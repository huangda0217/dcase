"""
End-to-end anomaly detection pipeline

Provides ASDPipeline class, encapsulating the complete fit → predict workflow.
Supports anomaly detection on a single machine or entire dataset.

Usage:
    from inference.pipeline import ASDPipeline
    import soundfile as sf
    
    # Create pipeline
    pipeline = ASDPipeline()
    
    # Load audio
    train_wavs = [sf.read(p)[0] for p in train_paths]
    train_domains = np.array(["source"]*990 + ["target"]*10)
    
    # Fit (on a single machine's training data)
    pipeline.fit(train_wavs, train_domains)
    
    # Predict
    test_wavs = [sf.read(p)[0] for p in test_paths]
    scores = pipeline.predict(test_wavs)
    # scores: (N_test,) anomaly scores
"""
import os
import numpy as np
import soundfile as sf
from typing import List, Optional, Dict
from sklearn.decomposition import PCA

from .features import FeatureExtractor
from .scorer import AnomalyScorer
from .fusion import AdaptiveFusion
from .preprocess import wiener_denoise, load_audio


class ASDPipeline:
    """
    Single-machine anomalous sound detection pipeline.
    
    Workflow:
        1. fit(): extract training features → compute domain gap → fit scorers per feature
        2. predict(): extract test features → score per feature → adaptive fusion
    
    Args:
        beats_ckpt: BEATs weight path (None = use default location)
        device: "cuda" or "cpu"
        features: feature list to use, default ["subband", "beats", "sc"]
                  full options: ["subband", "beats", "sc", "beats_ml", "cqt"]
    """
    
    def __init__(self, beats_ckpt: Optional[str] = None, 
                 device: str = "cuda",
                 features: Optional[List[str]] = None):
        self.extractor = FeatureExtractor(beats_ckpt=beats_ckpt, device=device)
        self.fuser = AdaptiveFusion()
        self.features = features or ["subband", "beats", "sc"]
        
        # State saved after fit
        self._scorers = {}        # {feature_name_reg: AnomalyScorer}
        self._domain_gap = 0.0
        self._params = {}
        self._train_feats = {}    # Saved training features (domain_gap needed during predict)
        self._train_domains = None
        self._ml_pca = None       # PCA model for ML features
        self._fitted = False
    
    def fit(self, train_wavs: List[np.ndarray], train_domains: np.ndarray):
        """
        Fit the model.
        
        Extract all features from training set, compute domain gap, fit scorers per feature.
        
        Args:
            train_wavs: list of training audio (each shape=(T,), 16kHz mono)
            train_domains: domain labels (N,), values "source" / "target"
        """
        self._train_domains = train_domains
        
        # 1. Extract features
        print(f"[fit] Extracting features: {self.features}")
        self._train_feats = self.extractor.extract_all(train_wavs, self.features)
        
        # 2. Compute domain gap
        if "beats" in self._train_feats:
            self._domain_gap = self.fuser.compute_domain_gap(
                self._train_feats["beats"], train_domains)
        print(f"[fit] Domain gap (bt_gap) = {self._domain_gap:.3f}, "
              f"tier = {self.fuser.get_tier(self._domain_gap)}")
        
        # 3. Get adaptive parameters
        self._params = self.fuser.get_params(self._domain_gap)
        
        # 4. PCA reduction for ML features
        if "beats_ml" in self._train_feats:
            self._ml_pca = PCA(n_components=576, random_state=42)
            self._train_feats["beats_ml"] = self._ml_pca.fit_transform(
                self._train_feats["beats_ml"])
        
        # 5. Fit scorers
        self._fit_scorers(train_domains)
        self._fitted = True
        print(f"[fit] Done, fitted {len(self._scorers)} scorers")
    
    def _fit_scorers(self, domains: np.ndarray):
        """Fit scorers per feature and regularization level"""
        params = self._params
        
        # SubBand: dual regularization
        if "subband" in self._train_feats:
            for name, reg, kl in [("sb_r1", 1e-4, 3), ("sb_r2", 3e-4, 4)]:
                s = AnomalyScorer(method="relative_max", reg=reg,
                                  k_local=kl, param=params["sb_param"])
                s.fit(self._train_feats["subband"], domains)
                self._scorers[name] = s
        
        # BEATs: dual regularization
        if "beats" in self._train_feats:
            bt_kl = params["bt_kl"]
            for name, reg in [("bt_r1", 1e-4), ("bt_r2", 5e-4)]:
                s = AnomalyScorer(method="relative_max", reg=reg,
                                  k_local=bt_kl, param=params["bt_param"])
                s.fit(self._train_feats["beats"], domains)
                self._scorers[name] = s
        
        # Spectral contrast
        if "sc" in self._train_feats and params["sc_w"] > 0:
            s = AnomalyScorer(method="relative_max", reg=3e-4,
                              k_local=5, param=0.0)
            s.fit(self._train_feats["sc"], domains)
            self._scorers["sc"] = s
        
        # BEATs multi-layer
        if "beats_ml" in self._train_feats and params["ml_w"] > 0:
            s = AnomalyScorer(method="relative_max", reg=1.2e-3,
                              k_local=5, param=0.40)
            s.fit(self._train_feats["beats_ml"], domains)
            self._scorers["ml"] = s
        
        # CQT
        if "cqt" in self._train_feats and params["aux_enabled"]:
            s = AnomalyScorer(method="relative_max", reg=2e-3,
                              k_local=5, param=0.4)
            s.fit(self._train_feats["cqt"], domains)
            self._scorers["cqt"] = s
    
    def predict(self, test_wavs: List[np.ndarray]) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            test_wavs: list of test audio (each shape=(T,), 16kHz mono)
            
        Returns:
            scores: (N_test,) anomaly scores, higher = more anomalous
        """
        if not self._fitted:
            raise RuntimeError("Please call fit() first to fit the model")
        
        # 1. Extract features
        test_feats = self.extractor.extract_all(test_wavs, self.features)
        
        # ML PCA
        if "beats_ml" in test_feats and self._ml_pca is not None:
            test_feats["beats_ml"] = self._ml_pca.transform(test_feats["beats_ml"])
        
        # 2. Score per feature
        scores_dict = {}
        for name, scorer in self._scorers.items():
            # Determine which feature to use
            if name.startswith("sb"):
                feat = test_feats["subband"]
            elif name.startswith("bt"):
                feat = test_feats["beats"]
            elif name == "sc":
                feat = test_feats["sc"]
            elif name == "ml":
                feat = test_feats["beats_ml"]
            elif name == "cqt":
                feat = test_feats["cqt"]
            else:
                continue
            scores_dict[name] = scorer.score(feat)
        
        # 3. Adaptive fusion
        return self.fuser.fuse(scores_dict, self._domain_gap)
    
    def fit_from_paths(self, train_paths: List[str], train_domains: np.ndarray):
        """
        Fit from file paths (convenience method).
        Auto-detects dual-channel audio and applies Wiener denoising.
        
        Args:
            train_paths: list of WAV file paths
            train_domains: domain labels
        """
        wavs = load_audio(train_paths)
        self.fit(wavs, train_domains)
    
    def predict_from_paths(self, test_paths: List[str]) -> np.ndarray:
        """
        Predict from file paths (convenience method).
        Auto-detects dual-channel audio and applies Wiener denoising.
        
        Args:
            test_paths: list of WAV file paths
            
        Returns:
            scores: anomaly scores
        """
        wavs = load_audio(test_paths)
        return self.predict(wavs)
    


# ================================================================
# CLI entrypoint
# ================================================================

# def main():
#     """CLI entrypoint: run inference from directory or file list"""
#     import argparse
#     import json
#     import glob
    
#     parser = argparse.ArgumentParser(
#         description="DCASE Task 2 Anomalous Sound Detection Inference Tool")
#     parser.add_argument("--train_dir", required=True,
#                         help="Training directory (WAV files)")
#     parser.add_argument("--test_dir", required=True,
#                         help="Test directory (WAV files)")
#     parser.add_argument("--domain_file", default=None,
#                         help="Domain label file (each line: filename domain). "
#                              "If not provided, all are treated as source")
#     parser.add_argument("--beats_ckpt", default=None,
#                         help="BEATs model path")
#     parser.add_argument("--output", default="scores.json",
#                         help="Output path for results")
#     parser.add_argument("--device", default="cuda")
#     args = parser.parse_args()
    
#     # Collect audio files
#     train_paths = sorted(glob.glob(os.path.join(args.train_dir, "*.wav")))
#     test_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.wav")))
    
#     if not train_paths:
#         print(f"Error: No WAV files found in {args.train_dir}")
#         return
#     if not test_paths:
#         print(f"Error: No WAV files found in {args.test_dir}")
#         return
    
#     print(f"Training set: {len(train_paths)} files")
#     print(f"Test set: {len(test_paths)} files")
    
#     # Domain labels
#     if args.domain_file and os.path.exists(args.domain_file):
#         domain_map = {}
#         with open(args.domain_file) as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) >= 2:
#                     domain_map[parts[0]] = parts[1]
#         train_domains = np.array([
#             domain_map.get(os.path.basename(p), "source") for p in train_paths
#         ])
#     else:
#         train_domains = np.array(["source"] * len(train_paths))
    
#     # Run
#     pipeline = ASDPipeline(beats_ckpt=args.beats_ckpt, device=args.device)
#     pipeline.fit_from_paths(train_paths, train_domains)
#     scores = pipeline.predict_from_paths(test_paths)
    
#     # Output
#     output = {
#         "files": [os.path.basename(p) for p in test_paths],
#         "scores": scores.tolist(),
#     }
#     with open(args.output, 'w') as f:
#         json.dump(output, f, indent=2, ensure_ascii=False)
#     print(f"\nResults saved to: {args.output}")
#     print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")


# if __name__ == "__main__":
#     main()
