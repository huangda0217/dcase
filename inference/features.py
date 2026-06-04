"""
Feature extraction module

Provides feature extraction interfaces for single and batch audio.
Supports 7 feature types, each callable independently.

Usage:
    from inference.features import FeatureExtractor
    
    extractor = FeatureExtractor(beats_ckpt="checkpoints/BEATs_iter3_plus_AS2M.pt")
    
    # Extract SubBand features for a single audio
    feat = extractor.extract_subband(wav)  # wav: np.array, shape=(160000,)
    
    # Extract BEATs features in batch
    feats = extractor.extract_beats(wav_list)  # list of np.array → (N, 768)
    
    # Extract all features
    all_feats = extractor.extract_all(wav_list)  # dict: {feat_name: np.array}
"""
import os
import sys
import numpy as np
import torch
import torchaudio
from typing import List, Dict, Optional

# Default device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureExtractor:
    """
    Feature extractor, encapsulating logic for 7 feature types.
    
    Args:
        beats_ckpt: BEATs model weight path (relative to this file or absolute)
        device: compute device ("cuda" / "cpu")
    """
    
    def __init__(self, beats_ckpt: Optional[str] = None, device: str = DEVICE):
        self.device = device
        # Default checkpoint path: checkpoints/ directory next to this file
        if beats_ckpt is None:
            beats_ckpt = os.path.join(os.path.dirname(__file__), 
                                       "checkpoints", "BEATs_iter3_plus_AS2M.pt")
            
        self.beats_ckpt = beats_ckpt
        if not os.path.isfile(self.beats_ckpt):
            raise FileNotFoundError(
                f"Model checkpoint not found! Program terminated.\n"
                f"Missing file: {self.beats_ckpt}\n"
                f"Please download BEATs_iter3_plus_AS2M.pt and place it in the checkpoints directory."
            )
        self._beats_model = None  # lazy load
    
    # ================================================================
    # SubBand LogMel statistical features (1292d)
    # ================================================================
    
    def extract_subband(self, wav: np.ndarray, sr: int = 16000,
                        n_mels: int = 256, n_fft: int = 4096, 
                        hop: int = 512, n_bands: int = 4) -> np.ndarray:
        """
        Extract sub-band LogMel statistical features.
        
        Input: single audio (np.array, shape=(T,), 16kHz mono)
        Output: 1292-dimensional feature vector.
        
        Feature composition:
            - 4 sub-bands × 4 statistics (mean/std/max/min) × 64 bands = 1024d
            - cross-band correlation statistics (mean/std/median) = 12d
            - full-band lag-1 autocorrelation = 256d
        """
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0
        )
        lm = torch.log(mel(wav_t) + 1e-8).squeeze(0).numpy()  # (n_mels, T')
        
        band_size = n_mels // n_bands
        feats = []
        for b in range(n_bands):
            band = lm[b * band_size: (b + 1) * band_size]
            feats.extend([band.mean(1), band.std(1), band.max(1), band.min(1)])
            if band.shape[0] > 1:
                corr_matrix = np.corrcoef(band)
                upper = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]
                feats.append(np.array([upper.mean(), upper.std(), np.median(upper)]))
        
        # Lag-1 autocorrelation
        centered = lm - lm.mean(1, keepdims=True)
        var = lm.var(1) + 1e-8
        autocorr1 = (centered[:, :-1] * centered[:, 1:]).mean(1) / var
        feats.append(autocorr1)
        
        return np.concatenate(feats)  # 1292d
    
    # ================================================================
    # BEATs pretrained embeddings (768d)
    # ================================================================
    
    def _load_beats(self):
        """Lazy load BEATs model"""
        if self._beats_model is not None:
            return self._beats_model
        
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models", "BEATs"))
        from .models.BEATs import BEATs, BEATsConfig
        
        ckpt = torch.load(self.beats_ckpt, map_location="cpu")
        cfg = BEATsConfig(ckpt['cfg'])
        model = BEATs(cfg)
        model.load_state_dict(ckpt['model'])
        model.eval().to(self.device)
        self._beats_model = model
        return model
    
    def extract_beats(self, wavs: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """
        Extract BEATs mean-pooled embeddings in batch.
        
        Args:
            wavs: list of audio, each shape=(T,), 16kHz mono
            batch_size: batch size
            
        Returns:
            (N, 768) feature matrix
        """
        model = self._load_beats()
        all_feats = []
        
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i+batch_size]
            max_len = max(len(w) for w in batch)
            padded = np.zeros((len(batch), max_len), dtype=np.float32)
            masks = np.ones((len(batch), max_len), dtype=bool)
            for j, w in enumerate(batch):
                padded[j, :len(w)] = w
                masks[j, :len(w)] = False
            
            with torch.no_grad():
                wav_t = torch.from_numpy(padded).to(self.device)
                mask_t = torch.from_numpy(masks).to(self.device)
                features = model.extract_features(wav_t, padding_mask=mask_t)[0]
                all_feats.append(features.mean(dim=1).cpu().numpy())
        
        return np.concatenate(all_feats, axis=0)  # (N, 768)
    
    # ================================================================
    # BEATs multi-layer features (12 layers concatenated, PCA → 576d)
    # ================================================================
    
    def extract_beats_multilayer(self, wavs: List[np.ndarray], 
                                  batch_size: int = 16) -> np.ndarray:
        """
        Extract BEATs 12-layer mean-pooled concatenated features (raw 9216d).
        
        Note: returns raw high-dim features; PCA reduction is handled by scorer during fit.
        
        Returns:
            (N, 9216) feature matrix
        """
        model = self._load_beats()
        
        # Register hooks to capture layer outputs
        layer_outputs = []
        hooks = []
        for layer in model.encoder.layers:
            def hook_fn(module, input, output, storage=layer_outputs):
                storage.append(output[0] if isinstance(output, tuple) else output)
            hooks.append(layer.register_forward_hook(hook_fn))
        
        all_feats = []
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i+batch_size]
            max_len = max(len(w) for w in batch)
            padded = np.zeros((len(batch), max_len), dtype=np.float32)
            masks = np.ones((len(batch), max_len), dtype=bool)
            for j, w in enumerate(batch):
                padded[j, :len(w)] = w
                masks[j, :len(w)] = False
            
            layer_outputs.clear()
            with torch.no_grad():
                wav_t = torch.from_numpy(padded).to(self.device)
                mask_t = torch.from_numpy(masks).to(self.device)
                model.extract_features(wav_t, padding_mask=mask_t)
            
            ml_feat = torch.cat([lo.mean(dim=1) for lo in layer_outputs], dim=-1)
            all_feats.append(ml_feat.cpu().numpy())
        
        for h in hooks:
            h.remove()
        
        return np.concatenate(all_feats, axis=0)  # (N, 9216)
    
    # ================================================================
    # Spectral contrast (21d)
    # ================================================================
    
    def extract_spectral_contrast(self, wav: np.ndarray, sr: int = 16000,
                                   n_fft: int = 2048, hop_length: int = 512,
                                   n_bands: int = 6) -> np.ndarray:
        """
        Extract spectral contrast features (21d).
        
        Spectral contrast measures the difference between peaks and valleys
        in each frequency band, capturing harmonic structure complementary to SubBand.
        """
        import librosa
        contrast = librosa.feature.spectral_contrast(
            y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands
        )
        return np.concatenate([contrast.mean(1), contrast.std(1), contrast.max(1)])  # 21d
    
    # ================================================================
    # CQT features (336d)
    # ================================================================
    
    def extract_cqt(self, wav: np.ndarray, sr: int = 16000,
                    n_bins: int = 84, hop_length: int = 512) -> np.ndarray:
        """
        Extract Constant Q Transform (CQT) statistical features (336d).
        
        CQT frequency resolution varies logarithmically, suitable for harmonic structure.
        Statistics: mean/std/max/min × 84 bins = 336d
        """
        import librosa
        cqt = np.abs(librosa.cqt(y=wav, sr=sr, n_bins=n_bins, hop_length=hop_length))
        log_cqt = np.log(cqt + 1e-8)
        return np.concatenate([
            log_cqt.mean(1), log_cqt.std(1), log_cqt.max(1), log_cqt.min(1)
        ])  # 336d
    
    # ================================================================
    # Batch extract all features
    # ================================================================
    
    def extract_all(self, wavs: List[np.ndarray], 
                    features: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Batch extract specified (or all) features.
        
        Args:
            wavs: list of audio arrays
            features: feature names to extract. Default ["subband", "beats", "sc"]
                      Options: "subband", "beats", "sc", "beats_ml", "cqt"
        
        Returns:
            dict: {feature_name: (N, D) feature matrix}
        """
        if features is None:
            features = ["subband", "beats", "sc"]
        
        result = {}
        
        if "subband" in features:
            result["subband"] = np.stack([self.extract_subband(w) for w in wavs])
        
        if "beats" in features:
            result["beats"] = self.extract_beats(wavs)
        
        if "sc" in features:
            result["sc"] = np.stack([self.extract_spectral_contrast(w) for w in wavs])
        
        if "beats_ml" in features:
            result["beats_ml"] = self.extract_beats_multilayer(wavs)
        
        if "cqt" in features:
            result["cqt"] = np.stack([self.extract_cqt(w) for w in wavs])
        
        return result
