"""
DCASE 2026 Task 2 Anomalous Sound Detection Inference Toolkit

Modules:
    features    - Feature extraction (SubBand, BEATs, SC, CQT, ML)
    scorer      - Anomaly scoring (Relative-Max, Mahalanobis)
    fusion      - Adaptive multi-feature fusion
    pipeline    - End-to-end pipeline (fit → predict)
    evaluate    - Evaluation metrics (AUC, pAUC, Omega)
    preprocess  - Dual-channel audio preprocessing (Wiener denoising)

Quick start:
    from inference import ASDPipeline
    
    pipeline = ASDPipeline()
    pipeline.fit(train_wavs, train_domains)
    scores = pipeline.predict(test_wavs)
"""
from .pipeline import ASDPipeline
from .features import FeatureExtractor
from .scorer import AnomalyScorer
from .fusion import AdaptiveFusion

from .preprocess import wiener_denoise, load_audio

__all__ = [
    "ASDPipeline",
    "FeatureExtractor", 
    "AnomalyScorer",
    "AdaptiveFusion",
    "compute_metrics",
    "compute_omega",
    "wiener_denoise",
    "load_audio",
]
