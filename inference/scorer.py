"""
Anomaly scoring module

Provides fit/score interface: fit on training data for a single machine, then score test data.

Usage:
    from inference.scorer import AnomalyScorer
    
    scorer = AnomalyScorer(method="relative_max", reg=1e-4, k_local=10)
    
    # Fit: pass training features and domain labels
    scorer.fit(train_feats, train_domains)
    
    # Score: returns anomaly scores (higher = more anomalous)
    scores = scorer.score(test_feats)
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

class AnomalyScorer:
    """
    Single-feature anomaly scorer.
    
    Supports three scoring methods:
        - "relative_max": Relative-Max local normalization + kNN-Ratio (recommended)
        - "mahalanobis": basic Mahalanobis distance
        - "knn_ratio": Mahalanobis + kNN proximity ratio adjustment
    
    Args:
        method: scoring method
        reg: covariance regularization (prevents singularity)
        k_local: top-k neighbors count for Relative-Max
        param: kNN-Ratio adjustment strength (0=no adjustment)
        k_tgt: target domain neighbor count
        k_src: source domain neighbor count
    """
    
    def __init__(self, method: str = "relative_max", reg: float = 1e-4,
                 k_local: int = 10, param: float = 0.35,
                 k_tgt: int = 10, k_src: int = 100):
        assert method in ("relative_max", "mahalanobis", "knn_ratio")
        self.method = method
        self.reg = reg
        self.k_local = k_local
        self.param = param
        self.k_tgt = k_tgt
        self.k_src = k_src
        
        # State saved after fit
        self._mu = None
        self._inv_cov = None
        self._train_feats = None
        self._train_scores = None
        self._train_domains = None
    
    def fit(self, train_feats: np.ndarray, train_domains: np.ndarray):
        """
        Fit normal distribution model.
        
        Args:
            train_feats: training feature matrix (N_train, D)
            train_domains: domain label array (N_train,), values "source" or "target"
        """
        dim = train_feats.shape[1]
        self._mu = train_feats.mean(0)
        cov = np.cov(train_feats.T) + self.reg * np.eye(dim)
        self._inv_cov = np.linalg.inv(cov)
        self._train_feats = train_feats
        self._train_domains = train_domains
        
        # Precompute training set Mahalanobis scores (needed for Relative-Max)
        diff = train_feats - self._mu
        self._train_scores = np.sqrt(np.sum(diff @ self._inv_cov * diff, axis=1))
    
    def score(self, test_feats: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            test_feats: test feature matrix (N_test, D)
            
        Returns:
            anomaly_scores: (N_test,) anomaly scores, higher = more anomalous
        """
        if self._mu is None:
            raise RuntimeError("Please call fit() first to fit the model")
        
        if self.method == "relative_max":
            return self._score_relative_max(test_feats)
        elif self.method == "knn_ratio":
            return self._score_knn_ratio(test_feats)
        else:
            return self._score_mahalanobis(test_feats)
    
    def _score_mahalanobis(self, test_feats: np.ndarray) -> np.ndarray:
        """Basic Mahalanobis distance scoring"""
        diff = test_feats - self._mu
        return np.sqrt(np.sum(diff @ self._inv_cov * diff, axis=1))
    
    def _score_relative_max(self, test_feats: np.ndarray) -> np.ndarray:
        """
        Relative-Max scoring:
        1. Compute test sample Mahalanobis distance
        2. Find k_local nearest training neighbors
        3. Divide by max Maha score among those neighbors (local normalization)
        4. Apply kNN-Ratio proximity adjustment
        """
        # Test sample Maha scores
        diff = test_feats - self._mu
        test_scores = np.sqrt(np.sum(diff @ self._inv_cov * diff, axis=1))
        
        # Local normalization
        k = min(self.k_local, len(self._train_feats))
        nn = NearestNeighbors(n_neighbors=k).fit(self._train_feats)
        _, nn_idx = nn.kneighbors(test_feats)
        local_ref = np.array([self._train_scores[nn_idx[i]].max() 
                              for i in range(len(test_feats))])
        relative = test_scores / (local_ref + 1e-8)
        
        # kNN-Ratio proximity adjustment
        if self.param > 0:
            relative = self._apply_knn_ratio(test_feats, relative)
        
        return relative
    
    def _score_knn_ratio(self, test_feats: np.ndarray) -> np.ndarray:
        """Mahalanobis with kNN-Ratio adjustment"""
        base = self._score_mahalanobis(test_feats)
        if self.param > 0:
            return self._apply_knn_ratio(test_feats, base)
        return base
    
    def _apply_knn_ratio(self, test_feats: np.ndarray, 
                         scores: np.ndarray) -> np.ndarray:
        """
        kNN-Ratio proximity adjustment:
        Reduce scores for test samples near target domain training samples.
        Uses geometric positions of the few target domain samples.
        """
        tgt_mask = self._train_domains == 'target'
        src_mask = self._train_domains == 'source'
        tgt_tr = self._train_feats[tgt_mask]
        src_tr = self._train_feats[src_mask]
        
        if len(tgt_tr) < 2:
            return scores
        
        nn_tgt = NearestNeighbors(
            n_neighbors=min(self.k_tgt, len(tgt_tr))).fit(tgt_tr)
        d_tgt, _ = nn_tgt.kneighbors(test_feats)
        nn_src = NearestNeighbors(
            n_neighbors=min(self.k_src, len(src_tr))).fit(src_tr)
        d_src, _ = nn_src.kneighbors(test_feats)
        
        ratio = d_src.mean(1) / (d_tgt.mean(1) + 1e-8)
        proximity = ratio / (ratio.max() + 1e-8)
        return scores / (1 + self.param * proximity)
