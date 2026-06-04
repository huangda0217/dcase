"""
Adaptive fusion module

Automatically selects scoring strategy and fusion weights based on source/target domain gap.

Usage:
    from inference.fusion import AdaptiveFusion
    
    fuser = AdaptiveFusion()
    
    # Compute domain gap
    gap = fuser.compute_domain_gap(train_beats_feats, train_domains)
    
    # Multi-feature fusion
    final_scores = fuser.fuse(scores_dict, domain_gap=gap)
"""
import numpy as np
from scipy.stats import rankdata
from typing import Dict, Tuple


# Four-tier thresholds and strategies
# bt_gap > 1.4  → High (BEATs only)
# 1.0 < gap ≤ 1.4 → Mid (dual feature, no aux)
# 0.8 < gap ≤ 1.0 → Near (dual feature + ML)
# gap ≤ 0.8     → Low (all features)
THRESHOLDS = {
    "high": 1.4,
    "mid": 1.0,
    "near": 0.8,
}


class AdaptiveFusion:
    """
    Adaptive domain-gap multi-feature fuser.
    
    Automatically selects fusion strategy based on source-target distribution
    distance (domain gap) in BEATs feature space. Larger gap → more reliance
    on domain-robust features (BEATs).
    """
    
    def compute_domain_gap(self, beats_feats: np.ndarray, 
                           domains: np.ndarray) -> float:
        """
        Compute Euclidean distance between source-target domains in BEATs feature space.
        
        Args:
            beats_feats: BEATs training features (N, 768)
            domains: domain labels (N,)
            
        Returns:
            domain gap (float), larger value indicates harder domain shift
        """
        src_mask = domains == 'source'
        tgt_mask = domains == 'target'
        if tgt_mask.sum() == 0 or src_mask.sum() == 0:
            return 0.0
        return float(np.linalg.norm(
            beats_feats[src_mask].mean(0) - beats_feats[tgt_mask].mean(0)))
    
    def get_tier(self, domain_gap: float) -> str:
        """Return tier name based on domain gap"""
        if domain_gap > THRESHOLDS["high"]:
            return "high"
        elif domain_gap > THRESHOLDS["mid"]:
            return "mid"
        elif domain_gap > THRESHOLDS["near"]:
            return "near"
        else:
            return "low"
    
    def get_params(self, domain_gap: float) -> Dict:
        """
        Return scoring parameters for the tier matching the given domain gap.
        
        Returns:
            dict containing:
                sb_param: SubBand kNN-Ratio strength
                bt_param: BEATs kNN-Ratio strength
                w_bt: BEATs fusion weight
                bt_kl: BEATs score k_local
                sc_w: spectral contrast weight
                ml_w: multi-layer BEATs weight
                aux_enabled: whether to enable aux features (EAT/CQT/BP256)
        """
        tier = self.get_tier(domain_gap)
        
        if tier == "high":
            return dict(sb_param=0.0, bt_param=0.40, w_bt=1.0,
                       bt_kl=25, sc_w=0.0, ml_w=0.0, aux_enabled=False)
        elif tier == "mid":
            return dict(sb_param=0.50, bt_param=0.20, w_bt=0.05,
                       bt_kl=25, sc_w=0.08, ml_w=0.0, aux_enabled=False)
        elif tier == "near":
            return dict(sb_param=0.35, bt_param=0.35, w_bt=0.12,
                       bt_kl=20, sc_w=0.0, ml_w=0.27, aux_enabled=False)
        else:  # low
            return dict(sb_param=0.35, bt_param=0.35, w_bt=0.12,
                       bt_kl=10, sc_w=0.15, ml_w=0.27, aux_enabled=True)
    
    def fuse(self, scores_dict: Dict[str, np.ndarray], 
             domain_gap: float) -> np.ndarray:
        """
        Geometric rank fusion for multi-feature scores.
        
        Takes raw anomaly scores from each feature, applies adaptive weighting
        based on domain gap, then rank-normalizes and geometrically fuses.
        
        Args:
            scores_dict: dict of per-feature scores, supported keys:
                - "sb_r1", "sb_r2": SubBand dual-regularized scores
                - "bt_r1", "bt_r2": BEATs dual-regularized scores
                - "sc": spectral contrast scores
                - "ml": BEATs multi-layer scores (optional)
                - "eat", "cqt", "bp256": auxiliary feature scores (optional)
            domain_gap: domain gap
            
        Returns:
            final_scores: (N,) fused anomaly scores
        """
        params = self.get_params(domain_gap)
        w_bt = params["w_bt"]
        sc_w = params["sc_w"]
        ml_w = params["ml_w"]
        aux_enabled = params["aux_enabled"]
        
        # SubBand dual-regularized geometric fusion
        rs1 = self._rank_normalize(scores_dict["sb_r1"])
        rs2 = self._rank_normalize(scores_dict["sb_r2"])
        rs = rs1**0.35 * rs2**0.65
        
        # BEATs dual-regularized geometric fusion
        rb1 = self._rank_normalize(scores_dict["bt_r1"])
        rb2 = self._rank_normalize(scores_dict["bt_r2"])
        rb = rb1**0.50 * rb2**0.50
        
        # Main feature fusion (SubBand + BEATs)
        scores = rs**(1 - w_bt) * rb**w_bt
        
        # Spectral contrast
        if sc_w > 0 and "sc" in scores_dict:
            rsc = self._rank_normalize(scores_dict["sc"])
            scores = scores**(1 - sc_w) * rsc**sc_w
        
        # BEATs multi-layer
        if ml_w > 0 and "ml" in scores_dict:
            rml = self._rank_normalize(scores_dict["ml"])
            scores = scores**(1 - ml_w) * rml**ml_w
        
        # Auxiliary features (EAT + CQT + BP256)
        if aux_enabled:
            aux_weights = {"eat": 0.135, "cqt": 0.205, "bp256": 0.055}
            # Only compute weights for existing aux features
            active_aux = {k: v for k, v in aux_weights.items() if k in scores_dict}
            
            if active_aux:
                total_aux = sum(active_aux.values())
                aux_terms = np.ones_like(scores)
                for feat_name, weight in active_aux.items():
                    aux_terms *= self._rank_normalize(scores_dict[feat_name]) ** weight
                scores = scores**(1 - total_aux) * aux_terms
        
        return scores
    
    @staticmethod
    def _rank_normalize(scores: np.ndarray) -> np.ndarray:
        """Rank normalization: convert scores to [0, 1] percentile ranks"""
        return np.clip(rankdata(scores) / len(scores), 1e-6, 1.0)
