"""
RAP2DGSLiteScorer: rule-based scoring for candidate Gaussians.

Computes per-Gaussian scores within a candidate_mask using lightweight
features. Never modifies the GaussianModel or optimizer state.
"""

import torch

from .feature_utils import (
    safe_minmax_normalize,
    safe_percentile_normalize,
    sanitize_score_tensor,
    compute_scale_area_and_axis_ratio,
    normalize_normals,
    compute_knn_indices_torch,
    compute_normal_consistency_score,
)


@torch.no_grad()
def _bounded_midrange_score(x, low_q=0.05, high_q=0.95):
    """Score tensor values by distance from the middle of the [low_q, high_q] range.

    1.0 at the centre, 0.0 at extremes. Returns zeros if range is too narrow.
    """
    x = x.float()
    if x.numel() < 4:
        return torch.ones_like(x) * 0.5
    lo = torch.quantile(x, low_q)
    hi = torch.quantile(x, high_q)
    if hi - lo < 1e-8:
        return torch.ones_like(x) * 0.5
    centre = (lo + hi) / 2.0
    half_range = (hi - lo) / 2.0
    score = 1.0 - torch.abs(x - centre) / half_range
    return score.clamp(0.0, 1.0)


class RAP2DGSLiteScorer:
    """Lightweight rule-based scorer for handoff candidate selection.

    Computes a weighted sum of feature scores for Gaussians within a
    candidate_mask. Each feature is normalised to [0, 1] before weighting.
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: dict with optional keys:
                weights: dict of per-feature weights
                features: dict of per-feature enable flags
                knn: dict with k, chunk_size
        """
        cfg = cfg or {}

        w = cfg.get("score_weights", {}) if isinstance(cfg, dict) else {}
        f = cfg.get("features", {}) if isinstance(cfg, dict) else {}
        k = cfg.get("knn", {}) if isinstance(cfg, dict) else {}

        self.w_support = float(w.get("support", 0.25))
        self.w_opacity = float(w.get("opacity", 0.20))
        self.w_observation = float(w.get("observation", 0.20))
        self.w_area = float(w.get("area", 0.10))
        self.w_normal = float(w.get("normal", 0.15))
        self.w_density = float(w.get("density", 0.10))

        self.use_support = bool(f.get("use_support", True))
        self.use_opacity = bool(f.get("use_opacity", True))
        self.use_observation = bool(f.get("use_observation", True))
        self.use_area = bool(f.get("use_area", True))
        self.use_normal = bool(f.get("use_normal", True))
        self.use_density = bool(f.get("use_density", True))

        self.knn_k = int(k.get("k", 16))
        self.knn_chunk_size = int(k.get("chunk_size", 4096))

    @torch.no_grad()
    def compute_scores(
        self,
        gaussians,
        candidate_mask,
        support_count=None,
        current_kf_id=None,
    ):
        """Compute RAP2DGS Lite scores for all Gaussians.

        Args:
            gaussians: GaussianModel instance
            candidate_mask: (N,) bool tensor
            support_count: (N,) optional long/float tensor, visibility support
            current_kf_id: unused (reserved for future use)

        Returns:
            scores: (N,) float tensor, -inf for non-candidates
            info: dict with per-component statistics
        """
        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device
        candidate_mask = candidate_mask.to(device=device)
        cand_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)  # (C,)
        C = cand_idx.numel()

        # Output: -inf for non-candidates
        scores_full = torch.full((N,), float("-inf"), dtype=torch.float, device=device)
        info = {
            "num_total": N,
            "num_candidates": int(C),
            "components": {},
        }

        if C == 0:
            return scores_full, info

        # ---- helpers to score only candidates, broadcast to full ----
        def _cand_to_full(cand_scores, default=0.0):
            out = torch.full((N,), float(default), dtype=torch.float, device=device)
            out[cand_idx] = cand_scores.float()
            return out

        total_weight = 0.0
        combined = torch.zeros(N, dtype=torch.float, device=device)

        # ---- 1. support_score ----
        if self.use_support and support_count is not None and support_count.numel() == N:
            supp = support_count.float()
            cand_supp = supp[cand_idx]
            cand_score = safe_percentile_normalize(cand_supp)
            full = _cand_to_full(cand_score, 0.0)
            full = sanitize_score_tensor(full, fill=0.0)
            combined += self.w_support * full
            total_weight += self.w_support
            info["components"]["support"] = {
                "min": float(cand_supp.min().item()) if C > 0 else 0.0,
                "max": float(cand_supp.max().item()) if C > 0 else 0.0,
                "score_min": float(cand_score.min().item()) if C > 0 else 0.0,
                "score_max": float(cand_score.max().item()) if C > 0 else 0.0,
            }
        else:
            info["components"]["support"] = {"enabled": False}

        # ---- 2. opacity_score ----
        if self.use_opacity:
            opacity = gaussians.get_opacity.squeeze()  # (N,)
            cand_op = opacity[cand_idx]
            cand_score = safe_minmax_normalize(cand_op)
            full = _cand_to_full(cand_score, 0.0)
            full = sanitize_score_tensor(full, fill=0.0)
            combined += self.w_opacity * full
            total_weight += self.w_opacity
            info["components"]["opacity"] = {
                "min": float(cand_op.min().item()) if C > 0 else 0.0,
                "max": float(cand_op.max().item()) if C > 0 else 0.0,
                "score_min": float(cand_score.min().item()) if C > 0 else 0.0,
                "score_max": float(cand_score.max().item()) if C > 0 else 0.0,
            }
        else:
            info["components"]["opacity"] = {"enabled": False}

        # ---- 3. observation_score ----
        if self.use_observation and hasattr(gaussians, "n_obs") and gaussians.n_obs.numel() == N:
            nobs = gaussians.n_obs.float().to(device)
            cand_nobs = nobs[cand_idx]
            cand_score = safe_percentile_normalize(torch.log1p(cand_nobs))
            full = _cand_to_full(cand_score, 0.0)
            full = sanitize_score_tensor(full, fill=0.0)
            combined += self.w_observation * full
            total_weight += self.w_observation
            info["components"]["observation"] = {
                "min": float(cand_nobs.min().item()) if C > 0 else 0.0,
                "max": float(cand_nobs.max().item()) if C > 0 else 0.0,
                "score_min": float(cand_score.min().item()) if C > 0 else 0.0,
                "score_max": float(cand_score.max().item()) if C > 0 else 0.0,
            }
        else:
            info["components"]["observation"] = {"enabled": False}

        # ---- 4. area_score (bounded midrange) ----
        if self.use_area:
            scaling = gaussians.get_scaling  # (N, 2) exp-activated
            area, _axis_ratio = compute_scale_area_and_axis_ratio(scaling)
            cand_area = area[cand_idx]
            cand_score = _bounded_midrange_score(torch.log1p(cand_area))
            full = _cand_to_full(cand_score, 0.0)
            full = sanitize_score_tensor(full, fill=0.0)
            combined += self.w_area * full
            total_weight += self.w_area
            info["components"]["area"] = {
                "score_min": float(cand_score.min().item()) if C > 0 else 0.0,
                "score_max": float(cand_score.max().item()) if C > 0 else 0.0,
            }
        else:
            info["components"]["area"] = {"enabled": False}

        # ---- Shared KNN (compute once for normal + density) ----
        need_knn = (self.use_normal or self.use_density) and C >= 2
        knn_idx_cand = None
        knn_dist_cand = None
        knn_ok = False
        if need_knn:
            cand_xyz = gaussians.get_xyz[cand_idx]  # (C, 3)
            knn_idx_cand, knn_dist_cand = compute_knn_indices_torch(
                cand_xyz, k=min(self.knn_k, 8), chunk_size=self.knn_chunk_size
            )
            knn_ok = knn_idx_cand.shape[1] > 0

        # ---- 5. normal_score (consistency with KNN) ----
        if self.use_normal and C >= 2:
            if knn_ok:
                normals = self._get_normals(gaussians, N, device)
                if normals is not None and normals.shape[0] == N:
                    # Map candidate-local knn indices back to global
                    knn_idx_global = cand_idx[knn_idx_cand.clamp(0, C - 1)]  # (C, k)
                    cand_normal_score = compute_normal_consistency_score(
                        normals[cand_idx], knn_idx_global
                    )
                    cand_normal_score = sanitize_score_tensor(cand_normal_score, fill=0.0)
                    full = _cand_to_full(cand_normal_score, 0.0)
                    full = sanitize_score_tensor(full, fill=0.0)
                    combined += self.w_normal * full
                    total_weight += self.w_normal
                    info["components"]["normal"] = {
                        "score_min": float(cand_normal_score.min().item()) if C > 0 else 0.0,
                        "score_max": float(cand_normal_score.max().item()) if C > 0 else 0.0,
                    }
                else:
                    info["components"]["normal"] = {"enabled": True, "skipped": "no_normals"}
            else:
                info["components"]["normal"] = {"enabled": True, "skipped": "knn_failed"}
        elif self.use_normal:
            info["components"]["normal"] = {"enabled": False}

        # ---- 6. density_score (local KNN density, reuses shared KNN) ----
        if self.use_density and C >= 2:
            if knn_ok:
                mean_dist = knn_dist_cand.mean(dim=-1)  # (C,)
                if mean_dist.numel() >= 4:
                    lo = torch.quantile(mean_dist, 0.05)
                    hi = torch.quantile(mean_dist, 0.95)
                    if hi - lo > 1e-8:
                        centre = (lo + hi) / 2.0
                        half_range = (hi - lo) / 2.0
                        cand_density = (
                            1.0 - torch.abs(mean_dist - centre) / half_range
                        ).clamp(0.0, 1.0)
                    else:
                        cand_density = torch.ones_like(mean_dist) * 0.5
                else:
                    cand_density = torch.ones_like(mean_dist) * 0.5
                cand_density = sanitize_score_tensor(cand_density, fill=0.0)
                full = _cand_to_full(cand_density, 0.0)
                full = sanitize_score_tensor(full, fill=0.0)
                combined += self.w_density * full
                total_weight += self.w_density
                info["components"]["density"] = {
                    "score_min": float(cand_density.min().item()) if C > 0 else 0.0,
                    "score_max": float(cand_density.max().item()) if C > 0 else 0.0,
                }
            else:
                info["components"]["density"] = {"enabled": True, "skipped": "knn_failed"}
        elif self.use_density:
            info["components"]["density"] = {"enabled": False}

        # ---- Normalise by total active weight ----
        if total_weight > 0:
            combined = combined / total_weight

        scores_full[cand_idx] = combined[cand_idx]
        scores_full = sanitize_score_tensor(scores_full, fill=float("-inf"))

        # Populate summary statistics
        cand_scores_final = scores_full[cand_idx]
        info["score_min"] = float(cand_scores_final.min().item()) if C > 0 else float("-inf")
        info["score_max"] = float(cand_scores_final.max().item()) if C > 0 else float("-inf")
        info["score_mean"] = float(cand_scores_final.mean().item()) if C > 0 else float("-inf")

        return scores_full, info

    @staticmethod
    def _get_normals(gaussians, N, device):
        """从 _rotation 实时推导法线（_normal 非优化参数，可能滞后于 _rotation）。"""
        if hasattr(gaussians, "_derive_normal_from_rotation"):
            normals = gaussians._derive_normal_from_rotation().to(device)
            if normals.shape[0] == N:
                return normalize_normals(normals)
        if hasattr(gaussians, "_normal") and gaussians._normal.shape[0] == N:
            normals = gaussians._normal.to(device).float()
            if normals.abs().sum() > 0:
                return normalize_normals(normals)
        return None
