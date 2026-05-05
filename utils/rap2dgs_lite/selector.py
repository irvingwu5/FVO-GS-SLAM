"""
RAP2DGSLiteSelector: top-K selection from RAP2DGS Lite scores.

Wraps RAP2DGSLiteScorer and handles min/max keep constraints,
empty-candidate fallback, and error recovery.
"""

import time
import torch

from .scorer import RAP2DGSLiteScorer
from .report import build_report_dict


class RAP2DGSLiteSelector:
    """Selects top-K Gaussians from candidate_mask using rule-based scoring.

    Safe by default: never prunes, never modifies the model, and always
    returns a valid mask. On error, signals fallback_required=True.
    """

    def __init__(self, cfg=None):
        """
        Args:
            cfg: dict with optional scorer config and selection params
        """
        self.scorer = RAP2DGSLiteScorer(cfg)
        cfg = cfg or {}

        sel = cfg.get("selection", {}) if isinstance(cfg, dict) else {}
        self._default_keep_percent = float(sel.get("keep_percent", 0.25))
        self._default_min_keep = int(sel.get("min_keep", 1000))
        self._default_max_keep = int(sel.get("max_keep", 8000))

    @torch.no_grad()
    def select(
        self,
        gaussians,
        candidate_mask,
        support_count=None,
        current_kf_id=None,
        keep_percent=None,
        min_keep=None,
        max_keep=None,
    ):
        """Select top-scoring Gaussians within candidate_mask.

        Args:
            gaussians: GaussianModel instance
            candidate_mask: (N,) bool tensor
            support_count: (N,) optional long/float tensor
            current_kf_id: optional int
            keep_percent: fraction of candidates to keep (overrides config)
            min_keep: minimum points to keep (overrides config)
            max_keep: maximum points to keep (overrides config)

        Returns:
            selected_mask: (N,) bool tensor
            scores: (N,) float tensor
            report: dict with selection metadata
        """
        keep_pct = keep_percent if keep_percent is not None else self._default_keep_percent
        min_k = min_keep if min_keep is not None else self._default_min_keep
        max_k = max_keep if max_keep is not None else self._default_max_keep

        N = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device
        candidate_mask = candidate_mask.to(device=device)

        t0 = time.time()

        report = build_report_dict(
            enabled=True,
            num_total_gaussians=N,
            num_candidates=0,
            num_selected=0,
            keep_percent=keep_pct,
            min_keep=min_k,
            max_keep=max_k,
        )

        # ---- Guard: empty candidate mask ----
        cand_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        C = cand_idx.numel()
        report["num_candidates"] = int(C)

        if C == 0:
            report["fallback_required"] = True
            report["fallback_reason"] = "empty_candidate_mask"
            report["elapsed_ms"] = (time.time() - t0) * 1000.0
            return (
                torch.zeros(N, dtype=torch.bool, device=device),
                torch.full((N,), float("-inf"), dtype=torch.float, device=device),
                report,
            )

        # ---- Compute scores ----
        try:
            scores, score_info = self.scorer.compute_scores(
                gaussians, candidate_mask, support_count=support_count,
                current_kf_id=current_kf_id,
            )
        except Exception as e:
            report["fallback_required"] = True
            report["fallback_reason"] = f"scorer_error: {e}"
            report["elapsed_ms"] = (time.time() - t0) * 1000.0
            return (
                torch.zeros(N, dtype=torch.bool, device=device),
                torch.full((N,), float("-inf"), dtype=torch.float, device=device),
                report,
            )

        # Merge scorer info into report
        report["score_min"] = score_info.get("score_min", float("-inf"))
        report["score_max"] = score_info.get("score_max", float("-inf"))
        report["score_mean"] = score_info.get("score_mean", float("-inf"))

        # ---- Sanitize scores ----
        candidate_scores = scores[cand_idx]
        if torch.isnan(candidate_scores).any() or torch.isinf(candidate_scores).any():
            candidate_scores = candidate_scores.clone()
            candidate_scores[torch.isnan(candidate_scores)] = float("-inf")
            candidate_scores[torch.isinf(candidate_scores) & (candidate_scores > 0)] = float("-inf")
            candidate_scores[torch.isinf(candidate_scores) & (candidate_scores < 0)] = float("-inf")

        # ---- Determine how many to keep ----
        target_count = max(min_k, int(C * keep_pct))
        target_count = min(target_count, max_k)
        target_count = max(target_count, 0)

        # ---- Select top K ----
        if target_count >= C or target_count <= 0:
            selected_mask = candidate_mask.clone()
            actual_kept = C
        else:
            # Stable top-k: sort by score descending
            sorted_idx = torch.argsort(candidate_scores, descending=True, stable=True)
            selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
            selected_mask[cand_idx[sorted_idx[:target_count]]] = True
            actual_kept = target_count

        elapsed_ms = (time.time() - t0) * 1000.0

        report["num_selected"] = int(selected_mask.sum().item())
        report["num_selected"] = actual_kept
        report["fallback_required"] = False
        report["fallback_reason"] = ""
        report["elapsed_ms"] = elapsed_ms

        return selected_mask, scores, report
