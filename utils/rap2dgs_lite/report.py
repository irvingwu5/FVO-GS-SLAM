"""
Report utilities for RAP2DGS Lite selection runs.

Provides structured dicts and optional JSON serialisation for debugging
and ablation tracking.
"""

import json
import os


def build_report_dict(
    enabled=True,
    num_total_gaussians=0,
    num_candidates=0,
    num_selected=0,
    keep_percent=0.0,
    min_keep=0,
    max_keep=0,
    knn_k=0,
    use_normal=True,
    use_density=True,
    score_min=float("-inf"),
    score_max=float("-inf"),
    score_mean=float("-inf"),
    fallback_required=False,
    fallback_reason="",
    elapsed_ms=0.0,
    **extra,
):
    """Build a standardised report dict for one selection run.

    All values are JSON-serialisable scalars.
    Extra keyword arguments are merged into the top-level dict.
    """
    report = {
        "enabled": enabled,
        "num_total_gaussians": int(num_total_gaussians),
        "num_candidates": int(num_candidates),
        "num_selected": int(num_selected),
        "keep_percent": float(keep_percent),
        "min_keep": int(min_keep),
        "max_keep": int(max_keep),
        "knn_k": int(knn_k),
        "use_normal": bool(use_normal),
        "use_density": bool(use_density),
        "score_min": float(score_min) if score_min != float("-inf") else None,
        "score_max": float(score_max) if score_max != float("-inf") else None,
        "score_mean": float(score_mean) if score_mean != float("-inf") else None,
        "fallback_required": bool(fallback_required),
        "fallback_reason": str(fallback_reason),
        "elapsed_ms": float(elapsed_ms),
    }
    report.update(extra)
    return report


def save_report_json(report, out_dir, label="rap2dgs_lite_report"):
    """Save a report dict as JSON to out_dir/<label>.json.

    Creates the output directory if needed.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{label}.json")

    def _safe(v):
        if isinstance(v, float):
            if v != v:  # NaN
                return None
            if v == float("inf") or v == float("-inf"):
                return None
        return v

    safe_report = {}
    for k, v in report.items():
        if isinstance(v, dict):
            safe_report[k] = {kk: _safe(vv) for kk, vv in v.items()}
        else:
            safe_report[k] = _safe(v)

    with open(path, "w") as f:
        json.dump(safe_report, f, indent=2, default=str)

    return path
