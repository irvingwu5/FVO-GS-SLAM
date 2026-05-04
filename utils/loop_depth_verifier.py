"""
Stage 4: RGB-D depth geometric verification for Reloc3R keyframe pair estimates.

Reloc3R produces a coarse T_target_from_source_raw. This module verifies the
relative pose by:
  1. Back-projecting sparse/downsampled points from source depth.
  2. Transforming them to the target camera using the candidate pose.
  3. Projecting onto the target image and comparing with target depth.
  4. Searching for an optimal translation scale that minimizes depth RMSE.

Only pairs that pass depth verification may proceed to PGO consideration.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple
from utils.keyframe_pgo import Reloc3RPairEstimate


# ============================================================================
# Data Structures
# ============================================================================


class DepthVerifiedPair:
    """Result of depth-based geometric verification for a Reloc3R keyframe pair.

    This is NOT a PGO edge — just a verified pair with geometry metrics.
    """
    __slots__ = (
        "source_keyframe_id", "target_keyframe_id",
        "source_submap_id", "target_submap_id",
        "T_target_from_source",  # 4x4 after scale search
        "scale_used", "depth_overlap", "depth_rmse",
        "depth_inlier_ratio", "valid_projected_points",
        "accepted_by_depth", "rejection_reason",
        "scale_search_diagnostics",
    )

    def __init__(self, source_kf: int, target_kf: int,
                 source_submap: int, target_submap: int):
        self.source_keyframe_id = source_kf
        self.target_keyframe_id = target_kf
        self.source_submap_id = source_submap
        self.target_submap_id = target_submap
        self.T_target_from_source: np.ndarray = np.eye(4)
        self.scale_used: float = 1.0
        self.depth_overlap: float = 0.0
        self.depth_rmse: float = 999.0
        self.depth_inlier_ratio: float = 0.0
        self.valid_projected_points: int = 0
        self.accepted_by_depth: bool = False
        self.rejection_reason: str = ""
        self.scale_search_diagnostics: Dict[str, Any] = {}


# ============================================================================
# Core Verification
# ============================================================================


def verify_reloc3r_pair_with_rgbd(
    estimate: Reloc3RPairEstimate,
    source_depth_path: Optional[str] = None,
    target_depth_path: Optional[str] = None,
    intrinsics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> DepthVerifiedPair:
    """Verify a Reloc3R pair estimate using RGB-D depth geometry.

    Args:
        estimate: Reloc3RPairEstimate with T_target_from_source_raw.
        source_depth_path: Path to source keyframe depth .pt file (H,W float32).
        target_depth_path: Path to target keyframe depth .pt file.
        intrinsics: dict with fx, fy, cx, cy, width, height.
        config: depth_verify config dict from LoopClosure config.

    Returns:
        DepthVerifiedPair with acceptance decision and metrics.
    """
    result = DepthVerifiedPair(
        estimate.source_keyframe_id, estimate.target_keyframe_id,
        estimate.source_submap_id, estimate.target_submap_id,
    )
    result.T_target_from_source = estimate.T_target_from_source_raw.copy()

    if config is None:
        result.rejection_reason = "no_depth_verify_config"
        return result

    enabled = config.get("enabled", True)
    if not enabled:
        result.rejection_reason = "depth_verify_disabled"
        return result

    # Load depth maps
    if source_depth_path is None or target_depth_path is None:
        result.rejection_reason = "missing_depth_path"
        return result

    try:
        import torch
        src_d = torch.load(source_depth_path, map_location="cpu")
        tgt_d = torch.load(target_depth_path, map_location="cpu")
        if isinstance(src_d, torch.Tensor):
            src_d = src_d.numpy()
        if isinstance(tgt_d, torch.Tensor):
            tgt_d = tgt_d.numpy()
    except Exception as e:
        result.rejection_reason = f"depth_load_error: {e}"
        return result

    src_depth = np.array(src_d, dtype=np.float32)
    tgt_depth = np.array(tgt_d, dtype=np.float32)

    if intrinsics is None:
        result.rejection_reason = "missing_intrinsics"
        return result

    fx = float(intrinsics.get("fx") or 525.0)
    fy = float(intrinsics.get("fy") or 525.0)
    cx = float(intrinsics.get("cx") or 319.5)
    cy = float(intrinsics.get("cy") or 239.5)
    H = int(intrinsics.get("height") or 480)
    W = int(intrinsics.get("width") or 640)

    # Reshape depth if needed
    if src_depth.ndim == 2:
        src_depth = src_depth.reshape(H, W)
    if tgt_depth.ndim == 2:
        tgt_depth = tgt_depth.reshape(H, W)

    # Config
    sample_count = config.get("depth_sample_count", 5000)
    min_overlap = config.get("min_overlap", 0.20)
    max_rmse = config.get("max_depth_rmse", 0.06)
    min_inlier_ratio = config.get("min_inlier_ratio", 0.35)
    max_scale_ratio = config.get("max_scale_ratio_from_init", 1.8)
    inlier_threshold = config.get("depth_inlier_threshold", 0.10)
    scale_search_steps = config.get("scale_search_steps", 10)

    # Back-project source depth to 3D points in source camera frame
    pts_3d, valid_src = _back_project(src_depth, fx, fy, cx, cy, sample_count)
    if len(pts_3d) < 100:
        result.rejection_reason = f"too_few_source_points_{len(pts_3d)}"
        return result

    # T_target_from_source_raw: transforms source-cam 3D points to target-cam
    R_raw = estimate.T_target_from_source_raw[:3, :3]
    t_raw = estimate.T_target_from_source_raw[:3, 3]
    t_raw_norm = np.linalg.norm(t_raw)

    if t_raw_norm < 1e-8:
        result.rejection_reason = "zero_translation_in_estimate"
        return result

    t_dir = t_raw / t_raw_norm

    # Scale search: multiply raw_t_norm by candidate scale factors.
    # Reloc3R raw translation is ~1m in unknown units. Log-spaced search
    # around 1.0 to densely cover plausible loop distances.
    scale_min = config.get("scale_search_min", 0.1)
    scale_max = config.get("scale_search_max", 20.0)
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), scale_search_steps)

    best_scale = 1.0
    best_rmse = 999.0
    best_overlap = 0.0
    best_inlier = 0.0
    best_valid = 0
    search_results = []

    for s in scales:
        t_scaled = t_dir * (t_raw_norm * s)
        overlap, rmse, inlier_ratio, valid_count = _evaluate_pose(
            pts_3d, R_raw, t_scaled, tgt_depth, fx, fy, cx, cy, W, H,
            inlier_threshold,
        )
        search_results.append({
            "scale": float(s), "overlap": float(overlap),
            "rmse": float(rmse), "inlier_ratio": float(inlier_ratio),
            "valid": int(valid_count),
        })
        if valid_count > 0 and rmse < best_rmse:
            best_rmse = rmse
            best_scale = s
            best_overlap = overlap
            best_inlier = inlier_ratio
            best_valid = valid_count

    result.scale_used = best_scale
    result.depth_overlap = best_overlap
    result.depth_rmse = best_rmse
    result.depth_inlier_ratio = best_inlier
    result.valid_projected_points = best_valid
    result.scale_search_diagnostics = {
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "search_results": search_results,
    }

    # Update T with best scale
    result.T_target_from_source[:3, 3] = t_dir * (t_raw_norm * best_scale)

    # Acceptance gate
    if best_overlap < min_overlap:
        result.rejection_reason = f"low_overlap_{best_overlap:.3f}_lt_{min_overlap}"
    elif best_rmse > max_rmse:
        result.rejection_reason = f"high_rmse_{best_rmse:.4f}_gt_{max_rmse}"
    elif best_inlier < min_inlier_ratio:
        result.rejection_reason = f"low_inlier_{best_inlier:.3f}_lt_{min_inlier_ratio}"
    else:
        result.accepted_by_depth = True

    return result


# ============================================================================
# Internal helpers
# ============================================================================


def _back_project(depth, fx, fy, cx, cy, max_samples=5000):
    """Back-project depth map to 3D points in camera frame.

    Returns (pts_3d, valid_mask) where pts_3d is (N,3) float32.
    """
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    valid = (depth > 0.01) & np.isfinite(depth)

    valid_indices = np.argwhere(valid)
    if len(valid_indices) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(valid_indices), max_samples, replace=False)
        valid_indices = valid_indices[idx]

    vi = valid_indices[:, 0]
    ui = valid_indices[:, 1]
    z = depth[vi, ui]

    x = (ui - cx) * z / fx
    y = (vi - cy) * z / fy

    return np.stack([x, y, z], axis=1).astype(np.float32), valid


def _evaluate_pose(pts_3d, R, t, tgt_depth,
                   fx, fy, cx, cy, W, H, inlier_threshold=0.10):
    """Evaluate a candidate pose by projecting points and comparing depth.

    Returns (overlap, rmse, inlier_ratio, valid_count).
    """
    # Transform points to target camera frame
    pts_tgt = (pts_3d @ R.T) + t.reshape(1, 3)  # (N,3)

    # Project to target image
    z_tgt = pts_tgt[:, 2]
    valid_z = z_tgt > 0.01
    if valid_z.sum() < 50:
        return 0.0, 999.0, 0.0, 0

    u_tgt = (pts_tgt[valid_z, 0] * fx / z_tgt[valid_z] + cx).astype(int)
    v_tgt = (pts_tgt[valid_z, 1] * fy / z_tgt[valid_z] + cy).astype(int)

    in_bounds = (u_tgt >= 0) & (u_tgt < W) & (v_tgt >= 0) & (v_tgt < H)
    if in_bounds.sum() < 50:
        return 0.0, 999.0, 0.0, 0

    u_in = u_tgt[in_bounds]
    v_in = v_tgt[in_bounds]
    z_proj = z_tgt[valid_z][in_bounds]

    # Compare with target depth
    z_tgt_val = tgt_depth[v_in, u_in]
    valid_tgt = (z_tgt_val > 0.01) & np.isfinite(z_tgt_val)

    if valid_tgt.sum() < 50:
        return 0.0, 999.0, 0.0, 0

    z_proj_valid = z_proj[valid_tgt]
    z_tgt_valid = z_tgt_val[valid_tgt]

    depth_diff = np.abs(z_proj_valid - z_tgt_valid)
    rmse = float(np.sqrt(np.mean(depth_diff ** 2)))
    inlier = float((depth_diff <= inlier_threshold).mean())
    overlap = float(len(z_proj_valid) / len(pts_3d))

    return overlap, rmse, inlier, len(z_proj_valid)
