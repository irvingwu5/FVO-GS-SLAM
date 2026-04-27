# utils/gsr_2dgs/pose_fusion_2dgs.py
# Weighted pose fusion from multiple candidates.

import numpy as np
import roma
from utils.logging_utils import Log


def fuse_relative_poses(candidates):
    """Fuse list of {T, loss, overlap, similarity}. Returns fused T_tgt_src (4x4) or None."""
    valid = [c for c in candidates if c.get("T") is not None]
    if len(valid) == 0:
        return None, []

    weights = []
    for c in valid:
        w = (c.get("similarity", 0.5) * c.get("overlap", 0.1)) / (c.get("loss", 0.1) + 1e-6)
        weights.append(max(w, 1e-6))
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    # Translation: weighted mean
    t_fused = np.zeros(3)
    for w, c in zip(weights, valid):
        t_fused += w * c["T"][:3, 3]

    # Rotation: weighted mean via SVD projection
    R_sum = np.zeros((3, 3))
    for w, c in zip(weights, valid):
        R_sum += w * c["T"][:3, :3]
    U, _, Vt = np.linalg.svd(R_sum)
    R_fused = U @ Vt
    if np.linalg.det(R_fused) < 0:
        R_fused = U @ np.diag([1, 1, -1]) @ Vt

    T_fused = np.eye(4)
    T_fused[:3, :3] = R_fused
    T_fused[:3, 3] = t_fused

    Log(f"[2DGS-GSReg] fused {len(valid)} candidates, weights={[f'{w:.3f}' for w in weights]}")
    return T_fused, [c.get("T") for c in valid]
