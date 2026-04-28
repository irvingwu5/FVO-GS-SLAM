# utils/gsr_2dgs/overlap_2dgs.py
# Full Gaussian overlap. No voxel downsample.

import torch, numpy as np
from utils.logging_utils import Log


def _chunked_knn(src, tgt, radius=0.10, chunk_size=20000):
    """Chunked nearest-neighbor using torch cdist. Returns ratio of src within radius of tgt."""
    Ns = len(src)
    count = 0
    for i in range(0, Ns, chunk_size):
        chunk = src[i:i + chunk_size]
        dists = torch.cdist(chunk.unsqueeze(0), tgt.unsqueeze(0)).squeeze(0)
        count += (dists.min(dim=1).values < radius).sum().item()
    return count / max(Ns, 1)


def _chunked_knn_with_normals(src, tgt, src_n, tgt_n, radius=0.10,
                              cos_threshold=0.707, chunk_size=20000):
    """Chunked KNN with normal-angle filter.
    A match counts only if the nearest neighbor is within radius AND
    the absolute cosine similarity of normals exceeds cos_threshold.
    """
    Ns = len(src)
    count = 0
    for i in range(0, Ns, chunk_size):
        chunk_xyz = src[i:i + chunk_size]
        chunk_n = src_n[i:i + chunk_size]
        dists = torch.cdist(chunk_xyz.unsqueeze(0), tgt.unsqueeze(0)).squeeze(0)
        nn_idx = dists.argmin(dim=1)
        nn_dist = dists[range(len(nn_idx)), nn_idx]
        # Check normal consistency for matches within radius
        in_radius = nn_dist < radius
        if in_radius.any():
            matched_chunk_n = chunk_n[in_radius]
            matched_tgt_n = tgt_n[nn_idx[in_radius]]
            cos_sim = torch.abs((matched_chunk_n * matched_tgt_n).sum(dim=1))
            count += (cos_sim > cos_threshold).sum().item()
    return count / max(Ns, 1)


def compute_overlap_2dgs(src_xyz, tgt_xyz, init_tsfm=None, radius=0.10,
                         src_normal=None, tgt_normal=None,
                         max_normal_angle_deg=45.0, chunk_size=20000):
    """Symmetric overlap using full Gaussian xyz with optional normal filter.

    If normals are provided, only counts matches where the angle between
    the two surface normals is below max_normal_angle_deg.
    """
    src = src_xyz.float()
    tgt = tgt_xyz.float()

    if init_tsfm is not None and not np.allclose(init_tsfm, np.eye(4), atol=1e-4):
        T = torch.from_numpy(init_tsfm).float().to(src.device)
        R = T[:3, :3]
        t = T[:3, 3]
        src_t = (R @ src.T).T + t
        if src_normal is not None:
            src_normal_t = (R @ src_normal.float().T).T
        else:
            src_normal_t = None
    else:
        src_t = src
        src_normal_t = src_normal

    use_normals = (src_normal_t is not None and tgt_normal is not None)
    cos_threshold = np.cos(np.deg2rad(max_normal_angle_deg))

    if use_normals:
        s2t = _chunked_knn_with_normals(
            src_t, tgt, src_normal_t, tgt_normal, radius, cos_threshold, chunk_size)
        t2s = _chunked_knn_with_normals(
            tgt, src_t, tgt_normal, src_normal_t, radius, cos_threshold, chunk_size)
    else:
        s2t = _chunked_knn(src_t, tgt, radius, chunk_size)
        t2s = _chunked_knn(tgt, src_t, radius, chunk_size)

    overlap = min(s2t, t2s)

    Log(f"[2DGS-GSReg] overlap src2tgt={s2t:.4f} tgt2src={t2s:.4f} overlap={overlap:.4f} "
        f"radius={radius:.3f} normal_filter={'on' if use_normals else 'off'}")
    return {"overlap": float(overlap), "src_to_tgt": float(s2t), "tgt_to_src": float(t2s)}
