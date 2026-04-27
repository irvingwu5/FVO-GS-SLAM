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


def compute_overlap_2dgs(src_xyz, tgt_xyz, init_tsfm=None, radius=0.10, chunk_size=20000):
    """Symmetric overlap using full Gaussian xyz. No downsample."""
    src = src_xyz.float()
    tgt = tgt_xyz.float()

    if init_tsfm is not None and not np.allclose(init_tsfm, np.eye(4), atol=1e-4):
        T = torch.from_numpy(init_tsfm).float().to(src.device)
        src_t = (T[:3, :3] @ src.T).T + T[:3, 3]
    else:
        src_t = src

    s2t = _chunked_knn(src_t, tgt, radius, chunk_size)
    t2s = _chunked_knn(tgt, src_t, radius, chunk_size)
    overlap = min(s2t, t2s)

    Log(f"[2DGS-GSReg] overlap src2tgt={s2t:.4f} tgt2src={t2s:.4f} overlap={overlap:.4f} radius={radius:.3f}")
    return {"overlap": float(overlap), "src_to_tgt": float(s2t), "tgt_to_src": float(t2s)}
