"""
Lightweight feature utilities for RAP2DGS Lite scoring.

All functions operate under torch.no_grad() and never modify inputs.
KNN is always chunked to avoid N×N distance matrices.
"""

import torch


@torch.no_grad()
def safe_minmax_normalize(x, eps=1e-6):
    """Normalize tensor x to [0, 1] via min-max, returning zeros if range ~0.

    Args:
        x: (N,) float tensor
        eps: minimum range to avoid division by zero

    Returns:
        (N,) float tensor in [0, 1]
    """
    x = x.float()
    x_min = x.min()
    x_max = x.max()
    denom = x_max - x_min
    if denom < eps:
        return torch.zeros_like(x)
    return (x - x_min) / denom


@torch.no_grad()
def safe_percentile_normalize(x, low=0.01, high=0.99, eps=1e-6):
    """Normalize x to [0, 1] using percentile clipping, then min-max.

    Values below the low percentile map toward 0, above high map toward 1.

    Args:
        x: (N,) float tensor
        low: lower percentile (0-1)
        high: upper percentile (0-1)
        eps: minimum range

    Returns:
        (N,) float tensor in roughly [0, 1]
    """
    x = x.float()
    if x.numel() < 4:
        return safe_minmax_normalize(x, eps=eps)
    x_low = torch.quantile(x, low)
    x_high = torch.quantile(x, high)
    x_clipped = x.clamp(x_low, x_high)
    return safe_minmax_normalize(x_clipped, eps=eps)


@torch.no_grad()
def sanitize_score_tensor(x, fill=0.0):
    """Replace NaN, Inf, -Inf with fill value.

    Args:
        x: any-shape float tensor
        fill: replacement value

    Returns:
        tensor of same shape, sanitized
    """
    x = x.clone()
    x[torch.isnan(x)] = fill
    x[torch.isinf(x) & (x > 0)] = fill
    x[torch.isinf(x) & (x < 0)] = fill
    return x


@torch.no_grad()
def compute_scale_area_and_axis_ratio(scaling, eps=1e-8):
    """Compute area and axis ratio from 2DGS scaling.

    For 2DGS, scaling is (N, 2) with exp-activated values.
    area = exp(s0) * exp(s1), axis_ratio = max(s0,s1) / min(s0,s1).

    Args:
        scaling: (N, 2) tensor, exp-activated scales
        eps: epsilon for ratio denominator

    Returns:
        area: (N,) float tensor
        axis_ratio: (N,) float tensor, >= 1.0
    """
    s0 = scaling[:, 0]
    s1 = scaling[:, 1]
    area = s0 * s1
    s_max = torch.max(s0, s1)
    s_min = torch.min(s0, s1)
    axis_ratio = s_max / (s_min + eps)
    return area, axis_ratio


@torch.no_grad()
def normalize_normals(normals, eps=1e-8):
    """L2-normalize normal vectors, returning zeros for near-zero inputs.

    Args:
        normals: (N, 3) float tensor
        eps: minimum norm

    Returns:
        (N, 3) normalized normals
    """
    norms = normals.norm(dim=-1, keepdim=True)
    valid = norms > eps
    normalized = torch.zeros_like(normals)
    normalized[valid.squeeze(-1)] = (
        normals[valid.squeeze(-1)] / norms[valid.squeeze(-1)]
    )
    return normalized


@torch.no_grad()
def compute_knn_indices_torch(xyz, k=16, chunk_size=4096):
    """Chunked KNN search using PyTorch cdist, avoiding N×N matrix.

    Args:
        xyz: (M, 3) float tensor
        k: number of neighbors (excluding self)
        chunk_size: max query points per chunk

    Returns:
        knn_idx: (M, min(k, M-1)) long tensor, neighbor indices
        knn_dist: (M, min(k, M-1)) float tensor, neighbor distances
    """
    M = xyz.shape[0]
    if M < 2:
        if M == 1:
            return (
                torch.zeros(1, 0, dtype=torch.long, device=xyz.device),
                torch.zeros(1, 0, dtype=torch.float, device=xyz.device),
            )
        return (
            torch.zeros(0, 0, dtype=torch.long, device=xyz.device),
            torch.zeros(0, 0, dtype=torch.float, device=xyz.device),
        )

    actual_k = min(k, M - 1)
    if actual_k < 1:
        return (
            torch.zeros(M, 0, dtype=torch.long, device=xyz.device),
            torch.zeros(M, 0, dtype=torch.float, device=xyz.device),
        )

    all_topk_idx = []
    all_topk_dist = []

    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        query = xyz[i:end]  # (chunk, 3)
        dist_chunk = torch.cdist(query, xyz)  # (chunk, M)
        # k+1 because closest point is self (distance ~0)
        topk = torch.topk(dist_chunk, k=actual_k + 1, dim=-1, largest=False)
        topk_idx = topk.indices   # (chunk, actual_k+1)
        topk_dist = topk.values

        # Remove self-neighbor (index == query index in full xyz)
        query_indices = torch.arange(i, end, device=xyz.device).unsqueeze(-1)  # (chunk, 1)
        is_self = (topk_idx == query_indices)  # (chunk, actual_k+1)

        # Build filtered results
        keep_mask = ~is_self
        # Keep first actual_k non-self entries
        cumsum = keep_mask.cumsum(dim=-1)  # (chunk, actual_k+1)
        select_mask = (cumsum <= actual_k) & keep_mask  # (chunk, actual_k+1)

        # Build output for this chunk
        chunk_idx_out = torch.zeros(end - i, actual_k, dtype=torch.long, device=xyz.device)
        chunk_dist_out = torch.zeros(end - i, actual_k, dtype=torch.float, device=xyz.device)
        for j in range(end - i):
            valid = topk_idx[j][select_mask[j]]
            valid_d = topk_dist[j][select_mask[j]]
            n_valid = min(valid.shape[0], actual_k)
            if n_valid > 0:
                chunk_idx_out[j, :n_valid] = valid[:n_valid]
                chunk_dist_out[j, :n_valid] = valid_d[:n_valid]

        all_topk_idx.append(chunk_idx_out)
        all_topk_dist.append(chunk_dist_out)

    knn_idx = torch.cat(all_topk_idx, dim=0)
    knn_dist = torch.cat(all_topk_dist, dim=0)
    return knn_idx, knn_dist


@torch.no_grad()
def compute_local_density_score(xyz, candidate_indices, k=16, chunk_size=4096):
    """Score based on local point density via KNN mean distance.

    Penalises both isolated points (large mean distance) and overly dense
    clusters (very small mean distance).  Returns scores in [0, 1] where
    higher = better (moderate density).

    Args:
        xyz: (N, 3) all point positions
        candidate_indices: (C,) long tensor, indices into xyz
        k: KNN k
        chunk_size: KNN chunk size

    Returns:
        scores: (N,) float tensor, zero for non-candidate
    """
    N = xyz.shape[0]
    scores = torch.zeros(N, dtype=torch.float, device=xyz.device)

    if candidate_indices.numel() < 2:
        if candidate_indices.numel() == 1:
            scores[candidate_indices[0]] = 0.5
        return scores

    cand_xyz = xyz[candidate_indices]  # (C, 3)
    _, knn_dist = compute_knn_indices_torch(cand_xyz, k=min(k, 8), chunk_size=chunk_size)
    # knn_dist: (C, k')

    if knn_dist.shape[1] == 0:
        return scores

    mean_dist = knn_dist.mean(dim=-1)  # (C,)

    # Bounded midrange: prefer points in the 5%-95% range
    if mean_dist.numel() >= 4:
        lo = torch.quantile(mean_dist, 0.05)
        hi = torch.quantile(mean_dist, 0.95)
        # Score 1.0 in middle, 0.0 at extremes
        clamped = mean_dist.clamp(lo, hi)
        if hi - lo > 1e-8:
            dist_score = 1.0 - torch.abs(clamped - (lo + hi) / 2.0) / ((hi - lo) / 2.0)
            dist_score = dist_score.clamp(0.0, 1.0)
        else:
            dist_score = torch.ones_like(mean_dist) * 0.5
    else:
        dist_score = torch.ones_like(mean_dist) * 0.5

    scores[candidate_indices] = dist_score
    return scores


@torch.no_grad()
def compute_normal_consistency_score(normals, knn_indices):
    """Score how consistent each point's normal is with its KNN neighbors.

    Uses |dot(n_i, n_j)| averaged over neighbors. Values near 1.0 mean the
    normal agrees with neighbors.

    Args:
        normals: (N, 3) unit normals
        knn_indices: (N, k) long tensor, pre-computed KNN indices

    Returns:
        scores: (N,) float tensor in [0, 1]
    """
    N = normals.shape[0]
    k = knn_indices.shape[1] if knn_indices.numel() > 0 else 0
    if N == 0 or k == 0:
        return torch.zeros(N, dtype=torch.float, device=normals.device)

    normals = normalize_normals(normals)
    # Gather neighbor normals: (N, k, 3)
    nbr_normals = normals[knn_indices.clamp(0, N - 1)]  # (N, k, 3)
    dots = torch.abs(torch.sum(normals.unsqueeze(1) * nbr_normals, dim=-1))  # (N, k)
    return dots.mean(dim=-1)  # (N,)
