"""
PAR RSKM (Pose-Aware Random Keyframe Replay) utility functions.

Pure functions with no class state. All computations use numpy/python float,
not torch tensors, to avoid autograd entanglement.
"""

import numpy as np
import random


def safe_float(value, default=0.0):
    """Convert tensors, numpy scalars, or Python numbers to float safely."""
    if value is None:
        return default
    if hasattr(value, "item"):
        v = value.item()
        if hasattr(v, "item"):
            return float(v.item())
        return float(v)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_pose_delta_metrics(c2w_a, c2w_b):
    """Compute translation error (meters) and rotation error (degrees)
    between two C2W poses.

    Args:
        c2w_a: np.ndarray (4,4) float64, first C2W
        c2w_b: np.ndarray (4,4) float64, second C2W

    Returns:
        trans_error: float (meters)
        rot_error_deg: float (degrees)
    """
    if c2w_a is None or c2w_b is None:
        return 0.0, 0.0

    # delta = inv(c2w_b) @ c2w_a, i.e., relative pose from b to a
    delta = np.linalg.inv(c2w_b) @ c2w_a
    trans_error = float(np.linalg.norm(delta[:3, 3]))

    R = delta[:3, :3]
    trace_val = (np.trace(R) - 1.0) / 2.0
    trace_val = max(-1.0, min(1.0, trace_val))
    rot_error_rad = float(np.arccos(trace_val))
    rot_error_deg = rot_error_rad * 180.0 / np.pi

    return trans_error, rot_error_deg


def compute_replay_weight(reliability, min_weight=0.25, max_weight=1.0):
    """Return clipped replay weight for mapping loss."""
    r = safe_float(reliability, default=1.0)
    return max(min_weight, min(max_weight, r))


def compute_par_sampling_score(
    reliability,
    replay_count=0,
    threshold=0.05,
    gamma=1.5,
    eps=1.0e-6,
):
    """Compute PAR sampling score for a single keyframe.

    score = indicator(r >= threshold) * (r + eps)^gamma / sqrt(1 + replay_count)

    Returns 0 if reliability is below threshold.
    """
    r = safe_float(reliability, default=1.0)
    if r < threshold:
        return 0.0
    base = (r + eps) ** gamma
    undersample = 1.0 / np.sqrt(1.0 + int(replay_count))
    return float(base * undersample)


def normalize_scores(scores):
    """Convert scores dict/list to normalized probabilities.

    Args:
        scores: dict {kf_id: score} or list of (kf_id, score)

    Returns:
        probs: dict {kf_id: prob}
        total: float total score
        need_fallback: bool, True if all scores are zero
    """
    if isinstance(scores, dict):
        items = list(scores.items())
    else:
        items = list(scores)

    total = sum(s for _, s in items)
    if total < 1e-12:
        need_fallback = True
        n = max(len(items), 1)
        probs = {kf_id: 1.0 / n for kf_id, _ in items}
        return probs, 0.0, need_fallback

    need_fallback = False
    probs = {kf_id: s / total for kf_id, s in items}
    return probs, total, need_fallback


def weighted_sample_index(scores, rng=None):
    """Sample one index by nonnegative scores.

    If all scores are zero or invalid, fallback to uniform sampling.

    Args:
        scores: dict {kf_id: score} or list of (kf_id, score)
        rng: random.Random instance or None (uses global random)

    Returns:
        kf_id: selected keyframe id
        fallback: bool, True if uniform fallback was used
    """
    rng = rng if rng is not None else random.Random()

    if isinstance(scores, dict):
        items = list(scores.items())
    else:
        items = list(scores)

    if len(items) == 0:
        return None, True

    total = sum(s for _, s in items)
    if total < 1e-12:
        # All zero: uniform fallback
        weights_uniform = [1.0 / len(items)] * len(items)
        idx = rng.choices(range(len(items)), weights=weights_uniform, k=1)[0]
        return items[idx][0], True

    weights = [s / total for _, s in items]
    idx = rng.choices(range(len(items)), weights=weights, k=1)[0]
    return items[idx][0], False


def compute_reliability_from_pose(trans_error, rot_error_deg, beta_pose=1.0):
    """Compute reliability from pose consistency alone.

    pose_error = trans_error + rot_error_deg / 30.0
    reliability = exp(-beta_pose * pose_error)

    Returns float in (0, 1].
    """
    pose_error = trans_error + rot_error_deg / 30.0
    return float(np.exp(-beta_pose * pose_error))


def assign_temporal_bins(viewpoints_dict, recent_frac=0.3, middle_frac=0.4, old_frac=0.3):
    """Assign keyframes to temporal bins based on frame ordering.

    Keyframes are sorted by their ID (proxy for temporal order).
    Bins: recent (last recent_frac), middle (middle middle_frac), old (first old_frac).

    Args:
        viewpoints_dict: dict {kf_id: viewpoint}
        recent_frac, middle_frac, old_frac: bin proportions

    Returns:
        bins: dict {kf_id: "recent" | "middle" | "old"}
    """
    kf_ids = sorted(list(viewpoints_dict.keys()))
    n = len(kf_ids)
    if n == 0:
        return {}

    old_end = max(1, int(n * old_frac))
    middle_end = old_end + max(1, int(n * middle_frac))

    bins = {}
    for i, kf_id in enumerate(kf_ids):
        if i < old_end:
            bins[kf_id] = "old"
        elif i < middle_end:
            bins[kf_id] = "middle"
        else:
            bins[kf_id] = "recent"
    return bins


def select_par_keyframes_binned(viewpoints_dict, current_window, num_samples,
                                 scores_dict, bin_probs, rng,
                                 current_frame_interval, iteration_count):
    """PAR weighted keyframe selection with temporal bin stratification.

    First selects a bin by probability, then weighted-samples within that bin.
    Falls back to non-binned select_par_keyframes if any bin is empty or all-zero.

    Args:
        bin_probs: dict {"recent": 0.5, "middle": 0.3, "old": 0.2}
    """
    bins = assign_temporal_bins(viewpoints_dict)
    if not bins:
        return select_par_keyframes(
            viewpoints_dict, current_window, num_samples,
            scores_dict, rng, current_frame_interval, iteration_count
        )

    active_kf_ids = sorted(list(viewpoints_dict.keys()))
    current_kf_id = current_window[-1] if len(current_window) > 0 else None

    selected = []
    any_fallback = False

    for s in range(num_samples):
        iter_id = iteration_count + s
        if iter_id % current_frame_interval == 0:
            if current_kf_id is not None and current_kf_id in viewpoints_dict:
                selected.append(current_kf_id)
                continue

        # Step 1: Select bin by probability
        bin_names = list(bin_probs.keys())
        bin_weights = [bin_probs[bn] for bn in bin_names]
        chosen_bin = rng.choices(bin_names, weights=bin_weights, k=1)[0]

        # Step 2: Filter keyframes in chosen bin
        bin_kf_ids = [kf_id for kf_id in active_kf_ids
                      if bins.get(kf_id) == chosen_bin]

        if len(bin_kf_ids) == 0:
            # Fallback: any keyframe
            bin_kf_ids = active_kf_ids[:]

        # Step 3: Weighted sample within bin
        bin_scores = {kf_id: scores_dict.get(kf_id, 0.0) for kf_id in bin_kf_ids}
        chosen, fb = weighted_sample_index(bin_scores, rng=rng)
        if fb:
            # Uniform within bin
            chosen = rng.choice(bin_kf_ids) if bin_kf_ids else current_kf_id
            any_fallback = True

        if chosen is not None:
            selected.append(chosen)
        elif current_kf_id is not None:
            selected.append(current_kf_id)

    return selected, any_fallback


def compute_reliabilities_batch(viewpoints_dict, par_config):
    """Compute/update per-keyframe reliability for all viewpoints that have
    PAR metadata but haven't had reliability computed yet.

    Uses par_pose_trans_error and par_pose_rot_error_deg stored on viewpoint.
    Falls back to default_reliability when pose errors are not available.

    Args:
        viewpoints_dict: dict {kf_id: viewpoint}
        par_config: dict with keys beta_pose, default_reliability

    Returns:
        reliabilities: dict {kf_id: float}
    """
    beta_pose = par_config.get("beta_pose", 1.0)
    default_r = par_config.get("default_reliability", 1.0)

    reliabilities = {}
    for kf_id, vp in viewpoints_dict.items():
        initialized = getattr(vp, "par_initialized", False)
        if not initialized:
            # Use default for frames without PAR metadata
            reliabilities[kf_id] = default_r
            continue

        trans_err = safe_float(getattr(vp, "par_pose_trans_error", None), default=0.0)
        rot_err = safe_float(getattr(vp, "par_pose_rot_error_deg", None), default=0.0)

        r = compute_reliability_from_pose(trans_err, rot_err, beta_pose)
        vp.par_reliability = r
        reliabilities[kf_id] = r

    return reliabilities


def select_par_keyframes(viewpoints_dict, current_window, num_samples,
                         scores_dict, rng, current_frame_interval, iteration_count):
    """PAR weighted keyframe selection with forced current-frame injection.

    Mimics the vanilla _select_rskm_keyframes loop structure but uses
    weighted sampling from scores_dict instead of uniform choice.

    Args:
        viewpoints_dict: dict {kf_id: viewpoint}
        current_window: list of current window kf_ids
        num_samples: int, number of supervision frames to select
        scores_dict: dict {kf_id: score} from compute_sampling_scores
        rng: random.Random instance
        current_frame_interval: int, force current frame every N samples
        iteration_count: int, current mapping iteration

    Returns:
        selected_ids: List[int]
        fallback_used: bool, True if any uniform fallback occurred
    """
    active_kf_ids = sorted(list(viewpoints_dict.keys()))
    current_kf_id = current_window[-1] if len(current_window) > 0 else None

    total_score = sum(scores_dict.get(kf_id, 0.0) for kf_id in active_kf_ids)
    need_fallback = total_score < 1e-12

    selected = []
    any_fallback = False

    for s in range(num_samples):
        iter_id = iteration_count + s
        if iter_id % current_frame_interval == 0:
            if current_kf_id is not None and current_kf_id in viewpoints_dict:
                selected.append(current_kf_id)
                continue

        if need_fallback or len(active_kf_ids) == 0:
            any_fallback = True
            if current_kf_id is not None and current_kf_id in viewpoints_dict:
                selected.append(current_kf_id)
            elif len(active_kf_ids) > 0:
                selected.append(rng.choice(active_kf_ids))
            continue

        # Weighted sampling
        kf_scores = {kf_id: scores_dict.get(kf_id, 0.0) for kf_id in active_kf_ids}
        chosen, fb = weighted_sample_index(kf_scores, rng=rng)
        if fb:
            any_fallback = True
        if chosen is not None:
            selected.append(chosen)
        elif current_kf_id is not None:
            selected.append(current_kf_id)

    return selected, any_fallback
