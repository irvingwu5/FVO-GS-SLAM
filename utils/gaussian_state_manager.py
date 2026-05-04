"""
Stage 10: Gaussian state lifecycle manager for active/inactive map strategy.

Provides interfaces for tracking Gaussian states (active, inactive, handoff)
and managing memory budgets. Does NOT implement full active/inactive rendering
or restore submap-level PGO.

Default behavior: all Gaussians are "active". Handoff Gaussians are "handoff".
This module is a preparation layer for future online loop closure integration.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


# ============================================================================
# 1. Gaussian States
# ============================================================================

class GaussianState:
    """Gaussian lifecycle states."""
    ACTIVE = "active"          # participates in tracking + mapping
    INACTIVE = "inactive"      # stored but excluded from tracking/mapping
    HANDOFF = "handoff"        # frozen boundary Gaussian (read-only, temporary)
    REACTIVATED = "reactivated"  # was inactive, reactivated by loop closure


# Valid state transitions
_VALID_TRANSITIONS = {
    GaussianState.ACTIVE: {GaussianState.INACTIVE},
    GaussianState.INACTIVE: {GaussianState.REACTIVATED},
    GaussianState.REACTIVATED: {GaussianState.ACTIVE, GaussianState.INACTIVE},
    GaussianState.HANDOFF: set(),  # handoff Gaussians are eventually dropped
}


# ============================================================================
# 2. State Manager
# ============================================================================


@dataclass
class GaussianStateManager:
    """Manages Gaussian lifecycle states and memory budgets.

    Keeps per-Gaussian metadata as numpy arrays for efficient indexing.
    Designed as a lightweight overlay that doesn't modify GaussianModel internals.

    Memory budget: caps on active (including reactivated) and handoff Gaussians.
    """
    # Per-Gaussian arrays (indexed by Gaussian index)
    states: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    owner_keyframe_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    last_observed_kf_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    observation_counts: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # Budget
    max_active_gaussians: int = 200_000
    max_handoff_gaussians: int = 80_000
    max_reactivated_per_loop: int = 50_000

    # Stats
    reactivated_count: int = 0
    total_inactivated: int = 0

    def initialize(self, N: int, owner_kf_id: int = 0):
        """Initialize state arrays for N Gaussians, all ACTIVE."""
        self.states = np.full(N, GaussianState.ACTIVE, dtype=object)
        self.owner_keyframe_ids = np.full(N, owner_kf_id, dtype=np.int32)
        self.last_observed_kf_ids = np.full(N, owner_kf_id, dtype=np.int32)
        self.observation_counts = np.zeros(N, dtype=np.int32)

    def extend(self, n_new: int, owner_kf_id: int, last_obs_kf_id: int):
        """Add n_new Gaussians (from densification or new KF seeding)."""
        if n_new == 0:
            return
        self.states = np.concatenate([
            self.states,
            np.full(n_new, GaussianState.ACTIVE, dtype=object),
        ])
        self.owner_keyframe_ids = np.concatenate([
            self.owner_keyframe_ids,
            np.full(n_new, owner_kf_id, dtype=np.int32),
        ])
        self.last_observed_kf_ids = np.concatenate([
            self.last_observed_kf_ids,
            np.full(n_new, last_obs_kf_id, dtype=np.int32),
        ])
        self.observation_counts = np.concatenate([
            self.observation_counts,
            np.zeros(n_new, dtype=np.int32),
        ])

    def prune(self, mask: np.ndarray):
        """Remove Gaussians by boolean mask (True = keep)."""
        self.states = self.states[mask]
        self.owner_keyframe_ids = self.owner_keyframe_ids[mask]
        self.last_observed_kf_ids = self.last_observed_kf_ids[mask]
        self.observation_counts = self.observation_counts[mask]

    def record_observations(self, visible_mask: np.ndarray, kf_id: int):
        """Update observation records for visible Gaussians."""
        self.observation_counts[visible_mask] += 1
        self.last_observed_kf_ids[visible_mask] = kf_id

    # ---- State queries ----

    def get_active_mask(self) -> np.ndarray:
        """Return boolean mask of Gaussians eligible for tracking/mapping."""
        return (self.states == GaussianState.ACTIVE) | \
               (self.states == GaussianState.REACTIVATED)

    def get_handoff_mask(self) -> np.ndarray:
        """Return boolean mask of handoff Gaussians."""
        return self.states == GaussianState.HANDOFF

    def count_state(self, state: str) -> int:
        """Count Gaussians in a given state."""
        return int((self.states == state).sum())

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics."""
        return {
            "total": len(self.states),
            "active": self.count_state(GaussianState.ACTIVE),
            "inactive": self.count_state(GaussianState.INACTIVE),
            "handoff": self.count_state(GaussianState.HANDOFF),
            "reactivated": self.count_state(GaussianState.REACTIVATED),
            "total_inactivated": self.total_inactivated,
            "reactivated_count": self.reactivated_count,
        }

    # ---- State transitions ----

    def mark_handoff(self, mask: np.ndarray):
        """Mark Gaussians as handoff (frozen, temporary)."""
        self.states[mask] = GaussianState.HANDOFF

    def drop_handoff(self):
        """Remove all handoff Gaussians (they are not pruned, just cleared)."""
        mask = self.states != GaussianState.HANDOFF
        self.prune(mask)


# ============================================================================
# 3. Active/Inactive Transition Functions
# ============================================================================


def mark_gaussians_inactive_by_observation_gap(
    manager: GaussianStateManager,
    current_kf_id: int,
    max_gap: int = 20,
    min_observations: int = 3,
    max_inactive_per_call: Optional[int] = None,
) -> int:
    """Mark ACTIVE Gaussians as INACTIVE if not observed for max_gap keyframes.

    Only affects Gaussians with observation_count >= min_observations
    (avoids prematurely deactivating newly seeded Gaussians).

    Args:
        manager: GaussianStateManager instance.
        current_kf_id: Current keyframe ID.
        max_gap: Observation gap threshold in keyframes.
        min_observations: Minimum observations before a Gaussian can go inactive.
        max_inactive_per_call: Budget cap per call (None = no limit).

    Returns:
        Number of Gaussians marked inactive.
    """
    active_mask = manager.states == GaussianState.ACTIVE
    if active_mask.sum() == 0:
        return 0

    gap = current_kf_id - manager.last_observed_kf_ids
    obs_ok = manager.observation_counts >= min_observations
    candidate_mask = active_mask & (gap > max_gap) & obs_ok

    if max_inactive_per_call is not None and candidate_mask.sum() > max_inactive_per_call:
        indices = np.nonzero(candidate_mask)[0]
        keep = np.random.RandomState(42).choice(
            indices, max_inactive_per_call, replace=False)
        candidate_mask = np.zeros(len(manager.states), dtype=bool)
        candidate_mask[keep] = True

    n = int(candidate_mask.sum())
    if n > 0:
        manager.states[candidate_mask] = GaussianState.INACTIVE
        manager.total_inactivated += n
    return n


def reactivate_gaussians_by_loop_keyframe(
    manager: GaussianStateManager,
    loop_keyframe_id: int,
    spatial_query_fn=None,
    max_radius: float = 1.0,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """Reactivate INACTIVE Gaussians near a verified loop keyframe.

    Only reactivates Gaussians whose owner_keyframe_id is near
    the loop keyframe (by frame_idx proximity) or whose xyz is
    spatially close to the loop KF's camera position.

    Args:
        manager: GaussianStateManager instance.
        loop_keyframe_id: The keyframe ID that triggered the loop closure.
        spatial_query_fn: Optional fn(kf_id) → (N,) bool mask of nearby Gaussians.
        max_radius: Max frame_idx radius for owner-based reactivation.
        config: Optional config dict.

    Returns:
        Number of Gaussians reactivated.
    """
    if config is None:
        config = {}
    max_budget = config.get("max_reactivated_gaussians_per_loop",
                            manager.max_reactivated_per_loop)
    inactive_mask = manager.states == GaussianState.INACTIVE
    if inactive_mask.sum() == 0:
        return 0

    # Owner-based: Gaussians owned by keyframes near the loop KF
    owner_dist = np.abs(manager.owner_keyframe_ids - loop_keyframe_id)
    owner_near = owner_dist <= max_radius

    # Spatial: use spatial query if provided
    spatial_near = np.zeros(len(manager.states), dtype=bool)
    if spatial_query_fn is not None:
        try:
            spatial_near = spatial_query_fn(loop_keyframe_id)
            if spatial_near.shape[0] != len(manager.states):
                spatial_near = np.zeros(len(manager.states), dtype=bool)
        except Exception:
            pass

    reactivate_mask = inactive_mask & (owner_near | spatial_near)

    # Budget cap
    n = int(reactivate_mask.sum())
    if n > max_budget:
        indices = np.nonzero(reactivate_mask)[0]
        chosen = np.random.RandomState(42).choice(indices, max_budget, replace=False)
        reactivate_mask = np.zeros(len(manager.states), dtype=bool)
        reactivate_mask[chosen] = True
        n = max_budget

    if n > 0:
        manager.states[reactivate_mask] = GaussianState.REACTIVATED
        manager.reactivated_count += n
    return n


def get_active_render_model(
    gaussian_model,
    manager: Optional[GaussianStateManager] = None,
    handoff_model=None,
):
    """Return the appropriate Gaussian model(s) for tracking/mapping.

    If manager is None, returns the full model (default behavior).
    If manager is set, returns only active + reactivated Gaussians,
    merged with handoff if provided.
    """
    if manager is None or len(manager.states) == 0:
        if handoff_model is not None:
            from gaussian_splatting.scene.gaussian_model import GaussianModel
            return GaussianModel.create_merged_for_render(gaussian_model, handoff_model)
        return gaussian_model

    # Future: return a view of only active Gaussians
    # For now, return full model (backward compatible)
    if handoff_model is not None:
        from gaussian_splatting.scene.gaussian_model import GaussianModel
        return GaussianModel.create_merged_for_render(gaussian_model, handoff_model)
    return gaussian_model
