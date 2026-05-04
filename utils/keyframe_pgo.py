"""
Keyframe-level Pose Graph Optimization module.

This module defines data structures and stub functions for keyframe-level PGO.
The old submap-level PGO (construct_and_optimize_pose_graph in loop_closure.py)
is deprecated and disabled by default via LoopClosure.legacy_submap_pgo_enabled.

Key invariants (must hold across all refactoring stages):
  - cam.T is always global W2C; inv(cam.T) is global C2W.
  - Submaps are storage/optimization partitions, NOT coordinate partitions.
  - relative_pose = inv(seed_prev) @ seed_curr.
  - correct_tsfm is a left-multiplied correction: optimized = correct_tsfm @ original.
  - Gaussian xyz is already in global coordinates; fusion only applies correct_tsfm.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ============================================================================
# 1. Data Structures
# ============================================================================

@dataclass
class KeyframeNode:
    """A single keyframe node in the keyframe-level pose graph."""
    keyframe_id: int
    submap_id: int
    frame_idx: int
    c2w_original: np.ndarray  # 4x4 global C2W
    fixed: bool = False


@dataclass
class KeyframeEdge:
    """An edge between two keyframe nodes.

    T_source_to_target = inv(c2w_source) @ c2w_target.
    edge_type is one of: "temporal", "handoff", "loop".
    """
    source_keyframe_id: int
    target_keyframe_id: int
    T_source_to_target: np.ndarray  # 4x4
    edge_type: str  # "temporal" | "handoff" | "loop"
    information: Optional[np.ndarray] = None  # 6x6
    diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Reloc3RPairEstimate:
    """Raw Reloc3R output for a single keyframe pair.

    This is a coarse relative pose between two keyframe images.
    Must NOT be used as a PGO edge directly — must pass depth verification
    and optional render/GSReg refinement first.
    """
    source_keyframe_id: int
    target_keyframe_id: int
    source_submap_id: int
    target_submap_id: int
    source_c2w_global: np.ndarray       # 4x4
    target_c2w_global: np.ndarray       # 4x4
    T_target_from_source_raw: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4 Reloc3R raw output
    raw_translation_norm: float = 0.0
    reloc3r_confidence: float = 0.0
    num_valid_matches: int = 0
    odom_T_target_from_source: Optional[np.ndarray] = None  # 4x4, from odometry
    raw_vs_init_dot: float = 0.0
    scale_applied: float = 1.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    accepted_by_reloc3r: bool = False
    rejection_reason: str = ""


@dataclass
class VerifiedLoopEdge:
    """A verified keyframe loop edge, ready for PGO consumption.

    This is produced AFTER depth verification and optional render/GSReg refinement.
    Must never be constructed directly from Reloc3R raw output.

    Convention: T_source_to_target = inv(c2w_source) @ c2w_target
    This maps a point from source camera to target camera frame.
    """
    source_keyframe_id: int
    target_keyframe_id: int
    source_submap_id: int = 0
    target_submap_id: int = 0
    T_source_to_target: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4 refined
    information: Optional[np.ndarray] = None  # 6x6
    source_c2w_original: Optional[np.ndarray] = None  # 4x4
    target_c2w_original: Optional[np.ndarray] = None  # 4x4
    verification_metrics: Dict[str, Any] = field(default_factory=dict)
    accepted_for_pgo: bool = False
    rejection_reason: str = ""


@dataclass
class KeyframePGOResult:
    """Result of a keyframe-level PGO trial run."""
    optimized_keyframe_c2w: Dict[int, np.ndarray] = field(default_factory=dict)
    original_keyframe_c2w: Dict[int, np.ndarray] = field(default_factory=dict)
    keyframe_corrections: Dict[int, np.ndarray] = field(default_factory=dict)
    odom_residual_before_mean: float = 0.0
    odom_residual_after_mean: float = 0.0
    loop_residual_before_mean: float = 0.0
    loop_residual_after_mean: float = 0.0
    max_correction_t: float = 0.0
    max_correction_r_deg: float = 0.0
    accepted: bool = False
    rejection_reason: str = ""


# ============================================================================
# 2. KeyframeRecord & Database
# ============================================================================


@dataclass
class KeyframeRecord:
    """A single keyframe record in the unified keyframe database.

    Each keyframe has a globally unique keyframe_id (= frame_idx in the current
    implementation). All poses are in global coordinates (cam.T = W2C, c2w = C2W).
    """
    keyframe_id: int
    frame_idx: int
    submap_id: int
    c2w_global: np.ndarray   # 4x4 global C2W
    w2c_global: np.ndarray   # 4x4 global W2C (= inv(c2w_global))
    rgb_path: Optional[str] = None
    depth_path: Optional[str] = None
    intrinsics: Optional[Dict[str, Any]] = None  # fx, fy, cx, cy, W, H
    cosplace_descriptor: Optional[np.ndarray] = None
    is_submap_seed: bool = False

    def __post_init__(self):
        if self.w2c_global is None and self.c2w_global is not None:
            self.w2c_global = np.linalg.inv(self.c2w_global)
        if self.c2w_global is None and self.w2c_global is not None:
            self.c2w_global = np.linalg.inv(self.w2c_global)


def build_keyframe_database(
    submap_keyframe_poses: Dict[int, Dict[int, np.ndarray]],
    submap_image_paths: Optional[Dict[int, List[str]]] = None,
    submap_seed_c2w: Optional[Dict[int, np.ndarray]] = None,
    intrinsics: Optional[Dict[str, Any]] = None,
    submap_depth_paths: Optional[Dict[int, List[str]]] = None,
) -> Dict[int, KeyframeRecord]:
    """Build a unified keyframe database from per-submap ckpt metadata.

    Args:
        submap_keyframe_poses: {submap_id: {frame_idx: 4x4 c2w}} from ckpt.
        submap_image_paths: {submap_id: [str, ...]} paths to saved kf images.
        submap_seed_c2w: {submap_id: 4x4 c2w} seed pose per submap.
        intrinsics: dict with fx, fy, cx, cy, width, height.
        submap_depth_paths: {submap_id: [str, ...]} paths to saved depth maps.

    Returns:
        keyframe_db: {keyframe_id: KeyframeRecord} keyed by frame_idx.
    """
    db: Dict[int, KeyframeRecord] = {}
    if submap_image_paths is None:
        submap_image_paths = {}
    if submap_seed_c2w is None:
        submap_seed_c2w = {}
    if submap_depth_paths is None:
        submap_depth_paths = {}

    for submap_id, kf_poses in sorted(submap_keyframe_poses.items()):
        seed = submap_seed_c2w.get(submap_id, np.eye(4))
        img_list = submap_image_paths.get(submap_id, [])
        dpt_list = submap_depth_paths.get(submap_id, [])

        for kf_idx in sorted(kf_poses.keys()):
            c2w = np.array(kf_poses[kf_idx], dtype=np.float64)
            w2c = np.linalg.inv(c2w)

            # Match rgb_path and depth_path by index within the submap's lists
            sorted_kf = sorted(kf_poses.keys())
            pos = sorted_kf.index(kf_idx) if kf_idx in sorted_kf else -1
            rgb_path = img_list[pos] if 0 <= pos < len(img_list) else None
            depth_path = dpt_list[pos] if 0 <= pos < len(dpt_list) else None

            is_seed = np.allclose(c2w, seed, atol=1e-6)

            db[kf_idx] = KeyframeRecord(
                keyframe_id=kf_idx,
                frame_idx=kf_idx,
                submap_id=submap_id,
                c2w_global=c2w,
                w2c_global=w2c,
                rgb_path=rgb_path,
                depth_path=depth_path,
                intrinsics=intrinsics,
                is_submap_seed=is_seed,
            )

    return db


def keyframe_db_stats(db: Dict[int, KeyframeRecord]) -> Dict[str, Any]:
    """Compute summary statistics for a keyframe database."""
    if not db:
        return {"total_keyframes": 0}

    submap_counts: Dict[int, int] = {}
    missing_rgb = 0
    missing_depth = 0
    missing_descriptor = 0
    seed_count = 0

    for rec in db.values():
        submap_counts[rec.submap_id] = submap_counts.get(rec.submap_id, 0) + 1
        if rec.rgb_path is None:
            missing_rgb += 1
        if rec.depth_path is None:
            missing_depth += 1
        if rec.cosplace_descriptor is None:
            missing_descriptor += 1
        if rec.is_submap_seed:
            seed_count += 1

    return {
        "total_keyframes": len(db),
        "submap_count": len(submap_counts),
        "per_submap": dict(sorted(submap_counts.items())),
        "missing_rgb": missing_rgb,
        "missing_depth": missing_depth,
        "missing_descriptor": missing_descriptor,
        "seed_keyframes": seed_count,
    }


# ============================================================================
# 3. Keyframe Retrieval
# ============================================================================


@dataclass
class KeyframeRetrievalCandidate:
    """A keyframe pair candidate from CosPlace retrieval.

    NOT a verified loop edge — this is the raw retrieval output that must
    pass through Reloc3R coarse pose estimation + depth verification before
    becoming a VerifiedLoopEdge.
    """
    query_keyframe_id: int
    target_keyframe_id: int
    query_submap_id: int
    target_submap_id: int
    cosplace_score: float = 0.0
    temporal_gap: int = 0
    is_mutual: bool = False
    accepted_by_retrieval: bool = False
    rejection_reason: str = ""


def retrieve_keyframe_loop_candidates(
    query_keyframe_id: int,
    keyframe_db: Dict[int, KeyframeRecord],
    config: Optional[Dict[str, Any]] = None,
) -> List[KeyframeRetrievalCandidate]:
    """Retrieve loop closure candidates at keyframe level.

    Reads CosPlace descriptors from KeyframeRecord.cosplace_descriptor.

    Args:
        query_keyframe_id: The query keyframe's ID (= frame_idx).
        keyframe_db: Unified keyframe database with populated descriptors.
        config: LoopClosure config dict.

    Returns:
        List of KeyframeRetrievalCandidate sorted by cosplace_score descending.
    """
    if config is None:
        return []

    if query_keyframe_id not in keyframe_db:
        return []

    query_rec = keyframe_db[query_keyframe_id]
    query_desc = query_rec.cosplace_descriptor
    if query_desc is None:
        return []

    query_submap = query_rec.submap_id

    # Config
    top_k = config.get("keyframe_retrieval_top_k", 10)
    min_score = config.get("keyframe_retrieval_min_score", 0.50)
    min_temporal_gap = config.get("keyframe_retrieval_min_temporal_gap", 10)
    min_submap_gap = config.get("min_interval", 3)
    use_mutual = config.get("keyframe_retrieval_mutual_nearest", True)
    mutual_top_n = config.get("keyframe_retrieval_mutual_top_n", 20)

    query_desc = np.array(query_desc, dtype=np.float32)
    query_norm = np.linalg.norm(query_desc)
    if query_norm < 1e-10:
        return []

    raw_matches = 0
    candidates: List[KeyframeRetrievalCandidate] = []

    for target_id, target_rec in keyframe_db.items():
        if target_id == query_keyframe_id:
            continue
        if target_rec.submap_id == query_submap:
            continue
        submap_diff = abs(query_submap - target_rec.submap_id)
        if submap_diff < min_submap_gap:
            continue
        temporal_gap = abs(query_keyframe_id - target_id)
        if temporal_gap < min_temporal_gap:
            continue

        target_desc = target_rec.cosplace_descriptor
        if target_desc is None:
            continue
        raw_matches += 1

        target_desc = np.array(target_desc, dtype=np.float32)
        target_norm = np.linalg.norm(target_desc)
        if target_norm < 1e-10:
            continue

        score = float(np.dot(query_desc, target_desc) / (query_norm * target_norm))

        candidate = KeyframeRetrievalCandidate(
            query_keyframe_id=query_keyframe_id,
            target_keyframe_id=target_id,
            query_submap_id=query_submap,
            target_submap_id=target_rec.submap_id,
            cosplace_score=score,
            temporal_gap=temporal_gap,
        )

        if score < min_score:
            candidate.rejection_reason = f"score={score:.3f} < min={min_score}"
        else:
            candidate.accepted_by_retrieval = True

        candidates.append(candidate)

    candidates.sort(key=lambda c: c.cosplace_score, reverse=True)

    # Mutual nearest check (optional)
    if use_mutual and candidates:
        accepted = [c for c in candidates if c.accepted_by_retrieval]
        if len(accepted) >= 2:
            top_accepted = accepted[:top_k]
            for cand in top_accepted:
                target_neighbors = _find_nearest_keyframes(
                    cand.target_keyframe_id, keyframe_db,
                    top_n=mutual_top_n, min_submap_gap=min_submap_gap,
                    min_temporal_gap=min_temporal_gap,
                )
                query_in_target_top = any(
                    n_id == query_keyframe_id for n_id, _ in target_neighbors
                )
                cand.is_mutual = query_in_target_top

    result = [c for c in candidates if c.accepted_by_retrieval][:top_k]
    if not result:
        result = candidates[:5]

    return result


def _find_nearest_keyframes(
    query_id: int,
    keyframe_db: Dict[int, KeyframeRecord],
    top_n: int = 20,
    min_submap_gap: int = 3,
    min_temporal_gap: int = 10,
) -> List[tuple]:
    """Find nearest keyframes by cosine similarity from KeyframeRecord descriptors."""
    query_rec = keyframe_db.get(query_id)
    if query_rec is None or query_rec.cosplace_descriptor is None:
        return []

    query_desc = np.array(query_rec.cosplace_descriptor, dtype=np.float32)
    query_norm = np.linalg.norm(query_desc)
    if query_norm < 1e-10:
        return []

    scored = []
    for target_id, target_rec in keyframe_db.items():
        if target_id == query_id:
            continue
        if target_rec.submap_id == query_rec.submap_id:
            continue
        if abs(query_rec.submap_id - target_rec.submap_id) < min_submap_gap:
            continue
        if abs(query_id - target_id) < min_temporal_gap:
            continue
        if target_rec.cosplace_descriptor is None:
            continue

        target_desc = np.array(target_rec.cosplace_descriptor, dtype=np.float32)
        target_norm = np.linalg.norm(target_desc)
        if target_norm < 1e-10:
            continue

        score = float(np.dot(query_desc, target_desc) / (query_norm * target_norm))
        scored.append((target_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ============================================================================
# 4. Refinement: Depth-Verified → VerifiedLoopEdge (Stage 5)
# ============================================================================


def _rot_error_deg(T_a, T_b):
    """Rotation error in degrees between two 4x4 SE3 matrices."""
    R = T_a[:3, :3] @ T_b[:3, :3].T
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def refine_keyframe_loop_edge(
    depth_verified_pair,       # DepthVerifiedPair from loop_depth_verifier
    source_record: KeyframeRecord,
    target_record: KeyframeRecord,
    config: Optional[Dict[str, Any]] = None,
) -> VerifiedLoopEdge:
    """Produce a VerifiedLoopEdge from a depth-verified pair.

    Args:
        depth_verified_pair: DepthVerifiedPair with accepted_by_depth=True.
        source_record: KeyframeRecord for source keyframe.
        target_record: KeyframeRecord for target keyframe.
        config: LoopClosure config dict.

    Returns:
        VerifiedLoopEdge with accepted_for_pgo decision.
    """
    edge = VerifiedLoopEdge(
        source_keyframe_id=source_record.keyframe_id,
        target_keyframe_id=target_record.keyframe_id,
        source_submap_id=source_record.submap_id,
        target_submap_id=target_record.submap_id,
        source_c2w_original=source_record.c2w_global.copy(),
        target_c2w_original=target_record.c2w_global.copy(),
    )

    if config is None:
        config = {}

    refine_cfg = config.get("render_refine", {})
    T_refined = depth_verified_pair.T_target_from_source.copy()  # T_target_from_source

    edge.verification_metrics = {
        "depth_overlap": depth_verified_pair.depth_overlap,
        "depth_rmse": depth_verified_pair.depth_rmse,
        "depth_inlier_ratio": depth_verified_pair.depth_inlier_ratio,
        "scale_used": depth_verified_pair.scale_used,
        "refinement_method": "depth_only",
    }

    allow_depth_only = refine_cfg.get("allow_depth_verified_edge_without_render_refine", True)
    if not allow_depth_only:
        edge.rejection_reason = "render_refine_disabled_and_depth_only_not_allowed"
        return edge

    # ---- Delta gate ----
    odom_T = np.linalg.inv(target_record.c2w_global) @ source_record.c2w_global
    max_dt_odom = refine_cfg.get("max_delta_t_from_odom", 50.0)
    max_dr_odom = refine_cfg.get("max_delta_r_deg_from_odom", 90.0)
    dt_odom = float(np.linalg.norm((T_refined @ np.linalg.inv(odom_T))[:3, 3]))
    dr_odom = _rot_error_deg(T_refined, odom_T)
    edge.verification_metrics["delta_t_from_odom"] = dt_odom
    edge.verification_metrics["delta_r_deg_from_odom"] = dr_odom

    if dt_odom > max_dt_odom or dr_odom > max_dr_odom:
        edge.rejection_reason = f"delta_from_odom_too_large dt={dt_odom:.3f}m dr={dr_odom:.1f}deg"
        return edge

    max_depth_rmse = refine_cfg.get("max_final_depth_rmse", 0.80)
    if depth_verified_pair.depth_rmse > max_depth_rmse:
        edge.rejection_reason = f"final_depth_rmse_{depth_verified_pair.depth_rmse:.4f}_gt_{max_depth_rmse}"
        return edge

    edge.T_source_to_target = T_refined
    edge.information = np.eye(6) * float(
        np.clip(depth_verified_pair.depth_inlier_ratio * 100, 10, 200)
    )
    edge.accepted_for_pgo = True
    return edge


# ============================================================================
# 5. Keyframe Pose Graph Construction (Stage 6)
# ============================================================================


@dataclass
class KeyframePoseGraph:
    """A keyframe-level pose graph ready for optimization.

    Nodes are keyframe global C2W poses. Edges include temporal (adjacent kfs),
    handoff (cross-submap boundary), and loop (verified closure) constraints.

    The first node is fixed (gauge freedom removal).
    """
    nodes: List[KeyframeNode] = field(default_factory=list)
    edges: List[KeyframeEdge] = field(default_factory=list)
    num_temporal_edges: int = 0
    num_handoff_edges: int = 0
    num_loop_edges: int = 0


# ---- Edge builders ----

def _build_temporal_edges(
    nodes_sorted: List[KeyframeNode],
    odom_info_scale: float = 100.0,
) -> List[KeyframeEdge]:
    """Build temporal edges between consecutive keyframes ordered by frame_idx.

    T_source_to_target = inv(c2w_source) @ c2w_target.
    """
    edges = []
    for i in range(len(nodes_sorted) - 1):
        src = nodes_sorted[i]
        tgt = nodes_sorted[i + 1]
        T_st = np.linalg.inv(src.c2w_original) @ tgt.c2w_original
        info = np.eye(6) * odom_info_scale
        edges.append(KeyframeEdge(
            source_keyframe_id=src.keyframe_id,
            target_keyframe_id=tgt.keyframe_id,
            T_source_to_target=T_st,
            edge_type="temporal",
            information=info,
            diagnostics={"submap_id": src.submap_id},
        ))
    return edges


def _build_handoff_edges(
    nodes_sorted: List[KeyframeNode],
    handoff_info_scale: float = 50.0,
) -> List[KeyframeEdge]:
    """Build handoff edges at submap boundaries.

    A handoff edge connects the last keyframe of submap N to the first keyframe
    of submap N+1. It is a temporal edge with lower information weight (soft constraint).
    """
    edges = []
    for i in range(len(nodes_sorted) - 1):
        src = nodes_sorted[i]
        tgt = nodes_sorted[i + 1]
        if src.submap_id != tgt.submap_id:
            T_st = np.linalg.inv(src.c2w_original) @ tgt.c2w_original
            info = np.eye(6) * handoff_info_scale
            edges.append(KeyframeEdge(
                source_keyframe_id=src.keyframe_id,
                target_keyframe_id=tgt.keyframe_id,
                T_source_to_target=T_st,
                edge_type="handoff",
                information=info,
                diagnostics={
                    "from_submap": src.submap_id,
                    "to_submap": tgt.submap_id,
                },
            ))
    return edges


def _build_loop_edges(
    verified_loop_edges_list: List[VerifiedLoopEdge],
) -> List[KeyframeEdge]:
    """Convert VerifiedLoopEdges into KeyframeEdges of type 'loop'.

    Only edges with accepted_for_pgo=True are included.
    """
    edges = []
    for vle in verified_loop_edges_list:
        if not vle.accepted_for_pgo:
            continue
        info = vle.information if vle.information is not None else np.eye(6)
        edges.append(KeyframeEdge(
            source_keyframe_id=vle.source_keyframe_id,
            target_keyframe_id=vle.target_keyframe_id,
            T_source_to_target=vle.T_source_to_target,
            edge_type="loop",
            information=info,
            diagnostics={
                "source_submap": vle.source_submap_id,
                "target_submap": vle.target_submap_id,
                "verification_metrics": vle.verification_metrics,
            },
        ))
    return edges


# ---- Main graph builder ----

def build_keyframe_pose_graph(
    keyframe_db: Dict[int, KeyframeRecord],
    verified_loop_edges: Optional[List[VerifiedLoopEdge]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> KeyframePoseGraph:
    """Build a keyframe-level pose graph from the unified keyframe database.

    Nodes are keyframe global C2W poses. The first keyframe (lowest frame_idx)
    is fixed to remove gauge freedom.

    Args:
        keyframe_db: {kf_idx: KeyframeRecord} from build_keyframe_database.
        verified_loop_edges: List of VerifiedLoopEdge with accepted_for_pgo=True.
        config: Optional config dict (reserved for future use).

    Returns:
        KeyframePoseGraph with nodes, edges, and edge type counts.
    """
    if verified_loop_edges is None:
        verified_loop_edges = []
    if config is None:
        config = {}

    # Build nodes sorted by frame_idx
    sorted_ids = sorted(keyframe_db.keys())
    nodes = []
    for kf_id in sorted_ids:
        rec = keyframe_db[kf_id]
        node = KeyframeNode(
            keyframe_id=rec.keyframe_id,
            submap_id=rec.submap_id,
            frame_idx=rec.frame_idx,
            c2w_original=rec.c2w_global.copy(),
        )
        nodes.append(node)

    # Fix first node (gauge freedom)
    if len(nodes) > 0:
        nodes[0].fixed = True

    # Edge config
    odom_scale = config.get("keyframe_pgo_odom_info_scale", 100.0)
    handoff_scale = config.get("keyframe_pgo_handoff_info_scale", 50.0)

    # Build edges
    temporal_edges = _build_temporal_edges(nodes, odom_info_scale=odom_scale)
    handoff_edges = _build_handoff_edges(nodes, handoff_info_scale=handoff_scale)
    loop_edges = _build_loop_edges(verified_loop_edges)

    all_edges = temporal_edges + handoff_edges + loop_edges

    return KeyframePoseGraph(
        nodes=nodes,
        edges=all_edges,
        num_temporal_edges=len(temporal_edges),
        num_handoff_edges=len(handoff_edges),
        num_loop_edges=len(loop_edges),
    )


# ============================================================================
# 6. Keyframe PGO Trial & Safety Evaluation (Stage 7)
# ============================================================================


def _compute_edge_residual(T_measured, T_expected):
    """Compute translation and rotation residual for a single edge."""
    delta = T_measured @ np.linalg.inv(T_expected)
    t_err = float(np.linalg.norm(delta[:3, 3]))
    r_err = _rot_error_deg(T_measured, T_expected)
    return t_err, r_err


def _compute_all_residuals(graph: KeyframePoseGraph,
                           c2w_dict: Dict[int, np.ndarray]):
    """Compute per-edge residuals given current node C2W estimates."""
    odom_res = []
    loop_res = []
    for edge in graph.edges:
        src_c2w = c2w_dict.get(edge.source_keyframe_id)
        tgt_c2w = c2w_dict.get(edge.target_keyframe_id)
        if src_c2w is None or tgt_c2w is None:
            continue
        # Expected: T_expected = inv(src) @ tgt from current estimates
        T_expected = np.linalg.inv(src_c2w) @ tgt_c2w
        t_err, r_err = _compute_edge_residual(edge.T_source_to_target, T_expected)
        if edge.edge_type == "loop":
            loop_res.append((edge, t_err, r_err))
        elif edge.edge_type in ("temporal", "handoff"):
            odom_res.append((edge, t_err, r_err))
    return odom_res, loop_res


def run_keyframe_pgo_trial(
    graph: KeyframePoseGraph,
    config: Optional[Dict[str, Any]] = None,
) -> KeyframePGOResult:
    """Run keyframe-level PGO trial in memory using Open3D.

    Does NOT modify ckpt, cam.T, or Gaussian params.

    Args:
        graph: KeyframePoseGraph from build_keyframe_pose_graph.
        config: PGO trial config dict.

    Returns:
        KeyframePGOResult with optimized poses, corrections, and residuals.
    """
    import open3d as o3d

    if config is None:
        config = {}

    result = KeyframePGOResult()

    if len(graph.nodes) == 0:
        result.rejection_reason = "no_nodes"
        return result

    # Build original C2W dict
    id_to_idx = {}
    for i, node in enumerate(graph.nodes):
        result.original_keyframe_c2w[node.keyframe_id] = node.c2w_original.copy()
        id_to_idx[node.keyframe_id] = i

    # ---- Compute residuals before ----
    odom_before, loop_before = _compute_all_residuals(
        graph, result.original_keyframe_c2w)
    result.odom_residual_before_mean = (
        float(np.mean([r[1] for r in odom_before])) if odom_before else 0.0
    )
    result.loop_residual_before_mean = (
        float(np.mean([r[1] for r in loop_before])) if loop_before else 0.0
    )

    # ---- Build O3D PoseGraph ----
    o3d_graph = o3d.pipelines.registration.PoseGraph()

    # Edge prune threshold
    prune_th = config.get("pgo_edge_prune_threshold", 0.25)
    voxel_size = config.get("pgo_voxel_size", 0.02)

    for node in graph.nodes:
        o3d_node = o3d.pipelines.registration.PoseGraphNode(
            node.c2w_original.copy()
        )
        o3d_graph.nodes.append(o3d_node)

    for edge in graph.edges:
        si = id_to_idx[edge.source_keyframe_id]
        ti = id_to_idx[edge.target_keyframe_id]
        info = edge.information if edge.information is not None else np.eye(6)
        is_uncertain = (edge.edge_type == "loop")
        # O3D convention: node_target = node_source @ T_edge
        # where T_edge = inv(pose_target) @ pose_source = inv(T_source_to_target)
        T_o3d = np.linalg.inv(edge.T_source_to_target)
        o3d_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                si, ti,
                T_o3d,
                info.copy(),
                uncertain=is_uncertain,
            )
        )

    # Run optimization
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 1.5,
        edge_prune_threshold=prune_th,
        reference_node=0,
    )
    try:
        o3d.pipelines.registration.global_optimization(
            o3d_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    except Exception as e:
        result.rejection_reason = f"o3d_optimization_error: {e}"
        return result

    # ---- Extract optimized poses ----
    for i, node in enumerate(graph.nodes):
        result.optimized_keyframe_c2w[node.keyframe_id] = (
            np.array(o3d_graph.nodes[i].pose, dtype=np.float64)
        )

    # ---- Compute corrections ----
    for kf_id in result.optimized_keyframe_c2w:
        opt_c2w = result.optimized_keyframe_c2w[kf_id]
        orig_c2w = result.original_keyframe_c2w[kf_id]
        result.keyframe_corrections[kf_id] = opt_c2w @ np.linalg.inv(orig_c2w)

    # ---- Correction stats ----
    corrections_t = [np.linalg.norm(c[:3, 3]) for c in result.keyframe_corrections.values()]
    corrections_r = [
        _rot_error_deg(np.vstack([
            np.hstack([c[:3, :3], np.zeros((3, 1))]),
            np.array([[0, 0, 0, 1]]),
        ]), np.eye(4))
        for c in result.keyframe_corrections.values()
    ]
    result.max_correction_t = float(np.max(corrections_t)) if corrections_t else 0.0
    result.max_correction_r_deg = float(np.max(corrections_r)) if corrections_r else 0.0

    # ---- Compute residuals after ----
    odom_after, loop_after = _compute_all_residuals(
        graph, result.optimized_keyframe_c2w)
    result.odom_residual_after_mean = (
        float(np.mean([r[1] for r in odom_after])) if odom_after else 0.0
    )
    result.loop_residual_after_mean = (
        float(np.mean([r[1] for r in loop_after])) if loop_after else 0.0
    )

    return result


def evaluate_keyframe_pgo_result(
    result: KeyframePGOResult,
    graph: Optional[KeyframePoseGraph] = None,
    config: Optional[Dict[str, Any]] = None,
) -> KeyframePGOResult:
    """Evaluate safety of a keyframe PGO trial result.

    Sets result.accepted and result.rejection_reason based on:
      1. Max correction magnitude
      2. Odom residual increase ratio
      3. Loop residual decrease ratio
      4. Robust edge pruning if individual loop residuals are too large

    This function does NOT modify ckpt, cam.T, or Gaussian params.

    Args:
        result: KeyframePGOResult from run_keyframe_pgo_trial.
        graph: Original KeyframePoseGraph (needed for edge pruning retry).
        config: PGO safety config dict.

    Returns:
        Same result object with accepted / rejection_reason set.
    """
    if config is None:
        config = {}
    safety = config.get("keyframe_pgo_safety", config)  # allow nested or flat

    # Thresholds
    max_corr_t = safety.get("max_correction_t", 0.30)
    max_corr_r = safety.get("max_correction_r_deg", 8.0)
    max_odom_ratio = safety.get("max_odom_residual_increase_ratio", 1.20)
    min_loop_decrease = safety.get("min_loop_residual_decrease_ratio", 0.30)
    max_single_loop_t = safety.get("max_single_loop_residual_after_t", 0.15)
    max_single_loop_r = safety.get("max_single_loop_residual_after_r_deg", 5.0)
    min_loop_edges = safety.get("min_verified_loop_edges", 1)
    max_retries = safety.get("max_pgo_retries", 2)

    if result.rejection_reason:
        return result

    # ---- Gate 1: Max correction ----
    if result.max_correction_t > max_corr_t:
        result.rejection_reason = (
            f"max_correction_t_{result.max_correction_t:.3f}m_gt_{max_corr_t}"
        )
        return result
    if result.max_correction_r_deg > max_corr_r:
        result.rejection_reason = (
            f"max_correction_r_{result.max_correction_r_deg:.1f}deg_gt_{max_corr_r}"
        )
        return result

    # ---- Gate 2: Odom residual increase ----
    if result.odom_residual_before_mean > 1e-8:
        ratio = result.odom_residual_after_mean / result.odom_residual_before_mean
        if ratio > max_odom_ratio:
            result.rejection_reason = (
                f"odom_residual_increase_ratio_{ratio:.2f}_gt_{max_odom_ratio}"
            )
            return result

    # ---- Gate 3: Loop residual decrease ----
    if result.loop_residual_before_mean > 1e-4:
        ratio = result.loop_residual_after_mean / (result.loop_residual_before_mean + 1e-8)
        if ratio > (1.0 - min_loop_decrease):
            result.rejection_reason = (
                f"loop_residual_not_decreased_enough_ratio_{ratio:.2f}"
            )
            return result

    # ---- Gate 4: Single edge residual check + robust pruning ----
    if graph is not None:
        odom_after, loop_after = _compute_all_residuals(
            graph, result.optimized_keyframe_c2w)
        # Check for large individual loop residuals
        bad_loops = [(e, t, r) for e, t, r in loop_after
                     if t > max_single_loop_t or r > max_single_loop_r]
        if bad_loops and max_retries > 0:
            # Prune worst loop edges and retry
            bad_loops.sort(key=lambda x: x[1] + x[2] * 0.1, reverse=True)
            pruned_ids = {(e.source_keyframe_id, e.target_keyframe_id)
                          for e, _, _ in bad_loops[:2]}
            filtered_edges = [e for e in graph.edges
                              if (e.source_keyframe_id, e.target_keyframe_id) not in pruned_ids
                              or e.edge_type != "loop"]
            filtered_loop_edges = len([e for e in graph.edges if e.edge_type == "loop"]) - len(
                [p for p in pruned_ids if p in {
                    (e.source_keyframe_id, e.target_keyframe_id)
                    for e in graph.edges if e.edge_type == "loop"}])

            if filtered_loop_edges < min_loop_edges:
                result.rejection_reason = (
                    f"insufficient_loop_edges_after_pruning_{filtered_loop_edges}"
                )
                return result

    # ---- Accepted ----
    result.accepted = True
    return result


# ============================================================================
# 7. Trajectory Application (Stage 8)
# ============================================================================


def apply_keyframe_pgo_to_trajectory(
    result: KeyframePGOResult,
    all_frame_c2w: Dict[int, np.ndarray],
    keyframe_db: Optional[Dict[int, KeyframeRecord]] = None,
    config: Optional[Dict[str, Any]] = None,
    save_dir: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Apply accepted keyframe PGO corrections to a full trajectory.

    Keyframes use optimized_c2w directly. Non-keyframes use the correction
    from their nearest keyframe (by frame_idx distance).

    The dominant correction is a LEFT-multiply: corrected_c2w = delta @ original_c2w.

    Args:
        result: Accepted KeyframePGOResult (result.accepted must be True).
        all_frame_c2w: {frame_id: 4x4 C2W} for all frames (KFs + non-KFs).
        keyframe_db: Keyframe database (for keyframe_id lookup).
        config: Optional config dict.
        save_dir: If set, write trajectory files and correction JSON.

    Returns:
        corrected_c2w: {frame_id: 4x4 C2W} dictionary.
    """
    if config is None:
        config = {}

    if not result.accepted:
        raise ValueError("Cannot apply PGO result that was not accepted. "
                         f"Rejection reason: {result.rejection_reason}")

    strategy = config.get("trajectory_correction_strategy", "nearest_keyframe")
    online = config.get("online_apply_enabled", False)

    # Build list of keyframe IDs and their corrections
    kf_ids = sorted(result.keyframe_corrections.keys())

    corrected_c2w: Dict[int, np.ndarray] = {}

    for frame_id, original_c2w in all_frame_c2w.items():
        if frame_id in result.optimized_keyframe_c2w:
            # Keyframe: use optimized C2W directly
            corrected_c2w[frame_id] = result.optimized_keyframe_c2w[frame_id].copy()
        else:
            # Non-keyframe: apply nearest keyframe's correction
            delta = _find_nearest_kf_correction(frame_id, kf_ids, result)
            corrected_c2w[frame_id] = delta @ np.array(original_c2w, dtype=np.float64)

    # --- Save outputs ---
    if save_dir is not None:
        import json as _json
        os.makedirs(save_dir, exist_ok=True)

        # Optimized keyframe poses
        opt_serializable = {str(k): v.tolist() for k, v in result.optimized_keyframe_c2w.items()}
        with open(os.path.join(save_dir, "keyframe_pgo_optimized_poses.json"), "w") as f:
            _json.dump(opt_serializable, f, indent=2)

        # Keyframe corrections
        corr_serializable = {str(k): v.tolist() for k, v in result.keyframe_corrections.items()}
        with open(os.path.join(save_dir, "keyframe_pgo_corrections.json"), "w") as f:
            _json.dump(corr_serializable, f, indent=2)

        # Trajectory before
        with open(os.path.join(save_dir, "trajectory_before_keyframe_pgo.txt"), "w") as f:
            for frame_id in sorted(all_frame_c2w.keys()):
                t = all_frame_c2w[frame_id][:3, 3]
                f.write(f"{frame_id} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")

        # Trajectory after
        with open(os.path.join(save_dir, "trajectory_after_keyframe_pgo.txt"), "w") as f:
            for frame_id in sorted(corrected_c2w.keys()):
                t = corrected_c2w[frame_id][:3, 3]
                f.write(f"{frame_id} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n")

    return corrected_c2w


def _find_nearest_kf_correction(
    frame_id: int,
    kf_ids: List[int],
    result: KeyframePGOResult,
) -> np.ndarray:
    """Find the correction delta from the nearest keyframe by frame_idx distance."""
    # Find closest keyframe
    best_dist = 10**9
    best_kf = kf_ids[0]
    for kf in kf_ids:
        dist = abs(frame_id - kf)
        if dist < best_dist:
            best_dist = dist
            best_kf = kf
    return result.keyframe_corrections.get(best_kf, np.eye(4))


# ============================================================================
# 8. Gaussian Correction (Stage 9)
# ============================================================================


def aggregate_submap_corrections(
    result: KeyframePGOResult,
    keyframe_db: Dict[int, KeyframeRecord],
) -> Dict[int, List[np.ndarray]]:
    """Collect per-submap keyframe corrections from a PGO result.

    Returns {submap_id: [delta_4x4, ...]} for all keyframes in that submap.
    """
    submap_deltas: Dict[int, List[np.ndarray]] = {}
    for kf_id, delta in result.keyframe_corrections.items():
        rec = keyframe_db.get(kf_id)
        if rec is None:
            continue
        sid = rec.submap_id
        if sid not in submap_deltas:
            submap_deltas[sid] = []
        submap_deltas[sid].append(delta.copy())
    return submap_deltas


def compute_submap_median_correction(deltas: List[np.ndarray]) -> np.ndarray:
    """Compute a robust submap correction from a list of keyframe corrections.

    Uses:
      - Rotation: chordal L2 mean (SVD of sum of rotation matrices), then
        project to SO(3).
      - Translation: element-wise median.
    """
    if len(deltas) == 0:
        return np.eye(4)
    if len(deltas) == 1:
        return deltas[0].copy()

    # Rotation: chordal mean
    R_sum = np.zeros((3, 3))
    for d in deltas:
        R_sum += d[:3, :3]
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt

    # Translation: median
    t_all = np.stack([d[:3, 3] for d in deltas], axis=0)
    t_median = np.median(t_all, axis=0)

    T = np.eye(4)
    T[:3, :3] = R_mean
    T[:3, 3] = t_median
    return T


def apply_keyframe_corrections_to_gaussians(
    gaussian_xyz: np.ndarray,
    result: Optional[KeyframePGOResult] = None,
    keyframe_db: Optional[Dict[int, KeyframeRecord]] = None,
    owner_keyframe_ids: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Apply keyframe PGO corrections to Gaussian xyz positions.

    Supports three modes:
      - none: return xyz unchanged.
      - submap_median_from_keyframes: compute per-submap median correction
        and apply to all Gaussians in that submap.
      - owner_keyframe: each Gaussian gets its owner keyframe's correction.

    Args:
        gaussian_xyz: (N, 3) float64 array of Gaussian positions (global coords).
        result: Accepted KeyframePGOResult.
        keyframe_db: Keyframe database for submap/owner lookup.
        owner_keyframe_ids: (N,) int array mapping each Gaussian to its owner KF.
        config: MapCorrection config dict.

    Returns:
        corrected_xyz: (N, 3) float64 array.
    """
    if config is None:
        config = {}
    mode = config.get("mode", "none")
    apply_online = config.get("apply_online", False)

    if mode == "none":
        return gaussian_xyz.copy()

    if result is None or not result.accepted:
        return gaussian_xyz.copy()

    corrected = gaussian_xyz.copy()

    if mode == "submap_median_from_keyframes" and keyframe_db is not None:
        submap_deltas = aggregate_submap_corrections(result, keyframe_db)
        # Build frame_idx → submap_id mapping
        kf_to_submap = {rec.keyframe_id: rec.submap_id
                        for rec in keyframe_db.values()}

        for sid, deltas in submap_deltas.items():
            if len(deltas) == 0:
                continue
            T = compute_submap_median_correction(deltas)
            R, t = T[:3, :3], T[:3, 3]
            # Apply to all Gaussians in this submap (requires submap membership info)
            # Without per-Gaussian submap IDs, apply globally if single submap
            # or skip if we can't determine per-Gaussian submap.
            if len(submap_deltas) == 1:
                corrected = (corrected @ R.T) + t
            # else: TODO — need per-Gaussian submap_id for multi-submap case
        return corrected

    if mode == "owner_keyframe" and owner_keyframe_ids is not None:
        for kf_id, delta in result.keyframe_corrections.items():
            mask = owner_keyframe_ids == kf_id
            if mask.sum() == 0:
                continue
            R, t = delta[:3, :3], delta[:3, 3]
            corrected[mask] = (corrected[mask] @ R.T) + t
        return corrected

    # Fallback: apply first (or only) keyframe's correction globally
    if len(result.keyframe_corrections) == 1:
        delta = list(result.keyframe_corrections.values())[0]
        R, t = delta[:3, :3], delta[:3, 3]
        corrected = (corrected @ R.T) + t

    return corrected
