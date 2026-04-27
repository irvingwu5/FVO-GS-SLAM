# utils/registration_2dgs.py
# LoopSplat-inspired 2DGS submap registration.
# Uses ckpt-stored _normal (FDN-derived) for normal-aware ICP.

import os, numpy as np, torch, torch.nn.functional as F, open3d as o3d
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import roma
from utils.logging_utils import Log


# ============================================================================
# Helpers: normal extraction from 2DGS params
# ============================================================================

def compute_normals_from_rotation(rotation, quat_order="wxyz"):
    """从 2DGS _rotation (wxyz) 提取世界坐标系 surfel 法线 (rotation 局部 z 轴)。"""
    if quat_order == "wxyz":
        rot_roma = rotation[:, [1, 2, 3, 0]]
    else:
        rot_roma = rotation
    rot_mat = roma.unitquat_to_rotmat(rot_roma)
    normal = rot_mat[:, :, 2]  # (N,3) world-frame normal
    return F.normalize(normal.float(), dim=-1, eps=1e-8)


def resolve_2dgs_normals(gp: dict, prefer_fdn=True) -> Tuple[torch.Tensor, str]:
    """从 ckpt gaussian_params 解析法线。
    返回 (normal, source_str)。
    """
    if prefer_fdn and "_normal" in gp and gp["_normal"] is not None:
        n = gp["_normal"].float()
        if n.ndim == 2 and n.shape[1] == 3 and n.shape[0] > 0:
            return F.normalize(n, dim=-1, eps=1e-8), "fdn_ckpt"
    if "_rotation" in gp and gp["_rotation"] is not None:
        return compute_normals_from_rotation(gp["_rotation"].float()), "rotation_fallback"
    return None, "none"


# ============================================================================
# Submap Data
# ============================================================================

@dataclass
class Submap2DGS:
    submap_id: int
    ckpt_path: str
    xyz: torch.Tensor          # (N,3) CPU
    normal: torch.Tensor       # (N,3) CPU, normalized
    opacity: torch.Tensor      # (N,) CPU
    rotation: Optional[torch.Tensor] = None  # (N,4) CPU wxyz
    gaussian_params: dict = field(default_factory=dict)
    normal_source: str = "unknown"
    keyframe_ids: List[int] = field(default_factory=list)
    seed_pose: Optional[np.ndarray] = None
    correction: Optional[np.ndarray] = None


def load_submap_from_ckpt(ckpt_path: str, submap_id: int = None) -> Submap2DGS:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    gp = ckpt.get("gaussian_params", ckpt)
    if submap_id is None:
        submap_id = int(os.path.splitext(os.path.basename(ckpt_path))[0])

    xyz = gp["_xyz"].float()
    opacity = gp["_opacity"].float().squeeze(-1)
    rotation = gp.get("_rotation", None)
    if rotation is not None:
        rotation = rotation.float()

    normal, normal_source = resolve_2dgs_normals(gp, prefer_fdn=True)
    if normal is None:
        Log(f"[2DGSReg] submap {submap_id}: no normal source, using O3D estimate")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.numpy().astype(np.float64))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.normalize_normals()
        normal = torch.from_numpy(np.asarray(pcd.normals).astype(np.float32))
        normal_source = "o3d_fallback"
    else:
        # 长度对齐
        n_xyz = len(xyz)
        n_norm = len(normal)
        if n_norm < n_xyz:
            Log(f"[2DGSReg] submap {submap_id}: normal shorter than xyz ({n_norm} vs {n_xyz}), padding")
            if rotation is not None:
                full = compute_normals_from_rotation(rotation)
                full[:n_norm] = normal
                normal = full
                normal_source = "fdn_ckpt+rotation_pad"
            else:
                pad = torch.zeros((n_xyz - n_norm, 3))
                pad[:, 2] = 1.0
                normal = torch.cat([normal, pad], dim=0)
        normal = F.normalize(normal.float(), dim=-1, eps=1e-8)

    Log(f"[2DGSReg] submap {submap_id}: N={len(xyz)}, normal_source={normal_source}, "
        f"norm_mean={normal.norm(dim=-1).mean():.4f}")

    return Submap2DGS(
        submap_id=submap_id, ckpt_path=ckpt_path,
        xyz=xyz, normal=normal, opacity=opacity, rotation=rotation,
        gaussian_params=gp, normal_source=normal_source,
        keyframe_ids=list(ckpt.get("submap_keyframes", [])),
        seed_pose=ckpt.get("seed_global_c2w"),
        correction=ckpt.get("correct_tsfm"),
    )


# ============================================================================
# Point cloud construction with ckpt normals preserved
# ============================================================================

def voxel_downsample_preserve_normal(src: Submap2DGS, voxel_size: float) -> o3d.geometry.PointCloud:
    """Voxel downsample 保留 ckpt 法线，不重新 estimate_normals。"""
    keep = torch.sigmoid(src.opacity) > 0.01
    xyz = src.xyz[keep].numpy().astype(np.float64)
    nrm = src.normal[keep].numpy().astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


# ============================================================================
# Overlap
# ============================================================================

def compute_submap_overlap(src: Submap2DGS, tgt: Submap2DGS, init_tsfm: np.ndarray,
                            radius: float = 0.05) -> float:
    src_pts = (init_tsfm[:3, :3] @ src.xyz.numpy().astype(np.float64).T).T + init_tsfm[:3, 3]
    tgt_pts = tgt.xyz.numpy().astype(np.float64)
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_pts)
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    c, step = 0, max(1, len(src_pts) // 2000)
    for i in range(0, len(src_pts), step):
        _, _, d2 = tgt_tree.search_knn_vector_3d(src_pts[i], 1)
        if d2[0] < radius * radius:
            c += 1
    return c / max(1, len(src_pts) // step)


# ============================================================================
# Normal-aware ICP
# ============================================================================

def _rot_error_deg(T_a, T_b):
    R = T_a[:3, :3] @ T_b[:3, :3].T
    t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(t))


def normal_aware_icp_2dgs(src_pcd: o3d.geometry.PointCloud,
                           tgt_pcd: o3d.geometry.PointCloud,
                           init_tsfm: np.ndarray, config: dict) -> Dict:
    """多尺度 point-to-plane ICP，使用 ckpt 法线。"""
    dist_th = config.get("icp_distance_th", 0.04)
    max_iters = config.get("icp_max_iters", 80)
    scales = [dist_th * 8, dist_th * 4, dist_th * 2]
    T = init_tsfm.copy()
    for s in scales:
        r = o3d.pipelines.registration.registration_icp(
            src_pcd, tgt_pcd, max_correspondence_distance=s, init=T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iters, relative_fitness=1e-6, relative_rmse=1e-6),
        )
        T = np.array(r.transformation, dtype=np.float64)

    # Quality: normal alignment
    src_pts = np.asarray(src_pcd.points)
    src_nrm = np.asarray(src_pcd.normals)
    tgt_pts = np.asarray(tgt_pcd.points)
    tgt_nrm = np.asarray(tgt_pcd.normals)
    R, t = T[:3, :3], T[:3, 3]
    src_t = (R @ src_pts.T).T + t
    src_nt = (R @ src_nrm.T).T

    tgt_kd = o3d.geometry.KDTreeFlann(tgt_pcd)
    normal_errors, dists, inliers = [], [], 0
    step = max(1, len(src_t) // 3000)
    nm_th = config.get("normal_th", 0.70)
    for i in range(0, len(src_t), step):
        _, idx, d2 = tgt_kd.search_knn_vector_3d(src_t[i], 1)
        d = np.sqrt(d2[0])
        dists.append(d)
        if d < dist_th:
            ad = abs(np.dot(src_nt[i], tgt_nrm[idx[0]]))
            if ad > nm_th:
                inliers += 1
                normal_errors.append(1.0 - ad)

    n_total = max(1, len(src_t) // step)
    fit = inliers / n_total
    rmse = float(np.sqrt(np.mean(np.array(dists) ** 2)))
    nm_err = float(np.mean(normal_errors)) if normal_errors else 1.0

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        src_pcd, tgt_pcd, dist_th, T)
    conf = float(np.clip(fit / max(rmse, 1e-3), 0.1, 10.0))
    info = np.array(info, dtype=np.float64) * conf

    delta_t = np.linalg.norm((T @ np.linalg.inv(init_tsfm))[:3, 3])
    delta_r = _rot_error_deg(T, init_tsfm)

    return {"successful": fit > 0.05,
            "transformation": T, "information": info,
            "fitness": fit, "inlier_rmse": rmse, "normal_score": 1.0 - nm_err,
            "overlap": fit, "delta_t": delta_t, "delta_r": delta_r}


# ============================================================================
# Main registration entry
# ============================================================================

def registration_2dgs(src: Submap2DGS, tgt: Submap2DGS, init_tsfm: np.ndarray,
                       config: dict = None, mode: str = "loop") -> Dict:
    if config is None:
        config = {}
    sid, tid = src.submap_id, tgt.submap_id
    Log(f"[2DGSReg] edge {sid} -> {tid} mode={mode} "
        f"N_src={len(src.xyz)} N_tgt={len(tgt.xyz)} "
        f"normal_src={src.normal_source} normal_tgt={tgt.normal_source}")

    init_is_I = np.allclose(init_tsfm, np.eye(4), atol=1e-4)
    Log(f"[2DGSReg] init_is_identity={init_is_I} init_t={np.linalg.norm(init_tsfm[:3,3]):.3f}m")

    # Overlap gate
    overlap = compute_submap_overlap(src, tgt, init_tsfm, radius=0.05)
    min_ov = config.get("min_overlap", 0.05)
    if overlap < min_ov:
        Log(f"[2DGSReg] {sid}->{tid} rejected: overlap={overlap:.3f} < {min_ov}")
        return _fail("low_overlap", overlap=overlap)

    # Build point clouds
    voxel = config.get("voxel_size", 0.03)
    src_pcd = voxel_downsample_preserve_normal(src, voxel)
    tgt_pcd = voxel_downsample_preserve_normal(tgt, voxel)
    Log(f"[2DGSReg] downsampled: src={len(src_pcd.points)} tgt={len(tgt_pcd.points)} overlap={overlap:.3f}")

    if len(src_pcd.points) < 50 or len(tgt_pcd.points) < 50:
        return _fail("too_few_points", overlap=overlap)

    # Normal-aware ICP
    icp_cfg = {"icp_distance_th": config.get("distance_th", 0.04),
               "icp_max_iters": config.get("max_iters", 80),
               "normal_th": config.get("normal_th", 0.70)}
    result = normal_aware_icp_2dgs(src_pcd, tgt_pcd, init_tsfm, icp_cfg)

    # Quality gates
    min_fit = config.get("min_fitness", 0.15)
    max_rmse = config.get("max_rmse", 0.08)
    max_dt = config.get("max_delta_t", 1.0)
    max_dr = config.get("max_delta_r_deg", 45.0)

    if result["fitness"] < min_fit:
        return _fail(f"fitness={result['fitness']:.3f} < {min_fit}", **result)
    if result["inlier_rmse"] > max_rmse:
        return _fail(f"rmse={result['inlier_rmse']:.3f} > {max_rmse}", **result)
    if result["delta_t"] > max_dt or result["delta_r"] > max_dr:
        return _fail(f"delta_t={result['delta_t']:.3f} delta_r={result['delta_r']:.1f}", **result)

    Log(f"[2DGSReg] {sid}->{tid} accepted: fitness={result['fitness']:.3f} "
        f"rmse={result['inlier_rmse']:.3f} nm_score={result['normal_score']:.3f}")
    result["successful"] = True
    result["overlap"] = overlap
    result["normal_source_src"] = src.normal_source
    result["normal_source_tgt"] = tgt.normal_source
    return result


def _fail(reason: str, **kw) -> Dict:
    return {"successful": False, "transformation": np.eye(4),
            "information": np.eye(6),
            "fitness": kw.get("fitness", 0), "inlier_rmse": kw.get("inlier_rmse", 99),
            "normal_score": kw.get("normal_score", 0), "overlap": kw.get("overlap", 0),
            "delta_t": 0, "delta_r": 0, "reason": reason}
