import os
import time
import torch
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp
import roma
from utils.logging_utils import Log
from utils.gsr_2dgs.solver_2dgs import registration_2dgs_gsreg
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# ============================================================================
# 1. CosPlace Model Architecture
# ============================================================================

def _gem(x, p=torch.ones(1) * 3, eps: float = 1e-6):
    """Generalized Mean Pooling (GeM)"""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return _gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
}


class CosPlaceNetwork(nn.Module):
    """
    CosPlace 视觉位置识别网络
    结构：ResNet backbone (去掉 avgpool + fc) → L2Norm → GeM → Flatten → Linear → L2Norm
    """
    def __init__(self, backbone_name: str = "ResNet18", fc_output_dim: int = 512):
        super().__init__()
        assert backbone_name in CHANNELS_NUM_IN_LAST_CONV, \
            f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"

        backbone_fn = getattr(models, backbone_name.lower())
        backbone = backbone_fn(weights=None)
        layers = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*layers)

        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


# ============================================================================
# 2. CosPlace Weight Loading
# ============================================================================

COSPLACE_WEIGHT_URL = (
    "https://github.com/gmberton/CosPlace/releases/download/v1.0/"
    "{backbone}_{fc_output_dim}_cosplace.pth"
)


def load_cosplace_model(backbone: str = "ResNet18",
                        fc_output_dim: int = 512,
                        weight_path: str = None,
                        device: str = "cuda") -> nn.Module:
    """
    加载 CosPlace 模型，支持三种方式：
      1. 从本地 .pth 文件加载（优先）
      2. 从 GitHub Releases 直链自动下载并缓存到本地
      3. 通过 torch.hub 加载（备用）
    """
    model = CosPlaceNetwork(backbone, fc_output_dim)

    if weight_path and os.path.isfile(weight_path):
        Log(f"[LoopClosure] 从本地文件加载 CosPlace 权重: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        Log("[LoopClosure] CosPlace 本地权重加载完成。")
        return model.eval().to(device)

    url = COSPLACE_WEIGHT_URL.format(backbone=backbone, fc_output_dim=fc_output_dim)
    Log(f"[LoopClosure] 本地权重未找到，从 GitHub 下载: {url}")
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict)
        Log("[LoopClosure] CosPlace 权重下载并加载完成。")

        if weight_path:
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            torch.save(state_dict, weight_path)
            Log(f"[LoopClosure] 权重已缓存到: {weight_path}")

        return model.eval().to(device)
    except Exception as e:
        Log(f"[LoopClosure] GitHub 下载失败: {e}")

    Log("[LoopClosure] 尝试通过 torch.hub 加载 CosPlace...")
    try:
        model = torch.hub.load(
            'gmberton/cosplace',
            'get_trained_model',
            backbone=backbone,
            fc_output_dim=fc_output_dim,
            trust_repo=True
        )
        Log("[LoopClosure] torch.hub 加载 CosPlace 完成。")
        return model.eval().to(device)
    except Exception as e:
        raise RuntimeError(
            f"[LoopClosure] 所有加载方式均失败！请手动下载权重文件：\n"
            f"  下载地址: {url}\n"
            f"  保存到: {weight_path or 'weights/ResNet18_512_cosplace.pth'}\n"
            f"  错误信息: {e}"
        )


# ============================================================================
# 3. 2DGS Rigid Transform
# ============================================================================

def rigid_transform_2dgs(gaussian_params, tsfm_matrix):
    tsfm_matrix = torch.from_numpy(tsfm_matrix).float().cuda()
    R = tsfm_matrix[:3, :3]
    t = tsfm_matrix[:3, 3]

    xyz = gaussian_params['_xyz']
    gaussian_params['_xyz'] = (xyz @ R.T) + t

    if '_rotation' in gaussian_params:
        rotation_q = gaussian_params['_rotation']
        rotation_q_roma = rotation_q[:, [1, 2, 3, 0]]  # xyzw
        cur_rot_mat = roma.unitquat_to_rotmat(rotation_q_roma)

        new_rot_mat = torch.einsum('ij,njk->nik', R, cur_rot_mat)

        new_rotation_q_roma = roma.rotmat_to_unitquat(new_rot_mat).squeeze()
        new_rotation_q = new_rotation_q_roma[:, [3, 0, 1, 2]]  # wxyz
        gaussian_params['_rotation'] = new_rotation_q

        if '_normal' in gaussian_params:
            gaussian_params['_normal'] = new_rot_mat[:, :, 2]

    return gaussian_params


# ============================================================================
# 4. Loop Closure Process
# ============================================================================

class LoopClosureProcess(mp.Process):
    # ========================================================================
    # 4.1 Initialization
    # ========================================================================
    def __init__(self, config, loop_queue):
        super().__init__()
        self.config = config
        self.loop_queue = loop_queue
        self.save_dir = self.config["Results"]["save_dir"]
        self.submaps_dir = os.path.join(self.save_dir, "submaps")
        self.device = "cuda"

        # 点云内存缓存（LRU 策略管理）
        self.submap_pcds = {}       # 非相邻子图用 sparse 点云（feature cloud）
        self.submap_records = {}    # 子图 ID → ckpt 路径（永不清理）
        self.submap_dense_pcds = {}  # 相邻子图 refine 用 dense cloud

        # 视觉特征缓存（永不清理）
        self.submap_features = {}      # [N, D]
        self.submap_thresholds = {}    # [N]
        self.min_similarity_ratio = 0.5

        # ===== 回环检测基本参数 =====
        self.min_interval = self.config.get("LoopClosure", {}).get("min_interval", 3)
        self.voxel_size = self.config.get("LoopClosure", {}).get("voxel_size", 0.05)
        self.icp_fitness_threshold = self.config.get("LoopClosure", {}).get("icp_fitness_threshold", 0.40)

        # ===== LRU 缓存参数 =====
        self.max_cached_submaps = self.config.get("LoopClosure", {}).get("keep_recent_submaps", 3)
        self.submap_access_order = []

        # ===== CosPlace 模型配置 =====
        self.cosplace_backbone = self.config.get("LoopClosure", {}).get("backbone", "ResNet18")
        self.cosplace_dim = self.config.get("LoopClosure", {}).get("feature_dim", 512)
        self.cosplace_weight_path = self.config.get("LoopClosure", {}).get(
            "weight_path", f"weights/{self.cosplace_backbone}_{self.cosplace_dim}_cosplace.pth"
        )

        # ===== 回环 ICP 一致性阈值 =====
        self.max_loop_delta_translation = self.config.get("LoopClosure", {}).get("max_loop_delta_translation", 0.80)
        self.max_loop_delta_rotation_deg = self.config.get("LoopClosure", {}).get("max_loop_delta_rotation_deg", 45.0)

        # ===== PGO 参数 =====
        self.default_odom_info_scale = self.config.get("LoopClosure", {}).get("default_odom_info_scale", 120.0)

        # ===== PGO 保护参数 (EAGS 风格) =====
        self.debug_disable_pgo_for_fftvo_test = self.config.get("LoopClosure", {}).get(
            "debug_disable_pgo_for_fftvo_test", False
        )
        self.min_loop_fitness_for_pgo = self.config.get("LoopClosure", {}).get(
            "min_loop_fitness_for_pgo", 0.40
        )
        self.max_loop_rmse_for_pgo = self.config.get("LoopClosure", {}).get(
            "max_loop_rmse_for_pgo", 0.04
        )
        self.max_loop_delta_t_for_pgo = self.config.get("LoopClosure", {}).get(
            "max_loop_delta_t_for_pgo", 1.5
        )
        self.max_loop_delta_r_for_pgo = self.config.get("LoopClosure", {}).get(
            "max_loop_delta_r_for_pgo", 45.0
        )

    # ========================================================================
    # 4.2 Feature Extraction
    # ========================================================================
    def init_feature_extractor(self):
        Log(f"[LoopClosure] 初始化 CosPlace ({self.cosplace_backbone}, {self.cosplace_dim}D)...")
        self.feature_extractor = load_cosplace_model(
            backbone=self.cosplace_backbone,
            fc_output_dim=self.cosplace_dim,
            weight_path=self.cosplace_weight_path,
            device=self.device
        )

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_submap_features_and_threshold(self, img_paths):
        feats = []
        for img_path in img_paths:
            img_tensor = torch.load(img_path, map_location="cpu")
            img_input = self.img_transform(img_tensor).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(img_input).squeeze().detach().cpu()
            feats.append(feat)
            del img_input

        submap_desc = torch.stack(feats)  # [N, D]
        self_sim = torch.mm(submap_desc, submap_desc.T)

        k = max(int(len(submap_desc) * self.min_similarity_ratio), 1)
        score_min, _ = self_sim.topk(k, dim=1)
        dynamic_thresholds = score_min[:, -1]

        return submap_desc, dynamic_thresholds

    # ========================================================================
    # 4.3 Point Cloud Extraction & Cache
    # ========================================================================
    def extract_pcd_from_2dgs_ckpt(self, ckpt_path):
        """Extract O3D point cloud from 2DGS ckpt using FDN normals directly.
        No curvature filtering. No O3D normal estimation.
        Returns (dense_pcd, feature_pcd) — both use FDN normals from Gaussian _rotation.
        """
        submap_ckpt = torch.load(ckpt_path, map_location="cpu")
        gp = submap_ckpt["gaussian_params"]
        xyz = gp["_xyz"].numpy()

        # Use FDN normals from rotation quaternion: 2DGS surfel normal = local z-axis
        rot_q = gp["_rotation"]
        rot_mat = roma.unitquat_to_rotmat(rot_q).numpy()
        normals = rot_mat[:, :, 2]
        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_norm[normals_norm < 1e-6] = 1.0
        normals = normals / normals_norm

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # Voxel downsample → both dense and feature are the same (FDN normals)
        pcd_dense = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        if len(pcd_dense.points) == 0:
            Log(f"[LoopClosure] 子图点云为空: {ckpt_path}")
            del submap_ckpt, gp
            return pcd_dense, pcd_dense

        Log(
            f"[LoopClosure] 点云提取 (FDN normals): 原始 {len(pcd.points)} "
            f"→ dense {len(pcd_dense.points)} "
            f"(voxel={self.voxel_size:.3f})"
        )

        del submap_ckpt, gp
        return pcd_dense, pcd_dense

    def _ensure_pcd_loaded(self, submap_id):
        if submap_id in self.submap_pcds and submap_id in self.submap_dense_pcds:
            return True

        ckpt_path = self.submap_records.get(submap_id)
        if not ckpt_path or not os.path.exists(ckpt_path):
            Log(f"[LoopClosure] 子图 {submap_id} 的 ckpt 不存在，无法重新加载点云")
            return False

        Log(f"[LoopClosure] 子图 {submap_id} 点云已被清理，从磁盘重新加载: {ckpt_path}")
        try:
            dense_pcd, feature_pcd = self.extract_pcd_from_2dgs_ckpt(ckpt_path)
            self.submap_dense_pcds[submap_id] = dense_pcd
            self.submap_pcds[submap_id] = feature_pcd

            if submap_id in self.submap_access_order:
                self.submap_access_order.remove(submap_id)
            self.submap_access_order.append(submap_id)
            return True
        except Exception as e:
            Log(f"[LoopClosure] 重新加载子图 {submap_id} 点云失败: {e}")
            return False

    def cleanup_old_submaps(self):
        if len(self.submap_access_order) > self.max_cached_submaps:
            to_evict = self.submap_access_order[:-self.max_cached_submaps]

            for submap_id in to_evict:
                if submap_id in self.submap_pcds:
                    del self.submap_pcds[submap_id]
                if submap_id in self.submap_dense_pcds:
                    del self.submap_dense_pcds[submap_id]
                Log(f"[LoopClosure] LRU 清理子图 {submap_id} 的 dense/feature 点云缓存（视觉特征保留）")

            self.submap_access_order = self.submap_access_order[-self.max_cached_submaps:]
            torch.cuda.empty_cache()
            Log(
                f"[LoopClosure] 点云缓存清理完毕，当前缓存: "
                f"{len(self.submap_access_order)} 个子图 dense/feature 点云, "
                f"{len(self.submap_features)} 个子图特征"
            )

    # ========================================================================
    # 4.4 Loop Detection (Visual)
    # ========================================================================
    def detect_closure(self, query_id):
        matched_ids = []
        if query_id not in self.submap_features:
            return matched_ids

        query_desc = self.submap_features[query_id].to(self.device)
        query_thresh = self.submap_thresholds[query_id].to(self.device)

        for db_id, db_desc in self.submap_features.items():
            if db_id <= query_id - self.min_interval:
                db_desc_cuda = db_desc.to(self.device)
                cross_sim = torch.mm(query_desc, db_desc_cuda.T)
                matches = torch.argwhere(cross_sim > query_thresh.unsqueeze(1))

                if len(matches) > 0:
                    max_sim = cross_sim.max().item()
                    Log(f"[*] 视觉粗筛命中: 子图 {query_id} -> {db_id} (相似度: {max_sim:.3f})")
                    matched_ids.append(db_id)

                del db_desc_cuda

        del query_desc, query_thresh
        torch.cuda.empty_cache()
        return matched_ids

    # ========================================================================
    # 4.5 ICP & Relative Transform
    # ========================================================================
    def _rotation_error_deg(self, T_a, T_b):
        R = T_a[:3, :3] @ T_b[:3, :3].T
        trace_val = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        return np.degrees(np.arccos(trace_val))

    def compute_relative_transform(self, source_id, target_id, current_pose_guesses):
        try:
            init_guess = (
                    np.linalg.inv(current_pose_guesses[target_id]) @
                    current_pose_guesses[source_id]
            )

            # 2DGS-GSReg registration (LoopSplat-style render-based)
            reg_method = self.config.get("LoopClosure", {}).get("registration_method", "2dgs_gsreg")
            src_ckpt, tgt_ckpt = self.submap_records.get(source_id), self.submap_records.get(target_id)
            if (reg_method == "2dgs_gsreg" and src_ckpt and tgt_ckpt
                    and os.path.exists(src_ckpt) and os.path.exists(tgt_ckpt)):
                # Merge camera intrinsics into the LC config for viewpoint construction
                reg_config = dict(self.config.get("LoopClosure", {}))
                dataset_cfg = self.config.get("Dataset", {})
                reg_config["cam"] = {
                    "fx": dataset_cfg.get("fx", 525.0),
                    "fy": dataset_cfg.get("fy", 525.0),
                    "cx": dataset_cfg.get("cx", 319.5),
                    "cy": dataset_cfg.get("cy", 239.5),
                    "W": dataset_cfg.get("width", 640),
                    "H": dataset_cfg.get("height", 480),
                }
                reg = registration_2dgs_gsreg(
                    src_ckpt, tgt_ckpt, init_guess, reg_config)
                if reg.get("success"):
                    info = np.eye(6)
                    delta_t = reg.get("delta_t_from_init", 0)
                    delta_r = reg.get("delta_r_from_init", 0)
                    overlap = reg.get("overlap", 0)
                    Log(
                        f"[2DGS-GSReg] 子图 {source_id}->{target_id} 配准成功! "
                        f"overlap={overlap:.3f} delta_t={delta_t:.3f}m delta_r={delta_r:.1f}deg"
                    )
                    return reg["T_tgt_src"], info, True, {
                        "fitness": float(overlap),
                        "rmse": 0.0,
                        "delta_t": float(delta_t),
                        "delta_r": float(delta_r),
                    }
                Log(f"[2DGS-GSReg] 子图 {source_id}->{target_id} failed: {reg.get('reason','unknown')}")

            # Fallback: single-stage ICP using FDN normals from 2DGS
            if not self._ensure_pcd_loaded(source_id):
                Log(f"[LoopClosure] 无法加载子图 {source_id} 的点云，跳过 ICP")
                return np.identity(4), np.identity(6), False, {}
            if not self._ensure_pcd_loaded(target_id):
                Log(f"[LoopClosure] 无法加载子图 {target_id} 的点云，跳过 ICP")
                return np.identity(4), np.identity(6), False, {}

            source_pcd = self.submap_pcds[source_id]
            target_pcd = self.submap_pcds[target_id]

            icp_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size * 4.0,
                init=init_guess,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6
                )
            )

            if icp_result.fitness < self.icp_fitness_threshold or icp_result.inlier_rmse > 0.04:
                Log(f"[!] ICP 配准失败: 子图 {source_id}->{target_id} "
                    f"(Fitness: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.3f}m)")
                return np.identity(4), np.identity(6), False, {}

            transformation = np.array(icp_result.transformation, dtype=np.float64)

            delta = transformation @ np.linalg.inv(init_guess)
            delta_t = np.linalg.norm(delta[:3, 3])
            delta_r = self._rotation_error_deg(transformation, init_guess)

            if delta_t > self.max_loop_delta_translation or delta_r > self.max_loop_delta_rotation_deg:
                Log(
                    f"[!] LOOP 一致性校验失败: 子图 {source_id}->{target_id} | "
                    f"delta_t={delta_t:.3f}m, delta_r={delta_r:.2f}deg | 拒绝该闭环边"
                )
                return np.identity(4), np.identity(6), False, {}

            information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                source_pcd,
                target_pcd,
                self.voxel_size * 2.0,
                transformation
            )

            confidence_scale = float(np.clip(icp_result.fitness, 0.1, 1.0))
            information = information * confidence_scale

            Log(
                f"[ICP] 子图 {source_id}->{target_id} 配准成功! "
                f"(Fitness: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.3f}m, "
                f"delta_t={delta_t:.3f}m, delta_r={delta_r:.2f}deg)"
            )

            metrics = {
                "fitness": float(icp_result.fitness),
                "rmse": float(icp_result.inlier_rmse),
                "delta_t": float(delta_t),
                "delta_r": float(delta_r),
            }
            return transformation, information, True, metrics

        except Exception as e:
            Log(f"[LoopClosure] ICP 异常: {source_id}->{target_id}: {e}")
            return np.identity(4), np.identity(6), False, {}

    # ========================================================================
    # 4.6 Pose Graph Optimization — Global
    # ========================================================================
    # --- PGO Helper: load transitions from ckpt ---
    def _load_prev_to_curr_transition(self, prev_sid, curr_sid):
        prev_ckpt_path = self.submap_records.get(prev_sid)
        if prev_ckpt_path is not None and os.path.exists(prev_ckpt_path):
            prev_ckpt = torch.load(prev_ckpt_path, map_location="cpu")
            rel = prev_ckpt.get("relative_pose", np.eye(4, dtype=np.float64))
            if isinstance(rel, torch.Tensor):
                rel = rel.numpy()
            return np.array(rel, dtype=np.float64)

        return np.eye(4, dtype=np.float64)

    def _load_prev_to_curr_information(self, prev_sid, curr_sid):
        return np.identity(6, dtype=np.float64) * float(
            self.default_odom_info_scale
        )

    # --- PGO Helper: anchor chain ---
    def _build_open_loop_anchors(self, all_submap_ids):
        anchors = {}

        if len(all_submap_ids) == 0:
            return anchors

        anchors[all_submap_ids[0]] = np.eye(4, dtype=np.float64)

        for i in range(1, len(all_submap_ids)):
            prev_sid = all_submap_ids[i - 1]
            curr_sid = all_submap_ids[i]

            rel_prev_from_curr = self._load_prev_to_curr_transition(prev_sid, curr_sid)
            anchors[curr_sid] = anchors[prev_sid] @ rel_prev_from_curr

        return anchors

    def _load_correct_tsfm_from_ckpt(self, sid):
        ckpt_path = self.submap_records.get(sid)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            return np.eye(4)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        corr = ckpt.get("correct_tsfm", np.eye(4))
        if isinstance(corr, torch.Tensor):
            corr = corr.numpy()
        return np.array(corr, dtype=np.float64)

    def _build_current_pose_guesses(self, all_submap_ids):
        open_loop_anchors = self._build_open_loop_anchors(all_submap_ids)
        current_pose_guesses = {}

        for sid in all_submap_ids:
            corr = self._load_correct_tsfm_from_ckpt(sid)
            current_pose_guesses[sid] = corr @ open_loop_anchors[sid]

        return open_loop_anchors, current_pose_guesses

    # --- PGO: Full Graph ---
    def construct_and_optimize_pose_graph(self):
        all_submap_ids = sorted(self.submap_records.keys())
        if len(all_submap_ids) < 2:
            Log(f"[LoopClosure] 子图数量不足 ({len(all_submap_ids)})，跳过 PGO")
            return []

        open_loop_anchors, current_pose_guesses = self._build_current_pose_guesses(all_submap_ids)

        pose_graph = o3d.pipelines.registration.PoseGraph()
        id_mapping = {sid: i for i, sid in enumerate(all_submap_ids)}

        for sid in all_submap_ids:
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.array(current_pose_guesses[sid], dtype=np.float64)
                )
            )

        for i in range(1, len(all_submap_ids)):
            prev_sid = all_submap_ids[i - 1]
            curr_sid = all_submap_ids[i]

            rel_prev_from_curr = self._load_prev_to_curr_transition(prev_sid, curr_sid)

            # Reject degenerate odom edges
            odom_t = np.linalg.norm(rel_prev_from_curr[:3, 3])
            odom_r = self._rotation_error_deg(rel_prev_from_curr, np.eye(4))
            if odom_t < 0.001 and odom_r < 0.01:
                Log(f"[LoopClosure] 跳过退化 odom 边 {prev_sid}→{curr_sid} "
                    f"(near-identity, t={odom_t:.4f}m r={odom_r:.2f}deg)")
                continue
            if odom_t > 5.0 or odom_r > 120.0:
                Log(f"[LoopClosure] 跳过异常 odom 边 {prev_sid}→{curr_sid} "
                    f"(implausible, t={odom_t:.3f}m r={odom_r:.1f}deg)")
                continue

            # Open3D convention: node_j = node_i @ T_ij
            # T_odom = inv(seed_prev) @ seed_curr = rel_prev_from_curr (already source→target)
            odom_source_to_target = rel_prev_from_curr.copy()
            info_odom = self._load_prev_to_curr_information(prev_sid, curr_sid)

            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i - 1,
                    i,
                    odom_source_to_target,
                    info_odom,
                    uncertain=False,
                )
            )

        recent_query_submaps = self.config.get("LoopClosure", {}).get("recent_query_submaps", 2)
        query_ids = all_submap_ids[-min(recent_query_submaps, len(all_submap_ids)):]
        loop_edges_added = 0

        for query_id in query_ids:
            matched_ids = self.detect_closure(query_id)

            for target_id in matched_ids:
                if target_id not in id_mapping:
                    continue
                if abs(query_id - target_id) < self.min_interval:
                    continue

                trans, info_loop, success, metrics = self.compute_relative_transform(
                    query_id, target_id, current_pose_guesses
                )

                if success:
                    fitness = metrics.get("fitness", 0)
                    rmse = metrics.get("rmse", 99)
                    delta_t = metrics.get("delta_t", 99)
                    delta_r = metrics.get("delta_r", 999)

                    if fitness < self.min_loop_fitness_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"fitness={fitness:.3f} < {self.min_loop_fitness_for_pgo}")
                    elif rmse > self.max_loop_rmse_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"rmse={rmse:.4f} > {self.max_loop_rmse_for_pgo}")
                    elif delta_t > self.max_loop_delta_t_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"delta_t={delta_t:.3f}m > {self.max_loop_delta_t_for_pgo}")
                    elif delta_r > self.max_loop_delta_r_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"delta_r={delta_r:.2f}deg > {self.max_loop_delta_r_for_pgo}")
                    else:
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                id_mapping[query_id],
                                id_mapping[target_id],
                                trans,
                                info_loop,
                                uncertain=True,
                            )
                        )
                        loop_edges_added += 1
                        Log(f"[LoopClosure] 添加回环边: 子图 {query_id} <-> {target_id}")

        if loop_edges_added < 3:
            Log(f"[LoopClosure] 回环边不足 ({loop_edges_added} < 3)，跳过 full-graph PGO")
            return []

        Log(f"[LoopClosure] 检测到有效闭环，启动全图 PGO ({len(all_submap_ids)} 个子图)...")

        prune_threshold = self.config.get("LoopClosure", {}).get("pgo_edge_prune_thres", 0.25)
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 1.5,
            edge_prune_threshold=prune_threshold,
            reference_node=0,
        )

        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        correction_list = []
        for i, sid in enumerate(all_submap_ids):
            optimized_pose = np.array(pose_graph.nodes[i].pose, dtype=np.float64)
            anchor_pose = open_loop_anchors[sid]
            correction = optimized_pose @ np.linalg.inv(anchor_pose)
            correction_list.append({
                "submap_id": sid,
                "correct_tsfm": correction,
            })

        return correction_list

    # ========================================================================
    # 4.7 Apply Correction to Submaps
    # ========================================================================
    def apply_correction_to_submaps(self, correction_list):
        """只写入 correct_tsfm 字段，不修改 ckpt 内的 gaussian_params 和 keyframe_poses。
        slam.py 终局合并时一次性应用 correct_tsfm，避免双重修正。
        """
        for correction in correction_list:
            submap_id = correction["submap_id"]
            new_correct_tsfm = np.array(correction["correct_tsfm"], dtype=np.float64)

            ckpt_path = self.submap_records.get(submap_id)
            if not ckpt_path or not os.path.exists(ckpt_path):
                continue

            submap_ckpt = torch.load(ckpt_path, map_location="cpu")
            submap_ckpt["correct_tsfm"] = new_correct_tsfm
            torch.save(submap_ckpt, ckpt_path)

            Log(
                f"[LoopClosure] PGO correction written to submap {submap_id}: "
                f"delta_t={np.linalg.norm(new_correct_tsfm[:3, 3] - np.eye(4)[:3, 3]):.4f}m"
            )

    # ========================================================================
    # 4.9 Main Loop
    # ========================================================================
    def run(self):
        Log("Loop Closure 进程已启动，后台静默监听中...")
        self.init_feature_extractor()
        while True:
            if not self.loop_queue.empty():
                data = self.loop_queue.get()
                if data[0] == "stop":
                    Log("Loop Closure 进程退出.")
                    break
                elif data[0] == "submap_saved":
                    submap_id = data[1]
                    ckpt_path = data[2]
                    img_paths = data[3]

                    self.submap_records[submap_id] = ckpt_path

                    Log(f"[LoopClosure] 提取并缓存子图 {submap_id} 的 3D 点云与特征...")
                    dense_pcd, feature_pcd = self.extract_pcd_from_2dgs_ckpt(ckpt_path)
                    self.submap_dense_pcds[submap_id] = dense_pcd
                    self.submap_pcds[submap_id] = feature_pcd

                    submap_desc, thresholds = self.extract_submap_features_and_threshold(img_paths)
                    self.submap_features[submap_id] = submap_desc
                    self.submap_thresholds[submap_id] = thresholds

                    Log(f"[LoopClosure] 接收并处理新子图: ID {submap_id} (包含 {len(img_paths)} 个关键帧)")

                    if submap_id in self.submap_access_order:
                        self.submap_access_order.remove(submap_id)
                    self.submap_access_order.append(submap_id)

                    if self.debug_disable_pgo_for_fftvo_test:
                        Log("[LoopClosure] PGO disabled for FFTVO ablation test")
                        # 仍运行回环检测（仅日志输出）
                        self.detect_closure(submap_id)
                    else:
                        # 2) 尝试闭环 PGO
                        correction_list = self.construct_and_optimize_pose_graph()
                        if len(correction_list) > 0:
                            self.apply_correction_to_submaps(correction_list)
                            Log("==> PGO 闭环校正及硬盘回写完毕！ <==")

                    # 清理旧子图缓存
                    self.cleanup_old_submaps()
            else:
                time.sleep(0.5)
