import json
import os
import time
import torch
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp
import roma
from utils.logging_utils import Log
from utils.gsr_2dgs.solver_2dgs import registration_2dgs_gsreg
from utils.reloc3r_adapter import Reloc3RSubmapRegistrator, estimate_keyframe_pair_pose
from utils.keyframe_pgo import (KeyframeRecord, Reloc3RPairEstimate, VerifiedLoopEdge,
                                 build_keyframe_database, build_keyframe_pose_graph,
                                 keyframe_db_stats, retrieve_keyframe_loop_candidates,
                                 KeyframeRetrievalCandidate, refine_keyframe_loop_edge,
                                 run_keyframe_pgo_trial, evaluate_keyframe_pgo_result)
from utils.loop_depth_verifier import verify_reloc3r_pair_with_rgbd
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
        self.submap_features = {}      # {submap_id: [N, D] tensor}
        self.submap_thresholds = {}    # {submap_id: [N] tensor}
        self.submap_keyframe_features = {}  # {submap_id: {kf_idx: (D,) np.ndarray}}
        self.min_similarity_ratio = 0.5

        # ===== 回环检测基本参数 =====
        self.min_interval = self.config.get("LoopClosure", {}).get("min_interval", 3)
        self.voxel_size = self.config.get("LoopClosure", {}).get("voxel_size", 0.05)
        self.icp_fitness_threshold = self.config.get("LoopClosure", {}).get("icp_fitness_threshold", 0.40)



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

        # ===== Reloc3R edge quality filters =====
        self.min_raw_vs_init_dot_for_pgo = self.config.get("LoopClosure", {}).get(
            "min_raw_vs_init_dot_for_pgo", 0.7
        )
        self.max_loop_delta_t_for_pgo_reloc3r = self.config.get("LoopClosure", {}).get(
            "max_loop_delta_t_for_pgo_reloc3r", 2.0
        )

        # PGO 增量门控：仅在新 loop edge 出现时运行
        self.last_loop_edge_count = 0

        # ===== Loop Closure Mode Control (Stage 0) =====
        self.mode = self.config.get("LoopClosure", {}).get("mode", "verify_only")
        self.legacy_submap_pgo_enabled = self.config.get("LoopClosure", {}).get(
            "legacy_submap_pgo_enabled", False
        )
        self.pgo_granularity = self.config.get("LoopClosure", {}).get(
            "pgo_granularity", "keyframe"
        )
        Log(
            f"[LoopClosure] mode={self.mode} "
            f"legacy_submap_pgo_enabled={self.legacy_submap_pgo_enabled} "
            f"pgo_granularity={self.pgo_granularity}"
        )

        # ===== Reloc3R 配置（Stage 1: 仅加载配置，不加载模型） =====
        reloc3r_cfg = self.config.get("LoopClosure", {}).get("Reloc3R", {})
        self.reloc3r_enabled = reloc3r_cfg.get("enabled", False)
        self.reloc3r_cfg = reloc3r_cfg
        # 子图元数据缓存（Reloc3R 需要 seed C2W + keyframe poses + 图像路径）
        self.submap_seed_c2w = {}          # {submap_id: 4x4 np.array}
        self.submap_keyframe_poses = {}    # {submap_id: {kf_idx: 4x4 np.array}}
        self.submap_image_paths = {}       # {submap_id: [str, ...]}
        self.submap_depth_paths = {}      # {submap_id: [str, ...]}
        self.keyframe_db = {}             # {kf_idx: KeyframeRecord}

        self.reloc3r_registrator = None
        if self.reloc3r_enabled:
            Log(f"[Reloc3R] config loaded (model NOT loaded in Stage 1) mode={reloc3r_cfg.get('scale_mode')}")
            self.reloc3r_registrator = Reloc3RSubmapRegistrator(reloc3r_cfg)
            Log(f"[Reloc3R] adapter initialized (mock={self.reloc3r_registrator.mock_mode})")

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

    def extract_submap_features_and_threshold(self, img_paths, kf_indices=None):
        """Extract CosPlace features from keyframe images.

        Returns:
            submap_desc: (N, D) tensor of N keyframe descriptors.
            dynamic_thresholds: (N,) tensor of per-keyframe thresholds.
            kf_features: dict {kf_idx: (D,) np.ndarray} for per-keyframe storage.
        """
        feats = []
        for img_path in img_paths:
            img_tensor = torch.load(img_path, map_location="cpu")
            img_input = self.img_transform(img_tensor).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(img_input).squeeze().detach().cpu()
            feats.append(feat)
            del img_input

        submap_desc = torch.stack(feats)  # [N, D]

        # Build per-keyframe descriptor dict
        kf_features = {}
        if kf_indices is not None and len(kf_indices) == len(feats):
            for kf_idx, feat in zip(kf_indices, feats):
                kf_features[int(kf_idx)] = feat.numpy().astype(np.float32)
        elif len(feats) > 0:
            # Fallback: assign descriptors by position
            for i, feat in enumerate(feats):
                kf_features[i] = feat.numpy().astype(np.float32)

        self_sim = torch.mm(submap_desc, submap_desc.T)

        k = max(int(len(submap_desc) * self.min_similarity_ratio), 1)
        score_min, _ = self_sim.topk(k, dim=1)
        dynamic_thresholds = score_min[:, -1]

        return submap_desc, dynamic_thresholds, kf_features

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
            return True
        except Exception as e:
            Log(f"[LoopClosure] 重新加载子图 {submap_id} 点云失败: {e}")
            return False

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

    def compute_relative_transform(self, source_id, target_id, current_pose_guesses,
                                    init_guess_override=None):
        try:
            pose_ref = init_guess_override if init_guess_override is not None else current_pose_guesses
            init_guess = (
                    np.linalg.inv(pose_ref[target_id]) @
                    pose_ref[source_id]
            )

            # 2DGS-GSReg registration (LoopSplat-style render-based)
            reg_method = self.config.get("LoopClosure", {}).get("registration_method", "2dgs_gsreg")
            src_ckpt, tgt_ckpt = self.submap_records.get(source_id), self.submap_records.get(target_id)

            # Reloc3R / reloc3r_mock branch (Stage 2)
            if reg_method == "reloc3r" or reg_method == "reloc3r_mock":
                if self.reloc3r_registrator is not None:
                    src_imgs = self.submap_image_paths.get(source_id, [])
                    tgt_imgs = self.submap_image_paths.get(target_id, [])
                    src_seed = self.submap_seed_c2w.get(source_id, np.eye(4))
                    tgt_seed = self.submap_seed_c2w.get(target_id, np.eye(4))
                    src_kf_poses = self.submap_keyframe_poses.get(source_id, {})
                    tgt_kf_poses = self.submap_keyframe_poses.get(target_id, {})

                    result = self.reloc3r_registrator.register_submaps(
                        source_id, target_id,
                        src_seed, tgt_seed,
                        src_kf_poses, tgt_kf_poses,
                        src_imgs, tgt_imgs,
                        init_guess,
                    )
                    if result["success"]:
                        metrics = result["metrics"]
                        Log(
                            f"[Reloc3R] 子图 {source_id}->{target_id} "
                            f"method={metrics.get('method')} scale={metrics.get('scale_value', 0):.3f}"
                        )
                        return result["T_tgt_src"], result["information"], True, metrics
                    r_metrics = result.get("metrics", {})
                    Log(f"[Reloc3R] 子图 {source_id}->{target_id} failed: {r_metrics.get('failure_reason', 'unknown')}")
                else:
                    Log(f"[Reloc3R] adapter not initialized, falling through to GSReg/ICP")

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
        # =====================================================================
        # DEPRECATED since Stage 0: Legacy submap-level PGO.
        # This path is disabled by default. Keyframe-level PGO via
        # build_keyframe_pose_graph + run_keyframe_pgo_trial is the
        # recommended path.
        # =====================================================================
        if not self.legacy_submap_pgo_enabled:
            Log("[LoopClosure] legacy_submap_pgo_enabled=false, skipping legacy submap PGO. "
                "Keyframe-level PGO is the recommended path. "
                "Set LoopClosure.legacy_submap_pgo_enabled=true to re-enable legacy (debug only).")
            return []

        Log("[LoopClosure-DEPRECATED] Running legacy submap-level PGO. "
            "This path is deprecated — consider using keyframe-level PGO instead.")

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

                    is_reloc3r = metrics.get("method", "").startswith("reloc3r")
                    if fitness < self.min_loop_fitness_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"fitness={fitness:.3f} < {self.min_loop_fitness_for_pgo}")
                    elif rmse > self.max_loop_rmse_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"rmse={rmse:.4f} > {self.max_loop_rmse_for_pgo}")
                    elif is_reloc3r and metrics.get("min_raw_vs_init_dot", 1.0) is not None \
                            and metrics.get("min_raw_vs_init_dot", 1.0) < self.min_raw_vs_init_dot_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"raw_vs_init_dot={metrics['min_raw_vs_init_dot']:.3f} < {self.min_raw_vs_init_dot_for_pgo}")
                    elif is_reloc3r and delta_t is not None \
                            and delta_t > self.max_loop_delta_t_for_pgo_reloc3r:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"delta_t={delta_t:.3f}m > {self.max_loop_delta_t_for_pgo_reloc3r} (reloc3r)")
                    elif not is_reloc3r and delta_t > self.max_loop_delta_t_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"delta_t={delta_t:.3f}m > {self.max_loop_delta_t_for_pgo}")
                    elif delta_r > self.max_loop_delta_r_for_pgo:
                        Log(f"[LoopClosure] 回环边 {query_id}<->{target_id} PGO过滤: "
                            f"delta_r={delta_r:.2f}deg > {self.max_loop_delta_r_for_pgo}")
                    else:
                        # Open3D convention: node_j = node_i @ T_ij
                        # Reloc3R outputs T_source_to_target = inv(C2W_tgt) @ C2W_src
                        # Open3D needs T_ij = inv(C2W_src) @ C2W_tgt = inv(T_source_to_target)
                        if is_reloc3r:
                            t_before = float(np.linalg.norm(trans[:3, 3]))
                            trans = np.linalg.inv(trans)
                            t_after = float(np.linalg.norm(trans[:3, 3]))
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
                        if is_reloc3r:
                            Log(f"[LoopClosure] 添加回环边: 子图 {query_id} <-> {target_id} "
                                f"(trans_original_t={t_before:.3f}m → inverted_t={t_after:.3f}m)")
                        else:
                            Log(f"[LoopClosure] 添加回环边: 子图 {query_id} <-> {target_id}")

        if loop_edges_added < 3:
            Log(f"[LoopClosure] 回环边不足 ({loop_edges_added} < 3)，跳过 full-graph PGO")
            return []

        if loop_edges_added <= self.last_loop_edge_count:
            Log(f"[LoopClosure] 无新增回环边 ({loop_edges_added} ≤ {self.last_loop_edge_count})，跳过重复 PGO")
            return []

        Log(f"[LoopClosure] 检测到有效闭环，启动全图 PGO ({len(all_submap_ids)} 个子图)...")
        for sid in all_submap_ids[:4]:
            t = float(np.linalg.norm(current_pose_guesses[sid][:3, 3]))
            Log(f"[LoopClosure] PGO前 submap {sid} pose: t={t:.3f}m")

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

        # PGO safety valve: reject if any correction exceeds max threshold
        safety_cfg = self.config.get("LoopClosure", {}).get("pgo_safety", {})
        if safety_cfg.get("enabled", False):
            max_correction_t = safety_cfg.get("max_correction_t", 1.0)
            for c in correction_list:
                dt = np.linalg.norm(c["correct_tsfm"][:3, 3])
                if dt > max_correction_t:
                    Log(f"[LoopClosure] PGO safety: submap {c['submap_id']} correction dt={dt:.3f}m > {max_correction_t}m, REJECTING PGO")
                    return []
            Log("[LoopClosure] PGO safety: all corrections within threshold, accepted")

        # PGO summary: per-submap corrections + aggregate stats
        corrections_t = [np.linalg.norm(c["correct_tsfm"][:3, 3]) for c in correction_list]
        corrections_r = [
            self._rotation_error_deg(c["correct_tsfm"], np.eye(4))
            for c in correction_list
        ]
        Log(f"[LoopClosure] PGO summary: {len(correction_list)} submaps, "
            f"{loop_edges_added} loop edges, "
            f"mean_correction_t={np.mean(corrections_t):.3f}m "
            f"max_correction_t={np.max(corrections_t):.3f}m "
            f"mean_correction_r={np.mean(corrections_r):.1f}deg "
            f"max_correction_r={np.max(corrections_r):.1f}deg")
        for i, sid in enumerate(all_submap_ids):
            opt_pose = np.array(pose_graph.nodes[i].pose, dtype=np.float64)
            opt_t = float(np.linalg.norm(opt_pose[:3, 3]))
            pre_t = float(np.linalg.norm(current_pose_guesses[sid][:3, 3]))
            Log(f"[LoopClosure] PGO后 submap {sid}: t={opt_t:.3f}m "
                f"(delta={opt_t - pre_t:+.3f}m, corr_t={corrections_t[i]:.3f}m)")

        self.last_loop_edge_count = loop_edges_added
        return correction_list

    # ========================================================================
    # 4.7 Apply Correction to Submaps
    # ========================================================================
    def apply_correction_to_submaps(self, correction_list):
        """DEPRECATED: Legacy submap-level PGO writeback.

        Writes correct_tsfm field only, does not modify gaussian_params or
        keyframe_poses within the ckpt. slam.py applies correct_tsfm at final
        fusion time to avoid double correction.

        This path is blocked by default (mode != keyframe_pgo and
        legacy_submap_pgo_enabled=false). The recommended path is keyframe-level
        PGO via apply_keyframe_pgo_to_trajectory + MapCorrection.
        """
        Log("[LoopClosure-DEPRECATED] apply_correction_to_submaps called. "
            "Legacy submap-level correct_tsfm writeback is deprecated. "
            "Use keyframe-level PGO via apply_keyframe_pgo_to_trajectory instead.")
        if self.mode != "keyframe_pgo":
            Log(f"[LoopClosure] mode={self.mode}, BLOCKING correct_tsfm write. "
                f"Set LoopClosure.mode=keyframe_pgo to enable PGO writeback.")
            return

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
        Log(f"Loop Closure 进程已启动 mode={self.mode}")
        if self.mode == "off":
            Log("[LoopClosure] mode=off, idle loop (no feature extraction, no PGO)")
            while True:
                if not self.loop_queue.empty():
                    data = self.loop_queue.get()
                    if data[0] == "stop":
                        Log("Loop Closure 进程退出.")
                        break
                    elif data[0] == "submap_saved":
                        Log(f"[LoopClosure] mode=off, submap {data[1]} received but ignored")
                else:
                    time.sleep(0.5)
            return

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
                    depth_paths = data[4] if len(data) > 4 else []

                    self.submap_records[submap_id] = ckpt_path
                    self.submap_image_paths[submap_id] = img_paths
                    self.submap_depth_paths[submap_id] = depth_paths

                    # 从 ckpt 读取 Reloc3R 所需的元数据（seed C2W + keyframe poses）
                    try:
                        ckpt = torch.load(ckpt_path, map_location="cpu")
                        self.submap_seed_c2w[submap_id] = np.array(
                            ckpt.get("seed_global_c2w", np.eye(4)), dtype=np.float64
                        )
                        kf_poses = ckpt.get("submap_keyframe_poses", {})
                        self.submap_keyframe_poses[submap_id] = {
                            int(k): np.array(v, dtype=np.float64) for k, v in kf_poses.items()
                        }
                        Log(f"[LoopClosure] 子图 {submap_id} 元数据: "
                            f"seed_c2w={'set' if not np.allclose(self.submap_seed_c2w[submap_id], np.eye(4)) else 'identity'}, "
                            f"kf_poses={len(self.submap_keyframe_poses[submap_id])}, "
                            f"images={len(img_paths)}")
                    except Exception as e:
                        Log(f"[LoopClosure] 读取子图 {submap_id} ckpt 元数据失败: {e}")

                    Log(f"[LoopClosure] 提取并缓存子图 {submap_id} 的 3D 点云与特征...")
                    dense_pcd, feature_pcd = self.extract_pcd_from_2dgs_ckpt(ckpt_path)
                    self.submap_dense_pcds[submap_id] = dense_pcd
                    self.submap_pcds[submap_id] = feature_pcd

                    # Get keyframe indices for per-keyframe descriptor assignment
                    kf_indices = sorted(self.submap_keyframe_poses.get(submap_id, {}).keys())
                    submap_desc, thresholds, kf_features = self.extract_submap_features_and_threshold(
                        img_paths, kf_indices=kf_indices
                    )
                    self.submap_features[submap_id] = submap_desc
                    self.submap_thresholds[submap_id] = thresholds
                    self.submap_keyframe_features[submap_id] = kf_features

                    Log(f"[LoopClosure] 接收并处理新子图: ID {submap_id} (包含 {len(img_paths)} 个关键帧)")

                    # Rebuild unified keyframe database from all loaded submaps
                    dataset_cfg = self.config.get("Dataset", {})
                    calib = dataset_cfg.get("Calibration", {})
                    intrinsics = {
                        "fx": calib.get("fx") or dataset_cfg.get("fx"),
                        "fy": calib.get("fy") or dataset_cfg.get("fy"),
                        "cx": calib.get("cx") or dataset_cfg.get("cx"),
                        "cy": calib.get("cy") or dataset_cfg.get("cy"),
                        "width": calib.get("width") or dataset_cfg.get("width"),
                        "height": calib.get("height") or dataset_cfg.get("height"),
                    }
                    self.keyframe_db = build_keyframe_database(
                        self.submap_keyframe_poses,
                        self.submap_image_paths,
                        self.submap_seed_c2w,
                        intrinsics,
                        self.submap_depth_paths,
                    )
                    stats = keyframe_db_stats(self.keyframe_db)
                    Log(f"[KeyframeDB] total={stats['total_keyframes']} "
                        f"submaps={stats['submap_count']} "
                        f"per_submap={stats['per_submap']} "
                        f"seeds={stats['seed_keyframes']} "
                        f"missing_rgb={stats['missing_rgb']} "
                        f"missing_depth={stats['missing_depth']} "
                        f"missing_descriptor={stats['missing_descriptor']}")

                    # ---- Keyframe-level retrieval (Stage 2) ----
                    if self.mode in ("detect_only", "verify_only", "keyframe_pgo"):
                        retrieval_log_path = os.path.join(
                            self.save_dir, "loop_keyframe_retrieval.jsonl"
                        )
                        # Query from the latest submap's keyframes (use tail kfs)
                        query_kfs = kf_indices[-min(3, len(kf_indices)):]
                        all_candidates = []
                        for qkf in query_kfs:
                            cands = retrieve_keyframe_loop_candidates(
                                qkf, self.keyframe_db,
                                submap_keyframe_features=self.submap_keyframe_features,
                                config=self.config.get("LoopClosure", {}),
                            )
                            all_candidates.extend(cands)

                        # Write JSONL log
                        try:
                            with open(retrieval_log_path, "a") as f:
                                for cand in all_candidates:
                                    record = {
                                        "query_keyframe_id": cand.query_keyframe_id,
                                        "target_keyframe_id": cand.target_keyframe_id,
                                        "query_submap_id": cand.query_submap_id,
                                        "target_submap_id": cand.target_submap_id,
                                        "cosplace_score": round(cand.cosplace_score, 4),
                                        "temporal_gap": cand.temporal_gap,
                                        "is_mutual": cand.is_mutual,
                                        "accepted_by_retrieval": cand.accepted_by_retrieval,
                                        "rejection_reason": cand.rejection_reason,
                                    }
                                    f.write(json.dumps(record) + "\n")
                        except Exception as e:
                            Log(f"[LoopClosure] JSONL write failed: {e}")

                        accepted = [c for c in all_candidates if c.accepted_by_retrieval]
                        mutual = [c for c in accepted if c.is_mutual]
                        Log(f"[KF Retrieval] {len(all_candidates)} candidates, "
                            f"{len(accepted)} accepted, {len(mutual)} mutual "
                            f"(query_kfs={query_kfs})")

                    # ---- Reloc3R keyframe-pair estimation (Stage 3) ----
                    if self.reloc3r_enabled and self.mode in ("verify_only", "keyframe_pgo"):
                        reloc3r_log_path = os.path.join(
                            self.save_dir, "reloc3r_keyframe_pairs.jsonl"
                        )
                        reloc3r_cfg = self.config.get("LoopClosure", {}).get("Reloc3R", {})
                        top_cands = [c for c in all_candidates if c.accepted_by_retrieval][:5]
                        reloc3r_accepted = 0
                        reloc3r_estimates = []
                        for cand in top_cands:
                            src_rec = self.keyframe_db.get(cand.query_keyframe_id)
                            tgt_rec = self.keyframe_db.get(cand.target_keyframe_id)
                            if src_rec is None or tgt_rec is None:
                                continue
                            try:
                                estimate = estimate_keyframe_pair_pose(src_rec, tgt_rec, reloc3r_cfg)
                            except Exception as e:
                                Log(f"[Reloc3R] pair ({cand.query_keyframe_id},{cand.target_keyframe_id}) crashed: {e}")
                                continue
                            reloc3r_estimates.append(estimate)
                            if estimate.accepted_by_reloc3r:
                                reloc3r_accepted += 1
                            # Write JSONL
                            try:
                                with open(reloc3r_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "source_keyframe_id": estimate.source_keyframe_id,
                                        "target_keyframe_id": estimate.target_keyframe_id,
                                        "source_submap_id": estimate.source_submap_id,
                                        "target_submap_id": estimate.target_submap_id,
                                        "raw_translation_norm": round(estimate.raw_translation_norm, 4),
                                        "raw_vs_init_dot": round(estimate.raw_vs_init_dot, 4),
                                        "scale_applied": round(estimate.scale_applied, 4),
                                        "accepted_by_reloc3r": estimate.accepted_by_reloc3r,
                                        "rejection_reason": estimate.rejection_reason,
                                    }) + "\n")
                            except Exception as e:
                                Log(f"[LoopClosure] Reloc3R JSONL write failed: {e}")
                        Log(f"[Reloc3R KF] {reloc3r_accepted}/{len(top_cands)} accepted")

                    # ---- Depth verification (Stage 4) ----
                    if self.mode in ("verify_only", "keyframe_pgo"):
                        depth_cfg = self.config.get("LoopClosure", {}).get("depth_verify", {})
                        depth_log_path = os.path.join(
                            self.save_dir, "loop_depth_verify.jsonl"
                        )
                        depth_accepted = 0
                        for est in reloc3r_estimates:
                            if not est.accepted_by_reloc3r:
                                continue
                            src_rec = self.keyframe_db.get(est.source_keyframe_id)
                            tgt_rec = self.keyframe_db.get(est.target_keyframe_id)
                            src_dpt = src_rec.depth_path if src_rec else None
                            tgt_dpt = tgt_rec.depth_path if tgt_rec else None
                            try:
                                verified = verify_reloc3r_pair_with_rgbd(
                                    est, src_dpt, tgt_dpt, intrinsics, depth_cfg,
                                )
                            except Exception as e:
                                Log(f"[Depth Verify] pair ({est.source_keyframe_id},{est.target_keyframe_id}) crashed: {e}")
                                continue
                            if verified.accepted_by_depth:
                                depth_accepted += 1
                            else:
                                Log(f"[Depth Verify Diag] ({est.source_keyframe_id},{est.target_keyframe_id}) "
                                    f"rejected: {verified.rejection_reason} "
                                    f"overlap={verified.depth_overlap:.3f} rmse={verified.depth_rmse:.4f} "
                                    f"inlier={verified.depth_inlier_ratio:.3f} valid_pts={verified.valid_projected_points}")
                            # Write JSONL
                            try:
                                with open(depth_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "source_kf": verified.source_keyframe_id,
                                        "target_kf": verified.target_keyframe_id,
                                        "scale_used": round(verified.scale_used, 4),
                                        "depth_overlap": round(verified.depth_overlap, 4),
                                        "depth_rmse": round(verified.depth_rmse, 4),
                                        "depth_inlier_ratio": round(verified.depth_inlier_ratio, 4),
                                        "valid_points": verified.valid_projected_points,
                                        "accepted_by_depth": verified.accepted_by_depth,
                                        "rejection_reason": verified.rejection_reason,
                                    }) + "\n")
                            except Exception as e:
                                Log(f"[LoopClosure] Depth verify JSONL write failed: {e}")
                        if depth_accepted < len(reloc3r_estimates):
                            for est in reloc3r_estimates:
                                if not est.accepted_by_reloc3r:
                                    continue
                                Log(f"[Depth Verify Diag] pair ({est.source_keyframe_id},{est.target_keyframe_id}) "
                                    f"raw_t={est.raw_translation_norm:.3f}m scale={est.scale_applied:.3f}")
                        Log(f"[Depth Verify] {depth_accepted}/{len(reloc3r_estimates)} accepted")

                    # ---- Refinement → VerifiedLoopEdge (Stage 5) ----
                    verified_loop_edges = []
                    if self.mode in ("verify_only", "keyframe_pgo"):
                        loop_cfg = self.config.get("LoopClosure", {})
                        refine_cfg = loop_cfg.get("render_refine", {})
                        refine_log_path = os.path.join(
                            self.save_dir, "loop_verified_edges.jsonl"
                        )
                        refine_accepted = 0
                        for est in reloc3r_estimates:
                            if not est.accepted_by_reloc3r:
                                continue
                            src_rec = self.keyframe_db.get(est.source_keyframe_id)
                            tgt_rec = self.keyframe_db.get(est.target_keyframe_id)
                            if src_rec is None or tgt_rec is None:
                                continue
                            try:
                                src_dpt = src_rec.depth_path if src_rec else None
                                tgt_dpt = tgt_rec.depth_path if tgt_rec else None
                                verified = verify_reloc3r_pair_with_rgbd(
                                    est, src_dpt, tgt_dpt, intrinsics, depth_cfg,
                                )
                                if not verified.accepted_by_depth:
                                    continue
                                src_ckpt = self.submap_records.get(est.source_submap_id)
                                tgt_ckpt = self.submap_records.get(est.target_submap_id)
                                edge = refine_keyframe_loop_edge(
                                    verified, src_rec, tgt_rec, loop_cfg,
                                    src_ckpt_path=src_ckpt, tgt_ckpt_path=tgt_ckpt,
                                )
                            except Exception as e:
                                Log(f"[Refine→Edge] pair ({est.source_keyframe_id},{est.target_keyframe_id}) crashed: {e}")
                                continue
                            verified_loop_edges.append(edge)
                            if edge.accepted_for_pgo:
                                refine_accepted += 1
                            # Write JSONL
                            try:
                                with open(refine_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "source_kf": edge.source_keyframe_id,
                                        "target_kf": edge.target_keyframe_id,
                                        "source_submap": edge.source_submap_id,
                                        "target_submap": edge.target_submap_id,
                                        "refinement_method": edge.verification_metrics.get("refinement_method", ""),
                                        "delta_t_from_odom": edge.verification_metrics.get("delta_t_from_odom", -1),
                                        "delta_r_from_odom": edge.verification_metrics.get("delta_r_deg_from_odom", -1),
                                        "accepted_for_pgo": edge.accepted_for_pgo,
                                        "rejection_reason": edge.rejection_reason,
                                    }) + "\n")
                            except Exception as e:
                                Log(f"[LoopClosure] Verified edge JSONL write failed: {e}")
                        Log(f"[Refine→Edge] {refine_accepted}/{len(reloc3r_estimates)} "
                            f"verified loop edges accepted for PGO")

                    # ---- Keyframe PGO trial (Stage 7) ----
                    if self.mode == "keyframe_pgo" and len(verified_loop_edges) > 0:
                        accepted_edges = [e for e in verified_loop_edges if e.accepted_for_pgo]
                        if len(accepted_edges) > 0:
                            Log(f"[KeyframePGO] {len(accepted_edges)} accepted edges, building graph...")
                            graph = build_keyframe_pose_graph(
                                self.keyframe_db, verified_loop_edges=accepted_edges,
                                config=self.config.get("LoopClosure", {}),
                            )
                            Log(f"[KeyframePGO] graph: {len(graph.nodes)} nodes, "
                                f"{graph.num_temporal_edges} temporal, "
                                f"{graph.num_handoff_edges} handoff, "
                                f"{graph.num_loop_edges} loop edges")

                            trial_result = run_keyframe_pgo_trial(
                                graph, config=self.config.get("LoopClosure", {}),
                            )
                            Log(f"[KeyframePGO] trial: max_corr_t={trial_result.max_correction_t:.3f}m "
                                f"max_corr_r={trial_result.max_correction_r_deg:.1f}deg "
                                f"odom_before={trial_result.odom_residual_before_mean:.4f} "
                                f"odom_after={trial_result.odom_residual_after_mean:.4f} "
                                f"loop_before={trial_result.loop_residual_before_mean:.4f} "
                                f"loop_after={trial_result.loop_residual_after_mean:.4f}")

                            trial_result = evaluate_keyframe_pgo_result(
                                trial_result, graph,
                                config=self.config.get("LoopClosure", {}),
                            )
                            if trial_result.accepted:
                                Log(f"[KeyframePGO] ACCEPTED! Saving results...")
                                pgo_result_path = os.path.join(
                                    self.save_dir, "keyframe_pgo_result.json"
                                )
                                try:
                                    with open(pgo_result_path, "w") as f:
                                        json.dump({
                                            "accepted": True,
                                            "num_nodes": len(graph.nodes),
                                            "num_loop_edges": graph.num_loop_edges,
                                            "max_correction_t": trial_result.max_correction_t,
                                            "max_correction_r_deg": trial_result.max_correction_r_deg,
                                            "odom_residual_before": trial_result.odom_residual_before_mean,
                                            "odom_residual_after": trial_result.odom_residual_after_mean,
                                            "loop_residual_before": trial_result.loop_residual_before_mean,
                                            "loop_residual_after": trial_result.loop_residual_after_mean,
                                            "keyframe_corrections": {
                                                str(k): v.tolist()
                                                for k, v in trial_result.keyframe_corrections.items()
                                            },
                                            "optimized_keyframe_c2w": {
                                                str(k): v.tolist()
                                                for k, v in trial_result.optimized_keyframe_c2w.items()
                                            },
                                        }, f, indent=2)
                                    Log(f"[KeyframePGO] result saved to {pgo_result_path}")
                                except Exception as e:
                                    Log(f"[KeyframePGO] save failed: {e}")
                            else:
                                Log(f"[KeyframePGO] REJECTED: {trial_result.rejection_reason}")
                        else:
                            Log(f"[KeyframePGO] no accepted edges, skipping trial")

                    # ---- Mode-aware PGO dispatch ----
                    if self.mode in ("detect_only", "verify_only", "keyframe_pgo"):
                        if self.debug_disable_pgo_for_fftvo_test:
                            Log("[LoopClosure] PGO disabled for FFTVO ablation test")
                            self.detect_closure(submap_id)
                        else:
                            correction_list = self.construct_and_optimize_pose_graph()
                            if len(correction_list) > 0:
                                self.apply_correction_to_submaps(correction_list)
                                Log("==> PGO 闭环校正及硬盘回写完毕！ <==")
                    # mode == "off" is handled above, before feature extraction

            else:
                time.sleep(0.5)
