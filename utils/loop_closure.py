import os
import time
import torch
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp
import roma  # 用于四元数和旋转矩阵的转换
from utils.logging_utils import Log
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# ============================================================================
# CosPlace 模型结构定义（内置，无需克隆 CosPlace 仓库）
# 来源：https://github.com/gmberton/CosPlace/tree/main/cosplace_model
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


# ResNet18 最后一个卷积层的输出通道数
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

        # 构建 backbone（去掉最后的 avgpool 和 fc 层）
        backbone_fn = getattr(models, backbone_name.lower())
        backbone = backbone_fn(weights=None)  # 不加载 ImageNet 权重，后面会加载 CosPlace 权重
        layers = list(backbone.children())[:-2]  # 去掉 avgpool 和 fc
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
# 权重下载 URL 模板
# 来源：https://github.com/gmberton/CosPlace/releases/tag/v1.0
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

    # ========== 方式 1：从本地文件加载 ==========
    if weight_path and os.path.isfile(weight_path):
        Log(f"[LoopClosure] 从本地文件加载 CosPlace 权重: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        Log("[LoopClosure] CosPlace 本地权重加载完成。")
        return model.eval().to(device)

    # ========== 方式 2：从 GitHub Releases 直链下载 ==========
    url = COSPLACE_WEIGHT_URL.format(backbone=backbone, fc_output_dim=fc_output_dim)
    Log(f"[LoopClosure] 本地权重未找到，从 GitHub 下载: {url}")
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict)
        Log("[LoopClosure] CosPlace 权重下载并加载完成。")

        # 自动保存到项目 weights 目录，方便下次直接使用
        if weight_path:
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            torch.save(state_dict, weight_path)
            Log(f"[LoopClosure] 权重已缓存到: {weight_path}")

        return model.eval().to(device)
    except Exception as e:
        Log(f"[LoopClosure] GitHub 下载失败: {e}")

    # ========== 方式 3：通过 torch.hub 加载（备用） ==========
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
# 子图高斯变换工具函数
# ============================================================================

def rigid_transform_2dgs(gaussian_params, tsfm_matrix):
    tsfm_matrix = torch.from_numpy(tsfm_matrix).float().cuda()
    R = tsfm_matrix[:3, :3]
    t = tsfm_matrix[:3, 3]

    # 1. 变换中心点 (xyz)
    xyz = gaussian_params['_xyz']
    gaussian_params['_xyz'] = (xyz @ R.T) + t  # 【修复】：正确的点云旋转公式 (N,3) @ (3,3) + (3,)

    # 2. 变换二维高斯的朝向
    if '_rotation' in gaussian_params:
        rotation_q = gaussian_params['_rotation']
        rotation_q_roma = rotation_q[:, [1, 2, 3, 0]]  # xyzw
        cur_rot_mat = roma.unitquat_to_rotmat(rotation_q_roma)

        # 【修复】：批量矩阵乘法，R 是 (3,3)，cur_rot_mat 是 (N,3,3)
        new_rot_mat = torch.einsum('ij,njk->nik', R, cur_rot_mat)

        new_rotation_q_roma = roma.rotmat_to_unitquat(new_rot_mat).squeeze()
        new_rotation_q = new_rotation_q_roma[:, [3, 0, 1, 2]]  # wxyz
        gaussian_params['_rotation'] = new_rotation_q

        if '_normal' in gaussian_params:
            # 法线是旋转矩阵的 Z 轴（第三列）
            gaussian_params['_normal'] = new_rot_mat[:, :, 2]

    return gaussian_params


# ============================================================================
# 回环检测主进程
# ============================================================================

class LoopClosureProcess(mp.Process):
    def __init__(self, config, loop_queue):
        super().__init__()
        self.config = config
        self.loop_queue = loop_queue
        self.save_dir = self.config["Results"]["save_dir"]
        self.submaps_dir = os.path.join(self.save_dir, "submaps")
        self.device = "cuda"

        # ==============================================================
        # 点云内存缓存（LRU 策略管理，按需从磁盘重新加载）
        #稀疏特征点云（用于非相邻ICP / loopclosure）
        #稠密点云（用于相邻 refine）
        # ==============================================================
        self.submap_pcds = {}       # 非相邻子图用sparse点云，内存缓存每个子图的点云数据（LRU 管理，可被清理后重新加载）
        self.submap_records = {}    # 记录子图 ID 与其对应的 ckpt 路径（永不清理）
        # 相邻子图refine要用dense点云
        self.submap_dense_pcds = {}  # 专门给相邻 refine 用  # raw dense downsampled cloud
        # ==============================================================
        # 视觉特征缓存（永不清理，内存占用极小）
        # CosPlace 特征向量每个子图约 N_kf * 512 * 4 bytes ≈ 几十 KB
        # ==============================================================
        self.submap_features = {}      # 保存每个子图的多帧特征矩阵 [N, D]
        self.submap_thresholds = {}    # 保存每个子图的内部动态阈值 [N]
        self.min_similarity_ratio = 0.5  # 自相似度 Top-K 比例

        self.min_interval = self.config.get("LoopClosure", {}).get("min_interval", 3)
        self.voxel_size = self.config.get("LoopClosure", {}).get("voxel_size", 0.05)
        self.icp_fitness_threshold = self.config.get("LoopClosure", {}).get("icp_fitness_threshold", 0.40)

        # LRU 缓存参数
        self.max_cached_submaps = self.config.get("LoopClosure", {}).get("keep_recent_submaps", 3)
        self.submap_access_order = []

        # CosPlace 模型配置
        self.cosplace_backbone = self.config.get("LoopClosure", {}).get("backbone", "ResNet18")
        self.cosplace_dim = self.config.get("LoopClosure", {}).get("feature_dim", 512)
        self.cosplace_weight_path = self.config.get("LoopClosure", {}).get(
            "weight_path", f"weights/{self.cosplace_backbone}_{self.cosplace_dim}_cosplace.pth"
        )

        # ===== 相邻子图边精炼参数 =====相邻子图icp用dense点云，且门槛更严格，避免误匹配引入错误的约束
        self.adjacent_icp_fitness_threshold = self.config.get("LoopClosure", {}).get("adjacent_icp_fitness_threshold", 0.45)
        self.adjacent_icp_rmse_threshold = self.config.get("LoopClosure", {}).get("adjacent_icp_rmse_threshold", 0.03)
        self.max_adjacent_delta_translation = self.config.get("LoopClosure", {}).get("max_adjacent_delta_translation", 0.25)
        self.max_adjacent_delta_rotation_deg = self.config.get("LoopClosure", {}).get("max_adjacent_delta_rotation_deg", 12.0)
        self.default_odom_info_scale = self.config.get("LoopClosure", {}).get("default_odom_info_scale", 120.0)
        # ========== Global-seed submap mode ==========
        # 如果前端切图时不再把 seed 帧重置为单位阵，而是继承 tracking 后的全局 c2w，
        # 那么每个子图 ckpt 中的 2DGS 点云已经处在同一个全局坐标系。
        # 此时 loop_closure 不能再按“局部子图 + relative_pose 串 anchor”的旧逻辑做 ChainPGO。
        self.use_global_seed_submap = self.config.get("Submap", {}).get("use_global_seed_submap", False)
        # global seed 模式下，没有真实非相邻闭环时，默认禁止 ChainPGO。
        default_enable_chain_pgo = False if self.use_global_seed_submap else True
        self.enable_chain_pgo_without_loop = self.config.get("LoopClosure", {}).get("enable_chain_pgo_without_loop", default_enable_chain_pgo)
        # global seed 模式下，没有 refined adjacent edge 时，只给极弱 odom 稳定项。
        self.global_seed_default_odom_info_scale = self.config.get("LoopClosure", {}).get("global_seed_default_odom_info_scale", 1e-3)

        # 新增：相邻边降权参数
        self.adjacent_edge_weight = self.config.get("LoopClosure", {}).get("adjacent_edge_weight", 0.25)
        self.adjacent_info_min_scale = self.config.get("LoopClosure", {}).get("adjacent_info_min_scale", 0.5)
        self.adjacent_info_max_scale = self.config.get("LoopClosure", {}).get("adjacent_info_max_scale", 8.0)

        self.chain_pgo_window = self.config.get("LoopClosure", {}).get("chain_pgo_window", 4)

        Log(
            f"[LoopClosure] use_global_seed_submap={self.use_global_seed_submap}, "
            f"enable_chain_pgo_without_loop={self.enable_chain_pgo_without_loop}"
        )

        # 回环边用feature cloud 新增：feature cloud 保留比例（建议保留 top 30% 曲率点）
        self.feature_keep_ratio = self.config.get("LoopClosure", {}).get("feature_keep_ratio", 0.30)

    def init_feature_extractor(self):
        """
        初始化 CosPlace 特征提取器。
        模型结构已内置在本文件中，无需克隆 CosPlace 仓库。
        权重加载优先级：本地文件 > GitHub 直链下载 > torch.hub
        """
        Log(f"[LoopClosure] 初始化 CosPlace ({self.cosplace_backbone}, {self.cosplace_dim}D)...")
        self.feature_extractor = load_cosplace_model(
            backbone=self.cosplace_backbone,
            fc_output_dim=self.cosplace_dim,
            weight_path=self.cosplace_weight_path,
            device=self.device
        )

        # 图像预处理：CosPlace 使用 ImageNet 标准化
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_submap_features_and_threshold(self, img_paths):
        """
        批量为一个子图的所有关键帧提取 CosPlace 视觉特征，并为每帧计算一个动态相似度阈值，用于后续视觉粗筛（回环候选过滤）
        所有中间计算在 CPU 上完成，仅推理时短暂使用 GPU。自相似矩阵就是一个描述同一子图内各帧特征两两相似度的矩阵这里用来自适应地为每一帧计算动态匹配阈值：对每行取 Top‑K 相似度，把第 K 大的值作为该帧的阈值。这样匹配时只保留与子图内部结构相符的较高相似度，能增强视觉粗筛的鲁棒性（抑制孤立或噪声帧带来的误匹配）。
        """
        feats = []
        for img_path in img_paths:
            img_tensor = torch.load(img_path, map_location="cpu")  # [3, H, W]
            img_input = self.img_transform(img_tensor).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(img_input).squeeze().detach().cpu() #前向得到每帧的特征向量，detach 并搬回 CPU，收集到 feats
            feats.append(feat)
            del img_input

        submap_desc = torch.stack(feats)  # [N, D]，已在 CPU 上，N：关键帧数，D：特征维度
        self_sim = torch.mm(submap_desc, submap_desc.T) #自相似矩阵

        k = max(int(len(submap_desc) * self.min_similarity_ratio), 1) #
        score_min, _ = self_sim.topk(k, dim=1) #取每行的 Top-K 相似度
        dynamic_thresholds = score_min[:, -1] #并把第 K 大的值作为该帧的动态阈值
        #通过每帧在子图内部的自相似性动态决定匹配阈值，提升视觉粗筛鲁棒性（根据子图内部相似结构自适应）。
        return submap_desc, dynamic_thresholds

    def extract_pcd_from_2dgs_ckpt(self, ckpt_path):
        """
        从 2DGS checkpoint 提取两类点云：
        1) dense cloud   : 去离群 + voxel 下采样后的点云（给相邻 refine 用）
        2) feature cloud : 在 dense cloud 基础上做曲率筛选后的点云（给非相邻 loop ICP 用）
        dense 分支：保留 voxel 平均后的 2DGS normals，不再 estimate_normals
        feature 分支：在 dense 副本上 estimate_normals，再做曲率筛选
        """
        submap_ckpt = torch.load(ckpt_path, map_location="cpu")
        gp = submap_ckpt["gaussian_params"]

        xyz = gp["_xyz"].numpy()
        rot_q = gp["_rotation"]

        # 2DGS 法线：由高斯旋转矩阵第 3 列近似得到
        rot_mat = roma.unitquat_to_rotmat(rot_q).numpy()
        normals = rot_mat[:, :, 2]

        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_norm[normals_norm < 1e-6] = 1.0
        normals = normals / normals_norm

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # 1) 去离群
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # 2) voxel 下采样：这一步的结果直接作为 dense cloud
        pcd_dense = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Open3D 的 voxel_down_sample 会对已有 normals 做平均
        # 所以 dense cloud 已经有 normals，可直接用于 point-to-plane ICP
        if len(pcd_dense.points) == 0:
            Log(f"[LoopClosure] 子图点云为空: {ckpt_path}")
            del submap_ckpt, gp
            return pcd_dense, pcd_dense

        # 3) feature cloud 在 dense 的副本上做，避免污染 dense normals
        pcd_for_feature = o3d.geometry.PointCloud(pcd_dense)

        # 这里建议保留重新估计法线，但只作用在 feature 分支
        pcd_for_feature.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        pcd_for_feature.normalize_normals()

        points = np.asarray(pcd_for_feature.points)
        normals_ds = np.asarray(pcd_for_feature.normals)

        if len(points) < 10:
            Log(f"[LoopClosure] 点数过少，dense 直接兼作 feature: {len(points)}")
            del submap_ckpt, gp
            return pcd_dense, pcd_for_feature

        tree = o3d.geometry.KDTreeFlann(pcd_for_feature)
        curvatures = []
        for i in range(len(points)):
            k, idx, _ = tree.search_knn_vector_3d(points[i], 10)
            neighbor_normals = normals_ds[idx]
            curvature = np.std(neighbor_normals, axis=0).mean()
            curvatures.append(curvature)

        curvatures = np.asarray(curvatures)

        # keep_ratio=0.30 表示保留 top 30% 曲率点
        keep_ratio = float(self.feature_keep_ratio)
        keep_ratio = min(max(keep_ratio, 0.05), 0.95)
        threshold = np.percentile(curvatures, 100.0 * (1.0 - keep_ratio))
        feature_mask = curvatures >= threshold
        feature_indices = np.where(feature_mask)[0]

        feature_points = points[feature_indices]
        feature_normals = normals_ds[feature_indices]

        pcd_feature = o3d.geometry.PointCloud()
        pcd_feature.points = o3d.utility.Vector3dVector(feature_points)
        pcd_feature.normals = o3d.utility.Vector3dVector(feature_normals)

        Log(
            f"[LoopClosure] 点云提取：原始 {len(pcd.points)} "
            f"→ dense {len(pcd_dense.points)} "
            f"→ feature {len(pcd_feature.points)}"
        )

        del submap_ckpt, gp
        return pcd_dense, pcd_feature

    def _ensure_pcd_loaded(self, submap_id):
        """
        确保指定子图的 dense / feature 两类点云都已加载。
        """
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

    def detect_closure(self, query_id):
        """纯视觉粗筛，使用 CosPlace 特征进行子图间相似度匹配"""
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


    def construct_and_optimize_pose_graph(self):
        all_submap_ids = sorted(self.submap_records.keys())
        if len(all_submap_ids) < 2:
            Log(f"[LoopClosure] 子图数量不足 ({len(all_submap_ids)})，跳过 PGO")
            return []

        open_loop_anchors, current_pose_guesses = self._build_current_pose_guesses(all_submap_ids)

        pose_graph = o3d.pipelines.registration.PoseGraph()
        id_mapping = {sid: i for i, sid in enumerate(all_submap_ids)}

        # 1) 全部子图建节点
        for sid in all_submap_ids:
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.array(current_pose_guesses[sid], dtype=np.float64)
                )
            )

        # 2) 全链路相邻 odom 边：使用 refined pose + refined info
        for i in range(1, len(all_submap_ids)):
            prev_sid = all_submap_ids[i - 1]
            curr_sid = all_submap_ids[i]

            rel_prev_from_curr = self._load_prev_to_curr_transition(prev_sid, curr_sid)
            odom_source_to_target = np.linalg.inv(rel_prev_from_curr)
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

        # 3) 真正闭环：只让最近几个新子图作为 query，但 target 可以是所有历史子图
        recent_query_submaps = self.config.get("LoopClosure", {}).get("recent_query_submaps", 2)
        query_ids = all_submap_ids[-min(recent_query_submaps, len(all_submap_ids)):]
        loop_found = False

        for query_id in query_ids:
            matched_ids = self.detect_closure(query_id)

            for target_id in matched_ids:
                if target_id not in id_mapping:
                    continue
                if abs(query_id - target_id) < self.min_interval:
                    continue

                trans, info_loop, success = self.compute_relative_transform(
                    query_id, target_id, current_pose_guesses
                )

                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            id_mapping[query_id],
                            id_mapping[target_id],
                            trans,
                            info_loop,
                            uncertain=True,
                        )
                    )
                    loop_found = True
                    Log(f"[LoopClosure] 添加回环边: 子图 {query_id} <-> {target_id}")

        if not loop_found:
            Log("[LoopClosure] 当前无有效非相邻闭环，跳过 full-graph PGO")
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

    def _load_relative_pose_from_ckpt(self, sid):
        ckpt_path = self.submap_records.get(sid)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            return np.eye(4)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        rel = ckpt.get("relative_pose", np.eye(4))
        if isinstance(rel, torch.Tensor):
            rel = rel.numpy()
        return np.array(rel, dtype=np.float64)

    def _build_open_loop_anchors(self, all_submap_ids):
        """
        anchors[sid]: 子图 sid 的 open-loop anchor。

        旧模式：
            子图内部是局部坐标，需要用 relative_pose 串 anchor。

        global seed 模式：
            子图内部的高斯点和关键帧位姿已经是全局坐标，
            所以每个子图的 open-loop anchor 都应该是 I。
        """
        anchors = {}

        if len(all_submap_ids) == 0:
            return anchors

        if self.use_global_seed_submap:
            for sid in all_submap_ids:
                anchors[sid] = np.eye(4, dtype=np.float64)
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
        """
        current_pose[sid]: 当前对子图 sid 的全局位姿估计（local -> global）
        = correct_tsfm @ open_loop_anchor
        """
        open_loop_anchors = self._build_open_loop_anchors(all_submap_ids)
        current_pose_guesses = {}

        for sid in all_submap_ids:
            corr = self._load_correct_tsfm_from_ckpt(sid)
            current_pose_guesses[sid] = corr @ open_loop_anchors[sid]

        return open_loop_anchors, current_pose_guesses

    def _rotation_error_deg(self, T_a, T_b):
        R = T_a[:3, :3] @ T_b[:3, :3].T
        trace_val = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        return np.degrees(np.arccos(trace_val))

    def _clear_refined_adjacent_edge(self, curr_sid):
        """
        相邻 ICP 失败时，必须清掉 curr.ckpt 中可能残留的 refined edge。
        否则后续 PGO 可能继续读取旧的 prev_submap_tsfm_refined，
        造成位姿链路被错误边污染。
        """
        ckpt_path = self.submap_records.get(curr_sid)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")

        dirty = False
        for key in [
            "prev_submap_tsfm_refined",
            "prev_submap_info_matrix",
            "prev_submap_metrics",
        ]:
            if key in ckpt:
                del ckpt[key]
                dirty = True

        if dirty:
            torch.save(ckpt, ckpt_path)
            Log(f"[AdjacentOdom] 已清理子图 {curr_sid} 的无效 refined adjacent edge")

    def _load_prev_to_curr_transition(self, prev_sid, curr_sid):
        """
        返回 curr -> prev 的相邻边变换。

        旧局部子图模式：
            可以从 relative_pose 串接子图 anchor。

        global seed 模式：
            2DGS 点云已经在全局坐标中；
            如果没有 ICP refined edge，就返回 I。
        """

        curr_ckpt_path = self.submap_records.get(curr_sid)

        # 1) 优先读取 curr.ckpt 中 refine 后的 curr -> prev
        if curr_ckpt_path is not None and os.path.exists(curr_ckpt_path):
            curr_ckpt = torch.load(curr_ckpt_path, map_location="cpu")

            if "prev_submap_tsfm_refined" in curr_ckpt:
                rel = curr_ckpt["prev_submap_tsfm_refined"]
                if isinstance(rel, torch.Tensor):
                    rel = rel.numpy()
                return np.array(rel, dtype=np.float64)

        # 2) global seed 模式下，没有 refined edge 就返回 I
        if self.use_global_seed_submap:
            return np.eye(4, dtype=np.float64)

        # 3) 旧模式兼容：读取 prev.ckpt 中的 relative pose
        prev_ckpt_path = self.submap_records.get(prev_sid)
        if prev_ckpt_path is not None and os.path.exists(prev_ckpt_path):
            prev_ckpt = torch.load(prev_ckpt_path, map_location="cpu")

            rel = prev_ckpt.get(
                "next_submap_relative_pose",
                prev_ckpt.get("relative_pose", np.eye(4, dtype=np.float64)),
            )

            if isinstance(rel, torch.Tensor):
                rel = rel.numpy()

            return np.array(rel, dtype=np.float64)

        return np.eye(4, dtype=np.float64)

    def _load_prev_to_curr_information(self, prev_sid, curr_sid):
        """
        返回 curr -> prev 这条相邻边的信息矩阵。

        global seed 模式下：
            如果相邻 ICP 没成功，就只给极弱默认边，不能让失败边强约束 PGO。
        """
        curr_ckpt_path = self.submap_records.get(curr_sid)

        if curr_ckpt_path is not None and os.path.exists(curr_ckpt_path):
            curr_ckpt = torch.load(curr_ckpt_path, map_location="cpu")
            info = curr_ckpt.get("prev_submap_info_matrix", None)

            if info is not None:
                if isinstance(info, torch.Tensor):
                    info = info.numpy()
                return np.array(info, dtype=np.float64)

        if self.use_global_seed_submap:
            return np.identity(6, dtype=np.float64) * float(
                self.global_seed_default_odom_info_scale
            )

        return np.identity(6, dtype=np.float64) * (
                float(self.default_odom_info_scale) * float(self.adjacent_edge_weight)
        )

    # compute_relative_transform()  -> self.submap_pcds回环边ICP用的点云是feature cloud，特征点云更稀疏但更稳定，适合非相邻子图的粗匹配；相邻子图的 refine 则用 dense cloud，提供更多点对支持更精细的配准。函数内部会自动判断当前边是相邻 refine 还是非相邻 loop ICP，并选择对应的点云和更严格的门槛进行配准和一致性校验。
    def compute_relative_transform(self, source_id, target_id, current_pose_guesses):
        try:
            if not self._ensure_pcd_loaded(source_id):
                Log(f"[LoopClosure] 无法加载子图 {source_id} 的点云，跳过 ICP")
                return np.identity(4), np.identity(6), False
            if not self._ensure_pcd_loaded(target_id):
                Log(f"[LoopClosure] 无法加载子图 {target_id} 的点云，跳过 ICP")
                return np.identity(4), np.identity(6), False

            source_pcd = self.submap_pcds[source_id]
            target_pcd = self.submap_pcds[target_id]

            # source -> target 的先验
            init_guess = (
                    np.linalg.inv(current_pose_guesses[target_id]) @
                    current_pose_guesses[source_id]
            )

            coarse_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size * 8.0,
                init=init_guess,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=60, relative_fitness=1e-6, relative_rmse=1e-6
                )
            )

            medium_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size * 4.0,
                init=coarse_icp.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100, relative_fitness=1e-7, relative_rmse=1e-7
                )
            )

            fine_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size * 2.0,
                init=medium_icp.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=150, relative_fitness=1e-8, relative_rmse=1e-8
                )
            )

            icp_result = fine_icp

            if icp_result.fitness < self.icp_fitness_threshold or icp_result.inlier_rmse > 0.04:
                Log(f"[!] ICP 配准失败: 子图 {source_id}->{target_id} "
                    f"(Fitness: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.3f}m)")
                return np.identity(4), np.identity(6), False

            transformation = np.array(icp_result.transformation, dtype=np.float64)

            # 与先验做一致性校验
            delta = transformation @ np.linalg.inv(init_guess)
            delta_t = np.linalg.norm(delta[:3, 3])
            delta_r = self._rotation_error_deg(transformation, init_guess)

            max_loop_delta_t = self.config.get("LoopClosure", {}).get("max_loop_delta_translation", 0.80)
            max_loop_delta_r = self.config.get("LoopClosure", {}).get("max_loop_delta_rotation_deg", 45.0)

            if delta_t > max_loop_delta_t or delta_r > max_loop_delta_r:
                Log(
                    f"[!] LOOP 一致性校验失败: 子图 {source_id}->{target_id} | "
                    f"delta_t={delta_t:.3f}m, delta_r={delta_r:.2f}deg | 拒绝该闭环边"
                )
                return np.identity(4), np.identity(6), False

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

            return transformation, information, True

        except Exception as e:
            Log(f"[LoopClosure] ICP 异常: {source_id}->{target_id}: {e}")
            return np.identity(4), np.identity(6), False

    #refine_adjacent_submap_edge() -> self.submap_dense_pcds相邻 refine 用 dense cloud，提供更多点对支持更精细的配准。函数内部会自动判断当前边是相邻 refine 还是非相邻 loop ICP，并选择对应的点云和更严格的门槛进行配准和一致性校验。refine_adjacent_submap_edge 专门针对相邻子图边进行精炼，使用更严格的 ICP 门槛和更密集的点云（dense cloud）来提升配准精度，同时引入偏差校验和降权机制，确保只有高质量的相邻边被写回并参与后续优化。
    def refine_adjacent_submap_edge(self, curr_sid):
        if curr_sid <= 0:
            return False

        prev_sid = curr_sid - 1

        if not self._ensure_pcd_loaded(curr_sid):
            return False
        if not self._ensure_pcd_loaded(prev_sid):
            return False

        all_ids = sorted(self.submap_records.keys())
        _, current_pose_guesses = self._build_current_pose_guesses(all_ids)

        init_guess = (
                np.linalg.inv(current_pose_guesses[prev_sid]) @
                current_pose_guesses[curr_sid]
        )

        # 改这里：相邻 refine 用 dense cloud
        source_pcd = self.submap_dense_pcds[curr_sid]
        target_pcd = self.submap_dense_pcds[prev_sid]

        coarse = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance=self.voxel_size * 8.0,
            init=init_guess,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=60, relative_fitness=1e-6, relative_rmse=1e-6
            ),
        )

        medium = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance=self.voxel_size * 4.0,
            init=coarse.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100, relative_fitness=1e-7, relative_rmse=1e-7
            ),
        )

        fine = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            max_correspondence_distance=self.voxel_size * 2.0,
            init=medium.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=150, relative_fitness=1e-8, relative_rmse=1e-8
            ),
        )

        if fine.fitness < self.adjacent_icp_fitness_threshold or fine.inlier_rmse > self.adjacent_icp_rmse_threshold:
            Log(
                f"[AdjacentOdom] 相邻子图 {curr_sid}->{prev_sid} 精炼失败 "
                f"(fitness={fine.fitness:.3f}, rmse={fine.inlier_rmse:.3f})"
            )
            self._clear_refined_adjacent_edge(curr_sid)
            return False

        refined = np.array(fine.transformation, dtype=np.float64)

        delta = refined @ np.linalg.inv(init_guess)
        delta_t = np.linalg.norm(delta[:3, 3])
        delta_r = self._rotation_error_deg(refined, init_guess)

        if delta_t > self.max_adjacent_delta_translation or delta_r > self.max_adjacent_delta_rotation_deg:
            Log(
                f"[AdjacentOdom] 相邻子图 {curr_sid}->{prev_sid} 偏差过大，拒绝写回 "
                f"(delta_t={delta_t:.3f}m, delta_r={delta_r:.2f}deg)"
            )
            return False

        large_delta = (
                delta_t > self.max_adjacent_delta_translation or
                delta_r > self.max_adjacent_delta_rotation_deg
        )

        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_pcd,
            target_pcd,
            self.voxel_size * 2.0,
            refined,
        )

        raw_conf = fine.fitness / max(fine.inlier_rmse, 1e-3)
        conf = float(np.clip(raw_conf, self.adjacent_info_min_scale, self.adjacent_info_max_scale))

        if large_delta:
            # 不拒绝，改为大幅降权
            penalty_t = np.exp(-max(0.0, delta_t - self.max_adjacent_delta_translation) * 4.0)
            penalty_r = np.exp(-max(0.0, delta_r - self.max_adjacent_delta_rotation_deg) * 0.15)
            soft_scale = max(0.05, penalty_t * penalty_r)
            Log(
                f"[AdjacentOdom] 相邻子图 {curr_sid}->{prev_sid} 偏差较大，但保留写回并降权 "
                f"(delta_t={delta_t:.3f}m, delta_r={delta_r:.2f}deg, soft_scale={soft_scale:.3f})"
            )
        else:
            soft_scale = 1.0

        info = np.array(info, dtype=np.float64) * conf * float(self.adjacent_edge_weight) * soft_scale

        ckpt_path = self.submap_records[curr_sid]
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["prev_submap_tsfm_refined"] = refined
        ckpt["prev_submap_info_matrix"] = info
        ckpt["prev_submap_metrics"] = {
            "fitness": float(fine.fitness),
            "rmse": float(fine.inlier_rmse),
            "delta_t": float(delta_t),
            "delta_r": float(delta_r),
            "raw_conf": float(raw_conf),
            "edge_weight": float(self.adjacent_edge_weight),
        }
        torch.save(ckpt, ckpt_path)

        Log(
            f"[AdjacentOdom] 已写回子图 {curr_sid} 的 refined prev edge | "
            f"fitness={fine.fitness:.3f}, rmse={fine.inlier_rmse:.3f}, "
            f"delta_t={delta_t:.3f}, delta_r={delta_r:.2f}, "
            f"raw_conf={raw_conf:.3f}, edge_weight={self.adjacent_edge_weight:.3f}"
        )
        return True

    def optimize_submap_chain(self, chain_submap_ids):
        """
        只对最近若干个子图做链式 PGO。
        没有真实闭环也允许优化，核心是把 refined adjacent edge 真正传播成 correction。
        """
        """
            只对最近若干个子图做链式 PGO。

            注意：
            global seed submap 模式下禁用该函数。
            因为此时子图已经是全局坐标，不能再用 relative_pose 链式传播 correction。
            """

        if self.use_global_seed_submap:
            Log("[ChainPGO] global-seed 子图模式下禁用无回环链式 PGO")
            return []

        if len(chain_submap_ids) < 2:
            return []

        all_submap_ids = sorted(self.submap_records.keys())
        open_loop_anchors, current_pose_guesses = self._build_current_pose_guesses(all_submap_ids)

        pose_graph = o3d.pipelines.registration.PoseGraph()
        id_mapping = {sid: i for i, sid in enumerate(chain_submap_ids)}

        for sid in chain_submap_ids:
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.array(current_pose_guesses[sid], dtype=np.float64)
                )
            )

        for i in range(1, len(chain_submap_ids)):
            prev_sid = chain_submap_ids[i - 1]
            curr_sid = chain_submap_ids[i]

            rel_prev_from_curr = self._load_prev_to_curr_transition(prev_sid, curr_sid)
            odom_source_to_target = np.linalg.inv(rel_prev_from_curr)
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

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 1.5,
            edge_prune_threshold=0.25,
            reference_node=0,
        )

        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        correction_list = []
        for i, sid in enumerate(chain_submap_ids):
            optimized_pose = np.array(pose_graph.nodes[i].pose, dtype=np.float64)
            anchor_pose = open_loop_anchors[sid]
            correction = optimized_pose @ np.linalg.inv(anchor_pose)
            correction_list.append({
                "submap_id": sid,
                "correct_tsfm": correction,
            })

        return correction_list

    def cleanup_old_submaps(self):
        """
        LRU 策略清理旧子图的点云缓存，只保留最近的 N 个。
        【关键改进】：只清理点云（占内存大），保留特征向量和阈值（占内存极小）。
        被清理的子图仍然可以参与回环检测（视觉粗筛），如果 ICP 需要其点云，
        会通过 _ensure_pcd_loaded 从磁盘自动重新加载。
        """
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

    def apply_correction_to_submaps(self, correction_list):
        """
        将 PGO correction 真正刚性应用到整个子图：
        1) 更新 gaussian_params['_xyz'] / '_rotation' / '_normal'
        2) 更新该子图内所有关键帧 c2w
        3) 写回 correct_tsfm
        4) 清空该子图点云缓存，避免 ICP 继续用旧点云
        """
        for correction in correction_list:
            submap_id = correction["submap_id"]
            new_correct_tsfm = np.array(correction["correct_tsfm"], dtype=np.float64)

            ckpt_path = self.submap_records.get(submap_id)
            if not ckpt_path or not os.path.exists(ckpt_path):
                continue

            submap_ckpt = torch.load(ckpt_path, map_location="cpu")

            prev_correct_tsfm = submap_ckpt.get(
                "correct_tsfm", np.eye(4, dtype=np.float64)
            )
            if isinstance(prev_correct_tsfm, torch.Tensor):
                prev_correct_tsfm = prev_correct_tsfm.cpu().numpy()
            prev_correct_tsfm = np.array(prev_correct_tsfm, dtype=np.float64)

            # 关键：如果之前已经应用过 correction，这次只能应用增量 delta，
            # 否则会重复变换整个子图。
            delta_tsfm = new_correct_tsfm @ np.linalg.inv(prev_correct_tsfm)

            if np.allclose(delta_tsfm, np.eye(4), atol=1e-6):
                submap_ckpt["correct_tsfm"] = new_correct_tsfm
                torch.save(submap_ckpt, ckpt_path)
                continue

            Log(
                f"[LoopClosure] Apply rigid correction to submap {submap_id}: "
                f"delta_t={np.linalg.norm(delta_tsfm[:3, 3]):.4f}m"
            )

            # 1) 整体刚性修正 2DGS 高斯地图
            gaussian_params = submap_ckpt["gaussian_params"]

            # rigid_transform_2dgs 内部会把矩阵转到 cuda；
            # 如果你的 loop closure 进程跑在 CPU 环境，需要把该函数里的 .cuda()
            # 改成根据 gaussian_params['_xyz'].device 自适应。
            device = gaussian_params["_xyz"].device
            gaussian_params = {
                k: (v.cuda() if torch.is_tensor(v) else v)
                for k, v in gaussian_params.items()
            }

            gaussian_params = rigid_transform_2dgs(gaussian_params, delta_tsfm)

            # 保存回 CPU，避免 ckpt 过度占 GPU 显存
            gaussian_params = {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in gaussian_params.items()
            }
            submap_ckpt["gaussian_params"] = gaussian_params

            # 2) 整体刚性修正该子图内所有关键帧 c2w
            kf_poses = submap_ckpt.get("submap_keyframe_poses", {})
            new_kf_poses = {}

            for kf_id, c2w in kf_poses.items():
                if isinstance(c2w, torch.Tensor):
                    c2w = c2w.cpu().numpy()
                c2w = np.array(c2w, dtype=np.float64)

                # correction 左乘到 c2w
                new_kf_poses[int(kf_id)] = (delta_tsfm @ c2w).astype(np.float64)

            submap_ckpt["submap_keyframe_poses"] = new_kf_poses

            # 3) 写入新的累计 correction
            submap_ckpt["correct_tsfm"] = new_correct_tsfm

            torch.save(submap_ckpt, ckpt_path)

            # 4) 清空缓存，防止后续 ICP 使用变换前的旧点云
            self.submap_pcds.pop(submap_id, None)
            self.submap_dense_pcds.pop(submap_id, None)

            if submap_id in self.submap_access_order:
                self.submap_access_order.remove(submap_id)


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

                    # 提取并缓存点云
                    Log(f"[LoopClosure] 提取并缓存子图 {submap_id} 的 3D 点云与特征...")
                    dense_pcd, feature_pcd = self.extract_pcd_from_2dgs_ckpt(ckpt_path)
                    self.submap_dense_pcds[submap_id] = dense_pcd
                    self.submap_pcds[submap_id] = feature_pcd

                    # 批量提取 CosPlace 特征并计算每帧的动态阈值
                    submap_desc, thresholds = self.extract_submap_features_and_threshold(img_paths)
                    self.submap_features[submap_id] = submap_desc
                    self.submap_thresholds[submap_id] = thresholds

                    Log(f"[LoopClosure] 接收并处理新子图: ID {submap_id} (包含 {len(img_paths)} 个关键帧)")

                    # 记录访问顺序（LRU 缓存）
                    if submap_id in self.submap_access_order:
                        self.submap_access_order.remove(submap_id)
                    self.submap_access_order.append(submap_id)

                    # 1) 先精炼相邻子图边
                    if submap_id > 0 and self.config.get("LoopClosure", {}).get("enable_adjacent_odom", False):
                        self.refine_adjacent_submap_edge(submap_id)

                    # 2) 即使没有真实闭环，也先对最近几段子图链做一次局部 PGO
                    if self.enable_chain_pgo_without_loop:
                        chain_ids = sorted(self.submap_records.keys())[-self.chain_pgo_window:]
                        local_chain_corr = self.optimize_submap_chain(chain_ids)
                        if len(local_chain_corr) > 0:
                            self.apply_correction_to_submaps(local_chain_corr)
                            Log(f"[ChainPGO] 已对最近 {len(chain_ids)} 个子图执行局部链式 PGO")

                    # 3) 再尝试真正的闭环 PGO
                    correction_list = self.construct_and_optimize_pose_graph()
                    if len(correction_list) > 0:
                        self.apply_correction_to_submaps(correction_list)
                        Log("==> PGO 闭环校正及硬盘回写完毕！ <==")

                    # 清理旧子图缓存
                    self.cleanup_old_submaps()
            else:
                time.sleep(0.5)
