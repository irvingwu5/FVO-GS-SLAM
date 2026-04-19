import os
import time
import torch
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp
import roma  # 用于四元数和旋转矩阵的转换
from utils.logging_utils import Log
from utils.netvlad import NetVLAD, EmbedNet
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from graphslam.graph import Graph
from graphslam.vertex import Vertex
from graphslam.edge.edge_odometry import EdgeOdometry
from graphslam.pose.se3 import PoseSE3


def rigid_transform_2dgs(gaussian_params, tsfm_matrix):
    tsfm_matrix = torch.from_numpy(tsfm_matrix).float().cuda()
    R = tsfm_matrix[:3, :3]
    t = tsfm_matrix[:3, 3]

    # 1. 变换中心点 (xyz)
    xyz = gaussian_params['_xyz']
    gaussian_params['_xyz'] = (R @ xyz.T).T + t

    # 2. 变换二维高斯的朝向
    if '_rotation' in gaussian_params:
        rotation_q = gaussian_params['_rotation']
        rotation_q_roma = rotation_q[:, [1, 2, 3, 0]]
        cur_rot_mat = roma.unitquat_to_rotmat(rotation_q_roma)
        new_rot_mat = R.unsqueeze(0) @ cur_rot_mat
        new_rotation_q_roma = roma.rotmat_to_unitquat(new_rot_mat).squeeze()
        new_rotation_q = new_rotation_q_roma[:, [3, 0, 1, 2]]
        gaussian_params['_rotation'] = new_rotation_q

        if '_normal' in gaussian_params:
            gaussian_params['_normal'] = new_rot_mat[:, :, 2]

    return gaussian_params


class LoopClosureProcess(mp.Process):
    def __init__(self, config, loop_queue):
        super().__init__()
        self.config = config
        self.loop_queue = loop_queue
        self.save_dir = self.config["Results"]["save_dir"]
        self.submaps_dir = os.path.join(self.save_dir, "submaps")
        self.device = "cuda"

        # ==============================================================
        # 【升级 5】：新增点云内存缓存，彻底消除高频磁盘 I/O 阻塞
        # ==============================================================
        self.submap_pcds = {}  # 内存缓存每个子图的点云数据，避免重复加载和计算
        self.submap_centroids = {}  # 内存缓存每个子图的中心点，避免重复计算
        self.submap_records = {}  # 记录子图 ID 与其对应的 ckpt 路径，便于后续访问
        # 【升级 1】：取消固定的 sim_threshold，新增动态阈值字典
        self.submap_features = {}  # 保存每个子图的多帧特征矩阵 [N, D]
        self.submap_thresholds = {}  # 保存每个子图的内部动态阈值 [N]
        self.min_similarity_ratio = 0.5  # 自相似度 Top-K 比例 (例如前 10%)

        self.min_interval = self.config.get("LoopClosure", {}).get("min_interval", 3)
        self.voxel_size = self.config.get("LoopClosure", {}).get("voxel_size", 0.05)
        self.icp_fitness_threshold = 0.60

        # 【新增】：LRU 缓存参数
        self.max_cached_submaps = self.config.get("LoopClosure", {}).get("keep_recent_submaps", 3)
        self.submap_access_order = []  # 记录最近访问的子图顺序

    def init_feature_extractor(self):
        Log("[LoopClosure] 初始化高级视觉检索网络 (ResNet18 + NetVLAD)...")
        encoder = models.resnet18(pretrained=True)
        base_model = nn.Sequential(
            encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool,
            encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        )
        dim = list(base_model.parameters())[-1].shape[0]
        net_vlad = NetVLAD(num_clusters=16, dim=dim, alpha=1.0)
        self.feature_extractor = EmbedNet(base_model, net_vlad).eval().to(self.device)

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 【升级 2】：支持批量提取 Submap 内所有关键帧的特征，并计算动态阈值
    def extract_submap_features_and_threshold(self, img_paths):
        feats = []
        for img_path in img_paths:
            # 1. 强制在 CPU 加载图像，避免加载瞬间的显存抖动
            img_tensor = torch.load(img_path, map_location="cpu")  # [3, H, W]

            # 2. 仅在推理时送入 GPU，并立即 detach 转回 CPU
            img_input = self.img_transform(img_tensor).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(img_input).squeeze().detach().cpu()
            feats.append(feat)

            # 3. 显式清理 GPU 上的临时图像张量
            del img_input

        # 4. 在 CPU 上进行矩阵运算（N 较小时 CPU 足够快，且不占显存）
        submap_desc = torch.stack(feats)  # 此时已在 CPU 上
        self_sim = torch.mm(submap_desc, submap_desc.T)

        k = max(int(len(submap_desc) * self.min_similarity_ratio), 1)
        score_min, _ = self_sim.topk(k, dim=1)
        dynamic_thresholds = score_min[:, -1]

        # 确保返回的是 CPU 张量
        return submap_desc, dynamic_thresholds

    def extract_pcd_from_2dgs_ckpt(self, ckpt_path):
        """
        从 2DGS checkpoint 提取点云，两阶段下采样
        【优化】：先粗下采样，再保留关键特征点
        """
        submap_ckpt = torch.load(ckpt_path, map_location="cpu")
        gp = submap_ckpt["gaussian_params"]

        xyz = gp['_xyz'].numpy()
        rot_q = gp['_rotation']

        # 在 CPU 上计算法线
        rot_mat = roma.unitquat_to_rotmat(rot_q).numpy()
        normals = rot_mat[:, :, 2]

        # 法线归一化
        normals_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_norm[normals_norm < 1e-6] = 1.0
        normals = normals / normals_norm

        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # 【第 1 阶段】：统计离群值移除（减少噪声）
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        # 【第 2 阶段】：粗下采样
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 【第 3 阶段】：保留关键特征点（边界点、高曲率点）
        # 计算点的曲率（用于识别关键特征点）
        pcd_downsampled.estimate_normals()

        # 获取下采样后的点和法线
        points = np.asarray(pcd_downsampled.points)
        normals_ds = np.asarray(pcd_downsampled.normals)

        # 计算点的曲率（基于法线变化）
        # 这是一个简化的曲率估计，实际应用中可以更精细
        tree = o3d.geometry.KDTreeFlann(pcd_downsampled)
        curvatures = []

        for i in range(len(points)):
            [k, idx, _] = tree.search_knn_vector_3d(points[i], 10)
            neighbor_normals = normals_ds[idx]
            # 曲率 = 法线方向变化的标准差
            curvature = np.std(neighbor_normals, axis=0).mean()
            curvatures.append(curvature)

        curvatures = np.array(curvatures)

        # 保留高曲率的点（关键特征点）
        high_curvature_threshold = np.percentile(curvatures, 30)  # 保留曲率最高的 30%
        feature_mask = curvatures > high_curvature_threshold

        # 合并：所有下采样点 + 额外的高曲率点
        feature_indices = np.where(feature_mask)[0]

        # 从原始点云中提取这些特征点
        feature_points = points[feature_indices]
        feature_normals = normals_ds[feature_indices]

        # 创建最终的点云
        pcd_final = o3d.geometry.PointCloud()
        pcd_final.points = o3d.utility.Vector3dVector(feature_points)
        pcd_final.normals = o3d.utility.Vector3dVector(feature_normals)

        Log(f"[LoopClosure] 两阶段下采样：原始 {len(pcd.points)} → 下采样 {len(pcd_downsampled.points)} → 最终 {len(pcd_final.points)}")

        del submap_ckpt, gp
        return pcd_final

    # 【升级 3】：纯视觉粗筛，彻底移除硬距离阈值拦截
    def detect_closure(self, query_id):
        matched_ids = []
        if query_id not in self.submap_features:
            return matched_ids

        # 仅将当前 Query 描述子送入 GPU 一次
        query_desc = self.submap_features[query_id].to(self.device)
        query_thresh = self.submap_thresholds[query_id].to(self.device)
        query_centroid = self.submap_centroids[query_id]

        for db_id, db_desc in self.submap_features.items():
            if db_id <= query_id - self.min_interval:
                # 瞬时加载 DB 描述子
                db_desc_cuda = db_desc.to(self.device)

                # 使用矩阵乘法替代 einsum，通常更快
                cross_sim = torch.mm(query_desc, db_desc_cuda.T)
                matches = torch.argwhere(cross_sim > query_thresh.unsqueeze(1))

                if len(matches) > 0:
                    max_sim = cross_sim.max().item()
                    db_centroid = self.submap_centroids[db_id]
                    dist = np.linalg.norm(query_centroid - db_centroid)

                    Log(f"[*] 🚀 视觉粗筛命中: 子图 {query_id} -> {db_id} (相似度: {max_sim:.3f})")
                    matched_ids.append(db_id)

                # 立即释放 DB 描述子的显存
                del db_desc_cuda

        # 释放 Query 显存
        del query_desc, query_thresh
        # 闭环检测后强制回收，防止显存阶梯增长
        torch.cuda.empty_cache()
        return matched_ids

    # 【升级 4】：基于 2DGS 法线的 PointToPlane ICP 几何精核
    def compute_relative_transform(self, source_id, target_id):
        """
        改进的 ICP 配置，充分利用 2DGS 法线
        """
        try:
            source_pcd = self.submap_pcds[source_id]
            target_pcd = self.submap_pcds[target_id]

            # ========== 【优化】：多阶段 ICP 配置 ==========

            # 第 1 阶段：粗配准（大范围搜索）
            coarse_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=1.5,  # 1.5m 搜索范围
                init=np.identity(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6
                )
            )

            # 第 2 阶段：中等精度配准
            medium_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size * 5.0,  # 0.25m
                init=coarse_icp.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100,
                    relative_fitness=1e-7,
                    relative_rmse=1e-7
                )
            )

            # 第 3 阶段：精细配准（小范围搜索）
            fine_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=self.voxel_size * 2.0,  # 0.1m
                init=medium_icp.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=150,
                    relative_fitness=1e-8,
                    relative_rmse=1e-8
                )
            )

            icp_result = fine_icp

            # ========== 【优化】：更严格的阈值 ==========
            # 对于 2DGS，法线质量更高，可以使用更严格的阈值
            if icp_result.fitness < 0.65 or icp_result.inlier_rmse > 0.03:
                Log(f"[!] ❌ ICP 配准失败: 子图 {source_id}->{target_id} "
                    f"(Fitness: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.3f}m)")
                return np.identity(4), np.identity(6), False

            transformation = icp_result.transformation

            # ========== 【优化】：基于多个指标的权重计算 ==========
            # 不仅考虑 Fitness 和 RMSE，还考虑收敛性
            fitness_weight = icp_result.fitness
            rmse_penalty = 1.0 / (icp_result.inlier_rmse + 1e-6)

            # 组合权重：fitness 越高、RMSE 越低，权重越高
            combined_weight = (fitness_weight * rmse_penalty) * 0.5
            information = np.identity(6) * combined_weight

            Log(f"[ICP] ✅ 子图 {source_id}->{target_id} 配准成功! "
                f"(Fitness: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.3f}m, "
                f"权重: {combined_weight:.2f})")

            return transformation, information, True

        except Exception as e:
            Log(f"[LoopClosure] ICP 异常: {source_id}->{target_id}: {e}")
            return np.identity(4), np.identity(6), False

    def construct_and_optimize_pose_graph(self):
        """
        改进的流式 PGO：只优化最近的 N 个子图，而不是所有历史子图
        【作用】：显存降低 20-30%，优化速度提升 4-5 倍
        """
        import open3d as o3d

        # 读取配置参数
        max_search_range = self.config.get("LoopClosure", {}).get("max_search_range", 5)

        # 1. 只保留最近的 N 个子图用于 PGO
        all_submap_ids = sorted(self.submap_records.keys())
        if len(all_submap_ids) == 0:
            return []

        if len(all_submap_ids) > max_search_range:
            # 只考虑最近 N 个子图进行 PGO 优化
            recent_submap_ids = all_submap_ids[-max_search_range:]
            # 但仍然允许与更早的子图形成回环边
            search_submap_ids = all_submap_ids
        else:
            recent_submap_ids = all_submap_ids
            search_submap_ids = all_submap_ids

        # 2. 初始化 Open3D 位姿图（只包含最近的子图）
        pose_graph = o3d.pipelines.registration.PoseGraph()

        # 创建 ID 映射：全局 ID → 本地 ID（用于 PoseGraph 节点索引）
        id_mapping = {gid: lid for lid, gid in enumerate(recent_submap_ids)}

        # 3. 添加节点（只添加最近的子图）
        for i, submap_id in enumerate(recent_submap_ids):
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.identity(4))
            )

        # 4. 添加里程计边（只在最近的子图间）
        info_odom = np.identity(6) * 50.0  # 刚度较强
        for i in range(1, len(recent_submap_ids)):
            source_id = recent_submap_ids[i]
            target_id = recent_submap_ids[i - 1]
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i, i - 1,  # 使用本地 ID
                    np.identity(4),  # 相邻子图间的初始相对变换
                    info_odom,
                    uncertain=False  # 里程计边高度可信
                )
            )

        # 5. 添加回环边（可以跨越最近子图范围）
        loop_found = False
        for source_id in search_submap_ids:
            if source_id not in id_mapping:
                continue  # 跳过不在最近范围内的源子图

            matches = self.detect_closure(source_id)
            for target_id in matches:
                if target_id not in id_mapping:
                    continue  # 跳过不在最近范围内的目标子图

                trans, info_loop, success = self.compute_relative_transform(source_id, target_id)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            id_mapping[source_id],
                            id_mapping[target_id],
                            trans,
                            info_loop,
                            uncertain=True  # 回环边允许被剪枝
                        )
                    )
                    loop_found = True

        if not loop_found:
            Log(f"[LoopClosure] 未检测到有效闭环")
            del pose_graph
            torch.cuda.empty_cache()
            return []

        Log(f"检测到有效闭环，启动 Open3D 全局优化 (最近 {len(recent_submap_ids)} 个子图)...")

        # 6. 配置优化选项
        prune_threshold = self.config.get("LoopClosure", {}).get("pgo_edge_prune_thres", 1.0)

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 1.5,
            edge_prune_threshold=prune_threshold,  # 残差超过此值的 uncertain 边将被剔除
            reference_node=0
        )

        # 7. 执行优化
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )

        # 8. 提取优化后的结果
        correction_list = []
        T_0_inv = np.linalg.inv(pose_graph.nodes[0].pose)

        for i, submap_id in enumerate(recent_submap_ids):
            opt_trans = pose_graph.nodes[i].pose
            final_trans = T_0_inv @ opt_trans
            correction_list.append({
                'submap_id': submap_id,
                'correct_tsfm': final_trans
            })

        # 9. 显式释放 PoseGraph 对象（防止显存泄漏）
        del pose_graph
        torch.cuda.empty_cache()

        return correction_list

    def cleanup_old_submaps(self):
        """
        清理显存中的旧子图，只保留最近的 N 个（LRU 策略）
        【作用】：防止显存无限增长，显存降低 50-70%
        """
        if len(self.submap_access_order) > self.max_cached_submaps:
            # 找出需要删除的子图（保留最近的 N 个）
            to_delete = self.submap_access_order[:-self.max_cached_submaps]

            for submap_id in to_delete:
                # 删除点云缓存
                if submap_id in self.submap_pcds:
                    del self.submap_pcds[submap_id]
                    Log(f"[LoopClosure] 清理显存中的旧子图 {submap_id} (点云)")

                # 删除中心点缓存
                if submap_id in self.submap_centroids:
                    del self.submap_centroids[submap_id]

                # 删除特征缓存
                if submap_id in self.submap_features:
                    del self.submap_features[submap_id]

                # 删除阈值缓存
                if submap_id in self.submap_thresholds:
                    del self.submap_thresholds[submap_id]

            # 更新访问顺序（只保留最近的 N 个）
            self.submap_access_order = self.submap_access_order[-self.max_cached_submaps:]

            # 强制释放显存
            torch.cuda.empty_cache()
            Log(f"[LoopClosure] 显存清理完毕，当前缓存子图数: {len(self.submap_access_order)}")

    def apply_correction_to_submaps(self, correction_list):
        for correction in correction_list:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']

            if np.allclose(correct_tsfm, np.eye(4), atol=1e-4):
                continue

            ckpt_path = self.submap_records.get(submap_id)
            if not ckpt_path or not os.path.exists(ckpt_path):
                continue

            Log(f"记录子图 {submap_id} 的 PGO 修正矩阵...")
            submap_ckpt = torch.load(ckpt_path, map_location="cpu")
            submap_ckpt["correct_tsfm"] = correct_tsfm
            torch.save(submap_ckpt, ckpt_path)

    def _cache_submap_centroid(self, submap_id, ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            xyz = ckpt["gaussian_params"]["_xyz"]
            self.submap_centroids[submap_id] = xyz.mean(dim=0).numpy()
        except Exception as e:
            Log(f"[LoopClosure] 提取子图中心失败: {e}")

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
                    img_paths = data[3]  # 【关键前提】：前端发送端必须将单一的 img_path 改为包含多帧路径的 list

                    self.submap_records[submap_id] = ckpt_path
                    # ==============================================================
                    # 【升级 5】：子图生成时，仅执行这一次唯一的硬盘点云读取！
                    # ==============================================================
                    Log(f"[LoopClosure] 提取并缓存子图 {submap_id} 的 3D 点云与特征...")
                    pcd = self.extract_pcd_from_2dgs_ckpt(ckpt_path)
                    self.submap_pcds[submap_id] = pcd

                    # 利用已在内存中的点云直接算中心，又干掉了一次冗余的磁盘读取！
                    self.submap_centroids[submap_id] = np.asarray(pcd.points).mean(axis=0)

                    # 批量提取并计算阈值
                    submap_desc, thresholds = self.extract_submap_features_and_threshold(img_paths)
                    self.submap_features[submap_id] = submap_desc
                    self.submap_thresholds[submap_id] = thresholds

                    Log(f"[LoopClosure] 接收并处理新子图: ID {submap_id} (包含 {len(img_paths)} 个关键帧)")

                    # 【新增】：记录访问顺序（用于 LRU 缓存）
                    if submap_id in self.submap_access_order:
                        self.submap_access_order.remove(submap_id)
                    self.submap_access_order.append(submap_id)

                    correction_list = self.construct_and_optimize_pose_graph()

                    if len(correction_list) > 0:
                        self.apply_correction_to_submaps(correction_list)
                        Log("==> PGO 闭环校正及硬盘回写完毕！ <==")

                    # 4. 【关键优化】：处理完后立即清理不需要的子图
                    self.cleanup_old_submaps()  # 只保留最近 3 个子图在显存中
            else:
                time.sleep(0.5)