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
        # 确保完全不经过 GPU
        submap_ckpt = torch.load(ckpt_path, map_location="cpu")
        gp = submap_ckpt["gaussian_params"]

        # 转换为 numpy，numpy 存储在内存（RAM）而非显存
        xyz = gp['_xyz'].numpy()
        rot_q = gp['_rotation']

        # 在 CPU 上计算法线
        rot_mat = roma.unitquat_to_rotmat(rot_q).numpy()
        normals = rot_mat[:, :, 2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # 显式清理大型字典
        del submap_ckpt, gp
        return pcd

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
        try:
            # ==============================================================
            # 【升级 5】：直接从内存字典中读取点云，省去高频的硬盘 I/O，速度飙升！
            # ==============================================================
            source_pcd = self.submap_pcds[source_id]
            target_pcd = self.submap_pcds[target_id]

            coarse_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 1.5, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane()  # 完美利用 2DGS 法线
            )

            fine_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, self.voxel_size * 2.0, coarse_icp.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()  # 完美利用 2DGS 法线
            )

            icp_result = fine_icp

            # ==============================================================
            # 【终极修复 2】：不仅查 Fitness，还要严查 RMSE！宁缺毋滥！
            # 对于室内高精度 SLAM，配准误差 (inlier_rmse) 超过 0.05 (5厘米) 直接斩杀
            # ==============================================================
            # 动态门限：可以在 __init__ 里把 self.icp_fitness_threshold 提高到 0.60
            if icp_result.fitness < self.icp_fitness_threshold or icp_result.inlier_rmse > 0.05:
                Log(f"[!] ❌ 几何精核失败 (ICP)! 子图 {source_id}->{target_id} 精度不达标 "
                    f"(重叠度: {icp_result.fitness:.3f}, RMSE误差: {icp_result.inlier_rmse:.3f}m)，已防毒剔除。")
                return np.identity(4), np.identity(6), False

            transformation = icp_result.transformation

            # 引入 RMSE 作为权重惩罚项：误差越大，PGO 中对这条边的信任度越低
            weight = (icp_result.fitness / (icp_result.inlier_rmse + 1e-6)) * 0.1
            information = np.identity(6) * weight

            Log(f"[ICP] ✅ 子图 {source_id}->{target_id} 几何配准成功! (Fitness: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.3f}m, 拉力权重: {weight:.1f})")
            return transformation, information, True

            transformation = icp_result.transformation
            weight = icp_result.fitness * 10.0
            information = np.identity(6) * weight

            Log(f"[ICP] ✅ 子图 {source_id}->{target_id} 几何配准成功! 严丝合缝! (Fitness: {icp_result.fitness:.3f}, 拉力权重: {weight:.1f})")
            return transformation, information, True

        except Exception as e:
            Log(f"[LoopClosure] ICP 计算异常中断 {source_id}->{target_id}: {e}")
            return np.identity(4), np.identity(6), False

    def construct_and_optimize_pose_graph(self):
        import open3d as o3d

        # 1. 初始化 Open3D 位姿图
        pose_graph = o3d.pipelines.registration.PoseGraph()
        n_submaps = max(self.submap_records.keys()) + 1

        # 2. 添加节点 (Nodes)
        # 初始假设所有子图都在原点（或当前的初步位姿），PGO 会优化它们
        for i in range(n_submaps):
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(np.identity(4))
            )

        # 3. 添加里程计边 (Odometry Edges) - 标记为 uncertain=False
        # 这些边是我们高度信任的，通常不会被剪枝
        info_odom = np.identity(6) * 50.0  # 刚度较强
        for i in range(1, n_submaps):
            source_id = i
            target_id = i - 1
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id,
                    np.identity(4),  # 相邻子图间的初始相对变换
                    info_odom,
                    uncertain=False
                )
            )

        # 4. 添加回环边 (Loop Closure Edges) - 标记为 uncertain=True
        # 这是剪枝机制生效的对象
        loop_found = False
        for source_id in range(1, n_submaps):
            matches = self.detect_closure(source_id)
            for target_id in matches:
                trans, info_loop, success = self.compute_relative_transform(source_id, target_id)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            source_id, target_id,
                            trans,
                            info_loop,
                            uncertain=True  # 🎯 关键：标记为不确定边，允许被剪枝
                        )
                    )
                    loop_found = True

        if not loop_found:
            return []

        Log(f"检测到有效闭环，启动 Open3D 全局优化 (带边剪枝机制)...")

        # 5. 🎯 配置优化选项 (实现 LoopSplat 机制)
        # 从 config 读取阈值，如果没有，建议设置在 0.1 ~ 5.0 之间
        prune_threshold = self.config.get("LoopClosure", {}).get("pgo_edge_prune_thres", 1.0)

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 1.5,
            edge_prune_threshold=prune_threshold,  # 🎯 机制核心：残差超过此值的 uncertain 边将被剔除
            reference_node=0
        )

        # 6. 执行优化
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )

        # 7. 提取优化后的结果
        correction_list = []
        # 以第一个节点为参考坐标系，计算相对修正
        T_0_inv = np.linalg.inv(pose_graph.nodes[0].pose)

        for i in range(len(pose_graph.nodes)):
            opt_trans = pose_graph.nodes[i].pose
            final_trans = T_0_inv @ opt_trans
            correction_list.append({
                'submap_id': i,
                'correct_tsfm': final_trans
            })

        return correction_list

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

                    correction_list = self.construct_and_optimize_pose_graph()

                    if len(correction_list) > 0:
                        self.apply_correction_to_submaps(correction_list)
                        Log("==> PGO 闭环校正及硬盘回写完毕！ <==")
            else:
                time.sleep(0.5)