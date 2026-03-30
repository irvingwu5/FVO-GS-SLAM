import os
import time
import torch
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp
import roma  # 用于四元数和旋转矩阵的转换
from utils.logging_utils import Log
# 【新增】：从你刚刚放进去的 netvlad.py 中导入模型
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

    # 2. 变换二维高斯的朝向 (严谨处理四元数顺序)
    if '_rotation' in gaussian_params:
        rotation_q = gaussian_params['_rotation']

        # 【核心修复】：3DGS 的四元数是 (w, x, y, z)，而 roma 需要 (x, y, z, w)
        # 将第 0 位 (w) 移到最后
        rotation_q_roma = rotation_q[:, [1, 2, 3, 0]]

        cur_rot_mat = roma.unitquat_to_rotmat(rotation_q_roma)
        new_rot_mat = R.unsqueeze(0) @ cur_rot_mat
        new_rotation_q_roma = roma.rotmat_to_unitquat(new_rot_mat).squeeze()

        # 计算完后再转换回 3DGS 的 (w, x, y, z) 格式
        # 将最后一位 (w) 移回最前面
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

        # 记录已保存子图的路径和其空间几何中心，用于启发式闭环检测
        self.submap_records = {}
        self.submap_centroids = {}
        # 视觉特征字典与相似度阈值
        self.submap_features = {}
        # 【重要】：NetVLAD 的向量非常紧凑，建议初始阈值设为 0.85 或 0.88
        # ==============================================================
        # 【精英法则 1】：提高视觉门槛！
        # 办公室里相似的东西太多，把 NetVLAD 的门槛从 0.85 提高到 0.92
        # ==============================================================
        self.sim_threshold = 0.92
        self.min_interval = self.config.get("LoopClosure", {}).get("min_interval", 3)
        self.voxel_size = self.config.get("LoopClosure", {}).get("voxel_size", 0.05)
        # 判定为有效闭环的 ICP Fitness 阈值 (重叠度下限)
        # ==============================================================
        # 【精英法则 2】：斩杀虚假 ICP！
        # 必须有至少 50% 的 3D 点云严丝合缝地贴在一起，才承认是回环！
        # 把之前的 0.25 改成 0.50，直接干掉那群 0.26 的“老鼠屎”！
        # ==============================================================
        self.icp_fitness_threshold = 0.50

    def init_feature_extractor(self):
        """
        利用你上传的 netvlad.py 官方写法初始化网络
        """
        Log("[LoopClosure] 初始化高级视觉检索网络 (ResNet18 + NetVLAD)...")

        # 1. 加载预训练的 ResNet18 作为特征提取骨干
        encoder = models.resnet18(pretrained=True)

        # 2. 丢弃最后的 Average Pooling 和 Linear 分类层
        base_model = nn.Sequential(
            encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool,
            encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        )

        # 获取基础网络输出的通道数 (ResNet18 是 512)
        dim = list(base_model.parameters())[-1].shape[0]

        # 3. 实例化 NetVLAD 层 (聚类中心设为 16 或 32，减小显存占用)
        net_vlad = NetVLAD(num_clusters=16, dim=dim, alpha=1.0)

        # 4. 使用 EmbedNet 将它们无缝组装
        self.feature_extractor = EmbedNet(base_model, net_vlad).eval().to(self.device)

        # 图像预处理流水线
        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def extract_image_feature(self, img_path):
        """提取图像的 NetVLAD 全局描述子"""
        img_tensor = torch.load(img_path).to(self.device)  # [3, H, W]
        img_tensor = self.img_transform(img_tensor).unsqueeze(0)

        with torch.no_grad():
            # net_vlad.py 里的实现已经自带了 L2 归一化
            feat = self.feature_extractor(img_tensor).squeeze()

        return feat.cpu().numpy()


    def extract_pcd_from_2dgs_ckpt(self, ckpt_path):
        """
        从 2DGS 的 .ckpt 文件中提取 Open3D 点云，并直接利用 2DGS 的旋转属性生成法线。
        这是 2DGS 进行 ICP 配准的核心优势：免去了耗时的 KD-Tree 法线估计。
        """
        submap_ckpt = torch.load(ckpt_path, map_location="cpu")
        gaussian_params = submap_ckpt["gaussian_params"]

        xyz = gaussian_params['_xyz'].numpy()

        # 提取 2DGS 法线：将四元数转为旋转矩阵，提取局部 Z 轴 (第三列) 作为法线
        rot_q = gaussian_params['_rotation']
        rot_mat = roma.unitquat_to_rotmat(rot_q).numpy()  # (N, 3, 3)
        normals = rot_mat[:, :, 2]  # 取每个旋转矩阵的第 3 列

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # 体素降采样以加速 ICP
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return pcd

    def detect_closure(self, query_id):
        """
        【全新升级】：NetVLAD 视觉特征匹配 + 空间距离双重校验
        """
        matched_ids = []
        if query_id not in self.submap_features:
            return matched_ids

        query_feat = self.submap_features[query_id]
        query_centroid = self.submap_centroids[query_id]

        for db_id, db_feat in self.submap_features.items():
            if abs(query_id - db_id) > self.min_interval:

                # 1. 核心：计算 NetVLAD 特征的余弦相似度 (由于已归一化，点乘即余弦)
                sim = np.dot(query_feat, db_feat)

                # 如果视觉相似度超过阈值
                if sim > self.sim_threshold:
                    # 2. 辅助校验：松弛几何距离，放宽到 4 米
                    db_centroid = self.submap_centroids[db_id]
                    dist = np.linalg.norm(query_centroid - db_centroid)

                    # =========================================================
                    # 【核心修复 2】：恢复理智的空间距离校验！
                    # 在室内场景，前端漂移不可能超过 2 米。
                    # 如果距离超过 2 米，就算长得一模一样，也肯定是“另一把椅子”！
                    # =========================================================
                    if dist < 2.0:
                        Log(f"[*] 🚀 NetVLAD 视觉回环触发! 子图 {query_id} -> {db_id} (相似度: {sim:.3f}, 距离: {dist:.2f}m)")
                        matched_ids.append(db_id)
                    else:
                        Log(f"[!] 拦截假回环: 子图 {query_id} -> {db_id} (视觉极度相似，但物理距离过远 {dist:.2f}m)")

        return matched_ids

    def compute_relative_transform(self, source_id, target_id):
        """
        加载 2DGS 点云，执行 ICP。
        """
        try:
            source_pcd = self.extract_pcd_from_2dgs_ckpt(self.submap_records[source_id])
            target_pcd = self.extract_pcd_from_2dgs_ckpt(self.submap_records[target_id])

            # ==============================================================
            # 【核心修复 1】：Coarse-to-Fine 且必须使用 PointToPoint！
            # 彻底杜绝平面滑动效应，只允许严丝合缝的几何镶嵌！
            # ==============================================================
            coarse_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 1.5, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint() # <== 必须是 PointToPoint
            )

            fine_icp = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, self.voxel_size * 2.0, coarse_icp.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint() # <== 必须是 PointToPoint
            )

            icp_result = fine_icp

            if icp_result.fitness < self.icp_fitness_threshold:
                return np.identity(4), np.identity(6), False

            transformation = icp_result.transformation

            # ==============================================================
            # 【核心修复 2】：降低闭环拉力！(50.0 -> 10.0)
            # ==============================================================
            weight = icp_result.fitness * 10.0  # <== 改成了 10.0
            information = np.identity(6) * weight

            Log(f"[ICP] 子图 {source_id}->{target_id} 匹配成功! Fitness: {icp_result.fitness:.3f}, 拉力权重: {weight:.1f}")
            return transformation, information, True

        except Exception as e:
            Log(f"[LoopClosure] ICP 计算失败 {source_id}->{target_id}: {e}")
            return np.identity(4), np.identity(6), False

    def construct_and_optimize_pose_graph(self):
        """
        使用纯 Python 的 python-graphslam
        """
        def mat2pose(mat):
            pos = mat[:3, 3].copy()
            rot_mat = mat[:3, :3].copy()
            rot_quat = Rotation.from_matrix(rot_mat).as_quat()
            return PoseSE3(pos, rot_quat)

        vertices = []
        edges = []
        n_submaps = max(self.submap_records.keys()) + 1

        for i in range(n_submaps):
            vertices.append(Vertex(i, PoseSE3(np.zeros(3), [0., 0., 0., 1.])))

        # ==============================================================
        # 【核心修复 3】：为里程计注入钢铁之魂！(5.0 -> 500.0)
        # 誓死捍卫前端 1.6cm 的极限精度，绝不向错误的闭环妥协！
        # ==============================================================
        info_odom = np.identity(6) * 500.0  # <== 改成了 500.0
        for i in range(1, n_submaps):
            source_id = i
            target_id = i - 1
            edges.append(EdgeOdometry([target_id, source_id], info_odom, mat2pose(np.identity(4))))

        loop_found = False
        for source_id in range(1, n_submaps):
            matches = self.detect_closure(source_id)
            for target_id in matches:
                trans, info_loop, success = self.compute_relative_transform(source_id, target_id)
                if success:
                    edges.append(EdgeOdometry([target_id, source_id], info_loop, mat2pose(trans)))
                    loop_found = True

        if not loop_found:
            return []

        Log("检测到有效闭环，启动纯 Python-GraphSLAM 全局优化...")
        graph = Graph(edges, vertices)

        graph.optimize(tol=1e-4, max_iter=100)

        correction_list = []
        T_0_inv = np.linalg.inv(graph._vertices[0].pose.to_matrix())

        for v in graph._vertices:
            opt_trans = v.pose.to_matrix()
            final_trans = T_0_inv @ opt_trans
            correction_list.append({
                'submap_id': v.id,
                'correct_tsfm': final_trans
            })

        return correction_list

    def apply_correction_to_submaps(self, correction_list):
        """
        仅仅将优化后的位姿修正矩阵记录到硬盘上的 .ckpt 文件中。
        真正的坐标变换放到 slam.py 合并大图时实时应用。杜绝递归旋转！
        """
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

            # 【绝对关键】：千万不要再调用 rigid_transform_2dgs 去改 gaussian_params 了！
            # 仅仅记录 PGO 计算出的全局修正矩阵
            submap_ckpt["correct_tsfm"] = correct_tsfm

            torch.save(submap_ckpt, ckpt_path)

    def _cache_submap_centroid(self, submap_id, ckpt_path):
        """
        加载 .ckpt 并缓存该子图的几何中心，用于启发式重叠检测。
        """
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            xyz = ckpt["gaussian_params"]["_xyz"]
            self.submap_centroids[submap_id] = xyz.mean(dim=0).numpy()
        except Exception as e:
            Log(f"[LoopClosure] 提取子图中心失败: {e}")

    def run(self):
        Log("Loop Closure 进程已启动，后台静默监听中...")
        # 启动即初始化基于 NetVLAD 的检索网络
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
                    img_path = data[3]

                    self.submap_records[submap_id] = ckpt_path
                    self._cache_submap_centroid(submap_id, ckpt_path)
                    # 提取图像特征
                    self.submap_features[submap_id] = self.extract_image_feature(img_path)

                    Log(f"[LoopClosure] 接收并处理新子图: ID {submap_id}")

                    # 尝试构建位姿图并优化
                    correction_list = self.construct_and_optimize_pose_graph()

                    if len(correction_list) > 0:
                        self.apply_correction_to_submaps(correction_list)
                        Log("==> PGO 闭环校正及硬盘回写完毕！ <==")
            else:
                time.sleep(0.5)