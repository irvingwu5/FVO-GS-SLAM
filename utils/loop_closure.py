import os
import time
import torch
import numpy as np
import open3d as o3d
import torch.multiprocessing as mp
import roma  # 用于四元数和旋转矩阵的转换
from utils.logging_utils import Log


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

        self.min_interval = self.config.get("LoopClosure", {}).get("min_interval", 3)
        self.voxel_size = self.config.get("LoopClosure", {}).get("voxel_size", 0.05)
        # 判定为有效闭环的 ICP Fitness 阈值 (重叠度下限)
        self.icp_fitness_threshold = 0.25

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
        基于空间距离启发式的闭环检测 (可平滑替换为 DBoW3 / NetVLAD)。
        此处返回潜在的闭环候选 ID。
        """
        matched_ids = []
        if query_id not in self.submap_centroids:
            return matched_ids

        query_centroid = self.submap_centroids[query_id]

        for db_id, db_centroid in self.submap_centroids.items():
            if abs(query_id - db_id) > self.min_interval:
                # 启发式距离检测：如果两个子图的几何中心距离小于 2.0 米，则认为有空间重叠的可能
                dist = np.linalg.norm(query_centroid - db_centroid)
                if dist < 2.0:
                    matched_ids.append(db_id)

        # TODO: 若后续接入 NetVLAD，只需将上述空间距离判断替换为:
        # cosine_similarity(query_desc, db_desc) > similarity_threshold

        return matched_ids

    def compute_relative_transform(self, source_id, target_id):
        """
        加载 2DGS 点云，执行 Point-to-Plane ICP。
        返回 4x4 变换矩阵、信息矩阵以及配准是否成功的布尔值。
        """
        try:
            source_pcd = self.extract_pcd_from_2dgs_ckpt(self.submap_records[source_id])
            target_pcd = self.extract_pcd_from_2dgs_ckpt(self.submap_records[target_id])

            max_correspondence_distance = self.voxel_size * 3.0

            # 利用 2DGS 原生高精度法线执行 Point-to-Plane ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, max_correspondence_distance,
                np.identity(4),  # 初始猜测 (对于SLAM，闭环通常处于同一全局坐标系附近)
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            # 通过 Fitness (重叠度) 和 Inlier RMSE 拒绝错误的闭环匹配
            if icp_result.fitness < self.icp_fitness_threshold:
                return np.identity(4), np.identity(6), False

            transformation = icp_result.transformation

            # # 计算信息矩阵，用于衡量这条闭环边在图优化中的可靠程度 (权重)
            # information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            #     source_pcd, target_pcd, max_correspondence_distance, transformation
            # )
            # 【核心修复 1】：废弃 Open3D 动辄几十万的暴力量化矩阵！
            # 强制将 ICP 回环边的信息矩阵（权重）设定为一个极小的软约束。
            # 让它只负责拉回全局累积的低频漂移，而不破坏局部高频轨迹的精确度。
            information = np.identity(6) * 0.1

            Log(f"[ICP] 子图 {source_id}->{target_id} 匹配成功! Fitness: {icp_result.fitness:.3f}, RMSE: {icp_result.inlier_rmse:.4f}")
            return transformation, information, True

        except Exception as e:
            Log(f"[LoopClosure] ICP 计算失败 {source_id}->{target_id}: {e}")
            return np.identity(4), np.identity(6), False

    def construct_and_optimize_pose_graph(self):
        """
        构建 Open3D 位姿图并执行 Levenberg-Marquardt 全局优化。
        """
        pose_graph = o3d.pipelines.registration.PoseGraph()
        n_submaps = max(self.submap_records.keys()) + 1

        for i in range(n_submaps):
            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.identity(4)))

        # 添加相邻里程计边
        for i in range(1, n_submaps):
            source_id = i
            target_id = i - 1
            # 相邻子图在保存时已经处于同一世界坐标系下，理论相对变换为单位阵
            trans = np.identity(4)
            # 【核心修复 1】：将极其刚硬的里程计权重(10.0)调低到(0.5)。
            # 这样位姿图才能变得“柔软”，允许 ICP 算出的回环误差将漂移的轨迹真正拉回正轨！
            info = np.identity(6) * 1000.0
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_id, target_id, trans, info, uncertain=False
                )
            )

        # 添加跨度闭环边
        loop_found = False
        for source_id in range(1, n_submaps):
            matches = self.detect_closure(source_id)
            for target_id in matches:
                trans, info, success = self.compute_relative_transform(source_id, target_id)
                if success:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            source_id, target_id, trans, info, uncertain=True
                        )
                    )
                    loop_found = True

        if not loop_found:
            return []

        Log("检测到有效闭环，正在执行全局位姿图优化 (PGO)...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=self.voxel_size * 2.0,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )

        correction_list = []
        for id in range(n_submaps):
            submap_correction = {
                'submap_id': id,
                'correct_tsfm': pose_graph.nodes[id].pose
            }
            correction_list.append(submap_correction)

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
        while True:
            if not self.loop_queue.empty():
                data = self.loop_queue.get()
                if data[0] == "stop":
                    Log("Loop Closure 进程退出.")
                    break
                elif data[0] == "submap_saved":
                    submap_id = data[1]
                    ckpt_path = data[2]
                    self.submap_records[submap_id] = ckpt_path
                    self._cache_submap_centroid(submap_id, ckpt_path)
                    Log(f"[LoopClosure] 接收并处理新子图: ID {submap_id}")

                    # 尝试构建位姿图并优化
                    correction_list = self.construct_and_optimize_pose_graph()

                    if len(correction_list) > 0:
                        self.apply_correction_to_submaps(correction_list)
                        Log("==> PGO 闭环校正及硬盘回写完毕！ <==")
            else:
                time.sleep(0.5)