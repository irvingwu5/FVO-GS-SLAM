import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import numpy as np
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import (get_loss_mapping,_save_normal_pair, _save_rendered_rgb,_save_gt_normal,check_normal_dir)
import torch.nn.functional as F
# 它主要负责全局地图构建（Mapping）和光束法平差（Bundle Adjustment）
# BackEnd 的核心设计模式是维护全局一致性。前端只关心“当前在哪里”，而后端关心“整个地图长什么样以及历史轨迹是否准确”。
class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None #后端到前端的通信队列
        self.backend_queue = None #前端到后端的通信队列

        # ========= 新增：预留 Loop Closure 队列的属性 =========
        self.loop_queue = None
        # ======================================================

        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # 兼容旧配置，如果没写 Ablation 字段，默认保持开启 (True)
        self.use_fdn = self.config.get("Ablation", {}).get("use_fdn", True)
        # self.use_surf_normal = config["Training"]["sagsslam"]["use_surf_normal"]
        # self.use_distortion_loss = config["Training"]["sagsslam"]["use_distortion_loss"]
        # self.use_plane_constraint = config["Training"]["sagsslam"]["use_plane_constraint"]

        self.current_submap_id = 0
        # 【新增】：读取显存管理参数
        self.empty_cache_on_submap_cut = self.config.get("MemoryManagement", {}).get("empty_cache_on_submap_cut", True)
        self.reuse_render_tensors = self.config.get("MemoryManagement", {}).get("reuse_render_tensors", True)
        self.keyframe_window_size = self.config.get("MemoryManagement", {}).get("keyframe_window_size", 10)
        self.gradient_accumulation_frames = self.config.get("MemoryManagement", {}).get("gradient_accumulation_frames",5)
        # ===== submap cut 前的局部收紧 =====
        self.enable_cut_local_ba = self.config.get("Submap", {}).get("enable_cut_local_ba", True)
        self.cut_local_ba_iters = self.config.get("Submap", {}).get("cut_local_ba_iters", 40)
        self.cut_local_prune_iters = self.config.get("Submap", {}).get("cut_local_prune_iters", 8)
        # ========== 【新增】方案 3：智能关键点选择配置 ==========
        self.use_critical_point_selection = self.config.get("Submap", {}).get("use_critical_point_selection", True)
        self.selection_strategy = self.config.get("Submap", {}).get("selection_strategy", "advanced")
        self.retention_ratio = self.config.get("Submap", {}).get("retention_ratio", 0.15)
        self.num_spatial_clusters = self.config.get("Submap", {}).get("num_spatial_clusters", 10)
        # ======================================================
        # 在 __init__ 中，约第 61 行后新增：
        self.true_independent_submap = self.config.get("Submap", {}).get("true_independent_submap", False)
        self.seed_init_iters = self.config.get("Submap", {}).get("seed_init_iters", 500)

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"] # 首帧初始化时优化的迭代次数
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"] # 初始化期间执行高斯点分裂/修剪的间隔次数
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"] # 初始化期间重置不透明度的迭代步数
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"] # 初始化期间用于高斯密化（新增点）的梯度阈值
        # 对densify(clone and split)的影响:init_gs_extent值越大，阈值=percent_dense*该值，更多gs被判为小，倾向于clone,该值越小，阈值越小，更多gs被判为大，倾向于split
        # 对prune的影响:init_gs_extent值越大,容忍上限变高,巨大浮空伪影无法被剔除导致画面朦胧，该值越小,容忍上限变低,正常背景墙面地板可能因为尺寸稍大而被误删，导致背景出现空洞
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"] # 建图阶段每处理一个新关键帧时的优化迭代次数150
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"] # 建图期间执行高斯点分裂/修剪的间隔次数
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"] # 建图期间首次执行高斯更新的起始偏移量
        self.gaussian_th = self.config["Training"]["gaussian_th"] # 建图期间用于高斯密化（新增点）的梯度阈值
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        ) # 建图期间场景范围6.0*30=180.0m，gs scaling>该值*0.1时被剔除
        self.gaussian_reset = self.config["Training"]["gaussian_reset"] # 建图期间重置不透明度的周期（用于去除漂浮物/噪声）
        self.size_threshold = self.config["Training"]["size_threshold"] # 高斯点的修剪阈值（过大的点会被移除）
        self.window_size = self.config["Training"]["window_size"] # 滑动窗口的大小（参与联合优化/BA的关键帧数量）
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        ) # 是否单线程运行后端
    #扩展地图: 调用 add_next_kf (即 gaussians.extend_from_pcd_seq)，利用新关键帧的深度图在未知区域初始化新的高斯点。
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # 只有在还有点的时候才执行 prune
        if len(self.gaussians._xyz) > 0:
            self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

        # 清空队列（注意：在子图切换场景下，队列中可能还有前端发来的 init 消息，
        # 这里不应该清空。但由于 new_submap 分支已经处理完毕，
        # 此时队列中的下一条消息就是 init，不会被误删。）
        # 保持原有逻辑不变
        while not self.backend_queue.empty():
            self.backend_queue.get()

    # ============================================================================
    # 【新增】方案 3：混合策略 - 智能关键点选择
    # ============================================================================

    def select_critical_gaussians_advanced(self, submap_id, retention_ratio=0.15,
                                           num_spatial_clusters=10):
        """
        使用混合策略选择关键高斯点（方案 3）

        结合不确定性、可见性和空间分布，选择最有价值的高斯点。

        Args:
            submap_id (int): 子图 ID
            retention_ratio (float): 保留比例（10-20%）
            num_spatial_clusters (int): 空间聚类数（8-12）

        Returns:
            np.ndarray: 选中的高斯点索引数组，或 None 表示失败
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            Log("[Error] sklearn 未安装，请运行: pip install scikit-learn scipy")
            return None

        try:
            # ========== 步骤 1：加载子图数据 ==========
            save_dir = self.config["Results"]["save_dir"]
            ckpt_path = os.path.join(save_dir, "submaps", f"{submap_id:06d}.ckpt")

            if not os.path.exists(ckpt_path):
                Log(f"[Warning] 子图文件不存在: {ckpt_path}，使用默认策略")
                return None

            ckpt = torch.load(ckpt_path, map_location="cpu")
            gaussian_params = ckpt.get("gaussian_params", {})

            xyz = gaussian_params.get("_xyz", torch.zeros((0, 3))).numpy()
            scaling = gaussian_params.get("_scaling", torch.zeros((len(xyz), 3))).numpy()

            if len(xyz) == 0:
                Log(f"[Warning] 子图 {submap_id} 没有高斯点")
                return None

            # ========== 步骤 2：计算不确定性 ==========
            uncertainty = np.mean(scaling, axis=1)
            uncertainty_min = uncertainty.min()
            uncertainty_max = uncertainty.max()
            if uncertainty_max > uncertainty_min:
                uncertainty_norm = (uncertainty - uncertainty_min) / (uncertainty_max - uncertainty_min)
            else:
                uncertainty_norm = np.zeros_like(uncertainty)

            # ========== 步骤 3：计算可见性 ==========
            opacity = gaussian_params.get("_opacity", torch.ones((len(xyz), 1))).numpy().squeeze()
            opacity_min = opacity.min()
            opacity_max = opacity.max()
            if opacity_max > opacity_min:
                visibility_norm = (opacity - opacity_min) / (opacity_max - opacity_min)
            else:
                visibility_norm = np.ones_like(opacity)

            # ========== 步骤 4：计算空间重要性 ==========
            k_neighbors = min(10, len(xyz) - 1)
            nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(xyz)
            distances, indices = nbrs.kneighbors(xyz)
            mean_distances = distances[:, 1:].mean(axis=1)

            dist_min = mean_distances.min()
            dist_max = mean_distances.max()
            if dist_max > dist_min:
                spatial_importance = (mean_distances - dist_min) / (dist_max - dist_min)
            else:
                spatial_importance = np.ones_like(mean_distances)

            # ========== 步骤 5：计算综合价值分数 ==========
            value_score = (1 - uncertainty_norm) * visibility_norm * spatial_importance

            # ========== 步骤 6：选择价值分数最高的前 N% 的点 ==========
            num_retain = max(100, int(len(xyz) * retention_ratio))
            top_indices = np.argsort(value_score)[-num_retain:]

            Log(f"[CriticalPointSelection] 子图 {submap_id}: "
                f"保留 {len(top_indices)}/{len(xyz)} 点 "
                f"({100 * len(top_indices) / len(xyz):.1f}%), "
                f"显存节省 {100 * (1 - len(top_indices) / len(xyz)):.1f}%")

            # ========== 步骤 7：对选中的点进行空间聚类 ==========
            selected_xyz = xyz[top_indices]
            num_clusters = min(num_spatial_clusters, len(top_indices) // 20)
            num_clusters = max(1, num_clusters)

            if num_clusters > 1:
                kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(selected_xyz)
            else:
                labels = np.zeros(len(selected_xyz), dtype=int)

            # ========== 步骤 8：从每个空间聚类中均匀采样 ==========
            final_indices = []

            for cluster_id in range(num_clusters):
                cluster_mask = labels == cluster_id
                cluster_local_indices = np.where(cluster_mask)[0]

                if len(cluster_local_indices) == 0:
                    continue

                points_per_cluster = max(10, len(cluster_local_indices) // 2)

                if len(cluster_local_indices) <= points_per_cluster:
                    sampled_local_indices = cluster_local_indices
                else:
                    cluster_value_score = value_score[top_indices[cluster_local_indices]]
                    importance_weights = cluster_value_score / cluster_value_score.sum()
                    sampled_local_indices = np.random.choice(
                        cluster_local_indices,
                        size=points_per_cluster,
                        replace=False,
                        p=importance_weights
                    )

                global_indices = top_indices[sampled_local_indices]
                final_indices.extend(global_indices)

            final_indices = np.array(final_indices, dtype=int)

            Log(f"[CriticalPointSelection] 最终保留 {len(final_indices)} 个点 "
                f"({100 * len(final_indices) / len(xyz):.1f}%)")

            return final_indices

        except Exception as e:
            Log(f"[Error] 关键点选择失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def select_critical_gaussians_simple(self, submap_id, retention_ratio=0.15):
        """
        简化版本的关键点选择（如果混合策略出现问题，可以使用这个）

        只基于不确定性和可见性，不进行空间聚类。

        Args:
            submap_id (int): 子图 ID
            retention_ratio (float): 保留比例（10-20%）

        Returns:
            np.ndarray: 选中的高斯点索引数组，或 None 表示失败
        """
        try:
            save_dir = self.config["Results"]["save_dir"]
            ckpt_path = os.path.join(save_dir, "submaps", f"{submap_id:06d}.ckpt")

            if not os.path.exists(ckpt_path):
                return None

            ckpt = torch.load(ckpt_path, map_location="cpu")
            gaussian_params = ckpt.get("gaussian_params", {})

            xyz = gaussian_params.get("_xyz", torch.zeros((0, 3))).numpy()
            scaling = gaussian_params.get("_scaling", torch.zeros((len(xyz), 3))).numpy()

            if len(xyz) == 0:
                return None

            # 计算不确定性
            uncertainty = np.mean(scaling, axis=1)
            uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-6)

            # 计算可见性
            opacity = gaussian_params.get("_opacity", torch.ones((len(xyz), 1))).numpy().squeeze()
            opacity_norm = (opacity - opacity.min()) / (opacity.max() - opacity.min() + 1e-6)

            # 计算价值分数
            value_score = (1 - uncertainty_norm) * opacity_norm

            # 选择价值分数最高的前 N% 的点
            num_retain = max(100, int(len(xyz) * retention_ratio))
            final_indices = np.argsort(value_score)[-num_retain:]

            return final_indices

        except Exception as e:
            Log(f"[Error] 简化关键点选择失败: {str(e)}")
            return None
    '''
    ------------------初始建图模块(Initialization)------------------
    作用: 处理系统启动时的第一帧数据。
    逻辑:
    清空旧地图。
    基于第一帧生成的点云初始化高斯模型。
    执行高频的迭代优化（init_itr_num），快速建立初始场景几何，为前端跟踪提供基础。
    #初始化阶段只需要把场景几何/颜色的高斯模型尽快收敛到一个可用的状态，位姿保持固定（单位矩阵）并不参与优化。
    在没有位姿优化的情况下快速建立稳定的几何/颜色基础，给前端提供可靠的跟踪基准（后续帧和 BA 才会优化相机位姿）。
    '''
    def initialize_map(self, cur_frame_idx, viewpoint, iters=None): #第一帧多次迭代优化，每次迭代独立计算并及时更新高斯参数，不是多次迭代累积梯度
        # 第一帧/子图 seed 帧多次迭代优化
        if iters is None:
            iters = self.init_itr_num

        for mapping_iteration in range(iters):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, initialization=True
            ) #0.4255

            loss_init.backward() #计算对gs模型参数的梯度（此阶段不更新相机位姿）

            with torch.no_grad(): #更新统计量
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                #
                if mapping_iteration % self.init_gaussian_update == 0: #在初始化建图阶段每隔self.init_gaussian_update次迭代执行一次高斯点的密化（densify）与修剪（prune）
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold, #基于梯度的密化阈值，表示哪些区域的梯度足够大，需要在该处增加新的高斯点以补充细节。
                        self.init_gaussian_th, #用于密化时的另一个阈值（如亮度/不透明度/半径方面的门限），限制新增点的条件
                        self.init_gaussian_extent, #在初始化阶段限定新增高斯点的空间范围（场景范围），防止在太远或不相关区域创建高斯。
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity() #将gs不透明度重置为0.01

                self.gaussians.optimizer.step() #执行gs参数更新
                self.gaussians.optimizer.zero_grad(set_to_none=True) #清空梯度
            # 可视化推送
            if mapping_iteration % 5 == 0:
                self.push_to_frontend()

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():  # 遍历窗口内所有关键帧
            if cam_idx in current_window_set:  # 跳过当前窗口内的帧
                continue
            random_viewpoint_stack.append(viewpoint)  # 其余帧加入随机采样列表

        # 获取当前窗口中最新的帧的索引
        for itr in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            # 🔴 修改点 1：删除原有的 loss_mapping = 0，不再将多帧 loss 拼接成一个巨大的计算图
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            keyframes_opt = []

            # 对多帧渲染计算联合损失，对当前滑动窗口内的关键帧进行优化
            for i in range(len(current_window)):
                viewpoint = viewpoint_stack[i]
                keyframes_opt.append(viewpoint)

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
                )
                # 解包数据
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                # 🔴 修改点 2：将全局 loss_mapping 改为单帧局部变量 loss_view
                loss_view = get_loss_mapping(
                    self.config, image, depth, viewpoint
                )

                if self.use_fdn and viewpoint.normal is not None:
                    rend_normal = render_pkg["rend_normal"]
                    rend_normal = F.normalize(rend_normal, p=2, dim=0)
                    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
                    sensor_normal = viewpoint.normal
                    gt_normal = (viewpoint.T[0:3, 0:3].T @ sensor_normal.view(3, -1)).view(
                        image.shape[0], image.shape[1], image.shape[2]
                    )
                    normal_error = (1 - (rend_normal * gt_normal * depth_pixel_mask).sum(dim=0))[None].mean()
                    loss_view += (self.config["opt_params"]["lambda_sensor_normal"] * normal_error)

                # 🔴 修改点 3：【核心】单帧计算完毕，立刻反向传播！
                # 此时 PyTorch 会把梯度累加到模型参数的 .grad 中，并立刻释放这一帧庞大的渲染计算图
                loss_view.backward()

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                # 【新增】：显式释放渲染包（防止显存泄漏）
                del render_pkg

                # 【新增】：每 5 帧强制清理一次显存
                if i % 5 == 0:
                    torch.cuda.empty_cache()
            # 随机选取的历史关键帧进行迭代优化
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                # 🔴 修改点 4：历史帧同样使用局部变量 loss_view
                loss_view = get_loss_mapping(
                    self.config, image, depth, viewpoint
                )

                # 🔴 修改点 5：立刻反向传播，释放历史帧计算图
                loss_view.backward()

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

                # 【新增】：显式释放渲染包（防止显存泄漏）
                del render_pkg
                torch.cuda.empty_cache()
            # 🔴 修改点 6：彻底删除原来在循环外面的 loss_mapping.backward() ！！！
            # (如果你有 Isotropic Loss 等全局损失，应在上方单独算完后调用 .backward())

            gaussian_split = False
            ## Deinsifying / Pruning Gaussians高斯密度自适应控制模块
            # 该模块负责动态调整高斯球的数量和分布，以适应场景的几何细节
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # 可见性修剪: 统计高斯球被观测的次数 (n_obs)。
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                        if prune_mode == "slam":
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                        self.iteration_count % self.gaussian_update_every
                        == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                # 只有在优化器被初始化后才执行 step
                if self.keyframe_optimizers is not None:
                    # 🔴 关键机制：所有上面累加在 .grad 里的梯度，在这里一次性更新！
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)

                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                    frames_to_optimize = self.config["Training"]["pose_window"]
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if getattr(viewpoint, "fixed_pose", False):
                            viewpoint.reset_pose_deltas()
                            continue
                        update_pose(viewpoint)
                else:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)

        return gaussian_split

    def finalize_submap_before_freeze(self):
        """
        在切图冻结前，对旧子图当前窗口做一次额外局部优化。
        目标：把边界关键帧位姿和局部几何尽量压稳，再写入 ckpt。
        """
        if not self.enable_cut_local_ba:
            return

        if len(self.current_window) == 0:
            return

        if self.keyframe_optimizers is None:
            Log("[SubmapLocalBA] skip: keyframe_optimizers is None")
            return

        ba_iters = max(int(self.cut_local_ba_iters), 0)
        prune_iters = max(int(self.cut_local_prune_iters), 0)

        Log(
            f"[SubmapLocalBA] freeze 前局部优化开始 | "
            f"window={len(self.current_window)}, "
            f"ba_iters={ba_iters}, prune_iters={prune_iters}"
        )

        if ba_iters > 0:
            self.map(self.current_window, prune=False, iters=ba_iters)

        if prune_iters > 0:
            self.map(self.current_window, prune=True, iters=prune_iters)

        Log("[SubmapLocalBA] freeze 前局部优化完成")
    '''
    ------------------离线精修模块(Color Refinement)------------------
    作用: 在 SLAM 过程结束后（或暂停时），对地图进行高质量的离线渲染优化。
    逻辑: 执行大量的迭代（如 26000 次），冻结几何结构调整（不再分裂/删除点），仅微调高斯球的颜色和不透明度，以获得最佳的视觉效果
    '''
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]

            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background, surf=False
            )
            image, viewspace_point_tensor, visibility_filter, radii, depth, opacity, n_touched = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
            # 【新增】：显式释放 render_pkg
            del render_pkg

            # 【新增】：每 100 次迭代清理一次显存
            if iteration % 100 == 0:
                torch.cuda.empty_cache()
        Log("Map refinement done")

    '''
    -------------------前后端同步模块(Frontend Sync)------------------
    作用: 将后端优化后的最新地图状态反馈给前端。
    逻辑:
    深拷贝当前的高斯模型 (clone_obj(self.gaussians))。
    打包优化后的关键帧位姿。
    通过 frontend_queue 发送给前端，确保前端跟踪线程使用的是经过后端精修的、更高质量的地图。
    '''
    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def get_dynamic_retention_frames(self):
        """动态调整保留帧数"""
        if not self.config.get("Submap", {}).get("dynamic_retention", False):
            return self.config.get("Submap", {}).get("retention_frames", 12)

        allocated_gb = torch.cuda.memory_allocated() / 1024 ** 3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        usage_ratio = allocated_gb / total_gb

        high_threshold = self.config.get("Submap", {}).get("vram_usage_threshold_high", 0.8)
        low_threshold = self.config.get("Submap", {}).get("vram_usage_threshold_low", 0.6)

        if usage_ratio > high_threshold:
            return self.config.get("Submap", {}).get("retention_frames_aggressive", 6)
        elif usage_ratio > low_threshold:
            return self.config.get("Submap", {}).get("retention_frames_balanced", 10)
        else:
            return self.config.get("Submap", {}).get("retention_frames_quality", 15)


    #作用: 整个后端进程的调度器，负责处理前端指令并执行持续的建图优化
    def run(self): #后端进程并行运行，主要响应前端的指令或在空闲时持续优化。
        while True:
            if self.backend_queue.empty(): #前端暂时未发送指令，空闲时持续优化 (队列为空时)
                if self.pause: #如果处于暂停状态、无关键帧或单线程模式，则挂起等待。
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                # 如果没有位姿优化器，说明正处于两个子图的交接点，且尚未收到新子图的第一个关键帧
                # 此时跳过 map，避免在 None 对象上纠结
                if self.keyframe_optimizers is None:
                    time.sleep(0.01)
                    continue

                if self.pause or len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window) #执行建图 (map): 对当前窗口进行优化。
                #周期性同步: 当累计迭代次数达到阈值（last_sent >= 10），执行一次带修剪的建图优化，并将最新状态同步给前端 (push_to_frontend)。
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else: #响应前端的指令，指令处理 (非空时)
                data = self.backend_queue.get()
                if data[0] == "stop":  # 终止循环，退出进程
                    save_dir = self.config["Results"]["save_dir"]
                    submaps_dir = os.path.join(save_dir, "submaps")
                    os.makedirs(submaps_dir, exist_ok=True)

                    gaussian_params = self.gaussians.capture_dict()
                    submap_keyframes = sorted(list(self.viewpoints.keys()))
                    ckpt_data = {
                        "gaussian_params": gaussian_params,
                        "submap_keyframes": submap_keyframes,
                        "correct_tsfm": np.eye(4)
                    }
                    ckpt_path = os.path.join(submaps_dir, f"{self.current_submap_id:06d}.ckpt")
                    torch.save(ckpt_data, ckpt_path)

                    # =========================================================
                    # 【核心修正】：终局保存也要保存多帧列表，而不是单张图片！
                    # =========================================================
                    kf_image_paths = []
                    if len(submap_keyframes) > 0:
                        for kf_idx in submap_keyframes:
                            kf_image = self.viewpoints[kf_idx].original_image.cpu()
                            # 文件名带上关键帧的 ID，防止覆盖
                            img_path = os.path.join(submaps_dir, f"{self.current_submap_id:06d}_img_{kf_idx}.pt")
                            torch.save(kf_image, img_path)
                            kf_image_paths.append(img_path)

                    if hasattr(self, 'loop_queue') and self.loop_queue is not None and len(kf_image_paths) > 0:
                        # 确保发送的是 list，而不是单字符串
                        self.loop_queue.put(["submap_saved", self.current_submap_id, ckpt_path, kf_image_paths])

                    Log(f"==> 终局保存：最后一块子图 {self.current_submap_id} 已存入硬盘。 <==")
                    break
                elif data[0] == "pause": #设置暂停状态标志
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement": #执行离线色彩精修，随后向前端推送结果。
                    self.color_refinement()
                    self.push_to_frontend()
                #系统重置->设置初始视点->扩展初始点云(add_next_kf)->执行初始建图优化(initialize_map)->不带修剪地推送到前端
                elif data[0] == "init": #系统重置并初始化第一帧
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    if self.true_independent_submap and len(self.gaussians._xyz) == 0 and self.current_submap_id > 0:
                        # 子图切换后的初始化：状态已在 new_submap 中清空，无需再 reset
                        Log("Initializing new submap from seed frame (state already clean)")
                        self.iteration_count = 0
                        self.occ_aware_visibility = {}
                        self.viewpoints = {}
                        self.current_window = []
                        self.initialized = not self.monocular
                        self.keyframe_optimizers = None
                    else:
                        # 系统冷启动：执行完整的 reset
                        Log("Resetting the system")
                        self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    if getattr(viewpoint, "fixed_pose", False):
                        viewpoint.reset_pose_deltas()
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )

                    # 子图 seed 初始化用单独的迭代数
                    if self.true_independent_submap and len(self.gaussians._xyz) == 0 and self.current_submap_id > 0:
                        init_iters = self.seed_init_iters
                    else:
                        init_iters = self.init_itr_num

                    self.initialize_map(cur_frame_idx, viewpoint, iters=init_iters)
                    self.push_to_frontend("init")
                # 接收前端发送的新关键帧，将其纳入后端优化体系
                # 添加新关键帧 -> 扩展地图 (add_next_kf) -> 配置关键帧位姿优化器 -> 执行特定迭代次数的建图优化 (map)
                # -> 执行带修剪的建图 (map(..., prune=True)) -> 推送到前端。
                elif data[0] == "keyframe": # 接收前端发送的新关键帧，将其纳入后端优化体系
                    # 读取前端发送来的数据
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4] # viewpoint里没有深度图吗？
                    # 更新内部状态: 存储新关键帧的视点信息，更新当前滑动窗口
                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    # 扩展地图: 调用 add_next_kf (即 gaussians.extend_from_pcd_seq)，利用新关键帧的深度图在未知区域初始化新的高斯点。
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    # 配置优化器: 为新关键帧的位姿参数（旋转、平移）和曝光参数（Exposure A/B）初始化独立的优化器参数组
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize and not getattr(viewpoint, "fixed_pose", False):
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    # 局部联合优化几何与位姿（BA）
                    self.map(self.current_window, iters=iter_per_kf)
                    # 做可选的修剪（移除低观测数的高斯点）
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")

                #在真正独立子图模式下，后端只需保存旧子图并彻底清空状态，不再做智能选择和关键帧保留。新子图的初始化将由随后到来的 "init" 消息触发
                elif data[0] == "new_submap":
                    #前端backend_queue.put(["new_submap"
                    completed_submap_id = data[1] #已经完成的子图id
                    # 接收前端传来的相对位姿（如果是第一个子图，可能没有这个参数，默认为单位阵）
                    relative_pose = data[2] if len(data) > 2 else np.eye(4)

                    self.current_submap_id = completed_submap_id + 1 #更新当前子图 ID，为下一个子图做准备
                    Log(f"==> Backend received new_submap signal. Freezing submap {completed_submap_id}...")
                    # ===== 新增：先对子图边界做一次局部收紧，再冻结 =====
                    self.finalize_submap_before_freeze()
                    save_dir = self.config["Results"]["save_dir"]
                    submaps_dir = os.path.join(save_dir, "submaps")
                    os.makedirs(submaps_dir, exist_ok=True)
                    # ========== 步骤 1：保存前一个子图的全部高斯参数到磁盘 ==========
                    gaussian_params = self.gaussians.capture_dict()
                    submap_keyframes = sorted(list(self.viewpoints.keys()))
                    ckpt_data = {
                        "gaussian_params": gaussian_params,
                        "submap_keyframes": submap_keyframes,
                        # PGO 修正矩阵
                        "correct_tsfm": np.eye(4),
                        # 兼容旧逻辑：保留旧字段
                        "relative_pose": relative_pose,
                        # 新字段：明确语义
                        # 这是“下一子图 seed 帧在当前(已完成)子图坐标系下”的位姿
                        # 也就是 T_prev<-next
                        "next_submap_relative_pose": relative_pose,
                    }

                    ckpt_path = os.path.join(submaps_dir, f"{completed_submap_id:06d}.ckpt")
                    torch.save(ckpt_data, ckpt_path)
                    Log(f"✓ Submap {completed_submap_id} parameters saved to {ckpt_path}")

                    # ========== 步骤 2：保存关键帧图像（用于回环检测） ==========
                    kf_image_paths = []
                    if len(submap_keyframes) > 0:
                        for kf_idx in submap_keyframes:
                            kf_image = self.viewpoints[kf_idx].original_image.cpu() #提取关键帧图像并保存，供后续回环检测使用
                            img_path = os.path.join(submaps_dir, f"{completed_submap_id:06d}_img_{kf_idx}.pt") #每张图像保存一个pt文件，文件名包含子图ID和关键帧ID，方便后续加载和对应
                            torch.save(kf_image, img_path)
                            kf_image_paths.append(img_path)

                    if hasattr(self, 'loop_queue') and self.loop_queue is not None and len(kf_image_paths) > 0:
                        self.loop_queue.put(["submap_saved", completed_submap_id, ckpt_path, kf_image_paths])
                        Log(f"✓ Submap {completed_submap_id} sent to loop closure")

                    # ========== 步骤 3：根据模式选择清理策略 ==========

                    if self.true_independent_submap:
                        # ====================================================
                        # 【真正独立子图模式】：彻底清空所有状态
                        # ====================================================
                        # 3a. 清空所有高斯点
                        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
                        Log("✓ Pruned ALL Gaussian points for true independent submap")
                        # 3b. 清空所有关键帧和优化器状态
                        self.viewpoints.clear()
                        self.current_window = []
                        self.occ_aware_visibility = {}
                        self.keyframe_optimizers = None
                        # 3c. 重新初始化高斯优化器（清空旧的动量和学习率状态）
                        self.gaussians.training_setup(self.opt_params)
                        # 3d. 彻底释放显存
                        torch.cuda.empty_cache()
                        Log("✓ Backend state fully reset. Waiting for seed frame init...")
                        # 注意：此模式下不调用 push_to_frontend()！
                        # 因为前端发送的是 "init" 请求，后端将在 "init" 分支中
                        # 执行 initialize_map() 并通过 push_to_frontend("init") 回传。
                    else:
                        # ====================================================
                        # 【流式子图模式】：保留部分旧点和关键帧（原有逻辑）
                        # ====================================================
                        retained_kfs = sorted(list(self.viewpoints.keys()))[-self.window_size:]
                        target_device = self.gaussians.unique_kfIDs.device
                        kf_mask = torch.zeros(len(self.gaussians._xyz), dtype=torch.bool, device=target_device)

                        for kf_id in retained_kfs:
                            kf_mask = kf_mask | (self.gaussians.unique_kfIDs == kf_id)

                        if self.use_critical_point_selection and kf_mask.sum() > 0:
                            try:
                                kf_indices_gpu = torch.where(kf_mask)[0]
                                xyz_subset = self.gaussians._xyz[kf_indices_gpu].detach().cpu().numpy()
                                scaling_subset = self.gaussians._scaling[kf_indices_gpu].detach().cpu().numpy()
                                opacity_subset = self.gaussians._opacity[
                                    kf_indices_gpu].detach().cpu().numpy().squeeze()
                                uncertainty = np.mean(scaling_subset, axis=1)
                                u_min, u_max = uncertainty.min(), uncertainty.max()
                                uncertainty_norm = (uncertainty - u_min) / (u_max - u_min + 1e-6)
                                o_min, o_max = opacity_subset.min(), opacity_subset.max()
                                visibility_norm = (opacity_subset - o_min) / (o_max - o_min + 1e-6)
                                value_score = (1 - uncertainty_norm) * visibility_norm
                                num_retain = max(100, int(len(xyz_subset) * self.retention_ratio))
                                top_local_indices = np.argsort(value_score)[-num_retain:]
                                top_global_indices = kf_indices_gpu[
                                    torch.from_numpy(top_local_indices).to(target_device)]
                                keep_mask = torch.zeros(len(self.gaussians._xyz), dtype=torch.bool,device=target_device)
                                keep_mask[top_global_indices] = True

                                Log(f"[Submap Cut] 智能选择：从 {len(retained_kfs)} 个关键帧的 "
                                    f"{kf_mask.sum().item()} 个点中保留 {keep_mask.sum().item()} 个")

                            except Exception as e:
                                Log(f"[Error] 智能选择失败: {e}，降级为仅保留关键帧归属点")
                                keep_mask = kf_mask
                        else:
                            keep_mask = kf_mask
                        old_mask = ~keep_mask
                        self.gaussians.prune_points(old_mask.cuda())
                        Log(f"✓ Pruned {old_mask.sum().item()} old Gaussian points")
                        self.gaussians.training_setup(self.opt_params)
                        retained_viewpoints = {kf: self.viewpoints[kf] for kf in retained_kfs if kf in self.viewpoints}
                        self.viewpoints.clear()
                        self.viewpoints.update(retained_viewpoints)
                        self.current_window = list(retained_kfs)
                        new_occ = {k: v for k, v in self.occ_aware_visibility.items() if k in retained_kfs}
                        self.occ_aware_visibility = new_occ
                        self.keyframe_optimizers = None
                        Log(f"✓ Retained {len(retained_kfs)} keyframes for next submap")

                        if self.empty_cache_on_submap_cut:
                            torch.cuda.empty_cache()

                        self.push_to_frontend("new_submap")
        # =======================================================
        # 【核心修复】：使用 get_nowait 替代阻塞的 get()
        # 防止在系统关闭的最后一刻，因为自己等自己而发生死锁
        # =======================================================
        while not self.backend_queue.empty():
            try:
                self.backend_queue.get_nowait()
            except:
                break

        while not self.frontend_queue.empty():
            try:
                self.frontend_queue.get_nowait()
            except:
                break

        return
