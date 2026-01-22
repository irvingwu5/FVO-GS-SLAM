import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping,get_loss_mapping_plane_constraint,get_depth_dist_loss,get_normal_consistency_loss,_save_normal_pair, _save_rendered_rgb

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

        self.use_normal = config["Training"]["sagsslam"]["use_normal"]
        self.use_dist = config["Training"]["sagsslam"]["use_dist"]
        self.use_plane_constraint = config["Training"]["sagsslam"]["use_plane_constraint"]

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"] # 首帧初始化时优化的迭代次数
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"] # 初始化期间执行高斯点分裂/修剪的间隔次数
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"] # 初始化期间重置不透明度的迭代步数
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"] # 初始化期间用于高斯密化（新增点）的梯度阈值
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        ) # 初始化期间场景范围180
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"] # 建图阶段每处理一个新关键帧时的优化迭代次数150
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"] # 建图期间执行高斯点分裂/修剪的间隔次数
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"] # 建图期间首次执行高斯更新的起始偏移量
        self.gaussian_th = self.config["Training"]["gaussian_th"] # 建图期间用于高斯密化（新增点）的梯度阈值
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        ) # 建图期间场景范围6.0
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

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
    '''
    ------------------初始建图模块(Initialization)------------------
    作用: 处理系统启动时的第一帧数据。
    逻辑:
    清空旧地图。
    基于第一帧生成的点云初始化高斯模型。
    执行高频的迭代优化（init_itr_num），快速建立初始场景几何，为前端跟踪提供基础。
    '''
    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
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
            #_save_normal_pair(render_pkg, "/home/wuxiangyu/Documents/PycharmProjects/SA-GS-SLAM/runtime_results/")
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            ) #0.4255
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg
    '''
    ------------------建图与联合优化模块(Mapping & Optimization)------------------
    后端最核心的功能，负责同时优化 3D 高斯场景参数和相机位姿
    作用: 对当前滑动窗口内的关键帧以及随机选取的历史关键帧进行迭代优化
    逻辑:
    数据采样: 选取当前窗口（current_window）内的关键帧，并随机采样之前的关键帧以防止灾难性遗忘。
    渲染与损失计算: 渲染选定视角的图像，计算渲染图与真实图像的损失（get_loss_mapping，包含 L1 loss 和 SSIM loss），以及各项同性正则化损失（Isotropic loss）。
    参数更新: 利用 PyTorch 的自动微分，同时更新 GaussianModel 参数（位置、颜色、协方差等）和 关键帧位姿参数（update_pose）。这里的位姿优化相当于后端 BA（Bundle Adjustment）。
    '''
    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for itr in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            keyframes_opt = []
            # 对多帧渲染计算联合损失，对当前滑动窗口内的关键帧进行优化
            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background,surf=True
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
                #_save_normal_pair(render_pkg, "/home/wuxiangyu/Documents/PycharmProjects/SA-GS-SLAM/runtime_results/", cam_idx)
                #_save_rendered_rgb(render_pkg, "/home/wuxiangyu/Documents/PycharmProjects/SA-GS-SLAM/runtime_results/", cam_idx)
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                ) #0.0202
                #-----------------------------------------------
                if self.use_normal:
                    rend_normal = render_pkg["rend_normal"]
                    surf_normal = render_pkg["surf_normal"]
                    normal_error = (1 - (rend_normal * (-surf_normal).detach()).sum(dim=0))[None]
                    normal_loss = self.config["opt_params"]["lambda_normal"] * normal_error.mean()
                    loss_mapping += normal_loss

                # 如果启用了 use_dist
                # 2DGS SLAM 对几何非常敏感，Distortion Loss 主要是为了压实表面。如果你的 Raw Loss 一直非常小（比如 1e-6 级别），
                # 说明你的高斯在光线方向上已经很“薄”了，甚至可能不需要太强的 Distortion Loss，此时该损失的意义主要在于防止未来产生新的浮空垃圾
                if self.use_dist:
                    # 你的 __init__.py 已经返回了 "rend_dist"
                    rend_dist = render_pkg["rend_dist"]
                    # 建议权重在 0.001 ~ 0.01 之间
                    loss_mapping += self.config["opt_params"][
                                        "lambda_dist"] * rend_dist.mean()  # 500*6.3175e-06=3.1587e-07
                #--------------------------------------------------
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            # 随机选取的历史关键帧进行迭代优化
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
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
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            # [修改点] 历史帧也要应用同样的 Warm-up 策略
            # if self.use_plane_constraint:
            #     lambda_plane = self.config["Training"]["sagsslam"]["weight"]
            #     loss_type = self.config["Training"]["sagsslam"]["loss_type"]
            #     warmup_iter = self.config["Training"]["sagsslam"]["warmup_iter"]
            #
            #     if self.iteration_count < warmup_iter:
            #         actual_lambda_plane = 0.0
            #     else:
            #         actual_lambda_plane = lambda_plane
            #
            #     if actual_lambda_plane > 0:
            #         loss_mapping += actual_lambda_plane * get_loss_mapping_plane_constraint(
            #             self.gaussians, viewpoint, loss_type
            #         )

            # if self.use_normal:
            #     rend_normal = render_pkg["rend_normal"]
            #     surf_normal = render_pkg["surf_normal"]
            #     normal_error =(1 - (rend_normal * (-surf_normal).detach()).sum(dim=0))[None]
            #     normal_loss = self.config["opt_params"]["lambda_normal"] * normal_error.mean()
            #     loss_mapping += normal_loss
            #
            # # 如果启用了 use_dist
            # #2DGS SLAM 对几何非常敏感，Distortion Loss 主要是为了压实表面。如果你的 Raw Loss 一直非常小（比如 1e-6 级别），
            # # 说明你的高斯在光线方向上已经很“薄”了，甚至可能不需要太强的 Distortion Loss，此时该损失的意义主要在于防止未来产生新的浮空垃圾
            # if self.use_dist:
            #     # 你的 __init__.py 已经返回了 "rend_dist"
            #     rend_dist = render_pkg["rend_dist"]
            #     # 建议权重在 0.001 ~ 0.01 之间
            #     loss_mapping += self.config["opt_params"]["lambda_dist"] * rend_dist.mean() #500*6.3175e-06=3.1587e-07

            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians高斯密度自适应控制模块
            # 该模块负责动态调整高斯球的数量和分布，以适应场景的几何细节
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                # 可见性修剪: 统计高斯球被观测的次数 (n_obs)。
                # 在 prune_mode="slam" 模式下，移除观测次数过少（n_obs <= 3）且不稳定的高斯球，去除噪声
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
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
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
                        # # make sure we don't split the gaussians, break here.
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

                ## Opacity reset
                # 不透明度重置: 定期调用 reset_opacity 或 reset_opacity_nonvisible，
                # 将高斯不透明度重置为低值。这有助于去除错误的“漂浮物”并重新收敛几何结构
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split
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
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background, surf=True
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
            #------------------------
            gt_depth = viewpoint_cam.gt_depth
            gt_depth_mask = gt_depth > 0.0
            Ll1_depth = l1_loss(depth * gt_depth_mask, gt_depth * gt_depth_mask)
            #------------------------
            # scaling = self.gaussians.get_scaling
            # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            #------------------------
            # loss = (1.0 - self.opt_params.lambda_dssim) * (
            #     Ll1
            # ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            #loss = Ll1 + 0.1 * isotropic_loss.mean() + 0.01 * Ll1_depth
            loss = Ll1 + 0.01 * Ll1_depth
            #------------------------
            if self.use_normal:
                rend_normal = render_pkg["rend_normal"]
                surf_normal = render_pkg["surf_normal"]
                normal_error =(1 - (rend_normal * (-surf_normal)).sum(dim=0))[None]
                normal_loss = 0.005 * normal_error.mean()
                loss += normal_loss
            #------------------------
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
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
            # keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
            keyframes.append((kf_idx, kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

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
                self.map(self.current_window) #执行建图 (map): 对当前窗口进行优化。
                #周期性同步: 当累计迭代次数达到阈值（last_sent >= 10），执行一次带修剪的建图优化，并将最新状态同步给前端 (push_to_frontend)。
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else: #响应前端的指令，指令处理 (非空时)
                data = self.backend_queue.get()
                if data[0] == "stop": #终止循环，退出进程
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
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    #扩展地图: 调用 add_next_kf (即 gaussians.extend_from_pcd_seq)，利用新关键帧的深度图在未知区域初始化新的高斯点。
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
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
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
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
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
