import time

import numpy as np
import torch
import torch.multiprocessing as mp
import os
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
import cv2
from utils.fft_filter import FFTFrequencyFilter

class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None # 后端到前端的通信队列
        self.backend_queue = None # 前端到后端的通信队列
        self.q_main2vis = None # 前端到可视化的通信队列
        self.q_vis2main = None # 可视化到前端的通信队列

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"] # 是否为单目模式True or False
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.use_gui = config["Results"]["use_gui"]  # 新增
        # ========== 新增：子图策略状态变量 ==========
        self.current_submap_id = 0
        self.submap_anchor_poses = {0: np.eye(4)}  # 子图 0 的锚点就是全局原点
        self.cumulative_anchor_c2w = np.eye(4)  # 当前累积的全局锚点位姿??????????????????
        self.submap_trans_thre = self.config["Submap"]["trans_thre"]
        self.submap_rot_thre = self.config["Submap"]["rot_thre"]
        self.frame_to_submap = {}  # <--- 新增这行：记录每帧属于哪个子图
        self.is_first_frame_of_submap = False  # <--- 新增这行：标记当前帧是否为子图的第一帧
        self.true_independent_submap = self.config.get("Submap", {}).get("true_independent_submap", False)
        # ★★★ 新增这一行，修复 AttributeError ★★★
        self.submap_anchor_pose = None #运动监控锚点
        self.cut_refine_iters = self.config.get("Submap", {}).get("cut_refine_iters", 0)
        # ============================================
        # ============================================
        self.fft_filter = None  # <--- 新增这行：频域滤波器实例
        # 【新增：读取消融实验开关，兼容旧版配置防止报错】
        self.use_submap = self.config.get("Ablation", {}).get("use_submap", True)
        # 【新增：消融实验开关】
        self.use_fft_mask = self.config.get("Ablation", {}).get("use_fft_mask", True)
        self.use_error_mask = self.config.get("Ablation", {}).get("use_error_mask", True)


    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"] # 结果保存路径
        self.save_results = self.config["Results"]["save_results"] # 是否保存结果
        self.save_trj = self.config["Results"]["save_trj"] # 是否保存轨迹
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"] # 保存轨迹的关键帧间隔，表示每增加多少个关键帧就进行一次轨迹保存或 ATE 评估

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"] # 跟踪迭代次数，针对每一帧图像优化相机位姿时的梯度下降迭代轮数
        self.kf_interval = self.config["Training"]["kf_interval"] # 关键帧最小间隔防止过于频繁插入关键帧
        self.window_size = self.config["Training"]["window_size"] # 滑动窗口大小，限制同时优化的关键帧数量，当关键帧数量超过此值时，会根据重叠度等策略移除旧的关键帧
        self.single_thread = self.config["Training"]["single_thread"] # 如果为 True，前端在请求关键帧或初始化后会主动等待（sleep），直到后端处理完成，表现为串行执行；否则前端和后端并行工作。

    # 加一个“给显示/评估用的全局化相机拷贝”函数，不改原始 self.cameras，只给 GUI 用
    def _camera_to_global_copy(self, cam):
        cam_g = clone_obj(cam)

        sid = self.frame_to_submap.get(cam.uid, self.current_submap_id)
        anchor_c2w = self.submap_anchor_poses.get(sid, np.eye(4))

        if isinstance(anchor_c2w, torch.Tensor):
            anchor_c2w = anchor_c2w.cpu().numpy()
        anchor_c2w = np.array(anchor_c2w, dtype=np.float64)

        local_w2c = cam.T.detach().cpu().numpy()
        local_c2w = np.linalg.inv(local_w2c)
        global_c2w = anchor_c2w @ local_c2w
        global_w2c = np.linalg.inv(global_c2w)

        cam_g.T = torch.from_numpy(global_w2c).to(cam.T.device).type_as(cam.T)
        return cam_g

    def compute_error_mask(self, render_pkg, viewpoint):
        """
        基于当前渲染结果与真实观测的差异，计算哪里需要补点 (Error Mask)
        grad_mask 管"在哪里优化位姿"，freq_mask 管"新高斯点怎么撒、撒多大"，error_mask 管"在哪里补新高斯点"
        """
        gt_image = viewpoint.original_image.cuda()  # [3, H, W]
        render_image = render_pkg["render"].detach()  # [3, H, W]
        render_opacity = render_pkg["opacity"].detach()  # [1, H, W]

        # 1. Silhouette / Opacity 掩膜 (寻找地图没覆盖到的“漏洞”)
        # 如果某像素的不透明度小于 0.95，说明这里高斯覆盖不足，存在空洞
        silhouette_mask = (render_opacity < 0.95).squeeze(0)  # [H, W]

        # 2. RGB 光度误差掩膜 (寻找颜色重建错的地方)
        rgb_error = torch.abs(gt_image - render_image).sum(dim=0)  # [H, W]
        rgb_error_mask = rgb_error > 0.5  # 阈值 0.5 (可根据场景光照情况微调 0.3 ~ 0.8)

        # 3. Depth 深度误差掩膜 (如果有 GT 深度的话)
        # 如果是单目模式且没有提供先验深度，这部分可以忽略
        depth_error_mask = torch.zeros_like(silhouette_mask, dtype=torch.bool)
        if not self.monocular and viewpoint.depth is not None:
            gt_depth = torch.from_numpy(viewpoint.depth).cuda()  # [H, W]
            render_depth = render_pkg["depth"].detach().squeeze(0)  # [H, W]
            depth_error = torch.abs(gt_depth - render_depth)

            valid_depth = gt_depth > 0.01
            if valid_depth.any():
                median_error = depth_error[valid_depth].median()
                # 渲染深度比GT深度远(说明渲染在背景上了，前景缺东西) 且 误差显著大于中位数
                depth_error_mask = valid_depth & (render_depth > gt_depth) & (depth_error > 10.0 * median_error)

        # 综合掩膜：没覆盖的 | 颜色错的 | 深度错的
        # 只要满足其一，就说明这个像素点处“需要加高斯”
        error_mask = silhouette_mask | rgb_error_mask | depth_error_mask

        return error_mask
    '''
    -------------------新关键帧数据准备模块---------------------
    作用: 为新关键帧准备深度图和不透明度数据，用于后续后端初始化新的高斯点。
    逻辑: 如果是单目模式（monocular），会使用深度估计或中值深度填充无效区域；如果是 RGB-D 模式，则直接使用观测深度
    '''
    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        # =================================================================
        # 【修改：FFT 频率掩膜条件计算】
        # =================================================================
        if self.use_fft_mask:
            if self.fft_filter is None:
                self.fft_filter = FFTFrequencyFilter(gt_img.shape[1], gt_img.shape[2])

            img_np = (gt_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            viewpoint.freq_mask = self.fft_filter.generate_frequency_mask(img_bgr)
        # =================================================================
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None] #首先根据图像像素亮度（RGB和）判断哪些像素是有效的（valid_rgb），过滤掉过暗的区域（通常视为无效边界）。
        if self.monocular: #单目模式下需要估计深度，由于单目相机没有深度传感器，系统无法获得真实的深度信息，必须通过“猜测”或利用已有的地图来估计深度，以便在 3D 空间中初始化新的高斯点。
            if depth is None: #初始帧 / 无先验深度的情况，通过设定一个统一的深度值（默认为 2）加上随机噪声来初始化深度图，在没有任何地图信息时，假设场景是一个距离相机 2 米的平面，以此作为 SLAM 系统初始化的起点（冷启动）
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2]) #生成一张所有像素值为 2.0 的深度图
                initial_depth += torch.randn_like(initial_depth) * 0.3 #给这个常数深度加上标准差为 0.3 的随机噪声
            else: #后续关键帧的情况，如果传入了当前模型渲染出的深度和不透明度，算法会计算**中值深度（Median Depth）**和标准差
                depth = depth.detach().clone() #克隆并分离（detach）当前渲染出的 depth（深度图）和 opacity（不透明度图），避免影响计算图的梯度
                opacity = opacity.detach()
                use_inv_depth = False #决定是在原始深度空间还是逆深度空间（1/depth）进行统计和滤波
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0) #非单目模式下，直接使用观测到的深度图
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels，valid_rgb标记了 RGB 图像中亮度足够、被视为有效的像素区域
        return initial_depth[0].numpy() #~valid_rgb 取反表示“无效的区域”（例如图像边缘的黑色区域或过暗区域）。代码将这些无效区域对应的深度值强制设为 0。这意味着系统在建图时会忽略这些区域的深度信息

    '''
    -------------------初始化模块---------------------
    作用: 系统的冷启动。
    逻辑: 清空队列和状态，接收第一帧数据，调用 add_new_keyframe 生成初始深度，
    并向后端发送初始化请求（request_init），建立初始的地图
    '''
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose 将当前相机的位姿（R 和 T）强制更新为上述的真值
        #viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt) #系统在初始化第一帧时，并不进行位姿估计，而是直接读取数据集提供的真实位姿
        viewpoint.T = viewpoint.T_gt #把第一帧真实位姿直接赋值给当前帧的估计位姿，确保系统从正确的状态开始
        viewpoint.fixed_pose = True
        viewpoint.reset_pose_deltas()
        viewpoint.cam_rot_delta.requires_grad_(False)
        viewpoint.cam_trans_delta.requires_grad_(False)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True) #生成初始深度图（忽略无效区域）
        self.request_init(cur_frame_idx, viewpoint, depth_map) ## 请求后端进行地图/高斯点的初始化（冷启动），并标记前端正在等待初始化完成
        self.reset = False #重置标志设为 False，表示系统已完成初始化

    '''
    -------------------跟踪模块---------------------
    作用: 估计当前帧的相机位姿（位置和旋转）
    逻辑: 基于当前帧与上一关键帧的相对位移（kf_translation）、视锥体重叠度（kf_overlap）以及可见性掩码来综合判断
    '''
    def tracking(self, cur_frame_idx, viewpoint):
        # 如果是子图的第一帧
        if self.is_first_frame_of_submap:
            Log(f"[DEBUG] tracking() consumed first-frame flag at frame {cur_frame_idx}")

            eye4 = torch.eye(4, device=viewpoint.T.device, dtype=viewpoint.T.dtype)
            viewpoint.T = eye4

            viewpoint.fixed_pose = False
            viewpoint.is_submap_seed = True
            viewpoint.seed_pose_prior = eye4.clone()

            viewpoint.seed_prior_weight_trans = self.config.get("Submap", {}).get(
                "seed_prior_weight_trans", 0.10
            )
            viewpoint.seed_prior_weight_rot = self.config.get("Submap", {}).get(
                "seed_prior_weight_rot", 0.05
            )

            viewpoint.reset_pose_deltas()
            viewpoint.cam_rot_delta.requires_grad_(True)
            viewpoint.cam_trans_delta.requires_grad_(True)

            self.is_first_frame_of_submap = False
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.T = prev.T.clone()
            viewpoint.fixed_pose = False

            # 普通帧不是 seed
            viewpoint.is_submap_seed = False
            viewpoint.seed_pose_prior = None
            viewpoint.seed_prior_weight_trans = 0.0
            viewpoint.seed_prior_weight_rot = 0.0

            viewpoint.cam_rot_delta.requires_grad_(True)
            viewpoint.cam_trans_delta.requires_grad_(True)
        # 优化参数与优化器构建
        opt_params = []
        if not viewpoint.fixed_pose:
            opt_params.append(
                {
                    "params": [viewpoint.cam_rot_delta],
                    "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                    "name": "rot_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.cam_trans_delta],
                    "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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
        # 迭代渲染与损失最小化（tracking loop）
        # 将当前地图渲染到该帧，计算渲染结果与观测图像之间的差异（损失函数），
        # 并通过梯度下降优化相机位姿参数以最小化该损失，当前帧位姿由上一帧位姿初始化到适应当前观测的位姿
        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step() # 执行 pose_optimizer.step() 更新参数
                if viewpoint.fixed_pose:
                    viewpoint.reset_pose_deltas()
                    converged = True
                else:
                    converged = update_pose(viewpoint) # update_pose 将增量应用到实际位姿，返回是否收敛（converged）
            # 每 10 次迭代把当前视点与 GT 图发到可视化队列（q_main2vis），用于前端显示
            if self.use_gui and tracking_itr % 10 == 0:
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged: #若 update_pose 表明已收敛，则提前退出迭代循环
                break
        #用最后一次渲染的 depth/opacity 计算并保存 self.median_depth（供关键帧判断等使用）
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg #返回最后一次的 render_pkg


    def refine_cut_frame_pose(self, viewpoint, extra_iters=30, lr_scale=0.25):
        """
        在真正切图前，对 cut frame 再额外做一小段 pose refine，
        减小 handoff 噪声被直接写进 relative_pose / anchor 链。
        """
        old_fixed = viewpoint.fixed_pose
        viewpoint.fixed_pose = False
        viewpoint.cam_rot_delta.requires_grad_(True)
        viewpoint.cam_trans_delta.requires_grad_(True)

        opt_params = [
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"] * lr_scale,
                "name": f"cut_refine_rot_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"] * lr_scale,
                "name": f"cut_refine_trans_{viewpoint.uid}",
            },
        ]
        pose_optimizer = torch.optim.Adam(opt_params)

        for _ in range(extra_iters):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )

            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
                if converged:
                    break

        viewpoint.fixed_pose = old_fixed

    '''
    -------------------关键帧判断---------------------
    作用: 判断当前帧是否提供了足够的新信息（或跟踪是否变得不稳定），从而需要将其作为新的关键帧。
    逻辑: 基于当前帧与上一关键帧的相对位移（kf_translation）、视锥体重叠度（kf_overlap）以及可见性掩码来综合判断
    当视野变化大（看到新场景）或者相机物理移动距离大时，系统都会判定需要插入新的关键帧，以便后端更新和扩展地图
    '''
    def is_keyframe( #判断当前帧是否应该被选为新的关键帧（Keyframe）。这是SLAM前端非常关键的一步，决定了何时向地图中插入新信息
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,## dict{keyframe_idx: tensor(N,)}一个字典（dictionary），用于存储各个关键帧（Keyframe）对场景中 3D 高斯点的可见性掩码（visibility mask）
    ):  #准备工作：获取配置和相机姿态
        kf_translation = self.config["Training"]["kf_translation"] # 较大位移阈值
        kf_min_translation = self.config["Training"]["kf_min_translation"] # 较小位移阈值
        kf_overlap = self.config["Training"]["kf_overlap"] # 共视重叠度阈值

        curr_frame = self.cameras[cur_frame_idx] # 当前帧对象
        last_kf = self.cameras[last_keyframe_idx] # 上一个关键帧对象

        # ====================================================================
        # 【新增防御性逻辑】：防止异步通信导致的状态覆写引发 KeyError
        # 如果发现当前的基准关键帧丢失，说明前端刚刚受到了旧消息的干扰。
        # 此时直接返回 True 强制生成一个新关键帧。这会迫使后端重新运行 map()，
        # 并下发一份绝对正确、新鲜的 occ_aware_visibility 字典，实现系统“自愈”。
        if last_keyframe_idx not in occ_aware_visibility:
            Log(f"[Warning] Keyframe {last_keyframe_idx} lost in visibility dict! Forcing a new keyframe to heal state.")
            return True
        # ====================================================================

        #计算相对位移（几何距离判断）这部分计算当前帧相对于上一关键帧移动了多少距离
        pose_CW = curr_frame.T
        last_kf_CW = last_kf.T
        last_kf_WC = torch.linalg.inv(last_kf_CW) # 上一关键帧的相机到世界变换矩阵 (Camera to World)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3]) # 计算相对变换矩阵：pose_CW @ last_kf_WC 提取位移向量的模长（欧氏距离）
        dist_check = dist > kf_translation * self.median_depth #判断 1：距离是否超过了一个较大的阈值 (kf_translation * main_depth)如果超过这个距离，说明相机移动很大，应该强制插入关键帧。
        dist_check2 = dist > kf_min_translation * self.median_depth #判断 2：距离是否超过了一个较小的阈值 (kf_min_translation * main_depth)
        # 这是一个辅助条件，结合共视度使用。如果不满足这个最小移动距离，哪怕共视度很低也不会由共视原因触发关键帧（防抖动）。
        #计算共视重叠度（视觉判断）这部分计算当前帧看到的场景内容（高斯点）和上一关键帧看到的有多少重叠。使用 IoU (Intersection over Union) 的概念。
        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero() #计算并集：当前帧看到的点 OR 上一帧看到的点
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero() #计算交集：当前帧看到的点 AND 上一帧看到的点
        point_ratio_2 = intersection / union # 计算重叠比率 (Intersection over Union)如果比率很低，说明当前帧看到了很多新东西，或者旧东西看不到了。
        '''
        返回 True 的条件（即选为关键帧）是以下两者之一：  
        1.视觉变化显著且有一定位移 (point_ratio_2 < kf_overlap and dist_check2)：  
        当前的视野内容与上一帧重叠度低（point_ratio_2 小于阈值），说明环境变了。
        并且，相机发生了至少一小段位移 (dist_check2)。防止因旋转或噪声导致的虚假低重叠。
        2.几何位移显著 (dist_check)：  
        无论视觉内容如何，只要相机移动的距离超过了较大的阈值 (kf_translation)，就强制插入关键帧，以保证轨迹跟踪的鲁棒性。
        '''
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    '''
    -------------------窗口维护---------------------
    作用: 维护一个固定大小的滑动窗口（current_window），用于限制优化规模。
    逻辑: 将新关键帧加入窗口。当窗口已满时，它会基于重叠度（Szymkiewicz–Simpson coefficient）
    和共视关系移除冗余的关键帧（通常是重叠度低或空间分布上对当前帧贡献最小的帧），即使在单目模式下，如果初始化失败也会触发重置
    '''

    def add_to_window(
            self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None

        # 策略一：基于视觉重叠度剔除
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]

            # 防御：如果旧关键帧 visibility 丢了，直接标记移除
            if kf_idx not in occ_aware_visibility:
                Log(
                    f"[Warning] Window keyframe {kf_idx} missing in visibility dict, "
                    f"mark it for removal."
                )
                to_remove.append(kf_idx)
                continue

            kf_visibility = occ_aware_visibility[kf_idx]

            intersection = torch.logical_and(
                cur_frame_visibility_filter, kf_visibility
            ).count_nonzero()

            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                kf_visibility.count_nonzero(),
            )

            if denom == 0:
                point_ratio_2 = torch.tensor(0.0, device=cur_frame_visibility_filter.device)
            else:
                point_ratio_2 = intersection.float() / denom.float()

            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )

            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        for kf_idx in to_remove:
            if kf_idx in window:
                window.remove(kf_idx)

        # 策略二：窗口过大时，按距离启发式移除
        if len(window) > self.window_size:
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = kf_i.T
                kf_i_WC = torch.linalg.inv(kf_i_CW)

                dists = []
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_CW = kf_j.T
                    kf_j_WC = torch.linalg.inv(kf_j_CW)
                    dist = torch.norm((kf_i_CW @ kf_j_WC)[0:3, 3])
                    dists.append(dist)

                kf_0_WC = torch.linalg.inv(curr_frame.T)
                kf_i_dist_to_0 = torch.norm((kf_i_CW @ kf_0_WC)[0:3, 3])

                inv_dist.append(1.0 / (sum(dists) + 1e-6) + kf_i_dist_to_0 * 0.0)

            idx = torch.argmax(torch.tensor(inv_dist)).item()
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
    # 用途是请求后端把当前帧作为新的关键帧插入地图（并提供位姿、窗口和深度信息以供初始化/优化）。
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap] # 向后端发送关键帧请求
        self.backend_queue.put(msg)
        self.requested_keyframe += 1
    # 用途是触发后端的建图/地图更新流程，通常在关键帧插入后调用。
    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)
    # 请求后端进行地图/高斯点的初始化（冷启动），并标记前端正在等待初始化完成
    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)

        # B1: seed 的局部原点重置已经在切图处完成；
        # 这里不再二次覆盖，只做状态标记。
        self.requested_init = True
    '''
    -------------------后端同步与通信模块---------------------
    作用: 保持前端与后端的地图数据一致。
    逻辑:
    发送请求: 通过 backend_queue 向后端发送 keyframe（插入关键帧）、init（初始化）、map（建图）等指令。
    接收更新: 通过 frontend_queue 接收后端优化好的全局高斯模型（gaussians）、关键帧位姿修正和可见性信息，并更新本地状态。
    '''

    def sync_backend(self, data):
        self.gaussians = data[1]
        backend_occ = data[2] if data[2] is not None else {}
        keyframes = data[3]

        if not isinstance(self.occ_aware_visibility, dict):
            self.occ_aware_visibility = {}

        if isinstance(backend_occ, dict) and len(backend_occ) > 0:
            self.occ_aware_visibility.update(backend_occ)

        # 先无条件更新位姿
        for kf_id, kf_T in keyframes:
            self.cameras[kf_id].T = kf_T

        # 只有 visibility 相关判断，才依赖 occ_aware_visibility

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 5 == 0: #10->5
            torch.cuda.empty_cache()

    def exceeds_motion_thresholds(self, current_c2w, anchor_c2w):
        """
        判断是否需要切分新子图。
        关键：保存切图时的位姿，确保新子图的位姿连续性。
        """
        if not self.use_submap:
            return False

        if anchor_c2w is None:
            return False

        # 计算相对变换矩阵
        delta_T = torch.linalg.inv(anchor_c2w) @ current_c2w

        # 1. 计算平移距离
        translation = torch.norm(delta_T[0:3, 3]).item()

        # 2. 计算旋转角度
        R = delta_T[0:3, 0:3]
        trace = torch.trace(R)
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta).item()
        angle_deg = angle_rad * 180.0 / np.pi

        if translation > self.submap_trans_thre or angle_deg > self.submap_rot_thre:
            Log(f"[Submap Cut] Translation: {translation:.3f}m (threshold: {self.submap_trans_thre}m), "
                f"Rotation: {angle_deg:.1f}° (threshold: {self.submap_rot_thre}°)")

            # 【新增】：保存切图时的位姿信息
            submap_cut_info = {
                "submap_id": self.current_submap_id,
                "cut_frame_id": len(self.cameras),  # 当前帧 ID
                "cut_pose_c2w": current_c2w.cpu().numpy(),  # 当前子图最后一帧的估计位姿
                "translation": translation,
                "rotation_deg": angle_deg
            }

            # 保存到文件供后续参考
            cut_info_path = os.path.join(
                self.config["Results"]["save_dir"],
                f"submap_cut_{self.current_submap_id:06d}.npy"
            )
            np.save(cut_info_path, submap_cut_info)

            return True

        return False

    '''
    -------------------前端 SLAM 主循环---------------------
    作用: 前端 SLAM 系统的主循环，负责处理数据流、执行跟踪和关键帧管理，并与后端进行通信。
    读取数据集（Camera.init_from_dataset）。
    处理 GUI 的暂停/恢复指令
    按顺序执行：初始化判断 -> 跟踪 (tracking) -> 关键帧判断 (is_keyframe) -> 窗口更新 (add_to_window) -> 向后端发送数据
    '''
    def run(self): # 前端slam主循环
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1) #由相机内参生成投影矩阵
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        #-------------进入前端主循环-------------
        while True:
            if self.q_vis2main.empty(): # 检查来自可视化模块的队列是否为空，以处理GUI中用户的暂停/恢复指令
                if self.pause:
                    continue #当系统处于暂停状态（self.pause 为 True）且没有收到新的可视化指令时，持续在循环头部空转等待，直到暂停解除或收到新指令
            else: #有来自可视化模块的消息指令
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])
            # ==================================================================
            if self.frontend_queue.empty(): # 前端队列为空（第一次执行为空），执行前端 SLAM 主循环的核心逻辑
                tic.record()
                if cur_frame_idx >= len(self.dataset): #所有帧处理完毕，保存结果并退出循环
                    # ========== 新增：退出前将从属关系和锚点位姿存入硬盘 ==========
                    torch.save(self.frame_to_submap, os.path.join(self.save_dir, "frame_to_submap.pt"))
                    torch.save(self.submap_anchor_poses, os.path.join(self.save_dir, "submap_anchor_poses.pt"))
                    # ====================================================
                    break
                # 当调用前端initialize函数时，会将 self.requested_init 设为 True，并向后端发送初始化请求。只有当后端完成初始化并通过队列发回 init 消息时（第 534-536 行），前端才会将此标志重置为 False
                if self.requested_init:
                    time.sleep(0.01)
                    continue #跳过后续的所有逻辑（如读取新帧、Tracking），直接回到 while True 开头。

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue
                # ---------------读取当前帧的具体数据---------------------
                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config) # 计算rgb l1损失时使用的梯度掩码,提取图像中纹理丰富或边缘明显的区域，忽略平坦区域，提高tracking计算的效率和稳定性
                # 读取当前帧数据后，将其存储在前端属性 self.cameras 字典中，以便后续处理和访问
                self.cameras[cur_frame_idx] = viewpoint #将当前正在处理的这一帧（由索引 cur_frame_idx 标识）的相机对象（viewpoint）保存到前端类的成员变量 self.cameras 字典中
                # ========== 新增：记录当前帧属于哪个子图 ==========
                self.frame_to_submap[cur_frame_idx] = self.current_submap_id
                # ==================================================
                # 系统初始化
                if self.reset: #系统重置标志为 True，说明需要重新初始化 SLAM 系统
                    self.initialize(cur_frame_idx, viewpoint) #系统的冷启动，前端将第一帧关键帧信息发送给后端建立初始地图，重置标志为 False表明初始化已完成
                    self.current_window.append(cur_frame_idx) #将当前帧作为第一个关键帧加入滑动窗口（current_window为list）
                    # =========================================================
                    # 📍 前端显存探针：放在当前帧处理完，马上要进入下一帧之前
                    # =========================================================
                    # print(f"[FrontEnd] 帧 {cur_frame_idx} 处理完毕 | "
                    #       f"分配显存: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB, "
                    #       f"保留显存: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
                    cur_frame_idx += 1 # 直接进入下一帧
                    continue #跳过后续的所有逻辑（如 Tracking），直接回到 while True 开头。

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )
                # 条件断点测试方法 1：在断点附近加一行代码
                # self.backend_queue.put(["pause"])  # 在断点前暂停后端
                # ------------------Tracking跟踪：估计当前帧的相机位姿（位置和旋转）---------------------
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                # ========== 新增：子图“切图”监控逻辑 ==========
                current_c2w = torch.linalg.inv(viewpoint.T)

                if self.submap_anchor_pose is None: #运动监控锚点，用来判断是否切图，初始为 None，第一帧处理时会被设置为当前帧的估计位姿
                    self.submap_anchor_pose = current_c2w.clone()

                # ========== 子图切换逻辑 ==========
                if self.exceeds_motion_thresholds(current_c2w, self.submap_anchor_pose):
                    Log(f"==> 启动新子图 (ID: {self.current_submap_id + 1}) <==")

                    if self.true_independent_submap:
                        cut_refine_iters = self.config.get("Submap", {}).get("cut_refine_iters", 20)
                        if cut_refine_iters > 0:
                            self.refine_cut_frame_pose(viewpoint, extra_iters=cut_refine_iters, lr_scale=0.25)
                            current_c2w = torch.linalg.inv(viewpoint.T)
                        # ====================================================
                        # 【真正独立子图模式】：种子帧初始化
                        # ====================================================
                        # 计算当前子图相对于上一个子图的相对位姿
                        # current_c2w 是新子图的起点在旧子图坐标系下的位姿，上一个子图最后一帧的估计位姿（局部位姿还是全局位姿？子图0没区别，但其他子图就有区别了，后面debug注意这里）
                        # 当前帧在“旧子图局部坐标系”下的 c2w
                        relative_pose = current_c2w.detach().cpu().numpy()
                        # ★★★ 新增：更新全局锚点位姿链 ★★★
                        # 第1个子图的全局锚点(第0子图最后一帧全局pose)=单位阵*第0个子图最后一帧全局pose
                        # 第2个子图的全局锚点(第1子图最后一帧全局pose)=第1个子图的全局锚点*第2个子图最后一帧的局部位姿？还是全局位姿？
                        # 更稳的 anchor 递推方式：由上一子图 anchor 显式计算
                        prev_anchor = np.array(
                            self.submap_anchor_poses[self.current_submap_id],
                            dtype=np.float64
                        )
                        new_submap_id = self.current_submap_id + 1
                        new_anchor = prev_anchor @ relative_pose

                        self.submap_anchor_poses[new_submap_id] = new_anchor
                        self.cumulative_anchor_c2w = new_anchor.copy()

                        # 1) 通知后端冻结旧子图
                        self.backend_queue.put(["new_submap", self.current_submap_id, relative_pose])

                        # 2) 前端切到新子图
                        self.current_submap_id = new_submap_id

                        # !!! 关键修复：当前帧已经不再属于旧子图，而是新子图 seed
                        self.frame_to_submap[cur_frame_idx] = self.current_submap_id
                        Log(
                            f"[DEBUG] cut frame {cur_frame_idx} reassigned to submap "
                            f"{self.frame_to_submap[cur_frame_idx]}"
                        )
                        # 3) 清空当前子图状态
                        self.current_window = []
                        self.occ_aware_visibility = {}
                        self.initialized = False

                        for cam in self.cameras.values():
                            if cam.cam_rot_delta is not None:
                                cam.cam_rot_delta.data.fill_(0)
                            if cam.cam_trans_delta is not None:
                                cam.cam_trans_delta.data.fill_(0)

                        # 4) 当前 cut frame 作为新子图 seed：局部原点 + 软锚定，不再硬锁死
                        eye4 = torch.eye(4, device=viewpoint.T.device, dtype=viewpoint.T.dtype)
                        viewpoint.T = eye4

                        viewpoint.fixed_pose = False
                        viewpoint.is_submap_seed = True
                        viewpoint.seed_pose_prior = eye4.clone()

                        viewpoint.seed_prior_weight_trans = self.config.get("Submap", {}).get(
                            "seed_prior_weight_trans", 0.10
                        )
                        viewpoint.seed_prior_weight_rot = self.config.get("Submap", {}).get(
                            "seed_prior_weight_rot", 0.05
                        )

                        viewpoint.reset_pose_deltas()
                        viewpoint.cam_rot_delta.requires_grad_(True)
                        viewpoint.cam_trans_delta.requires_grad_(True)

                        # 新子图内部的运动监控锚点：局部原点
                        self.submap_anchor_pose = eye4.clone()

                        # !!! 关键修复：当前帧已经是 seed，不要让下一帧再被重复 seed
                        self.is_first_frame_of_submap = False

                        # 5) 用当前帧初始化新子图
                        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)

                        # 6) 发送 init 请求
                        self.request_init(cur_frame_idx, viewpoint, depth_map)

                        # 7) 当前帧加入新窗口
                        self.current_window.append(cur_frame_idx)

                        cur_frame_idx += 1
                        continue

                    else:
                        # ====================================================
                        # 【流式子图模式】：保留部分旧点（原有逻辑）
                        # ====================================================
                        self.backend_queue.put(["new_submap", self.current_submap_id])
                        self.current_submap_id += 1
                        self.submap_anchor_pose = current_c2w.clone()
                        continue
                # ============================================

                # 窗口维护作用: 维护一个固定大小的滑动窗口（current_window），用于限制优化规模
                if self.use_gui:
                    current_window_dict = {}
                    current_window_dict[self.current_window[0]] = self.current_window[1:]

                    vis_current = self._camera_to_global_copy(viewpoint)
                    vis_keyframes = [
                        self._camera_to_global_copy(self.cameras[kf_idx])
                        for kf_idx in self.current_window
                    ]

                    self.q_main2vis.put(
                        gui_utils.GaussianPacket(
                            gaussians=clone_obj(self.gaussians),
                            current_frame=vis_current,
                            keyframes=vis_keyframes,
                            kf_window=current_window_dict,
                        )
                    )
                # 当已有未完成的关键帧请求时，前端不对当前帧做关键帧插入等进一步操作，而是清理当前帧并继续处理下一帧，避免并发冲突或重复请求。
                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()

                # 如果上一关键帧的 visibility 还没同步回来，先走保守策略
                if last_keyframe_idx not in self.occ_aware_visibility:
                    Log(f"[Warning] Keyframe {last_keyframe_idx} visibility missing, skipping point_ratio check.")
                    create_kf = check_time
                else:
                    create_kf = self.is_keyframe(
                        cur_frame_idx,
                        last_keyframe_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                    )
                '''
                这段代码的作用是加速填充窗口。在窗口未满时，系统不像通常那样严格考量几何位移距离，而是主要依赖视觉变化率。
                只要画面变化足够大（重叠度低）且满足最小间隔，就立即插入关键帧，以便尽快建立起局部地图并填满优化窗口。
                '''
                if len(self.current_window) < self.window_size:
                    # 只有当上一关键帧的 visibility 真实存在时，才允许做 point_ratio 计算
                    if last_keyframe_idx in self.occ_aware_visibility:
                        last_visibility = self.occ_aware_visibility[last_keyframe_idx]

                        union = torch.logical_or(
                            curr_visibility, last_visibility
                        ).count_nonzero()

                        intersection = torch.logical_and(
                            curr_visibility, last_visibility
                        ).count_nonzero()

                        if union > 0:
                            point_ratio = intersection.float() / union.float()
                        else:
                            point_ratio = torch.tensor(0.0, device=curr_visibility.device)

                        create_kf = (
                                check_time
                                and point_ratio < self.config["Training"]["kf_overlap"]
                        )
                    else:
                        # is_keyframe() 已经判定这里状态不一致，需要“自愈”
                        # 这里不能再访问缺失的 dict 项，直接强制关键帧
                        Log(
                            f"[Warning] Skip point_ratio check because keyframe "
                            f"{last_keyframe_idx} is missing in visibility dict."
                        )
                        create_kf = check_time
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf: # 当前帧是关键帧
                    # 将当前关键帧加入滑动窗口
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    # 【新增】：计算渲染误差掩膜，告诉后端“哪里破了”【修改：条件计算渲染误差掩膜】
                    # =========================================================
                    if self.use_error_mask:
                        viewpoint.error_mask = self.compute_error_mask(render_pkg, viewpoint)
                    # =========================================================
                    #为新关键帧准备深度图和不透明度数据，用于后续后端初始化新的高斯点。
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    # 向后端发送关键帧请求，请求后端扩展地图并进行精细优化。
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else: #当前帧不是关键帧
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1 # 直接进入下一帧

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf #当前帧被判定为关键帧
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0 #关键帧数满足保存/评估的间隔
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras, #传入全部帧的相机位姿数据（包括当前帧和之前的所有帧）
                        self.kf_indices, #传入当前已经选为关键帧的帧索引列表（kf_indices），后端优化完成后会更新这个列表
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                        frame_to_submap=self.frame_to_submap if self.true_independent_submap else None,
                        submap_anchor_poses=self.submap_anchor_poses if self.true_independent_submap else None,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            #==================================================================
            else: # 前端队列不为空，处理来自后端的同步数据，第二次及以后执行前端队列不为空
                data = self.frontend_queue.get() # 获取后端发来的第一帧以及后续帧的数据 data list(str:'init', GaussiansModel:, dict:, list:)
                if data[0] == "sync_backend":
                    # 如果系统刚刚请求了新子图初始化，还未完成时，忽略普通的同步信息
                    if not self.requested_init:
                        self.sync_backend(data) #前端同步来自后端的最新高斯模型、可见性信息、关键帧最新位姿

                elif data[0] == "keyframe":
                    # 只有当我们确实在等待 keyframe 返回时才处理它，
                    # 否则这就是一条来自旧子图的“幽灵消息”，直接丢弃。
                    if self.requested_keyframe > 0:
                        self.sync_backend(data)
                        self.requested_keyframe -= 1
                    else:
                        Log("[Frontend] 拦截到旧子图的幽灵 keyframe 消息，已安全丢弃。")

                elif data[0] == "init": #前端给后端第一个关键帧信息，后端初始化完成回传数据
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
