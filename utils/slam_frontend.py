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

        # ========== 新增：子图策略状态变量 ==========
        self.current_submap_id = 0
        self.submap_anchor_pose = None  # 记录当前子图第一帧的位姿 (World to Camera)
        self.submap_trans_thre = self.config["Submap"]["trans_thre"]
        self.submap_rot_thre = self.config["Submap"]["rot_thre"]
        self.frame_to_submap = {}  # <--- 新增这行：记录每帧属于哪个子图
        # ============================================
        self.fft_filter = None  # <--- 新增这行：频域滤波器实例
        # 【新增：读取消融实验开关，兼容旧版配置防止报错】
        self.use_submap = self.config.get("Ablation", {}).get("use_submap", True)
        # 【新增：消融实验开关】
        self.use_fft_mask = self.config.get("Ablation", {}).get("use_fft_mask", True)
        self.use_error_mask = self.config.get("Ablation", {}).get("use_error_mask", True)
        # 【新增】：读取位姿连续性检查参数
        self.pose_jump_check_enabled = self.config.get("Submap", {}).get("pose_jump_check_enabled", True)
        self.max_trans_jump = self.config.get("Submap", {}).get("max_trans_jump", 1.0)
        self.max_rot_jump = self.config.get("Submap", {}).get("max_rot_jump", 45.0)

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"] # 结果保存路径
        self.save_results = self.config["Results"]["save_results"] # 是否保存结果
        self.save_trj = self.config["Results"]["save_trj"] # 是否保存轨迹
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"] # 保存轨迹的关键帧间隔，表示每增加多少个关键帧就进行一次轨迹保存或 ATE 评估

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"] # 跟踪迭代次数，针对每一帧图像优化相机位姿时的梯度下降迭代轮数
        self.kf_interval = self.config["Training"]["kf_interval"] # 关键帧最小间隔防止过于频繁插入关键帧
        self.window_size = self.config["Training"]["window_size"] # 滑动窗口大小，限制同时优化的关键帧数量，当关键帧数量超过此值时，会根据重叠度等策略移除旧的关键帧
        self.single_thread = self.config["Training"]["single_thread"] # 如果为 True，前端在请求关键帧或初始化后会主动等待（sleep），直到后端处理完成，表现为串行执行；否则前端和后端并行工作。

    def compute_error_mask(self, render_pkg, viewpoint):
        """
        基于当前渲染结果与真实观测的差异，计算哪里需要补点 (Error Mask)
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
        viewpoint.T = viewpoint.T_gt

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
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames] # 上一帧位姿
        # 【新增】：检查位姿连续性
        # if not self.validate_pose_continuity(prev.T, prev.T):
        #     Log(f"[WARNING] 跳过第 {cur_frame_idx} 帧，因为检测到位姿跳变")
        #     return False  # 返回 False 表示 tracking 失败
        viewpoint.T = prev.T
        # 优化参数与优化器构建
        opt_params = []
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
                converged = update_pose(viewpoint) # update_pose 将增量应用到实际位姿，返回是否收敛（converged）
            # 每 10 次迭代把当前视点与 GT 图发到可视化队列（q_main2vis），用于前端显示
            if tracking_itr % 10 == 0:
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

    def validate_pose_continuity(self, prev_pose, curr_pose):
        """
        检查位姿是否发生跳变
        【作用】：防止子图切分时位姿跳变导致的 tracking 错误

        Args:
            prev_pose: 前一帧位姿 (4x4 变换矩阵)
            curr_pose: 当前帧位姿 (4x4 变换矩阵)

        Returns:
            True: 位姿连续，无跳变
            False: 检测到位姿跳变
        """
        if not self.pose_jump_check_enabled:
            return True

        try:
            # 计算平移差
            trans_diff = np.linalg.norm(curr_pose[:3, 3] - prev_pose[:3, 3])
            if trans_diff > self.max_trans_jump:
                Log(f"[WARNING] 检测到平移跳变: {trans_diff:.3f}m (阈值: {self.max_trans_jump}m)")
                return False

            # 计算旋转差（角度）
            R_diff = prev_pose[:3, :3].T @ curr_pose[:3, :3]
            # 使用迹计算旋转角度
            trace = np.clip(np.trace(R_diff), -1, 3)
            angle_diff = np.arccos((trace - 1) / 2) * 180 / np.pi

            if angle_diff > self.max_rot_jump:
                Log(f"[WARNING] 检测到旋转跳变: {angle_diff:.1f}° (阈值: {self.max_rot_jump}°)")
                return False

            return True

        except Exception as e:
            Log(f"[ERROR] 位姿连续性检查异常: {e}")
            return True  # 异常时默认认为连续
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
        N_dont_touch = 2 #这意味着最新的 2 个关键帧（当前帧和紧挨着的上一帧）被视为“活跃区”，受到保护，绝对不会被此算法移除。移除检查只针对窗口中索引为 2 及其之后的旧帧
        window = [cur_frame_idx] + window #将当前帧的索引 cur_frame_idx 插入到窗口列表的最前面
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        # 策略一：基于视觉重叠度剔除 (Visual Overlap Pruning)代码遍历保护区之外的所有旧关键帧，计算它们与当前帧的共视关系
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            # 使用了 Szymkiewicz–Simpson 系数（Overlap Coefficient）。它计算当前帧看到的点与旧帧看到的点的交集，除以两者中可见点数量较小的那个（min(当前, 旧)）。这比简单的 IoU 更能容忍视场大小差异。
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            # 如果某个旧关键帧与当前帧的视觉重叠度 point_ratio_2 低于阈值（cut_off，默认 0.4），说明该旧帧对当前视角的约束贡献很小
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx) # 该帧会被加入 to_remove 列表
        # 如果存在这样的帧，代码最后会移除其中一个（to_remove[-1]）
        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        # kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))
        kf_0_WC = torch.linalg.inv(curr_frame.T)
        # 策略二：基于几何分布剔除 (Geometric Pruning)如果经过策略一处理后，窗口大小仍然超过配置的限制（window_size），则强制基于空间几何关系移除一帧。
        # 解释：系统倾向于移除那些 “既离当前位置很远，又和其他旧帧挤在一起” 的关键帧。这样可以保留那些空间分布较均匀、或者离当前区域较近的关键帧。
        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = [] #打分逻辑：为每个候选旧帧计算一个分数 inv_dist。
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = kf_i.T
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(kf_j.T)
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item() # 距离项 (k): 计算该帧与当前帧（最新帧）的距离。该值越大，说明该帧离当前位置越远。
                inv_dist.append(k * sum(inv_dists)) # 密集度项 (sum(inv_dists)): 计算该帧与其他所有旧帧距离的倒数之和。该值越大，说明该帧周围越拥挤（冗余）

            idx = np.argmax(inv_dist) #决策：移除分数最高的帧。
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
        self.requested_init = True
    '''
    -------------------后端同步与通信模块---------------------
    作用: 保持前端与后端的地图数据一致。
    逻辑:
    发送请求: 通过 backend_queue 向后端发送 keyframe（插入关键帧）、init（初始化）、map（建图）等指令。
    接收更新: 通过 frontend_queue 接收后端优化好的全局高斯模型（gaussians）、关键帧位姿修正和可见性信息，并更新本地状态。
    '''
    def sync_backend(self, data): # 从后端接收更新的数据包，并同步前端的高斯模型、关键帧位姿和可见性信息
        self.gaussians = data[1] #更新全局高斯模型
        self.occ_aware_visibility = data[2] #更新可见性信息
        keyframes = data[3] #修正后的关键帧位姿列表，包含 (kf_id, kf_R, kf_T)

        for kf_id, kf_T in keyframes:
            self.cameras[kf_id].T = kf_T

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
                "cut_pose_c2w": current_c2w.cpu().numpy(),  # 切图时的位姿
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
                    # ========== 新增：退出前将从属关系存入硬盘 ==========
                    torch.save(self.frame_to_submap, os.path.join(self.save_dir, "frame_to_submap.pt"))
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

                # ------------------Tracking跟踪：估计当前帧的相机位姿（位置和旋转）---------------------
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                # ========== 新增：子图“切图”监控逻辑 ==========
                current_c2w = torch.linalg.inv(viewpoint.T)

                if self.submap_anchor_pose is None:
                    self.submap_anchor_pose = current_c2w.clone()

                if self.exceeds_motion_thresholds(current_c2w, self.submap_anchor_pose):
                    Log(f"==> 启动新子图 (ID: {self.current_submap_id + 1}) <==")

                    # 1. 仅发送切图信号给后端
                    self.backend_queue.put(["new_submap", self.current_submap_id])

                    # 2. 更新前端锚点，继续平滑 Tracking
                    self.current_submap_id += 1
                    self.submap_anchor_pose = current_c2w.clone()

                    # 【核心修复】：彻底删除坐标系强制转换！
                    # 保持全局坐标系连续 Tracking，不清理 current_window，
                    # 彻底消除轨迹指数级爆炸和失忆问题。

                    continue  # 跳过本次 is_keyframe 判断，平滑进入下一帧
                # ============================================

                # 窗口维护作用: 维护一个固定大小的滑动窗口（current_window），用于限制优化规模
                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0] #获取最近一个已添加的关键帧的索引，最新的关键帧总是被插入到列表的头部
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval #检查帧间隔约束，计算当前帧与上一个关键帧之间的帧数差，这是一种强制的时间/帧数限制，防止系统过于频繁地插入关键帧。
                '''
                -------------n_touched变量的含义和作用：
                #这个变量后续将用于计算当前帧与上一关键帧的共视程度（Overlap/Intersection over Union），这是判断是否需要插入新关键帧的核心依据之一（如果重叠度过低，说明到了新环境，通常需要插入关键帧）。
                # n_touched 其长度等于当前全局地图中所有高斯点（3D Gaussians）的总数量。它的值：表示每一个高斯点在当前这一帧图像的渲染过程中，投影并覆盖了多少个像素（或渲染图块 tiles）
                # 如果某个高斯点的 n_touched > 0：说明这个 3D 点在当前相机视角下是可见的（即它投影到了屏幕内，且参与了成像）
                # 如果某个高斯点的 n_touched == 0：说明这个点在当前视角下是不可见的（可能在相机背后、视野外，或者被深度剔除）
                -------------为什么要计算 curr_visibility？
                它是为了后续计算共视关系（Co-visibility / Overlap）（见第 464-470 行）：
                SLAM 前端需要判断当前帧和上一个关键帧之间有多少重叠。
                如果两帧看到的“可见高斯点集合”（即 curr_visibility）重合度很高，说明相机没怎么动。
                如果重合度变低（看到很多新点，旧点看不到了），说明相机移动到了新区域，系统就需要插入一个新的关键帧（Keyframe）
                '''
                curr_visibility = (render_pkg["n_touched"] > 0).long() #tensor(N,)
                # --------------------关键帧判断
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility, # dict{keyframe_idx: tensor(N,)}一个字典（dictionary），用于存储各个关键帧（Keyframe）对场景中 3D 高斯点的可见性掩码（visibility mask）
                )
                '''
                这段代码的作用是加速填充窗口。在窗口未满时，系统不像通常那样严格考量几何位移距离，而是主要依赖视觉变化率。
                只要画面变化足够大（重叠度低）且满足最小间隔，就立即插入关键帧，以便尽快建立起局部地图并填满优化窗口。
                '''
                if len(self.current_window) < self.window_size: #前端在滑动窗口未填满时的关键帧选择策略，这通常发生在系统刚启动、刚完成初始化或者重置之后。
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero() #计算当前帧看到的点与上一关键帧看到的点的并集数量
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero() #计算两帧共同看到的点的交集数量
                    point_ratio = intersection / union #计算交并比，这个值越接近 1，说明两帧画面基本一样，这个值越接近 0，说明当前帧看到了很多新区域，或者之前看到的区域看不到了
                    create_kf = (
                        check_time #必须满足最小帧间隔限制
                        and point_ratio < self.config["Training"]["kf_overlap"] #视觉重叠度必须低于阈值。这意味着视野发生了足够大的变化，需要新的关键帧来补充信息
                    ) #决定是否将当前帧选为关键帧
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
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
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
