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
    # ========================================================================
    # 1. Initialization
    # ========================================================================
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
        # ========== 子图策略状态变量 ==========
        self.current_submap_id = 0
        self.submap_anchor_poses = {0: np.eye(4)}  # 子图 0 的锚点就是全局原点
        self.submap_trans_thre = self.config["Submap"]["trans_thre"]
        self.submap_rot_thre = self.config["Submap"]["rot_thre"]
        self.frame_to_submap = {}  # <--- 记录每帧属于哪个子图
        # ========== LoopSplat-style submap handoff ==========
        # 新子图 seed 帧不再重置到单位阵，而是继承旧子图 tracking 后的全局估计位姿。
        self.use_global_seed_submap = self.config.get("Submap", {}).get("use_global_seed_submap", True)
        # 每个子图 seed 的全局 c2w，用于保存子图间 transition 和后续 PGO。
        self.submap_seed_global_c2w = {0: np.eye(4, dtype=np.float64)}
        self.last_submap_seed_global_c2w = None
        # ================================================
        self.is_first_frame_of_submap = False  # 标记当前帧是否为子图的第一帧
        self.submap_anchor_pose = None #运动监控锚点
        self.cut_refine_iters = self.config.get("Submap", {}).get("cut_refine_iters", 0)
        self.submap_start_frame_idx = 0
        self.fft_filter = None  # 频域滤波器实例
        # 消融实验开关
        self.use_submap = self.config.get("Ablation", {}).get("use_submap", True)
        self.use_fft_mask = self.config.get("Ablation", {}).get("use_fft_mask", True)
        self.use_error_mask = self.config.get("Ablation", {}).get("use_error_mask", True)

    # ========================================================================
    # 2. Hyperparameters
    # ========================================================================
    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"] # 结果保存路径
        self.save_results = self.config["Results"]["save_results"] # 是否保存结果
        self.save_trj = self.config["Results"]["save_trj"] # 是否保存轨迹
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"] # 保存轨迹的关键帧间隔，表示每增加多少个关键帧就进行一次轨迹保存或 ATE 评估

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"] # 跟踪迭代次数，针对每一帧图像优化相机位姿时的梯度下降迭代轮数
        self.kf_interval = self.config["Training"]["kf_interval"] # 关键帧最小间隔防止过于频繁插入关键帧
        self.window_size = self.config["Training"]["window_size"] # 滑动窗口大小，限制同时优化的关键帧数量，当关键帧数量超过此值时，会根据重叠度等策略移除旧的关键帧
        self.single_thread = self.config["Training"]["single_thread"] # 如果为 True，前端在请求关键帧或初始化后会主动等待（sleep），直到后端处理完成，表现为串行执行；否则前端和后端并行工作。

    # ========================================================================
    # 3. System Initialization (cold start)
    # ========================================================================
    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        self.submap_start_frame_idx = cur_frame_idx
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose 将当前相机的位姿（R 和 T）强制更新为上述的真值
        viewpoint.T = viewpoint.T_gt
        viewpoint.fixed_pose = True
        viewpoint.reset_pose_deltas()
        viewpoint.cam_rot_delta.requires_grad_(False)
        viewpoint.cam_trans_delta.requires_grad_(False)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        # =================================================================
        # FFT 频率掩膜条件计算
        # =================================================================
        if self.use_fft_mask:
            if self.fft_filter is None:
                self.fft_filter = FFTFrequencyFilter(gt_img.shape[1], gt_img.shape[2])

            img_np = (gt_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            viewpoint.freq_mask = self.fft_filter.generate_frequency_mask(img_bgr)
        # =================================================================
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
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
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    # ========================================================================
    # 4. Error Mask (rendering-guided densification hint)
    # ========================================================================
    def compute_error_mask(self, render_pkg, viewpoint):
        """
        基于当前渲染结果与真实观测的差异，计算哪里需要补点 (Error Mask)
        grad_mask 管"在哪里优化位姿"，freq_mask 管"新高斯点怎么撒、撒多大"，error_mask 管"在哪里补新高斯点"
        """
        gt_image = viewpoint.original_image.cuda()  # [3, H, W]
        render_image = render_pkg["render"].detach()  # [3, H, W]
        render_opacity = render_pkg["opacity"].detach()  # [1, H, W]

        # 1. Silhouette / Opacity 掩膜 (寻找地图没覆盖到的"漏洞")
        silhouette_mask = (render_opacity < 0.95).squeeze(0)  # [H, W]

        # 2. RGB 光度误差掩膜 (寻找颜色重建错的地方)
        rgb_error = torch.abs(gt_image - render_image).sum(dim=0)  # [H, W]
        rgb_error_mask = rgb_error > 0.5

        # 3. Depth 深度误差掩膜 (如果有 GT 深度的话)
        depth_error_mask = torch.zeros_like(silhouette_mask, dtype=torch.bool)
        if not self.monocular and viewpoint.depth is not None:
            gt_depth = torch.from_numpy(viewpoint.depth).cuda()  # [H, W]
            render_depth = render_pkg["depth"].detach().squeeze(0)  # [H, W]
            depth_error = torch.abs(gt_depth - render_depth)

            valid_depth = gt_depth > 0.01
            if valid_depth.any():
                median_error = depth_error[valid_depth].median()
                depth_error_mask = valid_depth & (render_depth > gt_depth) & (depth_error > 10.0 * median_error)

        # 综合掩膜：没覆盖的 | 颜色错的 | 深度错的
        error_mask = silhouette_mask | rgb_error_mask | depth_error_mask

        return error_mask

    # ========================================================================
    # 5. Tracking (per-frame pose estimation)
    # ========================================================================
    def tracking(self, cur_frame_idx, viewpoint):
        # 子图第一帧已经在旧子图上完成 tracking，并在 perform_submap_cut()
        # 中作为 seed 送给后端初始化；这里绝不能再把位姿重置成单位阵。
        if self.is_first_frame_of_submap:
            Log(f"[DEBUG] tracking() consumed first-frame flag at frame {cur_frame_idx}")

            viewpoint.fixed_pose = True
            viewpoint.reset_pose_deltas()
            viewpoint.cam_rot_delta.requires_grad_(False)
            viewpoint.cam_trans_delta.requires_grad_(False)

            self.is_first_frame_of_submap = False
        else:
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.T = prev.T.clone()
            viewpoint.fixed_pose = False
            viewpoint.cam_rot_delta.requires_grad_(True)
            viewpoint.cam_trans_delta.requires_grad_(True)

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
                pose_optimizer.step()
                if viewpoint.fixed_pose:
                    viewpoint.reset_pose_deltas()
                    converged = True
                else:
                    converged = update_pose(viewpoint)
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
            if converged:
                break
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    # ========================================================================
    # 6. Keyframe Detection
    # ========================================================================
    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]

        if last_keyframe_idx not in occ_aware_visibility:
            Log(f"[Warning] Keyframe {last_keyframe_idx} lost in visibility dict! Forcing a new keyframe to heal state.")
            return True

        pose_CW = curr_frame.T
        last_kf_CW = last_kf.T
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union

        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    # ========================================================================
    # 7. Sliding Window Management
    # ========================================================================
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

    # ========================================================================
    # 8. Submap Cutting — Motion Utility
    # ========================================================================
    def compute_submap_motion(self, current_c2w, anchor_c2w):
        if anchor_c2w is None:
            return 0.0, 0.0

        delta_T = torch.linalg.inv(anchor_c2w) @ current_c2w
        translation = torch.norm(delta_T[0:3, 3]).item()

        R = delta_T[0:3, 0:3]
        trace = torch.trace(R)
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta).item()
        angle_deg = angle_rad * 180.0 / np.pi
        return translation, angle_deg

    # ========================================================================
    # 9. Submap Cutting — Decision (motion-only)
    # ========================================================================
    def should_start_new_submap(self, current_c2w):
        if not self.use_submap:
            return False, None

        if self.submap_anchor_pose is None:
            return False, None

        translation, angle_deg = self.compute_submap_motion(current_c2w, self.submap_anchor_pose)

        if translation > self.submap_trans_thre or angle_deg > self.submap_rot_thre:
            metrics = {
                "translation": translation,
                "rotation_deg": angle_deg,
            }
            return True, metrics

        return False, None

    # ========================================================================
    # 10. Submap Cutting — Execution
    # ========================================================================
    def save_submap_cut_info(self, current_c2w, cut_metrics):
        submap_cut_info = {
            "submap_id": self.current_submap_id,
            "cut_frame_id": len(self.cameras),
            "cut_pose_c2w": current_c2w.detach().cpu().numpy(),
            "translation": cut_metrics["translation"],
            "rotation_deg": cut_metrics["rotation_deg"],
        }
        cut_info_path = os.path.join(
            self.config["Results"]["save_dir"],
            f"submap_cut_{self.current_submap_id:06d}.npy"
        )
        np.save(cut_info_path, submap_cut_info)

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

    def perform_submap_cut(self, cur_frame_idx, viewpoint, current_c2w, cut_metrics):
        Log(
            f"==> 启动新子图 (ID: {self.current_submap_id + 1}) | "
            f"trans={cut_metrics['translation']:.3f}m, "
            f"rot={cut_metrics['rotation_deg']:.1f}° <=="
        )

        self.save_submap_cut_info(current_c2w, cut_metrics)

        # 1) 当前帧已经在旧子图上完成 tracking；
        #    这里只允许小步 refine，不能重置位姿。
        cut_refine_iters = self.config.get("Submap", {}).get("cut_refine_iters", 20)
        if cut_refine_iters > 0:
            self.refine_cut_frame_pose(
                viewpoint,
                extra_iters=cut_refine_iters,
                lr_scale=0.25,
            )
            current_c2w = torch.linalg.inv(viewpoint.T)

        seed_global_c2w = current_c2w.detach().cpu().numpy().astype(np.float64)

        # 2) 保存子图间 transition：prev_seed -> new_seed
        if self.last_submap_seed_global_c2w is None:
            self.last_submap_seed_global_c2w = seed_global_c2w.copy()

        relative_pose = (
                np.linalg.inv(self.last_submap_seed_global_c2w) @ seed_global_c2w
        ).astype(np.float64)

        completed_submap_id = self.current_submap_id
        new_submap_id = completed_submap_id + 1

        self.submap_seed_global_c2w[new_submap_id] = seed_global_c2w.copy()
        self.last_submap_seed_global_c2w = seed_global_c2w.copy()

        self.submap_anchor_poses[new_submap_id] = np.eye(4, dtype=np.float64)

        # 3) 通知后端冻结旧子图。
        self.backend_queue.put(
            [
                "new_submap",
                completed_submap_id,
                relative_pose,
                seed_global_c2w,
            ]
        )

        self.current_submap_id = new_submap_id
        self.frame_to_submap[cur_frame_idx] = self.current_submap_id

        Log(
            f"[DEBUG] cut frame {cur_frame_idx} inherited global pose into "
            f"submap {self.current_submap_id}"
        )

        # 4) 清空当前子图窗口，但不要改 viewpoint.T。
        self.current_window = []
        self.occ_aware_visibility = {}
        self.initialized = False

        viewpoint.fixed_pose = True
        viewpoint.is_submap_seed = True
        viewpoint.seed_global_c2w = seed_global_c2w.copy()
        viewpoint.reset_pose_deltas()
        viewpoint.cam_rot_delta.requires_grad_(False)
        viewpoint.cam_trans_delta.requires_grad_(False)

        # 新子图运动监控锚点就是 seed 的全局位姿
        self.submap_anchor_pose = current_c2w.clone()
        self.submap_start_frame_idx = cur_frame_idx
        self.is_first_frame_of_submap = True

        # 5) 用当前 seed 帧初始化新子图。
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.current_window.append(cur_frame_idx)

        return True

    # ========================================================================
    # 11. Backend Communication
    # ========================================================================
    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        backend_occ = data[2] if data[2] is not None else {}
        keyframes = data[3]

        if not isinstance(self.occ_aware_visibility, dict):
            self.occ_aware_visibility = {}

        if isinstance(backend_occ, dict) and len(backend_occ) > 0:
            self.occ_aware_visibility.update(backend_occ)

        for kf_id, kf_T in keyframes:
            self.cameras[kf_id].T = kf_T

    # ========================================================================
    # 12. Coordinate Utility
    # ========================================================================
    def _camera_to_global_copy(self, cam):
        cam_g = clone_obj(cam)

        if self.config.get("Submap", {}).get("use_global_seed_submap", True):
            return cam_g

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

    # ========================================================================
    # 13. Cleanup
    # ========================================================================
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 5 == 0:
            torch.cuda.empty_cache()

    # ========================================================================
    # 14. Main Loop
    # ========================================================================
    def run(self):
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
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset):
                    torch.save(self.frame_to_submap, os.path.join(self.save_dir, "frame_to_submap.pt"))
                    torch.save(self.submap_anchor_poses, os.path.join(self.save_dir, "submap_anchor_poses.pt"))
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)
                self.cameras[cur_frame_idx] = viewpoint
                self.frame_to_submap[cur_frame_idx] = self.current_submap_id

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_c2w = torch.linalg.inv(viewpoint.T)

                if self.submap_anchor_pose is None:
                    self.submap_anchor_pose = current_c2w.clone()

                    seed_np = current_c2w.detach().cpu().numpy().astype(np.float64)
                    self.last_submap_seed_global_c2w = seed_np.copy()
                    self.submap_seed_global_c2w[self.current_submap_id] = seed_np.copy()

                # GUI update
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

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()

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

                if len(self.current_window) < self.window_size:
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
                        Log(
                            f"[Warning] Skip point_ratio check because keyframe "
                            f"{last_keyframe_idx} is missing in visibility dict."
                        )
                        create_kf = check_time

                if self.single_thread:
                    create_kf = check_time and create_kf

                # Submap cut decision
                should_cut_submap, cut_metrics = self.should_start_new_submap(current_c2w)

                if should_cut_submap:
                    did_cut = self.perform_submap_cut(
                        cur_frame_idx,
                        viewpoint,
                        current_c2w,
                        cut_metrics,
                    )
                    if did_cut:
                        cur_frame_idx += 1
                        continue

                if create_kf:
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

                    if self.use_error_mask:
                        viewpoint.error_mask = self.compute_error_mask(render_pkg, viewpoint)

                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

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
                        frame_to_submap=self.frame_to_submap,
                        submap_anchor_poses=self.submap_anchor_poses,
                        cameras_already_global=True,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    if not self.requested_init:
                        self.sync_backend(data)

                elif data[0] == "keyframe":
                    if self.requested_keyframe > 0:
                        self.sync_backend(data)
                        self.requested_keyframe -= 1
                    else:
                        Log("[Frontend] 拦截到旧子图的幽灵 keyframe 消息，已安全丢弃。")

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
