import copy
import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import numpy as np
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
import torch.nn.functional as F
import cv2
from utils.fft_filter import FFTFrequencyFilter

class BackEnd(mp.Process):
    # ========================================================================
    # 1. Initialization
    # ========================================================================
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ===== 外部注入的共享对象（由主进程在 start() 前赋值）=====
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None #后端到前端的通信队列
        self.backend_queue = None #前端到后端的通信队列
        self.loop_queue = None

        # ===== 进程控制 =====
        self.live_mode = False
        self.pause = False
        self.single_thread = False
        self.device = "cuda"
        self.dtype = torch.float32

        # ===== 建图状态 =====
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # ===== 消融开关 =====
        self.use_fdn = self.config.get("Ablation", {}).get("use_fdn", True)
        self.use_fft_mask = self.config.get("Ablation", {}).get("use_fft_mask", True)
        self.fft_filter = None

        # ===== 子图状态 =====
        self.current_submap_id = 0
        self.current_submap_seed_global_c2w = np.eye(4, dtype=np.float64)

        # ===== 子图切割参数 =====
        self.seed_init_iters = self.config.get("Submap", {}).get("seed_init_iters", 500)

        # ===== Gaussian Inheritance (替代 Handoff) =====
        submap_cfg = self.config.get("Submap", {})
        self.use_inheritance = submap_cfg.get("use_inheritance", True)
        self.inherit_keep_percent = submap_cfg.get("inherit_keep_percent", 0.45)
        self.inherit_min_keep = submap_cfg.get("inherit_min_keep", 3000)
        self.inherit_tail_kfs = submap_cfg.get("inherit_tail_kfs", 4)
        self.inherit_min_support = submap_cfg.get("inherit_min_support", 2)
        self.inherit_opacity_min = submap_cfg.get("inherit_opacity_min", 0.20)
        self.inherit_max_points = submap_cfg.get("inherit_max_points", 30000)
        Log(f"[Inheritance] enable={self.use_inheritance}, "
            f"keep_percent={self.inherit_keep_percent}")

        # ===== RAP2DGS Lite (用于 Inheritance 高斯评分) =====
        rap_cfg = self.config.get("RAP2DGSLite", {})
        self.rap2dgs_lite_enable = rap_cfg.get("enable", False)
        self.rap2dgs_lite_use_in_inheritance = rap_cfg.get("use_in_inheritance", False)
        self.rap2dgs_lite_cfg = rap_cfg
        self.rap2dgs_lite_selector = None
        Log(f"[RAP2DGS-Lite] enable={self.rap2dgs_lite_enable}, "
            f"use_in_inheritance={self.rap2dgs_lite_use_in_inheritance}")

        # ===== Backend pose policy (EAGS 风格: Gaussian only) =====
        self.optimize_keyframe_pose = self.config.get("Backend", {}).get(
            "optimize_keyframe_pose", False
        )
        self.optimize_keyframe_exposure = self.config.get("Backend", {}).get(
            "optimize_keyframe_exposure", False
        )
        self.backend_pose_sanity_check = self.config.get("Backend", {}).get(
            "backend_pose_sanity_check", True
        )
        self.backend_restore_pose_if_changed = self.config.get("Backend", {}).get(
            "backend_restore_pose_if_changed", True
        )
        self.pose_check_log_every = int(self.config.get("Backend", {}).get(
            "pose_check_log_every", 50
        ))
        self._pose_check_call_count = 0

        Log(
            f"[BackendPosePolicy] optimize_keyframe_pose={self.optimize_keyframe_pose}, "
            f"optimize_keyframe_exposure={self.optimize_keyframe_exposure}, "
            f"pose_sanity_check={self.backend_pose_sanity_check}"
        )

        # ===== RSKM (Random Sampling Keyframe Mapping) =====
        self.use_rskm = self.config.get("Training", {}).get("use_rskm", False)
        self.rskm_current_frame_interval = self.config.get("Training", {}).get("rskm_current_frame_interval", 4)
        rskm_seed = self.config.get("Training", {}).get("rskm_seed",
            self.config.get("Experiment", {}).get("seed", 42))
        self.rskm_rng = random.Random(rskm_seed)
        self.rskm_debug_log = self.config.get("Training", {}).get("rskm_debug_log", False)
        self._rskm_stats = None
        if self.use_rskm and self.rskm_debug_log:
            Log(f"[RSKM] enabled interval={self.rskm_current_frame_interval} seed={rskm_seed}")

        # ===== Normal Debug Log =====
        self.normal_debug_log = self.config.get("Training", {}).get("normal_debug_log", False)
        self.normal_debug_log_every = int(self.config.get("Training", {}).get(
            "normal_debug_log_every", 50
        ))
        self._normal_debug_iter_count = 0

    # ========================================================================
    # 2. Hyperparameters
    # ========================================================================
    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.nonvisible_reset_opacity = self.config["Training"].get("nonvisible_reset_opacity", 0.05)
        self.nonvisible_reset_stable_opacity = self.config["Training"].get("nonvisible_reset_stable_opacity", 0.08)
        self.nonvisible_reset_stable_n_obs = self.config["Training"].get("nonvisible_reset_stable_n_obs", 4)

    # ========================================================================
    # 3. State Management
    # ========================================================================
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        if len(self.gaussians._xyz) > 0:
            self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

        while not self.backend_queue.empty():
            self.backend_queue.get()

    # ========================================================================
    # 4. Seed Viewpoint Preparation
    # ========================================================================
    def prepare_seed_viewpoint_for_backend_init(self, viewpoint):
        viewpoint.is_submap_seed = True
        viewpoint.fixed_pose = True

        viewpoint.reset_pose_deltas()
        viewpoint.cam_rot_delta.requires_grad_(False)
        viewpoint.cam_trans_delta.requires_grad_(False)

    # ========================================================================
    # 5. Map Initialization
    # ========================================================================
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        # ==============================================================
        # 对齐 FGS-SLAM：在 Backend 内计算 freq_mask 和 error_mask
        # （而非 Frontend 中 tracking 时的一次性计算）
        # ==============================================================

        # ---- FFT 频率掩膜：描述图像纹理频率，用于 Gaussian 尺度初始化 ----
        if self.use_fft_mask:
            if self.fft_filter is None:
                H, W = viewpoint.image_height, viewpoint.image_width
                self.fft_filter = FFTFrequencyFilter(H, W)
            img_np = (viewpoint.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            viewpoint.freq_mask = self.fft_filter.generate_frequency_mask(img_bgr)

            # ---- 深度边缘过滤：纹理平面上 RGB 高频 ≠ 几何边缘，重新归类 ----
            evo_cfg = self.config.get("FFTEdgeVO", {})
            if evo_cfg.get("use_depth_edge_filter", False):
                depth_np = viewpoint.depth
                depth_f32 = depth_np.astype(np.float32, copy=False)
                gx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
                grad = np.sqrt(gx ** 2 + gy ** 2)
                valid = (depth_np > 0.01) & np.isfinite(depth_np)
                if valid.sum() > 100:
                    th = np.percentile(grad[valid], evo_cfg.get("depth_grad_percentile", 80))
                    depth_edge = grad > th
                    freq_np = viewpoint.freq_mask.cpu().numpy()
                    # 高频但不在深度边缘 → 降级为低频
                    demoted = freq_np & valid & (~depth_edge)
                    freq_np[demoted] = False
                    viewpoint.freq_mask = torch.from_numpy(freq_np).cuda()

            # ---- 可视化 FFT mask 播种区域 ----
            if self.config.get("Training", {}).get("debug_seed_vis", False):
                self._save_seed_vis(viewpoint, img_bgr, frame_idx)

        # ---- Error 掩膜：基于当前地图渲染的空洞 + 深度穿透检测 ----
        use_error_mask = self.config.get("Ablation", {}).get("use_error_mask", True)
        if (not init) and use_error_mask and len(self.gaussians._xyz) > 0:
            with torch.no_grad():
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
                )
                render_opacity = render_pkg["opacity"].detach()
                render_depth = render_pkg["depth"].detach()

                # Alpha 空洞掩膜（对齐 FGS-SLAM alpha_mask）
                alpha_mask = (render_opacity < 0.95).squeeze(0)

                # Depth 穿透误差掩膜（对齐 FGS-SLAM depth_error_mask）
                depth_error_mask = torch.zeros_like(alpha_mask, dtype=torch.bool)
                if (not self.monocular) and viewpoint.depth is not None:
                    gt_depth = torch.from_numpy(viewpoint.depth).to(
                        dtype=torch.float32, device=render_depth.device
                    )
                    depth_error = torch.abs(gt_depth - render_depth.squeeze(0))
                    valid_depth = gt_depth > 0.01
                    if valid_depth.any():
                        median_error = depth_error[valid_depth].median()
                        depth_error_mask = (
                            valid_depth
                            & (render_depth.squeeze(0) > gt_depth)
                            & (depth_error > 40.0 * median_error)
                        )

                viewpoint.error_mask = (alpha_mask | depth_error_mask)

        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def _save_seed_vis(self, viewpoint, img_bgr, frame_idx):
        """保存 FFT mask 播种可视化：高/低频区域 + 深度有效性叠加图。"""
        import os as _os
        save_dir = self.config["Results"]["save_dir"]
        _os.makedirs(save_dir, exist_ok=True)

        freq_np = viewpoint.freq_mask.detach().cpu().numpy()
        depth_np = viewpoint.depth
        depth_valid = (depth_np > 0.01) & np.isfinite(depth_np)

        # 背景：RGB 图
        vis = img_bgr.copy()
        # 绿色 = 高频 + 有效深度（小尺度 Gaussians）
        hf = freq_np & depth_valid
        vis[hf] = [0, 255, 0]
        # 蓝色 = 低频 + 有效深度（大尺度 Gaussians）
        lf = (~freq_np) & depth_valid
        vis[lf] = [255, 0, 0]
        # 红色 = 无效深度（不播种）
        no_d = ~depth_valid
        vis[no_d] = [0, 0, 255]

        # 半透明叠加
        blended = cv2.addWeighted(img_bgr, 0.5, vis, 0.5, 0)
        path = _os.path.join(save_dir, f"seed_vis_kf{frame_idx:04d}.png")
        cv2.imwrite(path, blended)

        Log(f"[SeedVis] 高频区点数={hf.sum()} 低频区点数={lf.sum()} 无效深度={no_d.sum()}  → {path}")

    def initialize_map(self, cur_frame_idx, viewpoint, iters=None):
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
            )

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

            if mapping_iteration % 5 == 0:
                self.push_to_frontend()

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        use_sa = self.config.get("pipeline_params", {}).get("use_sa", False)
        use_sa_depth = self.config.get("pipeline_params", {}).get("use_sa_depth", False)
        use_sa_dist = self.config.get("opt_params", {}).get("use_sa_dist", False)
        lambda_dist = self.config.get("opt_params", {}).get("lambda_dist", 0.0) # 仅供日志输出，实际是否启用 dist loss 还需同时满足 use_sa_dist=True 和 lambda_dist>0
        Log(f"[SA Config] use_sa={use_sa} use_sa_depth={use_sa_depth} use_sa_dist={use_sa_dist} lambda_dist={lambda_dist}")
        Log("Initialized map")

    # ========================================================================
    # 6. Map Optimization (Gaussian only)
    # ========================================================================
    def _backup_window_poses(self, current_window):
        pose_backup = {}
        for kf_idx in current_window:
            if kf_idx in self.viewpoints:
                pose_backup[int(kf_idx)] = self.viewpoints[kf_idx].T.detach().clone()
        return pose_backup

    @staticmethod
    def _pose_delta_stats(T_before, T_after):
        c2w_before = torch.linalg.inv(T_before.detach()).cpu()
        c2w_after = torch.linalg.inv(T_after.detach()).cpu()
        delta = torch.linalg.inv(c2w_before) @ c2w_after
        dt = torch.linalg.norm(delta[:3, 3]).item()
        R = delta[:3, :3]
        cos_angle = ((torch.trace(R) - 1.0) / 2.0).clamp(-1.0, 1.0)
        dr = torch.rad2deg(torch.acos(cos_angle)).item()
        return dt, dr

    def map(self, current_window, prune=False, iters=1,
            optimize_pose=None, optimize_exposure=None):
        if len(current_window) == 0:
            return

        if optimize_pose is None:
            optimize_pose = self.optimize_keyframe_pose
        if optimize_exposure is None:
            optimize_exposure = self.optimize_keyframe_exposure

        # ---- 后端 pose sanity check ----
        pose_backup = (
            self._backup_window_poses(current_window)
            if self.backend_pose_sanity_check else {}
        )

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for itr in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            if self.use_rskm and not prune:
                num_samples = len(current_window) + 2
                supervised_kf_ids = self._select_rskm_keyframes(current_window, num_samples)
                supervision_pairs = [(kf_idx, self.viewpoints[kf_idx]) for kf_idx in supervised_kf_ids]
                keyframes_opt = viewpoint_stack[:]
                current_kf_id = current_window[-1] if len(current_window) > 0 else None
                if self._rskm_stats is None:
                    self._rskm_stats = {
                        "total_iters": 0, "n_current": 0, "n_history": 0,
                        "history_kf_set": set(),
                    }
                self._rskm_stats["total_iters"] += 1
                if current_kf_id is not None:
                    for kf_idx in supervised_kf_ids:
                        if kf_idx == current_kf_id:
                            self._rskm_stats["n_current"] += 1
                        else:
                            self._rskm_stats["n_history"] += 1
                            self._rskm_stats["history_kf_set"].add(kf_idx)
            else:
                supervision_pairs = [(kf_idx, self.viewpoints[kf_idx]) for kf_idx in current_window]
                for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                    supervision_pairs.append((None, random_viewpoint_stack[cam_idx]))
                keyframes_opt = viewpoint_stack[:]

            # 法线 debug 统计量初始化
            if self.normal_debug_log and self.use_fdn:
                self._normal_debug_dot_sum = 0.0
                self._normal_debug_err_sum = 0.0
                self._normal_debug_count = 0

            for kf_idx, viewpoint in supervision_pairs:
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
                rend_dist = render_pkg["rend_dist"]

                loss_view = get_loss_mapping(
                    self.config, image, depth, viewpoint,
                    apply_exposure=optimize_exposure,
                    rend_dist=rend_dist,
                )

                if self.use_fdn and viewpoint.normal is not None:
                    rend_normal = render_pkg["rend_normal"]
                    rend_normal = F.normalize(rend_normal, p=2, dim=0)
                    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
                    sensor_normal = viewpoint.normal
                    gt_normal = (viewpoint.T[0:3, 0:3].T @ sensor_normal.view(3, -1)).view(
                        image.shape[0], image.shape[1], image.shape[2]
                    )
                    # normal_mask = gt_normal > 0
                    normal_error = (1 - (rend_normal * gt_normal * depth_pixel_mask).sum(dim=0))[None].mean()
                    loss_view += (self.config["opt_params"]["lambda_sensor_normal"] * normal_error)

                    if self.normal_debug_log:
                        # 累积法线对齐统计量（在有效深度像素上计算平均点积）
                        dot_per_pixel = (rend_normal * gt_normal).sum(dim=0)  # (H, W)
                        valid_mask = depth_pixel_mask.squeeze(0) > 0.5
                        if valid_mask.any():
                            self._normal_debug_dot_sum += float(dot_per_pixel[valid_mask].mean())
                            self._normal_debug_err_sum += float(normal_error)
                            self._normal_debug_count += 1

                loss_view.backward()

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append((kf_idx, n_touched))

                del render_pkg
                torch.cuda.empty_cache()

            # 法线 debug：每 N 次迭代输出一次对齐统计量
            if self.normal_debug_log and self.use_fdn:
                self._normal_debug_iter_count += 1
                if (self._normal_debug_iter_count % self.normal_debug_log_every == 0
                        and self._normal_debug_count > 0):
                    avg_dot = self._normal_debug_dot_sum / self._normal_debug_count
                    avg_err = self._normal_debug_err_sum / self._normal_debug_count
                    # 平均点积越接近 1.0，说明 rend_normal 与 sensor 法线方向越一致
                    Log(
                        f"[NormalAlign] iter={self._normal_debug_iter_count} "
                        f"avg_dot={avg_dot:.4f} avg_err={avg_err:.4f} "
                        f"n_pairs={self._normal_debug_count} "
                        f"(dot→1.0=法线方向一致)"
                    )

            gaussian_split = False

            with torch.no_grad():
                self.occ_aware_visibility = {}
                for kf_idx, n_touched in n_touched_acm:
                    if kf_idx is not None and kf_idx in current_window_set:
                        self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

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
                    actual_touched_filters = [(n_touched > 0) for _, n_touched in n_touched_acm]
                    self.gaussians.reset_opacity_nonvisible(
                        actual_touched_filters,
                        target_opacity=self.nonvisible_reset_opacity,
                        stable_opacity=self.nonvisible_reset_stable_opacity,
                        stable_n_obs=self.nonvisible_reset_stable_n_obs,
                    )
                    gaussian_split = True

                # Gaussian optimizer 始终 step
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                if optimize_pose and self.keyframe_optimizers is not None:
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
                    # optimize_pose=false: 冻结所有 keyframe pose
                    for viewpoint in viewpoint_stack:
                        viewpoint.reset_pose_deltas()
                        if viewpoint.cam_rot_delta is not None:
                            viewpoint.cam_rot_delta.requires_grad_(False)
                        if viewpoint.cam_trans_delta is not None:
                            viewpoint.cam_trans_delta.requires_grad_(False)

        # ---- 后端 pose sanity check（汇总输出，避免刷屏）----
        if self.backend_pose_sanity_check and pose_backup:
            checked = 0
            restored = 0
            max_dt = 0.0
            max_dr = 0.0
            restored_frames = []
            # float32 合理容忍阈值
            _eps_t = 1e-5      # m
            _eps_r_deg = 0.05  # deg

            for kf_idx, T_before in pose_backup.items():
                if kf_idx not in self.viewpoints:
                    continue
                T_after = self.viewpoints[kf_idx].T.detach()
                dt, dr = self._pose_delta_stats(T_before, T_after)
                checked += 1
                max_dt = max(max_dt, dt)
                max_dr = max(max_dr, dr)

                if not optimize_pose and (dt > _eps_t or dr > _eps_r_deg):
                    restored += 1
                    restored_frames.append((kf_idx, dt, dr))
                    if self.backend_restore_pose_if_changed:
                        self.viewpoints[kf_idx].T = T_before.clone()
                        self.viewpoints[kf_idx].reset_pose_deltas()
                        if self.viewpoints[kf_idx].cam_rot_delta is not None:
                            self.viewpoints[kf_idx].cam_rot_delta.requires_grad_(False)
                        if self.viewpoints[kf_idx].cam_trans_delta is not None:
                            self.viewpoints[kf_idx].cam_trans_delta.requires_grad_(False)

            self._pose_check_call_count += 1
            should_log = (
                restored > 0
                or (self._pose_check_call_count % self.pose_check_log_every == 0)
            )
            if restored == 0:
                if should_log:
                    Log(f"[BackendPoseCheck] checked={checked}, restored=0, "
                        f"max_dt={max_dt:.3e}m, max_dr={max_dr:.3e}deg — pose stable")
            else:
                Log(
                    f"[BackendPoseCheck] checked={checked}, restored={restored}, "
                    f"max_dt={max_dt:.6e}m, max_dr={max_dr:.6e}deg"
                )
                for fid, dt, dr in restored_frames:
                    Log(
                        f"[BackendPoseCheck][ERROR] Backend changed keyframe pose "
                        f"although optimize_pose=false! frame={fid} "
                        f"dt={dt:.6e}m dr={dr:.6e}deg"
                    )

        if self.use_rskm and self.rskm_debug_log and self._rskm_stats is not None and self._rskm_stats["total_iters"] > 0:
            stats = self._rskm_stats
            n_total = stats["n_current"] + stats["n_history"]
            pool_size = len(self.viewpoints)
            Log(f"[RSKM] submap={self.current_submap_id} "
                f"current_kf={current_window[-1] if len(current_window) > 0 else 'N/A'} "
                f"iters={stats['total_iters']} "
                f"pool={pool_size}kfs "
                f"total_samples={n_total} "
                f"current_samples={stats['n_current']} "
                f"history_samples={stats['n_history']} "
                f"distinct_history={len(stats['history_kf_set'])}")
            self._rskm_stats = None

        return gaussian_split

    # ========================================================================
    # 6.1 RSKM (Random Sampling Keyframe Mapping)
    # ========================================================================
    def _select_rskm_keyframes(self, current_window, num_samples):
        active_kf_ids = sorted(list(self.viewpoints.keys()))
        current_kf_id = current_window[-1] if len(current_window) > 0 else None

        selected = []
        for s in range(num_samples):
            iter_id = self.iteration_count + s
            if iter_id % self.rskm_current_frame_interval == 0:
                if current_kf_id is not None and current_kf_id in self.viewpoints:
                    selected.append(current_kf_id)
                    continue
            if len(active_kf_ids) <= 1 and current_kf_id is not None:
                selected.append(current_kf_id)
            elif len(active_kf_ids) > 0:
                selected.append(self.rskm_rng.choice(active_kf_ids))
            elif current_kf_id is not None:
                selected.append(current_kf_id)
        return selected

    # ========================================================================
    # 7. Submap Helpers
    # ========================================================================
    def pack_submap_keyframe_poses(self):
        submap_keyframe_poses = {}

        for kf_idx, viewpoint in self.viewpoints.items():
            with torch.no_grad():
                c2w = torch.linalg.inv(viewpoint.T.detach()).cpu().numpy()
            submap_keyframe_poses[int(kf_idx)] = c2w.astype(np.float64)

        return submap_keyframe_poses

    # ========================================================================
    # 7.1 Cross-Submap Covisibility Handoff
    # ========================================================================
    def _make_handoff_seed_viewpoint(self, c2w_np):
        """从现有 viewpoint 复制内参，设置新 C2W 位姿，构造临时渲染用 viewpoint。"""
        if len(self.viewpoints) == 0:
            return None
        src = next(iter(self.viewpoints.values()))
        vp = copy.copy(src)
        vp.T = torch.from_numpy(np.linalg.inv(c2w_np)).float().cuda()
        vp.fixed_pose = True
        return vp

    @staticmethod
    def _select_topk_mask(mask, score, max_points):
        """从 mask 中按 score 选最多 max_points 个元素，返回 bool mask。"""
        indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if indices.numel() <= max_points:
            return mask
        topk = torch.topk(score[indices], max_points, largest=True)
        selected = indices[topk.indices]
        result = torch.zeros_like(mask)
        result[selected] = True
        return result

    def _rap2dgs_lite_inheritance_select(self, cand_mask, support, opacity, max_keep):
        """RAP2DGS Lite inheritance selection with automatic fallback.

        Args:
            cand_mask: (N,) bool candidate mask
            support: (N,) long tensor, visibility support count
            opacity: (N,) float tensor, sigmoid-activated opacity
            max_keep: max points to keep
        Returns:
            selected: (N,) bool mask, always a subset of cand_mask
        """
        import time

        N = cand_mask.shape[0]
        t0 = time.time()

        if self.rap2dgs_lite_selector is None:
            from utils.rap2dgs_lite.selector import RAP2DGSLiteSelector
            self.rap2dgs_lite_selector = RAP2DGSLiteSelector(self.rap2dgs_lite_cfg)

        sel_cfg = self.rap2dgs_lite_cfg.get("selection", {})
        rap_max_keep = int(sel_cfg.get("max_keep", 8000))
        effective_max_keep = min(rap_max_keep, max_keep)

        current_kf_id = self.current_window[-1] if len(self.current_window) > 0 else None

        fallback = False
        fallback_reason = ""

        try:
            selected_mask, _scores, report = self.rap2dgs_lite_selector.select(
                gaussians=self.gaussians,
                candidate_mask=cand_mask,
                support_count=support,
                current_kf_id=current_kf_id,
                max_keep=effective_max_keep,
            )

            # ---- Safety checks ----
            if report.get("fallback_required", False):
                fallback = True
                fallback_reason = report.get("fallback_reason", "selector_fallback")
            elif selected_mask.sum() == 0:
                fallback = True
                fallback_reason = "empty_selected_mask"
            elif selected_mask.shape[0] != N:
                fallback = True
                fallback_reason = f"shape_mismatch:{selected_mask.shape[0]}!={N}"
            elif selected_mask.dtype != torch.bool:
                fallback = True
                fallback_reason = f"dtype_mismatch:{selected_mask.dtype}"
            elif (selected_mask & ~cand_mask).any():
                fallback = True
                fallback_reason = "selected_not_subset_of_candidate"
            else:
                pass  # all checks passed

            elapsed_ms = (time.time() - t0) * 1000.0

        except Exception as e:
            fallback = True
            fallback_reason = f"exception:{e}"
            elapsed_ms = (time.time() - t0) * 1000.0
            torch.cuda.empty_cache()

        if fallback:
            Log(f"[RAP2DGS-Lite] fallback: reason={fallback_reason} elapsed={elapsed_ms:.1f}ms")
            score = support.float() + 0.2 * opacity.float()
            return self._select_topk_mask(cand_mask, score, max_keep)

        n_cand = int(cand_mask.sum().item())
        n_sel = int(selected_mask.sum().item())
        Log(f"[RAP2DGS-Lite] inheritance selection: "
            f"candidates={n_cand} selected={n_sel} elapsed={elapsed_ms:.1f}ms fallback=False")

        # ---- Save report if configured ----
        if self.rap2dgs_lite_cfg.get("safety", {}).get("log_report", False):
            try:
                from utils.rap2dgs_lite.report import save_report_json

                report["num_total_gaussians"] = N
                report["num_candidates"] = n_cand
                report["num_selected"] = n_sel
                report["elapsed_ms"] = elapsed_ms
                report["fallback_required"] = False
                report["fallback_reason"] = ""

                save_dir = self.config["Results"]["save_dir"]
                report_dir_name = (
                    self.rap2dgs_lite_cfg.get("safety", {})
                    .get("report_dir", "rap2dgs_lite_reports")
                )
                out_dir = os.path.join(save_dir, report_dir_name)
                old_submap = self.current_submap_id
                new_submap = self.current_submap_id + 1
                label = f"inheritance_submap_{old_submap}_to_{new_submap}"
                save_report_json(report, out_dir, label=label)
            except Exception as e:
                Log(f"[RAP2DGS-Lite] failed to save report: {e}")

        return selected_mask

    def build_inheritance_mask(self, new_seed_global_c2w):
        """
        子图切换时评分旧高斯，返回 keep_mask 用于保留 top-K% 继承到新子图。
        与 Handoff 关键区别: 返回 mask 而非导出参数，保留的高斯仍为 active（可优化）。
        """
        if not self.use_inheritance:
            return torch.ones(self.gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")

        N = self.gaussians._xyz.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.bool, device="cuda")

        seed_vp = self._make_handoff_seed_viewpoint(new_seed_global_c2w)
        if seed_vp is None:
            return torch.ones(N, dtype=torch.bool, device="cuda")

        with torch.no_grad():
            render_pkg = render(
                seed_vp, self.gaussians, self.pipeline_params, self.background, surf=False
            )
            seed_visible = render_pkg["visibility_filter"]  # (N,) bool
            support = seed_visible.long().clone()

            tail_ids = list(self.current_window)[-self.inherit_tail_kfs:]
            valid_tail_ids = []
            for kf_id in tail_ids:
                if kf_id not in self.viewpoints:
                    continue
                kf_vp = self.viewpoints[kf_id]
                kf_render = render(
                    kf_vp, self.gaussians, self.pipeline_params, self.background, surf=False
                )
                kf_visible = kf_render["visibility_filter"]
                if kf_visible.shape[0] == N:
                    support += kf_visible.long()
                    valid_tail_ids.append(kf_id)

            opacity = self.gaussians.get_opacity.squeeze()
            cand_mask = (
                seed_visible
                & (support >= self.inherit_min_support)
                & (opacity > self.inherit_opacity_min)
            )

            # 从 candidate_mask 内按 keep_percent 选 top-K
            # inherit_min_keep 保证最少保留数，防止最后一个子图继承过少
            n_candidates = int(cand_mask.sum().item())
            n_keep = max(1, int(N * self.inherit_keep_percent))
            n_keep = max(n_keep, self.inherit_min_keep)
            n_keep = min(n_keep, n_candidates, self.inherit_max_points)

            if self.rap2dgs_lite_enable and self.rap2dgs_lite_use_in_inheritance:
                selected = self._rap2dgs_lite_inheritance_select(
                    cand_mask, support, opacity, n_keep
                )
                # RAP2DGS 内部 keep_percent 可能截断选择 → 用简单评分补足
                n_selected = int(selected.sum().item())
                if n_selected < n_keep:
                    remaining = cand_mask & ~selected
                    n_need = n_keep - n_selected
                    if remaining.any():
                        score = support.float() + 0.2 * opacity.float()
                        extra = self._select_topk_mask(remaining, score, n_need)
                        selected = selected | extra
            else:
                score = support.float() + 0.2 * opacity.float()
                selected = self._select_topk_mask(cand_mask, score, n_keep)

        n_kept = int(selected.sum().item())
        Log(f"[Inheritance] old_gaussians={N} seed_visible={int(seed_visible.sum().item())} "
            f"tail_kfs={tail_ids} valid_tail={valid_tail_ids}")
        Log(f"[Inheritance] candidates={n_candidates} kept={n_kept} "
            f"keep_ratio={n_kept / max(1, N):.4f}")

        return selected

    # ========================================================================
    # 8. Color Refinement (offline)
    # ========================================================================
    # ========================================================================
    # 10. Frontend Communication
    # ========================================================================
    @staticmethod
    def _parse_backend_msg(data):
        """Parse incoming message, handling old and new (meta) formats.

        Returns (tag, meta_or_None, payload_start_index).
        New format: [tag, meta_dict, payload...]  → payload starts at 2
        Old format: [tag, payload...]             → payload starts at 1
        """
        tag = data[0]
        if len(data) >= 2 and isinstance(data[1], dict) and "submap_id" in data[1]:
            return tag, data[1], 2
        return tag, None, 1

    def push_to_frontend(self, tag=None, meta=None):
        self.last_sent = 0
        keyframes = []

        if len(self.current_window) > 0:
            for kf_idx in self.current_window:
                if kf_idx in self.viewpoints:
                    kf = self.viewpoints[kf_idx]
                    keyframes.append((kf_idx, kf.T.clone()))
        else:
            if len(self.viewpoints) > 0:
                latest_kf_idx = sorted(self.viewpoints.keys())[-1]
                kf = self.viewpoints[latest_kf_idx]
                keyframes.append((latest_kf_idx, kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        # Stage 6: include meta for version checking
        if meta is None:
            meta = {
                "submap_id": self.current_submap_id,
                "request_id": -1,
                "frame_id": self.current_window[-1] if self.current_window else -1,
            }

        n_curr = self.gaussians._xyz.shape[0]
        if n_curr > 0:
            safe_occ = {k: v for k, v in self.occ_aware_visibility.items()
                        if isinstance(v, torch.Tensor) and v.shape[0] == n_curr}
        else:
            safe_occ = {}
        msg = [tag, meta, clone_obj(self.gaussians), safe_occ, keyframes]
        self.frontend_queue.put(msg)

    # ========================================================================
    # 11. Main Loop
    # ========================================================================
    def run(self):
        # ---- Seed Reproducibility (backend process) ----
        from utils.reproducibility import seed_everything
        base_seed = self.config.get("Experiment", {}).get("seed", 42)
        deterministic = self.config.get("Experiment", {}).get("deterministic", True)
        backend_seed = base_seed + 1
        seed_everything(backend_seed, deterministic=deterministic)
        Log(f"[Seed] backend_seed={backend_seed}")

        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue

                if self.pause or len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    save_dir = self.config["Results"]["save_dir"]
                    submaps_dir = os.path.join(save_dir, "submaps")
                    os.makedirs(submaps_dir, exist_ok=True)

                    gaussian_params = self.gaussians.capture_dict()
                    submap_keyframes = sorted(list(self.viewpoints.keys()))
                    ckpt_data = {
                        "gaussian_params": gaussian_params,
                        "submap_keyframes": submap_keyframes,
                        "seed_global_c2w": self.current_submap_seed_global_c2w,
                        "submap_keyframe_poses": self.pack_submap_keyframe_poses(),
                        "relative_pose": np.eye(4, dtype=np.float64),
                        "correct_tsfm": np.eye(4, dtype=np.float64),
                    }
                    ckpt_path = os.path.join(submaps_dir, f"{self.current_submap_id:06d}.ckpt")
                    torch.save(ckpt_data, ckpt_path)

                    kf_image_paths = []
                    kf_depth_paths = []
                    if len(submap_keyframes) > 0:
                        for kf_idx in submap_keyframes:
                            vp = self.viewpoints[kf_idx]
                            kf_image = vp.original_image.cpu()
                            img_path = os.path.join(submaps_dir, f"{self.current_submap_id:06d}_img_{kf_idx}.pt")
                            torch.save(kf_image, img_path)
                            kf_image_paths.append(img_path)
                            # Save depth map for Stage 4 depth verification
                            if not self.monocular and vp.depth is not None:
                                depth_tensor = torch.from_numpy(vp.depth.astype(np.float32))
                                depth_path = os.path.join(submaps_dir, f"{self.current_submap_id:06d}_depth_{kf_idx}.pt")
                                torch.save(depth_tensor, depth_path)
                                kf_depth_paths.append(depth_path)

                    if hasattr(self, 'loop_queue') and self.loop_queue is not None and len(kf_image_paths) > 0:
                        self.loop_queue.put(["submap_saved", self.current_submap_id, ckpt_path, kf_image_paths, kf_depth_paths])

                    Log(f"==> 终局保存：最后一块子图 {self.current_submap_id} 已存入硬盘。 <==")
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "init":
                    _, meta, off = self._parse_backend_msg(data)
                    cur_frame_idx = data[off]
                    viewpoint = data[off + 1]
                    depth_map = data[off + 2]

                    seed_global_c2w_from_viewpoint = (
                        torch.linalg.inv(viewpoint.T.detach()).cpu().numpy().astype(np.float64)
                    )
                    self.current_submap_seed_global_c2w = seed_global_c2w_from_viewpoint.copy()

                    if len(self.gaussians._xyz) == 0 and self.current_submap_id > 0:
                        Log("Initializing new submap from seed frame (state already clean)")
                        self.iteration_count = 0
                        self.occ_aware_visibility = {}
                        self.viewpoints = {}
                        self.current_window = []
                        self.initialized = not self.monocular
                        self.keyframe_optimizers = None
                    else:
                        Log("Resetting the system")
                        self.reset()

                    self.prepare_seed_viewpoint_for_backend_init(viewpoint)

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = [cur_frame_idx]
                    has_inherited = len(self.gaussians._xyz) > 0
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=not has_inherited
                    )
                    if has_inherited:
                        n_inherited = len(self.gaussians._xyz)
                        Log(f"[Inheritance] seed frame: {n_inherited} Gaussians present, "
                            f"skipping full seeding (init=False)")

                    if len(self.gaussians._xyz) == 0 and self.current_submap_id > 0:
                        init_iters = self.seed_init_iters
                    else:
                        init_iters = self.init_itr_num

                    self.initialize_map(cur_frame_idx, viewpoint, iters=init_iters)

                    self.push_to_frontend("init", meta=meta)
                elif data[0] == "keyframe":
                    _, meta, off = self._parse_backend_msg(data)
                    cur_frame_idx = data[off]
                    viewpoint = data[off + 1]
                    current_window = data[off + 2]
                    depth_map = data[off + 3]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    for cam_idx in range(len(self.current_window)):
                        vp = self.viewpoints[current_window[cam_idx]]
                        should_opt = (cam_idx < frames_to_optimize)

                        if should_opt and not getattr(vp, "fixed_pose", False):
                            if self.optimize_keyframe_pose:
                                rot_lr = self.config["Training"]["lr"]["cam_rot_delta"] * 0.5
                                trans_lr = self.config["Training"]["lr"]["cam_trans_delta"] * 0.5
                                opt_params.append({
                                    "params": [vp.cam_rot_delta],
                                    "lr": rot_lr,
                                    "name": "rot_{}".format(vp.uid),
                                })
                                opt_params.append({
                                    "params": [vp.cam_trans_delta],
                                    "lr": trans_lr,
                                    "name": "trans_{}".format(vp.uid),
                                })
                            else:
                                vp.reset_pose_deltas()
                                vp.cam_rot_delta.requires_grad_(False)
                                vp.cam_trans_delta.requires_grad_(False)

                            if self.optimize_keyframe_exposure:
                                opt_params.append({
                                    "params": [vp.exposure_a],
                                    "lr": 0.01,
                                    "name": "exposure_a_{}".format(vp.uid),
                                })
                                opt_params.append({
                                    "params": [vp.exposure_b],
                                    "lr": 0.01,
                                    "name": "exposure_b_{}".format(vp.uid),
                                })

                    if len(opt_params) > 0:
                        self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    else:
                        self.keyframe_optimizers = None
                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe", meta=meta)

                elif data[0] == "new_submap":
                    _, meta, off = self._parse_backend_msg(data)
                    completed_submap_id = data[off]
                    relative_pose_prev_seed_to_curr_seed = data[off + 1] if len(data) > off + 1 else np.eye(4, dtype=np.float64)
                    relative_pose_prev_seed_to_curr_seed = np.array(relative_pose_prev_seed_to_curr_seed, dtype=np.float64)
                    new_seed_global_c2w = (
                        data[off + 2] if len(data) > off + 2 else np.eye(4, dtype=np.float64)
                    )
                    new_seed_global_c2w = np.array(new_seed_global_c2w, dtype=np.float64)

                    completed_seed_global_c2w = np.array(
                        self.current_submap_seed_global_c2w, dtype=np.float64
                    )
                    self.current_submap_id = completed_submap_id + 1
                    self.current_submap_seed_global_c2w = new_seed_global_c2w.copy()
                    Log(f"==> Backend received new_submap signal. Freezing submap {completed_submap_id}...")
                    Log(
                        f"[SubmapSave] optimize_keyframe_pose={self.optimize_keyframe_pose}, "
                        f"keyframe poses are frontend tracking poses"
                    )

                    save_dir = self.config["Results"]["save_dir"]
                    submaps_dir = os.path.join(save_dir, "submaps")
                    os.makedirs(submaps_dir, exist_ok=True)

                    gaussian_params = self.gaussians.capture_dict()
                    submap_keyframes = sorted(list(self.viewpoints.keys()))
                    ckpt_data = {
                        "gaussian_params": gaussian_params,
                        "submap_keyframes": submap_keyframes,
                        "seed_global_c2w": completed_seed_global_c2w,
                        "submap_keyframe_poses": self.pack_submap_keyframe_poses(),
                        "relative_pose": relative_pose_prev_seed_to_curr_seed,
                        "correct_tsfm": np.eye(4, dtype=np.float64),
                    }

                    ckpt_path = os.path.join(submaps_dir, f"{completed_submap_id:06d}.ckpt")
                    torch.save(ckpt_data, ckpt_path)
                    Log(f"✓ Submap {completed_submap_id} parameters saved to {ckpt_path}")

                    kf_image_paths = []
                    kf_depth_paths = []
                    if len(submap_keyframes) > 0:
                        for kf_idx in submap_keyframes:
                            vp = self.viewpoints[kf_idx]
                            kf_image = vp.original_image.cpu()
                            img_path = os.path.join(submaps_dir, f"{completed_submap_id:06d}_img_{kf_idx}.pt")
                            torch.save(kf_image, img_path)
                            kf_image_paths.append(img_path)
                            # Save depth map for Stage 4 depth verification
                            if not self.monocular and vp.depth is not None:
                                depth_tensor = torch.from_numpy(vp.depth.astype(np.float32))
                                depth_path = os.path.join(submaps_dir, f"{completed_submap_id:06d}_depth_{kf_idx}.pt")
                                torch.save(depth_tensor, depth_path)
                                kf_depth_paths.append(depth_path)

                    if hasattr(self, 'loop_queue') and self.loop_queue is not None and len(kf_image_paths) > 0:
                        self.loop_queue.put(["submap_saved", completed_submap_id, ckpt_path, kf_image_paths, kf_depth_paths])
                        Log(f"✓ Submap {completed_submap_id} sent to loop closure")

                    # ---- Gaussian Inheritance: 保留 top-K% 旧高斯作为新子图 active 初始 ----
                    N_before = self.gaussians._xyz.shape[0]
                    if self.use_inheritance and N_before > 0:
                        keep_mask = self.build_inheritance_mask(new_seed_global_c2w)
                        prune_mask = ~keep_mask
                        self.gaussians.prune_points(prune_mask)
                        n_kept = self.gaussians._xyz.shape[0]
                        n_pruned = N_before - n_kept
                        Log(f"[Inheritance] kept {n_kept}/{N_before} Gaussians "
                            f"({n_kept / max(1, N_before) * 100:.1f}%), pruned {n_pruned}")
                    else:
                        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
                        Log("✓ Pruned ALL Gaussian points for true independent submap")

                    self.viewpoints.clear()
                    self.current_window = []
                    self.occ_aware_visibility = {}
                    self.keyframe_optimizers = None

                    # 给继承高斯加微小噪声，打破旧子图系统偏差
                    if self.use_inheritance and self.gaussians._xyz.shape[0] > 0:
                        noise = torch.randn_like(self.gaussians._xyz) * 0.001
                        self.gaussians._xyz.data += noise

                    self.gaussians.training_setup(self.opt_params)

                    torch.cuda.empty_cache()
                    Log("✓ Backend state reset. Waiting for seed frame init...")

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
