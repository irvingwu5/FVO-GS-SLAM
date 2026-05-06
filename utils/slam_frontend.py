import time

import numpy as np
import torch
import torch.multiprocessing as mp
import os
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
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
from utils.fft_edge_vo import FFTEdgeVO

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
        # Pose convention (global mode):
        #   cam.T is always global W2C.  inv(cam.T) is always global C2W.
        #   Submaps are memory/optimization partitions, NOT coordinate partitions.
        # ========== 子图策略状态变量 ==========
        self.current_submap_id = 0
        self.submap_trans_thre = self.config["Submap"]["trans_thre"]
        self.submap_rot_thre = self.config["Submap"]["rot_thre"]
        self.frame_to_submap = {}  # <--- 记录每帧属于哪个子图
        # ========== LoopSplat-style submap handoff ==========
        # 新子图 seed 帧不再重置到单位阵，而是继承旧子图 tracking 后的全局估计位姿。
        self.last_submap_seed_global_c2w = None
        # ================================================
        self.submap_motion_anchor_global_c2w = None #运动监控锚点（global C2W）
        self.submap_start_frame_idx = 0
        # ===== Cross-Submap Covisibility Handoff =====
        self.handoff_gaussians = None  # frozen GaussianModel from old submap boundary
        self.handoff_age_frames = 0
        _sub_cfg = self.config.get("Submap", {})
        self.handoff_warmup_keyframes = _sub_cfg.get("handoff_warmup_keyframes", 3)
        self.handoff_safety_age = 200  # 安全兜底: 超过此帧数未达标则强制退出
        self._handoff_dropped = False  # 防止 backend sync 重新激活已释放的 handoff
        self._handoff_eval = {}  # per-submap eval stats
        self.fft_filter = None  # 频域滤波器实例
        # 消融实验开关
        self.use_submap = self.config.get("Ablation", {}).get("use_submap", True)
        self.use_fft_mask = self.config.get("Ablation", {}).get("use_fft_mask", True)
        self.use_error_mask = self.config.get("Ablation", {}).get("use_error_mask", True)

        # ===== FFT Edge VO 模块 (dense DT alignment, Edge VO style) =====
        evo_cfg = self.config.get("FFTEdgeVO", {})
        self.use_fft_edge_vo = evo_cfg.get("use_fft_edge_vo", False)
        self.fft_edge_vo = None
        self.fft_edge_vo_initialized = False
        self.tracking_refine_iters = int(evo_cfg.get("tracking_refine_iters", 20))
        self.tracking_fallback_iters = int(evo_cfg.get("tracking_fallback_iters", 100))
        self.debug_log = evo_cfg.get("debug_log", True)

    # ========================================================================
    # 2. FFT Edge VO helpers
    # ========================================================================

    def _init_fft_edge_vo(self):
        if self.fft_edge_vo is not None:
            return
        self.fft_edge_vo = FFTEdgeVO(
            self.config,
            W=self.dataset.width, H=self.dataset.height,
            fx=self.dataset.fx, fy=self.dataset.fy,
            cx=self.dataset.cx, cy=self.dataset.cy,
        )
        Log("[FFTEdgeVO] tracker created (Edge VO style, dense FFT-mask DT alignment)")

    @staticmethod
    def _camera_rgb_to_bgr(cam):
        img = getattr(cam, "original_image", None)
        if img is None:
            return None
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)  # (3,H,W) → (H,W,3)
        if img.max() <= 1.01:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # ========================================================================
    # 3. Hyperparameters
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
        self.fft_edge_vo_initialized = False
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

        # Record submap 0 seed C2W for submap-to-submap transition computation
        seed_c2w = np.linalg.inv(viewpoint.T_gt.cpu().numpy()).astype(np.float64)
        self.last_submap_seed_global_c2w = seed_c2w.copy()

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()

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
        对齐 FGS-SLAM seeding_mask：alpha 空洞 + 深度穿透误差
        freq_mask 管"新高斯点尺度"，error_mask 管"在哪里补新高斯点"
        """
        gt_image = viewpoint.original_image.cuda()  # [3, H, W]
        render_opacity = render_pkg["opacity"].detach()  # [1, H, W]

        # 1. Alpha 掩膜 (地图空洞，对齐 FGS-SLAM alpha_mask)
        alpha_mask = (render_opacity < 0.95).squeeze(0)  # [H, W]

        # 2. Depth 穿透误差掩膜 (对齐 FGS-SLAM depth_error_mask)
        depth_error_mask = torch.zeros_like(alpha_mask, dtype=torch.bool)
        if not self.monocular and viewpoint.depth is not None:
            gt_depth = torch.from_numpy(viewpoint.depth).cuda()  # [H, W]
            render_depth = render_pkg["depth"].detach().squeeze(0)  # [H, W]
            depth_error = torch.abs(gt_depth - render_depth)

            valid_depth = gt_depth > 0.01
            if valid_depth.any():
                median_error = depth_error[valid_depth].median()
                # FGS-SLAM: render_depth > gt_depth AND depth_error > 40 * median
                depth_error_mask = (
                    valid_depth
                    & (render_depth > gt_depth)
                    & (depth_error > 40.0 * median_error)
                )

        # 综合掩膜：空洞 | 深度穿透（对齐 FGS-SLAM seeding_mask = alpha_mask | depth_error_mask）
        error_mask = alpha_mask | depth_error_mask

        return error_mask

    def _get_render_model(self):
        if self.handoff_gaussians is not None:
            return GaussianModel.create_merged_for_render(self.gaussians, self.handoff_gaussians)
        return self.gaussians

    def _maybe_drop_handoff(self, cur_frame_idx):
        """唯一退出条件: 关键帧数达标 AND 覆盖率达标。200 帧安全兜底防止死循环。"""
        if self.handoff_gaussians is None:
            return False, ""

        cov_th = self.config.get("Submap", {}).get("handoff_new_coverage_th", 0.85)

        # 安全兜底: 超过 safety_age 帧仍未达标，强制退出
        if self.handoff_age_frames >= self.handoff_safety_age:
            Log(f"[Handoff] SAFETY: age={self.handoff_age_frames} >= {self.handoff_safety_age}, "
                f"forcing drop (coverage never reached {cov_th})", tag="WARN")
            return True, "safety_age"

        # 正常退出: 关键帧数达标 AND 覆盖率达标（每 10 帧检查一次）
        kf_ready = len(self.current_window) >= self.handoff_warmup_keyframes
        if kf_ready and cur_frame_idx % 10 == 0 and self.gaussians is not None and len(self.gaussians._xyz) > 0:
            with torch.no_grad():
                viewpoint = self.cameras.get(cur_frame_idx)
                if viewpoint is None:
                    return False, ""
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
                )
                opacity = render_pkg["opacity"]
                coverage = (opacity > 0.95).float().mean().item()
                if coverage >= cov_th:
                    Log(f"[Handoff] keyframes={len(self.current_window)}>={self.handoff_warmup_keyframes} "
                        f"AND coverage={coverage:.3f}>={cov_th}, dropping")
                    return True, "keyframes+coverage"

        return False, ""

    def _drop_handoff(self, reason):
        n_handoff = self.handoff_gaussians._xyz.shape[0] if self.handoff_gaussians is not None else 0
        n_active = self.gaussians._xyz.shape[0] if self.gaussians is not None else 0
        Log(f"[Handoff] dropped: reason={reason} age_frames={self.handoff_age_frames} "
            f"handoff_points={n_handoff} active_points={n_active}")
        self.handoff_gaussians = None
        self.handoff_age_frames = 0
        self._handoff_dropped = True  # 阻止 backend sync 重新激活

    # ========================================================================
    # 5. Tracking (FFT Edge VO + render refinement)
    # ========================================================================
    def tracking(self, cur_frame_idx, viewpoint):
        if viewpoint.fixed_pose:
            with torch.no_grad():
                render_pkg = render(
                    viewpoint, self._get_render_model(), self.pipeline_params, self.background, surf=False
                )
            self.median_depth = get_median_depth(
                render_pkg["depth"], render_pkg["opacity"]
            )
            return render_pkg

        prev_cam = self.cameras.get(cur_frame_idx - 1)

        # ---- Step 1: FFT Edge VO pose estimation ---------------------------
        vo_success = False
        if self.use_fft_edge_vo:
            self._init_fft_edge_vo()
            rgb_bgr = self._camera_rgb_to_bgr(viewpoint)
            depth_np = viewpoint.depth
            init_c2w = (torch.linalg.inv(prev_cam.T).cpu().numpy().astype(np.float64)
                        if prev_cam is not None else None)

            if not self.fft_edge_vo_initialized:
                if init_c2w is None:
                    init_c2w = np.eye(4, dtype=np.float64)
                ok = self.fft_edge_vo.set_reference(rgb_bgr, depth_np, init_c2w)
                if ok:
                    self.fft_edge_vo_initialized = True
                    vo_success, est_c2w, _ = self.fft_edge_vo.track(rgb_bgr, depth_np, init_c2w)
                # else: vo_success stays False
            else:
                vo_success, est_c2w, _ = self.fft_edge_vo.track(rgb_bgr, depth_np, init_c2w)

            if vo_success:
                viewpoint.T = torch.from_numpy(np.linalg.inv(est_c2w)).float().cuda()
            elif prev_cam is not None:
                viewpoint.T = prev_cam.T.clone()
        elif prev_cam is not None:
            viewpoint.T = prev_cam.T.clone()

        # ---- Step 2: Refinement iteration count ----------------------------
        if self.use_fft_edge_vo:
            refine_iters = (self.tracking_refine_iters if vo_success
                            else self.tracking_fallback_iters)
        else:
            refine_iters = self.tracking_itr_num
        refine_iters = max(refine_iters, 5)  # minimum GPU refinement

        # ---- Step 3: Render refinement ------------------------------------
        viewpoint.fixed_pose = False
        viewpoint.cam_rot_delta.requires_grad_(True)
        viewpoint.cam_trans_delta.requires_grad_(True)

        opt_params = [
            {"params": [viewpoint.cam_rot_delta],
             "lr": self.config["Training"]["lr"]["cam_rot_delta"],
             "name": f"rot_{viewpoint.uid}"},
            {"params": [viewpoint.cam_trans_delta],
             "lr": self.config["Training"]["lr"]["cam_trans_delta"],
             "name": f"trans_{viewpoint.uid}"},
            {"params": [viewpoint.exposure_a], "lr": 0.01,
             "name": f"exposure_a_{viewpoint.uid}"},
            {"params": [viewpoint.exposure_b], "lr": 0.01,
             "name": f"exposure_b_{viewpoint.uid}"},
        ]
        pose_optimizer = torch.optim.Adam(opt_params)
        best_T, best_loss, best_render_pkg = None, float("inf"), None

        render_model = self._get_render_model()  # 一次创建，整帧复用
        for tracking_itr in range(refine_iters):
            render_pkg = render(
                viewpoint, render_model, self.pipeline_params, self.background, surf=False
            )
            image, depth, opacity = (
                render_pkg["render"], render_pkg["depth"], render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()
            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)
                current_loss = loss_tracking.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_T = viewpoint.T.clone()
                    best_render_pkg = {
                        "render": image.detach().clone(),
                        "depth": depth.detach().clone(),
                        "opacity": opacity.detach().clone(),
                        "n_touched": render_pkg.get("n_touched", None),
                    }
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

        if best_T is not None:
            viewpoint.T = best_T.clone()
        self.median_depth = get_median_depth(
            best_render_pkg["depth"], best_render_pkg["opacity"]
        )
        return best_render_pkg if best_render_pkg is not None else render_pkg

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

        last_vis = occ_aware_visibility[last_keyframe_idx]
        if last_vis.shape[0] != cur_frame_visibility_filter.shape[0]:
            return True  # densify 后 shape 变化，静默强制关键帧自愈

        pose_CW = curr_frame.T
        last_kf_CW = last_kf.T
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, last_vis
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, last_vis
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
            if kf_visibility.shape[0] != cur_frame_visibility_filter.shape[0]:
                to_remove.append(kf_idx)
                continue

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
    def compute_submap_motion(self, current_c2w, anchor_global_c2w):
        if anchor_global_c2w is None:
            return 0.0, 0.0

        delta_T = torch.linalg.inv(anchor_global_c2w) @ current_c2w
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
    def should_start_new_submap(self, current_c2w, cur_frame_idx):
        if not self.use_submap:
            return False, None

        if self.submap_motion_anchor_global_c2w is None:
            return False, None

        # guard: minimum frames between cuts (prevents infinite cut loops)
        min_frames = 3
        if cur_frame_idx - self.submap_start_frame_idx < min_frames:
            return False, None

        translation, angle_deg = self.compute_submap_motion(current_c2w, self.submap_motion_anchor_global_c2w)

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

    def perform_submap_cut(self, cur_frame_idx, viewpoint, current_c2w, cut_metrics):
        Log(
            f"==> 启动新子图 (ID: {self.current_submap_id + 1}) | "
            f"trans={cut_metrics['translation']:.3f}m, "
            f"rot={cut_metrics['rotation_deg']:.1f}° <=="
        )

        self.save_submap_cut_info(current_c2w, cut_metrics)

        # 切图帧的 global pose 作为新子图 seed
        seed_global_c2w = current_c2w.detach().cpu().numpy().astype(np.float64)

        # 2) 保存子图间 transition：prev_seed -> new_seed
        if self.last_submap_seed_global_c2w is None:
            self.last_submap_seed_global_c2w = seed_global_c2w.copy()

        relative_pose_prev_seed_to_curr_seed = (
                np.linalg.inv(self.last_submap_seed_global_c2w) @ seed_global_c2w
        ).astype(np.float64)

        completed_submap_id = self.current_submap_id
        new_submap_id = completed_submap_id + 1

        self.last_submap_seed_global_c2w = seed_global_c2w.copy()

        # 3) 通知后端冻结旧子图。
        self.backend_queue.put(
            [
                "new_submap",
                completed_submap_id,
                relative_pose_prev_seed_to_curr_seed,
                seed_global_c2w,
            ]
        )

        self.current_submap_id = new_submap_id
        self.frame_to_submap[cur_frame_idx] = self.current_submap_id

        Log(
            f"[DEBUG] cut frame {cur_frame_idx} inherited global pose into "
            f"submap {self.current_submap_id}"
        )

        # Reset FFT Edge VO so new submap gets its own reference
        self.fft_edge_vo_initialized = False
        self._handoff_dropped = False  # 新子图允许接收 handoff

        # 4) 清空当前子图窗口，但不要改 viewpoint.T。
        self.current_window = []
        self.occ_aware_visibility = {}
        self.initialized = False

        viewpoint.fixed_pose = True
        viewpoint.is_submap_seed = True
        viewpoint.reset_pose_deltas()
        viewpoint.cam_rot_delta.requires_grad_(False)
        viewpoint.cam_trans_delta.requires_grad_(False)

        # 新子图运动监控锚点就是 seed 的全局位姿
        self.submap_motion_anchor_global_c2w = current_c2w.clone()
        self.submap_start_frame_idx = cur_frame_idx

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
            if self.gaussians is not None and self.gaussians._xyz.shape[0] > 0:
                n_curr = self.gaussians._xyz.shape[0]
                self.occ_aware_visibility = {
                    k: v for k, v in backend_occ.items()
                    if isinstance(v, torch.Tensor) and v.shape[0] == n_curr
                }
            else:
                self.occ_aware_visibility = dict(backend_occ)

        for kf_id, kf_T in keyframes:
            self.cameras[kf_id].T = kf_T

        # Handoff: 接收 frozen GaussianModel（Stage 5）
        if len(data) > 4 and data[4] is not None:
            if self._handoff_dropped:
                pass  # 本子图 handoff 已退出，忽略 backend 后续推送
            else:
                (handoff_clone,) = data[4]
                if self.handoff_gaussians is None:
                    self.handoff_age_frames = 0  # 首次收到 Handoff 时重置 age
                self.handoff_gaussians = handoff_clone
        elif len(data) <= 4 or data[4] is None:
            self.handoff_gaussians = None
            self.handoff_age_frames = 0

    # ========================================================================
    # 12. Coordinate Utility
    # ========================================================================
    def _camera_to_global_copy(self, cam):
        """Return a deep copy of cam. Cameras are already in global coordinates."""
        return clone_obj(cam)

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

                # 延迟清理：帧 N-2 已不再被任何 prev_cam 引用，可安全释放
                stale_idx = cur_frame_idx - 2
                if stale_idx >= 0 and stale_idx in self.cameras and stale_idx not in self.current_window:
                    self.cleanup(stale_idx)

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                if self.handoff_gaussians is not None:
                    self.handoff_age_frames += 1
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                if self.handoff_gaussians is not None:
                    if self.handoff_age_frames <= 20 and self.handoff_age_frames % 5 == 0:
                        with torch.no_grad():
                            active_pkg = render(viewpoint, self.gaussians,
                                                self.pipeline_params, self.background, surf=False)
                            active_cov = (active_pkg["opacity"] > 0.95).float().mean().item()
                        hf_pts = self.handoff_gaussians._xyz.shape[0]
                        Log(f"[HandoffEval] submap={self.current_submap_id} frame={cur_frame_idx} "
                            f"age={self.handoff_age_frames} active_cov={active_cov:.3f} "
                            f"handoff_pts={hf_pts}")

                    dropped, reason = self._maybe_drop_handoff(cur_frame_idx)
                    if dropped:
                        self._drop_handoff(reason)

                current_c2w = torch.linalg.inv(viewpoint.T)

                # Auto-refresh FFT Edge VO reference when quality degrades
                # (large rotation / scene change causes mask overlap to drop)
                if self.use_fft_edge_vo and self.fft_edge_vo is not None:
                    _min_vis = max(300, self.fft_edge_vo.ref_count * 0.5)
                    if (self.fft_edge_vo.last_dt_mean > 8.0
                            or self.fft_edge_vo.last_visible < _min_vis):
                        rgb_bgr = self._camera_rgb_to_bgr(viewpoint)
                        depth_np = viewpoint.depth
                        c2w_np = current_c2w.cpu().numpy().astype(np.float64)
                        self.fft_edge_vo.set_reference(rgb_bgr, depth_np, c2w_np)

                if self.submap_motion_anchor_global_c2w is None:
                    self.submap_motion_anchor_global_c2w = current_c2w.clone()

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
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                # tracking render 使用 merged (active+handoff)，但 occ_aware_visibility 只有 active
                # 切片到 active-only 部分避免 shape mismatch
                if self.handoff_gaussians is not None and self.gaussians is not None:
                    n_active = self.gaussians._xyz.shape[0]
                    if n_active > 0 and n_active < curr_visibility.shape[0]:
                        curr_visibility = curr_visibility[:n_active]

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
                        if last_visibility.shape[0] != curr_visibility.shape[0]:
                            create_kf = check_time
                        else:
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
                should_cut_submap, cut_metrics = self.should_start_new_submap(current_c2w, cur_frame_idx)

                if should_cut_submap:
                    did_cut = self.perform_submap_cut(
                        cur_frame_idx,
                        viewpoint,
                        current_c2w,
                        cut_metrics,
                    )
                    if did_cut:
                        # Init FFT Edge VO reference from seed frame (has tracked
                        # pose + fresh data; otherwise next frame fails with stale ref)
                        if self.use_fft_edge_vo:
                            self._init_fft_edge_vo()
                            rgb_bgr = self._camera_rgb_to_bgr(viewpoint)
                            depth_np = viewpoint.depth
                            c2w = current_c2w.cpu().numpy().astype(np.float64)
                            self.fft_edge_vo_initialized = self.fft_edge_vo.set_reference(
                                rgb_bgr, depth_np, c2w)
                        else:
                            self.fft_edge_vo_initialized = False
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

                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                    # Refresh FFT Edge VO reference with this keyframe
                    if self.use_fft_edge_vo and self.fft_edge_vo is not None:
                        rgb_bgr = self._camera_rgb_to_bgr(viewpoint)
                        depth_np = viewpoint.depth
                        c2w = torch.linalg.inv(viewpoint.T).cpu().numpy().astype(np.float64)
                        self.fft_edge_vo.set_reference(rgb_bgr, depth_np, c2w)
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
