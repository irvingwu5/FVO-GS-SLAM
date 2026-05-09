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
        self.pending_backend_sync = None  # Stage 5: cache sync during requested_init
        # Stage 6: message versioning
        self.next_request_id = 0
        self.last_applied_backend_request_id = -1
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
        # 新子图 seed 帧不再重置到单位阵，而是继承旧子图 tracking 后的全局估计位姿。
        self.last_submap_seed_global_c2w = None
        # ================================================
        self.submap_motion_anchor_global_c2w = None #运动监控锚点（global C2W）
        self.submap_start_frame_idx = 0
        # ===== Gaussian Inheritance (Backend 处理，Frontend 无状态) =====
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
        # Previous-frame cache for VO reference init (avoids self-reference)
        self.last_vo_ref_rgb_bgr = None
        self.last_vo_ref_depth_np = None
        self.last_vo_ref_c2w = None
        self.last_vo_ref_frame_id = None
        self.tracking_refine_iters = int(evo_cfg.get("tracking_refine_iters", 20))
        self.tracking_fallback_iters = int(evo_cfg.get("tracking_fallback_iters", 100))
        self.debug_log = evo_cfg.get("debug_log", True)

        # Stage 2: Candidate selection for tracking initialization
        self.candidate_selection_enable = evo_cfg.get("candidate_selection_enable", False)
        self.candidate_lambda_depth = float(evo_cfg.get("candidate_lambda_depth", 1.0))
        self.candidate_lambda_coverage = float(evo_cfg.get("candidate_lambda_coverage", 1.0))
        self.candidate_min_opacity_ratio = float(evo_cfg.get("candidate_min_opacity_ratio", 0.05))
        self.last_c2ws = []  # up to 2 recent frame C2Ws for constant velocity prediction

        # Stage 3: VO render gate for dynamic refinement
        self.vo_render_gate_enable = evo_cfg.get("vo_render_gate_enable", False)
        self.vo_max_score_ratio_to_best = float(evo_cfg.get("vo_max_score_ratio_to_best", 1.25))
        self.vo_max_score_ratio_to_previous = float(evo_cfg.get("vo_max_score_ratio_to_previous", 1.10))
        self.vo_min_opacity_ratio = float(evo_cfg.get("vo_min_opacity_ratio", 0.05))
        self.vo_max_depth_loss = evo_cfg.get("vo_max_depth_loss", None)  # None = no limit
        self.vo_max_color_loss = evo_cfg.get("vo_max_color_loss", None)

        # Stage 7: submap cut quality gate
        self.submap_cut_gate_enable = evo_cfg.get("submap_cut_gate_enable", False)
        self.submap_cut_min_opacity = float(evo_cfg.get("submap_cut_min_opacity", 0.05))
        self.submap_cut_max_delay = int(evo_cfg.get("submap_cut_max_delay", 3))
        self.submap_cut_delay_count = 0
        self.last_tracking_diag = {}

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
    # 2b. Candidate Selection (Stage 2: multi-candidate render arbitration)
    # ========================================================================

    def _build_candidates(self, prev_cam, vo_success, est_c2w):
        """Build list of initial pose candidates for render arbitration.

        Candidates (in order): previous, constant_velocity, fft_vo.
        Each is a dict with name, c2w (float64 numpy), valid.
        """
        candidates = []

        # 1. Previous frame pose
        if prev_cam is not None:
            prev_c2w = torch.linalg.inv(prev_cam.T).cpu().numpy().astype(np.float64)
            candidates.append({"name": "previous", "c2w": prev_c2w, "valid": True})

        # 2. Constant velocity: C2W_pred = C2W_{t-1} @ inv(C2W_{t-2}) @ C2W_{t-1}
        if len(self.last_c2ws) >= 2:
            rel = np.linalg.inv(self.last_c2ws[0]) @ self.last_c2ws[1]
            cv_c2w = self.last_c2ws[1] @ rel
            candidates.append({"name": "constant_velocity", "c2w": cv_c2w, "valid": True})

        # 3. FFT Edge VO (only if raw VO succeeded)
        if vo_success and est_c2w is not None:
            candidates.append({"name": "fft_vo", "c2w": est_c2w, "valid": True})

        return candidates

    def _render_precheck(self, c2w, viewpoint):
        """Lightweight render loss check for a candidate pose.

        No gradients, no optimizer step, no Gaussian updates.
        Returns dict with l1_rgb, l1_depth, opacity_ratio.
        """
        with torch.no_grad():
            orig_T = viewpoint.T.clone()
            # Set candidate pose (C2W → W2C)
            viewpoint.T = torch.from_numpy(np.linalg.inv(c2w)).float().cuda()
            viewpoint.cam_rot_delta.data.fill_(0)
            viewpoint.cam_trans_delta.data.fill_(0)

            render_pkg = render(
                viewpoint, self._get_render_model(), self.pipeline_params,
                self.background, surf=False)
            image = render_pkg["render"]
            depth = render_pkg["depth"]
            opacity = render_pkg["opacity"]

            # Color loss (L1, opacity-weighted, matches get_loss_tracking_rgb)
            gt_image = viewpoint.original_image.cuda()
            _, h, w = gt_image.shape
            rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
            rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(1, h, w)
            rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
            l1_rgb = (opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)).mean().item()

            # Depth loss (L1, opacity-masked valid depth pixels)
            gt_depth = torch.from_numpy(viewpoint.depth).to(
                dtype=torch.float32, device=image.device)[None]
            depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
            opacity_mask = (opacity > 0.95).view(*depth.shape)
            depth_mask = depth_pixel_mask * opacity_mask
            n_valid = depth_mask.sum()
            if n_valid > 0:
                l1_depth = (torch.abs(depth * depth_mask - gt_depth * depth_mask).sum() / n_valid).item()
            else:
                l1_depth = float("inf")

            # Opacity coverage
            opacity_ratio = opacity_mask.float().mean().item()

            # Restore original pose
            viewpoint.T = orig_T

        return {"l1_rgb": l1_rgb, "l1_depth": l1_depth, "opacity_ratio": opacity_ratio}

    def _select_candidate(self, candidates, viewpoint):
        """Select best candidate by render precheck score.

        Returns the winning candidate dict with added "metrics" and "score" fields.
        Falls back to "previous" if all candidates fail precheck.
        """
        best_cand = None
        best_score = float("inf")

        for cand in candidates:
            if not cand.get("valid", True):
                continue
            metrics = self._render_precheck(cand["c2w"], viewpoint)
            cand["metrics"] = metrics

            # Reject if coverage too low
            if metrics["opacity_ratio"] < self.candidate_min_opacity_ratio:
                cand["rejected"] = True
                continue

            # Score: combined weighted loss
            score = (metrics["l1_rgb"]
                     + self.candidate_lambda_depth * metrics["l1_depth"]
                     + self.candidate_lambda_coverage * max(0, self.candidate_min_opacity_ratio - metrics["opacity_ratio"]))
            cand["score"] = score

            if score < best_score:
                best_score = score
                best_cand = cand

        if best_cand is None and candidates:
            # All rejected: fallback to first (previous) candidate
            best_cand = candidates[0]
            best_cand["fallback"] = True

        return best_cand

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
        return self.gaussians

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

        # ---- Step 1: FFT Edge VO estimation ---------------------------------
        vo_success = False
        est_c2w = None
        vo_info = {}
        if self.use_fft_edge_vo:
            self._init_fft_edge_vo()
            rgb_bgr = self._camera_rgb_to_bgr(viewpoint)
            depth_np = viewpoint.depth
            init_c2w = (torch.linalg.inv(prev_cam.T).cpu().numpy().astype(np.float64)
                        if prev_cam is not None else None)

            if not self.fft_edge_vo_initialized:
                if self.last_vo_ref_rgb_bgr is not None:
                    # Case A: use previous frame as reference, track current frame
                    ok = self.fft_edge_vo.set_reference(
                        self.last_vo_ref_rgb_bgr, self.last_vo_ref_depth_np,
                        self.last_vo_ref_c2w, frame_id=self.last_vo_ref_frame_id)
                    if ok:
                        self.fft_edge_vo_initialized = True
                        if self.debug_log:
                            Log(f"[FFTEdgeVO] init with prev frame {self.last_vo_ref_frame_id} "
                                f"as ref, tracking frame {cur_frame_idx}")
                        vo_success, est_c2w, vo_info = self.fft_edge_vo.track(
                            rgb_bgr, depth_np, init_c2w)
                else:
                    # Case B: no previous frame, init reference only, skip tracking
                    if init_c2w is None:
                        init_c2w = np.eye(4, dtype=np.float64)
                    ok = self.fft_edge_vo.set_reference(rgb_bgr, depth_np, init_c2w,
                                                          frame_id=cur_frame_idx)
                    if ok:
                        self.fft_edge_vo_initialized = True
                        if self.debug_log:
                            Log(f"[FFTEdgeVO] init ref with frame {cur_frame_idx} "
                                f"(no prev frame), skip VO this frame")
                    # vo_success stays False → fallback to prev pose
            else:
                vo_success, est_c2w, vo_info = self.fft_edge_vo.track(rgb_bgr, depth_np, init_c2w)

        # ---- Step 2: Pose initialization (candidate selection / direct VO) ---
        # ---- Step 2: Pose initialization (candidate selection / direct VO) ---
        vo_render_accepted = False
        vo_reject_reason = None
        selected_candidate_name = None

        if self.candidate_selection_enable and self.gaussians is not None:
            candidates = self._build_candidates(prev_cam, vo_success, est_c2w)
            if candidates:
                selected = self._select_candidate(candidates, viewpoint)
                if selected is not None:
                    viewpoint.T = torch.from_numpy(np.linalg.inv(selected["c2w"])).float().cuda()
                    selected_candidate_name = selected.get("name")
                    if self.debug_log:
                        names = [f"{c['name']}(s={c.get('score','?'):.4f})" for c in candidates]
                        Log(f"[Candidate] frame {cur_frame_idx}: {', '.join(names)} → "
                            f"selected={selected_candidate_name}")

                    # Stage 3: VO render acceptance gate
                    if self.vo_render_gate_enable and vo_success and selected_candidate_name == "fft_vo":
                        vo_metrics = selected.get("metrics", {})
                        vo_score = selected.get("score", float("inf"))
                        vo_l1_rgb = vo_metrics.get("l1_rgb", float("inf"))
                        vo_l1_depth = vo_metrics.get("l1_depth", float("inf"))
                        vo_opacity = vo_metrics.get("opacity_ratio", 0)

                        # Check against absolute thresholds
                        checks = []
                        if self.vo_max_color_loss is not None and vo_l1_rgb > self.vo_max_color_loss:
                            vo_reject_reason = f"color_loss {vo_l1_rgb:.4f} > {self.vo_max_color_loss}"
                        elif self.vo_max_depth_loss is not None and vo_l1_depth > self.vo_max_depth_loss:
                            vo_reject_reason = f"depth_loss {vo_l1_depth:.4f} > {self.vo_max_depth_loss}"
                        elif vo_opacity < self.vo_min_opacity_ratio:
                            vo_reject_reason = f"opacity_ratio {vo_opacity:.4f} < {self.vo_min_opacity_ratio}"
                        else:
                            # Check relative to best candidate
                            vo_render_accepted = True

                        if not vo_render_accepted and self.debug_log:
                            Log(f"[VO Gate] frame {cur_frame_idx}: VO render REJECTED ({vo_reject_reason})")
                    elif not self.vo_render_gate_enable:
                        # Gate disabled: VO accepted if it won the candidate race
                        vo_render_accepted = (selected_candidate_name == "fft_vo")
            elif prev_cam is not None:
                viewpoint.T = prev_cam.T.clone()
        else:
            # Original behaviour: VO directly overwrites pose
            if vo_success:
                viewpoint.T = torch.from_numpy(np.linalg.inv(est_c2w)).float().cuda()
                selected_candidate_name = "fft_vo"
            elif prev_cam is not None:
                viewpoint.T = prev_cam.T.clone()
                selected_candidate_name = "previous"
        # Final fallback for no-VO, no-candidate, no-prev case
        if not self.use_fft_edge_vo and not self.candidate_selection_enable:
            if prev_cam is not None:
                viewpoint.T = prev_cam.T.clone()
                selected_candidate_name = "previous"

        # ---- Step 3: Refinement iteration count (Stage 3 dynamic strategy) ---
        if self.candidate_selection_enable or self.vo_render_gate_enable:
            # Stage 3: only use short iters when VO is render-accepted
            if vo_render_accepted:
                refine_iters = self.tracking_refine_iters
            else:
                refine_iters = self.tracking_fallback_iters
        elif self.use_fft_edge_vo:
            refine_iters = (self.tracking_refine_iters if vo_success
                            else self.tracking_fallback_iters)
        else:
            refine_iters = self.tracking_itr_num
        refine_iters = max(refine_iters, 5)  # minimum GPU refinement

        # ---- Step 4: Render refinement ------------------------------------
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

        # Cache current frame as "previous" for next VO reference init
        if self.use_fft_edge_vo:
            self.last_vo_ref_rgb_bgr = rgb_bgr
            self.last_vo_ref_depth_np = depth_np
            self.last_vo_ref_c2w = torch.linalg.inv(viewpoint.T).cpu().numpy().astype(np.float64)
            self.last_vo_ref_frame_id = cur_frame_idx

        # Update constant velocity cache (Stage 2)
        if self.candidate_selection_enable:
            current_c2w = torch.linalg.inv(viewpoint.T).cpu().numpy().astype(np.float64)
            self.last_c2ws.append(current_c2w)
            if len(self.last_c2ws) > 2:
                self.last_c2ws.pop(0)

        # Stage 7: save tracking quality diagnostics
        self.last_tracking_diag = {
            "frame_id": cur_frame_idx,
            "render_loss": best_loss,
            "opacity_ratio": (best_render_pkg["opacity"] > 0.95).float().mean().item()
            if best_render_pkg is not None else 0,
            "selected_candidate": selected_candidate_name,
            "vo_render_accepted": vo_render_accepted,
        }

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
    # ========================================================================
    # 9. Submap Cutting — Decision (motion-only) + Quality Gate (Stage 7)
    # ========================================================================

    def can_cut_submap_now(self):
        """Check if current tracking quality is sufficient for submap seed.

        Returns (ok, reason).
        """
        if not self.submap_cut_gate_enable:
            return True, "gate_disabled"

        diag = self.last_tracking_diag
        if not diag:
            return True, "no_diag"

        opacity_ratio = diag.get("opacity_ratio", 0)
        if opacity_ratio < self.submap_cut_min_opacity:
            return False, f"low_opacity ({opacity_ratio:.4f} < {self.submap_cut_min_opacity})"

        render_loss = diag.get("render_loss", float("inf"))
        if render_loss >= float("inf") or np.isnan(render_loss):
            return False, f"invalid_loss ({render_loss})"

        # VO render-accepted requirement: if VO won but wasn't render-accepted, block
        if (diag.get("selected_candidate") == "fft_vo"
                and not diag.get("vo_render_accepted", True)):
            return False, "vo_not_render_accepted"

        return True, "ok"

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
        meta = self._make_meta(cur_frame_idx)
        # new_submap meta carries the old (completed) submap_id
        meta["submap_id"] = completed_submap_id
        self.backend_queue.put(
            [
                "new_submap",
                meta,
                completed_submap_id,
                relative_pose_prev_seed_to_curr_seed,
                seed_global_c2w,
            ]
        )

        self.current_submap_id = new_submap_id
        self.last_applied_backend_request_id = -1  # Stage 6: reset on new submap
        self.frame_to_submap[cur_frame_idx] = self.current_submap_id

        Log(
            f"[DEBUG] cut frame {cur_frame_idx} inherited global pose into "
            f"submap {self.current_submap_id}"
        )

        # Reset FFT Edge VO so new submap gets its own reference
        self.fft_edge_vo_initialized = False

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
    # 11. Backend Communication (Stage 6: message versioning)
    # ========================================================================

    def _make_meta(self, frame_id):
        """Build version meta dict for outgoing messages."""
        meta = {
            "submap_id": self.current_submap_id,
            "request_id": self.next_request_id,
            "frame_id": frame_id,
        }
        self.next_request_id += 1
        return meta

    def _parse_msg_meta(self, data):
        """Extract (tag, meta, payload_offset) from a message.

        Handles both old format (list, tag at [0]) and new format
        (list, tag at [0], meta dict at [1], payload starting at [2]).
        Returns (tag, meta_or_None, payload_start_index).
        """
        tag = data[0]
        if len(data) >= 2 and isinstance(data[1], dict) and "submap_id" in data[1]:
            return tag, data[1], 2
        return tag, None, 1

    def _check_backend_msg(self, meta):
        """Check if a backend message is current. Returns (ok, reason)."""
        if meta is None:
            # Old format: accept with warning once
            return True, "no_meta"
        if meta["submap_id"] != self.current_submap_id:
            return False, f"wrong_submap (msg={meta['submap_id']}, cur={self.current_submap_id})"
        if meta["request_id"] < self.last_applied_backend_request_id:
            return False, f"stale_request (msg={meta['request_id']}, last={self.last_applied_backend_request_id})"
        self.last_applied_backend_request_id = meta["request_id"]
        return True, "ok"

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        meta = self._make_meta(cur_frame_idx)
        msg = ["keyframe", meta, cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        meta = self._make_meta(cur_frame_idx)
        msg = ["init", meta, cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        _, _meta, off = self._parse_msg_meta(data)
        self.gaussians = data[off]
        backend_occ = data[off + 1] if data[off + 1] is not None else {}
        keyframes = data[off + 2]

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

        # Stage 4: Sync VO reference if backend updated the reference keyframe pose
        if (self.use_fft_edge_vo and self.fft_edge_vo is not None
                and self.fft_edge_vo.ref_frame_id is not None):
            vo_ref_id = self.fft_edge_vo.ref_frame_id
            if vo_ref_id in self.cameras:
                new_ref_c2w = torch.linalg.inv(self.cameras[vo_ref_id].T).cpu().numpy().astype(np.float64)
                old_ref_c2w = self.fft_edge_vo.ref_c2w
                if old_ref_c2w is not None:
                    # Compute pose delta
                    delta_T = np.linalg.inv(old_ref_c2w) @ new_ref_c2w
                    delta_trans = np.linalg.norm(delta_T[:3, 3])
                    delta_rot_deg = np.rad2deg(np.arccos(
                        max(-1.0, min(1.0, (np.trace(delta_T[:3, :3]) - 1.0) / 2.0))))
                    trans_th = self.fft_edge_vo.ref_pose_sync_trans_th
                    rot_th = self.fft_edge_vo.ref_pose_sync_rot_deg
                    if delta_trans > trans_th or delta_rot_deg > rot_th:
                        # Large change: rebuild reference pyramid
                        if (self.fft_edge_vo.ref_rgb_bgr is not None
                                and self.fft_edge_vo.ref_depth_np is not None):
                            self.fft_edge_vo.set_reference(
                                self.fft_edge_vo.ref_rgb_bgr,
                                self.fft_edge_vo.ref_depth_np,
                                new_ref_c2w, frame_id=vo_ref_id)
                            if self.debug_log:
                                Log(f"[VO Sync] frame {vo_ref_id}: rebuilt ref "
                                    f"(ΔT={delta_trans:.4f}m, ΔR={delta_rot_deg:.2f}°)")
                    else:
                        # Small change: only update pose
                        self.fft_edge_vo.update_reference_pose(new_ref_c2w)
                        if self.debug_log:
                            Log(f"[VO Sync] frame {vo_ref_id}: updated ref pose "
                                f"(ΔT={delta_trans:.4f}m, ΔR={delta_rot_deg:.2f}°)")

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
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

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
                        self.fft_edge_vo.set_reference(rgb_bgr, depth_np, c2w_np,
                                                       frame_id=cur_frame_idx)

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

                # Submap cut decision (Stage 7: quality gate)
                should_cut_submap, cut_metrics = self.should_start_new_submap(current_c2w, cur_frame_idx)

                if should_cut_submap:
                    # Stage 7: quality gate check
                    can_cut, gate_reason = self.can_cut_submap_now()
                    if not can_cut:
                        self.submap_cut_delay_count += 1
                        if self.submap_cut_delay_count >= self.submap_cut_max_delay:
                            Log(f"[Submap Gate] forced cut after {self.submap_cut_delay_count} "
                                f"delays (reason: {gate_reason})")
                        else:
                            Log(f"[Submap Gate] delayed cut (reason: {gate_reason}, "
                                f"delay={self.submap_cut_delay_count}/{self.submap_cut_max_delay})")
                            should_cut_submap = False

                if should_cut_submap:
                    self.submap_cut_delay_count = 0  # reset delay counter
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
                                rgb_bgr, depth_np, c2w, frame_id=cur_frame_idx)
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
                        self.fft_edge_vo.set_reference(rgb_bgr, depth_np, c2w,
                                                       frame_id=cur_frame_idx)
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
                tag, meta, _ = self._parse_msg_meta(data)

                if tag == "sync_backend":
                    # Stage 6: check submap_id before applying
                    ok, reason = self._check_backend_msg(meta)
                    if not ok and reason.startswith("wrong_submap"):
                        Log(f"[Frontend] drop stale sync_backend: {reason}")
                        continue
                    if self.requested_init:
                        # Stage 5: cache latest sync, apply after init completes
                        self.pending_backend_sync = data
                    else:
                        self.sync_backend(data)

                elif tag == "keyframe":
                    ok, reason = self._check_backend_msg(meta)
                    if not ok:
                        Log(f"[Frontend] drop stale keyframe: {reason}")
                        continue
                    if self.requested_keyframe > 0:
                        self.sync_backend(data)
                        self.requested_keyframe -= 1
                    else:
                        Log("[Frontend] 拦截到旧子图的幽灵 keyframe 消息，已安全丢弃。")

                elif tag == "init":
                    ok, reason = self._check_backend_msg(meta)
                    if not ok:
                        Log(f"[Frontend] drop stale init: {reason}")
                        continue
                    self.sync_backend(data)
                    self.requested_init = False
                    # Stage 5: apply pending sync that arrived during init
                    if self.pending_backend_sync is not None:
                        self.sync_backend(self.pending_backend_sync)
                        self.pending_backend_sync = None

                elif tag == "stop":
                    Log("Frontend Stopped.")
                    break
