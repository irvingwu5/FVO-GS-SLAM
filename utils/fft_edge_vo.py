# utils/fft_edge_vo.py
# FFT Edge VO — dense visual odometry, Edge VO (REVO) style.
#
# Architecture aligned with EAGS-SLAM Edge VO:
#   cur→ref direction: project current-frame 3D points into reference-frame
#   distance-transform pyramid.  Reference DT + gradients are built once and
#   reused for every subsequent frame — no per-frame DT computation.
#
#   Optimisation: damped Gauss-Newton (Levenberg-Marquardt style) with the
#   analytic SE(3) Jacobian from Kerl's MSc thesis (TU Munich, 2012, p.34).

import numpy as np
import torch
import cv2
import os
from utils.fft_filter import FFTFrequencyFilter
from utils.logging_utils import Log


# ============================================================================
# SE(3) helpers (torch, GPU)
# ============================================================================

def _skew(v):
    return torch.tensor([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], device=v.device, dtype=v.dtype)


def _so3_exp(omega):
    theta = torch.linalg.norm(omega)
    if theta < 1e-10:
        return torch.eye(3, device=omega.device, dtype=omega.dtype) + _skew(omega)
    axis = omega / theta
    K = _skew(axis)
    s, c = torch.sin(theta), torch.cos(theta)
    return torch.eye(3, device=omega.device, dtype=omega.dtype) + s * K + (1.0 - c) * (K @ K)


def _se3_exp(xi):
    """SE(3) exponential map.  xi = [omega, v] ∈ R^6 → 4×4 matrix."""
    omega, v = xi[:3], xi[3:]
    R = _so3_exp(omega)
    theta = torch.linalg.norm(omega)
    if theta < 1e-10:
        t = v
    else:
        axis = omega / theta
        K = _skew(axis)
        s, c = torch.sin(theta), torch.cos(theta)
        V = (torch.eye(3, device=xi.device, dtype=xi.dtype)
             + (1.0 - c) / theta * K + (theta - s) / theta * (K @ K))
        t = V @ v
    T = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ============================================================================
# FFTEdgeVO
# ============================================================================

class FFTEdgeVO:
    """Dense visual odometry — FFT mask alignment, Edge VO style.

    Direction:  cur → ref  (project current-frame 3D points into the reference
    distance-transform pyramid).  The reference optimisation structure (DT +
    Sobel gradients) is built once in set_reference() and reused for every
    subsequent track() call — no per-frame DT needed.

    Optimiser:  damped Gauss-Newton on SE(3) with the analytic Jacobian from
    Kerl (2012).  Typically converges in 3–8 iterations.
    """

    def __init__(self, config, W, H, fx, fy, cx, cy):
        cfg = config.get("FFTEdgeVO", {})

        self.W, self.H = W, H
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

        # ---- pyramid --------------------------------------------------------
        self.num_levels = int(cfg.get("num_levels", 3))
        self.level_scale = float(cfg.get("level_scale", 0.5))

        # ---- LM optimisation ------------------------------------------------
        self.max_iters_coarse = int(cfg.get("max_iters_coarse", 6))
        self.max_iters_fine   = int(cfg.get("max_iters_fine",   8))
        self.lm_lambda_init   = float(cfg.get("lm_lambda_init", 0.1))
        self.conv_eps         = float(cfg.get("conv_eps", 0.999))

        # ---- DT / Huber -----------------------------------------------------
        self.dt_max   = float(cfg.get("dt_max_dist", 50.0))
        self.dt_huber = float(cfg.get("dt_huber_delta", 5.0))

        # ---- point budget ---------------------------------------------------
        self.max_cur_pts = int(cfg.get("max_cur_points", 8000))
        self.min_cur_pts = int(cfg.get("min_cur_points", 300))

        # ---- depth gate (Stage 8) --------------------------------------------
        self.min_depth = float(cfg.get("min_depth", 0.1))
        self.max_depth = float(cfg.get("max_depth", 8.0))

        # ---- depth edge filter (EAGS-style geometric edge constraint) ------
        self.use_depth_edge_filter = cfg.get("use_depth_edge_filter", False)
        self.depth_grad_percentile = int(cfg.get("depth_grad_percentile", 80))

        # ---- sampling (Stage 8) ----------------------------------------------
        self.sampling_strategy = cfg.get("sampling_strategy", "grid")
        self.vo_random_seed = int(cfg.get("vo_random_seed",
            config.get("Experiment", {}).get("seed", 42)))

        # ---- quality --------------------------------------------------------
        self.dt_mean_fail       = float(cfg.get("dt_mean_fail_threshold", 15.0))
        self.require_visible_ratio = float(cfg.get("require_visible_ratio", 0.3))

        # ---- multi-condition success gate (Stage 2) ------------------------
        self.dt_all_p90_fail  = float(cfg.get("dt_all_p90_fail_threshold", 20.0))
        self.max_outside_ratio = float(cfg.get("max_outside_ratio", 0.25))
        self.min_near_edge_ratio = float(cfg.get("min_near_edge_ratio", 0.10))
        self.max_near_edge_ratio = float(cfg.get("max_near_edge_ratio", 0.95))
        self.min_good_bad_ratio = float(cfg.get("min_good_bad_ratio", 2.0))

        # ---- FFT filter (lazy) ----------------------------------------------
        self.fft_filter = None

        # ---- reference state ------------------------------------------------
        self.opt_struct_pyr = None   # list of torch float32 (H_lvl, W_lvl, 4)
        self.K_pyr          = None   # list of (fx, fy, cx, cy) per level
        self.T_wr           = None   # 4×4 world→ref camera, numpy float64
        self.ref_count      = 0
        self.ref_frame_id   = None   # frame id of the reference (Stage 4)
        self.ref_c2w        = None   # reference C2W, numpy float64 (Stage 4)
        self.ref_rgb_bgr    = None   # cached reference BGR image (Stage 4)
        self.ref_depth_np   = None   # cached reference depth (Stage 4)

        # ---- diagnostics ----------------------------------------------------
        self.last_dt_mean = float("inf")
        self.last_visible = 0
        self.last_iters   = 0

        # ---- Stage 4: reference pose sync ---------------------------------
        self.ref_pose_sync_trans_th = float(cfg.get("ref_pose_sync_trans_th", 0.01))
        self.ref_pose_sync_rot_deg = float(cfg.get("ref_pose_sync_rot_deg", 0.5))

        # ----
        self.debug       = cfg.get("debug_log", True)
        self.min_dt_mean = float(cfg.get("min_dt_mean", 0.5))

        # ---- debug visualization -------------------------------------------
        self.debug_save_images  = cfg.get("debug_save_images", False)
        self.debug_save_dir     = cfg.get("debug_save_dir", "debug_vo")
        self.debug_save_interval = int(cfg.get("debug_save_interval", 10))
        self._debug_frame_count = 0  # internal counter for interval gating

        # Resolve debug_save_dir under experiment save_dir if available
        exp_save_dir = config.get("Results", {}).get("save_dir")
        if exp_save_dir and not os.path.isabs(self.debug_save_dir):
            self.debug_save_dir = os.path.join(exp_save_dir, self.debug_save_dir)

        if self.debug and self.debug_save_images:
            Log(f"[FFTEdgeVO] debug viz ENABLED: saving to '{self.debug_save_dir}/' "
                f"every {self.debug_save_interval} frames")

    # ====================================================================
    # FFT mask
    # ====================================================================

    def _ensure_fft_filter(self):
        if self.fft_filter is None:
            self.fft_filter = FFTFrequencyFilter(self.H, self.W)

    def _compute_mask(self, image_bgr):
        self._ensure_fft_filter()
        return self.fft_filter.generate_frequency_mask(image_bgr)

    # ====================================================================
    # Pyramid helpers
    # ====================================================================

    @staticmethod
    def _pyramid_sizes(H, W, num_levels, scale):
        sizes = []
        h, w = H, W
        for _ in range(num_levels):
            sizes.append((h, w))
            h, w = max(1, int(h * scale)), max(1, int(w * scale))
            if h < 8 or w < 8:
                break
        return sizes[::-1]  # coarsest first

    # ====================================================================
    # Reference frame — build optimisation structure once
    # ====================================================================

    def _build_opt_struct_pyramid(self, mask_full):
        """Build multi-level optimisation structure (gx, gy, dt, 0).

        gx, gy are the Sobel gradients of the distance transform, pre-multiplied
        by fx, fy at each level so the Jacobian formula simplifies (Kerl 2012).
        dt is the (scaled) distance-transform value.
        """
        mask_np = mask_full.cpu().numpy().astype(np.uint8) * 255
        dt_full = cv2.distanceTransform(
            255 - mask_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
        )
        dt_full = np.clip(dt_full, 0.0, self.dt_max)

        sizes = self._pyramid_sizes(self.H, self.W, self.num_levels, self.level_scale)
        struct_pyr = []
        K_pyr = []

        for h, w in sizes:
            scale_x = w / self.W
            scale_y = h / self.H
            fx_l = self.fx * scale_x
            fy_l = self.fy * scale_y
            cx_l = self.cx * scale_x
            cy_l = self.cy * scale_y
            K_pyr.append((fx_l, fy_l, cx_l, cy_l))

            if h == self.H and w == self.W:
                dt = dt_full
            else:
                s = w / self.W
                dt = cv2.resize(dt_full, (w, h), interpolation=cv2.INTER_LINEAR)
                dt *= s  # physical px at this resolution

            # Sobel gradients of DT (pre-multiply by focal lengths)
            gx = cv2.Sobel(dt, cv2.CV_32F, 1, 0, ksize=3) * fx_l
            gy = cv2.Sobel(dt, cv2.CV_32F, 0, 1, ksize=3) * fy_l

            # Pack: (gx, gy, dt, 0) → (H, W, 4)
            ch4 = np.stack([gx, gy, dt, np.zeros_like(dt)], axis=-1).astype(np.float32)
            struct_pyr.append(torch.from_numpy(ch4).cuda())

        return struct_pyr, K_pyr

    def set_reference(self, image_bgr, depth_np, c2w, frame_id=None):
        """Build optimisation structure from reference FFT mask + store T_world_ref."""
        mask = self._compute_mask(image_bgr)
        n_mask = int(mask.sum().item())

        if n_mask < self.min_cur_pts:
            self.opt_struct_pyr = None
            self.ref_count = 0
            if self.debug:
                Log(f"[FFTEdgeVO] set_reference FAILED: {n_mask} mask px < {self.min_cur_pts}")
            return False

        self.opt_struct_pyr, self.K_pyr = self._build_opt_struct_pyramid(mask)
        c2w_f64 = c2w.astype(np.float64)
        self.T_wr = np.linalg.inv(c2w_f64)  # world→ref (w2c)
        self.ref_count = n_mask
        # Stage 4: cache reference metadata for backend pose sync
        self.ref_c2w = c2w_f64.copy()
        self.ref_frame_id = frame_id
        self.ref_rgb_bgr = image_bgr
        self.ref_depth_np = depth_np

        if self.debug:
            Log(f"[FFTEdgeVO] reference: {n_mask} mask px, "
                f"{len(self.opt_struct_pyr)} pyramid levels, T_wr ready")
        return True

    def update_reference_pose(self, ref_c2w):
        """Update only the global reference pose (T_wr).

        Does NOT rebuild the image pyramid — reference image/depth unchanged.
        Call this when the backend optimises the reference keyframe's pose.
        """
        c2w_f64 = ref_c2w.astype(np.float64)
        self.T_wr = np.linalg.inv(c2w_f64)
        self.ref_c2w = c2w_f64.copy()
        if self.debug:
            Log(f"[FFTEdgeVO] ref pose updated (frame {self.ref_frame_id}), "
                f"pyramid unchanged")

    # ====================================================================
    # Current-frame 3D points (camera frame, NOT world)
    # ====================================================================

    def _backproject_cur_mask(self, mask, depth_np):
        """Backproject current-frame FFT-mask pixels to 3D in CURRENT camera frame.

        Sampling strategy (Stage 8):
          - "grid": uniform grid sampling (deterministic)
          - "random": random sampling with fixed seed

        Returns  torch float32 (N,3) on CUDA, empty if too few valid points.
        """
        mask_np = mask.cpu().numpy()
        ys, xs = np.where(mask_np)

        if len(ys) == 0:
            return torch.zeros((0, 3), device="cuda", dtype=torch.float32)

        # Depth gate
        Z_all = depth_np[ys, xs]
        depth_ok = (Z_all > self.min_depth) & (Z_all < self.max_depth)
        ys, xs = ys[depth_ok], xs[depth_ok]

        if len(ys) == 0:
            return torch.zeros((0, 3), device="cuda", dtype=torch.float32)

        # Depth edge filter (EAGS-style: keep only geometric edges)
        if self.use_depth_edge_filter:
            depth_f32 = depth_np.astype(np.float32, copy=False)
            depth_grad_x = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
            depth_grad_y = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
            depth_grad = np.sqrt(depth_grad_x ** 2 + depth_grad_y ** 2)
            valid_depth_mask = (depth_np > self.min_depth) & (depth_np < self.max_depth)
            if valid_depth_mask.sum() > 100:
                grad_th = np.percentile(depth_grad[valid_depth_mask], self.depth_grad_percentile)
                depth_edge = depth_grad > grad_th
                edge_ok = depth_edge[ys, xs]
                ys, xs = ys[edge_ok], xs[edge_ok]

        if len(ys) == 0:
            return torch.zeros((0, 3), device="cuda", dtype=torch.float32)

        # Subsampling
        target = min(len(ys), self.max_cur_pts)
        if len(ys) > target:
            if self.sampling_strategy == "random":
                rng = np.random.RandomState(self.vo_random_seed)
                idx = rng.choice(len(ys), target, replace=False)
            else:  # "grid" (default)
                step = max(1, len(ys) // target)
                idx = np.arange(0, len(ys), step)[:target]
            ys, xs = ys[idx], xs[idx]

        Z = depth_np[ys, xs]

        if len(ys) < self.min_cur_pts:
            return torch.zeros((0, 3), device="cuda", dtype=torch.float32)

        Xc = np.zeros((len(ys), 3), dtype=np.float32)
        Xc[:, 0] = (xs - self.cx) * Z / self.fx
        Xc[:, 1] = (ys - self.cy) * Z / self.fy
        Xc[:, 2] = Z
        return torch.from_numpy(Xc).float().cuda()

    # ====================================================================
    # Tracking
    # ====================================================================

    def track(self, image_bgr, depth_np=None, init_c2w=None):
        """Estimate T_world_cur via damped Gauss-Newton on reference DT structure.

        Returns  (success, c2w: np.float64 (4,4), info: dict).
        """
        if self.opt_struct_pyr is None:
            return False, np.eye(4, dtype=np.float64), {
                "error": "no_reference", "dt_mean": float("inf"),
            }

        if init_c2w is None:
            init_c2w = np.linalg.inv(self.T_wr)  # world→ref inverted = ref→world ≈ cur

        # ---- 1.  Current-frame 3D points (camera frame) --------------------
        mask = self._compute_mask(image_bgr)
        X_cur = self._backproject_cur_mask(mask, depth_np)

        if X_cur.shape[0] < self.min_cur_pts:
            return False, np.eye(4, dtype=np.float64), {
                "error": "too_few_cur_pts", "n_cur": int(X_cur.shape[0]),
            }

        # ---- 2.  Initial T_ref_cur ------------------------------------------
        T_wc_init = init_c2w.astype(np.float64)
        T_rc_init = self.T_wr @ T_wc_init  # T_ref_cur

        # parameterise T_ref_cur = exp(xi) from actual initial guess
        T_rc_init_t = torch.from_numpy(T_rc_init).float().cuda()
        xi = _se3_log(T_rc_init_t)
        xi_init_for_diag = xi.clone()  # saved for delta-from-init diagnostics

        # ---- 3.  Coarse-to-fine LM -----------------------------------------
        num_levels = len(self.opt_struct_pyr)
        dt_mean = float("inf")
        n_vis = 0
        total_iters = 0

        # Per-level diagnostics accumulation
        lm_diag_agg = {"initial_error": float("inf"), "final_error": float("inf"),
                       "accepted_iters": 0, "rejected_iters": 0,
                       "inside_count": 0, "near_edge_count": 0}

        for level in range(num_levels):
            fx_l, fy_l, cx_l, cy_l = self.K_pyr[level]
            struct = self.opt_struct_pyr[level]

            max_it = (self.max_iters_coarse if level < num_levels - 1
                      else self.max_iters_fine)

            xi, n_vis, dt_mean, n_it, lm_diag = self._lm_optimise(
                X_cur, struct, fx_l, fy_l, cx_l, cy_l,
                xi.detach().clone(), max_it,
            )
            total_iters += n_it

            self.last_visible = n_vis
            self.last_dt_mean = dt_mean
            self.last_iters   = n_it

            # Accumulate diagnostics (finest level overwrites coarse)
            if level == 0:
                lm_diag_agg["initial_error"] = lm_diag["initial_error"]
            lm_diag_agg["final_error"] = lm_diag["final_error"]
            lm_diag_agg["accepted_iters"] += lm_diag["accepted_iters"]
            lm_diag_agg["rejected_iters"] += lm_diag["rejected_iters"]
            lm_diag_agg["inside_count"] = lm_diag["inside_count"]
            lm_diag_agg["near_edge_count"] = lm_diag["near_edge_count"]

            if dt_mean < self.min_dt_mean and level < num_levels - 1:
                if self.debug:
                    Log(f"[FFTEdgeVO] early conv level {level} dt_mean={dt_mean:.2f}px")
                break

        # ---- 4.  Final c2w --------------------------------------------------
        T_rc_opt = _se3_exp(xi).detach().cpu().numpy().astype(np.float64)
        c2w = np.linalg.inv(self.T_wr) @ T_rc_opt  # T_world_cur = T_world_ref @ T_ref_cur

        # ---- 5.  Quality ---------------------------------------------------
        visible_ratio = n_vis / max(1, X_cur.shape[0])
        min_visible = max(self.min_cur_pts, X_cur.shape[0] * self.require_visible_ratio)
        final_error = lm_diag_agg["final_error"]
        error_ok = (final_error < float("inf")
                    and final_error < self.dt_mean_fail * self.dt_mean_fail)

        # Compute simple diagnostics from LM data (no extra eval overhead)
        inside_count = lm_diag_agg["inside_count"]
        near_edge_count = lm_diag_agg["near_edge_count"]
        outside_ratio = (X_cur.shape[0] - inside_count) / max(1, X_cur.shape[0])
        near_edge_ratio = near_edge_count / max(1, inside_count)
        mask_density = mask.sum().item() / (self.H * self.W)
        ref_mask_density = (self.ref_count / (self.H * self.W)
                            if self.ref_count > 0 else 0.0)

        # EAGS-style good/bad ratio: bad_pts = outside FOV + inside but far from edge
        bad_pts = (X_cur.shape[0] - inside_count) + (inside_count - near_edge_count)
        good_bad_ratio = n_vis / max(1, bad_pts)

        success = (dt_mean < self.dt_mean_fail
                   and n_vis >= min_visible
                   and error_ok
                   and good_bad_ratio >= self.min_good_bad_ratio)

        # ---- 6.  Delta from init --------------------------------------------
        T_rc_init_for_diag = _se3_exp(xi_init_for_diag).detach().cpu().numpy().astype(np.float64)
        delta_T = np.linalg.inv(T_rc_init_for_diag) @ T_rc_opt
        delta_t = float(np.linalg.norm(delta_T[:3, 3]))
        delta_r_rad = float(np.arccos(
            np.clip((np.trace(delta_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
        delta_r_deg = delta_r_rad * 180.0 / np.pi

        # Build reject reason
        reject_reasons = []
        if dt_mean >= self.dt_mean_fail:
            reject_reasons.append("dt_mean_high")
        if n_vis < min_visible:
            reject_reasons.append("low_visible")
        if not error_ok:
            reject_reasons.append("high_error")
        if good_bad_ratio < self.min_good_bad_ratio:
            reject_reasons.append("good_bad_ratio_low")

        info = {
            "dt_mean": float(dt_mean) if not np.isinf(dt_mean) else float("inf"),
            "visible": int(n_vis),
            "total_cur": int(X_cur.shape[0]),
            "visible_ratio": float(visible_ratio),
            "iters": int(total_iters),
            "success": success,
            "n_mask_points": int(mask.sum().item()),
            "n_sampled_points": int(X_cur.shape[0]),
            "sampling": self.sampling_strategy,
            "initial_error": lm_diag_agg["initial_error"],
            "final_error": lm_diag_agg["final_error"],
            "accepted_iters": lm_diag_agg["accepted_iters"],
            "rejected_iters": lm_diag_agg["rejected_iters"],
            "inside_count": inside_count,
            "near_edge_count": near_edge_count,
            "delta_t_from_init": delta_t,
            "delta_r_deg_from_init": delta_r_deg,
            "outside_ratio": outside_ratio,
            "near_edge_ratio": near_edge_ratio,
            "good_bad_ratio": good_bad_ratio,
            "bad_pts": bad_pts,
            "mask_density": mask_density,
            "ref_mask_density": ref_mask_density,
            "reject_reason": " + ".join(reject_reasons) if reject_reasons else None,
        }

        if self.debug:
            Log(f"[FFTEdgeVO] track: dt_mean={dt_mean:.2f}px "
                f"good/bad={good_bad_ratio:.1f} "
                f"visible={n_vis}/{X_cur.shape[0]} "
                f"inside={inside_count} ne_ratio={near_edge_ratio:.2f} "
                f"outside_r={outside_ratio:.2f} "
                f"err={lm_diag_agg['initial_error']:.2f}→{final_error:.2f} "
                f"acc={lm_diag_agg['accepted_iters']}/{lm_diag_agg['rejected_iters']} "
                f"Δ_t={delta_t:.4f}m Δ_r={delta_r_deg:.2f}° "
                f"success={success}")

        # ---- Debug visualization (expensive: only when enabled) ------------
        if self.debug_save_images:
            self._debug_frame_count += 1
            if self._debug_frame_count == 1 or self._debug_frame_count % self.debug_save_interval == 0:
                try:
                    # Collect debug eval data on demand (expensive!)
                    fx_f, fy_f, cx_f, cy_f = self.K_pyr[-1]
                    struct_f = self.opt_struct_pyr[-1]
                    T_rc_t = _se3_exp(xi)
                    _, _, _, _, _, _, dbg_final = self._evaluate_residuals(
                        T_rc_t, X_cur, struct_f, fx_f, fy_f, cx_f, cy_f,
                        return_debug=True,
                    )
                    T_rc_init_t = _se3_exp(xi_init_for_diag)
                    _, _, _, _, _, _, dbg_init = self._evaluate_residuals(
                        T_rc_init_t, X_cur, struct_f, fx_f, fy_f, cx_f, cy_f,
                        return_debug=True,
                    )
                    self._save_debug_visualization(
                        image_bgr, mask, X_cur, xi_init_for_diag, xi,
                        dt_mean, n_vis, lm_diag_agg, info,
                        dbg_final=dbg_final, dbg_init=dbg_init,
                    )
                except Exception as e:
                    Log(f"[FFTEdgeVO] debug viz ERROR: {e}")

        return success, c2w.astype(np.float64), info

    # ====================================================================
    # Residual evaluation (shared by LM init and candidate re-evaluation)
    # ====================================================================

    def _evaluate_residuals(self, T, X_cur, opt_struct, fx, fy, cx, cy,
                            return_debug=False):
        """Evaluate weighted residuals at T_ref_cur.

        Returns (error, dt_mean, n_vis, inside_count, near_edge_count, data).

        data is a dict with keys (gx, gy, dt_vals, Xo, Yo, Zo, w) for
        Jacobian construction, or None when evaluation fails (too few points).

        When return_debug=True, also returns a debug dict with:
          u_raw, v_raw, dt_vals_all_inside, edge_ok, outside_count,
          dt_all_mean, dt_all_p50, dt_all_p75, dt_all_p90, dt_all_p95.
        """
        H, W = opt_struct.shape[0], opt_struct.shape[1]
        R = T[:3, :3]
        t = T[:3, 3]

        X_ref = (R @ X_cur.T).T + t                     # (N,3)
        Z = X_ref[:, 2]
        ok = Z > 0.05
        total_ok = int(ok.sum().item())
        if total_ok < 10:
            if return_debug:
                return float("inf"), float("inf"), 0, 0, 0, None, None
            return float("inf"), float("inf"), 0, 0, 0, None

        Xo, Yo, Zo = X_ref[ok, 0], X_ref[ok, 1], Z[ok]

        # Project — check inside BEFORE clamping (Bug 5 fix)
        u_raw = fx * Xo / Zo + cx
        v_raw = fy * Yo / Zo + cy
        inside = (u_raw >= 1.0) & (u_raw <= W - 2.0) & \
                 (v_raw >= 1.0) & (v_raw <= H - 2.0)
        inside_count = int(inside.sum().item())
        outside_count = total_ok - inside_count

        if inside_count < 10:
            if return_debug:
                dbg = {"outside_count": outside_count, "u_raw": None, "v_raw": None,
                       "dt_vals_all_inside": None, "edge_ok": None,
                       "dt_all_mean": float("inf"), "dt_all_p50": float("inf"),
                       "dt_all_p75": float("inf"), "dt_all_p90": float("inf"),
                       "dt_all_p95": float("inf")}
                return float("inf"), float("inf"), 0, inside_count, 0, None, dbg
            return float("inf"), float("inf"), 0, inside_count, 0, None

        u = u_raw[inside]
        v = v_raw[inside]
        Xo_in, Yo_in, Zo_in = Xo[inside], Yo[inside], Zo[inside]

        # Bilinear sample
        vals = _bilinear_sample_4ch(opt_struct, u, v)
        gx, gy, dt_vals_all, _ = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

        # ---- debug: all-inside DT statistics (before edge_ok filter) ---------
        debug_data = None
        if return_debug:
            dt_all_np = dt_vals_all.detach().cpu().numpy()
            dt_all_sorted = np.sort(dt_all_np)
            n_dt = len(dt_all_sorted)
            debug_data = {
                "u_raw": u_raw, "v_raw": v_raw, "inside_mask": inside,
                "dt_vals_all_inside": dt_vals_all,
                "outside_count": outside_count,
                "dt_all_mean": float(dt_all_np.mean()),
                "dt_all_p50": float(dt_all_sorted[int(n_dt * 0.50)] if n_dt > 1 else dt_all_np[0]),
                "dt_all_p75": float(dt_all_sorted[int(n_dt * 0.75)] if n_dt > 3 else dt_all_np[0]),
                "dt_all_p90": float(dt_all_sorted[int(n_dt * 0.90)] if n_dt > 9 else dt_all_np[0]),
                "dt_all_p95": float(dt_all_sorted[int(n_dt * 0.95)] if n_dt > 19 else dt_all_np[0]),
            }

        # Near-edge filter
        edge_ok = dt_vals_all < self.dt_huber * 3.0
        near_edge_count = int(edge_ok.sum().item())
        if near_edge_count < 10:
            if return_debug:
                return float("inf"), float("inf"), 0, inside_count, near_edge_count, None, debug_data
            return float("inf"), float("inf"), 0, inside_count, near_edge_count, None

        gx, gy, dt_vals = gx[edge_ok], gy[edge_ok], dt_vals_all[edge_ok]
        Xo_ne, Yo_ne, Zo_ne = Xo_in[edge_ok], Yo_in[edge_ok], Zo_in[edge_ok]

        # Huber weights
        abs_dt = torch.abs(dt_vals)
        w = torch.where(abs_dt <= self.dt_huber,
                        torch.ones_like(dt_vals),
                        self.dt_huber / (abs_dt + 1e-8))

        error = float((w * dt_vals * dt_vals).mean().item())
        dt_mean = float(dt_vals.mean().item())
        n_vis = near_edge_count

        # Update debug with edge_ok mask (on the inside subset)
        if debug_data is not None:
            debug_data["edge_ok"] = edge_ok
            debug_data["near_edge_count"] = near_edge_count

        data = {"gx": gx, "gy": gy, "dt_vals": dt_vals,
                "Xo": Xo_ne, "Yo": Yo_ne, "Zo": Zo_ne, "w": w}
        if return_debug:
            return error, dt_mean, n_vis, inside_count, near_edge_count, data, debug_data
        return error, dt_mean, n_vis, inside_count, near_edge_count, data

    # ====================================================================
    # Damped Gauss-Newton (LM-style) at one pyramid level
    # ====================================================================

    def _lm_optimise(self, X_cur, opt_struct, fx, fy, cx, cy, xi_init, max_iters):
        """Damped Gauss-Newton on SE(3).

        X_cur:  (N,3) current-camera-frame points  (GPU)
        opt_struct:  (H,W,4)  (gx, gy, dt, 0)       (GPU)
        xi:     6-vector, T_ref_cur = exp(xi)

        Returns  (xi_opt, n_visible, dt_mean, n_iters, lm_diag).
        """
        xi = xi_init.clone().detach().requires_grad_(False)
        H_img, W_img = opt_struct.shape[0], opt_struct.shape[1]

        lm_lambda = self.lm_lambda_init
        accepted_iters = 0
        rejected_iters = 0

        # ---- initial evaluation -------------------------------------------
        T_init = _se3_exp(xi)
        err_init, dt_init, nvis_init, inside_init, near_init, data_init = \
            self._evaluate_residuals(T_init, X_cur, opt_struct, fx, fy, cx, cy)

        best_error = err_init
        best_dt_mean = dt_init
        best_n_vis = nvis_init
        best_inside_count = inside_init
        best_near_edge_count = near_init
        best_xi = xi.clone()
        last_error = err_init

        initial_error = err_init

        if data_init is None:
            return best_xi.detach(), best_n_vis, best_dt_mean, 0, {
                "initial_error": initial_error, "final_error": best_error,
                "accepted_iters": 0, "rejected_iters": 0,
                "inside_count": inside_init, "near_edge_count": near_init,
            }

        # ---- LM loop -------------------------------------------------------
        for it in range(max_iters):
            # 1. Evaluate at current xi (build Jacobian)
            T_cur = _se3_exp(xi)
            err_cur, dt_cur, nvis_cur, inside_cur, near_cur, data_cur = \
                self._evaluate_residuals(T_cur, X_cur, opt_struct, fx, fy, cx, cy)

            if data_cur is None:
                break

            gx, gy, dt_vals, Xo, Yo, Zo, w = (
                data_cur["gx"], data_cur["gy"], data_cur["dt_vals"],
                data_cur["Xo"], data_cur["Yo"], data_cur["Zo"], data_cur["w"],
            )

            # ---- Build normal equations  J^T W J  Δξ = -J^T W r -----------
            z = 1.0 / Zo
            z2 = z * z
            px, py = Xo, Yo

            J0 = gx * z
            J1 = gy * z
            J2 = -(px * gx + py * gy) * z2
            J3 = -(px * py * z2) * gx - (1.0 + py * py * z2) * gy
            J4 = (1.0 + px * px * z2) * gx + (px * py * z2) * gy
            J5 = (-py * z) * gx + (px * z) * gy

            wr = w * dt_vals
            J_stack = torch.stack([J0, J1, J2, J3, J4, J5], dim=1)  # (M,6)

            H = (J_stack.unsqueeze(2) * J_stack.unsqueeze(1)
                 * w.unsqueeze(1).unsqueeze(2)).sum(dim=0)
            b = -(J_stack * wr.unsqueeze(1)).sum(dim=0)

            diag_H = torch.diag(H)
            H_damped = H + lm_lambda * torch.diag(diag_H)

            try:
                inc = torch.linalg.solve(H_damped, b)
            except torch.linalg.LinAlgError:
                break

            if not torch.isfinite(inc).all():
                break

            # 2. Re-evaluate at candidate  T_new = exp(inc) · T_cur (Bug 4 fix)
            T_new = _se3_exp(inc) @ T_cur
            err_new, dt_new, nvis_new, inside_new, near_new, data_new = \
                self._evaluate_residuals(T_new, X_cur, opt_struct, fx, fy, cx, cy)

            # 3. Accept / reject based on candidate error (Bug 4 fix)
            if err_new < err_cur and err_new < float("inf"):
                # Accept
                xi = _se3_log(T_new)
                accepted_iters += 1

                # Update best state AFTER accept (Bug 2 fix)
                if err_new < best_error:
                    best_error = err_new
                    best_dt_mean = dt_new
                    best_n_vis = nvis_new
                    best_inside_count = inside_new
                    best_near_edge_count = near_new
                    best_xi = xi.clone()

                # Convergence: check improvement ratio BEFORE overwriting last_error (Bug 1 fix)
                if last_error < float("inf"):
                    improvement_ratio = err_new / last_error
                    if improvement_ratio > self.conv_eps:
                        last_error = err_new
                        break

                last_error = err_new
                lm_lambda *= 0.5
            else:
                # Reject
                rejected_iters += 1
                lm_lambda *= 4.0
                if lm_lambda > 100:
                    break

            # Step-size convergence
            if inc.dot(inc) < 0.5 and it > 2:
                break

        lm_diag = {
            "initial_error": initial_error,
            "final_error": best_error,
            "accepted_iters": accepted_iters,
            "rejected_iters": rejected_iters,
            "inside_count": best_inside_count,
            "near_edge_count": best_near_edge_count,
        }
        return best_xi.detach(), best_n_vis, best_dt_mean, accepted_iters + rejected_iters, lm_diag


    # ====================================================================
    # Debug visualization
    # ====================================================================

    def _draw_projection_points(self, bg_img, u_raw, v_raw, inside_mask,
                                 edge_ok_on_inside, dt_vals_inside, H_l, W_l,
                                 mode="all_inside"):
        """Draw projected points on reference background.

        mode: "all_inside" | "used_near_edge" | "residual_colormap"
        Returns BGR image.
        """
        proj_img = bg_img.copy()
        u_np = u_raw.detach().cpu().numpy()
        v_np = v_raw.detach().cpu().numpy()
        inside_np = inside_mask.detach().cpu().numpy()

        u_inside = u_np[inside_np]
        v_inside = v_np[inside_np]

        if mode == "all_inside":
            # Green: inside FOV
            u_in = np.clip(u_inside, 0, W_l - 1).astype(int)
            v_in = np.clip(v_inside, 0, H_l - 1).astype(int)
            if len(u_in) > 0:
                proj_img[v_in, u_in] = [0, 255, 0]
            # Blue: outside FOV
            u_out = u_np[~inside_np]
            v_out = v_np[~inside_np]
            u_out_c = np.clip(u_out, 0, W_l - 1).astype(int)
            v_out_c = np.clip(v_out, 0, H_l - 1).astype(int)
            if len(u_out_c) > 0:
                proj_img[v_out_c, u_out_c] = [255, 0, 0]

        elif mode == "used_near_edge":
            if edge_ok_on_inside is not None and dt_vals_inside is not None:
                eo_np = edge_ok_on_inside.detach().cpu().numpy()
                u_ne = u_inside[eo_np]
                v_ne = v_inside[eo_np]
                u_ne_c = np.clip(u_ne, 0, W_l - 1).astype(int)
                v_ne_c = np.clip(v_ne, 0, H_l - 1).astype(int)
                if len(u_ne_c) > 0:
                    proj_img[v_ne_c, u_ne_c] = [0, 255, 255]  # yellow: near-edge (used)
                # Red: inside but NOT near-edge (filtered out)
                u_fe = u_inside[~eo_np]
                v_fe = v_inside[~eo_np]
                u_fe_c = np.clip(u_fe, 0, W_l - 1).astype(int)
                v_fe_c = np.clip(v_fe, 0, H_l - 1).astype(int)
                if len(u_fe_c) > 0:
                    proj_img[v_fe_c, u_fe_c] = [0, 0, 255]  # red: inside but far from edge

        elif mode == "residual_colormap":
            if dt_vals_inside is not None:
                dt_np = dt_vals_inside.detach().cpu().numpy()
                # Color by DT value
                # Green:  dt < 2
                # Yellow: 2 ≤ dt < 5
                # Orange: 5 ≤ dt < 15
                # Red:    dt ≥ 15
                mask_g = dt_np < 2
                mask_y = (dt_np >= 2) & (dt_np < 5)
                mask_o = (dt_np >= 5) & (dt_np < 15)
                mask_r = dt_np >= 15
                for mask, color in [(mask_g, [0, 255, 0]),
                                     (mask_y, [0, 255, 255]),
                                     (mask_o, [0, 165, 255]),
                                     (mask_r, [0, 0, 255])]:
                    if mask.sum() > 0:
                        u_m = np.clip(u_inside[mask], 0, W_l - 1).astype(int)
                        v_m = np.clip(v_inside[mask], 0, H_l - 1).astype(int)
                        proj_img[v_m, u_m] = color

        return proj_img

    def _save_debug_visualization(self, image_bgr, mask, X_cur,
                                   xi_init, xi_final, dt_mean, n_vis,
                                   lm_diag, info,
                                   dbg_final=None, dbg_init=None):
        """Save debug images for FFTEdgeVO diagnostics.

        Outputs (to self.debug_save_dir / frame_{id}_*.png):
          1. fft_mask.png            — current frame FFT mask overlay
          2. ref_dt.png              — reference DT heatmap
          3. projection_all_inside.png     — all inside (green) + outside (blue)
          4. projection_used_near_edge.png — near_edge used (yellow) vs filtered (red)
          5. projection_residual_colormap.png — DT color: g<2, y 2-5, o 5-15, r≥15
          6. projection_before_after.png  — left=init, right=final
          7. diagnostics.png              — bar chart
        """
        frame_id = info.get("frame_id", self._debug_frame_count)
        save_dir = self.debug_save_dir
        os.makedirs(save_dir, exist_ok=True)
        prefix = os.path.join(save_dir, f"frame_{frame_id:04d}")

        has_ref = self.ref_rgb_bgr is not None
        H_l = self.ref_rgb_bgr.shape[0] if has_ref else self.H
        W_l = self.ref_rgb_bgr.shape[1] if has_ref else self.W

        # --- 1. FFT mask overlay on current RGB -----------------------------
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
        overlay = image_bgr.copy()
        overlay[mask_np > 0] = [0, 255, 0]
        blended = cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)
        cv2.imwrite(f"{prefix}_fft_mask.png", blended)

        # --- 2. Reference DT heatmap ----------------------------------------
        ref_mask_np = None
        if has_ref:
            ref_mask = self._compute_mask(self.ref_rgb_bgr)
            ref_mask_np = ref_mask.cpu().numpy().astype(np.uint8) * 255
            dt_full = cv2.distanceTransform(
                255 - ref_mask_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            dt_full = np.clip(dt_full, 0.0, self.dt_max)
            dt_vis = (dt_full / max(self.dt_max, 1.0) * 255).astype(np.uint8)
            dt_color = cv2.applyColorMap(dt_vis, cv2.COLORMAP_JET)
            cv2.imwrite(f"{prefix}_ref_dt.png", dt_color)

        if not has_ref:
            return

        ref_bg = self.ref_rgb_bgr.copy()
        # Semi-transparent reference mask overlay (green contour on projection bg)
        if ref_mask_np is not None:
            ref_overlay = ref_bg.copy()
            ref_overlay[ref_mask_np > 0] = [0, 255, 0]
            ref_bg = cv2.addWeighted(ref_bg, 0.85, ref_overlay, 0.15, 0)

        # --- 3. projection_all_inside ---------------------------------------
        if dbg_final is not None and dbg_final.get("u_raw") is not None:
            img3 = self._draw_projection_points(
                ref_bg, dbg_final["u_raw"], dbg_final["v_raw"],
                dbg_final["inside_mask"], None, None, H_l, W_l,
                mode="all_inside")
            cv2.imwrite(f"{prefix}_projection_all_inside.png", img3)

        # --- 4. projection_used_near_edge -----------------------------------
        if dbg_final is not None and dbg_final.get("u_raw") is not None:
            img4 = self._draw_projection_points(
                ref_bg, dbg_final["u_raw"], dbg_final["v_raw"],
                dbg_final["inside_mask"],
                dbg_final.get("edge_ok"),
                dbg_final.get("dt_vals_all_inside"),
                H_l, W_l,
                mode="used_near_edge")
            cv2.imwrite(f"{prefix}_projection_used_near_edge.png", img4)

        # --- 5. projection_residual_colormap --------------------------------
        if dbg_final is not None and dbg_final.get("u_raw") is not None:
            img5 = self._draw_projection_points(
                ref_bg, dbg_final["u_raw"], dbg_final["v_raw"],
                dbg_final["inside_mask"],
                None,
                dbg_final.get("dt_vals_all_inside"),
                H_l, W_l,
                mode="residual_colormap")
            cv2.imwrite(f"{prefix}_projection_residual_colormap.png", img5)

        # --- 6. projection_before_after -------------------------------------
        if dbg_init is not None and dbg_final is not None:
            img_left = self._draw_projection_points(
                ref_bg, dbg_init["u_raw"], dbg_init["v_raw"],
                dbg_init["inside_mask"], None, None, H_l, W_l,
                mode="all_inside")
            img_right = self._draw_projection_points(
                ref_bg, dbg_final["u_raw"], dbg_final["v_raw"],
                dbg_final["inside_mask"], None, None, H_l, W_l,
                mode="all_inside")
            # Label left/right
            cv2.putText(img_left, "init", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img_right, "final", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            img_ba = np.hstack([img_left, img_right])
            cv2.imwrite(f"{prefix}_projection_before_after.png", img_ba)

        # --- 7. Diagnostics bar chart (matplotlib, optional) -----------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 5))
            metrics = [
                ("dt_mean", dt_mean),
                ("dt_all_p90", info.get("dt_all_p90", 0)),
                ("dt_all_p95", info.get("dt_all_p95", 0)),
                ("visible", n_vis),
                ("inside", lm_diag.get("inside_count", 0)),
                ("near_edge", lm_diag.get("near_edge_count", 0)),
                ("ne_ratio", info.get("near_edge_ratio", 0)),
                ("outside_r", info.get("outside_ratio", 0)),
                ("acc_iters", lm_diag.get("accepted_iters", 0)),
                ("rej_iters", lm_diag.get("rejected_iters", 0)),
                ("Δ_t(mm)", info.get("delta_t_from_init", 0) * 1000),
                ("Δ_r(deg)", info.get("delta_r_deg_from_init", 0)),
            ]
            names = [m[0] for m in metrics]
            values = [m[1] for m in metrics]
            bars = ax.bar(names, values)
            ax.bar_label(bars, fmt="%.2f", fontsize=6)
            ax.set_title(f"FFTEdgeVO Frame {frame_id}  "
                         f"err {lm_diag.get('initial_error',0):.1f}→{lm_diag.get('final_error',0):.1f}  "
                         f"success={info.get('success',False)}")
            ax.tick_params(axis="x", rotation=45, labelsize=6)
            fig.tight_layout()
            fig.savefig(f"{prefix}_diagnostics.png", dpi=100)
            plt.close(fig)
        except Exception:
            pass  # matplotlib unavailable — skip chart, images 1-6 still saved

        if self.debug:
            Log(f"[FFTEdgeVO] debug images saved to {prefix}_*.png")


# ============================================================================
# SE(3) logarithm & bilinear sampling
# ============================================================================

def _se3_log(T):
    """SE(3) logarithm — inverse of se3_exp.  Returns xi ∈ R^6."""
    R = T[:3, :3]
    t = T[:3, 3]

    # Rotation → so(3)
    cos_theta = ((torch.trace(R) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    if theta < 1e-10:
        omega = torch.zeros(3, device=T.device, dtype=T.dtype)
        V_inv = torch.eye(3, device=T.device, dtype=T.dtype)
    else:
        sin_theta = torch.sin(theta)
        omega = theta / (2.0 * sin_theta) * torch.tensor(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            device=T.device, dtype=T.dtype,
        )
        K = _skew(omega / theta)
        V_inv = (torch.eye(3, device=T.device, dtype=T.dtype)
                 - 0.5 * K
                 + (1.0 / (theta * theta) - (1.0 + cos_theta) / (2.0 * theta * sin_theta)) * (K @ K))

    v = V_inv @ t
    return torch.cat([omega, v])


def _bilinear_sample_4ch(im4, u, v):
    """Bilinear sample of a (H,W,4) tensor at fractional pixel coords (u,v).

    Returns (N,4) tensor.
    """
    H, W = im4.shape[0], im4.shape[1]
    u0 = u.floor().long().clamp(0, W - 1)
    v0 = v.floor().long().clamp(0, H - 1)
    u1 = (u0 + 1).clamp(0, W - 1)
    v1 = (v0 + 1).clamp(0, H - 1)

    du = u - u0.float()
    dv = v - v0.float()

    w00 = ((1.0 - du) * (1.0 - dv)).unsqueeze(1)
    w01 = (du * (1.0 - dv)).unsqueeze(1)
    w10 = ((1.0 - du) * dv).unsqueeze(1)
    w11 = (du * dv).unsqueeze(1)

    idx00 = v0 * W + u0
    idx01 = v0 * W + u1
    idx10 = v1 * W + u0
    idx11 = v1 * W + u1

    im_flat = im4.reshape(-1, 4)

    return w00 * im_flat[idx00] + w01 * im_flat[idx01] + w10 * im_flat[idx10] + w11 * im_flat[idx11]
