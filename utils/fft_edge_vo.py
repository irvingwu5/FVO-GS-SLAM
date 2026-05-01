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

        # ---- quality --------------------------------------------------------
        self.dt_mean_fail       = float(cfg.get("dt_mean_fail_threshold", 15.0))
        self.require_visible_ratio = float(cfg.get("require_visible_ratio", 0.3))

        # ---- FFT filter (lazy) ----------------------------------------------
        self.fft_filter = None

        # ---- reference state ------------------------------------------------
        self.opt_struct_pyr = None   # list of torch float32 (H_lvl, W_lvl, 4)
        self.K_pyr          = None   # list of (fx, fy, cx, cy) per level
        self.T_wr           = None   # 4×4 world→ref camera, numpy float64
        self.ref_count      = 0

        # ---- diagnostics ----------------------------------------------------
        self.last_dt_mean = float("inf")
        self.last_visible = 0
        self.last_iters   = 0

        # ----
        self.debug       = cfg.get("debug_log", True)
        self.min_dt_mean = float(cfg.get("min_dt_mean", 0.5))

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

    def set_reference(self, image_bgr, depth_np, c2w):
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
        self.T_wr = np.linalg.inv(c2w.astype(np.float64))  # world→ref (w2c)
        self.ref_count = n_mask

        if self.debug:
            Log(f"[FFTEdgeVO] reference: {n_mask} mask px, "
                f"{len(self.opt_struct_pyr)} pyramid levels, T_wr ready")
        return True

    # ====================================================================
    # Current-frame 3D points (camera frame, NOT world)
    # ====================================================================

    def _backproject_cur_mask(self, mask, depth_np):
        """Backproject current-frame FFT-mask pixels to 3D in CURRENT camera frame.

        Returns  torch float32 (N,3) on CUDA, empty if too few valid points.
        """
        mask_np = mask.cpu().numpy()
        ys, xs = np.where(mask_np)

        if len(ys) == 0:
            return torch.zeros((0, 3), device="cuda", dtype=torch.float32)

        # random subset
        target = min(len(ys), self.max_cur_pts)
        if len(ys) > target:
            idx = np.random.choice(len(ys), target, replace=False)
            ys, xs = ys[idx], xs[idx]

        Z = depth_np[ys, xs]
        ok = (Z > 0.1) & (Z < 8.0)
        ys, xs, Z = ys[ok], xs[ok], Z[ok]

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

        # ---- 3.  Coarse-to-fine LM -----------------------------------------
        num_levels = len(self.opt_struct_pyr)
        dt_mean = float("inf")
        n_vis = 0
        total_iters = 0

        for level in range(num_levels):
            fx_l, fy_l, cx_l, cy_l = self.K_pyr[level]
            struct = self.opt_struct_pyr[level]

            max_it = (self.max_iters_coarse if level < num_levels - 1
                      else self.max_iters_fine)

            xi, n_vis, dt_mean, n_it = self._lm_optimise(
                X_cur, struct, fx_l, fy_l, cx_l, cy_l,
                xi.detach().clone(), max_it,
            )
            total_iters += n_it

            self.last_visible = n_vis
            self.last_dt_mean = dt_mean
            self.last_iters   = n_it

            if dt_mean < self.min_dt_mean and level < num_levels - 1:
                if self.debug:
                    Log(f"[FFTEdgeVO] early conv level {level} dt_mean={dt_mean:.2f}px")
                break

        # ---- 4.  Final c2w --------------------------------------------------
        T_rc_opt = _se3_exp(xi).detach().cpu().numpy().astype(np.float64)
        c2w = np.linalg.inv(self.T_wr) @ T_rc_opt  # T_world_cur = T_world_ref @ T_ref_cur

        # ---- 5.  Quality ---------------------------------------------------
        success = (dt_mean < self.dt_mean_fail
                   and n_vis >= self.min_cur_pts * self.require_visible_ratio)

        info = {
            "dt_mean": float(dt_mean) if not np.isinf(dt_mean) else float("inf"),
            "visible": int(n_vis),
            "total_cur": int(X_cur.shape[0]),
            "visible_ratio": float(n_vis / max(1, X_cur.shape[0])),
            "iters": int(total_iters),
            "success": success,
        }

        if self.debug:
            Log(f"[FFTEdgeVO] track: dt_mean={dt_mean:.2f}px "
                f"visible={n_vis}/{X_cur.shape[0]} "
                f"iters={total_iters} success={success}")

        return success, c2w.astype(np.float64), info

    # ====================================================================
    # Damped Gauss-Newton (LM-style) at one pyramid level
    # ====================================================================

    def _lm_optimise(self, X_cur, opt_struct, fx, fy, cx, cy, xi_init, max_iters):
        """Damped Gauss-Newton on SE(3).

        X_cur:  (N,3) current-camera-frame points  (GPU)
        opt_struct:  (H,W,4)  (gx, gy, dt, 0)       (GPU)
        xi:     6-vector, T_ref_cur = exp(xi)

        Returns  (xi_opt, n_visible, dt_mean, n_iters).
        """
        xi = xi_init.clone().detach().requires_grad_(False)
        H, W, _ = opt_struct.shape

        lm_lambda = self.lm_lambda_init
        last_error = float("inf")
        best_xi = xi.clone()
        best_dt_mean = float("inf")
        best_n_vis = 0

        for it in range(max_iters):
            T = _se3_exp(xi)                              # T_ref_cur
            R = T[:3, :3]
            t = T[:3, 3]

            # Transform cur→ref: X_ref = R * X_cur + t
            X_ref = (R @ X_cur.T).T + t                  # (N,3)

            Z = X_ref[:, 2]
            ok = Z > 0.05
            if ok.sum() < 10:
                break

            Xo, Yo, Zo = X_ref[ok, 0], X_ref[ok, 1], Z[ok]

            # Project into reference image
            u = fx * Xo / Zo + cx
            v = fy * Yo / Zo + cy

            # Clamp to valid image region
            u = u.clamp(0.5, W - 1.5)
            v = v.clamp(0.5, H - 1.5)

            # Bilinear sample opt_struct (gx, gy, dt, _)
            vals = _bilinear_sample_4ch(opt_struct, u, v)
            gx, gy, dt_vals, _ = vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]

            # Skip points beyond edge distance
            edge_ok = dt_vals < self.dt_huber * 3.0
            if edge_ok.sum() < 10:
                break

            gx, gy, dt_vals = gx[edge_ok], gy[edge_ok], dt_vals[edge_ok]
            Xo, Yo, Zo = Xo[edge_ok], Yo[edge_ok], Zo[edge_ok]

            # Huber weights
            abs_dt = torch.abs(dt_vals)
            w = torch.where(abs_dt <= self.dt_huber,
                            torch.ones_like(dt_vals),
                            self.dt_huber / (abs_dt + 1e-8))

            # Weighted error
            error = float((w * dt_vals * dt_vals).mean().item())
            n_vis = int(edge_ok.sum().item())
            dt_mean = float(dt_vals.mean().item())

            if error < best_dt_mean:
                best_dt_mean = dt_mean
                best_n_vis = n_vis
                best_xi = xi.clone()

            # ---- Build normal equations  J^T W J  Δξ = -J^T W r -----------
            z = 1.0 / Zo
            z2 = z * z
            px, py = Xo, Yo

            # Jacobian rows (Kerl 2012, p.34 — gx, gy already pre-multiplied by fx, fy)
            J0 = gx * z                                      # v_x
            J1 = gy * z                                      # v_y
            J2 = -(px * gx + py * gy) * z2                   # v_z
            J3 = -(px * py * z2) * gx - (1.0 + py * py * z2) * gy   # ω_x
            J4 = (1.0 + px * px * z2) * gx + (px * py * z2) * gy    # ω_y
            J5 = (-py * z) * gx + (px * z) * gy             # ω_z

            # Weighted Jacobian and residual
            w_gx = w * gx
            w_gy = w * gy
            wr = w * dt_vals

            # Accumulate 6×6 Hessian and 6×1 rhs (loop-free, batched)
            # J^T W J  = sum_i w_i J_i^T J_i  (outer product)
            # J^T W r  = sum_i w_i J_i^T r_i
            J_stack = torch.stack([J0, J1, J2, J3, J4, J5], dim=1)  # (M, 6)

            # Weighted normal equations
            H = (J_stack.unsqueeze(2) * J_stack.unsqueeze(1) * w.unsqueeze(1).unsqueeze(2)).sum(dim=0)  # (6,6)
            b = -(J_stack * wr.unsqueeze(1)).sum(dim=0)  # (6,)

            # LM damping:  H_damped = H + λ * diag(H)
            diag_H = torch.diag(H)
            H_damped = H + lm_lambda * torch.diag(diag_H)

            # Solve
            try:
                inc = torch.linalg.solve(H_damped, b)
            except torch.linalg.LinAlgError:
                break

            if not torch.isfinite(inc).all():
                break

            # Update:  T_new = exp(inc) * T_old
            # This corresponds to left-multiplicative increment Δξ
            xi_new = _se3_log(_se3_exp(inc) @ T)  # I'll compute this more efficiently

            # Actually, for small inc, T_new ≈ exp(inc) * T_old,
            # and xi_new is the log of that.  For the next iteration we just need
            # the new xi.  But actually, it's easier: we're parameterising as
            # T = exp(xi), and updating T ← exp(inc)·T.
            # We need the new xi such that exp(xi_new) = exp(inc)·exp(xi_old).
            # Rather than computing the Baker-Campbell-Hausdorff formula, we
            # just compute the matrix product and take its log (inexpensive).

            # Simple update: evaluate T_new = exp(inc) @ exp(xi_old), then take log
            T_new = _se3_exp(inc) @ T

            # Check error reduction
            if error < last_error:
                # Accept
                xi = _se3_log(T_new)
                last_error = error
                lm_lambda *= 0.5

                if error / max(last_error, 1e-10) > self.conv_eps:
                    break
            else:
                # Reject
                lm_lambda *= 4.0
                if lm_lambda > 100:
                    break

            # Convergence check
            if inc.dot(inc) < 0.5 and it > 2:
                break

        return best_xi.detach(), best_n_vis, best_dt_mean, it + 1


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
