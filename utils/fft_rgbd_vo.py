# utils/fft_rgbd_vo.py
# FFT 高频区域辅助的 RGB-D Visual Odometry 初值估计器
# 帧间跟踪：FAST 角点 + Lucas-Kanade 金字塔光流（对动态模糊和反光更鲁棒）

import cv2
import torch
import numpy as np

from utils.fft_filter import FFTFrequencyFilter
from utils.logging_utils import Log


class FFTGuidedRGBDVO:
    def __init__(self, config):
        fft_cfg = config.get("FFTVO", {})

        self.use_fft_vo_init = fft_cfg.get("use_fft_vo_init", False)

        # ---- 特征点参数 ----
        self.max_features = int(fft_cfg.get("max_features", 1200))
        self.min_features = int(fft_cfg.get("min_features", 80))

        # ---- 光流与 PnP 参数 ----
        self.min_tracked = int(fft_cfg.get("min_tracked", 40))
        self.min_inliers = int(fft_cfg.get("min_inliers", 25))
        self.min_inlier_ratio = float(fft_cfg.get("min_inlier_ratio", 0.45))
        self.min_depth = float(fft_cfg.get("min_depth", 0.1))
        self.max_depth = float(fft_cfg.get("max_depth", 5.0))
        self.ransac_reproj_error = float(fft_cfg.get("ransac_reproj_error", 3.0))
        self.ransac_confidence = float(fft_cfg.get("ransac_confidence", 0.999))
        self.ransac_iterations = int(fft_cfg.get("ransac_iterations", 100))

        # ---- LK 光流参数 ----
        self.lk_win_size = (21, 21)
        self.lk_max_level = 3
        self.lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        # 最小特征值阈值，过滤低纹理角点
        self.lk_min_eig_threshold = 0.001

        # ---- 畸变参数（PnP 可选使用） ----
        calib = config.get("Dataset", {}).get("Calibration", {})
        self.use_distortion_in_pnp = fft_cfg.get(
            "use_distortion_in_pnp", calib.get("distorted", False)
        )
        if self.use_distortion_in_pnp:
            self.dist_coeffs = np.array([
                calib.get("k1", 0.0), calib.get("k2", 0.0),
                calib.get("p1", 0.0), calib.get("p2", 0.0),
                calib.get("k3", 0.0),
            ], dtype=np.float64)
        else:
            self.dist_coeffs = None

        # ---- 一致性检查参数 ----
        self.max_init_translation = float(fft_cfg.get("max_init_translation", 0.35))
        self.max_init_rotation_deg = float(fft_cfg.get("max_init_rotation_deg", 25.0))

        # ---- 调试 ----
        self.debug_log = fft_cfg.get("debug_log", False)

        # ---- FFT 滤波器（延迟初始化） ----
        self._fft_filter = None

        # ---- 缓存上一帧的特征点，用于光流跟踪（首次不缓存） ----
        self._prev_kp_pts = None   # [N, 2] float32, 上一帧在 FFT mask 内的角点像素坐标
        self._prev_gray = None     # 上一帧灰度图

        # ---- EAGS 风格：refined pose cache ----
        self.pose_cache = {}       # frame_id → c2w tensor [4, 4] on CUDA

    # ========================================================================
    # 对外接口
    # ========================================================================
    def update_pose(self, frame_id, refined_c2w):
        """EAGS 风格：保存 render tracking refinement 后的最终 pose。
        FFTVO 后续估计必须优先使用 refined pose 作为上一帧全局位姿基准。
        """
        if isinstance(refined_c2w, np.ndarray):
            refined_c2w = torch.from_numpy(refined_c2w.astype(np.float32))
        self.pose_cache[frame_id] = refined_c2w.float().cuda().detach()
        if self.debug_log:
            Log(f"[FFTVO] pose feedback updated: frame={frame_id}")
        # 限制缓存大小
        if len(self.pose_cache) > 100:
            oldest = min(self.pose_cache.keys())
            del self.pose_cache[oldest]

    def estimate(self, prev_cam, cur_cam, prev_c2w):
        """
        输入:
            prev_cam: 上一帧 Camera 对象
            cur_cam:  当前帧 Camera 对象
            prev_c2w: 上一帧 camera-to-world 矩阵, torch.Tensor 或 np.ndarray, shape [4, 4]

        输出:
            success: bool
            init_c2w: torch.Tensor, shape [4, 4], float32, on CUDA
            info: dict
        """
        # ---- 0. 类型统一 + EAGS 风格 refined pose feedback ----
        if isinstance(prev_c2w, np.ndarray):
            prev_c2w = torch.from_numpy(prev_c2w.astype(np.float32))
        prev_c2w = prev_c2w.float().cuda()

        # 优先使用 render tracking feedback 的 refined pose
        prev_uid = getattr(prev_cam, "uid", -1)
        if prev_uid >= 0 and prev_uid in self.pose_cache:
            prev_c2w = self.pose_cache[prev_uid].clone()

        # ---- 1. 读取 RGB 图像 ----
        rgb_prev_np = self._camera_rgb_to_numpy(prev_cam)
        rgb_cur_np = self._camera_rgb_to_numpy(cur_cam)
        if rgb_prev_np is None or rgb_cur_np is None:
            return False, torch.eye(4, device="cuda"), {"error": "missing_rgb"}

        H, W = rgb_prev_np.shape[:2]

        # ---- 2. 灰度图（光流需要单通道） ----
        gray_prev = cv2.cvtColor(rgb_prev_np, cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.cvtColor(rgb_cur_np, cv2.COLOR_BGR2GRAY)

        # ---- 3. 读取深度 ----
        depth_prev = prev_cam.depth
        if depth_prev is None:
            return False, torch.eye(4, device="cuda"), {"error": "missing_depth"}

        # ---- 4. 获取 / 生成 FFT 高频 mask ----
        freq_mask_prev = self._get_or_create_freq_mask(prev_cam)
        freq_mask_cur = self._get_or_create_freq_mask(cur_cam)
        freq_mask = (freq_mask_prev & freq_mask_cur).cpu().numpy().astype(np.uint8) * 255

        # ---- 5. 在 FFT mask 内检测 FAST 角点作为光流种子（仅上一帧需要） ----
        #      缓存策略：如果 prev_cam 的 uid 和缓存的一致则复用，否则重新检测
        prev_uid = getattr(prev_cam, "uid", -1)
        if self._prev_kp_pts is None or getattr(self, "_cached_uid", -1) != prev_uid:
            fast = cv2.FastFeatureDetector.create(
                threshold=15, nonmaxSuppression=True
            )
            kp_prev = fast.detect(gray_prev, mask=freq_mask)
            if kp_prev is None or len(kp_prev) < self.min_features:
                self._prev_kp_pts = None
                self._prev_gray = None
                self._cached_uid = -1
                return False, torch.eye(4, device="cuda"), {
                    "error": "too_few_features",
                    "kp_prev": len(kp_prev) if kp_prev is not None else 0,
                }

            # 按响应强度排序，限制最大数量
            kp_prev = sorted(kp_prev, key=lambda k: k.response, reverse=True)
            kp_prev = kp_prev[:self.max_features]

            self._prev_kp_pts = np.float32([kp.pt for kp in kp_prev]).reshape(-1, 1, 2)
            self._prev_gray = gray_prev.copy()
            self._cached_uid = prev_uid

        prev_pts = self._prev_kp_pts  # [N, 1, 2]
        n_prev = len(prev_pts)

        # ---- 6. LK 金字塔光流跟踪 prev → cur ----
        cur_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray_cur, prev_pts, None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=self.lk_criteria,
            minEigThreshold=self.lk_min_eig_threshold,
        )

        # status==1 表示跟踪成功
        status = status.ravel()
        err = err.ravel()
        tracked_mask = status == 1
        n_tracked = int(tracked_mask.sum())

        if n_tracked < self.min_tracked:
            return False, torch.eye(4, device="cuda"), {
                "error": "too_few_tracked",
                "n_tracked": n_tracked,
                "n_prev": n_prev,
            }

        # 过滤出成功跟踪的点
        pts_prev_valid = prev_pts[tracked_mask].reshape(-1, 2)  # [M, 2]
        pts_cur_valid = cur_pts[tracked_mask].reshape(-1, 2)    # [M, 2]

        # ---- 7. 用上一帧 depth 将跟踪点反投影为 3D 点 ----
        fx, fy = cur_cam.fx, cur_cam.fy
        cx, cy = cur_cam.cx, cur_cam.cy

        pts_3d, valid_mask = self._backproject(
            pts_prev_valid, depth_prev, fx, fy, cx, cy
        )

        n_valid_3d = int(valid_mask.sum())
        if n_valid_3d < self.min_tracked:
            return False, torch.eye(4, device="cuda"), {
                "error": "too_few_valid_depth",
                "n_valid_3d": n_valid_3d,
                "n_tracked": n_tracked,
            }

        pts_3d_ok = pts_3d[valid_mask]          # [K, 3]
        pts_cur_ok = pts_cur_valid[valid_mask]  # [K, 2]

        # ---- 8. solvePnPRansac ----
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        success_pnp, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d_ok, pts_cur_ok, K, self.dist_coeffs,
            reprojectionError=self.ransac_reproj_error,
            confidence=self.ransac_confidence,
            iterationsCount=self.ransac_iterations,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success_pnp or inliers is None:
            return False, torch.eye(4, device="cuda"), {
                "error": "pnp_failed",
                "n_3d": n_valid_3d,
            }

        n_inliers = len(inliers)
        inlier_ratio = n_inliers / max(1, n_valid_3d)

        if n_inliers < self.min_inliers:
            return False, torch.eye(4, device="cuda"), {
                "error": "too_few_inliers",
                "n_inliers": n_inliers,
                "n_3d": n_valid_3d,
                "inlier_ratio": inlier_ratio,
            }

        if inlier_ratio < self.min_inlier_ratio:
            return False, torch.eye(4, device="cuda"), {
                "error": "low_inlier_ratio",
                "n_inliers": n_inliers,
                "n_3d": n_valid_3d,
                "inlier_ratio": inlier_ratio,
            }

        # ---- 9. 构造 T_cur_prev ----
        #      solvePnPRansac 解出的 (rvec, tvec) 满足:
        #          X_cur = R @ X_prev + t
        #      即 T_cur_prev = [R | t; 0 | 1]
        R_cur_prev, _ = cv2.Rodrigues(rvec)
        T_cur_prev = np.eye(4, dtype=np.float32)
        T_cur_prev[:3, :3] = R_cur_prev
        T_cur_prev[:3, 3] = tvec.ravel()

        # ---- 10. 全局位姿：c2w_cur = c2w_prev @ inv(T_cur_prev) ----
        T_cur_prev_t = torch.from_numpy(T_cur_prev).float().cuda()
        T_prev_cur = torch.linalg.inv(T_cur_prev_t)
        init_c2w = prev_c2w @ T_prev_cur

        # ---- 11. 一致性检查 ----
        delta_t = torch.norm(T_cur_prev_t[:3, 3])
        delta_R = T_cur_prev_t[:3, :3]
        trace_val = torch.clamp((torch.trace(delta_R) - 1.0) * 0.5, -1.0, 1.0)
        delta_deg = torch.acos(trace_val).item() * 180.0 / np.pi

        if delta_t > self.max_init_translation or delta_deg > self.max_init_rotation_deg:
            return False, torch.eye(4, device="cuda"), {
                "error": "consistency_check_failed",
                "delta_t": float(delta_t),
                "delta_deg": float(delta_deg),
            }

        info = {
            "n_prev": n_prev,
            "n_tracked": n_tracked,
            "n_3d": n_valid_3d,
            "n_inliers": n_inliers,
            "inlier_ratio": inlier_ratio,
            "delta_t": float(delta_t),
            "delta_deg": float(delta_deg),
        }
        return True, init_c2w.float(), info

    # ========================================================================
    # 内部工具函数
    # ========================================================================
    def _camera_rgb_to_numpy(self, cam):
        """将 Camera.original_image [3, H, W] tensor 转为 numpy [H, W, 3] BGR uint8"""
        img = getattr(cam, "original_image", None)
        if img is None:
            return None
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = img.transpose(1, 2, 0)  # [3, H, W] → [H, W, 3]
        if img.max() <= 1.01:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _get_or_create_freq_mask(self, cam):
        """优先使用已有 freq_mask，否则现场生成"""
        freq_mask = getattr(cam, "freq_mask", None)
        if freq_mask is not None:
            return freq_mask  # torch.bool [H, W] on CUDA

        rgb_np = self._camera_rgb_to_numpy(cam)
        if rgb_np is None:
            H, W = cam.image_height, cam.image_width
            return torch.ones((H, W), dtype=torch.bool, device="cuda")

        if self._fft_filter is None:
            H, W = cam.image_height, cam.image_width
            self._fft_filter = FFTFrequencyFilter(H, W)

        mask = self._fft_filter.generate_frequency_mask(rgb_np)
        return mask

    def _backproject(self, pts_2d, depth_map, fx, fy, cx, cy):
        """
        将 2D 点通过深度图反投影为 3D 点（相机坐标系）。
        pts_2d: [N, 2] float32, (u, v) 像素坐标
        depth_map: [H, W] numpy, 深度值（米）
        返回:
            pts_3d: [N, 3] float32
            valid:  [N] bool
        """
        H, W = depth_map.shape
        pts_2d = np.round(pts_2d).astype(np.int32)

        valid_u = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W)
        valid_v = (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
        valid = valid_u & valid_v

        u = pts_2d[valid, 0]
        v = pts_2d[valid, 1]
        Z = depth_map[v, u]

        valid_depth = (Z > self.min_depth) & (Z < self.max_depth)

        full_valid = np.zeros(len(pts_2d), dtype=bool)
        full_valid[valid] = valid_depth

        pts_3d = np.zeros((len(pts_2d), 3), dtype=np.float32)
        idx = np.where(full_valid)[0]
        u_valid = u[valid_depth]
        v_valid = v[valid_depth]
        Z_valid = Z[valid_depth]

        pts_3d[idx, 0] = (u_valid - cx) * Z_valid / fx
        pts_3d[idx, 1] = (v_valid - cy) * Z_valid / fy
        pts_3d[idx, 2] = Z_valid

        return pts_3d, full_valid
