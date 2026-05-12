"""
Simple-RGBD-Odometry integration layer for FVO-GS-SLAM.

Wraps the upstream rgbd_odom.RGBDOdom (C++ pybind accelerated) as a
VO prior provider with the standard FVO tracking interface.

Requires the pybind module to be built:
  sh scripts/build_simple_rgbd_odom_bridge.sh

PYTHONPATH must include the upstream python directory:
  export PYTHONPATH="third_party/Simple-RGBD-Odometry/python:$PYTHONPATH"
"""

import io
import os
import sys
import time
from contextlib import redirect_stdout
from typing import Optional, Tuple

import numpy as np

_IMPORT_ERROR = None

# Ensure upstream python package is importable
_upstream_python = os.path.join(
    os.path.dirname(__file__), "..", "..", "third_party", "Simple-RGBD-Odometry", "python"
)
_upstream_python = os.path.abspath(_upstream_python)
if _upstream_python not in sys.path:
    sys.path.insert(0, _upstream_python)

try:
    from rgbd_odom.config import RGBDConfig
    from rgbd_odom.rgbd_odom import RGBDOdom as _UpstreamRGBDOdom
except ImportError as exc:
    _IMPORT_ERROR = exc
    RGBDConfig = None
    _UpstreamRGBDOdom = None


def version():
    if _IMPORT_ERROR is not None:
        return f"simple_rgbd_odom_bridge_v0 (unavailable: {_IMPORT_ERROR})"
    return "simple_rgbd_odom_bridge_v0 (pybind)"


def is_available():
    return _UpstreamRGBDOdom is not None


# ---------------------------------------------------------------------------
# SimpleRGBDVOProvider — FVO-compatible wrapper
# ---------------------------------------------------------------------------
class SimpleRGBDVOProvider:
    def __init__(self, config: dict, W: int, H: int, fx: float, fy: float, cx: float, cy: float):
        if _UpstreamRGBDOdom is None:
            raise RuntimeError(
                f"Upstream rgbd_odom package not available: {_IMPORT_ERROR}"
            )

        cfg = config.get("SimpleRGBDOdom", {})
        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Build upstream config
        rgbd_cfg = RGBDConfig()
        rgbd_cfg.data.min_range = float(cfg.get("min_range", 0.05))
        rgbd_cfg.data.max_range = float(cfg.get("max_range", 5.0))
        rgbd_cfg.mapping.voxel_size = float(cfg.get("voxel_size", 0.5))
        rgbd_cfg.mapping.max_points_per_voxel = int(cfg.get("max_points_per_voxel", 20))
        rgbd_cfg.registration.max_correspondence_distance = float(
            cfg.get("max_correspondence_distance", 0.20)
        )
        rgbd_cfg.registration.search_radius = float(cfg.get("search_radius", 5.0))
        rgbd_cfg.descriptor.num_descriptors = int(cfg.get("orb_nfeatures", 1000))

        self._odom = _UpstreamRGBDOdom(intrinsics=self.intrinsics, config=rgbd_cfg)

        # Quality gates
        self.min_valid_keypoints = int(cfg.get("min_valid_keypoints", 80))
        self.min_inliers = int(cfg.get("min_inliers", 20))
        self.min_inlier_ratio = float(cfg.get("min_inlier_ratio", 0.15))
        self.max_motion_trans = float(cfg.get("max_motion_trans", 0.50))
        self.max_motion_rot_deg = float(cfg.get("max_motion_rot_deg", 30.0))

        self.debug_log = cfg.get("debug_log", False)
        self.reset_on_tracking_lost = cfg.get("reset_on_tracking_lost", True)

        # State
        self.frame_id = 0

    def reset(self, initial_c2w: Optional[np.ndarray] = None):
        self._odom.local_map.clear()
        self._odom.poses = []
        self.frame_id = 0
        if initial_c2w is not None:
            self._odom.poses.append(initial_c2w.copy())

    # ------------------------------------------------------------------
    def track(
        self,
        rgb_img: np.ndarray,
        depth_np: np.ndarray,
        init_c2w: Optional[np.ndarray] = None,
    ) -> Tuple[bool, np.ndarray, dict]:
        t0 = time.perf_counter()

        # Record keypoint count before calling upstream
        gray = _cv2_cvtColor(rgb_img)
        raw_kps = self._odom.orb.detect(gray, mask=None)
        num_keypoints = len(raw_kps)

        # Call upstream register_frame (RGB image expected), suppress upstream prints
        with redirect_stdout(io.StringIO()):
            frame_pcd, corresp_tuple = self._odom.register_frame(rgb_img, depth_np)

        # Extract VO's local C2W (relative to VO's starting identity)
        vo_pose = self._odom.poses[-1].copy() if self._odom.poses else np.eye(4)

        # Use previous frame's known global C2W as base, apply VO's frame-to-frame delta.
        # This avoids accumulating VO drift: each frame starts from the last refined pose.
        if len(self._odom.poses) >= 2 and init_c2w is not None:
            vo_prev = self._odom.poses[-2]
            delta = np.linalg.inv(vo_prev) @ vo_pose  # relative motion in VO frame
            est_c2w = init_c2w @ delta
        else:
            est_c2w = init_c2w if init_c2w is not None else np.eye(4, dtype=np.float64)

        last_pose = init_c2w if init_c2w is not None else np.eye(4)

        # Extract quality info
        corresp_pts = None if corresp_tuple is None else corresp_tuple[1]
        num_inliers = 0 if corresp_pts is None else len(corresp_pts)
        # upstream RGBDOdom doesn't directly expose #matches; approximate from poses
        num_matches = num_inliers  # conservative

        # Compute valid depth keypoint count from upstream Frame
        # (upstream already filters them; we can approximate from frame_pcd)
        num_valid_depth = len(frame_pcd) if frame_pcd is not None else 0

        # Quality checks
        success = True
        reason = "ok"
        inlier_ratio = num_inliers / max(num_matches, 1)

        if num_valid_depth < self.min_valid_keypoints:
            success = False
            reason = f"too_few_valid_keypoints ({num_valid_depth} < {self.min_valid_keypoints})"
        elif num_inliers < self.min_inliers:
            success = False
            reason = f"too_few_inliers ({num_inliers} < {self.min_inliers})"
        elif num_matches > 0 and inlier_ratio < self.min_inlier_ratio:
            success = False
            reason = f"low_inlier_ratio ({inlier_ratio:.3f} < {self.min_inlier_ratio})"

        # Motion check
        motion_trans = float(np.linalg.norm(est_c2w[:3, 3] - last_pose[:3, 3]))
        rel_R = est_c2w[:3, :3].T @ last_pose[:3, :3]
        motion_rot_rad = float(np.arccos(max(-1.0, min(1.0, (np.trace(rel_R) - 1) / 2))))
        motion_rot_deg = float(np.rad2deg(motion_rot_rad))

        if motion_trans > self.max_motion_trans:
            success = False
            reason = f"excessive_translation ({motion_trans:.3f}m > {self.max_motion_trans}m)"
        if motion_rot_deg > self.max_motion_rot_deg:
            success = False
            reason = f"excessive_rotation ({motion_rot_deg:.1f}deg > {self.max_motion_rot_deg}deg)"

        dt = (time.perf_counter() - t0) * 1000.0
        self.frame_id += 1

        return success, est_c2w, {
            "backend": "simple_rgbd_odom",
            "success": success,
            "frame_id": self.frame_id - 1,
            "num_keypoints": num_keypoints,
            "num_valid_depth_keypoints": num_valid_depth,
            "num_map_points": 0,  # not directly exposed by upstream
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": float(inlier_ratio),
            "pose_source": "ransac_pnp" if num_inliers >= 3 else "prediction",
            "motion_trans": motion_trans,
            "motion_rot_deg": motion_rot_deg,
            "reason": reason,
            "runtime_ms": dt,
        }


def _cv2_cvtColor(rgb_img):
    """Avoid re-importing cv2 at module level (keep import cost low)."""
    import cv2
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
