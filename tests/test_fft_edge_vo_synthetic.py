# tests/test_fft_edge_vo_synthetic.py
# Synthetic tests for FFTEdgeVO LM optimisation.
# Requires CUDA (skips otherwise).

import numpy as np
import torch
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fft_edge_vo import FFTEdgeVO, _se3_exp, _se3_log


def __delta_trans(T_est, T_gt):
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def __delta_rot_deg(T_est, T_gt):
    dR = T_est[:3, :3].T @ T_gt[:3, :3]
    cos = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos) * 180.0 / np.pi)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 480, 640
FX, FY, CX, CY = 500.0, 500.0, 320.0, 240.0
Z_DEPTH = 2.0  # constant flat-plane depth (m)


def _make_config(**overrides):
    cfg = {
        "use_fft_edge_vo": True,
        "num_levels": 3,
        "level_scale": 0.5,
        "max_iters_coarse": 25,
        "max_iters_fine": 15,
        "dt_max_dist": 50.0,
        "dt_huber_delta": 5.0,
        "dt_mean_fail_threshold": 15.0,
        "require_visible_ratio": 0.15,
        "min_dt_mean": 0.1,
        "max_cur_points": 8000,
        "min_cur_points": 100,
        "min_depth": 0.05,
        "max_depth": 10.0,
        "sampling_strategy": "grid",
        "debug_log": False,
        "debug_save_images": False,
    }
    cfg.update(overrides)
    return {"FFTEdgeVO": cfg}


def _make_rectangle_bgr(offset_x=0, offset_y=0):
    """White background + black filled rectangle, offset from centre (pixels)."""
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    x0 = W // 2 - 100 + int(round(offset_x))
    y0 = H // 2 - 75 + int(round(offset_y))
    x1 = x0 + 200
    y1 = y0 + 150
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
    return img


def _make_depth():
    return np.full((H, W), Z_DEPTH, dtype=np.float32)


def _c2w_from_pose(tx=0.0, ty=0.0, tz=0.0, yaw_deg=0.0):
    """Build 4x4 C2W matrix for a camera at (tx,ty,tz) with yaw rotation."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 2] = s
    T[0, 3] = tx
    T[1, 1] = 1.0
    T[1, 3] = ty
    T[2, 0] = -s
    T[2, 2] = c
    T[2, 3] = tz
    return T


def _compute_expected_T_ref_cur(tx, ty, tz, yaw_deg):
    """Ground-truth T_ref_cur: ref at origin, cur at (tx,ty,tz,yaw)."""
    T_wr = np.eye(4, dtype=np.float64)  # ref at origin
    T_wc = _c2w_from_pose(tx, ty, tz, yaw_deg)
    return T_wr @ T_wc  # = T_wc since T_wr = I


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def _run_case(name, cur_offset_x, cur_offset_y, gt_tx, gt_ty, gt_tz, gt_yaw_deg,
              expect_success=True):
    """Run one VO test case and return (passed, details)."""
    config = _make_config()
    vo = FFTEdgeVO(config, W, H, FX, FY, CX, CY)

    # Reference: rectangle at centre
    ref_bgr = _make_rectangle_bgr(0, 0)
    ref_depth = _make_depth()
    ref_c2w = np.eye(4, dtype=np.float64)
    ok = vo.set_reference(ref_bgr, ref_depth, ref_c2w, frame_id=0)
    if not ok:
        return False, f"set_reference failed"

    # Current: rectangle shifted
    cur_bgr = _make_rectangle_bgr(cur_offset_x, cur_offset_y)
    cur_depth = _make_depth()
    init_c2w = _c2w_from_pose(gt_tx * 0.5, gt_ty * 0.5, 0.0, gt_yaw_deg * 0.5)  # perturbed initial guess

    success, est_c2w, info = vo.track(cur_bgr, cur_depth, init_c2w)

    T_rc_gt = _compute_expected_T_ref_cur(gt_tx, gt_ty, gt_tz, gt_yaw_deg)
    T_rc_est = np.linalg.inv(ref_c2w) @ est_c2w  # T_ref_cur estimate
    dt = _delta_trans(T_rc_est, T_rc_gt)
    dr = _delta_rot_deg(T_rc_est, T_rc_gt)

    dt_init = _delta_trans(np.linalg.inv(ref_c2w) @ init_c2w, T_rc_gt)

    # Checks
    checks = []
    if info["final_error"] <= info["initial_error"] + 1e-6:
        checks.append("err_decreased")
    else:
        checks.append(f"err_NOT_decreased: {info['initial_error']:.3f}→{info['final_error']:.3f}")

    if success == expect_success:
        checks.append("success_ok")
    else:
        checks.append(f"success_mismatch: got {success}, expected {expect_success}")

    if dt_init > 0:
        checks.append(f"init_dt={dt_init:.4f}m→final_dt={dt:.4f}m")
    else:
        checks.append("init_is_gt")

    if info["accepted_iters"] >= 1:
        checks.append(f"accepted_iters={info['accepted_iters']}")
    else:
        checks.append("NO_accepted_iters")

    if info["delta_t_from_init"] > 0 or info["delta_r_deg_from_init"] > 0:
        checks.append(f"delta_from_init: t={info['delta_t_from_init']:.4f}m r={info['delta_r_deg_from_init']:.2f}°")

    all_ok = (info["final_error"] <= info["initial_error"] + 1e-4
              and success == expect_success
              and info["accepted_iters"] >= 1)

    details = (f"[{name}] est_dt={dt:.4f}m est_dr={dr:.2f}° | "
               + " | ".join(checks)
               + f" | inside={info.get('inside_count','?')} near_edge={info.get('near_edge_count','?')}")

    return all_ok, details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True  # not a failure

    results = []

    # Case 1: identity
    ok, msg = _run_case("identity", 0, 0, 0.0, 0.0, 0.0, 0.0, expect_success=True)
    results.append((ok, msg))
    print(msg)

    # Case 2: small tx = 0.02 m → ~5 px shift at Z=2m, fx=500
    ok, msg = _run_case("tx_0.02m", -5, 0, 0.02, 0.0, 0.0, 0.0, expect_success=True)
    results.append((ok, msg))
    print(msg)

    # Case 3: small ty = 0.02 m
    ok, msg = _run_case("ty_0.02m", 0, -5, 0.0, 0.02, 0.0, 0.0, expect_success=True)
    results.append((ok, msg))
    print(msg)

    # Case 4: small yaw = 2 deg → ~17 px shift
    ok, msg = _run_case("yaw_2deg", 17, 0, 0.0, 0.0, 0.0, 2.0, expect_success=True)
    results.append((ok, msg))
    print(msg)

    # Case 5: large motion should fail
    ok, msg = _run_case("large_tx_0.5m", -125, 0, 0.5, 0.0, 0.0, 0.0, expect_success=False)
    results.append((ok, msg))
    print(msg)

    n_pass = sum(1 for ok, _ in results if ok)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"Results: {n_pass}/{n_total} passed")
    if n_pass < n_total:
        print("FAILURES:")
        for ok, msg in results:
            if not ok:
                print(f"  {msg}")

    return n_pass == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
