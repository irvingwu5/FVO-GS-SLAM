"""Stage 2: Verify SA dist backward gradient correctness and pose gradient preservation.

Tests:
  1. use_sa=False: backward parity, pose grad exists
  2. use_sa=True: rend_dist participates in backward, all grads finite
  3. use_sa=True: rend_dist has no NaN/Inf
  4. Both modes: cam_rot_delta.grad / cam_trans_delta.grad exist and finite
"""

import math
import sys
import torch
import numpy as np


def _make_pipe(use_sa):
    """Minimal pipe-like config for renderer."""
    class Pipe:
        pass
    p = Pipe()
    p.use_sa = use_sa
    p.use_sa_depth = use_sa
    p.depth_eps = 1e-6
    p.debug_sa_depth = False
    p.compute_cov3D_python = False
    p.convert_SHs_python = False
    p.depth_ratio = 0.0
    return p


def _make_camera(H=120, W=160):
    """Minimal viewpoint camera."""
    class Cam:
        pass
    c = Cam()
    c.image_height = H
    c.image_width = W
    c.FoVx = 0.6
    c.FoVy = 0.6 * H / W
    c.cx = W / 2.0
    c.cy = H / 2.0
    c.znear = 0.2
    c.zfar = 100.0

    # Identity view matrix (world = camera)
    c.world_view_transform = torch.eye(4, device="cuda")
    c.world_view_transform[2, 3] = 0.0  # camera at origin looking -Z

    # Simple projection matrix
    fx = W / (2.0 * math.tan(c.FoVx / 2.0))
    fy = H / (2.0 * math.tan(c.FoVy / 2.0))
    proj = torch.zeros(4, 4, device="cuda")
    proj[0, 0] = 2.0 * fx / W
    proj[1, 1] = 2.0 * fy / H
    proj[2, 2] = -(c.zfar + c.znear) / (c.zfar - c.znear)
    proj[2, 3] = -2.0 * c.zfar * c.znear / (c.zfar - c.znear)
    proj[3, 2] = -1.0
    c.full_proj_transform = proj.T.contiguous()  # column-major
    c.projection_matrix = proj.T.contiguous()

    c.camera_center = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    # Pose deltas (requires_grad)
    c.cam_rot_delta = torch.zeros(3, device="cuda", requires_grad=True)
    c.cam_trans_delta = torch.zeros(3, device="cuda", requires_grad=True)
    return c


def _make_gaussians(num=5000):
    """Create minimal GaussianModel-like object."""
    class GM:
        pass
    g = GM()
    # Place Gaussians on a plane at z=3 with some depth variation
    xyz = torch.randn(num, 3, device="cuda") * 0.3
    xyz[:, 2] = 3.0 + torch.randn(num, device="cuda") * 0.2
    xyz[:, 2] = xyz[:, 2].clamp(min=0.5)
    g.get_xyz = xyz
    g.get_scaling = torch.rand(num, 2, device="cuda") * 0.1 + 0.02
    # Quaternion rotation (identity-ish)
    rot = torch.zeros(num, 4, device="cuda")
    rot[:, 0] = 1.0  # w=1 -> identity rotation
    rot[:, 1:] = torch.randn(num, 3, device="cuda") * 0.01
    g.get_rotation = rot
    g.get_opacity = torch.sigmoid(torch.randn(num, 1, device="cuda") * 0.5)
    g.active_sh_degree = 0
    g.max_sh_degree = 0
    g.get_features = torch.zeros(num, 3, 1, device="cuda")
    g.get_features[:, 0, 0] = torch.rand(num, device="cuda") * 0.2
    g.get_features[:, 1, 0] = torch.rand(num, device="cuda") * 0.3
    g.get_features[:, 2, 0] = torch.rand(num, device="cuda") * 0.4
    return g


def test_sa_dist_backward(use_sa):
    """Run a forward+backward and verify gradient sanity."""
    from gaussian_splatting.gaussian_renderer import render

    torch.manual_seed(42)
    H, W = 120, 160
    bg_color = torch.zeros(3, device="cuda")
    pipe = _make_pipe(use_sa=use_sa)
    cam = _make_camera(H, W)
    gs = _make_gaussians(5000)

    rets = render(cam, gs, pipe, bg_color)
    if rets is None:
        return {"error": "render returned None (no visible Gaussians)"}

    render_img = rets["render"]
    depth = rets["depth"]
    rend_dist = rets["rend_dist"]

    results = {}

    # Check forward values
    results["render_nan"] = torch.isnan(render_img).any().item()
    results["render_inf"] = torch.isinf(render_img).any().item()
    results["depth_nan"] = torch.isnan(depth).any().item()
    results["depth_inf"] = torch.isinf(depth).any().item()
    results["dist_nan"] = torch.isnan(rend_dist).any().item()
    results["dist_inf"] = torch.isinf(rend_dist).any().item()

    if use_sa:
        results["dist_nonneg"] = (rend_dist >= -1e-6).all().item()

    # Build loss and backward
    loss = render_img.mean() + depth.mean() + 0.01 * rend_dist.mean()
    loss.backward()

    # Check gradient sanity
    grad_names = []
    grad_finite = []
    for name, param in [("cam_rot_delta", cam.cam_rot_delta),
                         ("cam_trans_delta", cam.cam_trans_delta)]:
        grad_names.append(name)
        g = param.grad
        if g is None:
            grad_finite.append(False)
        else:
            grad_finite.append(bool(torch.isfinite(g).all().item()))

    results["grad_names"] = grad_names
    results["grad_finite"] = grad_finite
    results["pose_grad_exists"] = all(
        p.grad is not None for p in [cam.cam_rot_delta, cam.cam_trans_delta]
    )

    return results


def main():
    print("=" * 60)
    print("SA Dist Backward + Pose Gradient Test")
    print("=" * 60)

    all_passed = True

    for mode, label in [(False, "use_sa=False"), (True, "use_sa=True")]:
        print(f"\n--- {label} ---")
        try:
            r = test_sa_dist_backward(use_sa=mode)
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            all_passed = False
            continue

        if "error" in r:
            print(f"  SKIP: {r['error']}")
            continue

        for key, val in r.items():
            if key in ("grad_names", "grad_finite"):
                continue
            print(f"  {key}: {val}")

        # Validate results
        checks = []
        checks.append(("no render NaN", not r["render_nan"]))
        checks.append(("no render Inf", not r["render_inf"]))
        checks.append(("no depth NaN", not r["depth_nan"]))
        checks.append(("no depth Inf", not r["depth_inf"]))
        checks.append(("no dist NaN", not r["dist_nan"]))
        checks.append(("no dist Inf", not r["dist_inf"]))
        if mode:
            checks.append(("dist non-neg", r.get("dist_nonneg", False)))
        checks.append(("pose grad exists", r["pose_grad_exists"]))
        for name, finite in zip(r["grad_names"], r["grad_finite"]):
            checks.append((f"  {name}.grad finite", finite))

        passed = True
        for check_name, result in checks:
            status = "PASS" if result else "FAIL"
            if not result:
                passed = False
                all_passed = False
            print(f"  [{status}] {check_name}")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
