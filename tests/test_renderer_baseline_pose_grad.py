"""
Stage 0: Baseline renderer test with pose gradient verification.

Tests the current diff-surfel-rasterization renderer without any modifications.
Verifies forward pass outputs, backward pass, and pose gradient propagation.

Uses the low-level rasterizer API directly to avoid heavy dependency chains.
"""

import math
import sys
import torch


def _build_projection_matrix(znear, zfar, cx, cy, fx, fy, W, H):
    """Replicate getProjectionMatrix2 inline to avoid import chain."""
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _build_minimal_rendering(device="cuda:0"):
    """Build minimal rasterizer inputs directly, avoiding project import chain."""
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    W, H = 640, 480
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    fovx = 2 * math.atan(W / (2 * fx))
    fovy = 2 * math.atan(H / (2 * fy))
    znear, zfar = 0.01, 100.0

    projmatrix = _build_projection_matrix(znear, zfar, cx, cy, fx, fy, W, H).to(device)
    projmatrix_raw = projmatrix.clone()

    # World-to-view: identity pose (camera at origin looking down +z)
    # The view matrix in this renderer is world_view_transform = T.transpose(0,1)
    # where T is the estimated camera pose (world-to-camera)
    T = torch.eye(4, device=device)
    viewmatrix = T.transpose(0, 1)  # world_view_transform
    full_proj = viewmatrix @ projmatrix  # full_proj_transform
    campos = viewmatrix.inverse()[3, :3]  # camera_center

    # Pose deltas (will receive gradients)
    cam_rot_delta = torch.tensor([0.01, -0.02, 0.005], device=device, requires_grad=True)
    cam_trans_delta = torch.tensor([0.005, -0.01, 0.002], device=device, requires_grad=True)

    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        cx=cx, cy=cy,
        tanfovx=fovx, tanfovy=fovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=full_proj,  # Note: this is full_proj_transform in original code
        projmatrix=full_proj,
        projmatrix_raw=projmatrix_raw,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
            use_sa=False,
    )

    # Wait - let me recheck. In the original render():
    # viewmatrix=viewpoint_camera.world_view_transform  (4x4)
    # projmatrix=viewpoint_camera.full_proj_transform   (4x4)
    # So viewmatrix is world->view, projmatrix is world->view->proj
    # Let me use the correct matrices.
    raster_settings2 = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        cx=cx, cy=cy,
        tanfovx=fovx, tanfovy=fovy,
        bg=bg,
        scale_modifier=1.0,
        viewmatrix=viewmatrix,       # world_view_transform
        projmatrix=full_proj,         # full_proj_transform
        projmatrix_raw=projmatrix_raw,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=False,
            use_sa=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings2)

    # Build minimal Gaussian point cloud (N=100 points)
    N = 100
    xs = torch.linspace(-0.5, 0.5, 10, device=device)
    ys = torch.linspace(-0.5, 0.5, 10, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    z_vals = torch.full((100,), 2.0, device=device)
    means3D = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z_vals], dim=-1)

    # Screen-space points (will get gradients populated)
    means2D = torch.zeros(N, 3, device=device, requires_grad=True)

    # SH features: (N, (sh_degree+1)^2, 3) = (N, 1, 3) for degree 0
    shs = torch.rand(N, 1, 3, device=device)

    colors_precomp = None  # use SH instead

    opacities = torch.sigmoid(torch.randn(N, 1, device=device))

    scales = torch.exp(torch.randn(N, 2, device=device) * 0.5 - 1.0)  # small 2D scales

    # Random rotation quaternions (normalized)
    rotations = torch.randn(N, 4, device=device)
    rotations = torch.nn.functional.normalize(rotations, dim=-1)

    cov3D_precomp = None  # use scales/rotations instead

    return rasterizer, means3D, means2D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp, cam_rot_delta, cam_trans_delta


def test_forward_outputs():
    """Test forward pass returns correct fields without NaN/Inf."""
    print("=== Test 1: Forward pass outputs ===")
    device = "cuda:0"

    (rasterizer, means3D, means2D, shs, colors_precomp,
     opacities, scales, rotations, cov3D_precomp,
     cam_rot_delta, cam_trans_delta) = _build_minimal_rendering(device)

    rets = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        w=cam_rot_delta,
        trans=cam_trans_delta,
    )

    rendered_image, radii, allmap, n_touched = rets

    # allmap layout:
    # [0] = expected depth accum (D)
    # [1] = alpha
    # [2:5] = normal
    # [5] = median depth
    # [6] = distortion

    render_alpha = allmap[1:2]
    render_depth_expected = allmap[0:1]
    render_depth_median = allmap[5:6]
    render_normal = allmap[2:5]
    render_dist = allmap[6:7]

    # surf_depth (with depth_ratio=1.0 median, matching default config)
    depth_ratio = 1.0
    depth_expected_safe = render_depth_expected / torch.clamp(render_alpha, min=1e-6)
    depth_expected_safe = torch.nan_to_num(depth_expected_safe, 0, 0)
    depth_median_safe = torch.nan_to_num(render_depth_median, 0, 0)
    surf_depth = depth_expected_safe * (1 - depth_ratio) + depth_ratio * depth_median_safe

    # Check all tensors
    checks = [
        ("rendered_image", rendered_image),
        ("radii", radii),
        ("allmap", allmap),
        ("surf_depth", surf_depth),
        ("render_alpha", render_alpha),
        ("render_normal", render_normal),
        ("render_dist", render_dist),
        ("n_touched", n_touched),
    ]

    all_good = True
    for name, t in checks:
        has_nan = torch.isnan(t).any().item() if t.dtype.is_floating_point else False
        has_inf = torch.isinf(t).any().item() if t.dtype.is_floating_point else False
        status = "OK" if not has_nan and not has_inf else "FAIL"
        if has_nan or has_inf:
            all_good = False
        print(f"  [{status}] {name}: shape={tuple(t.shape)}, NaN={has_nan}, Inf={has_inf}")

    print(f"  render       mean={rendered_image.mean().item():.6f}  min={rendered_image.min().item():.6f}  max={rendered_image.max().item():.6f}")
    print(f"  surf_depth   mean={surf_depth.mean().item():.6f}  min={surf_depth.min().item():.6f}  max={surf_depth.max().item():.6f}")
    print(f"  alpha        mean={render_alpha.mean().item():.6f}  min={render_alpha.min().item():.6f}  max={render_alpha.max().item():.6f}")
    print(f"  rend_normal  mean={render_normal.mean().item():.6f}  min={render_normal.min().item():.6f}  max={render_normal.max().item():.6f}")
    print(f"  rend_dist    mean={render_dist.mean().item():.6f}  min={render_dist.min().item():.6f}  max={render_dist.max().item():.6f}")
    print(f"  radii        num_visible={(radii > 0).sum().item()} / {len(radii)}")
    print(f"  n_touched    sum={n_touched.sum().item()}")
    print(f"  allmap       shape={tuple(allmap.shape)}")

    if all_good:
        print("  PASSED\n")
    else:
        print("  FAILED\n")
    return all_good


def test_backward_and_pose_grad():
    """Test backward pass: loss.backward() and check all key gradients."""
    print("=== Test 2: Backward pass and pose gradients ===")
    device = "cuda:0"

    (rasterizer, means3D, means2D, shs, colors_precomp,
     opacities, scales, rotations, cov3D_precomp,
     cam_rot_delta, cam_trans_delta) = _build_minimal_rendering(device)

    means3D = means3D.clone().requires_grad_(True)
    opacities = opacities.clone().requires_grad_(True)
    scales = scales.clone().requires_grad_(True)
    rotations = rotations.clone().requires_grad_(True)
    shs = shs.clone().requires_grad_(True)

    rets = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        w=cam_rot_delta,
        trans=cam_trans_delta,
    )

    rendered_image, radii, allmap, n_touched = rets

    # Build a composite loss
    render_alpha = allmap[1:2]
    render_depth_expected = allmap[0:1]
    render_dist = allmap[6:7]

    depth_safe = render_depth_expected / torch.clamp(render_alpha, min=1e-6)
    depth_safe = torch.nan_to_num(depth_safe, 0, 0)

    loss = rendered_image.mean() + depth_safe.mean() + 0.01 * render_dist.mean()
    loss.backward()

    checks = []

    # Pose gradients
    rot_grad = cam_rot_delta.grad
    trans_grad = cam_trans_delta.grad
    checks.append(("cam_rot_delta.grad not None", rot_grad is not None))
    checks.append(("cam_trans_delta.grad not None", trans_grad is not None))
    if rot_grad is not None:
        checks.append(("cam_rot_delta.grad finite", torch.isfinite(rot_grad).all().item()))
        print(f"  cam_rot_delta.grad    = {rot_grad.detach().cpu().tolist()}")
    if trans_grad is not None:
        checks.append(("cam_trans_delta.grad finite", torch.isfinite(trans_grad).all().item()))
        print(f"  cam_trans_delta.grad  = {trans_grad.detach().cpu().tolist()}")

    # Gaussian parameter gradients
    for name, param in [("means3D", means3D), ("opacities", opacities),
                        ("scales", scales), ("rotations", rotations), ("shs", shs)]:
        has_grad = param.grad is not None and torch.isfinite(param.grad).all().item()
        checks.append((f"{name}.grad finite", has_grad))
        if param.grad is not None:
            print(f"  {name}.grad norm = {param.grad.norm().item():.8f}")
        else:
            print(f"  {name}.grad = None (WARNING)")

    all_passed = all(passed for _, passed in checks)
    for name, passed in checks:
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if all_passed:
        print("  PASSED\n")
    else:
        print("  FAILED\n")
    return all_passed


def test_multiview_consistency():
    """Test forward/backward with different camera translations."""
    print("=== Test 3: Multi-view consistency ===")
    device = "cuda:0"
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    W, H = 640, 480
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    fovx = 2 * math.atan(W / (2 * fx))
    fovy = 2 * math.atan(H / (2 * fy))
    znear, zfar = 0.01, 100.0

    projmatrix = _build_projection_matrix(znear, zfar, cx, cy, fx, fy, W, H).to(device)
    projmatrix_raw = projmatrix.clone()
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Shared Gaussians
    N = 100
    xs = torch.linspace(-0.5, 0.5, 10, device=device)
    ys = torch.linspace(-0.5, 0.5, 10, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    z_vals = torch.full((100,), 2.0, device=device)
    means3D = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z_vals], dim=-1)
    means2D = torch.zeros(N, 3, device=device, requires_grad=True)
    shs = torch.rand(N, 1, 3, device=device)
    opacities = torch.sigmoid(torch.randn(N, 1, device=device))
    scales = torch.exp(torch.randn(N, 2, device=device) * 0.5 - 1.0)
    rotations = torch.randn(N, 4, device=device)
    rotations = torch.nn.functional.normalize(rotations, dim=-1)
    cov3D_precomp = None
    colors_precomp = None

    # Different camera poses
    Ts = [
        torch.eye(4, device=device),
        torch.tensor([[1, 0, 0, 0.2], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]], device=device, dtype=torch.float32),
        torch.tensor([[1, 0, 0, 0.0], [0, 1, 0, -0.1], [0, 0, 1, 0.5], [0, 0, 0, 1]], device=device, dtype=torch.float32),
    ]

    for i, T in enumerate(Ts):
        viewmatrix = T.transpose(0, 1)
        full_proj = viewmatrix @ projmatrix
        campos = viewmatrix.inverse()[3, :3]

        raster_settings = GaussianRasterizationSettings(
            image_height=H, image_width=W,
            cx=cx, cy=cy,
            tanfovx=fovx, tanfovy=fovy,
            bg=bg, scale_modifier=1.0,
            viewmatrix=viewmatrix, projmatrix=full_proj,
            projmatrix_raw=projmatrix_raw,
            sh_degree=0, campos=campos,
            prefiltered=False, debug=False,
            use_sa=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        cam_rot_delta = torch.tensor([0.01, -0.02, 0.005], device=device, requires_grad=True)
        cam_trans_delta = torch.tensor([0.005, -0.01, 0.002], device=device, requires_grad=True)

        rendered_image, radii, allmap, n_touched = rasterizer(
            means3D=means3D, means2D=means2D,
            shs=shs, colors_precomp=colors_precomp,
            opacities=opacities, scales=scales, rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            w=cam_rot_delta, trans=cam_trans_delta,
        )

        loss = rendered_image.mean() + allmap[0:1].mean()
        loss.backward()

        rot_ok = cam_rot_delta.grad is not None and torch.isfinite(cam_rot_delta.grad).all()
        trans_ok = cam_trans_delta.grad is not None and torch.isfinite(cam_trans_delta.grad).all()
        print(f"  View {i}: rot_grad OK={rot_ok}, trans_grad OK={trans_ok}, loss={loss.item():.6f}")
        assert rot_ok and trans_ok, f"View {i} gradient check failed"

    print("  PASSED\n")
    return True


def main():
    print("=" * 60)
    print("Stage 0: Baseline Renderer Test with Pose Gradient")
    print("=" * 60)
    print()

    results = []
    results.append(("Forward outputs", test_forward_outputs()))
    results.append(("Backward + pose grad", test_backward_and_pose_grad()))
    results.append(("Multi-view consistency", test_multiview_consistency()))

    print("=" * 60)
    all_passed = all(passed for _, passed in results)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name}")

    if all_passed:
        print("\nAll tests PASSED. Ready to proceed to Stage 1.")
    else:
        print("\nSome tests FAILED. Do not proceed to Stage 1.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
