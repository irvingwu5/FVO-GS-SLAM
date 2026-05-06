"""
Stage 1: Verify use_sa parameter plumbing without changing rendering behavior.

use_sa=False and use_sa=True must produce identical outputs (within FP tolerance).
"""

import math
import sys
import torch


def _build_projection_matrix(znear, zfar, cx, cy, fx, fy, W, H):
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


def _render_with_use_sa(use_sa, device="cuda:0"):
    """Run a single render pass with given use_sa setting."""
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    W, H = 640, 480
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    fovx = 2 * math.atan(W / (2 * fx))
    fovy = 2 * math.atan(H / (2 * fy))
    znear, zfar = 0.01, 100.0

    projmatrix = _build_projection_matrix(znear, zfar, cx, cy, fx, fy, W, H).to(device)
    projmatrix_raw = projmatrix.clone()

    T = torch.eye(4, device=device)
    viewmatrix = T.transpose(0, 1)
    full_proj = viewmatrix @ projmatrix
    campos = viewmatrix.inverse()[3, :3]

    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        cx=cx, cy=cy,
        tanfovx=fovx, tanfovy=fovy,
        bg=bg, scale_modifier=1.0,
        viewmatrix=viewmatrix, projmatrix=full_proj,
        projmatrix_raw=projmatrix_raw,
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False,
        use_sa=use_sa,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    N = 100
    xs = torch.linspace(-0.3, 0.3, 10, device=device)
    ys = torch.linspace(-0.3, 0.3, 10, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    z_vals = torch.full((100,), 2.0, device=device)
    means3D = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z_vals], dim=-1)
    means2D = torch.zeros(N, 3, device=device, requires_grad=True)
    shs = torch.rand(N, 1, 3, device=device)
    opacities = torch.sigmoid(torch.randn(N, 1, device=device))
    scales = torch.exp(torch.randn(N, 2, device=device) * 0.3 - 1.0)
    rotations = torch.randn(N, 4, device=device)
    rotations = torch.nn.functional.normalize(rotations, dim=-1)

    rendered_image, radii, allmap, n_touched = rasterizer(
        means3D=means3D, means2D=means2D,
        shs=shs, colors_precomp=None,
        opacities=opacities, scales=scales, rotations=rotations,
        cov3D_precomp=None,
        w=torch.tensor([0.0, 0.0, 0.0], device=device),
        trans=torch.tensor([0.0, 0.0, 0.0], device=device),
    )

    return rendered_image, radii, allmap, n_touched


def test_interface_parity():
    """Test that use_sa=False and use_sa=True produce identical results."""
    print("=== Stage 1: use_sa interface parity test ===")

    torch.manual_seed(42)
    img_false, radii_false, allmap_false, nt_false = _render_with_use_sa(False)

    torch.manual_seed(42)
    img_true, radii_true, allmap_true, nt_true = _render_with_use_sa(True)

    checks = []

    # Compare rendered image
    img_diff = (img_false - img_true).abs().max().item()
    checks.append(("rendered_image max diff == 0", img_diff == 0.0))
    print(f"  rendered_image max diff: {img_diff:.10f}")

    # Compare radii
    radii_diff = (radii_false - radii_true).abs().max().item()
    checks.append(("radii max diff == 0", radii_diff == 0))
    print(f"  radii max diff: {radii_diff}")

    # Compare allmap channels
    for ch, name in enumerate(["D (exp depth)", "alpha", "Nx", "Ny", "Nz", "median depth", "distortion"]):
        ch_diff = (allmap_false[ch] - allmap_true[ch]).abs().max().item()
        checks.append((f"allmap[{ch}] ({name}) max diff == 0", ch_diff == 0.0))
        print(f"  allmap[{ch}] ({name}) max diff: {ch_diff:.10f}")

    # Compare n_touched
    nt_diff = (nt_false - nt_true).abs().max().item()
    checks.append(("n_touched max diff == 0", nt_diff == 0))
    print(f"  n_touched max diff: {nt_diff}")

    all_passed = all(passed for _, passed in checks)
    for name, passed in checks:
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if all_passed:
        print("  PASSED\n")
    else:
        print("  FAILED\n")
    return all_passed


def test_backward_with_use_sa():
    """Test backward pass works with use_sa=True."""
    print("=== Stage 1: Backward pass with use_sa=True ===")
    device = "cuda:0"
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    W, H = 640, 480
    fx, fy = 525.0, 525.0
    cx, cy = 319.5, 239.5
    fovx = 2 * math.atan(W / (2 * fx))
    fovy = 2 * math.atan(H / (2 * fy))
    znear, zfar = 0.01, 100.0

    projmatrix = _build_projection_matrix(znear, zfar, cx, cy, fx, fy, W, H).to(device)
    T = torch.eye(4, device=device)
    viewmatrix = T.transpose(0, 1)
    full_proj = viewmatrix @ projmatrix

    ramp = torch.tensor([1, 2, 3], device=device)
    rot = torch.tensor([4, 5, 6], device=device)

    raster_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        cx=cx, cy=cy,
        tanfovx=fovx, tanfovy=fovy,
        bg=torch.tensor([0.0, 0.0, 0.0], device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix, projmatrix=full_proj,
        projmatrix_raw=projmatrix.clone(),
        sh_degree=0, campos=viewmatrix.inverse()[3, :3],
        prefiltered=False, debug=False,
        use_sa=True,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    N = 100
    xs = torch.linspace(-0.3, 0.3, 10, device=device)
    ys = torch.linspace(-0.3, 0.3, 10, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    z_vals = torch.full((100,), 2.0, device=device)
    means3D = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z_vals], dim=-1)
    means3D = means3D.clone().requires_grad_(True)
    means2D = torch.zeros(N, 3, device=device, requires_grad=True)
    shs = torch.rand(N, 1, 3, device=device, requires_grad=True)
    opacities = torch.sigmoid(torch.randn(N, 1, device=device)).clone().requires_grad_(True)
    scales = torch.exp(torch.randn(N, 2, device=device) * 0.3 - 1.0).clone().requires_grad_(True)
    rotations = torch.randn(N, 4, device=device)
    rotations = torch.nn.functional.normalize(rotations, dim=-1).clone().requires_grad_(True)

    cam_rot_delta = torch.tensor([0.01, -0.02, 0.005], device=device, requires_grad=True)
    cam_trans_delta = torch.tensor([0.005, -0.01, 0.002], device=device, requires_grad=True)

    rendered_image, radii, allmap, n_touched = rasterizer(
        means3D=means3D, means2D=means2D,
        shs=shs, colors_precomp=None,
        opacities=opacities, scales=scales, rotations=rotations,
        cov3D_precomp=None,
        w=cam_rot_delta, trans=cam_trans_delta,
    )

    render_alpha = allmap[1:2]
    depth = allmap[0:1] / torch.clamp(render_alpha, min=1e-6)
    loss = rendered_image.mean() + torch.nan_to_num(depth, 0, 0).mean()
    loss.backward()

    checks = []
    for name, param in [("means3D", means3D), ("opacities", opacities),
                        ("scales", scales), ("rotations", rotations),
                        ("shs", shs), ("cam_rot_delta", cam_rot_delta),
                        ("cam_trans_delta", cam_trans_delta)]:
        ok = param.grad is not None and torch.isfinite(param.grad).all().item()
        checks.append((f"{name}.grad finite", ok))
        if param.grad is not None:
            print(f"  {name}.grad norm = {param.grad.norm().item():.8f}")
        else:
            print(f"  {name}.grad = None")

    all_passed = all(passed for _, passed in checks)
    for name, passed in checks:
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {name}")

    if all_passed:
        print("  PASSED\n")
    else:
        print("  FAILED\n")
    return all_passed


def main():
    print("=" * 60)
    print("Stage 1: use_sa Parameter Plumbing Test")
    print("=" * 60)
    print()

    results = []
    results.append(("Interface parity (use_sa=False vs True)", test_interface_parity()))
    results.append(("Backward with use_sa=True", test_backward_with_use_sa()))

    print("=" * 60)
    all_passed = all(passed for _, passed in results)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name}")

    if all_passed:
        print("\nAll tests PASSED. Ready to proceed to Stage 2.")
    else:
        print("\nSome tests FAILED. Do not proceed to Stage 2.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
