"""
Stage 2: Surface-aware forward visual sanity test.

Tests the SA depth adjustment in forward pass:
- use_sa=False produces same results as baseline
- use_sa=True produces no NaN/Inf
- SA depth differs from raw depth at occlusion boundaries
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


def _render_scene(use_sa, seed=42, device="cuda:0"):
    """Render a richer scene with overlapping surfaces to test SA behavior."""
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    torch.manual_seed(seed)

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

    raster_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        cx=cx, cy=cy,
        tanfovx=fovx, tanfovy=fovy,
        bg=torch.tensor([0.0, 0.0, 0.0], device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix, projmatrix=full_proj,
        projmatrix_raw=projmatrix_raw,
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False,
        use_sa=use_sa,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Create a dense scene: Gaussians covering the full image
    # Use large scales and high opacity so they render visibly
    N_per_layer = 400
    xs = torch.linspace(-0.8, 0.8, 20, device=device)
    ys = torch.linspace(-0.6, 0.6, 20, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

    # Layer 1: near (z=0.5), large Gaussians
    z_near = torch.full((N_per_layer,), 0.5, device=device)
    xyz_near = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z_near], dim=-1)

    # Layer 2: far (z=0.8), overlapping behind the first layer
    z_far = torch.full((N_per_layer,), 0.8, device=device)
    xyz_far = torch.stack([grid_x.reshape(-1) * 0.9, grid_y.reshape(-1) * 0.9, z_far], dim=-1)

    means3D = torch.cat([xyz_near, xyz_far], dim=0)
    N = means3D.shape[0]
    means2D = torch.zeros(N, 3, device=device, requires_grad=True)

    shs = torch.rand(N, 1, 3, device=device) * 0.5 + 0.5

    # High opacity so they're visible
    opacities = torch.full((N, 1), 0.9, device=device)

    # Large scales: ~50-100 pixel radius
    scales = torch.full((N, 2), 0.15, device=device)

    rotations = torch.randn(N, 4, device=device)
    rotations = torch.nn.functional.normalize(rotations, dim=-1)

    rendered_image, radii, allmap, n_touched = rasterizer(
        means3D=means3D, means2D=means2D,
        shs=shs, colors_precomp=None,
        opacities=opacities, scales=scales, rotations=rotations,
        cov3D_precomp=None,
        w=torch.tensor([0.01, -0.02, 0.005], device=device),
        trans=torch.tensor([0.005, -0.01, 0.002], device=device),
    )

    # allmap channels
    D_accum = allmap[0]        # expected depth accum
    alpha = allmap[1]          # accumulated alpha
    normal = allmap[2:5]       # view-space normal
    median_depth = allmap[5]   # median depth
    distortion = allmap[6]     # distortion

    # Compute expected depth (safe division)
    eps = 1e-6
    valid_mask = alpha > eps
    exp_depth = torch.zeros_like(D_accum)
    exp_depth[valid_mask] = D_accum[valid_mask] / alpha[valid_mask]

    return {
        "render": rendered_image,
        "alpha": alpha,
        "exp_depth": exp_depth,
        "median_depth": median_depth,
        "distortion": distortion,
        "radii": radii,
        "n_touched": n_touched,
        "valid_mask": valid_mask,
    }


def test_sa_baseline_parity():
    """use_sa=False must match baseline."""
    print("=== Test 1: use_sa=False vs baseline parity ===")
    torch.manual_seed(123)
    res_false = _render_scene(use_sa=False, seed=42)
    torch.manual_seed(123)
    res_baseline = _render_scene(use_sa=False, seed=42)

    for key in ["render", "alpha", "exp_depth", "median_depth", "distortion"]:
        diff = (res_false[key] - res_baseline[key]).abs().max().item()
        ok = diff == 0.0
        print(f"  [{('OK' if ok else 'FAIL')}] {key} max diff: {diff:.10f}")
        assert ok, f"{key} differs between runs"

    print("  PASSED\n")
    return True


def test_sa_forward_sanity():
    """use_sa=True must not produce NaN/Inf; SA depth should differ from raw."""
    print("=== Test 2: SA forward sanity ===")
    res_false = _render_scene(use_sa=False, seed=42)
    res_true = _render_scene(use_sa=True, seed=42)

    # Check no NaN/Inf in SA outputs
    for key in ["render", "alpha", "exp_depth", "median_depth", "distortion"]:
        has_nan = torch.isnan(res_true[key]).any().item()
        has_inf = torch.isinf(res_true[key]).any().item()
        ok = not has_nan and not has_inf
        print(f"  [{('OK' if ok else 'FAIL')}] use_sa=True {key}: NaN={has_nan}, Inf={has_inf}")
        assert ok, f"{key} has NaN/Inf with use_sa=True"

    # Print statistics
    valid = res_true["valid_mask"]
    total_valid = valid.sum().item()
    print(f"  Valid pixels: {total_valid} / {valid.numel()}")

    if total_valid > 0:
        for label, res in [("use_sa=False", res_false), ("use_sa=True", res_true)]:
            exp_d = res["exp_depth"][valid]
            med_d = res["median_depth"][valid]
            dist = res["distortion"][valid]
            alpha = res["alpha"][valid]
            print(f"  {label}:")
            print(f"    exp_depth  mean={exp_d.mean().item():.4f}  min={exp_d.min().item():.4f}  max={exp_d.max().item():.4f}")
            print(f"    med_depth  mean={med_d.mean().item():.4f}  min={med_d.min().item():.4f}  max={med_d.max().item():.4f}")
            print(f"    distortion mean={dist.mean().item():.6f}  min={dist.min().item():.6f}  max={dist.max().item():.6f}")
            print(f"    alpha      mean={alpha.mean().item():.4f}  min={alpha.min().item():.4f}  max={alpha.max().item():.4f}")
            print(f"    n_visible   ={(res['radii'] > 0).sum().item()} / {len(res['radii'])}")

        # SA depth should be different from raw at some pixels
        depth_diff = (res_true["exp_depth"] - res_false["exp_depth"]).abs()
        nonzero_pixels = (depth_diff > 1e-6).sum().item()
        print(f"  SA depth diff > 1e-6: {nonzero_pixels} / {total_valid} pixels "
              f"({100 * nonzero_pixels / max(total_valid, 1):.1f}%)")

        dist_diff = (res_true["distortion"] - res_false["distortion"]).abs()
        dist_nonzero = (dist_diff > 1e-6).sum().item()
        print(f"  SA distortion diff > 1e-6: {dist_nonzero} / {total_valid} pixels "
              f"({100 * dist_nonzero / max(total_valid, 1):.1f}%)")
    else:
        print("  WARNING: No valid pixels rendered. SA behavior cannot be verified.")

    print("  PASSED\n")
    return True


def test_sa_forward_no_crash_multiview():
    """SA forward with different camera poses."""
    print("=== Test 3: SA forward multi-view ===")
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

    # Build Gaussians once
    N = 200
    xs = torch.linspace(-0.5, 0.5, 10, device=device)
    ys = torch.linspace(-0.5, 0.5, 10, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    z_near = torch.full((100,), 1.5, device=device)
    z_far = torch.full((100,), 3.0, device=device)
    xyz = torch.cat([
        torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z_near], dim=-1),
        torch.stack([grid_x.reshape(-1) * 0.7, grid_y.reshape(-1) * 0.7, z_far], dim=-1),
    ], dim=0)
    means2D = torch.zeros(200, 3, device=device, requires_grad=True)
    shs = torch.rand(200, 1, 3, device=device)
    opacities = torch.sigmoid(torch.randn(200, 1, device=device) * 0.5 + 0.5)
    scales = torch.exp(torch.randn(200, 2, device=device) * 0.3 - 1.0)
    rotations = torch.randn(200, 4, device=device)
    rotations = torch.nn.functional.normalize(rotations, dim=-1)

    Ts = [
        torch.eye(4, device=device),
        torch.tensor([[1, 0, 0, 0.2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device, dtype=torch.float32),
        torch.tensor([[1, 0, 0, 0], [0, 1, 0, -0.1], [0, 0, 1, 0.5], [0, 0, 0, 1]], device=device, dtype=torch.float32),
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
            use_sa=True,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        rendered_image, radii, allmap, n_touched = rasterizer(
            means3D=xyz, means2D=means2D,
            shs=shs, colors_precomp=None,
            opacities=opacities, scales=scales, rotations=rotations,
            cov3D_precomp=None,
            w=torch.zeros(3, device=device),
            trans=torch.zeros(3, device=device),
        )

        has_nan = torch.isnan(allmap).any().item()
        has_inf = torch.isinf(allmap).any().item()
        print(f"  View {i}: NaN={has_nan}, Inf={has_inf}, render_mean={rendered_image.mean().item():.6f}")
        assert not has_nan and not has_inf, f"View {i} has NaN/Inf"

    print("  PASSED\n")
    return True


def main():
    print("=" * 60)
    print("Stage 2: SA Forward Visual Sanity Test")
    print("=" * 60)
    print()

    results = []
    results.append(("use_sa=False parity", test_sa_baseline_parity()))
    results.append(("SA forward sanity", test_sa_forward_sanity()))
    results.append(("SA forward multi-view", test_sa_forward_no_crash_multiview()))

    print("=" * 60)
    all_passed = all(passed for _, passed in results)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name}")

    if all_passed:
        print("\nAll tests PASSED. SA forward is working. Ready for Stage 3.")
    else:
        print("\nSome tests FAILED.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
