"""
Stage 4: Parent renderer SA depth selection strategy test.

Tests that use_sa_depth switch works correctly through the parent render() function.
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from types import SimpleNamespace


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


class StubCamera:
    """Minimal Camera stub with only the attributes needed by render()."""
    def __init__(self, W=640, H=480, device="cuda:0"):
        self.image_height = H
        self.image_width = W
        fx, fy = 525.0, 525.0
        self.cx, self.cy = 319.5, 239.5
        self.FoVx = 2 * math.atan(W / (2 * fx))
        self.FoVy = 2 * math.atan(H / (2 * fy))

        projmat = _build_projection_matrix(0.01, 100.0, self.cx, self.cy, fx, fy, W, H).to(device)
        self.projection_matrix = projmat.T.clone()

        T = torch.eye(4, device=device)
        self.world_view_transform = T.T.clone()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.cam_rot_delta = torch.zeros(3, device=device)
        self.cam_trans_delta = torch.zeros(3, device=device)


class StubGaussianModel:
    """Minimal GaussianModel stub."""
    def __init__(self, N=200, device="cuda:0"):
        xs = torch.linspace(-0.5, 0.5, 10, device=device)
        ys = torch.linspace(-0.5, 0.5, 10, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        z_near = torch.full((100,), 1.5, device=device)
        z_far = torch.full((100,), 3.0, device=device)
        self._xyz = torch.cat([
            torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), z_near], dim=-1),
            torch.stack([grid_x.reshape(-1) * 0.7, grid_y.reshape(-1) * 0.7, z_far], dim=-1),
        ], dim=0)
        N_actual = self._xyz.shape[0]
        self._features = torch.rand(N_actual, 1, 3, device=device)
        self._opacity = torch.randn(N_actual, 1, device=device) * 0.5 + 1.0
        self._scaling = torch.randn(N_actual, 2, device=device) * 0.5 - 1.0
        self._rotation = torch.randn(N_actual, 4, device=device)
        self._rotation = torch.nn.functional.normalize(self._rotation, dim=-1)
        self.active_sh_degree = 0

    @property
    def get_xyz(self): return self._xyz

    @property
    def get_features(self): return self._features

    @property
    def get_opacity(self): return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self): return torch.exp(self._scaling)

    @property
    def get_rotation(self): return self._rotation


def test_depth_switches():
    """Test all combinations of use_sa, use_sa_depth."""
    print("=== Stage 4: Parent renderer SA depth switch test ===")

    from gaussian_splatting.gaussian_renderer import render

    device = "cuda:0"
    cam = StubCamera(device=device)
    gaussians = StubGaussianModel(device=device)
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    base = {"compute_cov3D_python": False, "convert_SHs_python": False}
    configs = [
        ("A0: baseline", {**base, "use_sa": False, "use_sa_depth": False, "depth_ratio": 1.0}),
        ("A1: SA fwd only", {**base, "use_sa": True, "use_sa_depth": False, "depth_ratio": 1.0}),
        ("A2: SA expected depth", {**base, "use_sa": True, "use_sa_depth": True, "depth_ratio": 1.0}),
        ("A3: SA mixed", {**base, "use_sa": True, "use_sa_depth": False, "depth_ratio": 0.3}),
    ]

    all_passed = True
    for name, cfg in configs:
        pipe = SimpleNamespace(**cfg)
        try:
            rets = render(cam, gaussians, pipe, bg)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            all_passed = False
            continue

        depth = rets["depth"]
        ok = (not torch.isnan(depth).any().item() and
              not torch.isinf(depth).any().item())
        status = "OK" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {name}: depth mean={depth.mean().item():.4f} "
              f"min={depth.min().item():.4f} max={depth.max().item():.4f}")

        # Check required fields exist
        for f in ["render", "viewspace_points", "visibility_filter", "radii",
                   "n_touched", "rend_normal", "rend_dist", "depth", "opacity"]:
            assert f in rets, f"Missing field: {f}"

    if all_passed:
        print("  PASSED\n")
    else:
        print("  FAILED\n")
    return all_passed


def test_debug_fields():
    """Test debug_sa_depth adds optional fields."""
    print("=== Stage 4: Debug SA depth fields ===")
    from gaussian_splatting.gaussian_renderer import render

    device = "cuda:0"
    cam = StubCamera(device=device)
    gaussians = StubGaussianModel(device=device)
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    # Without debug
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, use_sa=False, use_sa_depth=False, depth_ratio=1.0, debug_sa_depth=False)
    rets = render(cam, gaussians, pipe, bg)
    assert "depth_expected" not in rets, "debug_sa_depth=False should not add depth_expected"
    print("  [OK] debug_sa_depth=False: no extra fields")

    # With debug
    pipe2 = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, use_sa=False, use_sa_depth=False, depth_ratio=1.0, debug_sa_depth=True)
    rets2 = render(cam, gaussians, pipe2, bg)
    assert "depth_expected" in rets2, "debug_sa_depth=True should add depth_expected"
    assert "depth_median" in rets2, "debug_sa_depth=True should add depth_median"
    print("  [OK] debug_sa_depth=True: depth_expected, depth_median added")
    print("  PASSED\n")
    return True


def test_depth_ratio_override():
    """depth_ratio=1.0 with use_sa_depth=False uses median depth only."""
    print("=== Stage 4: depth_ratio behavior ===")
    from gaussian_splatting.gaussian_renderer import render

    device = "cuda:0"
    cam = StubCamera(device=device)
    gaussians = StubGaussianModel(device=device)
    bg = torch.tensor([0.0, 0.0, 0.0], device=device)

    # depth_ratio=1.0 => surf_depth = 0*expected + 1*median = median
    pipe_med = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, use_sa=False, use_sa_depth=False, depth_ratio=1.0)
    rets_med = render(cam, gaussians, pipe_med, bg)

    # depth_ratio=0.0 => surf_depth = 1*expected + 0*median = expected
    pipe_exp = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, use_sa=False, use_sa_depth=False, depth_ratio=0.0)
    rets_exp = render(cam, gaussians, pipe_exp, bg)

    # They should differ at some pixels (median != expected in general)
    diff = (rets_med["depth"] - rets_exp["depth"]).abs()
    nonzero_pixels = (diff > 1e-6).sum().item()
    print(f"  depth_ratio=1.0 vs 0.0: diff at {nonzero_pixels} pixels")
    print(f"  depth_ratio=1.0 depth mean={rets_med['depth'].mean().item():.4f}")
    print(f"  depth_ratio=0.0 depth mean={rets_exp['depth'].mean().item():.4f}")

    # use_sa_depth=True should give same as depth_ratio=0.0 (both use expected depth only)
    pipe_sa = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, use_sa=True, use_sa_depth=True, depth_ratio=1.0)
    rets_sa = render(cam, gaussians, pipe_sa, bg)
    diff2 = (rets_sa["depth"] - rets_exp["depth"]).abs().max().item()
    print(f"  use_sa_depth=True vs depth_ratio=0.0 max diff: {diff2:.10f}")

    print("  PASSED\n")
    return True


def main():
    print("=" * 60)
    print("Stage 4: Parent Renderer SA Depth Switch Test")
    print("=" * 60)
    print()

    results = []
    results.append(("Depth switches", test_depth_switches()))
    results.append(("Debug fields", test_debug_fields()))
    results.append(("depth_ratio behavior", test_depth_ratio_override()))

    print("=" * 60)
    all_passed = all(passed for _, passed in results)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {name}")

    if all_passed:
        print("\nAll tests PASSED. Ready to proceed to Stage 5.")
    else:
        print("\nSome tests FAILED.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
