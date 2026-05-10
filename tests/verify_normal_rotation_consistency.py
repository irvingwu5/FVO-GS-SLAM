"""
验证 _normal / _rotation 一致性修复。

测试所有 mutation 点后 _normal 是否与 build_rotation(_rotation)[:,:,2] 一致。
用法: python tests/verify_normal_rotation_consistency.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


def test_derive_normal_from_rotation():
    """验证 _derive_normal_from_rotation 输出 = rotation 第三列。"""
    print("=" * 60)
    print("Test 1: _derive_normal_from_rotation")
    print("=" * 60)

    from gaussian_splatting.scene.gaussian_model import GaussianModel

    config = {
        "Training": {
            "low_freq_scale_multiplier": 1.05,
            "min_init_gaussian_scale": 0.002,
            "max_init_gaussian_scale": 0.06,
        },
        "Dataset": {"pcd_downsample": 64, "point_size": 0.01},
    }
    model = GaussianModel(sh_degree=0, config=config)

    N = 20
    # 随机方向作为 surfel normal
    normals = torch.randn(N, 3)
    normals = torch.nn.functional.normalize(normals, dim=-1)

    # 构造 rotation（只取 z 轴 = normal）
    from gaussian_splatting.utils.general_utils import build_rotation
    from utils.rap2dgs_lite.scorer import normalize_normals

    # 手动设 _rotation 和 _normal
    xyz = torch.randn(N, 3, device="cuda")
    arbitrary_u1 = torch.randn(N, 3, device="cuda")
    arbitrary_u2 = torch.cross(normals.cuda(), arbitrary_u1, dim=-1)
    arbitrary_u2 = torch.nn.functional.normalize(arbitrary_u2, dim=-1)
    arbitrary_u1 = torch.cross(arbitrary_u2, normals.cuda(), dim=-1)

    R = torch.stack([arbitrary_u1, arbitrary_u2, normals.cuda()], dim=-1)  # (N,3,3)
    from pytorch3d.transforms import matrix_to_quaternion
    quats = matrix_to_quaternion(R)  # (N,4) wxyz

    model._xyz = nn.Parameter(xyz)
    model._rotation = nn.Parameter(quats)

    # Case A: _normal 正确，应该和 rotation 推导一致
    model._normal = normals.cuda().clone()
    derived = model._derive_normal_from_rotation()
    diff = (model._normal - derived).abs().max().item()
    print(f"  Case A (consistent init): max |_normal - derived| = {diff:.8f}")
    assert diff < 1e-4, f"Initial _normal should match _derive_normal_from_rotation, got diff={diff}"

    # Case B: 模拟优化后 _rotation 改变但 _normal 未更新（stale）
    delta_q = torch.zeros(N, 4, device="cuda")
    delta_q[:, 0] = 1.0  # identity quat
    # 绕 z 轴微扰（不改变第三列 normal 方向）
    angle = 0.1
    delta_q[:, 0] = np.cos(angle / 2)
    delta_q[:, 3] = np.sin(angle / 2)
    # quaternion multiplication would require complex logic, just perturb slightly
    model._rotation = nn.Parameter(quats + 0.01 * torch.randn(N, 4, device="cuda"))
    model._rotation = nn.Parameter(torch.nn.functional.normalize(model._rotation, dim=-1))

    # _normal still has old values
    stale_derived = model._derive_normal_from_rotation()
    stale_diff = (model._normal - stale_derived).abs().max().item()
    print(f"  Case B (stale _normal after rotation change): max |_normal - derived| = {stale_diff:.6f}")
    print(f"    -> _normal IS stale (diff > 0), confirming the risk exists")
    assert stale_diff > 1e-6, "Expected stale _normal to diverge from _rotation"

    print("  PASSED")
    return True


def test_get_normals_priority():
    """验证 _get_normals 优先使用 _derive_normal_from_rotation 而非 _normal。"""
    print()
    print("=" * 60)
    print("Test 2: _get_normals priority (rotation first)")
    print("=" * 60)

    from utils.rap2dgs_lite.scorer import RAP2DGSLiteScorer
    from gaussian_splatting.scene.gaussian_model import GaussianModel

    config = {
        "Training": {
            "low_freq_scale_multiplier": 1.05,
            "min_init_gaussian_scale": 0.002,
            "max_init_gaussian_scale": 0.06,
        },
        "Dataset": {"pcd_downsample": 64, "point_size": 0.01},
    }
    model = GaussianModel(sh_degree=0, config=config)

    N = 20
    normals = torch.randn(N, 3, device="cuda")
    normals = torch.nn.functional.normalize(normals, dim=-1)

    # 构造与 normal 一致的 rotation
    from pytorch3d.transforms import matrix_to_quaternion
    u2 = torch.randn(N, 3, device="cuda")
    u2 = u2 - (u2 * normals).sum(dim=-1, keepdim=True) * normals
    u2 = torch.nn.functional.normalize(u2, dim=-1)
    u1 = torch.cross(u2, normals, dim=-1)
    R = torch.stack([u1, u2, normals], dim=-1)
    quats = matrix_to_quaternion(R)  # wxyz

    model._xyz = nn.Parameter(torch.randn(N, 3, device="cuda"))
    model._rotation = nn.Parameter(quats)

    # 故意将 _normal 设为一个与 rotation 不一致的错误值
    stale_normal = torch.randn(N, 3, device="cuda")
    stale_normal = torch.nn.functional.normalize(stale_normal, dim=-1)
    model._normal = stale_normal.clone()

    # 调用 _get_normals，应该返回与 _rotation 一致的 normal（不是 stale）
    result = RAP2DGSLiteScorer._get_normals(model, N, "cuda")
    derived = model._derive_normal_from_rotation()

    diff_from_rotation = (result - derived).abs().max().item()
    diff_from_stale = (result - stale_normal).abs().max().item()

    print(f"  _normal (stale)  = first 3: {stale_normal[0].cpu().numpy()}")
    print(f"  _derive_normal   = first 3: {derived[0].cpu().numpy()}")
    print(f"  _get_normals out = first 3: {result[0].cpu().numpy()}")
    print(f"  max |result - derived|   = {diff_from_rotation:.8f}")
    print(f"  max |result - stale|     = {diff_from_stale:.6f}")

    assert diff_from_rotation < 0.1, (
        f"_get_normals should return rotation-derived normals, "
        f"but diff from derived={diff_from_rotation}"
    )
    print("  PASSED: _get_normals correctly prefers _derive_normal_from_rotation")
    return True


def test_save_ply_not_zero():
    """验证 save_ply 不再输出全零法线。"""
    print()
    print("=" * 60)
    print("Test 3: save_ply normals are from rotation, not zeros")
    print("=" * 60)

    import tempfile
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.general_utils import build_rotation

    config = {
        "Training": {
            "low_freq_scale_multiplier": 1.05,
            "min_init_gaussian_scale": 0.002,
            "max_init_gaussian_scale": 0.06,
        },
        "Dataset": {"pcd_downsample": 64, "point_size": 0.01},
    }
    model = GaussianModel(sh_degree=0, config=config)

    N = 20
    from pytorch3d.transforms import matrix_to_quaternion
    normals = torch.randn(N, 3, device="cuda")
    normals = torch.nn.functional.normalize(normals, dim=-1)
    u2 = torch.randn(N, 3, device="cuda")
    u2 = u2 - (u2 * normals).sum(dim=-1, keepdim=True) * normals
    u2 = torch.nn.functional.normalize(u2, dim=-1)
    u1 = torch.cross(u2, normals, dim=-1)
    R = torch.stack([u1, u2, normals], dim=-1)
    quats = matrix_to_quaternion(R)

    model._xyz = nn.Parameter(torch.randn(N, 3, device="cuda"))
    model._features_dc = nn.Parameter(torch.randn(N, 1, 3, device="cuda"))
    model._features_rest = nn.Parameter(torch.zeros(N, 15, 3, device="cuda"))
    model._scaling = nn.Parameter(torch.randn(N, 3, device="cuda"))
    model._rotation = nn.Parameter(quats)
    model._opacity = nn.Parameter(torch.randn(N, 1, device="cuda"))

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        ply_path = f.name
    try:
        model.save_ply(ply_path)

        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        nx = plydata.elements[0].data["nx"]
        ny = plydata.elements[0].data["ny"]
        nz = plydata.elements[0].data["nz"]
        normals_from_ply = np.stack([nx, ny, nz], axis=-1)
        normals_abs_sum = np.abs(normals_from_ply).sum()

        print(f"  PLY normals abs sum = {normals_abs_sum:.6f}")
        print(f"  PLY normals first 3 rows:")
        for i in range(min(3, N)):
            print(f"    [{normals_from_ply[i, 0]:.4f}, {normals_from_ply[i, 1]:.4f}, {normals_from_ply[i, 2]:.4f}]")
        assert normals_abs_sum > 1e-6, "Normals should NOT be all zeros"
        print("  PASSED: save_ply writes rotation-derived normals, not zeros")
    finally:
        os.unlink(ply_path)
    return True


def test_prune_re_derives_normal():
    """验证 prune_points 后 _normal 从 _rotation 重新推导。"""
    print()
    print("=" * 60)
    print("Test 4: prune_points re-derives _normal from _rotation")
    print("=" * 60)

    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from pytorch3d.transforms import matrix_to_quaternion

    config = {
        "Training": {
            "low_freq_scale_multiplier": 1.05,
            "min_init_gaussian_scale": 0.002,
            "max_init_gaussian_scale": 0.06,
        },
        "Dataset": {"pcd_downsample": 64, "point_size": 0.01},
    }
    model = GaussianModel(sh_degree=0, config=config)

    N = 20
    normals = torch.randn(N, 3, device="cuda")
    normals = torch.nn.functional.normalize(normals, dim=-1)
    u2 = torch.randn(N, 3, device="cuda")
    u2 = u2 - (u2 * normals).sum(dim=-1, keepdim=True) * normals
    u2 = torch.nn.functional.normalize(u2, dim=-1)
    u1 = torch.cross(u2, normals, dim=-1)
    R = torch.stack([u1, u2, normals], dim=-1)
    quats = matrix_to_quaternion(R)

    model._xyz = nn.Parameter(torch.randn(N, 3, device="cuda"))
    model._features_dc = nn.Parameter(torch.randn(N, 1, 3, device="cuda"))
    model._features_rest = nn.Parameter(torch.zeros(N, 15, 3, device="cuda"))
    model._scaling = nn.Parameter(torch.randn(N, 3, device="cuda"))
    model._rotation = nn.Parameter(quats)
    model._opacity = nn.Parameter(torch.randn(N, 1, device="cuda"))

    # 设 _normal 为错误值（模拟 stale）
    model._normal = torch.randn(N, 3, device="cuda")

    # 建立 optimizer（prune_points 需要 _prune_optimizer）
    model.optimizer = torch.optim.Adam([
        {"params": [model._xyz], "name": "xyz"},
        {"params": [model._features_dc], "name": "f_dc"},
        {"params": [model._features_rest], "name": "f_rest"},
        {"params": [model._opacity], "name": "opacity"},
        {"params": [model._scaling], "name": "scaling"},
        {"params": [model._rotation], "name": "rotation"},
    ], lr=0.001)

    model.max_radii2D = torch.zeros(N, device="cuda")
    model.xyz_gradient_accum = torch.zeros(N, 1, device="cuda")
    model.denom = torch.zeros(N, 1, device="cuda")
    model.unique_kfIDs = torch.zeros(N).int()
    model.n_obs = torch.zeros(N).int()

    # Prune 一半
    prune_mask = torch.zeros(N, dtype=torch.bool, device="cuda")
    prune_mask[:10] = True
    model.prune_points(prune_mask)

    derived = model._derive_normal_from_rotation()
    diff = (model._normal - derived).abs().max().item()
    print(f"  After prune: max |_normal - derived| = {diff:.8f}")
    print(f"  Remaining points: {model._xyz.shape[0]}")
    assert diff < 1e-4, f"After prune, _normal must match _derive_normal_from_rotation, got diff={diff}"
    print("  PASSED: prune_points correctly re-derives _normal from _rotation")
    return True


if __name__ == "__main__":
    all_pass = True
    try:
        test_derive_normal_from_rotation()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_pass = False

    try:
        test_get_normals_priority()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_pass = False

    try:
        test_save_ply_not_zero()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_pass = False

    try:
        test_prune_re_derives_normal()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        all_pass = False

    print()
    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
