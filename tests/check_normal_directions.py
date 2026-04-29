"""
2DGS 各阶段法线方向检查脚本（读取实际运行结果）
检查：ckpt 中 surfel FDN 法线、渲染法线、GT 传感器法线是否一致

用法：
    python tests/check_normal_directions.py --result_dir results/tum_results/tum_rgbd_dataset_freiburg3_long_office_household/2026-04-29-10-00-34/
    python tests/check_normal_directions.py --result_dir results/replica_results/datasets_replica/2026-04-29-...
"""

import argparse
import os
import sys
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def normal_to_rgb(normal_map):
    """法线 [3,H,W] → RGB [H,W,3] 可视化 (R=X, G=Y, B=Z, 映射 [-1,1]→[0,255])"""
    rgb = ((normal_map * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    return rgb


def save_comparison(images, titles, save_path, ncols=3):
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    h, w = images[0].shape[:2]
    canvas = np.zeros((h * nrows + 30 * nrows, w * ncols, 3), dtype=np.uint8)
    for i, (img, title) in enumerate(zip(images, titles)):
        r, c = i // ncols, i % ncols
        y0 = r * (h + 30)
        x0 = c * w
        canvas[y0:y0 + h, x0:x0 + w] = img
        cv2.putText(canvas, title, (x0 + 5, y0 + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f"Saved: {save_path}")


def load_ckpt_and_render(result_dir, submap_id=0, kf_ids=None):
    """从运行结果加载子图 ckpt，选关键帧渲染并返回法线对比数据。"""
    submaps_dir = os.path.join(result_dir, "submaps")
    ckpt_path = os.path.join(submaps_dir, f"{submap_id:06d}.ckpt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: ckpt not found: {ckpt_path}")
        return None

    print(f"\n===== Loading: {ckpt_path} =====")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ---- 1. 提取 surfel FDN 法线 ----
    gp = ckpt["gaussian_params"]
    xyz = gp["_xyz"]  # [N, 3]
    rot_q = gp["_rotation"]  # [N, 4] quaternion
    from gaussian_splatting.utils.general_utils import build_rotation
    rot_mat = build_rotation(rot_q)
    fdn_normals_world = rot_mat[:, :, 2].detach().cpu().numpy()  # [N, 3] FDN normal in world frame
    N = len(xyz)

    norms = np.linalg.norm(fdn_normals_world, axis=1)
    print(f"Surfels: {N}")
    print(f"FDN normal range: "
          f"x=[{fdn_normals_world[:,0].min():.3f}, {fdn_normals_world[:,0].max():.3f}] "
          f"y=[{fdn_normals_world[:,1].min():.3f}, {fdn_normals_world[:,1].max():.3f}] "
          f"z=[{fdn_normals_world[:,2].min():.3f}, {fdn_normals_world[:,2].max():.3f}]")
    print(f"FDN normal magnitude: mean={norms.mean():.4f} std={norms.std():.4f}")

    # 检查法线是否指向主要方向（对 Replica/TUM 室内场景，法线应分布在各方向）
    z_neg_ratio = (fdn_normals_world[:, 2] < -0.5).mean()
    z_pos_ratio = (fdn_normals_world[:, 2] > 0.5).mean()
    print(f"FDN z< -0.5 (pointing down): {z_neg_ratio:.1%}")
    print(f"FDN z> 0.5 (pointing up)  : {z_pos_ratio:.1%}")

    # ---- 2. 选择关键帧 ----
    kf_poses = ckpt.get("submap_keyframe_poses", {})
    kf_list = sorted([int(k) for k in kf_poses.keys()])
    print(f"Keyframes in ckpt: {len(kf_list)}")

    if kf_ids is None:
        # 选择首、中、尾三个关键帧
        kf_ids = [kf_list[0], kf_list[len(kf_list) // 2], kf_list[-1]]

    # ---- 3. 对每个关键帧渲染法线 ----
    from gaussian_splatting.gaussian_renderer import render
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from utils.camera_utils import Camera
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    import torch.nn as nn

    class Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        depth_ratio = 1.0
        debug = False

    pipe = Pipe()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    # Build Gaussian model from ckpt
    gm = GaussianModel(sh_degree=0)
    gm._xyz = nn.Parameter(gp["_xyz"].cuda())
    gm._features_dc = nn.Parameter(gp["_features_dc"].cuda())
    gm._features_rest = nn.Parameter(gp.get("_features_rest",
                                    torch.zeros(N, 15, 3)).cuda())
    gm._opacity = nn.Parameter(gp["_opacity"].cuda().unsqueeze(-1))
    gm._scaling = nn.Parameter(gp["_scaling"].cuda())
    gm._rotation = nn.Parameter(gp["_rotation"].cuda())
    gm._normal = nn.Parameter(torch.from_numpy(fdn_normals_world).float().cuda())
    gm.max_radii2D = torch.zeros(N, device="cuda")
    gm.xyz_gradient_accum = torch.zeros(N, 1, device="cuda")
    gm.denom = torch.zeros(N, 1, device="cuda")
    gm.unique_kfIDs = torch.zeros(N, device="cuda").int()
    gm.n_obs = torch.zeros(N, device="cuda").int()

    # Read camera intrinsics from ckpt or use defaults
    kf_imgs_dir = submaps_dir  # images saved alongside ckpt

    results = []
    for kf_id in kf_ids:
        print(f"\n--- Keyframe {kf_id} ---")

        # Load keyframe image
        img_path = os.path.join(kf_imgs_dir, f"{submap_id:06d}_img_{kf_id}.pt")
        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}")
            continue

        img_tensor = torch.load(img_path, map_location="cpu")
        if img_tensor.shape[0] == 3:
            H, W = img_tensor.shape[1], img_tensor.shape[2]
        else:
            print(f"  Unexpected image shape: {img_tensor.shape}")
            continue

        # Get keyframe pose
        c2w = kf_poses.get(str(kf_id)) or kf_poses.get(kf_id)
        if c2w is None:
            print(f"  Pose not found for kf {kf_id}")
            continue
        if isinstance(c2w, torch.Tensor):
            c2w = c2w.numpy()
        c2w = np.array(c2w, dtype=np.float64)
        w2c = np.linalg.inv(c2w)

        # Build minimal Camera
        fx = 525.0
        fy = 525.0
        cx = W / 2.0
        cy = H / 2.0
        proj_mat = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)

        viewpoint = Camera(
            uid=kf_id, color=img_tensor, depth=None,
            gt_T=torch.from_numpy(w2c).float(),
            dynamic_intrinsic=None, projection_matrix=proj_mat,
            fx=fx, fy=fy, cx=cx, cy=cy,
            fovx=float(2 * np.arctan(W / (2 * fx))),
            fovy=float(2 * np.arctan(H / (2 * fy))),
            image_height=H, image_width=W, device="cuda",
        )
        viewpoint.T = torch.from_numpy(w2c).float().cuda()

        # Render
        with torch.no_grad():
            render_pkg = render(viewpoint, gm, pipe, bg_color, surf=False)

        rend_rgb = render_pkg["render"].detach().cpu().numpy()
        rend_normal = render_pkg.get("rend_normal")
        rend_depth = render_pkg["depth"].detach().cpu().numpy()

        if rend_normal is not None:
            rn = rend_normal.detach().cpu().numpy()
            print(f"  rend_normal z-mean: {rn[2].mean():.3f} (negative=away from camera)")
            # 检查法线方向：在相机帧中，rend_normal z < 0 表示指向相机后方（远离）
            # 对于可见表面，法线应指向相机（z approx -1）
            z_neg_pct = (rn[2] < 0).mean()
            print(f"  rend_normal z<0 ratio: {z_neg_pct:.1%}")
            print(f"  rend_normal norm mean: {np.sqrt((rn**2).sum(axis=0)).mean():.4f}")
        else:
            rn = None
            print("  WARNING: rend_normal not in render output!")

        # GT RGB
        gt_rgb = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Rendered RGB
        rend_rgb_viz = (rend_rgb.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)

        results.append({
            "kf_id": kf_id,
            "gt_rgb": gt_rgb,
            "rend_rgb": rend_rgb_viz,
            "rend_normal": rn,
            "rend_depth": rend_depth,
        })

    del ckpt
    return results, fdn_normals_world, N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True,
                        help="Path to run results dir, e.g. results/tum_results/.../2026-04-29-10-00-34/")
    parser.add_argument("--submap_id", type=int, default=0,
                        help="Submap ID to inspect (default: 0)")
    parser.add_argument("--kf_ids", type=int, nargs="*", default=None,
                        help="Specific keyframe IDs (default: first/mid/last)")
    args = parser.parse_args()

    data = load_ckpt_and_render(args.result_dir, args.submap_id, args.kf_ids)
    if data is None:
        return
    kf_results, fdn_world, N = data

    # Build comparison images
    for r in kf_results:
        viz_images = []
        viz_titles = []

        # 1. GT RGB
        viz_images.append(r["gt_rgb"])
        viz_titles.append(f"KF{r['kf_id']}_GT_RGB")

        # 2. Rendered RGB
        viz_images.append(r["rend_rgb"])
        viz_titles.append(f"KF{r['kf_id']}_Rendered")

        # 3. Rendered normal
        if r["rend_normal"] is not None:
            viz_images.append(normal_to_rgb(r["rend_normal"]))
            titlesuffix = ""
            z_neg = (r["rend_normal"][2] < 0).mean()
            if z_neg < 0.1:
                titlesuffix = " [FLIPPED? z>0 dominant]"
            viz_titles.append(f"KF{r['kf_id']}_RendNormal{z_neg:.0%}z<0{titlesuffix}")

        # 4. Rendered depth
        depth_viz = (r["rend_depth"] / max(r["rend_depth"].max(), 1e-8) * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz.squeeze(), cv2.COLORMAP_INFERNO)
        viz_images.append(depth_viz)
        viz_titles.append(f"KF{r['kf_id']}_Depth")

        save_path = os.path.join(args.result_dir, "normal_debug",
                                 f"submap{args.submap_id}_kf{r['kf_id']}.png")
        save_comparison(viz_images, viz_titles, save_path)

    print(f"\nDone. Check {args.result_dir}/normal_debug/")


if __name__ == "__main__":
    main()
