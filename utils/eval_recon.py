import argparse
import os
import json
import random
from pathlib import Path
import numpy as np
import open3d as o3d
import torch
import trimesh
from tqdm import tqdm

# 导入外部评估库
try:
    from evaluate_3d_reconstruction import run_evaluation
except ImportError:
    print("[Warning] 无法导入 evaluate_3d_reconstruction。请确保克隆了官方仓库并正确配置环境变量。")


# =========================================================
# 1. GauS-SLAM 提供的 2D/3D 几何处理与评估辅助函数
# =========================================================
def normalize(x):
    return x / np.linalg.norm(x)


def get_align_transformation(rec_meshfile, gt_meshfile):
    o3d_rec_mesh = o3d.io.read_triangle_mesh(str(rec_meshfile))
    o3d_gt_mesh = o3d.io.read_triangle_mesh(str(gt_meshfile))
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return reg_p2p.transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).reshape(3, 3)
    ).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat([points, ones], dim=1).reshape(-1, 4, 1).cuda().float()
    cam_cord_homo = w2c @ homo_points
    cam_cord = cam_cord_homo[:, :3]
    cam_cord[:, 0] *= -1
    uv = K.float() @ cam_cord.float()
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (
            (0 <= -z[:, 0, 0].cpu().numpy())
            & (uv[:, 0] < W - edge) & (uv[:, 0] > edge)
            & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
    )
    return mask.sum() > 0


def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def calc_2d_metric(rec_meshfile, gt_meshfile, unseen_gt_pointcloud_file, align=True, n_imgs=1000):
    H, W, focal = 500, 500, 300
    fx, fy = focal, focal
    cx, cy = H / 2.0 - 0.5, W / 2.0 - 0.5

    gt_mesh = o3d.io.read_triangle_mesh(str(gt_meshfile))
    rec_mesh = o3d.io.read_triangle_mesh(str(rec_meshfile))
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    extents, transform = get_cam_position(gt_meshfile)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)  # 建议改为 False 防止弹窗干扰
    vis.get_render_option().mesh_show_back_face = True
    errors = []

    for i in tqdm(range(n_imgs), desc="Calculating 2D Novel View Depth L1"):
        while True:
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx, ty, tz = [round(random.uniform(-10000, +10000), 2) for _ in range(3)]
            target = np.array([tx, ty, tz]) - np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if ~seen:
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)
        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)

        # 渲染 GT 深度
        vis.add_geometry(gt_mesh, reset_bounding_box=True)
        ctr.convert_from_pinhole_camera_parameters(param, True)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = np.asarray(vis.capture_depth_float_buffer(True))
        vis.remove_geometry(gt_mesh, reset_bounding_box=True)

        # 渲染预测深度
        vis.add_geometry(rec_mesh, reset_bounding_box=True)
        ctr.convert_from_pinhole_camera_parameters(param, True)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = np.asarray(vis.capture_depth_float_buffer(True))
        vis.remove_geometry(rec_mesh, reset_bounding_box=True)

        if (ours_depth > 0).sum() > 0:
            errors.append(np.abs(gt_depth[ours_depth > 0] - ours_depth[ours_depth > 0]).mean())

    vis.destroy_window()
    return {"depth l1 (Random Novel View)": np.mean(errors) * 100}


def clean_mesh(mesh):
    print("[*] 开始执行 Mesh Cleaning (去除悬浮碎块)...")
    mesh_tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors),
    )
    components = trimesh.graph.connected_components(edges=mesh_tri.edges_sorted)
    min_len = 200
    components_to_keep = [c for c in components if len(c) >= min_len]

    new_vertices, new_faces, new_colors = [], [], []
    vertex_count = 0
    for component in components_to_keep:
        vertices = mesh_tri.vertices[component]
        colors = mesh_tri.visual.vertex_colors[component]
        index_mapping = {old_idx: vertex_count + new_idx for new_idx, old_idx in enumerate(component)}
        vertex_count += len(vertices)
        faces_in_component = mesh_tri.faces[np.any(np.isin(mesh_tri.faces, component), axis=1)]
        reindexed_faces = np.vectorize(index_mapping.get)(faces_in_component)

        new_vertices.extend(vertices)
        new_faces.extend(reindexed_faces)
        new_colors.extend(colors)

    cleaned_mesh_tri = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    if len(new_colors) > 0:
        cleaned_mesh_tri.visual.vertex_colors = np.array(new_colors)

    cleaned_mesh_tri.update_faces(cleaned_mesh_tri.nondegenerate_faces())
    cleaned_mesh_tri.update_faces(cleaned_mesh_tri.unique_faces())
    print(f"    清理对比: 顶点数 {len(mesh_tri.vertices)} -> {len(cleaned_mesh_tri.vertices)}, "
          f"面片数 {len(mesh_tri.faces)} -> {len(cleaned_mesh_tri.faces)}")

    cleaned_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(cleaned_mesh_tri.vertices),
        o3d.utility.Vector3iVector(cleaned_mesh_tri.faces),
    )
    if hasattr(cleaned_mesh_tri.visual, 'vertex_colors') and len(cleaned_mesh_tri.visual.vertex_colors) > 0:
        vertex_colors = np.asarray(cleaned_mesh_tri.visual.vertex_colors)[:, :3] / 255.0
        cleaned_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors.astype(np.float64))

    return cleaned_mesh


# =========================================================
# 2. 从渲染数据生成 TSDF 网格 (接轨我们之前的代码)
# =========================================================
def generate_tsdf_mesh(render_dir, out_mesh_path, intrinsics_dict):
    print(f"\n[*] 开始执行 TSDF 融合建图...")
    poses_file = os.path.join(render_dir, "render_poses.json")
    with open(poses_file, 'r') as f:
        poses_dict = json.load(f)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics_dict['W'], intrinsics_dict['H'],
        intrinsics_dict['fx'], intrinsics_dict['fy'],
        intrinsics_dict['cx'], intrinsics_dict['cy']
    )

    scale = 1.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=5.0 * scale / 512.0,
        sdf_trunc=0.04 * scale,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    valid_frames = 0
    for idx in tqdm(sorted([int(k) for k in poses_dict.keys()]), desc="TSDF Integration"):
        color_path = os.path.join(render_dir, f"color_{idx:05d}.png")
        depth_path = os.path.join(render_dir, f"depth_{idx:05d}.png")
        if not os.path.exists(color_path) or not os.path.exists(depth_path):
            continue

        color = o3d.io.read_image(color_path)
        depth = o3d.io.read_image(depth_path)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=30.0, convert_rgb_to_intensity=False)

        w2c = np.array(poses_dict[str(idx)])
        volume.integrate(rgbd, intrinsic, w2c)
        valid_frames += 1

    o3d_mesh = volume.extract_triangle_mesh()
    compensate_vector = (-0.0 * scale / 512.0, 2.5 * scale / 512.0, -2.5 * scale / 512.0)
    o3d_mesh = o3d_mesh.translate(compensate_vector)

    os.makedirs(os.path.dirname(out_mesh_path), exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_mesh_path), o3d_mesh)
    print(f"[*] TSDF 原始网格已生成至: {out_mesh_path}")
    return Path(out_mesh_path)


# =========================================================
# 3. 核心评估管线 (整合 GauS-SLAM 逻辑)
# =========================================================
def evaluate_reconstruction_pipeline(
        render_dir: Path,
        gt_mesh_path: Path,
        output_path: Path,
        intrinsics_dict: dict,
        unseen_pc_path: Path = None,
        to_clean=True,
        distance_thresh=0.01,
):
    os.makedirs(output_path / "mesh", exist_ok=True)

    # 1. 生成 TSDF Mesh
    raw_mesh_path = output_path / "mesh" / "raw_tsdf_mesh.ply"
    mesh_path = generate_tsdf_mesh(render_dir, raw_mesh_path, intrinsics_dict)

    # 2. 清理 Mesh (连通域分析去噪)
    if to_clean:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        cleaned_mesh = clean_mesh(mesh)
        cleaned_mesh_path = output_path / "mesh" / "cleaned_mesh.ply"
        o3d.io.write_triangle_mesh(str(cleaned_mesh_path), cleaned_mesh)
        mesh_path = cleaned_mesh_path

    # 3. 计算 3D 核心指标 (精确遵守 GauS-SLAM 调用方式)
    print("\n[*] 调用 evaluate_3d_reconstruction_lib 计算 F-score/Precision/Recall ...")
    # run_evaluation(rec_mesh_name, rec_mesh_dir, gt_mesh_name, distance_thresh, full_path_to_gt_ply, icp_align)
    result_3d = run_evaluation(
        str(mesh_path.parts[-1]),  # 文件名 e.g. "cleaned_mesh.ply"
        str(mesh_path.parent),  # 所在文件夹
        str(gt_mesh_path).split("/")[-1].split(".")[0],  # GT 名字
        distance_thresh=distance_thresh,
        full_path_to_gt_ply=str(gt_mesh_path),
        icp_align=False,
    )

    # 4. 计算 2D 随机视角深度指标
    result_2d = {"depth l1 (Random Novel View)": None}
    if unseen_pc_path and os.path.exists(unseen_pc_path):
        print("\n[*] 计算 2D Random Novel View Depth L1 (耗时较长)...")
        try:
            result_2d = calc_2d_metric(
                rec_meshfile=str(mesh_path),
                gt_meshfile=str(gt_mesh_path),
                unseen_gt_pointcloud_file=str(unseen_pc_path),
                align=True,
                n_imgs=1000
            )
        except Exception as e:
            print(f"[Error] 2D Metric 计算失败: {e}")
    else:
        print("\n[*] 未提供 unseen pointcloud，跳过 2D Random Novel View 评估。")

    # 5. 合并并保存结果
    result = {**result_3d, **result_2d}
    print("\n" + "=" * 55)
    print(f" 最终评估结果 (Threshold: {distance_thresh}m)")
    print("=" * 55)
    for k, v in result.items():
        print(f" {k}: \t{v}")
    print("=" * 55 + "\n")

    with open(str(output_path / "reconstruction_metrics.json"), "w") as f:
        json.dump(result, f, indent=4)
        print(f"[*] 结果已落盘至: {output_path / 'reconstruction_metrics.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-End TSDF Fusion & Evaluation for SLAM")
    parser.add_argument('--render_dir', type=str, required=True,
                        help="保存 color/depth/poses 的 mesh_rendering 文件夹路径")
    parser.add_argument('--gt_mesh', type=str, required=True, help="GT Mesh 路径 (.ply)")
    parser.add_argument('--output_dir', type=str, default="./eval_output", help="输出文件夹")
    parser.add_argument('--unseen_pc', type=str, default=None,
                        help="[可选] 未见区域点云文件路径 (.npy)，用于评测 Novel View")

    # 相机内参
    parser.add_argument('--W', type=int, required=True)
    parser.add_argument('--H', type=int, required=True)
    parser.add_argument('--fx', type=float, required=True)
    parser.add_argument('--fy', type=float, required=True)
    parser.add_argument('--cx', type=float, required=True)
    parser.add_argument('--cy', type=float, required=True)

    # 高级设置
    parser.add_argument('--threshold', type=float, default=0.01, help="评测距离阈值 (GauS-SLAM 默认 0.01m 即 1cm)")
    parser.add_argument('--no_clean', action='store_true', help="如果添加此参数，则不执行 Mesh 去噪清理")

    args = parser.parse_args()

    intrinsics = {'W': args.W, 'H': args.H, 'fx': args.fx, 'fy': args.fy, 'cx': args.cx, 'cy': args.cy}

    evaluate_reconstruction_pipeline(
        render_dir=Path(args.render_dir),
        gt_mesh_path=Path(args.gt_mesh),
        output_path=Path(args.output_dir),
        intrinsics_dict=intrinsics,
        unseen_pc_path=Path(args.unseen_pc) if args.unseen_pc else None,
        to_clean=not args.no_clean,
        distance_thresh=args.threshold
    )