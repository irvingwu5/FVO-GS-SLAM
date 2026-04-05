import open3d as o3d
import numpy as np
import json
import os
import glob
import argparse


def extract_mesh_tsdf(render_dir, out_mesh_path, voxel_size=0.01, fx=300.0, fy=300.0, cx=250.0, cy=250.0, W=500, H=500):
    # 【修改】：直接读取 rendering 文件夹里的专属位姿文件
    poses_path = os.path.join(render_dir, "render_poses.json")
    print(f"[*] 正在读取渲染位姿文件: {poses_path}")
    with open(poses_path, 'r') as f:
        pose_dict = json.load(f)

    print(f"[*] 正在初始化 TSDF Volume (体素大小: {voxel_size}m) ...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 5.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx, fy=fy, cx=cx, cy=cy)

    depth_files = sorted(glob.glob(os.path.join(render_dir, "depth_*.png")))
    color_files = sorted(glob.glob(os.path.join(render_dir, "color_*.png")))

    if len(depth_files) == 0:
        print(f"[Error] 在 {render_dir} 目录下没有找到 depth_*.png 文件！")
        return

    print(f"[*] 开始融合 {len(depth_files)} 帧深度图 ...")
    for depth_path, color_path in zip(depth_files, color_files):
        idx_str = os.path.basename(depth_path).split('_')[1].split('.')[0]

        # JSON 里的 key 是字符串
        uid_str = str(int(idx_str))

        if uid_str not in pose_dict:
            continue

        depth = o3d.io.read_image(depth_path)
        color = o3d.io.read_image(color_path)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )

        # =========================================================
        # 【至简提取】：render_poses 存的就是 W2C，直接用！
        # =========================================================
        w2c_matrix = np.array(pose_dict[uid_str])

        volume.integrate(rgbd, intrinsic, w2c_matrix)

    print("[*] 正在执行 Marching Cubes 提取 Mesh ...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"[*] 保存最终 Mesh 至: {out_mesh_path}")
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    print("[*] TSDF Fusion 提取完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TSDF Fusion Mesh Extraction")
    parser.add_argument('--render_dir', type=str, required=True,
                        help="包含 depth_*.png 和 render_poses.json 的 rendering 文件夹路径")
    parser.add_argument('--out_mesh', type=str, required=True, help="提取出的 mesh 保存路径 (如 recon_mesh.ply)")

    parser.add_argument('--W', type=int, default=1200)
    parser.add_argument('--H', type=int, default=680)
    parser.add_argument('--fx', type=float, default=600.0)
    parser.add_argument('--fy', type=float, default=600.0)
    parser.add_argument('--cx', type=float, default=599.5)
    parser.add_argument('--cy', type=float, default=339.5)

    args = parser.parse_args()

    extract_mesh_tsdf(
        render_dir=args.render_dir,
        out_mesh_path=args.out_mesh,
        W=args.W, H=args.H, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy
    )