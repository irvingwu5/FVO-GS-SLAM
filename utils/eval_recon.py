import argparse
import os
import json
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree as KDTree


def get_align_transformation(rec_points, gt_points, threshold=0.1):
    """
    使用 ICP 获取从重建点云到 GT 点云的对齐变换矩阵。
    用于补偿 SLAM 系统可能存在的全局坐标系原点漂移。
    """
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(rec_points))
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(gt_points))

    trans_init = np.eye(4)
    # 使用 Point-to-Point ICP 进行全局粗对齐
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation


def evaluate_3d_reconstruction(rec_ply_path, gt_mesh_path, dist_th=0.05, align=False):
    """
    核心评测函数：计算 Precision, Recall, F1-score 以及 Mean Accuracy/Completion。
    默认距离阈值 dist_th = 0.05 (5厘米)。
    """
    print(f"[*] 加载重建点云: {rec_ply_path}")
    try:
        rec_pcd = o3d.io.read_point_cloud(rec_ply_path)
        rec_points = np.asarray(rec_pcd.points)
        if len(rec_points) == 0:
            raise ValueError("重建点云为空！")
    except Exception as e:
        print(f"[Error] 读取重建点云失败: {e}")
        return None

    print(f"[*] 加载真实网格 GT Mesh: {gt_mesh_path}")
    try:
        mesh_gt = trimesh.load(gt_mesh_path, process=False)
        # 从 GT Mesh 表面均匀采样 20 万个点作为绝对 Ground Truth 参照
        gt_points, _ = trimesh.sample.sample_surface(mesh_gt, 200000)
    except Exception as e:
        print(f"[Error] GT Mesh 加载或采样失败: {e}")
        return None

    # 如果需要，执行 ICP 坐标系对齐
    if align:
        print("[*] 正在执行 ICP 全局坐标系对齐...")
        transformation = get_align_transformation(rec_points, gt_points)
        rec_pcd.transform(transformation)
        rec_points = np.asarray(rec_pcd.points)
        print("    对齐完成。")

    print("[*] 构建 KD-Tree 并计算空间几何距离...")
    rec_tree = KDTree(rec_points)
    gt_tree = KDTree(gt_points)

    # =========================================================
    # 1. 计算 Accuracy (Mean) 和 Precision (阈值内)
    # =========================================================
    # 对于每一个重建点，找到离它最近的 GT 点。
    # 距离越小，说明生成的点越准确 (没有飞点)。
    dist_rec_to_gt, _ = gt_tree.query(rec_points)
    acc_mean = np.mean(dist_rec_to_gt) * 100  # 转换为厘米 (cm)
    precision = np.mean((dist_rec_to_gt < dist_th).astype(float)) * 100  # 转换为百分比 (%)

    # =========================================================
    # 2. 计算 Completion (Mean) 和 Recall (阈值内)
    # =========================================================
    # 对于每一个 GT 点，找到离它最近的重建点。
    # 距离越小，说明 GT 的表面被覆盖得越完整 (没有破洞)。
    dist_gt_to_rec, _ = rec_tree.query(gt_points)
    comp_mean = np.mean(dist_gt_to_rec) * 100  # 转换为厘米 (cm)
    recall = np.mean((dist_gt_to_rec < dist_th).astype(float)) * 100  # 转换为百分比 (%)

    # =========================================================
    # 3. 计算 F1-score
    # =========================================================
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    results = {
        "Threshold_m": dist_th,
        "Accuracy_mean_cm": acc_mean,
        "Completion_mean_cm": comp_mean,
        "Precision_pct": precision,
        "Recall_pct": recall,
        "F1_score_pct": f1_score
    }

    # 打印最终结果表
    print("\n" + "=" * 55)
    print(f" 3D Reconstruction Metrics (Threshold: {dist_th * 100:.1f} cm)")
    print("=" * 55)
    print(f" Precision:       {precision:.2f} %  (越高越好，反映飞点噪声率)")
    print(f" Recall:          {recall:.2f} %  (越高越好，反映地图覆盖率)")
    print(f" F1-score:        {f1_score:.2f} %  (综合得分，越高越好)")
    print("-" * 55)
    print(f" Accuracy(Mean):   {acc_mean:.3f} cm (平均精度，越低越好)")
    print(f" Completion(Mean): {comp_mean:.3f} cm (平均完整度，越低越好)")
    print("=" * 55 + "\n")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline 3D Reconstruction Evaluation for SLAM")
    parser.add_argument('--rec_ply', type=str, required=True,
                        help="Path to your reconstructed point_cloud_points.ply")
    parser.add_argument('--gt_mesh', type=str, required=True,
                        help="Path to the dataset's ground truth mesh (.ply or .obj)")
    parser.add_argument('--threshold', type=float, default=0.05,
                        help="Distance threshold for Precision/Recall in meters (default: 0.05m = 5cm)")
    parser.add_argument('--align', action='store_true',
                        help="Use ICP to align reconstructed point cloud to GT mesh before evaluation")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save the JSON result")

    args = parser.parse_args()

    if not os.path.exists(args.rec_ply):
        print(f"Error: Reconstructed PLY not found at '{args.rec_ply}'")
        exit(1)

    if not os.path.exists(args.gt_mesh):
        print(f"Error: Ground Truth Mesh not found at '{args.gt_mesh}'")
        exit(1)

    # 运行评估
    metrics_result = evaluate_3d_reconstruction(
        rec_ply_path=args.rec_ply,
        gt_mesh_path=args.gt_mesh,
        dist_th=args.threshold,
        align=args.align
    )

    # 结果落盘
    if metrics_result is not None and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        out_file = os.path.join(args.output_dir, "3d_reconstruction_metrics.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_result, f, indent=4)
        print(f"[*] 评测结果已保存至: {out_file}")