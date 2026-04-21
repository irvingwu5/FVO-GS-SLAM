import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)
    return ape_stat


def _load_submap_correct_tsfms(save_dir):
    """
    从磁盘加载所有子图的 PGO 修正矩阵 correct_tsfm。

    Args:
        save_dir: 结果保存根目录，其下应有 submaps/ 子目录存放 *.ckpt 文件。

    Returns:
        dict: {submap_id (int): correct_tsfm (np.ndarray, 4x4)}
              如果 submaps 目录不存在或为空，返回空字典。
    """
    import glob
    submaps_dir = os.path.join(save_dir, "submaps")
    correct_tsfms = {}
    if not os.path.isdir(submaps_dir):
        return correct_tsfms
    for ckpt_path in sorted(glob.glob(os.path.join(submaps_dir, "*.ckpt"))):
        sid = int(os.path.basename(ckpt_path).split('.')[0])
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            tsfm = ckpt.get("correct_tsfm", np.eye(4))
            if isinstance(tsfm, torch.Tensor):
                tsfm = tsfm.numpy()
            correct_tsfms[sid] = np.array(tsfm, dtype=np.float64)
            del ckpt
        except Exception as e:
            Log(f"[eval_ate] 读取子图 {sid} 的 correct_tsfm 失败: {e}")
            correct_tsfms[sid] = np.eye(4)
    return correct_tsfms

def _rebuild_submap_anchors_from_ckpts(save_dir):
    submaps_dir = os.path.join(save_dir, "submaps")
    if not os.path.isdir(submaps_dir):
        return {}

    ckpt_files = sorted(
        [os.path.join(submaps_dir, f) for f in os.listdir(submaps_dir) if f.endswith(".ckpt")]
    )
    if len(ckpt_files) == 0:
        return {}

    sid_to_ckpt = {
        int(os.path.basename(p).split(".")[0]): p
        for p in ckpt_files
    }

    all_sids = sorted(sid_to_ckpt.keys())
    anchors = {all_sids[0]: np.eye(4)}

    for i in range(1, len(all_sids)):
        prev_sid = all_sids[i - 1]
        curr_sid = all_sids[i]

        curr_ckpt = torch.load(sid_to_ckpt[curr_sid], map_location="cpu")
        if "prev_submap_tsfm_refined" in curr_ckpt:
            rel_prev_from_curr = np.array(curr_ckpt["prev_submap_tsfm_refined"], dtype=np.float64)
        else:
            prev_ckpt = torch.load(sid_to_ckpt[prev_sid], map_location="cpu")
            rel_prev_from_curr = np.array(
                prev_ckpt.get("next_submap_relative_pose", prev_ckpt.get("relative_pose", np.eye(4))),
                dtype=np.float64
            )

        anchors[curr_sid] = anchors[prev_sid] @ rel_prev_from_curr

    return anchors

def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False,
             frame_to_submap=None, submap_anchor_poses=None,
             cameras_already_global=False):
    """
    评估绝对轨迹误差 (ATE)。

    新增参数:
        cameras_already_global (bool):
            如果为 True，表示 frames 中的 cam.T 已经被上游（如 slam.py 的
            轨迹拉扯步骤）修正为全局坐标系下的 W2C 矩阵，此时直接使用
            inv(cam.T) 作为全局 C2W，不再叠加 anchor 和 correct_tsfm，
            避免重复变换。

            如果为 False（默认），表示 cam.T 仍然是子图局部坐标系下的 W2C，
            需要依次叠加：
              1. submap_anchor_poses[sid]  —— 开环锚点（前端累积的全局 C2W）
              2. correct_tsfm[sid]         —— PGO 闭环修正矩阵
            来还原全局 C2W。
    """
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    # ------------------------------------------------------------------
    # 如果需要在线拼接（cameras_already_global=False），预加载 PGO 修正矩阵
    # ------------------------------------------------------------------
    correct_tsfms = {} #初始化一个空字典 correct_tsfms 用于存放每个子图的 4x4 修正矩阵
    rebuilt_anchor_poses = {}
    if save_dir is not None and frame_to_submap is not None:
        rebuilt_anchor_poses = _rebuild_submap_anchors_from_ckpts(save_dir)
    if not cameras_already_global and frame_to_submap is not None: #false表示当前frames中的相机位姿仍是子图局部坐标,表示存在子图分配信息
        correct_tsfms = _load_submap_correct_tsfms(save_dir)

    for kf_id in kf_ids:
        kf = frames[kf_id]

        # 1. 获取局部坐标系下的 C2W 矩阵
        local_c2w = np.linalg.inv(kf.T.cpu().numpy())

        # 2. 根据模式决定是否需要拼接全局位姿
        if cameras_already_global:
            # -------------------------------------------------------
            # 模式 A：cam.T 已经是全局 W2C（slam.py 终局评估阶段）
            # 直接使用 inv(cam.T) 即可，不做任何额外变换
            # -------------------------------------------------------
            pose_est = local_c2w


        elif frame_to_submap is not None:

            sid = frame_to_submap.get(kf_id, 0)

            # 优先使用从 ckpt 重建的 anchor

            anchor_source = rebuilt_anchor_poses if len(rebuilt_anchor_poses) > 0 else submap_anchor_poses

            if anchor_source is not None and sid in anchor_source:

                anchor_c2w = anchor_source[sid]

                if isinstance(anchor_c2w, torch.Tensor):
                    anchor_c2w = anchor_c2w.cpu().numpy()

                anchor_c2w = np.array(anchor_c2w, dtype=np.float64)

                ct = correct_tsfms.get(sid, np.eye(4))

                global_c2w = ct @ anchor_c2w @ local_c2w

                pose_est = global_c2w

            else:

                pose_est = local_c2w
        else:
            # -------------------------------------------------------
            # 模式 C：无子图信息（单子图 / 未启用子图策略）
            # -------------------------------------------------------
            pose_est = local_c2w

        pose_gt = np.linalg.inv(kf.T_gt.cpu().numpy())

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
            os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


'''
计算psnr、ssim、lpips用的是非关键帧，即NVS
关键帧参与建图优化，非关键帧测试泛化

计算depth l1是关键帧还是非关键帧还是全部帧？
全部帧，全局采样测试建图精度

计算precision、recall、f-score是用关键帧还是非关键帧还是全部帧？
因为使用了渲染rgb和渲染depth进行了TSDF生成recon mesh,这里渲染rgb和渲染depth是关键帧的还是非关键帧还是全部帧？
全部帧按固定间隔（Interval）采样的结果，它完全没有区分关键帧还是非关键帧。
'''

def eval_rendering(
        frames,
        gaussians,
        dataset,
        save_dir,
        pipe,
        background,
        kf_indices,
        iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []

    # 防止传入字符串时报错
    end_idx = len(frames) - 1 if isinstance(iteration, str) else iteration
    is_final_eval = isinstance(iteration, str)

    # 【解耦数据池】：分清 NVS(非关键帧) 和 All(全部采样帧)
    nvs_psnr_array, nvs_ssim_array, nvs_lpips_array = [], [], []
    all_depth_l1_array = []  # <== 专门用来装全部采样帧的深度误差

    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")

    # 创建必要的文件夹
    render_dir = os.path.join(save_dir, "rendering")
    mkdir_p(render_dir)

    if is_final_eval:
        mesh_render_dir = os.path.join(save_dir, "mesh_rendering")
        mkdir_p(mesh_render_dir)
        render_poses_dict = {}
        Log("Rendering and saving all sampled frames for TSDF Fusion...", tag="Eval")

    # =========================================================
    # 核心大循环：遍历所有采样帧（包含关键帧和非关键帧）
    # 在这里统一执行渲染，并一次性完成 Depth L1计算、TSDF数据保存 和 NVS评估
    # =========================================================
    for idx in range(0, end_idx, interval):
        frame = frames[idx]
        gt_image, gt_depth, _ = dataset[idx]

        # 统一执行一次渲染，绝不浪费算力
        render_pkg = render(frame, gaussians, pipe, background)
        rendering = render_pkg["render"]
        render_depth = render_pkg["depth"]

        # 提前把 clamped image 拿出来，供 TSDF 和 NVS 共用
        image = torch.clamp(rendering, 0.0, 1.0)

        # ---------------------------------------------------------
        # [逻辑分支 A]：全局 Depth L1 计算 (所有人都要算)
        # ---------------------------------------------------------
        if gt_depth is not None:
            if isinstance(gt_depth, np.ndarray):
                gt_d = torch.from_numpy(gt_depth).float().cuda().squeeze()
            else:
                gt_d = gt_depth.float().cuda().squeeze()

            rend_d = render_depth.squeeze()

            # 确保 GT 和 Pred 的分辨率维度完全一致
            if gt_d.shape != rend_d.shape:
                gt_d = gt_d.view(rend_d.shape)

            valid_depth_mask = gt_d > 0.0

            if valid_depth_mask.sum() > 0:
                depth_l1 = torch.abs(rend_d[valid_depth_mask] - gt_d[valid_depth_mask]).mean().item()
                all_depth_l1_array.append(depth_l1)

        # ---------------------------------------------------------
        # [逻辑分支 B]：保存用于 TSDF Mesh 生成的数据 (所有人都要存)
        # ---------------------------------------------------------
        if is_final_eval:
            # 转换 RGB
            pred_rgb = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{mesh_render_dir}/color_{idx:05d}.png", pred_bgr)

            # 转换深度为 uint16 (mm)
            depth_mm = (render_depth.squeeze().detach().cpu().numpy() * 1000.0).astype(np.uint16)
            cv2.imwrite(f"{mesh_render_dir}/depth_{idx:05d}.png", depth_mm)

            # 保存位姿
            render_poses_dict[str(idx)] = frame.T.cpu().numpy().tolist()

        # ---------------------------------------------------------
        # [逻辑分支 C]：NVS 专属 -> 如果是关键帧，直接跳过指标计算！
        # ---------------------------------------------------------
        if idx in kf_indices:
            continue

        # 以下代码仅针对【非关键帧】（NVS 测试）执行
        saved_frame_idx.append(idx)

        # 获取 GT 并计算图像指标
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred_bgr_nvs = cv2.cvtColor((image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR)
        gt_bgr = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

        cv2.imwrite(f"{render_dir}/pred_{idx:05d}.png", pred_bgr_nvs)

        img_pred.append(pred_bgr_nvs)
        img_gt.append(gt_bgr)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        nvs_psnr_array.append(psnr_score.item())
        nvs_ssim_array.append(ssim_score.item())
        nvs_lpips_array.append(lpips_score.item())

    # =========================================================
    # 循环结束：汇总与保存 JSON 结果
    # =========================================================
    output = dict()
    output["mean_psnr"] = float(np.mean(nvs_psnr_array)) if len(nvs_psnr_array) > 0 else 0.0
    output["mean_ssim"] = float(np.mean(nvs_ssim_array)) if len(nvs_ssim_array) > 0 else 0.0
    output["mean_lpips"] = float(np.mean(nvs_lpips_array)) if len(nvs_lpips_array) > 0 else 0.0
    output["mean_depth_l1"] = float(np.mean(all_depth_l1_array)) if len(all_depth_l1_array) > 0 else 0.0

    Log(
        f'NVS psnr: {output["mean_psnr"]:.4f}, ssim: {output["mean_ssim"]:.4f}, lpips: {output["mean_lpips"]:.4f} | ALL depth_l1: {output["mean_depth_l1"]:.4f}m',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )

    if is_final_eval:
        with open(os.path.join(mesh_render_dir, "render_poses.json"), "w") as f:
            json.dump(render_poses_dict, f, indent=4)

    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    gaussians.save_pointcloud_ply(os.path.join(point_cloud_path, "point_cloud_points.ply"))
