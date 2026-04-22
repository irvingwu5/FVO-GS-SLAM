import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
import random
from tqdm import tqdm
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
# ========= 新增：导入新建的 Loop Closure 进程类 =========
from utils.loop_closure import LoopClosureProcess
# ========================================================
# 引入刚体变换引擎
from utils.loop_closure import rigid_transform_2dgs
import glob
import threading
import subprocess
import time

def rebuild_submap_anchors_from_ckpts(save_dir):
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


class GPUMemoryMonitor:
    def __init__(self, physical_gpu_id=0):
        self.keep_measuring = True
        self.peak_memory = 0
        self.physical_gpu_id = physical_gpu_id
        # 🟢 设置为 daemon=True，主进程退出时它会自动陪葬，绝不死缠烂打
        self.thread = threading.Thread(target=self.measure_usage, daemon=True)

    def measure_usage(self):
        while self.keep_measuring:
            try:
                result = subprocess.check_output(
                    [
                        'nvidia-smi', f'--id={self.physical_gpu_id}',
                        '--query-gpu=memory.used',
                        '--format=csv,nounits,noheader'
                    ], encoding='utf-8')
                current_mem = int(result.strip())
                if current_mem > self.peak_memory:
                    self.peak_memory = current_mem

                # 新增：让守护进程每隔 2 秒播报一次当前物理显存
                #print(f"[Monitor] 物理显存监控: {current_mem} MB / 24576 MB")
            except Exception:
                pass
            time.sleep(2.0)  # 放慢采样率到 2 秒，避免终端被日志淹没

    def start(self):
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        # 🟢 取消 thread.join()。既然是 daemon 线程，我们拿完数据直接走人
        # 睡一小会儿（0.1s），给它最后一次记录的机会
        time.sleep(0.1)
        return self.peak_memory

class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True) #创建了两个事件（Event）对象，主要用于GPU上的时间测量
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        # 解析配置文件中的模型参数、优化参数和管道参数
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        # 赋值给类的成员变量
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )
        # 根据配置文件设置各种模式和选项
        # 通过检查配置文件中 Dataset 部分的 type 字段是否为 "realsense"。
        # 如果是 "realsense"，意味着使用 Intel RealSense 相机进行实时采集和建图；否则可能是离线读取数据集。
        self.live_mode = self.config["Dataset"]["type"] == "realsense" # False or True
        # 通过检查配置文件中 Dataset 部分的 sensor_type 字段是否为 "monocular"
        # 如果是 "monocular"，表示输入源是单目相机；否则可能是 RGB-D 或双目相机等其他传感器类型。
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular" # False or True
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"] # False or True
        self.use_gui = self.config["Results"]["use_gui"] # False or True
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"] # False or True
        # 【新增：读取消融实验开关，兼容旧版配置】
        # 优先读取 Ablation 中的开关，如果没有则读取原来 LoopClosure 里的 enable
        self.use_loop_closure = self.config.get("Ablation", {}).get("use_loop_closure", True)
        #model_params.sh_degree = 3 if self.use_spherical_harmonics else 0 # true设置为3，false设置为0
        # 将原来的全局优化拆分为两个独立开关
        self.use_global_ba = self.config.get("Ablation", {}).get("use_global_ba", True)
        self.use_color_refinement = self.config.get("Ablation", {}).get("use_color_refinement", True)
        # [修改] 尊重 yaml 配置。只有当开关关闭时，才强制设为 0。
        # 如果开关开启，则保留 model_params 中读取到的值 (比如 1 或 2)
        if not self.use_spherical_harmonics:
            model_params.sh_degree = 0
        # 初始化高斯模型和数据集
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config) # 执行__init_函数初始化高斯各属性
        self.gaussians.init_lr(6.0)
        # 加载数据集
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        ) #返回了数据集中每个文件的绝对路径还没有具体读取
        # 从config文件中读取的超参数设置训练超参数配置
        self.gaussians.training_setup(opt_params)
        # 设置背景颜色为黑色
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # 创建前端和后端进程之间的队列
        frontend_queue = mp.Queue() #后端传给前端的数据队列
        backend_queue = mp.Queue() #前端传给后端的数据队列

        # FakeQueue 是一个“假的”队列，它存在的目的是为了让程序在关闭 GUI 模式下，依然能够流畅运行原本为 GUI 通信设计的代码逻辑，而不会因为没有真实的队列对象而崩溃。
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue() #主进程（frontend）传给可视化进程的数据队列
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue() #可视化进程传给主进程（frontend）的数据队列

        # ========= 修改：条件创建 Loop Closure 专属通信队列 =========
        loop_queue = mp.Queue() if self.use_loop_closure else None
        # ==========================================================
        # 重新赋值保存目录和单目模式
        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular
        # 初始化前端和后端模块
        self.frontend = FrontEnd(self.config) # 执行__init_函数初始化前端各属性
        self.backend = BackEnd(self.config) # 执行__init_函数初始化后端各属性
        # 给前端一系列参数赋值
        self.frontend.dataset = self.dataset #让 FrontEnd 可以直接访问数据集，实际是引用赋值（不是拷贝）
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue #后端传给前端的数据队列
        self.frontend.backend_queue = backend_queue  #前端传给后端的数据队列
        self.frontend.q_main2vis = q_main2vis #主进程（frontend）传给可视化进程的数据队列
        self.frontend.q_vis2main = q_vis2main #可视化进程传给主进程（frontend）的数据队列
        self.frontend.set_hyperparams() # 设置前端的超参数
        # 给后端一系列参数赋值
        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0 #场景中相机的空间范围
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue #后端传给前端的数据队列
        self.backend.backend_queue = backend_queue #前端传给后端的数据队列
        self.backend.live_mode = self.live_mode # 实时模式标志
        self.backend.set_hyperparams() # 设置后端的超参数

        # ========= 新增：将 loop_queue 挂载给后端 =========
        self.backend.loop_queue = loop_queue
        # ====================================================

        # 给GUI的参数赋值
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main
        )
        # 仅仅是创建了一个进程对象，并将 self.backend.run 注册为该进程启动时要运行的目标函数，不会执行 run 方法
        backend_process = mp.Process(target=self.backend.run)

        # ========= 修改：按条件实例化并启动 Loop Closure 后台进程 =========
        if self.use_loop_closure:
            self.loop_closure_process = LoopClosureProcess(self.config, loop_queue)
            self.loop_closure_process.start()
            Log("Loop Closure Process started.")
        else:
            self.loop_closure_process = None
            Log("[Ablation] Loop Closure Process is DISABLED.")
        # ============================================================

        if self.use_gui:
            # 创建一个 GUI 进程，目标函数为 slam_gui.run，传递参数 self.params_gui
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            # 启动 GUI 进程
            gui_process.start()
            # 等待5秒，主要是等 GUI 界面加载好
            time.sleep(5)
        # 启动 backend_process 进程
        backend_process.start() #它会请求操作系统启动一个新的进程，并在该进程中执行 self.backend.run 方法

        # 主进程运行frontend
        self.frontend.run()
        # 前端运行结束了，利用队列传递信息，让后端暂停
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()

        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", FPS, tag="Eval")

        # =========================================================================
        # 1. 停止后端，促使其保存最后一块子图
        # =========================================================================
        backend_queue.put(["stop"])

        Log("等待后端清理显存并安全退出 (防死锁抽干机制启动)...")
        # 【核心修复】：只要后端还活着，主进程就疯狂抽干 frontend_queue！
        # 防止后端发送的遗留高斯地图塞爆底层 IPC 管道，导致互相死锁。
        while backend_process.is_alive():
            while not frontend_queue.empty():
                try:
                    # 拿出来直接丢弃，让 PyTorch 释放底层的共享内存
                    _ = frontend_queue.get_nowait()
                except:
                    break
            time.sleep(0.1)  # 稍微喘口气，避免吃满单核 CPU

        backend_process.join()
        Log("Backend stopped and saved final submap.")

        # =========================================================================
        # 2. 停止回环检测进程，确保所有 PGO 写入硬盘完成
        # =========================================================================
        if self.use_loop_closure and self.loop_closure_process is not None:
            loop_queue.put(["stop"])
            self.loop_closure_process.join()
            Log("Loop Closure stopped and PGO finalized.")
        else:
            Log("Loop Closure is disabled, skipping PGO finalize.")

        # ========= 新增：离线阶段前，释放前端历史相机中的重负载张量 =========
        for cam in self.frontend.cameras.values():
            if hasattr(cam, "release_mapping_payload"):
                cam.release_mapping_payload()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # ================================================================

        # =========================================================================
        # 3. STREAMING MERGE: 流式子图合并与内存边缘化
        # =========================================================================
        Log("==> 开启流式合并模式，正在拉取硬盘子图数据... <==")

        submaps_dir = os.path.join(self.save_dir, "submaps")
        ckpt_files = sorted(glob.glob(os.path.join(submaps_dir, "*.ckpt")))

        # 定义累加器列表，仅存储在 CPU 内存
        final_params = {
            "_xyz": [], "_features_dc": [], "_features_rest": [],
            "_scaling": [], "_rotation": [], "_opacity": []
        }
        has_normal = False

        # 加载锚点位姿
        frame_to_submap = torch.load(os.path.join(save_dir, "frame_to_submap.pt"))
        submaps_dir = os.path.join(save_dir, "submaps")
        submap_anchor_poses = rebuild_submap_anchors_from_ckpts(submaps_dir)

        # 如果重建失败，再回退到前端保存版
        if len(submap_anchor_poses) == 0:
            submap_anchor_poses_path = os.path.join(save_dir, "submap_anchor_poses.pt")
            if os.path.exists(submap_anchor_poses_path):
                submap_anchor_poses = torch.load(submap_anchor_poses_path)
            else:
                submap_anchor_poses = None

        submap_tsfms = {}

        # 🟢 第一步：仅读取 PGO 修正矩阵（体积极小）
        for ckpt_path in ckpt_files:
            sid = int(os.path.basename(ckpt_path).split('.')[0])
            # map_location="cpu" 确保加载时不占显存，仅加载 header 获取修正矩阵
            ckpt = torch.load(ckpt_path, map_location="cpu")
            correct_tsfm = ckpt.get("correct_tsfm", np.eye(4))

            # 消融拦截
            if not getattr(self, 'use_loop_closure', True):
                correct_tsfm = np.eye(4)

            submap_tsfms[sid] = torch.from_numpy(correct_tsfm).float()  # 保持在 CPU
            del ckpt  # 立即释放

        import gc  # 【新增】：在循环前导入，而不是每次都导入

        # 🟢 第二步：流式读取、变换、存入 CPU 累加器
        for idx, ckpt_path in enumerate(tqdm(ckpt_files, desc="Streaming Submap Concatenation")):
            sid = int(os.path.basename(ckpt_path).split('.')[0])

            # 1. 读入一个子图到内存 (CPU)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            gp = ckpt["gaussian_params"]

            # 2. 获取对应的完整变换矩阵 = correct_tsfm @ anchor_c2w
            ct = submap_tsfms[sid].numpy()
            anchor = np.eye(4)
            if submap_anchor_poses is not None and sid in submap_anchor_poses:
                anchor = submap_anchor_poses[sid]
                if isinstance(anchor, torch.Tensor):
                    anchor = anchor.cpu().numpy()
                anchor = np.array(anchor, dtype=np.float64)

            full_tsfm = ct @ anchor  # 完整的全局变换

            # 3. 执行空间变换 (增加安全检查)
            if not np.allclose(full_tsfm, np.eye(4), atol=1e-4):
                gp_cuda = {
                    k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                    for k, v in gp.items()
                }

                # 执行刚体变换 (传入完整的全局变换矩阵)
                gp_corrected = rigid_transform_2dgs(gp_cuda, full_tsfm)

                gp = {
                    k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in gp_corrected.items()
                }
                del gp_cuda, gp_corrected

            # 4. 存入流式累加器（仅存内容，确保 detach）
            for key in final_params.keys():
                if key in gp and isinstance(gp[key], torch.Tensor):
                    final_params[key].append(gp[key].detach().cpu())

            if "_normal" in gp and isinstance(gp["_normal"], torch.Tensor):
                has_normal = True
                if "_normal" not in final_params: final_params["_normal"] = []
                final_params["_normal"].append(gp["_normal"].detach().cpu())

            # 5. 🎯 核心：立即销毁临时对象
            del gp, ckpt

            # 【优化】：每处理 5 个子图清理一次
            if (idx + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # 🟢 第三步：流式拼接并推向 GPU（改进版）
        Log("==> 所有子图已离线变换完毕，正在构建全局高斯模型... <==")
        if len(final_params["_xyz"]) > 0:
            import torch.nn as nn

            # 【优化】：在 CPU 上先拼接，然后分批推到 GPU
            Log("Concatenating submap parameters on CPU...")
            # 在拼接前进行形状检查和修复
            features_dc_list = []
            for feat in final_params["_features_dc"]:
                if feat.dim() == 3 and feat.shape[1] == 1:
                    # 形状是 [N, 1, 3]，需要 squeeze 到 [N, 3]
                    features_dc_list.append(feat.squeeze(1))
                else:
                    # 形状已经是 [N, 3]，直接使用
                    features_dc_list.append(feat)
            # 在 CPU 上完成拼接
            cpu_xyz = torch.cat(final_params["_xyz"], dim=0)
            cpu_features_dc = torch.cat(final_params["_features_dc"], dim=0)
            cpu_features_rest = torch.cat(final_params["_features_rest"], dim=0)
            cpu_scaling = torch.cat(final_params["_scaling"], dim=0)
            cpu_rotation = torch.cat(final_params["_rotation"], dim=0)
            cpu_opacity = torch.cat(final_params["_opacity"], dim=0)

            # 【新增】：清空 final_params，立即释放 CPU 内存
            final_params.clear()
            import gc
            gc.collect()

            # 【优化】：分批推到 GPU（如果点数过多）
            total_points = cpu_xyz.shape[0]
            batch_size = 1000000  # 每批 100 万个点

            if total_points > batch_size:
                Log(f"Large point cloud detected ({total_points} points), using batch transfer...")

                # 初始化参数（空张量）
                self.gaussians._xyz = nn.Parameter(torch.zeros((total_points, 3), device="cuda"))
                self.gaussians._features_dc = nn.Parameter(torch.zeros((total_points, 3), device="cuda"))
                self.gaussians._features_rest = nn.Parameter(torch.zeros((total_points, 15), device="cuda"))
                self.gaussians._scaling = nn.Parameter(torch.zeros((total_points, 3), device="cuda"))
                self.gaussians._rotation = nn.Parameter(torch.zeros((total_points, 4), device="cuda"))
                self.gaussians._opacity = nn.Parameter(torch.zeros((total_points, 1), device="cuda"))

                # 分批传输
                for i in range(0, total_points, batch_size):
                    end_idx = min(i + batch_size, total_points)
                    Log(f"Transferring batch {i // batch_size + 1}/{(total_points + batch_size - 1) // batch_size}...")

                    self.gaussians._xyz.data[i:end_idx] = cpu_xyz[i:end_idx].cuda()
                    self.gaussians._features_dc.data[i:end_idx] = cpu_features_dc[i:end_idx].cuda()
                    self.gaussians._features_rest.data[i:end_idx] = cpu_features_rest[i:end_idx].cuda()
                    self.gaussians._scaling.data[i:end_idx] = cpu_scaling[i:end_idx].cuda()
                    self.gaussians._rotation.data[i:end_idx] = cpu_rotation[i:end_idx].cuda()
                    self.gaussians._opacity.data[i:end_idx] = cpu_opacity[i:end_idx].cuda()

                    torch.cuda.empty_cache()
            else:
                # 点数较少，直接一次性推到 GPU
                self.gaussians._xyz = nn.Parameter(cpu_xyz.cuda())
                self.gaussians._features_dc = nn.Parameter(cpu_features_dc.cuda())
                self.gaussians._features_rest = nn.Parameter(cpu_features_rest.cuda())
                self.gaussians._scaling = nn.Parameter(cpu_scaling.cuda())
                self.gaussians._rotation = nn.Parameter(cpu_rotation.cuda())
                self.gaussians._opacity = nn.Parameter(cpu_opacity.cuda())

            # 【新增】：处理法线
            if has_normal:
                cpu_normal = torch.cat(final_params["_normal"], dim=0)
                self.gaussians._normal = nn.Parameter(cpu_normal.cuda())

            # 【新增】：释放 CPU 内存
            del cpu_xyz, cpu_features_dc, cpu_features_rest, cpu_scaling, cpu_rotation, cpu_opacity
            if has_normal:
                del cpu_normal
            gc.collect()

            # 初始化辅助张量
            total_points = self.gaussians._xyz.shape[0]
            self.gaussians.max_radii2D = torch.zeros((total_points,), device="cuda")
            self.gaussians.xyz_gradient_accum = torch.zeros((total_points, 1), device="cuda")
            self.gaussians.denom = torch.zeros((total_points, 1), device="cuda")

            # =======================================================
            # 对前端相机的 4x4 矩阵进行逆向修正 (拉扯轨迹)
            # =======================================================
            Log("==> 开始拉扯前端相机轨迹... <==")

            for frame_id, cam in tqdm(self.frontend.cameras.items(), desc="Correcting Trajectory"):
                sid = frame_to_submap.get(frame_id, 0)

                # 1. 获取该子图的开环锚点位姿
                anchor_c2w = np.eye(4)
                if submap_anchor_poses is not None and sid in submap_anchor_poses:
                    anchor_c2w = submap_anchor_poses[sid]
                    if isinstance(anchor_c2w, torch.Tensor):
                        anchor_c2w = anchor_c2w.cpu().numpy()
                anchor_c2w = torch.from_numpy(np.array(anchor_c2w, dtype=np.float32)).to(cam.T.device)

                # 2. 获取 PGO 修正矩阵
                correct_tsfm = submap_tsfms.get(sid, torch.eye(4)).to(cam.T.device).float()

                # 3. 计算全局 C2W
                local_c2w = torch.linalg.inv(cam.T)
                global_c2w = correct_tsfm @ anchor_c2w @ local_c2w

                # 4. 更新为全局 W2C
                with torch.no_grad():
                    cam.T = torch.linalg.inv(global_c2w)

            Log(f"==> 拼接完成！全局高斯点总数: {self.gaussians._xyz.shape[0]} <==")

        # =========================================================================
        # 4. 评估与全局画质精修 (GLOBAL COLOR REFINEMENT)
        # =========================================================================
        if self.eval_rendering:
            kf_indices = self.frontend.kf_indices
            columns = ["tag", "psnr", "ssim", "lpips", "Depth L1", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)

            # -------------------------------------------------------------
            # 阶段 A：评估当前状态 (PGO 修正后，但未做离线精修)
            # -------------------------------------------------------------
            Log("Evaluating Tracking ATE (With PGO Correction if enabled)...")
            current_ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
                frame_to_submap=frame_to_submap,
                submap_anchor_poses=submap_anchor_poses,
                cameras_already_global=True  # ← 新增这一个参数
            )

            Log("Rendering Current Map Quality...")
            rendering_result_current = eval_rendering(
                self.frontend.cameras, self.gaussians, self.dataset, self.save_dir,
                self.pipeline_params, self.background, kf_indices=kf_indices,
                iteration="global_merged_before_opt",
            )

            metrics_table.add_data(
                "Before_Offline_Opt",
                rendering_result_current["mean_psnr"],
                rendering_result_current["mean_ssim"],
                rendering_result_current["mean_lpips"],
                rendering_result_current.get("mean_depth_l1", 0.0),
                current_ATE, FPS,
            )
            # ========= 新增：离线 color refinement 前，释放前端 camera 上的重负载张量 =========
            for cam in self.frontend.cameras.values():
                if hasattr(cam, "release_mapping_payload"):
                    cam.release_mapping_payload()
            Log("==> 已释放前端相机的映射相关张量，准备进入离线优化阶段... <==")
            gc.collect()
            torch.cuda.empty_cache()
            # =============================================================================
            # -------------------------------------------------------------
            # 阶段 B：离线联合优化 (严格遵守 MonoGS 原版 Color Refinement 逻辑)
            # -------------------------------------------------------------
            if self.use_color_refinement:
                Log(f"==> 开始离线优化 | Color Refinement: {self.use_color_refinement} <==")

                valid_cameras = list(self.frontend.cameras.values())

                if len(valid_cameras) > 0:
                    # 1. 重新初始化 Gaussian 的优化器 (使用标准参数，不再做任何学习率魔改)
                    self.gaussians.training_setup(self.opt_params)

                    # 2. 清空相关的状态张量
                    total_points = self.gaussians._xyz.shape[0]
                    self.gaussians.max_radii2D = torch.zeros((total_points,), device="cuda")

                    # 3. 原版 MonoGS 设定
                    iteration_total = 26000

                    # (保留你原本的图片缓存机制，防止读取全量高清图导致 CPU/GPU OOM)
                    cpu_image_cache = {}
                    MAX_CACHE_SIZE = 800

                    pbar = tqdm(total=iteration_total, desc="Offline Color Refinement")

                    for iteration in range(1, iteration_total + 1):
                        # 随机抽取一个有效的相机视角
                        viewpoint_cam = random.choice(valid_cameras)

                        # 读取 GT 图像 (带缓存逻辑)
                        if viewpoint_cam.uid in cpu_image_cache:
                            gt_image_raw = cpu_image_cache[viewpoint_cam.uid]
                        else:
                            gt_image_raw, _, _ = self.dataset[viewpoint_cam.uid]
                            if len(cpu_image_cache) >= MAX_CACHE_SIZE:
                                del_key = random.choice(list(cpu_image_cache.keys()))
                                del cpu_image_cache[del_key]
                            cpu_image_cache[viewpoint_cam.uid] = gt_image_raw

                        gt_image = gt_image_raw.cuda(non_blocking=True)

                        # 渲染当前视角
                        render_pkg = render(
                            viewpoint_cam, self.gaussians, self.pipeline_params, self.background, surf=False
                        )
                        image = render_pkg["render"]
                        visibility_filter = render_pkg["visibility_filter"]
                        radii = render_pkg["radii"]

                        # 计算损失 (严格对齐 MonoGS 原版 Loss)
                        Ll1 = l1_loss(image, gt_image)
                        loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (
                                1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

                        loss.backward()

                        with torch.no_grad():
                            # 记录最大半径
                            self.gaussians.max_radii2D[visibility_filter] = torch.max(
                                self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                            )
                            # 原汁原味的步进与清零
                            self.gaussians.optimizer.step()
                            self.gaussians.optimizer.zero_grad(set_to_none=True)

                            # 原汁原味的学习率更新 (会同步更新 xyz 等属性的学习率衰减)
                            self.gaussians.update_learning_rate(iteration)

                        # 【新增】：显式释放 render_pkg（防止显存泄漏）
                        del render_pkg, image, visibility_filter, radii

                        # 【新增】：释放 GT 图像
                        del gt_image

                        # 【新增】：每 100 次迭代强制清理显存
                        if iteration % 100 == 0:
                            torch.cuda.empty_cache()
                        pbar.update(1)

                    # 循环结束，清理缓存
                    del cpu_image_cache
                    gc.collect()

                    pbar.close()
                    Log("==> Map refinement done <==")

                    # -------------------------------------------------------------
                    # 阶段 C：评估精修后的最终状态
                    # -------------------------------------------------------------
                    Log("Rendering FINAL Map Quality (After Color Refinement)...")
                    rendering_result_after = eval_rendering(
                        self.frontend.cameras, self.gaussians, self.dataset, self.save_dir,
                        self.pipeline_params, self.background, kf_indices=kf_indices,
                        iteration="global_merged_after_opt",
                    )

                    # 因为没有 BA，相机位姿未变，直接复用 current_ATE
                    final_ATE = current_ATE

                    # 将精修后的指标加入 wandb 表格
                    metrics_table.add_data(
                        "After_Offline_Opt",
                        rendering_result_after["mean_psnr"],
                        rendering_result_after["mean_ssim"],
                        rendering_result_after["mean_lpips"],
                        rendering_result_after.get("mean_depth_l1", 0.0),
                        final_ATE, FPS,
                    )

                else:
                    Log("[Warning] 没有找到有效的相机帧，跳过离线优化。")
            else:
                Log("==> [Ablation] 离线优化模块 (Color Refinement) 已关闭！ <==")
                Log("==> 结果将以拼接后的初始状态直接保存。 <==")
                save_gaussians(self.gaussians, self.save_dir, "final_merged_no_opt", final=True)

                # 最终统一写入 wandb (这里会包含 Before 和 After 两行数据做对比)
            wandb.log({"Metrics": metrics_table})

            # 保存巅峰之作
            save_gaussians(self.gaussians, self.save_dir, "final_merged_after_opt", final=True)

        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")


    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser，解析命令行参数
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")
    # 加载配置文件
    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None
    # 如果命令行参数中包含--eval，则进入评估模式
    if args.eval:
        Log("Running SA-GS-SLAM in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=False")
        config["Results"]["use_wandb"] = False # True->False
    # 如果配置文件中指定了保存结果，则创建保存目录并初始化 wandb
    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")
    # =========================================================================
    # 【核心新增】：启动全局物理显存监控
    # 注意：如果你用 CUDA_VISIBLE_DEVICES=1 启动，这里的 physical_gpu_id 就是 1
    # 可以通过 os.environ 自动获取当前映射的物理 GPU ID
    # =========================================================================
    gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # 如果有多个，取第一个
    physical_gpu_id = int(gpu_id_str.split(',')[0])

    mem_monitor = GPUMemoryMonitor(physical_gpu_id=physical_gpu_id)
    mem_monitor.start()
    Log(f"Started tracking physical GPU {physical_gpu_id} memory...")
    # 整个SLAM系统作为一个类实现在SLAM.py中，而且在__init__()的时候就运行了所有的线程：gui_process、frontend_process、backend_process
    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    # =========================================================================
    # 【修改】：学术级指标统计 (Peak GPU Memory & Map Size)
    # =========================================================================
    if save_dir is not None:
        # 1. 停止监控并获取物理真实的 Peak GPU Memory
        real_peak_memory_mb = mem_monitor.stop()

        # 为了论文的严谨性，你可以同时把两个指标都打出来：
        # - Allocated Peak: 算法理论极限（不含进程开销，供消融实验参考）
        # - System Peak: 真实物理峰值（写在论文主表格里的数据）
        algo_allocated_mb = torch.cuda.max_memory_allocated(device="cuda") / (1024 * 1024)

        Log(f"🎯 Algorithm Allocated Peak: {algo_allocated_mb:.2f} MB", tag="Eval")
        Log(f"🎯 System Physical Peak (Paper Metric): {real_peak_memory_mb:.2f} MB", tag="Eval")

        # 2. 统计地图大小 (Map Size)
        final_ply_path = os.path.join(
            str(save_dir),
            "point_cloud",
            "final",
            "point_cloud.ply"
        )

        if os.path.exists(final_ply_path):
            map_size_mb = os.path.getsize(final_ply_path) / (1024 * 1024)
            Log(f"Final Map Size: {map_size_mb:.2f} MB", tag="Eval")
        else:
            Log(f"[Warning] 找不到最终的 PLY 文件: {final_ply_path}，无法计算 Map Size。", tag="Eval")
    wandb.finish()

    # All done
    Log("Done.")
