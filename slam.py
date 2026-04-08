import os
import sys
import time
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
from utils.pose_utils import update_pose
import glob
import threading
import subprocess
import time

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
                # 调用系统底层的 nvidia-smi 获取最真实的物理显存占用
                result = subprocess.check_output(
                    [
                        'nvidia-smi', f'--id={self.physical_gpu_id}',
                        '--query-gpu=memory.used',
                        '--format=csv,nounits,noheader'
                    ], encoding='utf-8')
                current_mem = int(result.strip())
                if current_mem > self.peak_memory:
                    self.peak_memory = current_mem
            except Exception:
                pass
            time.sleep(0.2)  # 每 0.2 秒高频采样一次

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

        #model_params.sh_degree = 3 if self.use_spherical_harmonics else 0 # true设置为3，false设置为0
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

        # ========= 新增：为 Loop Closure 创建专属通信队列 =========
        loop_queue = mp.Queue()
        # ==========================================================
        # 重新赋值保存目录和单目模式
        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular
        # 初始化前端和后端模块
        self.frontend = FrontEnd(self.config) # 执行__init_函数初始化前端各属性
        self.backend = BackEnd(self.config) # 执行__init_函数初始化后端各属性
        # 给前端一系列参数赋值
        self.frontend.dataset = self.dataset
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

        # ========= 新增：实例化并启动 Loop Closure 后台进程 =========
        self.loop_closure_process = LoopClosureProcess(self.config, loop_queue)
        self.loop_closure_process.start()
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
        loop_queue.put(["stop"])
        self.loop_closure_process.join()
        Log("Loop Closure stopped and PGO finalized.")

        # =========================================================================
        # 3. THE GRAND MERGE: 全局子图合并与相机轨迹深度校正
        # =========================================================================
        Log("==> 开始合并所有子图并校正全局相机轨迹... <==")

        submaps_dir = os.path.join(self.save_dir, "submaps")
        ckpt_files = sorted(glob.glob(os.path.join(submaps_dir, "*.ckpt")))

        merged_params = {
            "_xyz": [], "_features_dc": [], "_features_rest": [],
            "_scaling": [], "_rotation": [], "_opacity": []
        }
        has_normal = False
        frame_to_submap = torch.load(os.path.join(self.save_dir, "frame_to_submap.pt"))
        submap_tsfms = {}

        # 遍历读取所有存入硬盘的子图
        for ckpt_path in ckpt_files:
            ckpt = torch.load(ckpt_path, map_location="cuda")
            gp = ckpt["gaussian_params"]

            # 提取 PGO 修正矩阵
            submap_id = int(os.path.basename(ckpt_path).split('.')[0])
            correct_tsfm = ckpt.get("correct_tsfm", np.eye(4))
            submap_tsfms[submap_id] = torch.from_numpy(correct_tsfm).float().cuda()

            # 【核心】：在合并前，仅在此处执行一次刚体变换到高斯点云上！
            if not np.allclose(correct_tsfm, np.eye(4), atol=1e-4):
                gp = rigid_transform_2dgs(gp, correct_tsfm)

            # ==============================================================
            # 【显存修复 1：用完即焚，踢回 CPU】
            # 防止所有的子图在 GPU 显存里“大团圆”导致绝杀 OOM。
            # 将它们缓存到 CPU 列表里，组装完了再送去 GPU。
            # ==============================================================
            merged_params["_xyz"].append(gp["_xyz"].detach().cpu())
            merged_params["_features_dc"].append(gp["_features_dc"].detach().cpu())
            merged_params["_features_rest"].append(gp["_features_rest"].detach().cpu())
            merged_params["_scaling"].append(gp["_scaling"].detach().cpu())
            merged_params["_rotation"].append(gp["_rotation"].detach().cpu())
            merged_params["_opacity"].append(gp["_opacity"].detach().cpu())
            if "_normal" in gp:
                has_normal = True
                if "_normal" not in merged_params: merged_params["_normal"] = []
                merged_params["_normal"].append(gp["_normal"].detach().cpu())

            # 手动销毁当前循环的字典，及时腾出显存
            del gp, ckpt

        if len(merged_params["_xyz"]) > 0:
            import torch.nn as nn
            # 彻底重建全局高斯模型，在 CPU 上拼接 (cat) 后，最后统一 .cuda() 送回显存
            self.gaussians._xyz = nn.Parameter(torch.cat(merged_params["_xyz"], dim=0).cuda())
            self.gaussians._features_dc = nn.Parameter(torch.cat(merged_params["_features_dc"], dim=0).cuda())
            self.gaussians._features_rest = nn.Parameter(torch.cat(merged_params["_features_rest"], dim=0).cuda())
            self.gaussians._scaling = nn.Parameter(torch.cat(merged_params["_scaling"], dim=0).cuda())
            self.gaussians._rotation = nn.Parameter(torch.cat(merged_params["_rotation"], dim=0).cuda())
            self.gaussians._opacity = nn.Parameter(torch.cat(merged_params["_opacity"], dim=0).cuda())
            if has_normal:
                self.gaussians._normal = nn.Parameter(torch.cat(merged_params["_normal"], dim=0).cuda())

            self.gaussians.active_sh_degree = self.gaussians.max_sh_degree

            # =======================================================
            # 【终极修复】：同步扩充所有的辅助张量，防止 IndexError
            # =======================================================
            total_points = self.gaussians._xyz.shape[0]
            self.gaussians.max_radii2D = torch.zeros((total_points,), device="cuda")
            self.gaussians.xyz_gradient_accum = torch.zeros((total_points, 1), device="cuda")
            self.gaussians.denom = torch.zeros((total_points, 1), device="cuda")
            self.gaussians.unique_kfIDs = torch.zeros((total_points,), device="cuda", dtype=torch.int32)
            self.gaussians.n_obs = torch.zeros((total_points,), device="cuda", dtype=torch.int32)

            # =======================================================
            # 对前端相机的 4x4 矩阵进行逆向修正
            # =======================================================
            Log("==> 开始拉扯前端相机轨迹... <==")
            for frame_id, cam in self.frontend.cameras.items():
                sid = frame_to_submap.get(frame_id, 0)
                tsfm_tensor = submap_tsfms.get(sid, torch.eye(4).cuda())

                if torch.allclose(tsfm_tensor, torch.eye(4).cuda(), atol=1e-4):
                    continue

                inv_tsfm = torch.linalg.inv(tsfm_tensor)

                # 既然 cam.T 已经是 4x4 W2C 矩阵，直接矩阵相乘！
                cam.T = cam.T @ inv_tsfm

            Log(f"==> 拼接完成！全局高斯点总数: {self.gaussians._xyz.shape[0]} <==")

        # =========================================================================
        # 4. 评估与全局画质精修 (GLOBAL COLOR REFINEMENT)
        # =========================================================================
        if self.eval_rendering:
            kf_indices = self.frontend.kf_indices
            Log("Evaluating Global ATE with PGO Correction...")

            # 此时的相机已经全被 PGO 掰正了！
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            # 4.1 评估精修前的全局大地图 (Before)
            Log("Rendering Before Global Refinement...")
            rendering_result_before = eval_rendering(
                self.frontend.cameras, self.gaussians, self.dataset, self.save_dir,
                self.pipeline_params, self.background, kf_indices=kf_indices,
                iteration="global_merged_before_opt",
            )
            # 【修改 1】：列名加上 Depth L1
            columns = ["tag", "psnr", "ssim", "lpips", "Depth L1", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result_before["mean_psnr"],
                rendering_result_before["mean_ssim"],
                rendering_result_before["mean_lpips"],
                rendering_result_before.get("mean_depth_l1", 0.0),  # <== 新增
                ATE, FPS,
            )
            # 4.2 真正对全局地图执行 Global Bundle Adjustment (画质精修 + 几何缝合 + 位姿微调)
            Log("==> 开始全局大地图联合优化 (Global BA & Color Refinement)... <==")

            valid_cameras = list(self.frontend.cameras.values())

            if len(valid_cameras) > 0:
                self.gaussians.training_setup(self.opt_params)

                total_points = self.gaussians._xyz.shape[0]
                self.gaussians.max_radii2D = torch.zeros((total_points,), device="cuda")
                self.gaussians.xyz_gradient_accum = torch.zeros((total_points, 1), device="cuda")
                self.gaussians.denom = torch.zeros((total_points, 1), device="cuda")
                self.gaussians.unique_kfIDs = torch.zeros((total_points,), device="cuda", dtype=torch.int32)
                self.gaussians.n_obs = torch.zeros((total_points,), device="cuda", dtype=torch.int32)

                # 1. 允许几何自我缝合
                self.gaussians._xyz.requires_grad = True
                self.gaussians._scaling.requires_grad = True
                self.gaussians._rotation.requires_grad = True
                if hasattr(self.gaussians, '_normal'):
                    self.gaussians._normal.requires_grad = True

                for param_group in self.gaussians.optimizer.param_groups:
                    if param_group["name"] in ["xyz", "rotation", "scaling", "normal"]:
                        param_group["lr"] = param_group["lr"] * 0.2

                # 2. 【极速优化】：为每个相机绑定独立的优化器，拒绝全局遍历！
                for cam in valid_cameras:
                    if getattr(cam, 'cam_rot_delta', None) is None:
                        cam.cam_rot_delta = torch.nn.Parameter(torch.zeros(3, requires_grad=True, device="cuda"))
                    if getattr(cam, 'cam_trans_delta', None) is None:
                        cam.cam_trans_delta = torch.nn.Parameter(torch.zeros(3, requires_grad=True, device="cuda"))
                    if getattr(cam, 'exposure_a', None) is None:
                        cam.exposure_a = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True, device="cuda"))
                    if getattr(cam, 'exposure_b', None) is None:
                        cam.exposure_b = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True, device="cuda"))

                    cam_opt_params = [
                        {"params": [cam.cam_rot_delta], "lr": self.config["Training"]["lr"]["cam_rot_delta"] * 0.2},
                        {"params": [cam.cam_trans_delta], "lr": self.config["Training"]["lr"]["cam_trans_delta"] * 0.2},
                        {"params": [cam.exposure_a], "lr": 0.01},
                        {"params": [cam.exposure_b], "lr": 0.01}
                    ]
                    # 给每一个相机挂载自己的私有优化器
                    cam.optimizer = torch.optim.Adam(cam_opt_params)

                # =========================================================
                # 3. 极简智能动态缓存池 (最大缓存 800 张，通吃一切情况)
                # =========================================================
                iteration_total = 26000
                cpu_image_cache = {}
                MAX_CACHE_SIZE = 800

                pbar = tqdm(total=iteration_total, desc="Global BA (Ultra Speed)")

                for iteration in range(1, iteration_total + 1):
                    viewpoint_cam = random.choice(valid_cameras)

                    # 【智能命中机制】：如果在内存里，瞬间提取；如果不在，读硬盘并加入缓存。
                    if viewpoint_cam.uid in cpu_image_cache:
                        gt_image_raw = cpu_image_cache[viewpoint_cam.uid]
                    else:
                        gt_image_raw, _, _, _, _ = self.dataset[viewpoint_cam.uid]
                        # 超过安全上限就随机踢掉一个旧的，保护 RAM 永不爆炸
                        if len(cpu_image_cache) >= MAX_CACHE_SIZE:
                            del_key = random.choice(list(cpu_image_cache.keys()))
                            del cpu_image_cache[del_key]
                        cpu_image_cache[viewpoint_cam.uid] = gt_image_raw

                    gt_image = gt_image_raw.cuda(non_blocking=True)

                    render_pkg = render(
                        viewpoint_cam, self.gaussians, self.pipeline_params, self.background, surf=False
                    )
                    image = render_pkg["render"]
                    visibility_filter = render_pkg["visibility_filter"]
                    radii = render_pkg["radii"]

                    Ll1 = l1_loss(image, gt_image)
                    loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (
                            1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))

                    loss.backward()

                    with torch.no_grad():
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                        )
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                        self.gaussians.update_learning_rate(iteration)

                        # 【速度起飞核心】：只更新这 1 个相机的参数，省略了另外 499 个相机的无效开销！
                        viewpoint_cam.optimizer.step()
                        viewpoint_cam.optimizer.zero_grad(set_to_none=True)

                        update_pose(viewpoint_cam)

                        # 每 100 步统一衰减所有相机的学习率
                        if iteration % 100 == 0:
                            lr_factor = (0.01 ** (1.0 / (iteration_total / 100)))
                            for cam in valid_cameras:
                                for param_group in cam.optimizer.param_groups:
                                    param_group['lr'] *= lr_factor

                    del gt_image
                    pbar.update(1)

                # 用完即焚清理所有内存缓存
                del cpu_image_cache
                import gc
                gc.collect()

                pbar.close()
                Log("==> 全局大地图联合优化缝合完成！ <==")
            else:
                Log("[Warning] 没有找到有效的图像缓存，跳过全局画质精修。")

            # 4.3 评估精修后的超清大图 (After)
            Log("Evaluating FINAL ATE after Global Optimization...")

            # 【重要修复】：精修动了相机，必须在精修后重新计算真实的 ATE！
            final_ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result_after = eval_rendering(
                self.frontend.cameras, self.gaussians, self.dataset, self.save_dir,
                self.pipeline_params, self.background, kf_indices=kf_indices,
                iteration="global_merged_after_opt",
            )
            # 【修改 3】：塞入 After 阶段的数据
            metrics_table.add_data(
                "After",
                rendering_result_after["mean_psnr"],
                rendering_result_after["mean_ssim"],
                rendering_result_after["mean_lpips"],
                rendering_result_after.get("mean_depth_l1", 0.0),  # <== 新增
                final_ATE, FPS,
            )
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
