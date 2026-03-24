import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

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
# ========= 新增：导入新建的 Loop Closure 进程类 =========
from utils.loop_closure import LoopClosureProcess
# ========================================================

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
            q_vis2main=q_vis2main,
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
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")

        # ========= 新增：优雅关闭 Loop Closure 进程 =========
        loop_queue.put(["stop"])
        self.loop_closure_process.join()
        Log("Loop Closure stopped and joined the main thread")
        # ====================================================

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
    # 整个SLAM系统作为一个类实现在SLAM.py中，而且在__init__()的时候就运行了所有的线程：gui_process、frontend_process、backend_process
    slam = SLAM(config, save_dir=save_dir)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
