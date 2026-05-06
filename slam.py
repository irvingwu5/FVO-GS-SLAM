# ============================================================================
# Standard Library
# ============================================================================
import glob
import os
import random
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from datetime import datetime

# ============================================================================
# Third-party
# ============================================================================
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify
from tqdm import tqdm
import wandb

# ============================================================================
# Gaussian Splatting
# ============================================================================
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.system_utils import mkdir_p

# ============================================================================
# Local Modules
# ============================================================================
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.loop_closure import LoopClosureProcess, rigid_transform_2dgs
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


# ============================================================================
# Utility Functions
# ============================================================================

# ============================================================================
# GPU Memory Monitor
# ============================================================================

class GPUMemoryMonitor:
    def __init__(self, physical_gpu_id=0):
        self.keep_measuring = True
        self.peak_memory = 0
        self.baseline_memory = 0
        self.physical_gpu_id = physical_gpu_id
        self.thread = threading.Thread(target=self.measure_usage, daemon=True)

    def _query_gpu_mem(self):
        result = subprocess.check_output(
            [
                'nvidia-smi', f'--id={self.physical_gpu_id}',
                '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
        return int(result.strip())

    def measure_usage(self):
        while self.keep_measuring:
            try:
                current_mem = self._query_gpu_mem()
                if current_mem > self.peak_memory:
                    self.peak_memory = current_mem
            except Exception:
                pass
            time.sleep(2.0)

    def start(self):
        try:
            self.baseline_memory = self._query_gpu_mem()
        except Exception:
            self.baseline_memory = 0
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        time.sleep(0.1)
        return max(0, self.peak_memory - self.baseline_memory)


# ============================================================================
# SLAM System
# ============================================================================

class SLAM:
    def __init__(self, config, save_dir=None):
        # ---- Timing ----
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        self.config = config
        self.save_dir = save_dir

        # ---- Config Parsing ----
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        # ---- Mode Detection ----
        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        # ---- Feature Flags ----
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        # ---- Ablation Switches ----
        self.use_loop_closure = self.config.get("Ablation", {}).get("use_loop_closure", True)
        self.use_color_refinement = self.config.get("Ablation", {}).get("use_color_refinement", True)
        if not self.use_spherical_harmonics:
            model_params.sh_degree = 0

        # ---- Gaussian Model & Dataset ----
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        if config.get("max_frames", 0) > 0:
            self.dataset.num_imgs = min(config["max_frames"], self.dataset.num_imgs)
        self.gaussians.training_setup(opt_params)

        # ---- Background ----
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # ---- Queues ----
        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()
        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()
        loop_queue = mp.Queue() if self.use_loop_closure else None

        # ---- Frontend & Backend ----
        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode
        self.backend.loop_queue = loop_queue
        self.backend.set_hyperparams()

        # ---- GUI Params ----
        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main
        )

        # ---- Process Creation ----
        backend_process = mp.Process(target=self.backend.run)

        if self.use_loop_closure:
            self.loop_closure_process = LoopClosureProcess(self.config, loop_queue)
            self.loop_closure_process.start()
            Log("Loop Closure Process started.")
        else:
            self.loop_closure_process = None
            Log("[Ablation] Loop Closure Process is DISABLED.")

        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()

        # ---- Run Frontend (blocking) ----
        self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()

        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", FPS, tag="Eval")

        # =====================================================================
        # 1. Stop Backend (saves final submap)
        # =====================================================================
        backend_queue.put(["stop"])

        Log("等待后端清理显存并安全退出 (防死锁抽干机制启动)...")
        while backend_process.is_alive():
            while not frontend_queue.empty():
                try:
                    _ = frontend_queue.get_nowait()
                except:
                    break
            time.sleep(0.1)

        backend_process.join()
        Log("Backend stopped and saved final submap.")

        # =====================================================================
        # 2. Stop Loop Closure
        # =====================================================================
        if self.use_loop_closure and self.loop_closure_process is not None:
            loop_queue.put(["stop"])
            self.loop_closure_process.join()
            Log("Loop Closure stopped and PGO finalized.")
        else:
            Log("Loop Closure is disabled, skipping PGO finalize.")

        # =====================================================================
        # 3. Release Frontend Camera Tensors
        # =====================================================================
        for cam in self.frontend.cameras.values():
            if hasattr(cam, "release_mapping_payload"):
                cam.release_mapping_payload()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # =====================================================================
        # 4. Streaming Merge: Load & Transform Submaps
        # =====================================================================
        Log("==> 开启流式合并模式，正在拉取硬盘子图数据... <==")

        submaps_dir = os.path.join(self.save_dir, "submaps")
        ckpt_files = sorted(glob.glob(os.path.join(submaps_dir, "*.ckpt")))

        final_params = {
            "_xyz": [], "_features_dc": [], "_features_rest": [],
            "_scaling": [], "_rotation": [], "_opacity": []
        }
        has_normal = False

        frame_to_submap = torch.load(os.path.join(save_dir, "frame_to_submap.pt"))

        # Step A: Read PGO correction matrices (small).
        # NOTE: correct_tsfm is the LEGACY submap-level PGO correction field.
        # When keyframe-level PGO is used, corrections are applied separately
        # via apply_keyframe_pgo_to_trajectory. This field remains for backward
        # compatibility with old experiments. New experiments using
        # keyframe-level PGO will have correct_tsfm = identity.
        submap_tsfms = {}
        for ckpt_path in ckpt_files:
            sid = int(os.path.basename(ckpt_path).split('.')[0])
            ckpt = torch.load(ckpt_path, map_location="cpu")
            correct_tsfm = ckpt.get("correct_tsfm", np.eye(4))

            if not getattr(self, 'use_loop_closure', True):
                correct_tsfm = np.eye(4)

            submap_tsfms[sid] = torch.from_numpy(correct_tsfm).float()
            del ckpt

        # Load keyframe PGO result if available (Stage 7+)
        kf_pgo_path = os.path.join(save_dir, "keyframe_pgo_result.json")
        kf_pgo_corrections = {}
        if os.path.isfile(kf_pgo_path):
            import json
            with open(kf_pgo_path) as f:
                kf_pgo_data = json.load(f)
            if kf_pgo_data.get("accepted"):
                for kf_str, corr_list in kf_pgo_data.get("keyframe_corrections", {}).items():
                    kf_pgo_corrections[int(kf_str)] = torch.from_numpy(
                        np.array(corr_list, dtype=np.float64)
                    ).float()
                Log(f"Loaded {len(kf_pgo_corrections)} keyframe PGO corrections from {kf_pgo_path}")

        # Build per-submap median correction from keyframe PGO result
        submap_kf_corrections = {}
        if kf_pgo_corrections:
            for kf_id, delta in kf_pgo_corrections.items():
                sid = frame_to_submap.get(kf_id, 0)
                if sid not in submap_kf_corrections:
                    submap_kf_corrections[sid] = []
                submap_kf_corrections[sid].append(delta.numpy())

        submap_kf_tsfms = {}
        for sid, deltas in submap_kf_corrections.items():
            if len(deltas) == 0:
                continue
            # Median translation, chordal-mean rotation
            t_all = np.stack([d[:3, 3] for d in deltas])
            R_sum = np.sum([d[:3, :3] for d in deltas], axis=0)
            U, _, Vt = np.linalg.svd(R_sum)
            R_mean = U @ Vt
            if np.linalg.det(R_mean) < 0:
                R_mean = -R_mean
            T = np.eye(4)
            T[:3, :3] = R_mean
            T[:3, 3] = np.median(t_all, axis=0)
            submap_kf_tsfms[sid] = torch.from_numpy(T).float()

        import gc

        # Step B: Stream, transform, accumulate
        for idx, ckpt_path in enumerate(tqdm(ckpt_files, desc="Streaming Submap Concatenation")):
            sid = int(os.path.basename(ckpt_path).split('.')[0])

            ckpt = torch.load(ckpt_path, map_location="cpu")
            gp = ckpt["gaussian_params"]

            ct = submap_tsfms[sid].numpy()
            # Gaussians 已在全局坐标系（global seed mode），叠加 legacy correct_tsfm
            # 如果有 keyframe PGO 结果，再叠加 submap 级中位数 correction
            full_tsfm = ct
            if sid in submap_kf_tsfms:
                kf_tsfm = submap_kf_tsfms[sid].numpy()
                full_tsfm = kf_tsfm @ ct

            if not np.allclose(full_tsfm, np.eye(4), atol=1e-4):
                gp_cuda = {
                    k: (v.cuda() if isinstance(v, torch.Tensor) else v)
                    for k, v in gp.items()
                }
                gp_corrected = rigid_transform_2dgs(gp_cuda, full_tsfm)
                gp = {
                    k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in gp_corrected.items()
                }
                del gp_cuda, gp_corrected

            n_pts = gp["_xyz"].shape[0] if "_xyz" in gp else 0
            Log(f"[Fusion] submap {sid}: {n_pts} Gaussians")

            for key in final_params.keys():
                if key in gp and isinstance(gp[key], torch.Tensor):
                    final_params[key].append(gp[key].detach().cpu())

            if "_normal" in gp and isinstance(gp["_normal"], torch.Tensor):
                has_normal = True
                if "_normal" not in final_params: final_params["_normal"] = []
                final_params["_normal"].append(gp["_normal"].detach().cpu())

            del gp, ckpt

            if (idx + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Step C: Concatenate and push to GPU
        Log("==> 所有子图已离线变换完毕，正在构建全局高斯模型... <==")
        if len(final_params["_xyz"]) > 0:
            import torch.nn as nn

            Log("Concatenating submap parameters on CPU...")
            features_dc_list = []
            for feat in final_params["_features_dc"]:
                if feat.dim() == 3 and feat.shape[1] == 1:
                    features_dc_list.append(feat.squeeze(1))
                else:
                    features_dc_list.append(feat)

            cpu_xyz = torch.cat(final_params["_xyz"], dim=0)
            cpu_features_dc = torch.cat(final_params["_features_dc"], dim=0)
            cpu_features_rest = torch.cat(final_params["_features_rest"], dim=0)
            cpu_scaling = torch.cat(final_params["_scaling"], dim=0)
            cpu_rotation = torch.cat(final_params["_rotation"], dim=0)
            cpu_opacity = torch.cat(final_params["_opacity"], dim=0)
            cpu_normal = (torch.cat(final_params["_normal"], dim=0)
                          if has_normal and "_normal" in final_params else None)

            final_params.clear()
            import gc
            gc.collect()

            total_points = cpu_xyz.shape[0]
            batch_size = 1000000

            if total_points > batch_size:
                Log(f"Large point cloud detected ({total_points} points), using batch transfer...")

                self.gaussians._xyz = nn.Parameter(torch.zeros((total_points, 3), device="cuda"))
                self.gaussians._features_dc = nn.Parameter(torch.zeros((total_points, 3), device="cuda"))
                self.gaussians._features_rest = nn.Parameter(torch.zeros((total_points, 15), device="cuda"))
                self.gaussians._scaling = nn.Parameter(torch.zeros((total_points, 3), device="cuda"))
                self.gaussians._rotation = nn.Parameter(torch.zeros((total_points, 4), device="cuda"))
                self.gaussians._opacity = nn.Parameter(torch.zeros((total_points, 1), device="cuda"))

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
                self.gaussians._xyz = nn.Parameter(cpu_xyz.cuda())
                self.gaussians._features_dc = nn.Parameter(cpu_features_dc.cuda())
                self.gaussians._features_rest = nn.Parameter(cpu_features_rest.cuda())
                self.gaussians._scaling = nn.Parameter(cpu_scaling.cuda())
                self.gaussians._rotation = nn.Parameter(cpu_rotation.cuda())
                self.gaussians._opacity = nn.Parameter(cpu_opacity.cuda())

            if has_normal and cpu_normal is not None:
                self.gaussians._normal = cpu_normal.cuda()

            del cpu_xyz, cpu_features_dc, cpu_features_rest, cpu_scaling, cpu_rotation, cpu_opacity
            if has_normal and cpu_normal is not None:
                del cpu_normal
            gc.collect()

            total_points = self.gaussians._xyz.shape[0]
            self.gaussians.max_radii2D = torch.zeros((total_points,), device="cuda")
            self.gaussians.xyz_gradient_accum = torch.zeros((total_points, 1), device="cuda")
            self.gaussians.denom = torch.zeros((total_points, 1), device="cuda")

            # Step D: Correct frontend camera trajectory
            Log("==> 开始拉扯前端相机轨迹... <==")

            for frame_id, cam in tqdm(self.frontend.cameras.items(), desc="Correcting Trajectory"):
                sid = frame_to_submap.get(frame_id, 0)

                correct_tsfm = submap_tsfms.get(sid, torch.eye(4)).to(cam.T.device).float()

                # Apply keyframe PGO correction if available
                if frame_id in kf_pgo_corrections:
                    delta_kf = kf_pgo_corrections[frame_id].to(cam.T.device).float()
                elif kf_pgo_corrections:
                    # Non-keyframe: use nearest keyframe correction
                    nearest_kf = min(kf_pgo_corrections.keys(), key=lambda k: abs(k - frame_id))
                    delta_kf = kf_pgo_corrections[nearest_kf].to(cam.T.device).float()
                else:
                    delta_kf = torch.eye(4, device=cam.T.device)

                # cam.T 已经是全局 W2C（前端 global seed 模式下所有帧都在全局坐标系跟踪）
                # 叠加 legacy correct_tsfm + keyframe PGO correction
                global_c2w = delta_kf @ correct_tsfm @ torch.linalg.inv(cam.T)

                with torch.no_grad():
                    cam.T = torch.linalg.inv(global_c2w)

            Log(f"==> 拼接完成！全局高斯点总数: {self.gaussians._xyz.shape[0]} <==")

        # =====================================================================
        # 5. Evaluation & Color Refinement
        # =====================================================================
        if self.eval_rendering:
            kf_indices = self.frontend.kf_indices
            columns = ["tag", "psnr", "ssim", "lpips", "Depth L1", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)

            # Phase A: Evaluate ATE (after PGO, before offline opt)
            Log("Evaluating Tracking ATE (With PGO Correction if enabled)...")
            current_ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )
            # Skip pre-opt rendering to save time; only ATE is computed in Phase A

            for cam in self.frontend.cameras.values():
                if hasattr(cam, "release_mapping_payload"):
                    cam.release_mapping_payload()
            Log("==> 已释放前端相机的映射相关张量，准备进入离线优化阶段... <==")
            gc.collect()
            torch.cuda.empty_cache()

            # Phase B: Offline Color Refinement
            if self.use_color_refinement:
                Log(f"==> 开始离线优化 | Color Refinement: {self.use_color_refinement} <==")

                valid_cameras = list(self.frontend.cameras.values())

                if len(valid_cameras) > 0:
                    self.gaussians.training_setup(self.opt_params)

                    total_points = self.gaussians._xyz.shape[0]
                    self.gaussians.max_radii2D = torch.zeros((total_points,), device="cuda")

                    iteration_total = 26000

                    cpu_image_cache = {}
                    MAX_CACHE_SIZE = 800

                    pbar = tqdm(total=iteration_total, desc="Offline Color Refinement")

                    for iteration in range(1, iteration_total + 1):
                        viewpoint_cam = random.choice(valid_cameras)

                        if viewpoint_cam.uid in cpu_image_cache:
                            gt_image_raw = cpu_image_cache[viewpoint_cam.uid]
                        else:
                            gt_image_raw, _, _ = self.dataset[viewpoint_cam.uid]
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

                        del render_pkg, image, visibility_filter, radii
                        del gt_image

                        if iteration % 100 == 0:
                            torch.cuda.empty_cache()
                        pbar.update(1)

                    del cpu_image_cache
                    gc.collect()

                    pbar.close()
                    Log("==> Map refinement done <==")

                    # Phase C: Evaluate after refinement
                    Log("Rendering FINAL Map Quality (After Color Refinement)...")
                    rendering_result_after = eval_rendering(
                        self.frontend.cameras, self.gaussians, self.dataset, self.save_dir,
                        self.pipeline_params, self.background, kf_indices=kf_indices,
                        iteration="global_merged_after_opt",
                    )

                    final_ATE = current_ATE

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

            wandb.log({"Metrics": metrics_table})

            save_gaussians(self.gaussians, self.save_dir, "final_merged_after_opt", final=True)

        # =====================================================================
        # 6. GUI Cleanup
        # =====================================================================
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Limit dataset to N frames for smoke testing (0=unlimited)")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running FVO-GS-SLAM in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=False")
        config["Results"]["use_wandb"] = False

    if args.max_frames > 0:
        config["max_frames"] = args.max_frames
        Log(f"Limiting dataset to {args.max_frames} frames (smoke test)")

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

    # GPU Memory Monitor
    gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    physical_gpu_id = int(gpu_id_str.split(',')[0])

    mem_monitor = GPUMemoryMonitor(physical_gpu_id=physical_gpu_id)
    mem_monitor.start()
    Log(f"Started tracking physical GPU {physical_gpu_id} memory...")

    slam = SLAM(config, save_dir=save_dir)
    slam.run()

    # Metrics
    if save_dir is not None:
        real_peak_memory_mb = mem_monitor.stop()

        algo_allocated_mb = torch.cuda.max_memory_allocated(device="cuda") / (1024 * 1024)

        Log(f"Algorithm Allocated Peak: {algo_allocated_mb:.2f} MB", tag="Eval")
        Log(f"System Physical Peak (Paper Metric): {real_peak_memory_mb:.2f} MB", tag="Eval")

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

    Log("Done.")
