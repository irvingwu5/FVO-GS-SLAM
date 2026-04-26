import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import os
import numpy as np
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import (get_loss_mapping,_save_normal_pair, _save_rendered_rgb,_save_gt_normal,check_normal_dir)
import torch.nn.functional as F

class BackEnd(mp.Process):
    # ========================================================================
    # 1. Initialization
    # ========================================================================
    def __init__(self, config):
        super().__init__()
        self.config = config

        # ===== 外部注入的共享对象（由主进程在 start() 前赋值）=====
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None #后端到前端的通信队列
        self.backend_queue = None #前端到后端的通信队列
        self.loop_queue = None

        # ===== 进程控制 =====
        self.live_mode = False
        self.pause = False
        self.single_thread = False
        self.device = "cuda"
        self.dtype = torch.float32

        # ===== 建图状态 =====
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # ===== 消融开关 =====
        self.use_fdn = self.config.get("Ablation", {}).get("use_fdn", True)

        # ===== 子图状态 =====
        self.current_submap_id = 0
        self.current_submap_seed_global_c2w = np.eye(4, dtype=np.float64)

        # ===== 子图切割参数 =====
        self.enable_cut_local_ba = self.config.get("Submap", {}).get("enable_cut_local_ba", True)
        self.cut_local_ba_iters = self.config.get("Submap", {}).get("cut_local_ba_iters", 40)
        self.cut_local_prune_iters = self.config.get("Submap", {}).get("cut_local_prune_iters", 8)
        self.seed_init_iters = self.config.get("Submap", {}).get("seed_init_iters", 500)

    # ========================================================================
    # 2. Hyperparameters
    # ========================================================================
    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.nonvisible_reset_opacity = self.config["Training"].get("nonvisible_reset_opacity", 0.05)
        self.nonvisible_reset_stable_opacity = self.config["Training"].get("nonvisible_reset_stable_opacity", 0.08)
        self.nonvisible_reset_stable_n_obs = self.config["Training"].get("nonvisible_reset_stable_n_obs", 4)

    # ========================================================================
    # 3. State Management
    # ========================================================================
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        if len(self.gaussians._xyz) > 0:
            self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

        while not self.backend_queue.empty():
            self.backend_queue.get()

    # ========================================================================
    # 4. Seed Viewpoint Preparation
    # ========================================================================
    def prepare_seed_viewpoint_for_backend_init(self, viewpoint):
        viewpoint.is_submap_seed = True
        viewpoint.fixed_pose = True

        viewpoint.reset_pose_deltas()
        viewpoint.cam_rot_delta.requires_grad_(False)
        viewpoint.cam_trans_delta.requires_grad_(False)

    # ========================================================================
    # 5. Map Initialization
    # ========================================================================
    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def initialize_map(self, cur_frame_idx, viewpoint, iters=None):
        if iters is None:
            iters = self.init_itr_num

        for mapping_iteration in range(iters):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, initialization=True
            )

            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            if mapping_iteration % 5 == 0:
                self.push_to_frontend()

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")

    # ========================================================================
    # 6. Map Optimization (Local BA)
    # ========================================================================
    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for itr in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            keyframes_opt = []

            for i in range(len(current_window)):
                viewpoint = viewpoint_stack[i]
                keyframes_opt.append(viewpoint)

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_view = get_loss_mapping(
                    self.config, image, depth, viewpoint
                )

                if self.use_fdn and viewpoint.normal is not None:
                    rend_normal = render_pkg["rend_normal"]
                    rend_normal = F.normalize(rend_normal, p=2, dim=0)
                    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
                    sensor_normal = viewpoint.normal
                    gt_normal = (viewpoint.T[0:3, 0:3].T @ sensor_normal.view(3, -1)).view(
                        image.shape[0], image.shape[1], image.shape[2]
                    )
                    normal_mask = gt_normal > 0
                    normal_error = (1 - (rend_normal * gt_normal * depth_pixel_mask * normal_mask).sum(dim=0))[None].mean()
                    loss_view += (self.config["opt_params"]["lambda_sensor_normal"] * normal_error)

                loss_view.backward()

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                del render_pkg

                if i % 5 == 0:
                    torch.cuda.empty_cache()

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, surf=False
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_view = get_loss_mapping(
                    self.config, image, depth, viewpoint
                )

                loss_view.backward()

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

                del render_pkg
                torch.cuda.empty_cache()

            gaussian_split = False

            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                        if prune_mode == "slam":
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                        self.iteration_count % self.gaussian_update_every
                        == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                if (self.iteration_count % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    actual_touched_filters = [(n_touched > 0) for n_touched in n_touched_acm]
                    self.gaussians.reset_opacity_nonvisible(
                        actual_touched_filters,
                        target_opacity=self.nonvisible_reset_opacity,
                        stable_opacity=self.nonvisible_reset_stable_opacity,
                        stable_n_obs=self.nonvisible_reset_stable_n_obs,
                    )
                    gaussian_split = True

                if self.keyframe_optimizers is not None:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)

                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                    frames_to_optimize = self.config["Training"]["pose_window"]
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]

                        is_fixed_pose = getattr(viewpoint, "fixed_pose", False)

                        if is_fixed_pose:
                            viewpoint.reset_pose_deltas()
                            continue

                        update_pose(viewpoint)
                else:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)

        return gaussian_split

    # ========================================================================
    # 7. Submap Helpers
    # ========================================================================
    def pack_submap_keyframe_poses(self):
        submap_keyframe_poses = {}

        for kf_idx, viewpoint in self.viewpoints.items():
            with torch.no_grad():
                c2w = torch.linalg.inv(viewpoint.T.detach()).cpu().numpy()
            submap_keyframe_poses[int(kf_idx)] = c2w.astype(np.float64)

        return submap_keyframe_poses

    # ========================================================================
    # 8. Submap Freezing (Local BA before checkpoint)
    # ========================================================================
    def finalize_submap_before_freeze(self):
        if not self.enable_cut_local_ba:
            return

        if len(self.current_window) == 0:
            return

        if self.keyframe_optimizers is None:
            Log("[SubmapLocalBA] skip: keyframe_optimizers is None")
            return

        ba_iters = max(int(self.cut_local_ba_iters), 0)
        prune_iters = max(int(self.cut_local_prune_iters), 0)

        Log(
            f"[SubmapLocalBA] freeze 前局部优化开始 | "
            f"window={len(self.current_window)}, "
            f"ba_iters={ba_iters}, prune_iters={prune_iters}"
        )

        if ba_iters > 0:
            self.map(self.current_window, prune=False, iters=ba_iters)

        if prune_iters > 0:
            self.map(self.current_window, prune=True, iters=prune_iters)

        Log("[SubmapLocalBA] freeze 前局部优化完成")

    # ========================================================================
    # 9. Color Refinement (offline)
    # ========================================================================
    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]

            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background, surf=False
            )
            image, viewspace_point_tensor, visibility_filter, radii, depth, opacity, n_touched = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

            del render_pkg

            if iteration % 100 == 0:
                torch.cuda.empty_cache()
        Log("Map refinement done")

    # ========================================================================
    # 10. Frontend Communication
    # ========================================================================
    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []

        if len(self.current_window) > 0:
            for kf_idx in self.current_window:
                if kf_idx in self.viewpoints:
                    kf = self.viewpoints[kf_idx]
                    keyframes.append((kf_idx, kf.T.clone()))
        else:
            if len(self.viewpoints) > 0:
                latest_kf_idx = sorted(self.viewpoints.keys())[-1]
                kf = self.viewpoints[latest_kf_idx]
                keyframes.append((latest_kf_idx, kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    # ========================================================================
    # 11. Main Loop
    # ========================================================================
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue

                if self.keyframe_optimizers is None:
                    time.sleep(0.01)
                    continue

                if self.pause or len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    save_dir = self.config["Results"]["save_dir"]
                    submaps_dir = os.path.join(save_dir, "submaps")
                    os.makedirs(submaps_dir, exist_ok=True)

                    gaussian_params = self.gaussians.capture_dict()
                    submap_keyframes = sorted(list(self.viewpoints.keys()))
                    ckpt_data = {
                        "gaussian_params": gaussian_params,
                        "submap_keyframes": submap_keyframes,
                        "seed_global_c2w": self.current_submap_seed_global_c2w,
                        "submap_keyframe_poses": self.pack_submap_keyframe_poses(),
                        "relative_pose": np.eye(4, dtype=np.float64),
                        "correct_tsfm": np.eye(4, dtype=np.float64),
                    }
                    ckpt_path = os.path.join(submaps_dir, f"{self.current_submap_id:06d}.ckpt")
                    torch.save(ckpt_data, ckpt_path)

                    kf_image_paths = []
                    if len(submap_keyframes) > 0:
                        for kf_idx in submap_keyframes:
                            kf_image = self.viewpoints[kf_idx].original_image.cpu()
                            img_path = os.path.join(submaps_dir, f"{self.current_submap_id:06d}_img_{kf_idx}.pt")
                            torch.save(kf_image, img_path)
                            kf_image_paths.append(img_path)

                    if hasattr(self, 'loop_queue') and self.loop_queue is not None and len(kf_image_paths) > 0:
                        self.loop_queue.put(["submap_saved", self.current_submap_id, ckpt_path, kf_image_paths])

                    Log(f"==> 终局保存：最后一块子图 {self.current_submap_id} 已存入硬盘。 <==")
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]

                    seed_global_c2w_from_viewpoint = (
                        torch.linalg.inv(viewpoint.T.detach()).cpu().numpy().astype(np.float64)
                    )

                    if self.current_submap_id == 0 and len(self.viewpoints) == 0:
                        self.current_submap_seed_global_c2w = seed_global_c2w_from_viewpoint.copy()
                    else:
                        self.current_submap_seed_global_c2w = seed_global_c2w_from_viewpoint.copy()

                    if len(self.gaussians._xyz) == 0 and self.current_submap_id > 0:
                        Log("Initializing new submap from seed frame (state already clean)")
                        self.iteration_count = 0
                        self.occ_aware_visibility = {}
                        self.viewpoints = {}
                        self.current_window = []
                        self.initialized = not self.monocular
                        self.keyframe_optimizers = None
                    else:
                        Log("Resetting the system")
                        self.reset()

                    self.prepare_seed_viewpoint_for_backend_init(viewpoint)

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = [cur_frame_idx]
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )

                    if len(self.gaussians._xyz) == 0 and self.current_submap_id > 0:
                        init_iters = self.seed_init_iters
                    else:
                        init_iters = self.init_itr_num

                    self.initialize_map(cur_frame_idx, viewpoint, iters=init_iters)

                    self.push_to_frontend("init")
                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    for cam_idx in range(len(self.current_window)):
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        should_opt = (cam_idx < frames_to_optimize)

                        if should_opt and not getattr(viewpoint, "fixed_pose", False):
                            rot_lr = self.config["Training"]["lr"]["cam_rot_delta"] * 0.5
                            trans_lr = self.config["Training"]["lr"]["cam_trans_delta"] * 0.5

                            opt_params.append({
                                "params": [viewpoint.cam_rot_delta],
                                "lr": rot_lr,
                                "name": "rot_{}".format(viewpoint.uid),
                            })
                            opt_params.append({
                                "params": [viewpoint.cam_trans_delta],
                                "lr": trans_lr,
                                "name": "trans_{}".format(viewpoint.uid),
                            })
                            opt_params.append({
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            })
                            opt_params.append({
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            })
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)
                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")

                elif data[0] == "new_submap":
                    completed_submap_id = data[1]
                    relative_pose = data[2] if len(data) > 2 else np.eye(4, dtype=np.float64)
                    relative_pose = np.array(relative_pose, dtype=np.float64)
                    new_seed_global_c2w = (
                        data[3] if len(data) > 3 else np.eye(4, dtype=np.float64)
                    )
                    new_seed_global_c2w = np.array(new_seed_global_c2w, dtype=np.float64)

                    completed_seed_global_c2w = np.array(
                        self.current_submap_seed_global_c2w, dtype=np.float64
                    )
                    self.current_submap_id = completed_submap_id + 1
                    self.current_submap_seed_global_c2w = new_seed_global_c2w.copy()
                    Log(f"==> Backend received new_submap signal. Freezing submap {completed_submap_id}...")

                    self.finalize_submap_before_freeze()

                    save_dir = self.config["Results"]["save_dir"]
                    submaps_dir = os.path.join(save_dir, "submaps")
                    os.makedirs(submaps_dir, exist_ok=True)

                    gaussian_params = self.gaussians.capture_dict()
                    submap_keyframes = sorted(list(self.viewpoints.keys()))
                    ckpt_data = {
                        "gaussian_params": gaussian_params,
                        "submap_keyframes": submap_keyframes,
                        "seed_global_c2w": completed_seed_global_c2w,
                        "submap_keyframe_poses": self.pack_submap_keyframe_poses(),
                        "relative_pose": relative_pose,
                        "correct_tsfm": np.eye(4, dtype=np.float64),
                    }

                    ckpt_path = os.path.join(submaps_dir, f"{completed_submap_id:06d}.ckpt")
                    torch.save(ckpt_data, ckpt_path)
                    Log(f"✓ Submap {completed_submap_id} parameters saved to {ckpt_path}")

                    kf_image_paths = []
                    if len(submap_keyframes) > 0:
                        for kf_idx in submap_keyframes:
                            kf_image = self.viewpoints[kf_idx].original_image.cpu()
                            img_path = os.path.join(submaps_dir, f"{completed_submap_id:06d}_img_{kf_idx}.pt")
                            torch.save(kf_image, img_path)
                            kf_image_paths.append(img_path)

                    if hasattr(self, 'loop_queue') and self.loop_queue is not None and len(kf_image_paths) > 0:
                        self.loop_queue.put(["submap_saved", completed_submap_id, ckpt_path, kf_image_paths])
                        Log(f"✓ Submap {completed_submap_id} sent to loop closure")

                    self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
                    Log("✓ Pruned ALL Gaussian points for true independent submap")

                    self.viewpoints.clear()
                    self.current_window = []
                    self.occ_aware_visibility = {}
                    self.keyframe_optimizers = None

                    self.gaussians.training_setup(self.opt_params)

                    torch.cuda.empty_cache()
                    Log("✓ Backend state fully reset. Waiting for seed frame init...")

        while not self.backend_queue.empty():
            try:
                self.backend_queue.get_nowait()
            except:
                break

        while not self.frontend_queue.empty():
            try:
                self.frontend_queue.get_nowait()
            except:
                break

        return
