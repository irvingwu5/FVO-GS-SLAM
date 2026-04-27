# FVO-GS-SLAM: FFT Visual Odometry Guided RGBD 2D Gaussian Splatting SLAM

FVO-GS-SLAM is an RGBD SLAM system based on 2D Gaussian Splatting and differentiable surfel rendering. The current development branch extends the original MonoGS style front end and back end SLAM pipeline with FFT guided RGBD visual odometry initialization, RGB depth normal guided local mapping, motion based submap management, optional submap level loop closure, adjacent submap edge refinement, pose graph correction, streaming global Gaussian fusion, rendering evaluation, trajectory evaluation, and controlled ablation switches.

The maintained direction of this repository is RGBD indoor SLAM on datasets such as TUM RGBD, Replica, and ScanNet++. Some monocular or stereo files inherited from the upstream project may still exist, but the active system design centers on RGBD 2D Gaussian SLAM.

<p align="center">
  <a href="">
    <img src="./media/pipeline.png" alt="pipeline" width="100%">
  </a>
</p>

---

## Statement

This repository is developed from the MonoGS style Gaussian SLAM framework and has been extended into an RGBD 2D Gaussian SLAM system with FFT visual odometry guided tracking initialization and submap based global consistency handling.

The current system includes:

- RGBD dataset loading and camera model construction
- front end camera tracking
- FFT high frequency mask generation
- FFT guided RGBD visual odometry initialization
- previous pose, constant speed, and FFTVO candidate based tracking initialization
- render loss based candidate selection
- hard rejection, direct acceptance, and rescue refinement for FFTVO candidates
- keyframe selection and sliding window management
- asynchronous back end Gaussian mapping
- EAGS style back end pose policy with Gaussian only optimization by default
- 2D Gaussian map representation
- differentiable surfel rendering
- RGB and depth tracking loss
- RGB and depth mapping loss
- finite difference normal supervision
- Gaussian densification, opacity reset, and pruning
- visibility maintenance with `occ_aware_visibility`
- motion based submap cutting relative to the current submap anchor
- independent submap initialization from seed frames
- submap checkpoint saving with Gaussian parameters, keyframes, seed pose, relative pose, and correction transform
- optional CosPlace based submap loop candidate retrieval
- ICP based geometric verification
- adjacent submap edge refinement
- optional pose graph optimization
- 2D Gaussian rigid transform based global fusion
- streaming submap loading and global Gaussian concatenation
- optional offline color refinement
- ATE evaluation and rendering quality evaluation
- optional GUI visualization
- ablation switches for controlled experiments
- GPU memory and map size logging for evaluation

---

## Main Differences from MonoGS

### 1. RGBD oriented 2D Gaussian SLAM pipeline

The system uses RGBD observations to initialize, track, and optimize a 2D Gaussian map. The renderer outputs RGB, depth, opacity, visibility, radii, normal related tensors, and `n_touched`. These outputs are used by tracking loss, mapping loss, visibility update, Gaussian densification, Gaussian pruning, rendering evaluation, and global fusion.

### 2. FFT guided visual odometry initialization

The project adds an FFT guided RGBD visual odometry module. The frequency filter extracts high frequency image regions through FFT based high pass filtering and thresholding. The RGBD VO module uses FAST keypoints, Lucas Kanade optical flow, depth based 3D back projection, and PnP RANSAC to estimate a candidate camera pose between adjacent frames.

The front end does not blindly trust this VO output. Instead, it compares candidate initial poses, including previous pose, optional constant speed pose, and FFTVO pose, by rendering each candidate and measuring the tracking loss. FFTVO candidates are accepted only when their geometry and render loss pass the configured gates.

### 3. Three tier FFTVO candidate control

The current front end contains a conservative candidate selection policy:

- **Hard reject**: discard FFTVO when inlier ratio, inlier count, translation, or rotation is outside the safe range.
- **Direct accept**: accept FFTVO when geometry is reliable and render loss is clearly better than the previous pose.
- **Rescue trial**: run a small temporary pose refinement on the FFTVO candidate, then accept it only if the refined loss is better and the state can be restored safely.

This design makes FFTVO a tracking initialization aid instead of a replacement for differentiable render based tracking.

### 4. Front end tracking and keyframe management

The front end is responsible for camera tracking, keyframe insertion, sliding window maintenance, visibility synchronization, submap cut decisions, and communication with the back end. It receives Gaussian and visibility updates from the back end, optimizes the current camera pose using render based losses, then sends initialization, keyframe, and submap control messages to the back end.

### 5. Back end Gaussian only local mapping

The back end initializes and optimizes the Gaussian map from keyframes. By default, keyframe pose and exposure optimization in the back end are disabled. This keeps the back end focused on Gaussian parameters and avoids unexpected pose drift caused by back end pose updates. Pose sanity checks are kept to detect accidental keyframe pose changes.

### 6. Submap based SLAM

The current submap cut decision is motion based. A new submap can be started when the current camera motion relative to the current submap anchor exceeds the configured translation or rotation threshold. When a submap is closed, its Gaussian parameters and metadata are saved to disk, then the online map is cleared for the next independent submap.

Each submap checkpoint is expected to contain at least:

- `gaussian_params`
- `submap_keyframes`
- `seed_global_c2w`
- `submap_keyframe_poses`
- `relative_pose`
- `correct_tsfm`

### 7. Loop closure, adjacent edges, and pose graph correction

The loop closure process works at submap level. It extracts visual descriptors from saved keyframe images, keeps submap point cloud caches, verifies candidate edges through ICP, optionally refines adjacent submap edges, and writes correction related information back to submap checkpoints.

The system distinguishes these concepts:

- **Loop candidate retrieval**: visual retrieval through CosPlace style descriptors.
- **ICP verification**: geometric check and relative transform estimation.
- **Adjacent submap edge refinement**: local consistency check between neighboring submaps.
- **Global correction**: pose graph correction that generates transforms for final fusion.

Adjacent edges should not be treated as true loop closure edges.

### 8. Streaming global fusion and offline refinement

After the front end finishes, the back end saves the final submap. The main process then stops the loop closure process, loads submap checkpoints from disk, applies the stored anchor and correction transforms, streams Gaussian tensors to avoid excessive memory pressure, concatenates the global Gaussian model, corrects the front end camera trajectory, evaluates the result, and optionally runs offline color refinement.

---

## Repository Structure

```text
FVO-GS-SLAM
├── slam.py                         # main entry, process orchestration, evaluation, streaming fusion
├── run_ablation.py                 # ablation runner
├── run_all_slam.sh                 # batch running script
├── configs/
│   └── rgbd/
│       ├── tum/                    # TUM RGBD configs
│       ├── replica/                # Replica configs
│       └── scannetpp/              # ScanNet++ configs
├── gaussian_splatting/
│   ├── gaussian_renderer/          # differentiable 2DGS or surfel rendering
│   └── scene/gaussian_model.py     # Gaussian parameters, densify, prune, optimizer state
├── gui/                            # optional GUI visualization
├── scripts/                        # dataset and utility scripts
├── utils/
│   ├── slam_frontend.py            # tracking, FFTVO candidate selection, keyframes, submap decisions
│   ├── slam_backend.py             # Gaussian mapping, pruning, submap saving, queue handling
│   ├── fft_filter.py               # FFT high frequency mask generation
│   ├── fft_rgbd_vo.py              # FFT guided RGBD VO initialization
│   ├── loop_closure.py             # CosPlace, ICP verification, adjacent edges, PGO correction
│   ├── slam_utils.py               # tracking loss, mapping loss, normal utilities, depth utilities
│   ├── pose_utils.py               # pose update utilities
│   ├── camera_utils.py             # camera representation
│   ├── dataset.py                  # dataset loading
│   └── eval_utils.py               # ATE and rendering evaluation
└── weights/                        # CosPlace or other model weights
```

---

## System Architecture

```text
RGBD sequence
    ↓
Dataset loader and Camera objects
    ↓
FrontEnd
    ├── create current frame Camera
    ├── choose tracking initialization
    │   ├── previous pose candidate
    │   ├── optional constant speed candidate
    │   └── FFT guided RGBD VO candidate
    ├── render each candidate and compare tracking loss
    ├── optimize current camera pose by differentiable rendering
    ├── decide keyframe insertion
    ├── maintain current window and visibility
    └── decide whether to start a new submap
    ↓ queue messages
BackEnd
    ├── initialize Gaussian map from seed keyframe
    ├── extend Gaussian map from new keyframes
    ├── optimize Gaussian parameters
    ├── apply RGB depth normal supervision
    ├── densify, reset opacity, and prune
    ├── maintain occ_aware_visibility
    ├── push map snapshot back to FrontEnd
    └── save and reset independent submaps
    ↓ saved submap checkpoints
LoopClosureProcess, optional
    ├── load saved submap metadata
    ├── extract visual features from saved keyframe images
    ├── load sparse and dense point clouds
    ├── verify loop candidates by ICP
    ├── refine adjacent submap edge when enabled
    └── write correction transforms or refined edges
    ↓
Main process after tracking
    ├── stop backend and loop closure
    ├── stream submap checkpoints from disk
    ├── apply anchor and correction transforms
    ├── merge global Gaussian model
    ├── correct camera trajectory
    ├── evaluate ATE and rendering quality
    └── optional offline color refinement
```

---

## Module Roles and Data Flow

### `slam.py`

`slam.py` is the main entry and system controller. It loads configuration files, creates the Gaussian model, dataset, front end, back end, optional GUI, and optional loop closure process. It also handles evaluation mode overrides, W&B logging, GPU memory measurement, final submap loading, streaming global fusion, trajectory correction, rendering evaluation, offline color refinement, and final model saving.

Main responsibilities:

- parse and load YAML configs
- initialize `GaussianModel`
- load dataset
- create communication queues
- start back end process
- start loop closure process when enabled
- run front end in the main process
- stop back end safely
- stop loop closure safely
- merge submaps from checkpoint files
- apply `correct_tsfm` and submap anchor poses
- correct camera poses for evaluation
- evaluate and save results

### `utils/slam_frontend.py`

The front end is the online tracking and scheduling module.

Main responsibilities:

- construct per frame `Camera` objects
- initialize the first frame or submap seed frame
- generate FFT masks for keyframes when enabled
- estimate FFTVO candidates when enabled
- evaluate candidate poses by one render loss pass
- run rescue refinement without polluting formal tracking state
- optimize current camera pose by render based tracking
- insert keyframes
- manage sliding window
- synchronize Gaussian map and visibility from the back end
- compute motion relative to submap anchor
- trigger submap cut when motion threshold is exceeded
- send `init`, `keyframe`, `new_submap`, `pause`, and `stop` style messages to the back end

### `utils/fft_filter.py`

This module builds a high frequency mask from RGB images.

Main responsibilities:

- pad input image
- convert to grayscale and enhance local contrast
- apply FFT and Gaussian high pass filtering
- inverse FFT to recover high frequency response
- crop back to original image size
- threshold the response to produce a boolean mask

This mask is used by Gaussian sampling and FFTVO feature selection.

### `utils/fft_rgbd_vo.py`

This module provides FFT guided RGBD visual odometry initialization.

Main responsibilities:

- read adjacent RGB images from camera objects
- generate or reuse FFT high frequency masks
- detect FAST keypoints in high frequency regions
- track points by pyramidal Lucas Kanade optical flow
- back project valid previous frame points through depth
- estimate relative motion by `solvePnPRansac`
- reject motion if inlier count, depth validity, translation, or rotation is unsafe
- output a camera to world pose candidate for current frame

### `utils/slam_backend.py`

The back end is the asynchronous local mapping module.

Main responsibilities:

- receive initialization, keyframe, and submap messages
- initialize Gaussian map from seed keyframe
- add new keyframes into the Gaussian map
- optimize Gaussian parameters with mapping loss
- optionally apply finite difference normal supervision
- collect visibility and densification statistics
- densify, reset opacity, and prune Gaussian points
- keep `occ_aware_visibility` keyed by keyframe index
- push Gaussian snapshots and keyframe poses to the front end
- save submap checkpoints on `new_submap` and `stop`
- send saved submap metadata to loop closure
- prune all Gaussian points and reset state for independent submap initialization

### `utils/loop_closure.py`

This module manages submap level global consistency.

Main responsibilities:

- define and load CosPlace style visual retrieval model
- maintain submap checkpoint records
- cache sparse and dense submap point clouds
- extract image descriptors from saved keyframe images
- search potential loop candidates
- verify candidate constraints through ICP
- reject unreliable loop edges based on fitness, RMSE, translation, and rotation thresholds
- refine adjacent submap edges when configured
- write `prev_submap_tsfm_refined`, `prev_submap_info_matrix`, and metrics to checkpoints
- build anchor chains and correction transforms for final fusion
- apply rigid transforms to 2D Gaussian parameters through `rigid_transform_2dgs`

### `gaussian_splatting/scene/gaussian_model.py`

This file stores and updates the Gaussian map parameters.

Main responsibilities:

- keep Gaussian tensors such as `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation`, and `_normal`
- initialize Gaussian points from RGBD observations
- extend the map from keyframes
- maintain optimizer parameter groups
- perform densification and pruning
- reset opacity
- capture and restore Gaussian parameter dictionaries for checkpoints

### `gaussian_splatting/gaussian_renderer/__init__.py`

The renderer is the differentiable observation model used by both tracking and mapping.

Main outputs:

- `render`
- `viewspace_points`
- `visibility_filter`
- `radii`
- `depth`
- `opacity`
- `normal` or related normal fields
- `rend_normal`
- `surf_normal`
- `n_touched`

These outputs are consumed by tracking, mapping, normal supervision, visibility update, densification, pruning, and evaluation.

---

## Main Queue Messages

The front end and back end communicate through multiprocessing queues.

Typical front end to back end messages:

```text
["init", cur_frame_idx, viewpoint, depth_map]
["keyframe", cur_frame_idx, viewpoint, current_window, depth_map]
["new_submap", completed_submap_id, relative_pose, new_seed_global_c2w]
["pause"]
["unpause"]
["stop"]
```

Typical back end to front end messages:

```text
["init", gaussians, occ_aware_visibility, keyframes]
["keyframe", gaussians, occ_aware_visibility, keyframes]
["sync_backend", gaussians, occ_aware_visibility, keyframes]
```

Typical back end to loop closure messages:

```text
["submap_saved", submap_id, ckpt_path, kf_image_paths]
["stop"]
```

Do not change these message formats unless every sender and receiver is updated together.

---

## Configuration Overview

Important configuration groups:

- `Results`: save path, trajectory saving, GUI, rendering evaluation, W&B logging
- `Dataset`: dataset type, sensor type, point cloud sampling, camera settings
- `Training`: initialization, tracking, mapping, keyframe, window, learning rate, pruning, and loss settings
- `FFTVO`: FFTVO initialization, candidate selection, RANSAC, gate thresholds, warmup, rescue mode, and logging
- `Backend`: back end pose policy and pose sanity checking
- `Submap`: motion threshold and seed pose behavior
- `LoopClosure`: retrieval, ICP, adjacent edge, PGO, and protection parameters
- `opt_params`: Gaussian optimizer and densification parameters
- `model_params`: Gaussian model settings
- `pipeline_params`: renderer settings
- `Ablation`: switches for submap, loop closure, finite difference normal, FFT mask, error mask, and color refinement

Generic parameters should be added to these base configs together:

```text
configs/rgbd/tum/base_config.yaml
configs/rgbd/replica/base_config.yaml
configs/rgbd/scannetpp/base_config.yaml
```

Scene specific config files should only override existing parameters.

---

## Installation

```bash
git clone https://github.com/irvingwu5/FVO-GS-SLAM.git --recursive
cd FVO-GS-SLAM
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

Depending on your CUDA and PyTorch versions, you may need to adjust the versions in `environment.yml` or install PyTorch manually before installing the remaining packages.

---

## Download Datasets

TUM RGBD:

```bash
bash scripts/download_tum.sh
```

Replica:

```bash
bash scripts/download_replica.sh
```

EuRoC MAV, inherited from the original project:

```bash
bash scripts/download_euroc.sh
```

---

## Run

### TUM RGBD

```bash
python slam.py --config configs/rgbd/tum/fr1_desk.yaml --eval
```

```bash
python slam.py --config configs/rgbd/tum/fr3_office.yaml --eval
```

### Replica

```bash
python slam.py --config configs/rgbd/replica/office0.yaml --eval
```

Single process config, if provided:

```bash
python slam.py --config configs/rgbd/replica/office0_sp.yaml --eval
```

### ScanNet++

```bash
python slam.py --config configs/rgbd/scannetpp/your_scene.yaml --eval
```

---

## Evaluation

Use `--eval` to force evaluation mode. In this mode the script overrides several runtime options:

```text
save_results = True
use_gui = False
eval_rendering = True
use_wandb = False
```

Example:

```bash
python slam.py --config configs/rgbd/tum/fr1_desk.yaml --eval
```

The output directory is created under the configured `Results.save_dir`. A typical result folder contains:

```text
config.yml
frame_to_submap.pt
submap_anchor_poses.pt, when available
submaps/*.ckpt
point_cloud/final/point_cloud.ply
rendering evaluation outputs
trajectory and ATE outputs
```

Evaluation logs include FPS, ATE, rendering metrics, algorithm allocated GPU memory, physical GPU memory peak, and final map size when the final PLY exists.

---

## Ablation Notes

The following switches are commonly used for controlled experiments:

```yaml
Ablation:
  use_submap: True
  use_loop_closure: True
  use_fdn: True
  use_fft_mask: True
  use_error_mask: True
  use_color_refinement: True

FFTVO:
  use_fft_vo_init: true
  use_previous_candidate: true
  use_const_speed_candidate: false
  use_fft_vo_candidate: true
  enable_rescue_mode: true

Backend:
  optimize_keyframe_pose: false
  optimize_keyframe_exposure: false
```

For debugging FFTVO alone, you can disable PGO by setting:

```yaml
LoopClosure:
  debug_disable_pgo_for_fftvo_test: true
```

This is useful when you want to evaluate whether FFTVO improves tracking before global corrections are introduced.

---

## Reproducibility Notes

- The system is sensitive to CUDA, PyTorch, Open3D, and differentiable Gaussian rasterizer versions.
- The recommended workflow is to run a short smoke test first, then run full evaluation.
- For fair comparison, keep the same dataset sequence, image resolution, tracking iterations, mapping iterations, submap thresholds, and ablation switches.
- When analyzing submap experiments, report both tracking ATE and rendering metrics before and after global fusion or offline color refinement.
- When debugging FFTVO, log candidate losses, inlier count, inlier ratio, translation, rotation, selected candidate, and rescue result.

---

## Acknowledgement

This project is developed based on the MonoGS style Gaussian SLAM framework. The repository further adapts it toward RGBD 2D Gaussian SLAM with FFT visual odometry initialization, independent submaps, submap level consistency checks, and streaming global fusion.

---

## License

This modified version follows the license terms inherited from the original project. Please refer to `LICENSE.md` for details.

---

## Additional Notes

- Original framework: MonoGS
- Modified project: FVO-GS-SLAM
- Maintainer field in older files: `wxy`
- Current documentation update: 2026-04-27