# FVO-GS-SLAM: FFT Edge VO Guided RGBD 2D Gaussian Splatting SLAM

FVO-GS-SLAM is an RGBD SLAM system based on 2D Gaussian Splatting and differentiable surfel rendering. The system uses dense FFT Edge VO (EAGS-SLAM style DT alignment + LM optimization) for tracking initialization, render-based pose refinement, RGB/depth/normal joint Gaussian-only mapping, motion-based submap management, CosPlace retrieval + Reloc3R/2DGS-GSReg/ICP loop closure, pose graph optimization, and streaming global Gaussian fusion.

The maintained direction is RGBD indoor SLAM on TUM RGBD, Replica, and ScanNet++ datasets.

<p align="center">
  <a href="">
    <img src="./media/pipeline.png" alt="pipeline" width="100%">
  </a>
</p>

---

## Statement

This repository is developed from the MonoGS style Gaussian SLAM framework and has been extended into an RGBD 2D Gaussian SLAM system with FFT Edge VO guided tracking initialization, motion-based submap management, and submap-level global consistency handling.

The current system includes:

- RGBD dataset loading (TUM, Replica, ScanNet++, Realsense live)
- FFT high-frequency mask generation (CLAHE → FFT → Gaussian HPF → IFFT → threshold)
- FFT Edge VO: dense DT alignment + damped Gauss-Newton (EAGS-SLAM / Edge VO style)
- Render-based pose refinement (RGB + depth tracking loss)
- Keyframe selection and sliding window management
- Asynchronous back end Gaussian-only mapping (RGB + depth + normal)
- 2D Gaussian map representation with differentiable surfel rendering
- Finite difference normal (FDN) supervision
- Gaussian densification, opacity reset, and pruning
- Visibility maintenance with `occ_aware_visibility`
- Motion-based submap cutting (translation/rotation threshold relative to submap anchor)
- Independent submap initialization from seed frames
- Submap checkpoint saving (Gaussian params, keyframe poses, seed global C2W, relative/correct tsfm)
- RSKM (Random Sampling Keyframe Mapping): random keyframe replay within active submap
- Cross-submap covisibility handoff: frozen boundary Gaussians smooth submap transitions
- Active-only coverage for correct hole detection after submap cut
- CosPlace visual descriptor extraction from saved keyframe images
- Reloc3R (DUSt3R variant) top-K keyframe pair registration with consistency verification
- 2DGS-GSReg (LoopSplat-style render-based registration) as registration option
- ICP (point-to-plane with FDN normals) as fallback registration
- Pose graph optimization (Open3D) with incremental gating and PGO safety valve
- Reloc3R edge quality filters: raw_vs_init_dot direction alignment, delta_t/delta_r gates
- Streaming submap loading and global Gaussian concatenation
- `rigid_transform_2dgs` for applying correction transforms to 2DGS params
- ATE trajectory evaluation and rendering quality evaluation
- Optional GUI visualization
- Ablation switches for controlled experiments
- GPU memory monitoring (peak minus baseline)

---

## Main Differences from MonoGS

### 1. RGBD oriented 2D Gaussian SLAM pipeline

The system uses RGBD observations to initialize, track, and optimize a 2D Gaussian map. The renderer outputs RGB, depth, opacity, visibility, radii, normal, and `n_touched`. These outputs feed tracking loss, mapping loss, visibility update, densification, pruning, evaluation, and global fusion.

### 2. FFT Edge VO for tracking initialization

The project adds a dense FFT Edge VO module (`utils/fft_edge_vo.py`), aligned with EAGS-SLAM's Edge VO design:

- **cur→ref direction**: current-frame 3D points projected into reference-frame distance-transform pyramid.
- Reference DT + Sobel gradients are built once in `set_reference()` and reused for subsequent frames.
- Per-frame optimization: damped Gauss-Newton (LM-style) with analytic SE(3) Jacobian.
- Coarse-to-fine pyramid optimization with configurable levels and iteration budgets.
- The initial guess (`init_c2w`) is properly seeded into the optimizer via `_se3_log(T_rc_init)`.

FFTEdgeVO provides the tracking initial pose but does not replace differentiable render-based refinement.

### 3. Front end tracking and keyframe management

The front end is the main process, responsible for camera tracking, keyframe insertion, sliding window maintenance, visibility synchronization, submap cut decisions, and communication with the back end.

Tracking flow per frame:
1. FFT mask generation for the current frame
2. FFTEdgeVO.track() → initial pose from dense DT alignment
3. Render-based pose refinement (Adam on `cam_rot_delta` / `cam_trans_delta`)
4. Auto-refresh FFTEdgeVO reference when quality degrades
5. Keyframe decision (overlap ratio + translation threshold)
6. Submap motion monitoring and cut triggering

### 4. Back end Gaussian-only local mapping

The back end is an independent process that initializes and optimizes the 2D Gaussian map from keyframes. By default, `optimize_keyframe_pose` is `true` (EAGS-style: keyframe pose optimization enabled with pose sanity checks). `optimize_keyframe_exposure` is `false`.

Mapping includes:
- RGB L1 + DSSIM loss
- Depth L1 loss
- FDN normal supervision (when enabled)
- FFT frequency mask guided sampling and initial scale
- Error mask guided dynamic point insertion (holes + depth penetration)
- Periodic densify, opacity reset, and prune

### 5. Submap based SLAM

Motion-based submap cutting: a new submap starts when current camera motion relative to the submap anchor exceeds the configured translation or rotation threshold. Each submap is an independent memory and optimization partition.

Submap checkpoint fields:
- `gaussian_params` — `capture_dict()` output
- `submap_keyframes` — sorted keyframe indices
- `seed_global_c2w` — seed frame global C2W
- `submap_keyframe_poses` — all keyframe C2W poses
- `relative_pose` — previous submap seed → current submap seed
- `correct_tsfm` — loop closure / PGO correction (default identity)

On submap cut, the back end prunes ALL Gaussian points and resets optimizer state for the next independent submap.

When `use_handoff: true`, the system preserves a small set of boundary Gaussians from the old submap as short-term frozen tracking support (see [Handoff Mechanism](#7-cross-submap-covisibility-handoff)).

### 6. Loop closure and PGO

The loop closure process runs as an independent process:

1. **Visual retrieval**: Extract CosPlace descriptors (ResNet18 backbone + GeM pooling, weights loaded from disk) from saved keyframe images. Adaptive threshold from self-similarity.
2. **Reloc3R verification** (lightweight alternative): DUSt3R-variant submap registration via top-K keyframe pairs. Consistency verification on rotation std / translation direction std. `init_guess_norm` scale normalization from odometry chain. `raw_vs_init_dot` filter rejects edges where Reloc3R's raw translation direction contradicts odometry.
3. **2DGS-GSReg verification**: LoopSplat-style render-based multi-view bidirectional registration. Top-k viewpoint selection, render loss optimization, weighted pose fusion.
4. **ICP verification** (fallback): Point-to-plane ICP using FDN normals from 2DGS rotation quaternions. Consistency check on delta translation/rotation.
5. **PGO**: Builds pose graph from odometry edges (`relative_pose`) and loop edges (Reloc3R, GSReg, or ICP). Open3D `GlobalOptimizationLevenbergMarquardt`. Incremental gating avoids re-optimization when no new loop edges appear. Requires ≥3 valid loop edges. PGO safety valve rejects PGO when any submap correction exceeds `max_correction_t`.
6. **Correction**: Writes `correct_tsfm` to subgraph checkpoints (only on PGO success).

Odometry edges pass through validation gates (reject degenerate near-identity edges and implausible large jumps). Loop edges pass method-specific PGO thresholds on fitness, RMSE, delta_t, and delta_r. Reloc3R edges additionally pass `min_raw_vs_init_dot_for_pgo` and `max_loop_delta_t_for_pgo_reloc3r` filters.

### 7. Streaming global fusion

After front end finishes, the main process:
1. Stops the back end (saves final submap)
2. Stops loop closure (finalizes PGO)
3. Streams submap checkpoints from disk
4. Applies `correct_tsfm` via `rigid_transform_2dgs`
5. Concatenates all submap Gaussians into a single global model
6. Evaluates ATE and rendering quality
7. Optionally saves final PLY

### 8. Cross-Submap Covisibility Handoff

To mitigate tracking degradation after submap cuts (caused by sudden loss of all old Gaussians), the system supports boundary handoff:

1. **Selection**: At submap cut, the seed frame and old submap tail keyframes are rendered against the old Gaussian map. Boundary Gaussians visible from both the seed frame and tail keyframes are selected by support count and opacity score.
2. **Frozen container**: Selected Gaussians are exported via `capture_masked()` and stored as a frozen `GaussianModel` (no optimizer, no training).
3. **Tracking support**: During warmup, the front end creates a merged render model (`create_merged_for_render`) combining active new Gaussians with frozen handoff Gaussians, providing dense photometric constraints for tracking.
4. **Active-only insertion**: Error masks for new Gaussian insertion use active-only render (`self.gaussians`), preventing handoff from masking coverage holes.
5. **Auto-drop**: Handoff is removed when the new submap reaches `handoff_warmup_keyframes` keyframes, `handoff_warmup_frames` frames, or active opacity coverage exceeds `handoff_new_coverage_th`.
6. **Ckpt isolation**: Handoff Gaussians are never saved to new submap checkpoints and are cleared on handoff deactivation.

Configuration (all under `Submap`):
```yaml
use_handoff: false          # master switch
handoff_tail_kfs: 4         # old submap tail keyframes for covisibility
handoff_max_points: 3000    # max boundary Gaussians to retain
handoff_min_support: 2      # min keyframes that must observe a Gaussian
handoff_opacity_min: 0.20   # min opacity threshold
handoff_warmup_frames: 20   # max frames before forced drop
handoff_warmup_keyframes: 3 # max keyframes before forced drop
handoff_new_coverage_th: 0.85  # active coverage threshold for early drop
```

---

## Repository Structure

```text
FVO-GS-SLAM
├── slam.py                         # main entry, process orchestration, streaming fusion, evaluation
├── run_ablation.py                 # ablation experiment runner
├── run_all_slam.sh                 # batch running script
├── configs/
│   └── rgbd/
│       ├── tum/                    # TUM RGBD: base_config.yaml + scene overrides
│       ├── replica/                # Replica: base_config.yaml + scene overrides
│       └── scannetpp/              # ScanNet++: base_config.yaml + scene overrides
├── gaussian_splatting/
│   ├── gaussian_renderer/          # differentiable 2DGS surfel rendering
│   └── scene/gaussian_model.py     # Gaussian params, densify, prune, optimizer state
├── gui/                            # optional GUI visualization (OpenGL)
├── scripts/                        # dataset download scripts
├── tools/                          # debugging and testing tools
├── tests/                          # unit tests
├── weights/                        # CosPlace model weights
├── utils/
│   ├── slam_frontend.py            # tracking, keyframes, submap decisions, queue comms
│   ├── slam_backend.py             # Gaussian mapping, densify/prune, submap save
│   ├── fft_edge_vo.py              # FFT Edge VO: dense DT alignment + LM optimization
│   ├── fft_filter.py               # FFT high-frequency mask generation
│   ├── loop_closure.py             # CosPlace, Reloc3R/GSReg/ICP verification, PGO correction
│   ├── reloc3r_adapter.py          # Reloc3R top-K pair registration + consistency verification
│   ├── registration_2dgs.py        # normal-aware ICP for 2DGS submap registration
│   ├── gsr_2dgs/                   # LoopSplat-style 2DGS render-based registration
│   │   ├── solver_2dgs.py          # main registration entry
│   │   ├── gaussian_io_2dgs.py     # load 2DGS submap ckpt
│   │   ├── overlap_2dgs.py         # full Gaussian overlap with normal filter
│   │   ├── viewpoint_localizer_2dgs.py  # render-based viewpoint localization
│   │   └── pose_fusion_2dgs.py     # weighted pose fusion from multiple candidates
│   ├── tracking/                   # (future) HybridTracking: EdgeVO + RGB-D ICP
│   ├── slam_utils.py               # tracking/mapping loss functions
│   ├── pose_utils.py               # SE(3) pose update utilities
│   ├── camera_utils.py             # Camera class (viewpoint.T = W2C)
│   ├── dataset.py                  # dataset loading (TUM, Replica, ScanNet++, Realsense)
│   ├── eval_utils.py               # ATE and rendering evaluation
│   ├── normal_utils.py             # normal computation utilities
│   ├── point_utils.py              # point cloud utilities
│   ├── logging_utils.py            # logging
│   ├── config_utils.py             # YAML config loading
│   └── multiprocessing_utils.py    # FakeQueue for single-thread mode
└── submodules/                     # diff-surfel-rasterization, simple-knn
```

---

## System Architecture

```text
RGBD sequence
    ↓
Dataset loader → Camera objects
    ↓
FrontEnd (main process)
    ├── FFT mask generation (fft_filter.py)
    ├── FFT Edge VO initial pose (fft_edge_vo.py)
    │     cur→ref DT alignment + LM optimization
    ├── Render-based pose refinement
    │     Adam on cam_rot_delta / cam_trans_delta
    │     RGB L1 + DSSIM + depth L1 tracking loss
    ├── Keyframe decision + sliding window
    ├── Motion monitoring → submap cut trigger
    └── Auto-refresh FFTEdgeVO reference
    ↓ queue messages
BackEnd (independent process)
    ├── Seed frame init → Gaussian map initialization
    ├── Keyframe → extend Gaussian + mapping
    ├── RGB + depth + normal (FDN) loss
    ├── Densify / prune / opacity reset
    ├── occ_aware_visibility + pose sanity check
    ├── RSKM: randomly sampled keyframe supervision
    ├── Push Gaussian snapshot + Handoff → FrontEnd
    ├── Cross-submap boundary Handoff selection (seed + tail-kf covisibility)
    ├── Frozen Handoff: short-term read-only tracking support
    └── Save submap ckpt + notify loop closure
    ↓ saved submap checkpoints
LoopClosureProcess (independent process)
    ├── Extract CosPlace descriptors from keyframe images
    ├── Extract point clouds from 2DGS ckpt (FDN normals)
    ├── Detect loop candidates (cosine similarity + adaptive threshold)
    ├── Verify: Reloc3R / 2DGS-GSReg / ICP
    ├── Build + optimize pose graph (Open3D GlobalOptimization)
    │     Odometry edges from relative_pose
    │     Loop edges from Reloc3R/GSReg/ICP verification
    │     Incremental gating: only re-optimize with new edges
    │     PGO safety valve: reject if correction exceeds threshold
    └── Write correct_tsfm to submap checkpoints
    ↓
Main process after tracking
    ├── Stop backend + loop closure
    ├── Stream submap checkpoints from disk
    ├── Apply correct_tsfm via rigid_transform_2dgs
    ├── Concatenate global Gaussian model
    ├── Correct camera trajectory
    ├── Evaluate ATE + rendering quality
    └── Optional: save PLY, offline color refinement
```

---

## Module Roles and Data Flow

### `slam.py` — Main Entry

Main entry and system controller. Creates Gaussian model, dataset, front end, back end, optional GUI, and optional loop closure process. Handles evaluation mode overrides, W&B logging, GPU memory monitoring, streaming submap loading and fusion, trajectory correction, rendering evaluation, and final model saving.

### `utils/slam_frontend.py` — Front End

The front end is the main process (online tracking and scheduling).

Key responsibilities:
- Construct per-frame `Camera` objects (viewpoint.T = global W2C)
- Generate FFT masks for keyframes
- Run FFTEdgeVO for initial pose estimation
- Refine pose via render-based differentiable optimization
- Insert keyframes based on overlap ratio and translation
- Manage sliding window and visibility synchronization
- Compute motion relative to submap anchor
- Trigger submap cut on motion threshold
- Send `init`, `keyframe`, `new_submap`, `pause`, `stop` to back end
- Receive Gaussian snapshots and visibility from back end

### `utils/fft_edge_vo.py` — FFT Edge VO

Dense visual odometry, Edge VO (EAGS-SLAM) style.

**Direction**: cur→ref — project current-frame 3D points into reference-frame DT pyramid.

**Pipeline**:
1. `set_reference(image, depth, c2w)`: FFT mask → DT (full res) → pyramid → Sobel(DT) → pre-multiply fx/fy → store gradient structure pyramid
2. `track(image, depth, init_c2w)`: FFT mask → backproject 3D → coarse-to-fine LM with analytic SE(3) Jacobian → final C2W
3. `_lm_optimise()`: damped Gauss-Newton, DT gradient lookup, analytic Jacobian (Kerl 2012)

Returns `(success, est_c2w, info_dict)` with `dt_mean`, `visible`, `iters`.

### `utils/fft_filter.py` — FFT Mask

Builds a high-frequency mask from RGB: CLAHE → FFT → Gaussian HPF → IFFT → triangle threshold → bool mask. Used by Gaussian sampling (controls initial scale) and FFTEdgeVO feature selection.

### `utils/slam_backend.py` — Back End

Asynchronous local mapping (independent process).

Key responsibilities:
- Receive `init`, `keyframe`, `new_submap` messages from front end
- Initialize Gaussian map from seed keyframe
- Add new keyframes with FFT mask + error mask guided point insertion
- Optimize Gaussian parameters with RSKM-sampled keyframe supervision
- Collect visibility and densification statistics
- Densify, reset opacity, and prune Gaussian points
- Select boundary handoff Gaussians for submap transitions (frozen, read-only)
- Maintain `occ_aware_visibility` keyed by keyframe index
- Push Gaussian snapshots, keyframe poses, and handoff to front end
- Save submap checkpoints on `new_submap` and `stop`
- Notify loop closure on submap save
- Prune ALL Gaussian points and reset state for independent submap init

### `utils/loop_closure.py` — Loop Closure

Submap-level global consistency (independent process).

Key responsibilities:
- Implement CosPlace visual retrieval network (ResNet18 + GeM pooling)
- Maintain submap checkpoint records and point cloud caches
- Extract image descriptors from saved keyframe images
- Detect loop candidates (cosine similarity + adaptive threshold)
- Verify: Reloc3R / 2DGS-GSReg / ICP with FDN normals
- Reject unreliable loop edges (fitness, RMSE, delta_t, delta_r, raw_vs_init_dot)
- Build pose graph with odometry edges + loop edges
- Open3D GlobalOptimizationLevenbergMarquardt PGO
- Incremental PGO gating + PGO safety valve (correction magnitude guard)
- Write `correct_tsfm` to submap checkpoints (PGO success only)
- Apply rigid transforms to 2DGS params via `rigid_transform_2dgs`

### `utils/reloc3r_adapter.py` — Reloc3R Submap Registration

Lightweight submap registration using Reloc3R (DUSt3R variant).

Key responsibilities:
- Load Reloc3R model once, reuse across all submap pairs
- Build top-K keyframe pairs via CosPlace cross-similarity
- Run Reloc3R inference on each pair
- Convert per-pair camera-space transform to submap-level `T_src_to_tgt`
- `init_guess_norm`: scale Reloc3R's translation to metric using odometry chain
- Consistency verification: rotation std, translation direction std across pairs
- Track `raw_vs_init_dot` (direction alignment between Reloc3R raw output and odometry)
- Return metrics: `num_valid_pairs`, `rot_std_deg`, `min_raw_vs_init_dot`, `delta_t`, `delta_r`

### `utils/gsr_2dgs/` — 2DGS GSReg

LoopSplat-style render-based 2DGS submap registration.

| File | Role |
|---|---|
| `solver_2dgs.py` | Main entry: load ckpts, compute overlap, localize viewpoints, fuse poses |
| `gaussian_io_2dgs.py` | Load full 2DGS Gaussian params from ckpt (no downsample) |
| `overlap_2dgs.py` | Symmetric full-Gaussian overlap with optional normal-angle filter |
| `viewpoint_localizer_2dgs.py` | Render-based viewpoint localization: optimize pose delta only |
| `pose_fusion_2dgs.py` | Weighted pose fusion from multiple viewpoint candidates |

### `utils/registration_2dgs.py` — Normal-aware ICP

Normal-aware ICP registration for 2DGS submap pairs. Extracts normals from ckpt `_normal` (FDN) or derives from `_rotation` quaternion (local z-axis). Used as a utility in loop closure.

### `gaussian_splatting/scene/gaussian_model.py` — Gaussian Model

Stores and updates 2DGS Gaussian map parameters: `_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation`, `_normal`. Key methods: `extend_from_pcd_seq()`, `densify_and_prune()`, `prune_points()`, `capture_dict()`, `training_setup()`.

### `gaussian_splatting/gaussian_renderer/__init__.py` — Renderer

Differentiable 2DGS surfel renderer. Outputs: `render` (RGB), `depth`, `opacity`, `rend_normal`, `surf_normal`, `viewspace_points`, `visibility_filter`, `radii`, `n_touched`.

---

## Main Queue Messages

Front end → back end (multiprocessing Queue):

```text
["init", cur_frame_idx, viewpoint, depth_map]
["keyframe", cur_frame_idx, viewpoint, current_window, depth_map]
["new_submap", completed_submap_id, relative_pose, new_seed_global_c2w]
["pause"]
["unpause"]
["stop"]
```

Back end → front end:

```text
["init", gaussians, occ_aware_visibility, keyframes, handoff_data]
["keyframe", gaussians, occ_aware_visibility, keyframes, handoff_data]
["sync_backend", gaussians, occ_aware_visibility, keyframes, handoff_data]
```

`handoff_data` is `(frozen_gaussian_model, age_frames, warmup_frames)` or `None`.

Back end → loop closure:

```text
["submap_saved", submap_id, ckpt_path, kf_image_paths]
["stop"]
```

Do not change these message formats unless every sender and receiver is updated together.

---

## Pose Conventions

| Variable | Semantic | Source |
|---|---|---|
| `viewpoint.T` | global **W2C** (4×4) | FrontEnd tracking writes |
| `torch.linalg.inv(viewpoint.T)` | global **C2W** (4×4) | Computed as needed |
| `seed_global_c2w` | submap seed global C2W | Set at submap cut |
| `relative_pose` | prev_seed → curr_seed (4×4) | Computed at submap cut |
| `correct_tsfm` | PGO/loop correction (4×4) | Only loop closure writes |
| `cam_rot_delta` / `cam_trans_delta` | SE(3) delta for render refinement | Per-frame Adam optimizer |

**Do not change these conventions.**

---

## Configuration Overview

Important configuration groups:

| Group | Controls |
|---|---|
| `Results` | save path, trajectory saving, GUI, rendering eval, W&B |
| `Dataset` | dataset type, sensor type, camera params, point sampling |
| `Training` | init/mapping/tracking iters, keyframe, window, LR, densify/prune, RSKM |
| `FFTEdgeVO` | Edge VO pyramid, optimization, quality thresholds |
| `Backend` | keyframe pose policy, pose sanity check |
| `Submap` | motion thresholds (TUM: 2.0m/80°), seed init, handoff |
| `LoopClosure` | registration method, GSReg params, ICP params, PGO protection |
| `opt_params` | Gaussian optimizer and densification params |
| `model_params` | SH degree, data device |
| `pipeline_params` | renderer settings |
| `Ablation` | submap, loop closure, FDN, FFT mask, error mask, color refinement |

Three base configs must be kept in sync for generic parameters:

```text
configs/rgbd/tum/base_config.yaml
configs/rgbd/replica/base_config.yaml
configs/rgbd/scannetpp/base_config.yaml
```

Scene-specific configs should only override existing parameters.

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

Depending on your CUDA and PyTorch versions, you may need to adjust versions in `environment.yml` or install PyTorch manually.

---

## Download Datasets

```bash
bash scripts/download_tum.sh      # TUM RGBD
bash scripts/download_replica.sh  # Replica
bash scripts/download_euroc.sh    # EuRoC MAV (legacy)
```

---

## Run

### TUM RGBD

```bash
python slam.py --config configs/rgbd/tum/fr1_desk.yaml --eval
python slam.py --config configs/rgbd/tum/fr3_office.yaml --eval
```

### Replica

```bash
python slam.py --config configs/rgbd/replica/office0.yaml --eval
python slam.py --config configs/rgbd/replica/office0_sp.yaml --eval  # single process
```

### ScanNet++

```bash
python slam.py --config configs/rgbd/scannetpp/8b5caf3398.yaml --eval
```

---

## Evaluation

Use `--eval` to force evaluation mode, which overrides:

```text
save_results = True
use_gui = False
eval_rendering = True
use_wandb = False
```

Output directory contains:

```text
config.yml
frame_to_submap.pt
submaps/*.ckpt
submaps/*_img_*.pt   (keyframe images for CosPlace)
point_cloud/final/point_cloud.ply
rendering evaluation outputs
trajectory and ATE outputs
```

Evaluation logs include FPS, ATE, rendering metrics, GPU memory peak (minus baseline), and final map size.

---

## Ablation Switches

```yaml
Ablation:
  use_submap: True            # submap cutting (off → global single map)
  use_loop_closure: True      # loop detection + PGO correction
  use_fdn: True               # finite-difference normal supervision
  use_fft_mask: True          # FFT frequency mask for sampling + scale
  use_error_mask: True        # render error mask for dynamic point insertion
  use_color_refinement: False # offline color refinement after merge

FFTEdgeVO:
  use_fft_edge_vo: true       # enable FFT Edge VO initial pose

Backend:
  optimize_keyframe_pose: true        # keyframe pose optimization in back end
  optimize_keyframe_exposure: false   # keyframe exposure optimization

LoopClosure:
  registration_method: "2dgs_gsreg"  # "reloc3r", "2dgs_gsreg", "icp", or "reloc3r_mock"
  debug_disable_pgo_for_fftvo_test: false  # PGO enabled by default (set true for FFTVO ablation)
  pgo_safety:                           # PGO safety valve
    enabled: true
    max_correction_t: 2.0
  min_raw_vs_init_dot_for_pgo: 0.7     # Reloc3R direction alignment filter
  max_loop_delta_t_for_pgo_reloc3r: 2.0  # Reloc3R delta_t filter
```

For debugging FFTEdgeVO alone, disable PGO with `debug_disable_pgo_for_fftvo_test: true` — evaluates whether FFTEdgeVO improves tracking before global corrections are introduced.

---

## Reproducibility Notes

- The system is sensitive to CUDA, PyTorch, Open3D, and differentiable Gaussian rasterizer versions.
- Recommended workflow: run a short smoke test first, then full evaluation.
- For fair comparison, keep the same dataset sequence, image resolution, tracking/mapping iterations, submap thresholds, and ablation switches.
- When analyzing submap experiments, report both ATE and rendering metrics before/after global fusion or offline color refinement.
- When debugging FFTEdgeVO, log `dt_mean`, `visible`, `iters`, tracking source, and convergence behavior.

---

## Acknowledgement

This project is developed based on the MonoGS style Gaussian SLAM framework. The repository adapts it toward RGBD 2D Gaussian SLAM with FFT Edge VO initialization, motion-based submaps, CosPlace + 2DGS-GSReg/ICP loop closure, PGO correction, and streaming global fusion.

---

## License

This modified version follows the license terms inherited from the original project. Please refer to `LICENSE.md` for details.
