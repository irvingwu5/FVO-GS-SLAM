[comment]: <> (#SubGS-SLAM)

<!-- PROJECT LOGO -->

# SubGS-SLAM: A Submap based 2D Gaussian Splatting SLAM System with RGBD Guided Local Mapping and Pose Graph Optimization

SubGS-SLAM is an RGBD SLAM system based on 2D Gaussian Splatting and differentiable surfel rendering. The system is built upon [MonoGS](https://github.com/muskie82/MonoGS) and extends the original Gaussian based SLAM framework with submap management, RGB depth normal joint optimization, frequency and rendering error guided Gaussian expansion, optional loop closure, adjacent odometry constraints, chain PGO, global PGO, and ablation controls.

This repository focuses on RGBD indoor SLAM experiments on datasets such as TUM RGBD and Replica. Monocular and stereo configurations inherited from the original project may still exist in the repository, but the current maintained development direction is RGBD based 2D Gaussian SLAM.

<p align="center">
  <a href="">
    <img src="./media/pipeline.png" alt="pipeline" width="100%">
  </a>
</p>

---

# Statement

This repository is developed based on [MonoGS](https://github.com/muskie82/MonoGS). We retain the core frontend backend Gaussian SLAM logic of the original project and further extend it into a submap aware RGBD 2D Gaussian SLAM system.

The current system includes:

- RGBD input and camera model handling
- frontend camera tracking
- keyframe selection and sliding window management
- backend local mapping
- 2D Gaussian map representation
- differentiable surfel rendering
- RGB depth normal joint supervision
- finite difference normal supervision
- FFT frequency mask guided Gaussian sampling
- rendering error mask guided Gaussian expansion
- Gaussian densification and pruning
- visibility maintenance with `occ_aware_visibility`
- submap cutting and independent submap initialization
- submap freezing, checkpoint saving, and streaming global fusion
- optional CosPlace based submap loop candidate retrieval
- ICP based geometric verification
- AdjacentOdom constraints between neighboring submaps
- Chain PGO without reliable loop closure
- Global PGO with verified loop constraints
- PGO correction transform propagation
- offline color refinement
- trajectory and rendering quality evaluation
- optional GUI visualization
- ablation switches for controlled experiments
- memory management during submap switching and fusion

---

# Main Differences from MonoGS

Compared with the original MonoGS framework, this repository adds or modifies the following components:

## 1. RGBD Oriented 2D Gaussian SLAM Pipeline

The system is organized around RGBD input, differentiable 2D Gaussian or surfel rendering, and online pose tracking plus backend mapping. Rendering outputs such as RGB, depth, opacity, visibility, render normal, and surf normal are used for tracking, mapping, pruning, and evaluation.

## 2. Frontend Tracking and Keyframe Management

The frontend estimates the current camera pose by optimizing the rendered observation against the input RGBD frame. It also maintains keyframes, a local sliding window, visibility information, and communication with the backend.

## 3. Backend Mapping and Gaussian Optimization

The backend initializes and updates the Gaussian map from RGBD observations. It performs local mapping, Gaussian densification, Gaussian pruning, opacity reset, normal supervision, and local pose optimization.

## 4. Submap Based SLAM

The system supports submap cutting, independent submap initialization, submap freezing, submap checkpoint saving, and offline global fusion. Each submap can be saved with its Gaussian parameters, keyframe information, seed pose, relative pose, and correction transform.

## 5. Loop Closure and Pose Graph Optimization

The system optionally supports submap level loop closure. It distinguishes among three different pose graph related mechanisms:

- **AdjacentOdom**: odometry like constraints between neighboring submaps.
- **Chain PGO**: chain based pose graph optimization when no reliable loop closure is available.
- **Global PGO**: global pose graph optimization when verified loop closure constraints exist.

These mechanisms are designed to improve submap consistency and reduce accumulated drift.

## 6. Gaussian Sampling and Ablation Modules

The system includes optional modules for FFT frequency mask sampling, rendering error mask guided Gaussian insertion, finite difference normal supervision, and color refinement. These modules can be enabled or disabled through configuration files for ablation experiments.

---

# System Overview

The overall workflow is:

```text
RGBD sequence
    ↓
camera and dataset loader
    ↓
frontend tracking
    ↓
keyframe selection and sliding window update
    ↓
backend Gaussian mapping
    ↓
Gaussian densification, pruning, and local optimization
    ↓
submap cutting and independent submap initialization
    ↓
submap checkpoint saving
    ↓
optional loop candidate retrieval and ICP verification
    ↓
AdjacentOdom / Chain PGO / Global PGO
    ↓
streaming global Gaussian fusion
    ↓
trajectory evaluation and rendering evaluation
```

# Getting Started

## Installation

```
git clone https://github.com/irvingwu5/SubGS-SLAM.git --recursive
```

Setup the environment.
```
conda env create -f environment.yml
conda activate your_env_name
```
Based on your specific hardware and software setup, please modify the dependency versions for pytorch/cudatoolkit in the `environment.yml` file, following the instructions provided in [this PyTorch documentation](https://pytorch.org/get-started/previous-versions/).


## Downloading Datasets
When you run the following scripts, datasets will be downloaded automatically to the local `./datasets` directory.
### TUM-RGBD dataset
```bash
bash scripts/download_tum.sh
```

### Replica dataset
```bash
bash scripts/download_replica.sh
```

### EuRoC MAV dataset
```bash
bash scripts/download_euroc.sh
```



## Run
### Monocular
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

### RGB-D
```bash
python slam.py --config configs/rgbd/tum/fr3_office.yaml
```

```bash
python slam.py --config configs/rgbd/replica/office0.yaml
```
Or the single process version as
```bash
python slam.py --config configs/rgbd/replica/office0_sp.yaml
```


### Stereo (experimental)
```bash
python slam.py --config configs/stereo/euroc/mh02.yaml
```

# Evaluation
<!-- To evaluate the method, please run the SLAM system with `save_results=True` in the base config file. This setting automatically outputs evaluation metrics in wandb and exports log files locally in save_dir. For benchmarking purposes, it is recommended to disable the GUI by setting `use_gui=False` in order to maximise GPU utilisation. For evaluating rendering quality, please set the `eval_rendering=True` flag in the configuration file. -->
To evaluate our method, please add `--eval` to the command line argument:
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
```
This flag will automatically run our system in a headless mode, and log the results including the rendering metrics.

# Reproducibility
All experiments were conducted on an RTX 3090 graphics card. Performance discrepancies may arise when utilizing alternative GPU hardware configurations.

# Acknowledgement
Thanks to the original MonoGS project by muskie82, which provided a solid foundation for this work.

# License
The original MonoGS project is released under the license agreement specified in **LICENSE.md**. This modified version inherits the license agreement of the original project and does not alter the core license terms of the original project.

# Additional Notes
- Original Project Author: muskie82
- Modified Version Maintainer: [wxy]
- Modification Date: [2026-4-25]