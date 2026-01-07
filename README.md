[comment]: <> (# Gaussian Splatting SLAM)

<!-- PROJECT LOGO -->
# Statement
This repository is built upon [MonoGS](https://github.com/muskie82/MonoGS) (the original project repository) as the baseline for improvements, while retaining the core logic of the original project. The main modifications are as follows:

## 📌 Modifications Based on MonoGS
1. [Core Modification 1]: Added embedding of planar semantic information
2. [Core Modification 2]: Newly added optimization strategies for planar semantic information

<p align="center">
  <a href="">
    <img src="./media/pipeline.png" alt="gui" width="100%">
  </a>
</p>

# Getting Started

## Installation
```
git clone https://github.com/cuit-aicg-team/wxy.git --recursive
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
All experiments were conducted on an RTX 4090 graphics card. Performance discrepancies may arise when utilizing alternative GPU hardware configurations.

# Acknowledgement
Thanks to the original MonoGS project by muskie82, which provided a solid foundation for this work.

# License
The original MonoGS project is released under the license agreement specified in **LICENSE.md**. This modified version inherits the license agreement of the original project and does not alter the core license terms of the original project.

# Additional Notes
- Original Project Author: muskie82
- Modified Version Maintainer: [cuit-aicg-team:wxy]
- Modification Date: [2026-01-05]














