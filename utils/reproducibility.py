"""
Seed reproducibility utility for FVO-GS-SLAM.

Controls Python, NumPy, PyTorch, and cuDNN randomness to improve
experiment reproducibility across runs with the same seed.

NOTE: CUDA rasterizer atomic operations, multi-process scheduling,
and floating-point parallel accumulation may still cause minor
differences across runs.
"""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
