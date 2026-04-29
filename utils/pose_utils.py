import torch
import lietorch


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)
    T_w2c = camera.T

    # 【修复】：严格遵循 CUDA 梯度假设，直接在 W2C 矩阵上左乘扰动
    new_w2c = lietorch.SE3.exp(tau).matrix() @ T_w2c

    converged = (tau ** 2).sum() < (converged_threshold ** 2)
    camera.T = new_w2c

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged
