import queue

import cv2
import numpy as np
import open3d as o3d
import torch

from gaussian_splatting.utils.general_utils import (
    build_scaling_rotation,
    strip_symmetric,
)

cv_gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],
                [1.0, -0.5, 2],
                [-1.0, -0.5, 2],
                [1.0, 0.5, 2],
                [-1.0, 0.5, 2],
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


class GaussianPacket:
    def __init__(
        self,
        gaussians=None,
        keyframe=None,
        current_frame=None,
        gtcolor=None,
        gtdepth=None,
        gtnormal=None,
        keyframes=None,
        finish=False,
        kf_window=None,
    ):
        self.has_gaussians = False
        if gaussians is not None:
            self.has_gaussians = True
            self.get_xyz = gaussians.get_xyz.detach().clone()
            self.active_sh_degree = gaussians.active_sh_degree
            self.get_opacity = gaussians.get_opacity.detach().clone()
            self.get_scaling = gaussians.get_scaling.detach().clone()
            self.get_rotation = gaussians.get_rotation.detach().clone()
            self.max_sh_degree = gaussians.max_sh_degree
            self.get_features = gaussians.get_features.detach().clone()

            self._rotation = gaussians._rotation.detach().clone()
            self.rotation_activation = torch.nn.functional.normalize
            self.unique_kfIDs = gaussians.unique_kfIDs.clone()
            self.n_obs = gaussians.n_obs.clone()

        self.keyframe = keyframe
        self.current_frame = current_frame
        self.gtcolor = self.resize_img(gtcolor, 320)
        self.gtdepth = self.resize_img(gtdepth, 320)
        self.gtnormal = self.resize_img(gtnormal, 320)
        self.keyframes = keyframes
        self.finish = finish
        self.kf_window = kf_window

    def resize_img(self, img, width):
        if img is None:
            return None

        # check if img is numpy
        if isinstance(img, np.ndarray):
            height = int(width * img.shape[0] / img.shape[1])
            return cv2.resize(img, (width, height))
        height = int(width * img.shape[1] / img.shape[2])
        # img is 3xHxW
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        return img.squeeze(0)

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    # def build_covariance_from_scaling_rotation(  # 缩放矩阵,用于调整缩放矩阵的大小,四元数
    #         self, center, scaling, scaling_modifier, rotation
    # ):
    #     RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1),
    #                                 rotation,
    #                                 ).permute(0, 2, 1)  # S,q->R，L=RS，scaling_modifier用于调整缩放矩阵的大小
    #     trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
    #     trans[:, :3, :3] = RS
    #     trans[:, 3, :3] = center
    #     trans[:, 3, 3] = 1
    #     return trans
    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm


def get_latest_queue(q):
    message = None
    while True:
        try:
            message_latest = q.get_nowait()
            if message is not None:
                del message
            message = message_latest
        except queue.Empty:
            if q.qsize() < 1:
                break
    return message


class Packet_vis2main:
    flag_pause = None


class ParamsGUI: # 定义了一个名为 ParamsGUI 的 Python 类，其主要作用是一个数据容器或配置对象，用于封装和传递 GUI（图形用户界面）所需的各种参数和共享对象。
    def __init__(
        self,
        pipe=None,
        background=None,
        gaussians=None,
        q_main2vis=None,
        q_vis2main=None,
    ):
        self.pipe = pipe
        self.background = background
        self.gaussians = gaussians #要在 GUI 中可视化的 3D 高斯数据
        self.q_main2vis = q_main2vis #从前端（主进程/线程）发送消息到可视化 (GUI) 进程/线程。这通常用于传输更新的数据包
        self.q_vis2main = q_vis2main #从可视化 (GUI) 进程/线程发送消息回主进程/线程（例如用户在 GUI 上的操作指令，如暂停、调整视角等）
