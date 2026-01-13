#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0 # 当前球谐函数阶数
        self.max_sh_degree = sh_degree # 最大球谐函数阶数
        # 以下划线开头的变量表示这些变量是 GaussianModel 类的私有属性。它们用于存储高斯模型的内部状态，不应该被类的外部直接修改
        # 参与优化的高斯参数
        self._xyz = torch.empty(0, device="cuda") # 高斯位置(均值) (高斯数量,3)
        self._features_dc = torch.empty(0, device="cuda") # 第一个球谐系数，(x, 1, 3)(满足要求的高斯点数量,每个高斯的第一个球谐系数，RGB3个通道),每个高斯点的第一个球谐系数有 3 个通道，分别对应 RGB 颜色通道
        self._features_rest = torch.empty(0, device="cuda") # 其余球谐系数，(x,15,3)(,3阶除去0阶外有15个球谐系数,)
        self._scaling = torch.empty(0, device="cuda") # 高斯缩放值(xyz三个方向) (高斯数量,3)
        self._rotation = torch.empty(0, device="cuda") # 高斯旋转(四元数) (高斯数量,4)
        self._opacity = torch.empty(0, device="cuda") # 高斯不透明度  (高斯数量,1)
        # 不参与优化的辅助变量
        self._pxl_plane_id = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda") # 高斯投影后的最大半径(通过计算2D协方差矩阵的特征值，取其最大值的平方根，再乘以3并向上取整得到的) (高斯数量,)
        self.xyz_gradient_accum = torch.empty(0, device="cuda") # 高斯位置(均值)的累积梯度 (高斯数量,1)
        #这两个变量用于追踪高斯点的来源（哪个关键帧）和稳定性（被观测了多少次），主要用于增量式建图场景。
        self.unique_kfIDs = torch.empty(0).int() # 用于记录每个高斯点是由哪一帧图像生成的，有助于后续管理地图（例如剔除旧的关键帧对应的点）
        # 这是一个置信度指标。在 SLAM 系统中，如果一个点被多次观测到，说明它比较稳定；
        # 如果观测次数很少，可能是一个噪声点，可以在后续优化或剪枝步骤中被移除。
        self.n_obs = torch.empty(0).int()

        self.optimizer = None # 优化器
        # 激活函数的目的是对模型中的不同参数进行非线性变换，以确保它们在训练过程中保持在合理的范围内，并且具有适当的数值特性
        self.scaling_activation = torch.exp # 对缩放值进行指数变换，确保缩放值始终为正
        self.scaling_inverse_activation = torch.log # 对缩放值进行对数变换，通常用于将缩放值还原到其原始范围。

        self.covariance_activation = self.build_covariance_from_scaling_rotation # 通过缩放和旋转矩阵构建协方差矩阵，用于表示高斯分布的形状和方向

        self.opacity_activation = torch.sigmoid # 对不透明度进行 Sigmoid 变换，将其值限制在 0 到 1 之间
        self.inverse_opacity_activation = inverse_sigmoid # 对不透明度进行逆sigmoid变换，将不透明度值还原到原始值

        self.rotation_activation = torch.nn.functional.normalize # 对旋转四元数进行归一化，确保其表示有效的旋转

        self.config = config #dict{dict{}、dict{}、...}
        self.ply_input = None

        self.isotropic = False

    def build_covariance_from_scaling_rotation( # 缩放矩阵,用于调整缩放矩阵的大小,四元数
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation) # S,q->R，L=RS，scaling_modifier用于调整缩放矩阵的大小
        actual_covariance = L @ L.transpose(1, 2) # sigma=L*L^T=(RS)(RS)^T=RSS^TR^T
        symm = strip_symmetric(actual_covariance) # 协方差矩阵sigma属于对称矩阵，只需存储一半即可
        return symm

    # @property 装饰器将 get_scaling 方法伪装成属性，外部只能通过该接口获取缩放值（model.get_scaling）
    # 而无法直接访问 _scaling# @property 装饰器将 get_scaling 方法伪装成属性，
    # 外部只能通过该接口获取缩放值（model.get_scaling），而无法直接访问 _scaling
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) # 对缩放值进行指数变换，确保缩放值始终为正

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) # 对旋转四元数进行归一化，确保其表示有效的旋转

    @property
    def get_xyz(self):
        return self._xyz # 高斯位置(均值) (高斯数量,3)

    @property
    def get_features(self):
        features_dc = self._features_dc # 第一个球谐系数
        features_rest = self._features_rest # 后15个球谐系数
        return torch.cat((features_dc, features_rest), dim=1) # 3阶所有16个球谐系数

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity) # 不透明度

    @property
    def get_px_plane_id(self):
        return self._pxl_plane_id

    def get_covariance(self, scaling_modifier=1): # 根据缩放矩阵和四元数获取协方差矩阵
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree: # 如果当前球谐阶数小于最大球谐阶数
            self.active_sh_degree += 1 # 阶数+1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda() # 将点云转为tensor并送入GPU
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda()) # 将点云的RGB颜色转换为球谐系数
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )# 创建一个全零的三维张量，用于存储球谐系数特征。张量的形状为 (点云颜色数量, 3, (最大球谐函数阶数 + 1) ** 2)
        features[:, :3, 0] = fused_color # 将点云的 RGB 颜色值赋值给 features 张量的第一个球谐系数，将 features 张量的第一个维度的所有元素，第二个维度的前三个元素，最后一个维度的所有元素设置为 0.0
        features[:, 3:, 1:] = 0.0 # 将 features 张量的第一个维度的所有元素:，第二个维度的第三个元素之后3:的所有元素，第三个维度的第二个元素1:之后的所有元素设置为 0.0
        # 调用 distCUDA2 函数计算点之间的距离平方。torch.clamp_min 函数确保距离的最小值为 0.0000001，避免出现零距离
        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None] # 先对距离平方取平方根得到距离，然后取对数得到缩放值
        # Isotropic (各向同性)：意味着高斯体在 X, Y, Z 所有方向上的性质（缩放比例）是相同的。在几何上，这代表一个标准的 球体 (Sphere)。
        # Anisotropic (各向异性)：意味着不同方向的缩放比例可以不同。在几何上，这代表一个 椭球体 (Ellipsoid)。
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda") # 存储每个点的四元数表示
        rots[:, 0] = 1 # 将每个四元数的第一个元素设置为 1，其他元素保持为 0。这相当于将所有四元数初始化为单位四元数 [1, 0, 0, 0]，表示没有旋转
        # 生成的张量的每个元素都为0.1经过逆sigmoid变换得到真实不透明度 [0,1]->inv_sig->[-inf,inf]
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale # 高斯位置(均值)学习率缩放, mu_new = mu_old - lr*partial{loss}/partial{mu}

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        # 将新的高斯点属性（位置、球谐系数、不透明度、缩放值、旋转等）添加到现有的高斯点集合中，并更新优化器的参数
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        ) #基于相机信息生成点云相关数据并解包
        # 将这些数据传给另一个方法进行模型扩展
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        # 在高斯点的密化过程中，该参数决定了高斯点的缩放值阈值。具体来说，它用于判断哪些高斯点需要进行克隆或分裂操作，以调整高斯点的密度
        self.percent_dense = training_args.percent_dense # 控制高斯点密度的参数，percent_dense稠密化比例*scene_extent场景大小(最近最远相机连线长度*1.1)# 稠密化比例*场景大小作为高斯缩放阈值
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 高斯位置(均值)的累积梯度 (高斯数量,1)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda") # 高斯梯度更新次数 (高斯数量,1)# 高斯位置梯度更新次数，用于求平均位置梯度
        # 定义高斯各个优化参数的学习率，new = old - lr * gradient，lr可以动态变化(乘衰减因子)也可以为固定值
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # get_expon_lr_func 函数生成一个学习率调度函数，该函数根据训练参数计算位置参数的学习率
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    # 根据当前的迭代次数动态调整学习率，以便在训练过程中更好地优化模型
    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 通过剪枝操作，移除不需要的高斯点，并更新剩余高斯点的相关属性，原mask中大于梯度阈值的点被标记为True(要剔除的)
    # 小于阈值的点被标记为False(需要的)，新生成的高斯点会被标记为 False(但是需要保留)
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]

    # 将新的张量（tensors_dict 中的张量）添加到现有的优化器参数中，并更新优化器的状态
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 将新的高斯点属性（位置、球谐系数、不透明度、缩放值、旋转等）添加到现有的高斯点集合中，并更新优化器的参数
    # 满足梯度幅度以及缩放的高斯：高斯位置(均值)、第一个球谐系数、其余15个球谐系数、
    # 不透明度、缩放值、旋转、高斯半径(通过计算2D协方差矩阵的特征值，取其最大值的平方根，再乘以3并向上取整得到的)
    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ): # 创建一个字典 d，包含新的高斯点属性，这里new_xyz(219,3) 新增分裂(484,3)
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        # 优化后的高斯点覆盖原高斯点，cat_tensors_to_optimizer 函数将新的高斯点属性添加到现有的优化器参数中，并返回更新后的张量（在原N的基础上加减）
        optimizable_tensors = self.cat_tensors_to_optimizer(d) # 将新的高斯点属性添加到优化器的参数中，并返回更新后的张量
        self._xyz = optimizable_tensors["xyz"] # 将返回的更新后的张量分别赋值给高斯各个属性,(116045,3)，分裂完后(116529,3)
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    # 每个高斯点的位置(均值)平均梯度,位置梯度阈值、场景大小(最近最远相机连线长度*1.1)、分裂个数N
    # 位置梯度幅值大于阈值触发增密和分裂
    ## 若高斯尺寸超过阈值extent表明过重建区域，需要执行分裂操作
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0] # 克隆后高斯点个数(115826)->(116045)随后进行分裂操作
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze() # grads(115826,1)->(115826,)，grads复制到padded_grad(116045)前115826，将grads张量的值复制到 padded_grad 的前 grads.shape[0] 个位置上，并将 grads 张量压缩成一维
        # 高斯尺度>尺度阈值表示过重建需要分裂高斯
        # 选择那些位置平均梯度>=阈值的点，并且这些点的最大缩放值>场景范围的百分比，布尔数组标记满足条件的高斯
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        # 构造三维正态分布是为了生成新的高斯点位置。这些新的高斯点用于克隆或分裂现有的高斯点，以实现高斯点的密化过程
        # 通过生成符合正态分布的样本，可以确保新生成的高斯点在空间上分布合理，并且与原有高斯点的分布特性一致
        # 这样可以在保持原有高斯点分布特性的基础上，增加高斯点的数量，从而提高模型的精度和表现
        # 为后续的高斯点密化过程准备缩放值。通过重复缩放值，可以生成多个新的高斯点，这些点将用于克隆或分裂现有的高斯点
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1) # (242,3)->(484,3)直接在242尾部复制了一份，选择满足条件的高斯点的缩放值，将选中的缩放值沿第一个维度重复N次(分裂个数N)，生成一个新的张量stds(xyz三个方向标准差)
        means = torch.zeros((stds.size(0), 3), device="cuda") # (484,3)高斯位置(零均值)
        samples = torch.normal(mean=means, std=stds) # 根据提供的均值和标准差生成一个与 means 和 stds 形状相同服从正态分布的张量 (x,3)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1) # 四元数转旋转矩阵，对生成的旋转矩阵进行重复操作。N 表示重复的次数，1,1 表示在后两个维度上不进行重复。这样可以生成一个形状为 (N * 选定高斯点数量, 3, 3) 的张量
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        # 将选中的缩放值沿第一个维度重复N次，将重复后的缩放值除以0.8*N，进行缩放调整，对调整后的缩放值应用逆激活函数，生成新的缩放值(实际缩放值)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1) # 旋转(484,4)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1) # 第一个球谐系数(x,1,3)(高斯数量,系数个数,RGB三通道)沿着第一个维度重复N次
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1) # 其余球谐系数
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1) # 不透明度(x,1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)
        # 将新的高斯点属性（位置、球谐系数、不透明度、缩放值、旋转等）添加到现有的高斯点集合中，并更新优化器的参数
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )
        # prune_filter 用于标记哪些高斯点需要被剪枝。原mask中大于梯度阈值的点被标记为True(要剔除的)
        # 小于阈值的点被标记为False(需要的)，新生成的高斯点会被标记为 False(但是需要保留)
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        # 在密化和分裂高斯点后，移除不需要的高斯点，并更新剩余高斯点的相关属性
        self.prune_points(prune_filter)

    # 每个3D高斯点的位置(均值)平均梯度、位置梯度阈值、场景大小(最近最远相机连线长度*1.1)
    # 高斯位置梯度幅值大于阈值触发增密和分裂
    ## 若高斯尺寸小于阈值extent表明欠重建区域，需要执行克隆操作
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        # 高斯尺度<尺度阈值表示欠重建需要克隆高斯
        # 选择那些位置梯度模长(幅度)>=阈值的点，并且这些点的最大缩放值<=场景范围的百分比，布尔数组标记满足条件的高斯
        # self.get_scaling（N,3）每个高斯3个缩放值，dim=1选出每个高斯点三个缩放值中的最大值，得到value和indices两个张量，max.values最终结果每个元素表示对应高斯点的最大缩放值
        # 将这些最大值与 self.percent_dense * scene_extent 进行比较，筛选出符合条件的高斯点
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        # 将满足梯度模长和尺度缩放值的高斯属性进行重新赋值,其余不满足的直接剔除
        new_xyz = self._xyz[selected_pts_mask] # 高斯位置(均值) new_xyz(215,3)
        new_features_dc = self._features_dc[selected_pts_mask] # 第一个球谐系数，(215, 1, 3)(满足要求的高斯点数量,每个高斯的第一个球谐系数，RGB3个通道),每个高斯点的第一个球谐系数有 3 个通道，分别对应 RGB 颜色通道
        new_features_rest = self._features_rest[selected_pts_mask] # 其余球谐系数，(215,15,3)(,3阶除去0阶外有15个球谐系数,)
        new_opacities = self._opacity[selected_pts_mask] # 高斯不透明度
        new_scaling = self._scaling[selected_pts_mask] # 高斯缩放值(xyz三个方向)
        new_rotation = self._rotation[selected_pts_mask] # 高斯旋转(四元数)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        # 将新的高斯点属性（位置、球谐系数、不透明度、缩放值、旋转等）添加到现有的高斯点集合中，并更新优化器的参数
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    # 位置梯度阈值、最小透明度、场景半径(最近最远相机中心连线长度*1.1)、2D高斯最大半径(通过计算2D协方差矩阵的特征值，取其最大值的平方根，再乘以3并向上取整得到的)、3D高斯半径
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom # (N,1)每个高斯点的累积位置(均值)梯度/每个高斯点被更新的次数，计算了每个高斯点的位置(均值)平均梯度。这样可以得到每个高斯点的累积梯度的平均值，用于后续的密化和剪枝操作
        grads[grads.isnan()] = 0.0 # 将梯度中所有 NaN 值替换为 0.0。在训练过程中，梯度可能会因为数值不稳定性或其他原因变成 NaN，这会影响模型的训练
        ## 若高斯尺寸小于阈值extent表明欠重建区域，需要执行克隆操作,高斯数量增多
        self.densify_and_clone(grads, max_grad, extent)
        ## 若高斯尺寸超过阈值extent表明过重建区域，需要执行分裂操作,高斯数量增多
        self.densify_and_split(grads, max_grad, extent)
        # 剪枝操作：高斯不透明度小于阈值或者高斯尺寸过大则移除高斯,高斯数量减少
        prune_mask = (self.get_opacity < min_opacity).squeeze() # (116287,1)->(116287,)将不透明度小于min_opacity的高斯点标记出来，生成一个布尔掩码prune_mask
        if max_screen_size: # 如果提供了max_screen_size参数(2D高斯最大半径阈值)，则进一步筛选需要剪枝的高斯点
            big_points_vs = self.max_radii2D > max_screen_size # 将2D高斯半径大于max_screen_size的高斯点标记出来，生成布尔掩码big_points_vs
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # 将3D高斯缩放值大于场景范围10%的高斯点标记出来，生成布尔掩码big_points_ws

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )# 进行逻辑或操作，更新剪枝掩码prune_mask，标记出所有需要剪枝的高斯点
        self.prune_points(prune_mask) # 移除那些不透明度小于min_opacity的高斯点，以及那些2D高斯半径大于max_screen_size或3D高斯缩放值大于场景范围10%的高斯点

    # 所有高斯点的张量、半径大于0的高斯点的索引
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        ) # 更新梯度累积：对于半径大于0的高斯点，计算其梯度L2范数，并将其累加到 xyz_gradient_accum 张量中
        self.denom[update_filter] += 1 # 更新计数器：对于满足 update_filter 条件的高斯点，将 denom 张量中的对应值加 1
        # 用于记录每个高斯点在累积梯度时的计数器。它的作用是统计每个高斯点被更新的次数，denom 可以用来计算每个高斯点的平均梯度