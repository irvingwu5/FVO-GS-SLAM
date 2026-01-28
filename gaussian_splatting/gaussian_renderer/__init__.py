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

import math

import torch
# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
#修改为2dgs
from diff_surfel_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

#修改为2dgs
def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    surf=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        cx=viewpoint_camera.cx if viewpoint_camera.cx is not None else 0,
        cy=viewpoint_camera.cy if viewpoint_camera.cy is not None else 0,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = (
            torch.tensor(
                [
                    [W / 2, 0, 0, (W - 1) / 2],
                    [0, H / 2, 0, (H - 1) / 2],
                    [0, 0, far - near, near],
                    [0, 0, 0, 1],
                ]
            )
            .float()
            .cuda()
            .T
        )
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (
            (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]])
            .permute(0, 2, 1)
            .reshape(-1, 9)
        )  # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, allmap, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        w=viewpoint_camera.cam_rot_delta,
        trans=viewpoint_camera.cam_trans_delta,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets = {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
        "n_touched": n_touched,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (
            render_normal.permute(1, 2, 0)
            @ (viewpoint_camera.world_view_transform[:3, :3].T)
    ).permute(2, 0, 1)

    # get median depth map中值深度（render_depth_median）对比：期望深度对离群贡献更敏感（适合无界/稀疏场景以减少光盘锯齿），中值深度更鲁棒（适合有界场景）。
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    #期望深度图是每个像素沿视线的加权平均深度。简单来说，它表示“从相机看过去，所有被该像素贡献的高斯元的平均深度”。
    # get expected depth map：D_expected = sum_i (w_i * d_i) / sum_i w_i，其中 d_i 是第 i 个高斯元的深度，w_i 是该高斯元对像素的权重（累积到 render_alpha）
    render_depth_expected = allmap[0:1] #每个像素沿视线的加权平均深度，深度的累加量
    render_depth_expected = render_depth_expected / render_alpha
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    # get depth distortion map
    render_dist = allmap[6:7]
    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median

    if surf: #渲染深度图计算出来的法线图，宏观几何，后续让每个surfels都朝向对应的宏观法线方向，
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth) #已经转换到了世界坐标系下，assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = surf_normal.permute(2, 0, 1)
        surf_normal = surf_normal * (render_alpha).detach() #remember to multiply with accum_alpha since render_normal is unnormalized.
    else:
        surf_normal = None

    rets.update(
        {
            #"rend_alpha": render_alpha,
            "rend_normal": render_normal,
            "rend_dist": render_dist,
            "depth": surf_depth,
            "surf_normal": surf_normal,
            "opacity": render_alpha,
        }
    )

    return rets
'''
    1.Photometric loss（光度损失）: L1+DSSIM(结构相似性)
    Ll1 = l1_loss(rendered_image, gt_image)、lambda_dssim = 0.2
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) 
    确保渲染结果在视觉上逼真
    -----------------------------------------------------------------------------------------------------------
    2.Depth distortion loss（深度失真损失）: 
    lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0、lambda_dist = 0.0
    rend_dist = render_pkg["rend_dist"]
    dist_loss = lambda_dist * (rend_dist).mean()
    集中分布： 惩罚那些权重高但在深度上分布很散的情况。它强制高斯图元在沿射线的深度上尽可能集中
    消除伪影： 减少“浮空物”（floaters）和背景噪声。
    压实表面： 3DGS 容易用一团松散的“云”来模拟实心物体，这个损失强制这团“云”坍缩成一个薄且紧密的表面 。
    -----------------------------------------------------------------------------------------------------------
    3.Normal Consistency loss:
    lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0、lambda_normal = 0.05
    rend_normal  = render_pkg['rend_normal']
    surf_normal = render_pkg['surf_normal']
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    normal_loss = lambda_normal * (normal_error).mean()
    几何约束：强制每个微小的 2D 高斯圆盘的朝向（$n_i$）与由宏观深度图暗示的几何形状（$N$）(深度图梯度计算得到)保持一致
    平滑表面： 如果没有这一项，2D 圆盘可能会杂乱无章地排列（虽然颜色对了，但几何是乱的）。这个损失确保重建出的表面是光滑且物理上合理的
    -----------------------------------------------------------------------------------------------------------
render_normal  
    含义：来自 rasterizer 的输出（allmap[2:5]），表示每个像素由高斯元累积得到的法线分量（最初在视图/相机坐标系下）。这些法线是按贡献加权累积的，可能未归一化。
    转换：用相机的世界视图旋转矩阵将视图空间法线变换到世界坐标系（通过矩阵乘法和维度重排）。
surf_normal  
    含义：由深度图推导得到的“伪表面法线”，表示宏观几何方向（用于正则化）。深度图 surf_depth 是期望深度（期望深度图是每个像素沿视线的加权平均深度。简单来说，它表示“从相机看过去，所有被该像素贡献的高斯元的平均深度”）与中值深度按 pipe.depth_ratio 混合得到的。
    计算：调用 depth_to_normal(viewpoint_camera, surf_depth)（返回世界坐标系下的法线，通常通过深度的空间梯度或将临近像素反投影到 3D 再做叉乘得到），然后重排通道并乘以 render_alpha.detach()（按累积 alpha 加权，用于与 rasterizer 的法线匹配）
'''