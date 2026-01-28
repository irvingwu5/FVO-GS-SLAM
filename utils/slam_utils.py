import torch
import numpy as np

def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]

#这个函数实际上是在寻找内部有效点。如果一个像素位于有效区域的边缘（邻居中有无效点），或者自身无效，它都会被掩盖掉。
# 这在计算图像梯度（如函数 depth_reg 中用到）时非常重要，可以防止在深度断裂或无效边界处计算出错误的巨大梯度值
def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


# def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint): #主要用于计算 RGB 颜色跟踪损失 (Tracking Loss)。在 SLAM 系统中，这个损失值用于衡量当前渲染出的图像与真实观测图像之间的差异，通常用于优化当前的相机位姿（Tracking 过程）。
#     gt_image = viewpoint.original_image.cuda()
#     _, h, w = gt_image.shape
#     mask_shape = (1, h, w)
#     rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
#     rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape) #只要像素的 RGB 值之和大于该阈值，就被视为有效像素。这通常用于过滤掉全黑的图像边界或无效区域
#     rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask #进一步通过外部提供的梯度掩码来筛选纹理丰富区域
#     # 计算渲染图像与真值图像之间的绝对差值 (L1 Error)，并乘以不透明度掩码，确保只考虑那些被高置信度渲染的像素，这意味着模型渲染出不透明度高（实体表面）的区域，其颜色误差对损失函数的贡献更大；而透明或半透明区域的颜色误差权重较低
#     l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
#     return l1.mean() #返回整个图像所有像素误差的平均值
def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1 = opacity * torch.abs(
        image * viewpoint.rgb_pixel_mask - gt_image * viewpoint.rgb_pixel_mask
    )
    # huberloss = torch.nn.HuberLoss(reduction='mean', delta=0.005)

    # l1 = opacity * huberloss(image * viewpoint.rgb_pixel_mask,
    #  gt_image * viewpoint.rgb_pixel_mask)

    return l1.mean()


# def get_loss_tracking_rgbd(
#     config, image, depth, opacity, viewpoint, initialization=False
# ):
#     alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
#
#     gt_depth = torch.from_numpy(viewpoint.depth).to(
#         dtype=torch.float32, device=image.device
#     )[None]
#     depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
#     opacity_mask = (opacity > 0.95).view(*depth.shape)
#
#     l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint) #l1加权损失
#     depth_mask = depth_pixel_mask * opacity_mask
#     l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
#     return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()
def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    # if viewpoint.mask is not None:
    #     depth_pixel_mask = depth_pixel_mask * viewpoint.mask

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - viewpoint.gt_depth * depth_mask)
    # huberloss = torch.nn.HuberLoss(reduction='mean', delta=0.0005)
    # l1_depth =huberloss(depth * depth_mask, viewpoint.gt_depth * depth_mask)

    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


# def get_loss_mapping_rgb(config, image, depth, viewpoint):
#     gt_image = viewpoint.original_image.cuda()
#     _, h, w = gt_image.shape
#     mask_shape = (1, h, w)
#     rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
#
#     rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
#     l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
#
#     return l1_rgb.mean()
def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1_rgb = torch.abs(
        image * viewpoint.rgb_pixel_mask_mapping
        - gt_image * viewpoint.rgb_pixel_mask_mapping
    )

    return l1_rgb.mean()

# def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
#     alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
#     rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
#     gt_image = viewpoint.original_image.cuda()
#
#     gt_depth = torch.from_numpy(viewpoint.depth).to(
#         dtype=torch.float32, device=image.device
#     )[None]
#     rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
#     depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
#
#     l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
#     l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
#
#     return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()
def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    gt_image = viewpoint.original_image.cuda()

    rgb_pixel_mask = viewpoint.rgb_pixel_mask_mapping

    depth_pixel_mask = (viewpoint.gt_depth > 0.01).view(*depth.shape)
    # if viewpoint.mask is not None:
    #     depth_pixel_mask = depth_pixel_mask * viewpoint.mask

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(
        depth * depth_pixel_mask - viewpoint.gt_depth * depth_pixel_mask
    )

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()

def prepare_plane_data(viewpoint):
    # ------------准备数据，提取当前帧的平面参数和平面id----------------
    # Ensure the viewpoint has the necessary custom data
    if not hasattr(viewpoint, "plane_param_info") or not hasattr(viewpoint, "label_info"):
        return torch.tensor(0.0, device="cuda")
        # Extract info (assuming they might still be numpy arrays from the parsing functions)
        # Move plane equations to GPU: [N, 4] -> (a, b, c, d)
    plane_equations = viewpoint.plane_param_info["plane_equation"]
    if isinstance(plane_equations, np.ndarray):
        plane_equations = torch.from_numpy(plane_equations).float().cuda()

    # Move label map to GPU: [H, W]
    label_map = viewpoint.label_info["label_data"]
    if isinstance(label_map, np.ndarray):
        label_map = torch.from_numpy(label_map).long().cuda()

    num_planes = viewpoint.label_info["num_planes"]
    return plane_equations, label_map, num_planes


def world2camera(viewpoint, points_world):
    H, W = viewpoint.image_height, viewpoint.image_width
    # In 3DGS, world_view_transform is typically stored transposed.
    # P_cam = P_world @ R^T + t
    # Or if using the matrix directly: P_cam = P_world @ world_view_transform
    # We explicitly separate R and T for clarity matching standard math.
    w2c = viewpoint.world_view_transform.transpose(0, 1)  # Transpose back to [4, 4] row-major standard
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    # Transform: (M, 3) @ (3, 3).T + (1, 3)
    points_cam = points_world @ R.T + T.view(1, 3)

    # 4. Frustum Culling (Remove points outside image)
    # 4A. Depth Check (Keep points in front of camera)
    valid_z_mask = (points_cam[:, 2] > 0.01) & (points_cam[:, 2] < 100.0)

    if valid_z_mask.sum() == 0:
        return torch.tensor(0.0, device="cuda")

    points_cam_valid = points_cam[valid_z_mask]
    # 4B. Projection to Pixel Coordinates
    # Derive intrinsics from FOV if not explicitly stored
    if hasattr(viewpoint, "fx"):
        fx, fy = viewpoint.fx, viewpoint.fy
        cx, cy = viewpoint.cx, viewpoint.cy
    else:
        fovx = viewpoint.FoVx
        fovy = viewpoint.FoVy
        fx = W / (2 * torch.tan(torch.tensor(fovx) * 0.5))
        fy = H / (2 * torch.tan(torch.tensor(fovy) * 0.5))
        cx = W / 2.0
        cy = H / 2.0

    x, y, z = points_cam_valid[:, 0], points_cam_valid[:, 1], points_cam_valid[:, 2]

    # Project: u = fx * (x/z) + cx
    u = (x / z * fx + cx).long()
    v = (y / z * fy + cy).long()

    # 4C. Screen Space Check
    valid_pixel_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    if valid_pixel_mask.sum() == 0:
        return torch.tensor(0.0, device="cuda")

    # Filter points again based on screen bounds
    u_valid = u[valid_pixel_mask]
    v_valid = v[valid_pixel_mask]
    points_final = points_cam_valid[valid_pixel_mask]

    return points_final, u_valid, v_valid, valid_z_mask, valid_pixel_mask


def get_loss_mapping_plane_constraint(gaussians, viewpoint, loss_type=None):
    """
        计算点到平面的几何约束 Loss (基于当前视锥投影，解决跨帧ID不连续问题)

        Args:
            gaussians: GaussianModel 对象
            viewpoint: 当前视角的相机对象，需包含:
                       - world_view_transform: W2C 矩阵 [4, 4] (通常是转置的)
                       - projection_matrix: 投影矩阵 (用于获取FOV等，或直接使用 fx/fy)
                       - plane_params: 当前帧的平面参数 Tensor [K, 4] (ax+by+cz+d=0)
                       - plane_map: 当前帧的平面语义分割图 Tensor [H, W] (内容为 int 类型的 plane_id)
            loss_type: 'l1' 或 'l2'
    """

    # 1. Check & Prepare Data: plane_equations: [N, 4], label_map: [H, W], num_planes: int
    plane_equations, label_map, num_planes = prepare_plane_data(viewpoint)
    # -----------------------------------------------------------
    # [新增优化 1]: 筛选高质量高斯点 (Opacity Filtering)
    # -----------------------------------------------------------
    # 只约束比较"实"的点，忽略半透明的雾状点
    opacity = gaussians.get_opacity
    valid_opacity_mask = (opacity > 0.5).squeeze()

    if valid_opacity_mask.sum() == 0:
        return torch.tensor(0.0, device="cuda")

    # 提取符合条件的点
    points_world = gaussians.get_xyz[valid_opacity_mask]  # [M, 3]
    # 2. Extract XYZ from World Space
    #points_world = gaussians.get_xyz  # [M, 3]
    # --- 显存优化: 随机采样 (VRAM Fix) ---
    # 计算全部点的投影和索引极其消耗显存。采样 100k 点计算 Loss 已足够。
    total_points = points_world.shape[0]
    SAMPLE_SIZE = 100000
    if total_points > SAMPLE_SIZE:
        # 使用 randint 随机采样索引
        indices = torch.randint(0, total_points, (SAMPLE_SIZE,), device="cuda")
        points_world = points_world[indices]
    # -----------------------------------
    # 3. Transform World -> Camera Coordinate System -> Frustum Culling -> Projection to Pixel Coordinates
    points_cam, u, v, valid_z_mask, valid_pixel_mask = world2camera(viewpoint, points_world)
    # 4.Get Plane IDs and Parameters
    # Sample the plane ID for each projected point from the label map
    # Note: v corresponds to rows (height), u to columns (width)
    sampled_plane_ids = label_map[v, u]  # [M_final]
    is_plane_mask = (sampled_plane_ids < num_planes)

    if is_plane_mask.sum() == 0:
        return torch.tensor(0.0, device="cuda")

    # Apply mask
    final_plane_ids = sampled_plane_ids[is_plane_mask]
    final_points = points_cam[is_plane_mask]

    # Gather plane parameters: (a, b, c, d)
    # plane_equations is [Num_Planes, 4]
    target_plane_params = plane_equations[final_plane_ids]  # [K, 4]
    # 5. Compute Plane Constraint Loss
    # -----------------------------------------------------------
    # Plane Equation: ax + by + cz + d = 0
    # Your `parse_plane_param_from_file` returns normals (a,b,c) and d computed as -(n.center)
    # So we want to minimize: | n . p + d |

    # Extract normal (a,b,c) and d
    n = target_plane_params[:, :3]
    d = target_plane_params[:, 3]

    # Compute dot product (n . p)
    dot_prod = (final_points * n).sum(dim=1)

    # Distance to plane
    dist = dot_prod + d
    # -----------------------------------------------------------
    # [新增优化 2]: 深度一致性剔除 (Outlier Rejection)
    # -----------------------------------------------------------
    # 如果点到平面的距离本来就非常大(例如 > 10cm)，说明可能是误匹配(遮挡边界等)
    # 强行拉扯会破坏结构。我们只优化那些"离平面不远"的点。
    valid_dist_mask = torch.abs(dist) < 0.10  # 10cm 阈值
    if valid_dist_mask.sum() == 0: return torch.tensor(0.0, device="cuda")

    dist_filtered = dist[valid_dist_mask]

    # ===========================================================
    # [修改点]: 添加基于置信度的动态加权 (Confidence-based Weighting)
    # ===========================================================

    # 1. 计算每个点的基础 Loss (注意：必须使用 reduction='none' 以获得逐点 Loss)
    #if loss_type == "l1":
    #    per_point_loss = torch.abs(dist_filtered)
    #elif loss_type == "l2":
    #    per_point_loss = dist_filtered ** 2
    #else:
        # Huber loss 默认 reduction='mean'，这里必须改为 'none' 才能和权重相乘
    #    per_point_loss = torch.nn.functional.huber_loss(
    #        dist_filtered,
    #        torch.zeros_like(dist_filtered),
    #        delta=0.05,
    #        reduction='none'  # <--- 关键修改
    #    )

    # 2. 计算置信度权重 (Gaussian Kernel)
    # 逻辑：距离越小(dist -> 0)，权重越接近 1；距离越大(dist -> 10cm)，权重迅速衰减
    # sigma 控制衰减速度，建议设为 0.05 (5cm)
    #sigma = 0.05
    #confidence_weight = torch.exp(- (dist_filtered ** 2) / (sigma ** 2))

    # [可选] 停止梯度的传递给权重，防止网络试图通过移动点来增加权重（虽然在此场景下通常不需要，但更严谨）
    #confidence_weight = confidence_weight.detach()

    # 3. 计算加权后的 Loss
    # 方式 A (Soft Constraint - 推荐): 简单的加权平均。
    # 离得远的点不仅 Loss 大，但权重小，最终梯度会被抑制，不会强行拉扯。
    #loss = (per_point_loss * confidence_weight).mean()
    # -----------------------------------------------------------
    # [新增优化 3]: 使用 Huber Loss
    # -----------------------------------------------------------
    if loss_type == "l1":
        loss = torch.abs(dist_filtered).mean()
    elif loss_type == "l2":
        loss = (dist_filtered ** 2).mean()
    else: # Huber (默认推荐)
        # delta=0.01 (1cm): 小于1cm用L2平滑逼近，大于1cm用L1线性惩罚
        loss = torch.nn.functional.huber_loss(dist_filtered, torch.zeros_like(dist_filtered), delta=0.01)

    return loss #1.2667


# 在 slam_utils.py 或 map 函数外部辅助函数

# def build_combined_normal_gt(viewpoint, surf_normal, label_map, plane_equations, num_planes):
#     """
#     构建混合法线监督目标 (World Space)
#     """
#
#     # 获取旋转矩阵 (用于转换平面法线)
#     R_w2c = viewpoint.world_view_transform[:3, :3]
#
#     target_normal = surf_normal.clone().detach()  # [3, H, W]
#
#     # 2. 生成平面掩码
#     if isinstance(label_map, np.ndarray):
#         label_map = torch.from_numpy(label_map).long().cuda()
#
#     is_planar_mask = (label_map < num_planes) & (label_map >= 0)
#
#     if is_planar_mask.sum() > 0:
#         valid_plane_ids = label_map[is_planar_mask]
#
#         # 提取平面法线 (Camera Space)
#         plane_normals_cam = plane_equations[valid_plane_ids, :3]
#
#         # =========================================================
#         # [修正 2]: 保持平面法线的处理逻辑 (这个是对的)
#         # =========================================================
#         # 1. 坐标系对齐 (OpenCV -> OpenGL/3DGS)
#         axis_flip = torch.tensor([1.0, -1.0, -1.0], device="cuda", dtype=plane_normals_cam.dtype)
#         plane_normals_cam = plane_normals_cam * axis_flip
#
#         # 2. 旋转到世界坐标系 (Camera -> World)
#         plane_normals_world = plane_normals_cam @ R_w2c.T
#
#         # 3. 归一化
#         plane_normals_world = torch.nn.functional.normalize(plane_normals_world, dim=1)
#         # ========================================================
#         # [DEBUG START]: 插入调试代码查看数值
#         # ========================================================
#         # 为了避免每一帧都打印刷屏，建议只针对第0帧或者特定频率打印
#         # if viewpoint.uid == 0:
#         #     # 提取同一区域（Mask区域）的 surf_normal (Reference) 和 plane_normal (Candidate)
#         #     # target_normal 是 [3, H, W]，需要转置一下取 mask
#         #     surf_vectors = target_normal.permute(1, 2, 0)[is_planar_mask]  # [N, 3]
#         #     plane_vectors = plane_normals_world  # [N, 3]
#         #
#         #     # 计算均值
#         #     surf_mean = surf_vectors.mean(dim=0).cpu().numpy()
#         #     plane_mean = plane_vectors.mean(dim=0).cpu().numpy()
#         #
#         #     print(f"\n>>> DEBUG Normals (Frame {viewpoint.uid}) <<<")
#         #     print(f"Target (Surf) Mean: [{surf_mean[0]:.4f}, {surf_mean[1]:.4f}, {surf_mean[2]:.4f}]")
#         #     print(f"Pred   (Plane) Mean: [{plane_mean[0]:.4f}, {plane_mean[1]:.4f}, {plane_mean[2]:.4f}]")
#         #
#         #     # 自动判断差异
#         #     diff = surf_mean * plane_mean  # 逐元素相乘
#         #     axes = ['X', 'Y', 'Z']
#         #
#         #     print("--- Axis Check ---")
#         #     for i in range(3):
#         #         # 如果两个向量在某轴上的乘积是负数，且绝对值比较大（说明不是噪声），则该轴反了
#         #         if abs(surf_mean[i]) > 0.1:  # 只检查主要分量，忽略接近0的噪声轴
#         #             status = "✅ MATCH" if diff[i] > 0 else "❌ FLIPPED (Need to verify)"
#         #             print(f"Axis {axes[i]}: {status} (Val: {surf_mean[i]:.4f} vs {plane_mean[i]:.4f})")
#         #         else:
#         #             print(f"Axis {axes[i]}: IGNORE (Value too small: {surf_mean[i]:.4f})")
#         #     print("==========================================\n")
#         # ========================================================
#         # 5. 替换 Target 中平面区域的值
#         target_normal_permuted = target_normal.permute(1, 2, 0)
#         target_normal_permuted[is_planar_mask] = plane_normals_world
#         target_normal = target_normal_permuted.permute(2, 0, 1)
#
#     return target_normal

# def build_combined_normal_gt(viewpoint, sensor_normal, label_map, plane_equations, num_planes):
#     """
#     构建混合法线监督目标 (World Space)
#     输入:
#         sensor_normal: [1, 3, H, W] 或 [3, H, W] (Camera Space, OpenCV定义)
#         plane_equations: [M, 4] (Camera Space)
#     输出:
#         target_normal: [3, H, W] (World Space)
#     """
#     H, W = viewpoint.image_height, viewpoint.image_width
#
#     # 确保输入是 [3, H, W]
#     if sensor_normal.dim() == 4:
#         sensor_normal = sensor_normal.squeeze(0)  # 右x、下y、前z (OpenCV相机坐标系)
#
#     # =========================================================
#     # [关键修改]: 将 sensor_normal 从 Camera Space 转到 World Space
#     # =========================================================
#     # 1. 维度变换 [3, H, W] -> [N, 3] 以便矩阵乘法
#     normal_cam = sensor_normal.permute(1, 2, 0).reshape(-1, 3)
#
#     # 2. 坐标系对齐 (OpenCV -> OpenGL/3DGS)
#     # 必须与下面平面法线的处理保持一致！
#     axis_flip = torch.tensor([1.0, -1.0, -1.0], device="cuda", dtype=normal_cam.dtype)
#     normal_cam = normal_cam * axis_flip
#
#     # 3. 旋转到世界坐标系
#     # R_w2c 是 World-to-Camera 矩阵 (通常是 row-major 的 rotation)
#     # 所以 Camera-to-World 的变换是乘 R_w2c.T
#     R_w2c = viewpoint.world_view_transform[:3, :3]  # 右x、上y、后z (OpenGL坐标系)
#     normal_world = normal_cam @ R_w2c.T
#
#     # 4. 归一化 (防止插值或计算误差)
#     normal_world = torch.nn.functional.normalize(normal_world, dim=1)
#
#     # 5. 还原形状 [N, 3] -> [3, H, W] 并作为基础 target
#     # 此时 target_normal 存储的是全图的 Sensor Normal (World Space)
#     target_normal = normal_world.reshape(H, W, 3).permute(2, 0, 1)
#
#     # =========================================================
#     # 平面区域替换逻辑 (保持不变)
#     # =========================================================
#     if isinstance(label_map, np.ndarray):
#         label_map = torch.from_numpy(label_map).long().cuda()
#
#     is_planar_mask = (label_map < num_planes) & (label_map >= 0)
#
#     if is_planar_mask.sum() > 0:
#         valid_plane_ids = label_map[is_planar_mask]
#
#         # 提取平面法线 (Camera Space)
#         plane_normals_cam = plane_equations[valid_plane_ids, :3]
#
#         # [修正 2]: 保持平面法线的处理逻辑
#         # 这里的 axis_flip 已经定义过了，直接复用逻辑
#         plane_normals_cam = plane_normals_cam * axis_flip
#
#         # 旋转到世界坐标系 (Camera -> World)
#         plane_normals_world = plane_normals_cam @ R_w2c.T
#
#         # 归一化
#         plane_normals_world = torch.nn.functional.normalize(plane_normals_world, dim=1)
#
#         # ========================================================
#         # [DEBUG START]: 检查同一区域的 Sensor Normal 和 Plane Normal 是否一致
#         # ========================================================
#         # 建议仅在特定帧打印，防止刷屏 (例如第0帧或每100帧)
#         if viewpoint.uid == 0:
#             # 1. 提取 Mask 区域内的 Sensor Normal (作为 Reference)
#             # target_normal 目前是 [3, H, W]，先转为 [H, W, 3] 再用 mask 索引
#             sensor_vectors = target_normal.permute(1, 2, 0)[is_planar_mask]  # Shape: [N_planar, 3]
#
#             # 2. 获取对应的 Plane Normal (作为 Candidate)
#             plane_vectors = plane_normals_world  # Shape: [N_planar, 3]
#
#             # 3. 计算均值对比 (用于检查轴是否翻转)
#             s_mean = sensor_vectors.mean(dim=0).cpu().numpy()
#             p_mean = plane_vectors.mean(dim=0).cpu().numpy()
#
#             # 4. 计算余弦相似度 (点积)
#             # dim=1 表示对每个像素的 xyz 向量做点积
#             dot_prod = torch.sum(sensor_vectors * plane_vectors, dim=1)
#             mean_sim = dot_prod.mean().item()
#
#             print(f"\n>>> DEBUG Normals Alignment (Frame {viewpoint.uid}) <<<")
#             print(f"  Sensor Mean (World): [{s_mean[0]:.4f}, {s_mean[1]:.4f}, {s_mean[2]:.4f}]")
#             print(f"  Plane  Mean (World): [{p_mean[0]:.4f}, {p_mean[1]:.4f}, {p_mean[2]:.4f}]")
#             print(f"  Mean Cosine Similarity: {mean_sim:.4f}")
#
#             # 自动诊断
#             if mean_sim > 0.8:
#                 print("  ✅ ALIGNMENT: EXCELLENT (方向一致)")
#             elif mean_sim < -0.8:
#                 print("  ❌ ALIGNMENT: INVERTED (方向差180度) -> 需要给其中一个取反")
#             else:
#                 print("  ⚠️ ALIGNMENT: MISMATCH (可能轴错位或坐标系定义不同)")
#
#             # 轴向检查
#             axes = ['X', 'Y', 'Z']
#             print("  --- Axis Check ---")
#             for i in range(3):
#                 # 只有当分量绝对值足够大时才值得比较正负号
#                 if abs(s_mean[i]) > 0.1:
#                     match = "MATCH" if (s_mean[i] * p_mean[i] > 0) else "FLIPPED"
#                     print(f"  Axis {axes[i]}: {match} (Val: {s_mean[i]:.4f} vs {p_mean[i]:.4f})")
#             print("=======================================================\n")
#         # ========================================================
#         # [DEBUG END]
#
#         # 5. 替换 Target 中平面区域的值
#         target_normal_permuted = target_normal.permute(1, 2, 0)  # [H, W, 3]
#         target_normal_permuted[is_planar_mask] = plane_normals_world
#         target_normal = target_normal_permuted.permute(2, 0, 1)  # [3, H, W]
#
#     return target_normal
def build_plane_normal_gt(viewpoint):
    pass
def build_combined_normal_gt(viewpoint):

    """
    构建混合法线监督目标 (World Space)
    输入:
        sensor_normal: [1, 3, H, W] 或 [3, H, W] (Camera Space, OpenCV定义)
        plane_equations: [M, 4] (Camera Space)
    输出:
        target_normal: [3, H, W] (World Space)
    """
    # 获取传感器法线,将法线从相机空间转换到世界空间
    H, W = viewpoint.image_height, viewpoint.image_width
    sensor_depth2normal = viewpoint.normal #(1,3,H,W)
    sensor_depth2normal_world = (viewpoint.T[0:3, 0:3].T @ sensor_depth2normal.view(3, -1)).view(3, H, W) # (3,H,W)
    # 网络估计法线

    # 属于平面的像素对应网络估计的法线，非平面的像素对应传感器法线

    # 2. 坐标系对齐 (OpenCV -> OpenGL/3DGS)
    # 必须与下面平面法线的处理保持一致！
    axis_flip = torch.tensor([1.0, -1.0, -1.0], device="cuda", dtype=sensor_depth2normal.dtype)
    # normal_cam = normal_cam * axis_flip

    # 3. 旋转到世界坐标系
    # R_w2c 是 World-to-Camera 矩阵 (通常是 row-major 的 rotation)
    # 所以 Camera-to-World 的变换是乘 R_w2c.T
    R_w2c = viewpoint.world_view_transform[:3, :3]  # 右x、上y、后z (OpenGL坐标系)
    # normal_world = normal_cam @ R_w2c.T

    # 4. 归一化 (防止插值或计算误差)
    # normal_world = torch.nn.functional.normalize(normal_world, dim=1)

    # 5. 还原形状 [N, 3] -> [3, H, W] 并作为基础 target
    # 此时 target_normal 存储的是全图的 Sensor Normal (World Space)
    #target_normal = normal_world.reshape(H, W, 3).permute(2, 0, 1)
    target_normal = sensor_depth2normal_world  # [3, H, W]

    # =========================================================
    # 平面区域替换逻辑 (保持不变)
    # =========================================================
    plane_equations, label_map, num_planes = prepare_plane_data(viewpoint)

    if isinstance(label_map, np.ndarray):
        label_map = torch.from_numpy(label_map).long().cuda()

    is_planar_mask = (label_map < num_planes) & (label_map >= 0)

    if is_planar_mask.sum() > 0:
        valid_plane_ids = label_map[is_planar_mask]

        # 提取平面法线 (Camera Space)
        plane_normals_cam = plane_equations[valid_plane_ids, :3]

        # [修正 2]: 保持平面法线的处理逻辑
        # 这里的 axis_flip 已经定义过了，直接复用逻辑
        plane_normals_cam = plane_normals_cam * axis_flip

        # 旋转到世界坐标系 (Camera -> World)
        plane_normals_world = plane_normals_cam @ R_w2c.T
        # 归一化
        plane_normals_world = torch.nn.functional.normalize(plane_normals_world, dim=1)

        # ========================================================
        # [DEBUG START]: 检查同一区域的 Sensor Normal 和 Plane Normal 是否一致
        # ========================================================
        # 建议仅在特定帧打印，防止刷屏 (例如第0帧或每100帧)
        # if viewpoint.uid == 0:
        #     # 1. 提取 Mask 区域内的 Sensor Normal (作为 Reference)
        #     # target_normal 目前是 [3, H, W]，先转为 [H, W, 3] 再用 mask 索引
        #     sensor_vectors = target_normal.permute(1, 2, 0)[is_planar_mask]  # Shape: [N_planar, 3]
        #
        #     # 2. 获取对应的 Plane Normal (作为 Candidate)
        #     plane_vectors = plane_normals_world  # Shape: [N_planar, 3]
        #
        #     # 3. 计算均值对比 (用于检查轴是否翻转)
        #     s_mean = sensor_vectors.mean(dim=0).cpu().numpy()
        #     p_mean = plane_vectors.mean(dim=0).cpu().numpy()
        #
        #     # 4. 计算余弦相似度 (点积)
        #     # dim=1 表示对每个像素的 xyz 向量做点积
        #     dot_prod = torch.sum(sensor_vectors * plane_vectors, dim=1)
        #     mean_sim = dot_prod.mean().item()
        #
        #     print(f"\n>>> DEBUG Normals Alignment (Frame {viewpoint.uid}) <<<")
        #     print(f"  Sensor Mean (World): [{s_mean[0]:.4f}, {s_mean[1]:.4f}, {s_mean[2]:.4f}]")
        #     print(f"  Plane  Mean (World): [{p_mean[0]:.4f}, {p_mean[1]:.4f}, {p_mean[2]:.4f}]")
        #     print(f"  Mean Cosine Similarity: {mean_sim:.4f}")
        #
        #     # 自动诊断
        #     if mean_sim > 0.8:
        #         print("  ✅ ALIGNMENT: EXCELLENT (方向一致)")
        #     elif mean_sim < -0.8:
        #         print("  ❌ ALIGNMENT: INVERTED (方向差180度) -> 需要给其中一个取反")
        #     else:
        #         print("  ⚠️ ALIGNMENT: MISMATCH (可能轴错位或坐标系定义不同)")
        #
        #     # 轴向检查
        #     axes = ['X', 'Y', 'Z']
        #     print("  --- Axis Check ---")
        #     for i in range(3):
        #         # 只有当分量绝对值足够大时才值得比较正负号
        #         if abs(s_mean[i]) > 0.1:
        #             match = "MATCH" if (s_mean[i] * p_mean[i] > 0) else "FLIPPED"
        #             print(f"  Axis {axes[i]}: {match} (Val: {s_mean[i]:.4f} vs {p_mean[i]:.4f})")
        #     print("=======================================================\n")
        # ========================================================
        # [DEBUG END]

        # 5. 替换 Target 中平面区域的值
        target_normal_permuted = target_normal.permute(1, 2, 0)  # [H, W, 3]
        target_normal_permuted[is_planar_mask] = plane_normals_world
        target_normal = target_normal_permuted.permute(2, 0, 1)  # [3, H, W]

    return target_normal



def get_depth_dist_loss(render_pkg):
    rend_dist = render_pkg["rend_dist"]
    dist_loss = rend_dist.mean()
    return dist_loss

def get_normal_consistency_loss(render_pkg):
    rend_normal  = render_pkg['rend_normal'] #转到了世界坐标系下
    surf_normal = render_pkg['surf_normal'] #转到了世界坐标系下
    #render_alpha = render_pkg['opacity']
    #dot_product = (rend_normal * (-surf_normal)).sum(dim=0)
    normal_error = (1 - (rend_normal * (-surf_normal)).sum(dim=0))[None]
    normal_loss = normal_error.mean()
    #normal_loss = (render_alpha - dot_product).mean()
    return normal_loss

def _save_tensor_as_image(tensor, path, normal_map=False):
    import os
    from PIL import Image
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = tensor.detach().cpu()
    # support [3,H,W], [1,H,W], [H,W]
    if t.dim() == 3 and t.shape[0] == 3:
        img = t.permute(1, 2, 0)  # H,W,3
        if normal_map:
            img = (img * 0.5 + 0.5).clamp(0, 1)  # map from [-1,1] -> [0,1]
        else:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        arr = (img.numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path)
    else:
        if t.dim() == 3 and t.shape[0] == 1:
            t2 = t[0]
        elif t.dim() == 2:
            t2 = t
        else:
            # fallback: take first channel if exists
            t2 = t[0] if t.dim() >= 3 else t
        img = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)
        arr = (img.numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).convert("L").save(path)

def _save_normal_pair(render_pkg, save_dir, cam_idx, basename="normal"):
    import os
    rend_normal = render_pkg.get("rend_normal", None)
    surf_normal = render_pkg.get("surf_normal", None)
    surf_normal = -surf_normal
    if rend_normal is None or surf_normal is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    _save_tensor_as_image(rend_normal, os.path.join(save_dir, f"{basename}_{cam_idx}_rend.png"), normal_map=True)
    _save_tensor_as_image(surf_normal, os.path.join(save_dir, f"{basename}_{cam_idx}_surf.png"), normal_map=True)


def _save_gt_normal(tensor, save_dir, cam_idx, basename="normal_gt"):
    import os
    if tensor is None:
        return

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # [关键]: 保持与 _save_normal_pair 一致的视觉习惯
    # 如果你的 surf_normal 在可视化时需要取反 (-surf_normal)，
    # 那么基于它构建的 gt_normal 通常也需要取反才能得到正确的法线颜色图（如蓝色朝向相机）
    # 如果发现保存的图颜色不对（例如发黄而不是发蓝），请去掉这个负号
    vis_tensor = tensor

    file_path = os.path.join(save_dir, f"{basename}_{cam_idx}.png")
    _save_tensor_as_image(vis_tensor, file_path, normal_map=True)


def _save_rendered_rgb(render_pkg, save_dir, cam_idx, basename="rgb"):
        """
        保存渲染的 RGB 图像。
        支持 render_pkg 中固定键名：'render'。
        支持张量形状：[3,H,W], [H,W,3], [1,3,H,W], [B,3,H,W]（取第一个），并会将数值归一化到 [0,255]。
        """
        import os
        from PIL import Image
        import numpy as np
        import torch

        # 直接读取固定键名 'render'
        if "render" not in render_pkg:
            return
        tensor = render_pkg["render"]

        os.makedirs(save_dir, exist_ok=True)
        t = tensor.detach().cpu()

        # 规范为 H,W,3
        if t.dim() == 4 and t.shape[0] == 1 and t.shape[1] == 3:
            img = t[0].permute(1, 2, 0)
        elif t.dim() == 4 and t.shape[0] > 1 and t.shape[1] == 3:
            img = t[0].permute(1, 2, 0)
        elif t.dim() == 3 and t.shape[0] == 3:
            img = t.permute(1, 2, 0)
        elif t.dim() == 3 and t.shape[2] == 3:
            img = t
        else:
            # fallback: try to pick first 3 channels
            if t.dim() >= 3:
                c = t.shape[0]
                if c >= 3:
                    img = t[:3].permute(1, 2, 0) if t.dim() == 3 else t[0, :3].permute(1, 2, 0)
                else:
                    return
            else:
                return

        # 归一化到 [0,1] 再 *255
        img = img.float()
        minv = img.min()
        maxv = img.max()
        if (maxv - minv).abs() < 1e-8:
            img = img.clamp(0, 1)
        else:
            img = (img - minv) / (maxv - minv)

        arr = (img.numpy() * 255).astype(np.uint8)
        path = os.path.join(save_dir, f"{basename}_{cam_idx}.png")
        Image.fromarray(arr).convert("RGB").save(path)

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
