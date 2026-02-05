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
    #要切断对相机位姿的梯度，同时保留对高斯点的梯度,.detach() 会创建一个新的 Tensor，它与原 Tensor 共享数据，但不再存在于计算图中。因此，反向传播走到这里就会停止，不会更新生成该矩阵的相机参数。
    w2c = viewpoint.world_view_transform.transpose(0, 1).detach()  # Transpose back to [4, 4] row-major standard
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
    更加鲁棒的平面约束 Loss 计算，避免越界索引导致的 CUDA device-side assert。
    """
    # 1. Check & Prepare Data: plane_equations: [N, 4], label_map: [H, W], num_planes: int
    plane_equations, label_map, num_planes = prepare_plane_data(viewpoint)

    # 验证 prepare_plane_data 返回值
    if not (isinstance(plane_equations, torch.Tensor) and isinstance(label_map, torch.Tensor)):
        return torch.tensor(0.0, device="cuda")

    # 确保 plane_equations 是二维且至少有一行
    if plane_equations.ndim != 2 or plane_equations.shape[1] < 4 or plane_equations.shape[0] == 0:
        return torch.tensor(0.0, device="cuda")

    # 同步 num_planes 与实际 plane_equations 大小，避免 mismatch
    num_planes_actual = int(plane_equations.shape[0])
    try:
        num_planes = int(num_planes)
    except Exception:
        num_planes = num_planes_actual
    num_planes = min(num_planes, num_planes_actual)

    # 2. Opacity filtering
    opacity = gaussians.get_opacity
    valid_opacity_mask = (opacity > 0.5).squeeze()
    if valid_opacity_mask.sum() == 0:
        return torch.tensor(0.0, device="cuda")

    points_world = gaussians.get_xyz[valid_opacity_mask]  # [M, 3]
    total_points = points_world.shape[0]
    SAMPLE_SIZE = 100000
    if total_points > SAMPLE_SIZE:
        indices = torch.randint(0, total_points, (SAMPLE_SIZE,), device=points_world.device)
        points_world = points_world[indices]

    # 3. world2camera 可能返回标量表示失败，先判断
    w2c_out = world2camera(viewpoint, points_world)
    if isinstance(w2c_out, torch.Tensor) and w2c_out.dim() == 0:
        return torch.tensor(0.0, device="cuda")

    try:
        points_cam, u, v, valid_z_mask, valid_pixel_mask = w2c_out
    except Exception:
        return torch.tensor(0.0, device="cuda")

    # 再次检查投影结果有效性
    if not (isinstance(points_cam, torch.Tensor) and isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor)):
        return torch.tensor(0.0, device="cuda")

    H, W = viewpoint.image_height, viewpoint.image_width
    # 确保 u,v 在 [0,W-1]/[0,H-1] 范围内，否则过滤
    # u,v 可能已经是 long，但强制转 long
    u = u.long()
    v = v.long()
    within_x = (u >= 0) & (u < W)
    within_y = (v >= 0) & (v < H)
    within_both = within_x & within_y

    if within_both.sum() == 0:
        return torch.tensor(0.0, device="cuda")

    u_valid = u[within_both]
    v_valid = v[within_both]
    points_cam_valid = points_cam[within_both]

    # 4. Sample plane ids safely
    # label_map 应该为 long 在 GPU 上
    if label_map.device != u_valid.device:
        label_map = label_map.to(device=u_valid.device)
    sampled_plane_ids = label_map[v_valid, u_valid]

    # 强制为 long，防止类型问题
    sampled_plane_ids = sampled_plane_ids.long()

    # 过滤有效 plane id：非负并且小于实际 plane 数量
    is_plane_mask = (sampled_plane_ids >= 0) & (sampled_plane_ids < num_planes)
    if is_plane_mask.sum() == 0:
        return torch.tensor(0.0, device=plane_equations.device)

    final_plane_ids = sampled_plane_ids[is_plane_mask]
    final_points = points_cam_valid[is_plane_mask]

    # 再次确保 final_plane_ids 全为合法的索引范围（防御式编程）
    # clamp 为最后保险措施（但应尽量发现数据来源错误）
    final_plane_ids_clamped = torch.clamp(final_plane_ids, 0, plane_equations.shape[0] - 1).long()

    # Gather plane parameters safely
    target_plane_params = plane_equations[final_plane_ids_clamped]

    # 5. Compute Plane Constraint Loss (same as 原实现，但更稳健)
    n = target_plane_params[:, :3]
    d = target_plane_params[:, 3]

    dot_prod = (final_points * n).sum(dim=1)
    dist = dot_prod + d

    # Outlier rejection
    valid_dist_mask = torch.abs(dist) < 0.02
    if valid_dist_mask.sum() == 0:
        return torch.tensor(0.0, device=plane_equations.device)

    dist_filtered = dist[valid_dist_mask]

    if loss_type == "l1":
        loss = torch.abs(dist_filtered).mean()
    elif loss_type == "l2":
        loss = (dist_filtered ** 2).mean()
    else:
        # Huber (delta=0.01)
        loss = torch.nn.functional.huber_loss(dist_filtered, torch.zeros_like(dist_filtered), delta=0.01)

    return loss


def build_plane_normal_gt(viewpoint):
    H, W = viewpoint.image_height, viewpoint.image_width
    plane_equations, label_map, num_planes = prepare_plane_data(viewpoint)

    # 如果数据缺失，返回全0的法线图
    if not isinstance(plane_equations, torch.Tensor) or not isinstance(label_map, torch.Tensor):
        return torch.zeros((3, H, W), dtype=torch.float32, device="cuda")

    # 确保 label_map 与 plane_equations 在同一 device 且为 long
    if label_map.device != plane_equations.device:
        label_map = label_map.to(device=plane_equations.device)
    label_map = label_map.long()
    plane_equations = plane_equations.float().to(device=plane_equations.device)

    # 初始化为 0（三通道）
    target_normal = torch.zeros((3, H, W), dtype=torch.float32, device=plane_equations.device)

    # 安全地将 num_planes 转为 int
    try:
        num_planes_int = int(num_planes)
    except Exception:
        num_planes_int = int(plane_equations.shape[0])

    # 构造平面掩码并早期返回
    is_planar_mask = (label_map < num_planes_int) & (label_map >= 0)
    if is_planar_mask.sum() == 0:
        return target_normal

    # 从 label_map 中按掩码取出 plane id，并防护性 clamp 避免越界
    valid_plane_ids = label_map[is_planar_mask].long()
    if valid_plane_ids.numel() == 0:
        return target_normal
    valid_plane_ids = torch.clamp(valid_plane_ids, 0, plane_equations.shape[0] - 1)

    # 索引获取平面法线（相机系）
    try:
        plane_normals_cam = plane_equations[valid_plane_ids, :3]  # [N,3]
    except Exception:
        return target_normal

    # 防护：去除 NaN/Inf
    if not torch.isfinite(plane_normals_cam).all():
        plane_normals_cam = torch.nan_to_num(plane_normals_cam, nan=0.0, posinf=1e3, neginf=-1e3)

    # 坐标系对齐（与文件中其他函数保持一致）
    axis_flip = torch.tensor([-1.0, 1.0, 1.0], device=plane_normals_cam.device, dtype=plane_normals_cam.dtype)
    plane_normals_cam = plane_normals_cam * axis_flip

    # 旋转到世界坐标系，确保 R 在相同 device
    R_w2c = viewpoint.world_view_transform[:3, :3].to(device=plane_normals_cam.device, dtype=plane_normals_cam.dtype)
    plane_normals_world = plane_normals_cam @ R_w2c
    plane_normals_world = torch.nn.functional.normalize(plane_normals_world, dim=1)

    # 写回到 [3, H, W]（以 boolean mask 的顺序对应上面的 valid_plane_ids）
    target_normal_permuted = target_normal.permute(1, 2, 0).contiguous()  # [H, W, 3]
    # 确保掩码数量匹配，否则保守返回
    if plane_normals_world.shape[0] != is_planar_mask.sum().item():
        # 尝试使用 positions 索引写回，保证顺序一致
        positions = is_planar_mask.nonzero(as_tuple=False)
        if plane_normals_world.shape[0] == positions.shape[0]:
            rows = positions[:, 0]
            cols = positions[:, 1]
            target_normal_permuted[rows, cols] = plane_normals_world
            target_normal = target_normal_permuted.permute(2, 0, 1)
            return target_normal
        return target_normal

    # 直接按布尔掩码写回
    target_normal_permuted[is_planar_mask] = plane_normals_world
    target_normal = target_normal_permuted.permute(2, 0, 1)  # [3, H, W]

    return target_normal


def build_combined_normal_gt(viewpoint):
    """
    统一坐标系变换逻辑：将传感器法线与平面先验法线融合并转换至世界坐标系。
    采用 GS 渲染器兼容的 OpenGL 风格变换：n_world = (n_cam * axis_flip) @ R_w2c
    """
    try:
        H, W = viewpoint.image_height, viewpoint.image_width
        device = torch.device("cuda")

        # 1. 获取基础旋转矩阵 R_w2c (从 viewpoint.world_view_transform 提取)
        # 注意：GS 框架中此矩阵通常已是行优先存储
        R_w2c = viewpoint.world_view_transform[:3, :3].to(device)
        axis_flip = torch.tensor([-1.0, 1.0, 1.0], device=device).float()
        # 2. 处理传感器法线 (Sensor Depth-to-Normal)
        sensor_depth2normal = viewpoint.normal.to(device)  # 预期形状 [3, H, W]

        # 将传感器法线转换到世界系，逻辑必须与平面法线完全一致
        sensor_flat = sensor_depth2normal.view(3, -1).permute(1, 0).contiguous()  # [HW, 3]
        sensor_flat = sensor_flat * axis_flip
        sensor_world_flat = sensor_flat @ R_w2c
        sensor_depth2normal_world = sensor_world_flat.permute(1, 0).view(3, H, W)
        # 归一化，确保单位向量一致性
        sensor_depth2normal_world = torch.nn.functional.normalize(sensor_depth2normal_world, dim=0)

        # 初始目标设为传感器法线图
        target_normal = sensor_depth2normal_world

        # 3. 获取平面数据并进行融合
        plane_equations, label_map, num_planes = prepare_plane_data(viewpoint)

        # 验证平面数据有效性，若无效则直接返回传感器法线
        if not (isinstance(plane_equations, torch.Tensor) and isinstance(label_map, torch.Tensor)):
            return target_normal

        plane_equations = plane_equations.to(device).float()
        label_map = label_map.to(device).long()

        # 构造平面掩码
        try:
            num_planes_int = int(num_planes)
        except Exception:
            num_planes_int = int(plane_equations.shape[0])
        is_planar_mask = (label_map < num_planes_int) & (label_map >= 0)

        if is_planar_mask.sum() == 0:
            return target_normal

        # 获取平面区域像素的 ID 和位置
        valid_plane_ids = label_map[is_planar_mask]
        valid_id_mask = (valid_plane_ids >= 0) & (valid_plane_ids < plane_equations.shape[0])

        if valid_id_mask.sum() == 0:
            return target_normal

        positions = is_planar_mask.nonzero(as_tuple=False)[valid_id_mask]
        plane_ids_filtered = valid_plane_ids[valid_id_mask]

        # 4. 转换平面法线到世界系
        plane_normals_cam = plane_equations[plane_ids_filtered, :3]
        # 防护 NaN/Inf
        if not torch.isfinite(plane_normals_cam).all():
            plane_normals_cam = torch.nan_to_num(plane_normals_cam, nan=0.0)

        # 应用相同的坐标轴翻转与旋转
        plane_normals_cam = plane_normals_cam * axis_flip
        plane_normals_world = plane_normals_cam @ R_w2c
        plane_normals_world = torch.nn.functional.normalize(plane_normals_world, dim=1)

        # 5. 安全写回到目标法线图
        target_normal_permuted = target_normal.permute(1, 2, 0).contiguous()  # [H, W, 3]
        rows = positions[:, 0]
        cols = positions[:, 1]

        # 用高精度的平面法线覆盖传感器法线区域
        target_normal_permuted[rows, cols] = plane_normals_world.to(target_normal_permuted.dtype)

        # 转回 [3, H, W]
        target_normal = target_normal_permuted.permute(2, 0, 1)

        return target_normal

    except Exception as e:
        # 打印错误详情有助于调试坐标系问题
        print(f"[Error in build_combined_normal_gt]: {e}")
        # 出错时返回单位化的原法线防止崩溃
        return torch.nn.functional.normalize(viewpoint.normal, dim=0).to(viewpoint.normal.device)

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


def check_normal_dir(rend_normal, gt_normal, mask=None):
    """
    检查渲染法线和真值法线的方向一致性
    rend_normal: [3, H, W]
    gt_normal: [3, H, W]
    """
    with torch.no_grad():
        if mask is not None:
            # 仅在有效区域检查
            r_n = rend_normal[:, mask > 0]
            g_n = gt_normal[:, mask > 0]
        else:
            r_n = rend_normal.flatten(1)
            g_n = gt_normal.flatten(1)

        # 1. 计算点积 (Cosine Similarity)
        cos_sim = (r_n * g_n).sum(dim=0)
        avg_cos = cos_sim.mean().item()

        # 2. 计算平均角度 (Degrees)
        # 限制范围在 [-1, 1] 防止 acos 报错
        angle = torch.acos(cos_sim.clamp(-1.0, 1.0)) * (180.0 / 3.1415926)
        avg_angle = angle.mean().item()

        # 3. 检查各分量的对齐情况 (X, Y, Z)
        # 如果某一个轴的均值是负数，说明该轴镜像反了
        axis_alignment = (r_n * g_n).mean(dim=1)

        print("-" * 30)
        print(f"[Normal Check] Avg Cosine Similarity: {avg_cos:.4f}")
        print(f"[Normal Check] Avg Angle Error: {avg_angle:.2f} degrees")
        print(f"[Normal Check] Axis Alignment (X, Y, Z): {axis_alignment.tolist()}")

        if avg_cos < 0:
            print("警告：法线方向整体相反（钝角）！请检查坐标系定义。")
        elif abs(avg_cos) < 0.1:
            print("警告：法线几乎正交！极有可能是坐标轴顺序 (e.g., XYZ vs YZX) 错误。")
        print("-" * 30)


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
