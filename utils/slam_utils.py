import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
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
        return get_loss_tracking_rgb(config, image_ab, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, opacity, viewpoint): #主要用于计算 RGB 颜色跟踪损失 (Tracking Loss)。在 SLAM 系统中，这个损失值用于衡量当前渲染出的图像与真实观测图像之间的差异，通常用于优化当前的相机位姿（Tracking 过程）。
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape) #只要像素的 RGB 值之和大于该阈值，就被视为有效像素。这通常用于过滤掉全黑的图像边界或无效区域
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask #进一步通过外部提供的梯度掩码来筛选纹理丰富区域
    # 计算渲染图像与真值图像之间的绝对差值 (L1 Error)，并乘以不透明度掩码，确保只考虑那些被高置信度渲染的像素，这意味着模型渲染出不透明度高（实体表面）的区域，其颜色误差对损失函数的贡献更大；而透明或半透明区域的颜色误差权重较低
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean() #返回整个图像所有像素误差的平均值


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, opacity, viewpoint) #l1加权损失
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, initialization=False, apply_exposure=True):
    if initialization or not apply_exposure:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


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
