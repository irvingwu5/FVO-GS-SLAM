# utils/gsr_2dgs/viewpoint_localizer_2dgs.py
# Render-based viewpoint localization against a target 2DGS map.
# Optimizes only camera pose delta, not Gaussian params. No downsample.

import torch, numpy as np
from gaussian_splatting.gaussian_renderer import render as gs_render


def viewpoint_localize_2dgs(viewpoint, target_gaussians, pipeline_params, bg,
                             init_w2c, config, max_iters=80):
    """Localize a single viewpoint in a target 2DGS map. Returns T_tgt_src (4x4)."""
    dev = init_w2c.device
    # Save original state
    orig_T = viewpoint.T.clone()
    orig_fixed = getattr(viewpoint, "fixed_pose", False)

    viewpoint.T = init_w2c.clone()
    viewpoint.fixed_pose = False
    viewpoint.cam_rot_delta = torch.nn.Parameter(torch.zeros(3, device=dev))
    viewpoint.cam_trans_delta = torch.nn.Parameter(torch.zeros(3, device=dev))

    lr_rot = config.get("gsreg_lr_rot", 0.002)
    lr_trans = config.get("gsreg_lr_trans", 0.005)
    rw = config.get("gsreg_rgb_weight", 1.0)
    dw = config.get("gsreg_depth_weight", 1.0)
    nw = config.get("gsreg_normal_weight", 0.1)
    ow = config.get("gsreg_opacity_weight", 0.0)
    use_huber = config.get("gsreg_use_huber_loss", True)

    opt = torch.optim.Adam([
        {"params": [viewpoint.cam_rot_delta], "lr": lr_rot},
        {"params": [viewpoint.cam_trans_delta], "lr": lr_trans},
    ])

    has_depth = hasattr(viewpoint, "depth") and viewpoint.depth is not None
    best_loss, best_T = float("inf"), None

    for it in range(max_iters):
        pkg = gs_render(viewpoint, target_gaussians, pipeline_params, bg, surf=False)
        if pkg is None:
            break
        render_rgb, render_depth, render_alpha = pkg["render"], pkg["depth"], pkg["opacity"]
        gt_img = viewpoint.original_image.to(dev)

        # RGB loss with optional Huber
        if use_huber:
            rgb_loss = torch.nn.functional.huber_loss(render_rgb, gt_img, delta=0.1)
        else:
            rgb_loss = torch.abs(render_rgb - gt_img).mean()
        loss = rw * rgb_loss

        # Depth loss
        if has_depth:
            gt_d = torch.from_numpy(viewpoint.depth).float().to(dev).unsqueeze(0)
            mask = gt_d > 0.01
            if mask.any():
                if use_huber:
                    depth_loss = torch.nn.functional.huber_loss(
                        render_depth[mask], gt_d[mask], delta=0.05)
                else:
                    depth_loss = torch.abs(render_depth[mask] - gt_d[mask]).mean()
                loss += dw * depth_loss

        # Normal consistency loss: penalize deviation from GT normal in camera frame
        if nw > 0 and "rend_normal" in pkg and hasattr(viewpoint, "normal"):
            rn = pkg["rend_normal"]  # [3, H, W] in camera frame
            # viewpoint.normal is [1, 3, H, W] in camera frame (from FDN)
            gt_normal = viewpoint.normal.to(dev)
            # Mask: only where GT normal is valid
            if hasattr(viewpoint, "mask"):
                normal_mask = viewpoint.mask.to(dev)
            else:
                normal_mask = (gt_normal.norm(dim=1, keepdim=True) > 0.01).float()
            # Cosine similarity: 1 - |dot|
            cos_sim = torch.abs((rn.unsqueeze(1) * gt_normal).sum(dim=1, keepdim=True))
            normal_loss = (normal_mask * (1.0 - cos_sim)).mean()
            loss += nw * normal_loss

        # Opacity regularization
        if ow > 0:
            loss += ow * (1.0 - render_alpha).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        # Update pose
        from utils.pose_utils import update_pose
        update_pose(viewpoint)

        li = loss.item()
        if li < best_loss:
            best_loss = li
            best_T = viewpoint.T.clone()
        del pkg

    # Restore
    viewpoint.T = orig_T
    viewpoint.fixed_pose = orig_fixed

    if best_T is None:
        return {"success": False, "loss": float("inf"), "T": None}

    # T from viewpoint w2c: T_tgt_src = T_w2c^{-1}
    T_tgt_src = torch.linalg.inv(best_T).detach().cpu().numpy().astype(np.float64)
    return {"success": True, "loss": float(best_loss), "T": T_tgt_src}
