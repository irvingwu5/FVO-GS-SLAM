# utils/gsr_2dgs/viewpoint_localizer_2dgs.py
# Render-based viewpoint localization against a target 2DGS map.
# Optimizes only camera pose delta, not Gaussian params. No downsample.

import torch, numpy as np
from gaussian_splatting.gaussian_renderer import render as gs_render
from utils.logging_utils import Log


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
    rw, dw, nw = config.get("gsreg_rgb_weight", 1.0), config.get("gsreg_depth_weight", 1.0), config.get("gsreg_normal_weight", 0.1)

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

        loss = rw * torch.abs(render_rgb - gt_img).mean()
        if has_depth:
            gt_d = torch.from_numpy(viewpoint.depth).float().to(dev).unsqueeze(0)
            mask = gt_d > 0.01
            if mask.any():
                loss += dw * torch.abs(render_depth[mask] - gt_d[mask]).mean()
        if nw > 0 and "rend_normal" in pkg:
            rn = pkg["rend_normal"]
            loss += nw * (1.0 - torch.abs(rn).mean())

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
