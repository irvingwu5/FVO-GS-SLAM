# utils/gsr_2dgs/solver_2dgs.py
# LoopSplat-style 2DGS Gaussian registration main entry.

import torch, numpy as np
from utils.logging_utils import Log
from .gaussian_io_2dgs import load_2dgs_submap_ckpt
from .overlap_2dgs import compute_overlap_2dgs
from .viewpoint_localizer_2dgs import viewpoint_localize_2dgs
from .pose_fusion_2dgs import fuse_relative_poses


def _rot_error_deg(T_a, T_b):
    R = T_a[:3, :3] @ T_b[:3, :3].T
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def registration_2dgs_gsreg(src_ckpt, tgt_ckpt, init_guess, config,
                              pipeline_params=None, bg_color=None,
                              src_viewpoint=None, tgt_viewpoint=None):
    """LoopSplat-style 2DGS render-based registration.
    init_guess: T_src_to_tgt (4x4) frontend prior. Must not be identity for loop edges.
    Returns dict with success, T_tgt_src, metrics.
    """
    sid = int(src_ckpt.split("/")[-1].split(".")[0])
    tid = int(tgt_ckpt.split("/")[-1].split(".")[0])

    radius = config.get("gsreg_overlap_radius", 0.10)
    min_ov = config.get("gsreg_min_overlap", 0.08)
    topk = config.get("gsreg_topk_views", 3)
    max_iters = config.get("gsreg_max_iters", 80)
    max_loss = config.get("gsreg_max_loss", 0.08)
    min_views = config.get("gsreg_min_successful_views", 1)
    max_dt_init = config.get("gsreg_max_delta_t_from_init", 0.50)
    max_dr_init = config.get("gsreg_max_delta_r_from_init", 30.0)
    allow_id = config.get("allow_identity_loop_init", False)

    # Identity init gate
    init_is_id = np.allclose(init_guess, np.eye(4), atol=1e-4)
    init_t = np.linalg.norm(init_guess[:3, 3])

    Log(f"[2DGS-GSReg] edge {sid} -> {tid} init_t={init_t:.3f}m init_is_id={init_is_id}")

    if init_is_id and not allow_id:
        Log(f"[2DGS-GSReg] rejected edge {sid} -> {tid} reason=init_missing_identity_forbidden")
        return {"success": False, "reason": "init_missing", "overlap": 0}

    # Load Gothic params
    try:
        src = load_2dgs_submap_ckpt(src_ckpt)
        tgt = load_2dgs_submap_ckpt(tgt_ckpt)
    except Exception as e:
        Log(f"[2DGS-GSReg] load failed for {sid}->{tid}: {e}")
        return {"success": False, "reason": f"load_failed_{e}", "overlap": 0}

    # Overlap
    ov = compute_overlap_2dgs(src["xyz"], tgt["xyz"], init_guess, radius=radius)
    if ov["overlap"] < min_ov:
        Log(f"[2DGS-GSReg] rejected {sid} -> {tid} reason=overlap_too_low {ov['overlap']:.3f} < {min_ov}")
        return {"success": False, "reason": "overlap_too_low", "overlap": ov["overlap"]}

    # Viewpoint localization: need viewpoint objects from submap keyframes
    # Fallback: use provided viewpoints or construct minimal ones
    if src_viewpoint is None or tgt_viewpoint is None:
        Log(f"[2DGS-GSReg] no viewpoint provided for {sid}->{tid}, skipping render localization")
        return {"success": False, "reason": "no_viewpoint", "overlap": ov["overlap"]}

    # Build target Gaussian model from tgt ckpt (lightweight)
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    tgt_gm = _build_tmp_gaussian(tgt)
    src_gm = _build_tmp_gaussian(src)

    candidates = []
    iters_per = max_iters

    # src->tgt: localize src viewpoint in tgt map
    src_init_w2c = (torch.from_numpy(init_guess).float().cuda() @ src_viewpoint.T.cuda()).float()
    r1 = viewpoint_localize_2dgs(src_viewpoint, tgt_gm, pipeline_params, bg_color,
                                  src_init_w2c, config, max_iters=iters_per)
    if r1["success"]:
        candidates.append({"T": r1["T"], "loss": r1["loss"], "overlap": ov["overlap"], "similarity": 0.8})

    # tgt->src: localize tgt viewpoint in src map, then invert
    inv_guess = np.linalg.inv(init_guess)
    tgt_init_w2c = (torch.from_numpy(inv_guess).float().cuda() @ tgt_viewpoint.T.cuda())
    r2 = viewpoint_localize_2dgs(tgt_viewpoint, src_gm, pipeline_params, bg_color,
                                  tgt_init_w2c, config, max_iters=iters_per)
    if r2["success"]:
        T_inv = r2["T"]
        T_dir = np.linalg.inv(T_inv)
        candidates.append({"T": T_dir, "loss": r2["loss"], "overlap": ov["overlap"], "similarity": 0.8})

    Log(f"[2DGS-GSReg] {sid}->{tid} localization: src->tgt={'ok' if r1['success'] else 'fail'} "
        f"tgt->src={'ok' if r2['success'] else 'fail'} n_candidates={len(candidates)}")

    n_ok = len(candidates)
    if n_ok < min_views:
        Log(f"[2DGS-GSReg] rejected {sid} -> {tid} reason=no_successful_view_localization n_ok={n_ok} < {min_views}")
        return {"success": False, "reason": "no_view_loc", "overlap": ov["overlap"], "n_candidates": n_ok}

    # Fuse
    T_fused, all_Ts = fuse_relative_poses(candidates)
    if T_fused is None:
        return {"success": False, "reason": "fusion_failed", "overlap": ov["overlap"]}

    dt_i = np.linalg.norm((T_fused @ np.linalg.inv(init_guess))[:3, 3])
    dr_i = _rot_error_deg(T_fused, init_guess)

    Log(f"[2DGS-GSReg] {sid}->{tid} fused n={n_ok} dt_from_init={dt_i:.3f}m dr_from_init={dr_i:.1f}deg")

    if dt_i > max_dt_init or dr_i > max_dr_init:
        Log(f"[2DGS-GSReg] rejected {sid} -> {tid} reason=delta_from_init_too_large")
        return {"success": False, "reason": "delta_too_large", "overlap": ov["overlap"]}

    return {
        "success": True,
        "T_tgt_src": T_fused,
        "overlap": ov["overlap"],
        "n_candidates": n_ok,
        "delta_t_from_init": dt_i,
        "delta_r_from_init": dr_i,
    }


def _build_tmp_gaussian(sub_data, device="cuda"):
    """Build minimal GaussianModel for rendering from submap data dict."""
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    import torch.nn as nn
    gm = GaussianModel(sh_degree=0)
    gm._xyz = nn.Parameter(sub_data["xyz"].to(device).requires_grad_(False))
    f = sub_data.get("features")
    if f is not None:
        gm._features_dc = nn.Parameter(f.to(device).requires_grad_(False))
        gm._features_rest = nn.Parameter(torch.zeros(len(f), 15, 3, device=device).requires_grad_(False))
    else:
        gm._features_dc = nn.Parameter(torch.zeros(sub_data["N"], 1, 3, device=device).requires_grad_(False))
        gm._features_rest = nn.Parameter(torch.zeros(sub_data["N"], 15, 3, device=device).requires_grad_(False))
    gm._opacity = nn.Parameter(sub_data["opacity"].to(device).unsqueeze(-1).requires_grad_(False))
    gm._scaling = nn.Parameter(sub_data["scaling"].to(device).requires_grad_(False)
                                if sub_data.get("scaling") is not None
                                else torch.ones(sub_data["N"], 3, device=device).requires_grad_(False))
    gm._rotation = nn.Parameter(sub_data["rotation"].to(device).requires_grad_(False)
                                 if sub_data.get("rotation") is not None
                                 else torch.zeros(sub_data["N"], 4, device=device).requires_grad_(False))
    gm._rotation.data[:, 0] = 1.0
    gm.max_radii2D = torch.zeros(sub_data["N"], device=device)
    gm.xyz_gradient_accum = torch.zeros(sub_data["N"], 1, device=device)
    gm.denom = torch.zeros(sub_data["N"], 1, device=device)
    gm.unique_kfIDs = torch.zeros(sub_data["N"], device=device).int()
    gm.n_obs = torch.zeros(sub_data["N"], device=device).int()
    gm.optimizer = None
    return gm
