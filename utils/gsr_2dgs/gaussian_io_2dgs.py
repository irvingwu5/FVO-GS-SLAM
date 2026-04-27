# utils/gsr_2dgs/gaussian_io_2dgs.py
# Load 2DGS submap ckpt with proper _normal handling.
# No voxel downsample. No O3D estimate_normals.

import torch, torch.nn.functional as F, numpy as np
from utils.logging_utils import Log


def rotation_to_2dgs_normal(rotation, quat_order="wxyz"):
    """Convert _rotation quaternion to 2DGS surfel normal (local z-axis in world frame)."""
    import roma
    if quat_order == "wxyz":
        rot_roma = rotation[:, [1, 2, 3, 0]]
    else:
        rot_roma = rotation
    rot_mat = roma.unitquat_to_rotmat(rot_roma.float())
    N = rot_mat[:, :, 2]
    return F.normalize(N, dim=-1, eps=1e-8)


def load_2dgs_submap_ckpt(ckpt_path, device="cuda"):
    """Load full 2DGS Gaussian params from ckpt. No downsample.
    Returns dict with xyz, opacity, normal, rotation, scaling, features, normal_source, N.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gp = ckpt.get("gaussian_params", ckpt)

    sid = int(ckpt_path.split("/")[-1].split(".")[0])
    xyz = gp["_xyz"].float()
    opacity = gp["_opacity"].float().squeeze(-1)
    rotation = gp.get("_rotation")
    if rotation is not None:
        rotation = rotation.float()
    scaling = gp.get("_scaling")
    if scaling is not None:
        scaling = scaling.float()

    # Normal: _normal > rotation fallback
    normal_source = "unknown"
    if "_normal" in gp and gp["_normal"] is not None:
        n = gp["_normal"].float()
        if n.ndim == 2 and n.shape[1] == 3 and n.shape[0] > 0:
            normal_source = "fdn_ckpt"
        else:
            n = None
    else:
        n = None

    if n is None and rotation is not None:
        n = rotation_to_2dgs_normal(rotation)
        normal_source = "rotation_fallback"

    if n is None:
        raise RuntimeError(f"No normal source available in {ckpt_path}")

    # Pad/trim to match xyz
    N_xyz = len(xyz)
    N_n = len(n)
    if N_n < N_xyz:
        padding = rotation_to_2dgs_normal(rotation) if rotation is not None else torch.zeros(N_xyz - N_n, 3)
        if N_n > 0:
            padding[:N_n] = n
        n = padding
        normal_source += "+padded"
        Log(f"[2DGS-GSReg] submap {sid}: normal padded from {N_n} to {N_xyz}")
    elif N_n > N_xyz:
        n = n[:N_xyz]

    normal = F.normalize(n.float(), dim=-1, eps=1e-8)
    N = N_xyz

    # Validate & filter
    valid = (torch.isfinite(xyz).all(dim=1)
             & torch.isfinite(normal).all(dim=1)
             & torch.isfinite(opacity))
    if not valid.all():
        Log(f"[2DGS-GSReg] submap {sid}: filtering {valid.sum()}/{len(valid)} valid Gaussians")
        xyz = xyz[valid]
        normal = normal[valid]
        opacity = opacity[valid]
        if rotation is not None:
            rotation = rotation[valid]
        if scaling is not None:
            scaling = scaling[valid]
        N = len(xyz)

    result = {
        "xyz": xyz.to(device),
        "opacity": opacity.to(device),
        "normal": normal.to(device),
        "rotation": rotation.to(device) if rotation is not None else None,
        "scaling": scaling.to(device) if scaling is not None else None,
        "features": gp["_features_dc"].float().to(device),
        "normal_source": normal_source,
        "N": N,
        "submap_id": sid,
        "seed_pose": ckpt.get("seed_global_c2w"),
        "keyframe_ids": ckpt.get("submap_keyframes", []),
        "keyframe_poses": ckpt.get("submap_keyframe_poses", {}),
    }

    Log(f"[2DGS-GSReg] submap {sid}: N={N} normal_source={normal_source} norm_mean={normal.norm(dim=-1).mean():.4f}")
    return result
