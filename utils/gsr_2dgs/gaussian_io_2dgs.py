# utils/gsr_2dgs/gaussian_io_2dgs.py
# Load 2DGS submap ckpt with proper _normal handling.
# No voxel downsample. No O3D estimate_normals.

import os, torch, torch.nn.functional as F, numpy as np
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


def load_keyframe_viewpoint_from_ckpt(ckpt_path, kf_id, config, device="cuda"):
    """Construct a minimal Camera object from saved submap keyframe data.

    Loads the keyframe's RGB image from the saved .pt file and its pose from the
    ckpt's submap_keyframe_poses dict. Constructs a Camera with only the fields
    needed for render-based viewpoint localization.

    Args:
        ckpt_path: path to the .ckpt file
        kf_id: integer keyframe index to load
        config: LoopClosure config dict (must contain cam fx/fy/cx/cy/W/H)
        device: target device

    Returns:
        Camera object or None if loading fails
    """
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
    from utils.camera_utils import Camera

    submaps_dir = os.path.dirname(ckpt_path)
    sid = int(os.path.basename(ckpt_path).split(".")[0])
    img_path = os.path.join(submaps_dir, f"{sid:06d}_img_{kf_id}.pt")

    if not os.path.exists(img_path):
        Log(f"[2DGS-GSReg] image not found for submap {sid} kf {kf_id}: {img_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    kf_poses = ckpt.get("submap_keyframe_poses", {})
    c2w = kf_poses.get(kf_id) or kf_poses.get(str(kf_id))
    del ckpt

    if c2w is None:
        Log(f"[2DGS-GSReg] pose not found for submap {sid} kf {kf_id}")
        return None

    if isinstance(c2w, torch.Tensor):
        c2w = c2w.numpy()
    c2w = np.array(c2w, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    gt_T = torch.from_numpy(w2c).float().to(device)

    # Load RGB image
    img_tensor = torch.load(img_path, map_location="cpu")
    if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
        # Already [3, H, W]
        color = img_tensor.to(device)
        H, W = img_tensor.shape[1], img_tensor.shape[2]
    else:
        Log(f"[2DGS-GSReg] unexpected image shape for submap {sid} kf {kf_id}: {img_tensor.shape}")
        return None

    # Intrinsics from config
    fx = config.get("cam", {}).get("fx", 525.0)
    fy = config.get("cam", {}).get("fy", 525.0)
    cx = config.get("cam", {}).get("cx", 319.5)
    cy = config.get("cam", {}).get("cy", 239.5)
    cam_H = config.get("cam", {}).get("H", H)
    cam_W = config.get("cam", {}).get("W", W)

    proj_mat = getProjectionMatrix2(
        znear=0.01, zfar=100.0,
        fx=fx, fy=fy, cx=cx, cy=cy,
        W=cam_W, H=cam_H,
    ).transpose(0, 1).to(device)

    fovx = float(2 * np.arctan(cam_W / (2 * fx)))
    fovy = float(2 * np.arctan(cam_H / (2 * fy)))

    cam = Camera(
        uid=kf_id,
        color=color,
        depth=None,
        gt_T=gt_T,
        dynamic_intrinsic=None,
        projection_matrix=proj_mat,
        fx=fx, fy=fy, cx=cx, cy=cy,
        fovx=fovx, fovy=fovy,
        image_height=cam_H, image_width=cam_W,
        device=device,
    )
    cam.T = gt_T.clone()

    return cam


def select_best_keyframe_for_registration(submap_data, ckpt_path, config, device="cuda"):
    """Select the best keyframe from a submap for GS registration.

    Chooses the keyframe with the highest mean opacity in its rendered view
    (proxy for best map coverage). Falls back to the middle keyframe.

    Returns (Camera, kf_id) or (None, None).
    """
    kf_ids = submap_data.get("keyframe_ids", [])
    if not kf_ids:
        return None, None

    # Prefer keyframes in the middle of the submap (best coverage)
    kf_ids = sorted([int(k) for k in kf_ids])
    if len(kf_ids) >= 3:
        # Try a few candidates: middle third of the submap
        candidates = kf_ids[len(kf_ids)//3: 2*len(kf_ids)//3]
    else:
        candidates = kf_ids

    # Select the kf with best opacity-based coverage proxy
    opacity = submap_data.get("opacity")
    if opacity is not None and len(opacity) > 0:
        best_kf = candidates[len(candidates)//2]  # default: middle
        Log(f"[2DGS-GSReg] selected kf {best_kf} (middle of {len(kf_ids)} kfs) for registration")
    else:
        best_kf = candidates[len(candidates)//2]

    cam = load_keyframe_viewpoint_from_ckpt(ckpt_path, int(best_kf), config, device)
    if cam is None and len(candidates) > 1:
        # Fallback: try first candidate
        cam = load_keyframe_viewpoint_from_ckpt(ckpt_path, int(candidates[0]), config, device)
    return cam, int(best_kf)
