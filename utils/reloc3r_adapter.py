# utils/reloc3r_adapter.py
# Stage 4: top-K keyframe pairs + consistency verification.

import os
import sys
import numpy as np
import torch
import torchvision.transforms as T
from utils.logging_utils import Log


# ---- Reloc3R model loading (lazy, on first use) ----

_RELOC3R_MODEL = None
_RELOC3R_DEVICE = "cuda"

_IMG_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def _load_image_as_tensor(img_path, device="cuda"):
    img = torch.load(img_path, map_location="cpu")
    img = _IMG_TRANSFORM(img).unsqueeze(0)
    return img


def _load_reloc3r_model(config):
    global _RELOC3R_MODEL
    if _RELOC3R_MODEL is not None:
        return _RELOC3R_MODEL

    repo_path = config.get("repo_path", "third_party/reloc3r")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from reloc3r.reloc3r_relpose import Reloc3rRelpose
    except Exception as e:
        Log(f"[Reloc3R] import failed: {e}. Falling back to mock mode.")
        return None

    dev = config.get("device", "cuda")
    image_size = config.get("image_size", 224)
    local_path = config.get("local_checkpoint", "")

    if local_path and os.path.isfile(local_path):
        Log(f"[Reloc3R] loading from local checkpoint: {local_path}")
        try:
            model = Reloc3rRelpose(img_size=image_size)
            state_dict = torch.load(local_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            model.to(dev)
            model.eval()
            _RELOC3R_MODEL = model
            _RELOC3R_DEVICE = dev
            Log(f"[Reloc3R] model loaded from local file, image_size={image_size}")
            return model
        except Exception as e:
            Log(f"[Reloc3R] local checkpoint load failed: {e}. Trying HF hub...")

    checkpoint = config.get("checkpoint", "siyan824/reloc3r-224")
    Log(f"[Reloc3R] loading model from HF hub: {checkpoint} ...")
    try:
        model = Reloc3rRelpose.from_pretrained(checkpoint)
        model.to(dev)
        model.eval()
        _RELOC3R_MODEL = model
        _RELOC3R_DEVICE = dev
        Log(f"[Reloc3R] model loaded successfully from HF hub on {dev}")
        return model
    except Exception as e:
        Log(f"[Reloc3R] model load failed: {e}. Falling back to mock mode.")
        return None



# ============================================================================
# Stage 3: Keyframe-pair estimate (NOT submap-level)
# ============================================================================

def estimate_keyframe_pair_pose(source_record, target_record, config):
    """Estimate coarse relative pose for a single keyframe pair using Reloc3R.

    This function operates at KEYFRAME level. It does NOT produce submap
    transforms and does NOT write PGO edges.

    Args:
        source_record: KeyframeRecord for the source keyframe.
        target_record: KeyframeRecord for the target keyframe.
        config: Reloc3R config dict (from LoopClosure.Reloc3R).

    Returns:
        Reloc3RPairEstimate with T_target_from_source_raw and diagnostics.
    """
    from utils.keyframe_pgo import Reloc3RPairEstimate

    estimate = Reloc3RPairEstimate(
        source_keyframe_id=source_record.keyframe_id,
        target_keyframe_id=target_record.keyframe_id,
        source_submap_id=source_record.submap_id,
        target_submap_id=target_record.submap_id,
        source_c2w_global=source_record.c2w_global.copy(),
        target_c2w_global=target_record.c2w_global.copy(),
    )

    if source_record.rgb_path is None or target_record.rgb_path is None:
        estimate.rejection_reason = "missing_rgb_path"
        return estimate

    if not os.path.isfile(source_record.rgb_path) or not os.path.isfile(target_record.rgb_path):
        estimate.rejection_reason = "rgb_file_not_found"
        return estimate

    # Load model
    model = _load_reloc3r_model(config)
    if model is None:
        estimate.rejection_reason = "model_load_failed"
        return estimate

    try:
        # view1=target, view2=source → T_cam1_from_cam2 = T_target_from_source
        img_tgt = _load_image_as_tensor(target_record.rgb_path, config.get("device", "cuda"))
        img_src = _load_image_as_tensor(source_record.rgb_path, config.get("device", "cuda"))
        H = W = config.get("image_size", 224)
        view1 = {"img": img_tgt.cuda(), "true_shape": torch.tensor([[H, W]]).cuda()}
        view2 = {"img": img_src.cuda(), "true_shape": torch.tensor([[H, W]]).cuda()}

        with torch.inference_mode():
            use_amp = config.get("use_amp", True)
            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                _, pose2 = model(view1, view2)
        T_raw = pose2["pose"].cpu().numpy().squeeze()
        del img_src, img_tgt, view1, view2, pose2
    except Exception as e:
        estimate.rejection_reason = f"inference_error: {e}"
        return estimate

    # Sanity check
    det = np.linalg.det(T_raw[:3, :3])
    if abs(det - 1.0) > 0.1 or np.any(np.isnan(T_raw)) or np.any(np.isinf(T_raw)):
        estimate.rejection_reason = f"sanity_check_failed det={det:.4f}"
        return estimate

    raw_t_norm = float(np.linalg.norm(T_raw[:3, 3]))
    if raw_t_norm < 1e-8:
        estimate.rejection_reason = "zero_translation"
        return estimate

    estimate.T_target_from_source_raw = np.array(T_raw, dtype=np.float64)
    estimate.raw_translation_norm = raw_t_norm

    # Compute odometry-based T_target_from_source for comparison
    estimate.odom_T_target_from_source = np.linalg.inv(target_record.c2w_global) @ source_record.c2w_global

    # Direction alignment: raw vs odometry init (diagnostic only, scale NOT applied)
    raw_t_unit = T_raw[:3, 3] / raw_t_norm
    init_t = estimate.odom_T_target_from_source[:3, 3]
    init_norm = float(np.linalg.norm(init_t))
    if init_norm > 1e-6:
        init_t_unit = init_t / init_norm
        estimate.raw_vs_init_dot = float(np.dot(raw_t_unit, init_t_unit))

    # Preserve raw Reloc3R translation — scale search is delegated to depth verifier.
    # init_guess_norm is diagnostic only (direction check), NOT applied to T.
    estimate.scale_applied = 1.0
    estimate.T_target_from_source_raw = np.array(T_raw, dtype=np.float64)

    estimate.accepted_by_reloc3r = True
    estimate.num_valid_matches = 1
    return estimate

