# utils/reloc3r_adapter.py
# Stage 3: real Reloc3R single-pair inference with init_guess_norm scale.

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
    """Load a keyframe image from .pt file and convert to RGB tensor [1,3,H,W]."""
    img = torch.load(img_path, map_location="cpu")  # [3, H, W] float32 [0,1]
    img = _IMG_TRANSFORM(img).unsqueeze(0)
    return img


def _load_reloc3r_model(config):
    """Load Reloc3R model once. Called lazily on first use."""
    global _RELOC3R_MODEL
    if _RELOC3R_MODEL is not None:
        return _RELOC3R_MODEL

    repo_path = config.get("repo_path", "third_party/reloc3r")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from reloc3r.reloc3r_relpose import Reloc3rRelpose
    except ImportError as e:
        Log(f"[Reloc3R] import failed: {e}. Falling back to mock mode.")
        return None

    checkpoint = config.get("checkpoint", "siyan824/reloc3r-224")
    dev = config.get("device", "cuda")
    Log(f"[Reloc3R] loading model checkpoint={checkpoint} ...")
    try:
        model = Reloc3rRelpose.from_pretrained(checkpoint)
        model.to(dev)
        model.eval()
        _RELOC3R_MODEL = model
        _RELOC3R_DEVICE = dev
        Log(f"[Reloc3R] model loaded successfully on {dev}")
        return model
    except Exception as e:
        Log(f"[Reloc3R] model load failed: {e}. Falling back to mock mode.")
        return None


class Reloc3RSubmapRegistrator:
    """
    Reloc3R-based submap registration backend.

    Stage 3: real model loading (lazy) + single-pair inference + scale conversion.
    """

    def __init__(self, config):
        self.config = config
        self.topk_pairs = config.get("topk_pairs", 5)
        self.min_valid_pairs = config.get("min_valid_pairs", 2)
        self.scale_mode = config.get("scale_mode", "init_guess_norm")
        self.max_delta_t = config.get("max_delta_t", 2.0)
        self.max_delta_r_deg = config.get("max_delta_r_deg", 45.0)
        self.use_amp = config.get("use_amp", True)
        self.image_size = config.get("image_size", 224)
        self.mock_mode = config.get("mock_mode", False)
        self.model = None

    def _ensure_model(self):
        if self.mock_mode:
            return None
        if self.model is None:
            self.model = _load_reloc3r_model(self.config)
        return self.model

    def register_submaps(
        self,
        source_id, target_id,
        source_seed_c2w, target_seed_c2w,
        source_keyframe_poses, target_keyframe_poses,
        source_image_paths, target_image_paths,
        init_guess,
    ):
        if self.mock_mode or len(source_image_paths) == 0 or len(target_image_paths) == 0:
            return self._mock_result(source_id, target_id, init_guess,
                                     source_image_paths, target_image_paths)

        model = self._ensure_model()
        if model is None:
            return self._mock_result(source_id, target_id, init_guess,
                                     source_image_paths, target_image_paths)

        # Single-pair: use last source keyframe (closest to seed in time)
        # and CosPlace top-1 target keyframe (the most similar one)
        src_img_path = source_image_paths[-1]
        tgt_img_path = target_image_paths[0]
        src_kf_id = max(source_keyframe_poses.keys()) if source_keyframe_poses else None
        tgt_kf_id = min(target_keyframe_poses.keys()) if target_keyframe_poses else None

        if src_kf_id is None or tgt_kf_id is None:
            return self._fail_result("missing_keyframe_pose")

        try:
            img1 = _load_image_as_tensor(src_img_path, self.config.get("device", "cuda"))
            img2 = _load_image_as_tensor(tgt_img_path, self.config.get("device", "cuda"))
            H, W = self.image_size, self.image_size
            view1 = {"img": img1.cuda(), "true_shape": torch.tensor([[H, W]]).cuda()}
            view2 = {"img": img2.cuda(), "true_shape": torch.tensor([[H, W]]).cuda()}

            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=bool(self.use_amp)):
                    _, pose2 = model(view1, view2)
            T_cam1_from_cam2 = pose2["pose"].cpu().numpy().squeeze()  # 4x4
            del img1, img2, view1, view2, pose2
        except Exception as e:
            Log(f"[Reloc3R] inference error {source_id}->{target_id}: {e}")
            return self._fail_result(f"inference_error_{e}")

        # Sanity checks
        if not self._sanity_check(T_cam1_from_cam2):
            return self._fail_result("sanity_check_failed")

        # Convert image-pair pose → submap-level T_source_to_target:
        # T_cam1_from_cam2 = camera pose of src_kf in tgt_kf camera frame
        # T_source_to_target = T_tgt_seed_to_kf @ T_cam1_from_cam2 @ inv(T_src_seed_to_kf)
        C2W_src_seed = np.array(source_seed_c2w, dtype=np.float64)
        C2W_tgt_seed = np.array(target_seed_c2w, dtype=np.float64)
        C2W_src_kf = np.array(source_keyframe_poses[src_kf_id], dtype=np.float64)
        C2W_tgt_kf = np.array(target_keyframe_poses[tgt_kf_id], dtype=np.float64)

        T_tgt_seed_to_kf = np.linalg.inv(C2W_tgt_seed) @ C2W_tgt_kf      # T_tgt_seed→kf
        T_src_seed_to_kf = np.linalg.inv(C2W_src_seed) @ C2W_src_kf      # T_src_seed→kf

        T_source_to_target = T_tgt_seed_to_kf @ T_cam1_from_cam2 @ np.linalg.inv(T_src_seed_to_kf)
        T_source_to_target = np.array(T_source_to_target, dtype=np.float64)

        # Scale: use init_guess_norm
        init_norm = float(np.linalg.norm(init_guess[:3, 3]))
        reloc3r_norm = float(np.linalg.norm(T_source_to_target[:3, 3]))
        if reloc3r_norm > 1e-6:
            scale = init_norm / reloc3r_norm
        else:
            scale = 1.0
        T_source_to_target[:3, 3] *= scale

        dt = float(np.linalg.norm((T_source_to_target @ np.linalg.inv(init_guess))[:3, 3]))
        dr = self._rot_error_deg(T_source_to_target, init_guess)

        if dt > self.max_delta_t or dr > self.max_delta_r_deg:
            Log(f"[Reloc3R] {source_id}->{target_id} rejected: "
                f"dt={dt:.3f}m dr={dr:.1f}deg > thresholds")
            return self._fail_result("delta_too_large")

        Log(f"[Reloc3R] {source_id}->{target_id} success: "
            f"dt={dt:.3f}m dr={dr:.1f}deg scale={scale:.3f} "
            f"init_t={init_norm:.3f}m reloc3r_t_raw={reloc3r_norm:.3f}m")

        return {
            "success": True,
            "T_tgt_src": T_source_to_target,
            "information": np.eye(6),
            "metrics": {
                "method": "reloc3r_single_pair",
                "num_pairs": 1,
                "num_valid_pairs": 1,
                "scale_mode": self.scale_mode,
                "scale_value": float(scale),
                "delta_t": dt,
                "delta_r": dr,
                "failure_reason": None,
                "src_kf": int(src_kf_id),
                "tgt_kf": int(tgt_kf_id),
            },
        }

    @staticmethod
    def _rot_error_deg(T_a, T_b):
        R = T_a[:3, :3] @ T_b[:3, :3].T
        tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.degrees(np.arccos(tr)))

    def _sanity_check(self, T):
        det = np.linalg.det(T[:3, :3])
        if abs(det - 1.0) > 0.1:
            Log(f"[Reloc3R] sanity failed: rotation det={det:.3f}")
            return False
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            Log("[Reloc3R] sanity failed: NaN/Inf in pose")
            return False
        t_norm = np.linalg.norm(T[:3, 3])
        if t_norm < 1e-8:
            Log("[Reloc3R] sanity failed: zero translation")
            return False
        return True

    def _mock_result(self, source_id, target_id, init_guess, src_imgs, tgt_imgs):
        t = float(np.linalg.norm(init_guess[:3, 3]))
        Log(f"[Reloc3R-Mock] {source_id}->{target_id} init_t={t:.3f}m")
        return {
            "success": True,
            "T_tgt_src": init_guess.copy(),
            "information": np.eye(6),
            "metrics": {
                "method": "reloc3r_mock",
                "num_pairs": 0, "num_valid_pairs": 0,
                "scale_mode": self.scale_mode,
                "scale_value": t,
                "delta_t": 0.0, "delta_r": 0.0,
                "failure_reason": None,
            },
        }

    def _fail_result(self, reason):
        return {
            "success": False,
            "T_tgt_src": np.eye(4),
            "information": np.eye(6),
            "metrics": {
                "method": "reloc3r_single_pair",
                "num_pairs": 1, "num_valid_pairs": 0,
                "scale_mode": self.scale_mode,
                "scale_value": None,
                "delta_t": None, "delta_r": None,
                "failure_reason": reason,
            },
        }
