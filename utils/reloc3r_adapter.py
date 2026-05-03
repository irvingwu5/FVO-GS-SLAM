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


class Reloc3RSubmapRegistrator:
    """Reloc3R-based submap registration backend. Stage 4: top-K pairs + consistency."""

    def __init__(self, config):
        self.config = config
        self.topk_pairs = config.get("topk_pairs", 5)
        self.min_valid_pairs = config.get("min_valid_pairs", 2)
        self.scale_mode = config.get("scale_mode", "init_guess_norm")
        self.max_delta_t = config.get("max_delta_t", 2.0)
        self.max_delta_r_deg = config.get("max_delta_r_deg", 45.0)
        self.max_pair_rot_std_deg = config.get("max_pair_rot_std_deg", 15.0)
        self.max_pair_trans_angle_std_deg = config.get("max_pair_trans_angle_std_deg", 25.0)
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

    # ---- top-K pair construction ----

    def _build_pair_candidates(self, src_img_paths, tgt_img_paths,
                                src_kf_ids, tgt_kf_ids, K):
        """Build up to K keyframe pairs, preferring tail source kfs × tail target kfs."""
        pairs = []
        src_ids = sorted(src_kf_ids)
        tgt_ids = sorted(tgt_kf_ids)
        n_src = min(len(src_ids), K)
        n_tgt = min(len(tgt_ids), K)

        for i, si in enumerate(src_ids[-n_src:]):
            for j, ti in enumerate(tgt_ids[-n_tgt:]):
                if len(pairs) >= K:
                    break
                pairs.append((int(si), src_img_paths[src_ids.index(si)]
                                          if src_ids.index(si) < len(src_img_paths) else src_img_paths[-1],
                              int(ti), tgt_img_paths[tgt_ids.index(ti)]
                                          if tgt_ids.index(ti) < len(tgt_img_paths) else tgt_img_paths[-1]))
            if len(pairs) >= K:
                break
        return pairs

    # ---- single-pair inference core ----

    def _infer_single_pair(self, model, src_img_path, tgt_img_path):
        """Returns T_cam1_from_cam2 (src_cam → tgt_cam) or None."""
        img1 = _load_image_as_tensor(tgt_img_path, self.config.get("device", "cuda"))
        img2 = _load_image_as_tensor(src_img_path, self.config.get("device", "cuda"))
        H, W = self.image_size, self.image_size
        view1 = {"img": img1.cuda(), "true_shape": torch.tensor([[H, W]]).cuda()}
        view2 = {"img": img2.cuda(), "true_shape": torch.tensor([[H, W]]).cuda()}

        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=bool(self.use_amp)):
                _, pose2 = model(view1, view2)
        T = pose2["pose"].cpu().numpy().squeeze()
        del img1, img2, view1, view2, pose2
        return T if self._sanity_check(T) else None

    def _pair_to_submap_transform(self, T_cam, src_kf_id, tgt_kf_id,
                                   source_seed_c2w, target_seed_c2w,
                                   source_keyframe_poses, target_keyframe_poses,
                                   init_norm, src_img_path=None, tgt_img_path=None):
        """Convert image-pair pose to submap-level T_source_to_target with scale."""
        C2W_src_seed = np.array(source_seed_c2w, dtype=np.float64)
        C2W_tgt_seed = np.array(target_seed_c2w, dtype=np.float64)
        C2W_src_kf = np.array(source_keyframe_poses[src_kf_id], dtype=np.float64)
        C2W_tgt_kf = np.array(target_keyframe_poses[tgt_kf_id], dtype=np.float64)

        T_tgt_seed_to_kf = np.linalg.inv(C2W_tgt_seed) @ C2W_tgt_kf
        T_src_seed_to_kf = np.linalg.inv(C2W_src_seed) @ C2W_src_kf

        # init_guess_norm: scale Reloc3R's translation first, then chain
        raw_norm = float(np.linalg.norm(T_cam[:3, 3]))
        scale = init_norm / raw_norm if raw_norm > 1e-6 else 1.0
        T_cam_scaled = T_cam.copy()
        T_cam_scaled[:3, 3] *= scale

        T_s2t = T_tgt_seed_to_kf @ T_cam_scaled @ np.linalg.inv(T_src_seed_to_kf)
        T_s2t = np.array(T_s2t, dtype=np.float64)
        return T_s2t, scale

    # ---- consistency verification ----

    def _verify_consistency(self, candidates, init_guess):
        """Check rotation/translation consistency across candidate transforms.
        Returns (best_T, metrics) or (None, fail_metrics)."""
        n = len(candidates)
        if n < self.min_valid_pairs:
            return None, {"failure_reason": f"too_few_valid_pairs_{n}"}

        rotations = np.stack([c["T"][:3, :3] for c in candidates])
        translations = np.stack([c["T"][:3, 3] for c in candidates])

        # Rotation consistency: pairwise angular difference std
        rot_angles = []
        for i in range(n):
            for j in range(i + 1, n):
                dr = self._rot_error_deg(
                    np.vstack([np.hstack([rotations[i], np.zeros((3, 1))]),
                               np.array([[0, 0, 0, 1]])]),
                    np.vstack([np.hstack([rotations[j], np.zeros((3, 1))]),
                               np.array([[0, 0, 0, 1]])]))
                rot_angles.append(dr)
        rot_std = float(np.std(rot_angles)) if rot_angles else 0.0

        # Translation direction angular std
        trans_dirs = translations / (np.linalg.norm(translations, axis=1, keepdims=True) + 1e-8)
        mean_dir = np.mean(trans_dirs, axis=0)
        mean_dir /= np.linalg.norm(mean_dir) + 1e-8
        dir_angles = [float(np.degrees(np.arccos(np.clip(np.dot(d, mean_dir), -1.0, 1.0))))
                      for d in trans_dirs]
        dir_std = float(np.std(dir_angles))

        if rot_std > self.max_pair_rot_std_deg:
            return None, {"failure_reason": f"rot_std_{rot_std:.1f}deg", "num_valid_pairs": n,
                          "rot_std_deg": rot_std}
        if dir_std > self.max_pair_trans_angle_std_deg:
            return None, {"failure_reason": f"trans_dir_std_{dir_std:.1f}deg", "num_valid_pairs": n,
                          "rot_std_deg": rot_std, "trans_dir_std_deg": dir_std}

        # Select best: closest rotation to mean, then average translation
        best_idx = int(np.argmin(dir_angles))
        best_T = candidates[best_idx]["T"]

        return best_T, {"num_valid_pairs": n, "rot_std_deg": rot_std,
                        "trans_dir_std_deg": dir_std, "best_pair_idx": best_idx,
                        "pair_kf_ids": [(c["src_kf"], c["tgt_kf"]) for c in candidates]}

    # ---- main entry ----

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

        init_norm = float(np.linalg.norm(init_guess[:3, 3]))
        src_kf_ids = sorted(source_keyframe_poses.keys())
        tgt_kf_ids = sorted(target_keyframe_poses.keys())

        if len(src_kf_ids) == 0 or len(tgt_kf_ids) == 0:
            return self._fail_result("missing_keyframe_pose")

        # Build top-K pairs
        pairs = self._build_pair_candidates(
            source_image_paths, target_image_paths, src_kf_ids, tgt_kf_ids,
            K=self.topk_pairs)

        # Run inference on each pair
        candidates = []
        for src_kf, src_img, tgt_kf, tgt_img in pairs:
            try:
                T_cam = self._infer_single_pair(model, src_img, tgt_img)
            except Exception as e:
                Log(f"[Reloc3R] pair ({src_kf},{tgt_kf}) inference error: {e}")
                continue
            if T_cam is None:
                continue

            # ---- Reloc3R raw output diagnostic ----
            raw_R_det = float(np.linalg.det(T_cam[:3, :3]))
            raw_t_vec = T_cam[:3, 3].copy()
            raw_t_norm = float(np.linalg.norm(raw_t_vec))
            raw_t_unit = raw_t_vec / (raw_t_norm + 1e-8) if raw_t_norm > 1e-8 else raw_t_vec
            init_gt_vec = init_guess[:3, 3].copy()
            init_gt_norm = float(np.linalg.norm(init_gt_vec))
            init_gt_unit = init_gt_vec / (init_gt_norm + 1e-8) if init_gt_norm > 1e-8 else init_gt_vec
            raw_vs_init_dot = float(np.dot(raw_t_unit, init_gt_unit))
            Log(f"[Reloc3R-Diag] pair ({src_kf},{tgt_kf}): "
                f"input_order=view1(tgt)_view2(src) "
                f"raw_R_det={raw_R_det:.4f} raw_t_norm={raw_t_norm:.3f}m "
                f"raw_t_unit=[{raw_t_unit[0]:.3f},{raw_t_unit[1]:.3f},{raw_t_unit[2]:.3f}] "
                f"init_guess_t_unit=[{init_gt_unit[0]:.3f},{init_gt_unit[1]:.3f},{init_gt_unit[2]:.3f}] "
                f"raw_vs_init_dot={raw_vs_init_dot:.3f} "
                f"raw_vs_init_ratio={raw_t_norm / (init_gt_norm + 1e-8):.3f}")

            try:
                T_sub, scale = self._pair_to_submap_transform(
                    T_cam, src_kf, tgt_kf,
                    source_seed_c2w, target_seed_c2w,
                    source_keyframe_poses, target_keyframe_poses,
                    init_norm)
                candidates.append({"T": T_sub, "src_kf": src_kf, "tgt_kf": tgt_kf,
                                   "scale": scale, "raw_vs_init_dot": raw_vs_init_dot})

                # ---- Post-transform diagnostic ----
                dr_vs_init = self._rot_error_deg(
                    np.vstack([np.hstack([T_sub[:3,:3], np.zeros((3,1), dtype=T_sub.dtype)]),
                               np.array([[0,0,0,1]], dtype=T_sub.dtype)]),
                    init_guess)
                dt_vs_init = float(np.linalg.norm((T_sub @ np.linalg.inv(init_guess))[:3, 3]))
                Log(f"[Reloc3R-Diag] pair ({src_kf},{tgt_kf}) post_transform: "
                    f"T_s2t_norm={float(np.linalg.norm(T_sub[:3,3])):.3f}m "
                    f"scale={scale:.3f} dt_vs_init={dt_vs_init:.3f}m dr_vs_init={dr_vs_init:.1f}deg")
            except Exception as e:
                Log(f"[Reloc3R] pair ({src_kf},{tgt_kf}) transform error: {e}")
                continue

        Log(f"[Reloc3R] {source_id}->{target_id}: {len(candidates)}/{len(pairs)} valid pairs")

        # Aggregate raw Reloc3R direction alignment across pairs
        min_raw_dot = min((c["raw_vs_init_dot"] for c in candidates), default=None)
        mean_raw_dot = float(np.mean([c["raw_vs_init_dot"] for c in candidates])) if candidates else None

        # Top-K consistency or single-pair fallback
        if len(candidates) >= 2:
            best_T, consistency = self._verify_consistency(candidates, init_guess)
        elif len(candidates) == 1:
            best_T = candidates[0]["T"]
            consistency = {"num_valid_pairs": 1, "rot_std_deg": 0.0,
                           "trans_dir_std_deg": 0.0, "best_pair_idx": 0,
                           "pair_kf_ids": [(candidates[0]["src_kf"], candidates[0]["tgt_kf"])]}
        else:
            return self._fail_result("no_valid_pairs")

        if best_T is None:
            return self._fail_result(consistency.get("failure_reason", "consistency_failed"))

        dt = float(np.linalg.norm((best_T @ np.linalg.inv(init_guess))[:3, 3]))
        dr = self._rot_error_deg(best_T, init_guess)

        if dt > self.max_delta_t or dr > self.max_delta_r_deg:
            Log(f"[Reloc3R] {source_id}->{target_id} rejected: "
                f"dt={dt:.3f}m dr={dr:.1f}deg > thresholds "
                f"(n_pairs={consistency.get('num_valid_pairs', 0)})")
            r_metrics = {"method": "reloc3r_topk", "num_pairs": len(pairs),
                         "num_valid_pairs": consistency.get("num_valid_pairs", 0),
                         "scale_mode": self.scale_mode, "scale_value": None,
                         "delta_t": dt, "delta_r": dr,
                         "failure_reason": "delta_too_large",
                         "rot_std_deg": consistency.get("rot_std_deg"),
                         "trans_dir_std_deg": consistency.get("trans_dir_std_deg"),
                         "min_raw_vs_init_dot": min_raw_dot,
                         "mean_raw_vs_init_dot": mean_raw_dot}
            return {"success": False, "T_tgt_src": np.eye(4), "information": np.eye(6),
                    "metrics": r_metrics}

        Log(f"[Reloc3R] {source_id}->{target_id} success: "
            f"dt={dt:.3f}m dr={dr:.1f}deg n_pairs={consistency.get('num_valid_pairs',0)} "
            f"rot_std={consistency.get('rot_std_deg',0):.1f}deg")

        n_pairs = len(pairs)
        n_valid = consistency.get("num_valid_pairs", 1)
        return {
            "success": True,
            "T_tgt_src": best_T,
            "information": np.eye(6),
            "metrics": {
                "method": "reloc3r_topk",
                "fitness": float(n_valid) / max(n_pairs, 1),  # PGO gate needs fitness
                "rmse": 0.0,
                "num_pairs": n_pairs,
                "num_valid_pairs": n_valid,
                "scale_mode": "init_guess_norm",
                "scale_value": float(np.linalg.norm(best_T[:3, 3])),
                "delta_t": dt, "delta_r": dr,
                "failure_reason": None,
                "rot_std_deg": consistency.get("rot_std_deg"),
                "trans_dir_std_deg": consistency.get("trans_dir_std_deg"),
                "pair_kf_ids": consistency.get("pair_kf_ids", []),
                "min_raw_vs_init_dot": min_raw_dot,
                "mean_raw_vs_init_dot": mean_raw_dot,
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
            return False
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            return False
        if np.linalg.norm(T[:3, 3]) < 1e-8:
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
                "scale_mode": self.scale_mode, "scale_value": t,
                "delta_t": 0.0, "delta_r": 0.0,
                "failure_reason": None,
            },
        }

    def _fail_result(self, reason):
        return {
            "success": False,
            "T_tgt_src": np.eye(4), "information": np.eye(6),
            "metrics": {
                "num_pairs": 0, "num_valid_pairs": 0,
                "scale_mode": self.scale_mode,
                "scale_value": None, "delta_t": None, "delta_r": None,
                "failure_reason": reason,
            },
        }
