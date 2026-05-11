import json
import os
import time
import torch
import numpy as np
import torch.multiprocessing as mp
import roma
from utils.logging_utils import Log
from utils.reloc3r_adapter import estimate_keyframe_pair_pose
from utils.keyframe_pgo import (KeyframeRecord, Reloc3RPairEstimate, VerifiedLoopEdge,
                                 build_keyframe_database, build_keyframe_pose_graph,
                                 keyframe_db_stats, retrieve_keyframe_loop_candidates,
                                 KeyframeRetrievalCandidate, refine_keyframe_loop_edge,
                                 run_keyframe_pgo_trial, evaluate_keyframe_pgo_result,
                                 log_pgo_summary)
from utils.loop_depth_verifier import verify_reloc3r_pair_with_rgbd
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# ============================================================================
# 1. CosPlace Model Architecture
# ============================================================================

def _gem(x, p=torch.ones(1) * 3, eps: float = 1e-6):
    """Generalized Mean Pooling (GeM)"""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return _gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
}


class CosPlaceNetwork(nn.Module):
    """
    CosPlace 视觉位置识别网络
    结构：ResNet backbone (去掉 avgpool + fc) → L2Norm → GeM → Flatten → Linear → L2Norm
    """
    def __init__(self, backbone_name: str = "ResNet18", fc_output_dim: int = 512):
        super().__init__()
        assert backbone_name in CHANNELS_NUM_IN_LAST_CONV, \
            f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"

        backbone_fn = getattr(models, backbone_name.lower())
        backbone = backbone_fn(weights=None)
        layers = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*layers)

        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


# ============================================================================
# 2. CosPlace Weight Loading
# ============================================================================

COSPLACE_WEIGHT_URL = (
    "https://github.com/gmberton/CosPlace/releases/download/v1.0/"
    "{backbone}_{fc_output_dim}_cosplace.pth"
)


def load_cosplace_model(backbone: str = "ResNet18",
                        fc_output_dim: int = 512,
                        weight_path: str = None,
                        device: str = "cuda") -> nn.Module:
    """
    加载 CosPlace 模型，支持三种方式：
      1. 从本地 .pth 文件加载（优先）
      2. 从 GitHub Releases 直链自动下载并缓存到本地
      3. 通过 torch.hub 加载（备用）
    """
    model = CosPlaceNetwork(backbone, fc_output_dim)

    if weight_path and os.path.isfile(weight_path):
        Log(f"[LoopClosure] 从本地文件加载 CosPlace 权重: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        Log("[LoopClosure] CosPlace 本地权重加载完成。")
        return model.eval().to(device)

    url = COSPLACE_WEIGHT_URL.format(backbone=backbone, fc_output_dim=fc_output_dim)
    Log(f"[LoopClosure] 本地权重未找到，从 GitHub 下载: {url}")
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict)
        Log("[LoopClosure] CosPlace 权重下载并加载完成。")

        if weight_path:
            os.makedirs(os.path.dirname(weight_path), exist_ok=True)
            torch.save(state_dict, weight_path)
            Log(f"[LoopClosure] 权重已缓存到: {weight_path}")

        return model.eval().to(device)
    except Exception as e:
        Log(f"[LoopClosure] GitHub 下载失败: {e}")

    Log("[LoopClosure] 尝试通过 torch.hub 加载 CosPlace...")
    try:
        model = torch.hub.load(
            'gmberton/cosplace',
            'get_trained_model',
            backbone=backbone,
            fc_output_dim=fc_output_dim,
            trust_repo=True
        )
        Log("[LoopClosure] torch.hub 加载 CosPlace 完成。")
        return model.eval().to(device)
    except Exception as e:
        raise RuntimeError(
            f"[LoopClosure] 所有加载方式均失败！请手动下载权重文件：\n"
            f"  下载地址: {url}\n"
            f"  保存到: {weight_path or 'weights/ResNet18_512_cosplace.pth'}\n"
            f"  错误信息: {e}"
        )


# ============================================================================
# 3. 2DGS Rigid Transform
# ============================================================================

def rigid_transform_2dgs(gaussian_params, tsfm_matrix):
    tsfm_matrix = torch.from_numpy(tsfm_matrix).float().cuda()
    R = tsfm_matrix[:3, :3]
    t = tsfm_matrix[:3, 3]

    xyz = gaussian_params['_xyz']
    gaussian_params['_xyz'] = (xyz @ R.T) + t

    if '_rotation' in gaussian_params:
        rotation_q = gaussian_params['_rotation']
        rotation_q_roma = rotation_q[:, [1, 2, 3, 0]]  # xyzw
        cur_rot_mat = roma.unitquat_to_rotmat(rotation_q_roma)

        new_rot_mat = torch.einsum('ij,njk->nik', R, cur_rot_mat)

        new_rotation_q_roma = roma.rotmat_to_unitquat(new_rot_mat).squeeze()
        new_rotation_q = new_rotation_q_roma[:, [3, 0, 1, 2]]  # wxyz
        gaussian_params['_rotation'] = new_rotation_q

        if '_normal' in gaussian_params:
            gaussian_params['_normal'] = new_rot_mat[:, :, 2]

    return gaussian_params


# ============================================================================
# 4. Loop Closure Process
# ============================================================================

class LoopClosureProcess(mp.Process):
    # ========================================================================
    # 4.1 Initialization
    # ========================================================================
    def __init__(self, config, loop_queue):
        super().__init__()
        self.config = config
        self.loop_queue = loop_queue
        self.save_dir = self.config["Results"]["save_dir"]
        self.submaps_dir = os.path.join(self.save_dir, "submaps")
        self.device = "cuda"

        self.submap_records = {}    # 子图 ID → ckpt 路径（永不清理）

        # 视觉特征缓存（永不清理）
        self.submap_keyframe_features = {}  # {submap_id: {kf_idx: (D,) np.ndarray}}

        # ===== 回环检测基本参数 =====
        self.min_interval = self.config.get("LoopClosure", {}).get("min_interval", 3)
        self.voxel_size = self.config.get("LoopClosure", {}).get("voxel_size", 0.05)



        # ===== CosPlace 模型配置 =====
        self.cosplace_backbone = self.config.get("LoopClosure", {}).get("backbone", "ResNet18")
        self.cosplace_dim = self.config.get("LoopClosure", {}).get("feature_dim", 512)
        self.cosplace_weight_path = self.config.get("LoopClosure", {}).get(
            "weight_path", f"weights/{self.cosplace_backbone}_{self.cosplace_dim}_cosplace.pth"
        )

        # ===== Loop Closure Mode Control (Stage 0) =====
        self.mode = self.config.get("LoopClosure", {}).get("mode", "verify_only")
        Log(f"[LoopClosure] mode={self.mode}")

        # ===== Reloc3R 配置（Stage 1: 仅加载配置，不加载模型） =====
        reloc3r_cfg = self.config.get("LoopClosure", {}).get("Reloc3R", {})
        self.reloc3r_enabled = reloc3r_cfg.get("enabled", False)
        self.reloc3r_cfg = reloc3r_cfg
        # 子图元数据缓存（Reloc3R 需要 seed C2W + keyframe poses + 图像路径）
        self.submap_seed_c2w = {}          # {submap_id: 4x4 np.array}
        self.submap_keyframe_poses = {}    # {submap_id: {kf_idx: 4x4 np.array}}
        self.submap_image_paths = {}       # {submap_id: [str, ...]}
        self.submap_depth_paths = {}      # {submap_id: [str, ...]}
        self.keyframe_db = {}             # {kf_idx: KeyframeRecord}

    # ========================================================================
    # 4.2 Feature Extraction
    # ========================================================================
    def init_feature_extractor(self):
        Log(f"[LoopClosure] 初始化 CosPlace ({self.cosplace_backbone}, {self.cosplace_dim}D)...")
        self.feature_extractor = load_cosplace_model(
            backbone=self.cosplace_backbone,
            fc_output_dim=self.cosplace_dim,
            weight_path=self.cosplace_weight_path,
            device=self.device
        )

        self.img_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_submap_features_and_threshold(self, img_paths, kf_indices=None):
        """Extract per-keyframe CosPlace descriptors.

        Returns:
            kf_features: dict {kf_idx: (D,) np.ndarray} for keyframe retrieval.
        """
        feats = []
        for img_path in img_paths:
            img_tensor = torch.load(img_path, map_location="cpu")
            img_input = self.img_transform(img_tensor).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(img_input).squeeze().detach().cpu()
            feats.append(feat)
            del img_input

        kf_features = {}
        if kf_indices is not None and len(kf_indices) == len(feats):
            for kf_idx, feat in zip(kf_indices, feats):
                kf_features[int(kf_idx)] = feat.numpy().astype(np.float32)
        else:
            for i, feat in enumerate(feats):
                kf_features[i] = feat.numpy().astype(np.float32)

        return kf_features

    # ========================================================================
    # 4.9 Main Loop
    # ========================================================================
    def run(self):
        # ---- Seed Reproducibility (loop closure process) ----
        from utils.reproducibility import seed_everything
        base_seed = self.config.get("Experiment", {}).get("seed", 42)
        deterministic = self.config.get("Experiment", {}).get("deterministic", True)
        loop_seed = base_seed + 2
        seed_everything(loop_seed, deterministic=deterministic)
        Log(f"[Seed] loop_seed={loop_seed}")

        Log(f"Loop Closure 进程已启动 mode={self.mode}")
        if self.mode == "off":
            Log("[LoopClosure] mode=off, idle loop (no feature extraction, no PGO)")
            while True:
                if not self.loop_queue.empty():
                    data = self.loop_queue.get()
                    if data[0] == "stop":
                        Log("Loop Closure 进程退出.")
                        break
                    elif data[0] == "submap_saved":
                        Log(f"[LoopClosure] mode=off, submap {data[1]} received but ignored")
                else:
                    time.sleep(0.5)
            return

        self.init_feature_extractor()
        while True:
            if not self.loop_queue.empty():
                data = self.loop_queue.get()
                if data[0] == "stop":
                    Log("Loop Closure 进程退出.")
                    break
                elif data[0] == "submap_saved":
                    submap_id = data[1]
                    ckpt_path = data[2]
                    img_paths = data[3]
                    depth_paths = data[4] if len(data) > 4 else []

                    self.submap_records[submap_id] = ckpt_path
                    self.submap_image_paths[submap_id] = img_paths
                    self.submap_depth_paths[submap_id] = depth_paths

                    # 从 ckpt 读取 Reloc3R 所需的元数据（seed C2W + keyframe poses）
                    try:
                        ckpt = torch.load(ckpt_path, map_location="cpu")
                        self.submap_seed_c2w[submap_id] = np.array(
                            ckpt.get("seed_global_c2w", np.eye(4)), dtype=np.float64
                        )
                        kf_poses = ckpt.get("submap_keyframe_poses", {})
                        self.submap_keyframe_poses[submap_id] = {
                            int(k): np.array(v, dtype=np.float64) for k, v in kf_poses.items()
                        }
                        Log(f"[LoopClosure] 子图 {submap_id} 元数据: "
                            f"seed_c2w={'set' if not np.allclose(self.submap_seed_c2w[submap_id], np.eye(4)) else 'identity'}, "
                            f"kf_poses={len(self.submap_keyframe_poses[submap_id])}, "
                            f"images={len(img_paths)}")
                    except Exception as e:
                        Log(f"[LoopClosure] 读取子图 {submap_id} ckpt 元数据失败: {e}")


                    # Get keyframe indices for per-keyframe descriptor assignment
                    kf_indices = sorted(self.submap_keyframe_poses.get(submap_id, {}).keys())
                    kf_features = self.extract_submap_features_and_threshold(
                        img_paths, kf_indices=kf_indices
                    )
                    self.submap_keyframe_features[submap_id] = kf_features

                    # Diagnostic: per-keyframe descriptor stats
                    kf_norms = [float(np.linalg.norm(v)) for v in kf_features.values()]
                    Log(f"[LoopClosure] 子图 {submap_id}: {len(kf_features)} KFs, "
                        f"desc_shape={list(kf_features.values())[0].shape if kf_features else 'N/A'}, "
                        f"mean_norm={np.mean(kf_norms):.4f} min_norm={np.min(kf_norms):.4f}")

                    Log(f"[LoopClosure] 接收并处理新子图: ID {submap_id} (包含 {len(img_paths)} 个关键帧)")

                    # Rebuild unified keyframe database from all loaded submaps
                    dataset_cfg = self.config.get("Dataset", {})
                    calib = dataset_cfg.get("Calibration", {})
                    intrinsics = {
                        "fx": calib.get("fx") or dataset_cfg.get("fx"),
                        "fy": calib.get("fy") or dataset_cfg.get("fy"),
                        "cx": calib.get("cx") or dataset_cfg.get("cx"),
                        "cy": calib.get("cy") or dataset_cfg.get("cy"),
                        "width": calib.get("width") or dataset_cfg.get("width"),
                        "height": calib.get("height") or dataset_cfg.get("height"),
                    }
                    self.keyframe_db = build_keyframe_database(
                        self.submap_keyframe_poses,
                        self.submap_image_paths,
                        self.submap_seed_c2w,
                        intrinsics,
                        self.submap_depth_paths,
                    )

                    # Inject per-keyframe CosPlace descriptors into KeyframeRecord
                    for sid, kf_feats in self.submap_keyframe_features.items():
                        for kf_idx, desc in kf_feats.items():
                            if kf_idx in self.keyframe_db:
                                self.keyframe_db[kf_idx].cosplace_descriptor = desc

                    stats = keyframe_db_stats(self.keyframe_db)
                    Log(f"[KeyframeDB] total={stats['total_keyframes']} "
                        f"submaps={stats['submap_count']} "
                        f"per_submap={stats['per_submap']} "
                        f"seeds={stats['seed_keyframes']} "
                        f"missing_rgb={stats['missing_rgb']} "
                        f"missing_depth={stats['missing_depth']} "
                        f"missing_descriptor={stats['missing_descriptor']}")

                    # ---- Keyframe-level retrieval (Stage 2) ----
                    if self.mode in ("detect_only", "verify_only", "keyframe_pgo"):
                        retrieval_log_path = os.path.join(
                            self.save_dir, "loop_keyframe_retrieval.jsonl"
                        )
                        # Query from the latest submap's keyframes (use tail kfs)
                        query_kfs = kf_indices[-min(3, len(kf_indices)):]
                        all_candidates = []
                        for qkf in query_kfs:
                            cands = retrieve_keyframe_loop_candidates(
                                qkf, self.keyframe_db,
                                config=self.config.get("LoopClosure", {}),
                            )
                            all_candidates.extend(cands)

                        # Write JSONL log
                        try:
                            with open(retrieval_log_path, "a") as f:
                                for cand in all_candidates:
                                    record = {
                                        "query_keyframe_id": cand.query_keyframe_id,
                                        "target_keyframe_id": cand.target_keyframe_id,
                                        "query_submap_id": cand.query_submap_id,
                                        "target_submap_id": cand.target_submap_id,
                                        "cosplace_score": round(cand.cosplace_score, 4),
                                        "temporal_gap": cand.temporal_gap,
                                        "is_mutual": cand.is_mutual,
                                        "accepted_by_retrieval": cand.accepted_by_retrieval,
                                        "rejection_reason": cand.rejection_reason,
                                    }
                                    f.write(json.dumps(record) + "\n")
                        except Exception as e:
                            Log(f"[LoopClosure] JSONL write failed: {e}")

                        accepted = [c for c in all_candidates if c.accepted_by_retrieval]
                        mutual = [c for c in accepted if c.is_mutual]
                        Log(f"[KF Retrieval] {len(all_candidates)} candidates, "
                            f"{len(accepted)} accepted, {len(mutual)} mutual "
                            f"(query_kfs={query_kfs})")

                    # ---- Reloc3R keyframe-pair estimation (Stage 3) ----
                    reloc3r_estimates = []
                    if self.reloc3r_enabled and self.mode in ("verify_only", "keyframe_pgo"):
                        reloc3r_log_path = os.path.join(
                            self.save_dir, "reloc3r_keyframe_pairs.jsonl"
                        )
                        reloc3r_cfg = self.config.get("LoopClosure", {}).get("Reloc3R", {})
                        top_cands = [c for c in all_candidates if c.accepted_by_retrieval][:5]
                        reloc3r_accepted = 0
                        for cand in top_cands:
                            src_rec = self.keyframe_db.get(cand.query_keyframe_id)
                            tgt_rec = self.keyframe_db.get(cand.target_keyframe_id)
                            if src_rec is None or tgt_rec is None:
                                continue
                            try:
                                estimate = estimate_keyframe_pair_pose(src_rec, tgt_rec, reloc3r_cfg)
                            except Exception as e:
                                Log(f"[Reloc3R] pair ({cand.query_keyframe_id},{cand.target_keyframe_id}) crashed: {e}")
                                continue
                            reloc3r_estimates.append(estimate)
                            if estimate.accepted_by_reloc3r:
                                reloc3r_accepted += 1
                            # Write JSONL
                            try:
                                with open(reloc3r_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "source_keyframe_id": estimate.source_keyframe_id,
                                        "target_keyframe_id": estimate.target_keyframe_id,
                                        "source_submap_id": estimate.source_submap_id,
                                        "target_submap_id": estimate.target_submap_id,
                                        "raw_translation_norm": round(estimate.raw_translation_norm, 4),
                                        "raw_vs_init_dot": round(estimate.raw_vs_init_dot, 4),
                                        "scale_applied": round(estimate.scale_applied, 4),
                                        "accepted_by_reloc3r": estimate.accepted_by_reloc3r,
                                        "rejection_reason": estimate.rejection_reason,
                                    }) + "\n")
                            except Exception as e:
                                Log(f"[LoopClosure] Reloc3R JSONL write failed: {e}")
                        Log(f"[Reloc3R KF] {reloc3r_accepted}/{len(top_cands)} accepted")

                    # ---- Depth verification (Stage 4) ----
                    if self.mode in ("verify_only", "keyframe_pgo"):
                        depth_cfg = self.config.get("LoopClosure", {}).get("depth_verify", {})
                        depth_log_path = os.path.join(
                            self.save_dir, "loop_depth_verify.jsonl"
                        )
                        depth_accepted = 0
                        for est in reloc3r_estimates:
                            if not est.accepted_by_reloc3r:
                                continue
                            src_rec = self.keyframe_db.get(est.source_keyframe_id)
                            tgt_rec = self.keyframe_db.get(est.target_keyframe_id)
                            src_dpt = src_rec.depth_path if src_rec else None
                            tgt_dpt = tgt_rec.depth_path if tgt_rec else None
                            try:
                                verified = verify_reloc3r_pair_with_rgbd(
                                    est, src_dpt, tgt_dpt, intrinsics, depth_cfg,
                                )
                            except Exception as e:
                                Log(f"[Depth Verify] pair ({est.source_keyframe_id},{est.target_keyframe_id}) crashed: {e}")
                                continue
                            if verified.accepted_by_depth:
                                depth_accepted += 1
                            else:
                                Log(f"[Depth Verify Diag] ({est.source_keyframe_id},{est.target_keyframe_id}) "
                                    f"rejected: {verified.rejection_reason} "
                                    f"overlap={verified.depth_overlap:.3f} rmse={verified.depth_rmse:.4f} "
                                    f"inlier={verified.depth_inlier_ratio:.3f} valid_pts={verified.valid_projected_points}")
                            # Write JSONL
                            try:
                                with open(depth_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "source_kf": verified.source_keyframe_id,
                                        "target_kf": verified.target_keyframe_id,
                                        "scale_used": round(verified.scale_used, 4),
                                        "depth_overlap": round(verified.depth_overlap, 4),
                                        "depth_rmse": round(verified.depth_rmse, 4),
                                        "depth_inlier_ratio": round(verified.depth_inlier_ratio, 4),
                                        "valid_points": verified.valid_projected_points,
                                        "accepted_by_depth": verified.accepted_by_depth,
                                        "rejection_reason": verified.rejection_reason,
                                    }) + "\n")
                            except Exception as e:
                                Log(f"[LoopClosure] Depth verify JSONL write failed: {e}")
                        if depth_accepted < len(reloc3r_estimates):
                            for est in reloc3r_estimates:
                                if not est.accepted_by_reloc3r:
                                    continue
                                Log(f"[Depth Verify Diag] pair ({est.source_keyframe_id},{est.target_keyframe_id}) "
                                    f"raw_t={est.raw_translation_norm:.3f}m scale={est.scale_applied:.3f}")
                        Log(f"[Depth Verify] {depth_accepted}/{len(reloc3r_estimates)} accepted")

                    # ---- Refinement → VerifiedLoopEdge (Stage 5) ----
                    verified_loop_edges = []
                    if self.mode in ("verify_only", "keyframe_pgo"):
                        loop_cfg = self.config.get("LoopClosure", {})
                        # loop_edge_gates consumed inside refine_keyframe_loop_edge()
                        _gate_cfg = loop_cfg.get("loop_edge_gates", {})
                        refine_log_path = os.path.join(
                            self.save_dir, "loop_verified_edges.jsonl"
                        )
                        refine_accepted = 0
                        for est in reloc3r_estimates:
                            if not est.accepted_by_reloc3r:
                                continue
                            src_rec = self.keyframe_db.get(est.source_keyframe_id)
                            tgt_rec = self.keyframe_db.get(est.target_keyframe_id)
                            if src_rec is None or tgt_rec is None:
                                continue
                            try:
                                src_dpt = src_rec.depth_path if src_rec else None
                                tgt_dpt = tgt_rec.depth_path if tgt_rec else None
                                verified = verify_reloc3r_pair_with_rgbd(
                                    est, src_dpt, tgt_dpt, intrinsics, depth_cfg,
                                )
                                if not verified.accepted_by_depth:
                                    continue
                                edge = refine_keyframe_loop_edge(
                                    verified, src_rec, tgt_rec, loop_cfg,
                                )
                            except Exception as e:
                                Log(f"[Refine→Edge] pair ({est.source_keyframe_id},{est.target_keyframe_id}) crashed: {e}")
                                continue
                            verified_loop_edges.append(edge)
                            if edge.accepted_for_pgo:
                                refine_accepted += 1
                            # Write JSONL
                            try:
                                with open(refine_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "source_kf": edge.source_keyframe_id,
                                        "target_kf": edge.target_keyframe_id,
                                        "source_submap": edge.source_submap_id,
                                        "target_submap": edge.target_submap_id,
                                        "refinement_method": edge.verification_metrics.get("refinement_method", ""),
                                        "delta_t_from_odom": edge.verification_metrics.get("delta_t_from_odom", -1),
                                        "delta_r_from_odom": edge.verification_metrics.get("delta_r_deg_from_odom", -1),
                                        "accepted_for_pgo": edge.accepted_for_pgo,
                                        "rejection_reason": edge.rejection_reason,
                                    }) + "\n")
                            except Exception as e:
                                Log(f"[LoopClosure] Verified edge JSONL write failed: {e}")
                        Log(f"[Refine→Edge] {refine_accepted}/{len(reloc3r_estimates)} "
                            f"verified loop edges accepted for PGO")

                    # ---- Keyframe PGO trial (Stage 7) ----
                    if self.mode == "keyframe_pgo" and len(verified_loop_edges) > 0:
                        accepted_edges = [e for e in verified_loop_edges if e.accepted_for_pgo]
                        if len(accepted_edges) > 0:
                            Log(f"[KeyframePGO] {len(accepted_edges)} accepted edges, building graph...")
                            graph = build_keyframe_pose_graph(
                                self.keyframe_db, verified_loop_edges=accepted_edges,
                                config=self.config.get("LoopClosure", {}),
                            )
                            Log(f"[KeyframePGO] graph: {len(graph.nodes)} nodes, "
                                f"{graph.num_temporal_edges} temporal, "
                                f"{graph.num_handoff_edges} handoff, "
                                f"{graph.num_loop_edges} loop, "
                                f"{graph.num_skipped_duplicate_edges} skipped_duplicate ({'OK' if graph.num_skipped_duplicate_edges <= graph.num_handoff_edges else 'WARN'}) edges")

                            trial_result = run_keyframe_pgo_trial(
                                graph, config=self.config.get("LoopClosure", {}),
                            )
                            Log(f"[KeyframePGO] trial: max_corr_t={trial_result.max_correction_t:.3f}m "
                                f"max_corr_r={trial_result.max_correction_r_deg:.1f}deg "
                                f"odom_before={trial_result.odom_residual_before_mean:.4f} "
                                f"odom_after={trial_result.odom_residual_after_mean:.4f} "
                                f"loop_before={trial_result.loop_residual_before_mean:.4f} "
                                f"loop_after={trial_result.loop_residual_after_mean:.4f}")

                            trial_result = evaluate_keyframe_pgo_result(
                                trial_result, graph,
                                config=self.config.get("LoopClosure", {}),
                            )
                            log_pgo_summary(
                                graph, trial_result,
                                config=self.config.get("LoopClosure", {}),
                            )
                            if trial_result.accepted:
                                Log(f"[KeyframePGO] ACCEPTED! Saving results...")
                                pgo_result_path = os.path.join(
                                    self.save_dir, "keyframe_pgo_result.json"
                                )
                                try:
                                    with open(pgo_result_path, "w") as f:
                                        json.dump({
                                            "accepted": True,
                                            "num_nodes": len(graph.nodes),
                                            "num_loop_edges": graph.num_loop_edges,
                                            "max_correction_t": trial_result.max_correction_t,
                                            "max_correction_r_deg": trial_result.max_correction_r_deg,
                                            "odom_residual_before": trial_result.odom_residual_before_mean,
                                            "odom_residual_after": trial_result.odom_residual_after_mean,
                                            "loop_residual_before": trial_result.loop_residual_before_mean,
                                            "loop_residual_after": trial_result.loop_residual_after_mean,
                                            "keyframe_corrections": {
                                                str(k): v.tolist()
                                                for k, v in trial_result.keyframe_corrections.items()
                                            },
                                            "optimized_keyframe_c2w": {
                                                str(k): v.tolist()
                                                for k, v in trial_result.optimized_keyframe_c2w.items()
                                            },
                                        }, f, indent=2)
                                    Log(f"[KeyframePGO] result saved to {pgo_result_path}")
                                except Exception as e:
                                    Log(f"[KeyframePGO] save failed: {e}")
                            else:
                                Log(f"[KeyframePGO] REJECTED: {trial_result.rejection_reason}")
                        else:
                            Log(f"[KeyframePGO] no accepted edges, skipping trial")

                    # mode == "off" is handled above, before feature extraction

            else:
                time.sleep(0.5)
