#!/usr/bin/env python
"""
Stage 11: Keyframe-level PGO ablation experiment runner.

Defines 7 experiment configurations and runs SLAM with each, extracting
unified metrics for comparison. Outputs CSV, Markdown, and JSONL summaries.

Usage:
    python scripts/run_ablation.py --config configs/rgbd/tum/fr1_desk.yaml
    python scripts/run_ablation.py --config configs/rgbd/tum/fr1_desk.yaml --groups baseline,handoff
    python scripts/run_ablation.py --config configs/rgbd/tum/fr1_desk.yaml --dry-run
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


# ============================================================================
# 1. Experiment Definitions
# ============================================================================

ABLATION_GROUPS = {
    "baseline_no_loop": {
        "label": "A: Baseline (no loop)",
        "description": "Close loop closure entirely, keep tracking + mapping.",
        "overrides": {
            "LoopClosure.mode": "off",
            "Ablation.use_loop_closure": False,
        },
    },
    "handoff_only": {
        "label": "B: Handoff only",
        "description": "Enable handoff, disable loop closure and PGO.",
        "overrides": {
            "LoopClosure.mode": "off",
            "Ablation.use_loop_closure": False,
            "Submap.use_handoff": True,
        },
    },
    "keyframe_retrieval_only": {
        "label": "C: Keyframe retrieval only",
        "description": "Run CosPlace retrieval but no Reloc3R or PGO.",
        "overrides": {
            "LoopClosure.mode": "detect_only",
            "LoopClosure.Reloc3R.enabled": False,
        },
    },
    "reloc3r_verify_only": {
        "label": "D: Reloc3R + depth verify only",
        "description": "Retrieval + Reloc3R + depth verification, no PGO write.",
        "overrides": {
            "LoopClosure.mode": "verify_only",
        },
    },
    "keyframe_pgo_trial_only": {
        "label": "E: Keyframe PGO trial (no apply)",
        "description": "Full pipeline up to PGO trial, but do NOT apply to trajectory.",
        "overrides": {
            "LoopClosure.mode": "verify_only",
            "LoopClosure.legacy_submap_pgo_enabled": False,
        },
    },
    "keyframe_pgo_apply_trajectory": {
        "label": "F: PGO → trajectory",
        "description": "PGO accepted → apply to trajectory only (no Gaussian correction).",
        "overrides": {
            "LoopClosure.mode": "keyframe_pgo",
            "LoopClosure.legacy_submap_pgo_enabled": False,
            "GlobalFusion.correction_mode": "submap_rigid",
            "LoopClosure.MapCorrection.mode": "none",
        },
    },
    "keyframe_pgo_apply_map": {
        "label": "G: PGO → trajectory + map",
        "description": "PGO accepted → apply to trajectory and Gaussian map.",
        "overrides": {
            "LoopClosure.mode": "keyframe_pgo",
            "LoopClosure.legacy_submap_pgo_enabled": False,
            "LoopClosure.MapCorrection.mode": "submap_median_from_keyframes",
        },
    },
}


# ============================================================================
# 2. Metric Definitions
# ============================================================================

METRIC_KEYS = [
    "experiment",
    "ate_rmse",
    "ate_mean",
    "ate_median",
    "ate_max",
    "tracking_lost_count",
    "submap_count",
    "handoff_activation_count",
    "keyframe_count",
    "retrieval_candidate_count",
    "reloc3r_pair_count",
    "depth_verified_pair_count",
    "render_refined_edge_count",
    "accepted_loop_edge_count",
    "pgo_accepted_count",
    "pgo_rejected_count",
    "loop_residual_before_mean",
    "loop_residual_after_mean",
    "odom_residual_before_mean",
    "odom_residual_after_mean",
    "max_correction_t",
    "max_correction_r_deg",
]


def empty_metrics(experiment: str) -> Dict[str, Any]:
    """Return a metrics dict with default values."""
    m: Dict[str, Any] = {k: None for k in METRIC_KEYS}
    m["experiment"] = experiment
    m["tracking_lost_count"] = 0
    m["submap_count"] = 0
    m["handoff_activation_count"] = 0
    m["keyframe_count"] = 0
    m["retrieval_candidate_count"] = 0
    m["reloc3r_pair_count"] = 0
    m["depth_verified_pair_count"] = 0
    m["render_refined_edge_count"] = 0
    m["accepted_loop_edge_count"] = 0
    m["pgo_accepted_count"] = 0
    m["pgo_rejected_count"] = 0
    return m


# ============================================================================
# 3. Log Parsing
# ============================================================================


def parse_metrics_from_logs(log_dir: str, experiment: str) -> Dict[str, Any]:
    """Extract metrics from SLAM log files in log_dir."""
    metrics = empty_metrics(experiment)

    # Parse retrieval candidates from JSONL
    retrieval_path = os.path.join(log_dir, "loop_keyframe_retrieval.jsonl")
    if os.path.isfile(retrieval_path):
        with open(retrieval_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    metrics["retrieval_candidate_count"] += 1
                except json.JSONDecodeError:
                    pass

    # Parse Reloc3R estimates from JSONL
    reloc3r_path = os.path.join(log_dir, "reloc3r_keyframe_pairs.jsonl")
    if os.path.isfile(reloc3r_path):
        with open(reloc3r_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    metrics["reloc3r_pair_count"] += 1
                except json.JSONDecodeError:
                    pass

    # Parse depth verified from JSONL
    depth_path = os.path.join(log_dir, "loop_depth_verify.jsonl")
    if os.path.isfile(depth_path):
        with open(depth_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    metrics["depth_verified_pair_count"] += 1
                except json.JSONDecodeError:
                    pass

    # Parse verified edges from JSONL
    edges_path = os.path.join(log_dir, "loop_verified_edges.jsonl")
    if os.path.isfile(edges_path):
        with open(edges_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    metrics["render_refined_edge_count"] += 1
                    if rec.get("accepted_for_pgo"):
                        metrics["accepted_loop_edge_count"] += 1
                except json.JSONDecodeError:
                    pass

    # Parse ATE from eval output (if exists)
    ate_path = os.path.join(log_dir, "ate_result.txt")
    if os.path.isfile(ate_path):
        try:
            with open(ate_path) as f:
                content = f.read()
            # Simple parse: look for "rmse" line
            for line in content.split("\n"):
                if "rmse" in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "rmse" in p.lower() and i + 1 < len(parts):
                            try:
                                metrics["ate_rmse"] = float(parts[i + 1])
                            except ValueError:
                                pass
        except Exception:
            pass

    return metrics


# ============================================================================
# 4. Experiment Runner
# ============================================================================


def _deep_set(d, keys, value):
    """Set d[key1][key2]... = value from dot-notation key path."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def apply_overrides_to_config(base_config_path: str, overrides: Dict[str, Any],
                               save_dir: str, output_config_path: str):
    """Load base YAML, apply overrides (dot-notation keys), set save_dir, write temp config."""
    import yaml as _yaml
    with open(base_config_path) as f:
        cfg = _yaml.safe_load(f)

    cfg.setdefault("Results", {})["save_dir"] = save_dir

    for key_path, value in overrides.items():
        keys = key_path.split(".")
        # Convert string bools
        if isinstance(value, str):
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
        _deep_set(cfg, keys, value)

    with open(output_config_path, "w") as f:
        _yaml.dump(cfg, f)


def run_single_experiment(
    config_path: str,
    group_name: str,
    group_def: Dict[str, Any],
    output_dir: str,
    dry_run: bool = False,
    slam_cmd: str = f"{sys.executable} slam.py",
) -> Dict[str, Any]:
    """Run a single SLAM experiment and return metrics."""
    exp_dir = os.path.join(output_dir, group_name)
    os.makedirs(exp_dir, exist_ok=True)

    temp_config = os.path.join(exp_dir, "config.yaml")
    apply_overrides_to_config(config_path, group_def["overrides"], exp_dir, temp_config)

    cmd = f"{slam_cmd} --config {temp_config} --eval"

    metrics = empty_metrics(group_name)

    if dry_run:
        print(f"[DRY RUN] {cmd}")
        print(f"[DRY RUN] Output dir: {exp_dir}")
        return metrics

    print(f"[RUN] {group_def['label']}")
    t_start = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - t_start
        print(f"[DONE] {group_name} in {elapsed:.0f}s (exit={result.returncode})")
        if result.returncode != 0:
            print(f"[STDERR] {result.stderr.strip()[:500]}")
            print(f"[WARN] Non-zero exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {group_name} after 2h")
        return metrics
    except Exception as e:
        print(f"[ERROR] {group_name}: {e}")
        return metrics

    metrics = parse_metrics_from_logs(exp_dir, group_name)
    return metrics


# ============================================================================
# 5. Output Writers
# ============================================================================


def write_csv(metrics_list: List[Dict[str, Any]], path: str):
    """Write ablation summary as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_KEYS)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(m)


def write_markdown(metrics_list: List[Dict[str, Any]], path: str):
    """Write ablation summary as Markdown table."""
    with open(path, "w") as f:
        f.write("# Keyframe PGO Ablation Summary\n\n")
        # Header
        headers = ["Experiment", "ATE RMSE", "ATE Mean", "Submaps", "KFs",
                    "Retrieval", "Reloc3R", "Depth OK", "Loop Edges",
                    "PGO Accept", "Max Corr T"]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "---|" * len(headers) + "\n")
        for m in metrics_list:
            row = [
                m.get("experiment", "?"),
                fmt_opt(m.get("ate_rmse")),
                fmt_opt(m.get("ate_mean")),
                str(m.get("submap_count", 0)),
                str(m.get("keyframe_count", 0)),
                str(m.get("retrieval_candidate_count", 0)),
                str(m.get("reloc3r_pair_count", 0)),
                str(m.get("depth_verified_pair_count", 0)),
                str(m.get("accepted_loop_edge_count", 0)),
                str(m.get("pgo_accepted_count", 0)),
                fmt_opt(m.get("max_correction_t")),
            ]
            f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")


def write_jsonl(metrics_list: List[Dict[str, Any]], path: str):
    """Write ablation diagnostics as JSONL."""
    with open(path, "w") as f:
        for m in metrics_list:
            f.write(json.dumps(m) + "\n")


def fmt_opt(val: Any) -> str:
    """Format optional numeric value."""
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


# ============================================================================
# 6. Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Keyframe PGO ablation experiment runner"
    )
    parser.add_argument("--config", required=True,
                        help="Path to SLAM config YAML")
    parser.add_argument("--output-dir", default="results/ablation",
                        help="Output directory for ablation results")
    parser.add_argument("--groups", default="all",
                        help="Comma-separated experiment groups or 'all'")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument("--slam-cmd", default=f"{sys.executable} slam.py",
                        help="SLAM command prefix")
    args = parser.parse_args()

    if args.groups == "all":
        selected = list(ABLATION_GROUPS.keys())
    else:
        selected = [g.strip() for g in args.groups.split(",")]
        for g in selected:
            if g not in ABLATION_GROUPS:
                print(f"Unknown group: {g}. Available: {list(ABLATION_GROUPS.keys())}")
                sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_list = []
    for group_name in selected:
        group_def = ABLATION_GROUPS[group_name]
        metrics = run_single_experiment(
            args.config, group_name, group_def, args.output_dir,
            dry_run=args.dry_run, slam_cmd=args.slam_cmd,
        )
        metrics_list.append(metrics)

    write_csv(metrics_list, os.path.join(args.output_dir, "keyframe_pgo_ablation_summary.csv"))
    write_markdown(metrics_list, os.path.join(args.output_dir, "keyframe_pgo_ablation_summary.md"))
    write_jsonl(metrics_list, os.path.join(args.output_dir, "keyframe_pgo_diagnostics.jsonl"))

    print(f"\nAblation results saved to {args.output_dir}/")
    print("  - keyframe_pgo_ablation_summary.csv")
    print("  - keyframe_pgo_ablation_summary.md")
    print("  - keyframe_pgo_diagnostics.jsonl")


if __name__ == "__main__":
    main()
