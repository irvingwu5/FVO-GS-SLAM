#!/usr/bin/env python
"""
Stage 7: Normal mask ablation experiment runner.

Runs 5 config groups and outputs CSV + MD summaries comparing ATE,
mask ratios, fallback counts, and submap handoff stats.

Usage:
    python scripts/run_normal_mask_ablation.py --config configs/rgbd/tum/fr1_desk.yaml
    python scripts/run_normal_mask_ablation.py --config configs/rgbd/tum/fr1_desk.yaml --dry-run
"""

import argparse, csv, json, os, subprocess, sys, time
from typing import Any, Dict, List

GROUPS = {
    "baseline": {
        "label": "A: baseline (normal mask off)",
        "overrides": {"enabled": False},
    },
    "rgb_only": {
        "label": "B: RGB only",
        "overrides": {"enabled": True, "apply_to_rgb": True, "apply_to_depth": False},
    },
    "rgb_depth": {
        "label": "C: RGB + depth",
        "overrides": {"enabled": True, "apply_to_rgb": True, "apply_to_depth": True},
    },
    "no_handoff": {
        "label": "D: disable during handoff",
        "overrides": {"enabled": True, "disable_when_handoff_active": True},
    },
    "relaxed_handoff": {
        "label": "E: relaxed handoff",
        "overrides": {"enabled": True, "relaxed_when_handoff_active": True},
    },
}

METRIC_KEYS = [
    "experiment", "ate_rmse", "ate_mean", "submap_count",
    "handoff_activation_count", "normal_mask_fallback_count",
    "normal_mask_mean_ratio", "opacity_mask_mean_ratio", "final_mask_mean_ratio",
]

def empty_metrics(name):
    return {k: None for k in METRIC_KEYS} | {"experiment": name,
        "submap_count": 0, "handoff_activation_count": 0,
        "normal_mask_fallback_count": 0, "normal_mask_mean_ratio": 0.0,
        "opacity_mask_mean_ratio": 0.0, "final_mask_mean_ratio": 0.0,
    }

def apply_overrides(scene_path, overrides, save_dir, out_path):
    import yaml
    # Load base config first
    base_dir = os.path.dirname(scene_path)
    base_path = os.path.join(base_dir, "base_config.yaml")
    if os.path.isfile(base_path):
        with open(base_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    # Apply scene overrides on top
    with open(scene_path) as f:
        scene = yaml.safe_load(f)
    for k, v in scene.items():
        if k == "inherit_from":
            continue
        if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    # Apply ablation overrides
    cfg.setdefault("Results", {})["save_dir"] = save_dir
    nm = cfg.setdefault("Tracking", {}).setdefault("normal_mask", {})
    for k, v in overrides.items():
        nm[k] = v
    with open(out_path, "w") as f:
        yaml.dump(cfg, f)

def parse_metrics(log_dir, name):
    m = empty_metrics(name)
    ate_path = os.path.join(log_dir, "ate_result.txt")
    if os.path.isfile(ate_path):
        with open(ate_path) as f:
            for line in f:
                if "rmse" in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "rmse" in p.lower() and i + 1 < len(parts):
                            try: m["ate_rmse"] = float(parts[i + 1])
                            except ValueError: pass
    jl = os.path.join(log_dir, "normal_mask_debug", "normal_mask_stats.jsonl")
    if os.path.isfile(jl):
        ratios, falls = [], 0
        with open(jl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    ratios.append(r.get("final_ratio", 0))
                    if r.get("fallback"): falls += 1
                except json.JSONDecodeError: pass
        if ratios:
            m["normal_mask_mean_ratio"] = round(float(sum(ratios) / len(ratios)), 4)
        m["normal_mask_fallback_count"] = falls
    # Count submap and handoff from logs
    log_file = os.path.join(log_dir, "slam.log")
    if not os.path.isfile(log_file):
        for fname in os.listdir(log_dir):
            if fname.endswith(".log"):
                log_file = os.path.join(log_dir, fname); break
    if os.path.isfile(log_file):
        with open(log_file) as f:
            for line in f:
                if "启动新子图" in line: m["submap_count"] += 1
                if "selected" in line and "boundary Gaussians" in line: m["handoff_activation_count"] += 1
    return m

def run_one(config_path, group_name, group_def, output_dir, dry_run, slam_cmd):
    exp_dir = os.path.join(output_dir, group_name)
    os.makedirs(exp_dir, exist_ok=True)
    temp_cfg = os.path.join(exp_dir, "config.yaml")
    apply_overrides(config_path, group_def["overrides"], exp_dir, temp_cfg)
    cmd = f"{slam_cmd} --config {temp_cfg} --eval"
    if dry_run:
        print(f"[DRY] {cmd}"); return empty_metrics(group_name)
    print(f"[RUN] {group_def['label']}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        print(f"[DONE] {group_name} {time.time()-t0:.0f}s exit={r.returncode}")
        if r.returncode:
            print(f"[STDERR] {r.stderr.strip()[:400]}")
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {group_name}")
    return parse_metrics(exp_dir, group_name)

def write_csv(metrics, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_KEYS); w.writeheader()
        for m in metrics: w.writerow(m)

def write_md(metrics, path):
    with open(path, "w") as f:
        f.write("# Normal Mask Ablation\n\n")
        f.write("| Experiment | ATE RMSE | ATE Mean | Submaps | Handoffs | Fallback | NM Ratio | Op Ratio | Final Ratio |\n")
        f.write("|---" * 10 + "|\n")
        for m in metrics:
            f.write(f"| {m['experiment']} | {m['ate_rmse'] or '—'} | {m['ate_mean'] or '—'} | "
                    f"{m['submap_count']} | {m['handoff_activation_count']} | "
                    f"{m['normal_mask_fallback_count']} | {m['normal_mask_mean_ratio']} | "
                    f"{m['opacity_mask_mean_ratio']} | {m['final_mask_mean_ratio']} |\n")

def main():
    p = argparse.ArgumentParser(description="Normal mask ablation runner")
    p.add_argument("--config", required=True, help="Path to base config YAML")
    p.add_argument("--output-dir", default="results/normal_mask_ablation")
    p.add_argument("--groups", default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--slam-cmd", default=f"{sys.executable} slam.py")
    args = p.parse_args()
    selected = list(GROUPS) if args.groups == "all" else [g.strip() for g in args.groups.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = []
    for g in selected:
        metrics.append(run_one(args.config, g, GROUPS[g], args.output_dir, args.dry_run, args.slam_cmd))
    csv_p = os.path.join(args.output_dir, "normal_mask_ablation_summary.csv")
    md_p = os.path.join(args.output_dir, "normal_mask_ablation_summary.md")
    write_csv(metrics, csv_p); write_md(metrics, md_p)
    print(f"Saved {csv_p}, {md_p}")

if __name__ == "__main__":
    main()
