#!/bin/bash
#
# Stage 5: Surface-Aware Depth Rendering Smoke Test
#
# Runs quick smoke tests (50-100 frames) to compare SA depth modes against baseline.
# Each run logs config + metrics for later analysis.
#
# Usage:
#   bash scripts/run_sa_depth_smoke.sh <scene_config> [max_frames]
#
# Examples:
#   bash scripts/run_sa_depth_smoke.sh configs/rgbd/tum/fr1_desk.yaml 80
#   bash scripts/run_sa_depth_smoke.sh configs/rgbd/tum/fr2_xyz.yaml 100
#   bash scripts/run_sa_depth_smoke.sh configs/rgbd/replica/office0.yaml 60

set -euo pipefail

SCENE_CONFIG="${1:?Usage: $0 <scene_config> [max_frames]}"
MAX_FRAMES="${2:-80}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# Resolve output dir
SCENE_NAME=$(basename "$SCENE_CONFIG" .yaml)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/sa_depth_ablation/${SCENE_NAME}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "SA Depth Smoke Test"
echo "  Scene:   $SCENE_CONFIG"
echo "  Frames:  $MAX_FRAMES"
echo "  GPU:     $GPU"
echo "  Log dir: $LOG_DIR"
echo "============================================================"

# ------------------------------------------------------------------
# Helper: patch pipeline_params in a temp config and run
# ------------------------------------------------------------------
run_ablation() {
    local label="$1"
    local use_sa="$2"
    local use_sa_depth="$3"
    local depth_ratio="$4"
    local lambda_dist="${5:-0.0}"

    local tmp_config="$LOG_DIR/_tmp_${label}.yaml"
    local log_file="$LOG_DIR/${label}.log"

    echo ""
    echo "--- [$label] use_sa=$use_sa use_sa_depth=$use_sa_depth depth_ratio=$depth_ratio lambda_dist=$lambda_dist ---"

    # Generate a temporary config that inherits from the scene config
    # and overrides relevant pipeline_params only
    python3 -c "
import yaml

tmp = {
    'inherit_from': '$SCENE_CONFIG',
    'pipeline_params': {
        'use_sa': $use_sa,
        'use_sa_depth': $use_sa_depth,
        'depth_ratio': $depth_ratio,
        'debug_sa_depth': True,
    },
    'opt_params': {
        'lambda_dist': $lambda_dist,
    }
}

with open('$tmp_config', 'w') as f:
    yaml.dump(tmp, f)
"

    CUDA_VISIBLE_DEVICES="$GPU" python slam.py \
        --config "$tmp_config" \
        --eval \
        --max_frames "$MAX_FRAMES" \
        2>&1 | tee "$log_file"

    # Extract key metrics
    echo "  >> Log saved: $log_file"

    # Clean up temp config
    rm -f "$tmp_config"
}

# ------------------------------------------------------------------
# A0: Baseline
# ------------------------------------------------------------------
run_ablation "A0_baseline"       False False 1.0

# ------------------------------------------------------------------
# A1: SA forward only (SA depth adjustment in CUDA, but not used in loss)
# ------------------------------------------------------------------
run_ablation "A1_sa_fwd_only"    True  False 1.0

# ------------------------------------------------------------------
# A2: SA expected depth (use SA expected depth directly, no distortion loss)
# ------------------------------------------------------------------
run_ablation "A2_sa_exp_depth"   True  True  0.0

# ------------------------------------------------------------------
# A3: SA mixed depth (30% median + 70% expected)
# ------------------------------------------------------------------
run_ablation "A3_sa_mixed"       True  True  0.3

# ------------------------------------------------------------------
# A4: SA expected depth + weak distortion loss
# ------------------------------------------------------------------
run_ablation "A4_sa_depth_dist"  True  True  0.0  0.01

echo ""
echo "============================================================"
echo "Smoke test complete."
echo "Logs: $LOG_DIR"
echo ""
echo "To compare ATE, grep logs:"
echo "  grep -E 'ATE|tracking loss|depth loss' $LOG_DIR/*.log"
echo "============================================================"
