#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# User-tunable settings
# =============================================================================
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new.py
CHECKPOINT=./work_dirs/full_attn_cover_pyr_aabb_dice3d_new/epoch_15_lss_only.pth
GT_ROOT=/home/cvlab/Desktop/NIPS2026/datasets/efficientocf_gt_f3/GMO
GPU_ID=0
GPUS=1
PORT=50041

# final_score = sigmoid(logit(cls_conf) + DEPTH_BETA * depth_confidence)
DEPTH_BETA=1.0
FG_THRESHOLD=0.75
OCC_THRESHOLD=0.85
RENDER_WEIGHT=combined            # combined / cls / ones
TOPK=50                           # sweep 도구 기준; model config 기본도 50
NMS_RADIUS=0

EVAL_SAMPLES=0                    # 전체 5119
SAVE_RAW=1                        # query별 cls/depth dump(전체 val도 수십 MB 수준)
SAVE_VIS=1                        # 기존 query/mixture/attention 시각화 저장
VIS_EVERY=48                      # 기존 eval과 동일하게 48샘플마다 1회 저장
RUN_SWEEP=0                       # 1이면 저장된 mixture로 beta/FG-budget offline sweep
SWEEP_BETAS=0,0.1,0.2,0.3,0.5
SWEEP_BASE_FG_THRESHOLDS=0.95,0.90,0.85

# =============================================================================
# Runtime
# =============================================================================
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT_DIR"

ENV_BIN=/home/cvlab/anaconda3/envs/eof/bin
export PATH="$ENV_BIN:$PATH"
export PYTHONPATH="$ROOT_DIR"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS=4

RUN_TAG=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT_DIR/work_dirs/_analysis/eval_depth/$RUN_TAG"
EVAL_DUMP_DIR="$RUN_DIR/eval_dump"
mkdir -p "$RUN_DIR"

if [[ "$SAVE_RAW" == "1" || "$RUN_SWEEP" == "1" ]]; then
    mkdir -p "$EVAL_DUMP_DIR"
    export EOCF_EVAL_DEPTH_DUMP_DIR="$EVAL_DUMP_DIR"
else
    unset EOCF_EVAL_DEPTH_DUMP_DIR || true
fi
if [[ "$RUN_SWEEP" == "1" ]]; then
    export EOCF_EVAL_DEPTH_SAVE_MIXTURE=1
else
    export EOCF_EVAL_DEPTH_SAVE_MIXTURE=0
fi

echo "[eval_depth] depth-score eval"
PORT="$PORT" \
EOCF_EVAL_GT_ROOT="$GT_ROOT" \
EOCF_EVAL_MODE=1 \
EOCF_EVAL_MAX_SAMPLES="$EVAL_SAMPLES" \
EOCF_EVAL_VIS="$SAVE_VIS" \
EOCF_EVAL_VIS_EVERY="$VIS_EVERY" \
EOCF_EVAL_VIS_DIR="$RUN_DIR/eval_vis" \
EOCF_EVAL_ORACLE_MATCH=0 \
EOCF_EVAL_NMS_RADIUS="$NMS_RADIUS" \
EOCF_EVAL_OCC_THR="$OCC_THRESHOLD" \
EOCF_EVAL_FG_THR="$FG_THRESHOLD" \
EOCF_EVAL_DEPTH_SCORE=1 \
EOCF_EVAL_DEPTH_BETA="$DEPTH_BETA" \
EOCF_EVAL_DEPTH_RENDER_WEIGHT="$RENDER_WEIGHT" \
USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
    --cfg-options data.workers_per_gpu=0

if [[ "$RUN_SWEEP" == "1" ]]; then
    echo "[eval_depth] offline AABB score sweep (no model forward)"
    "$ENV_BIN/python" tools/dbg_probe/eval_depth_sweep.py "$EVAL_DUMP_DIR" \
        --gt-root "$GT_ROOT" \
        --output-dir "$RUN_DIR" \
        --betas "$SWEEP_BETAS" \
        --base-fg-thresholds "$SWEEP_BASE_FG_THRESHOLDS" \
        --occ-threshold "$OCC_THRESHOLD" \
        --topk "$TOPK" \
        --render-modes cls,combined
fi

echo "[eval_depth] done: $RUN_DIR"
