#!/usr/bin/env bash
set -euo pipefail

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new.py
CHECKPOINT=./work_dirs/full_attn_cover_pyr_aabb_dice3d_new/epoch_13_lss_only.pth
GPU_ID=0
GPUS=1
export PORT=50033

# ----- 실행 환경 (Blackwell + CPU 메모리 절약) -----
ENV_BIN=/home/cvlab/anaconda3/envs/eof/bin
export PATH="$ENV_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4

# ----- Oracle 평가 / GT / 기준 프레임 -----
export EOCF_EVAL_ORACLE_MATCH=1     # GT Hungarian 매칭 query만 선택 (진단용)
export EOCF_EVAL_GT_ROOT=/home/cvlab/Desktop/NIPS2026/datasets/efficientocf_gt_f3/GMO
export EOCF_EVAL_MODE=1             # 1=future 1~4 / 0=present 1프레임
export EOCF_EVAL_NMS_RADIUS=0       # oracle에서는 query score/NMS/top-k를 사용하지 않음

# ----- occupancy threshold / 평가 개수 -----
export EOCF_EVAL_OCC_THR=0.9
export EOCF_EVAL_METRIC_EVERY=200
export EOCF_EVAL_MAX_SAMPLES=5119

# ----- Oracle 시각화 -----
# query_debug/mixture3d/cam-gaussian 모두 Hungarian-matched query 기준.
export EOCF_EVAL_VIS=1
export EOCF_EVAL_VIS_EVERY=48

# 출력 메트릭:
#   IoU 2D/3D: nuScenes-Occ, AABB, Rot
#   Recall3d AABB/Rot: macro, micro, TP/FP/FN/bboxFP
#   AABB macro: Strict 및 A/B/C × Plain/Alpha/Density

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
  --cfg-options data.workers_per_gpu=0
