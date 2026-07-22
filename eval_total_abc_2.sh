#!/usr/bin/env bash
set -euo pipefail

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new.py
CHECKPOINT=./work_dirs/full_attn_cover_pyr_aabb_dice3d_new/epoch_15_lss_only.pth
GPU_ID=0
GPUS=1
export PORT=50032

# ----- 실행 환경 (Blackwell + CPU 메모리 절약) -----
ENV_BIN=/home/cvlab/anaconda3/envs/eof/bin
export PATH="$ENV_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4

# ----- 평가 GT / 기준 프레임 -----
export EOCF_EVAL_GT_ROOT=/home/cvlab/Desktop/NIPS2026/datasets/efficientocf_gt_f3/GMO
export EOCF_EVAL_MODE=1             # 1=future 1~4 / 0=present 1프레임
export EOCF_EVAL_NMS_RADIUS=0       # 0=distance NMS 끄기

# ----- threshold / 렌더 가중치 (필요한 항목만 주석 해제) -----
export EOCF_EVAL_OCC_THR=0.9      # occupancy 점유 threshold; config 값 override
export EOCF_EVAL_FG_THR=0.75      # foreground(query) score threshold; config 값 override
export EOCF_EVAL_DEPTH_SCORE=1     # final= sigmoid(logit(cls) + beta*depth_confidence), 0:기존, 1: depth score 사용
export EOCF_EVAL_DEPTH_BETA=1.0
export EOCF_EVAL_DEPTH_RENDER_WEIGHT=combined # selection과 렌더 모두 combined score 사용
# export EOCF_EVAL_WEIGHT_MODE=ones # ones=렌더 가중치 1.0 고정 / 주석 상태=config 기본 score

# ----- 중간 메트릭 / 평가 개수 -----
export EOCF_EVAL_METRIC_EVERY=200   # 10, 200, 400, ..., 마지막에 출력
export EOCF_EVAL_MAX_SAMPLES=0     # 0=전체 5119 / 예: 400=400개 조기 종료

# 출력 메트릭:
#   IoU 2D/3D: nuScenes-Occ, AABB, Rot
#   Recall3d AABB/Rot: macro, micro, TP/FP/FN/bboxFP
#   AABB macro: Strict 및 A/B/C × Plain/Alpha/Density

# ----- 시각화 -----
export EOCF_EVAL_VIS=1              # 1=기존 eval_total처럼 저장 / 0=끄기
export EOCF_EVAL_VIS_EVERY=48       # 몇 샘플마다 2D query + 3D mixture 시각화 저장
# export EOCF_EVAL_VIS_DIR=./work_dirs/custom_eval_vis

# dist_test.sh가 자동으로 아래에 로그를 저장함:
#   logs/<config 이름>/eval/<실행시각>.log
USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
  --cfg-options data.workers_per_gpu=0
