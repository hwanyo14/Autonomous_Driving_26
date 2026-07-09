#!/usr/bin/env bash
set -euo pipefail

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full_attn_cover_aabb_dice_pyr.py
CHECKPOINT=./work_dirs/full_attn_cover_aabb_dice_pyr/latest.pth
GPUS=8
export PORT=44031

# ----- 평가 동작 -----
export EOCF_EVAL_MODE=1            # 기준 프레임: 0=present(현재 1) / 1=future(미래 n_future). metric·viz 공통
# export EOCF_EVAL_OCC_THR=0.6    # occ 점유 threshold. config eval_occ_threshold override
# export EOCF_EVAL_FG_THR=0.75     # foreground(query) score threshold
export EOCF_EVAL_NMS_RADIUS=0      # distance NMS 반경(m). 0=off (스윕 승자 A 조합) / 주석처리=config 3.0m
# export EOCF_EVAL_WEIGHT_MODE=ones # 렌더 가중치 1.0 고정(fg/occ 독립화 실험용). 기본=score
# export EOCF_EVAL_MAX_SAMPLES=512 # 부분 eval: N샘플 후 조기 종료 (fixed_val이라 run간 비교 가능)

# ----- 시각화 (2D query_debug_vis + 3D mixture3d) -----
export EOCF_EVAL_VIS=1             # 1=켜기 / 0=끄기
export EOCF_EVAL_VIS_EVERY=48      # 몇 샘플마다 저장
# export EOCF_EVAL_VIS_DIR=...     # 저장 경로. 미설정 시 ./work_dirs/<config>/eval/<timestamp>

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS"
