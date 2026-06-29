#!/usr/bin/env bash
set -euo pipefail

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/shape_guide_128_dice_weight.py
CHECKPOINT=./work_dirs/shape_guide_128_dice_weight/latest.pth
GPUS=8
export PORT=20014

# ----- 평가 동작 -----
export EOCF_EVAL_MODE=1            # 기준 프레임: 0=present(현재 1) / 1=future(미래 n_future). metric·viz 공통
export EOCF_EVAL_OCC_THR=0.75    # occ 점유 threshold. config eval_occ_threshold override
# export EOCF_EVAL_FG_THR=0.75     # foreground(query) score threshold

# ----- 시각화 (2D query_debug_vis + 3D mixture3d) -----
export EOCF_EVAL_VIS=1             # 1=켜기 / 0=끄기
export EOCF_EVAL_VIS_EVERY=48      # 몇 샘플마다 저장
# export EOCF_EVAL_VIS_DIR=...     # 저장 경로. 미설정 시 ./work_dirs/<config>/eval/<timestamp>

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS"
