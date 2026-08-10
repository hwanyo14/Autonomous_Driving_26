#!/usr/bin/env bash
set -euo pipefail

# train-calibrated Depth+Speed query 재선택 eval (traincal 전용).
#   s_eff = sigmoid( logit(s) - a_depth*z_depth - a_speed*z_speed ),  컷: s_eff >= CUTOFF
#   z = clip( (log1p(feature) - MEAN) / STD , -3, 3 )   ← MEAN/STD 는 train 4,000 고정 상수
#
# 기존 fg 컷(score>=fg_thr)을 **대체**한다. rescue 가 아니라 global reselection 이라
# 원래 score>=0.9 였던 query 도 보정 후 컷 아래로 내려가면 탈락한다.
# baseline(score>=fg_thr) 대조군은 eval_total.sh 로 뽑는다.

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter.py
CHECKPOINT=./work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter/epoch_18_lss_only.pth
GPUS=1
export PORT=50607

# ----- 평가 동작 -----
export EOCF_EVAL_MODE=1            # 기준 프레임: 0=present / 1=future. metric 은 항상 둘 다 계산됨
export EOCF_EVAL_OCC_THR=0.80      # 0.75 대비 전 구간 우세(2026-08-03 실측). config eval_occ_threshold override
export EOCF_EVAL_NMS_RADIUS=0      # distance NMS 반경(m). 0=off / 주석처리=config 3.0m
export EOCF_EVAL_WEIGHT_MODE=ones  # 렌더 가중치 1.0 고정(fg/occ 독립화). 기본=score, 1고정=ones
export EOCF_EVAL_MAX_SAMPLES=0     # 부분 eval: N샘플 후 조기 종료. 0=5119 전체
export EOCF_EVAL_METRIC_EVERY=48   # 중간 metric 출력 주기. eval_total.sh 런들과 같은 N 에서 비교하려면 48

# ⚠ 이 줄을 **지우지 말 것**. 코드 기본값이 5.0 이라 줄을 지우면 TRAJ_CUT 이 켜진다.
# traincal 에서는 적용하지 않는다(EVAL_METHOD.md 고정조건에 없음).
export EOCF_EVAL_FG_TRAJ_ACC_MAX=0

# ----- Depth+Speed 가중치와 컷 -----
export EOCF_EVAL_TRAINCAL_CUTOFF=0.9        # s_eff 컷. 원래 score 기준 실효 범위는 0.536~0.988
export EOCF_EVAL_TRAINCAL_ALPHA_DEPTH=0.5   # z_depth 가중치. depth 가 speed 보다 2배 세다
export EOCF_EVAL_TRAINCAL_ALPHA_SPEED=0.25  # z_speed 가중치

# ----- Depth+Speed train 통계 (log1p population mean/std) -----
# 출처: depth_std+past_speed_calibration/normalization_stats.json
#       train split 4,000 sample / 모든 predicted foreground query 136,973개 / GT 미사용.
# ⚠ checkpoint 종속이다. 다른 checkpoint 로 평가하면 z-score 가 통째로 틀어지는데 에러는 안 난다.
#   checkpoint 를 바꾸면 반드시 calibration 을 다시 뽑을 것 (NOTES.md 2026-08-03 ①).
export EOCF_EVAL_TRAINCAL_MEAN_DEPTH=1.1490066687298401
export EOCF_EVAL_TRAINCAL_STD_DEPTH=0.2715431724116579
export EOCF_EVAL_TRAINCAL_MEAN_SPEED=1.5370710560583838
export EOCF_EVAL_TRAINCAL_STD_SPEED=0.6923375815042205
# JSON 에서 읽고 싶을 때만 주석 해제. 위 4개가 설정돼 있으면 그쪽이 이긴다.
# export EOCF_EVAL_TRAINCAL_STATS=./depth_std+past_speed_calibration/normalization_stats.json

# ----- 시각화 -----
export EOCF_EVAL_VIS=0             # 1=켜기 / 0=끄기. 1이면 GPU 메모리 약 2배
export EOCF_EVAL_VIS_EVERY=48      # 몇 샘플마다 저장
# export EOCF_EVAL_VIS_DIR=...     # 저장 경로. 미설정 시 ./work_dirs/<config>/eval/<timestamp>

echo "[eval_depth_speed] occ=$EOCF_EVAL_OCC_THR cutoff=$EOCF_EVAL_TRAINCAL_CUTOFF" \
     "alpha=$EOCF_EVAL_TRAINCAL_ALPHA_DEPTH/$EOCF_EVAL_TRAINCAL_ALPHA_SPEED" \
     "traj_cut=$EOCF_EVAL_FG_TRAJ_ACC_MAX ckpt=$(basename "$CHECKPOINT")"

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS"
