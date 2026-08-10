#!/usr/bin/env bash
set -euo pipefail

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_feat128.py
CHECKPOINT=./work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_feat128/epoch_16_lss_only.pth
GPUS=1
export PORT=50626

# ----- 평가 동작 -----
export EOCF_EVAL_MODE=1            # 기준 프레임: 0=present(현재 1) / 1=future(미래 n_future). metric·viz 공통
export EOCF_EVAL_OCC_THR=0.85    # occ 점유 threshold. config eval_occ_threshold override
export EOCF_EVAL_FG_THR=0.85     # foreground(query) score threshold
export EOCF_EVAL_NMS_RADIUS=0      # distance NMS 반경(m). 0=off (스윕 승자 A 조합) / 주석처리=config 3.0m
export EOCF_EVAL_WEIGHT_MODE=ones # 렌더 가중치 1.0 고정(fg/occ 독립화 실험용). 기본=score, 1고정=ones
export EOCF_EVAL_MAX_SAMPLES=0 # 부분 eval: N샘플 후 조기 종료 (fixed_val이라 run간 비교 가능)

# ----- TRAJ_CUT: 궤적 가속도 꼬리 제거 (코드 기본값 = 아래 값이므로 평소엔 주석 유지) -----
# 과거 3프레임 중심의 가속도 잔차 |c₂−2c₁+c₀| 가 임계를 넘는 query를 렌더에서 제외한다.
# 물리적으로 불가능한 가속도 = 궤적 추정 실패 신호. 실측(ep18 5119): 미래 FP −10.8%, TP −1%.
# 끄려면 ACC_MAX=0. scope=all 로 두면 현재 프레임에서도 제외(= present 도 −0.0003 손해).
export EOCF_EVAL_FG_TRAJ_ACC_MAX=5.0   # m. 5.0 = 20 m/s²(≈2g) 초과 배제. 0=끄기
export EOCF_EVAL_TRAJ_CUT_SCOPE=future # future=미래 프레임만 적용(권장) / all=query 통째 제거

# ----- 시각화 (2D query_debug_vis + 3D mixture3d) -----
export EOCF_EVAL_VIS=0             # 1=켜기 / 0=끄기
export EOCF_EVAL_VIS_EVERY=48      # 몇 샘플마다 저장
# export EOCF_EVAL_VIS_DIR=...     # 저장 경로. 미설정 시 ./work_dirs/<config>/eval/<timestamp>

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS"
