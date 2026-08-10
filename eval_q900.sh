#!/usr/bin/env bash
set -euo pipefail

# [ablation] q900 — query_num_queries 200 -> 900
#   main run(feat128, 200) 대비 query 수만 4.5배. 코드 변경 없이 config 파라미터만 다르다.
#   checkpoint 는 ablation_query 레포에 있고 config 만 이 레포에 있다.

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_q900_feat128.py
CHECKPOINT=/home/hwanhee/Autonomous_Driving_26_filter_ablation_query/work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_q900_feat128/epoch_15_lss_only.pth
GPUS=1
export PORT=50713

# ----- 평가 동작 (baseline 과 비교하려면 eval_total.sh 와 같은 값으로 둘 것) -----
export EOCF_EVAL_MODE=1            # 기준 프레임: 0=present(현재 1) / 1=future(미래 n_future). metric 은 항상 둘 다 나온다
export EOCF_EVAL_OCC_THR=0.75      # occ 점유 threshold. config occ_score_threshold override
export EOCF_EVAL_FG_THR=0.9        # foreground(query) score threshold
export EOCF_EVAL_NMS_RADIUS=0      # distance NMS 반경(m). 0=off / 주석처리=config 3.0m
export EOCF_EVAL_WEIGHT_MODE=ones  # 렌더 가중치 1.0 고정(fg/occ 독립화). 기본=score, 1고정=ones
export EOCF_EVAL_MAX_SAMPLES=0     # 부분 eval: N샘플 후 조기 종료. 0=5119 전체
export EOCF_EVAL_METRIC_EVERY=48   # 중간 metric 출력 주기

# ----- 시각화 -----
export EOCF_EVAL_VIS=0             # 1=켜기 / 0=끄기
export EOCF_EVAL_VIS_EVERY=48      # 몇 샘플마다 저장
# export EOCF_EVAL_VIS_DIR=...     # 저장 경로. 미설정 시 ./work_dirs/<config>/eval/<timestamp>

[ -f "$CHECKPOINT" ] || { echo "[eval_q900] checkpoint 없음: $CHECKPOINT"; exit 1; }
echo "[eval_q900] ckpt=$(basename "$CHECKPOINT") occ=$EOCF_EVAL_OCC_THR fg=$EOCF_EVAL_FG_THR maxs=$EOCF_EVAL_MAX_SAMPLES"

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS"
