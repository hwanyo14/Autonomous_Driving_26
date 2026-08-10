#!/usr/bin/env bash
set -euo pipefail

# traincal(depth+speed) 정규화 통계 수집기.
#   train split 을 **deterministic test pipeline** 으로 forward 하면서
#   예측 foreground query(pred_cls != bg)의 depth_std / past_speed 를 모아
#   log1p population mean/std 를 JSON 으로 저장한다. GT 는 쓰지 않는다.
#
# ⚠ checkpoint 종속이다. eval 에 쓸 checkpoint 와 **반드시 동일**해야 한다
#   (다르면 z-score 가 통째로 틀어지는데 에러는 안 난다 — eval_depth_speed.sh 주석 참조).
#
# ⚠ 중간 스냅샷이 EVERY 샘플마다 저장되고, JSON 에 누적합(_acc: n/sum/sumsq)까지 들어간다.
#   → 4,000 지점 스냅샷으로 먼저 eval 을 돌리고, 수집은 23,930 까지 계속 굴려도 된다.
#     나중에 이어붙이려고 처음부터 다시 돌릴 필요가 없다.

CFG_NAME=${CFG_NAME:-full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_feat128}
EP=${EP:-19}
OUT=${OUT:-./traincal_stats/feat128_ep${EP}_live.json}
MAXS=${MAXS:-0}            # 0 = train split 전체(23,930). 4000 등으로 제한 가능
EVERY=${EVERY:-100}        # 스냅샷 주기(샘플). 작을수록 중단 시 손실이 적다
PORT=${PORT:-50909}

CONFIG=./projects/configs/baselines/${CFG_NAME}_traincal_collect.py
CHECKPOINT=./work_dirs/${CFG_NAME}/epoch_${EP}_lss_only.pth
[ -f "$CONFIG" ]     || { echo "[collect] config 없음: $CONFIG"; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "[collect] checkpoint 없음: $CHECKPOINT"; exit 1; }

export PORT
export EOCF_EVAL_MODE=1
export EOCF_EVAL_MAX_SAMPLES=$MAXS
export EOCF_EVAL_METRIC_EVERY=100000        # train split metric 은 의미 없다 — 출력만 줄인다
export EOCF_EVAL_VIS=0
export EOCF_EVAL_NMS_RADIUS=0
export EOCF_EVAL_WEIGHT_MODE=ones
export EOCF_EVAL_FG_TRAJ_ACC_MAX=0          # 수집에는 컷을 적용하지 않는다

# 수집 모드 스위치
export EOCF_EVAL_TRAINCAL_COLLECT="$OUT"
export EOCF_EVAL_TRAINCAL_COLLECT_EVERY=$EVERY
export EOCF_EVAL_TRAINCAL_COLLECT_CKPT="$(readlink -f "$CHECKPOINT")"

# ⚠ 수집 중에는 traincal '선택'이 켜지면 안 된다(통계 없이 재선택하면 의미 없음).
unset EOCF_EVAL_TRAINCAL_STATS EOCF_EVAL_TRAINCAL_MEAN_DEPTH EOCF_EVAL_TRAINCAL_STD_DEPTH \
      EOCF_EVAL_TRAINCAL_MEAN_SPEED EOCF_EVAL_TRAINCAL_STD_SPEED 2>/dev/null || true

echo "[collect] cfg=$CFG_NAME ep=$EP maxs=$MAXS every=$EVERY"
echo "[collect] out=$OUT"
USE_MPS=0 bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" 1
