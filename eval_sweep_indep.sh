#!/usr/bin/env bash
# 실험 스윕: ① NMS off ② 렌더 가중치 독립화(weights=1) — 각 단독, 512샘플 순차 (NOTES 2026-07-06 스윕 후속)
#  A  : NMS=0        (weight=score, fg0.75/occ0.75 = 현 운영점에서 NMS만 끔)
#  B1~3: weights=ones (fg0.75 고정, occ 0.3/0.5/0.75 미니 스윕 — footprint 반경 1.55σ/1.18σ/0.76σ)
set -uo pipefail

CONFIG=./projects/configs/baselines/full.py
CHECKPOINT=./work_dirs/full/epoch_11_lss_only.pth
GPUS=8
export PORT=50036

export EOCF_EVAL_MODE=1
export EOCF_EVAL_MAX_SAMPLES=512
export EOCF_EVAL_VIS=1
export EOCF_EVAL_VIS_EVERY=48

# "TAG FG OCC NMS WEIGHT"
RUNS=(
  "nms0_fg0.75_occ0.75      0.75 0.75 0   score"
  "ones_fg0.75_occ0.3       0.75 0.3  -   ones"
  "ones_fg0.75_occ0.5       0.75 0.5  -   ones"
  "ones_fg0.75_occ0.75      0.75 0.75 -   ones"
)

# SWEEP2_WAIT=1일 때만 타 eval 종료 대기 (기본: 바로 얹어서 병행 실행)
if [ "${SWEEP2_WAIT:-0}" = "1" ]; then
  while pgrep -f "tools/test.py" > /dev/null; do
    echo "[sweep2] waiting for running eval... ($(date +%H:%M:%S))"
    sleep 120
  done
fi

CKPT_TAG=$(basename "$CHECKPOINT" .pth)
for r in "${RUNS[@]}"; do
  read -r TAG FG OCC NMS WEIGHT <<< "$r"
  export EOCF_EVAL_FG_THR=$FG
  export EOCF_EVAL_OCC_THR=$OCC
  export EOCF_EVAL_WEIGHT_MODE=$WEIGHT
  if [ "$NMS" = "-" ]; then unset EOCF_EVAL_NMS_RADIUS; else export EOCF_EVAL_NMS_RADIUS=$NMS; fi
  DIR=./work_dirs/full/eval_sweep/${CKPT_TAG}_${TAG}
  export EOCF_EVAL_VIS_DIR=$DIR
  mkdir -p "$DIR"
  echo "=== [sweep2] $TAG start $(date +%H:%M:%S) (fg=$FG occ=$OCC nms=$NMS w=$WEIGHT) ==="
  USE_MPS=0 bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" > "$DIR/stdout.log" 2>&1
  echo "=== [sweep2] $TAG done $(date +%H:%M:%S) ==="
  grep "FINAL\|recall3d comps" "$DIR/eval_metrics_live.log" 2>/dev/null || true
done
echo "[sweep2] ALL DONE"
