#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29501}

# ===== 콘솔 출력을 logs/<config명>/eval/<날짜시간>.log 파일로 저장 (터미널 출력 막힘=hang 방지) =====
# 출력이 터미널이 아니라 파일로 가므로 VSCode 터미널이 멈춰도 평가는 안 멈춤.
# 진행상황 보려면: tail -f <아래 경로>
LOG_DIR="logs/$(basename "$CONFIG" .py)/eval"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
echo "[dist_test] console logging -> $RUN_LOG  (실시간 보기: tail -f $RUN_LOG)"
exec >> "$RUN_LOG" 2>&1

if [[ "${EOCF_EVAL_VIS:-0}" == "1" && -z "${EOCF_EVAL_VIS_DIR:-}" ]]; then
    export EOCF_EVAL_TIMESTAMP=${EOCF_EVAL_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox
