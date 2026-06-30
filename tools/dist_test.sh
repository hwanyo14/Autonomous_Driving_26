#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29501}

LOG_DIR="logs/$(basename "$CONFIG" .py)/eval"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
echo "[dist_test] console logging -> $RUN_LOG  (tail -f $RUN_LOG)"
exec >> "$RUN_LOG" 2>&1

if [[ "${EOCF_EVAL_VIS:-0}" == "1" && -z "${EOCF_EVAL_VIS_DIR:-}" ]]; then
    export EOCF_EVAL_TIMESTAMP=${EOCF_EVAL_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
fi

USE_MPS=${USE_MPS:-1}
if [ "$USE_MPS" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
  export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[dist_test] MPS daemon already running (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
  else
    nvidia-cuda-mps-control -d && echo "[dist_test] MPS daemon started (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
  fi
fi

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox
