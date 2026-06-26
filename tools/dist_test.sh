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

# ===== MPS 상시적용 (멀티 eval 동시 실행 시 GPU SM 공유; 끄려면 USE_MPS=0) =====
# ⚠️ MPS는 컨텍스트 공유라, 한 eval이 치명적 GPU 오류로 죽으면 같은 GPU의 다른 eval도 영향받을 수 있음.
# ⚠️ 동시 eval 시 PORT는 잡마다 다르게 줄 것 (eval_total.sh: 20003, eval_total_2.sh: 20014 등).
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

# 멀티 eval 동시 실행 시 샘플당 GPU op이 2× 느려져 NCCL watchdog 600s 기본값을 초과할 수 있음.
# NCCL watchdog를 2시간으로 늘려 timeout SIGABRT 방지.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox
