#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ===== 콘솔 출력을 logs/<config명>_<날짜시간>.log 파일로 저장 (터미널 출력 막힘=hang 방지) =====
# 출력이 터미널이 아니라 파일로 가므로 VSCode 터미널이 멈춰도 학습은 안 멈춤.
# 진행상황 보려면: tail -f <아래 경로>
LOG_DIR="logs/$(basename "$CONFIG" .py)"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
echo "[dist_train] console logging -> $RUN_LOG  (실시간 보기: tail -f $RUN_LOG)"
exec >> "$RUN_LOG" 2>&1

# ===== MPS 상시적용 (멀티잡 시 빈 SM 채워 잡당 3x↑; 끄려면 앞에 USE_MPS=0) =====
# ⚠️ MPS는 컨텍스트 공유라, 한 잡이 치명적 GPU 오류로 죽으면 같은 GPU의 다른 잡도 영향받을 수 있음.
# ⚠️ 두 잡 동시 실행 시 PORT/--work-dir 는 잡마다 다르게 줄 것 (MPS가 자동으로 안 챙겨줌).
USE_MPS=${USE_MPS:-1}
if [ "$USE_MPS" = "1" ]; then
  export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
  export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[dist_train] MPS daemon already running (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
  else
    nvidia-cuda-mps-control -d && echo "[dist_train] MPS daemon started (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
  fi
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 2 \
    --launcher pytorch ${@:3}
    # --resume work_dirs/EfficientOCF_V1.1_1gpu/epoch_4_lss_only.pth