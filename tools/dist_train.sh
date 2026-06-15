#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ── NVIDIA MPS 자동 적용 (상시) ───────────────────────────────────────
# 한 노드 GPU 공유로 학습을 여러 개 동시에 돌릴 때 시분할 오버헤드를 없애기 위해
# 학습 시작 전에 MPS 데몬을 띄우고, 이 launch의 모든 프로세스를 MPS 클라이언트로 만든다.
# (CUDA 컨텍스트 생성 전에 env가 설정돼야 클라이언트가 됨 → 반드시 torchrun 호출 이전)
# 끄려면 USE_MPS=0 으로 실행. 데몬 종료/정리는 `bash stop_mps.sh`. 상세: MPS_MULTIJOB_RUNBOOK.md
USE_MPS=${USE_MPS:-1}
if [ "$USE_MPS" != "0" ]; then
    export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
    export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if pgrep -f "[n]vidia-cuda-mps-control -d" >/dev/null 2>&1; then
        echo "[dist_train] MPS daemon already running (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
    else
        nvidia-cuda-mps-control -d \
            && echo "[dist_train] MPS daemon started (pipe=$CUDA_MPS_PIPE_DIRECTORY)" \
            || echo "[dist_train] WARN: failed to start MPS daemon; continuing without MPS"
        sleep 1
    fi
fi
# ──────────────────────────────────────────────────────────────────────

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