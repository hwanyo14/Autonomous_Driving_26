#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
USE_MPS=${USE_MPS:-1}

if [ "$USE_MPS" != "0" ]; then
    if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
        export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
        mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

        if ! echo get_default_active_thread_percentage | nvidia-cuda-mps-control >/dev/null 2>&1; then
            if ! nvidia-cuda-mps-control -d; then
                echo "WARNING: failed to start MPS; continuing without MPS." >&2
                unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
            fi
        fi
    else
        echo "WARNING: nvidia-cuda-mps-control not found; continuing without MPS." >&2
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
    # --resume work_dirs/test_traj/epoch_8_lss_only.pth
