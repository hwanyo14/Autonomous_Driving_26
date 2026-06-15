#!/usr/bin/env bash
set -e

CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}

if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo quit | CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" nvidia-cuda-mps-control || true
fi

pkill -9 -f "[n]vidia-cuda-mps" || true
rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
