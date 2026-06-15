#!/usr/bin/env bash
# MPS 데몬 안전 종료 + 정리. (dist_train.sh가 상시적용으로 자동 시작하므로, 끄고 싶을 때 사용)
# 순서 중요: quit → 그 다음 rm. (pipe dir 먼저 지우면 데몬이 orphan 됨)
set -u
PIPE=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
LOG=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}

echo quit | CUDA_MPS_PIPE_DIRECTORY="$PIPE" nvidia-cuda-mps-control 2>/dev/null
sleep 1

if pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1 || pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1; then
  echo "[stop_mps] quit 안 먹음 → 강제 종료"
  pkill -9 -f "[n]vidia-cuda-mps"
  sleep 1
fi

rm -rf "$PIPE" "$LOG"
echo "[stop_mps] MPS 종료/정리 완료. GPU 메모리:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
