#!/usr/bin/env bash
# NVIDIA MPS 데몬 종료 + 정리 (학습 잡은 건드리지 않음).
# dist_train.sh가 학습 시작 시 자동으로 띄운 MPS 데몬을 끌 때 사용.
# 상세/주의사항: MPS_MULTIJOB_RUNBOOK.md §2-5
set -uo pipefail

export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}

# 1) 정상 종료: pipe dir가 살아있을 때 quit (순서 중요 — quit 먼저, rm 나중)
if echo quit | nvidia-cuda-mps-control 2>/dev/null; then
    echo "[stop_mps] quit sent to MPS control daemon"
else
    echo "[stop_mps] control daemon not reachable (already down?)"
fi

# 2) quit이 안 먹었으면 강제 종료 ([n] 대괄호로 pkill 자기 매칭 방지)
pkill -9 -f "[n]vidia-cuda-mps" 2>/dev/null && echo "[stop_mps] force-killed remaining MPS processes" || true

# 3) 마지막에 정리
rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
echo "[stop_mps] cleaned $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY"

# 확인: 메모리 ~수 MiB, util 0이면 완전 클린
echo "[stop_mps] GPU state:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
