#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_lss_pretrain.sh
#   bash run_lss_pretrain.sh <config> <gpus> [extra dist_train args...]

CONFIG=${1:-projects/configs/baselines/EfficientOCF_V1.1_query_lss.py}
GPUS=${2:-1}
shift $(( $# >= 1 ? 1 : 0 ))
shift $(( $# >= 1 ? 1 : 0 ))

if [ ! -f "$CONFIG" ]; then
  echo "Config not found: $CONFIG"
  exit 1
fi

echo "-------------"
echo "LSS pretrain config: $CONFIG"
echo "GPUS: $GPUS"

bash tools/dist_train.sh "$CONFIG" "$GPUS" \
  --cfg-options \
  model.pretrain_view_transform_only=True \
  "$@"
