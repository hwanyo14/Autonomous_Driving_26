#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=7,0,1,2,3,4,5,6 \
PORT=40004 \
USE_MPS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run.sh \
./projects/configs/baselines/subset_attn_cover.py \
8 \
# --resume ./work_dirs/subset_attn_cover/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
