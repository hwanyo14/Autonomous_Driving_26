#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=3,4,5,6,7,0,1,2 \
PORT=40003 \
USE_MPS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run.sh \
./projects/configs/baselines/subset_attn_cover_aabb.py \
8 \
--resume ./work_dirs/subset_attn_cover_aabb/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
