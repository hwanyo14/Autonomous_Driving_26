#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=6,7,0,1,2,3,4,5 \
PORT=40009 \
USE_MPS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run.sh \
./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_q100_feat128.py \
8 \
--resume ./work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_q100_feat128/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
