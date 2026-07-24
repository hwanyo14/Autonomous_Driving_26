#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=6,7,0,1,2,3,4,5 \
PATH=/home/hwanhee/anaconda3/envs/eo/bin:$PATH \
PORT=40003 \
USE_MPS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run.sh \
./projects/configs/baselines/subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft_bce.py \
8 \
# --resume work_dirs/subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft_bce/epoch_2_lss_only.pth
