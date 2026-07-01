#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 \
PORT=25003 \
USE_MPS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run.sh \
./projects/configs/baselines/shape_guide_128_dice_weight_full.py \
8 \
--resume /NHNHOME/WORKSPACE/0526040009_A/hwanhee/Autonomous_Driving_26_ksh_occ_shape_codeonly/work_dirs/shape_guide_128_dice_weight_full/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
