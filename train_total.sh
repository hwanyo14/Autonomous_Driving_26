#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=25001 \
USE_MPS=0 \
./run.sh \
./projects/configs/baselines/shape_guide_128_dice_weight_fl25_sz06_full.py \
8 \
# --resume /NHNHOME/WORKSPACE/0526040009_A/hwanhee/Autonomous_Driving_26_ksh_occ_shape_codeonly/work_dirs/shape_guide_128_dice_weight_fl25_sz06_full/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
