#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_MPS=0 \
PORT=27644 \
./run.sh \
./projects/configs/baselines/test_traj_large_adj_lr_crop_margin.py \
8
# --resume work_dirs/test_traj/epoch_8_lss_only.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
