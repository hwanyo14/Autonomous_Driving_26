#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_MPS=0 \
PORT=27658 \
./run.sh \
./projects/configs/baselines/test_traj_resnet_lr.py \
8 \
--resume work_dirs/test_traj_resnet_lr/epoch_1_lss_only.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
