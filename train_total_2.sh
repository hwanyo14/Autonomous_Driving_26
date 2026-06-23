#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_MPS=0 \
PORT=27659 \
./run.sh \
./projects/configs/baselines/test_traj_large_tf_simple_weight_bal.py \
8
# --resume work_dirs/test_traj/epoch_9_lss_only.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
