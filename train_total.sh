#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=22036 \
./run.sh \
./projects/configs/baselines/test_traj_shape.py \
8 \
--resume /NHNHOME/WORKSPACE/0526040009_A/hwanhee/Autonomous_Driving_26_ksh_shape/work_dirs/test_traj_shape/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
