#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=4,5,6,7 \
PORT=22036 \
./run.sh \
./projects/configs/baselines/test_traj_shape.py \
4 \
# --resume /home/hwanhee/Autonomous_Driving_26_cost/work_dirs/attn_suppress_mat_q200/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
