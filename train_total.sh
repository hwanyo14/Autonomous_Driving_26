#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PORT=22006 \
./run.sh \
./projects/configs/baselines/test_traj_mcls.py \
8 \
# --resume /home/hwanhee/Autonomous_Driving_26_cost/work_dirs/attn_suppress_mat_q200/latest.pth
# --cfg-options load_from=/home/hwanhee/Autonomous_Driving_26_0528/epoch_15_lss_only.pth
