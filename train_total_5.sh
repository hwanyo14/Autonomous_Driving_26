#!/usr/bin/env bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=6,7,0,1,2,3,4,5 \
PORT=40005 \
USE_MPS=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run.sh \
./projects/configs/baselines/subset_attn_cover_size.py \
8
# size note 실험 — from-scratch 전용 (head 파라미터 변경, resume/load_from 금지)
