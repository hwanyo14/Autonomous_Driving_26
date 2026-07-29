#!/usr/bin/env bash
set -euo pipefail

# ===== Oracle(=GT 치팅) eval =====
# confidence score 선택 대신, query를 GT에 Hungarian 매칭해서 고른다(학습과 100% 동일한 cost).
# baseline(eval_total_2.sh)과 동일 config/checkpoint/occ_thr/mode로 두고 선택 방식만 바꿔
# "스코어링이 병목인지" 진단한다. (oracle ≫ baseline → 스코어링 문제 / 비슷 → shape·매칭 문제)

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter.py
CHECKPOINT=./work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter/epoch_15_lss_only.pth
GPUS=8
export PORT=50091

# ----- 평가 동작 (baseline과 동일하게 맞춰 공정 비교) -----
export EOCF_EVAL_MODE=1            # 0=present / 1=future. baseline과 동일하게 future.
export EOCF_EVAL_OCC_THR=0.75      # baseline과 동일 occ threshold.
export EOCF_EVAL_MAX_SAMPLES=768
export EOCF_EVAL_TRAJ_REFINE=1     # trajectory xy-refine head를 추론에도 적용. 0=off(학습 전용) / 1=on
                                   # baseline(eval_total.sh)과 반드시 같은 값으로 둘 것 — 다르면 oracle vs baseline
                                   # 비교가 "선택 방식" 차이가 아니라 궤적 보정 유무 차이까지 섞인다.


# ----- Oracle 선택 (이 스크립트의 핵심) -----
export EOCF_EVAL_ORACLE_MATCH=1    # 1=학습과 100% 동일 매칭으로 override / 0=기존 score 선택

# ----- 시각화 (baseline과 동일하게 ON; viz도 oracle 선택 기준으로 일치시킴) -----
export EOCF_EVAL_VIS=1             # 1=켜기 / 0=끄기
export EOCF_EVAL_VIS_EVERY=48      # 몇 샘플마다 저장 (baseline과 동일)
# oracle 결과는 baseline과 섞이지 않게 별도 폴더(eval_oracle/<timestamp>)에 저장.
TS=$(date +%Y%m%d_%H%M%S)
export EOCF_EVAL_VIS_DIR=./work_dirs/eval_oracle/${TS}

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS"
