#!/usr/bin/env bash
set -euo pipefail

# feat128 (입력 896x1600 + bev_feat_dim 128, query 200) — train-calibrated query 재선택 eval.
#   s_eff = sigmoid( logit(s) - a_depth*z_depth - a_speed*z_traj_dev ),  컷: s_eff >= CUTOFF
#   z = clip( (log1p(feature) - MEAN) / STD , -3, 3 )   ← MEAN/STD 는 train 통계 고정 상수
#
# 기존 fg 컷(score>=fg_thr)을 **대체**한다. rescue 가 아니라 global reselection 이라
# 원래 score>=0.9 였던 query 도 보정 후 컷 아래로 내려가면 탈락한다.
# baseline(score>=fg_thr) 대조군은 eval_total.sh 로 뽑는다 (같은 config·같은 ckpt 로 맞출 것).

# ----- 대상 config / checkpoint / GPU -----
# checkpoint 는 Autonomous_Driving_26_filter 레포에 있다. ep19 가 best.
#   ep16~20 은 ep15 에서 resume 한 `_feat128_ep20` work_dir 에 들어 있다(`_feat128` 쪽은 ep1~15).
CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_feat128.py
CHECKPOINT=/home/hwanhee/Autonomous_Driving_26_filter/work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_feat128_ep20/epoch_19_lss_only.pth
GPUS=1
export PORT=50722

# ----- 평가 동작 -----
export EOCF_EVAL_MODE=1            # 기준 프레임: 0=present / 1=future. metric 은 항상 둘 다 나온다
export EOCF_EVAL_OCC_THR=0.85      # best. config occ_score_threshold override
export EOCF_EVAL_NMS_RADIUS=0      # distance NMS 반경(m). 0=off / 주석처리=config 3.0m
export EOCF_EVAL_WEIGHT_MODE=ones  # 렌더 가중치 1.0 고정(fg/occ 독립화). 기본=score, 1고정=ones
export EOCF_EVAL_MAX_SAMPLES=0     # 부분 eval: N샘플 후 조기 종료. 0=5119 전체
export EOCF_EVAL_METRIC_EVERY=48   # 중간 metric 출력 주기

# ----- 재선택 가중치와 컷 -----
export EOCF_EVAL_TRAINCAL_FEAT2=traj_dev    # 운동항 = 등속모델 잔차. ⚠ 생략하면 폐기된 past_speed 가 쓰인다
export EOCF_EVAL_TRAINCAL_CUTOFF=0.9        # s_eff 컷 (= fg 임계값 역할). best
export EOCF_EVAL_TRAINCAL_ALPHA_DEPTH=0.9   # z_depth 가중치. best
export EOCF_EVAL_TRAINCAL_ALPHA_SPEED=0.4   # z_traj_dev 가중치. best

# ----- train 통계 (log1p population mean/std) -----
# ⚠ traj_dev 키가 든 JSON 이어야 한다. traincal/ 21개 중 *_v3_*.json 2개만 해당하고,
#   나머지를 주면 에러 없이 baseline 으로 조용히 fallback 한다.
# 🔴 checkpoint 종속이다. 위 CHECKPOINT 를 바꾸면 이 통계도 반드시 다시 뽑을 것:
#      CFG_NAME=full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_feat128 EP=<에폭> \
#      OUT=./traincal/feat128_ep<에폭>_local.json MAXS=4000 bash collect_traincal_stats.sh
#    표본 4,000 이면 충분하다 (EVAL_SPEC §2-3).
#   ※ 이 JSON 의 checkpoint 필드는 다른 머신 경로(/home/cvlab/.../_feat128/epoch_19)를 가리키지만
#     같은 ep19 run 이다. 위 CHECKPOINT 와 이 config 로 load_state_dict 가
#     missing=0 / shape_mismatch=0 으로 통과함을 확인했다(ckpt meta epoch=19).
export EOCF_EVAL_TRAINCAL_STATS=./traincal/feat128_ep19_v3_n4000.json

# ⚠ 상수 env 가 JSON 을 이긴다. 옛 값이 셸에 남아 있으면 위 JSON 이 무시되므로 반드시 unset.
unset EOCF_EVAL_TRAINCAL_MEAN_DEPTH EOCF_EVAL_TRAINCAL_STD_DEPTH \
      EOCF_EVAL_TRAINCAL_MEAN_SPEED EOCF_EVAL_TRAINCAL_STD_SPEED 2>/dev/null || true

# ----- 시각화 -----
export EOCF_EVAL_VIS=0             # 1=켜기 / 0=끄기. 1이면 GPU 메모리 약 2배
export EOCF_EVAL_VIS_EVERY=48      # 몇 샘플마다 저장
# export EOCF_EVAL_VIS_DIR=...     # 저장 경로. 미설정 시 ./work_dirs/<config>/eval/<timestamp>

[ -f "$CHECKPOINT" ] || { echo "[eval_feat128] checkpoint 없음: $CHECKPOINT"; exit 1; }
[ -f "$EOCF_EVAL_TRAINCAL_STATS" ] || { echo "[eval_feat128] 통계 JSON 없음: $EOCF_EVAL_TRAINCAL_STATS"; exit 1; }
echo "[eval_feat128] ckpt=$(basename "$CHECKPOINT") occ=$EOCF_EVAL_OCC_THR cutoff=$EOCF_EVAL_TRAINCAL_CUTOFF" \
     "feat2=$EOCF_EVAL_TRAINCAL_FEAT2 alpha=$EOCF_EVAL_TRAINCAL_ALPHA_DEPTH/$EOCF_EVAL_TRAINCAL_ALPHA_SPEED" \
     "stats=$(basename "$EOCF_EVAL_TRAINCAL_STATS")"
echo "[eval_feat128] ⚠ 로그에 '[eval] traincal depth+traj_dev selection ON' 이 안 뜨면"
echo "                 통계 로드 실패로 baseline 으로 조용히 되돌아간 것이다."

USE_MPS=0 \
bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS"
