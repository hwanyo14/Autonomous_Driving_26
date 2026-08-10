#!/usr/bin/env bash
set -uo pipefail   # ⚠ -e 는 쓰지 않는다. 한 조합이 실패해도 나머지를 계속 돌려야 한다.

# q900(ep15) fg × occ 스윕. 800샘플, 순차 실행(동시 1개).
#
# 🔴 이 머신에서는 다른 학습이 8 GPU 로 돌고 있다. 이 스크립트는 프로세스를 **절대 죽이지 않는다**.
#    kill/pkill 을 쓰지 말 것. 중단하려면 아래 STOP 파일을 만들면 현재 런을 마치고 멈춘다:
#      touch work_dirs/_sweep_q900/STOP
#
# 결과: work_dirs/_sweep_q900/<TAG>/eval_metrics_live.log  (48샘플마다 한 줄)
#       work_dirs/_sweep_q900/<TAG>.out                    (해당 런의 콘솔 전체)
#       완주 판정은 "[eval][800]" + "[FINAL]" 이 있는 줄로 한다.
#       (중도 사망한 런도 [eval][48] 같은 줄을 남기므로 부분완료를 완주로 오인하지 말 것)

CONFIG=./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_q900_feat128.py
CHECKPOINT=/home/hwanhee/Autonomous_Driving_26_filter_ablation_query/work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_q900_feat128/epoch_15_lss_only.pth

FG_LIST=${FG_LIST:-"0.85 0.9 0.95"}
OCC_LIST=${OCC_LIST:-"0.75 0.85 0.9"}
MAXS=${MAXS:-800}
# ⚠ EOCF_EVAL_METRIC_EVERY 는 **iteration 기준**이다 (test.py: (i+1) % metric_every).
#   8 rank 면 iteration 1회 = 샘플 8개라, 글로벌 N샘플마다 찍으려면 N/GPUS 를 줘야 한다.
EVERY_GLOBAL=${EVERY_GLOBAL:-48}   # 글로벌 샘플 기준 출력 간격
GPUS=${GPUS:-8}               # rank 수. 학습이 8장을 쓰지만 장당 ~62GB, RAM 443GB 여유가 있다.
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
# ⚠ world_size 를 바꾸면 DistributedGroupSampler 의 분할이 달라져 MAX_SAMPLES 로 자른
#   부분집합이 통째로 바뀐다. GPUS 가 다른 런끼리는 수치를 비교하면 안 된다.
PORT_BASE=${PORT_BASE:-50730}
COOLDOWN=${COOLDOWN:-180}     # 런 사이 3분 (EVAL_SPEC §4 자원 규칙)

SW=work_dirs/_sweep_q900
mkdir -p "$SW"

# dist_test.sh 는 `python` 을 그대로 부른다. conda 가 활성화되지 않은 셸에서 실행하면
# base 의 python 이 잡혀 `No module named torch` 로 죽으므로 env 의 bin 을 앞에 붙인다.
CONDA_ENV_BIN=${CONDA_ENV_BIN:-/home/hwanhee/anaconda3/envs/eo/bin}
[ -x "$CONDA_ENV_BIN/python" ] || { echo "[sweep] python 없음: $CONDA_ENV_BIN/python"; exit 1; }

EVERY=$(( EVERY_GLOBAL / GPUS ))
[ "$EVERY" -ge 1 ] || EVERY=1
echo "[sweep] METRIC_EVERY=$EVERY (iteration) = 글로벌 $(( EVERY * GPUS )) 샘플마다"

[ -f "$CONFIG" ]     || { echo "[sweep] config 없음: $CONFIG"; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "[sweep] checkpoint 없음: $CHECKPOINT"; exit 1; }

done_p () {  # $1=TAG — 목표 샘플수와 정확히 일치하는 FINAL 줄이 있어야 완주
  [ -f "$SW/$1.out" ] && grep -qa "\[eval\]\[$MAXS\].*\[FINAL\]" "$SW/$1.out"
}

n_total=$(( $(echo $FG_LIST | wc -w) * $(echo $OCC_LIST | wc -w) ))
n_i=0
echo "[sweep] 시작 — 조합 ${n_total}개, MAXS=$MAXS, 글로벌 ${EVERY_GLOBAL}샘플마다 출력, GPUS=$GPUS ($GPU_IDS)"
echo "[sweep] ckpt=$(basename "$CHECKPOINT")"

for FG in $FG_LIST; do
  for OCC in $OCC_LIST; do
    n_i=$((n_i + 1))
    TAG="fg${FG}_occ${OCC}"

    if [ -f "$SW/STOP" ]; then echo "[sweep] STOP 파일 감지 — 중단"; exit 0; fi
    if done_p "$TAG"; then echo "[sweep] ($n_i/$n_total) $TAG 이미 완료 — skip"; continue; fi

    echo "[sweep] ($n_i/$n_total) $TAG 시작  $(date +%H:%M:%S)"
    rm -f "$SW/$TAG.out"
    mkdir -p "$SW/$TAG"
    : > "$SW/$TAG/eval_metrics_live.log"

    env \
      PATH="$CONDA_ENV_BIN:$PATH" \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      PORT=$((PORT_BASE + n_i)) \
      USE_MPS=0 \
      EOCF_EVAL_MODE=1 \
      EOCF_EVAL_FG_THR="$FG" \
      EOCF_EVAL_OCC_THR="$OCC" \
      EOCF_EVAL_NMS_RADIUS=0 \
      EOCF_EVAL_WEIGHT_MODE=ones \
      EOCF_EVAL_MAX_SAMPLES="$MAXS" \
      EOCF_EVAL_METRIC_EVERY="$EVERY" \
      EOCF_EVAL_VIS=0 \
      EOCF_EVAL_VIS_DIR="$SW/$TAG" \
      bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" > "$SW/$TAG.launch" 2>&1
    rc=$?

    # dist_test.sh 는 콘솔을 logs/<config>/eval/<ts>.log 로 리다이렉트한다 — 그 파일을 회수한다.
    RUN_LOG=$(grep -oE "logs/[^ ]+\.log" "$SW/$TAG.launch" | tail -1)
    if [ -n "$RUN_LOG" ] && [ -f "$RUN_LOG" ]; then cp "$RUN_LOG" "$SW/$TAG.out"; fi

    if done_p "$TAG"; then
      echo "[sweep] ($n_i/$n_total) $TAG 완주  $(grep -a '\[FINAL\]' "$SW/$TAG.out" | tail -1)"
    else
      echo "[sweep] ($n_i/$n_total) $TAG 실패(rc=$rc). 로그: $SW/$TAG.out / $SW/$TAG.launch"
    fi

    if [ "$n_i" -lt "$n_total" ]; then
      echo "[sweep] 쿨다운 ${COOLDOWN}s"
      sleep "$COOLDOWN"
    fi
  done
done

echo "[sweep] 전체 종료 $(date +%H:%M:%S)"
echo "===== 요약 ====="
for FG in $FG_LIST; do for OCC in $OCC_LIST; do
  TAG="fg${FG}_occ${OCC}"
  if done_p "$TAG"; then printf "  %-18s %s\n" "$TAG" "$(grep -a '\[FINAL\]' "$SW/$TAG.out" | tail -1)"
  else printf "  %-18s (미완주)\n" "$TAG"; fi
done; done
