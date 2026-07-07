#!/usr/bin/env bash
# threshold 스윕 래퍼 — eval_total.sh와 동일한 dist_test 경로로 (FG_THR, OCC_THR) 조합을
# 순차 실행. 부분 eval(EOCF_EVAL_MAX_SAMPLES)로 짧게 돌려 조합 간 비교 (fixed_val이라
# 같은 N끼리 비교 유효). 다른 eval이 돌고 있으면 끝날 때까지 대기 후 시작.
# 결과: 각 조합의 [FINAL] 줄 → work_dirs/<이름>/eval_sweep/<태그>/eval_metrics_live.log
set -uo pipefail

# ----- 대상 config / checkpoint / GPU -----
CONFIG=./projects/configs/baselines/full.py
CHECKPOINT=./work_dirs/full/epoch_11_lss_only.pth
GPUS=8
export PORT=50033

# ----- 스윕 조합: "FG_THR OCC_THR" -----
COMBOS=(
  "0.75 0.5"
  "0.5 0.5"
  "0.5 0.75"
)

# ----- 공통 평가 동작 -----
export EOCF_EVAL_MODE=1
export EOCF_EVAL_MAX_SAMPLES=512   # 부분 eval (±5% 수렴 지점)
export EOCF_EVAL_VIS=1             # 시각화 ON — 조합별 폴더(eval_sweep/<태그>)에 저장
export EOCF_EVAL_VIS_EVERY=48      # 조합당 ~10샘플 시각화

# 진행 중인 다른 eval 대기
while pgrep -f "tools/test.py" > /dev/null; do
  echo "[sweep] waiting for running eval to finish... ($(date +%H:%M:%S))"
  sleep 120
done

CKPT_TAG=$(basename "$CHECKPOINT" .pth)
for c in "${COMBOS[@]}"; do
  read -r FG OCC <<< "$c"
  export EOCF_EVAL_FG_THR=$FG
  export EOCF_EVAL_OCC_THR=$OCC
  TAG="${CKPT_TAG}_fg${FG}_occ${OCC}"
  export EOCF_EVAL_VIS_DIR=./work_dirs/full/eval_sweep/${TAG}
  mkdir -p "$EOCF_EVAL_VIS_DIR"
  echo "=== [sweep] $TAG start $(date +%H:%M:%S) ==="
  USE_MPS=0 bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
    > "./work_dirs/full/eval_sweep/${TAG}/stdout.log" 2>&1
  echo "=== [sweep] $TAG done $(date +%H:%M:%S) ==="
  grep "FINAL" "./work_dirs/full/eval_sweep/${TAG}/eval_metrics_live.log" 2>/dev/null || true
done
echo "[sweep] ALL DONE"
