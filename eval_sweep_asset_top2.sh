#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# eval_total_abc.sh와 동일한 기본 대상. 필요하면 실행 시 환경변수로 override.
CONFIG="${CONFIG:-./projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new.py}"
CHECKPOINT="${CHECKPOINT:-./work_dirs/full_attn_cover_pyr_aabb_dice3d_new/epoch_14_lss_only.pth}"
GPU_ID="${GPU_ID:-0}"
GPUS="${GPUS:-1}"
BASE_PORT="${BASE_PORT:-50100}"
WAIT_FOR_EVALS="${WAIT_FOR_EVALS:-1}"
FULL_VIS="${FULL_VIS:-1}"

FG_THRESHOLDS=(0.75 0.85 0.9)
OCC_THRESHOLDS=(0.75 0.8 0.85 0.9)
SWEEP_SAMPLES=600

ENV_BIN="${ENV_BIN:-/home/cvlab/anaconda3/envs/eof/bin}"
export PATH="$ENV_BIN:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export EOCF_EVAL_GT_ROOT=/home/cvlab/Desktop/NIPS2026/datasets/efficientocf_gt_f3/GMO
export EOCF_EVAL_ASSET_GT_ROOT=/home/cvlab/Desktop/NIPS2026/datasets/nuscenes_gmo_full_hybrid_solid_data_v1/val
export EOCF_EVAL_MODE=1
export EOCF_EVAL_NMS_RADIUS=0
export EOCF_EVAL_ORACLE_MATCH=0
export EOCF_EVAL_METRIC_EVERY=200

test -x "$ENV_BIN/python" || { echo "eof Python not found: $ENV_BIN/python" >&2; exit 1; }
test -f "$CONFIG" || { echo "Config not found: $CONFIG" >&2; exit 1; }
test -f "$CHECKPOINT" || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
test -d "$EOCF_EVAL_GT_ROOT" || { echo "Evaluation GT not found: $EOCF_EVAL_GT_ROOT" >&2; exit 1; }
test -d "$EOCF_EVAL_ASSET_GT_ROOT/val/segmentation_instance3d" \
  || test -d "$EOCF_EVAL_ASSET_GT_ROOT/segmentation_instance3d" \
  || { echo "Asset GT not found: $EOCF_EVAL_ASSET_GT_ROOT" >&2; exit 1; }

if [[ "$WAIT_FOR_EVALS" == "1" ]]; then
  while pgrep -f '[t]ools/test.py' >/dev/null; do
    echo "[asset-sweep] waiting for running evals... ($(date '+%H:%M:%S'))"
    sleep 60
  done
fi

RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./work_dirs/full_attn_cover_pyr_aabb_dice3d_new/asset_sweep/$RUN_TAG}"
SWEEP_RESULTS="$OUTPUT_ROOT/sweep_results.tsv"
RANKED_RESULTS="$OUTPUT_ROOT/sweep_ranked.tsv"
FULL_RESULTS="$OUTPUT_ROOT/top2_full_results.tsv"
mkdir -p "$OUTPUT_ROOT"
printf 'fg\tocc\tiou3d_asset\tmetrics_log\n' > "$SWEEP_RESULTS"

run_eval() {
  local phase="$1" fg="$2" occ="$3" samples="$4" run_index="$5"
  local run_dir="$OUTPUT_ROOT/${phase}_fg${fg}_occ${occ}"
  local launcher_log="$run_dir/launcher.log"
  local metrics_log="$run_dir/eval_metrics_live.log"
  mkdir -p "$run_dir"

  export PORT=$((BASE_PORT + run_index))
  export EOCF_EVAL_FG_THR="$fg"
  export EOCF_EVAL_OCC_THR="$occ"
  export EOCF_EVAL_MAX_SAMPLES="$samples"
  export EOCF_EVAL_VIS_DIR="$run_dir"
  if [[ "$phase" == "sweep" ]]; then
    export EOCF_EVAL_VIS=0
  else
    export EOCF_EVAL_VIS="$FULL_VIS"
    export EOCF_EVAL_VIS_EVERY=48
  fi

  echo "[asset-sweep] $phase start: FG=$fg OCC=$occ samples=$samples port=$PORT"
  if ! USE_MPS=0 bash tools/dist_test.sh "$CONFIG" "$CHECKPOINT" "$GPUS" \
      --cfg-options data.workers_per_gpu=0 > "$launcher_log" 2>&1; then
    cat "$launcher_log" >&2
    echo "[asset-sweep] evaluation failed: $phase FG=$fg OCC=$occ" >&2
    return 1
  fi

  LAST_ASSET_IOU="$(
    grep -E 'IoU3d .*asset=' "$metrics_log" 2>/dev/null \
      | tail -n 1 \
      | sed -nE 's/.*asset=([-+0-9.eE]+).*/\1/p' \
      || true
  )"
  if [[ -z "$LAST_ASSET_IOU" ]]; then
    cat "$launcher_log" >&2
    echo "[asset-sweep] IoU3d asset parse failed: $metrics_log" >&2
    return 1
  fi
  LAST_METRICS_LOG="$metrics_log"
  echo "[asset-sweep] $phase done: FG=$fg OCC=$occ IoU3d(asset)=$LAST_ASSET_IOU"
}

run_index=0
for fg in "${FG_THRESHOLDS[@]}"; do
  for occ in "${OCC_THRESHOLDS[@]}"; do
    run_eval sweep "$fg" "$occ" "$SWEEP_SAMPLES" "$run_index"
    printf '%s\t%s\t%s\t%s\n' \
      "$fg" "$occ" "$LAST_ASSET_IOU" "$LAST_METRICS_LOG" >> "$SWEEP_RESULTS"
    run_index=$((run_index + 1))
  done
done

{
  head -n 1 "$SWEEP_RESULTS"
  tail -n +2 "$SWEEP_RESULTS" | sort -t $'\t' -k3,3gr
} > "$RANKED_RESULTS"

mapfile -t top2 < <(tail -n +2 "$RANKED_RESULTS" | head -n 2)
if [[ "${#top2[@]}" -ne 2 ]]; then
  echo "[asset-sweep] expected two ranked combinations, got ${#top2[@]}" >&2
  exit 1
fi

echo "[asset-sweep] top-2 by 600-sample IoU3d(asset):"
printf '%s\n' "${top2[@]}"
printf 'rank\tfg\tocc\tsweep_iou3d_asset\tfull_iou3d_asset\tmetrics_log\n' > "$FULL_RESULTS"

rank=0
for row in "${top2[@]}"; do
  rank=$((rank + 1))
  IFS=$'\t' read -r fg occ sweep_iou _ <<< "$row"
  run_eval "full_rank${rank}" "$fg" "$occ" 0 "$run_index"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rank" "$fg" "$occ" "$sweep_iou" "$LAST_ASSET_IOU" "$LAST_METRICS_LOG" \
    >> "$FULL_RESULTS"
  run_index=$((run_index + 1))
done

echo "[asset-sweep] all done"
echo "  sweep : $SWEEP_RESULTS"
echo "  ranked: $RANKED_RESULTS"
echo "  full   : $FULL_RESULTS"
