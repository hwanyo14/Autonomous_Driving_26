# 1) 소규모 sanity check (5개 window, 시각화 2개, 단일 프로세스)
# python preprocess_nuscenes_occ_instance3d.py \
#   --ann-files data/nuscenes/nuscenes_occ_infos_train.pkl data/nuscenes/nuscenes_occ_infos_val.pkl \
#   --bbox-inst-dir data/efficientocf/GMO/segmentation_instance3d \
#   --bbox-cls-dir data/efficientocf_bboxcls/GMO/segmentation \
#   --occ-dir data/nuScenes-Occupancy \
#   --out-root data/nuScenes-Occupancy_inst3d \
#   --time-receptive-field 3 \
#   --n-future-frames 4 \
#   --foreground-classes 2 3 4 5 6 7 9 10 \
#   --overwrite \
#   --limit 100 \
#   --save-vis \
#   --vis-limit 100 \
#   --num-workers 1


# 2) 전체 실행 (멀티프로세스)
python preprocess_nuscenes_occ_instance3d.py \
  --ann-files data/nuscenes/nuscenes_occ_infos_train.pkl data/nuscenes/nuscenes_occ_infos_val.pkl \
  --bbox-inst-dir data/efficientocf/GMO/segmentation_instance3d \
  --bbox-cls-dir data/efficientocf_bboxcls/GMO/segmentation \
  --occ-dir data/nuScenes-Occupancy \
  --out-root data/nuScenes-Occupancy_inst3d \
  --time-receptive-field 3 \
  --n-future-frames 4 \
  --foreground-classes 2 3 4 5 6 7 9 10 \
  --save-vis \
  --vis-limit 10 \
  --num-workers 8 \
  --log-every 10


# 3) 기존 결과 덮어쓰기 + fail-fast
# python preprocess_nuscenes_occ_instance3d.py \
#   --ann-files data/nuscenes/nuscenes_occ_infos_train.pkl data/nuscenes/nuscenes_occ_infos_val.pkl \
#   --bbox-inst-dir data/efficientocf/GMO/segmentation_instance3d \
#   --bbox-cls-dir data/efficientocf_bboxcls/GMO/segmentation \
#   --occ-dir data/nuScenes-Occupancy \
#   --out-root data/nuScenes-Occupancy_inst3d \
#   --overwrite \
#   --fail-fast \
#   --num-workers 8
