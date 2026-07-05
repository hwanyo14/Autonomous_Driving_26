# Changelog

## 2026-07-04 15:49 KST

### LoadInstanceWithFlow global_idx cat crash 수정

**핵심 내용**: instance cache 사용 경로에서 scalar `global_idx`를 `torch.cat`하려다 DataLoader worker가 실패하는 문제 수정.

**주요 변경사항**:
- `loading_instance.py`: cache-hit/regenerate 후처리 루프의 cat 제외 key에 `global_idx` 추가.
- 검증: `python -m py_compile projects/occ_plugin/datasets/pipelines/loading_instance.py`, `git diff --check` 통과.

## 2026-07-04 15:19 KST

### Learnable Query KV Downsampling

**핵심 내용**: query transformer의 multi-resolution KV 생성에서 avg pooling을 경량 learnable convolution downsampling으로 교체.

**주요 변경사항**:
- `transformer.py`: `LightweightKVDownsample` 추가. downsample scale은 depthwise stride conv + pointwise 1x1 conv로 구성하고, 초기값은 기존 avg pooling + identity channel mixing에 가깝게 설정.
- `transformer.py`: `kv_resolutions[-1]`을 full `context_seq` 해상도로 검증하고, 각 layer KV token을 learnable downsampler 출력으로 생성.
- 검증: `python -m py_compile projects/occ_plugin/occupancy/image2bev/transformer.py` 통과. torch smoke는 현재 shell에 `torch`가 없어 실행하지 못함.

## 2026-06-30 13:12 KST

### Remaining Data Loader Pruning

**핵심 내용**: train에서만 불필요한 `segmentation_bev` 로드/생성을 차단하고, 사용처가 없는 이미지 `canvas` 복사를 제거.

**주요 변경사항**:
- `base_config.py`: `LoadInstanceWithFlow(load_segmentation_bev=False)`를 train에 적용하고 test는 evaluation metric용으로 `True` 유지.
- `loading_instance.py`: `load_segmentation_bev` 옵션 추가. `False`면 `segmentation_bev` cache check/read/dense 변환/regen/save/results 저장을 건너뜀.
- `loading_bevdet.py`: downstream 사용처가 없는 `canvas_seq`, per-frame `canvas`, `results['canvas']` 생성을 제거.
- 검증: 대상 파일 `py_compile`, config run-path key 점검, `results['canvas']` 잔여 사용처 검색 통과.

## 2026-06-30 12:58 KST

### Data Loading Bottleneck Pruning

**핵심 내용**: `base_config.py` 기준 미사용 GT/캐시 로드를 줄여 train dataloader CPU/RAM 부담을 완화.

**주요 변경사항**:
- `base_config.py`: train pipeline에서 `LoadOccupancy` 제거. train `Collect3D`에서 `gt_occ`, `occ_dt`, `segmentation_bev`, `instance_bev`, `segmentation_cls_instance3d` 제거.
- `base_config.py`: test pipeline은 evaluation용 `gt_occ`/`segmentation_bev`는 유지하고, `occ_dt`, `instance_bev`, `segmentation_cls_instance3d` 제거. `LoadOccupancy(load_occ_dt=False, load_height=False)` 명시.
- `loading_occupancy.py`: `load_height` 옵션 추가. `False`면 height cache read/write/regen을 건너뜀.
- `loading_instance.py`: `use_flow=False`일 때 `instance_bev`/`flow_bev` cache read, cache 필수조건, 재생성, 저장을 건너뜀.
- `loading_instance.py`: 수집되지 않는 `attribute_label` dense tensor를 results에 넣지 않도록 정리.
- 검증: 대상 파일 `py_compile`, config run-path key 점검, `git diff --check` 통과.

## 2026-06-30 12:34 KST

### Unified Gaussian Head Quaternion Rotation

**핵심 내용**: Gaussian mixture rotation 경로를 yaw 기반에서 quaternion(wxyz) 기반으로 정리.

**주요 변경사항**:
- `query_head.py`: 별도 Gaussian head placeholder를 제거하고 단일 `GaussianHead(out_dim=query_num_gaussians * 11)` 출력의 `offset(3)+sigma(3)+quat(4)+weight(1)` 구조만 사용하도록 정리.
- `query_head.py`: BEV debug Gaussian footprint 시각화가 `mixture_quat_tqg4`와 3D sigma를 받아 2D yaw 회전 대신 quaternion 기반 3D covariance의 BEV projection을 사용하도록 변경.
- `query_head.py`: GT2P instance coverage loss의 mixture Mahalanobis 거리를 quaternion 회전 로컬 좌표계에서 계산하도록 변경.
- `efficientocf.py`, `utils_loss.py`: 내부 로컬 변수명을 `mixture_quat_*`로 정리해 `mixture_quat_tqg4` 경로와 일관화.
- 검증: 대상 파일 `py_compile` 통과.

## 2026-06-30 11:52 KST

### Evaluation / Inference Visualization 구현

**핵심 내용**: `/home/hwanhee/EOCF_ksh_0626`의 평가·추론 시각화 경로를 현재 repo 구조에 맞춰 이식.

**주요 변경사항**:
- `efficientocf.py`: 현재 `extract_feat_query` 반환 인덱스에 맞춘 `forward_test`/`simple_test` 추가. query selection 기반 2D IoU, bbox-AABB IoU, 3D IoU/Recall 계산 및 eval 시각화 저장 경로 연결.
- `voxelizer.py`, `utils_visualization.py`, `query_head.py`: eval full-scene Gaussian mixture voxelization, 3D mixture 시각화, eval all-rank query debug 저장, `EOCF_EVAL_*` threshold/vis env 지원 추가.
- `occupancy/apis/test.py`, dataset evaluate: single/multi GPU 평가 누적, live metric log, bbox/3D metric optional 집계 추가.
- `tools/test.py`, `tools/dist_test.sh`: eval vis 디렉터리/env 설정, `global_idx` meta key 보장, 평가 로그 파일 저장 및 NCCL timeout/MPS 설정 추가.
- `efficientocf_config.py`, `base_config.py`: eval occ threshold, mixture eval 사용, mixture3d vis 설정 기본값 추가.
- 검증: 대상 Python 파일 `py_compile`, `tools/dist_test.sh`/`run_eval.sh` `bash -n`, `git diff --check` 통과. config print/runtime smoke는 현재 shell에 `mmcv`/`torch`가 없어 실행하지 못함.

## 2026-06-29 11:31 KST

### Query Classification Gaussian Params Concat

**핵심 내용**: query classification head가 선택적으로 query feature와 예측 Gaussian mixture 파라미터를 concat해 class logits를 예측하도록 변경.

**주요 변경사항**:
- `query_head.py`: `query_cls_use_gaussian_params` 옵션 추가. 활성화 시 cls 입력에 per-Gaussian `offset xyz + sigma xyz + yaw(sin,cos) + weight`를 detach 후 concat.
- `query_head.py`: lifted center 기반 Gaussian 계산 이후 `query_cls_logits_qc`, `query_cls_scores_qc`, `query_cls_scores_tqc`를 재계산하도록 `apply_lifted_centers_to_outputs` 경로 갱신.
- `efficientocf.py`: detector가 `apply_lifted_centers_to_outputs` 이후 갱신된 cls 출력을 사용하도록 로컬 변수 추출 위치 조정.
- `efficientocf_config.py`, `base_config.py`: 기본값은 기존 동작 유지(`False`), 현재 baseline에서는 `query_cls_use_gaussian_params=True` 활성화.

## 2026-06-29 10:39 KST

### Query Gaussian/Class Head Layer Config

- `query_head.py`: `GaussianHead`, `ClassificationHead`의 MLP layer 수를 인자로 받도록 변경.
- `efficientocf_config.py`, `efficientocf.py`: `query_gaussian_head_num_layers`, `query_cls_head_num_layers` config 기본값 및 전달 경로 추가.
- `test_traj_large_adj_lr_crop_margin_nodepth.py`, `EfficientOCF_V1.1_1gpu.py`: 기존 동작 보존을 위해 두 설정값을 `2`로 명시.

## 2026-06-26 23:35 KST

### Local AABB GMO summary sigma None crash 수정

- `utils_loss.py`: mixture-only 경로에서 `sigmas_world_tq3=None`인데 local AABB GMO 호출부가 `sigmas_world_tq3[:T]`를 무조건 슬라이스하던 문제 수정.
- mixture가 유효하면 summary sigma 없이 `None`을 전달하고, helper 내부의 mixture branch가 그대로 처리하도록 변경.
- 검증: `utils_loss.py` `py_compile`, `git diff --check` 통과.

## 2026-06-26 23:31 KST

### Gaussian별 독립 alpha 예측 및 mixture-only 경로 전환

- `query_head.py`: Gaussian별 `gaussian_weight_head`를 추가하고 `sigmoid` alpha `[0,1]`를 예측하도록 변경. Gaussian 간 softmax/sum-to-1 정규화는 사용하지 않음.
- `query_head.py`, `efficientocf.py`: `query_sigma_world_tq3` summary Gaussian 계산/사용을 중단하고 기존 key는 호환용 `None`으로 유지.
- `efficientocf.py`: present occupancy voxelization을 summary Gaussian 대신 mixture voxelization 후 query 확률 union으로 생성하도록 변경.
- `utils_loss.py`: matched GMO/sequence IoU/local AABB/DT/GT2P 보조 경로가 mixture tensors를 우선 사용하고 summary sigma가 없어도 동작하도록 정리.
- `voxelizer.py`: mixture weight 주석을 독립 alpha 설계에 맞게 갱신.
- 검증: 대상 파일 `py_compile`, `git diff --check` 통과. `QueryHead` dummy forward smoke는 현재 shell Python에 `torch`가 없어 실행하지 못함.

## 2026-06-26 23:03 KST

### 메인 config에 Local AABB GMO pair 시각화 설정 반영

- `test_traj_large_adj_lr_crop_margin.py`: `debug_gmo_local_aabb_pair_vis_every=48`, 전용 저장 경로, 최대 저장 pair 수 설정 추가.
- `NOTES.md`: 현재 메인 학습 config가 `projects/configs/baselines/test_traj_large_adj_lr_crop_margin.py`임을 기록.

## 2026-06-26 22:58 KST

### Local AABB GMO supervision pair 시각화 추가

- `utils_loss.py`: local AABB matched GMO loss에서 실제 학습에 쓰는 `pred_pairs`, `gt_pairs`, `valid_pairs`를 PNG/`.pt`로 저장하는 디버그 훅 추가.
- 저장 PNG는 pair별 BEV max-Z projection 기준으로 GT(green), pred(red), overlay를 함께 보여주며, `.pt`에는 crop tensor와 `query_idx`, `gt_instance_id`, valid frame 메타를 저장.
- `efficientocf_config.py`, `EfficientOCF_V1.1_1gpu.py`: `debug_gmo_local_aabb_pair_vis_every`, `debug_gmo_local_aabb_pair_vis_dir`, `debug_gmo_local_aabb_pair_vis_max_pairs` 설정 추가.

## 2026-06-25 11:43 KST

### Matched GMO AABB Local Shape Loss 추가

- `utils_loss.py`: matched pair별 GT AABB 중심/크기 기반 local crop을 만들고, pred Gaussian occupancy는 predicted query center 기준 local grid에서 직접 계산하는 `local_aabb` GMO shape loss 경로 추가.
- `efficientocf_config.py`, `efficientocf.py`, `EfficientOCF_V1.1_1gpu.py`: `query_gmo_shape_loss_mode`, local crop size/scale/margin, `query_gmo_loss_weight` config 추가 및 loss 호출에 연결.
- `test_traj_large_adj_lr.py`: 메인 실험 config에서 `query_gmo_shape_loss_mode='local_aabb'` 활성화.
- 기본값은 기존 global GMO loss 유지(`query_gmo_shape_loss_mode='global'`).
- 정정: `query_gmo_local_crop_size=None`일 때 `data/efficientocf_bboxcls`에서 로드된 bbox occupancy grid의 AABB voxel 크기를 그대로 사용하도록 변경. margin config는 grid voxel 단위인 `query_gmo_local_crop_margin_vox`로 변경.
- 검증: `py_compile`, `git diff --check` 통과.

## 2026-06-25 11:13 KST

### Query Gaussian weight 고정

- `query_head.py`: learnable Gaussian weight head를 제거하고 모든 mixture component weight를 고정 상수 `1.0`으로 생성하도록 변경.
- `query_head.py`: Gaussian weight regularization loss 및 관련 debug reg 항목 제거. `dbg_query_weight_sum_mean`은 고정 weight 총합 확인용으로 유지.
- `efficientocf.py`, `efficientocf_config.py`, baseline configs: 더 이상 사용하지 않는 `query_multi_gaussian_weight_*` 설정 전달과 config 항목 제거.
- 검증: 대상 파일 `py_compile` 통과.

## 2026-06-23 17:17 KST

### GMO quality bbox-vs-fine voxel alignment 시각화 추가

- `utils_visualization.py`: bbox 기반 dense instance GT와 fine instance voxel GT의 BEV 정합을 frame별 PNG로 저장하는 `maybe_save_gmo_quality_alignment_vis` 추가. 각 PNG는 bbox GT, fine voxel GT, overlap/diff 3개 panel을 포함함.
- `efficientocf.py`: matched GMO filtering에 쓰이는 동일 temporal slice를 시각화 함수에 전달. 모든 frame을 `gmo_quality_alignment_vis/iter_xxx_scene_lidar/frame_txx.png`로 저장함.
- `efficientocf_config.py`, `test_traj_large_adj_lr.py`, `mmdet_train.py`: `debug_gmo_quality_alignment_vis_every` 및 전용 저장 dir 설정 추가. `train_total.sh` 기준 main config에서는 매 iteration 저장되도록 `1`로 활성화.
- 검증: 대상 파일 `py_compile`, `git diff --check`, synthetic PNG 저장 테스트 통과.

## 2026-06-23 17:08 KST

### GMO quality TensorBoard 로그 누락 수정

- `utils_loss.py`: `_aggregate_training_losses`의 debug key 삭제 필터에서 `dbg_gmo_quality_*` prefix를 허용하도록 수정. `train_total.sh`의 `TensorboardLoggerHookSplitTabs` 경로에서 `dbg/gmo_quality_scene_pass_object_count`가 기록되도록 함.
- 원인: matched GMO quality debug 값은 생성됐지만 aggregation 마지막 단계에서 whitelist되지 않아 TensorBoard hook까지 전달되지 않았음.
- 검증: `utils_loss.py`, `test_traj_large_adj_lr.py`, `tensorboard_hooks.py` `py_compile` 및 `git diff --check` 통과.

## 2026-06-23 17:01 KST

### Matched GMO quality filter main config 반영 및 scene count 로그 추가

- `test_traj_large_adj_lr.py`: 실제 사용 config에 `use_query_gmo_quality_filter=True`, ROI 30m, occupancy ratio 0.4 설정 적용.
- `EfficientOCF_V1.1_1gpu.py`: 이전에 추가했던 quality filter 활성화 설정 제거.
- `utils_loss.py`: 필터를 모두 통과한 현재 scene/sample의 object 수를 `dbg_gmo_quality_scene_pass_object_count`로 추가. 기존 TensorBoard hook에서 `dbg/gmo_quality_scene_pass_object_count` scalar로 기록됨.
- 검증: 대상 파일 `py_compile`, quality-mask synthetic test, zero-pass scene count test, `git diff --check` 통과.

## 2026-06-23 16:55 KST

### Matched GMO quality filter 추가

- `utils_loss.py`: Hungarian matched GMO focal/BCE/dice loss 전에 pair-level quality filter를 추가. 현재 frame 기준 ROI 30m 이내이고 fine GT instance voxel / bbox instance voxel 비율이 설정 threshold 이상인 pair만 loss에 사용함.
- `efficientocf.py`: matched GMO loss 호출부에 bbox 기반 dense instance GT와 현재-frame index를 전달.
- `efficientocf_config.py`, `EfficientOCF_V1.1_1gpu.py`: `use_query_gmo_quality_filter`, `query_gmo_quality_roi_radius_m`, `query_gmo_quality_min_occupancy_ratio` 설정 추가. baseline은 30m / 0.4 기준으로 활성화.
- 검증: 대상 파일 `py_compile`, `git diff --check`, `eo` 환경 synthetic quality-mask 테스트 통과.

## 2026-06-23 13:51 KST

### QueryDepthHead depth supervision soft label CE 전환

- `utils_query_projection.py`: QueryDepthHead depth loss를 hard depth bin `F.cross_entropy`에서 Gaussian soft label CE로 변경. GT depth value 기준으로 bin-center 좌표 target을 만들고 `sigma=1.0 bin` 기본값을 사용함.
- `efficientocf_config.py`, `EfficientOCF_V1.1_1gpu.py`: `query_depth_soft_label_sigma_bins=1.0` 설정 추가.
- `test_traj_large_adj_lr.py`: 실제 사용 config에도 `query_depth_soft_label_sigma_bins=1.0` 추가.

## 2026-06-21 21:33 KST

### Binary query foreground 시각화 색상 변경 (녹색 → 노란색)

- `query_head.py` `_get_query_pred_palette`: binary foreground 색을 `[80,255,120]`(녹색)에서 `[255,255,0]`(노란색)으로 변경. GT semantic의 trailer 녹색(`[140,255,100]`)과 혼동되던 문제 해결. marker와 legend가 동일 색을 공유함.

## 2026-06-21 21:17 KST

### Query 분류를 multi-class → background/foreground 이진 분류로 전환

- 기존 `use_separate_classes` switch를 query branch에도 연결: `False`면 query가 binary(bg/fg) 분류, `True`면 종전 8-class 경로 유지. matcher/loss/scoring은 logit shape에서 class 수를 읽어 자동 적응함.
- `efficientocf_config.py`: `MODEL_CFG_DEFAULTS`에 `use_separate_classes`(기본 True) 추가, `apply_model_cfg`에서 `self.query_binary_cls = not use_separate_classes` 파생.
- `efficientocf.py`: binary일 때 `query_raw_to_compact_class_map`을 enumerate가 아니라 collapse로 구성(background raw id→0, 모든 foreground raw id→1). `query_binary_cls`를 `QueryHead`로 전달.
- `query_head.py`: `query_binary_cls` 인자 추가. binary에서는 `query_class_ids`가 라벨 공간이 아닌 foreground-id registry이므로 `len(query_class_ids)==num_query_classes` 검증을 건너뜀. 예측 marker/legend 색을 `_get_query_pred_palette()`(binary: [background, foreground])로 통일. GT semantic overlay는 per-class 색 유지.
- `test_traj_large_tf_simple.py`: `use_separate_classes` 분기에 `query_num_classes`/`query_cls_names`/`query_cls_loss_class_weights`(binary: `[0.1, 1.0]`, reference 구현과 동일) 추가, model_cfg가 분기 값과 `use_separate_classes`를 사용하도록 정리. `query_class_ids`는 foreground registry로 그대로 유지.
- 검증: 4개 파일 `py_compile` 통과, config exec 시 `query_num_classes=2`/`query_class_names=['background','foreground']`/weights 길이 2 확인, collapse map이 foreground ids→1·background→0·미등록 id→-1(필터) 매핑 확인. `efficientocf_bboxcls/GMO/segmentation` 캐시의 class 채널이 raw nuScenes id(`{2,3,4,7,9,10}`)를 보존함을 실데이터로 확인 → `use_separate_classes` 재사용이 query foreground GT를 손상시키지 않음(reference의 별도 `query_cls_mode` 키와 현재 config에서 기능적 동등).

## 2026-06-21 19:31 KST

### Trajectory debug DDP log key 불일치 수정

- `utils_loss.py`: trajectory loss 유효 여부와 무관하게 `loss_query_traj`와 6개 `dbg_query_traj_*` key를 0으로 선생성하고, 유효 loss가 있으면 실제 값으로 덮어쓰도록 수정.
- rank별 matched future pair 유무에 따라 `log_vars` 길이가 달라져 발생하던 `loss log variables are different across GPUs!` assert 방지.
- 검증: trajectory loss가 `None`인 경우와 유효 dict인 경우 aggregate 결과 key 집합이 동일함을 확인하고 config model build, `py_compile`, `git diff --check` 통과.

## 2026-06-21 19:28 KST

### Teacher forcing epoch 기준 전환

- `test_traj_large_cost_iou_resnet_lr_depthbin.py`: iter cutoff을 비활성화하고 `TrajectoryWarmupHook`으로 epoch 1~8 teacher forcing 활성, epoch 9부터 비활성화.
- `efficientocf.py`: `query_traj_teacher_forcing_until_iter=0`이면 iter 제한 없이 hook의 boolean 상태만 따르도록 변경. 양수인 기존 config는 종전 iter cutoff 유지.
- `efficiency_hooks.py`: 현재 단일 trajectory 경로의 `query_traj_teacher_forcing` 상태를 schedule debug 값에 반영하고 구형 key fallback 유지.
- post-match GMO loss 및 학습용 geometry 경로는 이미 `experiment/tf-traj-cost`와 동일하여 추가 변경하지 않음.
- 검증: 지정 config model/hook build 및 epoch 1·8 활성, epoch 9 비활성 전환 확인.

## 2026-06-21 19:23 KST

### Main trajectory config 단일 head 경로 정합

- `test_traj_large_cost_iou_resnet_lr_depthbin.py`: 제거된 multi-mode/routing/xy-refine/ratio teacher-forcing 인자와 `TrajectoryWarmupHook` 제거.
- matched query에 대해서만 trajectory head를 실행하도록 `query_traj_matched_only=True`를 명시.
- `experiment/tf-traj-cost`의 hard teacher forcing 설정인 `query_traj_teacher_forcing=True`, `query_traj_teacher_forcing_until_iter=30000` 적용.
- 검증: 지정 config model build, deferred 단계 head 미호출, GT prior 치환 후 matched-query head 1회 호출, center/GMM delta 누적 이동 smoke test 통과.

## 2026-06-21 19:15 KST

### Trajectory 경로를 experiment/tf-traj-cost 방식으로 복원

- `query_head.py`: multi-mode/Bernstein/refinement trajectory head를 제거하고, 과거 center delta와 미래 zero prior를 입력으로 받는 단일 MLP trajectory head 및 `[F,Q,2]` 출력 계약으로 복원.
- `efficientocf.py`: matching 이후 matched query만 trajectory를 예측하고, 유효한 과거 GT delta를 사용하는 hard teacher forcing 및 예측 delta 누적 기반 미래 center/Gaussian mixture 이동 경로로 복원.
- `utils_loss.py`: matched query의 미래 step delta에 대한 L1/L2 trajectory loss와 moving/static reweight만 유지.
- `efficientocf_config.py`: 단순 trajectory 경로에서 사용하는 설정만 유지하고 multi-mode/refinement/endpoint 관련 설정 제거.
- Pyramid query transformer CA, class/depth head, attention matching, Gaussian mixture head, voxelizer, dataset pipeline은 변경하지 않음.
- 검증: `py_compile`, `git diff --check`, 핵심 함수 AST 기준 브랜치 대조, `eo` 환경 trajectory head shape smoke test 및 baseline model build 통과.

## 2026-06-18 10:38 KST

### Hungarian Matching Feature Cost 제거 및 BEV IoU Cost 도입

- `utils_matcher.py`: query/GT feature matching 입력, cosine similarity cost, soft-assign diagnostic cost 제거. 매칭은 center/class/temporal/BEV IoU/attention cost만 사용.
- `utils_matcher.py`: BEV Dice cost 경로를 low-res BEV IoU cost로 교체. `matched_gmo_voxelizer`의 low-res 3D grid를 BEV max projection한 뒤 `1 - IoU` 계산.
- `efficientocf.py`: 매칭 전 GT instance context feature pooling 및 feature frame selection 제거.
- `efficientocf_config.py`: `query_bev_iou_match_cost_weight` 추가, feature/soft-assign match cost config 제거.
- `test_traj.py`, `test_traj_large_cost_iou.py`: `query_bev_iou_match_cost_weight=1.0`으로 IoU cost 활성화.
- `EfficientOCF_V1.1_1gpu.py`: 신규 key를 보수적으로 `0.0`으로 명시.
- `utils_loss.py`: `dbg_query_match_cost_bev_iou_*`, `dbg_query_bev_iou_match_cost_weight` 로깅으로 변경하고 feature/soft debug 항목 제거.
- `PROJECT_STRUCTURE.md`: trajectory/IoU 실험 config 항목 추가.

### BEV IoU Matching Cost 2D Rasterize 전환

- `utils_matcher.py`: matching cost용 BEV IoU pred map을 `64x64x20` 3D voxelize 후 Z축 max projection하던 방식에서, XY `64x64` 2D Gaussian mixture rasterize로 전환.
- GT instance map도 3D adaptive pooling 대신 Z축 occupancy union 후 `adaptive_max_pool2d`로 IoU grid에 맞추도록 변경.
- GMO focal/BCE/dice loss 경로는 기존 3D `query_matched_gmo_bce_occ_size=(64,64,20)` voxelize 유지.
- `conda run -n eo` one-off로 단일 Gaussian fallback 및 mixture 입력 모두 `matcher_bev_iou_2d` 검증 통과.

## 2026-06-16 13:05 KST

### test_traj_large ResNet-50 전환

- `projects/configs/baselines/test_traj_large.py`의 image backbone을 ResNet-18에서 ResNet-50으로 변경.
- ResNet-50 stage 출력에 맞춰 `img_neck.in_channels`를 `[256, 512, 1024, 2048]`로 변경.

## 2026-06-15 15:15 KST

### use_query_dt_loss=False 시 DT 데이터 로드 비활성화

- `tools/config_pipeline_utils.py` 추가: `model.model_cfg.use_query_dt_loss=False`이면 data pipeline의 `LoadOccupancy.load_occ_dt`를 false로 바꾸고 `Collect3D.keys`에서 `occ_dt` 제거.
- `tools/train.py`, `tools/test.py`에서 config 로드 직후 해당 동기화를 적용해 학습/평가 모두 불필요한 DT 파일 I/O를 생략.
- 신규 tools 유틸 반영을 위해 `PROJECT_STRUCTURE.md` 업데이트.

## 2026-06-15 15:08 KST

### MPS 멀티잡 실행 자동화

- `tools/dist_train.sh` 시작 시 CUDA MPS pipe/log 디렉터리를 준비하고 MPS 데몬을 자동 실행하도록 변경.
- `USE_MPS=0`이면 기존 non-MPS 실행 경로를 유지하도록 opt-out 추가.
- `stop_mps.sh` 추가: MPS 데몬 종료 후 pipe/log 디렉터리 정리.
- 신규 root 스크립트 반영을 위해 `PROJECT_STRUCTURE.md` 업데이트.

## 2026-06-15 15:01 KST

### Query Transformer Pyramid Cross-Attention

- `query_transformer_num_layers` 기본값을 3으로 변경하고 layer별 KV 해상도 기본값 `((14, 25), (28, 50), (56, 100))` 추가.
- `TransformerModule`에서 cross-attention key/value만 layer별 평균 풀링 토큰을 사용하도록 변경.
- 최종 layer는 full-res KV를 유지해 반환 attention map 및 downstream bbox/debug 경로의 `[T,Q,Ncam,H,W]` shape 호환성 유지.

## 2026-06-11 KST (9차)

### suppress weight 변경 (test_traj.py)

- `query_attn_bbox_other_weight` 0.1 → **0.3** (사용자 결정; cost 레포 검증 baseline은 0.1이었음).
- 시각화 score 가중치를 cls 단독으로 변경: `debug_query_score_cls_weight` 0.3→**1.0**, `cam_attn_weight` 0.7→**0.0** (iou는 0.0 유지). 이제 vis score = fg cls_prob 그대로.

## 2026-06-11 KST (8차)

### traj warmup/TF 스케줄 사용자 재설계 (test_traj.py)

- `mode_cls_loss_weight`를 warmup ramp 없이 **epoch 1부터 0.1 고정** (기존 0.02→0.05→0.10 계단 제거).
- TF gt_ratio를 **epoch 1~4 풀 유지(1.0)** → epoch 5~7 ramp(0.9/0.7/0.5) → epoch 8부터 0.0으로 변경. refine 활성(epoch 5)과 TF 전환 시작이 동시에 일어나는 구조.
- traj_loss_w / refine_w 계단은 기존 유지 (0.05/0.15/0.25/0.5, 0/0.02/0.05/0.10).
- 검증: 15 epoch 전체 stage 해석 표 확인.

## 2026-06-11 KST (7차)

### Teacher forcing을 iter 기준 → epoch 기준으로 전환 (test_traj.py)

- iter 스케줄(`schedule_iters/gt_ratios`)을 빈 튜플로 비활성 → 코드가 `query_traj_teacher_forcing_gt_ratio` 속성값을 직접 사용하는 경로로 전환 (efficientocf.py L323-324, 코드 수정 없음).
- `traj_warmup_schedule`에 epoch 2, 3 stage를 추가하고 모든 stage에 `query_traj_teacher_forcing_gt_ratio` 포함: epoch 1=1.0 → 2=0.5 → 3+=0.0. 원본(1-GPU 4000 iter/epoch 기준 epoch 0.75~2.0 전환)을 epoch 계단으로 근사.
- 이제 GPU 수가 바뀌어도 TF 스케줄 재스케일 불필요.
- 주의: TrajectoryWarmupHook은 마지막 매칭 stage "하나만" 적용하므로 각 stage가 전체 키를 보유해야 함 (config 주석으로 명시).
- 검증: epoch별 stage 해석 (1→1.0/2→0.5/3→0.0, weight 계단 유지) 확인. 기존 8-GPU run은 재시작해야 반영됨.

## 2026-06-11 KST (6차)

### 디버그 시각화 클래스 색상 얼라인

- `query_head.py::_get_query_vis_palette`: 위치 기반 palette → **raw id 고정 매핑**으로 변경. 8-class 전환(pedestrian 제외)으로 trailer/truck 색이 한 칸씩 밀리던 문제 해결, 클래스 목록이 바뀌어도 색 고정. 목록에 없는 raw id는 extra 색 순환.
- `utils_instance_img_debug.py::_instance_color`: 별도 팔레트를 query vis와 동일한 raw-id 매핑으로 통일 → 모든 디버그 PNG에서 같은 클래스 = 같은 색.
- 검증: trailer=(140,255,100), truck=(255,120,220) 등 원래 의도 색 복원 확인. 시각화 전용 변경이라 학습 무영향.

## 2026-06-11 KST (5차)

### Trajectory 2-mode 모듈 이식 (TRAJECTORY_CONFIG.md / cen10_6, traj 레포에서)

**핵심 내용**: `/home/hwanhee/Autonomous_Driving_26_traj`(같은 커밋 dd6c98d 기반)의 2-mode trajectory 브랜치를 3-way merge로 이식. 적용 config는 `test_traj.py` (test.py는 traj 기능 전부 기본값 off라 기존 동작 유지).

**3-way merge (base=dd6c98d, ours=cost 이식본, theirs=traj)**:
- `query_head.py`, `efficientocf.py`, `efficientocf_config.py`, `utils_visualization.py`: 클린 머지
- `utils_loss.py`: 충돌 2곳 수동 해결 (dbg prefill 중복 → total 버전 유지, dbg 유지목록 → cls_/traj_ 합집합)

**부수 반영**:
- `efficiency_hooks.py`: `TrajectoryWarmupHook` 추가 (traj 버전 복사)
- `loading_instance.py`/`loading_occupancy.py`: `exclude_occ_class_ids` 로드 단계 클래스 제거 지원 (traj 버전 복사)
- `mmdet_train.py`: `cfg.custom_hooks` 등록 루프 추가 (없으면 TrajectoryWarmupHook 미등록)
- `__init__.py` 2곳: TrajectoryWarmupHook export
- `tensorboard_hooks.py`: traj 버전 무시, 기존(cost) `TextLoggerHookNoDbg` 유지
- traj의 `tools/train.py` vis remap은 미이식 (cost의 mmdet_train 방식 유지)

**test_traj.py** (test.py 복사본에 추가):
- §3 전체: `query_traj_num_modes=2`, `semantic_routing_enabled=True`, `static_threshold_m=0.8`, `mode_infer_policy='argmax'`, xy refine(2층/64dim/weight 0.1), mode_cls_w=0.1, moving_w=5.0
- `traj_warmup_schedule` 4단계 (epoch 1/5/8/11) + `TrajectoryWarmupHook` 등록
- teacher forcing mix 스케줄: **1000 iter/epoch(4GPU) 기준 (750,1000,1250,1625,2000)으로 재스케일** (원본 4000 iter/epoch에 3000~8000)
- `exclude_occ_class_ids=(7,)`: pedestrian을 로드 단계에서 제거 (train instance/occupancy + test instance 로더, cen10_3과 동일하게 test LoadOccupancy는 제외)

**검증**: test_traj.py build_model 통과 (2-mode, refine 서브모듈 3개 생성, TF 스케줄 확인), test.py 호환성 통과 (num_modes=1, refine off), TrajectoryWarmupHook cfg 빌드 통과.

## 2026-06-11 KST (4차)

### cls match cost 활성화 (test.py)

- `query_cls_match_cost_weight=0.075` 추가 — Hungarian cost matrix에 `-log P_q(GT class)` NLL 항(clamp 30, 기여 상한 0.075×30=2.25) 활성화.
- 코드는 2차 이식 때 들어온 `utils_matcher.py` L586-605 그대로 사용, config 값만 켬. cost 레포 `*_clscost.py` run과 동일 값.
- 주의: from-scratch 학습 전제 (학습 도중 켜면 악화 관측, QUERY_ATTN_CHANGES.md 3번).

## 2026-06-11 KST (3차)

### ⑤ 하이퍼파라미터 일괄 반영 + pedestrian 제외 (test.py)

- `query_num_queries=200`, `query_center_match_cost_weight` 4.0→10.0, `query_temporal_offset_match_cost_weight` 2.0→0.5
- 클래스 구성을 문서/cls_5 기준 8-class로 변경: `class_names`에서 'pedestrian' 제거, `query_class_ids=[0,2,3,4,5,6,9,10]` (7 제외), `query_cls_loss_class_weights=[0.02, 1.4, 1.3, 0.3, 1.4, 1.4, 1.2, 0.9]`
- pedestrian은 사용자 결정으로 학습/추론/시각화 전체에서 제외 — gt_prep의 allowed_raw_ids 필터가 로드 직후 background로 처리
- cost 레포의 `projects/occ_plugin/utils/formating.py` 이식 (8-class 평가 클래스명 매핑)
- 검증: test.py `build_model` 통과, num_queries/class_ids/cls_weights/cost weight 모두 확인

## 2026-06-11 KST (2차)

### QUERY_ATTN_CHANGES.md 항목 이식 (①②③④⑥⑦⑧, cost 레포에서)

**핵심 내용**: `/home/hwanhee/Autonomous_Driving_26_cost`(같은 커밋 dd6c98d 기반)의 uncommitted 변경에서 query-attention/matching 개선을 이식. cost 버전 파일을 통째로 복사하는 방식 사용.

**복사한 파일** (cost → total):
- `detectors/utils_loss.py`: ① attn loss를 KL → inside-mass NLL(`-log(inside_mass)`)로 교체, ② suppress(타 객체 영역 질량 페널티) 항 추가, ⑩ 기존 수동 dbg prefill을 cost 버전으로 통일
- `detectors/utils_matcher.py`: ③ Hungarian cost에 `inside_log` metric 추가
- `detectors/utils_gt_prep.py` + `detectors/efficientocf.py`: ④ history-all-valid 인스턴스 필터 (`_build_history_all_valid_instance_ids`, `_filter_dense_instance_ids`)
- `detectors/efficientocf_config.py`: ⑥ 신규 키 등록 (`query_attn_bbox_other_*`, `query_require_history_all_valid`, `query_attn_overlap_loss_weight` 등)
- `apis/mmdet_train.py`: ⑦ `_configure_visualization_dirs` — vis 경로를 `work_dirs/<config>/vis/<timestamp>/` 아래로 자동 재매핑
- `dense_heads/query_head.py` + `detectors/utils_visualization.py`: ⑧ 디버그 시각화 개선 (selected query distance-NMS, 클래스 팔레트 수정)
- `image2bev/transformer.py`: ⑨ `query_attn_overlap_loss` — **계획에 없었으나 efficientocf.py가 constructor kwarg로 하드 의존하여 같이 이식. 기본 weight 0.0이라 비활성.**

**test.py 변경**:
- 추가: `query_attn_match_metric='inside_log'`, `query_attn_bbox_other_weight=0.1`, `query_attn_bbox_other_mode='union'`, `query_attn_bbox_unmatched_weight=0.0`, `query_require_history_all_valid=True`
- 변경: `query_attn_match_cost_weight` 0.5 → 0.3 (문서 권장값)
- 유지: `query_center_match_cost_weight=4.0`, `query_temporal_offset_match_cost_weight=2.0`, query 수, class weights (⑤ 미적용, 사용자 직접 조정 예정)

**검증**: test.py로 `build_model`까지 정상, 신규 attr 값 모두 확인.

## 2026-06-11 KST

### 1~8 GPU DDP 호환성 수정 (4_CHANGELOG.md의 2026-05-15 14:20 수정 이식)

**핵심 내용**: 이전 레포(`4_CHANGELOG.md`, `4_NOTES.md`)에서 검증된 multi-GPU DDP 안정화 수정을 현재 레포에 적용.

**주요 변경사항**:
- `tools/train.py`: distributed 초기화 전에 `torch.cuda.set_device(int(os.environ['LOCAL_RANK']))` 추가
- `projects/occ_plugin/occupancy/apis/mmdet_train.py`: `MMDistributedDataParallel`에 `init_sync=False`, `static_graph=True` 추가 (torch 2.7 지원 확인, `find_unused_parameters`는 유지)
- `projects/occ_plugin/occupancy/detectors/utils_loss.py`: `_aggregate_training_losses()`에서 조건부로 생성되던 `dbg_query_match_cost_*` 키 전체를 모든 rank에서 항상 0으로 선생성하도록 수정 → rank별 `log_vars` 키 불일치로 인한 `loss log variables are different across GPUs!` assert 방지

**실행기**: 별도 `train_total.sh`는 만들지 않음. 기존 `run.sh` → `tools/dist_train.sh`가 이미 config/GPU 수/PORT를 인자·환경변수로 받는 구조라 그대로 사용 (`PORT=24560 bash run.sh <config> 8`).

## 2026-06-30 12:13 KST

### Unified Gaussian Head Quaternion Rotation

- `query_head.py`: offset/sigma/rotation/weight Gaussian head 4개를 `GaussianHead(out_dim=query_num_gaussians * 11)` 단일 head로 통합.
- Gaussian component 출력을 `offset(3) + sigma(3) + quat(4, wxyz) + weight(1)`로 분해하고, downstream key를 `mixture_quat_tqg4`로 변경.
- `voxelizer.py`: Gaussian mixture grouped/scene voxelization을 yaw 기반 XY 회전에서 quaternion 기반 3D 회전으로 변경.
- matcher/loss/geometry/visualization 경로의 `mixture_yaw_*` key를 `mixture_quat_*` key로 교체하고, BEV matching은 quaternion으로부터 XY covariance를 투영해 rasterize하도록 변경.
- 검증: 대상 Python 파일 `py_compile` 통과. 현재 shell Python에 `torch`가 없어 dummy tensor smoke test는 실행하지 못함.

## 2026-06-11 10:20 KST

### 터미널 로그에서 dbg metric 숨김

- `projects/occ_plugin/core/evaluation/tensorboard_hooks.py`에 `TextLoggerHookNoDbg` 추가: `dbg_`/`dbg/` prefix 키를 터미널·JSON 로그에서만 필터링 (TensorBoard `dbg/*` 기록은 `TensorboardLoggerHookSplitTabs`로 유지).
- `test.py`, `EfficientOCF_V1.1_1gpu.py`, `EfficientOCF_V1.1_1gpu_traj_tf.py`의 text logger를 `TextLoggerHookNoDbg`로 교체.

## 2026-06-11 10:05 KST

### occ_pool_ext CUDA extension 재빌드

- `train_total.sh` 실행 시 `ImportError: cannot import name 'occ_pool_ext'` 발생 (`.so` 부재가 원인, circular import 아님).
- `projects/occ_plugin/ops/occ_pooling`에서 `setup.py build_ext --inplace`로 재빌드 후 import 정상 확인.

## 2026-05-26 11:01 KST

### BBox-based GT Instance Centers

**핵심 내용**: center supervision 계열 GT instance center를 fine occupancy voxel 평균 대신 bbox-volume instance의 3D bbox midpoint로 계산하도록 변경.

**주요 변경사항**:
- `loading_instance.py`: `build_instance_center_world_targets()`의 instance center 계산을 voxel 평균에서 `(min_xyz + max_xyz) / 2` bbox midpoint로 변경
- `efficientocf.py`: trajectory matching용 `gt_inst_center_world_tn3`가 `gt_occ_inst_bundle`보다 dataloader의 `gt_instance_centers_world`/`gt_instance_centers_valid`를 우선 사용하도록 변경
- `efficientocf.py`: center matching loss, Hungarian center cost, trajectory loss, trajectory teacher forcing이 bbox center source를 타도록 GT center routing 정렬

**유지된 동작**:
- `gt_occ_inst` 기반 dense occupancy / class / instance tensor는 그대로 유지
- `gmo_bce`/`focal`/`dice` 및 voxel mask 기반 GT는 기존 fine occupancy source 유지

## 2026-05-22

### Hungarian Matching Temporal Offset Cost

**핵심 내용**: Hungarian matching cost에 과거 프레임 간 query/GT center delta offset 차이를 추가해, birth/disappearance 주변의 jittering query가 덜 선택되도록 보강.

**주요 변경사항**:
- `utils_matcher.py`: `query_temporal_cost_frame_indices`로 선택된 과거 프레임들에 대해 `delta = center[t+1] - center[t]` 기반 temporal offset cost 추가
- `utils_matcher.py`: `valid_prev & valid_next`인 GT delta step만 평균에 반영하고, invalid step은 matching cost에서 제외
- `utils_loss.py`: `dbg_query_temporal_offset_match_cost_weight` 및 `dbg_query_match_cost_temporal_offset_*` tensorboard 디버그 로깅 추가
- `efficientocf_config.py`: `query_temporal_offset_match_cost_weight` 기본값/바인딩 추가
- `EfficientOCF_V1.1_1gpu.py`, `EfficientOCF_V1.1_1gpu_traj_tf.py`: `query_temporal_offset_match_cost_weight=2.0` 설정 추가

## 2026-05-22

### Trajectory Teacher Forcing Invalid Past Delta Zeroing

**핵심 내용**: teacher forcing 적용 시 matched query의 past offset prior를 먼저 0으로 초기화하고, 연속 두 프레임 모두 GT가 valid인 delta만 GT delta로 덮어쓰도록 변경.

**주요 변경사항**:
- `efficientocf.py`: birth/disappearance 등으로 `valid_prev & valid_next`가 false인 past step에 query-predicted delta가 남지 않고 0 prior가 들어가도록 수정

## 2026-05-16

### Trajectory Head Teacher Forcing

**핵심 내용**: trajectory head의 과거 delta offset prior를 학습 초반에 GT centers 기반으로 계산하도록 teacher forcing 적용. 특정 iteration 이후 hard switch로 prediction 기반으로 전환.

**주요 변경사항**:
- `efficientocf_config.py`: `MODEL_CFG_DEFAULTS`에 `query_traj_teacher_forcing` (bool), `query_traj_teacher_forcing_until_iter` (int) 추가 및 apply 바인딩
- `efficientocf.py`: `query_traj_matched_only` 분기 내 `_motion_skd` 생성 직후, `_predict_trajectory_from_inputs` 호출 이전에 GT override 삽입. `inst_match_result["gt_centers_tn3"]`와 `matched_inst_idx`를 활용해 매칭된 쿼리의 past prior `[:past_steps, :, -2:]`를 GT delta로 교체
- 새 config `EfficientOCF_V1.1_1gpu_traj_tf.py` 생성: `query_traj_teacher_forcing=True`, `query_traj_teacher_forcing_until_iter=4000`

**전환 조건**: `self.training and query_traj_teacher_forcing and _train_iter < query_traj_teacher_forcing_until_iter`

## 2026-05-15

### Matched-only Trajectory Head Input (Train)

**핵심 내용**: 학습 시 헝가리안 매칭을 trajectory head 이전에 수행하고, matched 쿼리만 trajectory head에 입력.

**주요 변경사항**:
- `query_head.py`: `apply_lifted_centers_to_outputs`에 `defer_trajectory: bool = False` 인자 추가. `True`이면 `traj_motion_input_tqd2`는 계산하되 `_predict_trajectory_from_inputs`는 건너뜀.
- `efficientocf.py`: `extract_feat_query`에 `defer_trajectory=False` 인자 추가. `traj_motion_input_tqd2`를 반환 튜플에 추가 (30번째 원소).
- `efficientocf.py`: `forward_train`에서 `extract_feat_query(defer_trajectory=True)` 호출. 헝가리안 매칭 후 matched K개 쿼리에 대해서만 trajectory 예측 → zero-scatter `[F, Q=100, 2]` → vis geometry 재빌드.

**설계 제약**: 헝가리안 매칭 cost는 과거/현재 프레임 기반 피처(center, sigma, cls logits, attn)만 사용 가능 (trajectory 미래 정보 배제).

**Config 키**: `query_traj_matched_only: bool = False` (`MODEL_CFG_DEFAULTS`). `True`로 설정 시 matched-only 동작 활성화.

**Inference 영향**: 없음 (`query_traj_matched_only=False` 기본값, test path 변경 없음).
