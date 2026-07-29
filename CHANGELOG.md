# Changelog

## 2026-07-28 — trajectory xy-refine 추론 적용 스위치 (EOCF_EVAL_TRAJ_REFINE)

- 기존에 `refine_trajectory_absolute_xy`(query_head.py:1620) 호출부는 `forward_train`
  (efficientocf.py:3550) 하나뿐이라 refine head가 **학습에서만** 적용되고 eval/oracle은 보정 전
  base 궤적을 썼다(train/eval mismatch, 체크포인트의 refine 가중치는 추론에서 미사용).
- `extract_feat_query`에서 traj 출력을 꺼낸 직후(efficientocf.py:1173~) opt-in 분기를 추가했다.
  `EOCF_EVAL_TRAJ_REFINE=1` + `query_traj_xy_refine_enabled=True` + `not self.training`일 때만
  `traj_offsets_fq2`를 refined offset으로 교체한다. 교체 지점이
  `_build_query_trajectory_geometry_from_present`와 반환 tuple보다 앞이라 metric·viz·oracle이
  전부 같은 refined 궤적을 쓴다. `not self.training` 게이트는 forward_train과의 이중 적용 방지용.
- `eval_total.sh` / `eval_oracle.sh` 양쪽에 `EOCF_EVAL_TRAJ_REFINE=1`을 추가했다. 두 스크립트는
  항상 같은 값이어야 한다 — 다르면 oracle vs baseline 비교에 궤적 보정 유무가 섞인다.
- 자매 레포 `Autonomous_Driving_26_filter_ablation_query`에도 동일 변경 적용(삽입부 diff 없음).

## 2026-07-23 19:58 KST — asset-all 학습 config visibility 필터 적용

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_filter.py`의 train/test dataset과
  `LoadInstanceWithFlow`에 `filter_f3_first_visibility=True`를 연결했다. 과거 3프레임 최초
  annotation이 `visibility_token==1`인 f3 raw instance ID가 train asset-union D,
  E(AABB), E에서 파생되는 center/valid/ID의 t0~t6 전체에서 제외된다.
- train/test `Collect3D.meta_keys`에 `gt_visibility_drop_ids`를 전달해 eval lazy AABB/rot 및
  asset 지표도 동일 blacklist를 사용하도록 맞췄다. filter 실험의 query/cam/mixture/attention
  시각화 경로를 전용 work_dir 아래로 분리했다.
- config 파싱·문법·배선 검사를 통과했다. asset train 캐시와 f3 캐시가 23,930개 sample key를
  전부 공유하며 무작위 100개 실제 7프레임 샘플의 instance ID 집합도 모두 일치해 f3 blacklist를
  asset instance 열에 적용할 수 있음을 확인했다.

## 2026-07-23 19:39 KST — visibility 필터 전후 D/E 시각 검증

- 실제 val 시퀀스 3개에 대해 D(nusocc instance3d)와 E(AABB)를 각각 `before`,
  `after`, `removed`로 나눈 7프레임 BEV 비교 이미지를
  `data_vis/visibility_filter_compare/`에 생성했다.
- 제거 대상은 instance별 동일 색상, 유지 대상은 before에서 회색/after에서 초록색으로 표시하고
  현재 프레임 t2는 노란 테두리로 구분했다. 세 샘플 모두 동일 blacklist가 D/E의 t0~t6 전체에
  적용되며 `before = after + removed`인 것을 픽셀 수와 시각 결과로 확인했다.

## 2026-07-23 19:31 KST — f3 최초등장 visibility=1 런타임 필터 추가

- `subset_attn_cover_pyr_aabb_dice3d_new_filter.py`에서만 과거 3프레임 내 instance 최초
  annotation의 `visibility_token==1`을 판정해 해당 f3 raw instance ID를 7프레임 전체에서
  제외하도록 활성화했다. 최초 visibility가 2~4면 이후 과거/미래 프레임의 visibility=1은
  제거 조건으로 사용하지 않는다.
- dataset이 원본 nuScenes annotation 순서와 `vehicle`/`human` 포함 규칙으로 f3 raw ID를 별도
  재현한다. 기존 `instance_dict`는 저가시성·사람 필터 후 ID라 f3와 다르므로 사용하지 않는다.
- `LoadInstanceWithFlow`가 D(`gt_occ_inst`)와 E(`gt_bbox_aabb`)를 읽은 직후 동일 blacklist로
  sparse row를 필터링하고, 그 다음 center/valid/ID/size를 생성한다. 따라서 matching/class,
  focal/Dice, center/trajectory와 query 시각화가 모두 필터된 D/E에서 파생된다.
- attention softargmax 시각화의 bbox 입력을 legacy fallback 대신 실제 필터된 AABB(E)로 수정했다.
  filter 실험의 query/cam/mixture/attn 시각화 출력 경로도 전용 work_dir 아래로 분리했다.
- test meta에도 blacklist를 전달해 eval lazy AABB/rot 및 부가 asset 지표가 동일 ID를 제거하도록
  맞췄다. 관련 파일 `py_compile`, config/dataset/loader build를 통과했다. val 100시퀀스에서
  D 429,613행/E 9,169,093행 제거, 후속 과거 visibility=1 객체 162개와 미래 visibility=1 객체
  307개 유지 확인. 실제 loader 샘플에서도 제거 ID의 D/E row와 파생 center ID가 모두 0개였다.

## 2026-07-23 18:29 KST — Asset IoU3D TP/FP/FN eval 출력 추가

- `IOU_3d_asset` 계산에서 이미 만들어지던 voxel TP/FP/FN을 버리지 않고 detector 결과로 반환하도록
  연결했다. 기존 Asset GT, 3D 정렬, occupancy threshold와 frame 범위를 그대로 공유한다.
- single/multi-GPU eval에서 성분을 전체 샘플·rank에 걸쳐 합산하고 final/live 로그에
  `[iou3d comps asset] TP=... FP=... FN=... | micro IoU3d=...`를 출력한다.
- dataset 최종 결과에 `IOU_3d_asset_TP`, `IOU_3d_asset_FP`, `IOU_3d_asset_FN`,
  `IOU_3d_asset_micro`를 추가했다. 기존 `IOU_3d_asset` sample-macro 평균은 변경하지 않았다.
- 관련 파일 `py_compile`, 합성 TP/FP/FN 계산과 dataset aggregation smoke test를 통과했다.

## 2026-07-23 16:22 KST — Asset scene BCE 빈-instance 배치 temporal guard 수정

- 첫 8GPU 실행에서 scene BCE는 현재+미래 5프레임으로 4 iteration까지 정상 계산됐지만,
  다섯 번째 배치의 GT center full-timeline pack이 비어 `use_full7_temporal_sup=False`가 되자
  BCE가 불필요하게 full7 center 정렬을 요구하며 rank 4에서 종료되는 문제를 확인했다.
- scene BCE는 GT instance center를 사용하지 않으므로, full-timeline 선택 조건을 예측 Gaussian
  timeline과 Asset occupancy GT의 존재 여부로 분리했다. full timeline을 사용할 수 없는 경우에는
  정렬된 present+future horizon으로 안전하게 fallback한다.
- 관련 detector/loss 파일 `py_compile`, config parse와 checkpoint 존재 여부를 다시 확인했다.
  scene BCE 가중치 `0.1`, hard-pair 비활성, 3 epoch/LR `1e-5`/warmup 100 설정은 변경하지 않았다.

## 2026-07-23 16:09 KST — 현재+미래 all-query Asset scene BCE fine-tuning 추가

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft_bce.py`에서 현재 1+미래 4프레임을 각각
  하나의 scene occupancy로 렌더하고 Asset 합집합 GT와 foreground-normalized BCE를 계산하도록
  활성화했다. 모든 200 query×48 Gaussian을 사용하며 query foreground confidence를 연속 opacity로
  곱한다. Hungarian/hard mining, score/occupancy threshold, NMS, top-k는 이 loss 경로에 없다.
- positive/negative voxel 가중치는 `1.0/1.0`, scene loss 가중치는 `0.1`이다. 프레임별 BCE 합을
  GT foreground voxel 수로 정규화한 뒤 현재+미래 5프레임을 평균한다. 빈 GT 프레임은 background
  평균 BCE만 계산한다.
- 평가용 in-place scene renderer는 backward 시 version error가 나므로, query별 full volume 대신
  실제 Gaussian bbox patch만 모아 `scatter_reduce(amax)`로 단일 scene을 만드는 differentiable
  경로를 추가했다. 기존 eval 경로와 합성 입력 출력 최대 오차 `0`, center/sigma/yaw/Gaussian
  opacity/query confidence gradient 유효를 확인했다.
- hard-pair gate는 config에 명시적으로 `False`로 고정했다. covariance/direction shape loss도
  기본 비활성이다. full epoch-15 checkpoint는 유지하고 전체 FT LR을 `1e-5`, image backbone을
  `1e-6`, linear warmup을 100 iter, cosine 3 epoch로 보수적으로 조정했다.
- 관련 Python 파일 `py_compile`, MMCV config/model/optimizer build, scene BCE 단독 backward smoke
  test를 통과했다. 지정 epoch-15 checkpoint도 정상 로드됐고 epoch metadata=15를 확인했다
  (출력된 unexpected key는 기존 checkpoint의 non-persistent buffer들뿐이며 학습 파라미터 누락은 없음).

## 2026-07-23 13:37 KST — Dice-loss top-k matched-pair Asset FT 추가

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft_hard_dice.py`를 full epoch-15 기반
  3-epoch fine-tuning 설정으로 구성했다. Hungarian matching 뒤 각 matched pair의 raw Dice loss를
  계산하고, rank별 상위 30%(`ceil`, 최소 1 pair)를 hard pair로 선택한다.
- 선택된 동일 hard pair에만 GMO focal/Dice loss를 적용하며 easy pair weight는 `0.0`으로 설정했다.
  선택 점수는 `detach`하므로 top-k 선택 자체에는 gradient가 없고, 선택된 loss에만 역전파된다.
- 공용 설정에 `geometry`/`loss_topk` 모드와 top-k 비율, easy weight, 최소 pair 수를 추가했다.
  기본 모드는 기존 `geometry`라 기존 config 동작은 유지된다.
- large/far 기준은 해당 설정에서 선택 조건이 아니라 진단값으로만 유지했다. hard/easy raw loss,
  선택 비율, Dice cutoff 및 선택된 hard 중 large/far 수를 DBG로 기록하며 hard DBG가 로그 필터에서
  제거되던 문제도 함께 수정했다.
- optimizer는 전체 `3e-5`, image backbone `3e-6`만 사용하고 특정 Gaussian head 2배 LR은 제거했다.
  기존 200-iteration warmup, cosine schedule, full epoch-15 `load_from`은 유지했다.
- 관련 파일 `py_compile`, 신규/기존 config parse, full model/optimizer build와 합성 gradient 검증을
  통과했다. 합성 검증에서 Dice loss 상위 pair에만 focal/Dice gradient가 생기고 geometry 모드의
  기존 large/far 선택도 유지됨을 확인했다.

## 2026-07-23 11:16 KST — 큰·원거리 matched-pair Asset FT 추가

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft_hard.py`에서 Hungarian matching은 유지하고,
  실제 annotation `max(w,l) >= 6m` 또는 present-frame LiDAR/BEV 거리 `>= 30m`인 pair만
  Asset focal/Dice loss에 포함하도록 설정했다. 회전 조건은 현재 미래 shape가 프레임 불변인 구조적
  한계 때문에 포함하지 않았다.
- 공용 hard-pair gate/threshold 기본값을 `efficientocf_config.py`에 비활성 상태로 추가하고,
  `utils_loss.py`에서 matched GT id 기준 실제 크기와 present-frame 중심 거리를 정렬해 focal/Dice만
  필터링했다. hard가 없는 rank는 기존 graph-connected zero loss 경로를 유지한다.
- 전체/hard/easy 및 large/far/overlap pair 수, metadata 유효 수, hard 비율·평균 크기·거리,
  hard/easy focal·Dice raw loss DBG를 추가했다.
- 전체 LR `3e-5`, image backbone `3e-6`은 유지하고 Gaussian offset/sigma/yaw/weight 출력 head만
  `6e-5`로 설정했다. 기존 200-iteration linear warmup, 3-epoch cosine, full epoch-15 load_from은 유지했다.

## 2026-07-23 10:31 KST — Asset focal/Dice 기본 fine-tuning 설정 분리

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft.py`에서 covariance/rendered-direction
  shape loss와 관련 진단 플래그를 제거하고, 기존 full epoch-15 모델을 asset-union focal/Dice로
  미세조정하는 설정만 유지했다.
- `train_ft.sh`가 새 `_asset_all_ft.py` 설정을 실행하도록 변경했다.

## 2026-07-21 18:44 KST — `_cov_weight` detached-opacity covariance variation 추가

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_cov_weight.py`에서 shape loss weight `0.1`은
  유지하고, 예측 Gaussian 중심 공분산만 detached opacity로 가중하도록 활성화했다.
- 공용 설정 `query_gmo_shape_pred_weight_mode`을 추가했다. 기본 `equal`은 기존 cov 수식을
  유지하며, `detached_opacity`는 opacity를 정규화해 weighted mean/covariance에 사용하되
  detach하여 shape loss가 opacity head를 직접 조절하지 못하게 한다.
- 검증: 기존 equal 수식과 최대 오차 `2.38e-7`, offset gradient 유효, opacity gradient 없음,
  기존 cov=`equal`/신규 cov_weight=`detached_opacity` 모델 build 및 py_compile 통과.

## 2026-07-21 11:30 KST — `_asset_all_cov` 학습 전용 covariance shape loss 추가

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_cov.py`에서만 matched asset-union GT BEV와
  48개 Gaussian 중심의 trace-normalized XY covariance를 비교하는 `loss_gmo_shape`를 활성화했다.
- shape loss weight는 `0.1`, 유효 GT 하한은 low-res BEV 4 voxel이며 opacity는 공분산 가중에
  사용하지 않는다. 기존 focal/Tversky/Gaussian/추론 설정은 유지했다.
- 공용 config 기본값은 비활성(`False`, `0.0`)이라 다른 config와 추론에는 영향이 없다.
- raw loss, valid pair-frame 수, 장축 각도 오차와 예측/GT 장단축비 debug 로그를 추가했다.
- 검증: 관련 파일 `py_compile`, MMCV config parse, 동일/45도 회전/정사각형 covariance 합성 검증 통과.

## 2026-07-20 19:39 KST — `_asset_all` Dice3D를 asset-union 단독 감독으로 변경

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all.py`의 Dice3D 비율을 inst3d/AABB `0.0/1.0`에서
  `1.0/0.0`으로 변경했다. 해당 config의 train `gt_occ_inst`는 병합 key
  `segmentation_instance_saved_list2`이므로 실제 Dice GT는 기존 inst3d와 asset의 합집합이다.
- focal은 기존대로 asset-union을 사용하며 center/trajectory/ID/size용 AABB 배선은 유지했다.

## 2026-07-20 19:23 KST — asset-union eval 지표 추가

- `efficientocf.py`에 eval 전용 asset GT lazy-loader를 추가했다. 기본 경로는
  `nuscenes_gmo_full_hybrid_solid_data_v1/val/segmentation_instance3d`, key는 train과 같은
  `segmentation_instance_saved_list2`이며 `EOCF_ASSET_GT_DIR`로 override할 수 있다.
- 기존 정렬을 그대로 적용해 `IOU_2d_asset`, `IOU_3d_asset`을 계산하고, 기존 inst3d를 필수 GT,
  asset-union을 추가 예측 허용 영역으로 쓰는 `Recall3d(asset)` 및 TP/FP/FN/assetFP 성분을 추가했다.
- single/multi-GPU 결과 수집, distributed running/final 로그와 dataset 최종 평가 출력을 모두 배선했다.

## 2026-07-20 15:18 KST — `_asset` train inst3d를 asset 보강 합집합 GT로 전환

- `subset_attn_cover_pyr_aabb_dice3d_new_asset.py`의 train `gt_occ_inst`를
  `nuscenes_gmo_full_hybrid_solid_data_v1/train/segmentation_instance3d`의
  `segmentation_instance_saved_list2`로 연결했다. 이 key는 기존 sparse inst3d를 보존하고 빈 xyz에
  asset voxel을 추가한 병합 완료 GT다.
- test `gt_occ_inst`와 AABB GT 경로는 기존 `efficientocf_gt_f3`를 유지한다.
- GMO focal은 병합 inst3d를 사용하고, Dice3D는 기존 설정대로 inst3d/AABB `0.0/1.0`을 사용한다.

## 2026-07-20 15:18 KST — `_asset` Dice3D를 AABB 단독 감독으로 변경

- `subset_attn_cover_pyr_aabb_dice3d_new_asset.py`에서 Dice3D 혼합 비율을
  inst3d/AABB `0.1/0.9`에서 `0.0/1.0`으로 변경했다.
- focal 비율(inst3d/AABB `1.0/0.0`)과 데이터 및 train/test 배선은 변경하지 않았다.

## 2026-07-16 18:56 KST — AABB 크기 기반 연속 Tversky (`_tver`)

- `subset_attn_cover_pyr_aabb_dice3d_new_tver.py`에서만 크기 스케줄 활성화. 현재 연결된
  `efficientocf_gt_f3/GMO/segmentation_aabb`에서 loader가 만든 `gt_instance_sizes`의 XY 최대 span을
  사용해 6m 이하 `α/β=0.7/0.3`, 10m 이상 `0.4/0.6`, 중간은 선형 보간한다.
- `efficientocf_config.py`: 크기 스케줄 gate/start/end/alpha_end 기본값 추가. gate 기본값은 `False`라
  기존 config 동작은 유지한다.
- `efficientocf.py`, `utils_loss.py`: matched instance id로 AABB size를 정렬해 pair별 Tversky α/β를
  적용. size가 없거나 id 정렬에 실패한 pair는 기존 고정 α/β로 fallback한다.
- 대상 config의 train `Collect3D`에 `gt_instance_sizes`를 추가해 loader에서 만든 AABB 크기가
  `forward_train`까지 전달되도록 배선했다. 최초 실행 18 iter에서 이 key 누락으로 size fallback만
  동작한 것을 TensorBoard(`alpha_mean=0.7`, `large_pair_count=0`)로 확인 후 보완했다.
- 디버그 값 `dbg_gmo_dice_alpha_mean`, `dbg_gmo_dice_beta_mean`,
  `dbg_gmo_dice_size_large_pair_count`를 추가했다.

## 2026-07-16 18:04 KST — 특정 config용 learnable KV pyramid downsampling

- `transformer.py`: coarse-to-fine KV pyramid의 고정 average pooling을 선택적으로 대체하는
  depthwise stride conv + pointwise 1x1 conv downsampler 추가. average pooling + identity와 같은
  초기값을 사용하고 마지막 full-resolution layer는 identity로 유지.
- `efficientocf_config.py`, `efficientocf.py`: `query_transformer_learnable_kv_downsample` gate 추가.
  기본값은 `False`라 기존 config 동작은 유지.
- `subset_attn_cover_pyr_aabb_dice3d_new_conv.py`에서만 gate를 `True`로 활성화.

## 2026-07-15 KST — subset_attn_cover_pyr_aabb_dice3d_new.py: 학습 GT를 rot→AABB로 전환

- **문제**: 이 config는 파일명이 `aabb`인데 실제 GT 배선은 `segmentation_rot`이었음(`_rot_new` 복사 후
  GT 미복원). `gt_bbox_aabb` 슬롯 하나가 train에서 `results['gt_bbox_aabb']`(dice bbox 0.9 타깃)·
  `gt_instance_centers_world`(center)·`gt_instance_ids`·`gt_instance_dims`(attn/size) 4개를 전부 파생
  (loading_instance.py:1314-1327) → dice·center·attn·size 감독이 전부 회전 박스 기준이었음.
- **변경(4줄, train+test)**: `gt_bbox_aabb_subdir` `segmentation_rot`→`segmentation_aabb`,
  `gt_bbox_aabb_key` `..._rot_saved_list2`→`..._aabb_saved_list2` (loader 기본값과 동일). 헤더/인라인
  주석도 aabb 서사로 갱신. `segmentation_rot` 잔여 참조 0 확인.
- **test도 바꾼 이유**: test Collect3D엔 `gt_bbox_aabb`(dice)가 없어 핵심 지표(IoU/Recall)엔 무관하나,
  `gt_instance_centers_world`/`gt_instance_ids`는 넘어감(eval vis·oracle-match·train-faithful 진단용).
  학습 감독(aabb)과 eval-time center 기준을 통일하려 val도 aabb로 맞춤.
- **안전성**: `data/efficientocf_gt_f3/GMO/segmentation_aabb` 29,049 파일 존재(rot과 동수). 이 config로
  학습/실행 중인 프로세스 없음(train_total*.sh 미참조) → pipeline 수정 리스크 없음. 파일명↔내용 일치 회복.

## 2026-07-14 KST — eval에 Recall3d(bbox_rot) 지표 추가 (관용 영역 AABB→rot OBB)

- **동기**: 기존 `Recall_3d`는 base(TP/FN/FP)=inst3d(nusocc)인데 관용항(bbox_fp)=**AABB box**라,
  축정렬로 퍼진 pred가 회전된 차 바깥·AABB 안 모서리로 새도 관용받아 penalty를 안 먹었음. `_rot`
  실험(회전 학습)의 효과를 측정하려면 관용 영역을 rot OBB로 좁힌 버전이 필요.
- **추가 지표**: `Recall3d(bbox_rot)` = base는 inst3d 그대로, 관용만 `segmentation_rot`(rot OBB)로
  교체. rot⊂aabb라 관용이 좁아져 **방향까지 맞아야 관용** — 더 엄격한 recall. 기존 `Recall_3d`는
  `Recall3d(bbox_aabb)`로 리네임(IoU 테이블 aabb/rot 네이밍과 통일). comps(TP/FP/FN/bboxFP/micro)도 동일 분리.
- **변경 파일(전부 eval 전용 경로 — 학습 중 run에 무영향)**:
  - `efficientocf.py` `simple_test`: `_iou_recall_3d(pred, gt3d=inst3d, bbox3d=rot3d_t, valid3d=valid_arg)`
    한 줄 추가(기존 rot IoU 계산 바로 뒤, `rot3d_t` 재활용). init/정상 return/empty return 3곳에
    `recall_3d_rot`·`recall_3d_rot_comps` 배선. `_iou_recall_3d` 자체는 무수정(bbox3d 인자 기존 지원).
  - `apis/test.py`: single/multi-gpu 두 루프에 `recall_3d_rot_metric`·`r3d_rot_comps_sum`(4-vector) 누적
    + res 배선. multi-gpu는 `collect_results_cpu`로 rank 합산(기존 aabb comps와 동일 규약).
  - `efficientocf_dataset.py` `evaluate`: `Recall3d(bbox_rot)` 및 comps 집계, 기존 키 aabb로 리네임.
- **판정법**: 축정렬 blob 상태면 `Recall3d(bbox_rot)` < `Recall3d(bbox_aabb)`로 벌어짐(AABB 모서리
  누출이 rot에선 비관용FP로 잡힘). 회전을 학습하면 격차가 좁혀짐.
- **주의**: 기존 로그의 `Recall_3d*` 키명이 `Recall3d(bbox_aabb)*`로 바뀜 — 이전 eval 결과와 키로
  join하던 도구가 있으면 갱신 필요.
- **live 로그(eval_metrics_live.log) 반영(2차)**: 최초엔 최종 `evaluate()` 출력에만 rot을 넣어서
  running용 `eval_metrics_live.log`엔 안 보였음. `_running_eval_msg`/`_distributed_running_eval_msg`에
  `recall3d_rot` 인자 + `Recall3d(bbox_aabb)`/`Recall3d(bbox_rot)` 둘 다 출력하도록 수정, 4개 호출부에
  `recall_3d_rot_metric` 전달. multi-gpu comps_msg도 `_comps_line(tag, ...)` 헬퍼로 리팩터해 `bbox_aabb`
  ·`bbox_rot` 2줄 출력(all_reduce는 전 rank 공동 호출 유지).
- **512샘플 실측(epoch_7)**: Recall3d(bbox_aabb)=0.197 → (bbox_rot)=0.143(−27%). bboxFP 21.68M→15.32M
  (차 6.36M = AABB 안·OBB 밖 누출). 단 전체 FP 132M 중 관용대상은 16%뿐 — 지배적 문제는 여전히
  전반 과확장(비관용FP 110M), 회전 누출은 2차.

## 2026-07-13 KST — [new_data 레포] GT 파이프라인 compute_D 버그 발견 + D/E/F 재생성 착수

- **버그 (GT 생성 스크립트, 데이터에 박제된 상태)**: `tools/gen_data/gen_new_gt_pipeline.py`의
  `compute_D`가 `occ = (dense != 0) & (dense != 255)`로 "AABB 안에서 아무거나 점유된" voxel을
  전부 keep → 도로(11)/건물(15)/초목(16) 등 이물질 voxel이 인스턴스 id를 달고 D(inst3d)에 들어감.
  - 실측: sample(1b5ef5ec...) D voxel 2,567개 중 자기 class 일치 79%, 도로/지면 8%, 기타(건물 등)
    13%. 회전각이 큰 차일수록 심함(축정렬 차는 AABB≈rot박스라 영향 미미 → "맞는 것도 많고 틀린
    것도 있는" 현상의 원인). 이물질은 주로 바닥 z층에 몰림.
  - `_gt`와 `_f3` 모두 같은 D 내용(실측 동일) — 둘 다 오염. E/F는 모양은 정상(A/B에서 옴),
    포함 목록만 D 게이트에 의존.
  - f3의 정체 확정: `_gt`에서 `derive_filtered_gt.py`(0630 레포, `--require past3all --keep-human`)로
    파생된 관측창(0∩1∩2) 필터판. 30샘플 실측으로 필터 의미 검증(제거분 = 교집합 밖 인스턴스, 정확 일치).
    id 재부여 없음(갭 있는 id가 정상).
- **수정 방식**: class-match 교집합 — `dense[x,y,z] == row.cls`인 voxel만 keep. 미리보기 검증:
  `data_vis/f3_check/classmatch_sample_{02,04,06}.png` (기울어진 차들이 F(rot)와 같은 각도 회복).
- **`tools/gen_data/regen_def_clsmatch.py` (신규)**: 기존 2단계(gen → derive_filtered)를 한 번에.
  A/B(raw, 버그 무관)는 `_gt` 저장본 재사용(rasterize 스킵 + **id가 A/B에 박제돼 있어 재번호
  위험 구조적으로 0**), C(점유)만 재계산 → class-match D → E/F 재게이트 → f3 필터(past3all,
  keep-human) → D/E/F 3레이어만 저장(A/B 미저장, 디스크 절약). train_capacity=23930 강제로
  subset config로도 전 키 커버.
- **8키 테스트 검증**: `data_vis/f3_check/regen_test_01~08.png` — id 정합 8/8 OK(새 id ⊆ 옛 id),
  D 12~48% 감소(이물질 제거), E/F 거의 동일(유령 인스턴스만 게이트 탈락).
- **전체 재생성 완료 (raw-먼저 방식)**: `--no-f3-filter`로 6 shard →
  `/home/hwanhee/datasets/efficientocf_gt_DEF_new` (raw D/E/F, 29,049키 × 3레이어, ~2.5시간).
  이어서 `derive_filtered_gt.py`(0630에서 md5 동일 복사, past3all+keep-human) 6 shard →
  `/home/hwanhee/datasets/efficientocf_gt_f3_new` (29,049키 × 3레이어, 0 skip).
- **검증 결과**: ①키 목록 — 새 raw = 기존 `_gt` = 기존 `_f3` 완전 일치(29,049). ②id 정합 —
  29,049키 **전수** 검사, 새 D id ⊆ A(bbox_aabb_raw) id 위반 0건(재번호 없음 보장). ③참고: 검사
  중 "옛 D와 다른 id 구성" ~21키 발견됐으나 전부 A에 존재하는 정당한 인스턴스 — 옛 파이프라인이
  dedup/any-occupied 부작용으로 잃었던 인스턴스가 class-match에서 복원된 케이스(개선, 버그 아님).
  ④f3 필터 동작 — 샘플별 raw 대비 인스턴스 0~5개 제거(관측창 교집합 밖) 확인.
- **시각화**: `data_vis/0713/` — `sample_01~05.png`(생성 중간 점검), `gt_sample_01~05.png`(새 raw vs
  기존 `_gt`), `f3_sample_01~05.png`(새 f3 vs 기존 `_f3`, 같은 5개 토큰) — D old의 축정렬 뭉개짐이
  D new에서 F(rot) 각도로 복원되는 것 일관 확인.
- **교체·삭제 완료 (사용자 최종 승인 후 실행)**: ①교체 직전 최종 정합 — 파일 수(2루트×3레이어
  전부 29,049), npz 무결성(600파일 손상 0), f3 rows ⊆ raw rows(위반 0), f3 필터 규칙(0∩1∩2 교집합)
  100/100키 정확 일치(인스턴스 24% 제거 — 옛 17%보다 큰 건 class-match로 유령 인스턴스가 추가로
  걸러진 것, 의도된 결과). ②`_gt`의 D/E/F 3폴더 교체 + `_f3` 루트 이름 스왑. 생성 로그는
  `efficientocf_gt/logs_regen_def_20260713/`에 보관. ③교체 후 기능 검증 — 레포 심링크·로더
  경로해석(inst3d/rot)·npz 키/7프레임 계약·eval lazy-loader 경로 전부 정상. ④옛 오염 데이터
  (`*_OLD_BROKEN` 4개)와 스크랩(`_clsmatch` 부분생성·`_TESTRUN`) 삭제 — 디스크 132→328GB 여유.
- **재학습 전 필수**: `work_dirs/subset_attn_cover_pyr_aabb_dice3d_rot/`의 옛 체크포인트(오염 GT로
  학습됨)를 비우고 from-scratch 시작할 것 — `train_total.sh`의 `--resume`이 latest.pth를 물면 안 됨.

## 2026-07-13 KST — [new_data 레포] `_rot` 학습 2차 크래시 수정: D(inst3d) 경로도 동일 shape 문제

- **증상**: 위 bbox shape 수정 후 재실행 → epoch1 iter4까지 정상 → iter5에서 이번엔 **전체 rank가**
  `AssertionError: loss log variables are different across GPUs!`로 동시에 죽음
  (`logs/subset_attn_cover_pyr_aabb_dice3d_rot/20260713_142302.log`). rank0=382키, rank4=375키 —
  정확히 7개 차이(diff로 확인): `dbg/gmo_dice_{alpha,beta,pair_count,inst3d,bbox,bbox_pair_count}`
  + `loss_query_attn_sigma`.
- **원인**: 바로 위 NOTES에서 "이론상 있을 수 있다"고 남겨둔 D(inst3d)/`gt_occ_inst` 경로의 동일
  shape-추론 버그가 실제로 발동함. `_prepare_gt_occ_inst_primary_targets`가 빈 sparse에서
  `None`을 리턴하면 `gt_instance_occ3d_txyz_primary`→`match_gt_instance_occ3d_txyz`가 `None`이
  되고, `efficientocf.py`의 matched-pair GMO loss 호출부(~3583)와 attn bbox/sigma loss 호출부
  (~3542)가 `torch.is_tensor(...)` 가드로 **함수 호출 자체를 통째로 스킵** — 이 두 함수 내부의
  "항상 채워지는 기본값(z placeholder)" 패턴(`_compute_matched_pair_gmo_losses` 최상단 `out={...}`
  등)까지 아예 실행이 안 돼서 그 rank만 키가 통째로 빠짐. rank4만 죽은 건 그 rank가 이 iter에서
  마침 GMO 인스턴스 0개인 샘플을 뽑았기 때문(우연, 배선 문제 아님).
- **수정 (`utils_gt_prep.py`)**: `_prepare_gt_occ_inst_primary_targets`와
  `_build_instance_center_world_targets_from_sparse`에도 동일하게 `spatial_shape=(voxelizer.W,H,D)`를
  전달하도록 수정(`_build_instance_center_world_targets_from_sparse`에도 `spatial_shape` 파라미터
  신규 추가) — 이제 D 쪽도 빈 샘플에서 `None` 대신 all-zero dense/빈(N=0) center 텐서를 리턴해서
  `match_gt_instance_occ3d_txyz`가 항상 유효한 텐서가 되고, matched-pair GMO/attn loss 함수들이
  항상 호출되어 자기 내부의 z-placeholder 기본값으로 전체 키 집합을 채움 → rank 간 키 개수 불일치
  해소.
- **검증**: 실제 함수로 (a) 빈 sparse(7프레임 전부 0행) → `bundle`이 `None` 대신 dict 리턴,
  `dense_inst_txyz` all-zero `[7,512,512,40]`, `centers_world_tn3` `[7,0,3]`(빈 N) 확인, (b) 정상
  sparse(row 있음) → 기존과 동일하게 올바른 위치에 값 채워지고 `instance_ids_n`/`centers_valid_tn`
  정상 확인. `py_compile` 통과.
- **후속 조치**: 이 커밋 직후 `full_attn_cover.py`(사용자 소유, 8GPU 점유 중이던 별도 job)를
  SIGTERM으로 정상 종료(GPU 전부 회수 확인) 후 `train_total.sh`를 8GPU로 재실행(`USE_MPS=0`,
  단독 실행이라 MPS 불필요) — 사용자 명시적 승인 하에 Claude가 직접 진행.

## 2026-07-13 KST — [new_data 레포] `_rot` 학습 크래시 수정: bbox dense shape 추론 실패

- **증상**: `train_total.sh`(`subset_attn_cover_pyr_aabb_dice3d_rot.py`) 실행 → epoch1 iter4까지
  정상(loss 값 정상 출력) → iter5에서 rank4가 `ValueError:
  query_gmo_{focal,dice}_bbox_weight>0 requires gt_bbox_aabb from the pipeline` 로 죽음
  (`logs/subset_attn_cover_pyr_aabb_dice3d_rot/20260713_141357.log`).
- **원인**: `utils_gt_prep.py:_infer_dense_shape_from_sparse`가 sparse row에서 x/y/z 최댓값을 보고
  dense grid 크기를 "추론"하는데, 그 샘플 윈도우(7프레임)에 GMO 인스턴스가 하나도 없어서(정상적으로
  발생 가능한 케이스 — 물체 없는 씬) rot sparse row가 전부 비어있으면 추론할 게 없어 `None`을 리턴 →
  `_build_dense_from_gt_occ_inst_sparse`도 `(None,None)` → `_prepare_gt_bbox_aabb_dense_txyz`도
  `None` → `forward_train`(efficientocf.py:2693)이 이걸 하드 에러로 처리. rank4만 죽고 다른 rank가
  iter4까지 살아있었던 건 이 샘플이 그 rank에만 배정된 데이터 문제였기 때문(배선 버그 아님). 같은
  구조(`dice_bbox_weight>0` + AABB/rot 기반 bbox GT)를 쓰는 다른 config(`subset_attn_cover_pyr_aabb_dice3d.py`
  등)도 이론상 동일 취약점 있음 — 지금까지 안 터진 건 우연히 빈 씬 샘플을 안 뽑았을 뿐.
- **수정 (`utils_gt_prep.py`)**: `_build_dense_from_gt_occ_inst_sparse`에 `spatial_shape` 파라미터
  추가(기본 None, 기존 호출부 전부 무변화) — sparse row에서 추론하는 대신 호출자가 미리 알고 있는
  고정 grid 크기를 바로 쓸 수 있게 함. `_prepare_gt_bbox_aabb_dense_txyz`에서
  `self.voxelizer.W/H/D`(항상 알려진 고정값)를 `spatial_shape`로 넘기도록 수정 — 이제 rot sparse가
  전부 비어도 shape 추론에 실패하지 않고 all-background dense tensor(정상적으로 "이 샘플엔 bbox GT
  없음"을 의미)를 만들어 리턴함. `_prepare_gt_occ_inst_primary_targets`(D/inst3d 경로)는 이번엔
  안 건드림 — 그쪽은 하드 크래시 없이(여러 `torch.is_tensor` 가드로) 이미 관대하게 처리되고 있어서
  범위 밖으로 둠(NOTES 참고).
- **검증**: 실제 함수로 (a) 크래시 재현 케이스(7프레임 전부 빈 sparse row) → 이제 `None` 대신
  `[7,512,512,40]` 전부-0 텐서 리턴 확인, (b) 정상 케이스(row 있음) → 기존과 동일하게 올바른 위치에
  instance id가 채워지는지 확인. `py_compile` 통과.

## 2026-07-13 KST — [new_data 레포] `_prepare_gt_instance_classes_for_matching` 벡터화 (캐싱 대신)

- **배경**: GT metadata 캐시(②) 도입을 검토하다가, `query_present_only=True`(이 레포 활성 config
  전부)라 class 판정 직전에 이미 7프레임→1프레임으로 줄어든다는 걸 확인 — 실측 비용이 예상보다
  작아서(T=1 기준 K=5~70에서 274ms~2488ms) 캐싱 인프라(mixin/config/심링크/raw-compact 분리, 15
  epoch 전체 기준 절약분 ≈ 8.6분)를 새로 만들 실익이 작다고 판단, 이 함수 자체를 벡터화하는 쪽으로
  방향 전환(캐싱 안 함).
  - 인스턴스별 class가 항상 단일값인지(다수결이 실제로 필요한지)도 실측 확인: `data/efficientocf_gt_f3/GMO/segmentation_instance3d`
    실 파일 200개·인스턴스 4,211개 전수 조사 — class 섞인 인스턴스 0건(0.000%). class·instance가
    같은 sparse row(`[x,y,z,cls,inst]`)에서 동시에 나와 구조적으로 섞일 수 없음(`_build_dense_from_gt_occ_inst_sparse`
    참고). 다만 다수결/tie-break 코드 자체는 안전망으로 유지(벡터화 버전도 동일 semantics 보존).
- **`projects/occ_plugin/occupancy/detectors/utils_gt_prep.py`**: `_prepare_gt_instance_classes_for_matching`의
  "인스턴스 K개만큼 grid 전체를 반복 스캔"하던 for-loop를 `torch.unique(pairs, dim=0)` 1회 호출로
  교체 — (instance,class) 쌍을 grid 전체에서 한 번에 집계한 뒤, 그 결과(보통 K개 안팎의 작은 목록)만
  놓고 그룹핑·tie-break. **함수 앞부분(윈도우 슬라이싱 `_select_query_present_frame_slice`,
  raw→compact 매핑)은 전혀 안 건드림** — 이 부분을 sparse GT 쪽으로 다시 짜는 건 프레임 선택 로직을
  복제해야 해서 위험 대비 이득이 작다고 판단, 대신 기존 로직 앞뒤는 그대로 두고 loop 구간만 교체.
  - `_build_intersection_instance_ids_from_dense_pair`/`_build_history_all_valid_instance_ids`(GT
    metadata 캐시 후보였던 나머지 2개)는 이전 NOTES(2026-07-13 "GT metadata 캐시... 이식 안 함")에
    적힌 이유(하나는 죽은 코드, 하나는 원래 저렴)로 손 안 댐.
  - 호출부는 `efficientocf.py`의 `forward_train`/`_eval_train_faithful_inst_match` 2곳뿐이고, 이
    함수가 다루는 데이터는 inst3d(D, `gt_occ_inst` 기반) 전용 — bbox/aabb(E)·rot(F)는 이 함수와
    무관한 별도 코드 경로(dice loss, center 계산)라 이번 변경과 무관.
- **검증**: 수정 전 전체 함수(윈도우 슬라이싱+매핑 포함, 원본 그대로 보존)와 새 구현을 200회
  랜덤 비교 — `query_present_only` True/False, K=0~12, 프레임 크기 2종, 일부러 class를 섞은 tie
  케이스(8% 확률), raw→compact 매핑 有/無, `gt_instance_ids_n=None`/빈 텐서 등 조합 전부
  bit-identical(200/200). `py_compile` 통과. 편집 후 학습 job(`full_attn_cover.py`, PID 3390668,
  3시간+ 경과) 생존 확인 — `utils_gt_prep.py`는 detector 코드라 dataloader worker가 안 읽어서
  mid-training 크래시 위험과 무관(기존 `_filter_dense_instance_ids` 변경과 동일 근거).
- **속도**: 실제 조건(T=1, occ_size 512×512×40) 기준 — K=5: 274→136ms(2.0x), K=20: 689→135ms(5.1x),
  K=50: 1777→137ms(13.0x), K=70: 2488→131ms(18.9x). 전체 iteration 대비 비중은 이전 항목과 동일하게
  미확인.

## 2026-07-13 KST — [new_data 레포] `_filter_dense_instance_ids` torch.isin 벡터화

- **배경**: 다른 레포(`EOCF_pyramid_multi`)에서 먼저 적용된 최적화(2026-07-09)를 이식. GT metadata
  캐시(같은 레포의 별도 변경)는 이 레포엔 이식하지 않기로 함 — 캐싱 대상 3개 함수 중 1개는 이
  레포에서 이미 죽은 코드(`_build_intersection_instance_ids_from_dense_pair`, 구 `segmentation_instance3d`
  삭제로 secondary input이 항상 None), 1개는 원래 저렴, 나머지 1개(`_prepare_gt_instance_classes_for_matching`)는
  raw→compact class 매핑을 캐시 경계 안에서 수행해 model config(`query_raw_to_compact_class_map`)가
  바뀌면 캐시가 이를 감지 못 하고 조용히 stale한 compact class를 반환하는 위험이 있어 보류(raw/compact
  분리 재설계 필요 — 필요 시 별도 작업).
- **`projects/occ_plugin/occupancy/detectors/utils_gt_prep.py`**: `_filter_dense_instance_ids`
  (history-valid 필터가 dense grid에서 non-kept instance를 0으로 지우는 함수, 4D/5D 두 분기)의
  `for instance_id in keep_instance_ids...: keep |= inst == id` python loop(인스턴스 수 K에 비례해
  느려짐)를 `torch.isin(inst, needle)` 1회 호출로 교체. `keep_instance_ids`를 `dense_gt.device`로
  옮겨서 device mismatch도 같이 방지.
  - 호출부는 `efficientocf.py`의 `forward_train`(학습)과 `_eval_train_faithful_inst_match`(eval,
    line~1571) 2곳뿐이고, 이 함수가 필터링하는 `gt_instance_occ3d_txyz_primary`/
    `gt_segmentation_cls_instance3d_for_match`/`gt_bbox_aabb_inst_txyz`는 학습·eval 양쪽에서 이후
    시각화(`_select_query_visualization_gt_slice` 등, query_debug_vis/mixture3d 등)에도 그대로
    재사용되므로 이 함수 하나만 bit-identical하게 고치면 학습/추론/시각화 전부 자동으로 얼라인됨
    (호출부 수정 불필요).
  - `utils_gt_prep.py`는 `datasets/pipelines/`가 아니라 detector 쪽 코드라 dataloader worker가
    안 읽음 — mid-training 크래시 위험(NOTES 07-07 사례)과 무관, 지금 도는 `full_attn_cover.py`
    학습 중에도 안전하게 적용.
- **검증**: 실제 파일에서 import한 새 함수 vs 수정 전 loop 구현(그대로 보존한 참조 구현)을
  4D/5D × K=0/1/5/30/100 = 10개 케이스 전부 `torch.equal` bit-identical 확인, `dense_gt`/
  `keep_instance_ids` 둘 다 non-tensor(None) 케이스 passthrough 확인, `py_compile` 통과. 편집 후
  학습 job(`full_attn_cover.py`, PID 3390668) 생존 확인.
- **속도**: 이 레포 실측 occ_size(512×512×40, T=7) 기준 loop vs isin 벤치마크 — K=5: 217ms→111ms(2.0x),
  K=20: 487ms→120ms(4.1x), K=50: 1128ms→105ms(10.7x), K=70: 1476ms→111ms(13.3x). isin은 K 무관 고정,
  loop는 K에 선형 비례. 전체 iteration 대비 실제 비중(%)은 미확인(로그 미확보) — 필요 시 프로파일링 권장.

## 2026-07-13 KST — [new_data 레포] `_rot` 학습 배선 + eval bbox GT를 f3로 재배선

- **배경**: `data/efficientocf_gt_f3`(symlink → `/home/hwanhee/datasets/efficientocf_gt_f3`) 실사용
  검증(경로/키/id-space/train-val 분리/완전성 5개 항목 전부 실측 재확인 완료) 후, `GMO/segmentation_rot`
  (29,049개, 회전 OBB GT, train/val 풀셋)이 로더에 전혀 안 붙어 있던 것을 발견하여 배선.
- **`projects/occ_plugin/datasets/pipelines/loading_instance.py`**:
  - `LoadInstanceWithFlow.__init__`에 `gt_bbox_aabb_subdir='segmentation_aabb'` 파라미터 추가.
    `resolve_gt_bbox_aabb_dir()`의 하드코딩 3곳을 `self.gt_bbox_aabb_subdir`로 교체 — 기본값이라
    기존 모든 config는 동작 무변화(회귀 없음, `python -c`로 default/override 양쪽 실제 경로 resolve
    직접 확인함).
  - `__call__` 상단 self-heal 블록(2026-07-07 mid-training AttributeError 사망 방지 패턴,
    `load_gt_bbox_aabb` 옆)에 `gt_bbox_aabb_subdir` 기본값 self-heal 추가 — **학습 도는 중에도
    안전하게 배포 가능**(현재 `full_attn_cover.py` 8GPU 학습이 돌고 있는 상태에서 적용, 이 self-heal
    없으면 다음 epoch worker respawn 시 크래시).
  - `gt_bbox_aabb_subdir='segmentation_rot'` + `gt_bbox_aabb_key='segmentation_rot_saved_list2'`로
    override하면 기존 `results['gt_bbox_aabb']`/dice bbox loss(0.9)/center·attn GT 배선을 그대로
    재사용하면서 소스만 AABB→ROT로 교체됨(새 배선 추가 아님, 소스 교체).
- **`projects/configs/baselines/subset_attn_cover_pyr_aabb_dice3d_rot.py`**: train+test 양쪽
  `LoadInstanceWithFlow`에 위 `gt_bbox_aabb_subdir`/`gt_bbox_aabb_key`를 `segmentation_rot`으로 설정
  — `subset_attn_cover_pyr_aabb_dice3d.py` 대비 단일변수(gt_bbox_aabb 소스만 AABB→ROT). val도
  train과 같은 rot 기준으로 통일(안 그러면 train 감독 소스와 val `gt_instance_centers_world` 기준이
  어긋남 — AABB는 겹치는 인스턴스 간 voxel dedup으로 centroid가 rot보다 더 밀릴 수 있음).
- **`projects/occ_plugin/occupancy/detectors/efficientocf.py`**: `_eval_load_bbox_gt_v2`의
  `EOCF_BBOX_GT_V2_DIR` 기본값을 구 캐시 `./data/efficientocf_bboxcls_v2/GMO` → `./data/efficientocf_gt_f3/GMO`로
  교체. 이 함수는 원래도 `{"aabb":..., "rot":...}`를 분리해서 반환해 `iou_3d_bbox`/`iou_3d_bbox_rot`을
  각각 따로 계산하고 있었음 — 로직 변경 없이 경로만 바꿔서 새 데이터의 aabb/rot 지표가 각각 맞는
  소스로 계산되게 함(변경 전엔 `eval_total.sh`/`eval_total_2.sh` 둘 다 override 안 해서 구 캐시를
  그대로 쓰고 있었음 — 신 데이터 미반영 상태였음). 이 파일은 모델 코드라 dataloader worker가
  안 씀 → 학습 중에도 무관하게 안전.
- **검증**: 3개 파일 `py_compile` 통과, `_rot` config `mmcv.Config.fromfile` 로드 후 필드값 확인,
  `LoadInstanceWithFlow.resolve_gt_bbox_aabb_dir()` 실제 인스턴스화해서 rot(29,049파일)/기본 aabb
  (backward-compat) 양쪽 실제 디스크 경로 resolve 확인.

## 2026-07-13 KST — [new_data 레포] 전체 config 새 GT(f3) 재배선 + 구 캐시 로딩 코드 완전 삭제

- **목적**: `newgt_f3_aabb_dice.py`에서 검증한 새 GT 배선 패턴을 이 레포의 나머지 실사용 config
  4개(`subset_attn.py`, `subset_attn_cover_aabb_dice.py`, `subset_attn_cover_pyr_aabb_dice3d.py`,
  `full_attn_cover_pyr_aabb_dice3d.py`)에도 동일 적용 + `loading_instance.py`의 구 캐시
  (`segmentation_instance3d`/`segmentation_cls_instance3d`) 로딩 코드 자체를 삭제 — 이제 이 레포의
  어떤 config도 구 캐시 분기를 타지 않으므로, 죽은 코드로 남겨두지 않고 완전히 제거.
- **config 5개 공통 변경** (`newgt_f3_aabb_dice.py` 포함, model_cfg는 전부 무수정):
  - `gt_occ_inst_dataset_path`/`gt_bbox_aabb_dataset_path` → `./data/efficientocf_gt_f3/` 단일 루트.
  - `load_segmentation_instance3d`/`load_segmentation_cls_instance3d`/`segmentation_cls_dataset_path`/
    `validate_segmentation_cls_instance3d_alignment` kwarg 전부 제거(생성자에서 완전히 없어진 인자라
    남겨두면 `TypeError: unexpected keyword argument`).
  - train+test 양쪽 `load_gt_bbox_aabb=True`(+ `gt_bbox_aabb_dataset_path`) — eval 쪽도 E 기반 center GT.
  - Collect3D keys에서 `segmentation_instance3d`/`segmentation_cls_instance3d` 제거. `gt_bbox_aabb`는
    train Collect3D에만 유지(raw bbox 텐서는 loss 혼합에만 필요, eval은 파생된 center만 사용 —
    `newgt_f3_aabb_dice.py`의 기존 검증 패턴과 동일하게 유지, test에는 추가하지 않음).
  - `subset_attn.py`는 원래 `gt_bbox_aabb` 관련 배선이 전혀 없어 신규 추가(center GT 소스 확보 목적).
- **`loading_instance.py`**: `LoadInstanceWithFlow`에서 구 캐시 전용 코드 전부 삭제 — 생성자 인자 7개,
  `resolve_segmentation_instance3d_dir`/`resolve_segmentation_cls_dir`/
  `_validate_segmentation_cls_instance3d_alignment` 메서드, 인라인 헬퍼 4개
  (`sparse_instance3d_to_dense`/`sparse_segmentation3d_to_dense`/`load_segmentation_cls_sparse_list_or_raise`/
  `build_segmentation_cls_tensor_from_sparse_list`), `__call__` 내 관련 분기 전체(캐시·재생성 경로 양쪽).
  `elif self.load_gt_bbox_aabb:` 로 있던 center/attn GT 생성 분기는 `if`로 승격(이제 유일 경로).
  `efficientocf.py`의 `segmentation_instance3d=None` 인자 처리는 범용 코드라 무수정(이미
  `newgt_f3_aabb_dice.py`에서 매 배치 None으로 검증된 경로).
- **검증**: 5개 config 전부 `mmcv.Config.fromfile`+`apply_model_cfg`+`LoadInstanceWithFlow` 생성자
  통과(구 kwarg 잔존 시 즉시 TypeError로 걸림). **실 데이터 dataloader 스모크**(`build_dataset` →
  `ds[0]`, train+test 양쪽): `full_attn_cover_pyr_aabb_dice3d.py`(len=23930, `gt_instance_centers_valid`
  249/252, centers 전부 finite) / `subset_attn.py`(신규 배선분 포함) 둘 다 결과 keys에 구 캐시 키
  없음·`gt_bbox_aabb`/`gt_instance_centers_world` 정상 확인. `ast.parse`+`py_compile`+전체 repo
  구-심볼 grep(0건, 주석 제외)으로 이중 확인.
- **`eval_total_2.sh`**: 삭제된 `full_attn_cover.py` 참조 → `full_attn_cover_pyr_aabb_dice3d.py`로 교체.
- **범위 밖으로 남긴 것 (발견만, 미수정)**: `eval_total.sh`가 여전히 삭제된 `full.py`를 참조(대응하는
  후속 config 없음 — 임의로 특정 config에 매핑하지 않음). `train_total_{2,4,5}.sh`,
  `eval_{oracle,sweep_thr,sweep_indep}.sh`, `eval_total_v1.sh`, `tools/dbg_probe/*.py`,
  `tools/gen_data/gen_bbox_gt_v2.py` 등에도 이번에 삭제된 8개 구버전 config명이 남아있음 — 이번
  작업 범위(config 5개 + loader) 밖이라 손대지 않음, 필요 시 별도 확인 요망.

## 2026-07-12 KST — [new_data 레포] f3 데이터 + 로더 사람필터 배선 완료 (학습 준비 상태)

- **config 신설**: `projects/configs/baselines/newgt_f3_aabb_dice.py` — gt 경로 ./data/efficientocf_gt_f3/
  (생성소멸 필터판: 관측창 0~2 교집합, 사람 포함) + `exclude_occ_class_ids=(7,)` (사람제거는 로더에서, CW 포함).
  raw(efficientocf_gt)와 완벽 호환 — 경로만 바꾸면 스왑, 사람 필터는 튜플 토글.
- **얼라인 검증**: ①로더 필터 E ped 12,158→0 / D ped 19,024→0, ②center GT(§8.3, 필터된 E 소스)에 사람 id 0,
  ③전체 파이프라인 3윈도우 스모크 PASS(ped 0, centers 정상). eval은 기존 내장 (gt!=7) 마스크 + EOCF_BBOX_GT_V2_DIR
  =./data/efficientocf_gt_f3/GMO. 시각 증빙: ksh_0630/data_vis/align_f3_wiring.png.
- **데이터 확정**: raw 125GB + f3 60GB(29,049키), f3nh/f7all/f3all 삭제. 디스크 여유 181GB.
- **다음**: GPU 확보 시 1-iter 학습 스모크 → 본 학습.


## 2026-07-11 KST — [new_data 레포] 새 GT(efficientocf_gt) 배선 완료 + center +0.1m 편향 수정 ①②

- **레포 성격**: ksh_0630을 rsync 복제한 새 GT 전환용 워킹 카피. 이 레포에서만 아래 수정 적용(원 레포는 무변경).
- **config 신설**: `projects/configs/baselines/newgt_aabb_dice.py` (base: subset_attn_cover_aabb_dice.py)
  - gt_occ_inst(D)/gt_bbox_aabb(E) 경로 → `./data/efficientocf_gt/` 단일 루트
  - `load_segmentation_instance3d=False`, `load_segmentation_cls_instance3d=False` — 구 캐시 완전 배제
  - `exclude_occ_class_ids=()` — raw GT 그대로(사람제거/생성소멸 필터는 추후 데이터 단계에서)
  - test pipeline에도 `load_gt_bbox_aabb=True`(E 기반 center GT), Collect keys에서 seg3d/cls3d 제거
- **loading_instance.py**: §8.3 배선 — `load_segmentation_instance3d=False`일 때
  `build_instance_center_world_targets(gt_bbox_aabb_sparse_list)`로 center/attn GT 생성(elif 분기,
  캐시 경로/재생성 경로 2곳). center fix ①: `:659` 역변환에 `-res/2` 추가 (NOTES ksh_0630 2026-07-11 예약 문구).
- **utils_gt_prep.py**: center fix ②: gt_occ_inst 폴백 center 빌더 `start = pc[:3] + 0.5*voxel` → `pc[:3]`.
  ①② 동시 적용 완료. cls GT는 기존 코드의 gt_occ_inst 번들 파생(seg_cls_inst 우선) 경로가 자동 담당 — 수정 불필요.
- **검증**: dataloader 스모크 PASS — ds[0] 전체 파이프라인 통과, D 7프레임 로드, centers (7,N,3)가 E에서 생성되고
  값이 올바른 공식(pc_min + v*res)과 일치(match=True), seg3d/cls3d 키 부재 확인.
- **주의(미해결)**: `segmentation`/`segmentation_bev`는 여전히 구 캐시(구 필터 기준) — raw GT와 내용 불일치 가능,
  해당 loss GT 소스 검토 후 유지/E-파생 재생성 결정 필요 (ksh_0630 NOTES 배선 체크리스트 (a)).


## 2026-07-10 KST — Query Cross-Attention Coarse-to-Fine KV 피라미드 이식 + `subset_attn_cover_pyr_aabb_dice3d.py` 신설

- **배경**: 사용자가 `Autonomous_Driving_26_ksh_local` 레포의 `subset_attn_cover_pyr.py`(2026-07-07,
  자매 레포 jhh_gi에서 이식된 coarse-to-fine KV 해상도 피라미드)를 이 레포(`ksh_0630`)에도 적용 요청.
  확인 결과 이 레포에는 `query_transformer_num_layers`만 있고 `kv_resolutions` 관련 코드 자체가
  없어(`_merge_cfg`가 unknown key를 조용히 무시하므로 config만 적어선 죽은 설정이 됨) 코드 이식이 필요.
  ksh_local과 diff 대조 결과 두 레포의 `transformer.py`는 이 기능 외엔 100% 동일 — 이식은 순수 추가.
- **`image2bev/transformer.py`**: ksh_local 버전 그대로 이식.
  - `TransformerModule.__init__`에 `kv_resolutions=None` 인자 추가 + `_normalize_kv_resolutions`
    (길이==num_layers 검증, 각 (h,w)>0 검증).
  - `_build_layer_kv_tokens(x, t, ncam, c, h, w)` 신설: `kv_resolutions=None`이면 기존과 동일하게
    전 레이어가 원본 해상도 K/V 공유(하위호환). 지정 시 레이어별 target(h,w)로 `avg_pool2d`
    (정수배)/`adaptive_avg_pool2d`(비정수배) 풀링해 레이어별 K/V 토큰 리스트 생성.
  - `forward()`: 매 timestep 루프 밖에서 `layer_kv_tokens`를 한 번만 계산(T·Ncam 벡터화), CA의
    `kv`를 `layer_kv_tokens[layer_idx][t]`로 교체(기존엔 timestep당 1회, 전 레이어 공유 `kv`).
- **`efficientocf_config.py`**: `MODEL_CFG_DEFAULTS`에 `query_transformer_kv_resolutions: None` 추가
  (하위호환 기본값). `apply_model_cfg`에서 `None`이면 그대로, 아니면 tuple 정규화 +
  `len() != query_transformer_num_layers`면 `ValueError`.
- **`efficientocf.py`**: `TransformerModule` 생성 시 `kv_resolutions=self.query_transformer_kv_resolutions` 전달.
- **새 config `subset_attn_cover_pyr_aabb_dice3d.py`**(`subset_attn_cover_aabb_dice3d.py` 기반):
  `query_transformer_num_layers=3`, `query_transformer_kv_resolutions=((14,25),(28,50),(56,100))`.
  마지막 해상도(56,100)는 이 레포 `data_config['input_size']=(896,1600)` + img_neck(SECONDFPN,
  stride16 통합) 기준 실제 원본 context feature 해상도와 일치(ksh_local과 img_neck/input_size 동일 확인).
- **검증**: `mmcv.Config.fromfile`로 신규 config 정상 로드, `apply_model_cfg` 통과. `TransformerModule`
  독립 스모크(feat_dim=embed_dim=96, num_layers=3, kv_resolutions 지정)로 forward/backward 성공(query
  grad 흐름 확인), `kv_resolutions=None` 레거시 경로 shape 동일, 길이 불일치 시 `ValueError` 발생 확인.
  `ast.parse`로 3개 수정 파일 문법 확인.
- **주의(단일변수 원칙 이탈)**: `subset_attn_cover`의 σ-matching + `_aabb_dice3d`(dice 3D) 위에 피라미드까지
  얹은 것이라 세 축이 동시에 바뀜. `load_from`/`resume_from` 전부 `None`(from-scratch 전제) — 기존
  1-layer 체크포인트로는 `ca_layers.1.*`/`ca_layers.2.*` 등 state_dict 키 불일치로 resume 불가.

## 2026-07-10 KST — `subset_attn_cover_aabb_dice3d.py` 신설 (dice 3D 모드)

- **목적**: `subset_attn_cover_aabb_dice.py`(2D z-collapse dice) 대비 단일변수 = `query_gmo_dice_3d=True`.
- **코드 변경 없음**: `utils_loss.py::_compute_matched_pair_gmo_losses`에 `query_gmo_dice_3d` 분기(z-collapse 여부)와
  `efficientocf_config.py`의 기본값(`False`)이 이미 구현돼 있어 config에서 플래그만 켜면 됨.
  bbox dice 혼합(`query_gmo_dice_inst3d_weight=0.1`/`query_gmo_dice_bbox_weight=0.9`)은 `dice_3d`와
  독립적으로 짜여 있어 3D에서도 동일하게 적용됨. `_compute_foreground_tversky_pair_loss`는
  shape-agnostic(`.sum()` 전체 reduce)이라 BEV든 3D든 그대로 동작.
- **GT 확인**: `gt_bbox_aabb`(v3 캐시, `[x,y,z,cls,inst]`)는 실제 z-extent를 가진 3D AABB — BEV 투영이 아니므로
  3D dice에 bbox GT를 섞는 것도 유효함(NOTES.md 참고).
- **주의**: `query_matched_gmo_bce_occ_size=(128,128,20)`이라 3D dice는 2D(BEV 1슬라이스) 대비 z=20배 voxel을
  reduce — 메모리/속도 영향은 미실측.

## 2026-07-08 KST — `segmentation_instance3d` 로더 버그 수정: `exclude_occ_class_ids` 오적용 제거

- **버그**: `loading_instance.py:1473`(구)가 `segmentation_instance3d` 캐시(행 포맷 `[x,y,z,...,inst]`, 마지막 컬럼=instance_id)에 `_filter_sparse_rows_by_class(rows, class_col=-1)`을 적용 — `exclude_occ_class_ids=(7,)`가 "class==7(사람) 제거"가 아니라 **"instance_id==7인 물체를 클래스 무관하게 매 샘플 통째로 드랍"**으로 오동작(NOTES.md 2026-07-06/2026-07-08). center GT·attn 카메라마스크 GT가 이 캐시에서 나오므로, id=7 물체는 매칭·center·traj·attn loss에서 조용히 빠지고 있었음.
- **실측(수정 전, raw 캐시 150키×전프레임=1,050프레임 직접 로드)**: instance_id==7이 649~727/1,050프레임(62~69%)에 실존 — 드문 edge case가 아니라 시퀀스 절반 이상에서 상시 발생.
- **수정**: 해당 필터 호출 제거(줄 삭제, 주석으로 이유 남김). 사람 제거는 `gt_occ_inst`(class_col=3, 정상)와의 id 교집합(`_build_intersection_instance_ids_from_dense_pair`, `efficientocf.py` forward_train)에서 이미 상시 보장되므로 안전 — `gt_occ_inst`엔 애초에 사람 voxel이 없어 교집합에서 자동 제외됨.
- **부수효과**: `query_require_history_all_valid=True`(과거 3프레임 전체-valid 교집합 필터, `utils_gt_prep.py::_build_history_all_valid_instance_ids`)가 이제 완전한 instance 목록을 받게 됨 — 이전엔 id=7 물체가 필터 입력에 도달하기 전에 이미 빠져 있었음.
- **적용 시점 주의**: 수정 시점에 `full_attn_cover`/`subset_attn_cover_aabb_dice`(epoch 11/15)/`subset_attn_cover_pyr` 3개 학습 job이 실행 중이었음 — `loading_instance.py`는 공유 dataloader 코드라 다음 epoch 워커 respawn 시 구인스턴스/신코드 불일치로 죽을 수 있음(2026-07-07 사고와 동일 메커니즘). 사용자 확인 하에 즉시 적용(체크포인트 resume으로 복구 가능 판단). 크래시 발생 시 `latest.pth`에서 resume하면 이 수정이 반영된 상태로 이어서 학습됨.
- **검증**: `ast.parse`로 문법 확인만 완료, 학습 재검증(smoke run)은 아직 안 함.

## 2026-07-07 KST — _aabb ep9 A/B 판정 + 후속 실험 `subset_attn_cover_aabb_dice.py` (dice GT 혼합)

- **_aabb ep9 512샘플 A/B 결과 (base=subset_attn_cover ep9, 동일 조건 8GPU eval)**: Recall3d 0.1862→0.1589(−15%), IoU3d(nusocc) −11%, IoU2d(bbox_aabb) −1.2%. 성분: TP +14%·FN −4%(벌리기는 성공)였으나 **비관용FP(box 밖) +30%** — focal이 FP에 관대해 box 밖에서 멈추는 힘이 없음. eval 로그: work_dirs/subset_attn_cover{,_aabb}/eval/20260707_1601*.
- **후속 구현 — dice(tversky) GT 혼합 (`utils_loss.py`)**: `_compute_matched_pair_gmo_losses`에 `dice_inst3d_weight/dice_bbox_weight` 추가 — pair별 `dice = inst3d_w×tversky(inst3d) + bbox_w×tversky(bbox)` (mixture·legacy 양 경로, focal과 동일한 빈-bbox 프레임 가드, bbox GT lowres는 focal용과 공유 1회 빌드). FP-heavy tversky(α=0.7)+bbox GT = "box 안 확장 허용, box 밖 강벌".
- **config**: `efficientocf_config.py`에 `query_gmo_dice_inst3d_weight`(1.0)/`query_gmo_dice_bbox_weight`(0.0) 신설. 신규 `subset_attn_cover_aabb_dice.py` = focal 복귀(1.0/0.0) + dice 0.1/0.9, dice 2D(z-collapse) 유지(사용자 지정). detector(`efficientocf.py`)의 bbox dense 준비 조건을 focal∨dice bbox weight>0으로 확장.
- **dbg**: `dbg_gmo_dice_inst3d/bbox`, `dbg_gmo_dice_bbox_pair_count` 추가 + 화이트리스트 `dbg_gmo_dice_` 허용(기존 dice pair_count/alpha/beta도 tensorboard로 나가게 됨).
- 검증: py_compile + config 로드·가중치·pipeline 키 assert 통과. **datasets/pipelines 무수정** — 돌고 있는 학습 안전 (NOTES 2026-07-07 사고 원칙 준수). 새 run 시작 시 dbg_gmo_dice_bbox_pair_count>0으로 혼합 작동 즉시 확인 가능.

## 2026-07-06 KST — eval 실험 스위치 2종: NMS off + 렌더 가중치 독립화 (`eval_sweep_indep.sh`)

- **`utils_visualization.py`**: `EOCF_EVAL_NMS_RADIUS` env — distance NMS 반경 override (0=off). 미설정 시 config 값(3.0m).
- **`efficientocf.py` simple_test**: `EOCF_EVAL_WEIGHT_MODE=ones` env — 렌더 가중치를 score 대신 1.0 고정. blob 높이의 score 종속(occ thr가 fg thr를 흡수하던 구조, NOTES 스윕 결론)을 끊어 fg(선택)/occ(footprint 반경 σ√(2ln(1/t))) 완전 독립화. 기본값 score = 기존 동작 무변경.
- **`eval_sweep_indep.sh` 신설**: A) NMS=0 (score 가중치, fg.75/occ.75) + B1~3) ones 가중치 occ {0.3,0.5,0.75} — 각 512샘플 순차, `SWEEP2_WAIT=1`이면 타 eval 대기(기본 병행).

## 2026-07-06 KST — focal3d GT 혼합(0.1 inst3d + 0.9 AABB bbox) 구현 + train AABB v3 캐시 생성

- **목적**: `loss_gmo_focal`(matched-pair focal, aka focal3d)의 GT를 pair별로 `0.1×gt_occ_inst3d + 0.9×AABB bbox`로 가중합 (`subset_attn_cover_aabb.py` 실험). dice/매칭/center는 inst3d 유지.
- **핵심 발견 — id 공간**: gt_occ_inst3d·segmentation_instance3d(구 캐시)는 **ped 포함 + `classes=['vehicle','human']`(원본 EfficientOCF) 번호 체계**. ① v2 val 캐시(ped 제외 번호)는 id 어긋남(실측 cover 0.45) → 학습 loss에 사용 금지. ② 현재 config `class_names`의 'bicycle' substring이 `static_object.bicycle_rack`을 등록해 rack 낀 시퀀스에서 번호가 밀림 — `['vehicle','human']`으로 역공학 재현 확인(probe 3/3키 100%).
- **`tools/gen_data/gen_bbox_gt_v2.py` 확장**: `--split train`(전 시퀀스 `len(dataset.indices)`=23,930 순회 — train subset은 매 run random이라 전체 필수), `--aabb_only`, `--include_ped_ids`(cache-parity: classes=['vehicle','human'] + ped rasterize cls=7).
- **v3 캐시 생성**: `data/efficientocf_bboxcls_v3/GMO/segmentation_aabb/` train 23,930키, [x,y,z,nusocc_cls,inst], ~23G. 4-shard 병렬 ~40분. 전수 무결성(23,930 모두 구조 검사) PASS. 심층 검증: 랜덤 키 id-pair coverage=1.0·dominant-cls 일치 (경계 1~2voxel 예외는 inst3d 캐시의 box-margin 특성 — NOTES 참조).
- **생성 중 잡은 함정 2건 (각각 전량 재생성으로 해결)**: ① 'bicycle' substring이 bicycle_rack을 id 번호에 등록 → `['vehicle','human']` cache-parity로 수정. ② 'construction' substring이 `human.pedestrian.construction_worker`(인부)를 cls5(차량)로 저장 → 로더 ped 필터(7)를 뚫고 학습에 들어갈 뻔 — CLS_MAP에서 pedestrian 매칭 최우선으로 수정. **v2 val 캐시엔 ②가 잔존**(bbox eval GT에 인부 box 소량 혼입, NOTES 참조).
- **로더 (`loading_instance.py`)**: `load_gt_bbox_aabb`/`gt_bbox_aabb_dataset_path`/`gt_bbox_aabb_key` 추가 — gt_occ_inst 경로 미러(ped는 cls col 필터로 제거). AABB는 인접 box끼리 voxel 겹침이 정상이라 duplicate-좌표 검사 없는 전용 validator 사용. `formating.py`에 DC(cpu_only) 번들 추가.
- **detector (`efficientocf.py`, `utils_gt_prep.py`)**: forward_train에 `gt_bbox_aabb` 수용 → `_prepare_gt_bbox_aabb_dense_txyz`(extract/query-class filter/dense 재사용)로 [T,X,Y,Z] dense → history_all_valid 필터·trajectory/vis slice를 inst3d GT와 동일 적용 → loss로 전달.
- **loss (`utils_loss.py`)**: `_prepare_grouped_matched_pair_lowres_occ`의 GT lowres 빌드를 `_build_grouped_pair_lowres_gt`로 분리(회귀 diff=0 검증) — **pred voxelize 1회 재사용**, bbox GT lowres만 추가 빌드. pair별 `focal = inst3d_w×focal(inst3d) + bbox_w×focal(bbox)`. whole-box drop 프레임(bbox GT 비었는데 inst3d GT 있음)은 bbox 항에서 프레임 단위 제외(빈 GT 0-감독 방지). 디버그: `dbg_gmo_focal_inst3d/bbox`, `dbg_gmo_focal_bbox_pair_count`.
- **config**: `efficientocf_config.py`에 `query_gmo_focal_inst3d_weight`(1.0)/`query_gmo_focal_bbox_weight`(0.0) 기본값. `subset_attn_cover_aabb.py`에 0.1/0.9 + 파이프라인 로드(train만; test split은 v3 키 없음·불필요) + Collect3D `gt_bbox_aabb`.
- **검증**: 로더 e2e(실 train 키: 로드·ped제거·id겹침 1.000), lowres GT 리팩토링 회귀(diff=0), 문제 키 8개 재생성 대조. 시각화: `work_dirs/bbox_aabb_v3_vis/*.png` (AABB vs inst3d 오버레이 + box-밖 voxel 판정).

## 2026-07-06 KST — Recall3d 성분(TP/FP/FN/bboxFP) 로깅 추가

- **`efficientocf.py`**: `_iou_recall_3d`가 (iou, recall, comps) 3-tuple 반환 (comps=voxel 수 {tp,fp,fn,bbox_fp}; 호출부 3곳 언팩 수정). simple_test 결과에 `recall_3d_comps` 추가.
- **`apis/test.py`**: 성분 voxel 합 누적(단일/분산) → `[FINAL]` 직후 `[recall3d comps] TP/FP/FN/bboxFP | 비관용FP | micro Recall3d` 줄을 stdout·live log에 기록 (분산은 all_reduce). collect에 `recall_3d_comps` 추가.
- **`efficientocf_dataset.py` evaluate**: `Recall_3d_TP/FP/FN/bboxFP/micro` 키 추가 (macro 평균과 별개로 voxel-pooled micro recall).
- **용도**: Recall3d를 갉아먹는 게 FN(못 채움)인지 비관용FP(box 밖 흘림)인지 정량 분리 — threshold/학습 개입 방향 결정용. 과거 run에는 성분이 저장 안 돼 소급 불가; 다음 eval부터 기록.

## 2026-07-06 KST — 부분 eval 스위치 `EOCF_EVAL_MAX_SAMPLES` 추가 (파라미터 튜닝용 짧은 eval)

- **`apis/test.py`**: 전 rank 합산 N샘플 처리 후 조기 종료(전 rank 동일 iter에서 break라 이후 all_reduce/collect 안전). `[FINAL]` 줄은 실제 처리 수(n_done)로 표기. `eval_total.sh`에 주석 예시 추가.
- **수렴 실측 근거 (3개 완주 run)**: 최종값 ±5% 안착 384~768샘플 / ±2% 1,152~1,536 / ±1% 1,536~4,224. `fixed_val=True(seed 2)`라 순회 순서가 run 간 동일 → 같은 N 부분 eval끼리 비교 유효. 속도 ~2.6샘플/초(8GPU, vis on) → 512샘플 ≈ 11분.

## 2026-07-06 KST — eval query_debug_vis 4행 교체: matched 점 → 'eval 실제 pred occ vs bbox AABB GT'

- **요구**: 4행(Hungarian-matched, eval에선 항상 matched=0으로 빈 행)을 GT=bbox AABB(초록) + 예측=**metric 채점에 쓰는 threshold된 occupancy**(빨강) + 겹침(흰색)으로 교체, 프레임별 IoU 라벨.
- **구현 (정합 보장 방식)**: simple_test가 metric 계산에 쓴 `pred_bev_t`/aligned `bbox_bev_t` 텐서를 `eval_cmp_pack`으로 시각화에 **그대로 전달** — 재계산 없음이라 threshold(EOCF_EVAL_OCC_THR)·FG 선택·align·기준프레임(EOCF_EVAL_MODE)이 정의상 metric과 동일. 이를 위해 `_maybe_save_eval_query_vis` 호출을 metric 계산 뒤로 이동.
- **파일**: `efficientocf.py`(호출 이동+pack 배선, `bbox_bev_vis` 캡처), `query_head.py`(`_maybe_save_prob_grid_vis`/`maybe_save_query_debug_vis`에 `eval_cmp_pack` kwarg — pack 있으면 matched_ov를 초록/빨강/흰 합성으로 교체 + 라벨 `pred occ vs bbox_aabb IoU=`). pack 없으면(학습 debug vis) 기존 matched 행 그대로 — 학습 시각화 무영향.
- **E2E 검증**: 1샘플 오프라인 forward(EOCF_EVAL_VIS=1)로 새 4행 PNG 생성 확인, 프레임별 IoU(0.135/0.174/0.174/0.174)가 사전 목업(metric 텐서 직접 캡처본)과 완전 일치. 목업: work_dirs/gt_align_vis/mock_row4_pred_occ_vs_aabb.png.
- **주의**: 진행 중이던 14:59 eval은 교체 전 코드 — 다음 eval부터 새 4행 저장.

## 2026-07-06 KST — eval 마감 로그 [FINAL] 추가 + E2E 검증 완료 + GT 4종 비교 시각화

- **`apis/test.py`**: 루프 종료 후 최종 누적값을 running과 같은 형식 + `[FINAL]` 접미로 stdout·`eval_metrics_live.log`에 기록 (single/multi GPU 모두; dataset 크기가 metric_every 배수가 아니어서 마지막 구간이 안 찍히던 문제). multi는 all_reduce collective라 전 rank 공동 호출 위치에 배치. **다음 eval부터 적용** (14:59 run은 옛 코드라 4992에서 끊김 — 종료 후 evaluate() 테이블로 확인).
- **E2E 확인 (14:59 run 첫 러닝 로그)**: 7지표 전부 정상 — `IoU2d(nusocc)=0.0886 IoU2d(bbox_aabb)=0.1520 IoU2d(bbox_rot)=0.1401 IoU3d(nusocc)=0.0301 IoU3d(bbox_aabb)=0.1137 IoU3d(bbox_rot)=0.1036 Recall3d=0.2632`. bbox_rot NaN 아님 = v2 detector-side 로드 실동작 확정.
- **시각화**: `work_dirs/gt_align_vis/figD_gt_sources_4way.png` — GT 4종(raw gt_occ/inst3d/AABB/rot) 개별+관계 오버레이 8패널 (실측: inst3d ≈ raw∩bbox 99.93%, inst3d 교집합 밖 0voxel — "교집합" 멘탈모델 확정).

## 2026-07-06 KST — bbox GT v2 생성 완료(val 5119) + eval 배선: bbox 메트릭 v2 교체 & bbox_rot 2D/3D 신설

- **생성 완료**: `data/efficientocf_bboxcls_v2/GMO/segmentation_{aabb,rot}/` val 5119샘플 × 2. **val 검증(30키)**: v2 == loader 기준 캐시(diff 0, ped 제외분만 차이) — 반면 **기존 bboxcls는 val에서 box 누락이 13/30**으로 이전 추정(5/39)보다 광범위했음이 드러남(소멸-유지 frozen pose 미반영 추정). 생성 스크립트는 compressed int32 저장으로 전환 + 기존 산출물 재압축(129G→축소, logs/recompress_bboxv2.log).
- **eval 배선 (`efficientocf.py`)**: ① `_eval_sample_key_from_metas`/`_eval_load_bbox_gt_v2` 헬퍼 신설 — test Collect의 `scene_token`(=present `<scene>_<lidar>` 결합키)으로 v2 npz를 detector에서 직접 lazy-load (**config/pipeline 무수정**; 경로 env `EOCF_BBOX_GT_V2_DIR`, 기본 `./data/efficientocf_bboxcls_v2/GMO`). ② bbox 메트릭 GT를 v2 aabb 우선으로 교체(부재 시 기존 bboxcls fallback — 그땐 rot 메트릭 NaN). ③ **IoU2d(bbox_rot)·IoU3d(bbox_rot)** 신설 — AABB와 동일 align 잠금·기준 프레임·수식(관용·마스크 없음). Recall_3d의 bbox 관용항도 자동으로 v2 사용.
- **`apis/test.py`**: `hist_for_iou_bbox_rot`/`iou_3d_bbox_rot` 수집(단일/분산, all_reduce scalars 6→8, CM 3개), running 로그 7지표로 확장. **`efficientocf_dataset.py` evaluate**: `IOU_bbox_rot_*`, `IOU_3d_bbox_rot` 키 추가.
- **검증**: py_compile 3파일 + 헬퍼 단위테스트(키 복원 2형태, dense [7,512,512,40] 로드, 캐시 hit, 부재 키 None). ⚠️ 실 eval E2E는 다음 eval 실행에서 확인 필요 — 로그에 `IoU2d(bbox_rot)=`가 NaN이 아니면 v2 로드 정상.
- **효과**: 이제 7지표 전부(2D/3D × nusocc/bbox_aabb/bbox_rot + Recall) 생성소멸·사람 필터가 학습과 정합. bbox 계열 절대값은 GT 교체로 이전 로그와 비교 불가.

## 2026-07-06 KST — 흔적코드 정리 + bbox GT v2 캐시 생성기(`gen_bbox_gt_v2.py`) [2단계 착수]

- **흔적코드 정리 (오해 유발 legacy)**: ① `query_head.compute_gmo_dice_loss` 삭제 (110줄 — 호출처 없는 옛 dice, gt_occ를 받아 "학습이 gt_occ를 쓴다"는 오해의 원천; 실제 dice는 utils_loss의 matched-pair 버전). ② `_select_query_gt_for_losses` → `_select_query_gt_for_debug_vis` rename (utils_gt_prep + efficientocf 호출부) — 이름과 달리 출력이 디버그 시각화에만 쓰임을 명시. 나머지 legacy(`*_ori.py` 보존본, loading_instance 주석 블록, height_l1 상시 0 배관)는 삭제하지 않고 NOTES에 목록화.
- **`tools/gen_data/gen_bbox_gt_v2.py` 신설**: val(eval) 5119샘플 전용 bbox GT 재생성 — `[x,y,z,cls,inst]` AABB + rotated(OBB) 두 벌을 `data/efficientocf_bboxcls_v2/GMO/segmentation_{aabb,rot}/`에 생성. 필터는 dataset 조립에서 자동 상속(사람: config class_names에 ped 없음 / 생성소멸·저가시성: record_instance가 skip). rasterize는 loader `get_poly_region`/`fill_occupancy` 규칙 재현(whole-box in-range, round 격자화), OBB는 box-local half-extent(+반voxel) 검사. category→nusocc id 매핑에 'vehicle.bicycle'만 bicycle 인정(bicycle_rack 배제). 빈 pipeline dataset으로 이미지 로딩 없이 동작, 존재 파일 skip(재실행/shard 안전).
- **스모크 검증 (3샘플)**: v2 AABB가 기존 bboxcls(ped 제외)와 voxel 완전 일치(-0/+0) / rot ⊂ aabb 위반 0, 부피 0.45~0.71× / inst3d 커버 aabb 100%·rot 97.8~100% / t≥3 신규 id 없음 / cls = GMO만. 전체 생성은 백그라운드 실행 (logs/gen_bbox_gt_v2.log).
- **미배선**: eval 메트릭은 아직 구 bboxcls 사용 — v2 생성 완료 후 로더/simple_test 배선 예정.

## 2026-07-06 KST — IoU3d(nusocc)/Recall_3d GT 교체: gt_occ(raw) → gt_inst(inst3d) [1단계 적용]

- **`efficientocf.py` simple_test**: 3D nusocc 메트릭의 GT를 `_eval_nusocc_3d_fg(gt_occ)` → `(gt_inst>0)`로 교체, `gt_occ_valid`(255 마스크)는 inst3d에 ignore 개념이 없어 None. gt_inst 부재 시에만 기존 gt_occ 경로 fallback. align/관용항/mode_slice 무수정(동일 그리드·방향 — inst3d ⊂ gt_occ 100% 실측).
- **근거 (NOTES 2026-07-06 실측)**: raw gt_occ는 movable voxel의 평균 29%(최악 70%)가 inst3d·bbox 밖 잔여 + 생성소멸 미필터 → 학습(전 loss가 inst3d 기반, gt_occ는 디버그 시각화 전용임을 코드 추적으로 확인)이 배우지 않는 voxel을 구조적 FN으로 채점하고 있었음.
- **영향**: 이제 5개 메트릭 중 nusocc 계열 3개(IoU2d/IoU3d/Recall_3d)가 전부 학습과 동일 GT·동일 필터(사람 제외·query_class·생성소멸). IoU3d·Recall_3d 절대값 상승 예상 — 이전 로그와 비교 금지. gt_occ는 eval에서 fallback 외 미사용.

## 2026-07-06 KST — IoU3d(bbox_aabb) 메트릭 추가 + 기존 IoU3d에 (nusocc) 라벨 명시

- **`efficientocf.py` simple_test**: `iou_3d_bbox` 신설 — 2D bbox 메트릭과 **동일 GT**(`segmentation_cls_instance3d` ch0=bboxcls 캐시, 로드 시 ped(7) 제거 + eval에서 7→0 이중 방어)를 3D 그대로 `_iou_recall_3d`로 채점 (nusocc valid(255) 마스크·bbox 관용항 없음 = IoU2d(bbox_aabb)와 규칙 일치). 3D align은 nusocc 3D와 같은 잠금(`bbox3d_t`) 재사용. `_pack`/return dict에 키 추가.
- **`apis/test.py`**: single/multi GPU 수집·running 로그에 `iou_3d_bbox` 배선. 로그 라벨 `IoU3d=` → `IoU3d(nusocc)=` + `IoU3d(bbox_aabb)=` 추가 (분산 all_reduce scalars 4→6개).
- **`efficientocf_dataset.py` evaluate**: `IOU_3d` → `IOU_3d_nusocc` rename + `IOU_3d_bbox_aabb` 추가 (save_best 등 키 소비처 없음 확인).
- **필터 체크 결과 (실데이터 스캔)**: bboxcls 캐시 클래스 = {2,3,4,5,6,7,9,10} — GMO뿐, 딴 클래스 없음 ✓ / ped는 로드+eval 이중 제거 ✓ / **생성소멸은 캐시에 미적용** — 20샘플 대조에서 2건이 미래 프레임 신규 box 포함, 1건은 과거 box 누락 (loader 자체 캐시 `data/efficientocf`는 visible_instance_set로 필터됨과 대조). **IoU2d(bbox_aabb)도 원래부터 같은 누수** — bbox 계열 두 메트릭 공통 한계로 기록 (NOTES 참조).

## 2026-07-06 KST — eval IoU2d(nusocc) GT 교체: bbox-AABB(segmentation_bev) → 진짜 nusocc inst3d z-collapse

- **문제**: `simple_test`의 "IoU2d(nusocc)" GT가 `segmentation_bev`였는데, 이는 nusocc가 아니라 **bbox 8코너의 AABB를 rasterize한 캐시**(`loading_instance.py` `get_poly_region`→`fill_occupancy`가 min/max로 채움)의 z-sum — 즉 bbox 메트릭이었음 (실데이터 probe로 확인: `data/efficientocf/GMO/segmentation_bev` sparse [x,y,1]).
- **수정 (`efficientocf.py` simple_test)**: GT를 `gt_inst`(=`_prepare_gt_occ_inst_primary_targets`의 `dense_inst_txyz`, nuScenes-Occupancy inst3d 캐시)로 교체 — `(gt_inst>0).any(z)` → [T,X,Y]. **학습 dice(`loss_gmo_dice`)가 쓰는 GT와 동일 소스·동일 필터**(로드 시 pedestrian cls7 제거 `exclude_occ_class_ids`, detector에서 query_class_ids 필터). gt_inst 부재 시에만 기존 segmentation_bev fallback. align/threshold/CM 로직은 무변경.
- **영향**: IoU2d(nusocc) 절대값이 과거 로그와 비교 불가(GT가 바뀜 — nusocc는 표면 sparse라 bbox-AABB보다 작음 → 수치 하락이 정상). IoU2d(bbox_aabb)/IoU3d/Recall3d는 무변경.
- **데이터 파악 (추가 예정 bbox3d·rotated-bbox 메트릭 사전조사)**: ① `segmentation_cls_instance3d` ch0 = bboxcls 캐시 [x,y,z,cls] — **AABB 채움**이라 bbox-3D(AABB) IoU는 지금 데이터로 즉시 가능. ② **회전(yaw)은 모든 캐시에서 소실** — `get_poly_region`이 회전 코너 8개를 계산하지만 rasterize 시 min/max(AABB)로 뭉갬. rotated bbox 2D/3D는 `instance_dict` annotation(translation/rotation/size)에서 OBB rasterize 필요(생성소멸 필터 `visible_instance_set`·pedestrian 제외 규칙 재사용 가능). ③ `gt_instance_dims`=(w,l) 원본 치수 있으나 yaw 미포함.

## 2026-07-05 KST — size note("크기 쪽지") 구현: aux size head + σ/depth 성분 주입 (`subset_attn_cover_size.py`)

- **배경 (NOTES 2026-07-02~04 실측 체인)**: 크기 정보가 모델 어디에도 없음(query feature R²=0.06, BEV dead code, instance-pool 0.015) + naive σ×depth 실패(시야각 단축·폭 바닥, spearman −0.05) → gaussian head에 크기를 **입력으로 직접 공급**하는 설계. 벌리는 힘(렌더-recall)은 사용자 결정으로 후속 스테이지.
- **`loading_instance.py`**: `build_instance_dims_targets`(annotation 원본 box 치수 (w,l)을 grid instance id에 정렬, 회전 무관 진짜 크기) 신설 → cache/non-cache 양 분기에서 `gt_instance_dims` [N,2] emit + skip-list 2곳 등록. `formating.py`: DC(stack=False) 처리 추가.
- **`query_head.py`**: ① `query_size_aux_head`(D→64→2, softplus로 (w,l) 예측), ② `query_size_note_proj`(5→D, **zero-init** — 시작 시 주입=no-op), ③ `apply_lifted_centers_to_outputs`에 `size_note_attn_sigma_tq2` kwarg — note 5성분 [logσu, logσv, log d, log w_pred, log l_pred]을 **detach 후** center_input에 additive 주입(gaussian head gradient가 attention/depth/aux로 역류 차단), outputs에 `query_size_aux_pred_tq2` 노출. 생성자 `query_size_note_enabled`(기본 False).
- **`efficientocf.py`**: `_compute_query_attn_sigma_note`(질량 최대 캠의 attn 가중 표준편차 [T,Q,2], no-grad) 신설, apply_lifted 호출부 배선(학습·추론 공용 경로), `_last_query_size_aux_pred` 캐시, forward_train에 `gt_instance_dims` param + aux loss 호출 + aggregate 전달.
- **`utils_loss.py`**: `_compute_query_size_aux_loss_from_match` — matched만, GT=gt_instance_dims(0=무효 제외), **크기-비례 가중** clamp(max(w,l)/2m, 1, cap=4) (버스 2.9% 희석 방지), L1. `_aggregate_training_losses`에 pack 전달 → `loss_query_size_aux` + dbg 4종(`l1_mean`/`l1_big`(>6m)/`pair_count`/`big_count`, z-prefill로 DDP 일관).
- **`efficientocf_config.py`**: DEFAULTS 3키(`query_size_note_enabled`=False/`query_size_aux_loss_weight`=0.0/`query_size_aux_big_weight_cap`=4.0, **기본 off=기존 동작**) + apply/검증(aux weight>0인데 note off면 에러).
- **config**: `subset_attn_cover_size.py`(사용자 생성 cover 복사본)에 3키 활성(aux weight 0.5) + Collect에 `gt_instance_sizes`/`gt_instance_dims` + 헤더 주석. cover 대비 단일변수. **head 파라미터 증가로 기존 ckpt 비호환 — from-scratch 전용.**
- **검증**: py_compile 7파일 / 합성: dims 빌더(무효 id→0), aux loss(가중 수식 일치 3.9349, gradient OK), σ 헬퍼(넓은 blob 3.94 vs 좁은 1.00) / 양 config build(cover=off 유지) + note E2E(apply_lifted 경로, proj zero-init, aux_pred>0, mixture 정상). ⚠️ 실 데이터 forward smoke 미실행 — 첫 학습에서 `loss_query_size_aux`·dbg 키 정상 출력 확인 필요.
- **판정 계획**: ① `dbg_query_size_aux_l1_big` 수렴(=feature가 대형 크기를 배우는가 — I/O로 보류된 probe를 학습이 대신), ② ep5 `vis_attn_by_size.py` 버스 BEV(before: work_dirs/subset_attn_cover/dbg_analysis/attn_vis_by_size.png), ③ ep15 eval Recall3d. 부족분 = 후속 렌더-recall 스테이지의 몫.

## 2026-07-03 KST — dbg 프로브 스크립트 7종을 `tools/dbg_probe/`로 영구 보존

- 세션 scratchpad(휘발)에 있던 체크포인트 오프라인 진단 스크립트들을 repo로 복사: `dbg_offset_spread.py`(spread/mixture 캡처), `plot_spread_results.py`, `cap_we_bev.py`(**두 run BEV 나란히 비교 — flag-pole 그림 생성기**), `probe_we_spread.py`(visible-gate 우회 판정), `probe_feat_size.py`(feature→크기 선형 프로브), `probe_bev_size.py`, `probe_attn_geometry.py`(attn 폭/coverage/σ×depth). 사용법·주의(하드코딩 경로 수정, vis 게이트, GPU 확인)는 `tools/dbg_probe/README.md`.
- 이 스크립트들의 실측 결과(ghost/flag-pole loophole, feature 크기 부재, attn 고정폭 등)는 NOTES.md 2026-07-02~03 항목 참조.

## 2026-07-02 KST — attn σ-matching(폭 감독) 추가: `loss_query_attn_sigma` + `subset_attn_cover.py`

- **배경 (ep10 attn 기하 probe, NOTES 2026-07-02 참조)**: attn은 카메라·중심은 정확(질량 80~90% 올바른 캠)하지만 **폭이 크기 무관 고정**(σ_attn 8~14px vs σ_mask 1.7~6.3px, corr(σ×depth, GT크기) spearman=0.004). 원인 = inside-mass가 '마스크 안 총량'만 채점해 2차 모멘트(폭)가 감독 없는 자유변수 → feature 유사도 길이(~8px)의 기본 블롭에 정착. 크기 신호의 유일 생존 원천(attn 기하×depth)을 살리려면 폭에 최초의 감독 필요.
- **`utils_loss.py`**: `_compute_query_attn_bbox_loss` 안에 σ-matching 블록 추가 — matched query마다 GT cam 마스크가 최대인 카메라에서 attn 분포의 가중 표준편차(σ_u,σ_v)를 계산해 마스크의 σ에 **log-ratio 회귀** `(logσ_attn−logσ_mask)²` (scale-free, floor 0.25px로 1px 마스크 σ=0 폭주 방지). **위치(1차 모멘트)는 loss 미포함 → center 경로 무접촉**. 게이트: matched valid + 마스크 ≥min_px + attn 질량 5%+가 해당 캠에 존재 + start_iter warmup. 신규 키 `loss_query_attn_sigma` + dbg(`ratio_u/v`(→1.0 목표), `raw`, `valid_count`) — DDP 일관성 위해 항상 prefill(z).
- **`efficientocf_config.py`**: DEFAULTS 3키 — `query_attn_sigma_match_loss_weight`(**0.0=off 기본**, 기존 run 무영향)/`query_attn_sigma_match_start_iter`/`query_attn_sigma_match_min_mask_px`(4) + apply/검증.
- **config**: `subset_attn_cover.py`(사용자 생성, subset_attn 복사본)에 weight **0.25**(초기 기여 ~0.3 = center_match의 1/5), start_iter **500**(≈1 epoch 후 개입) 활성. subset_attn 대비 단일변수 = σ-matching 3키.
- **검증**: py_compile 3파일 / 합성 테스트(넓은 blob vs 좁은 마스크 → loss 0.59·ratio 2.7, gradient가 matched query attn에만, unmatched grad 0, weight=0→0, start_iter 전→0) / 양 config build+플래그 전파(subset_attn off | cover 0.25) OK. ⚠️ 실 forward smoke 미실행.
- **판정**: `dbg_query_attn_sigma_ratio_u/v` → 1.0 수렴, `center_match`가 subset_attn 대비 ±5% 이내, 3~5 epoch 후 `probe_attn_geometry.py`(scratchpad) spearman 0.004→상승. ratio가 2 epoch 후에도 >2 정체면 weight ×2, center_match >10% 악화면 ÷2.

## 2026-07-02 KST — camera-attn loss 카메라간 픽셀좌표 충돌 fix (`utils_loss.py`, `utils_matcher.py`, `subset_attn.py` 신규)

- **배경 (대화 중 코드 리딩으로 발견, dbg 실측 아님)**: `_compute_query_attn_bbox_loss`(utils_loss.py)와 매칭 cost `_compute_query_attn_soft_iou_cost_qn`/`_compute_query_attn_inside_log_cost_qn`(utils_matcher.py) 전부, 예측 attn(`query_attn_weights_tqnhw` [T,Q,Ncam,H,W])을 `sum(dim=2)`로 카메라 축을 눌러 `[T,Q,H,W]`로, GT(`gt_inst_cam_mask_tnnhw`)는 `.any(dim=2)`로 OR해서 `[T,Ninst,H,W]`로 만든 뒤 비교하고 있었음. 두 값 다 카메라 정체성을 버리고 순수 배열 인덱스 `(h,w)`만으로 비교 — 서로 다른 카메라의 같은 `(h,w)`가 같은 슬롯으로 합쳐짐(원본 attn은 transformer cross-attention이 `S=Ncam*H*W` 전체에 대해 계산한 분포라 원래 카메라별로 고유 슬롯이 있었음, `transformer.py:558`).
- **영향**: 한 객체가 카메라 여러 대에 걸쳐 보이거나(정상 케이스는 무해), 서로 다른 두 객체가 서로 다른 카메라의 우연히 같은 `(h,w)` 배열 인덱스에 투영되면 loss/매칭 cost가 이를 구분 못 하고 섞임. `query_attn_match_metric='inside_log'`(현재 활성 매칭 지표)와 `loss_query_attn_bbox`(`use_query_attn_bbox_loss=True`) 둘 다 영향권.
- **수정**: 카메라 축을 미리 누르지 않고 `(Ncam,H,W)`를 하나의 축(`p = Ncam*H*W`)으로 flatten해서 비교. GT 타겟 빌더(`build_query_attn_bbox_targets`, utils_instance_img_debug.py)는 이미 카메라별로 안 뭉갠 `gt_inst_cam_mask_tnnhw`를 반환값에 포함하고 있었으나 아무도 안 쓰고 있었음 — 그걸 그대로 소비하도록 변경(빌더 자체는 무수정). Shape validation에 `ncam` 일치 체크 추가. `gt_inst_valid_tn`(인스턴스-프레임 유효성, `.any(dim=2)`)은 collision과 무관해 그대로 유지. 디버그 전용 `dbg_query_attn_bbox_empty_gt_mask`(efficientocf.py:3336, OR'd `gt_inst_mask_tnhw` 사용)는 그래디언트 무관 모니터링 값이라 무수정.
- **검증**: py_compile 2파일 OK. 합성 텐서 단위테스트(scratchpad `test_attn_fix.py`) — 카메라만 다르고 `(h,w)`는 동일한 두 GT instance를 만들어 old 방식이면 충돌했을 상황 재현: `_compute_query_attn_inside_log_cost_qn`/`_compute_query_attn_soft_iou_cost_qn`/`_compute_query_attn_bbox_loss` 셋 다 자기 카메라·자기 인스턴스는 cost≈0, 다른 카메라의 동일 `(h,w)` 인스턴스는 cost 크게(≥5) 분리되는 것 확인. 전체 forward/학습 smoke는 미실행.
- **config**: `subset_attn.py` 신규(내용은 `subset.py`와 100% 동일, 헤더에 실험 취지 주석만 추가) — 코드 fix가 config 무관(전역) 적용이라 새 하이퍼파라미터·값 변경 없음. 사용자 결정: GMO shape 계열(dice/tversky/weight/scale-spread) 안 얹고 순수 베이스라인 위 단일변수 ablation으로 검증.

## 2026-07-02 KST — spread 정칙항 weight-loophole 봉쇄: 'visible' soft gate + 'violators' 정규화 (`subset_scale_offset_we.py`)

- **배경 (dbg 실측, NOTES 2026-07-02 참조)**: subset_scale_offset ep1/ep10 forward 실측 결과 offset-only spread가 **크기 무관 ~3.5m 균일**(corr≈0), matched 42쌍 **전부 deficit=0 → 정칙항 무압력**. 원인 = σ-loophole의 weight 버전: 낮은 w '유령' 가우시안(~40/48개)이 정규화 질량의 60%+를 차지해 메트릭을 채우지만 점유(occ 임계 0.75)에는 안 나타남. 실제 가시 가우시안은 5~9개, 반경은 버스도 car-size(2.8m).
- **`query_head.py`**: spread 가중에 **soft visibility gate** 추가 — `query_scale_spread_weight_mode='visible'`이면 `w_eff = sigmoid((peak − vis_thr)/0.1)`, peak = w(sigmoid/union: w=α opacity) 또는 1−exp(−w)(poisson). `1-exp(-w)` 단독은 w≤1에서 근사-선형이라 유령 억제 실패 → 임계 기준 gate 필수(합성 검증: 유령 질량 62%→**4.6%**). `__init__`에 `query_scale_spread_weight_mode`('raw' 기본)/`query_scale_spread_vis_threshold`(0.75) param+검증. tau=0.1은 하드코딩(주석).
- **`utils_loss.py`**: `_compute_query_spread_reg_from_match`에 `norm_mode` param — 'matched'(기존 mean, 기본)/'violators'(deficit>0 쿼리 수로 정규화). visible 전환 후 위반자는 소수 대형뿐이라 matched-mean이면 1/K 희석으로 gradient 사멸(합성: 6쌍 중 1 위반 시 0.067 vs 0.40).
- **`efficientocf.py`**: QueryHead에 `query_scale_spread_weight_mode`/`query_scale_spread_vis_threshold=self.occ_score_threshold` 전달, loss 호출에 `norm_mode` 전달.
- **`efficientocf_config.py`**: DEFAULTS 2키 추가(`query_scale_spread_weight_mode`='raw', `query_scale_spread_norm_mode`='matched' — **기본값=기존 동작**, 진행 중인 run 재개 무영향) + apply/검증.
- **config**: `subset_scale_offset_we.py`(사용자 생성 복사본)에 `weight_mode='visible'`/`norm_mode='violators'` 활성 + 근거 주석. weight 0.2/frac 0.5 유지(단일변수 = loophole 봉쇄만).
- **검증**: py_compile 5파일 OK / 양 config build + 플래그 전파(qh.mode raw|visible, vis_thr 0.75) OK / 합성 수치: 버스형 raw spread 3.75m(충족·무압력)→visible 2.18m(deficit 1.17m 발생), gradient = 가시 가우시안 offset 바깥으로 + 유령 weight 올리기(opacity 레버), 균일 w(초기) 시 gate 균일→unweighted RMS 동일(초기 불안정 없음). ⚠️ 전체 forward smoke 미실행. ⚠️ CLAUDE.md가 언급하는 `EfficientOCF_V1.1_1gpu.py`는 repo에 부재 — efficientocf_config.py만 갱신.

## 2026-07-01 KST — scale head / Lreg / budget 커플링 코드 전면 제거 (forcing-only 확정)

- **배경**: forcing-only로 확정 → scale head의 유일한 소비처(천장 커플링)가 없어 dead code. 사용자 요청으로 완전 제거. (아래 2026-06-30~07-01의 scale head/Lreg/커플링 관련 항목은 이 제거로 **무효화**됨.)
- **제거**:
  - `query_head.py`: scale head 생성·`__init__` params/assignments·`query_inst_scale_q3` 예측·커플링 분기(→ 원래 고정 `off_max`/`sigmoid_to_sigma_from_range` 복원)·`_compute_gaussian_outputs` **6-tuple→5-tuple** 되돌림(두 unpack·양 outputs dict 포함).
  - `efficientocf.py`: QueryHead scale wiring 4줄·`_last_query_inst_scale` 캐시(init+write)·forward_train Lreg 블록·aggregate `query_scale_loss` 전달.
  - `utils_loss.py`: `_compute_query_scale_match_loss_from_match`·`_aggregate_training_losses` `query_scale_loss` param·`loss_query_scale` emit.
  - `efficientocf_config.py`: scale DEFAULTS 5키(`use_query_scale_loss`/`query_scale_loss_weight`/`query_scale_couple_budget`/`query_scale_off_floor_m`/`query_scale_min_m`)+apply.
  - `subset_scale.py`·`subset_scale_offset.py`: `use_query_scale_loss`/`query_scale_couple_budget` 줄.
- **유지(forcing 경로)**: `gt_instance_sizes` 플러밍(voxel-AABB extent, loading/formating/Collect3D), `query_offset_spread_tq3`(query_head apply_lifted), `_gather_matched_gt_sizes`/`_normalize_gt_instance_ids/sizes`/`_compute_query_spread_reg_from_match`(utils_loss), `query_scale_spread_loss_weight(0.2)/target_frac(0.5)`, `loss_query_spread`.
- **검증**: **repo 전체 제거 심볼 잔존 0** / py_compile 6파일 OK / Config.fromfile(subset_scale_offset) — spread 0.2 유지·scale 키 부재(KeyError 없음) / **QueryHead scale head 없이 빌드·`_compute_gaussian_outputs` 5-tuple 복귀·apply_lifted `query_offset_spread` 흐름·spread helper 동작**. ⚠️ 전체 build/forward smoke 미실행.

## 2026-07-01 KST — spread forcing 메트릭을 offset-only로 교체 (σ-loophole 차단) + subset_scale_offset 신규

- **배경**: 직전 forcing(subset_scale 실측, Epoch7)에서 `loss_query_spread`가 max 0.046/mean 0.0036으로 **사실상 미작동**. 원인 = 메트릭 `query_sigma=√(Σw(σ²+offset²))`가 **σ를 포함** → 모델이 offset을 분산 안 하고 **σ만 캡(3m)까지 키워** target 충족(=페널티 0). 로그상 spread가 epoch마다 0으로 수렴한 게 σ-loophole 증거.
- **수정 = 메트릭에서 σ 제외**: `query_head.apply_lifted_centers_to_outputs`에 **offset-only 분산** `query_offset_spread_tq3 = √(Σ w·(mixture_center−query_center)²)` 노출(σ 무관, 가우시안 **중심 분포**만). spread 정칙항 target을 이것으로 → σ를 아무리 키워도 페널티 안 줄어 **offset을 실제로 움직여야만** 감소. gradient가 **offset head로만** 흐르는 것 E2E로 검증(offset grad 539 / sigma grad None).
- **배선**: `efficientocf` 캐시 `self._last_query_sigma_world`→`self._last_query_offset_spread`(query_offset_spread_tq3). `utils_loss._compute_query_spread_reg_from_match` param `query_sigma_world_tq3`→`query_spread_tq3`(메트릭 무관), docstring/변수명 정리. 헬퍼 로직·정규화·one-sided·크기가중 없음(mean)·frac/weight 다 **동일**.
- **config**: **가중치 변경 없음**(weight 0.2, frac 0.5 그대로). `subset_scale_offset.py`(subset_scale 복사본, 별도 work_dir용)를 active로 — 주석만 offset-metric 명시. 코드가 offset-only가 기본이라 subset_scale.py도 자동으로 offset metric(옛 σ metric 경로 제거).
- **⚠️ 남은 watch-item(미변경)**: 희소한 버스가 mean-over-matched로 희석될 수 있음(gradient/K). 이번엔 단일변수(메트릭만) 원칙으로 안 건드림 — offset-metric으로 돌려보고 버스가 여전히 약하면 그때 크기가중/weight↑. center 아직 학습 중이라 초반 흔들림 가능(spread는 크기만 강제, 위치는 center_match).
- **검증**: py_compile(4파일) OK / offset_spread 수식 σ-독립성(collapsed→0, spread→3.07) / 실제 `EfficientOCFLossMixin` 헬퍼 offset-spread 동작 / **QueryHead apply_lifted E2E**: query_offset_spread_tq3 outputs 흐름 + backward가 offset head만(σ head 0). ⚠️ 전체 build/forward smoke 미실행.

## 2026-07-01 KST — forcing 추가: one-sided spread 정칙항 (`loss_query_spread`) — 큰 객체를 GT extent까지 펴기

- **배경**: 직전 budget 커플링은 **천장(ceiling)만** per-query로 정함. 그런데 `offset_max=12`라 천장은 원래부터 버스를 안 막았음 → 커플링만으론 버스가 안 커짐(주효과는 차 조이기). 사용자 확인: 타 레포 실측상 **모든 객체가 차 크기로 collapse**. 원인은 ceiling이 아니라 **collapsed init + 먼 영역에서 약한 FN gradient**로 가우시안이 큰 GT를 못 채우고 차 크기 평형에 갇히는 것. 평형을 옮기려면 **밖으로 미는 항**이 필요.
- **`_compute_query_spread_reg_from_match` (utils_loss.py) 신규**: matched query의 effective 2nd-moment 반경(`query_sigma_world`, 프레임 평균)을 `target_frac·0.5·GT_extent`까지 끌어올리는 **one-sided(relu) 정칙항**. 부족할 때만 밂 → 차/이미 큰 건 0, over-spread는 여전히 Tversky FP가 캡. xy는 `max(σx,σy)` vs `max(GTx,GTy)`(yaw/ego-transpose safe), z 별도. GT target은 detach, gradient는 offset/σ head로만(천장 off_max_q도 detached) → **scale head는 Lreg-only 유지**(충돌 0). query_sigma는 offset²를 포함 → 미는 힘이 σ(캡)보다 **offset을 키워 48개를 객체 전역에 분산** = collapsed init 직접 해소.
- **공유 gather**: `_gather_matched_gt_sizes`로 id-매칭 gather를 Lreg/spread가 공유(중복 제거). 기존 `_compute_query_scale_match_loss_from_match`를 이 gather 사용으로 리팩터(동작 동일, 재검증).
- **배선**: `efficientocf.py` `self._last_query_sigma_world` 캐시 + forward_train에서 `query_scale_spread_loss_weight>0`일 때 호출 → `_aggregate_training_losses(query_spread_loss=)` → `loss_query_spread` emit. (스칼라 placeholder 불필요 — offset/σ head는 occupancy loss로 매 iter grad 받음.)
- **config**: `query_scale_spread_loss_weight`(기본 0.0=off)/`query_scale_spread_target_frac`(0.5) DEFAULTS+apply. 다른 config는 0.0이라 무영향.
- **(최종) subset_scale = forcing-only**: 사용자 요청으로 **천장 커플링·scale head/Lreg를 OFF**(`use_query_scale_loss=False`, `query_scale_couple_budget=False`) → offset/sigma 캡은 **기존 전역 고정값(12 / 3·0.6) 그대로**, **spread 정칙항(0.2)만** 작동. 커플링/Lreg/scale-head 코드는 그대로 두고 flag로만 OFF(나중에 재활성 가능). spread의 GT-extent 타겟은 예측 scale이 아니라 voxel-AABB라 scale head 없이 동작. 검증: Config.fromfile로 게이트(F/F, spread 0.2) + 전역 캡(12/3·0.6) 유지 확인.
- **천장+forcing 짝**: 예측 size→천장이 방을 열고, spread 정칙항이 그 방을 GT extent까지 채움. `frac=0.5`는 균등채움 2nd-moment(≈half/√3≈0.577)보다 보수적이라 달성가능·over-spread 안전.
- **검증**: py_compile(4파일) OK / Config.fromfile(subset_scale) 키 확인 / **실제 `EfficientOCFLossMixin` helper 임포트·호출**(Lreg 스칼라, spread one-sided: 버스>차>0, weight=0→None·empty-match→placeholder 게이트) + 합성 gather 정렬·relu 거동 확인. ⚠️ 전체 build/forward smoke 미실행(데이터/GPU).

## 2026-06-30 KST — per-query instance-scale head + Lreg + budget 커플링 (큰 객체 커버, subset_scale ON)

- **목적**: 버스/트레일러 같은 큰 객체 under-coverage 해결. 지금까지 over-spread를 잡으려고 건 장치(Tversky α=0.7, sigma_max_z=0.6, offset_max 고정)가 **모든 query에 동일**해서, 차에 맞춰 조이면 큰 객체가 그 한도에 막혀 못 퍼짐. → query마다 객체 크기를 예측하고 그 크기로 가우시안 퍼짐 budget을 per-query로 정함.
- **scale head (query_head.py)**: `gaussian_scale_head=GaussianHead(out_dim=3)` 추가(`query_scale_head_enabled`일 때만 생성). present-frame feature에서 `scale_q3 = scale_min + softplus(.)` [Q,3] (world x/y/z extent) 예측. `_compute_gaussian_outputs` 5-tuple→6-tuple(+`query_inst_scale_q3`); 두 호출부·outputs dict 전부 갱신. 33-tuple(`extract_feat_query`)은 **미변경** → 캐시 attr(`self._last_query_inst_scale`)로 loss에 전달해 eval/oracle 위치불변 unpack 안전.
- **Lreg (utils_loss.py)**: `_compute_query_scale_match_loss_from_match` 신규 — 예측 scale과 **matched GT voxel-AABB span**의 L1. `_compute_query_center_match_loss_from_match`(실제 활성 경로)의 sibling로 추가(죽은 `utils_matcher.py:993` 안 건드림). GT size는 `gt_ids_n[matched_inst_idx]`로 **instance id 매칭 gather**(matcher 필터/순서 무관). `scale.sum()*0` placeholder 항상 포함 → zero-match 배치에서도 scale head가 grad 받아 **DDP static_graph 안전**. `_aggregate_training_losses`에 `query_scale_loss` param + `loss_query_scale` emit, forward_train에서 `use_query_scale_loss` gate로 호출.
- **budget 커플링 (query_head.py, `query_scale_couple_budget`)**: `off_max_q = min(clamp(0.5·scale.detach(), off_floor), offset_max_global)`, `sigma_max_q = min(clamp(0.25·scale.detach(), sigma_min), sigma_max_global)`. **scale는 detach** → scale head는 Lreg로만 학습(occupancy loss가 안 끌어당김). xy는 `max(w,l)`로 **yaw-safe**(회전 객체 under-cover 방지), z는 half-height. 전역 offset/sigma_max 안쪽으로 **layer(min)** + off_floor(차 반-크기) 하한 → 기존 config 전부 그대로 로드, 작은 객체는 floor에 머물러 조여짐. 검증: 버스 xy budget 6m / 트레일러 9m / 차 2.25m / 보행자 floor 2.0m, sigma 음수 없음·cap 준수.
- **size 타겟 소스 = voxel-AABB(max-over-frames)** (진짜 nuScenes box 대신): `build_instance_center_world_targets`가 center 계산하며 이미 `pts.min/max`를 뽑으므로 거기서 per-instance extent를 **프레임 최대값**으로 추가(occlusion 과소측정 완화). center와 **같은 id 공간**이라 정렬 무료. nuScenes `size`(loading_instance.py:344)는 cached voxel id↔annotation id 일치가 불확실 + plumbing hop 多라 보류. plumbing: build 4-tuple 반환 → `results['gt_instance_sizes']`(cache/non-cache 양 경로 + skip-list) → `formating.py` DC(stack=False) → `subset_scale.py` Collect3D(train+test).
- **config**: `efficientocf_config.py` DEFAULTS+apply에 `use_query_scale_loss`(F)/`query_scale_loss_weight`(0.3)/`query_scale_couple_budget`(F)/`query_scale_off_floor_m`(2,2,0.4)/`query_scale_min_m`(0.4,0.4,0.2). QueryHead `__init__` 4 param + 생성부 wiring. **`subset_scale.py`는 전부 ON**(loss+couple, weight 0.3). 다른 config는 DEFAULTS OFF라 **무영향**(byte-identical). `EfficientOCF_V1.1_1gpu.py`는 tree에 없어 미러 불필요.
- **검증**: 7개 수정파일 py_compile OK / mmcv Config.fromfile(subset_scale) 로드 + model_cfg 키 ON 확인 + Collect3D gt_instance_sizes(train+test) / 커플링·Lreg-gather 합성 단위테스트 통과(shape·size-ordering·cap·정렬). ⚠️ 진행 중 학습 미반영 — 다음 실행부터. 전체 모델 build/forward smoke는 미실행(데이터/GPU 필요).

## 2026-06-30 KST — oracle viz = normal viz로 통일 + footprint 복구 (두 레포)

- **footprint 복구**: footprint 시각화(그리기)는 occ threshold 통일이 목적이었지 제거 대상이 아니었음(오해). `query_head.py`를 HEAD에서 복구 + rename(occ_score/fg_score) 재적용 → footprint 그림 그대로, 이진화는 `occ_score_threshold`로 통일. 검증: 1-GPU VIS smoke로 png 정상 렌더 확인. (앞서 footprint 메서드 제거 때 고아로 남은 `@staticmethod`가 `_get_query_vis_palette`를 깨뜨렸던 버그도 fix.)
- **oracle viz = normal viz**: oracle은 이제 `sel_idx`(=metric 점유 렌더)만 바꾸고, viz bundle은 override 안 함 → oracle eval의 query_debug_vis가 normal eval(eval_total.sh)과 **완전 동일**. `_apply_oracle_selection_to_bundle`(call + method) 제거. metric은 그대로 oracle(GT Hungarian 매칭) 사용.

## 2026-06-30 KST — footprint inert 잔여 제거 + 고아 데코레이터 버그 fix (두 레포, VIS smoke-test 검증)

- footprint **그리기 제거 후 dead로 남은 잔여**까지 정리: `query_head._maybe_save_prob_grid_vis`의 `_project_gaussians` + gpts/sig/yaw/w 투영 setup + `bundle_has_gaussian`/`gaussian_desc` + 헤더 suffix + dead local(gaussian_vis_mode/prob_threshold/prob_alpha_scale) 7블록 제거. `_apply_oracle_selection_to_bundle`의 selected_sigmas/mixture + 관련 인자 제거(footprint 전용이라 dead). center 마커(pts/`_project_points`)·dense occ·GT viz는 유지.
- **⚠️ 버그 fix (VIS smoke-test가 잡음)**: 앞서 `_draw_gaussian_bev_footprints`(@staticmethod) 메서드를 `def`부터 지우면서 위의 `@staticmethod` 데코레이터가 **고아로 남아 바로 아래 `_get_query_vis_palette`를 잘못 staticmethod로** 만듦 → `self._get_query_vis_palette()`가 "missing self"로 viz 전체가 try/except에 먹혀 skip됐었음. 고아 데코레이터 제거로 해결. (metric엔 영향 없었음 — viz만 깨졌었고 try/except가 삼킴.)
- 검증: 1-GPU VIS=1 eval smoke → png 30장 생성, viz skip/error 0. 두 레포 동일.

## 2026-06-30 KST — threshold rename + fg_score를 model_cfg로 + footprint 그리기 제거 + gs oracle 이식 (두 레포 lockstep)

**모든 변경을 codeonly·gs 두 레포에 동일 적용. 검증: 4개 핵심코드 py_compile OK + sz06 config 로드 두 레포 동일(occ=0.5/fg=0.75/viz_has_fg=False).**

- **Rename (가족 전체)**: `eval_occ_threshold`→`occ_score_threshold`, `debug_query_score_*`→`fg_score_*`(threshold/iou_weight/cls_weight/cam_attn_weight/topk), `debug_query_distance_nms_radius_m`→`fg_score_distance_nms_radius_m`. config 28파일 + 코드(efficientocf_config/utils_visualization/utils_loss/query_head/efficientocf). 이유: `debug_` 접두사가 viz-전용처럼 보이나 실제론 baseline 선택(=metric)을 좌우하는 진짜 임계값.
- **fg_score 가족을 visualization_cfg → model_cfg 이동**: occ_score_threshold 옆에 "Thresholds" 블록으로 모음(자주 바꾸는 값 한곳 관리). selection 로직이 "visualization"에서 빠져나옴. `_merge_cfg`가 lenient라 config 11개+ defaults/apply 모두 일관 이동 필수 — 전 config 로드로 값 유지 검증.
- **footprint 그리기 제거**: `query_head.py`에서 `_draw_gaussian_bev_footprints` 메서드 + 5개 호출 블록(각 363줄) 제거. footprint는 viz-전용(metric/selection 무관, iou_q는 eval에서 0)이라 안전. dense 예측 occ 오버레이·center 마커·GT viz는 유지. **남은 inert**: 그리기 떼낸 뒤 안 쓰이는 gaussian 투영 setup(`_project_gaussians`+g* ~120줄) + bundle mixture 출력 필드 — 두 레포 동일하게 잔존(무해, 후속 정리 대상).
- **gs oracle 이식**: gs `efficientocf.py`에 `_eval_train_faithful_inst_match`+`_apply_oracle_selection_to_bundle` 삽입 + simple_test 배선(attn GT + oracle 블록) + `eval_oracle.sh`. main과 동일 동작.

## 2026-06-30 KST — viz 범례에서 redundant "Gaussian footprint" 회색 스와치 제거 (codeonly + _gs)

- `query_head.py` `_draw_generic_vis_legend`에서 `((140,140,140), "Gaussian footprint")` 범례 항목 + 안 쓰이는 `gaussian_label` 파라미터/호출 인자 제거. 실제 footprint는 행 색(cyan=conf≥thr / red=conf<thr)으로 그려지고 범례에 이미 그 항목이 있어 회색 스와치는 중복·혼동성 라벨이었음.
- footprint **렌더링 자체와 `gaussian_desc`(이미지 헤더 "Gaussian vis: ...")는 유지** — 그림은 안 바뀜.
- 이 viz 렌더러는 **train·eval 공유**라 학습/eval/시각화 범례가 자동으로 동일하게 정렬됨(따로 손댈 것 없음). 선택/metric 로직(`debug_query_score_*`)은 무수정.
- 동일 변경을 `_gs` 레포(`Autonomous_Driving_26_ksh_occ_shape_gs`, 해당 코드 byte-identical)에도 적용.

## 2026-06-30 KST — Oracle 매칭을 학습과 100% 동일하게(train-faithful) 재구현

- **배경**: 직전 oracle은 center 거리(기하) 근사라 학습 cost matrix와 불일치. 학습 매칭은 이 config 기준 **center(10·L1/diag) + cls(0.2) + attn soft-IoU(0.3)**, sim(0)은 feature **게이트**, 그리고 **full7 temporal**(`match_temporal_cost_frame_indices=range(time_receptive_field)`, `match_center_frame_idx=None`). 이를 그대로 재현.
- `efficientocf.py` 신규 `_eval_train_faithful_inst_match(...)`: `forward_train`의 매칭 orchestration(GT prep → intersection/history 필터 → trajectory/vis slice → cls prep → full center pack → ego-align(`_align_geom_pack`)/`_prepend_past_frames` → full7 분기 → `pool_gt_instance_context_features` + id reindex → `_select_matching_feature_frames` → 동일 가중치로 `_match_queries_to_gt_instances`)을 **동일 helper로 복제**. `forward_train`은 무수정(학습 안전). feat_out 인덱스는 학습 언패킹과 동일.
- `simple_test`: `EOCF_EVAL_ORACLE_MATCH=1`이면 학습-동일 매칭으로 선택 override. attn cost(0.3)용 attn target 생성을 위해 `extract_feat_query`에 `gt_segmentation_instance3d_txyz_for_attn`(=primary)·fallback 전달.
- matcher의 `matched_query_idx`를 `sel_idx`로 사용(중복 제거·정렬), viz bundle도 `_apply_oracle_selection_to_bundle`로 일치.
- 런타임 검증: 8GPU eval에서 첫 수십 샘플 크래시 없이 통과(학습-동일 경로 정상 동작).
- **옛 center-거리 기하 근사 oracle 완전 제거**: `_eval_oracle_match_select_queries` 메서드, `simple_test`의 `=center` 분기, env `EOCF_EVAL_ORACLE_DIST`/`EOCF_EVAL_ORACLE_MAX_DIST_M` 삭제 → oracle은 **학습-동일 경로 단일**(혼동 방지). (`_apply_oracle_selection_to_bundle`은 학습-동일 경로도 쓰므로 유지.)
- ⚠️ 한계: `forward_train` 공유가 아니라 **복제**라, 학습 코드가 바뀌면 이 메서드도 같이 갱신 필요(NOTES 참조).

## 2026-06-30 KST — Oracle(GT 치팅) eval 추가: query↔GT center Hungarian 선택으로 스코어링 병목 진단

- **목적**: 기존 eval은 query를 **confidence score**로 선택(`_build_query_visualization_bundle`→`selected_query_idx_q`). 이 선택을 GT instance center에 **Hungarian 매칭**한 결과로 override해 "스코어링이 병목인지" 진단. oracle ≫ baseline → 스코어링 문제 / 비슷 → query shape·trajectory·매칭 문제. (oracle도 선택만 완벽하게 해주는 상한선 — shape는 여전히 예측값으로 렌더되므로 shape 오차는 안 고쳐짐.)
- `efficientocf.py`:
  - 신규 메서드 `_eval_oracle_match_select_queries(...)`: GT instance center(present 프레임=traj timeline idx 0)와 query center를 `torch.cdist`로 cost 만들고 `utils_matcher._solve_unique_assignment`(Hungarian)로 매칭 → 매칭된 query만 `sel_idx`로 반환. matcher 본체(`_match_queries_to_gt_instances`)는 feature-sim base cost라 query/GT feature 없으면 early-return → 재사용 대신 기하 cost만 직접 구성.
  - `simple_test`: score 기반 `sel_idx/selected_score` 산출 직후 `EOCF_EVAL_ORACLE_MATCH=1`이면 oracle 결과로 override. metric만 영향, 학습/loss 무관.
  - 신규 `_apply_oracle_selection_to_bundle(...)`: oracle일 때 viz bundle의 `selected_*`(idx/score/cls/points/mixture)도 oracle 선택으로 덮어, 2D `query_debug_vis`·3D `query_mixture3d_vis`가 oracle-렌더 occ와 일치. (viz는 try/except라 실패해도 metric 안전.)
- 데이터: test pipeline이 이미 `gt_instance_centers_world/valid/ids`를 Collect3D로 로드 → `forward_test(**kwargs)`→`simple_test(**kwargs)`로 전달됨(추가 전처리 불필요).
- 신규 env flag(EOCF_EVAL_* 컨벤션): `EOCF_EVAL_ORACLE_MATCH`(0/1), `EOCF_EVAL_ORACLE_DIST`(xy|xyz), `EOCF_EVAL_ORACLE_MAX_DIST_M`(>0 매칭 거리 게이트). `efficientocf_config.py` env-doc 블록에 주석만 추가(설정값 아님, env 전용이라 config 키 미추가).
- 신규 스크립트 `eval_oracle.sh`: config `shape_guide_128_dice_weight_fl25_sz06`, ckpt `epoch_12_lss_only.pth`(=현재 `latest.pth` 심볼릭 타깃), MODE=1(future)·OCC_THR=0.75(baseline 동일), PORT=20025(충돌 회피), `EOCF_EVAL_ORACLE_MATCH=1`. VIS on(VIS_EVERY=48, baseline 동일)이되 결과는 baseline과 섞이지 않게 `eval_oracle/<timestamp>/` 별도 폴더에 저장.

## 2026-06-29 13:44 KST — dice/Tversky 3D 옵션(`query_gmo_dice_3d`) 추가 + fl25_sz06_dice3d config

- **배경/정정**: 직전 fl25_z(focal 가중 0.1→0.5)는 shape 조임에 부적합으로 판단. focal은 (a) `p.clamp(eps,1-eps)`(utils_loss.py:2634)로 포화 시 gradient 死, (b) FP를 전 격자(≈327k)로 평균(utils_loss.py:2664)해 per-cell FP gradient ≈0 → **over-coverage를 못 벌함**. 반면 Tversky(utils_loss.py:2684-2697)는 clamp 없음 + FP를 foreground 크기로 정규화 → **퍼짐을 실제로 벌하는 항**. 따라서 z 과확장 억제는 focal↑가 아니라 **dice를 3D로**가 맞음.
- `query_gmo_dice_3d` model config 키 추가: 기본값 `False`(= 현행 BEV `amax(dim=2)` 동작 → 기존 모든 config 불변).
- `utils_loss.py` `_compute_matched_pair_gmo_losses`의 dice 계산부 2곳(chunked/single)에서 `amax(dim=2)`(z collapse)를 `query_gmo_dice_3d`로 gate. True면 `[T,1,Z,Y,X]` 그대로 Tversky에 투입(z 과확장도 FP로 벌함). valid_mask 차원도 그에 맞춰 분기.
- `efficientocf_config.py`: 기본값 + self 할당 추가.
- 새 config `shape_guide_128_dice_weight_fl25_sz06_dice3d.py` (fl25 기반, 2변수): `sigma_max_z 1.5→0.6`(force) + `query_gmo_dice_3d=True`(reward: 브레이크를 z축에). focal weight는 기본 0.1 유지.

## 2026-06-29 11:34 KST — query_gmo_loss_weight(focal 항 가중치) config화 + fl25_z 콤보 config 추가

- `query_gmo_loss_weight` model config 키 추가: 기본값 `0.1` (기존 `efficientocf.py` 하드코딩과 동일 → 기존 모든 config 동작 불변).
- `efficientocf.py`의 `_compute_matched_pair_gmo_losses(loss_weight=0.1)` 하드코딩을 `self.query_gmo_loss_weight`로 교체.
  - 주의: 이 weight는 **focal/bce GMO 항에만** 곱해짐(`utils_loss.py:2889` `pair_loss * loss_weight`). dice는 별도 `query_gmo_dice_loss_weight`(`utils_loss.py:2891`)로 곱해지므로 영향 없음.
- `efficientocf_config.py`: `MODEL_CFG_DEFAULTS`에 기본값 추가 + self 할당 추가.
- 새 config `shape_guide_128_dice_weight_fl25_z.py` (fl25 기반, **2변수 콤보**):
  - `query_multi_gaussian_sigma_max_m` z `1.5→0.6` (force: 단일 가우시안의 수직 과확장 억제).
  - `query_gmo_loss_weight` `0.1→0.5` (reward: 3D z-aware focal 강화 → BEV dice와 동급).
  - 목적: 가우시안이 σ_z를 부풀려 기둥이 되는 대신 **z축으로 분산되는 수직 GMM**을 유도. (단일변수 σ_z 캡만으론 reward 부재로 분산이 안 일어나 납작해지기만 하는 문제 보완.)

## 2026-06-26 18:44:06 KST — matched GMO sigma floor config화 및 fl25=0.25 적용

- `query_matched_gmo_sigma_floor_vox` model config 키 추가: 기본값은 기존 동작과 같은 `0.5`.
- `efficientocf.py`의 matched GMO voxelizer floor 하드코딩 `0.5`를 config 값으로 교체.
- `shape_guide_128_dice_weight_fl25.py`: floor ablation을 위해 `query_matched_gmo_sigma_floor_vox=0.25` 설정. `query_multi_gaussian_sigma_min_m=(0.4,0.4,0.2)`는 유지.

## 2026-06-26 17:09:44 KST — eval mixture3d debug vis 해상도 128x128x10으로 축소

- `debug_query_mixture3d_vis_eval_occ_size`를 `(512, 512, 40)`에서 `(128, 128, 10)`으로 변경.
- 적용 범위: `efficientocf_config.py` 기본값, `shape_guide_128_dice.py`, `shape_guide_128_dice_weight.py`, `shape_guide_128_dice_weight_24.py`.
- 목적: train 중 eval/debug vis를 얹을 때 512 고해상도 mixture3d 렌더로 인한 CUDA/MPS 불안정성과 메모리 spike를 줄임.

## 2026-06-26 KST — 3D calib grid search 제거 (기본값 고정)

- `projects/occ_plugin/occupancy/detectors/efficientocf.py:1651`: `_calibrate_eval_3d_align` grid search를 BEV align과 동일한 패턴으로 교체.
- 기본값 `(t=0, transpose=True, fh=0, fw=0, fz=0)` 고정 — 64회 512³ IoU 연산 제거.
- `EOCF_EVAL_3D_ALIGN=auto` 시에만 기존 grid search 실행.
- `EOCF_EVAL_3D_ALIGN=0,1,0,0,0` 형태로 명시적 override 가능.
- **버그 수정**: 기존 calib은 첫 샘플 prediction이 비어있으면 `(0, False, 1, 1, 1)` 등 노이즈 alignment에 잠기는 문제 있었음.

## 2026-06-26 KST — dist_test.sh MPS 관리 코드 추가

- `tools/dist_test.sh`: `dist_train.sh`와 동일한 MPS(Multi-Process Service) 관리 블록 추가.
- `USE_MPS=1` 기본값. eval 동시 실행 시 GPU SM을 MPS로 공유해 CUDA illegal memory access 방지.
- `USE_MPS=0`으로 비활성화 가능 (eval_total.sh 등에서 환경변수 앞에 설정).

## 2026-06-25 21:39:56 KST — shape_guide_128_dice_weight_24 Gaussian 수 축소

- `projects/configs/baselines/shape_guide_128_dice_weight_24.py`: shape guide Gaussian ablation을 위해 `query_num_gaussians`를 48에서 24로 변경.

## 2026-06-25 KST — shape_guide_128_dice_weight 신규 config: opacity 부활(weight_mode 'ones'→'sigmoid')

- **배경**: 실행 중인 `shape_guide_128_dice` 로그 분석 → `loss_gmo_dice`가 iter 1부터 0.49→0.46으로 **평탄**(657 iter 평균 0.467, 기울기 ~0). over-spread equilibrium에 갇힘. 원인 진단: `weight_mode='ones'`(48개 가우시안 always-on, opacity off-switch 없음) + union이라, FP를 줄이는 길이 "sigma 축소"뿐 → 줄이면 FN(gap) 발생 → 적당히 퍼진 blob이 tversky 최소점.
- **변경**([shape_guide_128_dice_weight.py](projects/configs/baselines/shape_guide_128_dice_weight.py), 128_dice 복사본): **단일변수** `query_multi_gaussian_weight_mode='ones'→'sigmoid'` 한 줄만 변경. union p=1−Π(1−α·G)에서 α=sigmoid(logit)∈[0,1]가 per-gaussian opacity → α→0으로 잉여 blob OFF 가능. tversky FP항(α=0.7)이 그 압력 제공.
- **유지(미변경)**: `occ_combine_mode='union'`(sigmoid 필수쌍, 이미 union이라 단일변수 성립), `sigma_reg=0`, `weight_reg=0`(sparsity는 tversky에 위임), `softplus_bias_init=0.0`(→초기 α=0.5). offset_max=12/sigma_max=3 등 제한값 동일.
- **trade-off/주의**: opacity 부활로 (sigma↔weight) degeneracy 복귀 → sigma 폭주 시 sigma_reg 소량 켜는 것 고려. 관련 주석 3곳(loss 헤더/sigma 블록/weight 블록) 현실에 맞게 갱신.
- **영향 범위**: `efficientocf_config.py` 미변경(새 인자 아님, baseline override 값만 변경). 실행 중인 dice 런 무관(새 런부터 적용).

## 2026-06-25 KST — eval 3D mixture vis 전용 해상도 키 신설 (eval=512³ GT/metric 일치, train=128 유지)

- **문제**: 3D `mixture3d` vis가 train·eval 모두 `matched_gmo_voxelizer`(128, 0.8m)를 써서 그림. 그런데 eval **메트릭**은 `self.voxelizer`(512³, 0.2m)로 계산 → 같은 eval 런서 점수(0.2m)와 cube(0.8m) 해상도 불일치.
- **변경**: eval mixture3d vis만 GT/metric 해상도로 렌더하도록 분리. train vis는 128 유지(loss 격자=loss가 보는 것, every-48 상시라 속도 위해).
  - 신규 config 키 `debug_query_mixture3d_vis_eval_occ_size=(512,512,40)` ([efficientocf_config.py](projects/occ_plugin/occupancy/detectors/efficientocf_config.py): 기본값+apply, [shape_guide_128_dice.py](projects/configs/baselines/shape_guide_128_dice.py) 노출).
  - `efficientocf.py.__init__`: 그 키로 `self.mixture3d_eval_voxelizer` 생성(floor=0, 고해상; range_x/y/z 재사용).
  - `maybe_save_query_mixture_3d_vis`에 `is_eval: bool=False` 파라미터 추가 → eval일 때 eval voxelizer, train일 때 matched(128). **영속 플래그(`_eval_vis_all_ranks`) 대신 명시적 파라미터**로 train-after-eval 오염 방지.
  - eval 호출부 2곳([efficientocf.py:1381,1389](projects/occ_plugin/occupancy/detectors/efficientocf.py))만 `is_eval=True`; train 호출부(3142)는 미전달=128.
- **안전**: per-query bbox 렌더(K=1)+`occ_max_voxels_per_query=4000` 캡으로 512에서도 메모리/matplotlib 부담 bounded. eval은 가끔 도므로 OK. train은 미변경이라 속도 영향 0.
- **주의**: **실행 중인 dice 런엔 영향 없음**(런치 시 로드된 코드 사용). 새 런/eval부터 적용.
- 검증: 4개 파일 `py_compile` 통과. `is_eval=True`는 eval 2곳만, train 1곳 미전달 확인.

## 2026-06-25 KST — 헷갈리는 죽은 흔적 제거(sigma/score 관련 dead code·deprecated key)

- **(1) 레거시 단일-gaussian sigma 인자 제거**: `query_head.py` `gaussian_sigma_min`/`gaussian_sigma_max`(__init__ 파라미터 388-389 + `self.` 할당 577-578). **할당만 되고 어디서도 안 읽힘**(mixture 경로는 `query_multi_gaussian_sigma_min_m/max_m` 사용, efficientocf도 이것만 전달). 헷갈림 유발 → 삭제.
- **(2) deprecated 시각화 score 키 제거**: `efficientocf_config.py` `debug_query_mixture3d_vis_score_threshold`(DEFAULTS 키 + apply 할당). 이미 `debug_query_score_threshold`(+`EOCF_EVAL_FG_THR`)로 통일돼 무시되던 키. config에서도 미사용 확인 후 삭제. 관련 stale 주석(`utils_visualization.py:1335`)도 정리.
- **(3) `sigmoid_to_sigma_from_range` stale 기본값 제거**: 인자 default `(0.15..)/(4.0..)`가 실제(0.4/3.0)와 달라 오해 유발 + 유일 콜러(`query_head.py:1781`)가 항상 명시 전달 → default 삭제(필수 인자화).
- **남겨둔 것(미삭제, 의도)**: `query_multi_gaussian_sigma_min_m/max_m`의 DEFAULTS `(0.15..)/(4.0..)`(`efficientocf_config.py:108-109`, `query_head.py:392-393`)는 **functional fallback**(키 없는 config 대비). 모든 실제 config가 명시 override하므로 효과 없음 — 제거 시 KeyError 위험이라 유지.
- 검증: py_compile OK, 제거 심볼 잔존 참조 0. ⚠️ 진행 중 학습 무관(기능 무변).

## 2026-06-25 KST — eval voxelizer(512) floor 제거(0.0), train(matched) floor는 0.5 유지

- **대상**: `efficientocf.py:217-224` `self.voxelizer`(eval/512) → `gaussian_sigma_floor_vox=0.0`. `efficientocf.py:236` `matched_gmo_voxelizer`(train) → 상속 끊고 `0.5` 명시.
- **이유**: eval은 forward σ를 그대로 splat — sigma_min(head, ≥0.4m)이 이미 보장하므로 0.2m 격자에서 소멸 위험 없음 → 별도 floor 불필요(요청). floor의 '소멸 방지' 역할은 거친 train 격자(0.8m)에서만 필요.
- **영향(기능)**: **사실상 무변화(no-op)**. sigma_min=0.4m라 self.voxelizer floor(옛 0.1m)는 이미 dormant였음(σ 0.4m > 0.1m). self.voxelizer.floor를 공유하던 `utils_bev_pool`도 0이 되나 동일하게 dormant. 코드 의도("eval=raw forward")만 명확화.
- **주의**: 이후 sigma_min을 0.1m 밑으로 내리면 eval에서 sub-voxel 소멸 가능(현재 0.4m라 안전). py_compile OK. ⚠️ 진행 중 학습 미반영.

## 2026-06-25 KST — `sigma_min_m` (0.2,0.2,0.2)→(0.4,0.4,0.2): train/eval 최소 σ 통일

- **대상**: `shape_guide_128.py:451` `query_multi_gaussian_sigma_min_m`. (`efficientocf_config.py` DEFAULTS `(0.15,0.15,0.10)`는 다른 해상도 config가 공유 → 미변경. 각 config는 `0.5×자기 matched voxel`로 둘 것. ⚠️ `shape_guide_128_dice.py`도 같은 변경 필요 — 아래 별도 항목/확인.)
- **이유**: `floor_vox=0.5`는 voxel 단위라 train(matched 0.8m→**0.4m**)·eval(512 0.2m→**0.1m**) 물리 floor가 달라, σ<0.4m 구간에서 **train은 σ를 0.4m로 부풀려 학습 vs eval은 원값 렌더 → 미세 불일치**. `sigma_min_m`을 `0.5×coarsest(matched) voxel = (0.4,0.4,0.2)`로 올리면 σ가 항상 floor 이상 → **양 격자에서 floor 비활성, 네트워크 σ가 train·eval에 동일하게 흐름.**
- **효과**: train/eval 최소 σ 완전 일치(xy 0.4m / z 0.2m). 잃는 디테일 없음 — 모델은 어차피 0.8m matched 격자로만 학습(sub-0.4m σ는 학습 안 된 noise였음).
- **무관**: over-spread(퍼짐)와 별개(그건 offset_max·sigma_max·`weight_mode='ones'`).
- **주의**: matched 해상도 바꾸면 sigma_min도 `0.5×새 voxel`로 재산정(예: 64→0.8m). py_compile OK. ⚠️ 진행 중 학습 미반영 — 다음 실행부터.

## 2026-06-25 KST — `shape_guide_128_dice.py` 신규: occ shape over-spread 대응 dice(Tversky) 단독 실험

- **신규 config** `projects/configs/baselines/shape_guide_128_dice.py` ([shape_guide_128.py](projects/configs/baselines/shape_guide_128.py) 클린 카피에서 **단 한 가지만** 변경).
- **변경 내용**(`model_cfg`, 365–372줄):
  - `use_query_gmo_dice_loss=False → True` (occ loss에 foreground-Tversky 추가).
  - `query_gmo_dice_loss_weight=0.5`, `query_gmo_tversky_alpha=0.7`(FP 가중), `query_gmo_tversky_beta=0.3`(FN 가중) **명시**(값은 [efficientocf_config.py:8-10](projects/occ_plugin/occupancy/detectors/efficientocf_config.py#L8-L10) 기본과 동일 — 자기문서화 목적, 동작 불변).
  - `query_gmo_loss_type='focal'` 유지 → focal+dice 병행(focal=불균형, dice=shape 주력).
- **근거**: focal-only는 over-spread에서 `p.clamp` 포화 + 전격자 mean 희석으로 FP gradient≈0 → shape 못 잡음(폭발). Tversky는 FP를 foreground 크기로 정규화 → 격자 불변·비소멸 gradient, β가 recall 방어. 진단 전말은 [NOTES.md](NOTES.md) 2026-06-25 항목.
- **단일 변수 원칙**: 제한값(`offset_max=12`/`sigma_max=3`)·`weight_mode='ones'`·init **모두 그대로**. dice 효과만 귀속. 코드 변경 없음(메커니즘은 [utils_loss.py:2670-2697](projects/occ_plugin/occupancy/detectors/utils_loss.py#L2670-L2697) `_compute_foreground_tversky_pair_loss`로 기구현·wired).
- **검증**: `py_compile` 통과, 추가 키 3개 모두 `efficientocf_config.py`가 인식. **모니터**: `loss_gmo_dice`가 ~0.5에서 감소하는지 / matched footprint가 차 크기로 조여지는지 / recall(matched count) 유지.

## 2026-06-25 KST — `gaussian_sigma_floor_vox` 기본값 0.35→0.5 (σ 하한을 '반 voxel 방지턱'으로 정정)

- **대상(live 3곳, 0.35→0.5)**:
  - `voxelizer.py:139` `SoftVoxelizerOneAdd.__init__` 기본값(+근거 주석). `self.voxelizer`·`matched_gmo_voxelizer` 모두 이 기본값을 상속(전자는 미지정, 후자는 `float(self.voxelizer...)`).
  - `query_head.py:4618` `compute_query_point_dt_loss`의 floor 인자 기본값(콜러 `utils_loss.py:3672/3677`가 안 넘겨서 default 사용).
  - `utils_bev_pool.py:340` getattr fallback(실사용은 voxelizer attr=0.5라 사실상 표기 일치용).
  - dead 백업 `query_head_ori.py`는 미사용(import X)이라 제외.
- **의미**: floor는 σ가 voxel보다 작아져 grid에서 사라지는(occ 0 → dead-gradient) 것을 막는 **방지턱**. voxel 단위라 해상도 자동 추종(0.5=반 칸 → 128:0.4m / 64:0.8m / 256:0.2m). 엄밀 안전선(최악 정렬서 nearest voxel≥0.5)이 ~0.42 voxel이라 기존 0.35는 살짝 미달 → 0.5로 정정.
- **실효 σ 하한 = max(sigma_min_m, 0.5·voxel)**. `sigma_min_m=(0.2,0.2,0.2)`는 **사용자 요청대로 유지** → 128 matched(0.8m)에선 floor(0.4m)가 지배, eval 512(0.2m)에선 0.1m라 sigma_min(0.2m)이 지배(미세 불일치는 [[NOTES]] 참조). sigma_min_m을 0.4↑로 올리면 floor 비활성 + train/eval 완전 일치.
- **주의**: 이번 변경은 σ '소멸 방지(아래쪽)'만. shape **over-spread(퍼짐)** 와는 무관(그건 offset_max·sigma_max·`weight_mode='ones'` 쪽).
- config 주석 갱신: `shape_guide_128.py`의 floor 수치(0.28m→0.4m)·sigma_min 바닥 설명. 검증: py_compile 4파일 OK. ⚠️ 진행 중 학습 미반영 — 다음 실행부터.

## 2026-06-25 KST — `query_multi_gaussian_pair_chunk` 기본값 8→2

- 대상: `projects/occ_plugin/occupancy/detectors/efficientocf_config.py:112` (DEFAULTS). `8` → `2`.
- 이유: grouped voxelizer 루프 보폭(메모리/속도만, **결과 불변**). GPU 실측상 128격자에서 chunk=2가 속도 sweet spot + 메모리 적당(K80 9.6GB). 8은 128에서 ~35GB(OOM)·느림.
- **파급(override 없는 config)**: `shape_guide.py`(64격자)도 이제 기본 2 사용 → 64에선 chunk=8(45ms·10.5GB)이 最速이었으므로 **약간 느려지지만(71ms) 메모리 1/3(3.3GB)**. 결과/정확도 영향 0. `shape_guide_128.py`는 명시적으로 2라 무변화.
- 동기화: `EfficientOCF_V1.1_1gpu.py`는 현재 tree에 없음(불필요). py_compile·mmcv 로드 확인. ⚠️ 진행 중 학습 미반영 — 다음 실행부터.

## 2026-06-24 KST — mixture3d 2행: 예측 occ를 '예측 해상도 솔리드 큐브'로 (점 scatter → ax.voxels)

- 대상: `projects/occ_plugin/occupancy/detectors/utils_visualization.py` `maybe_save_query_mixture_3d_vis` (학습·추론 **공유** 함수 → 양쪽 다 적용).
- **변경 전**: mixture를 512-res(`self.voxelizer`)로 재복셀화 → 점유 복셀 센터를 sparse 점(s=1.4, alpha=0.18, query당 4000 subsample)으로 scatter → "퍼진 점"처럼 보임.
- **변경 후**: **모델 예측 해상도(`matched_gmo_voxelizer` = `query_matched_gmo_bce_occ_size`, 현재 64×64×20)** 로 복셀화 → 점유 복셀을 **실제 스케일 솔리드 큐브**(`ax.voxels`)로. query별 클래스색 유지, GT 회색 구름 레퍼런스 유지, 1행(1σ ellipsoid) 무변경.
  - **해상도 자동 적응**: 큐브 크기 = range/res → 64→1.6m, 128→0.8m. config(예: `shape_guide_128.py`)만 바꾸면 자동. 별도 vis 격자 기준 불필요.
  - occ 임계값은 통일된 `eval_occ_threshold`(+`EOCF_EVAL_OCC_THR`) 그대로 사용.
  - 속도: 예측 해상도라 큐브 수 적음(측정 64-res 43~949개 23~243ms / 128-res 115~3798개 45~611ms). 512-res 큐브(12만 개, 초 단위) 대비 빠름.
- **검증**: 매칭 48-gaussian 체크포인트(`work_dirs/shape_guide/epoch_1_lss_only.pth`)로 eval 3샘플 → `query_mixture3d_vis/iter_*.png` 2행이 솔리드 큐브로 정상 렌더(렌더 에러/OOM 0). py_compile 통과, dead 참조(gmm_xyz/occ_max_vox) 0.
- 주의: 진행 중 학습은 로드된 옛 코드라 미반영 — 다음 실행부터. (`debug_query_mixture3d_vis_occ_max_voxels_per_query`는 이제 미사용.)

## 2026-06-24 KST — shape_guide_128.py: 학습 occ-loss 격자 64→128(xy)

- **신규 실험 config** `projects/configs/baselines/shape_guide_128.py` (shape_guide.py 복제본). 단일 변경: `query_matched_gmo_bce_occ_size=(64,64,20)` → `(128,128,20)`.
- **배경**: 학습 occ-loss는 `matched_gmo_voxelizer`(loss 격자)에서, 평가 metric occupancy는 `self.voxelizer`(512)에서 가우시안 직접 splat. 64-grid(x,y 1.6m)에선 sigma floor(0.35vox)가 0.56m로 작동 → sigma_min_m=0.2m을 압도, σ_x,y∈[0.2,0.56m] 학습 dead-zone. 또한 자동차 폭이 ~1복셀이라 형상 감독 불가.
- **효과**: 128(x,y 0.8m)로 올리면 floor 0.56→0.28m(dead-zone 축소), 자동차 폭 ~2복셀로 형상 감독 가능. 단 x,y 실효 최소 σ=0.28m로 sigma_min(0.2m)보다 약간 높음(0.2m 도달하려면 격자≥256 필요).
- **해상도 분리 주의**: 평가 metric occupancy = 512 직접 splat(불변, 업스케일 없음). 단 3D mixture **디버그** 시각화(`maybe_save_query_mixture_3d_vis` 2행)는 최근 변경으로 **예측 해상도(`matched_gmo_voxelizer`)** 솔리드 큐브 렌더 → 128 적용 시 큐브 1.6m→0.8m 자동 적응(별도 수정 불필요).
- **메모리·속도 측정 + pair_chunk=2 추가**: grouped voxelizer가 청크당 `[chunk,G,bbox]` intermediate를 backward까지 retain. 흩어진 query를 묶으면 bbox≈전체 격자 → `query_multi_gaussian_pair_chunk` 기본 8이면 128격자에서 ~35GB(OOM). 결과는 chunk 무관(메모리/연산만 조절). GPU 실측(128격자, fwd+bwd, peak/ms): **K80** c8 35GB·132ms / c4 25GB·117 / **c2 9.6GB·89** / c1 1.3GB·131. **K160** c2 17.7GB·158 / c1 2.6GB·257. → **속도 sweet spot=2(모든 K 最速, c1보다 ~1.6배 빠름), 메모리 최소=1.** `shape_guide_128.py`에 **`query_multi_gaussian_pair_chunk=2`** 채택(속도 우선). 객체 多 프레임서 OOM 시 1로 폴백.
- 참고: `efficientocf_config.py` 기본값은 `(128,128,10)`/`pair_chunk=8` 유지(이 config가 override). ⚠️ 진행 중 학습엔 미반영 — 다음 실행부터.

## 2026-06-24 KST — 가우시안 footprint outline 임계값을 eval_occ_threshold로 통일

- **문제**: 2D query_debug_vis의 가우시안 footprint **outline 등고선**(prob 모드)이 별도 키 `debug_query_gaussian_prob_threshold`(0.5)를 탔음 → occ metric 임계값(`eval_occ_threshold`)과 따로 놀아 env override(`EOCF_EVAL_OCC_THR`)도 안 먹음.
- **변경**: outline + 타이틀 텍스트를 렌더러 `prob_threshold` 파라미터로 통일. 이 파라미터는 학습=`eval_occ_threshold`([efficientocf.py:3113](.)) / eval=`EOCF_EVAL_OCC_THR` override([efficientocf.py:1339](.)). → occ 임계값 한 키가 metric·occ-grid·3D mixture3d·**가우시안 footprint outline**까지 모두 통제.
  - `query_head.py`: `_maybe_save_prob_grid_vis` 3330(outline 소스)·3760/3764(타이틀) → `prob_threshold` 사용.
- **dead 키 제거** `debug_query_gaussian_prob_threshold`: `query_head.__init__`(param+검증), `efficientocf.py` QueryHead 생성 kwarg, `efficientocf_config.py`(DEFAULTS+주입+주석) 전부 삭제. 잔존 참조 0, py_compile 통과. (`debug_query_gaussian_prob_alpha_scale`=heatmap 밝기는 별개라 유지.)
- ⚠️ 진행 중 학습은 이미 로드된 코드라 미반영 — 다음 실행(train/eval)부터 적용.

## 2026-06-24 KST — PROJECT_STRUCTURE.md 현행화

- 실제 working tree에 맞춰 `PROJECT_STRUCTURE.md` 재정리.
- 루트: 삭제된 항목 제거(`GPU_MULTIJOB_NOTES.md`, `MPS_MULTIJOB_RUNBOOK.md`, `run_eval_occ_all.sh`, `run_lss_pretrain.sh`, `stop_mps.sh`, `gt_gaussian/`). 추가 항목 반영(`CLAUDE.md`, `GUIDE.pdf`, `train_total.sh`). `eval_total.sh` 설명 갱신.
- `projects/configs/baselines/`: `EfficientOCF_V1.1_*` 3종 → 현재의 `shape_guide.py`(주력), `test.py`로 교체.

## 2026-06-24 KST — eval env 정리(EOCF_EVAL_MODE 0/1 숫자화 + EOCF_EVAL_ALIGN 기본값화)

- 대상: `projects/occ_plugin/occupancy/detectors/efficientocf.py`, `tools/test.py`, `eval_total.sh`, `train_total.sh`.
- **`EOCF_EVAL_MODE` 숫자 인코딩**: `0=present`(현재 1프레임) / `1=future`(미래 n_future프레임). 기본 future.
  - `_eval_mode_is_present()` static helper 추가(`"0"|"present"` → present). 읽던 두 곳(simple_test의 `eval_mode`, `_eval_mode_slice`)을 helper로 통일. 기존 문자열 `present/future`도 하위호환 수용.
- **`EOCF_EVAL_ALIGN` 기본값화**: BEV/3D 기하 정렬 `(transpose,flipH,flipW)`는 데이터 그리드 동일 시 고정값 → 코드 기본값 `(2,1,0,0)`으로 내장(frame `2`는 BEV 정렬에 미사용, 호환용 자리). sh에서 export 제거.
  - env override 유지: `EOCF_EVAL_ALIGN="t,tr,fh,fw"` 강제 / `EOCF_EVAL_ALIGN=auto` 자동 캘리브레이션(IoU 최대화). 기존 forced/auto 2갈래를 default 포함 한 블록으로 정리.
- **eval viz 저장 경로**: `tools/test.py` 자동 dir을 `./work_dirs/eval/<config>/<timestamp>` → `./work_dirs/<config>/eval/<timestamp>`로 (학습 work_dir `./work_dirs/<config>/` 아래 `eval/` 하위로 통합).
- **sh 정리**: `eval_total.sh` CONFIG → `shape_guide.py`, 주석 간결화, `EOCF_EVAL_ALIGN` 줄 제거. `train_total.sh` CONFIG → `shape_guide.py`.

## 2026-06-24 KST — shape_guide.py config 정리(미사용 vis/dead 인자 삭제 + 임계값 2개 명시)

- 대상: `projects/configs/baselines/shape_guide.py` (= 옛 `test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide_real.py` 리네임, _codeonly repo).
- **임계값 2개 명시 추가** (이전엔 미설정 → 기본 0.5로 암묵 동작했음):
  - occ 점유: `model_cfg`에 `eval_occ_threshold=0.5` (env override `EOCF_EVAL_OCC_THR`).
  - fg query score: `visualization_cfg`에 `debug_query_score_threshold=0.5` (env override `EOCF_EVAL_FG_THR`).
  - 둘 다 "config=학습·추론 공통, env=추론만 override" 주석으로 위치 명확화. 머지 경로 검증(apply_model_cfg/apply_visualization_cfg)으로 self.* 정상 주입 확인.
- **삭제 — 안 쓰는 vis 타입(3종=2D query_debug/3D mixture3d/cam_gaussian 외)**: debug_cfg에서 `debug_query_inst_depth_lift_vis_every`, `debug_query_attn_softargmax_vis_every`; visualization_cfg에서 instance_img(dir/max_frames), gt_alignment(dir), inst_depth_lift(dir/max_frames/max_instances), attn_softargmax(dir), `query_attn_vis_dir`. (모두 every 기본값 0 → 키 제거시 자동 off.)
- **삭제 — deprecated/dead**: `debug_query_mixture3d_vis_score_threshold`(deprecated, score는 debug_query_score_threshold로 통일. 코드 실제 read 0건=utils_visualization.py:1335 주석에서만 언급).
- **(정정)** `debug_query_score_iou/cls/cam_attn_weight`는 처음에 dead로 오판해 지웠으나 **실제 live** — `utils_visualization.py:1638-1640`이 fg score 합성식 `score=(w_iou*iou+w_cls*cls+w_cam*cam)/norm`에 사용. 원래 cls-only(0.0/1.0/0.0)인데 삭제 시 DEFAULTS(0.5/0.5/0.0)로 떨어져 score 정의가 바뀜 → **복원**(cls-only 유지). 오판 원인: 이전 grep의 `-v "= float"` 필터가 `w_iou = float(self...)` 사용처를 자체 제거. (멀티에이전트 통일검증 워크플로가 적발.)
- model_cfg(실제 모델/loss)는 무수정. 옛 repo는 호환 위해 그대로 둠(_codeonly만 정리). py 로드/머지 검증 통과(occ=eval_occ_threshold, fg=debug_query_score_threshold, score합성=cls-only).

## 2026-06-24 KST — 중복 임계값 제거(confidence/objectness) → occ+score 2개로 정리

- **불필요 임계값 2개 제거**: `debug_query_confidence_vis_threshold`, `debug_query_objectness_vis_threshold`는 score 임계값과 중복(objectness는 순수 alias, confidence는 센터 마커 색칠 게이트로만 쓰임). 둘 다 삭제하고 동작을 `debug_query_score_threshold`(+`EOCF_EVAL_FG_THR`)로 일원화.
  - `query_head.py`: `__init__`에서 두 파라미터 제거 + alias/검증 블록 제거. 센터 마커 `conf_thr`(:3311)을 `EOCF_EVAL_FG_THR or debug_query_score_threshold`로 라우팅. bundle score_thr 기본값도 `debug_query_score_threshold`.
  - `efficientocf.py`: QueryHead 생성 시 두 인자 전달 제거.
  - `efficientocf_config.py`: DEFAULTS 키 2개 + 주입 라인 2개 삭제.
- **결과**: config의 의미적 임계값은 **occ(`eval_occ_threshold`) + query score(`debug_query_score_threshold`)** 둘뿐. 각각 env override(`EOCF_EVAL_OCC_THR`/`EOCF_EVAL_FG_THR`)로 추론만 덮어쓰기. (footprint 색칠 농도용 `debug_query_gaussian_prob_threshold`는 별개 렌더 디테일이라 유지.) 코드 잔존 참조 0, config 설정 0, py_compile 통과.

## 2026-06-24 KST — query score 임계값 통일(2D선택=2D표시=3D필터) + config 가시성 정리

- **score 임계값 통일**: 2D query 선택은 이미 `debug_query_score_threshold`(config)+`EOCF_EVAL_FG_THR`(env override)로 학습·추론 공통이었으나, **3D mixture3d 필터만 별도 `debug_query_mixture3d_vis_score_threshold`(0.75)** 였음. `utils_visualization.py:1337`을 동일 소스(`EOCF_EVAL_FG_THR or debug_query_score_threshold`)로 변경 → **2D선택/2D표시/3D필터가 단일 임계값**. 옛 3D score 키 deprecated(무시).
- **config 가시성**(`efficientocf_config.py` VISUALIZATION_CFG_DEFAULTS): 임계값 한눈에 보이게 주석 블록 추가 — occ는 `eval_occ_threshold`(+EOCF_EVAL_OCC_THR), query score는 `debug_query_score_threshold`(+EOCF_EVAL_FG_THR), 둘 다 "config=학습·추론 공통, env=추론 override" 패턴 명시.
- 결과: occ 임계값과 score 임계값 둘 다 동일 패턴(config 1개가 학습·추론 일관, env로 추론만 override). 별도/이상 임계값 없음. py_compile 통과.

## 2026-06-24 KST — mixture3d 2D방향 일치(측정검증) + 센터 검정/작게 + future all행 전체fg쿼리

- **2D↔3D 방향 불일치 측정으로 규명·수정**(`utils_visualization.py`): 워크플로 분석 + `proj3d.proj_transform` 직접 측정. 결과 **원래 azim(-60/-90)이 worldX 좌→우로 2D와 일치**했고 직전 +180(120/90)이 그걸 뒤집었음(되돌림). 추가로 **y축 반전**(`set_ylim(ymax,ymin)`)하니 X(좌→우)·Y(위→아래)·Z(위) 셋 다 2D BEV와 일치. occ Z는 z_idx=0=지면(-4.9m)으로 **뒤집힘 없음 확인**(차 바퀴 땅에 정상) — "바퀴 하늘"은 잘못된 azim 뷰각 탓이었음.
- 센터 마커: magenta+흰테두리 → **검정·테두리없음·s45**(요청). GT=lime 얇은 검은테두리. 컴포넌트 점 옅게(alpha 0.5).
- 겹침: bottom GT 진하게(0.45)/예측 옅게(0.18)로 둘 다 보이게.
- **future 2D "all"행 = 전체 fg 쿼리**(`efficientocf.py` `_build_eval_future_2d_bundle`에 conf/cls 전달, candidate=background 제외 전체) → present처럼 임계값 미달 저신뢰 쿼리까지 표시.
- 검증(noli occ epoch_13, 풀해상도, 8GPU 병렬): present/future 2D·3D 정상. 결과 `work_dirs/sample_test_vis_check/V6/{present,future}/`. ⚠️ 3D 센터 가시성은 matplotlib 깊이정렬 한계로 dense 영역에서 일부 묻힐 수 있음(필요시 2D 투영 오버레이로 강제). py_compile 통과.

## 2026-06-24 KST — ★future 미표시 진짜 버그(feat_out 인덱스 off-by-one) 수정 + mixture3d mode별 프레임/센터/투명도

**핵심 버그 수정**: future 2D/3D에 예측이 안 나오던 진짜 원인은 체크포인트가 아니라 **feat_out 인덱스 off-by-one**. future mixture를 `feat_out[30~33]`로 잡았는데 `[30]`은 centers가 아니라 **sigmas**(반환 순서: [29]centers/[30]sigmas/[31]yaw/[32]weights). 그래서 sigma(값 0.7~2.0)를 mixture center로 써서 좌표가 한 점에 뭉쳐 footprint가 전혀 안 그려졌음. 진단: `EOCF_VIS_DBG=1`로 mix_c 좌표가 x[0.7,2.0]임을 확인→`[29~32]`로 수정하니 x[-42.5,43.5](world)로 정상화. **결과 future 2D footprint 100px→8923/43870px**(present처럼 footprint, occ-overlay 불필요).
- `efficientocf.py`: mix_future 인덱스 [30~33]→**[29~32]**. occ-overlay 자동ON 제거(이제 footprint 동작). mixture3d도 **mode별 분기** — present=present bundle+full gt(present 프레임), future=future bundle+미래 tail gt+**frame_idx=1(2번째 미래 프레임)**. future bundle(인덱스 수정으로 valid)을 mixture3d가 받아 렌더(이전 0장→4장).
- `utils_visualization.py`: mixture3d에 `frame_idx` 파라미터(표시 프레임 override). 센터 점 s50→**s35**(작게), 예측=magenta·GT=lime, **흰 테두리+zorder=10**(구름 위 대비). 겹침/센터 가시성 위해 구름 더 투명(top GT 0.25, bottom GT 0.28, gmm 0.30).
- **검증(noli occ epoch_13, 풀해상도)**: present 2D 8884/8490·3D present프레임 / future 2D 8923/43870·3D 2번째미래프레임. 둘 다 footprint 풍부. 결과: `work_dirs/sample_test_vis_check/OCC_FINAL/{present,future}/`. (참고: ksh test_traj_mcls_full은 multi-class라 "class", noli가 "occ".) py_compile 통과.

## 2026-06-24 KST — future 2D 예측 표시(occ-overlay)·근본원인 규명 + epoch_15 검증 + 센터/투명도 디자인

- **future 2D 빈 예측의 진짜 원인 규명**: 체크포인트가 아니라 코드. epoch_15(완전학습 test_traj_mcls_full)로도 future **footprint는 100px(빔)** — trajectory-propagated mixture는 footprint 투영이 안 됨(present footprint는 8513px 정상). `_build_eval_future_2d_bundle`에 `score_thr=0.0`+표시키 추가해도 footprint는 안 떴음.
- **해결**: future는 예측 occ(dense)를 **blend 오버레이(행별 lo_color/hi_color 반투명)**로 표시. `_maybe_save_eval_query_vis`에서 future 시 `EOCF_VIS_PRED_STYLE=blend` 자동(미설정 시). 결과 future 2D 11733px. present는 footprint 그대로(8513px).
- **epoch_15 검증 경로 확보**: ksh의 `test_traj_mcls_full.py`+`epoch_15_lss_only.pth`가 이 repo와 모듈/dim/num_gaussians 호환 → `full15.py`(복사+test_capacity 4+3D score 0.05)로 가져와 정상 동작(buffer 키 mismatch만, 무해). noli latest는 epoch_13인데도 forecasting 미학습이라 future occ 약함.
- **mixture3d 디자인**(`utils_visualization.py`): 센터 점 s30→**s50**, 예측=**magenta**·GT=lime 고정색(클래스색은 구름과 안 보임), 검은 테두리. 겹침 가시성 위해 구름 **양쪽 반투명**(top GT 0.55→0.40, bottom GT 0.32→0.42, 예측 gmm 0.20→0.42).
- 결과: `work_dirs/sample_test_vis_check/EPOCH15/{present,future}/`. py_compile 통과.

## 2026-06-24 KST — GUIDE-real Phase A: union+opacity제거(weight_mode='ones') + 48가우시안 + focal-only (test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide_real)

**핵심 내용**: GUIDE 논문(eq3 `p=1-Π(1-G)`, weight 없음, peak=1)에 충실하게 occupancy 표현을 정렬. 기존 `guide`(union+sigmoid α opacity)에서 **학습 opacity α를 제거** → weight collapse/몰빵 + (sigma↔weight) degeneracy가 동시에 소멸하고, occ focal이 sigma를 직접 통제. 회전(yaw→full quaternion)은 8파일/138참조 교차변경이라 Phase B로 분리.

**변경 — query_head.py**: `weight_mode`에 `'ones'` 추가 (검증 튜플 611 + 분기). `weights_qg = ones_like(logits)` (weight head 출력 무시), surrogate는 uniform(sigma 2차모멘트용). softplus_bias_init 분기(696)엔 미포함(불필요).

**변경 — config (`..._guide_real.py` model_cfg)**:
- `query_multi_gaussian_weight_mode`: 'sigmoid' → **'ones'** (combine_mode='union' 유지 → `1-Π(1-G)`)
- `query_num_gaussians`: 16 → **48** (GUIDE 최고 성능; 실측 오버헤드 추론 +0%/+0MB, 학습 +1.1%/+0.6%)
- `use_query_gmo_dice_loss`: (기본 True) → **False** (GUIDE처럼 occ focal only)
- `query_multi_gaussian_sigma_max_m`: (2,2,0.7) → **(3,3,1.5)** (α 제거로 degeneracy 사라져 loss가 sigma 통제 → 난간만 느슨하게)
- `query_multi_gaussian_sigma_min_m`: (0.15…) → **(0.2…)** (voxel 0.2m 바닥)
- `query_multi_gaussian_sigma_reg_loss_weight`: 0.001 → **0.0** (α 없으니 불필요)
- offset_max (12,12,2)·weight_reg 0·eval_use_mixture True는 유지

**검증**: query_head/config py_compile 통과, `Config.fromfile` 파라미터 반영 확인, **`build_model` 성공**(weight_mode='ones'+48 구성).

**주의/다음(Phase B)**: yaw→full quaternion 회전은 `mixture_yaw_tkg`[T,K,G]가 head→voxelizer→loss→matcher→viz→geometry 8파일에 스칼라로 박혀 있어 쿼터니언[T,K,G,4]로 일관 교체 필요(렌더 hot-loop Mahalanobis도 full R 적용). 별도 careful 단계로 진행. binary fg(car~trailer 공유)라 학습 후 **car IoU/precision + over-coverage(α 없어 FP 코어) 모니터** 필수.

## 2026-06-24 KST — mixture3d 2뷰화/센터 작은점 + 풀해상도 검증(저해상도가 예측 붕괴 주범)

- **저해상도가 예측을 붕괴**시킴 규명: 320~448px에선 LSS feature가 작아져 query 예측이 거의 0(2D hi행 25px, 3D 0px). **풀해상도(896×1600)에선 epoch_2여도 present 예측 풍부**(2D hi 8490px, 3D 24963px). 그동안 "예측 없음"은 대부분 저해상도 아티팩트였음.
- **mixture3d 디자인**(`utils_visualization.py`): 뷰 3→**2개**(각도만 다른 중복 제거, `add_subplot(2,n_views,…)`, figsize `9*n_views×18`). 센터 마커 큰 원(s=160) → **작은 점(s=30)**, 모든 뷰 동일, GT=lime/예측=클래스색·검은 테두리. 1열/2열 차등·점수텍스트는 이전에 제거됨.
- **future 3D=present shape 유지**(`efficientocf.py`): mode-일관성으로 future bundle을 mixture3d에 넘기니 조기종료(빈 이미지) → 되돌림. mixture3d는 항상 present-frame **객체 SHAPE** viz, 2D query_debug_vis가 시간축(future occupancy) 담당. (즉 future 3D는 present shape를 보이고, future 2D는 미래 occ — epoch_2 forecasting 미학습이라 빔.)
- **검증(풀해상도)**: present/future 모두 2D·3D 정상, 3D 2뷰 1693×1903. 결과: `work_dirs/sample_test_vis_check/FINAL/{present,future}/`. 임시 config: `sample_test_fullres.py`(896×1600, test_capacity=4, 3D score 0.05). py_compile 통과.

## 2026-06-24 KST — 2D 예측 occ 오버레이 자동ON 해제(기존 footprint로 복귀) + future 빈예측 원인규명

사용자 피드백("future가 이상하게 찍히고 예측 없음, present처럼 나와야"). 진단: occ 오버레이(통짜 채움)가 present의 footprint와 달라 "이상"하게 보였고, future 예측이 비는 건 **렌더 문제가 아니라 epoch_2 체크포인트가 forecasting 미학습**이라 미래 mixture가 거의 비어서임(traj mixture는 efficientocf.py:1175-1186에서 present로 fallback하므로 데이터 자체는 유효).
- `efficientocf.py`: `_maybe_save_eval_query_vis`의 future 자동 `EOCF_VIS_PRED_STYLE=blend` 설정 **제거** → 예측은 기존 footprint(행별 lo/hi 색)로 그림. occ 오버레이는 `EOCF_VIS_PRED_STYLE` 명시 시에만(기본 OFF). future 2D bundle/overlay 코드는 유지(좋은 체크포인트에서 동작).
- **검증(저해상도, epoch_2)**: present 2D 2행 예측 footprint **6143px**(정상) vs future **100px**(빈≈모델 미예측). 둘 다 새 mixture3d(true-z, 열별 센터, 단일색 GT) 동일. 결과: `work_dirs/sample_test_vis_check/v3_clean/{present,future}/`. py_compile 통과.
- ⚠️ **future 예측을 보려면 더 학습된 체크포인트 필요**(epoch_2는 present만 예측됨). 디자인 미리보기는 present 모드 권장.

## 2026-06-24 KST — 2D 예측 오버레이 기존색화 + mixture3d 센터 재설계(열별 차등)

사용자 피드백 반영.
1. **query_debug_vis 예측 오버레이를 '기존처럼'**(`query_head.py`): 단일색+흰겹침 실험 → 행별 원래 색(all=lo_color red, hi=hi_color cyan)으로 반투명 blend. `EOCF_VIS_PRED_COLOR` 지정 시만 override. future 모드는 `_maybe_save_eval_query_vis`에서 `EOCF_VIS_PRED_STYLE=blend` 자동 설정(미설정 시) → 미래 예측 자동 표시.
2. **mixture3d 센터 재설계**(`utils_visualization.py`): 점수 텍스트 제거. 예측 센터를 흰→**검은 테두리(lw1.8)**로 진하게. 뷰(열)별 차등 — 1열=center 보기용 크게(예측 s160 + GT 센터 lime s150 복귀), 2열=중간(s85), 3열=작게(s55). GT 센터는 1열에만.

**검증(저해상도 320×576, test_capacity=4)**: Run A(기본 blend) 2D — 2행 all=red 4967·3행 hi=cyan 4999(기존색 OK). Run B(outline) 정상. 둘 다 3D 4장. OOM/에러 0. 결과: `work_dirs/sample_test_vis_check/v2_design/{run_A_blend_origcolor, run_B_outline}/`. py_compile 통과. ※ GPU 포화로 320×576 저해상도 검증(품질↓, 코드동작 확인용).

## 2026-06-24 KST — mixture3d true-z(z 과장 제거) + 2D 예측 occ 오버레이(디자인 변형)

future 2D에서 예측이 여전히 안 보이던 문제 재진단: gaussian footprint 투영이 미래(trajectory) mixture로는 거의 안 그려짐(예측 행 px ~100). 견고하게 해결.
1. **mixture3d z 원래 비율**(`utils_visualization.py`): 3D 3뷰 중 왼쪽 2뷰의 z 과장(z_aspect 0.22≈2.8x)을 제거하고 세 뷰 모두 true-z(0.078)로.
2. **2D 예측 occ 오버레이**(`query_head.py:_maybe_save_prob_grid_vis`): footprint 대신 예측 occ(dense, `pred ≥ prob_threshold`)를 BEV로 collapse해 행에 직접 그림. `EOCF_VIS_PRED_STYLE` 설정 시에만 동작(미설정=기존 유지):
   - `fill`/`overlap`: pred-only=지정색, GT∩pred 겹침=흰색(둘 다 보임)
   - `outline`: 예측 경계선만(GT 채움 그대로 비침)
   - `blend`: 예측 반투명 덧칠
   - 색은 `EOCF_VIS_PRED_COLOR`(red/magenta/yellow/cyan/orange).
   `gt_np` 옆 closure로 구현, all/hi/hi_cls 행에 적용.

**검증(저해상도)**: `sample_test_lowres.py`에 `test_capacity=4` 추가(소량 자연종료). 변형 생성: overlap+red(2행 예측 빨강1639+겹침흰8624 OK), blend+cyan(2행 cyan 4892 OK). outline+magenta는 GPU 경합 OOM으로 보류. 결과: `work_dirs/sample_test_vis_check/design_variants/`. py_compile 통과.

## 2026-06-24 KST — mixture3d 센터(예측만 점) + future 2D 예측 표시 복원

1. **mixture3d 센터 표시 변경**(`utils_visualization.py`, 학습·추론 공통): GT 센터 마커(lime 점) 제거(상·하단), GT 클래스 텍스트 라벨만 유지. 예측 query 센터는 점(o)으로 유지(흰 테두리·depthshade off).
2. **future 2D viz에 예측이 안 보이던 문제 수정**(`efficientocf.py`): 2D 렌더러는 예측을 gaussian footprint(bundle)로만 그리는데, `_build_query_visualization_bundle`이 `score_frame_count=min(time_receptive_field,…)`로 receptive(≤3)프레임만 반환 → 미래4와 프레임 불일치라 앞 단계에서 bundle을 비웠더니 예측이 전부 사라졌었음. 해결: `_build_eval_future_2d_bundle` 신설 — present bundle의 선택된 query를 미래 centers + 미래 mixture(`feat_out[30~33]` trajectory mixture)로 `[F,Qsel,…]` re-index해 직접 구성. selected/candidate 둘 다 채워 all/hi 행에 예측 footprint 표시. mixture3d/cam은 present full bundle 유지. 텐서 없으면 meta-only로 graceful fallback.

`py_compile` 통과 + future-bundle shape 단독검증.

**라이브 검증(저해상도)**: 풀해상도 eval은 ~23GB 필요한데 GPU가 전부 타 학습잡으로 포화(여유 15~19GB)라 OOM. 검증용으로 `sample_test_lowres.py`(noli 복사 + `input_size` 896×1600→448×800, LSS 메모리 ~1/4) 생성해 GPU2에서 future 모드 실행 성공(OOM 없음, frustum size mismatch는 무해한 load 경고—buffer라 모델 자체 저해상도 frustum 사용). 결과: 2D=2060폭(미래4프레임) 정상, **예측 footprint 픽셀 16,267개로 표시 확인**(수정 전 ~0). metric IoU2d=0.059/IoU3d=0.036. mixture3d/cam도 생성. 이미지: `work_dirs/sample_test_vis_check/lowres_future/`. ※ 저해상도라 예측 품질은 풀해상도보다 낮음(코드 동작 검증용). 풀해상도는 GPU 여유 시 `sample_test.py`로 재실행.

## 2026-06-23 KST — mixture3d 센터 가시성 + 추론 2D 하위폴더/미래4프레임 + EOCF_EVAL_MODE(present/future)

사용자 요청 4건 반영(`efficientocf.py`, `utils_visualization.py`).
1. **mixture3d 센터 가시성**(학습·추론 공통 렌더러): 가우시안에 가려 안 보이던 센터 수정. GMM occ alpha 0.28→0.20, 센터를 별(`*`,s150)→**점(`o`)**, 적당한 크기(GT lime s55·예측 클래스색 s60), `depthshade=False`로 깊이 흐림 제거, 구름 위에 마지막으로 그려 안 가리게. 예측 센터는 흰 테두리로 대비.
2. **추론 2D를 `query_debug_vis/` 하위폴더로**: 기존엔 `sample_*.png`가 EOCF_EVAL_VIS_DIR 루트에 그냥 떨어졌음 → 학습과 동일하게 하위폴더 저장(`_maybe_save_eval_query_vis`의 `head.debug_vis_dir`).
3. **추론 2D를 미래4프레임 표시**: 기존엔 present 1프레임만. future 모드면 미래 n_future 프레임(centers_eval/pred_occ_eval/gt future-tail, max_frames=n_future)으로 표시. 2D는 present-기반 bundle(receptive 3프레임)과 프레임수 불일치 → per-query gaussian footprint 행은 생략하고 GT occ vs 예측 occ(pred_occ_future)+센터점으로 미래4 비교(mixture3d/cam은 full bundle 유지).
4. **`EOCF_EVAL_MODE`(present/future, 기본 future) 신설**: metric과 viz가 함께 따름. `_eval_mode_slice` 헬퍼 추가(present=현재프레임 idx `T-n_future-1`, future=미래 tail), metric의 `_eval_future_tail` 4곳 교체, present 모드는 pred_txyz를 현재 1프레임으로 override. future 코드경로는 불변(회귀 없음).

**검증**(noli ckpt, val, 1GPU, OCC_THR=0.5): future → 2D 2060폭(4프레임)·metric IoU2d=0.036/IoU3d=0.013, query_debug_vis 하위폴더 확인·루트 누출 0, mixture3d/cam 정상. present → 2D 512폭(1프레임)·metric IoU2d=0.080/IoU3d=0.027. 크래시 0. `eval_total.sh`에 `EOCF_EVAL_MODE` 추가. 결과 이미지 `work_dirs/sample_test_vis_check/infer_{future,present}/`.

## 2026-06-23 KST — 추론 시각화 학습 정렬 (2D/3D/cam_gaussian 추가) + sample_test.py + 스모크 검증

추론(dist_test) 시각화를 학습과 정렬. 4종 중 **3종 추가 완료·검증**, 1종(attn_softargmax)은 매칭 의존으로 보류.
- **원칙**: 스타일/threshold 파라미터는 전부 config(`self.debug_query_*`, `eval_occ_threshold`)에서 읽어 학습=추론 동일. 추론 전용 override는 출력 dir + every(=1, 실제 샘플링은 `EOCF_EVAL_VIS_EVERY`가 담당)뿐.
- `efficientocf.py` 추론 경로(`simple_test`/`_maybe_save_eval_query_vis`): `extract_feat_query`를 `EOCF_EVAL_VIS=1`일 때 `return_instance_img_debug/_fullseq/return_query_cam_gaussian_vis_debug=True`로 호출 → `feat_out[14]`(attn weight)/`[16]`(instance_img bundle)/`[17]`(match_inputs) 캡처. `_maybe_save_eval_query_vis`에 인자 추가, **query_attn_softargmax_vis + query_cam_gaussian_vis 호출 추가**(mixture3d는 기존). `_eval_vis_all_ranks` 호이스트.
- `utils_visualization.py`/`utils_instance_img_debug.py`: attn·cam_gaussian 렌더러의 `_is_main_process` 게이트가 `_eval_vis_all_ranks` 존중(eval은 rank별 다른 샘플 → 전 rank 저장).
- **신규 config**: `projects/configs/baselines/sample_test.py` (noli config `_base_` 상속). 시각화 정렬 테스트용.
- **스모크 검증**: noli ckpt(epoch_2)+nuScenes val, 1GPU, `EOCF_EVAL_VIS=1 EOCF_EVAL_VIS_EVERY=1 EOCF_EVAL_OCC_THR=0.2`. 결과 — query_debug_vis(2D)✅ / query_mixture3d_vis(3D)✅ / query_cam_gaussian_vis✅ PNG 정상 생성, 에러 0. EOCF_EVAL_OCC_THR=0.2가 metric+2D+3D에 일관 적용됨 확인. attn 코드 제거 후 재검증 → 3종 정상·attn dir 미생성(clean) 확인.
- **query_attn_softargmax_vis 제외(사용자 결정)**: 이 viz는 GT 매칭 결과(inst_match_result) 필수인데 추론은 매칭을 안 함(`{}`). 추론에서 `_match_queries_to_gt_instances`를 돌리려면 GT 인스턴스 center/ids/cls prep + bev feature가 필요(별도 큰 작업)해서 **3종으로 확정**. 임시로 넣었던 attn 호출/인자/feat capture/rank-bypass는 전부 되돌려 추론 경로 clean(잔재 0).

## 2026-06-23 KST — 불필요 문서 정리 (stale 스냅샷 + 이식 끝난 포팅 가이드 삭제)

루트의 잉여 md 파일 5종 삭제로 문서 정리. 삭제 대상: `4_CHANGELOG.md`·`4_NOTES.md`(옛 `Autonomous_Driving_26_ksh/` repo 스냅샷, 05-15에서 멈춘 stale 복사본 — 현행 `CHANGELOG.md`/`NOTES.md`로 대체됨), `QUERY_ATTN_CHANGES.md`·`BINARY_FG_CLS_CHANGES.md`·`GATE_RECRU_GUIDE.md`(타 repo 이식용 일회성 가이드, 이식 완료). 추적 파일 3종은 `git rm`, untracked 2종은 `rm`. CHANGELOG/NOTES 본문의 과거 언급(이식 기록)은 사적 history라 그대로 보존. sh 스크립트는 `run.sh`/`run_eval.sh`가 `train_total.sh`·eval 래퍼의 의존성이고 `train_total_2~4.sh`는 동시학습 잡 변형이라 전부 유지.

## 2026-06-23 KST — query_mixture3d_vis 가독성 개선 (GT↑/예측↓/중심마커 위로)

`maybe_save_query_mixture_3d_vis` 렌더링 대비 재조정(학습·추론 viz 공통, `utils_visualization.py`). 하단(예측 GMM occ vs GT) 행에서 GT 회색 구름이 `alpha=0.12`로 너무 옅고 예측이 `alpha=0.5` 풀 클래스색으로 너무 세서 비교가 안 되던 문제.
- **GT 구름 진하게**: 하단 `c=0.6,s=0.6,alpha=0.12` → `c=0.45,s=1.2,alpha=0.32`. 상단(1σ) GT occ도 `s=1.0,alpha=0.35` → `s=1.6,alpha=0.55`.
- **예측 옅게**: 하단 GMM occ `s=2.0,alpha=0.5` → `s=1.6,alpha=0.28` (GT가 비쳐 보이게).
- **중심 마커 위로/크게**: 그리는 순서를 구름→중심으로 바꿔(예측 구름이 중심 dot을 덮던 문제) 중심이 맨 위. GT 중심 lime X `s=60→90, lw 2.0→2.8`(상·하단), 예측 query 중심은 하단에서 클래스색 **별(`*`, s=150, 검은 테두리)**로 강조.

py_compile 통과. 시각화 전용이라 학습/평가 수치 영향 없음.

## 2026-06-23 KST — `eval_total.sh` 추가 (단일 eval 원클릭 래퍼)

`train_total.sh`처럼 값만 수정하고 `bash eval_total.sh`만 실행하면 되는 평가 래퍼 신설. 상단에 `CONFIG`/`CHECKPOINT`/`GPUS`/`PORT` + `EOCF_EVAL_*`(ALIGN/OCC_THR/FG_THR/VIS/VIS_EVERY)를 export로 정의 후 `tools/dist_test.sh` 호출. `EOCF_EVAL_OCC_THR`은 이번 단일화로 metric+2D+3D viz에 동시 적용됨을 주석화. ⚠️ 기본 `CONFIG`(`test_traj_mcls_full_evalmix.py`)·`CHECKPOINT`는 `_ksh` repo 경로라 이 repo엔 없음(주신 값 그대로 박아둠, 사용 전 확인 필요). `bash -n` 통과, 실행권한 부여. PROJECT_STRUCTURE.md 갱신.

## 2026-06-23 18:23:42 KST — occupancy threshold 전역 단일화 (`eval_occ_threshold` 신설) + 추론 mixture3d 시각화 추가

**배경(audit)**: shape 관련 인자가 학습/추론 × 실제/시각화 4개 맥락에 흩어져 있는지 전역 조사. **shape 생성·합성**(query_head mixture + voxelizer `occ_combine_mode`/`weight_mode`/sigma·offset bound + `query_eval_occ_use_mixture` 기본 True)은 train/eval 일관 → 문제 없음. **occupancy threshold**가 0.5 근처 4개 노브로 출처가 제각각이라 꼬여 있었음: ①평가 metric=`EOCF_EVAL_OCC_THR`(env), ②2D viz occ-grid=학습 **하드코딩 0.5**/추론 env, ③가우시안 footprint 색칠=`debug_query_gaussian_prob_threshold`(config), ④3D mixture occ=`debug_query_mixture3d_vis_occ_threshold`(config, base 0.2/guide·noreg 0.5, **추론 미생성**).

**변경 — ①②④를 단일 키로 통일**:
- `efficientocf_config.py`: `eval_occ_threshold`(기본 0.5) **신설**(MODEL_CFG_DEFAULTS:129, 주입 371). occupancy 점유 판정의 단일 출처. 런타임 override는 `EOCF_EVAL_OCC_THR`(eval 한정) 유지.
- `efficientocf.py`: ①metric `thr = env or self.eval_occ_threshold`(1428). ②학습 viz occ-grid `prob_threshold=0.5`(하드코딩) → `self.eval_occ_threshold`(2943). 추론 viz occ-grid는 기존대로 `thr` 사용(이미 ①과 동일).
- `debug_query_mixture3d_vis_occ_threshold` **제거**(deprecated): DEFAULTS·주입(efficientocf_config.py)·렌더러 참조(utils_visualization.py:1364) 삭제. mixture3d occ_thr은 이제 `eval_occ_threshold` 참조(eval은 env-resolved 값을 `occ_threshold` 인자로 직접 받음). config 7개(`bg`,`head`,`head_gau`,`head_noli`,`noli`,`noli_guide`,`noli_noreg`)의 해당 라인·주석 삭제.

**변경 — ④ 추론 mixture3d 추가 (parity)**:
- `efficientocf.py:_maybe_save_eval_query_vis`: 2D viz 직후 `maybe_save_query_mixture_3d_vis` 호출 추가. `<EOCF_EVAL_VIS_DIR>/query_mixture3d_vis/`에 저장, occ_thr=metric과 동일(`prob_threshold`). 이제 metric 계산 시점(추론)에 3D shape 확인 가능.
- `utils_visualization.py:maybe_save_query_mixture_3d_vis`: `occ_threshold` 인자 추가, rank 게이트가 `_eval_vis_all_ranks` 존중(eval은 rank별 다른 샘플 처리 → 전 rank 저장, 2D와 동일 정책).

**미변경(의도)**: ③ footprint 색칠(`debug_query_gaussian_prob_threshold`)은 개별 G density 기준이라 의미가 달라 별도 노브 유지. union↔sigmoid / poisson↔softplus 쌍 guard는 이번 범위 외(NOTES에 후속 기록).

**검증**: 변경 3개 코드 + config 7개 py_compile 통과. `debug_query_mixture3d_vis_occ_threshold` 잔여 참조 0건.

## 2026-06-23 16:55:40 KST — eval occupancy mixture 정렬 전역 기본값화 (False→True) + 단일경로 dead-code 확인 (양쪽 레포)

**핵심 내용**: 추론/eval-시각화를 학습(16-gaussian mixture)과 완전히 정렬. `query_eval_occ_use_mixture` 기본값을 `False`→`True`로 변경(두 레포 `efficientocf_config.py` 동일: occ_shape 123행 / occ 117행). 이제 모든 config가 eval occupancy(present 정렬 + 미래 metric)와 eval 시각화 occupancy 패널을 단일 ellipsoid가 아닌 mixture로 splat.

**안전성**: mixture 텐서가 없는 모델은 `simple_test`의 `use_mix` 가드(텐서 유효성·dim·shape 검사)가 자동으로 단일 ellipsoid fallback → 비-mixture config도 무파손. configs/ 전체에 명시적 False override 없음(전 config True 적용). 양쪽 config.py py_compile 통과.

**audit 결과 — 학습 시각화는 변경 불필요**: `extract_feat_query`의 present 단일 voxelize(`self.voxelizer(...)`, occ 1203 / occ_shape 1204)는 두 호출부(occ 1494/2301, occ_shape 1369/2170)가 모두 `voxelize=False`라 **dead code**(`pred_occ`=None). 학습-시각화 occupancy 패널(`maybe_save_query_debug_vis`의 `pred_occ_prob`/`pred_occ_prob_all_obj`)은 이미 None을 받아 단일 ellipsoid를 안 그리고, shape 시각화는 mixture GMM 패널이 담당 → 학습 forward 핫패스 미변경(불필요한 per-query 루프 slowdown 위험 회피).

**관련**: `test_traj_mcls_full_evalmix.py`의 명시적 `query_eval_occ_use_mixture=True`는 이제 기본값과 동일(중복이지만 무해). base `test_traj_mcls_full.py`도 mixture로 전환되어 이전의 "평균 baseline vs 각각" 비교 구도는 사라짐(사용자가 전부 mixture로 통일 요청).

## 2026-06-23 15:43:38 KST — noli 계열 gaussian shape bound 데이터 기반 재설정 (offset 12 / sigma_max 2.0,2.0,0.7)

**핵심 내용**: noli 계열에서 16개 가우시안이 offset은 안 움직이고 sigma만 키워 **원형 blob**으로 수렴하는 문제. 원인은 sigma_max=3.5m가 과도해서 가우시안 1개로 객체 전체를 덮는 "쉬운 gradient 경로"가 열려 있었기 때문(offset 협응 학습 불필요).

**데이터 근거** (`gt_gaussian/sample_frame3_inst14_gmm16.npz`): 실제 인스턴스(extent 8.6×6.2×3.6m, 26,752 voxel)에 diagonal 16-GMM fit 시 per-gaussian sigma는 **xy median 0.90m / max 1.77m, z median 0.60m / max 0.67m**. 즉 차보다 큰 객체도 16개로 쪼개면 sigma는 ~0.9m, tail도 1.8m 미만 → 3.5m는 명백히 과대.

**변경** (4개 config: `_noli`, `_head_noli`, `_noli_guide`, `_noli_noreg`):
- `query_multi_gaussian_offset_max_m`: (10,10,2) → **(12,12,2)** — trailer(전장~19m) 커버를 sigma 아닌 offset 확장으로.
- `query_multi_gaussian_sigma_max_m`: (3.5,3.5,1.5) → **(2.0,2.0,0.7)** — xy는 fit max(1.77)에 맞춘 상한(blob path 차단), z는 fit max(0.67) 기준. sigma_max 작아지면 init sigma도 작아져 under-cover → offset gradient 활성화.
- `sigma_min`(0.15)·`sigma_reg`(0.001)·`num_gaussians`(16)은 미변경. 각 파일 [shape 실험] 주석도 데이터 근거로 갱신.

**주의/모니터**: binary fg라 car~trailer가 이 한 세트 공유. trailer가 비어 보이면 **sigma 다시 키우지 말고** num_gaussians(16→24)나 offset를 풀 것. car IoU/precision 회귀 여부 확인. `efficientocf_config.py` 전역 기본값은 미변경(이 config들이 명시적으로 override하므로 무영향).

## 2026-06-23 15:00:27 KST — eval occupancy를 학습과 정렬: mixture scene splat 신설 (noreg/guide 둘 다)

**핵심 내용**: 학습/매칭/시각화는 16개 가우시안 mixture(`forward_gaussian_mixture_grouped`)를 쓰는데, **eval(`simple_test`)만 쿼리당 16개를 moment-2로 요약한 단일 ellipsoid 1개**(`voxelizer.forward`→`_forward_gaussian`)로 찍고 있었음 → 학습 shape(특히 **yaw/회전**)가 eval에서 사라지고, 빈 공간 over-cover. (GUIDE는 학습=추론 모두 K개 union splat.) eval도 mixture를 그대로 splat하도록 정렬.

**왜 grouped를 eval에 그냥 못 쓰나**: `forward_gaussian_mixture_grouped`는 per-query 텐서 `[T,K,1,D,H,W]`를 할당 → eval 풀해상도(512×512×40)에서 K(선택 쿼리)배만큼 곱해져 수 GB~OOM. (그래서 기존 eval이 단일 경로를 썼던 것.)

**해결 — `forward_gaussian_mixture_scene` 신설** (`voxelizer.py`):
- per-query mixture를 **하나의 scene 볼륨 `[T,1,D,H,W]`에 누적**(쿼리별 저장 X → 풀해상도 메모리 안전).
- 쿼리 내부 G개: `self.gaussian_combine_mode`(poisson/union)로 합성. 쿼리 간: **max(occupancy union)**.
- yaw·offset·sigma 전부 사용(회전 살림).

**eval 경로 정렬** (`efficientocf.py` `simple_test`):
- 플래그 `query_eval_occ_use_mixture=True`이고 mixture 텐서가 유효할 때만 mixture 경로(아니면 기존 단일 ellipsoid로 안전 fallback; 프레임/쿼리 수 mismatch도 fallback).
- 헬퍼 `_eval_select_mixture_params`: present mixture에서 **프레임-불변 per-gaussian offset**(= mix_c[present] − center[present]) + sigma/yaw/weight 추출.
- 헬퍼 `_eval_mixture_scene_occ`: 그 offset을 present 프레임/trajectory 미래 프레임 중심에 다시 얹어 `forward_gaussian_mixture_scene` 호출. present는 `mix_c[present]`와 정확히 일치, 미래는 trajectory 중심 기반.
- present(pred_bev/정렬용)·미래(pred_txyz/metric용) 둘 다 mixture로. 출력 shape `[T,1,D,H,W]`는 단일 경로와 동일해 다운스트림 무변경.

**config**: `efficientocf_config.py`에 기본값 `query_eval_occ_use_mixture: False`(기존 config 동작 무변경) + apply. `_noreg`/`_guide` 둘 다 `True`로 설정 → 각각 poisson/union으로 학습=eval 정렬.

**검증**: 격리 torch 테스트로 (a) scene 출력 `[T,1,D,H,W]`, p∈[0,1], (b) **yaw=0→x로 길고 yaw=90→y로 길게** 회전 반영, (c) union peak(0.90)>poisson(0.74), (d) Q=60도 per-query 폭발 없이 동작 확인. py_compile 5파일 통과.
**주의**: scene 메서드는 per-query Python 루프(eval 전용, offline이라 OK). 학습 경로(grouped/loss/matcher)는 미변경.

## 2026-06-23 14:15:21 KST — GUIDE union + sigmoid opacity(추천2): voxelizer 곱-합성 분기 신설 (test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide)

**핵심 내용**: 추천1(합 공식 유지, weight 자유화)과 별개로, occupancy 합성을 GUIDE식 union으로 교체하는 경로를 신설. 합(Poisson) `p=1-exp(-Σ w·G)`는 활성 가우시안 하나의 중심 peak가 `1-exp(-w)`로 상한(w=1→0.63)인데, 곱(union) `p=1-Π(1-α·G)`는 중심에서 `p≈α`(α=1이면 정확히 1)에 자동 도달. weight를 mixture 지분(softmax/softplus)이 아니라 **α∈[0,1] opacity(sigmoid)** 로 재정의.

**매핑 결과(사전 조사)**: mixture occupancy를 만드는 가중-합성은 `voxelizer.py:forward_gaussian_mixture_grouped` **단일 메서드**뿐이고, 이를 (1) 학습 BCE loss(`utils_loss.py:2455`), (2) 매칭 cost(`utils_matcher.py:123`), (3) 시각화(`utils_visualization.py:1376`)가 공유. → 메서드 1곳 + voxelizer 인스턴스 2개에만 손대면 학습·매칭·시각화가 **일관되게** union으로 전환됨. (query_head의 coverage/hit loss 5052/5204는 unweighted 보조항이라 의도적으로 미변경. `_convert_occ_output`는 trilinear 경로라 무관.)

**코드 변경**:
- `voxelizer.py` `SoftVoxelizerOneAdd.__init__`: `gaussian_combine_mode="poisson"` 인자 추가(+검증). `forward_gaussian_mixture_grouped` 561행: `wg=w·gauss` 공유 후 `union`이면 `p=1-Π(1-wg).clamp(1e-6,1)`, 아니면 기존 `p=1-exp(-Σ wg)` 분기. factor clamp로 p∈[0,1] 수치 안전(비-sigmoid weight도 안전 캡).
- `query_head.py`: weight_mode에 `'sigmoid'` 추가 — validation(611), 분기(1807, `weights_qg=sigmoid(logits)`, surrogate는 sigma moment용 L1-norm), bias init 조건을 `softplus`→`(softplus, sigmoid)`로 확장(bias 0 → α start 0.5).
- `efficientocf_config.py`: 기본값 `query_multi_gaussian_occ_combine_mode="poisson"`(MODEL_CFG_DEFAULTS, `_merge_cfg`로 기존 config 전부 무변경) + apply에 `self.…` 설정.
- `efficientocf.py`: voxelizer 2개(viz `self.voxelizer`:220, 학습 `self.matched_gmo_voxelizer`:232) 모두 `gaussian_combine_mode=self.query_multi_gaussian_occ_combine_mode` 전달 → 두 인스턴스 일관.

**config 변경** (`test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide.py`):
- `query_multi_gaussian_weight_mode`: softplus → **sigmoid**
- `query_multi_gaussian_occ_combine_mode`: (default poisson) → **union**
- `query_multi_gaussian_softplus_bias_init`: -2.0 → **0.0** (sigmoid bias도 이 인자 공유; α start 0.5)
- `query_multi_gaussian_weight_reg_loss_weight`: 1e-3 → **0.0** (union엔 sum 제약 무의미)
- `debug_query_mixture3d_vis_occ_threshold`: 0.2 → **0.5** (peak 회복)
- shape(sigma/offset/truncate)는 추천1과 동일하게 고정.

**검증**: 격리 torch 테스트로 (a) shape 일치 [Kc,nx,ny,nz], (b) union+sigmoid가 활성 중심에서 0.98, α=1·G=1 단일에서 정확히 1.0, (c) poisson+softmax는 낮게 깔림, (d) 모든 p∈[0,1] 확인. py_compile 5파일 통과.
**주의**: union은 반드시 sigmoid와 함께(α∈[0,1]). default(poisson)는 기존 동작 그대로라 다른 config 영향 없음.

## 2026-06-23 14:00:59 KST — mixture weight 자유화(추천1): softplus weight_reg off + bias 0 (test_traj_mcls_occ_jhh_traj0_past_bg_noli_noreg)

**핵심 내용**: occupancy가 작고 흐린 타원으로 collapse하는 원인을 weight 설계에서 진단. voxelizer는 `p = 1 - exp(-Σ w·G)` (Poisson union, `voxelizer.py:561-562`)인데, softplus weight를 `weight_reg`가 `target_sum=1`로 묶어 16개 가우시안이 개당 평균 ~0.06을 나눠 가짐. 그 결과 한 컴포넌트 중심 peak가 `1-exp(-0.06)≈0.06`까지 눌려 다수가 겹치는 코어만 threshold를 통과(=작은 타원). **합/곱 공식 비교**: 합(Poisson)은 한 컴포넌트 peak = `1-exp(-w)` (w=1→0.63), 곱(GUIDE union `1-Π(1-G)`)은 = 1. 1차로 합 공식을 유지한 채 weight 예산 제약만 제거.

**변경** (`projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_noli_noreg.py`):
- `query_multi_gaussian_weight_reg_loss_weight`: 1e-3 → **0.0** (sum→1 제약 제거, weight를 occupancy loss가 직접 결정. target_sum은 loss=0이라 무의미)
- `query_multi_gaussian_softplus_bias_init`: -2.0 → **0.0** (init weight softplus(-2)=0.13 → softplus(0)=0.69, 시작 peak 정상화)
- `debug_query_mixture3d_vis_occ_threshold`: 0.2 → **0.5** (viz 전용. 0.2는 낮은 peak 보정용이었음 → peak 회복 후 faint tail까지 보여 over-cover로 오독)
- **shape(sigma_max/offset_max/truncate/sigma_reg)는 의도적 고정**: weight 효과 격리(ablation) + 모델이 σ 재적응. mixture voxelizer truncate=`gaussian_truncate_sigma`=3.0(넉넉, 강한 가우시안 d_eff~1.7-2.5σ 안 잘림). (`query_attn_cam_gaussian_truncate_sigma=1.777`은 attn-cam 경로라 mixture와 무관)

**기대 효과**: 한 컴포넌트 중심 peak 0.06→0.63, 코어 0.9+. 객체 몸통 전체가 점유로 채워짐.
**주의/모니터**: (1) over-coverage(뭉뚝 블롭) 나오면 weight 되돌리지 말고 `sigma_max` 3.5→~2.0 / `sigma_reg` 0.001→~0.005로 조일 것. (2) weight가 여전히 학습되므로 소수 가우시안 몰빵(active-set collapse)은 구조적으로 안 막힘 → `dbg_query_match_spread_ratio`와 활성 가우시안 수 모니터, 관측되면 weight entropy/load-balance reg 추가. (3) 추천2(GUIDE 곱 union + sigmoid opacity)는 voxelizer/head 코드 분기 필요 → `_guide` config에서 별도 진행.

## 2026-06-23 — shape(multi-gaussian) 제한 공격적 완화 (test_traj_mcls_occ_jhh_traj0_past_bg_head_noli)

**핵심 내용**: center가 안정화되어 shape 학습 단계로 전환. 그동안 center-first 목적으로 조여뒀던 multi-gaussian offset/sigma 제한을 GT 박스 크기 분포에 맞춰 공격적으로 해제. 기존 `offset_max=(3,3,0.7)`, `sigma_max=(1,1,1)`은 car 반길이(~2.6m)밖에 못 덮어 bus/trailer(반길이 6.6~9.5m)를 구조적으로 못 맞추던 문제 해결.

**근거 데이터** (train infos pkl, PC range 내 fg 박스 긴축 길이 percentile):
- bus L p50/p99 = 12.0/14.5m, trailer 13.1/18.9m, truck 6.0/13.1m, construction 6.1/16.0m, 폭 p99 construction W~6.3m
- offset_max는 객체 반길이를 덮어야 함(offset은 world 축별 독립) → trailer p99 반길이 ~9.45m 커버 위해 10 선택

**변경** (`projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_noli.py` model_cfg):
- `query_multi_gaussian_offset_max_m`: (3.0,3.0,0.7) → **(10.0,10.0,2.0)**
- `query_multi_gaussian_sigma_max_m`: (1.0,1.0,1.0) → **(3.5,3.5,1.5)**
- `query_multi_gaussian_sigma_reg_loss_weight`: 0.01 → **0.001** (동반 필수; 0.01은 sigma를 1m로 끌어내려 sigma_max 확대를 무효화)
- `sigma_min`(0.15,0.15,0.15), `num_gaussians`(16), `weight_mode`(softplus)는 유지

**주의**: binary fg라 car~trailer가 이 한 세트를 공유 → 학습 후 **car class IoU/precision 모니터** 필요. 소형 객체 over-coverage 시 sigma_max를 3.0으로 하향. weight_reg(softplus default 1e-3)는 구조적 제약이라 미변경.

## 2026-06-23 10:36:12 KST — query mixture 3D 시각화에 score gate + GMM occ 임계값 행 추가

**핵심 내용**: `maybe_save_query_mixture_3d_vis`를 2행 6패널로 확장. 공통 score 임계값(기본 0.75)으로 query를 1차 게이트한 뒤, 상단 3패널은 기존 1σ 등고면, 하단 3패널은 mixture density를 모델 voxelizer로 복셀화해 `p=1-exp(-Σ w_g G_g) >= occ_thr`(기본 0.2)인 복셀을 실제 occupancy처럼 렌더.

**변경**:
- `projects/occ_plugin/occupancy/detectors/utils_visualization.py`: `maybe_save_query_mixture_3d_vis`에 (1) `sel_score >= score_thr` 공통 게이트, (2) `self.voxelizer.forward_gaussian_mixture_grouped`를 query당 K=1로 호출(메모리 절약)해 GMM occ 복셀을 월드좌표로 변환·클래스색 부여, (3) 2×3 subplot(상=1σ, 하=gmm occ≥thr) 렌더 추가. 헬퍼 `_finalize`로 축/뷰 설정 공유
- `projects/occ_plugin/occupancy/detectors/efficientocf_config.py`: `debug_query_mixture3d_vis_score_threshold`(0.75), `..._occ_threshold`(0.2), `..._occ_max_voxels_per_query`(4000)를 `VISUALIZATION_CFG_DEFAULTS`+`apply_visualization_cfg`에 추가
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`: 위 3개 값 명시(0.75 / 0.2 / 4000)

**검증**:
- `python -m py_compile` 통과
- synthetic mixture로 `forward_gaussian_mixture_grouped` 직접 테스트: 출력 `[1,1,1,40,512,512]`, `p=occ[0,0,0]`은 `[z,y,x]` 레이아웃, 좌표 역변환이 입력 center(~10,-4,0.5) 복원 확인. softmax weight 기준 max p≈0.40이라 0.2에서 복셀 생성·0.5는 공백 → 0.2 임계값 적절성 확인

## 2026-06-23 10:17:34 KST — query mixture 3D 시각화 이식 (ksh_shape 레포 → 본 레포)

**핵심 내용**: 옆 레포 `Autonomous_Driving_26_ksh_shape`의 `ksh_shape` 브랜치(커밋 `dc828e1`)에 있던 학습용 query mixture 3D 시각화(`maybe_save_query_mixture_3d_vis`)를 본 레포로 이식. 3-패널 matplotlib 3D(oblique z×2.8 / oblique rear / low side true-z)로 GT instance 복셀 + GT center + 선별 query + mixture 16성분 점/상위 6성분 yaw 반영 1σ wireframe ellipsoid를 렌더.

**변경**:
- `projects/occ_plugin/occupancy/detectors/utils_visualization.py`: `EfficientOCFVisualizationMixin`에 `maybe_save_query_mixture_3d_vis`와 `_QUERY_CLS_NAMES_8` 추가. 기존 `_build_query_visualization_bundle`의 `selected_mixture_centers/sigmas/yaw/weights_tqg3` 등 bundle 키를 그대로 사용(본 레포에 이미 존재)
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: forward_train의 `maybe_save_query_debug_vis` 호출 직후 `maybe_save_query_mixture_3d_vis` 호출 추가 (`gt_instance_occ3d_txyz_query_vis`, `gt_inst_center_world_full_tn3` 등 기존 변수 재사용)
- `projects/occ_plugin/occupancy/detectors/efficientocf_config.py`: `debug_query_mixture3d_vis_every`는 `DEBUG_CFG_DEFAULTS`+`apply_debug_cfg`에, `debug_query_mixture3d_vis_dir/max_queries/max_gt_points`는 `VISUALIZATION_CFG_DEFAULTS`+`apply_visualization_cfg`에 추가
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`: `debug_cfg`에 `debug_query_mixture3d_vis_every=48`, `visualization_cfg`에 dir/max 플래그 추가하여 활성화

**검증**:
- `python -m py_compile` (수정 4개 파일) 통과
- bundle 키(`selected_mixture_*`)가 본 레포 `_build_query_visualization_bundle`에 이미 존재함을 확인 → 데이터 경로 호환

## 2026-06-22 18:48:41 KST — head_gau config에 Gaussian-soft GT GMO target 적용

**핵심 내용**: `test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`에서 matched GMO loss target을 hard binary 대신 inside=1, outside=Gaussian halo인 soft GT로 사용할 수 있게 구현하고 해당 config에 활성화.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf_config.py`: `query_gmo_soft_gt_enabled`, `query_gmo_soft_gt_sigma_vox`, `query_gmo_soft_gt_truncate_sigma` 기본값 추가
- `projects/occ_plugin/occupancy/detectors/utils_loss.py`: low-res matched GT occupancy에 3D dilation shell 기반 Gaussian-soft halo를 적용하는 `_maybe_apply_query_gmo_soft_gt` 추가
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`: soft GT 활성화, `(64,64,20)` low-res 기준 `sigma=2.5`, `truncate=3.0`, GMO dice 비활성화
- `projects/configs/baselines/test.py`: soft GT 옵션을 명시적으로 비활성화해 기본 실험 동작 유지
- `NOTES.md`: soft GT의 low-res voxel 단위 및 dilation-shell 근사 거리 주의사항 기록

**검증**:
- `python -m py_compile projects/occ_plugin/occupancy/detectors/efficientocf_config.py projects/occ_plugin/occupancy/detectors/utils_loss.py projects/configs/baselines/test.py projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py` 통과
- direct-file import synthetic tensor에서 center=1, 거리 증가에 따라 soft target 감소 확인

## 2026-06-22 18:43:15 KST — Gaussian-soft GT halo 3D 시각화 추가

**핵심 내용**: soft GT의 의미를 명확히 보기 위해 GT boundary와 soft target 등가면을 3D로 비교하는 시각화를 추가.

**변경**:
- `gt_gaussian/visualize_soft_gt_halo_3d.py`: GT boundary, `soft>=0.5`, `soft>=0.2` 등가면을 sigma별 3D scatter surface로 렌더
- `gt_gaussian/soft_halo_3d_frame3_inst14_sigmas_1p5_2p5_4p0.png`: sigma 1.5/2.5/4.0 voxel의 3D halo 퍼짐 비교
- `PROJECT_STRUCTURE.md`: 신규 3D soft halo 시각화 스크립트 및 산출물 설명 추가

**검증**: `python gt_gaussian/visualize_soft_gt_halo_3d.py` 실행 성공, PNG 렌더 확인.

## 2026-06-22 18:40:38 KST — Gaussian-soft GT halo sigma별 시각화 추가

**핵심 내용**: hard GT 내부는 1로 유지하고 외곽만 Gaussian distance-decay로 퍼뜨리는 soft target을 sigma별로 비교하는 시각화를 `gt_gaussian/`에 추가.

**변경**:
- `gt_gaussian/visualize_soft_gt_halo.py`: 실제 `segmentation_instance3d` 객체 mask에서 distance transform 기반 soft GT 생성 및 BEV/Z-slice 비교 PNG 저장
- `gt_gaussian/soft_halo_frame3_inst14_sigmas_0p8_1p5_2p5_4p0.png`: sigma 0.8/1.5/2.5/4.0 voxel의 퍼짐 정도 비교
- `gt_gaussian/soft_halo_frame3_inst14_sigmas_0p8_1p5_2p5_4p0.npz`: sigma별 soft target volume과 원 occupancy 저장
- `PROJECT_STRUCTURE.md`: 신규 soft halo 시각화 스크립트 및 산출물 설명 추가

**검증**: `python gt_gaussian/visualize_soft_gt_halo.py` 실행 성공, PNG 렌더 확인.

## 2026-06-22 18:15:44 KST — GT instance 3D의 GMM shape 데모 생성

**핵심 내용**: 실제 `segmentation_instance3d` GT 객체 하나를 K=16 diagonal Gaussian mixture로 피팅해 shape가 어떻게 보이는지 확인하는 데모를 `gt_gaussian/`에 추가.

**변경**:
- `gt_gaussian/visualize_gt_gmm.py`: GT instance voxel points를 로드해 sklearn `GaussianMixture(covariance_type="diag")`로 피팅하고 3D/BEV/density PNG와 파라미터 NPZ 저장
- `gt_gaussian/sample_frame3_inst14_gmm16.png`: frame 3, instance 14, 26752 voxels에 대한 K=16 GMM 시각화
- `gt_gaussian/sample_frame3_inst14_gmm16.npz`: fitted `means_xyz`, `sigmas_xyz`, `weights`, 원 GT `points_xyz` 저장
- `PROJECT_STRUCTURE.md`: 신규 `gt_gaussian/` 폴더 및 산출물 설명 추가

**검증**: `python gt_gaussian/visualize_gt_gmm.py` 실행 성공, PNG 렌더 확인.

## 2026-06-22 16:22:03 KST — Eval bbox GT source 및 pred weighting을 jhh 방식에 맞춤

**핵심 내용**: current eval의 남은 차이였던 bbox GT source와 selected query voxelizer weighting을 `efficientocf_jhh.py` 방식에 맞춤.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: eval `forward_test`/`simple_test`가 `segmentation`, `segmentation_cls_instance3d`를 명시적으로 받도록 확장
- bbox 보정 GT source 우선순위 변경: `segmentation_cls_instance3d` 우선, 없으면 `segmentation_instance3d`, `segmentation`, `gt_occ_inst` fallback
- `segmentation_cls_instance3d` 사용 시 jhh처럼 class channel을 bbox occupancy로 쓰고 pedestrian id 7은 제거
- selected query 렌더링 시 `selected_score_q`를 voxelizer `weights`로 전달해 jhh처럼 confidence가 occupancy intensity에 반영되도록 변경

**검증**: `python -m py_compile projects/occ_plugin/occupancy/detectors/efficientocf.py` 통과.

## 2026-06-22 16:10:19 KST — Eval metric을 present 단일 프레임에서 future tail로 변경

**핵심 내용**: current `efficientocf.py` eval metric을 `efficientocf_jhh.py`처럼 현재 프레임 1장 기준이 아니라 미래 horizon tail 전체 기준으로 누적 계산.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: 선택 query는 기존 actual inference scoring을 유지하되, metric용 prediction은 trajectory geometry(`centers_world_traj`, `sigmas_world_traj`)를 voxelizer로 렌더해 future tail `[T,X,Y,Z]`로 생성
- `IOU_movable objects`: future tail pred BEV와 `segmentation_bev` future tail을 누적해 2D confusion 계산
- `IOU_3d`: future tail pred 3D와 nusocc `gt_occ` future tail foreground로 계산
- `Recall_3d`: future tail nusocc 3D 기준 TP/FP/FN에 `gt_occ_inst` 기반 3D bbox AABB의 `bbox_FP`를 보정식으로 적용
- 3D metric은 jhh 방식에 맞춰 `gt_occ == 255`와 pedestrian class id 7을 valid mask에서 제외

**주의**: eval visualization PNG는 디버그 편의를 위해 기존처럼 present-frame query canvas를 저장하며, metric 계산만 future tail 기준으로 변경.

## 2026-06-22 15:13:45 KST — global_idx eval dataloader cat crash 수정

**핵심 내용**: eval 시각화 파일명을 위해 추가한 `global_idx`가 instance loader의 `torch.cat` 대상에 들어가 dataloader가 죽는 문제 수정.

**변경**:
- `projects/occ_plugin/datasets/pipelines/loading_instance.py`: cache-hit/cache-regenerate 양쪽 후처리 skip list에 `global_idx` 추가
- `global_idx`는 tensor sequence가 아니라 meta 값이므로 `torch.cat`하지 않고 뒤 pipeline/Collect3D meta로 전달

**검증**: 관련 dataset/eval 파일 `py_compile` 통과.

## 2026-06-22 15:07:52 KST — Recall_3d를 nusocc 3D + bbox 3D 보정식으로 변경

**핵심 내용**: `Recall_3d`를 기존 보정식 `((TP + bbox_FP) / (TP + FN + FP - bbox_FP))` 그대로 쓰되, TP/FP/FN은 nusocc `gt_occ`의 실제 3D foreground 기준으로 계산하고 `bbox_FP`만 bbox instance 3D AABB volume 기준으로 계산.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: eval `forward_test`/`simple_test`에서 `gt_occ`를 받아 3D metric에 사용
- `IOU_3d`: bbox instance GT가 아니라 nusocc `gt_occ` 3D foreground와 pred 3D voxel의 IoU로 계산
- `Recall_3d`: nusocc 3D 기준 TP/FP/FN에 bbox 3D AABB 내부 FP만 `bbox_FP`로 더하는 legacy 보정식 적용
- bbox 보정용 GT는 `gt_occ_inst`에서 instance별 XYZ axis-aligned bbox volume을 만들어 사용하며 height-map pseudo 3D는 사용하지 않음

**주의**: 이미 실행 중인 eval 프로세스에는 적용되지 않으며 새로 시작한 eval부터 동작.

## 2026-06-22 14:37:50 KST — Eval 중간 metric 터미널 출력 및 live log 저장

**핵심 내용**: 분산 eval 중간 metric을 48간격으로 터미널에 출력하고, 시각화 저장 폴더에 `eval_metrics_live.log`로 누적 저장.

**변경**:
- `projects/occ_plugin/occupancy/apis/test.py`: `EOCF_EVAL_METRIC_EVERY` 또는 fallback `EOCF_EVAL_VIS_EVERY` 간격마다 각 rank 누적 confusion/3D metric을 `dist.all_reduce`로 합산
- rank0에서 `[eval][N] ... (all ranks)` 형식으로 flush 출력
- `EOCF_EVAL_VIS_DIR`가 있으면 같은 폴더에 `eval_metrics_live.log` append 저장

**주의**: 이미 실행 중인 eval 프로세스에는 적용되지 않으며 새로 시작한 eval부터 동작.

## 2026-06-22 13:41:30 KST — Eval 시각화 파일명 global index 기반 변경

**핵심 내용**: 8GPU eval 시각화 파일명을 rank-offset(`iter_1000000`) 방식 대신 dataset global index 기반(`sample_000640`)으로 저장.

**변경**:
- `efficientocf_dataset.py`, `efficientocf_lyft_dataset.py`: sequence sample의 원본 dataset index를 `global_idx`로 pipeline results에 추가
- `tools/test.py`: eval/test pipeline의 `Collect3D.meta_keys`에 `global_idx`를 런타임으로 자동 추가
- `efficientocf.py`: eval vis 저장 시 `img_metas.global_idx`가 있으면 해당 값을 step으로 사용하고 filename prefix를 `sample`로 설정
- `query_head.py`: query debug 저장 파일 prefix를 bundle의 `_filename_prefix`로 받을 수 있게 변경 (`sample_000640_prob.png`, `sample_000640_query_vis.pt`)

**Fallback**: `global_idx`가 없으면 기존 rank-offset `iter_<rank*1000000+local>` 방식 유지.

## 2026-06-22 13:32:56 KST — Eval query 시각화 trajectory rows 제거

**핵심 내용**: eval 시각화에서 사용하지 않는 하단 trajectory row 2개(refined/base traj)를 생략해 캔버스를 4-row로 축소.

**변경**:
- `efficientocf.py`: eval vis 호출 시 query bundle에 `_skip_traj_rows=True` 플래그 추가
- `query_head.py`: 해당 플래그가 있으면 row5/refined traj 및 row6/base traj 렌더링·라벨·border·arrow drawing을 생략
- 학습 query debug vis는 기존 row 구성을 유지

## 2026-06-22 13:07:37 KST — Eval 시각화 기본 저장 경로 config 이름 기반 변경

**핵심 내용**: eval 시각화(`EOCF_EVAL_VIS=1`)의 기본 저장 위치를 실행 config 파일명 기준으로 자동 분리.

**변경**:
- `tools/test.py`: `EOCF_EVAL_VIS_DIR`가 명시되지 않은 경우 `./work_dirs/eval/<config_stem>/<timestamp>`를 기본값으로 설정하고 save dir 출력
- `tools/dist_test.sh`: 분산 rank별 저장 경로가 갈라지지 않도록 실행 시작 시 `EOCF_EVAL_TIMESTAMP`를 한 번 생성해 모든 rank에 상속
- 기존처럼 `EOCF_EVAL_VIS_DIR`를 직접 지정하면 해당 경로를 우선 사용

## 2026-06-21 KST — eval 시각화 추가(학습과 동일 캔버스, 전 랭크) + 15ep OCC 전체 eval 배치

**핵심 내용**: `simple_test`(eval 경로)에 학습과 **동일한 query debug 시각화**(`maybe_save_query_debug_vis`, 5-row prob.png)를 env-gated로 붙임. 8-GPU 분산 eval에서 **전 랭크가 저장**하도록 우회 플래그 추가. 15에포크 완주한 OCC 파생 6개를 최신순·8GPU·순차로 eval하는 배치 스크립트 작성.

**변경**:
- `efficientocf.py`: `EfficientOCF._maybe_save_eval_query_vis()` 신규 — `simple_test`가 이미 만든 `pred_occ`(선택 query 렌더)·`gt_inst`·`query_vis_bundle`·`cls_scores`를 학습용 `query_head.maybe_save_query_debug_vis`에 그대로 넘겨 동일 이미지를 생성. `simple_test` 본문에 `_pred_occ_vis` 캡처 + 1줄 호출 추가. **metrics엔 영향 0**(렌더만), 실패해도 eval 안 깨지게 try/except.
  - env: `EOCF_EVAL_VIS=1`(켜기, 기본 off), `EOCF_EVAL_VIS_DIR`(출력 dir), `EOCF_EVAL_VIS_EVERY`(샘플 게이트, 기본 50). 파일명 `iter_{rank*1e6+local}_prob.png`로 랭크별 유니크.
- `query_head.py`: `_eval_vis_enabled_this_rank()` 신규 + `_maybe_save_prob_grid_vis`의 `if not self._is_main_process(): return`를 이걸로 교체. `self._eval_vis_all_ranks=True`(eval 헬퍼가 set)면 rank0 외 랭크도 저장 → 분산 shard 전체 커버. 학습은 플래그 미설정이라 **기존 rank0-only 동작 유지**.
- `run_eval_occ_all.sh` 신규: 15ep OCC 6개(past_bg, res34, past, traj0, jhh, occ) 최신순 8-GPU 순차 eval. `EOCF_EVAL_ALIGN="2,1,0,0"`·occ thr 0.5(스모크에서 pred≈gt count 확인, placement 문제라 thr 무관). 결과/viz를 `work_dirs/<run>/eval/{eval_metrics.log, vis/}`에 저장.

**검증(스모크, 8GPU)**: past_bg 48샘플 eval 정상(exit0, ~1.1 task/s = 2GPU의 4배), viz ON에도 metrics 동일(movable IoU2d 0.083). 전 랭크 저장 확인(24샘플→24장, rank0~7 균등, skip 0). 이미지는 학습과 동일한 GT/candidates/selected/matched/refined-traj 5-row + legend.

## 2026-06-20 KST — QueryDepthHead 용량 ↑ (past_bg_head) + depth 성능 dbg

**핵심 내용**: query별 depth bin 분류기(`QueryDepthHead`)를 용량 파라미터화하고, depth 예측 품질을 보는 dbg 추가. `past_bg` 위에 head만 키운 변종 config 생성.

**동기**: depth는 죽은 LSS DepthNet이 아니라 **QueryDepthHead**(query_head.py:120, 2-layer MLP embed→64bin)가 담당. 이 depth로 soft-argmax expected depth를 구해 query를 3D center로 **lift**(efficientocf.py:1076 `_build_query_attn_soft_lift_pack` → :1086 `apply_lifted_centers_to_outputs`)하므로 depth 품질이 곧 lift center 정확도. 2-layer는 64-way 분류엔 작아 underfit 의심 → 용량↑로 depth 분포를 날카롭게.

**변경**:
- `query_head.py`: `QueryDepthHead.__init__`에 `num_layers/hidden_mult` 추가(기본 2/1 = 기존과 동일). `QueryHead.__init__`에 `query_depth_head_num_layers/hidden_mult` 추가 + QueryDepthHead 빌드(:697)에 전달.
- `efficientocf.py`: QueryHead 빌드(:181)에 두 인자 전달.
- `efficientocf_config.py`: `MODEL_CFG_DEFAULTS`에 `query_depth_head_num_layers=2`, `query_depth_head_hidden_mult=1` + apply 배선.
- 새 config `test_traj_mcls_occ_jhh_traj0_past_bg_head.py` = past_bg + `num_layers=3, hidden_mult=2` (96→192→192→64). past_bg 대비 head만 다른 깨끗한 A/B.

**depth 성능 dbg** (`utils_query_projection.py` `_compute_query_depth_loss_from_match`, CE 직후, matched+valid pair):
- `dbg/query_depth_top1_acc`, `within1_acc`: argmax bin이 GT bin과 일치 / ±1 이내 비율.
- `dbg/query_depth_bin_abs_err`: |argmax_bin − gt_bin| (bin 단위, 1bin≈0.875m).
- `dbg/query_depth_soft_bin_abs_err`: |soft-argmax(expected) bin − gt| (**lift가 실제 쓰는 값**).
- `dbg/query_depth_entropy`: depth 분포 엔트로피(낮을수록 날카로움).
- prefill(out dict)에 5키 추가 → 멀티GPU log_vars 키셋 일치. `with torch.no_grad()`로 grad 무영향.
- **strip allowlist 수정**(utils_loss.py:3682): `dbg_query_depth_` 접두사 허용 추가 — 안 그러면 잘림(기존 `dbg_query_depth_valid_count` 등도 이번에 같이 살아남).

**검증**: py_compile 6파일 통과. depth dbg 로직 합성 검증(sharp→top1=1/err=0/ent=0, flat→ent=ln8=2.08, off-by-one→within1=1/bin_abs=1). 기본값 2/1이라 다른 config 무영향(EfficientOCF_V1.1_1gpu.py 수정 불필요).

**비교법**: `past_bg`(head 기본) vs `past_bg_head`(3층/wider) → head 효과만 분리. depth dbg가 좋아지는지(top1↑/abs_err↓/entropy↓) + center_l2↓ 확인. 늘 그렇듯 IOU 기준점 먼저.

## 2026-06-19 KST — `test_traj_mcls_occ_jhh_traj0_res34.py` 백본 ResNet18 → ResNet34

**핵심 내용**: 해당 config의 `img_backbone`을 ResNet18에서 ResNet34로 교체.

**변경**:
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_res34.py`: `pretrained='torchvision://resnet18'` → `'torchvision://resnet34'`, `depth=18` → `depth=34`.

**범위 주의**: ResNet18/34 모두 `BasicBlock` 사용 → stage별 출력 채널 `[64, 128, 256, 512]` 동일하므로 `img_neck.in_channels` 무변경. ResNet50+ (`Bottleneck`, `[256,512,1024,2048]`)로 갈 때만 neck도 같이 수정 필요.

## 2026-06-19 KST — matched-pair GMO+center loss 과거3 제한 플래그 (`query_matched_loss_history_only`)

**핵심 내용**: matched-pair **GMO(occupancy BCE/dice)** 와 **center match** loss를 7프레임 전체가 아니라 **receptive-field(과거+현재) 프레임 `[0:time_receptive_field]`** 에만 걸도록 하는 플래그 추가. 미래4(traj로 민 프레임)는 **trajectory loss만** supervise.

**동기**: 미래 프레임 geometry는 present Gaussian을 traj offset으로 평행이동한 것(`_build_query_trajectory_geometry_from_present`)이고 sigma/yaw/weight는 present 복사 → (1) 미래 GMO는 shape 자유도 0이라 present sigma를 미래GT 쪽으로 끌어당겨 **present 모양 오염** 위험, (2) 미래 center loss는 `present center + traj offset` 학습이라 **traj loss와 중복**. 따라서 미래는 traj에 일임.

**변경**:
- `efficientocf_config.py`: `MODEL_CFG_DEFAULTS`에 `query_matched_loss_history_only`(기본 `False` = 기존 7프레임 동작 유지) 추가 + apply 배선.
- `efficientocf.py`: loss 섹션에 `_hist_slice` 헬퍼 추가 — 플래그 ON **및 `use_full7_temporal_sup` True**일 때만 `centers/sigmas/mixture_*/gt_instance_occ3d_txyz`를 `[:time_receptive_field]`로 슬라이스. `_compute_query_center_match_loss_from_match`, `_compute_matched_pair_gmo_losses` 두 호출에만 적용. center loss는 `t_match=min` 로직이 GT를 자동으로 history로 슬라이스, gmo는 `T=min(Tq,To)`라 centers+gt_occ 슬라이스로 history만 계산.
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past.py`: `query_matched_loss_history_only=True` 활성화.

**범위 주의**: matching(`_match_queries_to_gt_instances`)·recruit·center dbg는 그대로 7프레임 사용(매칭 cost는 이미 history 3프레임만 씀 → 영향 없음). traj/refine/cls/depth/attn 등 나머지 loss 무변경. 미래 occupancy IoU는 이제 `present shape + traj 위치`가 책임.

**검증**: py_compile 통과(3파일). `use_full7_temporal_sup=False` 경로(match 텐서가 present+future 5프레임)에서는 슬라이스 비활성 → 잘못된 history 해석 방지.

## 2026-06-18 KST — matcher cost 분석 dbg 2차 확장: fliprate / decisiveness / topk_colstd / range-coverage / spread / corr

**핵심 내용**: colstd+margin에 이어, 각 cost 항의 **실질적(인과) 영향**과 **공간 커버리지**를 측정하는 dbg 다수 추가. workflow로 in-scope 텐서 검증 후 추천 항목 전부 구현.

**추가 metric** (`projects/occ_plugin/occupancy/detectors/utils_loss.py`, match-cost dbg 블록):
- `dbg_query_match_cost_{name}_fliprate` (P0): term 제거 시 열별 argmin 승자가 바뀌는 비율 = **counterfactual 실질 영향**. cost=Σcontrib라 `cost-contrib`가 정확한 leave-one-out. 평평하면(flat) 0 → "평균 큰데 영향 0"을 직접 잡음.
- `dbg_query_match_center_err_{near,mid,far}_mean` + `_count` (P0): matched query→GT center err를 **ego-거리 버킷**으로 (R_max=`self.point_cloud_range` corner, 1/3·2/3 분할). 외곽 매칭 정확도 저하 진단. count로 빈 버킷(z) vs 진짜 0 구분.
- `dbg_query_match_cost_{name}_decisiveness_rate` (P1): matched 열에서 그 항이 **최대 margin(최다 타이브레이커)** 인 비율. 7항 합 ~1.
- `dbg_query_match_cost_{name}_topk_colstd_mean` (P1): 열별 **총비용 top-5 경쟁자 한정** contrib std = global colstd보다 결정영역 변별을 덜 희석.
- `dbg_query_match_far_gt_frac`, `_spread_q/_spread_g/_spread_ratio` (P2): far 버킷 비중 + matched query 공간 분산/GT 분산 비(중앙 콜랍스 지표).
- `dbg_query_match_cost_center_temporal_offset_corr` (P2): 두 거리계열 항 중복성(Pearson).

**SKIP**: argmin_share(fliprate에 포함), unmatched-GT coverage(Q≫N라 active GT 전부 1:1 매칭→측정 대상 없음).

**검증**: py_compile 통과. 합성 데이터로 6종 전부 로직 검증(spread 항=양수, flat 항=0; 버킷 stratify; corr=1.0). **prefill↔record 동기화**: 신규 키 전부 무조건 prefill→record는 덮어쓰기만이라 멀티GPU log_vars 키셋 일치 보장(빈 버킷/무경쟁 step은 z 유지).

**주의**: `self.point_cloud_range`는 mixin이 아니라 host detector(efficientocf.py:142) 속성 — runtime `self`로 접근. 새 dbg는 **새 run/재시작부터** 기록(돌고 있는 occ엔 미반영). config 인자 추가 없음.

**버그픽스(같은 날)**: traj0 검증 중 coverage/spread 키가 TB에 누락 발견. 원인 = `_aggregate_training_losses` 말미의 dbg strip allowlist(utils_loss.py:3674)가 `dbg_query_match_cost_` 접두사만 허용 → `dbg_query_match_center_err_*`/`far_gt_frac`/`spread_*`가 삭제됨(corr는 match_cost라 생존). allowlist를 `dbg_query_match_cost_` → `dbg_query_match_`로 넓혀 해결(전부 prefill돼 멀티GPU 안전). per-term fliprate/decisiveness/topk_colstd은 match_cost 접두사라 이미 정상 기록 중이었음.

## 2026-06-18 KST — matcher cost 변별력(discrimination) dbg 추가: colstd + margin

**핵심 내용**: Hungarian cost matrix에서 각 항(center/attn/cls/temporal_offset/...)의 **평균 비중(ratio)이 아니라 실제 변별력**을 측정하는 dbg 2종 추가. "평균은 크지만 쿼리마다 평평하면 매칭에 영향 없음" 문제를 진단하기 위함.

**주요 변경사항** (`projects/occ_plugin/occupancy/detectors/utils_loss.py`, match_cost dbg 블록만):
- `dbg_query_match_cost_{name}_colstd_mean`: GT 열(column)마다 그 항 contrib을 **후보 쿼리들 across로 std** → 후보 ≥2인 열 평균. = 변별력(작으면 평평→매칭 영향 0).
- `dbg_query_match_cost_{name}_margin_mean`: 매칭된 열에서 **runner-up(총비용 2등 쿼리) contrib − 선택 쿼리 contrib**. >0이면 그 항이 그 할당을 실제로 결정, ~0이면 무관(예: flat cls). runner-up은 항-독립이라 1회 계산 후 재사용.
- 7개 항 모두에 대해 prefill 루프와 record 지점 **양쪽에 키 추가**(멀티GPU log_vars assert 방지, NOTES:52 규약).

**검증**: py_compile 통과. 합성 cost matrix로 로직 검증 — spread 항(center) colstd>0/margin>0, flat 항(cls) colstd=0/margin=0으로 정확히 구분 확인.

**적용 범위/주의**: config 인자 추가 없음(순수 dbg). 돌고 있는 occ run은 모듈 이미 import돼 **재시작해야 반영**, occ_jhh/traj0 등 신규 run은 처음부터 기록. EfficientOCF_V1.1_1gpu.py/efficientocf_config.py 변경 불필요(신규 키 없음).

## 2026-06-18 KST — test_traj_mcls_occ_jhh.py: BEV dim 64→96 + DT loss off

**핵심 내용**: jhh occ 실험 config에서 BEV feature 채널을 64→96으로 키우고, query DT loss를 비활성화.

**주요 변경사항** (`projects/configs/baselines/test_traj_mcls_occ_jhh.py`만 수정):
- `bev_feat_dim = 64 → 96` (line 357). 단일 source-of-truth라 `numC_Trans`(358)·`query_embed_dim`(model_cfg 377)·`img_view_transformer numC_Trans`(507)로 자동 전파. 다른 키/소스 수정 불필요.
- model_cfg에 `use_query_dt_loss=False` 추가. `utils_loss.py:3340`의 `if self.use_query_dt_loss:` 브랜치가 통째로 skip → `loss_query_dt` 미생성. `load_occ_dt`는 True 유지(미사용 dead data, 무해).

**검증** (병렬 조사 + 적대적 검증, 신뢰도 high):
- BEV 채널 dim은 LSS view transformer→DepthNet context_conv→query transformer→query head까지 end-to-end **parametric** (채널 경로에 하드코딩 64 없음). attention `num_heads=4`(efficientocf.py:129), 96%4==0 → head 변경 불필요.
- 건드리면 안 되는 것: `img_neck out_channels=[128,128,128,128]`(497), `numC_input`(미설정, 512=4×128 SECONDFPN concat).
- DT는 weight 키가 없고 boolean toggle만 존재(efficientocf_config.py:13 기본 True, :216 wiring). loss_weight=0.1은 inline 하드코딩. weight=0 방식 불가, toggle만이 유일 제어.
- log_vars assert 무관(`loss_query_dt` prefill 없음; NOTES:52 경고는 `dbg_query_match_cost_*` 한정).

**주의**: BEV 채널 ~50%↑ → GPU 메모리 증가, OOM 모니터링 필요. efficientocf_config.py mirror 기본값(query_embed_dim=256, use_query_dt_loss=True)은 이 run의 model_cfg가 override하므로 수정 불필요(다른 config엔 영향 없음).

## 2026-06-16 KST — train 스크립트 USE_MPS=0 적용 (기존 MPS 데몬 재사용, 데몬 관리 안 함)

- 배경: `tools/dist_train.sh`의 `pgrep -x nvidia-cuda-mps-control` 데몬 감지 체크는 항상 실패함.
  커널이 comm 이름을 15자(`nvidia-cuda-mps`)로 자르는데 `-x`로 23자 풀네임을 찾기 때문.
  → `USE_MPS=1`(기본값)이면 매번 `nvidia-cuda-mps-control -d`로 새 데몬을 띄우려 시도.
- 대응: `train_total.sh`, `train_total_2.sh` env 블록에 `USE_MPS=0` 추가.
  - 스크립트가 MPS 데몬을 시작/관리하지 않음 (세션 종료 시 공유 데몬 영향 우려 제거).
  - 이미 떠 있는 데몬이 기본 파이프 `/tmp/nvidia-mps`에 있고 CUDA 클라이언트는 env 미설정 시
    해당 기본 경로로 자동 연결되므로, 기존 MPS 가속은 그대로 유지됨.

## 2026-06-15 18:16 KST — Gate + Recruit 구현 및 실험 설정

- `efficientocf_config.py`: Gate/Recruit 설정 기본값과 model attribute 바인딩 추가.
- `utils_matcher.py`: query-GT 최소 XY 거리 기반 8m hard gate, gate 이후 unmatched GT와 free query의 15m greedy recruit pairing 추가.
- `utils_loss.py`: recruit pair에 classification label 없이 center L1 pull loss와 DDP-safe debug key 추가.
- 미래 trajectory가 gate 판정에 섞이지 않도록 기존 temporal matching frame index를 거리 계산과 recruit loss에 동일하게 사용.
- `projects/configs/baselines/test_traj_mcls_gate.py`에 권장 설정 추가:
  `query_match_center_gate_radius_m=8.0`,
  `query_recruit_max_radius_m=15.0`,
  `query_recruit_loss_weight=0.1`.
- 검증: py_compile, config/model attribute 파싱, Gate off 회귀, hard gate/recruit pairing,
  recruit loss gradient, 미래 frame 제외 synthetic smoke test 통과.

## 2026-06-15 KST — MPS 멀티잡 실측 검증 + 런북 문서화

### 한 노드 8장에서 학습 2개 동시 실행 (MPS) 실험
- 실측(각 조건 40 iter 평균, test_traj + test_traj_mcls):
  - 1잡 단독: 8.2s/iter
  - 둘 다 non-MPS(겹침): 30s/iter (단독보다도 느림, 노드 처리량 0.066 iter/s)
  - 한쪽만 MPS: ~32s/iter (효과 0, 제로섬)
  - **둘 다 MPS: 9.0s/iter (3.3배 빠름, 노드 처리량 0.220 = 단독 1.79배)**
- 결론: MPS는 **두 잡 모두 클라이언트일 때만** 효과. 단독 GPU util 24~60%(launch-bound)라 빈 SM을 MPS가 채워 큰 이득.
- 신규 문서 `MPS_MULTIJOB_RUNBOOK.md` 추가 (복붙 절차 + 함정 정리). 이론은 기존 `GPU_MULTIJOB_NOTES.md`.

### MPS 상시적용
- `tools/dist_train.sh`: MPS 데몬 자동 시작 블록 추가(기본 ON, idempotent). 모든 학습 launch가 자동으로 MPS 클라이언트로 시작됨. `USE_MPS=0`으로 비활성 가능.
- 종료용 `stop_mps.sh` 추가 (quit→정리 순서 안전 처리).
- ⚠️ 트레이드오프: MPS는 컨텍스트 공유라 한 잡 치명오류 시 같은 GPU 다른 잡도 영향 가능(격리 약화). 단독 1잡은 이득 없음(무해).
- PORT/--work-dir 잡마다 다르게 주는 건 여전히 사용자 책임 (MPS가 안 챙김).

## 2026-06-15 KST — mode-cls moving 클래스 가중 (mcls A/B 실험)

### 배경
- epoch 14 분석: mode 분류기가 static으로 쏠려 moving을 11%만 예측 (GT 31%).
- 보정용으로 mode 분류 CE에서 moving(비정지) 클래스에 가중을 주는 신규 인자 추가.

### 신규 인자: `query_traj_mode_cls_moving_class_weight` (기본 1.0 = 무변화)
- `efficientocf_config.py`: DEFAULTS 등록 + 파싱.
- `efficientocf.py`: trajectory loss 호출에 전달.
- `utils_loss.py`: `_compute_query_trajectory_loss_from_match` 시그니처에 추가.
  static_gate **비활성** 경로(평범한 mode CE, L941 부근)에서 `F.cross_entropy(weight=...)`로 적용.
  weight 텐서 = 전 클래스 가중값, stationary 인덱스만 1.0으로 되돌림 (비정지 전부 가중).
  값이 1.0이거나 stationary_mode_idx가 없으면 weight=None(기존과 동일).

### config
- `test_traj_mcls.py`: `query_traj_mode_cls_moving_class_weight=4.0` 추가. 그 외 test_traj.py와 100% 동일 (A/B 비교).
- baseline `EfficientOCF_V1.1_1gpu.py`: traj-mode 키를 선언하지 않는 기존 패턴 유지 → DEFAULTS 1.0으로 동작 변화 없음.

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

---

## 2026-06-16 KST — EfficientOCF eval(test) 경로 신규 구현 + mcls vs gate IOU 측정

**배경**: 현재 detector `efficientocf.py`(class EfficientOCF)에는 `forward_train`만 있고 `forward_test`/`simple_test`가 없어, standard test(`tools/test.py`)가 base `BEVDepth.forward_test`로 떨어져 `img_inputs must be a list, got NoneType`로 크래시. 학습 중 eval(IOU/VPQ)이 한 번도 안 돈 것도 같은 이유. 옛 `efficientocf_ori.py`의 eval은 `pts_bbox_head/height_head/flow_head` 기반인데 현재 모델엔 그 헤드가 없음(query_head/voxelizer 구조).

**구현** (`projects/occ_plugin/occupancy/detectors/efficientocf.py`, forward_train 앞):
- `forward_test`/`simple_test` 추가. **실제 추론 scoring**(debug-vis 3행)을 그대로 재사용: `_build_query_visualization_bundle(...)` → `selected_query_idx_q` (fg argmax≠bg → score≥`debug_query_score_threshold`(0.5) → 거리 NMS(3m) → top-k). score 가중치는 config의 visualization_cfg(iou=0,cls=1,cam=0) → score=cls_prob.
- 선택된 query의 present-frame center를 `self.voxelizer`(단일 sigma)로 렌더 → Z축 max로 BEV collapse → `segmentation_bev`(dense BEV movable GT, 옛 eval과 동일 타깃)와 2x2 confusion(free/movable) → 기존 `EfficientOCFDataset.evaluate`(`cm_to_ious`)로 IOU.
- 정렬(present frame index, transpose, flip)은 좌표 규약 차이로 첫 샘플 auto-calibration 또는 env `EOCF_EVAL_ALIGN="frame,transpose,fH,fW"`로 고정. occupancy threshold는 env `EOCF_EVAL_OCC_THR`(기본 0.5).
- 헬퍼: `_binary_occ_cm`, `_apply_bev_align`, `_calibrate_eval_bev_align`. 진단 print(`[simple_test][diag]`) 포함.

**측정 결과** (val 100 samples, seed 0, 동일 정렬 `EOCF_EVAL_ALIGN=2,1,0,0`):
- **mcls ep9: IOU_movable = 0.074**, **gate ep9: IOU_movable = 0.060** → **mcls 우위(+23% rel)**.
- 주의: score≥0.5 임계로 장면당 1~3개 query만 선택(GT ~2000 vox/frame) → 절대 IOU 매우 낮음(under-predict). 이는 faithful한 실제-추론 동작. present-frame movable BEV IOU이며 legacy future-IOU/VPQ 아님. traj run은 ckpt 없어 미측정.
- 첫 자동정렬판(frame 미고정)에선 mcls가 frame3, gate가 frame2로 갈려 gate가 높게 나왔으나 **frame 고정 후 mcls 우위로 정정**됨.

**평가 실행**: `run_eval.sh CFG CKPT GPUS` 또는 `tools/test.py ... --launcher pytorch --eval bbox`(분산 런처 필수 — non-dist면 evaluate 미호출). 로그/CSV: `work_dirs/_analysis/`.

---

## 2026-06-17 KST — train 중 eval 완전 제거 (흔적 코드 포함)

**배경**: 사용자 요청 — train 도중 eval(validation)을 절대 하지 않음. 인자 비활성이 아니라 흔적 코드까지 제거.

**사실 확인**: 실제 train 경로(`tools/train.py` → `custom_train_model` → `custom_train_detector`)의 eval 훅 등록부는 이미 주석 처리 상태였음 → 사실상 train 중 eval은 이미 안 돌고 있었음. `Saving checkpoint` 직후 멈춘 듯 보인 건 eval이 아니라 체크포인트 저장 I/O + dataloader 워커 재기동.

**변경**:
- `projects/occ_plugin/occupancy/apis/mmdet_train.py`: 주석 처리돼 있던 eval 훅 등록 블록(val_dataset/val_dataloader/OccDistEvalHook) 완전 삭제. eval 전용 import 제거(`EvalHook`, `OccDistEvalHook/OccEvalHook`, `custom_build_dataset`, 미사용 `build_dataset/replace_ImageToTensor`). → 이제 어떤 config든 train 중 eval 훅이 등록될 수 없음(전역 보장).
- `projects/configs/baselines/test_traj_mcls_occ.py`: `evaluation = dict(...)` 블록 삭제, eval 전용 `val_config`/`data['val']` 삭제, 미사용 `import copy` 삭제. (`test_config`/`test_pipeline`은 tools/test.py용이라 유지)

**영향**: train 중 eval 없음(전역). test.py를 통한 별도 평가는 영향 없음. 다른 실험 config들(test_traj.py, EfficientOCF_V1.1_1gpu.py 등)에 남아있는 `evaluation` dict는 이제 dead code지만 별도 실험 소유라 미수정.
## 2026-07-22 10:00 KST — GMO covariance 장축 효과 오프라인 probe 추가

- `tools/dbg_probe/probe_gmo_axis.py` 추가: 모델의 실제 matched-pair low-resolution GMO 경로를 hook해 asset-union GT 대비 Gaussian center(equal/opacity-weighted)와 렌더링 BEV(0.5/0.75 threshold)의 주축 각도 오차 및 장단축비를 JSON으로 기록.
- 별도 매칭 재구현 없이 학습 시 사용되는 matching과 GT를 그대로 사용하며, 동일 sample/augmentation 비교를 위해 sample별 seed를 고정.
- `tools/dbg_probe/README.md`, `PROJECT_STRUCTURE.md`에 새 probe를 반영.

## 2026-07-22 11:40 KST — `_cov_dir` offset 장축 direction-only loss 추가

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_cov_dir.py`에 기존 covariance 원소 MSE 대신
  `[(Cxx-Cyy), 2*Cxy]`를 단위 방향으로 비교하는 `direction` shape-loss 모드를 적용했다.
- 실측상 불안정한 GT를 제외하도록 BEV 최소 10 voxel, GT 장단축비 최소 2.0을 사용하고,
  loss 범위(0~1)에 맞춰 weight를 0.02로 설정했다.
- 공용 기본 모드는 `covariance`, 최소 GT 장단축비는 1.0으로 두어 기존 no-cov/cov/cov_weight
  config의 동작을 유지했다. direction loss는 Gaussian center offset에만 gradient를 전달하며
  sigma/yaw/opacity 및 추론 경로는 변경하지 않는다.

## 2026-07-22 13:10 KST — GMO 장축 paired BEV 시각화 도구 추가

- `tools/dbg_probe/visualize_gmo_axis_pair.py` 추가: 동일 sample/augmentation/matched GT에서 두
  checkpoint의 asset BEV, 48 Gaussian centers, threshold 0.75 render, GT/예측 장축선을 나란히 표시한다.
- epoch1 probe 통계에 따라 개선/중립/악화 각 2건을 고정 선택해 성공 사례만 보는 편향을 방지했다.
### 2026-07-22 13:23 KST — paired axis visualization reproducibility fix

- `tools/dbg_probe/visualize_gmo_axis_pair.py` now advances the training dataset in the same sequential order as the source probe and immediately copies selected visualization data to CPU, preserving augmentation identity while avoiding retained GPU captures.
- Selected examples are resolved by the probe's stable matched-pair index; run-local remapped instance IDs are retained only as plot annotations.
- Default comparison cases use reproducible matched samples 5, 9, and 12 (two improved, two neutral, two degraded); sample 18 was excluded because the rerun augmentation produced no prepared matched pairs.
### 2026-07-22 14:25 KST — rendered-direction offset-only feasibility probe

- Added differentiable `render_direction` GMO shape supervision using the final soft rendered occupancy's XY covariance.
- Added diagnostic flags that freeze every parameter except `query_head.gaussian_offset_head` and expose only `loss_gmo_shape` to the runner.
- Extended shape-loss config validation to accept `render_direction` and explicitly apply both diagnostic flags.
- Added `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_render_dir_probe.py` and `train_render_dir_probe.sh`: load Asset-all epoch 15 weights, use fixed LR `1e-4`, and train one epoch without replaying trajectory warmup stages.
- Pinned the probe launcher to the `eo` Conda environment so distributed launch does not fall back to the base Python without PyTorch.
- Runtime incident: the probe reached iteration 13, then one rank received a batch with no valid shape pair and returned a detached zero loss; its backward failure tore down the shared NCCL job and also aborted the concurrently running `cov_dir` job. The probe was not relaunched.
- Fixed the zero-valid-pair DDP failure by connecting the zero shape loss to rendered mixture centers (and therefore the offset head). Updated the standalone probe recipe to 3 epochs with a short 200-iteration linear warmup followed by cosine decay; trajectory warmup remains removed.
### 2026-07-22 16:50 KST — FT 경로/이름 확정

- Fine-tuning config를 사용자가 지정한 `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_cov_dir_ft.py`로 통합하고, 초기 weight를 `/home/hwanhee/Autonomous_Driving_26_new_data/work_dirs/subset_attn_cover_pyr_aabb_dice3d_new/epoch_15_lss_only.pth`로 변경했다.
- 설정은 rendered-direction shape loss 단독, Gaussian offset head만 trainable, 3 epochs, LR `1e-4`, 200-iteration linear warmup 후 cosine decay이며 trajectory warmup hook은 제거했다.
- 실행 파일을 `train_ft.sh`로 교체하고 이전 `train_render_dir_probe.sh` 및 중복 probe config를 제거했다.
### 2026-07-22 16:55 KST — FT 전체 파라미터 unfreeze

- 사용자 요청에 따라 `_cov_dir_ft.py`의 `query_offset_only_finetune`과 `query_shape_only_finetune`을 모두 비활성화했다. 전체 모델과 기존 Asset focal/Dice/기타 loss가 다시 학습되며, rendered-direction 종류와 현재 shape weight `1.0`은 후속 결정 전까지 변경하지 않았다.
### 2026-07-22 17:00 KST — epoch-15 full-model fine-tuning schedule 정렬

- `_cov_dir_ft.py`의 전체 LR warmup을 제거했다. 3-epoch cosine schedule은 첫 iteration부터 적용된다.
- TrajectoryWarmupHook 없이 trajectory 설정을 기존 schedule의 마지막 stage로 고정했다: trajectory `0.5`, XY refine `0.1`, mode classification `0.1`, teacher-forcing GT ratio `0.0`.
- 전체 unfreeze에 맞춰 AdamW weight decay `0.01`과 image backbone LR multiplier `0.1`을 복구했다. 기본 LR은 `1e-4`로 유지했다.
### 2026-07-22 17:05 KST — FT 초기 checkpoint 변경

- `_cov_dir_ft.py`의 `load_from`을 `/home/hwanhee/Autonomous_Driving_26_new_data/work_dirs/full_attn_cover_pyr_aabb_dice3d_new/epoch_15_lss_only.pth`로 변경했다. 나머지 FT 설정과 현재 subset train capacity(4000)는 유지했다.
### 2026-07-22 17:10 KST — full epoch-15 안정 FT LR schedule

- `_cov_dir_ft.py`의 일반 LR을 `3e-5`로 낮추고 image backbone은 기존 multiplier `0.1`로 `3e-6` peak를 사용한다.
- 첫 200 iterations는 ratio `0.1`의 linear warmup(일반 `3e-6→3e-5`, backbone `3e-7→3e-6`)을 적용하고, 이후 cosine으로 3 epochs 종료 시 peak의 `0.1`까지 낮춘다. epoch-15 마지막 LR과 급격한 불연속을 줄이는 목적이다.
### 2026-07-22 17:15 KST — FT rendered-direction weight 확정

- `_cov_dir_ft.py`의 `query_gmo_shape_loss_weight`를 `1.0`에서 `0.1`로 확정했다. 기존 center `cov_dir`의 `0.02`보다 coefficient는 5배 강하지만 Asset focal `0.1`/Dice `0.5`와 병행 가능한 수준이다.

### 2026-07-23 14:25 KST — hard FT epoch 2 resume 연결

- `train_ft.sh`가 `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft_hard`의
  `epoch_2_lss_only.pth`에서 resume하도록 연결했다.
- checkpoint의 epoch/iteration, optimizer 및 LR scheduler 진행 상태를 복구하여
  warmup을 재실행하지 않고 epoch 3 학습을 이어간다.

### 2026-07-29 09:50 KST — eval NameError 수정

- `efficientocf.py` `extract_feat_query`에서 `EOCF_EVAL_TRAJ_REFINE` 환경변수를 읽을 때 `NameError: name 'os' is not defined`로 eval이 죽던 문제를 수정했다.
- 모듈 상단에 `import os`를 추가했다 (기존에는 일부 함수 내부에서만 local import 되어 있었음).

### 2026-07-29 10:20 KST — eval query_debug_vis 4행 GT를 AABB→asset-union으로 교체

- `efficientocf.py simple_test`의 `eval_cmp_pack`이 넘기던 `gt_bev_t`를 `bbox_bev_vis`(AABB BEV)에서 `asset_bev_vis`(asset-union BEV, `hist_for_iou_asset`/`iou_3d_asset` 채점에 쓰는 텐서 그대로)로 변경했다. align/eval_mode 슬라이스는 동일 경로.
- 더 이상 쓰이지 않는 `bbox_bev_vis` 변수를 제거했다.
- `query_head.py`의 4행 라벨을 `pred occ vs bbox_aabb` → `pred occ vs asset`로 수정했다.
- asset GT의 pedestrian(cls 7)은 `_eval_load_asset_gt`에서 이미 로드 시 제외되고 있어 별도 필터 추가는 없음.

### 2026-07-29 11:05 KST — ep20 연장 학습용 config 분리

- `projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_ep20.py` 추가 (기존 filter config 복사본).
- `max_epochs` 15 → 20. cosine은 `epoch/max_epochs`로 매 epoch 새로 계산되므로 ep15에서 resume하면 ep16 진행률 15/20=0.75 → LR 3.6e-6 → 4.4e-5로 warm restart (원래 ep12 수준). base LR 3e-4는 ckpt optimizer의 `initial_lr`로 보존됨.
- work_dir은 config 파일명 기준 자동 분리(`work_dirs/..._ep20/`). debug vis 4개 경로도 동일 폴더로 변경해 기존 run 산출물과 섞이지 않게 함.
- 학습은 `--resume work_dirs/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter/epoch_15_lss_only.pth`로 구 work_dir의 ckpt에서 이어받고, 저장만 새 폴더로 간다.
