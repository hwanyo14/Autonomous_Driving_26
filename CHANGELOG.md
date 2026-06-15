# Changelog

## 2026-06-15 KST (3차)

### MPS 상시 자동 적용 (tools/dist_train.sh) + stop_mps.sh

- 목표: 그냥 학습을 실행만 해도 NVIDIA MPS 클라이언트가 되게 (8장 공유 멀티잡 시 시분할 오버헤드 제거). MPS_MULTIJOB_RUNBOOK.md의 "상시적용" 절차를 실제 구현.
- `tools/dist_train.sh`: torchrun 호출 **이전**에 MPS 부트스트랩 블록 추가.
  - `USE_MPS`(기본 1). 0이면 스킵.
  - `CUDA_MPS_PIPE_DIRECTORY`/`CUDA_MPS_LOG_DIRECTORY` 기본 `/tmp/nvidia-mps[-log]`(override 가능) export → 이 launch의 모든 프로세스가 MPS 클라이언트가 됨 (컨텍스트 생성 전 env 설정이라 유효).
  - 데몬 멱등 시작: `pgrep -f "[n]vidia-cuda-mps-control -d"`로 떠있으면 재사용, 없으면 `nvidia-cuda-mps-control -d`로 시작(실패해도 학습은 계속). 대괄호 트릭으로 셸 자기매칭 방지.
  - 모든 실행 경로 커버: train_total.sh → run.sh → dist_train.sh, 직접 호출 모두. (run.sh는 단순 포워딩, dist_train.sh 이전에 CUDA 컨텍스트 생성 없음 확인.)
- `stop_mps.sh`(신규, 루트): MPS 데몬 종료 + 정리. 순서 준수(quit → pkill → rm), 학습 잡은 안 건드림.
- 검증: bash -n 통과(dist_train.sh/stop_mps.sh), 멱등 감지 동작 확인(이미 떠있는 데몬 재사용), `nvidia-cuda-mps-control` 응답 100.0. PROJECT_STRUCTURE.md에 stop_mps.sh/런북 항목 추가.

## 2026-06-15 KST (2차)

### GPU 멀티잡 운용 가이드 문서 추가 (GPU_MULTIJOB_NOTES.md)

- 한 노드 8 GPU에서 학습 잡 2개 동시 실행 시 iter 9s→40s(>4x) 슬로다운 분석 결과를 포터블 md로 정리.
- 원인: 두 잡이 같은 물리 GPU를 겹쳐 점유(oversubscription) → 시분할 + 컨텍스트 스위치 + CPU/NUMA 가중. 데이터/Lustre IO는 무죄(로그상 data_time ~6%).
- 내용: 원인 설명, 진단 명령(로그 time/data_time 분해, 프로세스→GPU 매핑, dmon, NUMA), 해결 3안(GPU 분할 4+4 / MPS 공유 / 순차) + 처리량 비교표, MPS 개념, 체크리스트, 오해 정리. 이 노드 실측값은 부록으로 분리.
- 멀티 GPU 다중 워크플로(병렬 조사 4갈래 + 종합)로 시스템 직접 측정해 도출. PROJECT_STRUCTURE.md에 항목 추가.

## 2026-06-15 KST

### mix3v(query mixture 3D) 시각화 저장 경로를 다른 vis와 동일 구조로 통일

- 기존에 다른 모든 vis는 `_configure_visualization_dirs`(`mmdet_train.py`)가 `{work_dir}/vis/{timestamp}/{folder}`로 자동 배치하는데, **mix3v만 이 map에서 누락**되어 있었음 → mix3v만 자체 `iter_NNNNNN/` 폴더로 따로 저장되던 문제.
- 수정: `_configure_visualization_dirs`의 `detector_dir_map`에 `"debug_query_mixture3d_vis_dir": "query_mixture3d_vis"` 한 줄 추가. → 다른 vis와 똑같이 `{work_dir}/vis/{timestamp}/query_mixture3d_vis/`로 자동 저장 (실행 config 이름·timestamp 자동).
- `utils_visualization.py`(mix3v 저장부): 위에서 설정된 `self.debug_query_mixture3d_vis_dir`를 그대로 사용, 파일명만 `iter_{step:06d}.png`로 변경 (이전 `iter_NNNNNN/` 하위폴더 + `mix3v_{scene}_{lidar}.png` → 하위폴더 제거, iter는 파일명에만).
- 최종 경로 예: `work_dirs/test_traj_shape/vis/20260615_111210/query_mixture3d_vis/iter_000048.png`
- mix3v는 iter당 1장 저장이라 파일명에 scene/lidar 토큰 없어도 충돌 없음.
- 검증: py_compile 통과 (train.py/mmdet_train.py/efficientocf_config.py/utils_visualization.py/test_traj_shape.py).

## 2026-06-12 KST (2차)

### shape Run 1: dice 3D화 + lowres 상향 + shape dbg 지표 (test_traj_shape.py)

- **dice 3D화** (`utils_loss.py`): 새 플래그 `query_gmo_dice_3d`(기본 False). True면 matched gmo dice를 BEV 2D(amax z-collapse) 대신 **3D tversky**로 계산 — z 모양이 처음으로 supervise됨. mixture grouped/surrogate single 두 경로 모두 적용, 공용 헬퍼 `_compute_pair_dice_bev_and_3d`로 정리.
- **lowres 상향** (test_traj_shape.py): `query_matched_gmo_bce_occ_size` (64,64,20)→**(128,128,40)** — xy 복셀 1.6m→0.8m (차 3×1→6×2 복셀). 코드 수정 없음 (voxelizer가 init에서 자동 재구성).
- **shape dbg 지표 추가** (matched pair lowres, hard thr 0.5, 매 iter TB 로깅):
  - `dbg_gmo_dice_bev` / `dbg_gmo_dice_3d`: 두 dice를 항상 모두 계산해 로깅 (loss 채택은 플래그 따름) → 기존 BEV-dice run과 비교 가능. `dbg_gmo_dice_is_3d`로 모드 확인.
  - `dbg_gmo_shape_iou3d` / `dbg_gmo_shape_iou_bev`: matched pair 평균 hard IoU.
  - `dbg_gmo_shape_z_extent_ratio`: pred/GT 점유 z-슬라이스 수 비율 (blob이면 ≫1, 목표 →1).
  - `dbg_gmo_shape_bev_area_ratio`, `dbg_gmo_shape_vol_ratio`: BEV 면적/3D 부피 비율 (blob 팽창도).
  - `dbg_gmo_shape_pair_count`: 지표 모집단 크기.
  - 헬퍼 `_gmo_pair_shape_stats` (detach, gradient 무관).
- config: `query_gmo_dice_3d` 기본값 efficientocf_config.py에 추가. test_traj_shape.py에 `query_gmo_dice_3d=True` + mixture 3D vis 키(`debug_query_mixture3d_vis_every=48`, dir)도 추가.
- 검증: py_compile 통과 + 더미 pair 단위 테스트 (3D dice > BEV dice for z-blob, z_ratio=3.6 정확, 가드 None, gradient OK).
- **미검증**: (128,128,40)에서의 voxelizer 메모리 — 학습 시작 직후 1 iter OOM 여부 확인 필요. OOM 시 1차 완충은 `query_multi_gaussian_pair_chunk` 축소(8→4), 2차는 (128,128,20).

## 2026-06-12 KST (1차)

### 학습 중 query mixture 3D 시각화 추가

- `utils_visualization.py`에 `maybe_save_query_mixture_3d_vis` 추가 (3D_SHAPE_ANALYSIS.md §3.2 mix3v 포팅).
  - present frame(t=2)만 렌더, 3-view 1장: oblique(z aspect 0.22≈2.8x) / oblique rear / low side(진비율 0.078).
  - GT instance 복셀 산점도(tab20, 상한 `debug_query_mixture3d_vis_max_gt_points=40000` 초과 시 stride 다운샘플) + GT center(lime X, 1번 패널에 클래스 라벨).
  - query 선별은 bundle의 기존 score 선별(selected) 재사용, matched는 검은 테두리. mixture 16성분 center 점(크기∝weight) + weight 상위 6성분 yaw 반영 1σ wireframe ellipsoid.
  - 렌더 예외는 print 후 무시 (학습 보호), rank0 only, matplotlib Agg.
- 호출: `efficientocf.py`의 `maybe_save_query_debug_vis` 직후, bundle + `gt_instance_occ3d_txyz_query_vis` + GT center/cls full 텐서 전달.
- config 키 추가 (efficientocf_config.py 기본값 + EfficientOCF_V1.1_1gpu.py):
  `debug_query_mixture3d_vis_every=48` (기본 0=off), `debug_query_mixture3d_vis_dir`, `_max_queries=50`, `_max_gt_points=40000`.
- 산출 구조: **호출 iter마다 새 폴더** `{vis_dir}/iter_000048/mix3v_{scene}_{lidar}.png`.
- 검증: py_compile 4파일 통과, 더미 텐서 standalone 스모크 테스트로 PNG 생성 확인.

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
