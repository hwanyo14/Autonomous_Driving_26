# NOTES

## 2026-06-30 KST — Remaining loader pruning 메모

- `segmentation_bev`는 test `simple_test`의 BEV metric에 필요하므로 test에서는 유지하고 train에서만 `load_segmentation_bev=False`로 차단했다.
- `LoadMultiViewImageFromFiles_BEVDet`의 `results['canvas']`는 `Collect3D`/model/debug 경로에서 읽히지 않는다. 검색상 다른 `canvas`는 각 시각화 함수 내부 로컬 변수라 이미지 로더 output과 무관하다.
- `projects/configs/baselines/EfficientOCF_V1.1_1gpu.py`는 현재 repo에 없어서 반영 대상에서 제외했다. 현재 존재하는 baseline config는 `base_config.py`, `EfficientOCF_V1.1_lyft.py`뿐이다.

## 2026-06-30 KST — Data loading pruning 기준

- 현재 사용 config는 `projects/configs/baselines/base_config.py`로 확정해서 판단했다.
- train pipeline에서는 `LoadOccupancy`를 제거했다. 현재 `use_segmentation_as_query_gt=True`, train eval hook 비활성 상태라 `gt_occ`는 학습 손실에 쓰이지 않는다.
- test pipeline의 `gt_occ`와 `segmentation_bev`는 evaluation metric 계산에 쓰이므로 유지했다.
- `segmentation_cls_instance3d`는 `gt_occ_inst` 기반 dense class/instance bundle이 detector 내부에서 우선 사용되므로 base config에서 로드를 껐다.
- `segmentation`/`segmentation_instance3d`/`gt_occ_inst`/`gt_instance_centers_*`는 matching, local AABB GMO, trajectory/depth target, shape 기준에 연결되어 있어 유지했다.

## 2026-06-30 KST — Eval/visualization 이식 검증 메모

## 2026-06-30 KST — Unified Gaussian quaternion rotation 검증 메모

- `mixture_quat_tqg4`는 wxyz 순서이며 shape은 `[T,Q,G,4]`이다. 기존 yaw scalar shape `[T,Q,G]`를 기대하는 신규 코드가 생기면 shape mismatch를 먼저 의심할 것.
- `QueryHead` dummy forward와 voxelizer smoke는 현재 shell Python에 `torch`가 없어 `ModuleNotFoundError: No module named 'torch'`로 실행하지 못함.
- 변경 파일 `py_compile`과 yaw key 잔여 검색은 통과.

- `tools/misc/print_config.py projects/configs/baselines/base_config.py`는 현재 shell Python에 `mmcv`가 없어 `ModuleNotFoundError: No module named 'mmcv'`로 실행하지 못함.
- `SoftVoxelizerOneAdd.forward_gaussian_mixture_scene` torch smoke는 현재 shell Python에 `torch`가 없어 `ModuleNotFoundError: No module named 'torch'`로 실행하지 못함.
- 변경 파일 `py_compile`과 `tools/dist_test.sh`/`run_eval.sh` `bash -n`은 통과.

## 2026-06-26 KST — Independent Gaussian alpha 검증 메모

- `QueryHead` dummy forward/backward smoke는 현재 shell Python에 `torch`가 없어 `ModuleNotFoundError: No module named 'torch'`로 실행하지 못함.
- 변경 파일 `py_compile`과 `git diff --check`는 통과.

## 2026-06-26 KST — Main training config

- 사용자가 현재 메인으로 사용하는 config는 `projects/configs/baselines/test_traj_large_adj_lr_crop_margin.py`이다. 학습 인자 변경이 필요하면 이 config를 우선 반영할 것.

## 2026-06-25 KST — Matched GMO local AABB loss 검증 메모

- `_compute_matched_pair_gmo_losses(shape_loss_mode='local_aabb')` 더미 torch smoke는 현재 shell Python에 `torch`가 없어 `ModuleNotFoundError: No module named 'torch'`로 실행하지 못함.
- 변경 파일 `py_compile`과 `git diff --check`는 통과.

## 2026-06-23 KST — QueryDepthHead soft depth CE 검증 메모

- `tools/misc/smoke_past_frame_matching.py` 실행은 현재 shell Python에 `torch`가 없어 `ModuleNotFoundError: No module named 'torch'`로 중단됨. 변경 파일 `py_compile`과 `git diff --check`는 통과.

## 2026-06-21 KST — Query 이진 분류(bg/fg) 전환 주의사항

- query 분류의 binary/multi-class 전환은 **`use_separate_classes` 하나로** 제어함(`False`=binary, `True`=8-class). 별도 query 전용 flag 없음. 내부적으로 `self.query_binary_cls = not use_separate_classes`.
- `query_class_ids`(`[0,2,3,4,5,6,9,10]`)는 binary에서도 **foreground-id registry**로 유지됨(`utils_gt_prep`의 instance 필터 `allowed_raw_ids` 및 collapse map 입력). 라벨 공간(`query_num_classes`)과 분리되므로 `query_class_ids`를 줄이지 말 것.
- `query_cls_loss_class_weights` 길이는 반드시 `query_num_classes`와 일치해야 함(binary=2, multi=8). config 분기에서 함께 관리.
- binary collapse map은 background raw id→0, foreground raw id→1, registry 밖 id→-1(필터). GT instance는 foreground이므로 매칭 target은 1, 미매칭 query는 0(background).
- 시각화: 예측 marker/legend는 binary 색(background/foreground)으로 통일했으나, **GT semantic overlay(`gt_cls_rgb`)는 per-class 색을 유지**함(GT 참조용). GT도 단일 fg 색으로 합치려면 `query_head.py`의 GT semantic 블록(`class_palette` 사용부)을 별도 수정해야 함.
- matcher/loss/scoring은 logit shape에서 class 수를 읽어 자동 적응함(코드 수정 불필요). `query_cls_match_cost_weight`는 binary log-softmax 기준으로 그대로 동작.
- `utils_loss.py`의 per-class debug stats(`dbg_query_cls_*_score_c{i}`/`_pred_count_c{i}`/`_target_count_c{i}`)는 binary에서 `c0`,`c1`만 생성됨(기존 `c0`~`c7`). 모든 rank가 동일 C=2라 DDP 문제는 없으나, 8-class key를 기대하는 TensorBoard/로그 파서는 조정 필요.

## 2026-06-21 KST — Trajectory DDP log key 안정성

- rank별 유효 trajectory pair 유무가 달라도 `_aggregate_training_losses()`가 `loss_query_traj`와 6개 `dbg_query_traj_*` key를 항상 반환해야 함. 새 trajectory diagnostic key를 추가할 때도 zero prefill 목록을 함께 갱신할 것.

## 2026-06-26 KST — Local AABB GMO pair visualization

- `debug_gmo_local_aabb_pair_vis_every > 0`이어도 `query_gmo_shape_loss_mode='local_aabb'`인 학습 branch에서만 pair PNG/sidecar가 저장된다.
- `EfficientOCF_V1.1_1gpu.py`에는 저장 주기/경로만 추가했으며, 기존 `query_gmo_shape_loss_mode` 값은 변경하지 않았다. 실제 crop supervision 검증 시 사용하는 config가 local AABB 모드인지 먼저 확인할 것.
- 저장 경로 기본값은 `work_dirs/gmo_local_aabb_pair_vis*`; PNG는 BEV max-Z projection이고 원본 crop tensor는 `local_aabb_pairs.pt`에 들어 있다.

## 2026-06-21 KST — tf-traj-cost 단일 trajectory 경로 적용 범위

- `query_traj_matched_only=True`이면 trajectory input만 matching 전에 만들고, head forward는 Hungarian matching 후 matched query에 대해 1회 실행함.
- teacher forcing은 matched GT의 유효한 과거 step delta를 head prior에 넣는 hard forcing임. `query_traj_teacher_forcing_until_iter>0`인 config는 해당 iter 이후 전환하고, 값이 `0`이면 epoch hook이 boolean 상태를 단독 제어함.
- teacher-forced head 출력은 미래 query center와 Gaussian mixture center의 누적 이동에 사용되지만, `experiment/tf-traj-cost` 원본과 동일하게 현재 post-match 재생성 결과는 `*_vis_full` 시각화 geometry에만 반영됨. matched GMO loss 등 학습용 geometry에도 반영하려면 별도 변경이 필요함.
- `test_traj_large_cost_iou_resnet_lr_depthbin.py`는 epoch 1~8 teacher forcing 활성, epoch 9부터 비활성으로 GPU 수와 무관하게 동작함.

## 2026-06-18 KST — Hungarian BEV IoU cost 변경
- feature cosine/soft-assign matching cost는 제거됨. `_match_queries_to_gt_instances()`는 더 이상 query/GT feature tensor가 없어도 매칭 cost map을 생성함.
- BEV IoU cost는 matching 전용 2D Gaussian rasterize로 계산됨. 현재 `test_traj.py`, `test_traj_large_cost_iou.py`는 `query_matched_gmo_bce_occ_size=(64,64,20)`의 XY와 동일한 BEV `64x64` 기준, weight `1.0`.
- GMO focal/BCE/dice loss는 여전히 matched pair 대상 3D `64x64x20` voxelize를 사용함. IoU matching cost와 loss voxelize 결과는 캐싱 공유하지 않음.
- `dbg_query_match_cost_*` 목록은 `feat`, `soft`, `bev_dice`가 사라지고 `bev_iou`로 변경됨. TensorBoard/로그 파서가 기존 key를 기대하면 같이 바꿔야 함.
- `tools/misc/smoke_past_frame_matching.py` 전체 실행은 기존 QueryHead trajectory output 검증에서 `traj_offsets_fq2=None`으로 실패함. 이번 변경 검증은 matcher BEV IoU one-off와 compile/config load로 수행.

## 2026-06-11 KST — Trajectory 2-mode 이식 관련
- ~~TF 스케줄 재스케일 주의~~ → **해결됨 (6/11)**: TF를 epoch 기준으로 전환. iter 스케줄은 빈 튜플로 비활성, `gt_ratio`를 TrajectoryWarmupHook stage(epoch 1=1.0, 2=0.5, 3+=0.0)가 제어. GPU 수 변경 시 재스케일 불필요.
- TrajectoryWarmupHook은 **마지막 매칭 stage 하나만 통째로 적용** (누적 merge 아님). stage 추가 시 모든 키를 다 들고 있어야 함.
- warmup 스케줄은 `max_epochs=15` 전제 (epoch 1/5/8/11 계단). epoch 수 바꾸면 비율 맞춰 조정할 것.
- epoch 1~4는 `loss_query_traj_refine_xy=0`이 정상 (warmup이 refine weight를 0으로 둠). 버그 아님.
- pedestrian 제거가 이제 **이중**으로 걸림: (a) test_traj.py의 `exclude_occ_class_ids=(7,)` 로드 단계 제거, (b) gt_prep의 query_class_ids 필터. test.py는 (b)만 사용 중. 둘은 양립하며 (a)가 더 상류.
- traj 알려진 한계 (TRAJECTORY_CONFIG.md §1): moving 과소예측 (pred ~11% vs GT ~33%). mode cls CE에 moving 클래스 가중 추가가 근본 해결책이나 미적용 상태로 이식됨. 모니터링: `dbg/query_traj_pred_mode1_ratio`가 0.33에 수렴하는지.
- 코드에 GT-anchor refine(`refine_trajectory_with_gt_anchor`, `query_traj_anchor_*`)도 들어왔으나 **미사용** (xy_refine만 사용).
- 이번 조합(2-mode traj + inside-mass matching + history-valid filter)은 어느 레포에서도 같이 학습된 적 없는 신규 조합 — cen10_6 수치 재현 보장 없음.
- merge 후 총 9개 코드 파일이 cost/traj 어느 쪽과도 다른 통합본이 됨. 이후 다른 레포에서 추가 이식 시 3-way merge 필수 (base=dd6c98d).

## 2026-06-11 KST — QUERY_ATTN_CHANGES 이식 관련
- ⑤도 적용 완료 (q200, class weights, center 10.0/temporal 0.5, 8-class). `formating.py`(8-class 평가 클래스명 매핑)도 같이 이식함.
- **pedestrian(id 7)은 사용자 결정으로 완전 제외**: `query_class_ids`에서 빠지면 `utils_gt_prep.py`의 allowed_raw_ids 필터가 GT 로드 직후 해당 voxel을 background(0)로 덮어써 cls/instance 모두 제거됨. 데이터 캐시 자체는 그대로이므로 다시 포함하려면 config의 class_names/query_class_ids/cls_weights만 되돌리면 됨.
- `query_attn_overlap_loss`(⑨)는 efficientocf.py 의존성 때문에 코드만 들어갔고 weight 0.0으로 꺼져 있음. 켜려면 transformer 쪽 `query_attn_overlap_loss_weight` 설정.
- cls match cost(`query_cls_match_cost_weight`)는 학습 도중 켜면 악화 관측 있음 — from-scratch에서 0.05~0.075로 시작 권장 (QUERY_ATTN_CHANGES.md 3번 주의사항).
- 이식은 cost 레포 파일 통째 복사 방식이라, 이후 total 쪽에서 해당 9개 파일 수정 시 cost 레포와 diverge함.

## 2026-06-11 KST — 1~8 GPU DDP 수정 관련
- `init_sync`, `static_graph`는 torch DDP kwarg이며 현재 env(`eo`, torch 2.7.0+cu128)에서 둘 다 지원됨을 확인. torch를 1.x대로 다운그레이드하면 `init_sync`가 없어 TypeError 발생 가능.
- `static_graph=True`는 iteration마다 graph 구조(사용 파라미터/분기)가 동일하다는 가정. 조건부로 모듈 일부가 빠지는 구조 변경을 하면 DDP 에러가 날 수 있으니 그 경우 이 옵션부터 의심할 것.
- `dbg_query_match_cost_*` prefix 키 목록은 `utils_loss.py`의 prefill 루프와 실제 기록 지점이 수동으로 동기화돼 있음. 새 match-cost dbg 키를 추가하면 prefill 루프에도 반드시 같이 추가해야 multi-GPU에서 `log_vars` assert가 재발하지 않음.
- baseline config는 `samples_per_gpu=1`이라 GPU 수 1~8 어느 쪽이든 per-rank batch shape은 동일. 검증은 아직 안 돌렸음 (smoke run 필요).
