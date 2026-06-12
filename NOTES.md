# NOTES

## 2026-06-12 KST — CUDA ext 빌드
- 이 레포는 `occ_pool_ext` CUDA extension이 빌드 안 된 상태였음 (학습/full import 시 ImportError). 소스가 cost 레포와 동일함을 diff로 확인 후 빌드된 `occ_pool_ext.cpython-310-x86_64-linux-gnu.so`를 cost 레포에서 복사해 해결. **python/torch 버전이나 서버가 바뀌면 `projects/occ_plugin/ops/occ_pooling/setup.py`로 재빌드 필요.**

## 2026-06-12 KST — shape Run 1 (dice 3D + lowres 128) 관련
- `loss_gmo_dice` 절대값은 기존 run과 **직접 비교 불가** (3D dice는 본질적으로 더 어려움). 비교는 `dbg_gmo_dice_bev`(항상 BEV 기준으로 같이 로깅됨)로 할 것.
- (128,128,40) voxelizer 메모리 미측정 — 학습 첫 iter OOM 시: ① `query_multi_gaussian_pair_chunk` 8→4, ② 그래도 안 되면 `query_matched_gmo_bce_occ_size=(128,128,20)` (z는 dice 3D가 0.4m에서도 잡으므로 손해 적음).
- shape 지표 해석: 학습이 잘 가면 `dbg_gmo_shape_z_extent_ratio`·`bev_area_ratio`·`vol_ratio`가 1로 수렴하고 `iou3d` 상승. z_ratio만 1로 가고 area_ratio가 안 내려오면 ①만 먹고 ②가 부족하다는 뜻 → bbox prior(Run 2) 근거.
- dice가 3D가 되면서 tversky 분모가 커져 gradient 스케일 변화 가능 — `query_gmo_dice_loss_weight=0.5` 기본 유지로 시작했으나 dice가 BCE를 압도/실종하면 조정.
- Run 2(bbox prior) 사전 확인사항: 이 파이프라인엔 3D GT bbox가 없음 (attn_bbox는 카메라 평면 mask). GT instance 복셀에서 box 유도(yaw=BEV PCA) 방식 권장.

## 2026-06-12 KST — query mixture 3D 시각화 관련
- 선별은 기존 score 기반(selected)을 재사용 (사용자 결정). 3D_SHAPE_ANALYSIS.md §4가 권장한 margin(fg−bg≥0.15) 선별은 미구현 — 필요 시 `debug_query_score_mode='margin'` 옵션을 `_build_query_visualization_bundle`에 추가해야 함.
- matplotlib 3D는 axis limit 밖 점을 clip하지 않음 — pc_range 밖 query가 패널 밖으로 비어져 보일 수 있음 (무해).
- 렌더는 present frame(t=2)만. future frame 렌더가 필요하면 `t_present` 부분을 루프로 확장.
- `debug_query_mixture3d_vis_every` 기본값은 0(off), EfficientOCF_V1.1_1gpu.py에서만 48로 켜져 있음. 다른 config(test_traj.py 등)에서 쓰려면 debug_cfg/visualization_cfg에 키 추가 필요.

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
