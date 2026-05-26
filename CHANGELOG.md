# Changelog

## 2026-05-26 KST (latest)

### Attn center distance auxiliary loss 추가

**핵심 내용**: 매칭된 (query, GT) 쌍에 대해 attention softargmax predicted 2D center와 GT mask centroid 간의 L2 거리를 보조 loss로 추가. on/off 및 weight 조절 가능.

**주요 변경사항**:
- `utils_loss.py`: `_compute_query_attn_center_dist_loss` 추가. matched pair에 대해 normalized L2 거리 계산
- `utils_loss.py` `_aggregate_training_losses`: `query_attn_center_dist_loss` 파라미터 추가 + `loss_query_attn_center_dist = raw * weight` 처리
- `efficientocf.py`: `query_attn_center_dist_loss` 초기화 및 호출, `_aggregate_training_losses` 전달
- `efficientocf_config.py`: `use_query_attn_center_dist_loss`, `query_attn_center_dist_loss_weight` 추가
- `EfficientOCF_V1.1_1gpu_traj_tf.py`: `use_query_attn_center_dist_loss=True`, `query_attn_center_dist_loss_weight=0.05` 설정

---

### Attn matching & scoring: IoU → center distance 기반으로 교체

**핵심 내용**: attention map이 넓게 분산될 경우 region IoU 기반 matching/scoring이 오염되는 문제 해결. softargmax 예측 2D center와 GT mask centroid 간 거리를 cost 및 score 기준으로 대체.

**주요 변경사항**:
- `utils_matcher.py`: `_compute_query_attn_center_dist_cost_qn` 추가. `_match_queries_to_gt_instances`에 `query_attn_match_metric="center_dist"` elif 분기 추가
- `utils_loss.py`: `_compute_query_attn_center_dist_score` 추가. GT mask centroid 기준 per-query 최소 거리 → `exp(-d/sigma)` 점수
- `efficientocf.py`: scoring 호출부를 `use_query_attn_cam_center_dist_score` 우선 분기 + 기존 Gaussian path elif로 변경
- `efficientocf_config.py`: `use_query_attn_cam_center_dist_score`, `query_attn_center_dist_score_sigma` 추가
- `EfficientOCF_V1.1_1gpu.py`: `query_attn_match_metric="center_dist"`, `use_query_attn_cam_center_dist_score=True`, `query_attn_center_dist_score_sigma=10.0` 추가

---

## 2026-05-26 KST

### DDP match-cost debug key pre-allocation fix (root cause)

**핵심 내용**: `_aggregate_training_losses` 끝에 `dbg_query_match_cost_` prefix가 아닌 모든 `dbg_*` 키를 삭제하는 필터가 있음을 확인. 실제 문제는 `dbg_query_match_cost_*` 32개 키가 `if isinstance(inst_match_result, dict)` + `if torch.is_tensor(cost_qn)` + 각 contrib 조건에 중첩 의존하여, 매칭 인스턴스가 없는 rank에서 키 누락 발생.

**주요 변경사항**:
- `detectors/utils_loss.py` `_aggregate_training_losses`: `if isinstance(inst_match_result, dict)` 블록 전에 32개 `dbg_query_match_cost_*` 키를 `z`로 선할당. 이전의 `dbg_query_traj_*` setdefault fix는 해당 키가 필터로 삭제되므로 무효였기에 원복.

---

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
