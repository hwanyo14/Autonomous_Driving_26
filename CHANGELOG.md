# Changelog

## 2026-05-27 19:19 KST

### Query Objectness Scoring

**핵심 내용**: query별 objectness head를 추가하고, Hungarian matched 여부로 supervision한 뒤 query 시각화 score에 gating으로 반영.

**주요 변경사항**:
- `query_head.py`: query feature에서 `[Q]` objectness logit/score를 예측하는 MLP head 추가
- `efficientocf.py`, `utils_loss.py`: Hungarian matched query는 1, unmatched query는 0으로 `loss_query_objectness` 계산 및 로깅 연결
- `utils_visualization.py`, `utils_query_vis.py`: 기존 cls/cam score를 objectness로 gating하고, row2는 all query에 high-score query를 강조 표시하며 row3/4는 threshold/top-k 통과 객체만 표시
- `efficientocf_config.py`, `EfficientOCF_V1.1_1gpu.py`: objectness loss weight/type 설정 추가

## 2026-05-27 18:55 KST

### Weak Class Cost for Hungarian Matching

**핵심 내용**: Hungarian matching에 GT class 확률 기반 weak class cost를 다시 추가.

**주요 변경사항**:
- `utils_matcher.py`: valid GT class에 대해 `-log p(gt_class)` class cost를 계산하고 `query_cls_match_cost_weight`로 cost에 반영
- `EfficientOCF_V1.1_1gpu_traj_tf.py`: 약한 class matching cost로 `query_cls_match_cost_weight=0.2` 설정
- `utils_loss.py`: class matching cost weight/contribution diagnostics 복구

## 2026-05-27 16:41 KST

### Scoring / Hungarian Matching 비활성 요소 제거

**핵심 내용**: `EfficientOCF_V1.1_1gpu_traj_tf.py` 기준으로 실제 score/cost에 포함되지 않던 scoring IoU, feature/soft/class/BEV-dice matching cost 및 관련 계산부를 제거.

**주요 변경사항**:
- scoring: query score를 class probability + cam attention score만 사용하도록 정리하고, cam attention score는 별도 false flag 없이 생성 조건이 맞으면 계산되도록 변경
- Hungarian matching: center distance, temporal offset, attention soft-IoU cost만 남기고 비활성 cost 계산/diagnostics/config 키 제거
- `EfficientOCF_V1.1_1gpu.py`, `EfficientOCF_V1.1_1gpu_traj_tf.py`: 제거된 scoring IoU weight 설정 삭제
- 사용되지 않게 된 GT feature pooling, matching feature frame selection, matched-query IoU score helper 제거

## 2026-05-27

### query_head.py 시각화 코드 분리 및 리팩토링

**핵심 내용**: `query_head.py`의 쿼리 시각화 관련 코드를 `utils_query_vis.py`로 분리. 4098줄 → 2581줄 (약 37% 감소).

**주요 변경사항**:
- `dense_heads/utils_query_vis.py` 신규 생성: `VisConfig` 데이터클래스, `draw_marker_splats`, `draw_cross_marker`, `draw_gaussian_bev_footprints`, `get_query_vis_palette`, `draw_query_vis_legend`, `draw_generic_vis_legend`, `normalize_gt_occ_semantic_for_vis`, `normalize_gt_occ_inst_for_vis`, `save_prob_grid_vis`, `is_main_process` 포함
- `query_head.py`: 위 함수들 제거 후 import로 교체, `_vis_cfg` property 추가, `_is_main_process` 1-line 위임, `_maybe_save_prob_grid_vis` thin wrapper로 대체

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
