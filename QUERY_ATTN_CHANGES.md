# Query-Attention / Matching 변경사항 (타 레포 이식용)

이 문서는 baseline 대비 추가한 6가지 변경의 **구현 방법 + 정확한 하이퍼파라미터**를 정리한 것.
표기: `pred`=쿼리 attention map(정규화된 확률분포), `gt`=인스턴스 마스크, `p`=BEV/attn 픽셀 수, `t`=프레임, `q`=쿼리, `n`=인스턴스, `k`=matched pair.

---

## 0. 공통 전제

- 각 쿼리는 cross-attention map `attn_tqnhw`를 가짐. 카메라축(n) 합산 후 픽셀축으로 L1 정규화하여
  확률분포 `pred_tqp` (Σ_p = 1)로 만든다.
  ```python
  attn_tqp = attn_weights.sum(dim=2).reshape(t, q, p)          # 카메라 합산
  pred_tqp = attn_tqp / attn_tqp.sum(-1, keepdim=True).clamp_min(eps)   # 픽셀 정규화 → 확률
  ```
- GT 인스턴스 마스크 `gt_tnp` (0/1), valid 플래그 `valid_tn`.
- **inside mass** 정의: 쿼리 attn 확률이 특정 GT 마스크 영역 안에 떨어진 질량의 합.
  ```python
  inside_mass = (pred * gt_mask).sum(-1).clamp(0, 1)   # ∈ [0,1]
  ```
  이 한 가지 정의가 (1) loss 와 (3) matching cost 양쪽에서 재사용된다.

---

## 1. Attention LOSS 를 inside-mass 기반으로 변경

**무엇**: matched 쿼리의 attn map이 자기 GT 마스크 안에 얼마나 모였는지를 `-log(inside_mass)`로 학습.
(기존 soft_iou / 다른 방식 → inside-mass NLL 로 교체)

**구현** (`utils_loss.py::_compute_query_attn_bbox_loss`, 핵심 L832~835):
```python
inside_mass_tk = (pred_tkp * target_mask_tkp).sum(-1).clamp(0.0, 1.0)   # matched only
loss_tk        = -torch.log(inside_mass_tk.clamp_min(eps))
matched_raw    = (loss_tk * valid_f).sum() / valid_cnt.clamp_min(1.0)   # valid 평균
```
- `pred_tkp` = matched 쿼리들의 정규화 attn (index_select(mq)), `target_mask_tkp` = 그 쿼리에 매칭된 GT 마스크.
- valid 조건: GT 마스크 비어있지 않고(`sum>0`), valid 플래그 True, pred finite.

**최종 loss 합성** (L907~908):
```python
loss_raw = matched_raw + other_w * other_raw + unmatched_w * unmatched_raw
loss_query_attn_bbox = loss_raw * total_w
```
→ 로그의 `loss_query_attn_bbox` 항.

**하이퍼파라미터**
| 이름 | 값 | 의미 |
|---|---|---|
| `query_attn_bbox_loss_weight` | `1.0` | 위 `total_w` |
| `query_attn_bbox_eps` | `1e-6` | log clamp |

---

## 2. Attention loss 에 SUPPRESS(타 객체 영역 억제) 추가

**무엇**: matched 쿼리가 **다른 객체** 영역(union − 자기 GT)에 attn 질량을 두면 페널티.
(쿼리 attn이 자기 객체에만 집중하도록 분리 유도)

**구현** (`utils_loss.py` L848~864):
```python
union_mask_tp = (gt_mask * valid).amax(dim=1)          # 모든 valid 인스턴스의 합집합
other_mask    = (union_mask_tp.unsqueeze(1) - target_mask).clamp(0, 1)   # 자기 GT 제외 = 타 객체
other_mass    = (pred_tkp * other_mask).sum(-1).clamp(0, 1)
if other_mode == "union_log":
    other_loss = -torch.log((1.0 - other_mass).clamp_min(eps))
else:                       # 'union'  (현재 사용)
    other_loss = other_mass        # 타 객체에 떨어진 질량을 그대로 페널티
other_raw = (other_loss * other_valid_f).sum() / other_valid_cnt.clamp_min(1.0)
```
(선택) unmatched 쿼리 억제도 동일 함수에 있음: union 전체에 떨어진 질량 페널티
(`unmatched_mode`, L866~886). 현재는 weight 0 으로 비활성.

**하이퍼파라미터**
| 이름 | 값 | 의미 |
|---|---|---|
| `query_attn_bbox_other_weight` | `0.1` | suppress 항 weight (위 `other_w`) |
| `query_attn_bbox_other_mode` | `'union'` | `'union'`=질량 직접 / `'union_log'`=−log(1−mass) |
| `query_attn_bbox_unmatched_weight` | `0.0` | unmatched 억제 (비활성) |
| `query_attn_bbox_unmatched_mode` | `'inverse_union'` | unmatched 모드 |

---

## 3. Matching COST 를 inside-mass(`inside_log`)로 변경

**무엇**: Hungarian 매칭 비용행렬에 쿼리-인스턴스 간 `-log(inside_mass)` 항을 추가.
(기존 `soft_iou` 옵션 → `inside_log` 로 교체)

**구현** (`utils_matcher.py::_compute_query_attn_inside_log_cost_qn`, L355~356):
```python
inside_mass_tqn = torch.einsum("tqp,tnp->tqn", pred_tqp, gt_tnp).clamp(0, 1)   # 모든 q×n 쌍
cost_tqn        = (-torch.log(inside_mass_tqn.clamp_min(eps))).clamp(0.0, 30.0)
# valid 프레임 평균 → cost_qn  (q×n 비용행렬)
```
**비용행렬 합성** (L777~811):
```python
if attn_match_metric == "inside_log":
    attn_cost_qn, valid = self._compute_query_attn_inside_log_cost_qn(...)
attn_contrib = query_attn_match_cost_weight * attn_cost_qn * valid
cost_qn = cost_qn + attn_contrib
...
row, col = linear_sum_assignment(cost_qn)   # _solve_unique_assignment, L893
```

전체 비용행렬은 여러 항의 가중합:
```
cost_qn = feat + cls + center + temporal_offset + bev_dice + attn(inside_log)
```
각 항의 weight(아래 표)와 구현 위치:
- cls cost: `-log_softmax(logits)[gt_cls]`, L600~605 (#6 매칭쪽, 기본 off)
- center cost: GT-pred center L1 / BEV대각선, clamp[0,2], L646
- temporal_offset cost: 프레임간 center delta L1 / `traj_residual_max`, L664, weight 0.5
- attn(inside_log): 위 설명

**하이퍼파라미터**
| 이름 | 값 | 의미 |
|---|---|---|
| `query_attn_match_metric` | `'inside_log'` | `'soft_iou'`/`'inside_log'` |
| `query_attn_match_cost_weight` | `0.3` | 비용행렬 attn 항 weight |
| `query_attn_match_eps` | `1e-6` | log clamp |
| `query_center_match_cost_weight` | `10.0` | center 항 weight |
| `query_temporal_offset_match_cost_weight` | `0.5` | temporal delta 항 weight |
| `query_cls_match_cost_weight` | `0.0` (실험: `0.05~0.075`) | 매칭에 cls 항 (기본 off) |

> 주의: cls match cost 는 **학습 도중 켜면 악화**된 관측이 있음(center L2 4.65→6.05).
> 켜려면 처음부터(from scratch) 작은 값(0.05~0.075)으로 시작 권장.

---

## 4. 쿼리 개수 200

**무엇**: query 수 100 → 200 (100 버전도 병행 검증중, 미확정).

**하이퍼파라미터**
| 이름 | 값 |
|---|---|
| `query_num_queries` | `200` (또는 `100`) |

---

## 5. Valid 한 인스턴스만 처리 (생성/소멸 제거)

**무엇**: history(과거 receptive field) 전 프레임에서 **계속 존재**한 인스턴스만 GT로 사용.
중간에 생성(appear)되거나 소멸(disappear)한 객체는 매칭/loss 대상에서 제외 → 라벨 안정화.

**구현** — 유지할 instance id 계산 (`utils_gt_prep.py::_build_history_all_valid_instance_ids`, L493):
```python
keep_n = valid_tn[:time_receptive_field].to(torch.bool).all(dim=0)  # 과거 모든 프레임 valid
return ids_n[keep_n]
```
이 id 집합으로 sparse target(center/valid/ids) 과 dense GT(occ/seg/cls)를 필터
(`efficientocf.py` L1238~1268, `_filter_instance_targets_by_intersection_ids` / `_filter_dense_instance_ids`).
dense 쪽은 제외 대상 voxel 을 background(0)로 덮어씀(`utils_gt_prep.py` L496~529).

**하이퍼파라미터**
| 이름 | 값 | 의미 |
|---|---|---|
| `query_require_history_all_valid` | `True` | 이 필터 on/off |
| `time_receptive_field` | `3` | "전 프레임" 정의 (과거 3프레임 모두 valid) |
| `query_present_only` | `True` | 현재프레임 기준 슬라이스 사용 |

---

## 6. 클래스 가중치 (class imbalance)

**무엇**: 분류 CE 에 per-class weight 적용. 지배 클래스(c0=배경/대다수)를 강하게 down-weight.

**구현** (`utils_loss.py` L260~266):
```python
ce_weight = class_weights.to(...)        # 길이 = num_classes
loss = F.cross_entropy(logits, targets, weight=ce_weight,
                       ignore_index=ignore_target, reduction="mean") * loss_weight
```

**가중치 튜닝 이력** (c0 을 계속 낮춤):
```
0.1  → 0.03 → 0.025 → 0.02     (c0, 지배 클래스)
```

**하이퍼파라미터**
| 이름 | 값 | 의미 |
|---|---|---|
| `query_cls_loss_class_weights` | `[0.02, 1.4, 1.3, 0.3, 1.4, 1.4, 1.2, 0.9]` | per-class CE weight (q200 baseline) |
| (q100 변형) | `[0.04, 1.4, 1.3, 0.3, 1.4, 1.4, 1.2, 0.9]` | c0=0.04 |
| `query_cls_loss_weight` | `1.0` | CE 전체 weight |
| `query_num_classes` | `8` | 클래스 수 |
| `query_class_ids` | `[0, 2, 3, 4, 5, 6, 9, 10]` | 원본 라벨 → 0..7 매핑 |

---

## 부록: 이식에 필요한 전체 하이퍼파라미터 (q200 baseline = `attn_suppress_mat_q200_valid_cls_5`)

```python
# --- queries ---
query_num_queries=200,           # (또는 100)
query_num_classes=8,
query_class_ids=[0, 2, 3, 4, 5, 6, 9, 10],
query_embed_dim=64,
query_pred_num_frames=1,
time_receptive_field=3,

# --- (1)(2) attn bbox loss + suppress ---
query_attn_bbox_loss_weight=1.0,
query_attn_bbox_eps=1e-6,
query_attn_bbox_other_weight=0.1,        # suppress
query_attn_bbox_other_mode='union',
query_attn_bbox_unmatched_weight=0.0,
query_attn_bbox_unmatched_mode='inverse_union',

# --- (3) matching cost ---
query_attn_match_metric='inside_log',
query_attn_match_cost_weight=0.3,
query_attn_match_eps=1e-6,
query_center_match_cost_weight=10.0,
query_temporal_offset_match_cost_weight=0.5,
query_cls_match_cost_weight=0.0,         # 실험시 0.05~0.075, from-scratch 권장

# --- (5) valid filter ---
query_require_history_all_valid=True,
query_present_only=True,

# --- (6) class weights ---
query_cls_loss_weight=1.0,
query_cls_loss_class_weights=[0.02, 1.4, 1.3, 0.3, 1.4, 1.4, 1.2, 0.9],

# --- 기타 관련(매칭/궤적) ---
query_traj_residual_max_m=(15.0, 15.0),
query_traj_loss_weight=0.5,
query_traj_teacher_forcing=True,
query_traj_teacher_forcing_until_iter=30000,
query_center_routed_loss_weight=0.3,
query_decor_loss_weight=1.0,             # (decor3 실험에서 3.0)

# --- gaussian attn 표현 ---
query_num_gaussians=16,
query_multi_gaussian_weight_mode='softplus',
query_multi_gaussian_sigma_min_m=(0.15, 0.15, 0.15),
query_multi_gaussian_sigma_max_m=(1.0, 1.0, 1.0),
query_multi_gaussian_offset_max_m=(3.0, 3.0, 0.7),
query_multi_gaussian_sigma_reg_loss_weight=0.01,
query_matched_gmo_bce_occ_size=(64, 64, 20),
query_gmo_loss_type='focal',
```

### 소스 위치 요약
| 변경 | 파일 | 함수 / 라인 |
|---|---|---|
| (1) attn loss | `utils_loss.py` | `_compute_query_attn_bbox_loss` L832~835, L907 |
| (2) suppress | `utils_loss.py` | 동 함수 L848~864 |
| (3) match cost | `utils_matcher.py` | `_compute_query_attn_inside_log_cost_qn` L355, 합성 L787~811 |
| (cls/center/temporal cost) | `utils_matcher.py` | L600~605 / L646 / L664 |
| (5) valid filter | `utils_gt_prep.py` L493, `efficientocf.py` L1238~1268 |
| (6) class weight CE | `utils_loss.py` | L260~266 |
| 기본값 | `efficientocf_config.py` | DEFAULT dict |
