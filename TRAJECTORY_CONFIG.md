# Trajectory Module Spec (cen10_6 / 2-mode)

다른 레포 포팅용. EfficientOCF query head의 **trajectory 예측 브랜치** 전체 설정 — 구조, 하이퍼파라미터, warmup/teacher-forcing 스케줄, 라우팅 룰, loss, 추론, 필요한 GT 입력까지.

기준 config: `8gpu_EfficientOCF_V1.1_multi_tf_refine_mix_nohuman_full_turn_pred_cen10_6.py`

---

## 1. 설계 요약 (2-mode)

객체(query)별 미래 궤적을 **2개 모드**로 예측:

| mode | 종류 | 출력 |
|---|---|---|
| **mode0** | static (built-in) | 항상 `(0,0)` (학습 파라미터 없음) |
| **mode1** | moving (learned) | MLP → `tanh(logits) * residual_max_m` per-step delta |

- **라우팅 룰**: GT 미래 총 변위 `|endpoint| ≤ static_threshold_m` → static(mode0), 초과 → moving(mode1)
- **선택**: 학습은 룰로 정답 라벨 생성, 추론은 `argmax(L0, L1)` (binary)
- **이유**: static(≈67%) vs moving(≈33%)은 균형 잡힌 binary라 분류 collapse가 구조적으로 적음. (5-mode static/straight/turn 분할은 turn 4%가 묻혀 추론에서 안 골라짐 → 폐기)

> ⚠️ 알려진 한계: binary로도 분류기가 다수(static)로 편향되어 **moving을 과소예측**(pred ~11% vs GT ~33%). 이는 모드 개수가 아니라 cls majority 편향 문제이며, static/moving CE에 moving 클래스 가중을 줘야 근본 해결됨(미적용).

---

## 2. 프레임 / 타임라인

```
time_receptive_field = 3      # 과거 입력 프레임 (t-2, t-1, t)
n_future_frames      = 4      # 예측 미래 프레임 (t+1 ~ t+4)
n_future_frames_plus = 6
query_present_only   = True   # query 브랜치는 present(t)만 사용
query_pred_num_frames= 1
# present 인덱스 = time_receptive_field - 1 = 2 (=t)
```

- 궤적 GT = 미래 4프레임 instance center의 **프레임간 변위**:
  `gt_delta[k] = center(t+k) - center(t+k-1)`, k=1..4
- endpoint = `Σ gt_delta = center(t+4) - center(t)`
- nuScenes 키프레임 2Hz(0.5s) → 4프레임 = 미래 2.0s

---

## 3. 핵심 하이퍼파라미터 (model_cfg)

```python
# --- 모드 구조 ---
query_traj_num_modes              = 2          # mode0=static, mode1=moving
query_traj_use_stationary_mode    = True       # mode0 = 정확히 0
query_traj_use_cv_mode            = False
query_traj_decoder_type           = 'offset'   # per-step xy offset 디코더
query_traj_residual_max_m         = (15.0, 15.0)  # learned delta clip: tanh()*15m

# --- 라우팅 (정답 라벨 생성) ---
query_traj_semantic_routing_enabled = True     # static_threshold로 static/moving 분기 (켜야 함)
query_traj_rule_turn_family_enabled = False    # turn family 미사용 (2-mode)
query_traj_rule_based_mode_enabled  = False    # (코드 미사용, 무시)
query_traj_static_threshold_m       = 0.8      # |endpoint| ≤ 0.8m → static
query_traj_rule_turn_threshold_deg  = 10.0     # turn family용, 여기선 dead
query_traj_cv_error_threshold_m     = 0.5      # CV mode용, 여기선 dead

# --- 추론 ---
query_traj_mode_infer_policy        = 'argmax' # binary 선택 (family_logsumexp는 5모드 강제라 불가)

# --- loss 가중 (warmup이 epoch별로 덮어씀, 아래 §4) ---
query_traj_loss_weight              = 0.5      # moving_reg loss 가중 (최종 stage 값)
query_traj_mode_cls_loss_weight     = 0.1      # static/moving 분류 CE 가중
query_traj_static_weight            = 1.0      # static GT reg 가중
query_traj_moving_weight            = 5.0      # moving GT reg 가중 (희소 moving 강조)
query_traj_moving_reweight_enabled  = True
query_traj_moving_threshold_m       = 0.8      # reg reweight 기준 (이동량 ≥ → moving)

# --- xy refine (선택된 궤적 후처리 보정) ---
query_traj_xy_refine_enabled        = True
query_traj_xy_refine_loss_weight    = 0.1
query_traj_xy_refine_num_layers     = 2
query_traj_xy_refine_hidden_dim     = 64

# --- 비활성 (구조상 끔) ---
query_traj_static_gate_enabled        = False  # 별도 static 게이트 (argmax와 양립 위해 off)
query_traj_derivative_routing_enabled = False  # 학습형 mode 라우터
query_traj_derivative_routing_hidden_dim = 16
```

---

## 4. Warmup 스케줄 (epoch 단위, `TrajectoryWarmupHook`)

매 epoch 시작 시 `begin_epoch ≤ 현재epoch`인 stage 값을 model 속성에 **덮어씀(계단식, 보간 아님)**.

```python
traj_warmup_schedule = [
    dict(begin_epoch=1,  query_traj_loss_weight=0.05, query_traj_xy_refine_loss_weight=0.00, query_traj_mode_cls_loss_weight=0.02, query_traj_teacher_forcing_enabled=True),
    dict(begin_epoch=5,  query_traj_loss_weight=0.15, query_traj_xy_refine_loss_weight=0.02, query_traj_mode_cls_loss_weight=0.05, query_traj_teacher_forcing_enabled=True),
    dict(begin_epoch=8,  query_traj_loss_weight=0.25, query_traj_xy_refine_loss_weight=0.05, query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True),
    dict(begin_epoch=11, query_traj_loss_weight=0.50, query_traj_xy_refine_loss_weight=0.10, query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True),
]
# hook 등록
custom_hooks = [ dict(type='TrajectoryWarmupHook', schedule=traj_warmup_schedule), ... ]
```

| epoch | traj_loss_w | xy_refine_w | mode_cls_w | 의도 |
|---|---|---|---|---|
| 1–4 | 0.05 | 0.0 (off) | 0.02 | 약하게 기본 궤적 학습, refine off |
| 5–7 | 0.15 | 0.02 | 0.05 | refine 켜고 점증 |
| 8–10 | 0.25 | 0.05 | 0.10 | 분류·정밀화 강화 |
| 11– | 0.50 | 0.10 | 0.10 | 최종 가중 |

> 총 15 epoch 학습 기준 (`runner = EpochBasedRunner(max_epochs=15)`).

---

## 5. Teacher Forcing 스케줄 (iter 단위)

궤적 계산 anchor를 GT present center(=teacher) vs 모델 예측 present center로 섞는 비율.

```python
query_traj_teacher_forcing_enabled        = True
query_traj_teacher_forcing_mix_enabled    = True
query_traj_teacher_forcing_gt_ratio       = 1.0   # 초기값
query_traj_teacher_forcing_schedule_iters     = (3000, 4000, 5000, 6500, 8000)
query_traj_teacher_forcing_schedule_gt_ratios = (1.0,  0.9,  0.7,  0.5,  0.0)
```

| iter | gt_ratio | 의미 |
|---|---|---|
| <3000 | 1.0 | 완전 teacher forcing (GT anchor, 안정적 초기 학습) |
| 3000→8000 | 1.0→0.0 | 점진적으로 free-running 전환 |
| ≥8000 | 0.0 | 완전 자기예측 (추론 조건) |

> 4000 iter/epoch 기준 ⇒ 대략 epoch 1~2에 걸쳐 GT→자기예측 전환.

---

## 6. 학습 흐름 (라우팅 → loss)

```
[Forward]  query_feat → mode_logits[Q,2], mode1_delta[Q,F,2] (+ mode0=0)
           → traj_deltas[Q,2,F,2]

[라우팅]   GT endpoint = Σ gt_delta
           |endpoint| ≤ static_threshold_m(0.8) → routed_mode = 0 (static)
           else                                  → routed_mode = 1 (moving)

[Loss]     reg_loss  = |pred[routed_mode] - GT|  (static GT는 ×static_weight=1, moving GT는 ×moving_weight=5)
                       × query_traj_loss_weight
           cls_loss  = CrossEntropy(mode_logits[Q,2], routed_mode) × query_traj_mode_cls_loss_weight   # binary
           refine_xy = |refine(pred) - GT|        × query_traj_xy_refine_loss_weight
```

로그에 찍히는 loss 키:
- `loss_query_traj` (브랜치 합), `loss_query_traj_moving_reg`, `loss_query_traj_moving_cls`, `loss_query_traj_refine_xy`, `loss_query_traj_static`(static gate off라 0)

---

## 6b. XY Refine (궤적 보정) — 이식용 상세

**선택된 mode의 base 궤적**을 입력으로, 학습형 correction을 sigmoid 게이트로 가산하는 후처리 모듈.
코드: 모듈 생성 `query_head.py` 모델 init(`if query_traj_xy_refine_enabled:` 블록), forward `QueryHead.refine_trajectory_absolute_xy(...)` (※ `refine_trajectory_with_gt_anchor`와 다름).

### 하이퍼파라미터
```python
query_traj_xy_refine_enabled     = True
query_traj_xy_refine_num_layers  = 2      # encoder MLP 깊이 (corr/gate head는 2층 고정)
query_traj_xy_refine_hidden_dim  = 64     # ≤0이면 embed_dim으로 대체
query_traj_xy_refine_loss_weight = 0.1    # 최종값 (warmup이 epoch별로 덮어씀, §4)
# 보정 클립은 trajectory와 공유: query_traj_residual_max_m = (15.0, 15.0)
```

### 서브모듈 3개 (전부 `MLP(use_ln=True, dropout=0)`)
```
encoder    : in = refine_input_dim,         hidden=64, out=64, layers=num_layers(2)
corr_head  : in = 64, out = num_steps*2,     layers=2     # per-step xy 보정 logit
gate_head  : in = 64, out = num_steps,       layers=2     # per-step 게이트 logit

refine_input_dim = (num_past_frames + num_steps)*2 + embed_dim + num_query_classes + 1
                 = (3 + 4)*2 + embed_dim + 8 + 1   # = 23 + embed_dim  (본 설정)
```

### 입력 feature (refine_input 구성, concat)
| 구성 | 차원 | 내용 |
|---|---|---|
| center_context | (num_past+num_steps)*2 = 14 | 과거 center xy(3프레임) + base 미래 궤적 xy(4스텝), 절대좌표 flatten |
| query feature | embed_dim | present query 임베딩 |
| cls_prob | num_query_classes = 8 | 클래스 확률 |
| entropy | 1 | cls 확률의 정규화 엔트로피 (불확실성) |

### Forward 수식
```
base_future_xy = present_center + cumsum(base_offsets)        # 선택 mode의 절대 미래 위치
feat = encoder(refine_input)
corr = tanh(corr_head(feat)) * residual_max_m(15m)            # per-step xy 보정, ±15m 클립
gate = sigmoid(gate_head(feat))                               # ∈[0,1], per-step
refined_xy = clip( base_future_xy + gate * corr, point_cloud_range[:2], [3:5] )   # BEV 범위로 클립
refined_offsets = refined_xy 를 다시 per-step delta로 변환
loss_query_traj_refine_xy = | refined_xy − GT_future_xy | * xy_refine_loss_weight
```
- base 궤적은 유지하고 **refined만 별도 loss**로 학습 (base reg와 분리).
- correction은 ±15m까지 가능하나 gate가 적용량 감쇠 → 실제 보정은 작게.

### ⚠️ warmup 게이팅 — refine은 epoch 5부터 활성
`xy_refine_loss_weight`는 warmup(§4)에서 **epoch 1–4 = 0.0** → 5/8/11에서 0.02/0.05/0.1.
즉 **초기 4 epoch 동안 refine loss·dbg 전부 0이 정상** (모듈은 빌드돼 있으나 학습 신호 없음).

### 모니터링 dbg 키
```
dbg/query_traj_sched_refine_loss_weight   # 현재 적용 weight (epoch<5면 0)
dbg/query_traj_refine_corr_abs_mean       # 원시 보정 크기(|corr|)
dbg/query_traj_refine_gate_mean           # 게이트 평균(적용 비율)
dbg/query_traj_refine_xy_motion_mag_mean  # 실제 적용된 보정 이동량(gate*corr)
dbg/query_traj_refine_xy_valid_count
loss_query_traj_refine_xy
```

### 이식 체크리스트 (refine)
1. 서브모듈 3개(encoder/corr/gate) — `MLP(use_ln=True)` 동일 구조로 생성
2. refine_input feature 4종(center_context/query_feat/cls_prob/entropy) 구성 — **cls_prob·entropy 입력이 있으므로 query 분류 head 출력이 필요**
3. correction 클립 = `tanh × residual_max_m`, 게이트 = sigmoid, 최종 `point_cloud_range`로 BEV 클립
4. refined에만 별도 L1 loss(weight는 warmup으로 epoch 5부터)
5. **선택된 mode 궤적**(argmax 결과)에 적용 — 5모드면 gather 후, 2모드면 동일

> 참고: 코드에 GT-anchor refine(`refine_trajectory_with_gt_anchor`, `query_traj_anchor_*`)도 있으나 본 설정 **미사용**. 여기서 쓰는 건 `xy_refine` 한 종류뿐.

---

## 7. 추론

```
mode_logits[Q,2] → traj_mode_idx = argmax(L0, L1)
  mode0 선택 → 정지 궤적 (0)
  mode1 선택 → mode1 궤적 (+ xy refine 보정)
최종 궤적 = present center에서 per-step delta 누적
```

---

## 8. 필요한 GT 입력 (데이터 파이프라인)

```
gt_instance_centers_world   # [T, N, 3] 프레임별 instance center (world 좌표)
gt_instance_centers_valid   # [T, N]    유효 마스크
gt_instance_ids             # instance 매칭용
gt_occ_inst                 # occupancy instance label
```
- query↔GT instance는 Hungarian 매칭으로 연결 후, 매칭된 쌍에 대해 위 궤적 loss 계산.

---

## 9. 모니터링용 dbg 키 (2-mode)

```
dbg/query_traj_route_mode0_ratio   # static GT 비율 (목표 ~0.67)
dbg/query_traj_route_mode1_ratio   # moving GT 비율 (목표 ~0.33)
dbg/query_traj_pred_mode0_ratio    # static 추론선택  ← route에 수렴해야 정상
dbg/query_traj_pred_mode1_ratio    # moving 추론선택
dbg/query_traj_mode1_endpoint_displacement  # moving 모드 이동량(m)
dbg/query_traj_mode1_straightness_ratio
dbg/query_traj_mode_cls_loss       # binary CE
dbg/query_traj_reg_loss
dbg/query_traj_sched_*             # 현재 적용중인 warmup/TF 값
```
**건강 지표**: `pred_mode1`이 `route_mode1`(~0.33)에 수렴하면 잘 학습된 것. (현재 ~0.11로 과소예측 = §1 한계)

---

## 9b. 학습 시각화 (BEV debug vis)

학습 중 매 `debug_query_vis_every` iter마다 BEV 디버그 이미지를 저장. trajectory는 **하단 2행**에 그려짐.
코드: `query_head.py`의 debug vis 렌더 블록 (`row5: refined traj ...` 헤더, `draw.text(... "base traj")`).

### 파일 / 제어 config
```python
# debug_cfg
debug_query_vis_every = 8     # 8 iter마다 저장 (0이면 off)
# visualization_cfg
debug_query_vis_dir = "./work_dirs/query_debug_vis_no_pretrain"
debug_query_gaussian_vis_mode = 'prob'
# 출력: work_dirs/<run>/vis/<timestamp>/query_debug_vis_no_pretrain/iter_XXXXXX_prob.png
```

### 그리드 구조 (6행 × 7열)
- **열** = 타임라인 7프레임 (t-2, t-1, t, t+1, t+2, t+3, t+4)
- **행**:

| 행 | 내용 |
|---|---|
| row1 | GT class BEV (gt_occ_inst cls) |
| row2 | candidates (all queries, bg+fg) |
| row3 | selected (class-colored) |
| row4 | Hungarian-matched query centers only |
| **row5** | **refined traj** (prev→cur; pred=red, GT=cyan) |
| **row6** | **base traj** (prev→cur; pred=red, GT=cyan) |

> row5/row6 비교 = refine 효과 (epoch<5면 refine off라 동일).

### 색상 (RGB) — 정확히 맞춤
**Legend (공통):**
| 색 | RGB | 의미 |
|---|---|---|
| 흰색 | (240,240,240) | GT occupied BEV |
| 초록 | (40,180,40) | GT overlay |
| 노랑 + | ego_color | ego center (+ marker) |
| 회색 | (140,140,140) | rotated Gaussian prob map |
| 시안 | (30,255,255) | query conf ≥ thr |
| 빨강 | (255,60,60) | query conf < thr |
| 마젠타 | matched_color | Hungarian-matched query |

**Trajectory 행(row5·6) 전용:**
| 색 | RGB | 의미 |
|---|---|---|
| **빨강** | **(255, 48, 48)** | **예측 궤적** (prev→cur 선) |
| **시안** | **(80, 255, 255)** | **GT 궤적** (prev→cur 선) |
| **노랑 숫자** | **(255, 255, 0)** | **선택된 mode index** |

### ★ mode 라벨 읽는 법 (2-mode)
궤적 행에서 각 매칭 query 옆 **노란 숫자 = 선택된 mode**:
- **`0`** = static mode → 모션 0 (점으로 멈춤)
- **`1`** = moving mode → 빨간 궤적 선이 그려짐

> 2-mode에선 0/1만 등장. (5-mode면 0=static, 1·2=straight, 3·4=turn으로 0~4)
> 대부분 `0`이면 over-static(분류기 majority 편향, §1) — dbg `pred_mode0` 높은 것과 시각적으로 일치.

### 읽기 요령
- 빨강(pred) 선과 시안(GT) 선이 겹치면 궤적 예측 정확.
- mode `1`인데 빨강 선이 GT(시안)와 방향/길이 맞으면 moving 예측 양호.
- refined(row5)와 base(row6) 차이 = xy_refine 보정량 (§6b).

---

## 10. 포팅 체크리스트

1. model_cfg에 §3 전체 + `query_pred_num_frames`, frame 설정(§2) 반영
2. `traj_warmup_schedule`(§4) 정의 + `custom_hooks`에 `TrajectoryWarmupHook` 등록
3. teacher forcing 스케줄(§5) — iter 수는 자기 데이터셋 iter/epoch에 맞게 재스케일
4. 데이터 파이프라인이 §8의 GT 키 제공하는지 확인
5. `mode_infer_policy='argmax'` (2-mode는 family_logsumexp 불가)
6. `semantic_routing_enabled=True` 필수 (이게 꺼지면 static_threshold 룰 라우팅 안 됨)

---

## 11. 참고: 모드 개수 바꾸려면

- **5-mode (static/straight/turn family)**: `num_modes=5`, `rule_turn_family_enabled=True`, `mode_infer_policy='family_logsumexp_argmax'` (static=0, straight=1:3, turn=3:5 하드코딩). 단 turn(4%) 추론 collapse 이슈 있음.
- dbg 로깅은 `range(mode_count)`로 자동 적응하므로 모드 수 바꿔도 코드 수정 불필요.
