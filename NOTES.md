# NOTES

## 2026-08-05 KST — 스윕 스크립트 STOP 플래그가 여러 루프를 한꺼번에 죽인다

`r704_traincal.sh` 에서 STOP 플래그 하나를 **통계 동결 루프와 알파 스윕 루프가 공유**하고 있었다.
스윕을 멈추려고 플래그를 세웠더니 4,000 표본 동결까지 같이 취소됐고, 다음 폴링(180초) 때
4,300 이 잡혔다. 누적 accumulator(`_acc: {n, sum, sumsq}`) 구조라 **되감기가 불가능**하다.

→ `traincal_stats/res704_ep18_n4300.json` 은 의도한 4,000 이 아니다. **재수집하지 않기로 함**
  (feat128 에서 n4000 vs n23900 이 future 소수 4자리까지 동일 → 7% 표본 차는 무의미).

**교훈**: 종료 플래그는 **취소해도 되는 작업**과 **끝까지 가야 하는 작업**을 반드시 분리할 것.
수집·동결처럼 되돌릴 수 없는 단계는 별도 플래그를 쓰거나 아예 STOP 을 무시하게 만든다.

---

## 2026-08-03 KST — Depth+Speed train-calibrated eval 이식 시 함정 4개

### ① 🔴 calibration JSON의 checkpoint 경로가 **다른 머신 것**이다

`depth_std+past_speed_calibration/normalization_stats.json`의 `checkpoint` 필드는
`/home/user/jhh/Projects/Autonomous_Driving_26_ikk_fi/epoch_18_lss_only.pth`다. 이 머신 경로가
아니다. 원본 문서(`CODEX_HANDOFF.md` §2.4)는 "stats checkpoint == eval checkpoint"를 계약으로
요구하지만, **경로 문자열 비교로 구현하면 repo를 옮길 때마다 튕긴다.** 검증이 필요하면 realpath가
아니라 md5로 비교할 것.

로컬 `epoch_18_lss_only.pth` md5 = `e6162823d9925a2d407a1399f8cba00c` (292,534,810 B).
**타 머신 원본과 동일한지는 아직 미검증** — 다르면 z-score가 통째로 틀어진다.

### ② `EOCF_EVAL_MODE=2`는 이 repo에 없다

원본 문서는 `MODE=2`(present 1 + future 4 joint volume)를 쓰지만 이 repo의 MODE는 `0=present /
1=future` 둘뿐이다. 게다가 `efficientocf.py`의 주석대로 **MODE는 시각화 기준 프레임에만 영향을
주고 metric은 항상 present/future 둘 다 계산**한다. 따라서 MODE 선택은 metric에 무의미하고,
원본의 `combined`(joint volume) 지표만 재현 불가다. `3D_avg=(future*4+present)/5`로 대체할 것 —
단 **joint volume IoU와 가중평균은 다른 값**이다(원본 CSV에서 0.15269 vs 0.15360).

### ③ `past_speed`의 z는 −3에 도달할 수 없다

`past_speed >= 0` → `log1p >= 0` → z 최솟값이 `(0 − 1.53707)/0.69234 = −2.22`다. 완전 정지
query조차 −2.22가 바닥이다(`depth_std`는 −4.23이라 −3으로 clip됨). 따라서 실효 보정 폭이
**비대칭**이다 — 최대 boost +2.055 / 최대 penalty −2.250.

결과적으로 원래 score의 실효 컷 범위는 **0.536 ~ 0.988**이고, **score 0.98 이상은 depth/speed가
아무리 나빠도 걸러지지 않는다**(logit(0.98)−logit(0.9) = 2.40 > penalty 상한 2.25).
FP를 더 줄이려면 α를 키우거나 clip을 ±3보다 넓혀야 한다.

### ④ depth bin 중심 규칙을 학습과 맞출 것

`depth_std` 계산에 쓰는 bin 중심은 `utils_query_projection.py`의
`depth_vals_d = (arange(D)+0.5)*bin_size + depth_min`과 **같아야** 한다. edge를 쓰면 mu가 반 bin
어긋나 train 통계와 validation feature가 다른 척도가 된다. range도 `query_inst_depth_range_mode`
(`custom` vs `dbound`) 분기를 그대로 따라야 한다.

---

## 2026-08-03 KST — 🔴 **고정 운영점으로 에폭 순위를 매기면 틀린다** + 평가 큐 운용 6원칙

측정 수치는 전부 `RESULTS.md` 로 옮겼다. 여기에는 **다음에 또 밟을 함정**만 남긴다.

### ① 에폭 비교는 반드시 **각자의 최적 운영점**에서 (실제로 뒤집힌 사례)

feat128 ep14 vs ep15 @5119:

| 비교 방식 | 승자 |
|---|---|
| 고정점 fg0.9/occ0.75 | ep14 (0.14716 vs 0.14666) |
| **각자 최적점** | **ep15** (0.15064 vs 0.14716, **+0.00348**) |

ep14 최적은 `fg0.90/occ0.75`, ep15 최적은 `fg0.85/occ0.85` 로 **에폭이 하나 바뀌는 동안 운영점이
이동했다**. 새 체크포인트를 기존 임계값으로만 재보면 저평가된다.
→ **에폭 스크리닝에도 최소 2×2 격자(`fg{0.85,0.90}×occ{0.75,0.85}`)를 편다.**

### ② `occ_thr` 은 fg 가 낮을 때만 일한다

fg0.90 에서는 선택된 query 의 cls 가 전부 0.9~1.0 이라 렌더 가중치가 1.0 과 같아져 occ 가 무영향이다
(dim96·res704 가 이 경우, 편차 ±0.0016). **fg 를 0.85 로 낮춰야 경계선 query 가 들어오고 그때부터
occ 가 작동한다** (feat128 이 이 경우, occ 0.70→0.85 에서 present +0.0036).
→ "occ 무영향"이라는 과거 결론(NOTES 2026-07-31)은 **fg0.90 조건에서만 유효**하다.

### ③ 부분 진행 중인 두 런은 **공통 N 에서만** 비교할 것

eval 순서가 deterministic 이라 같은 N = 같은 샘플이다. 진행률이 다른 두 런의 마지막 줄을 비교하면
엉뚱한 결론이 난다. 실제로 feat128 ep16 의 두 운영점 비교에서 **N=624 에서 −0.0014 이던 부호가
N=864 에서 뒤집혀** 끝까지 유지됐다 — 초반 구간만 보고 판단하면 정반대로 읽는다.
(ep15 는 같은 역전이 N=624 에서 일어났다. 최소 N≈1000~2000 은 봐야 부호가 안정된다.)

### ④ 평가 큐 스크립트 6원칙 (전부 실제 사고에서 나옴)

1. **`set -e` 금지** — 실패해도 `_STATUS`/DONE 마커를 남겨야 '죽은 run'과 '진행 중 run'을 구분한다.
2. **`.out` 은 프로젝트 안에도 복사** — `/tmp` 는 재부팅으로 소실된다(2026-08-02 실제 소실).
3. **워커는 `python -u tools/test.py` 로만 센다** — 부모(`torch.distributed.run`)까지 세면
   1 run 이 2로 잡혀 큐가 영구 대기한다.
4. **`setsid` 로 띄운다** — `nohup ... &` 만으로는 부모 셸 종료 시 프로세스 그룹째 SIGTERM 을 받는다
   (2026-08-03, 로딩 직후 `signal: 15` 사망).
5. **큐는 "프로세스 사망"이 아니라 "슬롯 확보"를 기다린다** — 전자로 짜면 슬롯을 놀린다(~2h 낭비 사례).
6. **TAG 에 모든 변수를 넣는다** — scope 를 빼먹어 `scope=all` 결과가 `scope=future` 로 덮인 적 있다.

### ⑤ 🔴 VIS=1 은 GPU 메모리를 약 2배로 쓴다 — 2개 동시 실행이 재부팅을 유발했다

feat128 + VIS=1 은 run 당 **42.8GB**(VIS off 23.3GB). 2026-08-02 14:01 에 이 조합 2개를 동시
실행(85.6GB, 87%)하다가 NVRM GSP RPC failure → FULLCHIP_RESET →
`CUDA error: unspecified launch failure` → **시스템 재부팅**. CPU/RAM 문제가 아니었다.
→ **대량 런은 전부 VIS=0, 시각화는 최종 승자 1개만 따로.** metric 에는 영향 없다.

---

## 2026-08-03 KST — 입력 해상도를 바꿀 때 **반드시 같이 바꿔야 하는 것 / 몰래 따라오는 것**

`_res704` config 작업(CHANGELOG 2026-08-03)에서 정리한 내용. 다음에 다른 해상도로 갈 때 재사용할 것.

### ① `kv_resolutions`는 선택이 아니라 **강제 종속**이다

img_neck(SECONDFPN, `upsample_strides=[0.25,0.5,1,2]`)이 backbone stride 4/8/16/32를 전부
**stride-16으로 통합**한다. 따라서 context feature = `input_size/16`이고,
`transformer.py:236-240`이 이것과 `kv_resolutions[-1]`의 불일치를 `ValueError`로 하드 검증한다.

    (896,1600) → (56,100)      (256,704) → (16,44)

**피라미드 중간 단계는 마지막 값의 약수로 잡을 것.** 정수배면 `avg_pool2d` 정확 경로,
아니면 `adaptive_avg_pool2d` fallback으로 조용히 넘어가 pooling 성질이 달라진다
(`transformer.py:247-253`). 16→8→4 / 44→22→11은 정수배 OK.

`learnable_kv_downsample`은 `efficientocf_config.py:217` 기본 **False** — 이 경우 pooling에
파라미터가 없다. True로 켤 때만 `kv_full_resolution` 의존 파라미터가 생겨 해상도 변경이
ckpt 호환을 깬다.

### ② ⚠ **crop이 몰래 따라 바뀐다** — 해상도 실험의 숨은 두 번째 변수

`loading_bevdet.py:190-215` `sample_augmentation`은 base resize를 `fW/W = fW/1600`으로 잡고
`crop_h = int((1-crop_h_jitter)*newH) - fH`로 **상단을 잘라낸다**(bottom crop 고정).

| input_size | base resize | newH | crop_h | 세로 FOV 유지율 |
|---|---|---|---|---|
| (896,1600) | 1.0 | 900 | 4 px | 99.6% |
| (256,704) | 0.44 | 396 | **140 px** | **64.6%** |

즉 704로 내리면 해상도만 주는 게 아니라 **세로 시야 상단 약 35%가 사라진다.** 가까운
트럭/버스 상단이 이미지 밖으로 나가므로 GMO/attn GT의 경계 clip 비율이 올라간다.
BEVDet 256×704 표준 레시피 그대로라 버그는 아니지만, **"해상도만 바꾼 단일변수 실험"이라고
판정문을 쓰면 안 된다.** 순수 해상도 효과만 보고 싶으면 `crop_h` jitter를 조정해 FOV를 맞춰야 한다.

부수: `resize` jitter가 base 아래로 내려가면(`resize<0.44`) `newW<704`가 되어 crop 우측이
이미지를 벗어나 **검은 패딩**이 생긴다. before(896)에서도 같은 성질이라 새로 생긴 문제는 아니다.

### ③ 해상도에 딸려가는 것 / 안 딸려가는 것

**전부 정확히 같은 비율로 감소** (stride-16 하나에 묶여 있음): 입력 픽셀, backbone 전 stage,
context 토큰, KV 토큰 총합, LSS frustum ray 수, depth loss 격자. 704는 896 대비 **1/8**.

**안 변함**: `occ_size`, `point_cloud_range`, `grid_config`, `bev_feat_dim`,
`query_matched_gmo_bce_occ_size`, gaussian/dice/tversky/traj 전부.
→ **BEV·loss 격자는 그대로이고 그 격자를 채우는 이미지 증거 밀도만 준다.**

**파라미터 shape은 해상도 무관** — FCN backbone/neck + `build_2d_sincos_pos_embed`가 매 forward
생성하는 sin-cos pos embed(학습 파라미터 아님) + 개수 고정 `query`/`query_id_embed`/`cam_id_embed`.
따라서 해상도가 달라도 **ckpt shape은 호환**된다(성능 보장과는 별개).

### ④ ⚠ 픽셀 단위 하이퍼파라미터는 자동으로 안 따라온다

`query_attn_sigma_match_min_mask_px=4`는 **896 기준 픽셀값**이다. 704에서는 GT cam 마스크가
픽셀 기준 ~2.3배 작아지므로 같은 값이 훨씬 공격적인 필터가 된다. `_res704`에서는
`query_attn_sigma_match_loss_weight=0.0`으로 축 자체를 꺼서 회피했으나(weight 0이면 loss 항
미생성 → `start_iter`/`min_mask_px` 동반 무효화), **되살릴 때 반드시 재스케일할 것.**

**σ-match를 꺼도 attn 감독이 다 꺼지는 게 아니다**: `use_query_attn_bbox_loss=True`(inside-mass),
`query_attn_bbox_other_weight=0.3`, 매칭 cost `inside_log`(0.3)는 그대로 살아 있고 이제
카메라당 (16,44)=704칸 격자에서 계산된다. 원거리 객체가 1칸 미만이 되어 inside-mass/soft-IoU가
거칠어지는 것이 저해상도 실험의 **주 리스크**(kv 레이어 수보다 이쪽과 LSS 밀도가 먼저 온다).

### ⑤ `visualization_cfg`의 4개 dir 문자열은 **죽은 값이다**

config에 뭘 적든 무시된다. 학습은 `mmdet_train.py:141` → `_configure_visualization_dirs()`가
`{work_dir}/vis/{timestamp}/{folder}`로 덮어쓰고(`mmdet_train.py:60-86`), eval은
`efficientocf.py:1333`이 `EOCF_EVAL_VIS_DIR`(기본 `./work_dirs/eval_vis`)를 쓴다.
**run 격리는 `work_dir`과 `EOCF_EVAL_VIS_DIR`로 하는 것이지 config 경로 rename으로 하는 게 아니다.**
경로 외 옵션(`gaussian_vis_mode`, `max_frames`, `topk_matched`, `max_queries`, `eval_occ_size`)은 실동작.

## 2026-07-31 KST — [실측] traj_acc 꼬리컷 = **유일하게 성공한 추론 손잡이**. 봉우리는 10~20% 제거

`traj_acc = |c₂−2c₁+c₀|`(과거 3프레임 xy 이차차분 = 가속도 잔차) 상위 꼬리를 fg 선택집합에서
**제거**한다. `EOCF_EVAL_FG_TRAJ_ACC_MAX=<m>`(기본 0=비활성). 구현/근거는 CHANGELOG 2026-07-31 참조.

**⚠ 재랭킹이 아니라 제거다.** 같은 traj_acc를 점수 가중치로 쓰는 4가지 방식은 전부 실패했고
(같은 날 아래 항목 참조), **버리는 축으로 쓸 때만 작동한다.** cls_prob은 "객체인가"를 재고
traj_acc는 "이 중심을 믿을 수 있나"를 재는데, IoU는 복셀 겹침으로 채점되므로 후자가 별도로 유효하다.

### ★ ep18 @5119 확정 곡선 — **봉우리 = 10% 제거(acc_max 4.85m)**

| 제거율 | acc_max(m) | IoU3d_future | baseline 대비 | IoU3d_present |
|---|---|---|---|---|
| 0% (baseline) | — | 0.1412 | — | (구코드 로그라 미측정) |
| 5% | 6.97 | 0.1432 | +0.0020 | 0.1791 |
| **10%** | **4.85** | **0.1439** | **+0.0027** | **0.1786** |
| 15% | 3.64 | (측정 예정) | | |
| 20% | 2.94 | 0.1437 | +0.0025 | 0.1763 |
| **30%** | **2.09** | **0.1407** | **−0.0005** | 0.1713 |

- **30%는 baseline보다 나쁘다.** 봉우리를 지나면 급격히 무너진다.
- **⚠ 800샘플 스크리닝은 봉우리 위치를 틀리게 짚었다**: @800에서는 20%(0.1482) > 10%(0.1477)였으나
  @5119에서는 10%(0.1439) > 20%(0.1437)로 뒤집혔다. 두 경우 모두 차이가 노이즈 수준이라
  **봉우리 확정은 5119로만 가능하다.** 800은 "이득이 있는 대역"까지만 알려준다.
- **10%가 단일 운영점으로 최적**: future 최고이면서 present 손실도 최소(5% 대비 −0.0005).
  20%는 future를 5%p 더 못 얻으면서 present를 0.0023 더 잃는다.

**★ 한계 유용비 예측이 실측으로 검증됐다.** 아래 표에서 20~30% 밴드 유용비 0.481 > 전체 평균 0.474 →
"그 구간부터 이득 소멸"이라 예측했고, 실측이 정확히 그 지점에서 꺾여 baseline 아래로 떨어졌다.
→ **앞으로 유사한 필터링 아이디어는 eval(3시간) 전에 덤프로 몇 분 만에 사전 판별 가능하다.**

**ep18 @800 컷 곡선** (스크리닝용. 절대값은 낙관 편향, 봉우리 위치는 신뢰 불가)

| 제거율 | acc_max(m) | IoU3d(asset)_future | 증분 |
|---|---|---|---|
| 0% (baseline) | — | 0.1450 | — |
| 5% | 6.97 | 0.1469 | +0.0019 |
| 10% | 4.85 | 0.1477 | +0.0008 |
| **20%** | **2.94** | **0.1482** | +0.0005 |
| 30% | 2.09 | 미실측(아래 근거로 스킵) | — |

- **ep20에서도 재현**: 0.1429 → 0.1447 (acc6.97, +0.0018). 기전도 동일 —
  ep18 TP −1.2%/FP −6.1%, ep20 TP −1.0%/FP −6.0%. **FP를 6% 버리고 TP는 1%만 잃는다.**

**한계 유용비 (ep18 N=300 덤프, fg0.9 선택집합 3,704개 / TP 1,756 / precision 0.474)**

| 밴드 | acc 범위(m) | 한계 유용비 |
|---|---|---|
| 0~2% | >11.16 | 0.147 |
| 2~5% | 6.97~11.16 | **0.126** (가장 나쁨) |
| 5~10% | 4.85~6.97 | 0.335 |
| 10~15% | 3.64~4.85 | 0.368 |
| 15~20% | 2.94~3.64 | 0.400 |
| **20~30%** | 2.09~2.94 | **0.481 > 평균 0.474** |
| 30~40% | 1.58~2.09 | 0.461 |

- **20%를 넘으면 밴드 유용비가 전체 평균에 도달한다** = 그때부터는 '나쁜 걸 골라 버리는' 게 아니라
  예산을 무작위로 깎는 것과 같아진다. 예산 감소 자체는 손해이므로(아래) **30% 이상은 볼 필요 없다.**
- 실측 증분(+0.0019 → +0.0008 → +0.0005)이 이 표와 정확히 같은 방향으로 감쇠한다.

**⚠ "자를수록 좋다"가 아니라 "제대로 고른 걸 자를 때만 좋다".** 같은 세션 실측 대조:

| 방식 | 버린 양 | 결과 |
|---|---|---|
| cls threshold 0.90→0.92 | ~19% | 0.1429 → **0.1391 (−0.0038)** |
| traj_acc 상위 10% 컷 | 10% | 0.1450 → **0.1477 (+0.0027)** |

둘 다 query를 버리는데 방향이 반대다. 이득의 원천은 예산 축소가 아니라 **traj_acc의 선별력**이다.

**⚠ 컷 값은 절대값(m)이고 체크포인트마다 재캘리브레이션해야 한다.** 위 acc_max는 **ep18 N=300**
덤프의 fg0.9 선택집합 백분위: 2%=11.16 / 5%=6.97 / 10%=4.85 / 15%=3.64 / 20%=2.94 / 30%=2.09.
ep20에 그대로 적용해도 이득은 났지만 '상위 N%' 라벨은 부정확해진다.

**⚠ offline 판정식이 이 아이디어를 죽일 뻔했다.** 제거 이득 조건을 미보정 임계(~0.123)로 재면
5% 제거의 유용비 0.134가 "경계선~소폭 손해"로 나온다. 실측은 +0.0019 이득이었다.
**추가 방향의 0.635 기준과 달리 제거 방향의 임계는 캘리브레이션된 적이 없다** — 판정에 쓰지 말 것.

## 2026-07-31 KST — [실측 N=300] attn soft-argmax vs hard-peak 괴리 = 닫힘. 3D 버전은 거리의 열화판

아이디어: attention map에서 soft-argmax로 중심을 잡는데, **최대峰(hard argmax)과 멀어지면**
중심을 잘못 잡은 것 아닌가. 거리는 두 가지 — 이미지 평면(2D)과 depth lifting 후(3D).
계측 구현: `EOCF_QFEAT_ATTN_PEAK=1`(기본 OFF) → `utils_query_projection.py`의
`_build_query_attn_soft_lift_pack`에서 산출, `_last_query_attn_soft_lift_pack` 경유로 qfeat 덤프에 실림.
덤프: `eval/qfeat_18ep_N300_attnpeak/qfeat` (ep18, fg0.9/occ0.75/nms0, feature 격자 **56×100**).

**3D 거리는 2D의 독립 인자가 아니다(해석적)**: 두 점이 같은 depth를 쓰고 회전이 노름을 보존하므로
`||Δ3D|| = depth · √((Δu/fx)² + (Δv/fy)²)` — 즉 **3D = 2D × depth/focal**. 거리 가중된 2D일 뿐이다.

| 인자 | 잔차화 AUC(⊥cls,dist) | ρ(cls) | ρ(dist) | ρ(xy중심오차) | 예산별 재랭킹 |
|---|---|---|---|---|---|
| d2d (feat/img, 동일) | 0.5166 | **−0.081** | −0.179 | **+0.096** | ~0 (fg0.95만 +0.011) |
| d3d | 0.5232 | −0.301 | +0.193 | +0.339 | ~0 (fg0.95만 +0.015) |
| peak_prob | 0.5073 | +0.115 | +0.175 | −0.030 | **전 예산 음수(−0.04~−0.13)** |
| (참고) traj_acc | 0.5668 | −0.349 | 0.183 | +0.424 | 음수 |
| (참고) dist | 0.5322 | −0.480 | 1.0 | **+0.381** | 음수 |

- **계측 자체는 의도대로 작동한다**: `spearman(peak_prob, d2d) = −0.552` — map이 뾰족할수록 괴리가 작다.
  배선/계산 오류가 아니라 **가설이 안 맞는 것**이다.
- **⚠ 가설 "괴리 크면 중심이 틀렸다"는 2D에서 성립하지 않는다.** ρ=+0.096이고 4분위가 **비단조**:
  Q1 1.524m / Q2 1.248m / Q3 1.385m / Q4 2.076m — **괴리가 가장 작은 Q1이 Q2·Q3보다 오히려 나쁘다.**
  (괴리 0 = attention이 한 칸에 붕괴한 퇴화 케이스로 보임. 양극단이 모두 나쁨 → GBDT가 LR보다
  약간 더 건지지만 그래도 부족.)
- **⚠ d3d의 상관(+0.339)은 괴리가 아니라 거리에서 온 것이다.** 거리 단독이 +0.381로 **더 높다**.
  즉 **d3d = 거리에 노이즈를 더한 열화판**. 3D로 올리는 것은 이득이 아니라 손해다.
- **왜 실패하는가**: peak 확률 중앙 **0.141** — 5,600칸(56×100) 중 최대칸이 질량의 14%만 갖는다.
  map이 너무 퍼져 있어 **hard argmax 한 칸이 "진짜 위치"의 신뢰할 만한 대안이 아니다.**
  전제가 깨진다.
- 5-fold: `cls+d2d` 1/5, `cls+d3d` 3/5, `cls+peak_prob` 1/5 — 전부 부호 불일치.
  `cls+신규4+acc`가 5/5(0.4802)지만 **`cls+acc` 단독(0.4818)보다 낮다** → 신규 4개는 순 기여 0.
- **추가분 유용비 최고 0.251** (d3d, cls≥0.85, q=0.7) ≪ 기준 **0.635** → **닫힘.**

**→ attention 기반 인자는 이걸로 정리. 재시도 금지.** 남은 인자 후보가 있다면 "attention이 퍼져
있다"는 사실 자체를 우회하는 것이어야 한다(예: map을 쓰지 않는 신호).

## 2026-07-31 KST — [실측 N=300] fg score 신규 인자 3축 탐색: `traj_acc`만 살아남음(단 크기 미미)

기존 덤프 `eval/qfeat_18ep_N300_depth/qfeat`(epoch_18, 60,000 query / GT 3,078) 재분석.
스크립트: scratchpad `qfeat_probe.py` / `qfeat_newfeat.py` / `qfeat_incremental.py` / `qfeat_cv.py`.
**파이프라인 검증**: cls_prob AUC **0.9527**(NOTES 대조 0.953), fg0.9 선택 3,704 / precision 0.4741 /
recall 0.5705 — 선행 분석 수치를 재현하므로 아래 신규 결과도 같은 기준으로 읽을 수 있다.
GT 프레임 정합은 실측으로 `t=2`(=`present_idx`)에서 최소거리 — 좌표계 정합 확인됨.

**탐색한 미탐색 3축** (선행 분석이 안 본 것만):
- **A. mixture 48-component 자기일관성** — 가중중심과 query 중심 불일치(`mix_off`), 성분 xy 산포
  (`mix_spread`), 가중치 엔트로피(`mix_went`)
- **B. trajectory 3프레임 기하** — 가속도 잔차 `|c₂−2c₁+c₀|`(`traj_acc`), 속도(`traj_vel`)
- **C. depth 분포 다봉성** — 주봉 ±3bin 밖 질량(`dep_multi`). entropy가 못 가르는 '넓은 단봉 vs 이봉'

**잔차화 AUC** (경계밴드 0.5≤cls<0.98, cls·dist 회귀 제거 후 남는 판별력):

| 인자 | 원본 | ⊥cls | ⊥cls,dist | ρ(dist) |
|---|---|---|---|---|
| **traj_acc** | 0.6239 | **0.5669** | **0.5668** | 0.183 |
| traj_vel | 0.5891 | 0.5377 | 0.5380 | 0.145 |
| dep_multi | 0.6383 | 0.5230 | 0.5116 | 0.182 |
| sharp | 0.6374 | 0.5226 | 0.5143 | −0.107 |
| mix_off / spread / went | 0.53~0.55 | 0.509~0.519 | 0.506~0.518 | 0.18~0.22 |

- **A축(mixture 자기일관성)과 C축(다봉성)은 닫혔다**: cls 제거 후 0.51~0.52로 붕괴 = cls의 재표현일 뿐.
- ⚠ **이들은 거리 대리변수가 아니다**(ρ(dist) 0.11~0.22). 선행 분석의 'range-bin 정규화 실패'와는
  **다른 이유로** 실패한 것 — cls_prob과 중복이기 때문. 거리 가설로 오해하지 말 것.
- **`traj_acc`만 cls·dist를 모두 제거하고도 0.567 유지.** 유일하게 독립 정보가 있다.

**5-fold CV 고정예산 precision** (샘플 단위 분할, budget≈740/fold):

| 구성 | 로지스틱 | GBDT | fold 부호일치 |
|---|---|---|---|
| 기준 (cls 원값 랭킹) | 0.4718 ±0.0163 | — | — |
| **cls + traj_acc** | 0.4794 ±0.0148 | **0.4818 ±0.0161** | **GBDT 5/5** |
| cls + traj_acc + traj_vel | 0.4788 | 0.4810 | GBDT 5/5 |
| cls + 전체9 | 0.4761 | 0.4850 | GBDT 4/5 |

- **`traj_acc`는 노이즈가 아니다** — GBDT 기준 5/5 fold 전부 개선, 평균 **+0.0100 (상대 +2.1%)**.
- ⚠ **단일 holdout(±0.0116 SE)만 봤으면 노이즈로 오판했을 크기다.** 반드시 fold 부호로 볼 것.
- ⚠ **단순 GD(300 iter) 로지스틱은 미수렴이라 "변수 추가 = 전부 하락"이라는 가짜 결론을 준다.**
  실제로 1차 시도에서 그렇게 나왔다. lbfgs/GBDT로 재확인할 것.

**⚠ 그래도 추론 손잡이로는 권하지 않는다.** 선행 실험에서 depth sharpness가 offline precision
**+11%**를 내고도 IoU는 동률(macro −0.5% / micro +1.2%)이었다. `traj_acc`의 **+2.1%**는 그보다
5배 작으므로 IoU 전이 기대값은 사실상 0이다. 예산 740개 중 7~8개가 교체되는 수준.

**[추가 실측] `traj_acc`를 fg score에 넣는 4가지 방식 — 전부 닫힘. 재시도 금지.**
스크립트: scratchpad `traj_score.py`.

| 방식 | 결과 |
|---|---|
| 단순 곱셈 `cls/(1+acc)^k` (k=0.25/0.5/1.0) | fg 0.80/0.85/0.90/0.95 **전 예산 음수** (−0.0013 ~ −0.0192) |
| 고정예산 재랭킹 (최적 k=0.1) | 620개 교체 → TP 183 유입 / 182 유출, **순증 1개** (GT 3078 대비 +0.0003) |
| 학습된 결합 (GBDT) | +0.010 precision (5/5 fold) — 실재하나 IoU 전이엔 부족 |
| 2차 게이트 (`cls≥0.85 & acc<중앙`) | 추가분 유용비 **0.272**, 최선 조합도 **0.442** ≪ 필요치 **0.635** |

- **기전**: 고정예산 교체 620개의 **유입 유용비 0.295 vs 유출 0.294** — 같은 품질끼리 자리만 바꾼다.
  `traj_acc`의 잔여 정보가 예산을 움직일 만큼 크지 않다.
- **2차 게이트 검증**: cls를 0.85로 낮추고 acc 하위 50%만 받으면 recall 0.5705→0.6257이지만
  추가 625개의 유용비가 0.272라 IoU는 떨어진다. 실측 fg 스윕(fg0.85 = 0.1374~0.1381
  < fg0.9 = 0.1430)과 정확히 일치 — **판정식과 실측이 서로를 확증한다.**
- ⚠ 단순 곱셈은 실패하는데 GBDT는 소폭 이득이 난다 = 유용한 형태가 **acc에 대해 단조가 아니다**.
  그래도 그 이득이 예산 교체를 이길 만큼은 아니므로 결론은 동일하다.
- **결론: 선행 기록의 "예산을 고정하면 cls_prob를 이기는 스코어 조합이 없다"가 trajectory 축까지
  확장 확인됐다.** 추론 재랭킹은 완전히 닫힌 문이다.

**→ 실질적 함의는 추론이 아니라 학습이다.** `traj_acc`(궤적 가속도 잔차)가 xy중심오차와
**ρ=+0.424**로 붙는다는 것은 "**시간적으로 흔들리는 query가 곧 중심이 틀린 query**"라는 뜻이다.
선행 결론(병목 = 중심 정확도, 추론에서 수정 불가)과 정확히 같은 지점을 가리키며,
**중심 궤적에 시간 일관성(constant-velocity 잔차) 정규화를 거는 학습 손실**이 근거 있는 다음 수다.
추론에서 재랭킹으로 뽑아 쓸 정보가 아니라, 애초에 그 흔들림을 없애는 쪽이 맞다.

## 2026-07-31 KST — [실측] fg threshold 봉우리 확정(0.9). occ는 5119에서도 무영향

epoch_20_lss_only, `eval_total.sh` 조건 고정(MODE=future / NMS off / weight=ones), fixed_val.
2026-07-30 스윕(epoch_18, N=300)이 **fg 0.9 아래쪽만** 봤던 것을 **0.95 위쪽까지 확장**해 봉우리를 가뒀다.

**800샘플 스크리닝** — `IoU3d(asset)_future`

| fg \ occ | 0.7 | 0.75 | 0.8 |
|---|---|---|---|
| 0.85 | 0.1365 | 0.1374 | 0.1381 |
| **0.9** | 0.1426 | 0.1429 | **0.1430** |
| 0.95 | 스킵 | **0.1124** | 스킵 |

- **fg 곡선은 0.9에서 봉우리, 그리고 비대칭이다.** 0.9→0.85는 −0.005(완만), 0.9→0.95는
  **−0.031(급락)**. 0.95에서 TP가 1.68e7→0.98e7로 **−42%** 날아간다(FP는 5.36e7→2.07e7로 줄지만
  판정식 `ΔTP/(ΔTP+ΔFP) > IoU` 기준으로 전혀 보상이 안 됨). **0.9 위로는 올리지 말 것.**
- 2026-07-30 기록의 fg 0.75(0.1256) / 0.5(0.0853)와 합치면 봉우리 양쪽이 모두 닫혔다.
  → 남은 탐색 여지는 **0.88~0.92 구간의 0.01 단위뿐**이고, 그 밖은 볼 필요 없다.
- **occ_thr(0.7/0.75/0.8)은 fg 0.9에서 편차 0.0004, fg 0.85에서 0.0016.** 무영향 재확인.
  ⚠ occ를 0.01 단위로 쪼개는 건 노이즈만 본다. **미세조정은 fg 축에만 걸 것.**

**5119 전체 검증**: `fg0.9/occ0.8` → **0.1407**. 별도로 돌린 `fg0.9/occ0.75` → **0.1406**(차이 0.0001).
occ 무영향이 부분표본 아티팩트가 아님이 전체 샘플에서 확정됐다.

**⚠ 코드 버전 등가성(중요)**: 위 0.1406은 `nusocc/bbox_aabb/bbox_rot/Recall3d`가 살아있던
**커밋 HEAD 코드**의 출력이고, 0.1407은 그 지표들이 제거된 **현재 워킹트리(test.py 리팩터)** 출력이다.
리팩터는 `iou_3d_asset` → `iou_3d_asset_{present,future}` **분리**일 뿐이며, `MODE=1`에서
옛 `IoU3d(asset)` ≡ 새 `IoU3d(asset)_future`임이 위 두 수치로 **실측 확인**됐다.
→ 구버전 로그의 `IoU3d(asset)` 값을 현재 코드 결과와 직접 비교해도 된다.

**⚠ 800샘플 대표성**: ref(`fg0.9/occ0.75`) 800샘플 = 0.1429 vs 5119 = 0.1406. **+0.0023 낙관 편향**이
있으나 조합 간 *순위*는 보존된다. 스크리닝용으로는 충분하지만, **절대값 기준(예: 0.13 컷)을
5119 수치에서 가져와 800에 적용할 때는 이 편향을 감안할 것.**

**⚠ 실행 환경**: eval은 conda env `eof`가 필요하다(`mmcv 1.7.2` / `torch 2.7.0+cu128`).
nohup/스크립트로 띄우면 base env로 떠서 `ModuleNotFoundError: mmcv`로 죽으므로
`source ~/anaconda3/etc/profile.d/conda.sh && conda activate eof`를 반드시 넣을 것.

**⚠ 동시 실행 상한은 GPU가 아니라 CPU다**: eval은 GPU util 5~7%로 **CPU 바운드**
(`workers_per_gpu=1`, run당 ~690% CPU = 6.9코어). 24코어 머신에서 2개 동시 → loadavg 21~22로 이미 포화.
GPU 메모리는 VIS off 17GB / VIS on 34GB per run이라 3개도 들어가지만, **CPU 초과구독 + swap(이미 4.7G/8G)
때문에 3개는 처리량 이득 없이 OOM 위험만 커진다. 2개가 상한.**
`EOCF_EVAL_VIS=1`은 48샘플마다의 렌더 외에 **매 샘플** attn/cam-gaussian 디버그 텐서를 계산하므로
(efficientocf.py의 `return_instance_img_debug` / `return_query_cam_gaussian_vis_debug`)
스크리닝에서는 순수 오버헤드다 — 끄면 GPU 메모리가 34GB→17GB로 반감된다(metric에는 무영향).

## 2026-07-30 KST — [실측 완결] 추론 손잡이 전수 스윕 = 현재 운영점이 최적. 남은 건 재학습뿐

epoch_18, N=300 고정, NMS off. 모두 같은 fixed_val 300샘플이라 직접 비교 가능.

| run | fg | occ | weight | IoU3d_fut | micro | TP | FP |
|---|---|---|---|---|---|---|---|
| **A** | 0.9 | 0.75 | score | **0.1463** | 0.1447 | 6.52e6 | 2.10e7 |
| E | 0.9 | 0.75 | ones | 0.1458 | 0.1442 | 6.71e6 | 2.24e7 |
| **F** (=`eval_total.sh`) | 0.9 | 0.8 | ones | 0.1461 | 0.1442 | 6.49e6 | 2.10e7 |
| G | 0.9 | 0.8 | score | 0.1464 | 0.1444 | 6.26e6 | 1.93e7 |
| sharp pow2 | — | 0.75 | score | 0.1455 | 0.1465 | 6.06e6 | 1.73e7 |
| B | 0.75 | 0.75 | score | 0.1256 | 0.1272 | 8.42e6 | 4.22e7 |
| C | 0.75 | 0.75 | ones | 0.1195 | 0.1206 | 8.98e6 | 5.04e7 |
| D | 0.5 | 0.5 | ones | 0.0853 | 0.0859 | 1.11e7 | 1.05e8 |

- **fg0.9 근방 4조합이 0.1455~0.1463(편차 0.6%)** — `weight_mode`(ones/score)와 `occ_thr`(0.75/0.8)은
  fg0.9에서 **무영향**이다. 선택된 query의 cls가 전부 0.9~1.0이라 렌더 가중치가 1.0과 거의 같기 때문.
  → **`eval_total.sh`의 ones/0.8 설정은 손해가 아니다. 바꿀 이유 없음.**
- **완화 계열은 전부 단조 손해**: B −14% / C −18% / D −42%. TP는 6.5→11.1e6(+70%) 늘지만
  FP가 2.1→10.5e7(**+400%**)로 6배 빠르게 늘어난다.
- **판정식**: IoU는 `ΔTP/(ΔTP+ΔFP) > 현재 IoU`일 때만 오른다(FN이 ΔTP만큼 줄므로).
  TP/FP 비율(0.31)이 아니라 **IoU 자체(0.1447)** 가 기준선이다. B의 실측 마진 유용비율은 0.0825로
  예측치 0.074와 12% 오차 내 일치 — **offline 겹침 proxy(3.65배 보정) 프레임워크는 신뢰 가능하다.**
  ⚠ 이 기준의 실무적 함의: **추가할 query 집합은 "지금 뽑고 있는 것과 비슷하게 좋아야" 한다**
  (`dmin<2m` 비율 > 0.635). 그래서 "무엇을 더 넣는" 계열은 원리적으로 이기기 어렵다.
- **σ/offset 균등 팽창(z 팽창) 아이디어도 이로써 폐기**. C/D가 사실상 그 실험이다.
  "IoU2d/IoU3d 격차로 z에 +43% 여지"는 상한 계산일 뿐 균등 팽창으로는 도달 불가.

### depth sharpness (`EOCF_EVAL_FG_SHARP_POW`) 결론: 원리적으로 닫힘
- `cls * sharp^2` (예산·실효렌더 모두 A와 정합): macro 0.1455(−0.5%) / micro 0.1465(+1.2%) = **동률**.
  성분은 TP −7% / FP −18%로 **단순 부피 감소**였다 — "더 잘 골라서" 오른 게 아니다.
- offline precision +11%가 IoU로 전이되지 않은 이유 2개:
  ① `weight_mode=score`에서 교체된 query는 cls가 낮아(중앙 0.841) **렌더 부피까지 함께 줄어든다**
     → 개수를 고정해도 부피가 고정되지 않아 단일변수가 깨진다.
  ② **`sharp`는 사실상 거리 정보다.** 대역별 sharp 중앙: 0–20m **0.513** / 20–30m 0.325 /
     30–40m 0.315 / 40m+ 0.338. **30m 밖에는 sharp≥0.5인 query가 0개**(최대 ~0.44).
     depth bin이 2~58m/64개(0.875m)라 원거리에서 depth 확신도를 얻는 것 자체가 불가능하다.
     cls가 이미 거리 정보를 갖고 있어 곱해도 새 정보가 없다.
- **"먼 것 중 날카로운 것만 추가"도 실패**: 30m+ 상위 1% sharp의 `dmin<2m`=0.600 < 0.635(기준).
  40m+는 0.100. 통과하는 건 20m+ 상위 5%(140개)뿐이고 IoU 기여 +0.4%로 노이즈.
  이유: 원거리 병목은 **검출 확신도가 아니라 위치 정밀도**(30m+ 중심오차 2.2~2.6m)이고,
  depth sharpness는 "거기 뭔가 있다"만 알려주고 "정확히 어디"는 못 알려준다.

## 2026-07-30 KST — [실측 N=300] fg score 재설계는 닫힌 문 / 병목은 중심정확도·크기

덤프: `work_dirs/full_.../eval/qfeat_18ep_N300_depth/qfeat` (epoch_18, fg0.9/occ0.75/nms0,
query 60,000개 = 300샘플 × 200query, GT 인스턴스 3,078개). 분석 스크립트는 scratchpad
`analyze_qfeat.py` (session-local — 재현 필요하면 이 NOTES의 수치로 대조할 것).

- **선택 recall 0.570 / precision 0.474**. 매칭된 query 1,325개(43%)가 fg0.9에서 탈락하고,
  선택된 3,698개 중 1,945개는 어떤 GT와도 매칭되지 않는다.
- **탈락 원인은 거리다(크기 아님)**: `spearman(score, ego거리) = -0.597`,
  `spearman(score, GT부피) = -0.137`. fg0.9 통과율은 0–20m 0.808 → 20–30m 0.647 →
  30–40m 0.392 → 40m+ **0.129**.
- **그러나 score는 miscalibrated가 아니다 — 위치정확도 추정기다.**
  `spearman(score, xy중심오차) = -0.397`. 탈락 pair는 xy중심오차 중앙 **2.42m**(<2m 비율 0.43),
  통과 pair는 1.16m(0.73). 렌더 블롭이 2.4×5.0m이므로 탈락 pair는 블롭 절반이 객체 밖이다.
  → **"억울한 탈락"이 아니라 "실제로 절반 어긋난 것"**. fg_thr을 내리면 TP/FP를 섞어 받는다.
- **⚠ range-bin 정규화(거리별 threshold)는 크게 손해다**: AP 0.476 → 0.189(평균정규화)/0.268(백분위).
  score의 거리 의존성 = 위치정확도의 거리 의존성이라, 정규화하면 실제 정보를 지운다. **재시도 금지.**
- **depth 분포 sharpness의 부호는 직관과 반대다**: normalized entropy가 matched 0.582 <
  unmatched 0.679 — **날카로운 쪽이 진짜**다. "뭉툭=하드샘플이니 가점" 은 FP를 부스트한다.
  신호는 있으나(AUC[-entropy] 0.877 단독, cls<0.9 조건부 0.808) cls_prob(0.953)와 겹쳐
  더해도 AP 0.4762 → 0.4742 (무이득).
- **큰 블롭은 거의 매칭되지 않지만 이미 걸러져 있다**: 블롭부피 4분위 매칭률
  0.163/0.029/0.010/**0.005**, 4분위 score 평균 0.037 · fg0.9 통과율 0.001.
  부피 penalty를 넣어도 동일 예산에서 recall 0.571→0.572 (노이즈).
- **결론: 예산을 고정하면 cls_prob를 이기는 스코어 조합이 없다.** 추론에서 남은 실질 손잡이는
  threshold뿐. cls 스윕 곡선(선택 recall/precision): 0.95→0.294/0.554, **0.90→0.571/0.474**,
  0.80→0.773/0.377, 0.75→0.817/0.353, 0.60→0.896/0.296, 0.50→0.928/0.269.
- **z 위치는 전 거리대에서 정확하다** (중심 z오차 중앙 0.13~0.19m). 따라서 IoU2d/IoU3d 격차(0.70)는
  z 배치가 아니라 **z extent(높이)만 짧은 것**이다.
- 파생 결론: 먼 객체의 진짜 병목은 **중심 정확도**이고 추론에서 고칠 수 없다.
  `_maybe_dump_eval_query_features`의 `EOCF_EVAL_CAM_SCORE`/`fg_score_cam_attn_weight` 경로도
  같은 이유로 기대값이 낮다.

**⚠ 덤프 텐서 레이아웃 함정**: `gt_instance_centers_world`는 **`[1, T7, N, 3]`
(batch, frame, instance, xyz)** 이고 `gt_instance_centers_valid`는 `[T7, N]`이다.
`[N,T,1,3]`으로 착각하면 `i==0`인 pair만 집계되고 **조용히 통과한다**(에러 없음).
실제로 이번 분석에서 1차로 이 버그가 났고, 유효 pair가 1,126개 대신 94개로 줄었다.
`gt_instance_sizes`는 `[N,3] = (w,l,h)`로 인스턴스 직접 인덱싱이 맞다.

## 2026-07-30 KST — `EOCF_EVAL_DUMP_QFEAT` (per-query 덤프) 사용 시 주의사항

- **기본 OFF.** 켜면 `_eval_train_faithful_inst_match()`가 매 샘플 추가로 돌아 eval이 느려진다
  (oracle 실행과 동일 비용). 또 `extract_feat_query`에 attn용 GT가 함께 넘어간다 —
  이건 attn **target** 산출용일 뿐 예측을 바꾸지 않지만, 플래그 ON일 때만 켜지는 경로다.
- **파일명 인덱스**: 이 repo의 test_pipeline `meta_keys`에 `global_idx`가 **없다**.
  따라서 `_extract_eval_global_idx`는 항상 None을 반환하고, 파일명은
  `rank*1_000_000 + 샘플순번`으로 떨어진다. fixed_val이라 run 간 순서는 재현되지만,
  **run 간 파일명을 sample id로 신뢰하려면 `global_idx`를 meta_keys에 추가해야 한다.**
- **`gt_instance_dims` / `gt_instance_sizes`는 덤프되지 않는다.** test_pipeline의 `Collect3D`
  keys에 없기 때문(로딩은 `LoadInstanceWithFlow`에서 되지만 Collect에서 떨어진다).
  GT 크기가 필요하면 config의 test `Collect3D` keys에 두 키를 추가하면 자동으로 덤프에 포함된다.
- **`score_q == cls_prob_q`인 이유**: 현재 config가 `fg_score_iou_weight=0.0,
  fg_score_cls_weight=1.0, fg_score_cam_attn_weight=0.0`이고, eval에서는 `inst_match_result={}`,
  `query_attn_cam_score_pack=None`이라 `iou_q`/`cam_attn_score_q`가 항등 0이다.
  → 덤프의 `iou_q`/`cam_attn_score_q`는 정보가 없는 0 벡터다(placeholder로만 저장).
- **`baseline_selected_idx_q`는 oracle override 이전 값**이다. `EOCF_EVAL_ORACLE_MATCH=1`과
  동시에 켜도 baseline/oracle을 나란히 비교할 수 있다.
- `EOCF_EVAL_VIS=0`이면 `dist_test.sh`/`tools/test.py`가 `EOCF_EVAL_VIS_DIR`를 자동 설정하지 않아
  덤프가 `./work_dirs/eval_vis/qfeat`로 떨어진다. **덤프만 쓸 거면 `EOCF_EVAL_VIS_DIR`를 직접 export할 것.**

## 2026-07-30 KST — eval 지표는 asset present/future 4종뿐 (+ 3D TP/FP/FN)

- 출력: `IOU_2d_asset_{present,future}`, `IOU_3d_asset_{present,future}`,
  `IOU_3d_asset_{present,future}_{TP,FP,FN,micro}`. 그 외(nusocc/bbox_aabb/bbox_rot/Recall3d)는
  코드에서 삭제됐다 — 이전 로그와 지표 이름이 다르니 비교 스크립트 주의.
- **`EOCF_EVAL_MODE`는 더 이상 metric을 바꾸지 않는다.** 시각화 기준 프레임에만 영향.
  present/future 지표는 항상 함께 계산된다.
- **aabb 로드는 두 갈래가 있다. 혼동 금지**:
  ① `LoadInstanceWithFlow(load_gt_bbox_aabb=True)` → `gt_instance_centers_world/valid/ids` 파생.
     **oracle 매칭에 필수라 절대 빼면 안 된다.**
  ② `_eval_load_bbox_gt_v2` → bbox 지표 전용 lazy 로더. 지표와 함께 사용 중단됨.
- test_pipeline에는 `LoadOccupancy`가 없다. `gt_occ`/`segmentation`/`segmentation_bev`/`instance_bev`가
  전부 None이므로, 이 값들을 참조하는 코드를 새로 추가하면 조용히 None을 받는다(가드 필수).
  train_pipeline에는 그대로 있으니 학습에는 영향 없다.

## 2026-07-30 KST — eval은 미래 4프레임만 채점한다 (present 열은 그림 전용)

- `EOCF_EVAL_MODE=1`(future)에서 metric은 `centers_eval_tq3`(= traj 텐서의 future tail 4프레임)
  으로만 계산된다. `query_debug_vis`에 present 열이 생겨도 **IoU2d/3d(asset)에는 present가
  들어가지 않는다.** 그림 열 수와 채점 프레임 수를 동일시하지 말 것.
- 행별 프레임 소스가 다르다: 1~3행은 `centers_future_tq3`(present+future), 4행은
  `eval_cmp_pack`(metric에 쓴 pred/GT). 두 텐서의 프레임 수가 다르면 **예외 없이 열이 밀린다**
  (렌더러가 `t < shape[0]` 가드로 조용히 건너뜀). 프레임 수를 바꿀 땐 반드시 양쪽을 같이 볼 것.
- present 프레임 GT를 4행에 쓰려면 metric과 **같은 align3d**를 적용해야 한다
  (`_apply_3d_align_sequence`). metric은 future tail만 정렬하므로 present는 따로 1프레임 정렬한다.

## 2026-07-28 KST — traj xy refine은 추론 적용 모듈이다 (train-only 아님)

- `refine_trajectory_absolute_xy`는 GT를 쓰지 않는다(입력: query feat / cls score / centers /
  예측 traj offset). 구조도 `corr_head`(보정량)+`gate_head`(신뢰도 게이트)의 residual refinement이고,
  loss가 refine 출력의 **절대 미래 XY**를 GT에 맞추도록 학습시킨다 → 추론에 적용해야 이득이 실현된다.
  (GT를 쓰는 `refine_trajectory_with_gt_anchor`는 별도 존재하며 미사용 — 둘을 혼동하지 말 것.)
- 2026-07-28 이전까지 `simple_test`에 배선이 없어 eval은 refine 이전 값을 썼다. 지금은 항상 적용된다.
- 레이아웃 함정: `centers_world_traj` / future mixture는 `frame_indices =
  [present_global_idx + step for step in range(F+1)]`로 만들어져 **T = F+1, index 0 = present**다.
  full-window `centers_world`(T=7)와 present 인덱스가 다르므로, 교체 helper에는 0을 넘겨야 한다.
  잘못 넘기면 helper가 조용히 원본을 반환해 **아무 일도 안 일어난다**(예외 없음).
- 학습 쪽 refine 영향은 제한적이다: `query_matched_loss_history_only=True`라 미래는 GMO loss를
  안 받고, 매칭 cost도 history 3프레임만 쓴다. 즉 refine은 자기 loss와 공유 feature gradient로만
  학습에 관여했다 — 그래서 추론에 붙여도 다른 학습 요소와 충돌할 여지는 작다.

## 2026-07-27 KST — eval GT 필터 적용 지점 정리 (사람/visibility)

- 사람(id=7) 제외는 **GT 소스마다 따로 구현**되어 있다. 한 곳만 고치면 지표 간 객체 집합이 어긋난다.
  - inst3d: config `LoadInstanceWithFlow(exclude_occ_class_ids=(7,))` — train/test 양쪽에 필요
  - asset: `_eval_load_asset_gt`의 `a[:,3] != 7` (하드코딩, config 무관)
  - bbox v2: `_eval_load_bbox_gt_v2`의 `a[:,3] == 7` 제거 (2026-07-27 추가. 그 전엔 누락)
  - nusocc gt_occ: `_eval_nusocc_3d_fg`의 `(gt != 255) & (gt != 7)`
- visibility `drop_ids`는 세 로더 모두에 이미 적용된다. 다만 조건은 "과거 3프레임 **최초** annotation의
  visibility==1"뿐 — 다른 visibility(2~4)나 나중에 1로 떨어지는 경우는 제거하지 않는다.
- `query_debug_vis` 행 구성(eval은 `_skip_traj_rows=True`라 4행):
  1=gt_cls, 2=all, 3=hi_cls는 `gt_inst`(inst3d), **4=eval 비교 행은 `eval_cmp_pack`의 GT**.
  4행만 다른 GT를 쓰므로 "1~3행엔 없는 게 4행에만 보인다"면 여기부터 의심할 것.

## 2026-07-27 KST — 이 머신 실행 전제 2가지 (torch 2.7 / occ_dt 부재)

- **torch 2.7.0 + mmcv 1.7.2**: mmcv의 `MMDistributedDataParallel._run_ddp_forward`가 torch에서
  사라진 `_use_replicated_tensor_module`을 참조한다. `mmdet_train.py`의
  `_Torch27MMDistributedDataParallel`(scatter/_run_ddp_forward 재정의)을 학습·eval 양쪽에서 써야 한다.
  이 shim이 빠지면 DDP forward 첫 호출에 AttributeError로 즉사한다 — 코드 되돌릴 때 특히 주의.
- **`./data/occ_dt` 없음**: DT 캐시가 이 머신에 없다. `use_query_dt_loss=False`면 `load_occ_dt=False`로
  두고 Collect3D `keys`에서도 `'occ_dt'`를 빼야 한다(둘 중 하나만 하면 FileNotFoundError 또는 KeyError).
- GPU는 1장(RTX PRO 6000, 97GB). 스크립트의 `GPUS=8`은 8 rank가 한 GPU에 몰리고 rank마다 NuScenes
  trainval 테이블을 통째로 로드해 62GB RAM이 먼저 고갈된다.

## 2026-07-23 KST — f3 visibility 필터 ID/시간 규칙

- `subset_attn_cover_pyr_aabb_dice3d_new_filter.py`의 제거 조건은 **7프레임 중 임의의
  visibility=1**이 아니다. 과거 3프레임에서 instance의 첫 annotation만 보고, 그 값이 1일 때만
  해당 instance를 7프레임 D/E 전체에서 제거한다. 첫 값이 2~4이면 이후 visibility=1은 유지하며
  미래 4프레임은 판정에 사용하지 않는다.
- f3 raw ID는 `vehicle`/`human` annotation을 visibility 필터 전에 등록한 순서다. 현재 dataset의
  `instance_dict` ID는 사람과 최초 저가시성 필터 뒤에 만들어져 숫자가 다를 수 있으므로 blacklist
  생성에 절대 재사용하지 말 것. `refine_instance_poly`는 attribute를 보정하므로 visibility 원본
  판정에도 사용하면 안 된다.
- 필터는 D(`segmentation_instance3d`)와 E(`segmentation_aabb`) sparse row에 동시에, center target
  생성 전에 적용해야 한다. 한쪽만 필터하면 matching/focal과 center/trajectory/AABB Dice가 다른
  instance 집합을 보게 된다.
- 실행 중인 worker에는 소스 변경이 반영되지 않는다. 이 config의 필터 실험은 반드시 새 학습
  프로세스로 시작할 것.

## 2026-07-23 KST — Asset IoU3D macro와 TP/FP/FN micro 해석

- 기존 `IOU_3d_asset`은 유효 샘플별 IoU를 평균한 macro 값이다.
- 신규 `IOU_3d_asset_TP/FP/FN`은 모든 평가 샘플·rank의 voxel 수를 합한 값이며,
  `IOU_3d_asset_micro = TP / (TP + FP + FN)`이다. 샘플 크기 가중 방식이 다르므로
  `IOU_3d_asset`과 `IOU_3d_asset_micro`가 서로 정확히 같지 않은 것이 정상이다.
- `Recall3d(asset)_TP/FP/FN`은 base inst3d와 asset 관용 영역으로 계산되는 별도 지표다.
  Asset 자체를 GT로 직접 비교하는 신규 `IOU_3d_asset_TP/FP/FN`과 혼동하지 말 것.

## 2026-07-23 KST — all-query Asset scene BCE 첫 실행 결과 및 수정

- 첫 실행은 `Epoch [1][4/500]`까지 정상 진행한 뒤, 빈 GT-center pack이 있는 다섯 번째 배치에서
  scene BCE의 과도한 full7 guard 때문에 종료됐다. 이는 Asset occupancy/예측 timeline의 정렬
  실패가 아니며 guard를 GT center 유무와 분리해 수정했다. 종료 전에 checkpoint는 생성되지 않았다.
- 정상 4개 배치에서 가중치 적용 후 `loss_query_scene_asset_bce=1.3253~3.0932`로 기존
  `loss_gmo_dice=0.2849~0.3513`보다 훨씬 컸다. `0.1`은 효과를 확실히 보는 공격적인 설정이며,
  안정적인 FT가 목적이면 후속 실험에서 전체 가중치를 낮추는 것을 검토할 것.
- 첫 배치 89초는 데이터/초기화 시간을 포함하며 이후 iteration은 약 11~13초, 관측 peak memory는
  rank 로그 기준 약 37.6GB였다.

## 2026-07-23 KST — all-query Asset scene BCE FT 주의사항

- `subset_attn_cover_pyr_aabb_dice3d_new_asset_all_ft_bce.py`의 scene BCE는 추론과 같은 query-axis
  `max`를 쓰므로 한 voxel에서는 최댓값 query에만 gradient가 간다. query confidence는 프레임별
  head가 아니라 현재/미래 5프레임이 공유하는 foreground score다.
- query별 full `[5,200,20,128,128]` volume은 만들지 않지만, 200×48 Gaussian의 유효 bbox patch는
  backward를 위해 유지된다. 실제 8GPU 첫 실행에서 peak memory와 iteration time을 확인할 것.
- `query_scene_asset_bce_pos_weight/neg_weight=1/1`로 시작한다. IoU가 오르며 recall이 급락하면
  geometry 개선보다 confidence/opacity 억제가 우세한 것이므로 neg weight 또는 전체 weight를
  낮춰야 한다. `dbg_query_scene_asset_{bce_pos_raw,bce_neg_raw,pred_gt_mass_ratio,conf_mean}`을 함께 본다.
- AGENTS.md가 함께 고려하라고 명시한 `projects/configs/baselines/EfficientOCF_V1.1_1gpu.py`는
  현재 repository에 존재하지 않는다. 공용 신규 키는 `efficientocf_config.py`에 기본 비활성으로
  추가했으며, 해당 1GPU config가 복구되면 필요할 때 명시적으로 배선할 것.

## 2026-07-13 KST — mixture3d 시각화의 GT class 라벨은 binary-fg 모드에서 전부 "bicycle"로 잘못 찍힘 (표시 전용, 수정 안 함)

`utils_visualization.py:1207`의 `_QUERY_CLS_NAMES_8` 이름표를 compact class id로 인덱싱하는데,
`query_cls_binary_fg=True` config에서는 모든 전경 인스턴스의 compact id가 1이라 표의 index 1인
"bicycle"이 항상 찍힘 (car든 truck이든). multi-class 모드에서는 표가 맞음. **loss/매칭은 id로
계산해서 정상 — 순수 표시 버그.** 사용자 결정으로 수정하지 않고 둠(2026-07-13). mixture3d에서
"bicycle" 라벨 보여도 class 배선 오류로 오인하지 말 것. 고치려면 `_cls_name()`(:1354)에
binary 분기 한 줄 추가하면 됨.

## 2026-07-13 KST — (해결됨) D(inst3d) 경로도 빈-씬 샘플에서 같은 shape-추론 문제 있었음

아래 있던 우려("하드 크래시 없어서 안 건드림")가 실제로 터짐 — 크래시가 아니라 rank마다 loss dict
키 개수가 달라지는 DDP AssertionError로 나타남. `_prepare_gt_occ_inst_primary_targets`와
`_build_instance_center_world_targets_from_sparse`에도 `spatial_shape=(voxelizer.W,H,D)`를 넘기도록
수정 완료. 상세 내용은 CHANGELOG.md "`_rot` 학습 2차 크래시 수정" 항목 참고. **결론: `torch.is_tensor`
가드로 관대하게 처리된다고 해서 안전한 게 아니었음 — None 전파가 상위에서 loss 블록 전체를 스킵시켜
다른 형태의(더 찾기 어려운) 버그로 나타날 수 있다는 교훈.**

## 2026-07-13 KST — binary-fg 전용 추가 단순화(class 판정 생략, 존재만 확인)는 보류

`_prepare_gt_instance_classes_for_matching` 벡터화 이후, `query_cls_binary_fg=True` config는 raw
class를 정확히 안 따지고 "instance에 유효 voxel이 있냐"만 봐도 되지 않냐는 아이디어가 나옴 — 맞는
지적이고 실측으로도 추가 ~2배 빨라짐(K=5~70에서 132~137ms → 69~74ms, 결과도 동일). 근데:
- `query_cls_binary_fg`가 없는 config(`full_attn_cover.py` 등 multi-class)는 raw class를 그대로
  써야 해서, 적용하려면 binary/multi-class 분기를 함수 안에 새로 넣어야 함.
- 절대 이득이 작음(스텝당 ~60ms, `_rot`(4000샘플×15epoch) 기준 전체 학습 통틀어 ~7.5분).
- 분기 추가 복잡도·검증 부담 대비 이득이 작다고 판단해 **적용 안 하기로 함**. 이미 적용된
  범용(binary/multi-class 공통) 벡터화 버전을 그대로 유지.

## 2026-07-13 KST — (최종 결론) GT metadata 캐시 대신 `_prepare_gt_instance_classes_for_matching` 벡터화로 종결

바로 아래 항목("GT metadata 캐시... 이식 안 함")에서 캐싱을 보류한 뒤, 실측을 더 해보니 캐싱 자체가
필요 없다는 결론까지 남. `query_present_only=True`라 class 판정 직전에 이미 7프레임→1프레임으로
줄어서 실비용이 작았고(T=1, K=70 기준 2488ms), 캐싱으로 아끼는 절대 시간이 15 epoch 전체 기준
8.6분 수준이라 캐싱 인프라(mixin/config/심링크/raw-compact 분리, model config 바뀌면 오염 위험)를
새로 만들 실익이 없다고 판단. 대신 loop를 `torch.unique(pairs, dim=0)` 1회 호출로 벡터화(2.0~18.9배)
— 디스크/config 없이, 위험 없이 적용. 상세 내용은 CHANGELOG.md 동일 날짜 항목 참고. **결론: 이
레포에서 GT metadata 캐시는 만들지 않기로 최종 확정.**

## 2026-07-13 KST — GT metadata 캐시(다른 레포 `EOCF_pyramid_multi`)는 이 레포에 이식 안 함 — 이유

- `EOCF_pyramid_multi`의 `utils_gt_meta_cache.py`(2026-07-08 그쪽 changelog)는 이 레포에 없음 — 의도적으로
  안 옮김. `_filter_dense_instance_ids` torch.isin 벡터화(같은 날짜 CHANGELOG 참고)만 이식함.
- **캐싱 대상 3개 함수 중 실제로 이 레포에 유효한 건 1개뿐**:
  - `_build_intersection_instance_ids_from_dense_pair` — 이 레포에서 **죽은 코드**. secondary
    input(`gt_segmentation_instance3d_txyz`)이 구 `segmentation_instance3d` 캐시 삭제 이후 항상
    `None`이라 매번 즉시 `None` 반환(`np.intersect1d`까지 안 감). 캐싱해도 이득 없음.
  - `_build_history_all_valid_instance_ids` — python loop 없는 순수 텐서 bool 연산이라 원래 저렴.
  - `_prepare_gt_instance_classes_for_matching` — 유일하게 진짜 비싼 함수(K개 인스턴스 python loop,
    실측 K=5~70에서 1.1~2.9초/call, CPU 기준 occ_size=512x512x40,T=7 실측치).
- **이 함수를 캐싱하려면 다른 레포 버전 그대로 쓰면 안 됨**: `utils_gt_prep.py:1228`
  `_prepare_gt_instance_classes_for_matching`이 raw class id → compact class id 매핑
  (`self.query_raw_to_compact_class_map`, `query_class_ids`/`query_cls_binary_fg` 등 model_cfg에서
  파생)을 함수 **안에서** 적용한 뒤 리턴함. 다른 레포의 캐시 검증 로직(`_gt_meta_cache_get_inst_cls`)은
  "instance id 리스트가 캐시된 것과 같은가"만 체크하고 class 매핑 스킴이 바뀐 건 감지 못 함 → 같은
  (scene,lidar) 토큰에 model config만 바꿔 재사용하면 **옛 매핑 기준 compact class를 조용히 반환**할
  위험이 있음(에러 없이 잘못된 라벨로 학습됨).
- **나중에 이식하려면**: 캐싱 경계를 raw class(매핑 전, majority-vote 직후)까지만으로 옮기고, raw→compact
  매핑은 캐시 밖에서 매번 새로 계산(단순 인덱싱이라 사실상 공짜, 캐싱 불필요)하도록 재설계할 것.
  이렇게 해야 GT 소스 변경뿐 아니라 model config(class 매핑) 변경에도 안전해짐.

## 2026-07-13 KST — [new_data 레포] `_rot` 배선 시 확인한 주의사항 / 미해결 항목

- **eval 스크립트는 아직 `_rot` 체크포인트를 안 가리킴**: `eval_total.sh`(CONFIG=`full.py`,
  이미 깨짐)/`eval_total_2.sh`(CONFIG=`full_attn_cover_pyr_aabb_dice3d.py`) 둘 다
  `subset_attn_cover_pyr_aabb_dice3d_rot.py`를 안 가리킴. `train_total.sh`가 지금 이 config를
  가리키고 있으니(`--resume ./work_dirs/subset_attn_cover_pyr_aabb_dice3d_rot/latest.pth`), 실제
  학습이 시작되어 체크포인트가 나오면 eval용 스크립트를 새로 만들거나 기존 스크립트의 CONFIG/CHECKPOINT를
  이 경로로 바꿔야 함(이번 작업 범위 밖 — 아직 안 함).
- **`gen_new_gt_pipeline.py` 필터 모순 (미해결, 재생성 시 주의)**: 이 스크립트 자체 docstring은
  "필터 전부 OFF(생성소멸 카운터 제거)"라고 명시하는데, `CHANGELOG.md`는 `efficientocf_gt_f3`를
  "생성소멸 필터판(관측창 0~2 교집합)"이라고 설명 — 서로 모순. 지금 디스크에 있는
  `efficientocf_gt_f3`(구조/키/id-space/train-val 분리/완전성은 전부 실측 검증 완료, 문제 없음)가
  실제로 어느 쪽으로 생성됐는지는 정적 분석만으론 확정 불가. **이 스크립트로 rot 데이터를 포함해
  재생성/추가생성할 계획이면, 재생성 전에 이 필터 상태부터 먼저 확정할 것** — 안 그러면 기존
  파일들과 필터 기준이 다른 데이터가 섞일 위험.
- **`segmentation_rot` 학습 배선 완료 상태**: `loading_instance.py`에 `gt_bbox_aabb_subdir` 파라미터
  추가로 rot GT를 `gt_bbox_aabb` 자리에 재사용하도록 배선함(CHANGELOG 2026-07-13 참고). 학습 중인
  `full_attn_cover.py`(mid-training 크래시 이력 있는 파일, NOTES 07-07 참조) 안전을 위해 self-heal
  패턴을 반드시 같이 추가했음 — 이 파일에 새 속성을 추가할 때는 항상 `__call__` 상단 self-heal
  블록에도 기본값을 추가할 것(안 하면 다음 epoch에 학습 중인 다른 job이 죽을 수 있음).

## 2026-07-13 KST — [new_data 레포] 구 config명을 아직 참조하는 스크립트/도구 목록 (이번 재배선 범위 밖)

- **무엇**: config 5개 전부 새 GT(f3) 재배선 + `loading_instance.py` 구 캐시 코드 삭제(CHANGELOG
  2026-07-13)를 하면서, 레포 전체에서 이미 삭제된 8개 구 config명(`full.py`/`subset.py`/
  `subset_scale*.py`/`subset_attn_cover{,_size,_aabb}.py`)을 참조하는 곳을 grep해봤더니 예상보다
  넓게 남아있었음: `eval_total.sh`(CONFIG=`full.py`, 대응 후속 config 없어 미수정), `train_total_{2,4,5}.sh`,
  `eval_{oracle,sweep_thr,sweep_indep}.sh`, `eval_total_v1.sh`, `tools/dbg_probe/*.py`,
  `tools/gen_data/gen_bbox_gt_v2.py`.
- **조치 안 함**: 이번 작업은 사용자가 명시적으로 확인한 범위(config 5개 + `loading_instance.py` +
  깨진 게 이미 확인된 `eval_total_2.sh`)만 처리. 위 목록은 실제로 깨졌는지(경로 존재 여부) 개별
  확인 안 된 상태 — 다음에 이 스크립트들을 실행하려 하면 먼저 CONFIG= 줄이 가리키는 파일이 실제로
  있는지 확인할 것.

## 2026-07-10 KST — NEW_GT_PIPELINE_SPEC.md 전면 실측 검증 결과 (에이전트 31개 + BEV 시각 대조)

- **스펙 정정 ① (v2/val)**: "ped 우선매칭 누락"이 아니라 **pedestrian 미포함 생성**이 실체 — 1,000파일 스캔에서 cls=7이 0건. v2는 `--include_ped_ids` 없이 default로 생성됐고, construction_worker만 'construction' substring으로 유입돼 cls=5를 받음(비겹침 CW 박스 12건 전부 100% cls=5).
- **스펙 정정 ② (gt_occ_inst)**: §0 표의 "inst3d CW cls=5 오염"은 **반대** — inst3d는 CW를 올바르게 cls=7로 저장(18곳 중 17곳, 유일한 cls=5는 진짜 vehicle.construction 겹침). cls=5 오염의 주체는 v2였음.
- **v3(train) 건전성 확정**: 23,930키 = inst3d−v2 정확히 일치, CW→7/adult→7/건설차량→5 전부 정상, 파일 손상 0. rot 없음(스펙 §8.5대로).
- **id 정렬**: train 3캐시(v3·inst3d·seg3d)는 인스턴스 id **100% 일치**, 빠진 인스턴스는 번호를 밀지 않고 구멍으로 남음(gap-not-shift). **val의 v2만 52.4% 일치** — 사람 제거 후 번호 재부여 탓. 로더/모델엔 id 재매핑 코드가 전혀 없어(캐시 숫자 그대로 소비, 유일 안전장치는 id 교집합 필터) val에서 v2를 inst3d와 id로 엮으면 즉시 오염됨.
- **npz collapse 함정 (신규 발견)**: 7프레임 row 수가 전부 같으면 `np.array(frames, dtype=object)`가 (7,N,5) 3-D object 배열로 붕괴 — v2 ~11%, v3 ~3% 실측. 새 파이프라인 저장부는 `arr=np.empty(T,object); arr[:]=frames` 방식으로 방어할 것(스펙 §7.3 코드 그대로 구현 금지). dtype도 캐시별 상이(v2/v3=int32, inst3d/seg3d=int64).
- **center GT +0.1m 편향은 2곳**: `loading_instance.py:659`(§9.4) 외에 `utils_gt_prep.py:1005~1006`(gt_occ_inst 폴백 center 빌더)에도 동일 편향. 고칠 땐 반드시 동시 수정(한쪽만 고치면 두 소스가 0.1m 어긋남).
- **§8.3 위치 정정**: `build_instance_center_world_targets`는 detector가 아니라 `loading_instance.py`(:583 정의, :1532/:1692 호출) — center GT 배선 변경은 로더 파이프라인 수정임.
- **§12.4 실측 입력**: 기존 inst3d는 중복 좌표 0(생성기가 이미 병합, summary.json의 class_mismatch_n=4.3M이 그 흔적). 새 D=A∩C는 겹침 AABB에서 중복이 생기므로 병합 규칙을 구현 전에 확정해야 strict 로더(`_validate_sparse_rows_basic`) 통과.
- **환경**: v3 캐시는 온전하나 `data/efficientocf_bboxcls_v3` 심볼릭 링크만 누락(7/10 링크 재생성 때 빠짐). aabb 계열 config 실행 전 `ln -s /home/hwanhee/datasets/efficientocf_bboxcls_v3 data/` 필요 — resolver는 조용한 폴백 없이 FileNotFoundError를 냄. 이 머신의 활성 8-GPU 학습은 **다른 레포**(EOCF_pyramid_multi) 소속.
- 검증 상세(판정 20건 전체표 + BEV 그림 7종): 아티팩트 "GT 파이프라인 스펙 검증 리포트" 및 대화 로그 참조. inst3d⊆occupancy 100.000%, inst3d 비보행자⊆v2 100%, rot⊆aabb 위반 0, config 상수 12/12 동일.

### [예약 문구] center GT +0.1m 편향 수정 — §8.3 배선 변경 시 함께 적용할 것 (2026-07-11 확정, 아직 미적용)
- 원인: voxel→world 역변환에서 `-res/2` 누락. 정변환(`get_poly_region`, `round((pts-pc_range_min)/res)`) 기준 voxel v의 참 중심은 `pc_range_min + v*res`인데 아래 두 곳은 `pc_range_min + v*res + res/2`를 계산 → 3축 각각 +0.1 m 공통 편향. **데이터 캐시는 무관(복셀 좌표 정상), 로더 산출값만 밀리는 로더 버그.**
- 수정 ① `projects/occ_plugin/datasets/pipelines/loading_instance.py:659` (`build_instance_center_world_targets`) — 아래로 교체:
  `centers_world = centers_voxel * self.resolution[:3].reshape(1, 1, 3) + self.start_position[:3].reshape(1, 1, 3) - self.resolution[:3].reshape(1, 1, 3) / 2.0`
- 수정 ② `projects/occ_plugin/occupancy/detectors/utils_gt_prep.py:~1004` (gt_occ_inst 폴백 center 빌더) — `start = pc[:3] + (0.5 * voxel)` → `start = pc[:3]`
- ①②는 반드시 동시 적용(한쪽만 고치면 두 center 소스가 서로 0.1 m 어긋남). 활성 run 단일변수 원칙 때문에 배선 작업 전까지 적용 금지.

### [결정사항] 새 GT 파이프라인 스펙 변경 (2026-07-11, 사용자 확정)
- **① CW/클래스 정합**: `cls_map = (("pedestrian",7),) + CLS_MAP` (ped 우선매칭 ON) — construction_worker→cls=7. nusocc(lidarseg)·inst3d·v3와 클래스 의미 일치, D(=A∩C)에서 bbox와 occ의 CW 클래스가 7로 정합. 스펙 §3.2 기본값(ped-last, CW→5)을 뒤집는 결정.
- **② 겹침 AABB 중복 좌표 병합**: **먼저 등장한 인스턴스(작은 id) 우선(first-wins)**. id가 최초 등장 순서로 부여되므로 미래-신규(생성) 인스턴스는 항상 더 큰 id → 추후 생성소멸 필터 방향과 일치, 신규 물체가 기존 물체 복셀을 잠식하지 않음. 구현: id 오름차순으로 채우되 이미 점유된 좌표는 skip(결정적, 추가 상태 불필요). 적용 지점은 D(nusocc_inst)의 dedup(strict 로더 통과용)뿐 — E/F(bbox 최종)는 지금처럼 중복 허용 유지.
- **배선 체크리스트(추가 확인 2건)**: (a) `forward_train`에 같이 들어가는 `segmentation`/`segmentation_bev`(구 필터 기준 바이너리 캐시, id 없음)는 이번 파이프라인 범위 밖 — raw GT(인스턴스 더 많음)와 내용 불일치 생기므로 해당 loss의 GT 소스 확인 후 유지/E-파생 재생성 중 택일. (b) id를 쓰는 GT 텐서는 현 배선상 전부 캐시 소스(runtime record_instance id는 모델에 안 들어감)임을 forward_train 인자 목록으로 확인함 — §8.3/8.4 배선 완료 후에는 id 소스가 D/E로 단일화됨.

## 2026-07-10 KST — `Autonomous_Driving_26_ksh_local`에서 이식한 기능은 이 레포에 자동 동기화되지 않음

- **무엇**: query cross-attention coarse-to-fine KV 해상도 피라미드(`kv_resolutions`)는 원래
  `Autonomous_Driving_26_ksh_local` 레포(자매 레포 jhh_gi에서 이식)에만 있던 기능이었음. 이 레포
  (`ksh_0630`)는 별도 워킹 카피라 코드가 자동으로 동기화되지 않고, `query_transformer_num_layers`만
  있고 `kv_resolutions` 관련 코드가 전혀 없는 상태였음 — config에 `query_transformer_kv_resolutions`를
  적어도 `_merge_cfg`가 unknown key를 조용히 무시해 **에러 없이 그냥 죽은 설정**이 되는 함정이 있었음.
- **조치**: `transformer.py`/`efficientocf_config.py`/`efficientocf.py` 3개 파일에 ksh_local과 동일한
  코드를 수동 이식(diff 대조로 이 기능 외엔 두 레포 `transformer.py`가 100% 동일함을 확인 후 진행).
  상세: CHANGELOG.md 2026-07-10.
- **일반화된 경고**: 앞으로 "다른 레포(ksh_local, jhh_gi 등)에 있던 X를 여기도 적용해줘" 요청이 오면,
  이 레포에 해당 코드가 실제로 있는지(config 키가 `efficientocf_config.py`의 `apply_model_cfg`에서
  실제로 읽히는지) **먼저 grep으로 확인**할 것 — 있는 것처럼 보여도(하위 num_layers 등 관련 키만 존재)
  실제 기능 코드가 없을 수 있음 (`query_gmo_dice_3d`처럼 이미 포팅된 케이스와 `kv_resolutions`처럼
  전혀 없던 케이스가 혼재).
- **미검증**: `subset_attn_cover_pyr_aabb_dice3d.py`(σ-matching + dice3D + 피라미드 3중 결합)는
  단독 스모크(forward/backward)만 확인했고 실제 학습(메모리/속도/σ-matching 상호작용)은 미실행.
  ksh_local NOTES 2026-07-07 항목의 "σ-matching과 피라미드 상호작용 미검증" 경고가 동일 적용됨.

## 2026-07-08 KST — [전수 감사] GT 캐시 6종 정밀 감사 결과 (6개 병렬 감사, train 300키+val 150키 × seed 2회 교차검증)

**건전 판정 (실측 확정, 더 의심할 필요 없음):**
- **id 번호체계**: GT1(gt_occ_inst)⊆GT2(segmentation_instance3d), GT1⊆GT3(v3) 프레임 단위 100%(2,100프레임×2seed, 위반 0), 공유 id voxel coverage 1.00000 — 번호 shift/충돌 없음. GT1에만 있어 center 못 받는 instance 0건.
- **center GT 품질**: GT2 AABB 중점 vs annotation box 중심(GT3) xy거리 중앙값/p95 = 0.000m, radial bias ±0.001m — systematic bias 없음(큰 물체 포함). 반면 GT1 voxel-mean은 ego쪽 -0.55m(bus/trailer -0.7~-1.7m) 쏠림 → 현 GT2 AABB 중점 설계가 정답임을 정량 확인.
- **생성소멸**: 현 config에서 loss로 새는 경로 0건. GT2는 미래신규 id가 구조적으로 0(원천 방어), GT3 미래신규(키의 15~16%)는 keep-id 필터(efficientocf.py L2803-2806)+_hist_slice 이중 방어로 학습 도달 0. 단 두 방어 모두 `query_require_history_all_valid`/`query_matched_loss_history_only` 플래그 의존 — 끄는 실험 시 재검토 필수.
- **v2 box 누락 해소**: GT1 present 물체(voxel≥10) non-ped 커버 100%(1,440/1,440, 1,519/1,519) — v1의 '누락 13/30' 결함은 v2에서 완전 해소. barrier 유입 0, rot⊆aabb 위반 0.
- **cw(공사인부) leak 정체 확정**: GT1 leak 155건 전부 '이웃 대형물체 id에 흡수'(out_frac 중앙값 0.966, id 총voxel 중앙값 1,772) — 사람 고유 id가 ped 필터를 뚫고 학습에 들어가는 케이스 **0건**. 실해는 차량 GT 안 사람모양 voxel 소량(155박스 합 16k voxel) 혼입뿐.
- **GT5(bboxcls cls, 학습 cls loss GT)는 cw를 cls=7로 정상 저장** — v2와 달리 오염 없음(0~0.4%).

**실질 문제 (심각도순):**
1. **[구조 한계] 교집합 필터로 center 감독 탈락 프레임당 평균 5.3~5.7개**(GT2 instance의 33~36%): ped 몫 ~3.8~4.3(nohuman 설계상 의도), **점유 0 박스 몫 ~1.5개/프레임**(가려짐/포인트 없음 — 카메라에 보일 수 있는 물체가 center 감독 없음). 미래 프레임일수록 증가(f0 4.0→f6 6.3).
2. **[상충 감독] surviving id의 4.5~4.7%: GT1 history 3프레임 중 1개+가 빈 채 matched focal에 도달** → focal(inst3d 1.0)은 all-negative, dice_bbox(0.9)는 positive를 가르침(같은 프레임 GT3 커버 98~100%). 해소책: focal에서 GT1-빈 프레임 마스킹 or history_all_valid를 GT1 점유 기준으로 변경.
3. **[GT1 cls 오염] cw가 GT1에도 cls=5로 저장돼 ped 필터 통과**(GT3는 cls=7로 드랍) → GT1/GT3 감독 불일치 소수(7~21 id/2,100프레임). + 겹침 box에서 (cls,inst) 교차오염 실측(truck id에 trailer cls 210행). 외부 캐시라 repo 내 수정 불가.
4. **[v2 cw 오염 정량 확정] aabb voxel의 0.068~0.080%, rot 0.067~0.086%** — GMO 전체 IoU 왜곡 상한 ~0.1%p 상대(무시 가능), 단 construction 클래스 한정 1.4~2.3%(클래스별 지표 쓸 때 유의).
5. **[신규 발견] refine_instance_poly 위치 동결**(x/y 이동≤1m/프레임이면 직전 위치 복사) + 미annotation 프레임 ghost box: 느린 물체의 GT box가 raw annotation과 최대 ~1m/프레임 어긋남(cw 사례 IoU 0.857→0.575 드리프트). train GT와 일관돼 랭킹 중립이나 절대 위치 정확도 저하. (v2/v3/GT2 공통 — 원본 파이프라인 상속)
6. **[버그픽스 영향 정량] id=7 드랍 버그(2026-07-08 수정)**: train 키의 48.7~50.7%에서 물체 1개(car ~75-80%) center 감독 제외였음. 수정 후 center 대상 +5.5~5.8%, attn pair +5.6~8.2%, focal/dice pair +6.2~8.8%. 영향 범위 정확히 id=7에 국한 검증(diff 위반 0). id=7이 사람인 키(34/28건)는 교집합에서 어차피 빠져 무영향.

기타: GT1 box-margin 삐져나옴 재측정 0.1%(기존 기록 ~1%보다 낮음, 필터 재현 후 측정 차이). GT2 z-extent는 GT3와 97~98% 완전 일치(placeholder는 cls뿐, 기하는 진짜 box). GT2≠GT3 차이(~9%)는 미래 진입 물체(GT2 부재)+가시성0 박스(GT3 드랍)로 전부 설명. 감사 스크립트/원시 출력은 세션 스크래치(임시)에 있으며 수치는 본 항목이 원본 기록.

## 2026-07-07 KST — ⚠ 사고 기록: 학습 중 pipeline 코드 수정으로 full·full_attn_cover 사망
- **무엇**: 07-06 21시경 `loading_instance.py`에 `load_gt_bbox_aabb` 속성 추가(aabb 실험 배선) → 돌고 있던 **full**(07-07 03:12, epoch 13 직후)과 **full_attn_cover**(07-06 23:23, epoch 5 직후)가 다음 epoch 경계에서 `AttributeError: no attribute 'load_gt_bbox_aabb'`로 사망.
- **메커니즘**: dataloader worker가 spawn 방식이라 매 epoch 워커가 **디스크의 새 코드를 import**하면서 **pickle된 구 인스턴스**(새 속성 없음)를 복원 → 새 `__call__`이 없는 속성을 참조. 즉 **학습 도는 동안 datasets/pipelines 파일에 속성·필드를 추가하면 그 이후 첫 epoch 경계에서 기존 run이 죽는다.**
- **피해**: 학습 진행분 손실 없음 — 둘 다 epoch checkpoint 저장(03:06/23:16) 직후 죽어서 latest.pth(=epoch_13/epoch_5)로 그대로 resume 가능.
- **재발 방지**: ① `__call__` 초입에 신규 속성 self-heal 블록 추가(구 인스턴스도 기본값으로 동작; 새 속성 추가 시 여기에 같이 등록). ② 원칙: **학습 도는 동안 datasets/pipelines 코드 수정 금지** (모델 쪽은 main process에만 로드돼 상대적으로 안전하나 재시작 전까지 미반영).

## 2026-07-06 KST — [스윕 완결] nms0+ones occ 곡선 + fg 교차 (epoch_11, 512, 총 12조합)
- **nms0+ones (fg.75)**: occ .3/.5/.6/.75 → aabb 0.1717/0.1720/0.1715/0.1697 (**~0.172 포화 — 벌리기로 얻을 aabb 상한**), recall 0.1909/0.2223/0.2382/0.2652 (단조; 비관용FP 227→113M와 역상관).
- **nms0+ones+fg0.5/occ.75**: 0.1462/0.1736 — ones에선 fg가 진짜 독립 손잡이가 되고, 낮추면 저품질 query가 unit 높이로 유입돼 전 지표 최악권. fg 0.75 유지 확정.
- **최종 우승 = A: nms0 + score 가중치 + fg.75/occ.75** — recall 0.3073(1위), aabb 0.1689(1위 0.1720과 1.8%차), rot/nusocc/3D 전 계열 1위. score 가중치가 '약한 blob 얇게' 소프트 필터 역할.
- 교훈 3: ① aabb는 확장으로 0.172가 천장 — 이후는 배치 정확도(학습) ② recall=비관용FP 싸움 ③ NMS off(물체 위 중복 허용)는 이득, 균일 확장(ones)은 손해.
- 남은 추론 손잡이: w_iou(단, ckpt에서 IoU head 학습 여부 먼저 확인). 학습 개입 우선순위: unmatched 렌더 억제 > 렌더-recall(matched, 물체 방향) > z/drift 교정.

## 2026-07-06 KST — [실측] NMS off / weights=ones 실험 + Recall3d 성분 분해 (epoch_11, 512샘플)
- **comps 진단 (fg.75/occ.75)**: TP 2.7M / FN 8.3M / FP 78M(box안 17M+밖 61M). pred 부피=GT 7.4배, 그중 75%가 box 밖. **병목 = 흘림(비관용FP가 FN의 7.4배)** — FP 절반이면 recall +0.20, FN 절반 전환은 +0.08 (레버리지 2.5배). 물체-수준 미탐은 적고 voxel-수준 정밀도 붕괴형. 2D>3D IoU 격차 → z 과확장/수직 어긋남 정황.
- **NMS off (EOCF_EVAL_NMS_RADIUS=0) = 전 지표 최고**: aabb 0.1478→**0.1689(+14%)**, IoU3d(aabb) +16%, Recall 0.3073, micro 0.2869. 메커니즘: 3m NMS가 물체당 blob 1개 강제 → 끄니 중복 query가 큰 물체를 나눠 채움(조준된 확장, TP+30%). **"큰 물체 커버리지"가 실병목이라는 방증.**
- **weights=ones (EOCF_EVAL_WEIGHT_MODE=ones) 단독은 손해**: occ.3/.5/.75 → recall 0.205/0.235/0.272 (전부 baseline 0.306 미만), aabb 최대 0.1594. 일괄 확장은 FP(최대 183M)가 TP 이득 초과. **확장은 '어디를'이 관건 — 균일 확장(ones) ✗, 물체 위 중복(NMS off) ✓.**
- fg↑↑ 전망: FP 주 출처(크기 초과·drift·z)가 score 무관 → 한계 예상. 올바른 손잡이는 w_iou(품질 필터).
- 결합(nms0+ones/occ.75) 측정 중. 산출물: work_dirs/full/eval_sweep/epoch_11_lss_only_*.

## 2026-07-06 KST — instance id 번호 체계 (⚠ bbox GT를 loss에 쓸 때 반드시 알아야 함)
- **기존 캐시(gt_occ_inst3d, efficientocf/segmentation_instance3d)의 번호 체계 = `classes=['vehicle','human']`** (원본 EfficientOCF config 실측: EfficientOCF_test/.../EfficientOCF_V1.1_subset_rd_light_fix_height.py:32). ped가 번호를 소비하고, `static_object.bicycle_rack`은 배제됨.
- **함정 2개 (둘 다 실측으로 확인)**: ① v2 val 캐시는 현재 config class_names(ped 없음)로 생성돼 ped가 번호를 안 먹음 → id 어긋남(같은 id 커버리지 0.45) — **v2를 pair-wise loss GT로 쓰면 안 됨**(eval bbox 메트릭은 id 무관이라 무해). ② 현재 config class_names의 'bicycle' substring은 record_instance에서 bicycle_rack까지 등록 → rack 낀 시퀀스에서 이후 id 전부 밀림. 역공학(probe_numbering): ['vehicle','human']이 문제 키 3/3에서 캐시 id→cls 100% 재현.
- **v3 캐시**(`data/efficientocf_bboxcls_v3`, train 23,930, [x,y,z,nusocc_cls,inst])는 cache-parity 모드(gen_bbox_gt_v2.py --include_ped_ids)로 생성 — 기존 캐시와 id 완전 호환. ped 행은 cls=7이라 로더 exclude가 제거(번호만 유지).
- **inst3d 캐시의 알려진 노이즈 (v3의 오류 아님, 검증 시 오판 주의)**: ① 인스턴스 경계 cls 혼입(예: truck id에 trailer cls 수십 voxel — dominant cls로 판별할 것) ② **inst3d 생성 당시 box를 약간 확장(margin)해서 voxel을 모은 흔적** — annotation box(AABB v3)보다 1~2 voxel 삐져나오는 pair가 ~1% 존재(실측 a6fa 키: t=0에서 pose가 raw annotation과 완전 동일한데도 15% 밖). v3는 loader get_poly_region과 동일한 tight annotation box가 정답. ③ annotation이 한 프레임뿐인 인스턴스는 refine이 frozen pose로 채우는데 inst3d voxel은 이동 → 후속 프레임 cover 하락(같은 a6fa, t≥1 cover 0.59).
- **whole-box in-range 규칙**: box 코너가 pc_range 밖이면 v3는 그 프레임을 통짜 drop하지만 inst3d(occupancy)는 남음 — focal bbox 항은 utils_loss의 프레임 가드가 이런 프레임을 자동 제외.
- **⚠ construction_worker 함정 (v3에서 수정, v2엔 잔존)**: `human.pedestrian.construction_worker`가 CLS_MAP의 ("construction",5) substring에 걸려 **공사장 인부가 cls5(차량)로 저장**되면 로더 ped 필터(7)를 통과함 — v3 생성기는 pedestrian 매칭을 최우선으로 두어 해결(gen_bbox_gt_v2.py). **v2 val 캐시와 구 bboxcls는 이 문제가 그대로 있음**(현 config class_names의 'construction'도 worker를 instance_dict에 등록) → bbox eval 메트릭 GT에 인부 box가 소량 섞여 있다는 뜻. eval 수치 연속성 때문에 v2는 재생성 안 함 — 필요 시 gen_bbox_gt_v2.py로 val도 재생성해 교체할 것.
- **loss 배선 요약**: focal3d GT 혼합은 `query_gmo_focal_inst3d_weight/query_gmo_focal_bbox_weight`(efficientocf_config 기본 1.0/0.0). bbox_w>0 + pipeline 로드 누락 시 forward_train에서 명시적 raise. focal3d는 history-only(`query_matched_loss_history_only`)라 bbox GT도 과거3프레임만 실사용.
- **✅ 2026-07-08 수정 완료 (loading_instance.py:1473, 구 1380)**: segmentation_instance3d 로드의 `_filter_sparse_rows_by_class(rows, class_col=-1)` 호출을 제거함 — 이 캐시의 마지막 컬럼은 **class가 아니라 instance id**라 exclude(7,)이 "**instance id==7인 인스턴스를 통째로 드랍**"으로 오동작하던 버그(실측: raw 캐시 1,050프레임 중 649~727프레임=62~69%에서 발동 조건 성립). ped 배제는 `gt_occ_inst`와의 id intersection(`_build_intersection_instance_ids_from_dense_pair`, class_col=3으로 정상 필터된 소스)이 이미 수행하므로 이 필터 제거는 안전. **부수효과**: `query_require_history_all_valid`(과거 3프레임 all-valid 교집합 필터)도 이제 완전한 instance 집합을 받게 됨. 수정 당시 `full_attn_cover`/`subset_attn_cover_aabb_dice`(ep11)/`subset_attn_cover_pyr` 3개 학습이 돌고 있었음 — 사용자 판단으로 즉시 적용(크래시 나면 latest.pth resume). **이 수정 이후 시작된 run과 그 이전 run(subset_attn_cover, subset_attn_cover_aabb{,_dice} 등 전부)은 GT가 다르므로 직접 비교 시 이 차이를 감안할 것.** 상세: CHANGELOG.md 2026-07-08.

## 2026-07-06 KST — [스윕 실측] epoch_11 threshold 스윕 결론 (512샘플, work_dirs/full/eval_sweep/)
- 결과(IoU2d(aabb)/Recall3d): fg.75/occ.5 = 0.1561/0.2487 · fg.5/occ.5 = 0.1289/0.1605 · fg.5/occ.75 = **0.0922…0.1478/0.3058** · fg.75/occ.75 = fg.5/occ.75와 **7지표 전부 소수점까지 동일**.
- **완전 동일의 메커니즘**: gaussian 렌더가 weight(=score)를 곱하므로 score<0.75인 query의 blob 최대값이 0.75를 못 넘음 → occ=0.75 이진화에서 **fg∈[0.5,0.75) query는 물리적으로 소멸**. 즉 occ threshold가 fg threshold를 흡수(占) — occ≥0.75에선 fg 스윕이 무의미.
- **운영점 결론**: fg 0.75 고정(낮추면 occ 낮을 때 크게 손해), occ는 trade-off — 0.5는 aabb +5%, 0.75는 Recall3d +23%. Recall 우선이면 0.75(현 기본값 유지). 중간값 0.6은 미측정.
- 선택 파이프라인 실체(full.py): score=binary fg 확률 그대로(w_iou=0, w_cam=0) → thr → **3m distance NMS(주 필터: 48후보→11)** → topk50(사실상 미작동). IoU head 점수는 선택에 미사용 — w_iou 활성화가 추론-만의 개선 후보.
- Recall3d 성분 로깅(comps) 추가됨 — FN vs 비관용FP 분해로 다음 개입(렌더-recall vs unmatched 억제) 결정 예정.

## 2026-07-06 KST — bbox v2 배선 후 알아둘 것
- **v2 로드는 detector-side lazy-load** (`_eval_load_bbox_gt_v2`): test Collect meta의 `scene_token`이 present `<scene>_<lidar>` 결합키라는 사실에 의존. dataset이 이 키 규약을 바꾸면 v2 로드가 조용히 실패(fallback으로 넘어감) — eval 로그에서 `IoU2d(bbox_rot)`가 NaN이면 이 경로부터 의심할 것.
- **v2 커버리지는 val 5119뿐** (train 키 없음) — train-time eval hook은 val이라 문제 없音. 기존 29k 캐시(train+val)와 달리 train 샘플로 simple_test를 돌리면 fallback(구 bboxcls)으로 감.
- **기존 bboxcls의 box 누락, val에서 13/30으로 재측정** (v2==loader 정합 확인 과정에서 발견; 이전 5/39는 train+val 혼합 표본) — 패턴상 소멸-유지(frozen pose) 미반영이 주원인으로 보임. 구 캐시 기반 bbox 수치는 이 정도로 과소평가였음.
- **재압축 완료** (int64 비압축 129G → int32 compressed **7.4G**): logs/recompress_bboxv2.log. eval 로더는 dtype 무관.
- rot GT의 OBB 채움 여유는 반voxel(`half + res/2`) — inst3d 커버 ~98–100%. 더 후하게 = gen 스크립트에서 `res/2`→`res`.

## 2026-07-06 KST — 남아있는 legacy/흔적 목록 (이번에 안 지운 것) + v2 캐시 주의
- **지운 것**: `compute_gmo_dice_loss`(query_head, 미호출), `_select_query_gt_for_losses`→`_select_query_gt_for_debug_vis` rename. **남긴 것(의도적)**: ① `*_ori.py` 3종(보존본, registry 미등록) ② loading_instance.py 640~1010행 주석 블록(옛 __call__ 보존) ③ `height_l1` 배관(항상 0.0 — eval 결과에 Height_L1로 나오지만 무의미) ④ `use_segmentation_as_query_gt` config 플래그(효과는 디버그 vis의 GT 선택뿐) ⑤ eval의 gt_occ/segmentation_bev fallback 경로(gt_inst 부재 시 안전망) ⑥ 학습 매칭의 bboxcls fallback(gt_occ_inst 부재 시).
- **v2 캐시 (`data/efficientocf_bboxcls_v2`)**: rot의 inst3d 커버가 97.8~100%로 100%가 아닐 수 있음 — annotation box가 타이트해서 nusocc 경계 voxel이 OBB(+반voxel 여유) 밖으로 나가는 케이스. 관용(Recall) GT로 쓸 때 이 ~2%는 관용 못 받음. 더 후하게 하려면 gen 스크립트의 `half + res/2`를 `half + res`로. AABB 쪽은 100%.
- **필터의 진짜 위치**: 사람 제외는 config `class_names`(ped 미포함)로 dataset instance_dict에서 이미 배제, 생성소멸·저가시성(visibility==1)은 `record_instance`가 skip → v2 생성기는 rasterize만 함. 즉 학습 GT(gt_instance_dims 등 instance_dict 파생물)와 규칙 원천이 동일.

## 2026-07-06 KST — [실측] gt_occ ↔ inst3d/bbox 불일치 + 승인 대기 중인 실행 계획 2단계
- **실측 (12샘플, present 프레임 raw↔캐시 항등좌표 대조)**: inst3d ⊂ gt_occ 100% / inst3d ⊂ bbox 100%(80프레임) / **gt_occ movable의 평균 29%(최악 70%)가 inst3d·bbox 어디에도 없음**(box 밖 잔여 — box 통째 drop된 물체 voxel로 추정). gt_occ는 생성소멸 필터도 없음(raw). → 현 IoU3d(nusocc)/Recall_3d는 모델이 절대 예측 못 하는 구조적 FN을 깔고 채점 중(IoU3d 0.03대의 원인 일부). inst3d는 생성소멸 사실상 필터됨(5/200 노이즈).
- **1단계 ✅ 적용 완료 (2026-07-06, CHANGELOG 참조)**: simple_test 3D GT를 `(gt_inst>0)`으로 교체, valid 마스크 제거. nusocc 계열 3개 메트릭 전부 학습 GT와 통일됨. 이후 eval의 IoU3d/Recall_3d는 이전 로그와 비교 금지.
- **2단계 계획 (bbox 재생성, val 5119샘플만)**: `tools/gen_data/gen_bbox_rot_gt.py` 신설 → `data/efficientocf_bboxcls_rot_v2/GMO/segmentation_rot/` per-frame `[x,y,z,cls,inst]`. get_label의 visible_instance_set(생성소멸)+query_class_ids 화이트리스트(ped·barrier 차단), rasterize만 point-in-OBB로 교체(z축 회전만이라 2D 회전 검사면 충분). 이후 로더 옵션 추가 + simple_test에 IoU2d/IoU3d(bbox_rot) 배선(기존 bbox_aabb 경로 복제 수준). 미결정: AABB 캐시도 v2로 재생성해 기존 오염(생성소멸 누수·box 누락) 제거할지 여부.

## 2026-07-06 KST — bbox 계열 메트릭의 생성소멸 누수 + rot 캐시(efficientocf_bboxcls_rot) 파악
- **bbox GT(bboxcls 캐시) 오염 정밀 재측정 (39샘플, ped 효과 분리 후)**: ① 생성소멸 누수(ped 제외 후에도 미래-신규 box) **5/39** — t≥3에서만 등장, box 1개(2~5k voxel)씩. ② 진짜 box 누락(ped 포함해도 없음) **5/39** — 단일 프레임 대형 box 누락 3건(7~31k voxel) + 전 시퀀스 누락 2건(pc_range 경계 물체). ⚠️ 초기 20샘플 스캔의 일부 "누수"는 ped box였음(bbox만 ped 제외하고 loader는 ped 포함이라 생긴 허상) — 비교 시 반드시 양쪽 클래스 기준 통일할 것. cls-only 캐시라 eval 시점 instance 필터 불가 → 근본 해결은 annotation 기반 재생성. nusocc inst3d는 5/200(거의 무결). **시각화: work_dirs/gt_align_vis/figA·figB·figC.png** (gt_occ 잔여 / 생성소멸 누수 / box 누락).
- **rot 캐시 구조** (`data/efficientocf_bboxcls_rot/GMO`, 29049 files): ① `segmentation` [x,y,z,cls] = **rotated(OBB) rasterize** (AABB 대비 voxel ~절반 — 회전 반영 확인) ② `instance_rotation` per-frame [inst_id, cls, yaw(rad), quaternion w,x,y,z] ③ `instance_bev` [x,y,inst_id]. **rotation 값 있음** ✓.
- **rot 캐시 필터 상태**: ⚠️ 클래스에 **1(barrier) 포함** — GMO 외 클래스 유입, 사용 시 query_class_ids 화이트리스트 필터 필수(현 로더의 exclude(7,)만으론 부족). ped(7) 존재 — 기존 `_filter_sparse_rows_by_class` 재사용으로 제거 가능 ✓. **생성소멸 미적용**(140/200 샘플에 미래-신규 instance).
- **rot 생성소멸 필터 가능성**: 2D는 가능 — `instance_bev`(id) + `instance_rotation`(id→cls 맵)으로 "t<time_receptive_field에 등장한 id만 유지" + ped/barrier id 제거. **3D는 `segmentation`에 per-voxel inst id가 없어 직접 불가** — ① instance_bev footprint로 z-컬럼 제거(겹침 box 과제거 위험), ② [x,y,z,cls,inst] 형태로 캐시 재생성(정공법) 중 택일 필요.

## 2026-07-06 KST — eval GT 지형도 (IoU2d(nusocc) GT 교체 시 확정한 사실)
- **GT 3계열 정리**: ① `gt_occ_inst`(data/nuScenes-Occupancy_inst3d, sparse [x,y,z,cls,inst] 512×512×40) = 진짜 nusocc per-instance voxel — 학습 dice·이번 교체 후 IoU2d(nusocc)·매칭 center의 소스. ② `segmentation`/`segmentation_bev`/`segmentation_cls_instance3d`(bboxcls 캐시) = **bbox AABB rasterize** (회전 코너를 min/max로 뭉갬) — IoU2d(bbox_aabb)·Recall3d 관용항 전용. ③ `gt_occ`(nuScenes-Occupancy dense semantic) = IoU3d/Recall3d의 3D GT.
- **필터 일치**: gt_occ_inst는 로드에서 pedestrian(cls7) 제거(`exclude_occ_class_ids`), detector `_prepare_gt_occ_inst_primary_targets`에서 query_class_ids 필터 → 학습 dice와 eval GT가 완전 동일 경로. 단 **생성소멸(`visible_instance_set`, 미래 신규 instance 제거) 필터는 bbox 렌더링(`get_label`) 전용** — gt_occ_inst sparse에는 그 필터가 없음(캐시 원본 그대로). 학습 dice도 같은 무필터 dense를 쓰므로 train/eval 일관은 유지되나, bbox 메트릭과 nusocc 메트릭의 instance 모집단이 미래 프레임에서 다를 수 있음을 기억할 것.
- **rotated bbox 메트릭 추가 시**: yaw는 어떤 캐시에도 없음. `instance_dict` annotation(translation/rotation quaternion/size) → `get_poly_region`식 egopose+ego2lidar 변환으로 회전 코너를 얻고 point-in-OBB로 rasterize해야 함 (AABB min/max로 채우면 기존 캐시와 동일해져 무의미). 생성소멸·pedestrian 필터는 `get_label` 규칙 재사용.
- **IoU2d(nusocc) 수치 단절**: 2026-07-06 이후 로그의 IoU2d(nusocc)는 이전 로그와 비교 금지(GT 교체). nusocc는 표면 sparse — bbox-AABB 대비 GT 양이 작아 수치가 내려가는 게 정상.

## 2026-07-05 KST — [probe 실측] cover ep15 feature도 크기 정보 없음 (자연 유입 가설 기각)
- 보류됐던 검증 완료 (per-sample 240s 타임아웃 버전, tools/dbg_probe/probe_feat_size.py 변형): cover ep15 matched query feature 133쌍, **best R²=-0.005, spearman≈0** (옛 모델 0.06과 동일하게 없음). bin별 예측이 buses 포함 전부 ~2.5m로 붕괴.
- **해석**: "cover로 attention이 객체를 커버하면 feature에 크기가 자연 유입된다" 가설 기각 — query feature는 attention 가중 **평균**이라 커버리지가 좋아져도 면적/extent 정보가 평균에서 소실됨(instance-pool 0.015와 같은 원리). σ-matching의 eval 이득은 feature "순도"(배경 오염 감소) 경로였지 크기 유입 경로가 아니었던 것.
- **함의 (size note 실험 해석 기준)**: 진행 중인 subset_attn_cover_size에서 ① `dbg_query_size_aux_l1_big`이 내려가면 = "감독하면 배운다"(자연 발생만 없던 것), ② 안 내려가면 = content feature로는 크기 불가가 **감독 유무 모두에서 확정** → 기하 성분(σu,σv,d)만 유효, 다음 수 = 렌더-recall + 기하 성분 중심 재설계.
- 참고: 학습 첫 iter smoke 통과 (`loss_query_size_aux: 2.98`, NaN 없음, live log = logs/subset_attn_cover_size/20260704_235120.log — 23:51 run이 본편, 23:53 중복 launch는 정리됨).

## 2026-07-05 KST — size note 구현 관련 주의 (subset_attn_cover_size)
- **from-scratch 전용**: aux head+note proj로 query_head 파라미터가 늘어 기존 ckpt 로드 불가(strict=False면 새 모듈만 랜덤 초기화되나 의도적 사용 금지). resume는 같은 config 내에서만.
- **note는 zero-init이라 초반 무영향**: 학습 초반 `loss_query_size_aux`만 내려가고 gaussian 분포 변화는 note_proj가 0에서 벗어난 뒤(수백 iter+) 시작 — 초반 BEV가 안 변해도 버그 아님.
- **벌리는 힘 없음이 의도**: 이번 run은 "쪽지만"(입력). 기대 경로 = 기존 matched-GMO dice가 크기 아는 head를 상대로 FN을 조임. 버스가 부분 개선에 그치면 그 갭이 렌더-recall(후속 스테이지)의 몫 — nusocc GT의 끝단 구멍 때문에 dice 압력이 약한 건 여전함을 기억할 것.
- **dbg 읽는 법**: `dbg_query_size_aux_l1_big`(>6m 쌍) — 초기 ~8-10(예측 2m vs 버스 12m 수준)에서 내려가야 정상. `pair_count` 대비 `big_count`로 대형 쌍 등장 빈도 감시(subset에서 ~3%). l1_big이 안 내려가면 feature에 크기 신호가 정말 없는 것 → 기하 성분(σ,d)만 남고 aux는 무용 — 그때 기하 성분의 기여를 ablation으로 분리할 것.
- **σ 성분은 no-grad·질량최대 캠 기준**: 학습·추론 공용 경로(apply_lifted 호출부)에서 계산되므로 eval에서도 동일 동작. attention 질량이 극소인 query는 σ가 노이즈지만 note가 detach라 학습 안정성엔 무해.
- **gt_instance_dims=0은 무효 마커**: annotation 미해결 instance(id 불일치 등)는 (0,0) — aux loss가 max(w,l)>0.05 게이트로 제외. 다른 config는 Collect에 안 실어 무영향.

## 2026-07-04 KST — ✅ cover eval 승리로 cls 우려 해소 + 매칭 수 해석 정정
- **eval (사용자 실행, cover 진행 중 1536/5119 시점 대조)**: cover가 **전 지표 우세** — IoU3d 0.0271 vs 0.0257(+5%), Recall3d 0.2479 vs 0.2336(+6%), IoU2d(bbox) 0.1213 vs 0.1206(동등+). → **cls +21%/focal +17.5% loss 격차는 eval 무해로 판정** (캘리브레이션 이동일 뿐). cover = 기하·loss·eval 3면 모두 확정 성공.
- **정정 (매칭 수 133 vs 89)**: 헝가리안은 유효 GT 수만큼 전부 매칭함(사용자 지적이 맞음 — 거리 게이트 `query_match_center_gate_radius_m`는 코드에 있으나 0.0=off, recruit도 off). 프로브 간 쌍 수 차이는 **각자 dataset을 다시 뽑아 랜덤 aug가 달랐던 측정 노이즈** (샘플별 양방향 요동으로 확인). "cover가 매칭을 더 한다"는 해석 철회.
- **⚠️ 프로브 방법론 교훈**: 두 모델 비교 시 **데이터를 한 번 뽑아 같은 배치를 양쪽에** 넣을 것 (cap_we_bev.py 방식). 각자 fetch하면 aug가 달라져 bin 평균/쌍 수가 흔들림 — run 내부 상관(spearman)은 유효하나 cross-run 절대값 비교는 오염됨.

## 2026-07-04 KST — ✅ cover(σ-matching) 완주 최종 판정: 크기 추종 확립, 다음 단계 GO
- **ep15 기하 (probe: dbg_out/attn_geom_cover_ep15.json, n=133)**: σ 크기 순위상관 **u 0.578 / v 0.803** (궤적 ep2 0.37/0.63 → ep5 0.47/0.73 → ep15 — 계속 상승, 대조군 ep15는 0.21/0.31). ratio 중앙값 2.13→**1.50**, x-large bin은 **0.95(사실상 목표 도달)**. small bin σ 5.5px — 우려했던 feature 해상도 바닥(~8px)보다 **아래로 조여짐**(바닥이 생각보다 낮음). inside_mass 0.57, 올바른 캠 질량 0.95/0.999 — 전 위생 지표 최고치.
- **⚠️ watch**: large bin(σ_mask~6px)이 ep5 13.1px→ep15 16.8px로 튐(ratio 2.76, n=24) — 다른 bin 전부 개선인데 이 bin만 역행. 샘플 구성 노이즈 가능성이 크나 재프로브 시 확인할 것.
- **cls +21%/focal +17.5% 열세의 해석**: cover가 **matched 쌍 133 vs 대조군 89 (+49%)** — cls loss는 matched=fg/unmatched=bg 구성에 의존하므로 **값 직접 비교가 불공정**(더 많은 쌍을 매칭하는 것 자체는 recall 관점 이득). 실질 악화 여부는 **eval IoU가 최종 심판** — 다음 단계 전 run_eval 권장. focal도 매칭↑로 fg 렌더 시도가 늘며 per-voxel 벌점이 늘어나는 구성 효과 가능성.
- **σ loss 0.35 정체 해석**: 잔여 ratio(small 3.5/mid 2.4)와 정합 — 정체가 아니라 완만한 수렴 중, 바닥 아직 아님.
- **GO 결정**: 크기 신호(σ_attn)가 attention 기하에 확립됨 → 다음 단계 = ① σ×depth→size embedding(head 입력 concat, detach), ② 렌더-recall forcing(matched instance FN-only, violators 정규화), ③ attn 계열 config에 `gt_instance_sizes` 플러밍 추가(현재 미포함 확인됨). head 입력 차원 변경이라 from-scratch run.

## 2026-07-03 KST — ✅ cover(σ-matching) ep5 관문 통과: attn 폭이 크기를 추종하기 시작, side-effect 0
- **ep5 vs ep5 동일-epoch 대조 (단일변수=σ loss; probe 데이터 dbg_out/attn_geom_{cover,attnfix}_ep5.json)**: σ 크기 순위상관 u축 0.286→**0.469**, v축 0.419→**0.728**. ratio(σ_attn/σ_mask): small 7.5→**4.0**, mid 3.9→3.2, large 2.5→2.1, x-large 1.3→1.25 — **큰 오차 구간일수록 세게 교정**되는 log-ratio 설계 그대로. inside_mass도 0.43→0.49 부수 개선.
- **궤적**: cover ep2→ep5 spearman 0.37/0.63→0.47/0.73 (계속 상승), σ loss 0.81→0.49 (붕괴 없이 하강). 대조군은 15 epoch 수렴해도 0.21/0.31 — σ 감독 없이는 절대 안 생기는 신호임이 재확인.
- **가드**: center_match 차이 ep4~6에서 −1.0~+0.3% (무손상, ep6엔 오히려 우세), dice는 전 구간 cover가 1~2% 우세(배경 오염 감소 부수효과 추정). 비용은 attn_bbox +2~3%뿐.
- **잔여 한계**: small 객체 σ 7.3px vs 목표 1.7px — feature 유사도 길이(~8px) 바닥에 걸리는 중일 가능성(예견됨). 순서/상관은 확보됐으므로 σ×depth 크기 신호로는 사용 가능 전망. mean ratio의 outlier(마스크 σ floor 걸린 초소형 쌍)는 중앙값으로 볼 것.
- **다음 관문**: cover 완주(ep15) 후 최종 프로브 → 통과 시 size embedding(σ×depth→head 입력, gt_instance_sizes 플러밍을 attn 계열 config에 추가 필요) + 렌더-recall forcing 구현 단계로.

## 2026-07-03 KST — ✅ subset_attn(collision fix) 완주 판정: 전면 개선, 이후 실험 베이스로 확정
- **로그 (정확한 ablation = fl25_sz06 ep15 vs subset_attn ep15, attn fix만 차이)**: center_match **−15.9%**(1.556→1.308), cls −16.3%, **focal −17.7%**, attn_bbox −7.9%(정의가 더 엄격한데도), dice −3.1%, depth −3.1%, total −5.7% — **전 항목 개선, 역행 0개**. 격차는 ep3부터 15까지 전 구간 일정(수렴해도 안 좁혀짐 = 구조적 개선). buggy subset_scale_offset과의 ep11 비교도 같은 결론(center −17.1%). 개선이 attn 항목에 국한되지 않음 = **매칭 cost 정확화의 연쇄 효과** — "loss 부풀림 2.2%p뿐"이라는 사전 예측은 matcher 경로를 과소평가했던 것.
- **ep15 attn 기하 probe (tools/dbg_probe/probe_attn_geometry.py 변형)**: 올바른 카메라 질량 80%→**93%(중앙값 99.8%)**, 마스크 안 질량 44%→**54%**, collision 크레딧 의존 0.022→0.006(소멸). 단 **폭은 여전히 크기-무시**(σ_attn 8.8~17px, spearman 0.21/0.31 — fix-only로 15 epoch 수렴해도 크기 추종은 절대 안 생김을 확인). ↔ cover는 ep2(σ loss 500 iter)에 이미 spearman 0.37/0.63 — **σ-matching 없이는 자연 발생하지 않는 신호**라는 대조 증거.
- **결론**: 이후 모든 실험 베이스 config = subset_attn 계열. 폭 감독(cover)의 필요성도 대조로 재확인.

## 2026-07-03 KST — 🔴 [dbg 실측] _we visible-gate spread도 우회당함 (3번째 loophole: flag-pole)
- **측정**: subset_scale_offset_we ep10, matched 158쌍 forward (scratchpad `probe_we_spread.py`, 데이터 `dbg_out/we_spread_ep10.json`).
- **결과**: visible spread ≈ 5m로 **크기 무관 균일**(corr spearman 0.017), deficit>0 **0%** → loss가 ep3부터 0.003→0.0004로 조용했던 이유 = 만족이 아니라 **우회**. 결정적 변화: **가시(w>0.75) 가우시안이 1.5~1.8개/48로 급감**(기존 raw run은 5~9개), 그 가시 1~2개가 **반경 ~9m**(기존 1.3~3.7m)에 배치됨.
- **메커니즘 = flag-pole gaming**: gate 통과 가우시안 1~2개를 멀리(9m) 꽂으면 visible-gate 2nd-moment는 target을 초과 충족. σ가 작으면 렌더 FP 비용(dice)은 voxel 몇 개 수준으로 미미한데 spread 메트릭 보상은 r² 가중이라 큼. 나머지 질량은 gate 아래(w<0.75)로 내려가 union 겹침으로 점유를 만듦 — 예견했던 '아령'+'전원-비가시' 도망의 조합.
- **교훈 (3연속 우회: σ-loophole → ghost/weight-loophole → flag-pole)**: 48-DOF mixture의 **내부 통계량**(모멘트류)을 forcing target으로 주면 점유에 안 보이는/저비용 자유도로 반드시 우회함(Goodhart). **다음 forcing은 렌더된 점유 자체에 걸어야 함** — matched instance별 one-sided recall(FN) 항 (matched_gmo 경로/128³ voxelizer 재사용, 버스 voxel을 실제로 덮어야만 감소 → 우회 불가). 기존 gmo dice는 FP/FN 균형이라 희소 대형의 FN을 못 조임 — FN-only + 크기/violator 가중 별도 항 필요.
- _we run 자체는 spread loss가 관성 0이라 baseline과 사실상 동일 학습(dice/focal 동등 확인) — 계속 돌릴 가치는 낮음.

## 2026-07-02 KST — [probe 실측] attn 기하: 중심·카메라는 정확, 폭이 크기-무시 고정 블롭 → σ-matching 구현
- **측정 (ep10, matched 148쌍, scratchpad `probe_attn_geometry.py`, 데이터 `dbg_out/attn_geom_ep10.json`)**: ① attn 질량 80%(중앙값 90%)가 GT-가시 카메라에 위치(카메라 선택 OK). ② 마스크 안 질량 44%(56% 배경 유출). ③ σ_attn/σ_mask = 2.4~3.1배 — 퍼짐 자체는 마스크보다 넓음. ④ **σ_attn이 크기 무관 8~14px 고정**(마스크 σ는 1.7→6.3px로 크기 추종) → corr(σ×depth, GT크기) spearman **0.004**. ⑤ collision 버그 inside_mass 부풀림 실측 **2.2%p**(0.457→0.436) — per-cam fix 단독의 학습 영향은 미미할 것(수정의 가치는 정확성+per-cam 감독의 기반).
- **해석**: "중심 응집"이 아니라 **크기-무시 고정폭 블롭**. 1차 모멘트(위치)는 center/depth/inside 3방향 감독인데 2차 모멘트(폭)는 어떤 loss에도 없음 → 폭이 feature 유사도 길이(~8px) 기본값에 정착. inside-mass는 '총량' 채점이라 마스크 안 웅크리기가 합법(마스크가 클수록 오히려 느슨), other-suppress(0.3)도 '남의 마스크' 금지일 뿐 폭 하한 없음.
- **조치 = σ-matching 구현(CHANGELOG 참조)**: `loss_query_attn_sigma` — 폭만 log-ratio 회귀, 위치 무접촉. `subset_attn_cover.py`에서 weight 0.25/start_iter 500 활성, 그 외 config는 0.0=off.
- ⚠️ **주의/한계**: (a) 조임의 바닥이 feature 유사도 길이에 걸릴 수 있음 — 자전거(마스크 1.7px)는 8→2px까지 못 갈 수 있고 부분 수렴도 정상, ratio_u/v가 1.0 근처가 아니라 1.5쯤에서 멈춰도 크기 '순위'가 생기면 성공. (b) σ만 맞추면 아령형 분포도 σ는 맞음 — size 용도(σ만 읽음)로는 무해, 모양 문제 생기면 정규화 soft-IoU/dice로 승급. (c) 가림/잘림 마스크는 σ_mask 과소 → min_px(4) 게이트만 있고 border-touch 제외는 미구현. (d) static_graph DDP: start_iter 게이트가 그래프를 바꾸지만 기존 손실들도 iter별 분기(z-tie 패턴)라 동일 패턴 — 첫 500 iter 전후 DDP 에러 없는지 확인.
- **부수 기대효과(실측 전 가설)**: 조이면 query feature의 배경 오염(56%)↓ → cls under-confidence(NOTES 6/16)·depth 노이즈·인접 instance 혼선 개선 여지. center는 depth 정화로 오히려 좋아질 수도.

## 2026-07-02 KST — 🔴 camera-attn loss 카메라간 픽셀좌표 충돌(collision) 버그 및 수정
- **버그**: `_compute_query_attn_bbox_loss`/매칭 cost 둘 다 예측 attn을 카메라 축 `sum`, GT를 카메라 축 `any`로 눌러 `H×W`로 비교 — 서로 다른 카메라의 같은 배열 인덱스 `(h,w)`가 같은 슬롯으로 취급됨. 카메라 extrinsic/intrinsic 자체는 GT 투영(`_project_gt_instances_to_cam_masks`)과 예측 attention(transformer cross-attn, 카메라별 pos embedding)에 다 올바르게 들어가지만, 마지막 비교 단계에서 "몇 번 카메라인지" 정보가 사라지는 구조였음.
- **수정**: `(Ncam,H,W)`를 통째로 flatten(`p=Ncam*H*W`)해서 비교하도록 `utils_loss.py`/`utils_matcher.py` 변경. GT 타겟 빌더는 이미 카메라별 마스크(`gt_inst_cam_mask_tnnhw`)를 만들어 반환하고 있었는데 소비하는 쪽이 OR'd 버전(`gt_inst_mask_tnhw`)만 쓰고 있었던 것 — 빌더 자체는 무수정, 소비부만 교체.
- ⚠️ **fix 이전에 학습된 checkpoint의 attn 관련 head/weight는 이 버그 하에서 학습된 것** — fix 이후 이어서 학습(resume)하면 loss 정의가 바뀌어 misalignment 가능. `subset_attn.py`(신규, subset.py와 동일값)로 처음부터 새로 학습해서 검증할 것.
- ⚠️ **미검증**: 합성 텐서 단위테스트만 통과(scratchpad `test_attn_fix.py`), 실제 forward/학습 smoke 미실행. 첫 학습 step에서 `loss_query_attn_bbox`/`dbg_query_attn_bbox_inside_mass_mean` 등이 NaN/shape-error 없이 나오는지 확인 필요.
- **후속 논의 (미구현)**: `inside_mass`(=`-log(Σ pred·gt_mask)`) 손실은 pred가 sum-to-1 분포라 "마스크 안 어딘가에 몰아넣기만 하면" 만족되고 마스크 전체를 덮도록(spread) 유도하는 항이 없음 — 실제로 관찰되는 point-concentration 경향의 원인일 수 있음. 대안(BCE/Focal/Tversky류 per-pixel independent 감독 등)은 아래 "camera-attn map spread" 항목 참고, 아직 미구현.

## 2026-07-02 KST — 🔴 [probe 실측] ep10 query feature에 크기 정보 사실상 없음 (사용자 가설 확인)
- **방법**: subset_scale_offset ep10, matched query feature(`query_feat_tqd`, D=96) 137쌍(14샘플)에서 GT half-extent(log) ridge 회귀, 샘플 단위 group 5-fold CV, λ 스윕(1~2e4) + shuffled-label 컨트롤. 스크립트 scratchpad `probe_feat_size.py`, 데이터 `dbg_out/feat_probe_ep10.npz`.
- **결과**: best groupCV **R²=0.06** (λ=100, 컨트롤 -0.13), Spearman≈0.11, big-vs-small 분류 acc 0.62 < 다수클래스 baseline 0.77. per-bin 예측이 tiny/car/truck/bus 전부 log-half≈0.86(≈2.4m, 승용차)으로 **일괄 수렴** — 버스(진값 5m)도 2.4m로 예측.
- **해석**: 선형 프로브 기준 ep10 feature는 크기를 못 담고 있음(하한이므로 비선형 가능성은 남지만 신호가 chance 수준). cls도 binary(fg/bg)라 크기/클래스 감독이 전무했던 상태 — "요구된 적 없음"과 "지금 없음"이 둘 다 사실로 확정. **추론 시 크기-조건부 행동이 현재로선 불가능** → 유령 loophole이 없었더라도 spread 규제가 즉시 작동하긴 어려웠을 것.
- **matched 쌍에서 bus/trailer는 4/137(2.9%)** — 데이터 불균형 정량치. `_we` 학습으로 spread gradient가 feature까지 흐르면 구분 표현이 생겨야 하며, **N epoch 후 이 프로브 재실행으로 R² 상승 여부가 "피쳐가 크기를 배우는 중인가"의 직접 지표**. R²가 안 오르고 loss_query_spread가 높게 정체하면 oversampling(버스/트레일러 포함 샘플)·클래스 감독 부활 등 데이터/감독 대응 필요.

## 2026-07-02 KST — 'visible' spread gate (`_we` config) 관련 주의
- **tau=0.1 하드코딩** (query_head.py gate). gate가 너무 가파르면(∵ w가 임계 근처에 몰릴 때) gradient 소실, 너무 완만하면 유령 재유입 — 문제 시 config 승격 검토.
- **vis_threshold = occ_score_threshold(0.75) 재사용**: occ 임계를 바꾸면 spread gate도 같이 움직임(의도된 커플링 — "점유에 보이는 것"의 정의 통일).
- **초기 학습**: w 균일(~0.5)이면 gate도 균일 → spread = unweighted RMS(≈init 산포 4~5m)라 초반 폭주 없음. weight 분화가 진행되며 gate가 조여지고 그때부터 실질 압력 발생 → `loss_query_spread`가 **초반 0 근처였다가 중반에 올라올 수 있음**(σ-metric 때의 "즉시 0"과 다른 패턴, 버그 아님).
- **모니터링**: ① `loss_query_spread`가 0으로 즉시 붕괴하지 않는지(또 loophole이면 재붕괴), ② dice/focal이 baseline 대비 안 밀리는지(이번엔 진짜 압력이라 trade-off 가능), ③ N epoch 후 dbg forward(scratchpad `dbg_offset_spread.py` 재사용, **vis 게이트 끄고**)로 corr(GT size, visible-spread)>0 확인. ④ 위반자 정규화라 loss 스케일이 이전 run보다 큼(수십 배) — 곡선 비교 시 스케일 주의.
- **d(loss)/d(w_visible)>0 부작용**: 중심 근처 가시 가우시안의 w를 살짝 낮추는 방향도 존재(spread 분모 축소) — dice가 반대 압력이라 실害 가능성 낮지만 w 붕괴 모니터.

## 2026-07-02 KST — 🔴 [dbg 실측] offset-only spread 정칙항 = 사실상 무압력 (weight-loophole 확인)
- **측정 방법**: subset_scale_offset ep1/ep10 ckpt를 동일 학습 샘플 5개(큰 객체 포함)에 forward, `_last_query_offset_spread`·`_last_inst_match_result`·mixture 텐서 캡처. 결과물 `work_dirs/subset_scale_offset/dbg_analysis/bev_weight_loophole.png`, 스크립트 scratchpad `dbg_offset_spread.py`.
- **핵심 실측 (matched 42쌍)**: offset spread(xy)가 **크기와 무관하게 전부 ~3.5m** — corr(GT half, spread) = 0.03(ep1) / **-0.15(ep10)**. 자전거(half 1.0m)도 3.4~4.1m(target의 7~8배), 버스(half 6.7m)도 3.5m(target 3.35m의 1.04배). **모든 쌍이 이미 deficit=0** → 정칙항 gradient 없음.
- **원인 = weight-loophole (σ-loophole의 weight 버전)**: spread 메트릭 `√(Σw_sur·off²)`의 w_sur는 정규화 weight라, **낮은 weight의 '유령' gaussian이 멀리 흩어져 있으면 메트릭은 충족**되지만 점유에는 안 나타남. 실제 ep10에서 개별 가시 gaussian(1-exp(-w)>0.5)은 5~9/48개뿐이고 그 반경 r_vis는 승용차 1.3~3.7m, **버스도 2.8m(car-size)**. ep1은 weight 미분화(w~0.5 균일, init 산포 4~5m) 상태였고 spread는 그때 이미 target 초과 → epoch1 초 loss 급감은 "정칙항이 offset을 벌린 것"이 아니라 init 산포가 원래 큰 것.
- **함의**: (1) 이미지 피쳐 구분력 문제라고 단정 불가 — 제약 자체가 공허(vacuous)해서 피쳐에 요구된 적이 없음. (2) **계속 학습해도 안 변함** (deficit≈0, plateau). (3) occupancy 손실은 baseline(fl25_sz06)과 동등(dice +0.01, focal은 더 좋음) — 정칙항이 무해했던 이유도 무압력이라서.
- **수정 방향(합의 전)**: spread 메트릭의 weight를 **splat-가시 질량으로 교체** — w_eff = 1-exp(-w) (poisson 단일 gaussian 피크 기여)로 2nd-moment 계산. 그러면 target을 채우려면 **높은 weight gaussian을 실제로 멀리 보내야** 하고(점유에 나타남), 작은 객체는 target이 작아 이미 충족 → 자동 크기-선택적. gradient가 offset과 weight(opacity) 양쪽으로 흐름 = "작은 객체는 opacity 떨구기" 아이디어의 올바른 방향 버전.
- ⚠️ dbg forward 부작용: 학습 vis 게이트가 열려 `vis/20260702_100726/`에 iter_005040 태그 stray PNG 21개가 남음(10:26 mtime, 삭제 권한 없어 방치). 재실행 시 visualization cfg를 꺼야 함. 또 dbg는 GPU당 ~27GB 점유 — 이 서버는 GPU당 학습 job 4개(70~160GB) 공유 중이므로 **실행 전 nvidia-smi 확인 필수**.

## 2026-07-01 KST — ✅ [제거] scale head / Lreg / budget 커플링 완전 삭제 (forcing-only만 남음)
- forcing-only 확정으로 scale head 계열(예측·Lreg·천장 커플링)이 dead code가 되어 **코드에서 전면 제거**함. repo 전체 잔존 심볼 0 검증.
- ⚠️ 따라서 **아래 scale head/Lreg/커플링 관련 주의 항목들(예측 under-size, detach, off_floor, DDP placeholder for scale head, 커플링 z 등)은 전부 무효**. 지금 남은 건 **forcing(spread 정칙항, offset-only 메트릭) + `gt_instance_sizes` 플러밍**뿐.
- 현재 큰 객체 메커니즘 = **고정 전역 캡(offset_max 12 / sigma_max 3·0.6) + spread 정칙항이 offset을 GT extent까지 밂**. scale 예측/천장 없음. eval도 offset head가 학습된 대로 뱉음(별도 scale 불필요).
- 재활성(explicit scale + 커플링) 원하면 CHANGELOG 2026-06-30~07-01 항목 참조해 재구현해야 함(코드 제거됨).

## 2026-07-01 KST — ⚠️ spread forcing 메트릭 = offset-only (σ 제외) 주의
- **왜 바꿈**: 첫 forcing run(subset_scale) 실측상 `loss_query_spread`≈0(미작동). 메트릭 `query_sigma`가 σ 포함이라 모델이 **σ만 키워** target 충족(σ-loophole). → 메트릭을 **offset-only 분산** `√(Σw·offset²)`(가우시안 중심 분포, σ 무관)로 교체. 이러면 σ 키워도 페널티 안 줄어 **offset을 움직여야만** 감소.
- **어디**: `query_head.apply_lifted_centers_to_outputs`가 `query_offset_spread_tq3` 노출 → `efficientocf._last_query_offset_spread` 캐시 → `_compute_query_spread_reg_from_match(query_spread_tq3=...)`. gradient가 offset head로만 흐름(σ head 0) — E2E 검증.
- ⚠️ **가중치 무변경**: weight 0.2·frac 0.5·mean 정규화 다 그대로(단일변수=메트릭만 교체). active config=`subset_scale_offset.py`(별도 work_dir).
- ⚠️ **희석 미해결(의도)**: 버스 희소 → mean-over-matched로 gradient/K 희석 여전. offset-metric으로 돌려서 버스 여전히 약하면 **그때** 크기가중(`Σsize·under/Σsize`)이나 weight↑ 추가. 지금은 단일변수로 효과 격리.
- ⚠️ **center 흔들림**: center 아직 학습 중(center_match 하강 중). spread는 자기-center 기준 "크기"만 강제 → 위치는 center_match, footprint는 Tversky가 잡음. 초반 노이즈 심하면 spread warmup(늦게 켜기) 고려.
- ⚠️ **옛 query_sigma 메트릭 경로 제거**됨(offset-only가 코드 기본). 첫 step에서 `loss_query_spread`가 이번엔 **0이 아니게** 뜨는지(작동 신호) 확인할 것.

## 2026-07-01 KST — ⚠️ forcing(spread 정칙항 `loss_query_spread`) 주의
- **왜 필요**: budget 커플링은 **천장만** 옮김. `offset_max=12`라 천장은 버스를 원래 안 막았으니 커플링만으론 버스 안 커짐(실측: 다 차 크기로 collapse). collapsed init + 약한 FN gradient로 가우시안이 차 크기 평형에 갇혀서, **밖으로 미는 항**(spread 정칙항)이 있어야 평형이 옮겨짐.
- **(현재 active) subset_scale = forcing-only**: 사용자 결정으로 **천장 커플링·scale head/Lreg OFF**(`use_query_scale_loss=False`,`query_scale_couple_budget=False`). offset/sigma 캡은 **기존 전역 고정(12/3·0.6)** 그대로고 **spread 정칙항(0.2)만** 작동. spread 타겟은 GT voxel-AABB라 scale head 불필요. 커플링/Lreg는 코드로 존재하나 flag OFF(재활성 가능). → 즉 지금 버스는 "고정 12m 천장 + forcing이 GT extent까지 밂"으로 커짐.
- **메커니즘**: matched query의 `query_sigma_world`(2nd-moment 반경, 프레임 평균)를 `0.5·GT_half_extent`까지 끌어올리는 one-sided relu. query_sigma가 offset²를 포함하므로 σ(캡)보다 **offset이 커져 가우시안이 분산**됨.
- ⚠️ **over-spread 재발 위험**: spread를 밀므로 Tversky FP와 길항. 안전장치 = one-sided(부족할 때만)+matched-only+GT target detach+`frac=0.5`(보수적). **그래도 `frac`↑나 weight↑ 시 카펫 재발 가능** → `loss_gmo_dice`/over-coverage·차 IoU를 함께 모니터. 재발 시 frac/weight↓.
- ⚠️ **scale head는 여전히 Lreg-only**: spread 정칙항의 gradient는 offset/σ head로만 흐름(천장 off_max_q·GT target 모두 detached). scale head는 Lreg(L1)만 받음 — 검증으로 grad 분리 확인.
- ⚠️ **query_sigma는 blob도 키울 수 있음**: σ(캡 sigma_max_q)로도 query_sigma가 부분 충족 가능 → 분산 대신 blob 성장 여지. Tversky FP가 박스 밖 spill을 막아 형상은 잡지만, 만약 "큰 blob"으로 나오면 target을 query_sigma 대신 **offset-spread(중심 분포 std)** 로 바꾸는 것 고려.
- ⚠️ **eval엔 spread 항 없음**(train-only). eval은 예측 size→천장 + 학습된 spread 거동으로 작동. 그래서 scale head(Lreg) 예측이 정확해야 eval 천장이 제대로 열림.
- gather는 `_gather_matched_gt_sizes`(Lreg와 공유, id 매칭). 첫 학습 step에서 `loss_query_scale`·`loss_query_spread` 둘 다 emit되는지·OOM·over-spread 거동 확인할 것.

## 2026-06-30 KST — ⚠️ per-query instance-scale head / Lreg / budget 커플링 주의 (subset_scale ON)
- **무엇**: query당 (world x/y/z) extent 예측 head + Lreg(L1 to matched GT voxel-AABB span) + 그 (detached) 예측으로 가우시안 offset/sigma budget을 per-query 설정(큰 객체↑/작은 객체 floor). `subset_scale.py`에 `use_query_scale_loss/query_scale_couple_budget=True`(weight 0.3). 메커니즘 전말은 CHANGELOG 2026-06-30 항목.
- ⚠️ **size 타겟 = voxel-AABB span**(진짜 box 아님): occupancy GT voxel에서 뽑은 span이라 **occlusion/range-clip 시 과소측정** 가능 → max-over-frames로 완화했지만 완전히 못 없앰. 모니터: 예측 `scale`이 클래스별로 버스>차>보행자 순서로 벌어지는지(Lreg가 수렴해도 under-size면 큰 객체 budget이 덜 큼). 안 되면 진짜 nuScenes `size`(loading_instance.py:344, 현재 미사용) plumbing 검토.
- ⚠️ **z는 sigma_z=0.6 캡 유지 + z-offset(전역 ±2.0)으로 커버**. 가우시안 중심은 `point_cloud_range` z=[-5,3]로 **하드 클램프**(query_head.py mixture_centers 직후) → 지면 위 3.5m급은 OK지만 **매우 높거나 들린 객체는 z=+3에서 잘릴 수 있음**. 단 타겟이 voxel(=in-range)에서 와서 보통은 자기-정합. 별도 z 데이터 precheck는 미실행(타겟이 범위 내라 self-consistent).
- ⚠️ **scale은 budget에서 detach** → scale head는 **Lreg로만** 학습. weight 0.3이 underfit면 큰 객체 budget이 floor에 갇힘 → weight↑ 검토. 반대로 occupancy loss는 offset/sigma head로만 흐름(기존과 동일).
- ⚠️ **budget은 전역 캡 안쪽 layer(min)** + off_floor(2,2,0.4) 하한. 작은 객체(보행자)는 floor에 머물러 sigma_z range가 0이 될 수 있음(sigma_max_q==sigma_min) → sigma 학습 정지(무해, sigma_min이 sane minimum). 
- ⚠️ **DDP static_graph**: Lreg helper가 `scale.sum()*0` placeholder를 항상 포함해 zero-match 배치에서도 scale head가 grad를 받음(없으면 'param did not receive grad'). center_only_mode/q=0 극단 엣지에선 placeholder 없이 z로 떨어짐(다른 loss도 degenerate라 허용).
- ⚠️ **eval 자동 전파**: 커플된 mixture offset/sigma가 그대로 33-tuple→simple_test/oracle에 흐름(scale은 dict/캐시로만 전달, 33-tuple 미변경이라 위치불변 unpack 안전). oracle 매칭 동기화 불필요(매칭 입력 무변).
- ⚠️ **미검증**: 전체 모델 build + forward smoke는 미실행(데이터/GPU 필요). py_compile·config 로드·핵심 수식 단위테스트만 통과. 첫 학습 step에서 `loss_query_scale` emit·shape·OOM 확인할 것.


## 2026-06-30 KST — ⚠️ Oracle 매칭 train-faithful 복제 유지보수 주의
- `_eval_train_faithful_inst_match`는 `forward_train`의 매칭 orchestration을 **복제**한 것(공유 메서드 아님 — 사용자가 학습 forward 무수정을 선택). 따라서 **`forward_train`의 매칭 준비 로직이 바뀌면 이 메서드도 손으로 동기화**해야 함. 동기화 안 하면 oracle이 학습과 달라짐.
- 동기화 체크포인트: GT prep(intersection/history_all_valid 필터), trajectory/vis slice, full center pack, ego-align(`_align_geom_pack`)·`_prepend_past_frames`, `use_full7_temporal_sup` 분기, `pool_gt_instance_context_features`+id reindex, `_select_matching_feature_frames`, `_match_queries_to_gt_instances` 인자/가중치, 그리고 `extract_feat_query` 반환 tuple 인덱스(feat_out[2,4,5,6,7,12,14,15,17,23~32]).
- attn cost(0.3) 재현 위해 `simple_test`가 oracle-train일 때만 `extract_feat_query`에 attn GT를 넘김 → attn target 생성. 이게 학습 경로와 동일 입력(primary+fallback).

## 2026-06-30 KST — ⚠️ Oracle eval(`EOCF_EVAL_ORACLE_MATCH=1`) 해석/주의
- **무엇**: query를 confidence score 대신 GT instance center에 Hungarian 매칭해 선택(GT 치팅). `eval_oracle.sh`로 실행. baseline(eval_total_2.sh)과 **반드시 같은 ckpt/occ_thr/mode**로 비교 (현재 `latest.pth`→`epoch_12_lss_only.pth` 심볼릭이라 동일 ckpt 보장).
- **해석**: oracle ≫ baseline → confidence **스코어링이 병목**(좋은 query를 못 고름). oracle ≈ baseline → 선택은 문제 아님 → **query shape/σ/trajectory 또는 매칭**이 한계.
- ⚠️ **oracle = 선택 상한선일 뿐**: 매칭된 query도 **예측 shape 그대로** 렌더 → shape 오차는 oracle로도 안 고쳐짐. "oracle인데도 낮음" = shape 문제로 읽으면 됨.
- ⚠️ **매칭은 min(Q,N_gt)개만**: 어떤 query도 안 잡은 GT는 oracle로도 못 살림(=query coverage 자체 부족 신호).
- ⚠️ **cost = 학습과 동일**(`EOCF_EVAL_ORACLE_MATCH=1`): center(10·L1/diag) + cls(0.2) + attn soft-IoU(0.3), full7 temporal, feature 게이트. `_eval_train_faithful_inst_match`가 `forward_train` 매칭을 복제(위 항목 참조). (옛 center-거리 기하 근사 oracle은 제거됨 — 학습-동일 경로만 존재.)
- ⚠️ viz(`EOCF_EVAL_VIS`)는 여전히 **score 선택** 기준 bundle을 그림 → oracle run에선 metric과 불일치하므로 `eval_oracle.sh`에서 VIS=0으로 끔.

## 2026-06-25 KST — ✅ [해소] sigma_min_m → (0.4,0.4,0.2)로 train/eval 최소 σ 통일
- 배경: `floor_vox=0.5`는 **voxel 단위**라 grid마다 물리 floor가 다름: matched 128(0.8m)→**0.4m**, eval/scene 512(0.2m)→**0.1m**. 옛 `sigma_min_m=(0.2,0.2,0.2)`에선 matched 0.4m(floor 지배)/eval 0.2m(sigma_min 지배)로 같은 σ가 train·eval에서 달라지는 미세 불일치가 있었음.
- **조치**: `sigma_min_m`을 `0.5×coarsest(matched) voxel = (0.4,0.4,0.2)`로 상향(`shape_guide_128.py`·`shape_guide_128_dice.py` 둘 다). → σ가 항상 floor 이상이라 **floor_vox가 양 격자에서 비활성**, 네트워크 σ가 train·eval에 동일하게 흐름 = **완전 일치**.
- ⚠️ **해상도 의존**: matched 해상도 바꾸면 sigma_min도 `0.5×새 voxel`로 재산정(64→0.8m, 256→0.2m). `efficientocf_config.py` DEFAULTS(0.15)는 미변경(공유) — config별로 직접 지정할 것.
- floor/sigma_min은 **over-spread(퍼짐)와 무관** — 그건 offset_max(12)/sigma_max(3)/`weight_mode='ones'` 쪽(아래 항목). 이 정리는 '소멸 방지(아래쪽)+train/eval 일관성'만 다룸.

## 2026-06-25 KST — ⚠️ shape over-spread(폭발) 근본원인: occ focal이 shape를 못 잡음 + `weight_mode='ones'`
- **증상**(shape_guide / shape_guide_128 공통): matched query 하나의 union-of-48-gaussian footprint가 차(≈4.5m)를 한참 넘어 퍼짐. 2D `query_debug_vis` row2~4(cand/sel/matched)·3D `mixture3d`·cam_gaussian 모두 거대 blob. 128 run은 occ≥0.5가 BEV 평면(~100×100m)을 카펫처럼 덮음.
- **dbg 결정 증거**: `loss_gmo_focal`이 0.14→**0.005**로 수렴하는데 shape는 폭발. 즉 **over-coverage가 focal의 최소값** → 조일 gradient 없음. `loss_gmo_dice=0`(off)·`loss_query_sigma_reg=0`(off)·`loss_query_weight_reg=0`(off)이라 **shape를 잡는 항이 하나도 안 돎**. (비교: `loss_query_center_match`~1.7–2.7). 64·128 두 run의 focal 궤적은 **동일** → 격자 변경은 폭발과 무관.
- **근본원인 A — focal이 spread를 못 벌함**(코드 검증):
  - `utils_loss.py:2634` `p.clamp(eps,1-eps)` → 카펫 **내부 p=1.0 포화**(union `1-Π(1-G)`는 gaussian 몇 개만 켜져도 1) → clamp가 gradient 0 통과. 큰 BCE 미분(1/(1-p))이 **실현 안 됨**.
  - `utils_loss.py:2660-2667` class-balanced **mean**: `loss_neg=Σ/neg.count`를 **전 격자(128³=327k 셀)로 평균** + α=0.25(neg 0.75배) + 0.5 + weight 0.1 → **FP 셀당 gradient ≈1e-7**. gradient 있는 곳은 얇은 경계 shell뿐인데, union-max라 한 gaussian 가장자리를 깎아도 **나머지 47개가 그 셀을 여전히 1로 덮음**.
  - ❗ 128은 64 대비 neg 셀 4배 → **FP 셀당 gradient 1/4로 더 약해짐**(스칼라 loss는 mean이라 동일하게 보이지만 per-cell gradient는 희석). 격자 키우면 shape는 더 안 잡힘.
- **근본원인 B(동급) — `weight_mode='ones'`**(`query_head.py:1807` `weights=ones_like(...)`, weight head 출력 폐기): 48개 gaussian이 **항상 peak=1·union**. **opacity off-switch 없음** → 잉여 47개를 끄는 길이 (sigma→floor & offset→0으로 다른 가우시안 속에 nest)뿐인 **measure-zero 기하 조건**. 완벽한 loss라도 sigma/offset만으로는 못 모음. `sigmoid_to_sigma`(min0.2~max3.0)·`tanh*offset_max(12,12,2)`엔 0으로 당기는 prior 없음.
- **vis 주의**: full-BEV 카펫은 일부 **scene/eval splat**(`voxelizer.py:598-709`, 512격자) 산물 — per-pair **학습 loss**는 grown bbox(~±21m)만 렌더. 카펫은 loss가 실제 보는 것보다 과장. 단 per-query 과확산 자체는 실재(matched blob·1σ 타원).
- **수정 우선순위**(적대 검증 후):
  1. **`use_query_gmo_dice_loss=True`(Tversky, FP-heavy α≈0.7/β≈0.3)** ← 단독 최고. `_compute_foreground_tversky_pair_loss(utils_loss.py:2690-2697)`는 FP를 **foreground 크기로 정규화**(전격자 327k 아님) → gradient 안 사라짐·격자 불변·β가 recall 방어. **단, config 주석상 GUIDE 논문 재현 위해 일부러 off**한 것이므로 trade-off 인지하고 켤 것.
  2. **`weight_mode='sigmoid'`**(opacity 복구, `query_head.py:1799` 지원) ← dice와 **함께만**. 단독은 opacity 닫으라는 gradient가 없어 무효. `weight_reg`는 `'ones'`에선 **폐기 텐서 정규화=완전 no-op**이니 `sigmoid`로 바꾼 뒤에만 의미.
  3. 기하 난간: `offset_max 12→4~6`, `sigma_max(xy) 3.0→1.5~2.0`. 싸고 안전하나 band-aid(너무 줄이면 트레일러 under-cover).
  4. `sigma_reg` 단독·focal weight↑는 **지양**: 전자는 무조건 축소→under-coverage/recall 붕괴, 후자는 포화·희석된 항을 상수배라 균형만 깸.
- **GUIDE 논문(GUIDE.pdf) 대조 — "focal only"는 GUIDE의 절반만 따라한 것**: GUIDE의 occ loss(Locc)도 focal only가 맞고(§3.5), opacity도 없음(Gaussian anchor=R^{K×10}=offset3+scale3+quat4, §3.3)이라 `weight_mode='ones'`·union·K=48(Table8)·focal-Locc는 **전부 GUIDE 충실**. 그러나 **가우시안 shape를 직접 감독하는 손실은 GUIDE도 Locc(focal) 하나뿐**(per-gaussian L1 없음) — 즉 손실로만 보면 GUIDE도 impl과 동급. GUIDE의 focal이 **작동하는 이유는 더 센 손실이 아니라 구조/초기화**: ①**instance anchor를 GT에서 k-means 초기화**(§3.2 BI∈R^{10}=pos+scale+yaw+vel, line349) → 가우시안이 처음부터 실제 객체·정위치에서 시작(focal은 *유지*만, *발견* 불필요). ②**Lreg = instance center/scale/velocity L1**(§3.5)이 가우시안의 **원점(instance center)을 GT에 고정**. ③**5-layer Gaussian decoder가 deformable 이미지 feature로 각 가우시안 anchor(offset/scale/rot)를 반복 정제**(§3.3) → 가우시안이 이미지 증거를 추종. ⚠️ Lreg는 instance box를 감독할 뿐 가우시안 offset을 기하적으로 clamp하진 않음(직전 메모의 "envelope 직접 감독"은 과장, 정정).
- ❗ **구현이 빠뜨린 것**: (a) instance **scale/extent L1 회귀가 없음**(emit되는 loss 키에 scale/box-size 항 부재; center는 `loss_query_center_match`만, `loss_query_attn_bbox`는 2D 카메라 attention bbox지 3D box envelope 아님), (b) k-means init 없음(offset=`tanh(logits)*offset_max` 고정상수 ±12m, init logits~0→offset 0으로 **중심에 뭉친 채 시작**, sigma는 중점 ~1.6m). 결과: collapsed→focal-pos가 차 덮으려 밖으로 밀고→focal-neg(희석/포화)는 overshoot 못 막음→envelope 없이 퍼짐.
- **GUIDE 충실 수정 = ①+②(opacity 아님!)**: GUIDE엔 opacity가 없으므로 `weight_mode='sigmoid'`는 GUIDE 이탈. GUIDE대로면 **per-query instance scale(wlh) 예측 head + 매칭 GT box에 L1**(Lreg) 추가가 정답, offset을 그 회귀 scale에 커플(고정 12m 대신), 가우시안 offset/sigma를 GT 기반 init. Tversky/dice는 GUIDE엔 없지만 효과적인 대체 엔지니어링 fix.

## 2026-06-24 KST — `forward_gaussian_mixture_grouped` 메모리는 `pair_chunk`가 좌우 (GT 격자 아님)

- 학습 occ-loss/매칭의 `matched_gmo_voxelizer.forward_gaussian_mixture_grouped`는 청크당 `[chunk, G, bbox]` intermediate를 만들고, 출력(`occ_tkzyx`)에 **slice 대입**(voxelizer.py:594)으로 누적 → **모든 청크의 intermediate가 backward까지 retained**. 따라서 peak ≈ (청크 수)×(청크 intermediate), K(매칭 쿼리 수)에 선형.
- ❗ **핵심**: bbox는 청크 내 pair들의 **min/max union**(voxelizer.py:542-547). 매칭 쿼리는 scene 전역에 흩어져 있어 `chunk≥2`면 bbox가 거의 **전체 격자**가 됨 → intermediate 폭증. **`chunk=1`이면 bbox가 객체 크기로 축소**되어 메모리 급감.
- GPU 실측(T=1, G=48, fwd+bwd, union, 128×128×20) **peak / ms**: K80 → c8 35GB·132 / c4 25GB·117 / **c2 9.6GB·89** / c1 1.3GB·131. K160 → c2 17.7GB·158 / c1 2.6GB·257. occ 출력 자체는 수십 MB뿐 — peak는 intermediate가 지배.
- ⚠️ `pair_chunk`는 **결과 불변**(메모리/연산 chunking만). 속도는 **chunk=2가 sweet spot**(모든 K 最速): chunk=1은 Python 루프 반복↑로 느리고(~1.6배), chunk≥4는 전체격자 낭비연산으로 느림+메모리 폭증. 메모리 최소는 chunk=1(K 무관 ≤2.6GB). **균형=2(채택), 빡빡한 GPU 안전=1.** config 키는 `efficientocf.py:3016`에서 loss로 전달됨.
- 시각화(`maybe_save_query_mixture_3d_vis`)는 쿼리당 K=1 호출이라 chunk 무관·메모리 무시 가능.

## 2026-06-23 KST — occupancy threshold 단일 출처(`eval_occ_threshold`) 주의
- occupancy 점유 판정 threshold는 이제 **`eval_occ_threshold`(기본 0.5) 한 곳**이 출처. 참조처: ①평가 metric 이진화(`efficientocf.py:1428`, env `EOCF_EVAL_OCC_THR`로만 override), ②2D `query_debug_vis` occ-grid(학습 `efficientocf.py:2943` / 추론은 ①의 `thr` 전달), ③3D `mixture3d` occ(`utils_visualization.py:1364`, eval은 env-resolved 값을 `occ_threshold` 인자로 받음).
- ⚠️ **`debug_query_mixture3d_vis_occ_threshold`는 제거됨(deprecated).** config에 남겨도 `_merge_cfg`가 non-strict라 무해하지만 **아무 효과 없음**. mixture3d occ threshold를 바꾸려면 `eval_occ_threshold`를 바꿀 것(metric도 같이 움직임 = 의도된 정렬).
- ⚠️ **가우시안 footprint 색칠**은 별개 노브 `debug_query_gaussian_prob_threshold`(기본 0.5)다. 개별 G density(`p_g≥thr`) 기준이라 합성 occupancy(`eval_occ_threshold`)와 의미가 다름 → 통일 대상 아님. 한 그림 안에 두 0.5가 공존함을 인지할 것.
- 추론 mixture3d는 `EOCF_EVAL_VIS=1`일 때 `<EOCF_EVAL_VIS_DIR>/query_mixture3d_vis/`에 생성(학습과 parity). 게이트: `_maybe_save_eval_query_vis`가 every=1·전 rank 저장으로 강제.
- ❗ **후속 TODO(미구현)**: `occ_combine_mode='union'`↔`weight_mode='sigmoid'`, `'poisson'`↔`softplus/softmax` 쌍 invariant를 config 단계에서 막는 guard/assert 없음. 잘못된 조합(union+softplus 등) 설정 가능 — 추후 `efficientocf_config.py`에 검증 추가 권장.

## 2026-06-23 KST — eval occupancy 경로(단일 ellipsoid vs mixture) 주의
- `simple_test`의 eval occupancy는 기본적으로 쿼리당 **단일 ellipsoid**(`voxelizer.forward`→`_forward_gaussian`, sigma=16개 moment-2 `sqrt(Σ w(σ²+offset²))`). **yaw를 안 씀** → 비스듬한 객체가 축정렬 타원으로 뭉개지고 빈 공간 over-cover. 학습/매칭/시각화(16개 mixture)와 occupancy 정의가 다름.
- `query_eval_occ_use_mixture=True`면 eval도 16개 mixture를 그대로 splat → 학습과 정렬(yaw/offset/sigma 다 살림). `_noreg`/`_guide`는 True.
- ⚠️ **`forward_gaussian_mixture_grouped`를 eval(512³)에 직접 쓰면 안 됨**: per-query `[T,K,D,H,W]` 할당이라 K(선택 쿼리)배로 수 GB~OOM. eval은 반드시 **`forward_gaussian_mixture_scene`**(scene 볼륨 1개에 union 누적, 쿼리별 저장 X)를 쓸 것.
- mixture는 **present 프레임만** 산출됨. eval 미래 프레임은 trajectory 중심 + **프레임-불변 offset**(`mix_c[present]−center[present]`)으로 재구성(`_eval_select_mixture_params`/`_eval_mixture_scene_occ`). present는 `mix_c[present]`와 정확히 일치.
- 쿼리 간 결합은 **max(union)**. 쿼리 내 16개 결합은 `gaussian_combine_mode`(poisson/union). eval voxelizer(`self.voxelizer`)도 config의 combine_mode를 받음.

## 2026-06-23 KST — mixture occupancy 합성 방식(poisson/union) + weight_mode 주의
- occupancy 합성은 `query_multi_gaussian_occ_combine_mode`로 선택: `'poisson'`(기본, `p=1-exp(-Σ w·G)`) / `'union'`(GUIDE, `p=1-Π(1-α·G)`). 가중-합성은 `voxelizer.py:forward_gaussian_mixture_grouped` **한 곳**뿐이고 학습 loss(`utils_loss.py:2455`)·매칭(`utils_matcher.py:123`)·시각화(`utils_visualization.py:1376`)가 공유 → 모드 바꾸면 셋 다 일관 전환.
- **union은 반드시 `weight_mode='sigmoid'`(α∈[0,1])와 함께 쓸 것.** softplus/softmax(α>1 가능)를 union에 쓰면 factor clamp가 p∈[0,1]은 지키지만 opacity 의미가 깨짐. 반대로 poisson은 softplus/softmax/sigmoid 아무거나 가능.
- voxelizer 인스턴스가 2개(`efficientocf.py`: viz용 `self.voxelizer`, 학습용 `self.matched_gmo_voxelizer`)라 **둘 다** 같은 `gaussian_combine_mode`를 받아야 viz=학습 일치. 현재 둘 다 `self.query_multi_gaussian_occ_combine_mode`에서 받음.
- 미변경(의도): query_head coverage/hit loss(`query_head.py:5052, 5204`)는 unweighted `1-exp(-Σ G)` 보조항이라 combine_mode와 무관. `voxelizer.py:_convert_occ_output`(308-316)은 trilinear forward 경로라 mixture와 무관.
- peak 회복(union α≈1 또는 추천1 weight 자유화) 시 viz `occ_threshold`는 0.2→0.5로 올릴 것(0.2는 softmax 저-peak 보정용이었음). 아래 옛 주석의 "0.2 권장"은 softmax+poisson 모델 한정.

## 2026-06-23 KST — query mixture 3D 시각화 (이식) 주의
- `maybe_save_query_mixture_3d_vis`는 `debug_query_mixture3d_vis_every>0`이고 main process일 때만 동작. iter가 every 배수일 때 `<dir>/iter_XXXXXX/mix3v_<scene>_<lidar>.png` 저장.
- 사용하는 bundle 키: `selected_points_tq3`, `selected_pred_cls_q`, `selected_score_q`, `selected_query_idx_q`, `matched_query_idx_q`, `selected_mixture_{centers_tqg3,sigmas_tqg3,yaw_tqg,weights_tqg}`. mixture 키가 없으면 점/ellipsoid는 생략되고 GT+query만 렌더(graceful degrade).
- present frame은 `time_receptive_field-1` 인덱스 기준. z축은 view별 box aspect로 과장(0.22≈2.8x) / true-z(0.078) 혼용 — 높이 구조 판독용.
- config 분리 주의: `debug_query_mixture3d_vis_every`는 **debug_cfg**(→`DEBUG_CFG_DEFAULTS`/`apply_debug_cfg`), `dir/max_queries/max_gt_points/score_threshold/occ_threshold/occ_max_voxels_per_query`는 **visualization_cfg**(→`VISUALIZATION_CFG_DEFAULTS`/`apply_visualization_cfg`). 그룹을 섞으면 `_merge_cfg`가 unknown key를 막지 않아 조용히 기본값으로 동작하므로 반드시 올바른 그룹에 넣을 것.
- 2행 레이아웃: 공통 `score_threshold`(기본 0.75)로 `selected_score_q` 게이트 후, 상단=1σ 등고면, 하단=GMM occ. 하단은 `self.voxelizer.forward_gaussian_mixture_grouped`를 query당 1개씩(K=1) 호출해 full 512³ grid에서 `p=1-exp(-Σ w_g G_g)>=occ_threshold`(기본 0.2) 복셀만 표시. voxelizer 출력은 `[T,K,1,D,H,W]`이고 `p=occ[0,0,0]`은 `[z,y,x]` → 월드 변환 시 `x=idx[:,2], y=idx[:,1], z=idx[:,0]` 주의.
- GMM occ 메모리/속도: query당 K=1 호출이라 transient는 grid 1개(≈42MB)뿐. query당 `occ_max_voxels_per_query`(기본 4000)로 서브샘플. softmax weight 모델은 보통 max p가 0.3~0.5라 occ_threshold를 너무 높이면(예 0.5) 하단이 비어 보일 수 있음 → 0.2 권장.

## 2026-06-22 KST — jhh eval 정합화 세부
- current eval은 bbox 보정 GT를 `segmentation_cls_instance3d` 우선으로 사용한다. 없으면 `segmentation_instance3d`, `segmentation`, `gt_occ_inst` 순서로 fallback한다.
- selected query는 score gate 통과 후에도 `selected_score_q`가 voxelizer weight로 들어간다. 따라서 jhh처럼 confidence가 낮은 query는 threshold 이후 occupancy가 줄어들 수 있다.
- `EOCF_EVAL_OCC_THR=0.2`로 실행하면 jhh의 `pred_occ >= 0.2` threshold와 맞는다.

## 2026-06-22 KST — current eval metric frame 범위
- 신규 eval부터 `projects/occ_plugin/occupancy/detectors/efficientocf.py`의 metrics는 present 단일 프레임이 아니라 future tail 전체 기준이다.
- 선택 query는 기존 actual inference scoring을 유지하고, 선택된 query의 trajectory geometry를 voxelizer로 렌더해 future tail pred occupancy를 만든다.
- eval PNG 시각화는 아직 present-frame query canvas다. metric과 visualization의 frame 범위가 다르므로 해석 시 주의한다.

## 2026-06-22 KST — global_idx pipeline 주의
- `global_idx`는 eval 시각화 filename/meta 용도인 scalar 값이다.
- `loading_instance.py`의 후처리 루프는 skip list에 없는 key를 모두 `torch.cat`하므로, `global_idx`는 반드시 skip list에 있어야 한다.
- 같은 패턴의 cache-hit/cache-regenerate 루프가 두 군데 있으므로 둘 다 동기화해야 한다.

## 2026-06-22 KST — Recall_3d GT 기준
- `Recall_3d`는 일반 recall `TP/(TP+FN)`이 아니라 legacy 보정식 `(TP + bbox_FP) / (TP + FN + FP - bbox_FP)`이다.
- 현재 구현에서 `TP/FP/FN`은 `gt_occ`의 nusocc 실제 3D foreground와 pred 3D voxel로 계산한다.
- `bbox_FP`만 `gt_occ_inst`에서 만든 instance별 3D AABB volume 내부의 false positive로 계산한다. height 기반 pseudo 3D는 쓰지 않는다.
- 따라서 `Recall_3d`는 bbox volume 내부 예측 FP를 맞춘 것으로 보정해 주는 지표라 일반 recall보다 높거나 직관과 다르게 움직일 수 있다.

## 2026-06-22 KST — GMO Gaussian-soft GT target 적용 주의
- `query_gmo_soft_gt_enabled=True`이면 matched GMO target은 hard binary가 아니라 **inside=1, outside=Gaussian distance-decay halo**로 바뀐다. GT를 16-Gaussian으로 새로 피팅하는 것이 아니라 기존 `segmentation_instance3d` instance mask에서 low-res target을 만든 뒤 외곽만 soft하게 확장한다.
- `query_gmo_soft_gt_sigma_vox` 단위는 원본 512x512x40 voxel이 아니라 `query_matched_gmo_bce_occ_size`의 low-res voxel 기준이다. 현재 head_gau config는 `(64,64,20)` 기준 `sigma=2.5`.
- 학습 경로에서는 SciPy EDT를 쓰지 않고 GPU tensor에서 3D max-pool dilation shell로 거리 껍질을 만든다. 따라서 정확한 Euclidean distance가 아니라 Chebyshev-like 근사 거리다. 시각화용 `visualize_soft_gt_halo.py`의 SciPy EDT 이미지와 세부 경계는 완전히 같지 않을 수 있다.
- `test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`는 soft GT만 보려는 실험 의도에 맞춰 `use_query_gmo_dice_loss=False`로 둔다. GMO shape 쪽은 focal loss 하나가 soft target을 본다.

## 2026-06-22 KST — eval 중간 metric live log
- 신규 eval 실행부터 `EOCF_EVAL_METRIC_EVERY`가 있으면 그 간격, 없으면 `EOCF_EVAL_VIS_EVERY` 간격으로 전체 rank 누적 metric을 터미널에 출력한다.
- 출력은 rank0 subset이 아니라 `dist.all_reduce`로 합산한 all-ranks 기준이다.
- `EOCF_EVAL_VIS_DIR`가 설정된 경우 같은 폴더에 `eval_metrics_live.log`가 append 저장된다.

## 2026-06-22 KST — eval 시각화 파일명
- 신규 eval 실행부터 `global_idx`가 meta에 전달되면 파일명이 `sample_000000_prob.png` 형식으로 저장된다. 8GPU 분산 속도는 유지하고 파일 정렬만 dataset index 기준으로 쉬워진다.
- `EOCF_EVAL_VIS_EVERY=48`은 여전히 rank-local step 기준 gate라, 저장되는 sample index는 전체 48 간격으로 완전히 균일하지 않을 수 있다. 그래도 파일명은 실제 dataset global index다.
- `global_idx`가 없는 예외 경로는 기존 `iter_<rank*1000000+local>_prob.png` fallback을 사용한다.

## 2026-06-22 KST — eval query 시각화 row 구성
- eval 시각화는 `_skip_traj_rows=True`로 하단 refined/base trajectory row를 생략한다. 최종 PNG는 GT class, candidates, selected(class), matched 4-row 구성이다.
- 학습 `query_debug_vis`는 기존처럼 trajectory row를 유지한다.

## 2026-06-22 KST — eval 시각화 기본 저장 경로
- `EOCF_EVAL_VIS=1`이고 `EOCF_EVAL_VIS_DIR`를 따로 주지 않으면 `tools/test.py`가 자동으로 `./work_dirs/eval/<실행 config 파일명>/<timestamp>`에 저장한다.
- `tools/dist_test.sh`는 분산 rank가 같은 하위 폴더에 쓰도록 `EOCF_EVAL_TIMESTAMP`를 실행 시작 시 한 번 export한다.
- `EOCF_EVAL_VIS_DIR`를 명시하면 기존처럼 그 경로가 우선한다. config별 자동 분리를 쓰려면 실행 커맨드에서 `EOCF_EVAL_VIS_DIR=...`를 빼야 한다.

## 2026-06-21 KST — eval 시각화(학습과 동일 캔버스) 사용법·주의
- **eval은 기본적으로 viz를 안 떨굼**(`simple_test`는 metrics만). 켜려면 env `EOCF_EVAL_VIS=1`. 출력 dir `EOCF_EVAL_VIS_DIR`, 샘플 게이트 `EOCF_EVAL_VIS_EVERY`(기본 50). `run_eval_occ_all.sh`가 이걸 자동 설정해 `work_dirs/<run>/eval/vis/`에 저장.
- **이미지 = 학습 query_debug_vis와 동일**(5-row: GT class BEV / candidates(cand·sel·drop) / selected / matched / refined-traj + legend). eval은 GT 매칭이 없어 **matched 행은 0**(정상).
- **전 랭크 저장**: 학습 viz는 rank0만 저장(`_is_main_process`). eval 헬퍼가 `query_head._eval_vis_all_ranks=True`를 set해 8랭크 모두 저장 → dataset shard 전체 커버. 파일명 `iter_{rank*1e6+local}_prob.png`로 충돌 없음.
- **"전체 5119 다 그리기"는 비현실적**: viz 렌더가 샘플당 ~0.5–1s라 다 그리면 viz만 ~6h 추가. EVERY=50이면 런당 ~100장(전 랭크). 더 필요하면 EVERY를 낮출 것.
- `.pt` 사이드카(query_vis tensors)는 여전히 rank0만 저장(`maybe_save_query_debug_vis` 내부 `_is_main_process` 게이트 유지). PNG만 전 랭크.
- viz 실패는 eval을 안 깨뜨림(try/except, `[eval_vis] skipped` 출력). metrics에는 영향 없음.

## 2026-06-16 KST — binary-fg(occ) run의 eval 임계 경고 (mcls 대비)
- **`test_traj_mcls_occ.py`(query_cls_binary_fg=True)는 cls 헤드가 2-way라 fg_prob이 구조적으로 부풀려짐**: epoch2 동일 step 기준 `dbg/query_cls_all_fg_prob_mean` occ 0.51 vs mcls(8-way) 0.18, `matched_fg_prob_mean` occ 0.66 vs mcls 0.31. 8-way는 확률이 8갈래로 쪼개져 낮고, binary는 fg 하나로 몰림.
- **결과: eval/선택의 `debug_query_score_threshold=0.5`를 occ에 그대로 쓰면 통과 query가 훨씬 많아짐**(vis에서 sel≈21, 8-way의 "1~3개"와 대조). 즉 **occ와 mcls를 동일 0.5 임계로 IOU 직접 비교하면 불공정** — occ는 recall↑/FP↑ 쪽으로 치우침. 공정 비교하려면 임계 재캘리브레이션(예: per-model 분위수 맞춤)하거나 IOU를 임계 sweep으로 볼 것.
- moving_gt_ratio=0은 occ 고유 문제 아님(mcls도 전 구간 0). 무의미 metric이므로 moving 모니터링은 `pred_mode1_ratio`로 할 것.
- **근본원인 = bg class weight(0.02)가 너무 낮음 (retrain 필요)**: weighted CE + reduction='mean'은 타깃 weight 합으로 정규화(`utils_loss.py:473`). bg=189.5·0.02 + fg=10·1.0 → 분모 13.79, **fg가 gradient mass의 72.5% 차지**(샘플은 5%). inverse-freq 균형점 bg:fg=10/189.5=**0.053**보다 0.02가 더 낮아 fg 과편향 → bg의 fg_prob가 0.45~0.55로 0.5 임계 근처까지 올라옴(mcls는 bg fg_prob median~0.10으로 임계서 멀리 떨어짐 — 이게 진짜 차이). 권장: bg weight 0.02→**0.05 시작, 필요시 0.1**, matched q10/target_fg recall 모니터링하며 과억제(all-bg 붕괴) 회피.
- **bg weight vs 시각화 fg 임계는 서로 다른 레버(등가 아님, 검증 high)**: bg weight=학습시 분포/랭킹·캘리브레이션을 이동, 임계=고정 랭킹 위 cutoff만 이동. occ의 matched(true-fg, mean 0.63, q10 0.43)와 unmatched(bg, mean 0.45)가 **겹쳐서** 어떤 단일 임계도 깨끗이 못 자름(0.6으로 올리면 true-fg 30~50% 손실). 추가로 **binary 헤드는 score_q==fg_prob이고 그 앞에 argmax(fg-vs-bg) 필터가 있어 `threshold<=0.5는 무효`(argmax와 중복), >0.5만 binding**(`utils_visualization.py:1343-1348`). 즉 임계는 PR 운영점 조정·공정비교용(무료), 근본 분리 개선은 bg weight(retrain)로.

## 2026-06-16 KST — eval(IOU) 측정 시 반드시 알 것
- **현재 detector는 eval 경로가 없었음**. `efficientocf.py`에 `forward_test`/`simple_test`를 새로 구현해서 추가함(2026-06-16, CHANGELOG 참조). 이전엔 `evaluation=dict(interval=1)`이어도 eval이 안 돌았음(텐서보드/로그에 IOU 없음) → "eval 안 됨"은 코드 부재 때문이지 데이터/설정 문제 아님.
- **반드시 분산 런처로 실행**: `tools/test.py`는 `if rank==0 and distributed:`에서만 `dataset.evaluate`를 호출함. `--launcher none`으로 돌리면 추론은 되지만 **IOU 출력이 안 됨**. 1 GPU여도 `torch.distributed.run --nproc_per_node=1 --launcher pytorch` 필요.
- **측정 metric은 present-frame movable BEV IOU**(legacy future-IOU/VPQ 아님). 현재 모델은 query→voxelizer로 present 1프레임 점유만 생성. pred = 실제추론으로 선택된 query(`_build_query_visualization_bundle.selected_query_idx_q`)를 voxelize 후 Z-max BEV, GT = `segmentation_bev`.
- **score≥0.5 임계 때문에 장면당 query 1~3개만 선택** → 절대 IOU 0.05~0.08로 매우 낮음. 모델이 under-confident(matched_fg_prob~0.47)라 임계를 넘는 query가 적음. 비교는 가능하나 절대값은 작음. 더 민감한 비교가 필요하면 `debug_query_score_threshold`를 낮추거나(실제추론과 달라짐) 샘플 수를 늘릴 것.
- **정렬(alignment) 주의**: pred(voxelizer)와 `segmentation_bev`는 좌표 규약이 달라 **transpose=True**(X/Y swap) 필요, present frame은 index **2**(=time_receptive_field-1)로 확인됨. 두 모델을 공정 비교하려면 **`EOCF_EVAL_ALIGN="2,1,0,0"`로 정렬을 강제**할 것 — 안 그러면 첫 샘플 auto-calibration이 모델마다 다른 frame을 골라 결과가 뒤집힘(실제로 frame 미고정 시 gate>mcls, 고정 시 mcls>gate로 뒤집혔음).
- traj run(`work_dirs/test_traj`)은 **체크포인트가 없어 eval 불가**(tf_logs만 존재).

## 2026-06-15 KST — Gate + Recruit 적용 관련
- `test_traj_mcls_gate.py`는 gate/recruit를 `8.0m / 15.0m / 0.1`로 활성화함. 다른 config는 기본값 `0.0`이라 기존 동작 유지.
- Gate/Recruit 거리와 pull loss는 미래 trajectory를 제외하고 matcher의 과거/현재 frame index만 사용함.
- 학습에서 `dbg/query_recruit_l1_mean` 감소, `dbg/query_recruit_count` 감소, matched count 회복 여부를 함께 확인할 것.
- 현재 기본 Python 환경에는 `mmcv`가 없어 전체 model build 검증은 실행하지 못함. py_compile, config 파싱, synthetic matcher/loss/gradient 검증은 통과.

## 2026-06-15 KST — mode-cls moving 가중 (`query_traj_mode_cls_moving_class_weight`)
- **적용 범위 주의**: weight는 `static_gate_enabled=False`일 때만 도는 평범한 mode CE 경로에만 적용됨.
  static_gate가 켜진 경로의 CE는 **moving 서브모드끼리만** 분류(static-vs-moving 아님)라 "moving 클래스 가중" 의미가 없어 미적용.
  → mcls 실험은 test_traj 설정(static_gate=False, num_modes=2, stationary=idx0)에서만 의도대로 동작.
- weighted CE + reduction="mean"은 PyTorch 규약상 **타깃 weight 합으로 정규화**됨(샘플 수 아님). 의도된 표준 동작.
- 학습 후 확인할 것: moving 예측 비율이 11% → GT 31%에 얼마나 근접하는지. 과보정 시 weight 하향(2.0 등) 검토.

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
## 2026-07-16 — `_tver` 최초 실행 재시작 필요

- 19:11 시작한 `subset_attn_cover_pyr_aabb_dice3d_new_tver` 최초 프로세스는 train `Collect3D`의
  `gt_instance_sizes` 누락 상태로 실행되어 크기 스케줄이 적용되지 않았다. TensorBoard 1~18 iter에서
  `dbg/gmo_dice_alpha_mean=0.7`, `dbg/gmo_dice_size_large_pair_count=0` 고정을 확인했다.
- config에 key를 보완했지만 실행 중 프로세스에는 반영되지 않으므로 재시작해야 한다. 재시작 후
  `large_pair_count>0`인 step에서 `alpha_mean<0.7`, `beta_mean>0.3`인지 확인할 것.
## 2026-07-22 — Rendered-direction feasibility probe

- `query_offset_only_finetune=True` is a diagnostic mode, not the final training recipe: only `query_head.gaussian_offset_head` is trainable.
- `query_shape_only_finetune=True` removes every other loss from the returned loss dictionary. It tests whether the frozen epoch-15 representation already contains enough orientation information to rotate Gaussian offsets.
- `projects/configs/baselines/EfficientOCF_V1.1_1gpu.py` is absent in this repository, so the new defaults were added to `efficientocf_config.py`; the dedicated probe config explicitly overrides them.
- A zero-valid-shape-pair rank is now handled by a zero loss connected to `mixture_centers_world_tqg3`, keeping DDP backward collective-safe. Even with this fix, do not co-run experimental NCCL jobs with a valuable job on the same GPUs without accepting shared-resource failure risk.

## 2026-07-23 — Hard matched-pair Asset FT 주의

- hard 조건은 GT 실제 `max(w,l) >= 6m` OR present-frame LiDAR/BEV 거리 `>= 30m`이며,
  matched GMO focal/Dice에만 적용된다. class/center/attention/trajectory 및 기타 regularization은
  전체 query/pair에 기존대로 적용되므로 전체 모델 gradient가 hard pair로만 제한되는 모드는 아니다.
- 시간에 따른 회전은 present Gaussian offset/sigma/yaw/weight를 미래에 그대로 복제하는 현재 구조로
  표현할 수 없어 hard 조건에서 제외했다. 이를 포함하면 실제 회전보다 축소·등방화로 타협할 수 있다.
- `dbg_gmo_hard_pair_ratio`, `dbg_gmo_hard_zero_batch`, metadata valid count를 먼저 확인할 것.
  hard 비율이 10% 미만이면 전체 LR을 올리기보다 hard sample 공급 빈도를 조정하는 편이 안전하다.
- `projects/configs/baselines/EfficientOCF_V1.1_1gpu.py`는 이 저장소에 없으므로 신규 기본값은
  `efficientocf_config.py`에 추가하고 hard FT config에서 명시적으로 override했다.

## 2026-07-23 — Dice-loss top-k hard FT 주의

- `_hard_dice.py`의 hard 선택은 각 rank/iteration에서 Hungarian matched pair의 raw Dice loss
  상위 30%를 동적으로 고른다. 점수는 `detach`하고 `ceil` 및 최소 1 pair를 적용하므로 유효 pair가
  있다면 모두 쉬운 batch여도 최소 1 pair는 선택된다.
- `query_gmo_hard_easy_weight=0.0`이므로 선택된 pair에만 GMO focal/Dice gradient가 흐른다.
  class/center/attention/trajectory 및 다른 regularization loss는 기존대로 전체 대상에 적용되므로
  모델 전체의 모든 gradient를 hard pair로만 제한하는 설정은 아니다.
- `large >= 6m`, `far >= 40m`는 `_hard_dice.py`에서 hard 선택 조건이 아니라 DBG 진단 기준이다.
  실제 선택 여부는 `dbg_gmo_hard_pair_ratio`, `dbg_gmo_hard_score_cutoff`,
  `dbg_gmo_hard_focal_raw`, `dbg_gmo_hard_dice_raw`로 확인할 것.
- 소스 수정 전에 시작된 학습 프로세스에는 새 로직이 반영되지 않는다. `_hard_dice.py` 설정으로
  새 프로세스를 시작해야 한다.
- `projects/configs/baselines/EfficientOCF_V1.1_1gpu.py`는 이 저장소에 없으므로 신규 기본값은
  `efficientocf_config.py`에만 추가하고 대상 config에서 명시적으로 override했다.

## 2026-07-30 — eval fg-score cam attention 항 주의사항

- eval score는 `(w_iou*iou_q + w_cls*cls_prob_q + w_cam*cam_attn_score_q*valid) / (w_iou+w_cls+w_cam*valid)`.
  `iou_q`는 eval에서 `inst_match_result={}`라 항상 0이며 **GT 기반이므로 절대 켜지 말 것**.
  `cam_attn_score_q`만 GT-free다(예측 mixture의 카메라 투영 vs. 자기 attention map의 KL).
- GT 미사용 검증 완료: `build_query_attn_cam_gaussian_targets`(utils_instance_img_debug.py:1487) →
  `_project_query_gaussians_to_cam_maps`(:1119)는 mixture(예측), 카메라 calib, `future_egomotion`(입력)만 쓴다.
  `_compute_query_attn_cam_gaussian_score`(utils_loss.py:2325)도 attention과 위 target만 쓴다.
- `EOCF_EVAL_CAM_SCORE=1`이어도 `EOCF_EVAL_W_CAM`(또는 config `fg_score_cam_attn_weight`)이 0이면
  score는 변하지 않는다. 현재 config 값은 `fg_score_cam_attn_weight=0.0`이므로 **두 env를 함께 설정해야** 한다.
- 유효하지 않은 query(투영 실패 등)는 `cam_attn_score_valid_q=False`가 되어 분모에서도 `w_cam`이 빠진다.
  즉 cam 항이 없는 query와 있는 query가 같은 스케일로 비교된다(0점 패널티가 아님).
- 비용: `_project_query_gaussians_to_cam_maps`는 chunk=32로 Q*G(=200*48) 성분을 카메라별 루프에서
  feat map(H*W)에 accumulate하고 카메라마다 `.item()` 동기화를 2회 한다. FLOP은 작지만 kernel launch/sync가
  많아 샘플당 수십 ms 수준의 오버헤드가 예상된다. `out_map`은 `[T,Q,N,H,W]`(≈27MB @ 1x200x6x56x100)로 상시 할당된다.

## 2026-08-08 — traincal 평가 시 반드시 지킬 것

**@800 은 순위 예측에 쓰면 안 된다.** 후보를 좁히는 용도로만 쓴다.
지금까지 다섯 번 뒤집혔다: ep19/ep20 · α0.5/0.1 vs α1.0/0.2 · α_s=0 "제거 가능" 오판 ·
res704 @800 1등이 @5119 7등 · feat128 @800 1등이 @5119 3등.
@800→@5119 편향이 조합마다 다르다(−0.0026 ~ −0.0050)는 게 원인이다.

**@5119 는 @800 보다 한 칸 높은 α_speed 를 선호한다.** 두 config 에서 같은 방향으로 나왔다
(feat128 0.2→0.3, res704 0.3→0.4). @800 으로 α_s 를 정하면 **과소평가**하게 된다.
새 checkpoint 를 튜닝할 때는 @800 봉우리보다 α_s 한 칸 위를 @5119 후보에 반드시 포함할 것.

**@5119 분해능은 약 0.0004** (1/5119). 이보다 작은 avg 차이로 조합 순위를 매기지 말 것.
res704 상위 5개는 0.00016 안에 몰려 있어 재실행하면 순위가 바뀔 수 있다.
논문에는 동점 집합 중 가장 단순한 값을 쓴다 (현재 res704 α0.6/0.3, 수치 1등은 α0.7/0.4).

**통계 파일은 checkpoint 마다 다시 뽑아야 한다.** 안 뽑으면 조용히 틀린다.
`feat128_ep19_v3_n4000.json` 은 GPU 크래시로 **3,800 에서 동결**됐다(마지막 200 구간 변화
0.06~0.25% 로 수렴 확인, 재수집 불필요). 파일명의 n4000 과 실제 표본수가 다르니 주의.

**해상도 어블레이션 수치는 baseline 인지 traincal 인지 명시할 것.**
baseline 격차 −4.4% 가 traincal 적용 후 −6.3% 로 벌어진다(traincal 이 feat128 에서 더 잘 먹힘).
