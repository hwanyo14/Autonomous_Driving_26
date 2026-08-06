# RES704 Ablation — 입력 해상도 896×1600 → 256×704

- **작성일**: 2026-07-28
- **대상 config**: `projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_res704.py`
- **베이스 config**: `projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter.py`
  (파생 시점에 두 파일은 byte-identical이었음)
- **코드 변경 없음** — config만 수정. 플러그인/모델 코드는 베이스와 100% 동일.

---

## 1. 변경 요약

| # | 항목 | before | after | 성격 |
|---|---|---|---|---|
| 1 | `data_config['input_size']` | `(896, 1600)` | `(256, 704)` | 실험 변수 |
| 2 | `query_transformer_kv_resolutions` | `((14,25),(28,50),(56,100))` | `((4,11),(8,22),(16,44))` | **필수 종속 변경** |
| 3 | `query_attn_sigma_match_loss_weight` | `0.25` | `0.0` | 의도적 비활성 (§5) |
| 4 | `visualization_cfg` work_dirs 3개 | `.../..._filter/` | `.../..._filter_res704/` | 출력 충돌 방지 |

그 외 model_cfg / GT 경로 / loss weight / optimizer / lr schedule / traj warmup은 전부 베이스와 동일.

---

## 2. 수정 불필요 — input_size로부터 자동 파생되는 것들

코드를 직접 확인한 결과 아래는 손댈 필요가 없다.

| 대상 | 근거 |
|---|---|
| 이미지 resize 배율 | `sample_augmentation`이 `resize = fW/W`로 계산 — `loading_bevdet.py:190-215` |
| LSS frustum | `ViewTransformerLSSBEVDepth.py:105-106`에서 `input_size // downsample(=16)` |
| depth GT downsample | `get_downsampled_gt_depth`가 동일한 `self.downsample` 사용 |
| transformer pos embed | sin-cos를 H,W에서 동적 생성 — `transformer.py:947` |
| grid_config / occ_size / point_cloud_range | 이미지 해상도와 무관 (BEV/복셀 측) |
| GT 파이프라인 전체 | `efficientocf_gt_f3` 로딩은 이미지 해상도 무관 |

플러그인 코드 전체 grep 결과 **하드코딩된 `896` / `1600` / `56,100` 없음**.

### 정수 나눗셈 검증 (실측)

```
input_size (256, 704)  →  feat(/16) = (16, 44)
  SECONDFPN stride4 : 64x176 -> 16x44   (upsample 0.25)
  SECONDFPN stride8 : 32x88  -> 16x44   (upsample 0.5)
  SECONDFPN stride16: 16x44  -> 16x44   (upsample 1)
  SECONDFPN stride32: 8x22   -> 16x44   (upsample 2)
  kv 4x11 : exact_avgpool=True
  kv 8x22 : exact_avgpool=True
  kv 16x44: exact_avgpool=True
```

4개 FPN 브랜치 전부, kv 피라미드 3단계 전부 정수배로 떨어진다 (adaptive pool fallback 없음).

---

## 3. ⚠️ kv_resolutions — 조용히 망가지는 항목

**이게 이 ablation에서 가장 위험한 지점이다.**

`transformer.py:236`에 `kv_resolutions[-1] != context 해상도`면 ValueError를 던지는 가드가 있지만,
**`learnable_kv_downsample`이 걸려 있을 때만 동작한다**:

```python
if self.learnable_kv_downsample and (int(h), int(w)) != self.kv_full_resolution:
    raise ValueError(...)
```

그리고 `query_transformer_learnable_kv_downsample`의 기본값은 **False** (`efficientocf_config.py:217`).
즉 이 config에서는 **가드가 꺼져 있다.**

kv_resolutions를 옛 값으로 두면 어떻게 되는지 실측:

```
input 256x704 (context 16x44) + stale kv ((14,25),(28,50),(56,100))
  → 크래시 없음
  → adaptive_avg_pool2d가 16x44 를 56x100 으로 '업샘플'
  → KV token: 4224 → 33600 (8배)
```

**학습은 정상적으로 돌아간다.** 다만 KV가 자기 자신을 늘려 만든 무의미한 feature가 되고,
메모리와 시간만 8배 먹는다. 로그·에러 어디에도 안 잡히므로 **반드시 수동 확인해야 한다.**

확인 방법:
```bash
python -c "
from mmcv import Config
c = Config.fromfile('projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter_res704.py')
H, W = c.data_config['input_size']
assert (H//16, W//16) == tuple(c.model_cfg['query_transformer_kv_resolutions'][-1])
print('kv OK')"
```

---

## 4. FoV — 상단 35%가 잘린다 (의도된 동작)

`crop_h=(0.0,0.0)`은 bottom-align crop이다 (`loading_bevdet.py:210-212`).

```
resize  = 704 / 1600 = 0.44
newH    = 900 × 0.44 = 396
crop_h  = 396 − 256  = 140      # 상단 140줄 drop (원본 기준 318줄, 35%)
```

시각 확인: `data_vis/res704_preprocess/01_fov_crop_CAM_FRONT.png`

### ⚠️ 35%는 test 값 — train에서는 고정이 아니다

`data_config['resize']` jitter는 **가산**이다 (`resize = fW/W; resize += U(-0.06, 0.11)`).
따라서 학습 중 crop 비율이 매 샘플 달라진다:

| jitter | resize | newH×newW | crop_h | top-drop |
|---|---|---|---|---|
| −0.06 | 0.380 | 342×608 | 86 | **25.1%** |
| 0 | 0.440 | 396×704 | 140 | 35.4% |
| +0.11 | 0.550 | 495×880 | 239 | **48.3%** |

test는 `resize_test=0.0`이라 35.4% 고정. 아래 §4 각도 계산은 nominal 기준이며,
최악(48.3%)에서는 남는 시야 상한이 +7.8°보다 더 내려간다. 결론(볼륨 기하와 정합)은 유지.

`crop_w = int(U(0, max(0, newW−fW)))`이므로 jitter<0(≈35% 샘플)이면 crop 창이 이미지보다
넓어 **우측 최대 96px가 검정 패딩**된다. 절대폭은 base와 같고 비율만 6%→13.6%.
반대로 base는 `crop_h`가 −50까지 음수로 내려가 **상단에도** 검정 패딩이 생겼는데,
res704는 crop_h가 항상 양수(86~239)라 이 문제가 없다.
시각 확인: `data_vis/res704_preprocess/04_train_jitter.png`

베이스(896×1600)는 `crop_h = 900−896 = 4`로 사실상 full-frame이었으므로, **이 ablation은 해상도와
FoV를 동시에 바꾼다.**

### 왜 안전한가

CAM_FRONT 기준 (f≈1266, cy≈491.5, 카메라 높이 ≈1.5m):

- 남는 수직 시야: 수평 위 **+7.8°** ~ 아래 **−17.9°**
- 버리는 구간: **+7.8° ~ +21.2°**

버려지는 +7.8° 광선이 지나가는 지면 기준 높이:

| 거리 | 높이 |
|---|---|
| 10 m | ≈ 2.9 m |
| 20 m | ≈ 4.3 m |
| **≥ 24 m** | **복셀 천장 밖** |

이 config의 `point_cloud_range` z-max = 3.0 (lidar 기준, 지면 약 4.8m)이므로,
버려지는 광선은 24m 이후 **어떤 voxel에도 기여하지 못한다.** LSS가 이미지 feature를 광선 따라
볼륨 안으로 lift하는 구조라, 볼륨을 즉시 벗어나는 광선을 계산하는 건 순수 낭비다.
즉 crop은 볼륨 기하와 정합적이다.

실질 손실은 **10~20m 이내 대형차(버스/트레일러) 지붕 상단** 정도.

### 저해상도 부작용이 아니라 설계다

BEVDet/Occ3D 계열은 해상도를 올려도 **같은 2.75:1 종횡비 = 같은 35% crop**을 유지한다:

| 설정 | resize | newH | crop_h | drop |
|---|---|---|---|---|
| 256×704 (R50) | 0.44 | 396 | 140 | 35% |
| 512×1408 (SwinB) | 0.88 | 792 | 280 | **35%** |
| 640×1600 | 1.0 | 900 | 260 | 29% |

512×1408은 픽셀이 4배인데도 동일 비율을 자른다 → "가벼우려고 어쩔 수 없이"가 아니라
"이 영역은 필요 없다"는 판단. 따라서 FoV 축소는 이 ablation의 리스크가 아니다.

**출처 확인 (2026-07-28)**: BEVDet 공식 Occ3D-nuScenes baseline
`configs/bevdet_occ/bevdet-occ-r50-4d-stereo-24e.py`의 `data_config`가
`input_size=(256,704)`, `src_size=(900,1600)`, `resize=(-0.06,0.11)`, `crop_h=(0.0,0.0)`,
`resize_test=0.00`으로 **이 config와 완전히 동일**하다. 차이는 `flip`(공식 True / 이 레포는
forecasting 유지를 위해 `sample_augmentation`에서 강제 None) 하나뿐.
즉 256×704 + 상단 crop은 Occ3D 벤치마크의 표준 입력 설정이지 이 레포의 임의 축소가 아니다.

---

## 5. σ-matching을 끈 이유 (`0.25 → 0.0`)

**이게 FoV보다 훨씬 실질적인 리스크다.**

σ-matching은 "GT 객체가 카메라 이미지에서 차지하는 마스크의 폭(σ)"에 "attention map의 폭"을
log-ratio 회귀시키는 loss다 (`utils_loss.py` `_compute_query_attn_bbox_loss` 내부).
문제는 σ가 **attn map 해상도에서** 계산된다는 것:

| | attn map | 같은 객체 마스크 면적 |
|---|---|---|
| 베이스 (896×1600) | 56×100 | 예: 4 px |
| _res704 (256×704) | 16×44 | **0.3 px** (12.25배 감소) |

`min_mask_px=4` 게이트는 이 attn map 픽셀 기준이라 딜레마가 생긴다:

- **4로 유지** → 거의 모든 객체가 게이트에 걸려 loss가 사실상 안 걸림 (무해하나 무의미)
- **1로 완화** → 1~2px 마스크는 분산이 0이라 σ가 `sig_floor = 0.25`로 clamp됨.
  이 0.25는 실제 GT 폭이 아니라 log 폭주 방지용 하한인데, loss가
  `(log σ_attn − log σ_mask)²`이므로 **"attention 폭을 0.25px까지 좁혀라"는 가짜 타깃**이 된다.
  attention은 물리적으로 그렇게 못 좁아지니 gradient가 한 방향으로만 계속 밀어 **attn이 점으로 붕괴**할 위험.

저해상도에서는 2차 모멘트 추정 자체가 성립하지 않으므로 **weight를 0으로 명시적 비활성화**했다.
"우연히 안 걸림"과 "의도적으로 끔"이 로그에서 구분되도록 하기 위함.

**해석 시 주의**: 이 run은 베이스의 `_cover`(σ-matching) 실험축이 **빠진 상태**다.
베이스와의 성능 차이를 전부 해상도 탓으로 돌리면 안 된다.

모니터: `dbg_query_attn_sigma_*` 키가 **아예 안 나오는 게 정상**.

---

## 6. 남은 리스크 / 해석 주의사항

1. **단일변수가 아니다.** 이 ablation은 (a) 입력 해상도, (b) FoV(상단 25~48%), (c) σ-matching on/off,
   (d) **scale augmentation 강도** 네 축을 동시에 바꾼다. 순수 BEV IoU 비교용으로만 해석할 것.
   (d)는 `resize` jitter가 가산이라 자동으로 따라오는 것으로, 같은 `(-0.06, 0.11)`이어도
   base는 base_ratio 1.0 위라 상대 **[−6.0%, +11.0%]**, res704는 0.44 위라 상대
   **[−13.6%, +25.0%]** — 증강이 2.3배 세진다. BEVDet 표준값이라 값 자체는 정상이나,
   base와의 성능 차를 해상도만으로 귀속시키면 안 되는 이유가 하나 더 있는 셈이다.
1-b. **이름의 `_cover`는 계보 표시일 뿐이다.** 파일명/`work_dirs` 경로에 `cover`(=σ-matching)가
   남아 있지만 이 run에서는 weight 0으로 꺼져 있다. 경로만 보고 켜진 run으로 오인하지 말 것.
2. **attn/center 감독 품질 저하.** GT cam mask가 16×44에서 생성되므로 작은 객체는 마스크가
   아예 비어버릴 수 있다. `query_attn_bbox_*`와 `inside_log` 매칭 cost 품질이 전반 하락한다.
   이건 FoV와 무관하게 남는, 이 레포 고유의 저해상도 비용이다 (표준 BEVDet엔 camera-attn 감독이 없음).
3. **resume 불가.** `load_from` / `resume_from` 모두 없음(from-scratch 전제). 896 체크포인트로는
   frustum shape와 kv 관련 shape가 달라 resume 불가.
4. **`min_mask_px=4`는 그대로 뒀다.** σ weight가 0이라 무의미하지만, 나중에 σ-matching을 다시
   켤 경우 이 값부터 재캘리브레이션해야 한다.

---

## 7. 검증 내역 (2026-07-28 실행)

`eo` 환경에서 실제로 확인한 것:

- ✅ config가 `mmcv.Config.fromfile`로 정상 파싱
- ✅ `input_size/16 == kv_resolutions[-1]` (16,44) 일치
- ✅ SECONDFPN 4개 브랜치 전부 16×44로 정수 수렴
- ✅ kv 피라미드 3단계 전부 exact avg_pool (adaptive fallback 없음)
- ✅ `TransformerModule._build_layer_kv_tokens` 실제 호출 → token shape
  `[(3,264,96), (3,1056,96), (3,4224,96)]` 기대값과 일치
- ✅ stale kv 시나리오 재현 → 크래시 없이 token 33600으로 부풀어오름을 실측 (§3의 근거)
- ✅ `sigma_match_loss_weight == 0.0`, vis dir에 `_res704` suffix 반영
- ✅ BEVDet 공식 Occ3D config와 `data_config` 대조 → flip 제외 전 항목 일치 (§4)
- ✅ 실제 val 이미지로 resize+crop 재현 → `data_vis/res704_preprocess/` PNG 4장
  (생성기 `make_res704_vis.py`, `loading_bevdet.py` 로직 복제)

**미검증**: 실제 학습 1 iteration end-to-end (데이터 로딩 + forward + backward)는 돌려보지 않았다.
첫 run 시작 시 1 iteration 로그와 GPU 메모리를 확인할 것.
