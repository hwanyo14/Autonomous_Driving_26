# Query 3D Shape 표현 · 시각화 · Supervision 분석

작성: 2026-06-12. 기준 checkpoint: `work_dirs/attn_suppress_mat_q200_valid_cls_5_clscost/latest.pth` (epoch ~14).
기준 config: `projects/configs/baselines/attn_suppress_mat_q200_valid_cls_5_clscost.py`.
다른 레포에서 query 3D shape 작업을 이어가기 위한 이관 문서. 결론만 보려면 §6, §7.

---

## 1. 모델의 3D shape 표현 구조

query 하나의 3D shape은 **gaussian 16개 mixture**로 표현된다 (단일 gaussian 아님).

| 항목 | 값 | 위치 |
|---|---|---|
| 성분 수 | `query_num_gaussians=16` | config:380 |
| 성분별 sigma 클램프 | `[0.15, 1.0]m` (xyz 각축) | config:382-383 (`query_multi_gaussian_sigma_min/max_m`) |
| 성분별 yaw | 있음 (xy 평면 회전) | `query_mixture_yaw_tqg` |
| 성분별 weight | 있음 | `query_mixture_weights_tqg` |
| query head 기본 클램프 | min (0.15,0.15,0.10), max (4.0,4.0,1.5) | `query_head.py:25-26` (config가 1.0으로 오버라이드) |

핵심 설계: **개별 gaussian은 일부러 작게(≤1m) 묶고, 퍼짐/이방성은 16개 성분의 배치 + yaw로 표현**한다.
따라서 성분 하나만 그리면 항상 작은 공처럼 보이는 게 정상.

관련 텐서 (matcher 호출 kwargs 기준, `efficientocf.py:1642` 부근):

- `centers_world` `[T,Q,3]` — query 대표 center (world 좌표, m)
- `query_sigmas_world_tq3` `[T,Q,3]` — **matcher용 단일 surrogate sigma** (mixture의 대표값; 시각화에 이것만 쓰면 구형으로 보임 — §5 함정)
- `query_mixture_centers_world_tqg3` `[T,Q,16,3]`, `..._sigmas_...` `[T,Q,16,3]`, `..._yaw_tqg` `[T,Q,16]`, `..._weights_tqg` `[T,Q,16]` — 실제 shape
- `query_cls_logits_qc` `[Q,8]` — cls logits (c0=bg, c1=bicycle, c2=bus, c3=car, c4=constr, c5=moto, c6=trailer, c7=truck)
- `gt_instance_occ3d_txyz` `[T,512,512,40]` — GT instance id 복셀 (0=빈공간)
- `gt_inst_center_world_tn3`, `gt_inst_cls_n`, `gt_inst_center_valid_tn` — GT center/클래스/유효

좌표계: `point_cloud_range = [-51.2,-51.2,-5.0, 51.2,51.2,3.0]`, voxel 0.2m, grid 512×512×40, `time_receptive_field=3` (현재 frame index = 2).

**중요**: 최종 3D occupancy 출력은 query mixture가 아니라 **dense 경로(BEV seg argmax + height 회귀 → 기둥 채우기)**가 만든다
(`efficientocf_ori.py:739-776`). query shape은 matching/cls/traj/시각화용 보조 표현이다.

---

## 2. shape supervision의 실체 (왜 차 모양이 안 나오는가)

matched query mixture는 자기 GT instance와 1:1로 dice/focal을 받지만, 디테일이 거칠다:

1. **저해상도**: `query_matched_gmo_bce_occ_size=(64,64,20)` (config:379) — 본 grid의 1/8, **복셀 1.6m**.
   차(4.5×1.8m)는 이 해상도에서 3×1 복셀 덩어리.
2. **dice는 BEV 2D**: `p_bev = p.amax(dim=2)` (`utils_loss.py:1736-1737`) — z를 amax로 collapse한 뒤 dice 계산.
   **높이 방향 모양은 dice가 전혀 보지 않는다.**
3. focal/BCE는 lowres 3D (64,64,20)에서 계산 (`dbg_gmo_bce_lowres_x/y/z`로 확인 가능).
4. voxelize 경로: `SoftVoxelizerOneAdd` (`matched_gmo_voxelizer`, `efficientocf.py:207`),
   pair 구성은 `utils_loss.py:1253` (`_prepare_..._matched_pair_lowres_occ`), GT는 `gt_occ == inst_id` → adaptive_max_pool3d로 다운샘플.

**결론**: "1.6m 해상도에서 BEV 발자국만 겹치면" loss가 만족 → gaussian들이 차의 직육면체로 정렬될 유인이 없음.
관찰된 blob은 현재 loss 기준으로는 거의 정답이다 (`loss_gmo_dice` ~0.38에서 정체되는 이유).
→ 모델 capacity 문제가 아니라 supervision 디테일 문제.

---

## 3. 시각화 도구 (이 레포에 만들어둔 임시 도구 2종)

공통 캡처 방식: 학습 코드를 안 건드리고, **detector의 `_match_queries_to_gt_instances`를 wrapper로 감싸 kwargs를 가로챈 뒤**
`model.train_step(data, None)` (no_grad, model.train())으로 forward 한 번 돌려 per-sample 텐서를 얻는다.
matched 정보는 forward 후 `det._last_inst_match_result["matched_query_idx"]`.

### 3.1 `vis/score_thr_sweep/run_vis.py` — BEV scoring 스윕

- GT 복셀 footprint + GT center + 필터 통과 query를 BEV 산점도로, 6개 설정 패널 비교
- 필터 모드 3종: `argmax` (argmax≠bg + fg thr), `bg<thr` (+ fg thr), `margin` (fg_max−bg ≥ thr, 정렬도 margin)
- 산출물: `sample_*.png` (저threshold 스윕), `hithr_sample_*.png` (0.35~0.45), `margin_sample_*.png` (margin 스윕)

### 3.2 `vis_3d/run_vis3d.py` — 3D 렌더

- GT instance 복셀 (tab20, instance별 색) + GT center(초록 X+클래스라벨)
  + margin≥0.15 통과 query (클래스색 구체, 검은 테두리=matched, 텍스트=margin 값)
  + **mixture 16성분**: 성분 center 점 (크기∝weight) + weight 상위 6개 성분의 yaw 반영 1σ wireframe ellipsoid
- view 3개: oblique(z×2.8 과장), oblique rear(z×2.8), low side(**진비율** z aspect 0.078=8m/102.4m)
  — z 과장 이유: 진비율로는 장면이 종잇장이라 높이 구조가 안 읽힘. `views` 리스트의 4번째 값으로 조절.
- 산출물: `sample_*_3d.png`(surrogate sigma만, 폐기), `mix_sample_*_3d.png`(mixture), `mix3v_sample_*_3d.png`(3-view 최종)

### 실행 방법 (이 서버 기준)

```bash
PYTHONPATH=/home/hwanhee/Autonomous_Driving_26_cost \
CUDA_VISIBLE_DEVICES=<여유 GPU> \
/home/hwanhee/anaconda3/envs/eo/bin/python vis_3d/run_vis3d.py
```

- python은 반드시 conda env `eo` (시스템 python에 torch 없음), repo root를 PYTHONPATH에.
- 학습 8 GPU 동시 점유 중이라 모델 로드+3샘플에 10~15분. CPU RAM 빡빡하므로 `workers_per_gpu=0`, 샘플 단위 처리 유지.
- 체크포인트 로드 시 voxelizer buffer 등 unexpected key warning은 무해.
- train dataset은 호출마다 같은 idx라도 다른 시퀀스 구성이 나올 수 있음(파이프라인 비결정성) — 패널 내 비교만 유효.

---

## 4. scoring(시각화 선별) 분석 결과 — margin 채택

현재 학습 vis 파이프라인의 선별: `argmax≠bg 필터 → score(=bg 제외 fg max prob) 정렬 → 3m xy NMS → top-50`
(`utils_visualization.py:1332-1374`; cls_5 계열 config는 `cls_weight=1.0, iou=0, cam=0, threshold=0`).

스윕으로 확인된 사실:

1. **fg score threshold는 0.3 이하에선 무의미** — argmax≠bg를 이긴 query는 대부분 score 0.3+.
   0.35~0.45는 실제로 자르지만 TP/FP score가 겹쳐 있어(matched 0.3~0.5, FP 0.3~0.58) TP도 같이 죽고, 희귀 클래스가 가장 먼저 죽음.
2. **수학적 관계**: `argmax≠bg ⟺ margin>0`, 그리고 `fg_max>bg ⟹ bg<0.5` (즉 argmax 필터에 bg<0.5 추가는 완전 중복).
   `bg<thr` 단독은 argmax보다 느슨한 방향으로만 의미 (bg 0.35~0.5에서 fg가 분산된 "거의 맞춘" matched query를 드러냄).
   `thr ≥ bg_thr`이 되면 두 룰은 수렴.
3. **margin(fg_max − bg)이 최우수 분리축**: FP는 fg와 bg가 같이 높아 margin이 작고(예: fg .45/bg .40 → +0.05),
   진짜 객체 위 query는 clscost 학습이 bg를 눌러놔 margin이 큼(fg .45/bg .20 → +0.25).
   margin 0.10~0.20 구간에서 **matched 무손실로 경계 FP만 제거**됨을 3샘플에서 확인.

**채택: `margin ≥ 0.15`, 정렬도 margin, 3m NMS, top-50.** 진단용은 `margin ≥ −0.10` (argmax가 숨기는 bg-경합 matched 표시).
참고: 본 파이프라인의 score 식 `(w_iou·IoU + w_cls·cls)/Σw`로는 −bg 항을 표현할 수 없어, 정식 채택하려면
`debug_query_score_mode='margin'` 같은 옵션을 `utils_visualization.py`에 추가해야 한다 (이 레포에는 미적용).

---

## 5. 3D 관찰 결과와 함정

1. **"구형으로 보인다" 함정**: surrogate 단일 sigma(`query_sigmas_world_tq3`)만 그리면 yaw 없는 ≤1m 타원체라 공처럼 보인다.
   실제 표현은 mixture — 반드시 mixture 텐서로 그릴 것.
2. mixture 렌더 결과: 성분들이 query 주변 2~4m로 흩어지고 ellipsoid가 겹쳐 차 크기의 외피는 만들지만,
   **차의 직육면체/방향성이 뚜렷한 모양은 아님** (blob). 원인은 §2의 supervision 거칠기.
3. truck/희귀 클래스 GT 옆에는 margin 통과 query가 없음 — cls 희귀 클래스 문제(별도 트랙: cls_7 실험)와 일관.
4. z축 시각화는 과장 배율을 명시할 것 (진비율 0.078은 판독 불가, 0.22≈2.8x가 적당).

---

## 6. 핵심 결론

- query 3D shape이 차 모양이 아닌 것은 **정상 동작** — supervision(1.6m lowres + BEV-only dice)이 그 이상을 요구하지 않는다.
- 시각화 선별 기준은 **margin(fg−bg) ≥ 0.15**가 현재 모델 상태(클래스 확률 calibration)에 가장 잘 맞는다.
  margin 분리력은 clscost(matcher cls cost)가 만든 fg/bg crossover 구조에 기대고 있으므로, cls supervision이 좋아질수록 더 좋아진다.
- shape 자체는 최종 출력 경로(dense head)와 분리되어 있으므로, **shape 정밀화가 필요한지는 query shape의 용도에 달려 있다** (§7).

---

## 7. 다음 단계 옵션 (다른 레포에서 진행 시)

**선결 질문: query 3D shape을 무엇에 쓸 것인가?**

### A. 보조 표현으로 유지 (매칭/cls/traj용)
- 아무것도 안 바꿔도 됨. blob이어도 기능에 문제없음. shape에 gradient를 더 쓰는 건 오히려 손해일 수 있음.

### B. instance shape 출력으로 승격 (query 기반 instance occupancy)
비용이 낮은 순서:
1. **dice를 3D로**: `utils_loss.py:1736` `amax(dim=2)` 제거 → z 모양이 처음으로 supervise됨. 한 줄 수정.
2. **lowres 해상도 상향**: (64,64,20) → (128,128,40). 메모리/속도 측정 필요 (voxelizer가 pair마다 호출됨).
3. **bbox prior loss**: GT bbox로 "bbox 밖 mixture density penalty" — 기존 `attn_bbox` loss와 같은 패턴이라 구현 결이 맞음.
4. sigma_max 상향(1.0→1.5~2.0)이나 성분 anchor 배치는 1~3 이후에.
- 판단 보조 도구: 학습과 동일한 voxelize(64,64,20 + BEV amax)로 pred/GT를 나란히 렌더하면 "loss가 보는 그림"을 직접 확인 가능 (이번에 미구현, 다음 단계로 적합).

### 검증 지표 (TB dbg)
- shape: `loss_gmo_dice`, `dbg_gmo_dice_pair_count`, `dbg_gmo_bce_lowres_x/y/z`
- 분리/선별: `dbg_query_cls_matched_fg_prob_*` vs `matched_bg_prob_*` (margin의 원천),
  `dbg_query_cls_nms_*` (3m NMS 기준 cls 통계 — 단, 정렬이 bg 포함 max라 vis와 모집단 다름에 주의)
- sigma: `dbg_query_sigma_mean_m`, `loss_query_sigma_reg`

---

## 8. 산출물 위치 (이 레포)

```
vis/score_thr_sweep/run_vis.py            # BEV scoring 스윕 도구
vis/score_thr_sweep/sample_*.png          # fg/bg thr 스윕 (1차)
vis/score_thr_sweep/hithr_sample_*.png    # 고threshold 스윕
vis/score_thr_sweep/margin_sample_*.png   # margin 스윕 (채택 근거)
vis_3d/run_vis3d.py                       # 3D mixture 렌더 도구
vis_3d/mix3v_sample_*_3d.png              # 최종 3-view 렌더
vis_3d/3D_SHAPE_ANALYSIS.md               # 본 문서
```
