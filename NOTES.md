# NOTES

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
