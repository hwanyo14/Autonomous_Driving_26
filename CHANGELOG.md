# Changelog

## 2026-06-25 KST — eval 3D mixture vis 전용 해상도 키 신설 (eval=512³ GT/metric 일치, train=128 유지)

- **문제**: 3D `mixture3d` vis가 train·eval 모두 `matched_gmo_voxelizer`(128, 0.8m)를 써서 그림. 그런데 eval **메트릭**은 `self.voxelizer`(512³, 0.2m)로 계산 → 같은 eval 런서 점수(0.2m)와 cube(0.8m) 해상도 불일치.
- **변경**: eval mixture3d vis만 GT/metric 해상도로 렌더하도록 분리. train vis는 128 유지(loss 격자=loss가 보는 것, every-48 상시라 속도 위해).
  - 신규 config 키 `debug_query_mixture3d_vis_eval_occ_size=(512,512,40)` ([efficientocf_config.py](projects/occ_plugin/occupancy/detectors/efficientocf_config.py): 기본값+apply, [shape_guide_128_dice.py](projects/configs/baselines/shape_guide_128_dice.py) 노출).
  - `efficientocf.py.__init__`: 그 키로 `self.mixture3d_eval_voxelizer` 생성(floor=0, 고해상; range_x/y/z 재사용).
  - `maybe_save_query_mixture_3d_vis`에 `is_eval: bool=False` 파라미터 추가 → eval일 때 eval voxelizer, train일 때 matched(128). **영속 플래그(`_eval_vis_all_ranks`) 대신 명시적 파라미터**로 train-after-eval 오염 방지.
  - eval 호출부 2곳([efficientocf.py:1381,1389](projects/occ_plugin/occupancy/detectors/efficientocf.py))만 `is_eval=True`; train 호출부(3142)는 미전달=128.
- **안전**: per-query bbox 렌더(K=1)+`occ_max_voxels_per_query=4000` 캡으로 512에서도 메모리/matplotlib 부담 bounded. eval은 가끔 도므로 OK. train은 미변경이라 속도 영향 0.
- **주의**: **실행 중인 dice 런엔 영향 없음**(런치 시 로드된 코드 사용). 새 런/eval부터 적용.
- 검증: 4개 파일 `py_compile` 통과. `is_eval=True`는 eval 2곳만, train 1곳 미전달 확인.

## 2026-06-25 KST — 헷갈리는 죽은 흔적 제거(sigma/score 관련 dead code·deprecated key)

- **(1) 레거시 단일-gaussian sigma 인자 제거**: `query_head.py` `gaussian_sigma_min`/`gaussian_sigma_max`(__init__ 파라미터 388-389 + `self.` 할당 577-578). **할당만 되고 어디서도 안 읽힘**(mixture 경로는 `query_multi_gaussian_sigma_min_m/max_m` 사용, efficientocf도 이것만 전달). 헷갈림 유발 → 삭제.
- **(2) deprecated 시각화 score 키 제거**: `efficientocf_config.py` `debug_query_mixture3d_vis_score_threshold`(DEFAULTS 키 + apply 할당). 이미 `debug_query_score_threshold`(+`EOCF_EVAL_FG_THR`)로 통일돼 무시되던 키. config에서도 미사용 확인 후 삭제. 관련 stale 주석(`utils_visualization.py:1335`)도 정리.
- **(3) `sigmoid_to_sigma_from_range` stale 기본값 제거**: 인자 default `(0.15..)/(4.0..)`가 실제(0.4/3.0)와 달라 오해 유발 + 유일 콜러(`query_head.py:1781`)가 항상 명시 전달 → default 삭제(필수 인자화).
- **남겨둔 것(미삭제, 의도)**: `query_multi_gaussian_sigma_min_m/max_m`의 DEFAULTS `(0.15..)/(4.0..)`(`efficientocf_config.py:108-109`, `query_head.py:392-393`)는 **functional fallback**(키 없는 config 대비). 모든 실제 config가 명시 override하므로 효과 없음 — 제거 시 KeyError 위험이라 유지.
- 검증: py_compile OK, 제거 심볼 잔존 참조 0. ⚠️ 진행 중 학습 무관(기능 무변).

## 2026-06-25 KST — eval voxelizer(512) floor 제거(0.0), train(matched) floor는 0.5 유지

- **대상**: `efficientocf.py:217-224` `self.voxelizer`(eval/512) → `gaussian_sigma_floor_vox=0.0`. `efficientocf.py:236` `matched_gmo_voxelizer`(train) → 상속 끊고 `0.5` 명시.
- **이유**: eval은 forward σ를 그대로 splat — sigma_min(head, ≥0.4m)이 이미 보장하므로 0.2m 격자에서 소멸 위험 없음 → 별도 floor 불필요(요청). floor의 '소멸 방지' 역할은 거친 train 격자(0.8m)에서만 필요.
- **영향(기능)**: **사실상 무변화(no-op)**. sigma_min=0.4m라 self.voxelizer floor(옛 0.1m)는 이미 dormant였음(σ 0.4m > 0.1m). self.voxelizer.floor를 공유하던 `utils_bev_pool`도 0이 되나 동일하게 dormant. 코드 의도("eval=raw forward")만 명확화.
- **주의**: 이후 sigma_min을 0.1m 밑으로 내리면 eval에서 sub-voxel 소멸 가능(현재 0.4m라 안전). py_compile OK. ⚠️ 진행 중 학습 미반영.

## 2026-06-25 KST — `sigma_min_m` (0.2,0.2,0.2)→(0.4,0.4,0.2): train/eval 최소 σ 통일

- **대상**: `shape_guide_128.py:451` `query_multi_gaussian_sigma_min_m`. (`efficientocf_config.py` DEFAULTS `(0.15,0.15,0.10)`는 다른 해상도 config가 공유 → 미변경. 각 config는 `0.5×자기 matched voxel`로 둘 것. ⚠️ `shape_guide_128_dice.py`도 같은 변경 필요 — 아래 별도 항목/확인.)
- **이유**: `floor_vox=0.5`는 voxel 단위라 train(matched 0.8m→**0.4m**)·eval(512 0.2m→**0.1m**) 물리 floor가 달라, σ<0.4m 구간에서 **train은 σ를 0.4m로 부풀려 학습 vs eval은 원값 렌더 → 미세 불일치**. `sigma_min_m`을 `0.5×coarsest(matched) voxel = (0.4,0.4,0.2)`로 올리면 σ가 항상 floor 이상 → **양 격자에서 floor 비활성, 네트워크 σ가 train·eval에 동일하게 흐름.**
- **효과**: train/eval 최소 σ 완전 일치(xy 0.4m / z 0.2m). 잃는 디테일 없음 — 모델은 어차피 0.8m matched 격자로만 학습(sub-0.4m σ는 학습 안 된 noise였음).
- **무관**: over-spread(퍼짐)와 별개(그건 offset_max·sigma_max·`weight_mode='ones'`).
- **주의**: matched 해상도 바꾸면 sigma_min도 `0.5×새 voxel`로 재산정(예: 64→0.8m). py_compile OK. ⚠️ 진행 중 학습 미반영 — 다음 실행부터.

## 2026-06-25 KST — `shape_guide_128_dice.py` 신규: occ shape over-spread 대응 dice(Tversky) 단독 실험

- **신규 config** `projects/configs/baselines/shape_guide_128_dice.py` ([shape_guide_128.py](projects/configs/baselines/shape_guide_128.py) 클린 카피에서 **단 한 가지만** 변경).
- **변경 내용**(`model_cfg`, 365–372줄):
  - `use_query_gmo_dice_loss=False → True` (occ loss에 foreground-Tversky 추가).
  - `query_gmo_dice_loss_weight=0.5`, `query_gmo_tversky_alpha=0.7`(FP 가중), `query_gmo_tversky_beta=0.3`(FN 가중) **명시**(값은 [efficientocf_config.py:8-10](projects/occ_plugin/occupancy/detectors/efficientocf_config.py#L8-L10) 기본과 동일 — 자기문서화 목적, 동작 불변).
  - `query_gmo_loss_type='focal'` 유지 → focal+dice 병행(focal=불균형, dice=shape 주력).
- **근거**: focal-only는 over-spread에서 `p.clamp` 포화 + 전격자 mean 희석으로 FP gradient≈0 → shape 못 잡음(폭발). Tversky는 FP를 foreground 크기로 정규화 → 격자 불변·비소멸 gradient, β가 recall 방어. 진단 전말은 [NOTES.md](NOTES.md) 2026-06-25 항목.
- **단일 변수 원칙**: 제한값(`offset_max=12`/`sigma_max=3`)·`weight_mode='ones'`·init **모두 그대로**. dice 효과만 귀속. 코드 변경 없음(메커니즘은 [utils_loss.py:2670-2697](projects/occ_plugin/occupancy/detectors/utils_loss.py#L2670-L2697) `_compute_foreground_tversky_pair_loss`로 기구현·wired).
- **검증**: `py_compile` 통과, 추가 키 3개 모두 `efficientocf_config.py`가 인식. **모니터**: `loss_gmo_dice`가 ~0.5에서 감소하는지 / matched footprint가 차 크기로 조여지는지 / recall(matched count) 유지.

## 2026-06-25 KST — `gaussian_sigma_floor_vox` 기본값 0.35→0.5 (σ 하한을 '반 voxel 방지턱'으로 정정)

- **대상(live 3곳, 0.35→0.5)**:
  - `voxelizer.py:139` `SoftVoxelizerOneAdd.__init__` 기본값(+근거 주석). `self.voxelizer`·`matched_gmo_voxelizer` 모두 이 기본값을 상속(전자는 미지정, 후자는 `float(self.voxelizer...)`).
  - `query_head.py:4618` `compute_query_point_dt_loss`의 floor 인자 기본값(콜러 `utils_loss.py:3672/3677`가 안 넘겨서 default 사용).
  - `utils_bev_pool.py:340` getattr fallback(실사용은 voxelizer attr=0.5라 사실상 표기 일치용).
  - dead 백업 `query_head_ori.py`는 미사용(import X)이라 제외.
- **의미**: floor는 σ가 voxel보다 작아져 grid에서 사라지는(occ 0 → dead-gradient) 것을 막는 **방지턱**. voxel 단위라 해상도 자동 추종(0.5=반 칸 → 128:0.4m / 64:0.8m / 256:0.2m). 엄밀 안전선(최악 정렬서 nearest voxel≥0.5)이 ~0.42 voxel이라 기존 0.35는 살짝 미달 → 0.5로 정정.
- **실효 σ 하한 = max(sigma_min_m, 0.5·voxel)**. `sigma_min_m=(0.2,0.2,0.2)`는 **사용자 요청대로 유지** → 128 matched(0.8m)에선 floor(0.4m)가 지배, eval 512(0.2m)에선 0.1m라 sigma_min(0.2m)이 지배(미세 불일치는 [[NOTES]] 참조). sigma_min_m을 0.4↑로 올리면 floor 비활성 + train/eval 완전 일치.
- **주의**: 이번 변경은 σ '소멸 방지(아래쪽)'만. shape **over-spread(퍼짐)** 와는 무관(그건 offset_max·sigma_max·`weight_mode='ones'` 쪽).
- config 주석 갱신: `shape_guide_128.py`의 floor 수치(0.28m→0.4m)·sigma_min 바닥 설명. 검증: py_compile 4파일 OK. ⚠️ 진행 중 학습 미반영 — 다음 실행부터.

## 2026-06-25 KST — `query_multi_gaussian_pair_chunk` 기본값 8→2

- 대상: `projects/occ_plugin/occupancy/detectors/efficientocf_config.py:112` (DEFAULTS). `8` → `2`.
- 이유: grouped voxelizer 루프 보폭(메모리/속도만, **결과 불변**). GPU 실측상 128격자에서 chunk=2가 속도 sweet spot + 메모리 적당(K80 9.6GB). 8은 128에서 ~35GB(OOM)·느림.
- **파급(override 없는 config)**: `shape_guide.py`(64격자)도 이제 기본 2 사용 → 64에선 chunk=8(45ms·10.5GB)이 最速이었으므로 **약간 느려지지만(71ms) 메모리 1/3(3.3GB)**. 결과/정확도 영향 0. `shape_guide_128.py`는 명시적으로 2라 무변화.
- 동기화: `EfficientOCF_V1.1_1gpu.py`는 현재 tree에 없음(불필요). py_compile·mmcv 로드 확인. ⚠️ 진행 중 학습 미반영 — 다음 실행부터.

## 2026-06-24 KST — mixture3d 2행: 예측 occ를 '예측 해상도 솔리드 큐브'로 (점 scatter → ax.voxels)

- 대상: `projects/occ_plugin/occupancy/detectors/utils_visualization.py` `maybe_save_query_mixture_3d_vis` (학습·추론 **공유** 함수 → 양쪽 다 적용).
- **변경 전**: mixture를 512-res(`self.voxelizer`)로 재복셀화 → 점유 복셀 센터를 sparse 점(s=1.4, alpha=0.18, query당 4000 subsample)으로 scatter → "퍼진 점"처럼 보임.
- **변경 후**: **모델 예측 해상도(`matched_gmo_voxelizer` = `query_matched_gmo_bce_occ_size`, 현재 64×64×20)** 로 복셀화 → 점유 복셀을 **실제 스케일 솔리드 큐브**(`ax.voxels`)로. query별 클래스색 유지, GT 회색 구름 레퍼런스 유지, 1행(1σ ellipsoid) 무변경.
  - **해상도 자동 적응**: 큐브 크기 = range/res → 64→1.6m, 128→0.8m. config(예: `shape_guide_128.py`)만 바꾸면 자동. 별도 vis 격자 기준 불필요.
  - occ 임계값은 통일된 `eval_occ_threshold`(+`EOCF_EVAL_OCC_THR`) 그대로 사용.
  - 속도: 예측 해상도라 큐브 수 적음(측정 64-res 43~949개 23~243ms / 128-res 115~3798개 45~611ms). 512-res 큐브(12만 개, 초 단위) 대비 빠름.
- **검증**: 매칭 48-gaussian 체크포인트(`work_dirs/shape_guide/epoch_1_lss_only.pth`)로 eval 3샘플 → `query_mixture3d_vis/iter_*.png` 2행이 솔리드 큐브로 정상 렌더(렌더 에러/OOM 0). py_compile 통과, dead 참조(gmm_xyz/occ_max_vox) 0.
- 주의: 진행 중 학습은 로드된 옛 코드라 미반영 — 다음 실행부터. (`debug_query_mixture3d_vis_occ_max_voxels_per_query`는 이제 미사용.)

## 2026-06-24 KST — shape_guide_128.py: 학습 occ-loss 격자 64→128(xy)

- **신규 실험 config** `projects/configs/baselines/shape_guide_128.py` (shape_guide.py 복제본). 단일 변경: `query_matched_gmo_bce_occ_size=(64,64,20)` → `(128,128,20)`.
- **배경**: 학습 occ-loss는 `matched_gmo_voxelizer`(loss 격자)에서, 평가 metric occupancy는 `self.voxelizer`(512)에서 가우시안 직접 splat. 64-grid(x,y 1.6m)에선 sigma floor(0.35vox)가 0.56m로 작동 → sigma_min_m=0.2m을 압도, σ_x,y∈[0.2,0.56m] 학습 dead-zone. 또한 자동차 폭이 ~1복셀이라 형상 감독 불가.
- **효과**: 128(x,y 0.8m)로 올리면 floor 0.56→0.28m(dead-zone 축소), 자동차 폭 ~2복셀로 형상 감독 가능. 단 x,y 실효 최소 σ=0.28m로 sigma_min(0.2m)보다 약간 높음(0.2m 도달하려면 격자≥256 필요).
- **해상도 분리 주의**: 평가 metric occupancy = 512 직접 splat(불변, 업스케일 없음). 단 3D mixture **디버그** 시각화(`maybe_save_query_mixture_3d_vis` 2행)는 최근 변경으로 **예측 해상도(`matched_gmo_voxelizer`)** 솔리드 큐브 렌더 → 128 적용 시 큐브 1.6m→0.8m 자동 적응(별도 수정 불필요).
- **메모리·속도 측정 + pair_chunk=2 추가**: grouped voxelizer가 청크당 `[chunk,G,bbox]` intermediate를 backward까지 retain. 흩어진 query를 묶으면 bbox≈전체 격자 → `query_multi_gaussian_pair_chunk` 기본 8이면 128격자에서 ~35GB(OOM). 결과는 chunk 무관(메모리/연산만 조절). GPU 실측(128격자, fwd+bwd, peak/ms): **K80** c8 35GB·132ms / c4 25GB·117 / **c2 9.6GB·89** / c1 1.3GB·131. **K160** c2 17.7GB·158 / c1 2.6GB·257. → **속도 sweet spot=2(모든 K 最速, c1보다 ~1.6배 빠름), 메모리 최소=1.** `shape_guide_128.py`에 **`query_multi_gaussian_pair_chunk=2`** 채택(속도 우선). 객체 多 프레임서 OOM 시 1로 폴백.
- 참고: `efficientocf_config.py` 기본값은 `(128,128,10)`/`pair_chunk=8` 유지(이 config가 override). ⚠️ 진행 중 학습엔 미반영 — 다음 실행부터.

## 2026-06-24 KST — 가우시안 footprint outline 임계값을 eval_occ_threshold로 통일

- **문제**: 2D query_debug_vis의 가우시안 footprint **outline 등고선**(prob 모드)이 별도 키 `debug_query_gaussian_prob_threshold`(0.5)를 탔음 → occ metric 임계값(`eval_occ_threshold`)과 따로 놀아 env override(`EOCF_EVAL_OCC_THR`)도 안 먹음.
- **변경**: outline + 타이틀 텍스트를 렌더러 `prob_threshold` 파라미터로 통일. 이 파라미터는 학습=`eval_occ_threshold`([efficientocf.py:3113](.)) / eval=`EOCF_EVAL_OCC_THR` override([efficientocf.py:1339](.)). → occ 임계값 한 키가 metric·occ-grid·3D mixture3d·**가우시안 footprint outline**까지 모두 통제.
  - `query_head.py`: `_maybe_save_prob_grid_vis` 3330(outline 소스)·3760/3764(타이틀) → `prob_threshold` 사용.
- **dead 키 제거** `debug_query_gaussian_prob_threshold`: `query_head.__init__`(param+검증), `efficientocf.py` QueryHead 생성 kwarg, `efficientocf_config.py`(DEFAULTS+주입+주석) 전부 삭제. 잔존 참조 0, py_compile 통과. (`debug_query_gaussian_prob_alpha_scale`=heatmap 밝기는 별개라 유지.)
- ⚠️ 진행 중 학습은 이미 로드된 코드라 미반영 — 다음 실행(train/eval)부터 적용.

## 2026-06-24 KST — PROJECT_STRUCTURE.md 현행화

- 실제 working tree에 맞춰 `PROJECT_STRUCTURE.md` 재정리.
- 루트: 삭제된 항목 제거(`GPU_MULTIJOB_NOTES.md`, `MPS_MULTIJOB_RUNBOOK.md`, `run_eval_occ_all.sh`, `run_lss_pretrain.sh`, `stop_mps.sh`, `gt_gaussian/`). 추가 항목 반영(`CLAUDE.md`, `GUIDE.pdf`, `train_total.sh`). `eval_total.sh` 설명 갱신.
- `projects/configs/baselines/`: `EfficientOCF_V1.1_*` 3종 → 현재의 `shape_guide.py`(주력), `test.py`로 교체.

## 2026-06-24 KST — eval env 정리(EOCF_EVAL_MODE 0/1 숫자화 + EOCF_EVAL_ALIGN 기본값화)

- 대상: `projects/occ_plugin/occupancy/detectors/efficientocf.py`, `tools/test.py`, `eval_total.sh`, `train_total.sh`.
- **`EOCF_EVAL_MODE` 숫자 인코딩**: `0=present`(현재 1프레임) / `1=future`(미래 n_future프레임). 기본 future.
  - `_eval_mode_is_present()` static helper 추가(`"0"|"present"` → present). 읽던 두 곳(simple_test의 `eval_mode`, `_eval_mode_slice`)을 helper로 통일. 기존 문자열 `present/future`도 하위호환 수용.
- **`EOCF_EVAL_ALIGN` 기본값화**: BEV/3D 기하 정렬 `(transpose,flipH,flipW)`는 데이터 그리드 동일 시 고정값 → 코드 기본값 `(2,1,0,0)`으로 내장(frame `2`는 BEV 정렬에 미사용, 호환용 자리). sh에서 export 제거.
  - env override 유지: `EOCF_EVAL_ALIGN="t,tr,fh,fw"` 강제 / `EOCF_EVAL_ALIGN=auto` 자동 캘리브레이션(IoU 최대화). 기존 forced/auto 2갈래를 default 포함 한 블록으로 정리.
- **eval viz 저장 경로**: `tools/test.py` 자동 dir을 `./work_dirs/eval/<config>/<timestamp>` → `./work_dirs/<config>/eval/<timestamp>`로 (학습 work_dir `./work_dirs/<config>/` 아래 `eval/` 하위로 통합).
- **sh 정리**: `eval_total.sh` CONFIG → `shape_guide.py`, 주석 간결화, `EOCF_EVAL_ALIGN` 줄 제거. `train_total.sh` CONFIG → `shape_guide.py`.

## 2026-06-24 KST — shape_guide.py config 정리(미사용 vis/dead 인자 삭제 + 임계값 2개 명시)

- 대상: `projects/configs/baselines/shape_guide.py` (= 옛 `test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide_real.py` 리네임, _codeonly repo).
- **임계값 2개 명시 추가** (이전엔 미설정 → 기본 0.5로 암묵 동작했음):
  - occ 점유: `model_cfg`에 `eval_occ_threshold=0.5` (env override `EOCF_EVAL_OCC_THR`).
  - fg query score: `visualization_cfg`에 `debug_query_score_threshold=0.5` (env override `EOCF_EVAL_FG_THR`).
  - 둘 다 "config=학습·추론 공통, env=추론만 override" 주석으로 위치 명확화. 머지 경로 검증(apply_model_cfg/apply_visualization_cfg)으로 self.* 정상 주입 확인.
- **삭제 — 안 쓰는 vis 타입(3종=2D query_debug/3D mixture3d/cam_gaussian 외)**: debug_cfg에서 `debug_query_inst_depth_lift_vis_every`, `debug_query_attn_softargmax_vis_every`; visualization_cfg에서 instance_img(dir/max_frames), gt_alignment(dir), inst_depth_lift(dir/max_frames/max_instances), attn_softargmax(dir), `query_attn_vis_dir`. (모두 every 기본값 0 → 키 제거시 자동 off.)
- **삭제 — deprecated/dead**: `debug_query_mixture3d_vis_score_threshold`(deprecated, score는 debug_query_score_threshold로 통일. 코드 실제 read 0건=utils_visualization.py:1335 주석에서만 언급).
- **(정정)** `debug_query_score_iou/cls/cam_attn_weight`는 처음에 dead로 오판해 지웠으나 **실제 live** — `utils_visualization.py:1638-1640`이 fg score 합성식 `score=(w_iou*iou+w_cls*cls+w_cam*cam)/norm`에 사용. 원래 cls-only(0.0/1.0/0.0)인데 삭제 시 DEFAULTS(0.5/0.5/0.0)로 떨어져 score 정의가 바뀜 → **복원**(cls-only 유지). 오판 원인: 이전 grep의 `-v "= float"` 필터가 `w_iou = float(self...)` 사용처를 자체 제거. (멀티에이전트 통일검증 워크플로가 적발.)
- model_cfg(실제 모델/loss)는 무수정. 옛 repo는 호환 위해 그대로 둠(_codeonly만 정리). py 로드/머지 검증 통과(occ=eval_occ_threshold, fg=debug_query_score_threshold, score합성=cls-only).

## 2026-06-24 KST — 중복 임계값 제거(confidence/objectness) → occ+score 2개로 정리

- **불필요 임계값 2개 제거**: `debug_query_confidence_vis_threshold`, `debug_query_objectness_vis_threshold`는 score 임계값과 중복(objectness는 순수 alias, confidence는 센터 마커 색칠 게이트로만 쓰임). 둘 다 삭제하고 동작을 `debug_query_score_threshold`(+`EOCF_EVAL_FG_THR`)로 일원화.
  - `query_head.py`: `__init__`에서 두 파라미터 제거 + alias/검증 블록 제거. 센터 마커 `conf_thr`(:3311)을 `EOCF_EVAL_FG_THR or debug_query_score_threshold`로 라우팅. bundle score_thr 기본값도 `debug_query_score_threshold`.
  - `efficientocf.py`: QueryHead 생성 시 두 인자 전달 제거.
  - `efficientocf_config.py`: DEFAULTS 키 2개 + 주입 라인 2개 삭제.
- **결과**: config의 의미적 임계값은 **occ(`eval_occ_threshold`) + query score(`debug_query_score_threshold`)** 둘뿐. 각각 env override(`EOCF_EVAL_OCC_THR`/`EOCF_EVAL_FG_THR`)로 추론만 덮어쓰기. (footprint 색칠 농도용 `debug_query_gaussian_prob_threshold`는 별개 렌더 디테일이라 유지.) 코드 잔존 참조 0, config 설정 0, py_compile 통과.

## 2026-06-24 KST — query score 임계값 통일(2D선택=2D표시=3D필터) + config 가시성 정리

- **score 임계값 통일**: 2D query 선택은 이미 `debug_query_score_threshold`(config)+`EOCF_EVAL_FG_THR`(env override)로 학습·추론 공통이었으나, **3D mixture3d 필터만 별도 `debug_query_mixture3d_vis_score_threshold`(0.75)** 였음. `utils_visualization.py:1337`을 동일 소스(`EOCF_EVAL_FG_THR or debug_query_score_threshold`)로 변경 → **2D선택/2D표시/3D필터가 단일 임계값**. 옛 3D score 키 deprecated(무시).
- **config 가시성**(`efficientocf_config.py` VISUALIZATION_CFG_DEFAULTS): 임계값 한눈에 보이게 주석 블록 추가 — occ는 `eval_occ_threshold`(+EOCF_EVAL_OCC_THR), query score는 `debug_query_score_threshold`(+EOCF_EVAL_FG_THR), 둘 다 "config=학습·추론 공통, env=추론 override" 패턴 명시.
- 결과: occ 임계값과 score 임계값 둘 다 동일 패턴(config 1개가 학습·추론 일관, env로 추론만 override). 별도/이상 임계값 없음. py_compile 통과.

## 2026-06-24 KST — mixture3d 2D방향 일치(측정검증) + 센터 검정/작게 + future all행 전체fg쿼리

- **2D↔3D 방향 불일치 측정으로 규명·수정**(`utils_visualization.py`): 워크플로 분석 + `proj3d.proj_transform` 직접 측정. 결과 **원래 azim(-60/-90)이 worldX 좌→우로 2D와 일치**했고 직전 +180(120/90)이 그걸 뒤집었음(되돌림). 추가로 **y축 반전**(`set_ylim(ymax,ymin)`)하니 X(좌→우)·Y(위→아래)·Z(위) 셋 다 2D BEV와 일치. occ Z는 z_idx=0=지면(-4.9m)으로 **뒤집힘 없음 확인**(차 바퀴 땅에 정상) — "바퀴 하늘"은 잘못된 azim 뷰각 탓이었음.
- 센터 마커: magenta+흰테두리 → **검정·테두리없음·s45**(요청). GT=lime 얇은 검은테두리. 컴포넌트 점 옅게(alpha 0.5).
- 겹침: bottom GT 진하게(0.45)/예측 옅게(0.18)로 둘 다 보이게.
- **future 2D "all"행 = 전체 fg 쿼리**(`efficientocf.py` `_build_eval_future_2d_bundle`에 conf/cls 전달, candidate=background 제외 전체) → present처럼 임계값 미달 저신뢰 쿼리까지 표시.
- 검증(noli occ epoch_13, 풀해상도, 8GPU 병렬): present/future 2D·3D 정상. 결과 `work_dirs/sample_test_vis_check/V6/{present,future}/`. ⚠️ 3D 센터 가시성은 matplotlib 깊이정렬 한계로 dense 영역에서 일부 묻힐 수 있음(필요시 2D 투영 오버레이로 강제). py_compile 통과.

## 2026-06-24 KST — ★future 미표시 진짜 버그(feat_out 인덱스 off-by-one) 수정 + mixture3d mode별 프레임/센터/투명도

**핵심 버그 수정**: future 2D/3D에 예측이 안 나오던 진짜 원인은 체크포인트가 아니라 **feat_out 인덱스 off-by-one**. future mixture를 `feat_out[30~33]`로 잡았는데 `[30]`은 centers가 아니라 **sigmas**(반환 순서: [29]centers/[30]sigmas/[31]yaw/[32]weights). 그래서 sigma(값 0.7~2.0)를 mixture center로 써서 좌표가 한 점에 뭉쳐 footprint가 전혀 안 그려졌음. 진단: `EOCF_VIS_DBG=1`로 mix_c 좌표가 x[0.7,2.0]임을 확인→`[29~32]`로 수정하니 x[-42.5,43.5](world)로 정상화. **결과 future 2D footprint 100px→8923/43870px**(present처럼 footprint, occ-overlay 불필요).
- `efficientocf.py`: mix_future 인덱스 [30~33]→**[29~32]**. occ-overlay 자동ON 제거(이제 footprint 동작). mixture3d도 **mode별 분기** — present=present bundle+full gt(present 프레임), future=future bundle+미래 tail gt+**frame_idx=1(2번째 미래 프레임)**. future bundle(인덱스 수정으로 valid)을 mixture3d가 받아 렌더(이전 0장→4장).
- `utils_visualization.py`: mixture3d에 `frame_idx` 파라미터(표시 프레임 override). 센터 점 s50→**s35**(작게), 예측=magenta·GT=lime, **흰 테두리+zorder=10**(구름 위 대비). 겹침/센터 가시성 위해 구름 더 투명(top GT 0.25, bottom GT 0.28, gmm 0.30).
- **검증(noli occ epoch_13, 풀해상도)**: present 2D 8884/8490·3D present프레임 / future 2D 8923/43870·3D 2번째미래프레임. 둘 다 footprint 풍부. 결과: `work_dirs/sample_test_vis_check/OCC_FINAL/{present,future}/`. (참고: ksh test_traj_mcls_full은 multi-class라 "class", noli가 "occ".) py_compile 통과.

## 2026-06-24 KST — future 2D 예측 표시(occ-overlay)·근본원인 규명 + epoch_15 검증 + 센터/투명도 디자인

- **future 2D 빈 예측의 진짜 원인 규명**: 체크포인트가 아니라 코드. epoch_15(완전학습 test_traj_mcls_full)로도 future **footprint는 100px(빔)** — trajectory-propagated mixture는 footprint 투영이 안 됨(present footprint는 8513px 정상). `_build_eval_future_2d_bundle`에 `score_thr=0.0`+표시키 추가해도 footprint는 안 떴음.
- **해결**: future는 예측 occ(dense)를 **blend 오버레이(행별 lo_color/hi_color 반투명)**로 표시. `_maybe_save_eval_query_vis`에서 future 시 `EOCF_VIS_PRED_STYLE=blend` 자동(미설정 시). 결과 future 2D 11733px. present는 footprint 그대로(8513px).
- **epoch_15 검증 경로 확보**: ksh의 `test_traj_mcls_full.py`+`epoch_15_lss_only.pth`가 이 repo와 모듈/dim/num_gaussians 호환 → `full15.py`(복사+test_capacity 4+3D score 0.05)로 가져와 정상 동작(buffer 키 mismatch만, 무해). noli latest는 epoch_13인데도 forecasting 미학습이라 future occ 약함.
- **mixture3d 디자인**(`utils_visualization.py`): 센터 점 s30→**s50**, 예측=**magenta**·GT=lime 고정색(클래스색은 구름과 안 보임), 검은 테두리. 겹침 가시성 위해 구름 **양쪽 반투명**(top GT 0.55→0.40, bottom GT 0.32→0.42, 예측 gmm 0.20→0.42).
- 결과: `work_dirs/sample_test_vis_check/EPOCH15/{present,future}/`. py_compile 통과.

## 2026-06-24 KST — GUIDE-real Phase A: union+opacity제거(weight_mode='ones') + 48가우시안 + focal-only (test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide_real)

**핵심 내용**: GUIDE 논문(eq3 `p=1-Π(1-G)`, weight 없음, peak=1)에 충실하게 occupancy 표현을 정렬. 기존 `guide`(union+sigmoid α opacity)에서 **학습 opacity α를 제거** → weight collapse/몰빵 + (sigma↔weight) degeneracy가 동시에 소멸하고, occ focal이 sigma를 직접 통제. 회전(yaw→full quaternion)은 8파일/138참조 교차변경이라 Phase B로 분리.

**변경 — query_head.py**: `weight_mode`에 `'ones'` 추가 (검증 튜플 611 + 분기). `weights_qg = ones_like(logits)` (weight head 출력 무시), surrogate는 uniform(sigma 2차모멘트용). softplus_bias_init 분기(696)엔 미포함(불필요).

**변경 — config (`..._guide_real.py` model_cfg)**:
- `query_multi_gaussian_weight_mode`: 'sigmoid' → **'ones'** (combine_mode='union' 유지 → `1-Π(1-G)`)
- `query_num_gaussians`: 16 → **48** (GUIDE 최고 성능; 실측 오버헤드 추론 +0%/+0MB, 학습 +1.1%/+0.6%)
- `use_query_gmo_dice_loss`: (기본 True) → **False** (GUIDE처럼 occ focal only)
- `query_multi_gaussian_sigma_max_m`: (2,2,0.7) → **(3,3,1.5)** (α 제거로 degeneracy 사라져 loss가 sigma 통제 → 난간만 느슨하게)
- `query_multi_gaussian_sigma_min_m`: (0.15…) → **(0.2…)** (voxel 0.2m 바닥)
- `query_multi_gaussian_sigma_reg_loss_weight`: 0.001 → **0.0** (α 없으니 불필요)
- offset_max (12,12,2)·weight_reg 0·eval_use_mixture True는 유지

**검증**: query_head/config py_compile 통과, `Config.fromfile` 파라미터 반영 확인, **`build_model` 성공**(weight_mode='ones'+48 구성).

**주의/다음(Phase B)**: yaw→full quaternion 회전은 `mixture_yaw_tkg`[T,K,G]가 head→voxelizer→loss→matcher→viz→geometry 8파일에 스칼라로 박혀 있어 쿼터니언[T,K,G,4]로 일관 교체 필요(렌더 hot-loop Mahalanobis도 full R 적용). 별도 careful 단계로 진행. binary fg(car~trailer 공유)라 학습 후 **car IoU/precision + over-coverage(α 없어 FP 코어) 모니터** 필수.

## 2026-06-24 KST — mixture3d 2뷰화/센터 작은점 + 풀해상도 검증(저해상도가 예측 붕괴 주범)

- **저해상도가 예측을 붕괴**시킴 규명: 320~448px에선 LSS feature가 작아져 query 예측이 거의 0(2D hi행 25px, 3D 0px). **풀해상도(896×1600)에선 epoch_2여도 present 예측 풍부**(2D hi 8490px, 3D 24963px). 그동안 "예측 없음"은 대부분 저해상도 아티팩트였음.
- **mixture3d 디자인**(`utils_visualization.py`): 뷰 3→**2개**(각도만 다른 중복 제거, `add_subplot(2,n_views,…)`, figsize `9*n_views×18`). 센터 마커 큰 원(s=160) → **작은 점(s=30)**, 모든 뷰 동일, GT=lime/예측=클래스색·검은 테두리. 1열/2열 차등·점수텍스트는 이전에 제거됨.
- **future 3D=present shape 유지**(`efficientocf.py`): mode-일관성으로 future bundle을 mixture3d에 넘기니 조기종료(빈 이미지) → 되돌림. mixture3d는 항상 present-frame **객체 SHAPE** viz, 2D query_debug_vis가 시간축(future occupancy) 담당. (즉 future 3D는 present shape를 보이고, future 2D는 미래 occ — epoch_2 forecasting 미학습이라 빔.)
- **검증(풀해상도)**: present/future 모두 2D·3D 정상, 3D 2뷰 1693×1903. 결과: `work_dirs/sample_test_vis_check/FINAL/{present,future}/`. 임시 config: `sample_test_fullres.py`(896×1600, test_capacity=4, 3D score 0.05). py_compile 통과.

## 2026-06-24 KST — 2D 예측 occ 오버레이 자동ON 해제(기존 footprint로 복귀) + future 빈예측 원인규명

사용자 피드백("future가 이상하게 찍히고 예측 없음, present처럼 나와야"). 진단: occ 오버레이(통짜 채움)가 present의 footprint와 달라 "이상"하게 보였고, future 예측이 비는 건 **렌더 문제가 아니라 epoch_2 체크포인트가 forecasting 미학습**이라 미래 mixture가 거의 비어서임(traj mixture는 efficientocf.py:1175-1186에서 present로 fallback하므로 데이터 자체는 유효).
- `efficientocf.py`: `_maybe_save_eval_query_vis`의 future 자동 `EOCF_VIS_PRED_STYLE=blend` 설정 **제거** → 예측은 기존 footprint(행별 lo/hi 색)로 그림. occ 오버레이는 `EOCF_VIS_PRED_STYLE` 명시 시에만(기본 OFF). future 2D bundle/overlay 코드는 유지(좋은 체크포인트에서 동작).
- **검증(저해상도, epoch_2)**: present 2D 2행 예측 footprint **6143px**(정상) vs future **100px**(빈≈모델 미예측). 둘 다 새 mixture3d(true-z, 열별 센터, 단일색 GT) 동일. 결과: `work_dirs/sample_test_vis_check/v3_clean/{present,future}/`. py_compile 통과.
- ⚠️ **future 예측을 보려면 더 학습된 체크포인트 필요**(epoch_2는 present만 예측됨). 디자인 미리보기는 present 모드 권장.

## 2026-06-24 KST — 2D 예측 오버레이 기존색화 + mixture3d 센터 재설계(열별 차등)

사용자 피드백 반영.
1. **query_debug_vis 예측 오버레이를 '기존처럼'**(`query_head.py`): 단일색+흰겹침 실험 → 행별 원래 색(all=lo_color red, hi=hi_color cyan)으로 반투명 blend. `EOCF_VIS_PRED_COLOR` 지정 시만 override. future 모드는 `_maybe_save_eval_query_vis`에서 `EOCF_VIS_PRED_STYLE=blend` 자동 설정(미설정 시) → 미래 예측 자동 표시.
2. **mixture3d 센터 재설계**(`utils_visualization.py`): 점수 텍스트 제거. 예측 센터를 흰→**검은 테두리(lw1.8)**로 진하게. 뷰(열)별 차등 — 1열=center 보기용 크게(예측 s160 + GT 센터 lime s150 복귀), 2열=중간(s85), 3열=작게(s55). GT 센터는 1열에만.

**검증(저해상도 320×576, test_capacity=4)**: Run A(기본 blend) 2D — 2행 all=red 4967·3행 hi=cyan 4999(기존색 OK). Run B(outline) 정상. 둘 다 3D 4장. OOM/에러 0. 결과: `work_dirs/sample_test_vis_check/v2_design/{run_A_blend_origcolor, run_B_outline}/`. py_compile 통과. ※ GPU 포화로 320×576 저해상도 검증(품질↓, 코드동작 확인용).

## 2026-06-24 KST — mixture3d true-z(z 과장 제거) + 2D 예측 occ 오버레이(디자인 변형)

future 2D에서 예측이 여전히 안 보이던 문제 재진단: gaussian footprint 투영이 미래(trajectory) mixture로는 거의 안 그려짐(예측 행 px ~100). 견고하게 해결.
1. **mixture3d z 원래 비율**(`utils_visualization.py`): 3D 3뷰 중 왼쪽 2뷰의 z 과장(z_aspect 0.22≈2.8x)을 제거하고 세 뷰 모두 true-z(0.078)로.
2. **2D 예측 occ 오버레이**(`query_head.py:_maybe_save_prob_grid_vis`): footprint 대신 예측 occ(dense, `pred ≥ prob_threshold`)를 BEV로 collapse해 행에 직접 그림. `EOCF_VIS_PRED_STYLE` 설정 시에만 동작(미설정=기존 유지):
   - `fill`/`overlap`: pred-only=지정색, GT∩pred 겹침=흰색(둘 다 보임)
   - `outline`: 예측 경계선만(GT 채움 그대로 비침)
   - `blend`: 예측 반투명 덧칠
   - 색은 `EOCF_VIS_PRED_COLOR`(red/magenta/yellow/cyan/orange).
   `gt_np` 옆 closure로 구현, all/hi/hi_cls 행에 적용.

**검증(저해상도)**: `sample_test_lowres.py`에 `test_capacity=4` 추가(소량 자연종료). 변형 생성: overlap+red(2행 예측 빨강1639+겹침흰8624 OK), blend+cyan(2행 cyan 4892 OK). outline+magenta는 GPU 경합 OOM으로 보류. 결과: `work_dirs/sample_test_vis_check/design_variants/`. py_compile 통과.

## 2026-06-24 KST — mixture3d 센터(예측만 점) + future 2D 예측 표시 복원

1. **mixture3d 센터 표시 변경**(`utils_visualization.py`, 학습·추론 공통): GT 센터 마커(lime 점) 제거(상·하단), GT 클래스 텍스트 라벨만 유지. 예측 query 센터는 점(o)으로 유지(흰 테두리·depthshade off).
2. **future 2D viz에 예측이 안 보이던 문제 수정**(`efficientocf.py`): 2D 렌더러는 예측을 gaussian footprint(bundle)로만 그리는데, `_build_query_visualization_bundle`이 `score_frame_count=min(time_receptive_field,…)`로 receptive(≤3)프레임만 반환 → 미래4와 프레임 불일치라 앞 단계에서 bundle을 비웠더니 예측이 전부 사라졌었음. 해결: `_build_eval_future_2d_bundle` 신설 — present bundle의 선택된 query를 미래 centers + 미래 mixture(`feat_out[30~33]` trajectory mixture)로 `[F,Qsel,…]` re-index해 직접 구성. selected/candidate 둘 다 채워 all/hi 행에 예측 footprint 표시. mixture3d/cam은 present full bundle 유지. 텐서 없으면 meta-only로 graceful fallback.

`py_compile` 통과 + future-bundle shape 단독검증.

**라이브 검증(저해상도)**: 풀해상도 eval은 ~23GB 필요한데 GPU가 전부 타 학습잡으로 포화(여유 15~19GB)라 OOM. 검증용으로 `sample_test_lowres.py`(noli 복사 + `input_size` 896×1600→448×800, LSS 메모리 ~1/4) 생성해 GPU2에서 future 모드 실행 성공(OOM 없음, frustum size mismatch는 무해한 load 경고—buffer라 모델 자체 저해상도 frustum 사용). 결과: 2D=2060폭(미래4프레임) 정상, **예측 footprint 픽셀 16,267개로 표시 확인**(수정 전 ~0). metric IoU2d=0.059/IoU3d=0.036. mixture3d/cam도 생성. 이미지: `work_dirs/sample_test_vis_check/lowres_future/`. ※ 저해상도라 예측 품질은 풀해상도보다 낮음(코드 동작 검증용). 풀해상도는 GPU 여유 시 `sample_test.py`로 재실행.

## 2026-06-23 KST — mixture3d 센터 가시성 + 추론 2D 하위폴더/미래4프레임 + EOCF_EVAL_MODE(present/future)

사용자 요청 4건 반영(`efficientocf.py`, `utils_visualization.py`).
1. **mixture3d 센터 가시성**(학습·추론 공통 렌더러): 가우시안에 가려 안 보이던 센터 수정. GMM occ alpha 0.28→0.20, 센터를 별(`*`,s150)→**점(`o`)**, 적당한 크기(GT lime s55·예측 클래스색 s60), `depthshade=False`로 깊이 흐림 제거, 구름 위에 마지막으로 그려 안 가리게. 예측 센터는 흰 테두리로 대비.
2. **추론 2D를 `query_debug_vis/` 하위폴더로**: 기존엔 `sample_*.png`가 EOCF_EVAL_VIS_DIR 루트에 그냥 떨어졌음 → 학습과 동일하게 하위폴더 저장(`_maybe_save_eval_query_vis`의 `head.debug_vis_dir`).
3. **추론 2D를 미래4프레임 표시**: 기존엔 present 1프레임만. future 모드면 미래 n_future 프레임(centers_eval/pred_occ_eval/gt future-tail, max_frames=n_future)으로 표시. 2D는 present-기반 bundle(receptive 3프레임)과 프레임수 불일치 → per-query gaussian footprint 행은 생략하고 GT occ vs 예측 occ(pred_occ_future)+센터점으로 미래4 비교(mixture3d/cam은 full bundle 유지).
4. **`EOCF_EVAL_MODE`(present/future, 기본 future) 신설**: metric과 viz가 함께 따름. `_eval_mode_slice` 헬퍼 추가(present=현재프레임 idx `T-n_future-1`, future=미래 tail), metric의 `_eval_future_tail` 4곳 교체, present 모드는 pred_txyz를 현재 1프레임으로 override. future 코드경로는 불변(회귀 없음).

**검증**(noli ckpt, val, 1GPU, OCC_THR=0.5): future → 2D 2060폭(4프레임)·metric IoU2d=0.036/IoU3d=0.013, query_debug_vis 하위폴더 확인·루트 누출 0, mixture3d/cam 정상. present → 2D 512폭(1프레임)·metric IoU2d=0.080/IoU3d=0.027. 크래시 0. `eval_total.sh`에 `EOCF_EVAL_MODE` 추가. 결과 이미지 `work_dirs/sample_test_vis_check/infer_{future,present}/`.

## 2026-06-23 KST — 추론 시각화 학습 정렬 (2D/3D/cam_gaussian 추가) + sample_test.py + 스모크 검증

추론(dist_test) 시각화를 학습과 정렬. 4종 중 **3종 추가 완료·검증**, 1종(attn_softargmax)은 매칭 의존으로 보류.
- **원칙**: 스타일/threshold 파라미터는 전부 config(`self.debug_query_*`, `eval_occ_threshold`)에서 읽어 학습=추론 동일. 추론 전용 override는 출력 dir + every(=1, 실제 샘플링은 `EOCF_EVAL_VIS_EVERY`가 담당)뿐.
- `efficientocf.py` 추론 경로(`simple_test`/`_maybe_save_eval_query_vis`): `extract_feat_query`를 `EOCF_EVAL_VIS=1`일 때 `return_instance_img_debug/_fullseq/return_query_cam_gaussian_vis_debug=True`로 호출 → `feat_out[14]`(attn weight)/`[16]`(instance_img bundle)/`[17]`(match_inputs) 캡처. `_maybe_save_eval_query_vis`에 인자 추가, **query_attn_softargmax_vis + query_cam_gaussian_vis 호출 추가**(mixture3d는 기존). `_eval_vis_all_ranks` 호이스트.
- `utils_visualization.py`/`utils_instance_img_debug.py`: attn·cam_gaussian 렌더러의 `_is_main_process` 게이트가 `_eval_vis_all_ranks` 존중(eval은 rank별 다른 샘플 → 전 rank 저장).
- **신규 config**: `projects/configs/baselines/sample_test.py` (noli config `_base_` 상속). 시각화 정렬 테스트용.
- **스모크 검증**: noli ckpt(epoch_2)+nuScenes val, 1GPU, `EOCF_EVAL_VIS=1 EOCF_EVAL_VIS_EVERY=1 EOCF_EVAL_OCC_THR=0.2`. 결과 — query_debug_vis(2D)✅ / query_mixture3d_vis(3D)✅ / query_cam_gaussian_vis✅ PNG 정상 생성, 에러 0. EOCF_EVAL_OCC_THR=0.2가 metric+2D+3D에 일관 적용됨 확인. attn 코드 제거 후 재검증 → 3종 정상·attn dir 미생성(clean) 확인.
- **query_attn_softargmax_vis 제외(사용자 결정)**: 이 viz는 GT 매칭 결과(inst_match_result) 필수인데 추론은 매칭을 안 함(`{}`). 추론에서 `_match_queries_to_gt_instances`를 돌리려면 GT 인스턴스 center/ids/cls prep + bev feature가 필요(별도 큰 작업)해서 **3종으로 확정**. 임시로 넣었던 attn 호출/인자/feat capture/rank-bypass는 전부 되돌려 추론 경로 clean(잔재 0).

## 2026-06-23 KST — 불필요 문서 정리 (stale 스냅샷 + 이식 끝난 포팅 가이드 삭제)

루트의 잉여 md 파일 5종 삭제로 문서 정리. 삭제 대상: `4_CHANGELOG.md`·`4_NOTES.md`(옛 `Autonomous_Driving_26_ksh/` repo 스냅샷, 05-15에서 멈춘 stale 복사본 — 현행 `CHANGELOG.md`/`NOTES.md`로 대체됨), `QUERY_ATTN_CHANGES.md`·`BINARY_FG_CLS_CHANGES.md`·`GATE_RECRU_GUIDE.md`(타 repo 이식용 일회성 가이드, 이식 완료). 추적 파일 3종은 `git rm`, untracked 2종은 `rm`. CHANGELOG/NOTES 본문의 과거 언급(이식 기록)은 사적 history라 그대로 보존. sh 스크립트는 `run.sh`/`run_eval.sh`가 `train_total.sh`·eval 래퍼의 의존성이고 `train_total_2~4.sh`는 동시학습 잡 변형이라 전부 유지.

## 2026-06-23 KST — query_mixture3d_vis 가독성 개선 (GT↑/예측↓/중심마커 위로)

`maybe_save_query_mixture_3d_vis` 렌더링 대비 재조정(학습·추론 viz 공통, `utils_visualization.py`). 하단(예측 GMM occ vs GT) 행에서 GT 회색 구름이 `alpha=0.12`로 너무 옅고 예측이 `alpha=0.5` 풀 클래스색으로 너무 세서 비교가 안 되던 문제.
- **GT 구름 진하게**: 하단 `c=0.6,s=0.6,alpha=0.12` → `c=0.45,s=1.2,alpha=0.32`. 상단(1σ) GT occ도 `s=1.0,alpha=0.35` → `s=1.6,alpha=0.55`.
- **예측 옅게**: 하단 GMM occ `s=2.0,alpha=0.5` → `s=1.6,alpha=0.28` (GT가 비쳐 보이게).
- **중심 마커 위로/크게**: 그리는 순서를 구름→중심으로 바꿔(예측 구름이 중심 dot을 덮던 문제) 중심이 맨 위. GT 중심 lime X `s=60→90, lw 2.0→2.8`(상·하단), 예측 query 중심은 하단에서 클래스색 **별(`*`, s=150, 검은 테두리)**로 강조.

py_compile 통과. 시각화 전용이라 학습/평가 수치 영향 없음.

## 2026-06-23 KST — `eval_total.sh` 추가 (단일 eval 원클릭 래퍼)

`train_total.sh`처럼 값만 수정하고 `bash eval_total.sh`만 실행하면 되는 평가 래퍼 신설. 상단에 `CONFIG`/`CHECKPOINT`/`GPUS`/`PORT` + `EOCF_EVAL_*`(ALIGN/OCC_THR/FG_THR/VIS/VIS_EVERY)를 export로 정의 후 `tools/dist_test.sh` 호출. `EOCF_EVAL_OCC_THR`은 이번 단일화로 metric+2D+3D viz에 동시 적용됨을 주석화. ⚠️ 기본 `CONFIG`(`test_traj_mcls_full_evalmix.py`)·`CHECKPOINT`는 `_ksh` repo 경로라 이 repo엔 없음(주신 값 그대로 박아둠, 사용 전 확인 필요). `bash -n` 통과, 실행권한 부여. PROJECT_STRUCTURE.md 갱신.

## 2026-06-23 18:23:42 KST — occupancy threshold 전역 단일화 (`eval_occ_threshold` 신설) + 추론 mixture3d 시각화 추가

**배경(audit)**: shape 관련 인자가 학습/추론 × 실제/시각화 4개 맥락에 흩어져 있는지 전역 조사. **shape 생성·합성**(query_head mixture + voxelizer `occ_combine_mode`/`weight_mode`/sigma·offset bound + `query_eval_occ_use_mixture` 기본 True)은 train/eval 일관 → 문제 없음. **occupancy threshold**가 0.5 근처 4개 노브로 출처가 제각각이라 꼬여 있었음: ①평가 metric=`EOCF_EVAL_OCC_THR`(env), ②2D viz occ-grid=학습 **하드코딩 0.5**/추론 env, ③가우시안 footprint 색칠=`debug_query_gaussian_prob_threshold`(config), ④3D mixture occ=`debug_query_mixture3d_vis_occ_threshold`(config, base 0.2/guide·noreg 0.5, **추론 미생성**).

**변경 — ①②④를 단일 키로 통일**:
- `efficientocf_config.py`: `eval_occ_threshold`(기본 0.5) **신설**(MODEL_CFG_DEFAULTS:129, 주입 371). occupancy 점유 판정의 단일 출처. 런타임 override는 `EOCF_EVAL_OCC_THR`(eval 한정) 유지.
- `efficientocf.py`: ①metric `thr = env or self.eval_occ_threshold`(1428). ②학습 viz occ-grid `prob_threshold=0.5`(하드코딩) → `self.eval_occ_threshold`(2943). 추론 viz occ-grid는 기존대로 `thr` 사용(이미 ①과 동일).
- `debug_query_mixture3d_vis_occ_threshold` **제거**(deprecated): DEFAULTS·주입(efficientocf_config.py)·렌더러 참조(utils_visualization.py:1364) 삭제. mixture3d occ_thr은 이제 `eval_occ_threshold` 참조(eval은 env-resolved 값을 `occ_threshold` 인자로 직접 받음). config 7개(`bg`,`head`,`head_gau`,`head_noli`,`noli`,`noli_guide`,`noli_noreg`)의 해당 라인·주석 삭제.

**변경 — ④ 추론 mixture3d 추가 (parity)**:
- `efficientocf.py:_maybe_save_eval_query_vis`: 2D viz 직후 `maybe_save_query_mixture_3d_vis` 호출 추가. `<EOCF_EVAL_VIS_DIR>/query_mixture3d_vis/`에 저장, occ_thr=metric과 동일(`prob_threshold`). 이제 metric 계산 시점(추론)에 3D shape 확인 가능.
- `utils_visualization.py:maybe_save_query_mixture_3d_vis`: `occ_threshold` 인자 추가, rank 게이트가 `_eval_vis_all_ranks` 존중(eval은 rank별 다른 샘플 처리 → 전 rank 저장, 2D와 동일 정책).

**미변경(의도)**: ③ footprint 색칠(`debug_query_gaussian_prob_threshold`)은 개별 G density 기준이라 의미가 달라 별도 노브 유지. union↔sigmoid / poisson↔softplus 쌍 guard는 이번 범위 외(NOTES에 후속 기록).

**검증**: 변경 3개 코드 + config 7개 py_compile 통과. `debug_query_mixture3d_vis_occ_threshold` 잔여 참조 0건.

## 2026-06-23 16:55:40 KST — eval occupancy mixture 정렬 전역 기본값화 (False→True) + 단일경로 dead-code 확인 (양쪽 레포)

**핵심 내용**: 추론/eval-시각화를 학습(16-gaussian mixture)과 완전히 정렬. `query_eval_occ_use_mixture` 기본값을 `False`→`True`로 변경(두 레포 `efficientocf_config.py` 동일: occ_shape 123행 / occ 117행). 이제 모든 config가 eval occupancy(present 정렬 + 미래 metric)와 eval 시각화 occupancy 패널을 단일 ellipsoid가 아닌 mixture로 splat.

**안전성**: mixture 텐서가 없는 모델은 `simple_test`의 `use_mix` 가드(텐서 유효성·dim·shape 검사)가 자동으로 단일 ellipsoid fallback → 비-mixture config도 무파손. configs/ 전체에 명시적 False override 없음(전 config True 적용). 양쪽 config.py py_compile 통과.

**audit 결과 — 학습 시각화는 변경 불필요**: `extract_feat_query`의 present 단일 voxelize(`self.voxelizer(...)`, occ 1203 / occ_shape 1204)는 두 호출부(occ 1494/2301, occ_shape 1369/2170)가 모두 `voxelize=False`라 **dead code**(`pred_occ`=None). 학습-시각화 occupancy 패널(`maybe_save_query_debug_vis`의 `pred_occ_prob`/`pred_occ_prob_all_obj`)은 이미 None을 받아 단일 ellipsoid를 안 그리고, shape 시각화는 mixture GMM 패널이 담당 → 학습 forward 핫패스 미변경(불필요한 per-query 루프 slowdown 위험 회피).

**관련**: `test_traj_mcls_full_evalmix.py`의 명시적 `query_eval_occ_use_mixture=True`는 이제 기본값과 동일(중복이지만 무해). base `test_traj_mcls_full.py`도 mixture로 전환되어 이전의 "평균 baseline vs 각각" 비교 구도는 사라짐(사용자가 전부 mixture로 통일 요청).

## 2026-06-23 15:43:38 KST — noli 계열 gaussian shape bound 데이터 기반 재설정 (offset 12 / sigma_max 2.0,2.0,0.7)

**핵심 내용**: noli 계열에서 16개 가우시안이 offset은 안 움직이고 sigma만 키워 **원형 blob**으로 수렴하는 문제. 원인은 sigma_max=3.5m가 과도해서 가우시안 1개로 객체 전체를 덮는 "쉬운 gradient 경로"가 열려 있었기 때문(offset 협응 학습 불필요).

**데이터 근거** (`gt_gaussian/sample_frame3_inst14_gmm16.npz`): 실제 인스턴스(extent 8.6×6.2×3.6m, 26,752 voxel)에 diagonal 16-GMM fit 시 per-gaussian sigma는 **xy median 0.90m / max 1.77m, z median 0.60m / max 0.67m**. 즉 차보다 큰 객체도 16개로 쪼개면 sigma는 ~0.9m, tail도 1.8m 미만 → 3.5m는 명백히 과대.

**변경** (4개 config: `_noli`, `_head_noli`, `_noli_guide`, `_noli_noreg`):
- `query_multi_gaussian_offset_max_m`: (10,10,2) → **(12,12,2)** — trailer(전장~19m) 커버를 sigma 아닌 offset 확장으로.
- `query_multi_gaussian_sigma_max_m`: (3.5,3.5,1.5) → **(2.0,2.0,0.7)** — xy는 fit max(1.77)에 맞춘 상한(blob path 차단), z는 fit max(0.67) 기준. sigma_max 작아지면 init sigma도 작아져 under-cover → offset gradient 활성화.
- `sigma_min`(0.15)·`sigma_reg`(0.001)·`num_gaussians`(16)은 미변경. 각 파일 [shape 실험] 주석도 데이터 근거로 갱신.

**주의/모니터**: binary fg라 car~trailer가 이 한 세트 공유. trailer가 비어 보이면 **sigma 다시 키우지 말고** num_gaussians(16→24)나 offset를 풀 것. car IoU/precision 회귀 여부 확인. `efficientocf_config.py` 전역 기본값은 미변경(이 config들이 명시적으로 override하므로 무영향).

## 2026-06-23 15:00:27 KST — eval occupancy를 학습과 정렬: mixture scene splat 신설 (noreg/guide 둘 다)

**핵심 내용**: 학습/매칭/시각화는 16개 가우시안 mixture(`forward_gaussian_mixture_grouped`)를 쓰는데, **eval(`simple_test`)만 쿼리당 16개를 moment-2로 요약한 단일 ellipsoid 1개**(`voxelizer.forward`→`_forward_gaussian`)로 찍고 있었음 → 학습 shape(특히 **yaw/회전**)가 eval에서 사라지고, 빈 공간 over-cover. (GUIDE는 학습=추론 모두 K개 union splat.) eval도 mixture를 그대로 splat하도록 정렬.

**왜 grouped를 eval에 그냥 못 쓰나**: `forward_gaussian_mixture_grouped`는 per-query 텐서 `[T,K,1,D,H,W]`를 할당 → eval 풀해상도(512×512×40)에서 K(선택 쿼리)배만큼 곱해져 수 GB~OOM. (그래서 기존 eval이 단일 경로를 썼던 것.)

**해결 — `forward_gaussian_mixture_scene` 신설** (`voxelizer.py`):
- per-query mixture를 **하나의 scene 볼륨 `[T,1,D,H,W]`에 누적**(쿼리별 저장 X → 풀해상도 메모리 안전).
- 쿼리 내부 G개: `self.gaussian_combine_mode`(poisson/union)로 합성. 쿼리 간: **max(occupancy union)**.
- yaw·offset·sigma 전부 사용(회전 살림).

**eval 경로 정렬** (`efficientocf.py` `simple_test`):
- 플래그 `query_eval_occ_use_mixture=True`이고 mixture 텐서가 유효할 때만 mixture 경로(아니면 기존 단일 ellipsoid로 안전 fallback; 프레임/쿼리 수 mismatch도 fallback).
- 헬퍼 `_eval_select_mixture_params`: present mixture에서 **프레임-불변 per-gaussian offset**(= mix_c[present] − center[present]) + sigma/yaw/weight 추출.
- 헬퍼 `_eval_mixture_scene_occ`: 그 offset을 present 프레임/trajectory 미래 프레임 중심에 다시 얹어 `forward_gaussian_mixture_scene` 호출. present는 `mix_c[present]`와 정확히 일치, 미래는 trajectory 중심 기반.
- present(pred_bev/정렬용)·미래(pred_txyz/metric용) 둘 다 mixture로. 출력 shape `[T,1,D,H,W]`는 단일 경로와 동일해 다운스트림 무변경.

**config**: `efficientocf_config.py`에 기본값 `query_eval_occ_use_mixture: False`(기존 config 동작 무변경) + apply. `_noreg`/`_guide` 둘 다 `True`로 설정 → 각각 poisson/union으로 학습=eval 정렬.

**검증**: 격리 torch 테스트로 (a) scene 출력 `[T,1,D,H,W]`, p∈[0,1], (b) **yaw=0→x로 길고 yaw=90→y로 길게** 회전 반영, (c) union peak(0.90)>poisson(0.74), (d) Q=60도 per-query 폭발 없이 동작 확인. py_compile 5파일 통과.
**주의**: scene 메서드는 per-query Python 루프(eval 전용, offline이라 OK). 학습 경로(grouped/loss/matcher)는 미변경.

## 2026-06-23 14:15:21 KST — GUIDE union + sigmoid opacity(추천2): voxelizer 곱-합성 분기 신설 (test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide)

**핵심 내용**: 추천1(합 공식 유지, weight 자유화)과 별개로, occupancy 합성을 GUIDE식 union으로 교체하는 경로를 신설. 합(Poisson) `p=1-exp(-Σ w·G)`는 활성 가우시안 하나의 중심 peak가 `1-exp(-w)`로 상한(w=1→0.63)인데, 곱(union) `p=1-Π(1-α·G)`는 중심에서 `p≈α`(α=1이면 정확히 1)에 자동 도달. weight를 mixture 지분(softmax/softplus)이 아니라 **α∈[0,1] opacity(sigmoid)** 로 재정의.

**매핑 결과(사전 조사)**: mixture occupancy를 만드는 가중-합성은 `voxelizer.py:forward_gaussian_mixture_grouped` **단일 메서드**뿐이고, 이를 (1) 학습 BCE loss(`utils_loss.py:2455`), (2) 매칭 cost(`utils_matcher.py:123`), (3) 시각화(`utils_visualization.py:1376`)가 공유. → 메서드 1곳 + voxelizer 인스턴스 2개에만 손대면 학습·매칭·시각화가 **일관되게** union으로 전환됨. (query_head의 coverage/hit loss 5052/5204는 unweighted 보조항이라 의도적으로 미변경. `_convert_occ_output`는 trilinear 경로라 무관.)

**코드 변경**:
- `voxelizer.py` `SoftVoxelizerOneAdd.__init__`: `gaussian_combine_mode="poisson"` 인자 추가(+검증). `forward_gaussian_mixture_grouped` 561행: `wg=w·gauss` 공유 후 `union`이면 `p=1-Π(1-wg).clamp(1e-6,1)`, 아니면 기존 `p=1-exp(-Σ wg)` 분기. factor clamp로 p∈[0,1] 수치 안전(비-sigmoid weight도 안전 캡).
- `query_head.py`: weight_mode에 `'sigmoid'` 추가 — validation(611), 분기(1807, `weights_qg=sigmoid(logits)`, surrogate는 sigma moment용 L1-norm), bias init 조건을 `softplus`→`(softplus, sigmoid)`로 확장(bias 0 → α start 0.5).
- `efficientocf_config.py`: 기본값 `query_multi_gaussian_occ_combine_mode="poisson"`(MODEL_CFG_DEFAULTS, `_merge_cfg`로 기존 config 전부 무변경) + apply에 `self.…` 설정.
- `efficientocf.py`: voxelizer 2개(viz `self.voxelizer`:220, 학습 `self.matched_gmo_voxelizer`:232) 모두 `gaussian_combine_mode=self.query_multi_gaussian_occ_combine_mode` 전달 → 두 인스턴스 일관.

**config 변경** (`test_traj_mcls_occ_jhh_traj0_past_bg_noli_guide.py`):
- `query_multi_gaussian_weight_mode`: softplus → **sigmoid**
- `query_multi_gaussian_occ_combine_mode`: (default poisson) → **union**
- `query_multi_gaussian_softplus_bias_init`: -2.0 → **0.0** (sigmoid bias도 이 인자 공유; α start 0.5)
- `query_multi_gaussian_weight_reg_loss_weight`: 1e-3 → **0.0** (union엔 sum 제약 무의미)
- `debug_query_mixture3d_vis_occ_threshold`: 0.2 → **0.5** (peak 회복)
- shape(sigma/offset/truncate)는 추천1과 동일하게 고정.

**검증**: 격리 torch 테스트로 (a) shape 일치 [Kc,nx,ny,nz], (b) union+sigmoid가 활성 중심에서 0.98, α=1·G=1 단일에서 정확히 1.0, (c) poisson+softmax는 낮게 깔림, (d) 모든 p∈[0,1] 확인. py_compile 5파일 통과.
**주의**: union은 반드시 sigmoid와 함께(α∈[0,1]). default(poisson)는 기존 동작 그대로라 다른 config 영향 없음.

## 2026-06-23 14:00:59 KST — mixture weight 자유화(추천1): softplus weight_reg off + bias 0 (test_traj_mcls_occ_jhh_traj0_past_bg_noli_noreg)

**핵심 내용**: occupancy가 작고 흐린 타원으로 collapse하는 원인을 weight 설계에서 진단. voxelizer는 `p = 1 - exp(-Σ w·G)` (Poisson union, `voxelizer.py:561-562`)인데, softplus weight를 `weight_reg`가 `target_sum=1`로 묶어 16개 가우시안이 개당 평균 ~0.06을 나눠 가짐. 그 결과 한 컴포넌트 중심 peak가 `1-exp(-0.06)≈0.06`까지 눌려 다수가 겹치는 코어만 threshold를 통과(=작은 타원). **합/곱 공식 비교**: 합(Poisson)은 한 컴포넌트 peak = `1-exp(-w)` (w=1→0.63), 곱(GUIDE union `1-Π(1-G)`)은 = 1. 1차로 합 공식을 유지한 채 weight 예산 제약만 제거.

**변경** (`projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_noli_noreg.py`):
- `query_multi_gaussian_weight_reg_loss_weight`: 1e-3 → **0.0** (sum→1 제약 제거, weight를 occupancy loss가 직접 결정. target_sum은 loss=0이라 무의미)
- `query_multi_gaussian_softplus_bias_init`: -2.0 → **0.0** (init weight softplus(-2)=0.13 → softplus(0)=0.69, 시작 peak 정상화)
- `debug_query_mixture3d_vis_occ_threshold`: 0.2 → **0.5** (viz 전용. 0.2는 낮은 peak 보정용이었음 → peak 회복 후 faint tail까지 보여 over-cover로 오독)
- **shape(sigma_max/offset_max/truncate/sigma_reg)는 의도적 고정**: weight 효과 격리(ablation) + 모델이 σ 재적응. mixture voxelizer truncate=`gaussian_truncate_sigma`=3.0(넉넉, 강한 가우시안 d_eff~1.7-2.5σ 안 잘림). (`query_attn_cam_gaussian_truncate_sigma=1.777`은 attn-cam 경로라 mixture와 무관)

**기대 효과**: 한 컴포넌트 중심 peak 0.06→0.63, 코어 0.9+. 객체 몸통 전체가 점유로 채워짐.
**주의/모니터**: (1) over-coverage(뭉뚝 블롭) 나오면 weight 되돌리지 말고 `sigma_max` 3.5→~2.0 / `sigma_reg` 0.001→~0.005로 조일 것. (2) weight가 여전히 학습되므로 소수 가우시안 몰빵(active-set collapse)은 구조적으로 안 막힘 → `dbg_query_match_spread_ratio`와 활성 가우시안 수 모니터, 관측되면 weight entropy/load-balance reg 추가. (3) 추천2(GUIDE 곱 union + sigmoid opacity)는 voxelizer/head 코드 분기 필요 → `_guide` config에서 별도 진행.

## 2026-06-23 — shape(multi-gaussian) 제한 공격적 완화 (test_traj_mcls_occ_jhh_traj0_past_bg_head_noli)

**핵심 내용**: center가 안정화되어 shape 학습 단계로 전환. 그동안 center-first 목적으로 조여뒀던 multi-gaussian offset/sigma 제한을 GT 박스 크기 분포에 맞춰 공격적으로 해제. 기존 `offset_max=(3,3,0.7)`, `sigma_max=(1,1,1)`은 car 반길이(~2.6m)밖에 못 덮어 bus/trailer(반길이 6.6~9.5m)를 구조적으로 못 맞추던 문제 해결.

**근거 데이터** (train infos pkl, PC range 내 fg 박스 긴축 길이 percentile):
- bus L p50/p99 = 12.0/14.5m, trailer 13.1/18.9m, truck 6.0/13.1m, construction 6.1/16.0m, 폭 p99 construction W~6.3m
- offset_max는 객체 반길이를 덮어야 함(offset은 world 축별 독립) → trailer p99 반길이 ~9.45m 커버 위해 10 선택

**변경** (`projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_noli.py` model_cfg):
- `query_multi_gaussian_offset_max_m`: (3.0,3.0,0.7) → **(10.0,10.0,2.0)**
- `query_multi_gaussian_sigma_max_m`: (1.0,1.0,1.0) → **(3.5,3.5,1.5)**
- `query_multi_gaussian_sigma_reg_loss_weight`: 0.01 → **0.001** (동반 필수; 0.01은 sigma를 1m로 끌어내려 sigma_max 확대를 무효화)
- `sigma_min`(0.15,0.15,0.15), `num_gaussians`(16), `weight_mode`(softplus)는 유지

**주의**: binary fg라 car~trailer가 이 한 세트를 공유 → 학습 후 **car class IoU/precision 모니터** 필요. 소형 객체 over-coverage 시 sigma_max를 3.0으로 하향. weight_reg(softplus default 1e-3)는 구조적 제약이라 미변경.

## 2026-06-23 10:36:12 KST — query mixture 3D 시각화에 score gate + GMM occ 임계값 행 추가

**핵심 내용**: `maybe_save_query_mixture_3d_vis`를 2행 6패널로 확장. 공통 score 임계값(기본 0.75)으로 query를 1차 게이트한 뒤, 상단 3패널은 기존 1σ 등고면, 하단 3패널은 mixture density를 모델 voxelizer로 복셀화해 `p=1-exp(-Σ w_g G_g) >= occ_thr`(기본 0.2)인 복셀을 실제 occupancy처럼 렌더.

**변경**:
- `projects/occ_plugin/occupancy/detectors/utils_visualization.py`: `maybe_save_query_mixture_3d_vis`에 (1) `sel_score >= score_thr` 공통 게이트, (2) `self.voxelizer.forward_gaussian_mixture_grouped`를 query당 K=1로 호출(메모리 절약)해 GMM occ 복셀을 월드좌표로 변환·클래스색 부여, (3) 2×3 subplot(상=1σ, 하=gmm occ≥thr) 렌더 추가. 헬퍼 `_finalize`로 축/뷰 설정 공유
- `projects/occ_plugin/occupancy/detectors/efficientocf_config.py`: `debug_query_mixture3d_vis_score_threshold`(0.75), `..._occ_threshold`(0.2), `..._occ_max_voxels_per_query`(4000)를 `VISUALIZATION_CFG_DEFAULTS`+`apply_visualization_cfg`에 추가
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`: 위 3개 값 명시(0.75 / 0.2 / 4000)

**검증**:
- `python -m py_compile` 통과
- synthetic mixture로 `forward_gaussian_mixture_grouped` 직접 테스트: 출력 `[1,1,1,40,512,512]`, `p=occ[0,0,0]`은 `[z,y,x]` 레이아웃, 좌표 역변환이 입력 center(~10,-4,0.5) 복원 확인. softmax weight 기준 max p≈0.40이라 0.2에서 복셀 생성·0.5는 공백 → 0.2 임계값 적절성 확인

## 2026-06-23 10:17:34 KST — query mixture 3D 시각화 이식 (ksh_shape 레포 → 본 레포)

**핵심 내용**: 옆 레포 `Autonomous_Driving_26_ksh_shape`의 `ksh_shape` 브랜치(커밋 `dc828e1`)에 있던 학습용 query mixture 3D 시각화(`maybe_save_query_mixture_3d_vis`)를 본 레포로 이식. 3-패널 matplotlib 3D(oblique z×2.8 / oblique rear / low side true-z)로 GT instance 복셀 + GT center + 선별 query + mixture 16성분 점/상위 6성분 yaw 반영 1σ wireframe ellipsoid를 렌더.

**변경**:
- `projects/occ_plugin/occupancy/detectors/utils_visualization.py`: `EfficientOCFVisualizationMixin`에 `maybe_save_query_mixture_3d_vis`와 `_QUERY_CLS_NAMES_8` 추가. 기존 `_build_query_visualization_bundle`의 `selected_mixture_centers/sigmas/yaw/weights_tqg3` 등 bundle 키를 그대로 사용(본 레포에 이미 존재)
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: forward_train의 `maybe_save_query_debug_vis` 호출 직후 `maybe_save_query_mixture_3d_vis` 호출 추가 (`gt_instance_occ3d_txyz_query_vis`, `gt_inst_center_world_full_tn3` 등 기존 변수 재사용)
- `projects/occ_plugin/occupancy/detectors/efficientocf_config.py`: `debug_query_mixture3d_vis_every`는 `DEBUG_CFG_DEFAULTS`+`apply_debug_cfg`에, `debug_query_mixture3d_vis_dir/max_queries/max_gt_points`는 `VISUALIZATION_CFG_DEFAULTS`+`apply_visualization_cfg`에 추가
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`: `debug_cfg`에 `debug_query_mixture3d_vis_every=48`, `visualization_cfg`에 dir/max 플래그 추가하여 활성화

**검증**:
- `python -m py_compile` (수정 4개 파일) 통과
- bundle 키(`selected_mixture_*`)가 본 레포 `_build_query_visualization_bundle`에 이미 존재함을 확인 → 데이터 경로 호환

## 2026-06-22 18:48:41 KST — head_gau config에 Gaussian-soft GT GMO target 적용

**핵심 내용**: `test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`에서 matched GMO loss target을 hard binary 대신 inside=1, outside=Gaussian halo인 soft GT로 사용할 수 있게 구현하고 해당 config에 활성화.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf_config.py`: `query_gmo_soft_gt_enabled`, `query_gmo_soft_gt_sigma_vox`, `query_gmo_soft_gt_truncate_sigma` 기본값 추가
- `projects/occ_plugin/occupancy/detectors/utils_loss.py`: low-res matched GT occupancy에 3D dilation shell 기반 Gaussian-soft halo를 적용하는 `_maybe_apply_query_gmo_soft_gt` 추가
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py`: soft GT 활성화, `(64,64,20)` low-res 기준 `sigma=2.5`, `truncate=3.0`, GMO dice 비활성화
- `projects/configs/baselines/test.py`: soft GT 옵션을 명시적으로 비활성화해 기본 실험 동작 유지
- `NOTES.md`: soft GT의 low-res voxel 단위 및 dilation-shell 근사 거리 주의사항 기록

**검증**:
- `python -m py_compile projects/occ_plugin/occupancy/detectors/efficientocf_config.py projects/occ_plugin/occupancy/detectors/utils_loss.py projects/configs/baselines/test.py projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past_bg_head_gau.py` 통과
- direct-file import synthetic tensor에서 center=1, 거리 증가에 따라 soft target 감소 확인

## 2026-06-22 18:43:15 KST — Gaussian-soft GT halo 3D 시각화 추가

**핵심 내용**: soft GT의 의미를 명확히 보기 위해 GT boundary와 soft target 등가면을 3D로 비교하는 시각화를 추가.

**변경**:
- `gt_gaussian/visualize_soft_gt_halo_3d.py`: GT boundary, `soft>=0.5`, `soft>=0.2` 등가면을 sigma별 3D scatter surface로 렌더
- `gt_gaussian/soft_halo_3d_frame3_inst14_sigmas_1p5_2p5_4p0.png`: sigma 1.5/2.5/4.0 voxel의 3D halo 퍼짐 비교
- `PROJECT_STRUCTURE.md`: 신규 3D soft halo 시각화 스크립트 및 산출물 설명 추가

**검증**: `python gt_gaussian/visualize_soft_gt_halo_3d.py` 실행 성공, PNG 렌더 확인.

## 2026-06-22 18:40:38 KST — Gaussian-soft GT halo sigma별 시각화 추가

**핵심 내용**: hard GT 내부는 1로 유지하고 외곽만 Gaussian distance-decay로 퍼뜨리는 soft target을 sigma별로 비교하는 시각화를 `gt_gaussian/`에 추가.

**변경**:
- `gt_gaussian/visualize_soft_gt_halo.py`: 실제 `segmentation_instance3d` 객체 mask에서 distance transform 기반 soft GT 생성 및 BEV/Z-slice 비교 PNG 저장
- `gt_gaussian/soft_halo_frame3_inst14_sigmas_0p8_1p5_2p5_4p0.png`: sigma 0.8/1.5/2.5/4.0 voxel의 퍼짐 정도 비교
- `gt_gaussian/soft_halo_frame3_inst14_sigmas_0p8_1p5_2p5_4p0.npz`: sigma별 soft target volume과 원 occupancy 저장
- `PROJECT_STRUCTURE.md`: 신규 soft halo 시각화 스크립트 및 산출물 설명 추가

**검증**: `python gt_gaussian/visualize_soft_gt_halo.py` 실행 성공, PNG 렌더 확인.

## 2026-06-22 18:15:44 KST — GT instance 3D의 GMM shape 데모 생성

**핵심 내용**: 실제 `segmentation_instance3d` GT 객체 하나를 K=16 diagonal Gaussian mixture로 피팅해 shape가 어떻게 보이는지 확인하는 데모를 `gt_gaussian/`에 추가.

**변경**:
- `gt_gaussian/visualize_gt_gmm.py`: GT instance voxel points를 로드해 sklearn `GaussianMixture(covariance_type="diag")`로 피팅하고 3D/BEV/density PNG와 파라미터 NPZ 저장
- `gt_gaussian/sample_frame3_inst14_gmm16.png`: frame 3, instance 14, 26752 voxels에 대한 K=16 GMM 시각화
- `gt_gaussian/sample_frame3_inst14_gmm16.npz`: fitted `means_xyz`, `sigmas_xyz`, `weights`, 원 GT `points_xyz` 저장
- `PROJECT_STRUCTURE.md`: 신규 `gt_gaussian/` 폴더 및 산출물 설명 추가

**검증**: `python gt_gaussian/visualize_gt_gmm.py` 실행 성공, PNG 렌더 확인.

## 2026-06-22 16:22:03 KST — Eval bbox GT source 및 pred weighting을 jhh 방식에 맞춤

**핵심 내용**: current eval의 남은 차이였던 bbox GT source와 selected query voxelizer weighting을 `efficientocf_jhh.py` 방식에 맞춤.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: eval `forward_test`/`simple_test`가 `segmentation`, `segmentation_cls_instance3d`를 명시적으로 받도록 확장
- bbox 보정 GT source 우선순위 변경: `segmentation_cls_instance3d` 우선, 없으면 `segmentation_instance3d`, `segmentation`, `gt_occ_inst` fallback
- `segmentation_cls_instance3d` 사용 시 jhh처럼 class channel을 bbox occupancy로 쓰고 pedestrian id 7은 제거
- selected query 렌더링 시 `selected_score_q`를 voxelizer `weights`로 전달해 jhh처럼 confidence가 occupancy intensity에 반영되도록 변경

**검증**: `python -m py_compile projects/occ_plugin/occupancy/detectors/efficientocf.py` 통과.

## 2026-06-22 16:10:19 KST — Eval metric을 present 단일 프레임에서 future tail로 변경

**핵심 내용**: current `efficientocf.py` eval metric을 `efficientocf_jhh.py`처럼 현재 프레임 1장 기준이 아니라 미래 horizon tail 전체 기준으로 누적 계산.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: 선택 query는 기존 actual inference scoring을 유지하되, metric용 prediction은 trajectory geometry(`centers_world_traj`, `sigmas_world_traj`)를 voxelizer로 렌더해 future tail `[T,X,Y,Z]`로 생성
- `IOU_movable objects`: future tail pred BEV와 `segmentation_bev` future tail을 누적해 2D confusion 계산
- `IOU_3d`: future tail pred 3D와 nusocc `gt_occ` future tail foreground로 계산
- `Recall_3d`: future tail nusocc 3D 기준 TP/FP/FN에 `gt_occ_inst` 기반 3D bbox AABB의 `bbox_FP`를 보정식으로 적용
- 3D metric은 jhh 방식에 맞춰 `gt_occ == 255`와 pedestrian class id 7을 valid mask에서 제외

**주의**: eval visualization PNG는 디버그 편의를 위해 기존처럼 present-frame query canvas를 저장하며, metric 계산만 future tail 기준으로 변경.

## 2026-06-22 15:13:45 KST — global_idx eval dataloader cat crash 수정

**핵심 내용**: eval 시각화 파일명을 위해 추가한 `global_idx`가 instance loader의 `torch.cat` 대상에 들어가 dataloader가 죽는 문제 수정.

**변경**:
- `projects/occ_plugin/datasets/pipelines/loading_instance.py`: cache-hit/cache-regenerate 양쪽 후처리 skip list에 `global_idx` 추가
- `global_idx`는 tensor sequence가 아니라 meta 값이므로 `torch.cat`하지 않고 뒤 pipeline/Collect3D meta로 전달

**검증**: 관련 dataset/eval 파일 `py_compile` 통과.

## 2026-06-22 15:07:52 KST — Recall_3d를 nusocc 3D + bbox 3D 보정식으로 변경

**핵심 내용**: `Recall_3d`를 기존 보정식 `((TP + bbox_FP) / (TP + FN + FP - bbox_FP))` 그대로 쓰되, TP/FP/FN은 nusocc `gt_occ`의 실제 3D foreground 기준으로 계산하고 `bbox_FP`만 bbox instance 3D AABB volume 기준으로 계산.

**변경**:
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`: eval `forward_test`/`simple_test`에서 `gt_occ`를 받아 3D metric에 사용
- `IOU_3d`: bbox instance GT가 아니라 nusocc `gt_occ` 3D foreground와 pred 3D voxel의 IoU로 계산
- `Recall_3d`: nusocc 3D 기준 TP/FP/FN에 bbox 3D AABB 내부 FP만 `bbox_FP`로 더하는 legacy 보정식 적용
- bbox 보정용 GT는 `gt_occ_inst`에서 instance별 XYZ axis-aligned bbox volume을 만들어 사용하며 height-map pseudo 3D는 사용하지 않음

**주의**: 이미 실행 중인 eval 프로세스에는 적용되지 않으며 새로 시작한 eval부터 동작.

## 2026-06-22 14:37:50 KST — Eval 중간 metric 터미널 출력 및 live log 저장

**핵심 내용**: 분산 eval 중간 metric을 48간격으로 터미널에 출력하고, 시각화 저장 폴더에 `eval_metrics_live.log`로 누적 저장.

**변경**:
- `projects/occ_plugin/occupancy/apis/test.py`: `EOCF_EVAL_METRIC_EVERY` 또는 fallback `EOCF_EVAL_VIS_EVERY` 간격마다 각 rank 누적 confusion/3D metric을 `dist.all_reduce`로 합산
- rank0에서 `[eval][N] ... (all ranks)` 형식으로 flush 출력
- `EOCF_EVAL_VIS_DIR`가 있으면 같은 폴더에 `eval_metrics_live.log` append 저장

**주의**: 이미 실행 중인 eval 프로세스에는 적용되지 않으며 새로 시작한 eval부터 동작.

## 2026-06-22 13:41:30 KST — Eval 시각화 파일명 global index 기반 변경

**핵심 내용**: 8GPU eval 시각화 파일명을 rank-offset(`iter_1000000`) 방식 대신 dataset global index 기반(`sample_000640`)으로 저장.

**변경**:
- `efficientocf_dataset.py`, `efficientocf_lyft_dataset.py`: sequence sample의 원본 dataset index를 `global_idx`로 pipeline results에 추가
- `tools/test.py`: eval/test pipeline의 `Collect3D.meta_keys`에 `global_idx`를 런타임으로 자동 추가
- `efficientocf.py`: eval vis 저장 시 `img_metas.global_idx`가 있으면 해당 값을 step으로 사용하고 filename prefix를 `sample`로 설정
- `query_head.py`: query debug 저장 파일 prefix를 bundle의 `_filename_prefix`로 받을 수 있게 변경 (`sample_000640_prob.png`, `sample_000640_query_vis.pt`)

**Fallback**: `global_idx`가 없으면 기존 rank-offset `iter_<rank*1000000+local>` 방식 유지.

## 2026-06-22 13:32:56 KST — Eval query 시각화 trajectory rows 제거

**핵심 내용**: eval 시각화에서 사용하지 않는 하단 trajectory row 2개(refined/base traj)를 생략해 캔버스를 4-row로 축소.

**변경**:
- `efficientocf.py`: eval vis 호출 시 query bundle에 `_skip_traj_rows=True` 플래그 추가
- `query_head.py`: 해당 플래그가 있으면 row5/refined traj 및 row6/base traj 렌더링·라벨·border·arrow drawing을 생략
- 학습 query debug vis는 기존 row 구성을 유지

## 2026-06-22 13:07:37 KST — Eval 시각화 기본 저장 경로 config 이름 기반 변경

**핵심 내용**: eval 시각화(`EOCF_EVAL_VIS=1`)의 기본 저장 위치를 실행 config 파일명 기준으로 자동 분리.

**변경**:
- `tools/test.py`: `EOCF_EVAL_VIS_DIR`가 명시되지 않은 경우 `./work_dirs/eval/<config_stem>/<timestamp>`를 기본값으로 설정하고 save dir 출력
- `tools/dist_test.sh`: 분산 rank별 저장 경로가 갈라지지 않도록 실행 시작 시 `EOCF_EVAL_TIMESTAMP`를 한 번 생성해 모든 rank에 상속
- 기존처럼 `EOCF_EVAL_VIS_DIR`를 직접 지정하면 해당 경로를 우선 사용

## 2026-06-21 KST — eval 시각화 추가(학습과 동일 캔버스, 전 랭크) + 15ep OCC 전체 eval 배치

**핵심 내용**: `simple_test`(eval 경로)에 학습과 **동일한 query debug 시각화**(`maybe_save_query_debug_vis`, 5-row prob.png)를 env-gated로 붙임. 8-GPU 분산 eval에서 **전 랭크가 저장**하도록 우회 플래그 추가. 15에포크 완주한 OCC 파생 6개를 최신순·8GPU·순차로 eval하는 배치 스크립트 작성.

**변경**:
- `efficientocf.py`: `EfficientOCF._maybe_save_eval_query_vis()` 신규 — `simple_test`가 이미 만든 `pred_occ`(선택 query 렌더)·`gt_inst`·`query_vis_bundle`·`cls_scores`를 학습용 `query_head.maybe_save_query_debug_vis`에 그대로 넘겨 동일 이미지를 생성. `simple_test` 본문에 `_pred_occ_vis` 캡처 + 1줄 호출 추가. **metrics엔 영향 0**(렌더만), 실패해도 eval 안 깨지게 try/except.
  - env: `EOCF_EVAL_VIS=1`(켜기, 기본 off), `EOCF_EVAL_VIS_DIR`(출력 dir), `EOCF_EVAL_VIS_EVERY`(샘플 게이트, 기본 50). 파일명 `iter_{rank*1e6+local}_prob.png`로 랭크별 유니크.
- `query_head.py`: `_eval_vis_enabled_this_rank()` 신규 + `_maybe_save_prob_grid_vis`의 `if not self._is_main_process(): return`를 이걸로 교체. `self._eval_vis_all_ranks=True`(eval 헬퍼가 set)면 rank0 외 랭크도 저장 → 분산 shard 전체 커버. 학습은 플래그 미설정이라 **기존 rank0-only 동작 유지**.
- `run_eval_occ_all.sh` 신규: 15ep OCC 6개(past_bg, res34, past, traj0, jhh, occ) 최신순 8-GPU 순차 eval. `EOCF_EVAL_ALIGN="2,1,0,0"`·occ thr 0.5(스모크에서 pred≈gt count 확인, placement 문제라 thr 무관). 결과/viz를 `work_dirs/<run>/eval/{eval_metrics.log, vis/}`에 저장.

**검증(스모크, 8GPU)**: past_bg 48샘플 eval 정상(exit0, ~1.1 task/s = 2GPU의 4배), viz ON에도 metrics 동일(movable IoU2d 0.083). 전 랭크 저장 확인(24샘플→24장, rank0~7 균등, skip 0). 이미지는 학습과 동일한 GT/candidates/selected/matched/refined-traj 5-row + legend.

## 2026-06-20 KST — QueryDepthHead 용량 ↑ (past_bg_head) + depth 성능 dbg

**핵심 내용**: query별 depth bin 분류기(`QueryDepthHead`)를 용량 파라미터화하고, depth 예측 품질을 보는 dbg 추가. `past_bg` 위에 head만 키운 변종 config 생성.

**동기**: depth는 죽은 LSS DepthNet이 아니라 **QueryDepthHead**(query_head.py:120, 2-layer MLP embed→64bin)가 담당. 이 depth로 soft-argmax expected depth를 구해 query를 3D center로 **lift**(efficientocf.py:1076 `_build_query_attn_soft_lift_pack` → :1086 `apply_lifted_centers_to_outputs`)하므로 depth 품질이 곧 lift center 정확도. 2-layer는 64-way 분류엔 작아 underfit 의심 → 용량↑로 depth 분포를 날카롭게.

**변경**:
- `query_head.py`: `QueryDepthHead.__init__`에 `num_layers/hidden_mult` 추가(기본 2/1 = 기존과 동일). `QueryHead.__init__`에 `query_depth_head_num_layers/hidden_mult` 추가 + QueryDepthHead 빌드(:697)에 전달.
- `efficientocf.py`: QueryHead 빌드(:181)에 두 인자 전달.
- `efficientocf_config.py`: `MODEL_CFG_DEFAULTS`에 `query_depth_head_num_layers=2`, `query_depth_head_hidden_mult=1` + apply 배선.
- 새 config `test_traj_mcls_occ_jhh_traj0_past_bg_head.py` = past_bg + `num_layers=3, hidden_mult=2` (96→192→192→64). past_bg 대비 head만 다른 깨끗한 A/B.

**depth 성능 dbg** (`utils_query_projection.py` `_compute_query_depth_loss_from_match`, CE 직후, matched+valid pair):
- `dbg/query_depth_top1_acc`, `within1_acc`: argmax bin이 GT bin과 일치 / ±1 이내 비율.
- `dbg/query_depth_bin_abs_err`: |argmax_bin − gt_bin| (bin 단위, 1bin≈0.875m).
- `dbg/query_depth_soft_bin_abs_err`: |soft-argmax(expected) bin − gt| (**lift가 실제 쓰는 값**).
- `dbg/query_depth_entropy`: depth 분포 엔트로피(낮을수록 날카로움).
- prefill(out dict)에 5키 추가 → 멀티GPU log_vars 키셋 일치. `with torch.no_grad()`로 grad 무영향.
- **strip allowlist 수정**(utils_loss.py:3682): `dbg_query_depth_` 접두사 허용 추가 — 안 그러면 잘림(기존 `dbg_query_depth_valid_count` 등도 이번에 같이 살아남).

**검증**: py_compile 6파일 통과. depth dbg 로직 합성 검증(sharp→top1=1/err=0/ent=0, flat→ent=ln8=2.08, off-by-one→within1=1/bin_abs=1). 기본값 2/1이라 다른 config 무영향(EfficientOCF_V1.1_1gpu.py 수정 불필요).

**비교법**: `past_bg`(head 기본) vs `past_bg_head`(3층/wider) → head 효과만 분리. depth dbg가 좋아지는지(top1↑/abs_err↓/entropy↓) + center_l2↓ 확인. 늘 그렇듯 IOU 기준점 먼저.

## 2026-06-19 KST — `test_traj_mcls_occ_jhh_traj0_res34.py` 백본 ResNet18 → ResNet34

**핵심 내용**: 해당 config의 `img_backbone`을 ResNet18에서 ResNet34로 교체.

**변경**:
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_res34.py`: `pretrained='torchvision://resnet18'` → `'torchvision://resnet34'`, `depth=18` → `depth=34`.

**범위 주의**: ResNet18/34 모두 `BasicBlock` 사용 → stage별 출력 채널 `[64, 128, 256, 512]` 동일하므로 `img_neck.in_channels` 무변경. ResNet50+ (`Bottleneck`, `[256,512,1024,2048]`)로 갈 때만 neck도 같이 수정 필요.

## 2026-06-19 KST — matched-pair GMO+center loss 과거3 제한 플래그 (`query_matched_loss_history_only`)

**핵심 내용**: matched-pair **GMO(occupancy BCE/dice)** 와 **center match** loss를 7프레임 전체가 아니라 **receptive-field(과거+현재) 프레임 `[0:time_receptive_field]`** 에만 걸도록 하는 플래그 추가. 미래4(traj로 민 프레임)는 **trajectory loss만** supervise.

**동기**: 미래 프레임 geometry는 present Gaussian을 traj offset으로 평행이동한 것(`_build_query_trajectory_geometry_from_present`)이고 sigma/yaw/weight는 present 복사 → (1) 미래 GMO는 shape 자유도 0이라 present sigma를 미래GT 쪽으로 끌어당겨 **present 모양 오염** 위험, (2) 미래 center loss는 `present center + traj offset` 학습이라 **traj loss와 중복**. 따라서 미래는 traj에 일임.

**변경**:
- `efficientocf_config.py`: `MODEL_CFG_DEFAULTS`에 `query_matched_loss_history_only`(기본 `False` = 기존 7프레임 동작 유지) 추가 + apply 배선.
- `efficientocf.py`: loss 섹션에 `_hist_slice` 헬퍼 추가 — 플래그 ON **및 `use_full7_temporal_sup` True**일 때만 `centers/sigmas/mixture_*/gt_instance_occ3d_txyz`를 `[:time_receptive_field]`로 슬라이스. `_compute_query_center_match_loss_from_match`, `_compute_matched_pair_gmo_losses` 두 호출에만 적용. center loss는 `t_match=min` 로직이 GT를 자동으로 history로 슬라이스, gmo는 `T=min(Tq,To)`라 centers+gt_occ 슬라이스로 history만 계산.
- `projects/configs/baselines/test_traj_mcls_occ_jhh_traj0_past.py`: `query_matched_loss_history_only=True` 활성화.

**범위 주의**: matching(`_match_queries_to_gt_instances`)·recruit·center dbg는 그대로 7프레임 사용(매칭 cost는 이미 history 3프레임만 씀 → 영향 없음). traj/refine/cls/depth/attn 등 나머지 loss 무변경. 미래 occupancy IoU는 이제 `present shape + traj 위치`가 책임.

**검증**: py_compile 통과(3파일). `use_full7_temporal_sup=False` 경로(match 텐서가 present+future 5프레임)에서는 슬라이스 비활성 → 잘못된 history 해석 방지.

## 2026-06-18 KST — matcher cost 분석 dbg 2차 확장: fliprate / decisiveness / topk_colstd / range-coverage / spread / corr

**핵심 내용**: colstd+margin에 이어, 각 cost 항의 **실질적(인과) 영향**과 **공간 커버리지**를 측정하는 dbg 다수 추가. workflow로 in-scope 텐서 검증 후 추천 항목 전부 구현.

**추가 metric** (`projects/occ_plugin/occupancy/detectors/utils_loss.py`, match-cost dbg 블록):
- `dbg_query_match_cost_{name}_fliprate` (P0): term 제거 시 열별 argmin 승자가 바뀌는 비율 = **counterfactual 실질 영향**. cost=Σcontrib라 `cost-contrib`가 정확한 leave-one-out. 평평하면(flat) 0 → "평균 큰데 영향 0"을 직접 잡음.
- `dbg_query_match_center_err_{near,mid,far}_mean` + `_count` (P0): matched query→GT center err를 **ego-거리 버킷**으로 (R_max=`self.point_cloud_range` corner, 1/3·2/3 분할). 외곽 매칭 정확도 저하 진단. count로 빈 버킷(z) vs 진짜 0 구분.
- `dbg_query_match_cost_{name}_decisiveness_rate` (P1): matched 열에서 그 항이 **최대 margin(최다 타이브레이커)** 인 비율. 7항 합 ~1.
- `dbg_query_match_cost_{name}_topk_colstd_mean` (P1): 열별 **총비용 top-5 경쟁자 한정** contrib std = global colstd보다 결정영역 변별을 덜 희석.
- `dbg_query_match_far_gt_frac`, `_spread_q/_spread_g/_spread_ratio` (P2): far 버킷 비중 + matched query 공간 분산/GT 분산 비(중앙 콜랍스 지표).
- `dbg_query_match_cost_center_temporal_offset_corr` (P2): 두 거리계열 항 중복성(Pearson).

**SKIP**: argmin_share(fliprate에 포함), unmatched-GT coverage(Q≫N라 active GT 전부 1:1 매칭→측정 대상 없음).

**검증**: py_compile 통과. 합성 데이터로 6종 전부 로직 검증(spread 항=양수, flat 항=0; 버킷 stratify; corr=1.0). **prefill↔record 동기화**: 신규 키 전부 무조건 prefill→record는 덮어쓰기만이라 멀티GPU log_vars 키셋 일치 보장(빈 버킷/무경쟁 step은 z 유지).

**주의**: `self.point_cloud_range`는 mixin이 아니라 host detector(efficientocf.py:142) 속성 — runtime `self`로 접근. 새 dbg는 **새 run/재시작부터** 기록(돌고 있는 occ엔 미반영). config 인자 추가 없음.

**버그픽스(같은 날)**: traj0 검증 중 coverage/spread 키가 TB에 누락 발견. 원인 = `_aggregate_training_losses` 말미의 dbg strip allowlist(utils_loss.py:3674)가 `dbg_query_match_cost_` 접두사만 허용 → `dbg_query_match_center_err_*`/`far_gt_frac`/`spread_*`가 삭제됨(corr는 match_cost라 생존). allowlist를 `dbg_query_match_cost_` → `dbg_query_match_`로 넓혀 해결(전부 prefill돼 멀티GPU 안전). per-term fliprate/decisiveness/topk_colstd은 match_cost 접두사라 이미 정상 기록 중이었음.

## 2026-06-18 KST — matcher cost 변별력(discrimination) dbg 추가: colstd + margin

**핵심 내용**: Hungarian cost matrix에서 각 항(center/attn/cls/temporal_offset/...)의 **평균 비중(ratio)이 아니라 실제 변별력**을 측정하는 dbg 2종 추가. "평균은 크지만 쿼리마다 평평하면 매칭에 영향 없음" 문제를 진단하기 위함.

**주요 변경사항** (`projects/occ_plugin/occupancy/detectors/utils_loss.py`, match_cost dbg 블록만):
- `dbg_query_match_cost_{name}_colstd_mean`: GT 열(column)마다 그 항 contrib을 **후보 쿼리들 across로 std** → 후보 ≥2인 열 평균. = 변별력(작으면 평평→매칭 영향 0).
- `dbg_query_match_cost_{name}_margin_mean`: 매칭된 열에서 **runner-up(총비용 2등 쿼리) contrib − 선택 쿼리 contrib**. >0이면 그 항이 그 할당을 실제로 결정, ~0이면 무관(예: flat cls). runner-up은 항-독립이라 1회 계산 후 재사용.
- 7개 항 모두에 대해 prefill 루프와 record 지점 **양쪽에 키 추가**(멀티GPU log_vars assert 방지, NOTES:52 규약).

**검증**: py_compile 통과. 합성 cost matrix로 로직 검증 — spread 항(center) colstd>0/margin>0, flat 항(cls) colstd=0/margin=0으로 정확히 구분 확인.

**적용 범위/주의**: config 인자 추가 없음(순수 dbg). 돌고 있는 occ run은 모듈 이미 import돼 **재시작해야 반영**, occ_jhh/traj0 등 신규 run은 처음부터 기록. EfficientOCF_V1.1_1gpu.py/efficientocf_config.py 변경 불필요(신규 키 없음).

## 2026-06-18 KST — test_traj_mcls_occ_jhh.py: BEV dim 64→96 + DT loss off

**핵심 내용**: jhh occ 실험 config에서 BEV feature 채널을 64→96으로 키우고, query DT loss를 비활성화.

**주요 변경사항** (`projects/configs/baselines/test_traj_mcls_occ_jhh.py`만 수정):
- `bev_feat_dim = 64 → 96` (line 357). 단일 source-of-truth라 `numC_Trans`(358)·`query_embed_dim`(model_cfg 377)·`img_view_transformer numC_Trans`(507)로 자동 전파. 다른 키/소스 수정 불필요.
- model_cfg에 `use_query_dt_loss=False` 추가. `utils_loss.py:3340`의 `if self.use_query_dt_loss:` 브랜치가 통째로 skip → `loss_query_dt` 미생성. `load_occ_dt`는 True 유지(미사용 dead data, 무해).

**검증** (병렬 조사 + 적대적 검증, 신뢰도 high):
- BEV 채널 dim은 LSS view transformer→DepthNet context_conv→query transformer→query head까지 end-to-end **parametric** (채널 경로에 하드코딩 64 없음). attention `num_heads=4`(efficientocf.py:129), 96%4==0 → head 변경 불필요.
- 건드리면 안 되는 것: `img_neck out_channels=[128,128,128,128]`(497), `numC_input`(미설정, 512=4×128 SECONDFPN concat).
- DT는 weight 키가 없고 boolean toggle만 존재(efficientocf_config.py:13 기본 True, :216 wiring). loss_weight=0.1은 inline 하드코딩. weight=0 방식 불가, toggle만이 유일 제어.
- log_vars assert 무관(`loss_query_dt` prefill 없음; NOTES:52 경고는 `dbg_query_match_cost_*` 한정).

**주의**: BEV 채널 ~50%↑ → GPU 메모리 증가, OOM 모니터링 필요. efficientocf_config.py mirror 기본값(query_embed_dim=256, use_query_dt_loss=True)은 이 run의 model_cfg가 override하므로 수정 불필요(다른 config엔 영향 없음).

## 2026-06-16 KST — train 스크립트 USE_MPS=0 적용 (기존 MPS 데몬 재사용, 데몬 관리 안 함)

- 배경: `tools/dist_train.sh`의 `pgrep -x nvidia-cuda-mps-control` 데몬 감지 체크는 항상 실패함.
  커널이 comm 이름을 15자(`nvidia-cuda-mps`)로 자르는데 `-x`로 23자 풀네임을 찾기 때문.
  → `USE_MPS=1`(기본값)이면 매번 `nvidia-cuda-mps-control -d`로 새 데몬을 띄우려 시도.
- 대응: `train_total.sh`, `train_total_2.sh` env 블록에 `USE_MPS=0` 추가.
  - 스크립트가 MPS 데몬을 시작/관리하지 않음 (세션 종료 시 공유 데몬 영향 우려 제거).
  - 이미 떠 있는 데몬이 기본 파이프 `/tmp/nvidia-mps`에 있고 CUDA 클라이언트는 env 미설정 시
    해당 기본 경로로 자동 연결되므로, 기존 MPS 가속은 그대로 유지됨.

## 2026-06-15 18:16 KST — Gate + Recruit 구현 및 실험 설정

- `efficientocf_config.py`: Gate/Recruit 설정 기본값과 model attribute 바인딩 추가.
- `utils_matcher.py`: query-GT 최소 XY 거리 기반 8m hard gate, gate 이후 unmatched GT와 free query의 15m greedy recruit pairing 추가.
- `utils_loss.py`: recruit pair에 classification label 없이 center L1 pull loss와 DDP-safe debug key 추가.
- 미래 trajectory가 gate 판정에 섞이지 않도록 기존 temporal matching frame index를 거리 계산과 recruit loss에 동일하게 사용.
- `projects/configs/baselines/test_traj_mcls_gate.py`에 권장 설정 추가:
  `query_match_center_gate_radius_m=8.0`,
  `query_recruit_max_radius_m=15.0`,
  `query_recruit_loss_weight=0.1`.
- 검증: py_compile, config/model attribute 파싱, Gate off 회귀, hard gate/recruit pairing,
  recruit loss gradient, 미래 frame 제외 synthetic smoke test 통과.

## 2026-06-15 KST — MPS 멀티잡 실측 검증 + 런북 문서화

### 한 노드 8장에서 학습 2개 동시 실행 (MPS) 실험
- 실측(각 조건 40 iter 평균, test_traj + test_traj_mcls):
  - 1잡 단독: 8.2s/iter
  - 둘 다 non-MPS(겹침): 30s/iter (단독보다도 느림, 노드 처리량 0.066 iter/s)
  - 한쪽만 MPS: ~32s/iter (효과 0, 제로섬)
  - **둘 다 MPS: 9.0s/iter (3.3배 빠름, 노드 처리량 0.220 = 단독 1.79배)**
- 결론: MPS는 **두 잡 모두 클라이언트일 때만** 효과. 단독 GPU util 24~60%(launch-bound)라 빈 SM을 MPS가 채워 큰 이득.
- 신규 문서 `MPS_MULTIJOB_RUNBOOK.md` 추가 (복붙 절차 + 함정 정리). 이론은 기존 `GPU_MULTIJOB_NOTES.md`.

### MPS 상시적용
- `tools/dist_train.sh`: MPS 데몬 자동 시작 블록 추가(기본 ON, idempotent). 모든 학습 launch가 자동으로 MPS 클라이언트로 시작됨. `USE_MPS=0`으로 비활성 가능.
- 종료용 `stop_mps.sh` 추가 (quit→정리 순서 안전 처리).
- ⚠️ 트레이드오프: MPS는 컨텍스트 공유라 한 잡 치명오류 시 같은 GPU 다른 잡도 영향 가능(격리 약화). 단독 1잡은 이득 없음(무해).
- PORT/--work-dir 잡마다 다르게 주는 건 여전히 사용자 책임 (MPS가 안 챙김).

## 2026-06-15 KST — mode-cls moving 클래스 가중 (mcls A/B 실험)

### 배경
- epoch 14 분석: mode 분류기가 static으로 쏠려 moving을 11%만 예측 (GT 31%).
- 보정용으로 mode 분류 CE에서 moving(비정지) 클래스에 가중을 주는 신규 인자 추가.

### 신규 인자: `query_traj_mode_cls_moving_class_weight` (기본 1.0 = 무변화)
- `efficientocf_config.py`: DEFAULTS 등록 + 파싱.
- `efficientocf.py`: trajectory loss 호출에 전달.
- `utils_loss.py`: `_compute_query_trajectory_loss_from_match` 시그니처에 추가.
  static_gate **비활성** 경로(평범한 mode CE, L941 부근)에서 `F.cross_entropy(weight=...)`로 적용.
  weight 텐서 = 전 클래스 가중값, stationary 인덱스만 1.0으로 되돌림 (비정지 전부 가중).
  값이 1.0이거나 stationary_mode_idx가 없으면 weight=None(기존과 동일).

### config
- `test_traj_mcls.py`: `query_traj_mode_cls_moving_class_weight=4.0` 추가. 그 외 test_traj.py와 100% 동일 (A/B 비교).
- baseline `EfficientOCF_V1.1_1gpu.py`: traj-mode 키를 선언하지 않는 기존 패턴 유지 → DEFAULTS 1.0으로 동작 변화 없음.

## 2026-06-11 KST (9차)

### suppress weight 변경 (test_traj.py)

- `query_attn_bbox_other_weight` 0.1 → **0.3** (사용자 결정; cost 레포 검증 baseline은 0.1이었음).
- 시각화 score 가중치를 cls 단독으로 변경: `debug_query_score_cls_weight` 0.3→**1.0**, `cam_attn_weight` 0.7→**0.0** (iou는 0.0 유지). 이제 vis score = fg cls_prob 그대로.

## 2026-06-11 KST (8차)

### traj warmup/TF 스케줄 사용자 재설계 (test_traj.py)

- `mode_cls_loss_weight`를 warmup ramp 없이 **epoch 1부터 0.1 고정** (기존 0.02→0.05→0.10 계단 제거).
- TF gt_ratio를 **epoch 1~4 풀 유지(1.0)** → epoch 5~7 ramp(0.9/0.7/0.5) → epoch 8부터 0.0으로 변경. refine 활성(epoch 5)과 TF 전환 시작이 동시에 일어나는 구조.
- traj_loss_w / refine_w 계단은 기존 유지 (0.05/0.15/0.25/0.5, 0/0.02/0.05/0.10).
- 검증: 15 epoch 전체 stage 해석 표 확인.

## 2026-06-11 KST (7차)

### Teacher forcing을 iter 기준 → epoch 기준으로 전환 (test_traj.py)

- iter 스케줄(`schedule_iters/gt_ratios`)을 빈 튜플로 비활성 → 코드가 `query_traj_teacher_forcing_gt_ratio` 속성값을 직접 사용하는 경로로 전환 (efficientocf.py L323-324, 코드 수정 없음).
- `traj_warmup_schedule`에 epoch 2, 3 stage를 추가하고 모든 stage에 `query_traj_teacher_forcing_gt_ratio` 포함: epoch 1=1.0 → 2=0.5 → 3+=0.0. 원본(1-GPU 4000 iter/epoch 기준 epoch 0.75~2.0 전환)을 epoch 계단으로 근사.
- 이제 GPU 수가 바뀌어도 TF 스케줄 재스케일 불필요.
- 주의: TrajectoryWarmupHook은 마지막 매칭 stage "하나만" 적용하므로 각 stage가 전체 키를 보유해야 함 (config 주석으로 명시).
- 검증: epoch별 stage 해석 (1→1.0/2→0.5/3→0.0, weight 계단 유지) 확인. 기존 8-GPU run은 재시작해야 반영됨.

## 2026-06-11 KST (6차)

### 디버그 시각화 클래스 색상 얼라인

- `query_head.py::_get_query_vis_palette`: 위치 기반 palette → **raw id 고정 매핑**으로 변경. 8-class 전환(pedestrian 제외)으로 trailer/truck 색이 한 칸씩 밀리던 문제 해결, 클래스 목록이 바뀌어도 색 고정. 목록에 없는 raw id는 extra 색 순환.
- `utils_instance_img_debug.py::_instance_color`: 별도 팔레트를 query vis와 동일한 raw-id 매핑으로 통일 → 모든 디버그 PNG에서 같은 클래스 = 같은 색.
- 검증: trailer=(140,255,100), truck=(255,120,220) 등 원래 의도 색 복원 확인. 시각화 전용 변경이라 학습 무영향.

## 2026-06-11 KST (5차)

### Trajectory 2-mode 모듈 이식 (TRAJECTORY_CONFIG.md / cen10_6, traj 레포에서)

**핵심 내용**: `/home/hwanhee/Autonomous_Driving_26_traj`(같은 커밋 dd6c98d 기반)의 2-mode trajectory 브랜치를 3-way merge로 이식. 적용 config는 `test_traj.py` (test.py는 traj 기능 전부 기본값 off라 기존 동작 유지).

**3-way merge (base=dd6c98d, ours=cost 이식본, theirs=traj)**:
- `query_head.py`, `efficientocf.py`, `efficientocf_config.py`, `utils_visualization.py`: 클린 머지
- `utils_loss.py`: 충돌 2곳 수동 해결 (dbg prefill 중복 → total 버전 유지, dbg 유지목록 → cls_/traj_ 합집합)

**부수 반영**:
- `efficiency_hooks.py`: `TrajectoryWarmupHook` 추가 (traj 버전 복사)
- `loading_instance.py`/`loading_occupancy.py`: `exclude_occ_class_ids` 로드 단계 클래스 제거 지원 (traj 버전 복사)
- `mmdet_train.py`: `cfg.custom_hooks` 등록 루프 추가 (없으면 TrajectoryWarmupHook 미등록)
- `__init__.py` 2곳: TrajectoryWarmupHook export
- `tensorboard_hooks.py`: traj 버전 무시, 기존(cost) `TextLoggerHookNoDbg` 유지
- traj의 `tools/train.py` vis remap은 미이식 (cost의 mmdet_train 방식 유지)

**test_traj.py** (test.py 복사본에 추가):
- §3 전체: `query_traj_num_modes=2`, `semantic_routing_enabled=True`, `static_threshold_m=0.8`, `mode_infer_policy='argmax'`, xy refine(2층/64dim/weight 0.1), mode_cls_w=0.1, moving_w=5.0
- `traj_warmup_schedule` 4단계 (epoch 1/5/8/11) + `TrajectoryWarmupHook` 등록
- teacher forcing mix 스케줄: **1000 iter/epoch(4GPU) 기준 (750,1000,1250,1625,2000)으로 재스케일** (원본 4000 iter/epoch에 3000~8000)
- `exclude_occ_class_ids=(7,)`: pedestrian을 로드 단계에서 제거 (train instance/occupancy + test instance 로더, cen10_3과 동일하게 test LoadOccupancy는 제외)

**검증**: test_traj.py build_model 통과 (2-mode, refine 서브모듈 3개 생성, TF 스케줄 확인), test.py 호환성 통과 (num_modes=1, refine off), TrajectoryWarmupHook cfg 빌드 통과.

## 2026-06-11 KST (4차)

### cls match cost 활성화 (test.py)

- `query_cls_match_cost_weight=0.075` 추가 — Hungarian cost matrix에 `-log P_q(GT class)` NLL 항(clamp 30, 기여 상한 0.075×30=2.25) 활성화.
- 코드는 2차 이식 때 들어온 `utils_matcher.py` L586-605 그대로 사용, config 값만 켬. cost 레포 `*_clscost.py` run과 동일 값.
- 주의: from-scratch 학습 전제 (학습 도중 켜면 악화 관측, QUERY_ATTN_CHANGES.md 3번).

## 2026-06-11 KST (3차)

### ⑤ 하이퍼파라미터 일괄 반영 + pedestrian 제외 (test.py)

- `query_num_queries=200`, `query_center_match_cost_weight` 4.0→10.0, `query_temporal_offset_match_cost_weight` 2.0→0.5
- 클래스 구성을 문서/cls_5 기준 8-class로 변경: `class_names`에서 'pedestrian' 제거, `query_class_ids=[0,2,3,4,5,6,9,10]` (7 제외), `query_cls_loss_class_weights=[0.02, 1.4, 1.3, 0.3, 1.4, 1.4, 1.2, 0.9]`
- pedestrian은 사용자 결정으로 학습/추론/시각화 전체에서 제외 — gt_prep의 allowed_raw_ids 필터가 로드 직후 background로 처리
- cost 레포의 `projects/occ_plugin/utils/formating.py` 이식 (8-class 평가 클래스명 매핑)
- 검증: test.py `build_model` 통과, num_queries/class_ids/cls_weights/cost weight 모두 확인

## 2026-06-11 KST (2차)

### QUERY_ATTN_CHANGES.md 항목 이식 (①②③④⑥⑦⑧, cost 레포에서)

**핵심 내용**: `/home/hwanhee/Autonomous_Driving_26_cost`(같은 커밋 dd6c98d 기반)의 uncommitted 변경에서 query-attention/matching 개선을 이식. cost 버전 파일을 통째로 복사하는 방식 사용.

**복사한 파일** (cost → total):
- `detectors/utils_loss.py`: ① attn loss를 KL → inside-mass NLL(`-log(inside_mass)`)로 교체, ② suppress(타 객체 영역 질량 페널티) 항 추가, ⑩ 기존 수동 dbg prefill을 cost 버전으로 통일
- `detectors/utils_matcher.py`: ③ Hungarian cost에 `inside_log` metric 추가
- `detectors/utils_gt_prep.py` + `detectors/efficientocf.py`: ④ history-all-valid 인스턴스 필터 (`_build_history_all_valid_instance_ids`, `_filter_dense_instance_ids`)
- `detectors/efficientocf_config.py`: ⑥ 신규 키 등록 (`query_attn_bbox_other_*`, `query_require_history_all_valid`, `query_attn_overlap_loss_weight` 등)
- `apis/mmdet_train.py`: ⑦ `_configure_visualization_dirs` — vis 경로를 `work_dirs/<config>/vis/<timestamp>/` 아래로 자동 재매핑
- `dense_heads/query_head.py` + `detectors/utils_visualization.py`: ⑧ 디버그 시각화 개선 (selected query distance-NMS, 클래스 팔레트 수정)
- `image2bev/transformer.py`: ⑨ `query_attn_overlap_loss` — **계획에 없었으나 efficientocf.py가 constructor kwarg로 하드 의존하여 같이 이식. 기본 weight 0.0이라 비활성.**

**test.py 변경**:
- 추가: `query_attn_match_metric='inside_log'`, `query_attn_bbox_other_weight=0.1`, `query_attn_bbox_other_mode='union'`, `query_attn_bbox_unmatched_weight=0.0`, `query_require_history_all_valid=True`
- 변경: `query_attn_match_cost_weight` 0.5 → 0.3 (문서 권장값)
- 유지: `query_center_match_cost_weight=4.0`, `query_temporal_offset_match_cost_weight=2.0`, query 수, class weights (⑤ 미적용, 사용자 직접 조정 예정)

**검증**: test.py로 `build_model`까지 정상, 신규 attr 값 모두 확인.

## 2026-06-11 KST

### 1~8 GPU DDP 호환성 수정 (4_CHANGELOG.md의 2026-05-15 14:20 수정 이식)

**핵심 내용**: 이전 레포(`4_CHANGELOG.md`, `4_NOTES.md`)에서 검증된 multi-GPU DDP 안정화 수정을 현재 레포에 적용.

**주요 변경사항**:
- `tools/train.py`: distributed 초기화 전에 `torch.cuda.set_device(int(os.environ['LOCAL_RANK']))` 추가
- `projects/occ_plugin/occupancy/apis/mmdet_train.py`: `MMDistributedDataParallel`에 `init_sync=False`, `static_graph=True` 추가 (torch 2.7 지원 확인, `find_unused_parameters`는 유지)
- `projects/occ_plugin/occupancy/detectors/utils_loss.py`: `_aggregate_training_losses()`에서 조건부로 생성되던 `dbg_query_match_cost_*` 키 전체를 모든 rank에서 항상 0으로 선생성하도록 수정 → rank별 `log_vars` 키 불일치로 인한 `loss log variables are different across GPUs!` assert 방지

**실행기**: 별도 `train_total.sh`는 만들지 않음. 기존 `run.sh` → `tools/dist_train.sh`가 이미 config/GPU 수/PORT를 인자·환경변수로 받는 구조라 그대로 사용 (`PORT=24560 bash run.sh <config> 8`).

## 2026-06-11 10:20 KST

### 터미널 로그에서 dbg metric 숨김

- `projects/occ_plugin/core/evaluation/tensorboard_hooks.py`에 `TextLoggerHookNoDbg` 추가: `dbg_`/`dbg/` prefix 키를 터미널·JSON 로그에서만 필터링 (TensorBoard `dbg/*` 기록은 `TensorboardLoggerHookSplitTabs`로 유지).
- `test.py`, `EfficientOCF_V1.1_1gpu.py`, `EfficientOCF_V1.1_1gpu_traj_tf.py`의 text logger를 `TextLoggerHookNoDbg`로 교체.

## 2026-06-11 10:05 KST

### occ_pool_ext CUDA extension 재빌드

- `train_total.sh` 실행 시 `ImportError: cannot import name 'occ_pool_ext'` 발생 (`.so` 부재가 원인, circular import 아님).
- `projects/occ_plugin/ops/occ_pooling`에서 `setup.py build_ext --inplace`로 재빌드 후 import 정상 확인.

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

---

## 2026-06-16 KST — EfficientOCF eval(test) 경로 신규 구현 + mcls vs gate IOU 측정

**배경**: 현재 detector `efficientocf.py`(class EfficientOCF)에는 `forward_train`만 있고 `forward_test`/`simple_test`가 없어, standard test(`tools/test.py`)가 base `BEVDepth.forward_test`로 떨어져 `img_inputs must be a list, got NoneType`로 크래시. 학습 중 eval(IOU/VPQ)이 한 번도 안 돈 것도 같은 이유. 옛 `efficientocf_ori.py`의 eval은 `pts_bbox_head/height_head/flow_head` 기반인데 현재 모델엔 그 헤드가 없음(query_head/voxelizer 구조).

**구현** (`projects/occ_plugin/occupancy/detectors/efficientocf.py`, forward_train 앞):
- `forward_test`/`simple_test` 추가. **실제 추론 scoring**(debug-vis 3행)을 그대로 재사용: `_build_query_visualization_bundle(...)` → `selected_query_idx_q` (fg argmax≠bg → score≥`debug_query_score_threshold`(0.5) → 거리 NMS(3m) → top-k). score 가중치는 config의 visualization_cfg(iou=0,cls=1,cam=0) → score=cls_prob.
- 선택된 query의 present-frame center를 `self.voxelizer`(단일 sigma)로 렌더 → Z축 max로 BEV collapse → `segmentation_bev`(dense BEV movable GT, 옛 eval과 동일 타깃)와 2x2 confusion(free/movable) → 기존 `EfficientOCFDataset.evaluate`(`cm_to_ious`)로 IOU.
- 정렬(present frame index, transpose, flip)은 좌표 규약 차이로 첫 샘플 auto-calibration 또는 env `EOCF_EVAL_ALIGN="frame,transpose,fH,fW"`로 고정. occupancy threshold는 env `EOCF_EVAL_OCC_THR`(기본 0.5).
- 헬퍼: `_binary_occ_cm`, `_apply_bev_align`, `_calibrate_eval_bev_align`. 진단 print(`[simple_test][diag]`) 포함.

**측정 결과** (val 100 samples, seed 0, 동일 정렬 `EOCF_EVAL_ALIGN=2,1,0,0`):
- **mcls ep9: IOU_movable = 0.074**, **gate ep9: IOU_movable = 0.060** → **mcls 우위(+23% rel)**.
- 주의: score≥0.5 임계로 장면당 1~3개 query만 선택(GT ~2000 vox/frame) → 절대 IOU 매우 낮음(under-predict). 이는 faithful한 실제-추론 동작. present-frame movable BEV IOU이며 legacy future-IOU/VPQ 아님. traj run은 ckpt 없어 미측정.
- 첫 자동정렬판(frame 미고정)에선 mcls가 frame3, gate가 frame2로 갈려 gate가 높게 나왔으나 **frame 고정 후 mcls 우위로 정정**됨.

**평가 실행**: `run_eval.sh CFG CKPT GPUS` 또는 `tools/test.py ... --launcher pytorch --eval bbox`(분산 런처 필수 — non-dist면 evaluate 미호출). 로그/CSV: `work_dirs/_analysis/`.

---

## 2026-06-17 KST — train 중 eval 완전 제거 (흔적 코드 포함)

**배경**: 사용자 요청 — train 도중 eval(validation)을 절대 하지 않음. 인자 비활성이 아니라 흔적 코드까지 제거.

**사실 확인**: 실제 train 경로(`tools/train.py` → `custom_train_model` → `custom_train_detector`)의 eval 훅 등록부는 이미 주석 처리 상태였음 → 사실상 train 중 eval은 이미 안 돌고 있었음. `Saving checkpoint` 직후 멈춘 듯 보인 건 eval이 아니라 체크포인트 저장 I/O + dataloader 워커 재기동.

**변경**:
- `projects/occ_plugin/occupancy/apis/mmdet_train.py`: 주석 처리돼 있던 eval 훅 등록 블록(val_dataset/val_dataloader/OccDistEvalHook) 완전 삭제. eval 전용 import 제거(`EvalHook`, `OccDistEvalHook/OccEvalHook`, `custom_build_dataset`, 미사용 `build_dataset/replace_ImageToTensor`). → 이제 어떤 config든 train 중 eval 훅이 등록될 수 없음(전역 보장).
- `projects/configs/baselines/test_traj_mcls_occ.py`: `evaluation = dict(...)` 블록 삭제, eval 전용 `val_config`/`data['val']` 삭제, 미사용 `import copy` 삭제. (`test_config`/`test_pipeline`은 tools/test.py용이라 유지)

**영향**: train 중 eval 없음(전역). test.py를 통한 별도 평가는 영향 없음. 다른 실험 config들(test_traj.py, EfficientOCF_V1.1_1gpu.py 등)에 남아있는 `evaluation` dict는 이제 dead code지만 별도 실험 소유라 미수정.
