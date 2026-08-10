# EVAL_SPEC — filter repo 평가 체계 이식 가이드

이 문서는 `Autonomous_Driving_26_filter` 의 **평가(eval) 체계**를 다른 repo(특히 7월 스냅샷에서
분기한 `ablation_*` repo)로 옮기기 위한 명세다. 대상은 두 덩어리다.

1. **metric 체계** — 한 번의 eval 런에서 `future` / `present` 를 **동시에** 뽑는 구조
2. **traincal** — depth + 운동항 기반 train-calibrated 재선택

작성 2026-08-10. 근거는 전부 `projects/occ_plugin/occupancy/detectors/efficientocf.py` 와
`projects/occ_plugin/occupancy/apis/test.py` 의 실제 코드다.

---

# 0. 먼저 — 새 repo 에서 eval 이 돌게 만들기

7월 스냅샷 repo 는 아래 4개를 고치지 않으면 **첫 샘플도 못 돌고 죽는다.** 순서대로 확인할 것.

### ① `ImportError: cannot import name 'occ_pool_ext'`
CUDA 확장이 repo 마다 따로 빌드돼 있어야 한다. ops 소스가 filter repo 와 같으면 **빌드하지 말고 복사**한다.

```bash
diff -rq <filter>/projects/occ_plugin/ops/occ_pooling <new>/projects/occ_plugin/ops/occ_pooling   # 먼저 확인
cp <filter>/projects/occ_plugin/ops/occ_pooling/occ_pool_ext.cpython-310-x86_64-linux-gnu.so \
   <new>/projects/occ_plugin/ops/occ_pooling/
```

### ② `FileNotFoundError: ./data/occ_dt/scene_*/dt/*.npz`
`occ_dt` 데이터셋은 이 머신에 없다. `use_query_dt_loss=False` 라 실제로 쓰이지도 않는다.
**config 의 `test_pipeline` 구간만** 고친다 (train_pipeline 은 건드리지 말 것):

- `load_occ_dt=True` → `False`
- `Collect3D` 의 `keys` 에서 `'occ_dt'` 제거

### ③ `AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'`
mmcv 1.x ↔ PyTorch 2.7 비호환. filter repo 의 셔틀을 이식한다.

`apis/mmdet_train.py`:
```python
from mmcv.parallel.scatter_gather import scatter_kwargs

class _Torch27MMDistributedDataParallel(MMDistributedDataParallel):
    """Keep MMCV 1.x DataContainer scatter compatible with PyTorch 2.7."""
    def scatter(self, inputs, kwargs, device_ids):
        devices = [torch.device("cuda", d) if isinstance(d, int) else d for d in device_ids]
        return scatter_kwargs(inputs, kwargs, devices, dim=self.dim)

    def _run_ddp_forward(self, *inputs, **kwargs):
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            inputs, kwargs = inputs[0], kwargs[0]
        if self._use_python_reducer:
            return self.module(*inputs, **kwargs)
        with self._inside_ddp_forward():
            return self.module(*inputs, **kwargs)
```
`tools/test.py`: `MMDistributedDataParallel(` → `_Torch27MMDistributedDataParallel(`

### ④ 백그라운드 런이 `SignalException: got signal: 15` 로 죽음
`nohup` 은 **SIGHUP 만** 막는다. 호출 셸이 끝나면 프로세스 그룹째 SIGTERM 을 받는다.

```bash
setsid <cmd> > log 2>&1 < /dev/null &     # nohup 아님
```

---

# 1. metric 체계 — future / present 동시 산출

## 1-1. 핵심 원리

**`EOCF_EVAL_MODE` 는 시각화 기준 프레임만 바꾸고, metric 은 항상 둘 다 계산한다.**

현재 프레임 예측 `pred_occ3d` 는 mode 와 무관하게 **이미 렌더돼 있다.** 그래서 present metric 은
GT 만 현재 프레임으로 잘라 IoU 를 한 번 더 재면 되고 — **추가 forward 비용이 0 이다.**

```python
pred_occ3d = (pred_occ[0, 0] > thr)          # [D,H,W] = [Z,Y,X]  ← 현재 프레임 예측
pred_bev   = pred_occ3d.amax(dim=0)          # [H,W]
pred_txyz  = (pred_occ_eval[:, 0] > thr).permute(0, 3, 2, 1)   # [F,X,Y,Z] ← 미래 프레임 예측
```

> ⚠ **구버전(7월) repo 는 여기서 갈린다.** 구버전은
> ```python
> if eval_mode == "present":
>     pred_txyz = pred_occ3d.permute(2, 1, 0).unsqueeze(0)   # metric 기준까지 바꿔버린다
> ```
> 이고, GT 도 `_eval_mode_slice(gt_asset_src)` 로 **한쪽만 잘라** asset 지표를 하나만 낸다.
> → 한 런에 한쪽만 나오므로 future/present 를 다 보려면 런을 두 번 돌려야 한다.
> **이식의 목표는 이 분기를 없애고 두 세트를 항상 계산하게 만드는 것이다.**

## 1-2. detector 쪽 (`efficientocf.py`, `simple_test`)

두 세트를 각각 계산해 **둘 다** 반환한다.

```python
pred_bev_t = pred_txyz.permute(0, 3, 2, 1).any(dim=1).contiguous()   # [F,Y,X]
cm_asset_future  = np.zeros((2, 2), dtype=np.int64)
cm_asset_present = np.zeros((2, 2), dtype=np.int64)
iou_3d_asset_future = iou_3d_asset_present = float("nan")
comps_future = comps_present = dict(tp=0.0, fp=0.0, fn=0.0)

gt_asset_src = self._eval_load_asset_gt(img_metas)      # [T,Z,Y,X]
_, transpose, fh, fw, fz = align3d                       # EOCF_EVAL_3D_ALIGN

# --- future: 미래 n_future 프레임 ---
asset3d_fut  = self._apply_3d_align_sequence(self._eval_future_tail(gt_asset_src), transpose, fh, fw, fz)
asset_bev_f  = asset3d_fut.any(dim=1).contiguous()
t_f = min(int(pred_bev_t.shape[0]), int(asset_bev_f.shape[0]))
cm_asset_future = self._binary_occ_cm(pred_bev_t[:t_f], asset_bev_f[:t_f])
pred_zyx_f = pred_txyz[:t_f].permute(0, 3, 2, 1).contiguous()
iou_3d_asset_future, _, comps_future = self._iou_recall_3d(pred_zyx_f, asset3d_fut[:t_f])

# --- present: 현재 1프레임 (pred_occ3d 재사용 → 추가 비용 없음) ---
asset3d_pres = self._apply_3d_align_sequence(
    gt_asset_src[present_idx:present_idx + 1], transpose, fh, fw, fz)   # [1,Z,Y,X]
asset_bev_p  = asset3d_pres.any(dim=1).contiguous()                     # [1,Y,X]
cm_asset_present = self._binary_occ_cm(pred_bev.unsqueeze(0), asset_bev_p)
iou_3d_asset_present, _, comps_present = self._iou_recall_3d(pred_occ3d.unsqueeze(0), asset3d_pres)

return dict(
    hist_for_iou_asset_present=cm_asset_present,
    hist_for_iou_asset_future=cm_asset_future,
    iou_3d_asset_present=iou_3d_asset_present,
    iou_3d_asset_future=iou_3d_asset_future,
    iou_3d_asset_present_comps=comps_present,
    iou_3d_asset_future_comps=comps_future,
)
```

**필요한 헬퍼** (7월 repo 에도 전부 있음):
`_eval_load_asset_gt` · `_eval_future_tail` · `_apply_3d_align_sequence` ·
`_binary_occ_cm` · `_iou_recall_3d` · `present_idx`(`feat_out[23]`)

## 1-3. apis/test.py 쪽

두 세트를 각각 누적하고 한 줄에 같이 출력한다.

```python
def _acc(result, cm_p, cm_f, i3_p, i3_f, comps_p, comps_f):
    cm_p.append(result['hist_for_iou_asset_present'])
    cm_f.append(result['hist_for_iou_asset_future'])
    ...

# 출력 형식 (running / all-ranks 공통)
"[eval][{n}] IoU3d(asset)_future={:.4f} IoU2d(asset)_future={:.4f} "
"IoU3d(asset)_present={:.4f} IoU2d(asset)_present={:.4f}"
```

분산 학습이므로 `collect_results_cpu` 로 rank 별 결과를 모은 뒤 all_reduce 한다.

## 1-4. asset 외 지표는 제거했다 (2026-07-30)

filter repo 는 `nusocc` · `bbox_aabb` · `bbox_rot` · `Recall3d` 계열을 **전부 제거**했다.
GT 로드(`_eval_load_bbox_gt_v2`, `LoadOccupancy`)도 함께 사라져 **eval CPU 비용이 크게 줄었다.**
7월 repo 는 이것들이 아직 살아 있다 — 이식할 때 asset 만 추가할지, 나머지를 지울지 결정할 것.

- `apis/test.py` 크기 비교: filter **389줄** vs 7월 스냅샷 **571줄**

## 1-5. live 로그

`EOCF_EVAL_VIS_DIR` 이 설정돼 있으면 매 `EOCF_EVAL_METRIC_EVERY` 샘플마다
`<VIS_DIR>/eval_metrics_live.log` 에 한 줄씩 append 된다. 진행 중에도 값을 볼 수 있다.

> ⚠ `tools/dist_test.sh` 가 콘솔을 `logs/<config>/eval/<timestamp>.log` 로 리다이렉트하므로
> stdout 을 파일로 받아도 지표가 안 들어온다. **지표는 live 로그에서 읽을 것.**

---

# 2. traincal — train-calibrated 재선택

## 2-1. 수식

```
s_eff = sigmoid( logit(s) − α_d · z_depth − α_s · z_motion )
keep if s_eff ≥ cutoff              # cutoff 가 곧 fg 임계값 역할
z = clip( (log1p(x) − train_mean) / train_std , −3, 3 )
```

- **전역 재선택(global reselection)** 이다. baseline-preserving rescue 가 **아니다** —
  원래 `s ≥ cutoff` 였던 query 도 보정 후 컷 아래로 내려가면 **탈락**하고, 아래였던 query 가 올라올 수 있다.
- 렌더 가중치는 **보정 전 score** 를 그대로 쓴다(선택 변경과 렌더 변경을 섞지 않는다).
- z 는 **train 통계로 고정 정규화**한다. validation 분포로 재정규화하지 않는다.
- 이 경로는 bundle 내부의 **distance-NMS / topk 를 우회**한다. 이후 combined logit 내림차순 정렬
  → `fg_score_topk` 만 적용된다.
- `pred_cls_q == query_bg_class` 인 query 는 제외된다.

## 2-2. feature 두 개

| feature | 정의 | 성격 |
|---|---|---|
| **`depth_std`** | present 프레임 depth-bin 분포의 표준편차<br>`mu=Σ p_d·depth_d`, `std=sqrt(Σ p_d (depth_d−mu)²)` | **모델이 뱉은 불확실성** — 진짜 품질 신호 |
| **`traj_dev`** ✅ | `mean_t ‖c_{t+1} − 2c_t + c_{t−1}‖` = **등속 모델 잔차** | 속도 변화 + **방향 변화**를 모두 잡는다 |
| `past_speed` ❌ | `mean ‖Δxy‖/dt` (dt=0.5s, present 까지) | 속력 **크기** — "빠르면 감점" = 난이도 prior. **폐기** |
| `speed_std` ❌ | 구간 속력의 std | 속력 변화만. 방향을 못 잡아 `traj_dev` 하위호환. **폐기** |

depth bin 중심은 학습/투영과 **동일 규칙**을 쓴다 (`utils_query_projection` 의 `depth_vals_d`):
```python
bin_size     = (d_hi - d_lo) / d_bins
depth_vals_d = (arange(d_bins) + 0.5) * bin_size + d_lo
```
`d_lo/d_hi` 는 `query_inst_depth_range_mode` 가 `custom` 이면 `query_inst_depth_min/max`,
아니면 `img_view_transformer.grid_config["dbound"]`.

> 🔴 **`depth_std` 는 depth head 가 학습된 모델에서만 의미가 있다.**
> `query_depth_loss_weight=0.0` 인 ablation(예: `_center`)에서는 depth head 가 학습된 적이 없어
> 출력이 무의미하다 → **그런 repo 에는 traincal 을 적용하면 안 된다.**

## 2-3. 통계 수집

```bash
EOCF_EVAL_TRAINCAL_COLLECT=1  EOCF_EVAL_TRAINCAL_COLLECT_EVERY=<N>  ...  # train split 으로 실행
```
결과 JSON 형식:
```json
{
  "schema_version": 1,
  "mode": "post_training_calibration",
  "checkpoint": "...", "split": "train", "sample_count": 3800,
  "scope": "all_predicted_foreground_queries",
  "uses_ground_truth": false,
  "features": {
    "depth_std": {"mean": 1.12142, "std": 0.27402, "count": 134498,
                  "transformed_log1p": true, "good_direction": "low",
                  "_acc": {"n": ..., "sum": ..., "sumsq": ...}},
    "traj_dev":  {"mean": 1.05696, "std": 0.61261, ...},
    "past_speed": {...}, "speed_std": {...}
  },
  "corr_with_depth_std": {"past_speed": 0.241, "traj_dev": 0.252, "speed_std": 0.164}
}
```

- `uses_ground_truth: false` — GT 를 전혀 안 쓴다. 추론 전용 보정이다.
- **표본 4,000 이면 충분**하다 (n4300 vs n23900 에서 future 소수 4자리 동일).
- 🔴 **통계는 checkpoint 마다 다시 뽑아야 한다.** 안 뽑으면 에러 없이 **조용히 틀어진다.**
- `corr_with_depth_std` 가 0.16~0.25 로 낮다 → 운동항이 depth 와 **독립적인 정보**를 준다.

## 2-4. env 인자

| env | 뜻 |
|---|---|
| `EOCF_EVAL_TRAINCAL_STATS` | 통계 JSON 경로 |
| `EOCF_EVAL_TRAINCAL_ALPHA_DEPTH` | `α_d` |
| `EOCF_EVAL_TRAINCAL_ALPHA_SPEED` | `α_s` (운동항 계수) |
| `EOCF_EVAL_TRAINCAL_CUTOFF` | 컷 (= fg 임계값 역할) |
| `EOCF_EVAL_TRAINCAL_FEAT2` | 둘째 feature: `traj_dev` (구 이름 `traj_acc` 도 받음) |
| `EOCF_EVAL_TRAINCAL_DT` | 과거 프레임 간격, 기본 0.5s |
| `EOCF_EVAL_TRAINCAL_COLLECT` / `_COLLECT_EVERY` / `_COLLECT_CKPT` | 통계 수집 모드 |
| `EOCF_EVAL_TRAINCAL_MEAN_DEPTH` / `_STD_DEPTH` / `_MEAN_SPEED` / `_STD_SPEED` | JSON 대신 상수 직접 지정 (**env 가 JSON 을 이긴다** — 옛 값이 남아 있으면 반드시 `unset`) |

## 2-5. 실패 시 동작

통계가 없거나 feature 계산이 실패하면 **조용히 baseline 으로 되돌아간다**(`return fallback`).
에러가 안 나므로, 로그에 아래 줄이 찍히는지 반드시 확인할 것:

```
[eval] traincal depth+traj_dev selection ON (a_depth=..., a_speed=..., cutoff=...)
```

---

# 3. 공통 eval 인자

| env | 뜻 | 스윕 시 값 |
|---|---|---|
| `EOCF_EVAL_MODE` | 1=future / 0=present. **metric 아님, 시각화 기준 프레임** | 1 |
| `EOCF_EVAL_FG_THR` | fg(query 선택) 임계값 | 스윕 대상 |
| `EOCF_EVAL_OCC_THR` | occ 점유 임계값 (Gaussian mixture 이진화) | 스윕 대상 |
| `EOCF_EVAL_NMS_RADIUS` | distance NMS 반경(m). **0=off** | 0 |
| `EOCF_EVAL_WEIGHT_MODE` | 렌더 가중치. `ones` = 1.0 고정 | ones |
| `EOCF_EVAL_TRAJ_REFINE` | traj xy-refine head 를 추론에도 적용 | 1 |
| `EOCF_EVAL_MAX_SAMPLES` | N 샘플 후 조기 종료. 0=전체(5119) | 800 / 0 |
| `EOCF_EVAL_METRIC_EVERY` | live 로그 주기 | 48 |
| `EOCF_EVAL_VIS` | 시각화 | 🔴 **스윕 시 반드시 0** |
| `EOCF_EVAL_FG_TRAJ_ACC_MAX` | TRAJ_CUT. **폐기된 방법 — 항상 0** | 0 |
| `EOCF_EVAL_3D_ALIGN` | GT↔pred 3D 정렬. `auto` 또는 `t,tr,fh,fw,fz` | 미설정(0,1,0,0,0) |

**fg score 가중치** (config 의 `model_cfg`):
```
score = (w_iou·iou_q + w_cls·cls_prob_q + w_cam·cam_attn_q·valid) / (w_iou + w_cls + w_cam·valid)
```
- `iou_q` 는 eval 에서 항상 0 이고 **GT 기반이라 절대 켜면 안 된다** → `fg_score_iou_weight=0.0`
- cls 단독으로 쓰려면 `cls_weight=1.0`, 나머지 0

---

# 4. 스윕 인프라 패턴

`work_dirs/_eval_scripts/` 에 세 조각으로 나눈다.

| 파일 | 역할 |
|---|---|
| `<x>_run.sh` | 단일 eval 런. env 를 받아 `dist_test.sh` 호출, TAG 로 결과 경로 결정 |
| `<x>_worker.sh` | 상시 워커. **큐 파일을 매 순회 다시 읽고**, 원자적 락(`mkdir`)으로 칸을 집는다 |
| `tasks_<x>.txt` | 큐. 한 줄 = 한 칸 |

**큐를 파일로 두는 이유**: 스크립트에 하드코딩하면 워커를 재시작할 때마다 남은 작업을 잃는다.
파일이면 돌리는 중에 줄을 추가/삭제해도 이어서 집어간다.

## 🔴 워커에서 실제로 터진 버그 4개 (같은 실수 반복 금지)

### ① `set -o pipefail` + `grep -q` → 판정이 뒤집힌다
```bash
# ❌ 절대 이렇게 쓰지 말 것
f() { pgrep -f X | while read p; do ... && echo x; done | grep -q x; }
```
`grep -q` 가 첫 매치에서 빠져나가면 앞단이 **SIGPIPE(141)** 로 죽고, `pipefail` 이 141 을
파이프라인 결과로 삼아 **true 가 false 가 된다.** 대화형 셸(pipefail 없음)에서 테스트하면 통과해서
더 위험하다. → **명령치환 안에서 개수만 세고 판정은 `[ ]` 로 한다.**
```bash
# ✅
f() { local n; n=$(ps -eo args= | grep -c "[X]pattern" || true); [ "${n:-0}" -gt 0 ]; }
```

### ② 부분 완료를 '완료'로 오인
중도 사망한 런도 `[eval][48]` 같은 줄을 남긴다. **목표 샘플수와 정확히 일치하는 줄**을 요구할 것.
```bash
done_p () { local want=$2; [ "$want" = 0 ] && want=5119      # MAXS=0 은 전체 → 마커가 [eval][5119]
            grep -qa "\[eval\]\[$want\]" "$SW/$1.out"; }
```
실제로 이 버그로 48샘플 값(0.0954)이 완주값(0.1060) 대신 기록될 뻔했다.

### ③ 실행 중인 스크립트를 수정해도 반영되지 않는다
bash 는 `while : ; do ... done` 을 **메모리에 파싱해두고** 돈다. 파일을 고쳐도 기존 프로세스는
옛 코드로 계속 간다. → **현재 런이 끝나는 쿨다운 창에 워커를 죽여 재기동**시킬 것.
(모르고 두면 새로 추가한 인자를 못 읽고 작업을 **조용히 건너뛴다.**)

### ④ `pkill -f <패턴>` 이 자기 셸을 죽인다
명령줄에 그 문자열이 들어 있으면 자기 자신이 매치된다(exit 144). → **PID 로만 죽일 것.**

## 자원 규칙

- **896×1600(feat128 급) eval 은 동시에 1개만.** VRAM 각 ~23GB 로 용량 문제는 아니지만,
  2개를 동시에 돌렸을 때 1초 차로 `CUDA error: unspecified launch failure` 동반 사망한 적이 있다.
- 256×704(res704 급)는 ~8GB 로 2개 동시 안전.
- 런 사이 **3분 쿨다운**.
- `USE_MPS=0` — MPS 는 컨텍스트 공유라 한쪽이 죽으면 다른 쪽도 같이 죽는다.
- 동시 실행 시 PORT 를 잡마다 다르게 줄 것.

---

# 5. 결과 해석 규칙

- **@800 은 후보 선별 전용.** 순위 예측에 쓰면 안 된다 — 지금까지 **5회** 뒤집혔다
  (ep19/ep20 · α0.5/0.1 vs α1.0/0.2 · α_s=0 판정 · res704 1등→7등 · feat128 1등→3등).
- **@800→@5119 편향은 모델마다 다르다.** feat128 −1.9% / res704 −4.4% / `_center` −3.7%.
  한 모델의 편향을 다른 모델에 환산하면 안 된다.
- **@5119 분해능 ≈ 0.0004** (1/5119). 이보다 작은 avg 차이로 순위를 매기지 말 것.
- **@5119 는 @800 보다 높은 α_s 를 선호한다** (feat128 0.2→0.3, res704 0.3→0.4).
  @800 으로 α_s 를 정하면 과소평가한다.
- 동점이면 **낮고 단순한 값**을 채택한다.
