# traincal 정규화 통계 (depth+speed)

`eval_depth_speed.sh` / `EOCF_EVAL_TRAINCAL_*` 가 쓰는 **train split 고정 통계**를 모아둔 폴더.

## 파일 이름 규칙

```
<모델>_ep<에폭>_n<샘플수>.json
```

- 모델: `dim96` / `feat128` / `res704`
- 에폭: 통계를 뽑은 **checkpoint 의 에폭**
- 샘플수: train split 에서 forward 한 sample 수 (query 수가 아님)
- `_live` 접미사 = 수집 진행 중인 스냅샷(계속 덮어써짐). eval 에 쓰지 말 것
- `_smoke` 접미사 = 배선 확인용 소표본. eval 에 쓰지 말 것

## 목록

| 파일 | 모델 | ckpt | 샘플 | query 수 | depth mean/std | speed mean/std |
|---|---|---|---|---|---|---|
| `dim96_ep18_n4000.json` | dim96 | ep18 | 4,000 | 136,973 | 1.14901 / 0.27154 | 1.53707 / 0.69234 |
| `dim96_ep18_n23930.json` | dim96 | ep18 | 23,930 | 821,803 | 1.14906 / 0.27078 | 1.53795 / 0.69588 |
| `feat128_ep19_n20_smoke.json` | feat128 | ep19 | 20 | 678 | 1.12624 / 0.27036 | 1.60634 / 0.65443 |
| `feat128_ep19_n8000/12000/16000/20000.json` | feat128 | ep19 | 마일스톤 | — | 수렴 곡선 참조 | |
| **`feat128_ep19_n23900.json`** | feat128 | ep19 | **23,900 (전량)** | 845,190 | 1.120834 / 0.273475 | 1.489227 / 0.631784 |
| **`feat128_ep19_n4000.json`** ← eval 에 사용 | feat128 | ep19 | 4,000 | 137,343 | 1.121052 / 0.274257 | 1.487886 / 0.627778 |

> dim96 두 파일은 **다른 프로젝트**(`Autonomous_Driving_26_ikk_fi`)에서 뽑아온 것이다. checkpoint 필드 참조.
>
> ⚠ `feat128_ep19_live.json` 은 `../depth_std+past_speed_calibration/feat128_ep19_normalization_stats.json`
> 로의 **심볼릭 링크**다. 수집기의 출력 경로가 프로세스 env 로 고정돼 있어 실행 중에는 바꿀 수 없어서,
> 링크로 이 폴더에서도 실시간 값이 보이게 했다. 수집이 끝나면 최종본을 `feat128_ep19_n23930.json`
> 으로 옮기고 링크는 지운다. **사본을 만들어 두면 갱신이 멈춘 채 이름만 `live` 인 함정이 된다.**

## 🔴 checkpoint 종속 — 재사용 금지

`depth_std`(예측 depth 분포의 표준편차)와 `past_speed`(예측 과거 궤적 속력)는 **모델 출력**이다.
다른 checkpoint 로 평가하면 z-score 가 통째로 틀어지는데 **에러가 나지 않고 조용히 잘못된 결과**가 나온다.

실측 차이 (log1p):

| feature | dim96 ep18 | feat128 ep19 (3,900) | 차이 |
|---|---|---|---|
| depth mean | 1.14901 | 1.12112 | −0.028 (std 의 10%) |
| speed mean | 1.53707 | 1.48845 | −0.049 (std 의 8%) |
| speed std | 0.69234 | 0.62737 | −0.065 (−9%) |

→ dim96 통계를 feat128 에 쓰면 z 가 약 0.1 밀리고, α=0.5/0.25 기준 combined logit 이 ~0.07 어긋난다.

## 4,000 이면 수렴한다 — 실측 곡선

feat128 ep19 수집 중 마일스톤 보존본 (log1p population mean/std):

| n | depth mean | depth std | speed mean | speed std |
|---|---|---|---|---|
| 20 (smoke) | 1.126237 | 0.270362 | 1.606336 | 0.654429 |
| **4,000** | **1.121052** | **0.274257** | **1.487886** | **0.627778** |
| 8,000 | 1.120817 | 0.273197 | 1.490745 | 0.630674 |
| 12,000 | 1.121325 | 0.272879 | 1.490036 | 0.631191 |
| 16,000 | 1.120875 | 0.273243 | 1.487747 | 0.631695 |
| 20,000 | 1.120745 | 0.273485 | 1.489784 | 0.632748 |
| **23,900 (전량)** | **1.120834** | **0.273475** | **1.489227** | **0.631784** |

**수집 완료 2026-08-05 07:11 (rc=0). train 23,900 샘플 / foreground query 845,190개.**

> ✅ **4,000 과 전량(23,900)의 차이가 무시할 수준으로 확정됐다.**
>
> | | 4,000 | 23,900 | 차이 | std 대비 |
> |---|---|---|---|---|
> | depth mean | 1.121052 | 1.120834 | **−0.000218** | 0.08% |
> | depth std | 0.274257 | 0.273475 | −0.000782 | 0.29% |
> | speed mean | 1.487886 | 1.489227 | **+0.001341** | 0.21% |
> | speed std | 0.627778 | 0.631784 | +0.004006 | 0.63% |
>
> 전 구간(4,000~23,900)에서 depth mean 폭 0.0006, speed mean 폭 0.0030 — **z-score 에 실질 영향 없다.**
> `α 0.75/0.25` 기준 combined logit 변화는 0.001 미만이라 선택 집합이 바뀌지 않는다.
> → **4,000 으로 돌린 eval 결과를 그대로 쓸 수 있다.** 재실행 불필요.
> ⚠ 반면 **20 샘플 smoke 는 speed mean 이 1.606 으로 0.12 나 어긋난다.** 소표본을 eval 에 쓰면 안 된다.
> dim96 도 4,000 vs 23,930 이 셋째 자리까지 같았다(1.14901 vs 1.14906).
>
> → **4,000 스냅샷으로 eval 을 시작해도 된다.** 다만 전량(23,930)을 확보해 두면
> "4,000 으로 충분함"을 두 점이 아니라 곡선으로 보일 수 있어 논거가 단단해진다.

## 이어붙이기 가능

각 feature 에 `_acc: {n, sum, sumsq}` 누적합이 들어 있다. population mean/std 는 이 셋으로 복원되고
**병합은 그냥 덧셈**이다 → 중단하거나 나중에 표본을 늘려도 처음부터 다시 돌 필요가 없다.

```python
import json
a = json.load(open("feat128_ep19_n4000.json"))["features"]["depth_std"]["_acc"]
b = json.load(open("feat128_ep19_n23930.json"))["features"]["depth_std"]["_acc"]
n = a["n"] + b["n"]; s = a["sum"] + b["sum"]; s2 = a["sumsq"] + b["sumsq"]
mean = s / n; std = (s2 / n - mean ** 2) ** 0.5
```

## 새로 뽑는 법

```bash
# EP=<에폭> MAXS=<샘플수, 0=전체> EVERY=<스냅샷 주기>
EP=19 MAXS=0 EVERY=100 OUT=./traincal/feat128_ep19_live.json \
  ./collect_traincal_stats.sh
```

train split 을 **deterministic test pipeline** 으로 forward 한다(`..._traincal_collect.py` config).
GT 는 쓰지 않고, 스코프는 `pred_cls != bg` 인 모든 예측 foreground query 다.

## 쓰는 법

```bash
export EOCF_EVAL_TRAINCAL_STATS=./traincal/feat128_ep19_n4000.json
# ⚠ EOCF_EVAL_TRAINCAL_MEAN_*/STD_* 가 설정돼 있으면 **그쪽이 JSON 을 이긴다**. 반드시 unset 할 것.
```
