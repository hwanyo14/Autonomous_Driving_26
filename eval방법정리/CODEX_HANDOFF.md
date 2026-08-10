# Codex용 Depth+Speed train-calibrated eval 구현 가이드

## 목적

`20260802_034946`에서 사용한 Depth+Speed query scoring과 Asset IoU3D eval을 다른 코드 버전에도
동일한 방식으로 구현하기 위한 명세다.

대상은 2026-08-02 당시의 고정 policy 하나다.

```text
depth_std alpha = 0.5
past_speed alpha = 0.25
combined cutoff = 0.90
OCC thresholds = 0.8, 0.9
```

현재 `eval_query_rescue_traincal_depth_speed_full.sh`의 alpha 6×6 sweep은 대상이 아니다. 성능 결과값도
이 문서에 기록하지 않는다.

세부 수식과 평가 정의는 같은 디렉터리의 `EVAL_METHOD.md`를 함께 참고한다.

## 1. 구현할 코드 위치

대상 프로젝트에서 아래 기능이 이미 있는지 먼저 검색하고, 없는 부분만 최소 구현한다.

| 파일/심볼 | 구현 책임 |
|---|---|
| `collect_query_rescue_train_stats.sh` | train 4,000 calibration 실행과 normalization JSON 생성 |
| `tools/test.py::_configure_train_calibration_split` | train annotation을 deterministic test pipeline으로 평가 |
| `efficientocf.py::_eval_query_calibration_feature_values` | foreground query의 `depth_std`, `past_speed` 추출 |
| `efficientocf.py::_eval_query_calibration_feature_stats` | `log1p`, finite filter, `count/sum/sum_sq` 수집 |
| `efficientocf.py::_eval_query_traincal_pairs_analysis` | stats 검증, z-normalize, fixed scoring, query selection, OCC render |
| `occupancy/apis/test.py::_finalize_query_rescue` | population mean/std 저장과 validation summary 집계 |
| `eval_query_rescue_traincal_pairs_full.sh` | fixed policy와 eval 환경변수 설정 |
| `eval_query_rescue_traincal_depth_speed_full.sh` | `PAIR=depth_speed` wrapper |

파일 전체를 다른 버전으로 교체하지 말고 함수와 환경변수 단위로 적용한다. 이 기능은 eval-only이므로
학습 loss, model head, checkpoint parameter, model config default는 변경하지 않는다.

## 2. Train 4,000 normalization 구현

### 2.1 데이터 범위

- train annotation을 사용한다.
- random training augmentation 대신 deterministic test pipeline을 사용한다.
- single GPU, non-shuffle 순서에서 앞쪽 4,000 sample을 처리한다.
- 모집단은 `predicted class != background`인 모든 predicted foreground query다.
- score window로 query를 제한하지 않는다.
- GT, Asset label, Hungarian matching, oracle 정보를 사용하지 않는다.

### 2.2 Feature 정의

현재 프레임 query의 depth-bin 확률 `p_d`와 실제 depth-bin 값 `depth_d`로 `depth_std`를 계산한다.

```text
mu_depth = sum_d p_d * depth_d
depth_std = sqrt(sum_d p_d * (depth_d - mu_depth)^2)
```

유효한 과거 query center의 인접 XY 이동으로 `past_speed`를 계산한다. 기본 frame 간격은 0.5초다.

```text
speed_i = ||xy_i - xy_(i-1)||_2 / 0.5
past_speed = mean(speed_i over valid intervals)
```

### 2.3 Population statistics

두 feature 모두 다음 변환을 적용한다.

```text
x = log1p(max(raw_value, 0))
```

전체 유효 foreground-query 값을 합쳐 population mean/std를 계산한다.

```text
mean = sum(x) / N
std = sqrt(max(0, sum(x^2) / N - mean^2))
```

`N-1`로 나누는 sample std를 사용하지 않는다.

### 2.4 Normalization JSON 계약

```text
schema_version = 1
mode = post_training_calibration
split = train
sample_count = 4000
scope = all_predicted_foreground_queries
uses_ground_truth = false
features = exactly {depth_std, past_speed}
transformed_log1p = true
good_direction = low
feature count > 0
feature std > 0
stats checkpoint = eval checkpoint
```

Validation에서는 이 train mean/std를 고정하며 다시 normalization 통계를 계산하지 않는다.

## 3. Validation scoring 구현

각 foreground query에 대해 다음 z-score를 계산한다.

```text
z_depth = clip((log1p(depth_std) - train_mean_depth) / train_std_depth, -3, 3)
z_speed = clip((log1p(past_speed) - train_mean_speed) / train_std_speed, -3, 3)
```

원래 query score를 `score`라고 할 때 combined logit은 다음과 같다.

```text
base_logit = logit(clip(score, 1e-6, 1 - 1e-6))
combined_logit = base_logit - 0.5*z_depth - 0.25*z_speed
```

Query 선택 조건은 다음과 같다.

```text
predicted foreground
AND z_depth, z_speed are finite
AND combined_logit >= logit(0.90)
```

선택 query는 combined logit 내림차순으로 정렬하고, `fg_score_topk > 0`이면 마지막에 top-k를 적용한다.

이 방식은 global reselection이다. 기존 `score >= 0.90` query를 강제로 보존하지 않는다.

## 4. Fixed policy와 runner 구현

Detector가 아래 policy를 인식해야 한다.

```python
(
    "depth_std_a0.5+past_speed_a0.25__global090",
    (("depth_std", 0.5), ("past_speed", 0.25)),
)
```

`depth_speed` runner 분기는 논리적으로 다음과 같아야 한다.

```bash
PAIR_POLICY=depth_std_a0.5+past_speed_a0.25__global090
PAIR_STATS="$DEPTH_SPEED_STATS"
EXPECTED_FEATURES=depth_std,past_speed
PAIR_LABEL=depth_std+past_speed
```

현재 alpha grid 기능을 유지해야 한다면 삭제하지 말고, historical fixed-policy wrapper 또는 policy
override를 분리한다. 이 실행에서는 `DEPTH_SPEED_ALPHAS` 이중 loop를 사용하지 않는다.

## 5. Occupancy render와 metric 구현

다음 eval 조건을 고정한다.

```text
EOCF_EVAL_MODE = 2
EOCF_EVAL_FG_THR = 0.9
EOCF_EVAL_NMS_RADIUS = 0
EOCF_EVAL_WEIGHT_MODE = ones
EOCF_EVAL_TRAJ_REFINE = 1
EOCF_EVAL_TRAINCAL_COMBINED_CUTOFF = 0.9
EOCF_EVAL_TRAINCAL_OCC_THRESHOLDS = 0.8,0.9
visualization = off
query-rescue cache = off
Asset metrics only = on
AABB IoU = off
Recall3D = off
```

선택 query의 full-resolution scene probability를 한 번 렌더하고 같은 probability에 OCC 0.8과 0.9를
각각 적용한다. OCC threshold마다 model inference를 반복하지 않는다.

평가 구간은 현재 1프레임, 미래 4프레임, 현재+미래 5프레임이다. Asset IoU3D present/future/combined의
macro와 micro를 집계한다. 대표 정렬 기준은 future macro다.

`score >= 0.90` baseline은 같은 forward에서 내부 비교용으로만 계산하고 최종 summary 행에서는 제외한다.

## 6. 검증 순서

### 6.1 정적 검사

```bash
bash -n collect_query_rescue_train_stats.sh
bash -n eval_query_rescue_traincal_pairs_full.sh
bash -n eval_query_rescue_traincal_depth_speed_full.sh
python -m py_compile tools/test.py
python -m py_compile projects/occ_plugin/occupancy/apis/test.py
python -m py_compile projects/occ_plugin/occupancy/detectors/efficientocf.py
git diff --check
```

### 6.2 Calibration smoke

`MAX_SAMPLES=1`로 다음을 확인한다.

- JSON이 생성되는가
- feature가 정확히 `depth_std`, `past_speed`인가
- split/scope/GT metadata가 계약과 같은가
- mean/std/count가 finite이고 std가 양수인가
- calibration 종료 후 불필요한 metric 계산을 하지 않는가

Smoke JSON은 full eval에 사용하지 않는다. 이후 `MAX_SAMPLES=4000`으로 정식 stats를 생성한다.

### 6.3 Eval smoke

`MAX_SAMPLES=1`과 새 `RUN_TAG`로 다음을 확인한다.

- active base policy가 정확히 하나인가
- output policy가 fixed Depth+Speed × OCC 2개인가
- baseline 행이 summary에서 제외되는가
- eval mode가 `present_future`, metric frame이 5인가
- alpha 6×6 policy가 섞이지 않았는가
- visualization/cache/AABB IoU/Recall3D가 비활성화됐는가

### 6.4 Full eval

Smoke가 통과한 다음 실행한다.

```bash
GPU_ID=<gpu-id> \
MAX_SAMPLES=5119 \
DEPTH_SPEED_STATS=<train-4000-normalization-json> \
RUN_TAG=<new-run-tag> \
bash eval_query_rescue_traincal_depth_speed_full.sh
```

`MAX_SAMPLES`는 처리 상한이며 실제 처리 sample 수는 summary metadata로 확인한다.

## 7. 구현 완료 기준

- GT 없이 deterministic train 4,000 subset에서 stats 생성
- validation 재정규화 없음
- fixed alpha `0.5/0.25` policy 하나만 활성화
- combined cutoff `0.90`, OCC `0.8/0.9`
- present 1 + future 4 평가
- summary CSV/JSON과 live log 생성
- 정적 검사와 calibration/eval smoke 통과
- opt-in이 꺼지면 기존 학습 및 일반 eval 동작이 바뀌지 않음

## 8. Codex에 입력할 요청문

```text
이 프로젝트의 AGENTS.md와 제공된 EVAL_METHOD.md, CODEX_HANDOFF.md를 먼저 읽어라.

목표는 `20260802_034946`에서 사용한 고정 Depth+Speed train-calibrated eval 방법을 현재 코드에
최소 수정으로 구현하는 것이다. 현재 alpha 6x6 sweep을 실행하는 작업이 아니다.

구현 요구사항:
- deterministic train subset 4,000 sample의 모든 predicted foreground query에서 depth_std와
  past_speed를 수집한다.
- 두 feature에 log1p를 적용하고 population mean/std를 normalization_stats.json에 저장한다.
- GT, Asset label, Hungarian/oracle 결과를 normalization이나 query selection에 사용하지 않는다.
- validation에서는 train stats를 고정하며 재정규화하지 않는다.
- combined logit은 logit(score) - 0.5*z_depth - 0.25*z_speed다.
- global cutoff 0.90으로 query를 다시 선택하며 기존 score>=0.90 query를 강제 보존하지 않는다.
- weight=ones, NMS=0, trajectory refine on, present 1 + future 4 조건을 사용한다.
- probability는 selection당 한 번 렌더하고 OCC 0.8/0.9를 후처리한다.
- Asset IoU3D만 집계하고 AABB IoU와 Recall3D는 제외한다.
- alpha 6x6 기능을 유지해야 한다면 fixed-policy 실행 경로를 별도로 분리한다.

작업 순서:
1. 문서의 코드 지도에 적힌 함수와 환경변수가 현재 코드에 있는지 확인한다.
2. 없는 부분과 현재 구현의 차이를 짧게 보고한다.
3. 함수/환경변수 단위의 최소 수정으로 fixed policy 실행 경로를 구현한다.
4. shell syntax, Python compile, git diff --check를 수행한다.
5. MAX_SAMPLES=1 calibration/eval smoke를 수행한다.
6. policy 이름과 개수, normalization metadata 계약을 확인한다.
7. smoke 통과 후 full eval 명령을 제시한다.

구현이나 로직이 불명확하면 임의로 가정하지 말고 질문해라.
기존 결과 디렉터리와 normalization JSON을 덮어쓰지 마라.
성능 결과값을 방법 문서에 추가하지 마라.
최종 답변에는 수정 파일, 구현한 식과 조건, 검증 결과, 미실행 항목을 구분해 적어라.
```
