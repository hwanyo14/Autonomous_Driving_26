# Depth+Speed train-calibrated eval 방법 (2026-08-02 버전)

## 문서 범위

이 문서는 아래 실행 디렉터리를 만든 **당시 평가 방법만** 기록한다.

```text
work_dirs/query_rescue_traincal_pairs_full/depth_speed/20260802_034946
```

IoU, delta, TP/FP/FN, win/loss/tie, 선택 query 수 등의 **평가 결과값은 기록하지 않는다**.

> 주의: 현재 `eval_query_rescue_traincal_depth_speed_full.sh`는 2026-08-03에
> Depth/Speed alpha 6×6 sweep으로 변경되었다. 현재 파일을 그대로 실행하면 이 문서의
> `20260802_034946` 평가와 정책 수 및 방법이 달라진다. 이 문서는 변경 전의 고정
> `depth_std=0.5`, `past_speed=0.25` 정책을 기준으로 한다.

## 1. 고정 입력

당시 runner의 기본 입력은 다음과 같았다.

| 항목 | 값 |
|---|---|
| config | `projects/configs/baselines/full_attn_cover_pyr_aabb_dice3d_new_asset_all_filter.py` |
| checkpoint | `epoch_18_lss_only.pth` |
| calibration split | train |
| calibration sample 상한 | 4,000 |
| eval split | validation |
| eval sample 상한 | 5,119 |
| GPU 구성 | 단일 GPU (`GPUS=1`) |
| train normalization 파일 | `work_dirs/query_rescue_train_calibration/epoch_18_lss_only/train_4000/depth_std+past_speed/20260801_224454/normalization_stats.json` |

Normalization 파일과 validation eval은 반드시 동일 checkpoint를 사용해야 한다.

## 2. Train 4,000개 normalization 수집 방법

### 2.1 데이터 구성

`collect_query_rescue_train_stats.sh`를 다음 의도로 실행했다.

```bash
FEATURES=depth_std,past_speed \
MAX_SAMPLES=4000 \
bash collect_query_rescue_train_stats.sh
```

수집 시 train annotation을 사용하되, 학습용 random augmentation이 아니라 validation과 같은
deterministic test pipeline으로 inference했다. 불필요한 GT instance/AABB loader와 GT collect key는
제외했다.

통계 모집단은 각 sample의 `predicted class != background`인 **모든 예측 foreground query**다.
score `[0.70, 0.90)` 같은 rescue window만 사용한 것이 아니며, GT, Asset label, Hungarian matching,
oracle 결과는 normalization에 사용하지 않았다.

### 2.2 `depth_std` 정의

현재 프레임의 query별 depth-bin 확률을 합이 1이 되도록 정규화한 뒤, 실제 depth-bin 값으로
기댓값과 표준편차를 계산했다.

```text
mu_depth = sum_d p_d * depth_d
depth_std = sqrt(sum_d p_d * (depth_d - mu_depth)^2)
```

즉 `depth_std`가 작을수록 현재 query의 depth 분포가 더 집중되어 있다고 보는 prediction-only
feature다.

### 2.3 `past_speed` 정의

present ego frame 기준으로 정렬된 query center의 유효한 과거 구간을 사용했다. 기본 frame 간격은
`0.5 s`이며, 각 인접 구간의 XY 속력 크기를 구한 뒤 유효 구간 평균을 사용했다.

```text
speed_i = ||xy_i - xy_(i-1)||_2 / 0.5
past_speed = mean(speed_i over valid intervals)
```

즉 방향을 상쇄한 평균 velocity가 아니라, 각 구간 **속력 크기의 평균**이다.

### 2.4 변환과 population 통계

두 feature 모두 음수를 0으로 제한한 뒤 `log1p` 변환했다.

```text
x_transformed = log(1 + max(x_raw, 0))
```

Train 4,000 sample에서 얻은 모든 유효 foreground-query 값을 합쳐 다음 population 통계를 저장했다.

```text
mean = sum(x_transformed) / N
std  = sqrt(max(0, sum(x_transformed^2) / N - mean^2))
```

여기서 `std`는 `N-1`로 나누는 sample standard deviation이 아니라 **population standard
deviation**이다. 실제 평균·표준편차 숫자는 이 문서에 복사하지 않고 위
`normalization_stats.json`을 source of truth로 사용한다.

저장 JSON은 최소한 다음 metadata를 만족해야 했다.

- `split == "train"`
- `sample_count == 4000`
- `scope == "all_predicted_foreground_queries"`
- `uses_ground_truth == false`
- checkpoint real path가 eval checkpoint와 동일
- feature 집합이 정확히 `depth_std`, `past_speed`
- 두 feature 모두 `transformed_log1p == true`, `good_direction == "low"`
- count가 양수이고 population std가 0보다 큼

## 3. Validation query scoring 및 selection

Validation에서는 train JSON의 평균·표준편차를 고정해서 사용했다. Validation feature 분포로
평균이나 표준편차를 다시 계산하지 않았다.

각 feature의 z-score는 다음과 같다.

```text
z_depth = clip((log1p(depth_std) - train_mean_depth) / train_std_depth, -3, 3)
z_speed = clip((log1p(past_speed) - train_mean_speed) / train_std_speed, -3, 3)
```

Query 원래 score를 `s`라고 할 때, 2026-08-02 당시의 고정 combined logit은 다음과 같다.

```text
base_logit     = logit(clip(s, 1e-6, 1 - 1e-6))
combined_logit = base_logit - 0.5 * z_depth - 0.25 * z_speed
```

두 feature 모두 `good_direction=low`이므로 train 평균보다 낮은 값은 combined score를 높이고,
높은 값은 낮춘다.

선택 조건은 다음과 같다.

```text
predicted foreground
AND depth/speed z-score가 finite
AND combined_logit >= logit(0.90)
```

선택된 query는 combined logit 내림차순으로 정렬하고, 모델의 `fg_score_topk`가 양수이면 마지막에
top-k를 적용했다.

이 방식은 baseline-preserving rescue가 아니라 **global reselection**이다. 따라서 원래
`score >= 0.90`인 query도 feature 보정 후 cutoff 아래로 내려가면 제외될 수 있으며, baseline query를
강제로 합치지 않았다.

## 4. Occupancy render 및 평가 조건

당시 eval의 주요 고정 조건은 다음과 같다.

- `EOCF_EVAL_MODE=2`: 현재 1프레임 + 미래 4프레임
- `EOCF_EVAL_FG_THR=0.9`
- `EOCF_EVAL_NMS_RADIUS=0`
- `EOCF_EVAL_WEIGHT_MODE=ones`
- `EOCF_EVAL_TRAJ_REFINE=1`
- `EOCF_EVAL_TRAINCAL_COMBINED_CUTOFF=0.9`
- `EOCF_EVAL_TRAINCAL_OCC_THRESHOLDS=0.8,0.9`
- query-rescue cache 비활성화
- 시각화 비활성화
- Asset metric만 계산하고 AABB IoU 및 Recall3D는 생략

고정 Depth+Speed selection의 full-resolution scene probability를 한 번 렌더한 뒤, 같은 probability에
OCC threshold `0.8`과 `0.9`를 각각 후처리했다. 즉 OCC threshold마다 모델 inference를 다시 실행한
것이 아니다.

`score >= 0.90` baseline도 같은 forward에서 내부 비교용으로 계산했지만, 당시 최종 CSV/JSON에서는
baseline 행을 제외하고 Depth+Speed 정책의 두 OCC 조건만 기록했다.

평가 구간 정의는 다음과 같다.

- `present`: 현재 1프레임
- `future`: 미래 4프레임 joint volume
- `combined`: 현재 1 + 미래 4프레임 joint volume

Asset IoU3D는 sample별 macro와 전체 TP/FP/FN 합산 micro를 집계했으며, train-calibrated summary의
대표 정렬 기준은 future macro Asset IoU3D였다.

## 5. 당시 실행 형태와 현재 재현 시 주의

변경 전 wrapper에서는 다음 형태로 실행했다.

```bash
RUN_TAG=20260802_034946 \
bash eval_query_rescue_traincal_depth_speed_full.sh
```

필요한 override가 있었다면 `GPU_ID`, `CHECKPOINT`, `DEPTH_SPEED_STATS`, `MAX_SAMPLES`, `RUN_DIR`를
환경변수로 지정하는 구조였다.

현재 wrapper는 alpha 6×6 sweep으로 바뀌었으므로 위 명령을 현재 코드에 그대로 적용하면 과거 방법을
재현하지 못한다. Claude/Codex로 재현 작업을 할 때는 다음 조건을 명시해야 한다.

1. 대상은 `20260803_123930`이 아니라 `20260802_034946` 방법이다.
2. active policy는 오직 `depth_std_a0.5+past_speed_a0.25__global090` 하나다.
3. train 4,000 normalization JSON을 고정하며 validation 재정규화를 금지한다.
4. combined cutoff는 `0.90`, OCC threshold는 `0.8/0.9`다.
5. 현재 6×6 alpha grid는 실행하지 않는다.
6. 평가 결과 수치를 이 방법 문서에 덧붙이지 않는다.

## 6. 생성 파일 확인

정상 완료 시 대상 run directory에는 다음 종류의 파일이 생성된다.

```text
eval_metrics_live.log
query_rescue_traincal_depth_speed_summary.csv
query_rescue_traincal_depth_speed_summary.json
```

이 문서에서는 파일의 존재와 schema만 안내하며 내부 결과값은 해석하거나 인용하지 않는다.
