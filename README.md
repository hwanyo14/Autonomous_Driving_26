# Model Summary
## 1. Scope
- Model: camera-only query-based 3D GMO occupancy forecasting
- Goal: 과거 multi-view camera sequence로 현재 및 미래 movable object의 3D occupancy, instance shape, trajectory를 예측한다.

## 2. Core Idea
- 기존 EfficientOCF는 dense BEV occupancy, height, flow를 예측하고 3D로 lift한다.
- 이 브랜치는 query-based instance forecasting 구조로 바꾼다.
- query 하나가 가능한 한 object instance 하나를 대표한다.
- query는 multi-view image feature를 attention으로 읽는다.
- query는 class, depth, 3D center, Gaussian occupancy shape, future trajectory를 예측한다.
- trajectory 학습 초기에 teacher forcing으로 GT 기반 past motion prior를 주입한다.

## 3. Input
- `img_inputs_seq`: 과거 3프레임의 6-view RGB image 및 camera geometry
- `future_egomotion`: frame 간 ego-motion transform
- `segmentation`, `segmentation_bev`: 3D 및 BEV occupancy target
- `segmentation_instance3d`, `gt_occ_inst`: instance-level 3D occupancy
- `segmentation_cls_instance3d`: semantic class와 instance label
- `gt_instance_centers_world`: instance별 temporal center target
- `gt_instance_centers_valid`, `gt_instance_ids`: instance validity 및 id
- `occ_dt`: distance transform supervision
```text
history = [t-2, t-1, t]
future  = [t+1, t+2, t+3, t+4]
time_receptive_field = 3
n_future_frames = 4
n_future_frames_plus = 6
```

## 4. Output
- `query_cls_logits`: query별 class 또는 background
- `query_depth_logits`: query별 depth bin distribution
- `centers_world`: attention과 depth로 lift된 3D center
- `mixture_*`: Gaussian mixture 기반 occupancy shape
- `traj_offsets`: future step별 xy displacement
- `soft occupancy`: Gaussian mixture를 voxelize한 occupancy probability

## 5. Architecture
- Image Encoder: 과거 3프레임, 6개 camera image를 backbone과 FPN neck으로 feature화한다.
- Depth and Context: BEVDepth, LSS 계열 depth branch로 proxy depth와 query context feature를 만든다.
- Query Transformer: learnable query가 camera feature token에 cross-attention하고, query self-attention과 FFN을 거친다.
- Temporal Query Update: frame을 순차 처리하며 query state를 다음 frame으로 넘긴다.
- Query Head: class, depth, center, Gaussian mixture, trajectory branch를 실행한다.

## 6. 3D Center Construction
center는 단순 MLP 회귀가 주 경로가 아니다.
```text
query feature
→ camera attention
→ query depth distribution
→ camera geometry 기반 3D lifting
→ query center
```
query가 image의 어디를 보는지와 그 위치의 depth를 결합해 3D center를 만든다.

## 7. Gaussian Occupancy
각 query는 object shape를 Gaussian mixture로 표현한다.
예측 항목은 component center offset, sigma, yaw, weight이다.
voxelizer가 mixture를 3D grid에 rasterize하여 query-level soft occupancy를 만든다.

## 8. Query-GT Matching
query가 instance를 담당하도록 Hungarian matching을 수행한다.
주요 cost는 다음과 같다.
- query center와 GT center의 L1 distance
- query attention map과 GT instance mask의 soft IoU
- temporal trajectory 단차
matching 결과는 class, center, depth, trajectory, occupancy loss에 공통으로 사용된다.

## 9. Teacher Forcing
teacher forcing은 trajectory head 입력 prior에만 적용된다.
학습 초기에는 predicted center가 불안정하므로 matched GT instance의 과거 center delta를 사용한다.
```text
GT past delta = GT center[s+1] - GT center[s]
normalized delta = clamp(delta / residual_max, -1, 1)
trajectory input의 motion prior slot을 normalized GT delta로 교체
```
future GT를 직접 입력으로 넣는 것이 아니라 과거 motion prior만 GT 기반으로 안정화한다.
설정된 iteration 이후에는 predicted center 기반 prior로 전환된다.

## 10. Losses
- query classification loss: matched query는 GT class, unmatched query는 background
- query depth loss: projected GT instance center depth supervision
- center routed loss: matched query center와 GT center의 L1 loss
- trajectory loss: predicted future xy offset과 GT center delta의 L1 loss
- attention bbox loss: query attention이 GT instance mask를 보도록 유도
- matched GMO occupancy loss: Gaussian occupancy와 GT instance occupancy 비교
- regularization: query decorrelation, Gaussian sigma regularization

## 11. Training Flow
```text
1. 과거 3프레임 6-view image 입력
2. image feature 및 context feature 추출
3. query transformer로 query feature 생성
4. query head로 class, depth, Gaussian parameter 예측
5. attention과 depth로 query center를 3D lift
6. query와 GT instance를 Hungarian matching
7. matched query에 대해 teacher-forcing motion prior 구성
8. trajectory head로 future xy offset 예측
9. present center와 offset을 누적해 future center 생성
10. Gaussian mixture를 voxelize해 occupancy 생성
11. 전체 loss로 end-to-end 학습
```

## 12. Key Interpretation
이 브랜치는 EfficientOCF를 dense BEV forecasting에서 query-based instance forecasting으로 바꾼 실험이다.
핵심은 query 하나가 instance 하나를 담당하고, geometry-aware lifting으로 center를 만들며, Gaussian mixture로 shape를 표현하고, teacher forcing으로 trajectory branch의 초기 학습을 안정화하는 것이다.
