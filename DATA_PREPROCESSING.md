# GT 데이터 전처리 절차 (efficientocf_gt / efficientocf_gt_f3)

최종 확정: 2026-07-13. 현재 학습·평가에 쓰이는 GT 캐시 2개가 정확히 어떤 파일을 어떤 순서로
실행해서 만들어졌는지 기록한다. 재현 시 이 문서의 명령을 순서대로 실행하면 된다.

## 0. 소스 데이터 (전처리 입력)

| 소스 | 경로 | 내용 |
|---|---|---|
| nuScenes annotations | `data/nuscenes/` + `data/nuscenes/nuscenes_occ_infos_{train,val}.pkl` | 3D 박스(위치·크기·회전·클래스), pose |
| nuScenes-Occupancy | `data/nuScenes-Occupancy/scene_*/occupancy/<lidar_token>.npy` | 씬 전체 semantic occupancy (voxel별 class 라벨, [z,y,x,...,label]) |

## 1. 산출물 구조

두 루트 모두 실제 위치는 `/home/hwanhee/datasets/`, 레포에서는 `data/` 심링크로 접근.

```
efficientocf_gt/GMO/            ← "raw" (필터 없음, 소스 캐시)
├── bbox_aabb_raw/    (A) annotation 박스 AABB 통짜 voxel — 중간 재료, 학습 미사용
├── bbox_rot_raw/     (B) 회전(OBB) 박스 통짜 voxel — 중간 재료, 학습 미사용
├── segmentation_instance3d/ (D) A ∩ occupancy(class 일치만) — inst3d
├── segmentation_aabb/       (E) A를 "D에 존재하는 (frame,inst)만" 게이트 (박스 안 깎음)
└── segmentation_rot/        (F) B를 같은 게이트로 필터

efficientocf_gt_f3/GMO/         ← 학습·평가용 (생성소멸 필터판)
├── segmentation_instance3d/  (D의 f3판)
├── segmentation_aabb/        (E의 f3판)
└── segmentation_rot/         (F의 f3판)
```

- 파일: `<scene_token>_<present_lidar_token>.npz`, 키당 29,049개 (train 23,930 + val 5,119)
- npz 내용: 7프레임(과거3+미래4, present=frame2) 리스트, 프레임당 `[N,5] int32` = `[x,y,z,cls,inst]`
- class id: 2=bicycle, 3=bus, 4=car, 5=construction, 6=motorcycle, 7=pedestrian(+CW), 9=trailer, 10=truck
  (nuScenes-Occupancy class id 공간과 동일. pedestrian은 데이터에 포함, 학습 로더의
  `exclude_occ_class_ids=(7,)`가 런타임 제거)
- instance id: 윈도우 내 annotation 등록 순서로 1부터 부여, **모든 레이어·모든 단계에서 동일 id
  공유(재부여 절대 없음)**. f3 필터 후에는 id에 빈 번호(갭)가 생기는 게 정상.

## 2. 실행 절차 (실제 실행 이력 그대로)

### Step 1 — A/B 생성 (+ 초판 D/E/F): `tools/gen_data/gen_new_gt_pipeline.py`
2026-07-11 실행(0630 레포에서, 스크립트는 양 레포 md5 동일). annotation을 rasterize해서 A/B를 만들고
occupancy와 교집합으로 D/E/F를 생성.

```bash
# shard 병렬 (--start/--end), split별
python tools/gen_data/gen_new_gt_pipeline.py --config <CONFIG> --split test  --out /home/hwanhee/datasets/efficientocf_gt
python tools/gen_data/gen_new_gt_pipeline.py --config <CONFIG> --split train --start 0 --end 6000 --out /home/hwanhee/datasets/efficientocf_gt
# ... (train을 4개 shard로 나눠 실행)
```

⚠ **재현 시 config 주의**: 이 스크립트는 config의 `train_capacity`를 그대로 따른다. 현재 레포의
subset config들은 `train_capacity=4000`(매 실행 랜덤 서브샘플)이라 **그대로 쓰면 train 전 키가
생성되지 않는다**. `<CONFIG>`는 반드시 `train_capacity=23930`, `test_capacity=5119`인 config를
쓰거나 임시로 그렇게 수정해서 쓸 것. (기본값 `--config .../full.py`는 삭제된 파일이라 실행 안 됨.
Step 2의 `regen_def_clsmatch.py`는 스크립트 내부에서 capacity를 강제해서 이 문제가 없음.)

⚠ **이 스크립트의 `compute_D`에는 버그가 있음**: AABB 안에서 "아무거나 점유된" voxel을 전부
keep해서 도로/건물 voxel이 인스턴스에 흡수됨(실측 이물질 ~21%). **이 단계 산출물 중 A/B만 현행
유효**하고, D/E/F는 Step 2에서 재생성됐다. 재현 시에도 이 스크립트는 A/B 생성용으로만 쓸 것.

### Step 2 — D/E/F 재생성 (class-match 수정판): `tools/gen_data/regen_def_clsmatch.py`
2026-07-13 실행. Step 1의 A/B를 재사용(재-rasterize 없음 = id 보존 보장)하고, occupancy만 재계산해서
**"점유 class == 인스턴스 class"인 voxel만** 교집합(D). E/F는 새 D로 재게이트.

```bash
# 6 shard 병렬 (test 2 + train 4), raw 모드(--no-f3-filter)
python tools/gen_data/regen_def_clsmatch.py --no-f3-filter --split test  --start 0     --end 2560  --out <staging>
python tools/gen_data/regen_def_clsmatch.py --no-f3-filter --split test  --start 2560  --end 5119  --out <staging>
python tools/gen_data/regen_def_clsmatch.py --no-f3-filter --split train --start 0     --end 6000  --out <staging>
python tools/gen_data/regen_def_clsmatch.py --no-f3-filter --split train --start 6000  --end 12000 --out <staging>
python tools/gen_data/regen_def_clsmatch.py --no-f3-filter --split train --start 12000 --end 18000 --out <staging>
python tools/gen_data/regen_def_clsmatch.py --no-f3-filter --split train --start 18000 --end 23930 --out <staging>
# 완료 후 <staging>/GMO/{D,E,F} 3폴더를 efficientocf_gt/GMO/로 이동(기존 D/E/F 대체)
```
소요 ~2.5시간(6 shard). 실행 로그: `efficientocf_gt/logs_regen_def_20260713/`.

핵심 규칙 (스크립트 내 구현):
- D = A의 row 중 `occupancy_dense[x,y,z] == row.cls`인 것만 (class-match 교집합)
- 중복 좌표는 first-wins dedup (등록 순 = 작은 instance id 우선)
- E/F 게이트: (frame, inst)가 D에 row 1개라도 있으면 해당 프레임의 박스 통짜 유지, 없으면 박스째 제거

### Step 3 — f3 파생 (생성소멸 필터): `tools/gen_data/derive_filtered_gt.py`
2026-07-13 실행. Step 2의 raw D/E/F에서 row 필터만 수행 (재계산·id 재부여 없음).

```bash
# 6 shard 병렬 (29,049키를 5,000 단위로)
python tools/gen_data/derive_filtered_gt.py \
  --src <raw D/E/F 루트> --out /home/hwanhee/datasets/efficientocf_gt_f3 \
  --require past3all --keep-human --start 0 --end 5000
# ... (5000 단위로 6개)
```
소요 ~30분. 실행 로그: `efficientocf_gt_f3/logs/`.

필터 규칙:
- `--require past3all`: 관측창(frame 0,1,2) **전부**에 D voxel이 있는 인스턴스만 윈도우 전체에서 유지
  (생성소멸 제거)
- `--keep-human`: pedestrian(7)은 데이터에 유지 (제거는 학습 로더의 `exclude_occ_class_ids`가 담당)

## 3. 검증 (2026-07-13 수행, 재현 시 동일하게 확인 권장)

| 항목 | 방법 | 결과 |
|---|---|---|
| 파일 수 | 레이어별 count == 29,049 | 통과 (gt 3레이어 + f3 3레이어) |
| 키 목록 | ann pkl의 train/val lidar_token과 전수 대조 | 23,930+5,119, 불일치 0 |
| id 정합 | 29,049키 전수: D/E/F의 id ⊆ A의 id (재번호 없음) | 위반 0 |
| npz 무결성 | 랜덤 600파일 로드 + 7프레임·5컬럼 계약 | 손상 0 |
| f3 필터 | 100키: f3 인스턴스 == raw D의 0∩1∩2 교집합 | 100/100 정확 (인스턴스 ~24% 제거) |
| 시간축 | 210프레임: D[t]⊆A[t], F[t]⊆B[t] (같은 t) + 이동 물체 centroid 추적 | 210/210, centroid 프레임별 일치 |
| 시각화 | `data_vis/0713/{gt,f3}_sample_*.png` (D old/new, E, F 비교) | 회전 물체가 rot 각도로 정상 |

## 4. 학습/평가 배선 (참고)

- 학습 config: `gt_occ_inst_dataset_path`(→D), `gt_bbox_aabb_dataset_path`(→E 또는, `_rot` 계열
  config는 `gt_bbox_aabb_subdir='segmentation_rot'`로 F)를 `./data/efficientocf_gt_f3/`로 설정
- eval bbox 지표: `efficientocf.py`의 lazy-loader가 같은 루트의 E(aabb 지표)/F(rot 지표)를 직접 로드
- 상세 배선·버그 이력: `CHANGELOG.md`/`NOTES.md`의 2026-07-13 항목들
