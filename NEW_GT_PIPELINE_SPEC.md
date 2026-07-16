# 새 GT 캐시 생성 파이프라인 — 완전 명세서 (v2)

이 문서는 EfficientOCF 레포(`Autonomous_Driving_26_ksh_0630`)의 기존 GT 캐시 체계를 대체할 **새로운, 자체 생성 가능한 GT 파이프라인**을 처음부터 구현하기 위한 명세서입니다. 이 레포에서 실제 코드를 정밀 검증(함수 원문, 라인번호, 실측 데이터)해서 작성했으며, 다른 환경/서버에서 이 문서만 보고 구현을 시작할 수 있도록 필요한 배경·근거·정확한 공식을 전부 포함합니다.

> **v2 변경 이력**: v1(최초 작성)은 4파일(bbox_aabb/bbox_rot/nusocc_inst + 게이트 메타데이터) 구조였으나, 대화 중 설계가 정제되어 **"raw 레이어 3개 + 파생 레이어 3개, 총 6개 파일"** 구조로 바뀌었다. 핵심 변화: (1) 게이트 판정을 별도 메타데이터로 저장하지 않고 `nusocc_inst`에 row가 있는지 자체로 판별하도록 단순화, (2) raw(가공 전) 레이어를 전부 디스크에 저장해서 나중에 재계산 없이 필터링 규칙만 바꿔 재파생할 수 있게 함.

---

## 0. 배경 — 왜 이걸 만드는가

기존에 이 레포가 쓰는 GT 소스들:

| 이름 | 실체 | 문제 |
|---|---|---|
| `gt_occ_inst` (`nuScenes-Occupancy_inst3d`) | annotation bbox ∩ 실제 LiDAR 점유(nuScenes-Occupancy) 교집합 | **생성 스크립트가 이 repo 안에 없음**(외부 프로젝트 생성, `reports/summary.json`만 남아 레시피만 역추적 가능). 소수 인스턴스 경계 혼입, construction_worker 오분류(cls=5) 등 알려진 오염 존재. |
| `segmentation_instance3d` (`data/efficientocf/GMO/segmentation_instance3d`) | annotation bbox를 AABB로 통짜 rasterize (class 컬럼은 항상 placeholder=1) | 마찬가지로 생성 스크립트 없음. |
| `efficientocf_bboxcls_v2/v3` (bbox aabb/rot) | `tools/gen_data/gen_bbox_gt_v2.py`로 생성 — **유일하게 자체 생성 가능** | v3(train)는 건전, v2(val)는 pedestrian 우선매칭이 빠져 construction_worker가 cls=5로 오분류됨. train/val이 서로 다른 옵션으로 따로 생성되어 둘 사이에 불일치가 생김. |

**목표**: `gen_bbox_gt_v2.py`가 이미 증명한 rasterize 알고리즘을 재사용해서, GT 전체를 **하나의 파이프라인에서, 하나의 instance 번호 체계를 공유하며** 새로 생성한다. 생성 스크립트를 이 repo 안에 두어 향후 완전히 재현·검증·수정 가능하게 만드는 것이 핵심 목적이다.

**이번 라운드에서 명시적으로 안 하는 것** (나중에 별도 단계로 분리):
- 생성소멸 필터 (미래 신규 등장 물체 제외) — 끔
- 사람(pedestrian) 제거 — 끔
- construction_worker 특수처리(pedestrian 우선매칭) — 끔, `CLS_MAP` 순서 그대로 둬서 "공사장 인부가 차량으로 찍히는" 현상도 이번엔 그냥 둔다

즉 이번 파이프라인은 **"최대한 raw하게, 아무것도 배제하지 않고, 다만 여러 GT 간의 정합성만 확실히 보장"**하는 것이 목표다.

---

## 1. 전체 아키텍처 — raw 3종 + 파생 3종, 총 6개 파일

```
                    ┌─────────────────────────────────────────┐
                    │  1단계: annotation → raw bbox 생성        │
                    │  (record_instance, 필터 전부 OFF)         │
                    └─────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
        [저장 A] bbox_aabb_raw          [저장 B] bbox_rot_raw
        (annotation box AABB 통짜,       (bbox_aabb_raw에 OBB 검사만
         게이트 없음, 필터 없음)           추가 적용, 역시 게이트 없음)

                    ┌─────────────────────────────────────────┐
                    │  독립: nuScenes-Occupancy 로딩 + present   │
                    │  프레임 정렬 (get_seq_occ 재사용)           │
                    └─────────────────────────────────────────┘
                                    │
                                    ▼
                    [저장 C] occ_raw
                    (씬 전체 실점유, class만 있고 instance 없음,
                     bbox와 교집합 안 한 원본, 필터 없음)

                    ═══════ 여기까지가 "raw 레이어" — 전부 저장됨 ═══════

        [저장 A] bbox_aabb_raw  ∩(voxel join)  [저장 C] occ_raw
                                    │
                                    ▼
                    [저장 D] nusocc_inst
                    (박스 voxel 중 실점유에도 있는 것만, voxel 단위로 trim,
                     instance id 있음 — 오늘의 gt_occ_inst와 정의 동일)
                                    │
                    "존재 게이트" = (frame, instance_id)가
                    nusocc_inst에 row를 1개라도 갖는지로 판정
                    (별도 계산/저장 없음 — nusocc_inst 자체가 게이트 기준)
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
  [저장 A]를 이 게이트로 필터           [저장 B]를 이 게이트로 필터
  (row 단위 필터, box는 안 깎음)         (row 단위 필터, box는 안 깎음)
              │                                             │
              ▼                                             ▼
    [저장 E] bbox_aabb (최종)                     [저장 F] bbox_rot (최종)
```

### 1.1 최종 산출물 6개 요약

| # | 파일명 | 정의 | instance id | 용도 |
|---|---|---|---|---|
| A | `bbox_aabb_raw` | annotation box AABB 통짜, 게이트/필터 전혀 없음 | 있음 | **연결 안 함, 그냥 만들어서 저장만**(향후 재파생용) |
| B | `bbox_rot_raw` | A에 OBB 검사만 추가, 게이트 없음 | 있음 | **연결 안 함, 저장만** |
| C | `occ_raw` | 씬 전체 실점유(nuScenes-Occupancy), bbox와 무관 | 없음 | **연결 안 함, 저장만** |
| D | `nusocc_inst` | A ∩ C (voxel 단위 trim) | 있음 | ✅ 연결함 — 오늘의 `gt_occ_inst` 대체 |
| E | `bbox_aabb` (최종) | A를 "D에 존재하는 (frame,inst)만" 게이트, 박스는 안 깎음 | 있음 | ✅ 연결함 — 오늘의 `segmentation_instance3d` + `v3` 대체 |
| F | `bbox_rot` (최종) | B를 E와 같은 게이트로 필터 | 있음 | ✅ 연결함 — 오늘의 `v2`(rot 부분) 대체 |

**왜 raw 3개(A/B/C)도 저장하는가**: 지금은 로더에 연결하지 않지만, ① 디버깅/감사 시 "게이트 전/후"를 직접 대조할 수 있고, ② 나중에 게이트 규칙(예: threshold, class 필터)을 바꾸고 싶을 때 nuScenes 원본부터 다시 만들 필요 없이 A/B/C만 가지고 D/E/F를 재계산하면 되고, ③ `occ_raw`(C)는 인스턴스 개념과 무관한 "씬 전체 occupancy" GT라서 그 자체로 다른 용도(예: 순수 occupancy prediction 과제)에 쓸 수도 있다.

### 1.2 왜 "voxel 교집합"이 최종 산출물이 아니라 "존재 게이트"인가

`nusocc_inst`(D)는 voxel 단위로 정확히 trim된 sparse 데이터인 반면, 최종 `bbox_aabb`/`bbox_rot`(E/F)는 **박스를 안 깎고 통짜로 유지**한다. 순수 교집합으로만 가면 최종 결과물이 사실상 `nusocc_inst`의 복제본이 되어, dice loss가 기대하는 "박스 안은 확장해도 된다"는 감독 신호가 사라지기 때문이다(§8 참조).

### 1.3 인스턴스 번호 규칙 (중요, 반드시 지킬 것)

- 번호는 **윈도우(7프레임 시퀀스) 단위로, `record_instance()` 알고리즘에 따라 1번부터 순차 부여**된다(§3).
- **한 윈도우 안에서 A→B→D→E→F(및 C)가 반드시 같은 `instance_dict`/`instance_map`을 공유**해야 한다. rot를 나중에 별도로 재-rasterize하거나, nusocc_inst를 다른 시점에 다시 만들면 안 된다.
- 실무적으로: **생성기 하나가 윈도우당 `prepare_test_data()`를 한 번 호출해서 `seq`를 얻고, 그 하나의 `seq`로 A→B→C→D→E→F를 순서대로 계산해 6개 출력을 한 번에 뽑는 구조**로 만들 것.
- 윈도우 간(슬라이딩 윈도우로 겹치는 다른 샘플) 번호가 달라지는 것은 **문제 아님** — 학습/검증은 윈도우(npz 파일) 하나씩만 보므로 무관.

---

## 2. 재사용할 기존 코드 (전부 실측 검증됨)

가장 중요한 재사용 대상은 **`tools/gen_data/gen_bbox_gt_v2.py`**(bbox 계열)와 **`projects/occ_plugin/datasets/pipelines/loading_occupancy.py`**(occupancy 계열)다.

### 2.1 `record_instance()` 전체 원문

파일: `projects/occ_plugin/datasets/efficientocf_dataset.py:164-236`

```python
def record_instance(self, idx, instance_map):
    """
    Record information about each visible instance in the sequence and assign a unique ID to it
    """
    rec = self.data_infos[idx]
    translation, rotation = self.get_lidar_pose(rec)
    self.egopose_list.append([translation, rotation])
    ego2lidar_translation, ego2lidar_rotation = self.get_ego2lidar_pose(rec)
    self.ego2lidar_list.append([ego2lidar_translation, ego2lidar_rotation])

    current_sample = self.nusc.get('sample', rec['token'])
    for annotation_token in current_sample['anns']:
        annotation = self.nusc.get('sample_annotation', annotation_token)
        # Filter out all non vehicle instances
        gmo_flag = False
        for class_name in self.classes:
            if class_name in annotation['category_name']:
                gmo_flag = True
                break
        if not gmo_flag:
            continue
        # Specify semantic id if use_separate_classes
        semantic_id = 1
        if self.use_separate_classes:
            if 'vehicle.bicycle' in annotation['category_name']: # rm static_object.bicycle_rack
                semantic_id = 1
            elif 'bus'  in annotation['category_name']:
                semantic_id = 2
            elif 'car'  in annotation['category_name']:
                semantic_id = 3
            elif 'construction'  in annotation['category_name']:
                semantic_id = 4
            elif 'motorcycle'  in annotation['category_name']:
                semantic_id = 5
            elif 'trailer'  in annotation['category_name']:
                semantic_id = 6
            elif 'truck'  in annotation['category_name']:
                semantic_id = 7
            elif 'pedestrian'  in annotation['category_name']:
                semantic_id = 8

        # Filter out invisible vehicles
        FILTER_INVISIBLE_VEHICLES = True
        if FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and annotation['instance_token'] not in self.visible_instance_set:
            continue
        # Filter out vehicles that have not been seen in the past
        if self.counter >= self.time_receptive_field and annotation['instance_token'] not in self.visible_instance_set:
            continue
        self.visible_instance_set.add(annotation['instance_token'])

        if annotation['instance_token'] not in instance_map:
            instance_map[annotation['instance_token']] = len(instance_map) + 1
        instance_id = instance_map[annotation['instance_token']]
        instance_attribute = int(annotation['visibility_token'])

        if annotation['instance_token'] not in self.instance_dict:
            # For the first occurrence of an instance
            self.instance_dict[annotation['instance_token']] = {
                'timestep': [self.counter],
                'translation': [annotation['translation']],
                'rotation': [annotation['rotation']],
                'size': annotation['size'],
                'instance_id': instance_id,
                'semantic_id': semantic_id,
                'attribute_label': [instance_attribute],
            }
        else:
            # For the instance that have appeared before
            self.instance_dict[annotation['instance_token']]['timestep'].append(self.counter)
            self.instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
            self.instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
            self.instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

    return instance_map
```

**필터 3곳 위치** (§3.2에서 수정 지시):
1. `gmo_flag` 판정 (line 178-184) — `self.classes`와 `category_name`의 substring 매칭
2. `visibility_token==1` 체크 (line 205-208) — 최초 등장 시 저가시성이면 스킵
3. `self.counter >= self.time_receptive_field` 체크 (line 209-211) — **이게 "생성소멸 필터"의 실체**

**동작 디테일**: `instance_map[token] = len(instance_map) + 1` — 등록 성공 순서대로 1부터 번호 부여, 재배정 없음. `instance_dict`의 `size`/`instance_id`/`semantic_id`는 최초 등장 프레임 값으로 고정(스칼라), `timestep`/`translation`/`rotation`/`attribute_label`은 프레임마다 append.

### 2.2 `refine_instance_poly()` — 위치 동결/보간 (유지 권장)

파일: `projects/occ_plugin/datasets/efficientocf_dataset.py:266-303`

```python
@staticmethod
def _check_consistency(translation, prev_translation, threshold=1.0):
    x, y = translation[:2]
    prev_x, prev_y = prev_translation[:2]
    if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
        return False
    return True

def refine_instance_poly(self, instance):
    """Fix the missing frames and disturbances of ground truth caused by noise"""
    pointer = 1
    for i in range(instance['timestep'][0] + 1, self.sequence_length):
        if i not in instance['timestep']:
            instance['timestep'].insert(pointer, i)
            instance['translation'].insert(pointer, instance['translation'][pointer-1])
            instance['rotation'].insert(pointer, instance['rotation'][pointer-1])
            instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer-1])
            pointer += 1
            continue
        if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer-1]):
            instance['translation'][pointer] = instance['translation'][pointer-1]
            instance['rotation'][pointer] = instance['rotation'][pointer-1]
            instance['attribute_label'][pointer] = instance['attribute_label'][pointer-1]
        pointer += 1
    return instance
```

호출 시점: `prepare_sequential_data()`에서 시퀀스 조립이 끝난 뒤 `instance_dict`의 각 항목에 대해 1회 실행(`efficientocf_dataset.py:363-364`).

### 2.3 `get_indices()` / `prepare_sequential_data()` — 슬라이딩 윈도우 및 `seq` 구조

파일: `projects/occ_plugin/datasets/efficientocf_dataset.py:115-142`, `:331-381`

```python
def get_indices(self):
    indices = []
    for index in range(len(self.data_infos)):
        is_valid_data = True
        previous_rec = None
        current_indices = []
        for t in range(self.sequence_length):
            index_t = index + t
            if index_t >= len(self.data_infos):
                is_valid_data = False
                break
            rec = self.data_infos[index_t]
            if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                is_valid_data = False
                break
            current_indices.append(index_t)
            previous_rec = rec
        if is_valid_data:
            indices.append(current_indices)
    return np.asarray(indices)
```

```python
def prepare_sequential_data(self, index):
    instance_map = {}
    input_seq_data = {}
    keys = ['input_dict','future_egomotion', 'sample_token']
    for key in keys:
        input_seq_data[key] = []
    scene_lidar_token = []

    for self.counter, index_t in enumerate(self.indices[index]):
        input_dict_per_frame = self.get_data_info(index_t)
        if input_dict_per_frame is None:
            return None
        input_seq_data['input_dict'].append(input_dict_per_frame)
        input_seq_data['sample_token'].append(input_dict_per_frame['sample_idx'])
        instance_map = self.record_instance(index_t, instance_map)
        future_egomotion = self.get_future_egomotion(index_t)
        input_seq_data['future_egomotion'].append(future_egomotion)
        scene_lidar_token.append(input_dict_per_frame['scene_token']+"_"+input_dict_per_frame['lidar_token'])
        if self.counter == self.time_receptive_field - 1:
            self.present_scene_lidar_token = input_dict_per_frame['scene_token']+"_"+input_dict_per_frame['lidar_token']

    if self.test_mode:
        test_idx_path = os.path.join(self.idx_root, "test_ids")
        if not os.path.exists(test_idx_path):
            os.mkdir(test_idx_path)
        np.savez(os.path.join(test_idx_path, self.present_scene_lidar_token), scene_lidar_token)

    for token in self.instance_dict.keys():
        self.instance_dict[token] = self.refine_instance_poly(self.instance_dict[token])

    input_seq_data.update(dict(
        global_idx=int(index), time_receptive_field=self.time_receptive_field,
        sequence_length=self.sequence_length, egopose_list=self.egopose_list,
        ego2lidar_list=self.ego2lidar_list, instance_dict=self.instance_dict,
        instance_map=instance_map, indices=self.indices[index],
        scene_token=self.present_scene_lidar_token,
    ))
    example = self.pipeline(input_seq_data)
    return example
```

`prepare_test_data(i)` 호출 전, 4개 상태 수동 리셋 필수(원래 `__getitem__`이 하던 일):
```python
dataset.egopose_list = []
dataset.ego2lidar_list = []
dataset.visible_instance_set = set()
dataset.instance_dict = {}
```

`present_scene_lidar_token`(=`scene_token + "_" + lidar_token`, receptive field 마지막 프레임 기준)이 **최종 npz 파일명**이 된다.

### 2.4 `get_lidar_pose()` / `get_ego2lidar_pose()` — pose 소스

파일: `projects/occ_plugin/datasets/efficientocf_dataset.py:144-161`

```python
def get_lidar_pose(self, rec):
    ego2global_translation = rec['ego2global_translation']
    ego2global_rotation = rec['ego2global_rotation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    return trans, rot

def get_ego2lidar_pose(self, rec):
    lidar2ego_translation = rec['lidar2ego_translation']
    lidar2ego_rotation = rec['lidar2ego_rotation']
    trans = -np.array(lidar2ego_translation)
    rot = Quaternion(lidar2ego_rotation).inverse
    return trans, rot
```

`nusc.get('ego_pose',...)`/`nusc.get('calibrated_sensor',...)`를 직접 호출하지 않고, `rec = self.data_infos[idx]`(사전 생성된 `.pkl` info)의 필드를 그대로 쓴다. `nusc.get()`이 실제 호출되는 곳은 `record_instance()` 안의 `self.nusc.get('sample', ...)`와 `self.nusc.get('sample_annotation', ...)` 두 곳뿐이다. 값은 부호 반전 + inverse quaternion으로 미리 저장돼 있어서, box 변환 시 `box.translate(+trans); box.rotate(+rot)`만 호출해도 global→ego / ego→lidar 변환이 성립한다.

---

## 3. Raw 레이어 생성 — [저장 A] `bbox_aabb_raw`, [저장 B] `bbox_rot_raw`

### 3.1 필터 제거 — `record_instance_raw()` (생성소멸 OFF / 사람 제거 OFF)

```python
def record_instance_raw(self, idx, instance_map):
    rec = self.data_infos[idx]
    translation, rotation = self.get_lidar_pose(rec)
    self.egopose_list.append([translation, rotation])
    ego2lidar_translation, ego2lidar_rotation = self.get_ego2lidar_pose(rec)
    self.ego2lidar_list.append([ego2lidar_translation, ego2lidar_rotation])

    current_sample = self.nusc.get('sample', rec['token'])
    for annotation_token in current_sample['anns']:
        annotation = self.nusc.get('sample_annotation', annotation_token)

        gmo_flag = False
        for class_name in self.classes:
            if class_name in annotation['category_name']:
                gmo_flag = True
                break
        if not gmo_flag:
            continue

        # ⛔ 제거: visibility_token==1 최초-등장 필터 (원본 §2.1 line 205-208)
        # ⛔ 제거: self.counter >= time_receptive_field 미래-신규 필터 (원본 line 209-211, "생성소멸 필터")
        self.visible_instance_set.add(annotation['instance_token'])

        if annotation['instance_token'] not in instance_map:
            instance_map[annotation['instance_token']] = len(instance_map) + 1
        instance_id = instance_map[annotation['instance_token']]
        instance_attribute = int(annotation['visibility_token'])

        if annotation['instance_token'] not in self.instance_dict:
            self.instance_dict[annotation['instance_token']] = {
                'timestep': [self.counter],
                'translation': [annotation['translation']],
                'rotation': [annotation['rotation']],
                'size': annotation['size'],
                'instance_id': instance_id,
                'semantic_id': 1,
                'attribute_label': [instance_attribute],
            }
        else:
            self.instance_dict[annotation['instance_token']]['timestep'].append(self.counter)
            self.instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
            self.instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
            self.instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

    return instance_map
```

**`dataset.classes` 설정** (사람도 포함시켜야 하므로 기존 `--include_ped_ids`와 동일하게):
```python
dataset.classes = ["vehicle", "human"]
```
(`record_with_category` 몽키패치(§4.1)를 그대로 씌워서 `category_name` 주입 필요.)

### 3.2 `CLS_MAP` 설정 (construction_worker 특수처리 OFF)

```python
CLS_MAP = (
    ("vehicle.bicycle", 2), ("bus", 3), ("car", 4), ("construction", 5),
    ("motorcycle", 6), ("trailer", 9), ("truck", 10),
)
# pedestrian을 "맨 뒤에" 추가 — 우선 배치(construction_worker 특수처리) 안 함
cls_map = CLS_MAP + (("pedestrian", 7),)
```
`category_to_nusocc_id()`가 순서대로 첫 매치를 반환하므로, `human.pedestrian.construction_worker`는 `"construction"`이 먼저 걸려 여전히 `cls=5`로 찍힌다 — 의도된 동작(§0). pedestrian 우선매칭까지 켜고 싶으면 `cls_map = (("pedestrian", 7),) + CLS_MAP`로 순서만 바꾸면 된다(v3와 동일해짐, 옵션으로 남겨둠).

### 3.3 `rasterize_sequence()` — AABB+OBB voxel 생성 (원문 그대로 재사용)

파일: `tools/gen_data/gen_bbox_gt_v2.py:64-143`

```python
def category_to_nusocc_id(name, cls_map=CLS_MAP):
    for sub, cid in cls_map:
        if sub in name:
            return cid
    return None

def rasterize_sequence(seq, pc_range, grid_size, cls_map=CLS_MAP, aabb_only=False):
    """seq: prepare_sequential_data 출력 dict. returns (aabb_list, rot_list)."""
    res = np.array([(pc_range[3 + i] - pc_range[i]) / grid_size[i] for i in range(3)])
    start = np.array([pc_range[i] + res[i] / 2.0 for i in range(3)])
    dims = np.asarray(grid_size, dtype=np.int64)

    trf = seq["time_receptive_field"]
    T = seq["sequence_length"]
    present_ego = seq["egopose_list"][trf - 1]
    present_e2l = seq["ego2lidar_list"][trf - 1]

    aabb_frames, rot_frames = [], []
    for t in range(T):
        rows_aabb, rows_rot = [], []
        for token, inst in seq["instance_dict"].items():
            if t not in inst["timestep"]:
                continue
            ptr = inst["timestep"].index(t)
            cls_id = category_to_nusocc_id(inst["category_name"], cls_map)
            if cls_id is None:
                continue

            box = Box(inst["translation"][ptr], inst["size"], Quaternion(inst["rotation"][ptr]))
            box.translate(present_ego[0]); box.rotate(present_ego[1])
            box.translate(present_e2l[0]); box.rotate(present_e2l[1])
            pts = box.corners().T  # [8,3] lidar coords

            lo, hi = pts.min(axis=0), pts.max(axis=0)
            if not (pc_range[0] <= lo[0] and hi[0] <= pc_range[3]
                    and pc_range[1] <= lo[1] and hi[1] <= pc_range[4]
                    and pc_range[2] <= lo[2] and hi[2] <= pc_range[5]):
                continue  # whole-box in-range 규칙

            vox = np.round((pts - start + res / 2.0) / res).astype(np.int64)
            vmin = np.clip(vox.min(axis=0), 0, dims - 1)
            vmax = np.clip(vox.max(axis=0), 0, dims - 1)
            xs = np.arange(vmin[0], vmax[0] + 1)
            ys = np.arange(vmin[1], vmax[1] + 1)
            zs = np.arange(vmin[2], vmax[2] + 1)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            cand = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # [M,3]

            iid = int(inst["instance_id"])
            tag = np.array([[cls_id, iid]], dtype=np.int64).repeat(len(cand), axis=0)
            rows_aabb.append(np.concatenate([cand, tag], axis=1))

            if aabb_only:
                continue
            centers_w = cand * res.reshape(1, 3) + start.reshape(1, 3) - res.reshape(1, 3) / 2.0
            local = (centers_w - box.center.reshape(1, 3)) @ box.rotation_matrix  # R^T x = x @ R
            w, l, h = box.wlh
            half = np.array([l / 2.0, w / 2.0, h / 2.0]) + res / 2.0
            inside = np.all(np.abs(local) <= half.reshape(1, 3), axis=1)
            if inside.any():
                rows_rot.append(np.concatenate([cand[inside], tag[inside]], axis=1))

        def pack(rows):
            return (np.concatenate(rows, axis=0).astype(np.int32)
                    if rows else np.zeros((0, 5), dtype=np.int32))
        aabb_frames.append(pack(rows_aabb))
        rot_frames.append(pack(rows_rot))
    return aabb_frames, rot_frames
```

**1단계 호출**: `bbox_aabb_raw, bbox_rot_raw = rasterize_sequence(seq, pc_range, grid_size, cls_map, aabb_only=False)` — 이번엔 `aabb_only=False`로 호출해서 **A와 B를 한 번에 뽑는다**(게이트가 없으니 rot도 이 시점에 바로 만들 수 있음 — 3단계에서 재-rasterize할 필요가 없어졌다).

각 row는 `[x, y, z, cls_id, instance_id]`(5열, int32). `box.wlh`는 `[width,length,height]` 순서인데 `half_extent`엔 `[l/2,w/2,h/2]`로 뒤바뀌어 들어가는 것이 원본 코드 그대로의 관례(§9.6에서 재확인) — 버그 아님, 그대로 재현.

---

## 4. Raw occupancy 레이어 — [저장 C] `occ_raw`

### 4.1 nuScenes-Occupancy 원본 데이터 포맷 (실측 확인됨)

파일: `data/nuScenes-Occupancy/scene_<scene_token>/occupancy/<lidar_token>.npy`

- `dtype=int32`, `shape=(N, 4)` — dense array가 아니라 **sparse point-list**. 점유되지 않은 voxel은 파일에 없음.
- 컬럼 순서: `[z, y, x, cls]` (col0=Z 0~39 / col1=Y 0~511 / col2=X 0~511 / col3=class 0~16).
- class 0도 실제로 존재(“empty”라는 실제 semantic 라벨, 아래 §4.2에서 처리 방식 설명).
- 좌표계는 **그 샘플 자신의 LIDAR_TOP 프레임 기준**(global 아님).

**class id 체계** (`projects/occ_plugin/utils/nusc_param.py:10-28`):
```python
nusc_class_names = [
    "empty", "barrier", "bicycle", "bus", "car", "construction", "motorcycle",
    "pedestrian", "trafficcone", "trailer", "truck", "driveable_surface",
    "other", "sidewalk", "terrain", "mannade", "vegetation",
]
```
인덱스가 곧 class id(0~16)이며, bbox GT(§3)가 쓰는 nusocc raw id와 완전히 동일한 id 공간이다.

### 4.2 로딩 및 present-frame 정렬 — `get_seq_occ()` (원문 그대로 재사용)

파일: `projects/occ_plugin/datasets/pipelines/loading_occupancy.py:221-320`, 좌표 헬퍼 `:456-466`

```python
def voxel2world(self, voxel):
    return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]

def world2voxel(self, world):
    return (world - self.pc_range[:3][None, :]) / self.voxel_size[None, :]

def get_seq_occ(self, results, only_gt_occ=True):
    sequence_length = results['sequence_length']
    gt_occ_seq = []
    for count in range(sequence_length):
        scene_token_cur = results['input_dict'][count]['scene_token']
        lidar_token_cur = results['input_dict'][count]['lidar_token']
        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(scene_token_cur, lidar_token_cur)

        pcd = np.load(os.path.join(self.occ_path, rel_path))          # [z,y,x,cls]
        pcd_label = pcd[..., -1:]
        pcd_label[pcd_label == 0] = 255                                # raw class 0(empty annotation) -> sentinel 255
        pcd_np_cor = self.voxel2world(pcd[..., [2, 1, 0]] + 0.5)       # [z,y,x]->[x,y,z], voxel center

        egopose_list = results['egopose_list']
        ego2lidar_list = results['ego2lidar_list']
        time_receptive_field = results['time_receptive_field']
        present_global2ego = egopose_list[time_receptive_field - 1]
        present_ego2lidar = ego2lidar_list[time_receptive_field - 1]
        cur_global2ego = egopose_list[count]
        cur_ego2lidar = ego2lidar_list[count]

        # cur_lidar -> cur_ego -> global -> present_ego -> present_lidar
        pcd_np_cor = np.dot(cur_ego2lidar[1].inverse.rotation_matrix, pcd_np_cor.T).T - cur_ego2lidar[0]
        pcd_np_cor = np.dot(cur_global2ego[1].inverse.rotation_matrix, pcd_np_cor.T).T - cur_global2ego[0]
        pcd_np_cor = pcd_np_cor + present_global2ego[0]
        pcd_np_cor = np.dot(present_global2ego[1].rotation_matrix, pcd_np_cor.T).T
        pcd_np_cor = pcd_np_cor + present_ego2lidar[0]
        pcd_np_cor = np.dot(present_ego2lidar[1].rotation_matrix, pcd_np_cor.T).T

        pcd_np_cor = self.world2voxel(pcd_np_cor)
        pcd_np_cor = np.clip(pcd_np_cor, np.array([0, 0, 0]), self.grid_size - 1)
        pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied  # =0
        processed_label = nb_process_label(processed_label, pcd_np)   # numba majority-vote merge

        gt_occ_seq.append(torch.from_numpy(processed_label))
    return gt_occ_seq
```

`nb_process_label`(numba, `loading_occupancy.py:546-560`)이 좌표 반올림으로 같은 voxel에 여러 class가 겹치면 **다수결**로 하나를 확정해 dense `[X,Y,Z]` uint8에 채운다. 파일에 없던 voxel은 자동 `0`(unoccupied).

**최종 dense 값 체계**: `0`=unoccupied(진짜 빈 공간) / `1~16`=정상 semantic class / `255`=noise(원본 raw class 0).

⚠ `loading_occupancy.py`의 `use_fine_occ` 플래그는 `__call__` 안에서 실제로 참조되지 않는다 — 이 로더는 그 값과 무관하게 항상 이 sparse npy 기반 파싱을 쓴다.

### 4.3 dense → sparse 변환 (저장용)

`get_seq_occ()`가 만드는 dense `[X,Y,Z]`를 그대로 저장하지 않고, 나머지 GT와 형식을 맞추기 위해 sparse row list로 변환해서 저장한다:

```python
def get_seq_occ_sparse(seq, occ_path, grid_size, pc_range):
    """get_seq_occ()의 dense 결과를 [x,y,z,cls] sparse row list로 변환. instance id 없음(4열)."""
    gt_occ_seq = get_seq_occ(seq, occ_path)  # list[T] of dense [X,Y,Z] uint8
    sparse_frames = []
    for dense in gt_occ_seq:
        dense_np = dense.numpy() if hasattr(dense, "numpy") else dense
        occ_mask = dense_np != 0   # unoccupied(0)만 제외. noise(255)는 raw 원칙상 그대로 포함.
        xs, ys, zs = np.nonzero(occ_mask)
        cls = dense_np[xs, ys, zs]
        rows = np.stack([xs, ys, zs, cls], axis=1).astype(np.int32)
        sparse_frames.append(rows)
    return sparse_frames
```

**설계 결정 — class 값 컨벤션**: `get_seq_occ()`가 이미 원본 class 0("empty" 라벨)을 255(noise sentinel)로 remap한 상태를 그대로 물려받는다(원본 파일의 진짜 raw 0 값이 아니라, 이 레포 기존 관례의 255-remap을 유지). 이유: `get_seq_occ()`를 통째로 재사용하는 게(좌표 등록 체인·다수결 병합을 직접 재구현하지 않아도 됨) 훨씬 안전하고, 이 레포 다른 코드가 이미 이 0/1-16/255 컨벤션을 알고 있어 일관성이 유지된다. 원본 그대로의(리맵 안 한) 값이 필요하면 `pcd_label[pcd_label==0]=255` 줄만 빼고 직접 재구현해야 한다 — 이번 버전은 재사용 우선으로 위 방식을 기본값으로 채택.

`occ_raw`는 **class 필터링을 전혀 하지 않는다**(도로/건물/식생 등 모든 class 포함) — "raw" 원칙(§0)에 따라 가장 완전한 형태로 저장하고, GMO 8종만 걸러 쓰고 싶으면 로드 시점에 필터링하면 된다.

---

## 5. 파생 레이어 1 — [저장 D] `nusocc_inst` (voxel 단위 교집합)

`bbox_aabb_raw`(A)와 `occ_raw`(C)를 좌표 join해서 만든다 — **박스 voxel 중 실점유(noise 아니고 unoccupied 아닌)에도 존재하는 것만** 남기고, 원본 bbox row의 `cls`/`inst` 태그를 그대로 붙인다.

```python
def compute_nusocc_inst(bbox_aabb_raw_frames, occ_raw_frames):
    """
    bbox_aabb_raw_frames: list[T] of [N,5] int32 [x,y,z,cls,inst]  (저장 A)
    occ_raw_frames:    list[T] of [M,4] int32 [x,y,z,cls]       (저장 C)
    Returns: list[T] of [K,5] int32 [x,y,z,cls,inst]               (저장 D)
    """
    nusocc_frames = []
    for bbox_rows, occ_rows in zip(bbox_aabb_raw_frames, occ_raw_frames):
        if bbox_rows.shape[0] == 0 or occ_rows.shape[0] == 0:
            nusocc_frames.append(np.zeros((0, 5), dtype=np.int32))
            continue
        occ_xyz_set = set(map(tuple, occ_rows[:, :3].tolist()))
        keep = np.array([tuple(row[:3]) in occ_xyz_set for row in bbox_rows], dtype=bool)
        nusocc_frames.append(bbox_rows[keep])
    return nusocc_frames
```

(실사용 시 `set` 대신 numpy 구조화 배열 `np.isin` 류로 벡터화하면 훨씬 빠름 — 위는 로직을 명확히 보여주기 위한 참조 구현.)

이 정의는 **오늘의 `gt_occ_inst`와 완전히 동일**하다(annotation box ∩ 실점유, voxel 단위 trim). 다만 오늘은 이걸 만드는 스크립트가 없었는데, 이제 A와 C만 있으면 재현·재계산 가능하다.

---

## 6. 파생 레이어 2 — [저장 E] `bbox_aabb`(최종), [저장 F] `bbox_rot`(최종)

### 6.1 존재 게이트 (별도 저장 없음, `nusocc_inst`로부터 즉석 계산)

```python
def compute_keep_set(nusocc_frames):
    """(t, inst_id) 쌍 중 nusocc_inst에 row가 1개 이상 있는 것들의 집합."""
    keep_set = set()
    for t, rows in enumerate(nusocc_frames):
        if rows.shape[0] > 0:
            for inst_id in np.unique(rows[:, 4]):
                keep_set.add((t, int(inst_id)))
    return keep_set
```

### 6.2 게이트 적용 — A/B를 필터링 (박스는 안 깎음, row 단위 keep/drop만)

```python
def apply_gate(raw_frames, keep_set):
    """raw_bbox_aabb 또는 raw_bbox_rot에 keep_set 게이트 적용."""
    gated_frames = []
    for t, rows in enumerate(raw_frames):
        if rows.shape[0] == 0:
            gated_frames.append(rows)
            continue
        keep_mask = np.array([(t, int(iid)) in keep_set for iid in rows[:, 4]], dtype=bool)
        gated_frames.append(rows[keep_mask])
    return gated_frames

# 사용
keep_set = compute_keep_set(nusocc_inst_frames)          # §5의 D로부터
bbox_aabb_final = apply_gate(bbox_aabb_raw_frames, keep_set)   # 저장 E
bbox_rot_final  = apply_gate(bbox_rot_raw_frames,  keep_set)   # 저장 F  (OBB 재계산 없이 B를 그대로 재사용)
```

**핵심 효율성**: `bbox_rot_raw`(B)를 이미 §3.3에서 A와 동시에 계산해뒀기 때문에, 최종 F는 **OBB 수학을 다시 계산하지 않고** B에 같은 게이트만 적용하면 된다. 이게 가능한 이유는 집합 연산의 결합법칙 때문이다:

```
(raw_aabb ∩ OBB조건) ∩ 게이트  =  (raw_aabb ∩ 게이트) ∩ OBB조건
        [이 구현 방식]                  [수학적으로 동일한 다른 경로]
```

`keep_mask`는 "실점유 voxel"이 아니라 "**게이트를 통과한 instance의 전체 박스 voxel**"을 남긴다 — 박스 shape는 절대 깎이지 않는다(§1.2).

---

## 7. 출력 포맷 스펙 (기존 관례 100% 준수 — 반드시 지킬 것)

### 7.1 저장 루트 및 폴더 구조 — 확정

**`out_root = data/efficientocf_gt/`** — 6개 산출물 전부 이 하나의 루트 밑에 모은다(기존처럼 캐시마다 서로 다른 최상위 심볼릭 링크로 흩어지는 걸 지양).

```
data/efficientocf_gt/
└── GMO/
    ├── bbox_aabb_raw/            [저장 A, 미연결]   key: bbox_aabb_raw_saved_list2
    ├── bbox_rot_raw/             [저장 B, 미연결]   key: bbox_rot_raw_saved_list2
    ├── occ_raw/                  [저장 C, 미연결]   key: occ_raw_saved_list2
    │
    ├── segmentation_instance3d/  [저장 D = nusocc_inst, 연결★]  key: segmentation_instance_saved_list2
    ├── segmentation_aabb/        [저장 E = bbox_aabb 최종, 연결★] key: segmentation_aabb_saved_list2
    └── segmentation_rot/         [저장 F = bbox_rot 최종, 연결★]  key: segmentation_rot_saved_list2
```

`<scene>_<lidar>` = `present_scene_lidar_token`(§2.3, receptive field 마지막 프레임 기준) — 각 폴더 안 파일명은 `<scene>_<lidar>.npz`.

### 7.1.1 왜 D/E/F만 이름이 "이상한지" (`segmentation_instance3d`, `segmentation_aabb`, `segmentation_rot`)

**A/B/C는 이름에 제약이 없다** — 어디에도 연결하지 않으므로(§8) `bbox_aabb_raw`/`bbox_rot_raw`/`occ_raw`처럼 내용 그대로 깔끔하게 지었다("segmentation" 접두사는 불필요해서 뺐다).

**D/E/F는 이름이 강제된다.** 기존 로더의 경로 탐색 함수(`resolve_gt_occ_inst_dir`, `resolve_gt_bbox_aabb_dir`, `loading_instance.py:57-116`)와 eval의 rot lazy-loader(`efficientocf.py:2350-2394`)가 하위 폴더명을 **하드코딩된 문자열**로 찾는다:

```python
# resolve_gt_occ_inst_dir — "segmentation_instance3d" 리터럴을 후보로 붙임
candidates.append(os.path.join(gt_occ_inst_dataset_path, prefix, "segmentation_instance3d"))
# resolve_gt_bbox_aabb_dir — "segmentation_aabb" 리터럴
candidates.append(os.path.join(gt_bbox_aabb_dataset_path, prefix, "segmentation_aabb"))
# eval rot lazy-loader — ("aabb","segmentation_aabb"), ("rot","segmentation_rot") 튜플이 하드코딩
```

이 폴더명을 바꾸면 로더 코드(3개 함수)를 같이 고쳐야 하므로, **"config 경로만 바꾸면 코드 무변경으로 작동"이라는 이번 계획의 핵심 장점을 지키기 위해 D/E/F는 기존 이름을 그대로 유지한다.** 이름이 내용과 안 맞아 보이는 건(`segmentation_instance3d`인데 실제로는 nusocc 교집합 결과) 미관 문제일 뿐 기능엔 문제없다 — 필요하면 나중에 resolver 3개 함수에 후보 문자열 하나씩만 추가하는 작은 패치로 자유 명명으로 바꿀 수 있다(§12에 옵션으로 남겨둠).

### 7.2 npz 저장 (atomic, 압축) — `atomic_savez` 그대로 재사용

파일: `tools/gen_data/gen_bbox_gt_v2.py:71-74`

```python
def atomic_savez(path_no_ext, **kw):
    tmp = path_no_ext + f".tmp{os.getpid()}"
    np.savez_compressed(tmp, **kw)
    os.replace(tmp + ".npz", path_no_ext + ".npz")
```

`np.savez_compressed`가 `tmp+'.npz'`를 만들고 `os.replace`로 원자적 교체(중간에 프로세스가 죽어도 반쪽 파일이 최종 경로에 노출되지 않음). `tmp` 파일명에 `os.getpid()`를 넣어 병렬 shard 실행 시 충돌 방지.

⚠ **`tools/gen_data/verify_and_merge_occ_cls_inst.py`의 `save_merged_npz`는 이 atomic 패턴을 쓰지 않는다** — 참고하지 말 것.

### 7.3 저장 호출 및 key 이름

```python
# A/B/C — 자유 명명(내용 그대로)
atomic_savez(path_A, bbox_aabb_raw_saved_list2=np.array(bbox_aabb_raw_frames, dtype=object))
atomic_savez(path_B, bbox_rot_raw_saved_list2=np.array(bbox_rot_raw_frames, dtype=object))
atomic_savez(path_C, occ_raw_saved_list2=np.array(occ_raw_frames, dtype=object))

# D/E/F — 로더 기본 key 그대로(강제는 아니지만 폴더명과 맞춰 혼란 방지, §7.1.1과 동일 이유)
atomic_savez(path_D, segmentation_instance_saved_list2=np.array(nusocc_inst_frames, dtype=object))   # gt_occ_inst_key 기본값과 동일
atomic_savez(path_E, segmentation_aabb_saved_list2=np.array(bbox_aabb_final, dtype=object))           # gt_bbox_aabb_key 기본값과 동일
atomic_savez(path_F, segmentation_rot_saved_list2=np.array(bbox_rot_final, dtype=object))             # eval lazy-loader가 하드코딩(변경 불가)
```

A/B/C의 key는 `LoadInstanceWithFlow`가 아예 모르는 이름이라(연결 안 하므로) 완전히 자유롭다. D/E는 로더의 `gt_occ_inst_key`/`gt_bbox_aabb_key` 파라미터로 오버라이드가 **가능은 하지만**, 폴더명(§7.1.1에서 이미 강제됨)과 key명을 다르게 두면 헷갈리므로 그냥 기본값을 그대로 쓴다. F의 key(`segmentation_rot_saved_list2`)는 `efficientocf.py`의 eval lazy-loader가 리터럴로 하드코딩하고 있어 **오버라이드 자체가 불가능** — 반드시 이 이름이어야 한다.

최상위 컨테이너는 **`dtype=object`인 numpy array**(길이 T=`sequence_length`, 각 원소가 `[N,5]` 또는 `[M,4]`(C만) int32 배열).

### 7.4 행 포맷

- A, B, D, E, F: `[x, y, z, cls, instance_id]` — **5열**.
- C(`occ_raw`)만: `[x, y, z, cls]` — **4열, instance id 없음**.

기존 로더의 `_normalize_sparse_rows(min_cols=5)`는 5열 파일(A/B/D/E/F)에 바로 호환된다. C는 instance 개념이 없으므로 기존 `gt_occ_inst`류 로더 경로와는 다른 별도 처리가 필요하다(현재 로더에 4열 전용 소비 경로가 없음 — §8에서 "연결 안 함"으로 명시한 이유).

### 7.5 중복 좌표 처리

- `bbox_aabb_raw`/`bbox_aabb`류(인접 instance AABB가 겹칠 수 있음): **중복 좌표 허용** — 로더의 `_validate_gt_bbox_aabb_sparse_list`가 중복 체크 안 함.
- `nusocc_inst`(D)를 만약 기존 `gt_occ_inst` 로더 경로로 그대로 태우려면(§8): 그 로더(`_validate_sparse_rows_basic`)는 **중복 좌표를 에러로 취급**한다. 인접 두 instance의 AABB가 겹치는 영역에서 D를 만들 때 중복이 생기면(같은 좌표가 서로 다른 inst로 두 번 등장) 병합 규칙(나중 id가 덮어씀 등)이 필요할 수 있음 — §10의 열린 질문 참조.

---

## 8. 로딩 매핑 — 기존 config/로더에 어떻게 연결되는지

**D, E, F만 연결한다. A, B, C는 만들어서 디스크에 저장만 하고 로더에 연결하지 않는다.**

### 8.1 `nusocc_inst`(D) → `gt_occ_inst` 자리

```python
load_gt_occ_inst=True,
gt_occ_inst_dataset_path="./data/<새경로>/nusocc_inst/",   # 경로만 교체, 코드 무변경
```
포맷이 기존 `gt_occ_inst`(`[x,y,z,cls,inst]`, class_col=3)와 100% 동일하므로 로더 코드 수정 없이 그대로 작동한다.

### 8.2 `bbox_aabb`(E) → `gt_bbox_aabb`(dice 90%) 자리

```python
load_gt_bbox_aabb=True,
gt_bbox_aabb_dataset_path="./data/<새경로>/bbox_aabb/",   # 경로만 교체, 코드 무변경
```

### 8.3 `bbox_aabb`(E) → center/attn GT 자리 (`segmentation_instance3d` 대체) — **모델 코드 배선 변경 필요**

기존엔 `segmentation_instance3d`가 이 역할을 했는데, 새 설계에서는 이 캐시 자체를 로딩 안 하고(`load_segmentation_instance3d=False`) 이미 로드된 `gt_bbox_aabb`(=E)를 재사용한다. `efficientocf.py`에서 `gt_instance_centers_world`를 만드는 `build_instance_center_world_targets()` 호출부의 입력을 `segmentation_instance3d` 대신 `gt_bbox_aabb`로 바꿔야 한다 — **로더 자체는 손댈 필요 없고, 모델(detector) 쪽 배선 한 곳만 변경**.

### 8.4 `bbox_aabb`(E) → cls GT 자리 (`segmentation_cls_instance3d` 대체) — **모델 코드 배선 변경 필요**

`load_segmentation_cls_instance3d=False`로 끄고, `loss_query_cls`가 `gt_bbox_aabb`(=E)의 `cls` 컬럼(3번째, 0-indexed)을 직접 읽도록 배선 변경.

### 8.5 `bbox_rot`(F) → eval rot 자리

`bbox_rot`은 오늘도 `LoadInstanceWithFlow`를 안 거친다 — `efficientocf.py`의 `_eval_load_bbox_gt_v2()`라는 **eval 전용 lazy-loader**(`EOCF_BBOX_GT_V2_DIR` 환경변수, 기본 `./data/efficientocf_bboxcls_v2/GMO`)가 담당한다:

```python
root = os.environ.get("EOCF_BBOX_GT_V2_DIR", "./data/efficientocf_bboxcls_v2/GMO")
for name, sub in (("aabb", "segmentation_aabb"), ("rot", "segmentation_rot")):
    p = os.path.join(root, sub, key + ".npz")
    with np.load(p, allow_pickle=True) as z:
        frames = list(z[sub + "_saved_list2"])
```
새 F를 여기 그대로 넣으면 된다(경로만 교체, 코드 무변경). v3는 rot가 아예 없어서 train 중엔 못 썼는데, 새 파이프라인은 train split에도 F가 있으니 향후 학습에 활용할 수도 있다(지금 당장 계획엔 없어도 됨).

### 8.6 요약표

| 새 파일 | 연결 여부 | 어디에 | 필요한 작업 |
|---|---|---|---|
| A `bbox_aabb_raw` | ❌ | — | 저장만 |
| B `bbox_rot_raw` | ❌ | — | 저장만 |
| C `occ_raw` | ❌ | — | 저장만 |
| D `nusocc_inst` | ✅ | `gt_occ_inst` 자리 | config 경로만 교체 |
| E `bbox_aabb` | ✅ | `gt_bbox_aabb`(dice 90%) | config 경로만 교체 |
| E `bbox_aabb` | ✅ | center/attn GT | **모델 코드 배선 변경** |
| E `bbox_aabb` | ✅ | cls GT | **모델 코드 배선 변경** |
| F `bbox_rot` | ✅ | eval rot | env var 경로만 교체 |

---

## 9. 좌표 변환 공식 — 완전판 (실측 검증, ⚠ 기존 코드의 버그 1건 포함)

### 9.1 grid/해상도 상수 (모든 11개 baseline config에서 100% 동일, 실측 확인)

```python
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]           # grid_size
res = [(point_cloud_range[3+i] - point_cloud_range[i]) / occ_size[i] for i in range(3)]  # [0.2, 0.2, 0.2]
start = [point_cloud_range[i] + res[i] / 2.0 for i in range(3)]  # voxel index 0의 world 중심 좌표

time_receptive_field = 3
n_future_frames = 4       # sequence_length(표준) = 3+4 = 7
n_future_frames_plus = 6  # 3D/BEV 확장 윈도 = 3+6 = 9 (이번 파이프라인은 표준 7프레임 기준)
```

`class_names`/`query_class_ids`/`exclude_occ_class_ids`는 **기존 학습 파이프라인의 필터값**이지, 이번 새 GT 생성 파이프라인에는 적용하지 않는다(§0).

### 9.2 정변환(world → voxel index)

```
vox = round((world_pt - start + res/2) / res) = round((world_pt - pc_range_min) / res)
```
`get_poly_region()`(`loading_instance.py:317`)과 `rasterize_sequence()`(`gen_bbox_gt_v2.py:113`)가 완전히 동일한 공식(dtype만 int32 vs int64 차이).

### 9.3 역변환(voxel index → world, voxel "중심" 좌표) — 정확한 공식

```
world_center = vox*res + start - res/2   (= pc_range_min + vox*res)
```
`rasterize_sequence()`의 OBB 로컬프레임 재구성 코드(§3.3의 `centers_w = cand*res + start - res/2`)가 정확히 이 공식을 쓴다.

### 9.4 ⚠ 발견된 버그 — `build_instance_center_world_targets()`의 역변환 공식이 틀림 (기존 운영 코드, 새 파이프라인은 절대 재현하지 말 것)

파일: `projects/occ_plugin/datasets/pipelines/loading_instance.py:659` (현재 학습의 **center GT**를 계산하는 실제 운영 코드)

```python
centers_world = centers_voxel * self.resolution[:3].reshape(1, 1, 3) + self.start_position[:3].reshape(1, 1, 3)
```

이건 `vox*res + start`이지 `vox*res + start - res/2`가 아니다 — `-res/2` 보정항이 빠져 있다. `start = pc_range_min + res/2`이므로 이 공식은 실제로 `pc_range_min + vox*res + res/2`를 계산하는데, §9.3의 정확한 값(`pc_range_min + vox*res`)보다 **매 축마다 `+res/2`(=0.1m, 이 grid 기준) 만큼 체계적으로 어긋나 있다.**

**영향**: 이 함수가 만드는 값이 지금 학습에서 쓰는 `gt_instance_centers_world`(center/traj loss의 GT)다. 즉 지금 이 순간도 모든 학습 run의 center GT가 각 축마다 0.1m씩 밀려있다. 새 파이프라인의 center 계산 코드(§8.3의 배선 변경 시)는 §9.3의 올바른 공식을 써야 한다. (기존 운영 코드 자체를 지금 고칠지는 별도 판단 — 활성 학습 run들과의 단일변수 원칙을 감안해서 결정할 것.)

### 9.5 Box/Quaternion 좌표 변환 순서 (정확히, 4단계)

```python
box = Box(translation, size, Quaternion(rotation))   # global frame
box.translate(present_ego_translation)   # global -> ego
box.rotate(present_ego_rotation)
box.translate(present_ego2lidar_translation)   # ego -> lidar
box.rotate(present_ego2lidar_rotation)
pts = box.corners().T   # [8,3] lidar frame corners
```
`get_poly_region()`(`loading_instance.py:297-305`)과 `rasterize_sequence()`(`gen_bbox_gt_v2.py:99-104`)가 완전히 동일한 순서/부호를 쓴다. `present_ego_translation/rotation`, `present_ego2lidar_translation/rotation`은 §2.4에서 이미 부호 반전 + inverse quaternion이 적용된 값이므로 여기선 그대로 `+`로 translate/rotate하면 된다.

### 9.6 OBB(회전 박스) 내부 판정

```python
local = (world_center - box.center) @ box.rotation_matrix
half_extent = np.array([box.wlh[1]/2, box.wlh[0]/2, box.wlh[2]/2]) + res/2   # ⚠ w,l이 뒤바뀜, 원본 그대로
inside = np.all(np.abs(local) <= half_extent, axis=1)
```
`box.wlh`는 `[width,length,height]` 순서인데 half_extent엔 `[l/2,w/2,h/2]`로 뒤바뀌어 들어간다 — `gen_bbox_gt_v2.py:133`의 실제 코드가 이렇고, v2/v3가 이미 이 공식으로 검증된 정상 출력을 내고 있으므로(rot⊆aabb 위반 0건 실측) **그대로 재현할 것**(고쳐야 할 버그 아님).

---

## 10. 실행 스크립트 골격 (제안)

```python
# tools/gen_data/gen_new_gt_pipeline.py (제안 파일명)
import argparse, os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

CLS_MAP = ( ("vehicle.bicycle",2), ("bus",3), ("car",4), ("construction",5),
            ("motorcycle",6), ("trailer",9), ("truck",10) )

def category_to_nusocc_id(name, cls_map=CLS_MAP):
    for sub, cid in cls_map:
        if sub in name:
            return cid
    return None

def atomic_savez(path_no_ext, **kw):
    tmp = path_no_ext + f".tmp{os.getpid()}"
    np.savez_compressed(tmp, **kw)
    os.replace(tmp + ".npz", path_no_ext + ".npz")

# §3.1 record_instance_raw, §3.3 rasterize_sequence(aabb_only=False),
# §4.2-4.3 get_seq_occ / get_seq_occ_sparse (loading_occupancy.py에서 import 재사용 권장),
# §5 compute_nusocc_inst, §6 compute_keep_set/apply_gate 함수들을 여기 정의/import

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/baselines/full.py")
    ap.add_argument("--out", default="./data/efficientocf_gt")   # §7.1 확정 루트, 기존 v2/v3와 안 겹침
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--occ-path", default="./data/nuScenes-Occupancy")
    args = ap.parse_args()

    # dataset 빌드: gen_bbox_gt_v2.py main()의 골격 재사용
    #   dataset.classes = ["vehicle", "human"]                          # §3.1
    #   dataset.record_instance = <record_instance_raw 몽키패치>         # §3.1
    #   dataset.record_instance = <record_with_category로 한 번 더 감싸기>  # category_name 주입, gen_bbox_gt_v2.py:182-198 원문 재사용
    #   cls_map = CLS_MAP + (("pedestrian", 7),)                        # §3.2

    for i in range(args.start, end):
        # §2.3의 4개 상태 리셋
        seq = dataset.prepare_test_data(i)
        if seq is None:
            continue
        key = dataset.present_scene_lidar_token
        # 6개 파일 전부 있으면 skip (재실행 안전)

        bbox_aabb_raw, bbox_rot_raw = rasterize_sequence(seq, pc_range, grid_size, cls_map, aabb_only=False)  # A, B
        occ_raw = get_seq_occ_sparse(seq, occ_path=args.occ_path, grid_size=grid_size, pc_range=pc_range)  # C
        nusocc_inst = compute_nusocc_inst(bbox_aabb_raw, occ_raw)                                          # D
        keep_set = compute_keep_set(nusocc_inst)
        bbox_aabb_final = apply_gate(bbox_aabb_raw, keep_set)                                                 # E
        bbox_rot_final = apply_gate(bbox_rot_raw, keep_set)                                                   # F

        atomic_savez(path_A, bbox_aabb_raw_saved_list2=np.array(bbox_aabb_raw, dtype=object))
        atomic_savez(path_B, bbox_rot_raw_saved_list2=np.array(bbox_rot_raw, dtype=object))
        atomic_savez(path_C, occ_raw_saved_list2=np.array(occ_raw, dtype=object))
        atomic_savez(path_D, segmentation_instance_saved_list2=np.array(nusocc_inst, dtype=object))
        atomic_savez(path_E, segmentation_aabb_saved_list2=np.array(bbox_aabb_final, dtype=object))
        atomic_savez(path_F, segmentation_rot_saved_list2=np.array(bbox_rot_final, dtype=object))
        # path_A..path_F = <out>/GMO/<§7.1 폴더명>/<key>.npz (key는 확장자 없이 atomic_savez에 전달)

if __name__ == "__main__":
    main()
```

**병렬화**: `--start`/`--end`로 range를 나눠 shard 병렬 실행 가능(기존 v2/v3 관례 그대로). `--split train`으로 전체 학습셋(23,930키), `--split test`(기본)로 val(5,119키) 생성.

---

## 11. 검증 방법 (생성 후 반드시 실행할 것)

1. **id 정렬 검증**: A/B/D/E/F가 같은 윈도우 안에서 같은 id=같은 물체인지(voxel coverage로 확인).
2. **raw vs 최종 대조**: E가 A의 부분집합인지(row 단위, 절대 A에 없던 voxel이 E에 생기면 버그), D가 A∩C인지 직접 재계산해서 대조.
3. **생성소멸 확인**: 의도한 대로 미래-신규 물체가 A/B/D/E/F 전부에 포함되는지(꺼놨으므로 존재해야 정상).
4. **사람 포함 확인**: pedestrian이 실제로 cls=7로 존재하는지(construction_worker는 여전히 cls=5로 새는 게 정상, §3.2).
5. **게이트 효과 확인**: E가 A 대비 몇 %가 걸러지는지, 걸러진 (instance,frame)이 정말 D에 없는지 랜덤 샘플로 직접 대조.
6. **rot ⊆ aabb 검증**: F ⊆ E, B ⊆ A 둘 다 정의상 항상 성립해야 함(위반 있으면 구현 버그).
7. **좌표 회귀 테스트**: 몇 개 샘플에 대해 nuScenes raw annotation → box corner → voxel 좌표를 직접 계산해서 생성된 npz와 픽셀 단위로 일치하는지 대조(§9 공식 검증용).
8. **occ_raw 부피 확인**: C가 D보다 voxel 수가 훨씬 많은지(교집합 안 했으니 당연히 많아야 함 — 안 그러면 버그).

---

## 12. 알려진 리스크 / 열린 설계 질문 (구현 전 재확인 권장)

1. **`refine_instance_poly`(위치 동결/보간, §2.2) 유지 여부**: 명시적으로 언급되지 않았음. 노이즈 보정 로직이지 "제외 필터"가 아니므로 **기본은 유지 권장**, raw 원칙에 더 철저히 따르고 싶다면 끌 수도 있음.
2. **`visibility_token==1` 필터**를 생성소멸 필터와 함께 끌지: 이번 문서는 raw 원칙 일관성을 위해 **같이 끄는 것으로 §3.1에 작성**. 다르게 하고 싶으면 해당 continue 조건만 복원.
3. **construction_worker 우선매칭**(§3.2): 기본은 "끔". 켜고 싶으면 `cls_map` 순서만 바꾸면 됨(§3.2 참조).
4. **`nusocc_inst`(D)를 만약 기존 `gt_occ_inst` 로더 그대로 태우려 할 때, 인접 instance AABB가 겹치는 중복 좌표를 어떻게 처리할지**(§7.5): 우선순위 규칙(나중 id가 덮어씀 등)이 필요할 수 있음 — 기존 `gt_occ_inst`도 이런 미세 침식이 있다고 알려져 있으므로 완전히 새로운 문제는 아님.
5. **§9.4의 center 좌표 버그를 기존 운영 코드에서 지금 고칠지**: 이 명세서 범위 밖. 고치기로 하면 `loading_instance.py:659`에 `- self.resolution[:3].reshape(1,1,3)/2.0` 항 추가. 활성 학습 run들과의 단일변수 비교 문제 감안해서 판단할 것(이 레포의 과거 사고 이력 참조: 학습 도는 중 `datasets/pipelines` 코드 수정 시 dataloader worker가 죽을 위험 있음).
6. **`occ_raw`(C)의 class 컨벤션**(§4.3): 이번 버전은 `get_seq_occ()`의 기존 0/1-16/255 remap 컨벤션을 그대로 물려받음. 진짜 원본(remap 안 한) 값이 필요하면 별도 재구현 필요.
7. **grid_size/pc_range가 config마다 다를 가능성**: 이번 조사에서는 11개 baseline config 전부 동일했으나(§9.1), 다른 서버/다른 config를 쓴다면 반드시 그 config의 실제 값을 다시 확인할 것.
8. **D/E/F 폴더명(`segmentation_instance3d`/`segmentation_aabb`/`segmentation_rot`)을 깔끔한 이름으로 바꾸고 싶다면**(§7.1.1): `loading_instance.py`의 `resolve_gt_occ_inst_dir`(57-100행 부근)과 `resolve_gt_bbox_aabb_dir`(102-116행 부근)에 하드코딩 문자열 후보를 하나씩 더 추가하고, `efficientocf.py`의 `_eval_load_bbox_gt_v2`(2350-2394행 부근)의 `("aabb","segmentation_aabb"),("rot","segmentation_rot")` 튜플도 같이 고치면 된다 — 3개 함수 모두 "후보 추가"라 기존 캐시 호환은 안 깨짐. 이번 버전은 이 작업을 안 하는 쪽(기존 이름 유지)을 기본값으로 채택했음.

---

## 부록 A: 관련 기존 파일 전체 목록 (참고용)

| 파일 | 역할 |
|---|---|
| `tools/gen_data/gen_bbox_gt_v2.py` | bbox 계열 베이스 스크립트(거의 그대로 재사용) |
| `projects/occ_plugin/datasets/efficientocf_dataset.py` | `record_instance`/`get_indices`/`prepare_sequential_data`/pose 함수 원본 |
| `projects/occ_plugin/datasets/pipelines/loading_instance.py` | `get_poly_region`/`fill_occupancy`/좌표 상수 원본, 기존 로더 스키마 |
| `projects/occ_plugin/datasets/pipelines/loading_occupancy.py` | `get_seq_occ`(occupancy 로딩) 원본 |
| `projects/occ_plugin/utils/nusc_param.py` | nusocc class id ↔ 이름 매핑 |
| `data/nuScenes-Occupancy_inst3d/reports/summary.json` | 기존 외부 `gt_occ_inst` 생성 레시피(참고, 이 repo엔 스크립트 자체는 없음) |
| `tools/gen_data/verify_and_merge_occ_cls_inst.py` | ⚠ 참고만, atomic 저장 패턴은 따라하지 말 것(§7.2) |

## 부록 B: 이 문서의 검증 방법

이 문서의 모든 코드 인용과 공식은 2026-07-08 시점 실제 레포 코드를 직접 읽고, 일부는 npy/npz 파일을 python으로 직접 로드해서 실측 검증했다(§4.1, §7.3 dtype/shape 등). 다른 서버/다른 시점에 구현할 때는 §12.7과 같이 config 값이 이 문서 작성 시점과 동일한지 먼저 확인할 것.
