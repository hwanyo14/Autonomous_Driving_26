# NOTES

## 2026-05-15 13:34 KST
- 이번 에러의 직접 원인은 circular import라기보다 `projects/occ_plugin/ops/occ_pooling/occ_pool_ext*.so`가 없어서 상대 import가 실패한 상태였음.
- 학습/평가 전에 해당 `.so`가 사라졌거나 새 환경으로 옮겨졌다면 아래 명령으로 다시 빌드해야 함.

```bash
cd /home/hwanhee/Autonomous_Driving_26_ksh/projects/occ_plugin/ops/occ_pooling
/home/hwanhee/anaconda3/envs/eo/bin/python setup.py build_ext --inplace
```

- 빌드 시 `TORCH_CUDA_ARCH_LIST` 미설정 경고가 있었지만, 현재 머신의 visible GPU 아키텍처 기준으로는 정상 빌드됨.

## 2026-05-15 13:35 KST
- `projects/configs/baselines/EfficientOCF_V1.1_8gpu.py`에는 `lss_pretrained_ckpt_path=''`가 들어있지만, 현재 코드베이스에서는 이 값을 실제로 사용하는 로직이 보이지 않음.
- 지금 적용한 수정은 이 키를 무시하도록 막는 수준이며, 추후 LSS pretrained checkpoint 자동 로드를 원하면 별도 로직 추가가 필요함.
- 같은 맥락으로 `freeze_lss_pretrained`, `lss_pretrained_strict`도 현재는 저장만 하고 실제 동작에는 연결하지 않음.

## 2026-05-15 13:37 KST
- 현재 baseline config는 custom 키를 평평하게 `model=dict(...)`에 넣는 legacy 형태이고, detector 쪽에서 이를 흡수하지 않으면 `CenterPoint.__init__()` unexpected keyword 에러가 연쇄적으로 발생함.
- 이번에는 config 파일 전체 구조를 갈아엎지 않고 `EfficientOCF.__init__()`에서 legacy flat 키를 흡수하는 방식으로 호환성을 맞춤.
- 재검증 중 `The model and loaded state dict do not match exactly` 및 `unexpected key in source state_dict: fc.weight, fc.bias` warning이 있었지만, 이는 import/init blocker는 아니고 pretrained checkpoint 헤드 불일치 경고로 보임.
- 마지막 프로세스 종료는 학습 실패가 아니라 검증용 `timeout 25s`에 의해 발생한 SIGTERM임.

## 2026-05-15 13:42 KST
- 이번 DDP 에러는 loss 값 자체보다 `log_vars`의 키 개수가 rank마다 달라서 발생한 것임.
- `dbg_query_match_cost_*`는 matching 결과나 cost tensor 유효 여부에 따라 일부 rank에서만 추가되던 구조였고, mmcv distributed loss parser는 이 경우 바로 assert로 중단함.
- 해결은 조건부 로그 키를 없애는 게 아니라, 모든 rank에서 동일한 키를 항상 생성하고 값이 없을 때만 0으로 두는 방식으로 맞춤.
- 재검증은 기존 `24534` 포트가 이미 점유 중이어서 `PORT=24536`으로 직접 실행했음.
- 검증 종료는 다시 `timeout`에 의한 SIGTERM이며, 종료 전 `Start running` 로그까지 확인되어 적어도 기존 DDP `log_vars` 불일치 blocker는 해소된 것으로 판단.

## 2026-05-15 13:50 KST
- query debug PNG는 이제 `row1=GT class`, `row2=all queries`, `row3=selected`, `row4=selected(class-colored)`, `row5=matched`, `row6=traj` 순서임.
- `traj` 행은 시간 `t-1 -> t` 구간 화살표를 각 tile에 그리는 방식이라, score 높은 query의 이동 방향을 바로 볼 수 있게 했음.
- GT trajectory는 clutter를 줄이기 위해 `matched GT`만 초록 화살표로 오버레이함.
- 추가 I/O나 별도 데이터 로드는 넣지 않았고, detector가 이미 들고 있는 GT center trajectory를 visualization bundle로 넘겨 재사용함.

## 2026-05-15 13:55 KST
- 현재 `dbg` 값들은 loss dict에 계속 존재하고 TensorBoard에도 계속 남음. 다만 터미널 출력만 숨기도록 logger hook 레벨에서 처리했음.
- 즉, 디버그 지표를 완전히 비활성화한 게 아니라 `TextLoggerHook`의 화면 출력만 필터링한 것이라 추후 필요하면 TensorBoard에서 그대로 확인 가능.

## 2026-05-15 13:56 KST
- 새로 추가한 `matched GT trajectory` 시각화 tensor는 detector bundle 단계에서 CPU로 detach되어 넘어올 수 있음.
- 반면 `_maybe_save_prob_grid_vis()`의 기준 좌표(`pc_min`, `voxel_size`)는 보통 CUDA 위에 있으므로, 시각화용 좌표 투영 함수에서 입력 points/sigmas를 기준 device로 먼저 맞춰줘야 안전함.
- 이번 수정으로 query/GT traj 시각화 경로에서 CPU/CUDA 혼합 입력이 들어와도 바로 좌표 투영 가능하도록 정리함.

## 2026-05-15 14:01 KST
- `traj` 행은 모든 프레임에서 selected query가 존재한다고 가정하면 안 됨.
- score/topk 기준 때문에 어떤 프레임에서는 selected query 수가 0일 수 있고, 이 경우 class-color 배열 생성에서 `np.stack([])`가 바로 터질 수 있음.
- 현재는 0개 프레임이면 해당 타일에서 selected traj만 생략하고, 다른 행/GT traj는 계속 저장되도록 처리함.

## 2026-05-15 14:07 KST
- 사용자 요구에 맞춰 query debug PNG는 기존 row들을 유지해야 하므로, 이전에 제거했던 `occ GT` row를 다시 복구함.
- `traj`는 기존 row를 대체하는 것이 아니라 마지막 1개 row를 추가하는 방식으로 유지.
- 시각화 저장 경로는 config 파일에 박힌 개별 경로를 직접 손대지 않고, train 시작 시점에 `cfg.model` 내부 `*_dir` 값을 일괄 재매핑하는 방식으로 통일함.
- 따라서 실행마다 시각화는 `work_dirs/<config명>/vis/<timestamp>/query_debug_vis_no_pretrain`, `.../instance_img_debug_vis_no_pretrain` 같은 하위 폴더들로 자동 정리됨.

## 2026-05-15 14:08 KST
- traj 행은 클래스색 그대로 쓰면 배경이나 다른 overlay와 섞여 잘 안 보일 수 있어, motion 가독성 우선으로 고정색을 쓰는 편이 안정적임.
- 현재 traj 색상은 `query=밝은 청록`, `GT=라임`이고, 검은 외곽선이 먼저 깔린 뒤 색 선이 그려지는 방식이라 어두운/밝은 배경 모두에서 더 잘 보이게 함.

## 2026-05-15 14:09 KST
- 채워진 삼각형 화살촉은 짧은 이동 벡터에서 종점 주변을 너무 많이 가려 가독성이 떨어질 수 있음.
- 현재는 끝점에서 양 갈래 선 두 개를 그리는 `선 화살표(V형 head)`로 바꿔, 작은 움직임도 더 읽기 쉽게 함.

## 2026-05-15 14:15 KST
- 현재 traj 화살표는 외곽선 포함 굵은 본선과 더 두꺼운 선형 head를 사용하도록 조정함.
- 이후에도 더 키우려면 같은 지점에서 `line width`만 올리면 되므로 수정 범위는 매우 작음.

## 2026-05-15 14:17 KST
- traj 색상 대비를 키우기 위해 `pred=주황/적색`, `GT=밝은 시안`으로 재조정함.
- 두 색은 밝기와 hue 차이가 커서 겹쳐도 이전보다 훨씬 구분이 쉬움.

## 2026-05-15 14:20 KST
- 옛 4GPU fix 문서의 핵심은 `LOCAL_RANK -> cuda device` 명시와 DDP constructor 옵션(`init_sync=False`, `static_graph=True`)이었고, 현재 레포에도 그대로 의미가 있음.
- 현재 baseline은 `samples_per_gpu=1`이라 GPU 수를 1~8로 바꿔도 per-rank batch shape 자체는 유지되므로, 우선 DDP 안정성 쪽만 고치면 범용성이 가장 높음.
- 검증 기준은 2GPU 짧은 스모크로 잡았고, 최소한 현재 레포에서는 DDP wrap/NCCL 초기화 단계에서 죽지 않고 runner `Start running`까지 진입함을 확인.
- 다만 NCCL이 `device used by this process is currently unknown` warning은 여전히 찍을 수 있음. 이번 수정 범위에서는 crash 제거와 1~8 GPU 공용 실행 호환성을 우선했음.
- 이제 실행 예시는 아래처럼 됨:

```bash
bash train_total.sh
bash train_total.sh ./projects/configs/baselines/EfficientOCF_V1.1_8gpu.py 1
bash train_total.sh ./projects/configs/baselines/EfficientOCF_V1.1_8gpu.py 4
PORT=24560 bash train_total.sh ./projects/configs/baselines/EfficientOCF_V1.1_8gpu.py 8
```
