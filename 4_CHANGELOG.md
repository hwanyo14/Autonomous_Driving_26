# CHANGELOG

## 2026-05-15 13:34 KST
- `projects/occ_plugin/ops/occ_pooling`의 CUDA extension `occ_pool_ext` 부재로 학습 시작 시 import 실패가 발생한 것을 확인.
- `/home/hwanhee/anaconda3/envs/eo/bin/python setup.py build_ext --inplace`로 extension을 제자리 빌드.
- 빌드 후 `from projects.occ_plugin.ops.occ_pooling import occ_pool` import가 정상 동작하는 것까지 검증.

## 2026-05-15 13:35 KST
- `occ_pool_ext` 문제 해결 후 재실행하여 다음 blocker가 `lss_pretrained_ckpt_path` unexpected keyword 에러임을 확인.
- `projects/occ_plugin/occupancy/detectors/bevdepth.py`에서 해당 인자를 명시적으로 받아 상위 `CenterPoint.__init__`로 전달되지 않도록 수정.
- 이어서 동일 계열 config 키인 `freeze_lss_pretrained`, `lss_pretrained_strict`도 같은 방식으로 흡수하도록 보완.

## 2026-05-15 13:37 KST
- `projects/occ_plugin/occupancy/detectors/efficientocf.py`에서 legacy flat config 키를 `model_cfg`, `debug_cfg`, `visualization_cfg`로 선흡수하도록 정리.
- 현재 코드에서 실제로 사용하지 않는 legacy 키(`pretrain_*`, `query_feat_align_loss_weight`, `query_cam_proj_consistency_*`, `debug_loss_grad_*` 등)는 상위 detector init으로 전달되지 않게 제외.
- `timeout 25s bash train_total.sh` 재검증에서 모델 파라미터 생성과 분산 초기화, pretrained weight 로드, `use 3 past frames to forecast 4 future frames` 출력까지 확인.

## 2026-05-15 13:42 KST
- DDP 학습 중 `loss log variables are different across GPUs!` 에러를 확인.
- 원인은 일부 rank에서만 `dbg_query_match_cost_*` 로그 키가 생성되고 다른 rank에서는 아예 빠지는 분기였음.
- `projects/occ_plugin/occupancy/detectors/utils_loss.py`에서 Hungarian matching diagnostic 키들을 항상 0 tensor로 먼저 채우도록 수정해 rank별 `log_vars` 키 집합을 고정.
- `PORT=24536`으로 재검증했을 때 기존 `log_vars` assert는 재발하지 않았고, 학습이 `Start running` 및 runner hook 초기화 단계까지 정상 진입함을 확인.

## 2026-05-15 13:50 KST
- `query_debug_vis_no_pretrain` PNG 레이아웃을 조정.
- 기존 1행 `occ gt`는 제거하고, 기존 6행의 `GT class BEV`를 1행으로 이동.
- 마지막 행을 `traj` 전용으로 바꿔 `selected(high-score) query trajectory`를 화살표로, `matched GT trajectory`를 초록 화살표로 함께 표시하도록 수정.
- 새 시각화 데이터는 추가 로드 없이, 이미 계산된 query/GT center tensor를 `query_vis_bundle`로 재사용하도록 구현.

## 2026-05-15 13:55 KST
- 터미널에 계속 출력되던 `dbg/*` 로그를 숨기기 위해 `TextLoggerHookNoDbg`를 추가.
- `dbg/` 및 `dbg_` 태그는 텍스트 로그에서만 필터링하고, TensorBoard 기록은 기존 `TensorboardLoggerHookSplitTabs`로 그대로 유지.
- 8GPU baseline config의 logger hook을 `TextLoggerHook`에서 `TextLoggerHookNoDbg`로 교체.

## 2026-05-15 13:56 KST
- query trajectory 시각화 추가 후 `matched GT traj`가 CPU tensor로 넘어오면서 `_maybe_save_prob_grid_vis()` 내부 좌표 투영에서 `cuda:0 vs cpu` device mismatch가 발생한 것을 수정.
- `projects/occ_plugin/occupancy/dense_heads/query_head.py`의 `_project_points`, `_project_gaussians`에서 입력 tensor를 기준 device(`pc_min.device`)로 명시 정렬하도록 보완.

## 2026-05-15 14:01 KST
- query debug traj 행에서 selected query가 0개인 프레임에 대해 `np.stack([])`가 발생하던 예외를 수정.
- `selected traj` 색상 배열 생성 전에 개수를 확인하고, 0개면 traj 화살표 그리기를 건너뛰도록 보완.

## 2026-05-15 14:07 KST
- query debug PNG에서 기존 시각화 행을 복구하고 `traj` 행만 맨 아래에 추가하도록 재조정.
- 현재 row 순서는 `occ GT`, `class GT`, `all`, `selected`, `selected(class)`, `matched`, `traj`.
- 학습 시작 시 `cfg.model` 내부의 모든 `*_dir` 시각화 경로를 `work_dirs/<config명>/vis/<timestamp>/<원래폴더명>/` 아래로 자동 재매핑하도록 `tools/train.py`를 수정.

## 2026-05-15 14:08 KST
- `traj` 화살표 가독성을 높이기 위해 선 두께와 화살촉 크기를 키우고, 검은 외곽선을 추가.
- `selected query traj`는 밝은 청록색, `matched GT traj`는 라임색 고정으로 바꿔 배경 위에서 더 잘 보이게 조정.

## 2026-05-15 14:09 KST
- traj 화살촉을 채운 삼각형에서 `V`자 선 화살표 형태로 변경.
- 짧게 움직이는 trajectory에서도 끝점이 덜 가려지도록 조정.

## 2026-05-15 14:15 KST
- traj 화살표 본선과 선형 화살촉 두께를 추가로 증가시켜 가독성을 더 높임.

## 2026-05-15 14:17 KST
- traj 색상을 더 명확히 구분되도록 변경.
- `pred traj`는 주황-적색, `GT traj`는 밝은 시안 계열로 조정.

## 2026-05-15 14:20 KST
- `CODEX_DDP_4GPU_FIX.md`를 참고해 현재 레포 기준 DDP 호환성 수정을 반영.
- `tools/train.py`에서 distributed 초기화 전에 `torch.cuda.set_device(int(os.environ['LOCAL_RANK']))`를 추가.
- `projects/occ_plugin/occupancy/apis/mmdet_train.py`의 `MMDistributedDataParallel`에 `init_sync=False`, `static_graph=True`를 추가하고 기존 `find_unused_parameters`는 유지.
- `train_total.sh`를 일반화해 `config path`, `GPU count`, `PORT`를 인자로/환경변수로 바꿔 1~8 GPU 공용 실행기로 정리.
- 2GPU 스모크 실행(`PORT=24552 bash train_total.sh ./projects/configs/baselines/EfficientOCF_V1.1_8gpu.py 2`)에서 DDP init 이후 `Start running`까지 진입하는 것을 확인.

## 2026-05-29 04:34 UTC
- `dbg/*`, `dbg_*` metric이 터미널 텍스트 로그에 계속 노출되어 가독성을 해치던 문제를 정리.
- `projects/occ_plugin/core/evaluation/tensorboard_hooks.py`에 `TextLoggerHookNoDbg`를 복구해 텍스트/JSON 로그에서만 debug metric을 필터링하도록 적용.
- `projects/configs/baselines/EfficientOCF_V1.1_1gpu.py`, `projects/configs/baselines/EfficientOCF_V1.1_1gpu_traj_tf.py`의 text logger를 `TextLoggerHookNoDbg`로 교체하고, TensorBoard `dbg` 기록은 기존대로 유지.

## 2026-05-29 04:40 UTC
- DDP 학습에서 `loss log variables are different across GPUs!`가 다시 발생한 원인을 `dbg_query_match_cost_*` 상세 진단 키의 조건부 생성으로 확인.
- `projects/occ_plugin/occupancy/detectors/utils_loss.py`에서 Hungarian matching 상세 dbg 키들을 항상 0 tensor로 먼저 채우고, 실제 값이 있을 때만 덮어쓰도록 보완.
- rank별 `log_vars` 키 집합을 고정해 일부 rank에 GT/match가 없더라도 `_parse_losses` assert가 나지 않도록 정리.
