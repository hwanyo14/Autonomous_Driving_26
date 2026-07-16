# Project Structure

```text
.
├── .gitignore  # 외부 저장소, 데이터셋, 체크포인트, work_dirs, 캐시 산출물 무시 규칙
├── AGENTS.md  # 이 repo에서 작업할 때 따라야 할 에이전트 지침
├── CLAUDE.md  # Claude Code용 작업 지침 (수정 원칙, 기록 규칙)
├── README.md  # 저장소 목적을 짧게 적은 루트 소개
├── PROJECT_STRUCTURE.md  # 이 문서
├── CHANGELOG.md  # 코드 수정/구현 기록
├── DATA_PREPROCESSING.md  # GT 캐시(efficientocf_gt/_f3) 전처리 절차 — 실행 파일·명령·순서·검증 기록 (재현용, 2026-07-13 확정)
├── NOTES.md  # 구현 중 주의사항·경고·후속 작업 메모
├── GUIDE.pdf  # 실험/사용 가이드 문서
├── run.sh  # 학습용 진입 스크립트; config 확인 후 tools/dist_train.sh 호출
├── run_eval.sh  # 평가용 진입 스크립트; config/checkpoint 확인 후 tools/dist_test.sh 호출
├── train_total.sh  # 8GPU 학습 원클릭 래퍼; CUDA_VISIBLE_DEVICES 지정 후 dist_train.sh 호출
├── eval_total.sh  # 단일 eval 원클릭 래퍼; CONFIG/checkpoint/GPU 값만 수정 후 dist_test.sh 호출
├── eval_oracle.sh  # Oracle(GT 치팅) eval 래퍼; EOCF_EVAL_ORACLE_MATCH=1로 query↔GT center Hungarian 선택 → 스코어링 병목 진단
├── eval_sweep_thr.sh  # (FG_THR,OCC_THR) 조합 순차 스윕 래퍼; 부분 eval(EOCF_EVAL_MAX_SAMPLES=512)로 짧게 비교, 타 eval 종료 대기 후 시작
├── data_vis/  # GT 파이프라인 검증 BEV 시각화 PNG (fig1~7=기존 캐시 검증, fign1~3=새 파이프라인 샘플; NOTES 2026-07-10 검증 섹션 참조)
├── data/  # 외부 데이터와 전처리 캐시를 가리키는 심볼릭 링크 모음
│   ├── efficientocf -> /home/user/jhh/Projects/EfficientOCF/data/efficientocf  # OCF instance/flow 전처리 캐시
│   ├── efficientocf_bboxcls -> /home/user/jhh/Projects/EOCF_qg_distil_dev/data/efficientocf_bboxcls  # bbox/class 기반 segmentation 캐시
│   ├── efficientocf_bboxcls_v2/  # (로컬 생성) val 전용 bbox GT v2 — GMO/segmentation_{aabb,rot}/ [x,y,z,cls,inst]; gen_bbox_gt_v2.py 산출물 (ped 제외 번호 — loss GT로 쓰지 말 것)
│   ├── efficientocf_bboxcls_v3/  # (로컬 생성) train 23,930 AABB GT — GMO/segmentation_aabb/ [x,y,z,cls,inst]; cache-parity id(ped 포함, ['vehicle','human']) — focal3d bbox 혼합 loss용 (NOTES 2026-07-06 id 체계 참조)
│   ├── nuScenes-Occupancy -> /home/user/jhh/Datasets/nuScenes-Occupancy-v0.1/  # nuScenes occupancy GT
│   ├── nuScenes-Occupancy_inst3d -> /home/user/jhh/Projects/EOCF_qpa_gmms_consist_aux_occ/data/nuScenes-Occupancy_inst3d  # GT 3D instance occupancy 캐시
│   ├── nuscenes -> /home/user/jhh/Datasets/nuscenes/  # nuScenes 원본 데이터
│   └── occ_dt -> /home/user/jhh/Datasets/occ_dt  # distance transform field 캐시
├── projects/
│   ├── __init__.py  # projects 패키지 마커
│   ├── configs/
│   │   ├── _base_/
│   │   │   ├── default_runtime.py  # checkpoint/logging/distributed/runtime 기본 설정
│   │   │   └── schedules/
│   │   │       ├── cosine.py  # CosineAnnealing 40epoch 스케줄 템플릿
│   │   │       ├── cyclic_20e.py  # cyclic LR 20epoch 스케줄 템플릿
│   │   │       ├── cyclic_40e.py  # cyclic LR 40epoch 스케줄 템플릿
│   │   │       ├── mmdet_schedule_1x.py  # MMDet 스타일 1x step scheduler 템플릿
│   │   │       ├── schedule_2x.py  # nuScenes 계열 24epoch step scheduler 템플릿
│   │   │       ├── schedule_3x.py  # 36epoch step scheduler 템플릿
│   │   │       ├── seg_cosine_150e.py  # segmentation용 cosine 150epoch 템플릿
│   │   │       ├── seg_cosine_200e.py  # segmentation용 cosine 200epoch 템플릿
│   │   │       └── seg_cosine_50e.py  # segmentation용 cosine 50epoch 템플릿
│   │   ├── baselines/  # 2026-07-13 기준 전부 새 GT(efficientocf_gt_f3) 배선 — 구버전 config 8개(full/subset/subset_scale*/subset_attn_cover{,_size,_aabb}) 삭제됨 (CHANGELOG 2026-07-13)
│   │   │   ├── subset_attn.py  # subset(4000) + camera-attn 충돌 fix. GT는 gt_bbox_aabb(E) 기반 (NOTES/CHANGELOG 2026-07-02, 2026-07-13)
│   │   │   ├── subset_attn_cover_aabb_dice.py  # subset_attn + σ-matching + dice(tversky) GT 혼합(0.1 inst3d+0.9 AABB, 2D z-collapse) (CHANGELOG 2026-07-07, 2026-07-13)
│   │   │   ├── subset_attn_cover_pyr_aabb_dice3d.py  # 위 + dice 3D(query_gmo_dice_3d=True) + query cross-attn coarse-to-fine KV 피라미드(3-layer) (CHANGELOG 2026-07-10, 2026-07-13)
│   │   │   ├── full_attn_cover_pyr_aabb_dice3d.py  # subset_attn_cover_pyr_aabb_dice3d와 동일 조합, train_capacity=23930(전체) (CHANGELOG 2026-07-13)
│   │   │   └── newgt_f3_aabb_dice.py  # subset_attn_cover_aabb_dice의 새 GT 배선 프로토타입(최초 검증본, model_cfg 동일) (CHANGELOG 2026-07-12)
│   │   └── datasets/
│   │       └── custom_nus-3d.py  # MMDet3D 기반 nuScenes 3D dataset/pipeline 기본 템플릿
│   └── occ_plugin/  # mmdetection3d plugin 진입점; datasets/models/hooks/ops 등록
│       ├── __init__.py  # plugin 서브모듈 import로 registry 등록 트리거
│       ├── core/
│       │   ├── __init__.py  # core 하위 공개 심볼 묶음
│       │   ├── evaluation/
│       │   │   ├── __init__.py  # custom eval/tensorboard/efficiency hook export
│       │   │   ├── efficiency_hooks.py  # 효율 측정 hook과 query center gradient debug hook
│       │   │   ├── eval_hooks.py  # occupancy용 single/distributed evaluation hook
│       │   │   └── tensorboard_hooks.py  # TensorBoard 태그를 train/dbg 탭으로 분리하는 logger hook
│       │   └── visualizer/
│       │       ├── __init__.py  # occupancy 시각화 유틸 export
│       │       └── show_occ.py  # pred/gt occupancy를 npy로 저장하는 시각화 유틸
│       ├── datasets/
│       │   ├── __init__.py  # custom dataset builder와 dataset class 등록
│       │   ├── builder.py  # custom dataloader/dataset builder와 worker seed 초기화
│       │   ├── efficientocf_dataset.py  # nuScenes용 시퀀스 occupancy forecasting dataset
│       │   ├── efficientocf_lyft_dataset.py  # Lyft용 시퀀스 occupancy forecasting dataset
│       │   ├── nuscenes_dataset.py  # 카메라 intrinsics/extrinsics와 temporal queue를 붙인 NuScenesDataset 확장
│       │   ├── pipelines/
│       │   │   ├── __init__.py  # pipeline component 등록
│       │   │   ├── formating.py  # occupancy 입력용 format bundle 정의
│       │   │   ├── loading_bevdet.py  # 멀티뷰 이미지, depth GT, BEVDet/BEVDepth 증강 로더
│       │   │   ├── loading_instance.py  # instance/flow/segmentation 캐시와 GT center를 읽는 로더
│       │   │   ├── loading_occupancy.py  # 현재 사용 occupancy 로더; DT field, height cache, 순차 occupancy 처리
│       │   │   ├── loading_occupancy_ori.py  # 예전 occupancy 로더 보존본
│       │   │   └── transform_3d.py  # pad/normalize/collect/scale 등 3D pipeline transform 모음
│       │   └── samplers/
│       │       ├── __init__.py  # sampler registry export
│       │       ├── distributed_sampler.py  # eval 쪽 non-shuffle distributed sampler
│       │       ├── group_sampler.py  # train 쪽 distributed group sampler
│       │       └── sampler.py  # sampler registry/build helper
│       ├── occupancy/
│       │   ├── __init__.py  # occupancy 하위 모듈 import 진입점
│       │   ├── apis/
│       │   │   ├── __init__.py  # custom train API export
│       │   │   ├── mmdet_train.py  # runner/dataloader/eval hook를 custom wiring하는 학습 API
│       │   │   ├── test.py  # custom single/multi GPU 평가 루프와 결과 수집 로직
│       │   │   └── train.py  # 모델 타입에 따라 custom_train_detector/train_detector를 선택하는 래퍼
│       │   ├── backbones/
│       │   │   ├── __init__.py  # 2D/3D backbone과 predictor 등록
│       │   │   ├── pred_block.py  # 미래 프레임 feature 예측용 residual predictor 블록
│       │   │   └── resnet3d.py  # occupancy branch용 custom 2D/3D ResNet backbone
│       │   ├── dense_heads/
│       │   │   ├── __init__.py  # occ/height/flow head 등록
│       │   │   ├── flow_head.py  # instance flow 예측 head와 flow loss
│       │   │   ├── height_head.py  # height map 예측 head와 height loss
│       │   │   ├── lovasz_softmax.py  # Lovasz/Jaccard 계열 segmentation loss 구현
│       │   │   ├── occ_head.py  # occupancy voxel 분류 head와 CE/geo/sem/lovasz loss
│       │   │   ├── query_head.py  # 현재 query 기반 instance head; center/sigma/class/depth/trajectory 예측
│       │   │   ├── query_head_ori.py  # 예전 query head; center+offset 기반 point set 예측 버전
│       │   │   ├── utils.py  # torchsparse 기반 point/voxel/range 변환 유틸
│       │   │   └── voxelizer.py  # query point/Gaussian을 occupancy grid로 바꾸는 soft voxelizer
│       │   ├── detectors/
│       │   │   ├── __init__.py  # 현재 detector 등록
│       │   │   ├── bevdepth.py  # BEVDet/BEVDepth 계열 detector base 구현
│       │   │   ├── efficientocf.py  # 현재 주력 EfficientOCF detector; mixin 조합형 구현
│       │   │   ├── efficientocf_config.py  # model/debug/visualization config default와 apply helper
│       │   │   ├── efficientocf_ori.py  # 예전 monolithic EfficientOCF detector 구현 보존본
│       │   │   ├── utils_bev_pool.py  # query/GT instance feature의 Gaussian BEV pooling 유틸
│       │   │   ├── utils_dbg.py  # detector 내부 디버그 출력/상태 추적 유틸
│       │   │   ├── utils_geometry.py  # detector용 좌표/궤적/egomotion 기하 보조 유틸
│       │   │   ├── utils_gt_prep.py  # query supervision용 GT center/id/class/occupancy 패킹 유틸
│       │   │   ├── utils_instance_img_debug.py  # instance 단위 이미지 디버그 시각화 유틸
│       │   │   ├── utils_loss.py  # query/GMO/DT/trajectory 관련 loss 계산 유틸
│       │   │   ├── utils_matcher.py  # Hungarian matching과 매칭 cost 계산 유틸
│       │   │   ├── utils_query_projection.py  # query attention/depth를 카메라 평면에 투영하고 soft-lift하는 유틸
│       │   │   └── utils_visualization.py  # BEV/query/GT alignment 디버그 시각화 유틸
│       │   ├── fuser/
│       │   │   ├── __init__.py  # fusion layer 등록
│       │   │   ├── addfuse.py  # camera/lidar voxel feature를 가중합으로 fusion
│       │   │   ├── convfuse.py  # camera/lidar voxel feature concat 후 conv fusion
│       │   │   └── visfuse.py  # visibility weight를 예측해 fusion하는 모듈
│       │   ├── image2bev/
│       │   │   ├── __init__.py  # view transformer 등록
│       │   │   ├── ViewTransformerLSSBEVDepth.py  # LSS+BEVDepth 기반 depth/view transform 핵심 구현
│       │   │   ├── ViewTransformerLSSVoxel.py  # voxel occupancy용 LSS transformer와 depth loss 구현
│       │   │   └── transformer.py  # camera/time cross-attention으로 query를 모으는 transformer 모듈; kv_resolutions 지정 시 레이어별 coarse-to-fine KV 해상도 피라미드 지원(2026-07-10 이식)
│       │   ├── necks/
│       │   │   ├── __init__.py  # 3D FPN 계열 neck 등록
│       │   │   ├── fpn3d.py  # 다중 해상도 3D feature를 합치는 일반 FPN
│       │   │   └── second_fpn_3d.py  # SECOND 스타일 3D deconv neck
│       │   └── voxel_encoder/
│       │       ├── __init__.py  # sparse lidar encoder 등록
│       │       └── sparse_lidar_enc.py  # spconv 기반 sparse 3D lidar encoder
│       ├── ops/
│       │   ├── __init__.py  # custom op export
│       │   └── occ_pooling/
│       │       ├── __init__.py  # occ_pool export
│       │       ├── OCC_Pool.py  # Python autograd 래퍼와 CUDA extension 호출부
│       │       ├── setup.py  # occ_pool CUDA extension 빌드 스크립트
│       │       └── src/
│       │           ├── occ_pool.cpp  # PyTorch extension binding과 forward/backward C++ entry
│       │           └── occ_pool_cuda.cu  # occupancy pooling CUDA kernel 구현
│       └── utils/
│           ├── __init__.py  # 공용 metric/geometry/format helper export
│           ├── coordinate_transform.py  # coarse-to-fine 좌표 확장과 3D point의 이미지 투영 유틸
│           ├── formating.py  # IoU/EPE 결과를 pretty table로 포맷하는 유틸
│           ├── gaussian.py  # heatmap/depth supervision용 Gaussian 생성 유틸
│           ├── geometry.py  # egopose 변환행렬 생성/역변환 유틸
│           ├── metric_util.py  # confusion matrix와 SSC metric 계산 유틸
│           ├── nusc_param.py  # nuScenes class 메타정보와 occupancy loss 유틸
│           ├── semkitti.py  # SemanticKITTI 계열 loss 유틸 모음
│           └── voxel_to_points.py  # voxel 예측을 point prediction으로 근사 매핑하는 유틸
└── tools/
    ├── dist_test.sh  # torch.distributed.run 기반 분산 평가 실행기
    ├── dist_train.sh  # torch.distributed.run 기반 분산 학습 실행기
    ├── test.py  # config/checkpoint 로드 후 custom test API를 호출하는 평가 엔트리포인트
    ├── train.py  # config 로드, plugin import, runner 구성 후 custom train API를 호출하는 학습 엔트리포인트
    ├── dbg_probe/  # 체크포인트 오프라인 진단 프로브 모음 (spread/feature/attn/BEV 크기 분석 + BEV 비교 플롯); 사용법은 내부 README.md
    ├── gen_data/
    │   ├── gen_bbox_gt_v2.py  # bbox GT 재생성기 (AABB+rotated OBB, [x,y,z,cls,inst], 생성소멸·사람 필터 상속). --split train/--aabb_only/--include_ped_ids(cache-parity id)로 v3 train 캐시도 생성
    │   ├── gen_depth_gt.py  # nuScenes lidar를 카메라로 투영해 depth GT bin 파일 생성
    │   ├── derive_filtered_gt.py  # raw D/E/F에서 f3 필터판 파생 (past3all/keep-human 등, row 필터만·id 재부여 없음). 0630 레포에서 무수정 복사(md5 동일)
    │   ├── gen_new_gt_pipeline.py  # 새 GT 캐시(A/B/D/E/F) 통합 생성기 (NEW_GT_PIPELINE_SPEC.md). ⚠ compute_D에 이물질 흡수 버그 있음 — 수정판은 regen_def_clsmatch.py
    │   ├── regen_def_clsmatch.py  # D/E/F 재생성기 (compute_D class-match 수정판, --no-f3-filter로 raw 모드). _gt의 A/B 재사용, D/E/F만 저장 (CHANGELOG 2026-07-13)
    │   └── verify_and_merge_occ_cls_inst.py  # sparse class/instance occupancy 캐시 정합성 검증과 병합 도구
    └── misc/
        ├── browse_dataset.py  # dataset pipeline을 시각적으로 브라우징하는 도구
        ├── fuse_conv_bn.py  # 추론용으로 Conv-BN을 fuse한 checkpoint 생성 도구
        ├── print_config.py  # mmcv config를 병합 후 pretty print하는 도구
        ├── smoke_past_frame_matching.py  # query matching/GT packing/projection 관련 smoke test 스크립트
        └── visualize_results.py  # 저장된 결과 pickle을 dataset.show로 렌더링하는 도구

```
