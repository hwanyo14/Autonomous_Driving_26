# Developed by Jingyi Xu based on the codebase of Cam4DOcc and PowerBEV
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF
#
# [occ 실험] semantic cls를 binary {0=bg, 1=fg}로 통합한 config (car/truck/... 구분 제거).
# 변경 방법/디버깅 가이드: 레포 루트의 BINARY_FG_CLS_CHANGES.md 참고.
# 스위치는 model_cfg의 query_cls_binary_fg=True (+ num_classes=2, weights 2개). moving/static과 무관.

# Basic params ******************************************
_base_ = ['../datasets/custom_nus-3d.py', '../_base_/default_runtime.py']

find_unused_parameters = True
# Whether to run only dataset generation without training.
only_generate_dataset = False

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

plugin = True
plugin_dir = "projects/occ_plugin/"

# Dataset paths ******************************************
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
depth_gt_path = './data/depth_gt'
ocf_dataset_path = "./data/efficientocf/"

# User-local dataset roots.
occ_path = "./data/nuScenes-Occupancy"
nusc_root = './data/nuscenes/'
occ_dt_path = "./data/occ_dt"
segmentation_cls_dataset_path = "./data/efficientocf_bboxcls/"
gt_occ_inst_dataset_path = "./data/nuScenes-Occupancy_inst3d/"

# Query/GMO foreground semantic classes use sparse raw nuScenes occupancy ids:
# [2, 3, 4, 5, 6, 9, 10] + background(0); pedestrian(7) excluded
class_names = [
    'bicycle',
    'bus',
    'car',
    'construction',
    'motorcycle',
    'trailer',
    'truck',
]
query_class_ids = [0, 2, 3, 4, 5, 6, 9, 10]
query_class_names = ['background'] + class_names
exclude_occ_class_ids = (7,)  # pedestrian: 로드 단계에서 제거 (nohuman)
validate_segmentation_cls_instance3d_alignment = True
strict_query_class_id_validation = True
use_separate_classes = False
use_fine_occ = False

# Forecasting-related params ******************************************
# Use time_receptive_field past frames to forecast n_future_frames.
# For 3D instance prediction, n_future_frames_plus should be > n_future_frames.
time_receptive_field = 3
n_future_frames = 4
n_future_frames_plus = 6

# Query present-only mode: converts the query branch output from the 6-frame
# plus window to a single present frame (global index = time_receptive_field - 1).
# BEV branch stays on n_future_frames_plus.
query_present_only = True
query_pred_num_frames = 1

# Occupancy-related params ******************************************
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
empty_idx = 0
if use_separate_classes:
    num_cls = len(class_names) + 1
else:
    num_cls = 2

img_norm_cfg = None

# Data-generation and pipeline params ******************************************
dataset_type = 'EfficientOCFDataset'
file_client_args = dict(backend='disk')
data_config = {
    'cams': [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ],
    'Ncams': 6,
    'input_size': (896, 1600),
    'src_size': (900, 1600),
    # Image-view augmentation.
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

# train_capacity = 23930  # default: use all sequences
train_capacity = 4000  # 3880
# train_capacity = 7  # 3880

test_capacity = 5119  # default: use all sequences
validate_instance_cache = False
write_instance_cache = False
write_height_cache = False

train_pipeline = [
    dict(
        type='LoadInstanceWithFlow',
        ocf_dataset_path=ocf_dataset_path,
        grid_size=occ_size,
        use_flow=False,
        background=empty_idx,
        pc_range=point_cloud_range,
        use_separate_classes=use_separate_classes,
        validate_cache=validate_instance_cache,
        write_cache=write_instance_cache,
        load_segmentation_instance3d=True,
        load_segmentation_cls_instance3d=True,
        segmentation_cls_dataset_path=segmentation_cls_dataset_path,
        validate_segmentation_cls_instance3d_alignment=validate_segmentation_cls_instance3d_alignment,
        load_gt_occ_inst=True,
        gt_occ_inst_dataset_path=gt_occ_inst_dataset_path,
        exclude_occ_class_ids=exclude_occ_class_ids,
    ),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=data_config,
        sequential=False,
        aligned=True,
        trans_only=False,
        depth_gt_path=depth_gt_path,
        data_root=nusc_root,
        mmlabnorm=True,
        load_depth=True,
        img_norm_cfg=img_norm_cfg,
    ),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        occ_path=occ_path,
        ocf_dataset_path=ocf_dataset_path,
        grid_size=occ_size,
        unoccupied=empty_idx,
        pc_range=point_cloud_range,
        use_fine_occ=use_fine_occ,
        test_mode=False,
        dt_path=occ_dt_path,
        load_occ_dt=True,
        dt_type='float16',
        strict_dt=True,
        validate_height_cache=False,
        write_height_cache=write_height_cache,
        exclude_occ_class_ids=exclude_occ_class_ids,
    ),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs_seq',
            'gt_occ',
            'future_egomotion',
            'segmentation',
            'segmentation_bev',
            'instance_bev',
            'segmentation_instance3d',
            'segmentation_cls_instance3d',
            'gt_occ_inst',
            'gt_instance_centers_world',
            'gt_instance_centers_valid',
            'gt_instance_ids',
            'occ_dt',
        ],
        meta_keys=(
            'filename',
            'ori_shape',
            'img_shape',
            'lidar2img',
            'depth2img',
            'cam2img',
            'pad_shape',
            'scale_factor',
            'flip',
            'pcd_horizontal_flip',
            'pcd_vertical_flip',
            'box_mode_3d',
            'box_type_3d',
            'img_norm_cfg',
            'pcd_trans',
            'sample_idx',
            'pcd_scale_factor',
            'pcd_rotation',
            'pcd_rotation_angle',
            'pts_filename',
            'transformation_3d_flow',
            'trans_mat',
            'affine_aug',
            'scene_token',
            'lidar_token',
            'pc_range',
            'occ_size',
        ),
    ),
]

test_pipeline = [
    dict(
        type='LoadInstanceWithFlow',
        ocf_dataset_path=ocf_dataset_path,
        grid_size=occ_size,
        use_flow=False,
        background=empty_idx,
        pc_range=point_cloud_range,
        use_separate_classes=use_separate_classes,
        validate_cache=validate_instance_cache,
        write_cache=write_instance_cache,
        load_segmentation_instance3d=True,
        load_segmentation_cls_instance3d=True,
        segmentation_cls_dataset_path=segmentation_cls_dataset_path,
        validate_segmentation_cls_instance3d_alignment=validate_segmentation_cls_instance3d_alignment,
        load_gt_occ_inst=True,
        gt_occ_inst_dataset_path=gt_occ_inst_dataset_path,
        exclude_occ_class_ids=exclude_occ_class_ids,
    ),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=data_config,
        depth_gt_path=depth_gt_path,
        data_root=nusc_root,
        sequential=False,
        aligned=True,
        trans_only=False,
        mmlabnorm=True,
        img_norm_cfg=img_norm_cfg,
        test_mode=True,
    ),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        occ_path=occ_path,
        ocf_dataset_path=ocf_dataset_path,
        grid_size=occ_size,
        unoccupied=empty_idx,
        pc_range=point_cloud_range,
        use_fine_occ=use_fine_occ,
        test_mode=True,
        dt_path=occ_dt_path,
        load_occ_dt=True,
        dt_type='float16',
        strict_dt=True,
        validate_height_cache=False,
        write_height_cache=write_height_cache,
    ),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs_seq',
            'gt_occ',
            'future_egomotion',
            'segmentation',
            'segmentation_bev',
            'instance_bev',
            'segmentation_instance3d',
            'segmentation_cls_instance3d',
            'gt_occ_inst',
            'gt_instance_centers_world',
            'gt_instance_centers_valid',
            'gt_instance_ids',
            'occ_dt',
        ],
        meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token'],
    ),
]

train_config = dict(
    type=dataset_type,
    data_root=nusc_root,
    occ_root=occ_path,
    idx_root=ocf_dataset_path,
    ori_data_root=ocf_dataset_path,
    save_local_root=ocf_dataset_path,
    ann_file=train_ann_file,
    pipeline=train_pipeline,
    classes=class_names,
    use_separate_classes=use_separate_classes,
    modality=input_modality,
    test_mode=False,
    use_valid_flag=True,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    box_type_3d='LiDAR',
    time_receptive_field=time_receptive_field,
    n_future_frames=n_future_frames,
    train_capacity=train_capacity,
    test_capacity=test_capacity,
)

test_config = dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=nusc_root,
    idx_root=ocf_dataset_path,
    ori_data_root=ocf_dataset_path,
    save_local_root=ocf_dataset_path,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    use_separate_classes=use_separate_classes,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    time_receptive_field=time_receptive_field,
    n_future_frames=n_future_frames,
    train_capacity=train_capacity,
    test_capacity=test_capacity,
)

# In our work we use 8 NVIDIA A100 GPUs.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=train_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# Model params ******************************************
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

bev_feat_dim = 96
numC_Trans = bev_feat_dim

gn_cfg = dict(type='GN', num_groups=16, requires_grad=True)
model_cfg = dict(
    use_segmentation_as_query_gt=True,
    use_gmo_bce_loss=True,
    query_gmo_loss_type='focal',
    # [GUIDE-real] dice off → GUIDE 논문처럼 occupancy 손실은 focal only.
    use_query_gmo_dice_loss=False,
    query_gt2p_instance_labeled_tau=0.3,
    query_gt2p_cooldown_iters=4000,
    query_class_ids=query_class_ids,           # full raw ids kept for GT loading/validation
    query_class_names=['background', 'foreground'],
    strict_query_class_id_validation=strict_query_class_id_validation,
    # [occ 실험] semantic cls를 binary {0=bg, 1=fg}로 통합 (car/truck/... 구분 제거).
    # query_class_ids는 8개 raw 유지, num_classes만 2, 가중치 2개. raw->compact는 many-to-one.
    query_cls_binary_fg=True,
    query_num_classes=2,
    query_cls_loss_class_weights=[0.05, 1.0],
    query_attn_match_metric='inside_log',
    query_attn_match_cost_weight=0.3,
    query_embed_dim=bev_feat_dim,
    use_query_dt_loss=False,  # DT loss off: utils_loss.py:3340 branch skipped, loss_query_dt 미생성
    query_id_reinject_scale=0.2,
    query_decor_loss_weight=1.0,
    use_query_attn_bbox_loss=True,
    query_attn_bbox_other_weight=0.3,
    query_attn_bbox_other_mode='union',
    query_attn_bbox_unmatched_weight=0.0,
    query_require_history_all_valid=True,
    query_attn_cam_gaussian_truncate_sigma=1.777,
    query_num_queries=200,
    query_cls_match_cost_weight=0.2,
    query_center_match_cost_weight=10.0,
    query_temporal_offset_match_cost_weight=0.0, #0으로 제거
    query_center_routed_loss_weight=0.3,
    # [past 실험] matched-pair GMO(occupancy)+center loss를 과거3(과거+현재)에만 적용.
    # 미래4(traj로 민 프레임)는 trajectory loss만 supervise. (efficientocf.py _hist_slice)
    query_matched_loss_history_only=True,
    # --- trajectory: 2-mode (TRAJECTORY_CONFIG.md / cen10_6) ---
    query_traj_num_modes=2,
    query_traj_use_stationary_mode=True,
    query_traj_use_cv_mode=False,
    query_traj_decoder_type='offset',
    query_traj_residual_max_m=(15.0, 15.0),
    # 라우팅 (정답 라벨 생성)
    query_traj_semantic_routing_enabled=True,
    query_traj_rule_turn_family_enabled=False,
    query_traj_static_threshold_m=0.8,
    # 추론
    query_traj_mode_infer_policy='argmax',
    # loss 가중 (warmup이 epoch별로 덮어씀)
    query_traj_loss_weight=0.5,
    query_traj_mode_cls_loss_weight=0.1,
    # [mcls 실험] mode 분류 CE에서 moving(비정지) 클래스 4배 가중 (static 쏠림 보정)
    query_traj_mode_cls_moving_class_weight=4.0,
    query_traj_static_weight=1.0,
    query_traj_moving_weight=5.0,
    query_traj_moving_reweight_enabled=True,
    query_traj_moving_threshold_m=0.8,
    # xy refine
    query_traj_xy_refine_enabled=True,
    query_traj_xy_refine_loss_weight=0.1,
    query_traj_xy_refine_num_layers=2,
    query_traj_xy_refine_hidden_dim=64,
    # 비활성
    query_traj_static_gate_enabled=False,
    query_traj_derivative_routing_enabled=False,
    # teacher forcing: iter 스케줄 비활성 → warmup hook이 gt_ratio를 epoch 단위로 제어
    # (GPU 수와 무관; 원본 1-GPU 기준 epoch 0.75~2.0 전환을 epoch 계단으로 근사)
    query_traj_teacher_forcing_enabled=True,
    query_traj_teacher_forcing_mix_enabled=True,
    query_traj_teacher_forcing_gt_ratio=1.0,
    query_traj_teacher_forcing_schedule_iters=(),
    query_traj_teacher_forcing_schedule_gt_ratios=(),
    # loss 격자 64→128(xy): 학습 occ-loss를 BEV 해상도(0.8m)에 맞춰 촘촘하게.
    # 효과: x,y sigma floor(=0.5 voxel, 방지턱)가 64→128에서 0.8m→0.4m, 자동차 폭이
    # ~1칸→~2칸으로 형상 감독 가능. 추론/시각화는 무관(항상 512에서 가우시안 직접 splat).
    query_matched_gmo_bce_occ_size=(128, 128, 20),
    # [메모리·속도] grouped voxelizer는 청크당 (chunk·G·bbox) intermediate를 backward까지 retain.
    # 흩어진 query를 묶으면 bbox≈전체 격자 → chunk 클수록 메모리·낭비연산 폭증. 결과는 chunk 무관.
    # GPU 실측(128격자, fwd+bwd, peak / ms): K80 → c8 35GB·132 / c4 25GB·117 / c2 9.6GB·89 / c1 1.3GB·131.
    #   K160 → c2 17.7GB·158 / c1 2.6GB·257.  → 속도 sweet spot=2(모든 K 최速), 메모리 최소=1.
    # 채택 2: 속도 우선(c1 대비 ~1.6배 빠름). 객체 多 프레임서 OOM(특히 여유 적은 GPU) 나면 1로 내릴 것.
    query_multi_gaussian_pair_chunk=2,
    query_num_gaussians=48,
    # [GUIDE-real] α 제거(weight_mode='ones') → (sigma↔weight) degeneracy 소멸 → occ focal이 sigma를
    # 직접 통제. 그래서 shape bound는 'tight 제약'이 아니라 '폭발 방지 난간'으로 느슨하게 둔다.
    # sigma_max 2.0→3.0(xy)/0.7→1.5(z): loss가 안에서 깎으므로 풀어줌.
    # sigma_min_m = 0.5 × matched voxel (= train/eval 최소 σ 통일). matched(0.8m xy/0.4m z)
    #   → (0.4, 0.4, 0.2). 이러면 σ가 항상 floor_vox 이상 → floor가 양 격자에서 비활성,
    #   네트워크 σ가 train·eval에 동일하게 흐름(불일치 제거). ※해상도 바꾸면 이 값도 0.5×새 voxel로.
    # sigma_reg=0: α 없으니 불필요. offset은 그대로 넉넉(트레일러 커버).
    # 주의: binary fg라 car~trailer가 이 한 세트 공유 → car IoU/precision + over-coverage 모니터.
    query_multi_gaussian_offset_max_m=(12.0, 12.0, 2.0),
    query_multi_gaussian_sigma_min_m=(0.4, 0.4, 0.2),
    query_multi_gaussian_sigma_max_m=(3.0, 3.0, 1.5),
    query_multi_gaussian_sigma_reg_loss_weight=0.0,
    # [GUIDE-real] GUIDE 논문 그대로: weight/opacity 없음. union p = 1 - Π(1 - G), 모든 가우시안
    # 중심 peak=1. weight_mode='ones'면 weight head 출력 무시하고 weights≡1 (sigmoid opacity 제거).
    # → weight collapse/몰빵 + (sigma↔weight) degeneracy 동시 소멸. combine_mode='union' 필수.
    query_multi_gaussian_weight_mode='ones',
    query_multi_gaussian_occ_combine_mode='union',
    query_multi_gaussian_softplus_bias_init=0.0,
    query_multi_gaussian_weight_reg_loss_weight=0.0,
    # eval도 학습과 동일하게 16개 mixture를 그대로 splat (평균 타원 X). occ_combine_mode=union 적용.
    query_eval_occ_use_mixture=True,
    # ===== [임계값 1/2] occ 점유 판정 =====================================
    # metric + 2D occ-grid + 3D mixture3d 에 공통 적용. config 값=학습·추론 공통.
    # 추론에서만 env EOCF_EVAL_OCC_THR 주면 그때 override. (fg score는 visualization_cfg)
    eval_occ_threshold=0.5, # occ 임계값 학습/추론/시각화 공통
)

debug_cfg = dict(
    debug_query_vis_every=8,                  # 2D query_debug_vis
    debug_query_cam_gaussian_vis_enabled=True,
    debug_query_cam_gaussian_vis_every=48,    # cam_gaussian
    debug_query_mixture3d_vis_every=48,       # 3D mixture3d
    debug_query_attn_softargmax_vis_every=48, # query attention map
)

visualization_cfg = dict(
    # ===== [임계값 2/2] query(fg) score =================================
    # query 선택 + 2D 표시 + 3D 필터 에 공통 적용. config 값=학습·추론 공통.
    # 추론에서만 env EOCF_EVAL_FG_THR 주면 그때 override. (occ는 model_cfg의 eval_occ_threshold)
    debug_query_score_threshold=0.75, # fg 임계값 학습/추론/시각화 공통
    # fg score '합성' 가중치 (무엇으로 점수를 매길지 결정 → 위 임계값으로 컷).
    # 이 config는 cls 확률만 사용(iou/cam off). utils_visualization.py:1655
    #   score = (w_iou*iou + w_cls*cls + w_cam*cam) / (w_iou+w_cls + w_cam)
    debug_query_score_iou_weight=0.0,
    debug_query_score_cls_weight=1.0,
    debug_query_score_cam_attn_weight=0.0,
    # ── 2D query_debug_vis ─────────────────────────────────────────────
    debug_query_vis_dir="./work_dirs/query_debug_vis_no_pretrain",
    debug_query_gaussian_vis_mode='prob',     # footprint 렌더 외형(prob heatmap)
    # ── cam_gaussian ───────────────────────────────────────────────────
    debug_query_cam_gaussian_vis_dir="./work_dirs/query_cam_gaussian_vis_no_pretrain",
    debug_query_cam_gaussian_vis_max_frames=3,
    debug_query_cam_gaussian_vis_gt_overlay_enabled=True,
    debug_query_cam_gaussian_vis_topk_matched=8,
    # ── 3D mixture3d ───────────────────────────────────────────────────
    debug_query_mixture3d_vis_dir="./work_dirs/query_mixture3d_vis_no_pretrain",
    debug_query_mixture3d_vis_max_queries=50,
    debug_query_mixture3d_vis_max_gt_points=40000,
    debug_query_mixture3d_vis_occ_max_voxels_per_query=4000,
)

model = dict(
    type='EfficientOCF',
    only_generate_dataset=only_generate_dataset,
    loss_norm=False,
    point_cloud_range=point_cloud_range,
    time_receptive_field=time_receptive_field,
    n_future_frames=n_future_frames,
    n_future_frames_plus=n_future_frames_plus,
    query_present_only=query_present_only,
    query_pred_num_frames=query_pred_num_frames,
    model_cfg=model_cfg,
    debug_cfg=debug_cfg,
    visualization_cfg=visualization_cfg,
    img_backbone=dict(
        pretrained='torchvision://resnet18',
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'
    ),
    img_neck=dict(
        type='SECONDFPN',
        # in_channels=[256, 512, 1024, 2048],
        in_channels=[64, 128, 256, 512],
        upsample_strides=[0.25, 0.5, 1, 2],
        # upsample_strides=[0.125, 0.25, 0.5, 1],
        out_channels=[128, 128, 128, 128],
        norm_cfg=gn_cfg,
    ),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        norm_cfg=gn_cfg,
        loss_depth_weight=3.,
        loss_depth_type='kld',
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False,
    ),
    empty_idx=empty_idx,
)

# Learning policy params ******************************************
optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=1,
    grad_clip=dict(max_norm=35, norm_type=2),
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=4000,
    # warmup_iters=800,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

runner = dict(type='EpochBasedRunner', max_epochs=15)
checkpoint_config = dict(interval=1, filename_tmpl='epoch_{}_lss_only.pth')

# 주의: hook은 매 epoch 시작 시 begin_epoch ≤ 현재 epoch인 "마지막 stage 하나"만 적용함.
# 따라서 각 stage는 전체 키를 다 들고 있어야 함 (누적 merge 아님).
# teacher forcing gt_ratio도 여기서 epoch 단위로 제어 (iter 스케줄은 비활성).
# mode_cls는 0.1 고정. TF는 epoch 1~4 풀 유지 → 5~7 ramp(0.9/0.7/0.5) → 8부터 0.
traj_warmup_schedule = [
    dict(begin_epoch=1, query_traj_loss_weight=0.05, query_traj_xy_refine_loss_weight=0.00,
         query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True,
         query_traj_teacher_forcing_gt_ratio=1.0),
    dict(begin_epoch=5, query_traj_loss_weight=0.15, query_traj_xy_refine_loss_weight=0.02,
         query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True,
         query_traj_teacher_forcing_gt_ratio=0.9),
    dict(begin_epoch=6, query_traj_loss_weight=0.15, query_traj_xy_refine_loss_weight=0.02,
         query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True,
         query_traj_teacher_forcing_gt_ratio=0.7),
    dict(begin_epoch=7, query_traj_loss_weight=0.15, query_traj_xy_refine_loss_weight=0.02,
         query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True,
         query_traj_teacher_forcing_gt_ratio=0.5),
    dict(begin_epoch=8, query_traj_loss_weight=0.25, query_traj_xy_refine_loss_weight=0.05,
         query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True,
         query_traj_teacher_forcing_gt_ratio=0.0),
    dict(begin_epoch=11, query_traj_loss_weight=0.50, query_traj_xy_refine_loss_weight=0.10,
         query_traj_mode_cls_loss_weight=0.10, query_traj_teacher_forcing_enabled=True,
         query_traj_teacher_forcing_gt_ratio=0.0),
]

custom_hooks = [
    dict(type='OccEfficiencyHook'),
    dict(type='TrajectoryWarmupHook', schedule=traj_warmup_schedule),
]

# W&B logging for sweeps
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHookNoDbg'),
        dict(type='TensorboardLoggerHookSplitTabs'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(project='efficientocf',
        #                      name='eocf_gc_light_260127_full',
        #                      resume="allow"),
        # ),
    ],
)
