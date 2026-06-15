# Developed by Jingyi Xu based on the codebase of Cam4DOcc and PowerBEV 
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF
import copy

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

# Copy and construct val config from test config.
val_config = copy.deepcopy(test_config)
val_config['test_capacity'] = 100

# In our work we use 8 NVIDIA A100 GPUs.
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=train_config,
    # val=test_config,
    val=val_config,
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

bev_feat_dim = 64
numC_Trans = bev_feat_dim

gn_cfg = dict(type='GN', num_groups=16, requires_grad=True)
model_cfg = dict(
    use_segmentation_as_query_gt=True,
    use_gmo_bce_loss=True,
    query_gmo_loss_type='focal',
    query_gt2p_instance_labeled_tau=0.3,
    query_gt2p_cooldown_iters=4000,
    query_class_ids=query_class_ids,
    query_class_names=query_class_names,
    strict_query_class_id_validation=strict_query_class_id_validation,
    query_num_classes=len(query_class_ids),
    query_cls_loss_class_weights=[0.02, 1.4, 1.3, 0.30, 1.4, 1.4, 1.2, 0.9],
    query_attn_match_metric='inside_log',
    query_attn_match_cost_weight=0.3,
    query_embed_dim=bev_feat_dim,
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
    query_temporal_offset_match_cost_weight=0.5,
    query_center_routed_loss_weight=0.3,
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
    query_matched_gmo_bce_occ_size=(64, 64, 20),
    query_num_gaussians=16,
    query_multi_gaussian_offset_max_m=(3.0, 3.0, 0.7),
    query_multi_gaussian_sigma_min_m=(0.15, 0.15, 0.15),
    query_multi_gaussian_sigma_max_m=(1.0, 1.0, 1.0),
    query_multi_gaussian_sigma_reg_loss_weight=0.01,
    query_multi_gaussian_weight_mode='softplus',
)

debug_cfg = dict(
    debug_query_vis_every=8,
    debug_query_cam_gaussian_vis_enabled=True,
    debug_query_cam_gaussian_vis_every=48,
    debug_query_inst_depth_lift_vis_every=48,
    debug_query_attn_softargmax_vis_every=(
        48
    ),
)

visualization_cfg = dict(
    debug_query_vis_dir="./work_dirs/query_debug_vis_no_pretrain",
    debug_query_gaussian_vis_mode='prob',
    debug_query_score_iou_weight=0.0,
    debug_query_score_cls_weight=1.0,
    debug_query_score_cam_attn_weight=0.0,
    debug_instance_img_vis_dir="./work_dirs/instance_img_debug_vis_no_pretrain",
    debug_instance_img_vis_max_frames=n_future_frames_plus,
    debug_query_cam_gaussian_vis_dir="./work_dirs/query_cam_gaussian_vis_no_pretrain",
    debug_query_cam_gaussian_vis_max_frames=3,
    debug_query_cam_gaussian_vis_gt_overlay_enabled=True,
    debug_query_cam_gaussian_vis_topk_matched=8,
    debug_gt_alignment_vis_dir="./work_dirs/gt_alignment_vis_no_pretrain",
    debug_query_inst_depth_lift_vis_dir="./work_dirs/query_inst_depth_lift_vis_no_pretrain",
    debug_query_inst_depth_lift_vis_max_frames=2,
    debug_query_inst_depth_lift_vis_max_instances=12,
    debug_query_attn_softargmax_vis_dir="./work_dirs/query_attn_softargmax_vis_no_pretrain",
    query_attn_vis_dir="./work_dirs/query_attn_vis_no_pretrain",
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
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='IOU_mean',
    rule='greater',
)

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
