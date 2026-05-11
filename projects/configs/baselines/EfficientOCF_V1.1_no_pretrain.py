# Developed by Jingyi Xu based on the codebase of Cam4DOcc and PowerBEV 
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF
import copy

# Basic params ******************************************
_base_ = ['../datasets/custom_nus-3d.py', '../_base_/default_runtime.py']

find_unused_parameters = False
ddp_static_graph = True
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
# [2, 3, 4, 5, 6, 7, 9, 10] + background(0)
class_names = [
    'bicycle',
    'bus',
    'car',
    'construction',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]
query_class_ids = [0, 2, 3, 4, 5, 6, 7, 9, 10]
query_class_names = ['background'] + class_names
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
        load_ocf_labels=False,
        load_segmentation_instance3d=False,
        load_segmentation_cls_instance3d=False,
        load_gt_occ_inst=True,
        gt_occ_inst_dataset_path=gt_occ_inst_dataset_path,
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
        load_height=False,
        validate_height_cache=False,
        write_height_cache=write_height_cache,
    ),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs_seq',
            'gt_occ',
            'future_egomotion',
            'gt_occ_inst',
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
        load_ocf_labels=False,
        load_segmentation_instance3d=False,
        load_segmentation_cls_instance3d=False,
        load_gt_occ_inst=True,
        gt_occ_inst_dataset_path=gt_occ_inst_dataset_path,
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
        load_height=False,
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
            'gt_occ_inst',
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
    workers_per_gpu=2,
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

# BEV / neck channel setup.
bev_feat_dim = 32
voxel_channels = [
    bev_feat_dim * time_receptive_field,
    bev_feat_dim * 2 * time_receptive_field,
    bev_feat_dim * 4 * time_receptive_field,
    bev_feat_dim * 8 * time_receptive_field,
]
pred_channels = [bev_feat_dim, bev_feat_dim * 2, bev_feat_dim * 4, bev_feat_dim * 8]
decoder_channels = [
    bev_feat_dim * n_future_frames_plus,
    bev_feat_dim * 2 * n_future_frames_plus,
    bev_feat_dim * 4 * n_future_frames_plus,
    bev_feat_dim * 8 * n_future_frames_plus,
]

# Backbone/head wiring.
numC_Trans = bev_feat_dim
occ_encoder_input_channel = (numC_Trans + 6) * time_receptive_field
voxel_out_channel = bev_feat_dim * (n_future_frames_plus)
voxel_out_channel_per_frame = bev_feat_dim
my_voxel_out_indices = (0, 1, 2, 3)

# Query matching / auxiliary losses.
gn_cfg = dict(type='GN', num_groups=16, requires_grad=True)
query_feat_align_loss_weight = 0.0

use_query_attn_bbox_loss = True
query_attn_bbox_loss_weight = 1.0
query_attn_bbox_unmatched_weight = 0.25
query_attn_bbox_eps = 1e-6

use_query_attn_cam_gaussian_score = False
query_attn_cam_frame_mode = "overlap_only"
query_attn_cam_target_mode = "prob"
query_attn_cam_metric = "kl"
query_attn_cam_camera_reduce_mode = "camera_aggregated_first"
query_attn_cam_eps = 1e-6
# Originally 3.0. 1.777 keeps roughly prob > 0.5 area for visualization.
query_attn_cam_gaussian_truncate_sigma = 1.777
query_attn_cam_target_binary_threshold = 0.5
query_attn_cam_score_norm_mode = "exp_neg"

use_query_cam_proj_consistency_loss = False
query_cam_proj_consistency_loss_weight = 1.0
query_cam_proj_consistency_metric = "bce"
query_cam_proj_consistency_eps = 1e-6
query_cam_proj_consistency_pred_norm = "amax"
query_attn_softargmax_tau = 1.0
query_depth_loss_weight = 1.0
query_depth_label_smoothing = 0.0
query_inst_depth_num_bins = 64
query_inst_depth_range_mode = "dbound"

center_only_mode = False

# Query classifier uses a compact contiguous label space:
# 0=background, then valid foreground raw ids [2,3,4,5,6,7,9,10].
query_num_classes = len(query_class_ids)
query_cls_loss_weight = 1.0
query_cls_loss_class_weights = [0.1] + [1.0] * (query_num_classes - 1)

query_attn_match_metric="soft_iou"
query_attn_match_pred_norm="amax"
query_attn_match_eps=1e-6

query_match_feature_source = 'query_img_feat_pooled'
query_soft_assign_temp = 0.10
# Hungarian cost 미사용(legacy diagnostics only)
query_soft_assign_cost_weight = 0.0
query_sim_match_cost_weight = 1.0
# Hungarian cost 미사용(legacy diagnostics only)
query_cls_match_cost_weight = 0.0
# Hungarian cost 미사용(legacy diagnostics only)
query_bev_dice_match_cost_weight = 0.0
query_attn_match_cost_weight = 0.5

query_embed_dim = bev_feat_dim
query_num_queries = 100
query_transformer_num_layers = 1
query_id_reinject_scale = 0.2
query_ca_kv_identity_init = False
query_ca_attn_tau = 1.0
query_decor_loss_weight = 1.0
query_attn_vis_dir = "./work_dirs/query_attn_vis_no_pretrain"

query_center_match_cost_weight = 4.0
query_center_match_loss_type = 'l1'
query_center_routed_loss_weight = 0.3
query_traj_loss_weight = 0.1
query_traj_loss_type = 'l1'
query_traj_residual_max_m = (8.0, 8.0)
query_traj_moving_reweight_enabled = True
query_traj_moving_threshold_m = 1.0
query_traj_moving_weight = 5.0
query_traj_static_weight = 1.0

# query_matched_gmo_bce_occ_size = (128, 128, 40)
query_matched_gmo_bce_occ_size = (64, 64, 20)

query_num_gaussians = 16
query_multi_gaussian_offset_max_m = (3.0, 3.0, 0.7)
query_multi_gaussian_sigma_min_m = (0.15, 0.15, 0.15)
query_multi_gaussian_sigma_max_m = (1.0, 1.0, 1.0)
query_multi_gaussian_sigma_reg_loss_weight = 0.01
query_multi_gaussian_sigma_reg_log_eps = 1e-6
query_multi_gaussian_pair_chunk = 8
query_multi_gaussian_weight_mode = 'softplus'
query_multi_gaussian_softplus_bias_init = -2.0
query_multi_gaussian_weight_reg_loss_weight = 1e-3
query_multi_gaussian_weight_reg_target_sum = 1.0

query_gmo_loss_type = 'bce'
query_gmo_focal_gamma = 2.0
query_gmo_focal_alpha = 0.25
query_gmo_dice_loss_weight = 0.5
query_gmo_tversky_alpha = 0.7
query_gmo_tversky_beta = 0.3

use_query_gt2p_instance_labeled_loss = False
query_gt2p_instance_labeled_loss_weight = 0.1
query_gt2p_instance_labeled_balance_weight = 0.25
query_gt2p_instance_labeled_tau = 0.3
query_gt2p_instance_labeled_assign_sigma_xyz = (4.0, 4.0, 1.5)
query_gt2p_instance_labeled_sigma_policy = 'fixed'  # Ignored when center_only_mode=True.
query_gt2p_cooldown_enabled = False
query_gt2p_cooldown_start_iter = 0
query_gt2p_cooldown_iters = 4000
query_gt2p_cooldown_min_scale = 0.0

# Debug / visualization controls.
debug_query_center_marker_radius = 3
debug_query_confidence_vis_threshold = 0.5
debug_query_objectness_vis_threshold = 0.5
debug_query_gaussian_vis_mode = 'prob'
debug_query_gaussian_prob_threshold = 0.5
debug_query_gaussian_prob_alpha_scale = 4.0
debug_query_score_topk = 50
debug_query_score_threshold = 0.5
debug_query_score_iou_weight = 0.0
debug_query_score_cls_weight = 0.6
debug_query_score_cam_attn_weight = 0.4
debug_instance_img_vis_every = 0
debug_instance_img_vis_dir = "./work_dirs/instance_img_debug_vis_no_pretrain"
debug_instance_img_vis_max_instances = 24
debug_query_cam_gaussian_vis_enabled = True
debug_query_cam_gaussian_vis_every = 48
debug_query_cam_gaussian_vis_dir = "./work_dirs/query_cam_gaussian_vis_no_pretrain"
debug_query_cam_gaussian_vis_max_queries = 50
debug_query_cam_gaussian_vis_max_frames = 3
debug_query_cam_gaussian_vis_gt_overlay_enabled = True
debug_query_cam_gaussian_vis_topk_matched = 8
debug_gt_alignment_vis_every = 0
debug_gt_alignment_vis_dir = "./work_dirs/gt_alignment_vis_no_pretrain"
debug_gt_alignment_vis_max_frames = 7
debug_query_inst_depth_lift_vis_every = 48
debug_query_inst_depth_lift_vis_dir = "./work_dirs/query_inst_depth_lift_vis_no_pretrain"
debug_query_inst_depth_lift_vis_max_frames = 2
debug_query_inst_depth_lift_vis_max_cams = 2
debug_query_inst_depth_lift_vis_max_instances = 12
debug_query_attn_softargmax_vis_enabled = True
debug_query_attn_softargmax_vis_every = 48
debug_query_attn_softargmax_vis_dir = "./work_dirs/query_attn_softargmax_vis_no_pretrain"
debug_query_attn_softargmax_vis_max_frames = 2
debug_query_attn_softargmax_vis_max_cams = 3
debug_query_attn_softargmax_vis_max_queries = 16

model = dict(
    type='EfficientOCF',

    # Runtime and pretrained initialization.
    only_generate_dataset=only_generate_dataset,
    loss_norm=False,
    lss_pretrained_ckpt_path='',
    freeze_lss_pretrained=False,
    lss_pretrained_strict=False,

    # Temporal / geometry setup.
    point_cloud_range=point_cloud_range,
    time_receptive_field=time_receptive_field,
    n_future_frames=n_future_frames,
    n_future_frames_plus=n_future_frames_plus,
    query_present_only=query_present_only,
    query_pred_num_frames=query_pred_num_frames,

    # Query supervision source.
    gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
    use_segmentation_as_query_gt=False,

    # Pretrain-only visualization.
    pretrain_view_transform_only=False,  # True: image->view_transform->BEV->occ_head only
    pretrain_lss_vis_every=0,  # >0 to save LSS-BEV occupancy visualization periodically
    pretrain_lss_vis_dir="./work_dirs/no_pretrain_vis",
    pretrain_lss_vis_prob_thr=0.5,
    pretrain_lss_vis_max_frames=6,

    # Query losses.
    use_gmo_bce_loss=True,
    query_gmo_loss_type=query_gmo_loss_type,
    query_gmo_focal_gamma=query_gmo_focal_gamma,
    query_gmo_focal_alpha=query_gmo_focal_alpha,
    use_lss_bev_occ_loss=False,
    use_query_gmo_dice_loss=True,
    query_gmo_dice_loss_weight=query_gmo_dice_loss_weight,
    query_gmo_tversky_alpha=query_gmo_tversky_alpha,
    query_gmo_tversky_beta=query_gmo_tversky_beta,
    use_query_inst_center_match_loss=True,
    use_query_dt_loss=True,
    use_query_gt2p_instance_labeled_loss=use_query_gt2p_instance_labeled_loss,
    query_gt2p_instance_labeled_loss_weight=query_gt2p_instance_labeled_loss_weight,
    query_gt2p_instance_labeled_balance_weight=query_gt2p_instance_labeled_balance_weight,
    query_gt2p_instance_labeled_tau=query_gt2p_instance_labeled_tau,
    query_gt2p_instance_labeled_assign_sigma_xyz=query_gt2p_instance_labeled_assign_sigma_xyz,
    query_gt2p_instance_labeled_sigma_policy=query_gt2p_instance_labeled_sigma_policy,
    query_gt2p_cooldown_enabled=query_gt2p_cooldown_enabled,
    query_gt2p_cooldown_start_iter=query_gt2p_cooldown_start_iter,
    query_gt2p_cooldown_iters=query_gt2p_cooldown_iters,
    query_gt2p_cooldown_min_scale=query_gt2p_cooldown_min_scale,

    # Query matching / class constraints.
    query_feat_align_loss_weight=query_feat_align_loss_weight,
    query_class_ids=query_class_ids,
    query_class_names=query_class_names,
    strict_query_class_id_validation=strict_query_class_id_validation,
    query_num_classes=query_num_classes,
    query_cls_loss_weight=query_cls_loss_weight,
    query_cls_loss_class_weights=query_cls_loss_class_weights,
    query_match_feature_source=query_match_feature_source,
    query_soft_assign_temp=query_soft_assign_temp,
    query_soft_assign_cost_weight=query_soft_assign_cost_weight,
    query_sim_match_cost_weight=query_sim_match_cost_weight,
    query_cls_match_cost_weight=query_cls_match_cost_weight,
    query_bev_dice_match_cost_weight=query_bev_dice_match_cost_weight,
    query_attn_match_cost_weight=query_attn_match_cost_weight,
    query_attn_match_metric=query_attn_match_metric,
    query_attn_match_pred_norm=query_attn_match_pred_norm,
    query_attn_match_eps=query_attn_match_eps,

    # Query Gaussian parameterization.
    query_matched_gmo_bce_occ_size=query_matched_gmo_bce_occ_size,
    query_num_gaussians=query_num_gaussians,
    query_multi_gaussian_offset_max_m=query_multi_gaussian_offset_max_m,
    query_multi_gaussian_sigma_min_m=query_multi_gaussian_sigma_min_m,
    query_multi_gaussian_sigma_max_m=query_multi_gaussian_sigma_max_m,
    query_multi_gaussian_sigma_reg_loss_weight=query_multi_gaussian_sigma_reg_loss_weight,
    query_multi_gaussian_sigma_reg_log_eps=query_multi_gaussian_sigma_reg_log_eps,
    query_multi_gaussian_pair_chunk=query_multi_gaussian_pair_chunk,
    query_multi_gaussian_weight_mode=query_multi_gaussian_weight_mode,
    query_multi_gaussian_softplus_bias_init=query_multi_gaussian_softplus_bias_init,
    query_multi_gaussian_weight_reg_loss_weight=query_multi_gaussian_weight_reg_loss_weight,
    query_multi_gaussian_weight_reg_target_sum=query_multi_gaussian_weight_reg_target_sum,

    # Query transformer.
    query_embed_dim=query_embed_dim,
    query_num_queries=query_num_queries,
    query_transformer_num_layers=query_transformer_num_layers,
    query_id_reinject_scale=query_id_reinject_scale,
    query_ca_kv_identity_init=query_ca_kv_identity_init,
    query_ca_attn_tau=query_ca_attn_tau,
    query_decor_loss_weight=query_decor_loss_weight,
    query_attn_vis_dir=query_attn_vis_dir,

    # Attention-based query losses.
    use_query_attn_bbox_loss=use_query_attn_bbox_loss,
    query_attn_bbox_loss_weight=query_attn_bbox_loss_weight,
    query_attn_bbox_unmatched_weight=query_attn_bbox_unmatched_weight,
    query_attn_bbox_eps=query_attn_bbox_eps,
    use_query_attn_cam_gaussian_score=use_query_attn_cam_gaussian_score,
    query_attn_cam_frame_mode=query_attn_cam_frame_mode,
    query_attn_cam_target_mode=query_attn_cam_target_mode,
    query_attn_cam_metric=query_attn_cam_metric,
    query_attn_cam_camera_reduce_mode=query_attn_cam_camera_reduce_mode,
    query_attn_cam_eps=query_attn_cam_eps,
    query_attn_cam_gaussian_truncate_sigma=query_attn_cam_gaussian_truncate_sigma,
    query_attn_cam_target_binary_threshold=query_attn_cam_target_binary_threshold,
    query_attn_cam_score_norm_mode=query_attn_cam_score_norm_mode,
    use_query_cam_proj_consistency_loss=use_query_cam_proj_consistency_loss,
    query_cam_proj_consistency_loss_weight=query_cam_proj_consistency_loss_weight,
    query_cam_proj_consistency_metric=query_cam_proj_consistency_metric,
    query_cam_proj_consistency_eps=query_cam_proj_consistency_eps,
    query_cam_proj_consistency_pred_norm=query_cam_proj_consistency_pred_norm,
    query_attn_softargmax_tau=query_attn_softargmax_tau,
    query_depth_loss_weight=query_depth_loss_weight,
    query_depth_label_smoothing=query_depth_label_smoothing,
    query_inst_depth_num_bins=query_inst_depth_num_bins,
    query_inst_depth_range_mode=query_inst_depth_range_mode,

    # Center matching.
    query_center_match_cost_weight=query_center_match_cost_weight,
    query_center_match_loss_type=query_center_match_loss_type,
    query_center_routed_loss_weight=query_center_routed_loss_weight,
    query_traj_loss_weight=query_traj_loss_weight,
    query_traj_loss_type=query_traj_loss_type,
    query_traj_residual_max_m=query_traj_residual_max_m,
    query_traj_moving_reweight_enabled=query_traj_moving_reweight_enabled,
    query_traj_moving_threshold_m=query_traj_moving_threshold_m,
    query_traj_moving_weight=query_traj_moving_weight,
    query_traj_static_weight=query_traj_static_weight,

    # Debug controls.
    debug_query_vis_every=48,
    debug_query_vis_dir="./work_dirs/query_debug_vis_no_pretrain",
    center_only_mode=center_only_mode,
    debug_query_center_marker_radius=debug_query_center_marker_radius,
    debug_query_confidence_vis_threshold=debug_query_confidence_vis_threshold,
    debug_query_objectness_vis_threshold=debug_query_objectness_vis_threshold,
    debug_query_gaussian_vis_mode=debug_query_gaussian_vis_mode,
    debug_query_gaussian_prob_threshold=debug_query_gaussian_prob_threshold,
    debug_query_gaussian_prob_alpha_scale=debug_query_gaussian_prob_alpha_scale,
    debug_query_score_topk=debug_query_score_topk,
    debug_query_score_threshold=debug_query_score_threshold,
    debug_query_score_iou_weight=debug_query_score_iou_weight,
    debug_query_score_cls_weight=debug_query_score_cls_weight,
    debug_query_score_cam_attn_weight=debug_query_score_cam_attn_weight,
    debug_instance_img_vis_every=debug_instance_img_vis_every,
    debug_instance_img_vis_dir=debug_instance_img_vis_dir,
    debug_instance_img_vis_max_frames=n_future_frames_plus,
    debug_instance_img_vis_max_instances=debug_instance_img_vis_max_instances,
    debug_query_cam_gaussian_vis_enabled=debug_query_cam_gaussian_vis_enabled,
    debug_query_cam_gaussian_vis_every=debug_query_cam_gaussian_vis_every,
    debug_query_cam_gaussian_vis_dir=debug_query_cam_gaussian_vis_dir,
    debug_query_cam_gaussian_vis_max_queries=debug_query_cam_gaussian_vis_max_queries,
    debug_query_cam_gaussian_vis_max_frames=debug_query_cam_gaussian_vis_max_frames,
    debug_query_cam_gaussian_vis_gt_overlay_enabled=debug_query_cam_gaussian_vis_gt_overlay_enabled,
    debug_query_cam_gaussian_vis_topk_matched=debug_query_cam_gaussian_vis_topk_matched,
    debug_gt_alignment_vis_every=debug_gt_alignment_vis_every,
    debug_gt_alignment_vis_dir=debug_gt_alignment_vis_dir,
    debug_gt_alignment_vis_max_frames=debug_gt_alignment_vis_max_frames,
    debug_query_inst_depth_lift_vis_every=debug_query_inst_depth_lift_vis_every,
    debug_query_inst_depth_lift_vis_dir=debug_query_inst_depth_lift_vis_dir,
    debug_query_inst_depth_lift_vis_max_frames=debug_query_inst_depth_lift_vis_max_frames,
    debug_query_inst_depth_lift_vis_max_cams=debug_query_inst_depth_lift_vis_max_cams,
    debug_query_inst_depth_lift_vis_max_instances=debug_query_inst_depth_lift_vis_max_instances,
    debug_query_attn_softargmax_vis_every=(
        debug_query_attn_softargmax_vis_every
        if debug_query_attn_softargmax_vis_enabled
        else 0
    ),
    debug_query_attn_softargmax_vis_dir=debug_query_attn_softargmax_vis_dir,
    debug_query_attn_softargmax_vis_max_frames=debug_query_attn_softargmax_vis_max_frames,
    debug_query_attn_softargmax_vis_max_cams=debug_query_attn_softargmax_vis_max_cams,
    debug_query_attn_softargmax_vis_max_queries=debug_query_attn_softargmax_vis_max_queries,

    # Backbone / neck / head.
    img_backbone=dict(
        pretrained='torchvision://resnet18',
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
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
    occ_encoder_backbone=dict(
        type='CustomResNet2D',
        depth=18,
        n_input_channels=occ_encoder_input_channel,
        block_inplanes=voxel_channels,
        out_indices=my_voxel_out_indices,
        norm_cfg=gn_cfg,
    ),
    occ_predictor=dict(
        type='Predictor',
        n_input_channels=pred_channels,
        in_timesteps=time_receptive_field,
        out_timesteps=n_future_frames_plus,
        norm_cfg=gn_cfg,
    ),
    occ_encoder_neck=dict(
        type='FPN',
        in_channels=decoder_channels,
        out_channels=voxel_out_channel,
        num_outs=4,
        norm_cfg=gn_cfg,
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=gn_cfg,
        soft_weights=True,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(my_voxel_out_indices),
        in_channels=[voxel_out_channel_per_frame] * len(my_voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    empty_idx=empty_idx,
)

# Learning policy params ******************************************
optimizer = dict(
    type='AdamW',
    lr=5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

optimizer_config = dict(
    # type='GradientCumulativeOptimizerHook',
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=8,
    grad_clip=dict(max_norm=35, norm_type=2),
    # Mixed precision training optiion.
    loss_scale='dynamic',
)

# Mixed precision training optiion.
# fp16 = dict(loss_scale='dynamic')

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    # warmup_iters=4000,
    warmup_iters=800,
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

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]

# W&B logging for sweeps
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHookSplitTabs'),
        # dict(
        #     type='WandbLoggerHook',
        #     init_kwargs=dict(project='efficientocf',
        #                      name='eocf_gc_light_260127_full',
        #                      resume="allow"),
        # ),
    ],
)
