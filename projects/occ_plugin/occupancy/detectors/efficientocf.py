# Developed by Jingyi Xu based on the codebase of Cam4DOcc, OpenOccupancy and PowerBEV
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

import torch
import torch.nn as nn

from mmdet.models import DETECTORS
from mmcv.runner import force_fp32
from .bevdepth import BEVDepth
from mmdet3d.models import builder
from projects.occ_plugin.occupancy.image2bev.transformer import TransformerModule
from projects.occ_plugin.occupancy.dense_heads.query_head import QueryHead
from projects.occ_plugin.occupancy.dense_heads.voxelizer import SoftVoxelizerOneAdd

from .utils_visualization import EfficientOCFVisualizationMixin
from .utils_dbg import EfficientOCFDebugMixin
from .utils_loss import EfficientOCFLossMixin
from .utils_matcher import EfficientOCFMatcherMixin
from .utils_bev_pool import EfficientOCFBEVPoolMixin
from .utils_gt_prep import EfficientOCFGTPrepMixin
from .utils_instance_img_debug import EfficientOCFInstanceImgDebugMixin
from .utils_query_projection import EfficientOCFQueryProjectionMixin
from .utils_geometry import EfficientOCFGeometryMixin


@DETECTORS.register_module()
class EfficientOCF(
    EfficientOCFVisualizationMixin,
    EfficientOCFDebugMixin,
    EfficientOCFInstanceImgDebugMixin,
    EfficientOCFQueryProjectionMixin,
    EfficientOCFLossMixin,
    EfficientOCFMatcherMixin,
    EfficientOCFBEVPoolMixin,
    EfficientOCFGTPrepMixin,
    EfficientOCFGeometryMixin,
    BEVDepth,
):
    def __init__(self, 
            only_generate_dataset=False,
            empty_idx=0,
            occ_encoder_backbone=None,
            occ_predictor=None,
            occ_encoder_neck=None,
            loss_norm=False,
            point_cloud_range=None,
            time_receptive_field=None,
            n_future_frames=None,
            n_future_frames_plus=None,
            query_present_only=False,
            query_pred_num_frames=None,
            gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
            use_segmentation_as_query_gt=False,
            use_gmo_bce_loss=False,
            query_gmo_loss_type='balanced_bce',
            query_gmo_focal_gamma=2.0,
            query_gmo_focal_alpha=0.25,
            query_gmo_dice_loss_weight=0.5,
            query_gmo_tversky_alpha=0.7,
            query_gmo_tversky_beta=0.3,
            use_lss_bev_occ_loss=True,
            pretrain_view_transform_only=False,
            pretrain_lss_vis_every=0,
            pretrain_lss_vis_dir="./work_dirs/lss_pretrain_vis",
            pretrain_lss_vis_prob_thr=0.5,
            pretrain_lss_vis_max_frames=6,
            use_query_gmo_dice_loss=True,
            use_query_inst_center_match_loss=True,
            use_query_dt_loss=True,
            use_query_gt2p_instance_labeled_loss=False,
            query_gt2p_instance_labeled_loss_weight=0.1,
            query_gt2p_instance_labeled_balance_weight=0.25,
            query_gt2p_instance_labeled_tau=1.0,
            query_gt2p_instance_labeled_assign_sigma_xyz=(4.0, 4.0, 1.5),
            query_gt2p_instance_labeled_sigma_policy='fixed',
            query_gt2p_cooldown_enabled=False,
            query_gt2p_cooldown_start_iter=0,
            query_gt2p_cooldown_iters=0,
            query_gt2p_cooldown_min_scale=0.0,
            debug_query_vis_every=0,
            debug_query_vis_dir="./work_dirs/query_debug_vis",
            center_only_mode=False,
            debug_query_center_marker_radius=3,
            debug_query_confidence_vis_threshold=0.5,
            debug_query_objectness_vis_threshold=0.5,
            debug_query_gaussian_vis_mode='ellipse',
            debug_query_gaussian_prob_threshold=0.5,
            debug_query_gaussian_prob_alpha_scale=4.0,
            debug_query_score_topk=50,
            debug_query_score_threshold=0.5,
            debug_query_score_iou_weight=0.5,
            debug_query_score_cls_weight=0.5,
            debug_query_score_cam_attn_weight=0.0,
            debug_instance_img_vis_every=0,
            debug_instance_img_vis_dir="./work_dirs/instance_img_debug_vis",
            debug_instance_img_vis_max_frames=3,
            debug_instance_img_vis_max_instances=24,
            debug_query_cam_gaussian_vis_enabled=False,
            debug_query_cam_gaussian_vis_every=0,
            debug_query_cam_gaussian_vis_dir="./work_dirs/query_cam_gaussian_vis",
            debug_query_cam_gaussian_vis_max_queries=50,
            debug_query_cam_gaussian_vis_max_frames=2,
            debug_gt_alignment_vis_every=0,
            debug_gt_alignment_vis_dir="./work_dirs/gt_alignment_vis",
            debug_gt_alignment_vis_max_frames=7,
            debug_query_inst_depth_lift_vis_every=0,
            debug_query_inst_depth_lift_vis_dir="./work_dirs/query_inst_depth_lift_vis",
            debug_query_inst_depth_lift_vis_max_frames=3,
            debug_query_inst_depth_lift_vis_max_cams=2,
            debug_query_inst_depth_lift_vis_max_instances=16,
            debug_query_attn_softargmax_vis_every=0,
            debug_query_attn_softargmax_vis_dir="./work_dirs/query_attn_softargmax_vis",
            debug_query_attn_softargmax_vis_max_frames=2,
            debug_query_attn_softargmax_vis_max_cams=3,
            debug_query_attn_softargmax_vis_max_queries=16,
            query_feat_cosine_threshold=0.30,
            query_center_distance_threshold_m=1.50,
            query_center_loss_detach_query_feat=False,
            query_feat_unmatched_neg_loss_weight=0.0,
            query_feat_unmatched_neg_margin=0.2,
            query_bev_pool_fixed_sigma_xyz=(4.0, 4.0, 1.5),
            query_class_ids=None,
            query_class_names=None,
            strict_query_class_id_validation=False,
            query_num_classes=3,
            query_cls_loss_weight=1.0,
            query_cls_loss_class_weights=None,
            query_match_feature_source='query_img_feat_pooled',
            query_soft_assign_temp=0.10,
            query_soft_assign_cost_weight=0.0,
            query_sim_match_cost_weight=1.0,
            query_cls_match_cost_weight=0.0,
            query_bev_dice_match_cost_weight=0.0,
            query_center_routed_loss_weight=0.1,
            query_traj_loss_weight=0.0,
            query_traj_loss_type='l1',
            query_traj_residual_max_m=(8.0, 8.0),
            query_traj_prior_detach=True,
            query_traj_moving_reweight_enabled=False,
            query_traj_moving_threshold_m=0.5,
            query_traj_moving_weight=5.0,
            query_traj_static_weight=1.0,
            # query_center_routed_loss_weight=1.0,
            query_center_match_cost_weight=0.0,
            query_center_match_loss_type='l1',
            query_matched_gmo_bce_occ_size=(128, 128, 10),
            query_num_gaussians=1,
            query_multi_gaussian_offset_max_m=(6.0, 6.0, 2.0),
            query_multi_gaussian_sigma_min_m=(0.15, 0.15, 0.10),
            query_multi_gaussian_sigma_max_m=(4.0, 4.0, 1.5),
            query_multi_gaussian_sigma_reg_loss_weight=0.0,
            query_multi_gaussian_sigma_reg_log_eps=1e-6,
            query_multi_gaussian_pair_chunk=8,
            query_multi_gaussian_weight_mode='softmax',
            query_multi_gaussian_softplus_bias_init=-2.0,
            query_multi_gaussian_weight_reg_loss_weight=1e-3,
            query_multi_gaussian_weight_reg_target_sum=1.0,
            query_embed_dim=256,
            query_num_queries=100,
            query_transformer_num_layers=1,
            query_id_reinject_scale=0.0,
            query_ca_kv_identity_init=False,
            query_ca_attn_tau=1.0,
            query_decor_loss_weight=0.0,
            query_attn_vis_dir="./work_dirs/query_attn_vis",
            use_query_attn_bbox_loss=False,
            query_attn_bbox_loss_weight=1.0,
            query_attn_bbox_unmatched_weight=0.25,
            query_attn_bbox_eps=1e-6,
            query_attn_bbox_camera_reduce_mode="camera_aggregated_first",
            query_attn_bbox_unmatched_mode="inverse_union",
            query_attn_match_cost_weight=0.0,
            query_attn_match_metric="soft_iou",
            query_attn_match_pred_norm="amax",
            query_attn_match_eps=1e-6,
            use_query_attn_cam_gaussian_score=False,
            query_attn_cam_frame_mode="overlap_only",
            query_attn_cam_target_mode="prob",
            query_attn_cam_metric="kl",
            query_attn_cam_camera_reduce_mode="camera_aggregated_first",
            query_attn_cam_eps=1e-6,
            query_attn_cam_gaussian_truncate_sigma=3.0,
            query_attn_cam_target_binary_threshold=0.5,
            query_attn_cam_score_norm_mode="exp_neg",
            query_attn_softargmax_tau=1.0,
            query_depth_loss_weight=1.0,
            query_depth_label_smoothing=0.0,
            query_inst_depth_num_bins=64,
            query_inst_depth_range_mode="dbound",
            query_inst_depth_min=0.0,
            query_inst_depth_max=0.0,
            debug_query_cam_gaussian_vis_gt_overlay_enabled=False,
            debug_query_cam_gaussian_vis_topk_matched=0,
            **kwargs):
        '''
        EfficientNet is our end-to-end baseline for 4D camera-only occupancy forecasting
        
        there is one stream for the forecasting task with aggregated voxel features as inputs:
            occ_encoder_backbone -> occ_predictor -> occ_encoder_neck -> pts_bbox_head
        
        time_receptive_field: number of historical frames used for forecasting (including the present one), default: 3
        n_future_frames: number of forecasted future frames, default: 4
        n_future_frames_plus: number of estimated frames (> n_future_frames), default: 6 (if only forecasting occupancy states rather than instances, n_future_frames=n_future_frames_plus can be set)
        '''
        # Keep backward compatibility with old configs while dropping dead options.
        for unused_key in (
            "loss_cfg",
            "disable_loss_depth",
            "test_present",
            "max_label",
            "iou_thresh_for_vpq",
            "record_time",
            "save_pred",
            "save_path",
            "query_bev_sim_cost_enabled",
            "query_bev_sim_cost_weight",
            "query_bev_sim_cost_warmup_start_iter",
            "query_bev_sim_cost_warmup_iters",
            "query_bev_sim_cost_norm",
            "use_soft_query_quality_routing",
            "query_quality_soft_temp",
            "query_quality_minmax_norm",
            "query_quality_min_soft_weight",
            "query_feat_softmax_temp",
            "query_feat_query_agg",

            "query_routing_warmup_iters",
            "height_encoder_backbone",
            "height_predictor",
            "height_encoder_neck",
            "height_head",
            "flow_encoder_backbone",
            "flow_predictor",
            "flow_encoder_neck",
            "flow_head",
            "detach_height_feat",
            "use_gt_occ_as_query_input",
            "gt_occ_query_downsample",
            "gt_occ_query_downsample_mode",
            "gt_occ_query_binary_mode",
            "gt_occ_query_encoder_dims",
            "use_query_objectness_loss",
            "query_objectness_loss_weight",
            "query_objectness_loss_type",
            "query_objectness_focal_gamma",
            "query_objectness_focal_alpha",
            "objectness_target_mode",
            "objectness_soft_pos_radius_m",
            "objectness_soft_neg_radius_m",
            "objectness_soft_ignore_radius_m",
            "objectness_soft_topk_per_gt",
            "objectness_soft_match_override",
            "use_query_self_bev_feat_align_loss",
            "query_self_bev_feat_align_loss_weight",
            "query_self_bev_feat_align_loss_type",
            "query_self_bev_feat_align_detach_bev_feats",
            "query_self_bev_match_cost_weight",
            "query_raw_bev_overlap_loss_weight",
            "query_objectness_match_cost_weight",
        ):
            kwargs.pop(unused_key, None)
        super().__init__(**kwargs)

        self.only_generate_dataset = only_generate_dataset
        self.loss_norm = loss_norm
        self.time_receptive_field = time_receptive_field
        self.n_future_frames = n_future_frames
        self.n_future_frames_plus = n_future_frames_plus
        self.query_present_only = bool(query_present_only)
        if query_pred_num_frames is None:
            self.query_pred_num_frames = int(self.n_future_frames_plus)
        else:
            self.query_pred_num_frames = int(query_pred_num_frames)
        self.query_present_global_idx = int(self.time_receptive_field - 1)
        self.eval_start_moment = self.n_future_frames_plus - self.n_future_frames - 1
        self.query_overlap_frames = max(0, int(self.n_future_frames_plus) - int(self.n_future_frames))

        self.empty_idx = empty_idx
        self.gmo_ids = tuple(int(v) for v in gmo_ids)
        self.use_segmentation_as_query_gt = bool(use_segmentation_as_query_gt)
        self.use_gmo_bce_loss = bool(use_gmo_bce_loss)
        self.query_gmo_loss_type = str(query_gmo_loss_type).lower()
        self.query_gmo_focal_gamma = float(query_gmo_focal_gamma)
        self.query_gmo_focal_alpha = float(query_gmo_focal_alpha)
        self.query_gmo_dice_loss_weight = float(query_gmo_dice_loss_weight)
        self.query_gmo_tversky_alpha = float(query_gmo_tversky_alpha)
        self.query_gmo_tversky_beta = float(query_gmo_tversky_beta)
        self.use_lss_bev_occ_loss = bool(use_lss_bev_occ_loss)
        self.pretrain_view_transform_only = bool(pretrain_view_transform_only)
        self.pretrain_lss_vis_every = int(pretrain_lss_vis_every)
        self.pretrain_lss_vis_dir = str(pretrain_lss_vis_dir)
        self.pretrain_lss_vis_prob_thr = float(pretrain_lss_vis_prob_thr)
        self.pretrain_lss_vis_max_frames = int(pretrain_lss_vis_max_frames)
        self._pretrain_lss_vis_iter = 0
        self.use_query_gmo_dice_loss = bool(use_query_gmo_dice_loss)
        self.use_query_inst_center_match_loss = bool(use_query_inst_center_match_loss)
        self.query_traj_loss_weight = float(query_traj_loss_weight)
        self.query_traj_loss_type = str(query_traj_loss_type).lower()
        self.query_traj_residual_max_m = tuple(float(v) for v in query_traj_residual_max_m)
        self.query_traj_prior_detach = bool(query_traj_prior_detach)
        self.query_traj_moving_reweight_enabled = bool(query_traj_moving_reweight_enabled)
        self.query_traj_moving_threshold_m = float(query_traj_moving_threshold_m)
        self.query_traj_moving_weight = float(query_traj_moving_weight)
        self.query_traj_static_weight = float(query_traj_static_weight)
        self.use_query_dt_loss = bool(use_query_dt_loss)
        self.use_query_gt2p_instance_labeled_loss = bool(use_query_gt2p_instance_labeled_loss)
        self.query_gt2p_instance_labeled_loss_weight = float(query_gt2p_instance_labeled_loss_weight)
        self.query_gt2p_instance_labeled_balance_weight = float(query_gt2p_instance_labeled_balance_weight)
        self.query_gt2p_instance_labeled_tau = float(query_gt2p_instance_labeled_tau)
        self.query_gt2p_cooldown_enabled = bool(query_gt2p_cooldown_enabled)
        self.query_gt2p_cooldown_start_iter = int(query_gt2p_cooldown_start_iter)
        self.query_gt2p_cooldown_iters = int(query_gt2p_cooldown_iters)
        self.query_gt2p_cooldown_min_scale = float(query_gt2p_cooldown_min_scale)
        self.query_gt2p_instance_labeled_assign_sigma_xyz = tuple(
            float(v) for v in query_gt2p_instance_labeled_assign_sigma_xyz
        )
        self.query_gt2p_instance_labeled_sigma_policy = str(
            query_gt2p_instance_labeled_sigma_policy
        ).lower()
        self.debug_query_vis_every = int(debug_query_vis_every)
        self.debug_query_vis_dir = str(debug_query_vis_dir)
        self.center_only_mode = bool(center_only_mode)
        self.debug_query_center_marker_radius = max(0, int(debug_query_center_marker_radius))
        self.debug_query_confidence_vis_threshold = float(debug_query_confidence_vis_threshold)
        self.debug_query_objectness_vis_threshold = float(debug_query_objectness_vis_threshold)
        self.debug_query_gaussian_vis_mode = str(debug_query_gaussian_vis_mode).lower()
        self.debug_query_gaussian_prob_threshold = float(debug_query_gaussian_prob_threshold)
        self.debug_query_gaussian_prob_alpha_scale = float(debug_query_gaussian_prob_alpha_scale)
        self.debug_query_score_topk = max(0, int(debug_query_score_topk))
        self.debug_query_score_threshold = float(debug_query_score_threshold)
        self.debug_query_score_iou_weight = float(debug_query_score_iou_weight)
        self.debug_query_score_cls_weight = float(debug_query_score_cls_weight)
        self.debug_query_score_cam_attn_weight = float(debug_query_score_cam_attn_weight)
        self.debug_instance_img_vis_every = int(debug_instance_img_vis_every)
        self.debug_instance_img_vis_dir = str(debug_instance_img_vis_dir)
        self.debug_instance_img_vis_max_frames = max(1, int(debug_instance_img_vis_max_frames))
        self.debug_instance_img_vis_max_instances = max(1, int(debug_instance_img_vis_max_instances))
        self.debug_query_cam_gaussian_vis_enabled = bool(debug_query_cam_gaussian_vis_enabled)
        self.debug_query_cam_gaussian_vis_every = int(debug_query_cam_gaussian_vis_every)
        self.debug_query_cam_gaussian_vis_dir = str(debug_query_cam_gaussian_vis_dir)
        self.debug_query_cam_gaussian_vis_max_queries = max(1, int(debug_query_cam_gaussian_vis_max_queries))
        self.debug_query_cam_gaussian_vis_max_frames = max(1, int(debug_query_cam_gaussian_vis_max_frames))
        self.debug_gt_alignment_vis_every = int(debug_gt_alignment_vis_every)
        self.debug_gt_alignment_vis_dir = str(debug_gt_alignment_vis_dir)
        self.debug_gt_alignment_vis_max_frames = max(1, int(debug_gt_alignment_vis_max_frames))
        self.debug_query_inst_depth_lift_vis_every = int(debug_query_inst_depth_lift_vis_every)
        self.debug_query_inst_depth_lift_vis_dir = str(debug_query_inst_depth_lift_vis_dir)
        self.debug_query_inst_depth_lift_vis_max_frames = max(1, int(debug_query_inst_depth_lift_vis_max_frames))
        self.debug_query_inst_depth_lift_vis_max_cams = max(1, int(debug_query_inst_depth_lift_vis_max_cams))
        self.debug_query_inst_depth_lift_vis_max_instances = max(1, int(debug_query_inst_depth_lift_vis_max_instances))
        self.debug_query_attn_softargmax_vis_every = int(debug_query_attn_softargmax_vis_every)
        self.debug_query_attn_softargmax_vis_dir = str(debug_query_attn_softargmax_vis_dir)
        self.debug_query_attn_softargmax_vis_max_frames = max(1, int(debug_query_attn_softargmax_vis_max_frames))
        self.debug_query_attn_softargmax_vis_max_cams = max(1, int(debug_query_attn_softargmax_vis_max_cams))
        self.debug_query_attn_softargmax_vis_max_queries = max(1, int(debug_query_attn_softargmax_vis_max_queries))
        self.debug_query_cam_gaussian_vis_gt_overlay_enabled = bool(
            debug_query_cam_gaussian_vis_gt_overlay_enabled
        )
        self.debug_query_cam_gaussian_vis_topk_matched = int(debug_query_cam_gaussian_vis_topk_matched)
        self._dbg_printed_instance_img_seq_len_warning = False
        self._last_inst_match_result = None
        self._last_gt_instance_bev_feat_cache = None
        self._last_query_inst_depth_target_pack = None
        self._last_query_attn_soft_lift_pack = None
        self.query_feat_cosine_threshold = float(query_feat_cosine_threshold)
        self.query_center_distance_threshold_m = float(query_center_distance_threshold_m)
        self.query_center_loss_detach_query_feat = bool(query_center_loss_detach_query_feat)

        self.query_feat_unmatched_neg_loss_weight = float(query_feat_unmatched_neg_loss_weight)
        self.query_feat_unmatched_neg_margin = float(query_feat_unmatched_neg_margin)
        self.query_bev_pool_fixed_sigma_xyz = tuple(float(v) for v in query_bev_pool_fixed_sigma_xyz)
        self.query_num_classes = int(query_num_classes)
        self.query_bg_class = 0
        if query_class_ids is None:
            query_class_ids = tuple(range(self.query_num_classes))
        else:
            query_class_ids = tuple(int(v) for v in query_class_ids)
        self.query_class_ids = query_class_ids
        if query_class_names is None:
            query_class_names = ["background"] + [
                f"class_{int(raw_id)}" for raw_id in self.query_class_ids[1:]
            ]
        else:
            query_class_names = list(query_class_names)
            if len(query_class_names) == (self.query_num_classes - 1):
                query_class_names = ["background"] + query_class_names
        self.query_class_names = tuple(str(v) for v in query_class_names)
        self.strict_query_class_id_validation = bool(strict_query_class_id_validation)
        query_raw_to_compact_size = max(256, max(self.query_class_ids) + 1)
        query_raw_to_compact = torch.full(
            (query_raw_to_compact_size,),
            fill_value=-1,
            dtype=torch.long,
        )
        for compact_id, raw_id in enumerate(self.query_class_ids):
            query_raw_to_compact[raw_id] = int(compact_id)
        self.register_buffer(
            "query_raw_to_compact_class_map",
            query_raw_to_compact,
            persistent=False,
        )
        self.register_buffer(
            "query_compact_to_raw_class_ids",
            torch.as_tensor(self.query_class_ids, dtype=torch.long),
            persistent=False,
        )
        self.query_cls_loss_weight = float(query_cls_loss_weight)
        if query_cls_loss_class_weights is None:
            query_cls_loss_class_weights = [1.0] * int(self.query_num_classes)
        query_cls_loss_class_weights = [float(v) for v in query_cls_loss_class_weights]
        self.register_buffer(
            "query_cls_loss_class_weights",
            torch.as_tensor(query_cls_loss_class_weights, dtype=torch.float32),
            persistent=False,
        )
        self.query_match_feature_source = str(query_match_feature_source)
        self.query_soft_assign_temp = float(query_soft_assign_temp)
        self.query_soft_assign_cost_weight = float(query_soft_assign_cost_weight)
        self.query_sim_match_cost_weight = float(query_sim_match_cost_weight)
        self.query_cls_match_cost_weight = float(query_cls_match_cost_weight)
        self.query_bev_dice_match_cost_weight = float(query_bev_dice_match_cost_weight)
        self.query_center_routed_loss_weight = float(query_center_routed_loss_weight)
        self.query_center_match_cost_weight = float(query_center_match_cost_weight)
        self.query_center_match_loss_type = str(query_center_match_loss_type).lower()
        self.query_matched_gmo_bce_occ_size = tuple(int(v) for v in query_matched_gmo_bce_occ_size)
        self.query_num_gaussians = int(query_num_gaussians)
        self.query_multi_gaussian_offset_max_m = tuple(float(v) for v in query_multi_gaussian_offset_max_m)
        self.query_multi_gaussian_sigma_min_m = tuple(float(v) for v in query_multi_gaussian_sigma_min_m)
        self.query_multi_gaussian_sigma_max_m = tuple(float(v) for v in query_multi_gaussian_sigma_max_m)
        self.query_multi_gaussian_sigma_reg_loss_weight = float(query_multi_gaussian_sigma_reg_loss_weight)
        self.query_multi_gaussian_sigma_reg_log_eps = float(query_multi_gaussian_sigma_reg_log_eps)
        self.query_multi_gaussian_pair_chunk = max(1, int(query_multi_gaussian_pair_chunk))
        self.query_multi_gaussian_weight_mode = str(query_multi_gaussian_weight_mode).lower()
        self.query_multi_gaussian_softplus_bias_init = float(query_multi_gaussian_softplus_bias_init)
        self.query_multi_gaussian_weight_reg_loss_weight = float(query_multi_gaussian_weight_reg_loss_weight)
        self.query_multi_gaussian_weight_reg_target_sum = float(query_multi_gaussian_weight_reg_target_sum)
        self.query_embed_dim = int(query_embed_dim)
        self.query_num_queries = int(query_num_queries)
        self.query_transformer_num_layers = int(query_transformer_num_layers)
        self.query_id_reinject_scale = float(query_id_reinject_scale)
        self.query_ca_kv_identity_init = bool(query_ca_kv_identity_init)
        self.query_ca_attn_tau = float(query_ca_attn_tau)
        self.query_decor_loss_weight = float(query_decor_loss_weight)
        self.query_attn_vis_dir = str(query_attn_vis_dir)
        self.use_query_attn_bbox_loss = bool(use_query_attn_bbox_loss)
        self.query_attn_bbox_loss_weight = float(query_attn_bbox_loss_weight)
        self.query_attn_bbox_unmatched_weight = float(query_attn_bbox_unmatched_weight)
        self.query_attn_bbox_eps = float(query_attn_bbox_eps)
        self.query_attn_bbox_camera_reduce_mode = str(query_attn_bbox_camera_reduce_mode).lower()
        self.query_attn_bbox_unmatched_mode = str(query_attn_bbox_unmatched_mode).lower()
        self.query_attn_match_cost_weight = float(query_attn_match_cost_weight)
        self.query_attn_match_metric = str(query_attn_match_metric).lower()
        self.query_attn_match_pred_norm = str(query_attn_match_pred_norm).lower()
        self.query_attn_match_eps = float(query_attn_match_eps)
        self.use_query_attn_cam_gaussian_score = bool(use_query_attn_cam_gaussian_score)
        self.query_attn_cam_frame_mode = str(query_attn_cam_frame_mode).lower()
        self.query_attn_cam_target_mode = str(query_attn_cam_target_mode).lower()
        self.query_attn_cam_metric = str(query_attn_cam_metric).lower()
        self.query_attn_cam_camera_reduce_mode = str(query_attn_cam_camera_reduce_mode).lower()
        self.query_attn_cam_eps = float(query_attn_cam_eps)
        self.query_attn_cam_gaussian_truncate_sigma = float(query_attn_cam_gaussian_truncate_sigma)
        self.query_attn_cam_target_binary_threshold = float(query_attn_cam_target_binary_threshold)
        self.query_attn_cam_score_norm_mode = str(query_attn_cam_score_norm_mode).lower()

        self.query_attn_softargmax_tau = float(query_attn_softargmax_tau)
        self.query_depth_loss_weight = float(query_depth_loss_weight)
        self.query_depth_label_smoothing = float(query_depth_label_smoothing)
        self.query_inst_depth_num_bins = int(query_inst_depth_num_bins)
        self.query_inst_depth_range_mode = str(query_inst_depth_range_mode).lower()
        self.query_inst_depth_min = float(query_inst_depth_min)
        self.query_inst_depth_max = float(query_inst_depth_max)
        self._query_train_iter = 0
        self._train_iter_synced = False
        if self.center_only_mode:
            self.use_gmo_bce_loss = False
            self.use_query_gmo_dice_loss = False

        context_feat_dim = self._get_context_feat_dim_from_depth_net()
        geo_input_dim = int(getattr(self.img_view_transformer, "cam_channels", 27))

        self.transformer = TransformerModule(
            feat_dim=context_feat_dim,
            geo_input_dim=geo_input_dim,
            num_queries=self.query_num_queries,
            num_heads=4,
            num_layers=self.query_transformer_num_layers,
            num_cams=6,
            embed_dim=self.query_embed_dim,
            max_time=self.time_receptive_field,
            query_id_reinject_scale=self.query_id_reinject_scale,
            ca_kv_identity_init=self.query_ca_kv_identity_init,
            ca_attn_tau=self.query_ca_attn_tau,
            query_decor_loss_weight=self.query_decor_loss_weight,
            attn_vis_dir=self.query_attn_vis_dir,
        )

        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_predictor = builder.build_neck(occ_predictor)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)

        self.point_cloud_range = point_cloud_range
        self.spatial_extent3d = (self.point_cloud_range[3]-self.point_cloud_range[0], \
                                    self.point_cloud_range[4]-self.point_cloud_range[1], \
                                         self.point_cloud_range[5]-self.point_cloud_range[2])
        self.ego_center_shift_proportion_x = abs(self.point_cloud_range[0])/(self.point_cloud_range[3]-self.point_cloud_range[0])
        self.ego_center_shift_proportion_y = abs(self.point_cloud_range[1])/(self.point_cloud_range[4]-self.point_cloud_range[1])
        self.ego_center_shift_proportion_z = abs(self.point_cloud_range[2])/(self.point_cloud_range[5]-self.point_cloud_range[2])


        self.query_head = QueryHead(
            embed_dim=self.query_embed_dim,
            num_past_frames=self.time_receptive_field,
            num_future_frames=self.n_future_frames_plus,
            query_present_only=self.query_present_only,
            query_pred_num_frames=self.query_pred_num_frames,
            query_traj_num_steps=self.n_future_frames,
            query_traj_residual_max_m=self.query_traj_residual_max_m,
            query_traj_prior_detach=self.query_traj_prior_detach,
            num_overlap_frames=self.query_overlap_frames,
            num_query_classes=self.query_num_classes,
            query_class_ids=self.query_class_ids,
            query_class_names=self.query_class_names,
            query_depth_num_bins=self.query_inst_depth_num_bins,
            center_out_dim=3,
            offset_out_dim=3,
            num_pcd=256,
            query_num_gaussians=self.query_num_gaussians,
            query_multi_gaussian_offset_max_m=self.query_multi_gaussian_offset_max_m,
            query_multi_gaussian_sigma_min_m=self.query_multi_gaussian_sigma_min_m,
            query_multi_gaussian_sigma_max_m=self.query_multi_gaussian_sigma_max_m,
            query_multi_gaussian_sigma_reg_loss_weight=self.query_multi_gaussian_sigma_reg_loss_weight,
            query_multi_gaussian_sigma_reg_log_eps=self.query_multi_gaussian_sigma_reg_log_eps,
            query_multi_gaussian_weight_mode=self.query_multi_gaussian_weight_mode,
            query_multi_gaussian_softplus_bias_init=self.query_multi_gaussian_softplus_bias_init,
            query_multi_gaussian_weight_reg_loss_weight=self.query_multi_gaussian_weight_reg_loss_weight,
            query_multi_gaussian_weight_reg_target_sum=self.query_multi_gaussian_weight_reg_target_sum,
            point_cloud_range=point_cloud_range,
            spatial_extent3d=self.spatial_extent3d,
            query_feat_cosine_threshold=self.query_feat_cosine_threshold,
            query_center_distance_threshold_m=self.query_center_distance_threshold_m,
            detach_query_for_center_default=self.query_center_loss_detach_query_feat,
            debug_vis_every=self.debug_query_vis_every,
            debug_vis_dir=self.debug_query_vis_dir,
            center_only_mode=self.center_only_mode,
            debug_query_center_marker_radius=self.debug_query_center_marker_radius,
            debug_query_confidence_vis_threshold=self.debug_query_confidence_vis_threshold,
            debug_query_objectness_vis_threshold=debug_query_objectness_vis_threshold,
            debug_query_gaussian_vis_mode=self.debug_query_gaussian_vis_mode,
            debug_query_gaussian_prob_threshold=self.debug_query_gaussian_prob_threshold,
            debug_query_gaussian_prob_alpha_scale=self.debug_query_gaussian_prob_alpha_scale)
        self.context_depth_proxy_head = nn.Conv2d(
            in_channels=int(context_feat_dim),
            out_channels=int(self.img_view_transformer.D),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        
        self.voxelizer = SoftVoxelizerOneAdd(
            point_cloud_range=point_cloud_range,
            voxel_size=0.2,
            occ_size=(512, 512, 40),
            as_prob=True,
            lambda_occ=1.0
        )
        low_x, low_y, low_z = self.query_matched_gmo_bce_occ_size
        range_x = float(self.point_cloud_range[3] - self.point_cloud_range[0])
        range_y = float(self.point_cloud_range[4] - self.point_cloud_range[1])
        range_z = float(self.point_cloud_range[5] - self.point_cloud_range[2])
        self.matched_gmo_voxelizer = SoftVoxelizerOneAdd(
            point_cloud_range=point_cloud_range,
            voxel_size=(range_x / float(low_x), range_y / float(low_y), range_z / float(low_z)),
            occ_size=(low_x, low_y, low_z),
            as_prob=True,
            lambda_occ=1.0,
            gaussian_truncate_sigma=float(self.voxelizer.gaussian_truncate_sigma),
            gaussian_sigma_floor_vox=float(self.voxelizer.gaussian_sigma_floor_vox),
        )

        self.n_cam = 6

        self.mean_weight= nn.Parameter(torch.ones(1) * 0.1, requires_grad=True)
        self.max_weight= nn.Parameter(torch.ones(1) * 1.0, requires_grad=True)

    def init_weights(self):
        super().init_weights()

    def set_train_iteration(self, train_iter: int, one_based: bool = True) -> None:
        step = int(train_iter)
        if not one_based:
            step += 1
        step = max(0, step)
        self._query_train_iter = step
        self._pretrain_lss_vis_iter = step
        self._train_iter_synced = True
        if hasattr(self, "query_head") and hasattr(self.query_head, "set_train_iteration"):
            self.query_head.set_train_iteration(step, one_based=True)
        if hasattr(self, "transformer") and hasattr(self.transformer, "set_train_iteration"):
            self.transformer.set_train_iteration(step, one_based=True)

    def _get_query_train_iteration(self, advance_if_unsynced: bool = False) -> int:
        if not self._train_iter_synced and advance_if_unsynced:
            self._query_train_iter += 1
        return int(self._query_train_iter)

    def _get_pretrain_vis_iteration(self, advance_if_unsynced: bool = False) -> int:
        if not self._train_iter_synced and advance_if_unsynced:
            self._pretrain_lss_vis_iter += 1
        return int(self._pretrain_lss_vis_iter)

    def _get_context_feat_dim_from_depth_net(self):

        depth_net = self.img_view_transformer.depth_net
        return int(depth_net.context_conv.out_channels)

    def train(self, mode=True):
        super().train(mode)
        return self

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        backbone_feats = self.img_backbone(imgs)
        if isinstance(backbone_feats, (list, tuple)):
            resnet_last = backbone_feats[-1]
        else:
            resnet_last = backbone_feats
        _, res_c, res_h, res_w = resnet_last.shape
        resnet_last = resnet_last.view(B, N, res_c, res_h, res_w)
        if self.with_img_neck:
            x = self.img_neck(backbone_feats)
            if type(x) in [list, tuple]:
                x = x[0]
        else:
            x = backbone_feats
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return {
            'x': x,
            'img_feats': [x.clone()],
            'resnet_last': resnet_last,
        }
    
    @force_fp32()
    def occ_encoder(self, x):
        b, t, _, _, _ = x.shape
        x = x.reshape(b, -1, *x.shape[3:])
        x = self.occ_encoder_backbone(x)
        x = self.occ_predictor(x)
        x = self.occ_encoder_neck(x)

        return x

    def mat2pose_vec(self, matrix: torch.Tensor):
        """
        Converts a 4x4 pose matrix into a 6-dof pose vector
        Args:
            matrix (ndarray): 4x4 pose matrix
        Returns:
            vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
            rotation components (rx, ry, rz)
        """

        # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
        rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

        # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
        cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
        roty = torch.atan2(matrix[..., 0, 2], cosy)

        # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
        rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

        rotation = torch.stack((rotx, roty, rotz), dim=-1)

        # Extract translation params
        translation = matrix[..., :3, 3]
        return torch.cat((translation, rotation), dim=-1)

    def pack_dbatch_and_dtime(self, x):
        b = x.shape[0]
        s = x.shape[1]
        x = x.view(b*s, *x.shape[2:])
        return x
    
    def unpack_dbatch_and_dtime(self, x, b, s):
        assert (b*s) == x.shape[0]
        x = x.view(b, s, *x.shape[1:])
        return x

    def _extract_depth_and_context_for_query(self, x, mlp_input_seq):


        depth_net = self.img_view_transformer.depth_net
        t, n, c, h, w = x.shape
        x_flat = x.reshape(t * n, c, h, w)
        reduced_x, mlp_norm = depth_net.prepare_inputs(x_flat, mlp_input_seq)

        context_se = depth_net.context_mlp(mlp_norm)[..., None, None]
        context_flat = depth_net.context_se(reduced_x, context_se)
        context_flat = depth_net.context_conv(context_flat)
        # Keep depth_net usage only for context feature generation.
        depth_digit = self.context_depth_proxy_head(context_flat.to(torch.float32))
        context_seq = context_flat.view(t, n, context_flat.shape[1], h, w)
        return depth_digit, context_flat, context_seq


    def extract_img_feat(
        self,
        img_inputs_seq,
        img_metas,
        run_query_transformer=True,
        return_instance_img_debug=False,
        return_query_match_inputs=False,
        frame_start_idx=0,
        frame_end_idx=None,
        update_internal_state=True,
    ):
        '''
        Extract features of sequential input images
        '''
        imgs_seq, rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq, _gt_depths_seq, _sensor2sensors_seq = img_inputs_seq
        del img_metas

        batch_size = int(imgs_seq.shape[0])
        seq_total = int(imgs_seq.shape[1])
        if update_internal_state:
            self.batch_size = batch_size
            self.sequence_length = seq_total

        s_idx = max(0, int(frame_start_idx))
        if frame_end_idx is None:
            if s_idx == 0:
                e_idx = min(seq_total, int(self.time_receptive_field))
            else:
                e_idx = seq_total
        else:
            e_idx = min(seq_total, int(frame_end_idx))

        imgs_seq = imgs_seq[:, s_idx:e_idx, ...].contiguous()
        rots_seq = rots_seq[:, s_idx:e_idx, ...].contiguous()
        trans_seq = trans_seq[:, s_idx:e_idx, ...].contiguous()
        intrins_seq = intrins_seq[:, s_idx:e_idx, ...].contiguous()
        post_rots_seq = post_rots_seq[:, s_idx:e_idx, ...].contiguous()
        post_trans_seq = post_trans_seq[:, s_idx:e_idx, ...].contiguous()
        seq_used = int(imgs_seq.shape[1])

        imgs_seq_bt = imgs_seq
        rots_seq_bt = rots_seq
        trans_seq_bt = trans_seq
        intrins_seq_bt = intrins_seq
        post_rots_seq_bt = post_rots_seq
        post_trans_seq_bt = post_trans_seq
        
        imgs_seq = self.pack_dbatch_and_dtime(imgs_seq)
        rots_seq = self.pack_dbatch_and_dtime(rots_seq)
        trans_seq = self.pack_dbatch_and_dtime(trans_seq)
        intrins_seq = self.pack_dbatch_and_dtime(intrins_seq)
        post_rots_seq = self.pack_dbatch_and_dtime(post_rots_seq)
        post_trans_seq = self.pack_dbatch_and_dtime(post_trans_seq)

        self.n_cam = imgs_seq.shape[1]
        
        img_enc_feats = self.image_encoder(imgs_seq)
        x = img_enc_feats['x']
        img_feats = img_enc_feats['img_feats']
        resnet_last_seq = img_enc_feats['resnet_last']

        mlp_input_seq = self.img_view_transformer.get_mlp_input(
            rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq
        )
        # NOTE for future query-only test path:
        # `context_seq` still depends on depth_net.bn/reduce_conv/context_*.
        # Query-only inference may skip depth prediction + lift/splat teacher path,
        # but it must continue loading those context-side modules from checkpoint.
        depth_digit, context_flat, context_seq = self._extract_depth_and_context_for_query(
            x, mlp_input_seq
        )

        query_inst, query_img_feat_pooled = None, None
        query_attn_weights = None
        if run_query_transformer:
            trans_out = self.transformer(
                context_seq,
                return_attn_pool=True,
                return_attn_weights=True,
                vis_images=imgs_seq,
            )
            query_inst, query_img_feat_pooled, query_attn_weights = trans_out

        geo_inputs = [rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq, None, mlp_input_seq]
        x, depth = self.img_view_transformer([x] + geo_inputs + [depth_digit, context_flat])

        instance_img_debug_bundle = None
        if return_instance_img_debug:
            context_seq_btnchw = self.unpack_dbatch_and_dtime(
                context_seq, batch_size, seq_used
            )
            instance_img_debug_bundle = {
                "context_seq_btnchw": context_seq_btnchw.detach(),
                "imgs_seq_bt": imgs_seq_bt.detach(),
                "rots_seq_bt": rots_seq_bt.detach(),
                "trans_seq_bt": trans_seq_bt.detach(),
                "intrins_seq_bt": intrins_seq_bt.detach(),
                "post_rots_seq_bt": post_rots_seq_bt.detach(),
                "post_trans_seq_bt": post_trans_seq_bt.detach(),
                "resnet_last_seq_btnchw": self.unpack_dbatch_and_dtime(
                    resnet_last_seq, batch_size, seq_used
                ).detach(),
                "frame_start_idx": int(s_idx),
                "frame_end_idx_exclusive": int(e_idx),
            }

        query_match_inputs = None
        if return_query_match_inputs:
            context_seq_btnchw = self.unpack_dbatch_and_dtime(
                context_seq, batch_size, seq_used
            )
            query_match_inputs = {
                "context_seq_tnchw": context_seq_btnchw[0].detach(),
                "rots_tn33": rots_seq_bt[0].detach(),
                "trans_tn3": trans_seq_bt[0].detach(),
                "intrins_tn33": intrins_seq_bt[0].detach(),
                "post_rots_tn33": post_rots_seq_bt[0].detach(),
                "post_trans_tn3": post_trans_seq_bt[0].detach(),
                "img_h": int(imgs_seq_bt.shape[-2]),
                "img_w": int(imgs_seq_bt.shape[-1]),
                "feat_h": int(context_seq_btnchw.shape[-2]),
                "feat_w": int(context_seq_btnchw.shape[-1]),
                "frame_start_idx": int(s_idx),
                "frame_end_idx_exclusive": int(e_idx),
            }

        return (
            query_inst,
            query_img_feat_pooled,
            depth,
            img_feats,
            x,
            query_attn_weights,
            instance_img_debug_bundle,
            query_match_inputs,
        )


    def warp_features(self, x, flow):
        '''
        Warp features by motion flow
        '''
        if flow is None:
            return x

        _, _, dx, dy, dz = x.shape

        # normalize 3D motion flow
        flow[:,0,-1] =flow[:,0,-1]*dx/self.spatial_extent3d[0]
        flow[:,1,-1] =flow[:,1,-1]*dy/self.spatial_extent3d[1]
        flow[:,2,-1] =flow[:,2,-1]*dz/self.spatial_extent3d[2]

        nx, ny, nz = torch.meshgrid(torch.arange(dx, dtype=torch.float, device=x.device), \
                                    torch.arange(dy, dtype=torch.float, device=x.device), \
                                    torch.arange(dz, dtype=torch.float, device=x.device))
        tmp = torch.ones((dx, dy, dz), device=x.device)
        grid = torch.stack((nx, ny, nz, tmp), dim=-1)

        # centralize shift
        shift_x = self.ego_center_shift_proportion_x * dx
        shift_y = self.ego_center_shift_proportion_y * dy
        shift_z = self.ego_center_shift_proportion_z * dz
        
        grid[:, :, :, 0] = grid[:, :, :, 0] - shift_x
        grid[:, :, :, 1] = grid[:, :, :, 1] - shift_y
        grid[:, :, :, 2] = grid[:, :, :, 2] - shift_z
        grid = grid.view(dx*dy*dz, grid.shape[-1]).unsqueeze(-1)

        transformation = flow.unsqueeze(1)
        transformed_grid = transformation @ grid
        transformed_grid = transformed_grid.squeeze(-1)
        transformed_grid = transformed_grid.view(-1, 4)

        # de-centralize
        transformed_grid[:, 0] = (transformed_grid[:, 0] + shift_x)
        transformed_grid[:, 1] = (transformed_grid[:, 1] + shift_y)
        transformed_grid[:, 2] = (transformed_grid[:, 2] + shift_z)
        transformed_grid = transformed_grid.round().long()

        # de-normalize
        grid = grid.squeeze(-1) 
        grid = grid.view(-1, 4)
        grid[:, 0] = (grid[:, 0] + shift_x)
        grid[:, 1] = (grid[:, 1] + shift_y)
        grid[:, 2] = (grid[:, 2] + shift_z)
        grid = grid.round().long()

        kept = (transformed_grid[:,0] >= 0) & (transformed_grid[:,0] <dx) \
               & (transformed_grid[:,1] >= 0) & (transformed_grid[:,1] <dy) \
               & (transformed_grid[:,2] >= 0) & (transformed_grid[:,2] < dz)

        transformed_grid = transformed_grid[kept]
        grid = grid[kept]

        warped_x =  torch.zeros_like(x, device=x.device)

        # loop for reduce memory cost
        interval_num = 32
        gap = transformed_grid.shape[0]//interval_num
        for tt in range(interval_num-1):
            start_idx_tt = int(tt*gap)
            end_idx_tt = int((tt+1)*gap)
            ixx = transformed_grid[start_idx_tt:end_idx_tt, 0]
            ixy = transformed_grid[start_idx_tt:end_idx_tt, 1]
            ixz = transformed_grid[start_idx_tt:end_idx_tt, 2]

            ixx_ori = grid[start_idx_tt:end_idx_tt, 0]
            ixy_ori = grid[start_idx_tt:end_idx_tt, 1]
            ixz_ori = grid[start_idx_tt:end_idx_tt, 2]

            warped_x[0, :, ixx, ixy, ixz] = x[0, :, ixx_ori, ixy_ori, ixz_ori]

        return warped_x 

    def cumulative_warp_occ(self, lifted_feature_seq, future_egomotion):
        '''
        Warp sequential voxel features to the present frame by ego pose updaextract_feattes
        '''
        
        future_egomotion = future_egomotion[:, :self.time_receptive_field, ...].contiguous()
        
        out = [lifted_feature_seq[:, -1]]
        cum_future_egomotion = future_egomotion[:, -2]
        for t in reversed(range(self.time_receptive_field - 1)): 
            out.append(self.warp_features(lifted_feature_seq[:, t], cum_future_egomotion))
            cum_future_egomotion = cum_future_egomotion @ future_egomotion[:, t - 1]
        
        return torch.stack(out[::-1], 1)

    @staticmethod
    def _yaw_delta_from_transform_xy(transform_44: torch.Tensor) -> torch.Tensor:
        rot2 = transform_44[:2, :2].to(torch.float32)
        return torch.atan2(rot2[1, 0], rot2[0, 0])

    def _build_query_trajectory_geometry_from_present(
        self,
        present_centers_world_tq3: torch.Tensor,
        present_sigmas_world_tq3: torch.Tensor = None,
        present_mix_centers_world_tqg3: torch.Tensor = None,
        present_mix_sigmas_world_tqg3: torch.Tensor = None,
        present_mix_yaw_tqg: torch.Tensor = None,
        present_mix_weights_tqg: torch.Tensor = None,
        traj_offsets_fq2: torch.Tensor = None,
        future_egomotion: torch.Tensor = None,
    ):
        """
        Build trajectory-horizon query geometry from present lifted centers.
        Output timeline is [t, t+1, ..., t+n_future_frames] in present-frame
        lidar coordinates. `traj_offsets_fq2` is interpreted as present-frame
        per-step delta: (t->t+1), (t+1->t+2), ...
        """
        out = {
            "centers_world_tq3": present_centers_world_tq3,
            "sigmas_world_tq3": present_sigmas_world_tq3,
            "mix_centers_world_tqg3": present_mix_centers_world_tqg3,
            "mix_sigmas_world_tqg3": present_mix_sigmas_world_tqg3,
            "mix_yaw_tqg": present_mix_yaw_tqg,
            "mix_weights_tqg": present_mix_weights_tqg,
            "traj_offsets_fq2": traj_offsets_fq2,
            "traj_frame_indices_t": None,
        }
        if (
            (not torch.is_tensor(present_centers_world_tq3))
            or present_centers_world_tq3.dim() != 3
            or int(present_centers_world_tq3.shape[0]) != 1
            or int(present_centers_world_tq3.shape[-1]) != 3
        ):
            return out
        future_steps = max(0, int(self.n_future_frames))
        if future_steps <= 0:
            return out

        q_count = int(present_centers_world_tq3.shape[1])
        if q_count <= 0:
            return out

        traj_offsets = traj_offsets_fq2
        if (not torch.is_tensor(traj_offsets)) or traj_offsets.dim() != 3:
            traj_offsets = present_centers_world_tq3.new_zeros((future_steps, q_count, 2))
        else:
            traj_offsets = traj_offsets.to(
                device=present_centers_world_tq3.device,
                dtype=present_centers_world_tq3.dtype,
            )
            if int(traj_offsets.shape[0]) != future_steps:
                if int(traj_offsets.shape[0]) < future_steps:
                    pad = present_centers_world_tq3.new_zeros(
                        (future_steps - int(traj_offsets.shape[0]), q_count, 2)
                    )
                    traj_offsets = torch.cat([traj_offsets, pad], dim=0)
                else:
                    traj_offsets = traj_offsets[:future_steps].contiguous()
        out["traj_offsets_fq2"] = traj_offsets

        present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
        frame_indices = [int(present_global_idx + step) for step in range(future_steps + 1)]
        present_centers_q3 = present_centers_world_tq3[0].to(torch.float32)

        centers_traj = []
        mix_centers_traj = []

        mix_enabled = (
            torch.is_tensor(present_mix_centers_world_tqg3)
            and torch.is_tensor(present_mix_sigmas_world_tqg3)
            and torch.is_tensor(present_mix_yaw_tqg)
            and torch.is_tensor(present_mix_weights_tqg)
            and present_mix_centers_world_tqg3.dim() == 4
            and present_mix_sigmas_world_tqg3.dim() == 4
            and present_mix_yaw_tqg.dim() == 3
            and present_mix_weights_tqg.dim() == 3
            and int(present_mix_centers_world_tqg3.shape[0]) == 1
            and tuple(present_mix_centers_world_tqg3.shape) == tuple(present_mix_sigmas_world_tqg3.shape)
            and tuple(present_mix_centers_world_tqg3.shape[:3]) == tuple(present_mix_yaw_tqg.shape)
            and tuple(present_mix_centers_world_tqg3.shape[:3]) == tuple(present_mix_weights_tqg.shape)
        )
        if mix_enabled:
            mix_q = int(present_mix_centers_world_tqg3.shape[1])
            present_mix_qg3 = present_mix_centers_world_tqg3[0].to(torch.float32)

        pc_min = present_centers_world_tq3.new_tensor(self.point_cloud_range[:3]).view(1, 3)
        pc_max = present_centers_world_tq3.new_tensor(self.point_cloud_range[3:]).view(1, 3)
        cur_centers_q3 = present_centers_q3
        cur_mix_qg3 = present_mix_qg3 if mix_enabled else None

        for step in range(future_steps + 1):
            if step > 0:
                delta_xy_q2 = traj_offsets[step - 1].to(torch.float32)
                cur_centers_q3 = cur_centers_q3.clone()
                cur_centers_q3[:, :2] = cur_centers_q3[:, :2] + delta_xy_q2
            cur_centers_q3 = torch.max(
                torch.min(cur_centers_q3, pc_max.to(torch.float32)),
                pc_min.to(torch.float32),
            )
            centers_traj.append(cur_centers_q3)

            if mix_enabled:
                if step > 0:
                    cur_mix_qg3 = cur_mix_qg3.clone()
                    cur_mix_qg3[..., :2] = cur_mix_qg3[..., :2] + delta_xy_q2[:, None, :]
                cur_mix_qg3 = torch.max(
                    torch.min(cur_mix_qg3, pc_max.to(torch.float32).view(1, 1, 3)),
                    pc_min.to(torch.float32).view(1, 1, 3),
                )
                mix_centers_traj.append(cur_mix_qg3)

        centers_world_traj_tq3 = torch.stack(centers_traj, dim=0).to(
            device=present_centers_world_tq3.device,
            dtype=present_centers_world_tq3.dtype,
        ).contiguous()
        out["centers_world_tq3"] = centers_world_traj_tq3
        out["traj_frame_indices_t"] = present_centers_world_tq3.new_tensor(
            frame_indices, dtype=torch.long
        )

        if torch.is_tensor(present_sigmas_world_tq3) and present_sigmas_world_tq3.dim() == 3:
            sigma_present_q3 = present_sigmas_world_tq3[0].to(
                device=present_centers_world_tq3.device, dtype=present_centers_world_tq3.dtype
            )
            out["sigmas_world_tq3"] = sigma_present_q3.unsqueeze(0).expand(
                int(centers_world_traj_tq3.shape[0]), -1, -1
            ).contiguous()

        if mix_enabled:
            mix_centers_world_traj = torch.stack(mix_centers_traj, dim=0).to(
                device=present_centers_world_tq3.device,
                dtype=present_centers_world_tq3.dtype,
            ).contiguous()
            sigma_present_qg3 = present_mix_sigmas_world_tqg3[0].to(
                device=present_centers_world_tq3.device,
                dtype=present_centers_world_tq3.dtype,
            )
            weight_present_qg = present_mix_weights_tqg[0].to(
                device=present_centers_world_tq3.device,
                dtype=present_centers_world_tq3.dtype,
            )
            out["mix_centers_world_tqg3"] = mix_centers_world_traj
            out["mix_sigmas_world_tqg3"] = sigma_present_qg3.unsqueeze(0).expand(
                int(centers_world_traj_tq3.shape[0]), -1, -1, -1
            ).contiguous()
            out["mix_yaw_tqg"] = present_mix_yaw_tqg[0].to(
                device=present_centers_world_tq3.device,
                dtype=present_mix_yaw_tqg.dtype,
            ).unsqueeze(0).expand(
                int(centers_world_traj_tq3.shape[0]), -1, -1
            ).contiguous()
            out["mix_weights_tqg"] = weight_present_qg.unsqueeze(0).expand(
                int(centers_world_traj_tq3.shape[0]), -1, -1
            ).contiguous()
        return out

    def _align_query_vis_geometry_to_present_frame(
        self,
        centers_world_tq3: torch.Tensor,
        traj_frame_indices_t: torch.Tensor = None,
        future_egomotion: torch.Tensor = None,
        gaussian_sigmas_world_tq3: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_yaw_tqg: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
    ):
        out = {
            "centers_world_tq3": centers_world_tq3,
            "gaussian_sigmas_world_tq3": gaussian_sigmas_world_tq3,
            "mixture_centers_world_tqg3": mixture_centers_world_tqg3,
            "mixture_sigmas_world_tqg3": mixture_sigmas_world_tqg3,
            "mixture_yaw_tqg": mixture_yaw_tqg,
            "mixture_weights_tqg": mixture_weights_tqg,
        }
        if (
            (not torch.is_tensor(centers_world_tq3))
            or centers_world_tq3.dim() != 3
            or int(centers_world_tq3.shape[0]) <= 1
            or (not torch.is_tensor(traj_frame_indices_t))
            or (not torch.is_tensor(future_egomotion))
        ):
            return out

        aligned_centers_tq3 = self._align_query_centers_to_present_frame(
            centers_world_tq3=centers_world_tq3,
            future_egomotion=future_egomotion,
            traj_frame_indices_t=traj_frame_indices_t,
        )
        out["centers_world_tq3"] = aligned_centers_tq3

        mix_enabled = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_yaw_tqg)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_yaw_tqg.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
        )
        if not mix_enabled:
            return out

        t_count, q_count, g_count, _ = [int(v) for v in mixture_centers_world_tqg3.shape]
        mix_flat_tkg3 = mixture_centers_world_tqg3.reshape(t_count, q_count * g_count, 3)
        aligned_mix_flat_tkg3 = self._align_query_centers_to_present_frame(
            centers_world_tq3=mix_flat_tkg3,
            future_egomotion=future_egomotion,
            traj_frame_indices_t=traj_frame_indices_t,
        )
        out["mixture_centers_world_tqg3"] = aligned_mix_flat_tkg3.reshape(
            t_count, q_count, g_count, 3
        ).contiguous()
        out["mixture_sigmas_world_tqg3"] = mixture_sigmas_world_tqg3
        out["mixture_weights_tqg"] = mixture_weights_tqg

        frame_idx = traj_frame_indices_t.to(torch.long).reshape(-1)
        if int(frame_idx.numel()) < t_count:
            return out
        frame_indices = [int(v) for v in frame_idx[:t_count].detach().cpu().tolist()]

        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[0]
        ego = ego.to(device=centers_world_tq3.device, dtype=torch.float32)
        present_global_idx = int(
            getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1))
        )
        lidar_target_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=frame_indices,
            present_global_idx=present_global_idx,
        )

        yaw_aligned = []
        yaw_src_tqg = mixture_yaw_tqg.to(torch.float32)
        for step in range(t_count):
            yaw_delta = self._yaw_delta_from_transform_xy(
                lidar_target_to_present[step].to(
                    device=centers_world_tq3.device, dtype=torch.float32
                )
            )
            yaw_step_qg = torch.atan2(
                torch.sin(yaw_src_tqg[step] + yaw_delta),
                torch.cos(yaw_src_tqg[step] + yaw_delta),
            )
            yaw_aligned.append(yaw_step_qg)
        out["mixture_yaw_tqg"] = torch.stack(yaw_aligned, dim=0).to(
            device=mixture_yaw_tqg.device,
            dtype=mixture_yaw_tqg.dtype,
        ).contiguous()
        return out


    def extract_feat_query(
        self,
        img_inputs_seq,
        img_metas,
        future_egomotion,
        gt_segmentation_instance3d_txyz_for_attn=None,
        fallback_segmentation_instance3d_txyz_for_attn=None,
        voxelize=True,
        return_bev_occ_feats=False,
        return_instance_img_debug=False,
        return_instance_img_debug_fullseq=True,
        return_query_cam_gaussian_vis_debug=False,
        return_query_match_inputs=False,
    ):
        '''
        Extract voxel features from input sequential images
        '''
        voxel_feats = None
        query_attn_weights = None
        query_attn_bbox_targets = None
        instance_img_debug_bundle = None
        query_match_inputs = None
        need_hist_debug_bundle = bool(return_instance_img_debug and (not return_instance_img_debug_fullseq))
        if img_inputs_seq is not None:
            (
                query_inst,
                query_img_feat_pooled,
                depth,
                img_feats,
                voxel_feats,
                query_attn_weights,
                instance_img_debug_bundle,
                query_match_inputs,
            ) = self.extract_img_feat(
                img_inputs_seq,
                img_metas,
                return_instance_img_debug=need_hist_debug_bundle,
                return_query_match_inputs=return_query_match_inputs,
                frame_start_idx=0,
                frame_end_idx=self.time_receptive_field,
                update_internal_state=True,
            )

        if return_instance_img_debug and return_instance_img_debug_fullseq:
            seq_total = int(img_inputs_seq[0].shape[1]) if (isinstance(img_inputs_seq, (list, tuple)) and torch.is_tensor(img_inputs_seq[0])) else int(self.time_receptive_field)
            vis_len = int(min(seq_total, int(self.n_future_frames_plus)))
            vis_start = max(0, seq_total - vis_len)
            vis_end = seq_total
            (
                _dbg_query_inst,
                _dbg_query_img_feat_pooled,
                _dbg_depth,
                _dbg_img_feats,
                _dbg_x,
                _dbg_query_attn_weights,
                debug_bundle_vis,
                _dbg_match_inputs,
            ) = self.extract_img_feat(
                img_inputs_seq,
                img_metas,
                run_query_transformer=False,
                return_instance_img_debug=True,
                return_query_match_inputs=False,
                frame_start_idx=vis_start,
                frame_end_idx=vis_end,
                update_internal_state=False,
            )
            if debug_bundle_vis is not None:
                vis_frame_indices = list(range(vis_start, vis_end))
                present_global = int(self.time_receptive_field - 1)
                present_local = int(present_global - vis_start)
                debug_bundle_vis["vis_global_frame_indices"] = vis_frame_indices
                debug_bundle_vis["present_global_frame_idx"] = present_global
                debug_bundle_vis["present_local_vis_idx"] = present_local
                debug_bundle_vis["sequence_length"] = seq_total
            instance_img_debug_bundle = debug_bundle_vis
        elif return_query_cam_gaussian_vis_debug and return_instance_img_debug_fullseq:
            seq_total = int(img_inputs_seq[0].shape[1]) if (
                isinstance(img_inputs_seq, (list, tuple)) and torch.is_tensor(img_inputs_seq[0])
            ) else int(self.time_receptive_field)
            vis_len = int(min(seq_total, int(self.n_future_frames_plus)))
            vis_start = max(0, seq_total - vis_len)
            vis_end = seq_total
            debug_bundle_vis = self._build_camera_debug_bundle_from_inputs(
                img_inputs_seq,
                frame_start_idx=vis_start,
                frame_end_idx=vis_end,
            )
            if debug_bundle_vis is not None:
                vis_frame_indices = list(range(vis_start, vis_end))
                present_global = int(self.time_receptive_field - 1)
                present_local = int(present_global - vis_start)
                debug_bundle_vis["vis_global_frame_indices"] = vis_frame_indices
                debug_bundle_vis["present_global_frame_idx"] = present_global
                debug_bundle_vis["present_local_vis_idx"] = present_local
                debug_bundle_vis["sequence_length"] = seq_total
            instance_img_debug_bundle = debug_bundle_vis
        elif return_instance_img_debug and isinstance(instance_img_debug_bundle, dict):
            vis_start = int(instance_img_debug_bundle.get("frame_start_idx", 0))
            vis_end = int(
                instance_img_debug_bundle.get(
                    "frame_end_idx_exclusive",
                    vis_start + int(instance_img_debug_bundle["imgs_seq_bt"].shape[1]),
                )
            )
            seq_total = int(img_inputs_seq[0].shape[1]) if (
                isinstance(img_inputs_seq, (list, tuple)) and torch.is_tensor(img_inputs_seq[0])
            ) else int(self.time_receptive_field)
            vis_frame_indices = list(range(vis_start, vis_end))
            present_global = int(self.time_receptive_field - 1)
            present_local = int(present_global - vis_start)
            instance_img_debug_bundle["vis_global_frame_indices"] = vis_frame_indices
            instance_img_debug_bundle["present_global_frame_idx"] = present_global
            instance_img_debug_bundle["present_local_vis_idx"] = present_local
            instance_img_debug_bundle["sequence_length"] = seq_total
            # CAM-vis-only path does not need heavyweight feature tensors.
            instance_img_debug_bundle.pop("context_seq_btnchw", None)
            instance_img_debug_bundle.pop("resnet_last_seq_btnchw", None)

        if (
            (
                self.use_query_attn_bbox_loss
                or (self.query_attn_match_cost_weight > 0.0)
            )
            and return_query_match_inputs
            and (
                torch.is_tensor(gt_segmentation_instance3d_txyz_for_attn)
                or torch.is_tensor(fallback_segmentation_instance3d_txyz_for_attn)
            )
            and isinstance(query_match_inputs, dict)
        ):
            query_attn_bbox_targets = self.build_query_attn_bbox_targets(
                segmentation_instance3d_txyz=gt_segmentation_instance3d_txyz_for_attn,
                future_egomotion=future_egomotion,
                gt_inst_ids_n=None,
                query_match_inputs=query_match_inputs,
                fallback_segmentation_instance3d_txyz=fallback_segmentation_instance3d_txyz_for_attn,
            )

        # query_inst transformer output is currently [T, 1, Q, D]; normalize to [T, Q, D].
        query_tokens_tqd = self._select_query_tokens_for_query_head(query_inst)
        query_head_outputs = self.query_head(
            query_tokens_tqd,
            return_query_feats=True,
            detach_query_for_center=self.query_center_loss_detach_query_feat,
            compute_direct_center=False,
        )
        query_cls_logits_qc = query_head_outputs["query_cls_logits_qc"]
        query_cls_scores_qc = query_head_outputs["query_cls_scores_qc"]
        query_cls_scores_tqc = query_head_outputs["query_cls_scores_tqc"]
        query_depth_logits_tqd = query_head_outputs["query_depth_logits_tqd"]
        query_depth_probs_tqd = query_head_outputs["query_depth_probs_tqd"]
        query_future_feat_tqd = query_head_outputs["query_feat_tqd"]

        self._last_query_attn_soft_lift_pack = None
        if (
            torch.is_tensor(query_attn_weights)
            and torch.is_tensor(query_depth_probs_tqd)
            and isinstance(query_match_inputs, dict)
            and torch.is_tensor(future_egomotion)
        ):
            query_attn_soft_lift_pack = self._build_query_attn_soft_lift_pack(
                query_attn_weights_tqnhw=query_attn_weights,
                query_depth_probs_tqd=query_depth_probs_tqd,
                query_match_inputs=query_match_inputs,
                future_egomotion=future_egomotion,
            )
            if isinstance(query_attn_soft_lift_pack, dict):
                self._last_query_attn_soft_lift_pack = query_attn_soft_lift_pack
        lifted_centers_world = self._last_query_attn_soft_lift_pack.get("lifted_center_world_tq3", None)
        lifted_valid_tq = self._last_query_attn_soft_lift_pack.get("lifted_valid_tq", None)
        query_head_outputs = self.query_head.apply_lifted_centers_to_outputs(
            query_head_outputs,
            lifted_centers_world,
            detach_query_for_center=self.query_center_loss_detach_query_feat,
        )
        centers_world = query_head_outputs["centers_world_tq3"]
        center_logits = query_head_outputs["center_logits_tq3"]
        gaussian_sigmas_world = query_head_outputs["query_sigma_world_tq3"]
        mixture_centers_world_tqg3 = query_head_outputs["mixture_centers_world_tqg3"]
        mixture_sigmas_world_tqg3 = query_head_outputs["mixture_sigmas_world_tqg3"]
        mixture_yaw_tqg = query_head_outputs["mixture_yaw_tqg"]
        mixture_weights_tqg = query_head_outputs["mixture_weights_tqg"]
        query_present_local_idx = int(query_head_outputs.get("query_present_local_idx", 0))
        if int(centers_world.shape[0]) > 0:
            query_present_local_idx = max(
                0, min(int(centers_world.shape[0]) - 1, int(query_present_local_idx))
            )
        else:
            query_present_local_idx = 0
        traj_offsets_fq2 = query_head_outputs.get("traj_offsets_fq2", None)

        present_centers_world_tq3 = centers_world.narrow(0, query_present_local_idx, 1).contiguous()
        present_sigmas_world_tq3 = (
            gaussian_sigmas_world.narrow(0, query_present_local_idx, 1).contiguous()
            if torch.is_tensor(gaussian_sigmas_world)
            and gaussian_sigmas_world.dim() == 3
            and int(gaussian_sigmas_world.shape[0]) > query_present_local_idx
            else None
        )
        present_mix_centers_world_tqg3 = (
            mixture_centers_world_tqg3.narrow(0, query_present_local_idx, 1).contiguous()
            if torch.is_tensor(mixture_centers_world_tqg3)
            and mixture_centers_world_tqg3.dim() == 4
            and int(mixture_centers_world_tqg3.shape[0]) > query_present_local_idx
            else None
        )
        present_mix_sigmas_world_tqg3 = (
            mixture_sigmas_world_tqg3.narrow(0, query_present_local_idx, 1).contiguous()
            if torch.is_tensor(mixture_sigmas_world_tqg3)
            and mixture_sigmas_world_tqg3.dim() == 4
            and int(mixture_sigmas_world_tqg3.shape[0]) > query_present_local_idx
            else None
        )
        present_mix_yaw_tqg = (
            mixture_yaw_tqg.narrow(0, query_present_local_idx, 1).contiguous()
            if torch.is_tensor(mixture_yaw_tqg)
            and mixture_yaw_tqg.dim() == 3
            and int(mixture_yaw_tqg.shape[0]) > query_present_local_idx
            else None
        )
        present_mix_weights_tqg = (
            mixture_weights_tqg.narrow(0, query_present_local_idx, 1).contiguous()
            if torch.is_tensor(mixture_weights_tqg)
            and mixture_weights_tqg.dim() == 3
            and int(mixture_weights_tqg.shape[0]) > query_present_local_idx
            else None
        )
        traj_geom_pack = self._build_query_trajectory_geometry_from_present(
            present_centers_world_tq3=present_centers_world_tq3,
            present_sigmas_world_tq3=present_sigmas_world_tq3,
            present_mix_centers_world_tqg3=present_mix_centers_world_tqg3,
            present_mix_sigmas_world_tqg3=present_mix_sigmas_world_tqg3,
            present_mix_yaw_tqg=present_mix_yaw_tqg,
            present_mix_weights_tqg=present_mix_weights_tqg,
            traj_offsets_fq2=traj_offsets_fq2,
            future_egomotion=future_egomotion,
        )
        if not isinstance(traj_geom_pack, dict):
            traj_geom_pack = dict()
        centers_world_traj_tq3 = traj_geom_pack.get("centers_world_tq3", None)
        if not torch.is_tensor(centers_world_traj_tq3):
            centers_world_traj_tq3 = centers_world
        traj_frame_indices_t = traj_geom_pack.get("traj_frame_indices_t", None)
        if torch.is_tensor(traj_frame_indices_t):
            traj_frame_indices_t = traj_frame_indices_t.to(
                device=centers_world_traj_tq3.device,
                dtype=torch.long,
            ).contiguous()
        gaussian_sigmas_world_traj_tq3 = traj_geom_pack.get("sigmas_world_tq3", None)
        if not torch.is_tensor(gaussian_sigmas_world_traj_tq3):
            gaussian_sigmas_world_traj_tq3 = gaussian_sigmas_world
        mixture_centers_world_traj_tqg3 = traj_geom_pack.get("mix_centers_world_tqg3", None)
        if not torch.is_tensor(mixture_centers_world_traj_tqg3):
            mixture_centers_world_traj_tqg3 = mixture_centers_world_tqg3
        mixture_sigmas_world_traj_tqg3 = traj_geom_pack.get("mix_sigmas_world_tqg3", None)
        if not torch.is_tensor(mixture_sigmas_world_traj_tqg3):
            mixture_sigmas_world_traj_tqg3 = mixture_sigmas_world_tqg3
        mixture_yaw_traj_tqg = traj_geom_pack.get("mix_yaw_tqg", None)
        if not torch.is_tensor(mixture_yaw_traj_tqg):
            mixture_yaw_traj_tqg = mixture_yaw_tqg
        mixture_weights_traj_tqg = traj_geom_pack.get("mix_weights_tqg", None)
        if not torch.is_tensor(mixture_weights_traj_tqg):
            mixture_weights_traj_tqg = mixture_weights_tqg
        pred_occ = None
        if voxelize:
            if self.center_only_mode:
                pred_occ = None
            else:
                # centers_world / gaussian_sigmas_world now span past frames
                # ([T_past, Q, *]) after the QueryHead projection was removed.
                # Voxelize only the present slice so inference-time occupancy
                # rendering keeps its original semantics (current frame only).
                present_sigmas_for_vox = present_sigmas_world_tq3
                if (
                    (not torch.is_tensor(present_sigmas_for_vox))
                    or present_sigmas_for_vox.dim() != 3
                ):
                    present_sigmas_for_vox = gaussian_sigmas_world.narrow(
                        0, query_present_local_idx, 1
                    ).contiguous()
                pred_occ = self.voxelizer(
                    present_centers_world_tq3,
                    sigmas_world=present_sigmas_for_vox,
                    weights=None,
                )

        bev_feats_enc = None
        bev_feats_src = None
        if return_bev_occ_feats and (voxel_feats is not None):
            voxel_feats = self.unpack_dbatch_and_dtime(
                voxel_feats, self.batch_size, self.time_receptive_field
            )
            voxel_feats = self.cumulative_warp_occ(voxel_feats.clone(), future_egomotion)
            seq_src = int(voxel_feats.shape[1])

            future_egomotion_vec = self.mat2pose_vec(future_egomotion)
            batch_size, sequence_length, nbr_pose_channels = future_egomotion_vec.shape
            dx, dy, dz = voxel_feats.shape[-3:]

            # Source BEV features: warped-only (before egomotion-channel concat).
            src_max_feats = self.voxel2bev_maxpooling(voxel_feats)
            src_mean_feats = self.voxel2bev(voxel_feats)
            src_bev_feats = src_max_feats * self.max_weight + src_mean_feats * self.mean_weight
            bev_feats_src = [src_bev_feats.reshape(batch_size, seq_src * int(src_bev_feats.shape[2]), *src_bev_feats.shape[3:])]

            future_egomotions_spatial = future_egomotion_vec.view(
                batch_size, sequence_length, nbr_pose_channels, 1, 1, 1
            ).expand(batch_size, sequence_length, nbr_pose_channels, dx, dy, dz)

            # at time 0, no egomotion so feed zero vector
            future_egomotions_spatial = torch.cat(
                [
                    torch.zeros_like(future_egomotions_spatial[:, :1]),
                    future_egomotions_spatial[:, :(self.time_receptive_field - 1)],
                ],
                dim=1,
            )
            voxel_feats = torch.cat([voxel_feats, future_egomotions_spatial], dim=-4)

            max_feats = self.voxel2bev_maxpooling(voxel_feats)
            mean_feats = self.voxel2bev(voxel_feats)
            bev_feats = max_feats * self.max_weight + mean_feats * self.mean_weight
            bev_feats_enc = self.occ_encoder(bev_feats)

            if type(bev_feats_enc) is not tuple:
                bev_feats_enc = [bev_feats_enc]

        return (
            pred_occ,
            center_logits,
            gaussian_sigmas_world,
            img_feats,
            centers_world,
            centers_world_traj_tq3,
            traj_frame_indices_t,
            query_cls_logits_qc,
            query_cls_scores_qc,
            query_cls_scores_tqc,
            query_depth_logits_tqd,
            query_depth_probs_tqd,
            query_img_feat_pooled,
            query_future_feat_tqd,
            bev_feats_src,
            bev_feats_enc,
            query_attn_weights,
            query_attn_bbox_targets,
            instance_img_debug_bundle,
            query_match_inputs,
            traj_offsets_fq2,
            query_present_local_idx,
            mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3,
            mixture_yaw_tqg,
            mixture_weights_tqg,
            gaussian_sigmas_world_traj_tq3,
            mixture_centers_world_traj_tqg3,
            mixture_sigmas_world_traj_tqg3,
            mixture_yaw_traj_tqg,
            mixture_weights_traj_tqg,
        )


    def extract_feat(self, img_inputs_seq, img_metas, future_egomotion):
        '''
        Extract voxel features from input sequential images
        '''
        voxel_feats = None
        depth, img_feats = None, None

        if img_inputs_seq is not None:
            _, _, depth, img_feats, voxel_feats, _, _, _ = self.extract_img_feat(
                img_inputs_seq,
                img_metas,
                run_query_transformer=False,
            )
        if depth is not None:
            depth = depth.view(-1, self.n_cam, *depth.shape[-3:])
        
        voxel_feats = self.unpack_dbatch_and_dtime(voxel_feats, self.batch_size, self.time_receptive_field)

        voxel_feats = self.cumulative_warp_occ(voxel_feats.clone(), future_egomotion)

        # egomotion-aware
        future_egomotion_vec = self.mat2pose_vec(future_egomotion)
        batch_size, sequence_length, nbr_pose_channels = future_egomotion_vec.shape
        dx, dy, dz = voxel_feats.shape[-3:]

        future_egomotions_spatial = future_egomotion_vec.view(batch_size, sequence_length, nbr_pose_channels, 1, 1, 1).expand(batch_size, sequence_length, nbr_pose_channels, dx, dy, dz)
        
        # at time 0, no egomotion so feed zero vector
        future_egomotions_spatial = torch.cat([torch.zeros_like(future_egomotions_spatial[:, :1]),
                                            future_egomotions_spatial[:, :(self.time_receptive_field-1)]], dim=1)
        voxel_feats = torch.cat([voxel_feats, future_egomotions_spatial], dim=-4)
        
        max_feats = self.voxel2bev_maxpooling(voxel_feats) # max pooling
        mean_feats = self.voxel2bev(voxel_feats) # average pooling
        bev_feats = max_feats * self.max_weight + mean_feats * self.mean_weight

        bev_feats_enc = self.occ_encoder(bev_feats)
        if type(bev_feats_enc) is not tuple:
            bev_feats_enc = [bev_feats_enc]

        return bev_feats_enc, img_feats, depth
    
    def voxel2bev(self, voxel_feats):
        bev_feats = torch.mean(voxel_feats,-1)
        return bev_feats

    def voxel2bev_maxpooling(self, voxel_feats):
        bev_feats = torch.max(voxel_feats,-1).values
        return bev_feats
    
    @force_fp32(apply_to=('voxel_feats'))
    def forward_pts_train(
            self,
            voxel_feats,
            segmentation_bev=None,
            points_occ=None,
            img_metas=None,
            transform=None,
            img_feats=None,
            return_outs=False,
        ):
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,)
        
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            target_voxels=segmentation_bev,
            target_points=points_occ,
            img_metas=img_metas,)

        if return_outs:
            return losses, outs
        return losses

    def _forward_train_view_transform_pretrain(
            self,
            img_inputs_seq=None,
            future_egomotion=None,
            segmentation_bev=None,
            img_metas=None,
            points_occ=None,
        ):
        """Pretrain route: image -> view transform -> BEV -> occ_head only."""

        bev_feats_enc, img_feats, _ = self.extract_feat(
            img_inputs_seq=img_inputs_seq,
            img_metas=img_metas,
            future_egomotion=future_egomotion,
        )

        if segmentation_bev.dim() >= 4:
            segmentation_bev = segmentation_bev[:, -self.n_future_frames_plus:, ...].contiguous()
        elif segmentation_bev.dim() == 3:
            segmentation_bev = segmentation_bev[-self.n_future_frames_plus:, ...].unsqueeze(0).contiguous()

        transform = img_inputs_seq[1:8] if img_inputs_seq is not None else None

        voxel_feats_seq = []
        for voxel_feats_stage in bev_feats_enc:
            bs, sfeatures = voxel_feats_stage.shape[:2]
            voxel_feats_stage_ = voxel_feats_stage.view(
                bs * self.n_future_frames_plus,
                sfeatures // self.n_future_frames_plus,
                *voxel_feats_stage.shape[2:],
            )
            voxel_feats_seq.append(voxel_feats_stage_)

        losses, occ_outs = self.forward_pts_train(
            voxel_feats=voxel_feats_seq,
            segmentation_bev=segmentation_bev,
            points_occ=points_occ,
            img_metas=img_metas,
            transform=transform,
            img_feats=img_feats,
            return_outs=True,
        )
        self._maybe_save_lss_pretrain_occ_vis(
            output_voxels=occ_outs.get("output_voxels", None),
            target_voxels=segmentation_bev,
        )
        return losses

    def _aggregate_training_losses(
        self,
        *,
        # BEV occupancy loss inputs
        bev_feats_enc,
        segmentation_bev,
        points_occ,
        img_metas,
        img_feats,
        transform,
        # Pre-computed query loss dicts
        query_cls_loss,
        query_depth_loss,
        query_attn_bbox_loss,
        matched_gmo_loss,
        center_match_loss,
        query_traj_loss,
        query_attn_cam_score_pack,
        # Query predictions
        centers_world,
        centers_world_dt_gt2p_tq3,
        gaussian_sigmas_world_loss,
        mixture_sigmas_world_loss,
        mixture_weights_loss,
        # DT loss inputs
        occ_dt,
        # GT2P loss inputs
        gt_inst_center_world_tn3,
        gt_inst_center_valid_tn,
    ):
        """Aggregate all training loss components into a single losses dict."""
        losses = dict()
        z = centers_world.sum() * 0.0

        # ---- BEV occupancy loss ----
        if self.use_lss_bev_occ_loss and (bev_feats_enc is not None) and (segmentation_bev is not None):
            voxel_feats_seq = []
            for voxel_feats_stage in bev_feats_enc:
                bs, sfeatures = voxel_feats_stage.shape[:2]
                voxel_feats_stage_ = voxel_feats_stage.view(
                    bs * self.n_future_frames_plus,
                    sfeatures // self.n_future_frames_plus,
                    *voxel_feats_stage.shape[2:],
                )
                voxel_feats_seq.append(voxel_feats_stage_)
            losses_occupancy = self.forward_pts_train(
                voxel_feats_seq, segmentation_bev, points_occ, img_metas,
                img_feats=img_feats, transform=transform,
            )
            losses.update(losses_occupancy)

        # ---- Query classification / feature losses ----
        if isinstance(query_cls_loss, dict):
            losses.update(query_cls_loss)
        else:
            losses["loss_query_cls"] = z
        if isinstance(query_depth_loss, dict):
            losses.update(query_depth_loss)
        else:
            losses["loss_query_depth"] = z

        if isinstance(query_attn_bbox_loss, dict):
            losses.update(
                {
                    k: v
                    for k, v in query_attn_bbox_loss.items()
                    if not k.startswith("dbg_")
                    and (
                        (torch.is_tensor(v) and (torch.is_floating_point(v) or torch.is_complex(v)))
                        or (
                            isinstance(v, list)
                            and all(
                                torch.is_tensor(v_i) and (torch.is_floating_point(v_i) or torch.is_complex(v_i))
                                for v_i in v
                            )
                        )
                    )
                }
            )
        else:
            losses["loss_query_attn_bbox"] = z

        if isinstance(matched_gmo_loss, dict):
            losses.update(matched_gmo_loss)
        else:
            losses["loss_gmo_focal"] = z
            losses["loss_gmo_dice"] = z

        losses.update(
            self.query_head.compute_multi_gaussian_sigma_reg_loss(
                mixture_sigmas_world_tqg3=mixture_sigmas_world_loss,
                mixture_weights_tqg=mixture_weights_loss,
            )
        )

        # ---- Query decorrelation loss ----
        query_decor_loss = getattr(self.transformer, "last_query_decor_loss", None)
        losses["loss_query_decor"] = query_decor_loss if torch.is_tensor(query_decor_loss) else z

        # ---- Center match loss ----
        losses["loss_query_center_match"] = center_match_loss if center_match_loss is not None else z

        # ---- Trajectory loss ----
        if isinstance(query_traj_loss, dict):
            losses.update(query_traj_loss)
        elif query_traj_loss is not None:
            losses["loss_query_traj"] = query_traj_loss
        else:
            losses["loss_query_traj"] = z

        # ---- Hungarian matching cost weights ----
        losses["dbg_query_sim_match_cost_weight"] = centers_world.new_tensor(
            float(self.query_sim_match_cost_weight)
        )
        losses["dbg_query_soft_assign_cost_weight"] = centers_world.new_tensor(
            float(self.query_soft_assign_cost_weight)
        )
        losses["dbg_query_cls_match_cost_weight"] = centers_world.new_tensor(
            float(self.query_cls_match_cost_weight)
        )
        losses["dbg_query_bev_dice_match_cost_weight"] = centers_world.new_tensor(
            float(self.query_bev_dice_match_cost_weight)
        )
        if isinstance(query_attn_cam_score_pack, dict):
            losses.update(
                {
                    k: v
                    for k, v in query_attn_cam_score_pack.items()
                    if not k.startswith("dbg_")
                    and (
                        (torch.is_tensor(v) and (torch.is_floating_point(v) or torch.is_complex(v)))
                        or (
                            isinstance(v, list)
                            and all(
                                torch.is_tensor(v_i) and (torch.is_floating_point(v_i) or torch.is_complex(v_i))
                                for v_i in v
                            )
                        )
                    )
                }
            )

        # ---- DT loss ----
        if self.use_query_dt_loss:
            if self.center_only_mode:
                dt_loss = self.query_head.compute_query_point_dt_loss(
                    points_world=centers_world_dt_gt2p_tq3, occ_dt=occ_dt,
                    loss_weight=0.1, mode='bilinear',
                )
            else:
                dt_loss = self.query_head.compute_query_point_dt_loss(
                    points_world=None,
                    gaussian_centers_world=centers_world_dt_gt2p_tq3,
                    gaussian_sigmas_world=gaussian_sigmas_world_loss,
                    occ_dt=occ_dt, loss_weight=0.1, mode='bilinear',
                )
            losses.update(dt_loss)

        # ---- GT2P loss ----
        gt2p_cooldown_scale = float(self._compute_gt2p_cooldown_scale())
        if self.use_query_gt2p_instance_labeled_loss:
            gt2p_inst_weight_eff = float(self.query_gt2p_instance_labeled_loss_weight) * gt2p_cooldown_scale
            gt2p_instance_labeled_loss = self.query_head.compute_query_gt2p_instance_labeled_loss(
                gaussian_centers_world=centers_world_dt_gt2p_tq3,
                gaussian_sigmas_world=gaussian_sigmas_world_loss,
                gt_inst_center_world_tn3=gt_inst_center_world_tn3,
                gt_inst_center_valid_tn=gt_inst_center_valid_tn,
                loss_weight=gt2p_inst_weight_eff,
                balance_weight=self.query_gt2p_instance_labeled_balance_weight,
                center_sigma_xyz=self.query_gt2p_instance_labeled_assign_sigma_xyz,
                sigma_policy=self.query_gt2p_instance_labeled_sigma_policy,
                tau=self.query_gt2p_instance_labeled_tau,
            )
            losses.update(gt2p_instance_labeled_loss)

        # ---- Normalize losses / strip stray dbg entries / namespace ----
        if self.loss_norm:
            for loss_key in losses.keys():
                if loss_key.startswith('loss'):
                    losses[loss_key] = losses[loss_key] / (losses[loss_key].detach() + 1e-9)
        _keep_dbg = {
            "dbg_query_sim_match_cost_weight",
            "dbg_query_soft_assign_cost_weight",
            "dbg_query_cls_match_cost_weight",
            "dbg_query_bev_dice_match_cost_weight",
        }
        for k in [k for k in losses if k.startswith("dbg_") and k not in _keep_dbg]:
            del losses[k]
        self._namespace_dbg_logs(losses)
        return losses

    def _build_query_visualization_bundle(
        self,
        centers_world_tq3: torch.Tensor,
        query_cls_scores_qc: torch.Tensor,
        inst_match_result: dict,
        gt_segmentation_instance3d_txyz: torch.Tensor,
        gaussian_sigmas_world_tq3: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_yaw_tqg: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        query_attn_cam_score_pack: dict = None,
    ):
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        if (not torch.is_tensor(query_cls_scores_qc)) or query_cls_scores_qc.dim() != 2:
            return None
        q_count, c_count = [int(v) for v in query_cls_scores_qc.shape]
        if q_count <= 0 or c_count <= 1:
            return None
        if int(centers_world_tq3.shape[1]) != q_count:
            return None
        if (not torch.is_tensor(gt_segmentation_instance3d_txyz)) or gt_segmentation_instance3d_txyz.dim() != 4:
            return None
        has_surrogate_sigma = (
            torch.is_tensor(gaussian_sigmas_world_tq3)
            and tuple(gaussian_sigmas_world_tq3.shape) == tuple(centers_world_tq3.shape)
        )
        has_mixture = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_yaw_tqg)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_yaw_tqg.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
            and int(mixture_centers_world_tqg3.shape[1]) == q_count
        )
        if (not has_surrogate_sigma) and (not has_mixture):
            return None

        score_frame_count = min(
            int(getattr(self, "time_receptive_field", 1)),
            int(centers_world_tq3.shape[0]),
            int(gt_segmentation_instance3d_txyz.shape[0]),
        )
        if score_frame_count <= 0:
            return None
        score_centers_tq3 = centers_world_tq3[:score_frame_count].contiguous()
        score_gt_occ_txyz = gt_segmentation_instance3d_txyz[:score_frame_count].contiguous()
        score_sigmas_tq3 = (
            gaussian_sigmas_world_tq3[:score_frame_count].contiguous()
            if has_surrogate_sigma
            else centers_world_tq3.new_zeros(score_centers_tq3.shape)
        )
        score_mix_centers_tqg3 = (
            mixture_centers_world_tqg3[:score_frame_count].contiguous()
            if has_mixture
            else None
        )
        score_mix_sigmas_tqg3 = (
            mixture_sigmas_world_tqg3[:score_frame_count].contiguous()
            if has_mixture
            else None
        )
        score_mix_yaw_tqg = (
            mixture_yaw_tqg[:score_frame_count].contiguous()
            if has_mixture
            else None
        )
        score_mix_weights_tqg = (
            mixture_weights_tqg[:score_frame_count].contiguous()
            if has_mixture
            else None
        )

        cls_scores_qc = query_cls_scores_qc.to(device=centers_world_tq3.device, dtype=torch.float32)
        pred_cls_q = torch.argmax(cls_scores_qc, dim=-1).to(torch.long)
        fg_scores_qc = cls_scores_qc.clone()
        fg_scores_qc[:, int(self.query_bg_class)] = 0.0
        cls_prob_q = fg_scores_qc.max(dim=-1).values.clamp(0.0, 1.0)

        if has_mixture:
            iou_q = self._compute_matched_query_sequence_iou_scores(
                centers_world_tq3=score_centers_tq3,
                sigmas_world_tq3=score_sigmas_tq3,
                mixture_centers_world_tqg3=score_mix_centers_tqg3,
                mixture_sigmas_world_tqg3=score_mix_sigmas_tqg3,
                mixture_yaw_tqg=score_mix_yaw_tqg,
                mixture_weights_tqg=score_mix_weights_tqg,
                gt_instance_occ3d_txyz_pred=score_gt_occ_txyz,
                inst_match_result=inst_match_result,
                objectness_scores_tq=None,
                pair_chunk_size=int(self.query_multi_gaussian_pair_chunk),
            )
        else:
            iou_q = self._compute_matched_query_sequence_iou_scores(
                centers_world_tq3=score_centers_tq3,
                sigmas_world_tq3=score_sigmas_tq3,
                gt_instance_occ3d_txyz_pred=score_gt_occ_txyz,
                inst_match_result=inst_match_result,
                objectness_scores_tq=None,
            )
        if not torch.is_tensor(iou_q) or int(iou_q.numel()) != q_count:
            iou_q = centers_world_tq3.new_zeros((q_count,), dtype=torch.float32)
        iou_q = iou_q.to(dtype=torch.float32).clamp(0.0, 1.0)

        w_iou = float(self.debug_query_score_iou_weight)
        w_cls = float(self.debug_query_score_cls_weight)
        w_cam = float(self.debug_query_score_cam_attn_weight)
        cam_attn_score_q = centers_world_tq3.new_zeros((q_count,), dtype=torch.float32)
        cam_attn_score_valid_q = torch.zeros((q_count,), device=centers_world_tq3.device, dtype=torch.bool)
        if isinstance(query_attn_cam_score_pack, dict):
            cam_q = query_attn_cam_score_pack.get("cam_attn_score_q", None)
            cam_valid_q = query_attn_cam_score_pack.get("cam_attn_score_valid_q", None)
            if torch.is_tensor(cam_q) and int(cam_q.numel()) == q_count:
                cam_attn_score_q = cam_q.to(device=centers_world_tq3.device, dtype=torch.float32).clamp(0.0, 1.0)
                if torch.is_tensor(cam_valid_q) and int(cam_valid_q.numel()) == q_count:
                    cam_attn_score_valid_q = cam_valid_q.to(device=centers_world_tq3.device, dtype=torch.bool)
                else:
                    cam_attn_score_valid_q = torch.isfinite(cam_attn_score_q)
            else:
                cam_attn_score_valid_q = torch.zeros((q_count,), device=centers_world_tq3.device, dtype=torch.bool)
        cam_valid_f_q = cam_attn_score_valid_q.to(torch.float32)
        score_num_q = (w_iou * iou_q) + (w_cls * cls_prob_q) + (w_cam * cam_attn_score_q * cam_valid_f_q)
        score_den_q = (w_iou + w_cls) + (w_cam * cam_valid_f_q)
        score_q = score_num_q / score_den_q.clamp_min(1e-6)
        score_q = score_q.clamp(0.0, 1.0)

        candidate_idx = torch.arange(q_count, device=score_q.device, dtype=torch.long)

        fg_mask_q = pred_cls_q != int(self.query_bg_class)
        selected_candidate_idx = torch.nonzero(fg_mask_q, as_tuple=False).squeeze(1)
        if int(selected_candidate_idx.numel()) > 0:
            candidate_scores = score_q.index_select(0, selected_candidate_idx)
            keep_thr = candidate_scores >= float(self.debug_query_score_threshold)
            selected_candidate_idx = selected_candidate_idx[keep_thr]

        if int(selected_candidate_idx.numel()) > 0 and int(self.debug_query_score_topk) > 0:
            cand_scores = score_q.index_select(0, selected_candidate_idx)
            topk = min(int(self.debug_query_score_topk), int(selected_candidate_idx.numel()))
            order = torch.topk(cand_scores, k=topk, largest=True, sorted=True).indices
            selected_idx = selected_candidate_idx.index_select(0, order)
        else:
            selected_idx = selected_candidate_idx.new_empty((0,), dtype=torch.long)

        candidate_points = (
            centers_world_tq3.index_select(1, candidate_idx)
            if int(candidate_idx.numel()) > 0 else centers_world_tq3[:, :0, :]
        )
        if has_surrogate_sigma:
            candidate_sigmas = (
                gaussian_sigmas_world_tq3.index_select(1, candidate_idx)
                if int(candidate_idx.numel()) > 0 else gaussian_sigmas_world_tq3[:, :0, :]
            )
        else:
            candidate_sigmas = centers_world_tq3[:, :0, :]
        candidate_mix_centers = (
            mixture_centers_world_tqg3.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        candidate_mix_sigmas = (
            mixture_sigmas_world_tqg3.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        candidate_mix_yaw = (
            mixture_yaw_tqg.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        candidate_mix_weights = (
            mixture_weights_tqg.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        selected_points = (
            centers_world_tq3.index_select(1, selected_idx)
            if int(selected_idx.numel()) > 0 else centers_world_tq3[:, :0, :]
        )
        if has_surrogate_sigma:
            selected_sigmas = (
                gaussian_sigmas_world_tq3.index_select(1, selected_idx)
                if int(selected_idx.numel()) > 0 else gaussian_sigmas_world_tq3[:, :0, :]
            )
        else:
            selected_sigmas = centers_world_tq3[:, :0, :]
        selected_mix_centers = (
            mixture_centers_world_tqg3.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        selected_mix_sigmas = (
            mixture_sigmas_world_tqg3.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        selected_mix_yaw = (
            mixture_yaw_tqg.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        selected_mix_weights = (
            mixture_weights_tqg.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        matched_idx = selected_idx.new_empty((0,), dtype=torch.long)
        matched_inst_idx = selected_idx.new_empty((0,), dtype=torch.long)
        matched_gt_ids = selected_idx.new_empty((0,), dtype=torch.long)
        if isinstance(inst_match_result, dict):
            match_q_idx = inst_match_result.get("matched_query_idx", None)
            if torch.is_tensor(match_q_idx) and match_q_idx.numel() > 0:
                match_q_idx = match_q_idx.to(device=score_q.device, dtype=torch.long).reshape(-1)
                valid_mask = (match_q_idx >= 0) & (match_q_idx < q_count)
                if bool(valid_mask.any().item()):
                    matched_idx = torch.unique(match_q_idx[valid_mask], sorted=True)
            if int(matched_idx.numel()) > 0:
                match_inst_idx = inst_match_result.get("matched_inst_idx", None)
                gt_ids_n = inst_match_result.get("gt_ids_n", None)
                if (
                    torch.is_tensor(match_q_idx)
                    and torch.is_tensor(match_inst_idx)
                    and torch.is_tensor(gt_ids_n)
                    and int(match_q_idx.numel()) == int(match_inst_idx.numel())
                    and int(gt_ids_n.numel()) > 0
                ):
                    mq = match_q_idx.to(device=score_q.device, dtype=torch.long).reshape(-1)
                    mi = match_inst_idx.to(device=score_q.device, dtype=torch.long).reshape(-1)
                    gt_ids_pool = gt_ids_n.to(device=score_q.device, dtype=torch.long).reshape(-1)
                    keep = (mq >= 0) & (mq < q_count) & (mi >= 0) & (mi < int(gt_ids_pool.numel()))
                    if bool(keep.any().item()):
                        mq = mq[keep]
                        mi = mi[keep]
                        q_to_src = {}
                        for src_idx, qid in enumerate(mq.tolist()):
                            if int(qid) not in q_to_src:
                                q_to_src[int(qid)] = int(src_idx)
                        out_inst = []
                        out_gt_ids = []
                        for qid in matched_idx.tolist():
                            src_idx = q_to_src.get(int(qid), None)
                            if src_idx is None:
                                out_inst.append(-1)
                                out_gt_ids.append(-1)
                            else:
                                inst_idx = int(mi[src_idx].item())
                                out_inst.append(inst_idx)
                                out_gt_ids.append(int(gt_ids_pool[inst_idx].item()))
                        matched_inst_idx = torch.as_tensor(
                            out_inst, device=score_q.device, dtype=torch.long
                        )
                        matched_gt_ids = torch.as_tensor(
                            out_gt_ids, device=score_q.device, dtype=torch.long
                        )
        matched_points = (
            centers_world_tq3.index_select(1, matched_idx)
            if int(matched_idx.numel()) > 0 else centers_world_tq3[:, :0, :]
        )
        if has_surrogate_sigma:
            matched_sigmas = (
                gaussian_sigmas_world_tq3.index_select(1, matched_idx)
                if int(matched_idx.numel()) > 0 else gaussian_sigmas_world_tq3[:, :0, :]
            )
        else:
            matched_sigmas = centers_world_tq3[:, :0, :]
        matched_mix_centers = (
            mixture_centers_world_tqg3.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )
        matched_mix_sigmas = (
            mixture_sigmas_world_tqg3.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )
        matched_mix_yaw = (
            mixture_yaw_tqg.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )
        matched_mix_weights = (
            mixture_weights_tqg.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )

        return {
            "candidate_points_tq3": candidate_points.detach(),
            "candidate_sigmas_tq3": candidate_sigmas.detach(),
            "candidate_pred_cls_q": pred_cls_q.index_select(0, candidate_idx).detach(),
            "candidate_cls_prob_q": cls_prob_q.index_select(0, candidate_idx).detach(),
            "candidate_iou_q": iou_q.index_select(0, candidate_idx).detach(),
            "candidate_score_q": score_q.index_select(0, candidate_idx).detach(),
            "selected_points_tq3": selected_points.detach(),
            "selected_sigmas_tq3": selected_sigmas.detach(),
            "selected_pred_cls_q": pred_cls_q.index_select(0, selected_idx).detach(),
            "selected_score_q": score_q.index_select(0, selected_idx).detach(),
            "selected_query_idx_q": selected_idx.detach(),
            "matched_points_tq3": matched_points.detach(),
            "matched_sigmas_tq3": matched_sigmas.detach(),
            "matched_pred_cls_q": pred_cls_q.index_select(0, matched_idx).detach(),
            "matched_score_q": score_q.index_select(0, matched_idx).detach(),
            "matched_query_idx_q": matched_idx.detach(),
            "matched_inst_idx_q": matched_inst_idx.detach(),
            "matched_gt_ids_q": matched_gt_ids.detach(),
            "score_q": score_q.detach(),
            "iou_q": iou_q.detach(),
            "cls_prob_q": cls_prob_q.detach(),
            "cam_attn_score_q": cam_attn_score_q.detach(),
            "cam_attn_score_valid_q": cam_attn_score_valid_q.detach(),
            "score_frame_count": int(score_frame_count),
            "pred_cls_q": pred_cls_q.detach(),
            "gaussian_sigmas_tq3": gaussian_sigmas_world_tq3.detach() if has_surrogate_sigma else None,
            "candidate_mixture_centers_tqg3": candidate_mix_centers.detach() if candidate_mix_centers is not None else None,
            "candidate_mixture_sigmas_tqg3": candidate_mix_sigmas.detach() if candidate_mix_sigmas is not None else None,
            "candidate_mixture_yaw_tqg": candidate_mix_yaw.detach() if candidate_mix_yaw is not None else None,
            "candidate_mixture_weights_tqg": candidate_mix_weights.detach() if candidate_mix_weights is not None else None,
            "selected_mixture_centers_tqg3": selected_mix_centers.detach() if selected_mix_centers is not None else None,
            "selected_mixture_sigmas_tqg3": selected_mix_sigmas.detach() if selected_mix_sigmas is not None else None,
            "selected_mixture_yaw_tqg": selected_mix_yaw.detach() if selected_mix_yaw is not None else None,
            "selected_mixture_weights_tqg": selected_mix_weights.detach() if selected_mix_weights is not None else None,
            "matched_mixture_centers_tqg3": matched_mix_centers.detach() if matched_mix_centers is not None else None,
            "matched_mixture_sigmas_tqg3": matched_mix_sigmas.detach() if matched_mix_sigmas is not None else None,
            "matched_mixture_yaw_tqg": matched_mix_yaw.detach() if matched_mix_yaw is not None else None,
            "matched_mixture_weights_tqg": matched_mix_weights.detach() if matched_mix_weights is not None else None,
            "top_k": int(self.debug_query_score_topk),
            "score_thr": float(self.debug_query_score_threshold),
            "w_iou": float(w_iou),
            "w_cls": float(w_cls),
            "w_cam": float(w_cam),
        }

    def forward_train(self,
            img_inputs_seq=None,
            segmentation=None,
            future_egomotion=None,
            gt_occ=None,
            segmentation_bev=None,
            segmentation_instance3d=None,
            segmentation_cls_instance3d=None,
            gt_occ_inst=None,
            gt_instance_centers_world=None,
            gt_instance_centers_valid=None,
            gt_instance_ids=None,
            img_metas=None,
            points_occ=None,
            occ_dt=None,
            **kwargs,
        ):
        '''
        Train EfficientOCF using the provided occupancy and instance supervision signals.
        '''
        # manually stop forward
        if self.only_generate_dataset:
            return {"pseudo_loss": torch.tensor(0.0, device=segmentation_bev.device, requires_grad=True)}

        # ======= prepare GT occupancy which contains cls logits and instance ids ============
        gt_segmentation_cls_instance3d_tcxyz = self._prepare_segmentation_cls_instance3d(
            segmentation_cls_instance3d=segmentation_cls_instance3d
        )
        gt_segmentation_instance3d_txyz = self._prepare_segmentation_instance3d(
            segmentation_instance3d=segmentation_instance3d
        )
        gt_occ_inst_bundle = self._prepare_gt_occ_inst_primary_targets(
            gt_occ_inst=gt_occ_inst,
            fallback_segmentation_instance3d_txyz=gt_segmentation_instance3d_txyz,
        )
        gt_instance_occ3d_txyz_primary = (
            gt_occ_inst_bundle["dense_inst_txyz"]
            if (isinstance(gt_occ_inst_bundle, dict) and torch.is_tensor(gt_occ_inst_bundle.get("dense_inst_txyz", None)))
            else gt_segmentation_instance3d_txyz
        )
        gt_segmentation_cls_instance3d_for_match = (
            gt_occ_inst_bundle["seg_cls_inst_tcxzy"]
            if (isinstance(gt_occ_inst_bundle, dict) and torch.is_tensor(gt_occ_inst_bundle.get("seg_cls_inst_tcxzy", None)))
            else gt_segmentation_cls_instance3d_tcxyz
        )
        _gt_instance_occ3d_txyz_query_present, gt_query_local_idx, gt_query_source_layout = self._select_query_temporal_gt_slice(
            gt_instance_occ3d_txyz_primary
        )
        _gt_segmentation_instance3d_txyz_query_present, _, _ = self._select_query_temporal_gt_slice(
            gt_segmentation_instance3d_txyz
        )
        _gt_segmentation_cls_instance3d_for_match_query_present, _, _ = self._select_query_temporal_gt_slice(
            gt_segmentation_cls_instance3d_for_match
        )
        gt_instance_occ3d_txyz_query, _, _ = self._select_query_trajectory_gt_slice(
            gt_instance_occ3d_txyz_primary
        )
        gt_instance_occ3d_txyz_query_vis, _, _ = self._select_query_visualization_gt_slice(
            gt_instance_occ3d_txyz_primary
        )
        gt_segmentation_instance3d_txyz_query, _, _ = self._select_query_trajectory_gt_slice(
            gt_segmentation_instance3d_txyz
        )
        gt_segmentation_instance3d_txyz_query_vis, _, _ = self._select_query_visualization_gt_slice(
            gt_segmentation_instance3d_txyz
        )
        gt_segmentation_cls_instance3d_for_match_query, _, _ = self._select_query_trajectory_gt_slice(
            gt_segmentation_cls_instance3d_for_match
        )
        # Visualization-horizon GT spans [t-(T_past-1), ..., t+n_future_frames],
        # i.e. 7 frames in the canonical config. Pad past frames with zeros if
        # the dataloader does not provide them.
        gt_segmentation_cls_instance3d_for_match_query_vis, _, _ = (
            self._select_query_visualization_gt_slice(gt_segmentation_cls_instance3d_for_match)
        )
        self._last_query_gt_temporal_local_idx = int(gt_query_local_idx) if gt_query_local_idx is not None else -1
        self._last_query_gt_temporal_source_layout = str(gt_query_source_layout)

        # LSS view transform pretrain route
        if self.pretrain_view_transform_only:
            return self._forward_train_view_transform_pretrain(
                img_inputs_seq=img_inputs_seq,
                future_egomotion=future_egomotion,
                segmentation_bev=segmentation_bev,
                img_metas=img_metas,
                points_occ=points_occ,
            )

        cur_train_iter = self._get_query_train_iteration(advance_if_unsynced=True)
        need_query_cam_gaussian_vis = bool(
            bool(self.debug_query_cam_gaussian_vis_enabled)
            and (int(self.debug_query_cam_gaussian_vis_every) > 0)
            and ((int(cur_train_iter) % int(self.debug_query_cam_gaussian_vis_every)) == 0)
            and self._is_main_process()
        )
        need_instance_img_debug = bool(
            (int(self.debug_instance_img_vis_every) > 0)
            and ((int(cur_train_iter) % int(self.debug_instance_img_vis_every)) == 0)
            and self._is_main_process()
        )
        # Query-CAM Gaussian visualization renders the query future horizon on
        # the last visible frames, so it must use the same full-sequence slice
        # that instance_img_debug_vis uses instead of the default 3-frame history.
        need_instance_img_debug_fullseq = bool(
            need_instance_img_debug or need_query_cam_gaussian_vis
        )
        if need_instance_img_debug and isinstance(img_inputs_seq, (list, tuple)) and torch.is_tensor(img_inputs_seq[0]):
            seq_len_loaded = int(img_inputs_seq[0].shape[1])
            seq_expected = int(self.time_receptive_field + self.n_future_frames)
            if (seq_len_loaded != seq_expected) and (not getattr(self, "_dbg_printed_instance_img_seq_len_warning", False)):
                print(
                    f"[EfficientOCF][instance-img-debug] sequence_length mismatch: loaded={seq_len_loaded}, expected={seq_expected}",
                    flush=True,
                )
                self._dbg_printed_instance_img_seq_len_warning = True

        # Select using gt_occ(nuScenes-Occupancy) or segmentation(nuScenes) as GT
        query_gt_occ, query_gmo_ids = self._select_query_gt_for_losses(
            gt_occ=gt_occ,
            segmentation=segmentation,
        )
        query_gt_occ_aligned, _, _ = self._select_query_trajectory_gt_slice(query_gt_occ)

        gt_inst_center_world_tn3 = None
        gt_inst_center_valid_tn = None
        gt_inst_ids_n = None
        gt_inst_center_world_hist_tn3 = None
        gt_inst_center_valid_hist_tn = None
        gt_inst_ids_hist_n = None
        if isinstance(gt_occ_inst_bundle, dict):
            gt_inst_center_world_tn3, gt_inst_center_valid_tn, gt_inst_ids_n = self._prepare_gt_instance_centers_for_trajectory_matching(
                gt_instance_centers_world=gt_occ_inst_bundle.get("centers_world_tn3", None),
                gt_instance_centers_valid=gt_occ_inst_bundle.get("centers_valid_tn", None),
                gt_instance_ids=gt_occ_inst_bundle.get("instance_ids_n", None),
            )
        if gt_inst_center_world_tn3 is None:
            gt_inst_center_world_tn3, gt_inst_center_valid_tn, gt_inst_ids_n = self._prepare_gt_instance_centers_for_trajectory_matching(
                gt_instance_centers_world=gt_instance_centers_world,
                gt_instance_centers_valid=gt_instance_centers_valid,
                gt_instance_ids=gt_instance_ids,
            )
        # Per-instance depth target uses bbox-based centers from segmentation_instance3d path.
        gt_inst_center_world_hist_tn3, gt_inst_center_valid_hist_tn, gt_inst_ids_hist_n = self._prepare_gt_instance_centers_for_history(
            gt_instance_centers_world=gt_instance_centers_world,
            gt_instance_centers_valid=gt_instance_centers_valid,
            gt_instance_ids=gt_instance_ids,
        )
        # Align instance targets by ID intersection between fine-grained and bbox occupancy sources.
        inst_ids_intersection_n = self._build_intersection_instance_ids_from_dense_pair(
            primary_instance3d_txyz=gt_instance_occ3d_txyz_primary,
            secondary_instance3d_txyz=gt_segmentation_instance3d_txyz,
        )
        if torch.is_tensor(inst_ids_intersection_n):
            gt_inst_center_world_tn3, gt_inst_center_valid_tn, gt_inst_ids_n = self._filter_instance_targets_by_intersection_ids(
                centers_world_tn3=gt_inst_center_world_tn3,
                centers_valid_tn=gt_inst_center_valid_tn,
                instance_ids_n=gt_inst_ids_n,
                intersection_ids_n=inst_ids_intersection_n,
            )
            gt_inst_center_world_hist_tn3, gt_inst_center_valid_hist_tn, gt_inst_ids_hist_n = self._filter_instance_targets_by_intersection_ids(
                centers_world_tn3=gt_inst_center_world_hist_tn3,
                centers_valid_tn=gt_inst_center_valid_hist_tn,
                instance_ids_n=gt_inst_ids_hist_n,
                intersection_ids_n=inst_ids_intersection_n,
            )
        gt_inst_cls_n, gt_inst_cls_valid_n = self._prepare_gt_instance_classes_for_matching(
            segmentation_cls_instance3d=gt_segmentation_cls_instance3d_for_match,
            gt_instance_ids_n=gt_inst_ids_n,
        )
        if (gt_inst_cls_n is None) and torch.is_tensor(gt_segmentation_cls_instance3d_tcxyz):
            gt_inst_cls_n, gt_inst_cls_valid_n = self._prepare_gt_instance_classes_for_matching(
                segmentation_cls_instance3d=gt_segmentation_cls_instance3d_tcxyz,
                gt_instance_ids_n=gt_inst_ids_n,
            )
        gt_inst_center_world_full_tn3 = gt_inst_center_world_tn3
        gt_inst_center_valid_full_tn = gt_inst_center_valid_tn
        gt_inst_ids_full_n = gt_inst_ids_n
        gt_inst_cls_full_n = gt_inst_cls_n
        gt_inst_cls_valid_full_n = gt_inst_cls_valid_n
        full_center_pack = self._build_full_query_instance_centers_from_history_and_traj(
            history_centers_tn3=gt_inst_center_world_hist_tn3,
            history_valid_tn=gt_inst_center_valid_hist_tn,
            history_ids_n=gt_inst_ids_hist_n,
            traj_centers_tn3=gt_inst_center_world_tn3,
            traj_valid_tn=gt_inst_center_valid_tn,
            traj_ids_n=gt_inst_ids_n,
        )
        if torch.is_tensor(full_center_pack[0]) and torch.is_tensor(full_center_pack[1]) and torch.is_tensor(full_center_pack[2]):
            gt_inst_center_world_full_tn3 = full_center_pack[0]
            gt_inst_center_valid_full_tn = full_center_pack[1]
            gt_inst_ids_full_n = full_center_pack[2]
            gt_inst_cls_full_n = self._reindex_vector_by_instance_ids(
                values_n=gt_inst_cls_n,
                instance_ids_n=gt_inst_ids_n,
                target_ids_n=gt_inst_ids_full_n,
            )
            gt_inst_cls_valid_full_n = self._reindex_vector_by_instance_ids(
                values_n=gt_inst_cls_valid_n,
                instance_ids_n=gt_inst_ids_n,
                target_ids_n=gt_inst_ids_full_n,
            )

        # centers_world: 51.2 미터 단위 스케일링 된 query center 좌표
        # center_logits: 미터 단위 스케일링 안 된 센터 좌표
        # gaussian_sigmas_world: query별 3D Gaussian sigma(m)
        (
            pred_occ,
            _center_logits,
            gaussian_sigmas_world,
            img_feats,
            centers_world,
            centers_world_traj_tq3,
            traj_frame_indices_t,
            query_cls_logits_qc,
            query_cls_scores_qc,
            query_cls_scores_tqc,
            query_depth_logits_tqd,
            query_depth_probs_tqd,
            query_img_feat_pooled_tqd,
            _query_future_feat_tqd,
            _bev_feats_src,
            bev_feats_enc,
            _query_attn_weights_tqnhw,
            _query_attn_bbox_targets,
            instance_img_debug_bundle,
            query_match_inputs,
            _traj_offsets_fq2,
            _query_present_local_idx,
            mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3,
            mixture_yaw_tqg,
            mixture_weights_tqg,
            gaussian_sigmas_world_traj_tq3,
            mixture_centers_world_traj_tqg3,
            mixture_sigmas_world_traj_tqg3,
            mixture_yaw_traj_tqg,
            mixture_weights_traj_tqg,
        ) = self.extract_feat_query(
            img_inputs_seq=img_inputs_seq,
            img_metas=img_metas,
            future_egomotion=future_egomotion,
            gt_segmentation_instance3d_txyz_for_attn=gt_instance_occ3d_txyz_primary,
            fallback_segmentation_instance3d_txyz_for_attn=gt_segmentation_instance3d_txyz,
            voxelize=False,
            return_bev_occ_feats=True,
            return_instance_img_debug=need_instance_img_debug,
            return_instance_img_debug_fullseq=need_instance_img_debug_fullseq,
            return_query_cam_gaussian_vis_debug=need_query_cam_gaussian_vis,
            return_query_match_inputs=True,
        )
        # Keep a minimal default loss path until full query-loss branches are enabled.
        _pick = self._pick_tensor
        centers_world_loss = _pick(centers_world_traj_tq3, centers_world)
        gaussian_sigmas_world_loss = _pick(gaussian_sigmas_world_traj_tq3, gaussian_sigmas_world)
        mixture_centers_world_loss = _pick(mixture_centers_world_traj_tqg3, mixture_centers_world_tqg3)
        mixture_sigmas_world_loss = _pick(mixture_sigmas_world_traj_tqg3, mixture_sigmas_world_tqg3)
        mixture_yaw_loss = _pick(mixture_yaw_traj_tqg, mixture_yaw_tqg)
        mixture_weights_loss = _pick(mixture_weights_traj_tqg, mixture_weights_tqg)
        expected_traj_horizon = int(self.n_future_frames) + 1
        centers_world_temporal_sup_tq3 = centers_world_loss
        traj_centers_are_present_aligned = bool(
            torch.is_tensor(centers_world_loss)
            and centers_world_loss.dim() == 3
            and int(centers_world_loss.shape[0]) == expected_traj_horizon
        )
        if (
            (bool(self.use_query_inst_center_match_loss) or float(self.query_traj_loss_weight) > 0.0)
            and torch.is_tensor(centers_world_loss)
            and centers_world_loss.dim() == 3
            and int(centers_world_loss.shape[0]) > 1
            and (not traj_centers_are_present_aligned)
        ):
            centers_world_temporal_sup_tq3 = self._align_query_centers_to_present_frame(
                centers_world_tq3=centers_world_loss,
                future_egomotion=future_egomotion,
                traj_frame_indices_t=traj_frame_indices_t,
            )
        centers_world_dt_gt2p_tq3 = centers_world_loss
        if (
            torch.is_tensor(centers_world_loss)
            and centers_world_loss.dim() == 3
            and int(centers_world_loss.shape[0]) > 1
            and torch.is_tensor(traj_frame_indices_t)
            and (not traj_centers_are_present_aligned)
        ):
            centers_world_dt_gt2p_tq3 = self._align_query_centers_to_present_frame(
                centers_world_tq3=centers_world_loss,
                future_egomotion=future_egomotion,
                traj_frame_indices_t=traj_frame_indices_t,
            )
        centers_world_vis_tq3 = centers_world_loss
        gaussian_sigmas_world_vis_tq3 = gaussian_sigmas_world_loss
        mixture_centers_world_vis_tqg3 = mixture_centers_world_loss
        mixture_sigmas_world_vis_tqg3 = mixture_sigmas_world_loss
        mixture_yaw_vis_tqg = mixture_yaw_loss
        mixture_weights_vis_tqg = mixture_weights_loss
        centers_world_loss_aligned_tq3 = centers_world_temporal_sup_tq3
        gaussian_sigmas_world_loss_aligned_tq3 = gaussian_sigmas_world_loss
        mixture_centers_world_loss_aligned_tqg3 = mixture_centers_world_loss
        mixture_sigmas_world_loss_aligned_tqg3 = mixture_sigmas_world_loss
        mixture_yaw_loss_aligned_tqg = mixture_yaw_loss
        mixture_weights_loss_aligned_tqg = mixture_weights_loss
        _needs_ego_align = (
            torch.is_tensor(centers_world_loss)
            and centers_world_loss.dim() == 3
            and int(centers_world_loss.shape[0]) > 1
            and torch.is_tensor(traj_frame_indices_t)
            and torch.is_tensor(future_egomotion)
            and (not traj_centers_are_present_aligned)
        )
        if _needs_ego_align:
            loss_geom_pack = self._align_geom_pack(
                centers_world_loss, traj_frame_indices_t, future_egomotion,
                gaussian_sigmas_world_loss, mixture_centers_world_loss,
                mixture_sigmas_world_loss, mixture_yaw_loss, mixture_weights_loss,
            )
            centers_world_loss_aligned_tq3 = loss_geom_pack.get("centers_world_tq3", centers_world_loss_aligned_tq3)
            gaussian_sigmas_world_loss_aligned_tq3 = loss_geom_pack.get("gaussian_sigmas_world_tq3", gaussian_sigmas_world_loss_aligned_tq3)
            mixture_centers_world_loss_aligned_tqg3 = loss_geom_pack.get("mixture_centers_world_tqg3", mixture_centers_world_loss_aligned_tqg3)
            mixture_sigmas_world_loss_aligned_tqg3 = loss_geom_pack.get("mixture_sigmas_world_tqg3", mixture_sigmas_world_loss_aligned_tqg3)
            mixture_yaw_loss_aligned_tqg = loss_geom_pack.get("mixture_yaw_tqg", mixture_yaw_loss_aligned_tqg)
            mixture_weights_loss_aligned_tqg = loss_geom_pack.get("mixture_weights_tqg", mixture_weights_loss_aligned_tqg)
        if _needs_ego_align:
            with torch.no_grad():
                vis_geom_pack = self._align_geom_pack(
                    centers_world_loss, traj_frame_indices_t, future_egomotion,
                    gaussian_sigmas_world_loss, mixture_centers_world_loss,
                    mixture_sigmas_world_loss, mixture_yaw_loss, mixture_weights_loss,
                    detach=True,
                )
            centers_world_vis_tq3 = vis_geom_pack.get("centers_world_tq3", centers_world_vis_tq3)
            gaussian_sigmas_world_vis_tq3 = vis_geom_pack.get("gaussian_sigmas_world_tq3", gaussian_sigmas_world_vis_tq3)
            mixture_centers_world_vis_tqg3 = vis_geom_pack.get("mixture_centers_world_tqg3", mixture_centers_world_vis_tqg3)
            mixture_sigmas_world_vis_tqg3 = vis_geom_pack.get("mixture_sigmas_world_tqg3", mixture_sigmas_world_vis_tqg3)
            mixture_yaw_vis_tqg = vis_geom_pack.get("mixture_yaw_tqg", mixture_yaw_vis_tqg)
            mixture_weights_vis_tqg = vis_geom_pack.get("mixture_weights_tqg", mixture_weights_vis_tqg)

        # Build 7-frame visualization tensors (past + present + future) by
        # prepending the lifted past frames to the trajectory horizon.
        #
        # centers_world is [T_past, Q, 3] expressed in present lidar coords
        # (returned by `_build_query_attn_soft_lift_pack`). Indices
        # [0, _query_present_local_idx) correspond to (t - time_receptive_field
        # + 1, ..., t - 1) — the strictly past frames.
        # centers_world_vis_tq3 / mixture_*_vis_* are the trajectory horizon
        # already aligned to present lidar coords; concatenating gives
        # [time_receptive_field + n_future_frames, Q, *] = 7 frames for the
        # canonical config (3 past + 5 traj-horizon - 1 overlap = 7).
        centers_world_vis_full_tq3 = centers_world_vis_tq3
        gaussian_sigmas_world_vis_full_tq3 = gaussian_sigmas_world_vis_tq3
        mixture_centers_world_vis_full_tqg3 = mixture_centers_world_vis_tqg3
        mixture_sigmas_world_vis_full_tqg3 = mixture_sigmas_world_vis_tqg3
        mixture_yaw_vis_full_tqg = mixture_yaw_vis_tqg
        mixture_weights_vis_full_tqg = mixture_weights_vis_tqg
        centers_world_loss_full_tq3 = centers_world_loss_aligned_tq3
        gaussian_sigmas_world_loss_full_tq3 = gaussian_sigmas_world_loss_aligned_tq3
        mixture_centers_world_loss_full_tqg3 = mixture_centers_world_loss_aligned_tqg3
        mixture_sigmas_world_loss_full_tqg3 = mixture_sigmas_world_loss_aligned_tqg3
        mixture_yaw_loss_full_tqg = mixture_yaw_loss_aligned_tqg
        mixture_weights_loss_full_tqg = mixture_weights_loss_aligned_tqg
        past_only_count = int(_query_present_local_idx)
        # Only prepend past frames when the vis geometry actually came from the
        # trajectory builder (shape[0] == n_future_frames + 1 = present + future).
        # In the fallback case where traj is missing, centers_world_vis_tq3 is
        # identical to centers_world and we must avoid duplicating past frames.
        traj_vis_has_present_anchor = (
            torch.is_tensor(centers_world_vis_tq3)
            and centers_world_vis_tq3.dim() == 3
            and int(centers_world_vis_tq3.shape[0]) == expected_traj_horizon
            and torch.is_tensor(centers_world)
            and centers_world.dim() == 3
            and int(centers_world.shape[0]) > past_only_count
            and past_only_count > 0
        )
        if traj_vis_has_present_anchor:
            _ppd = lambda p, t: self._prepend_past_frames(p, past_only_count, t, detach=True)
            centers_world_vis_full_tq3 = _ppd(centers_world, centers_world_vis_tq3)
            gaussian_sigmas_world_vis_full_tq3 = _ppd(gaussian_sigmas_world, gaussian_sigmas_world_vis_tq3)
            mixture_centers_world_vis_full_tqg3 = _ppd(mixture_centers_world_tqg3, mixture_centers_world_vis_tqg3)
            mixture_sigmas_world_vis_full_tqg3 = _ppd(mixture_sigmas_world_tqg3, mixture_sigmas_world_vis_tqg3)
            mixture_yaw_vis_full_tqg = _ppd(mixture_yaw_tqg, mixture_yaw_vis_tqg)
            mixture_weights_vis_full_tqg = _ppd(mixture_weights_tqg, mixture_weights_vis_tqg)
        traj_loss_has_present_anchor = (
            torch.is_tensor(centers_world_loss_aligned_tq3)
            and centers_world_loss_aligned_tq3.dim() == 3
            and int(centers_world_loss_aligned_tq3.shape[0]) == expected_traj_horizon
            and torch.is_tensor(centers_world)
            and centers_world.dim() == 3
            and int(centers_world.shape[0]) > past_only_count
            and past_only_count > 0
        )
        if traj_loss_has_present_anchor:
            _pp = lambda p, t: self._prepend_past_frames(p, past_only_count, t)
            centers_world_loss_full_tq3 = _pp(centers_world, centers_world_loss_aligned_tq3)
            gaussian_sigmas_world_loss_full_tq3 = _pp(gaussian_sigmas_world, gaussian_sigmas_world_loss_aligned_tq3)
            mixture_centers_world_loss_full_tqg3 = _pp(mixture_centers_world_tqg3, mixture_centers_world_loss_aligned_tqg3)
            mixture_sigmas_world_loss_full_tqg3 = _pp(mixture_sigmas_world_tqg3, mixture_sigmas_world_loss_aligned_tqg3)
            mixture_yaw_loss_full_tqg = _pp(mixture_yaw_tqg, mixture_yaw_loss_aligned_tqg)
            mixture_weights_loss_full_tqg = _pp(mixture_weights_tqg, mixture_weights_loss_aligned_tqg)
        use_full7_temporal_sup = bool(
            traj_loss_has_present_anchor
            and torch.is_tensor(centers_world_loss_full_tq3)
            and torch.is_tensor(gt_inst_center_world_full_tn3)
            and int(gt_inst_center_world_full_tn3.shape[0]) == int(centers_world_loss_full_tq3.shape[0])
        )
        centers_world_match_tq3 = centers_world_temporal_sup_tq3
        gaussian_sigmas_world_match_tq3 = gaussian_sigmas_world_loss
        mixture_centers_world_match_tqg3 = mixture_centers_world_loss
        mixture_sigmas_world_match_tqg3 = mixture_sigmas_world_loss
        mixture_yaw_match_tqg = mixture_yaw_loss
        mixture_weights_match_tqg = mixture_weights_loss
        match_gt_instance_occ3d_txyz = gt_instance_occ3d_txyz_query
        match_gt_centers_tn3 = gt_inst_center_world_tn3
        match_gt_valid_tn = gt_inst_center_valid_tn
        match_gt_ids_n = gt_inst_ids_n
        match_gt_cls_n = gt_inst_cls_n
        match_gt_cls_valid_n = gt_inst_cls_valid_n
        match_temporal_cost_frame_indices = None
        match_center_frame_idx = 0
        traj_loss_present_local_idx = 0
        if use_full7_temporal_sup:
            centers_world_match_tq3 = centers_world_loss_full_tq3
            gaussian_sigmas_world_match_tq3 = gaussian_sigmas_world_loss_full_tq3
            mixture_centers_world_match_tqg3 = mixture_centers_world_loss_full_tqg3
            mixture_sigmas_world_match_tqg3 = mixture_sigmas_world_loss_full_tqg3
            mixture_yaw_match_tqg = mixture_yaw_loss_full_tqg
            mixture_weights_match_tqg = mixture_weights_loss_full_tqg
            match_gt_instance_occ3d_txyz = gt_instance_occ3d_txyz_query_vis
            match_gt_centers_tn3 = gt_inst_center_world_full_tn3
            match_gt_valid_tn = gt_inst_center_valid_full_tn
            match_gt_ids_n = gt_inst_ids_full_n
            match_gt_cls_n = gt_inst_cls_full_n
            match_gt_cls_valid_n = gt_inst_cls_valid_full_n
            match_temporal_cost_frame_indices = list(range(int(self.time_receptive_field)))
            match_center_frame_idx = None
            traj_loss_present_local_idx = int(_query_present_local_idx)
        query_cls_loss = None
        query_depth_loss = None
        query_attn_bbox_loss = None
        query_attn_cam_score_pack = None
        query_inst_depth_target_pack = None
        matched_gmo_loss = None
        center_match_loss = None
        query_traj_loss = None
        inst_match_result = None
        num_matched_queries = 0
        num_total_queries = int(centers_world.shape[1]) if torch.is_tensor(centers_world) else 0

        gt_inst_img_feat_tnd, gt_inst_img_feat_valid_tn, gt_inst_img_feat_ids_n = self.pool_gt_instance_context_features(
            segmentation_instance3d_txyz=gt_instance_occ3d_txyz_primary,
            future_egomotion=future_egomotion,
            gt_inst_ids_n=gt_inst_ids_n,
            query_match_inputs=query_match_inputs,
            fallback_segmentation_instance3d_txyz=gt_segmentation_instance3d_txyz,
        )
        self._last_gt_instance_bev_feat_cache = None
        if torch.is_tensor(gt_inst_img_feat_tnd):
            self._last_gt_instance_bev_feat_cache = {
                "feat_tnd": gt_inst_img_feat_tnd.detach(),
                "valid_tn": gt_inst_img_feat_valid_tn.detach() if torch.is_tensor(gt_inst_img_feat_valid_tn) else None,
                "ids_n": gt_inst_img_feat_ids_n.detach() if torch.is_tensor(gt_inst_img_feat_ids_n) else None,
            }
        if (
            torch.is_tensor(gt_inst_ids_full_n)
            and torch.is_tensor(gt_inst_img_feat_ids_n)
            and torch.is_tensor(gt_inst_img_feat_tnd)
            and torch.is_tensor(gt_inst_img_feat_valid_tn)
        ):
            feat_full_tnd = self._reindex_temporal_tensor_by_instance_ids(
                values_tn=gt_inst_img_feat_tnd,
                instance_ids_n=gt_inst_img_feat_ids_n,
                target_ids_n=gt_inst_ids_full_n,
            )
            feat_valid_full_tn = self._reindex_temporal_tensor_by_instance_ids(
                values_tn=gt_inst_img_feat_valid_tn,
                instance_ids_n=gt_inst_img_feat_ids_n,
                target_ids_n=gt_inst_ids_full_n,
            )
            if torch.is_tensor(feat_full_tnd) and torch.is_tensor(feat_valid_full_tn):
                gt_inst_img_feat_tnd = feat_full_tnd
                gt_inst_img_feat_valid_tn = feat_valid_full_tn
                gt_inst_img_feat_ids_n = gt_inst_ids_full_n
        self._last_query_inst_depth_target_pack = None
        if (
            isinstance(query_match_inputs, dict)
            and torch.is_tensor(gt_inst_center_world_hist_tn3)
            and torch.is_tensor(gt_inst_center_valid_hist_tn)
        ):
            query_inst_depth_target_pack = self.build_query_inst_depth_targets(
                gt_inst_center_world_tn3=gt_inst_center_world_hist_tn3,
                gt_inst_center_valid_tn=gt_inst_center_valid_hist_tn,
                gt_inst_ids_n=gt_inst_ids_hist_n,
                future_egomotion=future_egomotion,
                query_match_inputs=query_match_inputs,
            )
            if isinstance(query_inst_depth_target_pack, dict):
                self._last_query_inst_depth_target_pack = {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in query_inst_depth_target_pack.items()
                }

        match_query_feat_tqd = None
        if self.query_match_feature_source == "query_img_feat_pooled":
            match_query_feat_tqd = query_img_feat_pooled_tqd

        q_sel_tqd, g_sel_tnd, g_sel_valid_tn = self._select_matching_feature_frames(
            query_feat_tqd=match_query_feat_tqd,
            gt_feat_tnd=gt_inst_img_feat_tnd,
            gt_valid_tn=gt_inst_img_feat_valid_tn,
        )
        inst_match_result = self._match_queries_to_gt_instances(
            centers_world=centers_world_match_tq3,
            query_sigmas_world_tq3=gaussian_sigmas_world_match_tq3,
            query_mixture_centers_world_tqg3=mixture_centers_world_match_tqg3,
            query_mixture_sigmas_world_tqg3=mixture_sigmas_world_match_tqg3,
            query_mixture_yaw_tqg=mixture_yaw_match_tqg,
            query_mixture_weights_tqg=mixture_weights_match_tqg,
            gt_instance_occ3d_txyz=match_gt_instance_occ3d_txyz,
            gt_inst_center_world_tn3=match_gt_centers_tn3,
            gt_inst_center_valid_tn=match_gt_valid_tn,
            gt_inst_ids_n=match_gt_ids_n,
            gt_inst_cls_n=match_gt_cls_n,
            gt_inst_cls_valid_n=match_gt_cls_valid_n,
            query_img_feat_tqd=q_sel_tqd,
            query_cls_logits_qc=query_cls_logits_qc,
            gt_inst_bev_feat_tnd=g_sel_tnd,
            gt_inst_bev_feat_valid_tn=g_sel_valid_tn,
            gt_inst_bev_feat_ids_n=gt_inst_img_feat_ids_n,
            query_attn_weights_tqnhw=_query_attn_weights_tqnhw,
            gt_attn_targets=_query_attn_bbox_targets,
            query_sim_cost_weight=self.query_sim_match_cost_weight,
            query_cls_cost_weight=self.query_cls_match_cost_weight,
            query_soft_assign_temp=self.query_soft_assign_temp,
            query_soft_assign_cost_weight=self.query_soft_assign_cost_weight,
            query_bev_dice_cost_weight=self.query_bev_dice_match_cost_weight,
            query_attn_match_cost_weight=self.query_attn_match_cost_weight,
            query_attn_match_metric=self.query_attn_match_metric,
            query_attn_match_pred_norm=self.query_attn_match_pred_norm,
            query_attn_match_eps=self.query_attn_match_eps,
            query_center_match_frame_idx=match_center_frame_idx,
            query_temporal_cost_frame_indices=match_temporal_cost_frame_indices,
        )
        self._last_inst_match_result = inst_match_result
        if isinstance(inst_match_result, dict):
            matched_query_idx = inst_match_result.get("matched_query_idx", None)
            if torch.is_tensor(matched_query_idx):
                num_matched_queries = int(matched_query_idx.numel())

        query_cls_loss = self._compute_query_cls_loss(
            query_cls_logits_qc=query_cls_logits_qc,
            inst_match_result=inst_match_result,
            loss_weight=float(self.query_cls_loss_weight),
            bg_index=int(self.query_bg_class),
            class_weights=self.query_cls_loss_class_weights,
        )
        query_depth_loss = self._compute_query_depth_loss_from_match(
            query_depth_logits_tqd=query_depth_logits_tqd,
            query_attn_weights_tqnhw=_query_attn_weights_tqnhw,
            inst_match_result=inst_match_result,
            query_inst_depth_target_pack=query_inst_depth_target_pack,
            loss_weight=float(self.query_depth_loss_weight),
            label_smoothing=float(self.query_depth_label_smoothing),
        )
        if self.use_query_inst_center_match_loss:
            center_match_loss = self._compute_query_center_match_loss_from_match(
                centers_world_tq3=centers_world_match_tq3,
                inst_match_result=inst_match_result,
                loss_weight=float(self.query_center_routed_loss_weight),
                loss_type=self.query_center_match_loss_type,
            )
        query_traj_loss = self._compute_query_trajectory_loss_from_match(
            centers_world_tq3=centers_world_match_tq3,
            pred_traj_offsets_fq2=_traj_offsets_fq2,
            inst_match_result=inst_match_result,
            loss_weight=float(self.query_traj_loss_weight),
            loss_type=self.query_traj_loss_type,
            present_local_idx=int(traj_loss_present_local_idx),
            moving_reweight_enabled=bool(self.query_traj_moving_reweight_enabled),
            moving_threshold_m=float(self.query_traj_moving_threshold_m),
            moving_weight=float(self.query_traj_moving_weight),
            static_weight=float(self.query_traj_static_weight),
        )
        query_attn_bbox_attn_export_enabled = float(
            1.0 if torch.is_tensor(_query_attn_weights_tqnhw) else 0.0
        )
        if (
            self.use_query_attn_bbox_loss
            and torch.is_tensor(_query_attn_weights_tqnhw)
            and isinstance(_query_attn_bbox_targets, dict)
            and isinstance(inst_match_result, dict)
        ):
            query_attn_bbox_loss = self._compute_query_attn_bbox_loss(
                query_attn_weights_tqnhw=_query_attn_weights_tqnhw,
                inst_match_result=inst_match_result,
                gt_attn_targets=_query_attn_bbox_targets,
            )
            if isinstance(query_attn_bbox_loss, dict):
                gt_mask_tnhw = _query_attn_bbox_targets.get("gt_inst_mask_tnhw", None)
                has_empty_gt_mask = 0.0
                if torch.is_tensor(gt_mask_tnhw) and gt_mask_tnhw.numel() > 0:
                    has_empty_gt_mask = float(1.0 if (not bool(gt_mask_tnhw.any().item())) else 0.0)
                query_attn_bbox_loss["dbg_query_attn_bbox_empty_gt_mask"] = centers_world.new_tensor(has_empty_gt_mask)
        if (
            self.use_query_attn_cam_gaussian_score
            and torch.is_tensor(_query_attn_weights_tqnhw)
            and isinstance(query_match_inputs, dict)
            and torch.is_tensor(mixture_centers_world_loss)
            and torch.is_tensor(mixture_sigmas_world_loss)
            and torch.is_tensor(mixture_yaw_loss)
            and torch.is_tensor(mixture_weights_loss)
        ):
            with torch.no_grad():
                query_attn_cam_targets = self.build_query_attn_cam_gaussian_targets(
                    mixture_centers_world_tqg3=mixture_centers_world_loss,
                    mixture_sigmas_world_tqg3=mixture_sigmas_world_loss,
                    mixture_yaw_tqg=mixture_yaw_loss,
                    mixture_weights_tqg=mixture_weights_loss,
                    future_egomotion=future_egomotion,
                    query_match_inputs=query_match_inputs,
                )
                if isinstance(query_attn_cam_targets, dict):
                    query_attn_cam_score_pack = self._compute_query_attn_cam_gaussian_score(
                        query_attn_weights_tqnhw=_query_attn_weights_tqnhw,
                        cam_targets=query_attn_cam_targets,
                    )
                    pass  # query_attn_cam_score_pack may be None or a valid score dict
        if (
            self.use_gmo_bce_loss
            and (not self.center_only_mode)
            and torch.is_tensor(mixture_centers_world_match_tqg3)
            and torch.is_tensor(mixture_sigmas_world_match_tqg3)
            and torch.is_tensor(mixture_yaw_match_tqg)
            and torch.is_tensor(mixture_weights_match_tqg)
            and torch.is_tensor(match_gt_instance_occ3d_txyz)
        ):
            matched_gmo_loss = self._compute_matched_pair_gmo_losses(
                centers_world_tq3=centers_world_match_tq3,
                sigmas_world_tq3=gaussian_sigmas_world_match_tq3,
                mixture_centers_world_tqg3=mixture_centers_world_match_tqg3,
                mixture_sigmas_world_tqg3=mixture_sigmas_world_match_tqg3,
                mixture_yaw_tqg=mixture_yaw_match_tqg,
                mixture_weights_tqg=mixture_weights_match_tqg,
                gt_instance_occ3d_txyz_pred=match_gt_instance_occ3d_txyz,
                objectness_scores_tq=None,
                inst_match_result=inst_match_result,
                loss_weight=0.1,
                loss_type=self.query_gmo_loss_type,
                focal_gamma=float(self.query_gmo_focal_gamma),
                focal_alpha=float(self.query_gmo_focal_alpha),
                compute_dice=bool(self.use_query_gmo_dice_loss),
                dice_loss_weight=float(self.query_gmo_dice_loss_weight),
                tversky_alpha=float(self.query_gmo_tversky_alpha),
                tversky_beta=float(self.query_gmo_tversky_beta),
                pair_chunk_size=int(self.query_multi_gaussian_pair_chunk),
            )

        if torch.is_tensor(query_cls_scores_tqc) and query_cls_scores_tqc.numel() > 0:
            query_conf_scores_tq = (1.0 - query_cls_scores_tqc[..., self.query_bg_class]).to(torch.float32)
            query_cls_pred_tq = torch.argmax(query_cls_scores_tqc, dim=-1)
        else:
            query_conf_scores_tq = centers_world.new_zeros(centers_world.shape[:-1], dtype=torch.float32)
            query_cls_pred_tq = centers_world.new_zeros(centers_world.shape[:-1], dtype=torch.long)
        if torch.is_tensor(centers_world_vis_full_tq3) and centers_world_vis_full_tq3.dim() == 3:
            t_vis = int(centers_world_vis_full_tq3.shape[0])
            if (
                torch.is_tensor(query_conf_scores_tq)
                and query_conf_scores_tq.dim() == 2
                and int(query_conf_scores_tq.shape[0]) != t_vis
            ):
                if int(query_conf_scores_tq.shape[0]) == 1:
                    query_conf_scores_tq = query_conf_scores_tq.expand(t_vis, -1).contiguous()
                else:
                    t_keep = min(t_vis, int(query_conf_scores_tq.shape[0]))
                    pad = query_conf_scores_tq.new_zeros(
                        (max(0, t_vis - t_keep), int(query_conf_scores_tq.shape[1]))
                    )
                    query_conf_scores_tq = torch.cat([query_conf_scores_tq[:t_keep], pad], dim=0)
            if (
                torch.is_tensor(query_cls_pred_tq)
                and query_cls_pred_tq.dim() == 2
                and int(query_cls_pred_tq.shape[0]) != t_vis
            ):
                if int(query_cls_pred_tq.shape[0]) == 1:
                    query_cls_pred_tq = query_cls_pred_tq.expand(t_vis, -1).contiguous()
                else:
                    t_keep = min(t_vis, int(query_cls_pred_tq.shape[0]))
                    pad = query_cls_pred_tq.new_zeros(
                        (max(0, t_vis - t_keep), int(query_cls_pred_tq.shape[1]))
                    )
                    query_cls_pred_tq = torch.cat([query_cls_pred_tq[:t_keep], pad], dim=0)
        _d = self._detach_if_tensor
        with torch.no_grad():
            query_vis_bundle = self._build_query_visualization_bundle(
                centers_world_tq3=_d(centers_world_vis_full_tq3),
                query_cls_scores_qc=_d(query_cls_scores_qc),
                inst_match_result=inst_match_result,
                gt_segmentation_instance3d_txyz=gt_instance_occ3d_txyz_query_vis,
                gaussian_sigmas_world_tq3=_d(gaussian_sigmas_world_vis_full_tq3),
                mixture_centers_world_tqg3=_d(mixture_centers_world_vis_full_tqg3),
                mixture_sigmas_world_tqg3=_d(mixture_sigmas_world_vis_full_tqg3),
                mixture_yaw_tqg=_d(mixture_yaw_vis_full_tqg),
                mixture_weights_tqg=_d(mixture_weights_vis_full_tqg),
                query_attn_cam_score_pack=query_attn_cam_score_pack,
            )

        # LSS view transform -> BEV feature -> occ head (2D occupancy) training route.
        if segmentation_bev is not None:
            if segmentation_bev.dim() >= 4:
                segmentation_bev = segmentation_bev[:, -self.n_future_frames_plus:, ...].contiguous()
            elif segmentation_bev.dim() == 3:
                segmentation_bev = segmentation_bev[-self.n_future_frames_plus:, ...].unsqueeze(0).contiguous()

        transform = img_inputs_seq[1:8] if img_inputs_seq is not None else None

        
        # Save query debug visualization regardless of individual loss switches.
        with torch.no_grad():
            self.query_head.maybe_save_query_debug_vis(
                pred_occ_prob=pred_occ.detach() if pred_occ is not None else None,
                gt_occ=query_gt_occ_aligned,
                gt_occ_inst=(
                    gt_segmentation_cls_instance3d_for_match_query_vis.detach()
                    if torch.is_tensor(gt_segmentation_cls_instance3d_for_match_query_vis)
                    else None
                ),
                gt_occ_semantic=(
                    gt_segmentation_cls_instance3d_for_match_query_vis[:, 0].detach()
                    if torch.is_tensor(gt_segmentation_cls_instance3d_for_match_query_vis)
                    else query_gt_occ_aligned
                ),
                gmo_ids=query_gmo_ids,
                ignore_index=255,
                points_world=centers_world_vis_full_tq3.detach(),
                step=cur_train_iter,
                # 7 frames: time_receptive_field (past+present) + n_future_frames.
                max_frames=int(self.time_receptive_field) + int(self.n_future_frames),
                voxel_center_offset=0.5,
                prob_threshold=0.5,
                pred_layout="zyx",
                pred_occ_prob_pos_obj=None,
                pred_occ_prob_all_obj=pred_occ.detach() if pred_occ is not None else None,
                points_world_all=centers_world_vis_full_tq3.detach(),
                point_conf_all=query_conf_scores_tq.detach(),
                point_class_ids_all=query_cls_pred_tq.detach(),
                query_vis_bundle=query_vis_bundle,
            )
        self.maybe_save_instance_img_debug(
            debug_bundle=instance_img_debug_bundle,
            segmentation_instance3d=gt_instance_occ3d_txyz_primary,
            segmentation_cls_instance3d=gt_segmentation_cls_instance3d_for_match,
            future_egomotion=future_egomotion,
            step=cur_train_iter,
        )
        self.maybe_save_query_cam_gaussian_vis(
            debug_bundle=instance_img_debug_bundle,
            query_vis_bundle=query_vis_bundle,
            future_egomotion=future_egomotion,
            step=cur_train_iter,
            gt_segmentation_instance3d=gt_instance_occ3d_txyz_query,
        )
        self.maybe_save_gt_alignment_bev_vis(
            segmentation=segmentation,
            gt_occ=gt_occ,
            gt_occ_inst=gt_occ_inst,
            img_metas=img_metas,
            step=cur_train_iter,
        )
        self.maybe_save_query_inst_depth_lift_vis(
            img_inputs_seq=img_inputs_seq,
            query_match_inputs=query_match_inputs,
            query_inst_depth_target_pack=query_inst_depth_target_pack,
            gt_segmentation_instance3d_txyz_fine=gt_instance_occ3d_txyz_primary,
            gt_segmentation_instance3d_txyz_bbox=gt_segmentation_instance3d_txyz,
            future_egomotion=future_egomotion,
            img_metas=img_metas,
            step=cur_train_iter,
        )
        self.maybe_save_query_attn_softargmax_vis(
            img_inputs_seq=img_inputs_seq,
            query_attn_weights_tqnhw=_query_attn_weights_tqnhw,
            query_attn_soft_lift_pack=self._last_query_attn_soft_lift_pack,
            query_match_inputs=query_match_inputs,
            inst_match_result=inst_match_result,
            gt_segmentation_instance3d_txyz_fine=gt_instance_occ3d_txyz_query_vis,
            gt_segmentation_instance3d_txyz_bbox=gt_segmentation_instance3d_txyz_query_vis,
            gt_source_start_global=0,
            future_egomotion=future_egomotion,
            img_metas=img_metas,
            step=cur_train_iter,
        )

        return self._aggregate_training_losses(
            bev_feats_enc=bev_feats_enc,
            segmentation_bev=segmentation_bev,
            points_occ=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            transform=transform,
            query_cls_loss=query_cls_loss,
            query_depth_loss=query_depth_loss,
            query_attn_bbox_loss=query_attn_bbox_loss,
            matched_gmo_loss=matched_gmo_loss,
            center_match_loss=center_match_loss,
            query_traj_loss=query_traj_loss,
            query_attn_cam_score_pack=query_attn_cam_score_pack,
            centers_world=centers_world,
            centers_world_dt_gt2p_tq3=centers_world_dt_gt2p_tq3,
            gaussian_sigmas_world_loss=gaussian_sigmas_world_loss,
            mixture_sigmas_world_loss=mixture_sigmas_world_loss,
            mixture_weights_loss=mixture_weights_loss,
            occ_dt=occ_dt,
            gt_inst_center_world_tn3=gt_inst_center_world_tn3,
            gt_inst_center_valid_tn=gt_inst_center_valid_tn,
        )
