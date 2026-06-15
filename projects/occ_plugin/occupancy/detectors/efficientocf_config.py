MODEL_CFG_DEFAULTS = {
    "gmo_ids": (2, 3, 4, 5, 6, 7, 9, 10),
    "use_segmentation_as_query_gt": False,
    "use_gmo_bce_loss": False,
    "query_gmo_loss_type": "balanced_bce",
    "query_gmo_focal_gamma": 2.0,
    "query_gmo_focal_alpha": 0.25,
    "query_gmo_dice_loss_weight": 0.5,
    "query_gmo_tversky_alpha": 0.7,
    "query_gmo_tversky_beta": 0.3,
    "use_query_gmo_dice_loss": True,
    "use_query_inst_center_match_loss": True,
    "use_query_dt_loss": True,
    "use_query_gt2p_instance_labeled_loss": False,
    "query_gt2p_instance_labeled_loss_weight": 0.1,
    "query_gt2p_instance_labeled_balance_weight": 0.25,
    "query_gt2p_instance_labeled_tau": 1.0,
    "query_gt2p_instance_labeled_assign_sigma_xyz": (4.0, 4.0, 1.5),
    "query_gt2p_instance_labeled_sigma_policy": "fixed",
    "query_require_history_all_valid": False,
    "query_gt2p_cooldown_enabled": False,
    "query_gt2p_cooldown_start_iter": 0,
    "query_gt2p_cooldown_iters": 0,
    "query_gt2p_cooldown_min_scale": 0.0,
    "center_only_mode": False,
    "query_feat_cosine_threshold": 0.30,
    "query_center_distance_threshold_m": 1.50,
    "query_center_loss_detach_query_feat": False,
    "query_feat_unmatched_neg_loss_weight": 0.0,
    "query_feat_unmatched_neg_margin": 0.2,
    "query_bev_pool_fixed_sigma_xyz": (4.0, 4.0, 1.5),
    "query_class_ids": None,
    "query_class_names": None,
    "strict_query_class_id_validation": False,
    "query_num_classes": 3,
    "query_cls_loss_weight": 1.0,
    "query_cls_loss_class_weights": None,
    "query_match_feature_source": "query_img_feat_pooled",
    "query_soft_assign_temp": 0.10,
    "query_soft_assign_cost_weight": 0.0,
    "query_sim_match_cost_weight": 0.0,
    "query_cls_match_cost_weight": 0.0,
    "query_bev_dice_match_cost_weight": 0.0,
    "query_center_routed_loss_weight": 0.1,
    "query_traj_matched_only": True,
    "query_traj_loss_weight": 0.0,
    "query_traj_loss_type": "l1",
    "query_traj_residual_max_m": (8.0, 8.0),
    "query_traj_prior_detach": True,
    "query_traj_num_modes": 1,
    "query_traj_decoder_type": "offset",
    "query_traj_bernstein_degree": 3,
    "query_traj_use_stationary_mode": False,
    "query_traj_use_cv_mode": False,
    "query_traj_static_gate_enabled": False,
    "query_traj_static_gate_loss_weight": 0.0,
    "query_traj_static_gate_threshold": 0.5,
    "query_traj_derivative_routing_enabled": False,
    "query_traj_derivative_routing_hidden_dim": 0,
    "query_traj_teacher_forcing_enabled": False,
    "query_traj_teacher_forcing_mix_enabled": False,
    "query_traj_teacher_forcing_gt_ratio": 1.0,
    "query_traj_teacher_forcing_schedule_iters": (),
    "query_traj_teacher_forcing_schedule_gt_ratios": (),
    "query_traj_pred_target_enabled": False,
    "query_traj_anchor_refine_enabled": False,
    "query_traj_xy_refine_enabled": False,
    "query_traj_xy_refine_loss_weight": 0.0,
    "query_traj_xy_refine_num_layers": 2,
    "query_traj_xy_refine_hidden_dim": 0,
    "query_traj_endpoint_conditioning": False,
    "query_endpoint_loss_weight": 0.0,
    "query_traj_semantic_routing_enabled": False,
    "query_traj_rule_turn_family_enabled": False,
    "query_traj_rule_based_mode_enabled": False,
    "query_traj_rule_turn_threshold_deg": 10.0,
    "query_traj_static_threshold_m": 0.8,
    "query_traj_cv_error_threshold_m": 0.5,
    "query_traj_mode_cls_loss_weight": 0.0,
    "query_traj_mode_cls_moving_class_weight": 1.0,
    "query_traj_mode_infer_policy": "argmax",
    "query_traj_moving_reweight_enabled": False,
    "query_traj_moving_threshold_m": 0.5,
    "query_traj_moving_weight": 5.0,
    "query_traj_static_weight": 1.0,
    "query_traj_teacher_forcing": False,
    "query_traj_teacher_forcing_until_iter": 0,
    "query_center_match_cost_weight": 0.0,
    "query_temporal_offset_match_cost_weight": 0.0,
    "query_center_match_loss_type": "l1",
    "query_matched_gmo_bce_occ_size": (128, 128, 10),
    "query_num_gaussians": 1,
    "query_multi_gaussian_offset_max_m": (6.0, 6.0, 2.0),
    "query_multi_gaussian_sigma_min_m": (0.15, 0.15, 0.10),
    "query_multi_gaussian_sigma_max_m": (4.0, 4.0, 1.5),
    "query_multi_gaussian_sigma_reg_loss_weight": 0.0,
    "query_multi_gaussian_sigma_reg_log_eps": 1e-6,
    "query_multi_gaussian_pair_chunk": 8,
    "query_multi_gaussian_weight_mode": "softmax",
    "query_multi_gaussian_softplus_bias_init": -2.0,
    "query_multi_gaussian_weight_reg_loss_weight": 1e-3,
    "query_multi_gaussian_weight_reg_target_sum": 1.0,
    "query_embed_dim": 256,
    "query_num_queries": 100,
    "query_transformer_num_layers": 1,
    "query_id_reinject_scale": 0.0,
    "query_ca_kv_identity_init": False,
    "query_ca_attn_tau": 1.0,
    "query_decor_loss_weight": 0.0,
    "query_attn_overlap_loss_weight": 0.0,
    "use_query_attn_bbox_loss": False,
    "query_attn_bbox_loss_weight": 1.0,
    "query_attn_bbox_unmatched_weight": 0.25,
    "query_attn_bbox_other_weight": 0.0,
    "query_attn_bbox_eps": 1e-6,
    "query_attn_bbox_camera_reduce_mode": "camera_aggregated_first",
    "query_attn_bbox_unmatched_mode": "inverse_union",
    "query_attn_bbox_other_mode": "union",
    "query_attn_match_cost_weight": 0.0,
    "query_attn_match_metric": "soft_iou",
    "query_attn_match_pred_norm": "amax",
    "query_attn_match_eps": 1e-6,
    "use_query_attn_cam_gaussian_score": False,
    "query_attn_cam_frame_mode": "overlap_only",
    "query_attn_cam_target_mode": "prob",
    "query_attn_cam_metric": "kl",
    "query_attn_cam_camera_reduce_mode": "camera_aggregated_first",
    "query_attn_cam_eps": 1e-6,
    "query_attn_cam_gaussian_truncate_sigma": 3.0,
    "query_attn_cam_target_binary_threshold": 0.5,
    "query_attn_cam_score_norm_mode": "exp_neg",
    "query_attn_softargmax_tau": 1.0,
    "query_depth_loss_weight": 1.0,
    "query_depth_label_smoothing": 0.0,
    "query_inst_depth_num_bins": 64,
    "query_inst_depth_range_mode": "dbound",
    "query_inst_depth_min": 0.0,
    "query_inst_depth_max": 0.0,
}

DEBUG_CFG_DEFAULTS = {
    "debug_query_vis_every": 0,
    "debug_instance_img_vis_every": 0,
    "debug_query_cam_gaussian_vis_enabled": False,
    "debug_query_cam_gaussian_vis_every": 0,
    "debug_gt_alignment_vis_every": 0,
    "debug_query_inst_depth_lift_vis_every": 0,
    "debug_query_attn_softargmax_vis_every": 0,
}

VISUALIZATION_CFG_DEFAULTS = {
    "debug_query_vis_dir": "./work_dirs/query_debug_vis",
    "debug_query_center_marker_radius": 3,
    "debug_query_confidence_vis_threshold": 0.5,
    "debug_query_objectness_vis_threshold": 0.5,
    "debug_query_gaussian_vis_mode": "ellipse",
    "debug_query_gaussian_prob_threshold": 0.5,
    "debug_query_gaussian_prob_alpha_scale": 4.0,
    "debug_query_score_topk": 50,
    "debug_query_score_threshold": 0.5,
    "debug_query_score_iou_weight": 0.5,
    "debug_query_score_cls_weight": 0.5,
    "debug_query_score_cam_attn_weight": 0.0,
    "debug_query_distance_nms_radius_m": 3.0,
    "debug_instance_img_vis_dir": "./work_dirs/instance_img_debug_vis",
    "debug_instance_img_vis_max_frames": 3,
    "debug_instance_img_vis_max_instances": 24,
    "debug_query_cam_gaussian_vis_dir": "./work_dirs/query_cam_gaussian_vis",
    "debug_query_cam_gaussian_vis_max_queries": 50,
    "debug_query_cam_gaussian_vis_max_frames": 2,
    "debug_query_cam_gaussian_vis_gt_overlay_enabled": False,
    "debug_query_cam_gaussian_vis_topk_matched": 0,
    "debug_gt_alignment_vis_dir": "./work_dirs/gt_alignment_vis",
    "debug_gt_alignment_vis_max_frames": 7,
    "debug_query_inst_depth_lift_vis_dir": "./work_dirs/query_inst_depth_lift_vis",
    "debug_query_inst_depth_lift_vis_max_frames": 3,
    "debug_query_inst_depth_lift_vis_max_cams": 2,
    "debug_query_inst_depth_lift_vis_max_instances": 16,
    "debug_query_attn_softargmax_vis_dir": "./work_dirs/query_attn_softargmax_vis",
    "debug_query_attn_softargmax_vis_max_frames": 2,
    "debug_query_attn_softargmax_vis_max_cams": 3,
    "debug_query_attn_softargmax_vis_max_queries": 16,
    "query_attn_vis_dir": "./work_dirs/query_attn_vis",
}


def _merge_cfg(defaults, cfg):
    merged = dict(defaults)
    if cfg:
        merged.update(cfg)
    return merged


def apply_model_cfg(self, cfg):
    cfg = _merge_cfg(MODEL_CFG_DEFAULTS, cfg)

    self.gmo_ids = tuple(int(v) for v in cfg["gmo_ids"])
    self.use_segmentation_as_query_gt = bool(cfg["use_segmentation_as_query_gt"])
    self.use_gmo_bce_loss = bool(cfg["use_gmo_bce_loss"])
    self.query_gmo_loss_type = str(cfg["query_gmo_loss_type"]).lower()
    self.query_gmo_focal_gamma = float(cfg["query_gmo_focal_gamma"])
    self.query_gmo_focal_alpha = float(cfg["query_gmo_focal_alpha"])
    self.query_gmo_dice_loss_weight = float(cfg["query_gmo_dice_loss_weight"])
    self.query_gmo_tversky_alpha = float(cfg["query_gmo_tversky_alpha"])
    self.query_gmo_tversky_beta = float(cfg["query_gmo_tversky_beta"])
    self.use_query_gmo_dice_loss = bool(cfg["use_query_gmo_dice_loss"])
    self.use_query_inst_center_match_loss = bool(cfg["use_query_inst_center_match_loss"])
    self.use_query_dt_loss = bool(cfg["use_query_dt_loss"])
    self.use_query_gt2p_instance_labeled_loss = bool(cfg["use_query_gt2p_instance_labeled_loss"])
    self.query_gt2p_instance_labeled_loss_weight = float(cfg["query_gt2p_instance_labeled_loss_weight"])
    self.query_gt2p_instance_labeled_balance_weight = float(cfg["query_gt2p_instance_labeled_balance_weight"])
    self.query_gt2p_instance_labeled_tau = float(cfg["query_gt2p_instance_labeled_tau"])
    self.query_gt2p_cooldown_enabled = bool(cfg["query_gt2p_cooldown_enabled"])
    self.query_gt2p_cooldown_start_iter = int(cfg["query_gt2p_cooldown_start_iter"])
    self.query_gt2p_cooldown_iters = int(cfg["query_gt2p_cooldown_iters"])
    self.query_gt2p_cooldown_min_scale = float(cfg["query_gt2p_cooldown_min_scale"])
    self.query_gt2p_instance_labeled_assign_sigma_xyz = tuple(
        float(v) for v in cfg["query_gt2p_instance_labeled_assign_sigma_xyz"]
    )
    self.query_gt2p_instance_labeled_sigma_policy = str(
        cfg["query_gt2p_instance_labeled_sigma_policy"]
    ).lower()
    self.query_require_history_all_valid = bool(cfg["query_require_history_all_valid"])
    self.center_only_mode = bool(cfg["center_only_mode"])
    self.query_feat_cosine_threshold = float(cfg["query_feat_cosine_threshold"])
    self.query_center_distance_threshold_m = float(cfg["query_center_distance_threshold_m"])
    self.query_center_loss_detach_query_feat = bool(cfg["query_center_loss_detach_query_feat"])
    self.query_feat_unmatched_neg_loss_weight = float(cfg["query_feat_unmatched_neg_loss_weight"])
    self.query_feat_unmatched_neg_margin = float(cfg["query_feat_unmatched_neg_margin"])
    self.query_bev_pool_fixed_sigma_xyz = tuple(float(v) for v in cfg["query_bev_pool_fixed_sigma_xyz"])
    self.query_num_classes = int(cfg["query_num_classes"])
    self.query_bg_class = 0

    query_class_ids = cfg["query_class_ids"]
    if query_class_ids is None:
        query_class_ids = tuple(range(self.query_num_classes))
    else:
        query_class_ids = tuple(int(v) for v in query_class_ids)
    self.query_class_ids = query_class_ids

    query_class_names = cfg["query_class_names"]
    if query_class_names is None:
        query_class_names = ["background"] + [
            f"class_{int(raw_id)}" for raw_id in self.query_class_ids[1:]
        ]
    else:
        query_class_names = list(query_class_names)
        if len(query_class_names) == (self.query_num_classes - 1):
            query_class_names = ["background"] + query_class_names
    self.query_class_names = tuple(str(v) for v in query_class_names)
    self.strict_query_class_id_validation = bool(cfg["strict_query_class_id_validation"])

    self.query_cls_loss_weight = float(cfg["query_cls_loss_weight"])
    self.query_match_feature_source = str(cfg["query_match_feature_source"])
    self.query_soft_assign_temp = float(cfg["query_soft_assign_temp"])
    self.query_soft_assign_cost_weight = float(cfg["query_soft_assign_cost_weight"])
    self.query_sim_match_cost_weight = float(cfg["query_sim_match_cost_weight"])
    self.query_cls_match_cost_weight = float(cfg["query_cls_match_cost_weight"])
    self.query_bev_dice_match_cost_weight = float(cfg["query_bev_dice_match_cost_weight"])
    self.query_center_routed_loss_weight = float(cfg["query_center_routed_loss_weight"])
    self.query_traj_matched_only = bool(cfg["query_traj_matched_only"])
    self.query_traj_loss_weight = float(cfg["query_traj_loss_weight"])
    self.query_traj_loss_type = str(cfg["query_traj_loss_type"]).lower()
    self.query_traj_residual_max_m = tuple(float(v) for v in cfg["query_traj_residual_max_m"])
    self.query_traj_prior_detach = bool(cfg["query_traj_prior_detach"])
    self.query_traj_num_modes = int(cfg["query_traj_num_modes"])
    self.query_traj_decoder_type = str(cfg["query_traj_decoder_type"]).lower()
    self.query_traj_bernstein_degree = int(cfg["query_traj_bernstein_degree"])
    self.query_traj_use_stationary_mode = bool(cfg["query_traj_use_stationary_mode"])
    self.query_traj_use_cv_mode = bool(cfg["query_traj_use_cv_mode"])
    self.query_traj_static_gate_enabled = bool(cfg["query_traj_static_gate_enabled"])
    self.query_traj_static_gate_loss_weight = float(cfg["query_traj_static_gate_loss_weight"])
    self.query_traj_static_gate_threshold = float(cfg["query_traj_static_gate_threshold"])
    self.query_traj_derivative_routing_enabled = bool(cfg["query_traj_derivative_routing_enabled"])
    self.query_traj_derivative_routing_hidden_dim = int(cfg["query_traj_derivative_routing_hidden_dim"])
    self.query_traj_teacher_forcing_enabled = bool(cfg["query_traj_teacher_forcing_enabled"])
    self.query_traj_teacher_forcing_mix_enabled = bool(cfg["query_traj_teacher_forcing_mix_enabled"])
    self.query_traj_teacher_forcing_gt_ratio = float(cfg["query_traj_teacher_forcing_gt_ratio"])
    self.query_traj_teacher_forcing_schedule_iters = tuple(
        int(v) for v in cfg["query_traj_teacher_forcing_schedule_iters"]
    )
    self.query_traj_teacher_forcing_schedule_gt_ratios = tuple(
        float(v) for v in cfg["query_traj_teacher_forcing_schedule_gt_ratios"]
    )
    self.query_traj_pred_target_enabled = bool(cfg["query_traj_pred_target_enabled"])
    self.query_traj_anchor_refine_enabled = bool(cfg["query_traj_anchor_refine_enabled"])
    self.query_traj_xy_refine_enabled = bool(cfg["query_traj_xy_refine_enabled"])
    self.query_traj_xy_refine_loss_weight = float(cfg["query_traj_xy_refine_loss_weight"])
    self.query_traj_xy_refine_num_layers = int(cfg["query_traj_xy_refine_num_layers"])
    self.query_traj_xy_refine_hidden_dim = int(cfg["query_traj_xy_refine_hidden_dim"])
    self.query_traj_endpoint_conditioning = bool(cfg["query_traj_endpoint_conditioning"])
    self.query_endpoint_loss_weight = float(cfg["query_endpoint_loss_weight"])
    self.query_traj_semantic_routing_enabled = bool(cfg["query_traj_semantic_routing_enabled"])
    self.query_traj_rule_turn_family_enabled = bool(cfg["query_traj_rule_turn_family_enabled"])
    self.query_traj_rule_based_mode_enabled = bool(cfg["query_traj_rule_based_mode_enabled"])
    self.query_traj_rule_turn_threshold_deg = float(cfg["query_traj_rule_turn_threshold_deg"])
    self.query_traj_static_threshold_m = float(cfg["query_traj_static_threshold_m"])
    self.query_traj_cv_error_threshold_m = float(cfg["query_traj_cv_error_threshold_m"])
    self.query_traj_mode_cls_loss_weight = float(cfg["query_traj_mode_cls_loss_weight"])
    self.query_traj_mode_cls_moving_class_weight = float(cfg["query_traj_mode_cls_moving_class_weight"])
    self.query_traj_mode_infer_policy = str(cfg["query_traj_mode_infer_policy"]).lower()
    self.query_traj_moving_reweight_enabled = bool(cfg["query_traj_moving_reweight_enabled"])
    self.query_traj_moving_threshold_m = float(cfg["query_traj_moving_threshold_m"])
    self.query_traj_moving_weight = float(cfg["query_traj_moving_weight"])
    self.query_traj_static_weight = float(cfg["query_traj_static_weight"])
    self.query_traj_teacher_forcing = bool(cfg["query_traj_teacher_forcing"])
    self.query_traj_teacher_forcing_until_iter = int(cfg["query_traj_teacher_forcing_until_iter"])
    self.query_center_match_cost_weight = float(cfg["query_center_match_cost_weight"])
    self.query_temporal_offset_match_cost_weight = float(cfg["query_temporal_offset_match_cost_weight"])
    self.query_center_match_loss_type = str(cfg["query_center_match_loss_type"]).lower()
    self.query_matched_gmo_bce_occ_size = tuple(int(v) for v in cfg["query_matched_gmo_bce_occ_size"])
    self.query_num_gaussians = int(cfg["query_num_gaussians"])
    self.query_multi_gaussian_offset_max_m = tuple(float(v) for v in cfg["query_multi_gaussian_offset_max_m"])
    self.query_multi_gaussian_sigma_min_m = tuple(float(v) for v in cfg["query_multi_gaussian_sigma_min_m"])
    self.query_multi_gaussian_sigma_max_m = tuple(float(v) for v in cfg["query_multi_gaussian_sigma_max_m"])
    self.query_multi_gaussian_sigma_reg_loss_weight = float(cfg["query_multi_gaussian_sigma_reg_loss_weight"])
    self.query_multi_gaussian_sigma_reg_log_eps = float(cfg["query_multi_gaussian_sigma_reg_log_eps"])
    self.query_multi_gaussian_pair_chunk = max(1, int(cfg["query_multi_gaussian_pair_chunk"]))
    self.query_multi_gaussian_weight_mode = str(cfg["query_multi_gaussian_weight_mode"]).lower()
    self.query_multi_gaussian_softplus_bias_init = float(cfg["query_multi_gaussian_softplus_bias_init"])
    self.query_multi_gaussian_weight_reg_loss_weight = float(cfg["query_multi_gaussian_weight_reg_loss_weight"])
    self.query_multi_gaussian_weight_reg_target_sum = float(cfg["query_multi_gaussian_weight_reg_target_sum"])
    self.query_embed_dim = int(cfg["query_embed_dim"])
    self.query_num_queries = int(cfg["query_num_queries"])
    self.query_transformer_num_layers = int(cfg["query_transformer_num_layers"])
    self.query_id_reinject_scale = float(cfg["query_id_reinject_scale"])
    self.query_ca_kv_identity_init = bool(cfg["query_ca_kv_identity_init"])
    self.query_ca_attn_tau = float(cfg["query_ca_attn_tau"])
    self.query_decor_loss_weight = float(cfg["query_decor_loss_weight"])
    self.query_attn_overlap_loss_weight = float(cfg["query_attn_overlap_loss_weight"])
    self.use_query_attn_bbox_loss = bool(cfg["use_query_attn_bbox_loss"])
    self.query_attn_bbox_loss_weight = float(cfg["query_attn_bbox_loss_weight"])
    self.query_attn_bbox_unmatched_weight = float(cfg["query_attn_bbox_unmatched_weight"])
    self.query_attn_bbox_other_weight = float(cfg["query_attn_bbox_other_weight"])
    self.query_attn_bbox_eps = float(cfg["query_attn_bbox_eps"])
    self.query_attn_bbox_camera_reduce_mode = str(cfg["query_attn_bbox_camera_reduce_mode"]).lower()
    self.query_attn_bbox_unmatched_mode = str(cfg["query_attn_bbox_unmatched_mode"]).lower()
    self.query_attn_bbox_other_mode = str(cfg["query_attn_bbox_other_mode"]).lower()
    self.query_attn_match_cost_weight = float(cfg["query_attn_match_cost_weight"])
    self.query_attn_match_metric = str(cfg["query_attn_match_metric"]).lower()
    self.query_attn_match_pred_norm = str(cfg["query_attn_match_pred_norm"]).lower()
    self.query_attn_match_eps = float(cfg["query_attn_match_eps"])
    self.use_query_attn_cam_gaussian_score = bool(cfg["use_query_attn_cam_gaussian_score"])
    self.query_attn_cam_frame_mode = str(cfg["query_attn_cam_frame_mode"]).lower()
    self.query_attn_cam_target_mode = str(cfg["query_attn_cam_target_mode"]).lower()
    self.query_attn_cam_metric = str(cfg["query_attn_cam_metric"]).lower()
    self.query_attn_cam_camera_reduce_mode = str(cfg["query_attn_cam_camera_reduce_mode"]).lower()
    self.query_attn_cam_eps = float(cfg["query_attn_cam_eps"])
    self.query_attn_cam_gaussian_truncate_sigma = float(cfg["query_attn_cam_gaussian_truncate_sigma"])
    self.query_attn_cam_target_binary_threshold = float(cfg["query_attn_cam_target_binary_threshold"])
    self.query_attn_cam_score_norm_mode = str(cfg["query_attn_cam_score_norm_mode"]).lower()
    self.query_attn_softargmax_tau = float(cfg["query_attn_softargmax_tau"])
    self.query_depth_loss_weight = float(cfg["query_depth_loss_weight"])
    self.query_depth_label_smoothing = float(cfg["query_depth_label_smoothing"])
    self.query_inst_depth_num_bins = int(cfg["query_inst_depth_num_bins"])
    self.query_inst_depth_range_mode = str(cfg["query_inst_depth_range_mode"]).lower()
    self.query_inst_depth_min = float(cfg["query_inst_depth_min"])
    self.query_inst_depth_max = float(cfg["query_inst_depth_max"])

    if self.center_only_mode:
        self.use_gmo_bce_loss = False
        self.use_query_gmo_dice_loss = False

    query_cls_loss_class_weights = cfg["query_cls_loss_class_weights"]
    if query_cls_loss_class_weights is None:
        query_cls_loss_class_weights = [1.0] * int(self.query_num_classes)
    query_cls_loss_class_weights = [float(v) for v in query_cls_loss_class_weights]
    return cfg, query_cls_loss_class_weights


def apply_debug_cfg(self, cfg):
    cfg = _merge_cfg(DEBUG_CFG_DEFAULTS, cfg)

    self.debug_query_vis_every = int(cfg["debug_query_vis_every"])
    self.debug_instance_img_vis_every = int(cfg["debug_instance_img_vis_every"])
    self.debug_query_cam_gaussian_vis_enabled = bool(cfg["debug_query_cam_gaussian_vis_enabled"])
    self.debug_query_cam_gaussian_vis_every = int(cfg["debug_query_cam_gaussian_vis_every"])
    self.debug_gt_alignment_vis_every = int(cfg["debug_gt_alignment_vis_every"])
    self.debug_query_inst_depth_lift_vis_every = int(cfg["debug_query_inst_depth_lift_vis_every"])
    self.debug_query_attn_softargmax_vis_every = int(cfg["debug_query_attn_softargmax_vis_every"])

    return cfg


def apply_visualization_cfg(self, cfg):
    cfg = _merge_cfg(VISUALIZATION_CFG_DEFAULTS, cfg)

    self.debug_query_vis_dir = str(cfg["debug_query_vis_dir"])
    self.debug_query_center_marker_radius = max(0, int(cfg["debug_query_center_marker_radius"]))
    self.debug_query_confidence_vis_threshold = float(cfg["debug_query_confidence_vis_threshold"])
    self.debug_query_objectness_vis_threshold = float(cfg["debug_query_objectness_vis_threshold"])
    self.debug_query_gaussian_vis_mode = str(cfg["debug_query_gaussian_vis_mode"]).lower()
    self.debug_query_gaussian_prob_threshold = float(cfg["debug_query_gaussian_prob_threshold"])
    self.debug_query_gaussian_prob_alpha_scale = float(cfg["debug_query_gaussian_prob_alpha_scale"])
    self.debug_query_score_topk = max(0, int(cfg["debug_query_score_topk"]))
    self.debug_query_score_threshold = float(cfg["debug_query_score_threshold"])
    self.debug_query_score_iou_weight = float(cfg["debug_query_score_iou_weight"])
    self.debug_query_score_cls_weight = float(cfg["debug_query_score_cls_weight"])
    self.debug_query_score_cam_attn_weight = float(cfg["debug_query_score_cam_attn_weight"])
    self.debug_query_distance_nms_radius_m = max(0.0, float(cfg["debug_query_distance_nms_radius_m"]))
    self.debug_instance_img_vis_dir = str(cfg["debug_instance_img_vis_dir"])
    self.debug_instance_img_vis_max_frames = max(1, int(cfg["debug_instance_img_vis_max_frames"]))
    self.debug_instance_img_vis_max_instances = max(1, int(cfg["debug_instance_img_vis_max_instances"]))
    self.debug_query_cam_gaussian_vis_dir = str(cfg["debug_query_cam_gaussian_vis_dir"])
    self.debug_query_cam_gaussian_vis_max_queries = max(1, int(cfg["debug_query_cam_gaussian_vis_max_queries"]))
    self.debug_query_cam_gaussian_vis_max_frames = max(1, int(cfg["debug_query_cam_gaussian_vis_max_frames"]))
    self.debug_query_cam_gaussian_vis_gt_overlay_enabled = bool(
        cfg["debug_query_cam_gaussian_vis_gt_overlay_enabled"]
    )
    self.debug_query_cam_gaussian_vis_topk_matched = int(cfg["debug_query_cam_gaussian_vis_topk_matched"])
    self.debug_gt_alignment_vis_dir = str(cfg["debug_gt_alignment_vis_dir"])
    self.debug_gt_alignment_vis_max_frames = max(1, int(cfg["debug_gt_alignment_vis_max_frames"]))
    self.debug_query_inst_depth_lift_vis_dir = str(cfg["debug_query_inst_depth_lift_vis_dir"])
    self.debug_query_inst_depth_lift_vis_max_frames = max(1, int(cfg["debug_query_inst_depth_lift_vis_max_frames"]))
    self.debug_query_inst_depth_lift_vis_max_cams = max(1, int(cfg["debug_query_inst_depth_lift_vis_max_cams"]))
    self.debug_query_inst_depth_lift_vis_max_instances = max(1, int(cfg["debug_query_inst_depth_lift_vis_max_instances"]))
    self.debug_query_attn_softargmax_vis_dir = str(cfg["debug_query_attn_softargmax_vis_dir"])
    self.debug_query_attn_softargmax_vis_max_frames = max(1, int(cfg["debug_query_attn_softargmax_vis_max_frames"]))
    self.debug_query_attn_softargmax_vis_max_cams = max(1, int(cfg["debug_query_attn_softargmax_vis_max_cams"]))
    self.debug_query_attn_softargmax_vis_max_queries = max(1, int(cfg["debug_query_attn_softargmax_vis_max_queries"]))
    self.query_attn_vis_dir = str(cfg["query_attn_vis_dir"])

    return cfg
