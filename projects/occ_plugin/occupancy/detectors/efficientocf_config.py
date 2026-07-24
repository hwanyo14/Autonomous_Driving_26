MODEL_CFG_DEFAULTS = {
    "gmo_ids": (2, 3, 4, 5, 6, 7, 9, 10),
    "use_segmentation_as_query_gt": False,
    "use_gmo_bce_loss": False,
    "query_gmo_loss_type": "balanced_bce",
    "query_gmo_focal_gamma": 2.0,
    "query_gmo_focal_alpha": 0.25,
    "query_gmo_loss_weight": 0.1,   # focal/bce GMO term weight (formerly hardcoded 0.1 in efficientocf.py)
    # focal(occ) GT 혼합: pair focal = inst3d_w*focal(gt_occ_inst3d) + bbox_w*focal(AABB bbox v3캐시).
    # bbox_w>0이면 pipeline에 load_gt_bbox_aabb=True + Collect3D 'gt_bbox_aabb' 필요. dice는 항상 inst3d.
    "query_gmo_focal_inst3d_weight": 1.0,
    "query_gmo_focal_bbox_weight": 0.0,
    # dice(tversky) GT 혼합: pair dice = inst3d_w*tversky(inst3d) + bbox_w*tversky(AABB bbox).
    # 의도: FP-heavy tversky(α)가 'box 밖' 확장을 강벌 — focal-bbox 실험의 비관용FP 문제 대응.
    "query_gmo_dice_inst3d_weight": 1.0,
    "query_gmo_dice_bbox_weight": 0.0,
    "query_gmo_dice_loss_weight": 0.5,
    "query_gmo_dice_3d": False,   # True면 dice/Tversky를 z collapse 없이 3D로 (z 과확장 억제)
    # All-query scene supervision on present+future frames. Continuous foreground
    # confidence gates each query; no threshold, Hungarian selection, NMS, or top-k.
    "use_query_scene_asset_bce_loss": False,
    "query_scene_asset_bce_loss_weight": 0.0,
    "query_scene_asset_bce_pos_weight": 1.0,
    "query_scene_asset_bce_neg_weight": 1.0,
    "query_scene_asset_bce_eps": 1e-6,
    # Matched-pair GMO focal/Dice hard mining. geometry는 크기/거리,
    # loss_topk는 pair raw Dice 상위 비율을 선택한다.
    "query_gmo_hard_pair_enabled": False,
    "query_gmo_hard_pair_mode": "geometry",
    "query_gmo_hard_topk_ratio": 0.3,
    "query_gmo_hard_easy_weight": 0.0,
    "query_gmo_hard_min_pairs": 1,
    "query_gmo_hard_large_min_m": 6.0,
    "query_gmo_hard_far_min_m": 30.0,
    # 학습 전용: matched Gaussian-center/GT-BEV의 trace-normalized XY covariance 감독.
    "use_query_gmo_shape_loss": False,
    "query_gmo_shape_loss_weight": 0.0,
    "query_gmo_shape_loss_type": "covariance",
    "query_gmo_shape_min_bev_voxels": 4,
    "query_gmo_shape_min_gt_axis_ratio": 1.0,
    "query_gmo_shape_pred_weight_mode": "equal",
    # Diagnostic fine-tuning: optimize only the Gaussian offset output head and
    # expose only the GMO shape loss to the runner.
    "query_offset_only_finetune": False,
    "query_shape_only_finetune": False,
    "query_gmo_tversky_alpha": 0.7,
    "query_gmo_tversky_beta": 0.3,
    "query_gmo_tversky_size_enabled": False,
    "query_gmo_tversky_size_start_m": 6.0,
    "query_gmo_tversky_size_end_m": 10.0,
    "query_gmo_tversky_size_alpha_end": 0.4,
    "query_gmo_soft_gt_enabled": False,
    "query_gmo_soft_gt_sigma_vox": 2.5,
    "query_gmo_soft_gt_truncate_sigma": 3.0,
    "use_query_gmo_dice_loss": True,
    "use_query_inst_center_match_loss": True,
    # Restrict matched-pair GMO(occupancy) + center losses to the receptive-field
    # (past+present) frames; leave the future horizon to the trajectory loss only.
    "query_matched_loss_history_only": False,
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
    # When True the semantic cls head is collapsed to binary {0=bg, 1=fg}:
    # every movable raw id is mapped to compact id 1 (see efficientocf.py raw->compact map),
    # and the head is built with num_query_classes=2. Keep query_class_ids as the full raw
    # id list (GT loading/validation still needs them). Default False = original N-way cls.
    "query_cls_binary_fg": False,
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
    "query_match_center_gate_radius_m": 0.0,
    "query_recruit_max_radius_m": 0.0,
    "query_recruit_loss_weight": 0.0,
    "query_center_match_loss_type": "l1",
    "query_matched_gmo_bce_occ_size": (128, 128, 10),
    "query_matched_gmo_sigma_floor_vox": 0.5,
    "query_num_gaussians": 1,
    "query_multi_gaussian_offset_max_m": (6.0, 6.0, 2.0),
    "query_multi_gaussian_sigma_min_m": (0.15, 0.15, 0.10),
    "query_multi_gaussian_sigma_max_m": (4.0, 4.0, 1.5),
    "query_multi_gaussian_sigma_reg_loss_weight": 0.0,
    "query_multi_gaussian_sigma_reg_log_eps": 1e-6,
    # grouped voxelizer 루프 보폭(메모리/속도만, 결과 불변). 흩어진 pair를 묶을수록 bbox≈전체격자라
    # 메모리·낭비연산↑. GPU 실측: 128격자에서 chunk=2가 속도 sweet spot(최速)+메모리 적당. → 기본 2.
    "query_multi_gaussian_pair_chunk": 2,
    "query_multi_gaussian_weight_mode": "softmax",
    "query_multi_gaussian_softplus_bias_init": -2.0,
    "query_multi_gaussian_weight_reg_loss_weight": 1e-3,
    "query_multi_gaussian_weight_reg_target_sum": 1.0,
    # occupancy 합성 방식: 'poisson'(기존) p=1-exp(-Σ w·G) / 'union'(GUIDE) p=1-Π(1-w·G).
    # 'union'은 weight_mode='sigmoid'(α∈[0,1] opacity)와 함께 써야 함.
    "query_multi_gaussian_occ_combine_mode": "poisson",
    # --- Per-query spread forcing (large-object coverage) ---
    # matched query의 OFFSET-only 분산(가우시안 중심 분포, σ 제외)을 GT half-extent의
    # target_frac까지 끌어올리는 one-sided(부족할 때만) 정칙항. σ-blob 대신 offset을
    # 움직이게 강제해 collapsed-to-car-size 평형을 깸. GT extent = voxel-AABB(max-over-frames).
    "query_scale_spread_loss_weight": 0.0,
    "query_scale_spread_target_frac": 0.5,
    # spread 메트릭의 가우시안 가중: 'raw'(기존, 정규화 w 그대로 — 낮은 w '유령' 가우시안이
    #   메트릭을 채우는 weight-loophole 있음, 2026-07-02 dbg 실측상 사실상 무압력) /
    #   'visible'(w_eff=1-exp(-w), poisson 단일 가우시안 피크 점유 기여 — 점유에 실제
    #   나타나는 질량만 집계, target을 채우려면 고-w 가우시안을 멀리 보내야 함).
    "query_scale_spread_weight_mode": "raw",
    # 정칙항 정규화: 'matched'(기존, matched 전체 mean — 위반자 소수면 1/K 희석) /
    #   'violators'(deficit>0 쿼리 수로 나눔 — 희소 대형 객체의 gradient 크기 보존).
    "query_scale_spread_norm_mode": "matched",
    # --- size note ("크기 쪽지", 2026-07-05) ---
    # gaussian head 입력에 [logσu, logσv, log d, log w_pred, log l_pred] 5성분을
    # zero-init 투영으로 additive 주입 (σ=attn 폭, d=center 거리, w/l=aux head 예측).
    # 전부 detach — head gradient가 attention/depth/aux로 역류 못 함. head 파라미터
    # 수가 바뀌므로(aux MLP+proj) 기존 checkpoint와 호환 안 됨 → from-scratch 전용.
    "query_size_note_enabled": False,
    # aux size head L1 (matched만, GT=annotation 원본 w,l — gt_instance_dims 플러밍 필요).
    "query_size_aux_loss_weight": 0.0,
    # 크기-비례 가중 상한: weight = clamp(max(w,l)/2m, 1, cap) — 버스(l~12m)는 cap배.
    "query_size_aux_big_weight_cap": 4.0,
    # eval occupancy를 학습과 정렬: True(기본)면 16개 mixture를 그대로 splat(union over queries,
    # occ_combine_mode 적용, yaw 살림). False면 16개를 평균낸 단일 ellipsoid로 찍음.
    # mixture 텐서가 없는 모델은 simple_test의 use_mix 가드가 자동으로 단일 ellipsoid로 fallback.
    "query_eval_occ_use_mixture": True,
    # ============================================================================
    # [Thresholds — 자주 바꾸는 값] 선택/metric 임계값. config 값=학습·추론 공통,
    #   추론에서만 바꾸려면 env. (둘 다 selection/metric 본 로직 — viz 아님)
    #   occ 점유(=metric) :  occ_score_threshold  (metric 이진화 + 2D occ-grid + 3D mixture3d)
    #                        env override: EOCF_EVAL_OCC_THR
    #   query(fg) 선택 컷  :  fg_score_threshold   (query 선택 + 2D 표시 + 3D 필터)
    #                        env override: EOCF_EVAL_FG_THR
    #
    # [Oracle eval] EOCF_EVAL_ORACLE_MATCH=1 → query score 선택을 끄고 GT에 Hungarian 매칭(학습 동일).
    #   구현: efficientocf.py `_eval_train_faithful_inst_match`. metric만 영향, 학습/loss 무관.
    # ============================================================================
    "occ_score_threshold": 0.5,
    "fg_score_threshold": 0.5,
    # fg score 합성 가중치: score = (w_iou*iou + w_cls*cls + w_cam*cam) / (w_iou+w_cls + w_cam)
    "fg_score_iou_weight": 0.5,
    "fg_score_cls_weight": 0.5,
    "fg_score_cam_attn_weight": 0.0,
    "fg_score_topk": 50,
    "fg_score_distance_nms_radius_m": 3.0,
    "query_embed_dim": 256,
    "query_num_queries": 100,
    "query_transformer_num_layers": 1,
    # Per-layer cross-attention KV 해상도 (coarse->fine 피라미드). None=기존 동작(전 layer가
    # 원본 해상도 KV 공유). 지정 시 len()이 query_transformer_num_layers와 일치해야 함.
    "query_transformer_kv_resolutions": None,
    "query_transformer_learnable_kv_downsample": False,
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
    # --- attention sigma-matching (폭 감독; 2026-07-02 실측 근거) ---
    # inside-mass는 총량만 봐서 attn 폭이 크기 무관 고정(σ 8~14px, 마스크는 1.7~6.3px)이 됨.
    # matched query의 attn 2차 모멘트(σ_u,σ_v)를 GT cam 마스크의 σ에 log-ratio 회귀.
    # 위치(1차 모멘트)는 loss에 미포함 → center 경로 무접촉. 0.0=off(기존 동작).
    "query_attn_sigma_match_loss_weight": 0.0,
    # warmup: 이 iter 전에는 계산 안 함 (center/attn이 자리 잡은 뒤 개입, 8GPU subset 기준 1 epoch≈500).
    "query_attn_sigma_match_start_iter": 0,
    # σ 추정이 노이즈인 초소형 마스크 제외 (px).
    "query_attn_sigma_match_min_mask_px": 4,
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
    # QueryDepthHead capacity (depth bin classifier MLP). Defaults reproduce the
    # original 2-layer, hidden=embed_dim head. hidden_mult widens hidden=embed_dim*mult.
    "query_depth_head_num_layers": 2,
    "query_depth_head_hidden_mult": 1,
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
    "debug_query_mixture3d_vis_every": 0,
}

VISUALIZATION_CFG_DEFAULTS = {
    "debug_query_vis_dir": "./work_dirs/query_debug_vis",
    "debug_query_center_marker_radius": 3,
    # (fg score 선택 임계값/가중치는 selection 로직이라 MODEL_CFG_DEFAULTS로 이동 — fg_score_* 참조)
    "debug_query_gaussian_vis_mode": "ellipse",
    "debug_query_gaussian_prob_alpha_scale": 4.0,
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
    "debug_query_mixture3d_vis_dir": "./work_dirs/query_mixture3d_vis",
    "debug_query_mixture3d_vis_max_queries": 50,
    "debug_query_mixture3d_vis_max_gt_points": 40000,
    # (deprecated) mixture3d occ threshold는 이제 occ_score_threshold로 통일됨 → 키 제거.
    "debug_query_mixture3d_vis_occ_max_voxels_per_query": 4000,
    # eval 전용 3D mixture vis 해상도. train 중 eval을 얹기 쉽게 기본은 lightweight 128x128x10.
    "debug_query_mixture3d_vis_eval_occ_size": (128, 128, 10),
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
    self.query_gmo_loss_weight = float(cfg["query_gmo_loss_weight"])
    self.query_gmo_focal_inst3d_weight = float(cfg["query_gmo_focal_inst3d_weight"])
    self.query_gmo_focal_bbox_weight = float(cfg["query_gmo_focal_bbox_weight"])
    self.query_gmo_dice_inst3d_weight = float(cfg["query_gmo_dice_inst3d_weight"])
    self.query_gmo_dice_bbox_weight = float(cfg["query_gmo_dice_bbox_weight"])
    self.query_gmo_dice_loss_weight = float(cfg["query_gmo_dice_loss_weight"])
    self.query_gmo_dice_3d = bool(cfg["query_gmo_dice_3d"])
    self.use_query_scene_asset_bce_loss = bool(cfg["use_query_scene_asset_bce_loss"])
    self.query_scene_asset_bce_loss_weight = float(cfg["query_scene_asset_bce_loss_weight"])
    self.query_scene_asset_bce_pos_weight = float(cfg["query_scene_asset_bce_pos_weight"])
    self.query_scene_asset_bce_neg_weight = float(cfg["query_scene_asset_bce_neg_weight"])
    self.query_scene_asset_bce_eps = float(cfg["query_scene_asset_bce_eps"])
    if min(
        self.query_scene_asset_bce_loss_weight,
        self.query_scene_asset_bce_pos_weight,
        self.query_scene_asset_bce_neg_weight,
    ) < 0.0:
        raise ValueError("query scene asset BCE weights must be >= 0")
    if self.query_scene_asset_bce_eps <= 0.0:
        raise ValueError("query_scene_asset_bce_eps must be > 0")
    self.query_gmo_hard_pair_enabled = bool(cfg["query_gmo_hard_pair_enabled"])
    self.query_gmo_hard_pair_mode = str(cfg["query_gmo_hard_pair_mode"]).lower()
    self.query_gmo_hard_topk_ratio = float(cfg["query_gmo_hard_topk_ratio"])
    self.query_gmo_hard_easy_weight = float(cfg["query_gmo_hard_easy_weight"])
    self.query_gmo_hard_min_pairs = int(cfg["query_gmo_hard_min_pairs"])
    self.query_gmo_hard_large_min_m = float(cfg["query_gmo_hard_large_min_m"])
    self.query_gmo_hard_far_min_m = float(cfg["query_gmo_hard_far_min_m"])
    if self.query_gmo_hard_pair_mode not in ("geometry", "loss_topk"):
        raise ValueError("query_gmo_hard_pair_mode must be 'geometry' or 'loss_topk'")
    if not 0.0 < self.query_gmo_hard_topk_ratio <= 1.0:
        raise ValueError("query_gmo_hard_topk_ratio must be in (0, 1]")
    if not 0.0 <= self.query_gmo_hard_easy_weight <= 1.0:
        raise ValueError("query_gmo_hard_easy_weight must be in [0, 1]")
    if self.query_gmo_hard_min_pairs < 1:
        raise ValueError("query_gmo_hard_min_pairs must be >= 1")
    if self.query_gmo_hard_large_min_m <= 0.0:
        raise ValueError("query_gmo_hard_large_min_m must be > 0")
    if self.query_gmo_hard_far_min_m <= 0.0:
        raise ValueError("query_gmo_hard_far_min_m must be > 0")
    self.use_query_gmo_shape_loss = bool(cfg["use_query_gmo_shape_loss"])
    self.query_gmo_shape_loss_weight = float(cfg["query_gmo_shape_loss_weight"])
    self.query_gmo_shape_loss_type = str(cfg["query_gmo_shape_loss_type"]).lower()
    self.query_gmo_shape_min_bev_voxels = int(cfg["query_gmo_shape_min_bev_voxels"])
    self.query_gmo_shape_min_gt_axis_ratio = float(cfg["query_gmo_shape_min_gt_axis_ratio"])
    self.query_gmo_shape_pred_weight_mode = str(cfg["query_gmo_shape_pred_weight_mode"]).lower()
    self.query_offset_only_finetune = bool(cfg["query_offset_only_finetune"])
    self.query_shape_only_finetune = bool(cfg["query_shape_only_finetune"])
    if self.query_gmo_shape_loss_weight < 0.0:
        raise ValueError("query_gmo_shape_loss_weight must be >= 0")
    if self.query_gmo_shape_min_bev_voxels < 2:
        raise ValueError("query_gmo_shape_min_bev_voxels must be >= 2")
    if self.query_gmo_shape_loss_type not in ("covariance", "direction", "render_direction"):
        raise ValueError(
            "query_gmo_shape_loss_type must be 'covariance', 'direction', or 'render_direction'"
        )
    if self.query_gmo_shape_min_gt_axis_ratio < 1.0:
        raise ValueError("query_gmo_shape_min_gt_axis_ratio must be >= 1")
    if self.query_gmo_shape_pred_weight_mode not in ("equal", "detached_opacity"):
        raise ValueError(
            "query_gmo_shape_pred_weight_mode must be 'equal' or 'detached_opacity'"
        )
    self.query_gmo_tversky_alpha = float(cfg["query_gmo_tversky_alpha"])
    self.query_gmo_tversky_beta = float(cfg["query_gmo_tversky_beta"])
    self.query_gmo_tversky_size_enabled = bool(cfg["query_gmo_tversky_size_enabled"])
    self.query_gmo_tversky_size_start_m = float(cfg["query_gmo_tversky_size_start_m"])
    self.query_gmo_tversky_size_end_m = float(cfg["query_gmo_tversky_size_end_m"])
    self.query_gmo_tversky_size_alpha_end = float(cfg["query_gmo_tversky_size_alpha_end"])
    if self.query_gmo_tversky_size_enabled:
        if self.query_gmo_tversky_size_end_m <= self.query_gmo_tversky_size_start_m:
            raise ValueError("query_gmo_tversky_size_end_m must be greater than start_m")
        if not 0.0 <= self.query_gmo_tversky_size_alpha_end <= 1.0:
            raise ValueError("query_gmo_tversky_size_alpha_end must be in [0, 1]")
    self.query_gmo_soft_gt_enabled = bool(cfg["query_gmo_soft_gt_enabled"])
    self.query_gmo_soft_gt_sigma_vox = float(cfg["query_gmo_soft_gt_sigma_vox"])
    self.query_gmo_soft_gt_truncate_sigma = float(cfg["query_gmo_soft_gt_truncate_sigma"])
    self.use_query_gmo_dice_loss = bool(cfg["use_query_gmo_dice_loss"])
    if (
        self.query_gmo_hard_pair_enabled
        and self.query_gmo_hard_pair_mode == "loss_topk"
        and not self.use_query_gmo_dice_loss
    ):
        raise ValueError("loss_topk hard mining requires use_query_gmo_dice_loss=True")
    self.use_query_inst_center_match_loss = bool(cfg["use_query_inst_center_match_loss"])
    self.query_matched_loss_history_only = bool(cfg["query_matched_loss_history_only"])
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
    self.query_cls_binary_fg = bool(cfg["query_cls_binary_fg"])

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
    self.query_match_center_gate_radius_m = float(cfg["query_match_center_gate_radius_m"])
    self.query_recruit_max_radius_m = float(cfg["query_recruit_max_radius_m"])
    self.query_recruit_loss_weight = float(cfg["query_recruit_loss_weight"])
    self.query_center_match_loss_type = str(cfg["query_center_match_loss_type"]).lower()
    self.query_matched_gmo_bce_occ_size = tuple(int(v) for v in cfg["query_matched_gmo_bce_occ_size"])
    self.query_matched_gmo_sigma_floor_vox = float(cfg["query_matched_gmo_sigma_floor_vox"])
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
    self.query_multi_gaussian_occ_combine_mode = str(cfg["query_multi_gaussian_occ_combine_mode"]).lower()
    self.query_scale_spread_loss_weight = float(cfg["query_scale_spread_loss_weight"])
    self.query_scale_spread_target_frac = float(cfg["query_scale_spread_target_frac"])
    self.query_scale_spread_weight_mode = str(cfg["query_scale_spread_weight_mode"]).lower()
    if self.query_scale_spread_weight_mode not in ("raw", "visible"):
        raise ValueError(
            f"query_scale_spread_weight_mode must be 'raw'|'visible', got {self.query_scale_spread_weight_mode}"
        )
    self.query_scale_spread_norm_mode = str(cfg["query_scale_spread_norm_mode"]).lower()
    if self.query_scale_spread_norm_mode not in ("matched", "violators"):
        raise ValueError(
            f"query_scale_spread_norm_mode must be 'matched'|'violators', got {self.query_scale_spread_norm_mode}"
        )
    self.query_size_note_enabled = bool(cfg["query_size_note_enabled"])
    self.query_size_aux_loss_weight = float(cfg["query_size_aux_loss_weight"])
    if self.query_size_aux_loss_weight < 0.0:
        raise ValueError(f"query_size_aux_loss_weight must be >= 0, got {self.query_size_aux_loss_weight}")
    if self.query_size_aux_loss_weight > 0.0 and not self.query_size_note_enabled:
        raise ValueError("query_size_aux_loss_weight > 0 requires query_size_note_enabled=True (aux head 모듈이 note에 포함됨)")
    self.query_size_aux_big_weight_cap = max(1.0, float(cfg["query_size_aux_big_weight_cap"]))
    self.query_eval_occ_use_mixture = bool(cfg["query_eval_occ_use_mixture"])
    self.occ_score_threshold = float(cfg["occ_score_threshold"])
    self.fg_score_threshold = float(cfg["fg_score_threshold"])
    self.fg_score_iou_weight = float(cfg["fg_score_iou_weight"])
    self.fg_score_cls_weight = float(cfg["fg_score_cls_weight"])
    self.fg_score_cam_attn_weight = float(cfg["fg_score_cam_attn_weight"])
    self.fg_score_topk = max(0, int(cfg["fg_score_topk"]))
    self.fg_score_distance_nms_radius_m = max(0.0, float(cfg["fg_score_distance_nms_radius_m"]))
    self.query_embed_dim = int(cfg["query_embed_dim"])
    self.query_num_queries = int(cfg["query_num_queries"])
    self.query_transformer_num_layers = int(cfg["query_transformer_num_layers"])
    if cfg["query_transformer_kv_resolutions"] is None:
        self.query_transformer_kv_resolutions = None
    else:
        self.query_transformer_kv_resolutions = tuple(
            tuple(int(v) for v in hw) for hw in cfg["query_transformer_kv_resolutions"]
        )
        if len(self.query_transformer_kv_resolutions) != self.query_transformer_num_layers:
            raise ValueError(
                "query_transformer_kv_resolutions length must match "
                f"query_transformer_num_layers: got {len(self.query_transformer_kv_resolutions)} "
                f"vs {self.query_transformer_num_layers}"
            )
    self.query_transformer_learnable_kv_downsample = bool(cfg["query_transformer_learnable_kv_downsample"])
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
    self.query_attn_sigma_match_loss_weight = float(cfg["query_attn_sigma_match_loss_weight"])
    if self.query_attn_sigma_match_loss_weight < 0.0:
        raise ValueError(
            f"query_attn_sigma_match_loss_weight must be >= 0, got {self.query_attn_sigma_match_loss_weight}"
        )
    self.query_attn_sigma_match_start_iter = max(0, int(cfg["query_attn_sigma_match_start_iter"]))
    self.query_attn_sigma_match_min_mask_px = max(1, int(cfg["query_attn_sigma_match_min_mask_px"]))
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
    self.query_depth_head_num_layers = int(cfg["query_depth_head_num_layers"])
    self.query_depth_head_hidden_mult = int(cfg["query_depth_head_hidden_mult"])
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
    self.debug_query_mixture3d_vis_every = int(cfg["debug_query_mixture3d_vis_every"])

    return cfg


def apply_visualization_cfg(self, cfg):
    cfg = _merge_cfg(VISUALIZATION_CFG_DEFAULTS, cfg)

    self.debug_query_vis_dir = str(cfg["debug_query_vis_dir"])
    self.debug_query_center_marker_radius = max(0, int(cfg["debug_query_center_marker_radius"]))
    self.debug_query_gaussian_vis_mode = str(cfg["debug_query_gaussian_vis_mode"]).lower()
    self.debug_query_gaussian_prob_alpha_scale = float(cfg["debug_query_gaussian_prob_alpha_scale"])
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
    self.debug_query_mixture3d_vis_dir = str(cfg["debug_query_mixture3d_vis_dir"])
    self.debug_query_mixture3d_vis_max_queries = max(1, int(cfg["debug_query_mixture3d_vis_max_queries"]))
    self.debug_query_mixture3d_vis_max_gt_points = max(1000, int(cfg["debug_query_mixture3d_vis_max_gt_points"]))
    self.debug_query_mixture3d_vis_occ_max_voxels_per_query = max(
        100, int(cfg["debug_query_mixture3d_vis_occ_max_voxels_per_query"])
    )
    self.debug_query_mixture3d_vis_eval_occ_size = tuple(
        int(v) for v in cfg["debug_query_mixture3d_vis_eval_occ_size"]
    )

    return cfg
