# Developed by Jingyi Xu based on the codebase of Cam4DOcc, OpenOccupancy and PowerBEV
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

import torch
import torch.nn as nn

from mmdet.models import DETECTORS
from .bevdepth import BEVDepth
from projects.occ_plugin.occupancy.image2bev.transformer import TransformerModule
from projects.occ_plugin.occupancy.dense_heads.query_head import QueryHead
from projects.occ_plugin.occupancy.dense_heads.voxelizer import (
    SoftVoxelizerOneAdd,
    quat_to_rotmat_wxyz,
    rotmat_to_quat_wxyz,
)

from .utils_visualization import EfficientOCFVisualizationMixin
from .utils_dbg import EfficientOCFDebugMixin
from .utils_loss import EfficientOCFLossMixin
from .utils_matcher import EfficientOCFMatcherMixin
from .utils_bev_pool import EfficientOCFBEVPoolMixin
from .utils_gt_prep import EfficientOCFGTPrepMixin
from .utils_instance_img_debug import EfficientOCFInstanceImgDebugMixin
from .utils_query_projection import EfficientOCFQueryProjectionMixin
from .utils_geometry import EfficientOCFGeometryMixin
from .efficientocf_config import (
    apply_debug_cfg,
    apply_model_cfg,
    apply_visualization_cfg,
)


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
    def __init__(
        self,
        only_generate_dataset=False,
        empty_idx=0,
        loss_norm=False,
        point_cloud_range=None,
        time_receptive_field=None,
        n_future_frames=None,
        n_future_frames_plus=None,
        query_present_only=False,
        query_pred_num_frames=None,
        model_cfg=None,
        debug_cfg=None,
        visualization_cfg=None,
        **kwargs
    ):
        '''
        EfficientNet is our end-to-end baseline for 4D camera-only occupancy forecasting
        
        time_receptive_field: number of historical frames used for forecasting (including the present one), default: 3
        n_future_frames: number of forecasted future frames, default: 4
        n_future_frames_plus: number of estimated frames (> n_future_frames), default: 6 (if only forecasting occupancy states rather than instances, n_future_frames=n_future_frames_plus can be set)
        '''
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
        _, query_cls_loss_class_weights = apply_model_cfg(self, model_cfg)
        apply_debug_cfg(self, debug_cfg)
        apply_visualization_cfg(self, visualization_cfg)
        self._dbg_printed_instance_img_seq_len_warning = False
        self._last_inst_match_result = None
        self._last_gt_instance_bev_feat_cache = None
        self._last_query_inst_depth_target_pack = None
        self._last_query_attn_soft_lift_pack = None
        query_raw_to_compact_size = max(256, max(self.query_class_ids) + 1)
        query_raw_to_compact = torch.full(
            (query_raw_to_compact_size,),
            fill_value=-1,
            dtype=torch.long,
        )
        if self.query_binary_cls:
            # background raw id -> 0, every foreground raw id -> 1 (foreground)
            query_raw_to_compact[int(self.query_class_ids[0])] = 0
            for raw_id in self.query_class_ids[1:]:
                query_raw_to_compact[int(raw_id)] = 1
        else:
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
        self.register_buffer(
            "query_cls_loss_class_weights",
            torch.as_tensor(query_cls_loss_class_weights, dtype=torch.float32),
            persistent=False,
        )
        self._query_train_iter = 0
        self._train_iter_synced = False

        context_feat_dim = self._get_context_feat_dim_from_depth_net()
        geo_input_dim = int(getattr(self.img_view_transformer, "cam_channels", 27))

        self.transformer = TransformerModule(
            feat_dim=context_feat_dim,
            geo_input_dim=geo_input_dim,
            num_queries=self.query_num_queries,
            num_heads=4,
            num_layers=self.query_transformer_num_layers,
            kv_resolutions=self.query_transformer_kv_resolutions,
            num_cams=6,
            embed_dim=self.query_embed_dim,
            max_time=self.time_receptive_field,
            query_id_reinject_scale=self.query_id_reinject_scale,
            ca_kv_identity_init=self.query_ca_kv_identity_init,
            ca_attn_tau=self.query_ca_attn_tau,
            query_decor_loss_weight=self.query_decor_loss_weight,
            query_attn_overlap_loss_weight=self.query_attn_overlap_loss_weight,
            attn_vis_dir=self.query_attn_vis_dir,
        )

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
            query_binary_cls=self.query_binary_cls,
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
            query_gaussian_head_num_layers=self.query_gaussian_head_num_layers,
            query_cls_head_num_layers=self.query_cls_head_num_layers,
            query_cls_use_gaussian_params=self.query_cls_use_gaussian_params,
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
            debug_query_objectness_vis_threshold=self.debug_query_objectness_vis_threshold,
            debug_query_gaussian_vis_mode=self.debug_query_gaussian_vis_mode,
            debug_query_gaussian_prob_threshold=self.debug_query_gaussian_prob_threshold,
            debug_query_gaussian_prob_alpha_scale=self.debug_query_gaussian_prob_alpha_scale,
            use_query_size_embedding=self.use_query_size_embedding,
            query_size_attn_threshold=self.query_size_attn_threshold,
            query_size_depth_ref_m=self.query_size_depth_ref_m,
            query_size_log_alpha=self.query_size_log_alpha,
            query_size_gate_init=self.query_size_gate_init)
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
            lambda_occ=1.0,
            gaussian_combine_mode=self.query_multi_gaussian_occ_combine_mode,
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
            gaussian_combine_mode=self.query_multi_gaussian_occ_combine_mode,
        )
        ev_x, ev_y, ev_z = self.debug_query_mixture3d_vis_eval_occ_size
        self.mixture3d_eval_voxelizer = SoftVoxelizerOneAdd(
            point_cloud_range=point_cloud_range,
            voxel_size=(range_x / float(ev_x), range_y / float(ev_y), range_z / float(ev_z)),
            occ_size=(ev_x, ev_y, ev_z),
            as_prob=True,
            lambda_occ=1.0,
            gaussian_truncate_sigma=float(self.voxelizer.gaussian_truncate_sigma),
            gaussian_sigma_floor_vox=0.0,
            gaussian_combine_mode=self.query_multi_gaussian_occ_combine_mode,
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
        self._train_iter_synced = True
        if hasattr(self, "query_head") and hasattr(self.query_head, "set_train_iteration"):
            self.query_head.set_train_iteration(step, one_based=True)
        if hasattr(self, "transformer") and hasattr(self.transformer, "set_train_iteration"):
            self.transformer.set_train_iteration(step, one_based=True)

    def _get_query_train_iteration(self, advance_if_unsynced: bool = False) -> int:
        if not self._train_iter_synced and advance_if_unsynced:
            self._query_train_iter += 1
        return int(self._query_train_iter)

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
        present_mix_quat_tqg4: torch.Tensor = None,
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
            "mix_quat_tqg4": present_mix_quat_tqg4,
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
            and torch.is_tensor(present_mix_quat_tqg4)
            and torch.is_tensor(present_mix_weights_tqg)
            and present_mix_centers_world_tqg3.dim() == 4
            and present_mix_sigmas_world_tqg3.dim() == 4
            and present_mix_quat_tqg4.dim() == 4
            and int(present_mix_quat_tqg4.shape[-1]) == 4
            and present_mix_weights_tqg.dim() == 3
            and int(present_mix_centers_world_tqg3.shape[0]) == 1
            and tuple(present_mix_centers_world_tqg3.shape) == tuple(present_mix_sigmas_world_tqg3.shape)
            and tuple(present_mix_centers_world_tqg3.shape[:3]) == tuple(present_mix_quat_tqg4.shape[:3])
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
            out["mix_quat_tqg4"] = present_mix_quat_tqg4[0].to(
                device=present_centers_world_tq3.device,
                dtype=present_mix_quat_tqg4.dtype,
            ).unsqueeze(0).expand(
                int(centers_world_traj_tq3.shape[0]), -1, -1, -1
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
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
    ):
        out = {
            "centers_world_tq3": centers_world_tq3,
            "gaussian_sigmas_world_tq3": gaussian_sigmas_world_tq3,
            "mixture_centers_world_tqg3": mixture_centers_world_tqg3,
            "mixture_sigmas_world_tqg3": mixture_sigmas_world_tqg3,
            "mixture_quat_tqg4": mixture_quat_tqg4,
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
            and torch.is_tensor(mixture_quat_tqg4)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and mixture_quat_tqg4.dim() == 4
            and int(mixture_quat_tqg4.shape[-1]) == 4
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_quat_tqg4.shape[:3])
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

        quat_aligned = []
        rot_src_tqg33 = quat_to_rotmat_wxyz(mixture_quat_tqg4.to(torch.float32))
        for step in range(t_count):
            rot_delta = lidar_target_to_present[step].to(
                device=centers_world_tq3.device, dtype=torch.float32
            )[:3, :3]
            rot_step_qg33 = torch.matmul(
                rot_delta.view(1, 1, 3, 3),
                rot_src_tqg33[step],
            )
            quat_aligned.append(rotmat_to_quat_wxyz(rot_step_qg33))
        out["mixture_quat_tqg4"] = torch.stack(quat_aligned, dim=0).to(
            device=mixture_quat_tqg4.device,
            dtype=mixture_quat_tqg4.dtype,
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
        return_instance_img_debug=False,
        return_instance_img_debug_fullseq=True,
        return_query_cam_gaussian_vis_debug=False,
        return_query_match_inputs=False,
        defer_trajectory=False,
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
        query_size_log_scalar_tq = None
        query_size_dbg = None
        if bool(getattr(self, "use_query_size_embedding", False)):
            query_size_log_scalar_tq, query_size_dbg = self.query_head.compute_query_size_log_scalar(
                query_attn_weights_tqnhw=query_attn_weights,
                query_depth_m_tq=self._last_query_attn_soft_lift_pack.get("depth_expect_tq", None),
            )
        query_head_outputs = self.query_head.apply_lifted_centers_to_outputs(
            query_head_outputs,
            lifted_centers_world,
            detach_query_for_center=self.query_center_loss_detach_query_feat,
            defer_trajectory=defer_trajectory,
            query_size_log_scalar_tq=query_size_log_scalar_tq,
            query_size_dbg=query_size_dbg,
        )
        centers_world = query_head_outputs["centers_world_tq3"]
        center_logits = query_head_outputs["center_logits_tq3"]
        query_cls_logits_qc = query_head_outputs["query_cls_logits_qc"]
        query_cls_scores_qc = query_head_outputs["query_cls_scores_qc"]
        query_cls_scores_tqc = query_head_outputs["query_cls_scores_tqc"]
        gaussian_sigmas_world = None
        mixture_centers_world_tqg3 = query_head_outputs["mixture_centers_world_tqg3"]
        mixture_sigmas_world_tqg3 = query_head_outputs["mixture_sigmas_world_tqg3"]
        mixture_quat_tqg4 = query_head_outputs["mixture_quat_tqg4"]
        mixture_weights_tqg = query_head_outputs["mixture_weights_tqg"]
        query_present_local_idx = int(query_head_outputs.get("query_present_local_idx", 0))
        if int(centers_world.shape[0]) > 0:
            query_present_local_idx = max(
                0, min(int(centers_world.shape[0]) - 1, int(query_present_local_idx))
            )
        else:
            query_present_local_idx = 0
        traj_offsets_fq2 = query_head_outputs.get("traj_offsets_fq2", None)
        traj_motion_input_tqd2 = query_head_outputs.get("traj_motion_input_tqd2", None)
        query_size_dbg = query_head_outputs.get("query_size_dbg", None)

        present_centers_world_tq3 = centers_world.narrow(0, query_present_local_idx, 1).contiguous()
        present_sigmas_world_tq3 = None
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
        present_mix_quat_tqg4 = (
            mixture_quat_tqg4.narrow(0, query_present_local_idx, 1).contiguous()
            if torch.is_tensor(mixture_quat_tqg4)
            and mixture_quat_tqg4.dim() == 4
            and int(mixture_quat_tqg4.shape[0]) > query_present_local_idx
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
            present_sigmas_world_tq3=None,
            present_mix_centers_world_tqg3=present_mix_centers_world_tqg3,
            present_mix_sigmas_world_tqg3=present_mix_sigmas_world_tqg3,
            present_mix_quat_tqg4=present_mix_quat_tqg4,
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
            gaussian_sigmas_world_traj_tq3 = None
        mixture_centers_world_traj_tqg3 = traj_geom_pack.get("mix_centers_world_tqg3", None)
        if not torch.is_tensor(mixture_centers_world_traj_tqg3):
            mixture_centers_world_traj_tqg3 = mixture_centers_world_tqg3
        mixture_sigmas_world_traj_tqg3 = traj_geom_pack.get("mix_sigmas_world_tqg3", None)
        if not torch.is_tensor(mixture_sigmas_world_traj_tqg3):
            mixture_sigmas_world_traj_tqg3 = mixture_sigmas_world_tqg3
        mixture_quat_traj_tqg = traj_geom_pack.get("mix_quat_tqg4", None)
        if not torch.is_tensor(mixture_quat_traj_tqg):
            mixture_quat_traj_tqg = mixture_quat_tqg4
        mixture_weights_traj_tqg = traj_geom_pack.get("mix_weights_tqg", None)
        if not torch.is_tensor(mixture_weights_traj_tqg):
            mixture_weights_traj_tqg = mixture_weights_tqg
        pred_occ = None
        if voxelize:
            if self.center_only_mode:
                pred_occ = None
            else:
                if (
                    torch.is_tensor(present_mix_centers_world_tqg3)
                    and torch.is_tensor(present_mix_sigmas_world_tqg3)
                    and torch.is_tensor(present_mix_quat_tqg4)
                    and torch.is_tensor(present_mix_weights_tqg)
                ):
                    pred_occ_q = self.voxelizer.forward_gaussian_mixture_grouped(
                        mixture_centers_world_tkg3=present_mix_centers_world_tqg3,
                        mixture_sigmas_world_tkg3=present_mix_sigmas_world_tqg3,
                        mixture_weights_tkg=present_mix_weights_tqg,
                        mixture_quat_tkg4=present_mix_quat_tqg4,
                        pair_weights_tk=None,
                        pair_chunk_size=int(self.query_multi_gaussian_pair_chunk),
                    ).to(torch.float32)
                    pred_occ = (1.0 - torch.prod((1.0 - pred_occ_q).clamp(0.0, 1.0), dim=1)).clamp(0.0, 1.0)

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
            query_attn_weights,
            query_attn_bbox_targets,
            instance_img_debug_bundle,
            query_match_inputs,
            traj_offsets_fq2,
            query_present_local_idx,
            mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3,
            mixture_quat_tqg4,
            mixture_weights_tqg,
            gaussian_sigmas_world_traj_tq3,
            mixture_centers_world_traj_tqg3,
            mixture_sigmas_world_traj_tqg3,
            mixture_quat_traj_tqg,
            mixture_weights_traj_tqg,
            traj_motion_input_tqd2,
            query_size_dbg,
        )


    def _maybe_save_eval_query_vis(self, pred_occ_prob, gt_inst, centers_world,
                                   present_idx, cls_scores_qc, bundle, prob_threshold,
                                   img_metas=None, img_inputs_seq=None,
                                   future_egomotion=None, instance_img_debug_bundle=None,
                                   eval_mode="future", centers_future_tq3=None,
                                   pred_occ_future=None, mix_future=None):
        import os
        if os.environ.get("EOCF_EVAL_VIS", "0") != "1":
            return
        try:
            from mmcv.runner import get_dist_info
            step_local = int(getattr(self, "_eval_vis_step", 0))
            self._eval_vis_step = step_local + 1
            every = max(1, int(os.environ.get("EOCF_EVAL_VIS_EVERY", "50")))
            if (step_local % every) != 0:
                return
            head = getattr(self, "query_head", None)
            if head is None or not hasattr(head, "maybe_save_query_debug_vis"):
                return
            vis_dir = os.environ.get("EOCF_EVAL_VIS_DIR", "./work_dirs/eval_vis")
            os.makedirs(vis_dir, exist_ok=True)
            qdv_dir = os.path.join(vis_dir, "query_debug_vis")
            os.makedirs(qdv_dir, exist_ok=True)
            head.debug_vis_dir = qdv_dir
            head.debug_vis_every = 1
            head._eval_vis_all_ranks = True
            self._eval_vis_all_ranks = True
            rank, _ = get_dist_info()
            step = int(rank) * 1000000 + step_local
            global_idx = self._extract_eval_global_idx(img_metas)
            if global_idx is not None:
                step = int(global_idx)
            pres = int(present_idx)
            conf = (1.0 - cls_scores_qc[:, int(self.query_bg_class)]).clamp(0.0, 1.0).detach()
            cls_ids = torch.argmax(cls_scores_qc, dim=-1).detach()
            if str(eval_mode).lower() != "present" and torch.is_tensor(centers_future_tq3) \
                    and centers_future_tq3.dim() == 3:
                pts = centers_future_tq3.detach()
                pred_vis = pred_occ_future.detach() if torch.is_tensor(pred_occ_future) else None
                gt_vis = self._eval_future_tail(gt_inst).detach() if torch.is_tensor(gt_inst) else None
                conf_seq = conf.unsqueeze(0).expand(int(pts.shape[0]), -1).contiguous()
                cls_seq = cls_ids.unsqueeze(0).expand(int(pts.shape[0]), -1).contiguous()
                vis_max_frames = int(pts.shape[0])
            else:
                pts = centers_world[pres:pres + 1].detach()
                pred_vis = pred_occ_prob.detach() if torch.is_tensor(pred_occ_prob) else None
                gt_vis = gt_inst.detach() if torch.is_tensor(gt_inst) else None
                conf_seq = conf.unsqueeze(0)
                cls_seq = cls_ids.unsqueeze(0)
                vis_max_frames = 1
            vis_bundle = dict(bundle) if isinstance(bundle, dict) else None
            if isinstance(vis_bundle, dict):
                vis_bundle["_skip_traj_rows"] = True
                if global_idx is not None:
                    vis_bundle["_filename_prefix"] = "sample"
            if str(eval_mode).lower() != "present" and vis_max_frames > 1 and isinstance(vis_bundle, dict):
                bundle_2d = self._build_eval_future_2d_bundle(vis_bundle, pts, mix_future, conf, cls_ids)
            else:
                bundle_2d = vis_bundle
            head.maybe_save_query_debug_vis(
                pred_occ_prob=pred_vis,
                gt_occ=None,
                gt_occ_inst=gt_vis,
                gt_occ_semantic=None,
                ignore_index=255,
                points_world=pts,
                step=step,
                max_frames=vis_max_frames,
                voxel_center_offset=0.5,
                prob_threshold=float(prob_threshold),
                pred_layout="zyx",
                pred_occ_prob_all_obj=pred_vis,
                points_world_all=pts,
                point_conf_all=conf_seq,
                point_class_ids_all=cls_seq,
                query_vis_bundle=bundle_2d,
            )
            if torch.is_tensor(gt_inst) and isinstance(vis_bundle, dict) and hasattr(self, "maybe_save_query_mixture_3d_vis"):
                m3d_dir = os.path.join(vis_dir, "query_mixture3d_vis")
                os.makedirs(m3d_dir, exist_ok=True)
                self.debug_query_mixture3d_vis_dir = m3d_dir
                self.debug_query_mixture3d_vis_every = 1
                self._eval_vis_all_ranks = True
                if str(eval_mode).lower() != "present" and isinstance(bundle_2d, dict) \
                        and torch.is_tensor(bundle_2d.get("selected_points_tq3", None)):
                    self.maybe_save_query_mixture_3d_vis(
                        query_vis_bundle=bundle_2d,
                        gt_instance_occ3d_txyz=self._eval_future_tail(gt_inst).detach(),
                        img_metas=img_metas, step=step,
                        occ_threshold=float(prob_threshold), frame_idx=1,
                        is_eval=True,
                    )
                else:
                    self.maybe_save_query_mixture_3d_vis(
                        query_vis_bundle=vis_bundle,
                        gt_instance_occ3d_txyz=gt_inst.detach(),
                        img_metas=img_metas, step=step,
                        occ_threshold=float(prob_threshold),
                        is_eval=True,
                    )
            gt_inst_d = gt_inst.detach() if torch.is_tensor(gt_inst) else None
            if isinstance(instance_img_debug_bundle, dict) and isinstance(vis_bundle, dict):
                self.debug_query_cam_gaussian_vis_dir = os.path.join(vis_dir, "query_cam_gaussian_vis")
                os.makedirs(self.debug_query_cam_gaussian_vis_dir, exist_ok=True)
                self.debug_query_cam_gaussian_vis_enabled = True
                self.debug_query_cam_gaussian_vis_every = 1
                self.maybe_save_query_cam_gaussian_vis(
                    debug_bundle=instance_img_debug_bundle,
                    query_vis_bundle=vis_bundle,
                    future_egomotion=future_egomotion,
                    step=step,
                    gt_segmentation_instance3d=gt_inst_d,
                )
        except Exception as e:
            print(f"[eval_vis] skipped (err={e})", flush=True)

    @staticmethod
    def _extract_eval_global_idx(img_metas):
        if img_metas is None:
            return None
        meta = img_metas
        while isinstance(meta, (list, tuple)) and len(meta) > 0:
            meta = meta[0]
        if not isinstance(meta, dict):
            return None
        val = meta.get("global_idx", None)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            val = val[0]
        if torch.is_tensor(val):
            if val.numel() == 0:
                return None
            val = val.reshape(-1)[0].item()
        try:
            return int(val)
        except Exception:
            return None

    def forward_test(self, img_inputs_seq=None, img_metas=None, future_egomotion=None,
                     segmentation=None, segmentation_bev=None, gt_occ=None, gt_occ_inst=None,
                     segmentation_instance3d=None, segmentation_cls_instance3d=None,
                     **kwargs):
        return self.simple_test(
            img_inputs_seq=img_inputs_seq, img_metas=img_metas,
            future_egomotion=future_egomotion, segmentation_bev=segmentation_bev,
            gt_occ=gt_occ, gt_occ_inst=gt_occ_inst,
            segmentation=segmentation, segmentation_instance3d=segmentation_instance3d,
            segmentation_cls_instance3d=segmentation_cls_instance3d, **kwargs)

    @torch.no_grad()
    def simple_test(self, img_inputs_seq=None, img_metas=None, future_egomotion=None,
                    segmentation=None, segmentation_bev=None, gt_occ=None, gt_occ_inst=None,
                    segmentation_instance3d=None, segmentation_cls_instance3d=None,
                    **kwargs):
        import os
        import numpy as np

        def _pack(cm):
            return dict(hist_for_iou=cm,
                        hist_for_iou_bbox=np.zeros((2, 2), dtype=np.int64),
                        height_l1=torch.tensor(0.0),
                        iou_3d=float("nan"), recall_3d=float("nan"))

        empty = _pack(np.zeros((2, 2), dtype=np.int64))
        eval_vis_on = os.environ.get("EOCF_EVAL_VIS", "0") == "1"
        feat_out = self.extract_feat_query(
            img_inputs_seq, img_metas, future_egomotion, voxelize=False,
            return_query_match_inputs=True,
            return_instance_img_debug=eval_vis_on,
            return_instance_img_debug_fullseq=True,
            return_query_cam_gaussian_vis_debug=eval_vis_on)
        centers_world = feat_out[4]
        centers_world_traj = feat_out[5]
        sigmas_world = feat_out[2]
        cls_scores_qc = feat_out[8]
        present_idx = feat_out[19]
        mix_c, mix_s, mix_y, mix_w = feat_out[20], feat_out[21], feat_out[22], feat_out[23]
        sigmas_world_traj = feat_out[24] if len(feat_out) > 24 else None
        mix_future = (
            self._eval_future_tail(feat_out[25]) if len(feat_out) > 25 and torch.is_tensor(feat_out[25]) else None,
            self._eval_future_tail(feat_out[26]) if len(feat_out) > 26 and torch.is_tensor(feat_out[26]) else None,
            self._eval_future_tail(feat_out[27]) if len(feat_out) > 27 and torch.is_tensor(feat_out[27]) else None,
            self._eval_future_tail(feat_out[28]) if len(feat_out) > 28 and torch.is_tensor(feat_out[28]) else None,
        )
        eval_instance_img_debug_bundle = feat_out[16]
        if not (torch.is_tensor(centers_world) and torch.is_tensor(cls_scores_qc)
                and torch.is_tensor(segmentation_bev)):
            return empty
        present_idx = int(present_idx) if present_idx is not None else centers_world.shape[0] - 1

        gt_seg_inst3d = self._prepare_segmentation_instance3d(
            segmentation_instance3d=segmentation_instance3d)
        gt_bundle = self._prepare_gt_occ_inst_primary_targets(
            gt_occ_inst=gt_occ_inst, fallback_segmentation_instance3d_txyz=gt_seg_inst3d)
        gt_inst = gt_bundle.get("dense_inst_txyz", None) if isinstance(gt_bundle, dict) else None
        if torch.is_tensor(gt_inst):
            while gt_inst.dim() > 4:
                gt_inst = gt_inst[0]
            gt_inst = gt_inst.to(centers_world.device)

        sel_idx = None
        selected_score = None
        bundle = self._build_query_visualization_bundle(
            centers_world_tq3=centers_world,
            query_cls_scores_qc=cls_scores_qc,
            inst_match_result={},
            gt_segmentation_instance3d_txyz=gt_inst,
            gaussian_sigmas_world_tq3=sigmas_world,
            mixture_centers_world_tqg3=mix_c, mixture_sigmas_world_tqg3=mix_s,
            mixture_quat_tqg4=mix_y, mixture_weights_tqg=mix_w,
            traj_mode_idx_q=None, query_attn_cam_score_pack=None)
        if isinstance(bundle, dict):
            sel_idx = bundle.get("selected_query_idx_q", None)
            selected_score = bundle.get("selected_score_q", None)

        thr = float(os.environ.get("EOCF_EVAL_OCC_THR", getattr(self, "eval_occ_threshold", 0.5)))
        eval_mode = "present" if self._eval_mode_is_present() else "future"
        H, W, D = int(self.voxelizer.H), int(self.voxelizer.W), int(self.voxelizer.D)
        n_sel = int(sel_idx.numel()) if torch.is_tensor(sel_idx) else 0
        pred_occ_vis = None
        pred_occ_eval = None
        centers_eval_tq3 = self._eval_future_tail(self._pick_tensor(centers_world_traj, centers_world))
        sigmas_eval_tq3 = self._eval_future_tail(self._pick_tensor(sigmas_world_traj, sigmas_world))
        if not (torch.is_tensor(centers_eval_tq3) and torch.is_tensor(sigmas_eval_tq3)):
            return empty
        if n_sel == 0:
            pred_bev = torch.zeros((H, W), dtype=torch.bool, device=centers_world.device)
            pred_occ3d = torch.zeros((D, H, W), dtype=torch.bool, device=centers_world.device)
            pred_txyz = centers_world.new_zeros(
                (int(centers_eval_tq3.shape[0]), W, H, D), dtype=torch.bool)
        else:
            cen = centers_world[present_idx].index_select(0, sel_idx)
            weights_sel = (
                selected_score.to(device=cen.device, dtype=cen.dtype).reshape(-1)
                if torch.is_tensor(selected_score) and int(selected_score.numel()) == n_sel
                else cen.new_ones((n_sel,), dtype=cen.dtype)
            )
            cen_eval = centers_eval_tq3.index_select(1, sel_idx)
            weights_tq = weights_sel.view(1, -1).expand(int(cen_eval.shape[0]), -1).contiguous()
            use_mix = (
                bool(getattr(self, "query_eval_occ_use_mixture", False))
                and all(torch.is_tensor(m) for m in (mix_c, mix_s, mix_y, mix_w))
                and mix_c.dim() == 4
                and int(mix_c.shape[0]) > int(present_idx)
                and int(mix_c.shape[1]) == int(centers_world.shape[1])
            )
            if use_mix:
                off_sg3, sig_sg3, yaw_sg, w_sg = self._eval_select_mixture_params(
                    present_idx, mix_c, mix_s, mix_y, mix_w, centers_world, sel_idx)
                pred_occ = self._eval_mixture_scene_occ(
                    cen.unsqueeze(0), off_sg3, sig_sg3, yaw_sg, w_sg, weights_sel.view(1, -1))
                pred_occ_eval = self._eval_mixture_scene_occ(
                    cen_eval, off_sg3, sig_sg3, yaw_sg, w_sg, weights_tq)
            else:
                sig = (sigmas_world[present_idx].index_select(0, sel_idx)
                       if torch.is_tensor(sigmas_world) else None)
                sig_eval = sigmas_eval_tq3.index_select(1, sel_idx)
                pred_occ = self.voxelizer(
                    cen.unsqueeze(0),
                    sigmas_world=(sig.unsqueeze(0) if torch.is_tensor(sig) else None),
                    weights=weights_sel.view(1, -1))
                pred_occ_eval = self.voxelizer(cen_eval, sigmas_world=sig_eval, weights=weights_tq)
            pred_occ3d = (pred_occ[0, 0] > thr)
            pred_bev = pred_occ3d.amax(dim=0)
            pred_occ_vis = pred_occ
            pred_txyz = (pred_occ_eval[:, 0] > thr).permute(0, 3, 2, 1).contiguous()

        if eval_mode == "present":
            pred_txyz = pred_occ3d.permute(2, 1, 0).unsqueeze(0).contiguous()

        self._maybe_save_eval_query_vis(
            pred_occ_prob=pred_occ_vis, gt_inst=gt_inst, centers_world=centers_world,
            present_idx=present_idx, cls_scores_qc=cls_scores_qc, bundle=bundle,
            prob_threshold=thr, img_metas=img_metas,
            img_inputs_seq=img_inputs_seq, future_egomotion=future_egomotion,
            instance_img_debug_bundle=eval_instance_img_debug_bundle,
            eval_mode=eval_mode, centers_future_tq3=centers_eval_tq3,
            pred_occ_future=pred_occ_eval, mix_future=mix_future)

        gt = segmentation_bev
        while gt.dim() > 3:
            gt = gt[0]
        gt = (gt > 0).to(pred_bev.device)
        pred_bev_t = pred_txyz.permute(0, 3, 2, 1).any(dim=1).contiguous()

        align = getattr(self, "_eval_occ_align", None)
        if align is None:
            env = os.environ.get("EOCF_EVAL_ALIGN", "").strip().lower()
            if env == "auto":
                print(f"[simple_test][diag] #selected={n_sel}/{int(cls_scores_qc.shape[0])} "
                      f"pred_bev#={int(pred_bev.sum())} pred_shape={tuple(pred_bev.shape)} "
                      f"gt_shape={tuple(gt.shape)} (thr={thr})", flush=True)
                align = self._calibrate_eval_bev_align(pred_bev, gt)
            else:
                t, tr, fh, fw = [int(x) for x in env.split(",")] if env else (2, 1, 0, 0)
                align = (t, bool(tr), (fh, fw))
            self._eval_occ_align = align
            print(f"[simple_test] BEV align {align}", flush=True)
        if align is None:
            return empty
        _, bev_transpose, bev_flips = align
        gt_bin_t = self._apply_bev_align_sequence(
            self._eval_mode_slice(gt), bev_transpose, bev_flips)
        t_bev = min(int(pred_bev_t.shape[0]), int(gt_bin_t.shape[0]))
        cm_nusocc = (
            self._binary_occ_cm(pred_bev_t[:t_bev], gt_bin_t[:t_bev])
            if t_bev > 0 else np.zeros((2, 2), dtype=np.int64)
        )

        iou_3d, recall_3d = float("nan"), float("nan")
        cm_bbox = np.zeros((2, 2), dtype=np.int64)
        gt_bbox_src = self._eval_bbox_cls_occ_txyz(segmentation_cls_instance3d)
        if not torch.is_tensor(gt_bbox_src):
            gt_bbox_src = self._eval_bbox_occ_txyz(
                segmentation=segmentation,
                segmentation_instance3d=segmentation_instance3d,
                gt_occ_inst=gt_occ_inst,
                fallback_shape=(tuple(gt_occ.shape) if torch.is_tensor(gt_occ) else None),
            )
        if torch.is_tensor(gt_bbox_src):
            gt_bbox_src = gt_bbox_src.to(device=pred_occ3d.device, dtype=torch.long)
        gt_occ_fg = self._eval_nusocc_3d_fg(gt_occ, device=pred_occ3d.device)
        gt_occ_valid = self._eval_nusocc_3d_valid(gt_occ, device=pred_occ3d.device)
        if torch.is_tensor(gt_occ_fg) and gt_occ_fg.dim() == 4:
            align3d = getattr(self, "_eval_3d_align", None)
            if align3d is None:
                env3d = os.environ.get("EOCF_EVAL_3D_ALIGN", "").strip().lower()
                if env3d == "auto":
                    align3d = self._calibrate_eval_3d_align(pred_occ3d, gt_occ_fg)
                else:
                    t, tr, fh, fw, fz = [int(x) for x in env3d.split(",")] if env3d else (0, 1, 0, 0, 0)
                    align3d = (t, bool(tr), fh, fw, fz)
                    print(f"[simple_test] 3D align {align3d}", flush=True)
                self._eval_3d_align = align3d
            if align3d is not None:
                _, transpose, fh, fw, fz = align3d
                gt3d_t = self._apply_3d_align_sequence(
                    self._eval_mode_slice(gt_occ_fg), transpose, fh, fw, fz)
                valid3d_t = (
                    self._apply_3d_align_sequence(self._eval_mode_slice(gt_occ_valid), transpose, fh, fw, fz)
                    if torch.is_tensor(gt_occ_valid) and gt_occ_valid.dim() == 4 else None
                )
                bbox3d_t = None
                if tuple(gt3d_t.shape[1:]) == tuple(pred_occ3d.shape):
                    if torch.is_tensor(gt_bbox_src) and gt_bbox_src.dim() == 4:
                        bbox3d_src_t = self._eval_mode_slice((gt_bbox_src > 0) & (gt_bbox_src != 255))
                        bbox3d_t = self._apply_3d_align_sequence(
                            bbox3d_src_t, transpose, fh, fw, fz)
                        if torch.is_tensor(bbox3d_t):
                            bbox_bev_t = bbox3d_t.any(dim=1).contiguous()
                            t_bbox = min(int(pred_bev_t.shape[0]), int(bbox_bev_t.shape[0]))
                            if t_bbox > 0 and tuple(bbox_bev_t.shape[1:]) == tuple(pred_bev_t.shape[1:]):
                                cm_bbox = self._binary_occ_cm(
                                    pred_bev_t[:t_bbox], bbox_bev_t[:t_bbox])
                    t_eval = min(int(pred_txyz.shape[0]), int(gt3d_t.shape[0]))
                    if t_eval > 0:
                        pred_zyx_t = pred_txyz[:t_eval].permute(0, 3, 2, 1).contiguous()
                        gt3d_t = gt3d_t[:t_eval]
                        if tuple(pred_zyx_t.shape) == tuple(gt3d_t.shape):
                            bbox_arg = bbox3d_t[:t_eval] if (
                                torch.is_tensor(bbox3d_t) and int(bbox3d_t.shape[0]) >= t_eval
                            ) else None
                            valid_arg = valid3d_t[:t_eval] if (
                                torch.is_tensor(valid3d_t) and int(valid3d_t.shape[0]) >= t_eval
                            ) else None
                            iou_3d, recall_3d = self._iou_recall_3d(
                                pred_zyx_t, gt3d_t, bbox3d=bbox_arg, valid3d=valid_arg)

        return dict(hist_for_iou=cm_nusocc, hist_for_iou_bbox=cm_bbox,
                    height_l1=torch.tensor(0.0),
                    iou_3d=iou_3d, recall_3d=recall_3d)

    @staticmethod
    def _binary_occ_cm(pred_bin, gt_bin):
        import numpy as np
        p = pred_bin.bool().reshape(-1)
        g = gt_bin.bool().reshape(-1)
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        tn = int((~p & ~g).sum())
        return np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    @staticmethod
    def _apply_bev_align_sequence(gt_txy, transpose, flips):
        g = gt_txy.transpose(1, 2).contiguous() if transpose else gt_txy
        for d, f in enumerate(flips):
            if f:
                g = torch.flip(g, dims=[d + 1])
        return g.contiguous()

    def _calibrate_eval_bev_align(self, pred_bev, gt):
        T = int(gt.shape[0])
        best, best_iou = None, -1.0
        for t in range(T):
            base = gt[t]
            for transpose in (False, True):
                g0 = base.transpose(0, 1).contiguous() if transpose else base
                if tuple(g0.shape) != tuple(pred_bev.shape):
                    continue
                for fh in (0, 1):
                    for fw in (0, 1):
                        g = g0
                        for d, fl in ((0, fh), (1, fw)):
                            if fl:
                                g = torch.flip(g, dims=[d])
                        inter = float((pred_bev & g).sum())
                        union = float((pred_bev | g).sum())
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou, best = iou, (t, transpose, (fh, fw))
        print(f"[simple_test] calib best BEV movable IoU={best_iou:.4f} at {best}", flush=True)
        return best

    @staticmethod
    def _apply_3d_align(fg_txyz, t_idx, transpose, flipH, flipW, flipZ):
        g = fg_txyz[t_idx].permute(2, 0, 1)
        if transpose:
            g = g.transpose(1, 2)
        if flipZ:
            g = torch.flip(g, dims=[0])
        if flipH:
            g = torch.flip(g, dims=[1])
        if flipW:
            g = torch.flip(g, dims=[2])
        return g.bool().contiguous()

    @staticmethod
    def _apply_3d_align_sequence(fg_txyz, transpose, flipH, flipW, flipZ):
        g = fg_txyz.permute(0, 3, 1, 2)
        if transpose:
            g = g.transpose(2, 3)
        if flipZ:
            g = torch.flip(g, dims=[1])
        if flipH:
            g = torch.flip(g, dims=[2])
        if flipW:
            g = torch.flip(g, dims=[3])
        return g.bool().contiguous()

    def _calibrate_eval_3d_align(self, pred_occ3d, fg_txyz):
        T = int(fg_txyz.shape[0])
        p = pred_occ3d.bool()
        best, best_iou = None, -1.0
        for t in range(T):
            for transpose in (False, True):
                for fz in (0, 1):
                    for fh in (0, 1):
                        for fw in (0, 1):
                            g = self._apply_3d_align(fg_txyz, t, transpose, fh, fw, fz)
                            if tuple(g.shape) != tuple(p.shape):
                                continue
                            inter = float((p & g).sum())
                            union = float((p | g).sum())
                            iou = inter / union if union > 0 else 0.0
                            if iou > best_iou:
                                best_iou, best = iou, (t, transpose, fh, fw, fz)
        if best_iou <= 0.0:
            return None
        print(f"[simple_test] calib best 3D movable IoU={best_iou:.4f} at {best}", flush=True)
        return best

    @staticmethod
    def _iou_recall_3d(pred3d, gt3d, bbox3d=None, valid3d=None):
        p = pred3d.bool()
        g = gt3d.bool()
        if torch.is_tensor(valid3d) and tuple(valid3d.shape) == tuple(p.shape):
            valid = valid3d.bool()
            p = p & valid
            g = g & valid
        tp = float((p & g).sum())
        fp = float((p & ~g).sum())
        fn = float((~p & g).sum())
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        bbox_fp = 0.0
        if torch.is_tensor(bbox3d) and tuple(bbox3d.shape) == tuple(p.shape):
            bbox_fp = float((p & ~g & bbox3d.bool()).sum())
        denom = tp + fn + fp - bbox_fp
        recall = (tp + bbox_fp) / denom if denom > 0 else float("nan")
        return iou, recall

    def _eval_nusocc_3d_fg(self, gt_occ, device=None):
        if gt_occ is None:
            return None
        if isinstance(gt_occ, (list, tuple)):
            frames = [x for x in gt_occ if torch.is_tensor(x)]
            if len(frames) <= 0:
                return None
            gt = torch.stack(frames, dim=0)
        elif torch.is_tensor(gt_occ):
            gt = gt_occ
        else:
            return None
        while gt.dim() > 4:
            gt = gt[0]
        if gt.dim() != 4:
            return None
        gt = gt.to(device=device, dtype=torch.long)
        ids = [
            int(v) for v in getattr(self, "query_class_ids", ())
            if int(v) > 0 and int(v) != int(self.query_bg_class)
        ]
        if len(ids) <= 0:
            return (gt > 0) & (gt != 255)
        mask = torch.zeros_like(gt, dtype=torch.bool)
        for cls_id in ids:
            mask |= (gt == int(cls_id))
        return mask

    @staticmethod
    def _eval_unwrap_single(x):
        x = getattr(x, "data", x)
        while isinstance(x, (list, tuple)) and len(x) == 1:
            x = getattr(x[0], "data", x[0])
        return x

    def _eval_tensor_to_txyz(self, x):
        x = self._eval_unwrap_single(x)
        if isinstance(x, (list, tuple)):
            x = [getattr(v, "data", v) for v in x]
            if len(x) == 0 or not all(torch.is_tensor(v) for v in x):
                return None
            x = torch.stack(x, dim=0)
        if not torch.is_tensor(x):
            return None
        if x.dim() == 6 and int(x.shape[0]) == 1 and int(x.shape[2]) == 1:
            x = x[0, :, 0]
        elif x.dim() == 5 and int(x.shape[0]) == 1:
            x = x[0]
        elif x.dim() == 5 and int(x.shape[1]) == 1:
            x = x[:, 0]
        elif x.dim() == 4:
            pass
        else:
            return None
        return x.to(torch.long).contiguous()

    def _eval_bbox_cls_occ_txyz(self, segmentation_cls_instance3d=None):
        if segmentation_cls_instance3d is None:
            return None
        seg_cls = self._eval_unwrap_single(segmentation_cls_instance3d)
        if not torch.is_tensor(seg_cls):
            return None
        if seg_cls.dim() == 6 and int(seg_cls.shape[0]) == 1 and int(seg_cls.shape[2]) == 2:
            cls_txyz = seg_cls[0, :, 0].to(torch.long).contiguous()
        elif seg_cls.dim() == 5 and int(seg_cls.shape[1]) == 2:
            cls_txyz = seg_cls[:, 0].to(torch.long).contiguous()
        else:
            return None
        cls_txyz = cls_txyz.clone()
        cls_txyz[cls_txyz == 7] = 0
        return cls_txyz

    def _eval_bbox_occ_txyz(
            self,
            segmentation=None,
            segmentation_instance3d=None,
            gt_occ_inst=None,
            fallback_shape=None,
        ):
        out = self._eval_tensor_to_txyz(segmentation_instance3d)
        if torch.is_tensor(out):
            return out
        out = self._eval_tensor_to_txyz(segmentation)
        if torch.is_tensor(out):
            return out

        gt_occ_inst_bundle = self._prepare_gt_occ_inst_primary_targets(
            gt_occ_inst=gt_occ_inst,
            fallback_segmentation_instance3d_txyz=None,
        )
        if isinstance(gt_occ_inst_bundle, dict) and torch.is_tensor(gt_occ_inst_bundle.get("dense_inst_txyz", None)):
            return gt_occ_inst_bundle["dense_inst_txyz"].to(torch.long).contiguous()

        if fallback_shape is None:
            return None
        shape = tuple(int(v) for v in fallback_shape)
        while len(shape) > 4 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 4:
            return None
        return torch.zeros(shape, dtype=torch.long)

    def _eval_nusocc_3d_valid(self, gt_occ, device=None):
        if gt_occ is None:
            return None
        if isinstance(gt_occ, (list, tuple)):
            frames = [x for x in gt_occ if torch.is_tensor(x)]
            if len(frames) <= 0:
                return None
            gt = torch.stack(frames, dim=0)
        elif torch.is_tensor(gt_occ):
            gt = gt_occ
        else:
            return None
        while gt.dim() > 4:
            gt = gt[0]
        if gt.dim() != 4:
            return None
        gt = gt.to(device=device, dtype=torch.long)
        return (gt != 255) & (gt != 7)

    def _eval_future_tail(self, x):
        target_t = int(self.n_future_frames)
        if torch.is_tensor(x) and x.dim() > 0 and target_t > 0 and int(x.shape[0]) > target_t:
            return x[-target_t:].contiguous()
        return x

    @staticmethod
    def _eval_mode_is_present():
        import os
        return os.environ.get("EOCF_EVAL_MODE", "1").strip().lower() in ("0", "present")

    def _eval_mode_slice(self, x):
        if self._eval_mode_is_present():
            if torch.is_tensor(x) and x.dim() > 0:
                t = int(x.shape[0])
                pi = max(0, t - int(self.n_future_frames) - 1)
                return x[pi:pi + 1].contiguous()
            return x
        return self._eval_future_tail(x)

    def _build_eval_future_2d_bundle(self, present_bundle, centers_future_tq3, mix_future,
                                     conf_q=None, cls_ids_q=None):
        meta_only = {k: v for k, v in present_bundle.items() if str(k).startswith("_")}
        sel_idx = present_bundle.get("selected_query_idx_q", None)
        if (not torch.is_tensor(sel_idx)) or int(sel_idx.numel()) == 0:
            return meta_only
        if (not isinstance(mix_future, (tuple, list))) or len(mix_future) != 4:
            return meta_only
        mc, ms, my, mw = mix_future
        if not all(torch.is_tensor(t) for t in (centers_future_tq3, mc, ms, my, mw)):
            return meta_only
        sel = sel_idx.to(dtype=torch.long, device=centers_future_tq3.device)

        def _isel(x):
            return x.index_select(1, sel).detach() if torch.is_tensor(x) and x.dim() >= 2 else None

        pts_sel = _isel(centers_future_tq3)
        mc_s, ms_s, my_s, mw_s = _isel(mc), _isel(ms), _isel(my), _isel(mw)
        if pts_sel is None or mc_s is None:
            return meta_only
        score_q = present_bundle.get("selected_score_q", None)
        cls_q = present_bundle.get("selected_pred_cls_q", None)
        fb = dict(meta_only)
        fb["selected_points_tq3"] = pts_sel
        fb["selected_mixture_centers_tqg3"] = mc_s
        fb["selected_mixture_sigmas_tqg3"] = ms_s
        fb["selected_mixture_quat_tqg4"] = my_s
        fb["selected_mixture_weights_tqg"] = mw_s
        fb["selected_score_q"] = score_q
        fb["selected_pred_cls_q"] = cls_q
        cand_idx = None
        if torch.is_tensor(cls_ids_q) and torch.is_tensor(conf_q):
            cand_idx = (cls_ids_q.to(device=sel.device) != int(self.query_bg_class)).nonzero(as_tuple=False).reshape(-1)
        if torch.is_tensor(cand_idx) and int(cand_idx.numel()) > 0:
            def _ic(x):
                return x.index_select(1, cand_idx).detach() if torch.is_tensor(x) and x.dim() >= 2 else None
            fb["candidate_points_tq3"] = _ic(centers_future_tq3)
            fb["candidate_mixture_centers_tqg3"] = _ic(mc)
            fb["candidate_mixture_sigmas_tqg3"] = _ic(ms)
            fb["candidate_mixture_quat_tqg4"] = _ic(my)
            fb["candidate_mixture_weights_tqg"] = _ic(mw)
            fb["candidate_score_q"] = conf_q.index_select(0, cand_idx).detach()
            fb["candidate_pred_cls_q"] = cls_ids_q.index_select(0, cand_idx).detach()
        else:
            for k in ("points_tq3", "mixture_centers_tqg3", "mixture_sigmas_tqg3",
                      "mixture_quat_tqg4", "mixture_weights_tqg", "score_q", "pred_cls_q"):
                fb[f"candidate_{k}"] = fb[f"selected_{k}"]
        fb["selected_query_idx_q"] = sel
        fb["score_thr"] = 0.0
        for k in ("top_k", "w_iou", "w_cls", "w_cam", "distance_nms_radius_m"):
            if k in present_bundle:
                fb[k] = present_bundle[k]
        return fb

    def _eval_select_mixture_params(self, present_idx, mix_c, mix_s, mix_y, mix_w,
                                    centers_world, sel_idx):
        pi = int(present_idx)
        off_qg3 = mix_c[pi] - centers_world[pi].unsqueeze(1)
        off_sg3 = off_qg3.index_select(0, sel_idx).contiguous()
        sig_sg3 = mix_s[pi].index_select(0, sel_idx).contiguous()
        yaw_sg = mix_y[pi].index_select(0, sel_idx).contiguous()
        w_sg = mix_w[pi].index_select(0, sel_idx).contiguous()
        return off_sg3, sig_sg3, yaw_sg, w_sg

    def _eval_mixture_scene_occ(self, centers_frames_ts3, off_sg3, sig_sg3, yaw_sg, w_sg,
                                pair_w_ts):
        T = int(centers_frames_ts3.shape[0])
        cen_tsg3 = centers_frames_ts3.unsqueeze(2) + off_sg3.unsqueeze(0)
        sig_tsg3 = sig_sg3.unsqueeze(0).expand(T, -1, -1, -1)
        yaw_tsg = yaw_sg.unsqueeze(0).expand(T, -1, -1, -1)
        w_tsg = w_sg.unsqueeze(0).expand(T, -1, -1)
        return self.voxelizer.forward_gaussian_mixture_scene(
            cen_tsg3, sig_tsg3, w_tsg, yaw_tsg, pair_weights_tq=pair_w_ts)


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
        gt_inst_center_world_tn3, gt_inst_center_valid_tn, gt_inst_ids_n = self._prepare_gt_instance_centers_for_trajectory_matching(
            gt_instance_centers_world=gt_instance_centers_world,
            gt_instance_centers_valid=gt_instance_centers_valid,
            gt_instance_ids=gt_instance_ids,
        )
        if gt_inst_center_world_tn3 is None and isinstance(gt_occ_inst_bundle, dict):
            gt_inst_center_world_tn3, gt_inst_center_valid_tn, gt_inst_ids_n = self._prepare_gt_instance_centers_for_trajectory_matching(
                gt_instance_centers_world=gt_occ_inst_bundle.get("centers_world_tn3", None),
                gt_instance_centers_valid=gt_occ_inst_bundle.get("centers_valid_tn", None),
                gt_instance_ids=gt_occ_inst_bundle.get("instance_ids_n", None),
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
        if bool(getattr(self, "query_require_history_all_valid", False)):
            history_all_valid_ids_n = self._build_history_all_valid_instance_ids(
                gt_instance_centers_valid=gt_inst_center_valid_hist_tn,
                gt_instance_ids=gt_inst_ids_hist_n,
                time_receptive_field=int(self.time_receptive_field),
            )
            if torch.is_tensor(history_all_valid_ids_n):
                gt_inst_center_world_tn3, gt_inst_center_valid_tn, gt_inst_ids_n = self._filter_instance_targets_by_intersection_ids(
                    centers_world_tn3=gt_inst_center_world_tn3,
                    centers_valid_tn=gt_inst_center_valid_tn,
                    instance_ids_n=gt_inst_ids_n,
                    intersection_ids_n=history_all_valid_ids_n,
                )
                gt_inst_center_world_hist_tn3, gt_inst_center_valid_hist_tn, gt_inst_ids_hist_n = self._filter_instance_targets_by_intersection_ids(
                    centers_world_tn3=gt_inst_center_world_hist_tn3,
                    centers_valid_tn=gt_inst_center_valid_hist_tn,
                    instance_ids_n=gt_inst_ids_hist_n,
                    intersection_ids_n=history_all_valid_ids_n,
                )
                gt_instance_occ3d_txyz_primary = self._filter_dense_instance_ids(
                    dense_gt=gt_instance_occ3d_txyz_primary,
                    keep_instance_ids=history_all_valid_ids_n,
                )
                gt_segmentation_instance3d_txyz = self._filter_dense_instance_ids(
                    dense_gt=gt_segmentation_instance3d_txyz,
                    keep_instance_ids=history_all_valid_ids_n,
                )
                gt_segmentation_cls_instance3d_for_match = self._filter_dense_instance_ids(
                    dense_gt=gt_segmentation_cls_instance3d_for_match,
                    keep_instance_ids=history_all_valid_ids_n,
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
        gt_segmentation_cls_instance3d_for_match_query_vis, _, _ = (
            self._select_query_visualization_gt_slice(gt_segmentation_cls_instance3d_for_match)
        )
        self._last_query_gt_temporal_local_idx = int(gt_query_local_idx) if gt_query_local_idx is not None else -1
        self._last_query_gt_temporal_source_layout = str(gt_query_source_layout)
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
            _query_attn_weights_tqnhw,
            _query_attn_bbox_targets,
            instance_img_debug_bundle,
            query_match_inputs,
            _traj_offsets_fq2,
            _query_present_local_idx,
            mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3,
            mixture_quat_tqg4,
            mixture_weights_tqg,
            gaussian_sigmas_world_traj_tq3,
            mixture_centers_world_traj_tqg3,
            mixture_sigmas_world_traj_tqg3,
            mixture_quat_traj_tqg,
            mixture_weights_traj_tqg,
            traj_motion_input_tqd2,
            query_size_dbg,
        ) = self.extract_feat_query(
            img_inputs_seq=img_inputs_seq,
            img_metas=img_metas,
            future_egomotion=future_egomotion,
            gt_segmentation_instance3d_txyz_for_attn=gt_instance_occ3d_txyz_primary,
            fallback_segmentation_instance3d_txyz_for_attn=gt_segmentation_instance3d_txyz,
            voxelize=False,
            return_instance_img_debug=need_instance_img_debug,
            return_instance_img_debug_fullseq=need_instance_img_debug_fullseq,
            return_query_cam_gaussian_vis_debug=need_query_cam_gaussian_vis,
            return_query_match_inputs=True,
            defer_trajectory=bool(getattr(self, "query_traj_matched_only", False)),
        )
        # Keep a minimal default loss path until full query-loss branches are enabled.
        _pick = self._pick_tensor
        centers_world_loss = _pick(centers_world_traj_tq3, centers_world)
        gaussian_sigmas_world_loss = _pick(gaussian_sigmas_world_traj_tq3, gaussian_sigmas_world)
        mixture_centers_world_loss = _pick(mixture_centers_world_traj_tqg3, mixture_centers_world_tqg3)
        mixture_sigmas_world_loss = _pick(mixture_sigmas_world_traj_tqg3, mixture_sigmas_world_tqg3)
        mixture_quat_loss = _pick(mixture_quat_traj_tqg, mixture_quat_tqg4)
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
        mixture_quat_vis_tqg = mixture_quat_loss
        mixture_weights_vis_tqg = mixture_weights_loss
        centers_world_loss_aligned_tq3 = centers_world_temporal_sup_tq3
        gaussian_sigmas_world_loss_aligned_tq3 = gaussian_sigmas_world_loss
        mixture_centers_world_loss_aligned_tqg3 = mixture_centers_world_loss
        mixture_sigmas_world_loss_aligned_tqg3 = mixture_sigmas_world_loss
        mixture_quat_loss_aligned_tqg = mixture_quat_loss
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
                mixture_sigmas_world_loss, mixture_quat_loss, mixture_weights_loss,
            )
            centers_world_loss_aligned_tq3 = loss_geom_pack.get("centers_world_tq3", centers_world_loss_aligned_tq3)
            gaussian_sigmas_world_loss_aligned_tq3 = loss_geom_pack.get("gaussian_sigmas_world_tq3", gaussian_sigmas_world_loss_aligned_tq3)
            mixture_centers_world_loss_aligned_tqg3 = loss_geom_pack.get("mixture_centers_world_tqg3", mixture_centers_world_loss_aligned_tqg3)
            mixture_sigmas_world_loss_aligned_tqg3 = loss_geom_pack.get("mixture_sigmas_world_tqg3", mixture_sigmas_world_loss_aligned_tqg3)
            mixture_quat_loss_aligned_tqg = loss_geom_pack.get("mixture_quat_tqg4", mixture_quat_loss_aligned_tqg)
            mixture_weights_loss_aligned_tqg = loss_geom_pack.get("mixture_weights_tqg", mixture_weights_loss_aligned_tqg)
        if _needs_ego_align:
            with torch.no_grad():
                vis_geom_pack = self._align_geom_pack(
                    centers_world_loss, traj_frame_indices_t, future_egomotion,
                    gaussian_sigmas_world_loss, mixture_centers_world_loss,
                    mixture_sigmas_world_loss, mixture_quat_loss, mixture_weights_loss,
                    detach=True,
                )
            centers_world_vis_tq3 = vis_geom_pack.get("centers_world_tq3", centers_world_vis_tq3)
            gaussian_sigmas_world_vis_tq3 = vis_geom_pack.get("gaussian_sigmas_world_tq3", gaussian_sigmas_world_vis_tq3)
            mixture_centers_world_vis_tqg3 = vis_geom_pack.get("mixture_centers_world_tqg3", mixture_centers_world_vis_tqg3)
            mixture_sigmas_world_vis_tqg3 = vis_geom_pack.get("mixture_sigmas_world_tqg3", mixture_sigmas_world_vis_tqg3)
            mixture_quat_vis_tqg = vis_geom_pack.get("mixture_quat_tqg4", mixture_quat_vis_tqg)
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
        mixture_quat_vis_full_tqg = mixture_quat_vis_tqg
        mixture_weights_vis_full_tqg = mixture_weights_vis_tqg
        centers_world_loss_full_tq3 = centers_world_loss_aligned_tq3
        gaussian_sigmas_world_loss_full_tq3 = gaussian_sigmas_world_loss_aligned_tq3
        mixture_centers_world_loss_full_tqg3 = mixture_centers_world_loss_aligned_tqg3
        mixture_sigmas_world_loss_full_tqg3 = mixture_sigmas_world_loss_aligned_tqg3
        mixture_quat_loss_full_tqg = mixture_quat_loss_aligned_tqg
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
            if torch.is_tensor(gaussian_sigmas_world) and torch.is_tensor(gaussian_sigmas_world_vis_tq3):
                gaussian_sigmas_world_vis_full_tq3 = _ppd(gaussian_sigmas_world, gaussian_sigmas_world_vis_tq3)
            mixture_centers_world_vis_full_tqg3 = _ppd(mixture_centers_world_tqg3, mixture_centers_world_vis_tqg3)
            mixture_sigmas_world_vis_full_tqg3 = _ppd(mixture_sigmas_world_tqg3, mixture_sigmas_world_vis_tqg3)
            mixture_quat_vis_full_tqg = _ppd(mixture_quat_tqg4, mixture_quat_vis_tqg)
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
            if torch.is_tensor(gaussian_sigmas_world) and torch.is_tensor(gaussian_sigmas_world_loss_aligned_tq3):
                gaussian_sigmas_world_loss_full_tq3 = _pp(gaussian_sigmas_world, gaussian_sigmas_world_loss_aligned_tq3)
            mixture_centers_world_loss_full_tqg3 = _pp(mixture_centers_world_tqg3, mixture_centers_world_loss_aligned_tqg3)
            mixture_sigmas_world_loss_full_tqg3 = _pp(mixture_sigmas_world_tqg3, mixture_sigmas_world_loss_aligned_tqg3)
            mixture_quat_loss_full_tqg = _pp(mixture_quat_tqg4, mixture_quat_loss_aligned_tqg)
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
        mixture_quat_match_tqg = mixture_quat_loss
        mixture_weights_match_tqg = mixture_weights_loss
        match_gt_instance_occ3d_txyz = gt_instance_occ3d_txyz_query
        match_bbox_instance_occ3d_txyz = gt_segmentation_instance3d_txyz_query
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
            mixture_quat_match_tqg = mixture_quat_loss_full_tqg
            mixture_weights_match_tqg = mixture_weights_loss_full_tqg
            match_gt_instance_occ3d_txyz = gt_instance_occ3d_txyz_query_vis
            match_bbox_instance_occ3d_txyz = gt_segmentation_instance3d_txyz_query_vis
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

        self._last_gt_instance_bev_feat_cache = None
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

        inst_match_result = self._match_queries_to_gt_instances(
            centers_world=centers_world_match_tq3,
            query_sigmas_world_tq3=gaussian_sigmas_world_match_tq3,
            query_mixture_centers_world_tqg3=mixture_centers_world_match_tqg3,
            query_mixture_sigmas_world_tqg3=mixture_sigmas_world_match_tqg3,
            query_mixture_quat_tqg4=mixture_quat_match_tqg,
            query_mixture_weights_tqg=mixture_weights_match_tqg,
            gt_instance_occ3d_txyz=match_gt_instance_occ3d_txyz,
            gt_inst_center_world_tn3=match_gt_centers_tn3,
            gt_inst_center_valid_tn=match_gt_valid_tn,
            gt_inst_ids_n=match_gt_ids_n,
            gt_inst_cls_n=match_gt_cls_n,
            gt_inst_cls_valid_n=match_gt_cls_valid_n,
            query_cls_logits_qc=query_cls_logits_qc,
            query_attn_weights_tqnhw=_query_attn_weights_tqnhw,
            gt_attn_targets=_query_attn_bbox_targets,
            query_cls_cost_weight=self.query_cls_match_cost_weight,
            query_bev_iou_cost_weight=self.query_bev_iou_match_cost_weight,
            query_attn_match_cost_weight=self.query_attn_match_cost_weight,
            query_attn_match_metric=self.query_attn_match_metric,
            query_attn_match_pred_norm=self.query_attn_match_pred_norm,
            query_attn_match_eps=self.query_attn_match_eps,
            query_center_match_frame_idx=match_center_frame_idx,
            query_temporal_cost_frame_indices=match_temporal_cost_frame_indices,
            query_temporal_offset_match_cost_weight=self.query_temporal_offset_match_cost_weight,
        )
        self._last_inst_match_result = inst_match_result
        if isinstance(inst_match_result, dict):
            matched_query_idx = inst_match_result.get("matched_query_idx", None)
            if torch.is_tensor(matched_query_idx):
                num_matched_queries = int(matched_query_idx.numel())
        # Deferred trajectory prediction: run traj head only on matched queries,
        # scatter results back to [F, Q=100, 2] with zeros for unmatched.
        if bool(getattr(self, "query_traj_matched_only", False)) and torch.is_tensor(traj_motion_input_tqd2):
            _n_future = int(self.n_future_frames)
            _q_total = int(centers_world.shape[1]) if torch.is_tensor(centers_world) else 0
            _mqi = inst_match_result.get("matched_query_idx", None) if isinstance(inst_match_result, dict) else None
            if torch.is_tensor(_mqi) and _mqi.numel() > 0 and _q_total > 0:
                _motion_skd = traj_motion_input_tqd2.index_select(1, _mqi)
                # Teacher forcing: replace past delta priors with GT-derived offsets.
                if (
                    self.training
                    and bool(getattr(self, "query_traj_teacher_forcing", False))
                    and (
                        int(getattr(self, "query_traj_teacher_forcing_until_iter", 0)) <= 0
                        or int(getattr(self.query_head, "_train_iter", 0))
                        < int(self.query_traj_teacher_forcing_until_iter)
                    )
                    and isinstance(inst_match_result, dict)
                ):
                    _mii = inst_match_result.get("matched_inst_idx", None)
                    _gt_c = inst_match_result.get("gt_centers_tn3", None)
                    _gt_v = inst_match_result.get("gt_valid_tn", None)
                    _past_steps = int(_query_present_local_idx)
                    if (
                        torch.is_tensor(_mii) and _mii.numel() > 0
                        and torch.is_tensor(_gt_c) and _gt_c.dim() == 3
                        and torch.is_tensor(_gt_v) and _gt_v.dim() == 2
                        and int(_gt_c.shape[0]) > _past_steps
                        and _past_steps > 0
                    ):
                        _gt_xy = _gt_c[:_past_steps + 1, :, :2].to(
                            device=_motion_skd.device, dtype=torch.float32
                        )
                        _gt_delta = _gt_xy[1:_past_steps + 1] - _gt_xy[:_past_steps]  # [past_steps, N, 2]
                        _gt_delta_k = _gt_delta.index_select(1, _mii)  # [past_steps, K, 2]
                        _gt_v_past = _gt_v[:_past_steps + 1].to(
                            device=_motion_skd.device, dtype=torch.bool
                        )
                        _valid_k = (_gt_v_past[:_past_steps].index_select(1, _mii)
                                    & _gt_v_past[1:_past_steps + 1].index_select(1, _mii))
                        _traj_max = _motion_skd.new_tensor(self.query_traj_residual_max_m).view(1, 1, 2)
                        _gt_delta_norm = (_gt_delta_k / _traj_max.clamp_min(1e-6)).clamp(-1.0, 1.0)
                        _motion_skd = _motion_skd.clone()
                        _motion_skd[:_past_steps, :, -2:] = 0.0
                        for _t in range(_past_steps):
                            _valid_t = _valid_k[_t]  # [K]
                            _motion_skd[_t, _valid_t, -2:] = _gt_delta_norm[_t, _valid_t]
                _feat_tkd = _query_future_feat_tqd.index_select(1, _mqi)
                _, _traj_fk2 = self.query_head._predict_trajectory_from_inputs(_motion_skd, _feat_tkd)
                _traj_offsets_fq2 = traj_motion_input_tqd2.new_zeros(_n_future, _q_total, 2)
                _traj_offsets_fq2.index_copy_(1, _mqi, _traj_fk2)
            elif _q_total > 0:
                _traj_offsets_fq2 = traj_motion_input_tqd2.new_zeros(_n_future, _q_total, 2)

            # Rebuild vis geometry with real (matched) traj offsets —
            # must update all positional vis tensors (centers + mixture) so that
            # Gaussian positions stay consistent with query center positions.
            if (
                torch.is_tensor(_traj_offsets_fq2)
                and torch.is_tensor(centers_world)
                and int(centers_world.shape[0]) > _query_present_local_idx
            ):
                def _safe_narrow(t, idx):
                    return (
                        t.narrow(0, idx, 1).contiguous()
                        if torch.is_tensor(t) and t.dim() >= 3 and int(t.shape[0]) > idx
                        else None
                    )
                _present_c  = _safe_narrow(centers_world, _query_present_local_idx)
                _present_sg = _safe_narrow(gaussian_sigmas_world, _query_present_local_idx)
                _present_mc = _safe_narrow(mixture_centers_world_tqg3, _query_present_local_idx)
                _present_ms = _safe_narrow(mixture_sigmas_world_tqg3, _query_present_local_idx)
                _present_my = _safe_narrow(mixture_quat_tqg4, _query_present_local_idx)
                _present_mw = _safe_narrow(mixture_weights_tqg, _query_present_local_idx)
                _traj_geom_real = self._build_query_trajectory_geometry_from_present(
                    present_centers_world_tq3=_present_c,
                    present_sigmas_world_tq3=None,
                    present_mix_centers_world_tqg3=_present_mc,
                    present_mix_sigmas_world_tqg3=_present_ms,
                    present_mix_quat_tqg4=_present_my,
                    present_mix_weights_tqg=_present_mw,
                    traj_offsets_fq2=_traj_offsets_fq2,
                )
                _ctraj_real = _traj_geom_real.get("centers_world_tq3", None)
                if torch.is_tensor(_ctraj_real):
                    if not traj_centers_are_present_aligned:
                        _ctraj_real = self._align_query_centers_to_present_frame(
                            centers_world_tq3=_ctraj_real,
                            future_egomotion=future_egomotion,
                            traj_frame_indices_t=traj_frame_indices_t,
                        )
                    if traj_vis_has_present_anchor:
                        _ppd_v = lambda p, t: self._prepend_past_frames(
                            p, past_only_count, t, detach=True
                        )
                        centers_world_vis_full_tq3 = _ppd_v(centers_world, _ctraj_real)
                        _sg = _traj_geom_real.get("sigmas_world_tq3", None)
                        if torch.is_tensor(_sg):
                            gaussian_sigmas_world_vis_full_tq3 = _ppd_v(gaussian_sigmas_world, _sg)
                        _mc = _traj_geom_real.get("mix_centers_world_tqg3", None)
                        if torch.is_tensor(_mc):
                            mixture_centers_world_vis_full_tqg3 = _ppd_v(mixture_centers_world_tqg3, _mc)
                        _ms = _traj_geom_real.get("mix_sigmas_world_tqg3", None)
                        if torch.is_tensor(_ms):
                            mixture_sigmas_world_vis_full_tqg3 = _ppd_v(mixture_sigmas_world_tqg3, _ms)
                        _my = _traj_geom_real.get("mix_quat_tqg4", None)
                        if torch.is_tensor(_my):
                            mixture_quat_vis_full_tqg = _ppd_v(mixture_quat_tqg4, _my)
                        _mw = _traj_geom_real.get("mix_weights_tqg", None)
                        if torch.is_tensor(_mw):
                            mixture_weights_vis_full_tqg = _ppd_v(mixture_weights_tqg, _mw)

        query_cls_loss = self._compute_query_cls_loss(
            query_cls_logits_qc=query_cls_logits_qc,
            inst_match_result=inst_match_result,
            loss_weight=float(self.query_cls_loss_weight),
            bg_index=int(self.query_bg_class),
            class_weights=self.query_cls_loss_class_weights,
            centers_world_tq3=centers_world_match_tq3,
            nms_radius_m=float(self.debug_query_distance_nms_radius_m),
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
            and torch.is_tensor(mixture_quat_loss)
            and torch.is_tensor(mixture_weights_loss)
        ):
            with torch.no_grad():
                query_attn_cam_targets = self.build_query_attn_cam_gaussian_targets(
                    mixture_centers_world_tqg3=mixture_centers_world_loss,
                    mixture_sigmas_world_tqg3=mixture_sigmas_world_loss,
                    mixture_quat_tqg4=mixture_quat_loss,
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
            and torch.is_tensor(mixture_quat_match_tqg)
            and torch.is_tensor(mixture_weights_match_tqg)
            and torch.is_tensor(match_gt_instance_occ3d_txyz)
        ):
            matched_gmo_loss = self._compute_matched_pair_gmo_losses(
                centers_world_tq3=centers_world_match_tq3,
                sigmas_world_tq3=gaussian_sigmas_world_match_tq3,
                mixture_centers_world_tqg3=mixture_centers_world_match_tqg3,
                mixture_sigmas_world_tqg3=mixture_sigmas_world_match_tqg3,
                mixture_quat_tqg4=mixture_quat_match_tqg,
                mixture_weights_tqg=mixture_weights_match_tqg,
                gt_instance_occ3d_txyz_pred=match_gt_instance_occ3d_txyz,
                bbox_instance_occ3d_txyz=match_bbox_instance_occ3d_txyz,
                objectness_scores_tq=None,
                inst_match_result=inst_match_result,
                loss_weight=float(self.query_gmo_loss_weight),
                loss_type=self.query_gmo_loss_type,
                shape_loss_mode=self.query_gmo_shape_loss_mode,
                local_crop_size=self.query_gmo_local_crop_size,
                local_crop_scale=float(self.query_gmo_local_crop_scale),
                local_crop_margin_vox=self.query_gmo_local_crop_margin_vox,
                focal_gamma=float(self.query_gmo_focal_gamma),
                focal_alpha=float(self.query_gmo_focal_alpha),
                compute_dice=bool(self.use_query_gmo_dice_loss),
                dice_loss_weight=float(self.query_gmo_dice_loss_weight),
                tversky_alpha=float(self.query_gmo_tversky_alpha),
                tversky_beta=float(self.query_gmo_tversky_beta),
                pair_chunk_size=int(self.query_multi_gaussian_pair_chunk),
                quality_filter_enabled=bool(self.use_query_gmo_quality_filter),
                quality_roi_radius_m=float(self.query_gmo_quality_roi_radius_m),
                quality_min_occupancy_ratio=float(self.query_gmo_quality_min_occupancy_ratio),
                quality_present_idx=int(traj_loss_present_local_idx),
            )
        self.maybe_save_gmo_quality_alignment_vis(
            fine_instance_occ3d_txyz=match_gt_instance_occ3d_txyz,
            bbox_instance_occ3d_txyz=match_bbox_instance_occ3d_txyz,
            inst_match_result=inst_match_result,
            img_metas=img_metas,
            step=cur_train_iter,
            present_idx=int(traj_loss_present_local_idx),
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
                gt_instance_centers_full_tn3=_d(gt_inst_center_world_full_tn3),
                gt_instance_valid_full_tn=_d(gt_inst_center_valid_full_tn),
                gt_instance_ids_full_n=_d(gt_inst_ids_full_n),
                gaussian_sigmas_world_tq3=_d(gaussian_sigmas_world_vis_full_tq3),
                mixture_centers_world_tqg3=_d(mixture_centers_world_vis_full_tqg3),
                mixture_sigmas_world_tqg3=_d(mixture_sigmas_world_vis_full_tqg3),
                mixture_quat_tqg4=_d(mixture_quat_vis_full_tqg),
                mixture_weights_tqg=_d(mixture_weights_vis_full_tqg),
                query_attn_cam_score_pack=query_attn_cam_score_pack,
            )

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
        # self.maybe_save_query_inst_depth_lift_vis(
        #     img_inputs_seq=img_inputs_seq,
        #     query_match_inputs=query_match_inputs,
        #     query_inst_depth_target_pack=query_inst_depth_target_pack,
        #     gt_segmentation_instance3d_txyz_fine=gt_instance_occ3d_txyz_primary,
        #     gt_segmentation_instance3d_txyz_bbox=gt_segmentation_instance3d_txyz,
        #     future_egomotion=future_egomotion,
        #     img_metas=img_metas,
        #     step=cur_train_iter,
        # )
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
            query_cls_loss=query_cls_loss,
            query_depth_loss=query_depth_loss,
            query_attn_bbox_loss=query_attn_bbox_loss,
            matched_gmo_loss=matched_gmo_loss,
            center_match_loss=center_match_loss,
            query_traj_loss=query_traj_loss,
            query_attn_cam_score_pack=query_attn_cam_score_pack,
            query_size_dbg=query_size_dbg,
            centers_world=centers_world,
            centers_world_match_tq3=centers_world_match_tq3,
            centers_world_dt_gt2p_tq3=centers_world_dt_gt2p_tq3,
            gaussian_sigmas_world_loss=gaussian_sigmas_world_loss,
            mixture_centers_world_loss=mixture_centers_world_loss,
            mixture_sigmas_world_loss=mixture_sigmas_world_loss,
            mixture_quat_loss=mixture_quat_loss,
            mixture_weights_loss=mixture_weights_loss,
            occ_dt=occ_dt,
            gt_inst_center_world_tn3=gt_inst_center_world_tn3,
            gt_inst_center_valid_tn=gt_inst_center_valid_tn,
            inst_match_result=inst_match_result,
        )
