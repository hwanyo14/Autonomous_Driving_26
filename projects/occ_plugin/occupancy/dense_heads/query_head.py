import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from .utils_query_vis import (
    is_main_process, VisConfig,
    draw_marker_splats, draw_cross_marker, draw_gaussian_bev_footprints,
    get_query_vis_palette, draw_query_vis_legend, draw_generic_vis_legend,
    normalize_gt_occ_semantic_for_vis, normalize_gt_occ_inst_for_vis,
    save_prob_grid_vis,
)

def sigmoid_to_world_from_range(logits: torch.Tensor,
                                point_cloud_range,
                                spatial_extent3d) -> torch.Tensor:
    """
    logits: (..., 3)  (unconstrained)
    point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    spatial_extent3d:  (x_range, y_range, z_range) = (x_max-x_min, y_max-y_min, z_max-z_min)

    return: (..., 3) world coords in the same unit as point_cloud_range (meters)
    """
    # make tensors on correct device/dtype
    pc_min = torch.tensor(point_cloud_range[:3], device=logits.device, dtype=logits.dtype)          # [3]
    extent = torch.tensor(spatial_extent3d, device=logits.device, dtype=logits.dtype)              # [3]

    return pc_min + torch.sigmoid(logits) * extent


def sigmoid_to_sigma_from_range(logits: torch.Tensor,
                                sigma_min=(0.15, 0.15, 0.10),
                                sigma_max=(4.0, 4.0, 1.5)) -> torch.Tensor:
    """
    logits: (..., 3) unconstrained
    sigma_min/max: axis-wise sigma bounds in meters

    return: (..., 3) sigma in meters, range [sigma_min, sigma_max]
    """
    smin = torch.tensor(sigma_min, device=logits.device, dtype=logits.dtype)  # [3]
    smax = torch.tensor(sigma_max, device=logits.device, dtype=logits.dtype)  # [3]
    return smin + torch.sigmoid(logits) * (smax - smin)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0, use_ln=True):
        super().__init__()
        assert num_layers >= 1
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(in_dim) if use_ln else nn.Identity()

    def forward(self, x):
        x = self.ln(x)
        return self.net(x)


class CenterHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 out_dim=3):
        super(CenterHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.mlp = MLP(in_dim=self.embed_dim,
                       hidden_dim=self.embed_dim,
                       out_dim=self.out_dim,
                       num_layers=2,
                       dropout=0.0,
                       use_ln=True)

    def forward(self, query_inst):
        center_logits = self.mlp(query_inst)  # [T, num_queries, out_dim]
        return center_logits
    

class GaussianHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 out_dim=3):
        super(GaussianHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.mlp = MLP(in_dim=self.embed_dim,
                       hidden_dim=self.embed_dim,
                       out_dim=self.out_dim,
                       num_layers=2,
                       dropout=0.0,
                       use_ln=True)

    def forward(self, query_inst):
        gaussian_logits = self.mlp(query_inst)  # [T, num_queries, 3]
        return gaussian_logits


class ClassificationHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 out_dim=3):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.mlp = MLP(in_dim=self.embed_dim,
                       hidden_dim=self.embed_dim,
                       out_dim=self.out_dim,
                       num_layers=2,
                       dropout=0.0,
                       use_ln=True)

    def forward(self, query_feat_qd):
        cls_logits = self.mlp(query_feat_qd)  # [Q, num_classes]
        return cls_logits


class QueryDepthHead(nn.Module):
    def __init__(self, embed_dim=128, out_dim=64):
        super(QueryDepthHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = int(out_dim)
        if self.out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {self.out_dim}")
        self.mlp = MLP(
            in_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            out_dim=self.out_dim,
            num_layers=2,
            dropout=0.0,
            use_ln=True,
        )

    def forward(self, query_feat_tqd):
        # [T,Q,D] -> [T,Q,Dbin]
        return self.mlp(query_feat_tqd)
    

class TrajectoryHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, out_dim=8):
        super(TrajectoryHead, self).__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {self.out_dim}")
        self.mlp = MLP(
            in_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            num_layers=2,
            dropout=0.0,
            use_ln=True,
        )

    def forward(self, query_feat_qd):
        # [Q,D] -> [Q,out_dim]
        return self.mlp(query_feat_qd)


class QueryHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_past_frames=3,
                 num_future_frames=6,
                 num_overlap_frames=0,
                 query_present_only=False,
                 query_pred_num_frames=None,
                 query_traj_num_steps=0,
                 query_traj_residual_max_m=(8.0, 8.0),
                 query_traj_prior_detach=True,
                 num_query_classes=3,
                 query_class_ids=None,
                 query_class_names=None,
                 center_out_dim=3,
                 offset_out_dim=3,
                 num_pcd=256,
                 gaussian_sigma_min=(0.15, 0.15, 0.10),
                 gaussian_sigma_max=(4.0, 4.0, 1.5),
                 query_num_gaussians=1,
                 query_multi_gaussian_offset_max_m=(6.0, 6.0, 2.0),
                 query_multi_gaussian_sigma_min_m=(0.15, 0.15, 0.10),
                 query_multi_gaussian_sigma_max_m=(4.0, 4.0, 1.5),
                 query_multi_gaussian_sigma_reg_loss_weight=0.0,
                 query_multi_gaussian_sigma_reg_log_eps=1e-6,
                 query_multi_gaussian_weight_mode="softmax",
                 query_multi_gaussian_softplus_bias_init=-2.0,
                 query_multi_gaussian_weight_reg_loss_weight=1e-3,
                 query_multi_gaussian_weight_reg_target_sum=1.0,
                 gaussian_truncate_sigma=3.0,
                 point_cloud_range=None,
                 spatial_extent3d=None,
                 query_depth_num_bins=64,
                 query_feat_cosine_threshold=0.30,
                 query_center_distance_threshold_m=1.50,
                 detach_query_for_center_default=False,
                 debug_vis_every=0,
                 debug_vis_dir="./work_dirs/query_debug_vis",
                 center_only_mode=False,
                 debug_query_center_marker_radius=3,
                 debug_query_confidence_vis_threshold=0.5,
                 debug_query_objectness_vis_threshold=None,
                 debug_query_gaussian_vis_mode="ellipse",
                 debug_query_gaussian_prob_threshold=0.5,
                 debug_query_gaussian_prob_alpha_scale=4.0):
        super(QueryHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.num_overlap_frames = int(num_overlap_frames)
        self.query_present_only = bool(query_present_only)
        # In present-only mode, query_pred_num_frames is overridden to num_past_frames:
        # the projection module is removed and query_inst (shape [T_past, Q, D])
        # is consumed directly so depth / 2D soft-argmax / 3D lifting can run on
        # every past frame. The user-supplied query_pred_num_frames (legacy) is
        # accepted for config compatibility but ignored in present-only mode.
        if self.query_present_only:
            self.query_pred_num_frames = int(num_past_frames)
        elif query_pred_num_frames is None:
            self.query_pred_num_frames = int(num_future_frames)
        else:
            self.query_pred_num_frames = int(query_pred_num_frames)
        if self.query_pred_num_frames <= 0:
            raise ValueError(
                f"query_pred_num_frames must be >= 1, got {self.query_pred_num_frames}"
            )
        self.query_traj_num_steps = int(query_traj_num_steps)
        if self.query_traj_num_steps < 0:
            raise ValueError(
                f"query_traj_num_steps must be >= 0, got {self.query_traj_num_steps}"
            )
        self.query_traj_residual_max_m = tuple(float(v) for v in query_traj_residual_max_m)
        self.query_traj_prior_detach = bool(query_traj_prior_detach)
        if len(self.query_traj_residual_max_m) != 2:
            raise ValueError(
                "query_traj_residual_max_m must be (x,y), "
                f"got {self.query_traj_residual_max_m}"
            )
        if self.query_traj_num_steps > 0:
            if (self.query_traj_residual_max_m[0] <= 0.0) or (self.query_traj_residual_max_m[1] <= 0.0):
                raise ValueError(
                    "query_traj_residual_max_m must be positive when query_traj_num_steps > 0, "
                    f"got {self.query_traj_residual_max_m}"
                )
        self.num_query_classes = int(num_query_classes)
        if query_class_ids is None:
            query_class_ids = tuple(range(self.num_query_classes))
        else:
            query_class_ids = tuple(int(v) for v in query_class_ids)
        if len(query_class_ids) != self.num_query_classes:
            raise ValueError(
                "query_class_ids length must match num_query_classes, "
                f"got len(query_class_ids)={len(query_class_ids)} vs num_query_classes={self.num_query_classes}"
            )
        if query_class_names is None:
            query_class_names = ["background"] + [
                f"class_{int(raw_id)}" for raw_id in query_class_ids[1:]
            ]
        else:
            query_class_names = list(query_class_names)
            if len(query_class_names) == (self.num_query_classes - 1):
                query_class_names = ["background"] + query_class_names
        if len(query_class_names) != self.num_query_classes:
            raise ValueError(
                "query_class_names length must match num_query_classes "
                "(or exclude background so it can be prepended automatically), "
                f"got len(query_class_names)={len(query_class_names)} vs num_query_classes={self.num_query_classes}"
            )
        self.query_class_ids = tuple(int(v) for v in query_class_ids)
        self.query_class_names = tuple(str(v) for v in query_class_names)
        self.point_cloud_range = point_cloud_range
        self.spatial_extent3d = spatial_extent3d
        self.query_depth_num_bins = int(query_depth_num_bins)
        if self.query_depth_num_bins <= 0:
            raise ValueError(
                f"query_depth_num_bins must be positive, got {self.query_depth_num_bins}"
            )
        self.num_pcd = num_pcd
        self.gaussian_sigma_min = tuple(float(v) for v in gaussian_sigma_min)
        self.gaussian_sigma_max = tuple(float(v) for v in gaussian_sigma_max)
        self.query_num_gaussians = int(query_num_gaussians)
        if self.query_num_gaussians <= 0:
            raise ValueError(f"query_num_gaussians must be >= 1, got {self.query_num_gaussians}")
        self.query_multi_gaussian_offset_max_m = tuple(float(v) for v in query_multi_gaussian_offset_max_m)
        self.query_multi_gaussian_sigma_min_m = tuple(float(v) for v in query_multi_gaussian_sigma_min_m)
        self.query_multi_gaussian_sigma_max_m = tuple(float(v) for v in query_multi_gaussian_sigma_max_m)
        self.query_multi_gaussian_sigma_reg_loss_weight = float(query_multi_gaussian_sigma_reg_loss_weight)
        self.query_multi_gaussian_sigma_reg_log_eps = float(query_multi_gaussian_sigma_reg_log_eps)
        if len(self.query_multi_gaussian_offset_max_m) != 3:
            raise ValueError(
                "query_multi_gaussian_offset_max_m must be xyz tuple, "
                f"got {self.query_multi_gaussian_offset_max_m}"
            )
        if len(self.query_multi_gaussian_sigma_min_m) != 3 or len(self.query_multi_gaussian_sigma_max_m) != 3:
            raise ValueError(
                "query_multi_gaussian_sigma_min_m/max_m must be xyz tuples, "
                f"got min={self.query_multi_gaussian_sigma_min_m}, max={self.query_multi_gaussian_sigma_max_m}"
            )
        if self.query_multi_gaussian_sigma_reg_loss_weight < 0.0:
            raise ValueError(
                "query_multi_gaussian_sigma_reg_loss_weight must be >= 0, "
                f"got {self.query_multi_gaussian_sigma_reg_loss_weight}"
            )
        if self.query_multi_gaussian_sigma_reg_log_eps <= 0.0:
            raise ValueError(
                "query_multi_gaussian_sigma_reg_log_eps must be > 0, "
                f"got {self.query_multi_gaussian_sigma_reg_log_eps}"
            )
        self.query_multi_gaussian_weight_mode = str(query_multi_gaussian_weight_mode).lower()
        if self.query_multi_gaussian_weight_mode not in ("softmax", "softplus"):
            raise ValueError(
                "query_multi_gaussian_weight_mode must be one of {'softmax','softplus'}, "
                f"got {self.query_multi_gaussian_weight_mode!r}"
            )
        self.query_multi_gaussian_softplus_bias_init = float(query_multi_gaussian_softplus_bias_init)
        self.query_multi_gaussian_weight_reg_loss_weight = float(query_multi_gaussian_weight_reg_loss_weight)
        if self.query_multi_gaussian_weight_reg_loss_weight < 0.0:
            raise ValueError(
                "query_multi_gaussian_weight_reg_loss_weight must be >= 0, "
                f"got {self.query_multi_gaussian_weight_reg_loss_weight}"
            )
        self.query_multi_gaussian_weight_reg_target_sum = float(query_multi_gaussian_weight_reg_target_sum)
        if self.query_multi_gaussian_weight_reg_target_sum <= 0.0:
            raise ValueError(
                "query_multi_gaussian_weight_reg_target_sum must be > 0, "
                f"got {self.query_multi_gaussian_weight_reg_target_sum}"
            )
        self.gaussian_truncate_sigma = float(gaussian_truncate_sigma)
        self.query_feat_cosine_threshold = float(query_feat_cosine_threshold)
        self.query_center_distance_threshold_m = float(query_center_distance_threshold_m)
        self.detach_query_for_center_default = bool(detach_query_for_center_default)
        self.iter = 0
        self._train_iter = 0
        self._train_iter_synced = False
        self.debug_vis_every = int(debug_vis_every)
        self.debug_vis_dir = str(debug_vis_dir)
        self.center_only_mode = bool(center_only_mode)
        self.debug_query_center_marker_radius = max(0, int(debug_query_center_marker_radius))
        if debug_query_objectness_vis_threshold is not None:
            debug_query_confidence_vis_threshold = debug_query_objectness_vis_threshold
        self.debug_query_confidence_vis_threshold = float(debug_query_confidence_vis_threshold)
        if not (0.0 <= self.debug_query_confidence_vis_threshold <= 1.0):
            raise ValueError(
                "debug_query_confidence_vis_threshold must be in [0,1], "
                f"got {self.debug_query_confidence_vis_threshold}"
            )
        self.debug_query_gaussian_vis_mode = str(debug_query_gaussian_vis_mode).lower()
        if self.debug_query_gaussian_vis_mode not in ("ellipse", "prob", "threshold"):
            raise ValueError(
                "debug_query_gaussian_vis_mode must be one of "
                "{'ellipse','prob','threshold'}, "
                f"got {self.debug_query_gaussian_vis_mode!r}"
            )
        self.debug_query_gaussian_prob_threshold = float(debug_query_gaussian_prob_threshold)
        if not (0.0 <= self.debug_query_gaussian_prob_threshold <= 1.0):
            raise ValueError(
                "debug_query_gaussian_prob_threshold must be in [0,1], "
                f"got {self.debug_query_gaussian_prob_threshold}"
            )
        self.debug_query_gaussian_prob_alpha_scale = float(debug_query_gaussian_prob_alpha_scale)
        if self.debug_query_gaussian_prob_alpha_scale < 0.0:
            raise ValueError(
                "debug_query_gaussian_prob_alpha_scale must be >= 0, "
                f"got {self.debug_query_gaussian_prob_alpha_scale}"
            )
        if self.num_query_classes <= 1:
            raise ValueError(f"num_query_classes must be >= 2 (fg + bg), got {self.num_query_classes}")
        if self.num_overlap_frames < 0:
            raise ValueError(f"num_overlap_frames must be >= 0, got {self.num_overlap_frames}")
        if self.num_overlap_frames > self.num_past_frames:
            raise ValueError(
                f"num_overlap_frames({self.num_overlap_frames}) exceeds num_past_frames({self.num_past_frames})"
            )

        # NOTE: The temporal projection module (3-past-frames -> T_pred frames) has
        # been removed. The transformer's last layer is already an FFN+LN, and we
        # now consume query_inst directly as [T_past, Q, D] in forward(). The
        # legacy projection collapsed past frames into a single present frame and
        # blocked depth / 2D soft-argmax / 3D lifting from running on past frames,
        # which is exactly what we need for trajectory motion priors.

        self.center_head = CenterHead(embed_dim=self.embed_dim,
                                      out_dim=center_out_dim)
        self.gaussian_head = None
        self.gaussian_offset_head = None
        self.gaussian_sigma_head = None
        self.gaussian_yaw_head = None
        self.gaussian_weight_head = None
        if not self.center_only_mode:
            g = int(self.query_num_gaussians)
            self.gaussian_offset_head = GaussianHead(embed_dim=self.embed_dim, out_dim=g * 3)
            self.gaussian_sigma_head = GaussianHead(embed_dim=self.embed_dim, out_dim=g * 3)
            self.gaussian_yaw_head = GaussianHead(embed_dim=self.embed_dim, out_dim=g * 2)
            self.gaussian_weight_head = GaussianHead(embed_dim=self.embed_dim, out_dim=g)
            if self.query_multi_gaussian_weight_mode == "softplus":
                last_layer = self.gaussian_weight_head.mlp.net[-1]
                if isinstance(last_layer, nn.Linear) and last_layer.bias is not None:
                    nn.init.constant_(last_layer.bias, self.query_multi_gaussian_softplus_bias_init)
            # Backward-compat alias used by debug grad logger.
            self.gaussian_head = self.gaussian_sigma_head
        self.cls_head = ClassificationHead(embed_dim=self.embed_dim, out_dim=self.num_query_classes)
        self.query_depth_head = QueryDepthHead(
            embed_dim=self.embed_dim,
            out_dim=self.query_depth_num_bins,
        )
        self.query_traj_head = None
        self.query_traj_input_steps = max(0, int(self.num_past_frames) - 1) + max(
            0, int(self.query_traj_num_steps)
        )
        self.query_traj_input_dim = int(self.embed_dim) + 2
        if self.query_traj_num_steps > 0:
            self.query_traj_head = TrajectoryHead(
                input_dim=(self.query_traj_input_steps * self.query_traj_input_dim),
                hidden_dim=self.embed_dim,
                out_dim=(self.query_traj_num_steps * 2),
            )

        # W,b of center-head final linear are scaled to widen initial center spread.
        center_last = self.center_head.mlp.net[-1]
        if isinstance(center_last, nn.Linear):
            with torch.no_grad():
                center_last.weight.mul_(4.0)
                # if center_last.bias is not None:
                    # center_last.bias.mul_(4.0)

    def _validate_query_inst_shape(self, query_inst: torch.Tensor) -> None:
        """Shape checks for the [T_past, Q, D] queries consumed in forward()."""
        if query_inst.dim() != 3:
            raise ValueError(
                f"query_inst must be [T,Q,D], got {tuple(query_inst.shape)}"
            )
        if query_inst.shape[0] != int(self.num_past_frames):
            raise ValueError(
                f"query_inst time dim mismatch: got T={int(query_inst.shape[0])}, "
                f"expected num_past_frames={int(self.num_past_frames)}"
            )
        if query_inst.shape[-1] != int(self.embed_dim):
            raise ValueError(
                f"query_inst channel mismatch: got D={int(query_inst.shape[-1])}, "
                f"expected embed_dim={int(self.embed_dim)}"
            )

    def _resolve_present_query_local_idx(self, t_pred: int) -> int:
        t_pred = int(t_pred)
        if t_pred <= 0:
            return 0
        # query_feat_tqd is now query_inst itself ([T_past, Q, D]); the present
        # frame is the last past frame (global index time_receptive_field - 1).
        if bool(self.query_present_only):
            return max(0, t_pred - 1)
        if int(self.num_overlap_frames) > 0:
            return max(0, min(t_pred - 1, int(self.num_overlap_frames) - 1))
        return max(0, min(t_pred - 1, int(self.num_past_frames) - 1))

    def _build_query_trajectory_input(
        self,
        query_feat_tqd: torch.Tensor,
        centers_world_tq3: torch.Tensor,
    ):
        if self.query_traj_head is None:
            return None
        if (not torch.is_tensor(query_feat_tqd)) or query_feat_tqd.dim() != 3:
            raise ValueError("query_feat_tqd must be [T,Q,D] for trajectory input")
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            raise ValueError("centers_world_tq3 must be [T,Q,3] for trajectory input")
        if tuple(query_feat_tqd.shape[:2]) != tuple(centers_world_tq3.shape[:2]):
            raise ValueError(
                "trajectory query feature / center shape mismatch: "
                f"{tuple(query_feat_tqd.shape)} vs {tuple(centers_world_tq3.shape)}"
            )

        t_count = int(query_feat_tqd.shape[0])
        q_count = int(query_feat_tqd.shape[1])
        if t_count <= 0 or q_count <= 0:
            return None

        present_local_idx = self._resolve_present_query_local_idx(t_count)
        if present_local_idx >= t_count:
            return None

        hist_feat_tqd = query_feat_tqd[:present_local_idx]
        present_feat_qd = query_feat_tqd[present_local_idx]
        future_feat_tqd = present_feat_qd.unsqueeze(0).expand(
            int(self.query_traj_num_steps), -1, -1
        )
        motion_feat_tqd = torch.cat([hist_feat_tqd, future_feat_tqd], dim=0).to(torch.float32)

        prior_offsets_tq2 = centers_world_tq3[1:(present_local_idx + 1), :, :2] - centers_world_tq3[
            :present_local_idx, :, :2
        ]
        if bool(self.query_traj_prior_detach):
            prior_offsets_tq2 = prior_offsets_tq2.detach()
        prior_offsets_tq2 = prior_offsets_tq2.to(device=query_feat_tqd.device, dtype=torch.float32)

        future_prior_tq2 = query_feat_tqd.new_zeros(
            (int(self.query_traj_num_steps), q_count, 2),
            dtype=query_feat_tqd.dtype,
        ).to(torch.float32)
        motion_prior_tq2 = torch.cat([prior_offsets_tq2, future_prior_tq2], dim=0)

        if int(motion_feat_tqd.shape[0]) != int(self.query_traj_input_steps):
            raise ValueError(
                "unexpected trajectory feature timeline length: "
                f"{int(motion_feat_tqd.shape[0])} vs expected {int(self.query_traj_input_steps)}"
            )
        if tuple(motion_feat_tqd.shape[:2]) != tuple(motion_prior_tq2.shape[:2]):
            raise ValueError(
                "trajectory feature / prior time-query mismatch: "
                f"{tuple(motion_feat_tqd.shape)} vs {tuple(motion_prior_tq2.shape)}"
            )

        traj_max_xy = query_feat_tqd.new_tensor(self.query_traj_residual_max_m).view(1, 1, 2).to(torch.float32)
        motion_prior_tq2 = motion_prior_tq2 / traj_max_xy.clamp_min(1e-6)
        motion_prior_tq2 = motion_prior_tq2.clamp(min=-1.0, max=1.0)
        return torch.cat([motion_feat_tqd, motion_prior_tq2], dim=-1).contiguous()

    def _predict_trajectory_from_inputs(
        self,
        motion_input_tqd2: torch.Tensor,
        query_feat_tqd: torch.Tensor,
    ):
        if self.query_traj_head is None or motion_input_tqd2 is None:
            return None, None
        if (not torch.is_tensor(motion_input_tqd2)) or motion_input_tqd2.dim() != 3:
            raise ValueError("motion_input_tqd2 must be [S,Q,D+2]")
        s_count, q_count, d_count = [int(v) for v in motion_input_tqd2.shape]
        if s_count != int(self.query_traj_input_steps):
            raise ValueError(
                f"trajectory input step mismatch: {s_count} vs {int(self.query_traj_input_steps)}"
            )
        if d_count != int(self.query_traj_input_dim):
            raise ValueError(
                f"trajectory input dim mismatch: {d_count} vs {int(self.query_traj_input_dim)}"
            )
        motion_input_qsd = motion_input_tqd2.permute(1, 0, 2).contiguous()
        motion_input_qd = motion_input_qsd.reshape(q_count, s_count * d_count)
        traj_logits_qf2 = self.query_traj_head(motion_input_qd).reshape(
            q_count, int(self.query_traj_num_steps), 2
        )
        traj_logits_fq2 = traj_logits_qf2.permute(1, 0, 2).contiguous()
        traj_max_xy = query_feat_tqd.new_tensor(self.query_traj_residual_max_m).view(1, 1, 2)
        traj_deltas_fq2 = torch.tanh(traj_logits_fq2) * traj_max_xy
        return traj_logits_fq2, traj_deltas_fq2

    @staticmethod
    def _resolve_center_branch_input(
        query_feat_tqd: torch.Tensor,
        detach_query_for_center: bool,
    ) -> torch.Tensor:
        """
        Split interface between query-feature path and center-prediction path.
        - detach_query_for_center=False: center branch keeps gradient to query path.
        - detach_query_for_center=True: center branch consumes detached features.
        """
        if detach_query_for_center:
            return query_feat_tqd.detach()
        return query_feat_tqd

    def _world_to_range_logits(self, centers_world: torch.Tensor) -> torch.Tensor:
        pc_min = centers_world.new_tensor(self.point_cloud_range[:3])
        extent = centers_world.new_tensor(self.spatial_extent3d).clamp_min(1e-6)
        p = ((centers_world - pc_min) / extent).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(p / (1.0 - p))

    def _compute_gaussian_outputs(
        self,
        center_input_tqd: torch.Tensor,
        centers_world: torch.Tensor,
    ):
        """
        Predict Gaussian-mixture parameters ONCE from a single frame's feature
        and broadcast the same mixture across every time slot in centers_world.

        Args:
            center_input_tqd: [T, Q, D] center-branch feature. The slice at
                `present_local_idx = T - 1` (last past frame, i.e. current frame
                in present-only mode) is used as the sole input to the Gaussian
                heads — this captures the "predict gaussian once at [t]" rule.
            centers_world:    [T, Q, 3] per-frame anchor centers (e.g. 3D-lifted
                centers, one per past frame). The shared mixture offsets are
                added to each frame's center so that each past frame gets the
                same Gaussian shape located at its own anchor.

        Returns:
            query_sigma_world_tq3:        [T, Q, 3]
            mixture_centers_world_tqg3:   [T, Q, G, 3]   (per-frame placement)
            mixture_sigmas_world_tqg3:    [T, Q, G, 3]   (shared across T)
            mixture_yaw_tqg:              [T, Q, G]      (shared across T)
            mixture_weights_tqg:          [T, Q, G]      (shared across T)
        """
        query_sigma_world_tq3 = None
        mixture_centers_world_tqg3 = None
        mixture_sigmas_world_tqg3 = None
        mixture_yaw_tqg = None
        mixture_weights_tqg = None

        if self.gaussian_sigma_head is None:
            return (
                query_sigma_world_tq3,
                mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3,
                mixture_yaw_tqg,
                mixture_weights_tqg,
            )

        t_count, q_count = int(center_input_tqd.shape[0]), int(center_input_tqd.shape[1])
        if t_count <= 0 or q_count <= 0:
            return (
                query_sigma_world_tq3,
                mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3,
                mixture_yaw_tqg,
                mixture_weights_tqg,
            )
        g_count = int(self.query_num_gaussians)

        # Run the four Gaussian heads on the present-frame feature only.
        present_local_idx = self._resolve_present_query_local_idx(t_count)
        center_input_qd = center_input_tqd[present_local_idx]  # [Q, D]

        offset_logits_qg3 = self.gaussian_offset_head(center_input_qd).reshape(q_count, g_count, 3)
        sigma_logits_qg3 = self.gaussian_sigma_head(center_input_qd).reshape(q_count, g_count, 3)
        yaw_basis_logits_qg2 = self.gaussian_yaw_head(center_input_qd).reshape(q_count, g_count, 2)
        weight_logits_qg = self.gaussian_weight_head(center_input_qd).reshape(q_count, g_count)

        off_max = centers_world.new_tensor(self.query_multi_gaussian_offset_max_m).view(1, 1, 3)
        offsets_world_qg3 = torch.tanh(offset_logits_qg3) * off_max

        sigmas_world_qg3 = sigmoid_to_sigma_from_range(
            sigma_logits_qg3,
            sigma_min=self.query_multi_gaussian_sigma_min_m,
            sigma_max=self.query_multi_gaussian_sigma_max_m,
        )

        yaw_basis_qg2 = F.normalize(yaw_basis_logits_qg2, dim=-1, eps=1e-6)
        yaw_qg = torch.atan2(yaw_basis_qg2[..., 0], yaw_basis_qg2[..., 1])

        if self.query_multi_gaussian_weight_mode == "softmax":
            weights_qg = torch.softmax(weight_logits_qg, dim=-1)
            weights_surrogate_qg = weights_qg
        elif self.query_multi_gaussian_weight_mode == "softplus":
            weights_qg = F.softplus(weight_logits_qg)
            weights_surrogate_qg = weights_qg / weights_qg.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
        else:
            raise RuntimeError(
                f"Unsupported query_multi_gaussian_weight_mode={self.query_multi_gaussian_weight_mode!r}"
            )

        # Broadcast the shared mixture to each frame:
        # - mixture centers move with the per-frame anchor (lifted center).
        # - sigmas/yaw/weights are identical across the T frames.
        mixture_centers_world_tqg3 = centers_world.unsqueeze(2) + offsets_world_qg3.unsqueeze(0)
        pc_min_g = centers_world.new_tensor(self.point_cloud_range[:3]).view(1, 1, 1, 3)
        pc_max_g = centers_world.new_tensor(self.point_cloud_range[3:]).view(1, 1, 1, 3)
        mixture_centers_world_tqg3 = torch.max(torch.min(mixture_centers_world_tqg3, pc_max_g), pc_min_g)

        mixture_sigmas_world_tqg3 = sigmas_world_qg3.unsqueeze(0).expand(t_count, -1, -1, -1).contiguous()
        mixture_yaw_tqg = yaw_qg.unsqueeze(0).expand(t_count, -1, -1).contiguous()
        mixture_weights_tqg = weights_qg.unsqueeze(0).expand(t_count, -1, -1).contiguous()
        weights_surrogate_tqg = weights_surrogate_qg.unsqueeze(0).expand(t_count, -1, -1).contiguous()

        effective_offset_tqg3 = mixture_centers_world_tqg3 - centers_world.unsqueeze(2)
        moment2_tq3 = (
            weights_surrogate_tqg.unsqueeze(-1)
            * (mixture_sigmas_world_tqg3.pow(2) + effective_offset_tqg3.pow(2))
        ).sum(dim=2)
        query_sigma_world_tq3 = torch.sqrt(moment2_tq3.clamp_min(1e-8))

        return (
            query_sigma_world_tq3,
            mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3,
            mixture_yaw_tqg,
            mixture_weights_tqg,
        )

    def apply_lifted_centers_to_outputs(
        self,
        outputs: dict,
        centers_world: torch.Tensor,
        detach_query_for_center: bool = None,
        defer_trajectory: bool = False,
    ) -> dict:
        if not isinstance(outputs, dict):
            raise TypeError("outputs must be a dict returned by QueryHead.forward")
        query_feat_tqd = outputs.get("query_feat_tqd", None)
        if (not torch.is_tensor(query_feat_tqd)) or query_feat_tqd.dim() != 3:
            raise ValueError("outputs['query_feat_tqd'] must be [T,Q,D]")
        if (not torch.is_tensor(centers_world)) or centers_world.dim() != 3:
            raise ValueError(f"centers_world must be [T,Q,3], got {type(centers_world)}")
        if tuple(centers_world.shape[:2]) != tuple(query_feat_tqd.shape[:2]) or int(centers_world.shape[2]) != 3:
            raise ValueError(
                "lifted centers must match query feature [T,Q], got "
                f"centers={tuple(centers_world.shape)} vs query_feat={tuple(query_feat_tqd.shape)}"
            )

        if detach_query_for_center is None:
            detach_query_for_center = bool(self.detach_query_for_center_default)
        center_input_tqd = self._resolve_center_branch_input(
            query_feat_tqd, bool(detach_query_for_center)
        )
        centers_world = centers_world.to(device=query_feat_tqd.device, dtype=query_feat_tqd.dtype)
        pc_min = centers_world.new_tensor(self.point_cloud_range[:3])
        pc_max = centers_world.new_tensor(self.point_cloud_range[3:])
        centers_world = torch.max(torch.min(centers_world, pc_max), pc_min)
        center_logits = self._world_to_range_logits(centers_world)

        (
            query_sigma_world_tq3,
            mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3,
            mixture_yaw_tqg,
            mixture_weights_tqg,
        ) = self._compute_gaussian_outputs(center_input_tqd, centers_world)

        traj_motion_input_tqd2 = None
        traj_logits_fq2 = None
        traj_offsets_fq2 = None
        if self.query_traj_head is not None:
            traj_motion_input_tqd2 = self._build_query_trajectory_input(
                query_feat_tqd=query_feat_tqd,
                centers_world_tq3=centers_world,
            )
            if not defer_trajectory:
                traj_logits_fq2, traj_offsets_fq2 = self._predict_trajectory_from_inputs(
                    motion_input_tqd2=traj_motion_input_tqd2,
                    query_feat_tqd=query_feat_tqd,
                )

        outputs = dict(outputs)
        outputs.update({
            "centers_world_tq3": centers_world,
            "center_logits_tq3": center_logits,
            "query_sigma_world_tq3": query_sigma_world_tq3,
            "mixture_centers_world_tqg3": mixture_centers_world_tqg3,
            "mixture_sigmas_world_tqg3": mixture_sigmas_world_tqg3,
            "mixture_yaw_tqg": mixture_yaw_tqg,
            "mixture_weights_tqg": mixture_weights_tqg,
            "traj_motion_input_tqd2": traj_motion_input_tqd2,
            "traj_logits_fq2": traj_logits_fq2,
            "traj_offsets_fq2": traj_offsets_fq2,
            "traj_deltas_fq2": traj_offsets_fq2,
            "query_traj_logits_fq2": traj_logits_fq2,
            "query_traj_offsets_fq2": traj_offsets_fq2,
            "query_traj_deltas_fq2": traj_offsets_fq2,
            "centers_world": centers_world,
            "center_logits": center_logits,
            "gaussian_sigmas_world": query_sigma_world_tq3,
        })
        return outputs

    def _build_query_cls_feature_qd(
        self,
        query_inst_tqd: torch.Tensor,
        query_future_feat_tqd: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build one class feature per query from the unique temporal support.

        Present-only semantics (query_present_only=True):
        - query_inst and query_future_feat are the same tensor [T_past, Q, D]
          since the projection module was removed. Average over the past
          temporal axis to produce a single context-aware class feature.

        Plus-window semantics (query_present_only=False, legacy):
        - input 3       : (t-2, t-1, t)
        - output T_pred : (t-1, t, t+1, ...)
        - Overlap with input excluded via num_overlap_frames.
        """
        if query_inst_tqd.dim() != 3 or query_future_feat_tqd.dim() != 3:
            raise ValueError("query_inst_tqd and query_future_feat_tqd must be [T,Q,D].")
        if query_inst_tqd.shape[1:] != query_future_feat_tqd.shape[1:]:
            raise ValueError(
                "query_inst_tqd and query_future_feat_tqd must share [Q,D], got "
                f"{tuple(query_inst_tqd.shape)} vs {tuple(query_future_feat_tqd.shape)}"
            )
        if bool(self.query_present_only):
            # query_future_feat_tqd == query_inst_tqd ([T_past, Q, D]); a single
            # mean over the past axis already gives a context-augmented feature.
            return query_future_feat_tqd.to(torch.float32).mean(dim=0)
        prefix_len = max(0, int(self.num_past_frames) - int(self.num_overlap_frames))
        temporal_chunks = []
        if prefix_len > 0:
            temporal_chunks.append(query_inst_tqd[:prefix_len])
        temporal_chunks.append(query_future_feat_tqd)
        cls_feat_tqd = torch.cat(temporal_chunks, dim=0)
        return cls_feat_tqd.to(torch.float32).mean(dim=0)

    def set_train_iteration(self, train_iter: int, one_based: bool = True) -> None:
        step = int(train_iter)
        if not one_based:
            step += 1
        self._train_iter = max(0, step)
        self._train_iter_synced = True

    def _get_visualization_step(self, advance_if_unsynced: bool = False) -> int:
        if self._train_iter_synced:
            return int(self._train_iter)
        if advance_if_unsynced:
            self.iter += 1
        return int(self.iter)

    def compute_query_feature_cosine(
        self,
        query_feat_tqd: torch.Tensor,
        gt_feat_tqd: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Computes cosine similarity s_{tq} = cos(q_{tq}, g_{tq}).
        Args:
            query_feat_tqd: [T, Q, D]
            gt_feat_tqd: [T, Q, D]
        Returns:
            cosine_sim_tq: [T, Q] in [-1, 1]
        """
        if query_feat_tqd.shape != gt_feat_tqd.shape:
            raise ValueError(
                "query/gt feature shape mismatch: "
                f"{tuple(query_feat_tqd.shape)} vs {tuple(gt_feat_tqd.shape)}"
            )
        return F.cosine_similarity(
            query_feat_tqd.to(torch.float32),
            gt_feat_tqd.to(torch.float32),
            dim=-1,
            eps=float(eps),
        )

    def compute_center_distance_m(
        self,
        pred_centers_world_tq3: torch.Tensor,
        gt_centers_world_tq3: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes center distance d_{tq} = ||c_hat_{tq} - c*_{tq}||_2 (meters).
        Args:
            pred_centers_world_tq3: [T, Q, 3]
            gt_centers_world_tq3: [T, Q, 3]
        Returns:
            center_dist_tq: [T, Q] in meters
        """
        if pred_centers_world_tq3.shape != gt_centers_world_tq3.shape:
            raise ValueError(
                "pred/gt center shape mismatch: "
                f"{tuple(pred_centers_world_tq3.shape)} vs {tuple(gt_centers_world_tq3.shape)}"
            )
        return torch.norm(
            pred_centers_world_tq3.to(torch.float32) - gt_centers_world_tq3.to(torch.float32),
            p=2,
            dim=-1,
        )

    def classify_query_cases(
        self,
        cosine_sim_tq: torch.Tensor,
        center_dist_tq_m: torch.Tensor,
        valid_mask_tq: torch.Tensor = None,
        cosine_threshold: float = None,
        center_distance_threshold_m: float = None,
    ) -> dict:
        """
        Four-case definition used by the query training policy.

        Binary criteria:
            I_feat(t,q)   = 1[s_{tq} >= tau_feat]
            I_center(t,q) = 1[d_{tq} <= tau_center]
            s_{tq} = cos(q_feat_{tq}, gt_feat_{tq})
            d_{tq} = ||c_hat_{tq} - c*_{tq}||_2  (meters)

        Cases:
            blue   : I_feat=1 and I_center=1
            green  : I_feat=1 and I_center=0
            orange : I_feat=0 and I_center=1
            purple : I_feat=0 and I_center=0

        Args:
            cosine_sim_tq: [T, Q]
            center_dist_tq_m: [T, Q]
            valid_mask_tq: optional [T, Q] bool mask. invalid entries are excluded.
            cosine_threshold: tau_feat. default self.query_feat_cosine_threshold.
            center_distance_threshold_m: tau_center. default self.query_center_distance_threshold_m.
        Returns:
            dict with bool masks and case_id_tq:
            - case_id: -1 invalid, 0 purple, 1 orange, 2 green, 3 blue
        """
        if cosine_sim_tq.shape != center_dist_tq_m.shape:
            raise ValueError(
                "metric shape mismatch: "
                f"{tuple(cosine_sim_tq.shape)} vs {tuple(center_dist_tq_m.shape)}"
            )
        tau_feat = (
            float(self.query_feat_cosine_threshold)
            if cosine_threshold is None else float(cosine_threshold)
        )
        tau_center = (
            float(self.query_center_distance_threshold_m)
            if center_distance_threshold_m is None else float(center_distance_threshold_m)
        )
        sim = cosine_sim_tq.to(torch.float32)
        dist = center_dist_tq_m.to(torch.float32)

        if valid_mask_tq is None:
            valid = torch.ones_like(sim, dtype=torch.bool)
        else:
            if valid_mask_tq.shape != sim.shape:
                raise ValueError(
                    "valid_mask shape mismatch: "
                    f"{tuple(valid_mask_tq.shape)} vs {tuple(sim.shape)}"
                )
            valid = valid_mask_tq.to(device=sim.device, dtype=torch.bool)

        is_feat_good = (sim >= tau_feat) & valid
        is_center_good = (dist <= tau_center) & valid

        case_blue = is_feat_good & is_center_good
        case_green = is_feat_good & (~is_center_good) & valid
        case_orange = (~is_feat_good) & is_center_good & valid
        case_purple = (~is_feat_good) & (~is_center_good) & valid

        case_id = torch.full(sim.shape, -1, dtype=torch.long, device=sim.device)
        case_id[case_purple] = 0
        case_id[case_orange] = 1
        case_id[case_green] = 2
        case_id[case_blue] = 3

        return {
            "tau_feat": tau_feat,
            "tau_center_m": tau_center,
            "is_valid": valid,
            "is_feat_good": is_feat_good,
            "is_center_good": is_center_good,
            "case_blue": case_blue,
            "case_green": case_green,
            "case_orange": case_orange,
            "case_purple": case_purple,
            "case_id": case_id,
        }

    def forward(
        self,
        query_inst,
        return_query_feats: bool = False,
        detach_query_for_center: bool = None,
        compute_direct_center: bool = False,
    ):
        '''
            query_inst: [T_past, num_queries, embed_dim]
            detach_query_for_center:
                - False: center/gaussian branch updates query feature path.
                - True: center/gaussian branch uses detached query features.
                - None: uses self.detach_query_for_center_default.
        '''
        self._validate_query_inst_shape(query_inst)
        # Consume query_inst directly as the per-past-frame feature [T_past, Q, D].
        # All downstream branches (depth, center, gaussian, traj, cls) now see one
        # feature slot per past frame instead of a single projected present slot.
        query_feat_tqd = query_inst
        if detach_query_for_center is None:
            detach_query_for_center = bool(self.detach_query_for_center_default)
        center_input_tqd = self._resolve_center_branch_input(
            query_feat_tqd, bool(detach_query_for_center)
        )

        query_depth_logits_tqd = self.query_depth_head(query_feat_tqd)  # [T,Q,Dbin]
        query_depth_probs_tqd = torch.softmax(query_depth_logits_tqd, dim=-1)
        query_cls_feat_qd = self._build_query_cls_feature_qd(query_inst, query_feat_tqd)
        query_cls_logits_qc = self.cls_head(query_cls_feat_qd)  # [Q,C]
        query_cls_scores_qc = torch.softmax(query_cls_logits_qc, dim=-1)
        query_cls_scores_tqc = query_cls_scores_qc.unsqueeze(0).expand(
            int(query_feat_tqd.shape[0]), -1, -1
        )
        query_present_local_idx = self._resolve_present_query_local_idx(
            int(query_feat_tqd.shape[0])
        )
        traj_logits_fq2 = None
        traj_offsets_fq2 = None
        traj_motion_input_tqd2 = None

        centers_world = None
        center_logits = None
        query_sigma_world_tq3 = None
        mixture_centers_world_tqg3 = None
        mixture_sigmas_world_tqg3 = None
        mixture_yaw_tqg = None
        mixture_weights_tqg = None
        if bool(compute_direct_center):
            center_logits = self.center_head(center_input_tqd)  # [T, Q, 3]
            centers_world = sigmoid_to_world_from_range(
                center_logits, self.point_cloud_range, self.spatial_extent3d
            )
            pc_min = centers_world.new_tensor(self.point_cloud_range[:3])
            pc_max = centers_world.new_tensor(self.point_cloud_range[3:])
            centers_world = torch.max(torch.min(centers_world, pc_max), pc_min)
            (
                query_sigma_world_tq3,
                mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3,
                mixture_yaw_tqg,
                mixture_weights_tqg,
            ) = self._compute_gaussian_outputs(center_input_tqd, centers_world)
            if self.query_traj_head is not None:
                traj_motion_input_tqd2 = self._build_query_trajectory_input(
                    query_feat_tqd=query_feat_tqd,
                    centers_world_tq3=centers_world,
                )
                traj_logits_fq2, traj_offsets_fq2 = self._predict_trajectory_from_inputs(
                    motion_input_tqd2=traj_motion_input_tqd2,
                    query_feat_tqd=query_feat_tqd,
                )

        outputs = {
            "centers_world_tq3": centers_world,
            "center_logits_tq3": center_logits,
            "query_sigma_world_tq3": query_sigma_world_tq3,
            "mixture_centers_world_tqg3": mixture_centers_world_tqg3,
            "mixture_sigmas_world_tqg3": mixture_sigmas_world_tqg3,
            "mixture_yaw_tqg": mixture_yaw_tqg,
            "mixture_weights_tqg": mixture_weights_tqg,
            "query_cls_logits_qc": query_cls_logits_qc,
            "query_cls_scores_qc": query_cls_scores_qc,
            "query_cls_scores_tqc": query_cls_scores_tqc,
            "query_depth_logits_tqd": query_depth_logits_tqd,
            "query_depth_probs_tqd": query_depth_probs_tqd,
            "query_feat_tqd": query_feat_tqd,
            "query_present_local_idx": int(query_present_local_idx),
            "traj_motion_input_tqd2": traj_motion_input_tqd2,
            "traj_logits_fq2": traj_logits_fq2,
            "traj_offsets_fq2": traj_offsets_fq2,
            "traj_deltas_fq2": traj_offsets_fq2,
            "query_traj_logits_fq2": traj_logits_fq2,
            "query_traj_offsets_fq2": traj_offsets_fq2,
            "query_traj_deltas_fq2": traj_offsets_fq2,
            # Legacy aliases for compatibility with older call sites.
            "centers_world": centers_world,
            "center_logits": center_logits,
            "gaussian_sigmas_world": query_sigma_world_tq3,
        }
        return outputs

    def compute_multi_gaussian_sigma_reg_loss(
        self,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
    ) -> dict:
        ref_tensor = None
        if torch.is_tensor(mixture_sigmas_world_tqg3):
            ref_tensor = mixture_sigmas_world_tqg3
        elif torch.is_tensor(mixture_weights_tqg):
            ref_tensor = mixture_weights_tqg
        else:
            ref_tensor = next(self.parameters())
        z = ref_tensor.sum() * 0.0
        out = {
            "loss_query_sigma_reg": z,
            "loss_query_weight_reg": z,
            "dbg_query_sigma_reg_raw": z,
            "dbg_query_sigma_mean_m": z,
            "dbg_query_sigma_log_abs_mean": z,
            "dbg_query_sigma_reg_weight": z.new_tensor(float(self.query_multi_gaussian_sigma_reg_loss_weight)),
            "dbg_query_sigma_reg_enabled": z.new_tensor(
                float(1.0 if self.query_multi_gaussian_sigma_reg_loss_weight > 0.0 else 0.0)
            ),
            "dbg_query_weight_sum_mean": z,
            "dbg_query_weight_reg_raw": z,
            "dbg_query_weight_reg_weight": z.new_tensor(float(self.query_multi_gaussian_weight_reg_loss_weight)),
            "dbg_query_weight_reg_target_sum": z.new_tensor(float(self.query_multi_gaussian_weight_reg_target_sum)),
            "dbg_query_weight_reg_enabled": z.new_tensor(
                float(
                    1.0
                    if (
                        self.query_multi_gaussian_weight_mode == "softplus"
                        and self.query_multi_gaussian_weight_reg_loss_weight > 0.0
                    )
                    else 0.0
                )
            ),
        }
        if mixture_sigmas_world_tqg3 is not None:
            if not torch.is_tensor(mixture_sigmas_world_tqg3):
                raise TypeError("mixture_sigmas_world_tqg3 must be a tensor or None")
            if mixture_sigmas_world_tqg3.dim() != 4 or int(mixture_sigmas_world_tqg3.shape[-1]) != 3:
                raise ValueError(
                    "mixture_sigmas_world_tqg3 must be [T,Q,G,3], "
                    f"got {tuple(mixture_sigmas_world_tqg3.shape)}"
                )
            if mixture_sigmas_world_tqg3.numel() > 0:
                sigmas = mixture_sigmas_world_tqg3.to(torch.float32).clamp_min(
                    self.query_multi_gaussian_sigma_reg_log_eps
                )
                log_abs = torch.log(sigmas).abs()
                raw = log_abs.sum(dim=-1).mean()
                out["dbg_query_sigma_reg_raw"] = raw.detach()
                out["dbg_query_sigma_mean_m"] = sigmas.mean().detach()
                out["dbg_query_sigma_log_abs_mean"] = log_abs.mean().detach()
                out["loss_query_sigma_reg"] = raw * float(self.query_multi_gaussian_sigma_reg_loss_weight)

        if mixture_weights_tqg is not None:
            if not torch.is_tensor(mixture_weights_tqg):
                raise TypeError("mixture_weights_tqg must be a tensor or None")
            if mixture_weights_tqg.dim() != 3:
                raise ValueError(
                    "mixture_weights_tqg must be [T,Q,G], "
                    f"got {tuple(mixture_weights_tqg.shape)}"
                )
            if mixture_weights_tqg.numel() > 0:
                weights = mixture_weights_tqg.to(torch.float32).clamp_min(0.0)
                weight_sum_tq = weights.sum(dim=-1)
                out["dbg_query_weight_sum_mean"] = weight_sum_tq.mean().detach()
                if self.query_multi_gaussian_weight_mode == "softplus":
                    raw_w = (weight_sum_tq - float(self.query_multi_gaussian_weight_reg_target_sum)).pow(2).mean()
                    out["dbg_query_weight_reg_raw"] = raw_w.detach()
                    out["loss_query_weight_reg"] = raw_w * float(self.query_multi_gaussian_weight_reg_loss_weight)
        return out

    def _is_main_process(self): return is_main_process()

    @property
    def _vis_cfg(self):
        return VisConfig(
            vis_every=self.debug_vis_every,
            vis_dir=self.debug_vis_dir,
            point_cloud_range=self.point_cloud_range,
            spatial_extent3d=self.spatial_extent3d,
            class_ids=self.query_class_ids,
            class_names=self.query_class_names,
            marker_radius=self.debug_query_center_marker_radius,
            conf_thr=self.debug_query_confidence_vis_threshold,
            gaussian_vis_mode=self.debug_query_gaussian_vis_mode,
            gaussian_prob_thr=self.debug_query_gaussian_prob_threshold,
            gaussian_prob_alpha_scale=self.debug_query_gaussian_prob_alpha_scale,
            gaussian_truncate_sigma=self.gaussian_truncate_sigma,
        )

    @torch.no_grad()
    def _maybe_save_points_gt_vis(
        self,
        points_world: torch.Tensor,
        gt_occ_gmo: torch.Tensor,
        step: int,
        gt_occ_semantic: torch.Tensor = None,
        max_frames: int = 6,
        voxel_center_offset: float = 0.5,
        points_world_all: torch.Tensor = None,
        point_conf_all: torch.Tensor = None,
        point_class_ids_all: torch.Tensor = None,
        point_weights_all: torch.Tensor = None,
        query_vis_bundle: dict = None,
    ) -> None:
        if point_conf_all is None:
            point_conf_all = point_weights_all
        if (not torch.is_tensor(gt_occ_gmo)) or gt_occ_gmo.dim() != 4:
            return
        self._maybe_save_prob_grid_vis(
            pred_occ_prob=gt_occ_gmo.to(torch.float32),
            points_world=points_world,
            gt_occ_gmo=gt_occ_gmo,
            gt_occ_semantic=gt_occ_semantic,
            step=step,
            max_frames=max_frames,
            voxel_center_offset=voxel_center_offset,
            prob_threshold=0.5,
            pred_layout="zxy",
            pred_occ_prob_pos_obj=None,
            pred_occ_prob_all_obj=None,
            points_world_all=points_world_all,
            point_conf_all=point_conf_all,
            point_class_ids_all=point_class_ids_all,
            point_weights_all=point_weights_all,
            query_vis_bundle=query_vis_bundle,
        )

    @torch.no_grad()
    def _maybe_save_prob_grid_vis(
        self,
        pred_occ_prob,
        points_world,
        gt_occ_gmo,
        step,
        gt_occ_semantic=None,
        max_frames=6,
        voxel_center_offset=0.5,
        prob_threshold=0.5,
        pred_layout="auto",
        pred_occ_prob_pos_obj=None,
        pred_occ_prob_all_obj=None,
        points_world_all=None,
        point_conf_all=None,
        point_class_ids_all=None,
        point_weights_all=None,
        query_vis_bundle=None,
    ):
        save_prob_grid_vis(
            pred_occ_prob, points_world, gt_occ_gmo, step, self._vis_cfg,
            gt_occ_semantic=gt_occ_semantic, max_frames=max_frames,
            voxel_center_offset=voxel_center_offset, prob_threshold=prob_threshold,
            pred_layout=pred_layout, pred_occ_prob_pos_obj=pred_occ_prob_pos_obj,
            pred_occ_prob_all_obj=pred_occ_prob_all_obj, points_world_all=points_world_all,
            point_conf_all=point_conf_all, point_class_ids_all=point_class_ids_all,
            point_weights_all=point_weights_all, query_vis_bundle=query_vis_bundle,
        )

    def gt_occ_to_gmo_binary(self, gt_occ: torch.Tensor,
                            gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
                            ignore_index: int = 255) -> torch.Tensor:
        gt = gt_occ.to(torch.long)
        out = torch.zeros_like(gt, dtype=torch.long)
        ignore_mask = (gt == ignore_index)
        gmo_ids_t = gt.new_tensor(gmo_ids, dtype=torch.long)
        gmo_mask = (gt.unsqueeze(-1) == gmo_ids_t).any(dim=-1)
        out[gmo_mask] = 1
        out[ignore_mask] = ignore_index
        return out

    def _pred_occ_to_zxy(self, pred_occ: torch.Tensor, pred_layout: str = "zyx") -> torch.Tensor:
        if pred_occ.dim() == 5 and pred_occ.shape[1] == 1:
            pred = pred_occ[:, 0]
        elif pred_occ.dim() == 4:
            pred = pred_occ
        else:
            raise ValueError(
                f"pred_occ must be [T,1,Z,*,*] or [T,Z,*,*], got {tuple(pred_occ.shape)}"
            )
        layout = str(pred_layout).lower()
        if layout == "zyx":
            pred = pred.permute(0, 1, 3, 2).contiguous()
        elif layout == "zxy":
            pred = pred.contiguous()
        else:
            raise ValueError(f"Unsupported pred_layout={pred_layout}. Use 'zyx' or 'zxy'.")
        return pred

    def _select_temporal_gt_for_prediction(self, gt_t: torch.Tensor, t_pred: int) -> torch.Tensor:
        if (not torch.is_tensor(gt_t)) or gt_t.dim() <= 0:
            return gt_t
        t_total = int(gt_t.shape[0])
        t_pred = int(t_pred)
        if t_total < t_pred:
            raise ValueError(f"GT time dim too short: {t_total} < {t_pred}")
        if t_total == t_pred:
            return gt_t.contiguous()
        if bool(getattr(self, "query_present_only", False)) and t_pred == 1:
            full_len = int(self.num_past_frames + self.num_future_frames - self.num_overlap_frames)
            if t_total >= full_len:
                local_idx = int(self.num_past_frames - 1)
            elif t_total >= int(self.num_future_frames):
                local_idx = int(max(0, self.num_overlap_frames - 1))
            elif t_total >= int(self.num_past_frames):
                local_idx = int(self.num_past_frames - 1)
            else:
                local_idx = 0
            local_idx = max(0, min(t_total - 1, local_idx))
            return gt_t.narrow(0, local_idx, 1).contiguous()
        return gt_t[-t_pred:].contiguous()

    def compute_gmo_loss(self,
                         pred_occ: torch.Tensor,
                         gt_occ: torch.Tensor,
                         gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
                         ignore_index: int = 255,
                         occ_dt=None,
                         points_world: torch.Tensor = None,
                         pred_occ_pos_vis: torch.Tensor = None,
                         pred_occ_obj_vis: torch.Tensor = None,
                         points_world_all_vis: torch.Tensor = None,
                         point_weights_all_vis: torch.Tensor = None,
                         pred_layout: str = "zyx",
                         loss_weight=1.0) -> torch.Tensor:
        pred_occ = self._pred_occ_to_zxy(pred_occ, pred_layout=pred_layout)
        T_pred = int(pred_occ.shape[0])

        if isinstance(gt_occ, (list, tuple)):
            gt_occ_t = torch.stack(gt_occ, dim=0)
        elif torch.is_tensor(gt_occ):
            gt_occ_t = gt_occ
        else:
            raise TypeError(f"gt_occ must be list/tuple/tensor, got {type(gt_occ)}")

        if gt_occ_t.dim() == 6 and gt_occ_t.shape[0] == 1 and gt_occ_t.shape[2] == 1:
            gt_occ_t = gt_occ_t[0, :, 0]
        elif gt_occ_t.dim() == 5 and gt_occ_t.shape[1] == 1:
            gt_occ_t = gt_occ_t[:, 0]
        elif gt_occ_t.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported gt_occ shape for gmo loss: {tuple(gt_occ_t.shape)}")

        if gt_occ_t.shape[0] < T_pred:
            raise ValueError(f"gt_occ time dim too short: {gt_occ_t.shape[0]} < {T_pred}")
        if gt_occ_t.shape[0] != T_pred:
            gt_occ_t = self._select_temporal_gt_for_prediction(gt_occ_t, T_pred)

        gt_occ_t = gt_occ_t.permute(0, 3, 1, 2).contiguous()
        gt_occ_gmo = self.gt_occ_to_gmo_binary(gt_occ_t, gmo_ids=gmo_ids, ignore_index=ignore_index)

        valid = (gt_occ_gmo != ignore_index)
        target_fg = (gt_occ_gmo == 1).float()

        eps = 1e-6
        p = pred_occ.clamp(eps, 1.0 - eps)
        t = target_fg

        pos_mask = valid & (t > 0.5)
        neg_mask = valid & (t < 0.5)
        loss_map = F.binary_cross_entropy(p, t, reduction="none")
        pos_cnt = pos_mask.sum().clamp(min=1).float()
        neg_cnt = neg_mask.sum().clamp(min=1).float()
        loss_pos = (loss_map * pos_mask.float()).sum() / pos_cnt
        loss_neg = (loss_map * neg_mask.float()).sum() / neg_cnt
        loss = 0.5 * loss_pos + 0.5 * loss_neg
        loss_dict = {"loss_gmo_bce": loss * loss_weight}

        pred_occ_pos_vis_zxy = None
        if pred_occ_pos_vis is not None:
            pred_occ_pos_vis_zxy = self._pred_occ_to_zxy(pred_occ_pos_vis, pred_layout=pred_layout)

        pred_occ_obj_vis_zxy = None
        if pred_occ_obj_vis is not None:
            pred_occ_obj_vis_zxy = self._pred_occ_to_zxy(pred_occ_obj_vis, pred_layout=pred_layout)

        step = self._get_visualization_step(advance_if_unsynced=True)
        self._maybe_save_prob_grid_vis(
            pred_occ_prob=pred_occ,
            points_world=points_world,
            gt_occ_gmo=gt_occ_gmo,
            gt_occ_semantic=gt_occ_t.permute(0, 2, 3, 1).contiguous(),
            step=step, max_frames=6, voxel_center_offset=0.5,
            prob_threshold=0.5, pred_layout="zxy",
            pred_occ_prob_pos_obj=pred_occ_pos_vis_zxy,
            pred_occ_prob_all_obj=pred_occ_obj_vis_zxy,
            points_world_all=points_world_all_vis,
            point_weights_all=point_weights_all_vis,
        )

        return loss_dict

    @torch.no_grad()
    def maybe_save_query_debug_vis(
        self,
        pred_occ_prob: torch.Tensor,
        gt_occ,
        gt_occ_inst=None,
        gt_occ_semantic: torch.Tensor = None,
        gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        ignore_index: int = 255,
        points_world: torch.Tensor = None,
        step: int = 0,
        max_frames: int = 6,
        voxel_center_offset: float = 0.5,
        prob_threshold: float = 0.5,
        pred_layout: str = "zyx",
        pred_occ_prob_pos_obj: torch.Tensor = None,
        pred_occ_prob_all_obj: torch.Tensor = None,
        points_world_all: torch.Tensor = None,
        point_conf_all: torch.Tensor = None,
        point_class_ids_all: torch.Tensor = None,
        point_weights_all: torch.Tensor = None,
        query_vis_bundle: dict = None,
    ) -> None:
        """
        Loss on/off와 무관하게 query debug visualization만 저장한다.
        """
        vis_every = int(getattr(self, "debug_vis_every", 0))
        if vis_every <= 0:
            return
        step = int(step)
        if (step % vis_every) != 0:
            return
        if points_world is None:
            return

        gt_occ_sem_t = None
        use_gt_occ_inst_binary = False
        gt_occ_inst_cls_t, gt_occ_inst_t = normalize_gt_occ_inst_for_vis(gt_occ_inst)
        if torch.is_tensor(gt_occ_inst_t):
            # gt_occ_inst: occupied iff instance-id > 0.
            gt_occ_t = (gt_occ_inst_t > 0).to(torch.long)
            gt_occ_sem_t = gt_occ_inst_cls_t
            use_gt_occ_inst_binary = True
        else:
            if gt_occ is None:
                return
            if isinstance(gt_occ, (list, tuple)):
                gt_occ_t = torch.stack(gt_occ, dim=0)
            elif torch.is_tensor(gt_occ):
                gt_occ_t = gt_occ
            else:
                return

            # Supported layouts:
            # - [T,1,X,Y,Z]
            # - [1,T,1,X,Y,Z]
            # - [T,X,Y,Z]
            if gt_occ_t.dim() == 6 and gt_occ_t.shape[0] == 1 and gt_occ_t.shape[2] == 1:
                gt_occ_t = gt_occ_t[0, :, 0]  # [T,X,Y,Z]
            elif gt_occ_t.dim() == 5 and gt_occ_t.shape[1] == 1:
                gt_occ_t = gt_occ_t[:, 0]     # [T,X,Y,Z]
            elif gt_occ_t.dim() == 4:
                pass
            else:
                return
            gt_occ_sem_t = normalize_gt_occ_semantic_for_vis(gt_occ_semantic)

        if gt_occ_sem_t is not None:
            if gt_occ_sem_t.shape[0] < gt_occ_t.shape[0]:
                gt_occ_sem_t = None
            elif gt_occ_sem_t.shape[0] != gt_occ_t.shape[0]:
                gt_occ_sem_t = gt_occ_sem_t[-gt_occ_t.shape[0]:].contiguous()
        if gt_occ_sem_t is None:
            gt_occ_sem_t = gt_occ_t

        if torch.is_tensor(points_world) and points_world.dim() >= 2 and int(points_world.shape[0]) > 0:
            t_vis = int(points_world.shape[0])
            if int(gt_occ_t.shape[0]) >= t_vis and int(gt_occ_t.shape[0]) != t_vis:
                try:
                    gt_occ_t = self._select_temporal_gt_for_prediction(gt_occ_t, t_vis)
                    if torch.is_tensor(gt_occ_sem_t) and int(gt_occ_sem_t.shape[0]) >= t_vis:
                        gt_occ_sem_t = self._select_temporal_gt_for_prediction(gt_occ_sem_t, t_vis)
                except Exception:
                    return

        gt_occ_zxy = gt_occ_t.permute(0, 3, 1, 2).contiguous()
        if use_gt_occ_inst_binary:
            gt_occ_gmo = gt_occ_zxy.to(torch.long).clamp(min=0, max=1)
        else:
            gt_occ_gmo = self.gt_occ_to_gmo_binary(gt_occ_zxy, gmo_ids=gmo_ids, ignore_index=ignore_index)

        if pred_occ_prob is None:
            self._maybe_save_points_gt_vis(
                points_world=points_world,
                gt_occ_gmo=gt_occ_gmo,
                gt_occ_semantic=gt_occ_sem_t,
                step=step,
                max_frames=max_frames,
                voxel_center_offset=voxel_center_offset,
                points_world_all=points_world_all,
                point_conf_all=point_conf_all,
                point_class_ids_all=point_class_ids_all,
                point_weights_all=point_weights_all,
                query_vis_bundle=query_vis_bundle,
            )
            return

        try:
            pred_occ_zxy = self._pred_occ_to_zxy(pred_occ_prob, pred_layout=pred_layout)
        except Exception:
            return

        T_pred = int(pred_occ_zxy.shape[0])
        if T_pred <= 0:
            return
        if gt_occ_zxy.shape[0] < T_pred:
            return
        if gt_occ_zxy.shape[0] != T_pred:
            gt_occ_t = self._select_temporal_gt_for_prediction(gt_occ_t, T_pred)
            gt_occ_zxy = gt_occ_t.permute(0, 3, 1, 2).contiguous()
            if use_gt_occ_inst_binary:
                gt_occ_gmo = gt_occ_zxy.to(torch.long).clamp(min=0, max=1)
            else:
                gt_occ_gmo = self.gt_occ_to_gmo_binary(
                    gt_occ_zxy, gmo_ids=gmo_ids, ignore_index=ignore_index
                )
        if gt_occ_sem_t is not None:
            if gt_occ_sem_t.shape[0] < T_pred:
                gt_occ_sem_t = None
            elif gt_occ_sem_t.shape[0] != T_pred:
                gt_occ_sem_t = self._select_temporal_gt_for_prediction(gt_occ_sem_t, T_pred)
        if gt_occ_sem_t is None:
            gt_occ_sem_t = gt_occ_t

        pred_occ_pos_zxy = None
        if pred_occ_prob_pos_obj is not None:
            try:
                pred_occ_pos_zxy = self._pred_occ_to_zxy(
                    pred_occ_prob_pos_obj, pred_layout=pred_layout
                )
            except Exception:
                pred_occ_pos_zxy = None

        pred_occ_obj_zxy = None
        if pred_occ_prob_all_obj is not None:
            try:
                pred_occ_obj_zxy = self._pred_occ_to_zxy(
                    pred_occ_prob_all_obj, pred_layout=pred_layout
                )
            except Exception:
                pred_occ_obj_zxy = None

        self._maybe_save_prob_grid_vis(
            pred_occ_prob=pred_occ_zxy,
            points_world=points_world,
            gt_occ_gmo=gt_occ_gmo,
            gt_occ_semantic=gt_occ_sem_t,
            step=step,
            max_frames=max_frames,
            voxel_center_offset=voxel_center_offset,
            prob_threshold=prob_threshold,
            pred_layout="zxy",
            pred_occ_prob_pos_obj=pred_occ_pos_zxy,
            pred_occ_prob_all_obj=pred_occ_obj_zxy,
            points_world_all=points_world_all,
            point_conf_all=point_conf_all,
            point_class_ids_all=point_class_ids_all,
            point_weights_all=point_weights_all,
            query_vis_bundle=query_vis_bundle,
        )

        if (
            isinstance(query_vis_bundle, dict)
            and self._is_main_process()
        ):
            try:
                import os
                vis_dir = str(getattr(self, "debug_vis_dir", "./work_dirs/query_debug_vis"))
                os.makedirs(vis_dir, exist_ok=True)
                out_path = os.path.join(vis_dir, f"iter_{int(step):06d}_query_vis.pt")
                sidecar = {}
                for key in (
                    "score_q",
                    "iou_q",
                    "cls_prob_q",
                    "cam_attn_score_q",
                    "cam_attn_score_valid_q",
                    "pred_cls_q",
                    "gaussian_sigmas_tq3",
                    "selected_sigmas_tq3",
                    "matched_sigmas_tq3",
                    "candidate_mixture_centers_tqg3",
                    "candidate_mixture_sigmas_tqg3",
                    "candidate_mixture_yaw_tqg",
                    "candidate_mixture_weights_tqg",
                    "selected_mixture_centers_tqg3",
                    "selected_mixture_sigmas_tqg3",
                    "selected_mixture_yaw_tqg",
                    "selected_mixture_weights_tqg",
                    "matched_mixture_centers_tqg3",
                    "matched_mixture_sigmas_tqg3",
                    "matched_mixture_yaw_tqg",
                    "matched_mixture_weights_tqg",
                    "selected_query_idx_q",
                ):
                    val = query_vis_bundle.get(key, None)
                    if torch.is_tensor(val):
                        sidecar[key] = val.detach().cpu()
                sidecar["top_k"] = int(query_vis_bundle.get("top_k", 0))
                sidecar["score_thr"] = float(query_vis_bundle.get("score_thr", 0.0))
                sidecar["w_iou"] = float(query_vis_bundle.get("w_iou", 0.5))
                sidecar["w_cls"] = float(query_vis_bundle.get("w_cls", 0.5))
                sidecar["w_cam"] = float(query_vis_bundle.get("w_cam", 0.0))
                torch.save(sidecar, out_path)
            except Exception:
                pass

    def compute_gmo_dice_loss(
        self,
        pred_occ: torch.Tensor,
        gt_occ,
        gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        ignore_index: int = 255,
        pred_layout: str = "zyx",
        loss_weight: float = 1.0,
        smooth: float = 1.0,
        eps: float = 1e-6,
        include_bg: bool = False,
        pred_is_logits: bool = False,
    ):
        """
        GMO binary Dice loss.

        Args:
            pred_occ:
                [T,1,Z,*,*] 또는 [T,Z,*,*]. (prob 또는 logits)
            gt_occ:
                list/tuple or tensor. 보통 list length>=T, each [1,X,Y,Z] 정수 라벨.
            gmo_ids:
                positive(1)로 매핑할 GT class id 집합.
            pred_layout:
                "zyx"이면 pred spatial dims를 [Z,Y,X]로 해석 후 [Z,X,Y]로 정렬.
                "zxy"이면 이미 [Z,X,Y]로 가정.
            include_bg:
                True면 background dice도 함께 평균.
            pred_is_logits:
                True면 sigmoid 후 Dice 계산.
        Returns:
            dict(loss_gmo_dice=...)
        """
        pred = self._pred_occ_to_zxy(pred_occ, pred_layout=pred_layout)

        T_pred = int(pred.shape[0])

        # gt_occ -> [T, X, Y, Z]
        if isinstance(gt_occ, (list, tuple)):
            gt_occ_t = torch.stack(gt_occ, dim=0)
        elif torch.is_tensor(gt_occ):
            gt_occ_t = gt_occ
        else:
            raise TypeError(f"gt_occ must be list/tuple/tensor, got {type(gt_occ)}")

        # Supported layouts:
        # - [T,1,X,Y,Z]
        # - [1,T,1,X,Y,Z]
        # - [T,X,Y,Z]
        if gt_occ_t.dim() == 6 and gt_occ_t.shape[0] == 1 and gt_occ_t.shape[2] == 1:
            gt_occ_t = gt_occ_t[0, :, 0]  # [T,X,Y,Z]
        elif gt_occ_t.dim() == 5 and gt_occ_t.shape[1] == 1:
            gt_occ_t = gt_occ_t[:, 0]     # [T,X,Y,Z]
        elif gt_occ_t.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported gt_occ shape for dice: {tuple(gt_occ_t.shape)}")

        if gt_occ_t.shape[0] < T_pred:
            raise ValueError(f"gt_occ time dim too short: {gt_occ_t.shape[0]} < {T_pred}")
        if gt_occ_t.shape[0] != T_pred:
            gt_occ_t = self._select_temporal_gt_for_prediction(gt_occ_t, T_pred)

        # [T,X,Y,Z] -> [T,Z,X,Y]
        gt_occ_t = gt_occ_t.permute(0, 3, 1, 2).contiguous()
        gt_occ_gmo = self.gt_occ_to_gmo_binary(
            gt_occ_t, gmo_ids=gmo_ids, ignore_index=ignore_index
        )

        valid = (gt_occ_gmo != ignore_index).float()
        target_fg = (gt_occ_gmo == 1).float()

        pred_f = pred.to(torch.float32)
        if pred_is_logits:
            pred_f = torch.sigmoid(pred_f)
        pred_f = pred_f.clamp(min=float(eps), max=1.0 - float(eps))

        # apply valid mask
        p_fg = pred_f * valid
        t_fg = target_fg * valid

        p_fg_flat = p_fg.reshape(T_pred, -1)
        t_fg_flat = t_fg.reshape(T_pred, -1)
        valid_flat = valid.reshape(T_pred, -1)

        inter_fg = (p_fg_flat * t_fg_flat).sum(dim=1)
        den_fg = p_fg_flat.sum(dim=1) + t_fg_flat.sum(dim=1)
        dice_fg = (2.0 * inter_fg + float(smooth)) / (den_fg + float(smooth) + float(eps))

        if include_bg:
            p_bg = (1.0 - pred_f) * valid
            t_bg = (1.0 - target_fg) * valid
            p_bg_flat = p_bg.reshape(T_pred, -1)
            t_bg_flat = t_bg.reshape(T_pred, -1)

            inter_bg = (p_bg_flat * t_bg_flat).sum(dim=1)
            den_bg = p_bg_flat.sum(dim=1) + t_bg_flat.sum(dim=1)
            dice_bg = (2.0 * inter_bg + float(smooth)) / (den_bg + float(smooth) + float(eps))
            dice = 0.5 * (dice_fg + dice_bg)
        else:
            dice = dice_fg

        valid_frames = (valid_flat.sum(dim=1) > 0)
        if not valid_frames.any():
            z = pred_f.sum() * 0.0
            return {"loss_gmo_dice": z}

        loss = (1.0 - dice[valid_frames]).mean() * float(loss_weight)
        return {"loss_gmo_dice": loss}

    def compute_objectness_loss(
        self,
        objectness_logits: torch.Tensor,
        objectness_targets: torch.Tensor,
        loss_weight: float = 0.1,
        valid_mask: torch.Tensor = None,
        loss_type: str = "balanced_bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        """
        Supervise per-query objectness.

        Args:
            objectness_logits: [T, Q] (before sigmoid)
            objectness_targets: [T, Q] in [0, 1]
            valid_mask: optional [T, Q] bool mask
        """
        if objectness_logits.shape != objectness_targets.shape:
            raise ValueError(
                f"objectness shape mismatch: {tuple(objectness_logits.shape)} vs "
                f"{tuple(objectness_targets.shape)}"
            )

        logits = objectness_logits.to(torch.float32)
        targets = objectness_targets.to(torch.float32).clamp_(0.0, 1.0)

        if valid_mask is None:
            valid = torch.ones_like(targets, dtype=torch.bool)
        else:
            if valid_mask.shape != targets.shape:
                raise ValueError(
                    f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(targets.shape)}"
                )
            valid = valid_mask.to(torch.bool)

        loss_type = str(loss_type).lower()
        if loss_type in ("balanced_bce", "bce"):
            loss_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        elif loss_type in ("balanced_focal", "focal"):
            bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            pt = probs * targets + (1.0 - probs) * (1.0 - targets)
            focal_w = (1.0 - pt).clamp(min=0.0).pow(float(focal_gamma))
            loss_map = bce_map * focal_w
            alpha = float(focal_alpha)
            if alpha >= 0.0:
                alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
                loss_map = loss_map * alpha_t
        else:
            raise ValueError(
                "Unsupported objectness loss_type. "
                f"Expected 'balanced_bce' or 'focal', got {loss_type!r}"
            )
        pos_mask = valid & (targets > 0.5)
        neg_mask = valid & (targets < 0.5)

        has_pos = bool(pos_mask.any().item())
        has_neg = bool(neg_mask.any().item())

        if has_pos:
            loss_pos = (loss_map * pos_mask.float()).sum() / pos_mask.sum().clamp(min=1).float()
        else:
            loss_pos = logits.sum() * 0.0
        if has_neg:
            loss_neg = (loss_map * neg_mask.float()).sum() / neg_mask.sum().clamp(min=1).float()
        else:
            loss_neg = logits.sum() * 0.0

        if has_pos and has_neg:
            loss = 0.5 * (loss_pos + loss_neg)
        elif has_pos:
            loss = loss_pos
        elif has_neg:
            loss = loss_neg
        else:
            loss = logits.sum() * 0.0

        scores = torch.sigmoid(logits)
        has_valid = bool(valid.any().item())
        pos_ratio = (targets[valid] > 0.5).float().mean() if has_valid else (logits.sum() * 0.0)
        return {
            "loss_query_objectness": loss * float(loss_weight),
            "dbg_query_objectness_score_mean": scores[valid].mean() if has_valid else (logits.sum() * 0.0),
            "dbg_query_objectness_pos_ratio": pos_ratio,
        }

    def _prepare_occ_dt(self, occ_dt, t_pred: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(occ_dt):
            if isinstance(occ_dt, np.ndarray):
                occ_dt = torch.from_numpy(occ_dt)
            else:
                raise TypeError(f"occ_dt must be torch.Tensor/np.ndarray, got {type(occ_dt)}")

        # collated by pipeline: [B, T, X, Y, Z], current query branch uses B=1.
        if occ_dt.dim() == 6 and occ_dt.shape[2] == 1:
            occ_dt = occ_dt.squeeze(2)
        if occ_dt.dim() == 5:
            if occ_dt.shape[0] != 1:
                raise ValueError(
                    f"current query path expects batch=1 for occ_dt, got {occ_dt.shape}"
                )
            occ_dt = occ_dt[0]
        elif occ_dt.dim() != 4:
            raise ValueError(f"occ_dt must be [B,T,X,Y,Z] or [T,X,Y,Z], got {occ_dt.shape}")

        if occ_dt.shape[0] < t_pred:
            raise ValueError(f"occ_dt time dim too short: {occ_dt.shape[0]} < {t_pred}")
        if occ_dt.shape[0] != t_pred:
            occ_dt = self._select_temporal_gt_for_prediction(occ_dt, t_pred)

        return occ_dt.to(device=device, dtype=torch.float32).contiguous()  # [T, X, Y, Z]

    def compute_query_point_dt_loss(
        self,
        points_world: torch.Tensor = None,
        occ_dt=None,
        loss_weight: float = 0.1,
        dt_clip_max: float = None,
        voxel_center_offset: float = 0.5,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
        gaussian_centers_world: torch.Tensor = None,
        gaussian_sigmas_world: torch.Tensor = None,
        gaussian_truncate_sigma: float = None,
        gaussian_sigma_floor_vox: float = 0.35,
    ):
        """
        p2gt loss.
        - gaussian 입력 시: E_{x~N(mu,sigma)}[DT(x)]를 근사 계산
        - legacy points 입력 시: point 샘플링 기반 DT loss
        """
        base_tensor = gaussian_centers_world if gaussian_centers_world is not None else points_world
        if occ_dt is None:
            if base_tensor is None:
                z = self.center_head.mlp.net[-1].weight.sum() * 0.0
                return {"loss_query_dt": z}
            z = base_tensor.sum() * 0.0
            return {"loss_query_dt": z}

        if self.center_only_mode and (gaussian_centers_world is not None) and (points_world is None):
            points_world = gaussian_centers_world
        use_gaussian = (not self.center_only_mode) and (
            (gaussian_centers_world is not None) or (gaussian_sigmas_world is not None)
        )
        if use_gaussian:
            if gaussian_centers_world is None or gaussian_sigmas_world is None:
                raise ValueError("gaussian_centers_world and gaussian_sigmas_world must be given together")
            if gaussian_centers_world.dim() != 3:
                raise ValueError(
                    f"gaussian_centers_world must be [T,Q,3], got {tuple(gaussian_centers_world.shape)}"
                )
            if gaussian_sigmas_world.shape != gaussian_centers_world.shape:
                raise ValueError(
                    f"gaussian_sigmas_world shape mismatch: {tuple(gaussian_sigmas_world.shape)} vs "
                    f"{tuple(gaussian_centers_world.shape)}"
                )

            centers = gaussian_centers_world.to(torch.float32)
            sigmas = gaussian_sigmas_world.to(torch.float32)
            T_pred, Q, _ = centers.shape

            occ_dt_t = self._prepare_occ_dt(occ_dt, T_pred, centers.device)
            if dt_clip_max is not None:
                occ_dt_t = occ_dt_t.clamp(max=float(dt_clip_max))

            X = int(occ_dt_t.shape[1])
            Y = int(occ_dt_t.shape[2])
            Z = int(occ_dt_t.shape[3])

            pc_min = centers.new_tensor(self.point_cloud_range[:3])
            extent = centers.new_tensor(self.spatial_extent3d)
            grid_size = centers.new_tensor([float(X), float(Y), float(Z)])
            voxel_size = extent / grid_size

            off = float(voxel_center_offset)
            cx = (centers[..., 0] - pc_min[0]) / voxel_size[0] - off
            cy = (centers[..., 1] - pc_min[1]) / voxel_size[1] - off
            cz = (centers[..., 2] - pc_min[2]) / voxel_size[2] - off

            sx = (sigmas[..., 0] / voxel_size[0]).clamp(min=float(gaussian_sigma_floor_vox))
            sy = (sigmas[..., 1] / voxel_size[1]).clamp(min=float(gaussian_sigma_floor_vox))
            sz = (sigmas[..., 2] / voxel_size[2]).clamp(min=float(gaussian_sigma_floor_vox))

            trunc = self.gaussian_truncate_sigma if gaussian_truncate_sigma is None else float(gaussian_truncate_sigma)
            trunc = max(1.0, float(trunc))

            loss_acc = centers.new_tensor(0.0)
            valid_cnt = 0

            for t in range(T_pred):
                dt_t = occ_dt_t[t]  # [X,Y,Z]
                for q in range(Q):
                    cxi, cyi, czi = cx[t, q], cy[t, q], cz[t, q]
                    sxi, syi, szi = sx[t, q], sy[t, q], sz[t, q]

                    rx = max(1, int(torch.ceil(trunc * sxi).item()))
                    ry = max(1, int(torch.ceil(trunc * syi).item()))
                    rz = max(1, int(torch.ceil(trunc * szi).item()))

                    x0 = max(0, int(torch.floor(cxi).item()) - rx)
                    x1 = min(X - 1, int(torch.floor(cxi).item()) + rx)
                    y0 = max(0, int(torch.floor(cyi).item()) - ry)
                    y1 = min(Y - 1, int(torch.floor(cyi).item()) + ry)
                    z0 = max(0, int(torch.floor(czi).item()) - rz)
                    z1 = min(Z - 1, int(torch.floor(czi).item()) + rz)

                    if (x1 < x0) or (y1 < y0) or (z1 < z0):
                        continue

                    xs = torch.arange(x0, x1 + 1, device=centers.device, dtype=torch.float32)
                    ys = torch.arange(y0, y1 + 1, device=centers.device, dtype=torch.float32)
                    zs = torch.arange(z0, z1 + 1, device=centers.device, dtype=torch.float32)

                    gx = torch.exp(-0.5 * ((xs - cxi) / sxi).pow(2))
                    gy = torch.exp(-0.5 * ((ys - cyi) / syi).pow(2))
                    gz = torch.exp(-0.5 * ((zs - czi) / szi).pow(2))
                    w = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]  # [Xw,Yw,Zw]

                    xx, yy, zz = torch.meshgrid(
                        xs.to(torch.long), ys.to(torch.long), zs.to(torch.long), indexing="ij"
                    )
                    dt_patch = dt_t[xx, yy, zz]  # [Xw,Yw,Zw]

                    wsum = w.sum().clamp(min=1e-6)
                    loss_acc = loss_acc + (w * dt_patch).sum() / wsum
                    valid_cnt += 1

            if valid_cnt == 0:
                z = centers.sum() * 0.0
                return {"loss_query_dt": z}

            loss = (loss_acc / float(valid_cnt)) * float(loss_weight)
            return {"loss_query_dt": loss}

        if points_world is None:
            raise ValueError("points_world is required when gaussian inputs are not provided")

        if points_world.dim() == 4:
            # [T, Nq, P, 3] -> [T, N, 3]
            T_pred = points_world.shape[0]
            pts_tn3 = points_world.reshape(T_pred, -1, 3)
        elif points_world.dim() == 3:
            # [T, N, 3]
            T_pred = points_world.shape[0]
            pts_tn3 = points_world
        else:
            raise ValueError(f"points_world must be [T,Nq,P,3] or [T,N,3], got {points_world.shape}")

        occ_dt_t = self._prepare_occ_dt(occ_dt, T_pred, points_world.device)
        if dt_clip_max is not None:
            occ_dt_t = occ_dt_t.clamp(max=float(dt_clip_max))

        # grid_sample expects input [N,C,D,H,W] with coord order (x,y,z) over (W,H,D).
        # our DT axis is [X,Y,Z], so permute to [Z,Y,X] as [D,H,W].
        dt_tzyx = occ_dt_t.permute(0, 3, 2, 1).unsqueeze(1).contiguous()  # [T,1,Z,Y,X]

        X = int(occ_dt_t.shape[1])
        Y = int(occ_dt_t.shape[2])
        Z = int(occ_dt_t.shape[3])

        pts = pts_tn3.to(dtype=torch.float32)
        pc_min = pts.new_tensor(self.point_cloud_range[:3])   # [x_min,y_min,z_min]
        extent = pts.new_tensor(self.spatial_extent3d)        # [x_range,y_range,z_range]
        grid_size = pts.new_tensor([float(X), float(Y), float(Z)])
        voxel_size = extent / grid_size

        # world xyz -> continuous voxel-center index in [x,y,z] axes.
        # ex) world at center of voxel k maps to index k when offset=0.5.
        off = float(voxel_center_offset)
        ix = (pts[..., 0] - pc_min[0]) / voxel_size[0] - off
        iy = (pts[..., 1] - pc_min[1]) / voxel_size[1] - off
        iz = (pts[..., 2] - pc_min[2]) / voxel_size[2] - off

        # voxel index -> normalized coords for grid_sample
        if X > 1:
            gx = (ix / float(X - 1)) * 2.0 - 1.0
        else:
            gx = torch.zeros_like(ix)
        if Y > 1:
            gy = (iy / float(Y - 1)) * 2.0 - 1.0
        else:
            gy = torch.zeros_like(iy)
        if Z > 1:
            gz = (iz / float(Z - 1)) * 2.0 - 1.0
        else:
            gz = torch.zeros_like(iz)

        sample_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(2).unsqueeze(2)  # [T,N,1,1,3]
        sampled_dt = F.grid_sample(
            dt_tzyx,
            sample_grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # [T,1,N,1,1]
        sampled_dt = sampled_dt[:, 0, :, 0, 0]  # [T,N]

        loss = sampled_dt.mean() * float(loss_weight)

        return {"loss_query_dt": loss}

    def compute_query_point_repel_loss(
        self,
        points_world: torch.Tensor,
        loss_weight: float = 0.01,
        k: int = 64,
        margin_m: float = 0.5,
    ):
        """
        Toy 코드의 repel_points와 동일한 의도:
        가까이 뭉친 query points에 약한 repulsion 페널티를 부여.

        Args:
            points_world:
                [T, Nq, P, 3] or [T, N, 3], world xyz(m).
            loss_weight:
                final scalar weight.
            k:
                frame마다 랜덤 샘플할 점 개수.
            margin_m:
                pair distance가 이 값보다 작을 때만 페널티.
        Returns:
            dict(loss_query_repel=...)
        """
        if points_world.dim() == 4:
            pts_tn3 = points_world.reshape(points_world.shape[0], -1, 3)
        elif points_world.dim() == 3:
            pts_tn3 = points_world
        else:
            raise ValueError(f"points_world must be [T,Nq,P,3] or [T,N,3], got {points_world.shape}")

        T, N, _ = pts_tn3.shape
        if N <= 1:
            z = pts_tn3.sum() * 0.0
            return {"loss_query_repel": z}

        kk = min(int(k), int(N))
        if kk <= 1:
            z = pts_tn3.sum() * 0.0
            return {"loss_query_repel": z}

        idx = torch.randint(0, N, (T, kk), device=pts_tn3.device)
        p = pts_tn3.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3)).to(torch.float32)  # [T,kk,3]

        d = torch.cdist(p, p)  # [T,kk,kk]
        eye = torch.eye(kk, device=p.device, dtype=torch.bool).unsqueeze(0)
        d = d.masked_fill(eye, 1e9)

        pen = F.relu(float(margin_m) - d)
        loss = pen.mean() * float(loss_weight)
        return {"loss_query_repel": loss}

    def compute_query_intra_repel_loss(
        self,
        points_world: torch.Tensor,
        loss_weight: float = 0.01,
        margin_m: float = 0.5,
        sample_k: int = 32,
        num_points_per_query: int = None,
        chunk_queries: int = 128,
    ):
        """
        Backward-compat name.
        실제 동작은 query center 간(inter-query) repulsion loss.
        프레임별로 query center들 사이 거리가 margin_m보다 가까우면 페널티를 준다.

        Args:
            points_world:
                [T, Q, P, 3] 또는 [T, N, 3] (N=Q*P).
            loss_weight:
                final scalar weight.
            margin_m:
                pair distance가 이 값보다 작을 때만 페널티.
            sample_k:
                프레임당 사용할 query center 수 (Q에서 랜덤 샘플).
                0 또는 음수면 전체 Q 사용.
            num_points_per_query:
                points_world가 [T,N,3]일 때 query당 point 수(P). None이면 self.num_pcd 사용.
            chunk_queries:
                메모리 절약용 frame chunk size.
        Returns:
            dict(loss_query_intra_repel=...)
        """
        if points_world.dim() == 4:
            centers_tq3 = points_world.mean(dim=2)
        elif points_world.dim() == 3:
            T, N, _ = points_world.shape
            P = int(num_points_per_query if num_points_per_query is not None else self.num_pcd)
            if P <= 0:
                raise ValueError(f"num_points_per_query must be positive, got {P}")
            if (N % P) == 0 and P > 1:
                Q = N // P
                centers_tq3 = points_world.view(T, Q, P, 3).mean(dim=2)
            else:
                # centers-only layout [T, Q, 3]
                centers_tq3 = points_world
        else:
            raise ValueError(f"points_world must be [T,Q,P,3] or [T,N,3], got {points_world.shape}")

        T, Q, _ = centers_tq3.shape
        if Q <= 1:
            z = centers_tq3.sum() * 0.0
            return {"loss_query_intra_repel": z}

        k = int(sample_k)
        if k <= 0 or k > Q:
            k = Q
        if k <= 1:
            z = centers_tq3.sum() * 0.0
            return {"loss_query_intra_repel": z}

        # frame별 query center 랜덤 샘플링
        if k < Q:
            # without replacement per frame
            idx = torch.rand(T, Q, device=centers_tq3.device).argsort(dim=1)[:, :k]
            centers = centers_tq3.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            centers = centers_tq3

        centers = centers.to(torch.float32)  # [T, k, 3]
        tri = torch.triu(torch.ones((k, k), device=centers.device, dtype=torch.bool), diagonal=1)
        pair_count = int(tri.sum().item())
        if pair_count <= 0:
            z = centers.sum() * 0.0
            return {"loss_query_intra_repel": z}

        step = max(1, int(chunk_queries))
        total_pen = centers.new_tensor(0.0)
        total_cnt = 0

        for s in range(0, T, step):
            c = centers[s:s + step]            # [mc, k, 3]
            d = torch.cdist(c, c)              # [mc, k, k]
            pen = F.relu(float(margin_m) - d[:, tri])  # [mc, pair_count]
            total_pen = total_pen + pen.sum()
            total_cnt += pen.numel()

        if total_cnt == 0:
            z = centers.sum() * 0.0
            return {"loss_query_intra_repel": z}

        loss = (total_pen / float(total_cnt)) * float(loss_weight)
        return {"loss_query_intra_repel": loss}

    def compute_query_gt2p_unlabeled_loss(
        self,
        gt_occ,
        points_world: torch.Tensor = None,
        gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        ignore_index: int = 255,
        loss_weight: float = 0.1,
        max_gt_samples: int = 512,
        pred_use_centers: bool = True,
        num_points_per_query: int = None,
        tau_softmin: float = 0.6,
        chunk_gt: int = 256,
        gaussian_centers_world: torch.Tensor = None,
        gaussian_sigmas_world: torch.Tensor = None,
        coverage_eps: float = 1e-6,
    ):
        """
        라벨-무관(unlabeled) GT->Pred coverage loss.
        - gaussian 입력 시: GT voxel이 Gaussian mixture에 커버되도록 NLL 최적화
          p(gt) = 1 - exp(-sum_q exp(-0.5 * mahalanobis^2))
        - legacy points 입력 시: 기존 softmin 거리 기반 loss
        """
        if (self.point_cloud_range is None) or (self.spatial_extent3d is None):
            raise ValueError("point_cloud_range/spatial_extent3d must be set for gt2p loss")

        if self.center_only_mode and (gaussian_centers_world is not None) and (points_world is None):
            points_world = gaussian_centers_world
        use_gaussian = (not self.center_only_mode) and (
            (gaussian_centers_world is not None) or (gaussian_sigmas_world is not None)
        )
        if use_gaussian:
            if gaussian_centers_world is None or gaussian_sigmas_world is None:
                raise ValueError("gaussian_centers_world and gaussian_sigmas_world must be given together")
            if gaussian_centers_world.dim() != 3:
                raise ValueError(
                    f"gaussian_centers_world must be [T,Q,3], got {tuple(gaussian_centers_world.shape)}"
                )
            if gaussian_sigmas_world.shape != gaussian_centers_world.shape:
                raise ValueError(
                    f"gaussian_sigmas_world shape mismatch: {tuple(gaussian_sigmas_world.shape)} vs "
                    f"{tuple(gaussian_centers_world.shape)}"
                )
            pred_tn3 = gaussian_centers_world.to(torch.float32)
            pred_sigma_tn3 = gaussian_sigmas_world.to(torch.float32).clamp(min=1e-3)
        else:
            if points_world is None:
                raise ValueError("points_world is required when gaussian inputs are not provided")
            if points_world.dim() == 4:
                # [T,Q,P,3]
                if pred_use_centers:
                    pred_tn3 = points_world.mean(dim=2)  # [T,Q,3]
                else:
                    pred_tn3 = points_world.reshape(points_world.shape[0], -1, 3)  # [T,Q*P,3]
            elif points_world.dim() == 3:
                # [T,N,3]
                if pred_use_centers:
                    T, N, _ = points_world.shape
                    P = int(num_points_per_query if num_points_per_query is not None else self.num_pcd)
                    if P <= 0:
                        raise ValueError(f"num_points_per_query must be positive, got {P}")
                    if (N % P) == 0 and P > 1:
                        pred_tn3 = points_world.view(T, N // P, P, 3).mean(dim=2)  # [T,Q,3]
                    else:
                        pred_tn3 = points_world  # centers-only layout
                else:
                    pred_tn3 = points_world  # [T,N,3]
            else:
                raise ValueError(f"points_world must be [T,Q,P,3] or [T,N,3], got {points_world.shape}")

        T_pred = int(pred_tn3.shape[0])

        # gt_occ -> [T, X, Y, Z]
        if isinstance(gt_occ, (list, tuple)):
            gt_occ_t = torch.stack(gt_occ, dim=0)
        elif torch.is_tensor(gt_occ):
            gt_occ_t = gt_occ
        else:
            raise TypeError(f"gt_occ must be list/tuple/tensor, got {type(gt_occ)}")

        # Common layouts:
        # - [T,1,X,Y,Z] (list stacked)
        # - [1,T,1,X,Y,Z] (batched single-sample)
        # - [T,X,Y,Z]
        if gt_occ_t.dim() == 6 and gt_occ_t.shape[0] == 1 and gt_occ_t.shape[2] == 1:
            gt_occ_t = gt_occ_t[0, :, 0]  # [T,X,Y,Z]
        elif gt_occ_t.dim() == 5 and gt_occ_t.shape[1] == 1:
            gt_occ_t = gt_occ_t[:, 0]      # [T,X,Y,Z]
        elif gt_occ_t.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported gt_occ shape for gt2p: {tuple(gt_occ_t.shape)}")

        if gt_occ_t.shape[0] < T_pred:
            raise ValueError(f"gt_occ time dim too short: {gt_occ_t.shape[0]} < {T_pred}")
        if gt_occ_t.shape[0] != T_pred:
            gt_occ_t = self._select_temporal_gt_for_prediction(gt_occ_t, T_pred)

        # [T,X,Y,Z] -> [T,Z,X,Y] (same convention used in compute_gmo_loss)
        gt_occ_t = gt_occ_t.permute(0, 3, 1, 2).contiguous()
        gt_bin = self.gt_occ_to_gmo_binary(gt_occ_t, gmo_ids=gmo_ids, ignore_index=ignore_index)

        Z = int(gt_bin.shape[1])
        X = int(gt_bin.shape[2])
        Y = int(gt_bin.shape[3])

        pred_f = pred_tn3.to(torch.float32)
        pc_min = pred_f.new_tensor(self.point_cloud_range[:3])
        extent = pred_f.new_tensor(self.spatial_extent3d)
        grid_size = pred_f.new_tensor([float(X), float(Y), float(Z)])
        voxel_size = extent / grid_size  # [vx,vy,vz]

        tau = float(tau_softmin)
        max_gt = int(max_gt_samples)
        step = max(1, int(chunk_gt))
        eps = float(coverage_eps)

        loss_acc = pred_f.new_tensor(0.0)
        valid_frames = 0

        for t in range(T_pred):
            pos = (gt_bin[t] == 1).nonzero(as_tuple=False)  # [M,3] in (z,x,y)
            if pos.numel() == 0:
                continue

            if max_gt > 0 and pos.shape[0] > max_gt:
                perm = torch.randperm(pos.shape[0], device=pos.device)[:max_gt]
                pos = pos[perm]

            pos = pos.to(device=pred_f.device)
            z = pos[:, 0].to(torch.float32)
            x = pos[:, 1].to(torch.float32)
            y = pos[:, 2].to(torch.float32)

            gt_xyz = torch.stack(
                [
                    pc_min[0] + (x + 0.5) * voxel_size[0],
                    pc_min[1] + (y + 0.5) * voxel_size[1],
                    pc_min[2] + (z + 0.5) * voxel_size[2],
                ],
                dim=-1,
            )  # [M,3]

            pred_xyz = pred_f[t]  # [N,3]
            if pred_xyz.numel() == 0:
                continue

            sum_t = pred_xyz.new_tensor(0.0)
            cnt_t = 0
            for s in range(0, gt_xyz.shape[0], step):
                g = gt_xyz[s:s + step]  # [mc,3]
                if use_gaussian:
                    pred_sig = pred_sigma_tn3[t]  # [N,3]
                    diff = (g[:, None, :] - pred_xyz[None, :, :]) / (pred_sig[None, :, :] + eps)
                    d2 = diff.pow(2).sum(dim=-1)  # [mc,N]
                    temp = tau if tau > 0.0 else 1.0
                    act = torch.exp(-0.5 * d2 / temp)  # [mc,N]
                    lam = act.sum(dim=1)               # [mc]
                    p_cov = (-torch.expm1(-lam)).clamp(min=eps, max=1.0)  # 1-exp(-lam)
                    dist = -torch.log(p_cov)
                else:
                    d = torch.cdist(g.unsqueeze(0), pred_xyz.unsqueeze(0)).squeeze(0)  # [mc,N]
                    if tau > 0.0:
                        dist = -tau * torch.logsumexp(-d / tau, dim=1)  # [mc]
                    else:
                        dist = d.min(dim=1).values

                sum_t = sum_t + dist.sum()
                cnt_t += int(dist.numel())

            if cnt_t > 0:
                loss_acc = loss_acc + (sum_t / float(cnt_t))
                valid_frames += 1

        if valid_frames == 0:
            z = pred_tn3.sum() * 0.0
            return {"loss_query_gt2p_unlabeled": z}

        loss = (loss_acc / float(valid_frames)) * float(loss_weight)
        return {"loss_query_gt2p_unlabeled": loss}

    def compute_query_gt2p_instance_labeled_loss(
        self,
        gaussian_centers_world: torch.Tensor,
        gaussian_sigmas_world: torch.Tensor = None,
        gt_inst_center_world_tn3: torch.Tensor = None,
        gt_inst_center_valid_tn: torch.Tensor = None,
        loss_weight: float = 0.1,
        balance_weight: float = 0.25,
        center_sigma_xyz=(4.0, 4.0, 1.5),
        sigma_policy: str = "fixed",
        tau: float = 1.0,
        coverage_eps: float = 1e-6,
    ):
        """
        Efficient center-only instance-labeled GT->Pred coverage loss using all queries.
        GT centers are assumed to come from instance-labeled annotations.

        목표:
        - 각 GT instance center가 전체 query center 집합으로 잘 커버되도록(coverage)
        - 각 query도 어떤 GT center 근처에 가도록(query coverage)
        - 동시에 query들이 특정 객체에만 몰리지 않고 present instances에 비교적 고르게 분배되도록(balance)

        Returns:
            dict(loss_query_gt2p_instance_labeled=..., dbg_*)
        """
        if not torch.is_tensor(gaussian_centers_world):
            raise TypeError("gaussian_centers_world must be a tensor")
        if gaussian_centers_world.dim() != 3:
            raise ValueError(
                f"gaussian_centers_world must be [T,Q,3], got {tuple(gaussian_centers_world.shape)}"
            )
        pred_mu_tq3 = gaussian_centers_world.to(torch.float32)
        T_pred, Q, _ = pred_mu_tq3.shape

        has_gt_centers = (
            torch.is_tensor(gt_inst_center_world_tn3)
            and torch.is_tensor(gt_inst_center_valid_tn)
            and gt_inst_center_world_tn3.dim() == 3
            and gt_inst_center_valid_tn.dim() == 2
            and gt_inst_center_world_tn3.shape[:2] == gt_inst_center_valid_tn.shape
            and gt_inst_center_world_tn3.shape[0] >= T_pred
            and gt_inst_center_world_tn3.shape[2] == 3
        )
        if not has_gt_centers:
            z = pred_mu_tq3.sum() * 0.0
            return {
                "loss_query_gt2p_instance_labeled": z,
                "dbg_query_gt2p_inst_cov_gt": z,
                "dbg_query_gt2p_inst_cov_q": z,
                "dbg_query_gt2p_inst_balance": z,
            }

        gt_centers_tn3 = gt_inst_center_world_tn3.to(device=pred_mu_tq3.device, dtype=torch.float32)
        gt_valid_tn = gt_inst_center_valid_tn.to(device=pred_mu_tq3.device, dtype=torch.bool)
        if gt_centers_tn3.shape[0] != T_pred:
            gt_centers_tn3 = self._select_temporal_gt_for_prediction(gt_centers_tn3, T_pred)
            gt_valid_tn = self._select_temporal_gt_for_prediction(gt_valid_tn, T_pred)

        fixed_sigma = None
        pred_sigma_tq3 = None
        if not self.center_only_mode:
            fixed_sigma = pred_mu_tq3.new_tensor(center_sigma_xyz, dtype=torch.float32)
            if fixed_sigma.numel() != 3:
                raise ValueError(f"center_sigma_xyz must be length-3, got {center_sigma_xyz}")
            fixed_sigma = fixed_sigma.clamp(min=1e-3).view(1, 1, 3)
            sigma_policy = str(sigma_policy).lower()
            if sigma_policy not in ("fixed", "pred_cov_q_only"):
                raise ValueError(
                    "sigma_policy must be one of {'fixed','pred_cov_q_only'}, "
                    f"got {sigma_policy!r}"
                )

            if sigma_policy == "pred_cov_q_only":
                if (not torch.is_tensor(gaussian_sigmas_world)) or gaussian_sigmas_world.dim() != 3:
                    raise ValueError(
                        "gaussian_sigmas_world must be [T,Q,3] when sigma_policy='pred_cov_q_only'"
                    )
                pred_sigma_tq3 = gaussian_sigmas_world.to(device=pred_mu_tq3.device, dtype=torch.float32)
                if pred_sigma_tq3.shape[0] < T_pred or pred_sigma_tq3.shape[1] != Q or pred_sigma_tq3.shape[2] != 3:
                    raise ValueError(
                        "gaussian_sigmas_world shape mismatch: "
                        f"got {tuple(pred_sigma_tq3.shape)}, expected [T>={T_pred},{Q},3]"
                    )
                if pred_sigma_tq3.shape[0] != T_pred:
                    pred_sigma_tq3 = pred_sigma_tq3[-T_pred:]
                pred_sigma_tq3 = pred_sigma_tq3.clamp(min=1e-3)

        eps = float(coverage_eps)
        tau_v = max(float(coverage_eps), float(tau))

        loss_cov_gt_acc = pred_mu_tq3.new_tensor(0.0)
        loss_cov_q_acc = pred_mu_tq3.new_tensor(0.0)
        cov_frame_cnt = 0
        loss_balance_acc = pred_mu_tq3.new_tensor(0.0)
        balance_frame_cnt = 0

        for t in range(T_pred):
            valid_n = gt_valid_tn[t]
            if not bool(valid_n.any().item()):
                continue

            gt_k3 = gt_centers_tn3[t, valid_n, :]   # [K,3]
            q_q3 = pred_mu_tq3[t]                   # [Q,3]
            if q_q3.numel() == 0:
                continue

            raw_diff_qk3 = q_q3[:, None, :] - gt_k3[None, :, :]          # [Q,K,3] in meters
            # GT->query attraction: Euclidean softmin distance in world space.
            d_qk_m = raw_diff_qk3.pow(2).sum(dim=-1).clamp(min=eps).sqrt()  # [Q,K], meters
            if self.center_only_mode:
                aff_balance_qk = torch.exp(-d_qk_m / tau_v)
                aff_cov_qk = aff_balance_qk
            else:
                fixed_diff_qk3 = raw_diff_qk3 / fixed_sigma
                d2_balance_qk = fixed_diff_qk3.pow(2).sum(dim=-1)            # [Q,K] normalized squared L2
                aff_balance_qk = torch.exp(-0.5 * d2_balance_qk / tau_v)     # [Q,K]
                if pred_sigma_tq3 is not None:
                    sigma_cov_q_q13 = pred_sigma_tq3[t][:, None, :]
                    diff_cov_q_qk3 = raw_diff_qk3 / sigma_cov_q_q13
                    d2_cov_qk = diff_cov_q_qk3.pow(2).sum(dim=-1)
                    aff_cov_qk = torch.exp(-0.5 * d2_cov_qk / tau_v)
                else:
                    aff_cov_qk = aff_balance_qk
            smin_k = -tau_v * torch.logsumexp(-d_qk_m / tau_v, dim=0)        # [K]
            loss_cov_gt_t = smin_k.mean()

            # Query coverage: every query should be close to at least one GT center.
            lam_q = aff_cov_qk.sum(dim=1)                    # [Q]
            p_hit_q = (-torch.expm1(-lam_q)).clamp(min=eps, max=1.0)
            loss_cov_q_t = -torch.log(p_hit_q).mean()

            loss_cov_gt_acc = loss_cov_gt_acc + loss_cov_gt_t
            loss_cov_q_acc = loss_cov_q_acc + loss_cov_q_t
            cov_frame_cnt += 1

            # Balance: distribute all queries across present GT instances roughly uniformly (Q/K).
            if float(balance_weight) > 0.0:
                resp_qk = aff_balance_qk / aff_balance_qk.sum(dim=1, keepdim=True).clamp(min=eps)  # [Q,K]
                mass_k = resp_qk.sum(dim=0)                                         # [K]
                k_inst = int(gt_k3.shape[0])
                target_mass = float(Q) / float(max(1, k_inst))
                target_k = mass_k.new_full((k_inst,), target_mass)
                loss_balance_t = F.smooth_l1_loss(
                    mass_k / float(max(1, Q)),
                    target_k / float(max(1, Q)),
                    reduction="mean",
                    beta=0.1,
                )
                loss_balance_acc = loss_balance_acc + loss_balance_t
                balance_frame_cnt += 1

        if cov_frame_cnt == 0:
            z = pred_mu_tq3.sum() * 0.0
            return {
                "loss_query_gt2p_instance_labeled": z,
                "dbg_query_gt2p_inst_cov_gt": z,
                "dbg_query_gt2p_inst_cov_q": z,
                "dbg_query_gt2p_inst_balance": z,
            }

        loss_cov_gt = loss_cov_gt_acc / float(cov_frame_cnt)
        loss_cov_q = loss_cov_q_acc / float(cov_frame_cnt)
        loss_cov = 0.5 * (loss_cov_gt + loss_cov_q)
        if (balance_frame_cnt > 0) and (float(balance_weight) > 0.0):
            loss_balance = loss_balance_acc / float(balance_frame_cnt)
        else:
            loss_balance = pred_mu_tq3.sum() * 0.0

        total = (loss_cov + float(balance_weight) * loss_balance) * float(loss_weight)
        return {
            "loss_query_gt2p_instance_labeled": total,
            "dbg_query_gt2p_inst_cov_gt": loss_cov_gt.detach(),
            "dbg_query_gt2p_inst_cov_q": loss_cov_q.detach(),
            "dbg_query_gt2p_inst_balance": loss_balance.detach(),
        }


    @torch.no_grad()
    def _sample_rows(self, idx: torch.Tensor, max_rows: int) -> torch.Tensor:
        # idx: (K, d)
        if max_rows is None or idx.shape[0] <= max_rows:
            return idx
        perm = torch.randperm(idx.shape[0], device=idx.device)[:max_rows]
        return idx[perm]

    def compute_gmo_center_knn_loss(
        self,
        center_logits: torch.Tensor,   # [T_pred, Nq, 3] (unconstrained)
        gt_occ,                        # list length >= T_pred, each [1, H, W, Z] (int labels)
        voxelizer,
        gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        ignore_index: int = 255,
        loss_weight: float = 0.1,
        max_gt: int = 4096,
        chunk_gt: int = 1024,
        max_pred: int = None,          # 보통 Nq=100이라 None 권장
        sqrt_dist: bool = True,
        eps: float = 1e-6,
    ):
        """
        목적:
          예측 query center들이 GT GMO 양성 영역(복셀 center 집합) 근처로 이동하도록 유도.

        정의(프레임별):
          pred_centers[t] = sigmoid(center_logits[t])를 world 범위로 변환한 [Nq,3]
          gt_points[t]    = GT에서 gmo 양성 복셀들의 world center 좌표 [K,3]
          loss[t] = mean_i min_j || pred_centers_i - gt_points_j ||  (KNN, pred->gt)

        반환:
          dict(loss_gmo_center_knn=...)
        """
        device = center_logits.device

        # 1) pred centers (world)
        pred_centers = sigmoid_to_world_from_range(
            center_logits, self.point_cloud_range, self.spatial_extent3d
        )  # [T_pred, Nq, 3]
        T_pred = pred_centers.shape[0]

        # 2) GT를 compute_gmo_loss와 동일한 방식으로 정렬: [T_pred, D, H, W]
        # gt_occ: [T_gt] list of [1,H,W,Z]
        gt_occ_raw = torch.stack(gt_occ, dim=0).squeeze(1)
        gt_occ_t = self._select_temporal_gt_for_prediction(gt_occ_raw, T_pred).permute(0, 3, 1, 2)  # [T_pred, Z, H, W]
        gt_bin = self.gt_occ_to_gmo_binary(gt_occ_t, gmo_ids=gmo_ids, ignore_index=ignore_index)

        # 3) voxelizer에서 world 변환 파라미터 확보
        # voxelizer.pc_min = [x_min,y_min,z_min], voxelizer.vs = [vx,vy,vz]
        if hasattr(voxelizer, "pc_min") and hasattr(voxelizer, "vs"):
            pc_min = voxelizer.pc_min.to(device=device, dtype=torch.float32)
            vs = voxelizer.vs.to(device=device, dtype=torch.float32)
            x_min, y_min, z_min = pc_min[0], pc_min[1], pc_min[2]
            vx, vy, vz = vs[0], vs[1], vs[2]
        else:
            pcr = voxelizer.point_cloud_range
            vsize = voxelizer.voxel_size
            if isinstance(vsize, (float, int)):
                vx = vy = vz = float(vsize)
            else:
                vx, vy, vz = float(vsize[0]), float(vsize[1]), float(vsize[2])
            x_min, y_min, z_min = float(pcr[0]), float(pcr[1]), float(pcr[2])
            x_min = torch.tensor(x_min, device=device, dtype=torch.float32)
            y_min = torch.tensor(y_min, device=device, dtype=torch.float32)
            z_min = torch.tensor(z_min, device=device, dtype=torch.float32)
            vx = torch.tensor(vx, device=device, dtype=torch.float32)
            vy = torch.tensor(vy, device=device, dtype=torch.float32)
            vz = torch.tensor(vz, device=device, dtype=torch.float32)

        losses = []
        for t in range(T_pred):
            g = gt_bin[t]              # [D,H,W]
            pos = (g == 1)

            if not pos.any():
                continue

            idx = pos.nonzero(as_tuple=False)  # [K,3] in (d,h,w)

            # GT 샘플링
            if max_gt is not None and idx.shape[0] > max_gt:
                with torch.no_grad():
                    perm = torch.randperm(idx.shape[0], device=device)[:max_gt]
                idx = idx[perm]

            d = idx[:, 0].to(torch.float32)
            h = idx[:, 1].to(torch.float32)
            w = idx[:, 2].to(torch.float32)

            # (d,h,w) -> world (x,y,z)
            gx = x_min + (w + 0.5) * vx
            gy = y_min + (h + 0.5) * vy
            gz = z_min + (d + 0.5) * vz
            gt_xyz = torch.stack([gx, gy, gz], dim=-1)  # [K,3]

            pred = pred_centers[t].to(torch.float32)    # [Nq,3]

            if max_pred is not None and pred.shape[0] > max_pred:
                with torch.no_grad():
                    perm = torch.randperm(pred.shape[0], device=device)[:max_pred]
                pred = pred[perm]

            # pred -> nearest GT (chunked)
            min_d2 = torch.full((pred.shape[0],), float("inf"), device=device, dtype=torch.float32)
            for s in range(0, gt_xyz.shape[0], chunk_gt):
                gt_chunk = gt_xyz[s:s + chunk_gt]  # [C,3]
                d2 = (pred[:, None, :] - gt_chunk[None, :, :]).pow(2).sum(dim=-1)  # [Np,C]
                min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)

            if sqrt_dist:
                loss_t = torch.sqrt(min_d2 + eps).mean()
            else:
                loss_t = min_d2.mean()

            losses.append(loss_t)

        if len(losses) == 0:
            # 모든 프레임에서 GT 양성이 없으면 0 loss
            z = center_logits.sum() * 0.0
            return {"loss_gmo_center_knn": z}

        loss = torch.stack(losses).mean() * float(loss_weight)
        return {"loss_gmo_center_knn": loss}




if __name__ == "__main__":
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    spatial_extent3d = [102.4, 102.4, 8.0]

    net = QueryHead(
        embed_dim=128,
        center_out_dim=3,
        offset_out_dim=3,  # legacy arg name, used as gaussian head out dim(=3)
        num_pcd=256,
        point_cloud_range=point_cloud_range,
        spatial_extent3d=spatial_extent3d,
    )

    query = torch.randn(3, 100, 128)  # [T, num_queries, embed_dim]
    out = net(query, compute_direct_center=True)

    print(out["centers_world_tq3"].shape)         # [T, Q, 3]
    print(out["center_logits_tq3"].shape)         # [T, Q, 3]
    print(out["query_sigma_world_tq3"].shape)     # [T, Q, 3]
    print(out["mixture_centers_world_tqg3"].shape)  # [T, Q, G, 3]
    print(out["mixture_sigmas_world_tqg3"].shape)   # [T, Q, G, 3]
    print(out["mixture_yaw_tqg"].shape)             # [T, Q, G]
    print(out["mixture_weights_tqg"].shape)         # [T, Q, G]
    print(out["query_cls_logits_qc"].shape)         # [Q, C]
