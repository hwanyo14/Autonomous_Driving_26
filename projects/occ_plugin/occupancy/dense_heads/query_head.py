import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import os

from projects.occ_plugin.occupancy.dense_heads.voxelizer import (
    normalize_quat_wxyz,
    quat_to_rotmat_wxyz,
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
                 out_dim=3,
                 num_layers=2):
        super(GaussianHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        self.mlp = MLP(in_dim=self.embed_dim,
                       hidden_dim=self.embed_dim,
                       out_dim=self.out_dim,
                       num_layers=self.num_layers,
                       dropout=0.0,
                       use_ln=True)

    def forward(self, query_inst):
        gaussian_logits = self.mlp(query_inst)  # [T, num_queries, 3]
        return gaussian_logits


class ClassificationHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 out_dim=3,
                 num_layers=2):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        self.mlp = MLP(in_dim=self.embed_dim,
                       hidden_dim=self.embed_dim,
                       out_dim=self.out_dim,
                       num_layers=self.num_layers,
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
    

class QuerySizeEmbedding(nn.Module):
    def __init__(self, embed_dim=128, gate_init=0.0):
        super(QuerySizeEmbedding, self).__init__()
        hidden_dim = max(16, int(embed_dim) // 4)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, int(embed_dim), bias=False),
        )
        self.gate = nn.Parameter(torch.ones(1) * float(gate_init), requires_grad=True)

    def forward(self, query_size_log_scalar_tq):
        return self.mlp(query_size_log_scalar_tq.unsqueeze(-1))


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
                 query_binary_cls=False,
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
                 query_gaussian_head_num_layers=2,
                 query_cls_head_num_layers=2,
                 query_cls_use_gaussian_params=False,
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
                 debug_query_gaussian_prob_alpha_scale=4.0,
                 use_query_size_embedding=False,
                 query_size_attn_threshold=0.5,
                 query_size_depth_ref_m=20.0,
                 query_size_log_alpha=10.0,
                 query_size_gate_init=0.0):
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
        # In binary mode query_class_ids stays the full foreground-id registry
        # (e.g. [0,2,3,4,5,6,9,10]) while the label space is just bg/fg, so the
        # length no longer matches num_query_classes.
        self.query_binary_cls = bool(query_binary_cls)
        if query_class_ids is None:
            query_class_ids = tuple(range(self.num_query_classes))
        else:
            query_class_ids = tuple(int(v) for v in query_class_ids)
        if not self.query_binary_cls and len(query_class_ids) != self.num_query_classes:
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
        self.query_gaussian_head_num_layers = int(query_gaussian_head_num_layers)
        if self.query_gaussian_head_num_layers <= 0:
            raise ValueError(
                "query_gaussian_head_num_layers must be positive, "
                f"got {self.query_gaussian_head_num_layers}"
            )
        self.query_cls_head_num_layers = int(query_cls_head_num_layers)
        if self.query_cls_head_num_layers <= 0:
            raise ValueError(
                "query_cls_head_num_layers must be positive, "
                f"got {self.query_cls_head_num_layers}"
            )
        self.query_cls_use_gaussian_params = bool(query_cls_use_gaussian_params)
        self.use_query_size_embedding = bool(use_query_size_embedding)
        self.query_size_attn_threshold = float(query_size_attn_threshold)
        self.query_size_depth_ref_m = float(query_size_depth_ref_m)
        self.query_size_log_alpha = float(query_size_log_alpha)
        self.query_size_gate_init = float(query_size_gate_init)
        if self.query_size_depth_ref_m <= 0.0:
            raise ValueError(
                "query_size_depth_ref_m must be positive, "
                f"got {self.query_size_depth_ref_m}"
            )
        if self.query_size_log_alpha < 0.0:
            raise ValueError(
                "query_size_log_alpha must be non-negative, "
                f"got {self.query_size_log_alpha}"
            )
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
        if not self.center_only_mode:
            g = int(self.query_num_gaussians)
            self.gaussian_head = GaussianHead(
                embed_dim=self.embed_dim,
                out_dim=g * 11,
                num_layers=self.query_gaussian_head_num_layers,
            )
        self.query_cls_gaussian_param_dim = int(self.query_num_gaussians) * 11
        self.query_cls_input_dim = self.embed_dim + (
            self.query_cls_gaussian_param_dim if self.query_cls_use_gaussian_params else 0
        )
        self.cls_head = ClassificationHead(
            embed_dim=self.query_cls_input_dim,
            out_dim=self.num_query_classes,
            num_layers=self.query_cls_head_num_layers,
        )
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
        self.query_size_embedding = (
            QuerySizeEmbedding(embed_dim=self.embed_dim, gate_init=self.query_size_gate_init)
            if self.use_query_size_embedding else None
        )
        self.last_query_size_dbg = None

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

    @staticmethod
    def _stats_dict(prefix, values, z):
        if (not torch.is_tensor(values)) or values.numel() <= 0:
            return {
                f"{prefix}_min": z,
                f"{prefix}_max": z,
                f"{prefix}_mean": z,
            }
        vals = values.to(torch.float32)
        valid = torch.isfinite(vals)
        if not bool(valid.any().item()):
            return {
                f"{prefix}_min": z,
                f"{prefix}_max": z,
                f"{prefix}_mean": z,
            }
        vals = vals[valid]
        return {
            f"{prefix}_min": vals.min().detach(),
            f"{prefix}_max": vals.max().detach(),
            f"{prefix}_mean": vals.mean().detach(),
        }

    def compute_query_size_log_scalar(
        self,
        query_attn_weights_tqnhw,
        query_depth_m_tq,
    ):
        if (
            (not self.use_query_size_embedding)
            or (not torch.is_tensor(query_attn_weights_tqnhw))
            or query_attn_weights_tqnhw.dim() != 5
            or (not torch.is_tensor(query_depth_m_tq))
            or query_depth_m_tq.dim() != 2
        ):
            return None, None

        t_attn, q_attn, _n_cam, h, w = [int(v) for v in query_attn_weights_tqnhw.shape]
        t_depth, q_depth = [int(v) for v in query_depth_m_tq.shape]
        t_count = min(t_attn, t_depth)
        q_count = min(q_attn, q_depth)
        if t_count <= 0 or q_count <= 0 or h <= 0 or w <= 0:
            return None, None

        attn = query_attn_weights_tqnhw[:t_count, :q_count].detach().to(torch.float32)
        depth = query_depth_m_tq[:t_count, :q_count].detach().to(device=attn.device, dtype=torch.float32)
        pixel_count_tqn = (attn > float(self.query_size_attn_threshold)).to(torch.float32).sum(dim=(-1, -2))
        pixel_count_tq = pixel_count_tqn.max(dim=-1).values
        area_norm = pixel_count_tq / float(h * w)
        depth_scale = (depth.clamp_min(0.0) / float(self.query_size_depth_ref_m)).pow(2)
        size_scalar_tq = area_norm * depth_scale
        log_scalar_tq = torch.log1p(float(self.query_size_log_alpha) * size_scalar_tq)

        z = log_scalar_tq.sum() * 0.0
        dbg = {}
        dbg.update(self._stats_dict("dbg_query_size_depth", depth, z))
        dbg.update(self._stats_dict("dbg_query_size_pixel_count", pixel_count_tq, z))
        dbg.update(self._stats_dict("dbg_query_size_log_scalar", log_scalar_tq, z))
        return log_scalar_tq.to(dtype=query_attn_weights_tqnhw.dtype), dbg

    def _apply_query_size_embedding(
        self,
        center_input_tqd,
        present_local_idx,
        query_size_log_scalar_tq=None,
    ):
        if (
            (not self.use_query_size_embedding)
            or self.query_size_embedding is None
            or (not torch.is_tensor(query_size_log_scalar_tq))
            or query_size_log_scalar_tq.dim() != 2
        ):
            self.last_query_size_dbg = None
            return center_input_tqd[int(present_local_idx)]

        q_feat = center_input_tqd[int(present_local_idx)]
        if (
            int(query_size_log_scalar_tq.shape[0]) <= int(present_local_idx)
            or int(query_size_log_scalar_tq.shape[1]) != int(q_feat.shape[0])
        ):
            self.last_query_size_dbg = None
            return q_feat

        e_q = query_size_log_scalar_tq[int(present_local_idx)].to(device=q_feat.device, dtype=q_feat.dtype)
        z_qd = self.query_size_embedding(e_q)
        q_out = q_feat + self.query_size_embedding.gate.to(dtype=q_feat.dtype) * z_qd

        z = q_feat.sum() * 0.0
        self.last_query_size_dbg = {
            "dbg_query_size_embedding_mean": z_qd.to(torch.float32).mean().detach(),
            "dbg_query_size_embedding_var": z_qd.to(torch.float32).var(unbiased=False).detach(),
            "dbg_query_size_target_feat_mean": q_feat.to(torch.float32).mean().detach(),
            "dbg_query_size_target_feat_var": q_feat.to(torch.float32).var(unbiased=False).detach(),
            "dbg_query_size_gate": self.query_size_embedding.gate.detach().reshape(()).to(torch.float32),
        }
        return q_out

    def _world_to_range_logits(self, centers_world: torch.Tensor) -> torch.Tensor:
        pc_min = centers_world.new_tensor(self.point_cloud_range[:3])
        extent = centers_world.new_tensor(self.spatial_extent3d).clamp_min(1e-6)
        p = ((centers_world - pc_min) / extent).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(p / (1.0 - p))

    def _compute_gaussian_outputs(
        self,
        center_input_tqd: torch.Tensor,
        centers_world: torch.Tensor,
        query_size_log_scalar_tq=None,
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
            query_sigma_world_tq3:        None (summary Gaussian disabled)
            mixture_centers_world_tqg3:   [T, Q, G, 3]   (per-frame placement)
            mixture_sigmas_world_tqg3:    [T, Q, G, 3]   (shared across T)
            mixture_quat_tqg4:              [T, Q, G, 4]   (shared across T, wxyz)
            mixture_weights_tqg:          [T, Q, G]      (independent sigmoid alpha)
        """
        query_sigma_world_tq3 = None
        mixture_centers_world_tqg3 = None
        mixture_sigmas_world_tqg3 = None
        mixture_quat_tqg4 = None
        mixture_weights_tqg = None

        if self.gaussian_head is None:
            return (
                query_sigma_world_tq3,
                mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3,
                mixture_quat_tqg4,
                mixture_weights_tqg,
            )

        t_count, q_count = int(center_input_tqd.shape[0]), int(center_input_tqd.shape[1])
        if t_count <= 0 or q_count <= 0:
            return (
                query_sigma_world_tq3,
                mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3,
                mixture_quat_tqg4,
                mixture_weights_tqg,
            )
        g_count = int(self.query_num_gaussians)

        # Run the unified Gaussian head on the present-frame feature only.
        present_local_idx = self._resolve_present_query_local_idx(t_count)
        center_input_qd = self._apply_query_size_embedding(
            center_input_tqd,
            present_local_idx,
            query_size_log_scalar_tq=query_size_log_scalar_tq,
        )  # [Q, D]

        gaussian_logits_qg11 = self.gaussian_head(center_input_qd).reshape(q_count, g_count, 11)
        offset_logits_qg3 = gaussian_logits_qg11[..., 0:3]
        sigma_logits_qg3 = gaussian_logits_qg11[..., 3:6]
        quat_logits_qg4 = gaussian_logits_qg11[..., 6:10]
        weight_logits_qg = gaussian_logits_qg11[..., 10]

        off_max = centers_world.new_tensor(self.query_multi_gaussian_offset_max_m).view(1, 1, 3)
        offsets_world_qg3 = torch.tanh(offset_logits_qg3) * off_max

        sigmas_world_qg3 = sigmoid_to_sigma_from_range(
            sigma_logits_qg3,
            sigma_min=self.query_multi_gaussian_sigma_min_m,
            sigma_max=self.query_multi_gaussian_sigma_max_m,
        )

        quat_qg4 = normalize_quat_wxyz(quat_logits_qg4)

        weights_qg = torch.sigmoid(weight_logits_qg).to(dtype=centers_world.dtype)

        # Broadcast the shared mixture to each frame:
        # - mixture centers move with the per-frame anchor (lifted center).
        # - sigmas/quaternion/weights are identical across the T frames.
        mixture_centers_world_tqg3 = centers_world.unsqueeze(2) + offsets_world_qg3.unsqueeze(0)
        pc_min_g = centers_world.new_tensor(self.point_cloud_range[:3]).view(1, 1, 1, 3)
        pc_max_g = centers_world.new_tensor(self.point_cloud_range[3:]).view(1, 1, 1, 3)
        mixture_centers_world_tqg3 = torch.max(torch.min(mixture_centers_world_tqg3, pc_max_g), pc_min_g)

        mixture_sigmas_world_tqg3 = sigmas_world_qg3.unsqueeze(0).expand(t_count, -1, -1, -1).contiguous()
        mixture_quat_tqg4 = quat_qg4.unsqueeze(0).expand(t_count, -1, -1, -1).contiguous()
        mixture_weights_tqg = weights_qg.unsqueeze(0).expand(t_count, -1, -1).contiguous()

        return (
            query_sigma_world_tq3,
            mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3,
            mixture_quat_tqg4,
            mixture_weights_tqg,
        )

    def apply_lifted_centers_to_outputs(
        self,
        outputs: dict,
        centers_world: torch.Tensor,
        detach_query_for_center: bool = None,
        defer_trajectory: bool = False,
        query_size_log_scalar_tq=None,
        query_size_dbg=None,
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
            mixture_quat_tqg4,
            mixture_weights_tqg,
        ) = self._compute_gaussian_outputs(
            center_input_tqd,
            centers_world,
            query_size_log_scalar_tq=query_size_log_scalar_tq,
        )
        if isinstance(query_size_dbg, dict) or isinstance(self.last_query_size_dbg, dict):
            merged_size_dbg = {}
            if isinstance(query_size_dbg, dict):
                merged_size_dbg.update(query_size_dbg)
            if isinstance(self.last_query_size_dbg, dict):
                merged_size_dbg.update(self.last_query_size_dbg)
        else:
            merged_size_dbg = None
        query_cls_logits_qc = outputs.get("query_cls_logits_qc", None)
        query_cls_scores_qc = outputs.get("query_cls_scores_qc", None)
        query_cls_scores_tqc = outputs.get("query_cls_scores_tqc", None)
        if self.query_cls_use_gaussian_params:
            (
                _raw_sigma_world_tq3,
                cls_mix_centers_world_tqg3,
                cls_mix_sigmas_world_tqg3,
                cls_mix_quat_tqg4,
                cls_mix_weights_tqg,
            ) = self._compute_gaussian_outputs(center_input_tqd, centers_world)
            query_inst_tqd = outputs.get("query_inst_tqd", query_feat_tqd)
            (
                query_cls_logits_qc,
                query_cls_scores_qc,
                query_cls_scores_tqc,
            ) = self._compute_query_cls_outputs(
                query_inst_tqd=query_inst_tqd,
                query_future_feat_tqd=query_feat_tqd,
                centers_world_tq3=centers_world,
                mixture_centers_world_tqg3=cls_mix_centers_world_tqg3,
                mixture_sigmas_world_tqg3=cls_mix_sigmas_world_tqg3,
                mixture_quat_tqg4=cls_mix_quat_tqg4,
                mixture_weights_tqg=cls_mix_weights_tqg,
            )

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
            "mixture_quat_tqg4": mixture_quat_tqg4,
            "mixture_weights_tqg": mixture_weights_tqg,
            "query_cls_logits_qc": query_cls_logits_qc,
            "query_cls_scores_qc": query_cls_scores_qc,
            "query_cls_scores_tqc": query_cls_scores_tqc,
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
            "query_size_dbg": merged_size_dbg,
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

    def _build_query_cls_gaussian_feature_qd(
        self,
        centers_world_tq3: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        q_count: int = None,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        g_count = int(self.query_num_gaussians)
        if q_count is None:
            for tensor in (
                centers_world_tq3,
                mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3,
                mixture_quat_tqg4,
                mixture_weights_tqg,
            ):
                if torch.is_tensor(tensor) and tensor.dim() >= 2:
                    q_count = int(tensor.shape[1])
                    break
        q_count = 0 if q_count is None else int(q_count)
        if device is None:
            for tensor in (
                centers_world_tq3,
                mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3,
                mixture_quat_tqg4,
                mixture_weights_tqg,
            ):
                if torch.is_tensor(tensor):
                    device = tensor.device
                    break
        if dtype is None:
            dtype = torch.float32

        valid = (
            torch.is_tensor(centers_world_tq3)
            and torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_quat_tqg4)
            and torch.is_tensor(mixture_weights_tqg)
            and centers_world_tq3.dim() == 3
            and mixture_centers_world_tqg3.dim() == 4
            and centers_world_tq3.shape[:2] == mixture_centers_world_tqg3.shape[:2]
            and mixture_sigmas_world_tqg3.shape == mixture_centers_world_tqg3.shape
            and mixture_quat_tqg4.dim() == 4
            and int(mixture_quat_tqg4.shape[-1]) == 4
            and mixture_quat_tqg4.shape[:3] == mixture_centers_world_tqg3.shape[:3]
            and mixture_weights_tqg.shape[:3] == mixture_centers_world_tqg3.shape[:3]
            and int(mixture_centers_world_tqg3.shape[1]) == q_count
            and int(mixture_centers_world_tqg3.shape[2]) == g_count
        )
        if not valid or q_count <= 0:
            return torch.zeros(
                (q_count, self.query_cls_gaussian_param_dim),
                device=device,
                dtype=dtype,
            )

        present_idx = self._resolve_present_query_local_idx(int(centers_world_tq3.shape[0]))
        center_q13 = centers_world_tq3[present_idx].unsqueeze(1)
        offset_qg3 = mixture_centers_world_tqg3[present_idx] - center_q13
        sigma_qg3 = mixture_sigmas_world_tqg3[present_idx]
        quat_qg4 = mixture_quat_tqg4[present_idx]
        weight_qg1 = mixture_weights_tqg[present_idx].unsqueeze(-1)
        params_qg11 = torch.cat(
            [
                offset_qg3,
                sigma_qg3,
                quat_qg4,
                weight_qg1,
            ],
            dim=-1,
        )
        return params_qg11.detach().to(device=device, dtype=dtype).reshape(q_count, g_count * 11)

    def _compute_query_cls_outputs(
        self,
        query_inst_tqd: torch.Tensor,
        query_future_feat_tqd: torch.Tensor,
        centers_world_tq3: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
    ):
        cls_feat_qd = self._build_query_cls_feature_qd(query_inst_tqd, query_future_feat_tqd)
        if self.query_cls_use_gaussian_params:
            gaussian_feat_qd = self._build_query_cls_gaussian_feature_qd(
                centers_world_tq3=centers_world_tq3,
                mixture_centers_world_tqg3=mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3=mixture_sigmas_world_tqg3,
                mixture_quat_tqg4=mixture_quat_tqg4,
                mixture_weights_tqg=mixture_weights_tqg,
                q_count=int(cls_feat_qd.shape[0]),
                device=cls_feat_qd.device,
                dtype=cls_feat_qd.dtype,
            )
            cls_feat_qd = torch.cat([cls_feat_qd, gaussian_feat_qd], dim=-1)
        cls_logits_qc = self.cls_head(cls_feat_qd)
        cls_scores_qc = torch.softmax(cls_logits_qc, dim=-1)
        cls_scores_tqc = cls_scores_qc.unsqueeze(0).expand(
            int(query_future_feat_tqd.shape[0]), -1, -1
        )
        return cls_logits_qc, cls_scores_qc, cls_scores_tqc

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

    @staticmethod
    def _draw_marker_splats(
        canvas: np.ndarray,
        x_idx: np.ndarray,
        y_idx: np.ndarray,
        color,
        radius: int,
        use_max: bool = False,
    ) -> None:
        if canvas.ndim != 3 or canvas.shape[-1] != 3:
            return
        if x_idx is None or y_idx is None:
            return
        if len(x_idx) <= 0 or len(y_idx) <= 0:
            return
        radius = max(0, int(radius))
        height, width = canvas.shape[:2]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xx = np.clip(x_idx + dx, 0, width - 1)
                yy = np.clip(y_idx + dy, 0, height - 1)
                if use_max:
                    canvas[yy, xx] = np.maximum(canvas[yy, xx], color)
                else:
                    canvas[yy, xx] = color

    @staticmethod
    def _draw_cross_marker(
        canvas: np.ndarray,
        center_x: int,
        center_y: int,
        color,
        arm: int = 6,
        thickness: int = 1,
        outline_color=(0, 0, 0),
        outline_thickness: int = 1,
        use_max: bool = False,
    ) -> None:
        if canvas.ndim != 3 or canvas.shape[-1] != 3:
            return
        height, width = canvas.shape[:2]
        cx = int(center_x)
        cy = int(center_y)
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            return

        arm = max(1, int(arm))
        thickness = max(1, int(thickness))
        outline_thickness = max(0, int(outline_thickness))

        def _draw_plus(draw_color, draw_arm: int, draw_thickness: int, draw_use_max: bool):
            xs = np.arange(cx - draw_arm, cx + draw_arm + 1, dtype=np.int64)
            ys = np.full_like(xs, fill_value=cy)
            QueryHead._draw_marker_splats(
                canvas,
                xs,
                ys,
                np.asarray(draw_color, dtype=np.uint8),
                radius=max(0, int(draw_thickness) - 1),
                use_max=bool(draw_use_max),
            )
            ys = np.arange(cy - draw_arm, cy + draw_arm + 1, dtype=np.int64)
            xs = np.full_like(ys, fill_value=cx)
            QueryHead._draw_marker_splats(
                canvas,
                xs,
                ys,
                np.asarray(draw_color, dtype=np.uint8),
                radius=max(0, int(draw_thickness) - 1),
                use_max=bool(draw_use_max),
            )

        if outline_thickness > 0 and outline_color is not None:
            _draw_plus(
                outline_color,
                draw_arm=arm + outline_thickness,
                draw_thickness=thickness + (2 * outline_thickness),
                draw_use_max=False,
            )
        _draw_plus(color, draw_arm=arm, draw_thickness=thickness, draw_use_max=use_max)

    @staticmethod
    def _draw_gaussian_bev_footprints(
        canvas: np.ndarray,
        x_idx: np.ndarray,
        y_idx: np.ndarray,
        sigma_x_px: np.ndarray,
        sigma_y_px: np.ndarray,
        color,
        sigma_z_px: np.ndarray = None,
        quat_wxyz: np.ndarray = None,
        component_weight: np.ndarray = None,
        valid_mask: np.ndarray = None,
        vis_mode: str = "ellipse",
        truncate_sigma: float = 3.0,
        fill_alpha: float = 0.18,
        outline_alpha: float = 0.60,
        outline_thickness_px: int = 1,
        prob_threshold: float = 0.5,
        prob_alpha_scale: float = 4.0,
        use_max: bool = True,
    ) -> None:
        if canvas.ndim != 3 or canvas.shape[-1] != 3:
            return
        if x_idx is None or y_idx is None or sigma_x_px is None or sigma_y_px is None:
            return

        vis_mode = str(vis_mode).lower()
        if vis_mode not in ("ellipse", "prob", "threshold"):
            return

        x_idx = np.asarray(x_idx)
        y_idx = np.asarray(y_idx)
        sigma_x_px = np.asarray(sigma_x_px, dtype=np.float32)
        sigma_y_px = np.asarray(sigma_y_px, dtype=np.float32)
        if x_idx.shape != y_idx.shape or x_idx.shape != sigma_x_px.shape or x_idx.shape != sigma_y_px.shape:
            return
        sigma_z_px = np.ones_like(sigma_x_px, dtype=np.float32) if sigma_z_px is None else np.asarray(sigma_z_px, dtype=np.float32)
        if sigma_z_px.shape != x_idx.shape:
            return
        if x_idx.ndim == 1:
            x_idx = x_idx[:, None]
            y_idx = y_idx[:, None]
            sigma_x_px = sigma_x_px[:, None]
            sigma_y_px = sigma_y_px[:, None]
            sigma_z_px = sigma_z_px[:, None]
        elif x_idx.ndim != 2:
            return

        num_queries, num_components = x_idx.shape
        if num_queries <= 0 or num_components <= 0:
            return

        color_arr = np.asarray(color, dtype=np.uint8)
        if color_arr.ndim == 1:
            color_arr = np.repeat(color_arr[None, :], repeats=num_queries, axis=0)
        elif color_arr.ndim != 2 or color_arr.shape[0] != num_queries or color_arr.shape[1] != 3:
            return

        if quat_wxyz is None:
            quat_arr = np.zeros((num_queries, num_components, 4), dtype=np.float32)
            quat_arr[..., 0] = 1.0
        else:
            quat_arr = np.asarray(quat_wxyz, dtype=np.float32)
            if quat_arr.shape != x_idx.shape + (4,):
                return

        if component_weight is None:
            weight_arr = np.ones((num_queries, num_components), dtype=np.float32)
        else:
            weight_arr = np.asarray(component_weight, dtype=np.float32)
            if weight_arr.shape != x_idx.shape:
                return

        if valid_mask is None:
            valid_arr = np.ones((num_queries, num_components), dtype=np.bool_)
        else:
            valid_arr = np.asarray(valid_mask, dtype=np.bool_)
            if valid_arr.shape != x_idx.shape:
                return

        height, width = canvas.shape[:2]
        trunc = max(1.0, float(truncate_sigma))
        thickness = max(1, int(outline_thickness_px))
        prob_threshold = float(np.clip(prob_threshold, 0.0, 1.0))
        prob_alpha_scale = max(0.0, float(prob_alpha_scale))

        def _bev_gaussian_params(q_idx, g_idx):
            sx = float(abs(sigma_x_px[q_idx, g_idx]))
            sy = float(abs(sigma_y_px[q_idx, g_idx]))
            sz = float(abs(sigma_z_px[q_idx, g_idx]))
            if (not np.isfinite(sx)) or (not np.isfinite(sy)) or (not np.isfinite(sz)):
                return None
            q = quat_arr[q_idx, g_idx].astype(np.float32)
            norm = float(np.linalg.norm(q))
            if (not np.isfinite(norm)) or norm <= 1e-6:
                q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                q = q / norm
            qw, qx, qy, qz = [float(v) for v in q]
            rot = np.array(
                [
                    [qw * qw + qx * qx - qy * qy - qz * qz, 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
                    [2.0 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2.0 * (qy * qz - qw * qx)],
                    [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz],
                ],
                dtype=np.float32,
            )
            sig2 = np.array([sx * sx, sy * sy, sz * sz], dtype=np.float32)
            cov_xx = float(np.sum(rot[0, :] * rot[0, :] * sig2))
            cov_xy = float(np.sum(rot[0, :] * rot[1, :] * sig2))
            cov_yy = float(np.sum(rot[1, :] * rot[1, :] * sig2))
            det = max(cov_xx * cov_yy - cov_xy * cov_xy, 1e-12)
            return (
                max(cov_xx, 1e-12),
                cov_xy,
                max(cov_yy, 1e-12),
                cov_yy / det,
                -cov_xy / det,
                cov_xx / det,
            )

        def _blend_patch(dst_patch, mask, rgb_color, alpha_scale):
            if not np.any(mask):
                return
            alpha = np.asarray(alpha_scale, dtype=np.float32)
            if alpha.ndim == 0:
                alpha = np.full(mask.shape, fill_value=float(alpha), dtype=np.float32)
            alpha = np.clip(alpha, 0.0, 1.0)
            if not np.any(alpha > 0.0):
                return
            src = np.clip(alpha[..., None] * rgb_color[None, None, :], 0.0, 255.0).astype(np.uint8)
            if use_max:
                dst_patch[mask] = np.maximum(dst_patch[mask], src[mask])
            else:
                dst_patch[mask] = src[mask]

        def _binary_outline(mask):
            if mask.shape[0] <= 2 or mask.shape[1] <= 2:
                return mask
            inner = np.zeros_like(mask, dtype=np.bool_)
            inner[1:-1, 1:-1] = (
                mask[1:-1, 1:-1]
                & mask[:-2, 1:-1]
                & mask[2:, 1:-1]
                & mask[1:-1, :-2]
                & mask[1:-1, 2:]
            )
            return mask & (~inner)

        for q_idx in range(num_queries):
            q_valid = valid_arr[q_idx]
            if not np.any(q_valid):
                continue
            color_i = color_arr[q_idx].astype(np.float32)

            if vis_mode == "ellipse":
                for g_idx in np.nonzero(q_valid)[0].tolist():
                    params = _bev_gaussian_params(q_idx, g_idx)
                    if params is None:
                        continue
                    cov_xx, cov_xy, cov_yy, inv_xx, inv_xy, inv_yy = params
                    rx = max(1, int(np.ceil(trunc * np.sqrt(cov_xx))))
                    ry = max(1, int(np.ceil(trunc * np.sqrt(cov_yy))))
                    cx = int(x_idx[q_idx, g_idx])
                    cy = int(y_idx[q_idx, g_idx])

                    x0 = max(0, cx - rx)
                    x1 = min(width - 1, cx + rx)
                    y0 = max(0, cy - ry)
                    y1 = min(height - 1, cy + ry)
                    if x1 < x0 or y1 < y0:
                        continue

                    xs = np.arange(x0, x1 + 1, dtype=np.float32)
                    ys = np.arange(y0, y1 + 1, dtype=np.float32)
                    xx, yy = np.meshgrid(xs, ys)
                    dx = xx - float(cx)
                    dy = yy - float(cy)
                    md2 = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy
                    norm = md2 / max(trunc * trunc, 1e-6)
                    fill_mask = norm <= 1.0
                    if not np.any(fill_mask):
                        continue

                    inner_scale = max((trunc - (float(thickness) / max(np.sqrt(max(cov_xx, cov_yy)), 1e-6))) / trunc, 1e-6)
                    inner_norm = norm / max(inner_scale * inner_scale, 1e-6)
                    outline_mask = fill_mask & (inner_norm >= 1.0)

                    patch = canvas[y0:y1 + 1, x0:x1 + 1]
                    if fill_alpha > 0.0:
                        _blend_patch(patch, fill_mask, color_i, float(fill_alpha))
                    if outline_alpha > 0.0 and np.any(outline_mask):
                        _blend_patch(patch, outline_mask, color_i, float(outline_alpha))
                continue

            rx_all = []
            ry_all = []
            cx_all = []
            cy_all = []
            for g_idx in np.nonzero(q_valid)[0].tolist():
                params = _bev_gaussian_params(q_idx, g_idx)
                if params is None:
                    continue
                cov_xx, _, cov_yy, _, _, _ = params
                rx_all.append(max(1, int(np.ceil(trunc * np.sqrt(cov_xx)))))
                ry_all.append(max(1, int(np.ceil(trunc * np.sqrt(cov_yy)))))
                cx_all.append(int(x_idx[q_idx, g_idx]))
                cy_all.append(int(y_idx[q_idx, g_idx]))
            if len(rx_all) <= 0:
                continue

            x0 = max(0, min(cx - rx for cx, rx in zip(cx_all, rx_all)))
            x1 = min(width - 1, max(cx + rx for cx, rx in zip(cx_all, rx_all)))
            y0 = max(0, min(cy - ry for cy, ry in zip(cy_all, ry_all)))
            y1 = min(height - 1, max(cy + ry for cy, ry in zip(cy_all, ry_all)))
            if x1 < x0 or y1 < y0:
                continue

            xs = np.arange(x0, x1 + 1, dtype=np.float32)
            ys = np.arange(y0, y1 + 1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)
            total_prob = np.zeros_like(xx, dtype=np.float32)

            for g_idx in np.nonzero(q_valid)[0].tolist():
                params = _bev_gaussian_params(q_idx, g_idx)
                if params is None:
                    continue
                _, _, _, inv_xx, inv_xy, inv_yy = params
                cx = float(x_idx[q_idx, g_idx])
                cy = float(y_idx[q_idx, g_idx])
                comp_w = float(weight_arr[q_idx, g_idx])
                if not np.isfinite(comp_w):
                    comp_w = 1.0
                comp_w = max(0.0, comp_w)
                if comp_w <= 0.0:
                    continue

                dx = xx - cx
                dy = yy - cy
                mahal_sq = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy
                support_mask = mahal_sq <= (trunc * trunc)
                if not np.any(support_mask):
                    continue
                comp_prob = np.exp(-0.5 * mahal_sq).astype(np.float32) * comp_w
                total_prob[support_mask] += comp_prob[support_mask]

            if not np.any(total_prob > 0.0):
                continue

            patch = canvas[y0:y1 + 1, x0:x1 + 1]
            total_prob = np.clip(total_prob, 0.0, 1.0)
            if vis_mode == "prob":
                prob_mask = total_prob > 0.0
                alpha_map = np.clip(total_prob * prob_alpha_scale * max(float(fill_alpha), 0.0), 0.0, 1.0)
                _blend_patch(patch, prob_mask, color_i, alpha_map)
                if outline_alpha > 0.0 and prob_threshold > 0.0:
                    thr_mask = total_prob >= prob_threshold
                    outline_mask = _binary_outline(thr_mask)
                    if np.any(outline_mask):
                        _blend_patch(patch, outline_mask, color_i, float(outline_alpha))
            else:
                fill_mask = total_prob >= prob_threshold
                if not np.any(fill_mask):
                    continue
                if fill_alpha > 0.0:
                    _blend_patch(patch, fill_mask, color_i, float(fill_alpha))
                if outline_alpha > 0.0:
                    outline_mask = _binary_outline(fill_mask)
                    if np.any(outline_mask):
                        _blend_patch(patch, outline_mask, color_i, float(outline_alpha))

    def _get_query_vis_palette(self):
        # Fixed raw-id -> color mapping: colors stay stable even if the
        # query_class_ids set changes (e.g. pedestrian excluded).
        fixed_palette = {
            0: [110, 110, 110],   # background query
            2: [255, 170, 40],    # bicycle
            3: [80, 180, 255],    # bus
            4: [255, 80, 80],     # car
            5: [255, 220, 80],    # construction
            6: [180, 100, 255],   # motorcycle
            7: [80, 255, 200],    # pedestrian
            9: [140, 255, 100],   # trailer
            10: [255, 120, 220],  # truck
        }
        extra_palette = [
            [255, 255, 120],
            [120, 220, 255],
            [255, 160, 120],
        ]
        color_map = {}
        extra_idx = 0
        for raw_id in self.query_class_ids:
            raw_id = int(raw_id)
            if raw_id in fixed_palette:
                color = fixed_palette[raw_id]
            else:
                color = extra_palette[extra_idx % len(extra_palette)]
                extra_idx += 1
            color_map[raw_id] = np.asarray(color, dtype=np.uint8)
        return color_map

    def _get_query_pred_palette(self):
        # Per predicted-class color list, indexed by the predicted class id.
        # Binary mode -> [background, foreground]; multi-class -> per query_class_id.
        class_palette = self._get_query_vis_palette()
        default = np.asarray([255, 255, 255], dtype=np.uint8)
        bg_color = class_palette.get(
            int(self.query_class_ids[0]), np.asarray([110, 110, 110], dtype=np.uint8)
        )
        if self.query_binary_cls:
            fg_color = np.asarray([255, 255, 0], dtype=np.uint8)
            return [bg_color, fg_color]
        return [class_palette.get(int(rid), default) for rid in self.query_class_ids]

    def _draw_query_vis_legend(self, draw, x_start: int, y_start: int, canvas_w: int) -> int:
        pred_palette = self._get_query_pred_palette()
        default = np.asarray([255, 255, 255], dtype=np.uint8)
        items = []
        for idx, name in enumerate(self.query_class_names):
            label = "background query" if idx == 0 else str(name)
            color = pred_palette[idx] if idx < len(pred_palette) else default
            items.append((color, label))
        if len(items) <= 0:
            return 0

        x = int(x_start)
        y = int(y_start)
        line_h = 16
        swatch = 10
        gap = 8
        for color, name in items:
            est_w = swatch + 4 + max(42, int(7 * len(name))) + gap
            if x + est_w > int(canvas_w) and x > int(x_start):
                x = int(x_start)
                y += line_h + 6
            color = tuple(int(v) for v in color)
            draw.rectangle([x, y + 3, x + swatch, y + 3 + swatch], fill=color)
            draw.text((x + swatch + 4, y), name, fill=(255, 255, 255))
            x += est_w
        return (y - int(y_start)) + line_h

    def _draw_generic_vis_legend(
        self,
        draw,
        x_start: int,
        y_start: int,
        canvas_w: int,
        conf_thr: float,
        matched_color=(255, 80, 255),
        ego_color=(255, 255, 0),
        gaussian_label: str = "Gaussian footprint (BEV xy)",
    ) -> int:
        items = [
            ((240, 240, 240), "GT occupied BEV"),
            ((0, 0, 0), "empty / non-query GT"),
            ((40, 180, 40), "GT overlay"),
            (ego_color, "ego center (+ marker)"),
            ((140, 140, 140), gaussian_label),
            ((30, 255, 255), f"query conf>={conf_thr:.2f}"),
            ((255, 60, 60), f"query conf<{conf_thr:.2f}"),
            (matched_color, "Hungarian-matched query"),
        ]
        x = int(x_start)
        y = int(y_start)
        line_h = 16
        swatch = 10
        gap = 10
        for color, name in items:
            est_w = swatch + 4 + max(72, int(7 * len(name))) + gap
            if x + est_w > int(canvas_w) and x > int(x_start):
                x = int(x_start)
                y += line_h + 6
            draw.rectangle([x, y + 3, x + swatch, y + 3 + swatch], fill=color)
            draw.text((x + swatch + 4, y), name, fill=(255, 255, 255))
            x += est_w
        return (y - int(y_start)) + line_h

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
        (
            query_cls_logits_qc,
            query_cls_scores_qc,
            query_cls_scores_tqc,
        ) = self._compute_query_cls_outputs(
            query_inst_tqd=query_inst,
            query_future_feat_tqd=query_feat_tqd,
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
        mixture_quat_tqg4 = None
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
                mixture_quat_tqg4,
                mixture_weights_tqg,
            ) = self._compute_gaussian_outputs(center_input_tqd, centers_world)
            (
                query_cls_logits_qc,
                query_cls_scores_qc,
                query_cls_scores_tqc,
            ) = self._compute_query_cls_outputs(
                query_inst_tqd=query_inst,
                query_future_feat_tqd=query_feat_tqd,
                centers_world_tq3=centers_world,
                mixture_centers_world_tqg3=mixture_centers_world_tqg3,
                mixture_sigmas_world_tqg3=mixture_sigmas_world_tqg3,
                mixture_quat_tqg4=mixture_quat_tqg4,
                mixture_weights_tqg=mixture_weights_tqg,
            )
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
            "mixture_quat_tqg4": mixture_quat_tqg4,
            "mixture_weights_tqg": mixture_weights_tqg,
            "query_cls_logits_qc": query_cls_logits_qc,
            "query_cls_scores_qc": query_cls_scores_qc,
            "query_cls_scores_tqc": query_cls_scores_tqc,
            "query_depth_logits_tqd": query_depth_logits_tqd,
            "query_depth_probs_tqd": query_depth_probs_tqd,
            "query_feat_tqd": query_feat_tqd,
            "query_inst_tqd": query_inst,
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
            "query_size_dbg": self.last_query_size_dbg,
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
            "dbg_query_sigma_reg_raw": z,
            "dbg_query_sigma_mean_m": z,
            "dbg_query_sigma_log_abs_mean": z,
            "dbg_query_sigma_reg_weight": z.new_tensor(float(self.query_multi_gaussian_sigma_reg_loss_weight)),
            "dbg_query_sigma_reg_enabled": z.new_tensor(
                float(1.0 if self.query_multi_gaussian_sigma_reg_loss_weight > 0.0 else 0.0)
            ),
            "dbg_query_weight_sum_mean": z,
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
        return out

    def _is_main_process(self) -> bool:
        if not dist.is_available():
            return True
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _eval_vis_enabled_this_rank(self) -> bool:
        return self._is_main_process() or bool(getattr(self, "_eval_vis_all_ranks", False))

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

    @staticmethod
    def _normalize_gt_occ_semantic_for_vis(gt_occ_semantic):
        if gt_occ_semantic is None:
            return None

        seg = gt_occ_semantic
        if isinstance(seg, (list, tuple)):
            if len(seg) <= 0:
                return None
            if len(seg) == 1:
                seg = seg[0]
            else:
                try:
                    seg = torch.stack(seg, dim=0)
                except Exception:
                    return None

        if not torch.is_tensor(seg):
            return None

        if seg.dim() == 6:
            if seg.shape[0] != 1:
                return None
            if seg.shape[2] in (1, 2):
                seg = seg[0, :, 0]
            else:
                return None
        elif seg.dim() == 5:
            if seg.shape[1] in (1, 2):
                seg = seg[:, 0]
            elif seg.shape[0] == 1:
                seg = seg[0]
            else:
                return None
        elif seg.dim() != 4:
            return None

        return seg.to(torch.long).contiguous()

    @staticmethod
    def _normalize_gt_occ_inst_for_vis(gt_occ_inst):
        """
        Normalize gt_occ_inst-like dense inputs for visualization.

        Supported layouts:
        - [T, 2, X, Y, Z] or [1, T, 2, X, Y, Z]
        - [T, X, Y, Z] (instance-id only)
        Returns:
            cls_txyz:  [T, X, Y, Z] long or None
            inst_txyz: [T, X, Y, Z] long or None
        """
        if gt_occ_inst is None:
            return None, None

        seg = gt_occ_inst
        if isinstance(seg, (list, tuple)):
            if len(seg) <= 0:
                return None, None
            if len(seg) == 1:
                seg = seg[0]
            else:
                try:
                    seg = torch.stack(seg, dim=0)
                except Exception:
                    return None, None

        if not torch.is_tensor(seg):
            return None, None

        # [1,T,2,X,Y,Z] or [1,T,1,X,Y,Z]
        if seg.dim() == 6:
            if seg.shape[0] != 1:
                return None, None
            if seg.shape[2] == 2:
                seg = seg[0]         # [T,2,X,Y,Z]
            elif seg.shape[2] == 1:
                seg = seg[0, :, 0]   # [T,X,Y,Z]
            else:
                return None, None
        elif seg.dim() == 5:
            # [T,2,X,Y,Z]
            if seg.shape[1] == 2:
                pass
            # [1,T,2,X,Y,Z] squeezed partially by caller
            elif seg.shape[0] == 1:
                seg = seg[0]
            # [T,1,X,Y,Z] -> [T,X,Y,Z]
            elif seg.shape[1] == 1:
                seg = seg[:, 0]
            else:
                return None, None
        elif seg.dim() != 4:
            return None, None

        cls_t = None
        inst_t = None
        if seg.dim() == 5 and seg.shape[1] == 2:
            cls_t = seg[:, 0]
            inst_t = seg[:, 1]
        elif seg.dim() == 4:
            inst_t = seg
        else:
            return None, None

        if not torch.is_tensor(inst_t):
            return None, None
        inst_t = inst_t.to(torch.long).contiguous()
        if torch.is_tensor(cls_t):
            cls_t = cls_t.to(torch.long).contiguous()
        return cls_t, inst_t

    @torch.no_grad()
    def _maybe_save_prob_grid_vis(
        self,
        pred_occ_prob: torch.Tensor,
        points_world: torch.Tensor,
        gt_occ_gmo: torch.Tensor,
        step: int,
        gt_occ_semantic: torch.Tensor = None,
        max_frames: int = 6,
        voxel_center_offset: float = 0.5,
        prob_threshold: float = 0.5,
        pred_layout: str = "auto",
        pred_occ_prob_pos_obj: torch.Tensor = None,
        pred_occ_prob_all_obj: torch.Tensor = None,
        points_world_all: torch.Tensor = None,
        point_conf_all: torch.Tensor = None,
        point_class_ids_all: torch.Tensor = None,
        point_weights_all: torch.Tensor = None,
        query_vis_bundle: dict = None,
    ) -> None:
        """
        Query-center-focused BEV visualization.
        - row1: GT semantic BEV (query classes, class-colored)
        - row2: all query centers (class-agnostic; high/low confidence contrasted)
        - row3: high-confidence query centers only (class-agnostic)
        - row4: high-confidence query centers only (class-colored)
        - row5: Hungarian-matched query centers only
        - row6: trajectory overlay
        """
        vis_every = int(getattr(self, "debug_vis_every", 0))
        if vis_every <= 0:
            return
        if (step % vis_every) != 0:
            return
        if not self._eval_vis_enabled_this_rank():
            return
        if pred_occ_prob is None or points_world is None:
            return
        if (self.point_cloud_range is None) or (self.spatial_extent3d is None):
            return

        try:
            import os
            from PIL import Image, ImageDraw
        except Exception:
            return

        if points_world.dim() == 4:
            pts = points_world
        elif points_world.dim() == 3:
            pts = points_world.unsqueeze(2)
        elif points_world.dim() == 5:
            pts = points_world[0]
        else:
            return

        if pred_occ_prob.dim() == 5 and pred_occ_prob.shape[1] == 1:
            pred_base = pred_occ_prob[:, 0]
        elif pred_occ_prob.dim() == 4:
            pred_base = pred_occ_prob
        else:
            return
        pred = pred_base

        if gt_occ_gmo.dim() != 4:
            return

        # gt_occ_gmo follows [T, Z, X, Y] in this branch.
        T_gt, Z, X, Y = gt_occ_gmo.shape
        T_pred = pts.shape[0]
        T_prob = pred.shape[0]
        if pred.shape[1] != Z:
            return

        shape_is_zyx = (pred.shape[2] == Y and pred.shape[3] == X)
        shape_is_zxy = (pred.shape[2] == X and pred.shape[3] == Y)
        if not (shape_is_zyx or shape_is_zxy):
            return

        layout = str(pred_layout).lower()
        if layout == "auto":
            if shape_is_zxy and not shape_is_zyx:
                layout = "zxy"
            elif shape_is_zyx and not shape_is_zxy:
                layout = "zyx"
            else:
                # Ambiguous case (e.g. X==Y). Canonical query loss path is [T,Z,X,Y].
                layout = "zxy"
        elif layout not in ("zyx", "zxy"):
            return

        T = int(min(T_gt, T_pred, T_prob, max_frames))
        if T <= 0:
            return

        pts = pts[:T].to(torch.float32)
        gt = gt_occ_gmo[:T]

        pc_min = pts.new_tensor(self.point_cloud_range[:3])
        extent = pts.new_tensor(self.spatial_extent3d)
        grid_size = pts.new_tensor([float(X), float(Y), float(Z)])
        voxel_size = extent / grid_size

        off = float(voxel_center_offset)
        ix = (pts[..., 0] - pc_min[0]) / voxel_size[0] - off
        iy = (pts[..., 1] - pc_min[1]) / voxel_size[1] - off
        iz = (pts[..., 2] - pc_min[2]) / voxel_size[2] - off
        valid = (ix >= 0.0) & (ix <= float(X - 1)) & (iy >= 0.0) & (iy <= float(Y - 1)) & (iz >= 0.0) & (iz <= float(Z - 1))

        def _normalize_points(points_tq3):
            if not torch.is_tensor(points_tq3):
                return None
            if points_tq3.dim() == 4:
                p = points_tq3
            elif points_tq3.dim() == 3:
                p = points_tq3.unsqueeze(2)
            elif points_tq3.dim() == 5:
                p = points_tq3[0]
            else:
                return None
            if int(p.shape[0]) <= 0:
                return None
            return p[:T].to(torch.float32)

        def _expand_attr_for_points(attr, points_t):
            if not torch.is_tensor(attr) or points_t is None:
                return None
            if attr.dim() == 1:
                x = attr[None, :].expand(T, -1)
            else:
                x = attr[:T]
            if x.dim() == 2 and points_t.dim() == 4 and points_t.shape[2] != 1:
                x = x[:, :, None].expand(-1, -1, points_t.shape[2])
            elif x.dim() == 2 and points_t.dim() == 4 and points_t.shape[2] == 1:
                x = x[:, :, None]
            elif x.dim() != points_t.dim() - 1:
                return None
            return x

        def _normalize_sigmas(sigmas_tq3, points_t):
            if (not torch.is_tensor(sigmas_tq3)) or points_t is None:
                return None
            if sigmas_tq3.dim() == 4:
                sig = sigmas_tq3
            elif sigmas_tq3.dim() == 3 and points_t.dim() == 4:
                sig = sigmas_tq3.unsqueeze(2)
            elif sigmas_tq3.dim() == 5:
                sig = sigmas_tq3[0]
            else:
                return None
            sig = sig[:T].to(torch.float32)
            if tuple(sig.shape) != tuple(points_t.shape):
                return None
            return sig

        def _normalize_scalar_attr(attr_tqg, points_t):
            if (not torch.is_tensor(attr_tqg)) or points_t is None:
                return None
            if attr_tqg.dim() == 3:
                attr = attr_tqg
            elif attr_tqg.dim() == 2 and points_t.dim() == 4 and int(points_t.shape[2]) == 1:
                attr = attr_tqg.unsqueeze(2)
            elif attr_tqg.dim() == 4 and attr_tqg.shape[0] == 1:
                attr = attr_tqg[0]
            else:
                return None
            attr = attr[:T].to(torch.float32)
            if tuple(attr.shape) != tuple(points_t.shape[:-1]):
                return None
            return attr

        def _normalize_vector_attr(attr_tqgd, points_t, dim):
            if (not torch.is_tensor(attr_tqgd)) or points_t is None:
                return None
            if attr_tqgd.dim() == 4:
                attr = attr_tqgd
            elif attr_tqgd.dim() == 3 and points_t.dim() == 4 and int(points_t.shape[2]) == 1:
                attr = attr_tqgd.unsqueeze(2)
            elif attr_tqgd.dim() == 5 and int(attr_tqgd.shape[0]) == 1:
                attr = attr_tqgd[0]
            else:
                return None
            attr = attr[:T].to(torch.float32)
            if tuple(attr.shape) != tuple(points_t.shape[:-1]) + (int(dim),):
                return None
            return attr

        def _project_points(points_t, scores_t=None, class_ids_t=None):
            if points_t is None:
                return None, None, None, None, None
            points_t = points_t.to(device=pc_min.device, dtype=torch.float32)
            ix_t = (points_t[..., 0] - pc_min[0]) / voxel_size[0] - off
            iy_t = (points_t[..., 1] - pc_min[1]) / voxel_size[1] - off
            iz_t = (points_t[..., 2] - pc_min[2]) / voxel_size[2] - off
            valid_t = (
                (ix_t >= 0.0) & (ix_t <= float(X - 1)) &
                (iy_t >= 0.0) & (iy_t <= float(Y - 1)) &
                (iz_t >= 0.0) & (iz_t <= float(Z - 1))
            )
            ix_i = torch.round(ix_t).to(torch.long).cpu().numpy()
            iy_i = torch.round(iy_t).to(torch.long).cpu().numpy()
            valid_np = valid_t.cpu().numpy()
            scores_np = None
            class_np = None
            if torch.is_tensor(scores_t):
                scores_np = scores_t.to(device=pc_min.device, dtype=torch.float32).clamp(0.0, 1.0).cpu().numpy()
            if torch.is_tensor(class_ids_t):
                class_np = class_ids_t.to(device=pc_min.device, dtype=torch.long).cpu().numpy()
            return ix_i, iy_i, valid_np, scores_np, class_np

        def _project_gaussians(points_t, sigmas_t, quat_t=None, weight_t=None):
            if points_t is None or sigmas_t is None:
                return None, None, None, None, None, None, None, None
            points_t = points_t.to(device=pc_min.device, dtype=torch.float32)
            sigmas_t = sigmas_t.to(device=pc_min.device, dtype=torch.float32)
            ix_t = (points_t[..., 0] - pc_min[0]) / voxel_size[0] - off
            iy_t = (points_t[..., 1] - pc_min[1]) / voxel_size[1] - off
            sigma_x_t = (sigmas_t[..., 0] / voxel_size[0]).to(torch.float32)
            sigma_y_t = (sigmas_t[..., 1] / voxel_size[1]).to(torch.float32)
            sigma_z_t = (sigmas_t[..., 2] / voxel_size[2]).to(torch.float32)
            valid_t = (
                (ix_t >= 0.0) & (ix_t <= float(X - 1)) &
                (iy_t >= 0.0) & (iy_t <= float(Y - 1)) &
                torch.isfinite(sigma_x_t) & torch.isfinite(sigma_y_t) & torch.isfinite(sigma_z_t) &
                (sigma_x_t > 0.0) & (sigma_y_t > 0.0) & (sigma_z_t > 0.0)
            )
            ix_i = torch.round(ix_t).to(torch.long).cpu().numpy()
            iy_i = torch.round(iy_t).to(torch.long).cpu().numpy()
            sx_np = sigma_x_t.cpu().numpy()
            sy_np = sigma_y_t.cpu().numpy()
            sz_np = sigma_z_t.cpu().numpy()
            quat_np = None
            weight_np = None
            if torch.is_tensor(quat_t):
                quat_np = quat_t[:T].to(device=pc_min.device, dtype=torch.float32).cpu().numpy()
            if torch.is_tensor(weight_t):
                weight_np = weight_t[:T].to(device=pc_min.device, dtype=torch.float32).cpu().numpy()
            valid_np = valid_t.cpu().numpy()
            return ix_i, iy_i, sx_np, sy_np, sz_np, quat_np, weight_np, valid_np

        use_score_bundle = isinstance(query_vis_bundle, dict)
        bundle_top_k = 0
        bundle_score_thr = float(getattr(self, "debug_query_confidence_vis_threshold", 0.5))
        bundle_w_iou = 0.5
        bundle_w_cls = 0.5
        bundle_w_cam = 0.0

        if use_score_bundle:
            pts_candidate = _normalize_points(query_vis_bundle.get("candidate_points_tq3", None))
            pts_selected = _normalize_points(query_vis_bundle.get("selected_points_tq3", None))
            pts_matched = _normalize_points(query_vis_bundle.get("matched_points_tq3", None))
            # Draw Gaussian footprints using full mixture components when available.
            gpts_candidate = _normalize_points(query_vis_bundle.get("candidate_mixture_centers_tqg3", None))
            gpts_selected = _normalize_points(query_vis_bundle.get("selected_mixture_centers_tqg3", None))
            gpts_matched = _normalize_points(query_vis_bundle.get("matched_mixture_centers_tqg3", None))
            sig_candidate = _normalize_sigmas(query_vis_bundle.get("candidate_mixture_sigmas_tqg3", None), gpts_candidate)
            sig_selected = _normalize_sigmas(query_vis_bundle.get("selected_mixture_sigmas_tqg3", None), gpts_selected)
            sig_matched = _normalize_sigmas(query_vis_bundle.get("matched_mixture_sigmas_tqg3", None), gpts_matched)
            quat_candidate = _normalize_vector_attr(query_vis_bundle.get("candidate_mixture_quat_tqg4", None), gpts_candidate, dim=4)
            quat_selected = _normalize_vector_attr(query_vis_bundle.get("selected_mixture_quat_tqg4", None), gpts_selected, dim=4)
            quat_matched = _normalize_vector_attr(query_vis_bundle.get("matched_mixture_quat_tqg4", None), gpts_matched, dim=4)
            w_candidate = _normalize_scalar_attr(query_vis_bundle.get("candidate_mixture_weights_tqg", None), gpts_candidate)
            w_selected = _normalize_scalar_attr(query_vis_bundle.get("selected_mixture_weights_tqg", None), gpts_selected)
            w_matched = _normalize_scalar_attr(query_vis_bundle.get("matched_mixture_weights_tqg", None), gpts_matched)
            if gpts_candidate is None:
                gpts_candidate = pts_candidate
                sig_candidate = _normalize_sigmas(query_vis_bundle.get("candidate_sigmas_tq3", None), gpts_candidate)
                quat_candidate = None
                w_candidate = None
            if gpts_selected is None:
                gpts_selected = pts_selected
                sig_selected = _normalize_sigmas(query_vis_bundle.get("selected_sigmas_tq3", None), gpts_selected)
                quat_selected = None
                w_selected = None
            if gpts_matched is None:
                gpts_matched = pts_matched
                sig_matched = _normalize_sigmas(query_vis_bundle.get("matched_sigmas_tq3", None), gpts_matched)
                quat_matched = None
                w_matched = None
            sc_candidate = _expand_attr_for_points(query_vis_bundle.get("candidate_score_q", None), pts_candidate)
            sc_selected = _expand_attr_for_points(query_vis_bundle.get("selected_score_q", None), pts_selected)
            sc_matched = _expand_attr_for_points(query_vis_bundle.get("matched_score_q", None), pts_matched)
            cls_candidate = _expand_attr_for_points(query_vis_bundle.get("candidate_pred_cls_q", None), pts_candidate)
            cls_selected = _expand_attr_for_points(query_vis_bundle.get("selected_pred_cls_q", None), pts_selected)
            cls_matched = _expand_attr_for_points(query_vis_bundle.get("matched_pred_cls_q", None), pts_matched)
            bundle_top_k = int(query_vis_bundle.get("top_k", 0))
            bundle_score_thr = float(query_vis_bundle.get("score_thr", bundle_score_thr))
            bundle_w_iou = float(query_vis_bundle.get("w_iou", bundle_w_iou))
            bundle_w_cls = float(query_vis_bundle.get("w_cls", bundle_w_cls))
            bundle_w_cam = float(query_vis_bundle.get("w_cam", bundle_w_cam))
        else:
            pts_candidate = _normalize_points(points_world_all if points_world_all is not None else pts)
            if point_conf_all is None:
                point_conf_all = point_weights_all
            sc_candidate = _expand_attr_for_points(point_conf_all, pts_candidate)
            cls_candidate = _expand_attr_for_points(point_class_ids_all, pts_candidate)
            pts_selected = pts_candidate
            sc_selected = sc_candidate
            cls_selected = cls_candidate
            pts_matched = None
            gpts_candidate = None
            gpts_selected = None
            gpts_matched = None
            sig_candidate = None
            sig_selected = None
            sig_matched = None
            quat_candidate = None
            quat_selected = None
            quat_matched = None
            w_candidate = None
            w_selected = None
            w_matched = None
            sc_matched = None
            cls_matched = None

        cand_x_i, cand_y_i, cand_valid_np, cand_score_np, cand_cls_np = _project_points(
            pts_candidate, sc_candidate, cls_candidate
        )
        sel_x_i, sel_y_i, sel_valid_np, sel_score_np, sel_cls_np = _project_points(
            pts_selected, sc_selected, cls_selected
        )
        mat_x_i, mat_y_i, mat_valid_np, mat_score_np, mat_cls_np = _project_points(
            pts_matched, sc_matched, cls_matched
        )
        cand_gx_i, cand_gy_i, cand_sx_np, cand_sy_np, cand_sz_np, cand_quat_np, cand_w_np, cand_gvalid_np = _project_gaussians(
            gpts_candidate, sig_candidate, quat_candidate, w_candidate
        )
        sel_gx_i, sel_gy_i, sel_sx_np, sel_sy_np, sel_sz_np, sel_quat_np, sel_w_np, sel_gvalid_np = _project_gaussians(
            gpts_selected, sig_selected, quat_selected, w_selected
        )
        mat_gx_i, mat_gy_i, mat_sx_np, mat_sy_np, mat_sz_np, mat_quat_np, mat_w_np, mat_gvalid_np = _project_gaussians(
            gpts_matched, sig_matched, quat_matched, w_matched
        )
        bundle_has_gaussian = any(
            x is not None for x in (cand_sx_np, sel_sx_np, mat_sx_np)
        )

        gt_np = gt.cpu().numpy()
        gt_sem_np = None
        if torch.is_tensor(gt_occ_semantic):
            gt_sem = gt_occ_semantic
            if gt_sem.dim() == 6 and gt_sem.shape[0] == 1 and gt_sem.shape[2] == 1:
                gt_sem = gt_sem[0, :, 0]
            elif gt_sem.dim() == 5 and gt_sem.shape[1] == 1:
                gt_sem = gt_sem[:, 0]
            elif gt_sem.dim() == 4:
                pass
            else:
                gt_sem = None
            if gt_sem is not None:
                gt_sem = gt_sem[:T].to(torch.long)
                if gt_sem.shape[-1] == Z and gt_sem.shape[1] == X and gt_sem.shape[2] == Y:
                    gt_sem_np = gt_sem.cpu().numpy()  # [T, X, Y, Z]
                elif gt_sem.shape[1] == Z and gt_sem.shape[2] == X and gt_sem.shape[3] == Y:
                    gt_sem_np = gt_sem.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [T, X, Y, Z]

        row_gt = []
        row_all = []
        row_hi_cls = []
        row_matched = []
        row_gt_cls = []
        row_traj = []
        row_base_traj = []
        stats_valid = []
        stats_hi = []
        stats_lo = []
        stats_matched = []

        conf_thr = float(os.environ.get(
            "EOCF_EVAL_FG_THR",
            getattr(self, "debug_query_score_threshold", getattr(self, "debug_query_confidence_vis_threshold", 0.5)),
        ))
        marker_radius = int(getattr(self, "debug_query_center_marker_radius", 3))
        gt_color = np.array([40, 180, 40], dtype=np.uint8)
        hi_color = np.array([30, 255, 255], dtype=np.uint8)
        lo_color = np.array([255, 60, 60], dtype=np.uint8)
        matched_color = np.array([255, 80, 255], dtype=np.uint8)
        traj_pred_color = np.array([255, 48, 48], dtype=np.uint8)
        traj_gt_color = np.array([80, 255, 255], dtype=np.uint8)
        ego_color = np.array([255, 255, 0], dtype=np.uint8)
        ego_outline_color = np.array([0, 0, 0], dtype=np.uint8)
        ego_cross_arm = max(2, marker_radius + 3)
        ego_cross_thickness = 1
        vx = float(voxel_size[0].item())
        vy = float(voxel_size[1].item())
        ego_ix_f = (0.0 - float(pc_min[0].item())) / max(vx, 1e-6) - off
        ego_iy_f = (0.0 - float(pc_min[1].item())) / max(vy, 1e-6) - off
        ego_visible = (
            (ego_ix_f >= 0.0) and (ego_ix_f <= float(X - 1))
            and (ego_iy_f >= 0.0) and (ego_iy_f <= float(Y - 1))
        )
        ego_ix = int(np.round(ego_ix_f)) if ego_visible else -1
        ego_iy = int(np.round(ego_iy_f)) if ego_visible else -1
        class_palette = self._get_query_vis_palette()
        pred_cls_palette = self._get_query_pred_palette()
        gaussian_vis_mode = str(getattr(self, "debug_query_gaussian_vis_mode", "ellipse")).lower()
        gaussian_prob_threshold = float(prob_threshold)
        gaussian_prob_alpha_scale = float(getattr(self, "debug_query_gaussian_prob_alpha_scale", 4.0))
        traj_points_tq3 = None
        base_traj_points_tq3 = None
        matched_gt_traj_tn3 = None
        matched_gt_valid_tn = None
        all_gt_traj_tn3 = None
        all_gt_valid_tn = None
        if isinstance(query_vis_bundle, dict):
            if torch.is_tensor(query_vis_bundle.get("matched_points_tq3", None)):
                traj_points_tq3 = query_vis_bundle["matched_points_tq3"]
            if torch.is_tensor(query_vis_bundle.get("base_matched_points_tq3", None)):
                base_traj_points_tq3 = query_vis_bundle["base_matched_points_tq3"]
            if torch.is_tensor(query_vis_bundle.get("matched_gt_traj_tn3", None)):
                matched_gt_traj_tn3 = query_vis_bundle["matched_gt_traj_tn3"]
            if torch.is_tensor(query_vis_bundle.get("matched_gt_valid_tn", None)):
                matched_gt_valid_tn = query_vis_bundle["matched_gt_valid_tn"]
            if torch.is_tensor(query_vis_bundle.get("all_gt_traj_tn3", None)):
                all_gt_traj_tn3 = query_vis_bundle["all_gt_traj_tn3"]
            if torch.is_tensor(query_vis_bundle.get("all_gt_valid_tn", None)):
                all_gt_valid_tn = query_vis_bundle["all_gt_valid_tn"]
        traj_x_i, traj_y_i, traj_valid_np, _, _ = _project_points(traj_points_tq3)
        base_traj_x_i, base_traj_y_i, base_traj_valid_np, _, _ = _project_points(base_traj_points_tq3)
        gt_traj_points_tn3 = all_gt_traj_tn3 if torch.is_tensor(all_gt_traj_tn3) else matched_gt_traj_tn3
        gt_traj_valid_tn = all_gt_valid_tn if torch.is_tensor(all_gt_valid_tn) else matched_gt_valid_tn
        gt_traj_x_i, gt_traj_y_i, gt_traj_valid_np, _, _ = _project_points(gt_traj_points_tn3)
        if torch.is_tensor(gt_traj_valid_tn) and gt_traj_valid_np is not None:
            gt_traj_valid_np = gt_traj_valid_np & gt_traj_valid_tn.to(torch.bool).cpu().numpy()

        def _draw_traj_arrow(draw_ctx, x0, y0, x1, y1, color, outline=(0, 0, 0)):
            dx = float(x1 - x0)
            dy = float(y1 - y0)
            norm = float(np.hypot(dx, dy))
            if norm < 1e-6:
                r = 3
                draw_ctx.ellipse((x1 - r - 1, y1 - r - 1, x1 + r + 1, y1 + r + 1), fill=outline)
                draw_ctx.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=color)
                return
            ux = dx / norm
            uy = dy / norm
            head_len = max(8.0, min(16.0, norm * 0.45))
            head_ang = 0.55
            c = float(np.cos(head_ang))
            s = float(np.sin(head_ang))
            lx = x1 - head_len * (ux * c - uy * s)
            ly = y1 - head_len * (uy * c + ux * s)
            rx = x1 - head_len * (ux * c + uy * s)
            ry = y1 - head_len * (uy * c - ux * s)
            draw_ctx.line((x0, y0, x1, y1), fill=outline, width=4)
            draw_ctx.line((x1, y1, lx, ly), fill=outline, width=5)
            draw_ctx.line((x1, y1, rx, ry), fill=outline, width=5)
            draw_ctx.line((x0, y0, x1, y1), fill=color, width=2)
            draw_ctx.line((x1, y1, lx, ly), fill=color, width=3)
            draw_ctx.line((x1, y1, rx, ry), fill=color, width=3)

        for t in range(T):
            gt_bev = (gt_np[t] == 1).any(axis=0).T.astype(np.bool_)
            gt_rgb = np.zeros((Y, X, 3), dtype=np.uint8)
            gt_rgb[gt_bev] = np.array([240, 240, 240], dtype=np.uint8)
            gt_cls_rgb = np.zeros((Y, X, 3), dtype=np.uint8)
            if gt_sem_np is not None:
                gt_sem_t = gt_sem_np[t]  # [X, Y, Z]
                class_count_xyk = []
                class_raw_ids = []
                for raw_id in self.query_class_ids:
                    raw_id = int(raw_id)
                    if raw_id <= 0:
                        continue
                    class_raw_ids.append(raw_id)
                    class_count_xyk.append((gt_sem_t == raw_id).sum(axis=-1))
                if len(class_count_xyk) > 0:
                    class_count_xyk = np.stack(class_count_xyk, axis=-1)  # [X, Y, K]
                    dom_idx_xy = np.argmax(class_count_xyk, axis=-1)
                    dom_count_xy = np.max(class_count_xyk, axis=-1)
                    gt_cls_xy = np.zeros((X, Y, 3), dtype=np.uint8)
                    for k, raw_id in enumerate(class_raw_ids):
                        mask_xy = (dom_count_xy > 0) & (dom_idx_xy == k)
                        if np.any(mask_xy):
                            gt_cls_xy[mask_xy] = class_palette.get(
                                int(raw_id), np.asarray([255, 255, 255], dtype=np.uint8)
                            )
                    gt_cls_rgb = np.transpose(gt_cls_xy, (1, 0, 2)).copy()

            all_ov = np.zeros((Y, X, 3), dtype=np.uint8)
            all_ov[gt_bev] = gt_color
            hi_ov = np.zeros((Y, X, 3), dtype=np.uint8)
            hi_ov[gt_bev] = gt_color
            hi_cls_ov = np.zeros((Y, X, 3), dtype=np.uint8)
            hi_cls_ov[gt_bev] = gt_color
            matched_ov = np.zeros((Y, X, 3), dtype=np.uint8)
            matched_ov[gt_bev] = gt_color
            traj_ov = np.zeros((Y, X, 3), dtype=np.uint8)
            traj_ov[gt_bev] = gt_color

            valid_count = 0
            hi_count = 0
            lo_count = 0
            matched_count = 0
            if use_score_bundle:
                if (cand_gx_i is not None) and (cand_gvalid_np is not None):
                    if cand_gvalid_np[t].any():
                        self._draw_gaussian_bev_footprints(
                            all_ov,
                            cand_gx_i[t],
                            cand_gy_i[t],
                            cand_sx_np[t],
                            cand_sy_np[t],
                            lo_color,
                            sigma_z_px=None if cand_sz_np is None else cand_sz_np[t],
                            quat_wxyz=None if cand_quat_np is None else cand_quat_np[t],
                            component_weight=None if cand_w_np is None else cand_w_np[t],
                            valid_mask=cand_gvalid_np[t],
                            vis_mode=gaussian_vis_mode,
                            truncate_sigma=self.gaussian_truncate_sigma,
                            fill_alpha=0.10,
                            outline_alpha=0.30,
                            outline_thickness_px=1,
                            prob_threshold=gaussian_prob_threshold,
                            prob_alpha_scale=gaussian_prob_alpha_scale,
                            use_max=True,
                        )
                if (sel_gx_i is not None) and (sel_gvalid_np is not None):
                    if sel_gvalid_np[t].any():
                        self._draw_gaussian_bev_footprints(
                            all_ov,
                            sel_gx_i[t],
                            sel_gy_i[t],
                            sel_sx_np[t],
                            sel_sy_np[t],
                            hi_color,
                            sigma_z_px=None if sel_sz_np is None else sel_sz_np[t],
                            quat_wxyz=None if sel_quat_np is None else sel_quat_np[t],
                            component_weight=None if sel_w_np is None else sel_w_np[t],
                            valid_mask=sel_gvalid_np[t],
                            vis_mode=gaussian_vis_mode,
                            truncate_sigma=self.gaussian_truncate_sigma,
                            fill_alpha=0.14,
                            outline_alpha=0.55,
                            outline_thickness_px=1,
                            prob_threshold=gaussian_prob_threshold,
                            prob_alpha_scale=gaussian_prob_alpha_scale,
                            use_max=True,
                        )
                        self._draw_gaussian_bev_footprints(
                            hi_ov,
                            sel_gx_i[t],
                            sel_gy_i[t],
                            sel_sx_np[t],
                            sel_sy_np[t],
                            hi_color,
                            sigma_z_px=None if sel_sz_np is None else sel_sz_np[t],
                            quat_wxyz=None if sel_quat_np is None else sel_quat_np[t],
                            component_weight=None if sel_w_np is None else sel_w_np[t],
                            valid_mask=sel_gvalid_np[t],
                            vis_mode=gaussian_vis_mode,
                            truncate_sigma=self.gaussian_truncate_sigma,
                            fill_alpha=0.14,
                            outline_alpha=0.55,
                            outline_thickness_px=1,
                            prob_threshold=gaussian_prob_threshold,
                            prob_alpha_scale=gaussian_prob_alpha_scale,
                            use_max=True,
                        )
                        if sel_cls_np is not None:
                            raw_cls_hi = sel_cls_np[t].reshape(-1)
                            hi_gauss_colors = np.stack(
                                [
                                    pred_cls_palette[int(cls_id)]
                                    if 0 <= int(cls_id) < len(pred_cls_palette)
                                    else hi_color
                                    for cls_id in raw_cls_hi.tolist()
                                ],
                                axis=0,
                            ).astype(np.uint8)
                        else:
                            hi_gauss_colors = np.repeat(hi_color[None, :], repeats=int(sel_gx_i[t].shape[0]), axis=0)
                        self._draw_gaussian_bev_footprints(
                            hi_cls_ov,
                            sel_gx_i[t],
                            sel_gy_i[t],
                            sel_sx_np[t],
                            sel_sy_np[t],
                            hi_gauss_colors,
                            sigma_z_px=None if sel_sz_np is None else sel_sz_np[t],
                            quat_wxyz=None if sel_quat_np is None else sel_quat_np[t],
                            component_weight=None if sel_w_np is None else sel_w_np[t],
                            valid_mask=sel_gvalid_np[t],
                            vis_mode=gaussian_vis_mode,
                            truncate_sigma=self.gaussian_truncate_sigma,
                            fill_alpha=0.16,
                            outline_alpha=0.60,
                            outline_thickness_px=1,
                            prob_threshold=gaussian_prob_threshold,
                            prob_alpha_scale=gaussian_prob_alpha_scale,
                            use_max=True,
                        )
                if (mat_gx_i is not None) and (mat_gvalid_np is not None):
                    if mat_gvalid_np[t].any():
                        self._draw_gaussian_bev_footprints(
                            matched_ov,
                            mat_gx_i[t],
                            mat_gy_i[t],
                            mat_sx_np[t],
                            mat_sy_np[t],
                            matched_color,
                            sigma_z_px=None if mat_sz_np is None else mat_sz_np[t],
                            quat_wxyz=None if mat_quat_np is None else mat_quat_np[t],
                            component_weight=None if mat_w_np is None else mat_w_np[t],
                            valid_mask=mat_gvalid_np[t],
                            vis_mode=gaussian_vis_mode,
                            truncate_sigma=self.gaussian_truncate_sigma,
                            fill_alpha=0.16,
                            outline_alpha=0.70,
                            outline_thickness_px=1,
                            prob_threshold=gaussian_prob_threshold,
                            prob_alpha_scale=gaussian_prob_alpha_scale,
                            use_max=True,
                        )
                if (cand_x_i is not None) and (cand_valid_np is not None):
                    xt_c = cand_x_i[t].reshape(-1)
                    yt_c = cand_y_i[t].reshape(-1)
                    vt_c = cand_valid_np[t].reshape(-1)
                    valid_count = int(vt_c.sum())
                    if vt_c.any():
                        self._draw_marker_splats(
                            all_ov,
                            xt_c[vt_c],
                            yt_c[vt_c],
                            lo_color,
                            radius=max(1, marker_radius),
                            use_max=True,
                        )

                if (sel_x_i is not None) and (sel_valid_np is not None):
                    xt_s = sel_x_i[t].reshape(-1)
                    yt_s = sel_y_i[t].reshape(-1)
                    vt_s = sel_valid_np[t].reshape(-1)
                    hi_count = int(vt_s.sum())
                    lo_count = max(0, valid_count - hi_count)
                    if vt_s.any():
                        self._draw_marker_splats(
                            all_ov,
                            xt_s[vt_s],
                            yt_s[vt_s],
                            hi_color,
                            radius=max(1, marker_radius + 1),
                            use_max=True,
                        )
                        self._draw_marker_splats(
                            hi_ov,
                            xt_s[vt_s],
                            yt_s[vt_s],
                            hi_color,
                            radius=max(1, marker_radius + 1),
                            use_max=True,
                        )
                        if sel_cls_np is not None:
                            raw_cls_hi = sel_cls_np[t].reshape(-1)[vt_s]
                            hi_colors = np.stack(
                                [
                                    pred_cls_palette[int(cls_id)]
                                    if 0 <= int(cls_id) < len(pred_cls_palette)
                                    else hi_color
                                    for cls_id in raw_cls_hi.tolist()
                                ],
                                axis=0,
                            ).astype(np.uint8)
                        else:
                            hi_colors = np.repeat(hi_color[None, :], repeats=int(hi_count), axis=0)
                        self._draw_marker_splats(
                            hi_cls_ov,
                            xt_s[vt_s],
                            yt_s[vt_s],
                            hi_colors,
                            radius=max(1, marker_radius + 1),
                            use_max=True,
                        )
                if (mat_x_i is not None) and (mat_valid_np is not None):
                    xt_m = mat_x_i[t].reshape(-1)
                    yt_m = mat_y_i[t].reshape(-1)
                    vt_m = mat_valid_np[t].reshape(-1)
                    matched_count = int(vt_m.sum())
                    if vt_m.any():
                        self._draw_marker_splats(
                            matched_ov,
                            xt_m[vt_m],
                            yt_m[vt_m],
                            matched_color,
                            radius=max(1, marker_radius + 1),
                            use_max=True,
                        )
            else:
                if (cand_x_i is not None) and (cand_valid_np is not None):
                    xt_all = cand_x_i[t].reshape(-1)
                    yt_all = cand_y_i[t].reshape(-1)
                    vt_all = cand_valid_np[t].reshape(-1)
                    if cand_score_np is not None:
                        st_all = np.clip(cand_score_np[t].reshape(-1).astype(np.float32), 0.0, 1.0)
                    else:
                        st_all = np.ones_like(xt_all, dtype=np.float32)
                    hi = vt_all & (st_all >= conf_thr)
                    lo = vt_all & (~hi)
                    valid_count = int(vt_all.sum())
                    hi_count = int(hi.sum())
                    lo_count = int(lo.sum())
                    if lo.any():
                        self._draw_marker_splats(
                            all_ov,
                            xt_all[lo],
                            yt_all[lo],
                            lo_color,
                            radius=max(1, marker_radius),
                            use_max=True,
                        )
                    if hi.any():
                        self._draw_marker_splats(
                            all_ov,
                            xt_all[hi],
                            yt_all[hi],
                            hi_color,
                            radius=max(1, marker_radius + 1),
                            use_max=True,
                        )
                        self._draw_marker_splats(
                            hi_ov,
                            xt_all[hi],
                            yt_all[hi],
                            hi_color,
                            radius=max(1, marker_radius + 1),
                            use_max=True,
                        )
                        if cand_cls_np is not None:
                            raw_cls_hi = cand_cls_np[t].reshape(-1)[hi]
                            hi_colors = np.stack(
                                [
                                    pred_cls_palette[int(cls_id)]
                                    if 0 <= int(cls_id) < len(pred_cls_palette)
                                    else hi_color
                                    for cls_id in raw_cls_hi.tolist()
                                ],
                                axis=0,
                            ).astype(np.uint8)
                        else:
                            hi_colors = np.repeat(hi_color[None, :], repeats=int(hi_count), axis=0)
                        self._draw_marker_splats(
                            hi_cls_ov,
                            xt_all[hi],
                            yt_all[hi],
                            hi_colors,
                            radius=max(1, marker_radius + 1),
                            use_max=True,
                        )
                matched_count = 0

            if ego_visible:
                for row_canvas in (gt_rgb, gt_cls_rgb, all_ov, hi_ov, hi_cls_ov, matched_ov, traj_ov):
                    self._draw_cross_marker(
                        row_canvas,
                        center_x=ego_ix,
                        center_y=ego_iy,
                        color=ego_color,
                        arm=ego_cross_arm,
                        thickness=ego_cross_thickness,
                        outline_color=ego_outline_color,
                        outline_thickness=1,
                        use_max=False,
                    )

            row_gt.append(gt_rgb)
            row_all.append(all_ov)
            row_hi_cls.append(hi_cls_ov)
            row_matched.append(matched_ov)
            row_gt_cls.append(gt_cls_rgb)
            row_traj.append(traj_ov)
            row_base_traj.append(traj_ov.copy())
            stats_valid.append(valid_count)
            stats_hi.append(hi_count)
            stats_lo.append(lo_count)
            stats_matched.append(matched_count)

        gap = 4
        text_h = 18
        row_h = Y
        canvas_w = T * X + (T - 1) * gap
        has_base_traj_row = (
            base_traj_x_i is not None
            and base_traj_y_i is not None
            and base_traj_valid_np is not None
        )
        num_rows = 6 if has_base_traj_row else 5
        legend_h = 120
        canvas_h = text_h + num_rows * row_h + (num_rows - 1) * gap + legend_h
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for t in range(T):
            x0 = t * (X + gap)
            y0 = text_h
            y1 = y0 + row_h + gap
            y2 = y1 + row_h + gap
            y3 = y2 + row_h + gap
            y4 = y3 + row_h + gap
            y5 = y4 + row_h + gap
            canvas[y0:y0 + row_h, x0:x0 + X] = row_gt_cls[t]
            canvas[y1:y1 + row_h, x0:x0 + X] = row_all[t]
            canvas[y2:y2 + row_h, x0:x0 + X] = row_hi_cls[t]
            canvas[y3:y3 + row_h, x0:x0 + X] = row_matched[t]
            canvas[y4:y4 + row_h, x0:x0 + X] = row_traj[t]
            if has_base_traj_row:
                canvas[y5:y5 + row_h, x0:x0 + X] = row_base_traj[t]

        img = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(img)
        traj_row_start_y = text_h + 4 * (row_h + gap)
        base_traj_row_start_y = text_h + 5 * (row_h + gap)
        gaussian_vis_mode = str(getattr(self, "debug_query_gaussian_vis_mode", "ellipse")).lower()
        if gaussian_vis_mode == "prob":
            gaussian_desc = (
                f"rotated Gaussian prob map (thr={float(getattr(self, 'debug_query_gaussian_prob_threshold', 0.5)):.2f})"
            )
        elif gaussian_vis_mode == "threshold":
            gaussian_desc = (
                f"rotated Gaussian prob>={float(getattr(self, 'debug_query_gaussian_prob_threshold', 0.5)):.2f} fill"
            )
        else:
            gaussian_desc = "rotated Gaussian ellipse footprint"
        if use_score_bundle:
            score_q = query_vis_bundle.get("score_q", None)
            iou_q = query_vis_bundle.get("iou_q", None)
            cls_prob_q = query_vis_bundle.get("cls_prob_q", None)
            cam_attn_score_q = query_vis_bundle.get("cam_attn_score_q", None)
            cam_attn_score_valid_q = query_vis_bundle.get("cam_attn_score_valid_q", None)
            score_mean = float(score_q.float().mean().item()) if torch.is_tensor(score_q) and score_q.numel() > 0 else 0.0
            iou_mean = float(iou_q.float().mean().item()) if torch.is_tensor(iou_q) and iou_q.numel() > 0 else 0.0
            cls_mean = float(cls_prob_q.float().mean().item()) if torch.is_tensor(cls_prob_q) and cls_prob_q.numel() > 0 else 0.0
            cam_mean = 0.0
            if torch.is_tensor(cam_attn_score_q) and cam_attn_score_q.numel() > 0:
                if torch.is_tensor(cam_attn_score_valid_q) and cam_attn_score_valid_q.numel() == cam_attn_score_q.numel():
                    valid_mask = cam_attn_score_valid_q.to(dtype=torch.bool)
                    if bool(valid_mask.any().item()):
                        cam_mean = float(cam_attn_score_q[valid_mask].float().mean().item())
                else:
                    cam_mean = float(cam_attn_score_q.float().mean().item())
            header = (
                f"row1: GT class BEV(gt_occ_inst cls)  "
                f"row2: candidates(all queries, bg+fg)  "
                f"row3: selected(class-colored)  "
                f"row4: Hungarian-matched query centers only  "
                f"row5: refined traj(prev->cur; pred=red, all GT=cyan)  "
                f"{'row6: base traj(prev->cur; pred=red, all GT=cyan)  ' if has_base_traj_row else ''}"
                f"| topk={bundle_top_k} thr={bundle_score_thr:.2f} "
                f"w_iou={bundle_w_iou:.2f} w_cls={bundle_w_cls:.2f} w_cam={bundle_w_cam:.2f} "
                f"mean(iou/cls/cam/score)=({iou_mean:.3f}/{cls_mean:.3f}/{cam_mean:.3f}/{score_mean:.3f})"
            )
            if bundle_has_gaussian:
                header += f" | Gaussian vis: {gaussian_desc}"
        else:
            header = (
                f"row1: GT class BEV(gt_occ_inst cls), class-colored  "
                f"row2: all query centers (green=gt, cyan=conf>={conf_thr:.2f}, red=conf<{conf_thr:.2f})  "
                f"row3: query centers with conf>={conf_thr:.2f} only, class-colored  "
                f"row4: Hungarian-matched query centers only  "
                f"row5: refined traj(prev->cur; pred=red, all GT=cyan)  "
                f"{'row6: base traj(prev->cur; pred=red, all GT=cyan)' if has_base_traj_row else ''}"
            )
        if ego_visible:
            header += f" | ego(+)=xy(0,0)->pix({ego_ix},{ego_iy})"
        else:
            header += " | ego(+)=xy(0,0) out-of-range"
        draw.text((2, 1), header, fill=(255, 255, 255))

        for t in range(T):
            x0 = t * (X + gap)
            y0 = text_h
            y1 = y0 + row_h + gap
            y2 = y1 + row_h + gap
            y3 = y2 + row_h + gap
            y4 = y3 + row_h + gap
            y5 = y4 + row_h + gap
            draw.text((x0 + 2, y0 + 2), "GT class", fill=(255, 255, 255))
            if use_score_bundle:
                draw.text((x0 + 2, y1 + 2), f"t={t} cand={stats_valid[t]} sel={stats_hi[t]} drop={stats_lo[t]}", fill=(255, 255, 255))
            else:
                draw.text((x0 + 2, y1 + 2), f"t={t} all={stats_valid[t]} hi={stats_hi[t]} lo={stats_lo[t]}", fill=(255, 255, 255))
            draw.text((x0 + 2, y2 + 2), f"hi(class)={stats_hi[t]}", fill=(255, 255, 255))
            draw.text((x0 + 2, y3 + 2), f"matched={stats_matched[t]}", fill=(255, 255, 255))
            draw.text((x0 + 2, y4 + 2), "refined traj", fill=(255, 255, 255))
            if has_base_traj_row:
                draw.text((x0 + 2, y5 + 2), "base traj", fill=(255, 255, 255))

            # Thin white border for readability across every grid tile.
            border_rows = (y0, y1, y2, y3, y4, y5) if has_base_traj_row else (y0, y1, y2, y3, y4)
            for yy in border_rows:
                draw.rectangle(
                    [x0, yy, x0 + X - 1, yy + row_h - 1],
                    outline=(255, 255, 255),
                    width=1,
                )

            if t > 0:
                if (
                    traj_x_i is not None
                    and traj_y_i is not None
                    and traj_valid_np is not None
                    and t < int(traj_valid_np.shape[0])
                ):
                    pred_valid = traj_valid_np[t - 1].reshape(-1) & traj_valid_np[t].reshape(-1)
                    if pred_valid.any():
                        x_prev = traj_x_i[t - 1].reshape(-1)[pred_valid]
                        y_prev = traj_y_i[t - 1].reshape(-1)[pred_valid]
                        x_cur = traj_x_i[t].reshape(-1)[pred_valid]
                        y_cur = traj_y_i[t].reshape(-1)[pred_valid]
                        for px0, py0, px1, py1 in zip(x_prev.tolist(), y_prev.tolist(), x_cur.tolist(), y_cur.tolist()):
                            _draw_traj_arrow(
                                draw,
                                x0 + int(px0),
                                traj_row_start_y + int(py0),
                                x0 + int(px1),
                                traj_row_start_y + int(py1),
                                tuple(int(v) for v in traj_pred_color.tolist()),
                            )
                if (
                    gt_traj_x_i is not None
                    and gt_traj_y_i is not None
                    and gt_traj_valid_np is not None
                    and t < int(gt_traj_valid_np.shape[0])
                ):
                    gt_valid = gt_traj_valid_np[t - 1].reshape(-1) & gt_traj_valid_np[t].reshape(-1)
                    if gt_valid.any():
                        gx_prev = gt_traj_x_i[t - 1].reshape(-1)[gt_valid]
                        gy_prev = gt_traj_y_i[t - 1].reshape(-1)[gt_valid]
                        gx_cur = gt_traj_x_i[t].reshape(-1)[gt_valid]
                        gy_cur = gt_traj_y_i[t].reshape(-1)[gt_valid]
                        for px0, py0, px1, py1 in zip(gx_prev.tolist(), gy_prev.tolist(), gx_cur.tolist(), gy_cur.tolist()):
                            _draw_traj_arrow(
                                draw,
                                x0 + int(px0),
                                traj_row_start_y + int(py0),
                                x0 + int(px1),
                                traj_row_start_y + int(py1),
                                tuple(int(v) for v in traj_gt_color.tolist()),
                            )
                if (
                    has_base_traj_row
                    and base_traj_x_i is not None
                    and base_traj_y_i is not None
                    and base_traj_valid_np is not None
                    and t < int(base_traj_valid_np.shape[0])
                ):
                    base_valid = base_traj_valid_np[t - 1].reshape(-1) & base_traj_valid_np[t].reshape(-1)
                    if base_valid.any():
                        bx_prev = base_traj_x_i[t - 1].reshape(-1)[base_valid]
                        by_prev = base_traj_y_i[t - 1].reshape(-1)[base_valid]
                        bx_cur = base_traj_x_i[t].reshape(-1)[base_valid]
                        by_cur = base_traj_y_i[t].reshape(-1)[base_valid]
                        for px0, py0, px1, py1 in zip(bx_prev.tolist(), by_prev.tolist(), bx_cur.tolist(), by_cur.tolist()):
                            _draw_traj_arrow(
                                draw,
                                x0 + int(px0),
                                base_traj_row_start_y + int(py0),
                                x0 + int(px1),
                                base_traj_row_start_y + int(py1),
                                tuple(int(v) for v in traj_pred_color.tolist()),
                            )
                if (
                    has_base_traj_row
                    and gt_traj_x_i is not None
                    and gt_traj_y_i is not None
                    and gt_traj_valid_np is not None
                    and t < int(gt_traj_valid_np.shape[0])
                ):
                    gt_valid = gt_traj_valid_np[t - 1].reshape(-1) & gt_traj_valid_np[t].reshape(-1)
                    if gt_valid.any():
                        gx_prev = gt_traj_x_i[t - 1].reshape(-1)[gt_valid]
                        gy_prev = gt_traj_y_i[t - 1].reshape(-1)[gt_valid]
                        gx_cur = gt_traj_x_i[t].reshape(-1)[gt_valid]
                        gy_cur = gt_traj_y_i[t].reshape(-1)[gt_valid]
                        for px0, py0, px1, py1 in zip(gx_prev.tolist(), gy_prev.tolist(), gx_cur.tolist(), gy_cur.tolist()):
                            _draw_traj_arrow(
                                draw,
                                x0 + int(px0),
                                base_traj_row_start_y + int(py0),
                                x0 + int(px1),
                                base_traj_row_start_y + int(py1),
                                tuple(int(v) for v in traj_gt_color.tolist()),
                            )

        legend_y = text_h + num_rows * row_h + (num_rows - 1) * gap + 4
        draw.text((2, legend_y), "Legend:", fill=(255, 255, 255))
        generic_h = self._draw_generic_vis_legend(
            draw,
            x_start=60,
            y_start=legend_y,
            canvas_w=canvas_w,
            conf_thr=conf_thr,
            matched_color=tuple(int(v) for v in matched_color.tolist()),
            ego_color=tuple(int(v) for v in ego_color.tolist()),
            gaussian_label=gaussian_desc,
        )
        self._draw_query_vis_legend(
            draw,
            x_start=60,
            y_start=legend_y + generic_h + 6,
            canvas_w=canvas_w,
        )

        vis_dir = str(getattr(self, "debug_vis_dir", "./work_dirs/query_debug_vis"))
        os.makedirs(vis_dir, exist_ok=True)
        out_path = os.path.join(vis_dir, f"iter_{int(step):06d}_prob.png")
        img.save(out_path)
    
    def gt_occ_to_gmo_binary(self, gt_occ: torch.Tensor,
                            gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
                            ignore_index: int = 255) -> torch.Tensor:
        """
        gt_occ: (..., ) 정수 라벨 텐서
        예: (B, T, Z, X, Y) 또는 (B, T, H, W) 등 어떤 shape도 가능
        return:
        동일 shape, 값은 {0, 1, ignore_index}
        """
        gt = gt_occ.to(torch.long)

        out = torch.zeros_like(gt, dtype=torch.long)

        # ignore는 보존
        ignore_mask = (gt == ignore_index)

        # torch.isin 없이도 동작하도록 구현 (버전 안전)
        gmo_ids_t = gt.new_tensor(gmo_ids, dtype=torch.long)
        gmo_mask = (gt.unsqueeze(-1) == gmo_ids_t).any(dim=-1)

        out[gmo_mask] = 1
        out[ignore_mask] = ignore_index
        return out

    def _pred_occ_to_zxy(self, pred_occ: torch.Tensor, pred_layout: str = "zyx") -> torch.Tensor:
        """
        Convert predicted occupancy tensor to canonical [T, Z, X, Y] layout.

        Args:
            pred_occ:
                [T,1,Z,*,*] or [T,Z,*,*]
            pred_layout:
                - "zyx": pred spatial dims are [Z, Y, X] (voxelizer output)
                - "zxy": pred spatial dims are [Z, X, Y]
        """
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
            # [T, Z, Y, X] -> [T, Z, X, Y]
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
        # pred_occ in voxelizer path is [T,1,Z,Y,X], convert to canonical [T,Z,X,Y]
        pred_occ = self._pred_occ_to_zxy(pred_occ, pred_layout=pred_layout)
        T_pred = int(pred_occ.shape[0])

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
            raise ValueError(f"Unsupported gt_occ shape for gmo loss: {tuple(gt_occ_t.shape)}")

        if gt_occ_t.shape[0] < T_pred:
            raise ValueError(f"gt_occ time dim too short: {gt_occ_t.shape[0]} < {T_pred}")
        if gt_occ_t.shape[0] != T_pred:
            gt_occ_t = self._select_temporal_gt_for_prediction(gt_occ_t, T_pred)

        # [T,X,Y,Z] -> [T,Z,X,Y]
        gt_occ_t = gt_occ_t.permute(0, 3, 1, 2).contiguous()
        gt_occ_gmo = self.gt_occ_to_gmo_binary(
            gt_occ_t, gmo_ids=gmo_ids, ignore_index=ignore_index
        )

        valid = (gt_occ_gmo != ignore_index)
        target_fg = (gt_occ_gmo == 1).float()

        # if self.iter % 50 == 0:
            # breakpoint()

        eps = 1e-6
        p = pred_occ.clamp(eps, 1.0 - eps)     # prob
        t = target_fg                         # 0/1

        # 마스크
        pos_mask = valid & (t > 0.5)
        neg_mask = valid & (t < 0.5)

        # elementwise BCE (shape 유지, boolean indexing 큰 벡터 생성 피함)
        loss_map = F.binary_cross_entropy(p, t, reduction="none")

        pos_cnt = pos_mask.sum().clamp(min=1).float()
        neg_cnt = neg_mask.sum().clamp(min=1).float()

        loss_pos = (loss_map * pos_mask.float()).sum() / pos_cnt
        loss_neg = (loss_map * neg_mask.float()).sum() / neg_cnt

        # 균형 합산
        loss = 0.5 * loss_pos + 0.5 * loss_neg
        loss_dict = {}
        loss_dict['loss_gmo_bce'] = loss * loss_weight

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
            step=step,
            max_frames=6,
            voxel_center_offset=0.5,
            prob_threshold=0.5,
            pred_layout="zxy",
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
        gt_occ_inst_cls_t, gt_occ_inst_t = self._normalize_gt_occ_inst_for_vis(gt_occ_inst)
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
            gt_occ_sem_t = self._normalize_gt_occ_semantic_for_vis(gt_occ_semantic)

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
            and self._eval_vis_enabled_this_rank()
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
                    "candidate_mixture_quat_tqg4",
                    "candidate_mixture_weights_tqg",
                    "selected_mixture_centers_tqg3",
                    "selected_mixture_sigmas_tqg3",
                    "selected_mixture_quat_tqg4",
                    "selected_mixture_weights_tqg",
                    "matched_mixture_centers_tqg3",
                    "matched_mixture_sigmas_tqg3",
                    "matched_mixture_quat_tqg4",
                    "matched_mixture_weights_tqg",
                    "selected_query_idx_q",
                    "base_points_tq3",
                    "base_mixture_centers_tqg3",
                ):
                    val = query_vis_bundle.get(key, None)
                    if torch.is_tensor(val):
                        sidecar[key] = val.detach().cpu()
                sidecar["top_k"] = int(query_vis_bundle.get("top_k", 0))
                sidecar["score_thr"] = float(query_vis_bundle.get("score_thr", 0.0))
                sidecar["distance_nms_radius_m"] = float(query_vis_bundle.get("distance_nms_radius_m", 0.0))
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
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        gaussian_truncate_sigma: float = None,
        gaussian_sigma_floor_vox: float = 0.35,
    ):
        """
        p2gt loss.
        - gaussian 입력 시: E_{x~N(mu,sigma)}[DT(x)]를 근사 계산
        - legacy points 입력 시: point 샘플링 기반 DT loss
        """
        base_tensor = mixture_centers_world_tqg3 if mixture_centers_world_tqg3 is not None else gaussian_centers_world
        if base_tensor is None:
            base_tensor = points_world
        if occ_dt is None:
            if base_tensor is None:
                z = self.center_head.mlp.net[-1].weight.sum() * 0.0
                return {"loss_query_dt": z}
            z = base_tensor.sum() * 0.0
            return {"loss_query_dt": z}

        use_mixture = (
            (not self.center_only_mode)
            and torch.is_tensor(mixture_centers_world_tqg3)
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
        if use_mixture:
            centers = mixture_centers_world_tqg3.to(torch.float32)
            sigmas = mixture_sigmas_world_tqg3.to(torch.float32)
            rot = quat_to_rotmat_wxyz(mixture_quat_tqg4.to(torch.float32))
            weights = mixture_weights_tqg.to(torch.float32).clamp(0.0, 1.0)
            T_pred, Q, G, _ = centers.shape

            occ_dt_t = self._prepare_occ_dt(occ_dt, T_pred, centers.device)
            if dt_clip_max is not None:
                occ_dt_t = occ_dt_t.clamp(max=float(dt_clip_max))

            X = int(occ_dt_t.shape[1])
            Y = int(occ_dt_t.shape[2])
            Z = int(occ_dt_t.shape[3])
            pc_min = centers.new_tensor(self.point_cloud_range[:3])
            extent = centers.new_tensor(self.spatial_extent3d)
            voxel_size = extent / centers.new_tensor([float(X), float(Y), float(Z)]).clamp_min(1.0)
            off = float(voxel_center_offset)
            c = centers.clone()
            c[..., 0] = (c[..., 0] - pc_min[0]) / voxel_size[0] - off
            c[..., 1] = (c[..., 1] - pc_min[1]) / voxel_size[1] - off
            c[..., 2] = (c[..., 2] - pc_min[2]) / voxel_size[2] - off
            s = sigmas.clone()
            s[..., 0] = (s[..., 0] / voxel_size[0]).clamp(min=float(gaussian_sigma_floor_vox))
            s[..., 1] = (s[..., 1] / voxel_size[1]).clamp(min=float(gaussian_sigma_floor_vox))
            s[..., 2] = (s[..., 2] / voxel_size[2]).clamp(min=float(gaussian_sigma_floor_vox))
            std_vox = torch.sqrt((rot.pow(2) * s.pow(2).unsqueeze(-2)).sum(dim=-1).clamp_min(1e-12))

            trunc = self.gaussian_truncate_sigma if gaussian_truncate_sigma is None else float(gaussian_truncate_sigma)
            trunc = max(1.0, float(trunc))
            trunc2 = trunc * trunc
            loss_acc = centers.new_tensor(0.0)
            weight_acc = centers.new_tensor(0.0)

            for t in range(T_pred):
                dt_t = occ_dt_t[t]
                for q in range(Q):
                    for g in range(G):
                        cxi, cyi, czi = c[t, q, g]
                        sxi, syi, szi = s[t, q, g]
                        rx = max(1, int(torch.ceil(trunc * std_vox[t, q, g, 0]).item()))
                        ry = max(1, int(torch.ceil(trunc * std_vox[t, q, g, 1]).item()))
                        rz = max(1, int(torch.ceil(trunc * std_vox[t, q, g, 2]).item()))
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
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                        dx = xx - cxi
                        dy = yy - cyi
                        dz = zz - czi
                        r = rot[t, q, g]
                        xr = r[0, 0] * dx + r[1, 0] * dy + r[2, 0] * dz
                        yr = r[0, 1] * dx + r[1, 1] * dy + r[2, 1] * dz
                        zr = r[0, 2] * dx + r[1, 2] * dy + r[2, 2] * dz
                        md2 = (xr / sxi).pow(2) + (yr / syi).pow(2) + (zr / szi).pow(2)
                        w = torch.exp(-0.5 * md2) * (md2 <= trunc2).to(md2.dtype)
                        xx_i, yy_i, zz_i = torch.meshgrid(
                            xs.to(torch.long), ys.to(torch.long), zs.to(torch.long), indexing="ij"
                        )
                        comp_dt = (w * dt_t[xx_i, yy_i, zz_i]).sum() / w.sum().clamp(min=1e-6)
                        alpha = weights[t, q, g]
                        loss_acc = loss_acc + alpha * comp_dt
                        weight_acc = weight_acc + alpha

            if float(weight_acc.detach().item()) <= 0.0:
                return {"loss_query_dt": centers.sum() * 0.0}
            return {"loss_query_dt": (loss_acc / weight_acc.clamp(min=1e-6)) * float(loss_weight)}

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
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
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

        use_mixture = (
            (not self.center_only_mode)
            and torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_quat_tqg4)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and mixture_quat_tqg4.dim() == 4
            and int(mixture_quat_tqg4.shape[-1]) == 4
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_quat_tqg4.shape[:3])
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
            and int(mixture_centers_world_tqg3.shape[0]) >= T_pred
            and int(mixture_centers_world_tqg3.shape[1]) == Q
        )
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

            if sigma_policy == "pred_cov_q_only" and not use_mixture:
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
        if use_mixture:
            comp_mu_tqg3 = mixture_centers_world_tqg3[:T_pred].to(device=pred_mu_tq3.device, dtype=torch.float32)
            comp_sigma_tqg3 = mixture_sigmas_world_tqg3[:T_pred].to(device=pred_mu_tq3.device, dtype=torch.float32).clamp(min=1e-3)
            comp_rot_tqg33 = quat_to_rotmat_wxyz(
                mixture_quat_tqg4[:T_pred].to(device=pred_mu_tq3.device, dtype=torch.float32)
            )
            comp_weight_tqg = mixture_weights_tqg[:T_pred].to(device=pred_mu_tq3.device, dtype=torch.float32).clamp(0.0, 1.0)
            loss_cov_gt_acc = pred_mu_tq3.new_tensor(0.0)
            loss_cov_q_acc = pred_mu_tq3.new_tensor(0.0)
            loss_balance_acc = pred_mu_tq3.new_tensor(0.0)
            cov_frame_cnt = 0
            balance_frame_cnt = 0

            for t in range(T_pred):
                valid_n = gt_valid_tn[t]
                if not bool(valid_n.any().item()):
                    continue
                gt_k3 = gt_centers_tn3[t, valid_n, :]
                comp_m3 = comp_mu_tqg3[t].reshape(-1, 3)
                sigma_m3 = comp_sigma_tqg3[t].reshape(-1, 3)
                rot_m33 = comp_rot_tqg33[t].reshape(-1, 3, 3)
                weight_m = comp_weight_tqg[t].reshape(-1)
                if comp_m3.numel() == 0:
                    continue

                diff_mk3 = comp_m3[:, None, :] - gt_k3[None, :, :]
                d_mk_m = diff_mk3.pow(2).sum(dim=-1).clamp(min=eps).sqrt()
                local_diff_mk3 = torch.einsum("mkc,mcd->mkd", diff_mk3, rot_m33)
                d2_mk = (local_diff_mk3 / sigma_m3[:, None, :]).pow(2).sum(dim=-1)
                aff_mk = weight_m[:, None] * torch.exp(-0.5 * d2_mk / tau_v)
                lam_k = aff_mk.sum(dim=0)
                p_hit_k = (-torch.expm1(-lam_k)).clamp(min=eps, max=1.0)
                loss_cov_gt_t = -torch.log(p_hit_k).mean()

                lam_m = aff_mk.sum(dim=1)
                p_hit_m = (-torch.expm1(-lam_m)).clamp(min=eps, max=1.0)
                weight_den = weight_m.sum().clamp(min=eps)
                loss_cov_q_t = (-torch.log(p_hit_m) * weight_m).sum() / weight_den

                loss_cov_gt_acc = loss_cov_gt_acc + loss_cov_gt_t
                loss_cov_q_acc = loss_cov_q_acc + loss_cov_q_t
                cov_frame_cnt += 1

                if float(balance_weight) > 0.0:
                    resp_mk = aff_mk / aff_mk.sum(dim=1, keepdim=True).clamp(min=eps)
                    mass_k = (resp_mk * weight_m[:, None]).sum(dim=0)
                    target_mass = weight_m.sum() / float(max(1, int(gt_k3.shape[0])))
                    target_k = mass_k.new_full((int(gt_k3.shape[0]),), float(target_mass.detach().item()))
                    loss_balance_acc = loss_balance_acc + F.smooth_l1_loss(
                        mass_k / weight_den,
                        target_k / weight_den,
                        reduction="mean",
                        beta=0.1,
                    )
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

    # def compute_point_gt_knn_loss(
    #         self,
    #         points_world: torch.Tensor,
    #         gt_occ: torch.Tensor,
    #         voxelizer,
    #         gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
    #         ignore_index: int = 255,
    #         use_centers_only: bool = True,
    #         num_points_per_query: int = None,
    #         max_pred: int = 512,
    #         max_gt: int = 4096,
    #         chunk_gt: int = 1024,
    #         sqrt_dist: bool = True,
    #         eps: float = 1e-6,
    #     ) -> torch.Tensor:
    #         """
    #         목적: overlap=0 이어도 query center가 GT 양성 영역 쪽으로 이동하게 만드는 attractive loss.

    #         points_world:
    #         (B, T, N, 3) where N = num_queries * num_points_per_query
    #         또는 (B, T, num_queries, 3) 같은 center만 모아둔 텐서도 허용(그 경우 use_centers_only=False로 두면 됨)

    #         gt_occ:
    #         (B, T, D, H, W) 정수 라벨 occupancy (ignore_index 포함 가능)

    #         voxelizer:
    #         point_cloud_range, voxel_size를 가지고 있어야 함
    #         """
    #         device = points_world.device

    #         # 1) GT를 GMO binary로 변환
    #         gt_occ = torch.stack(gt_occ, dim=0)[-6:].squeeze(1).permute(0,3,1,2)  # [6, 40, 512, 512]
    #         gt_bin = self.gt_occ_to_gmo_binary(gt_occ, gmo_ids=gmo_ids, ignore_index=ignore_index)
    #         # gt_bin: same shape, {0,1,ignore}

    #         # 2) centers만 쓰기 (연산 줄이기)
    #         if use_centers_only:
    #             if points_world.dim() != 4:
    #                 raise ValueError(f"points_world must be (B,T,N,3), got {points_world.shape}")
    #             if num_points_per_query is None:
    #                 num_points_per_query = getattr(self, "num_points", None)
    #             if num_points_per_query is None:
    #                 raise ValueError("num_points_per_query is required when use_centers_only=True")

    #             B, T, N, _ = points_world.shape
    #             step = int(num_points_per_query)
    #             if N % step != 0:
    #                 raise ValueError(f"N={N} is not divisible by num_points_per_query={step}")

    #             center_idx = torch.arange(0, N, step, device=device)
    #             pred_centers = points_world.index_select(2, center_idx)  # (B,T,Nq,3)
    #         else:
    #             pred_centers = points_world  # assume already (B,T,Nq,3) or (B,T,N,3)

    #         # voxelizer params
    #         pcr = voxelizer.point_cloud_range
    #         vsize = voxelizer.voxel_size
    #         vsize = voxelizer.voxel_size

    #         if isinstance(vsize, (float, int)):
    #             vx = vy = vz = float(vsize)
    #         else:
    #             vx, vy, vz = float(vsize[0]), float(vsize[1]), float(vsize[2])

    #         min_x, min_y, min_z = float(pcr[0]), float(pcr[1]), float(pcr[2])

    #         B = gt_bin.shape[0]
    #         T = gt_bin.shape[1]

    #         losses = []
    #         for b in range(B):
    #             for t in range(T):
    #                 g = gt_bin[b, t]  # (D,H,W)
    #                 pos_mask = (g == 1)

    #                 # GT 양성이 없으면 스킵
    #                 if not pos_mask.any():
    #                     continue

    #                 # (K,3) with order (z,x,y)
    #                 pos_idx = pos_mask.nonzero(as_tuple=False)

    #                 # GT 샘플링으로 비용 제한
    #                 if max_gt is not None and pos_idx.shape[0] > max_gt:
    #                     with torch.no_grad():
    #                         perm = torch.randperm(pos_idx.shape[0], device=device)[:max_gt]
    #                     pos_idx = pos_idx[perm]

    #                 iz = pos_idx[:, 0].to(torch.float32)
    #                 ix = pos_idx[:, 1].to(torch.float32)
    #                 iy = pos_idx[:, 2].to(torch.float32)

    #                 gx = min_x + (ix + 0.5) * vx
    #                 gy = min_y + (iy + 0.5) * vy
    #                 gz = min_z + (iz + 0.5) * vz
    #                 gt_xyz = torch.stack([gx, gy, gz], dim=-1)  # (K,3)

    #                 pred = pred_centers[b, t]  # (Np,3)

    #                 # pred 샘플링으로 비용 제한
    #                 if max_pred is not None and pred.shape[0] > max_pred:
    #                     with torch.no_grad():
    #                         perm = torch.randperm(pred.shape[0], device=device)[:max_pred]
    #                     pred = pred[perm]

    #                 # KNN: pred -> nearest GT
    #                 # 메모리 줄이기 위해 gt를 chunk로 쪼갬
    #                 min_d2 = torch.full((pred.shape[0],), float("inf"), device=device, dtype=torch.float32)

    #                 for s in range(0, gt_xyz.shape[0], chunk_gt):
    #                     gt_chunk = gt_xyz[s : s + chunk_gt]  # (C,3)
    #                     d2 = (pred[:, None, :] - gt_chunk[None, :, :]).pow(2).sum(dim=-1)  # (Np,C)
    #                     min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)

    #                 if sqrt_dist:
    #                     dist = torch.sqrt(min_d2 + eps)
    #                     loss_bt = dist.mean()
    #                 else:
    #                     loss_bt = min_d2.mean()

    #                 losses.append(loss_bt)

    #         if len(losses) == 0:
    #             # GT pos가 전부 없던 경우
    #             return pred_centers.sum() * 0.0

    #         return torch.stack(losses).mean()

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
    print(out["query_sigma_world_tq3"])           # None (summary Gaussian disabled)
    print(out["mixture_centers_world_tqg3"].shape)  # [T, Q, G, 3]
    print(out["mixture_sigmas_world_tqg3"].shape)   # [T, Q, G, 3]
    print(out["mixture_quat_tqg4"].shape)             # [T, Q, G]
    print(out["mixture_weights_tqg"].shape)         # [T, Q, G]
    print(out["query_cls_logits_qc"].shape)         # [Q, C]
