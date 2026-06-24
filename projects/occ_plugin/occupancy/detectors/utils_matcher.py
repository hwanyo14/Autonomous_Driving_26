import torch
import torch.nn.functional as F
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def _solve_unique_assignment(cost_qn_np: np.ndarray):
    """Solve optimal unique assignment using Hungarian algorithm or greedy fallback."""
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_qn_np)
        return row_ind.astype(np.int64), col_ind.astype(np.int64)

    # fallback without scipy: greedy unique matching
    qn_order = np.argsort(cost_qn_np, axis=None)
    used_q = np.zeros(cost_qn_np.shape[0], dtype=np.bool_)
    used_n = np.zeros(cost_qn_np.shape[1], dtype=np.bool_)
    rows, cols = [], []
    need = min(cost_qn_np.shape[0], cost_qn_np.shape[1])
    for flat_idx in qn_order:
        q_idx = int(flat_idx // cost_qn_np.shape[1])
        n_idx = int(flat_idx % cost_qn_np.shape[1])
        if used_q[q_idx] or used_n[n_idx]:
            continue
        used_q[q_idx] = True
        used_n[n_idx] = True
        rows.append(q_idx)
        cols.append(n_idx)
        if len(rows) >= need:
            break
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


class EfficientOCFMatcherMixin:
    """Hungarian matching methods for EfficientOCF."""

    @staticmethod
    def _build_empty_match_result(device):
        empty_idx = torch.empty((0,), device=device, dtype=torch.long)
        return {
            "matched_query_idx": empty_idx,
            "matched_inst_idx": empty_idx,
            "recruit_query_idx": empty_idx,
            "recruit_inst_idx": empty_idx,
            "recruit_frame_indices": empty_idx,
            "sim_qn": None,
            "soft_assign_qn": None,
            "center_frame_idx": None,
            "cost_qn": None,
            "cost_feat_qn": None,
            "cost_feat_contrib_qn": None,
            "cost_soft_qn": None,
            "cost_soft_contrib_qn": None,
            "cost_cls_qn": None,
            "cost_cls_contrib_qn": None,
            "cost_center_qn": None,
            "cost_center_contrib_qn": None,
            "cost_temporal_offset_qn": None,
            "cost_temporal_offset_contrib_qn": None,
            "cost_bev_dice_qn": None,
            "cost_bev_dice_contrib_qn": None,
            "cost_attn_iou_qn": None,
            "cost_attn_iou_contrib_qn": None,
            "gt_centers_tn3": None,
            "gt_valid_tn": None,
            "gt_ids_n": None,
            "gt_cls_n": None,
            "gt_match_feat_tnd": None,
            "gt_match_feat_valid_tn": None,
            "gt_match_feat_ids_n": None,
            "gt_bev_feat_tnd": None,
            "gt_bev_feat_valid_tn": None,
            "gt_bev_feat_ids_n": None,
            "matched_query_feat_tkd": None,
            "matched_gt_feat_tkd": None,
            "matched_gt_feat_valid_tk": None,
        }

    def _compute_query_gt_bev_dice_cost_qn(
        self,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        gt_instance_occ3d_txyz: torch.Tensor,
        gt_ids_n: torch.Tensor,
        gt_valid_tn: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_yaw_tqg: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        pair_chunk_size: int = 8,
        eps: float = 1e-6,
    ):
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None, None
        if (not torch.is_tensor(gt_instance_occ3d_txyz)) or gt_instance_occ3d_txyz.dim() != 4:
            return None, None
        if (not torch.is_tensor(gt_ids_n)) or gt_ids_n.numel() <= 0:
            return None, None

        t_count = int(min(centers_world_tq3.shape[0], gt_instance_occ3d_txyz.shape[0]))
        q_count = int(centers_world_tq3.shape[1])
        n_inst = int(gt_ids_n.numel())
        if t_count <= 0 or q_count <= 0 or n_inst <= 0:
            return None, None

        use_mixture = (
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
        if use_mixture:
            pred_tq1zyx = self.matched_gmo_voxelizer.forward_gaussian_mixture_grouped(
                mixture_centers_world_tkg3=mixture_centers_world_tqg3[:t_count].to(torch.float32),
                mixture_sigmas_world_tkg3=mixture_sigmas_world_tqg3[:t_count].to(torch.float32),
                mixture_weights_tkg=mixture_weights_tqg[:t_count].to(torch.float32),
                mixture_yaw_tkg=mixture_yaw_tqg[:t_count].to(torch.float32),
                pair_weights_tk=None,
                pair_chunk_size=int(max(1, pair_chunk_size)),
            ).to(torch.float32)
        else:
            if (not torch.is_tensor(sigmas_world_tq3)) or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape):
                return None, None
            pred_tq1zyx = self.matched_gmo_voxelizer(
                centers_world_tq3[:t_count],
                sigmas_world=sigmas_world_tq3[:t_count],
                weights=None,
            ).to(torch.float32)

        if pred_tq1zyx.dim() == 6:
            pred_tqzyx = pred_tq1zyx[:, :, 0]
        elif pred_tq1zyx.dim() == 5:
            pred_tqzyx = pred_tq1zyx
        else:
            return None, None
        if int(pred_tqzyx.shape[0]) != t_count or int(pred_tqzyx.shape[1]) != q_count:
            return None, None

        pred_bev_tqp = pred_tqzyx.amax(dim=2).reshape(t_count, q_count, -1).clamp(0.0, 1.0)
        gt_occ_sel = gt_instance_occ3d_txyz[:t_count].to(device=pred_tqzyx.device, dtype=torch.long)
        gt_ids = gt_ids_n.to(device=pred_tqzyx.device, dtype=torch.long)

        if (
            torch.is_tensor(gt_valid_tn)
            and gt_valid_tn.dim() == 2
            and int(gt_valid_tn.shape[0]) >= t_count
            and int(gt_valid_tn.shape[1]) == n_inst
        ):
            valid_tn = gt_valid_tn[:t_count].to(device=pred_tqzyx.device, dtype=torch.bool)
        else:
            valid_tn = torch.ones((t_count, n_inst), device=pred_tqzyx.device, dtype=torch.bool)

        cost_sum_qn = pred_bev_tqp.new_zeros((q_count, n_inst))
        cost_cnt_qn = pred_bev_tqp.new_zeros((q_count, n_inst))

        pred_spatial = tuple(int(v) for v in pred_tqzyx.shape[-3:])
        full_spatial = (int(gt_occ_sel.shape[3]), int(gt_occ_sel.shape[2]), int(gt_occ_sel.shape[1]))
        max_compare_elems = 128 * 1024 * 1024
        per_inst_elems = int(t_count * gt_occ_sel.shape[1] * gt_occ_sel.shape[2] * gt_occ_sel.shape[3])
        max_chunk = int(max_compare_elems // max(1, per_inst_elems))
        gt_chunk_size = max(1, min(int(max(1, pair_chunk_size)), max_chunk, n_inst))
        pool_dtype = torch.float16 if gt_occ_sel.is_cuda else torch.float32
        alpha = float(max(0.0, tversky_alpha))
        beta = float(max(0.0, tversky_beta))
        eps_v = float(max(1e-12, eps))

        for n0 in range(0, n_inst, gt_chunk_size):
            n1 = min(n_inst, n0 + gt_chunk_size)
            kc = int(n1 - n0)
            if kc <= 0:
                continue
            gt_ids_chunk = gt_ids[n0:n1]
            gt_chunk_tkxyz = (gt_occ_sel[:, None] == gt_ids_chunk[None, :, None, None, None])
            gt_chunk_tk1zyx = gt_chunk_tkxyz.permute(0, 1, 4, 3, 2).unsqueeze(2)
            if full_spatial != pred_spatial:
                gt_chunk_lowres = F.adaptive_max_pool3d(
                    gt_chunk_tk1zyx.reshape(t_count * kc, 1, *full_spatial).to(pool_dtype),
                    output_size=pred_spatial,
                ).reshape(t_count, kc, 1, *pred_spatial)
            else:
                gt_chunk_lowres = gt_chunk_tk1zyx.to(pool_dtype)
            gt_bev_tkp = gt_chunk_lowres[:, :, 0].amax(dim=2).reshape(t_count, kc, -1).to(torch.float32).clamp(0.0, 1.0)

            for t in range(t_count):
                valid_1k = valid_tn[t, n0:n1].to(dtype=torch.float32)[None, :]
                if not bool((valid_1k > 0.0).any().item()):
                    continue
                pred_qp = pred_bev_tqp[t]
                gt_kp = gt_bev_tkp[t]
                tp_qk = torch.matmul(pred_qp, gt_kp.t())
                pred_sum_q1 = pred_qp.sum(dim=1, keepdim=True)
                gt_sum_1k = gt_kp.sum(dim=1, keepdim=True).t()
                fp_qk = (pred_sum_q1 - tp_qk).clamp_min(0.0)
                fn_qk = (gt_sum_1k - tp_qk).clamp_min(0.0)
                denom_qk = tp_qk + (alpha * fp_qk) + (beta * fn_qk)
                score_qk = (tp_qk + eps_v) / (denom_qk + eps_v)
                cost_qk = (1.0 - score_qk).clamp(0.0, 1.0)
                cost_sum_qn[:, n0:n1] += cost_qk * valid_1k
                cost_cnt_qn[:, n0:n1] += valid_1k

        valid_qn = cost_cnt_qn > 0.0
        cost_qn = cost_sum_qn / cost_cnt_qn.clamp_min(1.0)
        return cost_qn, valid_qn

    def _compute_query_attn_soft_iou_cost_qn(
        self,
        query_attn_weights_tqnhw: torch.Tensor,
        gt_attn_targets: dict,
        pred_norm: str = "amax",
        eps: float = 1e-6,
    ):
        if (not torch.is_tensor(query_attn_weights_tqnhw)) or query_attn_weights_tqnhw.dim() != 5:
            return None, None
        if not isinstance(gt_attn_targets, dict):
            return None, None

        gt_inst_mask_tnhw = gt_attn_targets.get("gt_inst_mask_tnhw", None)
        gt_inst_valid_tn = gt_attn_targets.get("gt_inst_valid_tn", None)
        attn_t_idx_t = gt_attn_targets.get("attn_t_idx_t", None)
        if (
            (not torch.is_tensor(gt_inst_mask_tnhw))
            or gt_inst_mask_tnhw.dim() != 4
            or (not torch.is_tensor(gt_inst_valid_tn))
            or gt_inst_valid_tn.dim() != 2
        ):
            return None, None

        t_attn, q_count, _n_cam, h_attn, w_attn = [int(v) for v in query_attn_weights_tqnhw.shape]
        t_gt, n_inst, h_gt, w_gt = [int(v) for v in gt_inst_mask_tnhw.shape]
        if (
            t_attn <= 0
            or q_count <= 0
            or t_gt <= 0
            or n_inst <= 0
            or h_attn != h_gt
            or w_attn != w_gt
            or int(gt_inst_valid_tn.shape[0]) != t_gt
            or int(gt_inst_valid_tn.shape[1]) != n_inst
        ):
            return None, None

        t = min(t_attn, t_gt)
        if t <= 0:
            return None, None

        if torch.is_tensor(attn_t_idx_t) and int(attn_t_idx_t.numel()) >= t:
            attn_idx = attn_t_idx_t[:t].to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
            if not bool(((attn_idx >= 0) & (attn_idx < t_attn)).all().item()):
                return None, None
            attn_sel_tqnhw = query_attn_weights_tqnhw.index_select(0, attn_idx)
        else:
            attn_sel_tqnhw = query_attn_weights_tqnhw[:t]

        eps_v = float(max(1e-12, eps))
        pred_tqp = attn_sel_tqnhw.to(torch.float32).sum(dim=2).reshape(t, q_count, -1).clamp_min(0.0)

        pred_norm_mode = str(pred_norm).lower()
        if pred_norm_mode == "sum":
            pred_den_tq1 = pred_tqp.sum(dim=-1, keepdim=True)
            pred_tqp = pred_tqp / pred_den_tq1.clamp_min(eps_v)
        elif pred_norm_mode == "none":
            pred_tqp = pred_tqp.clamp(0.0, 1.0)
        else:
            pred_den_tq1 = pred_tqp.amax(dim=-1, keepdim=True)
            pred_tqp = pred_tqp / pred_den_tq1.clamp_min(eps_v)
        pred_tqp = pred_tqp.clamp(0.0, 1.0)

        gt_tnp = gt_inst_mask_tnhw[:t].to(device=pred_tqp.device, dtype=torch.float32).reshape(t, n_inst, -1)
        valid_tn = gt_inst_valid_tn[:t].to(device=pred_tqp.device, dtype=torch.bool)
        gt_sum_tn = gt_tnp.sum(dim=-1)
        valid_tn = valid_tn & (gt_sum_tn > 0.0)

        inter_tqn = torch.einsum("tqp,tnp->tqn", pred_tqp, gt_tnp)
        pred_sum_tq1 = pred_tqp.sum(dim=-1, keepdim=True)
        gt_sum_t1n = gt_sum_tn[:, None, :]
        union_tqn = (pred_sum_tq1 + gt_sum_t1n - inter_tqn).clamp_min(0.0)
        iou_tqn = inter_tqn / union_tqn.clamp_min(eps_v)
        cost_tqn = (1.0 - iou_tqn).clamp(0.0, 1.0)

        valid_t1n = valid_tn[:, None, :]
        finite_tqn = torch.isfinite(cost_tqn)
        valid_pair_tqn = finite_tqn & valid_t1n
        valid_count_qn = valid_pair_tqn.to(torch.float32).sum(dim=0)
        valid_pair_qn = valid_count_qn > 0.0
        if not bool(valid_pair_qn.any().item()):
            return None, None

        cost_qn = (torch.where(valid_pair_tqn, cost_tqn, torch.zeros_like(cost_tqn)).sum(dim=0)
                   / valid_count_qn.clamp_min(1.0))
        return cost_qn, valid_pair_qn

    def _compute_query_attn_inside_log_cost_qn(
        self,
        query_attn_weights_tqnhw: torch.Tensor,
        gt_attn_targets: dict,
        eps: float = 1e-6,
    ):
        if (not torch.is_tensor(query_attn_weights_tqnhw)) or query_attn_weights_tqnhw.dim() != 5:
            return None, None
        if not isinstance(gt_attn_targets, dict):
            return None, None

        gt_inst_mask_tnhw = gt_attn_targets.get("gt_inst_mask_tnhw", None)
        gt_inst_valid_tn = gt_attn_targets.get("gt_inst_valid_tn", None)
        attn_t_idx_t = gt_attn_targets.get("attn_t_idx_t", None)
        if (
            (not torch.is_tensor(gt_inst_mask_tnhw))
            or gt_inst_mask_tnhw.dim() != 4
            or (not torch.is_tensor(gt_inst_valid_tn))
            or gt_inst_valid_tn.dim() != 2
        ):
            return None, None

        t_attn, q_count, _n_cam, h_attn, w_attn = [int(v) for v in query_attn_weights_tqnhw.shape]
        t_gt, n_inst, h_gt, w_gt = [int(v) for v in gt_inst_mask_tnhw.shape]
        if (
            t_attn <= 0
            or q_count <= 0
            or t_gt <= 0
            or n_inst <= 0
            or h_attn != h_gt
            or w_attn != w_gt
            or int(gt_inst_valid_tn.shape[0]) != t_gt
            or int(gt_inst_valid_tn.shape[1]) != n_inst
        ):
            return None, None

        t = min(t_attn, t_gt)
        if t <= 0:
            return None, None

        if torch.is_tensor(attn_t_idx_t) and int(attn_t_idx_t.numel()) >= t:
            attn_idx = attn_t_idx_t[:t].to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
            if not bool(((attn_idx >= 0) & (attn_idx < t_attn)).all().item()):
                return None, None
            attn_sel_tqnhw = query_attn_weights_tqnhw.index_select(0, attn_idx)
        else:
            attn_sel_tqnhw = query_attn_weights_tqnhw[:t]

        eps_v = float(max(1e-12, eps))
        pred_tqp = attn_sel_tqnhw.to(torch.float32).sum(dim=2).reshape(t, q_count, -1).clamp_min(0.0)
        pred_tqp = pred_tqp / pred_tqp.sum(dim=-1, keepdim=True).clamp_min(eps_v)

        gt_tnp = gt_inst_mask_tnhw[:t].to(device=pred_tqp.device, dtype=torch.float32).reshape(t, n_inst, -1)
        valid_tn = gt_inst_valid_tn[:t].to(device=pred_tqp.device, dtype=torch.bool)
        valid_tn = valid_tn & (gt_tnp.sum(dim=-1) > 0.0)

        inside_mass_tqn = torch.einsum("tqp,tnp->tqn", pred_tqp, gt_tnp).clamp(0.0, 1.0)
        cost_tqn = (-torch.log(inside_mass_tqn.clamp_min(eps_v))).clamp(0.0, 30.0)

        valid_t1n = valid_tn[:, None, :]
        finite_tqn = torch.isfinite(cost_tqn)
        valid_pair_tqn = finite_tqn & valid_t1n
        valid_count_qn = valid_pair_tqn.to(torch.float32).sum(dim=0)
        valid_pair_qn = valid_count_qn > 0.0
        if not bool(valid_pair_qn.any().item()):
            return None, None

        cost_qn = (torch.where(valid_pair_tqn, cost_tqn, torch.zeros_like(cost_tqn)).sum(dim=0)
                   / valid_count_qn.clamp_min(1.0))
        return cost_qn, valid_pair_qn

    @staticmethod
    def _resolve_temporal_cost_frame_indices(
        frame_indices,
        t_limit: int,
    ):
        t_limit = int(t_limit)
        if t_limit <= 0:
            return None
        if frame_indices is None:
            return None
        if torch.is_tensor(frame_indices):
            idx_list = [int(v) for v in frame_indices.reshape(-1).tolist()]
        else:
            idx_list = [int(v) for v in frame_indices]
        idx_list = [idx for idx in idx_list if 0 <= int(idx) < t_limit]
        if len(idx_list) <= 0:
            return None
        return idx_list

    def _match_queries_to_gt_instances(
        self,
        centers_world: torch.Tensor,
        query_sigmas_world_tq3: torch.Tensor = None,
        query_mixture_centers_world_tqg3: torch.Tensor = None,
        query_mixture_sigmas_world_tqg3: torch.Tensor = None,
        query_mixture_yaw_tqg: torch.Tensor = None,
        query_mixture_weights_tqg: torch.Tensor = None,
        gt_instance_occ3d_txyz: torch.Tensor = None,
        gt_inst_center_world_tn3: torch.Tensor = None,
        gt_inst_center_valid_tn: torch.Tensor = None,
        gt_inst_ids_n: torch.Tensor = None,
        gt_inst_cls_n: torch.Tensor = None,
        gt_inst_cls_valid_n: torch.Tensor = None,
        query_img_feat_tqd: torch.Tensor = None,
        query_cls_logits_qc: torch.Tensor = None,
        gt_inst_bev_feat_tnd: torch.Tensor = None,
        gt_inst_bev_feat_valid_tn: torch.Tensor = None,
        gt_inst_bev_feat_ids_n: torch.Tensor = None,
        query_attn_weights_tqnhw: torch.Tensor = None,
        gt_attn_targets: dict = None,
        query_sim_cost_weight: float = 1.0,
        query_cls_cost_weight: float = 0.0,
        query_soft_assign_temp: float = 0.10,
        query_soft_assign_cost_weight: float = 0.0,
        query_bev_dice_cost_weight: float = 0.0,
        query_attn_match_cost_weight: float = 0.0,
        query_attn_match_metric: str = "soft_iou",
        query_attn_match_pred_norm: str = "amax",
        query_attn_match_eps: float = 1e-6,
        query_center_match_frame_idx: int = None,
        query_temporal_cost_frame_indices=None,
        query_temporal_offset_match_cost_weight: float = 0.0,
    ):
        match_result = self._build_empty_match_result(device=centers_world.device)

        if gt_inst_center_world_tn3 is None:
            return match_result
        if gt_inst_center_world_tn3.shape[0] != centers_world.shape[0]:
            raise ValueError(
                f"time mismatch between GT instance centers and predicted centers: "
                f"{tuple(gt_inst_center_world_tn3.shape)} vs {tuple(centers_world.shape)}"
            )

        match_gt_center_tn3 = gt_inst_center_world_tn3.to(device=centers_world.device, dtype=centers_world.dtype)
        match_valid_tn = gt_inst_center_valid_tn.to(device=centers_world.device, dtype=torch.bool)
        match_ids_n = None
        if torch.is_tensor(gt_inst_ids_n):
            match_ids_n = gt_inst_ids_n.to(device=centers_world.device, dtype=torch.long)
        match_cls_n = None
        match_cls_valid_n = None
        if torch.is_tensor(gt_inst_cls_n):
            match_cls_n = gt_inst_cls_n.to(device=centers_world.device, dtype=torch.long)
        if torch.is_tensor(gt_inst_cls_valid_n):
            match_cls_valid_n = gt_inst_cls_valid_n.to(device=centers_world.device, dtype=torch.bool)

        match_query_img_feat_tqd = None
        if torch.is_tensor(query_img_feat_tqd):
            if query_img_feat_tqd.dim() != 3:
                raise ValueError(f"query_img_feat_tqd must be [T,Q,D], got {tuple(query_img_feat_tqd.shape)}")
            if query_img_feat_tqd.shape[1] != centers_world.shape[1]:
                raise ValueError(
                    f"query_img_feat_tqd query dim mismatch with centers_world: "
                    f"{tuple(query_img_feat_tqd.shape)} vs {tuple(centers_world.shape)}"
                )
            match_query_img_feat_tqd = query_img_feat_tqd.to(device=centers_world.device, dtype=centers_world.dtype)

        match_gt_feat_tnd = None
        match_gt_feat_valid_tn = None
        match_gt_feat_ids_n = None
        if torch.is_tensor(gt_inst_bev_feat_tnd):
            if gt_inst_bev_feat_tnd.dim() != 3:
                raise ValueError(f"gt_inst_bev_feat_tnd must be [T,N,D], got {tuple(gt_inst_bev_feat_tnd.shape)}")
            match_gt_feat_tnd = gt_inst_bev_feat_tnd.to(device=centers_world.device, dtype=centers_world.dtype)
            if torch.is_tensor(gt_inst_bev_feat_valid_tn):
                if gt_inst_bev_feat_valid_tn.shape[:2] != match_gt_feat_tnd.shape[:2]:
                    raise ValueError(
                        "gt_inst_bev_feat_valid_tn shape mismatch: "
                        f"{tuple(gt_inst_bev_feat_valid_tn.shape)} vs {tuple(match_gt_feat_tnd.shape)}"
                    )
                match_gt_feat_valid_tn = gt_inst_bev_feat_valid_tn.to(device=centers_world.device, dtype=torch.bool)
            else:
                match_gt_feat_valid_tn = torch.ones(
                    match_gt_feat_tnd.shape[:2], device=centers_world.device, dtype=torch.bool
                )
            if torch.is_tensor(gt_inst_bev_feat_ids_n):
                match_gt_feat_ids_n = gt_inst_bev_feat_ids_n.to(device=centers_world.device, dtype=torch.long)

        if (match_ids_n is not None) and (match_ids_n.numel() == match_gt_center_tn3.shape[1]):
            id_order = torch.argsort(match_ids_n)
            match_gt_center_tn3 = match_gt_center_tn3[:, id_order, :]
            match_valid_tn = match_valid_tn[:, id_order]
            match_ids_n = match_ids_n[id_order]
            if (match_gt_feat_tnd is not None) and (match_gt_feat_ids_n is not None):
                if match_gt_feat_ids_n.numel() == match_gt_feat_tnd.shape[1]:
                    feat_order = torch.argsort(match_gt_feat_ids_n)
                    match_gt_feat_tnd = match_gt_feat_tnd[:, feat_order, :]
                    match_gt_feat_valid_tn = match_gt_feat_valid_tn[:, feat_order]
                    match_gt_feat_ids_n = match_gt_feat_ids_n[feat_order]
                    if match_gt_feat_ids_n.numel() == match_ids_n.numel() and not torch.equal(match_gt_feat_ids_n, match_ids_n):
                        match_gt_feat_tnd = None
                        match_gt_feat_valid_tn = None
                        match_gt_feat_ids_n = None
            elif match_gt_feat_tnd is not None:
                if match_gt_feat_tnd.shape[1] != match_gt_center_tn3.shape[1]:
                    match_gt_feat_tnd = None
                    match_gt_feat_valid_tn = None

        match_result["gt_centers_tn3"] = match_gt_center_tn3
        match_result["gt_valid_tn"] = match_valid_tn
        match_result["gt_ids_n"] = match_ids_n
        match_result["gt_cls_n"] = match_cls_n
        match_result["gt_match_feat_tnd"] = match_gt_feat_tnd
        match_result["gt_match_feat_valid_tn"] = match_gt_feat_valid_tn
        match_result["gt_match_feat_ids_n"] = match_gt_feat_ids_n
        match_result["gt_bev_feat_tnd"] = match_gt_feat_tnd
        match_result["gt_bev_feat_valid_tn"] = match_gt_feat_valid_tn
        match_result["gt_bev_feat_ids_n"] = match_gt_feat_ids_n

        q_count = int(centers_world.shape[1])
        n_inst = int(match_gt_center_tn3.shape[1])
        if (q_count <= 0) or (n_inst <= 0):
            return match_result
        if ((match_query_img_feat_tqd is None) or (match_gt_feat_tnd is None)):
            return match_result

        if match_gt_feat_valid_tn is None:
            match_gt_feat_valid_tn = torch.ones(
                match_gt_feat_tnd.shape[:2], device=centers_world.device, dtype=torch.bool
            )
            match_result["gt_bev_feat_valid_tn"] = match_gt_feat_valid_tn

        active_center = match_valid_tn.any(dim=0)
        active_feat = match_gt_feat_valid_tn.any(dim=0)
        active_inst_idx = torch.nonzero(active_center & active_feat, as_tuple=False).squeeze(1)
        if active_inst_idx.numel() <= 0:
            return match_result

        if active_inst_idx.numel() != n_inst:
            match_gt_center_tn3 = match_gt_center_tn3[:, active_inst_idx, :]
            match_valid_tn = match_valid_tn[:, active_inst_idx]
            match_gt_feat_tnd = match_gt_feat_tnd[:, active_inst_idx, :]
            match_gt_feat_valid_tn = match_gt_feat_valid_tn[:, active_inst_idx]
            if match_ids_n is not None:
                match_ids_n = match_ids_n[active_inst_idx]
            if match_cls_n is not None and int(match_cls_n.numel()) == n_inst:
                match_cls_n = match_cls_n[active_inst_idx]
            if match_cls_valid_n is not None and int(match_cls_valid_n.numel()) == n_inst:
                match_cls_valid_n = match_cls_valid_n[active_inst_idx]
            if (match_gt_feat_ids_n is not None) and (match_gt_feat_ids_n.numel() == n_inst):
                match_gt_feat_ids_n = match_gt_feat_ids_n[active_inst_idx]
            n_inst = int(match_gt_center_tn3.shape[1])
            match_result["gt_centers_tn3"] = match_gt_center_tn3
            match_result["gt_valid_tn"] = match_valid_tn
            match_result["gt_ids_n"] = match_ids_n
            match_result["gt_cls_n"] = match_cls_n
            match_result["gt_match_feat_tnd"] = match_gt_feat_tnd
            match_result["gt_match_feat_valid_tn"] = match_gt_feat_valid_tn
            match_result["gt_match_feat_ids_n"] = match_gt_feat_ids_n
            match_result["gt_bev_feat_tnd"] = match_gt_feat_tnd
            match_result["gt_bev_feat_valid_tn"] = match_gt_feat_valid_tn
            match_result["gt_bev_feat_ids_n"] = match_gt_feat_ids_n

        t_match = min(
            int(match_query_img_feat_tqd.shape[0]),
            int(match_gt_feat_tnd.shape[0]),
            int(match_gt_feat_valid_tn.shape[0]),
        )
        if t_match <= 0:
            return match_result

        q_feat_tqd = F.normalize(match_query_img_feat_tqd[-t_match:].to(torch.float32), dim=-1, eps=1e-6)
        gt_feat_tnd = F.normalize(match_gt_feat_tnd[:t_match].to(torch.float32), dim=-1, eps=1e-6)
        gt_feat_valid_tn = match_gt_feat_valid_tn[:t_match].to(device=centers_world.device, dtype=torch.bool)

        sim_tqn = torch.einsum("tqd,tnd->tqn", q_feat_tqd, gt_feat_tnd).clamp(-1.0, 1.0)
        valid_t1n = gt_feat_valid_tn.to(torch.float32)[:, None, :]
        sim_sum_qn = (sim_tqn * valid_t1n).sum(dim=0)
        valid_cnt_qn = valid_t1n.sum(dim=0)
        valid_pair_qn = valid_cnt_qn > 0.0
        if not torch.any(valid_pair_qn):
            return match_result

        sim_qn = sim_sum_qn / valid_cnt_qn.clamp_min(1.0)
        tau = max(1e-6, float(query_soft_assign_temp))
        soft_assign_qn = torch.softmax(sim_qn / tau, dim=1)
        cost_feat_qn = (1.0 - sim_qn).clamp(0.0, 2.0)
        cost_soft_qn = (-torch.log(soft_assign_qn.clamp_min(1e-8))).clamp(0.0, 30.0)

        sim_cost_w = float(query_sim_cost_weight)
        cost_feat_contrib_qn = sim_cost_w * cost_feat_qn
        # Legacy costs are kept for diagnostics compatibility but are never added to Hungarian cost.
        cost_soft_contrib_qn = torch.zeros_like(cost_feat_qn)
        cls_cost_qn = None
        cls_contrib_qn = torch.zeros_like(cost_feat_qn)
        cost_qn = cost_feat_contrib_qn
        cls_cost_w = float(query_cls_cost_weight)
        if (
            cls_cost_w > 0.0
            and torch.is_tensor(query_cls_logits_qc)
            and query_cls_logits_qc.dim() == 2
            and int(query_cls_logits_qc.shape[0]) == q_count
            and torch.is_tensor(match_cls_n)
            and int(match_cls_n.numel()) == n_inst
        ):
            num_cls = int(query_cls_logits_qc.shape[1])
            gt_cls = match_cls_n.to(device=centers_world.device, dtype=torch.long)
            cls_valid_n = (gt_cls >= 0) & (gt_cls < num_cls)
            if torch.is_tensor(match_cls_valid_n) and int(match_cls_valid_n.numel()) == n_inst:
                cls_valid_n = cls_valid_n & match_cls_valid_n.to(device=centers_world.device, dtype=torch.bool)
            if bool(cls_valid_n.any().item()):
                log_prob_qc = F.log_softmax(query_cls_logits_qc.to(torch.float32), dim=-1)
                cls_cost_qn = (-log_prob_qc.index_select(1, gt_cls.clamp(0, num_cls - 1))).clamp(0.0, 30.0)
                cls_valid_1n = cls_valid_n.to(dtype=cls_cost_qn.dtype)[None, :]
                cls_cost_qn = cls_cost_qn * cls_valid_1n
                cls_contrib_qn = cls_cost_w * cls_cost_qn
                cost_qn = cost_qn + cls_contrib_qn

        center_cost_qn = None
        center_contrib_qn = torch.zeros_like(cost_feat_qn)
        temporal_offset_cost_qn = None
        temporal_offset_contrib_qn = torch.zeros_like(cost_feat_qn)
        center_frame_idx = None
        center_cost_w = float(getattr(self, "query_center_match_cost_weight", 0.0))
        temporal_offset_cost_w = float(query_temporal_offset_match_cost_weight)
        t_center = min(
            int(centers_world.shape[0]),
            int(match_gt_center_tn3.shape[0]),
            int(match_valid_tn.shape[0]),
        )
        temporal_cost_idx = self._resolve_temporal_cost_frame_indices(
            query_temporal_cost_frame_indices,
            t_center,
        )
        if t_center > 0:
            if temporal_cost_idx is not None:
                pred_center_tq3 = centers_world.index_select(
                    0,
                    torch.as_tensor(temporal_cost_idx, device=centers_world.device, dtype=torch.long),
                ).to(torch.float32)
                gt_center_tn3 = match_gt_center_tn3.index_select(
                    0,
                    torch.as_tensor(temporal_cost_idx, device=centers_world.device, dtype=torch.long),
                ).to(torch.float32)
                gt_center_valid_tn = match_valid_tn.index_select(
                    0,
                    torch.as_tensor(temporal_cost_idx, device=centers_world.device, dtype=torch.long),
                ).to(device=centers_world.device, dtype=torch.bool)
                center_l1_tqn = torch.abs(
                    pred_center_tq3[:, :, None, :] - gt_center_tn3[:, None, :, :]
                ).sum(dim=-1)
                center_valid_t1n = gt_center_valid_tn.to(torch.float32)[:, None, :]
                center_num_qn = (center_l1_tqn * center_valid_t1n).sum(dim=0)
                center_den_qn = center_valid_t1n.sum(dim=0)
                center_valid_qn = center_den_qn > 0.0
                bev_diag_m = float(np.hypot(float(self.spatial_extent3d[0]), float(self.spatial_extent3d[1])))
                bev_diag_m = max(1e-6, bev_diag_m)
                center_cost_qn = (center_num_qn / center_den_qn.clamp_min(1.0) / bev_diag_m).clamp(0.0, 2.0)
                if center_cost_w > 0.0:
                    center_contrib_qn = center_cost_w * center_cost_qn * center_valid_qn.to(cost_qn.dtype)
                    cost_qn = cost_qn + center_contrib_qn
                if len(temporal_cost_idx) >= 2:
                    pred_delta_tq2 = pred_center_tq3[1:, :, :2] - pred_center_tq3[:-1, :, :2]
                    gt_delta_tn2 = gt_center_tn3[1:, :, :2] - gt_center_tn3[:-1, :, :2]
                    gt_delta_valid_tn = gt_center_valid_tn[1:] & gt_center_valid_tn[:-1]
                    traj_max_xy = centers_world.new_tensor(
                        self.query_traj_residual_max_m, dtype=torch.float32
                    ).view(1, 1, 1, 2).clamp_min(1e-6)
                    delta_l1_tqn = (
                        torch.abs(pred_delta_tq2[:, :, None, :] - gt_delta_tn2[:, None, :, :]) / traj_max_xy
                    ).sum(dim=-1)
                    delta_valid_t1n = gt_delta_valid_tn.to(torch.float32)[:, None, :]
                    delta_num_qn = (delta_l1_tqn * delta_valid_t1n).sum(dim=0)
                    delta_den_qn = delta_valid_t1n.sum(dim=0)
                    delta_valid_qn = delta_den_qn > 0.0
                    temporal_offset_cost_qn = (
                        delta_num_qn / delta_den_qn.clamp_min(1.0)
                    ).clamp(0.0, 2.0)
                    if temporal_offset_cost_w > 0.0:
                        temporal_offset_contrib_qn = (
                            temporal_offset_cost_w
                            * temporal_offset_cost_qn
                            * delta_valid_qn.to(cost_qn.dtype)
                        )
                        cost_qn = cost_qn + temporal_offset_contrib_qn
            elif query_center_match_frame_idx is not None:
                preferred_center_idx = int(query_center_match_frame_idx)
                preferred_center_idx = max(0, min(t_center - 1, preferred_center_idx))
                center_frame_idx = preferred_center_idx
                pred_center_q3 = centers_world[center_frame_idx].to(torch.float32)
                gt_center_n3 = match_gt_center_tn3[center_frame_idx].to(torch.float32)
                gt_center_valid_n = match_valid_tn[center_frame_idx].to(
                    device=centers_world.device, dtype=torch.bool
                )
                center_l1_qn = torch.abs(
                    pred_center_q3[:, None, :] - gt_center_n3[None, :, :]
                ).sum(dim=-1)
                bev_diag_m = float(np.hypot(float(self.spatial_extent3d[0]), float(self.spatial_extent3d[1])))
                bev_diag_m = max(1e-6, bev_diag_m)
                center_cost_qn = (center_l1_qn / bev_diag_m).clamp(0.0, 2.0)
                center_pair_valid_qn = gt_center_valid_n[None, :].expand_as(center_cost_qn)
                if center_cost_w > 0.0:
                    center_contrib_qn = center_cost_w * center_cost_qn * center_pair_valid_qn.to(cost_qn.dtype)
                    cost_qn = cost_qn + center_contrib_qn

        bev_dice_cost_qn = None
        bev_dice_contrib_qn = torch.zeros_like(cost_feat_qn)
        bev_dice_cost_w = float(query_bev_dice_cost_weight)
        if (
            bev_dice_cost_w > 0.0
            and torch.is_tensor(gt_instance_occ3d_txyz)
            and torch.is_tensor(match_ids_n)
            and match_ids_n.numel() > 0
        ):
            if temporal_cost_idx is not None:
                frame_idx_t = torch.as_tensor(
                    temporal_cost_idx,
                    device=centers_world.device,
                    dtype=torch.long,
                )
                bev_centers_tq3 = centers_world.index_select(0, frame_idx_t)
                bev_valid_tn = match_valid_tn.index_select(0, frame_idx_t)
                gt_occ_cost_txyz = gt_instance_occ3d_txyz.index_select(0, frame_idx_t.to(gt_instance_occ3d_txyz.device))
                bev_sigmas_tq3 = (
                    query_sigmas_world_tq3.index_select(0, frame_idx_t)
                    if torch.is_tensor(query_sigmas_world_tq3)
                    and tuple(query_sigmas_world_tq3.shape[:2]) == tuple(centers_world.shape[:2])
                    else None
                )
                bev_mix_centers_tqg3 = (
                    query_mixture_centers_world_tqg3.index_select(0, frame_idx_t)
                    if torch.is_tensor(query_mixture_centers_world_tqg3)
                    and int(query_mixture_centers_world_tqg3.shape[0]) == int(centers_world.shape[0])
                    else None
                )
                bev_mix_sigmas_tqg3 = (
                    query_mixture_sigmas_world_tqg3.index_select(0, frame_idx_t)
                    if torch.is_tensor(query_mixture_sigmas_world_tqg3)
                    and int(query_mixture_sigmas_world_tqg3.shape[0]) == int(centers_world.shape[0])
                    else None
                )
                bev_mix_yaw_tqg = (
                    query_mixture_yaw_tqg.index_select(0, frame_idx_t)
                    if torch.is_tensor(query_mixture_yaw_tqg)
                    and int(query_mixture_yaw_tqg.shape[0]) == int(centers_world.shape[0])
                    else None
                )
                bev_mix_weights_tqg = (
                    query_mixture_weights_tqg.index_select(0, frame_idx_t)
                    if torch.is_tensor(query_mixture_weights_tqg)
                    and int(query_mixture_weights_tqg.shape[0]) == int(centers_world.shape[0])
                    else None
                )
            else:
                bev_centers_tq3 = centers_world
                bev_valid_tn = match_valid_tn
                gt_occ_cost_txyz = gt_instance_occ3d_txyz
                bev_sigmas_tq3 = query_sigmas_world_tq3
                bev_mix_centers_tqg3 = query_mixture_centers_world_tqg3
                bev_mix_sigmas_tqg3 = query_mixture_sigmas_world_tqg3
                bev_mix_yaw_tqg = query_mixture_yaw_tqg
                bev_mix_weights_tqg = query_mixture_weights_tqg
            bev_dice_cost_qn, bev_dice_valid_qn = self._compute_query_gt_bev_dice_cost_qn(
                centers_world_tq3=bev_centers_tq3,
                sigmas_world_tq3=bev_sigmas_tq3,
                gt_instance_occ3d_txyz=gt_occ_cost_txyz,
                gt_ids_n=match_ids_n,
                gt_valid_tn=bev_valid_tn,
                mixture_centers_world_tqg3=bev_mix_centers_tqg3,
                mixture_sigmas_world_tqg3=bev_mix_sigmas_tqg3,
                mixture_yaw_tqg=bev_mix_yaw_tqg,
                mixture_weights_tqg=bev_mix_weights_tqg,
            )
            if (
                torch.is_tensor(bev_dice_cost_qn)
                and bev_dice_cost_qn.shape == cost_qn.shape
                and torch.is_tensor(bev_dice_valid_qn)
                and bev_dice_valid_qn.shape == cost_qn.shape
            ):
                bev_dice_contrib_qn = (
                    bev_dice_cost_w
                    * bev_dice_cost_qn.to(cost_qn.dtype)
                    * bev_dice_valid_qn.to(cost_qn.dtype)
                )
                cost_qn = cost_qn + bev_dice_contrib_qn
            else:
                bev_dice_cost_qn = None
                bev_dice_contrib_qn = torch.zeros_like(cost_feat_qn)
        attn_iou_cost_qn = None
        attn_iou_contrib_qn = torch.zeros_like(cost_feat_qn)
        attn_iou_cost_w = float(query_attn_match_cost_weight)
        attn_match_metric = str(query_attn_match_metric).lower()
        if (
            attn_iou_cost_w > 0.0
            and attn_match_metric in ("soft_iou", "inside_log")
            and torch.is_tensor(query_attn_weights_tqnhw)
            and isinstance(gt_attn_targets, dict)
        ):
            if attn_match_metric == "inside_log":
                attn_iou_cost_qn, attn_iou_valid_qn = self._compute_query_attn_inside_log_cost_qn(
                    query_attn_weights_tqnhw=query_attn_weights_tqnhw,
                    gt_attn_targets=gt_attn_targets,
                    eps=float(query_attn_match_eps),
                )
            else:
                attn_iou_cost_qn, attn_iou_valid_qn = self._compute_query_attn_soft_iou_cost_qn(
                    query_attn_weights_tqnhw=query_attn_weights_tqnhw,
                    gt_attn_targets=gt_attn_targets,
                    pred_norm=query_attn_match_pred_norm,
                    eps=float(query_attn_match_eps),
                )
            if (
                torch.is_tensor(attn_iou_cost_qn)
                and attn_iou_cost_qn.shape == cost_qn.shape
                and torch.is_tensor(attn_iou_valid_qn)
                and attn_iou_valid_qn.shape == cost_qn.shape
            ):
                attn_iou_contrib_qn = (
                    attn_iou_cost_w
                    * attn_iou_cost_qn.to(cost_qn.dtype)
                    * attn_iou_valid_qn.to(cost_qn.dtype)
                )
                cost_qn = cost_qn + attn_iou_contrib_qn
            else:
                attn_iou_cost_qn = None
                attn_iou_contrib_qn = torch.zeros_like(cost_feat_qn)

        valid_inst_n = valid_pair_qn.any(dim=0)
        valid_inst_idx = torch.nonzero(valid_inst_n, as_tuple=False).squeeze(1)
        if valid_inst_idx.numel() <= 0:
            return match_result

        cost_qn = cost_qn.index_select(1, valid_inst_idx)
        sim_qn = sim_qn.index_select(1, valid_inst_idx)
        soft_assign_qn = soft_assign_qn.index_select(1, valid_inst_idx)
        cost_feat_qn = cost_feat_qn.index_select(1, valid_inst_idx)
        cost_feat_contrib_qn = cost_feat_contrib_qn.index_select(1, valid_inst_idx)
        cost_soft_qn = cost_soft_qn.index_select(1, valid_inst_idx)
        cost_soft_contrib_qn = cost_soft_contrib_qn.index_select(1, valid_inst_idx)
        if cls_cost_qn is not None:
            cls_cost_qn = cls_cost_qn.index_select(1, valid_inst_idx)
        if cls_contrib_qn is not None:
            cls_contrib_qn = cls_contrib_qn.index_select(1, valid_inst_idx)
        if center_cost_qn is not None:
            center_cost_qn = center_cost_qn.index_select(1, valid_inst_idx)
        if center_contrib_qn is not None:
            center_contrib_qn = center_contrib_qn.index_select(1, valid_inst_idx)
        if temporal_offset_cost_qn is not None:
            temporal_offset_cost_qn = temporal_offset_cost_qn.index_select(1, valid_inst_idx)
        if temporal_offset_contrib_qn is not None:
            temporal_offset_contrib_qn = temporal_offset_contrib_qn.index_select(1, valid_inst_idx)
        if bev_dice_cost_qn is not None:
            bev_dice_cost_qn = bev_dice_cost_qn.index_select(1, valid_inst_idx)
        if bev_dice_contrib_qn is not None:
            bev_dice_contrib_qn = bev_dice_contrib_qn.index_select(1, valid_inst_idx)
        if attn_iou_cost_qn is not None:
            attn_iou_cost_qn = attn_iou_cost_qn.index_select(1, valid_inst_idx)
        if attn_iou_contrib_qn is not None:
            attn_iou_contrib_qn = attn_iou_contrib_qn.index_select(1, valid_inst_idx)

        if match_gt_center_tn3.shape[1] != valid_inst_idx.numel():
            match_gt_center_tn3 = match_gt_center_tn3.index_select(1, valid_inst_idx)
            match_valid_tn = match_valid_tn.index_select(1, valid_inst_idx)
            match_gt_feat_tnd = match_gt_feat_tnd.index_select(1, valid_inst_idx)
            match_gt_feat_valid_tn = match_gt_feat_valid_tn.index_select(1, valid_inst_idx)
            if match_ids_n is not None and match_ids_n.numel() == n_inst:
                match_ids_n = match_ids_n.index_select(0, valid_inst_idx)
            if match_cls_n is not None and match_cls_n.numel() == n_inst:
                match_cls_n = match_cls_n.index_select(0, valid_inst_idx)
            if match_cls_valid_n is not None and match_cls_valid_n.numel() == n_inst:
                match_cls_valid_n = match_cls_valid_n.index_select(0, valid_inst_idx)
            if match_gt_feat_ids_n is not None and match_gt_feat_ids_n.numel() == n_inst:
                match_gt_feat_ids_n = match_gt_feat_ids_n.index_select(0, valid_inst_idx)

        match_result["sim_qn"] = sim_qn
        match_result["soft_assign_qn"] = soft_assign_qn
        if center_frame_idx is not None:
            match_result["center_frame_idx"] = centers_world.new_tensor(float(center_frame_idx))
        match_result["cost_qn"] = cost_qn
        match_result["cost_feat_qn"] = cost_feat_qn
        match_result["cost_feat_contrib_qn"] = cost_feat_contrib_qn
        match_result["cost_soft_qn"] = cost_soft_qn
        match_result["cost_soft_contrib_qn"] = cost_soft_contrib_qn
        match_result["cost_cls_qn"] = cls_cost_qn
        match_result["cost_cls_contrib_qn"] = cls_contrib_qn
        match_result["cost_center_qn"] = center_cost_qn
        match_result["cost_center_contrib_qn"] = center_contrib_qn
        match_result["cost_temporal_offset_qn"] = temporal_offset_cost_qn
        match_result["cost_temporal_offset_contrib_qn"] = temporal_offset_contrib_qn
        match_result["cost_bev_dice_qn"] = bev_dice_cost_qn
        match_result["cost_bev_dice_contrib_qn"] = bev_dice_contrib_qn
        match_result["cost_attn_iou_qn"] = attn_iou_cost_qn
        match_result["cost_attn_iou_contrib_qn"] = attn_iou_contrib_qn
        match_result["gt_centers_tn3"] = match_gt_center_tn3
        match_result["gt_valid_tn"] = match_valid_tn
        match_result["gt_ids_n"] = match_ids_n
        match_result["gt_cls_n"] = match_cls_n
        match_result["gt_match_feat_tnd"] = match_gt_feat_tnd
        match_result["gt_match_feat_valid_tn"] = match_gt_feat_valid_tn
        match_result["gt_match_feat_ids_n"] = match_gt_feat_ids_n
        match_result["gt_bev_feat_tnd"] = match_gt_feat_tnd
        match_result["gt_bev_feat_valid_tn"] = match_gt_feat_valid_tn
        match_result["gt_bev_feat_ids_n"] = match_gt_feat_ids_n

        gate_radius_m = float(getattr(self, "query_match_center_gate_radius_m", 0.0))
        recruit_radius_m = float(getattr(self, "query_recruit_max_radius_m", 0.0))
        gate_blocked_qn = None
        min_dist_qn = None
        if gate_radius_m > 0.0 or recruit_radius_m > 0.0:
            if temporal_cost_idx is not None:
                distance_frame_indices = temporal_cost_idx
            elif center_frame_idx is not None:
                distance_frame_indices = [center_frame_idx]
            else:
                distance_frame_indices = list(range(t_center))
            frame_idx_t = torch.as_tensor(
                distance_frame_indices,
                device=centers_world.device,
                dtype=torch.long,
            )
            match_result["recruit_frame_indices"] = frame_idx_t
            if frame_idx_t.numel() > 0:
                query_xy_tq2 = centers_world.index_select(0, frame_idx_t)[..., :2].to(torch.float32)
                gt_xy_tn2 = match_gt_center_tn3.index_select(0, frame_idx_t)[..., :2].to(torch.float32)
                gt_valid_tn = match_valid_tn.index_select(0, frame_idx_t).to(torch.bool)
                dist_tqn = torch.linalg.vector_norm(
                    query_xy_tq2[:, :, None, :] - gt_xy_tn2[:, None, :, :],
                    dim=-1,
                )
                dist_tqn = torch.where(
                    gt_valid_tn[:, None, :],
                    dist_tqn,
                    torch.full_like(dist_tqn, float("inf")),
                )
                min_dist_qn = dist_tqn.min(dim=0).values
                if gate_radius_m > 0.0:
                    gate_blocked_qn = min_dist_qn > gate_radius_m
                    cost_qn = cost_qn + gate_blocked_qn.to(cost_qn.dtype) * 1e4
                    match_result["cost_qn"] = cost_qn

        row_ind, col_ind = _solve_unique_assignment(cost_qn.detach().cpu().numpy())
        if row_ind.size == 0:
            return match_result

        match_query_idx = torch.as_tensor(row_ind, device=centers_world.device, dtype=torch.long)
        match_inst_idx = torch.as_tensor(col_ind, device=centers_world.device, dtype=torch.long)
        inst_order = torch.argsort(match_inst_idx)
        match_query_idx = match_query_idx[inst_order]
        match_inst_idx = match_inst_idx[inst_order]
        if gate_blocked_qn is not None:
            keep = ~gate_blocked_qn[match_query_idx, match_inst_idx]
            match_query_idx = match_query_idx[keep]
            match_inst_idx = match_inst_idx[keep]

        if recruit_radius_m > 0.0 and torch.is_tensor(min_dist_qn):
            free_dist_qn = min_dist_qn.clone()
            if match_query_idx.numel() > 0:
                free_dist_qn[match_query_idx, :] = float("inf")
                free_dist_qn[:, match_inst_idx] = float("inf")
            free_dist_qn = torch.where(
                free_dist_qn <= recruit_radius_m,
                free_dist_qn,
                torch.full_like(free_dist_qn, float("inf")),
            )
            recruit_query_idx = []
            recruit_inst_idx = []
            n_inst_final = int(free_dist_qn.shape[1])
            while free_dist_qn.numel() > 0:
                flat_idx = int(torch.argmin(free_dist_qn).item())
                query_idx, inst_idx = divmod(flat_idx, n_inst_final)
                if not bool(torch.isfinite(free_dist_qn[query_idx, inst_idx]).item()):
                    break
                recruit_query_idx.append(query_idx)
                recruit_inst_idx.append(inst_idx)
                free_dist_qn[query_idx, :] = float("inf")
                free_dist_qn[:, inst_idx] = float("inf")
            if recruit_query_idx:
                match_result["recruit_query_idx"] = torch.as_tensor(
                    recruit_query_idx,
                    device=centers_world.device,
                    dtype=torch.long,
                )
                match_result["recruit_inst_idx"] = torch.as_tensor(
                    recruit_inst_idx,
                    device=centers_world.device,
                    dtype=torch.long,
                )

        match_result["matched_query_idx"] = match_query_idx
        match_result["matched_inst_idx"] = match_inst_idx
        if match_query_idx.numel() <= 0:
            return match_result
        if bool(getattr(self, "query_present_only", False)):
            query_feat_for_match = match_query_img_feat_tqd[:t_match]
        else:
            query_feat_for_match = match_query_img_feat_tqd[-t_match:]
        match_result["matched_query_feat_tkd"] = query_feat_for_match.index_select(1, match_query_idx)
        match_result["matched_gt_feat_tkd"] = match_gt_feat_tnd.index_select(1, match_inst_idx)
        match_result["matched_gt_feat_valid_tk"] = match_gt_feat_valid_tn.index_select(1, match_inst_idx)
        return match_result

    def _compute_query_inst_center_match_loss(
        self,
        centers_world: torch.Tensor,
        gt_inst_center_world_tn3: torch.Tensor = None,
        gt_inst_center_valid_tn: torch.Tensor = None,
        gt_inst_ids_n: torch.Tensor = None,
        gt_inst_cls_n: torch.Tensor = None,
        gt_inst_cls_valid_n: torch.Tensor = None,
        query_img_feat_tqd: torch.Tensor = None,
        query_cls_logits_qc: torch.Tensor = None,
        gt_inst_bev_feat_tnd: torch.Tensor = None,
        gt_inst_bev_feat_valid_tn: torch.Tensor = None,
        gt_inst_bev_feat_ids_n: torch.Tensor = None,
        query_sim_cost_weight: float = 1.0,
        query_cls_cost_weight: float = 0.0,
        query_soft_assign_temp: float = 0.10,
        query_soft_assign_cost_weight: float = 0.0,
        loss_weight: float = 0.1,
        return_match: bool = False,
    ):
        match_result = self._match_queries_to_gt_instances(
            centers_world=centers_world,
            gt_inst_center_world_tn3=gt_inst_center_world_tn3,
            gt_inst_center_valid_tn=gt_inst_center_valid_tn,
            gt_inst_ids_n=gt_inst_ids_n,
            gt_inst_cls_n=gt_inst_cls_n,
            gt_inst_cls_valid_n=gt_inst_cls_valid_n,
            query_img_feat_tqd=query_img_feat_tqd,
            query_cls_logits_qc=query_cls_logits_qc,
            gt_inst_bev_feat_tnd=gt_inst_bev_feat_tnd,
            gt_inst_bev_feat_valid_tn=gt_inst_bev_feat_valid_tn,
            gt_inst_bev_feat_ids_n=gt_inst_bev_feat_ids_n,
            query_sim_cost_weight=query_sim_cost_weight,
            query_cls_cost_weight=query_cls_cost_weight,
            query_soft_assign_temp=query_soft_assign_temp,
            query_soft_assign_cost_weight=query_soft_assign_cost_weight,
        )

        match_query_idx = match_result.get("matched_query_idx", None)
        match_inst_idx = match_result.get("matched_inst_idx", None)
        match_gt_center_tn3 = match_result.get("gt_centers_tn3", None)
        match_valid_tn = match_result.get("gt_valid_tn", None)
        if (
            (not torch.is_tensor(match_query_idx))
            or (not torch.is_tensor(match_inst_idx))
            or match_query_idx.numel() <= 0
            or match_query_idx.numel() != match_inst_idx.numel()
            or (not torch.is_tensor(match_gt_center_tn3))
            or (not torch.is_tensor(match_valid_tn))
        ):
            if return_match:
                return None, match_result
            return None

        matched_pred_tk3 = centers_world[:, match_query_idx, :]
        matched_gt_tk3 = match_gt_center_tn3[:, match_inst_idx, :]
        matched_valid_tk = match_valid_tn[:, match_inst_idx].to(dtype=centers_world.dtype)
        # Axis-wise L1 center distance loss on matched query/GT pairs.
        err_tk = torch.abs(matched_pred_tk3 - matched_gt_tk3).sum(dim=-1)
        inst_center_match_loss = (err_tk * matched_valid_tk).sum() / matched_valid_tk.sum().clamp_min(1.0)
        inst_center_match_loss = inst_center_match_loss * float(loss_weight)
        if return_match:
            return inst_center_match_loss, match_result
        return inst_center_match_loss
