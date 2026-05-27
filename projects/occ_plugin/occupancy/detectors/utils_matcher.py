import torch
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
            "center_frame_idx": None,
            "cost_qn": None,
            "cost_cls_qn": None,
            "cost_cls_contrib_qn": None,
            "cost_center_qn": None,
            "cost_center_contrib_qn": None,
            "cost_temporal_offset_qn": None,
            "cost_temporal_offset_contrib_qn": None,
            "cost_attn_iou_qn": None,
            "cost_attn_iou_contrib_qn": None,
            "gt_centers_tn3": None,
            "gt_valid_tn": None,
            "gt_ids_n": None,
            "gt_cls_n": None,
        }

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
        gt_inst_center_world_tn3: torch.Tensor = None,
        gt_inst_center_valid_tn: torch.Tensor = None,
        gt_inst_ids_n: torch.Tensor = None,
        gt_inst_cls_n: torch.Tensor = None,
        gt_inst_cls_valid_n: torch.Tensor = None,
        query_cls_logits_qc: torch.Tensor = None,
        query_cls_match_cost_weight: float = 0.0,
        query_attn_weights_tqnhw: torch.Tensor = None,
        gt_attn_targets: dict = None,
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

        if (match_ids_n is not None) and (match_ids_n.numel() == match_gt_center_tn3.shape[1]):
            id_order = torch.argsort(match_ids_n)
            match_gt_center_tn3 = match_gt_center_tn3[:, id_order, :]
            match_valid_tn = match_valid_tn[:, id_order]
            match_ids_n = match_ids_n[id_order]
            if match_cls_n is not None and int(match_cls_n.numel()) == int(id_order.numel()):
                match_cls_n = match_cls_n[id_order]
            if match_cls_valid_n is not None and int(match_cls_valid_n.numel()) == int(id_order.numel()):
                match_cls_valid_n = match_cls_valid_n[id_order]

        match_result["gt_centers_tn3"] = match_gt_center_tn3
        match_result["gt_valid_tn"] = match_valid_tn
        match_result["gt_ids_n"] = match_ids_n
        match_result["gt_cls_n"] = match_cls_n

        q_count = int(centers_world.shape[1])
        n_inst = int(match_gt_center_tn3.shape[1])
        if (q_count <= 0) or (n_inst <= 0):
            return match_result

        active_center = match_valid_tn.any(dim=0)
        active_inst_idx = torch.nonzero(active_center, as_tuple=False).squeeze(1)
        if active_inst_idx.numel() <= 0:
            return match_result

        if active_inst_idx.numel() != n_inst:
            match_gt_center_tn3 = match_gt_center_tn3[:, active_inst_idx, :]
            match_valid_tn = match_valid_tn[:, active_inst_idx]
            if match_ids_n is not None:
                match_ids_n = match_ids_n[active_inst_idx]
            if match_cls_n is not None and int(match_cls_n.numel()) == n_inst:
                match_cls_n = match_cls_n[active_inst_idx]
            if match_cls_valid_n is not None and int(match_cls_valid_n.numel()) == n_inst:
                match_cls_valid_n = match_cls_valid_n[active_inst_idx]
            n_inst = int(match_gt_center_tn3.shape[1])
            match_result["gt_centers_tn3"] = match_gt_center_tn3
            match_result["gt_valid_tn"] = match_valid_tn
            match_result["gt_ids_n"] = match_ids_n
            match_result["gt_cls_n"] = match_cls_n

        cost_qn = centers_world.new_zeros((q_count, n_inst), dtype=torch.float32)

        cls_cost_qn = None
        cls_contrib_qn = torch.zeros_like(cost_qn)
        cls_cost_w = float(query_cls_match_cost_weight)
        if (
            cls_cost_w > 0.0
            and torch.is_tensor(query_cls_logits_qc)
            and query_cls_logits_qc.dim() == 2
            and int(query_cls_logits_qc.shape[0]) == q_count
            and torch.is_tensor(match_cls_n)
            and int(match_cls_n.numel()) == n_inst
        ):
            cls_count = int(query_cls_logits_qc.shape[1])
            gt_cls = match_cls_n.to(device=centers_world.device, dtype=torch.long)
            cls_valid_n = (gt_cls >= 0) & (gt_cls < cls_count)
            if torch.is_tensor(match_cls_valid_n) and int(match_cls_valid_n.numel()) == n_inst:
                cls_valid_n = cls_valid_n & match_cls_valid_n.to(device=centers_world.device, dtype=torch.bool)
            if bool(cls_valid_n.any().item()):
                log_prob_qc = torch.log_softmax(
                    query_cls_logits_qc.to(device=centers_world.device, dtype=torch.float32),
                    dim=-1,
                )
                gather_cls_n = gt_cls.clamp(0, max(0, cls_count - 1))
                cls_cost_qn = (-log_prob_qc[:, gather_cls_n]).clamp(0.0, 30.0)
                cls_contrib_qn = cls_cost_w * cls_cost_qn * cls_valid_n.to(cost_qn.dtype)[None, :]
                cost_qn = cost_qn + cls_contrib_qn

        center_cost_qn = None
        center_contrib_qn = torch.zeros_like(cost_qn)
        temporal_offset_cost_qn = None
        temporal_offset_contrib_qn = torch.zeros_like(cost_qn)
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
                    cost_qn = cost_qn.masked_fill(~center_valid_qn, cost_qn.new_tensor(1e6))
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
                    cost_qn = cost_qn.masked_fill(~center_pair_valid_qn, cost_qn.new_tensor(1e6))

        attn_iou_cost_qn = None
        attn_iou_contrib_qn = torch.zeros_like(cost_qn)
        attn_iou_cost_w = float(query_attn_match_cost_weight)
        if (
            attn_iou_cost_w > 0.0
            and str(query_attn_match_metric).lower() == "soft_iou"
            and torch.is_tensor(query_attn_weights_tqnhw)
            and isinstance(gt_attn_targets, dict)
        ):
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
                attn_iou_contrib_qn = torch.zeros_like(cost_qn)

        if center_frame_idx is not None:
            match_result["center_frame_idx"] = centers_world.new_tensor(float(center_frame_idx))
        match_result["cost_qn"] = cost_qn
        match_result["cost_cls_qn"] = cls_cost_qn
        match_result["cost_cls_contrib_qn"] = cls_contrib_qn
        match_result["cost_center_qn"] = center_cost_qn
        match_result["cost_center_contrib_qn"] = center_contrib_qn
        match_result["cost_temporal_offset_qn"] = temporal_offset_cost_qn
        match_result["cost_temporal_offset_contrib_qn"] = temporal_offset_contrib_qn
        match_result["cost_attn_iou_qn"] = attn_iou_cost_qn
        match_result["cost_attn_iou_contrib_qn"] = attn_iou_contrib_qn
        match_result["gt_centers_tn3"] = match_gt_center_tn3
        match_result["gt_valid_tn"] = match_valid_tn
        match_result["gt_ids_n"] = match_ids_n
        match_result["gt_cls_n"] = match_cls_n

        row_ind, col_ind = _solve_unique_assignment(cost_qn.detach().cpu().numpy())
        if row_ind.size == 0:
            return match_result

        match_query_idx = torch.as_tensor(row_ind, device=centers_world.device, dtype=torch.long)
        match_inst_idx = torch.as_tensor(col_ind, device=centers_world.device, dtype=torch.long)
        inst_order = torch.argsort(match_inst_idx)
        match_query_idx = match_query_idx[inst_order]
        match_inst_idx = match_inst_idx[inst_order]
        match_result["matched_query_idx"] = match_query_idx
        match_result["matched_inst_idx"] = match_inst_idx
        return match_result

    def _compute_query_inst_center_match_loss(
        self,
        centers_world: torch.Tensor,
        gt_inst_center_world_tn3: torch.Tensor = None,
        gt_inst_center_valid_tn: torch.Tensor = None,
        gt_inst_ids_n: torch.Tensor = None,
        gt_inst_cls_n: torch.Tensor = None,
        gt_inst_cls_valid_n: torch.Tensor = None,
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
