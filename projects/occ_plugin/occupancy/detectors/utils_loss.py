import os

import torch
import torch.nn.functional as F
from projects.occ_plugin.occupancy.dense_heads.voxelizer import quat_to_rotmat_wxyz


class EfficientOCFLossMixin:
    """Loss computation methods for EfficientOCF."""

    @staticmethod
    def _safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        w = weights.to(dtype=values.dtype)
        num = (values * w).sum()
        den = w.sum().clamp_min(float(eps))
        return num / den

    def _align_query_centers_to_present_frame(
        self,
        centers_world_tq3: torch.Tensor,
        future_egomotion: torch.Tensor = None,
        traj_frame_indices_t: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert per-step local-lidar centers into present-frame lidar coordinates."""
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return centers_world_tq3
        t_count = int(centers_world_tq3.shape[0])
        if t_count <= 1:
            return centers_world_tq3


        frame_idx = traj_frame_indices_t.to(torch.long).reshape(-1)
        frame_indices = [int(v) for v in frame_idx[:t_count].detach().cpu().tolist()]

        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[0]
        ego = ego.to(torch.float32)
        ego_t = int(ego.shape[0])

        present_global_idx = int(
            getattr(self, "query_present_global_idx", int(getattr(self, "time_receptive_field", 1) - 1))
        )
        lidar_target_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=frame_indices,
            present_global_idx=present_global_idx,
        )

        aligned = []
        for step in range(t_count):
            tf = lidar_target_to_present[step].to(
                device=centers_world_tq3.device,
                dtype=torch.float32,
            )
            centers_step_q3 = centers_world_tq3[step].to(torch.float32)
            centers_step_q4 = torch.cat(
                [
                    centers_step_q3,
                    torch.ones(
                        (int(centers_step_q3.shape[0]), 1),
                        device=centers_step_q3.device,
                        dtype=centers_step_q3.dtype,
                    ),
                ],
                dim=-1,
            )
            aligned_step_q3 = (tf @ centers_step_q4.t()).t()[:, :3]
            aligned.append(aligned_step_q3)
        return torch.stack(aligned, dim=0).to(dtype=centers_world_tq3.dtype).contiguous()

    def _select_matching_feature_frames(
        self,
        query_feat_tqd: torch.Tensor,
        gt_feat_tnd: torch.Tensor = None,
        gt_valid_tn: torch.Tensor = None,
    ):
        """
        Matching feature frames:
        - query/src: use the full receptive-field history without dropping frames
        """
        if (not torch.is_tensor(query_feat_tqd)) or query_feat_tqd.dim() != 3:
            return None, None, None
        q_sel = query_feat_tqd.contiguous()

        gt_sel = None
        gt_valid_sel = None
        if torch.is_tensor(gt_feat_tnd) and gt_feat_tnd.dim() == 3 and int(gt_feat_tnd.shape[0]) > 0:
            t = min(int(query_feat_tqd.shape[0]), int(gt_feat_tnd.shape[0]))
            gt_sel = gt_feat_tnd[:t].contiguous()
            q_sel = query_feat_tqd[:t].contiguous()
        if torch.is_tensor(gt_valid_tn) and gt_valid_tn.dim() == 2 and int(gt_valid_tn.shape[0]) > 0:
            t = min(int(q_sel.shape[0]), int(gt_valid_tn.shape[0]))
            q_sel = q_sel[:t].contiguous()
            if gt_sel is not None:
                gt_sel = gt_sel[:t].contiguous()
            gt_valid_sel = gt_valid_tn[:t].contiguous()
        return q_sel, gt_sel, gt_valid_sel

    def _compute_query_cls_loss(
        self,
        query_cls_logits_qc: torch.Tensor,
        inst_match_result: dict = None,
        loss_weight: float = 1.0,
        bg_index: int = 0,
        class_weights: torch.Tensor = None,
        centers_world_tq3: torch.Tensor = None,
        nms_radius_m: float = 0.0,
    ) -> dict:
        z = query_cls_logits_qc.sum() * 0.0
        out = {
            "loss_query_cls": z,
            "dbg_query_cls_matched_count": z,
            "dbg_query_cls_bg_count": z,
            "dbg_query_cls_ignored_count": z,
        }
        if (not torch.is_tensor(query_cls_logits_qc)) or query_cls_logits_qc.dim() != 2:
            return out

        Q, C = [int(v) for v in query_cls_logits_qc.shape]
        if Q <= 0 or C <= 1:
            return out
        targets_q = torch.full(
            (Q,),
            fill_value=int(bg_index),
            device=query_cls_logits_qc.device,
            dtype=torch.long,
        )
        ignore_target = -100
        ignored_mask_q = torch.zeros((Q,), device=query_cls_logits_qc.device, dtype=torch.bool)
        matched_mask_q = torch.zeros((Q,), device=query_cls_logits_qc.device, dtype=torch.bool)
        matched_count = 0
        ignored_count = 0
        if isinstance(inst_match_result, dict):
            matched_query_idx = inst_match_result.get("matched_query_idx", None)
            matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
            gt_cls_n = inst_match_result.get("gt_cls_n", None)
            if (
                torch.is_tensor(matched_query_idx)
                and torch.is_tensor(matched_inst_idx)
                and torch.is_tensor(gt_cls_n)
                and matched_query_idx.numel() > 0
                and matched_query_idx.numel() == matched_inst_idx.numel()
            ):
                mq = matched_query_idx.to(device=query_cls_logits_qc.device, dtype=torch.long)
                mi = matched_inst_idx.to(device=query_cls_logits_qc.device, dtype=torch.long)
                gt_cls = gt_cls_n.to(device=query_cls_logits_qc.device, dtype=torch.long)
                keep = (
                    (mq >= 0) & (mq < Q)
                    & (mi >= 0) & (mi < int(gt_cls.numel()))
                )
                if bool(keep.any().item()):
                    mq = mq[keep]
                    mi = mi[keep]
                    cls_keep = gt_cls.index_select(0, mi)
                    valid_cls = (cls_keep >= 0) & (cls_keep < C)
                    invalid_cls = ~valid_cls
                    if bool(invalid_cls.any().item()):
                        ignored_mask_q[mq[invalid_cls]] = True
                        ignored_count = int(ignored_mask_q.sum().item())
                    if bool(valid_cls.any().item()):
                        mq = mq[valid_cls]
                        cls_keep = cls_keep[valid_cls]
                        targets_q[mq] = cls_keep
                        matched_mask_q[mq] = True
                        matched_count = int(mq.numel())
        targets_q = torch.where(
            ignored_mask_q,
            torch.full_like(targets_q, fill_value=ignore_target),
            targets_q,
        )

        cls_scores_qc = torch.softmax(query_cls_logits_qc.to(torch.float32), dim=-1)
        bg_prob_q = cls_scores_qc[:, int(bg_index)].clamp(0.0, 1.0)
        fg_scores_qc = cls_scores_qc.clone()
        fg_scores_qc[:, int(bg_index)] = 0.0
        fg_prob_q = fg_scores_qc.max(dim=-1).values.clamp(0.0, 1.0)
        pred_cls_q = torch.argmax(cls_scores_qc, dim=-1)
        valid_target_mask_q = targets_q != ignore_target
        unmatched_mask_q = valid_target_mask_q & (~matched_mask_q)

        def _add_stats(prefix, vals):
            vals = vals.reshape(-1).to(torch.float32)
            if int(vals.numel()) <= 0:
                for name in ("mean", "q10", "q50", "min", "max"):
                    out[f"{prefix}_{name}"] = z
                return
            vals_sorted = torch.sort(vals).values
            n = int(vals_sorted.numel())
            q10_idx = min(n - 1, max(0, int(round((n - 1) * 0.10))))
            q50_idx = min(n - 1, max(0, int(round((n - 1) * 0.50))))
            out[f"{prefix}_mean"] = vals.mean()
            out[f"{prefix}_q10"] = vals_sorted[q10_idx]
            out[f"{prefix}_q50"] = vals_sorted[q50_idx]
            out[f"{prefix}_min"] = vals_sorted[0]
            out[f"{prefix}_max"] = vals_sorted[-1]

        def _add_group(prefix, mask):
            _add_stats(f"{prefix}_bg_prob", bg_prob_q[mask])
            _add_stats(f"{prefix}_fg_prob", fg_prob_q[mask])
            out[f"{prefix}_pred_bg_count"] = ((pred_cls_q == int(bg_index)) & mask).sum().to(torch.float32)
            out[f"{prefix}_pred_fg_count"] = ((pred_cls_q != int(bg_index)) & mask).sum().to(torch.float32)
            out[f"{prefix}_count"] = mask.sum().to(torch.float32)
            for cls_idx in range(C):
                _add_stats(
                    f"{prefix}_score_c{cls_idx}",
                    cls_scores_qc[:, cls_idx][mask],
                )
                out[f"{prefix}_pred_count_c{cls_idx}"] = ((pred_cls_q == int(cls_idx)) & mask).sum().to(torch.float32)
                out[f"{prefix}_target_count_c{cls_idx}"] = ((targets_q == int(cls_idx)) & mask).sum().to(torch.float32)

        _add_group("dbg_query_cls_all", valid_target_mask_q)
        _add_group("dbg_query_cls_matched", matched_mask_q)
        _add_group("dbg_query_cls_unmatched", unmatched_mask_q)

        nms_mask_q = torch.zeros((Q,), device=query_cls_logits_qc.device, dtype=torch.bool)
        nms_radius = max(0.0, float(nms_radius_m))
        if (
            nms_radius > 0.0
            and torch.is_tensor(centers_world_tq3)
            and centers_world_tq3.dim() == 3
            and int(centers_world_tq3.shape[1]) == Q
        ):
            center_frame_idx = min(max(0, int(centers_world_tq3.shape[0]) - 1), 0)
            centers_xy_q2 = centers_world_tq3[center_frame_idx, :, :2].to(
                device=query_cls_logits_qc.device,
                dtype=torch.float32,
            )
            order = torch.argsort(cls_scores_qc.max(dim=-1).values, descending=True)
            kept = []
            radius_sq = float(nms_radius * nms_radius)
            for qid in order.tolist():
                qid = int(qid)
                if len(kept) > 0:
                    prev = torch.as_tensor(kept, device=centers_xy_q2.device, dtype=torch.long)
                    dist_sq = ((centers_xy_q2.index_select(0, prev) - centers_xy_q2[qid]) ** 2).sum(dim=-1)
                    if bool((dist_sq <= radius_sq).any().item()):
                        continue
                kept.append(qid)
            if len(kept) > 0:
                keep_idx = torch.as_tensor(kept, device=nms_mask_q.device, dtype=torch.long)
                nms_mask_q[keep_idx] = True
        nms_valid_mask_q = valid_target_mask_q & nms_mask_q
        _add_group("dbg_query_cls_nms_all", nms_valid_mask_q)
        _add_group("dbg_query_cls_nms_matched", nms_valid_mask_q & matched_mask_q)
        _add_group("dbg_query_cls_nms_unmatched", nms_valid_mask_q & unmatched_mask_q)
        out["dbg_query_cls_nms_radius_m"] = query_cls_logits_qc.new_tensor(float(nms_radius))
        out["dbg_query_cls_nms_keep_count"] = nms_valid_mask_q.sum().to(torch.float32)
        out["dbg_query_cls_target_fg_count"] = ((targets_q != int(bg_index)) & valid_target_mask_q).sum().to(torch.float32)
        out["dbg_query_cls_target_bg_count"] = (targets_q == int(bg_index)).sum().to(torch.float32)
        out["dbg_query_cls_target_ignored_count"] = query_cls_logits_qc.new_tensor(float(ignored_count))

        ce_weight = None
        if torch.is_tensor(class_weights):
            ce_weight = class_weights.to(device=query_cls_logits_qc.device, dtype=torch.float32)

        if not bool((targets_q != ignore_target).any().item()):
            out["dbg_query_cls_matched_count"] = query_cls_logits_qc.new_tensor(float(matched_count))
            out["dbg_query_cls_bg_count"] = query_cls_logits_qc.new_tensor(0.0)
            out["dbg_query_cls_ignored_count"] = query_cls_logits_qc.new_tensor(float(ignored_count))
            return out

        loss = F.cross_entropy(
            query_cls_logits_qc.to(torch.float32),
            targets_q,
            weight=ce_weight,
            ignore_index=ignore_target,
            reduction="mean",
        ) * float(loss_weight)
        out["loss_query_cls"] = loss
        out["dbg_query_cls_matched_count"] = query_cls_logits_qc.new_tensor(float(matched_count))
        out["dbg_query_cls_bg_count"] = query_cls_logits_qc.new_tensor(float((targets_q == int(bg_index)).sum().item()))
        out["dbg_query_cls_ignored_count"] = query_cls_logits_qc.new_tensor(float(ignored_count))
        return out

    def _compute_query_center_match_loss_from_match(
        self,
        centers_world_tq3: torch.Tensor,
        inst_match_result: dict = None,
        loss_weight: float = 1.0,
        loss_type: str = "l1",
    ):
        """
        Routed center loss on matched pairs.
        Expects centers_world_tq3 to be aligned in present-frame lidar coordinates.
        """
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        if not isinstance(inst_match_result, dict):
            return None
        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_centers_tn3 = inst_match_result.get("gt_centers_tn3", None)
        gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
            or (not torch.is_tensor(gt_centers_tn3))
            or (not torch.is_tensor(gt_valid_tn))
            or gt_centers_tn3.dim() != 3
            or gt_valid_tn.dim() != 2
        ):
            return None

        T, Q, _ = [int(v) for v in centers_world_tq3.shape]
        Tg, N, _ = [int(v) for v in gt_centers_tn3.shape]
        t_match = min(T, Tg, int(gt_valid_tn.shape[0]))
        if t_match <= 0:
            return None

        mq = matched_query_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        mi = matched_inst_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        keep = (mq >= 0) & (mq < Q) & (mi >= 0) & (mi < N)
        if not bool(keep.any().item()):
            return None
        mq = mq[keep]
        mi = mi[keep]
        if mq.numel() <= 0:
            return None

        pred_tk3 = centers_world_tq3[:t_match].index_select(1, mq).to(torch.float32)
        gt_tk3 = gt_centers_tn3[:t_match].to(device=centers_world_tq3.device, dtype=torch.float32).index_select(1, mi)
        valid_tk = gt_valid_tn[:t_match].to(device=centers_world_tq3.device, dtype=torch.bool).index_select(1, mi)
        valid_f = valid_tk.to(torch.float32)
        valid_cnt = valid_f.sum()
        if not bool((valid_cnt > 0).item()):
            return None

        loss_type = str(loss_type).lower()
        if loss_type == "l2":
            err_tk = (pred_tk3 - gt_tk3).pow(2).sum(dim=-1)
        else:
            err_tk = torch.abs(pred_tk3 - gt_tk3).sum(dim=-1)
        return ((err_tk * valid_f).sum() / valid_cnt.clamp_min(1.0)) * float(loss_weight)

    def _compute_query_center_match_dbg_from_match(
        self,
        centers_world_tq3: torch.Tensor,
        inst_match_result: dict = None,
    ):
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        z = centers_world_tq3.sum() * 0.0
        zero_dbg = {
            "dbg_query_matched_center_pair_count": z,
            "dbg_query_matched_center_valid_frame_count": z,
            "dbg_query_matched_center_l1_mean": z,
            "dbg_query_matched_center_l2_mean": z,
            "dbg_query_matched_center_xy_l2_mean": z,
            "dbg_query_matched_center_abs_x_mean": z,
            "dbg_query_matched_center_abs_y_mean": z,
            "dbg_query_matched_center_abs_z_mean": z,
            "dbg_query_matched_center_l2_max": z,
        }
        if not isinstance(inst_match_result, dict):
            return zero_dbg

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_centers_tn3 = inst_match_result.get("gt_centers_tn3", None)
        gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
            or (not torch.is_tensor(gt_centers_tn3))
            or (not torch.is_tensor(gt_valid_tn))
            or gt_centers_tn3.dim() != 3
            or gt_valid_tn.dim() != 2
        ):
            return zero_dbg

        T, Q, _ = [int(v) for v in centers_world_tq3.shape]
        Tg, N, _ = [int(v) for v in gt_centers_tn3.shape]
        t_match = min(T, Tg, int(gt_valid_tn.shape[0]))
        if t_match <= 0:
            return zero_dbg

        mq = matched_query_idx.to(device=centers_world_tq3.device, dtype=torch.long).reshape(-1)
        mi = matched_inst_idx.to(device=centers_world_tq3.device, dtype=torch.long).reshape(-1)
        keep = (mq >= 0) & (mq < Q) & (mi >= 0) & (mi < N)
        if not bool(keep.any().item()):
            return zero_dbg
        mq = mq[keep]
        mi = mi[keep]

        pred_tk3 = centers_world_tq3[:t_match].index_select(1, mq).to(torch.float32)
        gt_tk3 = gt_centers_tn3[:t_match].to(
            device=centers_world_tq3.device, dtype=torch.float32
        ).index_select(1, mi)
        valid_tk = gt_valid_tn[:t_match].to(
            device=centers_world_tq3.device, dtype=torch.bool
        ).index_select(1, mi)
        finite_tk = torch.isfinite(pred_tk3).all(dim=-1) & torch.isfinite(gt_tk3).all(dim=-1)
        valid_tk = valid_tk & finite_tk
        if not bool(valid_tk.any().item()):
            return zero_dbg

        diff_tk3 = pred_tk3 - gt_tk3
        abs_diff_tk3 = torch.abs(diff_tk3)
        l1_tk = abs_diff_tk3.sum(dim=-1)
        l2_tk = torch.linalg.norm(diff_tk3, dim=-1)
        xy_tk = torch.linalg.norm(diff_tk3[..., :2], dim=-1)
        valid_vals = valid_tk.to(torch.float32)
        valid_cnt = valid_vals.sum().clamp_min(1.0)
        matched_pair_count = centers_world_tq3.new_tensor(float(int(mq.numel())))

        def _masked_mean(x):
            return (x * valid_vals).sum() / valid_cnt

        return {
            "dbg_query_matched_center_pair_count": matched_pair_count,
            "dbg_query_matched_center_valid_frame_count": valid_vals.sum(),
            "dbg_query_matched_center_l1_mean": _masked_mean(l1_tk),
            "dbg_query_matched_center_l2_mean": _masked_mean(l2_tk),
            "dbg_query_matched_center_xy_l2_mean": _masked_mean(xy_tk),
            "dbg_query_matched_center_abs_x_mean": _masked_mean(abs_diff_tk3[..., 0]),
            "dbg_query_matched_center_abs_y_mean": _masked_mean(abs_diff_tk3[..., 1]),
            "dbg_query_matched_center_abs_z_mean": _masked_mean(abs_diff_tk3[..., 2]),
            "dbg_query_matched_center_l2_max": l2_tk[valid_tk].max() if bool(valid_tk.any().item()) else z,
        }

    def _compute_query_trajectory_loss_from_match(
        self,
        centers_world_tq3: torch.Tensor,
        pred_traj_offsets_fq2: torch.Tensor = None,
        inst_match_result: dict = None,
        loss_weight: float = 1.0,
        loss_type: str = "l1",
        present_local_idx: int = 0,
        moving_reweight_enabled: bool = False,
        moving_threshold_m: float = 0.5,
        moving_weight: float = 5.0,
        static_weight: float = 1.0,
    ):
        """
        Future trajectory loss on matched pairs.
        Supervises present-frame per-step xy deltas: (t->t+1), (t+1->t+2), ...
        """
        if float(loss_weight) <= 0.0:
            return None
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        if int(centers_world_tq3.shape[0]) <= 1:
            return None
        if not isinstance(inst_match_result, dict):
            return None

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_centers_tn3 = inst_match_result.get("gt_centers_tn3", None)
        gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
            or (not torch.is_tensor(gt_centers_tn3))
            or (not torch.is_tensor(gt_valid_tn))
            or gt_centers_tn3.dim() != 3
            or gt_valid_tn.dim() != 2
        ):
            return None

        T, Q, _ = [int(v) for v in centers_world_tq3.shape]
        Tg, N, _ = [int(v) for v in gt_centers_tn3.shape]
        t_match = min(T, Tg, int(gt_valid_tn.shape[0]))
        present_local_idx = max(0, min(t_match - 1, int(present_local_idx)))
        if t_match <= (present_local_idx + 1):
            return None

        mq = matched_query_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        mi = matched_inst_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        keep = (mq >= 0) & (mq < Q) & (mi >= 0) & (mi < N)
        if not bool(keep.any().item()):
            return None
        mq = mq[keep]
        mi = mi[keep]
        if mq.numel() <= 0:
            return None

        future_steps = int(t_match - present_local_idx - 1)
        if future_steps <= 0:
            return None

        gt_centers_xy_tn2 = gt_centers_tn3.to(
            device=centers_world_tq3.device, dtype=torch.float32
        )[..., :2]
        gt_deltas_tn2 = (
            gt_centers_xy_tn2[(present_local_idx + 1):t_match]
            - gt_centers_xy_tn2[present_local_idx:(t_match - 1)]
        )
        gt_tk2 = gt_deltas_tn2.index_select(1, mi)

        valid_prev_tk = gt_valid_tn[present_local_idx:(t_match - 1)].to(
            device=centers_world_tq3.device, dtype=torch.bool
        ).index_select(1, mi)
        valid_next_tk = gt_valid_tn[(present_local_idx + 1):t_match].to(
            device=centers_world_tq3.device, dtype=torch.bool
        ).index_select(1, mi)
        valid_tk = valid_prev_tk & valid_next_tk
        valid_f = valid_tk.to(torch.float32)
        if not bool((valid_f.sum() > 0).item()):
            return None

        pred_tk2 = None
        if (
            torch.is_tensor(pred_traj_offsets_fq2)
            and pred_traj_offsets_fq2.dim() == 3
            and int(pred_traj_offsets_fq2.shape[1]) == Q
            and int(pred_traj_offsets_fq2.shape[2]) == 2
        ):
            pred_steps = min(future_steps, int(pred_traj_offsets_fq2.shape[0]))
            pred_tk2 = pred_traj_offsets_fq2[:pred_steps].to(
                device=centers_world_tq3.device, dtype=torch.float32
            ).index_select(1, mq)
            if pred_steps < future_steps:
                gt_tk2 = gt_tk2[:pred_steps]
                valid_f = valid_f[:pred_steps]
                valid_tk = valid_tk[:pred_steps]
            if pred_tk2.numel() <= 0:
                return None
        else:
            pred_centers_xy_tq2 = centers_world_tq3.to(torch.float32)[..., :2]
            pred_tk2 = (
                pred_centers_xy_tq2[(present_local_idx + 1):t_match].index_select(1, mq)
                - pred_centers_xy_tq2[present_local_idx:(t_match - 1)].index_select(1, mq)
            )

        loss_type = str(loss_type).lower()
        if loss_type == "l2":
            err_tk = torch.norm(pred_tk2 - gt_tk2, p=2, dim=-1)
        else:
            err_tk = torch.abs(pred_tk2 - gt_tk2).sum(dim=-1)

        traj_weight_tk = torch.ones_like(valid_f)
        motion_mag_tk = torch.norm(gt_tk2, p=2, dim=-1)
        reweight_applied = bool(moving_reweight_enabled)
        if bool(moving_reweight_enabled):
            traj_weight_tk = torch.where(
                motion_mag_tk >= float(moving_threshold_m),
                torch.full_like(valid_f, float(moving_weight)),
                torch.full_like(valid_f, float(static_weight)),
            )

        eff_weight_tk = valid_f * traj_weight_tk
        eff_cnt = eff_weight_tk.sum()
        if not bool((eff_cnt > 0).item()):
            return None
        loss_val = ((err_tk * eff_weight_tk).sum() / eff_cnt.clamp_min(1.0)) * float(loss_weight)

        moving_mask_tk = motion_mag_tk >= float(moving_threshold_m)
        out = {
            "loss_query_traj": loss_val,
            "dbg_query_traj_reweight_enabled": loss_val.new_tensor(
                float(1.0 if moving_reweight_enabled else 0.0)
            ),
            "dbg_query_traj_reweight_applied": loss_val.new_tensor(
                float(1.0 if reweight_applied else 0.0)
            ),
            "dbg_query_traj_valid_count": valid_f.sum(),
            "dbg_query_traj_weight_mean": self._safe_weighted_mean(
                traj_weight_tk,
                valid_f,
            ),
            "dbg_query_traj_moving_ratio": self._safe_weighted_mean(
                moving_mask_tk.to(valid_f.dtype),
                valid_f,
            ),
            "dbg_query_traj_motion_mag_mean": self._safe_weighted_mean(
                motion_mag_tk,
                valid_f,
            ),
        }
        return out

    def _compute_query_raw_bev_overlap_loss(
        self,
        query_bev_feat_tqd: torch.Tensor,
        query_bev_valid_tq: torch.Tensor,
        inst_match_result: dict = None,
        gt_inst_bev_feat_tnd: torch.Tensor = None,
        gt_inst_bev_feat_valid_tn: torch.Tensor = None,
        loss_weight: float = 1.0,
    ) -> dict:
        z = None
        if torch.is_tensor(query_bev_feat_tqd):
            z = query_bev_feat_tqd.sum() * 0.0
        elif torch.is_tensor(gt_inst_bev_feat_tnd):
            z = gt_inst_bev_feat_tnd.sum() * 0.0
        else:
            z = self.mean_weight.sum() * 0.0
        out = {
            "loss_query_raw_bev_overlap": z,
            "dbg_query_raw_bev_overlap_pair_count": z,
            "dbg_query_raw_bev_overlap_valid_count": z,
        }
        if (
            (not torch.is_tensor(query_bev_feat_tqd))
            or (not torch.is_tensor(gt_inst_bev_feat_tnd))
            or (not isinstance(inst_match_result, dict))
        ):
            return out
        if query_bev_feat_tqd.dim() != 3 or gt_inst_bev_feat_tnd.dim() != 3:
            return out

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
        ):
            return out

        Tq, Q, Dq = [int(v) for v in query_bev_feat_tqd.shape]
        Tg, N, Dg = [int(v) for v in gt_inst_bev_feat_tnd.shape]
        if Dq != Dg:
            return out
        T = min(Tq, Tg)
        if T <= 0:
            return out
        q_feat_tqd = query_bev_feat_tqd[:T].to(torch.float32)
        g_feat_tnd = gt_inst_bev_feat_tnd[:T].to(device=query_bev_feat_tqd.device, dtype=torch.float32)
        valid_tq = torch.ones((T, Q), device=query_bev_feat_tqd.device, dtype=torch.bool)
        if (
            torch.is_tensor(query_bev_valid_tq)
            and query_bev_valid_tq.dim() == 2
            and int(query_bev_valid_tq.shape[0]) >= T
            and int(query_bev_valid_tq.shape[1]) == Q
        ):
            valid_tq = valid_tq & query_bev_valid_tq[:T].to(device=query_bev_feat_tqd.device, dtype=torch.bool)
        valid_tn = torch.ones((T, N), device=query_bev_feat_tqd.device, dtype=torch.bool)
        if (
            torch.is_tensor(gt_inst_bev_feat_valid_tn)
            and gt_inst_bev_feat_valid_tn.dim() == 2
            and int(gt_inst_bev_feat_valid_tn.shape[0]) >= T
            and int(gt_inst_bev_feat_valid_tn.shape[1]) == N
        ):
            valid_tn = valid_tn & gt_inst_bev_feat_valid_tn[:T].to(device=query_bev_feat_tqd.device, dtype=torch.bool)

        mq = matched_query_idx.to(device=query_bev_feat_tqd.device, dtype=torch.long)
        mi = matched_inst_idx.to(device=query_bev_feat_tqd.device, dtype=torch.long)
        keep = (mq >= 0) & (mq < Q) & (mi >= 0) & (mi < N)
        if not bool(keep.any().item()):
            return out
        mq = mq[keep]
        mi = mi[keep]
        if mq.numel() <= 0:
            return out

        q_sel_tkd = q_feat_tqd.index_select(1, mq)
        g_sel_tkd = g_feat_tnd.index_select(1, mi)
        valid_tk = valid_tq.index_select(1, mq) & valid_tn.index_select(1, mi)
        valid_f = valid_tk.to(torch.float32)
        valid_cnt = valid_f.sum()
        out["dbg_query_raw_bev_overlap_pair_count"] = query_bev_feat_tqd.new_tensor(float(mq.numel()))
        out["dbg_query_raw_bev_overlap_valid_count"] = valid_cnt
        if not bool((valid_cnt > 0).item()):
            return out

        qn = F.normalize(q_sel_tkd, dim=-1, eps=1e-6)
        gn = F.normalize(g_sel_tkd, dim=-1, eps=1e-6)
        cos_tk = F.cosine_similarity(qn, gn, dim=-1, eps=1e-6).clamp(-1.0, 1.0)
        dist_tk = (1.0 - cos_tk).clamp(0.0, 2.0)
        out["loss_query_raw_bev_overlap"] = (
            (dist_tk * valid_f).sum() / valid_cnt.clamp_min(1.0)
        ) * float(loss_weight)
        return out

    def _compute_query_attn_bbox_loss(
        self,
        query_attn_weights_tqnhw: torch.Tensor,
        inst_match_result: dict = None,
        gt_attn_targets: dict = None,
        loss_weight: float = None,
        unmatched_weight: float = None,
        eps: float = None,
    ) -> dict:
        if torch.is_tensor(query_attn_weights_tqnhw):
            z = query_attn_weights_tqnhw.sum() * 0.0
        else:
            z = self.mean_weight.sum() * 0.0

        out = {
            "loss_query_attn_bbox": z,
            "dbg_query_attn_bbox_matched_raw": z,
            "dbg_query_attn_bbox_unmatched_raw": z,
            "dbg_query_attn_bbox_other_raw": z,
            "dbg_query_attn_bbox_matched_count": z,
            "dbg_query_attn_bbox_unmatched_count": z,
            "dbg_query_attn_bbox_other_count": z,
            "dbg_query_attn_bbox_valid_frame_count": z,
            "dbg_query_attn_bbox_pred_entropy": z,
            "dbg_query_attn_bbox_target_entropy": z,
            "dbg_query_attn_bbox_shape_invalid_skip": z,
            "dbg_query_attn_bbox_nonfinite_skip": z,
            "dbg_query_attn_bbox_no_matched_pairs": z,
            "dbg_query_attn_bbox_empty_gt_mask": z,
            "dbg_query_attn_bbox_inside_mass_mean": z,
            "dbg_query_attn_bbox_inside_mass_min": z,
            "dbg_query_attn_bbox_inside_mass_max": z,
            "dbg_query_attn_bbox_outside_mass_mean": z,
            "dbg_query_attn_bbox_soft_iou_mean": z,
            "dbg_query_attn_bbox_soft_iou_min": z,
            "dbg_query_attn_bbox_soft_iou_max": z,
            "dbg_query_attn_bbox_other_mass_mean": z,
            "dbg_query_attn_bbox_other_mass_max": z,
            "dbg_query_attn_bbox_unmatched_mass_mean": z,
            "dbg_query_attn_bbox_unmatched_mass_max": z,
        }
        if (not torch.is_tensor(query_attn_weights_tqnhw)) or query_attn_weights_tqnhw.dim() != 5:
            out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
            return out
        if not isinstance(inst_match_result, dict) or (not isinstance(gt_attn_targets, dict)):
            out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        gt_inst_mask_tnhw = gt_attn_targets.get("gt_inst_mask_tnhw", None)
        gt_inst_valid_tn = gt_attn_targets.get("gt_inst_valid_tn", None)
        inverse_union_dist_thw = gt_attn_targets.get("inverse_union_dist_thw", None)
        attn_t_idx_t = gt_attn_targets.get("attn_t_idx_t", None)
        if (
            (not torch.is_tensor(gt_inst_mask_tnhw))
            or (not torch.is_tensor(gt_inst_valid_tn))
            or gt_inst_mask_tnhw.dim() != 4
            or gt_inst_valid_tn.dim() != 2
        ):
            out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        if (not torch.is_tensor(matched_query_idx)) or (not torch.is_tensor(matched_inst_idx)):
            matched_query_idx = torch.empty((0,), device=query_attn_weights_tqnhw.device, dtype=torch.long)
            matched_inst_idx = torch.empty((0,), device=query_attn_weights_tqnhw.device, dtype=torch.long)

        t_attn, q_attn, n_cam, h_attn, w_attn = [int(v) for v in query_attn_weights_tqnhw.shape]
        t_gt, n_inst, h_gt, w_gt = [int(v) for v in gt_inst_mask_tnhw.shape]
        t_valid, n_valid = [int(v) for v in gt_inst_valid_tn.shape]
        inv_shape_ok = torch.is_tensor(inverse_union_dist_thw) and inverse_union_dist_thw.dim() == 3
        if inv_shape_ok:
            t_inv, h_inv, w_inv = [int(v) for v in inverse_union_dist_thw.shape]
        else:
            t_inv, h_inv, w_inv = 0, h_attn, w_attn
        if (
            t_attn <= 0
            or q_attn <= 0
            or n_cam <= 0
            or t_gt <= 0
            or n_inst < 0
            or (h_attn != h_gt)
            or (w_attn != w_gt)
            or (t_gt != t_valid)
            or (inv_shape_ok and ((h_attn != h_inv) or (w_attn != w_inv)))
        ):
            out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        t = min(t_attn, t_gt, t_inv if inv_shape_ok else t_gt)
        if t <= 0:
            out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
            return out
        if torch.is_tensor(attn_t_idx_t) and int(attn_t_idx_t.numel()) >= t:
            attn_t_idx = attn_t_idx_t[:t].to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
            if not bool(((attn_t_idx >= 0) & (attn_t_idx < t_attn)).all().item()):
                out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
                return out
            attn_weights_sel_tqnhw = query_attn_weights_tqnhw.index_select(0, attn_t_idx)
        else:
            attn_weights_sel_tqnhw = query_attn_weights_tqnhw[:t]
        p = int(h_attn * w_attn)
        eps_v = float(getattr(self, "query_attn_bbox_eps", 1e-6) if eps is None else eps)
        eps_v = max(eps_v, 1e-12)
        total_w = float(
            getattr(self, "query_attn_bbox_loss_weight", 1.0)
            if loss_weight is None else loss_weight
        )
        unmatched_w = float(
            getattr(self, "query_attn_bbox_unmatched_weight", 0.0)
            if unmatched_weight is None else unmatched_weight
        )
        other_w = float(getattr(self, "query_attn_bbox_other_weight", 0.0))
        other_mode = str(getattr(self, "query_attn_bbox_other_mode", "union")).lower()

        attn_tqhw = attn_weights_sel_tqnhw.to(torch.float32).sum(dim=2)
        attn_tqp = attn_tqhw.reshape(t, q_attn, p)
        attn_denom_tq1 = attn_tqp.sum(dim=-1, keepdim=True).clamp_min(eps_v)
        pred_tqp = attn_tqp / attn_denom_tq1
        if not bool(torch.isfinite(pred_tqp).all().item()):
            out["dbg_query_attn_bbox_nonfinite_skip"] = z.new_tensor(1.0)
            return out

        mq = matched_query_idx.to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
        mi = matched_inst_idx.to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
        if mq.numel() != mi.numel():
            out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
            return out
        if mq.numel() > 0:
            keep = (mq >= 0) & (mq < q_attn) & (mi >= 0) & (mi < n_inst)
            mq = mq[keep]
            mi = mi[keep]
        k = int(mq.numel())
        out["dbg_query_attn_bbox_no_matched_pairs"] = z.new_tensor(float(1.0 if k <= 0 else 0.0))

        matched_raw = z
        other_raw = z
        unmatched_raw = z
        matched_valid_cnt = z
        other_valid_cnt = z
        unmatched_valid_cnt = z
        pred_entropy_terms = []
        target_entropy_terms = []
        union_mask_tp = None
        union_sum_t1 = None
        if n_inst > 0:
            gt_mask_tnp = gt_inst_mask_tnhw[:t].reshape(t, n_inst, p).to(device=pred_tqp.device, dtype=torch.float32)
            valid_mask_tn1 = gt_inst_valid_tn[:t].to(device=pred_tqp.device, dtype=torch.bool).unsqueeze(-1)
            union_mask_tp = (gt_mask_tnp * valid_mask_tn1.to(torch.float32)).amax(dim=1)
            union_sum_t1 = union_mask_tp.sum(dim=-1, keepdim=True)
        else:
            union_mask_tp = pred_tqp.new_zeros((t, p))
            union_sum_t1 = pred_tqp.new_zeros((t, 1))
        if k > 0:
            pred_tkp = pred_tqp.index_select(1, mq)
            pred_raw_tknp = attn_weights_sel_tqnhw.index_select(1, mq).to(
                device=pred_tqp.device, dtype=torch.float32
            ).clamp_min(0.0).reshape(t, k, n_cam, p)
            pred_prob_sum_tk11 = pred_raw_tknp.sum(dim=(2, 3), keepdim=True).clamp_min(eps_v)
            pred_prob_tknp = pred_raw_tknp / pred_prob_sum_tk11
            target_mask_tkp = gt_inst_mask_tnhw[:t].index_select(1, mi).reshape(t, k, p).to(
                device=pred_tqp.device, dtype=torch.float32
            )
            target_valid_tk = gt_inst_valid_tn[:t].index_select(1, mi).to(
                device=pred_tqp.device, dtype=torch.bool
            )
            target_sum_tk = target_mask_tkp.sum(dim=-1)
            if not bool((target_sum_tk > 0.0).any().item()):
                out["dbg_query_attn_bbox_empty_gt_mask"] = z.new_tensor(1.0)
            valid_tk = target_valid_tk & (target_sum_tk > 0.0)
            valid_tk = valid_tk & torch.isfinite(pred_prob_tknp).all(dim=(2, 3))
            valid_f = valid_tk.to(torch.float32)
            matched_valid_cnt = valid_f.sum()
            if bool((matched_valid_cnt > 0).item()):
                target_prob_tknp = (
                    target_mask_tkp.unsqueeze(2)
                    / (target_sum_tk[:, :, None, None] * float(n_cam)).clamp_min(eps_v)
                ).expand(t, k, n_cam, p)
                log_pred = torch.log(pred_prob_tknp.clamp_min(eps_v))
                log_tgt = torch.log(target_prob_tknp.clamp_min(eps_v))
                kl_tk = (target_prob_tknp * (log_tgt - log_pred)).sum(dim=(2, 3))
                matched_raw = (kl_tk * valid_f).sum() / matched_valid_cnt.clamp_min(1.0)

                inter_tk = (pred_prob_tknp * target_mask_tkp.unsqueeze(2)).sum(dim=(2, 3))
                pred_sum_tk = pred_prob_tknp.sum(dim=(2, 3))
                target_sum_cam_tk = target_sum_tk * float(n_cam)
                union_tk = (pred_sum_tk + target_sum_cam_tk - inter_tk).clamp_min(0.0)
                soft_iou_tk = (inter_tk / union_tk.clamp_min(eps_v)).clamp(0.0, 1.0)
                soft_iou_valid = soft_iou_tk[valid_tk]
                out["dbg_query_attn_bbox_soft_iou_mean"] = soft_iou_valid.mean()
                out["dbg_query_attn_bbox_soft_iou_min"] = soft_iou_valid.min()
                out["dbg_query_attn_bbox_soft_iou_max"] = soft_iou_valid.max()
                inside_mass_tk = inter_tk.clamp(0.0, 1.0)
                outside_mass_tk = (1.0 - inside_mass_tk).clamp(0.0, 1.0)
                inside_valid = inside_mass_tk[valid_tk]
                outside_valid = outside_mass_tk[valid_tk]
                out["dbg_query_attn_bbox_inside_mass_mean"] = inside_valid.mean()
                out["dbg_query_attn_bbox_inside_mass_min"] = inside_valid.min()
                out["dbg_query_attn_bbox_inside_mass_max"] = inside_valid.max()
                out["dbg_query_attn_bbox_outside_mass_mean"] = outside_valid.mean()
                pred_entropy_valid = -(pred_prob_tknp.clamp_min(eps_v) * log_pred).sum(dim=(2, 3))
                pred_entropy_terms.append(pred_entropy_valid[valid_tk])
                target_entropy_valid = -(target_prob_tknp.clamp_min(eps_v) * log_tgt).sum(dim=(2, 3))
                target_entropy_terms.append(target_entropy_valid[valid_tk])

                if other_w > 0.0:
                    other_mask_tkp = union_mask_tp.unsqueeze(1) - target_mask_tkp
                    other_mask_tkp = other_mask_tkp.clamp_(0.0, 1.0)
                    other_sum_tk = other_mask_tkp.sum(dim=-1)
                    other_valid_tk = valid_tk & (other_sum_tk > 0.0)
                    other_valid_f = other_valid_tk.to(torch.float32)
                    other_valid_cnt = other_valid_f.sum()
                    if bool((other_valid_cnt > 0).item()):
                        other_mass_tk = (
                            pred_prob_tknp * other_mask_tkp.unsqueeze(2)
                        ).sum(dim=(2, 3)).clamp(0.0, 1.0)
                        if other_mode == "union_log":
                            other_loss_tk = -torch.log((1.0 - other_mass_tk).clamp_min(eps_v))
                        else:
                            other_loss_tk = other_mass_tk
                        other_raw = (other_loss_tk * other_valid_f).sum() / other_valid_cnt.clamp_min(1.0)
                        other_mass_valid = other_mass_tk[other_valid_tk]
                        out["dbg_query_attn_bbox_other_mass_mean"] = other_mass_valid.mean()
                        out["dbg_query_attn_bbox_other_mass_max"] = other_mass_valid.max()

        if unmatched_w > 0.0:
            if (not inv_shape_ok) or t_inv < t:
                out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
                return out
            matched_mask_q = torch.zeros((q_attn,), device=pred_tqp.device, dtype=torch.bool)
            if k > 0:
                matched_mask_q[mq] = True
            uq = torch.nonzero(~matched_mask_q, as_tuple=False).squeeze(1)
            u = int(uq.numel())
            out["dbg_query_attn_bbox_unmatched_count"] = z.new_tensor(float(u))
            if u > 0:
                pred_raw_tunp = attn_weights_sel_tqnhw.index_select(1, uq).to(
                    device=pred_tqp.device, dtype=torch.float32
                ).clamp_min(0.0).reshape(t, u, n_cam, p)
                pred_prob_sum_tu11 = pred_raw_tunp.sum(dim=(2, 3), keepdim=True).clamp_min(eps_v)
                pred_prob_tunp = pred_raw_tunp / pred_prob_sum_tu11
                target_tp = inverse_union_dist_thw[:t].reshape(t, p).to(device=pred_tqp.device, dtype=torch.float32)
                target_sum_t1 = target_tp.sum(dim=-1, keepdim=True)
                target_prob_tnp = (
                    target_tp[:, None, :]
                    / (target_sum_t1[:, None, :] * float(n_cam)).clamp_min(eps_v)
                ).expand(t, n_cam, p)
                target_prob_tunp = target_prob_tnp.unsqueeze(1).expand(t, u, n_cam, p)
                unmatched_valid_tu = (target_sum_t1.squeeze(-1) > 0.0).unsqueeze(1).expand(t, u)
                unmatched_valid_tu = unmatched_valid_tu & torch.isfinite(pred_prob_tunp).all(dim=(2, 3))
                unmatched_valid_tu = unmatched_valid_tu & torch.isfinite(target_prob_tunp).all(dim=(2, 3))
                unmatched_valid_f = unmatched_valid_tu.to(torch.float32)
                unmatched_valid_cnt = unmatched_valid_f.sum()
                if bool((unmatched_valid_cnt > 0).item()):
                    log_pred_u = torch.log(pred_prob_tunp.clamp_min(eps_v))
                    log_tgt_u = torch.log(target_prob_tunp.clamp_min(eps_v))
                    kl_tu = (target_prob_tunp * (log_tgt_u - log_pred_u)).sum(dim=(2, 3))
                    unmatched_raw = (kl_tu * unmatched_valid_f).sum() / unmatched_valid_cnt.clamp_min(1.0)
                    union_mass_tu = (
                        pred_prob_tunp * union_mask_tp.unsqueeze(1).unsqueeze(2)
                    ).sum(dim=(2, 3)).clamp(0.0, 1.0)
                    unmatched_mass_valid = union_mass_tu[unmatched_valid_tu]
                    out["dbg_query_attn_bbox_unmatched_mass_mean"] = unmatched_mass_valid.mean()
                    out["dbg_query_attn_bbox_unmatched_mass_max"] = unmatched_mass_valid.max()
                    pred_entropy_unmatched = -(pred_prob_tunp.clamp_min(eps_v) * log_pred_u).sum(dim=(2, 3))
                    pred_entropy_terms.append(pred_entropy_unmatched[unmatched_valid_tu])

        out["dbg_query_attn_bbox_matched_raw"] = matched_raw
        out["dbg_query_attn_bbox_unmatched_raw"] = unmatched_raw
        out["dbg_query_attn_bbox_other_raw"] = other_raw
        out["dbg_query_attn_bbox_matched_count"] = z.new_tensor(float(k))
        out["dbg_query_attn_bbox_other_count"] = other_valid_cnt
        out["dbg_query_attn_bbox_valid_frame_count"] = matched_valid_cnt
        if len(pred_entropy_terms) > 0:
            pred_entropy_all = torch.cat(pred_entropy_terms, dim=0)
            if int(pred_entropy_all.numel()) > 0:
                out["dbg_query_attn_bbox_pred_entropy"] = pred_entropy_all.mean()
        if len(target_entropy_terms) > 0:
            target_entropy_all = torch.cat(target_entropy_terms, dim=0)
            if int(target_entropy_all.numel()) > 0:
                out["dbg_query_attn_bbox_target_entropy"] = target_entropy_all.mean()
        loss_raw = matched_raw + other_w * other_raw + unmatched_w * unmatched_raw
        out["loss_query_attn_bbox"] = loss_raw * total_w
        return out

    def _compute_query_attn_cam_gaussian_score(
        self,
        query_attn_weights_tqnhw: torch.Tensor,
        cam_targets: dict,
        eps: float = None,
    ) -> dict:
        if torch.is_tensor(query_attn_weights_tqnhw):
            z = query_attn_weights_tqnhw.sum() * 0.0
            q_count = int(query_attn_weights_tqnhw.shape[1]) if query_attn_weights_tqnhw.dim() >= 2 else 0
            score_q = torch.zeros((max(0, q_count),), device=query_attn_weights_tqnhw.device, dtype=torch.float32)
        else:
            z = self.mean_weight.sum() * 0.0
            score_q = torch.zeros((0,), device=z.device, dtype=torch.float32)

        out = {
            "cam_attn_score_q": score_q,
            "cam_attn_score_valid_q": torch.zeros_like(score_q, dtype=torch.bool),
            "dbg_query_attn_cam_score_mean": z,
            "dbg_query_attn_cam_metric_raw_mean": z,
            "dbg_query_attn_cam_valid_pair_count": z,
            "dbg_query_attn_cam_nonfinite_skip": z,
            "dbg_query_attn_cam_shape_invalid_skip": z,
        }
        if (not torch.is_tensor(query_attn_weights_tqnhw)) or query_attn_weights_tqnhw.dim() != 5:
            out["dbg_query_attn_cam_shape_invalid_skip"] = z.new_tensor(1.0)
            return out
        if not isinstance(cam_targets, dict):
            out["dbg_query_attn_cam_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        gauss_target_tqnhw = cam_targets.get("gauss_target_tqnhw", None)
        gauss_valid_tqn = cam_targets.get("gauss_valid_tqn", None)
        attn_t_idx_t = cam_targets.get("attn_t_idx_t", None)
        if (
            (not torch.is_tensor(gauss_target_tqnhw))
            or gauss_target_tqnhw.dim() != 5
            or (not torch.is_tensor(gauss_valid_tqn))
            or gauss_valid_tqn.dim() != 3
            or (not torch.is_tensor(attn_t_idx_t))
            or attn_t_idx_t.dim() != 1
        ):
            out["dbg_query_attn_cam_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        t_attn, q_attn, n_cam, h_attn, w_attn = [int(v) for v in query_attn_weights_tqnhw.shape]
        t_tgt, q_tgt, n_tgt, h_tgt, w_tgt = [int(v) for v in gauss_target_tqnhw.shape]
        if (
            t_tgt <= 0
            or q_attn <= 0
            or q_tgt != q_attn
            or n_tgt != n_cam
            or h_tgt != h_attn
            or w_tgt != w_attn
            or int(gauss_valid_tqn.shape[0]) != t_tgt
            or int(gauss_valid_tqn.shape[1]) != q_attn
            or int(gauss_valid_tqn.shape[2]) != n_cam
            or int(attn_t_idx_t.numel()) != t_tgt
        ):
            out["dbg_query_attn_cam_shape_invalid_skip"] = z.new_tensor(1.0)
            return out
        if not bool(((attn_t_idx_t >= 0) & (attn_t_idx_t < t_attn)).all().item()):
            out["dbg_query_attn_cam_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        eps_v = float(getattr(self, "query_attn_cam_eps", 1e-6) if eps is None else eps)
        eps_v = max(eps_v, 1e-12)
        if str(getattr(self, "query_attn_cam_camera_reduce_mode", "camera_aggregated_first")).lower() != "camera_aggregated_first":
            out["dbg_query_attn_cam_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        attn_sel_tqnhw = query_attn_weights_tqnhw.index_select(
            0, attn_t_idx_t.to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
        ).to(torch.float32)
        tgt_tqnhw = gauss_target_tqnhw.to(device=query_attn_weights_tqnhw.device, dtype=torch.float32)
        valid_tqn = gauss_valid_tqn.to(device=query_attn_weights_tqnhw.device, dtype=torch.bool)

        pred_tqhw = attn_sel_tqnhw.sum(dim=2)
        tgt_tqhw = tgt_tqnhw.sum(dim=2)
        p = int(h_attn * w_attn)
        pred_tqp_raw = pred_tqhw.reshape(t_tgt, q_attn, p).clamp_min(0.0)
        tgt_tqp_raw = tgt_tqhw.reshape(t_tgt, q_attn, p).clamp_min(0.0)
        valid_tq = valid_tqn.any(dim=2)

        target_mode = str(getattr(self, "query_attn_cam_target_mode", "prob")).lower()
        if target_mode == "binary_threshold":
            thr = float(getattr(self, "query_attn_cam_target_binary_threshold", 0.5))
            thr = max(0.0, min(1.0, thr))
            tgt_tqp_metric = (tgt_tqp_raw >= thr).to(torch.float32)
        else:
            tgt_tqp_metric = tgt_tqp_raw

        metric = str(getattr(self, "query_attn_cam_metric", "kl")).lower()
        dist_tq = torch.zeros((t_tgt, q_attn), device=pred_tqp_raw.device, dtype=torch.float32)
        if metric == "kl":
            pred_sum_tq1 = pred_tqp_raw.sum(dim=-1, keepdim=True)
            tgt_sum_tq1 = tgt_tqp_metric.sum(dim=-1, keepdim=True)
            valid_tq = valid_tq & (pred_sum_tq1[..., 0] > 0.0) & (tgt_sum_tq1[..., 0] > 0.0)
            pred_tqp = pred_tqp_raw / pred_sum_tq1.clamp_min(eps_v)
            tgt_tqp = tgt_tqp_metric / tgt_sum_tq1.clamp_min(eps_v)
            finite_tq = torch.isfinite(pred_tqp).all(dim=-1) & torch.isfinite(tgt_tqp).all(dim=-1)
            valid_tq = valid_tq & finite_tq
            log_pred = torch.log(pred_tqp.clamp_min(eps_v))
            log_tgt = torch.log(tgt_tqp.clamp_min(eps_v))
            dist_tq = (tgt_tqp * (log_tgt - log_pred)).sum(dim=-1)
        else:
            pred_den_tq1 = pred_tqp_raw.amax(dim=-1, keepdim=True)
            tgt_den_tq1 = tgt_tqp_metric.amax(dim=-1, keepdim=True)
            valid_tq = valid_tq & (pred_den_tq1[..., 0] > 0.0) & (tgt_den_tq1[..., 0] > 0.0)
            pred_tqp = pred_tqp_raw / pred_den_tq1.clamp_min(eps_v)
            tgt_tqp = tgt_tqp_metric / tgt_den_tq1.clamp_min(eps_v)
            finite_tq = torch.isfinite(pred_tqp).all(dim=-1) & torch.isfinite(tgt_tqp).all(dim=-1)
            valid_tq = valid_tq & finite_tq

            if metric in ("bce", "bce_dice"):
                bce_tqp = F.binary_cross_entropy(
                    pred_tqp.clamp(min=eps_v, max=(1.0 - eps_v)),
                    tgt_tqp.clamp(min=0.0, max=1.0),
                    reduction="none",
                )
                bce_tq = bce_tqp.mean(dim=-1)
            else:
                bce_tq = torch.zeros_like(dist_tq)
            if metric in ("dice", "bce_dice"):
                inter_tq = (pred_tqp * tgt_tqp).sum(dim=-1)
                denom_tq = pred_tqp.sum(dim=-1) + tgt_tqp.sum(dim=-1)
                dice_tq = (2.0 * inter_tq + eps_v) / (denom_tq + eps_v)
                dice_dist_tq = 1.0 - dice_tq
            else:
                dice_dist_tq = torch.zeros_like(dist_tq)

            if metric == "bce":
                dist_tq = bce_tq
            elif metric == "dice":
                dist_tq = dice_dist_tq
            else:  # bce_dice
                dist_tq = 0.5 * (bce_tq + dice_dist_tq)

        finite_dist_tq = torch.isfinite(dist_tq)
        valid_tq = valid_tq & finite_dist_tq
        dist_tq = torch.where(valid_tq, dist_tq, torch.zeros_like(dist_tq))
        valid_count = valid_tq.to(torch.float32).sum()
        out["dbg_query_attn_cam_valid_pair_count"] = valid_count
        if not bool((valid_count > 0).item()):
            out["cam_attn_score_q"] = score_q
            return out

        valid_f_tq = valid_tq.to(torch.float32)
        valid_q = valid_f_tq.sum(dim=0) > 0.0
        dist_sum_q = (dist_tq * valid_f_tq).sum(dim=0)
        dist_mean_q = torch.zeros((q_attn,), device=dist_sum_q.device, dtype=torch.float32)
        dist_mean_q[valid_q] = dist_sum_q[valid_q] / valid_f_tq.sum(dim=0)[valid_q].clamp_min(1.0)

        score_q = torch.zeros((q_attn,), device=dist_mean_q.device, dtype=torch.float32)
        norm_mode = str(getattr(self, "query_attn_cam_score_norm_mode", "exp_neg")).lower()
        if norm_mode == "inv_1plus":
            score_q[valid_q] = (1.0 / (1.0 + dist_mean_q[valid_q].clamp_min(0.0))).clamp(0.0, 1.0)
        elif norm_mode == "one_minus":
            score_q[valid_q] = (1.0 - dist_mean_q[valid_q]).clamp(0.0, 1.0)
        else:
            score_q[valid_q] = torch.exp(-dist_mean_q[valid_q]).clamp(0.0, 1.0)
        out["cam_attn_score_q"] = score_q
        out["cam_attn_score_valid_q"] = valid_q
        out["dbg_query_attn_cam_metric_raw_mean"] = (
            (dist_tq * valid_f_tq).sum() / valid_count.clamp_min(1.0)
        )
        out["dbg_query_attn_cam_score_mean"] = (
            score_q[valid_q].mean() if bool(valid_q.any().item()) else z.new_tensor(0.0)
        )
        if not bool(torch.isfinite(score_q).all().item()):
            out["dbg_query_attn_cam_nonfinite_skip"] = z.new_tensor(1.0)
            out["cam_attn_score_q"] = torch.where(
                torch.isfinite(score_q), score_q, torch.zeros_like(score_q)
            )
            out["cam_attn_score_valid_q"] = valid_q & torch.isfinite(score_q)
        return out

    def _compute_query_self_bev_align_loss(
        self,
        query_future_feat_tqd: torch.Tensor,
        query_bev_feat_tqd: torch.Tensor = None,
        query_bev_valid_tq: torch.Tensor = None,
        loss_weight: float = 1.0,
        loss_type: str = "l2",
    ) -> dict:
        """
        Self alignment between:
        - query future feature (from query tokens)
        - BEV feature pooled at the same predicted center/gaussian

        Default uses normalized L2 for scale-invariant feature matching.
        """
        if torch.is_tensor(query_future_feat_tqd):
            z = query_future_feat_tqd.sum() * 0.0
        elif torch.is_tensor(query_bev_feat_tqd):
            z = query_bev_feat_tqd.sum() * 0.0
        else:
            z = self.mean_weight.sum() * 0.0

        out = {
            "loss_query_self_bev_align": z,
            "dbg_query_self_bev_align_valid_count": z,
            "dbg_query_self_bev_align_cos_mean": z,
        }
        if (not torch.is_tensor(query_future_feat_tqd)) or query_future_feat_tqd.dim() != 3:
            return out
        if (not torch.is_tensor(query_bev_feat_tqd)) or query_bev_feat_tqd.dim() != 3:
            return out

        Tq, Qq, Dq = [int(v) for v in query_future_feat_tqd.shape]
        Tb, Qb, Db = [int(v) for v in query_bev_feat_tqd.shape]
        if (Qq != Qb) or (Dq != Db):
            return out
        T = min(Tq, Tb)
        if T <= 0:
            return out

        q_feat_tqd = query_future_feat_tqd[:T].to(torch.float32)
        b_feat_tqd = query_bev_feat_tqd[:T].to(
            device=query_future_feat_tqd.device, dtype=torch.float32
        )
        valid_tq = torch.ones((T, Qq), device=query_future_feat_tqd.device, dtype=torch.bool)
        if (
            torch.is_tensor(query_bev_valid_tq)
            and query_bev_valid_tq.dim() == 2
            and int(query_bev_valid_tq.shape[0]) >= T
            and int(query_bev_valid_tq.shape[1]) == Qq
        ):
            valid_tq = valid_tq & query_bev_valid_tq[:T].to(
                device=query_future_feat_tqd.device, dtype=torch.bool
            )
        valid_tq = valid_tq & torch.isfinite(q_feat_tqd).all(dim=-1) & torch.isfinite(b_feat_tqd).all(dim=-1)

        valid_f = valid_tq.to(torch.float32)
        valid_cnt = valid_f.sum()
        out["dbg_query_self_bev_align_valid_count"] = valid_cnt
        if not bool((valid_cnt > 0).item()):
            return out

        qn = F.normalize(q_feat_tqd, dim=-1, eps=1e-6)
        bn = F.normalize(b_feat_tqd, dim=-1, eps=1e-6)
        cos_tq = F.cosine_similarity(qn, bn, dim=-1, eps=1e-6).clamp(-1.0, 1.0)
        out["dbg_query_self_bev_align_cos_mean"] = self._safe_weighted_mean(
            cos_tq.to(torch.float32), valid_f
        )

        lt = str(loss_type).lower()
        if lt == "l1":
            dist_tq = torch.abs(qn - bn).mean(dim=-1)
        else:
            dist_tq = (qn - bn).pow(2).mean(dim=-1)
        loss_raw = (dist_tq * valid_f).sum() / valid_cnt.clamp_min(1.0)
        out["loss_query_self_bev_align"] = loss_raw * float(loss_weight)
        return out

    def _compute_query_self_bev_cost_tq(
        self,
        query_future_feat_tqd: torch.Tensor,
        query_bev_feat_tqd: torch.Tensor,
        query_bev_valid_tq: torch.Tensor = None,
    ):
        """
        Per-query self-BEV mismatch cost used for Hungarian routing.
        Returns:
            self_cost_tq: [T,Q] in [0,2], cosine distance (1-cos)
            valid_tq: [T,Q] bool
        """
        if (not torch.is_tensor(query_future_feat_tqd)) or query_future_feat_tqd.dim() != 3:
            return None, None
        if (not torch.is_tensor(query_bev_feat_tqd)) or query_bev_feat_tqd.dim() != 3:
            return None, None

        Tq, Qq, Dq = [int(v) for v in query_future_feat_tqd.shape]
        Tb, Qb, Db = [int(v) for v in query_bev_feat_tqd.shape]
        if (Qq != Qb) or (Dq != Db):
            return None, None
        T = min(Tq, Tb)
        if T <= 0:
            return None, None

        q_feat_tqd = query_future_feat_tqd[:T].to(torch.float32)
        b_feat_tqd = query_bev_feat_tqd[:T].to(device=query_future_feat_tqd.device, dtype=torch.float32)
        valid_tq = torch.ones((T, Qq), device=query_future_feat_tqd.device, dtype=torch.bool)
        if (
            torch.is_tensor(query_bev_valid_tq)
            and query_bev_valid_tq.dim() == 2
            and int(query_bev_valid_tq.shape[0]) >= T
            and int(query_bev_valid_tq.shape[1]) == Qq
        ):
            valid_tq = valid_tq & query_bev_valid_tq[:T].to(device=query_future_feat_tqd.device, dtype=torch.bool)
        valid_tq = valid_tq & torch.isfinite(q_feat_tqd).all(dim=-1) & torch.isfinite(b_feat_tqd).all(dim=-1)

        qn = F.normalize(q_feat_tqd, dim=-1, eps=1e-6)
        bn = F.normalize(b_feat_tqd, dim=-1, eps=1e-6)
        cos_tq = F.cosine_similarity(qn, bn, dim=-1, eps=1e-6).clamp(-1.0, 1.0)
        self_cost_tq = (1.0 - cos_tq).clamp(0.0, 2.0)
        return self_cost_tq, valid_tq

    def _prepare_single_matched_pair_lowres_occ(
        self,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        gt_occ_txyz: torch.Tensor,
        q_idx: int,
        inst_id: int,
        valid_frame_t: torch.Tensor,
        objectness_scores_tq: torch.Tensor = None,
    ):
        """
        Build one matched query-GT pair occupancy tensors on the same low-res grid.
        """
        if (
            (not torch.is_tensor(centers_world_tq3))
            or centers_world_tq3.dim() != 3
            or (not torch.is_tensor(sigmas_world_tq3))
            or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape)
            or (not torch.is_tensor(gt_occ_txyz))
            or gt_occ_txyz.dim() != 4
        ):
            return None, None, None
        t_count = int(min(centers_world_tq3.shape[0], gt_occ_txyz.shape[0]))
        q_count = int(centers_world_tq3.shape[1])
        if (t_count <= 0) or (q_idx < 0) or (q_idx >= q_count):
            return None, None, None
        if (not torch.is_tensor(valid_frame_t)) or int(valid_frame_t.numel()) < t_count:
            valid_frame_t = torch.ones((t_count,), device=centers_world_tq3.device, dtype=torch.bool)
        else:
            valid_frame_t = valid_frame_t[:t_count].to(device=centers_world_tq3.device, dtype=torch.bool)
        if not bool(valid_frame_t.any().item()):
            return None, None, None

        voxel_weight_t1 = None
        if (
            torch.is_tensor(objectness_scores_tq)
            and objectness_scores_tq.dim() == 2
            and int(objectness_scores_tq.shape[0]) >= t_count
            and int(objectness_scores_tq.shape[1]) == q_count
        ):
            voxel_weight_t1 = objectness_scores_tq[:t_count, q_idx:q_idx + 1].to(
                device=centers_world_tq3.device,
                dtype=torch.float32,
            ).clamp(0.0, 1.0)

        pred_k_t1zyx = self.matched_gmo_voxelizer(
            centers_world_tq3[:t_count, q_idx:q_idx + 1, :],
            sigmas_world=sigmas_world_tq3[:t_count, q_idx:q_idx + 1, :],
            weights=voxel_weight_t1,
        ).to(torch.float32)
        gt_k_t1zyx = (gt_occ_txyz[:t_count] == int(inst_id)).permute(0, 3, 2, 1).unsqueeze(1).to(torch.float32)
        if tuple(gt_k_t1zyx.shape[-3:]) != tuple(pred_k_t1zyx.shape[-3:]):
            gt_k_t1zyx = F.adaptive_max_pool3d(gt_k_t1zyx, output_size=pred_k_t1zyx.shape[-3:])
        return pred_k_t1zyx, gt_k_t1zyx, valid_frame_t

    def _prepare_grouped_matched_pair_lowres_occ(
        self,
        gt_occ_txyz: torch.Tensor,
        matched_query_idx: torch.Tensor,
        pair_gt_ids_k: torch.Tensor,
        gt_valid_sel_tk: torch.Tensor,
        mixture_centers_world_tqg3: torch.Tensor,
        mixture_sigmas_world_tqg3: torch.Tensor,
        mixture_quat_tqg4: torch.Tensor,
        mixture_weights_tqg: torch.Tensor,
        objectness_scores_tq: torch.Tensor = None,
        pair_chunk_size: int = 8,
    ):
        if (
            (not torch.is_tensor(gt_occ_txyz))
            or gt_occ_txyz.dim() != 4
            or (not torch.is_tensor(matched_query_idx))
            or matched_query_idx.dim() != 1
            or (not torch.is_tensor(pair_gt_ids_k))
            or pair_gt_ids_k.dim() != 1
            or (not torch.is_tensor(gt_valid_sel_tk))
            or gt_valid_sel_tk.dim() != 2
            or (not torch.is_tensor(mixture_centers_world_tqg3))
            or mixture_centers_world_tqg3.dim() != 4
            or (not torch.is_tensor(mixture_sigmas_world_tqg3))
            or tuple(mixture_sigmas_world_tqg3.shape) != tuple(mixture_centers_world_tqg3.shape)
            or (not torch.is_tensor(mixture_quat_tqg4))
            or (not torch.is_tensor(mixture_weights_tqg))
            or tuple(mixture_quat_tqg4.shape[:3]) != tuple(mixture_weights_tqg.shape)
            or tuple(mixture_quat_tqg4.shape[:3]) != tuple(mixture_centers_world_tqg3.shape[:3])
        ):
            return None, None, None

        T = int(min(gt_occ_txyz.shape[0], mixture_centers_world_tqg3.shape[0], gt_valid_sel_tk.shape[0]))
        K = int(matched_query_idx.numel())
        if T <= 0 or K <= 0 or int(pair_gt_ids_k.numel()) != K:
            return None, None, None

        mq = matched_query_idx.to(device=mixture_centers_world_tqg3.device, dtype=torch.long)
        q_count = int(mixture_centers_world_tqg3.shape[1])
        if q_count <= 0:
            return None, None, None
        if not bool(((mq >= 0) & (mq < q_count)).all().item()):
            return None, None, None

        mix_centers_tkg3 = mixture_centers_world_tqg3[:T].index_select(1, mq).to(torch.float32)
        mix_sigmas_tkg3 = mixture_sigmas_world_tqg3[:T].index_select(1, mq).to(torch.float32)
        mix_quat_tkg4 = mixture_quat_tqg4[:T].index_select(1, mq).to(torch.float32)
        mix_weights_tkg = mixture_weights_tqg[:T].index_select(1, mq).to(torch.float32)
        valid_tk = gt_valid_sel_tk[:T].to(device=mixture_centers_world_tqg3.device, dtype=torch.bool)

        pair_weights_tk = None
        if (
            torch.is_tensor(objectness_scores_tq)
            and objectness_scores_tq.dim() == 2
            and int(objectness_scores_tq.shape[0]) >= T
            and int(objectness_scores_tq.shape[1]) >= q_count
        ):
            pair_weights_tk = objectness_scores_tq[:T].to(device=mixture_centers_world_tqg3.device, dtype=torch.float32).index_select(1, mq)

        pred_tk1zyx = self.matched_gmo_voxelizer.forward_gaussian_mixture_grouped(
            mixture_centers_world_tkg3=mix_centers_tkg3,
            mixture_sigmas_world_tkg3=mix_sigmas_tkg3,
            mixture_weights_tkg=mix_weights_tkg,
            mixture_quat_tkg4=mix_quat_tkg4,
            pair_weights_tk=pair_weights_tk,
            pair_chunk_size=int(pair_chunk_size),
        ).to(torch.float32)

        gt_ids = pair_gt_ids_k.to(device=mixture_centers_world_tqg3.device, dtype=torch.long)
        gt_occ_sel = gt_occ_txyz[:T]
        pred_spatial = tuple(int(v) for v in pred_tk1zyx.shape[-3:])
        full_spatial = (int(gt_occ_sel.shape[3]), int(gt_occ_sel.shape[2]), int(gt_occ_sel.shape[1]))
        gt_voxels_per_pair = int(gt_occ_sel.numel())
        # Avoid materializing the full [T,K,X,Y,Z] one-vs-instance tensor on GPU.
        # Keep chunk peak around ~128M boolean elements, capped by pair_chunk_size.
        max_compare_elems = 128 * 1024 * 1024
        gt_chunk_size = max(1, min(int(pair_chunk_size), int(max_compare_elems // max(gt_voxels_per_pair, 1))))
        pool_dtype = torch.float16 if gt_occ_sel.is_cuda else torch.float32
        gt_chunks = []
        for k0 in range(0, K, gt_chunk_size):
            k1 = min(K, k0 + gt_chunk_size)
            kc = k1 - k0
            if kc <= 0:
                continue
            gt_ids_chunk = gt_ids[k0:k1]
            gt_chunk_tkxyz = (gt_occ_sel[:, None] == gt_ids_chunk[None, :, None, None, None])
            gt_chunk_tk1zyx = gt_chunk_tkxyz.permute(0, 1, 4, 3, 2).unsqueeze(2)
            if full_spatial != pred_spatial:
                gt_chunk_lowres = F.adaptive_max_pool3d(
                    gt_chunk_tk1zyx.reshape(T * kc, 1, *full_spatial).to(pool_dtype),
                    output_size=pred_spatial,
                ).reshape(T, kc, 1, *pred_spatial)
            else:
                gt_chunk_lowres = gt_chunk_tk1zyx.to(pool_dtype)
            gt_chunks.append(gt_chunk_lowres.to(torch.float32))
        if len(gt_chunks) != int((K + gt_chunk_size - 1) // gt_chunk_size):
            return None, None, None
        gt_tk1zyx = torch.cat(gt_chunks, dim=1) if len(gt_chunks) > 1 else gt_chunks[0]
        return pred_tk1zyx, gt_tk1zyx, valid_tk

    def _prepare_local_aabb_matched_pair_occ(
        self,
        gt_occ_txyz: torch.Tensor,
        bbox_occ_txyz: torch.Tensor,
        matched_query_idx: torch.Tensor,
        pair_gt_ids_k: torch.Tensor,
        gt_valid_sel_tk: torch.Tensor,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        objectness_scores_tq: torch.Tensor = None,
        crop_size=(32, 32, 12),
        crop_scale: float = 1.0,
        crop_margin_m=(0.0, 0.0, 0.0),
        eps: float = 1e-6,
    ):
        if (
            (not torch.is_tensor(gt_occ_txyz))
            or gt_occ_txyz.dim() != 4
            or (not torch.is_tensor(matched_query_idx))
            or matched_query_idx.dim() != 1
            or (not torch.is_tensor(pair_gt_ids_k))
            or pair_gt_ids_k.dim() != 1
            or (not torch.is_tensor(gt_valid_sel_tk))
            or gt_valid_sel_tk.dim() != 2
            or (not torch.is_tensor(centers_world_tq3))
            or centers_world_tq3.dim() != 3
        ):
            return None, None, None, None

        crop_x, crop_y, crop_z = [int(v) for v in crop_size]
        if crop_x <= 0 or crop_y <= 0 or crop_z <= 0:
            return None, None, None, None
        bbox_t = int(bbox_occ_txyz.shape[0]) if torch.is_tensor(bbox_occ_txyz) and bbox_occ_txyz.dim() == 4 else int(gt_occ_txyz.shape[0])
        T = int(min(gt_occ_txyz.shape[0], bbox_t, centers_world_tq3.shape[0], gt_valid_sel_tk.shape[0]))
        K = int(matched_query_idx.numel())
        if T <= 0 or K <= 0 or int(pair_gt_ids_k.numel()) != K:
            return None, None, None, None

        device = centers_world_tq3.device
        mq = matched_query_idx.to(device=device, dtype=torch.long)
        q_count = int(centers_world_tq3.shape[1])
        if q_count <= 0 or not bool(((mq >= 0) & (mq < q_count)).all().item()):
            return None, None, None, None

        use_mixture = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_quat_tqg4)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_quat_tqg4.shape[:3])
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
            and int(mixture_centers_world_tqg3.shape[0]) >= T
            and int(mixture_centers_world_tqg3.shape[1]) == q_count
        )
        if (not use_mixture) and (
            (not torch.is_tensor(sigmas_world_tq3))
            or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape)
        ):
            return None, None, None, None

        gt_occ = gt_occ_txyz[:T].to(device=device, dtype=torch.long)
        bbox_occ = bbox_occ_txyz[:T].to(device=device, dtype=torch.long) if torch.is_tensor(bbox_occ_txyz) and bbox_occ_txyz.dim() == 4 else gt_occ
        valid_tk = gt_valid_sel_tk[:T].to(device=device, dtype=torch.bool)
        pair_gt_ids = pair_gt_ids_k.to(device=device, dtype=torch.long)

        pc = torch.as_tensor(self.point_cloud_range, device=device, dtype=torch.float32)
        pc_min = pc[:3]
        pc_max = pc[3:]
        full_size = torch.tensor(
            [int(gt_occ.shape[1]), int(gt_occ.shape[2]), int(gt_occ.shape[3])],
            device=device,
            dtype=torch.float32,
        )
        fine_voxel = (pc_max - pc_min) / full_size.clamp_min(1.0)
        crop_size_t = torch.tensor([crop_x, crop_y, crop_z], device=device, dtype=torch.float32)
        margin_t = torch.as_tensor(crop_margin_m, device=device, dtype=torch.float32)
        if margin_t.numel() == 1:
            margin_t = margin_t.expand(3)
        margin_t = margin_t.reshape(3).clamp_min(0.0)
        scale = max(float(crop_scale), float(eps))

        x_alpha = ((torch.arange(crop_x, device=device, dtype=torch.float32) + 0.5) / float(crop_x)) - 0.5
        y_alpha = ((torch.arange(crop_y, device=device, dtype=torch.float32) + 0.5) / float(crop_y)) - 0.5
        z_alpha = ((torch.arange(crop_z, device=device, dtype=torch.float32) + 0.5) / float(crop_z)) - 0.5
        trunc = max(1.0, float(getattr(self.matched_gmo_voxelizer, "gaussian_truncate_sigma", 3.0)))
        trunc2 = trunc * trunc
        sigma_floor_vox = float(getattr(self.matched_gmo_voxelizer, "gaussian_sigma_floor_vox", 0.35))

        pred_tk1zyx = centers_world_tq3.new_zeros((T, K, 1, crop_z, crop_y, crop_x), dtype=torch.float32)
        gt_tk1zyx = pred_tk1zyx.new_zeros((T, K, 1, crop_z, crop_y, crop_x))
        valid_tk1zyx = torch.zeros((T, K, 1, crop_z, crop_y, crop_x), device=device, dtype=torch.bool)
        valid_pair_tk = torch.zeros((T, K), device=device, dtype=torch.bool)

        pair_weights_tq = None
        if (
            torch.is_tensor(objectness_scores_tq)
            and objectness_scores_tq.dim() == 2
            and int(objectness_scores_tq.shape[0]) >= T
            and int(objectness_scores_tq.shape[1]) >= q_count
        ):
            pair_weights_tq = objectness_scores_tq[:T].to(device=device, dtype=torch.float32)

        for t in range(T):
            bbox_frame = bbox_occ[t]
            gt_frame = gt_occ[t]
            for k in range(K):
                if not bool(valid_tk[t, k].item()):
                    continue
                inst_id = int(pair_gt_ids[k].item())
                if inst_id <= 0:
                    continue
                pts = torch.nonzero(bbox_frame == inst_id, as_tuple=False)
                if int(pts.numel()) <= 0:
                    continue

                mins = pts.min(dim=0).values.to(torch.float32)
                maxs = pts.max(dim=0).values.to(torch.float32) + 1.0
                gt_min = pc_min + mins * fine_voxel
                gt_max = pc_min + maxs * fine_voxel
                gt_center = 0.5 * (gt_min + gt_max)
                crop_extent = torch.maximum((gt_max - gt_min) * scale + 2.0 * margin_t, fine_voxel)
                crop_voxel = crop_extent / crop_size_t

                gt_x = gt_center[0] + x_alpha * crop_extent[0]
                gt_y = gt_center[1] + y_alpha * crop_extent[1]
                gt_z = gt_center[2] + z_alpha * crop_extent[2]
                ix = torch.floor((gt_x - pc_min[0]) / fine_voxel[0]).to(torch.long)
                iy = torch.floor((gt_y - pc_min[1]) / fine_voxel[1]).to(torch.long)
                iz = torch.floor((gt_z - pc_min[2]) / fine_voxel[2]).to(torch.long)
                in_x = (ix >= 0) & (ix < int(gt_occ.shape[1]))
                in_y = (iy >= 0) & (iy < int(gt_occ.shape[2]))
                in_z = (iz >= 0) & (iz < int(gt_occ.shape[3]))
                in_zyx = in_z[:, None, None] & in_y[None, :, None] & in_x[None, None, :]
                if not bool(in_zyx.any().item()):
                    continue

                ix = ix.clamp(0, int(gt_occ.shape[1]) - 1)
                iy = iy.clamp(0, int(gt_occ.shape[2]) - 1)
                iz = iz.clamp(0, int(gt_occ.shape[3]) - 1)
                gt_crop = gt_frame[
                    ix[None, None, :],
                    iy[None, :, None],
                    iz[:, None, None],
                ] == inst_id
                gt_tk1zyx[t, k, 0] = gt_crop.to(torch.float32) * in_zyx.to(torch.float32)
                valid_tk1zyx[t, k, 0] = in_zyx
                valid_pair_tk[t, k] = True

                q_idx = int(mq[k].item())
                pred_center = centers_world_tq3[t, q_idx].to(torch.float32)
                pred_x = pred_center[0] + x_alpha * crop_extent[0]
                pred_y = pred_center[1] + y_alpha * crop_extent[1]
                pred_z = pred_center[2] + z_alpha * crop_extent[2]
                zz, yy, xx = torch.meshgrid(pred_z, pred_y, pred_x, indexing="ij")

                if use_mixture:
                    comp_centers = mixture_centers_world_tqg3[t, q_idx].to(device=device, dtype=torch.float32)
                    comp_sigmas = mixture_sigmas_world_tqg3[t, q_idx].to(device=device, dtype=torch.float32)
                    comp_rot = quat_to_rotmat_wxyz(mixture_quat_tqg4[t, q_idx].to(device=device, dtype=torch.float32))
                    comp_weights = mixture_weights_tqg[t, q_idx].to(device=device, dtype=torch.float32).clamp_min(0.0)
                else:
                    comp_centers = centers_world_tq3[t, q_idx:q_idx + 1].to(torch.float32)
                    comp_sigmas = sigmas_world_tq3[t, q_idx:q_idx + 1].to(torch.float32)
                    comp_rot = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3)
                    comp_weights = comp_centers.new_ones((1,))

                sigma_floor = (crop_voxel * sigma_floor_vox).clamp_min(float(eps))
                comp_sigmas = torch.maximum(comp_sigmas, sigma_floor[None])
                dx = xx[None] - comp_centers[:, 0, None, None, None]
                dy = yy[None] - comp_centers[:, 1, None, None, None]
                dz = zz[None] - comp_centers[:, 2, None, None, None]
                xr = (
                    comp_rot[:, 0, 0, None, None, None] * dx
                    + comp_rot[:, 1, 0, None, None, None] * dy
                    + comp_rot[:, 2, 0, None, None, None] * dz
                )
                yr = (
                    comp_rot[:, 0, 1, None, None, None] * dx
                    + comp_rot[:, 1, 1, None, None, None] * dy
                    + comp_rot[:, 2, 1, None, None, None] * dz
                )
                zr = (
                    comp_rot[:, 0, 2, None, None, None] * dx
                    + comp_rot[:, 1, 2, None, None, None] * dy
                    + comp_rot[:, 2, 2, None, None, None] * dz
                )
                md2 = (
                    (xr / comp_sigmas[:, 0, None, None, None]).pow(2)
                    + (yr / comp_sigmas[:, 1, None, None, None]).pow(2)
                    + (zr / comp_sigmas[:, 2, None, None, None]).pow(2)
                )
                gauss = torch.exp(-0.5 * md2) * (md2 <= trunc2).to(md2.dtype)
                lam = (comp_weights[:, None, None, None] * gauss).sum(dim=0)
                pred_crop = (-torch.expm1(-lam)).clamp(0.0, 1.0)
                if pair_weights_tq is not None:
                    pred_crop = pred_crop * pair_weights_tq[t, q_idx].clamp(min=0.0)
                pred_tk1zyx[t, k, 0] = pred_crop

        return pred_tk1zyx, gt_tk1zyx, valid_tk1zyx, valid_pair_tk

    def _prepare_local_aabb_matched_pair_occ_pairs(
        self,
        gt_occ_txyz: torch.Tensor,
        bbox_occ_txyz: torch.Tensor,
        matched_query_idx: torch.Tensor,
        pair_gt_ids_k: torch.Tensor,
        gt_valid_sel_tk: torch.Tensor,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        objectness_scores_tq: torch.Tensor = None,
        crop_size=None,
        crop_scale: float = 1.0,
        crop_margin_vox=(0.0, 0.0, 0.0),
        eps: float = 1e-6,
    ):
        if (
            (not torch.is_tensor(gt_occ_txyz))
            or gt_occ_txyz.dim() != 4
            or (not torch.is_tensor(bbox_occ_txyz))
            or bbox_occ_txyz.dim() != 4
            or (not torch.is_tensor(matched_query_idx))
            or matched_query_idx.dim() != 1
            or (not torch.is_tensor(pair_gt_ids_k))
            or pair_gt_ids_k.dim() != 1
            or (not torch.is_tensor(gt_valid_sel_tk))
            or gt_valid_sel_tk.dim() != 2
            or (not torch.is_tensor(centers_world_tq3))
            or centers_world_tq3.dim() != 3
        ):
            return None, None, None, 0, None, None

        fixed_crop = None
        if crop_size is not None:
            fixed_crop = tuple(int(v) for v in crop_size)
            if len(fixed_crop) != 3 or min(fixed_crop) <= 0:
                return None, None, None, 0, None, None

        bbox_t = int(bbox_occ_txyz.shape[0])
        T = int(min(gt_occ_txyz.shape[0], bbox_t, centers_world_tq3.shape[0], gt_valid_sel_tk.shape[0]))
        K = int(matched_query_idx.numel())
        if T <= 0 or K <= 0 or int(pair_gt_ids_k.numel()) != K:
            return None, None, None, 0, None, None

        device = centers_world_tq3.device
        mq = matched_query_idx.to(device=device, dtype=torch.long)
        q_count = int(centers_world_tq3.shape[1])
        if q_count <= 0 or not bool(((mq >= 0) & (mq < q_count)).all().item()):
            return None, None, None, 0, None, None

        use_mixture = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_quat_tqg4)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_quat_tqg4.shape[:3])
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
            and int(mixture_centers_world_tqg3.shape[0]) >= T
            and int(mixture_centers_world_tqg3.shape[1]) == q_count
        )
        if (not use_mixture) and (
            (not torch.is_tensor(sigmas_world_tq3))
            or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape)
        ):
            return None, None, None, 0, None, None

        gt_occ = gt_occ_txyz[:T].to(device=device, dtype=torch.long)
        bbox_occ = bbox_occ_txyz[:T].to(device=device, dtype=torch.long)
        valid_tk = gt_valid_sel_tk[:T].to(device=device, dtype=torch.bool)
        pair_gt_ids = pair_gt_ids_k.to(device=device, dtype=torch.long)

        pc = torch.as_tensor(self.point_cloud_range, device=device, dtype=torch.float32)
        full_size = torch.tensor(
            [int(gt_occ.shape[1]), int(gt_occ.shape[2]), int(gt_occ.shape[3])],
            device=device,
            dtype=torch.float32,
        )
        fine_voxel = (pc[3:] - pc[:3]) / full_size.clamp_min(1.0)
        margin_vox_t = torch.as_tensor(crop_margin_vox, device=device, dtype=torch.float32)
        if margin_vox_t.numel() == 1:
            margin_vox_t = margin_vox_t.expand(3)
        margin_vox_t = margin_vox_t.reshape(3).clamp_min(0.0)
        scale = max(float(crop_scale), float(eps))
        trunc = max(1.0, float(getattr(self.matched_gmo_voxelizer, "gaussian_truncate_sigma", 3.0)))
        trunc2 = trunc * trunc
        sigma_floor_vox = float(getattr(self.matched_gmo_voxelizer, "gaussian_sigma_floor_vox", 0.35))

        pair_weights_tq = None
        if (
            torch.is_tensor(objectness_scores_tq)
            and objectness_scores_tq.dim() == 2
            and int(objectness_scores_tq.shape[0]) >= T
            and int(objectness_scores_tq.shape[1]) >= q_count
        ):
            pair_weights_tq = objectness_scores_tq[:T].to(device=device, dtype=torch.float32)

        pred_pairs, gt_pairs, valid_pairs, pair_meta = [], [], [], []
        crop_shape_sum = centers_world_tq3.new_zeros((3,), dtype=torch.float32)
        valid_count = 0
        for k in range(K):
            q_idx = int(mq[k].item())
            inst_id = int(pair_gt_ids[k].item())
            if inst_id <= 0:
                continue
            frames = []
            for t in range(T):
                if not bool(valid_tk[t, k].item()):
                    continue
                pts = torch.nonzero(bbox_occ[t] == inst_id, as_tuple=False)
                if int(pts.numel()) <= 0:
                    continue
                mins = pts.min(dim=0).values.to(torch.float32)
                maxs = pts.max(dim=0).values.to(torch.float32) + 1.0
                bbox_dims = (maxs - mins).clamp_min(1.0)
                crop_extent_vox = bbox_dims * scale + 2.0 * margin_vox_t
                if fixed_crop is None:
                    crop_dims = torch.ceil(crop_extent_vox).to(torch.long).clamp(min=1)
                else:
                    crop_dims = torch.tensor(fixed_crop, device=device, dtype=torch.long)
                crop_x, crop_y, crop_z = [int(v) for v in crop_dims.tolist()]
                crop_shape_sum = crop_shape_sum + crop_dims.to(dtype=torch.float32)
                valid_count += 1

                center_grid = 0.5 * (mins + maxs)
                x_alpha = ((torch.arange(crop_x, device=device, dtype=torch.float32) + 0.5) / float(crop_x)) - 0.5
                y_alpha = ((torch.arange(crop_y, device=device, dtype=torch.float32) + 0.5) / float(crop_y)) - 0.5
                z_alpha = ((torch.arange(crop_z, device=device, dtype=torch.float32) + 0.5) / float(crop_z)) - 0.5
                off_x = x_alpha * crop_extent_vox[0]
                off_y = y_alpha * crop_extent_vox[1]
                off_z = z_alpha * crop_extent_vox[2]

                gx = center_grid[0] + off_x
                gy = center_grid[1] + off_y
                gz = center_grid[2] + off_z
                ix = torch.floor(gx).to(torch.long)
                iy = torch.floor(gy).to(torch.long)
                iz = torch.floor(gz).to(torch.long)
                in_x = (ix >= 0) & (ix < int(gt_occ.shape[1]))
                in_y = (iy >= 0) & (iy < int(gt_occ.shape[2]))
                in_z = (iz >= 0) & (iz < int(gt_occ.shape[3]))
                in_zyx = in_z[:, None, None] & in_y[None, :, None] & in_x[None, None, :]
                if not bool(in_zyx.any().item()):
                    continue

                ix = ix.clamp(0, int(gt_occ.shape[1]) - 1)
                iy = iy.clamp(0, int(gt_occ.shape[2]) - 1)
                iz = iz.clamp(0, int(gt_occ.shape[3]) - 1)
                gt_crop = (gt_occ[t][ix[None, None, :], iy[None, :, None], iz[:, None, None]] == inst_id).to(torch.float32)
                gt_crop = gt_crop * in_zyx.to(torch.float32)

                pred_center = centers_world_tq3[t, q_idx].to(torch.float32)
                pred_x = pred_center[0] + off_x * fine_voxel[0]
                pred_y = pred_center[1] + off_y * fine_voxel[1]
                pred_z = pred_center[2] + off_z * fine_voxel[2]
                zz, yy, xx = torch.meshgrid(pred_z, pred_y, pred_x, indexing="ij")

                if use_mixture:
                    comp_centers = mixture_centers_world_tqg3[t, q_idx].to(device=device, dtype=torch.float32)
                    comp_sigmas = mixture_sigmas_world_tqg3[t, q_idx].to(device=device, dtype=torch.float32)
                    comp_rot = quat_to_rotmat_wxyz(mixture_quat_tqg4[t, q_idx].to(device=device, dtype=torch.float32))
                    comp_weights = mixture_weights_tqg[t, q_idx].to(device=device, dtype=torch.float32).clamp_min(0.0)
                else:
                    comp_centers = centers_world_tq3[t, q_idx:q_idx + 1].to(torch.float32)
                    comp_sigmas = sigmas_world_tq3[t, q_idx:q_idx + 1].to(torch.float32)
                    comp_rot = torch.eye(3, device=device, dtype=torch.float32).view(1, 3, 3)
                    comp_weights = comp_centers.new_ones((1,))

                step_world = (crop_extent_vox / crop_dims.to(torch.float32)) * fine_voxel
                sigma_floor = (step_world * sigma_floor_vox).clamp_min(float(eps))
                comp_sigmas = torch.maximum(comp_sigmas, sigma_floor[None])
                dx = xx[None] - comp_centers[:, 0, None, None, None]
                dy = yy[None] - comp_centers[:, 1, None, None, None]
                dz = zz[None] - comp_centers[:, 2, None, None, None]
                xr = (
                    comp_rot[:, 0, 0, None, None, None] * dx
                    + comp_rot[:, 1, 0, None, None, None] * dy
                    + comp_rot[:, 2, 0, None, None, None] * dz
                )
                yr = (
                    comp_rot[:, 0, 1, None, None, None] * dx
                    + comp_rot[:, 1, 1, None, None, None] * dy
                    + comp_rot[:, 2, 1, None, None, None] * dz
                )
                zr = (
                    comp_rot[:, 0, 2, None, None, None] * dx
                    + comp_rot[:, 1, 2, None, None, None] * dy
                    + comp_rot[:, 2, 2, None, None, None] * dz
                )
                md2 = (
                    (xr / comp_sigmas[:, 0, None, None, None]).pow(2)
                    + (yr / comp_sigmas[:, 1, None, None, None]).pow(2)
                    + (zr / comp_sigmas[:, 2, None, None, None]).pow(2)
                )
                gauss = torch.exp(-0.5 * md2) * (md2 <= trunc2).to(md2.dtype)
                pred_crop = (-torch.expm1(-(comp_weights[:, None, None, None] * gauss).sum(dim=0))).clamp(0.0, 1.0)
                if pair_weights_tq is not None:
                    pred_crop = pred_crop * pair_weights_tq[t, q_idx].clamp(min=0.0)
                frames.append((t, pred_crop, gt_crop, in_zyx))

            if len(frames) <= 0:
                continue
            max_z = max(int(v[1].shape[0]) for v in frames)
            max_y = max(int(v[1].shape[1]) for v in frames)
            max_x = max(int(v[1].shape[2]) for v in frames)
            pred_pair = centers_world_tq3.new_zeros((T, 1, max_z, max_y, max_x), dtype=torch.float32)
            gt_pair = pred_pair.new_zeros((T, 1, max_z, max_y, max_x))
            valid_pair = torch.zeros((T, 1, max_z, max_y, max_x), device=device, dtype=torch.bool)
            for t, pred_crop, gt_crop, valid_crop in frames:
                z, y, x = [int(v) for v in pred_crop.shape]
                pred_pair[t, 0, :z, :y, :x] = pred_crop
                gt_pair[t, 0, :z, :y, :x] = gt_crop
                valid_pair[t, 0, :z, :y, :x] = valid_crop
            pred_pairs.append(pred_pair)
            gt_pairs.append(gt_pair)
            valid_pairs.append(valid_pair)
            pair_meta.append({
                "pair_index": int(k),
                "query_idx": int(q_idx),
                "gt_instance_id": int(inst_id),
                "frames": [int(v[0]) for v in frames],
            })

        mean_shape = crop_shape_sum / float(max(valid_count, 1))
        return pred_pairs, gt_pairs, valid_pairs, valid_count, mean_shape, pair_meta

    @torch.no_grad()
    def _maybe_save_local_aabb_gmo_pair_vis(
        self,
        pred_pairs,
        gt_pairs,
        valid_pairs,
        pair_meta,
        step: int,
    ) -> None:
        vis_every = int(getattr(self, "debug_gmo_local_aabb_pair_vis_every", 0))
        if vis_every <= 0 or (int(step) % vis_every) != 0:
            return
        if (not getattr(self, "training", False)) or not self._is_main_process():
            return
        if not isinstance(pred_pairs, (list, tuple)) or len(pred_pairs) <= 0:
            return

        try:
            from PIL import Image, ImageDraw
            import numpy as np
        except Exception:
            return

        vis_dir = str(getattr(self, "debug_gmo_local_aabb_pair_vis_dir", "./work_dirs/gmo_local_aabb_pair_vis"))
        max_pairs = max(1, int(getattr(self, "debug_gmo_local_aabb_pair_vis_max_pairs", 8)))
        out_dir = os.path.join(vis_dir, f"iter_{int(step):06d}")
        os.makedirs(out_dir, exist_ok=True)

        sidecar = {"step": int(step), "pairs": []}
        selected = min(max_pairs, len(pred_pairs), len(gt_pairs), len(valid_pairs))
        for idx in range(selected):
            pred = pred_pairs[idx].detach().float().cpu()
            gt = gt_pairs[idx].detach().float().cpu()
            valid = valid_pairs[idx].detach().bool().cpu()
            meta = pair_meta[idx] if isinstance(pair_meta, (list, tuple)) and idx < len(pair_meta) else {}
            sidecar["pairs"].append({"meta": meta, "pred": pred, "gt": gt, "valid": valid})

            pred_bev = (pred[:, 0] * valid[:, 0].float()).amax(dim=1).numpy()
            gt_bev = (gt[:, 0] * valid[:, 0].float()).amax(dim=1).numpy()
            valid_bev = valid[:, 0].any(dim=1).numpy()
            t_count = int(pred_bev.shape[0])
            if t_count <= 0:
                continue
            h, w = int(pred_bev.shape[1]), int(pred_bev.shape[2])
            scale = max(1, min(8, 256 // max(h, w, 1)))
            canvas = np.zeros((t_count * h, 3 * w, 3), dtype=np.uint8)

            for t in range(t_count):
                p = np.clip(pred_bev[t], 0.0, 1.0)
                g = np.clip(gt_bev[t], 0.0, 1.0)
                v = valid_bev[t].astype(bool)
                pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                gt_rgb = np.zeros_like(pred_rgb)
                overlay = np.zeros_like(pred_rgb)
                pred_rgb[..., 0] = (p * 255).astype(np.uint8)
                gt_rgb[..., 1] = (g * 255).astype(np.uint8)
                overlay[..., 0] = (p * 255).astype(np.uint8)
                overlay[..., 1] = (g * 255).astype(np.uint8)
                pred_rgb[~v] = 32
                gt_rgb[~v] = 32
                overlay[~v] = 32
                row = slice(t * h, (t + 1) * h)
                canvas[row, 0:w] = gt_rgb
                canvas[row, w:2 * w] = pred_rgb
                canvas[row, 2 * w:3 * w] = overlay

            img = Image.fromarray(canvas, mode="RGB")
            if scale > 1:
                img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)
            draw = ImageDraw.Draw(img)
            draw.text(
                (4, 4),
                (
                    f"pair={idx} q={meta.get('query_idx', 'na')} "
                    f"gt_id={meta.get('gt_instance_id', 'na')} | GT green / pred red / overlay"
                ),
                fill=(255, 255, 255),
            )
            img.save(
                os.path.join(
                    out_dir,
                    f"pair_{idx:03d}_q{meta.get('query_idx', 'na')}_gt{meta.get('gt_instance_id', 'na')}.png",
                )
            )

        torch.save(sidecar, os.path.join(out_dir, "local_aabb_pairs.pt"))

    def _compute_matched_query_sequence_iou_scores(
        self,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        gt_instance_occ3d_txyz_pred: torch.Tensor,
        inst_match_result: dict = None,
        objectness_scores_tq: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        pair_chunk_size: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        if (not torch.is_tensor(gt_instance_occ3d_txyz_pred)) or gt_instance_occ3d_txyz_pred.dim() != 4:
            return None
        if not isinstance(inst_match_result, dict):
            return None

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_ids_n = inst_match_result.get("gt_ids_n", None)
        gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
        q_count = int(centers_world_tq3.shape[1])
        iou_q = centers_world_tq3.new_zeros((q_count,), dtype=torch.float32)
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
            or (not torch.is_tensor(gt_ids_n))
            or gt_ids_n.numel() <= 0
        ):
            return iou_q

        t_count = int(min(centers_world_tq3.shape[0], gt_instance_occ3d_txyz_pred.shape[0]))
        if t_count <= 0:
            return iou_q
        gt_occ_txyz = gt_instance_occ3d_txyz_pred[:t_count].to(device=centers_world_tq3.device, dtype=torch.long)
        mq = matched_query_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        mi = matched_inst_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        gt_ids = gt_ids_n.to(device=centers_world_tq3.device, dtype=torch.long)
        keep = (mq >= 0) & (mq < q_count) & (mi >= 0) & (mi < int(gt_ids.numel()))
        if not bool(keep.any().item()):
            return iou_q
        mq = mq[keep]
        mi = mi[keep]
        pair_gt_ids = gt_ids.index_select(0, mi)
        if torch.is_tensor(gt_valid_tn) and gt_valid_tn.dim() == 2 and int(gt_valid_tn.shape[0]) >= t_count:
            gt_valid_sel_tk = gt_valid_tn[:t_count].to(device=centers_world_tq3.device, dtype=torch.bool).index_select(1, mi)
        else:
            gt_valid_sel_tk = torch.ones((t_count, int(mq.numel())), device=centers_world_tq3.device, dtype=torch.bool)

        use_mixture = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_quat_tqg4)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_quat_tqg4.shape[:3])
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
        )
        if use_mixture:
            pred_tk1zyx, gt_tk1zyx, valid_tk = self._prepare_grouped_matched_pair_lowres_occ(
                gt_occ_txyz=gt_occ_txyz,
                matched_query_idx=mq,
                pair_gt_ids_k=pair_gt_ids,
                gt_valid_sel_tk=gt_valid_sel_tk,
                mixture_centers_world_tqg3=mixture_centers_world_tqg3[:t_count],
                mixture_sigmas_world_tqg3=mixture_sigmas_world_tqg3[:t_count],
                mixture_quat_tqg4=mixture_quat_tqg4[:t_count],
                mixture_weights_tqg=mixture_weights_tqg[:t_count],
                objectness_scores_tq=objectness_scores_tq,
                pair_chunk_size=int(pair_chunk_size),
            )
            if pred_tk1zyx is None:
                return iou_q
            pred_bin_tk = pred_tk1zyx[:, :, 0] > 0.5
            gt_bin_tk = gt_tk1zyx[:, :, 0] > 0.5
            inter_tk = (pred_bin_tk & gt_bin_tk).flatten(2).sum(dim=2).to(torch.float32)
            union_tk = (pred_bin_tk | gt_bin_tk).flatten(2).sum(dim=2).to(torch.float32)
            valid_iou_tk = valid_tk & (union_tk > 0.0)
            for k in range(int(mq.numel())):
                q_idx = int(mq[k].item())
                if not bool(valid_iou_tk[:, k].any().item()):
                    continue
                iou_t = inter_tk[:, k][valid_iou_tk[:, k]] / (union_tk[:, k][valid_iou_tk[:, k]] + float(eps))
                iou_q[q_idx] = iou_t.mean().clamp(0.0, 1.0)
            return iou_q

        if (not torch.is_tensor(sigmas_world_tq3)) or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape):
            return iou_q

        for k in range(int(mq.numel())):
            q_idx = int(mq[k].item())
            inst_id = int(pair_gt_ids[k].item())
            pred_k_t1zyx, gt_k_t1zyx, valid_frame_t = self._prepare_single_matched_pair_lowres_occ(
                centers_world_tq3=centers_world_tq3[:t_count],
                sigmas_world_tq3=sigmas_world_tq3[:t_count],
                gt_occ_txyz=gt_occ_txyz,
                q_idx=q_idx,
                inst_id=inst_id,
                valid_frame_t=gt_valid_sel_tk[:, k],
                objectness_scores_tq=objectness_scores_tq,
            )
            if pred_k_t1zyx is None:
                continue
            pred_bin_t = pred_k_t1zyx[:, 0] > 0.5
            gt_bin_t = gt_k_t1zyx[:, 0] > 0.5
            valid_t = valid_frame_t.to(torch.bool)
            inter_t = (pred_bin_t & gt_bin_t).flatten(1).sum(dim=1).to(torch.float32)
            union_t = (pred_bin_t | gt_bin_t).flatten(1).sum(dim=1).to(torch.float32)
            valid_iou_t = valid_t & (union_t > 0.0)
            if not bool(valid_iou_t.any().item()):
                continue
            iou_t = inter_t[valid_iou_t] / (union_t[valid_iou_t] + float(eps))
            iou_q[q_idx] = iou_t.mean().clamp(0.0, 1.0)
        return iou_q

    @staticmethod
    def _compute_balanced_binary_pair_loss(
        pred_occ: torch.Tensor,
        gt_occ: torch.Tensor,
        valid_mask: torch.Tensor,
        loss_type: str = "balanced_bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        eps: float = 1e-6,
    ):
        if (
            (not torch.is_tensor(pred_occ))
            or (not torch.is_tensor(gt_occ))
            or (not torch.is_tensor(valid_mask))
        ):
            return None
        p = pred_occ.to(torch.float32).clamp(float(eps), 1.0 - float(eps))
        t = gt_occ.to(device=p.device, dtype=torch.float32).clamp(0.0, 1.0)
        vm = valid_mask.to(device=p.device, dtype=torch.bool)
        if not bool(vm.any().item()):
            return None

        loss_type = str(loss_type).lower()
        if loss_type in ("balanced_bce", "bce"):
            loss_map = F.binary_cross_entropy(p, t, reduction="none")
        elif loss_type in ("balanced_focal", "focal"):
            bce_map = F.binary_cross_entropy(p, t, reduction="none")
            pt = p * t + (1.0 - p) * (1.0 - t)
            focal_w = (1.0 - pt).clamp(min=0.0).pow(float(focal_gamma))
            loss_map = bce_map * focal_w
            alpha = float(focal_alpha)
            if alpha >= 0.0:
                alpha_t = alpha * t + (1.0 - alpha) * (1.0 - t)
                loss_map = loss_map * alpha_t

        pos_mask = vm & (t > 0.5)
        neg_mask = vm & (t < 0.5)
        has_pos = bool(pos_mask.any().item())
        has_neg = bool(neg_mask.any().item())
        if (not has_pos) and (not has_neg):
            return None
        if has_pos:
            loss_pos = (loss_map * pos_mask.to(torch.float32)).sum() / pos_mask.sum().clamp(min=1).to(torch.float32)
        else:
            loss_pos = loss_map.sum() * 0.0
        if has_neg:
            loss_neg = (loss_map * neg_mask.to(torch.float32)).sum() / neg_mask.sum().clamp(min=1).to(torch.float32)
        else:
            loss_neg = loss_map.sum() * 0.0
        return 0.5 * (loss_pos + loss_neg) if (has_pos and has_neg) else (loss_pos if has_pos else loss_neg)

    @staticmethod
    def _compute_foreground_tversky_pair_loss(
        pred_occ: torch.Tensor,
        gt_occ: torch.Tensor,
        valid_mask: torch.Tensor,
        alpha: float = 0.7,
        beta: float = 0.3,
        eps: float = 1e-6,
    ):
        if (
            (not torch.is_tensor(pred_occ))
            or (not torch.is_tensor(gt_occ))
            or (not torch.is_tensor(valid_mask))
        ):
            return None
        p = pred_occ.to(torch.float32)
        t = gt_occ.to(device=p.device, dtype=torch.float32).clamp(0.0, 1.0)
        vm = valid_mask.to(device=p.device, dtype=torch.float32)
        if not bool((vm > 0).any().item()):
            return None

        p = p * vm
        t = t * vm
        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()
        denom = tp + (float(alpha) * fp) + (float(beta) * fn)
        score = (tp + float(eps)) / (denom + float(eps))
        return 1.0 - score.clamp(0.0, 1.0)

    @staticmethod
    def _build_matched_gmo_quality_mask(
        pair_gt_ids: torch.Tensor,
        matched_inst_idx: torch.Tensor,
        gt_occ_txyz: torch.Tensor,
        bbox_instance_occ3d_txyz: torch.Tensor,
        gt_centers_tn3: torch.Tensor,
        gt_valid_tn: torch.Tensor,
        present_idx: int,
        roi_radius_m: float,
        min_occupancy_ratio: float,
        eps: float = 1e-6,
    ):
        device = pair_gt_ids.device
        k_count = int(pair_gt_ids.numel())
        keep = torch.zeros((k_count,), device=device, dtype=torch.bool)
        roi_keep = torch.zeros_like(keep)
        ratio_keep = torch.zeros_like(keep)
        ratios = pair_gt_ids.new_zeros((k_count,), dtype=torch.float32)
        if k_count <= 0:
            return keep, roi_keep, ratio_keep, ratios
        if (
            (not torch.is_tensor(gt_occ_txyz))
            or gt_occ_txyz.dim() != 4
            or (not torch.is_tensor(bbox_instance_occ3d_txyz))
            or bbox_instance_occ3d_txyz.dim() != 4
            or (not torch.is_tensor(gt_centers_tn3))
            or gt_centers_tn3.dim() != 3
            or (not torch.is_tensor(gt_valid_tn))
            or gt_valid_tn.dim() != 2
        ):
            return keep, roi_keep, ratio_keep, ratios

        t_count = min(
            int(gt_occ_txyz.shape[0]),
            int(bbox_instance_occ3d_txyz.shape[0]),
            int(gt_centers_tn3.shape[0]),
            int(gt_valid_tn.shape[0]),
        )
        if t_count <= 0:
            return keep, roi_keep, ratio_keep, ratios
        pidx = max(0, min(int(present_idx), t_count - 1))

        mi = matched_inst_idx.to(device=device, dtype=torch.long)
        valid_mi = (mi >= 0) & (mi < int(gt_centers_tn3.shape[1])) & (mi < int(gt_valid_tn.shape[1]))
        if bool(valid_mi.any().item()):
            centers = gt_centers_tn3[pidx].to(device=device, dtype=torch.float32)
            valid = gt_valid_tn[pidx].to(device=device, dtype=torch.bool)
            pair_centers = centers.index_select(0, mi.clamp(0, max(int(centers.shape[0]) - 1, 0)))
            pair_valid = valid.index_select(0, mi.clamp(0, max(int(valid.shape[0]) - 1, 0))) & valid_mi
            dist_xy = torch.linalg.norm(pair_centers[:, :2], dim=-1)
            roi_keep = pair_valid & (dist_xy <= float(roi_radius_m))

        fine_frame = gt_occ_txyz[pidx].to(device=device, dtype=torch.long)
        bbox_frame = bbox_instance_occ3d_txyz[pidx].to(device=device, dtype=torch.long)
        for k in range(k_count):
            inst_id = int(pair_gt_ids[k].item())
            if inst_id <= 0:
                continue
            fine_count = (fine_frame == inst_id).sum().to(torch.float32)
            bbox_count = (bbox_frame == inst_id).sum().to(torch.float32)
            if float(bbox_count.item()) <= 0.0:
                continue
            ratio = fine_count / bbox_count.clamp_min(float(eps))
            ratios[k] = ratio
            ratio_keep[k] = ratio >= float(min_occupancy_ratio)

        keep = roi_keep & ratio_keep
        return keep, roi_keep, ratio_keep, ratios

    def _compute_matched_pair_gmo_losses(
        self,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        gt_instance_occ3d_txyz_pred: torch.Tensor,
        bbox_instance_occ3d_txyz: torch.Tensor = None,
        objectness_scores_tq: torch.Tensor = None,
        inst_match_result: dict = None,
        loss_weight: float = 0.1,
        loss_type: str = "balanced_bce",
        shape_loss_mode: str = "global",
        local_crop_size=None,
        local_crop_scale: float = 1.0,
        local_crop_margin_vox=(0.0, 0.0, 0.0),
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        compute_dice: bool = False,
        dice_loss_weight: float = 1.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_quat_tqg4: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        pair_chunk_size: int = 8,
        quality_filter_enabled: bool = False,
        quality_roi_radius_m: float = 30.0,
        quality_min_occupancy_ratio: float = 0.4,
        quality_present_idx: int = 0,
        eps: float = 1e-6,
    ) -> dict:
        z = centers_world_tq3.sum() * 0.0
        loss_type = str(loss_type).lower()
        shape_loss_mode = str(shape_loss_mode).lower()
        if shape_loss_mode in ("local", "aabb_local"):
            shape_loss_mode = "local_aabb"
        loss_key = "loss_gmo_focal" if loss_type in ("balanced_focal", "focal") else "loss_gmo_bce"
        if local_crop_size is None:
            crop_x, crop_y, crop_z = 0, 0, 0
        else:
            crop_x, crop_y, crop_z = [int(v) for v in local_crop_size]
        margin_vals = tuple(float(v) for v in local_crop_margin_vox)
        out = {
            "dbg_gmo_bce_pair_count": z,
            "dbg_gmo_bce_lowres_x": z,
            "dbg_gmo_bce_lowres_y": z,
            "dbg_gmo_bce_lowres_z": z,
            "dbg_gmo_shape_loss_mode": z.new_tensor(float(1.0 if shape_loss_mode == "local_aabb" else 0.0)),
            "dbg_gmo_local_crop_x": z.new_tensor(float(crop_x)),
            "dbg_gmo_local_crop_y": z.new_tensor(float(crop_y)),
            "dbg_gmo_local_crop_z": z.new_tensor(float(crop_z)),
            "dbg_gmo_local_crop_scale": z.new_tensor(float(local_crop_scale)),
            "dbg_gmo_local_crop_margin_vox_x": z.new_tensor(float(margin_vals[0])),
            "dbg_gmo_local_crop_margin_vox_y": z.new_tensor(float(margin_vals[1])),
            "dbg_gmo_local_crop_margin_vox_z": z.new_tensor(float(margin_vals[2])),
            "dbg_gmo_local_valid_pair_count": z,
            "loss_gmo_dice": z,
            "dbg_gmo_dice_pair_count": z,
            "dbg_gmo_dice_alpha": z.new_tensor(float(tversky_alpha)),
            "dbg_gmo_dice_beta": z.new_tensor(float(tversky_beta)),
            "dbg_gmo_quality_filter_enabled": z.new_tensor(float(bool(quality_filter_enabled))),
            "dbg_gmo_quality_pair_count_before": z,
            "dbg_gmo_quality_pair_count_after": z,
            "dbg_gmo_quality_roi_keep_count": z,
            "dbg_gmo_quality_ratio_keep_count": z,
            "dbg_gmo_quality_ratio_mean": z,
            "dbg_gmo_quality_scene_pass_object_count": z,
            "dbg_gmo_quality_roi_radius_m": z.new_tensor(float(quality_roi_radius_m)),
            "dbg_gmo_quality_min_ratio": z.new_tensor(float(quality_min_occupancy_ratio)),
        }
        out[loss_key] = z
        if loss_key == "loss_gmo_focal":
            out["dbg_gmo_bce_alias"] = z
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return out
        if (not torch.is_tensor(gt_instance_occ3d_txyz_pred)) or gt_instance_occ3d_txyz_pred.dim() != 4:
            return out
        if not isinstance(inst_match_result, dict):
            return out

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_ids_n = inst_match_result.get("gt_ids_n", None)
        gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
            or (not torch.is_tensor(gt_ids_n))
            or gt_ids_n.numel() <= 0
        ):
            return out

        Tq, Q, _ = centers_world_tq3.shape
        To, _, _, _ = gt_instance_occ3d_txyz_pred.shape
        T = min(int(Tq), int(To))
        if T <= 0:
            return out
        mq = matched_query_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        mi = matched_inst_idx.to(device=centers_world_tq3.device, dtype=torch.long)
        gt_ids = gt_ids_n.to(device=centers_world_tq3.device, dtype=torch.long)
        keep = (mq >= 0) & (mq < int(Q)) & (mi >= 0) & (mi < int(gt_ids.numel()))
        if not bool(keep.any().item()):
            return out
        mq = mq[keep]
        mi = mi[keep]
        if mq.numel() <= 0:
            return out
        pair_gt_ids = gt_ids.index_select(0, mi)
        out["dbg_gmo_quality_pair_count_before"] = centers_world_tq3.new_tensor(float(mq.numel()))

        gt_occ_txyz = gt_instance_occ3d_txyz_pred[:T].to(device=centers_world_tq3.device, dtype=torch.long)
        if torch.is_tensor(gt_valid_tn) and gt_valid_tn.dim() == 2 and int(gt_valid_tn.shape[0]) >= T:
            gt_valid_sel_tk = gt_valid_tn[:T].to(device=centers_world_tq3.device, dtype=torch.bool).index_select(1, mi)
        else:
            gt_valid_sel_tk = torch.ones((T, int(mq.numel())), device=centers_world_tq3.device, dtype=torch.bool)

        if bool(quality_filter_enabled):
            gt_centers_tn3 = inst_match_result.get("gt_centers_tn3", None)
            quality_keep, roi_keep, ratio_keep, quality_ratios = self._build_matched_gmo_quality_mask(
                pair_gt_ids=pair_gt_ids,
                matched_inst_idx=mi,
                gt_occ_txyz=gt_occ_txyz,
                bbox_instance_occ3d_txyz=bbox_instance_occ3d_txyz,
                gt_centers_tn3=gt_centers_tn3,
                gt_valid_tn=gt_valid_tn,
                present_idx=int(quality_present_idx),
                roi_radius_m=float(quality_roi_radius_m),
                min_occupancy_ratio=float(quality_min_occupancy_ratio),
                eps=float(eps),
            )
            out["dbg_gmo_quality_roi_keep_count"] = centers_world_tq3.new_tensor(float(roi_keep.sum().item()))
            out["dbg_gmo_quality_ratio_keep_count"] = centers_world_tq3.new_tensor(float(ratio_keep.sum().item()))
            valid_ratio = quality_ratios > 0.0
            if bool(valid_ratio.any().item()):
                out["dbg_gmo_quality_ratio_mean"] = quality_ratios[valid_ratio].mean()
            if not bool(quality_keep.any().item()):
                return out
            mq = mq[quality_keep]
            mi = mi[quality_keep]
            pair_gt_ids = pair_gt_ids[quality_keep]
            gt_valid_sel_tk = gt_valid_sel_tk.index_select(1, torch.nonzero(quality_keep, as_tuple=False).squeeze(1))
            out["dbg_gmo_quality_scene_pass_object_count"] = centers_world_tq3.new_tensor(
                float(torch.unique(pair_gt_ids).numel())
            )

        out["dbg_gmo_quality_pair_count_after"] = centers_world_tq3.new_tensor(float(mq.numel()))

        pair_losses = []
        dice_pair_losses = []
        use_mixture = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_quat_tqg4)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_quat_tqg4.shape[:3])
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
        )
        if (not use_mixture) and (
            (not torch.is_tensor(sigmas_world_tq3))
            or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape)
        ):
            return out
        if shape_loss_mode == "local_aabb":
            pred_pairs, gt_pairs, valid_pairs, valid_count, mean_shape, pair_meta = self._prepare_local_aabb_matched_pair_occ_pairs(
                gt_occ_txyz=gt_occ_txyz,
                bbox_occ_txyz=bbox_instance_occ3d_txyz,
                matched_query_idx=mq,
                pair_gt_ids_k=pair_gt_ids,
                gt_valid_sel_tk=gt_valid_sel_tk,
                centers_world_tq3=centers_world_tq3[:T],
                sigmas_world_tq3=sigmas_world_tq3[:T] if torch.is_tensor(sigmas_world_tq3) else None,
                mixture_centers_world_tqg3=mixture_centers_world_tqg3[:T] if use_mixture else None,
                mixture_sigmas_world_tqg3=mixture_sigmas_world_tqg3[:T] if use_mixture else None,
                mixture_quat_tqg4=mixture_quat_tqg4[:T] if use_mixture else None,
                mixture_weights_tqg=mixture_weights_tqg[:T] if use_mixture else None,
                objectness_scores_tq=objectness_scores_tq,
                crop_size=local_crop_size,
                crop_scale=float(local_crop_scale),
                crop_margin_vox=margin_vals,
                eps=float(eps),
            )
            if pred_pairs is None:
                return out
            step = self._get_query_train_iteration(advance_if_unsynced=False)
            self._maybe_save_local_aabb_gmo_pair_vis(
                pred_pairs=pred_pairs,
                gt_pairs=gt_pairs,
                valid_pairs=valid_pairs,
                pair_meta=pair_meta,
                step=step,
            )
            out["dbg_gmo_local_valid_pair_count"] = centers_world_tq3.new_tensor(float(valid_count))
            if torch.is_tensor(mean_shape):
                out["dbg_gmo_local_crop_x"] = mean_shape[0]
                out["dbg_gmo_local_crop_y"] = mean_shape[1]
                out["dbg_gmo_local_crop_z"] = mean_shape[2]
            for p_raw, t_raw, valid_mask in zip(pred_pairs, gt_pairs, valid_pairs):
                p = p_raw.clamp(float(eps), 1.0 - float(eps))
                t = t_raw.clamp(0.0, 1.0)
                pair_loss = self._compute_balanced_binary_pair_loss(
                    pred_occ=p,
                    gt_occ=t,
                    valid_mask=valid_mask,
                    loss_type=loss_type,
                    focal_gamma=float(focal_gamma),
                    focal_alpha=float(focal_alpha),
                    eps=float(eps),
                )
                if pair_loss is not None:
                    pair_losses.append(pair_loss)
                if compute_dice:
                    p_bev = p.amax(dim=2)
                    t_bev = t.amax(dim=2)
                    valid_bev = valid_mask.any(dim=2)
                    dice_loss = self._compute_foreground_tversky_pair_loss(
                        pred_occ=p_bev,
                        gt_occ=t_bev,
                        valid_mask=valid_bev,
                        alpha=float(tversky_alpha),
                        beta=float(tversky_beta),
                        eps=float(eps),
                    )
                    if dice_loss is not None:
                        dice_pair_losses.append(dice_loss)
        elif use_mixture:
            pred_tk1zyx, gt_tk1zyx, valid_tk = self._prepare_grouped_matched_pair_lowres_occ(
                gt_occ_txyz=gt_occ_txyz,
                matched_query_idx=mq,
                pair_gt_ids_k=pair_gt_ids,
                gt_valid_sel_tk=gt_valid_sel_tk,
                mixture_centers_world_tqg3=mixture_centers_world_tqg3[:T],
                mixture_sigmas_world_tqg3=mixture_sigmas_world_tqg3[:T],
                mixture_quat_tqg4=mixture_quat_tqg4[:T],
                mixture_weights_tqg=mixture_weights_tqg[:T],
                objectness_scores_tq=objectness_scores_tq,
                pair_chunk_size=int(pair_chunk_size),
            )
            if pred_tk1zyx is None:
                return out
            for k in range(int(mq.numel())):
                p = pred_tk1zyx[:, k:k + 1].clamp(float(eps), 1.0 - float(eps))
                t = gt_tk1zyx[:, k:k + 1].clamp(0.0, 1.0)
                valid_mask = valid_tk[:, k:k + 1, None, None, None, None].expand_as(t)
                pair_loss = self._compute_balanced_binary_pair_loss(
                    pred_occ=p,
                    gt_occ=t,
                    valid_mask=valid_mask,
                    loss_type=loss_type,
                    focal_gamma=float(focal_gamma),
                    focal_alpha=float(focal_alpha),
                    eps=float(eps),
                )
                if pair_loss is not None:
                    pair_losses.append(pair_loss)
                if compute_dice:
                    p_bev = p[:, :, 0].amax(dim=2)
                    t_bev = t[:, :, 0].amax(dim=2)
                    valid_bev = valid_tk[:, k:k + 1, None, None].expand_as(p_bev)
                    dice_loss = self._compute_foreground_tversky_pair_loss(
                        pred_occ=p_bev,
                        gt_occ=t_bev,
                        valid_mask=valid_bev,
                        alpha=float(tversky_alpha),
                        beta=float(tversky_beta),
                        eps=float(eps),
                    )
                    if dice_loss is not None:
                        dice_pair_losses.append(dice_loss)
        else:
            for k in range(int(mq.numel())):
                q_idx = int(mq[k].item())
                inst_id = int(pair_gt_ids[k].item())
                pred_k_t1zyx, gt_k_t1zyx, valid_frame_t = self._prepare_single_matched_pair_lowres_occ(
                    centers_world_tq3=centers_world_tq3[:T],
                    sigmas_world_tq3=sigmas_world_tq3[:T],
                    gt_occ_txyz=gt_occ_txyz,
                    q_idx=q_idx,
                    inst_id=inst_id,
                    valid_frame_t=gt_valid_sel_tk[:, k],
                    objectness_scores_tq=objectness_scores_tq,
                )
                if pred_k_t1zyx is None:
                    continue
                valid_mask = valid_frame_t[:, None, None, None, None].expand_as(gt_k_t1zyx)
                p = pred_k_t1zyx.clamp(float(eps), 1.0 - float(eps))
                t = gt_k_t1zyx.clamp(0.0, 1.0)
                pair_loss = self._compute_balanced_binary_pair_loss(
                    pred_occ=p,
                    gt_occ=t,
                    valid_mask=valid_mask,
                    loss_type=loss_type,
                    focal_gamma=float(focal_gamma),
                    focal_alpha=float(focal_alpha),
                    eps=float(eps),
                )
                if pair_loss is not None:
                    pair_losses.append(pair_loss)
                if compute_dice:
                    p_bev = p.amax(dim=2)
                    t_bev = t.amax(dim=2)
                    valid_bev = valid_frame_t[:, None, None, None].expand_as(p_bev)
                    dice_loss = self._compute_foreground_tversky_pair_loss(
                        pred_occ=p_bev,
                        gt_occ=t_bev,
                        valid_mask=valid_bev,
                        alpha=float(tversky_alpha),
                        beta=float(tversky_beta),
                        eps=float(eps),
                    )
                    if dice_loss is not None:
                        dice_pair_losses.append(dice_loss)

        if len(pair_losses) <= 0 and len(dice_pair_losses) <= 0:
            return out

        if shape_loss_mode == "local_aabb":
            low_x = float(out["dbg_gmo_local_crop_x"].detach().item())
            low_y = float(out["dbg_gmo_local_crop_y"].detach().item())
            low_z = float(out["dbg_gmo_local_crop_z"].detach().item())
        else:
            low_x, low_y, low_z = self.query_matched_gmo_bce_occ_size
        if len(pair_losses) > 0:
            pair_loss = torch.stack(pair_losses, dim=0).mean()
            out[loss_key] = pair_loss * float(loss_weight)
        if compute_dice and len(dice_pair_losses) > 0:
            out["loss_gmo_dice"] = torch.stack(dice_pair_losses, dim=0).mean() * float(dice_loss_weight)
        if loss_key == "loss_gmo_focal":
            out["dbg_gmo_bce_alias"] = out[loss_key].detach()
        out["dbg_gmo_bce_pair_count"] = centers_world_tq3.new_tensor(float(len(pair_losses)))
        out["dbg_gmo_bce_lowres_x"] = centers_world_tq3.new_tensor(float(low_x))
        out["dbg_gmo_bce_lowres_y"] = centers_world_tq3.new_tensor(float(low_y))
        out["dbg_gmo_bce_lowres_z"] = centers_world_tq3.new_tensor(float(low_z))
        out["dbg_gmo_dice_pair_count"] = centers_world_tq3.new_tensor(float(len(dice_pair_losses)))
        return out

    def _build_query_objectness_targets(
        self,
        objectness_scores: torch.Tensor,
        inst_match_result: dict,
        has_supervision: bool = True,
        query_centers_world_tq3: torch.Tensor = None,
    ):
        """
        Build per-frame per-query objectness targets.
        Supports:
        - hard (legacy): Hungarian matched query only positive
        - soft_distance_topk: distance-based soft positives + GT-topk expansion + ambiguous ignore
        """
        targets = torch.zeros_like(objectness_scores, dtype=torch.float32)
        valid_mask = torch.ones_like(objectness_scores, dtype=torch.bool)

        if not has_supervision:
            valid_mask = torch.zeros_like(valid_mask)
            return targets, valid_mask
        if not isinstance(inst_match_result, dict):
            return targets, valid_mask

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
        gt_centers_tn3 = inst_match_result.get("gt_centers_tn3", None)

        matched_query_idx_t = None
        matched_inst_idx_t = None
        if torch.is_tensor(matched_query_idx) and matched_query_idx.numel() > 0:
            matched_query_idx_t = matched_query_idx.to(device=targets.device, dtype=torch.long)
        if torch.is_tensor(matched_inst_idx) and matched_inst_idx.numel() > 0:
            matched_inst_idx_t = matched_inst_idx.to(device=targets.device, dtype=torch.long)

        can_use_temporal_valid = (
            matched_query_idx_t is not None
            and matched_inst_idx_t is not None
            and torch.is_tensor(gt_valid_tn)
            and gt_valid_tn.dim() == 2
            and gt_valid_tn.shape[0] == targets.shape[0]
            and matched_inst_idx_t.numel() == matched_query_idx_t.numel()
        )

        # Legacy hard one-hot targets.
        mode = str(getattr(self, "objectness_target_mode", "hard")).lower()
        if mode == "hard":
            if matched_query_idx_t is None:
                return targets, valid_mask
            if can_use_temporal_valid:
                gt_valid_tk = gt_valid_tn.to(device=targets.device, dtype=torch.float32).index_select(1, matched_inst_idx_t)
                targets[:, matched_query_idx_t] = gt_valid_tk
            else:
                targets[:, matched_query_idx_t] = 1.0
            return targets, valid_mask

        # Soft target path (distance + top-k). Fallback to hard if required tensors are missing.
        can_use_soft = (
            mode == "soft_distance_topk"
            and torch.is_tensor(query_centers_world_tq3)
            and query_centers_world_tq3.dim() == 3
            and query_centers_world_tq3.shape[:2] == targets.shape
            and torch.is_tensor(gt_centers_tn3)
            and gt_centers_tn3.dim() == 3
            and torch.is_tensor(gt_valid_tn)
            and gt_valid_tn.dim() == 2
            and gt_centers_tn3.shape[:2] == gt_valid_tn.shape
            and gt_centers_tn3.shape[0] == targets.shape[0]
        )
        if not can_use_soft:
            if matched_query_idx_t is None:
                return targets, valid_mask
            if can_use_temporal_valid:
                gt_valid_tk = gt_valid_tn.to(device=targets.device, dtype=torch.float32).index_select(1, matched_inst_idx_t)
                targets[:, matched_query_idx_t] = gt_valid_tk
            else:
                targets[:, matched_query_idx_t] = 1.0
            return targets, valid_mask

        q_centers = query_centers_world_tq3.to(device=targets.device, dtype=torch.float32)
        gt_centers = gt_centers_tn3.to(device=targets.device, dtype=torch.float32)
        gt_valid = gt_valid_tn.to(device=targets.device, dtype=torch.bool)

        pos_r = max(1e-6, float(self.objectness_soft_pos_radius_m))
        neg_r = max(pos_r + 1e-6, float(self.objectness_soft_neg_radius_m))
        ign_lo, ign_hi = [float(v) for v in self.objectness_soft_ignore_radius_m]
        if ign_lo > ign_hi:
            ign_lo, ign_hi = ign_hi, ign_lo
        topk_per_gt = max(0, int(self.objectness_soft_topk_per_gt))

        T, Q = targets.shape
        for t in range(T):
            valid_gt_idx = torch.nonzero(gt_valid[t], as_tuple=False).squeeze(1)
            if valid_gt_idx.numel() <= 0:
                # No GT at this frame: all queries are clean negatives.
                continue

            gt_t = gt_centers[t].index_select(0, valid_gt_idx)   # [Ng,3]
            q_t = q_centers[t]                                    # [Q,3]
            dist_qg = torch.cdist(q_t, gt_t)                      # [Q,Ng]
            min_dist_q = dist_qg.min(dim=1).values                # [Q]

            # Distance-based soft target: 1 (<=pos_r), 0 (>=neg_r), linear in between.
            soft_q = ((neg_r - min_dist_q) / (neg_r - pos_r)).clamp(0.0, 1.0)
            soft_q = torch.where(min_dist_q <= pos_r, torch.ones_like(soft_q), soft_q)
            soft_q = torch.where(min_dist_q >= neg_r, torch.zeros_like(soft_q), soft_q)

            # GT-wise top-k expansion: multiple near-tie queries can receive positive supervision.
            topk_pos_mask_q = torch.zeros((Q,), device=targets.device, dtype=torch.bool)
            if topk_per_gt > 0:
                kk = min(topk_per_gt, Q)
                for g in range(dist_qg.shape[1]):
                    d_q = dist_qg[:, g]
                    topk_idx = torch.topk(d_q, k=kk, largest=False).indices
                    topk_pos_mask_q[topk_idx] = True
                    # Use the same distance-to-score mapping for these candidates, but preserve the max score.
                    d_topk = d_q.index_select(0, topk_idx)
                    s_topk = ((neg_r - d_topk) / (neg_r - pos_r)).clamp(0.0, 1.0)
                    s_topk = torch.where(d_topk <= pos_r, torch.ones_like(s_topk), s_topk)
                    s_topk = torch.where(d_topk >= neg_r, torch.zeros_like(s_topk), s_topk)
                    soft_q[topk_idx] = torch.maximum(soft_q[topk_idx], s_topk)

            # Ambiguous distance band is ignored, unless query is a top-k candidate (or later matched override).
            ambiguous_q = (min_dist_q > ign_lo) & (min_dist_q < ign_hi)
            valid_mask[t, ambiguous_q & (~topk_pos_mask_q)] = False
            targets[t] = soft_q

        # Optionally preserve Hungarian positives as hard positives (reduces under-labeling).
        if self.objectness_soft_match_override and (matched_query_idx_t is not None):
            if can_use_temporal_valid:
                gt_valid_tk = gt_valid.to(dtype=torch.float32).index_select(1, matched_inst_idx_t)
                targets[:, matched_query_idx_t] = torch.maximum(targets[:, matched_query_idx_t], gt_valid_tk)
                valid_mask[:, matched_query_idx_t] = True
            else:
                targets[:, matched_query_idx_t] = torch.maximum(
                    targets[:, matched_query_idx_t],
                    torch.ones_like(targets[:, matched_query_idx_t]),
                )
                valid_mask[:, matched_query_idx_t] = True

        return targets.clamp_(0.0, 1.0), valid_mask

    def _build_query_objectness_targets_hungarian_binary(
        self,
        objectness_scores: torch.Tensor,
        inst_match_result: dict,
        has_supervision: bool = True,
    ):
        """
        Build strict 0/1 objectness targets from Hungarian matches only.

        - matched queries: 1 (or frame-wise GT-valid if available)
        - unmatched queries: 0
        """
        targets = torch.zeros_like(objectness_scores, dtype=torch.float32)
        valid_mask = torch.ones_like(objectness_scores, dtype=torch.bool)

        if not has_supervision:
            valid_mask = torch.zeros_like(valid_mask)
            return targets, valid_mask
        if not isinstance(inst_match_result, dict):
            valid_mask = torch.zeros_like(valid_mask)
            return targets, valid_mask

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
        if (not torch.is_tensor(matched_query_idx)) or matched_query_idx.numel() <= 0:
            return targets, valid_mask

        mq = matched_query_idx.to(device=targets.device, dtype=torch.long)
        can_use_temporal_valid = (
            torch.is_tensor(matched_inst_idx)
            and torch.is_tensor(gt_valid_tn)
            and matched_inst_idx.numel() == mq.numel()
            and gt_valid_tn.dim() == 2
            and gt_valid_tn.shape[0] == targets.shape[0]
        )
        if can_use_temporal_valid:
            mk = matched_inst_idx.to(device=targets.device, dtype=torch.long)
            gt_valid_tk = gt_valid_tn.to(device=targets.device, dtype=torch.float32).index_select(1, mk)
            targets[:, mq] = gt_valid_tk
        else:
            targets[:, mq] = 1.0
        return targets, valid_mask

    def _aggregate_training_losses(
        self,
        *,
        # Pre-computed query loss dicts
        query_cls_loss,
        query_depth_loss,
        query_attn_bbox_loss,
        matched_gmo_loss,
        center_match_loss,
        query_traj_loss,
        query_attn_cam_score_pack,
        query_size_dbg,
        # Query predictions
        centers_world,
        centers_world_match_tq3,
        centers_world_dt_gt2p_tq3,
        gaussian_sigmas_world_loss,
        mixture_centers_world_loss,
        mixture_sigmas_world_loss,
        mixture_quat_loss,
        mixture_weights_loss,
        # DT loss inputs
        occ_dt,
        # GT2P loss inputs
        gt_inst_center_world_tn3,
        gt_inst_center_valid_tn,
        # Hungarian match pack
        inst_match_result=None,
    ):
        """Aggregate all training loss components into a single losses dict."""
        losses = dict()
        z = centers_world.sum() * 0.0

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
                    if (
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
        query_attn_overlap_loss = getattr(self.transformer, "last_query_attn_overlap_loss", None)
        losses["loss_query_attn_overlap"] = (
            query_attn_overlap_loss if torch.is_tensor(query_attn_overlap_loss) else z
        )
        if isinstance(query_size_dbg, dict):
            losses.update({
                k: v
                for k, v in query_size_dbg.items()
                if torch.is_tensor(v) and (torch.is_floating_point(v) or torch.is_complex(v))
            })

        # ---- Center match loss ----
        losses["loss_query_center_match"] = center_match_loss if center_match_loss is not None else z
        center_match_dbg = self._compute_query_center_match_dbg_from_match(
            centers_world_tq3=centers_world_match_tq3,
            inst_match_result=inst_match_result,
        )
        if isinstance(center_match_dbg, dict):
            losses.update(center_match_dbg)

        # ---- Trajectory loss ----
        losses.update({
            "loss_query_traj": z,
            "dbg_query_traj_reweight_enabled": z,
            "dbg_query_traj_reweight_applied": z,
            "dbg_query_traj_valid_count": z,
            "dbg_query_traj_weight_mean": z,
            "dbg_query_traj_moving_ratio": z,
            "dbg_query_traj_motion_mag_mean": z,
        })
        if isinstance(query_traj_loss, dict):
            losses.update(query_traj_loss)
        elif query_traj_loss is not None:
            losses["loss_query_traj"] = query_traj_loss

        # ---- Hungarian matching cost diagnostics ----
        losses["dbg_query_cls_match_cost_weight"] = centers_world.new_tensor(
            float(self.query_cls_match_cost_weight)
        )
        losses["dbg_query_center_match_cost_weight"] = centers_world.new_tensor(
            float(getattr(self, "query_center_match_cost_weight", 0.0))
        )
        losses["dbg_query_temporal_offset_match_cost_weight"] = centers_world.new_tensor(
            float(getattr(self, "query_temporal_offset_match_cost_weight", 0.0))
        )
        losses["dbg_query_bev_iou_match_cost_weight"] = centers_world.new_tensor(
            float(self.query_bev_iou_match_cost_weight)
        )
        losses["dbg_query_attn_match_cost_weight"] = centers_world.new_tensor(
            float(getattr(self, "query_attn_match_cost_weight", 0.0))
        )
        losses["dbg_query_match_cost_total_mean"] = z
        losses["dbg_query_match_cost_matched_mean"] = z
        losses["dbg_query_match_cost_total_pair_count"] = z
        losses["dbg_query_match_cost_matched_pair_count"] = z
        for name in (
            "cls",
            "center",
            "temporal_offset",
            "bev_iou",
            "attn",
        ):
            losses[f"dbg_query_match_cost_{name}_total_mean"] = z
            losses[f"dbg_query_match_cost_{name}_matched_mean"] = z
            losses[f"dbg_query_match_cost_{name}_ratio_total"] = z
            losses[f"dbg_query_match_cost_{name}_ratio_matched"] = z

        if isinstance(inst_match_result, dict):
            cost_qn = inst_match_result.get("cost_qn", None)
            matched_query_idx = inst_match_result.get("matched_query_idx", None)
            matched_inst_idx = inst_match_result.get("matched_inst_idx", None)

            if torch.is_tensor(cost_qn) and cost_qn.dim() == 2 and int(cost_qn.numel()) > 0:
                cost_qn_f32 = cost_qn.to(torch.float32)
                finite_mask_qn = torch.isfinite(cost_qn_f32)
                finite_vals = cost_qn_f32[finite_mask_qn]
                if int(finite_vals.numel()) > 0:
                    total_mean = finite_vals.mean()
                else:
                    total_mean = z

                matched_vals = None
                if (
                    torch.is_tensor(matched_query_idx)
                    and torch.is_tensor(matched_inst_idx)
                    and int(matched_query_idx.numel()) > 0
                    and int(matched_query_idx.numel()) == int(matched_inst_idx.numel())
                ):
                    mq = matched_query_idx.to(device=cost_qn_f32.device, dtype=torch.long).reshape(-1)
                    mi = matched_inst_idx.to(device=cost_qn_f32.device, dtype=torch.long).reshape(-1)
                    keep = (
                        (mq >= 0)
                        & (mi >= 0)
                        & (mq < int(cost_qn_f32.shape[0]))
                        & (mi < int(cost_qn_f32.shape[1]))
                    )
                    if bool(keep.any().item()):
                        mq = mq[keep]
                        mi = mi[keep]
                        matched_raw = cost_qn_f32[mq, mi]
                        matched_vals = matched_raw[torch.isfinite(matched_raw)]

                if torch.is_tensor(matched_vals) and int(matched_vals.numel()) > 0:
                    matched_mean = matched_vals.mean()
                else:
                    matched_mean = z

                losses["dbg_query_match_cost_total_mean"] = total_mean
                losses["dbg_query_match_cost_matched_mean"] = matched_mean
                losses["dbg_query_match_cost_total_pair_count"] = cost_qn_f32.new_tensor(float(finite_vals.numel()))
                losses["dbg_query_match_cost_matched_pair_count"] = cost_qn_f32.new_tensor(
                    float(0 if matched_vals is None else int(matched_vals.numel()))
                )

                eps = float(1e-9)
                total_mean_safe = total_mean.detach().abs().clamp_min(eps) if torch.is_tensor(total_mean) else cost_qn_f32.new_tensor(eps)
                matched_mean_safe = matched_mean.detach().abs().clamp_min(eps) if torch.is_tensor(matched_mean) else cost_qn_f32.new_tensor(eps)

                contrib_map = {
                    "cls": inst_match_result.get("cost_cls_contrib_qn", None),
                    "center": inst_match_result.get("cost_center_contrib_qn", None),
                    "temporal_offset": inst_match_result.get("cost_temporal_offset_contrib_qn", None),
                    "bev_iou": inst_match_result.get("cost_bev_iou_contrib_qn", None),
                    "attn": inst_match_result.get("cost_attn_iou_contrib_qn", None),
                }
                for name, contrib_qn in contrib_map.items():
                    if torch.is_tensor(contrib_qn) and tuple(contrib_qn.shape) == tuple(cost_qn_f32.shape):
                        contrib_f32 = contrib_qn.to(torch.float32)
                        contrib_total_vals = contrib_f32[finite_mask_qn & torch.isfinite(contrib_f32)]
                        comp_total_mean = contrib_total_vals.mean() if int(contrib_total_vals.numel()) > 0 else z

                        comp_matched_mean = z
                        if (
                            torch.is_tensor(matched_query_idx)
                            and torch.is_tensor(matched_inst_idx)
                            and int(matched_query_idx.numel()) > 0
                            and int(matched_query_idx.numel()) == int(matched_inst_idx.numel())
                        ):
                            mq = matched_query_idx.to(device=contrib_f32.device, dtype=torch.long).reshape(-1)
                            mi = matched_inst_idx.to(device=contrib_f32.device, dtype=torch.long).reshape(-1)
                            keep = (
                                (mq >= 0)
                                & (mi >= 0)
                                & (mq < int(contrib_f32.shape[0]))
                                & (mi < int(contrib_f32.shape[1]))
                            )
                            if bool(keep.any().item()):
                                comp_matched = contrib_f32[mq[keep], mi[keep]]
                                comp_matched = comp_matched[torch.isfinite(comp_matched)]
                                if int(comp_matched.numel()) > 0:
                                    comp_matched_mean = comp_matched.mean()

                        losses[f"dbg_query_match_cost_{name}_total_mean"] = comp_total_mean
                        losses[f"dbg_query_match_cost_{name}_matched_mean"] = comp_matched_mean
                        losses[f"dbg_query_match_cost_{name}_ratio_total"] = comp_total_mean / total_mean_safe
                        losses[f"dbg_query_match_cost_{name}_ratio_matched"] = comp_matched_mean / matched_mean_safe

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
                    mixture_centers_world_tqg3=mixture_centers_world_loss,
                    mixture_sigmas_world_tqg3=mixture_sigmas_world_loss,
                    mixture_quat_tqg4=mixture_quat_loss,
                    mixture_weights_tqg=mixture_weights_loss,
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
                mixture_centers_world_tqg3=mixture_centers_world_loss,
                mixture_sigmas_world_tqg3=mixture_sigmas_world_loss,
                mixture_quat_tqg4=mixture_quat_loss,
                mixture_weights_tqg=mixture_weights_loss,
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
            "dbg_query_cls_match_cost_weight",
            "dbg_query_center_match_cost_weight",
            "dbg_query_temporal_offset_match_cost_weight",
            "dbg_query_bev_iou_match_cost_weight",
            "dbg_query_attn_match_cost_weight",
        }
        for k in [
            k for k in losses
            if k.startswith("dbg_")
            and (k not in _keep_dbg)
            and (not k.startswith("dbg_query_cls_"))
            and (not k.startswith("dbg_query_traj_"))
            and (not k.startswith("dbg_query_match_cost_"))
            and (not k.startswith("dbg_query_matched_center_"))
            and (not k.startswith("dbg_query_size_"))
            and (not k.startswith("dbg_gmo_quality_"))
        ]:
            del losses[k]
        self._namespace_dbg_logs(losses)
        return losses
