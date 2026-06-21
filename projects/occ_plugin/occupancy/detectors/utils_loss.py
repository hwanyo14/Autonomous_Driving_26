import torch
import torch.nn.functional as F


class EfficientOCFLossMixin:
    """Loss computation methods for EfficientOCF."""

    @staticmethod
    def _safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        w = weights.to(dtype=values.dtype)
        num = (values * w).sum()
        den = w.sum().clamp_min(float(eps))
        return num / den

    @staticmethod
    def _compute_traj_summary_from_deltas(
        traj_deltas_tk2: torch.Tensor,
        valid_f: torch.Tensor,
    ):
        if (
            (not torch.is_tensor(traj_deltas_tk2))
            or traj_deltas_tk2.dim() != 3
            or (not torch.is_tensor(valid_f))
            or valid_f.dim() != 2
            or int(traj_deltas_tk2.shape[0]) != int(valid_f.shape[0])
            or int(traj_deltas_tk2.shape[1]) != int(valid_f.shape[1])
        ):
            return None
        valid_w = valid_f.to(dtype=traj_deltas_tk2.dtype)
        step_norm_tk = torch.norm(traj_deltas_tk2, p=2, dim=-1)
        path_length_k = (step_norm_tk * valid_w).sum(dim=0)
        valid_count_k = valid_w.sum(dim=0).clamp_min(1.0)
        mean_speed_k = path_length_k / valid_count_k
        endpoint_vec_k2 = (traj_deltas_tk2 * valid_w.unsqueeze(-1)).sum(dim=0)
        endpoint_disp_k = torch.norm(endpoint_vec_k2, p=2, dim=-1)
        straightness_k = endpoint_disp_k / path_length_k.clamp_min(1e-6)
        return mean_speed_k, endpoint_disp_k, path_length_k, straightness_k.clamp(0.0, 1.0)

    def _apply_query_turn_family_routing(
        self,
        best_mode_k: torch.Tensor,
        mode_err_mk: torch.Tensor,
        gt_tk2: torch.Tensor,
        valid_f: torch.Tensor,
        gt_centers_xy_tn2: torch.Tensor,
        gt_valid_tn: torch.Tensor,
        present_local_idx: int,
        matched_inst_idx: torch.Tensor,
        matched_query_idx: torch.Tensor = None,
        pred_centers_xy_tq2: torch.Tensor = None,
    ):
        if (
            (not torch.is_tensor(best_mode_k))
            or (not torch.is_tensor(mode_err_mk))
            or (not torch.is_tensor(gt_tk2))
            or (not torch.is_tensor(valid_f))
            or (not torch.is_tensor(gt_centers_xy_tn2))
            or (not torch.is_tensor(gt_valid_tn))
            or (not torch.is_tensor(matched_inst_idx))
        ):
            return best_mode_k, None, None, None

        routed_mode_k = best_mode_k.clone()
        device = mode_err_mk.device
        stationary_enabled = bool(getattr(self, "query_traj_use_stationary_mode", False))
        cv_enabled = bool(getattr(self, "query_traj_use_cv_mode", False))
        stationary_mode_idx = 0 if stationary_enabled else None
        learned_start_idx = int(stationary_enabled) + int(cv_enabled)
        learned_mode_count = max(0, int(mode_err_mk.shape[1]) - learned_start_idx)

        gt_endpoint_k2 = (gt_tk2 * valid_f.unsqueeze(-1)).sum(dim=0)
        gt_endpoint_norm_k = torch.norm(gt_endpoint_k2, p=2, dim=-1)
        static_thr = float(getattr(self, "query_traj_static_threshold_m", 0.8))
        static_mask_k = gt_endpoint_norm_k <= static_thr

        turn_mask_k = static_mask_k.new_zeros(static_mask_k.shape)
        turn_family_enabled = bool(getattr(self, "query_traj_rule_turn_family_enabled", False))
        if (
            turn_family_enabled
            and learned_mode_count > 0
            and present_local_idx > 0
        ):
            past_valid_k = gt_valid_tn[present_local_idx - 1].to(
                device=device, dtype=torch.bool
            ).index_select(0, matched_inst_idx)
            cur_valid_k = gt_valid_tn[present_local_idx].to(
                device=device, dtype=torch.bool
            ).index_select(0, matched_inst_idx)
            past_pair_valid_k = past_valid_k & cur_valid_k
            if (
                torch.is_tensor(pred_centers_xy_tq2)
                and torch.is_tensor(matched_query_idx)
                and int(pred_centers_xy_tq2.shape[0]) > present_local_idx
            ):
                past_vec_k2 = (
                    pred_centers_xy_tq2[present_local_idx].to(device=device, dtype=torch.float32).index_select(0, matched_query_idx)
                    - pred_centers_xy_tq2[present_local_idx - 1].to(device=device, dtype=torch.float32).index_select(0, matched_query_idx)
                )
                past_pair_valid_k = torch.ones_like(static_mask_k, dtype=torch.bool)
            else:
                past_vec_k2 = (
                    gt_centers_xy_tn2[present_local_idx].to(device=device, dtype=torch.float32).index_select(0, matched_inst_idx)
                    - gt_centers_xy_tn2[present_local_idx - 1].to(device=device, dtype=torch.float32).index_select(0, matched_inst_idx)
                )
            past_norm_k = torch.norm(past_vec_k2, p=2, dim=-1)
            turn_thr_deg = float(getattr(self, "query_traj_rule_turn_threshold_deg", 10.0))
            cross_k = (past_vec_k2[:, 0] * gt_endpoint_k2[:, 1]) - (past_vec_k2[:, 1] * gt_endpoint_k2[:, 0])
            dot_k = (past_vec_k2 * gt_endpoint_k2).sum(dim=-1)
            heading_change_deg_k = torch.atan2(cross_k, dot_k).abs() * (180.0 / torch.pi)
            turn_mask_k = (
                (~static_mask_k)
                & past_pair_valid_k
                & (past_norm_k > 1e-6)
                & (gt_endpoint_norm_k > 1e-6)
                & (heading_change_deg_k >= turn_thr_deg)
            )

        straight_mask_k = (~static_mask_k) & (~turn_mask_k)
        if learned_mode_count > 0:
            learned_err_mk = mode_err_mk[:, learned_start_idx:]
            if turn_family_enabled and learned_mode_count >= 2:
                straight_family_count = min(2, learned_mode_count)
                turn_family_start = straight_family_count
                if straight_family_count > 0:
                    straight_best_local_k = torch.argmin(
                        learned_err_mk[:, :straight_family_count], dim=-1
                    )
                    straight_best_mode_k = straight_best_local_k + learned_start_idx
                    routed_mode_k = torch.where(
                        straight_mask_k,
                        straight_best_mode_k,
                        routed_mode_k,
                    )
                if turn_family_start < learned_mode_count:
                    turn_best_local_k = torch.argmin(
                        learned_err_mk[:, turn_family_start:], dim=-1
                    )
                    turn_best_mode_k = turn_best_local_k + learned_start_idx + turn_family_start
                    routed_mode_k = torch.where(
                        turn_mask_k,
                        turn_best_mode_k,
                        routed_mode_k,
                    )
                else:
                    learned_best_local_k = torch.argmin(learned_err_mk, dim=-1)
                    learned_best_mode_k = learned_best_local_k + learned_start_idx
                    routed_mode_k = torch.where(
                        (~static_mask_k),
                        learned_best_mode_k,
                        routed_mode_k,
                    )
            else:
                learned_best_local_k = torch.argmin(learned_err_mk, dim=-1)
                learned_best_mode_k = learned_best_local_k + learned_start_idx
                routed_mode_k = torch.where(
                    (~static_mask_k),
                    learned_best_mode_k,
                    routed_mode_k,
                )

        if stationary_mode_idx is not None:
            routed_mode_k = torch.where(
                static_mask_k,
                routed_mode_k.new_full(routed_mode_k.shape, int(stationary_mode_idx)),
                routed_mode_k,
            )
        return routed_mode_k, static_mask_k, straight_mask_k, turn_mask_k

    def _append_query_traj_mode_summary_debug(
        self,
        out: dict,
        pred_tkm2: torch.Tensor,
        valid_f: torch.Tensor,
        selected_mode_k: torch.Tensor = None,
    ) -> dict:
        if not isinstance(out, dict):
            out = {}
        if (
            (not torch.is_tensor(pred_tkm2))
            or pred_tkm2.dim() != 4
            or (not torch.is_tensor(valid_f))
            or valid_f.dim() != 2
            or int(pred_tkm2.shape[0]) != int(valid_f.shape[0])
            or int(pred_tkm2.shape[1]) != int(valid_f.shape[1])
        ):
            return out

        zero = pred_tkm2.sum() * 0.0
        mode_count = int(pred_tkm2.shape[2])
        if (
            torch.is_tensor(selected_mode_k)
            and selected_mode_k.dim() == 1
            and int(selected_mode_k.shape[0]) == int(pred_tkm2.shape[1])
        ):
            selected_tk2 = pred_tkm2.gather(
                2,
                selected_mode_k.view(1, -1, 1, 1).expand(int(pred_tkm2.shape[0]), -1, 1, 2),
            ).squeeze(2)
            selected_summary = self._compute_traj_summary_from_deltas(selected_tk2, valid_f)
            if selected_summary is not None:
                out["dbg_query_traj_summary_mean_speed"] = selected_summary[0].mean()
                out["dbg_query_traj_summary_endpoint_displacement"] = selected_summary[1].mean()
                out["dbg_query_traj_summary_path_length"] = selected_summary[2].mean()
                out["dbg_query_traj_summary_straightness_ratio"] = selected_summary[3].mean()
        else:
            out["dbg_query_traj_summary_mean_speed"] = zero
            out["dbg_query_traj_summary_endpoint_displacement"] = zero
            out["dbg_query_traj_summary_path_length"] = zero
            out["dbg_query_traj_summary_straightness_ratio"] = zero

        for mode_idx in range(mode_count):
            mode_summary = self._compute_traj_summary_from_deltas(
                pred_tkm2[:, :, mode_idx, :],
                valid_f,
            )
            if mode_summary is None:
                out[f"dbg_query_traj_mode{mode_idx}_mean_speed"] = zero
                out[f"dbg_query_traj_mode{mode_idx}_endpoint_displacement"] = zero
                out[f"dbg_query_traj_mode{mode_idx}_path_length"] = zero
                out[f"dbg_query_traj_mode{mode_idx}_straightness_ratio"] = zero
            else:
                out[f"dbg_query_traj_mode{mode_idx}_mean_speed"] = mode_summary[0].mean()
                out[f"dbg_query_traj_mode{mode_idx}_endpoint_displacement"] = mode_summary[1].mean()
                out[f"dbg_query_traj_mode{mode_idx}_path_length"] = mode_summary[2].mean()
                out[f"dbg_query_traj_mode{mode_idx}_straightness_ratio"] = mode_summary[3].mean()
        return out

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
        pred_traj_offsets_fqk2: torch.Tensor = None,
        pred_endpoint_deltas_qk2: torch.Tensor = None,
        pred_traj_mode_logits_qk: torch.Tensor = None,
        pred_traj_static_gate_logits_q: torch.Tensor = None,
        pred_traj_moving_mode_logits_qm: torch.Tensor = None,
        inst_match_result: dict = None,
        loss_weight: float = 1.0,
        endpoint_loss_weight: float = 0.0,
        loss_type: str = "l1",
        present_local_idx: int = 0,
        mode_cls_loss_weight: float = 0.0,
        static_gate_loss_weight: float = 0.0,
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

        use_pred_targets = bool(getattr(self, "query_traj_pred_target_enabled", False))

        gt_centers_xy_tn2 = gt_centers_tn3.to(
            device=centers_world_tq3.device, dtype=torch.float32
        )[..., :2]
        gt_present_xy_n2 = gt_centers_xy_tn2[present_local_idx]
        gt_deltas_tn2 = (
            gt_centers_xy_tn2[(present_local_idx + 1):t_match]
            - gt_centers_xy_tn2[present_local_idx:(t_match - 1)]
        )
        gt_tk2 = gt_deltas_tn2.index_select(1, mi)
        pred_centers_xy_tq2 = None
        if use_pred_targets:
            pred_centers_xy_tq2 = centers_world_tq3.detach().to(torch.float32)[..., :2]
            gt_present_xy_n2 = pred_centers_xy_tq2[present_local_idx]
            gt_tk2 = (
                pred_centers_xy_tq2[(present_local_idx + 1):t_match].index_select(1, mq)
                - pred_centers_xy_tq2[present_local_idx:(t_match - 1)].index_select(1, mq)
            ).detach()

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
        pred_tkm2 = None
        best_mode_k = None
        pred_mode_k = None
        routed_mode_k = None
        mode_cls_loss = None
        static_gate_loss = None
        endpoint_loss = None
        gt_endpoint_k2 = None
        gt_endpoint_valid_k = None
        moving_target_k = None
        if (
            torch.is_tensor(pred_traj_offsets_fqk2)
            and pred_traj_offsets_fqk2.dim() == 4
            and int(pred_traj_offsets_fqk2.shape[1]) == Q
            and int(pred_traj_offsets_fqk2.shape[3]) == 2
        ):
            pred_steps = min(future_steps, int(pred_traj_offsets_fqk2.shape[0]))
            pred_tkm2 = pred_traj_offsets_fqk2[:pred_steps].to(
                device=centers_world_tq3.device, dtype=torch.float32
            ).index_select(1, mq)
            if pred_steps < future_steps:
                gt_tk2 = gt_tk2[:pred_steps]
                valid_f = valid_f[:pred_steps]
                valid_tk = valid_tk[:pred_steps]
            if pred_tkm2.numel() <= 0:
                return None
        elif (
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
        if pred_tkm2 is not None:
            if loss_type == "l2":
                err_tkm = torch.norm(pred_tkm2 - gt_tk2.unsqueeze(2), p=2, dim=-1)
            else:
                err_tkm = torch.abs(pred_tkm2 - gt_tk2.unsqueeze(2)).sum(dim=-1)
            mode_weight_tkm = valid_f.unsqueeze(-1)
            mode_denom_m1 = mode_weight_tkm.sum(dim=0).clamp_min(1.0)
            mode_err_mk = (err_tkm * mode_weight_tkm).sum(dim=0) / mode_denom_m1
            best_mode_k = torch.argmin(mode_err_mk, dim=-1)
            routed_mode_k = best_mode_k
            routing_enabled = bool(getattr(self, "query_traj_semantic_routing_enabled", False))
            stationary_enabled = bool(getattr(self, "query_traj_use_stationary_mode", False))
            cv_enabled = bool(getattr(self, "query_traj_use_cv_mode", False))
            mode_count = int(pred_tkm2.shape[2])
            stationary_mode_idx = 0 if stationary_enabled else None
            cv_mode_idx = int(stationary_enabled) if cv_enabled else None
            learned_start_idx = int(stationary_enabled) + int(cv_enabled)
            learned_mode_count = max(0, mode_count - learned_start_idx)
            if routing_enabled:
                gt_endpoint_k2 = (gt_tk2 * valid_f.unsqueeze(-1)).sum(dim=0)
                gt_endpoint_norm_k = torch.norm(gt_endpoint_k2, p=2, dim=-1)
                static_thr = float(getattr(self, "query_traj_static_threshold_m", 0.8))
                static_mask_k = gt_endpoint_norm_k <= static_thr
                cv_mask_k = valid_f.new_zeros((int(mq.numel()),), dtype=torch.bool)
                if cv_mode_idx is not None:
                    hist_has_prev = present_local_idx > 0
                    if hist_has_prev:
                        if use_pred_targets and torch.is_tensor(pred_centers_xy_tq2):
                            hist_pair_valid_k = torch.ones(
                                (int(mq.numel()),),
                                device=centers_world_tq3.device,
                                dtype=torch.bool,
                            )
                            gt_hist_vel_k2 = (
                                pred_centers_xy_tq2[present_local_idx].index_select(0, mq)
                                - pred_centers_xy_tq2[present_local_idx - 1].index_select(0, mq)
                            ).detach()
                        else:
                            hist_prev_valid_k = gt_valid_tn[present_local_idx - 1].to(
                                device=centers_world_tq3.device, dtype=torch.bool
                            ).index_select(0, mi)
                            hist_cur_valid_k = gt_valid_tn[present_local_idx].to(
                                device=centers_world_tq3.device, dtype=torch.bool
                            ).index_select(0, mi)
                            hist_pair_valid_k = hist_prev_valid_k & hist_cur_valid_k
                            gt_hist_vel_k2 = (
                                gt_centers_xy_tn2[present_local_idx].index_select(0, mi)
                                - gt_centers_xy_tn2[present_local_idx - 1].index_select(0, mi)
                            )
                        cv_ref_tk2 = gt_hist_vel_k2.unsqueeze(0).expand_as(gt_tk2)
                        cv_err_tk = torch.norm(gt_tk2 - cv_ref_tk2, p=2, dim=-1)
                        cv_weight_tk = valid_f * hist_pair_valid_k.to(valid_f.dtype).view(1, -1)
                        cv_denom_k = cv_weight_tk.sum(dim=0)
                        cv_err_k = torch.full_like(gt_endpoint_norm_k, fill_value=1e6)
                        valid_cv_k = cv_denom_k > 0
                        if bool(valid_cv_k.any().item()):
                            cv_err_k[valid_cv_k] = (
                                (cv_err_tk * cv_weight_tk).sum(dim=0)[valid_cv_k]
                                / cv_denom_k[valid_cv_k].clamp_min(1.0)
                            )
                        cv_thr = float(getattr(self, "query_traj_cv_error_threshold_m", 0.5))
                        cv_mask_k = (~static_mask_k) & valid_cv_k & (cv_err_k <= cv_thr)
                if bool(getattr(self, "query_traj_rule_turn_family_enabled", False)) and not cv_enabled:
                    routed_mode_k, static_mask_k, _, _ = self._apply_query_turn_family_routing(
                        best_mode_k=best_mode_k,
                        mode_err_mk=mode_err_mk,
                        gt_tk2=gt_tk2,
                        valid_f=valid_f,
                        gt_centers_xy_tn2=gt_centers_xy_tn2,
                        gt_valid_tn=gt_valid_tn,
                        present_local_idx=present_local_idx,
                        matched_inst_idx=mi,
                        matched_query_idx=mq,
                        pred_centers_xy_tq2=pred_centers_xy_tq2 if use_pred_targets else None,
                    )
                else:
                    routed_mode_k = best_mode_k.clone()
                    if learned_mode_count > 0:
                        learned_err_mk = mode_err_mk[:, learned_start_idx:]
                        learned_best_local_k = torch.argmin(learned_err_mk, dim=-1)
                        learned_best_mode_k = learned_best_local_k + learned_start_idx
                        routed_mode_k = torch.where(
                            (~static_mask_k) & (~cv_mask_k),
                            learned_best_mode_k,
                            routed_mode_k,
                        )
                    if stationary_mode_idx is not None:
                        routed_mode_k = torch.where(
                            static_mask_k,
                            routed_mode_k.new_full(routed_mode_k.shape, int(stationary_mode_idx)),
                            routed_mode_k,
                        )
                if cv_mode_idx is not None and routed_mode_k is not None:
                    routed_mode_k = torch.where(
                        cv_mask_k,
                        routed_mode_k.new_full(routed_mode_k.shape, int(cv_mode_idx)),
                        routed_mode_k,
                    )
            pred_tk2 = pred_tkm2.gather(
                2,
                routed_mode_k.view(1, -1, 1, 1).expand(int(pred_tkm2.shape[0]), -1, 1, 2),
            ).squeeze(2)
            if (
                torch.is_tensor(pred_traj_mode_logits_qk)
                and pred_traj_mode_logits_qk.dim() == 2
                and int(pred_traj_mode_logits_qk.shape[0]) == Q
                and int(pred_traj_mode_logits_qk.shape[1]) == int(pred_tkm2.shape[2])
            ):
                mode_logits_mk = pred_traj_mode_logits_qk.to(
                    device=centers_world_tq3.device, dtype=torch.float32
                ).index_select(0, mq)
                pred_mode_k = torch.argmax(mode_logits_mk, dim=-1)
            if bool(getattr(self, "query_traj_static_gate_enabled", False)):
                if (
                    torch.is_tensor(pred_traj_static_gate_logits_q)
                    and pred_traj_static_gate_logits_q.dim() == 1
                    and int(pred_traj_static_gate_logits_q.shape[0]) == Q
                    and routed_mode_k is not None
                    and stationary_mode_idx is not None
                    and float(static_gate_loss_weight) > 0.0
                ):
                    static_logits_k = pred_traj_static_gate_logits_q.to(
                        device=centers_world_tq3.device, dtype=torch.float32
                    ).index_select(0, mq)
                    static_target_k = (routed_mode_k == int(stationary_mode_idx)).to(torch.float32)
                    static_gate_loss = F.binary_cross_entropy_with_logits(
                        static_logits_k,
                        static_target_k,
                        reduction="mean",
                    )
                if (
                    float(mode_cls_loss_weight) > 0.0
                    and torch.is_tensor(pred_traj_moving_mode_logits_qm)
                    and pred_traj_moving_mode_logits_qm.dim() == 2
                    and int(pred_traj_moving_mode_logits_qm.shape[0]) == Q
                    and routed_mode_k is not None
                ):
                    moving_logits_mk = pred_traj_moving_mode_logits_qm.to(
                        device=centers_world_tq3.device, dtype=torch.float32
                    ).index_select(0, mq)
                    moving_mask_k = routed_mode_k >= learned_start_idx
                    if bool(moving_mask_k.any().item()):
                        moving_target_k = routed_mode_k[moving_mask_k] - learned_start_idx
                        mode_cls_loss = F.cross_entropy(
                            moving_logits_mk[moving_mask_k],
                            moving_target_k,
                            reduction="mean",
                        )
            elif (
                float(mode_cls_loss_weight) > 0.0
                and pred_mode_k is not None
            ):
                mode_cls_loss = F.cross_entropy(mode_logits_mk, routed_mode_k, reduction="mean")

        if float(endpoint_loss_weight) > 0.0:
            future_valid_kn = gt_valid_tn[(present_local_idx + 1):t_match].to(
                device=centers_world_tq3.device, dtype=torch.bool
            ).transpose(0, 1).index_select(0, mi)
            future_xy_knf2 = gt_centers_xy_tn2[(present_local_idx + 1):t_match].to(
                device=centers_world_tq3.device, dtype=torch.float32
            ).permute(1, 0, 2).index_select(0, mi)
            gt_present_xy_k2 = gt_present_xy_n2.index_select(0, mi)
            if use_pred_targets and torch.is_tensor(pred_centers_xy_tq2):
                future_valid_kn = torch.ones_like(future_valid_kn, dtype=torch.bool)
                future_xy_knf2 = pred_centers_xy_tq2[(present_local_idx + 1):t_match].permute(1, 0, 2).index_select(0, mq)
                gt_present_xy_k2 = gt_present_xy_n2.index_select(0, mq)
            last_valid_idx_k = future_valid_kn.to(torch.long).sum(dim=1) - 1
            gt_endpoint_valid_k = last_valid_idx_k >= 0
            if bool(gt_endpoint_valid_k.any().item()):
                safe_last_valid_idx_k = last_valid_idx_k.clamp_min(0)
                gather_idx = safe_last_valid_idx_k.view(-1, 1, 1).expand(-1, 1, 2)
                gt_endpoint_abs_k2 = torch.gather(
                    future_xy_knf2, 1, gather_idx
                ).squeeze(1)
                gt_endpoint_k2 = gt_endpoint_abs_k2 - gt_present_xy_k2
                pred_endpoint_k2 = None
                if (
                    torch.is_tensor(pred_endpoint_deltas_qk2)
                    and pred_endpoint_deltas_qk2.dim() == 3
                    and int(pred_endpoint_deltas_qk2.shape[0]) == Q
                    and int(pred_endpoint_deltas_qk2.shape[2]) == 2
                ):
                    pred_endpoint_qk2 = pred_endpoint_deltas_qk2.to(
                        device=centers_world_tq3.device, dtype=torch.float32
                    ).index_select(0, mq)
                    if (
                        routed_mode_k is not None
                        and int(pred_endpoint_qk2.shape[1]) == int(pred_tkm2.shape[2])
                    ):
                        pred_endpoint_k2 = pred_endpoint_qk2.gather(
                            1,
                            routed_mode_k.view(-1, 1, 1).expand(-1, 1, 2),
                        ).squeeze(1)
                    elif int(pred_endpoint_qk2.shape[1]) > 0:
                        endpoint_err_km = torch.abs(
                            pred_endpoint_qk2 - gt_endpoint_k2.unsqueeze(1)
                        ).sum(dim=-1)
                        endpoint_best_mode_k = torch.argmin(endpoint_err_km, dim=-1)
                        pred_endpoint_k2 = pred_endpoint_qk2.gather(
                            1,
                            endpoint_best_mode_k.view(-1, 1, 1).expand(-1, 1, 2),
                        ).squeeze(1)
                if pred_endpoint_k2 is not None:
                    if loss_type == "l2":
                        endpoint_err_k = torch.norm(pred_endpoint_k2 - gt_endpoint_k2, p=2, dim=-1)
                    else:
                        endpoint_err_k = torch.abs(pred_endpoint_k2 - gt_endpoint_k2).sum(dim=-1)
                    endpoint_valid_f = gt_endpoint_valid_k.to(endpoint_err_k.dtype)
                    endpoint_denom = endpoint_valid_f.sum()
                    if bool((endpoint_denom > 0).item()):
                        endpoint_loss = (
                            (endpoint_err_k * endpoint_valid_f).sum()
                            / endpoint_denom.clamp_min(1.0)
                        ) * float(endpoint_loss_weight)

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

        moving_gt_mask_k = None
        if bool(getattr(self, "query_traj_static_gate_enabled", False)) and routed_mode_k is not None:
            moving_gt_mask_k = (routed_mode_k >= learned_start_idx).to(valid_f.dtype).view(1, -1)
        eff_weight_tk = valid_f * traj_weight_tk
        if moving_gt_mask_k is not None:
            eff_weight_tk = eff_weight_tk * moving_gt_mask_k
        eff_cnt = eff_weight_tk.sum()
        if bool((eff_cnt > 0).item()):
            loss_reg_val = ((err_tk * eff_weight_tk).sum() / eff_cnt.clamp_min(1.0)) * float(loss_weight)
        else:
            loss_reg_val = valid_f.sum() * 0.0
        loss_static_val = (
            static_gate_loss * float(static_gate_loss_weight)
            if torch.is_tensor(static_gate_loss) and float(static_gate_loss_weight) > 0.0
            else loss_reg_val.new_zeros(())
        )
        loss_moving_cls_val = (
            mode_cls_loss * float(mode_cls_loss_weight)
            if torch.is_tensor(mode_cls_loss) and float(mode_cls_loss_weight) > 0.0
            else loss_reg_val.new_zeros(())
        )
        loss_moving_reg_val = loss_reg_val
        loss_val = loss_static_val + loss_moving_cls_val + loss_moving_reg_val
        if torch.is_tensor(mode_cls_loss) and float(mode_cls_loss_weight) > 0.0:
            pass
        if torch.is_tensor(static_gate_loss) and float(static_gate_loss_weight) > 0.0:
            pass
        if torch.is_tensor(endpoint_loss):
            loss_val = loss_val + endpoint_loss

        moving_mask_tk = motion_mag_tk >= float(moving_threshold_m)
        static_target_k = (
            (routed_mode_k == int(stationary_mode_idx)).to(torch.float32)
            if (routed_mode_k is not None and stationary_mode_idx is not None)
            else loss_val.new_zeros((int(mq.numel()),))
        )
        static_prob_k = None
        if (
            torch.is_tensor(pred_traj_static_gate_logits_q)
            and pred_traj_static_gate_logits_q.dim() == 1
            and int(pred_traj_static_gate_logits_q.shape[0]) == Q
        ):
            static_prob_k = torch.sigmoid(
                pred_traj_static_gate_logits_q.to(
                    device=centers_world_tq3.device, dtype=torch.float32
                ).index_select(0, mq)
            )
        out = {
            "loss_query_traj": loss_val,
            "loss_query_traj_static": loss_static_val,
            "loss_query_traj_moving_cls": loss_moving_cls_val,
            "loss_query_traj_moving_reg": loss_moving_reg_val,
            "dbg_query_traj_reg_loss": loss_reg_val,
            "dbg_query_endpoint_loss": (
                endpoint_loss.detach()
                if torch.is_tensor(endpoint_loss)
                else loss_val.new_zeros(())
            ),
            "dbg_query_traj_mode_cls_loss": (
                mode_cls_loss.detach()
                if torch.is_tensor(mode_cls_loss)
                else loss_val.new_zeros(())
            ),
            "dbg_query_traj_static_gate_loss": (
                static_gate_loss.detach()
                if torch.is_tensor(static_gate_loss)
                else loss_val.new_zeros(())
            ),
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
            "dbg_query_endpoint_valid_count": (
                gt_endpoint_valid_k.to(torch.float32).sum()
                if torch.is_tensor(gt_endpoint_valid_k)
                else loss_val.new_zeros(())
            ),
            "dbg_query_traj_static_target_ratio": static_target_k.mean() if int(static_target_k.numel()) > 0 else loss_val.new_zeros(()),
            "dbg_query_traj_static_prob_mean": (
                static_prob_k.mean()
                if torch.is_tensor(static_prob_k) and int(static_prob_k.numel()) > 0
                else loss_val.new_zeros(())
            ),
            "dbg_query_traj_static_pred_ratio": (
                (static_prob_k >= float(getattr(self, "query_traj_static_gate_threshold", 0.5))).to(torch.float32).mean()
                if torch.is_tensor(static_prob_k) and int(static_prob_k.numel()) > 0
                else loss_val.new_zeros(())
            ),
            "dbg_query_traj_moving_gt_ratio": (
                moving_gt_mask_k.mean()
                if torch.is_tensor(moving_gt_mask_k) and int(moving_gt_mask_k.numel()) > 0
                else loss_val.new_zeros(())
            ),
        }
        selected_mode_k = pred_mode_k if pred_mode_k is not None else routed_mode_k
        out = self._append_query_traj_mode_summary_debug(
            out=out,
            pred_tkm2=pred_tkm2,
            valid_f=valid_f,
            selected_mode_k=selected_mode_k,
        )
        if best_mode_k is not None:
            mode_count = int(pred_tkm2.shape[2])
            builtin_mode_count = int(getattr(self, "query_traj_use_stationary_mode", False)) + int(
                getattr(self, "query_traj_use_cv_mode", False)
            )
            out["dbg_query_traj_best_mode_mean"] = best_mode_k.to(torch.float32).mean()
            if routed_mode_k is not None:
                out["dbg_query_traj_routed_mode_mean"] = routed_mode_k.to(torch.float32).mean()
            for mode_idx in range(mode_count):
                out[f"dbg_query_traj_pred_mode{mode_idx}_ratio"] = (
                    (
                        (pred_mode_k == int(mode_idx)).to(torch.float32).mean()
                        if pred_mode_k is not None
                        else (routed_mode_k == int(mode_idx)).to(torch.float32).mean()
                    )
                )
                if routed_mode_k is not None:
                    out[f"dbg_query_traj_route_mode{mode_idx}_ratio"] = (
                        (routed_mode_k == int(mode_idx)).to(torch.float32).mean()
                    )
            if pred_mode_k is not None:
                out["dbg_query_traj_pred_builtin_mode_ratio"] = (
                    (pred_mode_k < builtin_mode_count).to(torch.float32).mean()
                )
                out["dbg_query_traj_pred_learned_mode_ratio"] = (
                    (pred_mode_k >= builtin_mode_count).to(torch.float32).mean()
                )
            else:
                out["dbg_query_traj_pred_builtin_mode_ratio"] = (
                    (routed_mode_k < builtin_mode_count).to(torch.float32).mean()
                )
                out["dbg_query_traj_pred_learned_mode_ratio"] = (
                    (routed_mode_k >= builtin_mode_count).to(torch.float32).mean()
                )
            out["dbg_query_traj_route_builtin_mode_ratio"] = (
                (routed_mode_k < builtin_mode_count).to(torch.float32).mean()
            )
            out["dbg_query_traj_route_learned_mode_ratio"] = (
                (routed_mode_k >= builtin_mode_count).to(torch.float32).mean()
            )
        return out

    def _build_query_trajectory_mode_debug_from_match(
        self,
        centers_world_tq3: torch.Tensor,
        pred_traj_offsets_fqk2: torch.Tensor = None,
        pred_traj_mode_logits_qk: torch.Tensor = None,
        inst_match_result: dict = None,
        present_local_idx: int = 0,
    ):
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        if (
            (not torch.is_tensor(pred_traj_offsets_fqk2))
            or pred_traj_offsets_fqk2.dim() != 4
            or int(pred_traj_offsets_fqk2.shape[1]) != int(centers_world_tq3.shape[1])
        ):
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
        pred_steps = min(future_steps, int(pred_traj_offsets_fqk2.shape[0]))
        if pred_steps <= 0:
            return None

        gt_centers_xy_tn2 = gt_centers_tn3.to(
            device=centers_world_tq3.device, dtype=torch.float32
        )[..., :2]
        gt_deltas_tn2 = (
            gt_centers_xy_tn2[(present_local_idx + 1):t_match]
            - gt_centers_xy_tn2[present_local_idx:(t_match - 1)]
        )
        gt_tk2 = gt_deltas_tn2.index_select(1, mi)[:pred_steps]
        valid_prev_tk = gt_valid_tn[present_local_idx:(t_match - 1)].to(
            device=centers_world_tq3.device, dtype=torch.bool
        ).index_select(1, mi)[:pred_steps]
        valid_next_tk = gt_valid_tn[(present_local_idx + 1):t_match].to(
            device=centers_world_tq3.device, dtype=torch.bool
        ).index_select(1, mi)[:pred_steps]
        valid_tk = valid_prev_tk & valid_next_tk
        valid_f = valid_tk.to(torch.float32)
        if not bool((valid_f.sum() > 0).item()):
            return None

        pred_tkm2 = pred_traj_offsets_fqk2[:pred_steps].to(
            device=centers_world_tq3.device, dtype=torch.float32
        ).index_select(1, mq)
        if pred_tkm2.numel() <= 0:
            return None

        err_tkm = torch.abs(pred_tkm2 - gt_tk2.unsqueeze(2)).sum(dim=-1)
        mode_weight_tkm = valid_f.unsqueeze(-1)
        mode_denom_m1 = mode_weight_tkm.sum(dim=0).clamp_min(1.0)
        mode_err_mk = (err_tkm * mode_weight_tkm).sum(dim=0) / mode_denom_m1
        best_mode_k = torch.argmin(mode_err_mk, dim=-1)
        routed_mode_k = best_mode_k.clone()

        routing_enabled = bool(getattr(self, "query_traj_semantic_routing_enabled", False))
        stationary_enabled = bool(getattr(self, "query_traj_use_stationary_mode", False))
        cv_enabled = bool(getattr(self, "query_traj_use_cv_mode", False))
        mode_count = int(pred_tkm2.shape[2])
        stationary_mode_idx = 0 if stationary_enabled else None
        cv_mode_idx = int(stationary_enabled) if cv_enabled else None
        learned_start_idx = int(stationary_enabled) + int(cv_enabled)
        learned_mode_count = max(0, mode_count - learned_start_idx)
        if routing_enabled:
            gt_endpoint_k2 = (gt_tk2 * valid_f.unsqueeze(-1)).sum(dim=0)
            gt_endpoint_norm_k = torch.norm(gt_endpoint_k2, p=2, dim=-1)
            static_thr = float(getattr(self, "query_traj_static_threshold_m", 0.8))
            static_mask_k = gt_endpoint_norm_k <= static_thr
            cv_mask_k = static_mask_k.new_zeros(static_mask_k.shape)
            if cv_mode_idx is not None and present_local_idx > 0:
                hist_prev_valid_k = gt_valid_tn[present_local_idx - 1].to(
                    device=centers_world_tq3.device, dtype=torch.bool
                ).index_select(0, mi)
                hist_cur_valid_k = gt_valid_tn[present_local_idx].to(
                    device=centers_world_tq3.device, dtype=torch.bool
                ).index_select(0, mi)
                hist_pair_valid_k = hist_prev_valid_k & hist_cur_valid_k
                gt_hist_vel_k2 = (
                    gt_centers_xy_tn2[present_local_idx].index_select(0, mi)
                    - gt_centers_xy_tn2[present_local_idx - 1].index_select(0, mi)
                )
                cv_ref_tk2 = gt_hist_vel_k2.unsqueeze(0).expand_as(gt_tk2)
                cv_err_tk = torch.norm(gt_tk2 - cv_ref_tk2, p=2, dim=-1)
                cv_weight_tk = valid_f * hist_pair_valid_k.to(valid_f.dtype).view(1, -1)
                cv_denom_k = cv_weight_tk.sum(dim=0)
                cv_err_k = torch.full_like(gt_endpoint_norm_k, fill_value=1e6)
                valid_cv_k = cv_denom_k > 0
                if bool(valid_cv_k.any().item()):
                    cv_err_k[valid_cv_k] = (
                        (cv_err_tk * cv_weight_tk).sum(dim=0)[valid_cv_k]
                        / cv_denom_k[valid_cv_k].clamp_min(1.0)
                    )
                cv_thr = float(getattr(self, "query_traj_cv_error_threshold_m", 0.5))
                cv_mask_k = (~static_mask_k) & valid_cv_k & (cv_err_k <= cv_thr)
            if bool(getattr(self, "query_traj_rule_turn_family_enabled", False)) and not cv_enabled:
                routed_mode_k, static_mask_k, _, _ = self._apply_query_turn_family_routing(
                    best_mode_k=best_mode_k,
                    mode_err_mk=mode_err_mk,
                    gt_tk2=gt_tk2,
                    valid_f=valid_f,
                    gt_centers_xy_tn2=gt_centers_xy_tn2,
                    gt_valid_tn=gt_valid_tn,
                    present_local_idx=present_local_idx,
                    matched_inst_idx=mi,
                )
            else:
                if learned_mode_count > 0:
                    learned_err_mk = mode_err_mk[:, learned_start_idx:]
                    learned_best_local_k = torch.argmin(learned_err_mk, dim=-1)
                    learned_best_mode_k = learned_best_local_k + learned_start_idx
                    routed_mode_k = torch.where(
                        (~static_mask_k) & (~cv_mask_k),
                        learned_best_mode_k,
                        routed_mode_k,
                    )
                if stationary_mode_idx is not None:
                    routed_mode_k = torch.where(
                        static_mask_k,
                        routed_mode_k.new_full(routed_mode_k.shape, int(stationary_mode_idx)),
                        routed_mode_k,
                    )
            if cv_mode_idx is not None and routed_mode_k is not None:
                routed_mode_k = torch.where(
                    cv_mask_k,
                    routed_mode_k.new_full(routed_mode_k.shape, int(cv_mode_idx)),
                    routed_mode_k,
                )

        pred_mode_k = None
        if (
            torch.is_tensor(pred_traj_mode_logits_qk)
            and pred_traj_mode_logits_qk.dim() == 2
            and int(pred_traj_mode_logits_qk.shape[0]) == Q
            and int(pred_traj_mode_logits_qk.shape[1]) == mode_count
        ):
            mode_logits_mk = pred_traj_mode_logits_qk.to(
                device=centers_world_tq3.device, dtype=torch.float32
            ).index_select(0, mq)
            pred_mode_k = torch.argmax(mode_logits_mk, dim=-1)

        out = {
            "dbg_query_traj_best_mode_mean": best_mode_k.to(torch.float32).mean(),
            "dbg_query_traj_routed_mode_mean": routed_mode_k.to(torch.float32).mean(),
        }
        selected_mode_k = pred_mode_k if pred_mode_k is not None else routed_mode_k
        out = self._append_query_traj_mode_summary_debug(
            out=out,
            pred_tkm2=pred_tkm2,
            valid_f=valid_f,
            selected_mode_k=selected_mode_k,
        )
        builtin_mode_count = int(getattr(self, "query_traj_use_stationary_mode", False)) + int(
            getattr(self, "query_traj_use_cv_mode", False)
        )
        for mode_idx in range(mode_count):
            out[f"dbg_query_traj_pred_mode{mode_idx}_ratio"] = (
                (pred_mode_k == int(mode_idx)).to(torch.float32).mean()
                if pred_mode_k is not None
                else (routed_mode_k == int(mode_idx)).to(torch.float32).mean()
            )
            out[f"dbg_query_traj_route_mode{mode_idx}_ratio"] = (
                (routed_mode_k == int(mode_idx)).to(torch.float32).mean()
            )
        if pred_mode_k is not None:
            out["dbg_query_traj_pred_builtin_mode_ratio"] = (
                (pred_mode_k < builtin_mode_count).to(torch.float32).mean()
            )
            out["dbg_query_traj_pred_learned_mode_ratio"] = (
                (pred_mode_k >= builtin_mode_count).to(torch.float32).mean()
            )
        else:
            out["dbg_query_traj_pred_builtin_mode_ratio"] = (
                (routed_mode_k < builtin_mode_count).to(torch.float32).mean()
            )
            out["dbg_query_traj_pred_learned_mode_ratio"] = (
                (routed_mode_k >= builtin_mode_count).to(torch.float32).mean()
            )
        out["dbg_query_traj_route_builtin_mode_ratio"] = (
            (routed_mode_k < builtin_mode_count).to(torch.float32).mean()
        )
        out["dbg_query_traj_route_learned_mode_ratio"] = (
            (routed_mode_k >= builtin_mode_count).to(torch.float32).mean()
        )
        return out

    def _compute_query_trajectory_refine_xy_loss_from_match(
        self,
        pred_future_xy_fq2: torch.Tensor,
        inst_match_result: dict = None,
        loss_weight: float = 1.0,
        loss_type: str = "l1",
        present_local_idx: int = 0,
        moving_reweight_enabled: bool = False,
        moving_threshold_m: float = 0.5,
        moving_weight: float = 5.0,
        static_weight: float = 1.0,
    ):
        if float(loss_weight) <= 0.0:
            return None
        if (not torch.is_tensor(pred_future_xy_fq2)) or pred_future_xy_fq2.dim() != 3:
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

        future_steps, q_count, _ = [int(v) for v in pred_future_xy_fq2.shape]
        tg, n_count, _ = [int(v) for v in gt_centers_tn3.shape]
        t_match = min(int(gt_valid_tn.shape[0]), tg)
        present_local_idx = max(0, min(t_match - 1, int(present_local_idx)))
        avail_future = max(0, t_match - present_local_idx - 1)
        if future_steps <= 0 or avail_future <= 0:
            return None

        mq = matched_query_idx.to(device=pred_future_xy_fq2.device, dtype=torch.long)
        mi = matched_inst_idx.to(device=pred_future_xy_fq2.device, dtype=torch.long)
        keep = (mq >= 0) & (mq < q_count) & (mi >= 0) & (mi < n_count)
        if not bool(keep.any().item()):
            return None
        mq = mq[keep]
        mi = mi[keep]
        if mq.numel() <= 0:
            return None

        pred_steps = min(future_steps, avail_future)
        pred_fk2 = pred_future_xy_fq2[:pred_steps].to(torch.float32).index_select(1, mq)
        gt_fk2 = gt_centers_tn3[(present_local_idx + 1):(present_local_idx + 1 + pred_steps), :, :2].to(
            device=pred_future_xy_fq2.device,
            dtype=torch.float32,
        ).index_select(1, mi)
        valid_fk = gt_valid_tn[(present_local_idx + 1):(present_local_idx + 1 + pred_steps)].to(
            device=pred_future_xy_fq2.device,
            dtype=torch.bool,
        ).index_select(1, mi)
        valid_f = valid_fk.to(torch.float32)
        if not bool((valid_f.sum() > 0).item()):
            return None

        if loss_type == "l2":
            err_fk = torch.norm(pred_fk2 - gt_fk2, p=2, dim=-1)
        else:
            err_fk = torch.abs(pred_fk2 - gt_fk2).sum(dim=-1)

        traj_weight_fk = torch.ones_like(valid_f)
        if bool(moving_reweight_enabled):
            gt_deltas_fk2 = gt_fk2.clone()
            gt_prev_xy_k2 = gt_centers_tn3[present_local_idx, :, :2].to(
                device=pred_future_xy_fq2.device,
                dtype=torch.float32,
            ).index_select(0, mi)
            gt_deltas_fk2[0] = gt_fk2[0] - gt_prev_xy_k2
            if pred_steps > 1:
                gt_deltas_fk2[1:] = gt_fk2[1:] - gt_fk2[:-1]
            motion_mag_fk = torch.norm(gt_deltas_fk2, p=2, dim=-1)
            traj_weight_fk = torch.where(
                motion_mag_fk >= float(moving_threshold_m),
                torch.full_like(valid_f, float(moving_weight)),
                torch.full_like(valid_f, float(static_weight)),
            )
        else:
            motion_mag_fk = gt_fk2.new_zeros(gt_fk2.shape[:2])

        eff_weight_fk = valid_f * traj_weight_fk
        eff_cnt = eff_weight_fk.sum()
        if not bool((eff_cnt > 0).item()):
            return None
        loss_val = ((err_fk * eff_weight_fk).sum() / eff_cnt.clamp_min(1.0)) * float(loss_weight)
        return {
            "loss_query_traj_refine_xy": loss_val,
            "dbg_query_traj_refine_xy_valid_count": valid_f.sum(),
            "dbg_query_traj_refine_xy_weight_mean": self._safe_weighted_mean(traj_weight_fk, valid_f),
            "dbg_query_traj_refine_xy_motion_mag_mean": self._safe_weighted_mean(motion_mag_fk, valid_f),
        }

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
        if (
            t_attn <= 0
            or q_attn <= 0
            or n_cam <= 0
            or t_gt <= 0
            or n_inst < 0
            or (h_attn != h_gt)
            or (w_attn != w_gt)
            or (t_gt != t_valid)
        ):
            out["dbg_query_attn_bbox_shape_invalid_skip"] = z.new_tensor(1.0)
            return out

        t = min(t_attn, t_gt)
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
        unmatched_mode = str(getattr(self, "query_attn_bbox_unmatched_mode", "inverse_union")).lower()
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
            valid_tk = valid_tk & torch.isfinite(pred_tkp).all(dim=-1)
            valid_f = valid_tk.to(torch.float32)
            matched_valid_cnt = valid_f.sum()
            if bool((matched_valid_cnt > 0).item()):
                inside_mass_tk = (pred_tkp * target_mask_tkp).sum(dim=-1).clamp(0.0, 1.0)
                outside_mass_tk = (1.0 - inside_mass_tk).clamp(0.0, 1.0)
                loss_tk = -torch.log(inside_mass_tk.clamp_min(eps_v))
                matched_raw = (loss_tk * valid_f).sum() / matched_valid_cnt.clamp_min(1.0)
                inside_valid = inside_mass_tk[valid_tk]
                outside_valid = outside_mass_tk[valid_tk]
                out["dbg_query_attn_bbox_inside_mass_mean"] = inside_valid.mean()
                out["dbg_query_attn_bbox_inside_mass_min"] = inside_valid.min()
                out["dbg_query_attn_bbox_inside_mass_max"] = inside_valid.max()
                out["dbg_query_attn_bbox_outside_mass_mean"] = outside_valid.mean()
                pred_entropy_valid = -(pred_tkp.clamp_min(eps_v) * pred_tkp.clamp_min(eps_v).log()).sum(dim=-1)
                pred_entropy_terms.append(pred_entropy_valid[valid_tk])
                target_prob_tkp = target_mask_tkp / target_sum_tk.unsqueeze(-1).clamp_min(eps_v)
                target_entropy_valid = -(target_prob_tkp.clamp_min(eps_v) * target_prob_tkp.clamp_min(eps_v).log()).sum(dim=-1)
                target_entropy_terms.append(target_entropy_valid[valid_tk])

                if other_w > 0.0:
                    other_mask_tkp = union_mask_tp.unsqueeze(1) - target_mask_tkp
                    other_mask_tkp = other_mask_tkp.clamp_(0.0, 1.0)
                    other_sum_tk = other_mask_tkp.sum(dim=-1)
                    other_valid_tk = valid_tk & (other_sum_tk > 0.0)
                    other_valid_f = other_valid_tk.to(torch.float32)
                    other_valid_cnt = other_valid_f.sum()
                    if bool((other_valid_cnt > 0).item()):
                        other_mass_tk = (pred_tkp * other_mask_tkp).sum(dim=-1).clamp(0.0, 1.0)
                        if other_mode == "union_log":
                            other_loss_tk = -torch.log((1.0 - other_mass_tk).clamp_min(eps_v))
                        else:
                            other_loss_tk = other_mass_tk
                        other_raw = (other_loss_tk * other_valid_f).sum() / other_valid_cnt.clamp_min(1.0)
                        other_mass_valid = other_mass_tk[other_valid_tk]
                        out["dbg_query_attn_bbox_other_mass_mean"] = other_mass_valid.mean()
                        out["dbg_query_attn_bbox_other_mass_max"] = other_mass_valid.max()

        if unmatched_w > 0.0:
            matched_mask_q = torch.zeros((q_attn,), device=pred_tqp.device, dtype=torch.bool)
            if k > 0:
                matched_mask_q[mq] = True
            uq = torch.nonzero(~matched_mask_q, as_tuple=False).squeeze(1)
            u = int(uq.numel())
            out["dbg_query_attn_bbox_unmatched_count"] = z.new_tensor(float(u))
            if u > 0:
                pred_tup = pred_tqp.index_select(1, uq)
                union_mask_tup = union_mask_tp.unsqueeze(1)
                union_mass_tu = (pred_tup * union_mask_tup).sum(dim=-1).clamp(0.0, 1.0)
                unmatched_valid_tu = (union_sum_t1.squeeze(-1) > 0.0).unsqueeze(1).expand(t, u)
                unmatched_valid_tu = unmatched_valid_tu & torch.isfinite(pred_tup).all(dim=-1)
                unmatched_valid_f = unmatched_valid_tu.to(torch.float32)
                unmatched_valid_cnt = unmatched_valid_f.sum()
                if bool((unmatched_valid_cnt > 0).item()):
                    if unmatched_mode == "inverse_union_log":
                        unmatched_loss_tu = -torch.log((1.0 - union_mass_tu).clamp_min(eps_v))
                    else:
                        unmatched_loss_tu = union_mass_tu
                    unmatched_raw = (unmatched_loss_tu * unmatched_valid_f).sum() / unmatched_valid_cnt.clamp_min(1.0)
                    unmatched_mass_valid = union_mass_tu[unmatched_valid_tu]
                    out["dbg_query_attn_bbox_unmatched_mass_mean"] = unmatched_mass_valid.mean()
                    out["dbg_query_attn_bbox_unmatched_mass_max"] = unmatched_mass_valid.max()
                    pred_entropy_unmatched = -(pred_tup.clamp_min(eps_v) * pred_tup.clamp_min(eps_v).log()).sum(dim=-1)
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
        mixture_yaw_tqg: torch.Tensor,
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
            or (not torch.is_tensor(mixture_yaw_tqg))
            or (not torch.is_tensor(mixture_weights_tqg))
            or tuple(mixture_yaw_tqg.shape) != tuple(mixture_weights_tqg.shape)
            or tuple(mixture_yaw_tqg.shape) != tuple(mixture_centers_world_tqg3.shape[:3])
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
        mix_yaw_tkg = mixture_yaw_tqg[:T].index_select(1, mq).to(torch.float32)
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
            mixture_yaw_tkg=mix_yaw_tkg,
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

    def _compute_matched_query_sequence_iou_scores(
        self,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        gt_instance_occ3d_txyz_pred: torch.Tensor,
        inst_match_result: dict = None,
        objectness_scores_tq: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_yaw_tqg: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        pair_chunk_size: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        if (not torch.is_tensor(sigmas_world_tq3)) or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape):
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
            and torch.is_tensor(mixture_yaw_tqg)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_yaw_tqg.shape)
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
                mixture_yaw_tqg=mixture_yaw_tqg[:t_count],
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

    def _compute_matched_pair_gmo_losses(
        self,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        gt_instance_occ3d_txyz_pred: torch.Tensor,
        objectness_scores_tq: torch.Tensor = None,
        inst_match_result: dict = None,
        loss_weight: float = 0.1,
        loss_type: str = "balanced_bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        compute_dice: bool = False,
        dice_loss_weight: float = 1.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_yaw_tqg: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        pair_chunk_size: int = 8,
        eps: float = 1e-6,
    ) -> dict:
        z = centers_world_tq3.sum() * 0.0
        loss_type = str(loss_type).lower()
        loss_key = "loss_gmo_focal" if loss_type in ("balanced_focal", "focal") else "loss_gmo_bce"
        out = {
            "dbg_gmo_bce_pair_count": z,
            "dbg_gmo_bce_lowres_x": z,
            "dbg_gmo_bce_lowres_y": z,
            "dbg_gmo_bce_lowres_z": z,
            "loss_gmo_dice": z,
            "dbg_gmo_dice_pair_count": z,
            "dbg_gmo_dice_alpha": z.new_tensor(float(tversky_alpha)),
            "dbg_gmo_dice_beta": z.new_tensor(float(tversky_beta)),
        }
        out[loss_key] = z
        if loss_key == "loss_gmo_focal":
            out["dbg_gmo_bce_alias"] = z
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return out
        if (not torch.is_tensor(sigmas_world_tq3)) or tuple(sigmas_world_tq3.shape) != tuple(centers_world_tq3.shape):
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

        gt_occ_txyz = gt_instance_occ3d_txyz_pred[:T].to(device=centers_world_tq3.device, dtype=torch.long)
        if torch.is_tensor(gt_valid_tn) and gt_valid_tn.dim() == 2 and int(gt_valid_tn.shape[0]) >= T:
            gt_valid_sel_tk = gt_valid_tn[:T].to(device=centers_world_tq3.device, dtype=torch.bool).index_select(1, mi)
        else:
            gt_valid_sel_tk = torch.ones((T, int(mq.numel())), device=centers_world_tq3.device, dtype=torch.bool)

        pair_losses = []
        dice_pair_losses = []
        use_mixture = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_yaw_tqg)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_yaw_tqg.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
        )
        if use_mixture:
            pred_tk1zyx, gt_tk1zyx, valid_tk = self._prepare_grouped_matched_pair_lowres_occ(
                gt_occ_txyz=gt_occ_txyz,
                matched_query_idx=mq,
                pair_gt_ids_k=pair_gt_ids,
                gt_valid_sel_tk=gt_valid_sel_tk,
                mixture_centers_world_tqg3=mixture_centers_world_tqg3[:T],
                mixture_sigmas_world_tqg3=mixture_sigmas_world_tqg3[:T],
                mixture_yaw_tqg=mixture_yaw_tqg[:T],
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
        query_traj_refine_loss,
        query_attn_cam_score_pack,
        # Query predictions
        centers_world,
        centers_world_match_tq3,
        centers_world_dt_gt2p_tq3,
        gaussian_sigmas_world_loss,
        mixture_sigmas_world_loss,
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

        # ---- Center match loss ----
        losses["loss_query_center_match"] = center_match_loss if center_match_loss is not None else z
        center_match_dbg = self._compute_query_center_match_dbg_from_match(
            centers_world_tq3=centers_world_match_tq3,
            inst_match_result=inst_match_result,
        )
        if isinstance(center_match_dbg, dict):
            losses.update(center_match_dbg)

        # ---- Trajectory loss ----
        losses["loss_query_traj_static"] = z
        losses["loss_query_traj_moving_cls"] = z
        losses["loss_query_traj_moving_reg"] = z
        traj_dbg_defaults = {
            "dbg_query_traj_reg_loss": z,
            "dbg_query_endpoint_loss": z,
            "dbg_query_traj_mode_cls_loss": z,
            "dbg_query_traj_static_gate_loss": z,
            "dbg_query_traj_reweight_enabled": z,
            "dbg_query_traj_reweight_applied": z,
            "dbg_query_traj_valid_count": z,
            "dbg_query_traj_weight_mean": z,
            "dbg_query_traj_moving_ratio": z,
            "dbg_query_traj_motion_mag_mean": z,
            "dbg_query_endpoint_valid_count": z,
            "dbg_query_traj_static_target_ratio": z,
            "dbg_query_traj_static_prob_mean": z,
            "dbg_query_traj_static_pred_ratio": z,
            "dbg_query_traj_moving_gt_ratio": z,
            "dbg_query_traj_best_mode_mean": z,
            "dbg_query_traj_routed_mode_mean": z,
            "dbg_query_traj_pred_builtin_mode_ratio": z,
            "dbg_query_traj_pred_learned_mode_ratio": z,
            "dbg_query_traj_route_builtin_mode_ratio": z,
            "dbg_query_traj_route_learned_mode_ratio": z,
            "dbg_query_traj_mode_count": z.new_tensor(float(getattr(self, "query_traj_num_modes", 0))),
            "dbg_query_traj_builtin_mode_count": z.new_tensor(
                float(int(getattr(self, "query_traj_use_stationary_mode", False)) + int(getattr(self, "query_traj_use_cv_mode", False)))
            ),
            "dbg_query_traj_learned_mode_count": z.new_tensor(
                float(
                    max(
                        0,
                        int(getattr(self, "query_traj_num_modes", 0))
                        - int(getattr(self, "query_traj_use_stationary_mode", False))
                        - int(getattr(self, "query_traj_use_cv_mode", False)),
                    )
                )
            ),
            "dbg_query_traj_bernstein_degree": z.new_tensor(
                float(getattr(self, "query_traj_bernstein_degree", 0) if getattr(self, "query_traj_decoder_type", "offset") == "bernstein" else 0.0)
            ),
            "dbg_query_traj_summary_mean_speed": z,
            "dbg_query_traj_summary_endpoint_displacement": z,
            "dbg_query_traj_summary_path_length": z,
            "dbg_query_traj_summary_straightness_ratio": z,
            "dbg_query_traj_refine_xy_valid_count": z,
            "dbg_query_traj_refine_xy_weight_mean": z,
            "dbg_query_traj_refine_xy_motion_mag_mean": z,
            "dbg_query_traj_refine_gate_mean": z,
            "dbg_query_traj_refine_corr_abs_mean": z,
            "dbg_query_traj_sched_stage_begin_epoch": z,
            "dbg_query_traj_sched_loss_weight": z,
            "dbg_query_traj_sched_refine_loss_weight": z,
            "dbg_query_traj_sched_mode_cls_loss_weight": z,
            "dbg_query_traj_sched_teacher_forcing_enabled": z,
            "dbg_query_traj_sched_teacher_forcing_gt_ratio": z,
        }
        mode_count = int(getattr(self, "query_traj_num_modes", 0))
        stationary_enabled = bool(getattr(self, "query_traj_use_stationary_mode", False))
        cv_enabled = bool(getattr(self, "query_traj_use_cv_mode", False))
        learned_start_idx = int(stationary_enabled) + int(cv_enabled)
        for mode_idx in range(mode_count):
            traj_dbg_defaults[f"dbg_query_traj_pred_mode{mode_idx}_ratio"] = z
            traj_dbg_defaults[f"dbg_query_traj_route_mode{mode_idx}_ratio"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_is_builtin"] = z.new_tensor(
                float(1.0 if mode_idx < learned_start_idx else 0.0)
            )
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_is_stationary"] = z.new_tensor(
                float(1.0 if (stationary_enabled and mode_idx == 0) else 0.0)
            )
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_is_cv"] = z.new_tensor(
                float(1.0 if (cv_enabled and mode_idx == int(stationary_enabled)) else 0.0)
            )
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_is_learned"] = z.new_tensor(
                float(1.0 if mode_idx >= learned_start_idx else 0.0)
            )
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_bernstein_degree"] = z.new_tensor(
                float(
                    getattr(self, "query_traj_bernstein_degree", 0)
                    if (
                        getattr(self, "query_traj_decoder_type", "offset") == "bernstein"
                        and mode_idx >= learned_start_idx
                    )
                    else 0.0
                )
            )
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_mean_speed"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_endpoint_displacement"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_path_length"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_straightness_ratio"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_ctrl_abs_mean"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_d1_ctrl_abs_mean"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_d2_ctrl_abs_mean"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_score_mean"] = z
            traj_dbg_defaults[f"dbg_query_traj_mode{mode_idx}_score_abs_mean"] = z
        losses.update(traj_dbg_defaults)
        if isinstance(query_traj_loss, dict):
            losses.update(query_traj_loss)
        elif query_traj_loss is not None:
            losses["loss_query_traj"] = query_traj_loss
        else:
            losses["loss_query_traj"] = z
        if isinstance(query_traj_refine_loss, dict):
            losses.update(query_traj_refine_loss)
        elif query_traj_refine_loss is not None:
            losses["loss_query_traj_refine_xy"] = query_traj_refine_loss
        else:
            losses["loss_query_traj_refine_xy"] = z

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
        ]:
            del losses[k]
        self._namespace_dbg_logs(losses)
        return losses
