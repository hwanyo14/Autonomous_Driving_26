import os

import torch
import torch.nn.functional as F


class EfficientOCFQueryProjectionMixin:
    """Query-camera projection and attention-lift helper methods for EfficientOCF."""

    def _build_camera_debug_bundle_from_inputs(
        self,
        img_inputs_seq,
        frame_start_idx=0,
        frame_end_idx=None,
    ):
        if (not isinstance(img_inputs_seq, (list, tuple))) or len(img_inputs_seq) < 6:
            return None

        imgs_seq, rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq = img_inputs_seq[:6]
        if not all(torch.is_tensor(v) for v in (imgs_seq, rots_seq, trans_seq, intrins_seq, post_rots_seq, post_trans_seq)):
            return None

        seq_total = int(imgs_seq.shape[1])
        s_idx = max(0, int(frame_start_idx))
        e_idx = seq_total if frame_end_idx is None else min(seq_total, int(frame_end_idx))

        return {
            "imgs_seq_bt": imgs_seq[:, s_idx:e_idx, ...].contiguous().detach(),
            "rots_seq_bt": rots_seq[:, s_idx:e_idx, ...].contiguous().detach(),
            "trans_seq_bt": trans_seq[:, s_idx:e_idx, ...].contiguous().detach(),
            "intrins_seq_bt": intrins_seq[:, s_idx:e_idx, ...].contiguous().detach(),
            "post_rots_seq_bt": post_rots_seq[:, s_idx:e_idx, ...].contiguous().detach(),
            "post_trans_seq_bt": post_trans_seq[:, s_idx:e_idx, ...].contiguous().detach(),
            "frame_start_idx": int(s_idx),
            "frame_end_idx_exclusive": int(e_idx),
        }

    @staticmethod
    def _normalize_attention_prob_spatial_tqnhw(
        attn_tqnhw: torch.Tensor,
        tau: float = 1.0,
    ):
        if (not torch.is_tensor(attn_tqnhw)) or attn_tqnhw.dim() != 5:
            return None
        t_hist, q_count, n_cam, feat_h, feat_w = [int(v) for v in attn_tqnhw.shape]
        if t_hist <= 0 or q_count <= 0 or n_cam <= 0 or feat_h <= 0 or feat_w <= 0:
            return None

        prob_tqnp = attn_tqnhw.to(torch.float32).clamp_min(0.0).reshape(t_hist, q_count, n_cam, -1)
        tau_v = max(1e-6, float(tau))
        if abs(tau_v - 1.0) > 1e-6:
            # MultiheadAttention returns probabilities, not logits. Temperature
            # sharpening therefore uses p ** (1 / tau), equivalent to applying
            # temperature to the unknown pre-softmax logits up to a constant.
            prob_tqnp = prob_tqnp.clamp_min(1e-12).pow(1.0 / tau_v)
        denom = prob_tqnp.sum(dim=-1, keepdim=True)
        uniform = prob_tqnp.new_full(
            prob_tqnp.shape,
            1.0 / float(max(1, feat_h * feat_w)),
        )
        prob_tqnp = torch.where(denom > 0.0, prob_tqnp / denom.clamp_min(1e-12), uniform)
        return prob_tqnp.reshape(t_hist, q_count, n_cam, feat_h, feat_w)

    @staticmethod
    def _soft_argmax_attention_center_tqn2(
        attn_tqnhw: torch.Tensor,
        tau: float = 1.0,
    ):
        if (not torch.is_tensor(attn_tqnhw)) or attn_tqnhw.dim() != 5:
            return None, None
        t_hist, q_count, n_cam, feat_h, feat_w = [int(v) for v in attn_tqnhw.shape]
        if t_hist <= 0 or q_count <= 0 or n_cam <= 0 or feat_h <= 0 or feat_w <= 0:
            return None, None

        prob_tqnhw = EfficientOCFQueryProjectionMixin._normalize_attention_prob_spatial_tqnhw(
            attn_tqnhw,
            tau=tau,
        )
        if not torch.is_tensor(prob_tqnhw):
            return None, None
        prob_tqnp = prob_tqnhw.reshape(t_hist, q_count, n_cam, -1)
        gy, gx = torch.meshgrid(
            torch.arange(feat_h, device=attn_tqnhw.device, dtype=torch.float32),
            torch.arange(feat_w, device=attn_tqnhw.device, dtype=torch.float32),
            indexing="ij",
        )
        x_p = gx.reshape(1, 1, 1, -1)
        y_p = gy.reshape(1, 1, 1, -1)
        center_u_tqn = (prob_tqnp * x_p).sum(dim=-1)
        center_v_tqn = (prob_tqnp * y_p).sum(dim=-1)
        center_tqn2 = torch.stack([center_u_tqn, center_v_tqn], dim=-1)
        return prob_tqnhw, center_tqn2

    @staticmethod
    def _select_top1_camera_by_mass(
        cam_mass_tqn: torch.Tensor,
        valid_mask_tqn: torch.Tensor,
    ):
        if (
            (not torch.is_tensor(cam_mass_tqn))
            or (not torch.is_tensor(valid_mask_tqn))
            or cam_mass_tqn.dim() != 3
            or valid_mask_tqn.dim() != 3
            or tuple(cam_mass_tqn.shape) != tuple(valid_mask_tqn.shape)
        ):
            return None
        valid_mask_tqn = valid_mask_tqn.to(torch.bool)
        cam_mass_tqn = cam_mass_tqn.to(torch.float32)
        candidate_count_tq = valid_mask_tqn.to(torch.long).sum(dim=-1)
        selected_valid_tq = candidate_count_tq > 0
        masked_mass_tqn = cam_mass_tqn.masked_fill(~valid_mask_tqn, float("-inf"))
        selected_cam_idx_tq = torch.argmax(masked_mass_tqn, dim=-1)
        selected_cam_idx_tq = torch.where(
            selected_valid_tq,
            selected_cam_idx_tq,
            torch.zeros_like(selected_cam_idx_tq),
        )
        return {
            "selected_cam_idx_tq": selected_cam_idx_tq,
            "selected_valid_tq": selected_valid_tq,
            "selected_count": selected_valid_tq.to(torch.float32).sum(),
            "multi_cam_candidate_count": (candidate_count_tq > 1).to(torch.float32).sum(),
        }

    def _build_query_attn_soft_lift_pack(
        self,
        query_attn_weights_tqnhw: torch.Tensor,
        query_depth_probs_tqd: torch.Tensor,
        query_match_inputs: dict,
        future_egomotion: torch.Tensor,
    ):
        if (
            (not torch.is_tensor(query_attn_weights_tqnhw))
            or query_attn_weights_tqnhw.dim() != 5
            or (not torch.is_tensor(query_depth_probs_tqd))
            or query_depth_probs_tqd.dim() != 3
            or (not isinstance(query_match_inputs, dict))
            or (not torch.is_tensor(future_egomotion))
        ):
            return None

        rots_tn33 = query_match_inputs.get("rots_tn33", None)
        trans_tn3 = query_match_inputs.get("trans_tn3", None)
        intrins_tn33 = query_match_inputs.get("intrins_tn33", None)
        post_rots_tn33 = query_match_inputs.get("post_rots_tn33", None)
        post_trans_tn3 = query_match_inputs.get("post_trans_tn3", None)
        img_h = int(query_match_inputs.get("img_h", 0))
        img_w = int(query_match_inputs.get("img_w", 0))
        frame_start = int(query_match_inputs.get("frame_start_idx", 0))
        frame_end = int(query_match_inputs.get("frame_end_idx_exclusive", 0))
        if (
            (not torch.is_tensor(rots_tn33))
            or (not torch.is_tensor(trans_tn3))
            or (not torch.is_tensor(intrins_tn33))
            or (not torch.is_tensor(post_rots_tn33))
            or (not torch.is_tensor(post_trans_tn3))
            or img_h <= 1
            or img_w <= 1
        ):
            return None

        t_attn, q_count, n_cam_attn, feat_h, feat_w = [int(v) for v in query_attn_weights_tqnhw.shape]
        if t_attn <= 0 or q_count <= 0 or n_cam_attn <= 0:
            return None

        t_depth, q_depth, depth_bins = [int(v) for v in query_depth_probs_tqd.shape]
        if t_depth <= 0 or q_depth <= 0 or depth_bins <= 0:
            return None
        if q_depth != q_count:
            return None

        n_cam = int(
            min(
                n_cam_attn,
                int(rots_tn33.shape[1]),
                int(trans_tn3.shape[1]),
                int(intrins_tn33.shape[1]),
                int(post_rots_tn33.shape[1]),
                int(post_trans_tn3.shape[1]),
            )
        )
        t_cam = int(
            min(
                t_attn,
                int(rots_tn33.shape[0]),
                int(trans_tn3.shape[0]),
                int(intrins_tn33.shape[0]),
                int(post_rots_tn33.shape[0]),
                int(post_trans_tn3.shape[0]),
            )
        )
        if t_cam <= 0 or n_cam <= 0:
            return None

        if self.query_inst_depth_range_mode == "custom":
            depth_min = float(self.query_inst_depth_min)
            depth_max = float(self.query_inst_depth_max)
        else:
            dbound = None
            if hasattr(self, "img_view_transformer") and hasattr(self.img_view_transformer, "grid_config"):
                dbound = self.img_view_transformer.grid_config.get("dbound", None)
            if (not isinstance(dbound, (list, tuple))) or len(dbound) < 2:
                return None
            depth_min = float(dbound[0])
            depth_max = float(dbound[1])
        if not (depth_max > depth_min):
            return None

        present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
        # Three time-mapping regimes for output_frame_indices:
        #   - history:  t_depth == time_receptive_field  -> [0, 1, ..., T_past-1]
        #               (each past frame gets its own lifted center; this is the
        #                new path enabled when the QueryHead projection is removed)
        #   - present:  t_depth == 1                     -> [present_global_idx]
        #               (legacy / inference compatibility path)
        #   - future :  otherwise (query_present_only=False) ->
        #               trailing window of n_future_frames_plus future frames
        if (
            bool(getattr(self, "query_present_only", False))
            and t_depth == int(self.time_receptive_field)
        ):
            output_frame_indices = torch.arange(
                0,
                int(self.time_receptive_field),
                device=query_attn_weights_tqnhw.device,
                dtype=torch.long,
            )
        elif bool(getattr(self, "query_present_only", False)) and t_depth == 1:
            output_frame_indices = torch.full(
                (t_depth,),
                fill_value=present_global_idx,
                device=query_attn_weights_tqnhw.device,
                dtype=torch.long,
            )
        else:
            output_start_global = int(self.time_receptive_field + self.n_future_frames - t_depth)
            output_frame_indices = torch.arange(
                output_start_global,
                output_start_global + t_depth,
                device=query_attn_weights_tqnhw.device,
                dtype=torch.long,
            )
        attn_src_idx_t = (output_frame_indices - int(frame_start)).clamp(
            min=0,
            max=max(0, t_cam - 1),
        )

        attn_tqnhw = query_attn_weights_tqnhw.index_select(0, attn_src_idx_t)[:, :, :n_cam].to(torch.float32)
        prob_tqnhw, center_feat_tqn2 = self._soft_argmax_attention_center_tqn2(
            attn_tqnhw,
            tau=float(self.query_attn_softargmax_tau),
        )
        if (not torch.is_tensor(prob_tqnhw)) or (not torch.is_tensor(center_feat_tqn2)):
            return None

        query_depth_probs = query_depth_probs_tqd[:t_depth].to(torch.float32).clamp_min(0.0)
        query_depth_probs = query_depth_probs / query_depth_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        bin_size = (depth_max - depth_min) / float(depth_bins)
        depth_vals_d = (
            torch.arange(depth_bins, device=query_depth_probs.device, dtype=torch.float32) + 0.5
        ) * float(bin_size) + float(depth_min)
        depth_expect_tq = (query_depth_probs * depth_vals_d.view(1, 1, depth_bins)).sum(dim=-1)
        depth_expect_tqn = depth_expect_tq.unsqueeze(-1).expand(-1, -1, n_cam)

        scale_u = float(img_w - 1) / float(max(1, feat_w - 1))
        scale_v = float(img_h - 1) / float(max(1, feat_h - 1))
        center_img_tqn2 = center_feat_tqn2.clone()
        center_img_tqn2[..., 0] = center_img_tqn2[..., 0] * scale_u
        center_img_tqn2[..., 1] = center_img_tqn2[..., 1] * scale_v

        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[0]
        ego = ego.to(device=attn_tqnhw.device, dtype=torch.float32)
        frame_indices_hist = list(range(frame_start, max(frame_start, frame_end)))
        if len(frame_indices_hist) < t_cam:
            frame_indices_hist = list(range(int(self.time_receptive_field - t_cam), int(self.time_receptive_field)))
        frame_indices_hist = frame_indices_hist[:t_cam]
        frame_indices = [int(frame_indices_hist[int(i)]) for i in attn_src_idx_t.detach().cpu().tolist()]
        lidar_t_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=frame_indices,
            present_global_idx=present_global_idx,
        )
        if len(lidar_t_to_present) != t_depth:
            return None

        lifted_center_tqn3 = torch.zeros((t_depth, q_count, n_cam, 3), device=attn_tqnhw.device, dtype=torch.float32)
        lifted_valid_tqn = torch.zeros((t_depth, q_count, n_cam), device=attn_tqnhw.device, dtype=torch.bool)
        for t_idx in range(t_depth):
            src_t = int(attn_src_idx_t[t_idx].item())
            tf_lidar_to_present = lidar_t_to_present[t_idx].to(device=attn_tqnhw.device, dtype=torch.float32)
            for cam_idx in range(n_cam):
                uv_aug_q2 = center_img_tqn2[t_idx, :, cam_idx]
                depth_q = depth_expect_tqn[t_idx, :, cam_idx]

                post_rot_22 = post_rots_tn33[src_t, cam_idx].to(torch.float32)[:2, :2]
                post_trans_2 = post_trans_tn3[src_t, cam_idx].to(torch.float32)[:2]
                post_rot_22_inv = torch.inverse(post_rot_22)
                uv_raw_q2 = (post_rot_22_inv @ (uv_aug_q2 - post_trans_2).t()).t()

                valid_q = (
                    torch.isfinite(uv_raw_q2).all(dim=-1)
                    & torch.isfinite(depth_q)
                    & (depth_q > 1e-5)
                )
                if not bool(valid_q.any().item()):
                    continue
                q_idx = torch.nonzero(valid_q, as_tuple=False).squeeze(1)
                uv_raw = uv_raw_q2.index_select(0, q_idx)
                d = depth_q.index_select(0, q_idx).unsqueeze(-1)
                uv1 = torch.cat(
                    [
                        uv_raw,
                        torch.ones((int(uv_raw.shape[0]), 1), device=uv_raw.device, dtype=torch.float32),
                    ],
                    dim=-1,
                )
                intrin_inv = torch.inverse(intrins_tn33[src_t, cam_idx].to(torch.float32))
                cam_ray = (intrin_inv @ uv1.t()).t()
                cam_xyz = cam_ray * d

                rot_33 = rots_tn33[src_t, cam_idx].to(torch.float32)
                trans_3 = trans_tn3[src_t, cam_idx].to(torch.float32)
                lidar_t = (rot_33 @ cam_xyz.t()).t() + trans_3.view(1, 3)
                lidar_h = torch.cat(
                    [lidar_t, torch.ones((int(lidar_t.shape[0]), 1), device=lidar_t.device, dtype=torch.float32)],
                    dim=1,
                )
                present_xyz = (tf_lidar_to_present @ lidar_h.t()).t()[:, :3]
                lifted_center_tqn3[t_idx, q_idx, cam_idx] = present_xyz
                lifted_valid_tqn[t_idx, q_idx, cam_idx] = torch.isfinite(present_xyz).all(dim=-1)

        cam_mass_tqn = attn_tqnhw.reshape(t_depth, q_count, n_cam, -1).sum(dim=-1).clamp_min(0.0)
        selected_cam_pack = self._select_top1_camera_by_mass(
            cam_mass_tqn=cam_mass_tqn,
            valid_mask_tqn=lifted_valid_tqn,
        )
        if not isinstance(selected_cam_pack, dict):
            return None
        selected_cam_idx_tq = selected_cam_pack["selected_cam_idx_tq"]
        selected_valid_tq = selected_cam_pack["selected_valid_tq"]
        gather_idx_tq13 = selected_cam_idx_tq.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        lifted_center_tq3 = lifted_center_tqn3.gather(2, gather_idx_tq13).squeeze(2)
        lifted_valid_tq = selected_valid_tq.clone()
        finite_lift_tq = torch.isfinite(lifted_center_tq3).all(dim=-1)
        lifted_valid_tq = lifted_valid_tq & finite_lift_tq
        lifted_center_tq3 = torch.where(
            finite_lift_tq.unsqueeze(-1), lifted_center_tq3, torch.zeros_like(lifted_center_tq3)
        )

        # ---- soft-argmax vs hard-peak 괴리 계측 (opt-in: EOCF_QFEAT_ATTN_PEAK=1, 기본 OFF) ----
        # 가설: attention map이 단봉이면 soft-argmax와 최대峰이 일치하고, 이봉/퍼짐이면
        # soft-argmax가 봉우리 '사이'(=아무것도 없는 자리)에 떨어진다. 이 괴리가 크면 중심이 틀렸다는 신호.
        # depth 엔트로피(1D)와 달리 **2D 공간 다봉성**이라 기존 인자와 겹치지 않는다.
        # 3D 거리는 두 점이 같은 depth를 쓰므로 회전이 노름을 보존해 해석적으로 계산된다:
        #   ||Δ3D|| = depth * || K^-1 · [Δu_raw, Δv_raw, 0] || = depth * sqrt((Δu/fx)^2 + (Δv/fy)^2)
        attn_peak_pack = {}
        if os.environ.get("EOCF_QFEAT_ATTN_PEAK", "0") == "1":
            try:
                prob_tqnp = prob_tqnhw.reshape(t_depth, q_count, n_cam, -1)
                peak_flat_tqn = prob_tqnp.argmax(dim=-1)
                peak_prob_tqn = prob_tqnp.gather(-1, peak_flat_tqn.unsqueeze(-1)).squeeze(-1)
                peak_feat_tqn2 = torch.stack(
                    [(peak_flat_tqn % feat_w).to(torch.float32),
                     torch.div(peak_flat_tqn, feat_w, rounding_mode="floor").to(torch.float32)],
                    dim=-1,
                )
                d2d_feat_tqn = (center_feat_tqn2 - peak_feat_tqn2).norm(dim=-1)
                peak_img_tqn2 = peak_feat_tqn2.clone()
                peak_img_tqn2[..., 0] = peak_img_tqn2[..., 0] * scale_u
                peak_img_tqn2[..., 1] = peak_img_tqn2[..., 1] * scale_v
                d_uv_aug_tqn2 = center_img_tqn2 - peak_img_tqn2
                d2d_img_tqn = d_uv_aug_tqn2.norm(dim=-1)
                # 이미지 augmentation(post_rot) 역변환 후 intrinsic으로 3D 환산
                pr_tn22 = post_rots_tn33.index_select(0, attn_src_idx_t)[:, :n_cam, :2, :2].to(torch.float32)
                pr_inv_tn22 = torch.inverse(pr_tn22)
                d_uv_raw_tqn2 = torch.einsum("tnij,tqnj->tqni", pr_inv_tn22, d_uv_aug_tqn2)
                k_tn = intrins_tn33.index_select(0, attn_src_idx_t)[:, :n_cam].to(torch.float32)
                fx_tn = k_tn[..., 0, 0].abs().clamp_min(1e-6).unsqueeze(1)
                fy_tn = k_tn[..., 1, 1].abs().clamp_min(1e-6).unsqueeze(1)
                d3d_tqn = torch.sqrt(
                    (d_uv_raw_tqn2[..., 0] / fx_tn) ** 2 + (d_uv_raw_tqn2[..., 1] / fy_tn) ** 2
                ) * depth_expect_tqn
                _g = selected_cam_idx_tq.unsqueeze(-1)
                attn_peak_pack = {
                    "attn_soft_hard_d2d_feat_tq": d2d_feat_tqn.gather(2, _g).squeeze(-1),
                    "attn_soft_hard_d2d_img_tq": d2d_img_tqn.gather(2, _g).squeeze(-1),
                    "attn_soft_hard_d3d_tq": d3d_tqn.gather(2, _g).squeeze(-1),
                    "attn_peak_prob_tq": peak_prob_tqn.gather(2, _g).squeeze(-1),
                    "attn_feat_hw": center_feat_tqn2.new_tensor([float(feat_h), float(feat_w)]),
                }
            except Exception as e:  # 계측 실패가 추론을 죽이면 안 된다
                print(f"[attn_peak] skipped (err={e})", flush=True)
                attn_peak_pack = {}

        depth_expect_tq = depth_expect_tqn.gather(2, selected_cam_idx_tq.unsqueeze(-1)).squeeze(-1)
        depth_valid_tq = selected_valid_tq & torch.isfinite(depth_expect_tq) & (depth_expect_tq > 0.0)
        depth_mean = (
            depth_expect_tq[depth_valid_tq].mean()
            if bool(depth_valid_tq.any().item())
            else depth_expect_tq.new_tensor(0.0)
        )
        return {
            "center_feat_tqn2": center_feat_tqn2,
            "center_img_tqn2": center_img_tqn2,
            "depth_expect_tqn": depth_expect_tqn,
            "depth_expect_tq": depth_expect_tq,
            "lifted_center_world_tqn3": lifted_center_tqn3,
            "lifted_center_world_tq3": lifted_center_tq3,
            "lifted_valid_tqn": lifted_valid_tqn,
            "lifted_valid_tq": lifted_valid_tq,
            "selected_cam_idx_tq": selected_cam_idx_tq,
            "selected_cam_valid_tq": selected_valid_tq,
            "attn_src_idx_t": attn_src_idx_t,
            "output_frame_indices_t": output_frame_indices,
            "query_output_global_frame_indices": output_frame_indices,
            "query_present_global_idx": output_frame_indices.new_tensor(
                [present_global_idx], dtype=torch.long
            ),
            "dbg_query_attn_soft_lift_valid_cam_count": lifted_valid_tqn.to(torch.float32).sum(),
            "dbg_query_attn_soft_lift_valid_query_count": lifted_valid_tq.to(torch.float32).sum(),
            "dbg_query_attn_soft_lift_selected_cam_count": selected_cam_pack["selected_count"],
            "dbg_query_attn_soft_lift_multi_cam_candidate_count": selected_cam_pack["multi_cam_candidate_count"],
            "dbg_query_attn_soft_lift_depth_mean": depth_mean,
            "dbg_query_attn_softargmax_tau": center_feat_tqn2.new_tensor(float(self.query_attn_softargmax_tau)),
            **attn_peak_pack,
        }

    def _compute_query_depth_loss_from_match(
        self,
        query_depth_logits_tqd: torch.Tensor,
        query_attn_weights_tqnhw: torch.Tensor,
        inst_match_result: dict,
        query_inst_depth_target_pack: dict,
        loss_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> dict:
        if torch.is_tensor(query_depth_logits_tqd):
            z = query_depth_logits_tqd.sum() * 0.0
        else:
            z = self.mean_weight.sum() * 0.0
        out = {
            "loss_query_depth": z,
            "dbg_query_depth_valid_count": z,
            "dbg_query_depth_matched_pair_count": z,
            "dbg_query_depth_valid_pair_count": z,
            "dbg_query_depth_selected_count": z,
            "dbg_query_depth_multi_cam_candidate_count": z,
            "dbg_query_depth_loss_weight": z.new_tensor(float(loss_weight)),
            "dbg_query_depth_label_smoothing": z.new_tensor(float(label_smoothing)),
            # depth-head performance diagnostics (prefill kept in sync w/ record below
            # for multi-GPU log_vars; computed on matched+valid pairs at the CE site)
            "dbg_query_depth_top1_acc": z,
            "dbg_query_depth_within1_acc": z,
            "dbg_query_depth_bin_abs_err": z,
            "dbg_query_depth_soft_bin_abs_err": z,
            "dbg_query_depth_entropy": z,
        }
        if (
            (not torch.is_tensor(query_depth_logits_tqd))
            or query_depth_logits_tqd.dim() != 3
            or (not isinstance(inst_match_result, dict))
            or (not isinstance(query_inst_depth_target_pack, dict))
        ):
            return out

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        match_gt_ids_n = inst_match_result.get("gt_ids_n", None)
        depth_bin_tcn = query_inst_depth_target_pack.get("gt_inst_depth_bin_tcn", None)
        depth_valid_tcn = query_inst_depth_target_pack.get("gt_inst_depth_valid_tcn", None)
        depth_target_ids_n = query_inst_depth_target_pack.get("gt_inst_ids_n", None)
        depth_target_global_idx_t = query_inst_depth_target_pack.get(
            "query_output_global_frame_indices", None
        )
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or (not torch.is_tensor(match_gt_ids_n))
            or (not torch.is_tensor(depth_bin_tcn))
            or (not torch.is_tensor(depth_valid_tcn))
            or (not torch.is_tensor(depth_target_ids_n))
            or (not torch.is_tensor(depth_target_global_idx_t))
            or depth_bin_tcn.dim() != 3
            or depth_valid_tcn.dim() != 3
            or tuple(depth_bin_tcn.shape) != tuple(depth_valid_tcn.shape)
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
        ):
            return out

        t_query, q_total, d_bins = [int(v) for v in query_depth_logits_tqd.shape]
        t_tgt, n_cam, n_tgt_inst = [int(v) for v in depth_bin_tcn.shape]
        if d_bins <= 0 or t_query <= 0 or q_total <= 0 or t_tgt <= 0 or n_cam <= 0 or n_tgt_inst <= 0:
            return out

        mq = matched_query_idx.to(device=query_depth_logits_tqd.device, dtype=torch.long).reshape(-1)
        mi = matched_inst_idx.to(device=query_depth_logits_tqd.device, dtype=torch.long).reshape(-1)
        match_gt_ids = match_gt_ids_n.to(device=query_depth_logits_tqd.device, dtype=torch.long).reshape(-1)
        keep = (mq >= 0) & (mq < q_total) & (mi >= 0) & (mi < int(match_gt_ids.numel()))
        if not bool(keep.any().item()):
            return out
        mq = mq[keep]
        mi = mi[keep]
        if mq.numel() <= 0:
            return out

        matched_gt_ids_k = match_gt_ids.index_select(0, mi)
        tgt_ids = depth_target_ids_n.to(device=query_depth_logits_tqd.device, dtype=torch.long).reshape(-1)
        id_to_tgt_col = {int(tgt_ids[i].item()): int(i) for i in range(int(tgt_ids.numel()))}
        keep_k = []
        tgt_cols = []
        for k in range(int(matched_gt_ids_k.numel())):
            col = id_to_tgt_col.get(int(matched_gt_ids_k[k].item()), None)
            if col is None:
                continue
            keep_k.append(k)
            tgt_cols.append(col)
        if len(keep_k) <= 0:
            return out
        keep_k_t = torch.as_tensor(keep_k, device=query_depth_logits_tqd.device, dtype=torch.long)
        tgt_cols_t = torch.as_tensor(tgt_cols, device=query_depth_logits_tqd.device, dtype=torch.long)
        mq = mq.index_select(0, keep_k_t)
        if mq.numel() <= 0:
            return out

        depth_target_global_idx_t = depth_target_global_idx_t.to(
            device=query_depth_logits_tqd.device, dtype=torch.long
        ).reshape(-1)
        if int(depth_target_global_idx_t.numel()) != t_tgt:
            return out

        present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
        # Mirror the regime in `_build_query_attn_soft_lift_pack`:
        # history-mapping when t_query == time_receptive_field (3 past frames),
        # present-only when t_query == 1, else future-pointing window.
        if (
            bool(getattr(self, "query_present_only", False))
            and t_query == int(self.time_receptive_field)
        ):
            query_global_idx_t = torch.arange(
                0,
                int(self.time_receptive_field),
                device=query_depth_logits_tqd.device,
                dtype=torch.long,
            )
        elif bool(getattr(self, "query_present_only", False)) and t_query == 1:
            query_global_idx_t = torch.full(
                (t_query,),
                fill_value=present_global_idx,
                device=query_depth_logits_tqd.device,
                dtype=torch.long,
            )
        else:
            query_global_start = int(self.time_receptive_field + self.n_future_frames - t_query)
            query_global_idx_t = torch.arange(
                query_global_start,
                query_global_start + t_query,
                device=query_depth_logits_tqd.device,
                dtype=torch.long,
            )

        tgt_global_to_row = {
            int(depth_target_global_idx_t[row].item()): int(row)
            for row in range(t_tgt)
        }
        query_rows = []
        tgt_rows = []
        for q_row in range(t_query):
            tgt_row = tgt_global_to_row.get(int(query_global_idx_t[q_row].item()), None)
            if tgt_row is None:
                continue
            query_rows.append(int(q_row))
            tgt_rows.append(int(tgt_row))
        if len(query_rows) <= 0:
            return out

        query_row_idx_t = torch.as_tensor(
            query_rows, device=query_depth_logits_tqd.device, dtype=torch.long
        )
        tgt_row_idx_t = torch.as_tensor(
            tgt_rows, device=query_depth_logits_tqd.device, dtype=torch.long
        )
        t = int(query_row_idx_t.numel())

        logits_tkd = query_depth_logits_tqd.index_select(0, query_row_idx_t).index_select(1, mq).to(torch.float32)
        tgt_bin_tck = depth_bin_tcn.to(
            device=query_depth_logits_tqd.device, dtype=torch.long
        ).index_select(0, tgt_row_idx_t).index_select(2, tgt_cols_t)
        tgt_valid_tck = depth_valid_tcn.to(
            device=query_depth_logits_tqd.device, dtype=torch.bool
        ).index_select(0, tgt_row_idx_t).index_select(2, tgt_cols_t)
        tgt_valid_tck = tgt_valid_tck & (tgt_bin_tck >= 0) & (tgt_bin_tck < d_bins)

        valid_all_count = tgt_valid_tck.to(torch.float32).sum()
        out["dbg_query_depth_valid_count"] = valid_all_count
        out["dbg_query_depth_matched_pair_count"] = z.new_tensor(float(matched_query_idx.numel()))
        out["dbg_query_depth_valid_pair_count"] = z.new_tensor(float(mq.numel()))
        if not bool((valid_all_count > 0).item()):
            return out

        if (
            torch.is_tensor(query_attn_weights_tqnhw)
            and query_attn_weights_tqnhw.dim() == 5
            and int(query_attn_weights_tqnhw.shape[0]) >= int(query_row_idx_t.max().item()) + 1
            and int(query_attn_weights_tqnhw.shape[1]) == q_total
            and int(query_attn_weights_tqnhw.shape[2]) >= n_cam
        ):
            cam_mass_tkc = (
                query_attn_weights_tqnhw.index_select(0, query_row_idx_t)
                .to(device=query_depth_logits_tqd.device, dtype=torch.float32)
                .index_select(1, mq)[:, :, :n_cam]
                .reshape(t, int(mq.numel()), n_cam, -1)
                .sum(dim=-1)
                .clamp_min(0.0)
            )
        else:
            cam_mass_tkc = torch.ones(
                (t, int(mq.numel()), n_cam),
                device=query_depth_logits_tqd.device,
                dtype=torch.float32,
            )

        tgt_valid_tkc = tgt_valid_tck.permute(0, 2, 1).contiguous()
        tgt_bin_tkc = tgt_bin_tck.permute(0, 2, 1).contiguous()
        selected_cam_pack = self._select_top1_camera_by_mass(
            cam_mass_tqn=cam_mass_tkc,
            valid_mask_tqn=tgt_valid_tkc,
        )
        if not isinstance(selected_cam_pack, dict):
            return out
        selected_cam_idx_tk = selected_cam_pack["selected_cam_idx_tq"]
        selected_valid_tk = selected_cam_pack["selected_valid_tq"]
        out["dbg_query_depth_selected_count"] = selected_cam_pack["selected_count"]
        out["dbg_query_depth_multi_cam_candidate_count"] = selected_cam_pack["multi_cam_candidate_count"]

        selected_tgt_bin_tk = tgt_bin_tkc.gather(2, selected_cam_idx_tk.unsqueeze(-1)).squeeze(-1)
        logits_flat = logits_tkd.reshape(-1, d_bins)
        tgt_bin_flat = selected_tgt_bin_tk.reshape(-1)
        valid_flat = selected_valid_tk.reshape(-1)
        valid_count = valid_flat.to(torch.float32).sum()
        if not bool((valid_count > 0).item()):
            return out

        ce = F.cross_entropy(
            logits_flat[valid_flat],
            tgt_bin_flat[valid_flat],
            reduction="mean",
            label_smoothing=float(max(0.0, min(0.999, label_smoothing))),
        )
        out["loss_query_depth"] = ce * float(max(0.0, loss_weight))

        # depth-head performance diagnostics (does the head predict the GT depth bin
        # accurately/sharply?). argmax = hard pick; soft = expectation the lift uses.
        with torch.no_grad():
            v_logits = logits_flat[valid_flat]            # [M, d_bins]
            v_tgt = tgt_bin_flat[valid_flat]              # [M]
            v_probs = torch.softmax(v_logits, dim=-1)
            pred_bin = v_logits.argmax(dim=-1)
            bin_idx = torch.arange(d_bins, device=v_logits.device, dtype=torch.float32)
            exp_bin = (v_probs * bin_idx[None, :]).sum(dim=-1)  # soft-argmax bin
            tgt_f = v_tgt.to(torch.float32)
            out["dbg_query_depth_top1_acc"] = (pred_bin == v_tgt).to(torch.float32).mean()
            out["dbg_query_depth_within1_acc"] = ((pred_bin - v_tgt).abs() <= 1).to(torch.float32).mean()
            out["dbg_query_depth_bin_abs_err"] = (pred_bin.to(torch.float32) - tgt_f).abs().mean()
            out["dbg_query_depth_soft_bin_abs_err"] = (exp_bin - tgt_f).abs().mean()
            out["dbg_query_depth_entropy"] = (
                -(v_probs * torch.log(v_probs.clamp_min(1e-9))).sum(dim=-1)
            ).mean()
        return out

    def _select_query_tokens_for_query_head(self, query_inst: torch.Tensor) -> torch.Tensor:
        """
        Normalize transformer query output to [T, Q, D] before feeding QueryHead.

        Current TransformerModule returns [T, 1, Q, D] (singleton axis at dim=1),
        but this helper guards against shape drift.
        """

        if query_inst.dim() == 4:
            # Expected current layout: [T, 1, Q, D]
            q = query_inst[:, 0]
        elif query_inst.dim() == 3:
            # Backward-compatible fallback if transformer is changed to [T,Q,D].
            q = query_inst

        return q.contiguous()
