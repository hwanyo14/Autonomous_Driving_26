import torch
import torch.nn.functional as F
import numpy as np


class EfficientOCFBEVPoolMixin:
    """BEV feature pooling methods for EfficientOCF."""

    def _slice_bev_feats_by_time(
        self,
        bev_feats,
        total_time_length: int,
        start_idx: int,
        end_idx: int,
    ):
        if bev_feats is None:
            return None
        if not isinstance(bev_feats, (list, tuple)) or len(bev_feats) <= 0:
            raise ValueError("bev_feats must be a non-empty list/tuple.")
        total_t = int(total_time_length)
        s = max(0, int(start_idx))
        e = min(total_t, int(end_idx))
        if e <= s:
            return None
        sliced = []
        slice_t = int(e - s)
        for stage_feats in bev_feats:
            if stage_feats.dim() != 4:
                raise ValueError(
                    f"Each stage must be [B,T*C,H,W], got {tuple(stage_feats.shape)}"
                )
            if (int(stage_feats.shape[1]) % total_t) != 0:
                raise ValueError(
                    f"Stage channel dim {int(stage_feats.shape[1])} is not divisible by total_time_length={total_t}"
                )
            c_per_t = int(stage_feats.shape[1] // total_t)
            feat_btchw = stage_feats.view(
                int(stage_feats.shape[0]),
                total_t,
                c_per_t,
                int(stage_feats.shape[-2]),
                int(stage_feats.shape[-1]),
            )
            feat_slice = feat_btchw[:, s:e].reshape(
                int(stage_feats.shape[0]),
                slice_t * c_per_t,
                int(stage_feats.shape[-2]),
                int(stage_feats.shape[-1]),
            )
            sliced.append(feat_slice.contiguous())
        return sliced

    def _build_gt_instance_bev_sparse_from_occ3d(
        self,
        gt_instance_occ3d_txyz_pred: torch.Tensor,
        gt_instance_ids_n: torch.Tensor = None,
        ignore_index: int = 255,
    ):
        """
        Convert 3D instance occupancy [T, X, Y, Z] into sparse BEV instance pixels.

        Returns:
            sparse_t: list length T, each item=(inst_col_m, x_m, y_m)
            valid_tn: [T, N] bool
            ids_n: [N] long (sorted)
            bev_shape_xy: (X, Y)
        """
        if gt_instance_occ3d_txyz_pred is None:
            return None, None, None, None
        if gt_instance_occ3d_txyz_pred.dim() != 4:
            raise ValueError(
                "gt_instance_occ3d_txyz_pred must be [T,X,Y,Z], got "
                f"{tuple(gt_instance_occ3d_txyz_pred.shape)}"
            )

        occ = gt_instance_occ3d_txyz_pred.to(torch.long)
        T, X, Y, Z = occ.shape
        device = occ.device

        if gt_instance_ids_n is None:
            ids = torch.unique(occ)
            keep = (ids != 0) & (ids != int(ignore_index))
            ids = ids[keep].to(torch.long)
            if ids.numel() > 0:
                ids = torch.sort(ids).values
        else:
            ids = gt_instance_ids_n.to(device=device, dtype=torch.long).contiguous()
            if ids.numel() > 0:
                ids = torch.unique(ids, sorted=True)

        N = int(ids.numel())
        sparse_t = []
        valid_tn = torch.zeros((T, N), dtype=torch.bool, device=device)
        if N <= 0:
            for _ in range(T):
                empty = torch.empty((0,), dtype=torch.long, device=device)
                sparse_t.append((empty, empty, empty))
            return sparse_t, valid_tn, ids, (X, Y)

        xy_area = int(X * Y)
        for t in range(T):
            occ_t = occ[t]  # [X, Y, Z]
            flat = occ_t.reshape(-1)
            kept = (flat != 0) & (flat != int(ignore_index))
            if not torch.any(kept):
                empty = torch.empty((0,), dtype=torch.long, device=device)
                sparse_t.append((empty, empty, empty))
                continue

            flat_idx = torch.nonzero(kept, as_tuple=False).squeeze(1)
            inst_vals = flat.index_select(0, flat_idx)  # [M]

            pos = torch.searchsorted(ids, inst_vals)
            in_range = pos < N
            if not torch.any(in_range):
                empty = torch.empty((0,), dtype=torch.long, device=device)
                sparse_t.append((empty, empty, empty))
                continue
            pos = pos[in_range]
            flat_idx = flat_idx[in_range]
            inst_vals = inst_vals[in_range]

            exact = ids.index_select(0, pos) == inst_vals
            if not torch.any(exact):
                empty = torch.empty((0,), dtype=torch.long, device=device)
                sparse_t.append((empty, empty, empty))
                continue
            pos = pos[exact]
            flat_idx = flat_idx[exact]

            xy_flat = torch.div(flat_idx, Z, rounding_mode='floor')
            # Deduplicate by (instance_col, x, y) across z.
            pair_code = pos * xy_area + xy_flat
            pair_code = torch.unique(pair_code, sorted=True)
            if pair_code.numel() <= 0:
                empty = torch.empty((0,), dtype=torch.long, device=device)
                sparse_t.append((empty, empty, empty))
                continue

            pos_u = torch.div(pair_code, xy_area, rounding_mode='floor')
            xy_flat_u = torch.remainder(pair_code, xy_area)
            x_idx = torch.div(xy_flat_u, Y, rounding_mode='floor')
            y_idx = torch.remainder(xy_flat_u, Y)

            valid_tn[t, torch.unique(pos_u)] = True
            sparse_t.append((pos_u, x_idx, y_idx))

        return sparse_t, valid_tn, ids, (X, Y)

    def _pool_gt_instance_features_from_bev_feats_sparse(
        self,
        bev_feats_enc,
        gt_instance_bev_sparse_t,
        gt_instance_bev_valid_tn: torch.Tensor,
        gt_instance_bev_shape_xy,
        bev_time_length: int = None,
        detach_bev_feats: bool = True,
        eps: float = 1e-6,
    ):
        """
        Pool per-instance BEV features from multiscale BEV features using sparse BEV occupancy pixels.

        Args:
            bev_feats_enc: list of stage features, each [B, T*C, H, W]
            gt_instance_bev_sparse_t: list length T, item=(inst_col_m, x_m, y_m)
            gt_instance_bev_valid_tn: [T, N] bool
            gt_instance_bev_shape_xy: (X, Y)
            bev_time_length: optional temporal length used to split [T*C].
                When None, uses gt_instance_bev_valid_tn.shape[0].
        Returns:
            gt_feat_tnd: [T, N, D]
            gt_valid_tn: [T, N]
        """
        if bev_feats_enc is None or gt_instance_bev_sparse_t is None:
            return None, None
        if not isinstance(bev_feats_enc, (list, tuple)) or len(bev_feats_enc) <= 0:
            raise ValueError("bev_feats_enc must be a non-empty list/tuple.")
        if not isinstance(gt_instance_bev_sparse_t, (list, tuple)):
            raise ValueError(
                f"gt_instance_bev_sparse_t must be list/tuple, got {type(gt_instance_bev_sparse_t)}"
            )
        if not torch.is_tensor(gt_instance_bev_valid_tn) or gt_instance_bev_valid_tn.dim() != 2:
            raise ValueError("gt_instance_bev_valid_tn must be [T,N] tensor.")
        if (not isinstance(gt_instance_bev_shape_xy, (tuple, list))) or len(gt_instance_bev_shape_xy) != 2:
            raise ValueError("gt_instance_bev_shape_xy must be (X, Y).")

        t_gt, N = gt_instance_bev_valid_tn.shape
        if bev_time_length is None:
            T = int(t_gt)
        else:
            T = int(bev_time_length)
            if T <= 0:
                raise ValueError(f"bev_time_length must be positive, got {bev_time_length}")
            if T > int(t_gt):
                raise ValueError(
                    f"bev_time_length({T}) exceeds gt time length({int(t_gt)})."
                )

        if len(gt_instance_bev_sparse_t) < T:
            raise ValueError(
                f"gt_instance_bev_sparse_t length mismatch for T={T}: len={len(gt_instance_bev_sparse_t)}"
            )
        # Keep time alignment explicit. This is required for source BEV path:
        # source features are built from receptive-field history (e.g., T=3),
        # while prediction GT tensors are often [n_future_frames_plus] (e.g., T=6).
        gt_instance_bev_sparse_t = gt_instance_bev_sparse_t[:T]
        gt_instance_bev_valid_tn = gt_instance_bev_valid_tn[:T]

        X, Y = int(gt_instance_bev_shape_xy[0]), int(gt_instance_bev_shape_xy[1])
        device = gt_instance_bev_valid_tn.device
        if N <= 0:
            out = torch.zeros(
                (T, 0, int(self.query_head.embed_dim)),
                dtype=torch.float32,
                device=device,
            )
            valid = torch.zeros((T, 0), dtype=torch.bool, device=device)
            return out, valid

        stage_mean_chunks = []
        stage_max_chunks = []
        xy_area_stage = None  # local scratch only; assigned per stage

        for stage_feats in bev_feats_enc:
            feats = stage_feats.detach() if detach_bev_feats else stage_feats
            if feats.dim() != 4:
                raise ValueError(
                    f"Each bev_feats_enc stage must be [B,T*C,H,W], got {tuple(feats.shape)}"
                )
            if feats.shape[0] != 1:
                raise ValueError(
                    f"Current GT feature pooling path expects batch=1, got {tuple(feats.shape)}"
                )
            if (feats.shape[1] % T) != 0:
                raise ValueError(
                    f"Stage channel dim {int(feats.shape[1])} is not divisible by T={T}"
                )

            Hs, Ws = int(feats.shape[-2]), int(feats.shape[-1])
            c_per_frame = int(feats.shape[1] // T)
            feat_tchw = feats.view(1, T, c_per_frame, Hs, Ws)[0]
            xy_area_stage = int(Hs * Ws)

            mean_tnc = torch.zeros((T, N, c_per_frame), dtype=feat_tchw.dtype, device=feat_tchw.device)
            max_tnc = torch.zeros((T, N, c_per_frame), dtype=feat_tchw.dtype, device=feat_tchw.device)

            for t in range(T):
                feat_chw = feat_tchw[t]
                feat_chm = feat_chw.reshape(c_per_frame, -1)  # [C, HW]
                inst_col_t, x_t, y_t = gt_instance_bev_sparse_t[t]
                if inst_col_t.numel() <= 0:
                    continue

                # Build stage-resolution masks directly (avoid [T,N,512,512] dense masks + pooling).
                xs_t = torch.div(x_t * Hs, X, rounding_mode='floor').clamp_(0, Hs - 1)
                ys_t = torch.div(y_t * Ws, Y, rounding_mode='floor').clamp_(0, Ws - 1)
                stage_lin_t = xs_t * Ws + ys_t
                pair_lin_t = inst_col_t * xy_area_stage + stage_lin_t
                pair_lin_t = torch.unique(pair_lin_t, sorted=False)
                inst_u = torch.div(pair_lin_t, xy_area_stage, rounding_mode='floor')
                lin_u = torch.remainder(pair_lin_t, xy_area_stage)

                mask_nm = torch.zeros((N, xy_area_stage), dtype=torch.bool, device=feat_chm.device)
                mask_nm[inst_u, lin_u] = True
                count_n = mask_nm.sum(dim=-1)                 # [N]
                valid_n = count_n > 0
                if not torch.any(valid_n):
                    continue

                # Mean pooling via masked matmul.
                mask_f = mask_nm.to(dtype=feat_chm.dtype)
                sum_nc = torch.matmul(mask_f, feat_chm.transpose(0, 1))  # [N, C]
                denom_nc = count_n.to(dtype=feat_chm.dtype).unsqueeze(-1).clamp_min(float(eps))
                mean_tnc[t] = sum_nc / denom_nc

                # Max pooling with per-instance masked reduction (N is typically small).
                valid_idx = torch.nonzero(valid_n, as_tuple=False).squeeze(1)
                for n_idx in valid_idx.tolist():
                    vals = feat_chm[:, mask_nm[n_idx]]
                    if vals.numel() > 0:
                        max_tnc[t, n_idx] = vals.max(dim=1).values

            stage_mean_chunks.append(mean_tnc)
            stage_max_chunks.append(max_tnc)

        gt_feat_tnd = self._merge_pooled_stage_features(
            mean_chunks=stage_mean_chunks,
            max_chunks=stage_max_chunks,
            context="gt",
        )

        return gt_feat_tnd, gt_instance_bev_valid_tn.to(device=gt_feat_tnd.device, dtype=torch.bool)

    def _pool_query_features_from_bev_feats_gaussian(
        self,
        bev_feats_enc,
        centers_world_tq3: torch.Tensor,
        sigmas_world_tq3: torch.Tensor,
        detach_bev_feats: bool = True,
        eps: float = 1e-6,
    ):
        """
        Pool per-query BEV features at predicted center/gaussian.

        Returns:
            query_bev_feat_tqd: [T, Q, D]
            query_bev_valid_tq: [T, Q] bool
        """
        if bev_feats_enc is None:
            return None, None
        if not torch.is_tensor(centers_world_tq3) or not torch.is_tensor(sigmas_world_tq3):
            return None, None
        if centers_world_tq3.dim() != 3 or sigmas_world_tq3.shape != centers_world_tq3.shape:
            raise ValueError(
                "centers/sigmas must be [T,Q,3] with same shape, got "
                f"{tuple(centers_world_tq3.shape)} vs {tuple(sigmas_world_tq3.shape)}"
            )
        if not isinstance(bev_feats_enc, (list, tuple)) or len(bev_feats_enc) <= 0:
            raise ValueError("bev_feats_enc must be a non-empty list/tuple.")
        if self.point_cloud_range is None:
            raise ValueError("point_cloud_range must be set for query BEV pooling.")

        T, Q, _ = centers_world_tq3.shape
        device = centers_world_tq3.device
        if Q <= 0:
            out = torch.zeros(
                (T, 0, int(self.query_head.embed_dim)),
                dtype=torch.float32,
                device=device,
            )
            valid = torch.zeros((T, 0), dtype=torch.bool, device=device)
            return out, valid

        x_min, y_min, _, x_max, y_max, _ = [float(v) for v in self.point_cloud_range]
        range_x = max(1e-6, x_max - x_min)
        range_y = max(1e-6, y_max - y_min)

        trunc = float(getattr(self.voxelizer, "gaussian_truncate_sigma", 3.0))
        trunc = max(1.0, trunc)
        sigma_floor_vox = float(getattr(self.voxelizer, "gaussian_sigma_floor_vox", 0.35))
        sigma_floor_vox = max(1e-3, sigma_floor_vox)

        centers = centers_world_tq3.to(torch.float32)
        sigmas = sigmas_world_tq3.to(torch.float32)
        valid_tq = torch.isfinite(centers).all(dim=-1) & torch.isfinite(sigmas).all(dim=-1)

        # Fixed dimensionless Gaussian template for vectorized sampling.
        k_rad = max(1, int(np.ceil(trunc)))
        k_size = int(2 * k_rad + 1)
        offs = torch.linspace(
            -float(k_rad),
            float(k_rad),
            steps=k_size,
            device=device,
            dtype=torch.float32,
        )
        ox_kk, oy_kk = torch.meshgrid(offs, offs, indexing="ij")  # [K,K]
        w_kk = torch.exp(-0.5 * (ox_kk.pow(2) + oy_kk.pow(2)))    # [K,K]
        w_kk = (w_kk / w_kk.sum().clamp_min(float(eps))).to(torch.float32)

        stage_mean_chunks = []
        stage_max_chunks = []
        for stage_feats in bev_feats_enc:
            feats = stage_feats.detach() if detach_bev_feats else stage_feats
            if feats.dim() != 4:
                raise ValueError(
                    f"Each bev_feats_enc stage must be [B,T*C,H,W], got {tuple(feats.shape)}"
                )
            if feats.shape[0] != 1:
                raise ValueError(
                    f"Current query BEV pooling expects batch=1, got {tuple(feats.shape)}"
                )
            if (feats.shape[1] % T) != 0:
                raise ValueError(
                    f"Stage channel dim {int(feats.shape[1])} is not divisible by T={T}"
                )

            Hs, Ws = int(feats.shape[-2]), int(feats.shape[-1])  # Hs maps x-axis, Ws maps y-axis
            c_per_frame = int(feats.shape[1] // T)
            feat_tchw = feats.view(1, T, c_per_frame, Hs, Ws)[0]

            # world(m) -> stage voxel index (same center convention as gaussian voxelizer).
            vs_x = range_x / float(Hs)
            vs_y = range_y / float(Ws)
            cx_tq = (centers[..., 0] - x_min) / vs_x - 0.5
            cy_tq = (centers[..., 1] - y_min) / vs_y - 0.5
            sx_tq = (sigmas[..., 0] / vs_x).clamp(min=sigma_floor_vox)
            sy_tq = (sigmas[..., 1] / vs_y).clamp(min=sigma_floor_vox)

            mean_tqc = torch.zeros((T, Q, c_per_frame), dtype=feat_tchw.dtype, device=feat_tchw.device)
            max_tqc = torch.zeros((T, Q, c_per_frame), dtype=feat_tchw.dtype, device=feat_tchw.device)
            valid_stage_tq = valid_tq.clone()

            for t in range(T):
                valid_q_idx = torch.nonzero(valid_stage_tq[t], as_tuple=False).squeeze(1)
                if valid_q_idx.numel() <= 0:
                    continue

                feat_1chw = feat_tchw[t].unsqueeze(0)  # [1,C,Hs,Ws]
                cx_n = cx_tq[t].index_select(0, valid_q_idx)  # [N]
                cy_n = cy_tq[t].index_select(0, valid_q_idx)  # [N]
                sx_n = sx_tq[t].index_select(0, valid_q_idx)  # [N]
                sy_n = sy_tq[t].index_select(0, valid_q_idx)  # [N]
                n_valid = int(valid_q_idx.numel())

                # Dimensionless template -> query-specific stage coordinates.
                sample_x_nkk = cx_n[:, None, None] + (sx_n[:, None, None] * ox_kk[None, :, :])  # row (Hs)
                sample_y_nkk = cy_n[:, None, None] + (sy_n[:, None, None] * oy_kk[None, :, :])  # col (Ws)

                inb_nkk = (
                    (sample_x_nkk >= 0.0) & (sample_x_nkk <= float(Hs - 1))
                    & (sample_y_nkk >= 0.0) & (sample_y_nkk <= float(Ws - 1))
                )
                w_nkk = w_kk[None, :, :] * inb_nkk.to(torch.float32)
                wsum_n = w_nkk.sum(dim=(1, 2))  # [N]
                valid_sample_n = wsum_n > float(eps)

                # Normalize for grid_sample (x uses Ws/col, y uses Hs/row).
                if Ws > 1:
                    gx_nkk = (sample_y_nkk / float(Ws - 1)) * 2.0 - 1.0
                else:
                    gx_nkk = torch.zeros_like(sample_y_nkk)
                if Hs > 1:
                    gy_nkk = (sample_x_nkk / float(Hs - 1)) * 2.0 - 1.0
                else:
                    gy_nkk = torch.zeros_like(sample_x_nkk)
                grid_nkk2 = torch.stack([gx_nkk, gy_nkk], dim=-1).to(dtype=feat_1chw.dtype)  # [N,K,K,2]
                # One grid_sample per t-stage (vectorized over queries).
                grid_1hwo2 = grid_nkk2.reshape(1, n_valid * k_size, k_size, 2)
                sampled_1cnk = F.grid_sample(
                    feat_1chw,
                    grid_1hwo2,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                )  # [1,C,N*K,K]
                sampled_nckk = sampled_1cnk.view(1, c_per_frame, n_valid, k_size, k_size)[0].permute(1, 0, 2, 3)

                w_norm_n11 = (wsum_n.clamp_min(float(eps))).view(n_valid, 1, 1, 1).to(dtype=sampled_nckk.dtype)
                w_n11kk = w_nkk[:, None, :, :].to(dtype=sampled_nckk.dtype)
                mean_nc = (sampled_nckk * (w_n11kk / w_norm_n11)).sum(dim=(2, 3))

                # Max over valid sampled support.
                neg_inf = torch.finfo(sampled_nckk.dtype).min
                masked_nckk = sampled_nckk.masked_fill(~inb_nkk[:, None, :, :], neg_inf)
                max_nc = masked_nckk.amax(dim=(2, 3))
                max_nc = torch.where(valid_sample_n[:, None], max_nc, torch.zeros_like(max_nc))

                valid_q_keep = valid_q_idx[valid_sample_n]
                if valid_q_keep.numel() > 0:
                    mean_tqc[t, valid_q_keep] = mean_nc[valid_sample_n]
                    max_tqc[t, valid_q_keep] = max_nc[valid_sample_n]
                invalid_q_keep = valid_q_idx[~valid_sample_n]
                if invalid_q_keep.numel() > 0:
                    valid_stage_tq[t, invalid_q_keep] = False

            stage_mean_chunks.append(mean_tqc)
            stage_max_chunks.append(max_tqc)
            valid_tq = valid_tq & valid_stage_tq

        query_bev_feat_tqd = self._merge_pooled_stage_features(
            mean_chunks=stage_mean_chunks,
            max_chunks=stage_max_chunks,
            context="query",
        )
        return query_bev_feat_tqd, valid_tq.to(device=device, dtype=torch.bool)

    def _build_fixed_query_pool_sigmas(
        self,
        centers_world_tq3: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(centers_world_tq3):
            return None
        if centers_world_tq3.dim() != 3:
            raise ValueError(
                f"centers_world_tq3 must be [T,Q,3], got {tuple(centers_world_tq3.shape)}"
            )
        sigma = centers_world_tq3.new_tensor(
            self.query_bev_pool_fixed_sigma_xyz,
            dtype=torch.float32,
        ).clamp(min=1e-3)
        return sigma.view(1, 1, 3).expand_as(centers_world_tq3).contiguous()

    def _merge_pooled_stage_features(
        self,
        mean_chunks,
        max_chunks,
        context: str = "pool",
    ) -> torch.Tensor:
        """
        Merge multistage pooled features while keeping per-frame channel size.

        We intentionally avoid dimensional remapping here. Stage features are
        expected to be configured to the target channel size (query_embed_dim).
        """
        if (not isinstance(mean_chunks, (list, tuple))) or len(mean_chunks) <= 0:
            raise ValueError(f"{context}: mean_chunks must be non-empty.")
        if (not isinstance(max_chunks, (list, tuple))) or len(max_chunks) != len(mean_chunks):
            raise ValueError(
                f"{context}: max_chunks length mismatch. "
                f"len(mean)={len(mean_chunks)}, len(max)={len(max_chunks)}"
            )

        # Prefer mean pooled descriptors; average across stages to keep channel size.
        stage_dims = [int(t.shape[-1]) for t in mean_chunks]
        ref_dim = stage_dims[0]
        if any(d != ref_dim for d in stage_dims):
            raise ValueError(
                f"{context}: stage channel dims must match without remapping, got {stage_dims}"
            )

        if len(mean_chunks) == 1:
            return mean_chunks[0]
        return torch.stack(mean_chunks, dim=0).mean(dim=0)
