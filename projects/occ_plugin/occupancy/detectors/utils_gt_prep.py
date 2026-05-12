import torch
import numpy as np


class EfficientOCFGTPrepMixin:
    """Ground truth preparation methods for EfficientOCF."""

    def _select_query_present_frame_slice(
        self,
        tensor: torch.Tensor,
        source_layout: str,
    ):
        """
        Select one present frame for query-present-only mode.

        Layout contracts:
        - full7   : [t-2, t-1, t, t+1, t+2, t+3, t+4] -> local 2
        - plus6   : [t-1, t, t+1, t+2, t+3, t+4]      -> local 1
        - history3: [t-2, t-1, t]                     -> local 2
        """
        if (not torch.is_tensor(tensor)) or tensor.dim() <= 0:
            return tensor, None
        t = int(tensor.shape[0])
        if t <= 0:
            return tensor, None

        mode = str(source_layout).lower().strip()
        if mode == "plus6":
            local_idx = 1
        elif mode == "history3":
            local_idx = int(self.time_receptive_field - 1)
        elif mode == "full7":
            local_idx = int(self.time_receptive_field - 1)
        else:
            raise ValueError(f"Unsupported source_layout={source_layout}")
        local_idx = max(0, min(t - 1, int(local_idx)))
        return tensor.narrow(0, local_idx, 1).contiguous(), local_idx

    def _select_query_temporal_gt_slice(
        self,
        tensor: torch.Tensor,
    ):
        """
        Align a full-sequence dense GT tensor to the query prediction timeline.

        Dataset/full sequence layout is [t-2, t-1, t, t+1, t+2, t+3, t+4].
        The plus window layout is [t-1, t, t+1, t+2, t+3, t+4].
        In query_present_only mode, query local 0 must therefore supervise t.
        """
        if (not torch.is_tensor(tensor)) or tensor.dim() <= 0:
            return tensor, None, "none"
        t = int(tensor.shape[0])
        if t <= 0:
            return tensor, None, "empty"

        if bool(getattr(self, "query_present_only", False)):
            full_len = int(self.time_receptive_field + self.n_future_frames)
            if t >= full_len:
                local_idx = int(self.time_receptive_field - 1)
                source_layout = "full"
            elif t >= int(self.n_future_frames_plus):
                local_idx = int(getattr(self, "eval_start_moment", 1))
                source_layout = "plus"
            elif t >= int(self.time_receptive_field):
                local_idx = int(self.time_receptive_field - 1)
                source_layout = "history"
            else:
                local_idx = 0
                source_layout = "short"
            local_idx = max(0, min(t - 1, int(local_idx)))
            return tensor.narrow(0, local_idx, 1).contiguous(), local_idx, source_layout

        target_t = int(self.n_future_frames_plus)
        if t > target_t:
            start_idx = int(t - target_t)
            return tensor[start_idx:].contiguous(), start_idx, "tail"
        return tensor.contiguous(), 0, "as_is"

    def _select_query_visualization_gt_slice(
        self,
        tensor: torch.Tensor,
    ):
        """
        Align a dense GT tensor to the 7-frame BEV visualization horizon
        [t-(T_past-1), ..., t-1, t, t+1, ..., t+n_future_frames].

        - Source T >= time_receptive_field + n_future_frames : tail slice.
        - Source T == n_future_frames_plus (legacy plus window starting at t-1):
              prepend (time_receptive_field - eval_start_moment - 1) empty past
              frames so the present aligns at index time_receptive_field - 1.
        - Otherwise: zero-pad on the past side to reach the target length.

        The returned tensor matches the visualization grid order so that index
        time_receptive_field - 1 corresponds to the current frame.
        """
        target_t = int(self.time_receptive_field) + int(self.n_future_frames)
        if (not torch.is_tensor(tensor)) or tensor.dim() <= 0 or target_t <= 0:
            return tensor, None, "none"
        t = int(tensor.shape[0])
        if t <= 0:
            return tensor, None, "empty"

        if t >= target_t:
            start_idx = int(t - target_t)
            return tensor[start_idx:start_idx + target_t].contiguous(), start_idx, "tail_vis"

        # GT shorter than visualization horizon: pad past frames with zeros.
        pad_count = int(target_t - t)
        pad_shape = (pad_count,) + tuple(int(v) for v in tensor.shape[1:])
        pad_tensor = tensor.new_zeros(pad_shape)
        return torch.cat([pad_tensor, tensor], dim=0).contiguous(), 0, "past_pad_vis"

    def _select_query_trajectory_gt_slice(
        self,
        tensor: torch.Tensor,
    ):
        """
        Select trajectory-supervision horizon [t, t+1, ..., t+n_future_frames].

        This path is independent from query_present_only behavior and always
        returns tail (n_future_frames + 1) frames.
        """
        if (not torch.is_tensor(tensor)) or tensor.dim() <= 0:
            return tensor, None, "none"
        t = int(tensor.shape[0])
        if t <= 0:
            return tensor, None, "empty"

        target_t = int(self.n_future_frames) + 1
        if target_t <= 0:
            raise ValueError(f"trajectory target_t must be positive, got {target_t}")
        if t < target_t:
            raise ValueError(
                f"trajectory GT time dim too short: {t} < {target_t}"
            )
        if t > target_t:
            start_idx = int(t - target_t)
            return tensor[start_idx:].contiguous(), start_idx, "tail_traj"
        return tensor.contiguous(), 0, "as_is_traj"

    def _stack_gt_occ_to_txyz(self, gt_occ):
        """
        Returns:
            gt_occ_txyz: [T, X, Y, Z] long
        """
        if isinstance(gt_occ, (list, tuple)):
            gt_occ_t = torch.stack(gt_occ, dim=0)
        elif torch.is_tensor(gt_occ):
            gt_occ_t = gt_occ
        else:
            raise TypeError(f"gt_occ must be list/tuple/tensor, got {type(gt_occ)}")

        # Supported layouts for current query path (single sample):
        # - [T, 1, X, Y, Z]
        # - [1, T, 1, X, Y, Z]
        # - [T, X, Y, Z]
        if gt_occ_t.dim() == 6 and gt_occ_t.shape[0] == 1 and gt_occ_t.shape[2] == 1:
            gt_occ_t = gt_occ_t[0, :, 0]
        elif gt_occ_t.dim() == 5 and gt_occ_t.shape[1] == 1:
            gt_occ_t = gt_occ_t[:, 0]
        elif gt_occ_t.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported gt_occ shape for query input: {tuple(gt_occ_t.shape)}")

        if gt_occ_t.shape[0] < self.time_receptive_field:
            raise ValueError(
                f"gt_occ time dim too short for query input: {gt_occ_t.shape[0]} < {self.time_receptive_field}"
            )
        return gt_occ_t[:self.time_receptive_field].to(torch.long).contiguous()

    def _prepare_segmentation_query_gt(self, segmentation, ignore_index: int = 255):
        """
        Convert bbox-volume segmentation labels into query-loss supervision.

        Returns:
            [T, X, Y, Z] long, where:
            - occupied: 1
            - empty: 0
            - ignore: ignore_index
        """
        if segmentation is None:
            return None
        if isinstance(segmentation, (list, tuple)):
            seg_t = torch.stack(segmentation, dim=0)
        elif torch.is_tensor(segmentation):
            seg_t = segmentation
        else:
            raise TypeError(f"segmentation must be list/tuple/tensor, got {type(segmentation)}")

        # Supported layouts:
        # - [T, X, Y, Z]
        # - [T, 1, X, Y, Z]
        # - [1, T, X, Y, Z]   (current dataloader stack with samples_per_gpu=1)
        # - [1, T, 1, X, Y, Z]
        if seg_t.dim() == 6:
            if seg_t.shape[0] != 1:
                raise ValueError(
                    f"current query path expects batch=1 for segmentation, got {tuple(seg_t.shape)}"
                )
            if seg_t.shape[2] != 1:
                raise ValueError(
                    f"Unsupported segmentation shape (expect channel=1 at dim2): {tuple(seg_t.shape)}"
                )
            seg_t = seg_t[0, :, 0]  # [T, X, Y, Z]
        elif seg_t.dim() == 5:
            if seg_t.shape[0] == 1:
                seg_t = seg_t[0]      # [T, X, Y, Z]
            elif seg_t.shape[1] == 1:
                seg_t = seg_t[:, 0]   # [T, X, Y, Z]
            else:
                raise ValueError(
                    f"Unsupported segmentation shape for query GT: {tuple(seg_t.shape)}"
                )
        elif seg_t.dim() == 4:
            pass
        else:
            raise ValueError(
                f"Unsupported segmentation shape for query GT: {tuple(seg_t.shape)}"
            )

        seg_t = seg_t.to(torch.long).contiguous()
        ig = int(ignore_index)
        ignore_mask = (seg_t == ig)
        occ_mask = (seg_t > 0) & (~ignore_mask)

        out = torch.zeros_like(seg_t, dtype=torch.long)
        out[occ_mask] = 1
        out[ignore_mask] = ig
        return out

    def _select_query_gt_for_losses(self, gt_occ, segmentation):
        """
        Select supervision source for query losses.
        """
        if self.use_segmentation_as_query_gt and (segmentation is not None):
            # segmentation은 bbox volume 기반이라 non-zero를 occupied(1)로 이진화해서 사용.
            seg_gt = self._prepare_segmentation_query_gt(segmentation, ignore_index=255)
            return seg_gt, (1,)
        return gt_occ, self.gmo_ids

    def _prepare_gt_instance_centers_common(
        self,
        gt_instance_centers_world=None,
        gt_instance_centers_valid=None,
        gt_instance_ids=None,
        target_t=None,
        slice_from: str = "tail",
    ):
        """
        Normalize dataloader outputs to a target-ready layout.

        Args:
            target_t: required temporal length after slicing frames.
            slice_from: `tail` keeps the last `target_t` frames, `head` keeps
                the first `target_t` frames. This matters because the dataset
                sequence is `[history..., future...]`, while matching uses the
                forecast window and camera-depth supervision uses the visible
                history window.
        Returns:
            centers_world: [T_target, N_active, 3] float
            valid_mask:    [T_target, N_active] bool
            instance_ids:  [N_active] long (or None)
        """
        if (gt_instance_centers_world is None) or (gt_instance_centers_valid is None):
            return None, None, None

        c = gt_instance_centers_world
        v = gt_instance_centers_valid
        iids = gt_instance_ids

        if isinstance(c, (list, tuple)):
            if len(c) == 1:
                c = c[0]
            else:
                try:
                    c = torch.stack(c, dim=0)
                except Exception as e:
                    raise ValueError(
                        "gt_instance_centers_world list could not be stacked. "
                        "Use samples_per_gpu=1 for variable instance counts."
                    ) from e
        if isinstance(v, (list, tuple)):
            if len(v) == 1:
                v = v[0]
            else:
                try:
                    v = torch.stack(v, dim=0)
                except Exception as e:
                    raise ValueError(
                        "gt_instance_centers_valid list could not be stacked. "
                        "Use samples_per_gpu=1 for variable instance counts."
                    ) from e
        if isinstance(iids, (list, tuple)):
            if len(iids) == 0:
                iids = None
            elif len(iids) == 1:
                iids = iids[0]
            else:
                try:
                    iids = torch.stack(iids, dim=0)
                except Exception as e:
                    raise ValueError(
                        "gt_instance_ids list could not be stacked. "
                        "Use samples_per_gpu=1 for variable instance counts."
                    ) from e

        if c.dim() == 4 and c.shape[0] == 1:
            c = c[0]
        elif c.dim() != 3:
            raise ValueError(f"gt_instance_centers_world must be [1,T,N,3] or [T,N,3], got {tuple(c.shape)}")

        if v.dim() == 3 and v.shape[0] == 1:
            v = v[0]
        elif v.dim() != 2:
            raise ValueError(f"gt_instance_centers_valid must be [1,T,N] or [T,N], got {tuple(v.shape)}")

        if iids is not None:
            if iids.dim() == 2 and iids.shape[0] == 1:
                iids = iids[0]
            elif iids.dim() != 1:
                raise ValueError(f"gt_instance_ids must be [1,N] or [N], got {tuple(iids.shape)}")

        c = c.to(torch.float32).contiguous()
        v = v.to(torch.bool).contiguous()
        if iids is not None:
            iids = iids.to(torch.long).contiguous()

        t_target = int(target_t) if target_t is not None else int(c.shape[0])
        if t_target <= 0:
            raise ValueError(f"target_t must be positive, got {t_target}")
        if c.shape[0] < t_target:
            raise ValueError(f"gt instance center time dim too short: {c.shape[0]} < {t_target}")
        if c.shape[0] != t_target:
            slice_mode = str(slice_from).lower().strip()
            if slice_mode == "head":
                c = c[:t_target].contiguous()
                v = v[:t_target].contiguous()
            elif slice_mode == "tail":
                c = c[-t_target:].contiguous()
                v = v[-t_target:].contiguous()
            else:
                raise ValueError(f"Unsupported slice_from={slice_from}")

        active = v.any(dim=0)
        c = c[:, active, :].contiguous()
        v = v[:, active].contiguous()
        if iids is not None and iids.numel() == active.numel():
            iids = iids[active].contiguous()

        return c, v, iids

    def _prepare_gt_instance_centers_for_matching(
        self,
        gt_instance_centers_world=None,
        gt_instance_centers_valid=None,
        gt_instance_ids=None,
    ):
        """
        Normalize dataloader outputs to a matching-ready layout.
        Returns:
            centers_world: [T_pred, N_active, 3] float
            valid_mask:    [T_pred, N_active] bool
            instance_ids:  [N_active] long (or None)
        """
        centers, valid, ids = self._prepare_gt_instance_centers_common(
            gt_instance_centers_world=gt_instance_centers_world,
            gt_instance_centers_valid=gt_instance_centers_valid,
            gt_instance_ids=gt_instance_ids,
            target_t=int(self.n_future_frames_plus),
            slice_from="tail",
        )
        if bool(getattr(self, "query_present_only", False)):
            centers, _ = self._select_query_present_frame_slice(
                centers, source_layout="plus6"
            )
            valid, _ = self._select_query_present_frame_slice(
                valid, source_layout="plus6"
            )
        return centers, valid, ids

    def _prepare_gt_instance_centers_for_trajectory_matching(
        self,
        gt_instance_centers_world=None,
        gt_instance_centers_valid=None,
        gt_instance_ids=None,
    ):
        """
        Normalize dataloader outputs for trajectory matching/loss timeline.
        Returns:
            centers_world: [T_traj, N_active, 3] float, T_traj = n_future_frames + 1
            valid_mask:    [T_traj, N_active] bool
            instance_ids:  [N_active] long (or None)
        """
        centers, valid, ids = self._prepare_gt_instance_centers_common(
            gt_instance_centers_world=gt_instance_centers_world,
            gt_instance_centers_valid=gt_instance_centers_valid,
            gt_instance_ids=gt_instance_ids,
            target_t=(int(self.n_future_frames) + 1),
            slice_from="tail",
        )
        return centers, valid, ids

    def _prepare_gt_instance_centers_for_history(
        self,
        gt_instance_centers_world=None,
        gt_instance_centers_valid=None,
        gt_instance_ids=None,
    ):
        """
        Normalize dataloader outputs for history-frame camera supervision.
        Returns:
            centers_world: [T_hist, N_active, 3] float
            valid_mask:    [T_hist, N_active] bool
            instance_ids:  [N_active] long (or None)
        """
        centers, valid, ids = self._prepare_gt_instance_centers_common(
            gt_instance_centers_world=gt_instance_centers_world,
            gt_instance_centers_valid=gt_instance_centers_valid,
            gt_instance_ids=gt_instance_ids,
            target_t=int(self.time_receptive_field),
            slice_from="head",
        )
        return centers, valid, ids

    @staticmethod
    def _extract_valid_instance_ids_from_dense_txyz(
        segmentation_instance3d_txyz=None,
        background_index: int = 0,
        ignore_index: int = 255,
    ):
        if (
            (not torch.is_tensor(segmentation_instance3d_txyz))
            or segmentation_instance3d_txyz.dim() != 4
        ):
            return None
        ids = torch.unique(segmentation_instance3d_txyz.to(torch.long))
        keep = (ids != int(background_index)) & (ids != int(ignore_index))
        ids = torch.sort(ids[keep]).values.to(torch.long).contiguous()
        return ids

    def _build_intersection_instance_ids_from_dense_pair(
        self,
        primary_instance3d_txyz=None,
        secondary_instance3d_txyz=None,
        background_index: int = 0,
        ignore_index: int = 255,
    ):
        primary_ids = self._extract_valid_instance_ids_from_dense_txyz(
            segmentation_instance3d_txyz=primary_instance3d_txyz,
            background_index=background_index,
            ignore_index=ignore_index,
        )
        secondary_ids = self._extract_valid_instance_ids_from_dense_txyz(
            segmentation_instance3d_txyz=secondary_instance3d_txyz,
            background_index=background_index,
            ignore_index=ignore_index,
        )
        if (primary_ids is None) or (secondary_ids is None):
            return None
        if primary_ids.numel() <= 0 or secondary_ids.numel() <= 0:
            return primary_ids[:0]

        inter_np = np.intersect1d(
            primary_ids.detach().cpu().numpy(),
            secondary_ids.detach().cpu().numpy(),
            assume_unique=False,
        )
        if inter_np.size <= 0:
            return primary_ids[:0]
        return torch.from_numpy(inter_np.astype(np.int64, copy=False)).to(
            device=primary_ids.device,
            dtype=torch.long,
        ).contiguous()

    @staticmethod
    def _filter_instance_targets_by_intersection_ids(
        centers_world_tn3=None,
        centers_valid_tn=None,
        instance_ids_n=None,
        intersection_ids_n=None,
    ):
        if (not torch.is_tensor(instance_ids_n)) or (not torch.is_tensor(intersection_ids_n)):
            return centers_world_tn3, centers_valid_tn, instance_ids_n

        ids = instance_ids_n.to(torch.long).contiguous()
        inter_set = set(int(v) for v in intersection_ids_n.detach().cpu().tolist())
        keep_idx_list = [idx for idx, iid in enumerate(ids.detach().cpu().tolist()) if int(iid) in inter_set]
        keep_idx = torch.as_tensor(keep_idx_list, device=ids.device, dtype=torch.long)
        ids_out = ids.index_select(0, keep_idx) if keep_idx.numel() > 0 else ids[:0]

        centers_out = centers_world_tn3
        valid_out = centers_valid_tn
        if (
            torch.is_tensor(centers_world_tn3)
            and torch.is_tensor(centers_valid_tn)
            and centers_world_tn3.dim() == 3
            and centers_valid_tn.dim() == 2
            and tuple(centers_world_tn3.shape[:2]) == tuple(centers_valid_tn.shape)
            and int(centers_world_tn3.shape[1]) == int(ids.numel())
        ):
            if keep_idx.numel() > 0:
                centers_out = centers_world_tn3.index_select(1, keep_idx).contiguous()
                valid_out = centers_valid_tn.index_select(1, keep_idx).contiguous()
            else:
                centers_out = centers_world_tn3[:, :0, :].contiguous()
                valid_out = centers_valid_tn[:, :0].contiguous()

        return centers_out, valid_out, ids_out

    @staticmethod
    def _reindex_temporal_tensor_by_instance_ids(
        values_tn=None,
        instance_ids_n=None,
        target_ids_n=None,
    ):
        if (
            (not torch.is_tensor(values_tn))
            or values_tn.dim() < 2
            or (not torch.is_tensor(instance_ids_n))
            or instance_ids_n.dim() != 1
            or (not torch.is_tensor(target_ids_n))
            or target_ids_n.dim() != 1
            or int(values_tn.shape[1]) != int(instance_ids_n.numel())
        ):
            return None
        id_to_col = {
            int(instance_ids_n[idx].item()): int(idx)
            for idx in range(int(instance_ids_n.numel()))
        }
        gather = []
        for iid in target_ids_n.tolist():
            col = id_to_col.get(int(iid), None)
            if col is None:
                return None
            gather.append(col)
        if len(gather) != int(target_ids_n.numel()):
            return None
        gather_idx = torch.as_tensor(
            gather,
            device=values_tn.device,
            dtype=torch.long,
        )
        return values_tn.index_select(1, gather_idx).contiguous()

    @staticmethod
    def _reindex_vector_by_instance_ids(
        values_n=None,
        instance_ids_n=None,
        target_ids_n=None,
    ):
        if (
            (not torch.is_tensor(values_n))
            or values_n.dim() != 1
            or (not torch.is_tensor(instance_ids_n))
            or instance_ids_n.dim() != 1
            or (not torch.is_tensor(target_ids_n))
            or target_ids_n.dim() != 1
            or int(values_n.numel()) != int(instance_ids_n.numel())
        ):
            return None
        id_to_col = {
            int(instance_ids_n[idx].item()): int(idx)
            for idx in range(int(instance_ids_n.numel()))
        }
        gather = []
        for iid in target_ids_n.tolist():
            col = id_to_col.get(int(iid), None)
            if col is None:
                return None
            gather.append(col)
        if len(gather) != int(target_ids_n.numel()):
            return None
        gather_idx = torch.as_tensor(
            gather,
            device=values_n.device,
            dtype=torch.long,
        )
        return values_n.index_select(0, gather_idx).contiguous()

    def _build_full_query_instance_centers_from_history_and_traj(
        self,
        history_centers_tn3=None,
        history_valid_tn=None,
        history_ids_n=None,
        traj_centers_tn3=None,
        traj_valid_tn=None,
        traj_ids_n=None,
    ):
        if (
            (not torch.is_tensor(history_centers_tn3))
            or (not torch.is_tensor(history_valid_tn))
            or (not torch.is_tensor(history_ids_n))
            or (not torch.is_tensor(traj_centers_tn3))
            or (not torch.is_tensor(traj_valid_tn))
            or (not torch.is_tensor(traj_ids_n))
        ):
            return None, None, None
        if history_centers_tn3.dim() != 3 or history_valid_tn.dim() != 2:
            return None, None, None
        if traj_centers_tn3.dim() != 3 or traj_valid_tn.dim() != 2:
            return None, None, None
        if history_ids_n.dim() != 1 or traj_ids_n.dim() != 1:
            return None, None, None
        if tuple(history_centers_tn3.shape[:2]) != tuple(history_valid_tn.shape):
            return None, None, None
        if tuple(traj_centers_tn3.shape[:2]) != tuple(traj_valid_tn.shape):
            return None, None, None
        if int(history_centers_tn3.shape[1]) != int(history_ids_n.numel()):
            return None, None, None
        if int(traj_centers_tn3.shape[1]) != int(traj_ids_n.numel()):
            return None, None, None

        present_local_idx = int(self.time_receptive_field) - 1
        if int(history_centers_tn3.shape[0]) < (present_local_idx + 1):
            return None, None, None
        if int(traj_centers_tn3.shape[0]) <= 0:
            return None, None, None

        target_device = traj_centers_tn3.device
        history_centers_tn3 = history_centers_tn3.to(
            device=target_device,
        ).contiguous()
        history_valid_tn = history_valid_tn.to(
            device=target_device,
            dtype=torch.bool,
        ).contiguous()
        history_ids = history_ids_n.to(
            device=target_device,
            dtype=torch.long,
        ).contiguous()
        traj_centers_tn3 = traj_centers_tn3.to(
            device=target_device,
        ).contiguous()
        traj_valid_tn = traj_valid_tn.to(
            device=target_device,
            dtype=torch.bool,
        ).contiguous()
        traj_ids = traj_ids_n.to(
            device=target_device,
            dtype=torch.long,
        ).contiguous()
        shared_np = np.intersect1d(
            history_ids.detach().cpu().numpy(),
            traj_ids.detach().cpu().numpy(),
            assume_unique=False,
        )
        if shared_np.size <= 0:
            return None, None, None
        shared_ids_n = torch.from_numpy(shared_np.astype(np.int64, copy=False)).to(
            device=traj_ids.device,
            dtype=torch.long,
        ).contiguous()

        history_centers_sel = self._reindex_temporal_tensor_by_instance_ids(
            values_tn=history_centers_tn3,
            instance_ids_n=history_ids,
            target_ids_n=shared_ids_n,
        )
        history_valid_sel = self._reindex_temporal_tensor_by_instance_ids(
            values_tn=history_valid_tn,
            instance_ids_n=history_ids,
            target_ids_n=shared_ids_n,
        )
        traj_centers_sel = self._reindex_temporal_tensor_by_instance_ids(
            values_tn=traj_centers_tn3,
            instance_ids_n=traj_ids,
            target_ids_n=shared_ids_n,
        )
        traj_valid_sel = self._reindex_temporal_tensor_by_instance_ids(
            values_tn=traj_valid_tn,
            instance_ids_n=traj_ids,
            target_ids_n=shared_ids_n,
        )
        if (
            history_centers_sel is None
            or history_valid_sel is None
            or traj_centers_sel is None
            or traj_valid_sel is None
        ):
            return None, None, None

        past_only = history_centers_sel[:present_local_idx].contiguous()
        past_valid = history_valid_sel[:present_local_idx].contiguous()
        full_centers = torch.cat([past_only, traj_centers_sel], dim=0).contiguous()
        full_valid = torch.cat([past_valid, traj_valid_sel], dim=0).contiguous()
        active = full_valid.any(dim=0)
        if not bool(active.any().item()):
            return full_centers[:, :0, :].contiguous(), full_valid[:, :0].contiguous(), shared_ids_n[:0]
        full_centers = full_centers[:, active, :].contiguous()
        full_valid = full_valid[:, active].contiguous()
        shared_ids_n = shared_ids_n[active].contiguous()
        return full_centers, full_valid, shared_ids_n

    def _prepare_segmentation_instance3d(self, segmentation_instance3d=None):
        """
        Normalize dataloader outputs to [T, X, Y, Z] long.

        Expected inputs (current pipeline usually uses batch=1):
        - [T, X, Y, Z]
        - [1, T, X, Y, Z]
        - [T, 1, X, Y, Z]
        - [1, T, 1, X, Y, Z]
        """
        if segmentation_instance3d is None:
            return None

        seg = segmentation_instance3d
        if isinstance(seg, (list, tuple)):
            if len(seg) == 0:
                return None
            if len(seg) == 1:
                seg = seg[0]
            else:
                try:
                    seg = torch.stack(seg, dim=0)
                except Exception as e:
                    raise ValueError(
                        "segmentation_instance3d list could not be stacked. "
                        "Use samples_per_gpu=1 for variable shapes."
                    ) from e

        if not torch.is_tensor(seg):
            raise TypeError(
                f"segmentation_instance3d must be tensor/list/tuple, got {type(seg)}"
            )

        if seg.dim() == 6:
            if seg.shape[0] != 1:
                raise ValueError(
                    f"current path expects batch=1 for segmentation_instance3d, got {tuple(seg.shape)}"
                )
            if seg.shape[2] != 1:
                raise ValueError(
                    f"Unsupported segmentation_instance3d shape (expect channel=1 at dim2): {tuple(seg.shape)}"
                )
            seg = seg[0, :, 0]  # [T, X, Y, Z]
        elif seg.dim() == 5:
            if seg.shape[0] == 1:
                seg = seg[0]      # [T, X, Y, Z]
            elif seg.shape[1] == 1:
                seg = seg[:, 0]   # [T, X, Y, Z]
            else:
                raise ValueError(
                    f"Unsupported segmentation_instance3d shape: {tuple(seg.shape)}"
                )
        elif seg.dim() == 4:
            pass
        else:
            raise ValueError(
                f"Unsupported segmentation_instance3d shape: {tuple(seg.shape)}"
            )

        return seg.to(torch.long).contiguous()

    @staticmethod
    def _extract_gt_occ_inst_sparse_list(gt_occ_inst=None):
        """
        Normalize dataloader `gt_occ_inst` to list[T] of int64 arrays [K,5]:
        [x, y, z, cls, inst]
        """
        if gt_occ_inst is None:
            return None
        x = getattr(gt_occ_inst, "data", gt_occ_inst)
        if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
            x = x[0]
        if not isinstance(x, (list, tuple)):
            return None

        out = []
        for rows_raw in x:
            rows = getattr(rows_raw, "data", rows_raw)
            if torch.is_tensor(rows):
                rows = rows.detach().cpu().numpy()
            rows = np.asarray(rows)
            if isinstance(rows, np.ndarray) and rows.dtype == object:
                rows = np.vstack(rows) if rows.size > 0 else np.zeros((0, 5), dtype=np.int64)
            rows = np.asarray(rows, dtype=np.int64)
            if rows.size <= 0:
                rows = np.zeros((0, 5), dtype=np.int64)
            if rows.ndim != 2 or rows.shape[1] < 5:
                return None
            out.append(rows[:, :5].copy())
        return out

    @staticmethod
    def _infer_dense_shape_from_sparse(sparse_list):
        if not isinstance(sparse_list, (list, tuple)) or len(sparse_list) <= 0:
            return None
        x_max = -1
        y_max = -1
        z_max = -1
        for rows in sparse_list:
            if rows is None:
                continue
            arr = np.asarray(rows)
            if arr.size <= 0:
                continue
            if arr.ndim != 2 or arr.shape[1] < 3:
                return None
            x_max = max(x_max, int(arr[:, 0].max()))
            y_max = max(y_max, int(arr[:, 1].max()))
            z_max = max(z_max, int(arr[:, 2].max()))
        if x_max < 0 or y_max < 0 or z_max < 0:
            return None
        return int(x_max + 1), int(y_max + 1), int(z_max + 1)

    def _build_dense_from_gt_occ_inst_sparse(
        self,
        sparse_list,
        fallback_segmentation_instance3d_txyz=None,
    ):
        """
        Build dense instance-id and class volumes from sparse gt_occ_inst rows.
        Returns:
            dense_inst_txyz: [T,X,Y,Z] long
            dense_cls_txyz:  [T,X,Y,Z] long
        """
        if not isinstance(sparse_list, (list, tuple)) or len(sparse_list) <= 0:
            return None, None

        t_len = int(len(sparse_list))
        spatial = None
        if torch.is_tensor(fallback_segmentation_instance3d_txyz) and fallback_segmentation_instance3d_txyz.dim() == 4:
            spatial = tuple(int(v) for v in fallback_segmentation_instance3d_txyz.shape[1:])
            t_len = min(t_len, int(fallback_segmentation_instance3d_txyz.shape[0]))
        if spatial is None:
            spatial = self._infer_dense_shape_from_sparse(sparse_list)
        if spatial is None:
            return None, None
        x_dim, y_dim, z_dim = spatial
        if x_dim <= 0 or y_dim <= 0 or z_dim <= 0 or t_len <= 0:
            return None, None

        dense_inst = torch.zeros((t_len, x_dim, y_dim, z_dim), dtype=torch.long)
        dense_cls = torch.zeros((t_len, x_dim, y_dim, z_dim), dtype=torch.long)
        for t in range(t_len):
            rows = np.asarray(sparse_list[t], dtype=np.int64)
            if rows.size <= 0:
                continue
            if rows.ndim != 2 or rows.shape[1] < 5:
                continue
            xyz = rows[:, :3]
            cls = rows[:, 3]
            inst = rows[:, 4]
            keep = (
                (xyz[:, 0] >= 0) & (xyz[:, 0] < x_dim)
                & (xyz[:, 1] >= 0) & (xyz[:, 1] < y_dim)
                & (xyz[:, 2] >= 0) & (xyz[:, 2] < z_dim)
                & (inst > 0)
            )
            if not bool(np.any(keep)):
                continue
            xyz = xyz[keep]
            cls = cls[keep]
            inst = inst[keep]
            x_idx = torch.from_numpy(xyz[:, 0]).to(torch.long)
            y_idx = torch.from_numpy(xyz[:, 1]).to(torch.long)
            z_idx = torch.from_numpy(xyz[:, 2]).to(torch.long)
            dense_inst[t, x_idx, y_idx, z_idx] = torch.from_numpy(inst).to(torch.long)
            dense_cls[t, x_idx, y_idx, z_idx] = torch.from_numpy(cls).to(torch.long)
        return dense_inst, dense_cls

    def _build_instance_center_world_targets_from_sparse(
        self,
        sparse_list,
        fallback_segmentation_instance3d_txyz=None,
    ):
        """
        Convert sparse gt_occ_inst rows into per-instance temporal center targets.
        Returns:
            centers_world: [T,N,3] float32
            valid_mask:    [T,N] bool
            instance_ids:  [N] int64
        """
        if not isinstance(sparse_list, (list, tuple)) or len(sparse_list) <= 0:
            return None, None, None

        spatial = None
        t_len = int(len(sparse_list))
        if torch.is_tensor(fallback_segmentation_instance3d_txyz) and fallback_segmentation_instance3d_txyz.dim() == 4:
            spatial = tuple(int(v) for v in fallback_segmentation_instance3d_txyz.shape[1:])
            t_len = min(t_len, int(fallback_segmentation_instance3d_txyz.shape[0]))
        if spatial is None:
            spatial = self._infer_dense_shape_from_sparse(sparse_list)
        if spatial is None:
            return None, None, None
        x_dim, y_dim, z_dim = spatial
        if x_dim <= 0 or y_dim <= 0 or z_dim <= 0 or t_len <= 0:
            return None, None, None

        frame_maps = []
        all_ids = set()
        for t in range(t_len):
            rows = np.asarray(sparse_list[t], dtype=np.int64)
            if rows.size <= 0 or rows.ndim != 2 or rows.shape[1] < 5:
                frame_maps.append({})
                continue
            xyz = rows[:, :3].astype(np.float32, copy=False)
            inst = rows[:, 4].astype(np.int64, copy=False)
            keep = (
                (xyz[:, 0] >= 0) & (xyz[:, 0] < x_dim)
                & (xyz[:, 1] >= 0) & (xyz[:, 1] < y_dim)
                & (xyz[:, 2] >= 0) & (xyz[:, 2] < z_dim)
                & (inst > 0)
            )
            xyz = xyz[keep]
            inst = inst[keep]
            cur = {}
            if inst.size > 0:
                for iid in np.unique(inst):
                    pts = xyz[inst == iid]
                    if pts.shape[0] <= 0:
                        continue
                    cur[int(iid)] = pts.mean(axis=0).astype(np.float32, copy=False)
                    all_ids.add(int(iid))
            frame_maps.append(cur)

        instance_ids = np.asarray(sorted(all_ids), dtype=np.int64)
        n_inst = int(instance_ids.shape[0])
        if n_inst <= 0:
            return (
                torch.zeros((t_len, 0, 3), dtype=torch.float32),
                torch.zeros((t_len, 0), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.long),
            )

        centers_voxel = np.zeros((t_len, n_inst, 3), dtype=np.float32)
        valid_mask = np.zeros((t_len, n_inst), dtype=np.bool_)
        id_to_col = {int(iid): idx for idx, iid in enumerate(instance_ids.tolist())}
        for t, c_map in enumerate(frame_maps):
            for iid, c_xyz in c_map.items():
                col = id_to_col.get(int(iid), None)
                if col is None:
                    continue
                centers_voxel[t, col] = c_xyz
                valid_mask[t, col] = True

        pc = np.asarray(self.point_cloud_range, dtype=np.float32)
        voxel = np.asarray(
            [
                (pc[3] - pc[0]) / float(max(1, x_dim)),
                (pc[4] - pc[1]) / float(max(1, y_dim)),
                (pc[5] - pc[2]) / float(max(1, z_dim)),
            ],
            dtype=np.float32,
        )
        start = pc[:3] + (0.5 * voxel)
        centers_world = centers_voxel * voxel.reshape(1, 1, 3) + start.reshape(1, 1, 3)
        centers_world[~valid_mask] = 0.0
        return (
            torch.from_numpy(centers_world.astype(np.float32, copy=False)),
            torch.from_numpy(valid_mask),
            torch.from_numpy(instance_ids),
        )

    def _build_segmentation_cls_instance3d_from_gt_occ_inst_dense(
        self,
        dense_cls_txyz=None,
        dense_inst_txyz=None,
    ):
        if (
            (not torch.is_tensor(dense_cls_txyz))
            or (not torch.is_tensor(dense_inst_txyz))
            or dense_cls_txyz.dim() != 4
            or dense_inst_txyz.dim() != 4
            or tuple(dense_cls_txyz.shape) != tuple(dense_inst_txyz.shape)
        ):
            return None
        return torch.stack(
            (
                dense_cls_txyz.to(torch.long),
                dense_inst_txyz.to(torch.long),
            ),
            dim=1,
        ).contiguous()

    def _prepare_gt_occ_inst_primary_targets(
        self,
        gt_occ_inst=None,
        fallback_segmentation_instance3d_txyz=None,
    ):
        sparse = self._extract_gt_occ_inst_sparse_list(gt_occ_inst=gt_occ_inst)
        if sparse is None:
            return None
        dense_inst, dense_cls = self._build_dense_from_gt_occ_inst_sparse(
            sparse_list=sparse,
            fallback_segmentation_instance3d_txyz=fallback_segmentation_instance3d_txyz,
        )
        centers_world, centers_valid, ids_n = self._build_instance_center_world_targets_from_sparse(
            sparse_list=sparse,
            fallback_segmentation_instance3d_txyz=fallback_segmentation_instance3d_txyz,
        )
        seg_cls_inst = self._build_segmentation_cls_instance3d_from_gt_occ_inst_dense(
            dense_cls_txyz=dense_cls,
            dense_inst_txyz=dense_inst,
        )
        return {
            "sparse_list": sparse,
            "dense_inst_txyz": dense_inst,
            "dense_cls_txyz": dense_cls,
            "seg_cls_inst_tcxzy": seg_cls_inst,
            "centers_world_tn3": centers_world,
            "centers_valid_tn": centers_valid,
            "instance_ids_n": ids_n,
        }

    def _prepare_segmentation_cls_instance3d(self, segmentation_cls_instance3d=None):
        """
        Normalize dataloader outputs to [T, 2, X, Y, Z] long.

        Expected inputs (current pipeline usually uses batch=1):
        - [T, 2, X, Y, Z]
        - [1, T, 2, X, Y, Z]
        """
        if segmentation_cls_instance3d is None:
            return None

        seg = segmentation_cls_instance3d
        if isinstance(seg, (list, tuple)):
            if len(seg) == 0:
                return None
            if len(seg) == 1:
                seg = seg[0]
            else:
                try:
                    seg = torch.stack(seg, dim=0)
                except Exception as e:
                    raise ValueError(
                        "segmentation_cls_instance3d list could not be stacked. "
                        "Use samples_per_gpu=1 for variable shapes."
                    ) from e

        if not torch.is_tensor(seg):
            raise TypeError(
                f"segmentation_cls_instance3d must be tensor/list/tuple, got {type(seg)}"
            )

        if seg.dim() == 6:
            if seg.shape[0] != 1:
                raise ValueError(
                    "current path expects batch=1 for segmentation_cls_instance3d, "
                    f"got {tuple(seg.shape)}"
                )
            if seg.shape[2] != 2:
                raise ValueError(
                    "Unsupported segmentation_cls_instance3d shape "
                    f"(expect channel=2 at dim2): {tuple(seg.shape)}"
                )
            seg = seg[0]
        elif seg.dim() == 5:
            if seg.shape[1] != 2:
                raise ValueError(
                    "Unsupported segmentation_cls_instance3d shape "
                    f"(expect channel=2 at dim1): {tuple(seg.shape)}"
                )
        else:
            raise ValueError(
                f"Unsupported segmentation_cls_instance3d shape: {tuple(seg.shape)}"
            )

        return seg.to(torch.long).contiguous()

    def _validate_query_class_id_tensor(
        self,
        class_ids: torch.Tensor,
        ignore_index: int = 255,
        background_index: int = 0,
        source_name: str = "segmentation_cls_instance3d",
    ):
        if (not getattr(self, "strict_query_class_id_validation", False)) or (class_ids is None):
            return
        if not torch.is_tensor(class_ids):
            raise TypeError(f"class_ids must be a tensor, got {type(class_ids)}")

        cls = class_ids.to(torch.long)
        valid_mask = (cls != int(ignore_index)) & (cls != int(background_index))
        if not bool(valid_mask.any().item()):
            return

        present_ids = torch.unique(cls[valid_mask], sorted=True)
        expected_ids = sorted(
            int(v) for v in getattr(self, "query_class_ids", tuple()) if int(v) != int(background_index)
        )
        expected_id_set = set(expected_ids)
        invalid_ids = [
            int(v) for v in present_ids.tolist() if int(v) not in expected_id_set
        ]
        if len(invalid_ids) <= 0:
            return

        compact_upper = max(1, len(expected_ids))
        compact_like = all(1 <= int(v) <= compact_upper for v in invalid_ids)
        hint = (
            " Possible cause: bboxcls cache may be using compact 1..K labels instead of "
            "raw nuScenes occupancy ids."
            if compact_like else
            ""
        )
        raise ValueError(
            f"Unexpected raw class ids found in {source_name}: invalid_ids={invalid_ids}, "
            f"present_ids={present_ids.tolist()}, expected_foreground_raw_ids={expected_ids}.{hint}"
        )

    def _prepare_gt_instance_classes_for_matching(
        self,
        segmentation_cls_instance3d=None,
        gt_instance_ids_n: torch.Tensor = None,
        ignore_index: int = 255,
        background_index: int = 0,
    ):
        """
        Resolve one effective class label per instance over the prediction horizon.

        In the current pipeline, semantic voxels and instance voxels are loaded from
        separate caches and then stacked together. That means a single instance id can
        occasionally contain a small amount of mixed semantic labels. To keep training
        robust, we use a majority vote over valid voxels instead of failing hard on
        non-unique labels.

        Returns:
            gt_inst_cls_n: [N] long, invalid entries set to -1
            gt_inst_cls_valid_n: [N] bool
        """
        seg_tcxyz = self._prepare_segmentation_cls_instance3d(
            segmentation_cls_instance3d=segmentation_cls_instance3d
        )
        if seg_tcxyz is None:
            return None, None
        if seg_tcxyz.shape[0] < self.n_future_frames_plus:
            raise ValueError(
                "segmentation_cls_instance3d time dim too short: "
                f"{int(seg_tcxyz.shape[0])} < {int(self.n_future_frames_plus)}"
            )
        seg_tcxyz = seg_tcxyz[-self.n_future_frames_plus:].contiguous()
        if bool(getattr(self, "query_present_only", False)):
            seg_tcxyz, _ = self._select_query_present_frame_slice(
                seg_tcxyz, source_layout="plus6"
            )
        cls_txyz = seg_tcxyz[:, 0].to(torch.long)
        inst_txyz = seg_tcxyz[:, 1].to(torch.long)
        self._validate_query_class_id_tensor(
            class_ids=cls_txyz,
            ignore_index=ignore_index,
            background_index=background_index,
            source_name="segmentation_cls_instance3d[:, 0]",
        )

        if gt_instance_ids_n is None:
            inst_ids_n = torch.unique(inst_txyz)
            keep = (inst_ids_n != int(background_index)) & (inst_ids_n != int(ignore_index))
            inst_ids_n = torch.sort(inst_ids_n[keep]).values
        else:
            inst_ids_n = gt_instance_ids_n.to(device=seg_tcxyz.device, dtype=torch.long).contiguous()

        if inst_ids_n.numel() <= 0:
            return (
                torch.empty((0,), device=seg_tcxyz.device, dtype=torch.long),
                torch.empty((0,), device=seg_tcxyz.device, dtype=torch.bool),
            )

        gt_inst_cls_n = torch.full(
            (int(inst_ids_n.numel()),),
            fill_value=-1,
            device=seg_tcxyz.device,
            dtype=torch.long,
        )
        gt_inst_cls_valid_n = torch.zeros(
            (int(inst_ids_n.numel()),),
            device=seg_tcxyz.device,
            dtype=torch.bool,
        )

        for inst_col in range(int(inst_ids_n.numel())):
            inst_id = int(inst_ids_n[inst_col].item())
            mask = (inst_txyz == inst_id)
            if not bool(mask.any().item()):
                continue
            cls_vals = cls_txyz[mask]
            keep_cls = (cls_vals != int(background_index)) & (cls_vals != int(ignore_index))
            cls_vals = cls_vals[keep_cls]
            if cls_vals.numel() <= 0:
                continue
            uniq_cls, cls_counts = torch.unique(cls_vals, sorted=True, return_counts=True)
            if uniq_cls.numel() == 1:
                gt_inst_cls_n[inst_col] = uniq_cls[0]
                gt_inst_cls_valid_n[inst_col] = True
                continue
            top_count = cls_counts.max()
            top_mask = cls_counts == top_count
            if int(top_mask.sum().item()) != 1:
                # Exact tie: skip cls supervision for this instance rather than
                # injecting an arbitrary label into Hungarian cls cost / cls loss.
                continue
            gt_inst_cls_n[inst_col] = uniq_cls[top_mask][0]
            gt_inst_cls_valid_n[inst_col] = True

        # Convert raw semantic ids to compact classifier ids [0..C-1].
        raw_to_compact = getattr(self, "query_raw_to_compact_class_map", None)
        if torch.is_tensor(raw_to_compact) and raw_to_compact.dim() == 1:
            valid_idx = torch.nonzero(gt_inst_cls_valid_n, as_tuple=False).squeeze(1)
            if int(valid_idx.numel()) > 0:
                raw_cls = gt_inst_cls_n.index_select(0, valid_idx).to(torch.long)
                compact_cls = torch.full_like(raw_cls, fill_value=-1)
                in_range = (raw_cls >= 0) & (raw_cls < int(raw_to_compact.numel()))
                if bool(in_range.any().item()):
                    compact_cls[in_range] = raw_to_compact.to(device=raw_cls.device, dtype=torch.long).index_select(
                        0, raw_cls[in_range]
                    )
                gt_inst_cls_n[valid_idx] = compact_cls
                mapped_valid = compact_cls >= 0
                gt_inst_cls_valid_n[valid_idx] = mapped_valid
                if bool((~mapped_valid).any().item()):
                    invalid_idx = valid_idx[~mapped_valid]
                    gt_inst_cls_n[invalid_idx] = -1

        return gt_inst_cls_n, gt_inst_cls_valid_n
