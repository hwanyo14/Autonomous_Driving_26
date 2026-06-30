import torch


class EfficientOCFGeometryMixin:
    """Geometry helpers shared across forward_train."""

    @staticmethod
    def _detach_if_tensor(x):
        return x.detach() if torch.is_tensor(x) else x

    @staticmethod
    def _pick_tensor(preferred, fallback):
        """Return `preferred` if it is a tensor, else `fallback`."""
        return preferred if torch.is_tensor(preferred) else fallback

    @staticmethod
    def _prepend_past_frames(past, past_count, traj, *, detach=False):
        """Prepend `past[:past_count]` onto `traj` along dim 0.

        Returns `traj` unchanged if either input is not a tensor or the
        shapes are incompatible.
        """
        if not (torch.is_tensor(past) and torch.is_tensor(traj)):
            return traj
        if past.dim() != traj.dim() or int(past.shape[0]) <= past_count:
            return traj
        past_slice = past[:past_count].to(device=traj.device, dtype=traj.dtype)
        if detach:
            past_slice = past_slice.detach()
        return torch.cat([past_slice, traj], dim=0).contiguous()

    def _align_geom_pack(
        self,
        centers, frame_indices, egomotion,
        sigmas, mix_c, mix_s, mix_y, mix_w,
        *, detach=False,
    ):
        """Wrap _align_query_vis_geometry_to_present_frame with tensor guards
        and optional detach.  Returns a dict (empty on failure)."""
        def _t(x):
            if not torch.is_tensor(x):
                return None
            return x.detach() if detach else x

        def _e(x):
            if not torch.is_tensor(x):
                return x
            return x.detach() if detach else x

        pack = self._align_query_vis_geometry_to_present_frame(
            centers_world_tq3=_t(centers),
            traj_frame_indices_t=_e(frame_indices),
            future_egomotion=_e(egomotion),
            gaussian_sigmas_world_tq3=_t(sigmas),
            mixture_centers_world_tqg3=_t(mix_c),
            mixture_sigmas_world_tqg3=_t(mix_s),
            mixture_quat_tqg4=_t(mix_y),
            mixture_weights_tqg=_t(mix_w),
        )
        return pack if isinstance(pack, dict) else {}
