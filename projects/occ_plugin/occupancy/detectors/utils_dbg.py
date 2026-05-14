import torch.distributed as dist


class EfficientOCFDebugMixin:

    @staticmethod
    def _namespace_dbg_logs(losses: dict) -> dict:
        """Move dbg_* metrics under dbg/... namespace for cleaner TensorBoard grouping."""
        if not isinstance(losses, dict):
            return losses
        remap = []
        for key in list(losses.keys()):
            if isinstance(key, str) and key.startswith("dbg_"):
                remap.append((key, f"dbg/{key[4:]}"))
        for old_key, new_key in remap:
            losses[new_key] = losses.pop(old_key)
        return losses

    def _compute_gt2p_cooldown_scale(self, train_iter: int = None) -> float:
        if not self.query_gt2p_cooldown_enabled:
            return 1.0

        it1 = int(self._query_train_iter if train_iter is None else train_iter)
        it0 = max(0, it1 - 1)
        start = int(self.query_gt2p_cooldown_start_iter)
        dur = int(self.query_gt2p_cooldown_iters)
        min_s = float(self.query_gt2p_cooldown_min_scale)

        if it0 <= start:
            return 1.0
        if dur <= 0:
            return min_s

        progress = float(it0 - start) / float(dur)
        progress = min(max(progress, 0.0), 1.0)
        scale = max(min_s, 1.0 - progress)
        return float(scale)

    def _is_main_process(self) -> bool:
        if not dist.is_available():
            return True
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0
