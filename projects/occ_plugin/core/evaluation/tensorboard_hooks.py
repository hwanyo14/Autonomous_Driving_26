from mmcv.runner import HOOKS
from mmcv.runner.dist_utils import master_only

try:
    from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook
except Exception:
    from mmcv.runner.hooks.logger import TensorboardLoggerHook


@HOOKS.register_module()
class TensorboardLoggerHookSplitTabs(TensorboardLoggerHook):
    """Route dbg metrics to dbg/* and keep train/* focused on loss/grad_norm."""

    @staticmethod
    def _as_dbg_tag(tag: str):
        tag = str(tag)
        if tag.startswith("dbg/"):
            return tag
        if tag.startswith("dbg_"):
            return f"dbg/{tag[4:]}"
        return None

    @staticmethod
    def _allow_train_tag(tag: str) -> bool:
        tag = str(tag)
        return tag == "loss" or tag.startswith("loss") or ("grad_norm" in tag)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(
            runner,
            allow_text=True,
            add_mode=False,
        )
        mode = self.get_mode(runner)
        step = self.get_iter(runner)
        for tag, val in tags.items():
            tag = str(tag)
            dbg_tag = self._as_dbg_tag(tag)
            if dbg_tag is not None:
                out_tag = dbg_tag
            elif mode == "train":
                if not self._allow_train_tag(tag):
                    continue
                out_tag = f"train/{tag}"
            else:
                out_tag = f"{mode}/{tag}"
            if isinstance(val, str):
                self.writer.add_text(out_tag, val, step)
            else:
                self.writer.add_scalar(out_tag, val, step)
