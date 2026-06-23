import copy
from mmcv.runner import HOOKS, Hook
import time
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
import torch
import torch.distributed as dist


@HOOKS.register_module()
class TrajectoryWarmupHook(Hook):
    def __init__(self, schedule=None):
        self.schedule = self._normalize_schedule(schedule)

    @staticmethod
    def _normalize_schedule(schedule):
        if schedule is None:
            return ()
        normalized = []
        for stage in schedule:
            if not isinstance(stage, dict):
                continue
            item = copy.deepcopy(stage)
            item["begin_epoch"] = int(item["begin_epoch"])
            normalized.append(item)
        normalized.sort(key=lambda x: x["begin_epoch"])
        return tuple(normalized)

    @staticmethod
    def _get_model(runner):
        return runner.model.module if hasattr(runner.model, "module") else runner.model

    def _get_stage(self, epoch):
        if len(self.schedule) == 0:
            return None
        cur_stage = self.schedule[0]
        for stage in self.schedule:
            if int(stage["begin_epoch"]) <= int(epoch):
                cur_stage = stage
            else:
                break
        return cur_stage

    def _apply_stage(self, model, stage):
        if stage is None:
            return
        for key, value in stage.items():
            if key == "begin_epoch":
                continue
            setattr(model, key, value)
        model._query_traj_sched_stage_begin_epoch = int(stage["begin_epoch"])
        model._query_traj_sched_loss_weight = float(getattr(model, "query_traj_loss_weight", 0.0))
        model._query_traj_sched_refine_loss_weight = float(getattr(model, "query_traj_xy_refine_loss_weight", 0.0))
        model._query_traj_sched_mode_cls_loss_weight = float(getattr(model, "query_traj_mode_cls_loss_weight", 0.0))
        model._query_traj_sched_teacher_forcing_enabled = float(
            bool(
                getattr(
                    model,
                    "query_traj_teacher_forcing",
                    getattr(model, "query_traj_teacher_forcing_enabled", False),
                )
            )
        )

    def before_run(self, runner):
        self._apply_stage(self._get_model(runner), self._get_stage(runner.epoch + 1))

    def before_train_epoch(self, runner):
        self._apply_stage(self._get_model(runner), self._get_stage(runner.epoch + 1))

    def before_epoch(self, runner):
        self._apply_stage(self._get_model(runner), self._get_stage(runner.epoch + 1))


@HOOKS.register_module()
class OccEfficiencyHook(Hook):
    def __init__(self, dataloader,  **kwargs):
        self.dataloader = dataloader 
        self.warm_up = 5
        
    def construct_input(self, DUMMY_SHAPE=None, m_info=None):
        if m_info is None:
            m_info = next(iter(self.dataloader))
        img_metas = m_info['img_metas'].data
        input = dict(
            img_metas=img_metas,
        )
        if 'img_inputs' in m_info.keys():
            img_inputs = m_info['img_inputs']
            for i in range(len(img_inputs)):
                if isinstance(img_inputs[i], list):
                    for j in range(len(img_inputs[i])):
                        img_inputs[i][j] = img_inputs[i][j].cuda()
                else:
                    img_inputs[i] = img_inputs[i].cuda()
            input['img_inputs'] = img_inputs
            
        if 'points' in m_info.keys():
            points = m_info['points'].data[0]
            points[0] = points[0].cuda()
            input['points'] = points
        return input
    
    def before_run(self, runner):
        torch.cuda.reset_peak_memory_stats()
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier() 
        
    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

@HOOKS.register_module()
class QueryCenterGradHook(Hook):
    def __init__(self, interval=20, only_rank0=True, print_init=True):
        self.interval = int(interval)
        self.only_rank0 = bool(only_rank0)
        self.print_init = bool(print_init)
        self._iter = -1
        self._handles = []

    def _is_main(self):
        if not self.only_rank0:
            return True
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _get_model(self, runner):
        return runner.model.module if hasattr(runner.model, "module") else runner.model

    def _make_grad_hook(self, name):
        def _hook(grad):
            step = self._iter + 1  # 1-based print
            if self.interval > 0 and (step % self.interval) != 0:
                return grad
            if not self._is_main():
                return grad

            g = grad.detach()
            print(
                f"[CenterGrad][iter {step}] {name}: "
                f"mean_abs={g.abs().mean().item():.3e}, "
                f"max_abs={g.abs().max().item():.3e}, "
                f"l2={g.norm().item():.3e}"
            )
            return grad
        return _hook

    def before_run(self, runner):
        model = self._get_model(runner)
        center_head = model.query_head.center_head

        if self.print_init and self._is_main():
            opt_ids = {id(p) for group in runner.optimizer.param_groups for p in group["params"]}
            for n, p in center_head.named_parameters():
                print(
                    f"[CenterGrad][init] {n}: "
                    f"requires_grad={p.requires_grad}, in_optimizer={id(p) in opt_ids}"
                )

        for n, p in center_head.named_parameters():
            if p.requires_grad:
                self._handles.append(p.register_hook(self._make_grad_hook(n)))

    def before_train_iter(self, runner):
        self._iter = runner.iter

    # mmcv 버전 호환
    def before_iter(self, runner):
        self._iter = runner.iter

    def after_run(self, runner):
        for h in self._handles:
            h.remove()
        self._handles = []
