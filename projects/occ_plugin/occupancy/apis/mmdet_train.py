# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jingyi Xu, following OpenOccupancy of Xiaofeng Wang and Cam4DOcc of Junyi Ma
# ---------------------------------------------
import random
import warnings
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, Hook, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg
from mmdet.core import EvalHook
from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.occ_plugin.datasets.builder import build_dataloader
from projects.occ_plugin.core.evaluation.eval_hooks import OccDistEvalHook, OccEvalHook
from projects.occ_plugin.datasets import custom_build_dataset


class VisualizationIterSyncHook(Hook):
    def _unwrap_model(self, runner):
        return runner.model.module if hasattr(runner.model, "module") else runner.model

    def _sync(self, runner):
        model = self._unwrap_model(runner)
        if hasattr(model, "set_train_iteration"):
            model.set_train_iteration(int(runner.iter) + 1, one_based=True)

    def before_train_iter(self, runner):
        self._sync(runner)

    # mmcv version compatibility
    def before_iter(self, runner):
        self._sync(runner)

def custom_train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    
    logger = get_root_logger(cfg.log_level)
    
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        ) for ds in dataset
    ]
    
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        static_graph = cfg.get('ddp_static_graph', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    assert 'runner' in cfg
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting TODO
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # Debug: center head gradient monitor
    # runner.register_hook(
    #     build_from_cfg(
    #         dict(type='QueryCenterGradHook', interval=8, only_rank0=True, print_init=True),
    #         HOOKS
    #     ),
    #     priority='NORMAL'
    # )

    # if validate and cfg.data.get('val') is not None:
    #     val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))
    #     val_dataloader = build_dataloader(
    #         val_dataset,
    #         cfg.data.samples_per_gpu,
    #         cfg.data.workers_per_gpu,
    #         len(cfg.gpu_ids),
    #         dist=distributed,
    #         shuffle=False,
    #         seed=cfg.seed,
    #         shuffler_sampler=cfg.data.shuffler_sampler,
    #         nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    #     )
    #     eval_cfg = cfg.get('evaluation', {}).copy()
    #     eval_cfg.setdefault('interval', 1)
    #     eval_hook = OccDistEvalHook if distributed else OccEvalHook
    #     runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    runner.register_hook(VisualizationIterSyncHook(), priority='VERY_HIGH')
    
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    rank, world_size = get_dist_info()
    if cfg.resume_from:
        if rank == 0:
            print("-------------")
            print("resume from " + cfg.resume_from)
            print("-------------")
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        if rank == 0:
            print("-------------")
            print("load from " + cfg.load_from)
            print("-------------")
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
