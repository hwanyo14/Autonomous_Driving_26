# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jingyi Xu, following Zhiqi Li and Junyi Ma
# ---------------------------------------------
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core import encode_mask_results
import numpy as np
import pycocotools.mask as mask_util
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def _running_eval_msg(n, iou_cm, bbox_cm, iou3d, recall3d):
    from projects.occ_plugin.utils.formating import cm_to_ious

    def _mov(cm_list):
        return float(cm_to_ious(sum(cm_list))[1]) if cm_list else float('nan')

    def _mean(xs):
        return float(np.mean(xs)) if len(xs) else float('nan')

    return ("[eval][{}] IoU2d(nusocc)={:.4f} IoU2d(bbox_aabb)={:.4f} "
            "IoU3d={:.4f} Recall3d={:.4f}").format(
                n, _mov(iou_cm), _mov(bbox_cm), _mean(iou3d), _mean(recall3d))


def _append_live_eval_log(msg):
    vis_dir = os.environ.get("EOCF_EVAL_VIS_DIR", "")
    if not vis_dir:
        return
    try:
        os.makedirs(vis_dir, exist_ok=True)
        with open(osp.join(vis_dir, "eval_metrics_live.log"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _distributed_running_eval_msg(n, dataset_size, iou_cm, bbox_cm, iou3d, recall3d):
    from projects.occ_plugin.utils.formating import cm_to_ious

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    def _cm_tensor(cm_list):
        arr = sum(cm_list) if cm_list else np.zeros((2, 2), dtype=np.int64)
        return torch.as_tensor(arr, dtype=torch.float64, device=device)

    cm_iou = _cm_tensor(iou_cm)
    cm_bbox = _cm_tensor(bbox_cm)
    scalars = torch.tensor(
        [
            float(np.sum(iou3d)) if len(iou3d) else 0.0,
            float(len(iou3d)),
            float(np.sum(recall3d)) if len(recall3d) else 0.0,
            float(len(recall3d)),
        ],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(cm_iou, op=dist.ReduceOp.SUM)
    dist.all_reduce(cm_bbox, op=dist.ReduceOp.SUM)
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)

    cm_iou_np = cm_iou.cpu().numpy().astype(np.int64)
    cm_bbox_np = cm_bbox.cpu().numpy().astype(np.int64)
    iou3d_mean = float(scalars[0].item() / scalars[1].item()) if scalars[1].item() > 0 else float("nan")
    recall3d_mean = float(scalars[2].item() / scalars[3].item()) if scalars[3].item() > 0 else float("nan")
    n = min(int(n), int(dataset_size))
    return ("[eval][{}] IoU2d(nusocc)={:.4f} IoU2d(bbox_aabb)={:.4f} "
            "IoU3d={:.4f} Recall3d={:.4f} (all ranks)").format(
                n,
                float(cm_to_ious(cm_iou_np)[1]),
                float(cm_to_ious(cm_bbox_np)[1]),
                iou3d_mean,
                recall3d_mean,
            )


def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    logger.info(parameter_count_table(model))

    iou_metric, iou_bbox_metric = [], []
    iou_3d_metric, recall_3d_metric = [], []

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if 'hist_for_iou' in result.keys():
            iou_metric.append(result['hist_for_iou'])
        if 'hist_for_iou_bbox' in result.keys():
            iou_bbox_metric.append(result['hist_for_iou_bbox'])
        if 'iou_3d' in result.keys() and not np.isnan(result['iou_3d']):
            iou_3d_metric.append(result['iou_3d'])
        if 'recall_3d' in result.keys() and not np.isnan(result['recall_3d']):
            recall_3d_metric.append(result['recall_3d'])

        prog_bar.update()
        if (i + 1) % 50 == 0:
            print("\n" + _running_eval_msg(i + 1, iou_metric, iou_bbox_metric,
                                           iou_3d_metric, recall_3d_metric), flush=True)

    res = {
        'hist_for_iou': [sum(iou_metric)] if iou_metric else [],
        'hist_for_iou_bbox': [sum(iou_bbox_metric)] if iou_bbox_metric else [],
        'iou_3d': iou_3d_metric,
        'recall_3d': recall_3d_metric,
    }
    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()
    
    # init predictions
    iou_metric = []
    iou_bbox_metric = []
    height_l1_metric = []
    vpq_metric = []
    iou_3d_metric = []
    recall_3d_metric = []

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    metric_every = int(os.environ.get("EOCF_EVAL_METRIC_EVERY", os.environ.get("EOCF_EVAL_VIS_EVERY", "50")))
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    
    for i, data in enumerate(data_loader):

        with torch.no_grad():

            result = model(return_loss=False, rescale=True, **data)
            
            if 'hist_for_iou' in result.keys():
                iou_metric.append(result['hist_for_iou'])

            if 'hist_for_iou_bbox' in result.keys():
                iou_bbox_metric.append(result['hist_for_iou_bbox'])

            if 'height_l1' in result.keys():
                if not torch.isnan(result['height_l1']):
                    height_l1_metric.append(result['height_l1'])
            if 'vpq' in result.keys():
                vpq_metric.append(result['vpq'])
            
            if 'iou_3d' in result.keys():
                if not np.isnan(result['iou_3d']):
                    iou_3d_metric.append(result['iou_3d'])
            
            if 'recall_3d' in result.keys():
                if not np.isnan(result['recall_3d']):
                    recall_3d_metric.append(result['recall_3d'])

            batch_size = 1
                
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
        if metric_every > 0 and (i + 1) % metric_every == 0:
            msg = _distributed_running_eval_msg(
                (i + 1) * world_size,
                len(dataset),
                iou_metric,
                iou_bbox_metric,
                iou_3d_metric,
                recall_3d_metric,
            )
            if rank == 0:
                print("\n" + msg, flush=True)
                _append_live_eval_log(msg)

    # collect lists from multi-GPUs
    res = {}

    if 'hist_for_iou' in result.keys():
        iou_metric = [sum(iou_metric)]
        iou_metric = collect_results_cpu(iou_metric, len(dataset), tmpdir)
        res['hist_for_iou'] = iou_metric

    if 'hist_for_iou_bbox' in result.keys():
        iou_bbox_metric = [sum(iou_bbox_metric)]
        iou_bbox_metric = collect_results_cpu(iou_bbox_metric, len(dataset), tmpdir)
        res['hist_for_iou_bbox'] = iou_bbox_metric

    if 'height_l1' in result.keys():
        height_l1_metric = collect_results_cpu(height_l1_metric, len(dataset), tmpdir)
        res['height_l1'] = height_l1_metric
    
    if 'iou_3d' in result.keys():
        iou_3d_metric = collect_results_cpu(iou_3d_metric, len(dataset), tmpdir)
        res['iou_3d'] = iou_3d_metric

    if 'recall_3d' in result.keys():
        recall_3d_metric = collect_results_cpu(recall_3d_metric, len(dataset), tmpdir)
        res['recall_3d'] = recall_3d_metric

    if 'vpq' in result.keys():
        res['vpq_len'] = len(dataset)
        vpq_metric = [sum(vpq_metric)]
        vpq_metric = collect_results_cpu(vpq_metric, len(dataset), tmpdir)
        res['vpq_metric'] = vpq_metric

    return res

def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    # collect all parts
    if rank == 0:
    
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        # sort the results
        if type == 'list':
            ordered_results = []
            for res in part_list:  
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
        
        else:
            raise NotImplementedError
        
        # remove tmp dir
        shutil.rmtree(tmpdir)
    
    dist.barrier()

    if rank != 0:
        return None
    
    return ordered_results
