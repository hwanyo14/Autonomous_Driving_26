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

def _running_eval_msg(n, iou_cm, bbox_cm, bbox_rot_cm, iou3d, iou3d_bbox,
                      iou3d_bbox_rot, recall3d, recall3d_rot=None,
                      asset_cm=None, iou3d_asset=None, recall3d_asset=None):
    """One-line running eval summary (movable IoU2d nusocc/bbox_aabb/bbox_rot,
    IoU3d nusocc/bbox_aabb/bbox_rot, Recall3d aabb/rot) from metrics accumulated so far."""
    from projects.occ_plugin.utils.formating import cm_to_ious
    def _mov(cm_list):
        return float(cm_to_ious(sum(cm_list))[1]) if cm_list else float('nan')
    def _mean(xs):
        return float(np.mean(xs)) if xs else float('nan')
    return ("[eval][{}] IoU2d(nusocc)={:.4f} IoU2d(bbox_aabb)={:.4f} "
            "IoU2d(bbox_rot)={:.4f} IoU3d(nusocc)={:.4f} IoU3d(bbox_aabb)={:.4f} "
            "IoU3d(bbox_rot)={:.4f} Recall3d(bbox_aabb)={:.4f} "
            "Recall3d(bbox_rot)={:.4f} IoU2d(asset)={:.4f} "
            "IoU3d(asset)={:.4f} Recall3d(asset)={:.4f}").format(
                n, _mov(iou_cm), _mov(bbox_cm), _mov(bbox_rot_cm), _mean(iou3d),
                _mean(iou3d_bbox), _mean(iou3d_bbox_rot), _mean(recall3d),
                _mean(recall3d_rot if recall3d_rot is not None else []),
                _mov(asset_cm or []), _mean(iou3d_asset or []),
                _mean(recall3d_asset or []))


def _asset_msg(n, cm_p, cm_f, i3_p, i3_f, suffix=""):
    """asset present/future 요약 한 줄."""
    from projects.occ_plugin.utils.formating import cm_to_ious
    def _iou2d(cms):
        return float(cm_to_ious(sum(cms))[1]) if cms else float("nan")
    def _m(xs):
        return float(np.mean(xs)) if xs else float("nan")
    return ("[eval][{}] IoU3d(asset)_future={:.4f} IoU2d(asset)_future={:.4f} "
            "IoU3d(asset)_present={:.4f} IoU2d(asset)_present={:.4f}{}").format(
                n, _m(i3_f), _iou2d(cm_f), _m(i3_p), _iou2d(cm_p), suffix)


def _comps_msg(tag, comps_sum):
    tp, fp, fn = [float(v) for v in np.asarray(comps_sum, dtype=np.float64)[:3]]
    denom = tp + fp + fn
    return "[iou3d comps asset {}] TP={:.3e} FP={:.3e} FN={:.3e} | micro IoU3d={:.4f}".format(
        tag, tp, fp, fn, tp / denom if denom > 0 else float("nan"))


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


def _iou3d_asset_comps_line(comps_sum):
    tp, fp, fn = [float(v) for v in np.asarray(comps_sum, dtype=np.float64)[:3]]
    denom = tp + fp + fn
    return ("[iou3d comps asset] TP={:.3e} FP={:.3e} FN={:.3e} "
            "| micro IoU3d={:.4f}").format(
                tp, fp, fn, tp / denom if denom > 0 else float("nan"))


def _distributed_running_eval_msg(n, dataset_size, iou_cm, bbox_cm, bbox_rot_cm,
                                  iou3d, iou3d_bbox, iou3d_bbox_rot, recall3d,
                                  recall3d_rot=None, asset_cm=None,
                                  iou3d_asset=None, recall3d_asset=None):
    from projects.occ_plugin.utils.formating import cm_to_ious

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    def _cm_tensor(cm_list):
        arr = sum(cm_list) if cm_list else np.zeros((2, 2), dtype=np.int64)
        return torch.as_tensor(arr, dtype=torch.float64, device=device)

    cm_iou = _cm_tensor(iou_cm)
    cm_bbox = _cm_tensor(bbox_cm)
    cm_bbox_rot = _cm_tensor(bbox_rot_cm)
    cm_asset = _cm_tensor(asset_cm or [])
    scalars = torch.tensor(
        [
            float(np.sum(iou3d)) if len(iou3d) else 0.0,
            float(len(iou3d)),
            float(np.sum(recall3d)) if len(recall3d) else 0.0,
            float(len(recall3d)),
            float(np.sum(iou3d_bbox)) if len(iou3d_bbox) else 0.0,
            float(len(iou3d_bbox)),
            float(np.sum(iou3d_bbox_rot)) if len(iou3d_bbox_rot) else 0.0,
            float(len(iou3d_bbox_rot)),
            float(np.sum(recall3d_rot)) if (recall3d_rot and len(recall3d_rot)) else 0.0,
            float(len(recall3d_rot)) if recall3d_rot else 0.0,
            float(np.sum(iou3d_asset)) if iou3d_asset else 0.0,
            float(len(iou3d_asset)) if iou3d_asset else 0.0,
            float(np.sum(recall3d_asset)) if recall3d_asset else 0.0,
            float(len(recall3d_asset)) if recall3d_asset else 0.0,
        ],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(cm_iou, op=dist.ReduceOp.SUM)
    dist.all_reduce(cm_bbox, op=dist.ReduceOp.SUM)
    dist.all_reduce(cm_bbox_rot, op=dist.ReduceOp.SUM)
    dist.all_reduce(cm_asset, op=dist.ReduceOp.SUM)
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)

    cm_iou_np = cm_iou.cpu().numpy().astype(np.int64)
    cm_bbox_np = cm_bbox.cpu().numpy().astype(np.int64)
    cm_bbox_rot_np = cm_bbox_rot.cpu().numpy().astype(np.int64)
    cm_asset_np = cm_asset.cpu().numpy().astype(np.int64)
    iou3d_mean = float(scalars[0].item() / scalars[1].item()) if scalars[1].item() > 0 else float("nan")
    recall3d_mean = float(scalars[2].item() / scalars[3].item()) if scalars[3].item() > 0 else float("nan")
    iou3d_bbox_mean = float(scalars[4].item() / scalars[5].item()) if scalars[5].item() > 0 else float("nan")
    iou3d_bbox_rot_mean = float(scalars[6].item() / scalars[7].item()) if scalars[7].item() > 0 else float("nan")
    recall3d_rot_mean = float(scalars[8].item() / scalars[9].item()) if scalars[9].item() > 0 else float("nan")
    iou3d_asset_mean = float(scalars[10].item() / scalars[11].item()) if scalars[11].item() > 0 else float("nan")
    recall3d_asset_mean = float(scalars[12].item() / scalars[13].item()) if scalars[13].item() > 0 else float("nan")
    n = min(int(n), int(dataset_size))
    return ("[eval][{}] IoU2d(nusocc)={:.4f} IoU2d(bbox_aabb)={:.4f} "
            "IoU2d(bbox_rot)={:.4f} IoU3d(nusocc)={:.4f} IoU3d(bbox_aabb)={:.4f} "
            "IoU3d(bbox_rot)={:.4f} Recall3d(bbox_aabb)={:.4f} "
            "Recall3d(bbox_rot)={:.4f} IoU2d(asset)={:.4f} "
            "IoU3d(asset)={:.4f} Recall3d(asset)={:.4f} (all ranks)").format(
                n,
                float(cm_to_ious(cm_iou_np)[1]),
                float(cm_to_ious(cm_bbox_np)[1]),
                float(cm_to_ious(cm_bbox_rot_np)[1]),
                iou3d_mean,
                iou3d_bbox_mean,
                iou3d_bbox_rot_mean,
                recall3d_mean,
                recall3d_rot_mean,
                float(cm_to_ious(cm_asset_np)[1]),
                iou3d_asset_mean,
                recall3d_asset_mean,
            )


def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    logger.info(parameter_count_table(model))

    cm_p, cm_f, i3_p, i3_f = [], [], [], []
    comps_p = np.zeros(3, dtype=np.float64)
    comps_f = np.zeros(3, dtype=np.float64)

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        cm_p, cm_f, i3_p, i3_f, comps_p, comps_f = _accumulate(
            result, cm_p, cm_f, i3_p, i3_f, comps_p, comps_f)
        prog_bar.update()
        if (i + 1) % 50 == 0:
            print("\n" + _asset_msg(i + 1, cm_p, cm_f, i3_p, i3_f), flush=True)

    final_msg = _asset_msg(len(dataset), cm_p, cm_f, i3_p, i3_f, "  [FINAL]")
    print("\n" + final_msg, flush=True)
    _append_live_eval_log(final_msg)
    for tag, c in (("future", comps_f), ("present", comps_p)):
        msg = _comps_msg(tag, c)
        print(msg, flush=True)
        _append_live_eval_log(msg)

    return {
        'hist_for_iou_asset_present': [sum(cm_p)] if cm_p else [],
        'hist_for_iou_asset_future': [sum(cm_f)] if cm_f else [],
        'iou_3d_asset_present': i3_p,
        'iou_3d_asset_future': i3_f,
        'iou_3d_asset_present_comps': [comps_p],
        'iou_3d_asset_future_comps': [comps_f],
    }


def _accumulate(result, cm_p, cm_f, i3_p, i3_f, comps_p, comps_f):
    """result dict에서 asset present/future 지표를 누적."""
    if 'hist_for_iou_asset_present' in result:
        cm_p.append(result['hist_for_iou_asset_present'])
    if 'hist_for_iou_asset_future' in result:
        cm_f.append(result['hist_for_iou_asset_future'])
    for key, acc in (('iou_3d_asset_present', i3_p), ('iou_3d_asset_future', i3_f)):
        if key in result and not np.isnan(result[key]):
            acc.append(result[key])
    for key, acc in (('iou_3d_asset_present_comps', comps_p),
                     ('iou_3d_asset_future_comps', comps_f)):
        c = result.get(key, None)
        if isinstance(c, dict):
            acc += np.array([c.get('tp', 0.0), c.get('fp', 0.0), c.get('fn', 0.0)])
    return cm_p, cm_f, i3_p, i3_f, comps_p, comps_f


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Test model with multiple gpus (asset present/future 지표 전용)."""
    model.eval()
    cm_p, cm_f, i3_p, i3_f = [], [], [], []
    comps_p = np.zeros(3, dtype=np.float64)
    comps_f = np.zeros(3, dtype=np.float64)

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    metric_every = int(os.environ.get("EOCF_EVAL_METRIC_EVERY",
                                      os.environ.get("EOCF_EVAL_VIS_EVERY", "50")))
    # 부분 eval: 전 rank 합산 N샘플 처리 후 조기 종료. 0=전체.
    max_samples = int(os.environ.get("EOCF_EVAL_MAX_SAMPLES", "0"))

    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    logger = get_root_logger()
    logger.info(parameter_count_table(model))

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        cm_p, cm_f, i3_p, i3_f, comps_p, comps_f = _accumulate(
            result, cm_p, cm_f, i3_p, i3_f, comps_p, comps_f)
        if rank == 0:
            for _ in range(world_size):
                prog_bar.update()
        if metric_every > 0 and (i + 1) % metric_every == 0:
            msg = _dist_asset_msg((i + 1) * world_size, len(dataset),
                                  cm_p, cm_f, i3_p, i3_f)
            if rank == 0:
                print("\n" + msg, flush=True)
                _append_live_eval_log(msg)
        if max_samples > 0 and (i + 1) * world_size >= max_samples:
            if rank == 0:
                print(f"\n[eval] early stop at ~{(i + 1) * world_size} samples "
                      f"(EOCF_EVAL_MAX_SAMPLES={max_samples})", flush=True)
            break

    # 최종 요약 — all_reduce collective라 전 rank가 함께 호출(출력만 rank0).
    if metric_every > 0:
        n_done = (i + 1) * world_size if max_samples > 0 else len(dataset)
        final_msg = _dist_asset_msg(n_done, len(dataset), cm_p, cm_f, i3_p, i3_f) + "  [FINAL]"
        device = (torch.device("cuda", torch.cuda.current_device())
                  if torch.cuda.is_available() else torch.device("cpu"))
        comps_msgs = []
        for tag, c in (("future", comps_f), ("present", comps_p)):
            t = torch.as_tensor(c, dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            comps_msgs.append(_comps_msg(tag, t.cpu().numpy()))
        if rank == 0:
            print("\n" + final_msg, flush=True)
            _append_live_eval_log(final_msg)
            for m in comps_msgs:
                print(m, flush=True)
                _append_live_eval_log(m)

    res = {}
    if 'hist_for_iou_asset_present' in result.keys():
        res['hist_for_iou_asset_present'] = collect_results_cpu(
            [sum(cm_p)] if cm_p else [], len(dataset), tmpdir)
        res['hist_for_iou_asset_future'] = collect_results_cpu(
            [sum(cm_f)] if cm_f else [], len(dataset), tmpdir)
        res['iou_3d_asset_present'] = collect_results_cpu(i3_p, len(dataset), tmpdir)
        res['iou_3d_asset_future'] = collect_results_cpu(i3_f, len(dataset), tmpdir)
        res['iou_3d_asset_present_comps'] = collect_results_cpu([comps_p], len(dataset), tmpdir)
        res['iou_3d_asset_future_comps'] = collect_results_cpu([comps_f], len(dataset), tmpdir)
    return res


def _dist_asset_msg(n, dataset_size, cm_p, cm_f, i3_p, i3_f):
    """전 rank all_reduce 후 asset present/future 요약."""
    from projects.occ_plugin.utils.formating import cm_to_ious
    device = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))

    def _cm_t(cms):
        arr = sum(cms) if cms else np.zeros((2, 2), dtype=np.int64)
        return torch.as_tensor(arr, dtype=torch.float64, device=device)

    tp_cm = _cm_t(cm_p)
    tf_cm = _cm_t(cm_f)
    scalars = torch.tensor(
        [float(np.sum(i3_p)) if i3_p else 0.0, float(len(i3_p)),
         float(np.sum(i3_f)) if i3_f else 0.0, float(len(i3_f))],
        dtype=torch.float64, device=device)
    dist.all_reduce(tp_cm, op=dist.ReduceOp.SUM)
    dist.all_reduce(tf_cm, op=dist.ReduceOp.SUM)
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)

    def _mean(a, b):
        return float(scalars[a].item() / scalars[b].item()) if scalars[b].item() > 0 else float("nan")

    n = min(int(n), int(dataset_size))
    return ("[eval][{}] IoU3d(asset)_future={:.4f} IoU2d(asset)_future={:.4f} "
            "IoU3d(asset)_present={:.4f} IoU2d(asset)_present={:.4f} (all ranks)").format(
                n,
                _mean(2, 3),
                float(cm_to_ious(tf_cm.cpu().numpy().astype(np.int64))[1]),
                _mean(0, 1),
                float(cm_to_ious(tp_cm.cpu().numpy().astype(np.int64))[1]))


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
