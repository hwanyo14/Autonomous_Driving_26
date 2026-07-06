"""
dbg: matched query offset-only spread vs GT half-extent, by GT size bin.
Runs epoch_1 and epoch_10 checkpoints of subset_scale_offset on the SAME
train samples (containing large instances) and captures:
  - query_offset_spread_tq3 (offset-only spread, sigma excluded)
  - inst_match_result (matched query<->instance pairs)
  - mixture centers/sigmas/weights (for BEV vis)
Saves stats JSON + per-sample npz to OUT_DIR.
"""
import os, sys, json, pickle, argparse

REPO = '/NHNHOME/WORKSPACE/0526040009_A/hwanhee/Autonomous_Driving_26_ksh_0630'
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

import importlib
importlib.import_module('projects.occ_plugin')

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dbg_out')
os.makedirs(OUT_DIR, exist_ok=True)

CFG = os.path.join(REPO, 'work_dirs/subset_scale_offset/subset_scale_offset.py')
CKPTS = {
    'ep1': os.path.join(REPO, 'work_dirs/subset_scale_offset/epoch_1_lss_only.pth'),
    'ep10': os.path.join(REPO, 'work_dirs/subset_scale_offset/epoch_10_lss_only.pth'),
}
N_BIG = 6          # samples containing a large instance
N_NORMAL = 4       # extra ordinary samples (car-dominated)
BIG_LEN_M = 8.0    # full length threshold for "large" (bus/trailer)

CAPTURE_KEYS = (
    'mixture_centers_world_tqg3', 'mixture_sigmas_world_tqg3',
    'mixture_weights_tqg', 'centers_world_tq3', 'query_offset_spread_tq3',
)


def pick_candidate_indices(cfg):
    ann = cfg.data.train.ann_file
    with open(ann, 'rb') as f:
        d = pickle.load(f)
    infos = d['infos'] if isinstance(d, dict) else d
    big, normal = [], []
    for i, info in enumerate(infos):
        boxes = info.get('gt_boxes', None)
        names = info.get('gt_names', None)
        if boxes is None or len(boxes) == 0:
            continue
        boxes = np.asarray(boxes)
        horiz = boxes[:, 3:5].max(axis=1)  # max(w, l)
        has_big = bool((horiz > BIG_LEN_M).any())
        if names is not None:
            names = np.asarray(names)
            has_big = has_big and any(('bus' in n or 'trailer' in n or 'truck' in n) for n in names[horiz > BIG_LEN_M])
        if has_big:
            big.append(i)
        elif (horiz > 3.5).any():
            normal.append(i)
    return big, normal, len(infos)


def fetch_sample(dataset, idx):
    try:
        return dataset[idx]
    except Exception as e:
        print(f'[fetch] idx {idx} failed: {e}')
        return None


def run_model(model, data_gpu):
    model._last_query_offset_spread = None
    model._last_inst_match_result = None
    with torch.no_grad():
        model.forward_train(**data_gpu)
    cap = dict(model._dbg_capture) if getattr(model, '_dbg_capture', None) else {}
    spread = model._last_query_offset_spread
    match = model._last_inst_match_result
    return spread, match, cap


def install_capture(model):
    qh = model.query_head
    orig = qh.apply_lifted_centers_to_outputs

    def wrapped(*a, **k):
        out = orig(*a, **k)
        cap = {}
        for key in CAPTURE_KEYS:
            v = out.get(key, None)
            cap[key] = v.detach().float().cpu().numpy() if torch.is_tensor(v) else None
        cap['query_present_local_idx'] = int(out.get('query_present_local_idx', 0) or 0)
        model._dbg_capture = cap
        return out
    qh.apply_lifted_centers_to_outputs = wrapped


def matched_stats(model, spread_tq3, match, data_gpu, target_frac):
    """per matched pair: gt half extent, offset spread (mean over frames)."""
    if spread_tq3 is None or not isinstance(match, dict):
        return None
    mq, gt_size_k3 = model._gather_matched_gt_sizes(
        match, data_gpu.get('gt_instance_ids'), data_gpu.get('gt_instance_sizes'),
        spread_tq3.device)
    if mq is None:
        return None
    q_count = int(spread_tq3.shape[1])
    keep = (mq >= 0) & (mq < q_count)
    mq, gt_size_k3 = mq[keep], gt_size_k3[keep]
    sp_k3 = spread_tq3.mean(dim=0).index_select(0, mq)
    half = 0.5 * gt_size_k3
    return {
        'matched_query_idx': mq.cpu().numpy(),
        'gt_half_xyz': half.cpu().numpy(),
        'spread_xyz': sp_k3.cpu().numpy(),
        'tgt_xy': (target_frac * half[:, :2].amax(-1)).cpu().numpy(),
        'act_xy': sp_k3[:, :2].amax(-1).cpu().numpy(),
    }


def main():
    cfg = Config.fromfile(CFG)
    cfg.data.train.pop('type', None)
    cfg.data.train['type'] = 'EfficientOCFDataset'

    big_idx, normal_idx, n_infos = pick_candidate_indices(cfg)
    print(f'[scan] infos={n_infos} big-candidates={len(big_idx)} normal={len(normal_idx)}')

    dataset = build_dataset(cfg.data.train)
    n_ds = len(dataset)
    print(f'[dataset] len={n_ds}')
    # info index -> dataset index assumed aligned; clamp
    cand = [i for i in big_idx if i < n_ds][:N_BIG * 3] + [i for i in normal_idx if i < n_ds][:N_NORMAL]

    target_frac = float(cfg.model.get('query_scale_spread_target_frac', 0.5))
    print(f'[cfg] target_frac={target_frac}')

    models = {}
    for tag, ck in CKPTS.items():
        m = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        load_checkpoint(m, ck, map_location='cpu')
        m = m.cuda().eval()
        m.set_train_iteration(6000)
        install_capture(m)
        models[tag] = m
        print(f'[model] {tag} loaded from {os.path.basename(ck)}')

    results = {tag: [] for tag in models}
    n_big_done = 0
    n_norm_done = 0
    for idx in cand:
        want_big = idx in big_idx
        if want_big and n_big_done >= N_BIG:
            continue
        if (not want_big) and n_norm_done >= N_NORMAL:
            continue
        item = fetch_sample(dataset, idx)
        if item is None:
            continue
        batch = collate([item], samples_per_gpu=1)
        data_gpu = scatter(batch, [torch.cuda.current_device()])[0]
        sizes = data_gpu.get('gt_instance_sizes')
        s = sizes[0] if isinstance(sizes, (list, tuple)) else sizes
        if torch.is_tensor(s):
            s2 = s[0] if s.dim() == 3 else s
            max_len = float(s2[:, :2].max().item()) if s2.numel() else 0.0
        else:
            max_len = 0.0
        if want_big and max_len < BIG_LEN_M * 0.75:
            print(f'[skip] idx {idx}: pipeline max_len={max_len:.1f} < big thr')
            continue
        print(f'[run] idx={idx} big={want_big} pipeline_max_len={max_len:.1f}m')
        ok = True
        for tag, m in models.items():
            spread, match, cap = run_model(m, data_gpu)
            st = matched_stats(m, spread, match, data_gpu, target_frac)
            if st is None:
                print(f'  [{tag}] no matched stats'); ok = False; break
            rec = {'idx': int(idx), 'big': bool(want_big), 'max_len': max_len,
                   'gt_half_xyz': st['gt_half_xyz'].tolist(),
                   'spread_xyz': st['spread_xyz'].tolist(),
                   'tgt_xy': st['tgt_xy'].tolist(), 'act_xy': st['act_xy'].tolist(),
                   'matched_query_idx': st['matched_query_idx'].tolist()}
            results[tag].append(rec)
            np.savez_compressed(
                os.path.join(OUT_DIR, f'cap_{tag}_idx{idx}.npz'),
                **{k: v for k, v in cap.items() if isinstance(v, np.ndarray)},
                present_idx=np.array(cap.get('query_present_local_idx', 0)),
                matched_query_idx=st['matched_query_idx'],
                gt_half_xyz=st['gt_half_xyz'],
                spread_xyz=st['spread_xyz'],
                gt_centers=_gt_centers_np(data_gpu),
                gt_ids=_np(data_gpu.get('gt_instance_ids')),
                gt_sizes=_np(data_gpu.get('gt_instance_sizes')),
            )
            deficit = np.maximum(st['tgt_xy'] - st['act_xy'], 0)
            print(f"  [{tag}] matched={len(st['act_xy'])} "
                  f"mean_act_xy={st['act_xy'].mean():.2f} mean_tgt_xy={st['tgt_xy'].mean():.2f} "
                  f"deficit>0: {(deficit > 0).sum()}")
        if ok:
            if want_big: n_big_done += 1
            else: n_norm_done += 1
        if n_big_done >= N_BIG and n_norm_done >= N_NORMAL:
            break

    with open(os.path.join(OUT_DIR, 'spread_stats.json'), 'w') as f:
        json.dump(results, f, indent=1)
    print(f'[done] big={n_big_done} normal={n_norm_done} -> {OUT_DIR}/spread_stats.json')


def _np(x):
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.zeros(0, dtype=np.float32)


def _gt_centers_np(data_gpu):
    return _np(data_gpu.get('gt_instance_centers_world'))


if __name__ == '__main__':
    main()
