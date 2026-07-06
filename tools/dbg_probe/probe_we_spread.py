"""
_we ep10 판정: visible-gate spread가 '만족(성공)'인지 'vacuous(재붕괴)'인지.
per matched pair: visible-gated spread(모델이 loss에 쓰는 그 값), raw spread,
visible gaussian 수(w>0.75), deficit(vs 0.5*GT half). GT size와의 corr.
"""
import os, sys, pickle
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dbg_out')
CFG = os.path.join(REPO, 'projects/configs/baselines/subset_scale_offset_we.py')
CKPT = os.path.join(REPO, 'work_dirs/subset_scale_offset_we/epoch_10_lss_only.pth')
N_BIG, N_NORMAL, BIG_LEN_M = 8, 6, 8.0


def pick_candidates(cfg):
    with open(cfg.data.train.ann_file, 'rb') as f:
        d = pickle.load(f)
    infos = d['infos'] if isinstance(d, dict) else d
    big, normal = [], []
    for i, info in enumerate(infos):
        boxes = info.get('gt_boxes', None)
        if boxes is None or len(boxes) == 0:
            continue
        horiz = np.asarray(boxes)[:, 3:5].max(axis=1)
        (big if (horiz > BIG_LEN_M).any() else normal).append(i)
    return big, normal


def main():
    cfg = Config.fromfile(CFG)
    big, normal = pick_candidates(cfg)
    dataset = build_dataset(cfg.data.train)
    n = len(dataset)
    cand = [(i, True) for i in big if i < n][:N_BIG * 3] + [(i, False) for i in normal if i < n][:N_NORMAL * 2]

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT, map_location='cpu')
    model = model.cuda().eval()
    model.set_train_iteration(7)
    for attr in ('debug_query_vis_every', 'debug_query_cam_gaussian_vis_every',
                 'debug_query_mixture3d_vis_every', 'debug_query_attn_softargmax_vis_every'):
        setattr(model, attr, 0)
    model.debug_query_cam_gaussian_vis_enabled = False
    model.query_head.debug_vis_every = 0
    print(f'[cfg] spread weight_mode={model.query_scale_spread_weight_mode} '
          f'norm={model.query_scale_spread_norm_mode} vis_thr={model.query_head.query_scale_spread_vis_threshold}',
          flush=True)

    qh = model.query_head
    orig = qh.apply_lifted_centers_to_outputs
    cap = {}
    def wrapped(*a, **k):
        out = orig(*a, **k)
        for key in ('mixture_centers_world_tqg3', 'mixture_weights_tqg', 'centers_world_tq3',
                    'query_offset_spread_tq3'):
            v = out.get(key)
            cap[key] = v.detach().float().cpu() if torch.is_tensor(v) else None
        cap['present'] = int(out.get('query_present_local_idx', 0) or 0)
        return out
    qh.apply_lifted_centers_to_outputs = wrapped

    rows = []
    done_big = done_norm = 0
    for idx, want_big in cand:
        if (want_big and done_big >= N_BIG) or ((not want_big) and done_norm >= N_NORMAL):
            continue
        try:
            item = dataset[idx]
        except Exception as e:
            print(f'[fetch fail] {idx}: {e}', flush=True); continue
        data = scatter(collate([item], samples_per_gpu=1), [torch.cuda.current_device()])[0]
        cap.clear()
        with torch.no_grad():
            model.forward_train(**data)
        spread_vis = cap.get('query_offset_spread_tq3')     # visible-gated (config대로)
        match = model._last_inst_match_result
        if spread_vis is None or not isinstance(match, dict):
            continue
        mq, gt_size_k3 = model._gather_matched_gt_sizes(
            match, data.get('gt_instance_ids'), data.get('gt_instance_sizes'),
            torch.device('cpu'))
        if mq is None:
            continue
        Q = spread_vis.shape[1]
        keep = (mq >= 0) & (mq < Q)
        mq, gt_size_k3 = mq[keep].cpu(), gt_size_k3[keep].cpu()
        t = int(cap['present'])
        cen = cap['mixture_centers_world_tqg3'][t]           # [Q,G,3]
        w = cap['mixture_weights_tqg'][t]                    # [Q,G]
        qc = cap['centers_world_tq3'][t]                     # [Q,3]
        sp_vis = spread_vis.mean(dim=0)                      # [Q,3] frame-mean (loss와 동일)
        for k in range(mq.numel()):
            q = int(mq[k])
            half = 0.5 * gt_size_k3[k]
            hxy = float(half[:2].max())
            off = (cen[q, :, :2] - qc[q, :2]).norm(dim=-1)   # [G] radius
            wq = w[q]
            wn = wq / wq.sum().clamp_min(1e-6)
            raw_spread = float((wn * off ** 2).sum().sqrt())
            vis = wq > 0.75                                  # union: peak=w
            nvis = int(vis.sum())
            r_vis = float(off[vis].max()) if nvis else 0.0
            act = float(sp_vis[q, :2].max())
            tgt = 0.5 * hxy
            rows.append(dict(idx=idx, gt_half=hxy, act_vis=act, tgt=tgt,
                             deficit=max(0.0, tgt - act), raw=raw_spread,
                             nvis=nvis, r_vis=r_vis,
                             w_max=float(wq.max()), w_med=float(wq.median())))
        done_big += int(want_big); done_norm += int(not want_big)
        print(f'[ok] idx={idx} big={want_big} pairs={len(rows)} '
              f'gpu={torch.cuda.memory_allocated()/2**30:.1f}G', flush=True)
        torch.cuda.empty_cache()
        if done_big >= N_BIG and done_norm >= N_NORMAL:
            break

    import json as J
    with open(os.path.join(OUT, 'we_spread_ep10.json'), 'w') as f:
        J.dump(rows, f)
    gt = np.array([r['gt_half'] for r in rows]); act = np.array([r['act_vis'] for r in rows])
    dfc = np.array([r['deficit'] for r in rows]); raw = np.array([r['raw'] for r in rows])
    nv = np.array([r['nvis'] for r in rows]); rv = np.array([r['r_vis'] for r in rows])
    from scipy.stats import spearmanr
    print(f'\n[n={len(rows)}]')
    print(f'corr(GT half, visible spread): pearson={np.corrcoef(gt, act)[0,1]:.3f} '
          f'spearman={spearmanr(gt, act).correlation:.3f}')
    print(f'deficit>0: {(dfc>1e-3).mean()*100:.1f}%  mean_deficit={dfc.mean():.3f}m')
    for lo, hi, name in [(0, 1.5, 'tiny'), (1.5, 2.6, 'car'), (2.6, 4.0, 'truck'), (4.0, 99, 'bus/trailer')]:
        m = (gt >= lo) & (gt < hi)
        if m.any():
            print(f'  {name:>11}: n={m.sum():>3} gt_half={gt[m].mean():.2f} vis_spread={act[m].mean():.2f} '
                  f'(tgt {0.5*gt[m].mean():.2f}) raw={raw[m].mean():.2f} nvis={nv[m].mean():.1f}/48 '
                  f'r_vis={rv[m].mean():.2f}m')

if __name__ == '__main__':
    main()
