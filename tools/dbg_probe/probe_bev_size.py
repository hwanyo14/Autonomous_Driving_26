"""
Probe: is GT size decodable from (a) LSS BEV feature windows around GT centers,
(b) existing per-instance pooled context features (_last_gt_instance_bev_feat_cache)?
ep10 ckpt, same 14 samples as the camera-feature probe (which scored R2=0.06).
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
CFG = os.path.join(REPO, 'work_dirs/subset_scale_offset/subset_scale_offset.py')
CKPT = os.path.join(REPO, 'work_dirs/subset_scale_offset/epoch_10_lss_only.pth')
N_BIG, N_NORMAL, BIG_LEN_M = 8, 6, 8.0
PC_MIN, PC_MAX = -51.2, 51.2


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


def pool_window(bev_chw, cx, cy, half_m):
    """mean+max pool a (2*half_m)^2 window at world (cx,cy); returns [2C] or None."""
    C, H, W = bev_chw.shape
    cell = (PC_MAX - PC_MIN) / float(W)  # assume square
    r = max(1, int(round(half_m / cell)))
    ix = int((cx - PC_MIN) / cell)
    iy = int((cy - PC_MIN) / cell)
    out = []
    for (a, b) in ((ix, iy), (iy, ix)):  # both orientations (transpose ambiguity)
        x0, x1 = max(0, a - r), min(W, a + r + 1)
        y0, y1 = max(0, b - r), min(H, b + r + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        win = bev_chw[:, y0:y1, x0:x1]
        out.append(torch.cat([win.mean(dim=(1, 2)), win.amax(dim=(1, 2))]))
    return torch.stack(out)  # [2 orientations, 2C]


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

    # hook LSS output
    ivt = model.img_view_transformer
    orig_fwd = ivt.forward
    cap = {}
    def wrapped(*a, **k):
        out = orig_fwd(*a, **k)
        x = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(x):
            cap['bev'] = x.detach()
        return out
    ivt.forward = wrapped

    rows = {'w8': [], 'w16': [], 'instpool': []}
    sizes_all = {'w8': [], 'w16': [], 'instpool': []}
    sids = {'w8': [], 'w16': [], 'instpool': []}
    done_big = done_norm = 0
    for idx, want_big in cand:
        if (want_big and done_big >= N_BIG) or ((not want_big) and done_norm >= N_NORMAL):
            continue
        try:
            item = dataset[idx]
        except Exception as e:
            print(f'[fetch fail] {idx}: {e}', flush=True); continue
        data = scatter(collate([item], samples_per_gpu=1), [torch.cuda.current_device()])[0]
        cap.pop('bev', None)
        with torch.no_grad():
            model.forward_train(**data)

        def _np(x):
            if isinstance(x, (list, tuple)):
                x = x[0] if x else None
            return x.detach().float().cpu() if torch.is_tensor(x) else None
        gt_sizes = _np(data.get('gt_instance_sizes'))          # [N,3]
        gt_centers = _np(data.get('gt_instance_centers_world'))  # [T,N,3] (batch squeezed) or [1,T,N,3]
        gt_valid = _np(data.get('gt_instance_centers_valid'))
        gt_ids = _np(data.get('gt_instance_ids'))
        if gt_sizes is None or gt_centers is None:
            print(f'[skip] {idx}: no gt sizes/centers', flush=True); continue
        if gt_sizes.dim() == 3: gt_sizes = gt_sizes[0]
        if gt_centers.dim() == 4: gt_centers = gt_centers[0]
        if gt_valid is not None and gt_valid.dim() == 3: gt_valid = gt_valid[0]
        if gt_ids is not None and gt_ids.dim() == 2: gt_ids = gt_ids[0]
        t_pres = min(2, gt_centers.shape[0] - 1)
        cen = gt_centers[t_pres]                                # [N,3]
        val = gt_valid[t_pres].bool() if gt_valid is not None else torch.isfinite(cen).all(-1)

        bev = cap.get('bev', None)
        if bev is None:
            print(f'[skip] {idx}: no bev captured', flush=True); continue
        if idx == cand[0][0] or not rows['w16']:
            print(f'[bev] shape={tuple(bev.shape)}', flush=True)
        b = bev
        if b.dim() == 5:  # [TB,C,H,W,Z] (observed: 3,96,128,128,10) -> mean over Z(last)
            b = b.mean(dim=-1)
        # assume [T*B, C, H, W], B=1 -> take frame index t_pres (clamped)
        tb = b.shape[0]
        b_pres = b[min(t_pres, tb - 1)].float().cpu()           # [C,H,W]

        n_inst = gt_sizes.shape[0]
        for k in range(n_inst):
            if not bool(val[k]):
                continue
            sz = float(gt_sizes[k, :2].max()) * 0.5
            if sz <= 0.05:
                continue
            for half_m, key in ((4.0, 'w8'), (8.0, 'w16')):
                pw = pool_window(b_pres, float(cen[k, 0]), float(cen[k, 1]), half_m)
                if pw is not None:
                    rows[key].append(pw.numpy())               # [2, 2C]
                    sizes_all[key].append(sz)
                    sids[key].append(idx)
        # (b) existing instance-pooled cache
        cache = model._last_gt_instance_bev_feat_cache
        if isinstance(cache, dict) and torch.is_tensor(cache.get('feat_tnd')):
            f_tnd = cache['feat_tnd'].float().cpu()             # [T,Nc,D]
            v_tn = cache.get('valid_tn')
            v_tn = v_tn.cpu() if torch.is_tensor(v_tn) else v_tn
            ids_c = cache.get('ids_n')
            ids_c = ids_c.cpu() if torch.is_tensor(ids_c) else ids_c
            tp = min(t_pres, f_tnd.shape[0] - 1)
            for j in range(f_tnd.shape[1]):
                if v_tn is not None and not bool(v_tn[tp, j]):
                    continue
                # map cache id -> gt size row
                if ids_c is not None and gt_ids is not None:
                    m = (gt_ids == ids_c[j].long()).nonzero()
                    if m.numel() == 0: continue
                    k = int(m[0])
                else:
                    k = j if j < n_inst else -1
                if k < 0: continue
                sz = float(gt_sizes[k, :2].max()) * 0.5
                if sz <= 0.05: continue
                rows['instpool'].append(f_tnd[tp, j].numpy())
                sizes_all['instpool'].append(sz)
                sids['instpool'].append(idx)
        done_big += int(want_big); done_norm += int(not want_big)
        print(f'[ok] idx={idx} big={want_big} pairs w16={len(rows["w16"])} inst={len(rows["instpool"])} '
              f'gpu={torch.cuda.memory_allocated()/2**30:.1f}G', flush=True)
        torch.cuda.empty_cache()
        if done_big >= N_BIG and done_norm >= N_NORMAL:
            break

    np.savez_compressed(os.path.join(OUT, 'bev_probe_ep10.npz'),
                        **{f'{k}_X': np.array(rows[k]) for k in rows},
                        **{f'{k}_y': np.array(sizes_all[k]) for k in rows},
                        **{f'{k}_s': np.array(sids[k]) for k in rows})

    # ---- ridge probes ----
    def probe(X, y, S, tag):
        y = np.log(y)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
        groups = np.unique(S); rng = np.random.RandomState(0); rng.shuffle(groups)
        folds = np.array_split(groups, min(5, len(groups)))
        best = None
        for lam in (10., 100., 1000., 5000.):
            preds = np.zeros_like(y)
            for f in folds:
                te = np.isin(S, f); tr = ~te
                A, b = Xz[tr], y[tr]; mu = b.mean()
                w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (b - mu))
                preds[te] = Xz[te] @ w + mu
            r2 = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if best is None or r2 > best[0]:
                best = (r2, lam, preds)
        r2, lam, preds = best
        big = np.exp(y) > 2.6
        if 0 < big.sum() < len(big):
            thr = np.quantile(preds, 1 - big.mean())
            acc = ((preds > thr) == big).mean()
            base = max(big.mean(), 1 - big.mean())
        else:
            acc = base = float('nan')
        print(f'[probe:{tag}] n={len(y)} D={X.shape[1]} best_lam={lam:.0f} groupCV_R2={r2:.3f} '
              f'big-acc={acc:.3f} (base={base:.3f})', flush=True)

    for key in ('w8', 'w16'):
        X = np.array(rows[key])
        if len(X) == 0: continue
        y = np.array(sizes_all[key]); S = np.array(sids[key])
        for o, oname in ((0, 'orient0'), (1, 'orient1')):
            probe(X[:, o, :], y, S, f'{key}-{oname}')
    if rows['instpool']:
        probe(np.array(rows['instpool']), np.array(sizes_all['instpool']),
              np.array(sids['instpool']), 'instpool')


if __name__ == '__main__':
    main()
