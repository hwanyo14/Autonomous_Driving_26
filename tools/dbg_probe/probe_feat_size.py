"""
Probe: is GT size linearly decodable from matched query features (ep10)?
Captures query_feat_tqd (present frame) for matched queries + GT half extent.
No repo modification; vis gates disabled via cfg override.
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
    print(f'[dataset] len={n} candidates={len(cand)}', flush=True)

    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CKPT, map_location='cpu')
    model = model.cuda().eval()
    model.set_train_iteration(7)
    # vis_every <= 0 disables every training-vis path (verified in utils_visualization/query_head)
    for attr in ('debug_query_vis_every', 'debug_query_cam_gaussian_vis_every',
                 'debug_query_mixture3d_vis_every', 'debug_query_attn_softargmax_vis_every'):
        setattr(model, attr, 0)
    model.debug_query_cam_gaussian_vis_enabled = False
    model.query_head.debug_vis_every = 0

    qh = model.query_head
    orig = qh.apply_lifted_centers_to_outputs
    cap = {}
    def wrapped(*a, **k):
        out = orig(*a, **k)
        f = out.get('query_feat_tqd', None)
        cap['feat'] = f.detach().float().cpu().numpy() if torch.is_tensor(f) else None
        cap['present'] = int(out.get('query_present_local_idx', 0) or 0)
        return out
    qh.apply_lifted_centers_to_outputs = wrapped

    feats, halves, sample_ids = [], [], []
    done_big = done_norm = 0
    for idx, want_big in cand:
        if (want_big and done_big >= N_BIG) or ((not want_big) and done_norm >= N_NORMAL):
            continue
        try:
            item = dataset[idx]
        except Exception as e:
            print(f'[fetch fail] {idx}: {e}', flush=True); continue
        data = scatter(collate([item], samples_per_gpu=1), [torch.cuda.current_device()])[0]
        with torch.no_grad():
            model.forward_train(**data)
        spread = model._last_query_offset_spread
        match = model._last_inst_match_result
        mq, gt_size_k3 = model._gather_matched_gt_sizes(
            match, data.get('gt_instance_ids'), data.get('gt_instance_sizes'), spread.device)
        if mq is None or cap.get('feat') is None:
            print(f'[skip] {idx}: no match/feat', flush=True); continue
        q_count = spread.shape[1]
        keep = (mq >= 0) & (mq < q_count)
        mq, gt_size_k3 = mq[keep].cpu().numpy(), gt_size_k3[keep].cpu().numpy()
        f_qd = cap['feat'][cap['present']]           # [Q,D]
        feats.append(f_qd[mq])
        halves.append(0.5 * gt_size_k3)
        sample_ids.extend([idx] * len(mq))
        done_big += int(want_big); done_norm += int(not want_big)
        mx = (0.5 * gt_size_k3)[:, :2].max(axis=1)
        print(f'[ok] idx={idx} big={want_big} matched={len(mq)} half_xy max={mx.max():.1f} '
              f'gpu_mem={torch.cuda.memory_allocated()/2**30:.1f}G', flush=True)
        torch.cuda.empty_cache()
        if done_big >= N_BIG and done_norm >= N_NORMAL:
            break

    X = np.concatenate(feats); H = np.concatenate(halves); S = np.array(sample_ids)
    np.savez_compressed(os.path.join(OUT, 'feat_probe_ep10.npz'), X=X, H=H, S=S)
    print(f'[saved] {X.shape} pairs from {len(set(sample_ids))} samples', flush=True)

    # ---- ridge probe with group (sample) CV ----
    y = np.log(H[:, :2].max(axis=1))                 # log half_xy
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
    groups = np.unique(S)
    rng = np.random.RandomState(0); rng.shuffle(groups)
    folds = np.array_split(groups, min(5, len(groups)))
    preds = np.zeros_like(y)
    for fold in folds:
        te = np.isin(S, fold); tr = ~te
        A, b = Xz[tr], y[tr]
        lam = 10.0
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
        preds[te] = Xz[te] @ w
    ss_res = ((y - preds) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    rho = np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(preds)))[0, 1]
    big_true = H[:, :2].max(axis=1) > 2.6
    thr = np.quantile(preds, 1 - big_true.mean())
    acc = ((preds > thr) == big_true).mean()
    print(f'[probe] n={len(y)} groupCV R2={r2:.3f} spearman={rho:.3f} '
          f'big-vs-small acc={acc:.3f} (base={max(big_true.mean(),1-big_true.mean()):.3f})', flush=True)


if __name__ == '__main__':
    main()
