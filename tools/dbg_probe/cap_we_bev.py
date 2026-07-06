"""_we ep10 mixture 캡처 (idx 3, 6 — 어제 old-run 캡처와 동일 샘플) + BEV 비교 플롯."""
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
IDXS = [3, 6]

cfg = Config.fromfile(CFG)
dataset = build_dataset(cfg.data.train)
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT, map_location='cpu')
model = model.cuda().eval()
model.set_train_iteration(7)
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
    for key in ('mixture_centers_world_tqg3', 'mixture_sigmas_world_tqg3',
                'mixture_weights_tqg', 'centers_world_tq3'):
        v = out.get(key)
        cap[key] = v.detach().float().cpu().numpy() if torch.is_tensor(v) else None
    cap['present'] = int(out.get('query_present_local_idx', 0) or 0)
    return out
qh.apply_lifted_centers_to_outputs = wrapped

def _np(x):
    if isinstance(x, (list, tuple)): x = x[0] if x else None
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.zeros(0)

for idx in IDXS:
    item = dataset[idx]
    data = scatter(collate([item], samples_per_gpu=1), [torch.cuda.current_device()])[0]
    cap.clear()
    with torch.no_grad():
        model.forward_train(**data)
    match = model._last_inst_match_result
    mq, gt_size_k3 = model._gather_matched_gt_sizes(
        match, data.get('gt_instance_ids'), data.get('gt_instance_sizes'), torch.device('cpu'))
    np.savez_compressed(
        os.path.join(OUT, f'cap_we10_idx{idx}.npz'),
        **{k: v for k, v in cap.items() if isinstance(v, np.ndarray)},
        present_idx=np.array(cap['present']),
        matched_query_idx=mq.cpu().numpy(),
        gt_half_xyz=(0.5 * gt_size_k3).cpu().numpy(),
        gt_centers=_np(data.get('gt_instance_centers_world')),
    )
    print(f'[ok] idx={idx} matched={mq.numel()}', flush=True)
    torch.cuda.empty_cache()

# ---- plot: old ep10 (ghost) vs we ep10 (flag-pole), bus & car ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

SAVE = os.path.join(REPO, 'work_dirs/subset_scale_offset_we/dbg_analysis')
os.makedirs(SAVE, exist_ok=True)
cases = [('idx6', 1, 'bus/trailer (gt_half=6.7m)'), ('idx3', 2, 'car (gt_half=2.4m)')]
runs = [('cap_ep10_', 'old run ep10 (raw metric)'), ('cap_we10_', '_we ep10 (visible gate)')]
fig, axes = plt.subplots(2, 2, figsize=(13, 12))
for row, (idx, k, name) in enumerate(cases):
    for col, (pref, rname) in enumerate(runs):
        z = np.load(f'{OUT}/{pref}{idx}.npz')
        t = int(z['present_idx'])
        # old 캡처는 matched_query_idx/gt_half_xyz 저장 형식 동일
        mqs = z['matched_query_idx']; halfs = z['gt_half_xyz']
        kk = min(k, len(mqs) - 1)
        kk = int(np.argmax(halfs[:, :2].max(1)))  # 샘플 내 최대 instance 기준으로 통일
        q = int(mqs[kk]); half = halfs[kk]
        cen = z['mixture_centers_world_tqg3'][t, q]
        sig = z['mixture_sigmas_world_tqg3'][t, q]
        w = z['mixture_weights_tqg'][t, q]
        qc = z['centers_world_tq3'][t, q]
        gtc = z['gt_centers'].reshape(-1, 3)
        gtc = gtc[np.isfinite(gtc).all(1)]
        gtc = gtc[np.abs(gtc).sum(1) > 1e-3]
        g = np.linalg.norm(gtc[:, :2] - qc[:2], axis=1).argmin()
        gx, gy = gtc[g, 0], gtc[g, 1]
        ax = axes[row, col]
        vis = w > 0.75
        ax.add_patch(Rectangle((gx - half[0], gy - half[1]), 2 * half[0], 2 * half[1],
                               fill=False, ec='k', lw=2))
        for gi in range(len(w)):
            ax.add_patch(Circle((cen[gi, 0], cen[gi, 1]), radius=max(float(sig[gi, :2].mean()), .06),
                                alpha=min(.8, .08 + .7 * float(w[gi])),
                                fc='#d62728' if vis[gi] else '#999999',
                                ec='#d62728' if vis[gi] else 'none', lw=1.0))
        ax.scatter(cen[:, 0], cen[:, 1], s=8 + 50 * w / max(w.max(), 1e-6), c=w,
                   cmap='viridis', vmin=0, vmax=1, zorder=3)
        ax.plot(qc[0], qc[1], 'b*', ms=14, zorder=4)
        off = np.linalg.norm(cen[:, :2] - qc[:2], axis=1)
        rv = off[vis].max() if vis.any() else 0.0
        ax.set_title(f'{name}\n{rname} | visible(w>0.75)={int(vis.sum())}/48, r_vis={rv:.1f}m')
        ax.set_aspect('equal'); ax.grid(alpha=.3)
        pad = 11
        ax.set_xlim(qc[0] - pad, qc[0] + pad); ax.set_ylim(qc[1] - pad, qc[1] + pad)
plt.suptitle('빨강 테두리=가시(w>0.75) gaussian, 회색=gate 아래, 검정 박스=GT, ★=query center')
plt.tight_layout()
p = os.path.join(SAVE, 'bev_flagpole_vs_ghost.png')
plt.savefig(p, dpi=110, bbox_inches='tight')
print('saved', p, flush=True)
