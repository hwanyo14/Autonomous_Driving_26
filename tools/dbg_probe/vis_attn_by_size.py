"""
cover ep15: 크기 bin별(small/mid/large/x-large) 시각화 —
왼쪽: 카메라 이미지 + attention heatmap + GT cam mask 윤곽
오른쪽: BEV gaussian mixture(가시=빨강) + GT AABB
저장: work_dirs/subset_attn_cover/dbg_analysis/attn_vis_by_size.png
"""
import os, sys, pickle
REPO = '/NHNHOME/WORKSPACE/0526040009_A/hwanhee/Autonomous_Driving_26_ksh_0630'
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import importlib
importlib.import_module('projects.occ_plugin')
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

CFG = os.path.join(REPO, 'work_dirs/subset_attn_cover/subset_attn_cover.py')
CKPT = os.path.join(REPO, 'work_dirs/subset_attn_cover/epoch_15_lss_only.pth')
SAVE = os.path.join(REPO, 'work_dirs/subset_attn_cover/dbg_analysis')
os.makedirs(SAVE, exist_ok=True)
BINS = [(0, 1.5, 'small (bike급)'), (1.5, 2.6, 'mid (car급)'),
        (2.6, 4.0, 'large (truck급)'), (4.0, 99, 'x-large (bus/trailer급)')]
MEAN = np.array([123.675, 116.28, 103.53]); STD = np.array([58.395, 57.12, 57.375])


def pick_candidates(cfg):
    with open(cfg.data.train.ann_file, 'rb') as f:
        d = pickle.load(f)
    infos = d['infos'] if isinstance(d, dict) else d
    out = []
    for i, info in enumerate(infos):
        boxes = info.get('gt_boxes', None)
        if boxes is None or len(boxes) == 0:
            continue
        horiz = np.asarray(boxes)[:, 3:5].max(axis=1)
        if (horiz > 8.0).any():
            out.append(i)
    return out


def main():
    cfg = Config.fromfile(CFG)
    keys = list(cfg.data.train.pipeline[-1]['keys'])
    if 'gt_instance_sizes' not in keys:
        cfg.data.train.pipeline[-1]['keys'] = keys + ['gt_instance_sizes']
    cand = pick_candidates(cfg)
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

    cap = {}
    orig_loss = model._compute_query_attn_bbox_loss
    def wrap_loss(*a, **k):
        aw = k.get('query_attn_weights_tqnhw', a[0] if a else None)
        gt = k.get('gt_attn_targets', None)
        if torch.is_tensor(aw): cap['attn'] = aw.detach().float().cpu()
        if isinstance(gt, dict):
            cm = gt.get('gt_inst_cam_mask_tnnhw')
            cap['cmask'] = cm.detach().float().cpu() if torch.is_tensor(cm) else None
        return orig_loss(*a, **k)
    model._compute_query_attn_bbox_loss = wrap_loss
    qh = model.query_head
    orig_lift = qh.apply_lifted_centers_to_outputs
    def wrap_lift(*a, **k):
        out = orig_lift(*a, **k)
        for key in ('mixture_centers_world_tqg3', 'mixture_sigmas_world_tqg3',
                    'mixture_weights_tqg', 'centers_world_tq3'):
            v = out.get(key)
            cap[key] = v.detach().float().cpu() if torch.is_tensor(v) else None
        cap['present'] = int(out.get('query_present_local_idx', 0) or 0)
        return out
    qh.apply_lifted_centers_to_outputs = wrap_lift

    picked = {}  # bin_name -> dict of everything needed for plotting
    for idx in cand[:30]:
        if len(picked) >= len(BINS):
            break
        try:
            item = dataset[idx]
        except Exception:
            continue
        data = scatter(collate([item], samples_per_gpu=1), [torch.cuda.current_device()])[0]
        cap.clear()
        with torch.no_grad():
            model.forward_train(**data)
        attn = cap.get('attn'); cmask = cap.get('cmask')
        match = model._last_inst_match_result
        if attn is None or cmask is None or not isinstance(match, dict):
            continue
        def _norm(x):
            if isinstance(x, (list, tuple)): x = x[0] if x else None
            return x.cpu() if torch.is_tensor(x) else None
        gt_ids = _norm(data.get('gt_instance_ids')); gt_sizes = _norm(data.get('gt_instance_sizes'))
        gt_centers = _norm(data.get('gt_instance_centers_world'))
        if gt_ids is not None and gt_ids.dim() == 2: gt_ids = gt_ids[0]
        if gt_sizes is not None and gt_sizes.dim() == 3: gt_sizes = gt_sizes[0]
        if gt_centers is not None and gt_centers.dim() == 4: gt_centers = gt_centers[0]
        mq = match.get('matched_query_idx'); mi = match.get('matched_inst_idx'); gid = match.get('gt_ids_n')
        if not (torch.is_tensor(mq) and torch.is_tensor(mi)): continue
        mq, mi, gid = mq.cpu(), mi.cpu(), gid.cpu()
        t = min(cap['present'], attn.shape[0] - 1, cmask.shape[0] - 1)
        imgs = data['img_inputs_seq'][0]
        if imgs.dim() == 6: imgs = imgs[0]          # [T,N,3,H,W]
        img_t = imgs[min(t, imgs.shape[0] - 1)].cpu()  # [N,3,H,W]
        for k in range(mq.numel()):
            q, ii = int(mq[k]), int(mi[k])
            if ii >= cmask.shape[1] or q >= attn.shape[1]: continue
            sz = None
            if gt_ids is not None and gt_sizes is not None:
                hit = (gt_ids == gid[ii]).nonzero()
                if hit.numel(): sz = float(gt_sizes[int(hit[0]), :2].max()) * 0.5
            if sz is None: continue
            bname = next((n for lo, hi, n in BINS if lo <= sz < hi), None)
            if bname is None or bname in picked: continue
            M = cmask[t, ii]                        # [N,Ha,Wa]
            m_area = M.sum(dim=(1, 2))
            if float(m_area.max()) < 15: continue   # 잘 보이는 것만
            c = int(m_area.argmax())
            A = attn[t, q, c]                       # [Ha,Wa]
            if float(A.sum()) / float(attn[t, q].sum().clamp_min(1e-9)) < 0.2: continue
            picked[bname] = dict(
                idx=idx, gt_half=sz, cam=c,
                img=img_t[c].numpy(), attn=A.numpy(), mask=M[c].numpy(),
                cen=cap['mixture_centers_world_tqg3'][t, q].numpy(),
                sig=cap['mixture_sigmas_world_tqg3'][t, q].numpy(),
                w=cap['mixture_weights_tqg'][t, q].numpy(),
                qc=cap['centers_world_tq3'][t, q].numpy(),
                gt_centers=gt_centers.numpy() if gt_centers is not None else None,
                gt_size_row=(float(gt_sizes[int(hit[0]), 0]) * 0.5, float(gt_sizes[int(hit[0]), 1]) * 0.5),
                t=t,
            )
            print(f'[pick] {bname}: idx={idx} gt_half={sz:.1f}m cam={c}', flush=True)
        print(f'[scan] idx={idx} picked={list(picked)}', flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    n = len(picked)
    fig, axes = plt.subplots(n, 2, figsize=(17, 5.2 * n))
    if n == 1: axes = axes[None, :]
    for r, (lo, hi, bname) in enumerate([b for b in BINS if b[2] in picked]):
        d = picked[bname]
        axL, axR = axes[r]
        img = np.clip(d['img'].transpose(1, 2, 0) * STD + MEAN, 0, 255).astype(np.uint8)
        H, W = img.shape[:2]
        axL.imshow(img)
        A = torch.from_numpy(d['attn'])[None, None]
        Aup = F.interpolate(A, size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy()
        axL.imshow(Aup, cmap='jet', alpha=0.45 * Aup / max(Aup.max(), 1e-9))
        Mup = F.interpolate(torch.from_numpy(d['mask'])[None, None], size=(H, W), mode='nearest')[0, 0].numpy()
        axL.contour(Mup, levels=[0.5], colors='lime', linewidths=2)
        axL.set_title(f"{bname} | GT half={d['gt_half']:.1f}m | attention(heat) + GT mask(초록) | cam{d['cam']}")
        axL.axis('off')
        # BEV
        cen, sig, w, qc = d['cen'], d['sig'], d['w'], d['qc']
        vis = w > 0.75
        gx = gy = None
        if d['gt_centers'] is not None:
            gtc = d['gt_centers'].reshape(-1, 3)
            gtc = gtc[np.isfinite(gtc).all(1)]
            gtc = gtc[np.abs(gtc).sum(1) > 1e-3]
            if len(gtc):
                g = np.linalg.norm(gtc[:, :2] - qc[:2], axis=1).argmin()
                gx, gy = gtc[g, 0], gtc[g, 1]
        hx, hy = d['gt_size_row']
        if gx is not None:
            axR.add_patch(Rectangle((gx - hx, gy - hy), 2 * hx, 2 * hy, fill=False, ec='k', lw=2))
        for gi in range(len(w)):
            axR.add_patch(Circle((cen[gi, 0], cen[gi, 1]), radius=max(float(sig[gi, :2].mean()), .06),
                                 alpha=min(.8, .1 + .7 * float(w[gi])),
                                 fc='#d62728' if vis[gi] else '#999999',
                                 ec='#d62728' if vis[gi] else 'none'))
        axR.scatter(cen[:, 0], cen[:, 1], s=8 + 40 * w / max(w.max(), 1e-6), c=w, cmap='viridis', vmin=0, vmax=1, zorder=3)
        axR.plot(qc[0], qc[1], 'b*', ms=14, zorder=4)
        pad = max(hx, hy) * 2 + 4
        cx0, cy0 = (gx, gy) if gx is not None else (qc[0], qc[1])
        axR.set_xlim(cx0 - pad, cx0 + pad); axR.set_ylim(cy0 - pad, cy0 + pad)
        axR.set_aspect('equal'); axR.grid(alpha=.3)
        axR.set_title(f"BEV 예측: gaussian(빨강=가시 w>0.75) vs GT박스 | visible={int(vis.sum())}/48")
    plt.tight_layout()
    p = os.path.join(SAVE, 'attn_vis_by_size.png')
    plt.savefig(p, dpi=100, bbox_inches='tight')
    print('saved', p, flush=True)


if __name__ == '__main__':
    main()
