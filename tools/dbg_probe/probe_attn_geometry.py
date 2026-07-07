"""
Probe (ep10, buggy-attn baseline):
 (1) size_est = sigma_attn * depth / f  vs GT half-extent  (correlation baseline)
 (2) attention coverage vs GT cam masks: center-collapse or spread?
     - frac of attn mass on the GT-visible camera
     - inside_mass per-cam-correct vs legacy(cam-collided)  -> bug inflation
     - effective support (perplexity px) vs mask area
     - sigma_attn(px) vs sigma_mask(px)
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
    def wrapped(*a, **k):
        aw = k.get('query_attn_weights_tqnhw', a[0] if a else None)
        gt = k.get('gt_attn_targets', None)
        if torch.is_tensor(aw):
            cap['attn'] = aw.detach().float().cpu()
        if isinstance(gt, dict):
            cap['cam_mask'] = gt.get('gt_inst_cam_mask_tnnhw', None)
            cap['or_mask'] = gt.get('gt_inst_mask_tnhw', None)
            cap['valid'] = gt.get('gt_inst_valid_tn', None)
        return orig_loss(*a, **k)
    model._compute_query_attn_bbox_loss = wrapped

    qh = model.query_head
    orig_lift = qh.apply_lifted_centers_to_outputs
    def wrapped_lift(*a, **k):
        out = orig_lift(*a, **k)
        c = out.get('centers_world_tq3')
        cap['centers'] = c.detach().float().cpu() if torch.is_tensor(c) else None
        cap['present'] = int(out.get('query_present_local_idx', 0) or 0)
        return out
    qh.apply_lifted_centers_to_outputs = wrapped_lift

    rows = []  # per matched pair dict
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
        attn = cap.get('attn')            # [T,Q,N,H,W]
        cmask = cap.get('cam_mask')       # [T,Ninst,N,H,W] or None
        match = model._last_inst_match_result
        if attn is None or not isinstance(match, dict):
            print(f'[skip] idx={idx} attn={attn is not None}', flush=True); continue
        if torch.is_tensor(cmask):
            cmask = cmask.detach().to(torch.float32).cpu()
        else:
            print(f'[skip] idx={idx}: no per-cam gt mask', flush=True); continue

        # gather matched (query, inst, size)
        def _norm(x):
            if isinstance(x, (list, tuple)): x = x[0] if x else None
            if not torch.is_tensor(x): return None
            return x.cpu()
        gt_ids = _norm(data.get('gt_instance_ids'))
        gt_sizes = _norm(data.get('gt_instance_sizes'))
        if gt_ids is not None and gt_ids.dim() == 2: gt_ids = gt_ids[0]
        if gt_sizes is not None and gt_sizes.dim() == 3: gt_sizes = gt_sizes[0]
        mq = match.get('matched_query_idx'); mi = match.get('matched_inst_idx'); gid = match.get('gt_ids_n')
        if not (torch.is_tensor(mq) and torch.is_tensor(mi) and torch.is_tensor(gid)):
            continue
        mq, mi, gid = mq.cpu().long(), mi.cpu().long(), gid.cpu().long()
        centers = cap.get('centers')      # [T,Q,3]
        tpres = min(cap.get('present', 2), attn.shape[0] - 1)
        T_a, Q, NC, H, W = attn.shape
        tm = min(tpres, cmask.shape[0] - 1)

        # intrinsics: img_inputs_seq = [imgs, rots, trans, intrins, post_rots, post_trans, ...]
        f_eff = None
        try:
            ii = data.get('img_inputs_seq')
            intr = ii[3]; pr = ii[4]
            if torch.is_tensor(intr):
                if intr.dim() == 5: intr = intr[0]
                if pr.dim() == 5: pr = pr[0]
                eff = torch.matmul(pr[tpres].cpu().float(), intr[tpres].cpu().float())  # [N,3,3]
                f_eff = eff[:, 0, 0].abs()  # [N]
        except Exception as e:
            print(f'[warn] intrinsics parse failed: {e}', flush=True)

        uu = torch.arange(W).float().view(1, W)
        vv = torch.arange(H).float().view(H, 1)
        for k in range(mq.numel()):
            q, i_inst = int(mq[k]), int(mi[k])
            if q < 0 or q >= Q or i_inst < 0 or i_inst >= cmask.shape[1]:
                continue
            # size lookup by id
            sz = None
            if gt_ids is not None and gt_sizes is not None:
                hit = (gt_ids == gid[i_inst]).nonzero()
                if hit.numel():
                    sz = float(gt_sizes[int(hit[0]), :2].max()) * 0.5
            if sz is None or sz <= 0.05:
                continue
            A = attn[tpres, q]                          # [N,H,W]
            M = cmask[tm, i_inst]                       # [N,H,W]
            m_area = M.sum(dim=(1, 2))                  # [N]
            if float(m_area.sum()) <= 0:
                continue
            tot = float(A.sum().clamp_min(1e-9))
            P = A / tot                                 # normalized over cams+pixels
            # (2a) mass on GT-visible cams / correct inside mass vs legacy
            vis_cams = m_area > 0
            mass_on_vis = float(P[vis_cams].sum())
            inside_correct = float((P * M).sum())
            P_or = P.sum(dim=0)                         # cam-collided pred
            M_or = M.amax(dim=0)                        # cam-collided mask
            inside_legacy = float((P_or * M_or).sum())
            # dominant GT camera
            c = int(m_area.argmax())
            Pc = P[c] / max(float(P[c].sum()), 1e-9)
            # (2b) spread vs mask spread on cam c
            mu_u = float((Pc * uu).sum()); mu_v = float((Pc * vv).sum())
            s_u = float(((Pc * (uu - mu_u) ** 2).sum()) ** .5)
            s_v = float(((Pc * (vv - mu_v) ** 2).sum()) ** .5)
            mc = M[c] / max(float(M[c].sum()), 1e-9)
            gu = float((mc * uu).sum()); gv = float((mc * vv).sum())
            g_u = float(((mc * (uu - gu) ** 2).sum()) ** .5)
            g_v = float(((mc * (vv - gv) ** 2).sum()) ** .5)
            ent = float(-(Pc.clamp_min(1e-9) * Pc.clamp_min(1e-9).log()).sum())
            eff_px = float(np.exp(ent))
            # (1) size_est
            d_q = None
            if torch.is_tensor(centers):
                d_q = float(centers[tpres, q, :2].norm())
            f_c = float(f_eff[c]) if f_eff is not None else 800.0
            stride = 1600.0 / W
            size_est = (3 ** .5) * max(s_u, 1e-3) * stride * (d_q or 30.0) / max(f_c, 1.0)
            rows.append(dict(idx=idx, gt_half=sz, size_est=size_est,
                             s_u=s_u, s_v=s_v, g_u=g_u, g_v=g_v,
                             eff_px=eff_px, mask_px=float(m_area[c]),
                             mass_on_vis=mass_on_vis, inside_correct=inside_correct,
                             inside_legacy=inside_legacy, d_q=d_q or -1, f_c=f_c))
        done_big += int(want_big); done_norm += int(not want_big)
        print(f'[ok] idx={idx} big={want_big} pairs={len(rows)} gpu={torch.cuda.memory_allocated()/2**30:.1f}G', flush=True)
        torch.cuda.empty_cache()
        if done_big >= N_BIG and done_norm >= N_NORMAL:
            break

    import json
    with open(os.path.join(OUT, 'attn_geom_ep10.json'), 'w') as f:
        json.dump(rows, f)
    # ---- analysis ----
    r = rows
    gt = np.array([x['gt_half'] for x in r]); se = np.array([x['size_est'] for x in r])
    print(f'\n[n={len(r)} matched pairs]', flush=True)
    lg, ls = np.log(gt), np.log(np.maximum(se, 1e-3))
    pear = np.corrcoef(lg, ls)[0, 1]
    from scipy.stats import spearmanr
    rho = spearmanr(lg, ls).correlation
    print(f'(1) size_est vs GT half: pearson(log)={pear:.3f} spearman={rho:.3f}')
    for lo, hi, name in [(0, 1.5, 'tiny'), (1.5, 2.6, 'car'), (2.6, 4.0, 'truck'), (4.0, 99, 'bus/trailer')]:
        m = (gt >= lo) & (gt < hi)
        if m.any():
            print(f"   {name:>11}: n={m.sum():>3} gt_half={gt[m].mean():.2f}m size_est={se[m].mean():.2f}m "
                  f"s_u={np.mean([x['s_u'] for x, mm in zip(r, m) if mm]):.2f}px(grid) "
                  f"mask_sig_u={np.mean([x['g_u'] for x, mm in zip(r, m) if mm]):.2f}px(grid)")
    mov = np.array([x['mass_on_vis'] for x in r]); ic = np.array([x['inside_correct'] for x in r])
    il = np.array([x['inside_legacy'] for x in r])
    cover = np.array([x['eff_px'] for x in r]) / np.maximum(np.array([x['mask_px'] for x in r]), 1.0)
    sr = np.array([x['s_u'] for x in r]) / np.maximum(np.array([x['g_u'] for x in r]), 1e-3)
    print(f'(2) attn mass on GT-visible cam: mean={mov.mean():.3f} median={np.median(mov):.3f}')
    print(f'    inside_mass correct(per-cam)={ic.mean():.3f} vs legacy(collided)={il.mean():.3f} -> bug inflation={il.mean()-ic.mean():.3f}')
    print(f'    coverage: eff_support/mask_area mean={cover.mean():.3f} median={np.median(cover):.3f} (1.0=mask 전체 균등 커버)')
    print(f'    spread ratio sigma_attn/sigma_mask: mean={sr.mean():.3f} median={np.median(sr):.3f} (1.0=마스크만큼 퍼짐)')

if __name__ == '__main__':
    main()
