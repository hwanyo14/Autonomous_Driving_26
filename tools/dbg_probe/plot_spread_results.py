"""Plot dbg_offset_spread results: size-bin stats + BEV vis of biggest instances."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dbg_out')

with open(os.path.join(OUT, 'spread_stats.json')) as f:
    res = json.load(f)

TAGS = ['ep1', 'ep10']
BINS = [(0, 2.5, 'small (<5m)'), (2.5, 4.0, 'mid (5-8m)'), (4.0, 99, 'large (>8m)')]
# bins on gt half max-xy extent

def collect(tag):
    rows = []
    for rec in res[tag]:
        half = np.array(rec['gt_half_xyz'])  # [K,3]
        act = np.array(rec['act_xy'])
        tgt = np.array(rec['tgt_xy'])
        hxy = half[:, :2].max(axis=1)
        for h, a, t in zip(hxy, act, tgt):
            rows.append((h, a, t))
    return np.array(rows)  # [N,3] half_xy, act, tgt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
# --- panel 1: scatter act vs half extent
colors = {'ep1': '#9ecae1', 'ep10': '#d62728'}
for tag in TAGS:
    r = collect(tag)
    axes[0].scatter(r[:, 0], r[:, 1], s=18, alpha=.7, label=f'{tag} (n={len(r)})', color=colors[tag])
hh = np.linspace(0, 9, 50)
axes[0].plot(hh, 0.5 * hh, 'k--', lw=1, label='target = 0.5×half-extent')
axes[0].set_xlabel('GT half extent xy [m]'); axes[0].set_ylabel('offset-only spread xy [m]')
axes[0].set_title('matched query offset spread vs GT size'); axes[0].legend(); axes[0].grid(alpha=.3)

# --- panel 2: per-bin mean ratio act/tgt
w = 0.35
for j, tag in enumerate(TAGS):
    r = collect(tag)
    means, stds, ns = [], [], []
    for lo, hi, _ in BINS:
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        ratio = r[m, 1] / np.maximum(r[m, 2], 1e-6)  # act / tgt
        means.append(ratio.mean() if m.any() else 0)
        stds.append(ratio.std() if m.any() else 0)
        ns.append(int(m.sum()))
    x = np.arange(len(BINS)) + (j - .5) * w
    axes[1].bar(x, means, width=w, yerr=stds, capsize=3, label=tag, color=colors[tag])
    for xi, mi, ni in zip(x, means, ns):
        axes[1].text(xi, mi + .03, f'n={ni}', ha='center', fontsize=8)
axes[1].axhline(1.0, color='k', ls='--', lw=1)
axes[1].set_xticks(range(len(BINS))); axes[1].set_xticklabels([b[2] for b in BINS])
axes[1].set_ylabel('spread / target (≥1 = target met)')
axes[1].set_title('spread-target attainment by GT size bin'); axes[1].legend(); axes[1].grid(alpha=.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'spread_stats.png'), dpi=115)
print('saved spread_stats.png')

# --- BEV vis: biggest instance across samples, ep1 vs ep10 side by side
best = None
for rec in res['ep10']:
    half = np.array(rec['gt_half_xyz'])
    if len(half) == 0:
        continue
    k = half[:, :2].max(axis=1).argmax()
    v = half[k, :2].max()
    if best is None or v > best[2]:
        best = (rec['idx'], k, v)
print('biggest matched instance: sample idx', best[0], 'half_xy', best[2])

idx, k_big, _ = best
fig, axes = plt.subplots(1, len(TAGS), figsize=(13, 6.2), sharex=True, sharey=True)
for ax, tag in zip(axes, TAGS):
    z = np.load(os.path.join(OUT, f'cap_{tag}_idx{idx}.npz'))
    t = int(z['present_idx'])
    cen = z['mixture_centers_world_tqg3'][t]      # [Q,G,3]
    sig = z['mixture_sigmas_world_tqg3'][t]       # [Q,G,3]
    wgt = z['mixture_weights_tqg'][t]             # [Q,G]
    qc = z['centers_world_tq3'][t]                # [Q,3]
    mq = z['matched_query_idx']
    half = z['gt_half_xyz']
    spread = z['spread_xyz']
    gt_c = z['gt_centers']                        # [T?,N,3] or [N,3]
    gt_ids = z['gt_ids']
    gt_sz = z['gt_sizes']
    # this run: find matched pair with biggest gt
    kk = half[:, :2].max(axis=1).argmax()
    q = int(mq[kk])
    c = cen[q]; s = sig[q]; w_ = wgt[q]
    w_ = w_ / max(w_.max(), 1e-6)
    qcx, qcy = qc[q, 0], qc[q, 1]
    # GT rect: need center of that instance — approximate: nearest gt center to query center
    if gt_c.ndim == 3:
        gtc2 = gt_c.reshape(-1, gt_c.shape[-1])
    else:
        gtc2 = gt_c
    gtc2 = gtc2[np.isfinite(gtc2).all(axis=1)]
    d = np.linalg.norm(gtc2[:, :2] - np.array([qcx, qcy]), axis=1)
    g = d.argmin()
    gx, gy = gtc2[g, 0], gtc2[g, 1]
    hx, hy = half[kk, 0], half[kk, 1]
    ax.add_patch(Rectangle((gx - hx, gy - hy), 2 * hx, 2 * hy, fill=False, ec='k', lw=1.6, label='GT AABB'))
    for gi in range(c.shape[0]):
        ax.add_patch(Circle((c[gi, 0], c[gi, 1]), radius=max(s[gi, :2].mean(), .05),
                            alpha=.28 * w_[gi] + .04, fc='#d62728', ec='none'))
    ax.scatter(c[:, 0], c[:, 1], s=8 + 40 * w_, c='#d62728', alpha=.85, label='gaussian centers')
    ax.plot(qcx, qcy, 'b*', ms=14, label='query center')
    ax.set_title(f'{tag} | offset spread xy={spread[kk, :2].max():.2f}m (target {0.5*max(hx,hy):.2f}m)')
    ax.set_aspect('equal'); ax.grid(alpha=.3)
    pad = max(hx, hy) * 1.8 + 2
    ax.set_xlim(gx - pad, gx + pad); ax.set_ylim(gy - pad, gy + pad)
    ax.legend(fontsize=8, loc='upper right')
plt.suptitle(f'largest matched instance (sample {idx}) — gaussian mixture BEV, circle=σ̄xy, alpha∝weight')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'bev_biggest.png'), dpi=115)
print('saved bev_biggest.png')
