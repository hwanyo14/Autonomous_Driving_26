"""res704 전처리 시각화 — loading_bevdet.py sample_augmentation/img_transform_core 로직 그대로."""
import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = '/home/hwanhee/Autonomous_Driving_26_filter_ablation_res'
OUT = os.path.join(ROOT, 'data_vis', 'res704_preprocess')
os.makedirs(OUT, exist_ok=True)

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
SRC = (900, 1600)
RESIZE_JIT = (-0.06, 0.11)
CROP_H = (0.0, 0.0)


def sample_aug(input_size, H, W, is_train, jitter=0.0, crop_w_frac=0.5):
    """loading_bevdet.py:190-215 그대로. train의 random 값만 인자로 고정."""
    fH, fW = input_size
    resize = float(fW) / float(W) + (jitter if is_train else 0.0)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    if is_train:
        crop_h = int((1 - 0.0) * newH) - fH           # crop_h aug = (0,0)
        crop_w = int(crop_w_frac * max(0, newW - fW))
    else:
        crop_h = int((1 - np.mean(CROP_H)) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
    return resize, resize_dims, (crop_w, crop_h, crop_w + fW, crop_h + fH)


def transform(img, resize_dims, crop):
    return img.resize(resize_dims).crop(crop)   # img_transform_core


def pick_sample(infos):
    """근거리(<20m) 대형차가 있는 샘플 우선 — FoV crop 손실을 보이기 위함."""
    best, best_score = infos[0], -1
    for s in infos[:400]:
        b = s.get('gt_boxes')
        if b is None or len(b) == 0:
            continue
        d = np.linalg.norm(b[:, :2], axis=1)
        near = (d < 20) & (b[:, 5] > 2.0)          # 높이 2m 초과 = 버스/트럭류
        score = near.sum() * 10 + (d < 25).sum()
        if score > best_score:
            best, best_score = s, score
    return best


infos = pickle.load(open(f'{ROOT}/data/nuscenes/nuscenes_occ_infos_val.pkl', 'rb'))['infos']
S = pick_sample(infos)
paths = {c: os.path.join(ROOT, S['cams'][c]['data_path'].lstrip('./')) for c in CAMS}
front = Image.open(paths['CAM_FRONT'])
print('sample token:', S['token'], '| front size:', front.size)


# ── Fig 1. 원본 위에 살아남는 영역 표시 (test mode) ──────────────────────
r, rd, crop = sample_aug((256, 704), *SRC, is_train=False)
y0, y1 = crop[1] / r, crop[3] / r
x0, x1 = crop[0] / r, crop[2] / r

fig, ax = plt.subplots(figsize=(13, 7.6))
ax.imshow(front)
ax.add_patch(patches.Rectangle((0, 0), SRC[1], y0, fc='red', alpha=0.45))
ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, ec='lime', lw=3.5, fc='none'))
ax.axhline(y0, color='yellow', ls='--', lw=2)
ax.text(20, y0 / 2, f'DISCARDED  top {y0:.0f} px  ({y0/SRC[0]*100:.1f}% of height)',
        color='white', fontsize=15, fontweight='bold', va='center',
        bbox=dict(fc='darkred', alpha=.85, pad=6))
ax.text(20, y0 + 45, 'KEPT -> resized to 256x704', color='black', fontsize=15,
        fontweight='bold', bbox=dict(fc='lime', alpha=.85, pad=6))
ax.set_title(f'res704 test-mode crop on original {SRC[0]}x{SRC[1]} (CAM_FRONT)\n'
             f'resize={r:.3f}  ->  resized {rd[1]}x{rd[0]}  ->  crop_h={crop[1]}  ->  256x704',
             fontsize=13)
ax.axis('off')
fig.tight_layout()
fig.savefig(f'{OUT}/01_fov_crop_CAM_FRONT.png', dpi=110, bbox_inches='tight')
plt.close(fig)


# ── Fig 2. 6-cam, 네트워크가 실제로 보는 입력 ────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16.5, 4.6))
for ax, c in zip(axes.ravel(), CAMS):
    _, rd_, cr_ = sample_aug((256, 704), *SRC, is_train=False)
    ax.imshow(transform(Image.open(paths[c]), rd_, cr_))
    ax.set_title(c, fontsize=10)
    ax.axis('off')
fig.suptitle('What the network actually sees @ res704  (6 cams, 256x704 each, test mode)',
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/02_sixcam_res704.png', dpi=120, bbox_inches='tight')
plt.close(fig)


# ── Fig 3. base(896x1600) vs res704(256x704) ────────────────────────────
rb, rdb, cb = sample_aug((896, 1600), *SRC, is_train=False)
img_b, img_r = transform(front, rdb, cb), transform(front, rd, crop)

# 원본(900x1600) 좌표계의 관심 영역 -> 각 전처리 이미지로 사영해 '같은 실세계 영역'을 비교
ROI = (560, 400, 860, 580)          # 원경 차량 + 차선

def to_proc(box, resize_, crop_):
    x0_, y0_, x1_, y1_ = box
    return (x0_ * resize_ - crop_[0], y0_ * resize_ - crop_[1],
            x1_ * resize_ - crop_[0], y1_ * resize_ - crop_[1])

DISP_W = 1600                        # 두 이미지를 같은 표시 폭으로 -> 픽셀 밀도 차가 그대로 보임
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.72, 1.05])

for row, (im, rz, cr, tag, feat, tok) in enumerate([
        (img_b, rb, cb, 'BASE  (896,1600)', '56x100', 6 * 56 * 100),
        (img_r, r, crop, '_res704  (256,704)', '16x44', 6 * 16 * 44)]):
    disp = im.resize((DISP_W, int(im.size[1] * DISP_W / im.size[0])), Image.NEAREST)
    ax_ = fig.add_subplot(gs[row, :])
    ax_.imshow(disp); ax_.axis('off')
    px_, py_, px1_, py1_ = to_proc(ROI, rz, cr)
    s_ = DISP_W / im.size[0]
    ax_.add_patch(patches.Rectangle((px_ * s_, py_ * s_), (px1_ - px_) * s_,
                                    (py1_ - py_) * s_, ec='red', lw=2.5, fc='none'))
    ax_.set_title(f'{tag}   resize={rz:.3f}  crop_h={cr[1]}  ->  feature {feat},  '
                  f'KV tokens {tok}     [both drawn at the same display width]', fontsize=12)

for i, (im, rz, cr, tag) in enumerate([(img_b, rb, cb, 'BASE 896x1600'),
                                       (img_r, r, crop, '_res704 256x704')]):
    bx = tuple(int(v) for v in to_proc(ROI, rz, cr))
    patch = im.crop(bx).resize((560, 340), Image.NEAREST)
    p = fig.add_subplot(gs[2, i])
    p.imshow(patch)
    p.set_title(f'{tag} — identical real-world ROI\n'
                f'actually {bx[2]-bx[0]} x {bx[3]-bx[1]} pixels of data', fontsize=11)
    p.axis('off')
fig.tight_layout()
fig.savefig(f'{OUT}/03_base_vs_res704.png', dpi=115, bbox_inches='tight')
plt.close(fig)


# ── Fig 4. train-time resize jitter 범위 ────────────────────────────────
cases = [(RESIZE_JIT[0], 'jitter MIN  (-0.06)'), (0.0, 'jitter 0  (nominal)'),
         (RESIZE_JIT[1], 'jitter MAX  (+0.11)')]
fig, axes = plt.subplots(3, 1, figsize=(11, 7.2))
for ax, (j, tag) in zip(axes, cases):
    rr, rrd, cc = sample_aug((256, 704), *SRC, is_train=True, jitter=j, crop_w_frac=0.5)
    im = transform(front, rrd, cc)
    ax.imshow(im)
    drop = cc[1] / rrd[1] * 100
    pad = max(0, 704 - rrd[0])
    ax.set_title(f'{tag}   resize={rr:.3f}  resized={rrd[1]}x{rrd[0]}  '
                 f'crop_h={cc[1]}  top-drop={drop:.1f}%'
                 + (f'   <-- RIGHT BLACK PAD {pad}px' if pad else ''),
                 fontsize=11, color=('darkred' if pad else 'black'))
    ax.axis('off')
fig.suptitle("Train-time resize jitter: FoV is NOT fixed at 35%  "
             "(data_config['resize']=(-0.06,+0.11) is ADDITIVE)", fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT}/04_train_jitter.png', dpi=120, bbox_inches='tight')
plt.close(fig)

print('saved ->', OUT)
for f in sorted(os.listdir(OUT)):
    print('  ', f, os.path.getsize(os.path.join(OUT, f)) // 1024, 'KB')
