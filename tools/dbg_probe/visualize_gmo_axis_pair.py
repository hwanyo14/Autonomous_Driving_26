"""Visualize paired checkpoint Gaussian-center/rendered BEV principal axes."""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset

from probe_gmo_axis import _build_probe_model


DEFAULT_CASES = (
    ("improved", 12, 1, 14), ("improved", 9, 2, 2),
    ("neutral", 9, 1, 3), ("neutral", 9, 0, 1),
    ("worse", 5, 0, 5), ("worse", 12, 2, 0),
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config_a")
    p.add_argument("checkpoint_a")
    p.add_argument("config_b")
    p.add_argument("checkpoint_b")
    p.add_argument("--label-a", default="no-cov ep1")
    p.add_argument("--label-b", default="cov-dir ep1")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=20260722)
    return p.parse_args()


def axis_stats(xy):
    mean = xy.mean(axis=0)
    delta = xy - mean
    cov = delta.T @ delta / max(1, len(xy))
    val, vec = np.linalg.eigh(cov)
    return mean, vec[:, -1], float(val[-1] / max(val[0], 1e-8)), float(np.sqrt(max(val[-1], 1e-8)))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)), 0.0, 1.0))))


def draw_axis(ax, mean, vec, scale, color, label, width=2.2):
    length = max(1.0, 2.0 * scale)
    p0, p1 = mean - length * vec, mean + length * vec
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=width, label=label, zorder=6)


def extract_case(capture, frame, pair_idx, pc_range):
    pred, gt, valid = capture["prepared"]
    ids = capture["pair_gt_ids"].long()
    k = int(pair_idx)
    if not 0 <= k < ids.numel():
        raise RuntimeError(f"pair {k}: only {ids.numel()} pairs")
    if not bool(valid[frame, k].item()):
        raise RuntimeError(f"pair {k} frame {frame}: invalid pair")
    q = int(capture["matched_query_idx"][k].item())
    centers = capture["mixture_centers"][frame, q, :, :2].detach().cpu().numpy()
    gt_bev = gt[frame, k, 0].amax(dim=0)
    render_bev = pred[frame, k, 0].amax(dim=0)
    h, w = gt_bev.shape
    vx = (pc_range[3] - pc_range[0]) / w
    vy = (pc_range[4] - pc_range[1]) / h

    def world_points(mask):
        yx = torch.nonzero(mask, as_tuple=False).detach().cpu().numpy()
        return np.stack((pc_range[0] + (yx[:, 1] + 0.5) * vx,
                         pc_range[1] + (yx[:, 0] + 0.5) * vy), axis=-1)

    return centers, world_points(gt_bev > 0), world_points(render_bev >= 0.75), int(ids[k])


def main():
    args = parse_args()
    cfg_a, cfg_b = Config.fromfile(args.config_a), Config.fromfile(args.config_b)
    dataset = build_dataset(cfg_a.data.train)
    model_a, cap_a = _build_probe_model(cfg_a, args.checkpoint_a, torch.device("cuda:0"))
    model_b, cap_b = _build_probe_model(cfg_b, args.checkpoint_b, torch.device("cuda:1"))
    models = ((model_a, cap_a, 0), (model_b, cap_b, 1))
    wanted = {}
    for case in DEFAULT_CASES:
        wanted.setdefault(case[1], []).append(case)
    cases = {}
    for sample_idx in range(max(wanted) + 1):
        random.seed(args.seed + sample_idx)
        np.random.seed(args.seed + sample_idx)
        torch.manual_seed(args.seed + sample_idx)
        batch = collate([dataset[sample_idx]], samples_per_gpu=1)
        if sample_idx not in wanted:
            continue
        sample_cases = []
        for model, capture, device_idx in models:
            data = scatter(batch, [device_idx])[0]
            capture.clear()
            with torch.cuda.device(device_idx), torch.no_grad():
                model.forward_train(**data)
            sample_cases.append({
                (frame, pair_idx): extract_case(capture, frame, pair_idx,
                                             np.asarray(cfg_a.point_cloud_range, dtype=np.float32))
                for _, _, frame, pair_idx in wanted[sample_idx]
            })
        cases[sample_idx] = sample_cases
        print(f"[ok] sample={sample_idx}", flush=True)

    fig, axes = plt.subplots(len(DEFAULT_CASES), 3, figsize=(16, 27), squeeze=False)
    pc_range = np.asarray(cfg_a.point_cloud_range, dtype=np.float32)
    for row, (group, sample_idx, frame, pair_idx) in enumerate(DEFAULT_CASES):
        ca, cb = cases[sample_idx]
        a, b = ca[(frame, pair_idx)], cb[(frame, pair_idx)]
        gt = a[1]
        gt_mean, gt_axis, gt_ratio, gt_scale = axis_stats(gt)
        all_points = np.concatenate((gt, a[0], a[2], b[0], b[2]), axis=0)
        lo, hi = all_points.min(axis=0), all_points.max(axis=0)
        center = 0.5 * (lo + hi)
        radius = max(3.0, 0.6 * float((hi - lo).max()))

        ax = axes[row, 0]
        ax.scatter(gt[:, 0], gt[:, 1], s=20, c="black", marker="s", label="asset GT")
        draw_axis(ax, gt_mean, gt_axis, gt_scale, "black", "GT axis")
        ax.set_title(f"{group.upper()} | sample {sample_idx}, t{frame}, pair {pair_idx}, id {a[3]}\n"
                     f"asset GT | voxels={len(gt)}, ratio={gt_ratio:.2f}")

        for col, (label, item) in enumerate(((args.label_a, a), (args.label_b, b)), start=1):
            centers, _, rendered = item
            c_mean, c_axis, c_ratio, c_scale = axis_stats(centers)
            axp = axes[row, col]
            axp.scatter(gt[:, 0], gt[:, 1], s=13, c="#aaaaaa", marker="s", alpha=0.55,
                        label="asset GT")
            if len(rendered):
                axp.scatter(rendered[:, 0], rendered[:, 1], s=12, c="#53b7d2", marker="s",
                            alpha=0.42, label="render >= 0.75")
            axp.scatter(centers[:, 0], centers[:, 1], s=18, c="#d62728", alpha=0.85,
                        label="48 centers")
            draw_axis(axp, gt_mean, gt_axis, gt_scale, "black", "GT axis")
            draw_axis(axp, c_mean, c_axis, c_scale, "#d62728", "center axis")
            axp.set_title(f"{label}\ncenter error={angle_deg(c_axis, gt_axis):.1f} deg, "
                          f"ratio={c_ratio:.2f}")

        for ax in axes[row]:
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Asset GT vs no-cov/cov-dir epoch1 — matched Gaussian-center axes", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
