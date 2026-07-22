#!/usr/bin/env python3
"""Offline CLS+depth score sweep from eval depth dumps (no model forward)."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.occ_plugin.occupancy.dense_heads.voxelizer import SoftVoxelizerOneAdd


def csv_floats(value):
    return [float(v) for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--betas", type=csv_floats, default=csv_floats("0,0.1,0.2,0.3,0.5"))
    parser.add_argument("--base-fg-thresholds", type=csv_floats, default=csv_floats("0.95,0.90,0.85"))
    parser.add_argument("--render-modes", default="cls,combined")
    parser.add_argument("--occ-threshold", type=float, default=0.85)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--bg-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def final_score(payload, beta):
    cls_conf = payload["cls_conf_q"].float().clamp(1e-6, 1.0 - 1e-6)
    if beta == 0.0:
        return cls_conf
    depth_conf = payload["depth_conf_q"].float().clamp(0.0, 1.0)
    return torch.sigmoid(torch.logit(cls_conf) + beta * depth_conf)


def selected_indices(payload, score, threshold, topk, bg_index):
    candidate_idx = payload["candidate_query_idx_q"].long()
    valid = payload["pred_cls_q"].long().index_select(0, candidate_idx) != int(bg_index)
    candidate_idx = candidate_idx[valid]
    candidate_score = score.index_select(0, candidate_idx)
    keep = torch.nonzero(candidate_score >= threshold, as_tuple=False).reshape(-1)
    if int(keep.numel()) > topk:
        order = torch.argsort(candidate_score.index_select(0, keep), descending=True)
        keep = keep.index_select(0, order[:topk])
    return keep, candidate_idx, candidate_score


def total_selected(payloads, scores, threshold, topk, bg_index):
    return sum(
        int(selected_indices(p, s, threshold, topk, bg_index)[0].numel())
        for p, s in zip(payloads, scores)
    )


def calibrated_threshold(payloads, scores, target, topk, bg_index):
    if target <= 0:
        return 1.0, 0
    values = torch.unique(torch.cat([
        score.index_select(0, payload["candidate_query_idx_q"].long())
        for payload, score in zip(payloads, scores)
    ])).sort(descending=True).values
    best = (float(values[0]), math.inf, 0)
    for value in values:
        threshold = float(value)
        count = total_selected(payloads, scores, threshold, topk, bg_index)
        gap = abs(count - target)
        if gap < best[1]:
            best = (threshold, gap, count)
            if gap == 0:
                break
    return best[0], best[2]


def load_dense(gt_root, sample_key, subdir, npz_key):
    path = gt_root / subdir / f"{sample_key}.npz"
    with np.load(path, allow_pickle=True) as data:
        frames = list(data[npz_key] if npz_key in data.files else data[data.files[0]])
    dense = torch.zeros((len(frames), 512, 512, 40), dtype=torch.bool)
    for frame_idx, rows in enumerate(frames):
        array = np.asarray(rows)
        if array.dtype == object:
            array = np.vstack(array) if array.size else np.zeros((0, 5), dtype=np.int64)
        array = np.asarray(array, dtype=np.int64)
        if not array.size:
            continue
        array = array.reshape(-1, array.shape[-1])
        if array.shape[1] >= 4:
            array = array[array[:, 3] != 7]
        xyz = array[:, :3]
        valid = ((xyz >= 0) & (xyz < np.asarray([512, 512, 40]))).all(axis=1)
        xyz = xyz[valid]
        if xyz.size:
            dense[
                frame_idx,
                torch.from_numpy(xyz[:, 0]),
                torch.from_numpy(xyz[:, 1]),
                torch.from_numpy(xyz[:, 2]),
            ] = True
    return dense[-4:].permute(0, 3, 2, 1).contiguous()  # [T,Z,Y,X]


def main():
    args = parse_args()
    files = sorted(args.dump_dir.glob("*_depth.pt"))
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    required = {
        "sample_key", "cls_conf_q", "depth_conf_q", "pred_cls_q", "candidate_query_idx_q",
        "mixture_centers_tqg3", "mixture_sigmas_tqg3", "mixture_yaw_tqg", "mixture_weights_tqg",
    }
    if not payloads or any(not required.issubset(payload) for payload in payloads):
        raise RuntimeError("Mixture-enabled depth dumps are required (EOCF_EVAL_DEPTH_SAVE_MIXTURE=1)")
    output_dir = args.output_dir or args.dump_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    render_modes = [mode.strip() for mode in args.render_modes.split(",") if mode.strip()]
    if any(mode not in {"cls", "combined"} for mode in render_modes):
        raise ValueError("render modes must be cls and/or combined")

    baseline_scores = [final_score(p, 0.0) for p in payloads]
    budgets = [
        (threshold, total_selected(payloads, baseline_scores, threshold, args.topk, args.bg_index))
        for threshold in args.base_fg_thresholds
    ]
    thresholds = {}
    scores_by_beta = {}
    for beta in args.betas:
        scores = [final_score(p, beta) for p in payloads]
        scores_by_beta[beta] = scores
        for base_threshold, budget in budgets:
            thresholds[(beta, base_threshold)] = calibrated_threshold(
                payloads, scores, budget, args.topk, args.bg_index)

    settings = [
        (mode, base_threshold, budget, beta)
        for mode in render_modes
        for base_threshold, budget in budgets
        for beta in args.betas
    ]
    results = {
        setting: dict(a_tp=0.0, a_fp=0.0, a_fn=0.0, recalls=[], selected=0)
        for setting in settings
    }
    voxelizer = SoftVoxelizerOneAdd(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=0.2,
        occ_size=(512, 512, 40),
        as_prob=True,
        lambda_occ=1.0,
        gaussian_sigma_floor_vox=0.0,
        gaussian_combine_mode="union",
    ).to(args.device).eval()

    with torch.no_grad():
        for sample_idx, payload in enumerate(payloads, 1):
            sample_key = payload["sample_key"]
            inst = load_dense(
                args.gt_root, sample_key,
                "segmentation_instance3d", "segmentation_instance_saved_list2",
            ).to(args.device)
            aabb = load_dense(
                args.gt_root, sample_key,
                "segmentation_aabb", "segmentation_aabb_saved_list2",
            ).to(args.device)
            candidate_query_idx = payload["candidate_query_idx_q"].long()
            mixtures = [
                payload[key].to(args.device)
                for key in (
                    "mixture_centers_tqg3", "mixture_sigmas_tqg3",
                    "mixture_weights_tqg", "mixture_yaw_tqg",
                )
            ]
            for setting in settings:
                mode, base_threshold, _budget, beta = setting
                score = scores_by_beta[beta][sample_idx - 1]
                threshold, _ = thresholds[(beta, base_threshold)]
                keep, _, candidate_score = selected_indices(
                    payload, score, threshold, args.topk, args.bg_index)
                result = results[setting]
                result["selected"] += int(keep.numel())
                if keep.numel() == 0:
                    pred = torch.zeros_like(inst)
                else:
                    keep_device = keep.to(args.device)
                    if mode == "combined":
                        pair_score = candidate_score.index_select(0, keep)
                    else:
                        pair_score = payload["cls_conf_q"].float().index_select(
                            0, candidate_query_idx).index_select(0, keep)
                    pair_weight = pair_score.to(args.device).view(1, -1).expand(4, -1)
                    occ = voxelizer.forward_gaussian_mixture_scene(
                        *(value.index_select(1, keep_device) for value in mixtures),
                        pair_weights_tq=pair_weight,
                    )
                    pred = occ[:, 0] > args.occ_threshold
                a_tp = float((pred & aabb).sum())
                a_fp = float((pred & ~aabb).sum())
                a_fn = float((~pred & aabb).sum())
                tp = float((pred & inst).sum())
                fp = float((pred & ~inst).sum())
                fn = float((~pred & inst).sum())
                bbox_fp = float((pred & ~inst & aabb).sum())
                denominator = tp + fn + fp - bbox_fp
                result["a_tp"] += a_tp
                result["a_fp"] += a_fp
                result["a_fn"] += a_fn
                result["recalls"].append(
                    (tp + bbox_fp) / denominator if denominator > 0 else float("nan"))
            print(f"[{sample_idx}/{len(payloads)}] {sample_key}", flush=True)

    output = output_dir / "depth_score_sweep.csv"
    fieldnames = [
        "render_mode", "base_cls_threshold", "target_budget", "beta",
        "calibrated_threshold", "selected", "iou3d_aabb", "recall3d_aabb",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        rows = []
        for setting in settings:
            mode, base_threshold, budget, beta = setting
            result = results[setting]
            denominator = result["a_tp"] + result["a_fp"] + result["a_fn"]
            iou = result["a_tp"] / denominator if denominator > 0 else float("nan")
            recall = float(np.nanmean(result["recalls"]))
            threshold, _ = thresholds[(beta, base_threshold)]
            row = {
                "render_mode": mode,
                "base_cls_threshold": base_threshold,
                "target_budget": budget,
                "beta": beta,
                "calibrated_threshold": threshold,
                "selected": result["selected"],
                "iou3d_aabb": iou,
                "recall3d_aabb": recall,
            }
            rows.append(row)
            writer.writerow(row)
    finite_rows = [
        row for row in rows
        if math.isfinite(row["iou3d_aabb"]) and math.isfinite(row["recall3d_aabb"])
    ]
    summary = {}
    if finite_rows:
        summary = {
            "best_iou": max(finite_rows, key=lambda row: row["iou3d_aabb"]),
            "best_recall": max(finite_rows, key=lambda row: row["recall3d_aabb"]),
            "best_balanced": max(
                finite_rows,
                key=lambda row: math.sqrt(row["iou3d_aabb"] * row["recall3d_aabb"]),
            ),
        }
        (output_dir / "depth_score_best.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")
    if summary:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
