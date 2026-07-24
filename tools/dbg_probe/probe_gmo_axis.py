"""Measure matched asset-GT/Gaussian BEV principal-axis errors from a checkpoint."""
import argparse
import importlib
import json
import os
import random
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(REPO)
sys.path.insert(0, REPO)
importlib.import_module("projects.occ_plugin")

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config-b")
    parser.add_argument("--checkpoint-b")
    parser.add_argument("--output-b")
    return parser.parse_args()


def _axis_stats(xy, weight=None, eps=1e-6):
    if xy.shape[0] < 2:
        return None
    if weight is None:
        weight = torch.ones(xy.shape[0], device=xy.device, dtype=torch.float32)
    weight = weight.float()
    weight = weight / weight.sum().clamp_min(eps)
    delta = xy.float() - (weight[:, None] * xy.float()).sum(dim=0, keepdim=True)
    cov = torch.einsum("n,nc,nd->cd", weight, delta, delta)
    eigval, eigvec = torch.linalg.eigh(cov)
    ratio = float((eigval[-1] / eigval[0].clamp_min(eps)).item())
    return eigvec[:, -1], ratio


def _angle_deg(a, b):
    dot = (a * b).sum().abs().clamp(0.0, 1.0)
    return float(torch.rad2deg(torch.acos(dot)).item())


def _summarize(rows, key):
    values = np.asarray([r[key] for r in rows if r.get(key) is not None], dtype=np.float64)
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def _build_probe_model(cfg, checkpoint, device):
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint, map_location="cpu")
    model = model.to(device).eval()
    model.set_train_iteration(10**9)
    for name in (
        "debug_query_vis_every", "debug_query_cam_gaussian_vis_every",
        "debug_query_mixture3d_vis_every", "debug_query_attn_softargmax_vis_every",
    ):
        setattr(model, name, 0)
    model.debug_query_cam_gaussian_vis_enabled = False
    model.query_head.debug_vis_every = 0

    capture = {}
    original_prepare = model._prepare_grouped_matched_pair_lowres_occ

    def wrapped_prepare(*pos, **kw):
        result = original_prepare(*pos, **kw)
        capture["prepared"] = tuple(x.detach() if torch.is_tensor(x) else x for x in result)
        capture["mixture_centers"] = kw["mixture_centers_world_tqg3"].detach()
        capture["mixture_weights"] = kw["mixture_weights_tqg"].detach()
        capture["matched_query_idx"] = kw["matched_query_idx"].detach()
        capture["pair_gt_ids"] = kw["pair_gt_ids_k"].detach()
        return result

    model._prepare_grouped_matched_pair_lowres_occ = wrapped_prepare
    return model, capture


def _collect_rows(capture, sample_idx):
    rows = []
    if "prepared" not in capture:
        return rows
    pred, gt, valid = capture["prepared"]
    centers = capture["mixture_centers"]
    weights = capture["mixture_weights"]
    matched_query_idx = capture["matched_query_idx"].long()
    pair_gt_ids = capture["pair_gt_ids"].long()
    for t in range(pred.shape[0]):
        for k in range(pred.shape[1]):
            if not bool(valid[t, k].item()):
                continue
            gt_bev = gt[t, k, 0].amax(dim=0).float()
            gt_yx = torch.nonzero(gt_bev > 0, as_tuple=False)
            if gt_yx.shape[0] < 4:
                continue
            gt_stat = _axis_stats(gt_yx[:, [1, 0]])
            q = int(matched_query_idx[k].item())
            equal_stat = _axis_stats(centers[t, q, :, :2])
            weighted_stat = _axis_stats(centers[t, q, :, :2], weights[t, q])
            if gt_stat is None or equal_stat is None or weighted_stat is None:
                continue
            row = {
                "sample": sample_idx, "frame": t, "pair": k,
                "gt_instance_id": int(pair_gt_ids[k].item()),
                "gt_voxels_bev": int(gt_yx.shape[0]),
                "center_equal_angle_deg": _angle_deg(equal_stat[0], gt_stat[0]),
                "center_weighted_angle_deg": _angle_deg(weighted_stat[0], gt_stat[0]),
                "gt_axis_ratio": gt_stat[1],
                "center_equal_axis_ratio": equal_stat[1],
                "center_weighted_axis_ratio": weighted_stat[1],
            }
            pred_bev = pred[t, k, 0].amax(dim=0)
            for threshold in (0.5, 0.75):
                pred_yx = torch.nonzero(pred_bev >= threshold, as_tuple=False)
                stat = _axis_stats(pred_yx[:, [1, 0]]) if pred_yx.shape[0] >= 2 else None
                row[f"render_{threshold}_angle_deg"] = _angle_deg(stat[0], gt_stat[0]) if stat else None
                row[f"render_{threshold}_axis_ratio"] = stat[1] if stat else None
            rows.append(row)
    return rows


def main():
    args = parse_args()
    paired = all((args.config_b, args.checkpoint_b, args.output_b))
    if any((args.config_b, args.checkpoint_b, args.output_b)) and not paired:
        raise ValueError("--config-b, --checkpoint-b, --output-b must be provided together")
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    specs = [("a", cfg, args.checkpoint, args.output, torch.device("cuda:0"))]
    if paired:
        specs.append(("b", Config.fromfile(args.config_b), args.checkpoint_b,
                      args.output_b, torch.device("cuda:1")))
    probes = [(tag, *_build_probe_model(c, ck, dev), ck, out, dev)
              for tag, c, ck, out, dev in specs]
    all_rows = {tag: [] for tag, *_ in probes}
    completed = {tag: 0 for tag, *_ in probes}
    for idx in range(min(len(dataset), args.samples)):
        random.seed(args.seed + idx)
        np.random.seed(args.seed + idx)
        torch.manual_seed(args.seed + idx)
        try:
            batch = collate([dataset[idx]], samples_per_gpu=1)
        except Exception as exc:
            print(f"[skip] idx={idx}: dataset: {exc}", flush=True)
            continue
        for tag, model, capture, checkpoint, output, device in probes:
            try:
                data = scatter(batch, [device.index])[0]
                capture.clear()
                with torch.cuda.device(device), torch.no_grad():
                    model.forward_train(**data)
            except Exception as exc:
                print(f"[skip] {tag} idx={idx}: {exc}", flush=True)
                continue
            new_rows = _collect_rows(capture, idx)
            all_rows[tag].extend(new_rows)
            completed[tag] += 1
            print(f"[ok] {tag} sample={idx} completed={completed[tag]} "
                  f"pairs={len(all_rows[tag])}", flush=True)

    keys = ("center_equal_angle_deg", "center_weighted_angle_deg",
            "render_0.5_angle_deg", "render_0.75_angle_deg")
    for tag, _, _, checkpoint, output, _ in probes:
        rows = all_rows[tag]
        result = {"checkpoint": os.path.abspath(checkpoint), "requested_samples": args.samples,
                  "completed_samples": completed[tag],
                  "summary": {key: _summarize(rows, key) for key in keys}, "rows": rows}
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[{tag}] {json.dumps(result['summary'], indent=2)}", flush=True)
        print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
