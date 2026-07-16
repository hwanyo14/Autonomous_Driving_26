"""새 GT 캐시 통합 생성기 — NEW_GT_PIPELINE_SPEC.md v2 + NOTES 2026-07-11 결정사항.

한 윈도우당 prepare_test_data() 1회 → 같은 instance_dict/instance_map으로
A(bbox_aabb_raw), B(bbox_rot_raw), [C(occ_raw)], D(nusocc_inst), E(bbox_aabb), F(bbox_rot)를
한 번에 생성 (id/클래스/좌표 정합 구조 보장).

결정사항 반영:
  ① cls_map = (("pedestrian",7),) + CLS_MAP  — construction_worker→7 (nusocc/lidarseg 의미 정합)
  ② D 중복 좌표 first-wins(작은 id 우선) dedup — strict 로더 통과
  - 필터 전부 OFF: visibility==1 skip 제거, 생성소멸(counter>=trf) 제거, classes=['vehicle','human']
  - npz collapse 방어: np.empty(T,object) 채움 (NOTES: np.array(frames,dtype=object) 금지)
  - C(occ_raw)는 기본 저장 안 함(--save-occ-raw, 용량 ~130GB) — nuScenes-Occupancy에서 재계산 가능

출력: <out>/GMO/{bbox_aabb_raw,bbox_rot_raw[,occ_raw],segmentation_instance3d,segmentation_aabb,segmentation_rot}/<scene>_<lidar>.npz

사용 (shard 병렬 가능, 기존 key 전체 존재 시 skip):
  python tools/gen_data/gen_new_gt_pipeline.py --split test  --out /home/hwanhee/datasets/efficientocf_gt
  python tools/gen_data/gen_new_gt_pipeline.py --split train --start 0 --end 6000 --out ...
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_bbox_gt_v2 import CLS_MAP, atomic_savez, rasterize_sequence

CLS_MAP_PED_FIRST = (("pedestrian", 7),) + CLS_MAP

LAYERS = (
    ("bbox_aabb_raw", "bbox_aabb_raw_saved_list2"),
    ("bbox_rot_raw", "bbox_rot_raw_saved_list2"),
    ("segmentation_instance3d", "segmentation_instance_saved_list2"),
    ("segmentation_aabb", "segmentation_aabb_saved_list2"),
    ("segmentation_rot", "segmentation_rot_saved_list2"),
)
OCC_LAYER = ("occ_raw", "occ_raw_saved_list2")


def make_record_instance_raw(dataset):
    """record_instance에서 visibility==1 최초등장/미래-신규 필터 2개를 제거한 raw 버전
    + rasterize에 필요한 category_name 주입 (스펙 §3.1)."""
    def record_instance_raw(idx, instance_map):
        rec = dataset.data_infos[idx]
        translation, rotation = dataset.get_lidar_pose(rec)
        dataset.egopose_list.append([translation, rotation])
        e2l_t, e2l_r = dataset.get_ego2lidar_pose(rec)
        dataset.ego2lidar_list.append([e2l_t, e2l_r])

        sample = dataset.nusc.get("sample", rec["token"])
        for ann_token in sample["anns"]:
            ann = dataset.nusc.get("sample_annotation", ann_token)
            if not any(c in ann["category_name"] for c in dataset.classes):
                continue
            dataset.visible_instance_set.add(ann["instance_token"])
            if ann["instance_token"] not in instance_map:
                instance_map[ann["instance_token"]] = len(instance_map) + 1
            vis = int(ann["visibility_token"])
            d = dataset.instance_dict
            if ann["instance_token"] not in d:
                d[ann["instance_token"]] = {
                    "timestep": [dataset.counter],
                    "translation": [ann["translation"]],
                    "rotation": [ann["rotation"]],
                    "size": ann["size"],
                    "instance_id": instance_map[ann["instance_token"]],
                    "semantic_id": 1,
                    "attribute_label": [vis],
                    "category_name": ann["category_name"],
                }
            else:
                e = d[ann["instance_token"]]
                e["timestep"].append(dataset.counter)
                e["translation"].append(ann["translation"])
                e["rotation"].append(ann["rotation"])
                e["attribute_label"].append(vis)
        return instance_map
    return record_instance_raw


def get_seq_occ_dense(seq, occ_path, pc_range, grid_size, nb_process_label):
    """loading_occupancy.get_seq_occ와 동일한 등록 체인 + 다수결 병합 (스펙 §4.2)."""
    res = np.array([(pc_range[3 + i] - pc_range[i]) / grid_size[i] for i in range(3)])
    gs = np.array(grid_size)
    pc_lo = np.array(pc_range[:3])
    trf = seq["time_receptive_field"]
    dense_frames = []
    for t in range(seq["sequence_length"]):
        idic = seq["input_dict"][t]
        pcd = np.load(os.path.join(occ_path, f"scene_{idic['scene_token']}/occupancy/{idic['lidar_token']}.npy"))
        lab = pcd[..., -1:].copy()
        lab[lab == 0] = 255
        cor = (pcd[..., [2, 1, 0]] + 0.5) * res[None, :] + pc_lo[None, :]
        pge, pe2l = seq["egopose_list"][trf - 1], seq["ego2lidar_list"][trf - 1]
        cge, ce2l = seq["egopose_list"][t], seq["ego2lidar_list"][t]
        cor = np.dot(ce2l[1].inverse.rotation_matrix, cor.T).T - ce2l[0]
        cor = np.dot(cge[1].inverse.rotation_matrix, cor.T).T - cge[0]
        cor = cor + pge[0]
        cor = np.dot(pge[1].rotation_matrix, cor.T).T
        cor = cor + pe2l[0]
        cor = np.dot(pe2l[1].rotation_matrix, cor.T).T
        cor = np.clip((cor - pc_lo[None, :]) / res[None, :], np.zeros(3), gs - 1)
        rows = np.concatenate([cor, lab], axis=-1)
        rows = rows[np.lexsort((cor[:, 0], cor[:, 1], cor[:, 2])), :].astype(np.int64)
        dense = np.zeros(grid_size, dtype=np.uint8)
        dense_frames.append(nb_process_label(dense, rows))
    return dense_frames


def compute_D(aabb_frames, dense_frames):
    """D = A ∩ C실점유(0/255 제외), first-wins dedup(결정②).
    aabb_frames의 row는 instance 등록 순(id 오름차순 블록)이므로 첫 등장 row 유지가 곧 작은 id 우선."""
    out = []
    for rows, dense in zip(aabb_frames, dense_frames):
        if rows.shape[0] == 0:
            out.append(np.zeros((0, 5), np.int32))
            continue
        occ = (dense != 0) & (dense != 255)
        keep = rows[occ[rows[:, 0], rows[:, 1], rows[:, 2]]]
        enc = (keep[:, 0].astype(np.int64) * 512 + keep[:, 1]) * 64 + keep[:, 2]
        _, first_idx = np.unique(enc, return_index=True)
        out.append(np.ascontiguousarray(keep[np.sort(first_idx)]).astype(np.int32))
    return out


def apply_gate(raw_frames, D_frames):
    """존재 게이트(스펙 §6): D에 (frame,inst) row가 있으면 raw의 해당 인스턴스 박스 통짜 유지."""
    keep = {(t, int(i)) for t, r in enumerate(D_frames) for i in np.unique(r[:, 4])}
    out = []
    for t, rows in enumerate(raw_frames):
        if rows.shape[0] == 0:
            out.append(rows)
            continue
        mask = np.fromiter(((t, int(i)) in keep for i in rows[:, 4]), bool, len(rows))
        out.append(rows[mask])
    return out


def save_npz(out_dir, key, frames, npz_key):
    arr = np.empty(len(frames), dtype=object)  # collapse 방어
    arr[:] = [np.ascontiguousarray(f, dtype=np.int32) for f in frames]
    atomic_savez(os.path.join(out_dir, key), **{npz_key: arr})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/baselines/full.py")
    ap.add_argument("--out", default="/home/hwanhee/datasets/efficientocf_gt")
    ap.add_argument("--occ-path", default="./data/nuScenes-Occupancy")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--save-occ-raw", action="store_true",
                    help="C(occ_raw)도 저장 (키당 ~4.5MB, 전량 ~130GB — 기본 끔)")
    args = ap.parse_args()

    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        import importlib
        importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))
    from mmdet3d.datasets import build_dataset
    from projects.occ_plugin.datasets.pipelines.loading_occupancy import nb_process_label

    split_cfg = cfg.data.train if args.split == "train" else cfg.data.test
    split_cfg.pipeline = []
    dataset = build_dataset(split_cfg)
    dataset.classes = ["vehicle", "human"]
    dataset.record_instance = make_record_instance_raw(dataset)

    pc_range = list(cfg.point_cloud_range)
    grid_size = list(cfg.occ_size)

    dirs = {}
    for name, _ in LAYERS + ((OCC_LAYER,) if args.save_occ_raw else ()):
        dirs[name] = os.path.join(args.out, "GMO", name)
        os.makedirs(dirs[name], exist_ok=True)

    n_total = len(dataset.indices)
    end = n_total if args.end < 0 else min(args.end, n_total)
    made = skipped = 0
    for i in range(args.start, end):
        dataset.egopose_list = []
        dataset.ego2lidar_list = []
        dataset.visible_instance_set = set()
        dataset.instance_dict = {}
        seq = dataset.prepare_test_data(i)
        if seq is None:
            continue
        key = dataset.present_scene_lidar_token
        if all(os.path.exists(os.path.join(d, key + ".npz")) for d in dirs.values()):
            skipped += 1
            continue

        A, B = rasterize_sequence(seq, pc_range, grid_size, CLS_MAP_PED_FIRST, False)
        dense = get_seq_occ_dense(seq, args.occ_path, pc_range, grid_size, nb_process_label)
        D = compute_D(A, dense)
        E = apply_gate(A, D)
        F = apply_gate(B, D)

        for (name, npz_key), frames in zip(LAYERS, (A, B, D, E, F)):
            save_npz(dirs[name], key, frames, npz_key)
        if args.save_occ_raw:
            C = []
            for dn in dense:
                xs, ys, zs = np.nonzero(dn != 0)
                C.append(np.stack([xs, ys, zs, dn[xs, ys, zs]], 1).astype(np.int32))
            save_npz(dirs[OCC_LAYER[0]], key, C, OCC_LAYER[1])

        made += 1
        if made % 50 == 0:
            print(f"[gen_new_gt] {i + 1}/{end} made={made} skipped={skipped}", flush=True)
        if args.limit > 0 and made >= args.limit:
            break
    print(f"[gen_new_gt] done range=({args.start},{end}) made={made} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
