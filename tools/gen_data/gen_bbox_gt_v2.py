"""val(eval) 전용 bbox GT v2 캐시 생성기 — AABB + rotated(OBB), [x,y,z,cls,inst].

기존 bboxcls/bboxcls_rot 캐시의 오염(생성소멸 누수 5/39, box 누락 5/39, barrier 유입,
per-voxel inst id 부재)을 해결하기 위해 annotation에서 직접 재생성한다.

필터는 dataset 조립 단계에서 자동 상속된다 (NOTES 2026-07-06):
- 사람 제외: config class_names에 pedestrian이 없어 instance_dict에 아예 안 들어옴
- 생성소멸: record_instance가 t >= time_receptive_field 신규 instance를 skip
- 저가시성: visibility_token==1 && 미관측 instance skip (동일 함수)
좌표/rasterize 규칙은 LoadInstanceWithFlow.get_poly_region/fill_occupancy와 동일
(whole-box pc_range 체크, round((w-start+res/2)/res) 격자화). OBB는 AABB 후보 voxel
중심을 box 좌표계로 되돌려 half-extent(+반voxel 여유) 검사로 채운다.

출력: <out>/GMO/segmentation_aabb/<scene>_<lidar>.npz  key=segmentation_aabb_saved_list2
      <out>/GMO/segmentation_rot/<scene>_<lidar>.npz   key=segmentation_rot_saved_list2
      per-frame int64 [N,5] = [x, y, z, nusocc_cls, instance_id]

사용:
  python tools/gen_data/gen_bbox_gt_v2.py --config projects/configs/baselines/full.py \
      [--out ./data/efficientocf_bboxcls_v2] [--start 0 --end 5119] [--limit N]
이미 존재하는 key는 skip (재실행 안전 / shard 병렬 가능).
"""
import argparse
import os
import sys

import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

sys.path.insert(0, os.getcwd())

# category_name substring -> nusocc class id (efficientocf_dataset의 MMO 매핑과 동일
# 판정 순서; 'vehicle.bicycle'만 bicycle 인정 → static_object.bicycle_rack 배제)
CLS_MAP = (
    ("vehicle.bicycle", 2),
    ("bus", 3),
    ("car", 4),
    ("construction", 5),
    ("motorcycle", 6),
    ("trailer", 9),
    ("truck", 10),
)


def category_to_nusocc_id(name):
    for sub, cid in CLS_MAP:
        if sub in name:
            return cid
    return None


def atomic_savez(path_no_ext, **kw):
    tmp = path_no_ext + f".tmp{os.getpid()}"
    np.savez_compressed(tmp, **kw)
    os.replace(tmp + ".npz", path_no_ext + ".npz")


def rasterize_sequence(seq, pc_range, grid_size):
    """seq: prepare_sequential_data 출력 dict. returns (aabb_list, rot_list)."""
    res = np.array([(pc_range[3 + i] - pc_range[i]) / grid_size[i] for i in range(3)])
    start = np.array([pc_range[i] + res[i] / 2.0 for i in range(3)])
    dims = np.asarray(grid_size, dtype=np.int64)

    trf = seq["time_receptive_field"]
    T = seq["sequence_length"]
    present_ego = seq["egopose_list"][trf - 1]
    present_e2l = seq["ego2lidar_list"][trf - 1]

    aabb_frames, rot_frames = [], []
    for t in range(T):
        rows_aabb, rows_rot = [], []
        for token, inst in seq["instance_dict"].items():
            if t not in inst["timestep"]:
                continue
            ptr = inst["timestep"].index(t)
            cls_id = category_to_nusocc_id(inst["category_name"])
            if cls_id is None:
                continue

            box = Box(inst["translation"][ptr], inst["size"], Quaternion(inst["rotation"][ptr]))
            box.translate(present_ego[0])
            box.rotate(present_ego[1])
            box.translate(present_e2l[0])
            box.rotate(present_e2l[1])
            pts = box.corners().T  # [8,3] lidar coords

            lo, hi = pts.min(axis=0), pts.max(axis=0)
            # loader와 동일한 whole-box in-range 규칙
            if not (pc_range[0] <= lo[0] and hi[0] <= pc_range[3]
                    and pc_range[1] <= lo[1] and hi[1] <= pc_range[4]
                    and pc_range[2] <= lo[2] and hi[2] <= pc_range[5]):
                continue

            vox = np.round((pts - start + res / 2.0) / res).astype(np.int64)
            vmin = np.clip(vox.min(axis=0), 0, dims - 1)
            vmax = np.clip(vox.max(axis=0), 0, dims - 1)
            xs = np.arange(vmin[0], vmax[0] + 1)
            ys = np.arange(vmin[1], vmax[1] + 1)
            zs = np.arange(vmin[2], vmax[2] + 1)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            cand = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # [M,3]

            iid = int(inst["instance_id"])
            tag = np.array([[cls_id, iid]], dtype=np.int64).repeat(len(cand), axis=0)
            rows_aabb.append(np.concatenate([cand, tag], axis=1))

            # OBB: voxel center -> box local frame, half-extent(+반voxel) 검사
            centers_w = cand * res.reshape(1, 3) + start.reshape(1, 3) - res.reshape(1, 3) / 2.0
            local = (centers_w - box.center.reshape(1, 3)) @ box.rotation_matrix  # R^T x = x @ R
            w, l, h = box.wlh
            half = np.array([l / 2.0, w / 2.0, h / 2.0]) + res / 2.0
            inside = np.all(np.abs(local) <= half.reshape(1, 3), axis=1)
            if inside.any():
                rows_rot.append(np.concatenate([cand[inside], tag[inside]], axis=1))

        def pack(rows):
            return (np.concatenate(rows, axis=0).astype(np.int32)
                    if rows else np.zeros((0, 5), dtype=np.int32))

        aabb_frames.append(pack(rows_aabb))
        rot_frames.append(pack(rows_rot))
    return aabb_frames, rot_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/baselines/full.py")
    ap.add_argument("--out", default="./data/efficientocf_bboxcls_v2")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1, help="생성 개수 제한(스모크용)")
    args = ap.parse_args()

    from mmcv import Config
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, "plugin_dir"):
        import importlib
        importlib.import_module(cfg.plugin_dir.rstrip("/").replace("/", "."))
    else:
        import projects.occ_plugin  # noqa: F401
    from mmdet3d.datasets import build_dataset

    cfg.data.test.pipeline = []  # annotation 조립까지만 (이미지/캐시 로딩 생략)
    dataset = build_dataset(cfg.data.test)

    # instance_dict에 category_name이 없으므로 record_instance를 category 기록 버전으로 patch
    orig_record = dataset.record_instance

    def record_with_category(idx, instance_map):
        rec = dataset.data_infos[idx]
        sample = dataset.nusc.get("sample", rec["token"])
        cat = {}
        for ann_token in sample["anns"]:
            ann = dataset.nusc.get("sample_annotation", ann_token)
            cat[ann["instance_token"]] = ann["category_name"]
        out = orig_record(idx, instance_map)
        for token, inst in dataset.instance_dict.items():
            if "category_name" not in inst and token in cat:
                inst["category_name"] = cat[token]
        return out

    dataset.record_instance = record_with_category

    pc_range = list(cfg.point_cloud_range)
    grid_size = list(cfg.occ_size) if hasattr(cfg, "occ_size") else [512, 512, 40]

    out_aabb = os.path.join(args.out, "GMO", "segmentation_aabb")
    out_rot = os.path.join(args.out, "GMO", "segmentation_rot")
    os.makedirs(out_aabb, exist_ok=True)
    os.makedirs(out_rot, exist_ok=True)

    n_total = len(dataset)
    end = n_total if args.end < 0 else min(args.end, n_total)
    made = skipped = 0
    for i in range(args.start, end):
        # __getitem__이 하던 per-sequence 상태 리셋
        dataset.egopose_list = []
        dataset.ego2lidar_list = []
        dataset.visible_instance_set = set()
        dataset.instance_dict = {}
        seq = dataset.prepare_test_data(i)
        if seq is None:
            continue
        key = dataset.present_scene_lidar_token
        pa = os.path.join(out_aabb, key)
        pr = os.path.join(out_rot, key)
        if os.path.exists(pa + ".npz") and os.path.exists(pr + ".npz"):
            skipped += 1
            continue
        aabb, rot = rasterize_sequence(seq, pc_range, grid_size)
        atomic_savez(pa, segmentation_aabb_saved_list2=np.array(aabb, dtype=object))
        atomic_savez(pr, segmentation_rot_saved_list2=np.array(rot, dtype=object))
        made += 1
        if made % 100 == 0:
            print(f"[gen_bbox_gt_v2] {i + 1}/{end} made={made} skipped={skipped}", flush=True)
        if args.limit > 0 and made >= args.limit:
            break
    print(f"[gen_bbox_gt_v2] done range=({args.start},{end}) made={made} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
