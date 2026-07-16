"""D/E/F 재생성 (class-match 수정판) — compute_D의 도로/건물 voxel 흡수 버그 수정.

배경 (CHANGELOG 2026-07-13): 기존 compute_D는 AABB 안에서 "아무거나 점유된" voxel을 전부
keep해서(occ = dense!=0 & dense!=255), 도로(11)/건물(15) 등 이물질 voxel이 인스턴스 id를 달고
D(inst3d)에 들어감 — 실측 sample_04에서 D voxel의 21%가 이물질(도로 8%+기타 13%).
수정: 점유 class가 그 row의 class와 일치할 때만 keep (dense[x,y,z] == row.cls).

기존 2단계 파이프라인(gen_new_gt_pipeline.py → derive_filtered_gt.py)을 한 번에 수행:
  - A/B는 재생성하지 않고 --src(efficientocf_gt)에 저장된 것을 로드 (rasterize 스킵,
    instance id가 A/B에 이미 박혀있어 id 재부여 없음 = id 얼라인 자동 보장)
  - C(점유 dense)만 재계산 → D_new = A ∩(class-match) C → E/F = A/B를 새 D로 게이트
  - f3 필터(derive_filtered_gt.py의 past3all + keep-human과 동일 의미) 적용
  - D/E/F 3개 레이어만 --out에 저장 (A/B 미저장 — 디스크 절약, _gt 것 재사용)

사용:
  python tools/gen_data/regen_def_clsmatch.py --split test --start 0 --limit 8 --out <테스트루트>
  python tools/gen_data/regen_def_clsmatch.py --split train --start 0 --end 6000 --out <새루트>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_bbox_gt_v2 import atomic_savez
from gen_new_gt_pipeline import make_record_instance_raw, get_seq_occ_dense, apply_gate

LAYERS = (
    ("segmentation_instance3d", "segmentation_instance_saved_list2"),
    ("segmentation_aabb", "segmentation_aabb_saved_list2"),
    ("segmentation_rot", "segmentation_rot_saved_list2"),
)
SRC_RAW = (
    ("bbox_aabb_raw", "bbox_aabb_raw_saved_list2"),
    ("bbox_rot_raw", "bbox_rot_raw_saved_list2"),
)


def load_frames(path, npz_key):
    with np.load(path, allow_pickle=True) as z:
        return [np.asarray(f, dtype=np.int64).reshape(-1, 5) for f in z[npz_key]]


def compute_D_clsmatch(aabb_frames, dense_frames):
    """D = A ∩ C(class 일치만), first-wins dedup(기존 결정② 유지).
    dense에는 class 라벨이 들어있음(0=빈칸, 255=noise) — cls(2~10)와 같을 수 없어
    기존의 !=0/!=255 체크는 == 비교에 자동 포함됨."""
    out = []
    for rows, dense in zip(aabb_frames, dense_frames):
        if rows.shape[0] == 0:
            out.append(np.zeros((0, 5), np.int32))
            continue
        keep = rows[dense[rows[:, 0], rows[:, 1], rows[:, 2]] == rows[:, 3]]
        if keep.shape[0] == 0:
            out.append(np.zeros((0, 5), np.int32))
            continue
        enc = (keep[:, 0].astype(np.int64) * 512 + keep[:, 1]) * 64 + keep[:, 2]
        _, first_idx = np.unique(enc, return_index=True)
        out.append(np.ascontiguousarray(keep[np.sort(first_idx)]).astype(np.int32))
    return out


def apply_f3_filter(layer_frames_dict, D_frames, time_receptive_field=3):
    """derive_filtered_gt.py --require past3all --keep-human과 동일 의미:
    관측창(0..trf-1) 전 프레임에 D voxel이 있는 인스턴스만 윈도우 전체에서 유지, 사람(7)도 유지."""
    pres = {}
    for t, fr in enumerate(D_frames):
        for i in (np.unique(fr[:, 4]).tolist() if fr.shape[0] else []):
            pres.setdefault(int(i), set()).add(t)
    past = set(range(min(time_receptive_field, len(D_frames))))
    keep = {i for i, ts in pres.items() if past <= ts}
    out = {}
    for name, frames in layer_frames_dict.items():
        filtered = []
        for fr in frames:
            if fr.shape[0] == 0:
                filtered.append(fr)
                continue
            filtered.append(fr[np.isin(fr[:, 4], list(keep))] if keep else fr[:0])
        out[name] = filtered
    return out


def save_npz(out_dir, key, frames, npz_key):
    arr = np.empty(len(frames), dtype=object)  # collapse 방어
    arr[:] = [np.ascontiguousarray(f, dtype=np.int32) for f in frames]
    atomic_savez(os.path.join(out_dir, key), **{npz_key: arr})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="projects/configs/baselines/subset_attn_cover_pyr_aabb_dice3d_rot.py")
    ap.add_argument("--src", default="/home/hwanhee/datasets/efficientocf_gt",
                    help="A/B(raw 레이어)가 저장된 기존 루트")
    ap.add_argument("--out", default="/home/hwanhee/datasets/efficientocf_gt_f3_clsmatch")
    ap.add_argument("--occ-path", default="./data/nuScenes-Occupancy")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--no-f3-filter", action="store_true",
                    help="f3 필터(past3all) 생략 — raw D/E/F 생성 모드 (_gt 편입용)")
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
    # 전 키 커버 보장: subset config의 train_capacity(예: 4000)를 풀로 강제
    split_cfg.train_capacity = 23930
    split_cfg.test_capacity = 5119
    dataset = build_dataset(split_cfg)
    dataset.classes = ["vehicle", "human"]
    dataset.record_instance = make_record_instance_raw(dataset)

    pc_range = list(cfg.point_cloud_range)
    grid_size = list(cfg.occ_size)

    src_dirs = {name: os.path.join(args.src, "GMO", name) for name, _ in SRC_RAW}
    dst_dirs = {name: os.path.join(args.out, "GMO", name) for name, _ in LAYERS}
    for d in dst_dirs.values():
        os.makedirs(d, exist_ok=True)

    n_total = len(dataset.indices)
    end = n_total if args.end < 0 else min(args.end, n_total)
    made = skipped = missing_src = 0
    for i in range(args.start, end):
        dataset.egopose_list = []
        dataset.ego2lidar_list = []
        dataset.visible_instance_set = set()
        dataset.instance_dict = {}
        seq = dataset.prepare_test_data(i)
        if seq is None:
            continue
        key = dataset.present_scene_lidar_token
        if all(os.path.exists(os.path.join(d, key + ".npz")) for d in dst_dirs.values()):
            skipped += 1
            continue

        src_paths = {name: os.path.join(src_dirs[name], key + ".npz") for name, _ in SRC_RAW}
        if not all(os.path.exists(p) for p in src_paths.values()):
            missing_src += 1
            continue
        A = load_frames(src_paths["bbox_aabb_raw"], SRC_RAW[0][1])
        B = load_frames(src_paths["bbox_rot_raw"], SRC_RAW[1][1])

        dense = get_seq_occ_dense(seq, args.occ_path, pc_range, grid_size, nb_process_label)
        D = compute_D_clsmatch(A, dense)
        E = apply_gate(A, D)
        F = apply_gate(B, D)
        layers = {"segmentation_instance3d": D, "segmentation_aabb": E, "segmentation_rot": F}
        if args.no_f3_filter:
            filtered = layers
        else:
            filtered = apply_f3_filter(
                layers, D, time_receptive_field=int(cfg.time_receptive_field),
            )

        for name, npz_key in LAYERS:
            save_npz(dst_dirs[name], key, filtered[name], npz_key)

        made += 1
        if made % 50 == 0:
            print(f"[regen_clsmatch] {i + 1}/{end} made={made} skipped={skipped} missing_src={missing_src}", flush=True)
        if args.limit > 0 and made >= args.limit:
            break
    print(f"[regen_clsmatch] done range=({args.start},{end}) made={made} skipped={skipped} missing_src={missing_src}", flush=True)


if __name__ == "__main__":
    main()
