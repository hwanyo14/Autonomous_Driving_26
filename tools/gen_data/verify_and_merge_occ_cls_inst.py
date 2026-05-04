#!/usr/bin/env python3
"""Verify voxel-perfect alignment and merge sparse class/instance occupancy labels.

This script compares two cached sparse voxel datasets:

1. `segmentation_instance3d`: rows shaped like `[x, y, z, ..., inst_id]`
2. `segmentation`: rows shaped like `[x, y, z, cls_value]`

It follows the sparse cache conventions used in:
- `projects/occ_plugin/datasets/pipelines/loading_instance.py`
- `projects/occ_plugin/datasets/pipelines/loading_occupancy.py`

For every shared `.npz` window file, the script checks:
- same filename set
- same sequence length
- no duplicate voxel coordinates inside each frame
- no out-of-bound voxel coordinates
- exact voxel-coordinate equality after lexicographic sorting

If and only if a file passes all checks, it can also write merged sparse rows:
- `[x, y, z, cls_value, inst_id]`

For visualization, it can export 2D BEV images and optional ASCII `.ply`
point clouds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


DEFAULT_INST_DIR = "/home/hwanhee/dataset/efficientocf/GMO/segmentation_instance3d"
DEFAULT_CLS_DIR = "/home/hwanhee/dataset/efficientocf_bboxcls/GMO/segmentation"
DEFAULT_OUTPUT_DIR = "work_dirs/occ_cls_inst_verify"
DEFAULT_MERGED_KEY = "segmentation_cls_instance_saved_list2"
DEFAULT_BG_RGB = np.asarray([0, 0, 0], dtype=np.uint8)
DEFAULT_OCC_RGB = np.asarray([60, 60, 60], dtype=np.uint8)
INST_ONLY_RGB = np.asarray([255, 64, 64], dtype=np.uint8)
CLS_ONLY_RGB = np.asarray([64, 128, 255], dtype=np.uint8)
BOTH_MISMATCH_RGB = np.asarray([255, 230, 80], dtype=np.uint8)


CLASS_PALETTE = np.asarray(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
        [128, 128, 0],
        [255, 215, 180],
        [0, 0, 128],
        [128, 128, 128],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify voxel-perfect alignment between class and instance sparse occupancy caches."
    )
    parser.add_argument("--inst-dir", default=DEFAULT_INST_DIR, help="Directory with instance `.npz` files.")
    parser.add_argument("--cls-dir", default=DEFAULT_CLS_DIR, help="Directory with class `.npz` files.")
    parser.add_argument(
        "--inst-key",
        default="segmentation_instance_saved_list2",
        help="NPZ key for instance sparse lists.",
    )
    parser.add_argument(
        "--cls-key",
        default="segmentation_saved_list2",
        help="NPZ key for class sparse lists.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=3,
        default=[512, 512, 40],
        metavar=("X", "Y", "Z"),
        help="Voxel grid size used by the loaders.",
    )
    parser.add_argument(
        "--pc-range",
        type=float,
        nargs=6,
        default=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Point-cloud range used to convert voxel centers into world coordinates for optional PLY export.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for reports and visualization artifacts.",
    )
    parser.add_argument(
        "--write-merged",
        action="store_true",
        help="Write merged sparse `.npz` files for windows that pass the exact-match check.",
    )
    parser.add_argument(
        "--merged-out-dir",
        default=None,
        help="Output directory for merged `.npz` files. Defaults to `<output-dir>/merged_npz`.",
    )
    parser.add_argument(
        "--merged-key",
        default=DEFAULT_MERGED_KEY,
        help="NPZ key for merged sparse lists.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Export 2D BEV image visualizations for selected files.",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Also export optional 3D `.ply` point clouds alongside BEV images.",
    )
    parser.add_argument(
        "--vis-files",
        nargs="*",
        default=[],
        help="Exact `.npz` basenames to visualize. If empty, the first checked file is used.",
    )
    parser.add_argument(
        "--vis-frames",
        type=int,
        nargs="*",
        default=[0],
        help="Frame indices to visualize for each selected file.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Exact `.npz` basenames to check. If empty, checks the full shared set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Check only the first N shared files after sorting.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failing file.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Progress print interval while scanning many files.",
    )
    parser.add_argument(
        "--max-bad-reports",
        type=int,
        default=20,
        help="Maximum number of failing file reports to keep in the summary JSON.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_sparse_sequence(npz_path: Path, key: str) -> List[Any]:
    with np.load(npz_path, allow_pickle=True) as f:
        if key in f.files:
            data = f[key]
        elif "arr_0" in f.files:
            data = f["arr_0"]
        else:
            raise KeyError(f"{npz_path} does not contain key `{key}` or fallback `arr_0`.")

    if isinstance(data, np.ndarray) and data.dtype == object:
        return data.tolist()
    return list(data)


def normalize_sparse_rows(rows: Any, min_cols: int, path_label: str) -> np.ndarray:
    arr = np.asarray(rows)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.size == 0:
            arr = np.zeros((0, min_cols), dtype=np.int64)
        else:
            arr = np.vstack(arr)
    arr = np.asarray(arr)

    if arr.size == 0:
        return np.zeros((0, min_cols), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] < min_cols:
        raise ValueError(f"{path_label} row shape is invalid: {arr.shape}")

    return np.ascontiguousarray(arr)


def sort_rows_by_xyz(rows: np.ndarray) -> np.ndarray:
    if rows.shape[0] == 0:
        return rows
    xyz = rows[:, :3].astype(np.int64, copy=False)
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    return np.ascontiguousarray(rows[order])


def find_out_of_bounds(coords: np.ndarray, grid_size: np.ndarray) -> np.ndarray:
    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)

    invalid_mask = (
        (coords[:, 0] < 0)
        | (coords[:, 0] >= grid_size[0])
        | (coords[:, 1] < 0)
        | (coords[:, 1] >= grid_size[1])
        | (coords[:, 2] < 0)
        | (coords[:, 2] >= grid_size[2])
    )
    return coords[invalid_mask]


def find_duplicate_coords(sorted_coords: np.ndarray) -> np.ndarray:
    if sorted_coords.shape[0] <= 1:
        return np.zeros((0, 3), dtype=np.int64)

    dup_mask = np.all(sorted_coords[1:] == sorted_coords[:-1], axis=1)
    if not dup_mask.any():
        return np.zeros((0, 3), dtype=np.int64)

    dup_coords = sorted_coords[1:][dup_mask]
    unique_mask = np.ones(len(dup_coords), dtype=bool)
    if len(dup_coords) > 1:
        unique_mask[1:] = np.any(dup_coords[1:] != dup_coords[:-1], axis=1)
    return dup_coords[unique_mask]


def coords_to_struct(coords: np.ndarray) -> np.ndarray:
    coords = np.ascontiguousarray(coords.astype(np.int32, copy=False))
    dtype = np.dtype([("x", np.int32), ("y", np.int32), ("z", np.int32)])
    return coords.view(dtype).reshape(-1)


def struct_to_coords(struct_arr: np.ndarray) -> np.ndarray:
    if struct_arr.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return struct_arr.view(np.int32).reshape(-1, 3)


def compute_coord_diffs(inst_xyz_sorted: np.ndarray, cls_xyz_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inst_struct = coords_to_struct(inst_xyz_sorted)
    cls_struct = coords_to_struct(cls_xyz_sorted)
    inst_only = np.setdiff1d(inst_struct, cls_struct, assume_unique=False)
    cls_only = np.setdiff1d(cls_struct, inst_struct, assume_unique=False)
    return struct_to_coords(inst_only), struct_to_coords(cls_only)


def merge_sorted_rows(inst_rows_sorted: np.ndarray, cls_rows_sorted: np.ndarray) -> np.ndarray:
    merged_dtype = np.result_type(inst_rows_sorted.dtype, cls_rows_sorted.dtype, np.int64)
    merged = np.empty((inst_rows_sorted.shape[0], 5), dtype=merged_dtype)
    merged[:, :3] = inst_rows_sorted[:, :3]
    merged[:, 3] = cls_rows_sorted[:, -1]
    merged[:, 4] = inst_rows_sorted[:, -1]
    return merged


def compare_frame(
    inst_rows: Any,
    cls_rows: Any,
    grid_size: np.ndarray,
    file_name: str,
    frame_idx: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    frame_name = f"{file_name}:frame_{frame_idx:02d}"
    inst = normalize_sparse_rows(inst_rows, min_cols=4, path_label=f"{frame_name}:inst")
    cls = normalize_sparse_rows(cls_rows, min_cols=4, path_label=f"{frame_name}:cls")

    inst_sorted = sort_rows_by_xyz(inst)
    cls_sorted = sort_rows_by_xyz(cls)

    inst_xyz = inst_sorted[:, :3].astype(np.int64, copy=False)
    cls_xyz = cls_sorted[:, :3].astype(np.int64, copy=False)

    inst_oob = find_out_of_bounds(inst_xyz, grid_size)
    cls_oob = find_out_of_bounds(cls_xyz, grid_size)
    inst_dup = find_duplicate_coords(inst_xyz)
    cls_dup = find_duplicate_coords(cls_xyz)

    report: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "inst_points": int(inst_sorted.shape[0]),
        "cls_points": int(cls_sorted.shape[0]),
        "inst_duplicate_coords": int(inst_dup.shape[0]),
        "cls_duplicate_coords": int(cls_dup.shape[0]),
        "inst_out_of_bounds": int(inst_oob.shape[0]),
        "cls_out_of_bounds": int(cls_oob.shape[0]),
        "coord_match": False,
        "status": "failed",
    }

    if inst_dup.shape[0] > 0:
        report["inst_duplicate_preview"] = inst_dup[:10].tolist()
    if cls_dup.shape[0] > 0:
        report["cls_duplicate_preview"] = cls_dup[:10].tolist()
    if inst_oob.shape[0] > 0:
        report["inst_out_of_bounds_preview"] = inst_oob[:10].tolist()
    if cls_oob.shape[0] > 0:
        report["cls_out_of_bounds_preview"] = cls_oob[:10].tolist()

    if inst_dup.shape[0] > 0 or cls_dup.shape[0] > 0 or inst_oob.shape[0] > 0 or cls_oob.shape[0] > 0:
        return None, report

    if inst_sorted.shape[0] != cls_sorted.shape[0] or not np.array_equal(inst_xyz, cls_xyz):
        inst_only, cls_only = compute_coord_diffs(inst_xyz, cls_xyz)
        report["inst_only_coords"] = int(inst_only.shape[0])
        report["cls_only_coords"] = int(cls_only.shape[0])
        if inst_only.shape[0] > 0:
            report["inst_only_preview"] = inst_only[:10].tolist()
        if cls_only.shape[0] > 0:
            report["cls_only_preview"] = cls_only[:10].tolist()
        return None, report

    merged = merge_sorted_rows(inst_sorted, cls_sorted)
    report["coord_match"] = True
    report["status"] = "ok"
    return merged, report


def compare_file(
    inst_path: Path,
    cls_path: Path,
    inst_key: str,
    cls_key: str,
    grid_size: np.ndarray,
) -> Tuple[bool, List[np.ndarray], Dict[str, Any]]:
    inst_seq = load_sparse_sequence(inst_path, inst_key)
    cls_seq = load_sparse_sequence(cls_path, cls_key)

    report: Dict[str, Any] = {
        "file": inst_path.name,
        "inst_path": str(inst_path),
        "cls_path": str(cls_path),
        "inst_frames": int(len(inst_seq)),
        "cls_frames": int(len(cls_seq)),
        "frame_reports": [],
        "status": "failed",
    }

    if len(inst_seq) != len(cls_seq):
        report["reason"] = "sequence_length_mismatch"
        return False, [], report

    merged_frames: List[np.ndarray] = []
    ok = True
    for frame_idx, (inst_rows, cls_rows) in enumerate(zip(inst_seq, cls_seq)):
        merged_frame, frame_report = compare_frame(
            inst_rows=inst_rows,
            cls_rows=cls_rows,
            grid_size=grid_size,
            file_name=inst_path.name,
            frame_idx=frame_idx,
        )
        report["frame_reports"].append(frame_report)
        if merged_frame is None:
            ok = False
        else:
            merged_frames.append(merged_frame)

    if ok and len(merged_frames) != len(inst_seq):
        report["reason"] = "internal_merged_frame_count_mismatch"
        return False, [], report

    report["status"] = "ok" if ok else "failed"
    if not ok and "reason" not in report:
        report["reason"] = "frame_check_failed"
    return ok, merged_frames, report


def save_merged_npz(out_path: Path, merged_frames: Sequence[np.ndarray], merged_key: str) -> None:
    ensure_dir(out_path.parent)
    obj_arr = np.array([np.asarray(frame) for frame in merged_frames], dtype=object)
    np.savez_compressed(out_path, **{merged_key: obj_arr})


def voxel_centers_to_world(coords_xyz: np.ndarray, grid_size: np.ndarray, pc_range: np.ndarray) -> np.ndarray:
    voxel_size = (pc_range[3:] - pc_range[:3]) / grid_size.astype(np.float64)
    return (coords_xyz.astype(np.float64) + 0.5) * voxel_size[None, :] + pc_range[:3][None, :]


def xy_to_image_rc(xy_coords: np.ndarray, grid_size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cols = xy_coords[:, 0].astype(np.int64, copy=False)
    rows = (int(grid_size[1]) - 1) - xy_coords[:, 1].astype(np.int64, copy=False)
    return rows, cols


def project_top_rows_by_xy(rows: np.ndarray) -> np.ndarray:
    if rows.shape[0] == 0:
        return rows

    xyz = rows[:, :3].astype(np.int64, copy=False)
    order = np.lexsort((-xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    xy_sorted = xyz[order, :2]
    keep = np.ones(order.shape[0], dtype=bool)
    if order.shape[0] > 1:
        keep[1:] = np.any(xy_sorted[1:] != xy_sorted[:-1], axis=1)
    return np.ascontiguousarray(rows[order[keep]])


def build_projected_mask(rows: np.ndarray, grid_size: np.ndarray) -> np.ndarray:
    bev = np.zeros((int(grid_size[1]), int(grid_size[0])), dtype=bool)
    if rows.shape[0] == 0:
        return bev

    top_rows = project_top_rows_by_xy(rows)
    rr, cc = xy_to_image_rc(top_rows[:, :2], grid_size)
    bev[rr, cc] = True
    return bev


def build_top_label_map(rows: np.ndarray, grid_size: np.ndarray, label_col: int) -> Tuple[np.ndarray, np.ndarray]:
    height = int(grid_size[1])
    width = int(grid_size[0])
    if rows.shape[0] == 0:
        return np.zeros((height, width), dtype=np.int64), np.full((height, width), -1, dtype=np.int16)

    top_rows = project_top_rows_by_xy(rows)
    label_values = np.asarray(top_rows[:, label_col])
    label_map = np.zeros((height, width), dtype=label_values.dtype)
    z_map = np.full((height, width), -1, dtype=np.int16)

    rr, cc = xy_to_image_rc(top_rows[:, :2], grid_size)
    label_map[rr, cc] = label_values
    z_map[rr, cc] = top_rows[:, 2].astype(np.int16, copy=False)
    return label_map, z_map


def hash_ids_to_rgb(ids: np.ndarray) -> np.ndarray:
    ids = ids.astype(np.uint64, copy=False)
    colors = np.empty((ids.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = ((ids * 37) % 251).astype(np.uint8)
    colors[:, 1] = ((ids * 67 + 29) % 253).astype(np.uint8)
    colors[:, 2] = ((ids * 97 + 71) % 255).astype(np.uint8)
    zero_mask = ids == 0
    colors[zero_mask] = np.asarray([90, 90, 90], dtype=np.uint8)
    return colors


def class_values_to_rgb(cls_values: np.ndarray) -> np.ndarray:
    if cls_values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    if np.issubdtype(cls_values.dtype, np.integer):
        cls_idx = cls_values.astype(np.int64, copy=False)
        cls_idx = np.mod(cls_idx, CLASS_PALETTE.shape[0])
        return CLASS_PALETTE[cls_idx]

    finite = np.isfinite(cls_values)
    colors = np.zeros((cls_values.shape[0], 3), dtype=np.uint8)
    if not finite.any():
        return colors

    v = cls_values.astype(np.float64, copy=False)
    vmin = float(v[finite].min())
    vmax = float(v[finite].max())
    if vmax <= vmin:
        scaled = np.zeros_like(v, dtype=np.float64)
    else:
        scaled = (v - vmin) / (vmax - vmin)

    colors[:, 0] = np.clip(255.0 * scaled, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(255.0 * (1.0 - np.abs(2.0 * scaled - 1.0)), 0, 255).astype(np.uint8)
    colors[:, 2] = np.clip(255.0 * (1.0 - scaled), 0, 255).astype(np.uint8)
    colors[~finite] = np.asarray([255, 255, 255], dtype=np.uint8)
    return colors


def label_map_to_rgb(label_map: np.ndarray, color_fn) -> np.ndarray:
    flat_colors = color_fn(label_map.reshape(-1))
    return flat_colors.reshape(label_map.shape[0], label_map.shape[1], 3)


def save_rgb_image(out_path: Path, rgb: np.ndarray) -> Path:
    ensure_dir(out_path.parent)
    rgb = np.ascontiguousarray(rgb.astype(np.uint8, copy=False))

    if Image is not None:
        Image.fromarray(rgb).save(out_path)
        return out_path

    ppm_path = out_path.with_suffix(".ppm")
    with ppm_path.open("wb") as f:
        f.write(f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())
    return ppm_path


def save_merged_bev_images(out_dir: Path, frame_idx: int, merged_frame: np.ndarray, grid_size: np.ndarray) -> None:
    class_map, _ = build_top_label_map(merged_frame, grid_size=grid_size, label_col=3)
    inst_map, _ = build_top_label_map(merged_frame, grid_size=grid_size, label_col=4)

    save_rgb_image(
        out_dir / f"frame_{frame_idx:02d}_bev_class.png",
        label_map_to_rgb(class_map, class_values_to_rgb),
    )
    save_rgb_image(
        out_dir / f"frame_{frame_idx:02d}_bev_instance.png",
        label_map_to_rgb(inst_map, hash_ids_to_rgb),
    )


def save_mismatch_bev_images(
    out_dir: Path,
    frame_idx: int,
    inst_rows: np.ndarray,
    cls_rows: np.ndarray,
    inst_only: np.ndarray,
    cls_only: np.ndarray,
    grid_size: np.ndarray,
) -> None:
    inst_map, _ = build_top_label_map(inst_rows, grid_size=grid_size, label_col=-1)
    cls_map, _ = build_top_label_map(cls_rows, grid_size=grid_size, label_col=-1)

    save_rgb_image(
        out_dir / f"frame_{frame_idx:02d}_bev_instance.png",
        label_map_to_rgb(inst_map, hash_ids_to_rgb),
    )
    save_rgb_image(
        out_dir / f"frame_{frame_idx:02d}_bev_class.png",
        label_map_to_rgb(cls_map, class_values_to_rgb),
    )

    inst_bev = build_projected_mask(inst_rows, grid_size)
    cls_bev = build_projected_mask(cls_rows, grid_size)
    inst_only_bev = build_projected_mask(inst_only, grid_size)
    cls_only_bev = build_projected_mask(cls_only, grid_size)

    mismatch_rgb = np.zeros((int(grid_size[1]), int(grid_size[0]), 3), dtype=np.uint8)
    mismatch_rgb[:] = DEFAULT_BG_RGB
    mismatch_rgb[inst_bev & cls_bev] = DEFAULT_OCC_RGB
    mismatch_rgb[inst_only_bev] = INST_ONLY_RGB
    mismatch_rgb[cls_only_bev] = CLS_ONLY_RGB
    mismatch_rgb[inst_only_bev & cls_only_bev] = BOTH_MISMATCH_RGB

    save_rgb_image(out_dir / f"frame_{frame_idx:02d}_bev_mismatch.png", mismatch_rgb)


def write_ascii_ply(
    out_path: Path,
    points_xyz_world: np.ndarray,
    colors_rgb: np.ndarray,
    cls_values: Optional[np.ndarray] = None,
    inst_ids: Optional[np.ndarray] = None,
) -> None:
    ensure_dir(out_path.parent)
    n_points = int(points_xyz_world.shape[0])

    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if cls_values is not None:
            f.write("property float cls_value\n")
        if inst_ids is not None:
            f.write("property int inst_id\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for idx in range(n_points):
            x, y, z = points_xyz_world[idx]
            parts = [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
            if cls_values is not None:
                parts.append(f"{float(cls_values[idx]):.6f}")
            if inst_ids is not None:
                parts.append(str(int(inst_ids[idx])))
            r, g, b = colors_rgb[idx]
            parts.extend([str(int(r)), str(int(g)), str(int(b))])
            f.write(" ".join(parts) + "\n")


def export_visualizations(
    inst_path: Path,
    cls_path: Path,
    vis_root: Path,
    vis_frames: Sequence[int],
    inst_key: str,
    cls_key: str,
    grid_size: np.ndarray,
    pc_range: np.ndarray,
    save_ply: bool,
) -> None:
    inst_seq = load_sparse_sequence(inst_path, inst_key)
    cls_seq = load_sparse_sequence(cls_path, cls_key)
    valid_frames = [idx for idx in vis_frames if 0 <= idx < min(len(inst_seq), len(cls_seq))]

    file_stem = inst_path.stem
    out_dir = vis_root / file_stem
    ensure_dir(out_dir)

    for frame_idx in valid_frames:
        merged_frame, frame_report = compare_frame(
            inst_rows=inst_seq[frame_idx],
            cls_rows=cls_seq[frame_idx],
            grid_size=grid_size,
            file_name=inst_path.name,
            frame_idx=frame_idx,
        )

        report_path = out_dir / f"frame_{frame_idx:02d}_report.json"
        report_path.write_text(json.dumps(frame_report, indent=2), encoding="utf-8")

        if merged_frame is not None:
            save_merged_bev_images(out_dir, frame_idx=frame_idx, merged_frame=merged_frame, grid_size=grid_size)
            coords_world = voxel_centers_to_world(merged_frame[:, :3], grid_size=grid_size, pc_range=pc_range)
            cls_values = merged_frame[:, 3]
            inst_ids = merged_frame[:, 4]

            if save_ply:
                write_ascii_ply(
                    out_dir / f"frame_{frame_idx:02d}_merged_by_class.ply",
                    points_xyz_world=coords_world,
                    colors_rgb=class_values_to_rgb(cls_values),
                    cls_values=cls_values,
                    inst_ids=inst_ids,
                )
                write_ascii_ply(
                    out_dir / f"frame_{frame_idx:02d}_merged_by_instance.ply",
                    points_xyz_world=coords_world,
                    colors_rgb=hash_ids_to_rgb(inst_ids),
                    cls_values=cls_values,
                    inst_ids=inst_ids,
                )
            np.save(out_dir / f"frame_{frame_idx:02d}_merged.npy", merged_frame)
            continue

        inst_rows = sort_rows_by_xyz(normalize_sparse_rows(inst_seq[frame_idx], 4, f"{inst_path.name}:inst"))
        cls_rows = sort_rows_by_xyz(normalize_sparse_rows(cls_seq[frame_idx], 4, f"{cls_path.name}:cls"))
        inst_only, cls_only = compute_coord_diffs(
            inst_rows[:, :3].astype(np.int64, copy=False),
            cls_rows[:, :3].astype(np.int64, copy=False),
        )

        save_mismatch_bev_images(
            out_dir=out_dir,
            frame_idx=frame_idx,
            inst_rows=inst_rows,
            cls_rows=cls_rows,
            inst_only=inst_only,
            cls_only=cls_only,
            grid_size=grid_size,
        )

        if save_ply and inst_only.shape[0] > 0:
            inst_world = voxel_centers_to_world(inst_only, grid_size=grid_size, pc_range=pc_range)
            inst_color = np.repeat(INST_ONLY_RGB[None, :], inst_only.shape[0], axis=0)
            write_ascii_ply(
                out_dir / f"frame_{frame_idx:02d}_inst_only.ply",
                points_xyz_world=inst_world,
                colors_rgb=inst_color,
            )

        if save_ply and cls_only.shape[0] > 0:
            cls_world = voxel_centers_to_world(cls_only, grid_size=grid_size, pc_range=pc_range)
            cls_color = np.repeat(CLS_ONLY_RGB[None, :], cls_only.shape[0], axis=0)
            write_ascii_ply(
                out_dir / f"frame_{frame_idx:02d}_cls_only.ply",
                points_xyz_world=cls_world,
                colors_rgb=cls_color,
            )


def resolve_files(inst_dir: Path, cls_dir: Path, exact_files: Sequence[str], limit: Optional[int]) -> List[str]:
    inst_files = {p.name for p in inst_dir.glob("*.npz")}
    cls_files = {p.name for p in cls_dir.glob("*.npz")}
    shared = sorted(inst_files & cls_files)

    if exact_files:
        missing = [name for name in exact_files if name not in inst_files or name not in cls_files]
        if missing:
            raise FileNotFoundError(f"Requested files are missing from one of the directories: {missing}")
        shared = list(exact_files)

    if limit is not None:
        shared = shared[:limit]
    return shared


def to_serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {k: v for k, v in vars(args).items()}


def main() -> int:
    args = parse_args()

    inst_dir = Path(args.inst_dir)
    cls_dir = Path(args.cls_dir)
    output_dir = Path(args.output_dir)
    merged_out_dir = Path(args.merged_out_dir) if args.merged_out_dir else output_dir / "merged_npz"
    vis_root = output_dir / "vis"
    ensure_dir(output_dir)

    if not inst_dir.is_dir():
        raise FileNotFoundError(f"Instance directory does not exist: {inst_dir}")
    if not cls_dir.is_dir():
        raise FileNotFoundError(f"Class directory does not exist: {cls_dir}")

    grid_size = np.asarray(args.grid_size, dtype=np.int64)
    pc_range = np.asarray(args.pc_range, dtype=np.float64)

    inst_all = {p.name for p in inst_dir.glob("*.npz")}
    cls_all = {p.name for p in cls_dir.glob("*.npz")}
    shared_all = sorted(inst_all & cls_all)
    inst_only_all = sorted(inst_all - cls_all)
    cls_only_all = sorted(cls_all - inst_all)
    target_files = resolve_files(inst_dir, cls_dir, exact_files=args.files, limit=args.limit)
    vis_files = set(args.vis_files)
    if args.save_vis and not vis_files and target_files:
        vis_files.add(target_files[0])

    summary: Dict[str, Any] = {
        "args": to_serializable_args(args),
        "inst_dir": str(inst_dir),
        "cls_dir": str(cls_dir),
        "inst_files_total": int(len(inst_all)),
        "cls_files_total": int(len(cls_all)),
        "shared_files_total": int(len(shared_all)),
        "inst_only_files": int(len(inst_only_all)),
        "cls_only_files": int(len(cls_only_all)),
        "inst_only_preview": inst_only_all[:10],
        "cls_only_preview": cls_only_all[:10],
        "checked_files_requested": int(len(target_files)),
        "checked_files": 0,
        "ok_files": 0,
        "failed_files": 0,
        "written_merged_files": 0,
        "visualized_files": [],
        "bad_reports": [],
    }

    if not target_files:
        summary["status"] = "no_files_checked"
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[summary] no files to check. summary saved to {summary_path}")
        return 0

    processed_files = 0
    for file_idx, file_name in enumerate(target_files, start=1):
        inst_path = inst_dir / file_name
        cls_path = cls_dir / file_name

        processed_files += 1
        try:
            ok, merged_frames, file_report = compare_file(
                inst_path=inst_path,
                cls_path=cls_path,
                inst_key=args.inst_key,
                cls_key=args.cls_key,
                grid_size=grid_size,
            )
        except Exception as exc:
            ok = False
            merged_frames = []
            file_report = {
                "file": file_name,
                "inst_path": str(inst_path),
                "cls_path": str(cls_path),
                "status": "failed",
                "reason": "exception_while_comparing",
                "exception": repr(exc),
            }

        if ok:
            summary["ok_files"] += 1
            if args.write_merged:
                save_merged_npz(
                    out_path=merged_out_dir / file_name,
                    merged_frames=merged_frames,
                    merged_key=args.merged_key,
                )
                summary["written_merged_files"] += 1
        else:
            summary["failed_files"] += 1
            if len(summary["bad_reports"]) < args.max_bad_reports:
                summary["bad_reports"].append(file_report)

        if args.save_vis and file_name in vis_files:
            try:
                export_visualizations(
                    inst_path=inst_path,
                    cls_path=cls_path,
                    vis_root=vis_root,
                    vis_frames=args.vis_frames,
                    inst_key=args.inst_key,
                    cls_key=args.cls_key,
                    grid_size=grid_size,
                    pc_range=pc_range,
                    save_ply=bool(args.save_ply),
                )
                summary["visualized_files"].append(file_name)
            except Exception as exc:
                if len(summary["bad_reports"]) < args.max_bad_reports:
                    summary["bad_reports"].append(
                        {
                            "file": file_name,
                            "status": "failed",
                            "reason": "exception_while_visualizing",
                            "exception": repr(exc),
                        }
                    )

        if file_idx == 1 or file_idx % max(1, int(args.log_every)) == 0 or file_idx == len(target_files):
            print(
                f"[{file_idx}/{len(target_files)}] "
                f"ok={summary['ok_files']} failed={summary['failed_files']} current={file_name}"
            )

        if args.fail_fast and not ok:
            print(f"[fail-fast] stopped at {file_name}")
            break

    summary["checked_files"] = int(processed_files)
    summary["status"] = "ok" if summary["failed_files"] == 0 else "failed"
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print(f"[summary] shared_total={summary['shared_files_total']}")
    print(f"[summary] checked={summary['checked_files']}")
    print(f"[summary] ok={summary['ok_files']}")
    print(f"[summary] failed={summary['failed_files']}")
    print(f"[summary] merged_written={summary['written_merged_files']}")
    print(f"[summary] visualized={len(summary['visualized_files'])}")
    print(f"[summary] report={summary_path}")

    return 0 if summary["failed_files"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
