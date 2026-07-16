"""efficientocf_gt(raw)에서 필터판 파생 — 생성소멸 제거 + 사람 제거.

입력: segmentation_instance3d(D)/segmentation_aabb(E)/segmentation_rot(F) 세 캐시만 사용.
규칙 (윈도우별, 재계산 없음·row 필터만·id 재부여 없음):
  - 생성소멸: 관측창(frame 0..time_receptive_field-1)의 D에 복셀이 1개도 없는 인스턴스는
    윈도우 전체에서 제거 (D기준 == A기준: 사람제거 후 실질 차이 0% 실측, 2026-07-12)
  - 사람 제거: cls==7 인스턴스 전체 제거 (CW도 7이므로 함께 제거됨)
출력: <out>/GMO/{segmentation_instance3d,segmentation_aabb,segmentation_rot}/<key>.npz
      (동일 폴더명/npz key — 배선은 config 경로 교체만)

사용: python tools/gen_data/derive_filtered_gt.py --start 0 --end 15000 [--src ...] [--out ...]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_bbox_gt_v2 import atomic_savez

LAYERS = (
    ("segmentation_instance3d", "segmentation_instance_saved_list2"),
    ("segmentation_aabb", "segmentation_aabb_saved_list2"),
    ("segmentation_rot", "segmentation_rot_saved_list2"),
)


def load_frames(path, npz_key):
    with np.load(path, allow_pickle=True) as z:
        return [np.asarray(f, dtype=np.int32).reshape(-1, 5) for f in z[npz_key]]


def save_frames(path_no_ext, frames, npz_key):
    arr = np.empty(len(frames), dtype=object)
    arr[:] = [np.ascontiguousarray(f, dtype=np.int32) for f in frames]
    atomic_savez(path_no_ext, **{npz_key: arr})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/hwanhee/datasets/efficientocf_gt")
    ap.add_argument("--out", default="/home/hwanhee/datasets/efficientocf_gt_f3nh")
    ap.add_argument("--time-receptive-field", type=int, default=3)
    ap.add_argument("--require", choices=["past1", "past3all", "all7"], default="past1",
                    help="past1: 관측창 D-등장 1프레임 이상 / past3all: 관측창(0..trf-1) 전 프레임 D-등장(교집합) / all7: 시퀀스 전 프레임 D-등장")
    ap.add_argument("--keep-human", action="store_true",
                    help="사람(cls=7) 제거 생략 — 사람 필터는 로더 exclude_occ_class_ids=(7,)로 런타임 처리")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    src = {name: os.path.join(args.src, "GMO", name) for name, _ in LAYERS}
    dst = {name: os.path.join(args.out, "GMO", name) for name, _ in LAYERS}
    for d in dst.values():
        os.makedirs(d, exist_ok=True)

    keys = sorted(f[:-4] for f in os.listdir(src["segmentation_aabb"]))
    end = len(keys) if args.end < 0 else min(args.end, len(keys))
    made = skipped = 0
    for key in keys[args.start:end]:
        if all(os.path.exists(os.path.join(d, key + ".npz")) for d in dst.values()):
            skipped += 1
            continue
        frames = {name: load_frames(os.path.join(src[name], key + ".npz"), k) for name, k in LAYERS}
        D = frames["segmentation_instance3d"]

        human = set()
        pres = {}
        for t, fr in enumerate(D):
            for c, i in fr[:, 3:5].tolist():
                if c == 7:
                    human.add(int(i))
                pres.setdefault(int(i), set()).add(t)
        if args.require == "all7":
            allf = set(range(len(D)))
            base = {i for i, ts in pres.items() if ts == allf}
        elif args.require == "past3all":
            past = set(range(min(args.time_receptive_field, len(D))))
            base = {i for i, ts in pres.items() if past <= ts}
        else:
            past = set(range(min(args.time_receptive_field, len(D))))
            base = {i for i, ts in pres.items() if ts & past}
        keep = base if args.keep_human else (base - human)

        for name, k in LAYERS:
            out_frames = []
            for fr in frames[name]:
                if len(fr) == 0:
                    out_frames.append(fr)
                    continue
                out_frames.append(fr[np.isin(fr[:, 4], list(keep))] if keep else fr[:0])
            save_frames(os.path.join(dst[name], key), out_frames, k)
        made += 1
        if made % 500 == 0:
            print(f"[derive_filtered] {made} made, {skipped} skipped", flush=True)
    print(f"[derive_filtered] done range=({args.start},{end}) made={made} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
