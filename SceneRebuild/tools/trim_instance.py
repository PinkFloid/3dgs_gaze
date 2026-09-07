#!/usr/bin/env python
"""trim_instance.py -- 把实例吸进来的支撑面/背景点按"离中心的距离"修掉。

lift 的 mask 投票常把小物旁边的桌布边、车架边框一起吸进实例(v10 苹果粉吸车架、
v11 橘子/白杯2/网球R 吸桌布边)。物体本身是紧凑的:取该实例点的 xy 中位数当中心
(比均值抗离群),离中心水平距离 > radius 的点改标签(默认并回 --to 指定的支撑面
实例,如物品台 10;--to -1 则直接删掉)。同步改写 points.npz / instances.json 里
两个实例的 n_gaussians / centroid / bbox / size(n_views、n_masks 不动)。

用法:
    python tools/trim_instance.py --seg-dir lab_result/segmentation_sam --id 62 --radius 0.06 --to 10 [--dry-run]
    可选 --center X Y 手动指定中心;--zmax 再按高度砍(比如砍掉飘在上面的点)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(p: np.ndarray) -> dict:
    p = p.astype(np.float64)
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    return {"n_gaussians": int(len(p)), "centroid": p.mean(axis=0).round(3).tolist(),
            "bbox_min": lo.round(3).tolist(), "bbox_max": hi.round(3).tolist(),
            "size_m": (hi - lo).round(2).tolist()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--radius", type=float, required=True, help="保留的水平半径 (m)")
    ap.add_argument("--center", type=float, nargs=2, default=None, help="手动中心 X Y;默认 xy 中位数")
    ap.add_argument("--zmax", type=float, default=None, help="高于此 z 的点也修掉")
    ap.add_argument("--to", type=int, default=-1, help="修掉的点并入的实例 id;-1 = 删除")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    seg = Path(a.seg_dir)
    z = np.load(seg / "points.npz")
    xyz, label = z["xyz"], z["label"].copy()
    meta = json.loads((seg / "instances.json").read_text(encoding="utf-8"))
    by_id = {int(i["id"]): i for i in meta["instances"]}
    if a.id not in by_id:
        raise SystemExit(f"id {a.id} 不在 instances.json")
    m = label == a.id
    p = xyz[m]
    c = np.array(a.center) if a.center else np.median(p[:, :2], axis=0)
    d = np.linalg.norm(p[:, :2] - c, axis=1)
    bad = d > a.radius
    if a.zmax is not None:
        bad |= p[:, 2] > a.zmax
    print(f"id {a.id}: {m.sum()} 点,中心 ({c[0]:.3f},{c[1]:.3f}),半径 {a.radius} m 外 {bad.sum()} 点"
          f"(最远 {d.max():.3f} m) -> {'删除' if a.to < 0 else f'并入 {a.to}'}")
    if bad.sum() == 0 or a.dry_run:
        print("(dry-run,未写盘)" if a.dry_run else "无需修改")
        return 0
    idx = np.flatnonzero(m)[bad]
    if a.to < 0:
        keep = np.ones(len(label), bool)
        keep[idx] = False
        xyz, label = xyz[keep], label[keep]
    else:
        label[idx] = a.to
        if a.to in by_id:
            by_id[a.to].update(stats(xyz[label == a.to]))
    by_id[a.id].update(stats(xyz[label == a.id]))
    np.savez(seg / "points.npz", xyz=xyz.astype(np.float32), label=label.astype(np.int32))
    (seg / "instances.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写回:id {a.id} 剩 {by_id[a.id]['n_gaussians']} 点 size {by_id[a.id]['size_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
