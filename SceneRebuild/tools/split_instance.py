#!/usr/bin/env python
"""split_instance.py -- 把 SAM 分割里被并成一坨的实例按几何拆开。

SAM 对"托盘/白纸上摆一组小物"常给整体 mask,lift 的部件-整体合并会把球、
水果、杯子全吸进一个实例(2026-08-02 v4 地图实测:台面静物并成 id 25)。
重跑 SAM 又慢又不保证分开;但这类场景里物体彼此不接触、只共一个支撑面——
几何拆分是确定性的:

  1. 支撑面高度 z_cut:取该实例点云 z 直方图的众数(平面点最多)+ 余量;
  2. z > z_cut 的点做连通域聚类(linkage 半径 2.5cm);
  3. 每个 >= min-pts 的簇成为新实例(id 接在现有最大 id 之后),
     余下的点(纸/托盘沿)留在原实例;
  4. 同步改写 points.npz / instances.json / names.json(新 id 补空名)。

用法:
    python tools/split_instance.py --seg-dir lab_result/segmentation_sam --id 25
    # 预览不落盘:--dry-run;支撑面手动指定:--z-cut 0.765
改完翻 preview/ 或用本脚本打印的质心表命名 names.json(每球不同名)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--id", type=int, required=True, help="要拆的实例 id")
    p.add_argument("--z-cut", type=float, default=None,
                   help="支撑面高度(米);缺省 = z 众数 + 0.008")
    p.add_argument("--linkage", type=float, default=0.025, help="连通域半径 (m)")
    p.add_argument("--min-pts", type=int, default=40, help="新实例最少点数")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def connected_components(pts: np.ndarray, r: float) -> np.ndarray:
    tree = cKDTree(pts)
    lab = np.full(len(pts), -1, dtype=np.int32)
    cur = 0
    for seed in range(len(pts)):
        if lab[seed] >= 0:
            continue
        stack = [seed]
        lab[seed] = cur
        while stack:
            nxt = tree.query_ball_point(pts[stack.pop()], r)
            for j in nxt:
                if lab[j] < 0:
                    lab[j] = cur
                    stack.append(j)
        cur += 1
    return lab


def main() -> int:
    args = parse_args()
    seg = Path(args.seg_dir)
    pz = np.load(seg / "points.npz")
    xyz, label = pz["xyz"].copy(), pz["label"].copy()
    doc = json.loads((seg / "instances.json").read_text(encoding="utf-8"))
    inst = doc["instances"] if isinstance(doc, dict) else doc
    names = json.loads((seg / "names.json").read_text(encoding="utf-8"))
    by_id = {it["id"]: it for it in inst}
    if args.id not in by_id:
        raise SystemExit(f"instance {args.id} 不存在")

    m = label == args.id
    pts = xyz[m]
    idx = np.where(m)[0]
    if args.z_cut is None:
        hist, edges = np.histogram(pts[:, 2], bins=60)
        z_cut = float(edges[int(hist.argmax()) + 1] + 0.008)  # 众数=支撑面,加余量
    else:
        z_cut = args.z_cut
    above = pts[:, 2] > z_cut
    print(f"instance {args.id}: {m.sum()} 点,z_cut={z_cut:.3f},面上凸起 {above.sum()} 点")

    cc = connected_components(pts[above], args.linkage)
    next_id = int(max(max(label.max(), max(by_id)), 0)) + 1
    made = []
    for c in range(cc.max() + 1):
        sel = np.where(above)[0][cc == c]
        if len(sel) < args.min_pts:
            continue  # 碎屑留在原实例
        gi = idx[sel]
        p = xyz[gi]
        lo, hi = p.min(0), p.max(0)
        it = {"id": next_id, "n_gaussians": int(len(gi)),
              "n_views": by_id[args.id].get("n_views", 0),
              "n_masks": 0,  # 0 = 几何拆分所得,非 SAM 共识(审计可辨)
              "centroid": [round(float(v), 3) for v in p.mean(0)],
              "bbox_min": [round(float(v), 3) for v in lo],
              "bbox_max": [round(float(v), 3) for v in hi],
              "size_m": [round(float(v), 3) for v in hi - lo]}
        made.append(it)
        if not args.dry_run:
            label[gi] = next_id
            inst.append(it)
            names[str(next_id)] = ""
        next_id += 1

    rem = idx[~above] if args.dry_run else idx[label[idx] == args.id]
    it0 = by_id[args.id]
    if not args.dry_run:
        if len(rem):
            p = xyz[rem]
            lo, hi = p.min(0), p.max(0)
            it0.update(n_gaussians=int(len(rem)),
                       centroid=[round(float(v), 3) for v in p.mean(0)],
                       bbox_min=[round(float(v), 3) for v in lo],
                       bbox_max=[round(float(v), 3) for v in hi],
                       size_m=[round(float(v), 3) for v in hi - lo])
        else:  # 全拆光:原实例成空壳,连名字一起删,不留幽灵
            inst.remove(it0)
            names.pop(str(args.id), None)

    made.sort(key=lambda it: it["centroid"][0])
    print(f"拆出 {len(made)} 个新实例(按板系 x 从小到大):")
    for it in made:
        c = it["centroid"]
        s = it.get("size_m") or [round(h - l, 3) for h, l in zip(it["bbox_max"], it["bbox_min"])]
        print(f"  id {it['id']:4d} n={it['n_gaussians']:5d} "
              f"size=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})m centroid=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})")
    print(f"  原 {args.id} 余 {len(rem)} 点(纸/托盘沿)")

    if args.dry_run:
        print("(dry-run,未写盘)")
        return 0
    np.savez_compressed(seg / "points.npz", xyz=xyz, label=label)
    # 顶层键(background 名字表等)必须原样保留:v5/v6 实测这里重建 dict 丢键,
    # gaze_live/gaze_object 读 meta['background'] 当场 KeyError(2026-08-03)。
    if isinstance(doc, dict):
        doc["instances"] = inst
    else:
        doc = {"instances": inst}
    (seg / "instances.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (seg / "names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写回 points.npz / instances.json / names.json(新 id 待命名)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
