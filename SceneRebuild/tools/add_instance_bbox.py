#!/usr/bin/env python
"""add_instance_bbox.py -- 把 lift 漏掉的物体按世界系包围盒从 ckpt 高斯里补成一个实例。

lift_sam_instances 对"少视角 + 大块无纹理纸板"的物体可能只留碎片(2026-09-07 v11
纸箱子实测:splat 里 1.1 万个高斯完整,但只有两个 400 点的碎片过了跨帧共识)。
补救是几何的:

  1. 用与 lift 完全相同的 keep 规则(opacity>=0.5、max_scale<0.5、房间 1-99 分位
     ±0.2m)从 ckpt 载入高斯,保证新点与 points.npz 同源(fig_scene_map 会校验);
  2. 取包围盒内的高斯作为新实例,id = 现有最大 id + 1;
  3. points.npz 里已有的同坐标点改标签,没有的追加;--absorb 的碎片 id 整体并入,
     碎片里落在盒外的点丢弃;
  4. instances.json 增加条目(source=manual_bbox,n_views 取被吸收碎片的最大值),
     names.json 补名,顺手删掉被吸收的 id。

用法:
    python tools/add_instance_bbox.py --seg-dir lab_result/segmentation_sam \
        --bbox -0.30 0.13 -1.72 -1.26 0.795 1.00 --name 纸箱子 --absorb 29 33 [--dry-run]
包围盒怎么定:先看 splat 里该区域 1cm 分辨率的 z 直方图,支撑面(台面/架子板)是最高
的那根柱,zmin 取它上方 1-2cm;xy 取物体脚印外扩 2-3cm,别扩到支撑面边缘以外。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_ckpt_points(ckpt: Path):
    import torch
    sd = torch.load(ckpt, map_location="cpu")["pipeline"]
    g = lambda n: sd[f"_model.gauss_params.{n}"]
    xyz = g("means").numpy().astype(np.float32)
    opac = 1 / (1 + np.exp(-g("opacities").numpy().reshape(-1)))
    max_scale = np.exp(g("scales").numpy()).max(axis=1)
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    keep = (opac >= 0.5) & (max_scale < 0.5) & np.all((xyz > lo - 0.2) & (xyz < hi + 0.2), axis=1)
    return xyz[keep]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seg-dir", required=True)
    p.add_argument("--ckpt", default=None, help="默认取 instances.json 里记录的 ckpt")
    p.add_argument("--bbox", type=float, nargs=6, required=True,
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    p.add_argument("--name", default="", help="写进 names.json 的名字(可空)")
    p.add_argument("--absorb", type=int, nargs="*", default=[], help="并入新实例的碎片 id")
    p.add_argument("--id", type=int, default=None,
                   help="指定实例 id(默认现有最大 id+1);与 --absorb 里的 id 相同 = 原地重建该实例")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    seg = Path(a.seg_dir)
    inst_path, names_path, pts_path = seg / "instances.json", seg / "names.json", seg / "points.npz"
    meta = json.loads(inst_path.read_text(encoding="utf-8"))
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}
    z = np.load(pts_path)
    xyz, label = z["xyz"].astype(np.float32), z["label"].astype(np.int32)

    ckpt = Path(a.ckpt) if a.ckpt else Path(meta["ckpt"])
    xyz_k = load_ckpt_points(ckpt)
    x0, x1, y0, y1, z0, z1 = a.bbox
    sel = np.all((xyz_k > [x0, y0, z0]) & (xyz_k < [x1, y1, z1]), axis=1)
    new_pts = xyz_k[sel]
    if len(new_pts) < 80:
        raise SystemExit(f"盒内只有 {len(new_pts)} 个高斯(<80),包围盒不对?")

    ids = [int(i["id"]) for i in meta["instances"]] + [int(k) for k in names]
    new_id = a.id if a.id is not None else max(ids + [9]) + 1
    if a.id is not None and a.id in ids and a.id not in a.absorb:
        raise SystemExit(f"id {a.id} 已被占用;要原地重建请同时 --absorb {a.id}")

    # 已有点按坐标精确匹配(points.npz 是 ckpt 高斯的子集)
    index = {row.tobytes(): i for i, row in enumerate(np.ascontiguousarray(xyz))}
    hit = np.array([index.get(row.tobytes(), -1) for row in np.ascontiguousarray(new_pts)])
    n_relabel = int((hit >= 0).sum())
    old_labels = {}
    for i in hit[hit >= 0]:
        old_labels[int(label[i])] = old_labels.get(int(label[i]), 0) + 1
    absorbed = [i for i in meta["instances"] if int(i["id"]) in a.absorb]
    drop_outside = int(np.isin(label, a.absorb).sum() - sum(old_labels.get(i, 0) for i in a.absorb))

    bb_lo, bb_hi = np.percentile(new_pts.astype(np.float64), 2, axis=0), np.percentile(new_pts.astype(np.float64), 98, axis=0)
    entry = {
        "id": new_id, "n_gaussians": int(len(new_pts)),
        "n_views": max([int(i.get("n_views", 0)) for i in absorbed] + [0]),
        "n_masks": sum(int(i.get("n_masks", 0)) for i in absorbed),
        "centroid": new_pts.astype(np.float64).mean(axis=0).round(3).tolist(),
        "bbox_min": bb_lo.round(3).tolist(), "bbox_max": bb_hi.round(3).tolist(),
        "size_m": (bb_hi - bb_lo).round(2).tolist(),
        "source": "manual_bbox", "bbox_arg": list(a.bbox),
    }
    print(f"new id {new_id} '{a.name}': {len(new_pts)} gaussians in bbox; "
          f"{n_relabel} already in points.npz (old labels {old_labels}), "
          f"{len(new_pts) - n_relabel} appended; absorb {a.absorb}: {drop_outside} pts outside bbox dropped")
    print(f"  centroid {entry['centroid']} size {entry['size_m']} bbox {entry['bbox_min']}..{entry['bbox_max']}")
    if a.dry_run:
        print("(dry-run,未写盘)")
        return 0

    bak = seg / "_bak_before_add_instance"
    bak.mkdir(exist_ok=True)
    for f in (inst_path, names_path, pts_path):
        if f.exists():
            shutil.copy(f, bak / f.name)

    keep_rows = ~np.isin(label, a.absorb)
    label = label.copy()
    label[hit[hit >= 0]] = new_id
    keep_rows[hit[hit >= 0]] = True
    xyz2 = np.concatenate([xyz[keep_rows], new_pts[hit < 0]])
    lab2 = np.concatenate([label[keep_rows], np.full(int((hit < 0).sum()), new_id, np.int32)])
    np.savez(pts_path, xyz=xyz2.astype(np.float32), label=lab2.astype(np.int32))

    meta["instances"] = [i for i in meta["instances"] if int(i["id"]) not in a.absorb] + [entry]
    inst_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    for i in a.absorb:
        names.pop(str(i), None)
    names[str(new_id)] = a.name
    names_path.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写回 points.npz({len(xyz2)} 点) / instances.json / names.json;原件备份在 {bak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
