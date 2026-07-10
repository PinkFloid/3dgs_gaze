#!/usr/bin/env python
"""Offline splat segmentation v1: geometry-only instance clustering.

Splits the gaussians into background classes (floor / ceiling / walls) and
free-standing object instances found by voxel connected-components. No
semantics -- each instance gets highlight thumbnails so a human can name it
once in names.json. Good enough to answer "which object is the gaze on";
known v1 limitation: touching furniture (desk chains, chair against desk)
merges into one instance.

Outputs into --out-dir (default lab_result/segmentation):
  points.npz        xyz (N,3 float32) + label (N int32) for retained gaussians
                    labels: 0 floor, 1 ceiling, 2-5 walls, >=10 object instances
  instances.json    per instance: id, centroid, bbox, n_gaussians
  names.json        {"10": "", ...} template -- fill in object names by hand
  thumbs/inst_<id>_<k>.jpg   instance highlighted (others dimmed), 2 views
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLOOR, CEILING = 0, 1
WALL_XMIN, WALL_XMAX, WALL_YMIN, WALL_YMAX = 2, 3, 4, 5
OBJ0 = 10
BG_NAMES = {0: "floor", 1: "ceiling", 2: "wall", 3: "wall", 4: "wall", 5: "wall"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Default: newest step-*.ckpt under lab_result/.")
    p.add_argument("--out-dir", default=str(root / "lab_result/segmentation"))
    p.add_argument("--voxel", type=float, default=0.05, help="Clustering voxel size (m).")
    p.add_argument("--min-opacity", type=float, default=0.5)
    p.add_argument("--floor-z", type=float, default=0.08, help="z below this = floor.")
    p.add_argument("--ceiling-z", type=float, default=2.5, help="z above this = ceiling.")
    p.add_argument("--wall-margin", type=float, default=0.15, help="Distance to room bounds = wall.")
    p.add_argument("--min-gaussians", type=int, default=150, help="Min gaussians per instance.")
    p.add_argument("--min-size", type=float, default=0.12, help="Min instance bbox diagonal (m).")
    p.add_argument("--thumbs", type=int, default=2, help="Highlight views per instance (0 = skip).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from scipy import ndimage

    root = Path(__file__).resolve().parent.parent
    ckpt = Path(args.ckpt) if args.ckpt else max(
        (root / "lab_result").rglob("step-*.ckpt"), key=lambda p: p.stat().st_mtime)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = torch.load(ckpt, map_location="cpu")["pipeline"]
    g = lambda n: sd[f"_model.gauss_params.{n}"].numpy()
    xyz = g("means").astype(np.float32)
    opac = 1 / (1 + np.exp(-g("opacities").reshape(-1)))
    max_scale = np.exp(g("scales")).max(axis=1)

    # room bounds from the bulk of the mass
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    keep = (opac >= args.min_opacity) & (max_scale < 0.5) & \
           np.all((xyz > lo - 0.2) & (xyz < hi + 0.2), axis=1)
    print(f"{keep.sum()}/{len(xyz)} gaussians kept (opacity/scale/bounds), "
          f"room x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}] z[{lo[2]:.1f},{hi[2]:.1f}]")
    xyz_k = xyz[keep]

    label = np.full(len(xyz_k), -1, np.int32)
    z = xyz_k[:, 2]
    label[z < args.floor_z] = FLOOR
    label[z > args.ceiling_z] = CEILING
    m = label == -1
    label[m & (xyz_k[:, 0] < lo[0] + args.wall_margin)] = WALL_XMIN
    label[m & (xyz_k[:, 0] > hi[0] - args.wall_margin)] = WALL_XMAX
    m = label == -1
    label[m & (xyz_k[:, 1] < lo[1] + args.wall_margin)] = WALL_YMIN
    label[m & (xyz_k[:, 1] > hi[1] - args.wall_margin)] = WALL_YMAX

    # voxel connected components over the remaining (object) gaussians
    obj_mask = label == -1
    pts = xyz_k[obj_mask]
    ijk = np.floor((pts - (lo - 0.2)) / args.voxel).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    grid = np.zeros(dims, bool)
    grid[tuple(ijk.T)] = True
    comp, n_comp = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    comp_of_pt = comp[tuple(ijk.T)]
    print(f"{n_comp} raw components at {args.voxel*100:.0f}cm voxels")

    instances = []
    obj_label = np.full(len(pts), -1, np.int32)
    next_id = OBJ0
    for c in range(1, n_comp + 1):
        sel = comp_of_pt == c
        n = int(sel.sum())
        if n < args.min_gaussians:
            continue
        p = pts[sel]
        bb_lo, bb_hi = p.min(axis=0), p.max(axis=0)
        if np.linalg.norm(bb_hi - bb_lo) < args.min_size:
            continue
        obj_label[sel] = next_id
        instances.append({
            "id": next_id, "n_gaussians": n,
            "centroid": p.mean(axis=0).round(3).tolist(),
            "bbox_min": bb_lo.round(3).tolist(), "bbox_max": bb_hi.round(3).tolist(),
            "size_m": (bb_hi - bb_lo).round(2).tolist(),
        })
        next_id += 1
    label[obj_mask] = obj_label
    print(f"{len(instances)} instances kept (>= {args.min_gaussians} gaussians, >= {args.min_size}m)")

    retained = label >= 0
    np.savez(out_dir / "points.npz", xyz=xyz_k[retained], label=label[retained])
    (out_dir / "instances.json").write_text(json.dumps({
        "ckpt": str(ckpt), "voxel_m": args.voxel,
        "background": BG_NAMES, "instances": instances}, indent=2), encoding="utf-8")
    names_path = out_dir / "names.json"
    if names_path.exists():
        names = json.loads(names_path.read_text())
    else:
        names = {}
    for inst in instances:
        names.setdefault(str(inst["id"]), "")
    names_path.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")

    for inst in sorted(instances, key=lambda i: -i["n_gaussians"])[:20]:
        c, s = inst["centroid"], inst["size_m"]
        print(f"  id {inst['id']:>3}: {inst['n_gaussians']:>7} gaussians  "
              f"center ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})  size {s[0]}x{s[1]}x{s[2]}m")

    if args.thumbs > 0:
        render_thumbs(sd, keep, label, instances, out_dir, args.thumbs)
    print(f"done -> {out_dir}  (fill in names.json using thumbs/)")
    return 0


def render_thumbs(sd, keep, label, instances, out_dir: Path, n_views: int):
    import cv2
    import torch
    from gsplat import rasterization
    dev = torch.device("cuda")
    g = lambda n: sd[f"_model.gauss_params.{n}"].to(dev)
    means = g("means")
    quats = torch.nn.functional.normalize(g("quats"), dim=-1)
    scales = torch.exp(g("scales"))
    opac_full = torch.sigmoid(g("opacities")).squeeze(-1)
    colors = torch.cat([g("features_dc").unsqueeze(1), g("features_rest")], dim=1)

    full_label = np.full(len(means), -1, np.int32)
    full_label[np.flatnonzero(keep.copy())] = label
    full_label_t = torch.from_numpy(full_label).to(dev)

    thumbs = out_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    W = H = 480
    K = torch.tensor([[420.0, 0, W / 2], [0, 420.0, H / 2], [0, 0, 1]], device=dev).unsqueeze(0)

    for inst in instances:
        c = np.array(inst["centroid"])
        diag = np.linalg.norm(np.array(inst["bbox_max"]) - np.array(inst["bbox_min"]))
        dist = float(np.clip(2.2 * diag, 0.8, 4.0))
        opac = torch.where(full_label_t == inst["id"], opac_full, opac_full * 0.06)
        for k in range(n_views):
            az = 2 * np.pi * k / max(n_views, 1) + 0.5
            eye = c + dist * np.array([np.cos(az) * 0.87, np.sin(az) * 0.87, 0.5])
            eye[2] = max(eye[2], 0.3)
            zax = c - eye
            zax = zax / np.linalg.norm(zax)
            xax = np.cross(zax, [0, 0, 1.0])
            xax /= np.linalg.norm(xax)
            yax = np.cross(zax, xax)
            w2c = np.eye(4)
            w2c[:3, :3] = np.stack([xax, yax, zax])
            w2c[:3, 3] = -w2c[:3, :3] @ eye
            vm = torch.tensor(w2c, dtype=torch.float32, device=dev).unsqueeze(0)
            out, _, _ = rasterization(means, quats, scales, opac, colors, vm, K, W, H,
                                      sh_degree=3, render_mode="RGB")
            img = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, f"id {inst['id']}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imwrite(str(thumbs / f"inst_{inst['id']:03d}_{k}.jpg"), img)


if __name__ == "__main__":
    raise SystemExit(main())
