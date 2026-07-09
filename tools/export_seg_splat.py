#!/usr/bin/env python
"""Export a segmentation-colored 3DGS ply: see WHAT the object posterior sees.

Every gaussian is recolored by its instance label (same stable colors as
gaze_video boxes, same-named instances share one color), background/unlabeled
gaussians go dim gray, and each named object gets a bright bead-frame along
its union bbox edges. Open in SuperSplat / any 3DGS viewer to audit the
segmentation that gaze_object/gaze_live vote against.

  python tools/export_seg_splat.py                     # newest ckpt + segmentation_sam
  python tools/export_seg_splat.py --preview seg.jpg   # also render 2 check views

Output: <seg-dir>/splat_seg.ply (ignored by git like every ply).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
C0 = 0.28209479177387814  # SH band-0: color = 0.5 + C0 * f_dc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None, help="Default: newest step-*.ckpt under lab_result/.")
    p.add_argument("--seg-dir", default=str(ROOT / "lab_result/segmentation_sam"))
    p.add_argument("--out", default=None, help="Default: <seg-dir>/splat_seg.ply")
    p.add_argument("--bead-step", type=float, default=0.025, help="BBox bead spacing (m).")
    p.add_argument("--dim", type=float, default=0.35, help="Brightness of unlabeled/background gaussians.")
    p.add_argument("--preview", default=None, help="Also render top+oblique check views to this jpg.")
    p.add_argument("--preview-clip-z", type=float, default=2.6,
                   help="Drop gaussians above this height in the preview renders (ceiling/lamps).")
    return p.parse_args()


def stable_color(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(80, 255, 3).astype(np.float32) / 255.0  # rgb 0..1, same as gaze_video


def bead_frame(lo, hi, step):
    """Points along the 12 edges of an axis-aligned box."""
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    edges = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    pts = []
    for a, b in edges:
        seg = corners[b] - corners[a]
        n = max(2, int(np.linalg.norm(seg) / step))
        t = np.linspace(0, 1, n)[:, None]
        pts.append(corners[a] + t * seg)
    return np.concatenate(pts)


def main() -> int:
    args = parse_args()
    seg = Path(args.seg_dir)
    ckpt = Path(args.ckpt) if args.ckpt else max(
        (ROOT / "lab_result").rglob("step-*.ckpt"), key=lambda p: p.stat().st_mtime)
    out = Path(args.out) if args.out else seg / "splat_seg.ply"

    sd = torch.load(ckpt, map_location="cpu")["pipeline"]
    g = lambda name: sd[f"_model.gauss_params.{name}"].numpy()
    means = g("means").astype(np.float32)
    scales = g("scales").astype(np.float32)
    quats = g("quats").astype(np.float32)
    opac = g("opacities").reshape(-1, 1).astype(np.float32)
    n = len(means)
    print(f"ckpt: {ckpt.name}, {n} gaussians")

    z = np.load(seg / "points.npz")
    sxyz, slabel = z["xyz"], z["label"]
    meta = json.loads((seg / "instances.json").read_text(encoding="utf-8"))
    names = json.loads((seg / "names.json").read_text(encoding="utf-8")) if (seg / "names.json").exists() else {}

    # map labels back onto the full checkpoint: points.npz is the lift-retained
    # subset with identical coordinates, so nearest-neighbor at ~0 distance
    from scipy.spatial import cKDTree
    d, idx = cKDTree(sxyz).query(means, k=1, distance_upper_bound=1e-4, workers=-1)
    label = np.where(np.isfinite(d), slabel[np.clip(idx, 0, len(slabel) - 1)], -1)
    print(f"labels mapped: {np.isfinite(d).sum()}/{n} gaussians "
          f"({(~np.isfinite(d)).sum()} unlabeled -> dim gray)")

    # one color per NAME (same-named instances pooled, like the voter)
    name_of = {}
    for inst in meta["instances"]:
        nm = names.get(str(inst["id"]), "")
        if nm:
            name_of[inst["id"]] = nm
    group_color = {}
    for iid, nm in sorted(name_of.items()):
        group_color.setdefault(nm, stable_color(iid))

    rgb = np.full((n, 3), args.dim * 0.5, np.float32)      # unlabeled: dark gray
    rgb[label >= 0] = args.dim                              # bg + unnamed instances: gray
    counts = {}
    for iid, nm in name_of.items():
        m = label == iid
        rgb[m] = group_color[nm]
        counts[nm] = counts.get(nm, 0) + int(m.sum())

    # bead frames on named-object union bboxes
    union = {}
    for inst in meta["instances"]:
        nm = names.get(str(inst["id"]), "")
        if not nm:
            continue
        lo, hi = np.array(inst["bbox_min"]), np.array(inst["bbox_max"])
        u = union.get(nm)
        union[nm] = (lo, hi) if u is None else (np.minimum(u[0], lo), np.maximum(u[1], hi))
    beads_xyz, beads_rgb = [], []
    for nm, (lo, hi) in union.items():
        b = bead_frame(lo, hi, args.bead_step)
        beads_xyz.append(b)
        beads_rgb.append(np.tile(np.clip(group_color[nm] * 1.25, 0, 1), (len(b), 1)))
    beads_xyz = np.concatenate(beads_xyz).astype(np.float32)
    beads_rgb = np.concatenate(beads_rgb).astype(np.float32)
    nb = len(beads_xyz)

    all_xyz = np.concatenate([means, beads_xyz])
    all_rgb = np.concatenate([rgb, beads_rgb])
    all_scales = np.concatenate([scales, np.full((nb, 3), np.log(0.006), np.float32)])
    all_quats = np.concatenate([quats, np.tile(np.array([1, 0, 0, 0], np.float32), (nb, 1))])
    all_opac = np.concatenate([opac, np.full((nb, 1), 6.0, np.float32)])  # logit ~ opaque
    f_dc = ((all_rgb - 0.5) / C0).astype(np.float32)
    N = len(all_xyz)

    cols = (
        [("x", all_xyz[:, 0]), ("y", all_xyz[:, 1]), ("z", all_xyz[:, 2]),
         ("nx", np.zeros(N, np.float32)), ("ny", np.zeros(N, np.float32)), ("nz", np.zeros(N, np.float32))]
        + [(f"f_dc_{i}", f_dc[:, i]) for i in range(3)]
        + [(f"f_rest_{i}", np.zeros(N, np.float32)) for i in range(45)]
        + [("opacity", all_opac[:, 0])]
        + [(f"scale_{i}", all_scales[:, i]) for i in range(3)]
        + [(f"rot_{i}", all_quats[:, i]) for i in range(4)]
    )
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {N}"]
    header += [f"property float {name}" for name, _ in cols]
    header += ["end_header"]
    data = np.stack([c for _, c in cols], axis=1)
    with open(out, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.astype("<f4").tobytes())
    print(f"wrote {out}: {n} gaussians + {nb} bbox beads")
    for nm, c in group_color.items():
        r, gg, b = (c * 255).astype(int)
        print(f"  {nm:<12} rgb({r},{gg},{b})  {counts.get(nm, 0)} gaussians")

    if args.preview:
        keep = all_xyz[:, 2] < args.preview_clip_z
        render_preview(all_xyz[keep], all_rgb[keep], all_scales[keep],
                       all_quats[keep], all_opac[keep], args.preview)
    return 0


def render_preview(xyz, rgb, scales, quats, opac, out_jpg):
    import cv2
    from gsplat import rasterization
    dev = torch.device("cuda")
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    center = np.median(xyz[:: max(1, len(xyz) // 100000)], axis=0)
    views = []
    for name, eye_off, up in [("top", np.array([0, 0, 6.0]), np.array([0, 1, 0.0])),
                              ("oblique", np.array([-4.0, -4.0, 3.0]), np.array([0, 0, 1.0]))]:
        eye = center + eye_off
        z_ax = center - eye
        z_ax = z_ax / np.linalg.norm(z_ax)
        x_ax = np.cross(z_ax, up)
        x_ax /= np.linalg.norm(x_ax)
        y_ax = np.cross(z_ax, x_ax)
        w2c = np.eye(4)
        w2c[:3, :3] = np.stack([x_ax, y_ax, z_ax])
        w2c[:3, 3] = -w2c[:3, :3] @ eye
        W = H = 900
        K = np.array([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]])
        img, _, _ = rasterization(
            t(xyz), torch.nn.functional.normalize(t(quats), dim=-1), torch.exp(t(scales)),
            torch.sigmoid(t(opac)).squeeze(-1), t(rgb),
            t(w2c)[None], t(K)[None], W, H, render_mode="RGB", rasterize_mode="classic")
        views.append((np.clip(img[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8))
    cv2.imwrite(out_jpg, cv2.cvtColor(np.concatenate(views, axis=1), cv2.COLOR_RGB2BGR))
    print(f"wrote {out_jpg}")


if __name__ == "__main__":
    raise SystemExit(main())
