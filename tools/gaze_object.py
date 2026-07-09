#!/usr/bin/env python
"""Which object is the gaze on?

Two modes over world-space fixations + a segmentation (points.npz/names.json):

--cone (recommended): probabilistic gaze-cone posterior. Re-renders a depth
  patch down the fixation's mean ray (camera origin -> centroid), weights every
  pixel by an angular Gaussian (sigma = per-recording gaze accuracy from
  gaze_precision.json, else --sigma-deg), unprojects it through the rendered
  depth and assigns it to the nearest labeled gaussian. Only VISIBLE surface
  inside the cone votes -- fixes both known sphere-mode failures: the floor
  under a thin target stealing votes, and look-at-cup-hit-table edge misses.
  Needs the splat ckpt (one 33x33 render per fixation, seconds overall).
  Statistics note: one fixation = ONE observation. Calibration bias is shared
  across its samples, so evidence must not sharpen with sample count; sigma
  should be the post-correction residual, not the raw jitter.

default (legacy sphere): 1/d-weighted vote among gaussians within --radius of
  the 3D fixation point. View-independent, kept as fallback/baseline.

Both modes pool votes by resolved name: several ids sharing a name in
names.json (SAM part splits) count as one object.

Example:
  python tools/gaze_object.py --cone \
      --fixations ~/recordings/2026_07_05/002/world_fixations.json \
      --seg-dir lab_result/segmentation_sam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "SceneRebuild"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixations", required=True,
                   help="world_fixations.json (continuous) or fixations_world.json (per-event).")
    p.add_argument("--seg-dir", default=None,
                   help="Default: lab_result/segmentation_sam if present, else lab_result/segmentation.")
    p.add_argument("--radius", type=float, default=0.20, help="Sphere mode: vote neighborhood radius (m).")
    p.add_argument("--cone", action="store_true", help="Gaze-cone posterior instead of sphere vote.")
    p.add_argument("--ckpt", default=None,
                   help="Cone mode: splat ckpt (default: newest step-*.ckpt under lab_result/).")
    p.add_argument("--sigma-deg", type=float, default=None,
                   help="Cone mode: gaze angular sigma. Default: <recording>/gaze_precision.json, else 1.5.")
    p.add_argument("--span-sigmas", type=float, default=2.5, help="Cone half-angle in sigmas.")
    p.add_argument("--patch", type=int, default=33, help="Cone render patch size (px).")
    p.add_argument("--hit-eps", type=float, default=0.05,
                   help="Max unprojected-point-to-gaussian distance (m); also rejects silhouette "
                        "depth blends -- do not raise casually.")
    p.add_argument("--out", default=None, help="Default: <fixations file>_objects.json next to input.")
    return p.parse_args()


def cone_votes(splat, tree, label, origin, pt, sigma_rad, span, S, eps):
    """Angular-Gaussian-weighted object masses over the visible surface in the cone."""
    d0 = pt - origin
    dist0 = float(np.linalg.norm(d0))
    if dist0 < 0.05:
        return {}, 1.0
    depth, alpha, dirs, tmul = splat.patch_along_ray(origin, d0 / dist0, span * sigma_rad, S)
    cosang = np.clip(dirs @ (d0 / dist0), -1.0, 1.0)
    w = np.exp(-np.arccos(cosang) ** 2 / (2 * sigma_rad ** 2))
    ok = (depth > 0.05) & (depth < 12.0)
    X = origin + (depth * tmul)[..., None] * dirs
    dd, idx = tree.query(X[ok], k=1, distance_upper_bound=eps, workers=-1)
    hit = np.isfinite(dd)
    votes: dict[int, float] = {}
    m_ok = (w * alpha)[ok]
    for lab, wi in zip(label[idx[hit]], m_ok[hit]):
        votes[int(lab)] = votes.get(int(lab), 0.0) + float(wi)
    total_w = float(w.sum())
    p_none = 1.0 - sum(votes.values()) / total_w if total_w > 0 else 1.0
    return votes, max(0.0, p_none)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[2] / "SceneRebuild"
    if args.seg_dir:
        seg = Path(args.seg_dir)
    else:
        seg = root / "lab_result/segmentation_sam"
        if not seg.exists():
            seg = root / "lab_result/segmentation"
    print(f"segmentation: {seg}")
    z = np.load(seg / "points.npz")
    xyz, label = z["xyz"], z["label"]
    meta = json.loads((seg / "instances.json").read_text(encoding="utf-8"))
    names = json.loads((seg / "names.json").read_text(encoding="utf-8")) if (seg / "names.json").exists() else {}
    bg = {int(k): v for k, v in meta["background"].items()}
    centroids = {inst["id"]: inst["centroid"] for inst in meta["instances"]}

    def name_of(lab: int) -> str:
        if lab in bg:
            return bg[lab]
        return names.get(str(lab), "") or f"object#{lab}"

    tree = cKDTree(xyz)
    doc = json.loads(Path(args.fixations).expanduser().read_text(encoding="utf-8"))
    fixes = doc["fixations"]

    splat, sigma_deg = None, None
    if args.cone:
        import gaze_to_world as g2w
        ckpt = Path(args.ckpt) if args.ckpt else max(
            (root / "lab_result").rglob("step-*.ckpt"), key=lambda p: p.stat().st_mtime)
        splat = g2w.SplatDepth(ckpt)
        sigma_deg = args.sigma_deg
        if sigma_deg is None:
            pj = Path(args.fixations).expanduser().parent / "gaze_precision.json"
            if pj.exists():
                sigma_deg = float(json.loads(pj.read_text(encoding="utf-8"))["sigma_deg"])
                print(f"sigma from {pj.name}: {sigma_deg:.2f} deg")
        sigma_deg = sigma_deg or 1.5
        print(f"cone mode: sigma {sigma_deg:.2f} deg, half-angle {args.span_sigmas:.1f} sigma, "
              f"patch {args.patch}, hit-eps {args.hit_eps*100:.0f}cm")

    print(f"{len(fixes)} fixations vs {len(xyz)} labeled gaussians "
          f"({len(meta['instances'])} instances)")
    print(f"{'#':>3} {'t':>7} {'point/centroid xyz':<26} {'best':<22} {'vote':>5}  runner-up")
    results = []
    for k, fx in enumerate(fixes):
        pt = np.array(fx.get("centroid_world") or fx.get("point_world"))
        t = fx.get("t_start", fx.get("t_rel", 0.0))
        entry = dict(fx)
        if pt is None or (isinstance(pt, np.ndarray) and pt.dtype == object):
            entry["object"] = None
            results.append(entry)
            continue
        origin = fx.get("origin_world")
        if origin is None and fx.get("T_world_cam") is not None:
            origin = np.array(fx["T_world_cam"], float)[:3, 3]
        if args.cone and origin is not None:
            votes, p_none = cone_votes(splat, tree, label, np.asarray(origin, float), pt,
                                       np.radians(sigma_deg), args.span_sigmas,
                                       args.patch, args.hit_eps)
            entry["mode"] = "cone"
            entry["p_none"] = round(p_none, 3)
        else:
            if args.cone:
                entry["mode"] = "sphere-fallback"  # old fixations file without origin_world
            idx = tree.query_ball_point(pt, args.radius)
            votes = {}
            if idx:
                d = np.linalg.norm(xyz[idx] - pt, axis=1)
                w = 1.0 / np.maximum(d, 0.01)
                for lab, wi in zip(label[idx], w):
                    votes[int(lab)] = votes.get(int(lab), 0.0) + float(wi)
        if not votes:
            entry.update(object=None, note="nothing labeled in gaze neighborhood")
            print(f"{k:>3} {t:>7.1f} {str(np.round(pt,2)):<26} {'(nothing nearby)':<22}")
            results.append(entry)
            continue
        total = sum(votes.values())
        # pool by resolved name: ids hand-merged in names.json (same name on several
        # instances, e.g. a robot SAM splits along color boundaries) vote as one object
        pooled: dict[str, dict] = {}
        for lab, v in votes.items():
            p = pooled.setdefault(name_of(lab), {"v": 0.0, "labels": []})
            p["v"] += v
            p["labels"].append(lab)
        ranked = sorted(pooled.items(), key=lambda kv: -kv[1]["v"])
        best_name, bp = ranked[0]
        share = bp["v"] / total
        best = max(bp["labels"], key=lambda l: votes[l])
        second = f"{ranked[1][0]} {ranked[1][1]['v']/total:.0%}" if len(ranked) > 1 else "-"
        entry.update(object_label=best, object=best_name, vote_share=round(share, 3),
                     # gaze hit point (surface, varies) vs canonical object position (fixed):
                     object_centroid_world=centroids.get(best),
                     candidates=[{"name": n, "share": round(p["v"] / total, 3),
                                  "labels": sorted(p["labels"])} for n, p in ranked[:3]])
        results.append(entry)
        none_s = f"  none {entry['p_none']:.0%}" if "p_none" in entry else ""
        print(f"{k:>3} {t:>7.1f} {str(np.round(pt,2)):<26} {best_name:<22} {share:>5.0%}  {second}{none_s}")

    out = Path(args.out) if args.out else Path(args.fixations).expanduser().with_name(
        Path(args.fixations).stem + "_objects.json")
    out.write_text(json.dumps({"source": str(args.fixations), "seg_dir": str(seg),
                               "mode": "cone" if args.cone else "sphere",
                               "sigma_deg": sigma_deg, "radius_m": args.radius,
                               "fixations": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
