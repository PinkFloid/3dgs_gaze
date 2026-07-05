#!/usr/bin/env python
"""Which object is the gaze on? v1: neighborhood vote around the 3D fixation.

Takes world-space fixations (gaze_to_world.py --continuous output, or the
per-fixation json) and the segmentation from segment_splat.py, and assigns
each fixation to an object by voting among the gaussians within --radius of
the fixation point (weights 1/d). Reports the vote share of the top-2
candidates -- the placeholder that a Bayesian ray-vs-instance model will
replace later (looking-at-cup-but-hit-table needs the ray, not just the point).

Example:
  python tools/gaze_object.py --fixations ~/recordings/2026_07_05/000/world_fixations.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixations", required=True,
                   help="world_fixations.json (continuous) or fixations_world.json (per-event).")
    p.add_argument("--seg-dir", default=str(root / "lab_result/segmentation"))
    p.add_argument("--radius", type=float, default=0.20, help="Vote neighborhood radius (m).")
    p.add_argument("--out", default=None, help="Default: <fixations file>_objects.json next to input.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seg = Path(args.seg_dir)
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
        idx = tree.query_ball_point(pt, args.radius)
        if not idx:
            entry.update(object=None, note=f"no gaussians within {args.radius}m")
            print(f"{k:>3} {t:>7.1f} {str(np.round(pt,2)):<26} {'(nothing nearby)':<22}")
            results.append(entry)
            continue
        d = np.linalg.norm(xyz[idx] - pt, axis=1)
        w = 1.0 / np.maximum(d, 0.01)
        votes = {}
        for lab, wi in zip(label[idx], w):
            votes[int(lab)] = votes.get(int(lab), 0.0) + float(wi)
        total = sum(votes.values())
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        best, share = ranked[0][0], ranked[0][1] / total
        second = f"{name_of(ranked[1][0])} {ranked[1][1]/total:.0%}" if len(ranked) > 1 else "-"
        entry.update(object_label=best, object=name_of(best), vote_share=round(share, 3),
                     # gaze hit point (surface, varies) vs canonical object position (fixed):
                     object_centroid_world=centroids.get(best),
                     candidates=[{"label": l, "name": name_of(l), "share": round(v / total, 3)}
                                 for l, v in ranked[:3]])
        results.append(entry)
        print(f"{k:>3} {t:>7.1f} {str(np.round(pt,2)):<26} {name_of(best):<22} {share:>5.0%}  {second}")

    out = Path(args.out) if args.out else Path(args.fixations).expanduser().with_name(
        Path(args.fixations).stem + "_objects.json")
    out.write_text(json.dumps({"source": str(args.fixations), "seg_dir": str(seg),
                               "radius_m": args.radius, "fixations": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
