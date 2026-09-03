#!/usr/bin/env python
"""Which object is the gaze on?  (v2: area-normalised cone posterior)

Two modes over world-space fixations + a segmentation (points.npz/names.json):

--cone (recommended): gaze-cone posterior. Re-renders a depth patch down the
  fixation's mean ray (camera origin -> centroid), weights every pixel by an
  angular Gaussian (sigma = gaze accuracy, 1.0 deg unless stamped), unprojects
  it through the rendered depth and assigns it to the nearest labeled gaussian.
  Only VISIBLE surface inside the cone votes. Needs the splat ckpt (one 33x33
  render per fixation).

  v2 scoring (2026-09-02, see docs/CONE_POSTERIOR_V2.md):
  * Candidates = target vocabulary only: names.json minus places.json.
    Places (物品台/纸箱子), background, unnamed fragments, silhouette blends,
    alpha holes and out-of-margin depth are all INVALID mass -> p_none.
    The KD tree keeps every labeled gaussian, so fragment surface can no
    longer be adopted by a named neighbour within --hit-eps.
  * One denominator W = sum of kernel weights over the patch:
    q_k = m_k / W,  p_none = 1 - sum_k q_k.
  * Per-target expected capture c_k = kernel mass inside the target's own
    angular disk (RMS radius / fixation distance) / W. capture_k = q_k / c_k
    is size- and distance-invariant (~ exp(-d^2 / 2 sigma^2), d = angular
    miss) and is the area-normalised likelihood the posterior ranks by
    (--rank capture; --rank mass = the old projected-area vote, for E4).
  Statistics note: one fixation = ONE observation. Calibration bias is shared
  across its samples, so evidence must not sharpen with sample count.

default (legacy sphere): 1/d-weighted vote among gaussians within --radius of
  the 3D fixation point. View-independent baseline; no kernel -> no capture.

Example:
  python tools/gaze_object.py --cone \
      --fixations ~/recordings/2026_07_05/002/world_fixations.json \
      --seg-dir lab_result/segmentation_sam
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OBJ0 = 10                     # labels >= OBJ0 are instances; below: floor/ceiling/wall
BG_NAMES = ("floor", "ceiling", "wall")
DEFAULT_BG = {0: "floor", 1: "ceiling", 2: "wall", 3: "wall", 4: "wall", 5: "wall"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "SceneRebuild"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixations", required=True,
                   help="world_fixations.json (continuous) or fixations_world.json (per-event).")
    p.add_argument("--seg-dir", default=None,
                   help="Default: lab_result/segmentation_sam if present, else lab_result/segmentation.")
    p.add_argument("--places", default=None,
                   help="places.json (JSON list of names that are places, not targets). "
                        "Default: <seg-dir>/places.json if present.")
    p.add_argument("--rank", choices=["capture", "mass"], default="capture",
                   help="capture = area-normalised posterior (v2 default); mass = raw cone mass (v1 order).")
    p.add_argument("--radius", type=float, default=0.20, help="Sphere mode: vote neighborhood radius (m).")
    p.add_argument("--cone", action="store_true", help="Gaze-cone posterior instead of sphere vote.")
    p.add_argument("--ckpt", default=None,
                   help="Cone mode: splat ckpt (default: newest step-*.ckpt under lab_result/).")
    p.add_argument("--sigma-deg", type=float, default=None,
                   help="Cone mode: gaze angular sigma. Default: <recording>/gaze_precision.json, else 1.0 (sweep optimum = calibrated accuracy).")
    p.add_argument("--span-sigmas", type=float, default=2.0,
                   help="Cone half-angle in sigmas (2 = 86%% of the truth probability inside the cone).")
    p.add_argument("--patch", type=int, default=33, help="Cone render patch size (px).")
    p.add_argument("--patch-deg", type=float, default=6.0 / 33,
                   help="Angular pixel size (deg): patch = 2*span*sigma/patch-deg (odd, >= --patch); 33 px up to sigma 1.5.")
    p.add_argument("--hit-eps", type=float, default=0.05,
                   help="Max unprojected-point-to-gaussian distance (m); also rejects silhouette "
                        "depth blends -- do not raise casually.")
    p.add_argument("--out", default=None, help="Default: <fixations file>_objects.json next to input.")
    return p.parse_args()


# ------------------------------------------------------------ vocabulary / geometry helpers

def load_places(seg: Path, path=None) -> set:
    """Names that are places (destinations), never gaze targets."""
    pf = Path(path) if path else seg / "places.json"
    if not pf.exists():
        return set()
    return {str(n) for n in json.loads(pf.read_text(encoding="utf-8")) if n}


def load_background(meta: dict) -> dict:
    bg = {int(k): v for k, v in meta.get("background", {}).items()}
    if not bg:  # 新建图流水线导出的 instances.json 可能丢 background 段(v5 实测):
        bg = dict(DEFAULT_BG)
        print("[!] instances.json 缺 background,按约定 0=floor/1=ceiling/2-5=wall 兜底")
    return bg


def make_name_of(bg: dict, names: dict):
    def name_of(lab: int) -> str:
        if lab in bg:
            return bg[lab]
        return names.get(str(lab), "") or f"object#{lab}"
    return name_of


def object_radii_by_name(xyz, label, names, only=None) -> dict:
    """Radius (m) per named object: 84th-percentile distance-to-centroid of its
    LARGEST instance.

    This is the angular-disk size used for expected capture. A (partial)
    spherical shell returns ~its radius, where the RMS under-estimates it
    (v9 balls: RMS 2.8-3.1 cm vs 3.35 cm real). Same-named instances are
    usually SAM part splits of one object, but a stray fragment 15 cm away
    (v9 苹果粉 id 250, 85 pts) inflates a pooled radius from 4.5 to 7.6 cm,
    so the biggest part stands for the object. `only` restricts to a set of
    names (the target vocabulary).
    """
    by_name: dict[str, list] = {}
    for k, nm in names.items():
        if nm and (only is None or nm in only):
            by_name.setdefault(nm, []).append(int(k))
    out = {}
    for nm, labs in by_name.items():
        best = None
        for lab in labs:
            pts = xyz[label == lab]
            if len(pts) >= 4 and (best is None or len(pts) > len(best)):
                best = pts
        if best is None:
            continue
        c = best.mean(axis=0)
        out[nm] = float(np.percentile(np.linalg.norm(best - c, axis=1), 84))
    return out


def disk_capture(kern, rho: float):
    """Fraction of the cone's kernel mass inside an axis-centred disk of angular
    radius rho: what a perfectly fixated target of that angular size can capture
    at most. Computed on the same pixel grid as the votes, so patch truncation
    and discretisation cancel in q_k / c_k."""
    if kern is None or kern["W"] <= 0:
        return None
    return float(kern["w"][kern["theta"] <= rho].sum() / kern["W"])


# ------------------------------------------------------------ cone votes

def cone_votes(splat, tree, label, origin, pt, sigma_rad, span, S, eps,
               depth_margin=0.5):
    """Angular-Gaussian-weighted label masses over the visible surface in the cone.

    Returns (votes, kern): votes = {label: mass}, kern = {"W": total kernel
    mass, "theta": per-pixel angle to the axis, "w": per-pixel weight} (both
    flattened, S*S). kern is None when the fixation is degenerate (< 5 cm).

    depth_margin: surfaces farther than the fixation point by more than this
    (along the ray) do not vote. Thin structures (robot arm links) let cone
    pixels leak past the target onto whatever stands a meter behind it, and
    the background then outvotes the sparse foreground. Nearer surfaces keep
    voting -- an occluder in front of the fixation is usually the true target.
    """
    d0 = pt - origin
    dist0 = float(np.linalg.norm(d0))
    if dist0 < 0.05:
        return {}, None
    depth, alpha, dirs, tmul = splat.patch_along_ray(origin, d0 / dist0, span * sigma_rad, S)
    cosang = np.clip(dirs @ (d0 / dist0), -1.0, 1.0)
    theta = np.arccos(cosang)
    w = np.exp(-theta ** 2 / (2 * sigma_rad ** 2))
    ok = (depth > 0.05) & (depth < 12.0)
    if depth_margin > 0:
        ok &= (depth * tmul) < dist0 + depth_margin
    X = origin + (depth * tmul)[..., None] * dirs
    dd, idx = tree.query(X[ok], k=1, distance_upper_bound=eps, workers=-1)
    hit = np.isfinite(dd)
    votes: dict[int, float] = {}
    m_ok = (w * alpha)[ok]
    for lab, wi in zip(label[idx[hit]], m_ok[hit]):
        votes[int(lab)] = votes.get(int(lab), 0.0) + float(wi)
    return votes, {"W": float(w.sum()), "theta": theta.ravel(), "w": w.ravel()}


def pooled_centroids_by_name(instances, names):
    """Gaussian-count-weighted centroid for each resolved object name.

    Several SAM instances may be hand-merged by assigning them the same name.
    Their canonical coordinate must describe the whole named object, rather
    than whichever sub-instance happens to win the current gaze vote.
    """
    sums: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for inst in instances:
        lab = int(inst["id"])
        name = names.get(str(lab), "") or f"object#{lab}"
        weight = max(float(inst.get("n_gaussians", 1)), 1.0)
        sums[name] = sums.get(name, np.zeros(3, dtype=float)) + weight * np.asarray(inst["centroid"], float)
        weights[name] = weights.get(name, 0.0) + weight
    return {name: (total / weights[name]).tolist() for name, total in sums.items()}


# ------------------------------------------------------------ verdict (shared by gaze_live and offline)

def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def rank_votes(votes, kern, name_of, targets, object_centroids, radii=None,
               sigma_rad=None, dist=None, rank_by="capture"):
    """Pool votes by resolved name; only the TARGET vocabulary competes.

    Returns None when there is no mass at all (degenerate fixation). Otherwise
    a verdict dict; object is None when no target received mass (the gaze sat
    on a place / background / clutter -- `surface` says which).

    Per target:  q = m/W (absolute cone mass), share = m / sum(target m),
    c = disk_capture(rho_k), capture = q/c, miss_deg = sigma*sqrt(-2 ln capture).
    """
    if not votes:
        return None
    W = kern["W"] if kern is not None else float(sum(votes.values()))
    pooled: dict[str, dict] = {}
    for lab, v in votes.items():
        nm = name_of(lab)
        p = pooled.setdefault(nm, {"v": 0.0, "labels": []})
        p["v"] += v
        p["labels"].append(lab)
    tgt = {n: p for n, p in pooled.items() if n in targets}
    inv = {n: p for n, p in pooled.items() if n not in targets}
    T = float(sum(p["v"] for p in tgt.values()))
    p_none = max(0.0, 1.0 - T / W) if W > 0 else 1.0
    surf = max(inv.items(), key=lambda kv: kv[1]["v"]) if inv else None
    out = {"p_none": _r(p_none), "rank_by": rank_by,
           "surface": surf[0] if surf else None,
           "surface_q": _r(surf[1]["v"] / W) if (surf and W > 0) else 0.0}
    if T <= 0:
        out.update(object=None, object_label=-1, vote_share=0.0, q=0.0, capture=None,
                   miss_deg=None, object_centroid_world=None, candidates=[])
        return out
    sigma_deg = math.degrees(sigma_rad) if sigma_rad else None
    cands = []
    for n, p in tgt.items():
        q, share = p["v"] / W, p["v"] / T
        c = cap = miss = None
        if kern is not None and radii and n in radii and sigma_rad and dist:
            c = disk_capture(kern, math.atan2(radii[n], dist))
            if c and c > 0:
                cap = q / c
                miss = max(0.0, sigma_deg * math.sqrt(-2.0 * math.log(min(cap, 1.0)))) if cap > 0 else None
        cands.append({"name": n, "share": share, "q": q, "c": c, "capture": cap, "miss_deg": miss,
                      "labels": sorted(p["labels"]),
                      "_best": max(p["labels"], key=lambda l: votes[l])})
    by_capture = rank_by == "capture" and all(c["capture"] is not None for c in cands)
    cands.sort(key=(lambda c: c["capture"]) if by_capture else (lambda c: c["share"]), reverse=True)
    best = cands[0]
    out.update(object=best["name"], object_label=int(best["_best"]),
               vote_share=_r(best["share"]), q=_r(best["q"]), capture=_r(best["capture"]),
               miss_deg=_r(best["miss_deg"], 2),
               # Gaze hit point varies; this canonical coordinate is fixed and
               # pools every same-named SAM part into one whole-object centroid.
               object_centroid_world=object_centroids.get(best["name"]),
               candidates=[{"name": c["name"], "share": _r(c["share"]), "q": _r(c["q"]),
                            "capture": _r(c["capture"]), "miss_deg": _r(c["miss_deg"], 2),
                            "labels": c["labels"]} for c in cands[:3]])
    return out


def verdict_text(v) -> str:
    """One-line human summary used by both tools' logs."""
    if v is None:
        return "(no mass)"
    if v["object"] is None:
        return f"(none: {v['surface'] or '-'} {v['surface_q']:.0%})  none {v['p_none']:.0%}"
    s = f"{v['object']:<14} share {v['vote_share']:>4.0%}"
    if v.get("capture") is not None:
        s += f"  cap {v['capture']:.2f} (miss {v['miss_deg']:.1f}deg)"
    s += f"  none {v['p_none']:.0%}"
    if v.get("surface"):
        s += f"  on {v['surface']}"
    return s


# ------------------------------------------------------------ offline main

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
    bg = load_background(meta)
    name_of = make_name_of(bg, names)
    places = load_places(seg, args.places)
    targets = {v for v in names.values() if v and v not in places}
    radii = object_radii_by_name(xyz, label, names, only=targets)
    object_centroids = pooled_centroids_by_name(meta["instances"], names)
    print(f"targets {sorted(targets)}  places {sorted(places)}  "
          f"radii " + ", ".join(f"{n}:{r*100:.1f}cm" for n, r in sorted(radii.items())))
    tree = cKDTree(xyz)  # every labeled gaussian: invalid mass must stay invalid
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
        sigma_deg = sigma_deg or 1.0
        print(f"cone mode: sigma {sigma_deg:.2f} deg, half-angle {args.span_sigmas:.1f} sigma, "
              f"patch {args.patch}, hit-eps {args.hit_eps*100:.0f}cm, rank by {args.rank}")

    print(f"{len(fixes)} fixations vs {len(xyz)} labeled gaussians "
          f"({len(meta['instances'])} instances)")
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
        dist = float(np.linalg.norm(pt - np.asarray(origin, float))) if origin is not None else None
        sigma_rad = np.radians(sigma_deg) if sigma_deg else None
        if args.cone and origin is not None:
            S = max(args.patch, int(round(2 * args.span_sigmas * sigma_deg / args.patch_deg)) | 1)
            votes, kern = cone_votes(splat, tree, label, np.asarray(origin, float), pt,
                                     sigma_rad, args.span_sigmas, S, args.hit_eps)
            entry["mode"] = "cone"
        else:
            if args.cone:
                entry["mode"] = "sphere-fallback"  # old fixations file without origin_world
            idx = tree.query_ball_point(pt, args.radius)
            votes, kern = {}, None
            if idx:
                d = np.linalg.norm(xyz[idx] - pt, axis=1)
                w = 1.0 / np.maximum(d, 0.01)
                for lab, wi in zip(label[idx], w):
                    votes[int(lab)] = votes.get(int(lab), 0.0) + float(wi)
        v = rank_votes(votes, kern, name_of, targets, object_centroids, radii,
                       sigma_rad, dist, rank_by=args.rank)
        if v is None:
            entry.update(object=None, note="nothing labeled in gaze neighborhood")
            print(f"{k:>3} {t:>7.1f} {str(np.round(pt,2)):<26} (nothing nearby)")
            results.append(entry)
            continue
        entry.update(v)
        results.append(entry)
        print(f"{k:>3} {t:>7.1f} {str(np.round(pt,2)):<26} {verdict_text(v)}")

    out = Path(args.out) if args.out else Path(args.fixations).expanduser().with_name(
        Path(args.fixations).stem + "_objects.json")
    out.write_text(json.dumps({"source": str(args.fixations), "seg_dir": str(seg),
                               "mode": "cone" if args.cone else "sphere", "rank_by": args.rank,
                               "sigma_deg": sigma_deg, "span_sigmas": args.span_sigmas,
                               "radius_m": args.radius, "targets": sorted(targets),
                               "places": sorted(places), "fixations": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
