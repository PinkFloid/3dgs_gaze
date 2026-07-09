#!/usr/bin/env python
"""Per-recording gaze accuracy from the protocol's tag-stare stamps.

The recording protocol has the wearer stare at a surveyed wall tag for 2-3 s
right after pressing R and again before stopping. Tag centers are known to mm
(tags_world.json), so those two windows measure the gaze layer directly:

  bias  = median angular offset gaze-minus-tag. Systematic per wearing;
          gaze_to_world subtracts it automatically once this file exists.
  sigma = residual scatter after bias removal. Feeds the cone width of
          gaze_object --cone. This is the honest per-fixation uncertainty:
          bias does NOT average out over a fixation's samples, jitter does.
  drift = head-window bias vs tail-window bias (slow slippage; rec001's
          7-18 deg would light up here immediately).

Writes <recording>/gaze_precision.json and prints a verdict.

  python tools/gaze_precision.py --recording ~/recordings/<date>/<n> \
      [--poses <rec>/poses.jsonl] [--head-window 12] [--tail-window 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import gaze_to_world as g2w


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True)
    p.add_argument("--poses", default=None, help="Default: <recording>/poses.jsonl")
    p.add_argument("--tags", default=str(root / "world_size/tags_world.json"))
    p.add_argument("--calib", default=str(root / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--head-window", type=float, default=15.0, help="Search the first N s (0 = skip).")
    p.add_argument("--tail-window", type=float, default=30.0,
                   help="Search the last N s (0 = skip). Generous: rec002's tail stare sat "
                        "12-30 s before the end and a 12 s window missed it.")
    p.add_argument("--on-tag-deg", type=float, default=4.0,
                   help="Samples within this angle of a tag count as staring at it.")
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--max-gap", type=float, default=1.0)
    p.add_argument("--out", default=None, help="Default: <recording>/gaze_precision.json")
    return p.parse_args()


def window_stats(gaze, poses, tag_centers, K_img, D_fish, W, H, t_lo, t_hi, on_tag_rad, min_samples):
    """Best-tag angular offsets (normalized units ~ rad) for gaze samples in [t_lo, t_hi]."""
    per_tag: dict[str, list] = {tid: [] for tid in tag_centers}
    for g in gaze:
        t = g["timestamp"]
        if not (t_lo <= t <= t_hi):
            continue
        T = poses.query(t)
        if T is None:
            continue
        u = g["norm_pos"][0] * W
        v = (1.0 - g["norm_pos"][1]) * H
        pn = cv2.fisheye.undistortPoints(np.array([[[u, v]]], np.float64), K_img, D_fish).reshape(2)
        R, cam = T[:3, :3], T[:3, 3]
        for tid, c in tag_centers.items():
            x = R.T @ (c - cam)
            if x[2] < 0.3:
                continue
            per_tag[tid].append((t, pn - x[:2] / x[2]))   # gaze minus target
    best = None
    for tid, rows in per_tag.items():
        if len(rows) < min_samples:
            continue
        ts = np.array([r[0] for r in rows])
        offs = np.array([r[1] for r in rows])
        on = np.linalg.norm(offs, axis=1) < on_tag_rad
        if on.sum() < min_samples:
            continue
        med = np.linalg.norm(np.median(offs[on], axis=0))
        if best is None or med < best[0]:
            best = (med, tid, ts[on], offs[on])
    if best is None:
        return None
    _, tid, ts, on = best
    bias = np.median(on, axis=0)
    resid = on - bias
    sigma = float(np.sqrt(np.mean(resid ** 2)))      # pooled per-axis, 1D-equivalent
    return {"tag": tid, "n": int(len(on)), "t": round(float(np.median(ts)), 3),
            "bias_deg": [round(float(np.degrees(b)), 3) for b in bias],
            "sigma_deg": round(float(np.degrees(sigma)), 3)}


def main() -> int:
    args = parse_args()
    rec = Path(args.recording).expanduser()
    poses = g2w.PoseTrack(Path(args.poses) if args.poses else rec / "poses.jsonl", args.max_gap)
    gaze = g2w.load_gaze(rec, args.min_confidence)

    z = np.load(args.calib, allow_pickle=True)
    K_fish = np.asarray(z["camera_matrix"], np.float64)
    D_fish = np.asarray(z["dist_coeffs"], np.float64).reshape(-1, 1)[:4]
    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K_img = K_fish.copy()
    K_img[0] *= W / 1920.0
    K_img[1] *= H / 1080.0

    tags = json.loads(Path(args.tags).read_text(encoding="utf-8"))["tags"]
    tag_centers = {tid: np.array(t["T_world_tag"], float)[:3, 3] for tid, t in tags.items()}

    t0, t1 = gaze[0]["timestamp"], gaze[-1]["timestamp"]
    on_tag = np.tan(np.radians(args.on_tag_deg))
    windows = {}
    if args.head_window > 0:
        windows["head"] = window_stats(gaze, poses, tag_centers, K_img, D_fish, W, H,
                                       t0, t0 + args.head_window, on_tag, args.min_samples)
    if args.tail_window > 0:
        windows["tail"] = window_stats(gaze, poses, tag_centers, K_img, D_fish, W, H,
                                       t1 - args.tail_window, t1, on_tag, args.min_samples)
    found = {k: v for k, v in windows.items() if v}
    for k in windows:
        w = windows.get(k)
        if w:
            print(f"{k}: tag {w['tag']}, n={w['n']}, bias ({w['bias_deg'][0]:+.2f},{w['bias_deg'][1]:+.2f}) deg, "
                  f"sigma {w['sigma_deg']:.2f} deg")
        else:
            print(f"{k}: no tag stare found (need >= {args.min_samples} samples within {args.on_tag_deg} deg)")
    if not found:
        raise SystemExit("No usable tag-stare window -- was the protocol followed?")

    ws = list(found.values())
    n_tot = sum(w["n"] for w in ws)
    bias = np.sum([np.array(w["bias_deg"]) * w["n"] for w in ws], axis=0) / n_tot
    sigma = float(np.sqrt(sum(w["n"] * w["sigma_deg"] ** 2 for w in ws) / n_tot))
    drift = None
    if "head" in found and "tail" in found:
        drift = round(float(np.linalg.norm(
            np.array(found["head"]["bias_deg"]) - np.array(found["tail"]["bias_deg"]))), 3)
        # downstream interpolates bias(t) between the stamps, so only the
        # nonlinear remainder of the drift stays in sigma (~drift/4), not drift/2
        sigma = round(float(np.hypot(sigma, drift / 4)), 3)

    verdict = "ok"
    if sigma > 2.5:
        verdict = "poor-sigma: gaze layer noisy, consider re-calibration"
    if drift is not None and drift > 4.0:
        verdict = f"drifting {drift:.1f} deg head->tail: too much for linear correction, re-record"
    elif drift is not None and drift > 1.5:
        verdict = f"drift {drift:.1f} deg head->tail: linear bias(t) correction applied downstream"
    print(f"combined: bias ({bias[0]:+.2f},{bias[1]:+.2f}) deg, sigma {sigma:.2f} deg, "
          f"drift {drift if drift is not None else '-'} deg -> {verdict}")

    stamps = sorted((w for w in found.values()), key=lambda w: w["t"])
    out = Path(args.out) if args.out else rec / "gaze_precision.json"
    out.write_text(json.dumps({
        "recording": str(rec), "convention": "bias = gaze minus target, undistorted camera frame",
        "bias_deg": [round(float(b), 3) for b in bias], "sigma_deg": round(sigma, 3),
        "drift_deg": drift, "verdict": verdict,
        "stamps": stamps,  # gaze_to_world lerps bias(t) through these
        "windows": windows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
