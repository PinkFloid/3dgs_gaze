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

Each stamp must be one continuous on-tag episode; radial MAD rejects outliers.
  python tools/gaze_precision.py --recording ~/recordings/<date>/<n> \\
      [--poses <rec>/poses.jsonl] [--head-window 15] [--tail-window 30]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import gaze_to_world as g2w


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "SceneRebuild"
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
    p.add_argument("--min-dwell", type=float, default=0.25,
                   help="Minimum continuous on-tag episode (s). New recordings should target >=0.8s.")
    p.add_argument("--max-sample-gap", type=float, default=0.10,
                   help="Split an on-tag episode when consecutive samples are farther apart than this (s).")
    p.add_argument("--mad-k", type=float, default=3.5,
                   help="Radial MAD multiplier for rejecting gaze outliers inside a stamp.")
    p.add_argument("--head-tag", type=int, default=None,
                   help="Optional expected tag id for the head stamp; auto-select when omitted.")
    p.add_argument("--tail-tag", type=int, default=None,
                   help="Optional expected tag id for the tail stamp; auto-select when omitted.")
    p.add_argument("--dictionary", default="DICT_6X6_250",
                   help="ArUco dictionary for the stamp visibility check.")
    p.add_argument("--no-visibility-check", action="store_true",
                   help="Skip verifying that the stamp tag is actually DETECTED in the world "
                        "video. The on-tag test alone is purely geometric: pairing a recording "
                        "with a survey from another deployment era lets a phantom tag win "
                        "(002 + v2 survey picked tag 83, which did not physically exist yet, "
                        "and produced a sign-flipped tail bias).")
    p.add_argument("--out", default=None, help="Default: <recording>/gaze_precision.json")
    return p.parse_args()


def _split_contiguous_runs(rows, max_sample_gap: float):
    """Split sorted (timestamp, offset) rows at missing-sample gaps."""
    runs, current = [], []
    for row in rows:
        if current and row[0] - current[-1][0] > max_sample_gap:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return runs


def _robust_run_stats(tid, rows, min_samples: int, mad_k: float):
    """Robust bias/scatter estimate for one continuous on-tag episode."""
    ts = np.array([r[0] for r in rows], float)
    offsets = np.array([r[1] for r in rows], float)
    raw_ts = ts.copy()
    bias = np.median(offsets, axis=0)
    raw_sigma = float(np.sqrt(np.mean((offsets - bias) ** 2)))
    radial = np.linalg.norm(offsets - bias, axis=1)
    radial_med = float(np.median(radial))
    radial_mad = float(1.4826 * np.median(np.abs(radial - radial_med)))
    mad_floor = float(np.tan(np.radians(0.05)))
    cutoff = radial_med + mad_k * max(radial_mad, mad_floor)
    inlier = radial <= cutoff
    if int(inlier.sum()) < min_samples:
        return None

    ts, offsets = ts[inlier], offsets[inlier]
    bias = np.median(offsets, axis=0)
    resid = offsets - bias
    sigma_xy = np.sqrt(np.mean(resid ** 2, axis=0))
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    cov = np.cov(resid.T, bias=True) if len(resid) > 1 else np.zeros((2, 2))
    bias_deg = np.degrees(np.arctan(bias))
    sigma_xy_deg = np.degrees(np.arctan(sigma_xy))
    sigma_deg = float(np.degrees(np.arctan(sigma)))
    gaps = np.diff(raw_ts)
    return {
        "tag": tid,
        "n": int(len(offsets)),
        "n_raw": int(len(rows)),
        "inlier_ratio": round(float(len(offsets) / len(rows)), 3),
        "t": round(float(np.median(ts)), 3),
        "t_start": round(float(raw_ts[0]), 3),
        "t_end": round(float(raw_ts[-1]), 3),
        "duration_s": round(float(raw_ts[-1] - raw_ts[0]), 3),
        "inlier_span_s": round(float(ts[-1] - ts[0]), 3),
        "max_sample_gap_s": round(float(gaps.max()), 4) if len(gaps) else 0.0,
        "bias_norm": [float(v) for v in bias],
        "bias_deg": [round(float(v), 3) for v in bias_deg],
        "sigma_deg": round(sigma_deg, 3),
        "sigma_raw_deg": round(float(np.degrees(np.arctan(raw_sigma))), 3),
        "sigma_x_deg": round(float(sigma_xy_deg[0]), 3),
        "sigma_y_deg": round(float(sigma_xy_deg[1]), 3),
        "cov_norm": np.asarray(cov, float).tolist(),
        "radial_mad_deg": round(float(np.degrees(np.arctan(radial_mad))), 3),
    }


def window_stats(gaze, poses, tag_centers, K_img, D_fish, W, H, t_lo, t_hi,
                 on_tag_rad, min_samples, min_dwell, max_sample_gap, mad_k,
                 preferred_tag=None, visible=None):
    """Best continuous tag-stare episode inside [t_lo, t_hi].

    visible(tag_id, t) -> bool, when given, must confirm the tag is physically
    detected in the world video near t; candidates failing it are skipped in
    score order (phantom-tag guard for era-mismatched surveys)."""
    if preferred_tag is not None:
        preferred_tag = str(preferred_tag)
        if preferred_tag not in tag_centers:
            return None
        centers = {preferred_tag: tag_centers[preferred_tag]}
    else:
        centers = tag_centers

    per_tag: dict[str, list] = {tid: [] for tid in centers}
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
        for tid, c in centers.items():
            x = R.T @ (c - cam)
            if x[2] < 0.3:
                continue
            offset = pn - x[:2] / x[2]  # gaze minus target, normalized tangent plane
            if np.linalg.norm(offset) < on_tag_rad:
                per_tag[tid].append((t, offset))

    candidates = []
    for tid, rows in per_tag.items():
        for run in _split_contiguous_runs(rows, max_sample_gap):
            duration = run[-1][0] - run[0][0]
            if len(run) < min_samples or duration < min_dwell:
                continue
            stats = _robust_run_stats(tid, run, min_samples, mad_k)
            if stats is None:
                continue
            # Prefer a tag close to the raw gaze, but penalize a noisy accidental pass.
            score = np.linalg.norm(np.asarray(stats["bias_norm"])) + np.tan(np.radians(stats["sigma_raw_deg"]))
            stats["selection_score_deg"] = round(float(np.degrees(np.arctan(score))), 3)
            candidates.append((score, -stats["duration_s"], stats))
    if not candidates:
        return None
    for _, _, stats in sorted(candidates, key=lambda row: (row[0], row[1])):
        if visible is None or visible(stats["tag"], stats["t"]):
            return stats
    return None

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

    if not gaze:
        raise SystemExit("No gaze samples passed --min-confidence.")
    t0, t1 = gaze[0]["timestamp"], gaze[-1]["timestamp"]
    on_tag = np.tan(np.radians(args.on_tag_deg))
    quality_warnings = []

    # phantom-tag guard: the stamp tag must be ArUco-DETECTED near the episode
    # midpoint. Fail-open only when the video itself is unreadable.
    world_ts = np.load(rec / "world_timestamps.npy")
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary)),
        cv2.aruco.DetectorParameters())

    def tag_seen(tid, t_mid):
        if args.no_visibility_check:
            return True
        readable = False
        for dt in (0.0, -0.15, 0.15):
            fi = int(np.clip(np.searchsorted(world_ts, t_mid + dt), 0, len(world_ts) - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            readable = True
            _, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is not None and int(tid) in ids.flatten():
                return True
        if not readable:
            quality_warnings.append("visibility check skipped: world.mp4 frames unreadable")
            return True
        quality_warnings.append(
            f"rejected stamp candidate: tag {tid} never detected in video near t={t_mid:.1f} "
            "(phantom tag? recording paired with a survey from another era?)")
        return False
    head_range = [t0, min(t0 + args.head_window, t1)] if args.head_window > 0 else None
    tail_range = [max(t1 - args.tail_window, t0), t1] if args.tail_window > 0 else None
    if head_range and tail_range and head_range[1] > tail_range[0]:
        cut = (head_range[1] + tail_range[0]) / 2.0
        head_range[1] = cut
        tail_range[0] = np.nextafter(cut, np.inf)
        quality_warnings.append("head/tail search windows overlapped and were made disjoint")

    windows = {}
    if head_range:
        windows["head"] = window_stats(
            gaze, poses, tag_centers, K_img, D_fish, W, H, *head_range,
            on_tag, args.min_samples, args.min_dwell, args.max_sample_gap,
            args.mad_k, args.head_tag, visible=tag_seen)
    if tail_range:
        windows["tail"] = window_stats(
            gaze, poses, tag_centers, K_img, D_fish, W, H, *tail_range,
            on_tag, args.min_samples, args.min_dwell, args.max_sample_gap,
            args.mad_k, args.tail_tag, visible=tag_seen)
    found = {k: v for k, v in windows.items() if v}
    if len(windows) > 1 and len(found) < len(windows):
        missing = ", ".join(k for k, v in windows.items() if v is None)
        quality_warnings.append(
            f"missing usable {missing} stamp; drift cannot be measured")
    for k in windows:
        w = windows.get(k)
        if w:
            print(f"{k}: tag {w['tag']}, {w['duration_s']:.2f}s, n={w['n']}/{w['n_raw']}, "
                  f"bias ({w['bias_deg'][0]:+.2f},{w['bias_deg'][1]:+.2f}) deg, "
                  f"sigma {w['sigma_deg']:.2f} deg")
        else:
            print(f"{k}: no tag stare found (need a continuous >= {args.min_dwell:.2f}s episode, "
                  f">= {args.min_samples} samples within {args.on_tag_deg} deg)")
    if not found:
        raise SystemExit("No usable tag-stare window -- was the protocol followed?")

    ws = list(found.values())
    n_tot = sum(w["n"] for w in ws)
    bias_norm = np.sum([np.array(w["bias_norm"]) * w["n"] for w in ws], axis=0) / n_tot
    bias = np.degrees(np.arctan(bias_norm))
    sigma_measured = float(np.sqrt(sum(w["n"] * w["sigma_deg"] ** 2 for w in ws) / n_tot))
    drift = None
    drift_penalty = 0.0
    if "head" in found and "tail" in found:
        drift = round(float(np.linalg.norm(
            np.array(found["head"]["bias_deg"]) - np.array(found["tail"]["bias_deg"]))), 3)
        # Downstream interpolates bias(t); drift/4 is an empirical allowance
        # for the nonlinear remainder that a two-stamp linear model cannot remove.
        drift_penalty = drift / 4.0
    sigma = round(float(np.hypot(sigma_measured, drift_penalty)), 3)
    for key, stamp in found.items():
        if stamp["duration_s"] < 0.8:
            quality_warnings.append(
                f"{key} stamp is only {stamp['duration_s']:.2f}s; record >=0.8s on the tag center")
        if stamp["inlier_ratio"] < 0.8:
            quality_warnings.append(
                f"{key} stamp kept only {stamp['inlier_ratio']:.0%} after outlier rejection")

    verdict = "ok"
    if len(windows) > 1 and len(found) < len(windows):
        verdict = "partial: only one usable precision stamp; drift unavailable"
    if sigma > 2.5:
        verdict = "poor-sigma: gaze layer noisy, consider re-calibration"
    if drift is not None and drift > 4.0:
        verdict = f"drifting {drift:.1f} deg head->tail: too much for linear correction, re-record"
    elif drift is not None and drift > 1.5:
        verdict = f"drift {drift:.1f} deg head->tail: linear bias(t) correction applied downstream"
    print(f"combined: bias ({bias[0]:+.2f},{bias[1]:+.2f}) deg, "
          f"sigma {sigma:.2f} deg (measured {sigma_measured:.2f} + drift penalty {drift_penalty:.2f}), "
          f"drift {drift if drift is not None else '-'} deg -> {verdict}")
    for warning in quality_warnings:
        print(f"WARNING: {warning}")

    stamps = sorted((w for w in found.values()), key=lambda w: w["t"])
    out = Path(args.out) if args.out else rec / "gaze_precision.json"
    tags_path = Path(args.tags)
    tags_sha1 = hashlib.sha1(tags_path.read_bytes()).hexdigest()[:12]
    out.write_text(json.dumps({
        "recording": str(rec), "convention": "bias = gaze minus target, undistorted camera frame",
        # era pairing: stamps are only valid against the survey the poses were solved with
        "tags_file": str(tags_path), "tags_sha1": tags_sha1,
        "bias_norm": [float(b) for b in bias_norm],
        "bias_deg": [round(float(b), 3) for b in bias],
        "sigma_deg": round(sigma, 3),  # backward-compatible effective cone sigma
        "sigma_measured_deg": round(float(sigma_measured), 3),
        "drift_penalty_deg": round(float(drift_penalty), 3),
        "drift_deg": drift, "verdict": verdict,
        "quality_warnings": quality_warnings,
        "stamps": stamps,  # gaze_to_world lerps bias(t) through these
        "windows": windows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
