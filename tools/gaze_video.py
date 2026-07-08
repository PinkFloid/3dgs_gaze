#!/usr/bin/env python
"""Rebuild the gaze-overlay video from a Pupil Capture recording.

Draws the raw Pupil gaze on every world frame (like Pupil Player's world
video export, no Player needed): current gaze circle (color = confidence),
a short trail of recent gaze, and a timestamp HUD. Optionally overlays the
pipeline's world-space fixation verdicts (object names) if the
world_fixations_objects.json from gaze_object.py is given.

Examples:
  python tools/gaze_video.py --recording ~/recordings/2026_07_05/002
  python tools/gaze_video.py --recording ~/recordings/2026_07_05/002 \
      --objects ~/recordings/2026_07_05/002/world_fixations_objects.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import msgpack
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True, help="Pupil Capture recording dir.")
    p.add_argument("--out", default=None, help="Output mp4 (default: <recording>/gaze_overlay.mp4).")
    p.add_argument("--min-confidence", type=float, default=0.4,
                   help="Hide gaze below this confidence (blink/noise).")
    p.add_argument("--trail", type=float, default=0.4, help="Gaze trail length in seconds (0 = off).")
    p.add_argument("--objects", default=None,
                   help="world_fixations_objects.json -- overlay object verdicts during fixations.")
    p.add_argument("--poses", default=None,
                   help="Localizer --log JSONL; needed for any 3D box drawing.")
    p.add_argument("--boxes", choices=["target", "named", "both", "off"], default="target",
                   help="target: box the instance the current fixation was assigned to (works "
                        "unnamed); named: every named instance every frame; both; off.")
    p.add_argument("--seg-dir", default=None,
                   help="Default: lab_result/segmentation_sam if present, else lab_result/segmentation.")
    p.add_argument("--calib", default=str(Path(__file__).resolve().parent.parent / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--start", type=float, default=0.0, help="Start offset (s).")
    p.add_argument("--duration", type=float, default=None, help="Clip length (s), default all.")
    return p.parse_args()


BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def load_instances(seg_dir: Path) -> dict[int, dict]:
    """All instances (named or not) as {id: {name, corners, color}}."""
    meta = json.loads((seg_dir / "instances.json").read_text(encoding="utf-8"))
    names_p = seg_dir / "names.json"
    names = json.loads(names_p.read_text(encoding="utf-8")) if names_p.exists() else {}
    out = {}
    for inst in meta["instances"]:
        lo, hi = np.array(inst["bbox_min"]), np.array(inst["bbox_max"])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        rng = np.random.default_rng(inst["id"])
        color = tuple(int(c) for c in rng.integers(80, 255, 3))
        out[inst["id"]] = {"name": names.get(str(inst["id"]), ""),
                           "corners": corners, "color": color}
    return out


def draw_instances(frame, T_world_cam, instances, K, D, thick=2):
    w2c = np.linalg.inv(T_world_cam)
    rvec, _ = cv2.Rodrigues(np.ascontiguousarray(w2c[:3, :3]))
    tvec = np.ascontiguousarray(w2c[:3, 3]).reshape(3, 1)
    for inst in instances:
        cam_pts = (w2c[:3, :3] @ inst["corners"].T).T + w2c[:3, 3]
        if (cam_pts[:, 2] < 0.15).any():   # partially behind camera: skip, fisheye proj degenerates
            continue
        px, _ = cv2.fisheye.projectPoints(
            inst["corners"].reshape(-1, 1, 3).astype(np.float64), rvec, tvec, K, D)
        px = px.reshape(-1, 2)
        H, W = frame.shape[:2]
        if not ((px[:, 0] > -W) & (px[:, 0] < 2 * W) & (px[:, 1] > -H) & (px[:, 1] < 2 * H)).all():
            continue
        pts = np.int32(px)
        for a, b in BOX_EDGES:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), inst["color"], thick)
        if inst["name"]:
            top = pts[pts[:, 1].argmin()]
            cv2.putText(frame, inst["name"], (top[0] - 20, max(top[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, inst["color"], 2)


def load_gaze(rec: Path):
    ts, xy, conf = [], [], []
    with open(rec / "gaze.pldata", "rb") as f:
        for _topic, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            r = msgpack.unpackb(payload, strict_map_key=False)
            ts.append(r["timestamp"])
            xy.append(r["norm_pos"])
            conf.append(r.get("confidence", 0.0))
    order = np.argsort(ts)
    return np.array(ts)[order], np.array(xy)[order], np.array(conf)[order]


def main() -> int:
    args = parse_args()
    rec = Path(args.recording).expanduser()
    out_path = Path(args.out) if args.out else rec / "gaze_overlay.mp4"

    world_ts = np.load(rec / "world_timestamps.npy")
    g_ts, g_xy, g_conf = load_gaze(rec)
    print(f"{len(world_ts)} world frames, {len(g_ts)} gaze samples")

    fixations = []
    if args.objects:
        doc = json.loads(Path(args.objects).expanduser().read_text(encoding="utf-8"))
        fixations = [f for f in doc["fixations"] if f.get("object")]
        print(f"{len(fixations)} object-labeled fixations to overlay")

    poses = K_fish = D_fish = None
    inst_by_id: dict[int, dict] = {}
    named_instances: list[dict] = []
    if args.poses and args.boxes != "off":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from gaze_to_world import PoseTrack
        poses = PoseTrack(Path(args.poses), max_gap=1.0)
        root = Path(__file__).resolve().parent.parent
        seg = Path(args.seg_dir) if args.seg_dir else (
            root / "lab_result/segmentation_sam" if (root / "lab_result/segmentation_sam").exists()
            else root / "lab_result/segmentation")
        inst_by_id = load_instances(seg)
        named_instances = [v for v in inst_by_id.values() if v["name"]]
        z = np.load(args.calib, allow_pickle=True)
        K_fish = np.asarray(z["camera_matrix"], np.float64)
        D_fish = np.asarray(z["dist_coeffs"], np.float64).reshape(-1, 1)[:4]
        print(f"{len(inst_by_id)} instances from {seg.name} ({len(named_instances)} named), "
              f"box mode: {args.boxes}")

    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("cannot open VideoWriter (mp4v)")
    K_img = None
    if K_fish is not None:
        K_img = K_fish.copy()
        K_img[0] *= W / 1920.0
        K_img[1] *= H / 1080.0

    t0 = world_ts[0]
    fi = 0
    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok or fi >= len(world_ts):
            break
        t = world_ts[fi]
        fi += 1
        if t - t0 < args.start:
            continue
        if args.duration and t - t0 > args.start + args.duration:
            break

        T = poses.query(t) if poses is not None else None
        if T is not None and args.boxes in ("named", "both") and named_instances:
            draw_instances(frame, T, named_instances, K_img, D_fish)

        # trail: gaze samples in the last args.trail seconds
        if args.trail > 0:
            i0, i1 = np.searchsorted(g_ts, [t - args.trail, t])
            for k in range(i0, i1):
                if g_conf[k] < args.min_confidence:
                    continue
                age = (t - g_ts[k]) / args.trail          # 0 new .. 1 old
                px = (int(g_xy[k][0] * W), int((1 - g_xy[k][1]) * H))
                cv2.circle(frame, px, 6, (0, int(255 * (1 - age)), int(255 * age)), -1)

        # current gaze: nearest sample within 50 ms
        k = int(np.clip(np.searchsorted(g_ts, t), 1, len(g_ts) - 1))
        k = k if abs(g_ts[k] - t) < abs(g_ts[k - 1] - t) else k - 1
        if abs(g_ts[k] - t) < 0.05 and g_conf[k] >= args.min_confidence:
            u, v = int(g_xy[k][0] * W), int((1 - g_xy[k][1]) * H)
            good = g_conf[k] >= 0.8
            color = (0, 220, 0) if good else (0, 165, 255)
            cv2.circle(frame, (u, v), 28, color, 4)
            cv2.line(frame, (u - 40, v), (u + 40, v), color, 2)
            cv2.line(frame, (u, v - 40), (u, v + 40), color, 2)
            cv2.putText(frame, f"{g_conf[k]:.2f}", (u + 34, v - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # object verdict overlay during labeled fixations (+ box around the target)
        for fx in fixations:
            if fx["t_start"] <= t <= fx["t_end"] + 0.15:
                lab = fx.get("object_label")
                if T is not None and args.boxes in ("target", "both") and isinstance(lab, int):
                    labs = [lab]
                    cands = fx.get("candidates")
                    if cands and isinstance(cands[0], dict) and cands[0].get("labels"):
                        labs = cands[0]["labels"]  # all ids pooled under the winning name
                    members = [inst_by_id[l] for l in labs
                               if isinstance(l, int) and l >= 10 and l in inst_by_id]
                    if members:
                        col = members[0]["color"]  # one color for the whole named object
                        extra = [dict(m, color=col, name="") for m in members[1:]]
                        draw_instances(frame, T, extra, K_img, D_fish, thick=2)
                        draw_instances(frame, T, [members[0]], K_img, D_fish, thick=3)
                c = fx["centroid_world"]
                share = fx.get("vote_share", 0)
                cv2.putText(frame, f"-> {fx['object']} ({share:.0%})  [{c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}]m",
                            (30, H - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                break

        cv2.putText(frame, f"t={t - t0:6.2f}s", (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        writer.write(frame)
        n_written += 1
        if n_written % 300 == 0:
            print(f"  {n_written} frames written...")

    writer.release()
    print(f"wrote {out_path} ({n_written} frames, {n_written / fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
