#!/usr/bin/env python
"""Real-time gaze -> world -> object intent, with a live overlay UI.

Single process, event-driven:

  frame.world (ZMQ)  -> ArUco -> PnP -> T_world_cam      (pupil_localizer parts)
  gaze.* (ZMQ)       -> undistort -> bias -> ray -> splat depth intersect
                     -> incremental world-space clustering
  fixation closes    -> cone posterior over named instances (gaze_object parts)
  gaze passes a tag  -> online bias/sigma re-estimation (rolling precision stamp)

UI (cv2 window, CJK labels via PIL): live frame + named instance boxes +
gaze cross + verdict banner (object, vote share, world coordinate) + status.

Replay mode (no Pupil Capture needed, e.g. for UI work and regression tests):
  python tools/gaze_live.py --replay ~/recordings/2026_07_09/000
  --replay-speed 0 = as fast as possible; --headless --dump-video out.mp4
  renders the UI to a file instead of a window.

Live mode (Pupil Capture running, Frame Publisher plugin on, gaze calibrated):
  python tools/gaze_live.py [--publish 5581] [--log intents.jsonl]

The published intent events (topic 'gaze.intent', msgpack) are the interface
grasp_intent-style consumers (e.g. a robot arm) subscribe to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pupil_localizer import (connect_pupil, load_fisheye, load_tags,  # noqa: E402
                             recording_frames, scale_K, solve_pose)
from gaze_to_world import SplatDepth  # noqa: E402
from gaze_object import cone_votes  # noqa: E402
from gaze_video import CjkText, draw_instances, load_instances  # noqa: E402

SCENE = Path(__file__).resolve().parents[2] / "SceneRebuild"


def union_boxes_by_name(inst_by_id):
    """One box per object name: the union bbox of all same-named instances."""
    groups = {}
    for v in inst_by_id.values():
        if not v["name"]:
            continue
        lo, hi = v["corners"].min(axis=0), v["corners"].max(axis=0)
        g = groups.get(v["name"])
        if g is None:
            groups[v["name"]] = {"lo": lo, "hi": hi, "color": v["color"]}
        else:
            g["lo"] = np.minimum(g["lo"], lo)
            g["hi"] = np.maximum(g["hi"], hi)
    out = []
    for name, g in groups.items():
        lo, hi = g["lo"], g["hi"]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        out.append({"name": name, "corners": corners, "color": g["color"],
                    "diag": float(np.linalg.norm(hi - lo))})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pupil", default="127.0.0.1:50020", help="Pupil Remote host:port.")
    p.add_argument("--replay", default=None, help="Recording dir: replay world.mp4 + gaze.pldata instead of live ZMQ.")
    p.add_argument("--replay-speed", type=float, default=1.0, help="Replay pacing (1 = real time, 0 = flat out).")
    p.add_argument("--calib", default=str(SCENE / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--tags", default=str(SCENE / "world_size/tags_world.json"))
    p.add_argument("--seg-dir", default=None, help="Default: SceneRebuild/lab_result/segmentation_sam.")
    p.add_argument("--ckpt", default=None, help="Default: newest step-*.ckpt under SceneRebuild/lab_result/.")
    p.add_argument("--dictionary", default="DICT_6X6_250")
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--sample-hz", type=float, default=20.0, help="Gaze processing rate (intersections/s).")
    p.add_argument("--cluster-radius", type=float, default=0.15)
    p.add_argument("--min-fix-dur", type=float, default=0.25)
    p.add_argument("--idle-close", type=float, default=0.4,
                   help="Close the running cluster after this many s without a mappable gaze sample.")
    p.add_argument("--sigma-deg", type=float, default=None,
                   help="Cone sigma until the first online stamp (default: gaze_precision.json of --replay, else 1.5).")
    p.add_argument("--span-sigmas", type=float, default=2.5)
    p.add_argument("--patch", type=int, default=33)
    p.add_argument("--hit-eps", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=0.3, help="Pose EMA (0 = raw).")
    p.add_argument("--max-mean-reproj", type=float, default=0.006)
    p.add_argument("--max-jump", type=float, default=1.0)
    p.add_argument("--on-tag-deg", type=float, default=4.0, help="Gaze-to-tag angle that counts as staring at it.")
    p.add_argument("--stamp-samples", type=int, default=25, help="Samples on one tag before a bias re-estimate.")
    p.add_argument("--stamp-cooldown", type=float, default=5.0)
    p.add_argument("--display-scale", type=float, default=0.67)
    p.add_argument("--headless", action="store_true", help="No cv2 window (replay tests, remote shells).")
    p.add_argument("--dump-video", default=None, help="Write the UI frames to this mp4.")
    p.add_argument("--duration", type=float, default=None, help="Stop after N stream seconds.")
    p.add_argument("--publish", type=int, default=None, help="ZMQ PUB port for 'gaze.intent' events.")
    p.add_argument("--log", default=None, help="Append intent events to this jsonl.")
    return p.parse_args()


# ------------------------------------------------------------ streaming pieces

class RollingPoses:
    """Last few localized poses; query(t) interpolates like PoseTrack but in memory."""

    def __init__(self, max_gap=1.0, keep=90):
        self.buf = deque(maxlen=keep)  # (t, T)
        self.max_gap = max_gap

    def push(self, t, T):
        self.buf.append((t, T))

    def query(self, t):
        if not self.buf:
            return None
        ts = [b[0] for b in self.buf]
        i = np.searchsorted(ts, t)
        if i == 0:
            t1, T1 = self.buf[0]
            return T1 if t1 - t <= self.max_gap else None
        if i == len(self.buf):
            t0, T0 = self.buf[-1]
            return T0 if t - t0 <= self.max_gap else None
        t0, T0 = self.buf[i - 1]
        t1, T1 = self.buf[i]
        if t1 - t0 > self.max_gap:
            return T0 if t - t0 < t1 - t else T1
        a = (t - t0) / max(t1 - t0, 1e-6)
        out = np.eye(4)
        out[:3, 3] = (1 - a) * T0[:3, 3] + a * T1[:3, 3]
        rv, _ = cv2.Rodrigues(T0[:3, :3].T @ T1[:3, :3])
        R_step, _ = cv2.Rodrigues(rv * a)
        out[:3, :3] = T0[:3, :3] @ R_step
        return out


class Localizer:
    """pupil_localizer's per-frame detect -> PnP -> sanity gates, as a class."""

    def __init__(self, args, tags):
        self.args = args
        self.tags = tags
        allc = np.concatenate(list(tags.values()))
        self.bounds = (allc[:, 0].min() - 3, allc[:, 0].max() + 3,
                       allc[:, 1].min() - 3, allc[:, 1].max() + 3)
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.K = self.D = None
        self.smooth = None
        self.last_accept = None
        self.n_reject = 0

    def process(self, ts, img, K, D):
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        visible = [] if ids is None else [int(i) for i in ids.flatten()]
        known = [k for k, i in enumerate(visible) if i in self.tags]
        pose, n_inl, reproj = None, 0, None
        if known:
            obj = np.concatenate([self.tags[visible[k]] for k in known])
            px = np.concatenate([corners[k].reshape(4, 2) for k in known])
            pts_norm = cv2.fisheye.undistortPoints(
                px.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
            pose, n_inl, reproj = solve_pose(obj, pts_norm, 0.01)
        if pose is not None:
            x, y, z = pose[:3, 3]
            if not (self.bounds[0] < x < self.bounds[1] and self.bounds[2] < y < self.bounds[3]
                    and 0.15 < z < 2.8):
                pose = None
            elif reproj is not None and reproj > self.args.max_mean_reproj:
                pose = None
        if pose is not None and self.last_accept is not None:
            dt = ts - self.last_accept[0]
            if dt < 0.25 and np.linalg.norm(pose[:3, 3] - self.last_accept[1]) > self.args.max_jump:
                self.n_reject += 1
                if self.n_reject <= 5:
                    pose = None
        if pose is not None:
            self.n_reject = 0
            self.last_accept = (ts, pose[:3, 3].copy())
            if self.args.ema > 0:
                from pupil_localizer import ema_pose
                self.smooth = ema_pose(self.smooth, pose, self.args.ema)
                pose = self.smooth
        return pose, len(known)


class StreamCluster:
    """Incremental version of cluster_world_fixations: same radius/duration rules."""

    def __init__(self, radius, min_dur, idle_close):
        self.radius = radius
        self.min_dur = min_dur
        self.idle_close = idle_close
        self.reset()

    def reset(self):
        self.ts, self.pts, self.origins = [], [], []
        self.centroid = None

    def _close(self):
        fx = None
        if self.ts and self.ts[-1] - self.ts[0] >= self.min_dur and len(self.ts) >= 4:
            pts = np.array(self.pts)
            c = pts.mean(axis=0)
            mid = len(self.ts) // 2
            o = self.origins[mid]
            d = float(np.linalg.norm(c - o))
            fx = {"t_start": float(self.ts[0]), "t_end": float(self.ts[-1]),
                  "duration_s": float(self.ts[-1] - self.ts[0]),
                  "centroid_world": c.tolist(),
                  "spread_m": float(np.linalg.norm(pts - c, axis=1).mean()),
                  "n_samples": len(self.ts),
                  "origin_world": o.tolist(), "distance_m": round(d, 3),
                  "ang_spread_deg": round(float(np.degrees(np.arctan2(
                      float(np.linalg.norm(pts - c, axis=1).mean()), d))), 2)}
        self.reset()
        return fx

    def push(self, t, p, origin):
        closed = None
        if self.centroid is not None and np.linalg.norm(p - self.centroid) > self.radius:
            closed = self._close()
        self.ts.append(t)
        self.pts.append(p)
        self.origins.append(origin)
        self.centroid = np.mean(self.pts, axis=0)
        return closed

    def poll(self, now):
        if self.ts and now - self.ts[-1] > self.idle_close:
            return self._close()
        return None

    def running(self, now):
        if not self.ts or now - self.ts[-1] > self.idle_close:
            return None
        return self.ts[-1] - self.ts[0], len(self.ts)


class OnlineBias:
    """Rolling precision stamp: gaze passing over a surveyed tag re-estimates bias/sigma.

    Fed with FULL-rate gaze (it is cheap: no rendering) -- a 25-sample stamp
    needs only ~0.1s of staring at ~200Hz. On-tag test uses the bias-corrected
    direction so a large initial bias cannot push a genuine stare past the gate.
    """

    def __init__(self, tag_centers, on_tag_rad, n_needed, cooldown, bias0, sigma0):
        self.ids = np.array(sorted(tag_centers))
        self.C = np.stack([tag_centers[i] for i in self.ids])  # (N,3) world
        self.on_tag = on_tag_rad
        self.n_needed = n_needed
        self.cooldown = cooldown
        self.bias = np.asarray(bias0, float)  # undistorted-normalized units, gaze minus target
        self.sigma_deg = sigma0
        self.buf = deque(maxlen=600)  # (t, tag_id, raw_gaze_norm, target_norm)
        self.last_stamp_t = -1e9
        self.last_stamp_tag = None

    def feed(self, t, raw_norm, T_world_cam):
        w2c = np.linalg.inv(T_world_cam)
        pc = self.C @ w2c[:3, :3].T + w2c[:3, 3]              # (N,3) cam frame
        front = pc[:, 2] > 0.3
        if not front.any():
            return None
        corr = np.asarray(raw_norm, float) - self.bias
        g = np.array([corr[0], corr[1], 1.0])
        g /= np.linalg.norm(g)
        v = pc[front] / np.linalg.norm(pc[front], axis=1, keepdims=True)
        ang = np.arccos(np.clip(v @ g, -1, 1))
        j = int(np.argmin(ang))
        if ang[j] > self.on_tag:
            return None
        tid = int(self.ids[front][j])
        target = pc[front][j, :2] / pc[front][j, 2]
        self.buf.append((t, tid, np.asarray(raw_norm, float), target))
        if t - self.last_stamp_t < self.cooldown:
            return None
        recent = [b for b in self.buf if t - b[0] <= 2.0 and b[1] == tid]
        # need density AND dwell: a saccade sweeping past a tag must not stamp
        if len(recent) < self.n_needed or t - recent[0][0] < 0.3:
            return None
        d = np.array([r[2] - r[3] for r in recent])
        self.bias = np.median(d, axis=0)
        resid = d - self.bias
        self.sigma_deg = float(np.degrees(np.arctan(np.sqrt((resid ** 2).sum(axis=1).mean()))))
        self.last_stamp_t = t
        self.last_stamp_tag = tid
        return tid


# ------------------------------------------------------------ replay source

def replay_events(rec_dir: Path, min_conf: float):
    """Yield ('frame', t, img) / ('gaze', t, (norm_pos, conf)) in timestamp order."""
    gz = []
    with open(rec_dir / "gaze.pldata", "rb") as f:
        for _topic, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            r = msgpack.unpackb(payload, strict_map_key=False)
            if r.get("confidence", 0) >= min_conf:
                gz.append((float(r["timestamp"]), (tuple(r["norm_pos"]), float(r["confidence"]))))
    gz.sort(key=lambda x: x[0])
    gi = 0
    for t, img in recording_frames(rec_dir):
        while gi < len(gz) and gz[gi][0] <= t:
            yield "gaze", gz[gi][0], gz[gi][1]
            gi += 1
        yield "frame", t, img


def live_events(pupil_addr: str, min_conf: float):
    """Yield the same event stream from Pupil Capture (frame.world + gaze)."""
    import zmq
    req, sub = connect_pupil(pupil_addr)  # subscribes frame.world
    sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.")
    from pupil_localizer import recv_world_frame
    while True:
        try:
            parts = sub.recv_multipart()
        except zmq.Again:
            continue
        topic = parts[0].decode("utf-8", "replace")
        if topic.startswith("frame.world"):
            # drain to the newest frame: never localize against a stale image
            while sub.poll(0):
                more = sub.recv_multipart()
                if more[0].decode("utf-8", "replace").startswith("frame.world"):
                    parts = more
                else:
                    r = msgpack.unpackb(more[1], strict_map_key=False)
                    if r.get("confidence", 0) >= min_conf:
                        yield "gaze", float(r["timestamp"]), (tuple(r["norm_pos"]), float(r["confidence"]))
            meta = msgpack.unpackb(parts[1])
            buf = parts[2]
            fmt = meta.get("format", "jpeg")
            if fmt in ("jpeg", "mjpeg"):
                img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            elif fmt == "bgr":
                img = np.frombuffer(buf, np.uint8).reshape(meta["height"], meta["width"], 3)
            else:
                continue
            if img is not None:
                yield "frame", float(meta["timestamp"]), img
        elif topic.startswith("gaze"):
            r = msgpack.unpackb(parts[1], strict_map_key=False)
            if r.get("confidence", 0) >= min_conf:
                yield "gaze", float(r["timestamp"]), (tuple(r["norm_pos"]), float(r["confidence"]))


# ------------------------------------------------------------ verdict pooling (as gaze_object)

def rank_votes(votes, name_of, centroids):
    total = sum(votes.values())
    if not votes or total <= 0:
        return None
    pooled = {}
    for lab, v in votes.items():
        p = pooled.setdefault(name_of(lab), {"v": 0.0, "labels": []})
        p["v"] += v
        p["labels"].append(lab)
    ranked = sorted(pooled.items(), key=lambda kv: -kv[1]["v"])
    best_name, bp = ranked[0]
    best = max(bp["labels"], key=lambda l: votes[l])
    return {"object": best_name, "object_label": best,
            "vote_share": round(bp["v"] / total, 3),
            "object_centroid_world": centroids.get(best),
            "candidates": [{"name": n, "share": round(p["v"] / total, 3),
                            "labels": sorted(p["labels"])} for n, p in ranked[:3]]}


# ------------------------------------------------------------ main

def main() -> int:
    args = parse_args()
    K_calib, D = load_fisheye(args.calib)
    calib_wh = (1920, 1080)
    tags, _doc = load_tags(args.tags)
    tag_centers = {tid: c.mean(axis=0) for tid, c in tags.items()}

    seg = Path(args.seg_dir) if args.seg_dir else SCENE / "lab_result/segmentation_sam"
    z = np.load(seg / "points.npz")
    xyz, label = z["xyz"], z["label"]
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    meta = json.loads((seg / "instances.json").read_text(encoding="utf-8"))
    names = json.loads((seg / "names.json").read_text(encoding="utf-8")) if (seg / "names.json").exists() else {}
    bg = {int(k): v for k, v in meta["background"].items()}
    centroids = {i["id"]: i["centroid"] for i in meta["instances"]}

    def name_of(lab: int) -> str:
        if lab in bg:
            return bg[lab]
        return names.get(str(lab), "") or f"object#{lab}"

    inst_by_id = load_instances(seg)
    named_boxes = union_boxes_by_name(inst_by_id)  # one union box per object name
    print(f"segmentation: {seg.name}, {len(inst_by_id)} instances "
          f"({len(named_boxes)} named objects)")

    ckpt = Path(args.ckpt) if args.ckpt else max(
        (SCENE / "lab_result").rglob("step-*.ckpt"), key=lambda p: p.stat().st_mtime)
    splat = SplatDepth(ckpt)

    sigma0 = args.sigma_deg
    if sigma0 is None and args.replay:
        pj = Path(args.replay) / "gaze_precision.json"
        if pj.exists():
            sigma0 = float(json.loads(pj.read_text(encoding="utf-8"))["sigma_deg"])
    sigma0 = sigma0 or 1.5
    bias_est = OnlineBias(tag_centers, np.radians(args.on_tag_deg),
                          args.stamp_samples, args.stamp_cooldown,
                          bias0=(0.0, 0.0), sigma0=sigma0)
    print(f"cone sigma start: {sigma0:.2f} deg (online re-estimation on tag stares)")

    loc = Localizer(args, tags)
    poses = RollingPoses()
    cluster = StreamCluster(args.cluster_radius, args.min_fix_dur, args.idle_close)
    cjk = CjkText()

    pub = None
    if args.publish:
        import zmq
        pub = zmq.Context.instance().socket(zmq.PUB)
        pub.bind(f"tcp://*:{args.publish}")
    log_f = open(args.log, "a", encoding="utf-8") if args.log else None

    if args.replay:
        source = replay_events(Path(args.replay), args.min_confidence)
        print(f"replaying {args.replay} (speed {args.replay_speed or 'max'})")
    else:
        source = live_events(args.pupil, args.min_confidence)
        print("live: connected to Pupil Capture (Frame Publisher + calibrated gaze required)")

    writer = None
    win = not args.headless
    if win:
        cv2.namedWindow("gaze_live", cv2.WINDOW_NORMAL)

    K = None
    W = H = None
    last_gaze_proc = 0.0
    last_gaze_px = None       # (u, v, conf, t)
    verdict = None            # last closed fixation verdict (+ _shown_at)
    n_frames = n_loc = 0
    isect_ms = deque(maxlen=60)
    t_stream0 = wall0 = None
    t_now = 0.0

    def close_and_judge(fx):
        nonlocal verdict
        if fx is None:
            return
        t0 = time.perf_counter()
        votes, p_none = cone_votes(splat, tree, label,
                                   np.asarray(fx["origin_world"], float),
                                   np.asarray(fx["centroid_world"], float),
                                   np.radians(bias_est.sigma_deg), args.span_sigmas,
                                   args.patch, args.hit_eps)
        rank = rank_votes(votes, name_of, centroids)
        if rank is None:
            return
        fx.update(rank, p_none=round(p_none, 3), sigma_deg=round(bias_est.sigma_deg, 2),
                  mode="cone", judge_ms=round((time.perf_counter() - t0) * 1e3, 1))
        verdict = dict(fx, _shown_at=t_now)
        c = fx["centroid_world"]
        print(f"[{fx['t_start'] - (t_stream0 or 0):7.1f}s] {fx['object']:<18} "
              f"{fx['vote_share']:>4.0%}  ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})m "
              f"dur {fx['duration_s']:.2f}s  none {p_none:.0%}  [{fx['judge_ms']:.0f}ms]")
        payload = {k: v for k, v in fx.items() if not k.startswith("_")}
        payload["topic"] = "gaze.intent"
        if pub:
            pub.send_multipart([b"gaze.intent", msgpack.packb(payload)])
        if log_f:
            log_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            log_f.flush()

    try:
        for kind, t, payload in source:
            if t_stream0 is None:
                t_stream0, wall0 = t, time.time()
            t_now = t
            if args.duration and t - t_stream0 > args.duration:
                break
            if args.replay and args.replay_speed > 0:
                lag = (t - t_stream0) / args.replay_speed - (time.time() - wall0)
                if lag > 0.002:
                    time.sleep(lag)

            if kind == "gaze":
                (nx, ny), conf = payload
                if W is None:
                    continue
                u, v = nx * W, (1.0 - ny) * H
                last_gaze_px = (u, v, conf, t)
                T = poses.query(t)
                if T is None:
                    continue
                pn = cv2.fisheye.undistortPoints(
                    np.array([[[u, v]]], np.float64), K, D).reshape(2)
                # full-rate: the rolling precision stamp needs sample density
                stamped = bias_est.feed(t, pn, T)
                if stamped is not None:
                    b = np.degrees(np.arctan(bias_est.bias))
                    print(f"[{t - (t_stream0 or 0):7.1f}s] bias stamp @tag{stamped}: "
                          f"({b[0]:+.2f},{b[1]:+.2f})deg  sigma {bias_est.sigma_deg:.2f}deg")
                if t - last_gaze_proc < 1.0 / args.sample_hz:
                    continue                                 # intersection budget gate
                last_gaze_proc = t
                pn = pn - bias_est.bias                      # corrected ray for mapping
                ray = np.array([pn[0], pn[1], 1.0])
                ray /= np.linalg.norm(ray)
                t0 = time.perf_counter()
                depth = splat.depth_along_ray(T[:3, 3], T[:3, :3] @ ray)
                isect_ms.append((time.perf_counter() - t0) * 1e3)
                if depth is None:
                    continue
                p_world = T[:3, 3] + depth * (T[:3, :3] @ ray)
                close_and_judge(cluster.push(t, p_world, T[:3, 3]))
                continue

            # ---- frame ----
            img = payload
            n_frames += 1
            if K is None:
                W, H = img.shape[1], img.shape[0]
                K = scale_K(K_calib, calib_wh, (W, H))
                if (W, H) != calib_wh:
                    print(f"NOTE: frame {W}x{H} != calibration {calib_wh}, K rescaled")
            pose, n_tags = loc.process(t, img, K, D)
            if pose is not None:
                n_loc += 1
                poses.push(t, pose)
            close_and_judge(cluster.poll(t))

            if not win and writer is None and not args.dump_video:
                continue  # headless without dump: no UI work at all

            T_ui = poses.query(t)
            if T_ui is not None and named_boxes:
                draw_instances(img, T_ui, named_boxes, K, D, thick=1, cjk=cjk)
                if verdict is not None and t - verdict["_shown_at"] < 4.0:
                    hit = [b for b in named_boxes if b["name"] == verdict["object"]]
                    if hit:
                        draw_instances(img, T_ui, hit, K, D, thick=4, cjk=cjk)
            if last_gaze_px is not None and t - last_gaze_px[3] < 0.15:
                u, v, conf, _ = last_gaze_px
                color = (0, 220, 0) if conf >= 0.8 else (0, 165, 255)
                cv2.circle(img, (int(u), int(v)), 28, color, 4)
                cv2.line(img, (int(u) - 40, int(v)), (int(u) + 40, int(v)), color, 2)
                cv2.line(img, (int(u), int(v) - 40), (int(u), int(v) + 40), color, 2)
            run = cluster.running(t)
            if run is not None:
                cjk.put(img, f"fixating... {run[0]:.2f}s ({run[1]})", (30, H - 120), 30, (0, 200, 255))
            if verdict is not None and t - verdict["_shown_at"] < 4.0:
                c = verdict["centroid_world"]
                cjk.put(img, f"-> {verdict['object']}  {verdict['vote_share']:.0%}"
                             f"  [{c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}]m", (30, H - 60), 40, (0, 0, 255), 3)
            im = np.mean(isect_ms) if isect_ms else 0.0
            status = (f"t={t - t_stream0:6.1f}s  loc {n_loc}/{n_frames}"
                      f"  tags {n_tags}  sigma {bias_est.sigma_deg:.1f}deg"
                      f"  isect {im:.0f}ms")
            if bias_est.last_stamp_tag is not None:
                status += f"  bias@tag{bias_est.last_stamp_tag} ({np.degrees(np.arctan(bias_est.bias[0])):+.1f},{np.degrees(np.arctan(bias_est.bias[1])):+.1f})deg"
            pose_ok = T_ui is not None
            cv2.putText(img, status, (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255) if pose_ok else (0, 0, 255), 2)
            if not pose_ok:
                cv2.putText(img, "NO POSE (no tag in view)", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if args.dump_video:
                if writer is None:
                    writer = cv2.VideoWriter(args.dump_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             30.0, (W, H))
                writer.write(img)
            if win:
                disp = img if args.display_scale == 1.0 else cv2.resize(
                    img, None, fx=args.display_scale, fy=args.display_scale)
                cv2.imshow("gaze_live", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        close_and_judge(cluster.poll(t_now + 1e9))  # flush the running cluster
        if writer is not None:
            writer.release()
            print(f"wrote {args.dump_video}")
        if log_f:
            log_f.close()
        if win:
            cv2.destroyAllWindows()
    print(f"done: {n_frames} frames, {n_loc} localized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
