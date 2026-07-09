#!/usr/bin/env python
"""Online Pupil Core localization: world camera frames -> ArUco tags -> T_world_cam.

Consumes:
  - Pupil Capture's Frame Publisher stream (frame.world over ZMQ).
  - world_camera_calibration.npz  (cv2.fisheye model: camera_matrix, dist_coeffs)
  - tags_world.json from survey_aruco_tags.py (surveyed tag corners in the
    ChArUco board world frame, meters)

Per frame:
  1. Detect ArUco markers on the raw (distorted) image.
  2. cv2.fisheye.undistortPoints on the corner pixels only (no full-frame remap).
  3. solvePnP(RANSAC) against the surveyed 3D corners of ALL visible tags at
     once -> T_world_cam. Using corners_world directly means surveyed tag
     geometry is honored exactly (no nominal-size assumption).

Outputs any combination of:
  --print          human-readable pose lines
  --log FILE.jsonl one json per localized frame (pupil timestamp + T_world_cam)
  --publish PORT   ZMQ PUB socket, topic 'pose.world_cam', msgpack payload
  --preview-dir D  annotated jpg every N frames (headless-friendly)

--detect-only runs without tags_world.json and just reports visible tag ids
(useful to sanity-check detection before the survey file exists).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import zmq


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pupil", default="127.0.0.1:50020", help="Pupil Remote host:port.")
    p.add_argument("--recording", default=None,
                   help="Offline mode: a Pupil Capture recording dir (world.mp4 + world_timestamps.npy) "
                        "replayed instead of the live stream.")
    p.add_argument("--calib", default=str(Path(__file__).resolve().parents[2] / "SceneRebuild" / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--tags", default=str(Path(__file__).resolve().parents[2] / "SceneRebuild" / "world_size/tags_world.json"),
                   help="tags_world.json from survey_aruco_tags.py.")
    p.add_argument("--dictionary", default="DICT_6X6_250", help="OpenCV ArUco dictionary of the wall tags.")
    p.add_argument("--detect-only", action="store_true", help="No pose solving, just report detected tag ids.")
    p.add_argument("--min-tags", type=int, default=1, help="Min visible surveyed tags to attempt a pose.")
    p.add_argument("--max-reproj-norm", type=float, default=0.01,
                   help="RANSAC reprojection threshold in normalized image units (~0.01 = 8px at f=800).")
    p.add_argument("--ema", type=float, default=0.0,
                   help="Pose smoothing factor 0..1 (0 = raw). 0.7 is a reasonable start.")
    p.add_argument("--max-mean-reproj", type=float, default=0.006,
                   help="Reject poses with mean reprojection error above this (normalized units).")
    p.add_argument("--max-jump", type=float, default=1.0,
                   help="Reject poses further than this (m) from the last accepted pose within 0.25s. "
                        "Auto-resets after 5 consecutive rejections.")
    p.add_argument("--print", dest="do_print", action="store_true")
    p.add_argument("--log", default=None)
    p.add_argument("--publish", type=int, default=None, help="ZMQ PUB port for pose messages.")
    p.add_argument("--preview-dir", default=None)
    p.add_argument("--preview-every", type=int, default=30)
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: run forever).")
    return p.parse_args()


# ---------------------------------------------------------------- pupil I/O

def connect_pupil(addr: str):
    host, port = addr.rsplit(":", 1)
    ctx = zmq.Context.instance()
    req = ctx.socket(zmq.REQ)
    req.setsockopt(zmq.RCVTIMEO, 5000)
    req.setsockopt(zmq.SNDTIMEO, 5000)
    req.connect(f"tcp://{host}:{port}")
    try:
        req.send_string("SUB_PORT")
        sub_port = req.recv_string()
    except zmq.Again:
        raise SystemExit(f"No reply from Pupil Remote at {addr} -- is Pupil Capture running?")
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{sub_port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "frame.world")
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    return req, sub


def live_frames(sub):
    """Yield (timestamp, bgr) from the live frame.world stream; skips timeouts."""
    while True:
        got = recv_world_frame(sub)
        if got is not None:
            yield got


def recording_frames(rec_dir: Path):
    """Yield (timestamp, bgr) from a Pupil Capture recording directory."""
    video = next((rec_dir / n for n in ("world.mp4", "world.mjpeg", "world.avi") if (rec_dir / n).exists()), None)
    if video is None:
        raise SystemExit(f"No world video found in {rec_dir}")
    ts_file = rec_dir / "world_timestamps.npy"
    ts = np.load(ts_file) if ts_file.exists() else None
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = float(ts[i]) if ts is not None and i < len(ts) else i / 30.0
        i += 1
        yield t, img
    cap.release()


def recv_world_frame(sub):
    """Return (timestamp, bgr image) or None on timeout / undecodable frame."""
    try:
        parts = sub.recv_multipart()
    except zmq.Again:
        return None
    if len(parts) < 3:
        return None
    meta = msgpack.unpackb(parts[1])
    buf = parts[2]
    fmt = meta.get("format", "jpeg")
    if fmt in ("jpeg", "mjpeg"):
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    elif fmt == "bgr":
        img = np.frombuffer(buf, np.uint8).reshape(meta["height"], meta["width"], 3)
    elif fmt == "gray":
        img = cv2.cvtColor(np.frombuffer(buf, np.uint8).reshape(meta["height"], meta["width"]), cv2.COLOR_GRAY2BGR)
    else:
        return None
    if img is None:
        return None
    return float(meta["timestamp"]), img


# ---------------------------------------------------------------- geometry

def load_fisheye(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    K = np.asarray(z["camera_matrix"], np.float64)
    D = np.asarray(z["dist_coeffs"], np.float64).reshape(-1, 1)[:4]
    return K, D


def scale_K(K: np.ndarray, calib_wh, frame_wh):
    if calib_wh == frame_wh:
        return K
    sx = frame_wh[0] / calib_wh[0]
    sy = frame_wh[1] / calib_wh[1]
    K = K.copy()
    K[0] *= sx
    K[1] *= sy
    return K


def load_tags(path: str):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    tags = {int(k): np.asarray(v["corners_world"], np.float64) for k, v in doc["tags"].items()}
    return tags, doc


def solve_pose(obj_world: np.ndarray, pts_norm: np.ndarray, thresh: float):
    """PnP on fisheye-undistorted (normalized) points: K = I, dist = 0.

    All wall/floor tags are coplanar, so use ITERATIVE (homography-based init
    for planar sets); SQPNP asserts on degenerate/planar point clouds.
    """
    obj = np.asarray(obj_world, np.float64)
    pts = np.asarray(pts_norm, np.float64).reshape(-1, 1, 2)
    try:
        if len(obj) <= 4:  # single tag: nothing for RANSAC to reject
            ok, rvec, tvec = cv2.solvePnP(obj, pts, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
            inliers = np.arange(len(obj)).reshape(-1, 1) if ok else None
        else:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj, pts, np.eye(3), None,
                reprojectionError=thresh, iterationsCount=200, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inliers is None or len(inliers) < 4:
            return None, 0
        rvec, tvec = cv2.solvePnPRefineLM(obj[inliers.flatten()], pts[inliers.flatten()],
                                          np.eye(3), None, rvec, tvec)
    except cv2.error:
        return None, 0, None
    R, _ = cv2.Rodrigues(rvec)
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R
    T_cam_world[:3, 3] = tvec.flatten()
    proj, _ = cv2.projectPoints(obj[inliers.flatten()], rvec, tvec, np.eye(3), None)
    reproj = float(np.linalg.norm(proj.reshape(-1, 2) - pts[inliers.flatten()].reshape(-1, 2), axis=1).mean())
    return np.linalg.inv(T_cam_world), int(len(inliers)), reproj


def ema_pose(prev: np.ndarray | None, cur: np.ndarray, alpha: float) -> np.ndarray:
    """alpha = weight of the PREVIOUS pose (0 = no smoothing)."""
    if prev is None or alpha <= 0:
        return cur
    out = np.eye(4)
    out[:3, 3] = alpha * prev[:3, 3] + (1 - alpha) * cur[:3, 3]
    # slerp via rotation vector of the relative rotation
    R_rel = prev[:3, :3].T @ cur[:3, :3]
    rv, _ = cv2.Rodrigues(R_rel)
    R_step, _ = cv2.Rodrigues(rv * (1 - alpha))
    out[:3, :3] = prev[:3, :3] @ R_step
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    args = parse_args()
    K_calib, D = load_fisheye(args.calib)
    calib_wh = (1920, 1080)

    tags = {}
    bounds = None
    if not args.detect_only:
        if not args.tags:
            raise SystemExit("--tags tags_world.json is required (or use --detect-only).")
        tags, doc = load_tags(args.tags)
        allc = np.concatenate(list(tags.values()))
        bounds = (allc[:, 0].min() - 3, allc[:, 0].max() + 3,
                  allc[:, 1].min() - 3, allc[:, 1].max() + 3)  # x/y: tag extent + 3m margin
        print(f"Loaded {len(tags)} surveyed tags from {args.tags}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    pub = None
    if args.publish:
        pub = zmq.Context.instance().socket(zmq.PUB)
        pub.bind(f"tcp://*:{args.publish}")
    log_f = open(args.log, "w", encoding="utf-8") if args.log else None
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    if args.recording:
        frame_iter = recording_frames(Path(args.recording))
        print(f"Replaying recording {args.recording}")
    else:
        req, sub = connect_pupil(args.pupil)
        frame_iter = live_frames(sub)
        print("Connected to Pupil Capture, waiting for world frames "
              "(needs the Frame Publisher plugin enabled)...")

    K = None
    pose_smooth = None
    last_accept = None
    n_reject = 0
    n_frames = n_loc = 0
    t_start = time.time()
    t_report = t_start
    try:
        for ts, img in frame_iter:
            if args.duration and time.time() - t_start > args.duration:
                break
            n_frames += 1
            if K is None:
                K = scale_K(K_calib, calib_wh, (img.shape[1], img.shape[0]))
                if (img.shape[1], img.shape[0]) != calib_wh:
                    print(f"NOTE: frame {img.shape[1]}x{img.shape[0]} != calibration {calib_wh}, K rescaled")

            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            visible = [] if ids is None else [int(i) for i in ids.flatten()]

            if args.detect_only:
                if visible and args.do_print:
                    print(f"t={ts:.3f} tags: {sorted(visible)}")
            else:
                known = [k for k, i in enumerate(visible) if i in tags]
                pose, n_inl, reproj = (None, 0, None)
                if len(known) >= args.min_tags:
                    obj = np.concatenate([tags[visible[k]] for k in known])           # (4m,3)
                    px = np.concatenate([corners[k].reshape(4, 2) for k in known])    # (4m,2)
                    pts_norm = cv2.fisheye.undistortPoints(
                        px.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
                    pose, n_inl, reproj = solve_pose(obj, pts_norm, args.max_reproj_norm)
                # --- sanity gates: bounds, residual, temporal continuity ---
                if pose is not None:
                    x, y, z = pose[:3, 3]
                    if not (bounds[0] < x < bounds[1] and bounds[2] < y < bounds[3] and 0.15 < z < 2.8):
                        pose = None
                    elif reproj is not None and reproj > args.max_mean_reproj:
                        pose = None
                if pose is not None and last_accept is not None:
                    dt_a = ts - last_accept[0]
                    if dt_a < 0.25 and np.linalg.norm(pose[:3, 3] - last_accept[1]) > args.max_jump:
                        n_reject += 1
                        if n_reject <= 5:
                            pose = None
                        # after 5 consecutive rejections trust the new pose (real fast motion)
                if pose is not None:
                    n_reject = 0
                    last_accept = (ts, pose[:3, 3].copy())
                    n_loc += 1
                    pose_smooth = ema_pose(pose_smooth, pose, args.ema)
                    p = pose_smooth
                    payload = {"topic": "pose.world_cam", "timestamp": ts,
                               "T_world_cam": p.tolist(),
                               "n_tags": len(known), "n_inliers": n_inl,
                               "mean_reproj_norm": reproj}
                    if args.do_print:
                        pos = p[:3, 3]
                        print(f"t={ts:.3f} pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})m "
                              f"tags={sorted(visible[k] for k in known)} inl={n_inl}")
                    if log_f:
                        log_f.write(json.dumps(payload) + "\n")
                    if pub:
                        pub.send_multipart([b"pose.world_cam", msgpack.packb(payload)])

            if preview_dir and n_frames % args.preview_every == 0:
                vis = img.copy()
                if visible:
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imwrite(str(preview_dir / f"frame_{n_frames:06d}.jpg"), vis)

            if time.time() - t_report > 5:
                fps = n_frames / (time.time() - t_start)
                print(f"[{time.time()-t_start:5.1f}s] frames={n_frames} ({fps:.1f}/s) localized={n_loc}")
                t_report = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        if log_f:
            log_f.close()
    print(f"done: {n_frames} frames, {n_loc} localized, {time.time()-t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
