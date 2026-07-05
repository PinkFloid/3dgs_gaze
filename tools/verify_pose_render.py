#!/usr/bin/env python
"""Pose/calibration/model cross-check on a single recording frame.

Picks a frame with clearly visible surveyed tags (or a user-chosen one), then:
  1. PnP against the surveyed tag corners -> T_world_cam   (localization)
  2. Fisheye-undistort the real frame to a virtual pinhole K_new  (calibration)
  3. Render the 3DGS from T_world_cam with the same K_new         (map + pose)
  4. Overlay + metrics:
       - green crosses: surveyed 3D tag corners projected through the pose
       - red circles:  detected corners mapped into the undistorted image
       red-vs-green offset = end-to-end error, printed in px / degrees / mm@tag

Outputs <out-dir>/verify_f<N>_side.jpg (undistorted | render) and _blend.jpg
(50/50 mix). If everything is right the blend should look like ONE sharp image.

Needs the gsplat build env vars on first import (see PIPELINE.md).

Example:
  python tools/verify_pose_render.py --recording ~/recordings/2026_07_05/000 \
    --tags world_size/tags_world.json --out-dir lab_result/pose_verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pupil_localizer import load_fisheye, load_tags, scale_K, solve_pose  # noqa: E402
from gaze_to_world import SplatDepth  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True, help="Pupil Capture recording dir.")
    p.add_argument("--tags", default=str(root / "world_size/tags_world.json"))
    p.add_argument("--calib", default=str(root / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--ckpt", default=None, help="Default: newest step-*.ckpt under lab_result/.")
    p.add_argument("--frame", type=int, default=None,
                   help="Frame index to use. Default: scan and pick the frame seeing the most tags.")
    p.add_argument("--scan-step", type=int, default=5, help="Frame stride for the auto scan.")
    p.add_argument("--pinhole-scale", type=float, default=0.7,
                   help="Virtual pinhole focal = fisheye focal x this. Smaller keeps more FOV "
                        "(cv2.fisheye.estimateNewCameraMatrixForUndistortRectify returns garbage, built manually).")
    p.add_argument("--out-dir", default=str(root / "lab_result/pose_verify"))
    return p.parse_args()


def detect(gray, detector, tags):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return [], []
    keep_c, keep_i = [], []
    for quad, mid in zip(corners, ids.flatten()):
        if int(mid) in tags:
            keep_c.append(quad.reshape(4, 2))
            keep_i.append(int(mid))
    return keep_c, keep_i


def main() -> int:
    args = parse_args()
    rec = Path(args.recording).expanduser()
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K_calib, D = load_fisheye(args.calib)
    tags, _ = load_tags(args.tags)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    K = scale_K(K_calib, (1920, 1080), (W, H))

    # ---- pick the frame
    if args.frame is not None:
        fi = args.frame
    else:
        best, fi = -1, 0
        for i in range(0, n_total, args.scan_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, ids = detect(gray, detector, tags)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            score = len(ids) + min(sharp / 500.0, 2.0)  # tags dominate, sharpness breaks ties
            if len(ids) > 0 and score > best:
                best, fi = score, i
        print(f"auto-picked frame {fi} (score {best:.1f})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {fi}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    quads, ids = detect(gray, detector, tags)
    if not ids:
        raise SystemExit(f"frame {fi}: no surveyed tags detected")
    print(f"frame {fi}: tags {sorted(ids)}")

    # ---- pose from this frame alone
    obj = np.concatenate([tags[i] for i in ids])
    px = np.concatenate(quads)
    pts_norm = cv2.fisheye.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    T, n_inl, reproj = solve_pose(obj, pts_norm, 0.01)
    if T is None:
        raise SystemExit("PnP failed on this frame")
    pos = T[:3, 3]
    print(f"T_world_cam: pos ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m, "
          f"{n_inl}/{len(obj)} inliers, mean reproj {reproj*1000:.2f}e-3 norm")

    # ---- undistort the real frame to a virtual pinhole
    K_new = np.array([[K[0, 0] * args.pinhole_scale, 0, W / 2.0],
                      [0, K[1, 1] * args.pinhole_scale, H / 2.0],
                      [0, 0, 1.0]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2)
    real_pin = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    # ---- render the splat with the same pinhole
    ckpt = Path(args.ckpt) if args.ckpt else max(
        (root / "lab_result").rglob("step-*.ckpt"), key=lambda p: p.stat().st_mtime)
    splat = SplatDepth(ckpt)
    render = cv2.cvtColor(splat.render_view(T, K_new, W, H), cv2.COLOR_RGB2BGR)

    # ---- corner metrics on the pinhole image
    w2c = np.linalg.inv(T)
    rvec, _ = cv2.Rodrigues(w2c[:3, :3])
    proj_surv, _ = cv2.projectPoints(obj, rvec, w2c[:3, 3], K_new, None)   # green: map->image
    proj_surv = proj_surv.reshape(-1, 2)
    det_pin = (pts_norm @ np.diag([K_new[0, 0], K_new[1, 1]])) + np.array([K_new[0, 2], K_new[1, 2]])
    err = np.linalg.norm(proj_surv - det_pin, axis=1)
    dists = np.linalg.norm(obj - pos, axis=1)
    ang = np.degrees(err / K_new[0, 0])
    mm = err / K_new[0, 0] * dists * 1000
    print(f"corner residual: mean {err.mean():.2f}px max {err.max():.2f}px "
          f"| angular mean {ang.mean():.3f}deg | at-tag mean {mm.mean():.1f}mm")

    for canvas in (real_pin, render):
        for p_ in proj_surv:
            cv2.drawMarker(canvas, tuple(np.int32(p_)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    for p_ in det_pin:
        cv2.circle(real_pin, tuple(np.int32(p_)), 12, (0, 0, 255), 2)

    label = (f"f{fi} pos({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})m "
             f"err {err.mean():.1f}px/{ang.mean():.2f}deg")
    blend = cv2.addWeighted(real_pin, 0.5, render, 0.5, 0)  # blend before labels overlap
    cv2.putText(blend, f"50/50 blend  {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    for canvas, name in ((real_pin, "undistorted real"), (render, "3DGS render")):
        cv2.putText(canvas, f"{name}  {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    side = np.concatenate([real_pin, render], axis=1)
    cv2.imwrite(str(out_dir / f"verify_f{fi}_side.jpg"), side)
    cv2.imwrite(str(out_dir / f"verify_f{fi}_blend.jpg"), blend)
    print(f"wrote {out_dir}/verify_f{fi}_side.jpg and _blend.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
