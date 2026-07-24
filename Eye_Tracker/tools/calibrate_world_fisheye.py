#!/usr/bin/env python3
"""calibrate_world_fisheye.py -- Pupil 世界相机(鱼眼)ChArUco 内参标定。

吃 capture_world_frames.py 存的 frame_*.png,拟合三种模型(fisheye / standard /
rational),主模型取 fisheye,npz 键位与旧 world_camera_calibration.npz 完全一致
(pupil_localizer 只读 camera_matrix + dist_coeffs[:4] + model)。带逐帧误差剔除。

    python calibrate_world_fisheye.py                          # 新 A3 板(id 0 起,legacy)
    python calibrate_world_fisheye.py --marker-id-start 30     # 旧板(验证旧数据用)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    root = Path(__file__).resolve()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", default=str(root.parents[1] / "world_camera_calibration_imgs"))
    p.add_argument("--out", default=str(root.parents[2] / "SceneRebuild"
                                        / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--square-length", type=float, default=0.032)
    p.add_argument("--marker-length", type=float, default=0.024)
    p.add_argument("--dictionary", default="DICT_6X6_250")
    p.add_argument("--marker-id-start", type=int, default=0, help="新 A3 板=0,旧板=30")
    p.add_argument("--no-legacy", action="store_true")
    p.add_argument("--min-corners", type=int, default=20)
    p.add_argument("--max-image-err", type=float, default=1.0,
                   help="剔除均值重投影超过该值(px)的帧后重标;0 关闭")
    p.add_argument("--prune-rounds", type=int, default=2)
    return p.parse_args()


def fisheye_calibrate(objs, imgs, size):
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objs, imgs, size, K, D,
                                                    flags=flags, criteria=crit)
    return rms, K, D, rvecs, tvecs


def fisheye_image_errs(objs, imgs, K, D, rvecs, tvecs):
    errs = []
    for o, i, rv, tv in zip(objs, imgs, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(o, rv, tv, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))
    return errs


def main() -> int:
    args = parse_args()
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    n_markers = (args.squares_x * args.squares_y) // 2
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y),
                                   args.square_length, args.marker_length, dic,
                                   np.arange(args.marker_id_start,
                                             args.marker_id_start + n_markers))
    board.setLegacyPattern(not args.no_legacy)
    det = cv2.aruco.CharucoDetector(board)

    paths = sorted(Path(args.images).glob("frame_*.png")) or \
        sorted(p for p in Path(args.images).iterdir()
               if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise SystemExit(f"{args.images} 里没有图")
    objs, imgs, names, size = [], [], [], None
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if size is None:
            size = (im.shape[1], im.shape[0])
        cc, ci, _mc, _mi = det.detectBoard(im)
        n = 0 if ci is None else len(ci)
        if n < args.min_corners:
            print(f"[skip] {p.name}: {n} 角点")
            continue
        o, i = board.matchImagePoints(cc, ci)
        objs.append(o.reshape(1, -1, 3).astype(np.float64))
        imgs.append(i.reshape(1, -1, 2).astype(np.float64))
        names.append(p.name)
    print(f"{len(names)}/{len(paths)} 帧可用,分辨率 {size}")
    if len(names) < 10:
        raise SystemExit("有效帧太少(<10)")

    rejected = []
    for round_i in range(max(1, args.prune_rounds + 1)):
        rms, K, D, rvecs, tvecs = fisheye_calibrate(objs, imgs, size)
        errs = fisheye_image_errs(objs, imgs, K, D, rvecs, tvecs)
        order = np.argsort(errs)[::-1]
        print(f"round {round_i}: fisheye rms {rms:.3f}px / {len(objs)} 帧; worst: "
              + ", ".join(f"{names[k]} {errs[k]:.2f}" for k in order[:4]))
        if args.max_image_err <= 0 or round_i >= args.prune_rounds:
            break
        keep = [k for k in range(len(errs)) if errs[k] <= args.max_image_err]
        if len(keep) == len(errs) or len(keep) < 10:
            break
        rejected += [(names[k], round(errs[k], 3)) for k in range(len(errs))
                     if errs[k] > args.max_image_err]
        objs = [objs[k] for k in keep]
        imgs = [imgs[k] for k in keep]
        names = [names[k] for k in keep]
    if rejected:
        print(f"pruned {len(rejected)}: " + ", ".join(f"{n}({e})" for n, e in rejected))

    # 对照模型:同一批入选帧上拟合针孔 standard / rational(仅记录,主模型仍 fisheye)
    objs32 = [o.reshape(-1, 1, 3).astype(np.float32) for o in objs]
    imgs32 = [i.reshape(-1, 1, 2).astype(np.float32) for i in imgs]
    rms_s, K_s, D_s, _, _ = cv2.calibrateCamera(objs32, imgs32, size, None, None)
    rms_r, K_r, D_r, _, _ = cv2.calibrateCamera(objs32, imgs32, size, None, None,
                                                flags=cv2.CALIB_RATIONAL_MODEL)
    print(f"模型对比:fisheye {rms:.3f} | standard {rms_s:.3f} | rational {rms_r:.3f} px")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             model="fisheye",
             camera_matrix=K, dist_coeffs=D, reprojection_error=rms,
             camera_matrix_fisheye=K, dist_coeffs_fisheye=D, rms_fisheye=rms,
             camera_matrix_standard=K_s, dist_coeffs_standard=D_s, rms_standard=rms_s,
             camera_matrix_rational=K_r, dist_coeffs_rational=D_r, rms_rational=rms_r,
             image_size=np.array(size), used_images=np.array(names),
             board=json.dumps({"squares_x": args.squares_x, "squares_y": args.squares_y,
                               "square_length": args.square_length,
                               "marker_length": args.marker_length,
                               "dictionary": args.dictionary,
                               "marker_id_start": args.marker_id_start,
                               "legacy": not args.no_legacy}))
    print(f"fisheye K: fx {K[0,0]:.2f} fy {K[1,1]:.2f} cx {K[0,2]:.2f} cy {K[1,2]:.2f}")
    print(f"D: {D.ravel().round(5).tolist()}")
    print(f"已存 {out}({len(names)} 帧,rms {rms:.3f}px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
