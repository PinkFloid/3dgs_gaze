#!/usr/bin/env python
"""Align a nerfstudio (ns-process-data) dataset to a ChArUco board world frame.

Pipeline:
  1. Detect ChArUco chessboard corners (2D, with corner ids) in every dataset image.
  2. Triangulate each corner in the COLMAP/nerfstudio world frame using the camera
     poses + intrinsics from transforms.json (multi-view DLT on undistorted points).
  3. Fit a similarity transform (Umeyama: scale + rotation + translation) from the
     triangulated corners to their known metric positions on the physical board.
  4. Bake the transform into a new transforms_aligned.json and sparse_pc_aligned.ply.

After this, the world frame is the board frame: origin on the board, X/Y in the
board plane, Z perpendicular to it (flipped if needed so cameras sit at Z > 0),
and units are meters when --square-size is given in meters.

Train with the aligned frame kept intact:
  ns-train splatfacto --data <dataset>/transforms_aligned.json \
      nerfstudio-data --orientation-method none --center-method none --auto-scale-poses False
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="ns-process-data output dir containing transforms.json.")
    parser.add_argument("--square-size", type=float, default=1.0,
                        help="Physical chessboard square side length in meters. Default 1.0 = 'square units'.")
    parser.add_argument("--marker-size", type=float, default=None,
                        help="ArUco marker side length, same unit as --square-size. "
                             "Default: 22/30 of the square size (this project's board: 30mm square, 22mm marker). "
                             "Does not affect corner positions, only marker detection geometry.")
    parser.add_argument("--squares-x", type=int, default=11, help="Chessboard squares along X.")
    parser.add_argument("--squares-y", type=int, default=8, help="Chessboard squares along Y.")
    parser.add_argument("--dictionary", default="DICT_6X6_250", help="OpenCV ArUco dictionary name.")
    parser.add_argument("--marker-id-start", type=int, default=30, help="First ArUco marker id on the board.")
    parser.add_argument("--no-legacy", action="store_true", help="Board was generated with the new (non-legacy) pattern.")
    parser.add_argument("--origin", choices=["corner", "center"], default="corner",
                        help="Put the world origin at the board corner (OpenCV convention) or the board center.")
    parser.add_argument("--min-views", type=int, default=3, help="Min views per corner to triangulate.")
    parser.add_argument("--max-reproj-px", type=float, default=4.0, help="Reprojection outlier threshold (pixels).")
    parser.add_argument("--out-suffix", default="aligned", help="Suffix for output json/ply files.")
    return parser.parse_args()


def build_board(args):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, args.dictionary))
    n_markers = args.squares_x * args.squares_y // 2
    ids = np.arange(args.marker_id_start, args.marker_id_start + n_markers)
    marker_size = args.marker_size if args.marker_size is not None else args.square_size * 22.0 / 30.0
    board = aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_size,
                               marker_size, dictionary, ids=ids)
    board.setLegacyPattern(not args.no_legacy)
    return board


def load_transforms(dataset: Path):
    meta = json.loads((dataset / "transforms.json").read_text(encoding="utf-8"))
    K = np.array([[meta["fl_x"], 0, meta["cx"]],
                  [0, meta["fl_y"], meta["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array([meta.get("k1", 0.0), meta.get("k2", 0.0),
                     meta.get("p1", 0.0), meta.get("p2", 0.0)], dtype=np.float64)
    return meta, K, dist


def c2w_gl_to_w2c_cv(c2w_gl: np.ndarray) -> np.ndarray:
    c2w = c2w_gl.copy()
    c2w[0:3, 1:3] *= -1  # OpenGL (x right, y up, z back) -> OpenCV (x right, y down, z forward)
    return np.linalg.inv(c2w)


def detect_corners(dataset: Path, meta, board):
    """Return {corner_id: [(frame_idx, xy_pixels), ...]} over all images."""
    detector = cv2.aruco.CharucoDetector(board)
    observations: dict[int, list] = {}
    n_ok = 0
    for fi, frame in enumerate(meta["frames"]):
        image_path = dataset / frame["file_path"]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[skip] {image_path.name}: cannot read")
            continue
        ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
        if ch_ids is None or len(ch_ids) == 0:
            continue
        n_ok += 1
        for xy, cid in zip(ch_corners.reshape(-1, 2), ch_ids.flatten()):
            observations.setdefault(int(cid), []).append((fi, xy.astype(np.float64)))
        if (fi + 1) % 20 == 0:
            print(f"  detected in {n_ok}/{fi + 1} images...")
    print(f"Board seen in {n_ok}/{len(meta['frames'])} images, "
          f"{len(observations)} distinct corners observed.")
    return observations


def triangulate_dlt(projections, points_norm):
    """DLT triangulation from normalized image points. projections: list of 3x4 [R|t]."""
    rows = []
    for P, (x, y) in zip(projections, points_norm):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.stack(rows))
    X = vt[-1]
    return X[:3] / X[3]


def reproj_error_px(X, w2c, K, dist, xy):
    rvec, _ = cv2.Rodrigues(w2c[:3, :3])
    proj, _ = cv2.projectPoints(X.reshape(1, 3), rvec, w2c[:3, 3], K, dist)
    return float(np.linalg.norm(proj.reshape(2) - xy))


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity transform: dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (xs ** 2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return s, R, t


def transform_ply_ascii(src_path: Path, dst_path: Path, s: float, R: np.ndarray, t: np.ndarray):
    header, data_lines = [], []
    with open(src_path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            header.append(line.rstrip("\r\n"))
            if line.strip() == "end_header":
                break
        for line in f:
            if line.strip():
                data_lines.append(line.split())
    arr = np.array(data_lines)
    xyz = arr[:, :3].astype(np.float64)
    xyz = (s * (R @ xyz.T)).T + t
    with open(dst_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        rest = arr[:, 3:]
        for p, extra in zip(xyz, rest):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} " + " ".join(extra) + "\n")


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    unit = "m" if args.square_size != 1.0 else "square-units"

    board = build_board(args)
    meta, K, dist = load_transforms(dataset)
    w2c_all = [c2w_gl_to_w2c_cv(np.array(f["transform_matrix"], dtype=np.float64)) for f in meta["frames"]]

    observations = detect_corners(dataset, meta, board)

    # Triangulate every corner seen from enough views, with one outlier-rejection pass.
    board_corners = board.getChessboardCorners().astype(np.float64)  # (N,3), z=0
    tri_ids, tri_pts, reproj_all = [], [], []
    for cid, obs in sorted(observations.items()):
        for _ in range(2):
            if len(obs) < args.min_views:
                break
            Ps = [w2c_all[fi][:3, :4] for fi, _ in obs]
            pts_px = np.array([xy for _, xy in obs]).reshape(-1, 1, 2)
            pts_norm = cv2.undistortPoints(pts_px, K, dist).reshape(-1, 2)
            X = triangulate_dlt(Ps, pts_norm)
            errors = [reproj_error_px(X, w2c_all[fi], K, dist, xy) for fi, xy in obs]
            inliers = [o for o, e in zip(obs, errors) if e <= args.max_reproj_px]
            if len(inliers) == len(obs):
                break
            obs = inliers
        if len(obs) >= args.min_views:
            tri_ids.append(cid)
            tri_pts.append(X)
            reproj_all.extend(errors)

    if len(tri_ids) < 6:
        raise SystemExit(f"Only {len(tri_ids)} corners triangulated; not enough to align.")
    tri_pts = np.array(tri_pts)
    reproj_all = np.array(reproj_all)
    print(f"Triangulated {len(tri_ids)} corners "
          f"(reprojection px: median {np.median(reproj_all):.2f}, p95 {np.percentile(reproj_all, 95):.2f})")

    ref = board_corners[tri_ids]
    if args.origin == "center":
        ref = ref - np.array([args.squares_x, args.squares_y, 0.0]) * args.square_size / 2.0

    s, R, t = umeyama(tri_pts, ref)

    # Make sure cameras end up on the +Z side of the board (Z pointing toward cameras).
    cam_z = np.array([(s * R @ np.linalg.inv(w)[:3, 3] + t)[2] for w in w2c_all])
    if np.mean(cam_z > 0) < 0.5:
        # Mirror the board frame about its X axis (y -> H - y, z -> -z): the board
        # still spans positive X/Y but Z now points toward the cameras.
        if args.origin == "center":
            ref[:, 1] *= -1
        else:
            ref[:, 1] = args.squares_y * args.square_size - ref[:, 1]
        s, R, t = umeyama(tri_pts, ref)
        print("Flipped board frame so cameras sit at Z > 0.")
    residuals = np.linalg.norm((s * (R @ tri_pts.T)).T + t - ref, axis=1)

    scale_note = f"1 colmap-unit = {s:.6f} {unit}"
    print(f"\nAlignment fit over {len(tri_ids)} corners:")
    print(f"  scale     : {scale_note}")
    print(f"  residuals : RMS {np.sqrt((residuals**2).mean()):.5f} {unit}, max {residuals.max():.5f} {unit}")
    cam_pos = np.array([np.linalg.inv(w)[:3, 3] for w in w2c_all])
    cam_pos = (s * (R @ cam_pos.T)).T + t
    print(f"  camera Z  : min {cam_pos[:,2].min():.3f}, mean {cam_pos[:,2].mean():.3f}, "
          f"max {cam_pos[:,2].max():.3f} {unit}")

    # Bake into transforms.json (poses stay OpenGL convention; rotation kept orthonormal).
    S4 = np.eye(4)
    S4[:3, :3] = s * R
    S4[:3, 3] = t
    out = copy.deepcopy(meta)
    for frame in out["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[0:3, 1:3] *= -1  # GL -> CV
        c2w_new = S4 @ c2w
        c2w_new[:3, :3] /= s  # remove scale from rotation
        c2w_new[0:3, 1:3] *= -1  # CV -> GL
        frame["transform_matrix"] = c2w_new.tolist()
    if "applied_transform" in out:
        A4 = np.eye(4)
        A4[:3, :] = np.array(out["applied_transform"], dtype=np.float64)
        out["applied_transform"] = (S4 @ A4)[:3, :].tolist()

    ply_in = dataset / meta.get("ply_file_path", "sparse_pc.ply")
    if ply_in.exists():
        ply_out = f"sparse_pc_{args.out_suffix}.ply"
        transform_ply_ascii(ply_in, dataset / ply_out, s, R, t)
        out["ply_file_path"] = ply_out
        print(f"Wrote {dataset / ply_out}")

    json_out = dataset / f"transforms_{args.out_suffix}.json"
    json_out.write_text(json.dumps(out, indent=4), encoding="utf-8")
    print(f"Wrote {json_out}")
    if unit != "m":
        print("\nNOTE: run again with --square-size <meters> to get a metric world frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
