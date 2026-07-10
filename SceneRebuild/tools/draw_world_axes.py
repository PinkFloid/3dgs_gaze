#!/usr/bin/env python
"""Visual check for align_to_charuco.py output.

Projects the aligned world frame back into the original photos:
  - X axis (red), Y axis (green), Z axis (blue) drawn at the world origin
  - magenta dots on every chessboard grid intersection

If the alignment is correct, the axes sit exactly on the board corner and the
dots land on the printed grid. Also writes a colored axes point cloud merged
with the aligned sparse cloud for 3D viewers (CloudCompare / MeshLab).

Usage:
  python tools/draw_world_axes.py --dataset data/lab_colmap_v2 [--num-images 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Dataset dir containing transforms_aligned.json.")
    parser.add_argument("--transforms", default="transforms_aligned.json", help="Aligned transforms file name.")
    parser.add_argument("--num-images", type=int, default=4, help="How many check images to render.")
    parser.add_argument("--squares-x", type=int, default=11)
    parser.add_argument("--squares-y", type=int, default=8)
    parser.add_argument("--square-size", type=float, default=0.03, help="Square size in world units (m).")
    parser.add_argument("--axis-length", type=float, default=0.09, help="Drawn axis length in world units (m).")
    parser.add_argument("--out-dir", default="axes_check", help="Output subfolder for preview images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    meta = json.loads((dataset / args.transforms).read_text(encoding="utf-8"))
    K = np.array([[meta["fl_x"], 0, meta["cx"]], [0, meta["fl_y"], meta["cy"]], [0, 0, 1]])
    dist = np.array([meta.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")])

    sq, L = args.square_size, args.axis_length
    board_center = np.array([args.squares_x * sq / 2, args.squares_y * sq / 2, 0.0])
    axes_pts = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], dtype=np.float64)
    grid = np.array([[i * sq, j * sq, 0.0]
                     for i in range(args.squares_x + 1) for j in range(args.squares_y + 1)])

    # Rank frames: camera close to the board and looking at it, key points inside the image.
    candidates = []
    for frame in meta["frames"]:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[0:3, 1:3] *= -1  # GL -> CV
        w2c = np.linalg.inv(c2w)
        rvec, _ = cv2.Rodrigues(w2c[:3, :3])
        tvec = w2c[:3, 3]
        check = np.vstack([axes_pts, board_center])
        cam_z = (w2c[:3, :3] @ check.T + tvec.reshape(3, 1))[2]
        if (cam_z <= 0).any():
            continue
        proj, _ = cv2.projectPoints(check, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        if not ((proj >= 0) & (proj < [meta["w"], meta["h"]])).all():
            continue
        dist_to_board = np.linalg.norm(c2w[:3, 3] - board_center)
        candidates.append((dist_to_board, frame["file_path"], rvec, tvec))
    candidates.sort(key=lambda c: c[0])
    if not candidates:
        raise SystemExit("No frame sees the origin and axes; check the alignment.")

    out_dir = dataset / args.out_dir
    out_dir.mkdir(exist_ok=True)
    step = max(1, len(candidates) // args.num_images)
    picked = candidates[::step][: args.num_images]

    for dist_to_board, file_path, rvec, tvec in picked:
        img = cv2.imread(str(dataset / file_path))
        pts, _ = cv2.projectPoints(np.vstack([axes_pts, grid]), rvec, tvec, K, dist)
        pts = pts.reshape(-1, 2)
        o, x, y, z = pts[:4].astype(int)
        for g in pts[4:].astype(int):
            cv2.circle(img, tuple(g), 12, (255, 0, 255), -1)
        for end, color, label in [(x, (0, 0, 255), "X"), (y, (0, 255, 0), "Y"), (z, (255, 0, 0), "Z")]:
            cv2.line(img, tuple(o), tuple(end), color, 14, cv2.LINE_AA)
            cv2.putText(img, label, tuple(end + 30), cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 10, cv2.LINE_AA)
        cv2.circle(img, tuple(o), 22, (255, 255, 255), -1)
        cv2.circle(img, tuple(o), 22, (0, 0, 0), 4)
        scale = 2000.0 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out_path = out_dir / Path(file_path).name
        cv2.imwrite(str(out_path), img)
        print(f"Wrote {out_path}  (camera {dist_to_board:.2f} m from board center)")

    # Colored axes point cloud merged with the aligned sparse cloud for 3D viewers.
    ply_in = dataset / meta.get("ply_file_path", "sparse_pc_aligned.ply")
    if ply_in.exists():
        n_pts, spacing = 150, L * 2 / 150  # axes 2x drawn length, dense points
        rows = []
        for d, color in [((1, 0, 0), (255, 0, 0)), ((0, 1, 0), (0, 255, 0)), ((0, 0, 1), (0, 0, 255))]:
            for i in range(1, n_pts + 1):
                p = np.array(d) * spacing * i
                rows.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {color[0]} {color[1]} {color[2]}")
        text = ply_in.read_text(encoding="ascii", errors="replace")
        header_end = text.index("end_header")
        header, body = text[:header_end], text[text.index("\n", header_end) + 1:]
        old_count = int([l for l in header.splitlines() if l.startswith("element vertex")][0].split()[-1])
        header = header.replace(f"element vertex {old_count}", f"element vertex {old_count + len(rows)}")
        out_ply = dataset / "sparse_pc_axes_preview.ply"
        out_ply.write_text(header + "end_header\n" + body + "\n".join(rows) + "\n", encoding="ascii")
        print(f"Wrote {out_ply}  (viewing only; keep training on the clean aligned ply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
