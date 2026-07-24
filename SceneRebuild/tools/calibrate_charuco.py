#!/usr/bin/env python
"""Calibrate camera intrinsics from ChArUco board images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect a ChArUco calibration board in an image folder and estimate "
            "camera intrinsics. Board dimensions and lengths must match the "
            "physical printed board."
        )
    )
    parser.add_argument("--images", required=True, help="Image folder or glob pattern.")
    parser.add_argument("--out", default="camera_intrinsics.yml", help="Output YAML/JSON path.")
    parser.add_argument("--squares-x", type=int, required=True, help="Number of chessboard squares along X.")
    parser.add_argument("--squares-y", type=int, required=True, help="Number of chessboard squares along Y.")
    parser.add_argument("--square-length", type=float, required=True, help="Chessboard square side length.")
    parser.add_argument("--marker-length", type=float, required=True, help="ArUco marker side length.")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary name.")
    parser.add_argument("--marker-id-start", type=int, default=0,
                        help="First ArUco id on the board (calib.io 'Start Id').")
    parser.add_argument("--no-legacy", action="store_true",
                        help="Board was generated with the new (non-legacy) pattern. "
                             "calib.io boards are legacy, so legacy is the default.")
    parser.add_argument("--min-corners", type=int, default=8, help="Minimum ChArUco corners per image.")
    parser.add_argument("--max-image-err", type=float, default=1.5,
                        help="After each calibration round, drop images whose mean reprojection "
                             "error exceeds this (px), then recalibrate. 0 disables pruning.")
    parser.add_argument("--prune-rounds", type=int, default=2, help="Max prune+recalibrate rounds.")
    parser.add_argument(
        "--fix-k3",
        action="store_true",
        help=(
            "Fix k3=0 and fit only k1,k2,p1,p2. Use this when the intrinsics are "
            "consumed by COLMAP's OPENCV camera model, which has no k3."
        ),
    )
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="Optional folder for detection preview images.",
    )
    return parser.parse_args()


def import_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenCV is not installed. Install the contrib build first:\n"
            "  python -m pip install opencv-contrib-python numpy"
        ) from exc

    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "This OpenCV build does not include cv2.aruco. Install the contrib build:\n"
            "  python -m pip install opencv-contrib-python"
        )
    return cv2


def collect_images(images_arg: str) -> list[Path]:
    source = Path(images_arg)
    if source.is_dir():
        images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        images = [Path(p) for p in sorted(source.parent.glob(source.name))]
        images = [p for p in images if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    return sorted(images)


def get_dictionary(cv2, name: str):
    aruco = cv2.aruco
    if not hasattr(aruco, name):
        known = sorted(k for k in dir(aruco) if k.startswith("DICT_"))
        raise SystemExit(f"Unknown dictionary {name!r}. Known examples: {', '.join(known[:12])}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def create_charuco_board(cv2, squares_x: int, squares_y: int, square_length: float, marker_length: float,
                         dictionary, marker_id_start: int = 0, legacy: bool = True):
    aruco = cv2.aruco
    n_markers = (squares_x * squares_y) // 2
    ids = np.arange(marker_id_start, marker_id_start + n_markers)
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary, ids)
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(legacy)
        return board
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)
    raise SystemExit("This OpenCV build does not provide CharucoBoard APIs.")


def detect_charuco(cv2, gray, board, dictionary):
    aruco = cv2.aruco

    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary)
        charuco_corners, charuco_ids = None, None
        if marker_ids is not None and len(marker_ids) > 0:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )

    if charuco_corners is None or charuco_ids is None:
        return None, None
    return charuco_corners, charuco_ids


def draw_preview(cv2, image, charuco_corners, charuco_ids):
    preview = image.copy()
    cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners, charuco_ids)
    return preview


def upright_90cw(image_size, camera_matrix, dist_coeffs):
    """Intrinsics for the same frames rotated 90° clockwise to upright portrait
    ((x,y) -> (H-1-y, x)): fx/fy swap, principal point maps, tangential terms
    rotate as (p1', p2') = (p2, -p1); radial terms are rotation-invariant."""
    w, h = image_size
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    d = dist_coeffs.reshape(-1)
    return {
        "image_width": int(h), "image_height": int(w),
        "fx": float(fy), "fy": float(fx),
        "cx": float(h - 1 - cy), "cy": float(cx),
        "k1": float(d[0]), "k2": float(d[1]),
        "p1": float(d[3]), "p2": float(-d[2]),
    }


def save_output(cv2, out_path: Path, image_size, rms, camera_matrix, dist_coeffs, used_names, args,
                rejected=()):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "rms_reprojection_error": float(rms),
        "valid_images": len(used_names),
        "used_images": list(used_names),
        "rejected_images": [{"name": n, "mean_err_px": e} for n, e in rejected],
        "rejection": {"max_image_err_px": args.max_image_err, "rounds": args.prune_rounds},
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length": args.square_length,
            "marker_length": args.marker_length,
            "dictionary": args.dictionary,
            "marker_id_start": args.marker_id_start,
            "legacy": not args.no_legacy,
        },
        "upright_90cw": upright_90cw(image_size, camera_matrix, dist_coeffs),
    }

    if out_path.suffix.lower() == ".npz":
        np.savez(out_path,
                 camera_matrix=camera_matrix,
                 dist_coeffs=dist_coeffs.reshape(-1),
                 image_width=payload["image_width"],
                 image_height=payload["image_height"],
                 rms_reprojection_error=payload["rms_reprojection_error"],
                 valid_images=payload["valid_images"],
                 board=json.dumps(payload["board"]),
                 upright_90cw=json.dumps(payload["upright_90cw"]),
                 used_images=np.array(used_names))
        out_path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise SystemExit(f"Could not open output file for writing: {out_path}")
    fs.write("image_width", payload["image_width"])
    fs.write("image_height", payload["image_height"])
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("rms_reprojection_error", payload["rms_reprojection_error"])
    fs.write("valid_images", payload["valid_images"])
    fs.write("squares_x", args.squares_x)
    fs.write("squares_y", args.squares_y)
    fs.write("square_length", args.square_length)
    fs.write("marker_length", args.marker_length)
    fs.write("dictionary", args.dictionary)
    fs.release()


def main() -> int:
    args = parse_args()
    cv2 = import_cv2()

    images = collect_images(args.images)
    if not images:
        raise SystemExit(f"No images found from: {args.images}")

    dictionary = get_dictionary(cv2, args.dictionary)
    board = create_charuco_board(
        cv2,
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        dictionary,
        marker_id_start=args.marker_id_start,
        legacy=not args.no_legacy,
    )

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    all_corners = []
    all_ids = []
    used_names = []
    image_size = None

    print(f"Found {len(images)} image(s).")
    for image_path in images:
        # phones tag portrait shots via EXIF; load raw sensor orientation so all
        # frames share one (W,H) and one principal point convention
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            print(f"[skip] {image_path.name}: cannot read")
            continue

        height, width = image.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            print(f"[skip] {image_path.name}: image size {(width, height)} != {image_size}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids = detect_charuco(cv2, gray, board, dictionary)
        corner_count = 0 if charuco_ids is None else len(charuco_ids)
        if charuco_corners is None or charuco_ids is None or corner_count < args.min_corners:
            print(f"[skip] {image_path.name}: {corner_count} ChArUco corners")
            continue

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        used_names.append(image_path.name)
        print(f"[ok]   {image_path.name}: {corner_count} ChArUco corners")

        if preview_dir:
            preview = draw_preview(cv2, image, charuco_corners, charuco_ids)
            cv2.imwrite(str(preview_dir / image_path.name), preview)

    if image_size is None:
        raise SystemExit("No readable images.")

    if len(all_corners) < 5:
        raise SystemExit(
            f"Only {len(all_corners)} valid board image(s). Need at least 5, and 15-30 is better."
        )

    calib_flags = cv2.CALIB_FIX_K3 if args.fix_k3 else 0
    if not hasattr(board, "matchImagePoints"):
        # ancient OpenCV: single-shot charuco calibration, no pruning
        rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, image_size, None, None, flags=calib_flags)
        rejected = []
    else:
        # OpenCV >= 4.7 dropped calibrateCameraCharuco from the python bindings:
        # map charuco corner ids to board 3D corners, then plain calibrateCamera,
        # with per-image reprojection pruning (bad frames poison K silently)
        obj_pts, img_pts = [], []
        for corners, ids in zip(all_corners, all_ids):
            o, i = board.matchImagePoints(corners, ids)
            obj_pts.append(o)
            img_pts.append(i)
        rejected = []
        for round_i in range(max(1, args.prune_rounds + 1)):
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_pts, img_pts, image_size, None, None, flags=calib_flags)
            errs = []
            for o, i, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
                proj, _ = cv2.projectPoints(o, rv, tv, camera_matrix, dist_coeffs)
                errs.append(float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2)))))
            order = np.argsort(errs)[::-1]
            print(f"round {round_i}: rms {rms:.3f}px over {len(obj_pts)} images; worst: "
                  + ", ".join(f"{used_names[k]} {errs[k]:.2f}" for k in order[:5]))
            if args.max_image_err <= 0 or round_i >= args.prune_rounds:
                break
            keep = [k for k in range(len(errs)) if errs[k] <= args.max_image_err]
            if len(keep) == len(errs) or len(keep) < 5:
                break
            rejected += [(used_names[k], round(errs[k], 3)) for k in range(len(errs))
                         if errs[k] > args.max_image_err]
            obj_pts = [obj_pts[k] for k in keep]
            img_pts = [img_pts[k] for k in keep]
            used_names = [used_names[k] for k in keep]
    if rejected:
        print(f"pruned {len(rejected)}: " + ", ".join(f"{n}({e})" for n, e in rejected))

    out_path = Path(args.out)
    save_output(cv2, out_path, image_size, rms, camera_matrix, dist_coeffs, used_names, args,
                rejected)

    print()
    print(f"RMS reprojection error: {rms:.6f}")
    print("Camera matrix:")
    print(np.array2string(camera_matrix, precision=6, suppress_small=True))
    print("Distortion coefficients:")
    print(np.array2string(dist_coeffs.reshape(-1), precision=6, suppress_small=True))
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
