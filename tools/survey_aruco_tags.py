#!/usr/bin/env python
"""Survey ArUco tag poses in the ChArUco-aligned world frame.

Run AFTER align_to_charuco.py, on the same dataset. For every ArUco tag seen in
the capture photos:
  1. Detect the tag's 4 corners (subpixel-refined) in every dataset image.
  2. Triangulate each corner in the world frame using the camera poses +
     intrinsics from transforms_aligned.json (multi-view DLT, outlier pass).
  3. Fit a rigid transform (no scale -- the tag size is known) from the tag's
     canonical corner layout to the triangulated corners: T_world_tag.

The measured side lengths double as a metric sanity check: if the world frame
is in meters, they should come out at --tag-size.

Output tags_world.json is what the online eye-tracker localizer consumes:
detect tag -> solvePnP -> T_cam_tag, then T_world_cam = T_world_tag @ inv(T_cam_tag).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="ns-process-data output dir.")
    parser.add_argument("--transforms", default="transforms_aligned.json",
                        help="Which transforms json inside the dataset holds the world-frame poses.")
    parser.add_argument("--dictionary", default="DICT_6X6_250", help="OpenCV ArUco dictionary name.")
    parser.add_argument("--tag-ids", default="0-29", help="Tag id range 'a-b' or comma list. Ids outside are ignored.")
    parser.add_argument("--tag-size", type=float, default=0.10, help="Tag side length in meters (black border).")
    parser.add_argument("--tag-sizes", default=None,
                        help="Per-range sizes 'a-b:size,c-d:size' (m), overriding --tag-size for those "
                             "ids. Mixed deployments: e.g. '0-29:0.099,74-249:0.24' for the old 6-up "
                             "sheets plus the new A3 singles. Corner triangulation is size-free; this "
                             "only affects the rigid-fit template and the metric sanity check.")
    parser.add_argument("--min-views", type=int, default=3, help="Min views per corner to triangulate.")
    parser.add_argument("--max-reproj-px", type=float, default=4.0, help="Reprojection outlier threshold (pixels).")
    parser.add_argument("--out", default="tags_world.json", help="Output json (relative to dataset).")
    parser.add_argument("--preview-dir", default=None, help="Optional folder for detection/reprojection previews.")
    return parser.parse_args()


def parse_ids(spec: str) -> set[int]:
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        elif part:
            ids.add(int(part))
    return ids


def make_size_of(spec: str | None, default: float):
    """Resolve per-tag expected size from 'a-b:size,c:size' spec, else default."""
    ranges: list[tuple[set[int], float]] = []
    if spec:
        for part in spec.split(","):
            ids_part, s = part.rsplit(":", 1)
            ranges.append((parse_ids(ids_part), float(s)))

    def size_of(mid: int) -> float:
        for ids, s in ranges:
            if mid in ids:
                return s
        return default
    return size_of


def load_transforms(dataset: Path, name: str):
    meta = json.loads((dataset / name).read_text(encoding="utf-8"))
    K = np.array([[meta["fl_x"], 0, meta["cx"]],
                  [0, meta["fl_y"], meta["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array([meta.get("k1", 0.0), meta.get("k2", 0.0),
                     meta.get("p1", 0.0), meta.get("p2", 0.0)], dtype=np.float64)
    return meta, K, dist


def c2w_gl_to_w2c_cv(c2w_gl: np.ndarray) -> np.ndarray:
    c2w = c2w_gl.copy()
    c2w[0:3, 1:3] *= -1  # OpenGL -> OpenCV camera axes
    return np.linalg.inv(c2w)


def triangulate_dlt(projections, points_norm):
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


def kabsch(src: np.ndarray, dst: np.ndarray):
    """Rigid transform (rotation + translation, unit scale): dst ~= R @ src + t."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    cov = (dst - mu_d).T @ (src - mu_s)
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = mu_d - R @ mu_s
    return R, t


def tag_object_corners(size: float) -> np.ndarray:
    """Tag frame: origin at center, X right, Y up, Z out of the tag face.
    Corner order matches cv2.aruco detectMarkers: TL, TR, BR, BL."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset)
    wanted = parse_ids(args.tag_ids)

    meta, K, dist = load_transforms(dataset, args.transforms)
    w2c_all = [c2w_gl_to_w2c_cv(np.array(f["transform_matrix"], dtype=np.float64)) for f in meta["frames"]]

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    # observations[(tag_id, corner_idx)] = [(frame_idx, xy), ...]
    observations: dict[tuple[int, int], list] = {}
    frames_with_tag: dict[int, int] = {}
    for fi, frame in enumerate(meta["frames"]):
        image_path = dataset / frame["file_path"]
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[skip] {image_path.name}: cannot read")
            continue
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue
        hit = False
        for quad, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in wanted:
                continue
            hit = True
            frames_with_tag[mid] = frames_with_tag.get(mid, 0) + 1
            for ci, xy in enumerate(quad.reshape(4, 2)):
                observations.setdefault((mid, ci), []).append((fi, xy.astype(np.float64)))
        if hit and preview_dir:
            img = cv2.imread(str(image_path))
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            cv2.imwrite(str(preview_dir / image_path.name), img)
        if (fi + 1) % 20 == 0:
            print(f"  scanned {fi + 1}/{len(meta['frames'])} images, {len(frames_with_tag)} tags so far...")

    print(f"Tags seen: {sorted(frames_with_tag)} "
          f"(views per tag: {dict(sorted(frames_with_tag.items()))})")

    # Triangulate each corner with one outlier-rejection pass.
    corners_world: dict[int, dict[int, np.ndarray]] = {}
    for (mid, ci), obs in sorted(observations.items()):
        errors = []
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
            corners_world.setdefault(mid, {})[ci] = X

    size_of = make_size_of(args.tag_sizes, args.tag_size)
    result = {}
    print(f"\n{'tag':>4} {'views':>5} {'side_meas(m)':>13} {'expected':>9} {'fit_rms(mm)':>12}  status")
    for mid in sorted(corners_world):
        cw = corners_world[mid]
        if len(cw) < 4:
            print(f"{mid:>4} {frames_with_tag.get(mid, 0):>5} {'-':>13} {'-':>9} {'-':>12}  only {len(cw)}/4 corners triangulated")
            continue
        expected = size_of(mid)
        obj = tag_object_corners(expected)
        pts = np.stack([cw[ci] for ci in range(4)])
        sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
        R, t = kabsch(obj, pts)
        resid = np.linalg.norm((R @ obj.T).T + t - pts, axis=1)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        result[str(mid)] = {
            "T_world_tag": T.tolist(),
            "corners_world": pts.tolist(),
            "side_lengths_m": [float(s) for s in sides],
            "expected_size_m": expected,
            "fit_rms_m": float(np.sqrt((resid ** 2).mean())),
            "n_views": frames_with_tag.get(mid, 0),
        }
        print(f"{mid:>4} {frames_with_tag.get(mid, 0):>5} {np.mean(sides):>13.4f} {expected:>9.3f} "
              f"{np.sqrt((resid**2).mean())*1000:>12.2f}  ok")

    if not result:
        raise SystemExit("No tag could be surveyed -- check dictionary/ids and that tags are visible in the capture.")

    ratios = np.array([np.mean(v["side_lengths_m"]) / v["expected_size_m"] for v in result.values()])
    print(f"\nMeasured/expected side ratio over {len(result)} tags: {ratios.mean():.4f} "
          f"(per-tag range [{ratios.min():.4f}, {ratios.max():.4f}])")
    if abs(ratios.mean() - 1) > 0.05:
        print("WARNING: >5% off -- world frame is probably not metric (re-run align_to_charuco "
              "with --square-size in meters), or the per-range --tag-sizes are wrong.")

    out = dataset / args.out
    out.write_text(json.dumps({
        "world_frame": f"{args.transforms} (ChArUco board frame)",
        "tag_size_m": args.tag_size,
        "tag_sizes_spec": args.tag_sizes,
        "dictionary": args.dictionary,
        "tag_frame": "origin at tag center, X right, Y up, Z out of tag face; corners TL,TR,BR,BL",
        "tags": result,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
