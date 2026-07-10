"""Collapse duplicate COLMAP cameras into one shared camera.

COLMAP's feature_extractor creates a NEW camera each time it is (re)run on the
same database, even with --ImageReader.single_camera 1. If extraction was
interrupted and rerun, the sparse model ends up with several cameras that share
the locked fx/fy/cx/cy but drifted apart in refined distortion. Downstream
tools (align_to_charuco.py, survey_aruco_tags.py) expect a single camera in
transforms.json, so merge before ns-process-data.

Afterwards re-converge the shared distortion over all images:

  COLMAP.bat bundle_adjuster --input_path <out> --output_path <out> ^
      --BundleAdjustment.refine_focal_length 0 ^
      --BundleAdjustment.refine_principal_point 0 ^
      --BundleAdjustment.refine_extra_params 1
"""

import argparse
from collections import Counter
from pathlib import Path

from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=r"E:\Grasp\data\lab_colmap_init\colmap\sparse\0",
                        help="Input sparse model dir (cameras.bin/images.bin/points3D.bin).")
    parser.add_argument("--out", default=None,
                        help="Output dir. Default: sibling '0_merged' of the input dir.")
    args = parser.parse_args()

    model = Path(args.model)
    out = Path(args.out) if args.out else model.parent / "0_merged"

    cameras = read_cameras_binary(model / "cameras.bin")
    images = read_images_binary(model / "images.bin")
    points = read_points3D_binary(model / "points3D.bin")

    if len(cameras) == 1:
        print("Model already has a single camera; nothing to do.")
        return

    usage = Counter(im.camera_id for im in images.values())
    for cid in sorted(cameras):
        c = cameras[cid]
        print(f"camera {cid}: {usage.get(cid, 0)} images, params {[round(p, 4) for p in c.params]}")

    # All cameras must agree on the locked part (fx fy cx cy) for a loss-free merge.
    base = cameras[min(cameras)]
    for c in cameras.values():
        if list(c.params[:4]) != list(base.params[:4]) or (c.width, c.height) != (base.width, base.height):
            raise SystemExit("Cameras differ in focal/principal point or size -- not safe to merge.")

    keep_id = usage.most_common(1)[0][0]
    kept = cameras[keep_id]
    merged_camera = kept._replace(id=1)
    merged_images = {im_id: im._replace(camera_id=1) for im_id, im in images.items()}

    out.mkdir(parents=True, exist_ok=True)
    write_cameras_binary({1: merged_camera}, out / "cameras.bin")
    write_images_binary(merged_images, out / "images.bin")
    write_points3D_binary(points, out / "points3D.bin")
    print(f"Kept camera {keep_id}'s distortion as init for {len(images)} images.")
    print(f"Wrote merged model to {out} -- now run bundle_adjuster on it (see module docstring).")


if __name__ == "__main__":
    main()
