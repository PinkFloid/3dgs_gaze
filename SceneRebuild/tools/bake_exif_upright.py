#!/usr/bin/env python
"""Bake EXIF orientation into pixels before ns-process-data.

ns-process-data strips the EXIF Orientation tag without rotating pixels, so
phone photos (stored landscape + rotation tag) come out sideways. Run this
first: it writes upright copies (original files untouched, other EXIF kept).

Usage:
  python tools/bake_exif_upright.py --src data/lab2 --dst data/lab2_upright
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Folder with original photos.")
    parser.add_argument("--dst", required=True, help="Output folder for upright copies.")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95).")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    sizes: dict[tuple, int] = {}
    n = 0
    for path in sorted(src.iterdir()):
        if path.suffix.lower() not in EXTENSIONS:
            continue
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)  # rotate pixels, drop the orientation tag
        exif = im.info.get("exif")
        save_kwargs = {"quality": args.quality} if path.suffix.lower() in {".jpg", ".jpeg"} else {}
        if exif:
            save_kwargs["exif"] = exif
        im.save(dst / path.name, **save_kwargs)
        sizes[im.size] = sizes.get(im.size, 0) + 1
        n += 1
        if n % 50 == 0:
            print(f"  {n} done...")

    print(f"Baked {n} images -> {dst}")
    for size, count in sizes.items():
        print(f"  {size[0]}x{size[1]}: {count}")
    if len(sizes) > 1:
        print("WARNING: mixed image sizes (portrait + landscape shots?). "
              "COLMAP shared-intrinsics assumes one size - shoot all photos in the same orientation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
