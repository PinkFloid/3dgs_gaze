"""Bake EXIF orientation into pixels so COLMAP / nerfstudio see consistent portrait images.

Skips photos whose stored resolution doesn't match the calibrated 48MP mode
(5712x4284 raw landscape) -- e.g. IMG_1100-1104 were shot in 12MP mode and
don't match phone_camera_calibration.npz.
"""

import os
import sys

from PIL import Image, ImageOps

SRC = sys.argv[1] if len(sys.argv) > 1 else r"E:\Grasp\data\lab"
DST = sys.argv[2] if len(sys.argv) > 2 else r"E:\Grasp\data\lab_upright"
EXPECTED_RAW_SIZE = (5712, 4284)

os.makedirs(DST, exist_ok=True)

files = sorted(f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".jpeg")))
skipped = []
for i, name in enumerate(files, 1):
    im = Image.open(os.path.join(SRC, name))
    if im.size != EXPECTED_RAW_SIZE:
        skipped.append((name, im.size))
        continue
    im = ImageOps.exif_transpose(im)
    im.save(os.path.join(DST, name), quality=95)
    if i % 25 == 0:
        print(f"{i}/{len(files)}", flush=True)

print(f"converted: {len(files) - len(skipped)}")
for name, size in skipped:
    print(f"skipped (resolution {size[0]}x{size[1]} != calibrated mode): {name}")
