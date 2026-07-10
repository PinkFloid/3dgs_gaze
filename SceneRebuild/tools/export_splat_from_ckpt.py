#!/usr/bin/env python
"""Export a 3DGS splat.ply straight from a splatfacto checkpoint.

Bypasses `ns-export gaussian-splat`, which needs the training dataset present
and chokes on WindowsPath config.yml when the model was trained on Windows.
The gaussians are written exactly as stored (model/world space) -- for models
trained on transforms_aligned.json with orientation/center/auto-scale disabled
that IS the ChArUco board frame in meters.

Output layout matches the standard 3DGS / nerfstudio ply (x y z nx ny nz
f_dc_* f_rest_* opacity scale_* rot_*), readable by SuperSplat & co.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="step-XXXXXXXXX.ckpt from splatfacto.")
    parser.add_argument("--out", default=None, help="Output .ply (default: splat.ply next to nerfstudio_models/).")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out = Path(args.out) if args.out else ckpt_path.parent.parent / "splat.ply"

    sd = torch.load(ckpt_path, map_location="cpu")["pipeline"]
    g = lambda name: sd[f"_model.gauss_params.{name}"].numpy()

    means = g("means").astype(np.float32)
    scales = g("scales").astype(np.float32)          # log scale, as stored
    quats = g("quats").astype(np.float32)            # wxyz, as stored
    opac = g("opacities").reshape(-1, 1).astype(np.float32)  # logit, as stored
    f_dc = g("features_dc").astype(np.float32)
    f_rest = g("features_rest")                       # (N, 15, 3)
    f_rest = f_rest.transpose(0, 2, 1).reshape(len(means), -1).astype(np.float32)  # channel-major

    n = len(means)
    cols = (
        [("x", means[:, 0]), ("y", means[:, 1]), ("z", means[:, 2]),
         ("nx", np.zeros(n, np.float32)), ("ny", np.zeros(n, np.float32)), ("nz", np.zeros(n, np.float32))]
        + [(f"f_dc_{i}", f_dc[:, i]) for i in range(f_dc.shape[1])]
        + [(f"f_rest_{i}", f_rest[:, i]) for i in range(f_rest.shape[1])]
        + [("opacity", opac[:, 0])]
        + [(f"scale_{i}", scales[:, i]) for i in range(3)]
        + [(f"rot_{i}", quats[:, i]) for i in range(4)]
    )

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {name}" for name, _ in cols]
    header += ["end_header"]

    data = np.stack([c for _, c in cols], axis=1)
    with open(out, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.astype("<f4").tobytes())

    lo, hi = np.percentile(means, 5, axis=0), np.percentile(means, 95, axis=0)
    print(f"wrote {out}: {n} gaussians")
    print(f"extent p5..p95: x [{lo[0]:.2f}, {hi[0]:.2f}]  y [{lo[1]:.2f}, {hi[1]:.2f}]  z [{lo[2]:.2f}, {hi[2]:.2f}] (m if aligned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
