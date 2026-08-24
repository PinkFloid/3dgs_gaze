#!/usr/bin/env python3
"""make_offline_assets.py -- Fig.2 离线条四槽位真图(同一视角 frame_00061)。

    python paper/fig2_assets/make_offline_assets.py   (需 gsplat JIT 环境变量)

产出(本目录,统一中心裁成 2.47:1 横条,宽 1200px):
  offline_capture.jpg  原始建图照片
  offline_colmap.jpg   稀疏点云投影到同一相机(白底彩点)
  offline_3dgs.jpg     ckpt 渲染同一位姿
  offline_sam.jpg      SAM 实例点着色预览(现成图裁切)
顺带打印 names.json 非空命名,给第 5 槽排版用。
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("E:/Grasp/data/lab_colmap_v9")
CKPT = Path("E:/Grasp/outputs/lab_colmap_v9/splatfacto/2026-08-20_201525/nerfstudio_models/step-000029999.ckpt")
SEG = ROOT / "SceneRebuild/lab_result/segmentation_sam"
OUT = Path(__file__).resolve().parent
FRAME = "frame_00049"
ASPECT = 2.47          # 槽位图宽:高
BAND_CY = 0.45         # 裁切带中心(相对高度)
RENDER_DIV = 4         # 渲染分辨率 = 原图 / 4


def band_crop(img: np.ndarray, width_out: int = 1200) -> np.ndarray:
    h, w = img.shape[:2]
    bh = int(round(w / ASPECT))
    y0 = int(round(BAND_CY * h - bh / 2))
    y0 = max(0, min(h - bh, y0))
    band = img[y0:y0 + bh]
    return cv2.resize(band, (width_out, int(round(width_out / ASPECT))), interpolation=cv2.INTER_AREA)


def read_ply(path: Path):
    """极简 PLY 读取(binary_little_endian / ascii;取 x y z [red green blue])。"""
    f = path.open("rb")
    assert f.readline().strip() == b"ply"
    fmt, n, props = None, 0, []
    while True:
        line = f.readline().strip().split()
        if line[0] == b"format":
            fmt = line[1].decode()
        elif line[0] == b"element":
            if line[1] == b"vertex":
                n = int(line[2]); cur = props
            else:
                cur = []
        elif line[0] == b"property" and line[1] != b"list":
            cur.append((line[2].decode(), line[1].decode()))
        elif line[0] == b"end_header":
            break
    sizes = {"float": ("f", 4), "float32": ("f", 4), "double": ("d", 8),
             "uchar": ("B", 1), "uint8": ("B", 1), "int": ("i", 4), "uint": ("I", 4)}
    if fmt == "ascii":
        rows = np.loadtxt(f, max_rows=n)
        cols = {name: rows[:, i] for i, (name, _) in enumerate(props)}
    else:
        stfmt = "<" + "".join(sizes[t][0] for _, t in props)
        rec = struct.calcsize(stfmt)
        buf = f.read(rec * n)
        arr = np.array([struct.unpack_from(stfmt, buf, i * rec) for i in range(n)])
        cols = {name: arr[:, i] for i, (name, _) in enumerate(props)}
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    if "red" in cols:
        rgb = np.stack([cols["red"], cols["green"], cols["blue"]], axis=1)
        rgb = rgb / 255.0 if rgb.max() > 1.5 else rgb
    else:
        rgb = np.full((len(xyz), 3), 0.4)
    return xyz, rgb


def main():
    tj = json.load((DATA / "transforms_aligned.json").open())
    fr = next(f for f in tj["frames"] if FRAME in f["file_path"])
    c2w = np.array(fr["transform_matrix"], dtype=np.float64)
    c2w_cv = c2w.copy()
    c2w_cv[:3, 1:3] *= -1.0                      # OpenGL -> OpenCV
    W, H = int(tj["w"]), int(tj["h"])
    K = np.array([[tj["fl_x"], 0, tj["cx"]], [0, tj["fl_y"], tj["cy"]], [0, 0, 1]])

    # 1) capture
    photo = cv2.imread(str(DATA / fr["file_path"]))
    assert photo is not None and photo.shape[1] == W, f"photo {photo.shape} vs {W}x{H}"
    cv2.imwrite(str(OUT / "offline_capture.jpg"), band_crop(photo), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2) colmap sparse + camera frusta -- 脱开的第三视角俯瞰(经典 SfM 构图)
    xyz, rgb = read_ply(DATA / "sparse_pc_aligned.ply")
    lo, hi = np.percentile(xyz, 1, axis=0), np.percentile(xyz, 99, axis=0)
    keep = ((xyz > lo) & (xyz < hi)).all(axis=1) & (xyz[:, 2] < 2.6)
    P, Prgb = xyz[keep], rgb[keep]
    cams = []
    for f2 in tj["frames"]:
        cf = np.array(f2["transform_matrix"], dtype=np.float64)
        cf[:3, 1:3] *= -1.0
        cams.append(cf)
    ring = np.stack([c[:3, 3] for c in cams])
    target = np.median(P, axis=0); target[2] = 0.6
    outv = ring.mean(axis=0)[:2] - target[:2]
    outv = outv / (np.linalg.norm(outv) + 1e-9)
    radius = np.linalg.norm(ring[:, :2] - target[None, :2], axis=1).max()
    eye = np.array([*(target[:2] + outv * radius * 1.35), ring[:, 2].max() + 2.2])
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0])); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    if y[2] > 0:
        x, y = -x, -y
    c2w_o = np.eye(4); c2w_o[:3, :3] = np.stack([x, y, z], axis=1); c2w_o[:3, 3] = eye
    w2c_o = np.linalg.inv(c2w_o)
    cw, ch = 2400, int(round(2400 / ASPECT))
    pc = (w2c_o[:3, :3] @ P.T + w2c_o[:3, 3:4]).T
    fz = pc[:, 2] > 0.2
    xn, yn = pc[fz, 0] / pc[fz, 2], pc[fz, 1] / pc[fz, 2]
    f = min(0.47 * cw / np.percentile(np.abs(xn), 97), 0.47 * ch / np.percentile(np.abs(yn), 97))
    Ko = np.array([[f, 0, cw / 2], [0, f, ch / 2], [0, 0, 1.0]])
    canvas = np.full((ch, cw, 3), 255, np.uint8)
    uv = np.stack([xn * f + cw / 2, yn * f + ch / 2], axis=1).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < cw) & (uv[:, 1] >= 0) & (uv[:, 1] < ch)
    cols = (Prgb[fz][ok][:, ::-1] * 255).astype(np.uint8)
    for (u, v), c in zip(uv[ok], cols):
        cv2.circle(canvas, (u, v), 1, c.tolist(), -1)
    Kinv = np.linalg.inv(K)
    fr_d = 0.13
    n_frusta = 0
    for cf in cams:
        corners = [np.zeros(3)] + [Kinv @ np.array([u_, v_, 1.0]) * fr_d
                                   for u_, v_ in ((0, 0), (W, 0), (W, H), (0, H))]
        pw = np.stack([cf[:3, :3] @ c + cf[:3, 3] for c in corners])
        pv = (w2c_o[:3, :3] @ pw.T + w2c_o[:3, 3:4]).T
        if (pv[:, 2] < 0.2).any():
            continue
        puv = (Ko[:2, :2] @ (pv[:, :2] / pv[:, 2:3]).T).T + Ko[:2, 2]
        pts = puv.astype(int)
        for a, b in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (60, 60, 230), 1, cv2.LINE_AA)
        n_frusta += 1
    print(f"colmap overview: {ok.sum()} points, {n_frusta} frusta")
    out_img = cv2.resize(canvas, (1200, int(round(1200 / ASPECT))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "offline_colmap.jpg"), out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3) 3DGS render at the same pose
    sys.path.insert(0, str(ROOT / "Eye_Tracker/tools"))
    from gaze_to_world import SplatDepth
    sd = SplatDepth(CKPT)
    rw, rh = W // RENDER_DIV, H // RENDER_DIV
    Kr = K / RENDER_DIV
    Kr[2, 2] = 1.0
    rgb_img = sd.render_view(c2w_cv, Kr, rw, rh)
    cv2.imwrite(str(OUT / "offline_3dgs.jpg"), band_crop(rgb_img[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 4) SAM preview crop
    prev = cv2.imread(str(SEG / f"preview/{FRAME}.jpg"))
    cv2.imwrite(str(OUT / "offline_sam.jpg"), band_crop(prev), [cv2.IMWRITE_JPEG_QUALITY, 92])

    names = json.load((SEG / "names.json").open(encoding="utf-8"))
    named = {k: v for k, v in names.items() if v}
    print(f"registry: {len(names)} instances, {len(named)} named -> {named}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
