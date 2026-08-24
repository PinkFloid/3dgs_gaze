#!/usr/bin/env python3
"""make_hub_map.py -- Fig.2 中心大图:合成水平相机渲染 + 高斯实例 alpha 掩膜着色。

    python paper/fig2_assets/make_hub_map.py [--dist 1.9 --eye-height 1.55 --hfov 58 --yaw 0]

相机:不再继承手持位姿——在"物品台 -> 拍摄环质心"方向、人眼高度架一个
水平 look-at 合成相机(相机 x 轴严格水平,画面不歪),按槽位比例 3.34:1 原生渲染。
掩膜:points.npz 实例点最近邻回配到 ckpt 高斯,每个命名实例的高斯子集单独
光栅化,alpha 即软掩膜;与全场景深度比较做遮挡剔除;半透明着色+描边,
名字标质心(白名单 --labels + 贪心避让)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("E:/Grasp/data/lab_colmap_v9")
CKPT = Path("E:/Grasp/outputs/lab_colmap_v9/splatfacto/2026-08-20_201525/nerfstudio_models/step-000029999.ckpt")
SEG = ROOT / "SceneRebuild/lab_result/segmentation_sam"
OUT = Path(__file__).resolve().parent
ASPECT = 14.2 / 4.25

EN = {"球L": "ball_L", "球M": "ball_M", "球R": "ball_R", "苹果红": "apple_red",
      "苹果粉": "apple_pink", "白杯1": "cup_1", "白杯2": "cup_2", "红杯": "cup_red",
      "橘子": "orange", "香蕉": "banana", "水瓶": "bottle", "纸箱子": "box",
      "物品台": "cart"}
PALETTE = [(60, 76, 231), (18, 156, 243), (113, 204, 46), (219, 152, 52), (182, 89, 155),
           (34, 126, 230), (156, 188, 26), (43, 57, 192), (133, 160, 22), (241, 196, 15),
           (96, 174, 39), (185, 128, 41), (94, 73, 52), (15, 196, 241), (140, 140, 140)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=float, default=1.9, help="相机到物品台水平距离 m")
    ap.add_argument("--eye-height", type=float, default=1.55)
    ap.add_argument("--target-z", type=float, default=None, help="默认=命名实例点位 z 中位")
    ap.add_argument("--hfov", type=float, default=58.0)
    ap.add_argument("--yaw", type=float, default=0.0, help="绕目标水平旋转视点(度)")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--labels", default="ball_L,ball_M,ball_R,cup_1,cup_2,banana,bottle,orange")
    ap.add_argument("--out", default="hub_map.jpg")
    args = ap.parse_args()

    npz = np.load(SEG / "points.npz")
    xyz, label = npz["xyz"].astype(np.float64), npz["label"]
    names = json.load((SEG / "names.json").open(encoding="utf-8"))

    # 命名实例 -> id 组(同名 3D 质心 <0.35m 合并)
    by_name: dict[str, list[list]] = {}
    for iid, nm in names.items():
        if not nm or nm == "物品台":
            continue
        pts = xyz[label == int(iid)]
        if len(pts) < 20:
            continue
        cen = np.median(pts, axis=0)
        for grp in by_name.setdefault(nm, []):
            if np.linalg.norm(np.median(xyz[np.isin(label, grp)], axis=0) - cen) < 0.35:
                grp.append(int(iid))
                break
        else:
            by_name[nm].append([int(iid)])
    groups = [(nm, ids) for nm, gs in by_name.items() for ids in gs]
    named_pts = xyz[np.isin(label, [i for _, ids in groups for i in ids])]

    # 合成水平相机:物品台 -> 拍摄环方向,人眼高度
    tj = json.load((DATA / "transforms_aligned.json").open())
    ring = []
    for f2 in tj["frames"]:
        cf = np.array(f2["transform_matrix"], dtype=np.float64)
        ring.append(cf[:3, 3])
    ring = np.stack(ring)
    target = np.median(named_pts, axis=0)
    if args.target_z is not None:
        target[2] = args.target_z
    dirv = ring.mean(axis=0)[:2] - target[:2]
    dirv /= np.linalg.norm(dirv) + 1e-9
    a = np.radians(args.yaw)
    dirv = np.array([dirv[0] * np.cos(a) - dirv[1] * np.sin(a),
                     dirv[0] * np.sin(a) + dirv[1] * np.cos(a)])
    eye = np.array([*(target[:2] + dirv * args.dist), args.eye_height])
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0.0, 0.0, 1.0])); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, :3] = np.stack([x, y, z], axis=1); c2w[:3, 3] = eye
    w2c = np.linalg.inv(c2w)

    RW = 1200
    RH = int(round(RW / ASPECT))
    fpx = 0.5 * RW / np.tan(np.radians(args.hfov) / 2)
    K = np.array([[fpx, 0, RW / 2], [0, fpx, RH / 2], [0, 0, 1.0]])

    sys.path.insert(0, str(ROOT / "Eye_Tracker/tools"))
    from gaze_to_world import SplatDepth
    sd = SplatDepth(CKPT)
    full, _ = sd._render(w2c, K, RW, RH)          # RGB+ED
    img = (np.clip(full[..., :3], 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()  # BGR
    depth_full = full[..., 3]

    # 实例点回配到高斯索引
    from scipy.spatial import cKDTree
    means = sd.means.detach().cpu().numpy()
    tree = cKDTree(means)
    torch = sd.torch
    vm = torch.tensor(w2c, dtype=torch.float32, device=sd.dev).unsqueeze(0)
    Kt = torch.tensor(K, dtype=torch.float32, device=sd.dev).unsqueeze(0)

    fscale, th = 0.95, 2
    want = set(l.strip() for l in args.labels.split(",") if l.strip())
    tint = img.astype(np.float32)
    outlines, labels_todo, placed = [], [], []
    drawn = 0
    for gi, (nm, ids) in enumerate(groups):
        pts = xyz[np.isin(label, ids)]
        d, gidx = tree.query(pts, workers=-1)
        gidx = np.unique(gidx[d < 0.01])
        if len(gidx) < 15:
            continue
        gm = means[gidx]
        cen = np.median(gm, axis=0)
        dd = np.linalg.norm(gm - cen, axis=1)
        gidx = gidx[dd < 2.2 * (np.median(dd) + 1e-6)]   # 滤掉分割溢出的高斯
        if len(gidx) < 15:
            continue
        it = torch.tensor(gidx, dtype=torch.long, device=sd.dev)
        out, al, _ = sd.rasterization(
            sd.means[it], sd.quats[it], sd.scales[it], sd.opac[it], sd.colors[it],
            vm, Kt, RW, RH, sh_degree=3, render_mode="RGB+ED", rasterize_mode="classic")
        a2 = al[0, ..., 0].cpu().numpy()
        dsub = out[0, ..., 3].cpu().numpy()
        occl = (a2 > 0.25) & (dsub > depth_full + 0.10)   # 被别的东西挡住
        a2 = np.where(occl, 0.0, a2)
        if a2.max() < 0.4:
            continue
        col = np.array(PALETTE[gi % len(PALETTE)], np.float32)
        w_pix = (args.alpha * np.clip(a2, 0, 1))[..., None]
        tint = tint * (1 - w_pix) + col * w_pix
        mb = (a2 > 0.5).astype(np.uint8)
        ncc, cc, stats, _ = cv2.connectedComponentsWithStats(mb)
        for ci in range(1, ncc):
            if stats[ci, cv2.CC_STAT_AREA] < 60:
                mb[cc == ci] = 0
        if mb.sum() == 0:
            continue
        cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines.append((cnts, PALETTE[gi % len(PALETTE)]))
        en = EN.get(nm, nm)
        if not want or en in want:
            mm = cv2.moments(mb)
            labels_todo.append((en, int(mm["m10"] / mm["m00"]), int(mm["m01"] / mm["m00"])))
        drawn += 1
    band = tint.astype(np.uint8)
    for cnts, col in outlines:
        cv2.drawContours(band, cnts, -1, col, 2, cv2.LINE_AA)
    for en, mx, my in labels_todo:
        (tw, tth), _ = cv2.getTextSize(en, cv2.FONT_HERSHEY_SIMPLEX, fscale, th)
        cands = [(mx - tw // 2, my + tth // 2), (mx - tw // 2, my - 12),
                 (mx - tw // 2, my + tth + 12)]
        pos = None
        for tx, ty in cands:
            tx = int(np.clip(tx, 2, RW - tw - 2)); ty = int(np.clip(ty, tth + 2, RH - 4))
            r = (tx, ty - tth, tx + tw, ty + 4)
            if all(r[2] < p[0] or r[0] > p[2] or r[3] < p[1] or r[1] > p[3] for p in placed):
                pos = (tx, ty); placed.append(r); break
        if pos is None:
            continue
        cv2.putText(band, en, pos, cv2.FONT_HERSHEY_SIMPLEX, fscale, (20, 20, 20), th + 2, cv2.LINE_AA)
        cv2.putText(band, en, pos, cv2.FONT_HERSHEY_SIMPLEX, fscale, (255, 255, 255), th, cv2.LINE_AA)
    print(f"tinted {drawn}/{len(groups)} instances, eye={eye.round(2)}, target={target.round(2)}")
    cv2.imwrite(str(OUT / args.out), band, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("->", OUT / args.out)


if __name__ == "__main__":
    main()
