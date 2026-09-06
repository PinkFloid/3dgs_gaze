#!/usr/bin/env python3
"""fig_scene_map.py -- 论文"实验场景及其地图表示"三联图:真实帧 | 同位姿 3DGS 渲染 | 实例分割地图(逐实例着色),
下排为同一 ROI(桌面同类物体)的放大。位姿来自该帧的 tag PnP(与 verify_pose_render 同一链路),三张图像素对齐。

    conda run -n nerfstudio python Eye_Tracker/tools/fig_scene_map.py --recording ~/recordings/2026_08_20/000 --frame 230 \
        --seg-dir SceneRebuild/archive_envs/v9_rec --ckpt SceneRebuild/lab_result/splatfacto/2026-08-20_201525/nerfstudio_models/step-000029999.ckpt

产物:<out-dir>/scene_map_f<N>.png/.pdf(合成),panel_real/render/instances_f<N>.png 与 zoom_*.png(单图)。
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaze_to_world import SplatDepth  # noqa: E402
from pupil_localizer import load_fisheye, load_tags, scale_K, solve_pose  # noqa: E402
from verify_pose_render import detect  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "SceneRebuild"
EN = {"球L": "ball_L", "球M": "ball_M", "球R": "ball_R", "白杯1": "cup_1", "白杯2": "cup_2", "红杯": "red_cup",
      "水瓶": "bottle", "苹果粉": "apple_1", "苹果红": "apple_2", "橘子": "orange", "香蕉": "banana",
      "纸箱子": "box", "物品台": "cart"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recording", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tags", default=str(SCENE / "world_size/tags_world.json"))
    ap.add_argument("--calib", default=str(SCENE / "Calibration_result/world_camera_calibration.npz"))
    ap.add_argument("--pinhole-scale", type=float, default=0.7)
    ap.add_argument("--zoom-names", default="球L,球M,球R,白杯1,白杯2,红杯,水瓶,苹果粉,苹果红,橘子,香蕉",
                    help="放大框覆盖这些命名实例的投影")
    ap.add_argument("--zoom-margin", type=float, default=0.25)
    ap.add_argument("--label-zoom", action="store_true", help="放大图的实例面板上标英文名")
    ap.add_argument("--label-names", default="球L,球M,球R,白杯1,白杯2", help="只给这些实例标名(同类对)")
    ap.add_argument("--out-dir", default=str(ROOT / "paper/fig_scene"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = Path(a.recording).expanduser()

    # ---- 帧 + 位姿(同 verify_pose_render)
    K_calib, D = load_fisheye(a.calib)
    tags, _ = load_tags(a.tags)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
                                       cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = scale_K(K_calib, (1920, 1080), (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {a.frame}")
    quads, ids = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), detector, tags)
    if not ids:
        raise SystemExit("no surveyed tags in this frame")
    obj = np.concatenate([tags[i] for i in ids])
    px = np.concatenate(quads)
    pts_norm = cv2.fisheye.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    T, n_inl, reproj = solve_pose(obj, pts_norm, 0.01)
    print(f"frame {a.frame}: tags {sorted(ids)}, pos {np.round(T[:3, 3], 3)}, inliers {n_inl}/{len(obj)}")
    K_new = np.array([[K[0, 0] * a.pinhole_scale, 0, W / 2.0], [0, K[1, 1] * a.pinhole_scale, H / 2.0], [0, 0, 1.0]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2)
    real = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

    # ---- 3DGS 渲染 + 实例着色渲染
    splat = SplatDepth(Path(a.ckpt))
    render = cv2.cvtColor(splat.render_view(T, K_new, W, H), cv2.COLOR_RGB2BGR)
    seg = Path(a.seg_dir)
    z = np.load(seg / "points.npz")
    xyz, label = z["xyz"], z["label"]
    meta = json.loads((seg / "instances.json").read_text(encoding="utf-8"))
    names = json.loads((seg / "names.json").read_text(encoding="utf-8"))
    bg = {int(k): v for k, v in meta.get("background", {}).items()} or {0: "floor", 1: "ceiling", 2: "wall", 3: "wall", 4: "wall", 5: "wall"}
    from scipy.spatial import cKDTree
    means = splat.means.detach().cpu().numpy()
    d, idx = cKDTree(means).query(xyz, k=1)
    assert float(d.max()) < 1e-6, "points.npz 与 ckpt 高斯不对应"
    torch = splat.torch
    colors = np.full((len(means), 3), 0.86, np.float32)          # 未标注碎片:浅灰
    labs = np.unique(label)
    inst_labs = [int(l) for l in labs if int(l) not in bg]
    # 同名实例共色;命名实例饱和,无名实例淡一档
    name_of = {int(l): (names.get(str(int(l))) or "") for l in labs}
    groups = {}
    for l in inst_labs:
        groups.setdefault(name_of[l] or f"#{l}", []).append(l)
    keys = sorted(groups, key=lambda k: (k.startswith("#"), k))
    for r, k in enumerate(keys):
        hue = (r * 0.618033988749895) % 1.0
        named = not k.startswith("#")
        rgb = colorsys.hsv_to_rgb(hue, 0.92 if named else 0.55, 0.95 if named else 0.85)
        for l in groups[k]:
            colors[idx[label == l]] = rgb
    for l, nm in bg.items():
        colors[idx[label == l]] = {"floor": 0.72, "wall": 0.80, "ceiling": 0.84}.get(nm, 0.8)
    w2c = np.linalg.inv(T)
    vm = torch.tensor(w2c, dtype=torch.float32, device=splat.dev).unsqueeze(0)
    Kt = torch.tensor(K_new, dtype=torch.float32, device=splat.dev).unsqueeze(0)
    col_t = torch.tensor(colors, device=splat.dev)
    img, alpha, _ = splat.rasterization(splat.means, splat.quats, splat.scales, splat.opac, col_t, vm, Kt, W, H,
                                        sh_degree=None, render_mode="RGB", rasterize_mode="classic",
                                        backgrounds=torch.ones(1, 3, device=splat.dev))
    inst = (np.clip(img[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()

    # ---- ROI:命名实例投影包围框
    cents = {}
    inst_by_id = {int(r["id"]): r for r in meta["instances"]}
    for l, nm in name_of.items():
        if nm and l in inst_by_id:
            cents.setdefault(nm, []).append(np.asarray(inst_by_id[l]["centroid"], float))
    want = [n for n in a.zoom_names.split(",") if n in cents]
    pts = np.array([c for n in want for c in cents[n]])
    rvec, _ = cv2.Rodrigues(w2c[:3, :3])
    proj, _ = cv2.projectPoints(pts, rvec, w2c[:3, 3], K_new, None)
    proj = proj.reshape(-1, 2)
    x0, y0 = proj.min(0)
    x1, y1 = proj.max(0)
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * (1 + 2 * a.zoom_margin), (y1 - y0) * (1 + 2 * a.zoom_margin)
    bw = max(bw, bh * 16 / 9)
    bh = max(bh, bw * 9 / 16)
    X0, Y0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
    X1, Y1 = int(min(W, cx + bw / 2)), int(min(H, cy + bh / 2))
    print(f"zoom ROI x {X0}-{X1} y {Y0}-{Y1}")
    zooms = []
    for name, im in (("real", real), ("render", render), ("instances", inst)):
        cv2.imwrite(str(out / f"panel_{name}_f{a.frame}.png"), im)
        zc = im[Y0:Y1, X0:X1].copy()
        zc = cv2.resize(zc, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        if name == "instances" and a.label_zoom:
            # 同类各占一行:球标统一放在球排下方一行,杯标统一放在杯排上方一行,标签与物体用短引线相连
            lab_pts = []
            for n in [x for x in a.label_names.split(",") if x in cents]:
                c = np.mean(cents[n], axis=0)
                p, _ = cv2.projectPoints(c[None], rvec, w2c[:3, 3], K_new, None)
                u, v = (p.reshape(2) - [X0, Y0]) * 3
                lab_pts.append((n, float(u), float(v)))
            rows = {}
            for cls, dy in (("球", 58), ("杯", -46)):
                vs = [v for n, u, v in lab_pts if cls in n]
                if vs:
                    rows[cls] = (max(vs) if dy > 0 else min(vs)) + dy
            for n, u, v in lab_pts:
                cls = "球" if n.startswith("球") else "杯"
                if cls not in rows:
                    continue
                txt = EN.get(n, n)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                ox, oy = int(u - tw / 2), int(rows[cls])
                cv2.line(zc, (int(u), int(v)), (int(u), oy - th // 2), (0, 0, 0), 2)
                cv2.rectangle(zc, (ox - 4, oy - th - 4), (ox + tw + 4, oy + 6), (0, 0, 0), -1)
                cv2.putText(zc, txt, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"zoom_{name}_f{a.frame}.png"), zc)
        zooms.append(zc)

    # ---- 合成
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    zh = (Y1 - Y0) / (X1 - X0) * W   # 放大行按与上排同宽时的高度配比
    fig, axes = plt.subplots(2, 3, figsize=(18, 6 * (H + zh) / W + 0.8), dpi=150,
                             gridspec_kw={"height_ratios": [H, zh]}, constrained_layout=True)
    titles = ["(a) real scene (world camera, undistorted)", "(b) 3DGS rendering, same pose", "(c) instance map (one color per instance)"]
    for ax, im, t in zip(axes[0], (real, render, inst), titles):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec="#ff2d2d", lw=2))
        ax.set_title(t, fontsize=12)
        ax.axis("off")
    for ax, zc in zip(axes[1], zooms):
        ax.imshow(cv2.cvtColor(zc, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    fig.savefig(out / f"scene_map_f{a.frame}.png")
    fig.savefig(out / f"scene_map_f{a.frame}.pdf")
    print(f"-> {out}/scene_map_f{a.frame}.png/.pdf  ({len(inst_labs)} instances, {len(keys)} colour groups, "
          f"{sum(1 for k in keys if not k.startswith('#'))} named)")


if __name__ == "__main__":
    main()
