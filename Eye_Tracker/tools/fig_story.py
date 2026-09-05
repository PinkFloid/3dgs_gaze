#!/usr/bin/env python3
"""fig_story.py -- 一行多格的流程故事板(原始 world.mp4 去畸变抽帧,自己画叠加,不用烧录字幕)。

    conda run -n nerfstudio python Eye_Tracker/tools/fig_story.py

格:1 物品台放大  2 任务(说指令那一刻的第一人称)  3 视线与判定(测得视线点、候选框、选中实例高亮、箭头)
    4 狗在桌边抓取  5 放入纸箱。默认 08_27/005 苹果任务;改 SHOT_* 常量换实例。
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pupil_localizer import load_fisheye, load_tags, scale_K  # noqa: E402
from gaze_live import Localizer  # noqa: E402
from gaze_video import load_instances  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "SceneRebuild"
REC = Path("/home/liuchy/recordings/2026_08_27/005")
SEG = SCENE / "lab_result/segmentation_sam"   # 08-27 demo 用的就是当前图(无水瓶)
OUT = ROOT / "docs/E1_DATA/fig_story"
T_CMD, T_GRASP, T_PLACE = 59.6, 82.25, 92.0     # demo-video seconds (= world_ts[0] + t):说指令 / 臂弯到桌面抓取 / 苹果落进纸箱
BOUND = "苹果粉"                                   # binding event of session 155653: 拿一下这个苹果 -> 苹果粉
GAZE_MANUAL = None                                # (u, v) 去畸变像素:想手动把视线点放到所看物体上时填;None = 用测得视线
# 手工按画面对准的框(去畸变帧 T_CMD 的像素坐标)。地图投影有 ~0.4° 位姿/配准误差,示意图要准就手工对;球M 此刻已被狗取走,不画。
MANUAL_BOXES = {"苹果粉": [1001, 480, 1022, 503], "橘子": [1035, 479, 1055, 491], "苹果红": [1036, 486, 1056, 503],
                "球L": [1032, 497, 1050, 517], "香蕉": [1070, 486, 1100, 505], "白杯1": [1114, 484, 1134, 510],
                "红杯": [1131, 475, 1152, 502], "白杯2": [1153, 487, 1173, 515], "球R": [1131, 503, 1147, 522]}
HERE_BOX = [700, 424, 761, 476]                    # 纸箱子(桌上)在 T_CMD 帧里的位置,格 2 标 "here"
EN = {"球L": "ball L", "球M": "ball M", "球R": "ball R", "苹果粉": "apple 2", "苹果红": "apple 1", "橘子": "orange",
      "香蕉": "banana", "白杯1": "cup 1", "白杯2": "cup 2", "红杯": "red cup", "水瓶": "bottle", "纸箱子": "box", "物品台": "table"}


def frame_at(rec, t_pupil):
    ts = np.load(rec / "world_timestamps.npy")
    i = int(np.clip(np.searchsorted(ts, t_pupil), 0, len(ts) - 1))
    cap = cv2.VideoCapture(str(rec / "world.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, img = cap.read(); cap.release()
    return float(ts[i]), img


def gaze_px(rec, t0, t1, W, H, min_conf=0.6):
    pts = []
    with open(rec / "gaze.pldata", "rb") as f:
        for _t, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            r = msgpack.unpackb(payload, strict_map_key=False)
            if t0 <= float(r["timestamp"]) <= t1 and r.get("confidence", 0) >= min_conf:
                pts.append(r["norm_pos"])
    nx, ny = np.median(np.array(pts, float), axis=0)
    return nx * W, (1 - ny) * H


def main():
    K0, D = load_fisheye(str(SCENE / "Calibration_result/world_camera_calibration.npz"))
    tags, _ = load_tags(str(SCENE / "world_size/tags_world.json"))
    ts = np.load(REC / "world_timestamps.npy"); t_base = float(ts[0])
    t_cmd, frame_cmd = frame_at(REC, t_base + T_CMD)
    H, W = frame_cmd.shape[:2]
    K = scale_K(K0, (1920, 1080), (W, H))
    Kn = K.copy()   # 去畸变后的针孔内参直接用原 K:中心区域保真,边缘裁掉即可
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, (W, H), cv2.CV_16SC2)
    und = lambda im: cv2.remap(im, m1, m2, cv2.INTER_LINEAR)

    loc = Localizer(SimpleNamespace(dictionary="DICT_6X6_250", max_mean_reproj=0.006, max_jump=1.0, ema=0.0), tags)
    T, n = loc.process(t_cmd, frame_cmd, K, D)
    assert T is not None, "no pose at command time"
    print(f"pose from {n} tags at t={T_CMD}s")
    u, v = gaze_px(REC, t_base + T_CMD - 0.4, t_base + T_CMD + 0.4, W, H)
    gu, gv = cv2.fisheye.undistortPoints(np.array([[[u, v]]], np.float64), K, D, P=Kn).reshape(2)
    if GAZE_MANUAL is not None:
        gu, gv = GAZE_MANUAL
    inst = load_instances(SEG)
    pz = np.load(SEG / "points.npz"); xyz, lab = pz["xyz"].astype(np.float64), pz["label"]
    w2c = np.linalg.inv(T)
    boxes = {}
    for iid, i in inst.items():
        if i["name"] not in EN or i["name"] in ("物品台", "纸箱子"):   # 场所不画框
            continue
        cam = xyz[lab == iid] @ w2c[:3, :3].T + w2c[:3, 3]
        cam = cam[cam[:, 2] > 0.15]
        if len(cam) < 20:
            continue
        px = (Kn @ cam.T).T; px = px[:, :2] / px[:, 2:3]
        x0, y0 = np.percentile(px, 5, axis=0); x1, y1 = np.percentile(px, 95, axis=0)   # 紧框:去掉离群点
        b = boxes.setdefault(i["name"], [x0, y0, x1, y1])
        boxes[i["name"]] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
    if MANUAL_BOXES:
        boxes = {k: list(v) for k, v in MANUAL_BOXES.items()}
    print("boxes:", {EN[k]: [round(float(x)) for x in v] for k, v in boxes.items()})

    img_cmd = und(frame_cmd)
    _, f_grasp = frame_at(REC, t_base + T_GRASP); img_grasp = und(f_grasp)
    _, f_place = frame_at(REC, t_base + T_PLACE); img_place = und(f_place)
    apple = next(i for i in inst.values() if i["name"] == BOUND)
    dist = float(np.linalg.norm(apple["corners"].mean(axis=0) - T[:3, 3]))
    print(f"frame {W}x{H}; user->{BOUND} at command time: {dist:.2f} m; gaze px ({gu:.0f},{gv:.0f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyArrowPatch
    import matplotlib.patheffects as pe
    stroke = [pe.withStroke(linewidth=1.4, foreground="black")]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.5), gridspec_kw=dict(wspace=0.04))
    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    tb = np.array(list(boxes.values()))                                     # 台上实例的联合框
    tcx, tcy = (tb[:, 0].min() + tb[:, 2].max()) / 2, (tb[:, 1].min() + tb[:, 3].max()) / 2
    thw = (tb[:, 2].max() - tb[:, 0].min()) * 0.62 + 14

    def crop(ax, im, cx, cy, hw, aspect=4 / 3):
        hh = hw / aspect
        x0, x1 = int(round(cx - hw)), int(round(cx + hw)); y0, y1 = int(round(cy - hh)), int(round(cy + hh))
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, im.shape[1]), min(y1, im.shape[0])
        ax.imshow(rgb(im[y0:y1, x0:x1]), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)   # 锁定,免得补丁撑开坐标
        return x0, y0

    crop(axes[0], img_cmd, tcx, tcy - 2, thw * 0.85)                             # 1 台面放大
    x0, y0 = crop(axes[1], img_cmd, W / 2, H / 2 - 20, min(W / 2, H / 2 * 4 / 3) - 40)   # 2 说指令时的第一人称
    hb = HERE_BOX
    axes[1].add_patch(Rectangle((hb[0] - x0, hb[1] - y0), hb[2] - hb[0], hb[3] - hb[1], fill=False, ec="#ff9f1a", lw=1.2))
    axes[1].text(hb[0] - x0 - 8, (hb[1] + hb[3]) / 2 - y0, "here", color="#ff9f1a", fontsize=6.5, ha="right", va="center", fontweight="bold", path_effects=stroke)
    ab = boxes[BOUND]
    axes[1].add_patch(Rectangle((ab[0] - x0 - 3, ab[1] - y0 - 3), ab[2] - ab[0] + 6, ab[3] - ab[1] + 6, fill=False, ec="#2bd12b", lw=1.0))
    axes[1].text(ab[0] - x0 - 2, ab[1] - y0 - 10, "this apple", color="#2bd12b", fontsize=6.5, ha="left", va="bottom", fontweight="bold", path_effects=stroke)
    x0, y0 = crop(axes[2], img_cmd, tcx, tcy - 2, thw * 0.85)                    # 3 视线与判定(与 1 同一裁剪)
    for name, (bx0, by0, bx1, by1) in boxes.items():
        sel = name == BOUND
        axes[2].add_patch(Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False,
                                    ec="#2bd12b" if sel else "white", lw=1.5 if sel else 0.7, alpha=1.0 if sel else 0.9))
    bx0, by0, bx1, by1 = boxes[BOUND]
    axes[2].plot(gu - x0, gv - y0, marker="+", ms=9, mew=1.6, color="#ffdd00")
    axes[2].add_patch(FancyArrowPatch((gu - x0, gv - y0), ((bx0 + bx1) / 2 - x0, (by0 + by1) / 2 - y0),
                                      arrowstyle="-|>", mutation_scale=8, lw=1.2, color="#ffdd00", shrinkA=4, shrinkB=2))
    axes[2].text(bx0 - x0, by0 - y0 - 2, EN[BOUND], color="#2bd12b", fontsize=6, ha="left", va="bottom", fontweight="bold", path_effects=stroke)
    crop(axes[3], img_grasp, 1225, 530, 200)                                     # 4 臂弯到桌面抓取
    crop(axes[4], img_place, 1080, 470, 210)                                     # 5 苹果落进纸箱
    caps = ["(1) objects on the table", "(2) \u201cpick up this apple,\nput it here\u201d", "(3) measured gaze (+),\nbound instance (green)",
            "(4) robot grasps the instance", "(5) robot drops it into the box"]
    for ax, c in zip(axes, caps):
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#999999")
        ax.text(0.5, -0.05, c, transform=ax.transAxes, ha="center", va="top", fontsize=6.3, linespacing=1.25)
    fig.savefig(str(OUT) + ".png", dpi=300, bbox_inches="tight"); fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
