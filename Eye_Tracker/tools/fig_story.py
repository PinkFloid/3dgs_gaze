#!/usr/bin/env python3
"""fig_story.py -- 流程故事板(原始 world.mp4 去畸变抽帧,自己画叠加,不用烧录字幕)。

    conda run -n nerfstudio python Eye_Tracker/tools/fig_story.py                  # 默认 --story merged:2 行 x 5 列 -> fig_story
    conda run -n nerfstudio python Eye_Tracker/tools/fig_story.py --story hand     # 单行六格 -> fig_story_hand
    conda run -n nerfstudio python Eye_Tracker/tools/fig_story.py --story box      # 单行五格 -> fig_story_box

故事:
  hand  08-20/010(session 20260820-232900)"拿一下这个"→ 苹果粉 →"那给我":台面 / 指令 / 视线+绑定 / 抬离桌面 / 举回 / 递到手上
  box   08-27/005(session 20260827-155653)"拿一下这个苹果"→"放到这里":台面 / 指令(here=纸箱子)/ 视线+绑定 / 抓取 / 落进纸箱
  merged 上排 hand(去掉"举回"格)、下排 box,各 5 格
格 1 与格 3 用同一裁剪;格 3 的框是手工按画面对准的(地图投影有 ~0.4° 位姿/配准误差,示意图要准就手工对)。
视线点默认取绑定那段注视的 gaze 中位;gaze_manual 可手动指定。
"""
import argparse
import sys
from pathlib import Path

import cv2
import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pupil_localizer import load_fisheye, scale_K  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "SceneRebuild"
EN = {"球L": "ball L", "球M": "ball M", "球R": "ball R", "苹果粉": "apple 2", "苹果红": "apple 1", "橘子": "orange",
      "香蕉": "banana", "白杯1": "cup 1", "白杯2": "cup 2", "红杯": "red cup", "水瓶": "bottle", "纸箱子": "box", "物品台": "table"}

STORIES = {
    "hand": dict(
        rec=Path("/home/liuchy/recordings/2026_08_20/010"), out=ROOT / "docs/E1_DATA/fig_story_hand",
        t_cmd=62.2, gaze_win=(58.8, 65.6),           # binding 苹果粉 的注视段(vote 0.88);t 为视频秒 = pupil t - world_ts[0]
        bound="苹果粉", gaze_manual=None,
        boxes={"苹果粉": [975, 452, 993, 477], "苹果红": [1008, 455, 1026, 472], "球L": [1015, 469, 1033, 486],
               "水瓶": [1041, 428, 1059, 468], "白杯1": [1072, 448, 1088, 472], "红杯": [1077, 440, 1098, 466],
               "白杯2": [1107, 447, 1126, 470], "球R": [1094, 464, 1111, 483]},   # 橘子/香蕉被前排挡住、球M 已被取走:不画
        here=None, this_label="this", this_dy=75,
        shots=[(94.25, (1200, 445, 300), "robot grasps the instance"),
               (96.25, (1050, 640, 450), "carries it back"),
               (98.25, (1000, 800, 400), "“give it to me”:\nhands it over")],
        cap_task="“pick this up”\n(no object name, gaze only)"),
    "box": dict(
        rec=Path("/home/liuchy/recordings/2026_08_27/005"), out=ROOT / "docs/E1_DATA/fig_story_box",
        t_cmd=59.6, gaze_win=(59.2, 60.0),
        bound="苹果粉", gaze_manual=None,
        boxes={"苹果粉": [1001, 480, 1022, 503], "橘子": [1035, 479, 1055, 491], "苹果红": [1036, 486, 1056, 503],
               "球L": [1032, 497, 1050, 517], "香蕉": [1070, 486, 1100, 505], "白杯1": [1114, 484, 1134, 510],
               "红杯": [1131, 475, 1152, 502], "白杯2": [1153, 487, 1173, 515], "球R": [1131, 503, 1147, 522]},   # 球M 此刻已被取走
        here=[700, 424, 761, 476], this_label="this apple", this_dy=75,
        shots=[(82.25, (1225, 530, 200), "robot grasps the instance"),
               (92.0, (1080, 470, 210), "robot drops it into the box")],
        cap_task="“pick up this apple,\nput it here”"),
}


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


def load_story(S):
    """抽帧、去畸变、视线像素;返回画一行所需的一切。"""
    REC = S["rec"]
    K0, D = load_fisheye(str(SCENE / "Calibration_result/world_camera_calibration.npz"))
    ts = np.load(REC / "world_timestamps.npy"); t_base = float(ts[0])
    _, frame_cmd = frame_at(REC, t_base + S["t_cmd"])
    H, W = frame_cmd.shape[:2]
    K = scale_K(K0, (1920, 1080), (W, H))
    Kn = K.copy()   # 去畸变后的针孔内参直接用原 K:中心区域保真,边缘裁掉即可
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Kn, (W, H), cv2.CV_16SC2)
    und = lambda im: cv2.remap(im, m1, m2, cv2.INTER_LINEAR)
    u, v = gaze_px(REC, t_base + S["gaze_win"][0], t_base + S["gaze_win"][1], W, H)
    gu, gv = cv2.fisheye.undistortPoints(np.array([[[u, v]]], np.float64), K, D, P=Kn).reshape(2)
    if S["gaze_manual"] is not None:
        gu, gv = S["gaze_manual"]
    boxes = {k: list(v) for k, v in S["boxes"].items()}
    print(f"[{REC.name}] frame {W}x{H}; gaze px ({gu:.0f},{gv:.0f}); boxes {len(boxes)}")
    shots = [(und(frame_at(REC, t_base + t)[1]), crop, cap) for t, crop, cap in S["shots"]]
    return dict(img_cmd=und(frame_cmd), W=W, H=H, gaze=(gu, gv), boxes=boxes, shots=shots)


def draw_story(S, axes, shot_idx=None):
    """把一个故事画进一行 axes:格 1 台面、格 2 指令、格 3 视线+绑定、其余为机器人执行。"""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    import matplotlib.patheffects as pe
    stroke = [pe.withStroke(linewidth=1.4, foreground="black")]
    L = load_story(S)
    img_cmd, W, H, (gu, gv), boxes, shots, BOUND = L["img_cmd"], L["W"], L["H"], L["gaze"], L["boxes"], L["shots"], S["bound"]
    if shot_idx is not None:
        shots = [shots[i] for i in shot_idx]
    assert len(axes) == 3 + len(shots), (len(axes), len(shots))
    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    tb = np.array(list(boxes.values()))                                     # 台上实例的联合框
    tcx, tcy = (tb[:, 0].min() + tb[:, 2].max()) / 2, (tb[:, 1].min() + tb[:, 3].max()) / 2
    thw = ((tb[:, 2].max() - tb[:, 0].min()) * 0.62 + 14) * 0.85

    def crop(ax, im, cx, cy, hw, aspect=4 / 3):
        hh = hw / aspect
        x0, x1 = int(round(cx - hw)), int(round(cx + hw)); y0, y1 = int(round(cy - hh)), int(round(cy + hh))
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, im.shape[1]), min(y1, im.shape[0])
        ax.imshow(rgb(im[y0:y1, x0:x1]), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)   # 锁定,免得补丁撑开坐标
        return x0, y0

    # 1 台面放大
    crop(axes[0], img_cmd, tcx, tcy - 2, thw)
    # 2 说指令时的第一人称:标 this(以及 here)
    x0, y0 = crop(axes[1], img_cmd, W / 2, H / 2 - 20, min(W / 2, H / 2 * 4 / 3) - 40)
    ab = boxes[BOUND]
    axes[1].add_patch(Rectangle((ab[0] - x0 - 3, ab[1] - y0 - 3), ab[2] - ab[0] + 6, ab[3] - ab[1] + 6, fill=False, ec="#2bd12b", lw=1.0))
    lx, ly = (ab[0] + ab[2]) / 2 - x0, ab[3] - y0 + S["this_dy"]                     # 标签放到台面下方的空处,细引线指上去
    axes[1].plot([lx, lx], [ab[3] - y0 + 4, ly - 9], color="#2bd12b", lw=0.7)
    axes[1].text(lx, ly, S["this_label"], color="#2bd12b", fontsize=6.5, ha="center", va="top", fontweight="bold", path_effects=stroke)
    if S["here"]:
        hb = S["here"]
        axes[1].add_patch(Rectangle((hb[0] - x0, hb[1] - y0), hb[2] - hb[0], hb[3] - hb[1], fill=False, ec="#ff9f1a", lw=1.2))
        axes[1].text(hb[0] - x0 - 8, (hb[1] + hb[3]) / 2 - y0, "here", color="#ff9f1a", fontsize=6.5, ha="right", va="center", fontweight="bold", path_effects=stroke)
    # 3 视线与判定(与 1 同一裁剪)
    x0, y0 = crop(axes[2], img_cmd, tcx, tcy - 2, thw)
    for name, (bx0, by0, bx1, by1) in boxes.items():
        sel = name == BOUND
        axes[2].add_patch(Rectangle((bx0 - x0, by0 - y0), bx1 - bx0, by1 - by0, fill=False,
                                    ec="#2bd12b" if sel else "white", lw=1.5 if sel else 0.7, alpha=1.0 if sel else 0.9))
    bx0, by0, bx1, by1 = boxes[BOUND]
    axes[2].plot(gu - x0, gv - y0, marker="+", ms=9, mew=1.6, color="#ffdd00")
    axes[2].add_patch(FancyArrowPatch((gu - x0, gv - y0), ((bx0 + bx1) / 2 - x0, (by0 + by1) / 2 - y0),
                                      arrowstyle="-|>", mutation_scale=8, lw=1.2, color="#ffdd00", shrinkA=4, shrinkB=2))
    axes[2].text(bx0 - x0, by0 - y0 - 2, EN[BOUND], color="#2bd12b", fontsize=6, ha="left", va="bottom", fontweight="bold", path_effects=stroke)
    # 4.. 机器人执行
    for ax, (im, (cx, cy, hw), _cap) in zip(axes[3:], shots):
        crop(ax, im, cx, cy, hw)
    caps = ["objects on the table", S["cap_task"], "measured gaze (+),\nbound instance (green)"] + [c for _, _, c in shots]
    caps = [f"({k + 1}) {c}" for k, c in enumerate(caps)]
    n = len(axes)
    for ax, c in zip(axes, caps):
        for sp in ax.spines.values():
            sp.set_linewidth(0.5); sp.set_color("#999999")
        ax.text(0.5, -0.05, c, transform=ax.transAxes, ha="center", va="top", fontsize=6.3 if n <= 5 else 5.9, linespacing=1.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", choices=sorted(STORIES) + ["merged"], default="merged")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
                         "font.size": 7, "pdf.fonttype": 42})
    if args.story == "merged":
        OUT = ROOT / "docs/E1_DATA/fig_story"
        # 面板宽 = 7.16*0.98/(5+4*0.04);行高按 4:3 定,hspace 只留说明文字两行 + 行标
        fig, axes = plt.subplots(2, 5, figsize=(7.16, 2.95), gridspec_kw=dict(wspace=0.04, hspace=0.47, left=0.01, right=0.99, top=0.95, bottom=0.09))
        draw_story(STORIES["hand"], axes[0], shot_idx=[0, 2])      # 上排:去掉"举回"格,递到手上收尾
        draw_story(STORIES["box"], axes[1])                        # 下排:纸箱版
        for ax, lab in zip(axes[:, 0], ("(a)", "(b)")):
            ax.text(0.0, 1.03, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, fontweight="bold")
    else:
        S = STORIES[args.story]; OUT = S["out"]
        n = 3 + len(S["shots"])
        fig, axes = plt.subplots(1, n, figsize=(7.16, 1.5 * 5 / n + 0.35), gridspec_kw=dict(wspace=0.04))
        draw_story(S, axes)
    fig.savefig(str(OUT) + ".png", dpi=300, bbox_inches="tight"); fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
