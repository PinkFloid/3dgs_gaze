#!/usr/bin/env python3
"""make_wearer_assets.py -- Fig.2 佩戴端三槽:wearer_pnp / wearer_cluster / speech_wave。

    python paper/fig2_assets/make_wearer_assets.py

真实产物约定(MANIFEST):
- wearer_pnp.jpg   v9 录像鱼眼原帧 + ArUco 检测角点叠加(墙 tag,书签注视段里选检出最多的帧)
- wearer_cluster.png  世界系注视 final 落点散点(按绑定对象着色)+ 一次注视的聚类圈(spread_m)
- speech_wave.png  真实指令音频波形(brain 会话 utt 原始 16k WAV)
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REC = Path("/home/liuchy/recordings/2026_08_20/000")   # c1 综合卡(v9)
SEG = ROOT / "SceneRebuild/lab_result/segmentation_sam"

# ---- wearer_pnp:书签窗(前 14s)里挑墙 tag 检出最多的鱼眼帧 ----
def make_pnp():
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(str(REC / "world.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    best, best_n = None, 0
    for idx in range(int(3 * fps), int(14 * fps), int(fps // 3)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        n = 0 if ids is None else len(ids)
        if n > best_n:
            best, best_n = (f, corners, ids), n
    cap.release()
    f, corners, ids = best
    cv2.aruco.drawDetectedMarkers(f, corners, ids, borderColor=(0, 220, 0))
    for c in corners:                       # 角点加红点,远处小 tag 也看得清
        for x, y in c.reshape(4, 2):
            cv2.circle(f, (int(x), int(y)), 6, (0, 0, 255), -1)
    cv2.imwrite(str(HERE / "wearer_pnp.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wearer_pnp.jpg: {best_n} tags 检出")


# ---- wearer_cluster:final 落点散点 + 单次注视聚类圈 ----
PAL = {"球L": "#7a8a00", "球M": "#a7bb16", "球R": "#cfe22e",
       "白杯1": "#7f8c8d", "白杯2": "#b2babb", "红杯": "#c0392b",
       "苹果粉": "#e6a0b5", "苹果红": "#d0342c", "橘子": "#e67e22",
       "香蕉": "#d4ac0d", "水瓶": "#5dade2", "纸箱子": "#8d6e63"}

def make_cluster():
    finals = [e for e in map(json.loads, open(REC / "intents.jsonl", encoding="utf-8"))
              if not e.get("provisional") and e.get("object") in PAL]
    names = {int(k): v for k, v in
             json.loads((SEG / "names.json").read_text()).items() if v}
    inst = {r["id"]: r for r in
            json.loads((SEG / "instances.json").read_text())["instances"]}
    EN = {"球L": "ball-L", "球M": "ball-M", "球R": "ball-R",
          "白杯1": "cup-w1", "白杯2": "cup-w2", "红杯": "cup-red",
          "苹果粉": "apple-p", "苹果红": "apple-r", "橘子": "orange",
          "香蕉": "banana", "水瓶": "bottle"}
    finals = [e for e in finals if e["object"] != "纸箱子"]   # 箱子远在图外,裁掉保比例
    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=200)   # Fig.2 槽位 1.5:1(左字右图)
    seen = set()
    for e in finals:
        c = e["centroid_world"]
        nm = e["object"]
        ax.scatter(c[0], c[1], s=16, color=PAL[nm], alpha=.8, lw=0,
                   label=EN[nm] if nm not in seen else None)
        seen.add(nm)
    for i, nm in names.items():
        if nm in EN and i in inst:
            c = inst[i]["centroid"]
            ax.scatter(c[0], c[1], marker="x", s=48, color=PAL[nm], lw=1.8)
    balls = [e for e in finals if e["object"].startswith("球")]
    pick = max(balls, key=lambda e: e.get("duration_s", 0))
    c, r = pick["centroid_world"], max(pick.get("spread_m", .02), .015)
    ax.add_patch(plt.Circle((c[0], c[1]), 3 * r, fill=False,
                            color="#222", lw=1.3, ls="--"))
    ax.annotate("one fixation\n(3σ spread)", (c[0] + 3 * r, c[1]),
                textcoords="offset points", xytext=(6, -22), fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(0.55, 1.48); ax.set_ylim(0.12, 0.85)
    ax.set_xlabel("x (m)", fontsize=15); ax.set_ylabel("y (m)", fontsize=15)
    # 标题不画:槽位框自带标题(缩到 3.9cm 宽,字号按比例给足)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=.25)
    ax.legend(fontsize=11, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, handletextpad=.2)
    fig.tight_layout()
    fig.savefig(HERE / "wearer_cluster.png", bbox_inches="tight")
    print(f"wearer_cluster.png: {len(finals)} finals(纸箱子裁掉)")


# ---- speech_wave:真实指令波形 ----
def make_wave():
    # 8/18 放置会话的「把这个放到那里去。」:events 里按 asr 文本对回 utt 文件
    # 全部会话里挑"短而干净"的指令段:asr 是标准指令且整段 <4s(词占满段)
    CMDS = ("拿一下这个", "把这个放到那里去", "放到纸箱子", "拿一下球")
    cand = []
    for sess in sorted((ROOT / "Intension/logs").iterdir(), reverse=True):
        ev = sess / "events.jsonl"
        if not (ev.exists() and (sess / "utt").is_dir()):
            continue
        for ln in ev.open(encoding="utf-8"):
            e = json.loads(ln)
            if e.get("topic") != "asr":
                continue
            txt = e.get("text", "").strip("。 ")
            if not any(txt.startswith(c) for c in CMDS):
                continue
            t = e["t_end_wall"]
            for p in (sess / "utt").glob("utt_*.wav"):
                if abs(float(p.stem.split("_")[1]) - t) < 0.2:
                    dur = wave.open(str(p)).getnframes() / 16000
                    if dur < 4.0:
                        cand.append((dur, p, txt))
    assert cand, "没找到短指令段"
    cand.sort()
    _, pick_p, txt = cand[len(cand) // 2]     # 取中位时长的,太短的可能截音
    print(f"speech_wave 选段:「{txt}」 {pick_p}")
    w = wave.open(str(pick_p))
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16) / 32768.0
    t = np.arange(len(x)) / w.getframerate()
    # 裸波形:无标题无轴(文字由 Fig.2 里的矢量字承担),线加粗、幅值拉满
    fig, ax = plt.subplots(figsize=(10.0, 1.3), dpi=200)
    ax.plot(t, x, lw=1.8, color="#2c3e50")
    amp = max(1e-3, np.abs(x).max())
    ax.set_xlim(0, t[-1]); ax.set_ylim(-1.05 * amp, 1.05 * amp)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(HERE / "speech_wave_bare.png")
    ax.axis("on"); ax.set_yticks([])
    ax.set_xlabel("t (s)", fontsize=13)
    en = {"拿一下这个": "bring me that one", "把这个放到那里去": "put this over there",
          "放到纸箱子": "put it in the box", "拿一下球": "fetch the ball"}
    key = next((v for k, v in en.items() if txt.startswith(k)), txt)
    ax.set_title(f'utterance: "{key}"  (16 kHz)', fontsize=14)
    ax.tick_params(labelsize=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "speech_wave.png")
    print(f"speech_wave.png: {pick_p.name} {t[-1]:.2f}s")


if __name__ == "__main__":
    make_pnp()
    make_cluster()
    make_wave()
