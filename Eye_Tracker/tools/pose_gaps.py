#!/usr/bin/env python3
"""pose_gaps.py -- 定位更新间隔与旧姿态复用时长:对一条录像重跑 tag 定位(不渲染、不投票),输出
逐帧定位成败、姿态更新间隔分布、每个 gaze 样本用到的姿态"新鲜度"(插值/沿用旧姿态多久/无姿态),
以及按目标时段(card_windows 产物)统计每项的定位覆盖——用于检查行走录像的错误是否集中在定位中断期。

    conda run -n nerfstudio python Eye_Tracker/tools/pose_gaps.py 2026_08_25/u3 --era v9 \
        [--windows docs/E1_DATA/audit_0906/windows/08_25_u3.json] [--out docs/E1_DATA/audit_0906]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaze_live import Localizer, RollingPoses  # noqa: E402
from pupil_localizer import load_fisheye, load_tags, recording_frames, scale_K  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "SceneRebuild"
R = Path("/home/liuchy/recordings")
TAGS = {"v7": SCENE / "archive_envs/v7/tags_world.json", "v8": SCENE / "archive_envs/v8/tags_world.json",
        "v9": SCENE / "world_size/tags_world.json"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rec")
    ap.add_argument("--era", required=True, choices=["v7", "v8", "v9"])
    ap.add_argument("--windows", default=None)
    ap.add_argument("--out", default=str(ROOT / "docs/E1_DATA/audit_0906"))
    ap.add_argument("--sample-hz", type=float, default=20.0)
    a = ap.parse_args()
    rd = R / a.rec
    K_calib, D = load_fisheye(str(SCENE / "Calibration_result/world_camera_calibration.npz"))
    tags, _ = load_tags(str(TAGS[a.era]))
    args = SimpleNamespace(dictionary="DICT_6X6_250", max_mean_reproj=0.006, max_jump=1.0, ema=0.3)
    loc = Localizer(args, tags)
    frames = []  # (t, ok, n_tags)
    K = None
    for t, img in recording_frames(rd):
        if K is None:
            K = scale_K(K_calib, (1920, 1080), (img.shape[1], img.shape[0]))
        pose, n_tags = loc.process(t, img, K, D)
        frames.append((float(t), pose is not None, int(n_tags), pose[:3, 3].tolist() if pose is not None else None))
    ft = np.array([f[0] for f in frames])
    fok = np.array([f[1] for f in frames])
    t_ok = ft[fok]
    dts = np.diff(t_ok)
    # gaze 样本的姿态新鲜度(与 gaze_live 同一 RollingPoses 语义:间隔≤1s 插值;否则 1s 内取最近;再远无姿态)
    poses = RollingPoses()
    gz = []
    with open(rd / "gaze.pldata", "rb") as f:
        for _t, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            r = msgpack.unpackb(payload, strict_map_key=False)
            if r.get("confidence", 0) >= 0.6:
                gz.append(float(r["timestamp"]))
    gz.sort()
    status = []  # (t, kind, age)
    last_proc = -1e9
    fi = 0
    for tg in gz:
        while fi < len(frames) and frames[fi][0] <= tg:
            if frames[fi][1]:
                poses.push(frames[fi][0], np.eye(4))
            fi += 1
        if tg - last_proc < 1.0 / a.sample_hz:
            continue
        last_proc = tg
        buf = poses.buf
        if not buf:
            status.append((tg, "none", None))
            continue
        ts = [b[0] for b in buf]
        i = int(np.searchsorted(ts, tg))
        if i == len(ts):
            age = tg - ts[-1]
            status.append((tg, "hold" if age <= 1.0 else "none", age if age <= 1.0 else None))
        elif i == 0:
            status.append((tg, "hold" if ts[0] - tg <= 1.0 else "none", None))
        else:
            gap = ts[i] - ts[i - 1]
            if gap <= 1.0:
                status.append((tg, "interp", 0.0))
            else:
                d0, d1 = tg - ts[i - 1], ts[i] - tg
                if min(d0, d1) > 1.0:
                    status.append((tg, "none", None))
                else:
                    status.append((tg, "hold", min(d0, d1)))
    kinds = np.array([s[1] for s in status])
    ages = np.array([s[2] if s[2] is not None else np.nan for s in status], float)
    summary = {"rec": a.rec, "era": a.era, "n_frames": len(frames), "n_localized": int(fok.sum()),
               "loc_rate": round(float(fok.mean()), 3),
               "pose_dt_median": round(float(np.median(dts)), 3) if len(dts) else None,
               "pose_dt_p95": round(float(np.percentile(dts, 95)), 3) if len(dts) else None,
               "pose_dt_max": round(float(dts.max()), 3) if len(dts) else None,
               "gaps_gt_0.5s": int((dts > 0.5).sum()), "gaps_gt_1s": int((dts > 1.0).sum()),
               "time_in_gaps_gt_1s": round(float(dts[dts > 1.0].sum()), 2),
               "gaze_samples_20hz": len(status),
               "gaze_interp": int((kinds == "interp").sum()), "gaze_hold": int((kinds == "hold").sum()),
               "gaze_none": int((kinds == "none").sum()),
               "hold_age_median": round(float(np.nanmedian(ages[kinds == "hold"])), 3) if (kinds == "hold").any() else None,
               "hold_age_p95": round(float(np.nanpercentile(ages[kinds == "hold"], 95)), 3) if (kinds == "hold").any() else None}
    per_item = []
    if a.windows:
        doc = json.loads(Path(a.windows).read_text(encoding="utf-8"))
        for it in doc["items"]:
            w0, w1 = it["win_start_abs"], it["win_end_abs"]
            m = (ft >= w0) & (ft <= w1)
            gm = np.array([(w0 <= s[0] <= w1) for s in status])
            g_ok = int(((kinds != "none") & gm).sum())
            g_all = int(gm.sum())
            # 窗内最长无定位空档
            tt = np.concatenate([[w0], t_ok[(t_ok >= w0) & (t_ok <= w1)], [w1]])
            per_item.append({"k": it["k"], "target": it["target"], "loc_frames": f"{int((fok & m).sum())}/{int(m.sum())}",
                             "loc_rate": round(float((fok & m).sum() / max(m.sum(), 1)), 2),
                             "gaze_with_pose": f"{g_ok}/{g_all}", "max_gap_s": round(float(np.diff(tt).max()), 2),
                             "hold_frac": round(float(((kinds == "hold") & gm).sum() / max(g_all, 1)), 2)})
    out = Path(a.out) / "pose"
    out.mkdir(parents=True, exist_ok=True)
    tag = a.rec.replace("/", "_")[5:]
    (out / f"{tag}.json").write_text(json.dumps({"summary": summary, "per_item": per_item,
                                                  "frames": [(round(f[0], 3), f[1], f[2]) for f in frames]},
                                                 ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in per_item:
        print(r)
    print(f"-> {out / (tag + '.json')}")


if __name__ == "__main__":
    main()
