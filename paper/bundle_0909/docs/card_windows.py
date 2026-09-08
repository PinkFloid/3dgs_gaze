#!/usr/bin/env python3
"""card_windows.py -- 用卡片口播音频的提示音时刻表 + 原始眼动速度,为每条 E1 录像标出逐项目标时段。

目标时段来自任务记录(口播音频),不来自绑定算法输出:
口播结构(card_caller.render):开场书签(盯 tag)→ 叮 → [ i。物名。→ 盯 2.8s → 叮 → 1.5s ] × n
(第 n//2 项后插"校准"书签 3.4s+叮+1.0s)→ 收尾书签 → 叮。从 docs/e1_audio/<card>.wav 检出全部
1200Hz 叮(160ms),得到每项"报名起点 / 盯看起点 / 叮"的相对时刻;未知量只剩音频在录像时钟上的起点 τ。
τ 的估计(无音轨、无人工标注,只能借助注视时刻):用 eval_e1 的卡序对齐找到每个命中项的注视段,取
"注视段结束时刻 − 该项叮的相对时刻" 的中位数再减 0.25s 反应时(看向别处发生在叮之后)。识别结果只影响这一个
标量;逐项残差的 MAD 作一致性检验(n≥4 且 MAD≤0.8s 记 ok)。原始眼动速度/位移模板拟合在 4m 密排卡上
(视线位移仅 1–5°)多峰不可靠,已弃用。音频版本差异(08-16 17:55 前录的 000/001/s1/s2/s3 用的是无卡中场
校准块的旧版,项内结构 2.8/1.5s 不变)用前后两半各自估计 τ 吸收;一半不可靠时按版本已知的 Δτ 从另一半推得
(旧版 Δ=-(校准块时长 L),新版 Δ=0);两半都不可靠的录像退回卡序对齐窗口(fallback=lcs,逐项窗=注视段本身)。

每项分析窗 = [报名起点, 叮+0.5s];盯看窗 = [叮-2.8, 叮];项间(1.5s 空白+报名)与书签段为"无目标"时段。
校验:Pupil 自带 fixations.pldata(散度法)与本管线 final 流各算一次"盯看窗内有注视"的比例。

    python Eye_Tracker/tools/card_windows.py [--out docs/E1_DATA/audit_0906]
产物:<out>/windows/<rec_tag>.json、<out>/windows_index.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import cv2
import msgpack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_cards import CARDS  # noqa: E402
import eval_e1 as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = Path("/home/liuchy/recordings")
AUDIO = ROOT / "docs/e1_audio"
CALIB = ROOT / "SceneRebuild/Calibration_result/world_camera_calibration.npz"
ENV = {"v7": ROOT / "SceneRebuild/archive_envs/v7", "v8": ROOT / "SceneRebuild/archive_envs/v8",
       "v9": ROOT / "SceneRebuild/archive_envs/v9_rec"}
# rec, card, era, flags(与 collect_e1.RECS / run_e4.sh 同源;u1 为剔除条,一并标注但标 excluded)
OLD_AUDIO = {"2026_08_16/000", "2026_08_16/001", "2026_08_16/s1", "2026_08_16/s2", "2026_08_16/s3"}  # 17:55 前旧版口播(无校准块)
RECS = [("2026_08_16/000", "e1", "v7", ""), ("2026_08_16/001", "e2", "v7", "stress"),
        ("2026_08_16/s1", "s1", "v7", ""), ("2026_08_16/s2", "s2", "v7", ""), ("2026_08_16/s3", "s3", "v7", ""),
        ("2026_08_16/s4", "s4", "v8", ""), ("2026_08_16/s6", "s6", "v8", ""), ("2026_08_18/000", "s7", "v8", ""),
        ("2026_08_20/000", "c1", "v9", ""), ("2026_08_20/001", "c2", "v9", ""),
        ("2026_08_20/002", "c4", "v9", "beyond_occ"), ("2026_08_20/003", "c4", "v9", "beyond_occ"),
        ("2026_08_25/c1_1", "c4", "v9", "beyond_occ"), ("2026_08_25/c1_2", "c4", "v9", ""),
        ("2026_08_25/c1_3", "c4", "v9", ""), ("2026_08_25/u3", "u3", "v9", "walking"),
        ("2026_08_25/u1", "u1", "v9", "excluded")]
STARE, GAP, BEEP = 2.8, 1.5, 0.16
POST = 0.5   # 叮后宽限(眼比耳慢半拍)
OPEN_STARE, CAL_STARE, END_STARE = 5.2, 3.4, 3.2


def rec_tag(rec):
    d, n = rec.split("/")
    return f"{d[5:]}_{n}"


def beeps(wav):
    """1200Hz 提示音起点(秒):Goertzel 能量占比 >0.5 且持续 ≥100ms。"""
    with wave.open(str(wav)) as f:
        sr = f.getframerate()
        x = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(float) / 32768
    hop, win = int(sr * 0.01), int(sr * 0.02)
    n = np.arange(win)
    k = 2 * np.pi * 1200 / sr
    cs, sn = np.cos(k * n), np.sin(k * n)
    t, e = [], []
    for i in range(0, len(x) - win, hop):
        seg = x[i:i + win]
        t.append(i / sr)
        e.append(((seg @ cs) ** 2 + (seg @ sn) ** 2) / ((seg @ seg) + 1e-9) / win * 2)
    t, on = np.array(t), np.array(e) > 0.5
    starts, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= 10:
                starts.append(float(t[i]))
            i = j
        else:
            i += 1
    return starts, len(x) / sr


def schedule(card):
    """卡片音频的相对时刻表:items[k] = {target, cue_start, stare_start, beep, win_start, win_end, half}."""
    seq = CARDS[card][1]
    b, dur = beeps(AUDIO / f"{card}.wav")
    n, mid = len(seq), len(seq) // 2
    if len(b) != n + 3:
        raise SystemExit(f"{card}: 检到 {len(b)} 声叮,期望 {n + 3}(开场+{n}项+校准+收尾)")
    items = []
    for k in range(1, n + 1):
        bi = b[k] if k <= mid else b[k + 1]
        if k == 1:
            prev, gap = b[0], 1.0
        elif k == mid + 1:
            prev, gap = b[mid + 1], 1.0
        else:
            prev, gap = (b[k - 1] if k - 1 <= mid else b[k]), GAP
        cue = prev + BEEP + gap
        items.append({"k": k, "target": seq[k - 1], "cue_start": round(cue, 3),
                      "stare_start": round(bi - STARE, 3), "beep": round(bi, 3),
                      "win_start": round(cue, 3), "win_end": round(bi + POST, 3),
                      "half": "A" if k <= mid else "B"})
    marks = {"open_beep": b[0], "calib_beep": b[mid + 1], "final_beep": b[n + 2],
             "open_stare": [b[0] - OPEN_STARE, b[0]], "calib_stare": [b[mid + 1] - CAL_STARE, b[mid + 1]],
             "final_stare": [b[n + 2] - END_STARE, b[n + 2]], "audio_dur": dur}
    return items, marks, b


def raw_gaze_velocity(rec, K, D, W=1920, H=1080, min_conf=0.6):
    """原始 gaze -> 去畸变视线方向 -> 相邻样本角速度(deg/s),3 点中值滤波。"""
    gz = []
    with open(rec / "gaze.pldata", "rb") as f:
        for _t, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            r = msgpack.unpackb(payload, strict_map_key=False)
            if r.get("confidence", 0) >= min_conf:
                gz.append((float(r["timestamp"]), r["norm_pos"][0], r["norm_pos"][1]))
    gz.sort()
    g = np.array(gz)
    px = np.stack([g[:, 1] * W, (1 - g[:, 2]) * H], -1).reshape(-1, 1, 2).astype(np.float64)
    pn = cv2.fisheye.undistortPoints(px, K, D).reshape(-1, 2)
    v = np.concatenate([pn, np.ones((len(pn), 1))], 1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    t = g[:, 0]
    dt = np.diff(t)
    cosang = np.clip((v[1:] * v[:-1]).sum(1), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    ok = (dt > 1e-4) & (dt < 0.1)
    vel = np.where(ok, ang / np.maximum(dt, 1e-4), np.nan)
    tv = t[1:]
    vm = vel.copy()
    for i in range(1, len(vel) - 1):
        w3 = vel[i - 1:i + 2]
        if np.isfinite(w3).all():
            vm[i] = np.median(w3)
    return tv, vm


def lcs_episode_ends(rd, log, card, era, items, double_dwell=7.0):
    """卡序对齐(eval_e1 口径)得到的每个命中项:{item k: (注视段起点 abs, 注视段结束 abs)}。
    连击对(球L×2)三形态同 eval_e1:融合单段(时长≥double_dwell,两项共用一段,只有末项的结束时刻可当叮的锚)、
    两段分立(紧邻同名,各记各的)、半程(只记第一项)。返回值里 fit 只用 'end_anchor' 为 True 的项。"""
    named = E.load_named(ENV[era])
    seq = E.era_alias(CARDS[card][1], named)
    fin = E.finals(rd / log)
    eps = E.episodes(fin)
    runs = E.runs_of(seq)
    pairs = E.lcs_align([r[0] for r in runs], eps)
    used = set(pairs.values())
    out = {}
    kbase = 0
    for i, (want, k) in enumerate(runs):
        j = pairs.get(i)
        if j is not None:
            if k > 1 and eps[j]["dur"] < double_dwell and j + 1 < len(eps) and j + 1 not in used \
                    and eps[j + 1]["object"] == want:      # 两段分立
                used.add(j + 1)
                out[kbase + 1] = (eps[j]["t_start"], eps[j]["t_end"], True)
                for kk in range(kbase + 1, kbase + k):
                    out[kk + 1] = (eps[j + 1]["t_start"], eps[j + 1]["t_end"], kk == kbase + k - 1)
            else:                                            # 单项 / 融合单段 / 半程
                for kk in range(kbase, kbase + k):
                    out[kk + 1] = (eps[j]["t_start"], eps[j]["t_end"], kk == kbase + k - 1)
        kbase += k
    return out


def fit_half(items, ends, half, react=0.25):
    """τ = median(注视段结束 − 叮相对时刻) − 反应时;返回 (τ, n, MAD, 逐项残差)。"""
    vals = []
    for it in items:
        if it["half"] != half or it["k"] not in ends or not ends[it["k"]][2]:
            continue
        vals.append(ends[it["k"]][1] - it["beep"])
    if not vals:
        return None, 0, None, []
    v = np.array(vals)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med - react, len(v), mad, [round(float(x - med), 2) for x in v]


def pupil_fixations(rec):
    out = []
    with open(rec / "fixations.pldata", "rb") as f:
        for _t, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            r = msgpack.unpackb(payload, strict_map_key=False)
            out.append((float(r["timestamp"]), float(r["timestamp"]) + float(r["duration"]) / 1000.0))
    return out


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs/E1_DATA/audit_0906"))
    ap.add_argument("--log", default="intents_e4_v2s10.jsonl", help="校验用的 final 流(任一配置皆可)")
    ap.add_argument("--only", default="", help="只处理这些 rec(逗号分隔,如 2026_08_25/c1_2)")
    a = ap.parse_args()
    out = Path(a.out) / "windows"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(CALIB, allow_pickle=True)
    K = np.asarray(z["camera_matrix"], np.float64)
    D = np.asarray(z["dist_coeffs"], np.float64).reshape(-1, 1)[:4]
    index = []
    for rec, card, era, flags in RECS:
        if a.only and rec not in a.only.split(","):
            continue
        rd = R / rec
        info = json.loads((rd / "info.player.json").read_text())
        t_rec0 = float(info["start_time_synced_s"])
        items, marks, b = schedule(card)
        mid = len(items) // 2
        L_calib = marks["calib_beep"] - items[mid - 1]["beep"] - 0.5   # 校准块时长(旧版音频无此块)
        old_audio = rec in OLD_AUDIO
        d_version = -L_calib if old_audio else 0.0
        ends = lcs_episode_ends(rd, a.log, card, era, items) if (rd / a.log).exists() else {}
        tauA, nA, madA, resA = fit_half(items, ends, "A")
        tauB, nB, madB, resB = fit_half(items, ends, "B")
        okA = tauA is not None and nA >= 4 and madA <= 0.8
        okB = tauB is not None and nB >= 4 and madB <= 0.8
        method = "lcs_ends"
        if okA and not okB:
            tauB, method = tauA + d_version, "lcs_ends(B from A + version delta)"
        elif okB and not okA:
            tauA, method = tauB - d_version, "lcs_ends(A from B - version delta)"
        elif not okA and not okB:
            method = "fallback_lcs_windows"
        reliable = okA or okB
        sA = cA = sB = cB = 0.0
        # 校验:Pupil 自带注视 / 本管线 final 流
        pf = []  # Pupil 自带 fixations.pldata 全是固定 304ms 段,无校验价值
        fin = E.finals(rd / a.log) if (rd / a.log).exists() else []
        fin_any = []
        if (rd / a.log).exists():
            for ln in (rd / a.log).open(encoding="utf-8"):
                try:
                    e = json.loads(ln)
                except Exception:
                    continue
                if not e.get("provisional"):
                    fin_any.append((e["t_start"], e["t_end"]))
        n_pf = n_fin = 0
        abs_items = []
        for it in items:
            tau = tauA if it["half"] == "A" else tauB
            if method == "fallback_lcs_windows":
                if it["k"] in ends:
                    e0, e1 = ends[it["k"]][0], ends[it["k"]][1]
                    abs_items.append({**it, "tau": None, "cue_start_abs": round(e0 - 0.5, 3),
                                      "stare_start_abs": round(e0, 3), "beep_abs": round(e1, 3),
                                      "win_start_abs": round(e0 - 0.5, 3), "win_end_abs": round(e1 + 0.3, 3),
                                      "window_src": "lcs_episode"})
                else:
                    abs_items.append({**it, "tau": None, "cue_start_abs": None, "stare_start_abs": None,
                                      "beep_abs": None, "win_start_abs": None, "win_end_abs": None,
                                      "window_src": "none"})
                continue
            w0, w1 = tau + it["stare_start"], tau + it["beep"]
            if any(overlap(w0, w1, f0, f1) >= 0.3 for f0, f1 in fin_any):
                n_fin += 1
            abs_items.append({**it, "tau": round(tau, 3), "window_src": "schedule",
                              "cue_start_abs": round(tau + it["cue_start"], 3),
                              "stare_start_abs": round(tau + it["stare_start"], 3),
                              "beep_abs": round(tau + it["beep"], 3),
                              "win_start_abs": round(tau + it["win_start"], 3),
                              "win_end_abs": round(tau + it["win_end"], 3)})
        # LCS(eval_e1 口径)命中段是否落在窗内:识别无关的一致性检查
        n_lcs = n_lcs_in = 0
        if fin:
            named = E.load_named(ENV[era])
            seq = E.era_alias(CARDS[card][1], named)
            eps = E.episodes(fin)
            runs = E.runs_of(seq)
            pairs = E.lcs_align([r[0] for r in runs], eps)
            kbase = 0
            for i, (want, k) in enumerate(runs):
                j = pairs.get(i)
                if j is not None:
                    n_lcs += 1
                    wins = abs_items[kbase:kbase + k]
                    ep = eps[j]
                    if any(w["win_start_abs"] is not None and overlap(ep["t_start"], ep["t_end"], w["win_start_abs"], w["win_end_abs"]) > 0.3 for w in wins):
                        n_lcs_in += 1
                kbase += k
        doc = {"rec": rec, "card": card, "era": era, "flags": flags, "t_rec0": t_rec0,
               "duration_s": info["duration_s"], "audio": f"{card}.wav", "beeps_rel": [round(x, 3) for x in b],
               "method": method, "reliable": reliable, "old_audio": old_audio, "L_calib": round(L_calib, 2),
               "tau_A": (round(tauA, 3) if tauA is not None else None), "tau_B": (round(tauB, 3) if tauB is not None else None),
               "delta_tau": (round(tauB - tauA, 3) if (tauA is not None and tauB is not None) else None),
               "fit_A": {"n": nA, "mad": madA, "resid": resA, "ok": okA}, "fit_B": {"n": nB, "mad": madB, "resid": resB, "ok": okB},
               "marks_abs": ({"open_stare": [round(tauA + marks["open_stare"][0], 3), round(tauA + marks["open_stare"][1], 3)],
                             "calib_stare": [round(tauB + marks["calib_stare"][0], 3), round(tauB + marks["calib_stare"][1], 3)],
                             "final_stare": [round(tauB + marks["final_stare"][0], 3), round(tauB + marks["final_stare"][1], 3)]}
                             if method != "fallback_lcs_windows" else {}),
               "items": abs_items,
               "check": {"n_items": len(items), "pupil_fix_in_stare": n_pf, "pipeline_final_in_stare": n_fin,
                         "lcs_matched": n_lcs, "lcs_matched_in_window": n_lcs_in}}
        (out / f"{rec_tag(rec)}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        row = {"rec": rec, "card": card, "era": era, "flags": flags, "dur_s": round(info["duration_s"], 1),
               "method": method, "old_audio": old_audio,
               "tau_A_rel": (round(tauA - t_rec0, 2) if tauA is not None else ""), "tau_B_rel": (round(tauB - t_rec0, 2) if tauB is not None else ""),
               "delta_tau": (round(tauB - tauA, 2) if (tauA is not None and tauB is not None) else ""),
               "expected_delta": round(d_version, 2), "nA": nA, "madA": madA, "nB": nB, "madB": madB,
               "n_items": len(items), "pipeline_final_in_stare": n_fin, "lcs_in_window": f"{n_lcs_in}/{n_lcs}"}
        index.append(row)
        print(" ".join(f"{k}={v}" for k, v in row.items()))
    with (Path(a.out) / "windows_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    print(f"-> {out}  windows_index.csv")


if __name__ == "__main__":
    main()
