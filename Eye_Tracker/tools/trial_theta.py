#!/usr/bin/env python3
"""trial_theta.py -- 逐 trial 的 θ 与遮挡:用"该项目标时段内的真实头位"重算 θ(结果盲:头位来自 tag 定位流,不来自绑定),
并从该头位向目标圆盘投 13 条细射线算被别的候选物挡住的比例(hidden fraction)。走动/近站时头位随项变化,录像级中位 θ(theta_unit)会把 trial 放错箱——这里逐项纠正。

    conda run --no-capture-output -n nerfstudio python Eye_Tracker/tools/trial_theta.py [--out docs/E1_DATA] [--only 2026_09_07/v6_far,...]

输入:card_windows 的逐项时段(docs/E1_DATA/audit_0906/windows/<tag>.json)、station_theta 的定位流(<rec>/station_poses.jsonl)、
回放日志(老录像 intents_v10cfg.jsonl = 现行管线重放;v10 为 intents.jsonl,同配置)、各代分割档与 ckpt。
每项判定 = 时段内输出时长最多的物体(09-06 口径,不设闸;另给设闸版 gated 列),不用 LCS;头位 = 时段 [盯看起点−0.3, 叮+0.3] 内定位帧中位(没有则取 3 s 内最近帧,再没有退回录像中位,src 列注明);
θ_trial = theta_min(头位, 目标, 全部命名物)(eval_e1 同口径);遮挡 = 从头位向目标质心与顶点各投一条射线,
3DGS 深度首击点最近的命名高斯不是目标且比目标近 → 该射线被挡;两条都挡=1、一条=0.5、都不挡=0(blocker 列记挡它的物体)。
产物:<out>/trials_theta.csv;终端打印 θ_unit vs θ_trial 的分箱对照与"清晰/遮挡"两条曲线。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_cards import CARDS  # noqa: E402
from eval_e1 import era_alias, load_named, theta_min  # noqa: E402
from card_windows import ENV, RECS, rec_tag  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Intension"))
from binding_stats import GATES, canon, gate, load_finals, overlap  # noqa: E402  打分口径与审计报告同源

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "SceneRebuild"
R = Path("/home/liuchy/recordings")
CKPT = {"v7": SCENE / "lab_result/splatfacto/2026-08-15_173942/nerfstudio_models/step-000029999.ckpt",
        "v8": SCENE / "lab_result/splatfacto/2026-08-18_125627/nerfstudio_models/step-000029999.ckpt",
        "v9": SCENE / "lab_result/splatfacto/2026-08-20_201525/nerfstudio_models/step-000029999.ckpt",
        "v10": SCENE / "lab_result/splatfacto/2026-09-06_195439/nerfstudio_models/step-000029999.ckpt",
        "v11": SCENE / "lab_result/splatfacto/2026-09-07_171813/nerfstudio_models/step-000029999.ckpt"}
LOG_OF_ERA = {"v10": "intents.jsonl", "v11": "intents.jsonl"}   # v10/v11:score_card 的回放日志 = 现行配置
LOG_DEFAULT = "intents_v10cfg.jsonl"          # v7/v8/v9:现行管线重放(replay_v10cfg.sh),不混旧配置
BINS = [0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0, 20.0]
PAD, NEAR_S = 0.3, 3.0


def bin_of(th):
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= th < hi:
            return f"{lo:g}-{hi:g}"
    return "<0.5" if th < 0.5 else ">=20"


def window_outcomes(fin, items):
    """打分口径(09-06 用户裁定,与 binding_stats 同源):每项时段内输出时长最多的物体 = 该项判定;
    时段逐项独立,连击(球L×2)是两个独立时段。归入规则:段与 [盯看起点-0.3, 叮+0.5] 重叠 ≥0.5 s,
    或整段落在分析窗内。返回 {k: (多数物体, 设闸多数物体, 多数时长, 段数)},无时段 -> None。"""
    for e in fin:
        e["_acc"] = gate(e, GATES["live"])
    out = {}
    for it in items:
        if it.get("win_start_abs") is None:
            out[it["k"]] = None
            continue
        w0, w1 = it["win_start_abs"], it["win_end_abs"]
        s0 = (it["stare_start_abs"] - 0.3) if it.get("stare_start_abs") is not None else w0
        fl = [e for e in fin if overlap(e["t_start"], e["t_end"], s0, w1) >= 0.5 or (e["t_start"] >= w0 and e["t_end"] <= w1)]
        objf = [e for e in fl if e.get("object")]
        dur = defaultdict(float)
        for e in objf:
            dur[e["object"]] += e["duration_s"]
        maj = max(dur, key=dur.get) if dur else None
        gd = defaultdict(float)
        for e in objf:
            if e["_acc"]:
                gd[e["object"]] += e["duration_s"]
        gmaj = max(gd, key=gd.get) if gd else None
        out[it["k"]] = (maj, gmaj, (dur[maj] if maj else 0.0), len(objf))
    return out


class EraCtx:
    """一代地图的几何上下文:命名质心、带标签高斯、KD 树、半径、渲染器。"""
    def __init__(self, era, use_gpu=True):
        self.era = era
        seg = ENV[era]
        self.named = load_named(seg)
        z = np.load(seg / "points.npz")
        self.xyz, self.lab = z["xyz"], z["label"]
        names = json.loads((seg / "names.json").read_text(encoding="utf-8"))
        self.name_of = {int(k): v for k, v in names.items() if v}
        self.radius = {}
        for lid, nm in self.name_of.items():
            pts = self.xyz[self.lab == lid]
            if len(pts):
                ext = pts.max(0) - pts.min(0)
                self.radius[nm] = max(self.radius.get(nm, 0.0), float(ext.max()) / 2)
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.xyz)
        # 只有"候选物体"才算遮挡者:场所(物品台/桌子,places.json)与未命名碎片不算——
        # 论文关心的是被别的候选挡住;桌面前沿挡住球心射线是几何常态,不是消歧难度
        pl = seg / "places.json"
        places = set(json.loads(pl.read_text(encoding="utf-8"))) if pl.exists() else {"物品台", "桌子"}
        self.blockable = {n for n in self.named if n not in places}
        self.splat = None
        if use_gpu and CKPT[era].exists():
            from gaze_to_world import SplatDepth
            self.splat = SplatDepth(CKPT[era])

    def first_hit(self, origin, target_pt):
        """射线首击点最近的命名物(名字, 距离差 expected-depth);None=没打到东西。"""
        v = target_pt - origin
        dist = float(np.linalg.norm(v))
        u = v / dist
        # 细射线:半角 0.35° 的 9x9 小块取中心像素(depth_along_ray 的 33x33 中位在 3 m 外会被桌面前沿拉近)
        depth, alpha, dirs, tmul = self.splat.patch_along_ray(origin, u, half_angle=math.radians(0.35), S=9)
        c = 4
        if float(alpha[c, c]) < 0.5 or float(depth[c, c]) <= 0.05:
            return None, None
        t = float(depth[c, c] * tmul[c, c])
        hit = origin + dirs[c, c] * t
        _, ii = self.tree.query(hit)
        nm = self.name_of.get(int(self.lab[ii]), "")
        return nm, dist - t

    def occlusion(self, origin, target, n_ring=6, rings=(0.5, 0.9)):
        """目标在该头位看来被别的候选物挡住的比例 (0–1) 与挡它的物体名。
        在目标圆盘(半径 r,与视线垂直)上取 1 + 6×len(rings) 个采样点,各投一条细射线,首击点最近的命名高斯
        是别的候选且比采样点近 → 该射线被挡;hidden = 被挡射线数 / 总数。>=0.5 记 occluded,0<x<0.5 记 partial。"""
        if self.splat is None or target not in self.named:
            return None, ""
        c = np.asarray(self.named[target], float)
        o = np.asarray(origin, float)
        r = self.radius.get(target, 0.04)
        d = c - o; d /= np.linalg.norm(d)
        up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, up); u /= np.linalg.norm(u); v = np.cross(u, d)
        pts = [c]
        for rho in rings:
            for i in range(n_ring):
                ph = 2 * math.pi * i / n_ring
                pts.append(c + r * rho * (math.cos(ph) * u + math.sin(ph) * v))
        blocked, blockers, others = 0, [], []
        for pt in pts:
            nm, gap = self.first_hit(o, pt)
            if nm is None or gap is None or nm == target or gap <= max(0.3 * r, 0.03):
                continue
            if nm in self.blockable:
                blocked += 1
                blockers.append(nm)
            else:
                others.append(nm or "unnamed")
        tag = "/".join(dict.fromkeys(blockers))
        if others and not blockers:
            tag = "(" + "/".join(dict.fromkeys(others)) + ")"
        return round(blocked / len(pts), 2), tag


def load_poses(rec):
    p = R / rec / "station_poses.jsonl"
    if not p.exists():
        return None
    T, P = [], []
    for ln in p.open(encoding="utf-8"):
        e = json.loads(ln)
        if e.get("pos"):
            T.append(e["t"]); P.append(e["pos"])
    return np.array(T), np.array(P)


def head_in_window(T, P, w0, w1):
    m = (T >= w0 - PAD) & (T <= w1 + PAD)
    if m.sum() >= 3:
        return np.median(P[m], axis=0), "window", int(m.sum())
    mid = (w0 + w1) / 2
    j = int(np.argmin(np.abs(T - mid)))
    if abs(T[j] - mid) <= NEAR_S:
        return P[j], "nearest", 1
    return np.median(P, axis=0), "recording_median", len(T)


ROOM_OF = {"v7": "老房", "v8": "新房", "v9": "新房", "v10": "新房"}


def write_package(rows, out):
    """正式数据包:trials.csv(collect_e1 同字段 + 逐 trial 附加列)、curve.csv(站立主曲线,与 plot_fig4 的筛选一致)、
    curve_clear/curve_occluded(主曲线按视线是否被候选物挡住拆开)、curve_walking(边走录像单列)。"""
    fields = ["rec", "card", "room", "map", "station", "tags", "target", "outcome", "theta_deg", "theta_src",
              "vote", "dur_s", "dist_m", "k", "theta_unit", "occluded", "blocker", "head_frames", "window_src",
              "dwell", "dwell_obj", "dwell_s", "gated", "gated_obj", "n_seg"]
    trials = []
    for r in rows:
        tags = [t for t in r["tags"].split("|") if t]
        occ = r["occluded"]
        if occ not in ("", 0, 0.0, "0", "0.0"):
            tags.append("occluded" if float(occ) >= 0.5 else "partial")
        trials.append({"rec": r["rec"].split("/")[1], "card": r["card"], "room": ROOM_OF.get(r["map"], ""), "map": r["map"],
                       "station": "", "tags": "|".join(tags), "target": r["target"], "outcome": r["outcome"],
                       "theta_deg": r["theta_trial"], "theta_src": r["head_src"], "vote": "", "dur_s": "", "dist_m": r["dist_m"],
                       "k": r["k"], "theta_unit": r["theta_unit"], "occluded": occ, "blocker": r["blocker"],
                       "head_frames": r["head_frames"], "window_src": r["window_src"],
                       "dwell": r["dwell"], "dwell_obj": r["dwell_obj"], "dwell_s": r["dwell_s"],
                       "gated": r["gated"], "gated_obj": r["gated_obj"], "n_seg": r["n_seg"]})
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(trials)

    def curve(sub, fname):
        th = np.array([float(r["theta_deg"]) for r in sub]); ok = np.array([r["outcome"] == "hit" for r in sub])
        with (out / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["theta_lo", "theta_hi", "n", "hits", "acc"])
            for lo, hi in zip(BINS, BINS[1:]):
                m = (th >= lo) & (th < hi)
                if m.sum():
                    w.writerow([lo, hi, int(m.sum()), int(ok[m].sum()), round(float(ok[m].mean()), 3)])
    valid = [r for r in trials if r["theta_deg"] != ""]
    stand = [r for r in valid if not ({"stress", "walking"} & set(r["tags"].split("|")))]
    curve(stand, "curve.csv")
    curve([r for r in stand if not ({"occluded", "partial"} & set(r["tags"].split("|")))], "curve_clear.csv")
    curve([r for r in stand if {"occluded", "partial"} & set(r["tags"].split("|"))], "curve_occluded.csv")
    curve([r for r in valid if "walking" in r["tags"]], "curve_walking.csv")
    print(f"数据包 -> {out}/trials.csv({len(trials)} 行)curve.csv(站立主曲线 {len(stand)})curve_clear/occluded/walking.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs/E1_DATA"))
    ap.add_argument("--windows", default=str(ROOT / "docs/E1_DATA/audit_0906/windows"))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-gpu", action="store_true", help="不判遮挡(无 CUDA 时)")
    ap.add_argument("--log-override", default="", help="老录像改读这个日志名(回归对照用,如 intents_e4_v2s10.jsonl)")
    ap.add_argument("--log", default="", help="所有录像统一读这个日志名(消融配置,如 intents_abl_s05.jsonl)")
    ap.add_argument("--tag", default="", help="输出文件名后缀:trials_theta_<tag>.csv")
    ap.add_argument("--occ-from", default="", help="从已有 trials_theta CSV 复用逐 trial 遮挡标记(免 GPU 重算)")
    ap.add_argument("--package", action="store_true",
                    help="同时写正式数据包 <out>/trials.csv + curve.csv(+ curve_clear/curve_occluded/curve_walking.csv):"
                         "字段与 collect_e1 相同,theta_deg=θ_trial、theta_src=head_src,tags 加 occluded/partial;plot_fig4 直接可用")
    a = ap.parse_args()
    ctx = {}
    rows = []
    occ_cache = {}
    if a.occ_from:
        for r in csv.DictReader(open(a.occ_from, encoding="utf-8")):
            occ_cache[(r["rec"], int(r["k"]))] = (r["occluded"], r["blocker"])
    for rec, card, era, flags in RECS:
        if a.only and rec not in a.only.split(","):
            continue
        if "excluded" in flags:
            continue
        wp = Path(a.windows) / f"{rec_tag(rec)}.json"
        if not wp.exists():
            print(f"[!] {rec}: 无时段文件,跳过"); continue
        poses = load_poses(rec)
        if poses is None:
            print(f"[!] {rec}: 无 station_poses.jsonl(先跑 station_theta),跳过"); continue
        T, P = poses
        if a.log:
            logp = R / rec / a.log
        else:
            logp = R / rec / ((a.log_override if era != "v10" else LOG_OF_ERA["v10"]) if a.log_override else LOG_OF_ERA.get(era, LOG_DEFAULT))
        if not logp.exists():
            print(f"[!] {rec}: 无 {logp.name},跳过"); continue
        if era not in ctx:
            ctx[era] = EraCtx(era, use_gpu=(not a.no_gpu) and not occ_cache)
        C = ctx[era]
        fin = load_finals(logp)
        origins = np.array([e["origin_world"] for e in fin if e.get("object") and e.get("origin_world")], float)
        o_unit = np.median(origins, axis=0) if len(origins) else None   # eval_e1 的结果盲录像中位头位(对照用)
        win = json.loads(wp.read_text(encoding="utf-8"))
        outcomes = window_outcomes(fin, win["items"])
        seq = era_alias(CARDS[card][1], C.named)
        n_moved = 0
        for it in win["items"]:
            k = it["k"]; tgt = seq[k - 1]
            oc = outcomes.get(k)
            # 无时段的项(回退时段录像里没被卡序对齐命中的项)照审计口径记 none,计入分母;头位退回录像中位
            maj, gmaj, maj_dur, n_seg = oc if oc is not None else (None, None, 0.0, 0)
            res = "correct" if (maj and canon(maj) == canon(tgt)) else ("wrong" if maj else "none")
            gres = "correct" if (gmaj and canon(gmaj) == canon(tgt)) else ("wrong" if gmaj else "none")
            hit = res == "correct"
            tu = theta_min(o_unit, tgt, C.named) if o_unit is not None else None
            w0, w1 = it.get("stare_start_abs"), it.get("beep_abs")
            if w0 is None or w1 is None:
                head, src, nfr = np.median(P, axis=0), "no_window", len(T)
            else:
                head, src, nfr = head_in_window(T, P, w0, w1)
            th = theta_min(head, tgt, C.named)
            if occ_cache:
                o = occ_cache.get((rec, k), ("", ""))
                occ, blocker = ((float(o[0]) if o[0] != "" else None), o[1])
            else:
                occ, blocker = C.occlusion(head, tgt)
            dist = float(np.linalg.norm(np.asarray(C.named[tgt]) - head)) if tgt in C.named else None
            if th is not None and tu is not None and bin_of(th) != bin_of(tu):
                n_moved += 1
            rows.append({"rec": rec, "card": card, "map": era, "tags": flags, "k": k, "target": tgt,
                         "outcome": "hit" if hit else "miss", "dwell": res, "dwell_obj": maj or "", "dwell_s": round(maj_dur, 2),
                         "gated": gres, "gated_obj": gmaj or "", "n_seg": n_seg,
                         "theta_unit": ("" if tu is None else round(tu, 2)),
                         "theta_trial": ("" if th is None else round(th, 2)),
                         "head_src": src, "head_frames": nfr, "dist_m": ("" if dist is None else round(dist, 2)),
                         "occluded": ("" if occ is None else occ), "blocker": blocker,
                         "walking": int("walking" in flags), "window_src": it.get("window_src", "")})
        print(f"{rec:20} {card:3} {era:4} items {len(win['items'])}  换箱 {n_moved}  时段 {win.get('method','')}")
    out = Path(a.out) / (f"trials_theta_{a.tag}.csv" if a.tag else "trials_theta.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"-> {out}  ({len(rows)} trial)")

    main_rows = [r for r in rows if r["theta_trial"] != "" and not r["walking"] and "stress" not in r["tags"]]
    def curve(sub, key):
        tab = defaultdict(lambda: [0, 0])
        for r in sub:
            b = bin_of(float(r[key])); tab[b][1] += 1; tab[b][0] += (r["outcome"] == "hit")
        return tab
    tu = curve([r for r in main_rows if r["theta_unit"] != ""], "theta_unit")
    tt = curve(main_rows, "theta_trial")
    clear = curve([r for r in main_rows if r["occluded"] in (0, 0.0, "0", "0.0")], "theta_trial")
    occl = curve([r for r in main_rows if r["occluded"] not in ("", 0, 0.0, "0", "0.0")], "theta_trial")
    print(f"\n{'箱':>9} {'θ_unit(录像中位)':>16} {'θ_trial(时段头位)':>17} {'清晰':>12} {'有遮挡':>12}")
    for lo, hi in zip(BINS, BINS[1:]):
        b = f"{lo:g}-{hi:g}"
        cell = lambda t: (f"{t[b][0]:>3}/{t[b][1]:<3}{t[b][0]/t[b][1]:.2f}" if t[b][1] else f"{'—':>10}")
        print(f"{b:>9} {cell(tu):>16} {cell(tt):>17} {cell(clear):>12} {cell(occl):>12}")
    for b in ("<0.5", ">=20"):
        if tt[b][1]:
            print(f"{b:>9} {'':>16} {tt[b][0]:>3}/{tt[b][1]:<3}  (论文分箱外)")
    if a.package:
        write_package(rows, Path(a.out))
    moved = sum(1 for r in main_rows if r["theta_unit"] != "" and bin_of(float(r["theta_unit"])) != bin_of(float(r["theta_trial"])))
    print(f"\n主曲线 {len(main_rows)} trial;θ_trial 与 θ_unit 不同箱 {moved};头位来源 {dict(Counter(r['head_src'] for r in main_rows))};"
          f"遮挡分 {dict(Counter(str(r['occluded']) for r in main_rows))}")


if __name__ == "__main__":
    main()
