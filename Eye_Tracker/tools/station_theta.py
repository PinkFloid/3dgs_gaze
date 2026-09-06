#!/usr/bin/env python3
"""station_theta.py -- 录后现算:每条录像的实际站位 + 逐项 θ(论文口径)+ 分箱覆盖 + 下一站建议。

现场看不到显示器,所以不做实时——录完一批回到工作站跑一次:
  conda run --no-capture-output -n nerfstudio python Eye_Tracker/tools/station_theta.py --day 2026_09_07
  conda run --no-capture-output -n nerfstudio python Eye_Tracker/tools/station_theta.py ~/recordings/2026_09_07/p1_v6 ... [--card v6]
  conda run --no-capture-output -n nerfstudio python Eye_Tracker/tools/station_theta.py --plan   # 只看"站位→θ"速查表

每条录像:
  1. 卡号/佩戴者从目录名推断(含 v1…v6 即可,前缀 p1_/p2_ 为佩戴者,b/2 等后缀无妨;--card 可强制,
     Pupil 自动编号的 000 目录用 000=v1 这种写法指定)。
  2. 定位:tag→T_world_cam(与 pupil_localizer 同一套 PnP+三道门限),每 --every 帧取一帧,
     结果缓存 <rec>/station_poses.jsonl(重跑免定位;--relocalize 强制重算)。
  3. 站位 = 定位帧头位(世界相机)中位;给到 M/L/R 的卷尺距离、方位角 α(0=正对,右侧为正)、定位率、
     看到过的 tag;行走卡另给 p10/p90。
  4. 逐项 θ = theta_min(中位头位, 目标, 全部命名物)——与 eval_e1 的 theta_unit_deg 同口径(结果盲,
     注视命中与否都用它);球卡另给"只算三球"的 θ_ball 作参考(卡片文档标的 θ 是这个口径)。
  5. 汇总:theta_bins.json 的分箱里今日新增 trial 数(球卡/综合卡分开)+ 论文已有(docs/E1_DATA/curve.csv),
     对最缺的分箱列出可行站位(球卡用 V1 序列、综合卡用 V6 序列做前向预测,站位限定在可站地板内)。
产物:<rec>/station_theta.json;--day 时另写 ~/recordings/<day>/station_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_cards import CARDS  # noqa: E402
from eval_e1 import era_alias, load_named  # noqa: E402
from pupil_localizer import load_fisheye, load_tags, recording_frames, scale_K, solve_pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "SceneRebuild"
R = Path("/home/liuchy/recordings")
BINS = [(0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, 20.0)]
BALLS = ("网球L", "网球M", "网球R")
# 可站地板(v10 图实测):右侧 x≈2.3 起是桌子/墙线,左侧 x<-0.2 有杂物,桌前 y<0.3,最远 y≈-4.6
FLOOR = dict(xmin=-0.2, xmax=2.0, ymin=-4.6, ymax=0.2)
PLAN_D = (1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
PLAN_A = (-30, -15, 0, 15, 30, 45, 60)


def bin_of(th):
    if th is None:
        return None
    for lo, hi in BINS:
        if lo <= th < hi:
            return f"{lo:g}-{hi:g}"
    return "<0.5" if th < 0.5 else ">=20"


def theta_nb(origin, target, named):
    """theta_min 同款,但连同定义 θ 的邻居名一起返回。"""
    if origin is None or target not in named:
        return None, None
    o = np.asarray(origin, float)
    v0 = named[target] - o
    v0 = v0 / np.linalg.norm(v0)
    best, who = None, None
    for nm, c in named.items():
        if nm == target:
            continue
        v = c - o
        v = v / np.linalg.norm(v)
        ang = math.degrees(math.acos(float(np.clip(v0 @ v, -1, 1))))
        if best is None or ang < best:
            best, who = ang, nm
    return best, who


def infer_card(name: str):
    m = re.search(r"(?<![a-z])([vscue]\d)(?![0-9])", name.lower())
    return m.group(1) if m and m.group(1) in CARDS else None


def infer_person(name: str):
    m = re.search(r"(?:^|[_-])(p\d+)(?:[_-]|$)", name.lower())  # p1_v6 / v2_near_p2 都认
    return m.group(1) if m else "本人"


# ---------------------------------------------------------------- localization (same gates as pupil_localizer)

def localize(rec: Path, tags, K_calib, D, every: int, relocalize: bool):
    cache = rec / "station_poses.jsonl"
    if cache.exists() and not relocalize:
        rows = [json.loads(ln) for ln in cache.open(encoding="utf-8") if ln.strip()]
        if rows:
            return rows
    allc = np.concatenate(list(tags.values()))
    bounds = (allc[:, 0].min() - 3, allc[:, 0].max() + 3, allc[:, 1].min() - 3, allc[:, 1].max() + 3)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    K = None
    rows, last_accept, n_reject = [], None, 0
    for i, (ts, img) in enumerate(recording_frames(rec)):
        if i % every:
            continue
        if K is None:
            K = scale_K(K_calib, (1920, 1080), (img.shape[1], img.shape[0]))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        visible = [] if ids is None else [int(v) for v in ids.flatten()]
        known = [k for k, t in enumerate(visible) if t in tags]
        pose, n_inl, reproj = None, 0, None
        if known:
            obj = np.concatenate([tags[visible[k]] for k in known])
            px = np.concatenate([corners[k].reshape(4, 2) for k in known])
            pts = cv2.fisheye.undistortPoints(px.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
            pose, n_inl, reproj = solve_pose(obj, pts, 0.01)
        if pose is not None:
            x, y, z = pose[:3, 3]
            if not (bounds[0] < x < bounds[1] and bounds[2] < y < bounds[3] and 0.15 < z < 2.8):
                pose = None
            elif reproj is not None and reproj > 0.006:
                pose = None
        if pose is not None and last_accept is not None:
            if ts - last_accept[0] < 0.25 and np.linalg.norm(pose[:3, 3] - last_accept[1]) > 1.0:
                n_reject += 1
                if n_reject <= 5:
                    pose = None
        if pose is not None:
            n_reject = 0
            last_accept = (ts, pose[:3, 3].copy())
        rows.append({"t": ts, "pos": None if pose is None else [round(float(v), 4) for v in pose[:3, 3]],
                     "tags": sorted(visible[k] for k in known), "n_inl": n_inl,
                     "reproj": None if reproj is None else round(float(reproj), 5)})
        if len(rows) % 300 == 0:
            n_ok = sum(1 for r in rows if r["pos"])
            print(f"    定位中 {len(rows)} 帧(抽样 1/{every}),已定位 {n_ok}", flush=True)
    with cache.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


# ---------------------------------------------------------------- geometry

class Layout:
    def __init__(self, named):
        self.named = named
        self.balls = era_alias(list(BALLS), named)
        L, M, Rb = (named[b] for b in self.balls)
        self.L, self.M, self.R = L, M, Rb
        u = Rb - L
        u[2] = 0
        self.u_lr = u / np.linalg.norm(u)
        self.n_front = np.array([self.u_lr[1], -self.u_lr[0], 0.0])  # 指向 -y 侧(大 tag 76 那边)

    def station(self, o):
        v = o - self.M
        vf = v.copy()
        vf[2] = 0
        alpha = math.degrees(math.atan2(float(vf @ self.u_lr), float(vf @ self.n_front)))
        return dict(x=float(o[0]), y=float(o[1]), z=float(o[2]), dist=float(np.linalg.norm(v)),
                    alpha=alpha, dM=float(np.linalg.norm(o - self.M)),
                    dL=float(np.linalg.norm(o - self.L)), dR=float(np.linalg.norm(o - self.R)))

    def eye_at(self, dist, alpha_deg, eye_h):
        dz = eye_h - self.M[2]
        df = math.sqrt(max(dist * dist - dz * dz, 0.01))
        a = math.radians(alpha_deg)
        return self.M + self.n_front * df * math.cos(a) + self.u_lr * df * math.sin(a) + np.array([0, 0, dz])

    def theta_ball(self, o, target):
        if target not in self.balls:
            return None
        o = np.asarray(o, float)
        v0 = self.named[target] - o
        v0 = v0 / np.linalg.norm(v0)
        best = None
        for b in self.balls:
            if b == target:
                continue
            v = self.named[b] - o
            v = v / np.linalg.norm(v)
            ang = math.degrees(math.acos(float(np.clip(v0 @ v, -1, 1))))
            best = ang if best is None else min(best, ang)
        return best


def card_kind(seq, named):
    if not any(t in named for t in seq):
        return "负例"
    return "球卡" if set(seq) <= set(era_alias(list(BALLS), named)) else "综合卡"


def item_thetas(lay: Layout, o, card):
    seq = era_alias(CARDS[card][1], lay.named)
    out = []
    for k, tgt in enumerate(seq, 1):
        th, nb = theta_nb(o, tgt, lay.named)
        out.append(dict(k=k, target=tgt, theta=th, neighbor=nb, bin=bin_of(th),
                        theta_ball=lay.theta_ball(o, tgt)))
    return out


# ---------------------------------------------------------------- per recording

def analyze(rec: Path, card: str, person: str, lay: Layout, tags, K_calib, D, every, relocalize):
    rows = localize(rec, tags, K_calib, D, every, relocalize)
    P = np.array([r["pos"] for r in rows if r["pos"]], float)
    seen = Counter(t for r in rows for t in r["tags"])
    res = dict(rec=str(rec), name=rec.name, card=card, person=person, n_frames=len(rows), n_loc=int(len(P)),
               loc_rate=(len(P) / len(rows) if rows else 0.0),
               duration_s=(rows[-1]["t"] - rows[0]["t"]) if len(rows) > 1 else 0.0,
               tags_seen={str(k): v for k, v in seen.most_common()})
    if len(P) < 10:
        res["error"] = f"定位帧太少({len(P)}),算不了站位"
        return res
    o = np.median(P, axis=0)
    st = lay.station(o)
    dist_all = np.linalg.norm(P - lay.M, axis=1)
    alpha_all = np.array([lay.station(p)["alpha"] for p in P])
    st["dist_p10"], st["dist_p90"] = (float(np.percentile(dist_all, q)) for q in (10, 90))
    st["alpha_p10"], st["alpha_p90"] = (float(np.percentile(alpha_all, q)) for q in (10, 90))
    seq = era_alias(CARDS[card][1], lay.named)
    res.update(origin=[round(float(v), 3) for v in o], station=st, kind=card_kind(seq, lay.named),
               items=item_thetas(lay, o, card))
    return res


def show(res):
    print(f"\n=== {res['name']}  卡 {res['card']}({res.get('kind', '?')})  佩戴者 {res['person']}  "
          f"{res['duration_s']:.0f}s  定位率 {res['loc_rate']*100:.0f}%({res['n_loc']}/{res['n_frames']} 抽样帧)"
          f"  看到 tag {dict(list(res['tags_seen'].items())[:6])}")
    if "error" in res:
        print("  !!", res["error"])
        return
    s = res["station"]
    walk = (s["dist_p90"] - s["dist_p10"] > 0.6) or (s["alpha_p90"] - s["alpha_p10"] > 25)
    print(f"  站位(头位中位):到M {s['dM']:.2f}  到L {s['dL']:.2f}  到R {s['dR']:.2f} m,"
          f"  方位角 α {s['alpha']:+.0f}°({'右' if s['alpha'] > 0 else '左'}侧),眼高 {s['z']:.2f} m,"
          f"  世界系 ({s['x']:.2f},{s['y']:.2f})"
          + (f"  [行走:距离 {s['dist_p10']:.1f}–{s['dist_p90']:.1f} m,α {s['alpha_p10']:+.0f}…{s['alpha_p90']:+.0f}°]" if walk else ""))
    if res["kind"] == "负例":
        print("  负例卡:目标不是命名物,无 θ(站位照记)")
        return
    per = {}
    for it in res["items"]:
        per.setdefault(it["target"], it)
    print("  逐目标 θ(论文口径=对全部命名物的最小张角 | 只算三球):")
    for tgt, it in per.items():
        n = sum(1 for i in res["items"] if i["target"] == tgt)
        tb = "" if it["theta_ball"] is None else f" | 三球 {it['theta_ball']:.2f}°"
        if it["theta"] is None:
            print(f"    {tgt:5s} ×{n}: 不在命名物里,无 θ")
            continue
        print(f"    {tgt:5s} ×{n}: θ={it['theta']:.2f}° → 箱 {it['bin']:8s} 最近邻 {it['neighbor']}{tb}")
    cnt = Counter(it["bin"] for it in res["items"] if it["bin"])
    print("  本条 trial 落箱:" + "  ".join(f"{b}:{n}" for b, n in sorted(cnt.items(), key=lambda kv: float(kv[0].split('-')[0].lstrip('<>=')))))


# ---------------------------------------------------------------- coverage & plan

def existing_bins():
    p = ROOT / "docs/E1_DATA/curve.csv"
    out = {}
    if p.exists():
        for r in csv.DictReader(p.open(encoding="utf-8")):
            out[f"{float(r['theta_lo']):g}-{float(r['theta_hi']):g}"] = int(r["n"])
    return out

BIN_KEYS = [f"{lo:g}-{hi:g}" for lo, hi in BINS]


def coverage(results):
    have = existing_bins()
    new = {k: Counter() for k in ("球卡", "综合卡")}
    for r in results:
        if "items" not in r or r["kind"] == "负例":
            continue
        for it in r["items"]:
            if it["bin"] in BIN_KEYS:
                new[r["kind"]][it["bin"]] += 1
    print("\n=== θ 分箱覆盖(trial 数;新增 = 本次给定的全部录像)")
    print(f"  {'箱(°)':>9} {'论文已有':>8} {'新增球卡':>8} {'新增综合':>8} {'合计':>6}")
    tot = {}
    for k in BIN_KEYS:
        t = have.get(k, 0) + new["球卡"][k] + new["综合卡"][k]
        tot[k] = t
        print(f"  {k:>9} {have.get(k, 0):>8} {new['球卡'][k]:>8} {new['综合卡'][k]:>8} {t:>6}")
    return tot


def plan(lay: Layout, eye_h: float, tot: dict | None, ball_card="v1", comp_card="v6"):
    print(f"\n=== 站位 → θ 速查(眼高 {eye_h:.2f} m;格子 = 该卡各项 θ 的中位/最小(论文口径),× = 站不到的地板)")
    cands = []
    for kind, card in (("球卡", ball_card), ("综合卡", comp_card)):
        seq = era_alias(CARDS[card][1], lay.named)
        uniq = list(dict.fromkeys(seq))
        print(f"  -- {kind}(按 {card} 序列 {len(seq)} 项,{len(uniq)} 个目标)")
        hdr = "距离\\α"
        print("  " + f"{hdr:>7}" + "".join(f"{a:>+12d}°" for a in PLAN_A))
        for d in PLAN_D:
            line = f"  {d:>6.1f}m"
            for a in PLAN_A:
                o = lay.eye_at(d, a, eye_h)
                ok = FLOOR["xmin"] <= o[0] <= FLOOR["xmax"] and FLOOR["ymin"] <= o[1] <= FLOOR["ymax"]
                ths = [theta_nb(o, t, lay.named)[0] for t in seq]
                ths = [t for t in ths if t is not None]
                if not ok or not ths:
                    line += f"{'×':>13s}"
                    continue
                med, mn = float(np.median(ths)), float(min(ths))
                line += f"{med:6.1f}/{mn:4.1f}  "
                cands.append(dict(kind=kind, card=card, d=d, a=a, o=o, bins=Counter(bin_of(t) for t in ths),
                                  st=lay.station(o)))
            print(line)
    if not tot:
        return
    print("\n=== 最缺的分箱 → 建议站位(按能落进该箱的 trial 数排序;卷尺数=头到球的直线距离)")
    for k in sorted(BIN_KEYS, key=lambda b: tot.get(b, 0))[:4]:
        print(f"  箱 {k}°(现有 {tot.get(k, 0)}):")
        for kind in ("球卡", "综合卡"):
            best = sorted((c for c in cands if c["kind"] == kind and c["bins"][k] > 0),
                          key=lambda c: (-c["bins"][k], c["d"], abs(c["a"])))[:3]
            if not best:
                print(f"    {kind}:没有可站位置能落进这个箱")
                continue
            print(f"    {kind}:" + ";  ".join(
                f"{c['d']:.1f}m α{c['a']:+d}°→{c['bins'][k]}项(到M {c['st']['dM']:.2f} 到L {c['st']['dL']:.2f} 到R {c['st']['dR']:.2f})"
                for c in best))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recs", nargs="*", help="录像目录;可写 目录=卡号 或 目录=卡号=佩戴者 覆盖推断")
    ap.add_argument("--day", action="append", default=None, help="扫 ~/recordings/<day>/ 下全部目录(能推断卡号的才算);可重复给多天,覆盖统计跨天累计")
    ap.add_argument("--card", default=None, help="强制卡号(对所有给定目录)")
    ap.add_argument("--every", type=int, default=3, help="每 N 帧定位 1 帧(默认 3≈10Hz,一条 130s 录像实测约 35s)")
    ap.add_argument("--relocalize", action="store_true")
    ap.add_argument("--seg-dir", default=str(SCENE / "lab_result/segmentation_sam"))
    ap.add_argument("--tags", default=str(SCENE / "world_size/tags_world.json"))
    ap.add_argument("--calib", default=str(SCENE / "Calibration_result/world_camera_calibration.npz"))
    ap.add_argument("--eye-h", type=float, default=None, help="速查表用的眼高(默认取今日录像中位,无录像则 1.55)")
    ap.add_argument("--plan", action="store_true", help="只打印站位→θ 速查表")
    ap.add_argument("--no-plan", action="store_true", help="不打印速查与建议")
    a = ap.parse_args()

    named = load_named(Path(a.seg_dir))
    lay = Layout(named)
    print(f"命名物 {len(named)}:{' '.join(named)}  |  三球 {lay.balls},间距 "
          f"{np.linalg.norm(lay.M - lay.L)*100:.1f}/{np.linalg.norm(lay.R - lay.M)*100:.1f} cm")

    jobs = []
    for day in (a.day or []):
        for d in sorted((R / day).iterdir()):
            if d.is_dir() and (d / "world.mp4").exists():
                c = a.card or infer_card(d.name)
                if c:
                    jobs.append((d, c, infer_person(d.name)))
                else:
                    print(f"  跳过 {day}/{d.name}:目录名里没有卡号(用 目录=卡号 指定)")
    for spec in a.recs:
        parts = spec.split("=")
        d = Path(parts[0]).expanduser()
        if not d.is_absolute() and not d.exists():
            d = R / parts[0]
        c = parts[1] if len(parts) > 1 else (a.card or infer_card(d.name))
        p = parts[2] if len(parts) > 2 else infer_person(d.name)
        if not c:
            sys.exit(f"{d}: 推断不出卡号,用 {d}=v1 这种写法")
        if c not in CARDS:
            sys.exit(f"卡号 {c} 不在 e1_cards.py:{list(CARDS)}")
        jobs.append((d, c, p))

    results = []
    if jobs:
        K_calib, D = load_fisheye(a.calib)
        tags, _ = load_tags(a.tags)
        for d, c, p in jobs:
            print(f"\n[{d.name}] 卡 {c} 佩戴者 {p}:定位中…", flush=True)
            res = analyze(d, c, p, lay, tags, K_calib, D, a.every, a.relocalize)
            (d / "station_theta.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            results.append(res)
        for res in results:
            show(res)
        for day in (a.day or []):
            out = R / day / "station_summary.csv"
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["rec", "card", "kind", "person", "loc_rate", "dM", "dL", "dR", "alpha_deg", "eye_h",
                            "theta_med", "theta_min", "theta_max", "bins"])
                for r in results:
                    if Path(r["rec"]).parent.name != day:
                        continue
                    if "items" not in r:
                        w.writerow([r["name"], r["card"], "", r["person"], f"{r['loc_rate']:.2f}"] + [""] * 9)
                        continue
                    s, ths = r["station"], [i["theta"] for i in r["items"] if i["theta"] is not None]
                    cnt = Counter(i["bin"] for i in r["items"] if i["bin"])
                    w.writerow([r["name"], r["card"], r["kind"], r["person"], f"{r['loc_rate']:.2f}",
                                f"{s['dM']:.2f}", f"{s['dL']:.2f}", f"{s['dR']:.2f}", f"{s['alpha']:.0f}", f"{s['z']:.2f}",
                                f"{np.median(ths):.2f}" if ths else "", f"{min(ths):.2f}" if ths else "",
                                f"{max(ths):.2f}" if ths else "", " ".join(f"{b}:{n}" for b, n in sorted(cnt.items()))])
            print(f"\n汇总表 -> {out}")
    tot = coverage(results)
    if a.plan or (not a.no_plan):
        eye_h = a.eye_h or (float(np.median([r["station"]["z"] for r in results if "station" in r]))
                            if any("station" in r for r in results) else 1.55)
        plan(lay, eye_h, tot)


if __name__ == "__main__":
    main()
