#!/usr/bin/env python3
"""eval_e1.py -- E1 卡片自动打分:intents.jsonl + 盯看序列 -> 逐项 verdict 表。

    gaze_live --replay <rec> --headless --log intents.jsonl --stamp-exclude 126,124
    python Eye_Tracker/tools/eval_e1.py intents.jsonl --seq "R L M L M M R L L M R M R R L"
    # 或综合卡: --seq "苹果 网球M 石榴 香蕉 网球R 网球L ..."

规则(与 docs/E1_CARDS.md 约定一致):
- 只取 final(provisional=false)注视;**floor 与 background 一律当背景丢弃**
  (移开视线扫到地板不占序号);墙 tag 书签落在背景上,自然不占位。
- 第 k 个存活 final 对卡上第 k 项,严格按序;数量不齐如实报,不做智能对齐。
- 每项额外算:头到目标距离、该目标与最近其他命名物的**张角 θ_min**
  (Fig.4 横轴;正对/斜位/对角自动落到同一把尺上)。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SCENE = Path(__file__).resolve().parents[2] / "SceneRebuild"
TOKEN = {"L": "球L", "M": "球M", "R": "球R"}  # v9 改名 网球X->球X(旧卡 CSV 已归档)
DROP = {"", "floor", "background", "none", None}


def load_named(map_dir: Path):
    names = json.loads((map_dir / "names.json").read_text())
    inst = {r["id"]: r for r in
            json.loads((map_dir / "instances.json").read_text())["instances"]}
    out = {}
    for k, nm in names.items():
        if nm and int(k) in inst:
            out[nm] = np.array(inst[int(k)]["centroid"], float)
    return out


def finals(path: Path):
    rows = []
    for ln in path.open(encoding="utf-8"):
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("provisional"):
            continue
        if e.get("object") in DROP:
            continue
        rows.append(e)
    return rows


def theta_min(origin, target, named):
    """目标方向与其他命名物方向的最小夹角(度)——本 trial 的角分辨难度。"""
    if origin is None or target not in named:
        return None
    o = np.array(origin, float)
    v0 = named[target] - o
    v0 /= np.linalg.norm(v0)
    best = None
    for nm, c in named.items():
        if nm == target:
            continue
        v = c - o
        v /= np.linalg.norm(v)
        ang = math.degrees(math.acos(float(np.clip(v0 @ v, -1, 1))))
        best = ang if best is None else min(best, ang)
    return best


def episodes(ev, merge_gap=1.2, min_dur=1.0):
    """final 流 -> 注视段:同物且间隔<merge_gap 合并(一次盯看常被切成数条 final),
    累计时长<min_dur 的段丢弃(扫视路过的碎片,盯看协议是 2-3s)。
    每段带票面最高、时长最长的代表 final(取 vote/origin/dist 用)。"""
    eps = []
    for e in ev:
        if eps and eps[-1]["object"] == e["object"] and \
                e["t_start"] - eps[-1]["t_end"] < merge_gap:
            ep = eps[-1]
            ep["t_end"] = e["t_end"]
            ep["dur"] += e.get("duration_s", 0.0)
            if e.get("duration_s", 0) > ep["rep"].get("duration_s", 0):
                ep["rep"] = e
        else:
            eps.append({"object": e["object"], "t_start": e["t_start"],
                        "t_end": e["t_end"], "dur": e.get("duration_s", 0.0),
                        "rep": e})
    return [p for p in eps if p["dur"] >= min_dur]


def lcs_align(seq, eps):
    """卡序 × 注视段 最长公共子序列对齐:命中保序;未匹配卡项=缺失,
    未匹配注视段=多余(用户瞟错/系统误绑,人工核录像)。"""
    n, m = len(seq), len(eps)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = (1 + L[i + 1][j + 1]) if seq[i] == eps[j]["object"] \
                else max(L[i + 1][j], L[i][j + 1])
    pairs, i, j = {}, 0, 0
    while i < n and j < m:
        if seq[i] == eps[j]["object"]:
            pairs[i] = j; i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def runs_of(seq):
    """卡序压成连续同名段 [(名, 连击数)]:LL/RR 这类成对项,用户两次盯之间
    眼睛常不离场,感知只出一段——按段对齐,融合段凭时长顶多项。"""
    out = []
    for nm in seq:
        if out and out[-1][0] == nm:
            out[-1][1] += 1
        else:
            out.append([nm, 1])
    return out


def run(intents: Path, seq: list[str], map_dir: Path, out_csv: Path | None,
        merge_gap=1.2, min_dur=1.0, double_dwell=7.0):
    named = load_named(map_dir)
    ev = finals(intents)
    eps = episodes(ev, merge_gap, min_dur)
    runs = runs_of(seq)
    pairs = lcs_align([r[0] for r in runs], eps)
    used = set(pairs.values())
    rows, t0 = [], (eps[0]["t_start"] if eps else 0.0)

    def row(k, want, ep, verdict):
        e = ep["rep"] if ep else {}
        th = theta_min(e.get("origin_world"), want if want != "--" else
                       (ep["object"] if ep else ""), named)
        rows.append({
            "k": k, "want": want, "got": ep["object"] if ep else "--",
            "verdict": verdict,
            "dur_s": round(ep["dur"], 1) if ep else "",
            "vote": round(e.get("vote_share", 0.0), 2) if e else "",
            "dist_m": round(e.get("distance_m", 0.0), 2) if e else "",
            "theta_min_deg": round(th, 2) if th is not None else "",
            "t": round(ep["t_start"] - t0, 1) if ep else "",
        })

    j_extra = 0
    ok = miss = 0
    kbase = 1
    for i, (want, k) in enumerate(runs):
        j = pairs.get(i)
        while j is not None and j_extra < j:          # 之前夹着的多余段
            if j_extra not in used:
                row("+", "--", eps[j_extra], "＋多余")
            j_extra += 1
        if j is None:
            row(f"{kbase}" + (f"-{kbase + k - 1}" if k > 1 else ""),
                want + (f"×{k}" if k > 1 else ""), None, "✗缺失")
            miss += k
        else:
            # 连击段三种命中形态:融合单段(时长≥门)/ 两段分立(紧邻同名)/ 只到一半
            credit, jend = 1, j
            if k > 1:
                if eps[j]["dur"] >= double_dwell:
                    credit = k
                elif (j + 1 < len(eps) and j + 1 not in used
                      and eps[j + 1]["object"] == want):
                    credit, jend = k, j + 1
                    used.add(j + 1)
            else:
                credit = 1
            mark = "✓" * min(credit, 2) + ("(-1)" if credit < k else "")
            ep_show = dict(eps[j])
            if jend != j:  # 两段分立:表里合并显示,时长相加
                ep_show["dur"] = eps[j]["dur"] + eps[jend]["dur"]
                ep_show["t_end"] = eps[jend]["t_end"]
            row(f"{kbase}" + (f"-{kbase + k - 1}" if k > 1 else ""),
                want + (f"×{k}" if k > 1 else ""), ep_show, mark)
            ok += credit
            miss += k - credit
            j_extra = jend + 1
        kbase += k
    for j in range(j_extra, len(eps)):
        if j not in used:
            row("+", "--", eps[j], "＋多余")

    w = {"k": 3, "want": 6, "got": 6, "verdict": 5, "dur_s": 5, "vote": 5,
         "dist_m": 6, "theta_min_deg": 7, "t": 7}
    print("  ".join(f"{h:>{w[h]}}" for h in rows[0]))
    for r in rows:
        print("  ".join(f"{str(r[h]):>{w[h]}}" for h in r))
    extra = len(eps) - len(pairs)
    ths = [r["theta_min_deg"] for r in rows
           if r["verdict"].startswith("✓") and r["theta_min_deg"] != ""]
    med = f";θ_min 中位 {np.median(ths):.2f}°" if ths else ""
    print(f"\n命中 {ok}/{len(seq)},缺失 {miss},多余注视 {extra}"
          f"(合并 {len(ev)}→{len(eps)} 段,gap<{merge_gap}s,dur≥{min_dur}s,"
          f"连击融合门 {double_dwell}s){med}")
    if out_csv:
        import csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=list(rows[0]))
            cw.writeheader()
            cw.writerows(rows)
        print(f"CSV -> {out_csv}")
    return ok, len(seq)


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    (td / "names.json").write_text(json.dumps({"1": "球L", "2": "球M", "3": "苹果"}))
    (td / "instances.json").write_text(json.dumps({"instances": [
        {"id": 1, "centroid": [0.0, 0.0, 0.8]},
        {"id": 2, "centroid": [0.3, 0.0, 0.8]},
        {"id": 3, "centroid": [0.0, 1.0, 0.8]},
    ]}))
    F = dict(provisional=False, vote_share=0.9, origin_world=[0, -3, 1.6], distance_m=3.1)
    ev = [
        {"object": "球L", "t_start": 0.0, "t_end": 2.0, "duration_s": 2.0, **F},
        {"object": "floor", "t_start": 2.2, "t_end": 2.6, "duration_s": 0.4,
         "provisional": False},                                     # 背景,丢
        {"object": "球M", "t_start": 2.8, "t_end": 3.0, "duration_s": 0.2,
         "provisional": True},                                      # 非 final,丢
        {"object": "球M", "t_start": 3.0, "t_end": 3.8, "duration_s": 0.8, **F},
        {"object": "球M", "t_start": 4.1, "t_end": 5.0, "duration_s": 0.9, **F},  # 与上合并 ->1.7s
        {"object": "苹果", "t_start": 5.2, "t_end": 5.5, "duration_s": 0.3, **F},   # 扫视碎片,丢
        {"object": "香蕉", "t_start": 6.0, "t_end": 8.0, "duration_s": 2.0, **F},   # 多余注视
    ]
    (td / "names.json").write_text(json.dumps(
        {"1": "球L", "2": "球M", "3": "苹果", "4": "香蕉"}))
    (td / "instances.json").write_text(json.dumps({"instances": [
        {"id": 1, "centroid": [0.0, 0.0, 0.8]},
        {"id": 2, "centroid": [0.3, 0.0, 0.8]},
        {"id": 3, "centroid": [0.0, 1.0, 0.8]},
        {"id": 4, "centroid": [0.5, 1.0, 0.8]},
    ]}))
    ij = td / "intents.jsonl"
    ij.write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    seq = [TOKEN.get(t, t) for t in ["L", "球M"]]
    assert seq == ["球L", "球M"]
    ok, n = run(ij, seq, td, None)
    assert (ok, n) == (2, 2), (ok, n)   # 合并后 L/M 全中,香蕉记多余,苹果碎片被丢
    print("selftest OK(过滤+同物合并+时长门+LCS 对齐+多余段)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("intents", nargs="?", help="gaze_live --log 落盘的 intents.jsonl")
    ap.add_argument("--seq", default="", help="卡片序列,空格或逗号分隔;L/M/R 是网球简写")
    ap.add_argument("--card", default="", help="卡号 e1-e5/s1-s7(取 e1_cards.py 预注册序列)")
    ap.add_argument("--map-dir", default=str(SCENE / "lab_result/segmentation_sam"))
    ap.add_argument("--csv", default=None, help="逐项结果另存 CSV")
    ap.add_argument("--merge-gap", type=float, default=1.2,
                    help="同物 final 间隔小于此秒数合并为一段注视")
    ap.add_argument("--min-dur", type=float, default=1.0,
                    help="累计注视短于此秒数的段当扫视碎片丢弃")
    ap.add_argument("--double-dwell", type=float, default=7.0,
                    help="卡上连续同名对被融合成一段时,累计注视≥此秒数记满两项"
                         "(实测单项最长 6.7s、融合对最短 9.0s)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.intents or not (a.seq or a.card):
        ap.error("需要 intents.jsonl 与 --seq/--card(或 --selftest)")
    if a.card:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from e1_cards import CARDS
        if a.card not in CARDS:
            ap.error(f"卡号 {a.card} 不存在:{list(CARDS)}")
        seq = CARDS[a.card][1]
    else:
        seq = [TOKEN.get(t, t) for t in a.seq.replace(",", " ").split()]
    run(Path(a.intents), seq, Path(a.map_dir), Path(a.csv) if a.csv else None,
        merge_gap=a.merge_gap, min_dur=a.min_dur, double_dwell=a.double_dwell)


if __name__ == "__main__":
    main()
