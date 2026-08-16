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
TOKEN = {"L": "网球L", "M": "网球M", "R": "网球R"}
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


def run(intents: Path, seq: list[str], map_dir: Path, out_csv: Path | None):
    named = load_named(map_dir)
    ev = finals(intents)
    n = min(len(seq), len(ev))
    if len(ev) != len(seq):
        print(f"[!] final 数 {len(ev)} ≠ 卡项数 {len(seq)}:只对前 {n} 项,"
              f"多余的{'注视' if len(ev) > len(seq) else '卡项'}列在表尾")
    rows, ok = [], 0
    for k in range(max(len(seq), len(ev))):
        want = seq[k] if k < len(seq) else "--"
        e = ev[k] if k < len(ev) else {}
        got = e.get("object", "--")
        hit = want == got
        ok += hit and k < n
        th = theta_min(e.get("origin_world"), want, named)
        rows.append({
            "k": k + 1, "want": want, "got": got,
            "verdict": "✓" if hit else "✗",
            "vote": round(e.get("vote_share", 0.0), 2) if e else "",
            "dist_m": round(e.get("distance_m", 0.0), 2) if e else "",
            "theta_min_deg": round(th, 2) if th is not None else "",
            "t_end": round(e.get("t_end", 0.0), 2) if e else "",
        })
    w = {"k": 3, "want": 6, "got": 6, "verdict": 3, "vote": 5, "dist_m": 6,
         "theta_min_deg": 7, "t_end": 10}
    print("  ".join(f"{h:>{w[h]}}" for h in rows[0]))
    for r in rows:
        print("  ".join(f"{str(r[h]):>{w[h]}}" for h in r))
    ths = [r["theta_min_deg"] for r in rows[:n] if r["theta_min_deg"] != ""]
    med = f";θ_min 中位 {np.median(ths):.2f}°" if ths else ""
    print(f"\n{ok}/{n} 对{med}")
    if out_csv:
        import csv
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=list(rows[0]))
            cw.writeheader()
            cw.writerows(rows)
        print(f"CSV -> {out_csv}")
    return ok, n


def selftest():
    import tempfile
    td = Path(tempfile.mkdtemp())
    (td / "names.json").write_text(json.dumps({"1": "网球L", "2": "网球M", "3": "苹果"}))
    (td / "instances.json").write_text(json.dumps({"instances": [
        {"id": 1, "centroid": [0.0, 0.0, 0.8]},
        {"id": 2, "centroid": [0.3, 0.0, 0.8]},
        {"id": 3, "centroid": [0.0, 1.0, 0.8]},
    ]}))
    ev = [
        {"object": "网球L", "provisional": False, "vote_share": 0.9,
         "origin_world": [0, -3, 1.6], "distance_m": 3.1, "t_end": 1.0},
        {"object": "floor", "provisional": False, "t_end": 1.5},          # 该被丢
        {"object": "网球M", "provisional": True, "t_end": 1.7},            # 非 final,丢
        {"object": "苹果", "provisional": False, "vote_share": 0.8,
         "origin_world": [0, -3, 1.6], "distance_m": 4.0, "t_end": 2.0},
    ]
    ij = td / "intents.jsonl"
    ij.write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    seq = [TOKEN.get(t, t) for t in ["L", "网球M"]]
    assert seq == ["网球L", "网球M"]
    ok, n = run(ij, seq, td, None)
    assert (ok, n) == (1, 2), (ok, n)   # 第2项 want 网球M got 苹果 -> ✗
    print("selftest OK(floor/provisional 过滤 + 严格按序 + 命中判定)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("intents", nargs="?", help="gaze_live --log 落盘的 intents.jsonl")
    ap.add_argument("--seq", default="", help="卡片序列,空格或逗号分隔;L/M/R 是网球简写")
    ap.add_argument("--card", default="", help="卡号 e1-e5/s1-s7(取 e1_cards.py 预注册序列)")
    ap.add_argument("--map-dir", default=str(SCENE / "lab_result/segmentation_sam"))
    ap.add_argument("--csv", default=None, help="逐项结果另存 CSV")
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
    run(Path(a.intents), seq, Path(a.map_dir), Path(a.csv) if a.csv else None)


if __name__ == "__main__":
    main()
