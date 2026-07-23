#!/usr/bin/env python3
"""eval_binding.py -- E1 指代消歧打分:brain 的 events.jsonl -> 每 trial 一行 CSV -> 汇总表。

用法(两段式,对齐 docs/EXPERIMENT_PLAN.md §1 的真值协议):

  # 1) 每个 session 打分一次:--expect 按指令卡顺序给出每个 deictic trial 的真值
  python Intension/eval_binding.py Intension/logs/<sess>/events.jsonl \
      --expect 杯A,杯B,杯A --sep 0.25 --dist 1.5 --n 3 --out e1_results.csv

  # 2) 汇总出"角间隔-准确率"表(Fig.4 的数据)
  python Intension/eval_binding.py --table e1_results.csv

trial 定义:每条 kind=deictic 的 command 事件 = 一个 trial(全记录全报告,不许挑);
系统选择 = 其后 2s 内的 resolution 事件(mode=视线);没有 resolution = miss,同样入表。
角间隔 = 2*atan(间距/2/距离),CI 用 Wilson 95%。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

FIELDS = ["session", "trial", "t_word", "expected", "chosen", "correct", "mode",
          "cand1", "cand1_gap", "cand2", "cand2_gap",
          "n_inst", "sep_m", "dist_m", "ang_deg"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def score_session(path, expect, sep, dist, n_inst):
    """一份 events.jsonl -> trial 行列表(按 deictic command 顺序对齐指令卡)。"""
    evs = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    ang = math.degrees(2 * math.atan(sep / 2.0 / dist)) if sep and dist else 0.0
    rows, k = [], 0
    for i, e in enumerate(evs):
        if e.get("topic") != "command" or e.get("kind") != "deictic":
            continue
        if k >= len(expect):
            print(f"[!] {path}: 第 {k + 1} 个 deictic trial 超出 --expect 长度,忽略之后的")
            break
        t = float(e.get("t", 0.0))
        chosen, mode, binding = None, "", None
        for e2 in evs[i + 1:]:
            t2 = float(e2.get("t", e2.get("t_word", 0.0)))
            if t2 - t > 2.0:
                break
            if e2.get("topic") == "binding" and binding is None:
                binding = e2.get("candidates") or []
            if e2.get("topic") == "resolution" and not e2.get("goto"):
                chosen, mode = e2.get("object"), e2.get("mode", "")
                break
        c1 = binding[0] if binding else {}
        c2 = binding[1] if binding and len(binding) > 1 else {}
        rows.append({"session": Path(path).parent.name, "trial": k + 1,
                     "t_word": round(t, 2), "expected": expect[k],
                     "chosen": chosen or "", "correct": int(chosen == expect[k]),
                     "mode": mode,
                     "cand1": c1.get("object", ""), "cand1_gap": c1.get("gap", ""),
                     "cand2": c2.get("object", ""), "cand2_gap": c2.get("gap", ""),
                     "n_inst": n_inst, "sep_m": sep, "dist_m": dist,
                     "ang_deg": round(ang, 2)})
        k += 1
    if k < len(expect):
        print(f"[!] {path}: 指令卡 {len(expect)} 条,只找到 {k} 个 deictic trial")
    return rows


def print_table(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("空表")
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["n_inst"], r["ang_deg"])].append(int(r["correct"]))
    print(f"{'N':>3} {'角间隔°':>8} {'trials':>7} {'acc':>6} {'Wilson95%':>16} {'1/N基线':>8}")
    for (n, ang), cs in sorted(by.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        p, lo, hi = wilson(sum(cs), len(cs))
        base = 1.0 / float(n) if float(n) > 0 else 0.0
        print(f"{n:>3} {ang:>8} {len(cs):>7} {p:>6.2f} [{lo:>5.2f},{hi:>5.2f}]   {base:>8.2f}")
    total = [int(r["correct"]) for r in rows]
    p, lo, hi = wilson(sum(total), len(total))
    print(f"\n总计 {len(total)} trials  acc {p:.2f}  Wilson95% [{lo:.2f},{hi:.2f}]")
    miss = [r for r in rows if not r["chosen"]]
    if miss:
        print(f"其中 miss(未产生 resolution){len(miss)} 个 —— 按失败计入,不许剔除")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="*", help="brain 会话的 events.jsonl(可多个)")
    ap.add_argument("--expect", default="", help="逗号分隔的真值序列(按指令卡顺序)")
    ap.add_argument("--sep", type=float, default=0.0, help="相邻实例间距 (m)")
    ap.add_argument("--dist", type=float, default=0.0, help="用户到物体距离 (m)")
    ap.add_argument("--n", type=int, default=0, help="同类实例数 N")
    ap.add_argument("--out", default="e1_results.csv", help="追加写入的结果 CSV")
    ap.add_argument("--table", default=None, help="只汇总:读结果 CSV 打印角间隔-准确率表")
    args = ap.parse_args()

    if args.table:
        print_table(args.table)
        return 0
    if not args.events or not args.expect:
        ap.error("需要 events.jsonl 与 --expect(或用 --table 汇总)")
    expect = [s for s in args.expect.split(",") if s]
    rows = []
    for p in args.events:
        rows += score_session(p, expect, args.sep, args.dist, args.n)
    new = not Path(args.out).exists()
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    ok = sum(r["correct"] for r in rows)
    print(f"{len(rows)} trials -> {args.out}  (对 {ok} 错 {len(rows) - ok})")
    for r in rows:
        mark = "o" if r["correct"] else "x"
        print(f"  [{mark}] t={r['t_word']:>7} 期望 {r['expected']:<8} 选了 {r['chosen'] or '(miss)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
