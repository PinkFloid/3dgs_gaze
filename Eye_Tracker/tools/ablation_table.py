#!/usr/bin/env python3
"""ablation_table.py -- 消融汇总表(09-07 口径:现行 v2 管线、逐项时段多数判定、逐 trial θ、逐 trial 遮挡)。

    conda run -n nerfstudio python Eye_Tracker/tools/ablation_table.py [--dir docs/E1_DATA/ablation_v10cfg] [--out docs/E1_DATA]

输入:<dir>/trials_theta_<cfg>.csv(trial_theta.py --log intents_abl_<cfg>.jsonl --tag <cfg> 的产物)。
每配置一行:全部(站立,去 stress)不设闸正确率;按 θ_trial 三档(≥2.5° / 1–2.5° / <1°);清晰 / 有遮挡;边走;设闸版正确率。
产物:<out>/table2_v10cfg.csv 与 .md。
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORDER = [("full", "Full(现行方法,σ=1°)"), ("s01", "≈单射线 σ=0.1°"), ("s02", "σ=0.2°"), ("s05", "σ=0.5°"),
         ("s15", "σ=1.5°"), ("s25", "σ=2.5°"), ("s40", "σ=4°"),
         ("sphere", "球体投票(视角无关,无遮挡)"), ("mass", "去面积归一(按锥质量份额排序)"),
         ("table", "物品台入候选集"), ("cluster010", "固定半径聚类 0.1 m"),
         ("vis", "可见性=z 检验(无最近高斯认领)"), ("noocc", "忽略遮挡(逐实例单独渲染)")]


def acc(rows, key="outcome"):
    n = len(rows)
    if not n:
        return "—"
    h = sum(1 for r in rows if (r[key] == "hit" if key == "outcome" else r[key] == "correct"))
    return f"{h}/{n} ({h / n:.0%})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(ROOT / "docs/E1_DATA/ablation_v10cfg"))
    ap.add_argument("--out", default=str(ROOT / "docs/E1_DATA"))
    a = ap.parse_args()
    d = Path(a.dir)
    lines_md = ["| 配置 | 全部 不设闸 | ≥2.5° | 1–2.5° | <1° | 清晰 | 有遮挡 | 边走 | 全部 设闸 |", "|---|---|---|---|---|---|---|---|---|"]
    rows_csv = []
    for cfg, desc in ORDER:
        p = d / f"trials_theta_{cfg}.csv"
        if not p.exists():
            lines_md.append(f"| {desc} | (缺 {p.name}) | | | | | | | |")
            continue
        rows = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r["theta_trial"] != "" and "stress" not in r["tags"]]
        stand = [r for r in rows if r["walking"] == "0"]
        walk = [r for r in rows if r["walking"] == "1"]
        th = lambda r: float(r["theta_trial"])
        t_hi = [r for r in stand if th(r) >= 2.5]
        t_mid = [r for r in stand if 1.0 <= th(r) < 2.5]
        t_lo = [r for r in stand if th(r) < 1.0]
        clear = [r for r in stand if r["occluded"] in ("", "0", "0.0")]
        occl = [r for r in stand if r["occluded"] not in ("", "0", "0.0")]
        cells = [acc(stand), acc(t_hi), acc(t_mid), acc(t_lo), acc(clear), acc(occl), acc(walk), acc(stand, "gated")]
        lines_md.append(f"| {desc} | " + " | ".join(cells) + " |")
        rows_csv.append({"cfg": cfg, "desc": desc, "all": cells[0], "tier_hi": cells[1], "tier_mid": cells[2], "tier_lo": cells[3],
                         "clear": cells[4], "occluded": cells[5], "walking": cells[6], "all_gated": cells[7],
                         "n_stand": len(stand), "n_walk": len(walk),
                         "outcome": dict(Counter(r["dwell"] for r in stand))})
    out = Path(a.out)
    (out / "table2_v10cfg.md").write_text(
        "# Table II(09-07 口径):现行 v2 管线上的消融\n\n"
        "口径:v7–v10 全部 27 条录像按现行配置重放;判定 = 每项时段内输出时长最多的物体(不设闸;末列为设闸);"
        "θ = 该项时段内定位流头位到目标与最近命名物的张角(逐 trial);遮挡 = 从该头位看目标被别的候选物挡住(逐 trial 射线判定);"
        "'全部/三档/清晰/遮挡' 为站立录像(去 e2 压力段),'边走' 单列。\n\n" + "\n".join(lines_md) + "\n", encoding="utf-8")
    with (out / "table2_v10cfg.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0])) if rows_csv else None
        if w:
            w.writeheader(); w.writerows(rows_csv)
    print("\n".join(lines_md)); print(f"-> {out}/table2_v10cfg.md .csv")


if __name__ == "__main__":
    main()
