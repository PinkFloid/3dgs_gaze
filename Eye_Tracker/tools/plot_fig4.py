#!/usr/bin/env python3
"""plot_fig4.py -- docs/E1_DATA/trials.csv -> 论文正式版 Fig.4(IEEE 单栏)。

    python Eye_Tracker/tools/plot_fig4.py

产物:docs/E1_DATA/fig4.pdf(投稿用矢量)/ fig4.png(300dpi 预览)。
与 collect_e1.py 同一口径重新分箱(hit/miss、去 stress 与 walking、θ 非空;09-04 起 θ 为结果盲的单元 θ),并对
curve.csv 做一致性校验;额外算 Wilson 95% CI,顺带把逐箱数字打印出来
(正文 TBD{E1-near}/TBD{E1-far} 直接抄)。σ 竖线与遮挡带是常量,改这里。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/E1_DATA"

BINS = [0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0, 20.0]   # 与 collect_e1.py 一致
SIGMA_DEG = 1.0        # 论文口径:标定注视锥 σ≈1°(戳实测 0.7-0.85°)
OCC_LIMIT_DEG = 0.96   # 物理遮挡极限:球径 6.7cm @ 4m(低 θ 站位)
CHANCE = 1 / 3         # 三球卡乱猜
Z = 1.959964           # 95%

GREEN = "#4a7000"
RED = "#b23232"
GRAY = "#555555"


def wilson(hits: int, n: int) -> tuple[float, float]:
    p = hits / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return center - half, center + half


def load_bins():
    trials = [r for r in csv.DictReader((DATA / "trials.csv").open(encoding="utf-8"))
              if r["outcome"] in ("hit", "miss")
              and not ({"stress", "walking"} & set(r["tags"].split("|")))   # 主集口径,与 collect_e1 一致
              and r["theta_deg"]]
    th = np.array([float(r["theta_deg"]) for r in trials])
    ok = np.array([r["outcome"] == "hit" for r in trials])
    rows = []
    for lo, hi in zip(BINS, BINS[1:]):
        m = (th >= lo) & (th < hi)
        if m.sum() == 0:
            continue
        rows.append(dict(x=float(np.sqrt(lo * hi)), lo=lo, hi=hi,
                         n=int(m.sum()), hits=int(ok[m].sum())))
    # 与 collect_e1.py 产出的 curve.csv 校验(n/hits 必须一致)
    ref = list(csv.DictReader((DATA / "curve.csv").open(encoding="utf-8")))
    assert len(ref) == len(rows), "分箱数与 curve.csv 不一致,先重跑 collect_e1.py"
    for a, b in zip(rows, ref):
        assert (a["n"], a["hits"]) == (int(b["n"]), int(b["hits"])), \
            f"箱 [{a['lo']},{a['hi']}) 与 curve.csv 不一致,先重跑 collect_e1.py"
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    rows = load_bins()
    xs = [r["x"] for r in rows]
    accs = [r["hits"] / r["n"] for r in rows]
    ci = [wilson(r["hits"], r["n"]) for r in rows]
    err_lo = [a - lo for a, (lo, _) in zip(accs, ci)]
    err_hi = [hi - a for a, (_, hi) in zip(accs, ci)]

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    # 遮挡带 + σ 竖线 + chance 线(全部直接标注,不用图例)
    ax.axvspan(0.45, OCC_LIMIT_DEG, color=RED, alpha=0.08, lw=0, zorder=0)
    ax.text(np.sqrt(0.45 * OCC_LIMIT_DEG), 1.005, "past occlusion\nlimit (4 m)",
            fontsize=6.5, color=RED, ha="center", va="top")
    ax.axvline(SIGMA_DEG, ls="-.", lw=0.7, color=GRAY, zorder=1)
    ax.text(SIGMA_DEG * 1.06, 0.06, r"calibrated $\sigma\approx1^\circ$",
            fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.axhline(CHANCE, ls=":", lw=0.8, color=RED, zorder=1)
    ax.text(11.5, CHANCE + 0.025, "3-ball chance", fontsize=6.5, color=RED,
            ha="right", va="bottom")

    ax.errorbar(xs, accs, yerr=[err_lo, err_hi], fmt="o-", color=GREEN,
                lw=1.0, ms=3.2, elinewidth=0.7, capsize=1.5, capthick=0.7,
                zorder=3)
    for x, a, (lo, _), r in zip(xs, accs, ci, rows):
        ax.annotate(f"{r['n']}", (x, lo), textcoords="offset points",
                    xytext=(0, -8), ha="center", fontsize=6, color="#888888")

    ax.set_xscale("log")
    ax.set_xlim(0.45, 13)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Min. angular separation to nearest named object, $\theta$ (deg)")
    ax.set_ylabel("Top-1 binding accuracy")
    ax.grid(alpha=0.18, lw=0.4, which="major")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    fig.savefig(DATA / "fig4.pdf")
    fig.savefig(DATA / "fig4.png", dpi=300)

    print("bin [lo,hi)      n  hits   acc   Wilson95")
    for r, a, (lo, hi) in zip(rows, accs, ci):
        print(f"[{r['lo']:4},{r['hi']:4})  {r['n']:4}  {r['hits']:4}  "
              f"{a:5.1%}  [{lo:5.1%}, {hi:5.1%}]")
    n = sum(r["n"] for r in rows)
    h = sum(r["hits"] for r in rows)
    print(f"total {n} trials, {h} hits ({h/n:.1%})")
    print(f"-> {DATA}/fig4.pdf fig4.png")


if __name__ == "__main__":
    main()
