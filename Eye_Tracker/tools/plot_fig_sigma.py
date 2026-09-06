#!/usr/bin/env python3
"""plot_fig_sigma.py -- docs/E1_DATA/{sigma_curve,theta_bins}.json -> 锥宽 σ 扫描图(IEEE 单栏)。

    python Eye_Tracker/tools/plot_fig_sigma.py

产物:docs/E1_DATA/fig_sigma.pdf|png(全部 trial)、fig_sigma_clear.pdf|png(剔物理遮挡站位)。
三条 θ 档命中率(Wilson 95% CI)+ 响应率虚线(有目标判定的 final 占比,与 trial 子集无关)
+ σ_cal 竖线 + 三球乱猜线。样式与 plot_fig4.py 一致。

数据口径见 docs/CONE_POSTERIOR_V2.md「锥宽 σ 扫描」:v2 管线、半角 2σ、像素角尺寸固定。
分档(<1° / 1–2.5° / ≥2.5°)若在 Linux 侧按 09-01 结果盲 θ 重跑 e4_table.py 后变化,
重跑本脚本即可;各 σ 的总体与响应率不受分箱影响。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/E1_DATA"

SIGMA_CAL = 1.0        # 论文口径:标定注视锥 σ≈1°(戳实测 0.7-0.85°)
CHANCE = 1 / 3         # 三球卡乱猜
Z = 1.959964           # 95%

GREEN = "#4a7000"
AMBER = "#b8730f"
RED = "#b23232"
GRAY = "#555555"

TIERS = [("≥2.5°", 2.5, 99.0, GREEN, "o-", r"$\theta\geq2.5^\circ$"),
         ("1.0–2.5°", 1.0, 2.5, AMBER, "D-", r"$1^\circ\leq\theta<2.5^\circ$"),
         ("<1.0°", 0.0, 1.0, RED, "s-", r"$\theta<1^\circ$")]


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load():
    curve = sorted(json.loads((DATA / "sigma_curve.json").read_text(encoding="utf-8")),
                   key=lambda c: c["sigma"])
    bins = json.loads((DATA / "theta_bins.json").read_text(encoding="utf-8"))
    return curve, bins


def tier_rows(curve, bins, subset: str):
    """-> {tier: [(sigma, hits, n)]};subset='all' 取 sigma_curve 的 tiers,'no_occ' 从 theta_bins 聚合。"""
    out = {t[0]: [] for t in TIERS}
    for c in curve:
        for name, lo, hi, *_ in TIERS:
            if subset == "all":
                t = c["tiers"][name]
                out[name].append((c["sigma"], t["hit"], t["n"]))
            else:
                rows = [b for b in bins[c["cfg"]][subset] if b["lo"] >= lo and b["hi"] <= hi]
                out[name].append((c["sigma"], sum(b["hit"] for b in rows), sum(b["n"] for b in rows)))
    return out


def plot(curve, bins, subset: str, stem: str):
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

    rows = tier_rows(curve, bins, subset)
    sig = [c["sigma"] for c in curve]

    fig, ax = plt.subplots(figsize=(3.45, 2.7))

    ax.axvline(SIGMA_CAL, ls="-.", lw=0.7, color=GRAY, zorder=1)
    ax.text(SIGMA_CAL * 1.05, 0.03, r"calibrated $\sigma_{\rm cal}\approx1^\circ$",
            fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.axhline(CHANCE, ls=":", lw=0.8, color=RED, zorder=1)
    ax.text(0.56, CHANCE - 0.015, "3-ball chance", fontsize=6.5, color=RED,
            ha="left", va="top")
    ax.text(0.168, 1.035, r"$\approx$ single ray", fontsize=6.5, color=GRAY, ha="left", va="top")
    ax.text(4.75, 1.035, "neighbours\nenter cone", fontsize=6.5, color=GRAY, ha="right", va="top")

    for name, lo, hi, color, fmt, label in TIERS:
        xs = [r[0] for r in rows[name]]
        accs = [r[1] / r[2] for r in rows[name]]
        ci = [wilson(r[1], r[2]) for r in rows[name]]
        ns = sorted({r[2] for r in rows[name]})
        n_txt = f"n={ns[0]}" if len(ns) == 1 else f"n={ns[0]}–{ns[-1]}"
        ax.errorbar(xs, accs,
                    yerr=[[a - c[0] for a, c in zip(accs, ci)], [c[1] - a for a, c in zip(accs, ci)]],
                    fmt=fmt, color=color, mfc=color, lw=1.0, ms=3.0,
                    elinewidth=0.6, capsize=1.3, capthick=0.6,
                    label=f"{label} ({n_txt})", zorder=3)

    ax.plot(sig, [c["yield_finals"] for c in curve], "^--", color=GRAY, mfc="white",
            lw=0.9, ms=3.0, label="fixation events with a verdict", zorder=2)

    hs, ls = ax.get_legend_handles_labels()
    order = [k for k, l in enumerate(ls) if "verdict" not in l] + [k for k, l in enumerate(ls) if "verdict" in l]
    ax.legend([hs[k] for k in order], [ls[k] for k in order],
              loc="lower left", fontsize=6, frameon=False, ncol=2,
              handlelength=1.8, columnspacing=1.2, borderaxespad=0.0,
              bbox_to_anchor=(0.0, 1.01))   # 图外顶部两行,不压曲线

    ax.set_xscale("log")
    ax.set_xlim(0.16, 5.0)
    ax.set_ylim(0, 1.06)
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 1.5, 2.5, 4.0]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Gaze-cone scale $\sigma$ (deg)")
    ax.set_ylabel("Top-1 binding accuracy")
    ax.grid(alpha=0.18, lw=0.4, which="major")
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    fig.savefig(DATA / f"{stem}.pdf")
    fig.savefig(DATA / f"{stem}.png", dpi=300)
    plt.close(fig)

    print(f"\n[{stem}] subset={subset}")
    print("sigma   " + "  ".join(f"{t[0]:>10}" for t in TIERS) + "   verdict-rate")
    for i, c in enumerate(curve):
        cells = []
        for name, *_ in TIERS:
            s, h, n = rows[name][i]
            lo, hi = wilson(h, n)
            cells.append(f"{h:>3}/{n:<3}{h/n:5.0%}")
        print(f"{c['sigma']:<6}  " + "  ".join(f"{x:>10}" for x in cells) + f"   {c['yield_finals']:.0%}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    curve, bins = load()
    plot(curve, bins, "all", "fig_sigma_tiers")
    plot(curve, bins, "no_occ", "fig_sigma_clear")
    print("\nwrote", DATA / "fig_sigma_tiers.{pdf,png}", "and", DATA / "fig_sigma_clear.{pdf,png}")


if __name__ == "__main__":
    main()
