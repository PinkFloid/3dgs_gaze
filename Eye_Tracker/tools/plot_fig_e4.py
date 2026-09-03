#!/usr/bin/env python3
"""plot_fig_e4.py -- docs/E1_DATA/{theta_bins,sigma_curve}.json -> 论文两张图(IEEE 单栏,plot_fig4 同版式)。

    python Eye_Tracker/tools/plot_fig_e4.py

fig_theta.pdf/png:成功率对 θ(没答与答错同算失败,主集 204 项,剔压力段与边走)。
    曲线:ours(v2, σ=1°)、single ray(σ=0.2°,同管线锥收成射线)、nearest-centroid 基线、
    sphere vote(视角无关球投票,theta_bins.json 里有 v2sphere 时才画)。σ 竖线、乱猜线、遮挡带直接标注。
fig_sigma.pdf/png:成功率对锥宽 σ,按 θ 档三条线 + 有目标判定的 final 占比(灰虚线);σ=标定精度竖线。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/E1_DATA"
SIGMA_CAL = 1.0        # 标定精度 ≈ 1°;σ 扫描最优也在此
OCC_LIMIT_DEG = 0.96   # 物理遮挡极限:球径 6.7cm @ 4m
CHANCE = 1 / 3
GREEN, RED, GRAY, BLUE, ORANGE, PURPLE = "#4a7000", "#b23232", "#555555", "#2a5db0", "#c86a1e", "#6b4aa0"


def style(plt):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def series(bins, cfg, view="all"):
    rows = [b for b in bins[cfg][view] if b["n"]]
    x = [float(np.sqrt(b["lo"] * min(b["hi"], 12.0))) for b in rows]
    a = [b["rate"] for b in rows]
    lo = [b["rate"] - b["ci"][0] for b in rows]
    hi = [b["ci"][1] - b["rate"] for b in rows]
    return x, a, [lo, hi], rows


def fig_theta(plt, bins):
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.axvspan(0.45, OCC_LIMIT_DEG, color=RED, alpha=0.08, lw=0, zorder=0)
    ax.text(np.sqrt(0.45 * OCC_LIMIT_DEG), 1.005, "past occlusion\nlimit (4 m)",
            fontsize=6.5, color=RED, ha="center", va="top")
    ax.axvline(SIGMA_CAL, ls="-.", lw=0.7, color=GRAY, zorder=1)
    ax.text(SIGMA_CAL * 1.06, 0.06, r"calibrated $\sigma\approx1^\circ$", fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.axhline(CHANCE, ls=":", lw=0.8, color=RED, zorder=1)
    ax.text(11.5, CHANCE + 0.025, "3-ball chance", fontsize=6.5, color=RED, ha="right", va="bottom")
    lines = [("v2s10", r"ours (cone, $\sigma=1^\circ$)", GREEN, "o-", 3),
             ("v2s02", r"single ray ($\sigma=0.2^\circ$)", BLUE, "s--", 2),
             ("naive", "nearest centroid", GRAY, "^:", 2)]
    if "v2sphere" in bins:
        lines.insert(2, ("v2sphere", "sphere vote (view-independent)", ORANGE, "d--", 2))
    for cfg, lab, col, fmt, z in lines:
        x, a, err, rows = series(bins, cfg)
        ax.errorbar(x, a, yerr=err, fmt=fmt, color=col, lw=0.9, ms=3.0, elinewidth=0.6,
                    capsize=1.2, capthick=0.6, zorder=z, label=lab)
        if cfg == "v2s10":
            for xx, r in zip(x, rows):
                ax.annotate(f"{r['n']}", (xx, 0.0), textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=5.5, color="#888888")
    ax.set_xscale("log"); ax.set_xlim(0.45, 13); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Min. angular separation to nearest named object, $\theta$ (deg)")
    ax.set_ylabel("Success rate (no answer = failure)")
    ax.grid(alpha=0.18, lw=0.4, which="major"); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2, borderaxespad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(DATA / "fig_theta.pdf"); fig.savefig(DATA / "fig_theta.png", dpi=300)
    for cfg, lab, *_ in lines:
        x, a, err, rows = series(bins, cfg)
        print(f"fig_theta {cfg:8}: " + "  ".join(f"[{r['lo']},{r['hi']}) {r['hit']}/{r['n']}" for r in rows))


def fig_sigma(plt, curve):
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    sig = [c["sigma"] for c in curve]
    ax.axvline(SIGMA_CAL, ls="-.", lw=0.7, color=GRAY, zorder=1)
    ax.text(SIGMA_CAL * 1.05, 0.04, r"calibrated $\sigma\approx1^\circ$", fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.axhline(CHANCE, ls=":", lw=0.8, color=RED, zorder=1)
    ax.text(4.3, CHANCE + 0.025, "3-ball chance", fontsize=6.5, color=RED, ha="right", va="bottom")
    for tier, lab, col, fmt in [("≥2.5°", r"$\theta\geq2.5^\circ$", GREEN, "o-"),
                                ("1.0–2.5°", r"$1^\circ\leq\theta<2.5^\circ$", BLUE, "s-"),
                                ("<1.0°", r"$\theta<1^\circ$", ORANGE, "^-")]:
        a = [c["tiers"][tier]["rate"] for c in curve]
        lo = [c["tiers"][tier]["rate"] - c["tiers"][tier]["ci"][0] for c in curve]
        hi = [c["tiers"][tier]["ci"][1] - c["tiers"][tier]["rate"] for c in curve]
        ax.errorbar(sig, a, yerr=[lo, hi], fmt=fmt, color=col, lw=0.9, ms=3.0, elinewidth=0.6,
                    capsize=1.2, capthick=0.6, zorder=3, label=lab)
    ax.plot(sig, [c["yield_finals"] for c in curve], ls="--", lw=0.8, color=GRAY, marker="x", ms=3,
            zorder=2, label="fixations with a verdict")
    ax.set_xscale("log"); ax.set_xlim(0.16, 5); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(FixedLocator(sig))
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Cone width $\sigma$ (deg); half-angle $2\sigma$")
    ax.set_ylabel("Success rate")
    ax.grid(alpha=0.18, lw=0.4, which="major"); ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, handlelength=2.2, borderaxespad=0.3, ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(DATA / "fig_sigma.pdf"); fig.savefig(DATA / "fig_sigma.png", dpi=300)
    for c in curve:
        print(f"fig_sigma σ={c['sigma']:3}: overall {c['overall']['hit']}/{c['overall']['n']}  " +
              "  ".join(f"{k} {v['hit']}/{v['n']}" for k, v in c["tiers"].items()) + f"  yield {c['yield_finals']:.0%}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    bins = json.load(open(DATA / "theta_bins.json", encoding="utf-8"))
    curve = json.load(open(DATA / "sigma_curve.json", encoding="utf-8"))
    fig_theta(plt, bins)
    fig_sigma(plt, curve)
    print("wrote fig_theta.{pdf,png} fig_sigma.{pdf,png} ->", DATA)


if __name__ == "__main__":
    main()
