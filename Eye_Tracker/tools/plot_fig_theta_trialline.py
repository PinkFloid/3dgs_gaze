#!/usr/bin/env python3
"""plot_fig_theta_trialline.py -- 逐 trial 线的 θ 曲线图(论文 Fig.,IEEE 单栏,plot_fig_e4.fig_theta 同版式)。

    conda run -n nerfstudio python Eye_Tracker/tools/plot_fig_theta_trialline.py [--ray-cfg ray] [--out paper/Figures]

数据:docs/E1_DATA/ablation_v10cfg/trials_theta_<cfg>.csv(09-07 口径:站立录像、去 stress;没答与答错同算失败)。
分箱:[<1°) [1,1.5) [1.5,2.5) [2.5,4) [4,20);Wilson 95% CI;底部标各箱 n。
曲线:ours(cone σ=1°,full)、single ray(默认 ray = 真·单射线;--ray-cfg s01 复现 09-09 的 ≈单射线 σ=0.1° 版)、
     sphere vote(sphere)、nearest centroid(naive)。--extra noangw,noscale 可把两条消融线也画上。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "docs/E1_DATA/ablation_v10cfg"
SIGMA_CAL = 1.0
CHANCE = 1 / 3
GREEN, RED, GRAY, BLUE, ORANGE, PURPLE, TEAL = "#4a7000", "#b23232", "#555555", "#2a5db0", "#c86a1e", "#6b4aa0", "#1f8a8a"
BINS = [(0.0, 1.0, 0.5), (1.0, 1.5, 1.0), (1.5, 2.5, 1.5), (2.5, 4.0, 2.5), (4.0, 20.0, 4.0)]   # (lo, hi, x_lo 用于几何中点)
Z = 1.959964
LABEL = {"ray": "single ray", "s01": r"single ray ($\sigma=0.1^\circ$)",
         "noangw": "w/o angular weighting", "noscale": "w/o scale normalization"}


def wilson(h, n):
    p = h / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - half, c + half


def trials(cfg):
    th, hit = [], []
    for r in csv.DictReader((D / f"trials_theta_{cfg}.csv").open(encoding="utf-8")):
        if r["theta_trial"] == "" or "stress" in r["tags"] or r["walking"] != "0":
            continue
        th.append(float(r["theta_trial"])); hit.append(r["outcome"] == "hit")
    return np.array(th), np.array(hit)


def series(cfg):
    th, hit = trials(cfg)
    x, a, lo, hi, rows = [], [], [], [], []
    for b_lo, b_hi, x_lo in BINS:
        m = (th >= b_lo) & (th < b_hi)
        n = int(m.sum())
        if not n:
            continue
        h = int(hit[m].sum()); p = h / n; ci = wilson(h, n)
        x.append(float(np.sqrt(x_lo * min(b_hi, 12.0)))); a.append(p); lo.append(p - ci[0]); hi.append(ci[1] - p)
        rows.append(dict(lo=b_lo, hi=b_hi, n=n, hit=h))
    return x, a, [lo, hi], rows


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ray-cfg", default="ray", help="单射线那条线用哪个配置:ray(真·单射线)或 s01(≈单射线 σ=0.1°)")
    ap.add_argument("--extra", default="", help="额外画的配置,逗号分隔,如 noangw,noscale")
    ap.add_argument("--out", default=str(ROOT / "paper/Figures"))
    ap.add_argument("--name", default="fig_theta")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    style(plt)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = [("full", r"ours (cone, $\sigma=1^\circ$)", GREEN, "o-", 3),
             (a.ray_cfg, LABEL.get(a.ray_cfg, a.ray_cfg), BLUE, "s--", 2),
             ("sphere", "sphere vote (view-independent)", ORANGE, "d--", 2),
             ("naive", "nearest centroid", GRAY, "^:", 2)]
    for i, cfg in enumerate([c for c in a.extra.split(",") if c]):
        lines.insert(1 + i, (cfg, LABEL.get(cfg, cfg), [PURPLE, TEAL][i % 2], ["v-", "p-"][i % 2], 2))
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.axvline(SIGMA_CAL, ls="-.", lw=0.7, color=GRAY, zorder=1)
    ax.text(SIGMA_CAL * 1.06, 0.06, r"calibrated $\sigma\approx1^\circ$", fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.axhline(CHANCE, ls=":", lw=0.8, color=RED, zorder=1)
    ax.text(11.5, CHANCE + 0.025, "3-ball chance", fontsize=6.5, color=RED, ha="right", va="bottom")
    for cfg, lab, col, fmt, z in lines:
        x, acc, err, rows = series(cfg)
        ax.errorbar(x, acc, yerr=err, fmt=fmt, color=col, lw=0.9, ms=3.0, elinewidth=0.6,
                    capsize=1.2, capthick=0.6, zorder=z, label=lab)
        if cfg == "full":
            for xx, r in zip(x, rows):
                ax.annotate(f"{r['n']}", (xx, 0.0), textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=5.5, color="#888888")
        print(f"{cfg:8}: " + "  ".join(f"[{r['lo']:g},{r['hi']:g}) {r['hit']}/{r['n']}" for r in rows))
    ax.set_xscale("log"); ax.set_xlim(0.45, 13); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Min. angular separation to nearest named object, $\theta$ (deg)")
    ax.set_ylabel("Success rate (no answer = failure)")
    ax.grid(alpha=0.18, lw=0.4, which="major"); ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2, borderaxespad=0.3)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / f"{a.name}.pdf"); fig.savefig(out / f"{a.name}.png", dpi=300)
    print("wrote", out / f"{a.name}.pdf|png")


if __name__ == "__main__":
    main()
