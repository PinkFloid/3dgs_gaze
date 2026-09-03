#!/usr/bin/env python3
"""plot_fig_e4.py -- docs/E1_DATA/{theta_bins,sigma_curve}.json -> 论文两张图(IEEE 单栏,plot_fig4 同版式)。

    python Eye_Tracker/tools/plot_fig_e4.py

fig_theta.pdf/png:成功率对 θ(没答与答错同算失败;主集 204 项,剔压力段与边走,全部 trial 含被前排球遮挡的)。
    分箱 [0.5,1) [1,1.5) [1.5,2.5) [2.5,4) [4,∞)。
    曲线:ours(v2, σ=1°)、single ray(σ=0.2°,同管线锥收成射线)、nearest-centroid 基线、
    sphere vote(视角无关球投票,有 e4_v2sphere_score.csv 时才画)。σ 竖线、乱猜线直接标注。
fig_sigma.pdf/png:成功率对锥宽 σ,只画 1°≤θ<2.5° 这一档(膝盖区,锥宽最敏感);σ=标定精度竖线。
    分箱直接由 e4_table.rows_for 现算,不依赖 json。
"""
from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_table import rows_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/E1_DATA"
SIGMA_CAL = 1.0        # 标定精度 ≈ 1°;σ 扫描最优也在此
OCC_LIMIT_DEG = 0.96   # 物理遮挡极限:球径 6.7cm @ 4m
CHANCE = 1 / 3
GREEN, RED, GRAY, BLUE, ORANGE, PURPLE = "#4a7000", "#b23232", "#555555", "#2a5db0", "#c86a1e", "#6b4aa0"
BINS = [0.5, 1.0, 1.5, 2.5, 4.0, 20.0]
TIERS = [("<1°", 0.0, 1.0), ("1–2.5°", 1.0, 2.5), ("2.5–4°", 2.5, 4.0)]
SIGMAS = [("v2s02", 0.2), ("v2s05", 0.5), ("v2s10", 1.0), ("v2", 1.5), ("v2s25", 2.5), ("v2s40", 4.0)]
Z = 1.959964


def wilson(h, n):
    p = h / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - half, c + half


def trials(cfg, occluded=None):
    """主集(剔压力段与边走)、有 θ 的 hit/miss 行;occluded=None 全取,True/False 只取被/未被前排球挡住的。"""
    out = []
    for r in rows_for(cfg):
        tags = set(r["tags"].split("|"))
        if r["outcome"] not in ("hit", "miss") or ({"stress", "walking"} & tags) or r["theta_deg"] == "":
            continue
        if occluded is None or ("beyond_occ" in tags) == occluded:
            out.append((float(r["theta_deg"]), r["outcome"] == "hit"))
    return np.array([t for t, _ in out]), np.array([h for _, h in out])


def binned(cfg, occluded=None):
    th, hit = trials(cfg, occluded)
    rows = []
    for lo, hi in zip(BINS, BINS[1:]):
        m = (th >= lo) & (th < hi)
        n = int(m.sum())
        if n:
            rows.append(dict(lo=lo, hi=hi, n=n, hit=int(hit[m].sum()), x=float(np.sqrt(lo * min(hi, 12.0)))))
    return rows


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


def series(cfg, occluded=None):
    rows = binned(cfg, occluded)
    x = [r["x"] for r in rows]
    a = [r["hit"] / r["n"] for r in rows]
    ci = [wilson(r["hit"], r["n"]) for r in rows]
    return x, a, [[v - c[0] for v, c in zip(a, ci)], [c[1] - v for v, c in zip(a, ci)]], rows


def fig_theta(plt, have_sphere):
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    ax.axvline(SIGMA_CAL, ls="-.", lw=0.7, color=GRAY, zorder=1)
    ax.text(SIGMA_CAL * 1.06, 0.06, r"calibrated $\sigma\approx1^\circ$", fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.axhline(CHANCE, ls=":", lw=0.8, color=RED, zorder=1)
    ax.text(11.5, CHANCE + 0.025, "3-ball chance", fontsize=6.5, color=RED, ha="right", va="bottom")
    lines = [("v2s10", r"ours (cone, $\sigma=1^\circ$)", GREEN, "o-", 3),
             ("v2s02", r"single ray ($\sigma=0.2^\circ$)", BLUE, "s--", 2),
             ("naive", "nearest centroid", GRAY, "^:", 2)]
    if have_sphere:
        lines.insert(2, ("v2sphere", "sphere vote (view-independent)", ORANGE, "d--", 2))
    for cfg, lab, col, fmt, z in lines:
        x, a, err, rows = series(cfg)
        ax.errorbar(x, a, yerr=err, fmt=fmt, color=col, lw=0.9, ms=3.0, elinewidth=0.6,
                    capsize=1.2, capthick=0.6, zorder=z, label=lab)
        if cfg == "v2s10":
            for xx, r in zip(x, rows):
                ax.annotate(f"{r['n']}", (xx, 0.0), textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=5.5, color="#888888")
    ax.axvspan(0.45, OCC_LIMIT_DEG, color=RED, alpha=0.08, lw=0, zorder=0)
    ax.text(np.sqrt(0.45 * OCC_LIMIT_DEG), 1.005, "past occlusion\nlimit (4 m)",
            fontsize=6.5, color=RED, ha="center", va="top")
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
        x, a, err, rows = series(cfg)
        print(f"fig_theta {cfg:8}: " + "  ".join(f"[{r['lo']},{r['hi']}) {r['hit']}/{r['n']}" for r in rows))


def sigma_curve():
    """每个 σ:未遮挡 trial 按 TIERS 的 hit/n,≥4° 单独记,产出率读 intents 日志。"""
    import glob, json
    out = []
    for cfg, sd in SIGMAS:
        th, hit = trials(cfg)
        tiers = {}
        for name, lo, hi in TIERS + [("≥4°", 4.0, 99.0)]:
            m = (th >= lo) & (th < hi)
            tiers[name] = (int(hit[m].sum()), int(m.sum()))
        t = k = 0
        for p in glob.glob(f"/home/liuchy/recordings/*/*/intents_e4_{cfg}.jsonl"):
            for ln in open(p):
                e = json.loads(ln)
                if not e.get("provisional"):
                    t += 1
                    k += e.get("object") is not None
        out.append(dict(sigma=sd, tiers=tiers, yield_finals=k / t if t else 0.0))
    return out


def fig_sigma(plt, curve):
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    sig = [c["sigma"] for c in curve]
    ax.axvline(SIGMA_CAL, ls="-.", lw=0.7, color=GRAY, zorder=1)
    ax.text(SIGMA_CAL * 1.05, 0.04, r"calibrated $\sigma\approx1^\circ$", fontsize=6.5, color=GRAY, ha="left", va="bottom")
    ax.axhline(CHANCE, ls=":", lw=0.8, color=RED, zorder=1)
    ax.text(0.17, CHANCE + 0.025, "3-ball chance", fontsize=6.5, color=RED, ha="left", va="bottom")
    for tier, lab, col, fmt in [("1–2.5°", r"$1^\circ\leq\theta<2.5^\circ$ (n=%d)" % curve[0]["tiers"]["1–2.5°"][1], BLUE, "s-")]:
        hn = [c["tiers"][tier] for c in curve]
        a = [h / n for h, n in hn]
        ci = [wilson(h, n) for h, n in hn]
        ax.errorbar(sig, a, yerr=[[v - c[0] for v, c in zip(a, ci)], [c[1] - v for v, c in zip(a, ci)]],
                    fmt=fmt, color=col, lw=0.9, ms=3.0, elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=3, label=lab)
    ax.set_xscale("log"); ax.set_xlim(0.16, 5); ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(FixedLocator(sig))
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Cone width $\sigma$ (deg); half-angle $2\sigma$")
    ax.set_ylabel("Success rate")
    ax.grid(alpha=0.18, lw=0.4, which="major"); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.3, ncol=1)
    fig.tight_layout(pad=0.3)
    fig.savefig(DATA / "fig_sigma.pdf"); fig.savefig(DATA / "fig_sigma.png", dpi=300)
    for c in curve:
        print(f"fig_sigma σ={c['sigma']:3}: " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in c["tiers"].items()) + f"  yield {c['yield_finals']:.0%}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    have_sphere = all((Path("/home/liuchy/recordings") / rec / "e4_v2sphere_score.csv").exists()
                      for rec in ["2026_08_16/000", "2026_08_25/u3"])
    fig_theta(plt, have_sphere)
    fig_sigma(plt, sigma_curve())
    print("wrote fig_theta.{pdf,png} fig_sigma.{pdf,png} ->", DATA)


if __name__ == "__main__":
    main()
