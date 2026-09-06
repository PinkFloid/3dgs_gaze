#!/usr/bin/env python3
"""collect_e1.py -- 汇总所有 E1 打分 CSV -> 论文数据包(docs/E1_DATA/)。

    python Eye_Tracker/tools/collect_e1.py

产物:trials.csv(逐 trial 主表)/ curve.csv(θ 分箱精度)/ fig4_draft.png(草图)
/ README.md(来源与口径)。数据源与站位元信息硬编码在 RECS(=预注册台账,
docs/E1_RESULTS.md 的机器可读版);重跑任意录像的 score_card 后再跑本脚本即可刷新。
θ 口径:命中行用该 trial 实测 θ_min(从头位逐 trial 算);缺失行没有注视头位,
用该录像命中行的中位 θ 近似(theta_src=station 标记)。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/E1_DATA"
R = Path("/home/liuchy/recordings")

# (csv 路径, 卡, 房间, 地图, 站位描述, 标签集)
# 标签:stress=前排俘获压力段(不进主曲线);beyond_occ=超物理遮挡极限(进曲线,单独标记)
RECS = [
    (R/"2026_08_16/000/e1_score.csv", "e1", "老房", "v7", "1.4-1.7m 正对", set()),
    (R/"2026_08_16/001/e2_score.csv", "e2", "老房", "v7", "3.1m 正对", {"stress"}),
    (R/"2026_08_16/s1/s1_score.csv",  "s1", "老房", "v7", "~1.6m 正对", set()),
    (R/"2026_08_16/s2/s2_score.csv",  "s2", "老房", "v7", "~2.5m 正对", set()),
    (R/"2026_08_16/s3/s3_score.csv",  "s3", "老房", "v7", "~3.4m 正对", set()),
    (R/"2026_08_16/s4/s4_score.csv",  "s4", "新房", "v8", "3.9m 正对", set()),
    (R/"2026_08_16/s6/s6_score.csv",  "s6", "新房", "v8", "4.3m α57°", set()),
    (R/"2026_08_18/000/s7_score.csv", "s7", "新房", "v8", "3.1m α22.5°", set()),
    (R/"2026_08_20/000/c1_score.csv", "c1", "新房", "v9", "2m 正对", set()),
    (R/"2026_08_20/001/c2_score.csv", "c2", "新房", "v9", "2m 正对", set()),
    (R/"2026_08_20/002/c4_score.csv", "c4", "新房", "v9", "4.36m α20.6°", {"beyond_occ"}),
    (R/"2026_08_20/003/c4_score.csv", "c4", "新房", "v9", "4.52m α14.5°", {"beyond_occ"}),
    # 08-25 斜位三连 + 边走(车位偏移 5.8cm 实测:斜位站侧向分量 <1cm 不受害;
    # u1 正对站侧向 3.7cm+后排邻近受害,整条剔除并披露,不入 RECS)
    (R/"2026_08_25/c1_1/c4_score.csv", "c4", "新房", "v9", "4.25m α22.2°", {"beyond_occ"}),
    (R/"2026_08_25/c1_2/c4_score.csv", "c4", "新房", "v9", "4.31m α26.9°", set()),
    (R/"2026_08_25/c1_3/c4_score.csv", "c4", "新房", "v9", "3.77m α26.7°", set()),
    (R/"2026_08_25/u3/u3_score.csv",   "u3", "新房", "v9", "2.5m 边走", {"walking"}),
    # 09-06/07 v10 补录(lab_colmap_v10:球距 24cm,第二排在球正后方 15-23cm;站位=station_theta 头位中位;
    # p1/p2 = 两位同学佩戴,tags 标人;边走条同 u3 不进主曲线;p1_v3(3.60m)用户已删不入)
    (R/"2026_09_06/v1/v1_score.csv",           "v1", "新房", "v10", "2.15m 正对", set()),
    (R/"2026_09_06/v2/v2_score.csv",           "v2", "新房", "v10", "3.46m 正对", set()),
    (R/"2026_09_07/p1_v1/v1_score.csv",        "v1", "新房", "v10", "1.70m 正对", {"p1"}),
    (R/"2026_09_07/v4/v4_score.csv",           "v4", "新房", "v10", "2.5m 边走", {"walking"}),
    (R/"2026_09_07/p1_v4/v4_score.csv",        "v4", "新房", "v10", "2.2m 边走", {"walking", "p1"}),
    (R/"2026_09_07/v1_near/v1_score.csv",      "v1", "新房", "v10", "1.34m α-17°", set()),
    (R/"2026_09_07/v2_near/v2_score.csv",      "v2", "新房", "v10", "1.26m α+23°", set()),
    (R/"2026_09_07/v2_near_p2/v2_score.csv",   "v2", "新房", "v10", "1.05m α-25°", {"p2"}),
    (R/"2026_09_07/v6_far/v6_score.csv",       "v6", "新房", "v10", "2.99m 正对", set()),
    (R/"2026_09_07/v6_mid/v6_score.csv",       "v6", "新房", "v10", "1.80m 正对", set()),
    (R/"2026_09_07/v6_near/v6_score.csv",      "v6", "新房", "v10", "1.23m α-24°", set()),
    (R/"2026_09_07/v6_move/v6_score.csv",      "v6", "新房", "v10", "1.1m 边走", {"walking"}),
]
BINS = [0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0, 20.0]


def unit_rows(csv_path, card, room, mapv, station, tags):
    """一行打分 CSV -> 若干 unit trial(连击 want '球L×2' 展开;多余段单列)。"""
    rows = []
    thetas = []
    for r in csv.DictReader(csv_path.open(encoding="utf-8")):
        th = r["theta_min_deg"]
        if th:
            thetas.append(float(th))
    med = float(np.median(thetas)) if thetas else ""
    for r in csv.DictReader(csv_path.open(encoding="utf-8")):
        v = r["verdict"]
        m = re.match(r"^(.*?)(?:×(\d+))?$", r["want"])
        name, k = m.group(1), int(m.group(2) or 1)
        base = dict(rec=str(csv_path.parent.name), card=card, room=room, map=mapv,
                    station=station, tags="|".join(sorted(tags)),
                    target=name, vote=r["vote"], dur_s=r["dur_s"], dist_m=r["dist_m"])
        tu = r.get("theta_unit_deg", "")
        if v == "＋多余":
            rows.append(dict(base, target=r["got"], outcome="extra",
                             theta_deg=tu if tu else r["theta_min_deg"], theta_src="unit" if tu else "trial"))
            continue
        credit = 0 if v.startswith("✗") else (2 if v.startswith("✓✓") else 1)
        if "(-1)" in v:
            credit = max(credit - v.count("(-1)"), 1) if v.startswith("✓") else 0
        credit = min(credit, k)
        th = r["theta_min_deg"]
        for i in range(k):
            hit = i < credit
            if tu:  # 结果盲:同一(录像,目标)单元的 trial 共用 θ_unit,与命中无关
                rows.append(dict(base, outcome="hit" if hit else "miss", theta_deg=tu, theta_src="unit"))
            else:   # 旧 CSV 兜底:命中行用注视 θ,漏掉的用录像中位(横轴与结果挂钩,勿用于终稿)
                rows.append(dict(base, outcome="hit" if hit else "miss",
                                 theta_deg=th if th else med,
                                 theta_src="trial" if th else "station"))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for path, card, room, mapv, station, tags in RECS:
        if not path.exists():
            print(f"[!] 缺 {path}(跳过)")
            continue
        all_rows += unit_rows(path, card, room, mapv, station, tags)
    fields = ["rec", "card", "room", "map", "station", "tags", "target",
              "outcome", "theta_deg", "theta_src", "vote", "dur_s", "dist_m"]
    with (OUT / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    trials = [r for r in all_rows if r["outcome"] in ("hit", "miss")
              and not ({"stress", "walking"} & set(r["tags"].split("|")))
              and r["theta_deg"] != ""]
    th = np.array([float(r["theta_deg"]) for r in trials])
    ok = np.array([r["outcome"] == "hit" for r in trials])
    with (OUT / "curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["theta_lo", "theta_hi", "n", "hits", "acc"])
        pts = []
        for lo, hi in zip(BINS, BINS[1:]):
            m = (th >= lo) & (th < hi)
            if m.sum() == 0:
                continue
            acc = ok[m].mean()
            w.writerow([lo, hi, int(m.sum()), int(ok[m].sum()), round(float(acc), 3)])
            pts.append((np.sqrt(lo * hi), acc, int(m.sum())))
    n_all, n_hit = len(trials), int(ok.sum())
    print(f"主曲线 trial {n_all}(命中 {n_hit},{n_hit/n_all:.1%});"
          f"另 stress {sum(1 for r in all_rows if 'stress' in r['tags'] and r['outcome'] in ('hit','miss'))} trial、"
          f"多余注视 {sum(1 for r in all_rows if r['outcome']=='extra')} 段")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    xs, ys, ns = zip(*pts)
    ax.plot(xs, ys, "o-", color="#4a7000", lw=1.8, ms=5, zorder=3)
    for x, y, n in pts:
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="#666")
    ax.axhline(1/3, ls=":", color="#b23", lw=1)
    ax.text(BINS[-2], 1/3 + 0.02, "3-ball chance", fontsize=7.5, color="#b23", ha="right")
    ax.axvspan(BINS[0], 0.96, color="#c0392b", alpha=0.08)
    ax.text(0.7, 0.97, "past occlusion\nlimit (4m)", fontsize=7, color="#b23",
            ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("min angular separation to nearest named object θ (deg)")
    ax.set_ylabel("binding accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Gaze-object binding vs angular separation (draft)", fontsize=9)
    ax.grid(alpha=.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_draft.png")
    print(f"-> {OUT}/trials.csv curve.csv fig4_draft.png")


if __name__ == "__main__":
    main()
