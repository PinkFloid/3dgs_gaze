#!/usr/bin/env python3
"""e4_table.py -- 收割 E4 消融批跑 -> Table II(docs/E1_DATA/table2.csv + markdown)。

    python Eye_Tracker/tools/e4_table.py

五个配置:full(现役冻结配置)/ naive(最近质心基线)/ votescope_all(去词表过滤)
/ priors_off(去物品台先验)/ cluster0(去角聚类)。同一批录像同一打分口径,
按 θ 档(≥2.5° / 1.0-2.5° / <1.0°)与总体给命中率;主集口径与 Fig.4 一致
(剔 stress=e2 压力段与 walking=u3,两者单独列行)。θ_min 由地图几何决定,
与感知配置无关,取 full 稿逐 trial 值(缺失行用站位中位)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_e1 import RECS, unit_rows  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs/E1_DATA"
CFGS = ["full", "naive", "votescope_all", "priors_off", "cluster0", "v2", "v2mass", "v2selfcal", "v2tag", "v2selfcal2", "v2selfcal3", "v2s25", "v2s40", "v2s10", "v2s05", "v2s02", "v2sphere", "v2table", "v2cluster0", "v2sphere10", "v2table10", "v2cluster010"]  # v2*: 09-02 面积归一后验(docs/CONE_POSTERIOR_V2.md)
TIERS = [("≥2.5°", 2.5, 99.0), ("1.0–2.5°", 1.0, 2.5), ("<1.0°", 0.0, 1.0)]


def rows_for(cfg):
    """cfg 的全部 unit trial;theta 用 full 稿的逐 trial 值(几何属性,配置无关)。"""
    out = []
    for path, card, room, mapv, station, tags in RECS:
        p = path if cfg == "full" else path.parent / f"e4_{cfg}_score.csv"
        if not p.exists():
            print(f"[!] 缺 {p}")
            continue
        full_rows = unit_rows(path, card, room, mapv, station, tags)
        cfg_rows = unit_rows(p, card, room, mapv, station, tags) if cfg != "full" else full_rows
        # theta 对齐:同一卡同序,按序号借 full 的 theta(缺失行 full 也有站位中位)
        th = [r["theta_deg"] for r in full_rows if r["outcome"] in ("hit", "miss")]
        k = 0
        for r in cfg_rows:
            if r["outcome"] in ("hit", "miss"):
                if k < len(th):
                    r["theta_deg"] = th[k]
                k += 1
                out.append(r)
    return out


def acc(rows):
    n = len(rows)
    h = sum(r["outcome"] == "hit" for r in rows)
    return f"{h}/{n} ({h/n:.0%})" if n else "—"


def main():
    lines = [f"| 配置 | 总体 | " + " | ".join(t[0] for t in TIERS) + " | e2 压力段 | u3 边走 |",
             "|---|---|---|---|---|---|---|"]
    csv_rows = []
    for cfg in CFGS:
        rows = rows_for(cfg)
        tagged = lambda t: [r for r in rows if t in r["tags"]]
        main_rows = [r for r in rows if not ({"stress", "walking"} & set(r["tags"].split("|")))
                     and r["theta_deg"] != ""]
        cells = [acc(main_rows)]
        for _, lo, hi in TIERS:
            cells.append(acc([r for r in main_rows if lo <= float(r["theta_deg"]) < hi]))
        cells += [acc(tagged("stress")), acc(tagged("walking"))]
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
        csv_rows.append((cfg, cells))
    md = "\n".join(lines)
    print(md)
    (OUT / "table2.md").write_text(md + "\n", encoding="utf-8")
    import csv as _csv
    with (OUT / "table2.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["config", "overall"] + [t[0] for t in TIERS] + ["e2_stress", "u3_walking"])
        for cfg, cells in csv_rows:
            w.writerow([cfg] + cells)
    print(f"\n-> {OUT}/table2.md table2.csv")


if __name__ == "__main__":
    main()
