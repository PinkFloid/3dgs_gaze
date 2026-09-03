#!/usr/bin/env python3
"""e4_selective.py -- 难档的选择性精度:按置信量分桶看命中(hit)对多余/错答(extra)。

    python Eye_Tracker/tools/e4_selective.py [cfg ...]      # 缺省 v2;v1 = 冻结 intents.jsonl(置信量用 vote_share)

对每条录像重做 eval_e1 的段合并与卡序对齐,拿到每段代表 final 的 capture / vote_share /
歧义比(第二候选 capture / 第一候选 capture),按 θ_min 档(≥2.5° / 1.0–2.5° / <1.0°)与置信桶
列 hit、extra 与精度 hit/(hit+extra)。总体百分比被易档淹没,这张表才是难档该看的东西。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_e1 as E  # noqa: E402
from e1_cards import CARDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = Path("/home/liuchy/recordings")
A = {"v7": ROOT / "SceneRebuild/archive_envs/v7", "v8": ROOT / "SceneRebuild/archive_envs/v8",
     "v9": ROOT / "SceneRebuild/archive_envs/v9_rec"}
RECS = [("2026_08_16/000", "e1", "v7"), ("2026_08_16/001", "e2", "v7"), ("2026_08_16/s1", "s1", "v7"),
        ("2026_08_16/s2", "s2", "v7"), ("2026_08_16/s3", "s3", "v7"), ("2026_08_16/s4", "s4", "v8"),
        ("2026_08_16/s6", "s6", "v8"), ("2026_08_18/000", "s7", "v8"), ("2026_08_20/000", "c1", "v9"),
        ("2026_08_20/001", "c2", "v9"), ("2026_08_20/002", "c4", "v9"), ("2026_08_20/003", "c4", "v9"),
        ("2026_08_25/c1_1", "c4", "v9"), ("2026_08_25/c1_2", "c4", "v9"), ("2026_08_25/c1_3", "c4", "v9"),
        ("2026_08_25/u3", "u3", "v9")]
TIERS = [("≥2.5°", 2.5, 99), ("1.0–2.5°", 1.0, 2.5), ("<1.0°", 0, 1.0)]
BUCKETS = {"capture": [("≥0.5", 0.5, 9), ("0.3–0.5", 0.3, 0.5), ("0.2–0.3", 0.2, 0.3), ("<0.2", -1, 0.2)],
           "share": [("≥0.6", 0.6, 9), ("0.45–0.6", 0.45, 0.6), ("0.35–0.45", 0.35, 0.45), ("<0.35", -1, 0.35)],
           "ambiguity": [("≤0.25", -1, 0.25), ("0.25–0.5", 0.25, 0.5), ("0.5–0.75", 0.5, 0.75), (">0.75", 0.75, 9)]}


def tier_of(th):
    for name, lo, hi in TIERS:
        if th is not None and lo <= th < hi:
            return name


def confs(rep):
    c = rep.get("candidates") or []
    first = c[0].get("capture") if c else None
    second = (c[1].get("capture") or 0.0) if len(c) > 1 else 0.0
    return {"capture": rep.get("capture"), "share": rep.get("vote_share"),
            "ambiguity": (second / first) if first else None}


def analyse(cfg):
    rows = []
    for rec, card, era in RECS:
        path = R / rec / ("intents.jsonl" if cfg == "v1" else f"intents_e4_{cfg}.jsonl")
        if not path.exists():
            continue
        named = E.load_named(A[era])
        seq = E.era_alias(CARDS[card][1], named)
        ev = E.finals(path)
        _o = [e["origin_world"] for e in ev if e.get("origin_world")]
        o_unit = np.median(np.array(_o, float), axis=0).tolist() if _o else None   # 结果盲 θ,同 eval_e1
        eps = E.episodes(ev)
        runs = E.runs_of(seq)
        pairs = E.lcs_align([r[0] for r in runs], eps)
        used = set(pairs.values())
        for i, (want, k) in enumerate(runs):
            j = pairs.get(i)
            if j is None:
                rows.append((None, "miss", {}))
                continue
            rep = eps[j]["rep"]
            th = E.theta_min(o_unit, want, named)
            rows += [(tier_of(th), "hit", confs(rep))] * k
        for j, ep in enumerate(eps):
            if j not in used:
                rep = ep["rep"]
                th = E.theta_min(o_unit, ep["object"], named)
                rows.append((tier_of(th), "extra", confs(rep)))
    return rows


def main():
    cfgs = sys.argv[1:] or ["v2"]
    for cfg in cfgs:
        rows = analyse(cfg)
        keys = ["share"] if cfg == "v1" else ["capture", "share", "ambiguity"]
        for key in keys:
            print(f"\n===== {cfg}  按 {key}")
            print(f"{'θ 档':10} {'桶':10} {'hit':>4} {'extra':>5} {'精度':>6}")
            for tier, _, _ in TIERS:
                for bname, lo, hi in BUCKETS[key]:
                    h = sum(1 for t, k, c in rows if t == tier and k == "hit" and c.get(key) is not None and lo <= c[key] < hi)
                    x = sum(1 for t, k, c in rows if t == tier and k == "extra" and c.get(key) is not None and lo <= c[key] < hi)
                    if h + x:
                        print(f"{tier:10} {bname:10} {h:4d} {x:5d} {h / (h + x):6.0%}")
        print(f"{'':10} {'缺失项':10} {sum(1 for _, k, _ in rows if k == 'miss'):4d}")


if __name__ == "__main__":
    main()
