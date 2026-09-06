#!/usr/bin/env python3
"""binding_stats.py -- 完整绑定输出统计:目标时段 × 各配置 final 流 × brain 接受闸门。

    conda run -n nerfstudio python Eye_Tracker/tools/binding_stats.py [--out docs/E1_DATA/audit_0906]

输入:card_windows 的逐项目标时段(<out>/windows/*.json)、各配置 intents 日志(recordings/*/)、eval_e1 打分 CSV。
接受闸门 = Intension/core/attention.accepted(brain 现行默认 min-vote 0.45 / margin 1.4 / min-capture 0.2;
另算 E2 冻结口径 0.5 / 0 / 0.2)。一条 final 归入与其"盯看窗"[盯看起点-0.3s, 叮+0.5s] 重叠 ≥0.5s(或整段落在报名起点之后的分析窗内)的目标时段;
可归入多个相邻时段(连击对共用一段注视)。主时段 = 重叠最长的那个;未归入任何目标时段的接受判定 = 额外绑定。

产物(<out>/):finals_<cfg>.csv(逐段,含全部候选 m/q/c/S/s)、trials_<cfg>.csv(逐 trial)、recordings_<cfg>.csv、
ambiguity_<cfg>.csv / coverage_<cfg>.csv(歧义比分桶与拒绝曲线)、sweep_<cfg>.csv(阈值扫描)、
compare_<a>_vs_<b>.csv 与 cases_<a>_vs_<b>.csv(逐 trial 对照与不一致案例)、summary.md。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "Intension"))
from card_windows import ENV, RECS, rec_tag  # noqa: E402
from collect_e1 import unit_rows  # noqa: E402
from core.attention import accepted  # noqa: E402

R = Path("/home/liuchy/recordings")
CFGS = {"v1": "intents.jsonl", "ours": "intents_e4_v2s10.jsonl", "noocc": "intents_e4_v2noocc10.jsonl", "vis": "intents_e4_v2vis10.jsonl",
        "noocc_ng": "intents_e4_v2noocc10ng.jsonl", "mass10": "intents_e4_v2mass10.jsonl",
        "v2s15": "intents_e4_v2.jsonl", "mass15": "intents_e4_v2mass.jsonl", "noocc15": "intents_e4_v2noocc.jsonl",
        "sphere10": "intents_e4_v2sphere10.jsonl", "naive": "intents_e4_naive.jsonl"}
CSV_OF = {"v1": None, "ours": "e4_v2s10_score.csv", "noocc": "e4_v2noocc10_score.csv", "vis": "e4_v2vis10_score.csv",
          "noocc_ng": "e4_v2noocc10ng_score.csv", "mass10": "e4_v2mass10_score.csv", "v2s15": "e4_v2_score.csv",
          "mass15": "e4_v2mass_score.csv", "noocc15": "e4_v2noocc_score.csv", "sphere10": "e4_v2sphere10_score.csv",
          "naive": "e4_naive_score.csv"}
GATES = {"live": dict(min_vote=0.45, margin=1.4, min_capture=0.2), "e2": dict(min_vote=0.5, margin=0.0, min_capture=0.2)}
TIERS = [("≥2.5°", 2.5, 99.0), ("1.0–2.5°", 1.0, 2.5), ("<1.0°", 0.0, 1.0)]
PAIRS = [("ours", "noocc"), ("ours", "vis"), ("vis", "noocc"), ("ours", "mass10"), ("v2s15", "mass15"), ("ours", "noocc_ng"), ("ours", "v1")]


def canon(n):
    if n is None:
        return None
    return n.replace("网球", "球") if isinstance(n, str) else n


def size_class(name):
    if name is None:
        return ""
    if name.startswith("球"):
        return "ball(3cm)"
    if name.startswith("苹果") or name in ("橘子", "石榴"):
        return "fruit(4-5cm)"
    if "杯" in name:
        return "cup"
    if name == "香蕉":
        return "banana"
    if name == "水瓶":
        return "bottle"
    if name == "纸箱子":
        return "box"
    return "other"


def tier_of(th):
    for name, lo, hi in TIERS:
        if th is not None and lo <= th < hi:
            return name
    return ""


def occl_of(flags):
    if "beyond_occ" in flags:
        return "occluded(beyond limit)"
    if "stress" in flags:
        return "occluded(front-capture)"
    return "clear"


def load_finals(path):
    out = []
    for ln in path.open(encoding="utf-8"):
        try:
            e = json.loads(ln)
        except Exception:
            continue
        if e.get("provisional"):
            continue
        e["object"] = canon(e.get("object"))
        for c in e.get("candidates") or []:
            c["name"] = canon(c.get("name"))
        out.append(e)
    return out


def amb(e):
    c = e.get("candidates") or []
    if not c:
        return None, None
    a_cap = a_sh = None
    if c[0].get("capture") is not None:
        a_cap = (max(c[1].get("capture") or 0.0, 0.0) / c[0]["capture"]) if (len(c) > 1 and c[0]["capture"]) else 0.0
    if c[0].get("share"):
        a_sh = (c[1].get("share") or 0.0) / c[0]["share"] if len(c) > 1 else 0.0
    return (round(a_cap, 3) if a_cap is not None else None), (round(a_sh, 3) if a_sh is not None else None)


def gate(e, g):
    return accepted(dict(e), g["min_vote"], g["margin"], g["min_capture"])


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def theta_map(rec, card):
    p = R / rec / f"{card}_score.csv"
    m = {}
    if p.exists():
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r["verdict"] == "＋多余":
                continue
            nm = canon(r["want"].split("×")[0])
            if r.get("theta_unit_deg"):
                m[nm] = float(r["theta_unit_deg"])
    return m


def lcs_items(rec, card, era, cfg):
    """cfg 的 eval_e1 逐项判定(卡序,连击展开):[hit|miss ...],缺 CSV 返回 None。"""
    p = R / rec / (f"{card}_score.csv" if CSV_OF[cfg] is None else CSV_OF[cfg])
    if not p.exists():
        return None
    rows = unit_rows(p, card, "", era, "", set())
    return [r["outcome"] for r in rows if r["outcome"] in ("hit", "miss")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs/E1_DATA/audit_0906"))
    ap.add_argument("--cfgs", default=",".join(CFGS))
    a = ap.parse_args()
    out = Path(a.out)
    wins = {}
    for rec, card, era, flags in RECS:
        p = out / "windows" / f"{rec_tag(rec)}.json"
        if p.exists():
            wins[rec] = json.loads(p.read_text(encoding="utf-8"))
    cfgs = [c for c in a.cfgs.split(",") if c]
    all_trials = {}    # cfg -> list of trial dicts
    all_recs = {}
    finals_rows = {}
    for cfg in cfgs:
        trials, recrows, frows = [], [], []
        for rec, card, era, flags in RECS:
            if rec not in wins or "excluded" in flags:
                continue
            p = R / rec / CFGS[cfg]
            if not p.exists():
                continue
            W = wins[rec]
            fin = load_finals(p)
            th = theta_map(rec, card)
            lcs = lcs_items(rec, card, era, cfg)
            items = [it for it in W["items"]]
            for e in fin:
                e["_acc_live"] = gate(e, GATES["live"])
                e["_acc_e2"] = gate(e, GATES["e2"])
                e["_A"], e["_As"] = amb(e)
                e["_wins"] = []
            # 归入目标时段
            for it in items:
                if it.get("win_start_abs") is None:
                    it["_fin"] = []
                    continue
                w0, w1 = it["win_start_abs"], it["win_end_abs"]
                # 归入规则:与"盯看窗"[盯看起点-0.3, 叮+0.5] 重叠 ≥0.5s,或整段落在分析窗内且起于报名之后。
                # 只按分析窗重叠会把上一项盯看的尾巴(报名期间眼睛还在前一个目标上)算成本项的"首个错误绑定"。
                s0 = (it["stare_start_abs"] - 0.3) if it.get("stare_start_abs") is not None else w0
                it["_fin"] = []
                for e in fin:
                    ov = overlap(e["t_start"], e["t_end"], s0, w1)
                    if ov >= 0.5 or (e["t_start"] >= w0 and e["t_end"] <= w1):
                        it["_fin"].append(e)
                        e["_wins"].append((it["k"], max(ov, overlap(e["t_start"], e["t_end"], w0, w1))))
            for e in fin:
                e["_primary"] = max(e["_wins"], key=lambda x: x[1])[0] if e["_wins"] else None
            t0 = W["t_rec0"]
            # 逐段表
            for e in fin:
                prim = e["_primary"]
                tgt = canon(items[prim - 1]["target"]) if prim else ""
                frows.append({"rec": rec, "card": card, "era": era, "flags": flags, "cfg": cfg,
                              "t_start_rel": round(e["t_start"] - t0, 2), "t_end_rel": round(e["t_end"] - t0, 2),
                              "dur_s": round(e["duration_s"], 2), "dist_m": e.get("distance_m"),
                              "object": e.get("object") or "", "object_label": e.get("object_label"),
                              "share": e.get("vote_share"), "q": e.get("q"), "capture": e.get("capture"),
                              "A_cap": e["_A"], "A_share": e["_As"], "p_none": e.get("p_none"),
                              "surface": e.get("surface") or "", "W": e.get("W"),
                              "acc_live": int(bool(e["_acc_live"])), "acc_e2": int(bool(e["_acc_e2"])),
                              "window_k": prim or "", "target": tgt,
                              "correct": (int(e.get("object") == tgt) if (prim and e.get("object")) else ""),
                              "n_cands": len(e.get("candidates") or []),
                              "candidates": json.dumps(e.get("candidates") or [], ensure_ascii=False)})
            # 逐 trial
            for it in items:
                k = it["k"]
                tgt = canon(it["target"])
                base = {"rec": rec, "card": card, "era": era, "flags": flags, "cfg": cfg, "k": k, "target": tgt,
                        "size": size_class(tgt), "theta_unit": th.get(tgt), "tier": tier_of(th.get(tgt)),
                        "occl": occl_of(flags), "walking": int("walking" in flags), "stress": int("stress" in flags),
                        "window_src": it.get("window_src", ""), "reliable": int(bool(W.get("reliable"))),
                        "lcs": (lcs[k - 1] if (lcs and k - 1 < len(lcs)) else "")}
                if it.get("win_start_abs") is None:
                    trials.append(dict(base, has_window=0, outcome_first="unknown", outcome_any="unknown"))
                    continue
                fl = sorted(it["_fin"], key=lambda e: e["t_end"])
                cue = it.get("cue_start_abs") if it.get("cue_start_abs") is not None else it["win_start_abs"]
                objf = [e for e in fl if e.get("object")]
                acc = [e for e in fl if e["_acc_live"]]
                acc2 = [e for e in fl if e["_acc_e2"]]
                seqc = [e["object"] == tgt for e in acc]
                first = acc[0] if acc else None
                if first is None:
                    o_first = "none"
                else:
                    o_first = "correct" if first["object"] == tgt else "wrong"
                anyc, anyw = any(seqc), any(not x for x in seqc)
                o_any = "none" if not acc else ("correct_only" if anyc and not anyw else ("wrong_only" if anyw and not anyc else "mixed"))
                flip_wc = int(any((not seqc[i]) and any(seqc[i + 1:]) for i in range(len(seqc))))
                flip_cw = int(any(seqc[i] and any(not x for x in seqc[i + 1:]) for i in range(len(seqc))))
                dur_by = defaultdict(float)
                for e in objf:
                    dur_by[e["object"]] += e["duration_s"]
                maj = max(dur_by, key=dur_by.get) if dur_by else None
                gdur = defaultdict(float)
                for e in acc:
                    gdur[e["object"]] += e["duration_s"]
                gmaj = max(gdur, key=gdur.get) if gdur else None
                best_t = max([e.get("capture") or 0.0 for e in objf if e["object"] == tgt], default=None)
                best_w = max([e.get("capture") or 0.0 for e in objf if e["object"] != tgt], default=None)
                best_t_s = max([e.get("vote_share") or 0.0 for e in objf if e["object"] == tgt], default=None)
                best_w_s = max([e.get("vote_share") or 0.0 for e in objf if e["object"] != tgt], default=None)
                dists = [e.get("distance_m") for e in objf if e.get("distance_m")]
                trials.append(dict(
                    base, has_window=1, win_start_rel=round(it["win_start_abs"] - t0, 2),
                    win_end_rel=round(it["win_end_abs"] - t0, 2), n_finals=len(fl), n_obj_finals=len(objf),
                    n_acc=len(acc), n_acc_e2=len(acc2), outcome_first=o_first,
                    first_obj=(first["object"] if first else ""),
                    first_lat_end=(round(first["t_end"] - cue, 2) if first else ""),
                    first_lat_start=(round(first["t_start"] - cue, 2) if first else ""),
                    first_lat_stare=(round(first["t_end"] - it["stare_start_abs"], 2) if (first and it.get("stare_start_abs") is not None) else ""),
                    first_share=(first.get("vote_share") if first else ""), first_capture=(first.get("capture") if first else ""),
                    first_A=(first["_A"] if first else ""), first_p_none=(first.get("p_none") if first else ""),
                    first_dur=(round(first["duration_s"], 2) if first else ""),
                    outcome_any=o_any, any_correct=int(anyc), any_wrong=int(anyw), flip_wrong_then_correct=flip_wc,
                    flip_correct_then_wrong=flip_cw, wrong_objs="|".join(sorted({e["object"] for e in acc if e["object"] != tgt})),
                    first_e2=("correct" if (acc2 and acc2[0]["object"] == tgt) else ("wrong" if acc2 else "none")),
                    ungated_first=("correct" if (objf and objf[0]["object"] == tgt) else ("wrong" if objf else "none")),
                    ungated_majority=("correct" if maj == tgt else ("wrong" if maj else "none")),
                    ungated_majority_obj=(maj or ""), ungated_majority_dur=(round(dur_by[maj], 2) if maj else ""),
                    gated_majority=("correct" if gmaj == tgt else ("wrong" if gmaj else "none")),
                    gated_majority_obj=(gmaj or ""), gated_majority_dur=(round(gdur[gmaj], 2) if gmaj else ""),
                    best_target_capture=best_t, best_wrong_capture=best_w, best_target_share=best_t_s, best_wrong_share=best_w_s,
                    dist_m=(round(float(np.median(dists)), 2) if dists else "")))
            # 逐录像
            acc_all = [e for e in fin if e["_acc_live"]]
            extra = [e for e in acc_all if e["_primary"] is None]
            wrong_in = [e for e in acc_all if e["_primary"] and e["object"] != canon(items[e["_primary"] - 1]["target"])]
            dur = W["duration_s"]
            recrows.append({"rec": rec, "card": card, "era": era, "flags": flags, "cfg": cfg, "duration_s": round(dur, 1),
                            "window_src": W.get("method"), "reliable": int(bool(W.get("reliable"))),
                            "n_items": len(items), "n_items_with_window": sum(1 for it in items if it.get("win_start_abs") is not None),
                            "window_cover_s": round(sum(it["win_end_abs"] - it["win_start_abs"] for it in items if it.get("win_start_abs") is not None), 1),
                            "n_finals": len(fin), "n_obj_finals": sum(1 for e in fin if e.get("object")),
                            "n_acc": len(acc_all), "n_acc_in_window": len(acc_all) - len(extra),
                            "n_acc_wrong_in_window": len(wrong_in), "n_extra": len(extra),
                            "extra_per_min": round(len(extra) / (dur / 60.0), 2),
                            "extra_objs": "|".join(f"{k}:{v}" for k, v in Counter(e["object"] for e in extra).most_common()),
                            "extra_dur_s": round(sum(e["duration_s"] for e in extra), 1)})
        all_trials[cfg] = trials
        all_recs[cfg] = recrows
        finals_rows[cfg] = frows
        for name, rows in (("finals", frows), ("trials", trials), ("recordings", recrows)):
            if rows:
                keys = []
                for r in rows:
                    for kk in r:
                        if kk not in keys:
                            keys.append(kk)
                with (out / f"{name}_{cfg}.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows)
    # ---------------- 汇总
    md = []

    def subset(trs, which):
        if which == "main":
            return [t for t in trs if not t["walking"] and not t["stress"]]
        if which == "walking":
            return [t for t in trs if t["walking"]]
        if which == "stress":
            return [t for t in trs if t["stress"]]
        return trs

    def summ(trs):
        n = len(trs)
        hw = [t for t in trs if t.get("has_window")]
        c = sum(t["outcome_first"] == "correct" for t in hw)
        w = sum(t["outcome_first"] == "wrong" for t in hw)
        z = sum(t["outcome_first"] == "none" for t in hw)
        ac = sum(t.get("any_correct", 0) for t in hw)
        flips = sum(t.get("flip_wrong_then_correct", 0) for t in hw)
        lcs_h = sum(t["lcs"] == "hit" for t in trs)
        lcs_n = sum(t["lcs"] in ("hit", "miss") for t in trs)
        lat = [t["first_lat_end"] for t in hw if t["outcome_first"] == "correct" and t["first_lat_end"] != ""]
        lat2 = [t["first_lat_stare"] for t in hw if t["outcome_first"] == "correct" and t.get("first_lat_stare", "") != ""]
        ug = sum(t.get("ungated_first") == "correct" for t in hw)
        return {"n": n, "with_window": len(hw), "first_correct": c, "first_wrong": w, "none": z, "any_correct": ac,
                "flip_w2c": flips, "ungated_first_correct": ug,
                "lat_med": (round(float(np.median(lat)), 2) if lat else ""),
                "lat_stare_med": (round(float(np.median(lat2)), 2) if lat2 else ""), "lcs": f"{lcs_h}/{lcs_n}"}

    def fmt(s):
        hw = s["with_window"] or 1
        return (f"{s['n']} | {s['with_window']} | {s['first_correct']} ({s['first_correct']/hw:.0%}) | "
                f"{s['first_wrong']} ({s['first_wrong']/hw:.0%}) | {s['none']} ({s['none']/hw:.0%}) | "
                f"{s['any_correct']} ({s['any_correct']/hw:.0%}) | {s['flip_w2c']} | {s['ungated_first_correct']} ({s['ungated_first_correct']/hw:.0%}) | "
                f"{s['lat_med']} / {s['lat_stare_med']} | {s['lcs']}")

    hdr = ("| 配置 | 子集 | trial | 有时段 | 首个接受=正确 | 首个接受=错误 | 无接受 | 任一接受正确 | 先错后对 | 未过闸首判正确 | 正确首判时延中位 s(自报名起 / 自盯看起) | LCS 命中 |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|")
    md.append("## 1. 逐 trial 绑定结果(接受闸=brain 现行默认 0.45/1.4/0.2;时段=口播时刻表或回退卡序)\n")
    md.append(hdr)
    for cfg in cfgs:
        trs = all_trials.get(cfg, [])
        if not trs:
            continue
        for which in ("main", "stress", "walking"):
            sub = subset(trs, which)
            if sub:
                md.append(f"| {cfg} | {which} | " + fmt(summ(sub)) + " |")
    md.append("\n### 1b. 仅可靠口播时段的录像(reliable=1,10 条)\n")
    md.append(hdr)
    for cfg in cfgs:
        trs = [t for t in all_trials.get(cfg, []) if t["reliable"]]
        if trs:
            md.append(f"| {cfg} | main | " + fmt(summ(subset(trs, "main"))) + " |")
    md.append("\n### 1d. 顺序 + 最长注视法:每项按卡序取其时段内累计注视时长最长的对象作为系统答案(不设闸=全部判定;设闸=只算通过接受闸的判定)\n")
    md.append("| 配置 | 子集 | trial(有时段) | 不设闸:正确 | 错误 | 无 | 设闸:正确 | 错误 | 无 |\n|---|---|---|---|---|---|---|---|---|")
    for cfg in cfgs:
        trs = all_trials.get(cfg, [])
        for which in ("main", "walking", "stress"):
            hw = [t for t in subset(trs, which) if t.get("has_window")]
            if not hw:
                continue
            u = Counter(t.get("ungated_majority") for t in hw)
            g = Counter(t.get("gated_majority") for t in hw)
            md.append(f"| {cfg} | {which} | {len(hw)} | {u.get('correct',0)} ({u.get('correct',0)/len(hw):.0%}) | {u.get('wrong',0)} | {u.get('none',0)} | "
                      f"{g.get('correct',0)} ({g.get('correct',0)/len(hw):.0%}) | {g.get('wrong',0)} | {g.get('none',0)} |")
    md.append("\n最长注视法的两个诚实口径:(a) 仅口播时刻表可靠的 10 条录像(129 trial,时段与识别无关);"
              "(b) 主集全部 204 trial,回退录像里没有时段的项按'无'计(回退窗来自卡序对齐的命中段,其窗内多数判定必然正确,单看有窗项会高估)。\n")
    md.append("| 配置 | (a) 可靠 129:不设闸 正确/错误/无 | (a) 设闸 正确/错误/无 | (b) 204:不设闸 正确/错误/无 | (b) 设闸 正确/错误/无 | 论文 LCS 204 |\n|---|---|---|---|---|---|")
    for cfg in cfgs:
        trs = subset(all_trials.get(cfg, []), "main")
        if not trs:
            continue
        rel = [t for t in trs if t["reliable"] and t.get("has_window")]
        def cnt(rows, key, n_total):
            c = Counter(t.get(key) for t in rows)
            return f"{c.get('correct',0)}/{c.get('wrong',0)}/{n_total - c.get('correct',0) - c.get('wrong',0)}"
        md.append(f"| {cfg} | {cnt(rel,'ungated_majority',len(rel))} ({Counter(t.get('ungated_majority') for t in rel).get('correct',0)/max(len(rel),1):.0%}) | "
                  f"{cnt(rel,'gated_majority',len(rel))} ({Counter(t.get('gated_majority') for t in rel).get('correct',0)/max(len(rel),1):.0%}) | "
                  f"{cnt(trs,'ungated_majority',len(trs))} ({Counter(t.get('ungated_majority') for t in trs).get('correct',0)/len(trs):.0%}) | "
                  f"{cnt(trs,'gated_majority',len(trs))} ({Counter(t.get('gated_majority') for t in trs).get('correct',0)/len(trs):.0%}) | "
                  f"{sum(t['lcs']=='hit' for t in trs)}/{len(trs)} |")
    md.append("\n按 θ 档 / 遮挡(主集,最长注视法):\n")
    md.append("| 配置 | 分组 | trial | 不设闸:正确/错误/无 | 设闸:正确/错误/无 |\n|---|---|---|---|---|")
    for cfg in ("v1", "ours", "vis", "noocc", "mass10"):
        trs = [t for t in subset(all_trials.get(cfg, []), "main") if t.get("has_window")]
        groups = [(t[0], [x for x in trs if x["tier"] == t[0]]) for t in TIERS]
        groups += [(o, [x for x in trs if x["occl"] == o]) for o in ("clear", "occluded(beyond limit)")]
        for gname, g in groups:
            if not g:
                continue
            u = Counter(t.get("ungated_majority") for t in g)
            gg = Counter(t.get("gated_majority") for t in g)
            md.append(f"| {cfg} | {gname} | {len(g)} | {u.get('correct',0)}/{u.get('wrong',0)}/{u.get('none',0)} | {gg.get('correct',0)}/{gg.get('wrong',0)}/{gg.get('none',0)} |")
    md.append("\n### 1c. 主集,E2 冻结闸门(0.5 / 无 margin / 0.2)下的首个接受判定\n")
    md.append("| 配置 | 首个正确 | 首个错误 | 无接受 |\n|---|---|---|---|")
    for cfg in cfgs:
        hw = [t for t in subset(all_trials.get(cfg, []), "main") if t.get("has_window")]
        if hw:
            c = Counter(t.get("first_e2") for t in hw)
            md.append(f"| {cfg} | {c.get('correct',0)} ({c.get('correct',0)/len(hw):.0%}) | {c.get('wrong',0)} ({c.get('wrong',0)/len(hw):.0%}) | {c.get('none',0)} ({c.get('none',0)/len(hw):.0%}) |")
    # 分档
    md.append("\n## 2. 主集按 θ 档 / 遮挡(首个接受判定)\n")
    md.append("| 配置 | 分组 | trial | 首个正确 | 首个错误 | 无接受 | 任一正确 | LCS |\n|---|---|---|---|---|---|---|---|")
    for cfg in ("v1", "ours", "vis", "noocc", "mass10", "noocc_ng"):
        trs = subset(all_trials.get(cfg, []), "main")
        if not trs:
            continue
        groups = [(t[0], [x for x in trs if x["tier"] == t[0]]) for t in TIERS]
        groups += [(o, [x for x in trs if x["occl"] == o]) for o in ("clear", "occluded(beyond limit)")]
        groups += [("clear & <1.5°", [x for x in trs if x["occl"] == "clear" and x["theta_unit"] is not None and x["theta_unit"] < 1.5])]
        groups += [(f"size={s}", [x for x in trs if x["size"] == s]) for s in ("ball(3cm)", "fruit(4-5cm)", "cup", "banana", "bottle", "box")]
        groups += [(f"dist {lo}-{hi}m", [x for x in trs if x.get("dist_m") not in ("", None) and lo <= float(x["dist_m"]) < hi]) for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 9))]
        for gname, g in groups:
            if not g:
                continue
            s = summ(g)
            hw = s["with_window"] or 1
            md.append(f"| {cfg} | {gname} | {s['n']} | {s['first_correct']} ({s['first_correct']/hw:.0%}) | {s['first_wrong']} ({s['first_wrong']/hw:.0%}) | "
                      f"{s['none']} ({s['none']/hw:.0%}) | {s['any_correct']} ({s['any_correct']/hw:.0%}) | {s['lcs']} |")
    # 额外绑定
    md.append("\n## 3. 逐录像:额外绑定(接受但不在任何目标时段内)与录制时长\n")
    md.append("| 配置 | 录像 | 卡 | 时长 s | 时段来源 | finals | 接受 | 时段内接受 | 时段内错误 | 额外 | 额外/分钟 | 额外对象 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in ("v1", "ours", "noocc", "mass10"):
        for r in all_recs.get(cfg, []):
            md.append(f"| {cfg} | {r['rec']} | {r['card']} | {r['duration_s']} | {r['window_src']} | {r['n_finals']} | {r['n_acc']} | "
                      f"{r['n_acc_in_window']} | {r['n_acc_wrong_in_window']} | {r['n_extra']} | {r['extra_per_min']} | {r['extra_objs']} |")
        tot = all_recs.get(cfg, [])
        if tot:
            md.append(f"| {cfg} | **合计** | | {round(sum(r['duration_s'] for r in tot),0)} | | {sum(r['n_finals'] for r in tot)} | {sum(r['n_acc'] for r in tot)} | "
                      f"{sum(r['n_acc_in_window'] for r in tot)} | {sum(r['n_acc_wrong_in_window'] for r in tot)} | {sum(r['n_extra'] for r in tot)} | "
                      f"{round(sum(r['n_extra'] for r in tot)/(sum(r['duration_s'] for r in tot)/60),2)} | |")
    # 歧义比
    md.append("\n## 4. 歧义比 A(第二/第一名 capture)——时段内接受判定,正确 vs 错误\n")
    md.append("| 配置 | A 区间 | 正确 | 错误 | 错误率 |\n|---|---|---|---|---|")
    for cfg in ("ours", "noocc", "mass10", "v2s15"):
        fr = [f for f in finals_rows.get(cfg, []) if f["acc_live"] and f["window_k"] != "" and f["A_cap"] is not None
              and "walking" not in f["flags"] and "stress" not in f["flags"]]
        if not fr:
            continue
        rows_a = []
        for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 99)):
            g = [f for f in fr if lo <= f["A_cap"] < hi]
            if g:
                c = sum(f["correct"] == 1 for f in g)
                rows_a.append({"cfg": cfg, "A_lo": lo, "A_hi": hi, "correct": c, "wrong": len(g) - c, "err": round((len(g) - c) / len(g), 3)})
                md.append(f"| {cfg} | [{lo},{hi}) | {c} | {len(g)-c} | {(len(g)-c)/len(g):.0%} |")
        Ac = [f["A_cap"] for f in fr if f["correct"] == 1]
        Aw = [f["A_cap"] for f in fr if f["correct"] == 0]
        md.append(f"| {cfg} | 分布 | 正确 A 中位 {np.median(Ac):.2f} (P25 {np.percentile(Ac,25):.2f}, P75 {np.percentile(Ac,75):.2f}), n={len(Ac)} | "
                  f"错误 A 中位 {np.median(Aw):.2f} (P25 {np.percentile(Aw,25):.2f}, P75 {np.percentile(Aw,75):.2f}), n={len(Aw)} | |")
        with (out / f"ambiguity_{cfg}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_a[0]))
            w.writeheader()
            w.writerows(rows_a)
    # 拒绝高歧义:trial 级
    md.append("\n## 5. 逐步拒绝高歧义判定(主集,trial 级:首个 A≤阈值 的接受判定)\n")
    md.append("| 配置 | A 阈值 | 有输出 trial (覆盖率) | 正确 | 错误 | 准确率(有输出中) |\n|---|---|---|---|---|---|")
    for cfg in ("ours", "noocc", "mass10"):
        fr_by = defaultdict(list)
        for f in finals_rows.get(cfg, []):
            if f["acc_live"] and f["window_k"] != "" and "walking" not in f["flags"] and "stress" not in f["flags"]:
                fr_by[(f["rec"], f["window_k"])].append(f)
        trs = [t for t in subset(all_trials.get(cfg, []), "main") if t.get("has_window")]
        if not trs:
            continue
        rows_c = []
        for thr in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 9.0):
            c = w = 0
            for t in trs:
                fl = sorted([f for f in fr_by.get((t["rec"], t["k"]), []) if f["A_cap"] is not None and f["A_cap"] <= thr],
                            key=lambda f: f["t_end_rel"])
                if fl:
                    if fl[0]["object"] == t["target"]:
                        c += 1
                    else:
                        w += 1
            n = len(trs)
            rows_c.append({"cfg": cfg, "A_thr": thr, "n": n, "covered": c + w, "coverage": round((c + w) / n, 3),
                           "correct": c, "wrong": w, "acc_covered": round(c / max(c + w, 1), 3)})
            md.append(f"| {cfg} | ≤{thr} | {c+w}/{n} ({(c+w)/n:.0%}) | {c} | {w} | {c/max(c+w,1):.0%} |")
        with (out / f"coverage_{cfg}.csv").open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows_c[0]))
            wcsv.writeheader()
            wcsv.writerows(rows_c)
    # 归一化:阈值扫描(各自的原生分数),同覆盖率比错误数
    md.append("\n## 6. 归一化对照(σ=1°):capture 排序(ours) vs 质量份额排序(mass10)——各自原生分数扫阈值,比同覆盖率下的错误数\n")
    md.append("排序(接受前):ungated 首判/多数判正确率见第 1 节 `未过闸首判正确` 列与 trials_*.csv 的 ungated_majority。\n")
    md.append("| 配置 | 分数 | 阈值 | 覆盖率 | 正确 | 错误 | 准确率(有输出中) |\n|---|---|---|---|---|---|---|")
    for cfg, key in (("ours", "capture"), ("ours", "share"), ("mass10", "share"), ("mass10", "capture"), ("noocc", "capture"), ("noocc", "share"),
                     ("v2s15", "capture"), ("v2s15", "share"), ("mass15", "share")):
        fr_by = defaultdict(list)
        for f in finals_rows.get(cfg, []):
            if f["object"] and f["window_k"] != "" and "walking" not in f["flags"] and "stress" not in f["flags"]:
                fr_by[(f["rec"], f["window_k"])].append(f)
        trs = [t for t in subset(all_trials.get(cfg, []), "main") if t.get("has_window")]
        if not trs:
            continue
        rows_s = []
        grid = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0) if key == "capture" else (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9)
        for thr in grid:
            c = w = 0
            for t in trs:
                fl = sorted([f for f in fr_by.get((t["rec"], t["k"]), []) if f[key] is not None and f[key] >= thr], key=lambda f: f["t_end_rel"])
                if fl:
                    if fl[0]["object"] == t["target"]:
                        c += 1
                    else:
                        w += 1
            n = len(trs)
            rows_s.append({"cfg": cfg, "score": key, "thr": thr, "n": n, "coverage": round((c + w) / n, 3), "correct": c, "wrong": w,
                           "acc_covered": round(c / max(c + w, 1), 3)})
            md.append(f"| {cfg} | {key} | ≥{thr} | {(c+w)/n:.0%} | {c} | {w} | {c/max(c+w,1):.0%} |")
        with (out / f"sweep_{cfg}_{key}.csv").open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows_s[0]))
            wcsv.writeheader()
            wcsv.writerows(rows_s)
    # 隐藏竞争者子集:目标前面/旁边有被挡物体(noocc capture>=0.2 而 full capture<=0.05)的 trial
    md.append("\n## 6b. 有隐藏竞争者的 trial(时段内多数 final 里存在 noocc capture≥0.2 而 full capture≤0.05 的非目标候选)\n")
    hidden = {}
    for rec, card, era, flags in RECS:
        if rec not in wins or "excluded" in flags:
            continue
        p = R / rec / CFGS["noocc"]
        if not p.exists():
            continue
        fin = load_finals(p)
        for it in wins[rec]["items"]:
            if it.get("win_start_abs") is None:
                continue
            s0 = (it["stare_start_abs"] - 0.3) if it.get("stare_start_abs") is not None else it["win_start_abs"]
            w0, w1 = it["win_start_abs"], it["win_end_abs"]
            tgt = canon(it["target"])
            n = nh = 0
            names = set()
            for e in fin:
                ov = overlap(e["t_start"], e["t_end"], s0, w1)
                if not (ov >= 0.5 or (e["t_start"] >= w0 and e["t_end"] <= w1)) or not e.get("object"):
                    continue
                n += 1
                fullc = {canon(c["name"]): (c.get("capture") or 0.0) for c in ((e.get("full") or {}).get("candidates") or [])}
                hid = [c["name"] for c in e["candidates"] if c["name"] != tgt and (c.get("capture") or 0.0) >= 0.2 and fullc.get(c["name"], 0.0) <= 0.05]
                if hid:
                    nh += 1
                    names |= set(hid)
            if n:
                hidden[(rec, it["k"])] = (n, nh, names)
    H = {k for k, v in hidden.items() if v[1] >= max(1, v[0] // 2)}
    NH = {k for k, v in hidden.items() if v[1] == 0}
    md.append("| 配置 | 子集 | trial | 首个正确 | 首个错误 | 无接受 | 首判=隐藏物体 | LCS |\n|---|---|---|---|---|---|---|---|")
    for cfg in ("v1", "ours", "vis", "noocc", "mass10"):
        tmap = {(t["rec"], t["k"]): t for t in subset(all_trials.get(cfg, []), "main")}
        for label, keys in (("有隐藏竞争者", H), ("无隐藏竞争者", NH)):
            rows = [tmap[k] for k in keys if k in tmap]
            if not rows:
                continue
            c = Counter(r["outcome_first"] for r in rows)
            picked = sum(1 for r in rows if r["outcome_first"] == "wrong" and r.get("first_obj") in hidden[(r["rec"], r["k"])][2])
            md.append(f"| {cfg} | {label} | {len(rows)} | {c.get('correct',0)} | {c.get('wrong',0)} | {c.get('none',0)} | {picked} | "
                      f"{sum(r['lcs']=='hit' for r in rows)}/{len(rows)} |")
    md.append("隐藏竞争者构成:" + ", ".join(f"{n}×{c}" for n, c in Counter(n for k in H for n in hidden[k][2]).most_common(6)))
    # 配置对照
    md.append("\n## 7. 配置对照(逐 trial,主集+压力段+边走;首个接受判定的结果)\n")
    for a_, b_ in PAIRS:
        ta = {(t["rec"], t["k"]): t for t in all_trials.get(a_, [])}
        tb = {(t["rec"], t["k"]): t for t in all_trials.get(b_, [])}
        keys = [k for k in ta if k in tb]
        if not keys:
            continue
        cont = Counter((ta[k]["outcome_first"], tb[k]["outcome_first"]) for k in keys)
        contl = Counter((ta[k]["lcs"], tb[k]["lcs"]) for k in keys if ta[k]["lcs"] and tb[k]["lcs"])
        md.append(f"\n### {a_} vs {b_}(n={len(keys)})\n")
        md.append(f"| {a_} \\ {b_} | correct | wrong | none |\n|---|---|---|---|")
        for oa in ("correct", "wrong", "none"):
            md.append(f"| {oa} | " + " | ".join(str(cont.get((oa, ob), 0)) for ob in ("correct", "wrong", "none")) + " |")
        md.append(f"\nLCS 口径 hit/miss 列联:{dict(contl)}")
        rows_cmp, cases = [], []
        for k in keys:
            x, y = ta[k], tb[k]
            row = {"rec": k[0], "k": k[1], "target": x["target"], "tier": x["tier"], "occl": x["occl"], "flags": x["flags"],
                   "theta_unit": x["theta_unit"], "dist_m": x.get("dist_m"), "reliable": x["reliable"],
                   f"{a_}_first": x["outcome_first"], f"{a_}_obj": x.get("first_obj", ""), f"{a_}_cap": x.get("first_capture", ""),
                   f"{a_}_share": x.get("first_share", ""), f"{a_}_A": x.get("first_A", ""), f"{a_}_any": x.get("outcome_any", ""), f"{a_}_lcs": x["lcs"],
                   f"{b_}_first": y["outcome_first"], f"{b_}_obj": y.get("first_obj", ""), f"{b_}_cap": y.get("first_capture", ""),
                   f"{b_}_share": y.get("first_share", ""), f"{b_}_A": y.get("first_A", ""), f"{b_}_any": y.get("outcome_any", ""), f"{b_}_lcs": y["lcs"],
                   f"{a_}_best_target_cap": x.get("best_target_capture"), f"{b_}_best_target_cap": y.get("best_target_capture"),
                   f"{a_}_best_wrong_cap": x.get("best_wrong_capture"), f"{b_}_best_wrong_cap": y.get("best_wrong_capture")}
            rows_cmp.append(row)
            if x["outcome_first"] != y["outcome_first"] or x["lcs"] != y["lcs"]:
                cases.append(row)
        for name, rows in ((f"compare_{a_}_vs_{b_}", rows_cmp), (f"cases_{a_}_vs_{b_}", cases)):
            if rows:
                with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        md.append(f"\n不一致案例 {len(cases)} 条 -> cases_{a_}_vs_{b_}.csv;按分组:")
        by = Counter((c["tier"], c["occl"], f"{a_}={c[f'{a_}_first']}/{b_}={c[f'{b_}_first']}") for c in cases if c[f"{a_}_first"] != c[f"{b_}_first"])
        for (tier, occl, what), n in sorted(by.items(), key=lambda kv: -kv[1]):
            md.append(f"- {tier} {occl}: {what} × {n}")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:60]))
    print(f"-> {out}/summary.md")


if __name__ == "__main__":
    main()
