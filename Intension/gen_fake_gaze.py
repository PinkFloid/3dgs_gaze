#!/usr/bin/env python3
"""gen_fake_gaze.py -- 造一段假的 gaze.intent 流,用于本地模拟(不开眼动仪/不开 gaze_live)。

场景:瞥碗0.5s -> 瞥地板 -> 盯水杯2.0s -> 瞥地板0.3s -> 续盯水杯3.4s(跨瞥离合并,
因果 dwell 在 4.8s 处触发)-> 移开。配合 stare_to_grasp.py --replay 使用:

    python Intension/gen_fake_gaze.py /tmp/fake.jsonl
    python Intension/stare_to_grasp.py --replay /tmp/fake.jsonl \
        --skill-endpoint tcp://127.0.0.1:5583      # 到 4.8s 会弹确认,敲 y
"""
import json
import sys

CUP_C = [0.50, -0.30, 0.86]
BOWL_C = [0.80, -0.10, 0.85]


def ev(t0, dur, obj, label, centroid, vote=0.78, prov=True):
    t1 = t0 + dur
    return {"t_start": t0, "t_end": t1, "duration_s": round(dur, 3),
            "centroid_world": centroid or [1.0, 1.0, 0.0], "spread_m": 0.02,
            "n_samples": max(4, int(dur * 30)), "origin_world": [2.0, -1.5, 1.4],
            "distance_m": 1.7, "ang_spread_deg": 0.7,
            "object": obj, "object_label": label, "vote_share": vote,
            "object_centroid_world": centroid,
            "candidates": [{"name": obj, "share": vote, "labels": [label]}],
            "p_none": 0.08, "sigma_deg": 1.2, "mode": "cone",
            "provisional": prov, "judge_ms": 3.0, "topic": "gaze.intent"}


def fixation(t0, total, obj, label, centroid, vote=0.78, prov_every=0.4):
    out, d = [], prov_every
    while d < total - 1e-9:
        out.append(ev(t0, d, obj, label, centroid, vote, prov=True))
        d += prov_every
    out.append(ev(t0, total, obj, label, centroid, vote, prov=False))
    return out


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake.jsonl"
    evs = []
    evs += fixation(100.0, 0.5, "碗", 12, BOWL_C)      # 短瞥:不触发
    evs += fixation(100.8, 0.9, "floor", 3, None)      # 背景:被门滤掉
    evs += fixation(102.0, 2.0, "水杯", 11, CUP_C)     # 第一段注视
    evs += fixation(104.1, 0.3, "floor", 3, None)      # 瞥离 < merge-gap
    evs += fixation(104.5, 3.4, "水杯", 11, CUP_C)     # 合并后 4.8s 处触发
    evs += fixation(108.4, 1.2, "碗", 12, BOWL_C)      # 移开,结账
    evs.sort(key=lambda e: e["t_end"])
    with open(out, "w", encoding="utf-8") as f:
        for e in evs:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{out}: {len(evs)} events (盯水杯累计 5.4s,4.8s 处触发)")


if __name__ == "__main__":
    main()


