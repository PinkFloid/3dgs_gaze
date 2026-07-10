#!/usr/bin/env python
"""Grasp-intent detector v0: dwell-based "which object do I want to grasp".

Reads gaze_object.py's *_objects.json (per-fixation object verdicts) and turns
the fixation stream into grasp-intent events: when gaze dwells on one object
long enough, emit an intent = {target object, world coordinate for the arm,
dwell, revisit count, confidence}. This is the smallest bridge from
"what am I looking at" to "what do I want to grab".

v0 is deliberately simple (dwell threshold + revisit count). It does NOT yet
use a grasp POINT (which part) -- that needs the Bayesian ray model. And it can
only be validated on data where the true grasp target is known: record a task
("pick object X") so each trial has a ground-truth intent.

Example:
  python tools/grasp_intent.py --objects ~/recordings/2026_07_05/002/world_fixations_objects.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OBJ0 = 10  # instance labels >= this are objects; below are floor/ceiling/wall


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", required=True, help="world_fixations_objects.json from gaze_object.py.")
    p.add_argument("--dwell", type=float, default=0.8,
                   help="Accumulated dwell (s) on one object to declare grasp intent.")
    p.add_argument("--merge-gap", type=float, default=0.6,
                   help="Look-away shorter than this still counts as the same visit.")
    p.add_argument("--min-vote", type=float, default=0.5,
                   help="Ignore fixations whose object vote share is below this.")
    p.add_argument("--out", default=None, help="Default: <objects>_grasp_intent.json.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    doc = json.loads(Path(args.objects).expanduser().read_text(encoding="utf-8"))
    fx = [f for f in doc["fixations"]
          if f.get("object_label", -1) >= OBJ0 and f.get("vote_share", 0) >= args.min_vote
          and f.get("object_centroid_world")]
    fx.sort(key=lambda f: f.get("t_start", 0.0))

    # Group the fixation stream into visits (a run of gaze on one object,
    # tolerating brief look-aways up to --merge-gap).
    visits = []
    for f in fx:
        t0 = f.get("t_start", 0.0)
        t1 = f.get("t_end", t0 + f.get("duration_s", 0.0))
        if visits and visits[-1]["object"] == f["object"] and t0 - visits[-1]["t_end"] <= args.merge_gap:
            v = visits[-1]
            v["t_end"] = t1
            v["dwell"] += t1 - t0
            v["shares"].append(f["vote_share"])
            v["points"].append(f["object_centroid_world"])
        else:
            visits.append({"object": f["object"], "label": f["object_label"],
                           "t_start": t0, "t_end": t1, "dwell": t1 - t0,
                           "shares": [f["vote_share"]], "points": [f["object_centroid_world"]]})

    revisits = {}
    for v in visits:
        revisits[v["object"]] = revisits.get(v["object"], 0) + 1

    t0_all = fx[0].get("t_start", 0.0) if fx else 0.0
    intents = []
    for v in visits:
        if v["dwell"] < args.dwell:
            continue
        mean_share = sum(v["shares"]) / len(v["shares"])
        # confidence: how firmly gaze settled (dwell past threshold) x how clean
        # the object verdict was (vote share). Simple, monotone, in [0,1].
        conf = min(1.0, v["dwell"] / (2 * args.dwell)) * mean_share
        intents.append({
            "target_object": v["object"],
            "target_world": v["points"][-1],        # instance centroid: arm-consumable coordinate
            "t_intent": round(v["t_start"] - t0_all + args.dwell, 2),  # moment dwell threshold crossed
            "dwell_s": round(v["dwell"], 2),
            "revisits": revisits[v["object"]],
            "mean_vote_share": round(mean_share, 2),
            "confidence": round(conf, 2),
        })

    print(f"{len(fx)} object fixations -> {len(visits)} visits -> {len(intents)} grasp-intent events "
          f"(dwell>={args.dwell}s)\n")
    print(f"{'t(s)':>6} {'target':<16} {'dwell':>6} {'revisit':>7} {'vote':>5} {'conf':>5}  target_world (m)")
    for it in intents:
        w = it["target_world"]
        print(f"{it['t_intent']:>6.1f} {it['target_object']:<16} {it['dwell_s']:>5.1f}s "
              f"{it['revisits']:>7} {it['mean_vote_share']:>5.0%} {it['confidence']:>5.0%}  "
              f"({w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f})")

    # Also report total attention per object -- useful even below the intent threshold.
    dwell_by_obj = {}
    for v in visits:
        dwell_by_obj[v["object"]] = dwell_by_obj.get(v["object"], 0.0) + v["dwell"]
    print("\nattention summary (total dwell per object):")
    for obj, d in sorted(dwell_by_obj.items(), key=lambda kv: -kv[1]):
        print(f"  {obj:<16} {d:5.1f}s  ({revisits[obj]} visits)")

    out = Path(args.out) if args.out else Path(args.objects).expanduser().with_name(
        Path(args.objects).stem + "_grasp_intent.json")
    out.write_text(json.dumps({"source": args.objects, "params": vars(args),
                               "grasp_intents": intents}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
