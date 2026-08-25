#!/usr/bin/env python3
"""e4_naive.py -- Table II 的朴素基线:最近质心指派(无锥体投票/无词表/无先验)。

    python Eye_Tracker/tools/e4_naive.py

对每条 E1 录像的 full 稿 intents.jsonl 做后处理:每条 final 的注视落点
(centroid_world)直接指派给 0.35 m 内最近的命名物质心,没有就当背景——
即"视线射线打到哪、谁近算谁"的裸方法,同一打分口径出 e4_naive_score.csv。
不需要重回放:落点由几何管线给出,与投票层无关,基线只换指派规则。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = Path("/home/liuchy/recordings")
ENV = {"v7": ROOT / "SceneRebuild/archive_envs/v7",
       "v8": ROOT / "SceneRebuild/archive_envs/v8",
       "v9": ROOT / "SceneRebuild/lab_result/segmentation_sam"}
RECS = [  # 与 run_e4.sh 同源
    (R/"2026_08_16/000", "e1", "v7"), (R/"2026_08_16/001", "e2", "v7"),
    (R/"2026_08_16/s1", "s1", "v7"), (R/"2026_08_16/s2", "s2", "v7"),
    (R/"2026_08_16/s3", "s3", "v7"), (R/"2026_08_16/s4", "s4", "v8"),
    (R/"2026_08_16/s6", "s6", "v8"), (R/"2026_08_18/000", "s7", "v8"),
    (R/"2026_08_20/000", "c1", "v9"), (R/"2026_08_20/001", "c2", "v9"),
    (R/"2026_08_20/002", "c4", "v9"), (R/"2026_08_20/003", "c4", "v9"),
    (R/"2026_08_25/c1_1", "c4", "v9"), (R/"2026_08_25/c1_2", "c4", "v9"),
    (R/"2026_08_25/c1_3", "c4", "v9"), (R/"2026_08_25/u3", "u3", "v9"),
]
RADIUS = 0.35


def named_centroids(era):
    d = ENV[era]
    names = {int(k): v for k, v in
             json.loads((d / "names.json").read_text()).items() if v}
    inst = {r["id"]: r for r in
            json.loads((d / "instances.json").read_text())["instances"]}
    return {nm: np.array(inst[i]["centroid"]) for i, nm in names.items() if i in inst}


def main():
    import subprocess
    for rec, card, era in RECS:
        src = rec / "intents.jsonl"
        if not src.exists():
            print(f"[!] 缺 {src}")
            continue
        C = named_centroids(era)
        names = list(C)
        M = np.stack([C[n] for n in names])
        out = rec / "intents_e4_naive.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for ln in src.open(encoding="utf-8"):
                e = json.loads(ln)
                p = e.get("centroid_world")
                if not e.get("provisional") and p:
                    d = np.linalg.norm(M - np.array(p), axis=1)
                    j = int(d.argmin())
                    if d[j] <= RADIUS:
                        e["object"] = names[j]
                        e["vote_share"] = 1.0
                    else:
                        e["object"] = "background"
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        r = subprocess.run(
            ["/home/liuchy/miniconda3/envs/nerfstudio/bin/python",
             str(ROOT / "Eye_Tracker/tools/eval_e1.py"), str(out),
             "--card", card, "--map-dir", str(ENV[era]),
             "--csv", str(rec / "e4_naive_score.csv")],
            capture_output=True, text=True)
        hit = [l for l in r.stdout.splitlines() if "命中" in l]
        print(f"{rec.name}/{card}: {hit[0] if hit else 'eval 失败'}")


if __name__ == "__main__":
    main()
