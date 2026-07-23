"""名字消解:物体表加载(池化质心)与模糊匹配。"""

from __future__ import annotations

import difflib
import json
from pathlib import Path


def load_object_table(map_dir):
    """物体名 -> 池化质心(n_gaussians 加权,与 gaze_live 同语义)。"""
    d = Path(map_dir)
    inst = json.load(open(d / "instances.json", encoding="utf-8"))["instances"]
    names = json.load(open(d / "names.json", encoding="utf-8"))
    acc = {}
    for it in inst:
        name = names.get(str(int(it["id"])), "")
        if not name:
            continue
        w = max(float(it.get("n_gaussians", 1)), 1.0)
        s, tw = acc.get(name, ([0.0, 0.0, 0.0], 0.0))
        acc[name] = ([s[i] + w * float(it["centroid"][i]) for i in range(3)], tw + w)
    return {n: [s[i] / w for i in range(3)] for n, (s, w) in acc.items()}


def resolve_named(query, table):
    """返回 (唯一命中名 或 None, 前三候选[(score,name)])。"""
    if not query or not table:
        return None, []
    if query in table:  # 精确命中直接赢:名字是标识符,残余风险交给确认门
        return query, [(1.0, query)]
    scored = []
    for name in table:
        if query == name:
            s = 1.0
        elif query in name or name in query:
            s = 0.8 + 0.2 * min(len(query), len(name)) / max(len(query), len(name))
        else:
            s = difflib.SequenceMatcher(None, query, name).ratio()
        scored.append((round(s, 3), name))
    scored.sort(reverse=True)
    best_s, best = scored[0]
    if best_s < 0.55:
        return None, scored[:3]
    if len(scored) > 1 and scored[1][0] >= max(0.55, best_s - 0.08):
        return None, scored[:3]
    return best, scored[:3]
