#!/usr/bin/env python3
"""rejudge.py -- 离线重判:对已落盘 intents 日志里的每条 final 注视重算候选证据(同一份注视记录,只换证据/排序口径)。

回放里的注视聚类不依赖投票层:同一录像在不同投票配置下 final 流逐段相同(实测 v2s10 / v2sphere10 /
v2table10 的 t_start 序列一致)。本工具直接读 intents_e4_v2s10.jsonl 的 final 记录
(origin_world / centroid_world / distance_m / sigma_deg),对每条重算候选证据,写出同格式的
intents_e4_<cfg>.jsonl,eval_e1 / e4_table 照常打分。这样"完全相同的输入记录"由构造保证。

    conda run -n nerfstudio python Eye_Tracker/tools/rejudge.py <rec>/intents_e4_v2s10.jsonl \
        --seg-dir SceneRebuild/archive_envs/v9_rec --ckpt <ckpt> --mode noocc --out <rec>/intents_e4_v2noocc10.jsonl

--mode full   现行 v2 证据:渲染整幅 3DGS,只有可见表面投票;像素标签 = 反投影点 5cm 内最近标注高斯
              m_k = Σ_{i: L(x_i)=k} w_i α_i(cone_votes,与 gaze_live 同一函数)。
--mode noocc  忽略遮挡的锥形查询:对目标词表里每个实例 k 单独渲染(只含该实例的高斯,其他一切物体移除),
              m_k = Σ_i w_i α_i^{(k)} 1[valid_i^{(k)}],α^{(k)} 为该实例单独渲染的累计不透明度;
              valid = 0.05<depth<12 且(--noocc-range-gate on 时)depth·tmul < D + depth_margin —— 与 full 同一距离闸,
              D 取同一条注视记录的落点距离。其余全部相同:核 w_i、W、候选集(names 减 places)、r_k / c_k、
              capture=q/c、share、排序键、接受闸门。被遮挡的实例在此得到"仿佛没有遮挡物"的证据。
--mode vis    可见性=z 检验、不做最近高斯认领:逐实例渲染同 noocc,但像素只在该实例自身深度 ≤ 整幅渲染深度 + vis_tol
              时计票(即它就是该像素最前面的表面)。与 full 的差别只剩"标签来自实例自身渲染 vs 反投影点 5cm 内最近标注高斯";
              与 noocc 的差别只剩可见性。三者并列可把 full-noocc 的差异拆成 可见性 / 最近高斯认领 两部分。
--rank capture|mass  排序键:capture = 面积归一;mass = 原始锥质量份额(v1 口径)。

输出记录比 gaze_live 多:W(核总质量)、candidates 全列表(m/q/c/capture/share/miss_deg/labels)、ambiguity
(第二/第一名 capture)、ambiguity_share;--mode noocc 时附 full 口径判定(字段 full)供逐段对照。
mode 字段仍写 "cone"(brain 的接受闸只认 cone),证据口径写在 evid 字段。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaze_object import (cone_votes, disk_capture, load_background, load_places,  # noqa: E402
                         make_name_of, object_radii_by_name, pooled_centroids_by_name)
from gaze_to_world import SplatDepth  # noqa: E402

KEEP = ("t_start", "t_end", "duration_s", "centroid_world", "spread_m", "n_samples",
        "origin_world", "distance_m", "ang_spread_deg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inp", help="输入 intents jsonl(final 记录来源,通常 intents_e4_v2s10.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--seg-dir", required=True, help="录制时地图存档(points.npz/instances.json/names.json/places.json)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--places", default=None)
    p.add_argument("--mode", choices=["full", "noocc", "vis"], default="full")
    p.add_argument("--vis-tol", type=float, default=0.05, help="vis 模式:实例自身深度与整幅深度之差 ≤ 此值(m)的像素才算可见")
    p.add_argument("--rank", choices=["capture", "mass"], default="capture")
    p.add_argument("--sigma-deg", type=float, default=None, help="默认取每条记录的 sigma_deg")
    p.add_argument("--span-sigmas", type=float, default=2.0)
    p.add_argument("--patch", type=int, default=33)
    p.add_argument("--patch-deg", type=float, default=6.0 / 33)
    p.add_argument("--hit-eps", type=float, default=0.05)
    p.add_argument("--depth-margin", type=float, default=0.5)
    p.add_argument("--noocc-range-gate", choices=["on", "off"], default="on")
    p.add_argument("--check", action="store_true", help="full 模式:与输入记录的 object/capture 对账")
    return p.parse_args()


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


def patch_cam(origin, direction, half_angle, S):
    """与 SplatDepth.patch_along_ray 完全相同的相机几何(w2c, K, 像素射线方向, z深度->射线长度倍率)。"""
    z = direction / np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([x, y, z])
    w2c[:3, 3] = -w2c[:3, :3] @ origin
    f = (S / 2) / np.tan(half_angle)
    K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]])
    n = (np.arange(S) + 0.5 - S / 2) / f
    nx, ny = np.meshgrid(n, n)
    dirs_cam = np.stack([nx, ny, np.ones_like(nx)], axis=-1)
    tmul = np.linalg.norm(dirs_cam, axis=-1)
    dirs = (dirs_cam / tmul[..., None]) @ np.stack([x, y, z])
    return w2c, K, dirs, tmul


def render_subset(splat, sub, w2c, K, S):
    torch = splat.torch
    vm = torch.tensor(w2c, dtype=torch.float32, device=splat.dev).unsqueeze(0)
    Kt = torch.tensor(K, dtype=torch.float32, device=splat.dev).unsqueeze(0)
    out, alpha, _ = splat.rasterization(
        splat.means[sub], splat.quats[sub], splat.scales[sub], splat.opac[sub], splat.colors[sub],
        vm, Kt, S, S, sh_degree=3, render_mode="RGB+ED", rasterize_mode="classic")
    return out[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


def noocc_votes(splat, lab_idx, origin, pt, sigma_rad, span, S, depth_margin, range_gate=True,
                full_depth=None, vis_tol=0.05):
    """full_depth 非空时 = vis 模式:实例像素只在 depth_k <= full_depth + vis_tol 时计票(z 检验可见性)。"""
    d0 = pt - origin
    dist0 = float(np.linalg.norm(d0))
    if dist0 < 0.05:
        return {}, None
    w2c, K, dirs, tmul = patch_cam(origin, d0 / dist0, span * sigma_rad, S)
    theta = np.arccos(np.clip(dirs @ (d0 / dist0), -1.0, 1.0))
    w = np.exp(-theta ** 2 / (2 * sigma_rad ** 2))
    votes = {}
    for lab, sub in lab_idx.items():
        depth, alpha = render_subset(splat, sub, w2c, K, S)
        ok = (depth > 0.05) & (depth < 12.0) & (alpha > 0)
        if range_gate and depth_margin > 0:
            ok &= (depth * tmul) < dist0 + depth_margin
        if full_depth is not None:
            ok &= depth <= full_depth + vis_tol
        m = float((w * alpha)[ok].sum())
        if m > 0:
            votes[int(lab)] = m
    return votes, {"W": float(w.sum()), "theta": theta.ravel(), "w": w.ravel()}


def rank_all(votes, kern, name_of, targets, cents, radii, sigma_rad, dist, rank_by):
    """rank_votes 的全候选版:同样的池化/分母/capture 定义,候选不截断,多记 m/c/W 与歧义比。"""
    if not votes:
        return None
    W = kern["W"] if kern is not None else float(sum(votes.values()))
    pooled = {}
    for lab, v in votes.items():
        nm = name_of(lab)
        p = pooled.setdefault(nm, {"v": 0.0, "labels": {}})
        p["v"] += v
        p["labels"][int(lab)] = p["labels"].get(int(lab), 0.0) + v
    tgt = {n: p for n, p in pooled.items() if n in targets}
    inv = {n: p for n, p in pooled.items() if n not in targets}
    T = float(sum(p["v"] for p in tgt.values()))
    p_none = max(0.0, 1.0 - T / W) if W > 0 else 1.0
    surf = max(inv.items(), key=lambda kv: kv[1]["v"]) if inv else None
    out = {"p_none": _r(p_none), "rank_by": rank_by, "W": _r(W, 4),
           "surface": surf[0] if surf else None,
           "surface_q": _r(surf[1]["v"] / W) if (surf and W > 0) else 0.0}
    if T <= 0:
        out.update(object=None, object_label=-1, vote_share=0.0, q=0.0, capture=None, miss_deg=None,
                   object_centroid_world=None, candidates=[], ambiguity=None, ambiguity_share=None)
        return out
    sigma_deg = math.degrees(sigma_rad)
    cands = []
    for n, p in tgt.items():
        m = p["v"]
        q, share = m / W, m / T
        c = cap = miss = None
        if kern is not None and n in radii and dist:
            c = disk_capture(kern, math.atan2(radii[n], dist))
            if c and c > 0:
                cap = q / c
                miss = max(0.0, sigma_deg * math.sqrt(-2.0 * math.log(min(cap, 1.0)))) if cap > 0 else None
        cands.append({"name": n, "m": _r(m, 5), "q": _r(q, 4), "c": _r(c, 4), "capture": _r(cap),
                      "share": _r(share), "miss_deg": _r(miss, 2), "labels": sorted(p["labels"]),
                      "_best": max(p["labels"], key=lambda l: p["labels"][l]),
                      "_cap": cap if cap is not None else -1.0, "_share": share})
    by_cap = rank_by == "capture" and all(c["_cap"] >= 0 for c in cands)
    cands.sort(key=(lambda c: c["_cap"]) if by_cap else (lambda c: c["_share"]), reverse=True)
    best = cands[0]
    amb = amb_s = None
    if len(cands) > 1:
        if cands[0]["_cap"] > 0:
            amb = _r(max(cands[1]["_cap"], 0.0) / cands[0]["_cap"])
        amb_s = _r(cands[1]["_share"] / cands[0]["_share"]) if cands[0]["_share"] > 0 else None
    else:
        amb, amb_s = 0.0, 0.0
    out.update(object=best["name"], object_label=int(best["_best"]), vote_share=best["share"], q=best["q"],
               capture=best["capture"], miss_deg=best["miss_deg"],
               object_centroid_world=cents.get(best["name"]), ambiguity=amb, ambiguity_share=amb_s,
               candidates=[{k: v for k, v in c.items() if not k.startswith("_")} for c in cands])
    return out


def main():
    a = parse_args()
    seg = Path(a.seg_dir)
    z = np.load(seg / "points.npz")
    xyz, label = z["xyz"], z["label"]
    meta = json.loads((seg / "instances.json").read_text(encoding="utf-8"))
    names = json.loads((seg / "names.json").read_text(encoding="utf-8")) if (seg / "names.json").exists() else {}
    bg = load_background(meta)
    name_of = make_name_of(bg, names)
    places = load_places(seg, a.places)
    targets = {v for v in names.values() if v and v not in places}
    radii = object_radii_by_name(xyz, label, names, only=targets)
    cents = pooled_centroids_by_name(meta["instances"], names)
    tree = cKDTree(xyz)
    splat = SplatDepth(Path(a.ckpt))
    splat.depth_along_ray(np.zeros(3), np.array([0.0, 0.0, 1.0]))  # CUDA 预热(首次渲染有冷启动伪影)
    lab_idx = None
    if a.mode in ("noocc", "vis"):
        means = splat.means.detach().cpu().numpy()
        d, idx = cKDTree(means).query(xyz, k=1)
        if float(d.max()) > 1e-6:
            raise SystemExit(f"points.npz 与 ckpt 高斯不一一对应(max d={d.max():.4f})")
        torch = splat.torch
        lab_idx = {}
        for lab in np.unique(label):
            nm = names.get(str(int(lab)), "")
            if nm and nm in targets:
                lab_idx[int(lab)] = torch.as_tensor(idx[label == lab], device=splat.dev)
        print(f"noocc: {len(lab_idx)} target instances -> " +
              ", ".join(f"{names[str(l)]}#{l}:{len(v)}" for l, v in lab_idx.items()))
        w2c, K, _, _ = patch_cam(np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.05, a.patch)
        render_subset(splat, next(iter(lab_idx.values())), w2c, K, a.patch)  # 子集渲染预热
    print(f"targets {sorted(targets)}  places {sorted(places)}  mode {a.mode}  rank {a.rank}")

    n = n_same = n_obj_same = 0
    t0 = time.time()
    with open(a.out, "w", encoding="utf-8") as fo:
        for ln in open(a.inp, encoding="utf-8"):
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("provisional"):
                continue
            if e.get("origin_world") is None or e.get("centroid_world") is None:
                fo.write(json.dumps(e, ensure_ascii=False) + "\n")
                continue
            o = np.asarray(e["origin_world"], float)
            c = np.asarray(e["centroid_world"], float)
            sigma_deg = a.sigma_deg if a.sigma_deg else float(e.get("sigma_deg") or 1.0)
            sr = np.radians(sigma_deg)
            S = max(a.patch, int(round(2 * a.span_sigmas * sigma_deg / a.patch_deg)) | 1)
            dist = e.get("distance_m") or float(np.linalg.norm(c - o))
            v_full, kern = cone_votes(splat, tree, label, o, c, sr, a.span_sigmas, S, a.hit_eps, a.depth_margin)
            rk_full = rank_all(v_full, kern, name_of, targets, cents, radii, sr, dist, a.rank)
            if a.mode in ("noocc", "vis"):
                fd = None
                if a.mode == "vis":
                    fd, _, _, _ = splat.patch_along_ray(o, (c - o) / np.linalg.norm(c - o), a.span_sigmas * sr, S)
                v_no, kern2 = noocc_votes(splat, lab_idx, o, c, sr, a.span_sigmas, S, a.depth_margin,
                                          range_gate=(a.noocc_range_gate == "on"), full_depth=fd, vis_tol=a.vis_tol)
                rk = rank_all(v_no, kern2, name_of, targets, cents, radii, sr, dist, a.rank)
                if rk is None and kern2 is not None:  # 没有任何目标实例落进锥:无目标
                    rk = {"p_none": 1.0, "rank_by": a.rank, "W": _r(kern2["W"], 4), "surface": None,
                          "surface_q": 0.0, "object": None, "object_label": -1, "vote_share": 0.0, "q": 0.0,
                          "capture": None, "miss_deg": None, "object_centroid_world": None, "candidates": [],
                          "ambiguity": None, "ambiguity_share": None}
                if rk is not None and rk_full is not None:  # 无效票表面(场所/地板)只有整幅渲染知道
                    rk["surface"], rk["surface_q"] = rk_full["surface"], rk_full["surface_q"]
            else:
                rk = rk_full
            if rk is None:  # 退化注视(gaze_live 不落盘);保留一条空判定维持流对齐
                rk = {"p_none": 1.0, "rank_by": a.rank, "W": None, "surface": None, "surface_q": 0.0,
                      "object": None, "object_label": -1, "vote_share": 0.0, "q": 0.0, "capture": None,
                      "miss_deg": None, "object_centroid_world": None, "candidates": [],
                      "ambiguity": None, "ambiguity_share": None}
            rec = {k: e[k] for k in KEEP if k in e}
            rec.update(rk)
            rec.update(sigma_deg=round(sigma_deg, 2), mode="cone", evid=a.mode, provisional=False,
                       topic="gaze.intent", src_object=e.get("object"), src_capture=e.get("capture"),
                       src_share=e.get("vote_share"))
            if a.mode in ("noocc", "vis") and rk_full is not None:
                rec["full"] = {k: rk_full.get(k) for k in
                               ("object", "object_label", "vote_share", "q", "capture", "miss_deg",
                                "p_none", "ambiguity", "ambiguity_share")}
                rec["full"]["candidates"] = rk_full["candidates"][:5]
            if a.check and a.mode == "full":
                n += 1
                n_obj_same += rec["object"] == e.get("object")
                n_same += (rec["object"] == e.get("object")
                           and abs((rec.get("capture") or 0) - (e.get("capture") or 0)) < 0.02)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.check and n:
        print(f"check: {n} finals, object 一致 {n_obj_same}/{n}, object+capture(±0.02) 一致 {n_same}/{n}")
    print(f"wrote {a.out}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
