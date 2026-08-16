"""repair_balls.py -- v7 网球标签几何修复。

从 ckpt 找台面区荧光黄绿球形高斯簇(3 只网球),每球以中心 4.8cm 半径
认领全部 keep 高斯,重写 points.npz 标签(已有点改标,缺的点补录),
instances.json 同步重算几何账目。球是刚性球体,半径认领是精确修复;
E1 投票的标签密度从"顶盖薄壳"变成"整球"。
"""
import json, sys
import numpy as np
import torch
from scipy.spatial import cKDTree

CKPT = r"E:/Grasp/outputs/lab_colmap_v7/splatfacto/2026-08-15_173942/nerfstudio_models/step-000029999.ckpt"
SEG = r"E:/Grasp/data/lab_colmap_v7/segmentation_sam"
C0 = 0.28209479177387814
REGION = dict(x=(0.85, 1.95), y=(0.85, 1.70), z=(0.70, 0.92))
BALL_R = 0.0335         # 网球半径
OWN_R = 0.048           # 认领半径(球半径 + splat 毛边)
CL_R = 0.02             # 聚类连通半径

pl = torch.load(CKPT, map_location="cpu", weights_only=False)["pipeline"]
g = lambda k: pl[f"_model.gauss_params.{k}"].detach().numpy()
means = g("means").astype(np.float32)
op = 1 / (1 + np.exp(-g("opacities").squeeze()))
sc = np.exp(g("scales")).max(axis=1)
rgb = np.clip(0.5 + C0 * g("features_dc"), 0, 1)
keep = (op >= 0.5) & (sc < 0.5)
print(f"gaussians {len(means)}  keep {keep.sum()}")

m = keep.copy()
for a, (lo, hi) in zip("xyz", [REGION['x'], REGION['y'], REGION['z']]):
    i = "xyz".index(a)
    m &= (means[:, i] > lo) & (means[:, i] < hi)
r_, g_, b_ = rgb[:, 0], rgb[:, 1], rgb[:, 2]
# 网球荧光黄绿(chartreuse):g 显著大于 r 和 b;苹果 r>g、香蕉 r≈g 都进不来
tennis = (g_ > 0.40) & (g_ - r_ > 0.03) & (g_ - b_ > 0.15)
cand = np.where(m & tennis)[0]
print(f"台面区荧光绿高斯 {len(cand)}")

# 连通域聚类
pts = means[cand]
tree = cKDTree(pts)
pairs = tree.query_pairs(CL_R, output_type="ndarray")
parent = np.arange(len(pts))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a, b in pairs:
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
roots = np.array([find(i) for i in range(len(pts))])
blobs = []
for rt in np.unique(roots):
    idx = np.where(roots == rt)[0]
    if len(idx) < 50: continue
    P = pts[idx]
    lo, hi = P.min(0), P.max(0)
    ext = hi - lo
    diag = float(np.linalg.norm(ext))
    elong = float(ext.max() / max(ext.min(), 1e-6))
    blobs.append(dict(n=len(idx), c=P.mean(0), ext=ext, diag=diag, elong=elong))
print("黄绿簇:")
for b in blobs:
    print(f"  n={b['n']:>5} c=({b['c'][0]:+.2f},{b['c'][1]:+.2f},{b['c'][2]:.2f}) "
          f"ext=({b['ext'][0]:.3f},{b['ext'][1]:.3f},{b['ext'][2]:.3f}) elong={b['elong']:.1f}")
balls = [b for b in blobs if b["diag"] < 0.19 and b["elong"] < 2.6 and all(e < 0.14 for e in b["ext"])]
balls.sort(key=lambda b: -b["n"])
balls = sorted(balls[:3], key=lambda b: float(b["c"][0]))
if len(balls) != 3:
    print(f"!! 球候选 {len(balls)} 个,不是 3——中止,人工看参数"); sys.exit(1)

# 每球认领半径内全部 keep 高斯(球心用簇质心再迭代一次:半径内点的质心)
own_sets = []
kd_all = cKDTree(means[keep]); keep_idx = np.where(keep)[0]
for b in balls:
    c = b["c"]
    for _ in range(2):
        near = keep_idx[kd_all.query_ball_point(c, OWN_R)]
        c = means[near].mean(0)
    own_sets.append(np.array(sorted(near)))
    b["center"] = c

# 载入 points.npz,改标 + 补点
z = np.load(f"{SEG}/points.npz")
xyz, label = z["xyz"].copy(), z["label"].copy()
npz_tree = cKDTree(xyz)
inst_meta = json.load(open(f"{SEG}/instances.json", encoding="utf-8"))
old_by_id = {it["id"]: it for it in inst_meta["instances"]}
new_id0 = max(old_by_id) + 1
add_xyz, add_lab = [], []
stolen = {}
for bi, (b, own) in enumerate(zip(balls, own_sets)):
    bid = new_id0 + bi
    P = means[own]
    d, j = npz_tree.query(P, distance_upper_bound=1e-6)
    hit = np.isfinite(d)
    for jj in j[hit]:
        stolen[int(label[jj])] = stolen.get(int(label[jj]), 0) + 1
    label[j[hit]] = bid
    add_xyz.append(P[~hit]); add_lab.append(np.full((~hit).sum(), bid, label.dtype))
    print(f"球{bi+1} id={bid} center=({b['center'][0]:+.3f},{b['center'][1]:+.3f},{b['center'][2]:.3f}) "
          f"own={len(own)} 改标={int(hit.sum())} 补点={int((~hit).sum())}")
print("被夺点的旧标签:", {k: v for k, v in sorted(stolen.items()) if k >= 10})
xyz = np.vstack([xyz] + add_xyz).astype(np.float32)
label = np.concatenate([label] + add_lab)

# 重算 instances.json 几何账(保留 n_views/n_masks 旧值;新球 n_views=13)
out_inst = []
for i in np.unique(label):
    if i < 10: continue
    sel = label == i
    if sel.sum() < 10: continue
    P = xyz[sel]
    old = old_by_id.get(int(i), {})
    out_inst.append({"id": int(i), "n_gaussians": int(sel.sum()),
                     "n_views": old.get("n_views", 13), "n_masks": old.get("n_masks", 0),
                     "centroid": [round(float(v), 4) for v in P.mean(0)],
                     "bbox_min": [round(float(v), 4) for v in P.min(0)],
                     "bbox_max": [round(float(v), 4) for v in P.max(0)]})
out_inst.sort(key=lambda x: -x["n_gaussians"])
inst_meta["instances"] = out_inst
np.savez(f"{SEG}/points.npz", xyz=xyz, label=label)
json.dump(inst_meta, open(f"{SEG}/instances.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"写回:{len(out_inst)} 实例;球 id = {new_id0},{new_id0+1},{new_id0+2}(x 从小到大=左中右)")
