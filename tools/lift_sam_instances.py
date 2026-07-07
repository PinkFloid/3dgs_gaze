#!/usr/bin/env python
"""Offline splat segmentation v2: SAM masks lifted onto the gaussians, 3D-consensus clustering.

Replaces the object layer of segment_splat.py (whose voxel connected-components
cannot separate touching objects: cup-on-desk merges into the desk chain). The
background layer (floor / ceiling / walls by height + room-bound rules) is kept
unchanged. No queries, no CLIP -- naming stays manual via names.json + thumbs.

Per view: undistort photo -> SAM automatic masks -> back-project mask pixels
through the gsplat-rendered depth -> nearest-gaussian match, so each mask
becomes a set of gaussian IDs. Masks from different views that hit the same
gaussians are the same object (edge if gaussian-set IoU >= --edge-iou), and
connected components over that mask graph are the instances. A part->whole
containment merge then reassembles SAM's part-level masks (dog legs, desk
corners) into their parent object. Per-gaussian majority vote over the
instances' masks yields the final labels: association lives in 3D on the
gaussians themselves -- the 2D bbox + CLIP matching that broke the old Grasp
build_object_map.py is gone entirely.

Runs on the mapping machine (4090: photos + ckpt + segment-anything needed).
Output contract identical to segment_splat.py, into --out-dir
(default lab_result/segmentation_sam; point gaze_object.py --seg-dir at it):

  points.npz        xyz + label (0 floor, 1 ceiling, 2-5 walls, >=10 instances)
  instances.json    id, centroid, bbox, n_gaussians, n_views, n_masks
  names.json        {"10": "", ...} template -- fill names by hand (kept if exists)
  thumbs/           per-instance highlight renders (segment_splat.render_thumbs)
  render_check.jpg  photo|render|blend of the first view -- must look aligned

SAM vit_h strongly preferred over vit_b (vit_b's low mask scores helped the old
pipeline miss the robot dog): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLOOR, CEILING = 0, 1
WALL_XMIN, WALL_XMAX, WALL_YMIN, WALL_YMAX = 2, 3, 4, 5
OBJ0 = 10
BG_NAMES = {0: "floor", 1: "ceiling", 2: "wall", 3: "wall", 4: "wall", 5: "wall"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=r"E:\Grasp\data\lab_colmap",
                   help="Dataset dir with transforms json + images/ (mapping machine).")
    p.add_argument("--transforms", default="transforms_aligned.json")
    p.add_argument("--ckpt", default=None,
                   help="Default: newest step-*.ckpt under lab_result/, else under E:\\Grasp\\outputs.")
    p.add_argument("--out-dir", default=str(root / "lab_result/segmentation_sam"))
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="Debug: only the first N selected frames.")
    # SAM
    p.add_argument("--sam-checkpoint", default=None,
                   help="Default: best sam_vit_*.pth in tools/ or E:\\Grasp\\tools (h > l > b).")
    p.add_argument("--long-side", type=int, default=2048, help="Photo/depth working resolution.")
    p.add_argument("--points-per-side", type=int, default=32)
    p.add_argument("--crop-layers", type=int, default=1,
                   help="SAM crop_n_layers; 1 = extra 2x-zoom pass, rescues small objects (cups).")
    p.add_argument("--crop-points-downscale", type=int, default=1)
    p.add_argument("--points-per-batch", type=int, default=16,
                   help="SAM decoder batch; its GPU post-processing runs at full crop res, 64 OOMs at 2048px+vit_h.")
    p.add_argument("--pred-iou", type=float, default=0.8)
    p.add_argument("--stability", type=float, default=0.9)
    p.add_argument("--min-mask-area", type=int, default=300, help="px at working resolution.")
    p.add_argument("--max-mask-frac", type=float, default=0.8,
                   help="Larger masks dropped. Keep high: close-up whole-desk masks are legit "
                        "instances (bg-frac already kills floor/wall spans).")
    # lifting
    p.add_argument("--max-px-samples", type=int, default=4000, help="Mask pixels sampled for unprojection.")
    p.add_argument("--match-eps", type=float, default=0.04,
                   help="Max point-to-gaussian distance (m); also kills silhouette depth fliers.")
    p.add_argument("--depth-max", type=float, default=12.0)
    p.add_argument("--bg-frac", type=float, default=0.6,
                   help="Drop mask if more than this fraction of hits land on floor/wall/ceiling gaussians.")
    # clustering / voting
    p.add_argument("--edge-iou", type=float, default=0.3,
                   help="Gaussian-set IoU linking two masks. Same-object masks from different "
                        "viewpoints hit different surfaces, so cross-view IoU runs low.")
    p.add_argument("--merge-containment", type=float, default=0.7,
                   help="Merge a component into a bigger well-supported one covering this "
                        "fraction of its gaussians (reassembles SAM part masks: legs->dog).")
    p.add_argument("--min-views", type=int, default=3, help="Distinct views required per instance.")
    p.add_argument("--min-votes", type=int, default=2, help="Masks required to claim a gaussian.")
    p.add_argument("--min-gaussians", type=int, default=80)
    p.add_argument("--min-size", type=float, default=0.06, help="Min instance bbox diagonal (m).")
    # background rules (same defaults as segment_splat.py)
    p.add_argument("--min-opacity", type=float, default=0.5)
    p.add_argument("--floor-z", type=float, default=0.08)
    p.add_argument("--ceiling-z", type=float, default=2.5)
    p.add_argument("--wall-margin", type=float, default=0.15)
    p.add_argument("--thumbs", type=int, default=2)
    p.add_argument("--previews", type=int, default=8,
                   help="Instance-colored photo overlays for QA/naming (0 = skip).")
    return p.parse_args()


def color_for(oid: int) -> tuple[int, int, int]:
    import cv2
    rng = np.random.default_rng(oid * 7919 + 13)
    hsv = np.array([[[int(rng.integers(0, 180)), 220, 255]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def find_ckpt(arg: str | None, root: Path) -> Path:
    if arg:
        return Path(arg)
    for base in (root / "lab_result", Path(r"E:\Grasp\outputs")):
        cands = list(base.rglob("step-*.ckpt")) if base.exists() else []
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    raise SystemExit("No step-*.ckpt found; pass --ckpt.")


def find_sam(arg: str | None, root: Path) -> tuple[Path, str]:
    if arg:
        path = Path(arg)
    else:
        path = None
        for kind in ("vit_h", "vit_l", "vit_b"):
            for base in (root / "tools", Path(r"E:\Grasp\tools")):
                hits = sorted(base.glob(f"sam_{kind}_*.pth")) if base.exists() else []
                if hits:
                    path = hits[0]
                    break
            if path:
                break
        if path is None:
            raise SystemExit("No sam_vit_*.pth found; pass --sam-checkpoint.")
    for kind in ("vit_h", "vit_l", "vit_b"):
        if kind in path.name:
            return path, kind
    raise SystemExit(f"Cannot tell vit_b/l/h from filename: {path.name}")


def dedup_masks(masks: list[dict]) -> list[dict]:
    """Drop near-identical masks (SAM crop passes re-find the same object)."""
    masks = sorted(masks, key=lambda m: -m["predicted_iou"])
    kept: list[dict] = []
    for m in masks:
        seg = m["segmentation"][::4, ::4]
        dup = False
        for k in kept:
            ks = k["segmentation"][::4, ::4]
            inter = np.logical_and(seg, ks).sum()
            if inter / max(seg.sum() + ks.sum() - inter, 1) > 0.9:
                dup = True
                break
        if not dup:
            kept.append(m)
    return kept


def main() -> int:
    args = parse_args()
    import cv2
    import torch
    from PIL import Image
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from gsplat import rasterization
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    import segment_splat  # background rules + thumbs live there

    root = Path(__file__).resolve().parent.parent
    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = find_ckpt(args.ckpt, root)
    sam_path, sam_kind = find_sam(args.sam_checkpoint, root)
    dev = torch.device("cuda")
    print(f"ckpt: {ckpt}\nSAM:  {sam_path} ({sam_kind})")

    # ---- gaussians + background labels (identical rules to segment_splat) ----
    sd = torch.load(ckpt, map_location="cpu")["pipeline"]
    g = lambda n: sd[f"_model.gauss_params.{n}"]
    xyz = g("means").numpy().astype(np.float32)
    opac = 1 / (1 + np.exp(-g("opacities").numpy().reshape(-1)))
    max_scale = np.exp(g("scales").numpy()).max(axis=1)
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    keep = (opac >= args.min_opacity) & (max_scale < 0.5) & \
           np.all((xyz > lo - 0.2) & (xyz < hi + 0.2), axis=1)
    xyz_k = xyz[keep]
    print(f"{keep.sum()}/{len(xyz)} gaussians kept, "
          f"room x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}] z[{lo[2]:.1f},{hi[2]:.1f}]")

    label = np.full(len(xyz_k), -1, np.int32)
    z = xyz_k[:, 2]
    label[z < args.floor_z] = FLOOR
    label[z > args.ceiling_z] = CEILING
    m = label == -1
    label[m & (xyz_k[:, 0] < lo[0] + args.wall_margin)] = WALL_XMIN
    label[m & (xyz_k[:, 0] > hi[0] - args.wall_margin)] = WALL_XMAX
    m = label == -1
    label[m & (xyz_k[:, 1] < lo[1] + args.wall_margin)] = WALL_YMIN
    label[m & (xyz_k[:, 1] > hi[1] - args.wall_margin)] = WALL_YMAX
    tree = cKDTree(xyz_k)

    # ---- renderer (full model: depth must match what the photos saw) ----
    means_t = g("means").to(dev)
    quats_t = torch.nn.functional.normalize(g("quats").to(dev), dim=-1)
    scales_t = torch.exp(g("scales").to(dev))
    opac_t = torch.sigmoid(g("opacities").to(dev)).squeeze(-1)
    colors_t = torch.cat([g("features_dc").unsqueeze(1), g("features_rest")], dim=1).to(dev)

    meta = json.loads((dataset / args.transforms).read_text(encoding="utf-8"))
    if "fl_x" not in meta:
        raise SystemExit("Expected single-camera transforms json.")
    s = args.long_side / max(meta["w"], meta["h"])
    W, H = int(round(meta["w"] * s)), int(round(meta["h"] * s))
    K = np.array([[meta["fl_x"] * s, 0, meta["cx"] * s],
                  [0, meta["fl_y"] * s, meta["cy"] * s], [0, 0, 1]])
    dist = np.array([meta.get(k, 0.0) for k in ("k1", "k2", "p1", "p2")])
    Kt = torch.from_numpy(K).float().to(dev).unsqueeze(0)

    frames = [f for f in sorted(meta["frames"], key=lambda f: f["file_path"])
              if (dataset / f["file_path"]).exists()][:: args.every]
    if args.limit:
        frames = frames[: args.limit]
    print(f"{len(frames)} views at {W}x{H}")

    sam = sam_model_registry[sam_kind](checkpoint=str(sam_path)).to(dev)
    mask_gen = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side, points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou, stability_score_thresh=args.stability,
        crop_n_layers=args.crop_layers,
        crop_n_points_downscale_factor=args.crop_points_downscale, min_mask_region_area=100)

    # ---- per view: masks -> gaussian ID sets ----
    mask_frame: list[int] = []
    mask_gids: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for i, frame in enumerate(frames):
        stem = Path(frame["file_path"]).stem
        img = Image.open(dataset / frame["file_path"]).convert("RGB")
        arr = np.array(img.resize((W, H), Image.LANCZOS))
        arr = cv2.undistort(arr, K, dist, None, K)  # pinhole K = render K

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)  # nerfstudio GL c2w
        c2w_cv = c2w.copy()
        c2w_cv[:3, 1] *= -1
        c2w_cv[:3, 2] *= -1
        vm = torch.from_numpy(np.linalg.inv(c2w_cv)).float().to(dev).unsqueeze(0)
        with torch.no_grad():
            out, alpha_t, _ = rasterization(means_t, quats_t, scales_t, opac_t, colors_t,
                                            vm, Kt, W, H, sh_degree=3, render_mode="RGB+ED")
        depth = out[0, ..., 3].cpu().numpy()
        alpha = alpha_t[0, ..., 0].cpu().numpy()
        if i == 0:  # alignment insurance: photo | render | blend must coincide
            render = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            blend = cv2.addWeighted(arr, 0.5, render, 0.5, 0)
            cv2.imwrite(str(out_dir / "render_check.jpg"),
                        cv2.cvtColor(np.concatenate([arr, render, blend], axis=1), cv2.COLOR_RGB2BGR))

        masks = [m for m in mask_gen.generate(arr)
                 if args.min_mask_area <= m["area"] <= args.max_mask_frac * H * W]
        torch.cuda.empty_cache()
        masks = dedup_masks(masks)

        lifted = 0
        for mk in masks:
            ys, xs = np.nonzero(mk["segmentation"])
            if len(ys) > args.max_px_samples:
                sel = rng.choice(len(ys), args.max_px_samples, replace=False)
                ys, xs = ys[sel], xs[sel]
            d = depth[ys, xs]
            ok = (d > 0.05) & (d < args.depth_max) & (alpha[ys, xs] > 0.7)
            if ok.sum() < 30:
                continue
            ys, xs, d = ys[ok], xs[ok], d[ok]
            pc = np.stack([(xs + 0.5 - K[0, 2]) / K[0, 0] * d,
                           (ys + 0.5 - K[1, 2]) / K[1, 1] * d, d], axis=1)
            pts = pc @ c2w_cv[:3, :3].T + c2w_cv[:3, 3]
            dd, idx = tree.query(pts, k=1, distance_upper_bound=args.match_eps, workers=-1)
            hit = np.isfinite(dd)
            if hit.sum() < 30:
                continue
            gids = idx[hit]
            if (label[gids] >= 0).mean() > args.bg_frac:
                continue  # floor / wall / ceiling mask
            gids = np.unique(gids[label[gids] == -1])
            if len(gids) < 30:
                continue
            mask_frame.append(i)
            mask_gids.append(gids)
            lifted += 1
        print(f"[{i + 1}/{len(frames)}] {stem}: {len(masks)} masks, {lifted} lifted "
              f"-> {len(mask_gids)} total")

    del sam, mask_gen
    torch.cuda.empty_cache()
    if not mask_gids:
        raise SystemExit("No masks lifted -- check render_check.jpg for pose/undistort mismatch.")

    # ---- mask graph: edge = gaussian-set IoU, components = instances ----
    n_masks = len(mask_gids)
    rows = np.concatenate([np.full(len(gi), j, np.int64) for j, gi in enumerate(mask_gids)])
    cols = np.concatenate(mask_gids)
    M = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(n_masks, len(xyz_k)))
    sizes = np.asarray(M.sum(axis=1)).ravel()
    inter = (M @ M.T).tocoo()
    iou = inter.data / (sizes[inter.row] + sizes[inter.col] - inter.data)
    e = (iou >= args.edge_iou) & (inter.row != inter.col)
    adj = sparse.csr_matrix((np.ones(e.sum(), np.int8), (inter.row[e], inter.col[e])),
                            shape=(n_masks, n_masks))
    n_comp, comp = connected_components(adj, directed=False)
    frame_ids = np.array(mask_frame)
    print(f"{n_masks} masks -> {n_comp} components at IoU >= {args.edge_iou}")

    # part -> whole merge. IoU edges only join same-granularity masks, so SAM's
    # part/whole ambiguity (dog legs vs whole dog, desk corners vs desk) leaves
    # parts as separate components. A component whose covered gaussians lie
    # mostly inside the coverage of a bigger component is a part of it. The
    # attractor must itself be seen in >= min-views views: a one-off union-blob
    # mask (desk+cup in one frame) never gets that support, so it cannot swallow
    # real objects.
    for _ in range(5):
        cids = np.unique(comp)
        covs, views_n = [], []
        for c in cids:
            mrows = np.flatnonzero(comp == c)
            hits = np.asarray(M[mrows].sum(axis=0)).ravel()
            covs.append(np.flatnonzero(hits >= min(2, len(mrows))))
            views_n.append(len(set(frame_ids[mrows])))
        sizes_c = np.array([len(cv) for cv in covs])
        crow = np.concatenate([np.full(len(cv), k, np.int64) for k, cv in enumerate(covs)])
        Cov = sparse.csr_matrix((np.ones(len(crow), np.float32), (crow, np.concatenate(covs))),
                                shape=(len(cids), len(xyz_k)))
        inter_c = (Cov @ Cov.T).tocoo()
        target: dict[int, tuple[float, int]] = {}
        for a, b, v in zip(inter_c.row, inter_c.col, inter_c.data):
            if a == b or views_n[b] < args.min_views or sizes_c[b] <= sizes_c[a]:
                continue
            cont = v / max(sizes_c[a], 1)
            if cont >= args.merge_containment and cont > target.get(a, (0.0, -1))[0]:
                target[a] = (cont, b)
        if not target:
            break
        remap = np.arange(len(cids))
        for a, (_, b) in target.items():
            remap[a] = b
        while True:  # follow chains (corner -> desk-half -> desk); acyclic since size strictly grows
            nxt = remap[remap]
            if np.array_equal(nxt, remap):
                break
            remap = nxt
        lut = {c: cids[remap[k]] for k, c in enumerate(cids)}
        comp = np.array([lut[c] for c in comp])
        print(f"  part-merge: absorbed {len(target)} components")

    comp_views = {c: len(set(frame_ids[comp == c])) for c in np.unique(comp)}
    good = [c for c in comp_views if comp_views[c] >= args.min_views]
    print(f"{len(comp_views)} components after part-merge, {len(good)} with >= {args.min_views} views")

    # ---- per-gaussian vote ----
    rank = {c: r for r, c in enumerate(good)}
    in_good = np.array([comp[j] in rank for j in range(n_masks)])
    C = sparse.csr_matrix((np.ones(in_good.sum(), np.float32),
                           (np.array([rank[comp[j]] for j in range(n_masks) if in_good[j]]),
                            np.flatnonzero(in_good))), shape=(len(good), n_masks))
    counts = (C @ M).tocsc()
    best = np.asarray(counts.argmax(axis=0)).ravel()
    maxv = np.asarray(counts.max(axis=0).todense()).ravel()
    voted = np.where((maxv >= args.min_votes) & (label == -1), best, -1)

    # ---- instances: filter, renumber by size, write ----
    instances = []
    next_id = OBJ0
    order = sorted(range(len(good)), key=lambda r: -(voted == r).sum())
    for r in order:
        sel = voted == r
        n = int(sel.sum())
        if n < args.min_gaussians:
            continue
        p = xyz_k[sel]
        bb_lo, bb_hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
        if np.linalg.norm(bb_hi - bb_lo) < args.min_size:
            continue
        label[sel] = next_id
        c = good[r]
        instances.append({
            "id": next_id, "n_gaussians": n,
            "n_views": comp_views[c], "n_masks": int((comp == c).sum()),
            "centroid": p.mean(axis=0).round(3).tolist(),
            "bbox_min": bb_lo.round(3).tolist(), "bbox_max": bb_hi.round(3).tolist(),
            "size_m": (bb_hi - bb_lo).round(2).tolist(),
        })
        next_id += 1
    print(f"{len(instances)} instances kept (>= {args.min_gaussians} gaussians, >= {args.min_size}m)")

    retained = label >= 0
    np.savez(out_dir / "points.npz", xyz=xyz_k[retained], label=label[retained])
    (out_dir / "instances.json").write_text(json.dumps({
        "ckpt": str(ckpt), "method": "sam-lift-consensus",
        "sam": sam_path.name, "every": args.every, "long_side": args.long_side,
        "edge_iou": args.edge_iou, "min_views": args.min_views,
        "background": BG_NAMES, "instances": instances}, indent=2), encoding="utf-8")
    names_path = out_dir / "names.json"
    names = json.loads(names_path.read_text(encoding="utf-8")) if names_path.exists() else {}
    for inst in instances:
        names.setdefault(str(inst["id"]), "")
    names_path.write_text(json.dumps(names, indent=2, ensure_ascii=False), encoding="utf-8")

    for inst in instances[:30]:
        c, sz = inst["centroid"], inst["size_m"]
        print(f"  id {inst['id']:>3}: {inst['n_gaussians']:>7} gaussians  {inst['n_views']:>3} views  "
              f"center ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})  size {sz[0]}x{sz[1]}x{sz[2]}m")

    # instance-colored overlays on the real photos: one glance shows what every
    # pixel region got assigned to (and what got nothing) -- use for naming
    if args.previews > 0 and instances:
        pv_dir = out_dir / "preview"
        pv_dir.mkdir(exist_ok=True)
        rng_pv = np.random.default_rng(2)
        step = max(1, len(frames) // args.previews)
        for frame in frames[::step][: args.previews]:
            stem = Path(frame["file_path"]).stem
            img = Image.open(dataset / frame["file_path"]).convert("RGB")
            arr = cv2.undistort(np.array(img.resize((W, H), Image.LANCZOS)), K, dist, None, K)
            canvas = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            c2w_cv = c2w.copy()
            c2w_cv[:3, 1] *= -1
            c2w_cv[:3, 2] *= -1
            w2c = np.linalg.inv(c2w_cv)
            vm = torch.from_numpy(w2c).float().to(dev).unsqueeze(0)
            with torch.no_grad():
                out_pv, _, _ = rasterization(means_t, quats_t, scales_t, opac_t, colors_t,
                                             vm, Kt, W, H, sh_degree=3, render_mode="ED")
            depth = out_pv[0, ..., 0].cpu().numpy()
            for inst in instances:
                oid = inst["id"]
                pts = xyz_k[label == oid]
                if len(pts) > 400:
                    pts = pts[rng_pv.choice(len(pts), 400, replace=False)]
                cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
                zc = cam[:, 2]
                ok = zc > 0.05
                u = (K[0, 0] * cam[ok, 0] / zc[ok] + K[0, 2]).astype(int)
                v = (K[1, 1] * cam[ok, 1] / zc[ok] + K[1, 2]).astype(int)
                zin = zc[ok]
                inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                u, v, zin = u[inb], v[inb], zin[inb]
                vis = np.abs(depth[v, u] - zin) < 0.12
                if vis.sum() < 12:
                    continue
                col = color_for(oid)
                for x, y in zip(u[vis], v[vis]):
                    cv2.circle(canvas, (x, y), 2, col, -1)
                cx_, cy_ = int(np.median(u[vis])), int(np.median(v[vis]))
                cv2.putText(canvas, str(oid), (cx_, cy_), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, str(oid), (cx_, cy_), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 1, cv2.LINE_AA)
            cv2.imwrite(str(pv_dir / f"{stem}.jpg"), canvas)
        print(f"instance-colored previews -> {pv_dir}")

    if args.thumbs > 0:
        segment_splat.render_thumbs(sd, keep, label, instances, out_dir, args.thumbs)
    print(f"done -> {out_dir}  (check render_check.jpg + preview/, name via names.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
