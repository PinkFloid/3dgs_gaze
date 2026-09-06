"""Multi-view rigid tag-pose fit (known square, free scale) against fixed camera poses.
Well-conditioned where per-corner triangulation slips along grazing rays.
usage: python tag_pnp_fit.py <dataset> <tag-ids> <tag-sizes> [--write --only ids --min-views N]"""
import sys, json, pickle, numpy as np, cv2, shutil
from pathlib import Path
from scipy.optimize import least_squares
sys.path.insert(0, r"E:\3dgs_gaze\SceneRebuild\tools"); import survey_aruco_tags as S
ds = Path(sys.argv[1]); wanted = S.parse_ids(sys.argv[2]); size_of = S.make_size_of(sys.argv[3], 0.1)
write = "--write" in sys.argv
only = S.parse_ids(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
min_views = int(sys.argv[sys.argv.index("--min-views") + 1]) if "--min-views" in sys.argv else 3
meta, K, dist = S.load_transforms(ds, "transforms_aligned.json"); frames = meta["frames"]
w2c = [S.c2w_gl_to_w2c_cv(np.array(f["transform_matrix"], dtype=np.float64)) for f in frames]
cache = Path(f"dets_{ds.name}.pkl")
if cache.exists():
    dets = pickle.load(open(cache, "rb"))
else:
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, prm); dets = {}
    for fi, fr in enumerate(frames):
        g = cv2.imread(str(ds / fr["file_path"]), cv2.IMREAD_GRAYSCALE); c, ids, _ = det.detectMarkers(g)
        if ids is None: continue
        for quad, mid in zip(c, ids.flatten()):
            if int(mid) in wanted: dets.setdefault(int(mid), []).append((fi, quad.reshape(4, 2).astype(np.float64)))
    pickle.dump(dets, open(cache, "wb"))
def project(Xw, fi):
    rvec, _ = cv2.Rodrigues(w2c[fi][:3, :3]); p, _ = cv2.projectPoints(Xw, rvec, w2c[fi][:3, 3], K, dist); return p.reshape(-1, 2)
def floor_hit(fi, xy):
    n = cv2.undistortPoints(xy.reshape(1, 1, 2), K, dist).reshape(2); R, t = w2c[fi][:3, :3], w2c[fi][:3, 3]
    C = -R.T @ t; d = R.T @ np.array([n[0], n[1], 1.0]); return C + (-C[2] / d[2]) * d
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
def corners_from(p):
    R, _ = cv2.Rodrigues(p[:3]); return (R @ S.tag_object_corners(np.exp(p[6])).T).T + p[3:6]
def fit(obs, size0):
    P = np.array([[floor_hit(fi, q[k]) for k in range(4)] for fi, q in obs])  # (n,4,3)
    ctr = np.median(P.mean(1), 0); v = P[:, 1] - P[:, 0]; yaw = np.arctan2(np.median(v[:, 1]), np.median(v[:, 0]))
    def resid(p):
        Xw = corners_from(p); return np.concatenate([(project(Xw, fi) - q).ravel() for fi, q in obs])
    best = None
    for dyaw in (0, np.pi / 2, np.pi, -np.pi / 2):
        r0, _ = cv2.Rodrigues(Rz(yaw + dyaw)); p0 = np.concatenate([r0.ravel(), [ctr[0], ctr[1], 0.0], [np.log(size0)]])
        sol = least_squares(resid, p0, loss="soft_l1", f_scale=3.0, max_nfev=400)
        if best is None or sol.cost < best.cost: best = sol
    p = best.x; Xw = corners_from(p)
    per_view = [float(np.linalg.norm(project(Xw, fi) - q, axis=1).mean()) for fi, q in obs]
    R, _ = cv2.Rodrigues(p[:3]); tilt = np.degrees(np.arccos(abs(R[2, 2])))
    return p, Xw, per_view, tilt
tw_path = ds / "tags_world.json"; tw = json.loads(tw_path.read_text(encoding="utf-8"))
print(f"{'tag':>4} {'views':>5} {'cur_px':>7} {'fit_px':>7} {'size_fit_mm':>11} {'exp_mm':>6} {'shift_mm':>8} {'z_mm':>5} {'tilt':>5}  note")
new = {}
for mid in sorted(dets):
    if only and mid not in only: continue
    obs = dets[mid]
    if len(obs) < min_views: print(f"{mid:>4} {len(obs):>5}  skipped (<{min_views} views)"); continue
    exp = size_of(mid); p, Xw, pv, tilt = fit(obs, exp)
    cur = tw["tags"].get(str(mid)); 
    if cur:
        Xc = np.array(cur["corners_world"]); cur_px = float(np.mean([np.linalg.norm(project(Xc, fi) - q, axis=1).mean() for fi, q in obs]))
        shift = float(np.linalg.norm(Xw.mean(0) - Xc.mean(0)) * 1000)
    else: cur_px, shift = float("nan"), float("nan")
    size_fit = float(np.exp(p[6])); z = float(Xw.mean(0)[2] * 1000)
    note = []
    if abs(size_fit / exp - 1) > 0.04: note.append("SIZE?")
    if max(pv) > 8: note.append("worst view %.1fpx" % max(pv))
    if tilt > 5: note.append("TILT")
    print(f"{mid:>4} {len(obs):>5} {cur_px:7.2f} {np.mean(pv):7.2f} {size_fit*1000:11.1f} {exp*1000:6.0f} {shift:8.1f} {z:5.0f} {tilt:5.1f}  {' '.join(note)}")
    R, _ = cv2.Rodrigues(p[:3]); T = np.eye(4); T[:3, :3], T[:3, 3] = R, p[3:6]
    obj = S.tag_object_corners(exp); Rk, tk = S.kabsch(obj, Xw); resid_k = np.linalg.norm((Rk @ obj.T).T + tk - Xw, axis=1)
    new[str(mid)] = {"T_world_tag": T.tolist(), "corners_world": Xw.tolist(),
                     "side_lengths_m": [float(np.linalg.norm(Xw[i] - Xw[(i + 1) % 4])) for i in range(4)],
                     "expected_size_m": exp, "fit_rms_m": float(np.sqrt((resid_k ** 2).mean())), "n_views": len(obs),
                     "method": "multiview_pnp_fit", "reproj_px_mean": float(np.mean(pv)), "size_fit_m": size_fit}
if write:
    bak = ds / "tags_world.before_pnp.json"
    if not bak.exists(): shutil.copy(tw_path, bak)
    for k, v in new.items(): tw["tags"][k] = v
    tw["note"] = "tags with method=multiview_pnp_fit: rigid known-square fit over all views (tag_pnp_fit.py); others from survey RANSAC"
    tw_path.write_text(json.dumps(tw, indent=2), encoding="utf-8"); print(f"wrote {tw_path}: {len(new)} tags (backup {bak.name})")
