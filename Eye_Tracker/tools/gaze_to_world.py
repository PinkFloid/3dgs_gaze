#!/usr/bin/env python
"""Gaze -> 3D world coordinates: intersect gaze rays with the 3DGS scene.

Offline, over a Pupil Capture recording:
  1. Load camera poses from pupil_localizer.py's --log JSONL; interpolate
     short gaps (position lerp + rotation slerp, up to --max-gap seconds).
  2. Load Pupil's online fixation detections (fixations.pldata).
  3. For each fixation: gaze pixel -> fisheye-undistort -> ray in camera frame
     -> world ray; render a small depth patch with gsplat looking straight
     down the ray; depth at the patch center gives the 3D fixation point.
  4. Write fixations_world.json, print a human-readable table, and (optionally)
     save annotated video frames + GS renders for visual verification.

Run inside the nerfstudio env. First-ever gsplat import on a fresh cache needs
the CUDA build env vars (see PIPELINE.md); afterwards it loads from cache.

Example:
  python tools/gaze_to_world.py \
    --recording ~/recordings/2026_07_05/000 \
    --poses out.jsonl \
    --splat lab_result/splatfacto/<run>/splat.ply-adjacent ckpt \
    --annotate-dir fixation_previews
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2] / "SceneRebuild"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", required=True, help="Pupil Capture recording dir.")
    p.add_argument("--poses", required=True, help="JSONL from pupil_localizer.py --log.")
    p.add_argument("--ckpt", default=None,
                   help="splatfacto checkpoint (default: newest step-*.ckpt under lab_result/).")
    p.add_argument("--calib", default=str(root / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--max-gap", type=float, default=1.0,
                   help="Max pose gap (s) to interpolate across; fixations in longer gaps are skipped.")
    p.add_argument("--min-confidence", type=float, default=0.6, help="Min fixation confidence.")
    p.add_argument("--min-duration", type=float, default=150.0, help="Min fixation duration (ms).")
    p.add_argument("--out", default=None, help="Output json (default: <recording>/fixations_world.json).")
    p.add_argument("--annotate-dir", default=None, help="Save annotated frame + GS render per fixation.")
    p.add_argument("--continuous", action="store_true",
                   help="Map ALL high-confidence gaze samples to 3D and detect fixations by "
                        "WORLD-space clustering instead of Pupil's image-space detector. "
                        "Catches walking-while-staring (VOR) fixations the image-space detector misses.")
    p.add_argument("--sample-hz", type=float, default=30.0, help="Continuous mode: gaze subsample rate.")
    p.add_argument("--cluster-radius", type=float, default=0.15,
                   help="Continuous mode: max distance (m) from cluster centroid.")
    p.add_argument("--min-fix-dur", type=float, default=0.25, help="Continuous mode: min fixation duration (s).")
    p.add_argument("--bias-deg", default=None,
                   help="Gaze bias correction 'dx,dy' in deg (gaze minus target). Default: read "
                        "<recording>/gaze_precision.json (written by gaze_precision.py) if present.")
    p.add_argument("--no-bias", action="store_true", help="Disable bias correction.")
    return p.parse_args()


# ------------------------------------------------------------ recording I/O

def load_pldata(path: Path):
    import msgpack
    out = []
    with open(path, "rb") as f:
        for _topic, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            out.append(msgpack.unpackb(payload, strict_map_key=False))
    return out


def load_fixations(rec: Path, min_conf: float, min_dur_ms: float):
    """Online detector emits growing updates per fixation id -- keep the last."""
    by_id = {}
    for r in load_pldata(rec / "fixations.pldata"):
        by_id[r["id"]] = r
    fx = [r for r in by_id.values()
          if r.get("confidence", 0) >= min_conf and r.get("duration", 0) >= min_dur_ms]
    return sorted(fx, key=lambda r: r["timestamp"])


# ------------------------------------------------------------ poses

class PoseTrack:
    def __init__(self, jsonl: Path, max_gap: float):
        from scipy.spatial.transform import Rotation, Slerp
        recs = [json.loads(l) for l in open(jsonl)]
        recs.sort(key=lambda r: r["timestamp"])
        self.ts = np.array([r["timestamp"] for r in recs])
        T = np.array([r["T_world_cam"] for r in recs])
        self.pos = T[:, :3, 3]
        self.rot = Rotation.from_matrix(T[:, :3, :3])
        self.max_gap = max_gap
        self._Slerp = Slerp

    def query(self, t: float):
        """Return T_world_cam at t, or None if inside a gap longer than max_gap."""
        i = np.searchsorted(self.ts, t)
        if i == 0 or i == len(self.ts):
            edge = 0 if i == 0 else -1
            if abs(t - self.ts[edge]) > 0.05:  # only snap to the very edge
                return None
            i = max(1, min(i, len(self.ts) - 1))
        t0, t1 = self.ts[i - 1], self.ts[i]
        if t1 - t0 > self.max_gap:
            return None
        a = 0.0 if t1 == t0 else min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        T = np.eye(4)
        T[:3, 3] = (1 - a) * self.pos[i - 1] + a * self.pos[i]
        slerp = self._Slerp([0, 1], self.rot[[i - 1, i]])
        T[:3, :3] = slerp([a])[0].as_matrix()
        return T


# ------------------------------------------------------------ splat renderer

class SplatDepth:
    def __init__(self, ckpt_path: Path):
        import torch
        from gsplat import rasterization
        self.torch = torch
        self.rasterization = rasterization
        self.dev = torch.device("cuda")
        sd = torch.load(ckpt_path, map_location=self.dev)["pipeline"]
        g = lambda n: sd[f"_model.gauss_params.{n}"]
        self.means = g("means")
        self.quats = torch.nn.functional.normalize(g("quats"), dim=-1)
        self.scales = torch.exp(g("scales"))
        self.opac = torch.sigmoid(g("opacities")).squeeze(-1)
        self.colors = torch.cat([g("features_dc").unsqueeze(1), g("features_rest")], dim=1)
        print(f"splat: {len(self.means)} gaussians from {ckpt_path.name}")

    def _render(self, w2c: np.ndarray, K: np.ndarray, w: int, h: int):
        vm = self.torch.tensor(w2c, dtype=self.torch.float32, device=self.dev).unsqueeze(0)
        Kt = self.torch.tensor(K, dtype=self.torch.float32, device=self.dev).unsqueeze(0)
        out, alpha, _ = self.rasterization(
            self.means, self.quats, self.scales, self.opac, self.colors,
            vm, Kt, w, h, sh_degree=3, render_mode="RGB+ED", rasterize_mode="classic")
        return out[0].cpu().numpy(), alpha[0].cpu().numpy()

    def depth_along_ray(self, origin: np.ndarray, direction: np.ndarray):
        """Median depth of a 3x3 center patch of a 33x33 render looking down the ray."""
        z = direction / np.linalg.norm(direction)
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, up); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        w2c = np.eye(4)
        w2c[:3, :3] = np.stack([x, y, z])
        w2c[:3, 3] = -w2c[:3, :3] @ origin
        S = 33
        K = np.array([[256.0, 0, S / 2], [0, 256.0, S / 2], [0, 0, 1]])
        out, alpha = self._render(w2c, K, S, S)
        c = S // 2
        patch_d = out[c - 1:c + 2, c - 1:c + 2, 3]
        patch_a = alpha[c - 1:c + 2, c - 1:c + 2, 0]
        if np.median(patch_a) < 0.5:
            return None
        return float(np.median(patch_d))

    def patch_along_ray(self, origin: np.ndarray, direction: np.ndarray,
                        half_angle: float, S: int = 33):
        """Full depth/alpha patch for a cone of half-angle (rad) around the ray.

        Returns (depth SxS z-depth, alpha SxS, dirs SxSx3 unit world ray dirs,
        tmul SxS) with ray length t = depth * tmul (depth is z-depth, not range).
        gaze_object --cone integrates the object posterior over this patch.
        """
        z = direction / np.linalg.norm(direction)
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        x = np.cross(z, up); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        w2c = np.eye(4)
        w2c[:3, :3] = np.stack([x, y, z])
        w2c[:3, 3] = -w2c[:3, :3] @ origin
        f = (S / 2) / np.tan(half_angle)
        K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]])
        out, alpha = self._render(w2c, K, S, S)
        n = (np.arange(S) + 0.5 - S / 2) / f
        nx, ny = np.meshgrid(n, n)                    # [row=v, col=u]
        dirs_cam = np.stack([nx, ny, np.ones_like(nx)], axis=-1)
        tmul = np.linalg.norm(dirs_cam, axis=-1)      # z-depth -> ray length
        dirs = (dirs_cam / tmul[..., None]) @ np.stack([x, y, z])
        return out[..., 3], alpha[..., 0], dirs, tmul

    def render_view(self, T_world_cam: np.ndarray, K: np.ndarray, w: int, h: int):
        rgb, _ = self._render(np.linalg.inv(T_world_cam), K, w, h)
        return (np.clip(rgb[..., :3], 0, 1) * 255).astype(np.uint8)


def resolve_bias(args, rec: Path):
    """Bias(t) as (timestamps, biases) in undistorted-normalized units, gaze-minus-target.

    gaze_precision.json stamps (head/tail tag stares) are lerped over time:
    rec002 measured 2.6 deg of slow drift head->tail -- a constant correction
    leaves late fixations off by more than a 13cm target at 4m.
    """
    if args.no_bias:
        return np.array([0.0]), np.zeros((1, 2))
    if args.bias_deg:
        b = np.array([float(v) for v in args.bias_deg.split(",")])
        print(f"bias correction (cli, constant): ({b[0]:+.2f},{b[1]:+.2f}) deg")
        return np.array([0.0]), np.tan(np.radians(b))[None]
    pj = rec / "gaze_precision.json"
    if not pj.exists():
        return np.array([0.0]), np.zeros((1, 2))
    d = json.loads(pj.read_text(encoding="utf-8"))
    stamps = d.get("stamps") or [{"t": 0.0, "bias_deg": d["bias_deg"]}]
    ts = np.array([s["t"] for s in stamps], float)
    bs = np.tan(np.radians(np.array([s["bias_deg"] for s in stamps], float)))
    desc = ", ".join(f"({s['bias_deg'][0]:+.2f},{s['bias_deg'][1]:+.2f})deg" for s in stamps)
    print(f"bias correction ({pj.name}): {len(stamps)} stamp(s) {desc}"
          + (", lerped over time" if len(stamps) > 1 else ""))
    return ts, bs


def bias_at(bias, t: float) -> np.ndarray:
    ts, bs = bias
    if len(ts) == 1:
        return bs[0]
    return np.array([np.interp(t, ts, bs[:, 0]), np.interp(t, ts, bs[:, 1])])


# ------------------------------------------------------------ continuous mode

def load_gaze(rec: Path, min_conf: float):
    gz = [r for r in load_pldata(rec / "gaze.pldata") if r.get("confidence", 0) >= min_conf]
    return sorted(gz, key=lambda r: r["timestamp"])


def cluster_world_fixations(times, points, radius: float, min_dur: float):
    """Greedy sequential clustering of 3D gaze points -> world-space fixations."""
    clusters = []
    cur_idx = []
    centroid = None
    for i, (t, p) in enumerate(zip(times, points)):
        if centroid is not None and np.linalg.norm(p - centroid) <= radius:
            cur_idx.append(i)
            pts = points[cur_idx]
            centroid = pts.mean(axis=0)
        else:
            if cur_idx:
                clusters.append(cur_idx)
            cur_idx = [i]
            centroid = p
    if cur_idx:
        clusters.append(cur_idx)
    out = []
    for idx in clusters:
        t0, t1 = times[idx[0]], times[idx[-1]]
        if t1 - t0 < min_dur or len(idx) < 4:
            continue
        pts = points[idx]
        out.append({
            "t_start": float(t0), "t_end": float(t1), "duration_s": float(t1 - t0),
            "centroid_world": pts.mean(axis=0).tolist(),
            "spread_m": float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).mean()),
            "n_samples": len(idx),
            "mid_index": idx[len(idx) // 2],
        })
    return out


def run_continuous(args, rec, poses, splat, K_img, D_fish, W, H, world_ts, cap, bias):
    gaze = load_gaze(rec, args.min_confidence)
    step = max(1, int(round(len(gaze) / ((gaze[-1]["timestamp"] - gaze[0]["timestamp"]) * args.sample_hz))))
    gaze = gaze[::step]
    print(f"continuous: {len(gaze)} gaze samples after subsampling to ~{args.sample_hz:.0f}Hz")

    times, points, px_uv = [], [], []
    n_nopose = n_nosurf = 0
    for g in gaze:
        T = poses.query(g["timestamp"])
        if T is None:
            n_nopose += 1
            continue
        u = g["norm_pos"][0] * W
        v = (1.0 - g["norm_pos"][1]) * H
        pn = cv2.fisheye.undistortPoints(np.array([[[u, v]]], np.float64), K_img, D_fish).reshape(2)
        pn = pn - bias_at(bias, g["timestamp"])
        ray = np.array([pn[0], pn[1], 1.0])
        ray /= np.linalg.norm(ray)
        depth = splat.depth_along_ray(T[:3, 3], T[:3, :3] @ ray)
        if depth is None:
            n_nosurf += 1
            continue
        times.append(g["timestamp"])
        points.append(T[:3, 3] + depth * (T[:3, :3] @ ray))
        px_uv.append((u, v))
    times, points = np.array(times), np.array(points)
    print(f"mapped {len(points)} points ({n_nopose} no-pose, {n_nosurf} no-surface)")

    fixes = cluster_world_fixations(times, points, args.cluster_radius, args.min_fix_dur)
    # camera origin at the mid sample: gaze_object --cone re-renders the gaze
    # cone from here, and distance turns cluster spread into angular spread
    for fx in fixes:
        T = poses.query(float(times[fx["mid_index"]]))
        if T is not None:
            o = T[:3, 3]
            d = float(np.linalg.norm(np.array(fx["centroid_world"]) - o))
            fx["origin_world"] = o.tolist()
            fx["distance_m"] = round(d, 3)
            fx["ang_spread_deg"] = round(float(np.degrees(np.arctan2(fx["spread_m"], d))), 2)
    t0 = world_ts[0]
    print(f"\n{len(fixes)} world-space fixations "
          f"(radius {args.cluster_radius}m, min {args.min_fix_dur}s):")
    print(f"{'#':>3} {'t(s)':>6} {'dur(s)':>6} {'n':>4}  {'centroid xyz (m)':<28} {'spread':>6}")
    annotate_dir = Path(args.annotate_dir) if args.annotate_dir else None
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)
    for k, fx in enumerate(fixes):
        c = fx["centroid_world"]
        print(f"{k:>3} {fx['t_start']-t0:>6.1f} {fx['duration_s']:>6.2f} {fx['n_samples']:>4}  "
              f"({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})          {fx['spread_m']*100:>5.1f}cm")
        if annotate_dir:
            mi = fx.pop("mid_index")
            fi = int(np.clip(np.searchsorted(world_ts, times[mi]), 0, len(world_ts) - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok:
                u, v = px_uv[mi]
                cv2.circle(frame, (int(u), int(v)), 25, (0, 0, 255), 4)
                cv2.putText(frame, f"wfix{k} t={fx['t_start']-t0:.1f}s ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})m",
                            (30, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.imwrite(str(annotate_dir / f"wfix_{k:03d}.jpg"), frame)
        else:
            fx.pop("mid_index", None)

    out = Path(args.out) if args.out else rec / "world_fixations.json"
    out.write_text(json.dumps({
        "recording": str(rec), "mode": "continuous world-space clustering",
        "params": {"sample_hz": args.sample_hz, "cluster_radius_m": args.cluster_radius,
                   "min_fix_dur_s": args.min_fix_dur, "min_confidence": args.min_confidence},
        "world_frame": "ChArUco board frame, meters",
        "fixations": fixes,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


# ------------------------------------------------------------ main

def main() -> int:
    args = parse_args()
    rec = Path(args.recording).expanduser()
    root = Path(__file__).resolve().parents[2] / "SceneRebuild"

    ckpt = Path(args.ckpt) if args.ckpt else max(
        (root / "lab_result").rglob("step-*.ckpt"), key=lambda p: p.stat().st_mtime)

    z = np.load(args.calib, allow_pickle=True)
    K_fish = np.asarray(z["camera_matrix"], np.float64)
    D_fish = np.asarray(z["dist_coeffs"], np.float64).reshape(-1, 1)[:4]

    poses = PoseTrack(Path(args.poses), args.max_gap)
    fixations = load_fixations(rec, args.min_confidence, args.min_duration)
    print(f"{len(fixations)} fixations (conf>={args.min_confidence}, dur>={args.min_duration}ms), "
          f"pose track {poses.ts[0]:.1f}..{poses.ts[-1]:.1f}")

    world_ts = np.load(rec / "world_timestamps.npy")
    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # calibration was 1920x1080; rescale if the recording differs
    K_img = K_fish.copy()
    K_img[0] *= W / 1920.0
    K_img[1] *= H / 1080.0

    splat = SplatDepth(ckpt)
    bias = resolve_bias(args, rec)

    if args.continuous:
        run_continuous(args, rec, poses, splat, K_img, D_fish, W, H, world_ts, cap, bias)
        return 0

    annotate_dir = Path(args.annotate_dir) if args.annotate_dir else None
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = world_ts[0]
    print(f"\n{'id':>4} {'t(s)':>6} {'dur(ms)':>8} {'conf':>5}  {'world xyz (m)':<26} {'dist':>5}  status")
    for fx in fixations:
        t = fx["timestamp"]
        row = {"id": fx["id"], "t": t, "t_rel": t - t0, "duration_ms": fx["duration"],
               "confidence": fx["confidence"], "norm_pos": list(fx["norm_pos"])}
        T = poses.query(t)
        if T is None:
            row["status"] = "no-pose"
        else:
            u = fx["norm_pos"][0] * W
            v = (1.0 - fx["norm_pos"][1]) * H   # pupil norm_pos: origin bottom-left
            pn = cv2.fisheye.undistortPoints(
                np.array([[[u, v]]], np.float64), K_img, D_fish).reshape(2)
            pn = pn - bias_at(bias, t)
            ray_cam = np.array([pn[0], pn[1], 1.0])
            ray_cam /= np.linalg.norm(ray_cam)
            origin = T[:3, 3]
            ray_w = T[:3, :3] @ ray_cam
            depth = splat.depth_along_ray(origin, ray_w)
            if depth is None:
                row["status"] = "no-surface"
            else:
                point = origin + depth * ray_w
                row.update(status="ok", point_world=point.tolist(),
                           distance_m=depth, T_world_cam=T.tolist())
        results.append(row)
        pt = row.get("point_world")
        pts = f"({pt[0]:+.2f},{pt[1]:+.2f},{pt[2]:+.2f})" if pt else "-"
        dist = f"{row['distance_m']:.2f}" if "distance_m" in row else "-"
        print(f"{fx['id']:>4} {t - t0:>6.1f} {fx['duration']:>8.0f} {fx['confidence']:>5.2f}  "
              f"{pts:<26} {dist:>5}  {row['status']}")

        if annotate_dir and row["status"] == "ok":
            fi = int(np.clip(np.searchsorted(world_ts, t), 0, len(world_ts) - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if ok:
                cv2.circle(frame, (int(u), int(v)), 25, (0, 0, 255), 4)
                label = f"id{fx['id']} ({pt[0]:+.2f},{pt[1]:+.2f},{pt[2]:+.2f})m d={depth:.2f}m"
                cv2.putText(frame, label, (30, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                gs = splat.render_view(T, K_img / 2, W // 2, H // 2)
                gs = cv2.resize(cv2.cvtColor(gs, cv2.COLOR_RGB2BGR), (W, H))
                cv2.imwrite(str(annotate_dir / f"fix_{fx['id']:03d}.jpg"),
                            np.concatenate([frame, gs], axis=1))

    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"\n{n_ok}/{len(results)} fixations mapped to world coordinates")
    out = Path(args.out) if args.out else rec / "fixations_world.json"
    out.write_text(json.dumps({
        "recording": str(rec), "poses": str(args.poses), "ckpt": str(ckpt),
        "world_frame": "ChArUco board frame, meters",
        "fixations": results,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
