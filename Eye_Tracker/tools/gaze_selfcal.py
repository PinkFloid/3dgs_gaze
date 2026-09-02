"""Object-anchored gaze-bias self-calibration (v2 self-cal, 2026-09-02).

Every FINAL cone verdict is a candidate observation of the calibration bias:
the residual between the (bias-corrected) fixation ray and the winning
object's centroid direction, in the camera's undistorted-normalised plane,
i.e. the same units gaze_live subtracts as `bias`. The only way such a loop
can run away is by treating a wrong verdict as truth, so an observation is
accepted only when the verdict is unambiguous by construction:

  capture >= min_capture         the winner really is under the cone
  2nd candidate capture < max_2  nearest other target >= ~2.4 sigma off-axis
  rho_k <= max_rho_sigma*sigma   small object: its centroid is what you look at
  duration >= min_dur, head travel <= max_travel

Aggregation is a robust median over a short window that must contain >= 3
distinct objects with consistent implied biases (MAD < 1.5 sigma; pure noise
gives ~1.2 sigma); the bias
moves a fraction alpha of the way per update, step- and norm-clamped, and
never decays toward zero (the last observation is the prior, not zero).
Bias-corrected residuals are stored as *implied true bias* (= bias used + r)
so older observations stay valid after updates. See docs/CONE_POSTERIOR_V2.md.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class ObjectSelfCal:
    def __init__(self, min_capture=0.3, max_second=0.05, max_rho_sigma=1.0, min_dur=0.5,
                 max_travel=0.15, window=8, min_obs=3, min_objects=3, max_mad_sigma=1.5,
                 alpha=0.3, step_deg=0.5, clamp_deg=3.0):
        self.min_capture, self.max_second = min_capture, max_second
        self.max_rho_sigma, self.min_dur, self.max_travel = max_rho_sigma, min_dur, max_travel
        self.min_obs, self.min_objects = min_obs, min_objects
        self.max_mad_sigma = max_mad_sigma   # window MAD limit in sigmas: noise alone gives ~1.2 sigma
        self.alpha = alpha
        self.step = math.tan(math.radians(step_deg))
        self.clamp = math.tan(math.radians(clamp_deg))
        self.obs = deque(maxlen=window)          # (t, name, implied bias (2,) normalised)
        self.bias = np.zeros(2)                  # normalised units, gaze minus target
        self.n_obs = self.n_updates = 0
        self.skips: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _norm(R, o, p):
        v = R.T @ (np.asarray(p, float) - o)
        return v[:2] / v[2] if v[2] > 1e-6 else None

    @staticmethod
    def deg(v):
        return np.degrees(np.arctan(np.asarray(v, float))).round(2).tolist()

    def gate(self, fx, sigma_rad, radius_m):
        if fx.get("provisional"):
            return "provisional"
        if fx.get("object") is None or not fx.get("object_centroid_world"):
            return "none"
        cap = fx.get("capture")
        if cap is None or cap < self.min_capture:
            return "capture"
        cands = fx.get("candidates") or []
        if len(cands) > 1 and (cands[1].get("capture") or 0.0) >= self.max_second:
            return "ambiguous"
        dist = float(fx.get("distance_m") or 0.0)
        if radius_m is None or dist <= 0.05:
            return "noradius"
        if math.atan2(radius_m, dist) > self.max_rho_sigma * sigma_rad:
            return "big"
        if float(fx.get("duration_s", 0.0)) < self.min_dur:
            return "short"
        if float(fx.get("_travel", 0.0)) > self.max_travel:
            return "moving"
        return None

    # -- main entry --------------------------------------------------------
    def observe(self, fx, T, sigma_rad, radius_m, bias_used):
        """One final verdict in. Returns a dict for logging: {"skip": why} or
        {"resid_deg": [...], "bias_deg": [...] (when the bias moved)}."""
        why = self.gate(fx, sigma_rad, radius_m) or (None if T is not None else "noT")
        if why:
            self.skips[why] = self.skips.get(why, 0) + 1
            return {"skip": why}
        T = np.asarray(T, float)
        R, o = T[:3, :3], T[:3, 3]
        g = self._norm(R, o, fx["centroid_world"])
        tn = self._norm(R, o, fx["object_centroid_world"])
        if g is None or tn is None:
            self.skips["behind"] = self.skips.get("behind", 0) + 1
            return {"skip": "behind"}
        r = g - tn                                   # corrected gaze minus target
        implied = np.asarray(bias_used, float) + r   # = estimate of the true bias
        self.obs.append((float(fx.get("t_end", 0.0)), fx["object"], implied))
        self.n_obs += 1
        out = {"resid_deg": self.deg(r), "object": fx["object"]}
        if len(self.obs) < self.min_obs or len({n for _, n, _ in self.obs}) < self.min_objects:
            out["skip"] = "warmup"
            return out
        M = np.array([b for _, _, b in self.obs])
        med = np.median(M, axis=0)
        mad = float(np.median(np.linalg.norm(M - med, axis=1)))
        if mad > self.max_mad_sigma * math.tan(sigma_rad):
            out["skip"] = "inconsistent"
            out["mad_deg"] = round(math.degrees(math.atan(mad)), 2)
            return out
        step = self.alpha * (med - self.bias)
        n = float(np.linalg.norm(step))
        if n > self.step:
            step *= self.step / n
        b = self.bias + step
        nb = float(np.linalg.norm(b))
        if nb > self.clamp:
            b *= self.clamp / nb
        self.bias = b
        self.n_updates += 1
        out["bias_deg"] = self.deg(self.bias)
        out["target_deg"] = self.deg(med)
        return out


# ------------------------------------------------------------ self-test
def _selftest():
    rng = np.random.default_rng(0)
    sigma = math.radians(1.5)
    true_b = np.array([math.tan(math.radians(0.5)), math.tan(math.radians(-1.7))])
    # camera at origin looking +x: cam x = world -y (viewer right), cam y = world -z (down), cam z = world +x
    R = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])   # columns = cam axes in world
    T = np.eye(4); T[:3, :3] = R
    objs = {f"o{i}": np.array([3.0, y, 0.75]) for i, y in enumerate((-0.6, -0.2, 0.2, 0.6))}
    sc = ObjectSelfCal()
    hist = []
    for k in range(80):
        name = list(objs)[k % 4]
        tn = ObjectSelfCal._norm(R, np.zeros(3), objs[name])
        raw = tn + true_b + rng.normal(0, math.tan(sigma), 2)
        corr = raw - sc.bias
        miss = np.degrees(np.arctan(np.linalg.norm(corr - tn)))
        cap = math.exp(-(math.radians(miss) ** 2) / (2 * sigma ** 2))
        pw = R @ np.array([corr[0], corr[1], 1.0]) * 3.0
        fx = {"object": name, "object_centroid_world": objs[name].tolist(), "centroid_world": pw.tolist(),
              "capture": cap, "candidates": [{"capture": cap}, {"capture": 0.0}], "distance_m": 3.0,
              "duration_s": 1.0, "t_end": float(k)}
        out = sc.observe(fx, T, sigma, 0.0335, sc.bias)
        hist.append((k, out.get("skip"), sc.deg(sc.bias)))
    print("true bias deg:", ObjectSelfCal.deg(true_b), " final estimate:", sc.deg(sc.bias),
          " updates:", sc.n_updates, " skips:", sc.skips)
    for k, s, b in hist[:12]:
        print(f"  obs {k:2d} {s or 'update':12s} bias {b}")
    err = np.degrees(np.arctan(np.linalg.norm(sc.bias - true_b)))
    assert err < 0.6, err
    print(f"selftest ok: |error| {err:.2f} deg")


if __name__ == "__main__":
    _selftest()
