#!/usr/bin/env python3
"""check_cart_pose.py -- 车(物品台)有没有被挪过:非车 tag 定位相机,反解车上
tag(126/124)的实测位姿,与 tags_world 存档对比。

    python SceneRebuild/tools/check_cart_pose.py ~/recordings/2026_08_16/s1/world.mp4
    python SceneRebuild/tools/check_cart_pose.py photo.jpg --cam none   # 手机照片(仅粗查)

挪车复位就边挪边跑:Δ 中心 <2cm 且 Δyaw <2° 即视为对齐。
相机默认用眼镜世界相机标定(鱼眼);抽多帧取中位,单帧检测噪声不背锅。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SCENE = Path(__file__).resolve().parents[1]
CART_IDS = (126, 124)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", help="world.mp4 或单张图片")
    p.add_argument("--tags", default=str(SCENE / "world_size/tags_world.json"))
    p.add_argument("--cam", default=str(SCENE / "Calibration_result/world_camera_calibration.npz"),
                   help="鱼眼标定 npz;'none' = 无标定(手机照片粗查,只报像素残差)")
    p.add_argument("--frames", type=int, default=9, help="均匀抽帧数(图片输入忽略)")
    return p.parse_args()


def detect(gray, dic):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dic))
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel())}


def main():
    a = parse_args()
    J = json.loads(Path(a.tags).read_text())
    tt = J.get("tags", J)
    dic = J.get("dictionary", "DICT_6X6_250")
    world_corners = {int(k): np.array(v["corners_world"], float).reshape(4, 3)
                     for k, v in tt.items() if isinstance(v, dict) and "corners_world" in v}
    ref = {k: np.array(tt[str(k)]["T_world_tag"], float) for k in CART_IDS if str(k) in tt}

    K = D = None
    if a.cam != "none":
        z = np.load(a.cam, allow_pickle=True)
        K = np.asarray(z["camera_matrix"], np.float64)
        D = np.asarray(z["dist_coeffs"], np.float64).reshape(-1, 1)[:4]

    src = Path(a.video)
    frames = []
    if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
        frames = [cv2.imread(str(src))]
    else:
        cap = cv2.VideoCapture(str(src))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(n * 0.1, n * 0.9, a.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()

    deltas = {k: [] for k in CART_IDS}
    used = 0
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        det = detect(gray, dic)
        anchors = [i for i in det if i in world_corners and i not in CART_IDS]
        carts = [i for i in det if i in ref]
        if len(anchors) < 2 or not carts or K is None:
            continue
        obj = np.concatenate([world_corners[i] for i in anchors]).astype(np.float64)
        img = np.concatenate([det[i] for i in anchors]).astype(np.float64).reshape(-1, 1, 2)
        und = cv2.fisheye.undistortPoints(img, K, D).reshape(-1, 2)  # 归一化平面
        ok, rvec, tvec = cv2.solvePnP(obj, und, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        Rcw, _ = cv2.Rodrigues(rvec)   # world -> cam
        used += 1
        for cid in carts:
            e = tt[str(cid)]
            size = float(e.get("expected_size_m") or J.get("tag_size_m", 0.1))
            half = size / 2.0
            local = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], np.float64)
            uc = cv2.fisheye.undistortPoints(
                det[cid].astype(np.float64).reshape(-1, 1, 2), K, D).reshape(-1, 2)
            ok2, rv2, tv2 = cv2.solvePnP(local, uc, np.eye(3), None,
                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok2:
                continue
            Rct, _ = cv2.Rodrigues(rv2)
            Rwt = Rcw.T @ Rct
            cw = Rcw.T @ (tv2.ravel() - tvec.ravel())
            T = ref[cid]
            dxy = cw[:2] - T[:2, 3]
            yaw_meas = np.arctan2(Rwt[1, 0], Rwt[0, 0])
            yaw_ref = np.arctan2(T[1, 0], T[0, 0])
            dyaw = np.degrees((yaw_meas - yaw_ref + np.pi) % (2 * np.pi) - np.pi)
            deltas[cid].append([dxy[0], dxy[1], cw[2] - T[2, 3], dyaw])

    print(f"用帧 {used}/{len(frames)}")
    moved = False
    for cid, rows in deltas.items():
        if not rows:
            print(f"tag{cid}: 未检出(距离远/太斜正常,换近距帧)")
            continue
        m = np.median(np.array(rows), axis=0)
        norm = float(np.hypot(m[0], m[1]))
        flag = "  <-- 挪过" if (norm > 0.03 or abs(m[3]) > 3) else "(在位)"
        moved |= norm > 0.03 or abs(m[3]) > 3
        print(f"tag{cid}: Δ中心 ({m[0]:+.3f},{m[1]:+.3f})m |Δ|={norm*100:.1f}cm "
              f"Δz {m[2]:+.3f}m Δyaw {m[3]:+.1f}°  n={len(rows)}{flag}")
    if moved:
        print("复位:朝 Δ 的反方向挪车,重跑本工具直到 |Δ|<2cm 且 |Δyaw|<2°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
