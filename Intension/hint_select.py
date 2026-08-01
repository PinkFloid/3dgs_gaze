#!/usr/bin/env python3
"""hint_select.py -- 交付狗端:object_hint 投影选框(同类多实例的最后一米消歧)。

接入(检测节点里三行):
    from hint_select import load_tags, pick_box
    TAGS, DICT = load_tags("tags_world.json")     # 随地图版本更新(与请求 frame 对版本)
    idx, uv, why = pick_box(frame_bgr, K, D, TAGS, object_hint, yolo_boxes, DICT)
      # idx: 选中的 YOLO 框下标;None = 拒抓(why: no_tag / hint_mismatch / pnp_failed)
      # yolo_boxes: [[x0,y0,x1,y1], ...] 像素 xyxy;object_hint: 请求里的 [x,y,z](板系)

原理:抓取相机同画面认 ArUco → 对 tags_world 的测绘角点联合 PnP(ITERATIVE;
tag 共面,SQPNP 断言崩)→ T_board_cam → 把板系 hint 投影到像素 → 落在哪个框抓
哪只;点不在任何框内且离最近框心 > 半框宽 → 拒抓(宁可不抓不抓错)。
**不需要机器人基座定位/外参链,也不需要意图机的模型或点云**:tags_world.json +
每单请求里的 object_hint 就是全部输入。K/D 用抓取相机自己的内参(无畸变传 None)。

角点顺序约定:corners_world 与 cv2.aruco.detectMarkers 返回角点同序(survey 按
检测序三角化)。现场自检两步:①hint 换成某 tag 角点均值,投影应落回该 tag 中心;
②hint 用在场物体的地图质心,应落进它的 YOLO 框。

无真机自检:python hint_select.py --selftest
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np


def load_tags(path):
    """tags_world.json -> ({id: 4x3 世界角点}, 字典名)。"""
    doc = json.load(open(path, encoding="utf-8"))
    tags = {int(k): np.asarray(v["corners_world"], np.float64)
            for k, v in doc["tags"].items()}
    return tags, doc.get("dictionary", "DICT_6X6_250")


def project_hint(matched, K, D, hint):
    """matched: [(4x3 世界角点, 4x2 像素角点), ...] -> hint 的像素投影 (u,v) 或 None。
    多 tag 联合 PnP:角点越多位姿越稳(与 pupil_localizer 同做法)。"""
    obj = np.concatenate([m[0] for m in matched]).astype(np.float64)
    img = np.concatenate([m[1] for m in matched]).astype(np.float64)
    dist = None if D is None else np.asarray(D, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, np.float64), dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    uv, _ = cv2.projectPoints(np.asarray([hint], np.float64), rvec, tvec,
                              np.asarray(K, np.float64), dist)
    return float(uv[0, 0, 0]), float(uv[0, 0, 1])


def select_box(uv, boxes):
    """落框判定。返回 (框下标|None, why)。规则:点在框内取框心最近者;
    不在任何框内但离最近框心 < 半框宽(容忍 2-3cm 投影误差)也取;再远拒抓。"""
    if uv is None:
        return None, "pnp_failed"
    u, v = uv
    inside, near = [], []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        d = float(np.hypot(u - cx, v - cy))
        if x0 <= u <= x1 and y0 <= v <= y1:
            inside.append((d, i))
        else:
            near.append((d, i, max(x1 - x0, y1 - y0)))
    if inside:
        return min(inside)[1], ""
    if near:
        d, i, side = min(near, key=lambda t: t[0])
        if d < 0.5 * side:
            return i, ""
    return None, "hint_mismatch"


def pick_box(frame_bgr, K, D, tags, hint, boxes, dict_name="DICT_6X6_250"):
    """整帧入口:检测 ArUco -> 联合 PnP -> 投影 -> 选框。"""
    ar = cv2.aruco
    det = ar.ArucoDetector(ar.getPredefinedDictionary(getattr(ar, dict_name)),
                           ar.DetectorParameters())
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    matched = []
    if ids is not None:
        for c, tid in zip(corners, ids.flatten()):
            if int(tid) in tags:
                matched.append((tags[int(tid)], c.reshape(4, 2)))
    if not matched:
        return None, None, "no_tag"
    uv = project_hint(matched, K, D, hint)
    idx, why = select_box(uv, boxes)
    return idx, uv, why


# ------------------------------------------------------------ 无真机自检

def _selftest():
    """合成一个已知位姿的相机看桌面 tag + 两只球:检验 PnP 投影闭环与选框规则。"""
    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
    s = 0.04  # 8cm tag 半边
    tc = np.array([0.20, -2.30, 0.75])                     # 桌面 tag 中心(板系)
    tag = tc + np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    b1, b2 = np.array([0.05, -2.30, 0.78]), np.array([0.35, -2.30, 0.78])

    C, L = np.array([0.20, -1.60, 1.20]), tc               # 相机位置/看向
    f = (L - C) / np.linalg.norm(L - C)
    x = np.cross([0.0, 0.0, 1.0], f)
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(f, x), f])                    # 右手系,z 朝前
    rvec, _ = cv2.Rodrigues(R)
    tvec = (-R @ C).reshape(3, 1)

    def proj(pts):
        uv, _ = cv2.projectPoints(np.asarray(pts, np.float64), rvec, tvec, K, None)
        return uv.reshape(-1, 2)

    det_tag = proj(tag)                                     # 合成"检测到的角点"
    p1, p2 = proj([b1])[0], proj([b2])[0]
    boxes = [[p[0] - 25, p[1] - 25, p[0] + 25, p[1] + 25] for p in (p1, p2)]

    # ① 零位:PnP 复原位姿后,hint=tag 中心的投影应与真值投影重合
    #(注意不能拿角点像素均值当参照:透视下四角均值≠中心投影)
    uv = project_hint([(tag, det_tag)], K, None, tc)
    assert np.hypot(*(np.array(uv) - proj([tc])[0])) < 0.5, uv
    # ② hint=球2 -> 选框 1
    uv = project_hint([(tag, det_tag)], K, None, b2)
    idx, why = select_box(uv, boxes)
    assert idx == 1 and why == "", (idx, why)
    # ③ hint 偏出(30cm 外的幽灵)-> 拒抓
    uv = project_hint([(tag, det_tag)], K, None, b2 + np.array([0.30, 0, 0]))
    idx, why = select_box(uv, boxes)
    assert idx is None and why == "hint_mismatch", (idx, why)
    print("hint_select selftest ok(零位/选框/拒抓 三关全过)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
