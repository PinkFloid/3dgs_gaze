#!/usr/bin/env python3
"""export_undistorted.py -- 把 Pupil 录像的世界相机导出为去畸变 mp4(可叠加视线点)。

    python Eye_Tracker/tools/export_undistorted.py ~/recordings/2026_07_30/004
    # -> ~/recordings/2026_07_30/004/world_undistorted.mp4(+ 中点预览 jpg)

去畸变配方与 verify_pose_render.py 相同:虚拟针孔 K_new = 鱼眼焦距 x --pinhole-scale
(默认 0.7,保住大部分 FOV)、主点居中,cv2.fisheye.initUndistortRectifyMap 整帧重映射;
视线点(norm_pos,y 翻转)用 cv2.fisheye.undistortPoints(P=K_new) 映到同一针孔平面。
视线是原始 gaze(未做 bias 修正)——看视频够用;精度活走 process_recording。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import msgpack
import numpy as np

SCENE = Path(__file__).resolve().parents[2] / "SceneRebuild"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", help="Pupil 录像目录(含 world.mp4 / gaze.pldata)")
    p.add_argument("--calib", default=str(SCENE / "Calibration_result/world_camera_calibration.npz"))
    p.add_argument("--out", default=None, help="输出 mp4(默认 <录像>/world_undistorted.mp4)")
    p.add_argument("--pinhole-scale", type=float, default=0.7,
                   help="虚拟针孔焦距 = 鱼眼焦距 x 此值,小=视野大边缘拉伸多")
    p.add_argument("--no-gaze", action="store_true", help="不叠加视线点")
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--max-frames", type=int, default=0, help=">0 只导前 N 帧(试跑)")
    return p.parse_args()


def load_gaze(rec: Path):
    """同 gaze_video.load_gaze:pldata -> (ts, norm_xy, conf),按时间排序。"""
    ts, xy, conf = [], [], []
    with open(rec / "gaze.pldata", "rb") as f:
        for _topic, payload in msgpack.Unpacker(f, use_list=False, strict_map_key=False):
            r = msgpack.unpackb(payload, strict_map_key=False)
            ts.append(r["timestamp"])
            xy.append(r["norm_pos"])
            conf.append(r.get("confidence", 0.0))
    order = np.argsort(ts)
    return np.array(ts)[order], np.array(xy)[order], np.array(conf)[order]


def main() -> int:
    args = parse_args()
    rec = Path(args.recording).expanduser()
    out_path = Path(args.out) if args.out else rec / "world_undistorted.mp4"

    z = np.load(args.calib, allow_pickle=True)
    K = np.asarray(z["camera_matrix"], np.float64)
    D = np.asarray(z["dist_coeffs"], np.float64).reshape(-1, 1)[:4]

    cap = cv2.VideoCapture(str(rec / "world.mp4"))
    if not cap.isOpened():
        print(f"打不开 {rec/'world.mp4'}")
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != (1920, 1080):  # 标定分辨率 1920x1080,其他分辨率按比例缩放内参
        K = K.copy()
        K[0] *= W / 1920.0
        K[1] *= H / 1080.0
        print(f"[!] 分辨率 {W}x{H} ≠ 标定 1920x1080,内参已按比例缩放")

    world_ts = np.load(rec / "world_timestamps.npy")
    n = len(world_ts)
    fps = (n - 1) / max(world_ts[-1] - world_ts[0], 1e-6)
    print(f"{n} 帧 @ {fps:.1f}fps,{world_ts[-1]-world_ts[0]:.1f}s")

    K_new = np.array([[K[0, 0] * args.pinhole_scale, 0, W / 2.0],
                      [0, K[1, 1] * args.pinhole_scale, H / 2.0],
                      [0, 0, 1.0]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2)

    g_ts = g_px = g_conf = None
    if not args.no_gaze and (rec / "gaze.pldata").exists():
        g_ts, g_xy, g_conf = load_gaze(rec)
        # norm_pos -> 鱼眼像素(y 翻转) -> 针孔像素(逐点去畸变,一次算完)
        px = np.stack([g_xy[:, 0] * W, (1.0 - g_xy[:, 1]) * H], axis=1)
        g_px = cv2.fisheye.undistortPoints(
            px.reshape(-1, 1, 2).astype(np.float64), K, D, R=np.eye(3), P=K_new
        ).reshape(-1, 2)
        print(f"{len(g_ts)} 个视线样本(conf≥{args.min_confidence} 才画)")

    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                    quality=7, macro_block_size=1)
        use_imageio = True
    except Exception:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        use_imageio = False
        print("[!] imageio/ffmpeg 不可用,回退 mp4v(兼容性差些)")

    preview = rec / "world_undistorted_preview.jpg"
    total = min(n, args.max_frames) if args.max_frames > 0 else n
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            total = i
            break
        und = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        if g_ts is not None and i < n:
            j = int(np.searchsorted(g_ts, world_ts[i]))
            j = min(max(j, 0), len(g_ts) - 1)
            if j > 0 and abs(g_ts[j - 1] - world_ts[i]) < abs(g_ts[j] - world_ts[i]):
                j -= 1
            if abs(g_ts[j] - world_ts[i]) < 0.08 and g_conf[j] >= args.min_confidence:
                x, y = int(round(g_px[j, 0])), int(round(g_px[j, 1]))
                if 0 <= x < W and 0 <= y < H:
                    cv2.circle(und, (x, y), 18, (0, 255, 0), 3)
                    cv2.line(und, (x - 28, y), (x + 28, y), (0, 255, 0), 2)
                    cv2.line(und, (x, y - 28), (x, y + 28), (0, 255, 0), 2)
        if i == total // 2:
            cv2.imwrite(str(preview), und)
        if use_imageio:
            writer.append_data(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
        else:
            writer.write(und)
        if i % 300 == 0:
            print(f"  {i}/{total}", flush=True)
    if use_imageio:
        writer.close()
    else:
        writer.release()
    cap.release()
    print(f"完成:{out_path}({total} 帧)\n预览:{preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
