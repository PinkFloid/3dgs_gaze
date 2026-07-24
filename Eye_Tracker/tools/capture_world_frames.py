#!/usr/bin/env python3
"""capture_world_frames.py -- 世界相机标定图自动采集(需 Pupil Capture 在跑 + Frame Publisher 插件)。

订阅 frame.world,检测 ChArUco 板;角点数达标、离上一张拉开距离且间隔够时自动存 PNG。
挥板要慢、盖满视野——鱼眼的畸变主战场在边角,边角机位多给。

    python capture_world_frames.py                     # 存满 80 张到 ../world_camera_calibration_imgs
    python capture_world_frames.py --target 60 --min-corners 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pupil_localizer import live_frames  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pupil", default="127.0.0.1:50020", help="Pupil Remote host:port")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "world_camera_calibration_imgs"))
    p.add_argument("--target", type=int, default=80, help="存满多少张自动退出")
    p.add_argument("--min-corners", type=int, default=40, help="入选帧的最少 ChArUco 角点")
    p.add_argument("--interval", type=float, default=0.7, help="两张之间最短间隔 (s)")
    p.add_argument("--min-move", type=float, default=60.0,
                   help="板中心相对上一张至少移动多少像素(逼出机位多样性)")
    p.add_argument("--squares-x", type=int, default=11)
    p.add_argument("--squares-y", type=int, default=8)
    p.add_argument("--marker-id-start", type=int, default=0, help="新 A3 板=0(旧板=30)")
    p.add_argument("--no-legacy", action="store_true")
    p.add_argument("--dictionary", default="DICT_6X6_250")
    return p.parse_args()


def wait_pupil(addr):
    """连不上就一直等(Capture 可以后开);返回已订阅 frame.world 的 SUB。"""
    import zmq
    ctx = zmq.Context.instance()
    host, port = addr.rsplit(":", 1)
    while True:
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.connect(f"tcp://{host}:{port}")
        req.send_string("SUB_PORT")
        if req.poll(2000):
            sub_port = req.recv_string()
            req.close()
            sub = ctx.socket(zmq.SUB)
            sub.connect(f"tcp://{host}:{sub_port}")
            sub.setsockopt_string(zmq.SUBSCRIBE, "frame.world")
            sub.setsockopt(zmq.RCVTIMEO, 2000)
            return sub
        req.close()
        print("  等待 Pupil Capture ...(先开 Capture,并确认 Frame Publisher 插件开启)", flush=True)
        time.sleep(2)


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n0 = len(list(out.glob("frame_*.png")))
    if n0:
        print(f"[!] {out} 已有 {n0} 张 frame_*.png,新帧接着编号(混批请先清空)")

    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    n_markers = (args.squares_x * args.squares_y) // 2
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), 0.032, 0.024, dic,
                                   np.arange(args.marker_id_start,
                                             args.marker_id_start + n_markers))
    board.setLegacyPattern(not args.no_legacy)
    det = cv2.aruco.CharucoDetector(board)

    print(f"连接 {args.pupil} ...")
    sub = wait_pupil(args.pupil)
    print(f"开始采集:目标 {args.target} 张,角点>={args.min_corners},"
          f"间隔>={args.interval}s,移动>={args.min_move:.0f}px。Ctrl-C 提前结束。")

    saved, last_t, last_c, t_msg = 0, 0.0, None, 0.0
    grid = set()  # 板中心走过的 4x4 画面格,提示覆盖率
    try:
        for _ts, bgr in live_frames(sub):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cc, ci, _mc, _mi = det.detectBoard(gray)
            n = 0 if ci is None else len(ci)
            now = time.time()
            if n < args.min_corners:
                if now - t_msg > 5:
                    print(f"  ...等板子(当前角点 {n})", flush=True)
                    t_msg = now
                continue
            c = cc.reshape(-1, 2).mean(0)
            if last_c is not None and float(np.hypot(*(c - last_c))) < args.min_move:
                continue
            if now - last_t < args.interval:
                continue
            cv2.imwrite(str(out / f"frame_{n0 + saved:03d}.png"), bgr)
            saved += 1
            last_t, last_c, t_msg = now, c, now
            h, w = gray.shape
            grid.add((min(3, int(c[0] / w * 4)), min(3, int(c[1] / h * 4))))
            print(f"[{saved}/{args.target}] 角点 {n}  画面覆盖 {len(grid)}/16 格", flush=True)
            if saved >= args.target:
                break
    except KeyboardInterrupt:
        pass
    print(f"完成:存了 {saved} 张到 {out}(覆盖 {len(grid)}/16 格;边角格没到就再补拍)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
