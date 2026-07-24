#!/usr/bin/env python3
"""capture_world_frames.py -- 世界相机标定图采集(需 Pupil Capture 在跑 + Frame Publisher 插件)。

默认手动挡:终端里**按回车拍一张**,q 回车收工。状态行实时报当前 ChArUco 角点数
和 4x4 视野覆盖,对准了再按。--auto 换回自动挡(角点够+移动够+间隔够就存)。

    python capture_world_frames.py                     # 回车拍照,存到 ../world_camera_calibration_imgs
    python capture_world_frames.py --auto --target 80  # 自动挡
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
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
    p.add_argument("--auto", action="store_true", help="自动挡(默认手动:回车拍)")
    p.add_argument("--target", type=int, default=80, help="自动挡存满多少张退出")
    p.add_argument("--min-corners", type=int, default=40,
                   help="自动挡入选门槛;手动挡低于它只警告不拦")
    p.add_argument("--interval", type=float, default=0.7, help="自动挡两张最短间隔 (s)")
    p.add_argument("--min-move", type=float, default=60.0, help="自动挡板中心最小位移 (px)")
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

    keys: queue.Queue[str] = queue.Queue()
    if not args.auto:
        def stdin_reader():
            for line in sys.stdin:
                keys.put(line.strip().lower())
        threading.Thread(target=stdin_reader, daemon=True).start()

    print(f"连接 {args.pupil} ...")
    sub = wait_pupil(args.pupil)
    print("自动挡:角点/位移/间隔达标即存" if args.auto
          else "手动挡:[回车]拍一张  [q回车]收工(角点数看状态行,对准了再按)")

    saved, last_t, last_c, t_line = 0, 0.0, None, 0.0
    grid = set()
    try:
        for _ts, bgr in live_frames(sub):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cc, ci, _mc, _mi = det.detectBoard(gray)
            n = 0 if ci is None else len(ci)
            h, w = gray.shape
            c = cc.reshape(-1, 2).mean(0) if n else None

            def snap(warn_low):
                nonlocal saved, last_t, last_c
                cv2.imwrite(str(out / f"frame_{n0 + saved:03d}.png"), bgr)
                saved += 1
                last_t, last_c = time.time(), c
                if c is not None:
                    grid.add((min(3, int(c[0] / w * 4)), min(3, int(c[1] / h * 4))))
                note = "  [!] 角点偏少,标定时可能被剔" if warn_low and n < args.min_corners else ""
                print(f"\r[{saved}] 存 frame_{n0 + saved - 1:03d}.png  角点 {n}"
                      f"  覆盖 {len(grid)}/16 格{note}" + " " * 12, flush=True)

            if args.auto:
                if n >= args.min_corners and c is not None \
                        and (last_c is None or float(np.hypot(*(c - last_c))) >= args.min_move) \
                        and time.time() - last_t >= args.interval:
                    snap(False)
                    if saved >= args.target:
                        break
                continue

            # 手动挡:状态行 + 回车触发
            now = time.time()
            if now - t_line > 0.25:
                print(f"\r角点 {n:2d} | 已存 {saved} | 覆盖 {len(grid)}/16"
                      f"   [回车]拍  [q回车]收工   ", end="", flush=True)
                t_line = now
            try:
                k = keys.get_nowait()
            except queue.Empty:
                continue
            if k == "q":
                break
            snap(True)
    except KeyboardInterrupt:
        pass
    print(f"\n完成:本次存 {saved} 张到 {out}(覆盖 {len(grid)}/16 格;边角格没到就再补)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
