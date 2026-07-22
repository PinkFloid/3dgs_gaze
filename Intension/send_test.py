#!/usr/bin/env python3


import argparse
import json
import time

import msgpack
import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="SERVER IP")
    ap.add_argument("--rep", type=int, default=5583, help="技能请求端口")
    ap.add_argument("--pub", type=int, default=5584, help="进度广播端口")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)                      # 先订阅,2 秒等待顺便让订阅生效
    sub.connect(f"tcp://{args.host}:{args.pub}")
    sub.setsockopt(zmq.SUBSCRIBE, b"skill.status")

    req = {"v": 1, "type": "skill.request",
           "req_id": time.strftime("test-%H%M%S"),
           "sent_at": time.time(),
           "frame": "board/v2",
           "skill": "grasp",
           "params": {"object_name": "黄色机器人",
                      "target_world": [-0.185, 3.413, 0.829],
                      "deliver_to": [2.0, -1.5, 1.4]},
           "intent_summary": "send_test 固定样例"}

    print(f"2 秒后发送到 tcp://{args.host}:{args.rep} ...")
    time.sleep(2)

    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{args.host}:{args.rep}")
    s.send(msgpack.packb(req))
    print("[发送]", json.dumps(req, ensure_ascii=False))
    if not s.poll(2000):
        print("[回执] 2s 超时 —— 狗端没开、IP 不对或防火墙拦了")
        return
    rep = msgpack.unpackb(s.recv(), strict_map_key=False)
    s.close()
    print("[回执]", json.dumps(rep, ensure_ascii=False))
    if not rep.get("accepted"):
        return

    print("等待进度广播")
    while True:
        if not sub.poll(120000):
            print("auto quit")
            return
        st = msgpack.unpackb(sub.recv_multipart()[-1], strict_map_key=False)
        print("[进度]", json.dumps(st, ensure_ascii=False))
        if st.get("state") in ("done", "failed", "stopped"):
            return


if __name__ == "__main__":
    main()
