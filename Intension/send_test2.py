#!/usr/bin/env python3
"""send_test2.py -- 顺序双步测试(狗端联调,不需要视线管线):

    步骤1  grasp{"苹果", T1}        去 T1 抓苹果(等终态)
    步骤2  grasp{null,  T2}        纯导航走回 T2(冻结定义:空 object=只走不抓)

    python send_test2.py                              # 本机 dog_link
    python send_test2.py --host 192.168.1.7           # 狗机
    python send_test2.py --t1 -0.1,-1.1,0.5 --t2 0,0,0
"""

import argparse
import json
import time

import msgpack
import zmq

TERMINAL = ("done", "failed", "stopped")


def xyz(s):
    v = [float(x) for x in s.split(",")]
    assert len(v) == 3, "坐标格式 x,y,z"
    return v


def send(ctx, host, port, req):
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{host}:{port}")
    s.send(msgpack.packb(req))
    print("[发送]", json.dumps(req, ensure_ascii=False))
    if not s.poll(2000):
        s.close()
        print("[回执] 2s 超时 —— 狗端没开、IP 不对或防火墙拦了")
        return None
    rep = msgpack.unpackb(s.recv(), strict_map_key=False)
    s.close()
    print("[回执]", json.dumps(rep, ensure_ascii=False))
    return rep


def wait_terminal(sub, req_id, timeout=120):
    """跟着 5584 的进度走到本请求的终态;返回终态名或 None(超时)。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not sub.poll(1000):
            continue
        st = msgpack.unpackb(sub.recv_multipart()[-1], strict_map_key=False)
        if st.get("req_id") != req_id:
            continue
        print("[进度]", json.dumps(st, ensure_ascii=False))
        if st.get("state") in TERMINAL:
            return st["state"]
    print(f"[进度] {timeout}s 无终态,放弃等待")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="狗机 IP")
    ap.add_argument("--rep", type=int, default=5583)
    ap.add_argument("--pub", type=int, default=5584)
    ap.add_argument("--t1", type=xyz, default=[-0.1, -1.1, -2.0],
                    help="步骤1 抓取目标 x,y,z(注意 z 需在夹爪可达内,狗端默认 0.02~0.90)")
    ap.add_argument("--t2", type=xyz, default=[0.0, 0.0, 0.0], help="步骤2 导航目标 x,y,z")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)              # 先订阅,2 秒等待顺便让订阅生效
    sub.connect(f"tcp://{args.host}:{args.pub}")
    sub.setsockopt(zmq.SUBSCRIBE, b"skill.status")
    print(f"2 秒后开始,目标 tcp://{args.host}:{args.rep} ...")
    time.sleep(2)

    base = {"v": 1, "type": "skill.request", "frame": "board/v2"}
    tag = time.strftime("%H%M%S")

    print("---- 步骤1:去抓苹果 ----")
    rep = send(ctx, args.host, args.rep, {
        **base, "req_id": f"test2-{tag}-1", "sent_at": time.time(),
        "skill": "grasp",
        "params": {"object_name": "苹果", "target_world": args.t1},
        "intent_summary": "send_test2 步骤1:抓苹果"})
    state = None
    if rep and rep.get("accepted"):
        state = wait_terminal(sub, f"test2-{tag}-1")
    if state != "done":
        print(f"[!] 步骤1 未完成(", (rep or {}).get("reason") or state, "),仍继续测步骤2链路")

    print("---- 步骤2:纯导航回去 ----")
    rep = send(ctx, args.host, args.rep, {
        **base, "req_id": f"test2-{tag}-2", "sent_at": time.time(),
        "skill": "grasp",
        "params": {"object_name": None, "target_world": args.t2},
        "intent_summary": "send_test2 步骤2:导航返回"})
    if rep and rep.get("accepted"):
        wait_terminal(sub, f"test2-{tag}-2")
    print("---- 结束 ----")


if __name__ == "__main__":
    main()
