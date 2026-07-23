"""传输与显示:技能派发(REQ)、狗状态订阅(SUB)、事件源、单行进度打印。"""

from __future__ import annotations

import json
import time


class Printer:
    """One-line live progress that never collides with real lines."""

    def __init__(self):
        self._open = False

    def progress(self, s):
        print(f"\r  {s}   ", end="", flush=True)
        self._open = True

    def say(self, s):
        if self._open:
            print()
            self._open = False
        print(s, flush=True)


def dispatch(req, endpoint):
    """Skill call, PROTOCOL.md REQ shape. Print-only unless an endpoint is given."""
    if not endpoint:
        return {"accepted": True, "reason": "print-only (no --skill-endpoint)"}
    import msgpack
    import zmq
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(endpoint)
    s.send(msgpack.packb(req))
    if s.poll(2000):
        rep = msgpack.unpackb(s.recv(), strict_map_key=False)
    else:
        rep = {"accepted": False, "reason": "skill endpoint timeout (2s)"}
    s.close()
    return rep


def status_listener(endpoint, printer, seen, logev):
    """Print the dog's skill.status broadcasts into our console (daemon thread)."""
    import msgpack
    import zmq
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"skill.status")
    while True:
        st = msgpack.unpackb(sub.recv_multipart()[-1], strict_map_key=False)
        seen[st.get("req_id", "?")] = st.get("state", "?")
        pose = st.get("pose") or {}
        line = f"[狗] {st.get('state', '?'):<10} req={st.get('req_id', '?')}"
        if pose:
            line += f"  pose=({pose.get('x', 0):+.2f},{pose.get('y', 0):+.2f})"
        if st.get("detail"):
            line += f"  {st['detail']}"
        printer.say(line)
        logev({"topic": "skill.status", **st})


def gaze_events(endpoint):
    """订阅 gaze.intent;200ms 无事件也产出 None 心跳,指令处理不依赖视线流。"""
    import msgpack
    import zmq
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"gaze.intent")
    while True:
        if sub.poll(200):
            yield msgpack.unpackb(sub.recv_multipart()[-1], strict_map_key=False)
        else:
            yield None


def replay_events(path):
    """回放 jsonl;流放完后继续心跳,让脚本指令与终态等待能收尾。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
    while True:
        time.sleep(0.05)
        yield None
