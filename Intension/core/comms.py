"""传输与显示:技能派发(REQ)、狗状态订阅(SUB)、事件源、单行进度打印。"""

from __future__ import annotations

import json
import time


class Printer:
    """One-line live progress that never collides with real lines.

    plain=True 时进度也走整行输出——配 prompt_toolkit 的 patch_stdout 用:
    \r 覆写在那种终端下会被拆成一行行残迹,整行打印才干净。
    """

    def __init__(self, plain=False):
        self.plain = plain
        self._open = False

    def progress(self, s):
        if self.plain:
            self.say(f"  {s}")
            return
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
    """Print the dog's skill.status broadcasts into our console (daemon thread).

    抢占语义下大脑不做忙闲记账,seen 只服务显示与回放收尾(等终态)。
    """
    import msgpack
    import zmq
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"skill.status")
    terminal = ("done", "failed", "stopped")
    while True:
        st = msgpack.unpackb(sub.recv_multipart()[-1], strict_map_key=False)
        rid, state = st.get("req_id", "?"), st.get("state", "?")
        if seen.get(rid) in terminal and state not in terminal:
            logev({"topic": "skill.status", **st, "stale_after_terminal": True})
            continue  # 终态粘滞:真实狗端 done 后会晚到一条 running,只进日志不改状态
        seen[rid] = state
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


def replay_events(path, pace=0.0):
    """回放 jsonl;pace>0 = 按流时间 1/pace 倍实速播放(交互试跑,行为对齐 live),
    0 = 全速灌入(回归用,指令必须走 --script)。流放完后继续心跳收尾。"""
    last = None
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    e = json.loads(line)
                except ValueError:
                    bad += 1  # 手拼文件接缝的粘行等:跳过坏行,别一行毒死全场
                    if bad == 1:
                        print(f"[!] 回放流有坏行,已跳过(首个于第 {bad} 次出现;"
                              f"检查文件是否缺换行拼接)", flush=True)
                    continue
                if pace > 0:
                    t = float(e.get("t_end", e.get("t_start", 0.0)))
                    if last is not None and t > last:
                        time.sleep(min((t - last) / pace, 1.0))
                    last = t
                yield e
    while True:
        time.sleep(0.05)
        yield None
