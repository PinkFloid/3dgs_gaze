#!/usr/bin/env python3
"""stare_to_grasp.py -- 盯住即夹取:AGENT_DESIGN 的最小可跑切片(无 LLM)。

    gaze_live.py --publish 5581        感知层(已有,另开终端)
        |  PUB 'gaze.intent'           per-fixation verdict(provisional + final)
        v
    VisitTracker(本文件,层A内核)    因果 visit/dwell 累积,每 visit 单发
        |  sustained: 同一物体因果注视 >= --dwell(默认 4.8s)
        v
    规则脑(本文件,层B的极简替身)   sustained -> 控制台问一句 -> y 确认
        |
        v  dispatch: 打印 grasp 调用(或 --skill-endpoint REQ 发给技能端)

用法:
    # 上游(另一终端,与 gaze_live 同一个 python 环境):
    python Eye_Tracker/tools/gaze_live.py --publish 5581 [...]
    # 本脚本(需要 pyzmq + msgpack;回放模式纯 stdlib):
    python Intension/stare_to_grasp.py
    python Intension/stare_to_grasp.py --raw-log                       # 录原始流供回放
    python Intension/stare_to_grasp.py --replay logs/<sess>/raw.jsonl  # 确定性回放,无需上游

保留自 AGENT_DESIGN.md、将来直接长成层A/层B的部分:
  * 吃 provisional 做实时 dwell,final 结账(§4.1:只吃 final 的仲裁器永远不触发);
  * visit 语义 = grasp_intent.py 的因果化:merge-gap 瞥离容忍、revisit 只数过去(§4.2);
  * 每 visit 单发 + 对话后抑制期(§4.4/§9,Midas touch 防线);
  * 确认门在代码里(§2.3):没有 y 就没有 dispatch;
  * 一切事件落 jsonl;--raw-log + --replay 给仲裁层确定性回放(§2.5);
  * --publish 5582 可选发布 attention.* 事件,即未来大脑进程的订阅口。
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import threading
import time
from collections import deque
from pathlib import Path

OBJ0 = 10  # instance labels >= OBJ0 are objects; below: floor/wall/ceiling (same as grasp_intent.py)
YES_WORDS = {"y", "yes", "是", "好", "嗯", "要", "ok", "行"}
QUIT_WORDS = {"q", "quit", "停", "stop", "exit"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sub", default="tcp://127.0.0.1:5581",
                   help="gaze_live 的 PUB 端点(话题 gaze.intent)")
    p.add_argument("--dwell", type=float, default=4.8,
                   help="触发夹取提议的因果 dwell 阈值 (s)")
    p.add_argument("--merge-gap", type=float, default=0.6,
                   help="visit 内的瞥离容忍(同 grasp_intent)(s)")
    p.add_argument("--min-vote", type=float, default=0.5,
                   help="接受 verdict 的最小 vote_share(同 grasp_intent)")
    p.add_argument("--confirm-timeout", type=float, default=8.0,
                   help="y/n 确认的等待时长,超时按取消处理 (s)")
    p.add_argument("--suppress", type=float, default=30.0,
                   help="一次问答后(无论 y/n)同一物体的静默期,流时间 (s)")
    p.add_argument("--skill-endpoint", default=None,
                   help="技能端 REQ 端点(如 tcp://127.0.0.1:5583);缺省只打印调用")
    p.add_argument("--frame", default="board/v2",
                   help="target_world 的坐标系/地图版本标识,须与狗端一致(见 PROTOCOL.md §4)")
    p.add_argument("--status-endpoint", default=None,
                   help="狗端 skill.status 的 PUB 端点;缺省由 --skill-endpoint 推导"
                        "(同主机 :5584);传 off 关闭订阅")
    p.add_argument("--publish", type=int, default=None,
                   help="可选:attention.* 事件的 PUB 端口(层A对外接口,如 5582)")
    p.add_argument("--replay", default=None,
                   help="从 --raw-log 录下的 jsonl 回放,替代 ZMQ 订阅")
    p.add_argument("--raw-log", action="store_true",
                   help="把每条进来的 gaze.intent 原样落盘,供 --replay")
    p.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    return p.parse_args()


# ------------------------------------------------------------ 层A内核:因果 visit 累积

class VisitTracker:
    """Causal visit/dwell accumulation over fixation verdicts (intent-agnostic).

    Live dwell comes from provisional verdicts of the still-open fixation;
    finals settle the books (closed_s). Same-object fixations merge across
    gaps <= merge_gap; revisits count past visits only; 'sustained' fires at
    most once per visit. What a sustained visit *means* is the caller's call.
    """

    def __init__(self, fire_dwell, merge_gap, revisit_window=90.0, release_grace=0.6):
        self.fire_dwell = fire_dwell
        self.merge_gap = merge_gap
        self.revisit_window = revisit_window
        self.release_grace = release_grace
        self.visit = None    # object/label/t_start/t_last_end/closed_s/shares/fired/last
        self.run_fx = None   # still-open fixation: t_start/dur/object
        self.past = deque()  # (t_close, object) -- causal revisit counting

    def _dwell(self):
        v = self.visit
        run = 0.0
        if v and self.run_fx and self.run_fx["object"] == v["object"] \
                and self.run_fx["t_start"] >= v["t_start"] - 1e-9:
            run = self.run_fx["dur"]
        return v["closed_s"] + run if v else 0.0

    def _revisits(self, obj, now):
        while self.past and now - self.past[0][0] > self.revisit_window:
            self.past.popleft()
        return sum(1 for _, o in self.past if o == obj)

    def _close(self, t):
        v = self.visit
        if v is None:
            return []
        dwell = self._dwell()
        # a run_fx that belongs to the closed span must not leak into the next visit
        if self.run_fx and self.run_fx["object"] == v["object"] \
                and self.run_fx["t_start"] <= v["t_last_end"] + 1e-9:
            self.run_fx = None
        self.past.append((t, v["object"]))
        self.visit = None
        return [("released", {"object": v["object"], "dwell_s": round(dwell, 2),
                              "fired": v["fired"], "t": round(t, 3)})]

    def advance(self, t):
        """Clock tick from gated-out events: merge-gap timeout release.

        Grace on top of merge_gap: a merging same-object fixation announces
        itself only ~0.4-0.5s after its t_start (first provisional), so closing
        on the raw gap would kill visits the offline semantics would merge.
        feed() still applies the strict t_start-based gap, so dwell accounting
        stays exactly grasp_intent's; grace only delays the released event.
        """
        if self.visit and t - self.visit["t_last_end"] > self.merge_gap + self.release_grace:
            return self._close(t)
        return []

    def feed(self, e):
        """One accepted verdict in; a list of (kind, payload) events out."""
        out = []
        t0, t1 = float(e["t_start"]), float(e["t_end"])
        dur, obj = float(e["duration_s"]), e["object"]
        if e.get("provisional"):
            self.run_fx = {"t_start": t0, "dur": dur, "object": obj}
        elif self.run_fx and abs(self.run_fx["t_start"] - t0) < 1e-9:
            self.run_fx = None  # this fixation's final settles it below

        v = self.visit
        if v is None or obj != v["object"] or t0 - v["t_last_end"] > self.merge_gap:
            out += self._close(t1)
            self.visit = v = {"object": obj, "label": e.get("object_label"),
                              "t_start": t0, "t_last_end": t1, "closed_s": 0.0,
                              "shares": [], "fired": False, "last": e,
                              "revisits": self._revisits(obj, t0)}
        v["t_last_end"] = max(v["t_last_end"], t1)
        v["shares"].append(float(e.get("vote_share", 0.0)))
        v["last"] = e
        if not e.get("provisional"):
            v["closed_s"] += dur

        dwell = self._dwell()
        out.append(("progress", {"object": obj, "dwell_s": round(dwell, 2),
                                 "share": float(e.get("vote_share", 0.0))}))
        if not v["fired"] and dwell >= self.fire_dwell - 1e-9:
            v["fired"] = True
            last = v["last"]
            out.append(("sustained", {
                "object": v["object"],
                "target_world": last.get("object_centroid_world"),
                "dwell_s": round(dwell, 2),
                "revisits": v["revisits"],
                "mean_vote_share": round(sum(v["shares"]) / len(v["shares"]), 2),
                "p_none": last.get("p_none"),
                "sigma_deg": last.get("sigma_deg"),
                "candidates": last.get("candidates"),
                "t": round(t1, 3),
            }))
        return out


# ------------------------------------------------------------ IO

def accepted(e, min_vote):
    """Same gate as grasp_intent.py: objects only, clean cone verdicts."""
    return (e.get("object_label", -1) >= OBJ0
            and e.get("vote_share", 0.0) >= min_vote
            and e.get("mode") == "cone"
            and e.get("object_centroid_world"))


def zmq_source(endpoint):
    import msgpack
    import zmq
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.connect(endpoint)
    sub.setsockopt(zmq.SUBSCRIBE, b"gaze.intent")
    print(f"订阅 {endpoint} 话题 gaze.intent ... (Ctrl-C 退出)")
    while True:
        parts = sub.recv_multipart()
        yield msgpack.unpackb(parts[-1], strict_map_key=False)


def replay_source(path):
    print(f"回放 {path}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dispatch(req, endpoint):
    """Skill call, AGENT_DESIGN §8 REQ shape. Print-only unless an endpoint is given."""
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


def ask(printer, question, timeout):
    printer.say(question)
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        printer.say(f"  (无输入 {timeout:.0f}s,按取消处理)")
        return ""
    return sys.stdin.readline().strip()


# ------------------------------------------------------------ main

def main() -> int:
    args = parse_args()
    sess = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
    sess.mkdir(parents=True, exist_ok=True)
    ev_f = open(sess / "events.jsonl", "w", encoding="utf-8")
    # line-buffered: the raw dump must survive an unclean kill mid-stream
    raw_f = open(sess / "raw.jsonl", "w", encoding="utf-8", buffering=1) if args.raw_log else None

    apub = None
    if args.publish:
        import msgpack
        import zmq
        apub = zmq.Context.instance().socket(zmq.PUB)
        apub.bind(f"tcp://*:{args.publish}")
        apub_pack = msgpack.packb

    loglock = threading.Lock()

    def logev(rec):
        with loglock:
            ev_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ev_f.flush()

    def emit(kind, payload):
        rec = {"topic": f"attention.{kind}", **payload}
        logev(rec)
        if apub:
            apub.send_multipart([rec["topic"].encode(), apub_pack(rec)])

    tracker = VisitTracker(args.dwell, args.merge_gap)
    suppress = {}  # object -> stream-time until which it stays silent
    P = Printer()
    status_seen = None
    if args.skill_endpoint and args.status_endpoint != "off":
        sep = args.status_endpoint or args.skill_endpoint.rsplit(":", 1)[0] + ":5584"
        status_seen = {}
        threading.Thread(target=status_listener, args=(sep, P, status_seen, logev),
                         daemon=True).start()
    n_ask = n_go = n_req = 0
    source = replay_source(args.replay) if args.replay else zmq_source(args.sub)
    P.say(f"规则:同一物体因果注视 >= {args.dwell:.1f}s -> 询问 -> y 确认 -> grasp"
          f"  (抑制期 {args.suppress:.0f}s, 日志 {sess})")

    try:
        for e in source:
            if raw_f:
                raw_f.write(json.dumps(e, ensure_ascii=False) + "\n")
            t = float(e.get("t_end", e.get("t_start", 0.0)))
            # accepted events do their own (strict, t_start-based) gap check in
            # feed(); only gated-out events drive the timeout clock
            if accepted(e, args.min_vote):
                events = tracker.feed(e)
            else:
                events = tracker.advance(t)
            for kind, pl in events:
                if kind == "progress":
                    P.progress(f"盯 {pl['object']:<14} {pl['dwell_s']:4.1f}/{args.dwell:.1f}s"
                               f"  vote {pl['share']:3.0%}")
                    continue
                if kind == "released":
                    emit(kind, pl)
                    if pl["dwell_s"] >= 1.0:
                        P.say(f"[·] 离开 {pl['object']}(累计 {pl['dwell_s']:.1f}s)")
                    continue
                # ---- sustained: the one rule of this demo ----
                emit(kind, pl)
                obj, tw = pl["object"], pl["target_world"]
                until = suppress.get(obj, float("-inf"))
                if t < until:
                    P.say(f"[×] {obj} 在抑制期(剩 {until - t:.0f}s),本次注视忽略")
                    logev({"topic": "dialog.skipped", "object": obj, "t": t,
                           "suppressed_until": until})
                    continue
                n_ask += 1
                ans = ask(P, f"[?] 盯了 {pl['dwell_s']:.1f}s -> 夹取「{obj}」"
                             f" ({tw[0]:+.2f},{tw[1]:+.2f},{tw[2]:+.2f})m ?"
                             f"  y=确认 其他=取消 ({args.confirm_timeout:.0f}s):",
                          args.confirm_timeout)
                if ans.lower() in QUIT_WORDS:
                    P.say("[!] 退出")
                    raise KeyboardInterrupt
                yes = ans.lower() in YES_WORDS
                suppress[obj] = t + args.suppress
                logev({"topic": "dialog", "object": obj, "answer": ans, "yes": yes, "t": t})
                if not yes:
                    P.say(f"[-] 已取消,{obj} 静默 {args.suppress:.0f}s")
                    continue
                n_req += 1
                req = {"v": 1, "type": "skill.request", "skill": "grasp",
                       "params": {"object_name": obj, "target_world": tw},
                       "req_id": f"{sess.name}-{n_req:03d}",
                       "frame": args.frame, "sent_at": time.time(), "t_stream": t,
                       "intent_summary": f"用户注视 {pl['dwell_s']:.1f}s 并确认夹取 {obj}"}
                rep = dispatch(req, args.skill_endpoint)
                logev({"topic": "skill.req", **req, "rep": rep})
                n_go += 1
                P.say(f"[GRASP] {json.dumps(req, ensure_ascii=False)}")
                P.say(f"        -> {json.dumps(rep, ensure_ascii=False)}")
        # 事件流走完但狗可能还在执行:等它到终态再退出(回放模式常见)
        if status_seen is not None and n_go:
            P.say("[·] 事件流结束,等待狗端执行完成(最多60s,Ctrl-C 跳过)")
            t_wait = time.time()
            while time.time() - t_wait < 60:
                if status_seen and all(s in ("done", "failed", "stopped")
                                       for s in status_seen.values()):
                    break
                time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    finally:
        P.say(f"结束:询问 {n_ask} 次,确认执行 {n_go} 次。日志:{sess}")
        ev_f.close()
        if raw_f:
            raw_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
