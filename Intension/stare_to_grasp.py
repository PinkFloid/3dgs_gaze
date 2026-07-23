#!/usr/bin/env python3
"""stare_to_grasp.py -- 盯住即夹取:最早的最小切片,留作对照与回归(无 LLM)。

    gaze_live.py --publish 5581 ──▶ VisitTracker(core.attention)──▶
    盯满 --dwell(默认 4.8s)──▶ 控制台问一句 ──▶ y 确认 ──▶ dispatch

组件已抬进 core/(brain.py 同用);本文件只剩这一条规则的编排。

    python Intension/stare_to_grasp.py [--skill-endpoint tcp://狗机:5583]
    python Intension/stare_to_grasp.py --raw-log                       # 录原始流供回放
    python Intension/stare_to_grasp.py --replay logs/<sess>/raw.jsonl  # 确定性回放
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.attention import VisitTracker, accepted           # noqa: E402
from core.comms import Printer, dispatch, status_listener   # noqa: E402

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
