#!/usr/bin/env python3
"""say_to_fetch.py -- 指令驱动的取物入口(键盘先行,ASR 后换):

    拿一下黄色机器人        名字消解:模糊匹配地图物体名,不需要注视
    把这个杯子拿来          视线消解:眼-声窗口取近期注视目标,名词过滤类别
    停                      急停旁路

与 stare_to_grasp.py(盯 4.8s 主动问询)共用层A/确认/派发/协议,互不影响。

    python Intension/say_to_fetch.py [--skill-endpoint tcp://狗机:5583]
    # 回放回归(确定性):
    python Intension/say_to_fetch.py --replay /tmp/fake.jsonl --yes \
        --script "106.5:把这个杯子拿来" --script "111:拿一下碗"
"""

from __future__ import annotations

import argparse
import difflib
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stare_to_grasp import (Printer, VisitTracker, accepted, dispatch,  # noqa: E402
                            status_listener)

DEICTIC = ("这个", "那个", "这只", "那只", "这台", "那台", "这", "那")
ACTIONS = ("给我拿", "帮我拿", "拿一下", "拿个", "拿来", "拿", "取", "抓", "给我", "要")
TAILS = ("拿过来", "拿给我", "拿来", "过来", "一下", "拿", "取", "抓", "来", "吧", "啊", "了", "的")
YES_WORDS = {"y", "yes", "是", "好", "嗯", "要", "ok", "行"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sub", default="tcp://127.0.0.1:5581")
    p.add_argument("--map-dir", default=str(Path(__file__).resolve().parents[1]
                                            / "SceneRebuild/lab_result/segmentation_sam"),
                   help="instances.json + names.json 所在目录(名字消解用)")
    p.add_argument("--merge-gap", type=float, default=0.6)
    p.add_argument("--min-vote", type=float, default=0.5)
    p.add_argument("--lookback", type=float, default=4.0,
                   help="视线消解回看窗:说指代词前多少秒内的注视算数")
    p.add_argument("--confirm-timeout", type=float, default=10.0)
    p.add_argument("--proactive", type=float, default=0.0,
                   help=">0 时开启第三模式:盯满该秒数主动问询(如 4.8);默认关")
    p.add_argument("--suppress", type=float, default=30.0,
                   help="主动问询被拒后同物体静默期 (s)")
    p.add_argument("--skill-endpoint", default=None)
    p.add_argument("--status-endpoint", default=None)
    p.add_argument("--frame", default="board/v2")
    p.add_argument("--replay", default=None)
    p.add_argument("--script", action="append", default=[],
                   help="回归用脚本指令 '流时间:指令文本',可重复")
    p.add_argument("--yes", action="store_true", help="自动确认(回归测试用)")
    p.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    return p.parse_args()


def _noun_match(noun, obj):
    """类别启发式:"杯子"命中"水杯"、"机器人"不误中"机械臂"。真类别字段随新地图上。"""
    return noun in obj or obj in noun or (len(noun) >= 2 and noun[:-1] in obj)


# ------------------------------------------------------------ 层A + 近期注意缓冲

class AttentionBuffer(VisitTracker):
    """VisitTracker 原语义,外加最近 visit 的富记录,供眼-声绑定查询。"""

    def __init__(self, merge_gap, fire_dwell=0.0):
        # fire_dwell<=0 = 纯缓冲不触发;>0 = 同时兼任主动问询的触发器(--proactive)
        super().__init__(fire_dwell=fire_dwell if fire_dwell > 0 else float("inf"),
                         merge_gap=merge_gap)
        self.recent = []  # 已关闭 visit 的富记录,按关闭时间升序

    def _close(self, t):
        v = self.visit
        out = super()._close(t)
        if v is not None and out:
            last = v["last"]
            self.recent.append({"object": v["object"], "t_start": v["t_start"],
                                "t_end": v["t_last_end"],
                                "dwell_s": out[0][1]["dwell_s"],
                                "vote": float(last.get("vote_share", 0.0)),
                                "target_world": last.get("object_centroid_world")})
            self.recent = self.recent[-50:]
        return out

    def candidates(self, t_word, lookback, noun=""):
        """说指代词时刻往前看:仍在盯的目标排最前,其余按离开时间近排序。"""
        out = []
        if self.visit is not None:
            v, last = self.visit, self.visit["last"]
            out.append({"object": v["object"],
                        "gap": max(0.0, t_word - v["t_last_end"]),
                        "dwell_s": self._dwell(),
                        "vote": float(last.get("vote_share", 0.0)),
                        "target_world": last.get("object_centroid_world")})
        for r in reversed(self.recent):
            gap = t_word - r["t_end"]
            if gap > lookback:
                break
            if gap >= -0.5:
                out.append({**r, "gap": max(0.0, gap)})
        if noun:
            out = [c for c in out if _noun_match(noun, c["object"])]
        out.sort(key=lambda c: c["gap"])
        return out


# ------------------------------------------------------------ 指令解析与名字消解

def parse_command(text):
    t = "".join(text.split())
    if not t:
        return None
    if t.lower() in ("停", "stop", "s"):
        return {"kind": "stop"}
    if not any(a in t for a in ("拿", "取", "抓", "要", "给")):
        return {"kind": "help"}
    for m in DEICTIC:
        if m in t:
            noun = t.split(m, 1)[1]
            for suf in TAILS:
                noun = noun.replace(suf, "")
            return {"kind": "deictic", "noun": noun}
    q = t
    for a in sorted(ACTIONS, key=len, reverse=True):
        q = q.replace(a, "")
    for suf in TAILS:
        q = q.replace(suf, "")
    return {"kind": "named", "query": q}


def load_object_table(map_dir):
    """物体名 -> 池化质心(n_gaussians 加权,与 gaze_live 同语义)。"""
    d = Path(map_dir)
    inst = json.load(open(d / "instances.json", encoding="utf-8"))["instances"]
    names = json.load(open(d / "names.json", encoding="utf-8"))
    acc = {}
    for it in inst:
        name = names.get(str(int(it["id"])), "")
        if not name:
            continue
        w = max(float(it.get("n_gaussians", 1)), 1.0)
        s, tw = acc.get(name, ([0.0, 0.0, 0.0], 0.0))
        acc[name] = ([s[i] + w * float(it["centroid"][i]) for i in range(3)], tw + w)
    return {n: [s[i] / w for i in range(3)] for n, (s, w) in acc.items()}


def resolve_named(query, table):
    """返回 (唯一命中名 或 None, 前三候选[(score,name)])。"""
    if not query or not table:
        return None, []
    scored = []
    for name in table:
        if query == name:
            s = 1.0
        elif query in name or name in query:
            s = 0.8 + 0.2 * min(len(query), len(name)) / max(len(query), len(name))
        else:
            s = difflib.SequenceMatcher(None, query, name).ratio()
        scored.append((round(s, 3), name))
    scored.sort(reverse=True)
    best_s, best = scored[0]
    if best_s < 0.55:
        return None, scored[:3]
    if len(scored) > 1 and scored[1][0] >= max(0.55, best_s - 0.08):
        return None, scored[:3]
    return best, scored[:3]


# ------------------------------------------------------------ 事件源(带心跳,指令不依赖视线流)

def zmq_events(endpoint):
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
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
    while True:  # 流放完后继续心跳,让脚本指令与终态等待能收尾
        time.sleep(0.05)
        yield None


# ------------------------------------------------------------ main

def main() -> int:
    args = parse_args()
    sess = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
    sess.mkdir(parents=True, exist_ok=True)
    ev_f = open(sess / "events.jsonl", "w", encoding="utf-8")
    loglock = threading.Lock()

    def logev(rec):
        with loglock:
            ev_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ev_f.flush()

    P = Printer()
    try:
        table = load_object_table(args.map_dir)
        P.say(f"物体表:{len(table)} 个命名物体(名字消解可用)")
    except Exception as e:  # 地图缺失只失去名字模式,视线模式照常
        table = {}
        P.say(f"[!] 物体表加载失败({e}),名字消解不可用")

    buf = AttentionBuffer(args.merge_gap, args.proactive)
    suppress = {}  # object -> 流时间,主动问询被拒后的静默截止
    status_seen = None
    if args.skill_endpoint and args.status_endpoint != "off":
        sep = args.status_endpoint or args.skill_endpoint.rsplit(":", 1)[0] + ":5584"
        status_seen = {}
        threading.Thread(target=status_listener, args=(sep, P, status_seen, logev),
                         daemon=True).start()

    last_req = {"id": None}

    def dog_busy():
        """上一个已接受的请求还没到终态 -> 狗在忙,主动问询让路。"""
        if status_seen is None or last_req["id"] is None:
            return False
        return status_seen.get(last_req["id"]) not in ("done", "failed", "stopped")

    cmd_q = queue.Queue()
    if not args.script:
        def stdin_reader():
            for line in sys.stdin:
                cmd_q.put(line.rstrip("\n"))
        threading.Thread(target=stdin_reader, daemon=True).start()
    scripted = sorted((float(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in args.script)

    n_req = 0
    pending = None  # {"req":..., "since": stream_t}
    clock = {"stream": 0.0, "wall": time.time()}

    def stream_now():
        return clock["stream"] + (time.time() - clock["wall"])

    def send(req):
        nonlocal n_req
        rep = dispatch(req, args.skill_endpoint)
        logev({"topic": "skill.req", **req, "rep": rep})
        P.say(f"[派发] {json.dumps(req, ensure_ascii=False)}")
        P.say(f"       -> {json.dumps(rep, ensure_ascii=False)}")
        if rep.get("accepted") and req.get("skill") != "stop":
            last_req["id"] = req["req_id"]

    def propose(obj, tw, mode, t_word):
        nonlocal n_req, pending
        n_req += 1
        req = {"v": 1, "type": "skill.request", "skill": "grasp",
               "params": {"object_name": obj, "target_world": tw},
               "req_id": f"{sess.name}-{n_req:03d}", "frame": args.frame,
               "sent_at": time.time(), "t_stream": round(t_word, 3),
               "intent_summary": f"指令({mode})消解为 {obj}"}
        logev({"topic": "resolution", "mode": mode, "object": obj, "t": t_word})
        if args.yes:
            send(req)
        else:
            pending = {"req": req, "since": t_word, "mode": mode}
            ask = (f"[?] 你在看「{obj}」——要我拿来吗?" if mode == "主动"
                   else f"[?] 去拿「{obj}」({tw[0]:+.2f},{tw[1]:+.2f},{tw[2]:+.2f})m ?")
            P.say(ask + " y=确认 其他=取消")

    def handle(t_word, text):
        nonlocal pending
        cmd = parse_command(text)
        if pending is not None:
            prev, pending = pending, None
            if text.strip().lower() in YES_WORDS:
                send(prev["req"])
                if prev["mode"] == "主动":  # 执行完还盯着,也别立刻再问
                    suppress[prev["req"]["params"]["object_name"]] = t_word + args.suppress
                return
            if prev["mode"] == "主动":  # 拒绝主动提议 -> 抑制该物体,免得追着问
                suppress[prev["req"]["params"]["object_name"]] = t_word + args.suppress
            P.say("[-] 已取消")
            if cmd is None or cmd["kind"] == "help":
                return  # 纯拒绝;若输入本身是新指令,则继续往下执行它
        if cmd is None:
            return
        logev({"topic": "command", "text": text, "t": t_word, "kind": cmd["kind"]})
        if cmd["kind"] == "stop":
            send({"v": 1, "type": "skill.request", "skill": "stop",
                  "req_id": f"{sess.name}-stop", "frame": args.frame,
                  "sent_at": time.time(), "params": {}})
            return
        if cmd["kind"] == "help":
            P.say("用法:拿一下<名字> | 把这个<类别>拿来 | 停")
            return
        if cmd["kind"] == "named":
            name, top = resolve_named(cmd["query"], table)
            if name is None:
                P.say(f"[×] 「{cmd['query']}」没有唯一命中,最像的:"
                      + " / ".join(f"{n}({s:.2f})" for s, n in top))
                return
            propose(name, table[name], "名字", t_word)
            return
        # deictic
        cands = buf.candidates(t_word, args.lookback, cmd["noun"])
        if not cands:
            if cmd["noun"]:
                name, _ = resolve_named(cmd["noun"], table)
                if name:
                    P.say(f"[·] 近期没注视「{cmd['noun']}」,按名字兜底 -> {name}")
                    propose(name, table[name], "名字兜底", t_word)
                    return
            P.say(f"[×] 最近 {args.lookback:.0f}s 没有可用注视目标"
                  + (f"(类别「{cmd['noun']}」)" if cmd["noun"] else ""))
            return
        c = cands[0]
        logev({"topic": "binding", "t_word": t_word, "noun": cmd["noun"],
               "candidates": cands[:3]})
        propose(c["object"], c["target_world"], "视线", t_word)

    source = replay_events(args.replay) if args.replay else zmq_events(args.sub)
    P.say(f"指令入口就绪:拿一下<名字> / 把这个<类别>拿来 / 停"
          f"  (回看窗 {args.lookback:.0f}s,日志 {sess})")
    try:
        for e in source:
            if e is not None:
                t = float(e.get("t_end", e.get("t_start", 0.0)))
                clock["stream"], clock["wall"] = t, time.time()
                if accepted(e, args.min_vote):
                    for kind, pl in buf.feed(e):
                        if kind == "progress":
                            P.progress(f"看 {pl['object']:<14} {pl['dwell_s']:4.1f}s"
                                       f"  vote {pl['share']:3.0%}")
                        elif kind == "sustained" and args.proactive > 0:
                            if pending is not None:
                                logev({"topic": "proactive.skipped", "object": pl["object"],
                                       "why": "pending"})
                            elif dog_busy():
                                logev({"topic": "proactive.skipped", "object": pl["object"],
                                       "why": "dog_busy"})
                            elif pl["t"] < suppress.get(pl["object"], float("-inf")):
                                P.say(f"[×] {pl['object']} 抑制期,略过主动问询")
                            elif pl.get("target_world"):
                                propose(pl["object"], pl["target_world"], "主动", pl["t"])
                else:
                    buf.advance(t)
            st = stream_now()
            while scripted and st >= scripted[0][0]:
                _, text = scripted.pop(0)
                P.say(f"[指令] {text}")
                handle(st, text)
            while not cmd_q.empty():
                handle(st, cmd_q.get())
            if pending and st - pending["since"] > args.confirm_timeout:
                P.say("[-] 确认超时,已取消")
                if pending["mode"] == "主动":
                    suppress[pending["req"]["params"]["object_name"]] = st + args.suppress
                pending = None
            if args.replay and not scripted and pending is None:
                done = status_seen is None or not status_seen or \
                    all(s in ("done", "failed", "stopped") for s in status_seen.values())
                if e is None and done:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        P.say(f"结束。日志:{sess}")
        ev_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
