#!/usr/bin/env python3
"""brain.py -- Intension 层本体:多模态指令 -> 指代消解 -> 确认 -> 派发。

    拿一下黄色机器人            名字消解:模糊匹配地图物体名,不需要注视
    把这个杯子拿来              视线消解:眼-声窗口取近期注视目标,名词过滤类别
    帮我把那个黄颜色的…弄过来    开放句式:语法接不住时交给 LLM 解析(--llm)
    停                          急停旁路
    (--proactive 4.8 加开第三模式:盯满主动问询)

解析:除"停"与 y/n 两个硬旁路外,指令全部由 LLM 转结构(OpenAI 直连,默认
gpt-5-mini;key 放环境变量 OPENAI_API_KEY 或 Intension/.openai_key,已 gitignore)。
同一句话的解析结果进 parse_cache.json:缓存命中 0ms 且完全确定——demo 台词
预热一遍后不再依赖网络;--llm off = 只走缓存(离线回归模式)。
LLM 只做"文本->结构";绑定/几何/确认永远是确定性代码。

    python Intension/brain.py [--skill-endpoint tcp://狗机:5583]
    # 回放回归(确定性):
    python Intension/brain.py --llm off --replay /tmp/fake.jsonl --yes \
        --script "106.5:把这个杯子拿来" --script "111:拿一下碗"
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stare_to_grasp import (Printer, VisitTracker, accepted, dispatch,  # noqa: E402
                            status_listener)

YES_WORDS = {"y", "yes", "是", "好", "嗯", "要", "ok", "行"}
NO_WORDS = {"n", "no", "不", "不用", "不要", "算了", "否", "取消"}
STOP_WORDS = {"停", "stop", "s"}
SCHEMA = Path(__file__).resolve().parent / "parse_schema.json"
PARSE_SCHEMA = json.loads(SCHEMA.read_text(encoding="utf-8"))


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
    p.add_argument("--llm", choices=["on", "off"], default="on",
                   help="off=只走 parse_cache.json(离线/回归);on=缓存未命中时调 OpenAI(默认)")
    p.add_argument("--llm-model", default="gpt-5-mini",
                   help="OpenAI 解析模型;key 读 OPENAI_API_KEY 或 Intension/.openai_key")
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


# ------------------------------------------------------------ 名字消解

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
    if query in table:  # 精确命中直接赢:名字是标识符,残余风险交给确认门
        return query, [(1.0, query)]
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
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not openai_key:
        kf = Path(__file__).resolve().parent / ".openai_key"
        if kf.exists():
            openai_key = kf.read_text(encoding="utf-8").strip()
    if args.llm == "on" and not openai_key:
        P.say("[!] 未配置 OPENAI_API_KEY(环境变量或 Intension/.openai_key),只用解析缓存")
        args.llm = "off"

    cache_path = Path(__file__).resolve().parent / "parse_cache.json"
    try:
        parse_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        parse_cache = {}

    def llm_parse(text):
        """指令文本 -> 结构。缓存命中 0ms 且确定;未命中走 OpenAI;都不行 None。"""
        data = parse_cache.get(text)
        cached = data is not None
        if not cached:
            if args.llm == "off":
                return None
            prompt = (
                "把这句对机器人说的中文指令解析成 JSON(只输出 JSON)。\n"
                "机器人技能(action 取值):\n"
                "- fetch: 去拿某个物体并送回来(需要一个物体目标)\n"
                "- goto: 只移动过去,不抓取——去某物体旁边,或来用户身边('过来')\n"
                "- stop: 让它立刻停下\n"
                "- none: 都不是\n"
                "场景中已命名的物体(object_query 与 location_hint 只能取其中之一或 null):\n"
                f"{'、'.join(sorted(table))}\n"
                "字段规则:\n"
                "- deictic: 用了'这个/那个/那边'等现场指代、且没指名是上表中哪一个时为 true\n"
                "- object_query: 目标物体(fetch=要拿的物;goto=要去的参照物)->\n"
                "  上表中最匹配的一个;没指名、或说法同时匹配多个而无法确定时为 null\n"
                "- noun_class: 指代或泛指时的类别词(如 杯、机器人);没有则 null\n"
                "- location_hint: 顺带提到的地点参照物 -> 上表中的一个;没有则 null\n"
                "- deliver_to_user: fetch=是否要求送到用户身边;goto=目的地是否就是用户身边\n"
                f"指令:「{text}」")
            t0 = time.time()
            P.say("[LLM] 解析中…")
            body = json.dumps({
                "model": args.llm_model,
                "reasoning_effort": "minimal",  # 解析任务不需要深思,省时省钱
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "robot_command", "strict": True,
                                                    "schema": PARSE_SCHEMA}},
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {openai_key}"})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    resp = json.loads(r.read())
                data = json.loads(resp["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as e:
                P.say(f"[LLM] API {e.code}: {e.read().decode(errors='ignore')[:160]}")
                return None
            except Exception as e:
                P.say(f"[LLM] 解析失败({type(e).__name__}: {e})")
                return None
            parse_cache[text] = data
            try:  # 缓存落盘:demo 台词预热一遍后离线可用
                cache_path.write_text(json.dumps(parse_cache, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
            except Exception:
                pass
            P.say(f"[LLM] {json.dumps(data, ensure_ascii=False)}  ({time.time() - t0:.1f}s)")
        logev({"topic": "llm_parse", "text": text, "result": data, "cached": cached})
        if data.get("action") == "stop":
            return {"kind": "stop"}
        if data.get("action") == "goto":
            return {"kind": "goto", "query": data.get("object_query"),
                    "noun": data.get("noun_class") or "",
                    "deictic": bool(data.get("deictic")),
                    "to_user": bool(data.get("deliver_to_user"))}
        if data.get("action") != "fetch":
            return {"kind": "help"}
        if data.get("deictic"):
            return {"kind": "deictic", "noun": data.get("noun_class") or ""}
        if data.get("object_query"):
            return {"kind": "named", "query": data["object_query"],
                    "location": data.get("location_hint")}
        if data.get("noun_class"):
            return {"kind": "deictic", "noun": data["noun_class"]}
        return {"kind": "help"}
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

    user_pos = {"xyz": None, "t": 0.0}  # 每条 verdict 的 origin_world = 用户头部位置

    def propose(obj, tw, mode, t_word, goto=False):
        """goto=True 时发狗端真实的 move_to(纯导航);否则 grasp。"""
        nonlocal n_req, pending
        n_req += 1
        if goto:
            params = {"x": round(tw[0], 3), "y": round(tw[1], 3)}
        else:
            params = {"object_name": obj, "target_world": tw}
            if user_pos["xyz"] is not None:  # 确认时刻的用户位置:带它=送达,缺省=原地done
                params["deliver_to"] = user_pos["xyz"]
        req = {"v": 1, "type": "skill.request",
               "skill": "move_to" if goto else "grasp",
               "params": params,
               "req_id": f"{sess.name}-{n_req:03d}", "frame": args.frame,
               "sent_at": time.time(), "t_stream": round(t_word, 3),
               "intent_summary": f"指令({mode}){'导航至' if goto else '消解为'} {obj}"}
        logev({"topic": "resolution", "mode": mode, "object": obj, "goto": goto, "t": t_word})
        if args.yes:
            send(req)
        else:
            pending = {"req": req, "since": t_word, "mode": mode}
            ask = (f"[?] 你在看「{obj}」——要我拿来吗?" if mode == "主动"
                   else f"[?] 过去「{obj}」({tw[0]:+.2f},{tw[1]:+.2f},{tw[2]:+.2f})m ?" if goto
                   else f"[?] 去拿「{obj}」({tw[0]:+.2f},{tw[1]:+.2f},{tw[2]:+.2f})m ?")
            P.say(ask + " y=确认 其他=取消")

    def handle(t_word, text):
        nonlocal pending
        t = "".join(text.split())
        if not t:
            return
        was_decline = False
        if pending is not None:
            prev, pending = pending, None
            if t.lower() in YES_WORDS:
                send(prev["req"])
                if prev["mode"] == "主动":  # 执行完还盯着,也别立刻再问
                    suppress[prev["req"]["params"]["object_name"]] = t_word + args.suppress
                return
            if prev["mode"] == "主动":  # 拒绝主动提议 -> 抑制该物体,免得追着问
                suppress[prev["req"]["params"]["object_name"]] = t_word + args.suppress
            P.say("[-] 已取消")
            if t.lower() in NO_WORDS:
                return
            was_decline = True  # 可能是"取消旧的换新指令":往下试,解析不出就保持安静
        if t.lower() in STOP_WORDS:  # 急停硬旁路:永不过 LLM
            send({"v": 1, "type": "skill.request", "skill": "stop",
                  "req_id": f"{sess.name}-stop", "frame": args.frame,
                  "sent_at": time.time(), "params": {}})
            return
        cmd = llm_parse(t)
        logev({"topic": "command", "text": text, "t": t_word,
               "kind": cmd["kind"] if cmd else "parse_fail"})
        if cmd is None:
            if not was_decline:
                P.say("[×] 解析不可用(缓存未命中且 LLM 不可达)")
            return
        if cmd["kind"] == "stop":
            send({"v": 1, "type": "skill.request", "skill": "stop",
                  "req_id": f"{sess.name}-stop", "frame": args.frame,
                  "sent_at": time.time(), "params": {}})
            return
        if cmd["kind"] == "help":
            if not was_decline:
                P.say("我能做:拿取场景里的物体。例:拿一下显示器 / 把这个杯子拿来 / 停")
            return
        if cmd["kind"] == "goto":  # 高维 move_to:目的地=物体旁/注视处/用户身边
            if cmd["to_user"] and not cmd["query"] and not cmd["deictic"]:
                if user_pos["xyz"] is None:
                    P.say("[×] 还不知道你的位置(视线流没送来定位)")
                    return
                propose("你这里", user_pos["xyz"], "导航", t_word, goto=True)
                return
            if cmd["deictic"]:
                cands = buf.candidates(t_word, args.lookback, cmd["noun"])
                if cands and cands[0].get("target_world"):
                    propose(f"{cands[0]['object']}那边", cands[0]["target_world"],
                            "导航", t_word, goto=True)
                    return
            if cmd["query"]:
                name, top = resolve_named(cmd["query"], table)
                if name:
                    propose(f"{name}那边", table[name], "导航", t_word, goto=True)
                    return
                P.say(f"[×] 「{cmd['query']}」没有唯一命中,最像的:"
                      + " / ".join(f"{n}({s:.2f})" for s, n in top))
                return
            P.say("[×] 说不清去哪:指个名字,或看一眼目的地再说")
            return
        if cmd["kind"] == "named":
            name, top = resolve_named(cmd["query"], table)
            if name is None and cmd.get("location"):
                loc, _ = resolve_named(cmd["location"], table)
                near = [n for s, n in top if s >= 0.55]
                if loc and near:  # 名字打平时按地点参照就近消歧
                    name = min(near, key=lambda n: sum(
                        (table[n][i] - table[loc][i]) ** 2 for i in range(3)))
                    P.say(f"[·] 按「{cmd['location']}」就近消歧 -> {name}")
            if name is None:
                P.say(f"[×] 「{cmd['query']}」没有唯一命中,最像的:"
                      + " / ".join(f"{n}({s:.2f})" for s, n in top))
                return
            propose(name, table[name], "名字", t_word)
            return
        # deictic / 类别泛指
        cands = buf.candidates(t_word, args.lookback, cmd["noun"])
        if not cands:
            if cmd["noun"]:
                name, top = resolve_named(cmd["noun"], table)
                if name:
                    P.say(f"[·] 近期没注视「{cmd['noun']}」,按名字兜底 -> {name}")
                    propose(name, table[name], "名字兜底", t_word)
                    return
                if top:
                    P.say(f"[×] 「{cmd['noun']}」类有多个且最近没注视——看一眼目标再说,或指名:"
                          + " / ".join(f"{n}({s:.2f})" for s, n in top))
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
                ow = e.get("origin_world")
                if ow:  # 背景注视也带头位姿:每条 verdict 都在更新用户位置
                    user_pos["xyz"], user_pos["t"] = [round(v, 3) for v in ow], t
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
