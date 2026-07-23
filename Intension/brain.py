#!/usr/bin/env python3
"""brain.py -- Intension 层入口:指令 -> 指代消解 -> 确认 -> 派发。

    拿一下黄色机器人            名字消解:模糊匹配地图物体名,不需要注视
    把这个杯子拿来              视线消解:眼-声窗口取近期注视目标,名词过滤类别
    过来 / 去凳子那边           goto:空 object 的 grasp = 纯导航
    停                          急停旁路(永不过 LLM)
    (--proactive 4.8 加开第三模式:盯满主动问询)

模块分工:agent.py = 文本->结构(LLM+缓存);core/attention = 层A与注意缓冲;
core/resolve = 名字消解;core/comms = 派发/状态订阅/事件源。本文件只做编排:
状态(pending/suppress/user_pos)+ 消解分支 + 确认门 + 事件循环。

    python Intension/brain.py [--skill-endpoint tcp://狗机:5583]
    # 回放回归(确定性,只吃 parse_cache):
    python Intension/brain.py --llm off --replay /tmp/fake.jsonl --yes \
        --script "106.5:把这个杯子拿来"
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import CommandParser, load_openai_key            # noqa: E402
from core.attention import AttentionBuffer, accepted        # noqa: E402
from core.comms import (Printer, dispatch, gaze_events,     # noqa: E402
                        replay_events, status_listener)
from core.resolve import load_object_table, resolve_named   # noqa: E402

YES_WORDS = {"y", "yes", "是", "好", "嗯", "要", "ok", "行"}
NO_WORDS = {"n", "no", "不", "不用", "不要", "算了", "否", "取消"}
STOP_WORDS = {"停", "stop", "s"}


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
    p.add_argument("--standoff", type=float, default=0.6,
                   help="站位距离:target 发在目标前多少米(狗端自留 standoff 时设 0)")
    p.add_argument("--detect-names", default=str(Path(__file__).resolve().parent
                                                 / "detect_names.json"),
                   help="地图名->检测器类名映射(狗端 /detect_grasp 只认英文类名)")
    p.add_argument("--replay", default=None)
    p.add_argument("--script", action="append", default=[],
                   help="回归用脚本指令 '流时间:指令文本',可重复")
    p.add_argument("--yes", action="store_true", help="自动确认(回归测试用)")
    p.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    return p.parse_args()


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

    try:  # 狗端检测器只认类名(如 orange):发送前把地图名翻译过去
        dmap = json.loads(Path(args.detect_names).read_text(encoding="utf-8"))
    except Exception:
        dmap = {}
        P.say(f"[!] 检测名映射不可用({args.detect_names}),object_name 将按地图原名直发")
    unmapped = set()

    def detect_name(obj):
        if obj in dmap:
            return dmap[obj]
        hit = max((k for k in dmap if k in obj), key=len, default=None)  # 杯A -> 杯 -> cup
        if hit:
            return dmap[hit]
        if dmap and obj not in unmapped:
            unmapped.add(obj)
            P.say(f"[!] 「{obj}」不在 detect_names.json,按原名直发(狗端检测器可能不认)")
        return obj

    key = load_openai_key()
    if args.llm == "on" and not key:
        P.say("[!] 未配置 OPENAI_API_KEY(环境变量或 Intension/.openai_key),只用解析缓存")
        args.llm = "off"
    parser = CommandParser(table, model=args.llm_model, mode=args.llm, key=key,
                           say=P.say, logev=logev)

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
    pending = None  # {"req":..., "since": stream_t, "mode":...}
    clock = {"stream": 0.0, "wall": time.time()}
    user_pos = {"xyz": None, "t": 0.0}  # 每条 verdict 的 origin_world = 用户头部位置

    def stream_now():
        return clock["stream"] + (time.time() - clock["wall"])

    def send(req):
        rep = dispatch(req, args.skill_endpoint)
        logev({"topic": "skill.req", **req, "rep": rep})
        P.say(f"[派发] {json.dumps(req, ensure_ascii=False)}")
        P.say(f"       -> {json.dumps(rep, ensure_ascii=False)}")
        if rep.get("accepted") and req.get("skill") != "stop":
            last_req["id"] = req["req_id"]

    def send_stop():
        send({"v": 1, "type": "skill.request", "skill": "stop",
              "req_id": f"{sess.name}-stop", "frame": args.frame,
              "sent_at": time.time(), "params": {}})

    def stand_pose(goal, approach_from=None):
        """狗基座站位:沿 approach_from(缺省=用户位置,再缺省=原点)→goal 方向,
        在 goal 前 --standoff 米停;yaw 指向 goal(弧度,板系 +x=0,逆时针正)。"""
        src = approach_from or user_pos["xyz"] or (0.0, 0.0, 0.0)
        vx, vy = goal[0] - src[0], goal[1] - src[1]
        n = math.hypot(vx, vy)
        if n < 1e-6:
            vx, vy, n = 1.0, 0.0, 1.0
        d = min(args.standoff, n)  # 出发侧离目标太近时,别退越过出发点
        return ([round(goal[0] - vx / n * d, 3), round(goal[1] - vy / n * d, 3),
                 round(goal[2], 3)], round(math.atan2(vy, vx), 3))

    def propose(obj, tw, mode, t_word, goto=False, approach_from=None):
        """唯一技能 grasp:goto=True 时 object_name 置空 = 纯导航(冻结定义)。
        狗端导航吃 (x,y)+yaw、高度自调:target_world = 站位,z 仅是目标高度参考。"""
        nonlocal n_req, pending
        n_req += 1
        stand, yaw = stand_pose(tw, approach_from)
        params = {"object_name": None if goto else detect_name(obj),
                  "target_world": stand, "yaw": yaw}
        if not goto and user_pos["xyz"] is not None:  # 确认时刻的用户位置:带它=送达
            params["deliver_to"] = user_pos["xyz"]
        req = {"v": 1, "type": "skill.request", "skill": "grasp",
               "params": params,
               "req_id": f"{sess.name}-{n_req:03d}", "frame": args.frame,
               "sent_at": time.time(), "t_stream": round(t_word, 3),
               "intent_summary": f"指令({mode}){'导航至' if goto else '消解为'} {obj}"}
        logev({"topic": "resolution", "mode": mode, "object": obj, "goto": goto,
               "t": t_word, "goal": tw, "stand": stand, "yaw": yaw})
        if args.yes:
            send(req)
        else:
            pending = {"req": req, "since": t_word, "mode": mode, "object": obj}
            pose_txt = f"站位({stand[0]:+.2f},{stand[1]:+.2f}) 朝向{math.degrees(yaw):+.0f}°"
            ask = (f"[?] 你在看「{obj}」——要我拿来吗?" if mode == "主动"
                   else f"[?] 过去「{obj}」{pose_txt} ?" if goto
                   else f"[?] 去拿「{obj}」{pose_txt} ?")
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
                    suppress[prev["object"]] = t_word + args.suppress
                return
            if prev["mode"] == "主动":  # 拒绝主动提议 -> 抑制该物体,免得追着问
                suppress[prev["object"]] = t_word + args.suppress
            P.say("[-] 已取消")
            if t.lower() in NO_WORDS:
                return
            was_decline = True  # 可能是"取消旧的换新指令":往下试,解析不出就保持安静
        if t.lower() in STOP_WORDS:  # 急停硬旁路:永不过 LLM
            send_stop()
            return
        cmd = parser.parse(t)
        logev({"topic": "command", "text": text, "t": t_word,
               "kind": cmd["kind"] if cmd else "parse_fail"})
        if cmd is None:
            if not was_decline:
                P.say("[×] 解析不可用(缓存未命中且 LLM 不可达)")
            return
        if cmd["kind"] == "stop":
            send_stop()
            return
        if cmd["kind"] == "help":
            if not was_decline:
                P.say("我能做:拿取场景里的物体。例:拿一下显示器 / 把这个杯子拿来 / 停")
            return
        if cmd["kind"] == "goto":  # 目的地 = 用户身边 / 注视处 / 名字
            if cmd["to_user"] and not cmd["query"] and not cmd["deictic"]:
                if user_pos["xyz"] is None:
                    P.say("[×] 还不知道你的位置(视线流没送来定位)")
                    return
                # 从最近注视处那一侧接近:站到用户视野里,yaw 朝向用户
                ref = (buf.visit["last"].get("object_centroid_world")
                       if buf.visit is not None
                       else buf.recent[-1].get("target_world") if buf.recent else None)
                propose("你这里", user_pos["xyz"], "导航", t_word, goto=True,
                        approach_from=ref)
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

    source = replay_events(args.replay) if args.replay else gaze_events(args.sub)
    P.say(f"指令入口就绪:拿一下<名字> / 把这个<类别>拿来 / 过来 / 停"
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
                    suppress[pending["object"]] = st + args.suppress
                pending = None
            if args.replay and not scripted and pending is None and e is None:
                rid = last_req["id"]  # 等最后一个已接受请求到终态,而不是"暂时没状态"就走
                if status_seen is None or rid is None or \
                        status_seen.get(rid) in ("done", "failed", "stopped"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        P.say(f"结束。日志:{sess}")
        ev_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
