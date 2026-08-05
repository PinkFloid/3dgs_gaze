#!/usr/bin/env python3
"""brain.py -- Intension 层入口:指令 -> 双槽位消解 -> 确认 -> 派发。

    拿一下黄色机器人            object=名字:模糊匹配地图物体,不需要注视
    把这个杯子拿来              object=视线:眼-声窗口取近期注视物体,名词过滤类别
    去这个地方拿橘子            place=注视处落点(看着目标处的物体/家具说)+ 物名给到位检测
    把这个拿到物品台那边        dest=名字:送达目的地显式给出,不送用户
    过来 / 去凳子那边 / 去那边  goto:用户身边 / 名字 / 注视处落点
    停 / 停下 / 停止            急停旁路(永不过 LLM;标点/大小写归一,"停。"同样命中)
    (--proactive 4.8 加开第三模式:盯满主动问询;
     --proactive 3 --proactive-goto --yes = 看哪去哪,盯满即去注视物体旁;
     狗忙时最新盯视目标入补发槽,状态流报终态后自动补发——先看A再看B=先去A再去B)

模块分工:agent.py = 文本->槽位(LLM+缓存);core/attention = 物体通道(眼-声
绑定,E1 语义)+ 落点通道(place/dest 槽);core/resolve = 名字消解;
core/comms = 派发/状态订阅/事件源。本文件只做编排:三个槽位 resolver
(object/place/dest)+ 确认门 + 事件循环。
大脑不记忙闲、不排队、发完即忘:狗端忙时拒单(busy)就提示"本单作废",
要打断先「停」再重下;狗端将来升级抢占语义(archive/dog_link_preempt.py
是参考实现)时本文件零改动。

    python Intension/brain.py --skill-endpoint tcp://192.168.123.164:5583   # 真发狗机
    python Intension/brain.py                        # 缺省干跑:请求只打印不发送
    # 回放回归(确定性,只吃 parse_cache_v2):
    python Intension/brain.py --llm off --replay /tmp/fake.jsonl --yes \
        --skill-endpoint off --script "106.5:把这个杯子拿来"
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
from agent import CommandParser, load_openai_key, norm_cmd  # noqa: E402
from core.attention import (AttentionBuffer, PlaceBuffer,   # noqa: E402
                            accepted)
from core.comms import (Printer, dispatch, gaze_events,     # noqa: E402
                        replay_events, status_listener)
from core.resolve import load_object_table, resolve_named   # noqa: E402

# 词表匹配一律吃 norm_cmd(去两端标点+lower):语音转写的"好。""停。""Stop!"
# 才进得了门——尤其急停,带个句号就掉进 2.2s LLM 是不可接受的。
YES_WORDS = {"y", "yes", "是", "好", "嗯", "要", "ok", "行", "好的", "可以"}
NO_WORDS = {"n", "no", "不", "不用", "不要", "算了", "否", "取消"}
STOP_WORDS = {"停", "停下", "停止", "停一下", "stop", "s"}


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
    p.add_argument("--place-dwell", type=float, default=0.4,
                   help="落点通道入册的最短驻留 (s),防扫视噪声")
    p.add_argument("--confirm-timeout", type=float, default=10.0)
    p.add_argument("--proactive", type=float, default=0.0,
                   help=">0 时开启第三模式:盯满该秒数主动问询(如 4.8);默认关")
    p.add_argument("--proactive-goto", action="store_true",
                   help="主动模式只导航不抓取:盯满即去注视物体旁(看哪去哪)")
    p.add_argument("--suppress", type=float, default=30.0,
                   help="主动问询被拒后同目标静默期 (s)")
    p.add_argument("--llm", choices=["on", "off"], default="on",
                   help="off=只走 parse_cache_v2.json(离线/回归);on=缓存未命中时调 OpenAI(默认)")
    p.add_argument("--llm-model", default="gpt-5-mini",
                   help="OpenAI 解析模型;key 读 OPENAI_API_KEY 或 Intension/.openai_key")
    p.add_argument("--skill-endpoint", default=None,
                   help="狗机 REP 端点(如 tcp://192.168.123.164:5583);缺省=干跑只打印")
    p.add_argument("--status-endpoint", default=None)
    p.add_argument("--frame", default="board/v6")  # 2026-08-02 lab_colmap_v4 重建
    p.add_argument("--standoff", type=float, default=0.6,
                   help="[已废弃,仅兼容旧脚本] v2 起站位/避障由狗端自留,此参不再使用")
    p.add_argument("--detect-names", default=str(Path(__file__).resolve().parent
                                                 / "detect_names.json"),
                   help="地图名->检测器类名映射(狗端 /detect_grasp 只认英文类名)")
    p.add_argument("--replay", default=None)
    p.add_argument("--replay-pace", type=float, default=0.0,
                   help=">0 按流时间实速回放(如 1),可交互敲指令,行为对齐 live;"
                        "0=全速灌入(回归,指令走 --script)")
    p.add_argument("--script", action="append", default=[],
                   help="回归用脚本指令 '流时间:指令文本',可重复")
    p.add_argument("--voice", action="store_true",
                   help="开麦语音指令(faster-whisper CPU;与打字并行,同一条消解路径)")
    p.add_argument("--voice-model", default="small",
                   help="whisper 模型(small 够用;tiny 更快中文略差)")
    p.add_argument("--voice-device", default="Rx",
                   help="麦克风设备:序号或名字子串(voice_input.py --list 查;缺省=DJI 无线麦。"
                        "PortAudio 的'系统默认'是板载卡,故显式点名;没插会开麦报错,fail loud)")
    p.add_argument("--voice-rms", type=float, default=54,
                   help="语音能量闸:低于此 rms 的段当环境底噪丢弃(另有超长段/幻听闸,"
                        "见 voice_input.py 模块注释;默认按 DJI 麦 --meter 校准,换麦必重跑)")
    p.add_argument("--yes", action="store_true", help="自动确认(回归测试用)")
    p.add_argument("--log-dir", default=str(Path(__file__).resolve().parent / "logs"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.skill_endpoint in ("off", "none", "print", ""):
        args.skill_endpoint = None  # 显式干跑写法也认,与缺省等价
    sess = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
    sess.mkdir(parents=True, exist_ok=True)
    ev_f = open(sess / "events.jsonl", "w", encoding="utf-8")
    loglock = threading.Lock()

    def logev(rec):
        with loglock:
            ev_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ev_f.flush()

    logev({"topic": "session.args", **{k: v for k, v in vars(args).items()}})

    use_pt = False
    if not args.script:  # 交互输入走 prompt_toolkit:后台输出永远打在输入行上方
        try:
            import prompt_toolkit  # noqa: F401
            use_pt = sys.stdin.isatty()
        except ImportError:
            use_pt = False
    P = Printer(plain=use_pt)

    # 物体表热加载:names.json 是用户随分割命名反复改的活文件,按 mtime 跟进,
    # 原地更新 dict —— resolver 与 CommandParser 的提示词共享同一引用。
    table: dict = {}
    boxes: list = []  # [(名字, [minx,miny,maxx,maxy])] 逐实例占地,站位避障用
    tbl = {"sig": None, "t": 0.0}
    tbl_files = [Path(args.map_dir) / "instances.json", Path(args.map_dir) / "names.json"]

    def refresh_table(force=False):
        now = time.time()
        if not force and now - tbl["t"] < 1.0:
            return
        tbl["t"] = now
        try:
            sig = tuple(f.stat().st_mtime_ns for f in tbl_files)
        except OSError as e:
            if tbl["sig"] is None and force:
                P.say(f"[!] 物体表加载失败({e}),名字消解不可用")
            return
        if sig == tbl["sig"]:
            return
        first = tbl["sig"] is None
        try:
            new, nbox = load_object_table(args.map_dir, with_boxes=True)
        except Exception as e:
            P.say(f"[!] 物体表{'加载' if first else '重载'}失败({e})"
                  + (",沿用旧表" if not first else ",名字消解不可用"))
            tbl["sig"] = sig
            return
        tbl["sig"] = sig
        table.clear()
        table.update(new)
        boxes[:] = nbox
        P.say(f"{'物体表' if first else '[·] 物体表已刷新'}:{len(table)} 个命名物体"
              + ("(名字消解可用)" if first else ""))

    refresh_table(force=True)

    if args.proactive_goto and args.proactive <= 0:
        P.say("[!] --proactive-goto 只是修饰符,触发器是 --proactive <秒>(如 6)——"
              "当前没给,盯视主动模式是关的,盯多久都不会出发")
    buf = AttentionBuffer(args.merge_gap, args.proactive)   # 物体通道(E1 语义勿动)
    place_buf = PlaceBuffer(min_dwell=args.place_dwell)     # 落点通道:物体表面落点
    suppress = {}  # 目标名 -> 流时间,主动问询被拒后的静默截止
    hinted = set()  # 已提示过"未命名"的实例:盯满只提醒一次,不刷屏

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
        if obj.isascii():  # 英文命名直接小写直发:检测器类名惯例是小写(Bottle->bottle)
            return obj.lower()
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

    # 状态订阅纯粹是显示 + 回放收尾等终态;抢占语义下大脑不做任何忙闲记账
    status_seen = None
    if args.skill_endpoint and args.status_endpoint != "off":
        sep = args.status_endpoint or args.skill_endpoint.rsplit(":", 1)[0] + ":5584"
        status_seen = {}
        threading.Thread(target=status_listener, args=(sep, P, status_seen, logev),
                         daemon=True).start()

    last_req = {"id": None}  # 最近已接受请求:回放收尾 + 盯视补发的空闲判据
    # 盯视补发单槽:主动模式派发被拒 busy 时暂存最新一单,状态流报终态后补发。
    # 不是忙闲记账——不看表不设看门狗,纯反应式:被拒才存,终态才发,45s 放弃。
    parked = {"req": None, "name": None, "deadline": 0.0, "next_try": 0.0}
    # 上一单派发的终点(deliver_to 优先,其次 target_world)= 狗当前位置的最佳已知值。
    # 「抓XX」原地抓取用它当 target:狗端 Move 段零长度,等效纯 Pick。
    # 状态流有 pose 后(待对齐)换真实位姿,此处即删。
    last_stand = {"tw": None}
    last_cmd = {"text": None}  # 最近成功解析的原文:确认(y/--yes)后 parser.confirm 落盘

    cmd_q = queue.Queue()  # 条目 = (文本, 说完时刻wall|None);None 条目=退出哨兵
    if not args.script:
        if use_pt:
            def stdin_reader():  # 输入行常驻底部,后台 say/进度全部打在上方
                from prompt_toolkit import PromptSession
                from prompt_toolkit.patch_stdout import patch_stdout
                ps = PromptSession()
                with patch_stdout(raw=True):
                    while True:
                        try:
                            cmd_q.put((ps.prompt("指令> "), None))
                        except (EOFError, KeyboardInterrupt):
                            cmd_q.put(None)  # 哨兵:主循环转成正常退出
                            return
        else:
            def stdin_reader():
                for line in sys.stdin:
                    cmd_q.put((line.rstrip("\n"), None))
                # 管道 EOF 不塞哨兵:stdin 关闭(如重定向跑回放)不该杀事件循环
        threading.Thread(target=stdin_reader, daemon=True).start()
    notify = lambda kind: None  # 提示音钩子:--voice 时换成 voice_input.chime  # noqa: E731
    if args.voice:  # 语音与打字并行:同一队列同一 handle,t_word=说完时刻回填
        from voice_input import VoiceReader, chime
        notify = chime
        _say = P.say  # 失败/取消行自动低音:语音场景用户看不到终端,
        # "说了没反应"最像死机;成功(ok)/等确认(ask)在 send/propose 显式发

        def _say_chimed(msg):
            _say(msg)
            if msg.startswith(("[×]", "[-]", "[狗] failed", "[狗] stopped")):
                chime("fail")
            elif msg.startswith("[狗] done"):
                chime("ok")  # 狗跑完终态也报音:不看屏幕知道干完没
        P.say = _say_chimed

        def _on_voice(text, t_end, asr_s):
            P.say(f"[语音] 「{text}」 (asr {asr_s:.2f}s)")
            logev({"topic": "asr", "text": text, "t_end_wall": t_end, "asr_s": asr_s})
            cmd_q.put((text, t_end))

        dev = args.voice_device
        if dev is not None and dev.isdigit():
            dev = int(dev)
        vr = VoiceReader(_on_voice, model=args.voice_model, vocab=table.keys(),
                         say=P.say, device=dev, min_rms=args.voice_rms,
                         dump_dir=sess / "utt")  # 每条语音段存 WAV(Pupil 不录音)
        threading.Thread(target=vr.run, daemon=True).start()
    scripted = sorted((float(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in args.script)

    n_req = 0
    pending = None  # {"req":..., "since": stream_t, "mode":..., "object":...}
    last_prog = None
    clock = {"stream": 0.0, "wall": time.time()}
    user_pos = {"xyz": None, "t": 0.0}  # 每条 verdict 的 origin_world = 用户头部位置

    def stream_now():
        return clock["stream"] + (time.time() - clock["wall"])

    def send(req, retry_name=None):
        """retry_name 非空 = 盯视(主动)单:被拒 busy 进补发槽而不是作废。
        任何一单被接受都清空补发槽——新意图覆盖旧盯视目标。"""
        rep = dispatch(req, args.skill_endpoint)
        logev({"topic": "skill.req", **req, "rep": rep})
        P.say(f"[派发] {json.dumps(req, ensure_ascii=False)}")
        P.say(f"       -> {json.dumps(rep, ensure_ascii=False)}")
        notify("ok" if rep.get("accepted") else "fail")
        if rep.get("accepted"):
            parked["req"] = None
            if req.get("skill") != "stop":
                last_req["id"] = req["req_id"]
                p = req.get("params") or {}
                end = p.get("deliver_to") or p.get("target_world")
                if end:
                    last_stand["tw"] = list(end)
        elif rep.get("reason") == "busy":
            if retry_name and status_seen is not None:  # 没状态流就无从判空闲,不存
                fresh = parked["req"] is None or req["req_id"] != parked.get("rid")
                parked.update(req=req, rid=req["req_id"], name=retry_name,
                              next_try=time.time() + 2.0)
                if fresh:  # 首次入槽才刷新存活期;补发重试沿用原 deadline
                    parked["deadline"] = time.time() + 45.0
                    P.say(f"[·] 狗在忙,已排队最新盯视目标:{retry_name}(空闲后自动补发)")
            else:  # 打字指令维持作废语义:发完即忘,打断权在「停」
                P.say("[!] 狗在忙,本单作废——要打断先「停」,再重下指令")

    def send_stop():
        send({"v": 1, "type": "skill.request", "skill": "stop",
              "req_id": f"{sess.name}-stop", "frame": args.frame,
              "sent_at": time.time(), "params": {}})

    def aim(goal, approach_from=None):
        """v2 线格式:狗端自算站位与避障,brain 只给 目标本体 (x,y) + 建议接近
        方位 yaw(从参考侧指向目标,板系弧度;狗端可据此选站位侧,也可忽略)。
        参考侧 = approach_from(缺省=用户位置,再缺省=原点)——"站到你视野里"
        的语义以建议的形式过线,不再由 brain 硬算站位。"""
        src = approach_from or user_pos["xyz"] or (0.0, 0.0, 0.0)
        ang = math.atan2(goal[1] - src[1], goal[0] - src[0])
        return [round(goal[0], 3), round(goal[1], 3)], round(ang, 3)

    def propose(obj, tw, mode, t_word, goto=False, approach_from=None, dest=None,
                via=None, raw_pose=False, obj_point=None):
        """唯一技能 grasp:goto=True 时 object_name 置空 = 纯导航(冻结定义)。
        dest = (落点, 标签) 或 None:送达站位从物体那侧接近,停在落点前
        standoff、yaw 朝向落点(dest=用户位置时即"送回你这")。无 dest = 原地 done。
        via = 地点标签(地点/视线地点单):tw 是地点的站位而非物体位置,
        确认行写明"去「地点」拿「物」",免得坐标被读成物体在哪。"""
        nonlocal n_req, pending
        n_req += 1
        if raw_pose:  # 原地抓:重发上一单目标三元组——狗端站位计算幂等=原地不动
            stand, yaw = [round(tw[0], 3), round(tw[1], 3)], round(float(tw[2]), 3)
        else:
            stand, yaw = aim(tw, approach_from)
        params = {"object_name": None if goto else detect_name(obj),
                  "target_world": [stand[0], stand[1], yaw]}  # v2:目标本体+建议方位
        if not goto and obj_point is not None:
            # 实例消歧过线(坑2):选中实例的板系质心。狗端检测出多只同类时取离
            # hint 最近的,>0.3m 拒抓(hint_mismatch)。加字段=兼容,旧狗端自动忽略。
            params["object_hint"] = [round(float(obj_point[0]), 3),
                                     round(float(obj_point[1]), 3),
                                     round(float(obj_point[2]) if len(obj_point) > 2 else 0.0, 3)]
        if not goto and dest is not None:
            dxy, dyaw = aim(dest[0], approach_from=tw)  # 送达落点+朝向建议(自物体侧)
            params["deliver_to"] = [dxy[0], dxy[1], dyaw]
        req = {"v": 1, "type": "skill.request", "skill": "grasp",
               # 2026-08-05 起线上语义=目标本体+建议方位(站位狗端自留);版本号按用户
               # 裁定维持 1,双方靠约定同步——切换日后旧语义 server 不可再接单
               "params": params,
               "req_id": f"{sess.name}-{n_req:03d}", "frame": args.frame,
               "sent_at": time.time(), "t_stream": round(t_word, 3),
               "intent_summary": f"指令({mode}){'导航至' if goto else '消解为'} {obj}"}
        logev({"topic": "resolution", "mode": mode, "object": obj, "goto": goto,
               "t": t_word, "goal": list(tw), "stand": stand, "yaw": yaw, "wire_v": 1,
               "dest": list(dest[0]) if dest else None,
               "hint": params.get("object_hint")})
        src_text = None if mode == "主动" else last_cmd["text"]  # 主动单没有指令原文
        if args.yes:
            send(req, retry_name=obj if mode == "主动" else None)
            if src_text:
                parser.confirm(src_text)
        else:
            pending = {"req": req, "since": t_word, "mode": mode, "object": obj,
                       "text": src_text}
            pose_txt = (f"目标({stand[0]:+.2f},{stand[1]:+.2f}) "
                        f"接近方位{math.degrees(yaw):+.0f}°(站位狗端定)")
            if not goto:
                pose_txt += f" 送到{dest[1]}" if dest else "(不送)"
            ask = (f"[?] 你在看「{obj}」——{'过去吗' if goto else '要我拿来吗'}?" if mode == "主动"
                   else f"[?] 过去「{obj}」{pose_txt}"
                        + (f"(从「{via}」侧接近)" if via else "") + " ?" if goto
                   else f"[?] 原地抓「{obj}」(狗不挪窝)?" if raw_pose
                   else f"[?] 去「{via}」拿「{obj}」{pose_txt} ?" if via
                   else f"[?] 去拿「{obj}」{pose_txt} ?")
            P.say(ask + " y=确认 其他=取消")
            notify("ask")  # 升调:该说"好/不"了(语音确认走同一 handle)

    def proactive_fire(pl):
        """盯满触发的公共闸:命名/pending/抑制期。狗忙不设闸——直发,被拒 busy
        时最新目标进补发槽(send 的 retry_name 路径),空闲后自动补发。"""
        if pl["object"] not in table:  # 只去有名字的物体旁
            logev({"topic": "proactive.skipped", "object": pl["object"], "why": "unnamed"})
            if pl["object"] not in hinted:
                hinted.add(pl["object"])
                P.say(f"[·] 你盯的「{pl['object']}」还没命名——names.json 起名后才可用(热加载)")
        elif pending is not None:
            logev({"topic": "proactive.skipped", "object": pl["object"], "why": "pending"})
        elif pl["t"] < suppress.get(pl["object"], float("-inf")):
            P.say(f"[×] {pl['object']} 抑制期,略过主动问询")
        elif pl.get("target_world"):
            propose(pl["object"], pl["target_world"], "主动", pl["t"],
                    goto=args.proactive_goto)

    def place_label(r):
        """落点的人话标签:命名物体用名字,未命名实例报坐标。"""
        o = r.get("object") or ""
        if not o or o.startswith("object#"):
            p = r["point"]
            return f"位置({p[0]:+.1f},{p[1]:+.1f})"
        return o

    def slot_point(deictic, query, t_word, role):
        """place/dest 槽公共消解。指代词在场 = 视线优先(说"这里"时看哪就是哪),
        名字只做兜底;纯名字则只按名字。
        返回 (状态, 落点, 标签):状态 ok=拿到 / fail=给了但消解不了 / none=没给。"""
        if deictic:
            r = place_buf.latest(t_word, args.lookback)
            if r:
                return "ok", r["point"], place_label(r)
        if query:
            name, top = resolve_named(query, table)
            if name:
                return "ok", table[name], name
            P.say(f"[×] {role}「{query}」没有唯一命中,最像的:"
                  + " / ".join(f"{n}({s:.2f})" for s, n in top))
            return "fail", None, None
        if deictic:
            P.say(f"[×] 最近 {args.lookback:.0f}s 没注视到{role}——看一眼再说"
                  f"(先看后说,回看窗只朝前看 {args.lookback:.0f}s)")
            return "fail", None, None
        return "none", None, None

    def handle(t_word, text):
        nonlocal pending
        t = "".join(text.split())
        key = norm_cmd(t)  # 词表匹配用归一形:"好。""停。"(语音标点)才进得了门
        if not key:
            return  # 空行 / 纯标点(转写噪声)
        refresh_table(force=True)  # 名字消解前跟一次 names.json,拿最新命名
        was_decline = False
        if pending is not None:
            prev, pending = pending, None
            if key in YES_WORDS:
                send(prev["req"],
                     retry_name=prev["object"] if prev["mode"] == "主动" else None)
                if prev.get("text"):  # 点头 = 解析经过人工校验 -> 此刻才落盘
                    parser.confirm(prev["text"])
                if prev["mode"] == "主动":  # 执行完还盯着,也别立刻再问
                    suppress[prev["object"]] = t_word + args.suppress
                return
            if prev["mode"] == "主动":  # 拒绝主动提议 -> 抑制该目标,免得追着问
                suppress[prev["object"]] = t_word + args.suppress
            P.say("[-] 已取消")
            if key in NO_WORDS:
                return
            was_decline = True  # 可能是"取消旧的换新指令":往下试,解析不出就保持安静
        if key in STOP_WORDS:  # 急停硬旁路:永不过 LLM
            send_stop()
            return
        cmd = parser.parse(t)
        if cmd is not None:
            last_cmd["text"] = t  # 供 propose 记进 pending:确认后按原文落盘
        # kind 词表与旧版对齐:E1 打分认 kind=="deictic"(object 槽走视线的 trial)
        obj_gaze = (cmd is not None and cmd["kind"] in ("fetch", "grab")
                    and (cmd["object_deictic"] or (not cmd["object"] and cmd["noun"])))
        kind = ("parse_fail" if cmd is None else
                "grab" if cmd["kind"] == "grab" else  # 不混进 deictic:E1 只数 fetch 类
                "deictic" if obj_gaze else
                "named" if cmd["kind"] == "fetch" and cmd["object"] else
                cmd["kind"])
        logev({"topic": "command", "text": text, "t": t_word, "kind": kind})
        if cmd is None:
            if not was_decline:
                P.say("[×] 解析不可用(缓存未命中且 LLM 不可达)")
            return
        if cmd["kind"] == "stop":
            send_stop()
            return
        if cmd["kind"] == "help":
            if not was_decline:
                P.say("我能做:拿取场景里的物体。例:拿一下显示器 / 把这个杯子拿来"
                      " / 去这个地方拿橘子 / 停")
            return
        def user_pos_stale():
            """定位新鲜度:>10s 没有带头位姿的视线事件 = 你可能已走动(定位空窗)。
            幽灵位置比没有位置更糟——狗会认真地走去你不在的地方。"""
            age = t_word - user_pos["t"]
            if user_pos["xyz"] is not None and age > 10.0:
                P.say(f"[!] 你的定位是 {age:.0f}s 前的——中途走动过的话,先扫一眼 tag 再说")

        if cmd["kind"] == "goto":  # 目的地 = 用户身边 / 名字 / 视线落点
            if cmd["to_user"] and not cmd["place"] and not cmd["place_deictic"]:
                if user_pos["xyz"] is None:
                    P.say("[×] 还不知道你的位置(视线流没送来定位)")
                    return
                user_pos_stale()
                # 从最近注视处那一侧接近:站到用户视野里,yaw 朝向用户。
                # 这同时是控制权:回车前盯稳(≥0.5s)哪一侧,狗就停哪一侧
                r = place_buf.latest(t_word, 30.0)
                propose("你这里", user_pos["xyz"], "导航", t_word, goto=True,
                        approach_from=r["point"] if r else None,
                        via=place_label(r) if r else None)
                return
            st, pt, label = slot_point(cmd["place_deictic"], cmd["place"], t_word, "目的地")
            if st == "ok":
                disp = label if label.startswith("位置(") else f"{label}那边"
                propose(disp, pt, "导航", t_word, goto=True)
            elif st == "none":
                P.say("[×] 说不清去哪:指个名字,或看一眼目的地再说")
            return
        if cmd["kind"] == "grab":  # 原地抓:不算站位不导航,target=上一单终点(狗就在那)
            if last_stand["tw"] is None:
                P.say("[×] 不知道狗现在站哪(本会话还没派发过)——先「去XX」把它派过去,"
                      "或说完整的「拿XX」")
                return
            tw = last_stand["tw"]
            if obj_gaze:
                cands = [c for c in buf.candidates(t_word, args.lookback, cmd["noun"])
                         if c["object"] in table]
                if not cands:
                    P.say(f"[×] 最近 {args.lookback:.0f}s 没有可用注视目标——"
                          "看一眼要抓的再说,或指名「抓orange」")
                    return
                logev({"topic": "binding", "t_word": t_word, "noun": cmd["noun"],
                       "candidates": cands[:3]})
                propose(cands[0]["object"], tw, "原地抓", t_word, raw_pose=True,
                        obj_point=cands[0]["target_world"])
            elif cmd["object"]:  # 名字透传检测器;但名字在表里时把质心当 hint 带上
                nm, _ = resolve_named(cmd["object"], table)  # (同类多实例的选框锚)
                propose(nm or cmd["object"], tw, "原地抓", t_word, raw_pose=True,
                        obj_point=table.get(nm))
            else:
                P.say("[×] 抓什么?指个名字(抓orange)或看它一眼说「抓这个」")
            return
        # ---- fetch:先消解 dest(送到哪),再 place(去哪拿),最后 object(拿什么)
        dest = None
        if cmd["dest"] or cmd["dest_deictic"]:
            st, pt, label = slot_point(cmd["dest_deictic"], cmd["dest"], t_word, "送达地")
            if st != "ok":
                return  # 显式说了送哪却消解不了:宁可不动
            dest = (pt, label)
        elif cmd["to_user"]:
            if user_pos["xyz"] is not None:  # 确认时刻的用户位置 = 送回站位的锚
                user_pos_stale()
                dest = (user_pos["xyz"], "你这")
            else:  # 拿取但不知用户在哪:降级可用,但必须说出来,不许静默丢送回
                P.say("[!] 不知道你在哪(视线流未定位)——本单不带送回,接近方向退化为原点侧")
        place = None
        if cmd["place"] or cmd["place_deictic"]:
            st, pt, label = slot_point(cmd["place_deictic"], cmd["place"], t_word, "取货地点")
            if st == "ok":
                place = (pt, label, cmd["place_deictic"])
            elif cmd["place_deictic"]:
                return  # 指了"这里"却没有注视落点:没得兜底
            else:
                P.say(f"[!] 地点「{cmd['place']}」没认出,退回按物名的地图位置")
        if obj_gaze:  # object=视线:眼-声窗口绑定(E1 语义)
            cands = [c for c in buf.candidates(t_word, args.lookback, cmd["noun"])
                     if c["object"] in table]  # 只有命名物体可被指代:碎片不进绑定,
            # 免得"拿这个"派发 object#38 这种检测器不认的名字(碎片仍参与投票与落点)
            if cands:
                c = cands[0]
                logev({"topic": "binding", "t_word": t_word, "noun": cmd["noun"],
                       "candidates": cands[:3]})
                propose(c["object"], place[0] if place else c["target_world"],
                        "视线", t_word, dest=dest, via=place[1] if place else None,
                        obj_point=c["target_world"])
                return
            if cmd["noun"]:
                name, top = resolve_named(cmd["noun"], table)
                if name:  # 地点槽已解出时仍去注视处检测:LLM 错标 deictic 也能落对地方
                    P.say(f"[·] 近期没注视「{cmd['noun']}」,按名字兜底 -> {name}"
                          + (f"(仍去注视处「{place[1]}」检测)" if place else ""))
                    propose(name, place[0] if place else table[name],
                            "名字兜底", t_word, dest=dest,
                            via=place[1] if place else None, obj_point=table[name])
                    return
                if top:
                    P.say(f"[×] 「{cmd['noun']}」类有多个且最近没注视——看一眼目标再说,或指名:"
                          + " / ".join(f"{n}({s:.2f})" for s, n in top))
                    return
            P.say(f"[×] 最近 {args.lookback:.0f}s 没有可用注视目标"
                  + (f"(类别「{cmd['noun']}」)" if cmd["noun"] else ""))
            return
        if cmd["object"]:
            if place:  # 地点优先:物品会被挪动/可能不在图里,到位后按名检测
                P.say(f"[·] 按{'注视处' if place[2] else '地点'}「{place[1]}」导航,"
                      f"到位后检测「{cmd['object']}」")
                propose(cmd["object"], place[0], "视线地点" if place[2] else "地点",
                        t_word, dest=dest, via=place[1])
                return
            name, top = resolve_named(cmd["object"], table)
            if name is None:
                P.say(f"[×] 「{cmd['object']}」没有唯一命中,最像的:"
                      + " / ".join(f"{n}({s:.2f})" for s, n in top))
                return
            propose(name, table[name], "名字", t_word, dest=dest,
                    obj_point=table[name])
            return
        P.say("我能做:拿取场景里的物体。例:拿一下显示器 / 把这个杯子拿来 / 停")

    # live 会话把收到的每条 gaze.intent 原样落盘:任何一次 live 都能事后用
    # --replay logs/<sess>/gaze.jsonl --script "<t>:<指令>" 逐位复现(t 抄 events.jsonl)
    gz_f = None if args.replay else open(sess / "gaze.jsonl", "w", encoding="utf-8")
    source = (replay_events(args.replay, args.replay_pace) if args.replay
              else gaze_events(args.sub))
    P.say(f"指令入口就绪:拿一下<名字> / 把这个<类别>拿来 / 去这个地方拿<名字> / 过来 / 停"
          f"  (回看窗 {args.lookback:.0f}s,日志 {sess})")
    try:
        for e in source:
            if e is not None:
                if gz_f is not None:
                    gz_f.write(json.dumps(e, ensure_ascii=False) + "\n")
                t = float(e.get("t_end", e.get("t_start", 0.0)))
                if t < clock["stream"] - 5.0:
                    # 流时间大幅倒退 = 新的一遍回放接进了同一个 brain:旧注意状态
                    # 会把重放的注视当成"已触发过的延续"而整段静默——重置自愈。
                    # (阈值 5s:乱序 final 最多晚 1-2s,不许误伤)
                    P.say(f"[·] 流时间倒退 {clock['stream'] - t:.0f}s(新流/回放接入)"
                          "——注意缓冲、排队与确认已重置")
                    buf = AttentionBuffer(args.merge_gap, args.proactive)
                    place_buf = PlaceBuffer(min_dwell=args.place_dwell)
                    suppress.clear()
                    parked["req"] = None
                    pending, last_prog = None, None
                clock["stream"], clock["wall"] = t, time.time()
                ow = e.get("origin_world")
                if ow:  # 背景注视也带头位姿:每条 verdict 都在更新用户位置
                    user_pos["xyz"], user_pos["t"] = [round(v, 3) for v in ow], t
                place_buf.feed(e)  # 落点通道:物体表面落点记账(place/dest 槽用)
                if accepted(e, args.min_vote):
                    for kind, pl in buf.feed(e):
                        if kind == "progress":
                            if pl["object"] not in table:
                                continue  # 未命名碎片不上进度条(用户裁定:眼不见为净)
                            key = (pl["object"], int(pl["dwell_s"]))
                            if key == last_prog:  # 1Hz/换目标才刷,少打扰输入行
                                continue
                            last_prog = key
                            P.progress(f"看 {pl['object']:<14} {pl['dwell_s']:4.1f}s"
                                       f"  vote {pl['share']:3.0%}")
                        elif kind == "sustained" and args.proactive > 0:
                            proactive_fire(pl)
                else:
                    buf.advance(t)
            st = stream_now()
            refresh_table()
            while scripted and st >= scripted[0][0]:
                _, text = scripted.pop(0)
                P.say(f"[指令] {text}")
                handle(st, text)
            while not cmd_q.empty():
                item = cmd_q.get()
                if item is None:  # 输入端收摊(Ctrl-C/Ctrl-D)-> 正常退出
                    raise KeyboardInterrupt
                text, t_said = item
                # 语音单 t_word 回填到说完时刻:VAD 尾判+转写延迟不歪眼-声回看窗
                tw = st if t_said is None else max(0.0, st - max(0.0, time.time() - t_said))
                handle(tw, text)
            if parked["req"] is not None:  # 盯视补发:终态即空闲,晚到 running 被粘滞挡住
                now = time.time()
                if now > parked["deadline"]:
                    P.say(f"[·] 排队目标「{parked['name']}」等不到空闲,放弃补发")
                    parked["req"] = None
                elif now >= parked["next_try"] and \
                        (last_req["id"] is None or status_seen.get(last_req["id"])
                         in ("done", "failed", "stopped")):
                    rq, name = parked["req"], parked["name"]
                    parked["req"] = None  # 先摘再发:再被拒会按原 deadline 重新入槽
                    P.say(f"[·] 狗空闲,补发盯视目标:{name}")
                    rq["sent_at"] = time.time()
                    send(rq, retry_name=name)
            if pending and st - pending["since"] > args.confirm_timeout:
                P.say("[-] 确认超时,已取消")
                if pending["mode"] == "主动":
                    suppress[pending["object"]] = st + args.suppress
                pending = None
            if args.replay and not scripted and pending is None \
                    and parked["req"] is None and e is None:
                rid = last_req["id"]  # 等最后一个已接受请求到终态,而不是"暂时没状态"就走
                if status_seen is None or rid is None or \
                        status_seen.get(rid) in ("done", "failed", "stopped"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        P.say(f"结束。日志:{sess}")
        ev_f.close()
        if gz_f is not None:
            gz_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
