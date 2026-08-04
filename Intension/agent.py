"""agent.py -- 指令文本 -> 结构化槽位(LLM + 解析缓存)。

只做"文本->槽位"这一件事;绑定/几何/确认永远是外面的确定性代码。
槽位模型(v2,双通道消解的输入):object(拿什么)/ place(去哪拿)/
dest(送到哪),每个槽位独立地要么给名字(*_query),要么标指代(*_deictic
——指什么由视线决定,LLM 永远不猜)。
缓存命中 0ms 且完全确定(parse_cache_v2.json;schema 变更即换文件名,
旧缓存不许毒化新解析);未命中且 mode=='on' 时直连 OpenAI(strict
json_schema,reasoning minimal);都不行返回 None。
落盘时机 = 用户确认后(confirm()):确认门就是人工校验,新解析先只进
内存(本进程内复用),没确认的误判活不过本次进程,不毒化缓存文件。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

_DIR = Path(__file__).resolve().parent
PARSE_SCHEMA = json.loads((_DIR / "parse_schema.json").read_text(encoding="utf-8"))

# 槽位卫生:LLM 偶尔把指代词本身塞进 *_query(实测:"去这里拿Orange" ->
# place_query="这里")。查询字段里出现指代词 = 确定性清洗成 deictic 标记,
# 不指望提示词能管住模型。
_DEICTIC_WORDS = {"这", "那", "这个", "那个", "这里", "那里", "这边", "那边",
                  "这儿", "那儿", "此处", "这块", "那块", "这个地方", "那个地方",
                  "这个位置", "那个位置"}

# 语音转写带标点/大小写抖动("回来"vs"回来。"),同一句话会裂成多个缓存键:
# 每个变体首次都付一次 LLM,且解析可能不一致(实测"回来。"被判成 stop)。
# 归一化 = 去两端标点空白 + lower,统一用于缓存键与 brain 的 停/y/n 词表匹配;
# LLM 仍吃原文——句中标点("哦,不是,去X")是真实内容,不动。
_STRIP = " \t\r\n。,、!?;:…~·“”‘’\"'()()《》〈〉【】[].,!?;:~-"


def norm_cmd(text: str) -> str:
    return text.strip(_STRIP).lower()


def _sanitize(data):
    for q, d in (("object_query", "object_deictic"),
                 ("place_query", "place_deictic"),
                 ("dest_query", "dest_deictic")):
        v = data.get(q)
        if v and v.strip() in _DEICTIC_WORDS:
            data[q], data[d] = None, True
    # 指名即指称:object_query 是真名字、且没有独立类别词(noun 缺失或只是名字的
    # 重复)时,object_deictic 视为地点槽漏过来的错标,按命名处理。
    # ("去这里拿Orange"实测:'这里'的 deictic 被同时标到了 object 上。
    #  "这个杯子"类真指代 noun_class=杯 ≠ query,不受影响。)
    oq, nc = data.get("object_query"), data.get("noun_class")
    if oq and (not nc or nc.strip().lower() == oq.strip().lower()):
        data["object_deictic"] = False
        data["noun_class"] = None
    return data


def load_openai_key():
    """OPENAI_API_KEY 环境变量优先,其次 Intension/.openai_key(已 gitignore)。"""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        kf = _DIR / ".openai_key"
        if kf.exists():
            key = kf.read_text(encoding="utf-8").strip()
    return key


class CommandParser:
    def __init__(self, table, model="gpt-5-mini", mode="on", key="",
                 cache_path=None, say=print, logev=lambda rec: None):
        self.table = table            # 物体名 -> 质心(名字表进提示词,口语->规范名)
        self.model, self.mode, self.key = model, mode, key
        self.say, self.logev = say, logev
        self.cache_path = Path(cache_path or _DIR / "parse_cache_v2.json")
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        self.cache = {}
        for k, v in raw.items():  # 旧键就地归一合并;冲突时首见(预热条目)胜
            self.cache.setdefault(norm_cmd(k), v)
        self._unsaved = set()  # 只在内存、未过确认门的键:落盘时过滤掉

    # -------------------------------------------------- LLM
    def _prompt(self, text):
        return (
            "把这句对机械狗说的中文指令解析成 JSON(只输出 JSON)。\n"
            "动作 action:\n"
            "- fetch: 去拿某个物体(之后可能要送到某处)\n"
            "- grab: 原地抓取——机器人已在目标面前,不要移动,直接抓。动词是\n"
            "  '抓/夹'且没说去哪、没说拿来/给我时用它('抓orange''抓这个');\n"
            "  出现'去/来/给我/带'的一律 fetch\n"
            "- goto: 只移动过去,不抓取\n"
            "- stop: 让它立刻停下\n"
            "- none: 都不是\n"
            "场景中已命名的物体:\n"
            f"{'、'.join(sorted(self.table))}\n"
            "三个槽位各自独立填。用了'这个/那个/这里/那边/这边'等现场指代词时只把\n"
            "对应 *_deictic 置 true;**指代词本身永远不要写进 *_query**(query 只放\n"
            "物体名字),也不要猜指代词指什么——指什么由视线决定。\n"
            "例:'去这里拿Orange' -> place_deictic=true, place_query=null,\n"
            "object_query=orange, object_deictic=false(Orange 是明确指名,不是指代)。\n"
            "- object_query/object_deictic/noun_class: 要拿的物体(仅 fetch 用,goto\n"
            "  一律填 place_*)。指名 -> 上表最匹配的一个;明确指名但不在表中的照抄\n"
            "  原文;没指名 -> null。'这个杯子' -> object_deictic=true 且 noun_class=杯。\n"
            "  noun_class 是指代/泛指时的类别词(杯、狗…),没有则 null\n"
            "- place_query/place_deictic: 在哪里拿 / 要去哪(fetch 的取货地点、goto\n"
            "  的目的地)。'去这里拿X' -> place_deictic=true;'去桌子那边' ->\n"
            "  place_query=桌子\n"
            "- dest_query/dest_deictic: 拿到之后送去哪。'拿到桌子那边' ->\n"
            "  dest_query=桌子;'拿去那边' -> dest_deictic=true;没说送哪 -> 都空\n"
            "- to_user: fetch=送到用户身边('拿来/给我/带过来';没说送哪也默认 true,\n"
            "  除非给了 dest_* 或明确只是拿着不送);goto=目的地是用户('过来')\n"
            f"指令:「{text}」")

    def _call_api(self, text, key):
        t0 = time.time()
        self.say("[LLM] 解析中…")
        body = json.dumps({
            "model": self.model,
            "reasoning_effort": "minimal",  # 解析任务不需要深思,省时省钱
            "messages": [{"role": "user", "content": self._prompt(text)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "robot_command", "strict": True,
                                                "schema": PARSE_SCHEMA}},
        }).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                resp = json.loads(r.read())
            data = json.loads(resp["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            self.say(f"[LLM] API {e.code}: {e.read().decode(errors='ignore')[:160]}")
            return None
        except Exception as e:
            self.say(f"[LLM] 解析失败({type(e).__name__}: {e})")
            return None
        self.cache[key] = data      # 先只进内存(本进程复用);confirm() 后才落盘
        self._unsaved.add(key)
        self.say(f"[LLM] {json.dumps(data, ensure_ascii=False)}  ({time.time() - t0:.1f}s)")
        return data

    # -------------------------------------------------- 对外
    def confirm(self, text):
        """text 的解析过了确认门(用户点头)-> 落盘。demo 预热 = 跑一遍台词并
        确认(或 --yes);没确认过的解析永不落地,缓存文件 = 全部人工校验过。"""
        key = norm_cmd(text)
        if key not in self._unsaved:
            return
        self._unsaved.discard(key)
        keep = {k: v for k, v in self.cache.items() if k not in self._unsaved}
        try:
            self.cache_path.write_text(json.dumps(keep, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        except Exception:
            pass

    def parse(self, text):
        """返回槽位 dict(kind: stop/fetch/goto/help + 各槽位)或 None(不可解析)。"""
        key = norm_cmd(text)
        if not key:
            return None  # 纯标点/空白 = 转写噪声
        data = self.cache.get(key)
        cached = data is not None
        if not cached:
            if self.mode == "off":
                return None
            data = self._call_api(text, key)
            if data is None:
                return None
        data = _sanitize(dict(data))
        # grab 是私有约定:只跟"抓/夹"动词。LLM 把「拿一下这个」误判 grab 实测过——
        # 原地抓需要狗位,拿类动词被拦在那道门上。动词不符一律降级 fetch。
        if data.get("action") == "grab" and not any(v in key for v in ("抓", "夹")):
            data["action"] = "fetch"
            if not data.get("dest_query") and not data.get("dest_deictic"):
                data["to_user"] = True  # 拿类缺省送回
        self.logev({"topic": "llm_parse", "text": text, "result": data, "cached": cached})
        act = data.get("action")
        if act == "stop":
            return {"kind": "stop"}
        if act not in ("fetch", "grab", "goto"):
            return {"kind": "help"}
        return {"kind": act,
                "object": data.get("object_query"),
                "object_deictic": bool(data.get("object_deictic")),
                "noun": data.get("noun_class") or "",
                "place": data.get("place_query"),
                "place_deictic": bool(data.get("place_deictic")),
                "dest": data.get("dest_query"),
                "dest_deictic": bool(data.get("dest_deictic")),
                "to_user": bool(data.get("to_user"))}
