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
                  "这个位置", "那个位置", "哪里", "哪儿", "什么地方"}

# 送回用户的明说标记(用户裁定 2026-08-20:「拿一下X」=过去拿住,不默认送回;
# 送回要么句里带这些词,要么拿完另说一句"给我")。判定前先抠掉否定说法:
# "拿apple不需要给我"(缓存实测句)不许因子串"给我"被翻成送回。
_GIVE_WORDS = ("给我", "拿来", "过来", "带来", "递", "送")
_NO_GIVE = ("不需要给我", "不用给我", "不要给我", "别给我", "不给我",
            "不需要拿来", "不用拿来", "别拿来", "先不给我", "不用送", "不送")


def _says_give(key):
    for w in _NO_GIVE:
        key = key.replace(w, "")
    return any(w in key for w in _GIVE_WORDS)

# 语音转写带标点/大小写抖动("回来"vs"回来。"),同一句话会裂成多个缓存键:
# 每个变体首次都付一次 LLM,且解析可能不一致(实测"回来。"被判成 stop)。
# 归一化 = 去两端标点空白 + lower,统一用于缓存键与 brain 的 停/y/n 词表匹配;
# LLM 仍吃原文——句中标点("哦,不是,去X")是真实内容,不动。
_STRIP = " \t\r\n。,、!?;:…~·“”‘’\"'()()《》〈〉【】[].,!?;:~-"


# 句首动词的高频听岔(whisper small + DJI 麦实测):确定性纠回去,纠完的键
# 直接命中已有缓存(「麻衣下这个」-> 「拿一下这个」0ms),不再抽 LLM 盲盒。
# 只治句首、只治动词:物体名的同音错让 LLM 按词表纠(提示词里有)。
_ASR_HEAD = (("麻衣下", "拿一下"), ("那一下", "拿一下"), ("麻一下", "拿一下"),
             ("拿衣下", "拿一下"), ("纳一下", "拿一下"), ("那衣下", "拿一下"))
# 句中同音错字(head 表只管句首):实测「把这个网球拿给我」→"往球",LLM 直接懵。
# 只收指令域内无歧义的替换,实测一个加一个。
_ASR_SUBS = (("往球", "网球"), ("王球", "网球"))


def norm_cmd(text: str) -> str:
    t = text.strip(_STRIP).lower()
    for bad, good in _ASR_SUBS:
        t = t.replace(bad, good)
    for bad, good in _ASR_HEAD:
        if t.startswith(bad):
            t = good + t[len(bad):]
            break
    return t


def _sanitize(data):
    for q, d in (("object_query", "object_deictic"),
                 ("place_query", "place_deictic"),
                 ("dest_query", "dest_deictic")):
        v = data.get(q)
        if v and v.strip() in _DEICTIC_WORDS:
            data[q], data[d] = None, True
    # 指代词不是类别:光杆「拿一下这个给我」实测 LLM 把 noun_class 填成"这个",
    # 类过滤拿它去筛注视缓冲全灭(苹果粉明明在盯着,候选全 0.00)。指代词一律
    # 从类槽清掉,指代标志补上——失败解析不落盘,但同会话内存缓存会复读这口毒。
    nc = data.get("noun_class")
    if nc and nc.strip() in _DEICTIC_WORDS:
        data["noun_class"], data["object_deictic"] = None, True
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
            "输入是语音转写,常有同音错字:'那一下/麻衣下'='拿一下'、'被子'='杯子'、\n"
            "'和子'='盒子'。先按读音对着下面的动作和物体名把错字纠回来再解析;\n"
            "发音接近词表里哪个名字就当哪个。但听不出跟指挥狗有关的内容\n"
            "(闲聊、感叹、外语音节如'way way way')-> action=none,不要硬凑。\n"
            "动作 action:\n"
            "- fetch: 去拿某个物体(之后可能要送到某处)。'拿/拿一下/去拿/帮我拿'\n"
            "  都是 fetch\n"
            "- grab: 原地抓取——机器人已在目标面前,不要移动,直接抓。只跟动词\n"
            "  '抓/夹',或明确说'原地拿/就地拿'('抓orange''原地拿这个');\n"
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
            "  '把这个放到那里去'/'放到哪里去' -> object_deictic=true 且\n"
            "  dest_deictic=true(放置=fetch+送达,'哪里'是手指方向不是疑问)\n"
            "放置句(说'放/放回/放到/搁/摆',但没说要拿什么):狗手里已经拿着东西,\n"
            "只需要知道放哪儿 -> object_query/object_deictic/noun_class 全部留空,\n"
            "地点填 dest_*,to_user=false。句中的物体名是**目的地**不是要拿的东西,\n"
            "哪怕它是个能抓的小东西也一样:'放回球L' = 放到球L那个位置 ->\n"
            "dest_query=球L, object_query=null;'放回原来的地方' -> dest_deictic=true。\n"
            "只有同时说了拿什么才填 object:'把球M放到纸箱子' -> object_query=球M,\n"
            "dest_query=纸箱子。\n"
            "- to_user: fetch=是否送到用户身边:只有明说'拿来/给我/带过来/递给我/\n"
            "  拿过来'才 true;只说'拿(一下)X'= 过去拿住,不送 -> false。\n"
            "  goto=目的地是用户('过来')\n"
            "单说'给我/拿过来/递过来'(没提任何物体):把手里的送到我这 ->\n"
            "action=goto, to_user=true(狗过来,东西在爪上不放下)。\n"
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
    def _known(self, q):
        """q 是否像物体表里的某个名字(双向子串粗判:"网球L"/"球L的位置" 都算命中
        球L)。只用来在几个槽位候选里挑哪个像地名,真消解仍在 core/resolve。"""
        q = (q or "").strip()
        return bool(q) and any(q in nm or nm in q for nm in self.table)

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
        # 指代必须有指代词撑腰:LLM 会给"那一下白杯1"(语音同音错字,本意"拿一下")
        # 标 object_deictic,于是明明指了名却掉进视线绑定,没注视就报"「杯」类有多个"。
        # 句里根本没出现指代词、且名字对得上物体表 -> 按指名处理。
        # ("那"单字不算:它多半就是"拿"的同音错字;"这"单字算。)
        if data.get("object_deictic") and self._known(data.get("object_query")) \
                and not any(w in key for w in ("这", "那个", "那只", "那颗", "那些",
                                               "那俩", "那瓶", "那杯", "那台")):
            data["object_deictic"] = False
            data["noun_class"] = None
        # 指代句里 LLM 给的具体名,按"名字的字都在句里吗"分两路(各有实测反例):
        # ①「这个红苹果」→苹果红:语序归一的真指名(字全在句中)——指名即指称,
        #   名字压过指代,不盯着也拿对;之前按字面 not in key 一刀切把它冤杀了。
        # ②「这个苹果」→苹果红:凭空多出"红"=幻觉,清掉让视线裁决。
        # 单字类词(球M 的"球")不算指名,仍走视线绑定。
        oq = data.get("object_query")
        if data.get("object_deictic") and oq:
            hz = [ch for ch in oq if not ch.isascii()]
            if len(hz) >= 2 and all(ch in key for ch in hz) and self._known(oq):
                data["object_deictic"] = False
                data["noun_class"] = None
            else:
                data["object_query"] = None
        # 「过来」类:目的地是人,不是地名。LLM 时不时把 place_query 填成
        # "用户"/"user"(实测),brain 会拿它去物体表里找一个叫"用户"的东西而报错;
        # 也见过标成 place_deictic 让狗走去注视点。归一到 to_user 这一条路上。
        if data.get("action") == "goto":
            if (data.get("place_query") or "").strip().lower() in (
                    "用户", "user", "我", "你", "我这", "我这里", "你这", "主人"):
                data["place_query"], data["to_user"] = None, True
            if any(w in key for w in ("过来", "回来", "到我这", "来我这", "跟我")) \
                    and not self._known(data.get("place_query")):
                data["place_query"], data["place_deictic"] = None, False
                data["to_user"] = True
        # 裸放置句(「放到那里去」「放回球L」):只说放、没说拿什么 = 狗手里已经
        # 有东西,句中出现的物体名是"放到哪"而不是"抓什么"。LLM 两头都会搞反:
        # 标 object_deictic 会让视线绑定把盯着的落点当成要抓的(实测 -009 把纸箱子
        # 抓走了);填 object_query 会变成再去拿一个(实测「放回网球L」-> grab 球L)。
        # 提示词管不住,这里确定性纠偏。带"把/将"或物指代词的不算裸放置。
        if any(v in key for v in ("放", "搁", "摆")) \
                and not any(v in key for v in ("拿", "抓", "取", "夹", "递", "带", "给")) \
                and not any(w in key for w in ("把", "将", "放下", "放开")) \
                and not any(w in key for w in ("这个", "那个", "这只", "这颗", "这些", "这俩")):
            if not data.get("dest_deictic") and not self._known(data.get("dest_query")):
                # 送达地空着,或填了句外词(实测用户嘴瓢说出"place",LLM 就把
                # dest_query 填成 place):按 dest>object>place 取第一个对得上
                # 物体表的候选,都对不上就留原样让消解去报"没有唯一命中"
                cands = [data.get("dest_query"), data.get("object_query"),
                         data.get("place_query")]
                data["dest_query"] = next((c for c in cands if self._known(c)),
                                          next((c for c in cands if c), None))
            data["object_query"] = data["noun_class"] = None
            data["object_deictic"] = data["to_user"] = False
            if not data.get("dest_query") and not data.get("dest_deictic"):
                data["dest_deictic"] = True  # 放类句必有去处:落点走视线
        # grab 是私有约定:只跟"抓/夹"动词。LLM 把「拿一下这个」误判 grab 实测过——
        # 原地抓需要狗位,拿类动词被拦在那道门上。动词不符一律降级 fetch。
        if data.get("action") == "grab" and \
                not any(v in key for v in ("抓", "夹", "原地", "就地")):
            data["action"] = "fetch"
        # 光杆"给我/拿过来"(哪个槽都没填):把手里的送来 = goto 到用户身边,
        # 不撒手。LLM 会硬凑成没物体的 fetch,handle 只能干瞪眼出帮助语。
        if data.get("action") == "fetch" \
                and not any(data.get(k) for k in
                            ("object_query", "object_deictic", "noun_class",
                             "place_query", "place_deictic",
                             "dest_query", "dest_deictic")) \
                and _says_give(key):
            data["action"], data["to_user"] = "goto", True
        # to_user 只认明说:「拿一下X」=过去拿住,不送(用户裁定 8-20)。
        # 旧缓存按旧默认("拿=送回")存了一堆 to_user=true,这里统一按新语义
        # 确定性纠偏,缓存不用升版。
        if data.get("action") == "fetch":
            data["to_user"] = bool(
                not data.get("dest_query") and not data.get("dest_deictic")
                and _says_give(key))
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
