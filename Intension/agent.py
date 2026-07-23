"""agent.py -- 指令文本 -> 结构化命令(LLM + 解析缓存)。

只做"文本->结构"这一件事;绑定/几何/确认永远是外面的确定性代码。
缓存命中 0ms 且完全确定(parse_cache.json);未命中且 mode=='on' 时
直连 OpenAI(strict json_schema,reasoning minimal);都不行返回 None。
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
        self.cache_path = Path(cache_path or _DIR / "parse_cache.json")
        try:
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            self.cache = {}

    # -------------------------------------------------- LLM
    def _prompt(self, text):
        return (
            "把这句对机器人说的中文指令解析成 JSON(只输出 JSON)。\n"
            "机器人技能(action 取值):\n"
            "- fetch: 去拿某个物体并送回来(需要一个物体目标)\n"
            "- goto: 只移动过去,不抓取——去某物体旁边,或来用户身边('过来')\n"
            "- stop: 让它立刻停下\n"
            "- none: 都不是\n"
            "场景中已命名的物体(object_query 与 location_hint 只能取其中之一或 null):\n"
            f"{'、'.join(sorted(self.table))}\n"
            "字段规则:\n"
            "- deictic: 用了'这个/那个/那边'等现场指代、且没指名是上表中哪一个时为 true\n"
            "- object_query: 目标物体(fetch=要拿的物;goto=要去的参照物)->\n"
            "  上表中最匹配的一个;没指名、或说法同时匹配多个而无法确定时为 null\n"
            "- noun_class: 指代或泛指时的类别词(如 杯、机器人);没有则 null\n"
            "- location_hint: 顺带提到的地点参照物 -> 上表中的一个;没有则 null\n"
            "- deliver_to_user: fetch=是否要求送到用户身边;goto=目的地是否就是用户身边\n"
            f"指令:「{text}」")

    def _call_api(self, text):
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
        self.cache[text] = data
        try:  # 缓存落盘:demo 台词预热一遍后离线可用
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
        except Exception:
            pass
        self.say(f"[LLM] {json.dumps(data, ensure_ascii=False)}  ({time.time() - t0:.1f}s)")
        return data

    # -------------------------------------------------- 对外
    def parse(self, text):
        """返回内部指令 dict(kind: stop/goto/named/deictic/help)或 None(不可解析)。"""
        data = self.cache.get(text)
        cached = data is not None
        if not cached:
            if self.mode == "off":
                return None
            data = self._call_api(text)
            if data is None:
                return None
        self.logev({"topic": "llm_parse", "text": text, "result": data, "cached": cached})
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
