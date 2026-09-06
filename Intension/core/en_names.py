"""en_names.py -- 英文演示的物名对照(地图名是中文,狗端/字幕/LLM 都要英文形)。

EN_GLOSS: 地图名 -> 英文说法(LLM 提示词对照、whisper 热词、demo 字幕)。
EN_CLASS: 英文类别词 -> 中文类别子串(指代句 "this cup" 的类过滤走 noun_match 子串匹配)。
"""
from __future__ import annotations

EN_GLOSS = {
    "网球L": "left tennis ball", "网球M": "middle tennis ball", "网球R": "right tennis ball",
    "球L": "left tennis ball", "球M": "middle tennis ball", "球R": "right tennis ball",
    "网球1": "tennis ball 1", "网球2": "tennis ball 2", "网球3": "tennis ball 3",
    "苹果红": "red apple", "苹果粉": "pink apple", "苹果": "apple",
    "香蕉": "banana", "橘子": "orange", "石榴": "pomegranate",
    "白杯1": "white cup 1", "白杯2": "white cup 2", "红杯": "red cup",
    "水瓶": "bottle", "纸箱子": "box", "物品台": "table",
}
EN_CLASS = {
    "ball": "球", "balls": "球", "tennis": "球", "tennis ball": "球", "tennisball": "球",
    "apple": "苹果", "apples": "苹果", "cup": "杯", "cups": "杯", "mug": "杯", "glass": "杯",
    "banana": "香蕉", "orange": "橘子", "fruit": "果", "box": "箱", "bottle": "瓶",
    "table": "物品台", "stand": "物品台", "desk": "物品台",
}
_INV = {v.lower(): k for k, v in EN_GLOSS.items()}


def gloss(name: str) -> str:
    """地图名 -> 英文;没有对照的原样返回。"""
    return EN_GLOSS.get(name, name)


def ungloss(q, table=None):
    """英文说法 -> 地图名(大小写/冠词不敏感);table 给了就只认表里有的名字。"""
    if not q or not isinstance(q, str):
        return q
    s = q.strip().lower()
    for art in ("the ", "a ", "an ", "my "):
        if s.startswith(art):
            s = s[len(art):]
    name = _INV.get(s)
    if name is None:
        return q
    if table is not None and name not in table:
        return q
    return name


def en_class_to_zh(noun: str) -> str:
    """英文类别词 -> 中文类别子串;中文/未知原样返回。"""
    if not noun or not noun.isascii():
        return noun
    s = noun.strip().lower()
    for art in ("the ", "a ", "an ", "this ", "that "):
        if s.startswith(art):
            s = s[len(art):]
    if s in EN_CLASS:
        return EN_CLASS[s]
    last = s.split()[-1] if s.split() else s
    return EN_CLASS.get(last, noun)
