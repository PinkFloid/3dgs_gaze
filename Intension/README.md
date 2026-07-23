# Intension — 多模态指令层(语言出结构,注视出指代)

主入口 **`brain.py`**:指令 → 指代消解 → y/n 确认 → 派发给狗。三种交互一个进程:

| 说法 | 消解 |
|---|---|
| `拿一下黄色机器人` | 名字:模糊匹配地图物体表(池化质心) |
| `把这个杯子拿来` | 视线:眼-声回看窗取近期注视 + 类别过滤 |
| `帮我把那个黄颜色的机器人弄过来` | 同上:口语说法由 LLM 规范化成物体表名字 |
| `过来` / `去凳子那边` / `去那边` | goto:发空 object 的 grasp = 纯导航(目的地=用户/名字/注视) |
| 盯满 4.8s(`--proactive 4.8` 时) | 主动问询:"要我拿来吗?" |
| `停` | 急停旁路,永不过 LLM,不确认 |

解析:**除"停"和 y/n 外全部由 LLM 转结构**(gpt-5-mini 直连,~2s),结果进
`parse_cache.json` —— 同一句话第二次起 0ms 且完全确定。demo 台词预热一遍即可
离线跑;`--llm off` = 只走缓存(回归/实验模式)。LLM 只做文本→结构,
绑定/几何/确认永远是确定性代码。

## 运行

```bash
# 感知(另一终端): python Eye_Tracker/tools/gaze_live.py --publish 5581 ...
python Intension/brain.py                          # 纯本机,派发只打印
python Intension/brain.py --skill-endpoint tcp://狗机:5583   # 接真狗/模拟器
python Intension/brain.py --proactive 4.8          # 加开盯视主动问询

# 无硬件回放回归(确定性):
python Intension/gen_fake_gaze.py /tmp/fake.jsonl
python Intension/brain.py --llm off --replay /tmp/fake.jsonl --yes \
    --script "106.5:把这个杯子拿来"
```

## 文件地图

- `brain.py` — Intension 层本体(解析级联 + AttentionBuffer 眼-声绑定 + 确认 + 派发)
- `stare_to_grasp.py` — 遗产入口:纯"盯4.8s→问"(层A `VisitTracker` 的宿主,brain 复用其组件;留作对照与回归)
- `dog_link.py` — **发给狗端同学的唯一文件**:通信壳封好,他只填 `execute/on_stop/get_pose`
- `send_test.py` — 意图机替身:2 秒后发固定样例,狗端联调用
- `gen_fake_gaze.py` — 合成 gaze.intent 流,无硬件回放
- `parse_schema.json` — LLM 解析的输出 JSON schema
- `PROTOCOL.md` — 通信契约 v1(端口、消息、急停语义、坐标系)

端口:5581 感知入 / 5583 命令出(REQ)/ 5584 狗状态回(SUB);日志每次运行落
`logs/<时间戳>/events.jsonl`(指令、消解、绑定候选、问答、派发、狗状态全在)。
