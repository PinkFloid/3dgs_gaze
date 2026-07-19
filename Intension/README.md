# Intension — 盯住即夹取(最小 demo)

一个脚本:订阅 `gaze_live` 的 `gaze.intent` 流,同一物体**因果注视 ≥ 4.8s** →
控制台问一句"夹取「水杯」?" → 键入 `y` → 发出 `grasp(object_name, target_world)`
技能调用(缺省只打印;`target_world` 是该实例的池化质心)。
是 `docs/AGENT_DESIGN.md` 的最小可跑切片:层A内核(VisitTracker)+ 规则脑(单条规则),无 LLM。

## 运行

```bash
# 终端 1:感知层(与平时一样,加 --publish)
python Eye_Tracker/tools/gaze_live.py --publish 5581 [...]

# 终端 2:本 demo(同一个 python 环境,需 pyzmq + msgpack)
python Intension/stare_to_grasp.py
```

盯目标时终端有实时进度(`盯 水杯 2.4/4.8s vote 78%`)——dwell 交互没有反馈就没法用。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dwell` | 4.8 | 触发阈值 (s) |
| `--merge-gap` | 0.6 | visit 内瞥离容忍 (s),同 grasp_intent |
| `--min-vote` | 0.5 | verdict 准入,同 grasp_intent |
| `--confirm-timeout` | 8 | y/n 等待,超时=取消 |
| `--suppress` | 30 | 问答后(无论 y/n)同物体静默期 (s) |
| `--skill-endpoint` | 无 | 技能端 REQ 地址,如 `tcp://127.0.0.1:5583`;缺省只打印 |
| `--publish` | 无 | 把 `attention.*` 事件 PUB 出去(未来大脑进程的订阅口,如 5582) |
| `--raw-log` | 关 | 原样落盘进来的 `gaze.intent` 流 |
| `--replay` | 无 | 用 `--raw-log` 录的 jsonl 回放,替代 ZMQ(纯 stdlib 可跑) |

在确认提示处输入 `q` / `停` 退出。

## 行为细节(即层A的触发纪律)

- 实时 dwell 来自 **provisional** verdict,final 结账——只吃 final 永远不会触发(设计稿 §4.1);
- 每 visit 只问一次;拒绝或确认后该物体静默 `--suppress` 秒,静默期后需**重新累积** 4.8s;
- 背景表面(label < 10)、`vote_share < 0.5`、无质心的 verdict 不参与,但其时间戳仍推动 visit 超时。

## 日志与回放

每次运行建 `Intension/logs/<时间戳>/`:
- `events.jsonl` — attention 事件、问答、技能调用(自包含证据包);
- `raw.jsonl` — 开 `--raw-log` 时的原始输入流。

回归:`--replay raw.jsonl` 是确定性的,改参数在同一段录像上 A/B(设计稿 §11 的验证方案 A)。

## 技能调用格式(REQ,msgpack)

```json
{"skill": "grasp",
 "params": {"object_name": "水杯", "target_world": [1.32, -0.45, 0.86]},
 "req_id": "req-001", "t": 123.4,
 "intent_summary": "用户注视 4.8s 并确认夹取 水杯"}
```

技能端应答 `{"accepted": bool, "reason": str}`(设计稿 §8)。

## 与 AGENT_DESIGN.md 的关系

保留:provisional/final 双流会计、visit 因果语义、单发+抑制、确认门在代码里、
jsonl+回放、`attention.*` 接口形状。砍掉(等下一步):LLM 大脑、alternation/point 等
事件、技能层进程、坐标桥接。`VisitTracker` 以后原样抬进 `attention_arbiter.py`。
