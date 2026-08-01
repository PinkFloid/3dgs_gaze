# Intension — 多模态指令层(语言出槽位,注视出指代)

主入口 **`brain.py`**:指令 → 三槽位消解(object/place/dest)→ y/n 确认 → 派发给狗。
大脑**不记忙闲、不排队、发完即忘**:狗忙时拒单(busy)就提示"本单作废",
要打断先「停」再重下;狗端将来升级抢占语义时 brain 零改动。

| 说法 | 消解 |
|---|---|
| `拿一下黄色机器人` | object=名字:模糊匹配地图物体表(池化质心) |
| `把这个杯子拿来` | object=视线:眼-声回看窗取近期注视 + 类别过滤 |
| `帮我把那个黄颜色的机器人弄过来` | 同上:口语说法由 LLM 规范化成物体表名字 |
| `去这个地方拿橘子` | place=注视处落点(看着目标处的物体/家具说)+ 物名给狗端到位检测 |
| `把这个拿到物品台那边` | dest=名字:送达目的地显式给出,不送用户 |
| `过来` / `去凳子那边` / `去那边` | goto:目的地=用户 / 名字 / 注视处落点 |
| 盯满 4.8s(`--proactive 4.8` 时) | 主动问询:"要我拿来吗?" |
| `停` / `停下` / `停止` | 急停旁路,永不过 LLM,不确认(标点/大小写归一:转写成"停。"同样命中) |

解析:**除"停"和 y/n 外全部由 LLM 转槽位**(gpt-5-mini 直连,~2s),结果
**过确认门后**才落盘进 `parse_cache_v2.json`(键做标点/大小写归一,转写抖动
不裂键;没确认的解析只活在本进程内存——确认即人工校验,误判进不了文件)。
同一句话第二次起 0ms 且完全确定。demo 台词预热 = 跑一遍并确认(或 `--yes`),
之后离线可用;`--llm off` = 只走缓存(回归/实验模式)。LLM 只做文本→槽位
(指代词只标记不猜指什么),绑定/几何/确认永远是确定性代码。

两条注意通道(core/attention):**物体通道** AttentionBuffer = 眼-声绑定的
E1 语义(物体身份+质心,投票门把关);**落点通道** PlaceBuffer = 物体注视的
**表面实际落点**(物品台的那个角而非质心,未命名实例也算,无投票门),
供 place/dest 槽。地板/墙注视不当地点(噪声大,无需求)。

## 运行

```bash
# 感知(另一终端): python Eye_Tracker/tools/gaze_live.py --publish 5581 ...
python Intension/brain.py                          # 纯本机,派发只打印
python Intension/brain.py --skill-endpoint tcp://狗机:5583   # 接真狗/模拟器
python Intension/brain.py --voice                  # 开麦语音指令(与打字并行)
#   免屏幕提示音:断句"叮"=听到了 / 升调=等确认(说"好/嗯"即确认) / 双音=已派发或狗done
#   / 低音=失败取消;词表与缓存键均归一(标点/大小写),转写"好。"照样确认
#   噪声四闸:能量 --voice-rms / 超长段丢弃 / silero 复核 / 幻听过滤——风扇键盘不触发指令
python Intension/voice_input.py --once             # 单独试麦:说一句看转写
python Intension/brain.py --proactive 4.8          # 加开盯视主动问询
python Intension/brain.py --proactive 3 --proactive-goto --yes  # 看哪去哪(盯满即去物体旁)

# 无硬件回放回归(确定性):
python Intension/gen_fake_gaze.py /tmp/fake.jsonl
python Intension/brain.py --llm off --replay /tmp/fake.jsonl --yes \
    --script "106.5:把这个杯子拿来"

# 一键回归(7 项:绑定/视线地点/地板负例/goto落点/显式dest/急停/忙拒作废):
./Intension/run_regression.sh
```

## 文件地图

- `brain.py` — Intension 层本体(三槽位 resolver + 眼-声绑定 + 确认 + 派发)
- `dog_link.py` — **发给狗端同学的唯一文件**:通信壳封好,他只填 `execute/on_stop/get_pose`
- `send_test.py` / `send_test2.py` — 意图机替身:发固定样例,狗端联调用
- `gen_fake_gaze.py` — 合成 gaze.intent 流,无硬件回放
- `eval_binding.py` — E1 指代消歧打分(events.jsonl → CSV → 角间隔-准确率表)
- `parse_schema.json` — LLM 槽位输出 JSON schema(v2:object/place/dest 三槽)
- `PROTOCOL.md` — 通信契约 v1(端口、消息、急停语义、坐标系)
- `archive/stare_to_grasp.py` — 遗产入口"盯4.8s→问",已被 brain 全面取代
- `archive/dog_link_preempt.py` — 抢占语义版 dog_link(已验但未启用:狗端暂不改)

端口:5581 感知入 / 5583 命令出(REQ)/ 5584 狗状态回(SUB);日志每次运行落
`logs/<时间戳>/events.jsonl`(指令、消解、绑定候选、问答、派发、狗状态全在)。
