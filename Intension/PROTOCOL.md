# 视线意图机 ↔ 机械狗 通信协议 v1

一句话:**狗机是服务端**(bind),意图机是客户端(connect);请求走 REQ/REP 拿"立即回执",
执行进度走 PUB 广播;所有消息 msgpack 编码的 UTF-8 字典。

```
意图机(视线管线)                         狗机(Go2 控制)
stare_to_grasp.py ── REQ ──────────────▶ REP :5583   技能请求(<100ms 立即回执)
       (可选)SUB ◀────────────────────── PUB :5584   skill.status 进度广播
```

依赖(狗机):`pip install pyzmq msgpack`。同一局域网,狗机用固定 IP,放行 5583/5584。

**狗端拿到的文件是 `dog_link.py`**:通信壳已封好(收包/解析/回执/广播/急停),
只需改「你的区」三个函数 —— `execute()`(写技能逻辑)、`on_stop()`(接急停)、
`get_pose()`(报位姿);`MY_SKILLS` 列表声明支持哪些技能名。execute 抛异常自动
广播 failed、忘发终态自动补 done,不会把对方卡死。
它自带 3 秒假执行示例,不接真机直接跑,就是全链路模拟器(意图机侧自测同样用它)。

## 1. 技能请求(意图机 → 狗机,REQ)

```json
{"v": 1, "type": "skill.request",
 "req_id": "20260719-153012-001",
 "sent_at": 1789456123.4,
 "frame": "board/v2",
 "skill": "grasp",
 "params": {"object_name": "黄色机器人",
            "target_world": [-0.185, 3.413, 0.829]},
 "intent_summary": "用户注视 4.8s 并确认夹取 黄色机器人"}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `v` | int | 协议版本,当前 1。不认识的版本直接拒绝 |
| `req_id` | str | 全局唯一(会话时间戳+序号),后续所有状态用它对账 |
| `sent_at` | float | 发送方墙钟(epoch 秒)。注意:不要用我日志里的流时间 |
| `frame` | str | 坐标系标识 = 地图版本号。**不匹配必须拒绝**,见 §4 |
| `skill` | str | `grasp` / `move_to` / `stop` / `get_state` |
| `params` | dict | 按技能,见下表 |
| `intent_summary` | str | 人读的因果记录,狗端只需要原样进日志 |

| skill | params | 语义 |
|---|---|---|
| `grasp` | `object_name: str`, `target_world: [x,y,z]` 米, `deliver_to: [x,y,z]`(可选) | 走到目标附近→抓取→带回 `deliver_to`(=确认时刻的用户头位置;缺省回停放点) |
| `move_to` | `x, y, yaw` | 板坐标系位姿 |
| `stop` | 无 | **急停,最高优先级**,见 §3 |
| `get_state` | 无 | 回执里带当前位姿与忙闲 |

## 2. 回执(狗机 → 意图机,REP,必须 <100ms)

```json
{"v": 1, "req_id": "20260719-153012-001", "accepted": true, "reason": ""}
```

**硬规则:回执只表示"收到并合法",不等执行。** REP 循环里不允许任何阻塞的机器人调用——
执行丢给工作线程,进度走 5584。拒绝时 `accepted: false` + reason,约定的 reason:
`busy`(v1 同时只执行一个技能)/ `frame_mismatch` / `unknown_skill` / `bad_params` / `out_of_workspace`。
意图机侧等回执超时 2s,超时按失败处理。

## 3. 进度广播(狗机 PUB :5584,话题 `skill.status`)

```json
{"v": 1, "req_id": "20260719-153012-001", "state": "moving",
 "pose": {"x": 0.5, "y": 1.2, "yaw": 1.57}, "detail": "", "t": 1789456125.0}
```

`state` 顺序:`accepted → moving → grasping → returning → done`,任何时刻可终止于
`failed` 或 `stopped`。`pose` 是板坐标系狗位姿,随手带上(以后意图机要用它做
"看狗"检测)。**急停语义**:收到 `stop` 请求 → 立即回执 → 中断当前动作(unitree
damp/stop + 臂急停)→ 给被中断的 req_id 广播 `stopped`。急停链路上不许有任何模型/慢逻辑。

## 4. 坐标系(最容易悄悄出错的地方)

- 板坐标系(board frame):**米制、z-up、地板 z≈0**,由 `tags_world.json` 的标定定义。
- `frame` 字段 = 地图版本(如 `board/v2`)。两台机器必须持有**同一版本**的标定结果;
  狗端自己定位到板系(v0.5 停放点静态外参,v1 狗头 ArUco)。
- 请求的 frame 与狗端配置不一致 → 拒绝 `frame_mismatch`,**宁可不动不要走错**。
- **联调期约定**:`dog_link.py` 当前版本不校验 frame(收到即忽略);上真狗或出现第二张
  地图后,在狗端 execute 前加一行版本比对即可。发送方始终带上该字段,成本为零。

## 5. 联调三步(不需要真狗)

```bash
# 1) 狗机同学:python dog_link.py                   # 打印收到的请求+模拟执行
# 2) 意图机:python Intension/stare_to_grasp.py \
#        --skill-endpoint tcp://<狗机IP>:5583        # 盯 4.8s → y → 请求发出
# 3) 看狗端终端:请求 JSON + accepted→moving→…→done 的状态流
# 之后他把 dog_link 里 execute() 的 sleep 换成 unitree_sdk2 调用,协议层零改动
```

## 6. 演进规则

加字段=兼容(收到不认识的字段必须忽略,不许报错);改语义/删字段=升 `v`。
新技能(place/handover…)只是新的 `skill` 名+params,双方各自扩一张表。
