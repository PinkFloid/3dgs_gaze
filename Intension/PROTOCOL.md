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
真机只需填一个 `RealDog` 适配器类(get_pose / send_velocity / stand_still /
gripper_close / gripper_open,Go2 对应关系写在类注释里);技能逻辑(standoff
接近、对准、夹取、送达)、每阶段看门狗超时、卡死检测、工作空间校验都已实现。
execute 抛异常自动广播 failed、忘发终态自动补 done,不会把对方卡死。
不接真机直接 `python dog_link.py --fake` 跑速度积分假狗 = 全链路模拟器
(无 RealDog 时自动回落假狗;意图机侧自测同样用它)。

## 1. 技能请求(意图机 → 狗机,REQ)

```json
{"v": 1, "type": "skill.request",
 "req_id": "20260719-153012-001",
 "sent_at": 1789456123.4,
 "frame": "board/v2",
 "skill": "grasp",
 "params": {"object_name": "黄色机器人",
            "target_world": [-0.185, 3.413, 1.571]},
 "intent_summary": "用户注视 4.8s 并确认夹取 黄色机器人"}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `v` | int | 协议版本,**维持 1**(2026-08-05 用户裁定:站位语义切换不升版,双方靠约定同步部署——切换日后旧语义 server 不可再用)。不认识的版本拒绝 |
| `req_id` | str | 全局唯一(会话时间戳+序号),后续所有状态用它对账 |
| `sent_at` | float | 发送方墙钟(epoch 秒)。注意:不要用我日志里的流时间 |
| `frame` | str | 坐标系标识 = 地图版本号。**不匹配必须拒绝**,见 §4 |
| `skill` | str | `grasp` / `place` / `move_to` / `stop` / `get_state` |
| `params` | dict | 按技能,见下表 |
| `intent_summary` | str | 人读的因果记录,狗端只需要原样进日志 |

| skill | params | 语义 |
|---|---|---|
| `grasp` | `object_name: str \| null`, `target_world: [x, y, yaw]`, `deliver_to: [x, y, yaw]`(可选), `object_hint: [x, y, z]`(可选) | **唯一技能,两种用法**。**v2(2026-08-05 起)**:`target_world` = **目标本体中心** x,y(米)+ **建议接近方位** yaw(弧度,板系 +x=0,逆时针正,从用户/参考侧指向目标;狗端可据此选站位侧,也可忽略)——**站位与避障由狗端自留**(参考实现 STANDOFF 0.6m,沿建议方位后退);`deliver_to` 同语义 = 送达**落点**(用户位置/指定地点)+ 朝向建议。协议仍不传高度。(2026-08-05 前的旧语义 = 意图机算好的站位,已废止;线上版本号仍为 1。)①`object_name` 有值 = 站定→按名检测(多候选按 `object_hint` 投影选框,见待对齐 #7)→抓取,有 `deliver_to` 则送达、无则原地 done;②**`object_name` 空 = 纯导航** |
| `place` | `target_world: [x,y,yaw]`(放置坐标), `place_name: str`(可选检测名,如 "storage box")——**不带 object 字段**,放的就是手里那件 | **把手里的东西放下**(2026-08-18 定):与 grasp 是两个独立方法,**编排在意图机**——带送达的取物 = 先派 grasp(纯抓,无送达字段),状态流报 done 后意图机自动补发 place 链单(req_id 带 p 后缀);grasp failed/stopped/急停/新指令 → 链废弃。狗端有 `place_name` 按名检测对准(箱被挪也稳),否则按坐标放 |
| `move_to` | — | 不使用:导航一律用 `object_name=null` 的 grasp 表达(服务器里残留的实现无害) |
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
(意图机 2026-07-28 起**不做忙闲记账**:收到 `busy` 拒单即提示"本单作废",不排队
不重试,要打断先发 `stop` 再重下。狗端将来若改为抢占语义(新单取消旧单,参考实现
`archive/dog_link_preempt.py`),意图机零改动。)

## 3. 进度广播(狗机 PUB :5584,话题 `skill.status`)

```json
{"v": 1, "req_id": "20260719-153012-001", "state": "moving",
 "pose": {"x": 0.5, "y": 1.2, "yaw": 1.57}, "detail": "", "t": 1789456125.0}
```

`state` 顺序:`accepted → moving → grasping [→ returning] → done`,任何时刻可终止于
`failed` 或 `stopped`(对准阶段仍是 `moving`,`detail:"aligning"`;失败原因走 detail:
`moving_timeout` / `stuck` / `unlocalized` / `grasp_missed` 等)。`pose` 是板坐标系
狗位姿,随手带上(以后意图机要用它做"看狗"检测)。另有独立话题 **`dog.heartbeat`**
(1Hz,pose+busy):意图机据此区分"空闲"和"死机",且不污染 skill.status 的日志流。**急停语义**:收到 `stop` 请求 → 立即回执 → 中断当前动作(unitree
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
# 2) 意图机:python Intension/brain.py \
#        --skill-endpoint tcp://<狗机IP>:5583        # 下指令 → y → 请求发出
# 3) 看狗端终端:请求 JSON + accepted→moving→…→done 的状态流
# 之后他把 dog_link 里 execute() 的 sleep 换成 unitree_sdk2 调用,协议层零改动
```

## 6. 演进规则

加字段=兼容(收到不认识的字段必须忽略,不许报错);改语义/删字段=升 `v`。
新技能(place/handover…)只是新的 `skill` 名+params,双方各自扩一张表。

## 7. 真机联调状态(2026-07-23,B2+Z1 全链已通)

狗端真实栈 = ROS2 action(`/b2_move_path` 直线 waypoint 导航、`/detect_grasp`、
`/gripper_control`、`/z1_move_ee`),server 把请求编成 `task=[Move, Pick]`。
已实测通过:导航→按名检测(返回 6DoF grasp_pose)→开爪→伸臂→合爪→抬臂→done。

**已定(2026-07-27 定稿)**:`target_world = [x, y, yaw]`——第三位就是偏航角,协议不传高度(狗端自调)。

**待对齐(每条一句话就能定)**:
1. `yaw` 单位与零轴:意图机发**弧度、板系 +x=0、逆时针正**——狗端确认或换算。
2. ~~站位 standoff 由意图机留~~ **已定(2026-08-05,v2)**:站位/避障狗端自留,
   意图机只发目标本体+建议方位;`dog_link._v2_stance` 是参考实现(沿建议方位退 0.6m)。
3. **检测类名词表**:`/detect_grasp` 只认检测器类名("苹果"被 goal reject,"orange" 通过)。
   意图机已有映射(`Intension/detect_names.json`,发送前地图名→检测名);
   狗端给一份支持的类名列表,填进这张表即完事。
4. ~~送达段未实现~~ **已实现(2026-08-18,skill=place)**:见技能表 place 行。
   `deliver_to`/`deliver_name` 字段废止——grasp 回归纯抓,送达一律走 place 链单
   (意图机编排:grasp done → place)。放置目的地词表现仅 "storage box"(棕色纸箱),
   映射在 detect_names.json(纸箱子→storage box)。
5. `stop` 抢占尚未在真机验证(`/b2_move_path` 需支持 cancel)。
6. 障碍:直线插值路径不会绕桌子/门,跨房 demo 前确认规划器如何避障。
7. **`object_hint` 投影选框(同类多实例消歧的最后一米,带干扰物实验前必须)**:
   意图机已随每单带上选中实例的板系质心 `object_hint:[x,y,z]`(2026-07-31 起)。
   **推荐实现(2026-07-31 定,与意图机共识)**:抓取相机同画面认桌面 tag →
   对 tags_world.json 一步 PnP 得 T_board_cam(**不经过基座定位/外参链,狗站哪
   都不影响精度**)→ hint 投影到图像 → **落在哪个 YOLO 框里抓哪只**。判定规则:
   点在框内→选之;不在任何框但离最近框心 <半框宽→选之;否则拒抓报
   `grasp_missed`(detail `hint_mismatch`)——宁可不抓不抓错;无 hint 维持现状。
   误差预算:桌面 tag PnP(1-2cm)+ 地图质心(cm)≈2-3cm ≪ 球半径,**密排间距
   也能分**。注意:PnP 用 ITERATIVE(tag 共面,SQPNP 断言崩);桌面 tag 必须
   经 survey 入 tags_world(现成:id=79 @ (+0.16,-2.73,0.85) rms 2.2mm;
   最后一次重建时在球桌再贴 1-2 张对角 tag 同场入镜即自动测绘)。
   备选(tag 不入画时兜底):用基座 tag 标定链把 hint 变换到相机系做 3D 就近
   匹配,阈值 0.3m。**参考实现已交付:`Intension/hint_select.py`**(load_tags +
   pick_box 三行接入,含无真机自检 --selftest;输入只需 tags_world.json + 请求里的
   object_hint + 他自己的相机内参与 YOLO 框——不需要意图机的模型/点云)。
8. **status/heartbeat 请把 `pose` 填上**(2026-07-30 实测全空):意图机三个等着用的
   场景——"回来"就近侧接近、原地抓用真实狗位、看狗校验。
