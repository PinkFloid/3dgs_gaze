# 架构 v2:薄大脑、抢占狗、共享地图定位

> 2026-07-28 设计稿(待过目)。配套阅读:`Intension/PROTOCOL.md`(v1 现行)、`docs/CODE_TOUR.md`。
>
> 起因三个症状:① brain 526 行里近半是忙碌记账/覆盖排队/120s 看门狗/手动重置,全是在绕
> 协议的"忙时拒单";② "去这个地方拿X"盯着地板说必失败——地板注视被物体门挡在指令消解
> 之外(brain.py:417 的视线地点分支只吃 AttentionBuffer,而 attention.py:15 的门只放
> object_label≥10);③ 狗 pose = 停放点静态外参 + 里程计硬算,漂移且与板系没有可靠对齐。
> **三个症状同一个根:分工错位——大脑替狗管执行状态、替狗算站位,狗却不管自己在哪。**

## 0. 目标分工

**brain = 感知与意图**:视线缓冲 + 语音/文本→槽位 + 名字消解 + 确认门,产出一张带坐标的
任务单,发完即忘。**dog = 执行**:自己定位、自己算站位避障、自己管抢占;进度是广播,不是握手。
**协议 = 任务单 + 最新优先**:新任务永远抢占旧任务;急停 = 抢占为"空任务"。

| 职责 | v1(现状) | v2 归属 |
|---|---|---|
| 忙/闲、排队、卡死解锁 | brain(dog_busy / queued / 重置 / busy-timeout) | **删除**——狗端抢占后无此概念 |
| 站位 standoff + 家具避障 | brain stand_pose + instances 占地表 | 狗端(规划器份内事) |
| 送达站位 | brain 算 deliver_to 站位 | brain 填 deliver_to **落点**,站位狗端算 |
| 狗定位 | 停放外参 + 里程计 | 狗端锚定 + 里程计(§4) |
| 指代→坐标 | 只认物体注视 | 双槽位:object 槽 + place 槽(§2) |
| 文本→结构 | agent.py(LLM+缓存) | 不变,schema 扁平化 |
| 名字→检测类名 | brain detect_names.json | 不变;get_state 回detector 词表供早警 |
| 急停 | 旁路永不过 LLM | 不变(神圣不可侵犯) |

## 1. 协议:抢占语义(核心变更;wire 仍 v1)

**冻结定义:狗端收到新 grasp/goto 时,若在忙,立即回执 accepted,给旧 req 广播
`stopped`(detail=`preempted_by:<new_req_id>`),取消动作后执行新单。** busy 拒单从
reason 词表删除。安全语义沿用急停:站定 + 夹爪冻结不松开,cancel 完成才起新任务。
版本策略(实现时修正):消息形状零变化,线上版本**不升**——否则新 brain 对还没升级
的真实 server 会被 unsupported protocol 拒掉,联调直接断;这是服务端行为修订,
v2 编号留给步 3 的形状变更。旧服务器忙时仍回 busy,
新 brain 把它当"没抢占成功,本单作废"提示,不排队不重试。
**状态(2026-07-28)**:抢占参考实现已写完并回归全绿,但按用户方针**暂不启用**
(狗端零改动),归档 `Intension/archive/dog_link_preempt.py`;现行行为 = 狗端忙拒、
brain 作废提示。brain 端的"发完即忘"对两种服务端行为都成立,狗端何时升级都零改动。

- REP 回执仍 <100ms:取消是异步的。新 worker 先等旧 worker 退出(上限 5s;旧 worker 的
  各阶段看门狗保证会退)。等不到 → 新 req 广播 `failed`(detail=`preempt_timeout`),
  **绝不双控制器并跑**。
- 前置条件 = 急停前置条件,同一个:`/b2_move_path` 等 ROS2 action 支持 cancel
  (PROTOCOL 待对齐 #5)。抢占和急停是同一条代码路径,做一个送一个。
- brain 侧因此删除:`dog_busy()`、`queued` 排队补发、`重置`命令、`--busy-timeout`、
  comms 的终态粘滞逻辑(它只为忙碌记账服务;状态流回归纯显示+日志)。
- brain 加一个 1 行防抖:同一目标(同名或落点 <1m)不重复下发,防注视抖动刷单。
- 为什么"最新优先"是安全的:所有下发都过确认门(y / --yes 是显式选择),
  抢占 = 用户最新意志优先,不是机器自作主张。

**站位语义搬家(升 v2,字段换新名避免语义撞车)**:`params.target = [x, y]` 指**目标物/
目的地本体**(不再是站位;高度继续不传)。`params.deliver_to = [x, y]` **保留且显式**,
同样降为落点语义:送达目的地本体(用户位置,或"拿到Y那边"里的 Y),**缺省 = 原地 done
(v1 冻结语义不变);送到哪永远写在请求里,不从别的字段推断**。新增可选
`params.user_pose = [x, y, yaw]`(确认时刻用户头位姿,yaw 取注视方位角),只当接近方向
先验用(站到用户可见的那一侧),与是否送达解耦——否则表达不了"拿着别动"(有 user_pose
无 deliver_to)和"拿到桌子那边"(deliver_to=桌子,非用户)。狗端负责:standoff 站位与
家具避障(target 与 deliver_to 走同一套站位函数:落点前 0.6m、yaw 朝向落点)、
`object_name` 非空时到位检测抓取、有 deliver_to 则抓完走送达站位。已知简化:brain 填
deliver_to 用确认时刻的用户位置,用户中途走动不追(房间尺度可接受,记入失败分类)。

## 2. 消解:双槽位模型(治"去这个地方拿X")

每条指令解析为两个槽位,每个槽位独立走"名字或视线"两条消解路:

| 槽 | 语音给名字 | 语音给指代词 | 特殊值 |
|---|---|---|---|
| **object**(拿什么) | 查物体表→透传检测类名;不在表也照发(到位检测兜底) | 眼-声窗口内**物体注视**候选(现 AttentionBuffer 语义,不动) | null = 纯导航 |
| **place**(去哪拿) | 查物体表取质心 | 眼-声窗口内**最近注视落点**——物体表面实际落点,未命名实例算数;地板/墙不算(2026-07-28 用户裁定:噪声大且无需求,"去哪"看目标处的物体/家具表达) | user = 来用户身边 |
| **destination**(送到哪) | "拿到Y那边"→查表取 Y 质心 | 说"那边"时的最近注视落点(复用落点通道) | "拿来/带来"=用户位置;全无 = null(抓住不送,原地 done) |

- attention 层新增**落点通道**:所有注视事件的世界落点(`centroid_world`,即视线实际
  打到的点,不是物体质心),驻留 ≥0.4s 才入册防扫视噪声。物体通道原语义不动——
  **E1 的绑定实验不受重构影响**(预注册纪律)。
- place 槽取"落点"、object 槽取"物体质心",两者刻意不同:"这个地方"要的是你看的那一点。
- `--goto-floor / floor_buf / VisitTracker 网格分桶`整套删除,不再有地板注视功能
  (用户裁定);看哪去哪保持物体版(AttentionBuffer 盯满触发),排队机制消失。
- parse_schema 扁平化为 `action(fetch/goto/stop/none) + object_query + noun_class +
  place_query + place_deictic + object_deictic + dest_query + dest_deictic + to_user`;
  `去这个地方拿X` → `fetch + object_query=X + place_deictic=true`;`把这个拿到桌子那边` →
  `fetch + object_deictic=true + dest_query=桌子`。缓存文件升版(parse_cache_v2.json),
  demo 台词重新预热;eval_binding 加"盯地板取物"与"送到指定地点"用例。
- ⚠ 一句话里两个指代词("把这个拿到那边"):打字指令只有整句一个时间戳,绑定规则=
  object 槽取最近**物体驻留**、dest 槽取最近**落点**——先看物体后看目的地的自然语序下
  两者各得其所;反序会绑错,记入失败分类。根治要等语音上词级时间戳(faster-whisper
  给得出,每个指代词绑各自时刻的注视)。
- brain `handle()` 的六分支塌缩为两个 resolver(先填 place,再填 object,发送)。

## 3. brain 删除/保留清单

**删**:dog_busy、queued+空闲补发、重置命令、busy-timeout、终态粘滞、stand_pose+boxes
占地表、standoff 参数、送达**站位**计算(deliver_to 落点本身照填)、floor_buf 三 flag
(估 -180 行,brain ≤350 行)。
**留**:AttentionBuffer(+落点通道)、resolve 名字消解、CommandParser、确认门+suppress、
急停旁路、events.jsonl 日志、--replay/--script 回归、物体表热加载。
**归档**:stare_to_grasp.py(被 brain 全面取代,PROTOCOL 联调示例改指 brain)。

## 4. 狗定位:锚 + 里程计,共享 3DGS 地图

**冻结公式:`T_world_base = T_world_odom ∘ T_odom_base`。** 里程计连续跟踪(已有),
锚定修正 `T_world_odom`(新)。锚定时机:接单时、到位检测前、途中每 ~2m 机会性一次;
锚龄超限+里程超限 → status 报 `unlocalized`(词表已有)。心跳与 status 的 pose 一律
报世界系,`frame` 校验从"收到即忽略"改为真比对(统一 frame 缝就此封死)。

锚从哪来,三条路线(按 bring-up 顺序,不互斥):

| 路线 | 做法 | 依赖 | 定位 |
|---|---|---|---|
| **A. 狗头 ArUco** | 狗相机认墙面 tag,PnP 对 tags_world.json——就是 pupil_localizer 的活,连三道门限都照抄 | 狗端 OpenCV + tag 入画 | 一两天通,demo 保底 |
| **B. 3DGS 渲染锚定** | 工作站开 `loc.anchor` REP(:5585):狗发 {jpg, K, 位姿猜测} → 在猜测位姿渲 RGB+ED → 特征匹配真图↔渲染图 → 匹配点用**渲染深度**(米制板系)抬到 3D → PnP → 回 T_world_cam + 内点数。猜测差时渲 8 个 yaw 扇面重试,再不行退路线 A | 工作站 TITAN X(gsplat 环境现成);锚定级延迟(几百 ms)够用 | 无 tag 依赖、全屋可用;**论文点:人与狗共享同一 3DGS 世界模型定位** |
| **C. LiDAR 地图 ICP** | 若狗导航栈自带建图定位:一次性把它的地图 ICP 对齐 splat.ply,得固定 T_board_slam,狗定位直接权威 | LiDAR 存在待确认 | 若有,最省最稳 |

⚠ 路线 B 的已知风险:建图照片拍摄高度 0.93–2.03m,狗头相机 ~0.5m,低机位渲染质量和
匹配会打折——**下次重建补拍一圈低机位照片**(成本≈0);光照变化与挪动过的家具同理
(verify 的 blend 重影 = 免费预警)。验收:verify_pose_render 式 blend 交叉校验 +
E2 的"到位站位残差 <0.3m"同源指标。

## 5. 迁移顺序(每步独立可回归)

| 步 | 内容 | 回归口径 | 封的缝 |
|---|---|---|---|
| 1 ✅* | brain 删记账(狗端不改:忙拒=作废提示,wire v1 原样) | --replay 全绿;busy 拒单出作废提示——**2026-07-28 已过**。*抢占 dog_link 已实现并全绿(连环抢占/急停/版本闸)但**未启用**,归档 archive/dog_link_preempt.py:用户裁定狗端零改动,同学做 cancel 时可整体采纳,brain 届时零改动 | — |
| 2 ✅ | 三槽位 + 落点通道(物体表面落点,无地板)+ schema/缓存升版 | **2026-07-28 已过**:注视物品台"去这个地方拿橘子"/goto 落点/显式 dest/盯地板拒绝/E1 打分兼容 | — |
| 3 | 站位/避障搬狗端(协议 v2,target/deliver_to 降为落点);tools 出 2D occupancy(splat 地板切片+实例占地)喂规划器 | 假狗:站位不进家具、送达朝向用户 | standoff、deliver_to(站位侧) |
| 4 | 定位:路线 A 通 → 路线 B 上;frame 真校验 | blend 校验;到位残差 <0.3m | 统一 frame |
| 5 | get_state 回 detector 词表,brain 启动即对表早警 | 未映射名字启动时报,不再运行时踩 | 名字映射 |

步 1-2 是 E2 的关键路径且不依赖狗机同学;步 3-5 与狗机侧并行。⚠ 缝 4 的送达段
**不必等步 3**:狗端现在就能按 v1 形状实现(Pick 成功后 Move 到 deliver_to 并转向),
步 3 只把"直接走到"换成"自算站位",复用他给 target 写的同一个站位函数。
截稿 2026-09-15,步 1-2 本周可完,步 4 路线 B 是论文增量,demo 以 A 保底。

## 6. 不变量(重构护栏)

LLM 只做文本→结构;急停旁路永不过 LLM 且最高优先;REQ/REP+PUB 拓扑与 msgpack 不动;
events.jsonl 事件流字段只增不改(E4 回放消融依赖它);E1 物体绑定语义分毫不动。

## 7. 导航设计(2026-07-29 定稿;步 3 的 occupancy 即为此服务)

**三层分工,狗端只需一句话改动:**

```
brain 全局规划(3DGS→occupancy→A*→拐点)      ← 新增,全在工作站
   │ waypoints:[[x,y],…](协议加字段=兼容,旧狗端自动忽略)
狗端直线段跟随(/b2_move_path 现有能力,零新依赖)
   │ 到位 ±0.3m 级即可
末端检测闭环(/detect_grasp 6DoF,已存在)      ← 兜住终端误差
```

- **occupancy 从 3DGS 来**(`SceneRebuild/tools/export_occupancy.py`,换图重跑):
  splat.ply 取 z∈[0.15, 1.0](狗+臂高度带,可调)切片 → 5cm 栅格 → 形态学去噪
  (3DGS floater)→ 并上 instances 占地 bbox → 膨胀 0.35m(B2 半宽+安全边)→
  occupancy.npz(网格+原点+分辨率+地图版本)+ 预览 PNG 人工验收。
  墙/门/走廊天然在图里,不用额外标注。
- **规划在 brain**:A* (8 邻接,~180×200 格,毫秒级)+ 视线缩点 → ≤N 个拐点;
  起点=狗回传 pose(订阅 dog.heartbeat,brain 存最新位姿),终点=站位。
  **无路 → 拒绝派发**(宁可不动);狗位姿未知/过期 → 退化为只发终点(现状直线)+ 警告。
  门口段拐点垂直穿门中线,少斜穿窄口。
- **协议**:`params.waypoints` 可选字段。按演进规则加字段=兼容:同学的 server 把
  waypoints 透传给 `/b2_move_path`(他的 action 本来就吃 waypoint 列表,只是把
  "起点→终点直线插值"换成透传)——**狗端全部改动就这一句**;没改之前旧行为不变。
- **误差预算**:跨房 ~10m 里程计漂移 1-5% → 10-50cm,吃在膨胀余量+门宽里,
  末端由检测闭环修正——导航只需"把狗送进检测范围"。中期(步 4 锚定)把这项压掉。
- **归属声明**:规划器放 brain 是"狗端零改动"约束下的落点;接口(waypoints 可选)
  设计成狗端将来长出自己的规划器时直接忽略该字段即可,协议不用再动。
  brain 侧 `stand_pose` 的家具避让届时被栅格终点合法性检查自然取代。
- **落地顺序**:① exporter+planner+路径可视化(不动协议,规划结果先只画)→
  ② dog_link(模拟)与协议加 waypoints,假狗回归 → ③ 同学一句话接入 →
  ④ 真机跨房空跑(不抓取)验门口通过率。
