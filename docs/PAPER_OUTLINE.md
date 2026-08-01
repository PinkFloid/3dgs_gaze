# 论文概要(写作地面真值,2026-07-31)

> 目标:ICRA 2027(首尔),截稿 **2026-09-15**,8 页含引用;保底 RA-L 随投。
> 系统冻结 8/30。本文件是**给人和写作 Agent 共用的宪法**:声明口径、术语、
> 数字来源以此为准;与本文件冲突的表述一律不许写进论文。

## 0. 一句话

戴眼动仪的用户说一句带指代的话并看一眼目标("把这个网球拿来"),一只带机械臂的
四足机器人从房间外(无视线接触)走进来,取走**被注视的那一只**并送回用户——
指代消解发生在一张**人机共享的持久米制 3DGS 实例地图**中,而非机器人的相机画面里。

## 1. 声明(Claim,只许合取,不许拆开吹)

**三支柱合取**(novelty 检查 13:0 存活的形态,2026-07-19):

1. **可穿戴注视消解口语指代**——语言说不清同类多实例("那个网球"= 1/N 抽签),
   注视说得清;消解到**实例**级,物体无任何标记。
2. **消解发生在持久全局地图中,且指令时目标在机器人视野外**——机器人在门外,
   与目标无视线接触;地图中介使机器人视野无关紧要(E3 的不变性)。
3. **真实移动取物送回**——B2 四足 + Z1 臂,穿门进入、导航、检测、抓取、送达,真硬件。

**明确不声明的(负面清单,Agent 必须遵守)**:
- 不写 "cross-room/multi-room" —— 写 *the robot starts outside the room, behind a
  doorway, with no line of sight to the target*(按实测措辞);
- 不声明动态场景(静态地图假设,见 Limitations);不声明多用户(E5 可选才做);
- 主动意图推断(盯满问询)**不进声明**(被超越的踏脚石,最多一句 mention);
- 抢占/补发/原地抓/看哪去哪 = 工程便利,不进 contributions;
- 不声明"通用家庭机器人/开放词汇"——检测词表是有限类名集合。

## 2. 固定术语表(EN,Agent 不得另造)

| 概念 | 论文用词 |
|---|---|
| 共享持久地图 | persistent metric 3DGS instance map / shared world model |
| 注视指代消解 | gaze-based deictic reference resolution(实例级:instance-level)|
| 眼-声绑定窗 | eye-voice binding window |
| 锥后验投票 | cone-posterior object voting(over labeled Gaussians)|
| 板坐标系 | board frame(metric, z-up, defined by ChArUco calibration)|
| 角间隔 | angular separation(把间距×距离折叠为一根轴)|
| 最后一米消歧 | last-meter disambiguation(object hint + projection-based selection)|
| 视野外 | out of the robot's view / no line of sight |
| 槽位解析 | LLM slot parsing(text→slots only; deictics are flagged, never guessed)|
| 三通道指代 | object / place / destination slots, each fillable by name or gaze |

## 3. 结构骨架与图表

**Abstract** → **I. Introduction** → **II. Related Work** → **III. System** →
**IV. Reference Resolution**(方法核心)→ **V. Experiments** → **VI. Limitations**
→ **VII. Conclusion**。

图表清单(编号冻结,写作时引用):
- **Fig.1** teaser:一镜串联(用户注视三只同款球之一→狗从门外进来→抓对→送回),配时间轴;
- **Fig.2** 系统架构:眼镜/工作站/狗 三栏,共享地图居中;
- **Fig.3** 场地平面图:两空间、门、tag 分布、桌位、停放点(SHOOTLIST 视频6 的测量);
- **Fig.4**(**全文最硬**)消歧准确率 vs 角间隔,σ≈1° 竖线,基线 1/N 与最近同类启发式;
- **Fig.5** 定性条带:overlay 帧(注视十字+实例框+判定横幅)×若干场景;
- **Table I** E2 端到端:分阶段成功率 + 时延分解;
- **Table II** E4 消融:绑定窗/类别过滤/在线 bias 戳/票面阈;
- (可选 Table III)相关工作对照矩阵:in-view? markers? persistent map? mobile? real grasp?

## 4. Introduction 论证链(逐段)

1. 取物是移动操作的基本任务;自然指令里**语言天然欠指定实例**——"拿那个网球"在
   三只同款面前语义上就是 1/N。
2. 注视是人类消歧的天然通道(说指代词时正看着目标——eye-voice span 文献支撑)。
3. 已有注视-机器人工作两个前提之一:目标在机器人当前视野内(FAM-HRI 类),或
   物体贴标记(iBotAssistant 的 AprilTag)。两者都把"机器人此刻看得见什么"
   变成了系统边界。
4. 我们的关键一步:把消解的场地搬进**人机共享的持久米制 3DGS 实例地图**——
   人的视线在地图里求交、投票、选实例;机器人在同一坐标系里定位与执行。
   两个前提同时消失:目标可在视野外,物体无需标记。
4.5 **Why is this hard(显式一段,防"太简单/纯集成"指控的正面回答)**:
   视线通道 σ≈1°,3m 外即 5cm;同款实例间距 0.25m ⇒ 全链(gaze 标定/头定位/
   地图米制/漂移/末端检测)**每环必须厘米-亚度级**,任何一环松弛系统归零。
   逐环给"天真版的实证失败":佩戴内 2.6° 慢漂曾整段报废(→双戳+bias 按龄衰减);
   朴素球投票自信错判 84%(→锥后验);图像空间 fixation 抓不到边走边盯(→世界系
   聚类);类名检测最后一米丢消歧(→hint 投影选框)。**误差预算即论证,不用形容词。**
5. 贡献列表(编号,3-4 条):①实例级注视指代消解管线(锥后验+眼-声窗,含在线
   精度戳);②共享持久地图架构(建图→命名实例→双端定位→最后一米投影消歧);
   ③真机系统与预注册评测(角间隔曲线 + 端到端 + 视野外不变性 + 全回放消融)。

## 5. Related Work 分组与划界句

- **Gaze-based HRI**:选物/意图推断,均要求目标在机器人图像内 → 我们地图中介;
- **指令取物 / language grounding**:语言到地图/物体,通常无注视、或无实例消歧;
- **语义建图 / 开放词汇地图**:地图供机器人;我们的地图同时被**人的视线**查询;
- **可穿戴 + 机器人**:近邻逐个划界——iBotAssistant(注视选 AprilTag ID,非无标记
  实例);FAM-HRI(目标须在机器人视野);IntenBot(多房间为虚拟环境,真机无臂)。
- 素材:`docs/novelty_check_20260719/merged.md` 可直接当骨架。

## 6. System 节要点(数字见 §8)

眼镜端:Pupil Core,世界相机鱼眼标定;墙面 ArUco PnP 定位(三道门限);
片头片尾 tag 精度戳 + 在线 bias 重估(戳龄衰减)。
建图:手机照片→COLMAP(锁内参)→ChArUco 对齐米制板系→splatfacto(三 no-scale
flag)→SAM mask 提升到高斯→跨视角共识实例→**命名即合并**(名字=全栈主键)。
视线→实例:射线在 3DGS 渲染深度上求交(深度预言机)→世界系聚类成注视事件→
锥后验在带标签高斯上投票(输出 top-k + p_none)。
语音/语言:LLM 只做文本→槽位(object/place/dest 各 name|deictic + to_user),
**指代词只标记不猜**;确定性卫生层清洗 LLM 槽位错标;眼-声窗回看绑定。
机器人:B2+Z1;对同一套 tag 标定进板系;导航→到位按名检测→**投影消歧**
(抓取相机同帧认桌面 tag 一步 PnP,object hint 投影进图像,落在哪个检测框抓哪只,
不在框内拒抓)→抓取→送达用户(确认时刻头位姿)。
设计纪律(可写成一段落):急停旁路永不过 LLM;绑定/几何全确定性可回放;
全事件日志使每个 trial 可逐位复现(E4 的基础)。

## 7. Experiments(预注册纪律:指标与失败分类采集前定死,全记录全报告)

- **E1 消歧准确率(核心,离线,不用狗)**:N∈{2,3,5} 同款网球,准确率 vs
  **角间隔**(球排固定一次建图,走位扫距离,子集控 N);基线:纯语言 1/N、
  最近同类启发式;Wilson 95% CI,McNemar 对基线。**Fig.4 的数据源**。
  ⚠ **曲线必须测出"膝盖"**:走到 7m/0.25m 间距把角间隔压进 1-2°,让准确率在
  σ 竖线附近塌陷——"在几何预言处失效"是科学命题;全程 100% 的平线才是真·太简单。
- **E2 端到端**:狗自门外入,K≥25 trial,≥2 类目标;分阶段成功率(消解/导航到位
  <0.3m/抓取/送达)、时延分解(说完→dispatch→到位→抓稳→送达)、人工干预计数
  (0 干预才算 success);**失败七分类**:定位空窗/选错相邻实例/绑定窗错过/
  ASR 错/导航失败/抓取失败/通信超时。
- **E3 视野外不变性**:同任务,(a) 目标在狗视野内 vs (b) 狗在门外,各 10-12 trial,
  预期成功率无差——支柱二的量化形态。
- **E4 回放消融**:绑定窗起点/宽度、类别过滤开关、在线 bias 戳开关、票面阈扫描
  ——全部离线回放,零真机成本。
- 真值协议:指令卡定目标、位置-实例对应表拍照存档、系统选择=binding 日志实例 id。

## 8. 数字清单

**已有(可直接引用,注明出处)**:几何链端到端 ~0.1°(verify blend 0.93px/tag 处
4.6mm);gaze 层刚标定 1-2°,佩戴内慢漂 2.6°/段(戳协议压制);tag 测绘 RMS 毫米级
(桌面 tag id79 2.2mm);板拟合 RMS 0.48mm;v3 地图 336 图注册、SAM 623 实例;
深度求交 3ms、锥判定 4-6ms @TITAN X(快于实时);rec002 验收三狗全中零错误翻转,
趴姿狗中位票面 76%;投影消歧误差预算 2-3cm(tag PnP+地图质心)≪ 球半径。
**待采(写作时留 `[TBD-E1]` 等占位)**:E1 曲线全部数据点;E2/E3 全部率与时延;
E4 消融表;最终重建的地图统计(实例数/命名数/球间距实测)。

## 9. Limitations(主动写,别等审稿人)

静态地图假设(物体挪动即深度失效;变化检测 blend 只是预警);单用户单场景
(E5 跨用户可选);gaze 标定有效期=一次佩戴,靠戳协议维持;送达位置取确认时刻
用户位置(走动不追);检测词表为有限类名集合;同类消歧的最后一米依赖桌面 tag
可见(拒抓兜底,宁可不动)。

## 10. 给写作 Agent 的硬规则

1. **不得发明任何数字**:§8 之外的数字一律写 `[TBD-xxx]` 占位,不许编;
2. **不得升级声明**:§1 负面清单是红线;每个贡献句必须能映射到 §7 的某个实验;
3. 术语用 §2,首次出现给全称;引用一律 `[REF-主题]` 占位(如 `[REF-eye-voice-span]`),
   由人工/检索后补,不许幻觉 bibkey;
4. 语气:测量型(we measure / we observe)。**卖"能力+机制",少卖合取**:
   Abstract 说 unmarked, instance-level, out-of-view fetching via a shared
   persistent map;"to our knowledge, no prior system combines..." 全文至多
   出现一次(related work 划界处)。"为什么难"一律用误差预算回答,禁用形容词;
5. 系统事实以本仓库为准(PROTOCOL.md / REDESIGN.md / PIPELINE.md / CODE_TOUR.md);
   与本文件冲突时,本文件优先,并向作者报告冲突;
6. 中文括号注释是给作者的工作笔记,不是论文内容;
7. 八页预算(含引用):Intro 1、Related 0.75、System 2、Experiments 2.5、
   其余 1.75——超页先砍 System 细节(挪引用 CODE_TOUR 式附录/网站),不砍实验。
