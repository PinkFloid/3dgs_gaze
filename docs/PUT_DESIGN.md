# 逐词对齐指代 + 放置(put)——设计一页

**场景**:看着 A 说"把这个…",视线移到 B,接着说"…放到那里去" → 系统自动填
object=A、送达点=B,狗取 A 放到 B。

**一句话原理**:现在整句只有一个时间戳(说完时刻),三个槽位共用;升级成
**每个指示词各带自己的时刻**——"这个"出口的瞬间查 A 的注视,"那里"出口的
瞬间查 B 的落点。链路其余部分(三槽位解析、AttentionBuffer、PlaceBuffer、
deliver_to 派发)全部现成,零架构改动。

## 1. 词时刻从哪来(voice_input.py,~15 行)

`model.transcribe(..., word_timestamps=True)`(faster-whisper 自带强制对齐;
中文吐的是逐字/小词组 token,自测 CPU 开销 +20-50%,可接受)。
段起点墙钟 = t_end − 音频时长,词墙钟 = 段起点 + word.start。
队列条目从 `(text, t_end)` 扩成 `(text, t_end, words)`,
`words = [(字, wall_t0, wall_t1), ...]`;打字输入 words=None,一切照旧。

## 2. 指示词 → 时刻(纯确定性,LLM/缓存零改动)

LLM 只负责说"object_deictic=true / dest_deictic=true"(现状不变,缓存不升版);
**时刻由词表匹配从 words 流里找**,函数 `locate_deictics(text, words)`:

- 物指示词表:这个/那个/这只/这颗/这些;地点表:这里/那里/这边/那边/哪里
- 中文 token 常被切成单字 → 按**字符滑窗**在 words 流上匹配("这"+"个"相邻);
- 命中取词中点:t_obj = mid(这个),t_dest = mid(那里);同词多次出现取
  与槽位顺序一致的第一个(object 在前、dest 在后);
- **fallback 链(必须无感降级)**:找不到该词 / words=None / 打字输入
  → 该槽用句末 t_end(= 今天的行为,E1 语义不受任何影响)。

## 3. brain 消解(把 t_word 变成 per-slot,~20 行)

`handle(t_word, text)` → `handle(t_word, text, words=None)`;
cmd 增加 `t_obj / t_place / t_dest`(缺省都 = t_word)。
- object 视线绑定(binding / obj_gaze):用 **t_obj**;
- place 槽 slot_point:用 t_place;dest 槽 slot_point:用 **t_dest**;
- 绑定窗口从"只回看"改成 **[t−lookback, t+0.6]**:ASR 在整句说完后才到,
  流是完整的,允许小幅"向后看"——吃掉眼睛比嘴慢半拍的情形;
- **前向等待(dest 专属,用户实测的真实说法:"说完话目光才指过去")**:
  dest 为指示词且窗内无新鲜落点 → 不立刻失败,挂起本单等 **3s 内第一个
  新落点**(PlaceBuffer 出点即结算派发,提示音 ask 一声"看哪放哪");
  超时才报"没看到要放哪"。object 槽不等(物必须先看,防误绑)。
- 附赠:单指示词指令("拿一下这个")的 t 也换成词时刻,比句末更贴注视,
  E1 型指令的绑定精度免费+。

## 4. 派发与狗端(协议零改动或一行改动,取决于 put 的形态)

维持**唯一技能 grasp**:`object_name=A + object_hint + deliver_to=[Bx,By,yaw]`
——"把A放到B"在线上语义里 = 取A、送达B,`deliver_to` 本来就是干这个的
(R5 已回归)。同学的 put 接口两种可能:

- **形态 a(推荐)**:put = 他终于实现送达段(待对齐 #4):Pick 成功后追加
  Move(deliver_to)+ 放下。**协议零改动**,我们今天发的单他明天就能吃。
- **形态 b**:put 是独立原语 `put(x,y,yaw)`:他 server 收到带 deliver_to 的
  grasp 单,内部组合 [Move,Pick,Move,Put]。协议仍是一张单;
  只有当他坚持要意图机发两张单时才加 skill="put"(不推荐:两张单之间
  狗端持物状态要我们记账,违背 stateless brain)。

**要同学确认的三件事**:①接口签名与放置高度(桌面 z?地面?——建议第一版
只支持"放地上/放回物品台"两档);②放置失败态(placed_missed?);
③deliver_to 的 yaw 他用不用(不用就忽略,协议不改)。

## 5. 解析与缓存

- prompt 补一个例:"把这个放到那里去" → object_deictic=true, dest_deictic=true,
  action=fetch("放"不进抓取白名单,天然走 fetch+dest,语义正确);
- 首次说会走确认,确认后进 parse_cache,后续零延迟。

## 6. 回归 + 实验角度

- **R-新(双指示词逐词绑定)**:假流 = 盯 cup 3s(t 2-5)→ 盯落点(t 6-7);
  指令带 words:「把这个(t≈4.6)放到那里(t≈6.5)去」;断言 object=cup 且
  deliver_to≈落点。再来一条 words=None 的同文本,断言退化为今日行为。
- 论文角度:word-aligned deixis(逐词视线-语音共指)是干净的加分点,
  eye-voice span 有文献背书;可选 E5:双指示词放置任务成功率
  (对照:句末单时间戳 vs 逐词)。demo 视频5 第二幕直接用这个场景。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| whisper 对齐误差 ±0.1-0.2s | 窗口 [t−lookback, t+0.6] 远大于误差 |
| 幻听词/漏词 | 词表匹配不上 → 句末 fallback,行为=今天 |
| 两词间视线没挪(说得太快) | 退化成 dwell 竞争,=今天的行为 |
| 逐字 token 切分 | 字符滑窗匹配,不依赖分词 |

**改动量**:voice_input ~15 行,brain ~20 行,agent prompt 1 例,回归 1 组。
狗端零改动(形态 a)。
