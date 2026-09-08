# ICRA 演示视频剧本(v1,2026-09-09;只展示结果,四幕 + 收尾,目标 2:45,上限 3:00)

> 素材:`~/recordings/2026_09_08/{007,008,010}/demo_voice_gaze_clean.mp4`(第一人称 + 视线准星 + 英文字幕,已静音无效语句,时间轴与原录像一致),
> 机器人小窗 `…/outputs/robot_demo/*_RGB_aligned.mp4`(你重新对齐后的版本),v11 地图渲染图(2 m 正对、斜前方各一张,scratchpad 里有)。
> 全片英文字幕/字卡,不配旁白也能看懂;若配旁白,用下面"VO"那行。快进段右上角打 "4×" 角标,说话与绑定的瞬间一律 1×。
> **网球一幕只用 007**(三只球三次都是日志真绑对的);010 开头那单日志是 M,改标 left 的版本别拿来证明消歧。

## 第 0 幕 片头(0:00–0:08)

| 画面 | 字卡 |
|---|---|
| v11 地图渲染图 → 叠第一人称画面(准星停在球上)| 论文标题 / 作者 / 单位;一行副标:*Say "this", look at it, and a robot that cannot see it fetches it.* |

## 第 1 幕 问题(0:08–0:32)

| 时间 | 画面 | 字卡 / VO |
|---|---|---|
| 0:08 | 第一人称:桌上三只一模一样的网球,准星在中间那只上;人说 "Bring me this tennis ball"(007 的 1:50 那句,只留语音,不留派发字幕)| **"This" is ambiguous to a robot** — three identical balls, and the robot is not even looking at the table. |
| 0:16 | 机器人小窗单独放大:狗的相机里只有墙和门(用 008/010 小窗开头狗在走廊时的画面)| **The robot cannot see what you see.** No shared viewpoint, no pointing gesture. |
| 0:24 | v11 地图渲染图上叠三条信息:眼动仪定位(头位)、视线射线、命中的实例框(可用现成的 fig_pipeline / overlay 截图)| **Our answer:** a persistent 3DGS instance map shared by the eye tracker and the robot. Gaze picks the instance in the map; speech says what to do. |

VO(可选):"Deictic words like *this* only make sense if the listener sees what you see. Our robot does not. Instead, both the eye tracker and the robot are localized in one persistent 3D Gaussian Splatting map, so a gaze can name an instance the robot has never looked at."

## 第 2 幕 三只网球:无需共视角、同款消歧、眼神指代(0:32–1:22)

全部来自 **007**。每次:说话 + 绑定字幕 1×(约 5 s),狗导航/抓取 4×(约 6 s),送达 1×(约 3 s)。

| 时间 | 007 片段 | 速度 | 字卡 |
|---|---|---|---|
| 0:32 | 0:15.8–0:21.5 说 "Bring me this tennis ball",字幕 "→ left tennis ball (gaze) · grasp dispatched" | 1× | **Trial 1 — the left one.** 小窗:狗此时在门口/走廊 |
| 0:38 | 0:21.5–0:58.6 狗走到桌前、抓、送到手上(两个 done)| 4× | 角标 4×;done 出现时切 1× 半秒 |
| 0:47 | 1:01.7–1:07.0 说 "Grab this tennis ball" → right tennis ball | 1× | **Trial 2 — the right one.** 同一句话,不同的注视 |
| 0:52 | 1:07–1:24.6 抓取;1:24.4–1:29 "Bring it to me" → 1:45.6 done | 4×,递送段 1× 收尾 | **"Bring it to me"** — the user's own position comes from the same map |
| 1:04 | 1:50.0–1:54.1 说 "Grab this tennis ball" → middle tennis ball | 1× | **Trial 3 — the middle one.** |
| 1:09 | 1:54–2:09.8 抓取 | 4× | |
| 1:14 | 定格三只球的画面,叠三次绑定的字幕 | 静帧 4 s | **Same words, three different balls.** Disambiguation comes from gaze in the map, not from the robot's camera. (可加一行:*instance-level: 3 identical balls, 24 cm apart*)|

## 第 3 幕 名字指代;类别词 + 眼神辅助消解(1:22–1:58)

| 时间 | 片段 | 速度 | 字卡 |
|---|---|---|---|
| 1:22 | 008 0:57.2–1:02.6 "Grab the red apple" → red apple (by name);人可以看别处(准星不在苹果上更好)| 1× | **A name needs no gaze.** "the red apple" resolves against the map's object table. |
| 1:28 | 008 1:02.6–1:19.7 抓取 done | 4× | |
| 1:33 | 008 2:04.7–2:09.5 "Bring me this cup" → white cup 1 (gaze, nearest);画面里两只一模一样的白杯 | 1× | **A class word + gaze picks the instance.** Two identical white cups; "this cup" + gaze → cup 1. |
| 1:39 | 008 2:09.5–2:24.6 抓取 + 送达 | 4× | |
| 1:46 | 008 0:09.9–0:14.8 "Grab this apple" → pink apple(gaze),桌上有红、粉两只苹果 | 1× | **"this apple"** among a red and a pink one → the pink one you looked at. |
| 1:52 | 008 0:14.8–0:36.7 | 6× | |

(若时长紧,第 3 幕留"red apple 按名"和"this cup 眼神+类别"两条即可。)

## 第 4 幕 眼神指定放置地点(1:58–2:30)

| 时间 | 片段 | 速度 | 字卡 |
|---|---|---|---|
| 1:58 | 010 0:44.3–0:49.2 手里拿着球,看着地上的纸箱说 "Put it here" → 字幕 "place held object at gazed spot" | 1× | **Where is "here"?** The gaze landing point in the map becomes the destination. |
| 2:03 | 010 0:49–1:00.5 狗走到箱子边放进去,done | 4× | 小窗:狗相机里的箱子 |
| 2:09 | 007 2:10.9–2:16 "Put it in the box" → 按名字送达 | 1× | **Or name the place:** "in the box" → the box instance. |
| 2:14 | 007 2:16–2:27.2 放置 done | 4× | |
| 2:20 | 008 0:37.1–0:41.6 "Put it in the box"(苹果放进箱子)| 1×,后 4× | 可选;不够时长就删 |

## 第 5 幕 急停与收尾(2:30–2:45)

| 时间 | 片段 | 速度 | 字卡 |
|---|---|---|---|
| 2:30 | 010 1:46.6–1:50.1 狗走动中说 "Stop" → ✓ Stopped,狗停住 | 1× | **"Stop" bypasses everything** — no parsing, no confirmation, ~2 s from speech end to halt. |
| 2:35 | 010 1:51.2–1:56.4 "Bring it to me" → 狗继续过来递到手上,2:10.4 done | 1× 起,4×,交接 1× | |
| 2:41 | 尾卡 | | 三行:*One persistent 3DGS map · gaze + speech deixis · no shared viewpoint*;项目主页/二维码;致谢 |

## 需要另外准备的东西

1. **片头/尾卡 + 第 1 幕的三张静帧**:论文标题;狗相机"只见墙门"截图(小窗素材里截);地图上"头位–射线–实例框"示意图(可用 docs/E1_DATA/fig_pipeline.png 或我用 v11 渲染一张带射线的)。
2. **小窗对齐版**:你重新对齐后的 007/008/010 画中画;若来不及,只有 007 一幕用小窗,其余用第一人称全屏也成立。
3. **一段 3–4 s 的"三次绑定"定格**:三只球画面上叠三条字幕(第 2 幕结尾),我可以用 ffmpeg 直接做。
4. **字幕风格统一**:烧进去的绿色系统行已经是英文;新加的字卡用同一字体(Noto Sans/DejaVu),白字黑边。
5. **时长核对**:按上表 1× 段约 75 s、快进段折算约 60 s、字卡 30 s,合计 2:45;ICRA 附件时长上限按当年 CFP 为准,通常 3 min 内最稳。

## 剪辑口径(写在这里免得忘)

- 只用 `_clean` 版本做源;时间码以它为准(与原录像一致)。
- 每个 1× 段包含:说话开始前 0.5 s → 派发字幕出现后 1.5 s。
- 快进用 setpts,音频段静音(快进段本来只有狗的动作,没有语音)。
- 不要把 010 开头改标的那单放进"消歧"叙事里。
