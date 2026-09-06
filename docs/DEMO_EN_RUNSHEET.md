# 英文实时演示流程(2026-09-07,v10 地图,与 8-27 中文 demo 同一拍法)

管线已支持 `--lang en`:whisper 英文转写 + 英文热词、停词 `stop`、LLM 提示词带中英物名对照、
英文类别词(cup/ball/apple)映射到中文类过滤、demo_mux 英文字幕。地图名仍是中文,英文说法对照:

| 地图名 | 英文说法(口播就用这些词) |
|---|---|
| 网球L / 网球M / 网球R | left / middle / right tennis ball(指代时只说 "this tennis ball") |
| 苹果红 / 苹果粉 | red apple / pink apple |
| 香蕉 / 橘子 | banana / orange |
| 白杯1 / 白杯2 | white cup 1 / white cup 2(指代时说 "this cup") |
| 物品台 | table |

## 1. 开机顺序(四个终端)

```bash
# ① Pupil Capture:开 Frame Publisher 插件;眼动标定;世界相机 1920x1080;录制键 R(先别按)
# ② 感知
Eye_Tracker/tools/gaze_live.sh --publish 5581
# ③ 狗端二选一
#    真狗:同学的 server(5583 命令 / 5584 状态),brain 用它的地址
#    假狗(没狗也能出片,✓ Task done 字幕照出):
conda run --no-capture-output -n nerfstudio python Intension/dog_link.py --fake
# ④ 大脑(英文 + 语音 + 免确认)
conda run --no-capture-output -n nerfstudio python Intension/brain.py --lang en --voice --yes \
    --skill-endpoint tcp://127.0.0.1:5583          # 真狗换成 tcp://<狗机IP>:5583
```

brain 终端要有代理变量(`https_proxy=http://127.0.0.1:10808`),台词没进缓存的那句才走 LLM(约 2 s);
下面台词已全部预热进 `parse_cache_v2.json`,现场 0 ms。DJI 无线麦要插好(`--voice-device Rx` 缺省)。

## 2. 录前三查(各 30 秒)

1. **视线**:站 2 m 正对,盯中间球,gaze_live 窗口判定应是 网球M;盯白杯1 应是 白杯1。
2. **英文听写**:`conda run --no-capture-output -n nerfstudio python Intension/voice_input.py --once --lang en`,
   说 "Bring me this tennis ball." 看转写;说 "Stop." 看转写是不是 stop。
3. **急停**:brain 开着说 "Stop.",终端应立刻打印 skill=stop 派发(不过 LLM)。

提示音:每句话说完 0.6 s 静音 → 「叮」= 听到;双音 = 已派发/狗 done;低音 = 失败。
`--yes` 免确认;若不加 `--yes`,升调后说 "Yes." / "Okay." 确认,"No." 取消。

## 3. 台词(按 8-27 四幕改的英文版;先盯 1–2 s 再开口,说完继续盯到叮)

| 幕 | 看哪里 | 说什么 | 系统应做 |
|---|---|---|---|
| 1 | 盯**中间**的网球 | **Bring me this tennis ball.** | 视线绑 网球M → 抓取 → 送到你面前(两单) |
| 2 | 盯粉苹果 | **Grab this apple.** | 视线绑 苹果粉 → 抓取(不送) |
|   | 盯桌上一块空处 | **Put it here.** | 放到注视落点 |
| 3 | 盯白杯1 | **Grab this cup.** | 视线绑 白杯1 → 抓取 |
|   | 看狗 | **Give it to me.** | 携物导航到你身边 |
|   | 狗走到一半 | **Stop.** | 急停(不过 LLM,旁路) |
|   | — | **Come here.** | 继续过来 |
| 4 | 不用看 | **Grab the orange.** | 按名字 橘子 → 抓取 |
|   | 不用看 | **Put it on the table.** | 放到物品台 |
| 5 | — | **Come back.** | 回到你身边 |

可选:盯左边球说 "Go there."(纯导航到注视点);"Grab this red apple."(名字压过指代,不盯也拿对)。
别说的:光说 "Bring me this ball." 又没盯着 → 系统会要你先看一眼(三只球都候选)。

## 4. 录像与出片

- Pupil Capture 按 R 开录 → 全部台词 → 按 R 停录。录像在 `~/recordings/<日期>/NNN`,改名 `demo_en_1`。
- brain 会话目录在终端结尾打印(`Intension/logs/<时间戳>`),语音段 WAV 在其 `utt/`。
- 出片(准星渲染 + 人声/提示音入轨 + 英文字幕烧录):

```bash
conda run --no-capture-output -n nerfstudio python Intension/demo_mux.py ~/recordings/<日期>/demo_en_1 Intension/logs/<时间戳> --lang en
```

产物 `<录像>/demo_voice_gaze.mp4`;字幕:底部白字 = 你说的话,上一行绿字 = `→ middle tennis ball (gaze) · grasp dispatched` / `✓ Task done`。
剪辑版做法同 8-27:复制会话目录删掉不要的 asr 事件再出片。

## 5. 已知边界

- 视线判定用的是 v10 地图;桌上东西一毫米别挪。
- 英文 "grab" 与 "bring me" 都派 grasp,区别只在送不送到你身边;"put it …" 一律当手里有东西的放置。
- 停词只认 stop(连喊、"stop it" 都算);置信闸照旧,狗行走噪声下的幻听会被丢弃并打印原因。
- 回归:`./Intension/run_regression.sh`(中文 23 项)全绿;英文台词在 v6_near 视线回放上全过(2026-09-07 01:40)。
