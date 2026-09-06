#!/usr/bin/env python3
"""demo_mux.py -- 实验录像 + brain 会话 -> 带语音/提示音/字幕的第一人称 demo。

    python Intension/demo_mux.py ~/recordings/2026_08_05/004 Intension/logs/20260805-225227
    # 1) 若无 world_undistorted.mp4 先跑 export_undistorted.py(视线准星渲染)
    # 2) 会话 utt/*.wav 按文件名墙钟回放入轨,还原 heard/ok 提示音
    # 3) events.jsonl -> ASS 字幕(说的话/消解结果/任务完成),libass 烧录 + AAC 混流

对齐原理:utt 文件名 = 语音结束墙钟;录像墙钟 = world_ts - start_time_synced_s
+ start_time_system_s(info.player.json)。狗端 done 的 timestamp 与工作站钟差
在亚秒级,直接用。录像停止后的语音段自动丢弃(报告里列出)。"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description="拼第一人称语音+视线 demo")
ap.add_argument("recording", help="Pupil 录像目录(004)")
ap.add_argument("session", help="brain 会话目录(logs/2026...)")
ap.add_argument("--out", default=None, help="输出 mp4(默认 <录像>/demo_voice_gaze.mp4)")
ap.add_argument("--crf", type=int, default=19)
ap.add_argument("--lang", choices=["zh", "en"], default="zh", help="字幕语言(英文演示用 en:系统行英文、物名用英文对照)")
A = ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.en_names import gloss  # noqa: E402
EN = A.lang == "en"
name = (lambda o: gloss(o)) if EN else (lambda o: o)
REC, SESS = Path(A.recording).expanduser(), Path(A.session).expanduser()
OUT = REC
SR = 16000

und = REC / "world_undistorted.mp4"
if not und.exists():  # 视线准星渲染是唯一的重活,现成就复用
    tool = Path(__file__).resolve().parents[1] / "Eye_Tracker/tools/export_undistorted.py"
    subprocess.run([sys.executable, str(tool), str(REC)], check=True)

info = json.loads((REC / "info.player.json").read_text())
synced0, system0 = info["start_time_synced_s"], info["start_time_system_s"]
world_ts = np.load(REC / "world_timestamps.npy")
vid_start_wall = world_ts[0] - synced0 + system0   # 渲染视频 t=0 的墙钟
vid_dur = float(world_ts[-1] - world_ts[0])
p2w = lambda t: t - synced0 + system0              # pupil 钟 -> 墙钟

n_frames = len(world_ts)
fps = (n_frames - 1) / max(vid_dur, 1e-6)
def w2v(wall):
    """墙钟 -> 渲染视频时间轴。渲染按恒定 fps 铺帧,真实帧间隔有抖动,
    线性换算会攒漂移(本录 101s 处 +0.26s)——逐帧插值把每个墙钟时刻
    映到它真正落在的帧号。"""
    pupil = wall - system0 + synced0
    idx = float(np.interp(pupil, world_ts, np.arange(n_frames)))
    return idx / fps

asr, res, dones = [], [], []
for ln in (SESS / "events.jsonl").open(encoding="utf-8"):
    e = json.loads(ln)
    tp = e.get("topic")
    if tp == "asr":
        asr.append((e["t_end_wall"], e["text"]))
    elif tp == "resolution":
        res.append((p2w(e["t"]), e.get("object"), e.get("mode"), e.get("goto")))
    elif tp == "skill.status" and e.get("state") in ("done", "failed", "stopped"):
        dones.append((e.get("timestamp"), e.get("state"), e.get("req_id")))

# ---- 音轨:人声(按 utt 文件名 t_end 回放)+ 还原当时响过的提示音 ----
n_total = int((vid_dur + 1.0) * SR)
voice = np.zeros(n_total, np.float64)
placed, skipped = [], []
utts = []            # (t_end, pcm, dur) 给字幕锚定复用
peak = 1.0
wavs = sorted((SESS / "utt").glob("utt_*.wav"))
for w in wavs:
    t_end = float(w.stem.split("_")[1])
    with wave.open(str(w)) as f:
        assert f.getframerate() == SR and f.getnchannels() == 1
        pcm = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float64)
    dur = len(pcm) / SR
    off = w2v(t_end - dur)
    if not (0 <= t_end - dur - vid_start_wall <= vid_dur):
        skipped.append((w.name, t_end - dur - vid_start_wall))
        continue
    utts.append((t_end, pcm, dur))
    i0 = max(int(off * SR), 0)
    seg = pcm[:max(n_total - i0, 0)]        # 跨视频尾的段截断而不是整段丢
    voice[i0:i0 + len(seg)] += seg
    peak = max(peak, np.abs(pcm).max())
    placed.append((off, dur, w.name))

gain = min(0.70 * 32767 / peak, 40.0)  # 人声拉到 -3dB 附近,DJI 电平低,限最大增益
voice *= gain

def tone(seq):  # 与 voice_input.chime 同配方,采样率换 16k
    def seg(f, ms):
        t = np.linspace(0.0, ms / 1000.0, int(SR * ms / 1000), False)
        v = np.sin(2 * np.pi * f * t)
        n = SR // 100
        v[:n] *= np.linspace(0, 1, n)
        v[-n:] *= np.linspace(1, 0, n)
        return v
    return np.concatenate([seg(f, ms) for f, ms in seq])

CHIME = {"heard": tone([(880, 90)]), "ok": tone([(523, 90), (784, 90)])}
def put(buf, t_wall, kind, amp):
    if not (0 <= t_wall - vid_start_wall <= vid_dur):
        return
    i0 = int(w2v(t_wall) * SR)
    c = CHIME[kind] * amp * 32767
    if 0 <= i0 and i0 + len(c) <= n_total:
        buf[i0:i0 + len(c)] += c

for t_end, _ in asr:
    put(voice, t_end + 0.05, "heard", 0.18)     # 断句即响
for t_wall, _, _, _ in res:
    put(voice, t_wall + 0.35, "ok", 0.18)       # 派单成功音

mix = np.clip(voice, -32767, 32767).astype(np.int16)
with wave.open(str(OUT / "demo_audio.wav"), "wb") as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
    f.writeframes(mix.tobytes())

# ---- 字幕 ASS:底部=说的话,上一行=系统消解结果 ----
def ts(sec):
    sec = max(sec, 0.0)
    h = int(sec // 3600); m = int(sec % 3600 // 60); s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

hdr = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Speech,Noto Sans CJK SC,64,&H00FFFFFF,&H00000000,&H7F000000,-1,3,1,2,60,60,52
Style: Sys,Noto Sans CJK SC,48,&H0000FF00,&H00000000,&H7F000000,-1,3,1,2,60,60,150

[Events]
Format: Layer, Start, End, Style, Text
"""
def speech_island(pcm):
    """VAD 段可含数秒环境音(实测「OK」段 9.8s,词在段尾)——字幕锚到
    段内**最后一节连续语音**:30ms 帧能量 > max(峰值 25%, 底闸),
    往前回溯到 >0.4s 的静默为界。整段都是话的短指令不受影响。"""
    fr = int(0.03 * SR)
    m = len(pcm) // fr
    if m == 0:
        return 0.0, len(pcm) / SR
    e = np.sqrt((pcm[:m * fr].reshape(m, fr) ** 2).mean(1))
    idx = np.flatnonzero(e > max(e.max() * 0.25, 60.0))
    if not len(idx):
        return 0.0, len(pcm) / SR
    brk = np.flatnonzero(np.diff(idx) > int(0.4 / 0.03))
    s0 = idx[brk[-1] + 1] if len(brk) else idx[0]
    return s0 * 0.03, (idx[-1] + 1) * 0.03

lines = []
for t_end, text in asr:
    u = min(utts, key=lambda u: abs(u[0] - t_end), default=None)
    if u is None or abs(u[0] - t_end) > 0.2:
        continue                                  # 无对应入轨段(录像窗外)
    seg_start = t_end - u[2]
    s0, s1 = speech_island(u[1])
    a = max(w2v(seg_start + s0) - 0.15, 0.0)
    b = min(w2v(seg_start + s1) + 0.7, vid_dur)
    quoted = f"\u201c{text.rstrip('。.')}\u201d" if EN else f"「{text.rstrip('。')}」"
    lines.append((a, f"Dialogue: 0,{ts(a)},{ts(b)},Speech,"
                     f"{{\\fad(120,120)}}{quoted}"))
for t_wall, obj, mode, goto in res:
    if not (0 <= t_wall - vid_start_wall <= vid_dur):
        continue
    a = w2v(t_wall) + 0.3
    if EN:
        who = {"你这里": "you", None: ""}.get(obj, obj)
        modes = {"视线": "gaze", "名字": "by name", "主动": "proactive", "导航": "navigate", "放置": "place"}
        if goto or mode == "导航":
            txt = f"→ {name(who) if who != 'you' else 'to you'} · navigation dispatched"
        elif mode == "放置":
            txt = "→ place held object at gazed spot · dispatched"
        else:
            txt = f"→ {name(obj)} ({modes.get(mode, mode)}) · grasp dispatched"
    elif goto or mode == "导航":
        txt = f"→ {obj} · 已派导航"
    elif mode == "放置":  # 裸放置:没有 object(手里有什么放什么),别打 None
        txt = "→ 手里的放到注视处 · 已派放置"
    else:
        txt = f"→ {obj}({mode})· 已派抓取"
    lines.append((a, f"Dialogue: 0,{ts(a)},{ts(min(a + 2.8, vid_dur))},Sys,"
                     f"{{\\fad(120,200)}}{txt}"))
for t_wall, state, rid in dones:
    if not (0 <= (t_wall or 0) - vid_start_wall <= vid_dur):
        continue
    a = w2v(t_wall)
    txt = ({"done": "✓ Task done", "failed": "✗ Failed", "stopped": "⏹ Stopped"} if EN
           else {"done": "✓ 任务完成", "failed": "✗ 失败", "stopped": "⏹ 已急停"})[state]
    lines.append((a, f"Dialogue: 0,{ts(a)},{ts(min(a + 2.2, vid_dur))},Sys,"
                     f"{{\\fad(120,200)}}{txt}"))
lines.sort()
(OUT / "subs.ass").write_text(hdr + "\n".join(l for _, l in lines) + "\n", encoding="utf-8")

print(f"视频墙钟起点 {vid_start_wall:.3f},时长 {vid_dur:.2f}s")
print(f"人声 {len(placed)} 段入轨(增益 x{gain:.1f}),窗口外跳过 {len(skipped)} 段:")
for name, off in skipped:
    print(f"  跳过 {name} @ {off:+.1f}s")
print("\n时间轴(前 40 条):")
for a, l in lines[:40]:
    print(f"  {a:7.2f}s  {l.split('Speech,')[-1].split('Sys,')[-1]}")
print(f"\ndone/failed/stopped 状态事件 {len(dones)} 条")

# ---- 烧字幕 + 混音(imageio-ffmpeg 自带静态 ffmpeg,含 libass)----
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
final = Path(A.out) if A.out else REC / "demo_voice_gaze.mp4"
subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(und), "-i", str(OUT / "demo_audio.wav"),
                "-vf", f"ass={OUT / 'subs.ass'}",
                "-c:v", "libx264", "-crf", str(A.crf), "-preset", "medium",
                "-c:a", "aac", "-b:a", "128k", "-shortest", str(final)], check=True)
print(f"\n完成:{final}")
