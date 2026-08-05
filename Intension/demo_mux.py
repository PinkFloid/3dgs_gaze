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
A = ap.parse_args()
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
peak = 1.0
wavs = sorted((SESS / "utt").glob("utt_*.wav"))
for w in wavs:
    t_end = float(w.stem.split("_")[1])
    with wave.open(str(w)) as f:
        assert f.getframerate() == SR and f.getnchannels() == 1
        pcm = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float64)
    dur = len(pcm) / SR
    off = t_end - dur - vid_start_wall
    i0 = int(off * SR)
    if i0 < 0 or i0 + len(pcm) > n_total:
        skipped.append((w.name, off))
        continue
    voice[i0:i0 + len(pcm)] += pcm
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
    i0 = int((t_wall - vid_start_wall) * SR)
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
lines = []
utt_dur = {round(float(w.stem.split("_")[1]), 2): wave.open(str(w)).getnframes() / SR
           for w in wavs}
for t_end, text in asr:
    d = utt_dur.get(round(t_end, 2), 1.5)
    a, b = t_end - d - vid_start_wall, t_end - vid_start_wall + 0.9
    if b < 0 or a > vid_dur:
        continue
    lines.append((a, f"Dialogue: 0,{ts(a)},{ts(min(b, vid_dur))},Speech,"
                     f"{{\\fad(120,120)}}「{text.rstrip('。')}」"))
for t_wall, obj, mode, goto in res:
    a = t_wall - vid_start_wall + 0.3
    if a < 0 or a > vid_dur:
        continue
    if goto or mode == "导航":
        txt = f"→ {obj} · 已派导航"
    else:
        txt = f"→ {obj}({mode})· 已派抓取"
    lines.append((a, f"Dialogue: 0,{ts(a)},{ts(min(a + 2.8, vid_dur))},Sys,"
                     f"{{\\fad(120,200)}}{txt}"))
for t_wall, state, rid in dones:
    a = (t_wall or 0) - vid_start_wall
    if 0 <= a <= vid_dur:
        txt = {"done": "✓ 任务完成", "failed": "✗ 失败", "stopped": "⏹ 已急停"}[state]
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
