#!/usr/bin/env python3
"""card_caller.py -- E1 卡片口播音频生成:不用背卡,手机播着跟读盯看。

    python Eye_Tracker/tools/card_caller.py --all          # 全部 12 张卡 -> docs/e1_audio/
    python Eye_Tracker/tools/card_caller.py --card s5      # 单张
    python Eye_Tracker/tools/card_caller.py --list

节奏(与卡片规则一致):报物名 → 盯 2.8s → 「叮」= 移开 → 1.5s → 下一项;
开头口播"开录,先盯墙上 tag 三秒",结尾"再盯 tag 三秒……停录"。
工作站无扬声器:产物是 .wav + .m4a,拷到手机播放即可(微信传自己/数据线)。
TTS 默认 Edge 晓晓神经语音(需代理网络,人声级;逐句缓存只请求一次),
--engine espeak 为离线兜底(机器人腔);网球念 一号/二号/三号球(斜位站
视觉左右会翻,数字名不带方位暗示)。
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_cards import CARDS, SPOKEN, SPOKEN_TITLE  # noqa: E402

LIB = "libespeak-ng.so.1"
AUDIO_OUTPUT_SYNCHRONOUS = 2
espeakCHARS_UTF8 = 1
espeakRATE, espeakPITCH = 1, 3

_buf: list[np.ndarray] = []
_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short),
                       ctypes.c_int, ctypes.c_void_p)


def _cb(wav, n, _ev):
    if n > 0:
        _buf.append(np.ctypeslib.as_array(wav, (n,)).copy())
    return 0


_cb_keep = _CB(_cb)  # 防 GC:espeak 持有回调指针


class TTS:
    def __init__(self, rate=150):
        self.es = ctypes.CDLL(LIB)
        self.sr = self.es.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0)
        if self.sr <= 0:
            raise RuntimeError("espeak_Initialize 失败(查 espeak-ng-data 是否在)")
        self.es.espeak_SetSynthCallback(_cb_keep)
        for v in (b"cmn", b"zh", b"zh-cmn", b"cmn-latn-pinyin"):
            if self.es.espeak_SetVoiceByName(v) == 0:
                self.voice = v.decode()
                break
        else:
            raise RuntimeError("普通话语音不可用(试过 cmn/zh/zh-cmn)")
        self.es.espeak_SetParameter(espeakRATE, rate, 0)
        self.es.espeak_SetParameter(espeakPITCH, 55, 0)

    def say(self, text: str) -> np.ndarray:
        _buf.clear()
        b = text.encode("utf-8")
        self.es.espeak_Synth(b, len(b) + 1, 0, 1, 0, espeakCHARS_UTF8, None, None)
        self.es.espeak_Synchronize()
        return (np.concatenate(_buf) if _buf else np.zeros(1, np.int16)).astype(np.int16)


class EdgeTTS:
    """微软 Edge 神经语音(晓晓)——需网络(走 socks 代理),音质人声级;
    逐句渲染进 outdir/.tts_cache 缓存,重复短语(数字+物名)只请求一次。"""

    def __init__(self, cache: Path, voice="zh-CN-XiaoxiaoNeural",
                 proxy="socks5://127.0.0.1:10808", rate="+0%"):
        import imageio_ffmpeg
        self.ff = imageio_ffmpeg.get_ffmpeg_exe()
        self.exe = str(Path(sys.executable).parent / "edge-tts")
        self.voice, self.proxy, self.rate = voice, proxy, rate
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.sr = 24000

    def say(self, text: str) -> np.ndarray:
        import hashlib
        import subprocess
        key = hashlib.md5(f"{self.voice}|{self.rate}|{text}".encode()).hexdigest()[:16]
        mp3 = self.cache / f"{key}.mp3"
        if not mp3.exists():
            subprocess.run([self.exe, "--proxy", self.proxy, "--voice", self.voice,
                            "--rate", self.rate, "--text", text,
                            "--write-media", str(mp3)], check=True, timeout=60)
        out = subprocess.run([self.ff, "-loglevel", "error", "-i", str(mp3),
                              "-f", "s16le", "-ac", "1", "-ar", str(self.sr), "-"],
                             check=True, capture_output=True)
        return np.frombuffer(out.stdout, np.int16)


def beep(sr, f=1200, ms=160, amp=0.5):
    t = np.linspace(0, ms / 1000, int(sr * ms / 1000), False)
    w = amp * np.sin(2 * np.pi * f * t)
    n = max(sr // 100, 1)
    w[:n] *= np.linspace(0, 1, n)
    w[-n:] *= np.linspace(1, 0, n)
    return (w * 32767).astype(np.int16)


def sil(sr, s):
    return np.zeros(int(sr * s), np.int16)


def render(key, tts, stare, gap, outdir):
    title, seq = CARDS[key]
    sr = tts.sr
    spoken_title = SPOKEN_TITLE.get(key, title)
    parts = [tts.say(f"{spoken_title}。开始录像。先盯墙上的 tag,三秒。"), sil(sr, 3.2), beep(sr), sil(sr, 1.0)]
    for i, nm in enumerate(seq, 1):
        parts += [tts.say(f"{i}。{SPOKEN[nm]}。"), sil(sr, stare), beep(sr), sil(sr, gap)]
    parts += [tts.say("最后。再盯墙上的 tag,三秒。"), sil(sr, 3.2), beep(sr),
              tts.say("完成,停止录像。")]
    pcm = np.concatenate(parts)
    wav_p = outdir / f"{key}.wav"
    with wave.open(str(wav_p), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())
    dur = len(pcm) / sr
    m4a = None
    try:
        import subprocess
        import imageio_ffmpeg
        m4a = wav_p.with_suffix(".m4a")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(wav_p), "-c:a", "aac", "-b:a", "64k", str(m4a)],
                       check=True)
    except Exception as e:
        print(f"[!] m4a 转码跳过:{e}")
    print(f"{key}: {title}  {len(seq)} 项  {dur:.0f}s  -> {wav_p.name}"
          + (f" + {m4a.name}" if m4a else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--card", help="卡号:e1-e5 / s1-s7")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stare", type=float, default=2.8, help="每项盯看秒数")
    ap.add_argument("--gap", type=float, default=1.5, help="移开间隔秒数")
    ap.add_argument("--engine", choices=("edge", "espeak"), default="edge",
                    help="edge=晓晓神经语音(需代理网络,人声级);espeak=离线兜底(机器人腔)")
    ap.add_argument("--edge-rate", default="+0%", help="edge 语速,如 -10%% / +15%%")
    ap.add_argument("--proxy", default="socks5://127.0.0.1:10808")
    ap.add_argument("--rate", type=int, default=150, help="espeak 语速(词/分)")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / "docs/e1_audio"))
    a = ap.parse_args()
    if a.list:
        for k, (t, seq) in CARDS.items():
            print(f"{k}: {t}  {len(seq)} 项")
        return
    keys = list(CARDS) if a.all else ([a.card] if a.card in CARDS else None)
    if not keys:
        ap.error("--card 无效(--list 看卡号)或用 --all")
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if a.engine == "edge":
        tts = EdgeTTS(outdir / ".tts_cache", proxy=a.proxy, rate=a.edge_rate)
    else:
        tts = TTS(rate=a.rate)
    for k in keys:
        render(k, tts, a.stare, a.gap, outdir)


if __name__ == "__main__":
    main()
