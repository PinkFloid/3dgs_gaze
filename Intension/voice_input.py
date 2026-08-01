#!/usr/bin/env python3
"""voice_input.py -- 麦克风 -> VAD 断句 -> faster-whisper 转写(CPU int8)。

brain 用法:`brain.py --voice`(内部起 VoiceReader 线程,转写结果进指令队列;
t_word 用"说完时刻"回填——VAD 尾判 + 转写的 1-2s 延迟不歪眼-声回看窗)。
单独试麦(不动 brain):
    python Intension/voice_input.py --once      # 说一句,打印转写与耗时
    python Intension/voice_input.py             # 连续模式,Ctrl-C 退出

设计:16kHz 单声道;webrtcvad 30ms 帧,语音 >=0.3s、静音 0.6s 判结束,
前导 240ms 环形缓冲防吃句首;whisper CPU int8(TITAN X sm_52 不碰,
GPU 留给 gaze_live);initial_prompt 每句现喂物体表词表(热加载跟随命名),
中英混说更稳。首次运行自动下载模型(~460MB,需 https_proxy)。
"""

from __future__ import annotations

import argparse
import collections
import queue
import threading
import time

import numpy as np


class VoiceReader:
    """开麦 -> 断句 -> 转写;每句回调 on_text(text, t_end_wall, asr_s)。"""

    def __init__(self, on_text, model="small", vocab=(), say=print,
                 aggressiveness=2, silence_s=0.6, min_speech_s=0.3, device=None):
        import sounddevice as sd
        import webrtcvad
        from faster_whisper import WhisperModel
        self.on_text, self.say, self.vocab = on_text, say, vocab
        self.sd, self.vad = sd, webrtcvad.Vad(aggressiveness)
        self.silence_s, self.min_speech_s = silence_s, min_speech_s
        self.device = device  # None=系统默认;int 序号或名字子串(--list 查)
        t0 = time.time()
        say(f"[语音] 加载 whisper {model}(CPU int8)…")
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        say(f"[语音] 就绪({time.time() - t0:.1f}s)。开麦:说话即指令,静音 {silence_s:.1f}s 断句")

    def run(self):
        """阻塞循环,放线程里跑。"""
        FR = 16000
        FRAME = FR * 30 // 1000  # 30ms = webrtcvad 合法帧长
        q: queue.Queue = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(bytes(indata))

        buf: list[bytes] = []
        ring = collections.deque(maxlen=8)  # 前导 240ms
        voiced, silence = False, 0
        with self.sd.RawInputStream(samplerate=FR, blocksize=FRAME, channels=1,
                                    dtype="int16", callback=cb, device=self.device):
            while True:
                frame = q.get()
                if len(frame) != FRAME * 2:
                    continue
                is_sp = self.vad.is_speech(frame, FR)
                if not voiced:
                    ring.append(frame)
                    if is_sp:
                        voiced, buf, silence = True, list(ring), 0
                else:
                    buf.append(frame)
                    silence = 0 if is_sp else silence + 1
                    if silence * 0.03 >= self.silence_s:
                        t_end = time.time() - self.silence_s  # 语音实际结束时刻
                        speech_s = len(buf) * 0.03 - self.silence_s
                        voiced = False
                        ring.clear()
                        seg, buf = b"".join(buf), []
                        if speech_s >= self.min_speech_s:
                            self._transcribe(seg, t_end)

    def _transcribe(self, pcm: bytes, t_end: float):
        t0 = time.time()
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        vocab = list(self.vocab)
        prompt = ("机器人指令。词表:" + "、".join(vocab[:40])
                  + "、拿这个、抓这个、去这里、过来、回来、停。") if vocab else None
        segs, _ = self.model.transcribe(audio, language="zh", beam_size=1,
                                        initial_prompt=prompt, vad_filter=False)
        text = "".join(s.text for s in segs).strip()
        if text:
            self.on_text(text, t_end, time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="small")
    ap.add_argument("--once", action="store_true", help="转写一句即退出(试麦)")
    ap.add_argument("--device", default=None, help="输入设备:序号或名字子串(缺省=系统默认)")
    ap.add_argument("--list", action="store_true", help="列出音频设备后退出")
    args = ap.parse_args()
    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    dev = args.device
    if dev is not None and dev.isdigit():
        dev = int(dev)
    got = []

    def on_text(text, t_end, asr_s):
        print(f"[转写] 「{text}」  (asr {asr_s:.2f}s + 尾判 0.6s)")
        got.append(text)

    vr = VoiceReader(on_text, model=args.model, device=dev)
    threading.Thread(target=vr.run, daemon=True).start()
    try:
        while not (args.once and got):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
