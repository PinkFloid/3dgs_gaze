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
断句即「叮」= 听到了(转写/LLM 再慢,用户不看屏幕也知道系统活着);
chime() 供 brain 复用(ask/ok/fail 三音,见函数注释)。

噪声五闸(webrtcvad 会把风扇/键盘当语音,whisper 对非语音段会幻听出
"⑽④③③…"这类垃圾,实测直通到 LLM):① 超长段丢弃 max_speech_s——
指令不该 >12s,连续噪声/闲聊整段不转写,也防转写线程被长段卡死;
② 能量阈 min_rms——低于它的段是环境底噪,不进 ASR(丢弃时打印实测
rms 供调参);③ 转写时 vad_filter=True(silero 复核,专杀幻听源);
④ 转写卫生 _sane——真指令必以中英文为主,符号占多数 = 幻听,丢弃;
⑤ 置信闸 lp/ns——prompt 硬掰出的词表词形好分低,阈值见 _transcribe。
急停快诊另有自己的物理闸(成片话音)+ 同款置信闸,见 _sniff_stop。
"""

from __future__ import annotations

import argparse
import collections
import queue
import re
import threading
import time

import numpy as np

# ---------------------------------------------------------------- 提示音
# 免屏幕反馈(实验时用户看不到终端):heard=听到了(断句即响)/ ask=等确认
# (升调,该说"好/不"了)/ ok=已派发(双音)/ fail=失败或取消(低音)。
# 全部 <0.3s = 短于 VAD 的 min_speech,音箱串回麦克风也凑不成一句"话";
# 播放非阻塞,出错静默——提示音是增强,没声卡不许影响主流程。
_TONES: dict = {}
_CHIME = {"off": False}
_CHIME_LOCK = threading.Lock()


def chime(kind):
    """输出设备打不开时(默认输出指向没接的 HDMI 等),PortAudio 在 C 层向
    stderr 喷 ALSA 报错(Python 捕获不到)且每响一次喷一次——播放期间把 fd2
    按到 /dev/null,失败一次即永久禁用并只报一行。提示音是增强,不许污染主流程。"""
    if _CHIME["off"]:
        return
    try:
        import os
        import sounddevice as sd
        if kind not in _TONES:
            def seg(f, ms):
                t = np.linspace(0.0, ms / 1000.0, int(48 * ms), False)
                w = 0.3 * np.sin(2 * np.pi * f * t)
                n = 480  # 10ms 淡入出防爆音
                w[:n] *= np.linspace(0.0, 1.0, n)
                w[-n:] *= np.linspace(1.0, 0.0, n)
                return w
            seq = {"heard": [(880, 90)], "ask": [(660, 110), (880, 110)],
                   "ok": [(523, 90), (784, 90)], "fail": [(233, 250)]}[kind]
            _TONES[kind] = np.concatenate([seg(f, ms) for f, ms in seq]).astype(np.float32)
        with _CHIME_LOCK:  # fd2 换装是进程级操作:多线程同时 chime 必须串行
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_fd = os.dup(2)
            try:
                os.dup2(devnull, 2)
                sd.play(_TONES[kind], 48000)  # 48k=输出设备原生率
            finally:
                os.dup2(old_fd, 2)
                os.close(old_fd)
                os.close(devnull)
    except Exception:
        _CHIME["off"] = True
        print("[语音] 提示音输出设备不可用,已禁用提示音(不影响识别与派发;"
              "想要提示音:系统默认输出设为有效设备后重启)")


def _open_stream(sd, device, frame16, cb):
    """48k 硬编开流,3:1 降采样到 16k。固定麦(DJI Rx)与板载 ALSA 都只认 44.1/48k,
    先试 16k 的探测必失败且 PortAudio 在 C 层往 stderr 喷 paInvalidSampleRate
    (Python 拦不住,还绕过 prompt_toolkit 直污"指令>"行)。返回 (stream, 降采样因子)。"""
    return sd.RawInputStream(samplerate=48000, blocksize=frame16 * 3, channels=1,
                             dtype="int16", callback=cb, device=device), 3


def _decimate(pcm: bytes, dec: int) -> bytes:
    """整数因子降采样:每 dec 点取均值(粗低通防混叠,ASR 足够)。"""
    x = np.frombuffer(pcm, np.int16).astype(np.float32)
    n = len(x) // dec * dec
    return x[:n].reshape(-1, dec).mean(1).astype(np.int16).tobytes()


def _sane(text: str) -> bool:
    """真指令必然以中英文为主;whisper 幻听是带圈数字/符号串(见模块注释④)。"""
    core = [ch for ch in text if not ch.isspace()]
    ok = sum(1 for ch in core
             if (ch.isascii() and ch.isalnum()) or "一" <= ch <= "鿿")
    return ok >= len(core) * 0.5 and ok > 0


class VoiceReader:
    """开麦 -> 断句 -> 转写;每句回调 on_text(text, t_end_wall, asr_s)。"""

    def __init__(self, on_text, model="small", vocab=(), say=print, stop_test=None,
                 busy_probe=None, aggressiveness=2, silence_s=0.6, min_speech_s=0.3,
                 device=None, min_rms=100, max_speech_s=12.0, dump_dir=None):
        import sounddevice as sd
        import webrtcvad
        from faster_whisper import WhisperModel
        self.on_text, self.say, self.vocab = on_text, say, vocab
        self.stop_test = stop_test  # 急停快诊判词(brain 注入 is_stop∘norm)
        self.busy_probe = busy_probe  # 任务在飞才快诊(None=总是):空闲时停无意义,
        # 短拿类指令也不必白付快诊;执行中的新拿单反正会被狗拒 busy,多付无感
        # (试过 tiny 专职快诊:在真实停片段上复读崩溃「停。」x112 反而 3.7s
        #  且有幻听漏停——快诊沿用主模型,重复的停词 prompt 对 small 无此问题)
        self.sd, self.vad = sd, webrtcvad.Vad(aggressiveness)
        self.silence_s, self.min_speech_s = silence_s, min_speech_s
        self.min_rms, self.max_speech_s = min_rms, max_speech_s
        self.device = device  # None=系统默认;int 序号或名字子串(--list 查)
        # Pupil Capture 不录音频:每条过闸语音段存 WAV(ASR 审计/配音素材)。
        # 文件名=说完时刻 epoch,与 events.jsonl 的 asr 事件对账。
        self.dump_dir = dump_dir
        if dump_dir:
            from pathlib import Path
            Path(dump_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        say(f"[语音] 加载 whisper {model}(CPU int8)…")
        try:  # 先离线:实验现场不联网;hf 的在线版本检查还会撞 socks:// 代理(httpx 不认该 scheme)
            self.model = WhisperModel(model, device="cpu", compute_type="int8",
                                      local_files_only=True)
        except Exception:
            say(f"[语音] 本地缓存无 {model},联网下载(~460MB 需 https_proxy;"
                "若报 Unknown scheme for proxy 是 all_proxy=socks:// 所致,"
                "改 socks5:// 或 unset all_proxy 再跑)")
            self.model = WhisperModel(model, device="cpu", compute_type="int8")
        say(f"[语音] 就绪({time.time() - t0:.1f}s)。开麦:说话即指令,静音 {silence_s:.1f}s 断句")

    def run(self):
        """阻塞循环,放线程里跑。"""
        FR = 16000
        FRAME = FR * 30 // 1000  # 30ms = webrtcvad 合法帧长
        q: queue.Queue = queue.Queue()

        def cb(indata, frames, t, status):
            q.put(bytes(indata))

        stream, dec = _open_stream(self.sd, self.device, FRAME, cb)
        buf: list[bytes] = []
        ring = collections.deque(maxlen=8)  # 前导 240ms
        voiced, silence = False, 0
        sniff_at, fired = 1e9, False
        with stream:
            while True:
                frame = q.get()
                if dec > 1:
                    frame = _decimate(frame, dec)
                if len(frame) != FRAME * 2:
                    continue
                is_sp = self.vad.is_speech(frame, FR)
                if not voiced:
                    ring.append(frame)
                    if is_sp:
                        voiced, buf, silence = True, list(ring), 0
                        sniff_at, fired = 1.2, False
                else:
                    buf.append(frame)
                    # 段中急停快诊:狗行走噪声会压着 VAD 关不了段(实测「停一下」
                    # 在 11.6s 长段里等到段尾才转写,人到面前才停)。段开着且任务
                    # 在飞时,每 ~1.5s 对最近 2.5s 音频先诊一遍,不等断句。
                    cur = len(buf) * 0.03
                    if (not fired and self.stop_test and cur >= sniff_at
                            and (self.busy_probe is None or self.busy_probe())):
                        sniff_at = cur + 1.5
                        tail = b"".join(buf[-int(2.5 / 0.03):])
                        a = np.frombuffer(tail, np.int16).astype(np.float32)
                        if float(np.sqrt(np.mean(a ** 2))) >= self.min_rms:
                            fired = self._sniff_stop(tail, time.time())
                    if len(buf) * 0.03 > self.max_speech_s:  # 闸①:超长段整段丢弃
                        self.say(f"[语音] 丢弃 >{self.max_speech_s:.0f}s 无停顿长段"
                                 "(指令请短说;连续噪声/闲聊不转写)")
                        voiced, buf = False, []
                        ring.clear()
                        continue
                    silence = 0 if is_sp else silence + 1
                    if silence * 0.03 >= self.silence_s:
                        t_end = time.time() - self.silence_s  # 语音实际结束时刻
                        speech_s = len(buf) * 0.03 - self.silence_s
                        voiced = False
                        ring.clear()
                        seg, buf = b"".join(buf), []
                        if fired:
                            # 段中已急停:段仍要全量转写——停后紧跟的指令会落在
                            # 同一个噪声未关段的尾巴里(实测「停一下…拿给我」6.4s
                            # 段,第一声拿给我被整段吞)。skip_sniff 防重复快诊,
                            # 句首停词由 _transcribe 掐掉,只上行停后的增量。
                            self._transcribe(seg, t_end, skip_sniff=True)
                            continue
                        if speech_s < self.min_speech_s:
                            continue
                        rms = float(np.sqrt(np.mean(
                            np.frombuffer(seg, np.int16).astype(np.float32) ** 2)))
                        if rms < self.min_rms:  # 闸②:环境底噪不进 ASR
                            self.say(f"[语音] 丢弃低能量段 {speech_s:.1f}s"
                                     f"(rms {rms:.0f} < {self.min_rms:.0f},嫌严调 --min-rms)")
                            continue
                        chime("heard")  # 断句即响:比转写结果早 ~1s 告诉用户"听到了"
                        self._transcribe(seg, t_end)

    def _dump(self, pcm: bytes, t_end: float):
        if not self.dump_dir:
            return
        import wave
        from pathlib import Path
        try:
            with wave.open(str(Path(self.dump_dir) / f"utt_{t_end:.2f}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(pcm)
        except Exception:
            pass

    def _sniff_stop(self, pcm: bytes, wall: float) -> bool:
        """急停快诊:裁掉逐词对齐/长prompt 的极简转写,诊出停词立即上行。
        vad_filter 必须留着:狗噪音段(rms~115)配停词 prompt 会幻听出「停。」
        (实测本段裸转写=「字幕製作人」),silero 恰好杀这种鬼,真停无损。"""
        a = np.frombuffer(pcm, np.int16).astype(np.float32)
        # 停必须喊、且是成片的话音(物理闸 v2,8-27 17:24 幻听现行后重定):旧闸
        # (峰值>=220 且 2x中位)的闸形恰好就是行走冲击的形态——幻听窗峰值实测
        # 1500-3438,反超真停金标(223-644),峰值是反向指标,对狗噪全放行。
        # 分得开的是持续性:0.6s(20帧)滑窗内 rms>=220 的帧数,指令音量人声
        # 18-20 片成片,行走/伺服冲击最坏 14(8-27 全天 6 段噪声窗标定),阈 16
        # 两侧各留 2。弱声段(太好了=13、嗯=8)也被拦——快诊只管喊出来的停,
        # 轻声话走段尾全量路径,不白烧每 1.5s 一口解码。
        fr = 480  # 30ms 帧
        n = len(a) // fr
        if n:
            e = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(1))
            k = min(20, n)
            run = int(np.convolve((e >= 220).astype(int), np.ones(k, int), "valid").max())
            if run < min(16, k):
                return False
        audio = a / 32768.0
        t0 = time.time()
        kw = dict(language="zh", beam_size=1,
                  initial_prompt="停一下。",  # 规约 v2:唯一停词;多热一个形就多一种幻听
                  condition_on_previous_text=False,
                  temperature=0.0,  # 单温度:不走回退梯子,时延封顶
                  vad_filter=True)  # (金标 1/8 曾梯到 11.5s;
        # 没梯子最坏也就是本轮没诊出,段尾全量还兜着)
        try:  # 连喊会诱发贪心复读塌缩(实测「停。」x112 白烧 3.8s):解码长度封顶
            segs, _ = self.model.transcribe(audio, max_new_tokens=16, **kw)
        except (TypeError, ValueError):  # 旧版 faster-whisper 无此参数
            segs, _ = self.model.transcribe(audio, **kw)
        segs = list(segs)
        txt = "".join(s.text for s in segs).strip()
        if not (txt and self.stop_test(txt)):
            return False
        # 快诊也过置信闸(17:24 两次「停下。」幻听现行:此前只有全量路径有闸):
        # 双轴复合,阈值来自 8-27 全天语料(真指令最低 给我 lp-0.64/ns0.01 必须活)
        lp = min((s.avg_logprob for s in segs), default=0.0)
        ns = max((s.no_speech_prob for s in segs), default=0.0)
        if lp < -0.8 or ns > 0.4 or (lp < -0.55 and ns > 0.15):
            self.say(f"[语音] 快诊置信不足丢弃「{txt[:16]}」(lp={lp:.2f} ns={ns:.2f})")
            return False
        self.say(f"[语音] 快诊原文「{txt[:24]}」(lp={lp:.2f} ns={ns:.2f})")
        # fast 参数带原文+置信上行:events.jsonl 里只剩归一「停一下」时,真停
        # 幻听事后不可分辨(17:24 排障靠的是 WAV 人耳复核,观测性欠账在此还上)
        self.on_text("停一下", wall, time.time() - t0, None,
                     f"{txt[:24]}|lp{lp:.2f}|ns{ns:.2f}")
        return True

    def _transcribe(self, pcm: bytes, t_end: float, skip_sniff: bool = False):
        t0 = time.time()
        self._dump(pcm, t_end)  # 幻听/空转写的段也存:被丢弃的才最需要人耳复核
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        # 急停快诊:短段先跑裁掉慢件(逐词对齐/silero/长prompt)的极简转写,
        # 诊出停词立即上行、跳过 ~2s 全量——急停不需要词时刻(实测「停一下」
        # 全量 1.99s,任务本体都没这么久)。非停短段照走全量,多付快诊一口
        # (~0.3s,只有 ≤1.6s 的段付);停在句中的长句不走此门,由 LLM 慢裁。
        if not skip_sniff and self.stop_test and len(audio) <= 16000 * 1.6 \
                and (self.busy_probe is None or self.busy_probe()):
            if self._sniff_stop(pcm, t_end):
                return
        vocab = list(self.vocab)
        # 高频指令短语也进热词:偏置解码——"拿一下"实测被听成"麻衣下/那一下",
        # 光给物体名压不住动词错字
        # 「停一下」必须显式在场:prompt 里只有"拿一下这个"时,"X一下"的
        # 先验全在拿/看那边,实测把「停一下」听成「看一下」——急停丢失。
        prompt = ("机器人指令。词表:" + "、".join(vocab[:40])
                  + "、拿一下这个、把这个拿来、抓这个、去这里拿、"
                  "放到那里、放回、过来、回来、停、停一下、停下。") if vocab else None
        segs, _ = self.model.transcribe(audio, language="zh", beam_size=1,
                                        initial_prompt=prompt,
                                        word_timestamps=True,  # 逐词墙钟:指示词各带时刻
                                        vad_filter=True)  # 闸③:silero 复核,专杀幻听源
        segs = list(segs)
        text = "".join(s.text for s in segs).strip()
        if not text:
            return
        if not _sane(text):  # 闸④:符号占多数 = 对噪声的幻听,不进指令队列
            self.say(f"[语音] 疑似幻听,丢弃「{text[:24]}」")
            return
        # 闸⑤ 置信闸:有声非话(伺服声/哼气/感叹)被热词 prompt 硬掰成词表词,
        # silero 判不了"是不是话",但 whisper 自己的信心分得开——8-27 全天实测:
        # 真指令 lp>=-0.33 ns<=0.04,真停 lp>=-0.56;嗯/哎/闲聊 lp<=-0.60 或 ns>=0.13。
        lp = min((s.avg_logprob for s in segs), default=0.0)
        ns = max((s.no_speech_prob for s in segs), default=0.0)
        if lp < -0.8 or ns > 0.4 or (lp < -0.55 and ns > 0.15):
            self.say(f"[语音] 置信不足丢弃「{text[:24]}」(lp={lp:.2f} ns={ns:.2f})")
            return
        if skip_sniff:  # 段中已急停的段:掐掉句首停词,只上行停后的增量指令
            text = re.sub(r"^[\s。,,、!!??]*(?:停一下|停下|停止|别动|停)+[\s。,,、!!??]*",
                          "", text)
            if not text:
                return  # 整段只有停词:快诊已消费,不再重复上行
            self.say(f"[语音] 停后增量指令「{text[:24]}」")
        t_seg0 = t_end - len(audio) / 16000.0   # 段起点墙钟
        words = [(w.word.strip(), t_seg0 + w.start, t_seg0 + w.end)
                 for s in segs for w in (s.words or []) if w.word.strip()]
        self.on_text(text, t_end, time.time() - t0, words or None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="small")
    ap.add_argument("--once", action="store_true", help="转写一句即退出(试麦)")
    ap.add_argument("--device", default="Rx",
                    help="输入设备:序号或名字子串(缺省=DJI 无线麦;序号会随插拔变,--list 查)")
    ap.add_argument("--min-rms", type=float, default=100,
                    help="能量闸:低于此 rms 的段当环境底噪丢弃(丢弃时打印实测值;"
                         "默认=DJI 麦 --meter 校准值,底噪14/说话207 的几何均值,换麦必重跑)")
    ap.add_argument("--list", action="store_true", help="列出音频设备后退出")
    ap.add_argument("--meter", type=float, default=0.0,
                    help="电平表 N 秒:每 0.3s 打 rms 条+VAD 判定,前半安静后半说话,"
                         "结束给 --min-rms 建议值(换麦必跑)")
    args = ap.parse_args()
    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    dev = args.device
    if dev is not None and dev.isdigit():
        dev = int(dev)
    if args.meter > 0:  # 不加载模型,纯电平:换麦/调阈值用
        import sounddevice as sd
        import webrtcvad
        vad, FR, FRAME = webrtcvad.Vad(2), 16000, 480
        vals = []
        print(f"电平表 {args.meter:.0f}s:前半保持安静(量底噪),后半正常说几句指令")
        mq: queue.Queue = queue.Queue()
        st, dec = _open_stream(sd, dev, FRAME, lambda d, f, t, s: mq.put(bytes(d)))
        with st:
            t0 = time.time()
            while time.time() - t0 < args.meter:
                pcm = b"".join(_decimate(mq.get(), dec) if dec > 1 else mq.get()
                               for _ in range(10))  # 0.3s
                x = np.frombuffer(pcm, np.int16).astype(np.float32)
                rms = float(np.sqrt(np.mean(x ** 2)))
                sp = sum(vad.is_speech(pcm[i * 960:(i + 1) * 960], FR) for i in range(10))
                vals.append((rms, sp))
                print(f"  rms {rms:6.0f}  vad {sp:2d}/10  " + "#" * min(60, int(rms / 50)))
        quiet = sorted(r for r, s in vals if s <= 2)
        loud = sorted(r for r, s in vals if s >= 8)
        if quiet:
            print(f"底噪中位 rms ≈ {quiet[len(quiet) // 2]:.0f}")
        if loud:
            print(f"说话中位 rms ≈ {loud[len(loud) // 2]:.0f}")
        if quiet and loud:
            rec = (quiet[len(quiet) // 2] * loud[len(loud) // 2]) ** 0.5
            print(f"建议 --min-rms / --voice-rms ≈ {rec:.0f}(几何均值;宁低勿高,能量闸只是兜底)")
        return 0
    got = []

    def on_text(text, t_end, asr_s, *rest):  # 逐词流/快诊标记只有 brain 用,这里吞掉
        print(f"[转写] 「{text}」  (asr {asr_s:.2f}s + 尾判 0.6s)")
        got.append(text)

    vr = VoiceReader(on_text, model=args.model, device=dev, min_rms=args.min_rms)
    threading.Thread(target=vr.run, daemon=True).start()
    try:
        while not (args.once and got):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
