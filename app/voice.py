# -*- coding: utf-8 -*-
"""voice.py — J022 Ghoul 桌宠语音模块（多声线可选 + 语速可调）

特性
  - 三种女声声线，可在桌宠右键菜单「语音」子菜单里自主选择；全部经过赛博机械处理：
      * cyber  赛博 Ghoo   —— Microsoft Xiaoxiao Online 女声（默认）
      * xiaoyi 晓伊 Xiaoyi —— Microsoft Xiaoyi Online 女声（在线）
      * huihui 慧慧 Huihui —— Microsoft Huihui SAPI5 女声（离线）
  - 语速三档：慢 / 正常 / 快（在线用百分比、离线 SAPI 用速率整数）。
  - 赛博/机械音色：三种声线统一启用，对 PCM 做 环形调制 + 预加重 +
            软削波 + 降比特，做出电子/机器人质感。
  - 播放：写入临时 WAV，用标准库 winsound 播放（可随时打断）。
  - 后台线程，不阻塞桌宠主消息循环。
  - 选择会写入 voice_config.json，重启桌宠后保持。

注意：本模块只被桌宠（desktop_pet.py）使用；网页对话(web_app.py)不调用，
因此网页永远不会发声。

环境变量（高级微调，可选）：
  VOICE_CYBER_RING_FREQ 环形调制载波频率 Hz（默认 70，越低越"机器人"）
  VOICE_CYBER_RING_DEPTH 环形调制深度 0~1（默认 0.5）
  VOICE_CYBER_BITS     量化比特 8~16（默认 14，越小越"数字颗粒"）
  VOICE_CYBER_GAIN     软削波增益（默认 1.15）
"""
import os
import sys
import time
import wave
import json
import queue
import threading
import tempfile

import numpy as np

try:
    import winsound
except Exception:  # 非 Windows 环境
    winsound = None


# ---------------------------------------------------------------- 配置
def _env(name, default):
    v = os.environ.get(name)
    return v if v is not None else default


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "voice_config.json")

# 三种声线：key -> 配置（全部女声，全部赛博化处理）
PROFILES = {
    "cyber":  {"label": "赛博 Ghoo（云潇·女声）",   "name": "zh-CN-XiaoxiaoNeural",
               "cyber": True, "engine": "edge"},
    "xiaoyi": {"label": "晓伊 Xiaoyi（在线·女声）",  "name": "zh-CN-XiaoyiNeural",
               "cyber": True, "engine": "edge"},
    "huihui": {"label": "慧慧 Huihui（离线·女声）",   "name": None,  # SAPI 自动选中文女声
               "cyber": True, "engine": "sapi"},
}

# 语速三档：key -> 百分比偏移（在线 Edge 用；离线 SAPI 由百分比换算速率）
SPEEDS = {
    "slow":   {"label": "慢",   "pct": -22},
    "normal": {"label": "正常", "pct": -8},
    "fast":   {"label": "快",   "pct": 18},
}

RING_FREQ = float(_env("VOICE_CYBER_RING_FREQ", "70"))
RING_DEPTH = float(_env("VOICE_CYBER_RING_DEPTH", "0.5"))
CYBER_BITS = int(_env("VOICE_CYBER_BITS", "14"))
CYBER_GAIN = float(_env("VOICE_CYBER_GAIN", "1.15"))

SND_FILENAME = 0x00020000
SND_ASYNC = 0x0001
SND_NODEFAULT = 0x0002
SND_PURGE = 0x0040


# ---------------------------------------------------------------- 配置读写
def _load_cfg():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            c = json.load(f)
        p = c.get("profile")
        s = c.get("speed")
        if p not in PROFILES:
            p = "cyber"
        if s not in SPEEDS:
            s = "normal"
        return p, s, bool(c.get("enabled", True))
    except Exception:
        return "cyber", "normal", True


def _save_cfg(profile, speed, enabled):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"profile": profile, "speed": speed, "enabled": enabled},
                      f, ensure_ascii=False)
    except Exception:
        pass


class Voice:
    """后台语音引擎：FIFO 队列 + 可打断 + 多声线 + 语速 + winsound 播放。"""

    def __init__(self):
        self.profile, self.speed, self.enabled = _load_cfg()
        self.ring_freq = RING_FREQ
        self.ring_depth = RING_DEPTH
        self.bits = CYBER_BITS
        self.gain = CYBER_GAIN

        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._gen_token = 0
        self._abort_token = 0
        self._stop = threading.Event()
        self._play_lock = threading.Lock()
        self._play_ev = None          # 当前播放的"打断"事件（每次播放新建）
        self._seq = 0
        self._tmp = tempfile.mkdtemp(prefix="j022voice_")

        self._apply_profile()   # 根据声线设置 voice_name / cyber / engine
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---------------------------------------------------- 公共接口
    def set_enabled(self, v):
        self.enabled = bool(v)
        if not self.enabled:
            self._purge_audio()
        _save_cfg(self.profile, self.speed, self.enabled)

    def set_profile(self, key):
        if key not in PROFILES:
            return
        self.profile = key
        self._apply_profile()
        _save_cfg(self.profile, self.speed, self.enabled)

    def set_speed(self, key):
        if key not in SPEEDS:
            return
        self.speed = key
        _save_cfg(self.profile, self.speed, self.enabled)

    @staticmethod
    def profile_list():
        """返回 [(key, label), ...] 供菜单渲染。"""
        return [(k, PROFILES[k]["label"]) for k in PROFILES]

    @staticmethod
    def speed_list():
        return [(k, SPEEDS[k]["label"]) for k in SPEEDS]

    # 当前速率百分比
    @property
    def rate_pct(self):
        return SPEEDS[self.speed]["pct"]

    def _apply_profile(self):
        p = PROFILES[self.profile]
        self.voice_name = p["name"]
        self.cyber = p["cyber"]
        self._engine = p["engine"]

    def speak(self, text, interrupt=False):
        text = (text or "").strip()
        if not text or not self.enabled:
            return
        with self._lock:
            self._gen_token += 1
            tok = self._gen_token
            if interrupt:
                self._abort_token = tok
        if interrupt:
            self._purge_audio()
        else:
            if self._q.qsize() > 2:
                try:
                    self._q.get_nowait()
                    self._q.task_done()
                except queue.Empty:
                    pass
        self._q.put((tok, text, interrupt))

    # ---------------------------------------------------- 后台循环
    def _run(self):
        while not self._stop.is_set():
            try:
                tok, text, prio = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if not self.enabled:
                self._q.task_done()
                continue
            if tok < self._abort_token and not prio:
                self._q.task_done()
                continue
            try:
                self._say(tok, text, prio)
            except Exception as e:  # noqa: BLE001
                print("[voice] 合成/播放失败:", e, flush=True)
            self._q.task_done()

    def _purge_audio(self):
        # 停掉当前正在播放的音频
        if winsound is not None:
            try:
                winsound.PlaySound(None, SND_PURGE)
            except Exception:
                pass
        # 唤醒正在等待播放结束的 worker（打断当前这句）
        with self._play_lock:
            ev = self._play_ev
        if ev is not None:
            ev.set()
        # 清空积压的待播队列
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass

    def _say(self, tok, text, prio):
        # 合成前：若已被更新的打断作废，直接跳过
        if tok < self._abort_token and not prio:
            return
        pcm, fs = self._synthesize(text)
        # 合成可能耗时，播前再判一次，避免播出一句已作废的旧话
        if tok < self._abort_token and not prio:
            return
        if self.cyber:
            pcm = self._cyberify(pcm, fs)
        path = os.path.join(self._tmp, "v%d.wav" % self._seq)
        self._seq += 1
        self._write_wav(path, pcm, fs)
        dur = len(pcm) / fs
        ev = threading.Event()
        with self._play_lock:
            self._play_ev = ev
        if winsound is not None:
            try:
                winsound.PlaySound(path, SND_FILENAME | SND_ASYNC | SND_NODEFAULT)
            except Exception as e:  # noqa: BLE001
                print("[voice] 播放失败:", e, flush=True)
        # 等本句自然播完（dur + 少量缓冲）；期间若被 _purge_audio 打断则提前返回。
        # 关键：这里绝不主动 purge，避免把正在播的句子从中间掐断（"说着说着没声"）。
        self._wait_play(dur + 0.15, ev)
        with self._play_lock:
            if self._play_ev is ev:
                self._play_ev = None
        # 播完（或被打断）后再删本段音频，避免播放中途删文件导致断音
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def _wait_play(self, dur, ev):
        end = time.time() + dur
        while time.time() < end:
            if self._stop.is_set() or ev.is_set():
                return
            time.sleep(0.05)

    # ---------------------------------------------------- 合成
    def _synthesize(self, text):
        if self._engine == "edge":
            # 在线合成偶发失败（网络抖动）时先重试一次，再回退离线 SAPI5
            for attempt in (1, 2):
                try:
                    return self._synth_edge(text)
                except Exception as e:  # noqa: BLE001
                    print("[voice] Edge TTS 第%d次失败：%s" % (attempt, e), flush=True)
                    if attempt == 1:
                        time.sleep(0.25)
            print("[voice] Edge 连续失败，回退 SAPI5 离线合成", flush=True)
        return self._synth_sapi(text)

    def _synth_edge(self, text):
        import asyncio
        import edge_tts
        import miniaudio

        rate = "%+d%%" % self.rate_pct

        async def _gen():
            comm = edge_tts.Communicate(text, self.voice_name, rate=rate)
            buf = b""
            async for d in comm.stream():
                if isinstance(d, dict) and d.get("type") == "audio":
                    buf += (d.get("data") or b"")
            if not buf:
                raise RuntimeError("Edge TTS 未返回音频")
            return buf

        async def _main():
            # 超时保护：网络卡住时不让后台语音线程永久阻塞
            return await asyncio.wait_for(_gen(), timeout=20)

        buf = asyncio.run(_main())
        sf = miniaudio.decode(buf,
                              output_format=miniaudio.SampleFormat.FLOAT32,
                              nchannels=1, sample_rate=24000)
        pcm = np.frombuffer(sf.samples, dtype=np.float32)
        fs = sf.sample_rate
        pcm = np.clip(pcm, -1.0, 1.0) * 32767.0
        return pcm.astype(np.int16), int(fs)

    def _synth_sapi(self, text):
        import pyttsx3

        # 关键修复：每次都用"全新引擎"。复用同一个 pyttsx3 引擎时，第二次
        # save_to_file+runAndWait 会偶发死锁，导致后台语音线程卡死、之后彻底没声音。
        eng = pyttsx3.init()
        path = os.path.join(self._tmp, "sapi_%d.wav" % self._seq)
        self._seq += 1
        try:
            voices = eng.getProperty("voices") or []
            vid = None
            for v in voices:
                nm = (getattr(v, "name", "") or "")
                if ("Chinese" in nm) or ("Huihui" in nm) or ("中文" in nm):
                    vid = getattr(v, "id", None)
                    break
            if vid:
                eng.setProperty("voice", vid)
            # 离线声线按语速档换算（基准 200）
            eng.setProperty("rate", int(round(200 * (1.0 + self.rate_pct / 100.0))))
        except Exception:  # noqa: BLE001
            pass
        eng.save_to_file(text, path)
        eng.runAndWait()
        try:
            eng.stop()
        except Exception:  # noqa: BLE001
            pass
        with wave.open(path, "rb") as w:
            fs = w.getframerate()
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16), int(fs)

    # ---------------------------------------------------- 赛博机械音色
    def _cyberify(self, pcm_int16, fs):
        x = pcm_int16.astype(np.float32) / 32768.0
        x = x - np.mean(x)
        n = len(x)
        if n == 0:
            return pcm_int16
        t = np.arange(n) / fs

        carrier = np.sin(2.0 * np.pi * self.ring_freq * t).astype(np.float32)
        y = x * (1.0 + self.ring_depth * carrier)

        prev = np.concatenate(([0.0], x[:-1]))
        y = y + 0.18 * (x - prev)

        y = np.tanh(y * self.gain)

        if self.bits < 16:
            levels = float(2 ** self.bits)
            y = np.round(y * levels) / levels

        y = np.clip(y, -1.0, 1.0)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = y / peak * 0.95
        return (y * 32767.0).astype(np.int16)

    # ---------------------------------------------------- 工具
    @staticmethod
    def _write_wav(path, pcm_int16, fs):
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(fs)
            w.writeframes(pcm_int16.tobytes())


# ---------------------------------------------------------------- 单例
_voice = None


def get_voice():
    global _voice
    if _voice is None:
        _voice = Voice()
    return _voice


def speak(text, interrupt=False):
    """便捷函数：直接让桌宠发声。"""
    get_voice().speak(text, interrupt=interrupt)
