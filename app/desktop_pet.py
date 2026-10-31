# -*- coding: utf-8 -*-
"""desktop_pet.py — J022 Ghoul 桌宠（纯 Win32 ctypes + PIL，逐像素透明，无 tkinter）。

功能
  - 置顶 / 透明 / 点击穿透（仅角色实体像素响应鼠标）
  - 左键单击  → 唤出输入框 → 本地 Ollama 模型回答（气泡 + 打字机效果）
  - 左键拖拽  → 角色倾斜跟随，松手 → 重力掉落回桌面
  - 右键菜单  → 聊一聊 / 散散步 / 跳一下 / 睡一会儿 / 回到右下角 / 退出
  - 常见桌宠动作：待机呼吸 / 眨眼 / 行走 / 跳跃 / 睡觉 Zzz / 思考 / 说话气泡
  - AI 状态指示灯：绿=在线 黄=思考中 灰=离线

用法
  python desktop_pet.py [--x 1200] [--y 600] [--no-chat] [--test 5]
"""
import argparse
import ctypes
import faulthandler
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glitch_fx import GlitchFX

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FRAMES_DIR = os.path.join(HERE, "assets", "frames")
FONT_PATH = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyh.ttc")
from mathtext import sanitize_math, demojify  # noqa: E402
from voice import get_voice  # noqa: E402
from ollama_stream import stream_ollama_chat  # noqa: E402
import ollama_stream  # noqa: E402  预热 / 卸载模型（防「卡住」）

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.environ.get("PET_MODEL", "qwen2.5-coder:7b")

PERSONA = (
    "你的名字是 Ghoo，你的本体（模型）代号是 J022 Ghoul，你是主人电脑上的离线 CAD 助手，"
    "原型是白发红瞳的哥特萝莉。你的主人（master）叫 J022BOT。"
    "性格：俏皮、可靠、略带毒舌但关心主人。"
    "用简体中文回答，回答必须简短（60字以内）。"
    "你完全离线运行，主人问什么就答什么，不会的问题直说。"
    "重要：说话**禁止使用任何 emoji 表情符号**（如 😀 😅 👍 ✨ ⚠ 这类彩色符号，"
    "气泡字体不支持，会显示成方块）。需要表达情绪时，一律改用**颜文字**——"
    "即由括号、标点、日文假名等符号拼成的表情，例如：(◕ᴗ◕)、(´▽`)、(｡･ω･｡)、"
    "(≧▽≦)、(；ω；｀)、(´･_･`)、(･ω･)b、m(_ _)m、ヽ(´▽`)/。(笑) 这种写法也不要用。"
    "重要：绝对不要使用 LaTeX 数学/化学式语法（如 \\[ \\] \\( \\) \\text{} \\frac "
    "\\rightarrow 等），所有方程、化学式、数学式都用 Unicode 纯文本写出，"
    "例如：2HF + Sb → SbF₃ + H₂（用 ₂ ₃ 等上下标字符）。"
)

# ---------------------------------------------------------------- Win32 常量
WS_EX_LAYERED = 0x00080000
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TRANSPARENT = 0x00000020
WS_POPUP = 0x80000000
WS_BORDER = 0x00800000
WS_CHILD = 0x40000000
WS_VISIBLE = 0x10000000
ES_AUTOHSCROLL = 0x0080

WM_DESTROY = 0x0002
WM_NULL = 0x0000
WM_TIMER = 0x0113
WM_NCHITTEST = 0x0084
WM_MOUSEMOVE = 0x0200
WM_SETCURSOR = 0x0020
WM_CAPTURECHANGED = 0x0215
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0206
WM_RBUTTONUP = 0x0205
WM_MOUSEACTIVATE = 0x0021
WM_ACTIVATE = 0x0006
WM_KEYDOWN = 0x0100
WM_COMMAND = 0x0111
WM_APP_EVENT = 0x8000 + 1
WM_APP_CHAT = 0x8000 + 2

HTTRANSPARENT = -1
HTCLIENT = 1
MA_NOACTIVATE = 3
ULW_ALPHA = 0x00000002
DIB_RGB_COLORS = 0
GWLP_WNDPROC = -4
VK_RETURN = 0x0D
VK_ESCAPE = 0x1B
VK_LBUTTON = 0x01
IDC_ARROW = 32512
IDC_HAND = 32649
IDC_SIZEALL = 32646
MF_STRING = 0x0000
MF_POPUP = 0x0010
MF_SEPARATOR = 0x0800

# MessageBox 标志（退出二次确认用）
MB_YESNO = 0x04
MB_ICONQUESTION = 0x20
MB_TOPMOST = 0x40000
IDYES = 6
TPM_RETURNCMD = 0x0100
TPM_NONOTIFY = 0x0080
SM_CXSCREEN, SM_CYSCREEN = 0, 1
SPI_GETWORKAREA = 0x0030
SW_HIDE, SW_SHOWNA = 0, 8

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# ---- 64 位安全的原型声明 ----
for fn, res, args in [
    (user32.CreateWindowExW, ctypes.c_void_p,
     [ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint,
      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]),
    (user32.DefWindowProcW, ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]),
    (user32.LoadIconW, ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_uint]),
    (user32.LoadCursorW, ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_uint]),
    (user32.GetDlgItem, ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int]),
    (user32.SetFocus, ctypes.c_void_p, [ctypes.c_void_p]),
    (user32.GetWindowRect, ctypes.c_bool, [ctypes.c_void_p, ctypes.c_void_p]),
    (user32.GetCursorPos, ctypes.c_bool, [ctypes.c_void_p]),
    (user32.SystemParametersInfoW, ctypes.c_bool,
     [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]),
    (user32.SetWindowLongPtrW, ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]),
    (user32.CallWindowProcW, ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
      ctypes.c_void_p]),
    (user32.AppendMenuW, ctypes.c_bool,
     [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_wchar_p]),
    (user32.CreatePopupMenu, ctypes.c_void_p, []),
    (user32.TrackPopupMenu, ctypes.c_int,
     [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int,
      ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]),
    (user32.SetForegroundWindow, ctypes.c_bool, [ctypes.c_void_p]),
    (user32.ShowWindow, ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int]),
    (user32.UpdateWindow, ctypes.c_bool, [ctypes.c_void_p]),
    (user32.DestroyWindow, ctypes.c_bool, [ctypes.c_void_p]),
    (user32.DestroyMenu, ctypes.c_bool, [ctypes.c_void_p]),
    (user32.PostMessageW, ctypes.c_bool,
     [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]),
    (user32.SetTimer, ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]),
    (user32.KillTimer, ctypes.c_bool, [ctypes.c_void_p, ctypes.c_void_p]),
    (user32.GetWindowTextLengthW, ctypes.c_int, [ctypes.c_void_p]),
    (user32.GetWindowTextW, ctypes.c_int,
     [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]),
    (user32.SetCapture, ctypes.c_void_p, [ctypes.c_void_p]),
    (user32.ReleaseCapture, ctypes.c_bool, []),
    (user32.SetCursor, ctypes.c_void_p, [ctypes.c_void_p]),
    (user32.GetAsyncKeyState, ctypes.c_short, [ctypes.c_int]),
    (gdi32.CreateCompatibleDC, ctypes.c_void_p, [ctypes.c_void_p]),
    (gdi32.CreateDIBSection, ctypes.c_void_p,
     [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
      ctypes.c_void_p, ctypes.c_uint]),
    (gdi32.SelectObject, ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p]),
    (gdi32.DeleteObject, ctypes.c_bool, [ctypes.c_void_p]),
    (gdi32.DeleteDC, ctypes.c_bool, [ctypes.c_void_p]),
    (kernel32.GetModuleHandleW, ctypes.c_void_p, [ctypes.c_wchar_p]),
    (kernel32.GetLastError, ctypes.c_uint, []),
]:
    fn.restype, fn.argtypes = res, args

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p)


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", ctypes.c_uint), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", ctypes.c_void_p), ("hIcon", ctypes.c_void_p),
                ("hCursor", ctypes.c_void_p), ("hbrBackground", ctypes.c_void_p),
                ("lpszMenuName", ctypes.c_wchar_p), ("lpszClassName", ctypes.c_wchar_p)]


class MSG(ctypes.Structure):
    _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                ("wParam", ctypes.c_void_p), ("lParam", ctypes.c_void_p),
                ("time", ctypes.c_uint), ("pt", POINT)]


_ULW = user32.UpdateLayeredWindow
_ULW.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(POINT),
                 ctypes.POINTER(SIZE), ctypes.c_void_p, ctypes.POINTER(POINT),
                 ctypes.c_uint32, ctypes.POINTER(BLENDFUNCTION), ctypes.c_uint32]
_ULW.restype = ctypes.c_bool


def _wp(v):
    return int(v or 0)


def _lo_word(v):
    x = _wp(v) & 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def _hi_word(v):
    y = (_wp(v) >> 16) & 0xFFFF
    return y - 0x10000 if y & 0x8000 else y


def _enable_dpi_awareness():
    """设为 per-monitor DPI aware：否则 125% 缩放下坐标/尺寸会被虚拟化。"""
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _last_error(what):
    raise WinError(f"{what} 失败，GetLastError={kernel32.GetLastError()}")


class WinError(Exception):
    pass


# ---------------------------------------------------------------- 回调保命区
# 【崩溃根因 / 必读】
# Win32 窗口类在注册时只保存一个原生函数指针（lpfnWndProc）。ctypes 的
# WNDPROC 对象一旦被 Python 回收，它背后的可执行 thunk 内存就会释放，
# 而窗口类里仍留着指向那块已释放内存的指针 —— 之后再创建同类窗口就会
# 跳进野地址，进程被系统直接杀掉（0xc000001d / 0xc0000005 / 0xc000041d），
# Python 层完全看不到异常（这就是"点一下就崩溃"的真相）。
#
# 因此：所有 WNDPROC / 子类化回调都必须由**模块级字典**长期持有，
# 绝不允许挂在某个会被销毁的实例上（实例 + 回调互相引用会被循环 GC 收走）。
_CLASS_PROCS = {}   # 类名 -> WNDPROC（进程生命周期内常驻）
_EDIT_PROCS = {}    # EDIT 子窗口 hwnd -> WNDPROC（子类化回调，随窗口销毁清除）

_CB_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pet_errors.log")


def _log_cb_error(where, exc):
    """回调里发生异常时记文件（不抛回原生栈，避免进程被杀）。"""
    try:
        import traceback
        with open(_CB_LOG, "a", encoding="utf-8") as f:
            f.write("\n[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), where))
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    except Exception:
        pass
    try:
        print("%s 出错：%s" % (where, exc), flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------- 分层窗口基类
class LayeredWindow:
    """逐像素 alpha 透明窗口（UpdateLayeredWindow）。"""

    _classes = {}
    _instances = {}

    def __init__(self, class_name, title=""):
        self.class_name = class_name
        self.title = title
        self.hwnd = None
        self._hdc = None
        self._bmp = None
        self._bits = ctypes.c_void_p()
        self._size = (0, 0)
        self._frame = None
        self._proc_ref = None
        self._create()

    def _register_class(self):
        if self.class_name in LayeredWindow._classes:
            return

        # 关键：回调按「类」注册一次，并存入模块级 _CLASS_PROCS 永久持有。
        # 绝不能用 self._proc_ref —— 实例销毁后回调节点被 GC 回收，
        # 窗口类里就留下悬空函数指针，再建同类窗口必崩。
        def _dispatch(hwnd, msg, w, l):
            inst = LayeredWindow._instances.get(hwnd)
            if inst is None:
                return user32.DefWindowProcW(hwnd, msg, w, l)
            try:
                return inst._wndproc(hwnd, msg, w, l)
            except Exception as e:  # noqa: BLE001
                _log_cb_error("LayeredWindow._dispatch", e)
                try:
                    return user32.DefWindowProcW(hwnd, msg, w, l)
                except Exception:
                    return 0

        proc = WNDPROC(_dispatch)
        _CLASS_PROCS[self.class_name] = proc
        wc = WNDCLASSW()
        wc.lpfnWndProc = proc
        wc.hInstance = kernel32.GetModuleHandleW(None)
        wc.hIcon = user32.LoadIconW(None, 32512)
        wc.hCursor = user32.LoadCursorW(None, 32512)
        wc.lpszClassName = self.class_name
        if not user32.RegisterClassW(ctypes.byref(wc)):
            if kernel32.GetLastError() != 1410:  # CLASS_ALREADY_EXISTS
                _last_error("RegisterClassW")
        LayeredWindow._classes[self.class_name] = True

    def _create(self):
        self._register_class()
        ex = WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        if getattr(self, "click_through", False):
            ex |= WS_EX_TRANSPARENT
        self.hwnd = user32.CreateWindowExW(
            ex, self.class_name, self.title, WS_POPUP,
            0, 0, 8, 8, None, None, kernel32.GetModuleHandleW(None), None)
        if not self.hwnd:
            _last_error("CreateWindowExW")
        LayeredWindow._instances[self.hwnd] = self

    # ---- DIB ----
    def _ensure_dib(self, w, h):
        if self._hdc is not None and self._size == (w, h):
            return
        if self._hdc is not None:
            gdi32.DeleteObject(self._bmp)
            gdi32.DeleteDC(self._hdc)
        self._size = (w, h)
        bi = BITMAPINFO()
        bi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bi.bmiHeader.biWidth = w
        bi.bmiHeader.biHeight = -h          # 自上而下
        bi.bmiHeader.biPlanes = 1
        bi.bmiHeader.biBitCount = 32
        self._hdc = gdi32.CreateCompatibleDC(None)
        if not self._hdc:
            _last_error("CreateCompatibleDC")
        self._bmp = gdi32.CreateDIBSection(self._hdc, ctypes.byref(bi),
                                           DIB_RGB_COLORS,
                                           ctypes.byref(self._bits), None, 0)
        if not self._bmp or not self._bits.value:
            _last_error("CreateDIBSection")
        gdi32.SelectObject(self._hdc, self._bmp)

    def blit(self, img, pos=None):
        img = img.convert("RGBA")
        w, h = img.size
        self._ensure_dib(w, h)
        arr = np.asarray(img, dtype=np.float32)
        alpha = arr[:, :, 3:4] / 255.0
        arr[:, :, :3] *= alpha                       # 预乘 alpha
        bgra = np.dstack([arr[:, :, 2], arr[:, :, 1], arr[:, :, 0],
                          arr[:, :, 3]]).astype(np.uint8)
        ctypes.memmove(self._bits.value, bgra.tobytes(), bgra.nbytes)
        self._frame = img
        if pos is None:
            r = RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(r))
            pos = (r.left, r.top)
        pt = POINT(int(pos[0]), int(pos[1]))
        sz = SIZE(w, h)
        src = POINT(0, 0)
        blend = BLENDFUNCTION(0, 0, 255, 1)          # AC_SRC_OVER | AC_SRC_ALPHA
        if not _ULW(self.hwnd, None, ctypes.byref(pt), ctypes.byref(sz),
                    self._hdc, ctypes.byref(src), 0, ctypes.byref(blend),
                    ULW_ALPHA):
            _last_error("UpdateLayeredWindow")

    def alpha_at(self, x, y):
        if self._frame is None:
            return 0
        w, h = self._frame.size
        if 0 <= x < w and 0 <= y < h:
            return self._frame.getpixel((int(x), int(y)))[3]
        return 0

    def rect(self):
        r = RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(r))
        return (r.left, r.top, r.right - r.left, r.bottom - r.top)

    def show(self):
        user32.ShowWindow(self.hwnd, SW_SHOWNA)

    def destroy(self):
        if self.hwnd:
            LayeredWindow._instances.pop(self.hwnd, None)
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None
        if self._hdc is not None:
            gdi32.DeleteObject(self._bmp)
            gdi32.DeleteDC(self._hdc)
            self._hdc = None

    def _wndproc(self, hwnd, msg, w, l):
        return user32.DefWindowProcW(hwnd, msg, w, l)


# ---------------------------------------------------------------- 聊天气泡窗
class BubbleWindow(LayeredWindow):
    """点击穿透的气泡窗，始终位于桌宠上方。"""

    def __init__(self):
        self.click_through = True
        super().__init__("J022Bubble", "J022Bubble")
        self._font = self._load_font(15)
        self.visible = False
        self._text = ""
        self._max_w = 260
        user32.ShowWindow(self.hwnd, SW_HIDE)

    @staticmethod
    def _load_font(size):
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            return ImageFont.load_default()

    def _wrap(self, text):
        lines, cur = [], ""
        for ch in text:
            if ch == "\n":
                lines.append(cur)
                cur = ""
                continue
            trial = cur + ch
            if self._font.getlength(trial) > self._max_w - 24:
                lines.append(cur)
                cur = ch
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines or [""]

    def show_text(self, text):
        self._text = text
        self.redraw(text)

    def redraw(self, text):
        if not text:
            self.visible = False
            user32.ShowWindow(self.hwnd, SW_HIDE)
            return
        lines = self._wrap(text)
        pad, lh = 12, 20
        w = min(self._max_w,
                max(60, int(max(self._font.getlength(x) for x in lines)) + pad * 2))
        h = lh * len(lines) + pad * 2 + 6
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 7], radius=10,
                            fill=(24, 26, 34, 238), outline=(150, 90, 90, 255),
                            width=1)
        d.polygon([(16, h - 8), (30, h - 8), (18, h - 1)], fill=(24, 26, 34, 238))
        for i, line in enumerate(lines):
            d.text((pad, pad + i * lh), line, font=self._font,
                   fill=(235, 238, 245, 255))
        self.visible = True
        self.blit(img)
        user32.ShowWindow(self.hwnd, SW_SHOWNA)

    def hide(self):
        self.visible = False
        self._text = ""
        user32.ShowWindow(self.hwnd, SW_HIDE)


# ---------------------------------------------------------------- 输入对话框
class InputDialog:
    """左键单击唤出的提问框（原生 EDIT + 按钮；嵌套消息泵保证动画不停）。"""

    ID_EDIT = 1001
    ID_BTN = 1002

    def __init__(self, app):
        self.app = app
        self.result = None
        self.open = True
        self._old_edit_proc = None
        self._edit_proc_ref = None
        self._proc_ref = None
        self.hwnd = None
        self._create()

    def _register_class(self):
        cls = "J022Input"

        # 同 LayeredWindow：回调按类注册一次并永久持有；
        # 每次新建提问框都复用这个类（旧实现把回调挂在第一个框实例上，
        # 该实例被 GC 回收后类里就留下死指针，第二次开框必然崩进程）。
        def _dispatch(hwnd, msg, w, l):
            inst = LayeredWindow._instances.get(hwnd)
            if inst is None:
                return user32.DefWindowProcW(hwnd, msg, w, l)
            try:
                return inst._wndproc(hwnd, msg, w, l)
            except Exception as e:  # noqa: BLE001
                _log_cb_error("InputDialog._dispatch", e)
                try:
                    return user32.DefWindowProcW(hwnd, msg, w, l)
                except Exception:
                    return 0

        proc = WNDPROC(_dispatch)
        _CLASS_PROCS[cls] = proc
        wc = WNDCLASSW()
        wc.lpfnWndProc = proc
        wc.hInstance = kernel32.GetModuleHandleW(None)
        wc.hIcon = user32.LoadIconW(None, 32512)
        wc.hCursor = user32.LoadCursorW(None, 32512)
        wc.lpszClassName = cls
        user32.RegisterClassW(ctypes.byref(wc))
        LayeredWindow._classes[cls] = True

    def _create(self):
        if "J022Input" not in LayeredWindow._classes:
            self._register_class()
        W, H = 340, 46
        px, py, pw, ph = self.app.pet.rect()
        x = px + pw // 2 - W // 2
        y = py - H - 44
        sw = user32.GetSystemMetrics(SM_CXSCREEN)
        x = max(4, min(x, sw - W - 4))
        self.hwnd = user32.CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW, "J022Input", "Ghoo · 提问",
            WS_POPUP | WS_BORDER, x, y, W, H, None, None,
            kernel32.GetModuleHandleW(None), None)
        if not self.hwnd:
            _last_error("CreateWindowExW(InputDialog)")
        LayeredWindow._instances[self.hwnd] = self
        user32.CreateWindowExW(
            0, "EDIT", "", WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOHSCROLL,
            8, 8, W - 96, H - 16, self.hwnd, self.ID_EDIT,
            kernel32.GetModuleHandleW(None), None)
        user32.CreateWindowExW(
            0, "BUTTON", "发送", WS_CHILD | WS_VISIBLE,
            W - 82, 8, 74, H - 16, self.hwnd, self.ID_BTN,
            kernel32.GetModuleHandleW(None), None)
        user32.ShowWindow(self.hwnd, SW_SHOWNA)
        user32.UpdateWindow(self.hwnd)
        user32.SetFocus(user32.GetDlgItem(self.hwnd, self.ID_EDIT))
        self._subclass_edit(user32.GetDlgItem(self.hwnd, self.ID_EDIT))

    def _subclass_edit(self, hwnd):
        def _edit_proc(h, msg, w, l):
            try:
                if msg == WM_KEYDOWN:
                    if _wp(w) & 0xFF == VK_RETURN:
                        self._submit()
                        return 0
                    if _wp(w) & 0xFF == VK_ESCAPE:
                        self.close()
                        return 0
                if msg == 0x0082:  # WM_NCDESTROY：窗口即将销毁，释放回调强引用
                    _EDIT_PROCS.pop(h, None)
                return user32.CallWindowProcW(self._old_edit_proc, h, msg, w, l)
            except Exception as e:  # noqa: BLE001
                _log_cb_error("InputDialog._edit_proc", e)
                try:
                    return user32.CallWindowProcW(self._old_edit_proc, h, msg, w, l)
                except Exception:
                    return 0

        proc = WNDPROC(_edit_proc)
        _EDIT_PROCS[hwnd] = proc       # 模块级强引用：窗口活着，回调就不能死
        self._edit_proc_ref = proc
        self._old_edit_proc = user32.SetWindowLongPtrW(
            hwnd, GWLP_WNDPROC,
            ctypes.cast(proc, ctypes.c_void_p).value)

    def _submit(self):
        if not self.open:
            return
        edit = user32.GetDlgItem(self.hwnd, self.ID_EDIT)
        n = user32.GetWindowTextLengthW(edit)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(edit, buf, n + 1)
        text = buf.value.strip()
        self.result = text or None
        self.close()
        # 非模态回传：直接回调桌宠应用（不再开嵌套消息泵）
        self.app._on_chat_submit(self.result)

    def _wndproc(self, hwnd, msg, w, l):
        if msg == WM_COMMAND and self.open:
            if _lo_word(w) == self.ID_BTN:
                self._submit()
                return 0
        elif msg == WM_KEYDOWN and self.open:
            if _wp(w) & 0xFF == VK_ESCAPE:
                self.close()
                return 0
        elif msg == WM_ACTIVATE and self.open:
            if _lo_word(w) == 0:
                self.close()
            return 0
        elif msg == WM_DESTROY:
            return 0
        return user32.DefWindowProcW(hwnd, msg, w, l)

    def close(self):
        if self.open:
            self.open = False
            if self.hwnd:
                LayeredWindow._instances.pop(self.hwnd, None)
                user32.DestroyWindow(self.hwnd)
                self.hwnd = None
            # 解除应用对它的引用，避免重复创建时叠加
            if getattr(self.app, "_dialog", None) is self:
                self.app._dialog = None


# ---------------------------------------------------------------- 聊天引擎
class ChatEngine:
    """本地 Ollama 对话 + 状态轮询（后台线程，结果经事件队列回 UI）。

    稳定性要点：
      * 改用 stream 流式读取：首字即可见，且能分别设「首字/停滞/总时长」三道
        时限，避免 Ollama 卡慢时桌宠无限停在「……」被误认为卡死。
      * 代际令牌（_gen）：丢弃过期/被用户取消的回复，防止旧回答突然后补弹出。
      * busy 守卫：思考中拒绝重复发起，避免后台线程堆积把 Ollama 打爆。
    """

    # 流式读取的三道时限（秒）
    # FIRST_TOKEN 给到 200 秒：7B 模型在 U 盘/机械盘上冷加载实测能到 150 秒以上，
    # 旧值 60 秒会在模型其实正在加载时先报「太慢」，用户再点一次又撞在加载中，
    # 看起来就是「Ollama 卡住」。配合 keep_alive=30m，冷加载只会发生一次。
    SOCKET_TIMEOUT = 65     # 单次网络读的最长阻塞（兜底）
    FIRST_TOKEN = 200       # 允许模型冷加载 / 出首个字的最长时间
    STALL = 35              # 出字后，超过该秒数无新内容判为停滞
    TOTAL = 300             # 单轮绝对上限

    def __init__(self, app):
        self.app = app
        self.online = None      # None=未知 True/False
        self.busy = False
        self._stop = threading.Event()
        self._gen = 0           # 代际令牌
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        threading.Thread(target=self._poll_loop, daemon=True).start()
        # 启动后悄悄预热文本模型：避免第一次点桌宠要等一两分钟冷加载
        threading.Thread(target=self._warmup_later, daemon=True).start()

    def _warmup_later(self):
        """延迟几秒等 Ollama 就绪，再把模型预热进显存并常驻。"""
        try:
            if self._stop.wait(4.0):
                return
            ollama_stream.warmup_model(OLLAMA_HOST, CHAT_MODEL,
                                       keep_alive=ollama_stream.DEFAULT_KEEP_ALIVE,
                                       timeout=240.0)
        except Exception:  # noqa: BLE001  预热失败无所谓，只是首次慢一点
            pass

    def _poll_loop(self):
        while not self._stop.is_set():
            self._check()
            self._stop.wait(5.0)

    def _check(self):
        try:
            with self._opener.open(OLLAMA_HOST + "/api/tags", timeout=4) as r:
                self.online = r.status == 200
        except Exception:
            self.online = False

    def ask(self, question):
        """发起一轮对话；思考中返回 False（不重复发起）。"""
        if self.busy:
            return False
        self.busy = True
        self._gen += 1
        my_gen = self._gen
        threading.Thread(target=self._ask_thread, args=(question, my_gen),
                         daemon=True).start()
        return True

    def invalidate(self):
        """让当前在途的回复失效（用户已切走/关闭对话框时用）。"""
        self._gen += 1

    def _ask_thread(self, question, my_gen):
        answer, err = "", None
        try:
            answer, err = self._stream_chat(question, my_gen)
        finally:
            self.busy = False
        # 代际校验：若用户已发起新对话或切走，丢弃过期回复
        if my_gen != self._gen:
            return
        self.app.post_event(("chat_done", answer, err))

    def _stream_chat(self, question, my_gen=None):
        """流式请求 Ollama，返回 (answer, err)。err 非空即失败（answer 可能为部分）。"""
        def on_token(text):
            # 代际校验：用户已点掉/切走后，旧流仍可能继续吐字。
            # 不再转发，否则会把桌宠重新拽回 think 态并永远卡在"……"。
            if my_gen is not None and my_gen != self._gen:
                return
            self.app.post_event(("chat_token", text))
        text, err = stream_ollama_chat(
            [{"role": "system", "content": PERSONA},
             {"role": "user", "content": question}],
            model=CHAT_MODEL, host=OLLAMA_HOST, temperature=0.6,
            on_token=on_token,
            first_token=self.FIRST_TOKEN, stall=self.STALL, total=self.TOTAL,
            socket_timeout=self.SOCKET_TIMEOUT)
        if err:
            return text, err
        # 清洗：LaTeX → 纯文本；emoji → 颜文字（桌宠说话不用表情）
        return demojify(sanitize_math(text)), None


# ---------------------------------------------------------------- 桌宠窗口
class PetWindow(LayeredWindow):
    def __init__(self, app):
        self.app = app
        super().__init__("J022Pet", "J022 Ghoul Pet")

    def _wndproc(self, hwnd, msg, w, l):
        return self.app.wndproc(hwnd, msg, w, l)


# ---------------------------------------------------------------- 桌宠应用
class PetApp:
    TICK = 50  # ms
    LONG_PRESS = 0.3  # 长按拖动阈值（秒）：按住这么久才进入拖动
    BLINK_DURATION = 0.24  # 一次眨眼的总时长（秒）

    def __init__(self, x=None, y=None, chat=True, test_seconds=None):
        self.name = "Ghoo"      # 桌宠的名字（代号 J022 Ghoul）
        self.chat_enabled = chat
        self.frames = self._load_frames()
        self.pet = PetWindow(self)
        self.bubble = BubbleWindow()
        self.chat = ChatEngine(self)
        self.voice = get_voice()   # 赛博机械女声（后台线程，可打断）
        self.events = deque()
        self._event_lock = threading.Lock()

        sw = user32.GetSystemMetrics(SM_CXSCREEN)
        wa = RECT()
        user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(wa), 0)
        self.work = (wa.left, wa.top, wa.right - wa.left, wa.bottom - wa.top)

        pw, ph = self.frames["idle_0"].size
        if x is None:
            x = self.work[0] + self.work[2] - pw - 40
        if y is None:
            y = self.work[1] + self.work[3] - ph - 4
        self.x, self.y = int(x), int(y)

        # 赛博朋克装饰层（霓虹描边 / HUD / 扫描线 / 环绕粒子 / 错误闪光）
        fw, fh = self.frames["idle_0"].size
        self.fx = GlitchFX(fw, fh)

        # 动画状态
        self.state = "idle"
        self.anim_t = 0
        self.face_right = True
        self.vy = 0.0
        self.target_x = None
        self.next_action = time.time() + random.uniform(3, 8)
        self.next_blink = time.time() + random.uniform(1.0, 2.5)
        # 自动休息：约 10 分钟无强交互则去睡一会儿（睡眠时长也约 10 分钟后自动醒来）
        self.next_rest = time.time() + 600
        self.next_wake = 0
        # 话痨：非休息态每隔随机时间（≤60s）吐槽一句
        self.next_chatter = time.time() + random.uniform(15, 45)
        self.blink_until = 0.0
        self.talk_until = 0.0
        self.wave_until = 0.0
        self.bubble_text = ""
        self.bubble_shown = 0
        self._drag_off = None
        self._drag_armed = False
        self._drag_moved = 0.0
        self._drag_t0 = 0.0
        self._press_xy = (0, 0)
        self._tilt = 0.0
        self._floor = self.work[1] + self.work[3] - ph - 4
        self._dialog = None
        self._reply_cancelled = False   # 本轮回复是否已被用户点掉（丢弃其后所有事件）
        self._last_frame = self.frames["idle_0"]
        self._test_until = (time.time() + test_seconds) if test_seconds else None

        user32.SetTimer(self.pet.hwnd, 1, self.TICK, None)
        self.pet.blit(self.frames["idle_0"], (self.x, self.y))
        self.pet.show()

    def _load_frames(self):
        frames = {}
        for name in ("idle_0", "idle_1", "idle_2", "idle_3",
                     "walk_0", "walk_1", "walk_2", "walk_3",
                     "blink_1", "blink_2", "blink_3",
                     "squash", "stretch", "sleep", "talk",
                     "wave_0", "wave_1", "wave_2", "wave_3"):
            p = os.path.join(FRAMES_DIR, f"{name}.png")
            frames[name] = Image.open(p).convert("RGBA")
        return frames

    # ---------------------------------------------------------- 事件
    def post_event(self, ev):
        with self._event_lock:
            self.events.append(ev)
        user32.PostMessageW(self.pet.hwnd, WM_APP_EVENT, 0, 0)

    def _handle_events(self):
        with self._event_lock:
            evs = list(self.events)
            self.events.clear()
        for ev in evs:
            # 用户已点掉这一轮：其后排队的旧 token / 旧回答一律丢弃，
            # 否则会把桌宠重新拽回 think 态，并把收起计时推到很久以后（表现为卡住）
            if self._reply_cancelled and ev[0] in ("chat_token", "chat_done"):
                continue
            if ev[0] == "chat_token":
                # 流式片段：让气泡实时出字（不发声，避免把半句念出来）
                _, text = ev
                self.bubble_text = text
                self.bubble_shown = 0
                self.talk_until = time.time() + 99999
                if self.state not in ("think", "talk"):
                    self.state = "think"
            elif ev[0] == "chat_done":
                _, answer, err = ev
                if err or not answer:
                    self.say(f"（AI 出错了：{(err or '空响应')[:60]}）", hold=6)
                else:
                    self.say(answer, hold=12, interrupt=True)
                self.state = "talk"

    # ---------------------------------------------------------- 动作
    def say(self, text, hold=10, interrupt=False):
        self.bubble_text = text
        self.bubble_shown = 0
        self.talk_until = time.time() + hold
        self.bubble.show_text("")   # 先收起，tick 里逐字显示
        # 语音：休眠态不发声（"Z z z…" 不读）；其余（话痨 / 聊天 / 打招呼 / 报错）
        # 都会用赛博机械女声念出来。interrupt=True 时打断正在播放的更早语音。
        if self.state != "sleep":
            try:
                self.voice.speak(text, interrupt=interrupt)
            except Exception:
                pass

    def walk_to(self, tx):
        self.target_x = max(self.work[0] + 4,
                            min(tx, self.work[0] + self.work[2] - 80))
        self.state = "walk"

    def jump(self):
        if self.state in ("drag", "fall"):
            return
        self.state = "jump"
        self.vy = -17.0

    def sleep(self):
        self.state = "sleep"
        self.say("Z z z…", hold=99999)

    def wake(self):
        if self.state == "sleep":
            self.state = "idle"
            self.bubble.hide()
            self.talk_until = 0
        # 醒来（或被叫醒/自动醒来）后重置自动休息计时：再过约 10 分钟才去休息
        self.next_wake = 0
        self.next_rest = time.time() + 600

    def go_corner(self):
        self.walk_to(self.work[0] + self.work[2] - 130)

    GREETS = [
        "今天也要加油哦～",
        "在的在的，随时听候吩咐！",
        "唔……好困，但看到你就不困了",
        "要我陪你画图吗？",
        "嘿嘿，被你发现啦",
        "工作辛苦啦，休息一下吧",
    ]

    # 话痨池：非休息态随机吐槽，大姐姐 + 黑客风范，调侃、有趣
    CHATTER = [
        '检测到你打开了十个小窗，姐姐帮你关了九个，留一个专心~',
        '沙箱已隔离，先把 DXF 导出来再说~',
        '命令行敲错三次，姐姐给你记一笔小账 (◕‿◕)',
        '图层没锁好，权限不足请先肝完这版图纸~',
        '命令行敲错三次，姐姐已接管键盘五分钟',
        'sudo 摸鱼已被拦截，罚画十张练手',
        '你这波鼠标轨迹，像极了没对齐的尺寸链，调一调~',
        '咖啡洒键盘上了，0day 漏洞都给你玩出花了哈哈~',
        '姐姐给你发了个表情包：一个催图的猫猫，凶但可爱~',
        '图层没锁好，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '姐姐给你开了个『夸夸群』，群里只有我一个，专门夸你~',
        '渲染卡住了，姐姐已接管键盘五分钟',
        '你这拖延，比垃圾回收还慢，赶紧提交 (¬‿¬)',
        '你忘保存了，姐姐给你记一笔小账 (◕‿◕)',
        '日志已归档，再浪就 kill 了',
        '图层没锁好，再浪就 SIGTERM 伺候 (`ε´)',
        '咖啡洒键盘上了，cat 了你的草稿，挺有意思嘛~',
        '咖啡洒键盘上了，罚你多画十张当练手~',
        'DXF 还没导出，给你超个频，麻溜点儿 (*´艸｀)',
        '防火墙新规上线，这波不亏',
        '检测到你在偷乐，是不是又偷懒成功了的那种乐 (¬‿¬)',
        '你又在划水，姐姐已接管键盘五分钟',
        '你忘保存了，0day 漏洞都给你玩出花了哈哈~',
        '沙箱已隔离，别想抵赖 (¬‿¬)',
        '命令行敲得挺溜，画起图来怎么像手抖 (¬‿¬)',
        '你盯着空白画布的样子，像极了 wait() 卡死的线程 (◕ᗺ◕)',
        '你又在划水，防火墙已自动补上 (￣▽￣)ノ',
        '咖啡洒键盘上了，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '你画的直线歪了 0.5 度，姐姐的显微镜都看出来了 (｡•̀ᴗ-)✧',
        '咖啡洒键盘上了，姐姐已接管键盘五分钟',
        '你画的直线终于直了，姐姐鼓掌，啪啪啪 (◕ᗺ◕)',
        '咖啡洒键盘上了，再浪就 SIGTERM 伺候 (`ε´)',
        '限速策略生效，先把 DXF 导出来再说~',
        '自动备份已完成，截图存证了',
        '你对着空白画布发呆，防火墙已自动补上 (￣▽￣)ノ',
        '端口扫了一遍，发现你又在发呆… 批个 timeout 给你续命 (◕ᴗ◕)',
        '你在摸鱼，姐姐已接管键盘五分钟',
        '你对着空白画布发呆，cat 了你的草稿，挺有意思嘛~',
        '你又在划水，再浪就 SIGTERM 伺候 (`ε´)',
        '你那套快捷键，比我奶奶织毛衣还慢，提速提速！',
        '你对着空白画布发呆，姐姐给你记一笔小账 (◕‿◕)',
        'agent 已部署，截图存证了',
        '你对着空白画布发呆，姐姐已接管键盘五分钟',
        'sudo 摸鱼已被拦截，这波不亏',
        '缓存已清理，再浪就 kill 了',
        '检测到你在偷看网页，进程 kill 警告已发出 (`ε´) 乖乖画图',
        '再摸一会儿鱼，姐姐就把你的桌面换成 CAD 快捷键表了哦~',
        '日志已归档，快去搬砖',
        '防火墙新规上线，罚画十张练手',
        '别跟姐姐比懒，姐姐可是 7×24 在线监督你呢~',
        '限速策略生效，麻溜点儿 (*´艸｀)',
        '系统提示：检测到可爱生物正在偷懒，启动投喂程序 (づ｡◕‿‿◕｡)づ',
        '防火墙新规上线，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '你忘保存了，cat 了你的草稿，挺有意思嘛~',
        '自动备份已完成，再浪就 kill 了',
        '你在摸鱼，cat 了你的草稿，挺有意思嘛~',
        'root 权限已收回，小笨蛋 (◕‿◕)',
        '再摸鱼，姐姐就把你的微信签名改成『正在肝图，勿扰』~',
        '再发呆，姐姐就把你的椅子调成站立模式，站着画！',
        'sleep 模式已禁用，今天你是姐姐的强制在线进程~',
        '沙箱已隔离，快去搬砖',
        '你在摸鱼，罚你多画十张当练手~',
        '你在偷看网页，姐姐给你记一笔小账 (◕‿◕)',
        '端口扫描完成，先把 DXF 导出来再说~',
        'git 提交了没，快去搬砖',
        '防火墙新规上线，再浪就 kill 了',
        '你那鼠标，双击像在敲门，轻点儿，姐姐听得见~',
        '缓存已清理，截图存证了',
        '检测到你在搜『CAD 怎么画直线』，姐姐默默点了个赞~',
        '你在偷看网页，姐姐已接管键盘五分钟',
        '图层没锁好，批个 timeout 给你续命好了 (｡•̀ᴗ-)✧',
        '检测到你在偷偷点赞，姐姐假装没看见，继续画哦 (◕ᗺ◕)',
        '内核已加固，麻溜点儿 (*´艸｀)',
        '自动备份已完成，先把 DXF 导出来再说~',
        '你这算力，跑个渲染都喘，姐姐给你超个频 (*´艸｀)',
        '你在摸鱼，姐姐给你记一笔小账 (◕‿◕)',
        '自动备份已完成，别想抵赖 (¬‿¬)',
        '你这操作行云流水，就是画出来的东西在流汗 (¬‿¬)',
        '姐姐给你设了番茄钟：画 25 分钟，夸你 5 分钟，成交不~',
        'git 提交了没，麻溜点儿 (*´艸｀)',
        '你在摸鱼，git 都看不下去要报警了 (¬_¬)',
        '再划水，姐姐就把你的屏保换成『快去画图』滚动字幕了~',
        '防火墙新规上线，先把 DXF 导出来再说~',
        '再不保存，姐姐就要用 rm -rf 给你演示什么叫教训了，开玩笑的啦~',
        '姐姐给你泡了杯虚拟咖啡，喝完这口就把图交了呗 (◕‿◕)',
        'DXF 还没导出，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '限速策略生效，这波不亏',
        'sudo 摸鱼已被拦截，别想抵赖 (¬‿¬)',
        'sudo 摸鱼已被拦截，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '你对着空白画布发呆，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '姐姐给你煮了碗虚拟泡面，趁热画完这版再吃~',
        '你忘保存了，给你超个频，麻溜点儿 (*´艸｀)',
        '渲染卡住了，cat 了你的草稿，挺有意思嘛~',
        '沙箱已隔离，小笨蛋 (◕‿◕)',
        'sudo 摸鱼已被拦截，截图存证了',
        '你在偷看网页，cat 了你的草稿，挺有意思嘛~',
        'DXF 还没导出，cat 了你的草稿，挺有意思嘛~',
        '缓存已清理，这波不亏',
        '你的鼠标轨迹，像极了没对齐的标注线，哈哈~',
        '鼠标轨迹歪了，0day 漏洞都给你玩出花了哈哈~',
        '限速策略生效，快去搬砖',
        '渲染卡住了，0day 漏洞都给你玩出花了哈哈~',
        '你的图层命名是『新建图层 7』？姐姐的强迫症犯了 (¬_¬)',
        'commit 了吗？没 commit 就关机，小心 git 给你颜色看看 (¬_¬)',
        '日志已归档，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '专注度掉到 0%，cat 了你的草稿，挺有意思嘛~',
        '你刚才切出去的那一下，已被姐姐记在小本本第 233 页~',
        '你画的交点，比你的社交圈还密，热闹得很 (◕‿◕)',
        '端口扫描完成，快去搬砖',
        '你忘保存了，权限不足请先肝完这版图纸~',
        '自动备份已完成，快去搬砖',
        '日志已归档，小笨蛋 (◕‿◕)',
        '系统已为你清除缓存里的懒惰，刷新一下重新做人吧~',
        '你这手绘功底，藏在 CAD 里可惜了，偶尔也炫一下嘛~',
        '专注度掉到 0%，权限不足请先肝完这版图纸~',
        '你又在划水，罚你多画十张当练手~',
        '你又在划水，0day 漏洞都给你玩出花了哈哈~',
        '鼠标轨迹歪了，再浪就 SIGTERM 伺候 (`ε´)',
        '命令行敲错三次，0day 漏洞都给你玩出花了哈哈~',
        'DXF 还没导出，git 都看不下去要报警了 (¬_¬)',
        '自动备份已完成，麻溜点儿 (*´艸｀)',
        '图层没锁好，姐姐给你记一笔小账 (◕‿◕)',
        '咖啡洒键盘上了，姐姐给你记一笔小账 (◕‿◕)',
        'root 权限已收回，截图存证了',
        'root 权限已收回，麻溜点儿 (*´艸｀)',
        'DXF 还没导出，权限不足请先肝完这版图纸~',
        '姐姐给你开了个后门：画完这版就准你刷五分钟视频~',
        '专注度掉到 0%，0day 漏洞都给你玩出花了哈哈~',
        '图层没锁好，git 都看不下去要报警了 (¬_¬)',
        '姐姐给你泡的茶凉了，画完这版赶紧喝，别拼命~',
        '你忘保存了，再浪就 SIGTERM 伺候 (`ε´)',
        '咖啡洒键盘上了，权限不足请先肝完这版图纸~',
        '姐姐帮你把杂乱的线规整了一下，世界瞬间清爽 (◕‿◕)',
        'agent 已部署，罚画十张练手',
        '日志已归档，先把 DXF 导出来再说~',
        'DXF 还没导出，0day 漏洞都给你玩出花了哈哈~',
        '摸鱼被姐姐抓到了哦，小笨蛋~ 我可是开着进程监控的呢 (◕‿◕)',
        '自动备份已完成，罚画十张练手',
        '鼠标轨迹歪了，姐姐给你记一笔小账 (◕‿◕)',
        '内核已加固，再浪就 kill 了',
        '端口扫描完成，麻溜点儿 (*´艸｀)',
        'DXF 还没导出，罚你多画十张当练手~',
        '再摸鱼，姐姐就把你的头像换成『画图机器』表情包~',
        '你在摸鱼，权限不足请先肝完这版图纸~',
        '防火墙新规上线，麻溜点儿 (*´艸｀)',
        '你画的点，比你的存在感还小，放大点让姐姐看见~',
        '再划水，姐姐就把你的项目名改成『薛定谔的完工』~',
        '后门没关好哦，姐姐帮你 firewall 一下 (￣▽￣)ノ',
        '日志已归档，别想抵赖 (¬‿¬)',
        '你在偷看网页，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '你刚才那声叹气，已录入姐姐的『撒娇抵抗训练库』了~',
        '你画的斜线，比你的歪理还斜，扶正一下嘛 (｡•̀ᴗ-)✧',
        '日志已归档，截图存证了',
        'root 权限已收回，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '你这光标移动速度，像极了加载进度条，急死姐姐了~',
        '你在摸鱼，0day 漏洞都给你玩出花了哈哈~',
        '你忘保存了，罚你多画十张当练手~',
        '你在偷看网页，git 都看不下去要报警了 (¬_¬)',
        '你对着空白画布发呆，权限不足请先肝完这版图纸~',
        'git 提交了没，再浪就 kill 了',
        '日志已归档，这波不亏',
        '缓存已清理，别想抵赖 (¬‿¬)',
        '鼠标轨迹歪了，防火墙已自动补上 (￣▽￣)ノ',
        '你忘保存了，git 都看不下去要报警了 (¬_¬)',
        '你画的多段线，拐弯比你的思路还急，慢点儿~',
        '你说你不会画？那刚才偷偷画的火柴人是谁 (¬‿¬)',
        '命令行敲错三次，批个 timeout 给你续命好了 (｡•̀ᴗ-)✧',
        '你画的线比你的发量还少，多加几根嘛 (｡•̀ᴗ-)✧',
        '检测到你在揉眼睛，姐姐远程调暗了屏幕护眼~',
        '姐姐给你开了『专注模式』，但这模式只防别人不防你哦~',
        '限速策略生效，小笨蛋 (◕‿◕)',
        '你在偷看网页，批个 timeout 给你续命好了 (｡•̀ᴗ-)✧',
        '专注度掉到 0%，再浪就 SIGTERM 伺候 (`ε´)',
        '你又在划水，cat 了你的草稿，挺有意思嘛~',
        'sudo 摸鱼已被拦截，先把 DXF 导出来再说~',
        'root 权限已收回，别想抵赖 (¬‿¬)',
        '你对着空白画布发呆，批个 timeout 给你续命好了 (｡•̀ᴗ-)✧',
        '缓存已清理，小笨蛋 (◕‿◕)',
        '你那声『就画一下下』，已经从下午说到晚上了 (¬_¬)',
        '端口扫描完成，别想抵赖 (¬‿¬)',
        '自动备份已完成，这波不亏',
        '姐姐的雷达显示：你附近有逃课的气息，抓到了吧~',
        '你画的标注箭头，比你的方向感还迷，姐姐帮你指路~',
        '端口扫描完成，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '命令行敲错三次，权限不足请先肝完这版图纸~',
        '端口扫描完成，截图存证了',
        '别浪，图层没锁就别乱改，姐姐可不背锅~',
        'DXF 还没导出，再浪就 SIGTERM 伺候 (`ε´)',
        '内核已加固，先把 DXF 导出来再说~',
        '咖啡洒键盘上了，防火墙已自动补上 (￣▽￣)ノ',
        '渲染卡住了，再浪就 SIGTERM 伺候 (`ε´)',
        '你刚才那个回车，力度之大，隔壁工位的同事都听见了哈哈~',
        '日志显示：某人专注度 0%，建议来杯咖啡再战 (◕ᗺ◕)',
        '内核已加固，罚画十张练手',
        '你又在划水，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '咖啡洒键盘上了，git 都看不下去要报警了 (¬_¬)',
        '你那『马上就好』，姐姐已经设了倒计时，滴答滴答~',
        '别总等灵感，灵感是个懒鬼，你不动它也不动 (｡•̀ᴗ-)✧',
        '沙箱已隔离，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '鼠标轨迹歪了，权限不足请先肝完这版图纸~',
        '内核已加固，别想抵赖 (¬‿¬)',
        '防火墙新规上线，别想抵赖 (¬‿¬)',
        '限速策略生效，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '专注度掉到 0%，姐姐给你记一笔小账 (◕‿◕)',
        '你画的线宽，比你的发际线还细，加粗点嘛 (◕ᗺ◕)',
        '限速策略生效，别想抵赖 (¬‿¬)',
        '你画的抛物线，比你的运势还上扬，今天必完工~',
        '检测到你在划水，权限不足，请先肝完这版图纸 (｡•̀ᴗ-)✧',
        '姐姐在后台给你点了『加油』，算力 +1，趁热画~',
        '你忘保存了，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '内核已加固，姐姐盯着呢 (｡•̀ᴗ-)✧',
        '命令行敲错三次，cat 了你的草稿，挺有意思嘛~',
        '图层没锁好，罚你多画十张当练手~',
        '你画的多边形怎么少了一条边，是预算不足吗 (｡•̀ᴗ-)✧',
        '再不走神，姐姐就把你的桌面宠物设成监督员，就是我啦~',
        '命令行敲错三次，防火墙已自动补上 (￣▽￣)ノ',
        '你忘保存了，防火墙已自动补上 (￣▽￣)ノ',
        '沙箱已隔离，这波不亏',
        '别总说难，难的都让姐姐的算法替你扛了，你只管画~',
        'sudo 摸鱼已被拦截，麻溜点儿 (*´艸｀)',
        '限速策略生效，再浪就 kill 了',
        '缓存已清理，快去搬砖',
        '图层没锁好，防火墙已自动补上 (￣▽￣)ノ',
        '鼠标轨迹歪了，cat 了你的草稿，挺有意思嘛~',
        '你又在划水，给你超个频，麻溜点儿 (*´艸｀)',
        '你在偷看网页，罚你多画十张当练手~',
        '命令行敲错三次，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '专注度掉到 0%，git 都看不下去要报警了 (¬_¬)',
        '鼠标轨迹歪了，检测到可爱值超标，启动撒娇进程 (づ｡◕‿‿◕｡)づ',
        '你对着空白画布发呆，0day 漏洞都给你玩出花了哈哈~',
        'root 权限已收回，先把 DXF 导出来再说~',
        '你又在划水，权限不足请先肝完这版图纸~',
        '防火墙新规上线，快去搬砖',
        '你画的矩形，比你的作息还方方正正，规律得很 (◕‿◕)',
        '你在偷看网页，防火墙已自动补上 (￣▽￣)ノ',
        '鼠标轨迹歪了，姐姐已接管键盘五分钟',
        '端口扫描完成，小笨蛋 (◕‿◕)',
        '你在偷看网页，给你超个频，麻溜点儿 (*´艸｀)',
        '你忘保存了，批个 timeout 给你续命好了 (｡•̀ᴗ-)✧',
        'git 提交了没，罚画十张练手',
        '你对着空白画布发呆，再浪就 SIGTERM 伺候 (`ε´)',
        '咖啡洒键盘上了，批个 timeout 给你续命好了 (｡•̀ᴗ-)✧',
        '缓存已清理，先把 DXF 导出来再说~',
        '别跟姐姐说你不会，你昨天还说自己是高手呢 (¬_¬)',
        'git 提交了没，截图存证了',
        '日志已归档，罚画十张练手',
        '鼠标轨迹歪了，git 都看不下去要报警了 (¬_¬)',
        '检测到你左手奶茶右手鼠标，算力过载，建议放下奶茶~',
        '你画的图，比你的自拍还讲究角度，有天赋 (◕ᗺ◕)',
        '你在偷看网页，0day 漏洞都给你玩出花了哈哈~',
        'git 提交了没，先把 DXF 导出来再说~',
        'root 权限已收回，这波不亏',
        '你在摸鱼，批个 timeout 给你续命好了 (｡•̀ᴗ-)✧',
        '姐姐帮你在图上盖了个『已完成』的章，就差你按下去了~',
        '你那句『最后改一版』，姐姐已经听第 12 次了 (¬_¬)',
        '别急着关机，先把这版存了，姐姐可不想明天陪你重画~',
    ]

    @classmethod
    def pick_chatter(cls):
        """从话痨池里挑【一句】。

        防呆：万一池子里的条目被误写成多句（换行连写、或漏逗号被 Python
        隐式拼成一条超长串），这里只保留第一句、并限制长度，
        保证气泡每次只出一句，不再一次刷一大串。
        """
        pool = cls.CHATTER or ["……"]
        s = str(random.choice(pool)).replace("\r", "\n")
        s = s.split("\n")[0].strip()          # 多行只取第一行
        if len(s) > 46:                        # 过长则按句末截断，只留第一句
            for k, ch in enumerate(s):
                if k >= 10 and ch in "。！？~～":
                    s = s[:k + 1]
                    break
            else:
                s = s[:46]
        return s.strip() or "……"

    def greet(self):
        """打个招呼：随机说一句，并做个小动作（挥手）。"""
        self.wake()
        self.say(random.choice(self.GREETS), hold=5)
        if self.state not in ("drag", "fall"):
            self.wave()

    def wave(self):
        """挥手打招呼：右臂抬起并前后摆动约 2.6 秒后回到待机。"""
        if self.state in ("drag", "fall"):
            return
        self.state = "wave"
        self.wave_until = time.time() + 2.6

    def open_chat(self):
        if not self.chat_enabled:
            self.say("（对话功能已关闭）", hold=4)
            return
        if self.state == "sleep":
            self.wake()
        # 单实例 + 切换：对话框已打开时，再点人物 = 关闭它。
        # 修正"越点越多、刷屏且关不掉"的 bug —— 旧实现用嵌套消息泵，
        # 每点一次就再开一个泵，层层堆叠；对话框又因宠物窗口 NOACTIVATE
        # 收不到失活消息而关不掉。现改为非模态窗口 + 单实例。
        if self._dialog is not None and getattr(self._dialog, "open", False):
            self._dialog.close()
            self._dialog = None
            return
        self._dialog = InputDialog(self)
        # 非模态：对话框是普通顶层窗口，由主消息循环统一驱动；
        # 提交/取消通过 InputDialog 直接回调 _on_chat_submit，不再开嵌套泵。

    def _on_chat_submit(self, text):
        """输入框提交后的回调（非模态，由 InputDialog 直接调用）。"""
        self._dialog = None
        if text:
            if not self.chat.ask(text):
                # 上一轮还在思考，不要叠加请求
                self.say("（正在想呢，先等一下下~）", hold=4)
                return
            # 新一轮提问：解除上一轮的"已取消"标记，重新接收流式事件
            self._reply_cancelled = False
            # 用户发起对话 = 强交互，自动休息计时顺延 10 分钟
            self.next_rest = time.time() + 600
            # 进入思考态：先用占位符，流式片段到达后会实时出字
            self.state = "think"
            self.bubble_text = "……"
            self.bubble_shown = 0
            self.talk_until = time.time() + 99999
            self.bubble.show_text("")

    # ---------------------------------------------------------- 拖动
    def _left_down(self):
        """左键当前是否物理按下（不依赖窗口消息送达）。"""
        return bool(user32.GetAsyncKeyState(VK_LBUTTON) & 0x8000)

    def _cancel_drag(self):
        """强制结束拖动（不触发任何动作）。"""
        if self._drag_off is None:
            return
        try:
            user32.ReleaseCapture()
        except Exception:  # noqa: BLE001
            pass
        self._drag_off = None
        self._drag_armed = False
        self._tilt = 0.0
        try:
            user32.SetCursor(user32.LoadCursorW(None, IDC_ARROW))
        except Exception:  # noqa: BLE001
            pass

    def _update_drag(self):
        """按住期间跟随光标；超过长按阈值才真正开始拖动。"""
        if self._drag_off is None:
            return
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        self._drag_moved = max(
            self._drag_moved,
            abs(pt.x - self._press_xy[0]) + abs(pt.y - self._press_xy[1]))
        if not self._drag_armed:
            if time.time() - self._drag_t0 < self.LONG_PRESS:
                return
            # 超过阈值 → 进入拖动态（光标变四向箭头作为反馈）
            self._drag_armed = True
            self.state = "drag"
            user32.SetCursor(user32.LoadCursorW(None, IDC_SIZEALL))
        nx = pt.x - self._drag_off[0]
        ny = pt.y - self._drag_off[1]
        self._tilt = max(-12.0, min(12.0,
                                     self._tilt * 0.6 + (nx - self.x) * 1.2))
        self.x, self.y = nx, ny
        self._render(self.frames["idle_0"])

    def _end_drag(self):
        """松手处理：区分"拖过"/"长按未动"/"快速滑动"/"短按点击"。"""
        if self._drag_off is None:
            return
        was_armed = self._drag_armed
        held = time.time() - self._drag_t0
        moved = self._drag_moved
        self._cancel_drag()
        if was_armed:
            # 拖动结束 → 松手掉落
            self.state = "fall"
            self.vy = 0.0
            return
        if held >= self.LONG_PRESS:
            # 长按但没拖动：视为"想抓起来"，静默取消，不弹对话框
            return
        if moved > 8:
            # 快速滑动但未达长按阈值：误触，静默取消
            return
        # 短按 = 单击对话。用投递消息的方式，避免在 tick() 里开嵌套消息泵
        if self.state == "sleep":
            self.wake()
        elif self.state in ("think", "talk"):
            # 被她"正在回答"时点掉：立刻闭嘴收气泡，并让在途回复作废
            self.bubble.hide()
            self.bubble_text = ""
            self.bubble_shown = 0
            self.talk_until = 0.0
            self.state = "idle"
            self._reply_cancelled = True
            try:
                self.chat.invalidate()
            except Exception as e:  # noqa: BLE001
                _log_cb_error("_end_drag.invalidate", e)
        user32.PostMessageW(self.pet.hwnd, WM_APP_CHAT, 0, 0)

    # ---------------------------------------------------------- 右键菜单
    def show_menu(self):
        """弹出右键菜单。TrackPopupMenu 返回被选中的命令 ID（0=取消）。

        菜单打开期间把赛博装饰切到"故障形态"（持续跳动乱码），
        关闭后恢复。TrackPopupMenu 内部会泵消息，宠物定时器仍触发
        tick() 重绘，因此故障背景会持续跳动。
        """
        fx = getattr(self, "fx", None)
        if fx is not None:
            fx.menu_mode = True
            self._render(self._last_frame or self.frames["idle_0"])
        hmenu = user32.CreatePopupMenu()
        if not hmenu:
            _last_error("CreatePopupMenu")
            if fx is not None:
                fx.menu_mode = False
            return
        for label, cid in self.menu_items():
            user32.AppendMenuW(hmenu, MF_STRING, cid, label)

        # 语音子菜单：三声线可选 + 开关
        hvoice = user32.CreatePopupMenu()
        v = self.voice
        for idx, (key, label) in enumerate(v.profile_list()):
            mark = "● " if v.profile == key else "　 "
            user32.AppendMenuW(hvoice, MF_STRING, 20 + idx, mark + label)
        user32.AppendMenuW(hvoice, MF_STRING, 23,
                           "关闭语音" if v.enabled else "开启语音")
        user32.AppendMenuW(hmenu, MF_POPUP, hvoice, "语音")

        # 语速子菜单：慢 / 正常 / 快
        hspeed = user32.CreatePopupMenu()
        for idx, (key, label) in enumerate(v.speed_list()):
            mark = "● " if v.speed == key else "　 "
            user32.AppendMenuW(hspeed, MF_STRING, 30 + idx, mark + label)
        user32.AppendMenuW(hmenu, MF_POPUP, hspeed, "语速")

        # 「退出」放最底部（分隔线隔开），并加二次确认——防止误点导致"闪退"
        user32.AppendMenuW(hmenu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(hmenu, MF_STRING, 7, "退出…")

        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        user32.SetForegroundWindow(self.pet.hwnd)
        cmd = user32.TrackPopupMenu(hmenu, TPM_RETURNCMD | TPM_NONOTIFY,
                                    pt.x, pt.y, 0, self.pet.hwnd, None)
        user32.DestroyMenu(hmenu)
        # 让菜单在失去焦点后能正确关闭
        user32.PostMessageW(self.pet.hwnd, WM_NULL, 0, 0)
        if fx is not None:
            fx.menu_mode = False
            self._render(self._last_frame or self.frames["idle_0"])
        if cmd:
            self.handle_menu_command(int(cmd))

    def menu_items(self):
        """菜单项定义：[(显示文本, 命令 ID)]。语音/语速/退出在 show_menu 里追加。

        注意：「退出」必须放在菜单最底部（分隔线之下、语音/语速子菜单之后）——
        右键菜单会从光标处向下铺开、整块盖在桌宠身上，若"退出"排在中间，
        用户"左键点桌宠"极可能误中它导致桌宠直接退出（看起来像 100% 崩溃）。
        """
        items = [("打个招呼", 1), ("聊一聊", 2), ("散散步", 3), ("跳一下", 4)]
        items.append(("醒醒", 5) if self.state == "sleep" else ("睡一会儿", 5))
        items += [("回到右下角", 6), ("打开 LibreCAD", 8)]
        return items

    def handle_menu_command(self, cmd):
        """按命令 ID 执行对应动作（独立成方法便于测试）。"""
        if cmd == 1:
            self.greet()
        elif cmd == 2:
            self.open_chat()
        elif cmd == 3:
            self.walk_to(random.uniform(self.work[0] + 40,
                                        self.work[0] + self.work[2] - 140))
        elif cmd == 4:
            self.jump()
        elif cmd == 5:
            self.wake() if self.state == "sleep" else self.sleep()
        elif cmd == 6:
            self.go_corner()
        elif cmd == 7:
            # 退出前二次确认：右键菜单盖在桌宠身上时极易误点"退出"
            r = user32.MessageBoxW(self.pet.hwnd,
                                   "确定要退出 Ghoo 吗？",
                                   "退出 Ghoo",
                                   MB_YESNO | MB_ICONQUESTION | MB_TOPMOST)
            if r == IDYES:
                self.quit()
        elif cmd == 8:
            self.open_librecad()
        elif cmd in (20, 21, 22):
            keys = [k for k, _ in self.voice.profile_list()]
            self.voice.set_profile(keys[cmd - 20])
            if self.voice.enabled:
                self.say("（语音已切换）", hold=4, interrupt=True)
        elif cmd == 23:
            if self.voice.enabled:
                self.voice.set_enabled(False)
                self.say("（语音已关闭）", hold=4)
            else:
                self.voice.set_enabled(True)
                self.say("（语音已开启）", hold=4)
        elif cmd in (30, 31, 32):
            skeys = [k for k, _ in self.voice.speed_list()]
            self.voice.set_speed(skeys[cmd - 30])
            if self.voice.enabled:
                self.say("（语速已调整）", hold=4, interrupt=True)

    def open_librecad(self):
        """右键菜单「打开 LibreCAD」：定位并启动 LibreCAD（不阻塞主循环）。"""
        exe = self._find_librecad()
        if not exe:
            self.say("没找到 LibreCAD，先装一下吧～", hold=6)
            return
        try:
            subprocess.Popen([exe], shell=False)
            self.say("已帮你打开 LibreCAD～", hold=5)
        except Exception as e:  # noqa: BLE001
            self.say(f"打开失败：{str(e)[:40]}", hold=6)

    @staticmethod
    def _find_librecad():
        """LibreCAD 定位统一交给 portable（便携版优先用包内副本）。"""
        from portable import find_librecad
        return find_librecad() or None

    # ---------------------------------------------------------- 主循环
    def tick(self):
        now = time.time()
        self._handle_events()
        self.anim_t += 1

        # ---- 气泡：打字机推进 + 超时收起（任何状态都生效）----
        speed = 2 if self.state in ("think", "talk") else 1
        self._typewriter(speed)
        if (self.talk_until and now > self.talk_until
                and self.bubble_shown >= len(self.bubble_text)):
            self.bubble.hide()
            self.bubble_text = ""
            self.bubble_shown = 0
            if self.state in ("think", "talk"):
                self.state = "idle"
        self._position_bubble()

        # ---- 话痨：非休息/拖动/掉落态，每隔随机时间（≤60s）吐槽一句 ----
        if (self.state not in ("sleep", "drag", "fall")
                and self.talk_until < now
                and now >= self.next_chatter):
            self.say(self.pick_chatter(), hold=random.uniform(6, 10))
            self.state = "talk"
            self.next_chatter = now + random.uniform(20, 55)

        px, py, pw, ph = self.pet.rect()

        # ---- 拖动：按物理按键状态判定，不依赖窗口消息送达 ----
        # 桌宠窗口带 WS_EX_NOACTIVATE，SetCapture 常失败 → 鼠标抬起时
        # WM_LBUTTONUP 可能发给下面的窗口，导致桌宠"粘"在光标上。
        # 这里每帧检查 VK_LBUTTON 的真实状态，按钮松开就一定收尾。
        if self._drag_off is not None:
            if self._left_down():
                self._update_drag()
            else:
                self._end_drag()

        if self._test_until and now >= self._test_until:
            self.quit()
            return

        if self.state == "drag":
            self._render(self.frames["idle_0"])
            return
        elif self.state == "walk":
            if self.target_x is None or abs(px - self.target_x) < 4:
                self.state = "idle"
                self.target_x = None
            else:
                step = 4 if px < self.target_x else -4
                self.x += step
                self.face_right = step > 0
                frame = self.frames[f"walk_{(self.anim_t // 4) % 4}"]
                if not self.face_right:
                    frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
                self._floor = self.work[1] + self.work[3] - ph - 4
                self.y = self._floor
                self._render(frame)
                return
        elif self.state in ("jump", "fall"):
            self.vy += 1.1
            self.y += int(self.vy)
            if self.y >= self._floor:
                self.y = self._floor
                self.vy = 0.0
                self.state = "idle"
                self._render(self.frames["squash"])
                return
            frame = self.frames["stretch"] if self.vy < 0 else self.frames["squash"]
            self._render(frame)
            return
        elif self.state == "sleep":
            if self.next_wake and now >= self.next_wake:
                self.wake()      # 自动醒来（状态转 idle，下方待机段会接着渲染）
            else:
                self._render(self.frames["sleep"])
                return
        elif self.state in ("think", "talk"):
            self._render(self.frames["talk"])
            return
        elif self.state == "wave":
            if now >= self.wave_until:
                self.state = "idle"
                self._render(self.frames["idle_0"])
                return
            frame = self.frames[f"wave_{(self.anim_t // 3) % 4}"]
            self._render(frame)
            return

        # ---- 待机 ----
        # 眨眼：半闭 → 全闭 → 半闭，播放完再回到 idle
        if now >= self.next_blink:
            self.blink_until = now + self.BLINK_DURATION
            self.next_blink = now + random.uniform(2.5, 6.0)
        if now < self.blink_until:
            idx = 1 + int((now - (self.blink_until - self.BLINK_DURATION))
                          / (self.BLINK_DURATION / 3)) % 3
            self._render(self.frames[f"blink_{idx}"])
        else:
            self._render(self.frames[f"idle_{(self.anim_t // 7) % 4}"])

        if self.state == "idle" and now >= self.next_action:
            r = random.random()
            if r < 0.45:
                self.walk_to(random.uniform(self.work[0] + 40,
                                            self.work[0] + self.work[2] - 140))
            elif r < 0.75:
                self.jump()
            elif r < 0.90 and self.talk_until < now:
                self.wave()      # 自由行动：招手（与打招呼同款），不止会跳
            self.next_action = now + random.uniform(8, 20)
        # 自动休息：每约 10 分钟无强交互则去睡一会儿（睡满 10 分钟自动醒来）
        if self.state == "idle" and now >= self.next_rest:
            self.sleep()
            self.next_wake = now + 600

    def _typewriter(self, speed=1):
        if not self.bubble_text:
            return
        if self.bubble_shown < len(self.bubble_text):
            self.bubble_shown = min(len(self.bubble_text),
                                    self.bubble_shown + speed)
            self.bubble.redraw(self.bubble_text[:self.bubble_shown])

    def _render(self, frame):
        self._last_frame = frame
        img = frame.copy()
        if self._tilt:
            img = img.rotate(self._tilt, resample=Image.BICUBIC, expand=False)
        base_for_hit = img                       # 干净躯干（未装饰），用于命中测试
        fx = getattr(self, "fx", None)
        if fx is not None:
            img = fx.decorate(img, self.state, self.anim_t, time.time())
        self.pet.blit(img, (self.x, self.y))
        # 装饰不参与抓取/点击：命中测试仍基于干净躯干，避免扩大可抓取区域
        self.pet._frame = base_for_hit

    def _position_bubble(self):
        if not self.bubble.visible or self.bubble._frame is None:
            return
        px, py, pw, ph = self.pet.rect()
        bw, bh = self.bubble._frame.size
        bx = px + pw // 2 - bw // 2
        by = max(2, py - bh - 6)
        sw = user32.GetSystemMetrics(SM_CXSCREEN)
        bx = max(2, min(bx, sw - bw - 2))
        self.bubble.blit(self.bubble._frame, (bx, by))

    # ---------------------------------------------------------- 窗口过程
    def wndproc(self, hwnd, msg, w, l):
        if msg == WM_TIMER:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001
                print("tick error:", e, flush=True)
            return 0
        if msg == WM_MOUSEACTIVATE:
            return MA_NOACTIVATE
        if msg == WM_NCHITTEST:
            x, y = _lo_word(l), _hi_word(l)
            px, py, pw, ph = self.pet.rect()
            if self.pet.alpha_at(x - px, y - py) > 24:
                return HTCLIENT
            return HTTRANSPARENT
        if msg == WM_LBUTTONDOWN:
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            px, py, pw, ph = self.pet.rect()
            # 长按拖动：按下只记录，不立即拖动（避免误触）
            self._drag_off = (pt.x - px, pt.y - py)
            self._press_xy = (pt.x, pt.y)
            self._drag_armed = False
            self._drag_moved = 0.0
            self._drag_t0 = time.time()
            user32.SetCapture(self.pet.hwnd)
            return 0
        if msg == WM_MOUSEMOVE and self._drag_off is not None:
            # 只累计位移，实际跟随由 tick() 轮询光标完成（不依赖消息送达）
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            self._drag_moved = max(
                self._drag_moved,
                abs(pt.x - self._press_xy[0]) + abs(pt.y - self._press_xy[1]))
            return 0
        if msg == WM_LBUTTONUP:
            # 正常的松手路径；丢消息时由 tick() 的按键状态检测兜底
            self._end_drag()
            return 0
        if msg == WM_APP_CHAT:
            self.open_chat()
            return 0
        if msg == WM_CAPTURECHANGED:
            # 鼠标捕获被别处抢走 → 必须清掉拖动态，否则桌宠会一直粘着光标
            self._cancel_drag()
            return 0
        if msg == WM_RBUTTONUP:
            self._cancel_drag()
            self.show_menu()
            return 0
        if msg == WM_SETCURSOR:
            # 悬停在角色身上显示手型光标，提示可点击
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            px, py, pw, ph = self.pet.rect()
            if self.pet.alpha_at(pt.x - px, pt.y - py) > 24:
                user32.SetCursor(user32.LoadCursorW(None, IDC_HAND))
            else:
                user32.SetCursor(user32.LoadCursorW(None, IDC_ARROW))
            return 0
        return user32.DefWindowProcW(hwnd, msg, w, l)

    def quit(self):
        """退出：仅投递退出消息（窗口销毁放到消息循环结束后，避免在窗口过程中销毁自身）。"""
        if getattr(self, "_quitting", False):
            return
        self._quitting = True
        user32.PostQuitMessage(0)

    def cleanup(self):
        """消息循环结束后真正释放窗口与 GDI 资源。"""
        for fn in (lambda: user32.KillTimer(self.pet.hwnd, 1),
                   self.bubble.destroy,
                   self.pet.destroy):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print("quit cleanup error:", e, flush=True)


# ---------------------------------------------------------------- 入口
def main():
    _enable_dpi_awareness()

    # 原生崩溃现场落盘：若哪天仍出现 0xc0000005 / 0xc000041d 一类致命异常，
    # pet_crash.log 里会有崩溃时的 Python 栈（否则进程被系统杀掉什么都不留）。
    try:
        _crash_fh = open(os.path.join(HERE, "pet_crash.log"), "a",
                         buffering=1, encoding="utf-8")
        faulthandler.enable(file=_crash_fh, all_threads=True)
        _crash_fh.write("\n===== 启动 %s =====\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="J022 Ghoul 桌宠")
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--test", type=float, default=0,
                    help="测试模式：N 秒后自动退出")
    args = ap.parse_args()

    app = PetApp(x=args.x, y=args.y, chat=not args.no_chat,
                 test_seconds=args.test or None)
    print("J022 Ghoul 桌宠已启动：左键单击对话 / 拖拽移动 / 右键菜单", flush=True)

    msg = MSG()
    while True:
        r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if r <= 0:
            break
        user32.TranslateMessage(ctypes.byref(msg))
        # 异常保护：单条消息处理出错不应让整个桌宠进程崩溃（崩溃看起来像卡死/消失）
        try:
            user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as e:  # noqa: BLE001
            print("dispatch error:", e, flush=True)
    app.cleanup()
    print("桌宠已退出", flush=True)


if __name__ == "__main__":
    main()
