# -*- coding: utf-8 -*-
"""glitch_fx.py — 桌宠周身赛博朋克装饰（霓虹描边 / HUD 角框 / 扫描线 /
环绕粒子 / 偶发"错误闪光"故障）。

所有效果都画在角色**之外或之上**的装饰层，并不修改原始的动画帧 PNG，
也不参与命中测试（桌宠的点击/抓取区域仍由干净的躯干决定，避免之前
"绿点误抓取"的同类问题）。

用法：
    fx = GlitchFX(w, h)
    decorated = fx.decorate(frame, state, anim_t, now)
    # frame: PIL RGBA 246x364；返回新的 RGBA 图像（不会改动入参）
"""
import math
import os
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyh.ttc")

# 霓虹色板（用户要求：把"蓝色赛博装饰"整体换成暗红 155,0,0）
NEON_DARKRED = (155, 0, 0)   # 暗红（替代原来的青色霓虹）
NEON_MAGENTA = (255, 40, 200)
NEON_GREEN = (80, 255, 130)
NEON_RED = (255, 55, 70)
NEON_YELLOW = (255, 220, 60)


class GlitchFX:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.rng = random.Random(20261025)
        self.enabled = True
        # 右键菜单打开时为 True：强制持续故障（跳动乱码）
        self.menu_mode = False
        # 故障（错误闪光）调度
        self.next_glitch = time.time() + self.rng.uniform(3.0, 9.0)
        self.glitch_until = 0.0
        self.glitch_seed = 0
        # 静态扫描线叠加层（整幅淡蓝细线，营造 CRT 质感）
        self._scan = self._build_scanlines()
        # 环绕漂移的霓虹粒子（加号 / 方块）
        self.particles = self._build_particles()
        self._font_small = self._load_font(12)
        self._font_tiny = self._load_font(9)
        self._t0 = time.time()

    # ---------------------------------------------------------- 资源
    @staticmethod
    def _load_font(size):
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            return ImageFont.load_default()

    def _build_scanlines(self):
        img = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for y in range(0, self.h, 3):
            d.line([(0, y), (self.w, y)], fill=(155, 0, 0, 18), width=1)
        return img

    def _build_particles(self):
        ps = []
        for _ in range(16):
            ps.append(dict(
                ang=self.rng.uniform(0, 2 * math.pi),
                rad=self.rng.uniform(0.55, 1.08),
                spd=self.rng.uniform(0.25, 0.8) * (1 if self.rng.random() < 0.5 else -1),
                size=self.rng.choice([1, 2, 3]),
                color=self.rng.choice([NEON_DARKRED, NEON_MAGENTA, NEON_GREEN, NEON_YELLOW]),
                phase=self.rng.uniform(0, 6.28),
            ))
        return ps

    # ---------------------------------------------------------- 入口
    def decorate(self, frame, state="idle", anim_t=0, now=None):
        if not self.enabled:
            return frame
        if now is None:
            now = time.time()
        img = frame.copy().convert("RGBA")
        arr = np.asarray(np.array(frame), dtype=np.float32)
        body = arr[..., 3] > 24
        ys, xs = np.where(body)
        if len(xs) == 0:
            return img
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0, y1 - y0

        # ① 霓虹描边（暗红/品红 RGB 错位）
        self._neon_outline(img, body)
        # ② HUD 角框 + 标签 + 活动刻度
        self._hud(img, x0, y0, x1, y1, now, anim_t)
        # ③ 扫描线
        img.alpha_composite(self._scan, (0, 0))
        # ④ 环绕漂移粒子
        self._particles(img, x0, y0, bw, bh, now)
        # ⑤ 偶发"错误闪光"故障（右键菜单打开时强制持续故障）
        self._maybe_glitch(now)
        forced = bool(getattr(self, "menu_mode", False))
        if now < self.glitch_until or forced:
            self._glitch(img, frame, x0, y0, x1, y1, now, forced=forced)

        return img

    # ---------------------------------------------------------- ① 霓虹描边
    def _ring_layer(self, ring, color, strength=1.0, blur=0.0):
        """把 ring(0..255) 掩膜染成 color 后返回 RGBA 层（可选模糊）。"""
        out = np.zeros((self.h, self.w, 4), np.uint8)
        m = ring > 4
        out[m, 0:3] = color
        out[..., 3] = (np.clip(ring, 0, 255) * strength).astype(np.uint8)
        layer = Image.fromarray(out, "RGBA")
        if blur > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        return layer

    def _neon_outline(self, img, body):
        alpha = Image.fromarray((body.astype(np.uint8)) * 255, "L")
        dil = alpha.filter(ImageFilter.MaxFilter(9))          # 外扩 ~4px
        da = np.asarray(dil, np.float32)
        ba = np.asarray(alpha, np.float32)
        outer = np.clip(da - ba, 0, 255)                       # 轮廓外环带
        if outer.max() <= 0:
            return
        # 暗红向左错位、品红向右错位 → 经典 RGB 分离霓虹边
        cyan = self._ring_layer(np.roll(outer, -2, axis=1), NEON_DARKRED, 1.0, 1.6)
        mag = self._ring_layer(np.roll(outer, 2, axis=1), NEON_MAGENTA, 0.55, 1.6)
        img.alpha_composite(mag)
        img.alpha_composite(cyan)

    # ---------------------------------------------------------- ② HUD 角框
    def _hud(self, img, x0, y0, x1, y1, now, anim_t):
        pad = 11
        ax0, ay0, ax1, ay1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        ax0, ay0 = max(2, ax0), max(2, ay0)
        ax1, ay1 = min(self.w - 3, ax1), min(self.h - 3, ay1)
        L = 15
        col = NEON_DARKRED + (165,)
        d = ImageDraw.Draw(img, "RGBA")
        # 四角 L 形括号
        corners = [
            (ax0, ay0, +1, +1), (ax1, ay0, -1, +1),
            (ax0, ay1, +1, -1), (ax1, ay1, -1, -1),
        ]
        for cx, cy, sx, sy in corners:
            d.line([(cx + sx * L, cy), (cx, cy), (cx, cy + sy * L)],
                   fill=col, width=2)
        # 左侧活动刻度条（随动画帧上下跳动）
        bar_x = ax0 - 5
        if bar_x >= 1:
            seg = (ay1 - ay0) // 6
            for i in range(6):
                yy = ay0 + i * seg
                on = ((anim_t // 3 + i) % 6) < 3
                c = NEON_DARKRED + (200 if on else 50,)
                d.line([(bar_x, yy), (bar_x, yy + seg - 2)], fill=c, width=2)
        # 顶部标签 + 闪烁指示三角
        label = "J022//GHOUL"
        ty = max(1, ay0 - 15)
        d.text((ax0, ty), label, font=self._font_tiny, fill=(155, 0, 0, 150))
        blink = (int(now * 4) % 2) == 0
        if blink:
            tri = ax0 + len(label) * 6 + 6
            d.polygon([(tri, ty + 2), (tri + 8, ty + 2), (tri + 4, ty + 9)],
                      fill=NEON_DARKRED + (200,))

    # ---------------------------------------------------------- ④ 环绕粒子
    def _particles(self, img, x0, y0, bw, bh, now):
        cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
        rx, ry = max(8, bw * 0.62), max(8, bh * 0.62)
        t = now - self._t0
        d = ImageDraw.Draw(img, "RGBA")
        for p in self.particles:
            a = p["ang"] + t * p["spd"]
            px = cx + math.cos(a) * rx * p["rad"]
            py = cy + math.sin(a) * ry * p["rad"]
            if not (0 <= px < self.w and 0 <= py < self.h):
                continue
            if math.sin(t * 3.0 + p["phase"]) < 0.25:       # 闪烁
                continue
            s = p["size"]
            col = p["color"] + (200,)
            d.line([(px - s, py), (px + s, py)], fill=col, width=1)
            d.line([(px, py - s), (px, py + s)], fill=col, width=1)

    # ---------------------------------------------------------- ⑤ 错误闪光
    def _maybe_glitch(self, now):
        if now >= self.glitch_until and now >= self.next_glitch:
            self.glitch_until = now + self.rng.uniform(0.35, 0.85)
            self.glitch_seed = self.rng.randint(0, 1 << 30)
            self.next_glitch = self.glitch_until + self.rng.uniform(4.0, 11.0)

    def _tag(self, d, text, x, y, font, fg, bg=(8, 10, 14)):
        """带深色底衬的文字标签，确保在泛红故障背景上清晰可读。"""
        tw = d.textlength(text, font=font)
        d.rectangle([x - 2, y - 1, x + tw + 2, y + 13], fill=bg + (165,))
        d.text((x, y), text, font=font, fill=fg)

    def _glitch(self, img, frame, x0, y0, x1, y1, now, forced=False):
        W, H = self.w, self.h
        if forced:
            # 菜单打开期间持续故障：种子随墙钟快速变化 → 乱码"跳动"
            self.glitch_until = now + 0.25
            base = (int(now * 18) ^ self.glitch_seed) & 0x7FFFFFFF
        else:
            base = self.glitch_seed + (int((now - self.glitch_until) * 1000) // 60)
        rng = random.Random(base)
        # 横向撕裂（数据损坏质感）
        n = rng.randint(5, 9) if forced else rng.randint(3, 7)
        for _ in range(n):
            yy = rng.randint(max(0, y0 - 6), min(H - 2, y1 + 6))
            hh = rng.randint(3, 16)
            dx = rng.randint(-16, 16)
            band = frame.crop((0, yy, W, min(H, yy + hh)))
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ov.alpha_composite(band, (dx, 0))
            img.alpha_composite(ov)
        # 红色故障脉冲（错误闪光核心，必然出现，强度随时间起伏）
        prog = max(0.0, min(1.0, (self.glitch_until - now) / 0.85))
        pulse = int(28 + 60 * abs(math.sin(prog * math.pi)))
        img.alpha_composite(
            Image.new("RGBA", (W, H), (255, 35, 45, pulse)), (0, 0))
        # 红色边框闪光（错误标志，必然出现）
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([1, 1, W - 2, H - 2], outline=NEON_RED + (210,), width=2)
        # 故障文字（ERR / 致命错误，必然出现，带深色底衬醒目可读）
        texts = ["ERR", "⚠ ERR", "SYS.CORRUPT", "0xDEAD", "FATAL!!", "GLITCH"]
        txt = rng.choice(texts)
        tx = max(3, min(W - 70, x0 + rng.randint(-8, 8)))
        ty = max(3, y0 - 22 + rng.randint(-3, 3))
        self._tag(d, txt, tx, ty, self._font_small, NEON_RED + (255,))
        # 十六进制错误码（红 + 浅红各一行，必然出现，置于画布内安全位置）
        ay1 = min(H - 3, y1 + 11)
        hy = min(H - 13, ay1 + 2)
        hexs = "".join(rng.choice("0123456789ABCDEF") for _ in range(8))
        self._tag(d, "0x" + hexs, x0, hy, self._font_tiny, (255, 90, 100, 255))
        if hy - 13 >= 2:
            hx2 = "".join(rng.choice("0123456789ABCDEF") for _ in range(8))
            self._tag(d, "0x" + hx2, x0 + 4, hy - 13, self._font_tiny,
                      (255, 120, 130, 255))
        # 跳动的乱码（菜单态更强、更密，平时偶发）
        self._garble(d, x0, y0, x1, y1, rng, forced=forced)

    def _garble(self, d, x0, y0, x1, y1, rng, forced=False):
        """随机乱码串（十六进制 + 符号），每帧重新生成 → 视觉上"跳动"。"""
        chars = "0123456789ABCDEF#@%&<>/\\=+*$"
        n = rng.randint(4, 7) if forced else rng.randint(2, 4)
        for _ in range(n):
            s = "".join(rng.choice(chars) for _ in range(rng.randint(3, 9)))
            gx = max(2, min(self.w - 70, rng.randint(x0, max(x0, x1 - 6))))
            gy = max(2, min(self.h - 13, rng.randint(y0, max(y0, y1 - 10))))
            col = rng.choice([(255, 90, 100, 255), (255, 200, 210, 235),
                              (255, 140, 150, 255)])
            self._tag(d, s, gx, gy, self._font_tiny, col)
