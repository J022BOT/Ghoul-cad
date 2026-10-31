"""concepts.py — 现实物体的「概念库」：词汇 → 参数化、可描摹的形状。

为什么需要它
------------
本地 7B 模型对「葫芦」「雨伞」「台灯」这类**具体物体**没有稳定的形状记忆，
让它凭想象报坐标，经常得到一堆互不相干的折线（"画一个葫芦"就会错乱）。
本模块把这些物体预先拆成「部件 + 轮廓控制点」的确定性配方：

- 命中词汇（含别名、英文）→ 服务器直接生成几何，毫秒级、零误差、可调尺寸；
- 未命中 → 返回 (None, "")，调用方回退大模型（并给模型一份"部件拆解"提示）。

尺寸说法：高/长 120、宽 80、直径 60、半径 30、大小 100，单位可写 mm / cm。
数量说法："三个苹果"（最多 6 个，横排）。
坐标系：毫米(mm)，原点在物体包围盒中心，X 向右，Y 向上。
"""

import math
import re

try:                                    # 复用 shapes.py 的中文数字解析
    from shapes import _num as _num_any
except Exception:                       # pragma: no cover
    def _num_any(s):
        try:
            return float(s)
        except Exception:
            return None


# ============================================================ 基础几何工具
def _pt(x, y):
    return [round(float(x), 3), round(float(y), 3)]


def _poly(pts, closed=True):
    pts = [[round(float(x), 3), round(float(y), 3)] for x, y in pts]
    ent = {"type": "polyline", "points": pts, "closed": bool(closed), "layer": "0"}
    # 配方里的"多点是采样出来的光滑轮廓"（_revolve/_smooth/_ellipse/_arcpts），
    # 交给 curves.curveify 升级成真 B 样条；只有几个点的折线是刻意画的折角
    # （车顶、机架、屋顶），保持多段线。
    if len(pts) >= 10:
        ent["smooth"] = True
    return ent


def _line(p1, p2):
    return {"type": "line", "p1": _pt(*p1), "p2": _pt(*p2), "layer": "0"}


def _circle(cx, cy, r):
    return {"type": "circle", "center": _pt(cx, cy), "radius": round(float(r), 3), "layer": "0"}


def _rect(x, y, w, h):
    return {"type": "rect", "x": round(float(x), 3), "y": round(float(y), 3),
            "w": round(float(w), 3), "h": round(float(h), 3), "layer": "0"}


def _cr(p0, p1, p2, p3, t):
    """Catmull-Rom 样条在 t∈[0,1] 处的点。"""
    t2, t3 = t * t, t * t * t

    def f(a, b, c, d):
        return 0.5 * ((2 * b) + (-a + c) * t
                      + (2 * a - 5 * b + 4 * c - d) * t2
                      + (-a + 3 * b - 3 * c + d) * t3)
    return (f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1]))


def _dedupe(pts, eps=1e-6):
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    return out


def _smooth(pts, closed=False, samples=6):
    """过控制点的平滑曲线（Catmull-Rom）；closed=True 首尾相连成环。"""
    P = [(float(x), float(y)) for x, y in pts]
    n = len(P)
    if n < 3:
        return P
    out = []
    if closed:
        ext = [P[-1]] + P + [P[0], P[1]]
        for i in range(1, n + 1):
            p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
            for k in range(samples):
                out.append(_cr(p0, p1, p2, p3, k / float(samples)))
        out.append(out[0])
    else:
        ext = [P[0]] + P + [P[-1]]
        out = [P[0]]
        for i in range(1, len(ext) - 2):
            p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
            for k in range(1, samples + 1):
                out.append(_cr(p0, p1, p2, p3, k / float(samples)))
    return _dedupe(out)


def _quad(p0, c, p1, n=14):
    """二次贝塞尔采样。"""
    out = []
    for i in range(n + 1):
        t = i / float(n)
        m = 1 - t
        out.append((m * m * p0[0] + 2 * m * t * c[0] + t * t * p1[0],
                    m * m * p0[1] + 2 * m * t * c[1] + t * t * p1[1]))
    return out


def _arcpts(cx, cy, r, a0, a1, n=24):
    """圆弧采样（角度制，逆时针为正）。"""
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / float(n))
        out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def _ellipse(cx, cy, rx, ry, n=36):
    return [(cx + rx * math.cos(2 * math.pi * i / n),
             cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _rot(pts, deg, cx=0.0, cy=0.0):
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa,
             cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts]


def _mirror_x(pts):
    return [(-x, y) for x, y in pts]


def _revolve(profile, w, h, samples=5):
    """把「(y 比例, 半径比例) 列表」转成左右对称的闭合轮廓。

    profile 自下而上；半径为**相对半宽的比例**，函数内部会重新归一到 w/2，
    保证画出来的最大宽度正好等于 w（这样"宽 80"就是真的 80mm 宽）。
    """
    half = w / 2.0
    ctrl = [(r, yf * h) for yf, r in profile]
    right = _smooth(ctrl, closed=False, samples=samples)
    mx = max([abs(x) for x, _ in right] or [1.0]) or 1.0
    k = half / mx
    right = [(x * k, y) for x, y in right]
    left = [(-x, y) for x, y in reversed(right)]
    merged = _dedupe(right + left[1:])
    if len(merged) > 2 and abs(merged[0][0] - merged[-1][0]) < 1e-6 \
            and abs(merged[0][1] - merged[-1][1]) < 1e-6:
        merged.pop()
    return merged


def _crescent(R, r, d, n=40):
    """月牙/香蕉形：大圆 R 减去偏置 d 的小圆 r 后剩下的外轮廓。"""
    x = (d * d + R * R - r * r) / (2 * d)
    y = math.sqrt(max(0.0, R * R - x * x))
    t1 = math.atan2(y, x)                       # 上交点
    t2 = 2 * math.pi - t1                       # 下交点
    pts = [(R * math.cos(t1 + (t2 - t1) * i / n),
            R * math.sin(t1 + (t2 - t1) * i / n)) for i in range(n + 1)]
    fA = math.atan2(y, x - d)                   # 小圆上的上交点角
    fB = -fA
    start = fB
    end = fA - 2 * math.pi                      # 经过 180° 的那一侧（反向）
    for i in range(1, n):
        a = start + (end - start) * i / n
        pts.append((d + r * math.cos(a), r * math.sin(a)))
    return pts


def _bbox(entities):
    xs, ys = [], []
    for e in entities:
        t = e["type"]
        if t == "circle":
            c, r = e["center"], e["radius"]
            xs += [c[0] - r, c[0] + r]
            ys += [c[1] - r, c[1] + r]
        elif t == "rect":
            xs += [e["x"], e["x"] + e["w"]]
            ys += [e["y"], e["y"] + e["h"]]
        elif t == "line":
            xs += [e["p1"][0], e["p2"][0]]
            ys += [e["p1"][1], e["p2"][1]]
        elif t == "polyline":
            xs += [p[0] for p in e["points"]]
            ys += [p[1] for p in e["points"]]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)


def _shift(entities, dx, dy):
    out = []
    for e in entities:
        e = dict(e)
        t = e["type"]
        if t == "circle":
            e["center"] = _pt(e["center"][0] + dx, e["center"][1] + dy)
        elif t == "rect":
            e["x"], e["y"] = round(e["x"] + dx, 3), round(e["y"] + dy, 3)
        elif t == "line":
            e["p1"] = _pt(e["p1"][0] + dx, e["p1"][1] + dy)
            e["p2"] = _pt(e["p2"][0] + dx, e["p2"][1] + dy)
        elif t == "polyline":
            e["points"] = [_pt(p[0] + dx, p[1] + dy) for p in e["points"]]
        out.append(e)
    return out


def _center(entities):
    x0, y0, x1, y1 = _bbox(entities)
    return _shift(entities, -(x0 + x1) / 2.0, -(y0 + y1) / 2.0)


def _scale(entities, s):
    """等比缩放（圆仍是圆，不会被压成椭圆）。"""
    out = []
    for e in entities:
        e = dict(e)
        t = e["type"]
        if t == "circle":
            e["center"] = _pt(e["center"][0] * s, e["center"][1] * s)
            e["radius"] = round(e["radius"] * s, 3)
        elif t == "rect":
            e["x"], e["y"] = round(e["x"] * s, 3), round(e["y"] * s, 3)
            e["w"], e["h"] = round(e["w"] * s, 3), round(e["h"] * s, 3)
        elif t == "line":
            e["p1"] = _pt(e["p1"][0] * s, e["p1"][1] * s)
            e["p2"] = _pt(e["p2"][0] * s, e["p2"][1] * s)
        elif t == "polyline":
            e["points"] = [_pt(p[0] * s, p[1] * s) for p in e["points"]]
        out.append(e)
    return out


def _fit(entities, w, h):
    """等比缩放到恰好填满 w×h 的一边并居中（保证圆不被压扁）。"""
    x0, y0, x1, y1 = _bbox(entities)
    bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    s = min(w / bw, h / bh)
    return _center(_scale(entities, s))


# ============================================================ 各物体配方
# 约定：recipe(w, h) → [实体...]，包围盒大致填满 w×h，中心在原点。

def _gourd(w, h):
    """葫芦：下大上小两个鼓包 + 收腰 + 顶蒂。"""
    prof = [(-0.50, 0.03), (-0.45, 0.40), (-0.38, 0.70), (-0.28, 0.93),
            (-0.18, 0.99), (-0.09, 0.82), (-0.03, 0.58), (0.02, 0.47),
            (0.07, 0.54), (0.13, 0.67), (0.19, 0.71), (0.25, 0.64),
            (0.31, 0.49), (0.37, 0.29), (0.43, 0.12), (0.47, 0.03)]
    ents = [_poly(_revolve(prof, w, h))]
    ents.append(_poly([(-0.045 * w, 0.46 * h), (-0.030 * w, 0.55 * h),
                       (0.030 * w, 0.55 * h), (0.045 * w, 0.46 * h)], closed=True))
    return ents


def _mug(w, h):
    """马克杯：杯身 + 杯口 + 右侧把手。"""
    prof = [(-0.50, 0.36), (-0.30, 0.39), (0.0, 0.43), (0.30, 0.46), (0.47, 0.47)]
    ents = [_poly(_revolve(prof, w, h))]
    ents.append(_line((-0.47 * w, 0.47 * h), (0.47 * w, 0.47 * h)))
    ents.append(_poly(_arcpts(0.44 * w, 0.02 * h, 0.22 * h, -80, 80, 26), closed=False))
    return ents


def _bottle(w, h):
    """瓶子：瓶身 + 肩部收口 + 瓶颈 + 瓶盖。"""
    prof = [(-0.50, 0.34), (-0.42, 0.37), (-0.12, 0.38), (0.02, 0.37),
            (0.10, 0.27), (0.19, 0.14), (0.33, 0.11), (0.44, 0.11), (0.50, 0.14)]
    ents = [_poly(_revolve(prof, w, h))]
    ents.append(_line((-0.13 * w, 0.43 * h), (0.13 * w, 0.43 * h)))
    return ents


def _bowl(w, h):
    """碗：外壁圆弧 + 平底 + 口沿。"""
    prof = [(-0.50, 0.17), (-0.45, 0.26), (-0.30, 0.36), (-0.10, 0.45),
            (0.20, 0.49), (0.50, 0.50)]
    ents = [_poly(_revolve(prof, w, h))]
    ents.append(_line((-0.40 * w, -0.50 * h), (0.40 * w, -0.50 * h)))
    return ents


def _plate(w, h):
    """盘子：浅口碟形。"""
    prof = [(-0.50, 0.10), (-0.42, 0.20), (-0.24, 0.32), (-0.04, 0.43),
            (0.26, 0.48), (0.50, 0.50)]
    ents = [_poly(_revolve(prof, w, h))]
    ents.append(_line((-0.36 * w, -0.50 * h), (0.36 * w, -0.50 * h)))
    return ents


def _spoon(w, h):
    """勺子：椭圆勺头 + 细长柄。"""
    ents = [_poly(_ellipse(0, -0.30 * h, 0.24 * w, 0.18 * h, 36))]
    ents.append(_poly([(-0.05 * w, -0.13 * h), (-0.035 * w, 0.48 * h),
                       (0.035 * w, 0.48 * h), (0.05 * w, -0.13 * h)], closed=True))
    return ents


def _umbrella(w, h):
    """雨伞：穹形伞面（下缘扇贝）+ 伞杆 + 弯手柄 + 伞骨。"""
    n = 24
    dome = []
    for i in range(n + 1):
        t = i / float(n)
        dome.append((-0.5 * w + w * t, 0.22 * h + 0.28 * h * (1 - (2 * t - 1) ** 2)))
    scallop = []
    for k in range(3):
        x0 = 0.5 * w - w * k / 3.0
        x1 = x0 - w / 3.0
        for i in range(1, 7):
            t = i / 6.0
            scallop.append((x0 + (x1 - x0) * t,
                            0.22 * h - 0.075 * h * math.sin(math.pi * t)))
    ents = [_poly(dome + scallop, closed=True)]
    ents.append(_line((0, 0.50 * h), (0, -0.36 * h)))
    ents.append(_poly(_arcpts(0.055 * w, -0.36 * h, 0.055 * w, 180, 360, 16), closed=False))
    for xv in (-w / 6.0, w / 6.0):
        ents.append(_line((0, 0.50 * h), (xv, 0.20 * h)))
    return ents


def _key(w, h):
    """钥匙：环 + 杆 + 两个齿。"""
    ents = [_poly(_ellipse(-0.30 * w, 0, 0.17 * h, 0.17 * h, 32))]
    ents.append(_poly([(-0.14 * w, -0.055 * h), (0.48 * w, -0.055 * h),
                       (0.48 * w, 0.055 * h), (-0.14 * w, 0.055 * h)], closed=True))
    ents.append(_poly([(0.28 * w, -0.055 * h), (0.28 * w, -0.20 * h),
                       (0.35 * w, -0.20 * h), (0.35 * w, -0.055 * h)], closed=True))
    ents.append(_poly([(0.39 * w, -0.055 * h), (0.39 * w, -0.15 * h),
                       (0.45 * w, -0.15 * h), (0.45 * w, -0.055 * h)], closed=True))
    return ents


def _bulb(w, h):
    """灯泡：圆形玻璃泡 + 收颈 + 螺纹灯头。"""
    r = 0.5 * w
    cy = 0.16 * h
    ents = [_poly(_ellipse(0, cy, r, r * 1.02, 44))]
    ents.append(_poly([(-0.18 * w, cy - r * 0.82), (-0.13 * w, -0.34 * h),
                       (0.13 * w, -0.34 * h), (0.18 * w, cy - r * 0.82)], closed=False))
    ents.append(_rect(-0.14 * w, -0.50 * h, 0.28 * w, 0.16 * h))
    ents.append(_line((-0.14 * w, -0.44 * h), (0.14 * w, -0.44 * h)))
    ents.append(_line((-0.14 * w, -0.38 * h), (0.14 * w, -0.38 * h)))
    return ents


def _pencil(w, h):
    """铅笔：笔身 + 笔尖 + 尾部橡皮（宽度即笔杆粗细）。"""
    ents = [_poly([(-0.5 * w, -0.28 * h), (-0.5 * w, 0.40 * h),
                   (0.5 * w, 0.40 * h), (0.5 * w, -0.28 * h)], closed=True)]
    ents.append(_poly([(-0.5 * w, -0.28 * h), (0, -0.50 * h),
                       (0.5 * w, -0.28 * h)], closed=True))
    ents.append(_rect(-0.5 * w, 0.40 * h, w, 0.10 * h))
    ents.append(_line((-0.5 * w, 0.34 * h), (0.5 * w, 0.34 * h)))
    return ents


def _book(w, h):
    """书：封面 + 书脊 + 页线。"""
    ents = [_rect(-0.45 * w, -0.50 * h, 0.90 * w, 1.00 * h)]
    ents.append(_line((-0.34 * w, -0.50 * h), (-0.34 * w, 0.50 * h)))
    for y in (-0.20, 0.0, 0.20):
        ents.append(_line((-0.27 * w, y * h), (0.34 * w, y * h)))
    return ents


def _clock(w, h):
    """闹钟：表盘 + 刻度 + 指针 + 双铃 + 脚。"""
    r = 0.38 * min(w, h)
    cy = -0.03 * h
    ents = [_circle(0, cy, r)]
    for k in range(12):
        a = math.radians(90 - 30 * k)
        ents.append(_line((r * 0.84 * math.cos(a), cy + r * 0.84 * math.sin(a)),
                          (r * 0.95 * math.cos(a), cy + r * 0.95 * math.sin(a))))
    ents.append(_line((0, cy), (0, cy + r * 0.58)))
    ents.append(_line((0, cy), (r * 0.46 * math.cos(math.radians(-35)),
                                cy + r * 0.46 * math.sin(math.radians(-35)))))
    for sgn in (-1, 1):
        ents.append(_poly(_arcpts(sgn * r * 0.70, cy + r * 0.78, r * 0.25, 0, 180, 14),
                          closed=False))
        ents.append(_line((sgn * r * 0.52, cy - r * 0.86), (sgn * r * 0.78, cy - r * 1.04)))
    return ents


def _house(w, h):
    """房子：墙 + 三角屋顶 + 门 + 两扇带十字窗。"""
    ents = [_rect(-0.5 * w, -0.5 * h, 1.0 * w, 0.60 * h)]
    ents.append(_poly([(-0.57 * w, 0.10 * h), (0.57 * w, 0.10 * h), (0, 0.50 * h)],
                      closed=True))
    ents.append(_rect(-0.11 * w, -0.50 * h, 0.22 * w, 0.34 * h))
    ents.append(_circle(0.07 * w, -0.34 * h, 0.02 * w))
    for sx in (-0.36 * w, 0.18 * w):
        ents.append(_rect(sx, -0.06 * h, 0.18 * w, 0.15 * h))
        ents.append(_line((sx + 0.09 * w, -0.06 * h), (sx + 0.09 * w, 0.09 * h)))
        ents.append(_line((sx, 0.015 * h), (sx + 0.18 * w, 0.015 * h)))
    return ents


def _door(w, h):
    """门：门框 + 上下门板 + 门把手。"""
    ents = [_rect(-0.45 * w, -0.50 * h, 0.90 * w, 1.00 * h)]
    ents.append(_rect(-0.32 * w, 0.04 * h, 0.64 * w, 0.32 * h))
    ents.append(_rect(-0.32 * w, -0.42 * h, 0.64 * w, 0.38 * h))
    ents.append(_circle(0.32 * w, 0.02 * h, 0.03 * w))
    return ents


def _window(w, h):
    """窗：外框 + 内框 + 十字窗棂。"""
    ents = [_rect(-0.5 * w, -0.5 * h, 1.0 * w, 1.0 * h)]
    ents.append(_rect(-0.44 * w, -0.44 * h, 0.88 * w, 0.88 * h))
    ents.append(_line((0, -0.44 * h), (0, 0.44 * h)))
    ents.append(_line((-0.44 * w, 0), (0.44 * w, 0)))
    return ents


def _lamp(w, h):
    """台灯：底座 + 灯杆 + 梯形灯罩。"""
    ents = [_poly(_ellipse(0, -0.45 * h, 0.34 * w, 0.06 * h, 32))]
    ents.append(_line((0, -0.45 * h), (0, 0.17 * h)))
    ents.append(_poly([(-0.24 * w, 0.50 * h), (0.24 * w, 0.50 * h),
                       (0.37 * w, 0.17 * h), (-0.37 * w, 0.17 * h)], closed=True))
    return ents


def _chair(w, h):
    """椅子：侧视——座面 + 宽靠背 + 两条腿 + 横撑。"""
    ents = [_rect(-0.36 * w, -0.05 * h, 0.72 * w, 0.10 * h)]
    ents.append(_rect(-0.40 * w, 0.02 * h, 0.20 * w, 0.46 * h))
    ents.append(_line((-0.30 * w, -0.05 * h), (-0.30 * w, -0.50 * h)))
    ents.append(_line((0.30 * w, -0.05 * h), (0.30 * w, -0.50 * h)))
    ents.append(_line((-0.30 * w, -0.26 * h), (0.30 * w, -0.26 * h)))
    return ents


def _table(w, h):
    """桌子：桌板 + 两条腿 + 横撑。"""
    ents = [_rect(-0.5 * w, 0.34 * h, 1.0 * w, 0.10 * h)]
    ents.append(_rect(-0.42 * w, -0.50 * h, 0.09 * w, 0.84 * h))
    ents.append(_rect(0.33 * w, -0.50 * h, 0.09 * w, 0.84 * h))
    ents.append(_line((-0.42 * w, -0.30 * h), (0.42 * w, -0.30 * h)))
    return ents


def _tree(w, h):
    """树：树干 + 分瓣的团状树冠。"""
    ents = [_rect(-0.07 * w, -0.50 * h, 0.14 * w, 0.42 * h)]
    crown = [(-0.50 * w, 0.16 * h), (-0.38 * w, 0.36 * h), (-0.20 * w, 0.30 * h),
             (-0.10 * w, 0.48 * h), (0.06 * w, 0.40 * h), (0.20 * w, 0.46 * h),
             (0.34 * w, 0.30 * h), (0.46 * w, 0.26 * h), (0.50 * w, 0.08 * h),
             (0.30 * w, -0.02 * h), (0.00, -0.04 * h), (-0.28 * w, -0.02 * h)]
    ents.append(_poly(_smooth(crown, closed=True, samples=5)))
    return ents


def _cloud(w, h):
    """云：几个明显鼓包相连，底部略平。"""
    pts = [(-0.50 * w, -0.08 * h), (-0.44 * w, 0.12 * h), (-0.30 * w, 0.14 * h),
           (-0.26 * w, 0.32 * h), (-0.12 * w, 0.42 * h), (0.00, 0.28 * h),
           (0.12 * w, 0.44 * h), (0.26 * w, 0.38 * h), (0.32 * w, 0.22 * h),
           (0.44 * w, 0.18 * h), (0.48 * w, 0.02 * h), (0.50 * w, -0.08 * h),
           (0.28 * w, -0.18 * h), (0.00, -0.20 * h), (-0.28 * w, -0.18 * h)]
    return [_poly(_smooth(pts, closed=True, samples=5))]


def _sun(w, h):
    """太阳：圆 + 12 道光芒。"""
    R = 0.26 * min(w, h)
    ents = [_circle(0, 0, R)]
    for k in range(12):
        a = math.radians(15 + 30 * k)
        ents.append(_line((R * 1.22 * math.cos(a), R * 1.22 * math.sin(a)),
                          (R * 1.55 * math.cos(a), R * 1.55 * math.sin(a))))
    return ents


def _moon(w, h):
    """月亮：大圆减偏置小圆得到的月牙。"""
    R = 0.5 * min(w, h) * 0.96
    pts = _crescent(R, R * 0.86, R * 0.58, 44)
    return [_poly(pts, closed=True)]


def _mountain(w, h):
    """山：双峰折线 + 雪线。"""
    ents = [_poly([(-0.50 * w, -0.50 * h), (-0.18 * w, 0.28 * h), (0.02 * w, 0.02 * h),
                   (0.22 * w, 0.50 * h), (0.50 * w, -0.50 * h)], closed=True)]
    ents.append(_poly([(-0.32 * w, 0.12 * h), (-0.20 * w, 0.19 * h),
                       (-0.08 * w, 0.11 * h), (0.03 * w, 0.17 * h),
                       (0.13 * w, 0.30 * h)], closed=False))
    return ents


def _flower(w, h):
    """花：6 片花瓣 + 花心 + 花茎 + 叶。"""
    b = min(w, h)
    cy = 0.16 * h
    ents = [_circle(0, cy, 0.10 * b)]
    for k in range(6):
        a = 60 * k
        px = 0.26 * b * math.cos(math.radians(a))
        py = cy + 0.26 * b * math.sin(math.radians(a))
        pet = _ellipse(px, py, 0.17 * b, 0.10 * b, 20)
        ents.append(_poly(_rot(pet, a, px, py)))
    ents.append(_line((0, cy - 0.10 * b), (0, -0.50 * h)))
    leaf = _quad((0, -0.24 * h), (0.20 * w, -0.30 * h), (0.04 * w, -0.10 * h))
    leaf2 = _quad((0.04 * w, -0.10 * h), (0.20 * w, -0.16 * h), (0, -0.24 * h))
    ents.append(_poly(leaf + leaf2[1:], closed=True))
    return ents


def _leaf(w, h):
    """叶子：两条对称弧线合成叶形 + 主脉 + 侧脉。"""
    left = _quad((0, -0.50 * h), (-1.02 * w, 0.0), (0, 0.50 * h), 26)
    right = _quad((0, 0.50 * h), (1.02 * w, 0.0), (0, -0.50 * h), 26)
    ents = [_poly(left + right[1:], closed=True)]
    ents.append(_line((0, -0.46 * h), (0, 0.46 * h)))
    for yv in (-0.24, 0.0, 0.24):
        ents.append(_line((0, yv * h), (0.30 * w, (yv + 0.13) * h)))
        ents.append(_line((0, yv * h), (-0.30 * w, (yv + 0.13) * h)))
    return ents


def _snowflake(w, h):
    """雪花：6 主枝 + 每枝两组分叉。"""
    R = 0.5 * min(w, h)
    ents = []
    for k in range(6):
        a = math.radians(60 * k)
        ents.append(_line((0, 0), (R * math.cos(a), R * math.sin(a))))
        for frac in (0.45, 0.70):
            bx, by = R * frac * math.cos(a), R * frac * math.sin(a)
            for sgn in (1, -1):
                b = a + sgn * math.radians(38)
                L = R * 0.24
                ents.append(_line((bx, by), (bx + L * math.cos(b), by + L * math.sin(b))))
    return ents


def _cat(w, h):
    """猫（正面）：头 + 耳朵 + 眼 + 鼻嘴 + 胡须。"""
    ents = [_poly(_ellipse(0, -0.06 * h, 0.42 * w, 0.38 * h, 44))]
    ents.append(_poly([(-0.36 * w, 0.10 * h), (-0.24 * w, 0.44 * h),
                       (-0.04 * w, 0.24 * h)], closed=True))
    ents.append(_poly([(0.36 * w, 0.10 * h), (0.24 * w, 0.44 * h),
                       (0.04 * w, 0.24 * h)], closed=True))
    for sx in (-0.16 * w, 0.16 * w):
        ents.append(_poly(_ellipse(sx, -0.02 * h, 0.055 * w, 0.075 * h, 18)))
    ents.append(_poly([(-0.035 * w, -0.16 * h), (0.035 * w, -0.16 * h),
                       (0, -0.21 * h)], closed=True))
    ents.append(_poly([(0, -0.21 * h), (-0.035 * w, -0.27 * h)], closed=False))
    ents.append(_poly([(0, -0.21 * h), (0.035 * w, -0.27 * h)], closed=False))
    for sgn in (-1, 1):
        for dy in (0.0, -0.05 * h):
            ents.append(_line((sgn * 0.22 * w, -0.17 * h + dy),
                              (sgn * 0.47 * w, -0.12 * h + dy)))
    return ents


def _dog(w, h):
    """狗（正面）：头 + 垂耳 + 口鼻 + 鼻头 + 眼。"""
    ents = [_poly(_ellipse(0, 0.02 * h, 0.36 * w, 0.34 * h, 44))]
    for sgn in (-1, 1):
        ear = [(sgn * 0.28 * w, 0.26 * h), (sgn * 0.48 * w, 0.10 * h),
               (sgn * 0.46 * w, -0.20 * h), (sgn * 0.32 * w, -0.02 * h)]
        ents.append(_poly(_smooth(ear, closed=True, samples=5)))
    ents.append(_poly(_ellipse(0, -0.20 * h, 0.21 * w, 0.15 * h, 28)))
    ents.append(_poly([(-0.055 * w, -0.13 * h), (0.055 * w, -0.13 * h),
                       (0, -0.20 * h)], closed=True))
    for sx in (-0.13 * w, 0.13 * w):
        ents.append(_poly(_ellipse(sx, 0.10 * h, 0.045 * w, 0.055 * h, 16)))
    return ents


def _fish(w, h):
    """鱼：纺锤身体 + 尾鳍 + 背鳍 + 眼 + 鳃线。"""
    body = [(-0.40 * w, 0.0), (-0.22 * w, 0.22 * h), (0.10 * w, 0.25 * h),
            (0.32 * w, 0.11 * h), (0.42 * w, 0.0), (0.32 * w, -0.11 * h),
            (0.10 * w, -0.25 * h), (-0.22 * w, -0.22 * h)]
    ents = [_poly(_smooth(body, closed=True, samples=5))]
    ents.append(_poly([(-0.38 * w, 0.0), (-0.56 * w, 0.20 * h),
                       (-0.50 * w, 0.0), (-0.56 * w, -0.20 * h)], closed=True))
    ents.append(_poly([(0.00, 0.21 * h), (0.08 * w, 0.35 * h), (0.18 * w, 0.19 * h)],
                      closed=True))
    ents.append(_circle(0.26 * w, 0.03 * h, 0.035 * w))
    ents.append(_poly(_arcpts(0.14 * w, 0, 0.16 * w, -55, 55, 12), closed=False))
    return ents


def _bird(w, h):
    """鸟：身体 + 头 + 喙 + 翅 + 尾 + 腿。"""
    ents = [_poly(_ellipse(0, -0.06 * h, 0.30 * w, 0.24 * h, 36))]
    ents.append(_circle(0.20 * w, 0.22 * h, 0.13 * w))
    ents.append(_poly([(0.31 * w, 0.26 * h), (0.50 * w, 0.19 * h),
                       (0.31 * w, 0.13 * h)], closed=True))
    wing = [(0.06 * w, 0.08 * h), (-0.06 * w, 0.20 * h),
            (-0.24 * w, 0.10 * h), (-0.10 * w, -0.06 * h)]
    ents.append(_poly(_smooth(wing, closed=True, samples=5)))
    ents.append(_poly([(-0.26 * w, 0.00 * h), (-0.50 * w, 0.16 * h),
                       (-0.44 * w, -0.10 * h)], closed=True))
    ents.append(_circle(0.24 * w, 0.26 * h, 0.022 * w))
    for sx in (-0.08 * w, 0.08 * w):
        ents.append(_line((sx, -0.28 * h), (sx, -0.42 * h)))
        ents.append(_line((sx - 0.04 * w, -0.44 * h), (sx + 0.04 * w, -0.44 * h)))
    return ents


def _butterfly(w, h):
    """蝴蝶：身体 + 上下四翅 + 触角。"""
    ents = [_poly(_ellipse(0, 0.0, 0.045 * w, 0.32 * h, 24))]
    for sgn in (-1, 1):
        up = [(sgn * 0.04 * w, 0.12 * h), (sgn * 0.24 * w, 0.42 * h),
              (sgn * 0.46 * w, 0.34 * h), (sgn * 0.42 * w, 0.06 * h),
              (sgn * 0.10 * w, 0.02 * h)]
        lo = [(sgn * 0.06 * w, -0.02 * h), (sgn * 0.34 * w, -0.16 * h),
              (sgn * 0.42 * w, -0.36 * h), (sgn * 0.18 * w, -0.30 * h)]
        ents.append(_poly(_smooth(up, closed=True, samples=5)))
        ents.append(_poly(_smooth(lo, closed=True, samples=5)))
        ents.append(_line((sgn * 0.03 * w, 0.30 * h), (sgn * 0.18 * w, 0.46 * h)))
        ents.append(_circle(sgn * 0.18 * w, 0.46 * h, 0.015 * w))
    return ents


def _rabbit(w, h):
    """兔子：身体 + 头 + 长耳 + 眼 + 脚 + 尾。"""
    ents = [_poly(_ellipse(0, -0.26 * h, 0.30 * w, 0.24 * h, 36))]
    ents.append(_poly(_ellipse(0, 0.14 * h, 0.21 * w, 0.18 * h, 36)))
    for sgn in (-1, 1):
        ear = [(sgn * 0.06 * w, 0.24 * h), (sgn * 0.10 * w, 0.48 * h),
               (sgn * 0.19 * w, 0.46 * h), (sgn * 0.15 * w, 0.22 * h)]
        ents.append(_poly(_smooth(ear, closed=True, samples=5)))
        ents.append(_poly(_ellipse(sgn * 0.09 * w, 0.10 * h, 0.03 * w, 0.045 * h, 14)))
        ents.append(_poly(_ellipse(sgn * 0.15 * w, -0.47 * h, 0.10 * w, 0.04 * h, 16)))
    ents.append(_circle(-0.29 * w, -0.34 * h, 0.06 * w))
    return ents


def _snail(w, h):
    """蜗牛：螺旋壳 + 长条身体 + 两根触角。"""
    b = min(w, h)
    cx, cy = 0.10 * w, 0.12 * h
    pts = []
    n, turns = 100, 2.8
    for i in range(n + 1):
        t = i / float(n)
        a = t * 2 * math.pi * turns
        r = (0.04 + 0.62 * t) * b * 0.5
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    ents = [_poly(pts, closed=False)]
    body = [(-0.50 * w, -0.40 * h), (-0.30 * w, -0.26 * h), (-0.05 * w, -0.22 * h),
            (0.10 * w, -0.28 * h), (0.22 * w, -0.34 * h), (0.18 * w, -0.46 * h),
            (-0.44 * w, -0.46 * h)]
    ents.append(_poly(_smooth(body, closed=True, samples=5)))
    ents.append(_line((-0.36 * w, -0.32 * h), (-0.50 * w, -0.10 * h)))
    ents.append(_line((-0.44 * w, -0.36 * h), (-0.58 * w, -0.18 * h)))
    ents.append(_circle(-0.50 * w, -0.10 * h, 0.025 * w))
    ents.append(_circle(-0.58 * w, -0.18 * h, 0.025 * w))
    return ents


def _penguin(w, h):
    """企鹅：蛋形身体 + 白肚 + 眼 + 喙 + 两只脚。"""
    body = [(0, 0.50 * h), (0.30 * w, 0.42 * h), (0.45 * w, 0.18 * h),
            (0.47 * w, -0.18 * h), (0.32 * w, -0.44 * h), (0, -0.50 * h),
            (-0.32 * w, -0.44 * h), (-0.47 * w, -0.18 * h),
            (-0.45 * w, 0.18 * h), (-0.30 * w, 0.42 * h)]
    ents = [_poly(_smooth(body, closed=True, samples=5))]
    belly = [(0, 0.30 * h), (0.26 * w, 0.22 * h), (0.28 * w, -0.12 * h),
             (0, -0.32 * h), (-0.28 * w, -0.12 * h), (-0.26 * w, 0.22 * h)]
    ents.append(_poly(_smooth(belly, closed=True, samples=5)))
    for sx in (-0.14 * w, 0.14 * w):
        ents.append(_poly(_ellipse(sx, 0.32 * h, 0.035 * w, 0.042 * h, 14)))
    ents.append(_poly([(-0.09 * w, 0.22 * h), (0.09 * w, 0.22 * h), (0, 0.12 * h)],
                      closed=True))
    for sx in (-0.20 * w, 0.20 * w):
        ents.append(_poly(_ellipse(sx, -0.48 * h, 0.13 * w, 0.035 * h, 18)))
    return ents


def _car(w, h):
    """汽车（侧视）：车身轮廓 + 车窗 + 两轮。"""
    body = [(-0.50 * w, -0.10 * h), (-0.46 * w, 0.10 * h), (-0.28 * w, 0.14 * h),
            (-0.16 * w, 0.30 * h), (0.12 * w, 0.30 * h), (0.26 * w, 0.14 * h),
            (0.46 * w, 0.10 * h), (0.50 * w, -0.10 * h)]
    ents = [_poly(_smooth(body, closed=True, samples=4))]
    ents.append(_poly([(-0.13 * w, 0.15 * h), (-0.07 * w, 0.26 * h),
                       (0.04 * w, 0.26 * h), (0.09 * w, 0.15 * h)], closed=True))
    ents.append(_poly([(-0.17 * w, 0.15 * h), (-0.22 * w, 0.24 * h),
                       (-0.26 * w, 0.15 * h)], closed=True))
    for sx in (-0.27 * w, 0.27 * w):
        ents.append(_circle(sx, -0.13 * h, 0.12 * h))
        ents.append(_circle(sx, -0.13 * h, 0.05 * h))
    return ents


def _boat(w, h):
    """帆船：船体 + 桅杆 + 两面帆。"""
    ents = [_poly([(-0.50 * w, 0.04 * h), (-0.34 * w, -0.28 * h),
                   (0.34 * w, -0.28 * h), (0.50 * w, 0.04 * h)], closed=True)]
    ents.append(_line((-0.50 * w, 0.04 * h), (0.50 * w, 0.04 * h)))
    ents.append(_line((0.01 * w, 0.04 * h), (0.01 * w, 0.50 * h)))
    ents.append(_poly([(0.04 * w, 0.46 * h), (0.36 * w, 0.10 * h),
                       (0.04 * w, 0.10 * h)], closed=True))
    ents.append(_poly([(-0.01 * w, 0.42 * h), (-0.26 * w, 0.10 * h),
                       (-0.01 * w, 0.10 * h)], closed=True))
    return ents


def _plane(w, h):
    """飞机（俯视）：机身 + 主翼 + 尾翼。"""
    ents = [_poly(_ellipse(0, 0, 0.10 * w, 0.46 * h, 36))]
    ents.append(_poly([(-0.48 * w, 0.02 * h), (-0.48 * w, -0.08 * h),
                       (-0.08 * w, 0.12 * h), (-0.08 * w, 0.22 * h)], closed=True))
    ents.append(_poly([(0.48 * w, 0.02 * h), (0.48 * w, -0.08 * h),
                       (-0.08 * w, 0.12 * h), (-0.08 * w, 0.22 * h)], closed=True))
    ents.append(_poly([(-0.24 * w, -0.34 * h), (-0.24 * w, -0.44 * h),
                       (-0.04 * w, -0.36 * h), (-0.04 * w, -0.28 * h)], closed=True))
    ents.append(_poly([(0.24 * w, -0.34 * h), (0.24 * w, -0.44 * h),
                       (-0.04 * w, -0.36 * h), (-0.04 * w, -0.28 * h)], closed=True))
    for k in range(4):
        ents.append(_circle(0, 0.30 * h - k * 0.10 * h, 0.018 * w))
    return ents


def _bicycle(w, h):
    """自行车：两轮 + 三角车架 + 车把 + 座。"""
    r = 0.20 * h
    ents = [_circle(-0.32 * w, -0.18 * h, r), _circle(0.32 * w, -0.18 * h, r),
            _circle(-0.32 * w, -0.18 * h, r * 0.14),
            _circle(0.32 * w, -0.18 * h, r * 0.14)]
    ents.append(_poly([(-0.32 * w, -0.18 * h), (0.02 * w, -0.18 * h),
                       (0.08 * w, 0.10 * h), (-0.14 * w, 0.10 * h),
                       (-0.32 * w, -0.18 * h)], closed=False))
    ents.append(_line((0.08 * w, 0.10 * h), (0.32 * w, -0.18 * h)))
    ents.append(_line((-0.14 * w, 0.10 * h), (0.02 * w, -0.18 * h)))
    ents.append(_line((0.08 * w, 0.10 * h), (0.14 * w, 0.18 * h)))
    ents.append(_line((0.06 * w, 0.18 * h), (0.20 * w, 0.18 * h)))
    ents.append(_line((-0.14 * w, 0.10 * h), (-0.24 * w, 0.16 * h)))
    ents.append(_line((-0.31 * w, 0.16 * h), (-0.17 * w, 0.16 * h)))
    return ents


def _traffic(w, h):
    """红绿灯：灯箱 + 三灯 + 灯杆 + 底座。"""
    r = 0.095 * min(w, h)
    ents = [_rect(-0.22 * w, 0.00, 0.44 * w, 0.50 * h)]
    for yy in (0.40 * h, 0.25 * h, 0.10 * h):
        ents.append(_circle(0, yy, r))
    ents.append(_rect(-0.05 * w, -0.50 * h, 0.10 * w, 0.52 * h))
    ents.append(_rect(-0.22 * w, -0.50 * h, 0.44 * w, 0.05 * h))
    return ents


def _apple(w, h):
    """苹果：圆形果身 + 顶部内凹 + 果柄 + 小叶。"""
    right = [(0.02 * w, -0.50 * h), (0.26 * w, -0.46 * h), (0.42 * w, -0.30 * h),
             (0.48 * w, -0.06 * h), (0.43 * w, 0.18 * h), (0.28 * w, 0.35 * h),
             (0.12 * w, 0.42 * h), (0.03 * w, 0.35 * h)]
    ents = [_poly(right + _mirror_x(right[::-1])[1:], closed=True)]
    ents.append(_line((0.0, 0.36 * h), (0.02 * w, 0.48 * h)))
    ents.append(_poly(_ellipse(0.13 * w, 0.40 * h, 0.10 * w, 0.045 * h, 18)))
    return ents


def _pear(w, h):
    """梨：下宽上窄的梨形 + 蒂。"""
    prof = [(-0.50, 0.03), (-0.45, 0.32), (-0.36, 0.56), (-0.24, 0.68),
            (-0.12, 0.66), (-0.03, 0.52), (0.05, 0.38), (0.14, 0.28),
            (0.24, 0.21), (0.34, 0.16), (0.43, 0.12), (0.50, 0.09)]
    ents = [_poly(_revolve(prof, w, h))]
    ents.append(_line((0, 0.47 * h), (0.01 * w, 0.50 * h)))
    return ents


def _banana(w, h):
    """香蕉：弯月形果身（中间粗、两端收尖）+ 两端小梗。"""
    p0 = (-0.46 * w, -0.34 * h)
    pc = (0.0, 0.88 * h)
    p1 = (0.46 * w, -0.34 * h)
    n = 36
    thick = 0.20 * h
    centers, normals = [], []
    for i in range(n + 1):
        t = i / float(n)
        m = 1 - t
        x = m * m * p0[0] + 2 * m * t * pc[0] + t * t * p1[0]
        y = m * m * p0[1] + 2 * m * t * pc[1] + t * t * p1[1]
        dx = 2 * m * (pc[0] - p0[0]) + 2 * t * (p1[0] - pc[0])
        dy = 2 * m * (pc[1] - p0[1]) + 2 * t * (p1[1] - pc[1])
        L = math.hypot(dx, dy) or 1.0
        centers.append((x, y))
        normals.append((-dy / L, dx / L))
    outer, inner = [], []
    for i, (c, nv) in enumerate(zip(centers, normals)):
        t = i / float(n)
        d = thick * (math.sin(math.pi * t) ** 0.72)
        outer.append((c[0] + nv[0] * d, c[1] + nv[1] * d))
        inner.append((c[0] - nv[0] * d, c[1] - nv[1] * d))
    ents = [_poly(outer + inner[::-1], closed=True)]
    ents.append(_line(p0, (p0[0] - 0.05 * w, p0[1] + 0.07 * h)))
    ents.append(_line(p1, (p1[0] + 0.05 * w, p1[1] + 0.07 * h)))
    return ents


def _watermelon(w, h):
    """西瓜片：半圆 + 双道瓜皮 + 瓜籽。"""
    cy = -0.45 * h
    R = 0.5 * w
    ents = [_poly(_arcpts(0, cy, R, 0, 180, 48), closed=False)]
    ents.append(_poly(_arcpts(0, cy, R * 0.90, 0, 180, 44), closed=False))
    ents.append(_line((-R, cy), (R, cy)))
    for sx, sy in ((-0.22, 0.10), (0.02, 0.22), (0.24, 0.10), (-0.10, 0.05), (0.14, 0.05)):
        ents.append(_poly(_ellipse(sx * R, cy + sy * h, 0.03 * w, 0.05 * h, 12)))
    return ents


def _icecream(w, h):
    """冰淇淋：甜筒（网格）+ 两个球 + 樱桃。"""
    ents = [_poly([(-0.5 * w, 0.02 * h), (0.5 * w, 0.02 * h), (0, -0.50 * h)],
                  closed=True)]
    for k in (1, 2):
        yv = 0.02 * h - k * 0.15 * h
        half = 0.5 * w * (1 - k * 0.33)
        ents.append(_line((-half, yv), (half, yv)))
    ents.append(_line((-0.30 * w, -0.02 * h), (0.22 * w, 0.02 * h)))
    ents.append(_line((0.30 * w, -0.02 * h), (-0.22 * w, 0.02 * h)))
    ents.append(_poly(_arcpts(-0.17 * w, 0.06 * h, 0.22 * w, 0, 180, 20), closed=False))
    ents.append(_poly(_arcpts(0.17 * w, 0.06 * h, 0.22 * w, 0, 180, 20), closed=False))
    ents.append(_poly(_arcpts(0, 0.12 * h, 0.25 * w, 0, 180, 20), closed=False))
    ents.append(_circle(0, 0.44 * h, 0.05 * w))
    return ents


def _cake(w, h):
    """蛋糕：两层胚 + 蜡烛 + 火苗 + 樱桃。"""
    ents = [_rect(-0.50 * w, -0.50 * h, 1.0 * w, 0.30 * h)]
    ents.append(_line((-0.50 * w, -0.22 * h), (0.50 * w, -0.22 * h)))
    ents.append(_rect(-0.36 * w, -0.20 * h, 0.72 * w, 0.26 * h))
    ents.append(_line((-0.36 * w, 0.02 * h), (0.36 * w, 0.02 * h)))
    ents.append(_rect(-0.02 * w, 0.06 * h, 0.04 * w, 0.26 * h))
    ents.append(_poly(_ellipse(0, 0.38 * h, 0.04 * w, 0.07 * h, 18)))
    for sx in (-0.26 * w, 0.26 * w):
        ents.append(_circle(sx, 0.10 * h, 0.035 * w))
    return ents


def _gear(w, h, teeth=12):
    """齿轮：方齿外轮廓 + 轴孔 + 减重孔。"""
    R = 0.5 * min(w, h)
    rr = R * 0.76
    step = 2 * math.pi / teeth
    tw = step * 0.28
    pts = []
    for k in range(teeth):
        a = step * k
        pts += [(rr * math.cos(a - step / 2), rr * math.sin(a - step / 2)),
                (R * math.cos(a - tw), R * math.sin(a - tw)),
                (R * math.cos(a + tw), R * math.sin(a + tw)),
                (rr * math.cos(a + step / 2), rr * math.sin(a + step / 2))]
    ents = [_poly(pts, closed=True), _circle(0, 0, R * 0.18)]
    for k in range(4):
        a = math.radians(45 + 90 * k)
        ents.append(_circle(R * 0.48 * math.cos(a), R * 0.48 * math.sin(a), R * 0.13))
    return ents


def _nut(w, h):
    """六角螺母：六边形 + 内孔 + 螺纹线。"""
    R = 0.5 * min(w, h)
    hexa = [(R * math.cos(math.radians(90 + 60 * k)),
             R * math.sin(math.radians(90 + 60 * k))) for k in range(6)]
    ents = [_poly(hexa, closed=True), _circle(0, 0, R * 0.56)]
    ents.append(_poly(_arcpts(0, 0, R * 0.50, 20, 340, 40), closed=False))
    return ents


def _screw(w, h):
    """螺丝：圆柱头 + 杆 + 螺纹。"""
    ents = [_poly([(-0.32 * w, 0.50 * h), (0.32 * w, 0.50 * h),
                   (0.18 * w, 0.34 * h), (-0.18 * w, 0.34 * h)], closed=True)]
    ents.append(_poly([(-0.11 * w, 0.34 * h), (-0.09 * w, -0.26 * h),
                       (0, -0.50 * h), (0.09 * w, -0.26 * h),
                       (0.11 * w, 0.34 * h)], closed=True))
    for k in range(6):
        yv = 0.28 * h - k * 0.10 * h
        ents.append(_line((-0.105 * w, yv), (0.105 * w, yv - 0.045 * h)))
    return ents


def _envelope(w, h):
    """信封：矩形 + 上翻盖 + 下折角。"""
    ents = [_rect(-0.5 * w, -0.36 * h, 1.0 * w, 0.72 * h)]
    ents.append(_line((-0.5 * w, 0.36 * h), (0, 0.0)))
    ents.append(_line((0, 0.0), (0.5 * w, 0.36 * h)))
    ents.append(_line((-0.5 * w, -0.36 * h), (0, 0.0)))
    ents.append(_line((0, 0.0), (0.5 * w, -0.36 * h)))
    return ents


def _pin(w, h):
    """图钉：圆头 + 锥体 + 针尖。"""
    ents = [_poly(_ellipse(0, 0.32 * h, 0.21 * w, 0.14 * h, 28))]
    ents.append(_poly([(-0.13 * w, 0.22 * h), (0.13 * w, 0.22 * h),
                       (0.05 * w, -0.12 * h), (-0.05 * w, -0.12 * h)], closed=True))
    ents.append(_line((0, -0.12 * h), (0, -0.50 * h)))
    return ents


def _battery(w, h):
    """电池：外壳 + 正极凸头 + 分隔线 + 正负极标记。"""
    ents = [_rect(-0.50 * w, -0.42 * h, 1.0 * w, 0.84 * h)]
    ents.append(_rect(-0.16 * w, 0.42 * h, 0.32 * w, 0.08 * h))
    for yv in (-0.14, 0.14):
        ents.append(_line((-0.50 * w, yv * h), (0.50 * w, yv * h)))
    ents.append(_line((0.15 * w, 0.27 * h), (0.29 * w, 0.27 * h)))
    ents.append(_line((0.22 * w, 0.20 * h), (0.22 * w, 0.34 * h)))
    ents.append(_line((-0.29 * w, 0.27 * h), (-0.15 * w, 0.27 * h)))
    return ents


def _stick(w, h):
    """火柴人：头 + 躯干 + 双臂 + 双腿。"""
    ents = [_circle(0, 0.36 * h, 0.12 * h)]
    ents.append(_line((0, 0.24 * h), (0, -0.10 * h)))
    ents.append(_line((-0.26 * w, 0.12 * h), (0.26 * w, 0.12 * h)))
    ents.append(_line((0, -0.10 * h), (-0.20 * w, -0.48 * h)))
    ents.append(_line((0, -0.10 * h), (0.20 * w, -0.48 * h)))
    return ents


def _crown(w, h):
    """皇冠：五尖冠体 + 冠带 + 宝石。"""
    pts = [(-0.48 * w, -0.30 * h), (-0.48 * w, 0.26 * h), (-0.30 * w, 0.04 * h),
           (-0.16 * w, 0.46 * h), (0, 0.12 * h), (0.16 * w, 0.46 * h),
           (0.30 * w, 0.04 * h), (0.48 * w, 0.26 * h), (0.48 * w, -0.30 * h)]
    ents = [_poly(pts, closed=True)]
    ents.append(_line((-0.48 * w, -0.12 * h), (0.48 * w, -0.12 * h)))
    for sx in (-0.32 * w, 0, 0.32 * w):
        ents.append(_circle(sx, -0.03 * h, 0.035 * w))
    return ents


def _hat(w, h):
    """帽子：帽檐 + 帽身 + 帽带。"""
    ents = [_poly(_ellipse(0, -0.26 * h, 0.5 * w, 0.09 * h, 40))]
    ents.append(_poly([(-0.30 * w, -0.24 * h), (-0.26 * w, 0.16 * h),
                       (-0.14 * w, 0.40 * h), (0.14 * w, 0.40 * h),
                       (0.26 * w, 0.16 * h), (0.30 * w, -0.24 * h)], closed=True))
    ents.append(_line((-0.27 * w, -0.02 * h), (0.27 * w, -0.02 * h)))
    return ents


# ============================================================ 概念表
# size = (默认宽, 默认高) mm；parts = 给人看的部位说明（也用于提示词）
CONCEPTS = [
    {"id": "gourd", "cn": "葫芦", "aliases": ("葫芦", "hulu", "gourd"),
     "size": (100, 150), "recipe": _gourd,
     "parts": "上小下大两个鼓球、中间收腰、顶部一小段蒂"},
    {"id": "mug", "cn": "杯子", "aliases": ("马克杯", "茶杯", "咖啡杯", "杯子", "杯", "mug", "cup"),
     "size": (90, 100), "recipe": _mug, "parts": "杯身微收口、杯口、右侧耳朵形把手"},
    {"id": "bottle", "cn": "瓶子", "aliases": ("瓶子", "酒瓶", "水瓶", "瓶", "bottle"),
     "size": (70, 160), "recipe": _bottle, "parts": "圆柱瓶身、肩部收口、瓶颈、瓶盖"},
    {"id": "bowl", "cn": "碗", "aliases": ("饭碗", "碗", "bowl"),
     "size": (120, 70), "recipe": _bowl, "parts": "半球形碗壁、小圆底、宽口沿"},
    {"id": "plate", "cn": "盘子", "aliases": ("盘子", "碟子", "盘", "plate", "dish"),
     "size": (140, 40), "recipe": _plate, "parts": "浅弧碟面 + 小底圈"},
    {"id": "spoon", "cn": "勺子", "aliases": ("勺子", "汤匙", "匙", "spoon"),
     "size": (40, 130), "recipe": _spoon, "parts": "椭圆勺头 + 细直柄"},
    {"id": "umbrella", "cn": "雨伞", "aliases": ("雨伞", "伞", "umbrella"),
     "size": (130, 150), "recipe": _umbrella, "parts": "半圆伞面（下缘三个扇贝）+ 伞杆 + 弯手柄 + 伞骨"},
    {"id": "key", "cn": "钥匙", "aliases": ("钥匙", "锁匙", "key"),
     "size": (130, 60), "recipe": _key, "parts": "左端圆环 + 长杆 + 右端两枚齿"},
    {"id": "bulb", "cn": "灯泡", "aliases": ("灯泡", "电灯泡", "bulb", "lamp bulb"),
     "size": (80, 130), "recipe": _bulb, "parts": "梨形玻璃泡 + 螺纹灯头 + 底触点"},
    {"id": "pencil", "cn": "铅笔", "aliases": ("铅笔", "pencil"),
     "size": (32, 160), "recipe": _pencil, "parts": "长条笔身 + 锥形笔尖 + 尾部橡皮"},
    {"id": "book", "cn": "书", "aliases": ("书本", "书籍", "书", "book"),
     "size": (110, 140), "recipe": _book, "parts": "矩形封面 + 书脊 + 几条页线"},
    {"id": "clock", "cn": "闹钟", "aliases": ("闹钟", "时钟", "钟", "clock", "alarm clock"),
     "size": (120, 130), "recipe": _clock, "parts": "圆形表盘 + 12 刻度 + 时针分针 + 顶部双铃 + 两只脚"},
    {"id": "house", "cn": "房子", "aliases": ("房子", "房屋", "小屋", "house", "home"),
     "size": (150, 140), "recipe": _house, "parts": "方墙 + 三角屋顶 + 门 + 两扇十字窗"},
    {"id": "door", "cn": "门", "aliases": ("房门", "门", "door"),
     "size": (90, 180), "recipe": _door, "parts": "门框 + 上下门板 + 门把手"},
    {"id": "window", "cn": "窗", "aliases": ("窗户", "窗", "window"),
     "size": (120, 120), "recipe": _window, "parts": "外框 + 内框 + 十字窗棂"},
    {"id": "lamp", "cn": "台灯", "aliases": ("台灯", "落地灯", "lamp", "desk lamp"),
     "size": (120, 150), "recipe": _lamp, "parts": "扁圆底座 + 细灯杆 + 梯形灯罩"},
    {"id": "chair", "cn": "椅子", "aliases": ("椅子", "座椅", "椅", "chair"),
     "size": (100, 130), "recipe": _chair, "parts": "座面 + 靠背 + 两条腿 + 横撑"},
    {"id": "table", "cn": "桌子", "aliases": ("桌子", "书桌", "餐桌", "桌", "table", "desk"),
     "size": (160, 100), "recipe": _table, "parts": "长桌板 + 两条腿 + 横撑"},
    {"id": "tree", "cn": "树", "aliases": ("树", "大树", "树木", "tree"),
     "size": (140, 170), "recipe": _tree, "parts": "短树干 + 团状（云朵形）树冠"},
    {"id": "cloud", "cn": "云", "aliases": ("云朵", "云", "cloud"),
     "size": (160, 90), "recipe": _cloud, "parts": "几个相连的圆弧鼓包，底部略平"},
    {"id": "sun", "cn": "太阳", "aliases": ("太阳", "sun"),
     "size": (150, 150), "recipe": _sun, "parts": "圆盘 + 12 道放射光芒"},
    {"id": "moon", "cn": "月亮", "aliases": ("月牙", "月亮", "moon", "crescent"),
     "size": (110, 130), "recipe": _moon, "parts": "大圆减去偏置小圆形成的月牙"},
    {"id": "mountain", "cn": "山", "aliases": ("山峰", "山脉", "山", "mountain"),
     "size": (180, 110), "recipe": _mountain, "parts": "两座三角峰 + 雪线"},
    {"id": "flower", "cn": "花", "aliases": ("花朵", "花", "flower"),
     "size": (120, 160), "recipe": _flower, "parts": "6 片椭圆花瓣 + 圆花心 + 花茎 + 一片叶"},
    {"id": "leaf", "cn": "叶子", "aliases": ("树叶", "叶子", "叶", "leaf"),
     "size": (100, 150), "recipe": _leaf, "parts": "两条对称弧线合成叶形 + 主脉 + 侧脉"},
    {"id": "snowflake", "cn": "雪花", "aliases": ("雪花", "snowflake"),
     "size": (130, 130), "recipe": _snowflake, "parts": "6 条主枝，每条两组 38° 分叉"},
    {"id": "cat", "cn": "猫", "aliases": ("猫咪", "小猫", "猫", "cat"),
     "size": (120, 130), "recipe": _cat, "parts": "圆头 + 两只三角耳 + 两眼 + 三角鼻 + 六根胡须"},
    {"id": "dog", "cn": "狗", "aliases": ("小狗", "狗狗", "狗", "dog"),
     "size": (130, 130), "recipe": _dog, "parts": "圆头 + 两只垂耳 + 口鼻 + 鼻头 + 两眼"},
    {"id": "fish", "cn": "鱼", "aliases": ("金鱼", "小鱼", "鱼", "fish"),
     "size": (150, 90), "recipe": _fish, "parts": "纺锤身体 + 三角尾鳍 + 背鳍 + 圆眼 + 鳃线"},
    {"id": "bird", "cn": "鸟", "aliases": ("小鸟", "鸟", "bird"),
     "size": (150, 120), "recipe": _bird, "parts": "蛋形身体 + 圆头 + 三角喙 + 翅膀 + 尾羽 + 两条腿"},
    {"id": "butterfly", "cn": "蝴蝶", "aliases": ("蝴蝶", "butterfly"),
     "size": (160, 130), "recipe": _butterfly, "parts": "细长身体 + 左右各两片翅膀 + 两根触角"},
    {"id": "rabbit", "cn": "兔子", "aliases": ("兔子", "小兔", "兔", "rabbit", "bunny"),
     "size": (110, 170), "recipe": _rabbit, "parts": "椭圆身体 + 圆头 + 两只长耳 + 眼 + 两只脚 + 短尾"},
    {"id": "snail", "cn": "蜗牛", "aliases": ("蜗牛", "snail"),
     "size": (140, 100), "recipe": _snail, "parts": "螺旋壳 + 长条身体 + 两根触角"},
    {"id": "penguin", "cn": "企鹅", "aliases": ("企鹅", "penguin"),
     "size": (110, 170), "recipe": _penguin, "parts": "蛋形身体 + 白色腹部 + 两眼 + 三角喙 + 两只脚"},
    {"id": "car", "cn": "汽车", "aliases": ("小汽车", "轿车", "汽车", "车", "car"),
     "size": (190, 100), "recipe": _car, "parts": "车身轮廓（前后翼子板、车顶）+ 车窗 + 两个圆轮"},
    {"id": "boat", "cn": "船", "aliases": ("帆船", "小船", "船", "boat", "ship"),
     "size": (160, 150), "recipe": _boat, "parts": "梯形船体 + 桅杆 + 两面三角帆"},
    {"id": "plane", "cn": "飞机", "aliases": ("飞机", "客机", "plane", "airplane"),
     "size": (190, 170), "recipe": _plane, "parts": "细长机身 + 一对主翼 + 一对尾翼 + 一排舷窗"},
    {"id": "bicycle", "cn": "自行车", "aliases": ("自行车", "单车", "bicycle", "bike"),
     "size": (180, 110), "recipe": _bicycle, "parts": "两个圆轮 + 三角车架 + 车把 + 座垫"},
    {"id": "traffic", "cn": "红绿灯", "aliases": ("红绿灯", "信号灯", "traffic light"),
     "size": (80, 180), "recipe": _traffic, "parts": "竖长灯箱 + 三个圆灯 + 灯杆 + 底座"},
    {"id": "apple", "cn": "苹果", "aliases": ("苹果", "apple"),
     "size": (110, 120), "recipe": _apple, "parts": "圆形果身 + 顶部内凹 + 果柄 + 一片小叶"},
    {"id": "pear", "cn": "梨", "aliases": ("梨子", "梨", "pear"),
     "size": (100, 130), "recipe": _pear, "parts": "下宽上窄的梨形轮廓 + 细小果柄"},
    {"id": "banana", "cn": "香蕉", "aliases": ("香蕉", "banana"),
     "size": (150, 90), "recipe": _banana, "parts": "弯月形果身，两端收尖"},
    {"id": "watermelon", "cn": "西瓜", "aliases": ("西瓜片", "西瓜", "watermelon"),
     "size": (150, 90), "recipe": _watermelon, "parts": "半圆切片 + 两圈瓜皮 + 几粒瓜子"},
    {"id": "icecream", "cn": "冰淇淋", "aliases": ("冰淇淋", "冰激凌", "雪糕", "ice cream"),
     "size": (110, 170), "recipe": _icecream, "parts": "倒三角甜筒（网格纹）+ 两个半圆球 + 樱桃"},
    {"id": "cake", "cn": "蛋糕", "aliases": ("蛋糕", "cake"),
     "size": (140, 150), "recipe": _cake, "parts": "两层蛋糕胚 + 蜡烛 + 火苗 + 两颗樱桃"},
    {"id": "gear", "cn": "齿轮", "aliases": ("齿轮", "gear"),
     "size": (140, 140), "recipe": _gear, "parts": "12 个方齿外轮廓 + 中心轴孔 + 4 个减重孔"},
    {"id": "nut", "cn": "螺母", "aliases": ("六角螺母", "螺母", "nut"),
     "size": (110, 110), "recipe": _nut, "parts": "正六边形外形 + 中心圆孔 + 螺纹圈"},
    {"id": "screw", "cn": "螺丝", "aliases": ("螺丝", "螺钉", "螺栓", "screw", "bolt"),
     "size": (70, 170), "recipe": _screw, "parts": "圆柱/锥头 + 细杆 + 一排斜螺纹"},
    {"id": "envelope", "cn": "信封", "aliases": ("信封", "envelope"),
     "size": (150, 110), "recipe": _envelope, "parts": "矩形信封 + 上翻盖 + 两道下折线"},
    {"id": "pin", "cn": "图钉", "aliases": ("图钉", "大头针", "pin"),
     "size": (70, 140), "recipe": _pin, "parts": "扁圆头 + 锥形颈 + 细针"},
    {"id": "battery", "cn": "电池", "aliases": ("电池", "battery"),
     "size": (70, 150), "recipe": _battery, "parts": "矩形壳体 + 顶部正极凸头 + 分隔线 + 正负极符号"},
    {"id": "stick", "cn": "火柴人", "aliases": ("火柴人", "小人", "stick figure"),
     "size": (110, 170), "recipe": _stick, "parts": "圆头 + 直线躯干 + 双臂 + 双腿"},
    {"id": "crown", "cn": "皇冠", "aliases": ("皇冠", "王冠", "crown"),
     "size": (150, 110), "recipe": _crown, "parts": "五个尖的冠体 + 底边冠带 + 三颗宝石"},
    {"id": "hat", "cn": "帽子", "aliases": ("礼帽", "帽子", "帽", "hat"),
     "size": (150, 110), "recipe": _hat, "parts": "椭圆帽檐 + 梯形帽身 + 帽带"},
]


# ============================================================ 指令解析
_NEG = ("不要", "无需", "不需要", "别", "勿", "不画", "不生成", "不做")
_EDIT_PREFIX = re.compile(r"^\s*(把|将|改|修改|删除|移动|旋转|缩放|镜像|阵列|标注|测量)")

_NUM = r"([0-9]+(?:\.[0-9]+)?|[零一二两三四五六七八九十百千]+)"
_UNIT = r"\s*(mm|毫米|cm|厘米|公分)?"

_PAT_WH = re.compile(r"宽\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT +
                     r"\s*[，,、/]?\s*高\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT)
_PAT_HW = re.compile(r"高\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT +
                     r"\s*[，,、/]?\s*宽\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT)
_PAT_D = re.compile(r"(?:直径|口径)\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT)
_PAT_R = re.compile(r"半径\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT)
_PAT_H = re.compile(r"(?:高|高度|长|长度|身高)\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT)
_PAT_W = re.compile(r"(?:宽|宽度|大小|尺寸|边长)\s*(?:为|是|有|=|:|：)?\s*" + _NUM + _UNIT)
_PAT_BARE = re.compile(_NUM + r"\s*(mm|毫米|cm|厘米|公分)")
_PAT_COUNT = re.compile(r"([0-9]+|[一二两三四五六七八九十]+)\s*(?:个|只|条|棵|朵|辆|座|张|把|枚|顶)")


def _mm(v, unit):
    if v is None:
        return None
    return v * 10.0 if (unit in ("cm", "厘米", "公分")) else v


def _parse_size(text):
    """从文本里取 宽/高/直径/半径/裸尺寸；返回 {'w','h'}（可能有缺项）。"""
    out = {}
    m = _PAT_WH.search(text)
    if m:
        return {"w": _mm(_num_any(m.group(1)), m.group(2)),
                "h": _mm(_num_any(m.group(3)), m.group(4))}
    m = _PAT_HW.search(text)
    if m:
        return {"h": _mm(_num_any(m.group(1)), m.group(2)),
                "w": _mm(_num_any(m.group(3)), m.group(4))}
    m = _PAT_D.search(text)
    if m:
        d = _mm(_num_any(m.group(1)), m.group(2))
        if d:
            out["w"] = d
    m = _PAT_R.search(text)
    if m:
        r = _mm(_num_any(m.group(1)), m.group(2))
        if r:
            out["w"] = 2 * r
    m = _PAT_H.search(text)
    if m:
        v = _mm(_num_any(m.group(1)), m.group(2))
        if v:
            out["h"] = v
    m = _PAT_W.search(text)
    if m:
        v = _mm(_num_any(m.group(1)), m.group(2))
        if v:
            out["w"] = v
    if not out:
        m = _PAT_BARE.search(text)
        if m:
            v = _mm(_num_any(m.group(1)), m.group(2))
            if v:
                out["h"] = v
    return out


def _parse_count(text):
    m = _PAT_COUNT.search(text)
    n = _num_any(m.group(1)) if m else 1
    n = int(n) if n else 1
    return max(1, min(6, n))


def _match(text):
    """扫描文本中的概念词（长词优先、去否定、去编辑类）。"""
    low = text.lower()
    hits = []
    for c in CONCEPTS:
        for al in c["aliases"]:
            for m in re.finditer(re.escape(al.lower()), low):
                hits.append((m.start(), m.end(), c))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    picked, last, seen = [], -1, set()
    for s, e, c in hits:
        if s < last or id(c) in seen:
            continue
        if any(w in text[max(0, s - 6):s] for w in _NEG):
            continue
        picked.append((s, e, c))
        seen.add(id(c))
        last = e
    return picked


def _row(items, gap):
    """把多组实体横向排开并垂直居中；items: [(entities, w, h)]。"""
    if len(items) == 1:
        return items[0][0]
    total = sum(it[1] for it in items) + gap * (len(items) - 1)
    out, x = [], -total / 2.0
    for ents, w_, h_ in items:
        out.extend(_shift(ents, x + w_ / 2.0, 0))
        x += w_ + gap
    return out


def parse_concept_instruction(instruction):
    """把自然语言指令解析成概念库几何。

    返回 (spec, desc)：命中 → spec 可直接给 dxf_writer；未命中 → (None, "")。
    """
    if not instruction or not instruction.strip():
        return None, ""
    text = instruction.strip()
    if _EDIT_PREFIX.match(text):
        return None, ""
    hits = _match(text)
    if not hits:
        return None, ""

    size = _parse_size(text)
    count = _parse_count(text) if len(hits) == 1 else 1
    items, descs, names = [], [], []
    for _s, _e, c in hits:
        w0, h0 = c["size"]
        w = size.get("w") or None
        h = size.get("h") or None
        if w and h:
            pass
        elif w:
            h = w * h0 / float(w0)          # 只给宽 → 按默认比例推高
        elif h:
            w = h * w0 / float(h0)          # 只给高 → 按默认比例推宽
        else:
            w, h = w0, h0
        w = max(6.0, min(2000.0, float(w)))
        h = max(6.0, min(2000.0, float(h)))
        ents = _fit(_center(c["recipe"](w, h)), w, h)
        for _ in range(count):
            items.append((ents, w, h))
        descs.append(f"{c['cn']}（{c['parts']}；{w:g}×{h:g}mm）")
        names.append(c["id"])

    gap = max(12.0, 0.12 * max(it[2] for it in items))

    # ---- 同一句里若还点名了几何基本形（五角星、正六边形、心形…），一起排版 ----
    # 只在出现并列连词时才合并，避免把"圆形的杯子"里的"圆形"当成另一个图元。
    rest = text
    for s, e, _c in hits:
        rest = rest[:s] + " " * max(1, e - s) + rest[e:]
    if re.search(r"[和与及、,，+]", rest):
        try:
            from shapes import parse_shape_instruction
            extra_spec, extra_desc = parse_shape_instruction(rest)
        except Exception:                       # noqa: BLE001
            extra_spec, extra_desc = None, ""
        if extra_spec and extra_spec.get("entities"):
            ex = extra_spec["entities"]
            x0, y0, x1, y1 = _bbox(ex)
            items.append((ex, max(1.0, x1 - x0), max(1.0, y1 - y0)))
            descs.append(extra_desc)

    entities = _row(items, gap)
    if not entities:
        return None, ""
    spec = {"name": ("_".join(names))[:48] or "concept", "unit": "mm",
            "entities": entities}
    if count > 1:
        descs = [f"{descs[0].rstrip('）')}，共 {count} 个）"]
    return spec, "；".join(descs)


def catalog():
    """返回 [(中文名, 部位说明)]，供帮助信息/调试使用。"""
    return [(c["cn"], c["parts"]) for c in CONCEPTS]


def concept_words():
    """所有可识别的主词（含别名），用于提示与测试。"""
    words = []
    for c in CONCEPTS:
        words.append(c["cn"])
        words.extend(a for a in c["aliases"] if len(a) > len(c["cn"]))
    return words


__all__ = ["parse_concept_instruction", "catalog", "concept_words", "CONCEPTS"]
