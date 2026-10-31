"""shapes.py — 服务器端确定性参数化图形库。

为什么需要它：本地 7B 模型处理需要三角计算的图形时不可靠。实测：
- "画一个五角星"              → 输出 9 顶点乱多边形
- "画一个五角星，外接圆半径50" → 画成了五边形
- "画一个正六边形"            → 侥幸正确
五角星需要 10 个顶点、外半径 R 与内半径 0.382R 交替、角度每次 +36°，
7B 模型算不准三角函数。本模块用确定性几何直接生成这类图形：
不过模型、毫秒级、零误差。

支持：N 角星、正 N 边形、矩形/正方形、圆、三角形（直角/等腰/等边）、
菱形、十字、箭头、心形、线段，以及"圆和五角星"这类简单组合（自动排版）。
未命中或无法可靠解析时返回 (None, "")，由调用方回退到大模型。
"""

import math
import re

# ---------------------------------------------------------------- 数字解析
_CN = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
       "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_NUM = r"([0-9]+(?:\.[0-9]+)?|[零一二两三四五六七八九十百千]+)"


def _num(s):
    """'50' / '3.5' / '五十' / '一百零五' → float；无法解析返回 None。"""
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", s):
        return float(s)
    if not re.fullmatch(r"[零一二两三四五六七八九十百千]+", s):
        return None
    total, section, number = 0, 0, 0
    for ch in s:
        if ch in _CN:
            number = _CN[ch]
        elif ch in _CN_UNIT:
            section += (number if number else 1) * _CN_UNIT[ch]
            number = 0
        else:
            return None
    return float(total + section + number)


# ---------------------------------------------------------------- 图形识别
# (类型, 正则)。顺序即优先级；更具体的词放在前面（如"直角三角形"先于"三角形"）。
_SHAPE_DEFS = [
    ("star",     r"(?:([零一二两三四五六七八九十百0-9]+)\s*角星|★|☆|星形|星星)"),
    ("polygon",  r"(?:(?:正\s*)?([零一二两三四五六七八九十百0-9]+)\s*边形|等边三角形|正多边形|多边形)"),
    ("rect",     r"(?:矩形|长方形|正方形|方形)"),
    ("circle",   r"(?:圆形|圆)"),
    ("triangle", r"(?:直角三角形|等腰三角形|等边三角形|三角形)"),
    ("diamond",  r"(?:菱形)"),
    ("cross",    r"(?:十字架|十字|十字符)"),
    ("arrow",    r"(?:箭头形|箭头)"),
    ("heart",    r"(?:心形|爱心)"),
    ("line",     r"(?:线段|直线)"),
]

# 否定语：出现在图形词前 6 字内 → 不算图形指令（"不要画五角星"）。
# 注意：只用完整否定词，不要放单字"不"——否则"不规则七边形"会被误判为否定。
_NEG = ("不要", "无需", "不需要", "别", "勿", "莫", "不画", "不生成", "不做")

# 多图形组合时的"语义黑名单"：出现这些词说明是带孔/嵌套的机械件，
# 简单横排会出错，整体交给大模型
_MULTI_BLACKLIST = ("孔", "中间", "内部", "里面", "穿", "嵌", "槽", "配合", "嵌套")

# 编辑类指令开头 → 不走图形库（交给模型理解意图）
_EDIT_PREFIX = re.compile(r"^\s*(把|将|改|修改|删除|移动|旋转|缩放|镜像|阵列|标注|测量)")


def _find_mentions(text):
    """扫描文本中的所有图形提及：去重叠、去否定语、去半圆/圆弧误匹配。"""
    # "外接圆/内切圆/圆心"里的"圆"不是圆形图元——先掩掉再扫描（等长替换，位置不变）
    scan = text
    for phrase in ("外接圆", "内切圆", "内接圆", "外圆", "圆心"):
        scan = scan.replace(phrase, "○" * len(phrase))
    raw = []
    for typ, pat in _SHAPE_DEFS:
        for m in re.finditer(pat, scan):
            raw.append({"type": typ, "start": m.start(), "end": m.end(),
                        "kw": m.group(0),
                        "n": _num(m.group(1)) if m.groups() and m.group(1) else None})
    raw.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    picked, last_end = [], -1
    for m in raw:
        if m["start"] < last_end:
            continue                      # 与已选中的更长/更早匹配重叠
        if m["type"] == "circle":
            if m["start"] > 0 and scan[m["start"] - 1] == "半":
                continue                  # 半圆
            if scan[m["end"]:m["end"] + 1] == "弧":
                continue                  # 圆弧
        ctx = scan[max(0, m["start"] - 6):m["start"]]
        if any(w in ctx for w in _NEG):
            continue                      # "不要画五角星"
        picked.append(m)
        last_end = m["end"]
    return picked


# ---------------------------------------------------------------- 参数解析
# 参数名前可接的连接词（含"都是/都为"，用于组合图形的共享参数）
_CONN = r"(?:都是|都为|都|为|是|等于|=|:|：)?"

# (参数名, 正则)。顺序即优先级（"长30宽20"整体先于单个"长30"）。
_PARAM_DEFS = [
    ("legs",     r"直角边\s*" + _CONN + r"\s*" + _NUM + r"\s*[和,、与及]\s*" + _NUM),
    ("diags",    r"对角线\s*" + _CONN + r"\s*" + _NUM + r"\s*[和,、与及]\s*" + _NUM),
    ("wh",       r"长\s*" + _CONN + r"\s*" + _NUM + r"\s*[，,、]?\s*宽\s*" + _CONN + r"\s*" + _NUM),
    ("hw",       r"宽\s*" + _CONN + r"\s*" + _NUM + r"\s*[，,、]?\s*长\s*" + _CONN + r"\s*" + _NUM),
    ("base_h",   r"底(?:边)?\s*" + _CONN + r"\s*" + _NUM + r"\s*[，,、]?\s*高\s*" + _CONN + r"\s*" + _NUM),
    ("diameter", r"直径\s*" + _CONN + r"\s*" + _NUM),
    ("inner",    r"内(?:切圆|接圆)?(?:半径|径)\s*" + _CONN + r"\s*" + _NUM),
    ("radius",   r"(?:外接圆|外圆)?半径\s*" + _CONN + r"\s*" + _NUM),
    ("R",        r"R\s*[=为是]?\s*" + _NUM),
    ("side",     r"边长\s*" + _CONN + r"\s*" + _NUM),
    ("size",     r"大小\s*" + _CONN + r"\s*" + _NUM),
    ("arm",      r"(?:臂长|总长|全长|长度)\s*" + _CONN + r"\s*" + _NUM),
    ("width",    r"(?:宽|粗细|厚度)\s*" + _CONN + r"\s*" + _NUM),
    ("len",      r"长\s*" + _CONN + r"\s*" + _NUM),
]


def _collect_params(text):
    """按优先级收集文本中的全部参数匹配 → [(start, end, kind, values)]。"""
    found = []
    for kind, pat in _PARAM_DEFS:
        for m in re.finditer(pat, text):
            vals = [_num(g) for g in m.groups()]
            if vals and all(v is not None and v > 0 for v in vals):
                found.append((m.start(), m.end(), kind, vals))
    return found


def _params_for(text):
    """单图形：全文参数（同区间被多个模式命中时保留高优先级的）。"""
    params, spans = {}, []
    for pos, end, kind, vals in _collect_params(text):
        if any(pos < e and s < end for s, e in spans):
            continue
        params[kind] = vals
        spans.append((pos, end))
    return params


# ---------------------------------------------------------------- 不规则多边形（给定各边长）
# 示例："画一个边长为 10、20、30、25、15 的五边形" → 各边长度不同的五边形。
# 经典做法：把顶点放到一个圆上，使相邻顶点间的弦长恰好等于给定边长。
# 设外接圆半径为 R，则第 i 条边的中心角 α_i = 2·asin(s_i / 2R)。要让多边形闭合，
# 需满足 Σα_i = 2π。对于给定的边长，存在唯一 R（> max(s)/2）满足该方程，用二分法解出。
_SIDES_PAT = re.compile(
    r"边长(?:\s*(?:分别|依次)?\s*(?:为|是|=|:|：))?\s*"
    r"([0-9]+(?:\.[0-9]+)?(?:[，,、，\s]+[0-9]+(?:\.[0-9]+)?){2,})"
)


def _fmt_side(v):
    return str(int(v)) if float(v).is_integer() else ("%.1f" % v)


def _extract_sides(text):
    """提取"边长 a、b、c …"形式的边长列表（≥3 个才有效）。"""
    m = _SIDES_PAT.search(text)
    if not m:
        return None
    out = []
    for x in re.split(r"[，,、\s]+", m.group(1).strip()):
        x = x.strip()
        if not x:
            continue
        v = _num(x)
        if v and v > 0:
            out.append(v)
    return out if len(out) >= 3 else None


def _irregular_polygon_entity(sides):
    """给定各边长度生成闭合不规则多边形（顶点共圆、弦长=边长，必闭合）。"""
    sides = [float(s) for s in sides if s and float(s) > 0]
    n = len(sides)
    if n < 3 or n > 64:
        return None
    maxs = max(sides)

    def total_angle(R):
        return sum(2 * math.asin(min(1.0, s / (2 * R))) for s in sides)

    # f(R)=Σα_i-2π=0；R→maxs/2 时 Σα_i≈n·π>2π，R→∞ 时 Σα_i→0；二分求根
    lo, hi = maxs / 2.0 + 1e-6, maxs * 1e6
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if total_angle(mid) > 2 * math.pi:
            lo = mid
        else:
            hi = mid
    R = (lo + hi) / 2.0
    angles = [math.pi / 2.0]
    for s in sides[:-1]:
        a = 2 * math.asin(min(1.0, s / (2 * R)))
        angles.append(angles[-1] + a)
    verts = [(R * math.cos(th), R * math.sin(th)) for th in angles]
    return _poly(_centered(verts), closed=True)


def _radius_of(p):
    for kind, div in (("radius", 1), ("R", 1), ("diameter", 2), ("size", 1)):
        if kind in p:
            return p[kind][0] / div
    return None


# ---------------------------------------------------------------- 几何生成
_CN_NUM = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


def _cn(n):
    """5 → '五'，12 → '十二'（用于中文描述）。"""
    n = int(n)
    if n <= 10:
        return _CN_NUM[n]
    if n < 20:
        return "十" + _CN_NUM[n - 10]
    return _CN_NUM[n // 10] + "十" + (_CN_NUM[n % 10] if n % 10 else "")
def _poly(pts, closed=True):
    # no_curve：正多边形/星形的折角是设计意图，别让曲线化把它们磨圆
    return {"type": "polyline",
            "points": [[round(x, 3), round(y, 3)] for x, y in pts],
            "closed": closed, "layer": "0", "no_curve": True}


def _star_entity(n=5, R=50.0, ratio=None):
    """N 角星：2N 个顶点，外半径 R 与内半径 r 交替，自正上方起每 180/N 度一个。
    默认内半径比 r/R = sin(π/2N)/sin(3π/2N)——N=5 时恰为经典的 0.382。"""
    n = max(3, min(20, int(n)))
    if ratio is None:
        ratio = math.sin(math.pi / (2 * n)) / math.sin(3 * math.pi / (2 * n))
    ratio = min(0.95, max(0.05, ratio))
    r = R * ratio
    pts = []
    for k in range(2 * n):
        ang = math.radians(90.0 + k * 180.0 / n)
        rad = R if k % 2 == 0 else r
        pts.append((rad * math.cos(ang), rad * math.sin(ang)))
    return _poly(pts)


def _polygon_entity(n=6, R=50.0):
    """正 N 边形：顶点均匀分布在半径 R 的外接圆上，自正上方起。"""
    n = max(3, min(64, int(n)))
    pts = [(R * math.cos(math.radians(90.0 + k * 360.0 / n)),
            R * math.sin(math.radians(90.0 + k * 360.0 / n))) for k in range(n)]
    return _poly(pts)


def _centered(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    dx, dy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return [(x - dx, y - dy) for x, y in pts]


def _heart_entity(scale=50.0):
    """心形（参数方程），按最大边归一化到 2*scale 的包围盒。"""
    pts = []
    for i in range(72):
        t = 2 * math.pi * i / 72
        pts.append((16 * math.sin(t) ** 3,
                    13 * math.cos(t) - 5 * math.cos(2 * t)
                    - 2 * math.cos(3 * t) - math.cos(4 * t)))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    k = (2 * scale) / max(max(xs) - min(xs), max(ys) - min(ys))
    return _poly(_centered([(x * k, y * k) for x, y in pts]))


def _build_shape(mention, params, sides=None):
    """按图形类型 + 参数生成 (entities, desc, name)；无法可靠生成返回 None。"""
    t = mention["type"]
    kw = mention["kw"]
    n = mention.get("n")

    def val(*kinds):
        for k in kinds:
            if k in params:
                return params[k]
        return None

    if t == "polygon":
        # 给定各边长 → 不规则多边形（弦长=边长，必闭合）
        if sides and len(sides) >= 3:
            e = _irregular_polygon_entity(sides)
            if e:
                nn = len(sides)
                sd = "/".join(_fmt_side(s) for s in sides)
                return ([e], f"不规则{_cn(nn)}边形(边长 {sd})",
                        f"polygon_irr_{nn}")
        if not n:
            n = 3 if "三角形" in kw else 0
        if not n or not 3 <= int(n) <= 64:
            return None
        n = int(n)
        R = _radius_of(params)
        if R is None:
            iv = val("inner")
            if iv:
                R = iv[0] / math.cos(math.pi / n)   # 内切圆半径 → 外接圆半径
            else:
                side = val("side")
                R = side[0] / (2 * math.sin(math.pi / n)) if side else 50.0
        return ([_polygon_entity(n, R)],
                f"正{_cn(n)}边形(外接圆R{R:g}mm)", f"polygon_{n}_r{R:g}")

    if t == "star":
        n = int(n or 5)
        if not 3 <= n <= 12:
            return None
        R = _radius_of(params) or 50.0
        ratio = None
        iv = val("inner")
        if iv:
            v = iv[0]
            ratio = v / R if v >= 2 else v
        return ([_star_entity(n, R, ratio)],
                f"{_cn(n)}角星(外接圆R{R:g}mm)", f"star_{n}_r{R:g}")

    if t == "polygon":
        if not n:
            n = 3 if "三角形" in kw else 0
        if not n or not 3 <= int(n) <= 64:
            return None
        n = int(n)
        R = _radius_of(params)
        if R is None:
            iv = val("inner")
            if iv:
                R = iv[0] / math.cos(math.pi / n)   # 内切圆半径 → 外接圆半径
            else:
                side = val("side")
                R = side[0] / (2 * math.sin(math.pi / n)) if side else 50.0
        return ([_polygon_entity(n, R)],
                f"正{_cn(n)}边形(外接圆R{R:g}mm)", f"polygon_{n}_r{R:g}")

    if t == "rect":
        square = (kw == "正方形")
        wh, hw = val("wh"), val("hw")
        if wh:
            w, h = wh[0], wh[1]
        elif hw:
            w, h = hw[1], hw[0]
        else:
            side = val("side")
            if side:
                w = h = side[0]
            else:
                ln = val("len", "arm")
                w = ln[0] if ln else 80.0
                h = w if square else round(w * 0.6, 3)
        if square and not (wh or hw or val("side")):
            w = h = 80.0
        return ([{"type": "rect", "x": round(-w / 2, 3), "y": round(-h / 2, 3),
                  "w": round(w, 3), "h": round(h, 3), "layer": "0"}],
                f"{'正方形' if square else '矩形'}({w:g}×{h:g}mm)",
                f"rect_{w:g}x{h:g}")

    if t == "circle":
        r = _radius_of(params) or 50.0
        return ([{"type": "circle", "center": [0, 0], "radius": round(r, 3),
                  "layer": "0"}],
                f"圆(半径{r:g}mm)", f"circle_r{r:g}")

    if t == "triangle":
        # 给定三条边长 → 不等边三角形（由两边交点确定第三顶点，必能拼出）
        if sides and len(sides) == 3 and all(s > 0 for s in sides):
            a, b, c = sides
            if a + b > c and a + c > b and b + c > a:
                x = (a * a + c * c - b * b) / (2 * a)   # 顶点 C 的 x（A=(0,0), B=(a,0)）
                yy = max(0.0, c * c - x * x)
                y = math.sqrt(yy)
                return ([_poly(_centered([(0, 0), (a, 0), (x, y)]))],
                        f"不等边三角形(边长 {_fmt_side(a)}/{_fmt_side(b)}/{_fmt_side(c)})",
                        f"tri_scalene_{_fmt_side(a)}_{_fmt_side(b)}_{_fmt_side(c)}")
        if "直角" in kw:
            legs = val("legs")
            a, b = (legs[0], legs[1]) if legs else (40.0, 30.0)
            return ([_poly(_centered([(0, 0), (a, 0), (0, b)]))],
                    f"直角三角形(直角边{a:g}×{b:g}mm)",
                    f"tri_right_{a:g}x{b:g}")
        if "等腰" in kw:
            bh = val("base_h")
            b, h = (bh[0], bh[1]) if bh else (80.0, 60.0)
            return ([_poly([(-b / 2, -h / 3), (b / 2, -h / 3), (0, 2 * h / 3)])],
                    f"等腰三角形(底{b:g}mm 高{h:g}mm)",
                    f"tri_isos_{b:g}x{h:g}")
        R = _radius_of(params)
        if R is None:
            side = val("side")
            R = side[0] / math.sqrt(3) if side else 60.0 / math.sqrt(3)
        return ([_polygon_entity(3, R)],
                f"等边三角形(外接圆R{R:g}mm)", f"tri_eq_r{R:g}")

    if t == "diamond":
        dg = val("diags")
        if dg:
            d1, d2 = dg[0], dg[1]
        else:
            R = _radius_of(params)
            d1 = d2 = 2 * R if R else 80.0
            d2 = d1 * 0.75
        return ([_poly([(d1 / 2, 0), (0, d2 / 2), (-d1 / 2, 0), (0, -d2 / 2)])],
                f"菱形(对角线{d1:g}×{d2:g}mm)", f"diamond_{d1:g}x{d2:g}")

    if t == "cross":
        wh, hw = val("wh"), val("hw")
        if wh:                       # "长80宽30的十字" → 臂长80、宽30
            L, w = wh[0], wh[1]
        elif hw:
            L, w = hw[1], hw[0]
        else:
            L = (val("arm") or val("len") or [60.0])[0]
            w = (val("width") or [L * 0.4])[0]
        w = min(w, L)
        pts = [(w / 2, L / 2), (w / 2, w / 2), (L / 2, w / 2), (L / 2, -w / 2),
               (w / 2, -w / 2), (w / 2, -L / 2), (-w / 2, -L / 2), (-w / 2, -w / 2),
               (-L / 2, -w / 2), (-L / 2, w / 2), (-w / 2, w / 2), (-w / 2, L / 2)]
        return ([_poly(pts)], f"十字(总长{L:g}mm 宽{w:g}mm)", f"cross_{L:g}x{w:g}")

    if t == "arrow":
        L = (val("arm") or val("len") or [60.0])[0]
        hl, hw = L * 0.35, L * 0.28
        sw = max(4.0, L * 0.12)
        pts = [(0, sw / 2), (L - hl, sw / 2), (L - hl, hw / 2), (L, 0),
               (L - hl, -hw / 2), (L - hl, -sw / 2), (0, -sw / 2)]
        return ([_poly([(x - L / 2, y) for x, y in pts])],
                f"箭头(长{L:g}mm)", f"arrow_{L:g}")

    if t == "heart":
        s = _radius_of(params)
        if s is None and val("size"):
            s = val("size")[0]
        s = s or 50.0
        return ([_heart_entity(s)], f"心形(高{2 * s:g}mm)", f"heart_{s:g}")

    if t == "line":
        L = (val("len") or val("arm") or [100.0])[0]
        return ([{"type": "line", "p1": [round(-L / 2, 3), 0],
                  "p2": [round(L / 2, 3), 0], "layer": "0"}],
                f"线段(长{L:g}mm)", f"line_{L:g}")

    return None


# ---------------------------------------------------------------- 组合排版
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


def _translate(ents, dx, dy):
    out = []
    for e in ents:
        e = dict(e)
        t = e["type"]
        if t == "circle":
            e["center"] = [e["center"][0] + dx, e["center"][1] + dy]
        elif t == "rect":
            e["x"] = round(e["x"] + dx, 3)
            e["y"] = round(e["y"] + dy, 3)
        elif t == "line":
            e["p1"] = [e["p1"][0] + dx, e["p1"][1] + dy]
            e["p2"] = [e["p2"][0] + dx, e["p2"][1] + dy]
        elif t == "polyline":
            e["points"] = [[round(p[0] + dx, 3), round(p[1] + dy, 3)]
                           for p in e["points"]]
        out.append(e)
    return out


def _layout(built_all):
    """多个图形横向排开（整体居中对齐），避免叠在一起。"""
    if len(built_all) == 1:
        return built_all[0][0]
    boxes = [_bbox(ents) for ents, _ in built_all]
    gap = 20.0
    total = sum(b[2] - b[0] for b in boxes) + gap * (len(boxes) - 1)
    out, x = [], -total / 2
    for (ents, _), (x0, y0, x1, y1) in zip(built_all, boxes):
        out.extend(_translate(ents, x - x0, -(y0 + y1) / 2))
        x += (x1 - x0) + gap
    return out


# ---------------------------------------------------------------- 主入口
def parse_shape_instruction(instruction):
    """把文本指令解析成确定性几何 spec。

    返回 (spec, desc)：
      命中图形库 → spec 可直接交给 dxf_writer.build_from_entities，desc 为中文说明；
      未命中     → (None, "")，调用方应回退到大模型。
    """
    if not instruction or not instruction.strip():
        return None, ""
    text = instruction.strip()
    if _EDIT_PREFIX.match(text):
        return None, ""
    mentions = _find_mentions(text)
    if not mentions:
        return None, ""

    if len(mentions) == 1:
        built = _build_shape(mentions[0], _params_for(text), _extract_sides(text))
        if not built:
            return None, ""
        entities, desc, name = built
        return {"name": name, "unit": "mm", "entities": entities}, desc

    # ---- 多图形：必须有连接词、且无机械语义黑名单，才做组合排版 ----
    between = text[mentions[0]["end"]:mentions[-1]["start"]]
    if not re.search(r"[，,。;；、]|和|跟|与|及|\+|&|加", between):
        return None, ""
    if any(w in text for w in _MULTI_BLACKLIST):
        return None, ""

    found = _collect_params(text)
    per = [dict() for _ in mentions]
    spans = [[] for _ in mentions]
    for pos, end, kind, vals in found:
        # "半径都是40" → 广播给所有图形；否则分配给距离最近的图形
        if "都" in text[pos:end]:
            targets = range(len(mentions))
        else:
            targets = [min(range(len(mentions)),
                           key=lambda j: abs(pos - mentions[j]["start"]))]
        for i in targets:
            if any(pos < e and s < end for s, e in spans[i]):
                continue
            per[i][kind] = vals
            spans[i].append((pos, end))

    built_all, names = [], []
    for i, m in enumerate(mentions):
        sides = _extract_sides(text) if m["type"] in ("polygon", "triangle") else None
        built = _build_shape(m, per[i], sides)
        if not built:
            return None, ""      # 有一个说不清 → 整体交给模型
        ents, desc, name = built
        built_all.append((ents, desc))
        names.append(name)
    desc = " + ".join(d for _, d in built_all)
    return ({"name": "_".join(names)[:48] or "shapes", "unit": "mm",
             "entities": _layout(built_all)}, desc)
