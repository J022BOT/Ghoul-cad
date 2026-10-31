# -*- coding: utf-8 -*-
"""纯 Python 的 DXF 读取器（零第三方依赖）。

存在的唯一目的：**「导入已有 DXF」不能依赖 ezdxf**。

背景（真实踩坑）：web_app 的重依赖（numpy / cv2 / ezdxf）全是懒加载的，
所以用任意一个"干净"的 Python（例如系统自带的 3.13）启动网页服务都能正常
打开界面，只有点「导入 DXF」时才炸出「未安装 ezdxf」。便携包里明明装了
ezdxf，用户却完全摸不着头脑。

本模块用组码（group code）直接解析 ASCII DXF，输出与
``dxf_writer.parse_dxf_to_entities`` 完全一致的图元字典，支持：

    LINE / LWPOLYLINE / POLYLINE(+VERTEX) / CIRCLE / ARC /
    SPLINE / ELLIPSE / TEXT / MTEXT

约定与 ezdxf 路线保持一致：
  · ARC 保留 ``arc`` 类型（同时给 points 细分点），角度已归一化到 0~360
  · SPLINE 保留 ``spline`` 类型，按控制点 + 节点向量用 de Boor 采样成 points
  · ELLIPSE 交给 ``curves.ellipse_entity`` 生成（与写出一致的参数化）
  · 图层名 / 线型名原样保留，带参虚线名（DASHED10_4）反解出 dashLen/dashGap
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["parse_dxf", "parse_dxf_text", "looks_like_binary"]


# --------------------------------------------------------------------------
# 解码 / 预处理
# --------------------------------------------------------------------------
def looks_like_binary(raw: bytes) -> bool:
    """二进制 DXF（AutoCAD Binary DXF 或 AC10xx 版本号开头）。"""
    head = raw[:32]
    return head.startswith(b"AutoCAD Binary DXF") or head[:2] == b"AC"


def _decode(raw: bytes) -> str:
    """尽量把 DXF 解成文本。国产软件常写 GBK，老图纸可能是 cp1252。"""
    if looks_like_binary(raw):
        raise ValueError("这是二进制 DXF（Binary DXF），内置读取器只支持 ASCII DXF；"
                         "请另存为 ASCII DXF 后重试")
    for enc in ("utf-8-sig", "gbk", "cp1252"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _pairs(text: str) -> List[Tuple[int, str]]:
    """DXF 是「组码 / 值」严格成对的文本。容错：跳过不成对与非法组码。"""
    lines = text.splitlines()
    out: List[Tuple[int, str]] = []
    i = 0
    n = len(lines)
    while i < n - 1:
        code_s = lines[i].strip()
        if not code_s:
            i += 1
            continue
        try:
            code = int(code_s)
        except ValueError:
            i += 1
            continue
        out.append((code, lines[i + 1].rstrip("\r")))
        i += 2
    return out


# --------------------------------------------------------------------------
# 实体分组
# --------------------------------------------------------------------------
def _split_entities(pairs: Sequence[Tuple[int, str]]) -> List[Tuple[str, List[Tuple[int, str]]]]:
    """只取 ENTITIES 段，按组码 0 切成 (类型, 组码列表)。"""
    out: List[Tuple[str, List[Tuple[int, str]]]] = []
    in_entities = False
    section = ""
    cur_type: Optional[str] = None
    cur: List[Tuple[int, str]] = []
    for code, val in pairs:
        if code == 0:
            tag = val.strip().upper()
            if tag == "SECTION":
                if cur_type:
                    out.append((cur_type, cur))
                cur_type, cur = None, []
                section = ""
                continue
            if tag == "ENDSEC":
                if cur_type:
                    out.append((cur_type, cur))
                cur_type, cur = None, []
                in_entities = False
                continue
            if in_entities:
                if cur_type:
                    out.append((cur_type, cur))
                cur_type, cur = tag, []
                continue
            # 段外的 0 标记（如 EOF）直接忽略
            cur_type, cur = None, []
            continue
        if code == 2 and not cur:
            section = val.strip().upper()
            if section == "ENTITIES":
                in_entities = True
            continue
        if in_entities and cur_type is not None:
            cur.append((code, val))
    if cur_type:
        out.append((cur_type, cur))
    return out


# --------------------------------------------------------------------------
# 组码取值小工具
# --------------------------------------------------------------------------
def _f(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _first(g: Sequence[Tuple[int, str]], code: int, default: Optional[str] = None) -> Optional[str]:
    for c, v in g:
        if c == code:
            return v
    return default


def _all(g: Sequence[Tuple[int, str]], code: int) -> List[str]:
    return [v for c, v in g if c == code]


def _xy(g: Sequence[Tuple[int, str]], ix: int, iy: int) -> Tuple[float, float]:
    return (_f(_first(g, ix) or "0"), _f(_first(g, iy) or "0"))


def _vert2d(g: Sequence[Tuple[int, str]]) -> List[Tuple[float, float]]:
    """LWPOLYLINE 的顶点：10/20 成串出现，按出现顺序配对。"""
    xs = [_f(v) for v in _all(g, 10)]
    ys = [_f(v) for v in _all(g, 20)]
    n = min(len(xs), len(ys))
    return [(xs[i], ys[i]) for i in range(n)]


# --------------------------------------------------------------------------
# B 样条采样（de Boor）
# --------------------------------------------------------------------------
def _de_boor(u: float, ctrl: Sequence[Tuple[float, float]],
             knots: Sequence[float], p: int) -> Tuple[float, float]:
    n = len(ctrl) - 1
    k = p
    for i in range(p, n + 1):
        if knots[i] <= u < knots[i + 1]:
            k = i
            break
    else:
        k = n if u >= knots[n + 1] else p
    d = [list(ctrl[j + k - p]) for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            den = knots[j + 1 + k - r] - knots[j + k - p]
            a = 0.0 if den <= 1e-12 else (u - knots[j + k - p]) / den
            d[j][0] = (1.0 - a) * d[j - 1][0] + a * d[j][0]
            d[j][1] = (1.0 - a) * d[j - 1][1] + a * d[j][1]
    return (d[p][0], d[p][1])


def _sample_bspline(ctrl: Sequence[Tuple[float, float]], knots: Sequence[float],
                    degree: int, closed: bool) -> List[List[float]]:
    n = len(ctrl) - 1
    p = max(1, min(int(degree), n))
    if len(knots) != n + p + 2:                       # 节点向量不合法 → 退化为控制多边形
        return [[round(x, 4), round(y, 4)] for x, y in ctrl]
    u0, u1 = knots[p], knots[n + 1]
    if u1 <= u0:
        return [[round(x, 4), round(y, 4)] for x, y in ctrl]
    seg = max(48, min(400, n * 12))
    pts: List[List[float]] = []
    for i in range(seg + 1):
        u = u0 + (u1 - u0) * i / seg
        x, y = _de_boor(u, ctrl, knots, p)
        pts.append([round(x, 4), round(y, 4)])
    if closed and pts and (pts[0] != pts[-1]):
        pts.append(list(pts[0]))
    return pts


# --------------------------------------------------------------------------
# 单实体 → 编辑器图元
# --------------------------------------------------------------------------
def _common(g: Sequence[Tuple[int, str]]) -> Dict[str, Any]:
    layer = str(_first(g, 8, "0") or "0")
    lt = str(_first(g, 6, "") or "")
    dash = lt.upper() not in ("", "CONTINUOUS", "BYLAYER", "BYBLOCK")
    return {"layer": layer, "dash": dash, "lt": lt}


def _entity_from(type_name: str, g: Sequence[Tuple[int, str]],
                 verts: Optional[List[Tuple[float, float]]] = None) -> Optional[Dict[str, Any]]:
    base = _common(g)

    if type_name == "LINE":
        p1 = _xy(g, 10, 20)
        p2 = _xy(g, 11, 21)
        return {"type": "line", "p1": [p1[0], p1[1]], "p2": [p2[0], p2[1]],
                "layer": base["layer"], "dash": base["dash"]}

    if type_name == "LWPOLYLINE":
        pts = _vert2d(g)
        closed = bool(int(_f(_first(g, 70) or "0")) & 1)
        if len(pts) >= 2:
            return {"type": "polyline", "points": [[x, y] for x, y in pts],
                    "closed": closed, "layer": base["layer"], "dash": base["dash"]}
        return None

    if type_name == "POLYLINE":
        pts = verts or []
        flags = int(_f(_first(g, 70) or "0"))
        if len(pts) >= 2:
            return {"type": "polyline", "points": [[x, y] for x, y in pts],
                    "closed": bool(flags & 1), "layer": base["layer"], "dash": base["dash"]}
        return None

    if type_name == "CIRCLE":
        c = _xy(g, 10, 20)
        r = _f(_first(g, 40) or "0")
        if r > 0:
            return {"type": "circle", "center": [c[0], c[1]], "radius": r,
                    "layer": base["layer"], "dash": base["dash"]}
        return None

    if type_name == "ARC":
        c = _xy(g, 10, 20)
        r = _f(_first(g, 40) or "0")
        a0 = _f(_first(g, 50) or "0")
        a1 = _f(_first(g, 51) or "0")
        if r <= 0:
            return None
        if a1 <= a0:
            a1 += 360.0
        seg = max(16, int((a1 - a0) / 4) + 1)
        pts = [[round(c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * k / seg)), 4),
                round(c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * k / seg)), 4)]
               for k in range(seg + 1)]
        return {"type": "arc", "center": [c[0], c[1]], "radius": r,
                "start": a0 % 360.0, "end": a1 % 360.0, "points": pts,
                "layer": base["layer"], "dash": base["dash"]}

    if type_name == "SPLINE":
        flags = int(_f(_first(g, 70) or "0"))
        closed = bool(flags & 1)
        degree = int(_f(_first(g, 71) or "3"))
        xs = [_f(v) for v in _all(g, 10)]
        ys = [_f(v) for v in _all(g, 20)]
        ctrl = list(zip(xs, ys))
        knots = [_f(v) for v in _all(g, 40)]
        if len(ctrl) >= 2 and len(knots) == len(ctrl) + degree + 1:
            pts = _sample_bspline(ctrl, knots, degree, closed)
        else:
            # 只有拟合点（11/21）或节点不全：两种都退化成穿过这些点的折线，
            # 至少形状是对的（导出时仍按 spline 写出）。
            fx = [_f(v) for v in _all(g, 11)]
            fy = [_f(v) for v in _all(g, 21)]
            fit = [[round(x, 4), round(y, 4)] for x, y in zip(fx, fy)]
            pts = fit if len(fit) >= 2 else [[round(x, 4), round(y, 4)] for x, y in ctrl]
        if len(pts) >= 2:
            return {"type": "spline", "points": pts, "closed": closed,
                    "layer": base["layer"], "dash": base["dash"]}
        return None

    if type_name == "ELLIPSE":
        c = _xy(g, 10, 20)
        mx, my = _xy(g, 11, 21)
        rx = math.hypot(mx, my)
        ratio = _f(_first(g, 40) or "1") or 1.0
        rot = math.degrees(math.atan2(my, mx))
        start = math.degrees(_f(_first(g, 41) or "0"))
        end = math.degrees(_f(_first(g, 42) or ("%.17g" % (2 * math.pi))))
        if rx <= 0:
            return None
        try:
            from curves import ellipse_entity
            ent = ellipse_entity([c[0], c[1]], rx, rx * ratio, rot,
                                 base["layer"], start, end)
            ent["dash"] = base["dash"]
            return ent
        except Exception:                              # noqa: BLE001 - curves 不可用就手工采样
            pass
        a = math.radians(rot)
        ux, uy = math.cos(a) * rx, math.sin(a) * rx
        vx, vy = -math.sin(a) * rx * ratio, math.cos(a) * rx * ratio
        span = (end - start) % 360.0 or 360.0
        pts = [[round(c[0] + math.cos(math.radians(start + span * k / 64)) * ux
                      + math.sin(math.radians(start + span * k / 64)) * vx, 4),
                round(c[1] + math.cos(math.radians(start + span * k / 64)) * uy
                      + math.sin(math.radians(start + span * k / 64)) * vy, 4)]
               for k in range(65)]
        return {"type": "ellipse", "center": [c[0], c[1]], "rx": rx, "ry": rx * ratio,
                "rotation": rot, "start": start, "end": end, "points": pts,
                "layer": base["layer"], "dash": base["dash"]}

    if type_name == "MTEXT":
        chunks = _all(g, 3)                            # 长文本会被切成 3 组码
        tail = _first(g, 1, "") or ""
        text = "".join(chunks) + tail
        p = _xy(g, 10, 20)
        h = _f(_first(g, 40) or "5") or 5.0
        return {"type": "text", "p": [p[0], p[1]], "height": h,
                "value": _clean_text(text), "layer": base["layer"]}

    if type_name == "TEXT":
        p = _xy(g, 10, 20)
        h = _f(_first(g, 40) or "5") or 5.0
        return {"type": "text", "p": [p[0], p[1]], "height": h,
                "value": _clean_text(_first(g, 1, "") or ""), "layer": base["layer"]}

    return None


def _clean_text(s: str) -> str:
    """MTEXT 里的格式码（\\P 换行、{\\f...;} 字体等）粗清一下，留可读文本。"""
    if "\\" not in s and "{" not in s:
        return s
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in ("P", "p"):
                out.append("\n")
                i += 2
                continue
            if nxt in ("\\", "{", "}"):
                out.append(nxt)
                i += 2
                continue
            # \fArial|b0; \H2.5x; \C1; 等 → 跳到分号
            j = s.find(";", i)
            if j == -1:
                i += 2
                continue
            i = j + 1
            continue
        if ch in "{}":
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out).strip()


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def parse_dxf_text(text: str) -> List[Dict[str, Any]]:
    """解析 DXF 文本，返回图元字典列表（与 ezdxf 路线同构）。"""
    ents = _split_entities(_pairs(text))
    out: List[Dict[str, Any]] = []
    i = 0
    n = len(ents)
    while i < n:
        name, g = ents[i]
        if name == "POLYLINE":
            verts: List[Tuple[float, float]] = []
            j = i + 1
            while j < n and ents[j][0] == "VERTEX":
                vx = _first(ents[j][1], 10)
                vy = _first(ents[j][1], 20)
                if vx is not None and vy is not None:
                    verts.append((_f(vx), _f(vy)))
                j += 1
            if j < n and ents[j][0] == "SEQEND":
                j += 1
            ent = _entity_from("POLYLINE", g, verts)
            if ent:
                _apply_dash_params(ent, _common(g)["lt"])
                out.append(ent)
            i = j
            continue
        if name in ("VERTEX", "SEQEND", "ATTRIB"):
            i += 1
            continue
        ent = _entity_from(name, g)
        if ent:
            _apply_dash_params(ent, _common(g)["lt"])
            out.append(ent)
        i += 1
    return out


def _apply_dash_params(ent: Dict[str, Any], lt: str) -> None:
    """带参线型名（DASHED10_4）→ 恢复段长/间隔，编辑器才能显示原始间隔。"""
    if not ent.get("dash") or not lt:
        return
    try:
        from dxf_writer import DEFAULT_DASH, parse_dash_name
        pat = parse_dash_name(lt)
        if pat and pat != DEFAULT_DASH:
            ent["dashLen"], ent["dashGap"] = pat
    except Exception:                                  # noqa: BLE001
        pass


def parse_dxf(path: str) -> List[Dict[str, Any]]:
    """读取 DXF 文件（ASCII），不需要 ezdxf。"""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw:
        raise ValueError("文件是空的")
    return parse_dxf_text(_decode(raw))
