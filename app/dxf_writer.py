"""
dxf_writer.py — 纯标准库 DXF (R2000 / AC1015) 生成器
零依赖、可在完全离线环境下运行。LibreCAD / AutoCAD / ezdxf 均可读取。

支持的图元类型：
  line      线段
  circle    圆
  arc       圆弧
  rect      矩形（轴对齐，转为闭合多段线）
  polyline  多段线（可闭合）
  spline    三次 B 样条（真曲线，过给定点；闭合样条无接缝）
  ellipse   椭圆 / 椭圆弧
  text      文字
  point     点

坐标单位：毫米(mm)。原点在左下角，Y 轴向上。

已按 DXF R2000 规范补全：实体子类标记 (100,AcDb*) 与唯一句柄 (5)，
可通过 ezdxf / AutoCAD 严格解析。
"""

from __future__ import annotations
import math
import re
import sys
from typing import Any, Dict, List, Sequence, Tuple


# ---------------------------------------------------------------- 文字样式
# 文字样式名。这不是随便起的：LibreCAD 读 DXF 时**把样式名当字体名**去
# resources/fonts/ 里找 <样式名>.lff，找不到就回落到 standard（纯 ASCII 字形）。
# 用户名一进 LibreCAD，Φ 直径符号和中文标注就全变成小方框——所以样式名必须
# 等于包内自带的中文字体文件名。noto-sans-sc 是 LibreCAD 随包的简体中文字体。
TEXT_STYLE = "noto-sans-sc"
# 同一份 DXF 还要给 AutoCAD / 浩辰 / CAXA 看，它们不看样式名、只看 font 字段，
# 所以这里写系统自带的宋体文件名（simsun.ttc），中文一样能显示。
TEXT_STYLE_FONT = "simsun.ttc"


def _fmt(v: float) -> str:
    """按 DXF 精度格式化浮点数，去除多余尾零。"""
    s = f"{float(v):.6f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _num(v, default: float = 0.0) -> float:
    """宽松数值转换：字符串/None/坏值都不再抛异常。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _pt2(v, default=(0.0, 0.0)):
    """宽松取二维点：接受 [x,y] / [x,y,z]（忽略z） / {"x":..,"y":..} / 标量。"""
    if isinstance(v, dict):
        return _num(v.get("x")), _num(v.get("y"))
    if isinstance(v, (list, tuple)):
        if len(v) >= 2:
            return _num(v[0]), _num(v[1])
        if len(v) == 1:
            return _num(v[0]), 0.0
    return default


def _bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "是")
    if v is None:
        return default
    return bool(v)


# 模型可能输出的图元类型别名（中英文都认）
_TYPE_ALIASES = {
    "line": "line", "线段": "line", "直线": "line", "linesegment": "line",
    "circle": "circle", "圆": "circle", "圆形": "circle",
    "arc": "arc", "圆弧": "arc",
    "rect": "rect", "rectangle": "rect", "矩形": "rect", "方形": "rect",
    "polyline": "polyline", "多段线": "polyline", "多边形": "polyline",
    "spline": "spline", "bspline": "spline", "nurbs": "spline",
    "curve": "spline", "bezier": "spline", "曲线": "spline",
    "样条": "spline", "样条曲线": "spline", "自由曲线": "spline",
    "ellipse": "ellipse", "oval": "ellipse", "椭圆": "ellipse", "椭圆形": "ellipse",
    "text": "text", "文字": "text", "标注": "text", "label": "text",
    "point": "point", "点": "point",
}


# ---------------------------------------------------------------- 虚线线型
# 默认虚线（毫米级）：5mm 实 + 3mm 空。图案随图纸大小自定义时，
# 线型名带参数（如 DASHED10_4 = 10mm 实 4mm 空），一份 DXF 里可共存多种图案。
DEFAULT_DASH = (5.0, 3.0)


def dash_ltype_name(dash: float, gap: float) -> str:
    """按段长/间隔生成线型名；默认参数直接用标准名 DASHED。"""
    def f(v: float) -> str:
        s = f"{float(v):.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    d, g = f(dash), f(gap)
    if (d, g) == ("5", "3"):
        return "DASHED"
    return f"DASHED{d}_{g}"


def parse_dash_name(name: str):
    """从线型名反解 (段长, 间隔)；不是带参虚线名则返回 None。"""
    n = str(name or "").strip()
    if n == "DASHED":
        return DEFAULT_DASH
    m = re.match(r"^DASHED(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$", n)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


def is_arrowable(ent: Dict[str, Any]) -> bool:
    """箭头只加在开放线条的端点上：线段 / 开多段线 / 开样条。"""
    if not isinstance(ent, dict):
        return False
    t = str(ent.get("type") or ent.get("kind") or "").strip().lower()
    t = _TYPE_ALIASES.get(t, t)
    if t not in ("line", "polyline", "spline"):
        return False
    if t == "line":
        return True
    return not _bool(ent.get("closed", ent.get("close")))


def arrow_triangles(ent: Dict[str, Any]):
    """计算箭头三角形（世界坐标，毫米）。

    图元字段：arrow = 0 无 / 1 起点 / 2 终点 / 3 两端；arrowSize = 箭头长度 mm。
    返回 [(tip, base1, base2), ...]，tip 在线条端点上，三角形指向行进方向的反向
    （即箭头"尖"顶在端点，两翼向线条内侧张开）。
    """
    if not isinstance(ent, dict):
        return []
    t = str(ent.get("type") or ent.get("kind") or "").strip().lower()
    t = _TYPE_ALIASES.get(t, t)
    arrow = int(_num(ent.get("arrow"), 0.0)) or 0
    arrow &= 3
    if not arrow or not is_arrowable(ent):
        return []
    size = _num(ent.get("arrowSize"), 4.0)
    if size <= 0:
        return []
    if t == "line":
        pts = [_pt2(ent.get("p1", ent.get("start", ent.get("from")))),
               _pt2(ent.get("p2", ent.get("end", ent.get("to"))))]
    elif t == "polyline":
        pts = [_pt2(p) for p in (ent.get("points") or ent.get("vertices") or [])
               if isinstance(p, (list, tuple, dict))]
    else:  # spline：取拟合采样点
        pts = [_pt2(p) for p in (ent.get("points") or ent.get("fit_points")
                                 or ent.get("vertices") or [])
               if isinstance(p, (list, tuple, dict))]
    if len(pts) < 2:
        return []
    ends = []
    if arrow & 1:                          # 起点：方向 = 第2点 → 第1点
        ends.append((pts[0], pts[1]))
    if arrow & 2:                          # 终点：方向 = 倒数第2点 → 最后1点
        ends.append((pts[-1], pts[-2]))
    half = size * 0.3                      # 半翼宽 ≈ 0.3×长（半角约 17°）
    tris = []
    for tip, prev in ends:
        dx, dy = tip[0] - prev[0], tip[1] - prev[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ux, uy = dx / L, dy / L            # 指向端点外侧
        px, py = -uy, ux                   # 垂直方向
        bx, by = tip[0] - ux * size, tip[1] - uy * size   # 底边中点
        tris.append(((tip[0], tip[1]),
                     (bx + px * half, by + py * half),
                     (bx - px * half, by - py * half)))
    return tris


class DXF:
    def __init__(self, unit: str = "mm") -> None:
        self.entities: List[List[Tuple[int, Any]]] = []
        self.unit = unit
        self.skipped = 0  # 因数据异常被跳过的图元数
        self._handle = 100  # 实体句柄从 100 起，避免与表记录(1)冲突
        self._dash_patterns: Dict[str, Tuple[float, float]] = {}  # 线型名 → (段长, 间隔)

    def _h(self) -> str:
        h = self._handle
        self._handle += 1
        return f"{h:X}"

    # ---------- 低级接口 ----------
    def _add(self, groups: List[Tuple[int, Any]]) -> None:
        self.entities.append(groups)

    def _lt_group(self, linetype: str):
        """线型组码（6）：实线(CONTINUOUS)省略，保持旧文件兼容；虚线名写入并登记。"""
        if not linetype or linetype == "CONTINUOUS":
            return []
        name = str(linetype)
        if name not in self._dash_patterns:      # 直接调方法没登记图案时兜底
            self._dash_patterns[name] = parse_dash_name(name) or DEFAULT_DASH
        return [(6, name)]

    def set_dash(self, name: str, dash: float, gap: float) -> str:
        """登记一个带参虚线图案（段长 dash mm / 间隔 gap mm），返回规范化线型名。"""
        name = name or dash_ltype_name(dash, gap)
        self._dash_patterns[name] = (max(0.1, float(dash)), max(0.1, float(gap)))
        return name

    def solid(self, pts3: Sequence[Sequence[float]], layer: str = "0",
              linetype: str = "CONTINUOUS") -> None:
        """实心三角形（SOLID 实体，第 4 顶点复用第 3 个）——画箭头用。"""
        p = [tuple(q) for q in pts3]
        while len(p) < 3:
            p.append(p[-1])
        self._add([
            (0, "SOLID"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
        ] + self._lt_group(linetype) + [
            (100, "AcDbTrace"),
            (10, _fmt(p[0][0])), (20, _fmt(p[0][1])), (30, "0"),
            (11, _fmt(p[1][0])), (21, _fmt(p[1][1])), (31, "0"),
            (12, _fmt(p[2][0])), (22, _fmt(p[2][1])), (32, "0"),
            (13, _fmt(p[2][0])), (23, _fmt(p[2][1])), (33, "0"),
        ])

    def line(self, p1: Sequence[float], p2: Sequence[float], layer: str = "0",
             linetype: str = "CONTINUOUS") -> None:
        self._add([
            (0, "LINE"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
        ] + self._lt_group(linetype) + [
            (100, "AcDbLine"),
            (10, _fmt(p1[0])), (20, _fmt(p1[1])), (30, "0"),
            (11, _fmt(p2[0])), (21, _fmt(p2[1])), (31, "0"),
        ])

    def circle(self, center: Sequence[float], radius: float, layer: str = "0",
                linetype: str = "CONTINUOUS") -> None:
        self._add([
            (0, "CIRCLE"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
        ] + self._lt_group(linetype) + [
            (100, "AcDbCircle"),
            (10, _fmt(center[0])), (20, _fmt(center[1])), (30, "0"),
            (40, _fmt(radius)),
        ])

    def arc(self, center: Sequence[float], radius: float,
            start_deg: float, end_deg: float, layer: str = "0",
            linetype: str = "CONTINUOUS") -> None:
        self._add([
            (0, "ARC"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
        ] + self._lt_group(linetype) + [
            (100, "AcDbCircle"), (100, "AcDbArc"),
            (10, _fmt(center[0])), (20, _fmt(center[1])), (30, "0"),
            (40, _fmt(radius)),
            (50, _fmt(start_deg)), (51, _fmt(end_deg)),
        ])

    def polyline(self, points: Sequence[Sequence[float]],
                 closed: bool = False, layer: str = "0",
                 linetype: str = "CONTINUOUS") -> None:
        pts = list(points)
        flags = 1 if closed else 0
        groups: List[Tuple[int, Any]] = [
            (0, "LWPOLYLINE"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
        ] + self._lt_group(linetype) + [
            (100, "AcDbPolyline"),
            (90, len(pts)), (70, flags),
        ]
        for x, y in pts:
            groups += [(10, _fmt(x)), (20, _fmt(y))]
        self._add(groups)

    def spline(self, points: Sequence[Sequence[float]], closed: bool = False,
               layer: str = "0", linetype: str = "CONTINUOUS") -> bool:
        """拟合一条过给定点的三次 B 样条并写成 SPLINE 实体（真曲线）。

        控制点/节点向量由 curves.py 反解（开曲线 A9.1 全局插值、闭曲线周期
        逆平滑）。点数不足 3 时退回多段线，保证不丢图元。
        """
        from curves import spline_definition                   # 延迟导入，避免循环
        fit = spline_definition({"points": [list(p) for p in points],
                                 "closed": bool(closed)})
        ctrl = fit.get("ctrl") or []
        knots = fit.get("knots") or []
        degree = int(fit.get("degree") or 3)
        if len(ctrl) < 2 or len(knots) != len(ctrl) + degree + 1:
            if len(points) >= 2:                               # 兜底：退化成多段线
                self.polyline(points, closed=closed, layer=layer, linetype=linetype)
                return False
            return False
        flags = 8 | (1 if (closed or fit.get("closed")) else 0)   # 8=平面, 1=闭合
        groups: List[Tuple[int, Any]] = [
            (0, "SPLINE"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
        ] + self._lt_group(linetype) + [
            (100, "AcDbSpline"),
            (70, flags), (71, degree), (72, len(knots)),
            (73, len(ctrl)), (74, 0),
        ]
        for k in knots:
            groups.append((40, _fmt(k)))
        for (x, y) in ctrl:
            groups += [(10, _fmt(x)), (20, _fmt(y)), (30, "0")]
        self._add(groups)
        return True

    def ellipse(self, center: Sequence[float], rx: float, ry: float,
                rotation: float = 0.0, start: float = 0.0, end: float = 360.0,
                layer: str = "0", linetype: str = "CONTINUOUS") -> None:
        """椭圆 / 椭圆弧。DXF 要求 ratio=短轴/长轴 ∈ (0,1]，故 ry>rx 时交换主轴。"""
        rx, ry = abs(float(rx)), abs(float(ry))
        rot = float(rotation)
        if rx <= 0 or ry <= 0:
            return
        if ry > rx:                       # 让"长轴"真的是长的那根
            rx, ry = ry, rx
            rot += 90.0
        ratio = max(1e-6, min(1.0, ry / rx))
        a = math.radians(rot)
        mx, my = rx * math.cos(a), rx * math.sin(a)
        s, e = float(start), float(end)
        while e <= s:
            e += 360.0
        self._add([
            (0, "ELLIPSE"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
        ] + self._lt_group(linetype) + [
            (100, "AcDbEllipse"),
            (10, _fmt(center[0])), (20, _fmt(center[1])), (30, "0"),
            (11, _fmt(mx)), (21, _fmt(my)), (31, "0"),
            (40, _fmt(ratio)),
            (41, _fmt(math.radians(s))), (42, _fmt(math.radians(e))),
        ])

    def rect(self, x: float, y: float, w: float, h: float,
             layer: str = "0", linetype: str = "CONTINUOUS") -> None:
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        self.polyline(pts, closed=True, layer=layer, linetype=linetype)

    def text(self, p: Sequence[float], height: float, value: str,
             rotation: float = 0.0, layer: str = "0") -> None:
        # 组码写全（41/51/72/11/21/31/73 + 末尾第二个 100 AcDbText）：
        # 少写这些时 ezdxf 会补默认值，但 LibreCAD 的解析器对残缺 TEXT 更挑，
        # 容易把"没有样式"的文字直接吞掉不画。
        h = max(0.1, float(height or 5.0))
        self._add([
            (0, "TEXT"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
            (100, "AcDbText"),
            (10, _fmt(p[0])), (20, _fmt(p[1])), (30, "0"),
            (40, _fmt(h)), (1, value), (50, _fmt(rotation)),
            (41, "1"), (51, "0"), (7, TEXT_STYLE), (71, "0"), (72, "0"),
            (11, _fmt(p[0])), (21, _fmt(p[1])), (31, "0"),
            (100, "AcDbText"), (73, "0"),
        ])

    def point(self, p: Sequence[float], layer: str = "0") -> None:
        self._add([
            (0, "POINT"), (5, self._h()), (100, "AcDbEntity"), (8, layer),
            (100, "AcDbPoint"),
            (10, _fmt(p[0])), (20, _fmt(p[1])), (30, "0"),
        ])

    # ---------- 图层 ----------
    # AutoCAD 颜色索引(ACI)调色板：图层名若能解析为数字，按序取色，
    # 让 1/2/3… 各层在 LibreCAD 里颜色分明、便于区分。
    _LAYER_COLORS = [1, 2, 3, 4, 5, 6, 30, 210, 40, 170, 150, 220, 8, 50, 130, 200]

    def _layer_color(self, name: str) -> int:
        try:
            n = int(str(name))
        except (TypeError, ValueError):
            return 7  # 非数字层名 → 默认白
        if n < 1:
            return 7
        return self._LAYER_COLORS[(n - 1) % len(self._LAYER_COLORS)]

    def flatten(self, target: str = "0") -> None:
        """把所有实体的图层统一改写到 target（用于"合成进一个图层"）。"""
        for g in self.entities:
            for i, (code, val) in enumerate(g):
                if code == 8:
                    g[i] = (8, target)
                    break

    def layer_names(self) -> List[str]:
        """返回所有被使用到的图层名（含 '0'）。"""
        names = []
        seen = set()
        for g in self.entities:
            for code, val in g:
                if code == 8:
                    v = str(val)
                    if v not in seen:
                        seen.add(v)
                        names.append(v)
                    break
        return names

    # ---------- 从模型输出的实体字典构建（宽松容错版） ----------
    def add_entity(self, ent: Dict[str, Any]) -> bool:
        """把一个图元字典写入 DXF。

        对模型输出的常见不规范性全面容错：
          - 字段别名：p1/p2、start/end、from/to、center/c、radius/r、value/text...
          - 三维坐标 [x,y,z]（自动忽略 z）
          - 字符串数字、缺 layer、rect 用 p1/p2 或 x1/y1/x2/y2 表示
          - 中英文图元类型名（"矩形"/"圆"/"line"...）
          - 单个图元数据损坏只跳过该图元，不影响整张图纸
        返回 True 表示成功写入。
        """
        if not isinstance(ent, dict):
            self.skipped += 1
            return False
        t = str(ent.get("type") or ent.get("kind") or "").strip().lower()
        t = _TYPE_ALIASES.get(t, t)
        layer = str(ent.get("layer") or "0")
        linetype = ent.get("linetype")
        if not linetype:
            if _bool(ent.get("dash")):
                linetype = self.set_dash(dash_ltype_name(_num(ent.get("dashLen"), 5.0),
                                                         _num(ent.get("dashGap"), 3.0)),
                                         _num(ent.get("dashLen"), 5.0),
                                         _num(ent.get("dashGap"), 3.0))
            else:
                linetype = "CONTINUOUS"
        try:
            if t == "line":
                p1 = _pt2(ent.get("p1", ent.get("start", ent.get("from", ent.get("a")))))
                p2 = _pt2(ent.get("p2", ent.get("end", ent.get("to", ent.get("b")))))
                self.line(p1, p2, layer, linetype)
            elif t == "circle":
                center = _pt2(ent.get("center", ent.get("c", ent.get("p"))))
                r = ent.get("radius", ent.get("r"))
                if r is None and ent.get("diameter") is not None:
                    r = _num(ent.get("diameter")) / 2.0
                if r is None:
                    raise KeyError("缺少 radius")
                self.circle(center, _num(r), layer, linetype)
            elif t == "arc":
                center = _pt2(ent.get("center", ent.get("c")))
                r = ent.get("radius", ent.get("r"))
                if r is None and ent.get("diameter") is not None:
                    r = _num(ent.get("diameter")) / 2.0
                start = ent.get("start", ent.get("start_angle",
                               ent.get("startAngle", ent.get("begin"))))
                end = ent.get("end", ent.get("end_angle",
                             ent.get("endAngle", ent.get("finish"))))
                if r is None or start is None or end is None:
                    raise KeyError("arc 参数不全")
                s, e = _num(start), _num(end)
                while e <= s:          # DXF 圆弧逆时针，保证 end > start
                    e += 360.0
                self.arc(center, _num(r), s, e, layer, linetype)
            elif t == "rect":
                if all(k in ent for k in ("x", "y", "w", "h")):
                    x, y = _num(ent["x"]), _num(ent["y"])
                    w, h = _num(ent["w"]), _num(ent["h"])
                elif "p1" in ent and "p2" in ent:
                    x1, y1 = _pt2(ent["p1"])
                    x2, y2 = _pt2(ent["p2"])
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                elif all(k in ent for k in ("x1", "y1", "x2", "y2")):
                    x1, y1 = _num(ent["x1"]), _num(ent["y1"])
                    x2, y2 = _num(ent["x2"]), _num(ent["y2"])
                    x, y = min(x1, x2), min(y1, y2)
                    w, h = abs(x2 - x1), abs(y2 - y1)
                else:
                    raise KeyError("rect 参数不全")
                self.rect(x, y, w, h, layer, linetype)
            elif t == "polyline":
                raw = ent.get("points") or ent.get("vertices") or ent.get("pts") or []
                pts = [_pt2(p) for p in raw if isinstance(p, (list, tuple, dict))]
                if len(pts) < 2:
                    raise ValueError("polyline 顶点不足")
                self.polyline(pts, _bool(ent.get("closed", ent.get("close"))), layer, linetype)
            elif t == "spline":
                # 曲线：points 是"曲线经过的点"（模型/配方给的采样点都算），
                # 也接受 fit_points / control_points 别名。
                raw = (ent.get("points") or ent.get("fit_points") or ent.get("fit")
                       or ent.get("vertices") or ent.get("control_points") or [])
                pts = [_pt2(p) for p in raw if isinstance(p, (list, tuple, dict))]
                closed = _bool(ent.get("closed", ent.get("close")))
                if len(pts) < 3:
                    raise ValueError("spline 控制点不足")
                self.spline(pts, closed=closed, layer=layer, linetype=linetype)
            elif t == "ellipse":
                center = _pt2(ent.get("center", ent.get("c")))
                rx = ent.get("rx", ent.get("a", ent.get("major", ent.get("radius"))))
                ry = ent.get("ry", ent.get("b", ent.get("minor", ent.get("radius2"))))
                if ry is None:
                    ry = rx
                if rx is None:
                    raise KeyError("ellipse 缺少 rx/ry")
                self.ellipse(center, _num(rx), _num(ry),
                             _num(ent.get("rotation", ent.get("rotate",
                                                           ent.get("angle"))), 0.0),
                             _num(ent.get("start", ent.get("start_angle")), 0.0),
                             _num(ent.get("end", ent.get("end_angle")), 360.0),
                             layer, linetype)
            elif t == "text":
                p = _pt2(ent.get("p", ent.get("position", ent.get("at", ent.get("point")))))
                value = ent.get("value", ent.get("text",
                                ent.get("content", ent.get("label", ""))))
                self.text(p, _num(ent.get("height"), 5.0), str(value),
                          _num(ent.get("rotation", ent.get("angle"))), layer)
            elif t == "point":
                p = _pt2(ent.get("p", ent.get("position", ent.get("at", ent.get("point")))))
                self.point(p, layer)
            else:
                print(f"[warn] 未知图元类型被忽略: {t or '(空)'}")
                self.skipped += 1
                return False
            for tri in arrow_triangles(ent):       # 箭头三角形跟在主体图元之后
                self.solid(tri, layer)
            return True
        except Exception as e:  # noqa - 单图元损坏不应毁掉整张图
            print(f"[warn] 图元数据异常被跳过: {t or '?'} ({e})")
            self.skipped += 1
            return False

    # ---------- 序列化 ----------
    def to_string(self) -> str:
        out: List[str] = []
        # HEADER
        # $DWGCODEPAGE：R2000(AC1015) 本身不带编码信息，不声明代码页时读取方
        # （LibreCAD/ezdxf）会按 cp1252 解码，中文文本就变成乱码。这里显式声明
        # 简体中文代码页 ANSI_936，save() 也按 GBK 落盘，文本标注才不会花。
        out += ["0", "SECTION", "2", "HEADER",
                "9", "$ACADVER", "1", "AC1015",
                "9", "$DWGCODEPAGE", "3", "ANSI_936",
                "9", "$INSUNITS",
                "70", "4" if self.unit == "mm" else "1",
                "0", "ENDSEC"]
        # TABLES (LAYER) —— 自动登记所有被使用到的图层，否则 LibreCAD 里
        # 非 0 层虽然能渲染，但图层表里查不到、颜色也不可控。
        used = self.layer_names()
        if "0" not in used:
            used = ["0"] + used
        out += ["0", "SECTION", "2", "TABLES",
                "0", "TABLE", "2", "LAYER", "70", str(len(used)), "100", "AcDbLayerTable"]
        for idx, name in enumerate(used):
            color = 7 if name == "0" else self._layer_color(name)
            out += ["0", "LAYER", "5", self._h(), "100", "AcDbLayerTableRecord",
                    "2", name, "70", "0", "62", str(color), "6", "CONTINUOUS"]
        out += ["0", "ENDTAB"]
        # LTYPE 表：CONTINUOUS 是实线默认（LAYER 记录里引用的就是它，没有这张表
        # LibreDWG 转 DWG 会报 "Unknown LAYER.ltype CONTINUOUS"）；带参虚线
        # （DASHED / DASHED10_4 …）按实际用到的图案逐一生成，毫米级，
        # 避免英寸级默认在毫米图纸里几乎连成实线。
        lts = dict(self._dash_patterns)            # 线型名 → (段长, 间隔)
        lts.setdefault("CONTINUOUS", None)         # 实线永远要有
        out += ["0", "TABLE", "2", "LTYPE", "70", str(len(lts)),
                "100", "AcDbSymbolTable"]
        for name, pat in lts.items():
            if name == "CONTINUOUS":
                out += ["0", "LTYPE", "5", self._h(), "100", "AcDbSymbolTableRecord",
                        "100", "AcDbLinetypeTableRecord",
                        "2", "CONTINUOUS", "70", "0", "3", "Solid line", "72", "65",
                        "73", "0", "40", "0.0"]
            else:
                d, g = pat if pat else (parse_dash_name(name) or DEFAULT_DASH)
                out += ["0", "LTYPE", "5", self._h(), "100", "AcDbSymbolTableRecord",
                        "100", "AcDbLinetypeTableRecord",
                        "2", name, "70", "0", "3",
                        "Dashed __ __ __ __ __ __ __ __ __ __ __ __ __ __ __",
                        "72", "65", "73", "2",
                        "40", _fmt(d + g),
                        "49", _fmt(d), "74", "0", "49", _fmt(-g), "74", "0"]
        out += ["0", "ENDTAB"]
        # STYLE 表：没有它，LibreCAD 只能用内置 standard 字体，Φ 和中文字形全缺
        # → 标注显示成一串小方框。样式名同时充当 LibreCAD 的字体名（见 TEXT_STYLE）。
        out += ["0", "TABLE", "2", "STYLE", "70", "1", "100", "AcDbSymbolTable",
                "0", "STYLE", "5", self._h(),
                "100", "AcDbSymbolTableRecord", "100", "AcDbTextStyleTableRecord",
                "2", TEXT_STYLE, "70", "0", "40", "0", "41", "1", "50", "0",
                "71", "0", "42", "2.5", "3", TEXT_STYLE_FONT,
                "0", "ENDTAB", "0", "ENDSEC"]
        # ENTITIES
        out += ["0", "SECTION", "2", "ENTITIES"]
        for groups in self.entities:
            for code, val in groups:
                out.append(str(code))
                out.append(str(val))
        out += ["0", "ENDSEC", "0", "EOF"]
        return "\n".join(out) + "\n"

    def save(self, path: str) -> None:
        # 与 HEADER 里的 $DWGCODEPAGE ANSI_936 对应：GBK 落盘，中文标注不乱码；
        # 少量 GBK 无法表示的字符（如 emoji）退化成 '?'，绝不让保存失败。
        with open(path, "w", encoding="gbk", errors="replace", newline="\r\n") as f:
            f.write(self.to_string())


def build_from_entities(entities: Sequence[Dict[str, Any]],
                        unit: str = "mm",
                        flatten_layer: str = None) -> DXF:
    d = DXF(unit=unit)
    for ent in entities:
        d.add_entity(ent)
    if flatten_layer is not None:
        d.flatten(flatten_layer)
    return d


# ----------------------------------------------------------------------------
# 反向解析：把已有 DXF 文件读成编辑器可用的图元字典（与 build_from_entities 对应）
# ----------------------------------------------------------------------------
def _parse_without_ezdxf(path: str, cause: BaseException = None) -> List[Dict[str, Any]]:
    """兜底：用内置的纯 Python 组码读取器解析（不需要 ezdxf）。

    这一步是必需的 —— 网页工作台的 numpy/cv2/ezdxf 都是懒加载，用错了解释器
    也能把界面跑起来，只有「导入 DXF」时才暴露依赖缺失。有了这条兜底，
    导入功能不再看 ezdxf 的脸色。
    """
    import dxf_reader
    try:
        return dxf_reader.parse_dxf(path)
    except Exception as e:                             # noqa: BLE001
        extra = ("；ezdxf 侧原因：%s" % cause) if cause else ""
        raise RuntimeError(
            "DXF 解析失败（%s）%s；当前 Python 解释器：%s"
            % (e, extra, sys.executable))


def parse_dxf_to_entities(path: str) -> List[Dict[str, Any]]:
    """读取一个 DXF 文件，返回编辑器可消费的图元字典列表。

    支持：LINE / LWPOLYLINE / POLYLINE / CIRCLE / ARC / SPLINE / ELLIPSE / TEXT / MTEXT。
    ARC / SPLINE / ELLIPSE 保留为曲线图元（同时带 points 细分点，编辑器直接可画，
    导出时按各自的参数写回 ARC / SPLINE / ELLIPSE）。
    POINT / INSERT(块) / DIMENSION 等暂不解析（跳过并在末尾给出跳过数）。
    图层名直接沿用 DXF 原始层名。单个图元出错只跳过该图元。

    ezdxf 优先（兼容性最好）；没装或读不动时，自动退回内置的纯 Python 读取器
    （``dxf_reader``），保证「导入 DXF」永远不因缺第三方库而报废。
    """
    ezdxf = None
    cause: BaseException = None
    try:
        import ezdxf as _ezdxf
        ezdxf = _ezdxf
    except ImportError as e:                           # 没装 ezdxf 也能导入
        cause = e
    if ezdxf is None:
        return _parse_without_ezdxf(path, cause)
    try:
        doc = ezdxf.readfile(path)
    except Exception as e:                             # noqa: BLE001
        # ezdxf 读不动的（版本怪、结构残缺）再给内置读取器一次机会
        try:
            return _parse_without_ezdxf(path, e)
        except Exception:                              # noqa: BLE001
            raise
    msp = doc.modelspace()
    out: List[Dict[str, Any]] = []
    skipped = 0
    for e in msp:
        try:
            mark = len(out)                        # 本图元写入的条目起点（SOLID 箭头会被跳过）
            t = e.dxftype()
            layer = str(getattr(e.dxf, "layer", "0") or "0")
            lt = str(getattr(e.dxf, "linetype", "") or "")
            # ezdxf 会把线型名规范化成大写（BYLAYER/BYBLOCK），必须大小写不敏感地排除
            dash = lt.upper() not in ("", "CONTINUOUS", "BYLAYER", "BYBLOCK")
            pat = parse_dash_name(lt) if dash else None   # 带参虚线名 → 段长/间隔
            if t == "LINE":
                s = e.dxf.start
                en = e.dxf.end
                out.append({"type": "line",
                            "p1": [float(s.x), float(s.y)],
                            "p2": [float(en.x), float(en.y)],
                            "layer": layer, "dash": dash})
            elif t == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in e.get_points("xy")]
                closed = bool(int(getattr(e.dxf, "flags", 0)) & 1)
                if len(pts) >= 2:
                    out.append({"type": "polyline",
                                "points": [[x, y] for x, y in pts],
                                "closed": closed, "layer": layer, "dash": dash})
            elif t == "POLYLINE":
                pts = [(float(v.dxf.location.x), float(v.dxf.location.y))
                       for v in e.vertices()]
                closed = bool(e.is_closed)
                if len(pts) >= 2:
                    out.append({"type": "polyline",
                                "points": [[x, y] for x, y in pts],
                                "closed": closed, "layer": layer, "dash": dash})
            elif t == "CIRCLE":
                c = e.dxf.center
                out.append({"type": "circle",
                            "center": [float(c.x), float(c.y)],
                            "radius": float(e.dxf.radius), "layer": layer, "dash": dash})
            elif t == "ARC":
                c = e.dxf.center
                r = float(e.dxf.radius)
                a0 = float(e.dxf.start_angle)
                a1 = float(e.dxf.end_angle)
                if a1 <= a0:
                    a1 += 360.0
                seg = max(16, int((a1 - a0) / 4) + 1)
                pts = [[round(c.x + r * math.cos(math.radians(a0 + (a1 - a0) * k / seg)), 4),
                        round(c.y + r * math.sin(math.radians(a0 + (a1 - a0) * k / seg)), 4)]
                       for k in range(seg + 1)]
                # 保留 arc 类型（同时给出细分点）：编辑器按点画，导出时再按圆心/半径/角度
                # 写回 ARC —— 这样"只画了一段"的圆弧不会在往返中被摊成多段线，
                # 也不会因为读回时变成 polyline 而丢掉"这是一段弧"的语义。
                out.append({"type": "arc", "center": [float(c.x), float(c.y)],
                            "radius": r, "start": a0 % 360.0, "end": a1 % 360.0,
                            "points": pts, "layer": layer, "dash": dash})
            elif t == "SPLINE":
                # 曲线 → 保留 spline 类型，同时给出细分点（编辑器/预览直接画）
                try:
                    # 摊平精度要细：旧写法 flattening(0.4, segments=8) 每个节点段只切
                    # 8 刀，实测大曲率样条偏差能到 4mm（编辑器里看得出折角）。
                    # 这里按 0.15mm 弦高摊平，点数过多再均匀抽稀（保首尾）。
                    pts = [[round(float(v.x), 4), round(float(v.y), 4)]
                           for v in e.flattening(0.15)]
                    if len(pts) > 3000:
                        step = len(pts) / 3000.0
                        keep = [pts[int(i * step)] for i in range(3000)]
                        if keep[-1] != pts[-1]:
                            keep.append(pts[-1])
                        pts = keep
                except Exception:                     # noqa: BLE001
                    pts = [[float(p[0]), float(p[1])] for p in e.control_points]
                if len(pts) >= 2:
                    out.append({"type": "spline", "points": pts,
                                "closed": bool(e.closed), "layer": layer, "dash": dash})
                else:
                    skipped += 1
            elif t == "ELLIPSE":
                c = e.dxf.center
                major = e.dxf.major_axis
                rx = math.hypot(float(major.x), float(major.y))
                ratio = float(getattr(e.dxf, "ratio", 1.0) or 1.0)
                rot = math.degrees(math.atan2(float(major.y), float(major.x)))
                start = math.degrees(float(getattr(e.dxf, "start_param", 0.0) or 0.0))
                end = math.degrees(float(getattr(e.dxf, "end_param", 2 * math.pi)))
                from curves import ellipse_entity
                _ell = ellipse_entity([float(c.x), float(c.y)], rx, rx * ratio,
                                      rot, layer, start, end)
                _ell["dash"] = dash
                out.append(_ell)
            elif t == "TEXT":
                p = e.dxf.insert
                out.append({"type": "text",
                            "p": [float(p.x), float(p.y)],
                            "height": float(getattr(e.dxf, "height", 5) or 5),
                            "value": str(getattr(e.dxf, "text", "") or ""),
                            "layer": layer})
            elif t == "MTEXT":
                p = e.dxf.insert
                out.append({"type": "text",
                            "p": [float(p.x), float(p.y)],
                            "height": float(getattr(e.dxf, "char_height", 5) or 5),
                            "value": str(getattr(e, "text", "") or ""),
                            "layer": layer})
            else:
                skipped += 1
            if dash and pat and pat != DEFAULT_DASH:   # 带参虚线：恢复段长/间隔
                for e2 in out[mark:]:
                    e2["dashLen"], e2["dashGap"] = pat
        except Exception:  # noqa - 单个图元异常不影响整张图
            skipped += 1
    return out


if __name__ == "__main__":
    # 自测：生成一个简单的示意图
    d = DXF()
    d.rect(0, 0, 100, 60)
    d.circle([50, 30], 20)
    d.line([0, 0], [100, 60])
    d.text([5, 65], 5, "TEST")
    d.save("sample.dxf")
    print("sample.dxf written")
