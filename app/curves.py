"""curves.py — 真曲线：把「过点的折线」变成 B 样条 / 椭圆，并给出细分采样。

为什么需要
----------
CAD 里「直线 + 圆」画不出有机形状：葫芦的腰、云朵的边、动物的背线、手柄的弯
都是自由曲线。原来的管线里，模型和参数化配方都只会输出折线（一百多个点的多段
线），放大看全是一段段直线——这就是"只会用直线和圆圈"的根源。

本模块提供三件事（只依赖 numpy，不需要 ezdxf）：

1. ``fit_curve(points, closed)``
   把「曲线经过的一组点」反解成三次 B 样条：
   - 开曲线：Piegl & Tiller《The NURBS Book》A9.1 全局插值
     （弦长参数化 + 平均法节点 + 带状方程组），曲线**精确过每一个点**；
   - 闭曲线：周期均匀 B 样条的逆平滑。周三次 B 样条在节点处满足
     S(i) = (c[i-1] + 4c[i] + c[i+1]) / 6，这是一个循环三对角方程组，
     解出控制点 c 就得到一条**无接缝**的闭合曲线（首尾切向也连续）。
2. ``sample_fit(fit)``
   把样条细分采样成点列，供前端画布 / 预览图 / 橡皮擦使用。
3. ``curveify(entities)``
   判断哪些折线其实本来就是"采样过的曲线"（顶点多、转角小、不是正多边形），
   把它们升级成 ``{"type":"spline"}``。

为什么不用"把所有折线都变样条"：五角星、正六边形、正八边形的折角是设计意图，
变样条就成了圆角。判据见 ``should_curve``。
"""

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except Exception:                                    # pragma: no cover
    np = None

Point = Tuple[float, float]

#: spline 的类型别名（模型/前端可能传中英文、大小写混写）
SPLINE_ALIASES = ("spline", "bspline", "nurbs", "curve", "bezier",
                  "曲线", "样条", "样条曲线", "自由曲线")
#: ellipse 的类型别名
ELLIPSE_ALIASES = ("ellipse", "oval", "椭圆", "椭圆形")
#: polyline 的类型别名
POLYLINE_ALIASES = ("polyline", "lwpolyline", "多段线", "折线", "多段折线")

DEFAULT_DEGREE = 3

#: 宽松曲线化（loose）的判据参数：给经典视觉（OpenCV）识别结果用。
#: 用户要求"尽可能用曲线、少用直线"——识别图里的碎弦只要能连成一条缓弯
#: 就该升级成样条，不能因为"段数不够多"而留在直线状态。单条直线、
#: 直角链（工程图的边）依然被挡在外面，精确测绘结果不会被改坏。
LOOSE_KW = {"max_turn": 62.0, "min_open": 4, "min_closed": 5,
            "sharp_ratio": 0.32}


# ------------------------------------------------------------------ 基础工具
def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _xy(p) -> Any:
    """宽松取二维点。接受 [x,y]、[x,y,z]、(x,y)、{"x":..,"y":..}。"""
    if isinstance(p, dict):
        x, y = _f(p.get("x")), _f(p.get("y"))
    elif isinstance(p, (list, tuple)) and len(p) >= 2:
        x, y = _f(p[0]), _f(p[1])
    else:
        return None
    if x is None or y is None:
        return None
    return (x, y)


def norm_points(raw: Iterable) -> List[Point]:
    out: List[Point] = []
    for p in raw or ():
        q = _xy(p)
        if q is None:
            continue
        if out and math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) < 1e-9:
            continue                                    # 去掉零长度重复点
        out.append(q)
    return out


def is_closed_pts(pts: Sequence[Point], eps: float = 1e-6) -> bool:
    if len(pts) < 4:
        return False
    return math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= eps


def _drop_closing_dup(pts: List[Point]) -> Tuple[List[Point], bool]:
    if is_closed_pts(pts):
        return pts[:-1], True
    return pts, False


def _seg_len(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def polyline_length(pts: Sequence[Point], closed: bool = False) -> float:
    n = len(pts)
    if n < 2:
        return 0.0
    L = sum(_seg_len(pts[i], pts[i + 1]) for i in range(n - 1))
    if closed:
        L += _seg_len(pts[-1], pts[0])
    return L


# ------------------------------------------------------- 形状判据（要不要变曲线）
def turn_angles(pts: Sequence[Point], closed: bool = False) -> List[float]:
    """各顶点处的转角（度，0=直着走，180=原地折返）。"""
    n = len(pts)
    if n < 3:
        return []
    out = []
    rng = range(n) if closed else range(1, n - 1)
    for i in rng:
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        n1, n2 = math.hypot(*v1), math.hypot(*v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        out.append(math.degrees(math.acos(cos)))
    return out


def is_regular_polygon(pts: Sequence[Point], closed: bool = True,
                       len_tol: float = 0.06, ang_tol: float = 6.0) -> bool:
    """顶点均匀分布 + 边长一致 + 转角一致 → 是"正多边形"，不能变样条。

    采样出来的圆/椭圆虽然转角也一致，但边长会随曲率变化不大、转角却极均匀——
    所以还必须要求**顶点数少**（≤16）才算正多边形，见 ``should_curve``。
    """
    n = len(pts)
    if n < 3:
        return False
    segs = [_seg_len(pts[i], pts[(i + 1) % n]) for i in range(n if closed else n - 1)]
    if not segs:
        return False
    mean = sum(segs) / len(segs)
    if mean <= 1e-9:
        return False
    if any(abs(s - mean) > len_tol * mean for s in segs):
        return False
    turns = turn_angles(pts, closed)
    if len(turns) < 2:
        return True
    tm = sum(turns) / len(turns)
    return all(abs(t - tm) <= ang_tol for t in turns)


def should_curve(pts: Sequence[Point], closed: bool = False,
                 max_turn: float = 50.0, min_open: int = 6, min_closed: int = 8,
                 sharp_ratio: float = 0.20) -> bool:
    """这串点"是不是一条被采样过的曲线"。

    判据（全部满足才升级成样条）：
      1) 顶点够多：开曲线 ≥6 点、闭曲线 ≥8 点（少于此的折线通常是刻意画的折角）；
      2) **绝大多数**顶点是缓拐（转角 < max_turn 的占 ≥80%）——允许个别尖点
         （葫芦顶的凹口、云朵的扇贝接缝就是尖点，整条轮廓仍应是曲线）；
         正多边形/星形的顶点全是尖点，会在这里被挡掉；
      3) 不是正多边形（八边形 45° 刚好卡在门槛内，靠这条挡住；顶点 ≥10 的
         均匀分布视为"在近似圆"，允许转曲线）；
      4) 不是一条近似直线（那种东西用 line 表达更清楚）。
    """
    if np is None:
        return False
    # 首尾重合点要去掉：否则下面"近似直线"判据里 a=pts[0]、b=pts[-1] 是同一个点，
    # L=0 直接判死，闭合轮廓永远升级不了（concepts 的 _smooth 就带这个尾巴）。
    pts, auto_closed = _drop_closing_dup(list(pts))
    closed = bool(closed or auto_closed)
    n = len(pts)
    if closed:
        if n < min_closed:
            return False
    elif n < min_open:
        return False
    turns = turn_angles(pts, closed)
    if not turns:
        return False
    sharp = sum(1 for t in turns if t > max_turn)
    if sharp > sharp_ratio * len(turns):
        return False
    # 正多边形保护只对「顶点少」的生效（三~九边形）：12 个均匀点更可能是
    # 模型在近似一个圆（真正的正十二边形由 shapes.py 直出，带 no_curve 标记）。
    if closed and n <= 9 and is_regular_polygon(pts, True):
        return False
    # 近似直线：所有点到首尾连线的距离都很小
    a, b = pts[0], pts[-1]
    L = _seg_len(a, b)
    if L < 1e-9:
        return False
    dev = 0.0
    for p in pts[1:-1]:
        dev = max(dev, abs((b[0] - a[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (b[1] - a[1])) / L)
    if dev < 0.01 * L:
        return False
    return True


def resample(pts: Sequence[Point], count: int, closed: bool = False,
             curve_bias: float = 3.0) -> List[Point]:
    """按"弧长 × 局部弯曲程度"重采样到 count 个点（首尾保留）。

    - 均匀化控制点规模：DXF 里 30~40 个控制点的样条足够光滑，文件也不大；
    - **按转角加权**：拐得急的地方（云朵的扇贝接缝、葫芦的腰、花瓣尖）多放点，
      否则重采样会把尖点"抹过"，拟合结果塌角、出现台阶。
    """
    pts = list(pts)
    if len(pts) < 3 or count >= len(pts):
        return pts
    count = max(4, int(count))
    seq = pts + ([pts[0]] if closed else [])
    # 每个顶点的权重：越弯越大
    w = [1.0] * len(seq)
    try:
        turns = turn_angles(pts, closed)
        base = list(range(1, len(pts) - 1)) if not closed else list(range(len(pts)))
        for k, t in enumerate(turns):
            if k < len(base):
                w[base[k]] = 1.0 + curve_bias * min(90.0, t) / 90.0
        w[-1] = w[0] if closed else w[-1]
    except Exception:                                  # noqa: BLE001
        pass
    acc = [0.0]
    for i in range(len(seq) - 1):
        seg = _seg_len(seq[i], seq[i + 1]) * 0.5 * (w[i] + w[i + 1])
        acc.append(acc[-1] + seg)
    total = acc[-1]
    if total <= 1e-9:
        return pts
    nseg = count if closed else count - 1
    out: List[Point] = []
    j = 0
    for k in range(nseg if closed else nseg + 1):
        target = total * k / nseg
        while j < len(acc) - 2 and acc[j + 1] < target:
            j += 1
        span = acc[j + 1] - acc[j]
        t = 0.0 if span <= 1e-12 else (target - acc[j]) / span
        x = seq[j][0] + (seq[j + 1][0] - seq[j][0]) * t
        y = seq[j][1] + (seq[j + 1][1] - seq[j][1]) * t
        out.append((x, y))
    return out


# ------------------------------------------------------------ B 样条：开曲线插值
def _find_span(knots: Sequence[float], nctrl: int, degree: int, u: float) -> int:
    """u 落在哪个节点区间（Cox-de Boor 的非零区间下标）。"""
    if u >= knots[nctrl]:
        return nctrl - 1
    if u <= knots[degree]:
        return degree
    lo, hi = degree, nctrl
    mid = (lo + hi) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


def _basis_funs(span: int, u: float, degree: int, knots: Sequence[float]) -> List[float]:
    """区间 span 上非零的 degree+1 个基函数值（Piegl A2.2，数值稳定版）。"""
    N = [0.0] * (degree + 1)
    left = [0.0] * (degree + 1)
    right = [0.0] * (degree + 1)
    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knots[span + 1 - j]
        right[j] = knots[span + j] - u
        saved = 0.0
        for r in range(j):
            den = right[r + 1] + left[j - r]
            temp = 0.0 if abs(den) < 1e-12 else N[r] / den
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def _chord_params(pts: Sequence[Point]) -> List[float]:
    d = [0.0]
    for i in range(1, len(pts)):
        d.append(d[-1] + _seg_len(pts[i - 1], pts[i]))
    total = d[-1]
    if total <= 1e-12:
        return [i / max(1, len(pts) - 1) for i in range(len(pts))]
    return [v / total for v in d]


def _averaging_knots(ubar: Sequence[float], degree: int) -> List[float]:
    """A9.1 的节点向量：两端 degree+1 重，中间用参数平均值。"""
    n = len(ubar) - 1
    p = degree
    knots = [0.0] * (p + 1)
    for j in range(1, n - p + 1):
        knots.append(sum(ubar[j:j + p]) / p)
    knots += [1.0] * (p + 1)
    return knots


def fit_open(pts: Sequence[Point], degree: int = DEFAULT_DEGREE) -> Dict[str, Any]:
    """开曲线全局插值：返回 {degree, ctrl, knots, closed, n}，曲线精确过每个点。"""
    pts = list(pts)
    if len(pts) < 2:
        return {}
    degree = max(1, min(degree, len(pts) - 1))
    if degree == 1:                                    # 两点 → 一次样条（即直线）
        knots = [0.0, 0.0, 1.0, 1.0]
        return {"degree": 1, "ctrl": [tuple(p) for p in pts],
                "knots": knots, "closed": False, "n": len(pts)}
    ubar = _chord_params(pts)
    knots = _averaging_knots(ubar, degree)
    n = len(pts) - 1
    nctrl = n + 1
    A = np.zeros((nctrl, nctrl))
    for k, u in enumerate(ubar):
        span = _find_span(knots, nctrl, degree, u)
        N = _basis_funs(span, u, degree, knots)
        for j in range(degree + 1):
            col = span - degree + j
            if 0 <= col < nctrl:
                A[k, col] = N[j]
    rhs = np.asarray(pts, dtype=float)
    # A9.1 的列归一化：中间方程除以基函数和，条件数从 1e6 降到 1e2 量级
    for k in range(1, nctrl - 1):
        s = A[k].sum()
        if s > 1e-12:
            A[k] /= s
            rhs[k] /= s
    try:
        ctrl = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:                      # pragma: no cover
        ctrl = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return {"degree": degree, "ctrl": [tuple(p) for p in ctrl],
            "knots": knots, "closed": False, "n": len(pts)}


# ---------------------------------------------------------- B 样条：闭合周期插值
def fit_closed(pts: Sequence[Point], degree: int = DEFAULT_DEGREE) -> Dict[str, Any]:
    """闭合周期 B 样条插值（三次）→ 转成"夹紧节点 + 节点重数 3"的标准表示。

    两步：
    1. 周期三次 B 样条在整数节点处 c[i-1]、c[i]、c[i+1] 的权重是 1/6、4/6、1/6，
       所以控制点 = 求解循环三对角方程组（严格对角占优，必然可逆）。
       这样得到的闭合曲线**没有接缝**，首尾切向也连续。
    2. 直接把周期节点向量写进 DXF 是不行的：非夹紧节点会让曲线的合法定义域缩到
       [p, n+p]，有些读图器（含 ezdxf 的 flattening）会从 0 开始采样，凭空多出一段
       外推的"钩子"。所以再把每个节点区间转成一段三次贝塞尔
       （B1 = S(k)+S'(k)/3，B2 = S(k+1)-S'(k+1)/3），用**节点重数 3** 拼成一条
       夹紧 B 样条：几何与周期曲线完全一致，定义域就是 [0, n]，谁读都不会外推。
    """
    pts = list(pts)
    pts, _ = _drop_closing_dup(pts)
    n = len(pts)
    if n < 5:
        return {}
    degree = 3                                        # CAD 里三次够用且最稳
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i - 1) % n] = 1.0 / 6.0
        A[i, i] = 4.0 / 6.0
        A[i, (i + 1) % n] = 1.0 / 6.0
    rhs = np.asarray(pts, dtype=float)
    ctrl = np.linalg.solve(A, rhs)
    cp = [tuple(p) for p in ctrl]

    def s(k: int) -> Point:                           # 曲线上的点 S(k)
        return ((cp[(k - 1) % n][0] + 4 * cp[k % n][0] + cp[(k + 1) % n][0]) / 6.0,
                (cp[(k - 1) % n][1] + 4 * cp[k % n][1] + cp[(k + 1) % n][1]) / 6.0)

    def d(k: int) -> Point:                           # 一阶导 S'(k)
        return ((cp[(k + 1) % n][0] - cp[(k - 1) % n][0]) / 2.0,
                (cp[(k + 1) % n][1] - cp[(k - 1) % n][1]) / 2.0)

    bez: List[Point] = []
    p0 = s(0)
    bez.append(p0)
    for k in range(n):
        p_k, p_k1 = s(k), s(k + 1)
        d_k, d_k1 = d(k), d(k + 1)
        bez.append((p_k[0] + d_k[0] / 3.0, p_k[1] + d_k[1] / 3.0))
        bez.append((p_k1[0] - d_k1[0] / 3.0, p_k1[1] - d_k1[1] / 3.0))
        bez.append(p_k1)
    knots = [0.0] * 4
    for k in range(1, n):
        knots += [float(k)] * 3
    knots += [float(n)] * 4
    return {"degree": degree, "ctrl": bez, "knots": knots,
            "closed": True, "n": n, "periodic_ctrl": cp}


def fit_curve(pts: Sequence[Point], closed: bool = False,
              degree: int = DEFAULT_DEGREE) -> Dict[str, Any]:
    """统一入口：自动选择开/闭曲线拟合。点数不足返回 {}。"""
    pts = norm_points(pts)
    if len(pts) < 2:
        return {}
    if closed or is_closed_pts(pts):
        fit = fit_closed(pts)
        if fit:
            return fit
        pts, _ = _drop_closing_dup(pts)
    return fit_open(pts, degree)


# --------------------------------------------------------------------- 采样
def _eval_open(ctrl, knots, degree: int, u: float) -> Point:
    nctrl = len(ctrl)
    u = min(max(u, knots[degree]), knots[nctrl])
    span = _find_span(knots, nctrl, degree, u)
    d = [tuple(ctrl[span - degree + j]) for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = span - degree + j
            den = knots[i + degree - r + 1] - knots[i]
            a = 0.0 if abs(den) < 1e-12 else (u - knots[i]) / den
            x0, y0 = d[j - 1]
            x1, y1 = d[j]
            d[j] = (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
    return d[degree]


def _eval_periodic(ctrl, t: float) -> Point:
    """周期均匀三次 B 样条求值：t ∈ [0, n)，节点整数处精确落在采样点上。"""
    n = len(ctrl)
    k = int(math.floor(t)) % n
    f = t - math.floor(t)
    b0 = (1 - f) ** 3 / 6.0
    b1 = (3 * f ** 3 - 6 * f ** 2 + 4) / 6.0
    b2 = (-3 * f ** 3 + 3 * f ** 2 + 3 * f + 1) / 6.0
    b3 = f ** 3 / 6.0
    p0 = ctrl[(k - 1) % n]
    p1 = ctrl[k % n]
    p2 = ctrl[(k + 1) % n]
    p3 = ctrl[(k + 2) % n]
    x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
    y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
    return (x, y)


def sample_fit(fit: Dict[str, Any], per_span: float = 6.0,
               max_points: int = 400) -> List[Point]:
    """把拟合结果细分采样（前端画布 / 预览叠加图 / 橡皮擦都用它）。"""
    if not fit:
        return []
    degree = int(fit.get("degree") or 3)
    if fit.get("closed") and fit.get("periodic_ctrl"):
        ctrl = fit["periodic_ctrl"]
        n = len(ctrl)
        steps = max(4, min(int(round(n * per_span)), max_points))
        return [tuple(round(v, 4) for v in _eval_periodic(ctrl, n * k / steps))
                for k in range(steps)]
    ctrl = fit.get("ctrl") or []
    knots = fit.get("knots") or []
    if len(ctrl) < 2 or len(knots) < len(ctrl) + degree + 1:
        return [tuple(p) for p in ctrl]
    spans = max(1, len(ctrl) - degree)
    steps = max(4, min(int(round(spans * per_span)), max_points))
    lo, hi = knots[degree], knots[len(ctrl)]
    if hi - lo <= 1e-12:
        return [tuple(p) for p in ctrl]
    return [tuple(round(v, 4) for v in _eval_open(ctrl, knots, degree,
                                                  lo + (hi - lo) * k / steps))
            for k in range(steps + 1)]


# --------------------------------------------------------------- 椭圆（含圆弧）
def ellipse_points(center: Sequence[float], rx: float, ry: float,
                   rotation: float = 0.0, start: float = 0.0, end: float = 360.0,
                   samples: int = 96) -> List[Point]:
    """椭圆/椭圆弧采样（角度为度，rotation 为长轴倾角）。"""
    cx, cy = float(center[0]), float(center[1])
    rx, ry = abs(float(rx)), abs(float(ry))
    r = math.radians(float(rotation))
    ca, sa = math.cos(r), math.sin(r)
    a0, a1 = math.radians(float(start)), math.radians(float(end))
    if a1 <= a0:
        a1 += 2 * math.pi
    full = abs((a1 - a0) - 2 * math.pi) < 1e-6
    n = max(8, int(samples))
    out = []
    for k in range(n if full else n + 1):
        a = a0 + (a1 - a0) * k / n
        x, y = rx * math.cos(a), ry * math.sin(a)
        out.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return out


# ------------------------------------------------------------- 实体级：曲线化
def spline_entity(points: Sequence[Point], closed: bool = False, layer: str = "0",
                  per_span: float = 8.0, max_ctrl: int = 40) -> Dict[str, Any]:
    """折线点 → spline 实体（points 存的就是细分后的光滑点列）。"""
    pts, auto_closed = _drop_closing_dup(norm_points(points))
    closed = bool(closed or auto_closed)
    if len(pts) < 3:
        return {"type": "polyline", "points": [list(p) for p in pts],
                "closed": closed, "layer": layer}
    base = resample(pts, min(max_ctrl, len(pts)), closed)
    fit = fit_curve(base, closed)
    dense = sample_fit(fit, per_span=per_span) if fit else base
    if not dense:
        dense = base
    return {"type": "spline", "points": [[round(float(x), 3), round(float(y), 3)]
                                         for x, y in dense],
            "closed": closed, "layer": layer}


def ellipse_entity(center, rx, ry, rotation: float = 0.0, layer: str = "0",
                   start: float = 0.0, end: float = 360.0,
                   samples: int = 96) -> Dict[str, Any]:
    """椭圆实体：既存椭圆参数（DXF 写成真 ELLIPSE），也存细分点（前端直接画）。"""
    return {"type": "ellipse",
            "center": [round(float(center[0]), 3), round(float(center[1]), 3)],
            "rx": round(abs(float(rx)), 3), "ry": round(abs(float(ry)), 3),
            "rotation": round(float(rotation), 4),
            "start": float(start), "end": float(end),
            "points": [[round(x, 3), round(y, 3)]
                       for x, y in ellipse_points(center, rx, ry, rotation,
                                                  start, end, samples)],
            "closed": True, "layer": layer}


def _poly_points(e: Dict[str, Any]) -> Tuple[List[Point], bool]:
    raw = (e.get("points") or e.get("vertices") or e.get("pts")
           or e.get("fit") or e.get("fit_points") or [])
    pts = norm_points(raw)
    closed = bool(e.get("closed") or e.get("close"))
    if not closed and is_closed_pts(pts):
        closed = True
    return pts, closed


def curveify(entities: Sequence[Dict[str, Any]], per_span: float = 6.0,
             max_turn: float = 50.0, loose: bool = False) -> List[Dict[str, Any]]:
    """把"本来就是曲线"的折线升级成 spline，并规范化已有的 spline/ellipse。

    保守规则（见 ``should_curve``）：只动那些顶点多、处处缓拐、非正多边形的折线。
    显式标记优先：``smooth=True`` 强制曲线化、``no_curve=True`` / ``polygon=True``
    保持折线（shapes.py 的多边形都带 no_curve，避免把五角星磨圆）。
    ``loose=True`` 时用 ``LOOSE_KW`` 的放宽判据（经典视觉识别的碎弦链专用）。
    """
    if loose:
        max_turn = LOOSE_KW["max_turn"]
    out: List[Dict[str, Any]] = []
    for e in entities or []:
        if not isinstance(e, dict):
            out.append(e)
            continue
        t = str(e.get("type") or e.get("kind") or "").strip().lower()
        layer = str(e.get("layer") or "0")
        if t in SPLINE_ALIASES:
            pts, closed = _poly_points(e)
            if len(pts) >= 3 and not e.get("points"):
                e = spline_entity(pts, closed, layer, per_span)
            out.append(e)
        elif t in ELLIPSE_ALIASES:
            if not e.get("points"):
                c = e.get("center") or e.get("c") or [0, 0]
                rx = _f(e.get("rx", e.get("a", e.get("major"))))
                ry = _f(e.get("ry", e.get("b", e.get("minor"))))
                if rx is None and _f(e.get("radius")) is not None:
                    rx = ry = _f(e.get("radius"))
                if rx and ry:
                    e = ellipse_entity(c, rx, ry, _f(e.get("rotation"), 0.0) or 0.0,
                                       layer, _f(e.get("start"), 0.0) or 0.0,
                                       _f(e.get("end"), 360.0) or 360.0)
            out.append(e)
        elif t in POLYLINE_ALIASES:
            if e.get("no_curve") or e.get("polygon") or e.get("smooth") is False:
                out.append(e)
                continue
            pts, closed = _poly_points(e)
            force = e.get("smooth") is True
            if loose:
                ok = len(pts) >= LOOSE_KW["min_closed"] if closed \
                    else len(pts) >= LOOSE_KW["min_open"]
            else:
                ok = len(pts) >= 3
            if ok and (force or should_curve(pts, closed, max_turn,
                                             min_open=LOOSE_KW["min_open"] if loose else 6,
                                             min_closed=LOOSE_KW["min_closed"] if loose else 8,
                                             sharp_ratio=LOOSE_KW["sharp_ratio"] if loose else 0.20)):
                out.append(spline_entity(pts, closed, layer, per_span))
            else:
                out.append(e)
        else:
            out.append(e)
    return out


def count_curves(entities: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """统计曲线实体数量（给提示语用）。"""
    c = {"spline": 0, "ellipse": 0}
    for e in entities or ():
        if not isinstance(e, dict):
            continue
        t = str(e.get("type") or "").strip().lower()
        if t in SPLINE_ALIASES:
            c["spline"] += 1
        elif t in ELLIPSE_ALIASES:
            c["ellipse"] += 1
    return c


# ------------------------------------------------ 实体级：一串短直线 → 一条样条
_LINE_TYPES = ("line", "直线", "line2d")
_LINE_P1 = ("p1", "start", "from", "a", "x1y1")
_LINE_P2 = ("p2", "end", "to", "b", "x2y2")


def _line_ends(e: Dict[str, Any]) -> Tuple[Any, Any]:
    """宽松取直线端点（p1/p2、start/end、from/to 都认）。"""
    a = b = None
    for k in _LINE_P1:
        if k in e:
            a = _xy(e[k])
            if a is not None:
                break
    for k in _LINE_P2:
        if k in e:
            b = _xy(e[k])
            if b is not None:
                break
    return a, b


def _cell(p: Point, tol: float) -> Tuple[int, int]:
    return (int(math.floor(p[0] / tol)), int(math.floor(p[1] / tol)))


def chain_lines_to_curves(entities: Sequence[Dict[str, Any]], tol: float = 0.05,
                          min_seg: int = 5, per_span: float = 8.0,
                          max_seg: int = 400,
                          max_join_turn: float = 45.0,
                          loose: bool = False) -> List[Dict[str, Any]]:
    """把「一串首尾相接的短直线」串成一条样条曲线。

    为什么还要这一步：模型（以及部分参数化配方）用直边形近似弧时，输出的不是
    一条 polyline，而是 **几十条独立 line**。它们各自只有 2 个点，``curveify``
    按单个实体判断时永远不像曲线，于是图纸上仍是一圈锯齿。这里先做几何连通性
    串链，再对整条链跑 ``should_curve``：

    - 端点按 ``tol`` 网格聚类（±1 格相邻都算相连，避免正好卡在格子边界上断开）；
    - 只沿"唯一候选"前进：遇到断头路或岔口（真正的节点，比如支架的横梁与立柱
      交汇）立即停下——这样 L 形支架、矩形框、桁架不会被误连成一条曲线；
    - **跨拐角不连**：相邻两段的夹角 > ``max_join_turn``（默认 45°）就停。曲线
      被离散成短线时，相邻段夹角很小（20 段圆的每段才 18°）；而"一段圆弧接一条
      直边"在接缝处是 90°，那种地方必须保留尖角，不能被样条抹圆；
    - 只串同一图层的直线（不同图层通常语义不同）；
    - 整条链必须通过 ``should_curve``（顶点够多、处处缓拐、不是正多边形、
      不是一条近似直线）才升级，升级后在同一槽位换成 spline，其余直线标记丢弃。
    - ``loose=True``：放宽判据（见 ``LOOSE_KW``）——升链门槛降到 3 段、
      缓拐上限 62°、允许 32% 尖点。给经典视觉识别结果用：霍夫把缓弯切成
      3~4 段碎弦是常态，按保守判据它们永远凑不够"顶点数"。

    返回新列表；顺序、图层、未参与串链的实体全部原样保留。
    """
    lst = list(entities or [])
    if np is None or len(lst) < min_seg:
        return lst

    segs: List[Tuple[int, Point, Point, str]] = []      # (原下标, 起点, 终点, 图层)
    for i, e in enumerate(lst):
        if not isinstance(e, dict):
            continue
        t = str(e.get("type") or e.get("kind") or "").strip().lower()
        if t not in _LINE_TYPES:
            continue
        a, b = _line_ends(e)
        if a is None or b is None or _seg_len(a, b) < 1e-9:
            continue
        segs.append((i, a, b, str(e.get("layer") or "0")))
    if len(segs) < min_seg:
        return lst

    # 端点 → [(segs 下标, 端点序号 0/1)]，3×3 邻格都查
    buckets: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for si, (_, a, b, _lay) in enumerate(segs):
        buckets.setdefault(_cell(a, tol), []).append((si, 0))
        buckets.setdefault(_cell(b, tol), []).append((si, 1))

    def _near(p: Point, used: set, layer: str) -> List[Tuple[int, int]]:
        cx, cy = _cell(p, tol)
        out: List[Tuple[int, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for si, end in buckets.get((cx + dx, cy + dy), ()):
                    if si in used:
                        continue
                    q = segs[si][1] if end == 0 else segs[si][2]
                    if _seg_len(p, q) <= tol and segs[si][3] == layer:
                        out.append((si, end))
        return out

    used: set = set()
    replaced: Dict[int, Any] = {}                       # 原下标 → spline / None(丢弃)
    for si in range(len(segs)):
        if si in used:
            continue
        i0, a0, b0, lay0 = segs[si]
        used.add(si)
        members = [si]
        back: List[Point] = [a0]                        # 从 a0 往回走到的点（逆序）
        fwd: List[Point] = [a0, b0]                     # 从 a0 往前的点（正序）
        head, tail = a0, b0

        def _walk(from_pt: Point, prev_pt: Point, forward: bool) -> None:  # noqa: F811
            """沿唯一候选一路走，遇到岔口 / 断头 / 大拐角 / 闭合就停。"""
            nonlocal head, tail
            cur, prev = from_pt, prev_pt
            while len(members) < max_seg:
                cands = _near(cur, used, lay0)
                if len(cands) != 1:                     # 断头 / 岔口 → 停
                    return
                cj, ce = cands[0]
                pa, pb = segs[cj][1], segs[cj][2]
                far = pb if ce == 0 else pa
                # 拐角闸门：进段方向与出段方向夹角过大 = 尖角，不属于同一条曲线
                v1 = (cur[0] - prev[0], cur[1] - prev[1])
                v2 = (far[0] - cur[0], far[1] - cur[1])
                n1, n2 = math.hypot(*v1), math.hypot(*v2)
                if n1 > 1e-9 and n2 > 1e-9:
                    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
                    if math.degrees(math.acos(cos)) > max_join_turn:
                        return
                used.add(cj)
                members.append(cj)
                if forward:
                    fwd.append(far)
                    tail = far
                else:
                    back.append(far)
                    head = far
                prev, cur = cur, far
                # 已经绕回起点 → 闭合链，停止
                if _seg_len(far, fwd[0]) <= tol * 2:
                    return

        _walk(tail, head, True)
        if _seg_len(fwd[-1], fwd[0]) > tol * 2:          # 没闭合才需要往回走
            _walk(head, tail, False)

        if len(members) < min_seg:
            continue
        pts = list(reversed(back)) + fwd[1:]
        pts = norm_points(pts)
        closed = len(pts) > 3 and _seg_len(pts[0], pts[-1]) <= tol * 2
        if closed:
            pts = pts[:-1]
        if len(pts) < min_seg:
            continue
        if loose:
            sc = should_curve(pts, closed, LOOSE_KW["max_turn"],
                              min_open=LOOSE_KW["min_open"],
                              min_closed=LOOSE_KW["min_closed"],
                              sharp_ratio=LOOSE_KW["sharp_ratio"])
        else:
            sc = should_curve(pts, closed)
        if not sc:
            continue
        spl = spline_entity(pts, closed, lay0, per_span)
        if spl.get("type") != "spline":                  # 拟合失败就别动原图
            continue
        replaced[i0] = spl
        for m in members[1:]:
            replaced[segs[m][0]] = None

    if not replaced:
        return lst
    out: List[Dict[str, Any]] = []
    for i, e in enumerate(lst):
        if i in replaced:
            if replaced[i] is not None:
                out.append(replaced[i])
            continue
        out.append(e)
    return out



# ------------------------------------------------ 给 DXF 写出器：实体 → 样条定义
def spline_definition(ent: Dict[str, Any]) -> Dict[str, Any]:
    """DXF SPLINE 需要的 {degree, ctrl, knots, closed}；数据不足返回 {}。

    输入点先按弧长均匀压到 ≤40 个再拟合：形状不变，但 DXF 里的控制点少、
    文件小、任何 CAD 打开都快（密集点直接拟合会得到上百个控制点）。
    """
    pts, closed = _poly_points(ent)
    if len(pts) < 2:
        return {}
    base = resample(pts, min(40, len(pts)), closed) if len(pts) > 40 else pts
    fit = fit_curve(base, closed)
    return fit or {}


if __name__ == "__main__":                             # pragma: no cover
    # 自测：圆上取 12 点 → 拟合闭合样条 → 采样，检查是否过点且够光滑
    n = 12
    pts = [(30 * math.cos(2 * math.pi * k / n) + 5,
            20 * math.sin(2 * math.pi * k / n) - 7) for k in range(n)]
    ent = spline_entity(pts, closed=True)
    print("spline points:", len(ent["points"]))
    # 采样点应全部落在椭圆上（误差 <= 0.5mm）
    err = max(abs(math.hypot((x - 5) / 30.0, (y + 7) / 20.0) - 1.0) for x, y in ent["points"])
    print("相对径向误差: %.4f" % err)
    print("open:", len(spline_entity([(0, 0), (10, 20), (30, 25), (50, 5)], False)["points"]))
