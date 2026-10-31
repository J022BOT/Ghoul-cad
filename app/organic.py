"""organic.py — 有机线稿矢量化：墨迹 → 骨架 → 有序路径 → 直线 / 圆 / 样条曲线。

为什么需要
----------
`line_extractor` 的霍夫路线是为**工程图**设计的：直边、圆、矩形，一条边一条线。
遇到螺旋、云纹、手绘曲线这种「有机线稿」，霍夫只能把曲线切成几十上百段短弦
（实测一张 492×382 的漩涡图 → 232 条直线，其中 95 条还是互相重叠的重复段），
放大看全是棱角 —— 这就是"AI 只会用直线"的真正来源。

而且这些短弦是**交叉重叠**的，不是首尾相接的：端点 3px 邻域里挤着十几个邻居，
所以"端点串链"（curves.chain_lines_to_curves）在这里一条也串不起来。

做法
----
1. 墨迹二值图 → 形态学开/闭（去孤立噪点、补 1~2px 断口）；
2. Zhang-Suen 细化得到 1px 骨架（笔画中心线）；
3. 沿骨架走路径：遇到岔点**按方向穿过去**（取转角最小的分支），
   这样 6 条螺旋各走各的，不会在相切处串线；
4. 路径 → RDP 简化（1.2px）→ 判形状：
   闭合且贴合圆（径向偏差 <5%）→ circle；近似直线 → line；其余 → spline；
5. 只有当"弯曲墨水"占多数时才启用（否则是工程图，交给霍夫的精确结果）。

坐标一律是**图像像素**（y 向下），由调用方统一变换到 DXF 毫米坐标。
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import curves

Point = Tuple[float, float]

_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


# ------------------------------------------------------------------ 骨架
def thin(ink: np.ndarray, max_it: int = 80) -> np.ndarray:
    """Zhang-Suen 细化，输入 0/1，输出 1px 骨架（同尺寸 uint8）。"""
    img = (ink > 0).astype(np.uint8)
    for _ in range(max_it):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            P2, P3, P4 = p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:]
            P5, P6, P7 = p[2:, 2:], p[2:, 1:-1], p[2:, 0:-2]
            P8, P9 = p[1:-1, 0:-2], p[0:-2, 0:-2]
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = (P2, P3, P4, P5, P6, P7, P8, P9, P2)
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
            if step == 0:
                c1, c2 = (P2 * P4 * P6) == 0, (P4 * P6 * P8) == 0
            else:
                c1, c2 = (P2 * P4 * P8) == 0, (P2 * P6 * P8) == 0
            kill = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            break
    return img


def _nbmap(sk: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """每个骨架像素 → 8 邻域里的骨架像素。"""
    h, w = sk.shape
    out: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for y, x in zip(*np.nonzero(sk)):
        nbs = []
        for dy, dx in _N8:
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and sk[yy, xx]:
                nbs.append((yy, xx))
        out[(y, x)] = nbs
    return out


def _heading(path: Sequence[Tuple[int, int]], back: int = 6) -> Tuple[float, float]:
    a = path[max(0, len(path) - back)]
    b = path[-1]
    dx, dy = b[1] - a[1], b[0] - a[0]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (1.0, 0.0)


def despeckle(ink: np.ndarray, min_area: int = 6) -> np.ndarray:
    """删掉面积过小的孤立连通域（噪点/JPEG 碎屑）。

    **不能用 3×3 开运算**：3px 宽的曲线斜着走时，正交核的腐蚀会把它切成一截截
    （实测一条螺旋被切成 52 段，直接毁掉矢量化）。按连通域面积删才是对的：
    噪点是孤立的几像素团，笔画无论多细都是连通的。
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8),
                                                        connectivity=8)
    out = np.zeros_like(ink)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def prepare_ink(ink: np.ndarray) -> np.ndarray:
    """去噪 + 补断口，**自适应**选择要不要做开运算。

    两张实测图的诉求正好相反：
      - 手机截图/照片：笔画周围有一圈浅色晕边，二值化后会带出一堆 1~2px 的碎屑，
        骨架被碎屑拖成毛球（566 个连通域）。这里必须开运算把这些浮屑削掉，
        开完反而更干净（566 → 25），路径从 72 段降到 21 段。
      - 干净细线稿：笔画只有 2~3px、又是斜着走的，3×3 正交核的腐蚀会把笔画**切断**
        （1 个连通域 → 52 个），开运算直接毁掉矢量化。

    判据就看开运算前后：连通域变少、墨迹还剩一半以上 → 说明只是在削浮屑，用它；
    否则（连通域变多 = 把笔画切断了）→ 改用按面积删小连通域。
    """
    n0 = cv2.connectedComponents((ink > 0).astype(np.uint8), connectivity=8)[0] - 1
    k3 = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k3)
    n1 = cv2.connectedComponents(opened, connectivity=8)[0] - 1
    area0 = float((ink > 0).sum())
    if n1 <= n0 and (opened > 0).sum() >= 0.5 * area0:
        ink = opened
    else:
        ink = despeckle(ink, 6)
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k3)


def prune_spurs(sk: np.ndarray, min_len: int = 12, rounds: int = 3) -> np.ndarray:
    """剪掉骨架上的短悬挂分支（毛刺）。

    粗笔画/模糊图细化后，主线上会长出很多 5~20px 的小岔枝（实测 1400px 的漩涡图
    能长出几十根），不剪的话每一根都会变成一条独立样条，图纸立刻变成一片乱线。
    做法：从每个端点沿"度=2"的链往里走，走不到 min_len 就撞上岔点 → 整条剪掉。
    """
    min_len = max(4, int(min_len))
    for _ in range(max(1, rounds)):
        nbm = _nbmap(sk)
        if not nbm:
            break
        deg = {k: len(v) for k, v in nbm.items()}
        removed = 0
        for ep in [k for k, v in deg.items() if v == 1]:
            if not sk[ep[0], ep[1]]:
                continue
            chain = [ep]
            prev, cur = None, ep
            while len(chain) <= min_len:
                nbs = [n for n in nbm.get(cur, ()) if n != prev]
                if len(nbs) != 1:
                    break
                n = nbs[0]
                chain.append(n)
                if deg.get(n, 0) != 2:
                    break
                prev, cur = cur, n
            if len(chain) <= min_len and deg.get(chain[-1], 0) > 2:
                for (y, x) in chain[:-1]:
                    sk[y, x] = 0
                removed += 1
        if not removed:
            break
    return sk


def trace_paths(sk: np.ndarray, min_len: int = 10,
                max_turn: float = 55.0) -> List[List[Point]]:
    """沿骨架走有序路径（像素坐标，x,y）。

    关键：岔点上**按方向穿过去**。骨架在"两条笔画相切"的地方会连成 X 形节点，
    若象普通连通域那样"到岔点就停"，一条螺旋会被切成十几段（实测 6 条螺旋 →
    51 段碎片）；按最小转角继续走才能保持一整条笔画。
    """
    nbm = _nbmap(sk)
    if not nbm:
        return []
    deg = {k: len(v) for k, v in nbm.items()}
    used = set()

    def walk(start, first):
        path = [start, first]
        used.add((start, first)); used.add((first, start))
        cur = first
        while True:
            cands = [n for n in nbm.get(cur, ()) if (cur, n) not in used]
            if not cands:
                break
            if len(cands) == 1:
                nxt = cands[0]
            else:
                h = _heading(path)
                nxt, best = None, 1e9
                for n in cands:
                    dx, dy = n[1] - cur[1], n[0] - cur[0]
                    m = math.hypot(dx, dy)
                    c = max(-1.0, min(1.0, (h[0] * dx + h[1] * dy) / (m or 1.0)))
                    a = math.degrees(math.acos(c))
                    if a < best:
                        best, nxt = a, n
                if nxt is None or best > max_turn:
                    break
            used.add((cur, nxt)); used.add((nxt, cur))
            path.append(nxt)
            cur = nxt
            if len(path) > 20000:
                break
        return path

    paths: List[List[Point]] = []
    seeds = [k for k in nbm if deg[k] != 2] + list(nbm)
    for s in seeds:
        for n in nbm[s]:
            if (s, n) not in used:
                paths.append(walk(s, n))
    out = []
    for p in paths:
        if len(p) < min_len:
            continue
        out.append([(float(x), float(y)) for y, x in p])
    return out


# ------------------------------------------------------------------ 几何
def rdp(pts: Sequence[Point], eps: float) -> List[Point]:
    """Ramer-Douglas-Peucker 简化（递归版，点数不多够用）。"""
    pts = list(pts)
    if len(pts) < 3:
        return pts
    a = np.asarray(pts[0], float)
    b = np.asarray(pts[-1], float)
    ab = b - a
    L = float(np.hypot(*ab))
    P = np.asarray(pts, float)
    if L < 1e-9:
        d = np.hypot(P[:, 0] - a[0], P[:, 1] - a[1])
    else:
        d = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L
    i = int(np.argmax(d))
    if d[i] > eps:
        return rdp(pts[:i + 1], eps)[:-1] + rdp(pts[i:], eps)
    return [pts[0], pts[-1]]


def path_length(pts: Sequence[Point]) -> float:
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def smooth_path(pts: Sequence[Point], win: int = 7, closed: bool = False) -> List[Point]:
    """沿路径做滑动平均：抹掉 1px 骨架的锯齿/毛刺。

    不做这一步，样条会精确插值每一个像素抖动 → 曲线像蛇一样抖（实测放大后
    满是 1~2px 的波浪）。骨架已经把笔画缩到 1px，这些抖动是二值化/细化引入的
    量化噪声，不是真实形状。
    """
    n = len(pts)
    win = max(3, int(win) | 1)
    if n < win + 2:
        return [(float(p[0]), float(p[1])) for p in pts]
    half = win // 2
    ker = np.ones(win) / win
    P = np.asarray(pts, float)
    if closed:
        core = P[:-1] if math.hypot(P[-1][0] - P[0][0], P[-1][1] - P[0][1]) < 1e-6 else P
        ext = np.vstack([core[-half:], core, core[:half]])       # 环形填充
        xs = np.convolve(ext[:, 0], ker, mode="valid")
        ys = np.convolve(ext[:, 1], ker, mode="valid")
        out = np.c_[xs, ys]
        out = np.vstack([out, out[:1]])
        return [(float(a), float(b)) for a, b in out]
    ext = np.vstack([np.repeat(P[:1], half, axis=0), P, np.repeat(P[-1:], half, axis=0)])
    xs = np.convolve(ext[:, 0], ker, mode="valid")
    ys = np.convolve(ext[:, 1], ker, mode="valid")
    out = np.c_[xs, ys]
    out[0], out[-1] = P[0], P[-1]
    return [(float(a), float(b)) for a, b in out]


def is_closed(pts: Sequence[Point], tol: float) -> bool:
    return len(pts) > 3 and math.hypot(pts[-1][0] - pts[0][0],
                                       pts[-1][1] - pts[0][1]) <= tol


def max_deviation(pts: Sequence[Point]) -> float:
    """到首尾弦的最大垂距（判断"是不是一条直线"）。"""
    if len(pts) < 3:
        return 0.0
    a, b = np.asarray(pts[0], float), np.asarray(pts[-1], float)
    ab = b - a
    L = float(np.hypot(*ab))
    P = np.asarray(pts, float)
    if L < 1e-9:
        return float(max(np.hypot(P[:, 0] - a[0], P[:, 1] - a[1])))
    return float(max(np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L))


def fit_circle(pts: Sequence[Point]) -> Optional[Tuple[float, float, float, float]]:
    """代数最小二乘拟合圆（Kåsa）。返回 (cx, cy, r, 相对径向偏差)，失败 None。"""
    P = np.asarray(pts, float)
    if len(P) < 6:
        return None
    x, y = P[:, 0], P[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:                                     # noqa: BLE001
        return None
    cx, cy = float(sol[0]), float(sol[1])
    r2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = math.sqrt(float(r2))
    if r <= 1e-6:
        return None
    dev = float(np.max(np.abs(np.hypot(x - cx, y - cy) - r)) / r)
    return cx, cy, r, dev


# --------------------------------------------------------------- 主入口
def vectorize(bw: np.ndarray, D: float, min_ratio: float = 0.02,
              rdp_eps: float = 1.2, curved_share: float = 0.55,
              min_paths: int = 4, max_turn: float = 55.0,
              blank_boxes: Optional[Sequence[Sequence[float]]] = None) -> Dict[str, Any]:
    """墨迹二值图 → 曲线实体列表（像素坐标）。判定为工程图时返回空 entities。

    blank_boxes：先抹白的矩形（文字行），避免"70""R24"这类标注被当成笔画读成曲线。
    返回 {"entities": [...], "total_len": px, "curved_share": 0~1, "paths": n}
    """
    empty = {"entities": [], "total_len": 0.0, "curved_share": 0.0, "paths": 0}
    if bw is None or not hasattr(bw, "shape") or D <= 0:
        return empty
    ink = (bw > 0).astype(np.uint8)
    if ink.size == 0:
        return empty
    ratio = float(ink.mean())
    if ratio > 0.45:                     # 整张都是墨：不是线稿
        return empty
    if blank_boxes:
        h, w = ink.shape[:2]
        for (bx, by, bw_, bh_) in blank_boxes:
            x0, y0 = max(0, int(bx) - 2), max(0, int(by) - 2)
            x1, y1 = min(w, int(bx + bw_) + 2), min(h, int(by + bh_) + 2)
            if x1 > x0 and y1 > y0:
                ink[y0:y1, x0:x1] = 0
    # 骨架化限制在固定尺度：**必须在去噪之前缩**，否则同一张图大尺寸的晕边
    # 比小尺寸宽一倍，3×3 的核削不干净（实测 1400px 版本 63 段碎片 vs 492px 21 段）。
    inv = 1.0
    m = max(ink.shape[:2])
    if m > 900:
        sc = 900.0 / m
        # 注意 ink 是 0/1：先放大到 0/255 再插值，否则 INTER_AREA 后全部落到 0，
        # 再按 127 阈值就整张图变空白（曾经踩过）。
        small = cv2.resize(ink.astype(np.float32) * 255.0,
                           (max(1, int(round(ink.shape[1] * sc))),
                            max(1, int(round(ink.shape[0] * sc)))),
                           interpolation=cv2.INTER_AREA)
        ink = (small > 127).astype(np.uint8)
        inv = 1.0 / sc
        D = D * sc

    ink = prepare_ink(ink)
    sk = thin(ink)
    sk = prune_spurs(sk, int(max(6, 0.012 * D)))
    min_len = int(max(8, 0.02 * D))
    paths = trace_paths(sk, min_len=min_len, max_turn=max_turn)
    if not paths:
        return empty

    keep_len = max(0.02 * D, 6.0)
    simples = []                        # (rdp 点, 长度, 是否闭合, 是否"弯的")
    total = curved = 0.0
    for p in paths:
        L = path_length(p)
        if L < keep_len:
            continue
        closed = is_closed(p, max(2.5, 0.004 * D))
        sp = smooth_path(p, 7, closed)
        simp = rdp(sp, rdp_eps)
        total += L
        curvy = (len(simp) >= 5 and max_deviation(sp) > 0.01 * L) or \
                (closed and len(simp) >= 6)
        if curvy:
            curved += L
        simples.append((sp, simp, L, closed, curvy))

    share = (curved / total) if total > 1e-9 else 0.0
    n_curved = sum(1 for s in simples if s[4])
    if share < curved_share or n_curved < min_paths:
        return {"entities": [], "total_len": total * inv, "curved_share": share,
                "paths": len(simples), "reason": "not-organic"}

    ents: List[Dict[str, Any]] = []
    for p, simp, L, closed, curvy in simples:
        fit = fit_circle(p) if closed else None
        if fit is None and not closed:
            # 开口弧也可能是"几乎画满的圆"：两个端点离得近（不到弧长 20%）且
            # 径向偏差小 → 当成整圆。漩涡图外围那圈就是这么被判成圆的。
            gap = math.hypot(p[-1][0] - p[0][0], p[-1][1] - p[0][1])
            if L > 0 and gap < 0.20 * L:
                fit = fit_circle(list(p) + [p[0]])
        if fit and fit[3] < 0.05 and fit[2] > 0.02 * D:
            ents.append({"type": "circle",
                         "center": [round(fit[0], 3), round(fit[1], 3)],
                         "radius": round(fit[2], 3)})
            continue
        if not closed and max_deviation(p) < 0.008 * L:
            ents.append({"type": "line",
                         "p1": [round(p[0][0], 3), round(p[0][1], 3)],
                         "p2": [round(p[-1][0], 3), round(p[-1][1], 3)]})
            continue
        ent = curves.spline_entity(simp, closed)
        if ent.get("type") == "spline":
            ent.pop("layer", None)
            # 夹紧：样条是过点插值，路径在岔口处若有个小急拐，拟合出来的曲线可能
            # 甩到包围盒外面很远（实测一张图上有一条曲线跑到图像外 30px）。把细分点
            # 限制在"路径包围盒 + 1%路径长"内，形状不变但不会飞出去。
            px = [q[0] for q in p]
            py = [q[1] for q in p]
            pad = 0.01 * L + 2.0
            lo_x, hi_x = min(px) - pad, max(px) + pad
            lo_y, hi_y = min(py) - pad, max(py) + pad
            ent["points"] = [[min(max(float(x), lo_x), hi_x),
                              min(max(float(y), lo_y), hi_y)]
                             for x, y in ent["points"]]
            ents.append(ent)
        elif len(simp) >= 3:                              # 拟合失败退回折线
            ents.append({"type": "polyline",
                         "points": [[round(x, 3), round(y, 3)] for x, y in simp],
                         "closed": closed})
    if inv != 1.0:                        # 骨架化时缩过 → 坐标乘回去
        for e in ents:
            if e["type"] == "line":
                e["p1"] = [round(v * inv, 3) for v in e["p1"]]
                e["p2"] = [round(v * inv, 3) for v in e["p2"]]
            elif e["type"] == "circle":
                e["center"] = [round(v * inv, 3) for v in e["center"]]
                e["radius"] = round(e["radius"] * inv, 3)
            else:
                e["points"] = [[round(p[0] * inv, 3), round(p[1] * inv, 3)]
                               for p in e["points"]]
    return {"entities": ents, "total_len": total * inv, "curved_share": share,
            "paths": len(simples)}

def looks_organic(segs: Sequence[Any], D: float) -> bool:
    """霍夫结果是否"碎成一片"：段数多、且多数段远短于图幅 → 有机线稿。

    这是最省事也最准的一刀：工程图的边是长的（一条轮廓就是一段），螺旋/云纹被
    霍夫切成几十上百段短弦（实测漩涡图 232 段，中位长度只有图幅的 11%）。
    只用霍夫结果算，不用建骨架，判错也没有副作用（后面还有"弯曲墨水占比"闸门）。
    """
    if not segs or D <= 0 or len(segs) < 10:
        return False
    Ls = sorted(math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)
    med = Ls[len(Ls) // 2]
    short = sum(1 for L in Ls if L < 0.12 * D)
    return med < 0.20 * D and short > 0.5 * len(Ls)


def looks_organic_loose(segs: Sequence[Any], D: float) -> bool:
    """二次机会预检：多遍融合会把短弦拼成中段，looks_organic 的"短段占比"
    可能差一点（实测螺旋图 6 遍融合后 243 段、中位 0.14×D，但短段占比只有
    0.38）。这里只要求"段数多且中位段长仍远短于图幅"，放它进 vectorize ——
    真正的裁决是 vectorize 内部的"弯曲墨水占比 ≥0.55 且弯曲路径 ≥4"闸门：
    工程图的墨迹以直为主，进了也会被弹出来（只浪费 ~0.1s，无副作用）。
    门槛从 60 段放宽到 40：用户要求"尽可能用曲线"，挂钩/法兰这类"直边为主、
    带几条大弧"的图，碎弧弦被串弧收编后剩 40~60 段，也需要骨架矢量化兜底。
    """
    if not segs or D <= 0 or len(segs) < 40:
        return False
    Ls = sorted(math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)
    return Ls[len(Ls) // 2] < 0.20 * D
