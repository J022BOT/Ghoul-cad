# -*- coding: utf-8 -*-
"""homotope_bridge.py — 用 Homotope 引擎（@homotope/engine，MIT，纯本地）辅助识图。

为什么引入
----------
OpenCV 路线（line_extractor）靠"霍夫直线 + 圆检测 + 骨架"硬拼几何，遇到这三类图
会明显吃亏：
  · 手绘/照片/有抗锯齿软边的图 —— 霍夫对模糊边会出双线或断线；
  · 大量自由曲线（云纹、螺旋、logo）—— 只能切成几百段短弦再靠有机线稿兜底；
  · 彩色/灰底图 —— 墨迹场判定容易失手。

Homotope 是"拓扑优先"的矢量化引擎：先在 OKLab 里量化颜色、按区域分割，每条**共享
边界只拟合一次**，所以相邻色块既不重叠也无缝隙；`centerline` 预设还能把笔画直接描
成**单条中线开路径**（而不是线两侧的双轮廓），正好就是 CAD 线稿要的东西。

它的输出是 SVG，我们解析出点列，转成与 cv-lines 一致的实体结构，再交给
``cad_agent`` 的曲线化通道（cv-lines 现在恒走 loose 曲线化）产出真样条。

依赖与降级
----------
引擎是纯 TypeScript（零运行时依赖，1.7MB），但需要 **Node ≥ 22** 才能跑：
  · 先找包内 ``tools/node/node.exe``（若打包时自带）；
  · 再找系统 PATH 里的 ``node``。
两者都没有时 ``available()`` 返回 False，调用方自动退回 OpenCV 路线，不影响使用。
像素数据自始至终在本机，不上传任何服务器。
"""

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

try:
    import portable
except Exception:                                        # pragma: no cover
    portable = None

#: 本模块给 cad_agent 用的方法名（与 concepts/shapes/vlm/cv-lines 并列）
METHOD = "homotope"

#: 预设 → 说明。centerline 最贴合线稿；line-art 是填充轮廓（线变双边界）
PRESETS = ("centerline", "line-art", "black-white", "illustration",
           "logo", "pixel-art", "photo")

#: 默认预设：中线描摹
DEFAULT_PRESET = "centerline"

_SVG_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ 环境探测
def _pkg_root() -> str:
    try:
        return portable.PKG_ROOT
    except Exception:                                     # noqa: BLE001
        return _SVG_DIR


def _tools_dir() -> str:
    return os.path.join(_pkg_root(), "tools", "homotope")


def find_node() -> str:
    """Node 可执行文件：包内自带优先，其次系统 PATH。找不到返回 ''。"""
    cands = [
        os.path.join(_pkg_root(), "tools", "node", "node.exe"),
        os.path.join(_pkg_root(), "tools", "node", "node"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    p = shutil.which("node") or shutil.which("node.exe")
    return p or ""


def cli_path() -> str:
    """homotope_cli.mjs 的绝对路径（包内 tools/homotope/）。"""
    return os.path.join(_tools_dir(), "homotope_cli.mjs")


#: Homotope 需要 Node ≥22（用了较新的 ESM / 语法特性）
MIN_NODE_MAJOR = 22

_NODE_VER_CACHE = {"path": "", "major": None}


def node_major(node: str = None) -> int:
    """返回 node 主版本号；查不出来返回 None（宽容放行，不因探测失败禁用）。"""
    node = node or find_node()
    if not node:
        return None
    if _NODE_VER_CACHE.get("path") == node and _NODE_VER_CACHE["major"] is not None:
        return _NODE_VER_CACHE["major"]
    major = None
    try:
        pr = subprocess.run([node, "--version"], capture_output=True, timeout=20)
        v = (pr.stdout or b"").decode("utf-8", "ignore").strip().lstrip("vV")
        if v:
            major = int(v.split(".")[0])
    except Exception:                                     # noqa: BLE001
        major = None
    if major is not None:
        _NODE_VER_CACHE["path"] = node
        _NODE_VER_CACHE["major"] = major
    return major


def available() -> bool:
    """引擎是否可用：node（≥MIN_NODE_MAJOR）+ CLI + node_modules 三者齐备。"""
    if not (find_node() and os.path.isfile(cli_path())
            and os.path.isdir(os.path.join(_tools_dir(), "node_modules"))):
        return False
    mj = node_major()
    return mj is None or mj >= MIN_NODE_MAJOR


def reason_unavailable() -> str:
    if not find_node():
        return "未找到 Node（需要 ≥22；可把 node.exe 放到 tools/node/ 或装在系统 PATH）"
    if not os.path.isfile(cli_path()):
        return "缺少 tools/homotope/homotope_cli.mjs"
    if not os.path.isdir(os.path.join(_tools_dir(), "node_modules")):
        return "缺少 tools/homotope/node_modules（npm install @homotope/engine pngjs jpeg-js）"
    mj = node_major()
    if mj is not None and mj < MIN_NODE_MAJOR:
        return "Node 版本过低（当前 v%s，需要 ≥v%d）；请换用 tools/node/node.exe 或升级系统 Node" \
               % (mj, MIN_NODE_MAJOR)
    return ""


# ------------------------------------------------------------- 调用 CLI
def vectorize_svg(image_path: str, preset: str = DEFAULT_PRESET,
                  timeout: float = 120.0):
    """跑 Homotope，返回 (svg_text, stats_dict)；失败返回 ('', {'ok':False,...})。

    ``stats`` 含 paths/colors/ms 等，供提示语与诊断用。
    """
    if not available():
        return "", {"ok": False, "error": reason_unavailable()}
    if not image_path or not os.path.isfile(image_path):
        return "", {"ok": False, "error": "图片不存在：%s" % image_path}
    preset = preset if preset in PRESETS else DEFAULT_PRESET
    image_path = os.path.abspath(image_path)
    cmd = [find_node(), cli_path(), "--in", image_path, "--preset", preset]
    try:
        # cwd 固定到引擎目录：U 盘环境下 python 进程的 cwd 可能是别的盘，
        # 固定后模块解析与相对路径行为在任何机器上一致。
        pr = subprocess.run(cmd, capture_output=True, timeout=timeout,
                            cwd=_tools_dir())
    except subprocess.TimeoutExpired:
        return "", {"ok": False, "error": "Homotope 超时（%ss）" % timeout}
    except Exception as e:                                # noqa: BLE001
        return "", {"ok": False, "error": str(e)}
    err = (pr.stderr or b"").decode("utf-8", "ignore")
    svg = (pr.stdout or b"").decode("utf-8", "ignore")
    stats = {"ok": False}
    m = re.search(r"__STATS__\s*(\{.*?\})\s*$", err, re.S)
    if not m:
        m = re.search(r"__STATS__\s*(\{.*?\})", err, re.S)
    if m:
        try:
            import json
            stats = json.loads(m.group(1))
        except Exception:                                 # noqa: BLE001
            stats = {"ok": False, "error": "stats 解析失败"}
    if pr.returncode != 0 and not svg:
        stats.setdefault("error", "CLI 退出码 %s" % pr.returncode)
        stats["ok"] = False
        return "", stats
    return svg, stats


# ------------------------------------------------------- SVG path 解析
_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_CMD = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])")


def _numbers(chunk: str):
    return [float(v) for v in _NUM.findall(chunk)]


def _path_tokens(d: str):
    """把 path 的 d 属性切成 [(cmd, [操作数...]), ...]。"""
    out = []
    cur = None
    buf = ""
    for ch in d:
        if _CMD.match(ch):
            # 注意：Z/z 没有操作数，但它是真命令，必须无条件记下来。
            # 写成 `if cur and buf.strip()` 的话，"...03Zm300" 里的 Z 会被吞掉，
            # 第一条子路径就永远闭不上。
            if cur:
                out.append((cur, _numbers(buf)))
            cur = ch
            buf = ""
        else:
            buf += ch
    if cur is not None:
        out.append((cur, _numbers(buf)))
    return out


def _arc_points(p0, rx, ry, phi_deg, large, sweep, p1, steps=24):
    """SVG 椭圆弧（A）→ 点列。按 W3C 附录 F 的端点参数化实现。"""
    x1, y1 = p0
    x2, y2 = p1
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-9 or ry < 1e-9:
        return []
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cp * dx2 + sp * dy2
    y1p = -sp * dx2 + cp * dy2
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:                       # 半径太小：按比例放大到刚好够
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    co = math.sqrt(max(0.0, num / den)) if den > 1e-12 else 0.0
    if large == sweep:
        co = -co
    cxp, cyp = co * rx * y1p / ry, -co * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def ang(ux, uy, vx, vy):
        d = math.hypot(ux, uy) * math.hypot(vx, vy)
        if d < 1e-12:
            return 0.0
        c = max(-1.0, min(1.0, (ux * vx + uy * vy) / d))
        a = math.acos(c)
        return -a if (ux * vy - uy * vx) < 0 else a

    th1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dth = ang((x1p - cxp) / rx, (y1p - cyp) / ry,
              (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dth > 0:
        dth -= 2 * math.pi
    elif sweep and dth < 0:
        dth += 2 * math.pi
    pts = []
    for k in range(1, steps + 1):
        th = th1 + dth * k / steps
        x = rx * math.cos(th)
        y = ry * math.sin(th)
        pts.append((cx + cp * x - sp * y, cy + sp * x + cp * y))
    return pts


def _bezier3(p0, c1, c2, p1, steps=16):
    out = []
    for k in range(1, steps + 1):
        t = k / steps
        mt = 1 - t
        x = (mt ** 3 * p0[0] + 3 * mt * mt * t * c1[0]
             + 3 * mt * t * t * c2[0] + t ** 3 * p1[0])
        y = (mt ** 3 * p0[1] + 3 * mt * mt * t * c1[1]
             + 3 * mt * t * t * c2[1] + t ** 3 * p1[1])
        out.append((x, y))
    return out


def _bezier2(p0, c, p1, steps=12):
    out = []
    for k in range(1, steps + 1):
        t = k / steps
        mt = 1 - t
        out.append((mt * mt * p0[0] + 2 * mt * t * c[0] + t * t * p1[0],
                    mt * mt * p0[1] + 2 * mt * t * c[1] + t * t * p1[1]))
    return out


def parse_path_d(d: str, steps: int = 16):
    """一条 path 的 d → 若干条子路径点列 [[(x,y),...], ...]（像素坐标，y 向下）。

    M/m 开新子路径；Z/z 闭合；曲线命令按 steps 细分成折线点；A 按端点参数化。
    """
    subs = []
    cur = None
    cur_pt = (0.0, 0.0)
    start = (0.0, 0.0)
    prev_c2 = None                      # S/s 反射控制点
    prev_q = None                       # T/t 反射控制点
    for cmd, args in _path_tokens(d or ""):
        rel = cmd.islower()
        C = cmd.upper()
        i = 0
        if C == "Z":
            if cur is not None and len(cur) >= 2:
                cur.append(cur[0])
                subs.append(cur)
            # Z 之后当前点回到子路径起点（SVG 规范）：后面紧跟的相对 m/l/c 都
            # 以它为基准。不复位的话第二条子路径会整体平移（实测偏移 400px）。
            cur_pt = start
            cur, prev_c2, prev_q = None, None, None
            continue
        if C == "M":
            while i + 1 < len(args) + 1 and i + 2 <= len(args):
                x, y = args[i], args[i + 1]
                i += 2
                if rel:
                    x += cur_pt[0]
                    y += cur_pt[1]
                if cur is not None and len(cur) >= 2:
                    subs.append(cur)
                cur = [(x, y)]
                cur_pt = start = (x, y)
                prev_c2 = prev_q = None
                cmd = "l" if rel else "L"       # 后续数字按 L 处理
                C = "L"
                continue
            continue
        while True:
            if C == "L":
                if i + 2 > len(args):
                    break
                x, y = args[i], args[i + 1]
                i += 2
                if rel:
                    x += cur_pt[0]
                    y += cur_pt[1]
                if cur is None:
                    cur = [cur_pt]
                cur.append((x, y))
                cur_pt = (x, y)
            elif C == "H":
                if i + 1 > len(args):
                    break
                x = args[i] + (cur_pt[0] if rel else 0.0)
                i += 1
                y = cur_pt[1]
                if cur is None:
                    cur = [cur_pt]
                cur.append((x, y))
                cur_pt = (x, y)
            elif C == "V":
                if i + 1 > len(args):
                    break
                y = args[i] + (cur_pt[1] if rel else 0.0)
                i += 1
                x = cur_pt[0]
                if cur is None:
                    cur = [cur_pt]
                cur.append((x, y))
                cur_pt = (x, y)
            elif C == "C":
                if i + 6 > len(args):
                    break
                c1 = (args[i], args[i + 1])
                c2 = (args[i + 2], args[i + 3])
                p1 = (args[i + 4], args[i + 5])
                i += 6
                if rel:
                    c1 = (c1[0] + cur_pt[0], c1[1] + cur_pt[1])
                    c2 = (c2[0] + cur_pt[0], c2[1] + cur_pt[1])
                    p1 = (p1[0] + cur_pt[0], p1[1] + cur_pt[1])
                if cur is None:
                    cur = [cur_pt]
                cur += _bezier3(cur_pt, c1, c2, p1, steps)
                prev_c2, prev_q = c2, None
                cur_pt = p1
            elif C == "S":
                if i + 4 > len(args):
                    break
                c2 = (args[i], args[i + 1])
                p1 = (args[i + 2], args[i + 3])
                i += 4
                if rel:
                    c2 = (c2[0] + cur_pt[0], c2[1] + cur_pt[1])
                    p1 = (p1[0] + cur_pt[0], p1[1] + cur_pt[1])
                c1 = (2 * cur_pt[0] - prev_c2[0], 2 * cur_pt[1] - prev_c2[1]) \
                    if prev_c2 else cur_pt
                if cur is None:
                    cur = [cur_pt]
                cur += _bezier3(cur_pt, c1, c2, p1, steps)
                prev_c2, prev_q = c2, None
                cur_pt = p1
            elif C == "Q":
                if i + 4 > len(args):
                    break
                c = (args[i], args[i + 1])
                p1 = (args[i + 2], args[i + 3])
                i += 4
                if rel:
                    c = (c[0] + cur_pt[0], c[1] + cur_pt[1])
                    p1 = (p1[0] + cur_pt[0], p1[1] + cur_pt[1])
                if cur is None:
                    cur = [cur_pt]
                cur += _bezier2(cur_pt, c, p1, steps)
                prev_q, prev_c2 = c, None
                cur_pt = p1
            elif C == "T":
                if i + 2 > len(args):
                    break
                p1 = (args[i], args[i + 1])
                i += 2
                if rel:
                    p1 = (p1[0] + cur_pt[0], p1[1] + cur_pt[1])
                c = (2 * cur_pt[0] - prev_q[0], 2 * cur_pt[1] - prev_q[1]) \
                    if prev_q else cur_pt
                if cur is None:
                    cur = [cur_pt]
                cur += _bezier2(cur_pt, c, p1, steps)
                prev_q, prev_c2 = c, None
                cur_pt = p1
            elif C == "A":
                if i + 7 > len(args):
                    break
                rx, ry = args[i], args[i + 1]
                phi = args[i + 2]
                large = int(args[i + 3])
                sweep = int(args[i + 4])
                x, y = args[i + 5], args[i + 6]
                i += 7
                if rel:
                    x += cur_pt[0]
                    y += cur_pt[1]
                if cur is None:
                    cur = [cur_pt]
                cur += _arc_points(cur_pt, rx, ry, phi, large, sweep, (x, y))
                prev_c2 = prev_q = None
                cur_pt = (x, y)
            else:                                     # 未知命令：跳过
                break
    if cur is not None and len(cur) >= 2:
        subs.append(cur)
    return subs


def svg_to_subpaths(svg: str, steps: int = 16):
    """SVG 文本 → 全部子路径点列（像素坐标，y 向下）。"""
    if not svg:
        return []
    out = []
    for m in re.finditer(r"<path\b[^>]*>", svg, re.I):
        tag = m.group(0)
        dm = re.search(r'\sd\s*=\s*"([^"]*)"', tag) or \
            re.search(r"\sd\s*=\s*'([^']*)'", tag)
        if not dm:
            continue
        out += parse_path_d(dm.group(1), steps)
    return out


# --------------------------------------------------- 子路径 → cad 实体（毫米）
def subpaths_to_entities(subpaths, max_dim: float = 210.0,
                         view_box=None, min_len_ratio: float = 0.004):
    """点列（像素，y 向下）→ cad_agent 实体列表（毫米，y 向上，居中）。

    与 line_extractor 的坐标约定一致：整体缩放使最长边 = max_dim，y 翻转。
    太短的碎路径（< min_len_ratio × 图幅）直接丢掉，避免噪点变图元。
    """
    pts_all = [p for sp in subpaths for p in sp]
    if not pts_all:
        return []
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    if view_box:
        # 有 viewBox 就用它算图幅（否则按点包围盒，缩放会偏大）
        vx, vy, vw, vh = view_box
        xs = xs + [vx, vx + vw]
        ys = ys + [vy, vy + vh]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    bw_ = max(max_x - min_x, 1.0)
    bh_ = max(max_y - min_y, 1.0)
    scale = max_dim / max(bw_, bh_)

    def T(p):
        return (round((p[0] - (min_x + max_x) / 2) * scale, 3),
                round(((min_y + max_y) / 2 - p[1]) * scale, 3))

    D = math.hypot(max_x - min_x, max_y - min_y)
    min_len = max(1.0, min_len_ratio * D)
    ents = []
    for sp in subpaths:
        if len(sp) < 2:
            continue
        L = sum(math.hypot(sp[i + 1][0] - sp[i][0], sp[i + 1][1] - sp[i][1])
                for i in range(len(sp) - 1))
        if L < min_len:
            continue
        closed = len(sp) > 3 and \
            math.hypot(sp[0][0] - sp[-1][0], sp[0][1] - sp[-1][1]) <= max(1.0, 0.004 * D)
        pts = [T(p) for p in (sp[:-1] if closed else sp)]
        # 去掉连续重复点（细分后常见）
        ded = [pts[0]]
        for q in pts[1:]:
            if math.hypot(q[0] - ded[-1][0], q[1] - ded[-1][1]) > 1e-6:
                ded.append(q)
        if len(ded) < 2:
            continue
        if len(ded) == 2:
            ents.append({"type": "line", "p1": list(ded[0]), "p2": list(ded[1])})
        else:
            ents.append({"type": "polyline", "points": [list(p) for p in ded],
                         "closed": closed})
    return ents


def _view_box(svg: str):
    m = re.search(r'viewBox\s*=\s*"([^"]*)"', svg or "")
    if not m:
        return None
    v = _numbers(m.group(1))
    if len(v) == 4:
        return (v[0], v[1], v[2], v[3])
    return None


def vectorize_entities(image_path: str, max_dim: float = 210.0,
                       preset: str = DEFAULT_PRESET, timeout: float = 120.0):
    """一步到位：图片 → (实体列表, 统计字典)。引擎不可用时返回 ([], stats)。"""
    svg, stats = vectorize_svg(image_path, preset=preset, timeout=timeout)
    if not svg:
        return [], stats
    subs = svg_to_subpaths(svg)
    ents = subpaths_to_entities(subs, max_dim=max_dim, view_box=_view_box(svg))
    stats["entities"] = len(ents)
    stats["subpaths"] = len(subs)
    return ents, stats


def build_spec(image_path: str, instruction: str = "", max_dim: float = 210.0,
               preset: str = DEFAULT_PRESET, name: str = "homotope",
               timeout: float = 120.0):
    """返回与 line_extractor.extract_spec 同构的 spec 字典，供 cad_agent 直接用。"""
    ents, stats = vectorize_entities(image_path, max_dim=max_dim,
                                     preset=preset, timeout=timeout)
    counts = {}
    for e in ents:
        t = e.get("type", "?")
        counts[t] = counts.get(t, 0) + 1
    ok = bool(ents) and stats.get("ok")
    return {"name": name or "homotope", "unit": "mm", "entities": ents,
            "_meta": {"ok": bool(ok), "method": METHOD, "counts": counts,
                      "homotope": stats,
                      "reason": "" if ok else (stats.get("error") or "未识别到图元")}}


# ------------------------------------------------------------------ 自检
if __name__ == "__main__":                                 # pragma: no cover
    print("node      :", find_node() or "(未找到)")
    print("cli       :", cli_path())
    print("available :", available(), reason_unavailable())
    if len(sys.argv) > 1:
        ents, st = vectorize_entities(sys.argv[1],
                                      preset=sys.argv[2] if len(sys.argv) > 2
                                      else DEFAULT_PRESET)
        print("stats     :", st)
        for e in ents[:20]:
            if e["type"] == "line":
                print("   line", e["p1"], "->", e["p2"])
            else:
                print("   %s %d 点 closed=%s 首=%s"
                      % (e["type"], len(e["points"]), e.get("closed"),
                         e["points"][0]))
        print("共 %d 个图元" % len(ents))
