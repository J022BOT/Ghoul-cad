"""曲线能力离线自测（curves.py / dxf_writer SPLINE、ELLIPSE）。

验证八件事：
  1) 开曲线插值：曲线**精确过每个点**（A9.1 全局插值，误差 < 1e-6）；
  2) 闭曲线插值：首尾无接缝（间隙≈0）、首尾切向一致（夹角≈0），且精确过点；
  3) spline 实体：细分点足够密，且紧贴原始轮廓（偏差 < 3% 包围盒对角线）；
  4) 该曲线的必须变曲线：大模型常见的"平滑折线"（无标记）会被识别并升级；
  5) 不该变的必须不变：五角星、正六边形、正方形、矩形一律保持折线；
  6) DXF 读写闭环：写出的 SPLINE / ELLIPSE 能被 ezdxf 读回，几何与参数一致；
  7) 退化输入不崩：2 点、重复点、全同点、空列表；
  8) 幂等：已经曲线化的实体再跑一遍 curveify 不变形。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import curves
import dxf_writer

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  OK   %-40s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-40s %s" % (name, detail))


def dist_point_seg(p, a, b):
    ax, ay = b[0] - a[0], b[1] - a[1]
    L2 = ax * ax + ay * ay
    if L2 < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / L2))
    return math.hypot(p[0] - (a[0] + ax * t), p[1] - (a[1] + ay * t))


def dist_to_poly(p, pts, closed):
    n = len(pts)
    segs = n if closed else n - 1
    return min(dist_point_seg(p, pts[i], pts[(i + 1) % n]) for i in range(segs))


def bbox_size(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)


# ---------------------------------------------------------------- 1) 开曲线插值
print("[1] 开曲线插值（A9.1 全局插值）")
P = [(0, 0), (10, 20), (30, 25), (50, 5), (60, 30), (72, 12)]
fit = curves.fit_open(P)
ub = curves._chord_params(P)
err = 0.0
for k, p in enumerate(P):
    q = curves._eval_open(fit["ctrl"], fit["knots"], fit["degree"], ub[k])
    err = max(err, math.hypot(q[0] - p[0], q[1] - p[1]))
check("曲线精确通过每个插值点", err < 1e-6, "最大误差 %.2e mm" % err)
check("节点数 = 控制点 + 次数 + 1",
      len(fit["knots"]) == len(fit["ctrl"]) + fit["degree"] + 1,
      "%d = %d + %d + 1" % (len(fit["knots"]), len(fit["ctrl"]), fit["degree"]))
check("端点被夹紧（0/1 各重复 4 次）",
      fit["knots"][:4] == [0, 0, 0, 0] and fit["knots"][-4:] == [1, 1, 1, 1])

# ---------------------------------------------------------------- 2) 闭曲线插值
print("[2] 闭曲线插值（周期反平滑 + 贝塞尔链）")
n = 16
ring = [(30 * math.cos(2 * math.pi * k / n) + 5,
         18 * math.sin(2 * math.pi * k / n) - 7) for k in range(n)]
fc = curves.fit_closed(ring)
err = 0.0
for k, p in enumerate(ring):
    q = curves._eval_open(fc["ctrl"], fc["knots"], 3, float(k))
    err = max(err, math.hypot(q[0] - p[0], q[1] - p[1]))
check("闭曲线精确通过每个点", err < 1e-6, "最大误差 %.2e mm" % err)
q0 = curves._eval_open(fc["ctrl"], fc["knots"], 3, 0.0)
qn = curves._eval_open(fc["ctrl"], fc["knots"], 3, float(n))
check("首尾闭合（无接缝）", math.hypot(q0[0] - qn[0], q0[1] - qn[1]) < 1e-9,
      "间隙 %.2e mm" % math.hypot(q0[0] - qn[0], q0[1] - qn[1]))
a = curves._eval_open(fc["ctrl"], fc["knots"], 3, 1e-4)
b = curves._eval_open(fc["ctrl"], fc["knots"], 3, float(n) - 1e-4)
t0 = (a[0] - q0[0], a[1] - q0[1])
t1 = (qn[0] - b[0], qn[1] - b[1])
cs = (t0[0] * t1[0] + t0[1] * t1[1]) / (math.hypot(*t0) * math.hypot(*t1) + 1e-12)
ang = math.degrees(math.acos(max(-1.0, min(1.0, cs))))
check("首尾切向连续（G1）", ang < 0.5, "夹角 %.3f°" % ang)
check("定义域就是 [0, n]（不会外推）",
      abs(fc["knots"][3] - 0.0) < 1e-9 and abs(fc["knots"][-4] - n) < 1e-9,
      "[%g, %g]" % (fc["knots"][3], fc["knots"][-4]))

# ------------------------------------------------------------- 3) spline 实体
print("[3] spline 实体：细分点密度与形状贴合")
ent = curves.spline_entity(ring, closed=True)
check("类型为 spline", ent["type"] == "spline")
check("细分点足够密（>100）", len(ent["points"]) > 100, "%d 点" % len(ent["points"]))
pts = curves.norm_points(ent["points"])
dev = max(dist_to_poly(p, ring, True) for p in pts) / bbox_size(ring)
check("细分辨点贴合原轮廓（<2%）", dev < 0.02, "偏差 %.2f%%" % (dev * 100))

# ------------------------------------------------------- 4) 该曲线化的要曲线化
print("[4] 平滑折线（大模型常见输出）应被升级成样条")
smooth = [[20 * math.cos(2 * math.pi * k / 24), 12 * math.sin(2 * math.pi * k / 24)]
          for k in range(24)]
out = curves.curveify([{"type": "polyline", "points": smooth, "closed": True}])
check("平滑闭合折线 → spline", out[0]["type"] == "spline", out[0]["type"])
wave = [[x, 10 * math.sin(x / 12.0)] for x in range(0, 120, 5)]
out = curves.curveify([{"type": "polyline", "points": wave, "closed": False}])
check("平滑开折线（波浪）→ spline", out[0]["type"] == "spline", out[0]["type"])
cusp = [[0, 0], [30, 0], [30, 30], [0, 30]]          # 四边形：全是折角
out = curves.curveify([{"type": "polyline", "points": cusp, "closed": True}])
check("四边形保持 polyline", out[0]["type"] == "polyline", out[0]["type"])

# --------------------------------------------------- 5) 不该变的必须保持不变
print("[5] 正多边形 / 星形必须保持折线（折角是设计意图）")
star = []
for k in range(10):
    R = 50 if k % 2 == 0 else 19.1
    a = math.radians(90 + 36 * k)
    star.append([R * math.cos(a), R * math.sin(a)])
for name, pts2 in (("五角星", star),
                   ("正六边形", [[50 * math.cos(math.radians(60 * k)),
                                 50 * math.sin(math.radians(60 * k))] for k in range(6)]),
                   ("正八边形", [[50 * math.cos(math.radians(45 * k)),
                                 50 * math.sin(math.radians(45 * k))] for k in range(8)])):
    o = curves.curveify([{"type": "polyline", "points": pts2, "closed": True}])
    check("%s 保持 polyline" % name, o[0]["type"] == "polyline", o[0]["type"])
# 顶点 ≥10 的"均匀分布"视为在近似圆 → 允许曲线化；真正的正十二边形由
# shapes.py 直出并带 no_curve，不会被磨圆
twelve = [[50 * math.cos(math.radians(30 * k)), 50 * math.sin(math.radians(30 * k))]
          for k in range(12)]
o = curves.curveify([{"type": "polyline", "points": twelve, "closed": True}])
check("12 点均匀折线（在近似圆）→ spline", o[0]["type"] == "spline", o[0]["type"])
o = curves.curveify([{"type": "polyline", "points": twelve, "closed": True,
                      "no_curve": True}])
check("带 no_curve 的正十二边形保持折线", o[0]["type"] == "polyline", o[0]["type"])
o = curves.curveify([{"type": "polyline", "points": star, "closed": True,
                      "smooth": True}])                # 显式要求曲线化才动
check("显式 smooth=True 才允许强制曲线化", o[0]["type"] == "spline", o[0]["type"])
o = curves.curveify([{"type": "polyline", "points": smooth, "closed": True,
                      "no_curve": True}])
check("no_curve 标记优先（shapes.py 用）", o[0]["type"] == "polyline", o[0]["type"])

# ------------------------------------------------------------ 6) DXF 读写闭环
print("[6] DXF 写出 / ezdxf 读回")
try:
    import ezdxf
    has_ez = True
except Exception:                                      # noqa: BLE001
    has_ez = False
if has_ez:
    d = dxf_writer.build_from_entities([
        {"type": "spline", "points": [list(p) for p in ring], "closed": True},
        {"type": "spline", "points": [[0, 0], [20, 15], [45, 5], [60, 25]],
         "closed": False},
        {"type": "ellipse", "center": [0, 90], "rx": 30, "ry": 12, "rotation": 25},
    ])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_curve_tmp.dxf")
    d.save(path)
    doc = ezdxf.readfile(path)
    types = [e.dxftype() for e in doc.modelspace()]
    check("DXF 里确实是 SPLINE + ELLIPSE",
          types.count("SPLINE") == 2 and types.count("ELLIPSE") == 1, str(types))
    sp = [e for e in doc.modelspace() if e.dxftype() == "SPLINE"][0]
    check("闭合样条被读回为 closed", bool(sp.closed))
    verts = [(v.x, v.y) for v in sp.flattening(0.3)]
    gap = math.hypot(verts[0][0] - verts[-1][0], verts[0][1] - verts[-1][1])
    check("读回后仍闭合（间隙≈0）", gap < 0.05, "%.4f mm" % gap)
    dev = max(dist_to_poly(p, ring, True) for p in verts) / bbox_size(ring)
    check("读回几何贴合原轮廓（<2%）", dev < 0.02, "偏差 %.2f%%" % (dev * 100))
    el = [e for e in doc.modelspace() if e.dxftype() == "ELLIPSE"][0]
    rx = math.hypot(el.dxf.major_axis.x, el.dxf.major_axis.y)
    check("椭圆长轴/短轴比正确", abs(rx - 30) < 0.01 and abs(rx * el.dxf.ratio - 12) < 0.01,
          "rx=%.2f ry=%.2f" % (rx, rx * el.dxf.ratio))
    back = dxf_writer.parse_dxf_to_entities(path)
    kinds = [e["type"] for e in back]
    check("反向解析保留曲线类型", kinds.count("spline") == 2 and kinds.count("ellipse") == 1,
          str(kinds))
    os.remove(path)

# ------------------------------------------------------------- 7) 退化输入
print("[7] 退化输入不崩")
for name, bad in (("2 点折线", [(0, 0), (10, 10)]),
                  ("重复点", [(5, 5)] * 8),
                  ("只有一个点", [(3, 4)]),
                  ("空列表", [])):
    try:
        e = curves.spline_entity(bad, closed=True)
        ok = isinstance(e, dict) and e.get("type") in ("spline", "polyline")
        check("%s 安全返回" % name, ok, e.get("type"))
    except Exception as ex:                            # noqa: BLE001
        check("%s 安全返回" % name, False, repr(ex))
check("curveify(None) 不崩", curves.curveify(None) == [])
check("curveify 保留未知类型", curves.curveify([{"type": "text", "p": [0, 0]}])[0]["type"]
      == "text")

# ------------------------------------------------------------------ 8) 幂等
print("[8] 幂等：已曲线化的实体再跑一遍不变形")
once = curves.curveify([{"type": "polyline", "points": smooth, "closed": True}])
twice = curves.curveify(once)
check("类型仍为 spline", twice[0]["type"] == "spline", twice[0]["type"])
d1 = max(dist_to_poly(p, curves.norm_points(once[0]["points"]), True)
         for p in curves.norm_points(twice[0]["points"])) / bbox_size(curves.norm_points(twice[0]["points"]))
check("形状不再漂移（<0.5%）", d1 < 0.005, "偏差 %.3f%%" % (d1 * 100))

# ------------------------------------- 9) 一串短直线拼的弧 → 串成一条样条
print("[9] 短直线串链（模型画弧的常见手法）")
R = 40.0
ring20 = [(R * math.cos(2 * math.pi * k / 20), R * math.sin(2 * math.pi * k / 20))
          for k in range(20)]
lines20 = [{"type": "line", "p1": list(ring20[k]), "p2": list(ring20[(k + 1) % 20])}
           for k in range(20)]
o = curves.chain_lines_to_curves(lines20)
check("20 条短线拼的圆 → 1 条 spline", len(o) == 1 and o[0]["type"] == "spline",
      "%d 个实体：%s" % (len(o), [e["type"] for e in o]))
if o and o[0]["type"] == "spline":
    e_r = max(abs(math.hypot(x, y) - R) for x, y in o[0]["points"])
    check("曲线化后仍是圆（<1%）", e_r < 0.01 * R, "最大径向误差 %.4f mm" % e_r)
    check("输出的样条是闭合的", bool(o[0].get("closed")))

# 每边 10 条短线的矩形框：4 个 90° 拐角必须断开，且直边不该变曲线
sq = []
for k in range(10):
    sq.append(((k, 0), (k + 1, 0)))
    sq.append(((10, k), (10, k + 1)))
    sq.append(((10 - k, 10), (9 - k, 10)))
    sq.append(((0, 10 - k), (0, 9 - k)))
sq_lines = [{"type": "line", "p1": [float(a[0]), float(a[1])],
             "p2": [float(b[0]), float(b[1])]} for a, b in sq]
o = curves.chain_lines_to_curves(sq_lines)
check("40 条短线拼的方框 → 全部保持 line",
      all(e["type"] == "line" for e in o) and len(o) == 40,
      "%d 个实体" % len(o))

# 圆弧接直边（圆角弯钩）：弧变曲线，直边必须保留——接缝是尖角，不能被抹圆
hook = [{"type": "line", "p1": [100.0, 0.0], "p2": [100.0, 20.0]}]
for k in range(16):
    a0 = math.radians(k * 90.0 / 16)
    a1 = math.radians((k + 1) * 90.0 / 16)
    hook.append({"type": "line",
                 "p1": [20 * math.cos(a0), 20 + 20 * math.sin(a0)],
                 "p2": [20 * math.cos(a1), 20 + 20 * math.sin(a1)]})
hook.append({"type": "line", "p1": [0.0, 40.0], "p2": [0.0, 100.0]})
o = curves.chain_lines_to_curves(hook)
kinds = [e["type"] for e in o]
check("弧+直边：弧变 spline、两条直边保留",
      kinds.count("spline") == 1 and kinds.count("line") == 2, str(kinds))

# 不同图层不串链：两个半圆弧各自成一条曲线（不能被合并成一条整圆）
two_layer = ([{"type": "line", "p1": list(ring20[k]), "p2": list(ring20[(k + 1) % 20]),
               "layer": "A"} for k in range(10)]
             + [{"type": "line", "p1": list(ring20[k]), "p2": list(ring20[(k + 1) % 20]),
                 "layer": "B"} for k in range(10, 20)])
o = curves.chain_lines_to_curves(two_layer)
check("跨图层不串链（半圈各自成曲线、图层保留）",
      len(o) == 2 and [e["type"] for e in o] == ["spline", "spline"]
      and sorted(e["layer"] for e in o) == ["A", "B"],
      "%d 个实体：%s" % (len(o), [(e["type"], e.get("layer")) for e in o]))

# 非直线实体必须原样保留
mixed = [{"type": "circle", "center": [0, 0], "radius": 5}] + lines20
o = curves.chain_lines_to_curves(mixed)
check("不参与串链的实体原样保留",
      o[0]["type"] == "circle" and len([e for e in o if e["type"] == "spline"]) == 1,
      "%d 个实体" % len(o))
check("chain_lines_to_curves([]) 不崩", curves.chain_lines_to_curves([]) == [])
check("chain_lines_to_curves(None) 不崩", curves.chain_lines_to_curves(None) == [])

# 回归：带"首尾重合点"的闭合轮廓也必须能被判为曲线（否则 concepts 的闭合
# 轮廓永远升级不了——a=pts[0]、b=pts[-1] 同点，L=0 被"近似直线"判据误杀）
ring_dup = [[30 * math.cos(2 * math.pi * k / 24), 30 * math.sin(2 * math.pi * k / 24)]
            for k in range(24)]
ring_dup.append(list(ring_dup[0]))
check("带重复首尾点的闭合折线仍判为曲线",
      curves.should_curve(ring_dup, closed=True))

print("\n===== 曲线测试：%d 通过 / %d 失败 =====" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
