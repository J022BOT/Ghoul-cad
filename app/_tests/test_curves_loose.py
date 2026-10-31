# -*- coding: utf-8 -*-
"""test_curves_loose.py — 宽松曲线化（loose）模式测试。

用户要求"尽可能用曲线、少用直线"：cv-lines 识别结果开 loose 通道后，
首尾相接的碎弦链、弯曲 polyline 应升级成样条；单条直线、直角链不动。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import curves

PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name} {extra}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def arc_chain(cx, cy, r, a0, a1, n, jitter=0.0):
    """圆弧上取 n 段弦，返回 line 实体列表（首尾相接）。"""
    ents = []
    pts = [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
            cy + r * math.sin(math.radians(a0 + (a1 - a0) * k / n)))
           for k in range(n + 1)]
    for i in range(n):
        ents.append({"type": "line", "p1": list(pts[i]), "p2": list(pts[i + 1])})
    return ents


def main():
    import random
    random.seed(42)

    # ---- 1) 4 段缓弯链：保守模式不升级，loose 升级 ----
    ents4 = arc_chain(0, 0, 50, 0, 80, 4)
    keep = curves.chain_lines_to_curves(ents4, tol=0.05, min_seg=5)
    check("保守模式：4 段链不升级", all(e["type"] == "line" for e in keep))
    loose = curves.chain_lines_to_curves(ents4, tol=0.05, min_seg=3, loose=True)
    n_spl = sum(1 for e in loose if e["type"] == "spline")
    check("loose：4 段弧弦链升级成 1 条样条", n_spl == 1,
          f"(spline={n_spl}, 其余直线={sum(1 for e in loose if e['type'] == 'line')})")
    if n_spl == 1:
        spl = next(e for e in loose if e["type"] == "spline")
        # 升级后的样条应贴着原圆弧（半径误差 < 3%）
        err = max(abs(math.hypot(p[0], p[1]) - 50.0) for p in spl["points"])
        check("loose 样条几何贴原弧", err < 1.5, f"(最大径向误差 {err:.3f})")

    # ---- 2) 3 段直角链：loose 也不许升级（工程图的边）----
    corner = [{"type": "line", "p1": [0, 0], "p2": [30, 0]},
              {"type": "line", "p1": [30, 0], "p2": [30, 30]},
              {"type": "line", "p1": [30, 30], "p2": [60, 30]}]
    out2 = curves.chain_lines_to_curves(corner, tol=0.05, min_seg=3, loose=True)
    check("loose：直角链保持直线", all(e["type"] == "line" for e in out2))

    # ---- 3) 单条直线：怎么都不动 ----
    one = [{"type": "line", "p1": [0, 0], "p2": [100, 0]}]
    out3 = curves.chain_lines_to_curves(one, tol=0.05, min_seg=3, loose=True)
    check("loose：单条直线原样保留", len(out3) == 1 and out3[0]["type"] == "line")

    # ---- 4) 弯曲 polyline：loose curveify 升级，保守不升 ----
    import random as _r
    _r.seed(7)
    # 缓波（相邻转角 ~29°）：保守模式就该升级
    poly_gentle = [{"type": "polyline",
                    "points": [[k * 10.0, 20.0 * math.sin(k * 0.5)] for k in range(6)]}]
    c_cons = curves.curveify([dict(e) for e in poly_gentle])
    c_loose = curves.curveify([dict(e) for e in poly_gentle], loose=True)
    check("保守 curveify：6 点缓弯 polyline 升级", c_cons[0]["type"] == "spline")
    check("loose curveify：同链升级", c_loose[0]["type"] == "spline")
    # 陡波（相邻转角 ~52°，超过保守的 50° 上限）：只有 loose 肯收
    poly_steep = [{"type": "polyline",
                   "points": [[k * 10.0, 20.0 * math.sin(k * 0.9)] for k in range(6)]}]
    c_cons2 = curves.curveify([dict(e) for e in poly_steep])
    c_loose2 = curves.curveify([dict(e) for e in poly_steep], loose=True)
    check("保守 curveify：陡弯折线保持 polyline", c_cons2[0]["type"] != "spline")
    check("loose curveify：陡弯折线也升级", c_loose2[0]["type"] == "spline")

    # 直角 polyline（工程图链）：loose 不许磨圆
    rect_poly = [{"type": "polyline",
                  "points": [[0, 0], [80, 0], [80, 60], [0, 60]], "closed": False}]
    c_rect = curves.curveify([dict(rect_poly[0])], loose=True)
    check("loose curveify：直角折线保持 polyline", c_rect[0]["type"] != "spline")

    # ---- 5) 完整圆的 8 段弦：loose 串成闭合样条，直线数 8→0 ----
    circ8 = arc_chain(0, 0, 40, 0, 330, 8)  # 8 段覆盖 330°，留 30° 缺口避免闭合判定干扰
    out5 = curves.chain_lines_to_curves(circ8, tol=0.05, min_seg=3, loose=True)
    n_spl5 = sum(1 for e in out5 if e["type"] == "spline")
    n_line5 = sum(1 for e in out5 if e["type"] == "line")
    check("loose：8 段圆弦链升级", n_spl5 == 1 and n_line5 == 0,
          f"(spline={n_spl5}, line={n_line5})")

    # ---- 6) loose 判据参数可追溯 ----
    check("LOOSE_KW 参数齐全",
          curves.LOOSE_KW["min_open"] == 4 and curves.LOOSE_KW["min_closed"] == 5
          and curves.LOOSE_KW["sharp_ratio"] > 0.2)

    print(f"\n===== 宽松曲线化测试：{PASS} 通过 / {FAIL} 失败 =====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
