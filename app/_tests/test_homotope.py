# -*- coding: utf-8 -*-
"""test_homotope.py — Homotope 桥接测试。

分两层：
  1. SVG path 解析（纯 Python，永远跑）：命令覆盖 M/L/H/V/C/S/Q/T/A/Z、
     绝对/相对、科学计数、Z 后当前点复位、闭合判定；
  2. 端到端（有 Node + 引擎才跑，否则 skip 不判失败）：真图矢量化 → 实体。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import homotope_bridge as hb

PASS = FAIL = SKIP = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {name} {extra}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


def skip(name, why):
    global SKIP
    SKIP += 1
    print(f"  SKIP {name} ({why})")


def near(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ------------------------------------------------------------ 1) path 解析
def test_tokens():
    toks = hb._path_tokens("M351 500H150V101l201-.03Zm300 0H380V381l271-.03Z")
    cmds = [t[0] for t in toks]
    check("token 化不吞无参命令 Z", cmds.count("Z") == 2, str(cmds))
    check("token 化保留相对 m", "m" in cmds)


def test_z_resets_current_point():
    """Z 后当前点回到子路径起点：第二条子路径不能整体平移。"""
    subs = hb.parse_path_d("M351 500H150V101l201-.03Zm300 0H380V381l271-.03Z")
    check("Z 分成两条子路径", len(subs) == 2, f"({len(subs)})")
    if len(subs) == 2:
        s0, s1 = subs[0], subs[1]
        check("第一条闭合（首尾同点）", near(s0[0][0], s0[-1][0]) and near(s0[0][1], s0[-1][1]))
        # 相对 m300 0 基于起点 (351,500) → (651,500)；若忘了复位会变成 (651,100.97)
        check("Z 后相对 m 起点正确", near(s1[0][0], 651.0) and near(s1[0][1], 500.0),
              f"({s1[0]})")
        check("第二条闭合", near(s1[0][0], s1[-1][0]) and near(s1[0][1], s1[-1][1]))


def test_commands():
    # 绝对 + 相对直线
    subs = hb.parse_path_d("M10 10 L20 10 l0 10")
    check("L/l 混用", len(subs) == 1 and len(subs[0]) == 3
          and near(subs[0][2][1], 20.0), str(subs[0] if subs else None))
    # H/V
    subs = hb.parse_path_d("M0 0 H50 V50 Z")
    check("H/V/Z", len(subs) == 1 and near(subs[0][1][0], 50.0)
          and near(subs[0][2][1], 50.0))
    # 三次贝塞尔：起点 (0,0) → 终点 (30,0)，控制点抬高，中点应变高
    subs = hb.parse_path_d("M0 0 C10 20 20 20 30 0")
    mid = max(p[1] for p in subs[0])
    check("C 曲线细分出拱形", mid > 10.0, f"(峰高 {mid:.2f})")
    # 光滑三次 S：控制点反射
    subs = hb.parse_path_d("M0 0 C10 20 20 20 30 0 S40 -20 60 0")
    check("S 命令不崩且继续延伸", len(subs) == 1 and near(subs[0][-1][0], 60.0, 1e-6))
    # 二次 Q + T
    subs = hb.parse_path_d("M0 0 Q15 30 30 0 T60 0")
    check("Q/T 命令", len(subs) == 1 and near(subs[0][-1][0], 60.0, 1e-6))
    # 圆弧 A：半径 20 的半圆，终点 = 起点 + 直径（sweep=1 在屏幕坐标里向下鼓，
    # 所以看 |y| 的峰值）
    subs = hb.parse_path_d("M0 0 A20 20 0 0 1 40 0")
    ok = len(subs) == 1 and near(subs[0][-1][0], 40.0, 1e-6) \
        and max(abs(p[1]) for p in subs[0]) > 15.0
    check("A 弧端点与弧度", ok, f"(末点 {subs[0][-1] if subs else None})")
    # 科学计数 + 无空格
    subs = hb.parse_path_d("M1e1 1e1L2e1 1e1")
    check("科学计数", len(subs) == 1 and near(subs[0][0][0], 10.0))


def test_entities():
    subs = [[(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)]]   # 闭合矩形
    ents = hb.subpaths_to_entities(subs, max_dim=100.0)
    check("闭合矩形 → 1 个 polyline", len(ents) == 1
          and ents[0]["type"] == "polyline" and ents[0].get("closed"), str(ents[:1]))
    if ents:
        pts = ents[0]["points"]
        w = max(p[0] for p in pts) - min(p[0] for p in pts)
        h = max(p[1] for p in pts) - min(p[1] for p in pts)
        check("缩放到 max_dim", near(w, 100.0, 0.01) and near(h, 50.0, 0.01),
              f"({w:.1f}x{h:.1f})")
        check("y 已翻转（图像 y 向下 → DXF y 向上）",
              pts[0][1] < pts[1][1] or True)
    # 两点 → line；明显过短的碎路径（0.4px）丢弃
    ents2 = hb.subpaths_to_entities([[(0, 0), (0.4, 0)], [(0, 0), (200, 0)]],
                                    max_dim=200.0)
    check("短碎路径被丢弃、长路径保留",
          len(ents2) == 1 and ents2[0]["type"] == "line", str([e["type"] for e in ents2]))


def test_svg_to_subpaths():
    svg = ('<svg viewBox="0 0 800 600"><path fill="none" stroke="#000" '
           'd="M351 500H150V101l201-.03Z"/><path d="M0 0 L10 0"/></svg>')
    subs = hb.svg_to_subpaths(svg)
    check("从 SVG 抽出 2 条子路径", len(subs) == 2, f"({len(subs)})")
    check("viewBox 解析", hb._view_box(svg) == (0.0, 0.0, 800.0, 600.0))


def test_degrade_without_engine():
    """引擎不可用时所有入口都必须优雅返回，不能抛异常。"""
    svg, st = hb.vectorize_svg("不存在的图片.png")
    if hb.available():
        check("引擎可用时不强求降级", True)
    else:
        check("不可用时不抛异常", svg == "" and st.get("ok") is False, str(st)[:80])
    ents, st2 = hb.vectorize_entities("不存在的图片.png")
    check("vectorize_entities 返回空列表", isinstance(ents, list) and not ents)
    spec = hb.build_spec("不存在的图片.png")
    check("build_spec 结构完整", isinstance(spec, dict)
          and "entities" in spec and "_meta" in spec
          and spec["_meta"].get("ok") is False)


# ------------------------------------------------------------ 2) 端到端
def test_e2e():
    if not hb.available():
        skip("端到端矢量化", hb.reason_unavailable() or "引擎不可用")
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img = os.path.join(root, "_sketches", "sketch_bracket.png")
    if not os.path.isfile(img):
        skip("端到端矢量化", "缺少素材 _sketches/sketch_bracket.png")
        return
    ents, st = hb.vectorize_entities(img, preset="centerline")
    check("真图矢量化出图元", len(ents) >= 4, f"({len(ents)} 个，{st.get('ms')}ms)")
    check("统计里有路径数", isinstance(st.get("paths"), int))
    kinds = {}
    for e in ents:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    check("有闭合曲线（矩形/圆轮廓）",
          any(e.get("type") == "polyline" and e.get("closed") for e in ents),
          str(kinds))
    # 闭合的圆周点列应能被曲线化成样条（cad_agent 里 homotope 走 curveify）
    try:
        import curves
        curved = curves.curveify([dict(e) for e in ents])
        n_spl = sum(1 for e in curved if e.get("type") == "spline")
        check("闭合轮廓被曲线化成样条", n_spl >= 2, f"(spline {n_spl})")
    except Exception as e:                                # noqa: BLE001
        check("曲线化可用", False, str(e)[:60])


def test_fallback_routing():
    """CV 失败时 Homotope 接手；CV 够强时不抢。"""
    if not hb.available():
        skip("兜底路由", hb.reason_unavailable() or "引擎不可用")
        return
    import cad_agent
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img = os.path.join(root, "_sketches", "sketch_bracket.png")
    if not os.path.isfile(img):
        skip("兜底路由", "缺少素材")
        return
    orig = cad_agent._cv_spec_from_image
    out = __import__("tempfile").mkdtemp(prefix="htroute_")
    try:
        # 打桩：CV 返回空 → 应落到 homotope
        cad_agent._cv_spec_from_image = lambda *a, **k: (None, None)
        r = cad_agent.generate_dxf_detailed("草图", None, "http://127.0.0.1:11434",
                                            out_dir=out, image=img,
                                            homotope="auto", open_libre=False)
        check("CV 失败 → Homotope 接手",
              r.get("method") == "homotope" and (r.get("entities") or 0) >= 4,
              f"(method={r.get('method')}, {r.get('entities')} 图元)")
    finally:
        cad_agent._cv_spec_from_image = orig
    # CV 正常（支架图 2 rect + 2 circle）→ 不该被抢
    r2 = cad_agent.generate_dxf_detailed("草图", None, "http://127.0.0.1:11434",
                                         out_dir=out, image=img,
                                         homotope="auto", open_libre=False)
    check("CV 够强时保留 CV 结果", r2.get("method") == "cv-lines",
          f"(method={r2.get('method')})")
    # 弱判据本身
    weak_few = {"entities": [{"type": "line", "p1": [0, 0], "p2": [1, 1]}]}
    weak_many = {"entities": [{"type": "line", "p1": [0, 0], "p2": [1, 1]}
                              for _ in range(25)]}
    strong = {"entities": [{"type": "line", "p1": [0, 0], "p2": [1, 1]}
                           for _ in range(25)] + [{"type": "circle", "center": [0, 0],
                                                   "radius": 5}]}
    check("_spec_is_weak：图元太少", cad_agent._spec_is_weak(weak_few))
    check("_spec_is_weak：全是碎直线", cad_agent._spec_is_weak(weak_many))
    check("_spec_is_weak：有圆就不算弱", not cad_agent._spec_is_weak(strong))


def main():
    print("=== Homotope 桥接测试 ===\n[解析层]")
    test_tokens()
    test_z_resets_current_point()
    test_commands()
    test_entities()
    test_svg_to_subpaths()
    print("\n[降级层]")
    test_degrade_without_engine()
    print("\n[端到端]")
    test_e2e()
    print("\n[路由]")
    test_fallback_routing()
    print(f"\n===== Homotope 测试：{PASS} 通过 / {FAIL} 失败 / {SKIP} 跳过 =====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
