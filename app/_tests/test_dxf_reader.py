# -*- coding: utf-8 -*-
"""内置 DXF 读取器（dxf_reader.py）回归。

真实 bug：用户点「导入 DXF」弹出
    「未安装 ezdxf，无法解析 DXF（请先 pip install ezdxf）」
但便携包里明明装了 ezdxf——原因是 web_app 的 numpy/cv2/ezdxf 都是懒加载，
用系统自带的干净 Python 启动网页服务也能开界面，只有导入/生成时才炸。

修法：①`dxf_writer.parse_dxf_to_entities` 在 ezdxf 缺失或读不动时，自动退回
      内置的纯 Python 组码读取器（本测试的第 1~7 项）；
      ②启动自检 `runtime_guard.ensure()` 发现缺 numpy/cv2 就用包内 Python
      重启自己，不再让用户撞见这种幽灵报错。
"""
import os
import sys
import math
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILS = []


def check(name, cond, extra=""):
    print(("  OK   " if cond else "  FAIL ") + name + (("  " + str(extra)) if extra else ""))
    if not cond:
        FAILS.append(name)


def _types(ents):
    return [e.get("type") for e in ents]


def _bbox(ents):
    xs, ys = [], []
    for e in ents:
        t = e.get("type")
        if t == "line":
            pts = [e["p1"], e["p2"]]
        elif t in ("circle", "arc"):
            c, r = e["center"], e["radius"]
            pts = [[c[0] - r, c[1] - r], [c[0] + r, c[1] + r]]
        else:
            pts = e.get("points") or []
        for q in pts:
            xs.append(float(q[0]))
            ys.append(float(q[1]))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def test_roundtrip():
    """本程序写出的 DXF → 内置读取器，图元构成与几何必须对得上。"""
    print("\n[1] 往返解析（LINE/CIRCLE/ARC/POLYLINE/SPLINE/ELLIPSE/TEXT/虚线）")
    import dxf_writer as dw
    import dxf_reader as dr

    d = dw.DXF(unit="mm")
    d.line([0, 0], [50, 0])
    d.circle([20, 20], 8)
    d.arc([100, 0], 30, 10, 120)
    d.rect(0, 0, 100, 60)                       # → LWPOLYLINE 闭合
    d.polyline([[0, 0], [10, 10], [20, 0]], closed=True)
    d.spline([[0, 0], [10, 20], [30, 25], [50, 10]], closed=False)
    d.ellipse([160, 40], 20, 12, 30, 0, 270)
    d.text([5, 70], 5, "中文标注 Φ10")
    d.line([0, 0], [100, 100], linetype="DASHED10_4")

    p = os.path.join(tempfile.gettempdir(), "_dr_roundtrip.dxf")
    d.save(p)

    ez = dw.parse_dxf_to_entities(p)             # 走 ezdxf
    mine = dr.parse_dxf(p)                       # 纯 Python
    check("图元数量一致", len(ez) == len(mine), "%d vs %d" % (len(ez), len(mine)))
    check("图元类型逐一一致", _types(ez) == _types(mine), str(_types(mine)))
    b1, b2 = _bbox(ez), _bbox(mine)
    check("包围盒一致（<0.05mm）",
          b1 and b2 and all(abs(x - y) < 0.05 for x, y in zip(b1, b2)),
          "%s vs %s" % (b1, b2))

    line = [e for e in mine if e["type"] == "line"][0]
    check("LINE 端点", line["p1"] == [0.0, 0.0] and line["p2"] == [50.0, 0.0], line)
    cir = [e for e in mine if e["type"] == "circle"][0]
    check("CIRCLE 圆心/半径", cir["center"] == [20.0, 20.0] and abs(cir["radius"] - 8) < 1e-9, cir)

    arc = [e for e in mine if e["type"] == "arc"][0]
    check("ARC 保留 arc 类型 + 细分点",
          arc["type"] == "arc" and abs(arc["radius"] - 30) < 1e-9 and len(arc["points"]) > 10, )
    check("ARC 起止角", abs(arc["start"] - 10) < 1e-6 and abs(arc["end"] - 120) < 1e-6,
          (arc["start"], arc["end"]))

    closed = [e for e in mine if e["type"] == "polyline" and e.get("closed")]
    check("闭合多段线 closed 标记", len(closed) >= 1, [e.get("closed") for e in mine if e["type"] == "polyline"])

    sp = [e for e in mine if e["type"] == "spline"][0]
    check("SPLINE 采样成曲线点", len(sp["points"]) >= 20, len(sp["points"]))
    # 样条是"过给定点"的插值曲线：必须真的穿过原来那 4 个点
    fit = [(0, 0), (10, 20), (30, 25), (50, 10)]

    def seg_dist(pt, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / L2))
        return math.hypot(pt[0] - (ax + t * dx), pt[1] - (ay + t * dy))

    def poly_dist(pt, pts):
        if len(pts) < 2:
            return min(math.hypot(pt[0] - q[0], pt[1] - q[1]) for q in pts)
        return min(seg_dist(pt, pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    d_fit = max(poly_dist(q, sp["points"]) for q in fit)
    check("SPLINE 穿过原始拟合点（≤0.6mm）", d_fit <= 0.6, "%.3f" % d_fit)
    # 与 ezdxf 路线的摊平结果比曲线偏差（点到折线，而不是点到点——
    # 点到点会混入两边的采样密度差异，不是几何误差）
    sp_ez = [e for e in ez if e["type"] == "spline"][0]
    dev = max(poly_dist(q, sp_ez["points"]) for q in sp["points"])
    dev2 = max(poly_dist(q, sp["points"]) for q in sp_ez["points"])
    check("SPLINE 曲线偏差 ≤0.3mm（双向）", max(dev, dev2) <= 0.3,
          "%.3f / %.3f" % (dev, dev2))

    ell = [e for e in mine if e["type"] == "ellipse"][0]
    check("ELLIPSE 长短轴", abs(ell["rx"] - 20) < 1e-6 and abs(ell["ry"] - 12) < 1e-6,
          (ell.get("rx"), ell.get("ry")))

    txt = [e for e in mine if e["type"] == "text"][0]
    check("TEXT 中文不乱码", "中文标注" in txt["value"] and "Φ10" in txt["value"], txt["value"])

    dashed = [e for e in mine if e["type"] == "line" and e.get("dash")]
    check("虚线线型被识别", len(dashed) == 1)
    check("虚线参数反解（段 10 / 间隔 4）",
          dashed and abs(dashed[0].get("dashLen", 0) - 10) < 1e-9
          and abs(dashed[0].get("dashGap", 0) - 4) < 1e-9,
          dashed[0] if dashed else None)


def test_handmade():
    """手写最小 DXF：LWPOLYLINE / POLYLINE+VERTEX / MTEXT 格式码清理。"""
    print("\n[2] 手写组码（LWPOLYLINE / POLYLINE / MTEXT）")
    import dxf_reader as dr

    txt = "\r\n".join([
        "0", "SECTION", "2", "ENTITIES",
        # 闭合 LWPOLYLINE，含 4 个顶点
        "0", "LWPOLYLINE", "8", "OUTLINE", "70", "1",
        "10", "0", "20", "0",
        "10", "10", "20", "0",
        "10", "10", "20", "10",
        "10", "0", "20", "10",
        # 旧式 POLYLINE + VERTEX + SEQEND
        "0", "POLYLINE", "8", "OLD", "70", "0",
        "0", "VERTEX", "10", "100", "20", "0",
        "0", "VERTEX", "10", "120", "20", "50",
        "0", "SEQEND",
        # 多行文字，带 \P 换行与 {\f...;} 字体码
        "0", "MTEXT", "8", "TEXT", "10", "5", "20", "80", "40", "3.5",
        "3", "第一行\\P", "1", "{\\fSimSun|b0;第二行}",
        "0", "ENDSEC", "0", "EOF",
    ])
    ents = dr.parse_dxf_text(txt)
    check("解析出 3 个图元", len(ents) == 3, _types(ents))

    lw = [e for e in ents if e.get("layer") == "OUTLINE"]
    check("LWPOLYLINE 4 个顶点", lw and len(lw[0]["points"]) == 4, lw)
    check("LWPOLYLINE 闭合标记", lw and lw[0]["closed"] is True)

    poly = [e for e in ents if e.get("layer") == "OLD"]
    check("POLYLINE 吃到 VERTEX 顶点",
          poly and len(poly[0]["points"]) == 2
          and poly[0]["points"][1] == [120.0, 50.0], poly)
    check("POLYLINE 不闭合", poly and poly[0]["closed"] is False)

    mt = [e for e in ents if e.get("layer") == "TEXT"]
    check("MTEXT 拼接 3/1 组码并去格式码",
          mt and "第一行" in mt[0]["value"] and "第二行" in mt[0]["value"],
          mt[0]["value"] if mt else None)
    check("MTEXT \\P → 换行", mt and "\n" in mt[0]["value"], repr(mt[0]["value"]) if mt else None)


def test_no_ezdxf():
    """没装 ezdxf 时，parse_dxf_to_entities 必须自动走内置读取器。"""
    print("\n[3] ezdxf 缺失时的自动兜底")
    import builtins
    import dxf_writer as dw

    d = dw.DXF(unit="mm")
    d.circle([0, 0], 5)
    p = os.path.join(tempfile.gettempdir(), "_dr_noez.dxf")
    d.save(p)

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "ezdxf" or name.startswith("ezdxf."):
            raise ImportError("模拟：未安装 ezdxf")
        return real_import(name, *a, **k)

    builtins.__import__ = fake
    try:
        ents = dw.parse_dxf_to_entities(p)
        ok = len(ents) == 1 and ents[0]["type"] == "circle"
    except Exception as e:                             # noqa: BLE001
        ents, ok = [], False
        print("      异常：%s" % e)
    finally:
        builtins.__import__ = real_import
    check("缺 ezdxf 也能解析出图元", ok, _types(ents))

    # 兜底也解析不了时，报错里必须带解释器路径（便于排查"是哪个 python 在跑"）
    bad = os.path.join(tempfile.gettempdir(), "_dr_binary.dxf")
    with open(bad, "wb") as f:
        f.write(b"AutoCAD Binary DXF\r\n\x1a\x00" + b"\x00" * 40)
    msg = ""
    try:
        dw.parse_dxf_to_entities(bad)
    except Exception as e:                             # noqa: BLE001
        msg = str(e)
    check("二进制 DXF 报错说明清楚", "二进制" in msg or "Binary" in msg, msg[:60])
    check("报错带解释器路径", "python.exe" in msg.lower() or "interpreter" in msg.lower(),
          msg[:80])


def test_runtime_guard():
    """启动自检：环境齐全时什么都不做；描述信息能看清是谁在跑。"""
    print("\n[4] 启动自检 runtime_guard")
    import runtime_guard as rg

    info = rg.describe()
    check("describe 给出解释器路径", bool(info.get("python")))
    check("describe 给出缺失清单", isinstance(info.get("missing"), list))
    check("环境齐全时不重启（ensure 返回 False）", rg.ensure(quiet=True) is False,
          info.get("missing"))
    check("包内 python 探测（本机包内应存在）",
          rg.bundled_python() is None or os.path.isfile(rg.bundled_python()),
          rg.bundled_python())


def main():
    test_roundtrip()
    test_handmade()
    test_no_ezdxf()
    test_runtime_guard()
    print("\n=== 结果：%d 项失败 ===" % len(FAILS))
    if FAILS:
        for f in FAILS:
            print("  -", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
