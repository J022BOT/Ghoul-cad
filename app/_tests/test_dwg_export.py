"""DWG 导出回归测试（dwg_export.py + cad_agent 的 fmt 分支）。

DWG 是 Autodesk 私有二进制格式，我们只写 DXF，再由外部转换器转 DWG。
这个测试不去"验图"（拿 LibreCAD/AutoCAD 打开看），只盯住**契约**：

  1) 后端探测找得到包内 LibreDWG（找不到时整条链路要优雅降级，不能崩）；
  2) 产出的文件必须是真 DWG（文件头 AC10xx）且不是空壳；
  3) DXF 永远是主产物，DWG 失败不拖累 DXF（这是用户最怕的回退）；
  4) fmt=dxf / dwg / both 三种取值的行为边界。

跑法：python _tests/test_dwg_export.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dwg_export  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  OK   %-46s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-46s %s" % (name, detail))


ENTS = [
    {"type": "line", "p1": [0, 0], "p2": [80, 0], "layer": "0"},
    {"type": "rect", "x": 10, "y": 10, "w": 40, "h": 20, "layer": "1"},
    {"type": "circle", "center": [50, 50], "radius": 12, "layer": "1"},
    {"type": "arc", "center": [20, 60], "radius": 15, "start": 20, "end": 200,
     "layer": "2"},
    {"type": "polyline", "points": [[0, 0], [10, 5], [20, 0]], "closed": False,
     "layer": "2"},
    {"type": "spline", "points": [[0, 70], [10, 85], [25, 60], [40, 75]],
     "closed": False, "layer": "3"},
    {"type": "text", "value": "标注 70", "p": [5, 90], "height": 5, "layer": "0"},
]


def main():
    tmp = tempfile.mkdtemp(prefix="dwgt_")

    print("[1] 后端探测")
    info = dwg_export.probe()
    check("探测不抛异常", isinstance(info, dict))
    check("找到可用后端", bool(info.get("ok")),
          info["backends"][0]["name"] if info.get("backends") else info.get("note"))
    if not info.get("ok"):
        print("\n  没有 DWG 转换器，后续用例跳过（DXF 路径不受影响）")
        print("\n%d pass, %d fail" % (PASS, FAIL))
        return 0
    st = dwg_export.status()
    check("status 给出人话提示", bool(st.get("note")), st.get("note"))

    print("\n[2] 规范 DXF（ezdxf 重建，喂给转换器）")
    clean = os.path.join(tmp, "clean.dxf")
    n = dwg_export.build_clean_dxf(ENTS, clean)
    check("重建图元数 == 7", n == 7, "写入 %d 个" % n)
    check("文件已落盘", os.path.isfile(clean) and os.path.getsize(clean) > 500)

    print("\n[3] DXF → DWG")
    dwg = os.path.join(tmp, "out.dwg")
    res = dwg_export.convert(clean, dwg)
    check("转换成功", bool(res.get("ok")), res.get("note") or res.get("log", "")[-200:])
    check("产物存在且非空壳",
          os.path.isfile(dwg) and os.path.getsize(dwg) > 128,
          "%d 字节" % (os.path.getsize(dwg) if os.path.isfile(dwg) else 0))
    ver = dwg_export.dwg_version_of(dwg) if os.path.isfile(dwg) else ""
    check("文件头是真 DWG（AC10xx）", bool(ver), "版本 %s" % (ver or "无"))
    check("版本在已知表内", ver in dwg_export.DWG_VERSIONS.values() or not ver,
          ver or "")

    print("\n[4] 失败要优雅，不能留半成品")
    bad = dwg_export.convert(os.path.join(tmp, "nope.dxf"))
    check("源文件不存在 → ok=False", bad.get("ok") is False)
    check("给出原因", bool(bad.get("note")), bad.get("note"))
    # 伪装成 DXF 的垃圾文件：转换器会失败，且不能留下 .dwg
    junk = os.path.join(tmp, "junk.dxf")
    open(junk, "w", encoding="utf-8").write("this is not a dxf\n")
    r2 = dwg_export.convert(junk, os.path.join(tmp, "junk.dwg"))
    check("垃圾 DXF 不产出假 DWG",
          (not r2.get("ok")) and not os.path.isfile(os.path.join(tmp, "junk.dwg")),
          r2.get("note"))

    print("\n[5] cad_agent 的 fmt 分支（DXF 永远是主产物）")
    import cad_agent

    def build(fmt, name):
        spec = {"name": name, "unit": "mm", "entities": ENTS}
        return cad_agent._build_dxf_from_spec(spec, tmp, open_libre=False,
                                              method="editor", fmt=fmt)

    o1 = build("dxf", "f_dxf")
    check("fmt=dxf → 只有 DXF",
          os.path.isfile(o1["path"]) and not o1.get("dwg"), o1["path"])
    check("fmt=dxf → main 指向 DXF", o1.get("main") == o1["path"])

    o2 = build("dwg", "f_dwg")
    check("fmt=dwg → DWG 已生成",
          bool(o2.get("dwg")) and os.path.isfile(o2["dwg"]),
          o2.get("dwg_error") or o2.get("dwg"))
    check("fmt=dwg → DXF 仍在（转换失败也有退路）", os.path.isfile(o2["path"]))
    check("fmt=dwg → main 指向 DWG", o2.get("main") == o2.get("dwg"))
    check("提示语里写了 DWG", "DWG" in (o2.get("message") or ""),
          (o2.get("message") or "")[-60:])

    o3 = build("both", "f_both")
    check("fmt=both → 两份都在",
          os.path.isfile(o3["path"]) and bool(o3.get("dwg")))
    check("fmt=both → main 仍是 DXF", o3.get("main") == o3["path"])

    print("\n[6] 坏入参不崩")
    check("entities=None 也能跑（只是不重建规范 DXF）",
          dwg_export.convert(clean, os.path.join(tmp, "n2.dwg"), entities=None)
          .get("ok") is True)
    check("版本回退：请求 R2018 但后端只到 R2000 时不报错",
          dwg_export.convert(clean, os.path.join(tmp, "n3.dwg"),
                             version="R2018").get("ok") is True)

    print("\n%d pass, %d fail" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
