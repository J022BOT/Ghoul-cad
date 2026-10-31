"""虚线/实线 线型功能回归测试（web_app.py 编辑器 + dxf_writer + dwg_export）。

功能：编辑器选中线条后可切「虚线 / 实线」（dash 属性），导出 DXF 时虚线实体
带组码 6=DASHED、DXF 自带 LTYPE 表（毫米级图案，5mm 实 + 3mm 空）；
DWG 路径经 build_clean_dxf(ezdxf) 重建时同样携带 DASHED（标准图案在毫米图上
几乎连成实线，必须重定义）。

盯住的契约：
  1) 手写 DXF 有 LTYPE 表（CONTINUOUS + DASHED），实线实体不写组码 6（旧文件兼容）；
  2) dash=True / linetype 两种入参都生效；
  3) 导入回读（parse_dxf_to_entities）虚线仍是虚线、实线仍是实线
     —— ezdxf 会把 ByLayer 规范化成大写 BYLAYER，排除清单必须大小写不敏感；
  4) DWG 重建路径同样携带 DASHED 且图案是毫米级；
  5) 编辑器 JS：按钮在、applyDash 只作用"可描线"图元（text 不动）、
     rotateEntity / 橡皮擦碎片保留 dash、新线默认虚线开关接上。

跑法：python _tests/test_dash_linetype.py   （node 可选，缺了就跳过交互段）
"""
import ast
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dxf_writer  # noqa: E402
from dxf_writer import build_from_entities, parse_dxf_to_entities  # noqa: E402

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


def gbk_read(path):
    return open(path, encoding="gbk", errors="replace").read()


def load_js():
    """把 web_app.py 里的 HTML 常量解析出来，取出唯一的 <script> 块。"""
    src = open(os.path.join(ROOT, "web_app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    html = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "HTML":
            html = node.value.value
            break
    assert html, "web_app.py 里找不到 HTML 常量"
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    assert blocks, "HTML 里找不到 <script> 块"
    return html, blocks[0]


def need_node():
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=15, check=True)
        return True
    except Exception:
        return False


ENTS = [
    {"type": "line", "p1": [0, 0], "p2": [60, 0], "layer": "1", "dash": True},
    {"type": "line", "p1": [0, 20], "p2": [60, 20], "layer": "4"},   # 实线（异层，避免回读键覆盖）
    {"type": "circle", "center": [30, 30], "radius": 12, "layer": "2"},
    {"type": "rect", "x": 5, "y": 5, "w": 40, "h": 20, "layer": "3", "dash": True},
]


def main():
    tmp = tempfile.mkdtemp(prefix="dash_")

    print("\n[1] 手写 DXF：LTYPE 表")
    p1 = os.path.join(tmp, "a.dxf")
    d = dxf_writer.DXF()
    d.line((0, 0), (100, 0), layer="1", linetype="DASHED")
    d.line((0, 20), (100, 20), layer="1")            # 实线
    d.circle((50, 50), 10, layer="2", linetype="DASHED")
    d.save(p1)
    txt = gbk_read(p1)
    check("LTYPE 表里定义了 CONTINUOUS", "2\nCONTINUOUS\n" in txt)
    check("LTYPE 表里定义了 DASHED", "2\nDASHED\n" in txt)
    check("DASHED 是毫米级图案（40=总长8 / 49=实5 / -空3）",
          "73\n2\n40\n8\n49\n5\n" in txt and "49\n-3\n" in txt)
    ent = txt[txt.find("2\nENTITIES"):]
    check("虚线实体带组码 6=DASHED（2 处）", ent.count("6\nDASHED\n") == 2)
    check("实线实体不写组码 6（保持旧文件兼容）", ent.count("6\nCONTINUOUS\n") == 0)

    print("\n[2] build_from_entities：dash / linetype 入参")
    p2 = os.path.join(tmp, "b.dxf")
    build_from_entities(ENTS).save(p2)
    t2 = gbk_read(p2)
    e2 = t2[t2.find("2\nENTITIES"):]
    check("dash=True 的 line / rect 各带 DASHED", e2.count("6\nDASHED\n") == 2)
    check("实线 circle 不带组码 6", e2.count("6\nCONTINUOUS\n") == 0)
    p2b = os.path.join(tmp, "b2.dxf")
    build_from_entities([{"type": "line", "p1": [0, 0], "p2": [1, 1],
                          "layer": "0", "linetype": "DASHED"}]).save(p2b)
    check("linetype 直传也生效", "6\nDASHED\n" in gbk_read(p2b))

    print("\n[2b] 自定义段长/间隔 + 箭头")
    p2c = os.path.join(tmp, "b3.dxf")
    build_from_entities([
        {"type": "line", "p1": [0, 0], "p2": [60, 0], "layer": "1",
         "dash": True, "dashLen": 10, "dashGap": 4, "arrow": 3, "arrowSize": 5},
        {"type": "polyline", "points": [[0, 40], [20, 40], [40, 55]],
         "layer": "3", "dash": True, "dashLen": 2.5, "dashGap": 1.5,
         "arrow": 2, "arrowSize": 3},
    ]).save(p2c)
    t3 = gbk_read(p2c)
    e3 = t3[t3.find("2\nENTITIES"):]
    check("LTYPE 表按需生成 DASHED10_4 / DASHED2.5_1.5",
          "2\nDASHED10_4\n" in t3 and "2\nDASHED2.5_1.5\n" in t3)
    check("不同图案的实体各带自己的组码 6",
          e3.count("6\nDASHED10_4\n") == 1 and e3.count("6\nDASHED2.5_1.5\n") == 1)
    check("箭头三角形落成 SOLID（2+1=3 个）", e3.count("\nSOLID\n") == 3,
          str(e3.count("\nSOLID\n")))
    back3 = parse_dxf_to_entities(p2c)
    ln = next(e for e in back3 if e["type"] == "line")
    pl = next(e for e in back3 if e["type"] == "polyline")
    check("回读恢复段长/间隔", ln.get("dashLen") == 10.0 and ln.get("dashGap") == 4.0
          and pl.get("dashLen") == 2.5 and pl.get("dashGap") == 1.5)

    print("\n[3] 导入回读：虚线仍是虚线（BYLAYER 大小写钉子）")
    back = parse_dxf_to_entities(p2)
    got = {(e.get("type"), e.get("layer")): e.get("dash") for e in back}
    check("手写 DXF: 虚线 line/rect → True",
          got.get(("line", "1")) is True and got.get(("polyline", "3")) is True)
    check("手写 DXF: 实线 line/circle → False",
          got.get(("line", "4")) is False and got.get(("circle", "2")) is False)

    print("\n[4] DWG 重建路径（build_clean_dxf / ezdxf）")
    import ezdxf
    p_out = os.path.join(tmp, "c.dxf")
    import dwg_export
    n = dwg_export.build_clean_dxf(ENTS, p_out, unit="mm")
    check("重建实体数 = 4", n == 4, str(n))
    doc = ezdxf.readfile(p_out)
    line_lts = [str(getattr(e.dxf, "linetype", "")).upper()
                for e in doc.modelspace() if e.dxftype() == "LINE"]
    circ_lt = [str(getattr(e.dxf, "linetype", "")).upper()
               for e in doc.modelspace() if e.dxftype() == "CIRCLE"]
    check("虚线 LINE linetype=DASHED", "DASHED" in line_lts, str(line_lts))
    check("实线 LINE/CIRCLE linetype=BYLAYER",
          "BYLAYER" in line_lts and circ_lt == ["BYLAYER"])
    # 图案看落盘文本（pattern_tags 是对象直接比对不可靠；ezdxf 组码右对齐带空格、CRLF）
    raw = open(p_out, "rb").read().decode("utf-8", errors="replace").replace(" ", "")
    check("DASHED 图案被重定义为毫米级(总长 8.0 / 实 5.0 / 空 3.0)",
          "73\r\n2\r\n40\r\n8.0\r\n49\r\n5.0\r\n74\r\n0\r\n49\r\n-3.0\r\n74\r\n0" in raw)
    back2 = parse_dxf_to_entities(p_out)
    got2 = {(e.get("type"), e.get("layer")): e.get("dash") for e in back2}
    check("重建 DXF 回读：虚线 True / 实线 False",
          got2.get(("line", "1")) is True and got2.get(("circle", "2")) is False
          and got2.get(("line", "4")) is False)

    print("\n[5] ODA 转 DWG（后端可用时才跑）")
    st = dwg_export.status()
    if st.get("ok"):
        p_dwg = os.path.join(tmp, "c.dwg")
        res = dwg_export.convert(p_out, p_dwg, version="R2018")
        check("转换 ok", res.get("ok") is True, str(res.get("note"))[:40])
        with open(p_dwg, "rb") as f:
            head = f.read(6)
        check("DWG 文件头 AC10xx", head[:4] == b"AC10", head.decode("ascii", "replace"))
    else:
        print("  SKIP 本机无 DWG 转换器（%s）" % (st.get("note") or ""))

    print("\n[6] 编辑器 JS（静态）")
    html, js = load_js()
    check("实线/虚线按钮在 HTML 里",
          'id="dashSolid"' in html and 'id="dashDashed"' in html)
    check("新线默认虚线开关在 HTML 里", 'id="newDash"' in html)
    check("applyDash 已定义并绑定两个按钮",
          "function applyDash(on" in js
          and "$('#dashSolid').onclick=()=>applyDash(false)" in js
          and "$('#dashDashed').onclick=()=>applyDash(true,true)" in js)
    check("isStrokable 覆盖 7 种可描线图元",
          all(k in js for k in ("'line'", "'rect'", "'circle'", "'arc'",
                                "'polyline'", "'spline'", "'ellipse'"))
          and "function isStrokable" in js)
    check("drawEntity 按 dash 设置/复位 setLineDash（像素随缩放换算）",
          "ctx.setLineDash(e.dash?dashPx(e):[])" in js and "ctx.setLineDash([]);" in js)
    check("rotateEntity 各分支保留线型/箭头（withSt 6 处）",
          js.count("withSt(") >= 6 and "const st={dash:e.dash" in js,
          str(js.count("withSt(")))
    check("创建点接上新线默认（line/circle/pen/poly/spline）",
          js.count("dash:newDash") >= 5 and js.count("dashLen:newDashLen") >= 5,
          str(js.count("dashLen:newDashLen")))
    check("箭头 UI / 帮助函数 / 后端字段齐全",
          'id="arrBoth"' in html and "function applyArrow" in js
          and "function arrowTris" in js and "function isArrowable" in js
          and "arrow:newArrow" in js and "arrowSize:newArrowSize" in js)
    dwg_src = open(os.path.join(ROOT, "dwg_export.py"), encoding="utf-8").read()
    check("DWG 重建路径写箭头 SOLID / 带参虚线",
          "arrow_triangles" in dwg_src and "add_solid" in dwg_src
          and "ensure_linetype" in dwg_src)
    check("空选时不静默成功（提示先选中）",
          "请先选中要设置线型的线条" in js)
    check("移至图层控件 / applyLayer / 下拉框同步齐全",
          'id="moveLayerSel"' in html and 'id="moveLayerBtn"' in html
          and "function applyLayer(n)" in js
          and "$('#moveLayerBtn').onclick=()=>applyLayer" in js
          and "moveLayerSel" in js.split("function renderLayerChips")[1][:600])

    if need_node():
        print("\n[7] 编辑器 JS（node 交互）")
        i = js.index("function isStrokable")
        j = js.index("$('#undoBtn')", i)             # 取到全部线型/箭头处理函数为止
        h0 = js.index("function dashPx")             # 帮助函数区：dashPx/isArrowable/arrowTris
        h1 = js.index("function drawEntity")
        block = js[h0:h1] + "\n" + js[i:j]
        harness = (
            "function eq(a,b){return JSON.stringify(a)===JSON.stringify(b);}\n"
            "let ok=0, bad=0;\n"
            "function t(name,c){ if(c){ok++;console.log('  JSOK '+name);}"
            "else{bad++;console.log('  JSBAD '+name);} }\n"
            "let statusMsg='';\n"
            "function setStatus(m){ statusMsg = (statusMsg?statusMsg+' | ':'')+m; }\n"
            "function pushUndo(){}\n"
            "function updateSelInfo(){}\n"
            "function render(){}\n"
            "function fireChange(){}\n"
            "function $(sel){ return {value:'5', checked:false, addEventListener(){},"
            " dispatchEvent(){}}; }\n"
            "const view={cx:0,cy:0,scale:10};\n"
            "let newDashLen=5, newDashGap=3, newArrow=0, newArrowSize=4;\n"
            "const line={type:'line',p1:[0,0],p2:[10,0],layer:'1'};\n"
            "const txt={type:'text',p:[0,0],height:5,value:'x',layer:'1'};\n"
            "const circ={type:'circle',center:[5,5],radius:3,layer:'2'};\n"
            "let selSet=new Set([line,txt,circ]);\n"
            + block +
            "applyDash(true);\n"
            "t('虚线：line.dash=true', line.dash===true);\n"
            "t('虚线：circle.dash=true', circ.dash===true);\n"
            "t('虚线：text 不被改动', txt.dash===undefined);\n"
            "t('虚线带段长/间隔', line.dashLen===5 && line.dashGap===3);\n"
            "t('提示里带数量（text 不计入 → 2 条）', /2 条/.test(statusMsg));\n"
            "applyDash(false);\n"
            "t('切回实线', line.dash===false && circ.dash===false);\n"
            "applyLayer(parseInt($('#moveLayerSel').value,10));\n"
            "t('移至图层：选中图元 → 图层5', line.layer==='5' && circ.layer==='5' && txt.layer==='5');\n"
            "t('移至图层：提示带数量（3 个）', /3 个图元/.test(statusMsg));\n"
            "applyArrow(3);\n"
            "t('箭头：line 两端 + 大小取输入(5)', line.arrow===3 && line.arrowSize===5);\n"
            "t('箭头：circle 不可加', circ.arrow===undefined && txt.arrow===undefined);\n"
            "selSet.clear(); statusMsg='';\n"
            "applyDash(true);\n"
            "t('空选给出提示', /请先选中/.test(statusMsg));\n"
            "applyLayer(3);\n"
            "t('空选移层也给提示', /请先选中/.test(statusMsg));\n"
            "applyArrow(2);\n"
            "t('空选箭头 → 新线默认终点箭头', newArrow===2);\n"
            "t('箭头三角形：两端两个', arrowTris({type:'line',p1:[0,0],p2:[10,0],"
            "arrow:3,arrowSize:4}).length===2);\n"
            "t('箭头三角形：闭合多段线不加', arrowTris({type:'polyline',"
            "points:[[0,0],[10,0],[10,10]],closed:true,arrow:3,arrowSize:4}).length===0);\n"
            "console.log('JSRESULT '+ok+' pass, '+bad+' fail');\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as f:
            f.write(harness)
            hpath = f.name
        try:
            r = subprocess.run(["node", hpath], capture_output=True,
                               text=True, timeout=30)
            out = (r.stdout or "") + (r.stderr or "")
            m = re.search(r"JSRESULT (\d+) pass, (\d+) fail", out)
            if m:
                jp, jf = int(m.group(1)), int(m.group(2))
                for ln in out.splitlines():
                    if "JSBAD" in ln:
                        print("       " + ln.strip())
                check("JS 交互断言 %d pass / 0 fail" % jp, jf == 0)
            else:
                check("JS 交互段能跑起来", False, out[:200])
        finally:
            try:
                os.remove(hpath)
            except OSError:
                pass
    else:
        print("\n[7] 编辑器 JS（node 交互）：跳过（未装 node）")

    print("\n%d pass, %d fail" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
