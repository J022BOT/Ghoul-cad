"""编辑器（web_app.py 内嵌 HTML/JS）交互回归测试 —— 需要 node。

这次踩的坑值得留个钉子：画布上的「选中 + 拖动」在 mousemove 里调用
`selSet.forEach(en => moveEntity(en, dx, dy))`，而 moveEntity 当时签名是
`moveEntity(i, dx, dy)` 且内部 `const e = entities[i]`。传进去的是图元**对象**，
于是 `entities[对象]` = undefined，`e.type` 立刻抛 TypeError —— 整条 mousemove
回调死掉（连 drag.last 都没更新），表现就是"能选中、但拖不动"。

验证五件事：
  1) 整页 JS 语法（node --check）；
  2) moveEntity 同时认「图元对象」和「数组下标」；
  3) 无效引用（undefined / null / {} / 越界下标）不抛异常；
  4) 命中测试 + 拖动 的完整交互链：点中 → 选中 → 多帧累加位移 → 引用不断；
  5) spline / ellipse 图元走同一条路也能选中、能拖动。
"""
import ast
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  OK   %-44s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-44s %s" % (name, detail))


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


def grab(js, start, end):
    i = js.index(start)
    j = js.index(end, i)
    return js[i:j]


def need_node():
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=15, check=True)
        return True
    except Exception:
        return False


STUB = (
    "let view = {cx:0, cy:0, scale:10};\n"
    "let entities = [];\n"
    "const HIT = 6;\n"
    "const canvas = {width:800, height:600};\n"
    "const ctx = {font:'', measureText:(s)=>({width:(s||'').length*8,"
    " actualBoundingBoxAscent:8, actualBoundingBoxDescent:2})};\n"
)

CASES = r"""
function eq(a,b){return JSON.stringify(a)===JSON.stringify(b);}
let ok=0, bad=0;
function t(n,c){ if(c) ok++; else { bad++; console.log('  FAIL js: '+n); } }

/* ---- 1) moveEntity 认对象 / 认下标 / 认全 ---- */
entities=[{type:'line',p1:[0,0],p2:[10,0],layer:'0'}];
let e0=entities[0];
moveEntity(e0,5,3);
t('传对象也能移动(原来崩在这)', eq(e0.p1,[5,3]) && eq(e0.p2,[15,3]));

entities=[{type:'circle',center:[1,1],radius:2,layer:'0'}];
moveEntity(0,-1,2);
t('传下标仍然可用', eq(entities[0].center,[0,3]));

entities=[{type:'spline',points:[[0,0],[1,1],[2,0]],closed:false,layer:'A'}];
moveEntity(entities[0],2,2);
t('spline 对象可移动', eq(entities[0].points,[[2,2],[3,3],[4,2]]));

entities=[{type:'ellipse',points:[[0,0]],center:[5,5],rx:3,ry:2,rotation:0,layer:'A'}];
moveEntity(entities[0],1,-1);
t('ellipse 对象可移动', eq(entities[0].center,[6,4]) && eq(entities[0].points,[[1,-1]]));

let threw=false;
try{ moveEntity(undefined,1,1); moveEntity(null,1,1); moveEntity({},1,1); moveEntity(99,1,1); }
catch(err){ threw=true; }
t('无效引用不抛异常', !threw);

/* ---- 2) 命中 + 拖动 交互链 ---- */
entities=[{type:'line',p1:[0,0],p2:[10,0],layer:'0'}];
view={cx:5,cy:0,scale:10};
const selSet=new Set();
t('点中线条', hitTest([400,300])===0);
t('点空白落空', hitTest([400,120])===-1);
t('偏 5px 仍命中(HIT=6)', hitTest([400,305])===0);
t('偏 8px 不命中', hitTest([400,308])===-1);

selSet.add(entities[0]);
selSet.forEach(en=>moveEntity(en,2,-1));
t('拖动后两端一起走', eq(entities[0].p1,[2,-1]) && eq(entities[0].p2,[12,-1]));
t('选区引用未断', selSet.has(entities[0]));
for(let k=0;k<5;k++) selSet.forEach(en=>moveEntity(en,0.5,0));
t('多帧位移累加', eq(entities[0].p1,[4.5,-1]));

/* spline 走同一路径 */
entities=[{type:'spline',points:[[0,0],[5,5],[10,0]],closed:false,layer:'1'}];
view={cx:5,cy:2,scale:10};
t('spline 可命中', hitTest(w2s([5,5]))===0);
selSet.clear(); selSet.add(entities[0]);
selSet.forEach(en=>moveEntity(en,1,1));
t('spline 可拖动(点全动)', eq(entities[0].points[0],[1,1]) && eq(entities[0].points[1],[6,6]));

/* 圆弧：只有 center/radius/start/end（没给 points）也要能画、能选中、能拖 */
entities=[{type:'arc',center:[0,0],radius:10,start:0,end:90,layer:'2'}];
view={cx:0,cy:0,scale:10};
const ap=entityPoints(entities[0]);
t('arc 无 points 也能算出端点', ap.length>=2 && Math.abs(ap[0][0]-10)<1e-9 && Math.abs(ap[0][1])<1e-9);
t('arc 可命中(弧上一点)', hitTest(w2s([10,0]))===0);
t('arc 圆心处不命中(不是整圆)', hitTest(w2s([0,0]))===-1);
selSet.clear(); selSet.add(entities[0]);
selSet.forEach(en=>moveEntity(en,3,-2));
t('arc 可拖动(圆心+点一起)', eq(entities[0].center,[3,-2]) && eq(entities[0].points[0],[13,-2]));

/* ---- 3) 与水平夹角（0°=水平，0~180） ---- */
t('0° = 水平', Math.abs(dirAngleDeg([0,0],[5,0]))<1e-9);
t('90° = 竖直', Math.abs(dirAngleDeg([0,0],[0,5])-90)<1e-9);
t('60° 右上', Math.abs(dirAngleDeg([0,0],[5,8.660254])-60)<1e-6);
t('150° 左上', Math.abs(dirAngleDeg([0,0],[-5,2.8867513])-150)<1e-6);
t('方向无正反(反向同角)', Math.abs(dirAngleDeg([5,0],[0,0]))<1e-9);
t('多段线取最长段方向', Math.abs(entAngleDeg({type:'polyline',points:[[0,0],[1,0.1],[1,9]],closed:false})-90)<1e-6);
t('圆没有方向', entAngleDeg({type:'circle',center:[0,0],radius:2})===null);
t('多选取最长线作代表角', Math.abs(selAngleDeg(new Set([
    {type:'line',p1:[0,0],p2:[1,0]}, {type:'line',p1:[0,0],p2:[0,10]}]))-90)<1e-9);

/* ---- 4) 线条扶正：近水平/竖直 → 精确 0°/90°，保长度 ---- */
let hz={type:'line',p1:[0,0],p2:[10,0.35],layer:'0'};
straightenSelLines([hz],3);
t('近水平线扶正(保长度)', hz.p2[1]===0 && Math.abs(hz.p2[0]-Math.hypot(10,0.35))<1e-6);
let vt={type:'line',p1:[0,0],p2:[0.4,10],layer:'0'};
straightenSelLines([vt],3);
t('近竖直线扶正', vt.p2[0]===0 && Math.abs(vt.p2[1]-Math.hypot(0.4,10))<1e-6);
let st={type:'line',p1:[0,0],p2:[10,3],layer:'0'};
straightenSelLines([st],3);
t('超容差不扶正', Math.abs(st.p2[1]-3)<1e-9);
let dg={type:'line',p1:[0,0],p2:[10,0.35],layer:'0'};
straightenSelLines([dg],0);
t('容差 0 = 关闭扶正', Math.abs(dg.p2[1]-0.35)<1e-9);

/* ---- 5) 吸附：顶点优先、线段其次 ---- */
entities=[{type:'line',p1:[0,0],p2:[10,0],layer:'0'},
          {type:'line',p1:[0,20],p2:[10,20],layer:'0'}];
let mv=entities[1], bs=[[mv, JSON.parse(JSON.stringify(mv))]];
let tg=collectSnapTargets(new Set([mv]));
t('候选点含目标两端+中点', tg.pts.length===3);
applyMoveFromBases(bs,-1,-19,{on:false,deg:0});   // p1=(-1,1) p2=(9,1)
let off=snapAnchorOffset({bases:bs,targets:tg},{on:true,w:2,deg:0});
t('顶点吸附补齐到目标端点',
  off && Math.abs(off.d[0]-1)<1e-9 && Math.abs(off.d[1]+1)<1e-9 && off.hint.kind==='pt');
t('远离时不吸附',
  snapAnchorOffset({bases:bs,targets:tg},{on:true,w:0.3,deg:0})===null);

entities=[{type:'polyline',points:[[0,0],[0,100]],closed:false},
          {type:'line',p1:[1.5,50],p2:[20,50],layer:'0'}];
mv=entities[1]; bs=[[mv, JSON.parse(JSON.stringify(mv))]];
tg=collectSnapTargets(new Set([mv]));
off=snapAnchorOffset({bases:bs,targets:tg},{on:true,w:2,deg:0});
t('无顶点命中时投影到线段', off && off.hint.kind==='seg'
  && Math.abs(off.d[0]+1.5)<1e-9 && Math.abs(off.d[1])<1e-9);

/* ---- 6) 快照重算：多处拖拽不累积漂移 ---- */
entities=[{type:'line',p1:[0,0],p2:[10,0],layer:'0'}];
mv=entities[0]; bs=[[mv, JSON.parse(JSON.stringify(mv))]];
applyMoveFromBases(bs,3,4,{on:false,deg:0});
applyMoveFromBases(bs,3,4,{on:false,deg:0});
t('重复应用同一位移不叠加', eq(mv.p1,[3,4]) && eq(mv.p2,[13,4]));

/* ---- 7) 旋转取绝对角：填 60° 就是与水平成 60° ---- */
let R={type:'line',p1:[0,0],p2:[10,0],layer:'0'};
let cur=selAngleDeg(new Set([R])), cc=combinedCentroid(new Set([R]));
Object.assign(R, rotateEntity(R,cc,(60-cur)*Math.PI/180));
t('填 60° → 夹角 60°', Math.abs(dirAngleDeg(R.p1,R.p2)-60)<0.05);   // 坐标保留 3 位小数，留 0.05° 余量
let cur2=selAngleDeg(new Set([R]));
Object.assign(R, rotateEntity(R,cc,(60-cur2)*Math.PI/180));
t('再填 60° 不动(绝对角)', Math.abs(dirAngleDeg(R.p1,R.p2)-60)<0.05);
Object.assign(R, rotateEntity(R,cc,(0-selAngleDeg(new Set([R])))*Math.PI/180));
t('填 0° → 回水平', Math.abs(dirAngleDeg(R.p1,R.p2))<0.05 && Math.abs(R.p1[1]-R.p2[1])<1e-2);

console.log('JS_RESULT '+ok+' '+bad);
"""


def main():
    print("[1] 取页面 JS")
    html, js = load_js()
    check("取出 <script> 块", len(js) > 20000, "%d 字符" % len(js))

    if not need_node():
        print("\n[跳过] 本机没有 node，JS 部分跳过")
        print("\n%d pass, %d fail" % (PASS, FAIL))
        return 1 if FAIL else 0

    print("\n[2] 整页 JS 语法检查")
    tmp = tempfile.mkdtemp(prefix="wbt_")
    page = os.path.join(tmp, "page.js")
    open(page, "w", encoding="utf-8").write(js)
    r = subprocess.run(["node", "--check", page], capture_output=True, text=True)
    check("node --check 通过", r.returncode == 0, (r.stderr or "").strip().splitlines()[:1])

    print("\n[3] 编辑器交互链（真实源码切片）")
    parts = [
        STUB,
        grab(js, "function arcPoints(e)", "function worldBounds"),
        grab(js, "function w2s(p)", "function dist2seg"),
        grab(js, "function dist2seg(p,a,b)", "function textBoxScreen"),
        grab(js, "function textBoxScreen(e)", "function hitTest"),
        grab(js, "function hitTest(sp)", "function rnd(p)"),
        grab(js, "function rnd(p)", "function moveEntity"),
        grab(js, "function moveEntity(", "/* ----"),
        grab(js, "function rotatePts(pts,cx,cy,a)", "/* ---- n边形"),
        CASES,
    ]
    hz = os.path.join(tmp, "harness.js")
    open(hz, "w", encoding="utf-8").write("\n".join(parts))
    r = subprocess.run(["node", hz], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.rstrip())
    m = re.search(r"JS_RESULT (\d+) (\d+)", out)
    check("交互链全部通过", bool(m) and int(m.group(2)) == 0,
          "%s pass / %s fail" % (m.group(1), m.group(2)) if m else "无结果")

    print("\n[3b] 导入 DXF 的图层归属（真实源码切片）")
    # 踩过的坑：appendEntities 原来一律 c.layer=String(activeLayer)，把「导入到图层 N」
    # 硬改成当前图层 —— 用户选了图层 3，结果全落进图层 1，看着像"点了没反应"。
    APPEND_STUB = (
        "let entities = [];\nlet selSet = new Set();\nlet activeLayer = 1;\n"
        "function pushUndo(){}\nfunction updateSelInfo(){}\nfunction render(){}\n"
        "function fireChange(){}\nfunction fitView(){}\n"
    )
    APPEND_CASES = r"""
let ok2=0,bad2=0;
function t2(n,c){ if(c) ok2++; else { bad2++; console.log('  FAIL js: '+n); } }
const added=appendEntities([{type:'line',p1:[0,0],p2:[1,0],layer:'3'},
                            {type:'circle',center:[0,0],radius:1}]);
t2('自带图层的图元保留原图层', added[0].layer==='3');
t2('没带图层的图元才落到当前图层', added[1].layer==='1');
t2('appendEntities 返回新加入的图元引用', added.length===2 && added[0]===entities[0]);
console.log('JS_RESULT2 '+ok2+' '+bad2);
"""
    parts2 = [APPEND_STUB,
              grab(js, "function appendEntities(list)", "/* 选中一批图元"),
              APPEND_CASES]
    hz2 = os.path.join(tmp, "harness2.js")
    open(hz2, "w", encoding="utf-8").write("\n".join(parts2))
    r2 = subprocess.run(["node", hz2], capture_output=True, text=True)
    out2 = (r2.stdout or "") + (r2.stderr or "")
    print(out2.rstrip())
    m2 = re.search(r"JS_RESULT2 (\d+) (\d+)", out2)
    check("图层归属测试全部通过", bool(m2) and int(m2.group(2)) == 0,
          "%s pass / %s fail" % (m2.group(1), m2.group(2)) if m2 else "无结果")

    print("\n[4] 静态一致性（调用方传对象 ↔ 函数认对象）")
    check("移动改为快照重算（applyMoveFromBases）",
          "functionapplyMoveFromBases(bases,dx,dy,cfg)" in js.replace(" ", ""))
    check("moveEntity 兼容对象入参", "typeof ref === 'number'" in js)
    check("吸附控件齐备（开关/距离/角度）",
          all(k in html for k in ('id="snapChk"', 'id="snapPxIn"', 'id="snapDegIn"')))
    check("应用旋转按「与水平夹角」取绝对角", "delta=deg-cur" in js.replace(" ", ""))
    check("选中信息显示与水平夹角", js.count("与水平夹角") >= 2)
    check("移动路径挂了吸附（候选缓存 + 顶点/线段）",
          "collectSnapTargets(selSet)" in js.replace(" ", "")
          and "snapAnchorOffset(drag,cfg)" in js.replace(" ", ""))
    check("entityPoints 覆盖 spline/ellipse",
          "e.type==='spline'||e.type==='ellipse'" in js.replace(" ", ""))
    check("drawEntity 会画 spline",
          "elseif(e.type==='polyline'||e.type==='spline'||e.type==='ellipse'||e.type==='arc')"
          in js.replace(" ", ""))
    check("arc 有独立取点函数（无 points 也能画）", "functionarcPoints(e)" in js.replace(" ", ""))

    print("\n[4b] 导入 DXF 的前端接线（别再静默无反应）")
    flat0 = js.replace(" ", "").replace("\n", "")
    check("图元自带图层优先（不再被 activeLayer 覆盖）",
          "c.layer=(e.layer!=null&&String(e.layer)!=='')?String(e.layer):String(activeLayer)"
          in flat0)
    check("导入后重新适配视野", "Editor.fitAll()" in flat0)
    check("导入后选中并高亮所导入图元", "Editor.select(added)" in flat0)
    check("解析期间禁用「导入」按钮", "goBtn.disabled=true" in flat0)
    check("没解析结果时给出提示而不是静默 return", "还没有解析结果" in js)
    check("导入成功同时写 layerHint 与状态栏", "setStatus(msg+" in flat0)

    print("\n[5] DWG 导出接线（前端必须真的把 fmt 传上去）")
    flat = js.replace(" ", "").replace("\n", "")
    check("有输出格式选择框", 'id="fmtSel"' in html)
    check("生成请求带 fmt", "ocr_text:wantOcr,fmt:fmtSel.value" in flat)
    check("勾选生成请求带 fmt", "keep_ids:keepIds,fmt:fmtSel.value" in flat)
    check("编辑器导出走 fmt 参数", "asyncfunctionexportCanvas(fmt,btn)" in flat)
    check("导出 DWG 按钮已绑定", "$('#exportDwg').onclick" in js)
    check("合成弹窗有 DWG 选项", 'id="combineDwg"' in html)
    check("DWG 结果链接会被赋值", "linkEl.href='/open?file='+encodeURIComponent(data.dwg)" in flat)

    print("\n[6] 后端路由（Python 侧）")
    src = open(os.path.join(ROOT, "web_app.py"), encoding="utf-8").read()
    check("/api/dwg_status 已注册", '"/api/dwg_status": self._api_dwg_status' in src)
    check("/api/save 透传 fmt 给 _build_dxf_from_spec",
          'flatten_layer="0" if flatten else None, fmt=fmt' in src)
    check("/open 对 DWG 走资源管理器",
          'if f.lower().endswith(".dwg")' in src)

    print("\n%d pass, %d fail" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
