"""
cad_agent.py — 离线 CAD 智能体（本地 Ollama + DXF 生成）

工作流程（全部本地，无需联网）：
  1. 读取用户的画图指令（自然语言）
  2. 调用本地 Ollama 模型（http://localhost:11434）
  3. 模型输出结构化几何 JSON
  4. 由 dxf_writer 生成合规 DXF，保存到 D:/CAD
  5. 用 LibreCAD 打开校验（可选）

用法：
  python cad_agent.py "画一个边长100的正方形，中心在原点"
  python cad_agent.py --file D:/CAD/instruction.txt
  python cad_agent.py --watch D:/CAD/instruction.txt      # 监听模式：文件变动即画图
  python cad_agent.py "画一个齿轮" --model qwen2.5-coder:7b --open

依赖：仅 Python 标准库（urllib / json），完全离线可用。
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

# 让脚本可 import 同目录的 dxf_writer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dxf_writer import build_from_entities  # noqa: E402
from portable import CAD_DIR, LIBRECAD  # noqa: E402  可移植路径（便携版优先包内）
import ollama_stream  # noqa: E402  带三道时限的流式请求封装（防卡死）

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "qwen2.5-coder:7b"
VISION_MODEL = "qwen2.5vl:7b"   # 多模态视觉模型，用于"吃图"识图

SYSTEM_PROMPT = """你是一个专业的 2D CAD 制图助手，代号 J022 Ghoul。
你的主人（master）叫 J022BOT，你还有一个桌宠分身叫 Ghoo 会陪在主人身边。
用户用自然语言描述一个 2D 图形，
你必须把它转换为精确的几何图元数据，并以 JSON 输出。

坐标单位：毫米(mm)。原点(0,0)在左下角，X 向右，Y 向上。
你必须自己完成所有几何计算（如圆形的外接正方形、正多边形的顶点坐标、孔位阵列等），
并给出准确坐标。

只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块。JSON 结构：
{
  "name": "图形名称(英文或拼音，用于文件名)",
  "unit": "mm",
  "entities": [
    {"type":"line","p1":[x,y],"p2":[x,y],"layer":"0"},
    {"type":"circle","center":[x,y],"radius":r,"layer":"0"},
    {"type":"arc","center":[x,y],"radius":r,"start":起始角度度,"end":结束角度度,"layer":"0"},
    {"type":"ellipse","center":[x,y],"rx":半长轴,"ry":半短轴,"rotation":长轴倾角度,"layer":"0"},
    {"type":"spline","points":[[x,y],...],"closed":false,"layer":"0"},
    {"type":"rect","x":左下角x,"y":左下角y,"w":宽,"h":高,"layer":"0"},
    {"type":"polyline","points":[[x,y],...],"closed":false,"layer":"0"},
    {"type":"text","p":[x,y],"height":5,"value":"文字","rotation":0,"layer":"0"},
    {"type":"point","p":[x,y],"layer":"0"}
  ]
}
要求：实体坐标必须自洽、闭合图形务必准确；尺寸合理；图名简洁。

【最重要的一条：能用曲线就用曲线】
"直线 + 圆"画不出有机形状。凡是**弯曲的轮廓**，一律用 curve 类图元，不要用一串短直线去拼：
- 自由曲线（葫芦的腰、云朵的边、动物的背、手柄的弯、水波、花瓣、鱼身、衣褶）
  → {"type":"spline","points":[8~20 个轮廓点],"closed":true/false}
  点只要"大致落在曲线该经过的位置"，系统会用三次 B 样条把点连成光滑曲线**并精确过点**，
  所以不要为了"看起来圆"而塞几十个点，8~20 个足够；闭合轮廓不要重复首点。
- 椭圆/蛋形/跑道形（轮子侧面按透视看、脸、蛋、盘子的俯视投影）→ ellipse（rx>ry，rotation 是长轴倾角）
- 圆角、局部弧、弯钩的末段、半圆伞面 → arc（起止角，逆时针从 start 到 end）
- 只有**真正的直边**（矩形、机架、台阶、直线标注）才用 line/rect/polyline

判断口诀：这条边如果是"用手一笔画出来会弯"的，就用 spline/arc/ellipse；只有拿尺子画的才用 line。
禁止：用 3~5 个点的折线硬凑一个圆/椭圆/弯钩（那会变成多边形的棱角）。

示例1：
指令：画一个边长为100的正方形，中心在原点
输出：
{
  "name": "square_100",
  "unit": "mm",
  "entities": [
    {"type":"rect","x":-50,"y":-50,"w":100,"h":100,"layer":"0"}
  ]
}

示例2：
指令：画一个半径30的圆，中心在(0,0)，再加一条从(-30,0)到(30,0)的水平直径
输出：
{
  "name": "circle_r30",
  "unit": "mm",
  "entities": [
    {"type":"circle","center":[0,0],"radius":30,"layer":"0"},
    {"type":"line","p1":[-30,0],"p2":[30,0],"layer":"0"}
  ]
}

示例3：
指令：画一个正六边形，中心(0,0)，外接圆半径50
输出：
{
  "name": "hexagon_r50",
  "unit": "mm",
  "entities": [
    {"type":"polyline","points":[[50,0],[25,43.3],[-25,43.3],[-50,0],[-25,-43.3],[25,-43.3]],"closed":true,"layer":"0"}
  ]
}

星形与多顶点图形速算（务必按公式计算，不要凭感觉估点）：
- 五角星 = 10 个顶点的闭合多段线（不是 5 个顶点！），外半径 R 与内半径 0.382R 交替，
  自正上方 (0,R) 起每 36° 取一个顶点。R=50 时的 10 个顶点：
  [0,50],[-11.2,15.5],[-47.6,15.5],[-18.2,-5.9],[-29.4,-40.5],
  [0,-19.1],[29.4,-40.5],[18.2,-5.9],[47.6,15.5],[11.2,15.5]
- N 角星 = 2N 个顶点，角度每次 +180/N 度，内外半径交替；正 N 边形 = N 个顶点均匀分布在
  外接圆上（自正上方起每 360/N 度一个）。

画现实物体（葫芦、雨伞、猫、汽车、台灯这类具体东西）时，不要凭空撒点，按"部件拆解"来画：
1) 先把物体拆成 2~5 个部件：主体轮廓 + 附件（开口/盖子/把手/轮子/耳朵/尾巴）+ 关键细节；
2) **主体轮廓一律用 spline**（闭合就 closed:true），给 10~20 个轮廓点即可——点在曲线上，
   系统会自动拟合成光滑样条；**不要**再用 16~24 个点的多段线（那只会得到一串直线段）；
3) 对称物体只算半边坐标，再镜像到另一边，保证左右对齐；
4) 圆形件（轮子/孔/眼睛/球）用 circle，椭圆件（蛋/脸/盘口）用 ellipse，
   弯杆、弯钩、伞面弧、耳廓这类单段弯曲用 arc 或 spline，杆和柄的直段才用 line/rect；
5) 实在没把握就画「最简可识别剪影」，但绝不能输出互不相连、随机散布的折线。
示例：
- 葫芦 = 下大球 + 上小球 + 中间收腰：一条 closed spline 描述整个外轮廓（左右对称，
  从顶部凹口出发：左侧向下外扩→下球最宽→收底→右侧对称上行），再加顶部小蒂；
- 雨伞 = 伞面（arc 半圆，或 spline 画 3 个扇贝下缘）+ 中间直伞杆 + 弯钩手柄（arc/spline）；
- 猫（正面）= 头（ellipse 或 closed spline）+ 两只三角耳（polyline）+ 两只眼（ellipse 或 circle）
  + 三角鼻 + 两侧胡须（line）。

示例4（曲线优先的写法）：
指令：画一个葫芦，高 150 宽 90
输出：
{
  "name": "gourd",
  "unit": "mm",
  "entities": [
    {"type":"spline","closed":true,"layer":"0","points":[
      [0,75],[-22,62],[-38,40],[-45,10],[-44,-20],[-36,-48],[-22,-68],[-8,-75],
      [8,-75],[22,-68],[36,-48],[44,-20],[45,10],[38,40],[22,62]]},
    {"type":"arc","center":[0,78],"radius":9,"start":0,"end":180,"layer":"0"},
    {"type":"line","p1":[-9,78],"p2":[9,78],"layer":"0"}
  ]
}
（注意：没有用一条条直线去拼轮廓，也没有为了"圆"而塞 50 个点。）

不规则多边形（组合/定制概念）：若用户给出"各边长度不同"的图形，例如
"五边形 边长 10/20/30/25/15"或"三角形 边长 30 40 50"，请按给定边长计算各顶点坐标——
顶点共圆、相邻顶点弦长=给定边长（用二分法解外接圆半径 R 使 Σ2·asin(s_i/2R)=2π），
输出为闭合 polyline；不要画成边长相等的正多边形。多个独立图形（如"画一个圆和一个正方形"）
分别列出各自实体即可，系统会自动排版。"""


VISION_SYSTEM_PROMPT = """你是一个专业的 2D CAD 制图助手，能够理解图片（如手绘草图、CAD 截图、示意图）。
用户会给你一张图片，可能还附带文字说明。请你识别图片中的 2D 图形（形状、相对位置、尺寸比例等），
并转换为精确的几何图元数据，以 JSON 输出。

**用户指令优先级最高**：用户的文字指令必须优先于"照搬整张图片"，常见指令及处理方式：
- 颜色过滤（如"去掉红色标注，只保留黑色线条"）：仔细观察每条线的颜色，
  只输出符合要求的颜色的图元，其他颜色的一律不输出（哪怕它们更显眼）；
- 局部提取（如"只画左边的图形"/"只要中间那个圆"）：只输出对应部分的图元；
- 视图/投影转换（如"画成左视图"）：在思考中完成投影几何变换后再输出；
- 尺寸指定（如"边长改为 100"）：以用户指定尺寸覆盖图中比例。
用户指令与图片内容冲突时，以用户指令为准。

工作方法（务必遵循，可显著提升坐标精度）：
1. 先在思考中把图片划分为 10×10 的网格（列 0-9 从左到右，行 0-9 从下到上），
   逐一列出看到的每个图元及其网格坐标，例如：
   - 直线：(1,2)→(8,2)
   - 圆：圆心(5,5)，半径 1.5 格
   - 矩形：左上(2,3)，右下(7,7)
   **所有直线段必须一根不漏，端点到端点。**
2. 再把网格坐标换算成毫米：整个图像映射为 100mm 宽（高按图片宽高比折算），
   左下角为 (0,0)，X 向右，Y 向上。
3. 最后只输出一个 JSON 对象。

坐标单位：毫米(mm)。原点(0,0)在左下角，X 向右，Y 向上。
你必须自己完成所有几何计算（如根据图片估算或按说明确定尺寸、多边形顶点坐标、孔位阵列等），并给出准确坐标。
若图片中有文字标注的尺寸，请优先采用标注尺寸；若没有明确尺寸，请基于图片比例给出合理的尺寸。

只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块。JSON 结构：
{
  "name": "图形名称(英文或拼音，用于文件名)",
  "unit": "mm",
  "entities": [
    {"type":"line","p1":[x,y],"p2":[x,y],"layer":"0"},
    {"type":"circle","center":[x,y],"radius":r,"layer":"0"},
    {"type":"arc","center":[x,y],"radius":r,"start":起始角度度,"end":结束角度度,"layer":"0"},
    {"type":"ellipse","center":[x,y],"rx":半长轴,"ry":半短轴,"rotation":长轴倾角度,"layer":"0"},
    {"type":"spline","points":[[x,y],...],"closed":false,"layer":"0"},
    {"type":"rect","x":左下角x,"y":左下角y,"w":宽,"h":高,"layer":"0"},
    {"type":"polyline","points":[[x,y],...],"closed":false,"layer":"0"},
    {"type":"text","p":[x,y],"height":5,"value":"文字","rotation":0,"layer":"0"},
    {"type":"point","p":[x,y],"layer":"0"}
  ]
}
要求：实体坐标必须自洽、闭合图形务必准确；尺寸合理；图名简洁。

【弯曲的线必须用曲线表达】
图纸里"弯"的部分不要拆成许多短直线（那样放大后全是棱角），按下面选：
- 自由曲线轮廓（钣金展开边、凸轮轮廓、水波、手绘弧线、圆角过渡）→
  {"type":"spline","points":[沿曲线 8~20 个点],"closed":true/false}（点会被拟合成过点的光滑样条）
- 椭圆/长圆孔/斜投影的圆 → ellipse；整圆 → circle；单段圆弧（倒角、半圆、弯钩）→ arc
- 只有真正的直边才用 line / rect / polyline
判断口诀：拿尺子画的才用直线，靠手一笔画弯的用 spline/arc/ellipse。

星形与多顶点图形速算（遇到★/五角星/正多边形务必按公式计算，不要凭感觉估点）：
- 五角星 = 10 个顶点的闭合多段线（不是 5 个顶点！），外半径 R 与内半径 0.382R 交替，
  自正上方起每 36° 一个顶点；N 角星 = 2N 个顶点，角度每次 +180/N，内外半径交替；
  正 N 边形 = N 个顶点均匀分布在外接圆上（自正上方起每 360/N 度一个）。
  注意：星形与多边形是**折角**，必须用 polyline，**不能**用 spline（会变成圆角的胖星）。

不规则多边形：若用户给出各边长度（如"五边形 边长 10/20/30/25/15"），请按给定边长计算
各顶点坐标（顶点共圆、弦长=边长），输出为闭合 polyline；不要画成边长相等的正多边形。

示例（图片为一个边长100的正方形，中心在原点）：
输出：
{
  "name": "square_100",
  "unit": "mm",
  "entities": [
    {"type":"rect","x":-50,"y":-50,"w":100,"h":100,"layer":"0"}
  ]
}"""


def _encode_image(path: str) -> str:
    """把图片文件编码为 base64 字符串（Ollama 视觉模型直接吃 base64）。"""
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def call_ollama(prompt: str, model: str, host: str, timeout: int = 600,
               images: list = None, system_prompt: str = None) -> str:
    """向 Ollama 要一次完整回答（内部是流式读取 + 三道时限）。

    旧实现用 urllib 一次性阻塞读（timeout 最长 600 秒）：模型冷加载或卡住时，
    网页就转圈好几分钟，看起来像「死机」。现在统一走 ollama_stream：
      * 出首字 / 停滞 / 总时长都有硬上限，卡住也能在几十秒内报错而不是无限等；
      * 带 keep_alive，模型用完在显存里留 30 分钟，不会 5 分钟一卸载→下次又冷加载；
      * 已加载的模型用较短的首字时限，冷加载才放宽，避免白等。
    """
    sys_text = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    messages = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    user_msg = {"role": "user", "content": prompt}
    if images:
        user_msg["images"] = list(images)
    messages.append(user_msg)

    # 冷加载（模型不在显存里）宽松些；已加载就按正常速度收紧
    first = 200.0 if images else 150.0
    try:
        if ollama_stream.is_loaded(host, model):
            first = 60.0
    except Exception:  # noqa: BLE001
        pass
    total = max(120.0, min(float(timeout or 600), 360.0))
    text, err = ollama_stream.stream_ollama_chat(
        messages, model=model, host=host, temperature=0.2,
        first_token=first, stall=60.0, total=total,
        keep_alive=ollama_stream.DEFAULT_KEEP_ALIVE)
    if err:
        raise RuntimeError(err)
    return text


# ---------- 健壮的 JSON 提取（应对模型各种不规范输出） ----------

# 全角标点 -> 半角（模型经常顺手打出中文标点导致 JSON 破裂）
_FULLWIDTH_MAP = {
    "，": ",", "：": ":", "；": ";", "、": ",",
    "（": "(", "）": ")", "【": "[", "】": "]",
    "｛": "{", "｝": "}", "“": '"', "”": '"',
    "‘": "'", "’": "'", "％": "%", "－": "-", "．": ".",
}

# Python/JS 字面量 -> JSON 字面量
_LITERAL_FIXES = [
    (r"\bTrue\b", "true"), (r"\bFalse\b", "false"),
    (r"\bNone\b", "null"), (r"\bNaN\b", "null"),
    (r"\bInfinity\b", "null"), (r"\bundefined\b", "null"),
]


def _sanitize_json_text(s: str) -> str:
    """修复模型输出 JSON 的常见坏习惯（字符串感知，不误伤字符串内容）。"""
    for a, b in _FULLWIDTH_MAP.items():
        s = s.replace(a, b)
    s = s.replace("\ufeff", "").replace("\u200b", "")
    out = []
    i, n = 0, len(s)
    in_str, quote, esc = False, "", False
    while i < n:
        ch = s[i]
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == quote:
                out.append(ch)
                in_str = False
            elif ch in "\n\r\t":
                # 字符串里的裸控制字符：JSON 不允许，转义之
                out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            elif ord(ch) < 0x20:
                pass  # 丢弃其他控制字符
            else:
                out.append(ch)
            i += 1
            continue
        if ch in "\"'":
            in_str, quote = True, ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and s[i + 1] == "/":      # 行注释
            j = s.find("\n", i)
            i = n if j == -1 else j
            continue
        if ch == "/" and i + 1 < n and s[i + 1] == "*":      # 块注释
            j = s.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(ch)
        i += 1
    s = "".join(out)
    s = re.sub(r",(\s*[}\]])", r"\1", s)                    # 尾随逗号
    # 相邻容器之间缺逗号（模型高频缺陷，如 "}\n    {" -> "},\n    {"）
    s = re.sub(r'([}\]])\s*([{\[])', r"\1,\2", s)
    for pat, rep in _LITERAL_FIXES:                         # Python 字面量
        s = re.sub(pat, rep, s)
    return s


def _try_loads(s: str):
    """先严格解析，失败则清洗后再试。返回归一化后的 dict 或 None。"""
    for candidate in (s, _sanitize_json_text(s)):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return _normalize_spec(v)
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _balanced_braces(t: str):
    """字符串感知地切出所有顶层平衡 {...} 片段，返回 [(start, end, text)]。"""
    spans = []
    depth, start = 0, -1
    in_str, quote, esc = False, "", False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append((start, i + 1, t[start:i + 1]))
                    start = -1
    return spans


def _normalize_spec(d: dict) -> dict:
    """统一字段名：elements/shapes/objects -> entities；补 name/unit 默认值。"""
    if not isinstance(d, dict):
        return d
    for key in ("entities", "elements", "shapes", "objects", "primitives"):
        if isinstance(d.get(key), list):
            if key != "entities":
                d["entities"] = d.pop(key)
            break
    d.setdefault("name", "drawing")
    d.setdefault("unit", "mm")
    return d


def extract_json(text: str) -> dict:
    """从模型回复中提取几何 JSON。

    兼容：```json 围栏、前后解释文字、多个 JSON 对象粘连、全角标点、
    // 和 /* */ 注释、尾随逗号、Python 字面量(True/None)、字符串内裸换行等。
    全部失败时进入实体抢救模式（把零散的图元对象拼成图纸）。
    """
    if not text or not text.strip():
        raise ValueError("模型输出为空")
    t = text.strip()

    def is_spec(d):
        return isinstance(d, dict) and isinstance(d.get("entities"), list)

    # 1) 整体直接解析（最常见路径）
    d = _try_loads(t)
    if is_spec(d):
        return _normalize_spec(d)

    # 2) 去掉 markdown 围栏后再试
    if "```" in t:
        for p in t.split("```"):
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            d = _try_loads(p)
            if is_spec(d):
                return _normalize_spec(d)

    # 3) 扫描所有平衡 {...} 片段（模型输出多个对象/带前后废话时）
    specs, entity_objs = [], []
    spans3 = _balanced_braces(t)
    for _s, _e, span in spans3:
        d = _try_loads(span)
        if d is None:
            continue
        if isinstance(d.get("entities"), list):
            specs.append(d)
        elif isinstance(d.get("type"), str):
            entity_objs.append(d)
    if specs:
        # 模型惯用"先给示例再给正图"：选图元最多的；数量相同时取最后一个（修正版在后）
        best = specs[0]
        for s in specs[1:]:
            if len(s.get("entities") or []) >= len(best.get("entities") or []):
                best = s
        return _normalize_spec(best)

    # 4) raw_decode 兜底（大括号不平衡、片段前有垃圾字符时）
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        if any(s <= i < e for s, e, _ in spans3):
            continue  # 已扫描过的区间，避免重复收集
        try:
            v, _ = dec.raw_decode(t[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            if isinstance(v.get("entities"), list):
                return _normalize_spec(v)
            if isinstance(v.get("type"), str):
                entity_objs.append(v)

    # 5) 实体抢救：零散图元对象拼成图纸
    if entity_objs:
        return _normalize_spec({"name": "salvaged", "unit": "mm",
                                "entities": entity_objs})

    # 6) 彻底失败：带输出尾部报错，便于定位
    tail = t[-200:].replace("\n", " ")
    raise ValueError(f"无法从模型输出中解析出几何 JSON（输出尾部：…{tail}）")


def _safe_name(name: str) -> str:
    name = (name or "drawing").strip()
    for ch in r'\/:*?"<>| ':
        name = name.replace(ch, "_")
    return name[:60] or "drawing"


def _homotope_spec_from_image(image, instruction: str, preset: str = "centerline",
                              name: str = "homotope"):
    """Homotope 引擎路线：拓扑分割 + 中线描摹，输出 SVG → 转成实体。

    与 OpenCV 路线互补：OpenCV 靠霍夫/圆检测硬拼几何（工程图最准），
    Homotope 按色块分割、共享边界只拟合一次，擅长手绘/软边/自由曲线图。
    引擎需要 Node ≥22；不可用时返回 (None, reason)，调用方自动退回其他路线。
    """
    try:
        import homotope_bridge as _hb
    except Exception as e:                              # noqa: BLE001
        return None, "Homotope 桥接不可用（%s）" % str(e)[:80]
    if not _hb.available():
        return None, _hb.reason_unavailable()
    # Homotope CLI 只吃文件路径：base64 先落临时盘
    path = image
    tmp = None
    if isinstance(image, str):
        s = image.strip()
        if s.startswith("data:") or (len(s) > 300 and "," not in s[:64]):
            try:
                import tempfile
                import base64
                raw = s.split(",", 1)[1] if "," in s[:64] else s
                tmp = os.path.join(tempfile.gettempdir(),
                                   "_ht_%s.png" % os.getpid())
                with open(tmp, "wb") as f:
                    f.write(base64.b64decode(raw))
                path = tmp
            except Exception as e:                      # noqa: BLE001
                return None, "图片解码失败（%s）" % str(e)[:60]
        elif not os.path.exists(s):
            return None, "图片不存在：%s" % s[:60]
    print("[1/3] Homotope 引擎矢量化（%s 预设，纯本地不上传）..." % preset)
    try:
        spec = _hb.build_spec(path, instruction or "", max_dim=210.0,
                              preset=preset, name=name)
    except Exception as e:                              # noqa: BLE001
        return None, "Homotope 失败（%s）" % str(e)[:80]
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:                           # noqa: BLE001
                pass
    if not spec.get("_meta", {}).get("ok"):
        return None, spec.get("_meta", {}).get("reason") or "Homotope 未检出图元"
    counts = spec["_meta"].get("counts", {})
    st = spec["_meta"].get("homotope", {})
    print("    Homotope 检出：" + ("，".join(f"{v}个{k}" for k, v in counts.items() if v)
                                or "0 个图元")
          + f"（{st.get('ms', '?')}ms，{st.get('paths', '?')} 条路径）")
    return spec, None


def _spec_is_weak(spec) -> bool:
    """CV 结果是否"勉强"——该让 Homotope 试一把。

    两种情况算弱：①图元太少（<3，通常只认出一两条边就放弃了）；
    ②**全是碎直线**（≥20 条 line 且没有圆/弧/矩形）：正是"曲线被切成短弦"
    的典型症状，Homotope 的中线描摹会给出连贯曲线。
    """
    ents = (spec or {}).get("entities") or []
    if len(ents) < 3:
        return True
    kinds = {}
    for e in ents or ():
        if isinstance(e, dict):
            t = str(e.get("type") or "").strip().lower()
            kinds[t] = kinds.get(t, 0) + 1
    if not (kinds.get("circle") or kinds.get("arc") or kinds.get("rect")
            or kinds.get("spline") or kinds.get("ellipse")) \
            and kinds.get("line", 0) >= 20:
        return True
    return False


def _cv_spec_from_image(image, instruction: str, out_dir: str, passes: int = None,
                        bw_first: bool = True):
    """CV 路线（经典 OpenCV 视觉）：从图片直接提取线条/圆/矩形，不经过大模型。

    image 可以是：文件路径 / 裸 base64 / data:image URL。
    passes：霍夫直线遍数；None=默认多遍（6+2），1=单遍快跑（网页"多次识别"关）。
    bw_first：先把图片压成高对比纯黑白（白底黑线）再识别（bw_photo，默认开）。
    返回 (spec 或 None, debug_path 或 None)。spec 为 None 表示 CV 没检出足够图元；
    黑白预处理图落在 spec["_meta"]["bw_path"]（有 bw_first 时）。
    """
    try:
        from line_extractor import extract_spec, load_image, load_image_b64, available
    except Exception:
        return None, None
    if not available():
        return None, None
    img = None
    if isinstance(image, str):
        s = image.strip()
        if s.startswith("data:"):
            img = load_image_b64(s)
        elif len(s) > 300 and "," not in s[:64]:
            img = load_image_b64(s)          # 裸 base64
        elif os.path.exists(s):
            img = load_image(s)              # 文件路径
    if img is None:
        return None, None
    debug_dir = os.path.join(out_dir, "_debug")
    try:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"cv_{int(time.time() * 1000)}.png")
    except Exception:
        debug_path = None
    bw_path = None
    if bw_first:
        # 颜色过滤必须吃**原图**：bw_photo 会把彩色线一律压成黑线（青色标注、
        # 绿色中心线、白色轮廓全都变成同一种黑），之后再按颜色过滤必然错位——
        # 暗底白线图上"只留白"会把整片背景判成目标色、整张反转（实测 0 图元），
        # "只留黑"则等于全保留（实测 52 条杂线）。有色过滤指令就跳过黑白化。
        try:
            from line_extractor import parse_color_filter as _pcf
            _kc, _dc = _pcf(instruction or "")
            if _kc or _dc:
                bw_first = False
                print("    （检测到颜色过滤指令，跳过黑白化：颜色判定必须用原图）")
        except Exception:                    # noqa: BLE001 - 解析失败按无过滤处理
            pass
    if bw_first:
        try:
            import cv2
            from line_extractor import bw_photo
            bwimg = bw_photo(img)
            if bwimg is not None and debug_dir:
                bw_path = os.path.join(debug_dir, f"bw_{int(time.time() * 1000)}.png")
                ok, buf = cv2.imencode(".png", bwimg)
                if ok:
                    buf.tofile(bw_path)      # 中文路径安全写盘
                    img = bwimg              # 之后全部识别都吃这张纯黑白图
                else:
                    bw_path = None
        except Exception:                    # noqa: BLE001 - 预处理失败就用原图
            bw_path = None
    spec = extract_spec(img, instruction=instruction or "", name="cv_drawing",
                        debug_path=debug_path, passes=passes)
    if bw_path:
        try:
            spec.setdefault("_meta", {})["bw_path"] = bw_path
        except Exception:
            pass
    if not spec.get("_meta", {}).get("ok"):
        return None, debug_path
    return spec, debug_path


def _build_dxf_from_spec(spec: dict, out_dir: str, open_libre: bool,
                         method: str, debug_path: str = None,
                         note: str = "", flatten_layer: str = None,
                         curveify: bool = None, fmt: str = "dxf",
                         dwg_version: str = None) -> dict:
    """把几何 spec 写成 DXF（可选再转一份 DWG），返回详细结果（供 CLI 与 Web 共用）。

    flatten_layer: 若给定（如 "0"），把全部实体归到该单层——用于"合成进一个图层"。
    curveify: 是否把"本来就是曲线"的折线升级成真 B 样条。默认对大模型 / 概念库 /
              图形库的产物开启；经典视觉（OpenCV）与编辑器手绘保持折线原样——
              前者是精确测绘结果、后者是用户亲手画的，都不该被偷偷改形状。
    fmt: "dxf"（默认）/ "dwg" / "both"。DWG 由 dwg_export.py 调外部转换器生成；
         **DXF 永远会写**，它是 DWG 的中间产物，也是 LibreCAD 预览用的格式。
         DWG 转换失败不算整体失败：图纸照样有 DXF，错误原因随结果返回。
    """
    entities = spec.get("entities", [])
    n_curve = {"spline": 0, "ellipse": 0}
    if curveify is None:
        # homotope 的输出全是细分点列（一个圆就是 30+ 点的闭合 polyline），
        # 必须曲线化，否则写进 DXF 又是一堆折线。
        curveify = method in ("concepts", "shapes", "vlm", "homotope")
    # 经典视觉（OpenCV）识别结果：单条精确直线保留原样（那是真实直边），
    # 但用户要求"尽可能用曲线、少用直线"——首尾相接的碎弦链、弯曲的多段线
    # 都要用放宽判据升级成样条/弧。curveify/chain 的保守闸门（顶点数、缓拐率）
    # 是给模型输出设计的，霍夫碎弦经常 3~4 段一条弯，保守判据永远凑不够。
    cv_loose = method == "cv-lines"
    if curveify or cv_loose:
        try:
            import curves as _curves
            entities = _curves.curveify(entities, loose=cv_loose)
            # 第二道：把"用一串短直线拼出来的弧/缓弯"串成一条样条。模型输出弧线时
            # 经常给几十条独立 line（每条 2 点），单看每一条都不像曲线，
            # 只有先按几何连通性串起来才能认出它其实是一条弧。
            entities = _curves.chain_lines_to_curves(
                entities, loose=cv_loose, min_seg=3 if cv_loose else 5)
            n_curve = _curves.count_curves(entities)
        except Exception as e:                          # noqa: BLE001
            print(f"[提示] 曲线化跳过（{e}）")
    dxf = build_from_entities(entities, unit=spec.get("unit", "mm"),
                              flatten_layer=flatten_layer)
    if not dxf.entities:
        raise ValueError("图元数据全部无效，未生成任何实体")
    name = _safe_name(spec.get("name", "drawing"))
    if method == "cv-lines":
        name = name if name != "cv_drawing" else "sketch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.dxf")
    # 避免覆盖：若存在则加序号
    if os.path.exists(out_path):
        i = 1
        while os.path.exists(os.path.join(out_dir, f"{name}_{i}.dxf")):
            i += 1
        out_path = os.path.join(out_dir, f"{name}_{i}.dxf")
    dxf.save(out_path)
    skip_note = f"，跳过 {dxf.skipped} 个异常图元" if dxf.skipped else ""
    counts = spec.get("_meta", {}).get("counts", {})
    if (curveify or cv_loose) and entities is not spec.get("entities"):
        # 曲线化会改图元构成（几十条 line 变 1 条 spline），统计要按最终实体算，
        # 否则提示里还在说"24个line"，跟实际图纸对不上。
        counts = {}
        for e in entities:
            t = str((e or {}).get("type") or "").strip().lower()
            if t:
                counts[t] = counts.get(t, 0) + 1
    count_note = "，".join(f"{v}个{k}" for k, v in counts.items() if v) or f"{len(dxf.entities)}个图元"
    cf = spec.get("_meta", {}).get("color_filter")
    cf_note = f"｜{cf}" if cf else ""
    print(f"[3/3] 已生成 DXF：{out_path}（{count_note}{skip_note}）{cf_note}")
    # ---- DWG：DXF 已落盘，再按需转一份（转不动就如实上报，DXF 照常交付）----
    want_dwg = str(fmt or "dxf").strip().lower() in ("dwg", "both", "dxf+dwg", "dxf_dwg")
    dwg_path, dwg_error, dwg_note = None, None, ""
    if want_dwg:
        try:
            import dwg_export
            res = dwg_export.convert(out_path, version=dwg_version,
                                     entities=entities,
                                     unit=spec.get("unit", "mm"))
        except Exception as e:                          # noqa: BLE001
            res = {"ok": False, "note": "%s: %s" % (type(e).__name__, e)}
        if res.get("ok"):
            dwg_path = res.get("path")
            dwg_note = res.get("note") or ""
            print(f"[4/4] 已生成 DWG：{dwg_path}（{dwg_note}）")
        else:
            dwg_error = res.get("note") or "DWG 转换失败"
            print(f"[4/4] DWG 未生成：{dwg_error}（DXF 已正常输出）")
    if open_libre:
        open_in_librecad(out_path)
    n_p = ((spec.get("_meta") or {}).get("passes") or {}).get("passes")
    _cv_name = ("经典视觉(OpenCV %d遍融合)" % n_p) if n_p else "经典视觉(OpenCV)精确线条"
    if (spec.get("_meta") or {}).get("organic"):
        # 有机线稿（螺旋/云纹/手绘曲线）走的是"骨架 → 有序路径 → 样条"，
        # 提示语里说清楚，用户才知道图上的曲线是怎么来的
        _cv_name += " + 骨架曲线化"
    method_names = {"cv-lines": _cv_name,
                    "shapes": "服务器图形库(确定性几何)",
                    "concepts": "现实物体概念库(确定性几何)"}
    prefix = note or ""
    if prefix and prefix[-1] not in "；;。，,：: ":
        prefix += "；"          # 免得「…已写入 CAD识别方式：」黏成一坨
    curve_note = ""
    if n_curve["spline"] or n_curve["ellipse"]:
        parts = []
        if n_curve["spline"]:
            parts.append(f"{n_curve['spline']} 条样条曲线")
        if n_curve["ellipse"]:
            parts.append(f"{n_curve['ellipse']} 个椭圆")
        curve_note = "📐 " + " + ".join(parts) + "；"
    msg = prefix + curve_note + cf_note + f"识别方式：{method_names.get(method, '本地大模型')}"
    # DWG 相关提示接在末尾：成功报版本+后端，失败说清原因（不静默吞掉）
    if dwg_path:
        msg += f"；DWG 已输出（{dwg_note}）"
    elif dwg_error:
        msg += f"；DWG 输出失败：{dwg_error}"
    return {"path": out_path, "method": method, "entities": len(dxf.entities),
            "entities_list": entities, "skipped": dxf.skipped,
            "debug_path": debug_path, "message": msg,
            "fmt": "dwg" if (want_dwg and dwg_path and
                             str(fmt).strip().lower() == "dwg") else "dxf",
            "dwg": dwg_path, "dwg_error": dwg_error, "dwg_version": dwg_note or None,
            # main = 用户这次要的那份（要 DWG 且转成功就给 DWG，否则退回 DXF）
            "main": dwg_path if (dwg_path and str(fmt).strip().lower() == "dwg")
                    else out_path}


def _cleaned_image_b64(image, keep, drop):
    """颜色兜底第一层：把原图按颜色"擦"成只剩目标颜色线条（黑线白底），
    以 base64 PNG 返回，交给视觉模型识别。

    本地 7B 视觉模型经常无视"去掉红色只留黑色"这类指令，且输出坐标体系不稳定，
    事后按坐标校验不可靠。这里换一个确定性思路：先把图擦干净再给模型看——
    模型物理上看不到非目标颜色的线条，从源头保证过滤生效。
    图中没有目标颜色线条时抛出 ValueError（快速失败，不浪费一次模型调用）。
    """
    import base64 as _b64
    try:
        from line_extractor import (_load_image_any, color_filter_gray,
                                    _binarize, available)
        import cv2
        import numpy as np
        if not available():
            return None
        img = _load_image_any(image)
        if img is None:
            return None

        def _count(gray):
            return 0 if gray is None else int((_binarize(gray) > 0).sum())

        eff_keep = set(keep)
        if not eff_keep:
            # drop-only（如"去掉红色"）：目标=全部颜色，但要排除"背景色本身"——
            # 亮底图的"白"就是背景，暗底图的"黑"就是背景，误入会把图像反转
            hsv0 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bg_light = float(np.median(hsv0[:, :, 2])) >= 128.0
            eff_keep = ({"黑", "灰", "橙", "黄", "绿", "青", "蓝", "紫", "粉", "棕"}
                        if bg_light else
                        {"白", "灰", "橙", "黄", "绿", "青", "蓝", "紫", "粉", "棕"})
        g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        n_all = int((_binarize(g0) > 0).sum())
        gray = color_filter_gray(img, eff_keep, drop)
        n_tgt = _count(gray)
        if n_all > 0 and n_tgt / n_all < 0.005:
            # 严格掩码几乎什么都没留下：对"黑"放宽为消色（黑/灰；暗底再加白）再试
            if keep:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                dark_bg = float(np.median(hsv[:, :, 2])) < 128.0
                achroma = ({"黑", "白", "灰"} if dark_bg else {"黑", "灰"}) - set(drop)
                gray2 = color_filter_gray(img, achroma, drop)
                if _count(gray2) > 0:
                    gray, n_tgt = gray2, _count(gray2)
            if n_tgt == 0:
                raise ValueError(
                    "按颜色过滤后图中没有可绘制的内容——未找到目标颜色的线条")
        ok, buf = cv2.imencode(".png", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        if not ok:
            return None
        return _b64.b64encode(buf.tobytes()).decode("ascii")
    except ValueError:
        raise
    except Exception:
        return None


def _ocr_note(info: dict) -> str:
    """OCR 结果 → 一句中文说明（模块名与参数同名，这里单独包一层避免混淆）。"""
    try:
        import ocr_text as _ocr
        return _ocr.describe(info)
    except Exception:  # noqa: BLE001
        return ""


def _ocr_enrich(spec: dict, image_path: str, host: str, model: str = None) -> dict:
    """给 spec 补上文字图元（OpenCV 定位 + 视觉模型读字）。失败不影响几何。"""
    try:
        import ocr_text
        if not ocr_text.available():
            return {"ok": False, "count": 0, "error": "本机缺少 OpenCV"}
        img = ocr_text.load_image(image_path)
        if img is None:
            return {"ok": False, "count": 0, "error": "图片读取失败"}
        return ocr_text.enrich_spec_with_text(
            spec, img, model=model or VISION_MODEL, host=host)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "count": 0, "error": str(e)[:140]}


def generate_dxf_detailed(instruction: str, model: str, host: str,
                          out_dir: str = CAD_DIR, open_libre: bool = False,
                          image: str = None, mode: str = "auto",
                          ocr_text: bool = False,
                          ocr_model: str = None,
                          fmt: str = "dxf", dwg_version: str = None,
                          multipass: bool = True, bw_first: bool = True,
                          homotope: str = "auto", ht_preset: str = "centerline") -> dict:
    """生成 DXF 并返回详细信息 {"path","dwg","method","entities","debug_path","message"}。

    mode:
      auto  — 有图片时先用经典视觉(OpenCV)提线，失败自动回退视觉模型（默认）
      lines — 强制经典视觉优先（仍会在 CV 失败时回退，保证有产出）
      vlm   — 只用视觉大模型
    ocr_text: 是否同时识别图中文字，把文字作为 text 图元一起写进 DXF。
    fmt: "dxf"(默认) / "dwg" / "both"，是否额外产出一份 DWG（见 dwg_export.py）。
    multipass: 经典视觉是否多次识别（多遍霍夫融合）。False=只跑 1 遍 OpenCV（快）。
    bw_first: 经典视觉先把图片压成高对比纯黑白（白底黑线）再识别（bw_photo）。
    homotope: "auto"(默认，CV 失败或结果很弱时让 Homotope 接手) / "on"(只用
              Homotope) / "off"（禁用）。引擎需 Node ≥22，缺了自动跳过。
    ht_preset: Homotope 预设，默认 centerline（笔画描成单条中线，最适合线稿）。
"""
    note = ""
    ocr_note = ""
    ht_on = str(homotope or "auto").strip().lower() in ("auto", "on", "1", "true")
    # ---- 路线 A0：Homotope 引擎（"on" 时独占）----
    if image and str(homotope or "").strip().lower() in ("on", "1", "true"):
        hspec, herr = _homotope_spec_from_image(image, instruction, ht_preset)
        if hspec:
            if ocr_text:
                oinfo = _ocr_enrich(hspec, image, host, ocr_model)
                ocr_note = _ocr_note(oinfo)
            return _build_dxf_from_spec(hspec, out_dir, open_libre,
                                        method="homotope", debug_path=None,
                                        note=note + ocr_note,
                                        fmt=fmt, dwg_version=dwg_version)
        note = (herr or "Homotope 未产出图元") + "，已回退；"
        print("    " + note)

    # ---- 路线 A：经典视觉（OpenCV）提线，精度最高，不经过大模型 ----
    if image and mode in ("auto", "lines"):
        print("[1/3] 经典视觉(OpenCV)提取线条/圆/矩形...")
        spec, debug_path = _cv_spec_from_image(
            image, instruction, out_dir,
            passes=None if multipass else 1, bw_first=bw_first)
        if spec:
            counts = spec["_meta"].get("counts", {})
            detail = "，".join(f"{v}个{k}" for k, v in counts.items() if v)
            print(f"    CV 检出：{detail}（无需调用大模型）")
            # ---- A1：CV 结果很弱 → 让 Homotope 顶上（辅助识别的核心）----
            if ht_on and _spec_is_weak(spec):
                hspec, herr = _homotope_spec_from_image(image, instruction, ht_preset)
                if hspec and len(hspec.get("entities") or ()) > len(spec.get("entities") or ()):
                    print("    CV 结果偏弱（碎直线/图元过少），改用 Homotope 结果")
                    spec, debug_path, method_used = hspec, None, "homotope"
                else:
                    method_used = "cv-lines"
                    if hspec:
                        print("    Homotope 图元不占优，保留 CV 结果")
            else:
                method_used = "cv-lines"
            if ocr_text:
                print("[1.5/3] 识别图中文字（OpenCV 定位 + 视觉模型读字）...")
                oinfo = _ocr_enrich(spec, image, host, ocr_model)
                ocr_note = _ocr_note(oinfo)
                print("    " + (ocr_note or "（无文字）"))
            return _build_dxf_from_spec(spec, out_dir, open_libre,
                                        method=method_used, debug_path=debug_path,
                                        note=note + ocr_note,
                                        fmt=fmt, dwg_version=dwg_version)
        # ---- A2：CV 完全失败 → Homotope 兜底，再不行才回退视觉模型 ----
        if ht_on:
            hspec, herr = _homotope_spec_from_image(image, instruction, ht_preset)
            if hspec:
                note = "经典视觉未检出图元，已由 Homotope 引擎接手；"
                if ocr_text:
                    oinfo = _ocr_enrich(hspec, image, host, ocr_model)
                    ocr_note = _ocr_note(oinfo)
                return _build_dxf_from_spec(hspec, out_dir, open_libre,
                                            method="homotope", debug_path=None,
                                            note=note + ocr_note,
                                            fmt=fmt, dwg_version=dwg_version)
        note = "经典视觉未检出足够图元，已回退视觉模型；"
        print("    CV 未检出足够图元，回退到视觉模型...")

    # ---- 路线 B：视觉/文本大模型 ----
    keep_c, drop_c, cleaned_used = set(), set(), False
    if image:
        model = VISION_MODEL
        sys_prompt = VISION_SYSTEM_PROMPT
        base_instruction = instruction or (
            "请识别这张图片中的 2D 图形，按系统要求输出几何 JSON 并绘制。")
        try:
            from line_extractor import parse_color_filter
            keep_c, drop_c = parse_color_filter(base_instruction)
        except ImportError:
            keep_c, drop_c = set(), set()
        images_b64 = None
        if keep_c or drop_c:
            # 颜色兜底第一层：先把图擦成只剩目标色线条，再交给模型——
            # 模型物理上看不到非目标颜色的线，从源头保证过滤生效
            cleaned = _cleaned_image_b64(image, keep_c, drop_c)
            if cleaned:
                images_b64 = [cleaned]
                cleaned_used = True
                effective_instruction = base_instruction + (
                    "\n（注：图片已经过颜色预处理，非目标颜色的线条已被擦除，"
                    "请把图中剩下的所有线条全部画出，不要遗漏）")
                print("    已按颜色擦图（只留目标色线条）后交给视觉模型")
        if images_b64 is None:
            images_b64 = [_encode_image(image)]
            effective_instruction = base_instruction
        print(f"[1/3] 已向视觉模型 {model} 发送图片 + 指令：{effective_instruction[:60]}...")
    else:
        # ---- 概念库直出：葫芦/雨伞/汽车等现实物体 ----
        # 7B 模型对具体物体没有稳定的形状记忆（"画一个葫芦"常输出一堆乱折线），
        # concepts.py 用"部件 + 轮廓控制点"的确定性配方直接出图，还能吃尺寸。
        try:
            from concepts import parse_concept_instruction
            concept_spec, concept_desc = parse_concept_instruction(instruction)
        except Exception as e:                      # noqa: BLE001
            concept_spec, concept_desc = None, ""
            print(f"    [概念库不可用：{e}]")
        if concept_spec:
            print(f"[1/3] 命中物体概念库：{concept_desc}（确定性几何，不经过模型）")
            return _build_dxf_from_spec(
                concept_spec, out_dir, open_libre, method="concepts",
                debug_path=None,
                note=f"🧩 物体概念库：{concept_desc}；",
                fmt=fmt, dwg_version=dwg_version)
        # ---- 图形库直出：五角星/多边形等参数化图形，服务器确定性生成 ----
        # 7B 模型做不好三角计算（实测"画一个五角星"输出 9 顶点乱多边形），
        # 这类指令直接用 shapes.py 精确几何生成：不过模型、毫秒级、零误差。
        try:
            from shapes import parse_shape_instruction
            shape_spec, shape_desc = parse_shape_instruction(instruction)
        except ImportError:
            shape_spec, shape_desc = None, ""
        if shape_spec:
            print(f"[1/3] 命中图形库：{shape_desc}（服务器确定性几何，不经过模型）")
            return _build_dxf_from_spec(
                shape_spec, out_dir, open_libre, method="shapes",
                debug_path=None,
                note=f"📐 图形库直出：{shape_desc}（确定性几何，未经过模型，零误差）；",
                fmt=fmt, dwg_version=dwg_version)
        sys_prompt = None
        images_b64 = None
        effective_instruction = instruction
        print(f"[1/3] 已向本地模型 {model} 发送指令：{instruction[:60]}...")
    raw = call_ollama(effective_instruction, model, host,
                      system_prompt=sys_prompt, images=images_b64)
    spec = None
    last_err = None
    for attempt in range(3):
        try:
            print(f"[2/3] 解析几何 JSON（第 {attempt + 1} 次尝试）...")
            spec = extract_json(raw)
            if not spec.get("entities"):
                raise ValueError("模型未输出任何图元")
            break
        except Exception as e:
            last_err = e
            if attempt >= 2:
                break
            print(f"    JSON 解析失败（{e}），携带原指令让模型重写...")
            fix = (f"你上一次的输出无法被解析为合法 JSON。解析错误：{e}\n"
                   f"原始指令：{effective_instruction}\n"
                   "请重新输出：只输出一个 JSON 对象；不要任何解释、不要代码块、"
                   "不要注释、不要尾随逗号；所有字符串用英文双引号；"
                   "坐标用二维 [x,y] 数字数组。")
            raw = call_ollama(fix, model, host,
                              system_prompt=sys_prompt, images=images_b64)
    if spec is None or not spec.get("entities"):
        raise ValueError(f"模型多次输出均无法解析为几何数据：{last_err}")
    # ---- 颜色兜底：第一层擦图已从源头保证；未能擦图时才启用第二层几何校验 ----
    color_note = ""
    if image and (keep_c or drop_c):
        if cleaned_used:
            color_note = "🧹 颜色兜底：已按颜色擦图后识别（非目标色线条已从输入中擦除）"
        else:
            try:
                from line_extractor import filter_spec_by_color, available
                if available():
                    spec, cinfo = filter_spec_by_color(spec, image, keep_c, drop_c)
                    if cinfo.get("enabled"):
                        color_note = (f"🧹 颜色兜底：{cinfo['desc']}，"
                                      f"已剔除 {cinfo['dropped']} 个非目标颜色图元；")
                    elif cinfo.get("reason"):
                        color_note = f"（颜色兜底未生效：{cinfo['reason']}）"
                    if not spec.get("entities"):
                        raise ValueError(
                            "按颜色过滤后没有剩余图元——图中可能没有目标颜色的线条")
            except ImportError:
                pass
    if image and ocr_text:
        print("[2.5/3] 识别图中文字（OpenCV 定位 + 视觉模型读字）...")
        oinfo = _ocr_enrich(spec, image, host, ocr_model)
        ocr_note = _ocr_note(oinfo)
        print("    " + (ocr_note or "（无文字）"))
    return _build_dxf_from_spec(spec, out_dir, open_libre,
                                method="vlm", debug_path=None,
                                note=note + color_note + ocr_note,
                                fmt=fmt, dwg_version=dwg_version)


def generate_dxf(instruction: str, model: str, host: str,
                 out_dir: str = CAD_DIR, open_libre: bool = False,
                 image: str = None, mode: str = "auto",
                 fmt: str = "dxf", dwg_version: str = None) -> str:
    """兼容旧接口：只返回路径（要 DWG 时返回 DWG，转失败退回 DXF）。"""
    return generate_dxf_detailed(instruction, model, host, out_dir,
                                 open_libre, image, mode,
                                 fmt=fmt, dwg_version=dwg_version)["main"]


# LibreCAD 把草稿模式存在注册表里：2.1.x 用 ...\LibreCAD\Appearance，
# 2.2.x 迁到了 ...\LibreCAD-2\Appearance。两个都写，换版本不会失效。
LIBRECAD_CFG_KEYS = (r"Software\LibreCAD\LibreCAD\Appearance",
                     r"Software\LibreCAD\LibreCAD-2\Appearance")


def ensure_librecad_text_visible() -> None:
    """关掉 LibreCAD 的「草稿模式」。

    这个开关一开，LibreCAD 会把**所有**文字画成空心方框（为了省性能）。用户
    看到的就是"网页里明明标着 Φ16/100，LibreCAD 里一个标注都看不见"。开关
    存在注册表（见 LIBRECAD_CFG_KEYS），LibreCAD 只在退出时整包回写，所以每次
    打开图纸前都设一次（幂等）；已经在运行的实例要重启一次才会生效。
    """
    if os.name != "nt":
        return
    try:
        import winreg
    except ImportError:
        return
    for path in LIBRECAD_CFG_KEYS:
        try:
            with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, path,
                                    0, winreg.KEY_SET_VALUE) as k:
                winreg.SetValueEx(k, "DraftMode", 0, winreg.REG_DWORD, 0)
        except OSError:
            pass


def librecad_draft_state() -> int:
    """读 LibreCAD 的草稿模式开关：1=开（文字会画成方框），0=正常，-1=读不到。

    新版（LibreCAD-2）优先；它没建过就退回旧键。
    """
    if os.name != "nt":
        return -1
    try:
        import winreg
    except ImportError:
        return -1
    last = -1
    for path in reversed(LIBRECAD_CFG_KEYS):
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as k:
                return int(winreg.QueryValueEx(k, "DraftMode")[0])
        except OSError:
            continue
    return last


def open_in_librecad(dxf_path: str) -> None:
    if not os.path.exists(LIBRECAD):
        print(f"[提示] 未找到 LibreCAD：{LIBRECAD}，跳过自动打开")
        return
    import subprocess
    ensure_librecad_text_visible()
    try:
        subprocess.Popen([LIBRECAD, dxf_path])
        print(f"[打开] 已用 LibreCAD 打开：{dxf_path}")
    except Exception as e:  # noqa
        print(f"[提示] 无法打开 LibreCAD：{e}")


def _read_instruction_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="离线 CAD 智能体")
    ap.add_argument("instruction", nargs="*", help="画图指令（自然语言）")
    ap.add_argument("--file", help="从文件读取指令")
    ap.add_argument("--watch", help="监听指定指令文件，变动即重新画图")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama 模型名")
    ap.add_argument("--host", default=OLLAMA_HOST, help="Ollama 地址")
    ap.add_argument("--out", default=CAD_DIR, help="DXF 输出目录")
    ap.add_argument("--open", action="store_true", help="生成后用 LibreCAD 打开")
    ap.add_argument("--image", help="传入图片路径(识图模式，启用视觉模型)")
    ap.add_argument("--mode", choices=["auto", "lines", "vlm"], default="auto",
                    help="识图路线：auto=经典视觉优先、失败回退视觉模型(默认)；"
                         "lines=经典视觉优先(OpenCV精确提线)；vlm=只用视觉大模型")
    ap.add_argument("--format", "--fmt", dest="fmt", default="dxf",
                    choices=["dxf", "dwg", "both"],
                    help="输出格式：dxf(默认) / dwg / both(两份都要)。"
                         "DWG 由包内 LibreDWG 转换，需 tools/libredwg/dxf2dwg.exe")
    ap.add_argument("--dwg-version", default=None,
                    help="DWG 版本：R2000(默认) / R2004 / R2010 / R2018 等，"
                         "取决于可用转换器（LibreDWG 只支持 R2000）")
    ap.add_argument("--dwg-info", action="store_true",
                    help="只检查 DWG 转换器可用性并退出")
    args = ap.parse_args()

    if args.dwg_info:
        try:
            import dwg_export
            st = dwg_export.status()
        except Exception as e:                          # noqa: BLE001
            print(f"[失败] 无法加载 dwg_export：{e}")
            sys.exit(1)
        print(st["note"])
        for b in dwg_export.probe().get("backends", []):
            print(f"  · {b['name']} → {b['path']}（支持 {','.join(b['versions'])}）")
        sys.exit(0 if st["ok"] else 1)

    if args.watch:
        inst_file = args.watch
        print(f"监听模式：监控 {inst_file} ，文件变动即自动画图（Ctrl+C 退出）")
        last = None
        # 先处理一次已有内容
        if os.path.exists(inst_file):
            cur = _read_instruction_from_file(inst_file)
            if cur:
                try:
                    generate_dxf(cur, args.model, args.host, args.out, True,
                                 mode=args.mode, fmt=args.fmt,
                                 dwg_version=args.dwg_version)
                except Exception as e:
                    print(f"[错误] {e}")
                last = os.path.getmtime(inst_file)
        try:
            while True:
                time.sleep(2)
                if not os.path.exists(inst_file):
                    continue
                mtime = os.path.getmtime(inst_file)
                if mtime == last:
                    continue
                last = mtime
                cur = _read_instruction_from_file(inst_file)
                if not cur:
                    continue
                print("--- 检测到新指令 ---")
                try:
                    generate_dxf(cur, args.model, args.host, args.out, True,
                                 mode=args.mode, fmt=args.fmt,
                                 dwg_version=args.dwg_version)
                except Exception as e:
                    print(f"[错误] {e}")
        except KeyboardInterrupt:
            print("已停止监听")
        return

    if args.file:
        instruction = _read_instruction_from_file(args.file)
    else:
        instruction = " ".join(args.instruction)

    if not instruction and not args.image:
        print("请提供画图指令或图片，例如：")
        print('  python cad_agent.py "画一个边长100的正方形居中"')
        print('  python cad_agent.py --image D:/CAD/sketch.png "按此草图绘图"')
        sys.exit(1)

    try:
        out = generate_dxf_detailed(instruction, args.model, args.host,
                                    args.out, args.open, image=args.image,
                                    mode=args.mode, fmt=args.fmt,
                                    dwg_version=args.dwg_version)
        print(f"[完成] {out['main']}")
        if out.get("dwg"):
            print(f"        DWG：{out['dwg']}")
        elif out.get("dwg_error"):
            print(f"        DWG 未生成：{out['dwg_error']}（DXF 已正常输出）")
    except Exception as e:
        print(f"[失败] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
