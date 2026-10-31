"""
web_app.py — J022 Ghoul 本地 Web 界面（零依赖，仅 Python 标准库）

功能：
  - 浏览器中打开 http://127.0.0.1:8765
  - 左栏：与本地 AI 对话（qwen2.5-coder:7b，离线答疑）
  - 右栏：自然语言指令 + 可选上传图片 → DXF → LibreCAD
      自动(推荐) / 线条精确 — 经典视觉(OpenCV)精确提取，识别后可先勾选
                            保留哪些线段再生成 DXF
      AI 理解 — 只用本地视觉大模型（qwen2.5vl）
  - 生成的 DXF 存到 D:/CAD 并用 LibreCAD 打开
  - 识图结果附带叠加图与编号预览图

全程本地，断网可用。Ollama 需已在 localhost:11434 运行。

启动： python web_app.py   然后浏览器访问 http://127.0.0.1:8765
"""

from __future__ import annotations
import base64
import json
import math
import os
import sys
import subprocess
import tempfile
import threading
import time
import uuid
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cad_agent  # noqa: E402
import dxf_writer  # noqa: E402
import line_extractor  # noqa: E402
import ocr_text  # noqa: E402
import ollama_stream  # noqa: E402
from mathtext import sanitize_math, demojify  # noqa: E402
from ollama_stream import stream_ollama_chat  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765
LIBRECAD = cad_agent.LIBRECAD
CAD_DIR = cad_agent.CAD_DIR
CHAT_MODEL = cad_agent.DEFAULT_MODEL  # qwen2.5-coder:7b
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# 关草稿模式（文字变方框的元凶）——实现在 cad_agent 里，CLI 与网页共用一份
ensure_librecad_text_visible = cad_agent.ensure_librecad_text_visible

CHAT_SYSTEM_PROMPT = (
    "你是 J022 Ghoul，一个完全运行在用户本机上的离线 CAD 助手"
    "（Ollama 本地模型 + LibreCAD，断网可用）。"
    "你有一个桌宠叫 Ghoo（你的桌面化身，会陪在主人身边）。你的主人（master）叫 J022BOT。"
    "用简体中文回答，风格简洁、专业、直接，不要客套。"
    "你熟悉机械制图、几何计算、DXF 格式与 LibreCAD 的使用。"
    "关于本工具：右侧 CAD 面板支持自然语言/手绘草图生成 DXF；"
    "自动与线条模式用 OpenCV 精确提线（坐标是算出来的，精度高）；"
    "AI 理解模式用本地视觉大模型（适合语义描述，精度一般）；"
    "识别出的线段可先勾选保留再出图。"
    "做不到的事（如真正的 3D 投影视图）请如实说明，不要编造。"
    "重要：绝对不要使用 LaTeX 数学/化学式语法，所有方程与化学式都用 Unicode 纯文本写出"
    "（例如 2HF + Sb → SbF₃ + H₂，用 ₂ ³ 等上下标字符）。"
    "重要：回复中禁止使用任何 emoji 表情符号（如 😀 👍 ✨ ⚠ 这类彩色符号）；"
    "需要表达情绪时改用颜文字，即由符号拼成的表情，如 (´▽`)、(◕ᴗ◕)、(｡･ω･｡)、(･ω･)b。"
)

# 看图问答 / 识图（OCR 文字 + 图内信息）专用系统提示：走本地视觉大模型 qwen2.5vl
CHAT_VISION_SYSTEM_PROMPT = (
    "你是 J022 Ghoul，一个完全运行在用户本机上的离线 CAD 助手"
    "（本地视觉大模型 + LibreCAD，断网可用），代号 J022 Ghoul。"
    "你的主人（master）叫 J022BOT，你还有一个桌宠分身叫 Ghoo。"
    "用户可能会给你发送一张图片（截图、文档照片、草图、说明书、带文字的图片等）并附带文字提问。"
    "你要仔细阅读图片内容：识别并转写出图中的文字（OCR，忠实原文、保留关键排版与序号），"
    "理解图中的图形、表格、标注与说明，然后以简体中文自然、准确地回答用户的问题。"
    "回答风格：简洁、专业、直接，不要客套；需要时可以分条列出图中要点或原文。"
    "如果用户只是让你「把图中的文字提取出来/转写出来」，就直接给出识别到的文字内容。"
    "如果图片里有 2D 图形且用户问到尺寸/形状/坐标，可结合你的 CAD 知识解读；"
    "但若用户想把图变成 DXF，请引导他去右侧 CAD 面板（那里有更精确的提线能力），不要在此处编造坐标。"
    "看不清楚或图片里没有用户所问内容时，如实说明，不要编造。"
    "重要：回复中禁止使用任何 emoji 表情符号（如 😀 👍 ✨ ⚠ 这类彩色符号）；"
    "需要表达情绪时改用颜文字，即由符号拼成的表情，如 (´▽`)、(◕ᴗ◕)、(｡･ω･｡)、(･ω･)b。"
)

# ---------------------------------------------------------------- 线段选择任务存储
JOBS = {}
_JOBS_LOCK = threading.Lock()
MAX_JOBS = 40

# ---------------------------------------------------------------- 本地模型并发闸门
# Ollama 同一时刻只跑一个生成任务。网页端如果同时发起两个请求（一边聊天、一边
# 识图），后到的会在服务端排队，前端看起来就是「卡住」。这里显式串行，排队过久
# 就给一句人话提示，而不是让浏览器干等到超时。
_MODEL_LOCK = threading.Lock()


def _model_slot(wait: float = 45.0) -> bool:
    """申请模型执行权；等太久就放弃（调用方回「模型正忙」）。"""
    deadline = time.time() + max(0.0, wait)
    while time.time() < deadline:
        if _MODEL_LOCK.acquire(timeout=0.5):
            return True
    return _MODEL_LOCK.acquire(blocking=False)


def _warmup_async():
    """启动后在后台把文本模型预热进显存，避免第一次对话/识图冷加载等半天。

    注意：这里不占用 _MODEL_LOCK——预热只是让 Ollama 把权重读进显存，
    不生成内容；若此时用户就发消息，Ollama 会把请求排在加载之后、加载完立刻出字，
    比让用户先被「模型正忙」挡一次体验更好。
    """
    def _run():
        try:
            ok = ollama_stream.warmup_model(
                cad_agent.OLLAMA_HOST, CHAT_MODEL,
                keep_alive=ollama_stream.DEFAULT_KEEP_ALIVE,
                timeout=240.0)
            if ok:
                print("[warmup] 文本模型已预热并常驻显存")
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_run, daemon=True).start()

_TYPE_NAME = {"line": "直线", "rect": "矩形", "circle": "圆",
              "polyline": "多段线", "arc": "圆弧", "text": "文本",
              "spline": "样条曲线", "ellipse": "椭圆"}
_TYPE_COLOR = {"rect": (0, 0, 255), "circle": (255, 0, 0),
              "line": (0, 180, 0), "polyline": (0, 140, 255),
              "text": (255, 0, 255), "spline": (0, 200, 120),
              "ellipse": (200, 120, 0), "arc": (0, 200, 200)}


def _job_store(spec: dict) -> str:
    with _JOBS_LOCK:
        if len(JOBS) >= MAX_JOBS:
            for k in sorted(JOBS, key=lambda k: JOBS[k]["ts"])[:len(JOBS) - MAX_JOBS + 1]:
                JOBS.pop(k, None)
        jid = uuid.uuid4().hex[:12]
        JOBS[jid] = {"spec": spec, "ts": time.time()}
    return jid


def _entity_desc(e: dict) -> str:
    """图元的一句话中文描述（勾选清单用）。"""
    t = str(e.get("type", ""))
    try:
        if t == "rect":
            return "矩形 %.1f×%.1f mm，左下角(%.1f, %.1f)" % (
                float(e.get("w", 0)), float(e.get("h", 0)),
                float(e.get("x", 0)), float(e.get("y", 0)))
        if t == "circle":
            c = e.get("center") or [0, 0]
            return "圆 半径 %.1f mm，中心(%.1f, %.1f)" % (
                float(e.get("radius", 0)), float(c[0]), float(c[1]))
        if t == "line":
            p1, p2 = e.get("p1") or [0, 0], e.get("p2") or [0, 0]
            L = math.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1]))
            return "直线 (%.1f, %.1f)→(%.1f, %.1f)，长 %.1f mm" % (
                float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), L)
        if t == "polyline":
            pts = e.get("points") or []
            return "多段线 %d 个顶点%s" % (
                len(pts), "（闭合）" if e.get("closed") else "")
        if t == "spline":
            pts = e.get("points") or []
            return "样条曲线 %d 个拟合点%s" % (
                len(pts), "（闭合）" if e.get("closed") else "")
        if t == "ellipse":
            c = e.get("center") or [0, 0]
            return "椭圆 %.1f×%.1f mm，中心(%.1f, %.1f)" % (
                float(e.get("rx", 0) or 0) * 2, float(e.get("ry", 0) or 0) * 2,
                float(c[0]), float(c[1]))
        if t == "arc":
            c = e.get("center") or [0, 0]
            try:
                span = (float(e.get("end", 0) or 0) - float(e.get("start", 0) or 0)) % 360.0
            except (TypeError, ValueError):
                span = 0.0
            return "圆弧 R%.1f mm，%d°，中心(%.1f, %.1f)" % (
                float(e.get("radius", 0) or 0), round(span),
                float(c[0]), float(c[1]))
        if t == "text":
            p = e.get("p") or [0, 0]
            val = str(e.get("value") or e.get("text") or "")
            if len(val) > 28:
                val = val[:28] + "…"
            return "文本「%s」，字高 %.1f mm，(%.1f, %.1f)" % (
                val, float(e.get("height", 5) or 5),
                float(p[0]), float(p[1]))
    except (TypeError, ValueError):
        pass
    return t


def _annotate_preview(img, spec: dict) -> str:
    """在原图上给每个识别图元画彩色轮廓 + 白底编号，返回 data URL。"""
    import cv2
    import numpy as np
    meta = spec.get("_meta") or {}
    frame = meta.get("frame")
    if not frame:
        ok, buf = cv2.imencode(".png", img)
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    cx, cy, scale = frame
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, e in enumerate(spec.get("entities") or []):
        col = _TYPE_COLOR.get(str(e.get("type")), (200, 200, 200))
        mm = line_extractor._entity_points_mm(e)
        if not mm:
            continue
        px = [(cx + x / scale, cy - y / scale) for (x, y) in mm]
        closed = str(e.get("type")) in ("rect", "circle", "text", "ellipse") \
            or e.get("closed")
        rng = range(len(px)) if closed else range(len(px) - 1)
        for k in rng:
            p1, p2 = px[k], px[(k + 1) % len(px)]
            cv2.line(vis, (int(p1[0]), int(p1[1])),
                     (int(p2[0]), int(p2[1])), col, 2)
        if str(e.get("type")) == "text":
            # 文本的编号放到框外左上角，别盖住字本身
            mx = min(p[0] for p in px) + 9
            my = max(6, min(p[1] for p in px) - 11)
        else:
            mx = sum(p[0] for p in px) / len(px)
            my = sum(p[1] for p in px) / len(px)
        cv2.circle(vis, (int(mx), int(my)), 11, (255, 255, 255), -1)
        cv2.circle(vis, (int(mx), int(my)), 11, (90, 90, 90), 1)
        cv2.putText(vis, str(i + 1), (int(mx) - 6, int(my) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", vis)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>J022 Ghoul</title>
<link rel="icon" type="image/png" href="/assets/joZZbot.png">
<style>
  :root{--bg:#140a0c;--card:#261115;--accent:#9b0000;--ok:#36c275;--err:#ff6b6b;--txt:#eee7e7;--mut:#b18f92}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,"Microsoft YaHei",sans-serif;background:var(--bg);color:var(--txt);padding:20px}
  .wrap{max-width:1440px;margin:0 auto}
  h1{font-size:22px;margin:0 0 4px}
  .sub{color:var(--mut);font-size:13px;margin-bottom:16px}
  .cols{display:grid;grid-template-columns:minmax(340px,1fr) minmax(380px,1.1fr);gap:16px;align-items:start}
  @media (max-width:1020px){.cols{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid #4a2024;border-radius:12px;padding:16px;margin-bottom:14px}
  label{display:block;font-size:13px;color:var(--mut);margin-bottom:6px}
  textarea{width:100%;background:#180b0d;border:1px solid #4a2024;border-radius:8px;color:var(--txt);padding:10px;font-size:14px;resize:vertical}
  #instr{height:84px}
  #chatin{height:64px;font-size:13px}
  .row{display:flex;gap:12px;align-items:center;margin-top:12px;flex-wrap:wrap}
  input[type=file]{color:var(--mut);font-size:13px}
  select{background:#180b0d;border:1px solid #4a2024;border-radius:8px;color:var(--txt);padding:8px 10px;font-size:13px}
  select:disabled{opacity:.55}
  button{background:var(--accent);color:#fff;border:none;border-radius:8px;padding:10px 18px;font-size:14px;cursor:pointer}
  button:disabled{opacity:.5;cursor:default}
  button.ghost{background:#3a161a;border:1px solid #4a2024;color:var(--txt);font-size:13px;padding:8px 14px}
  button.sm{padding:6px 12px;font-size:12px}
  #preview{max-width:150px;max-height:150px;border:1px solid #4a2024;border-radius:8px;display:none;margin-top:10px}
  #status{padding:12px;border-radius:8px;font-size:14px;display:none;white-space:pre-wrap;line-height:1.5}
  .s-ok{background:rgba(54,194,117,.12);border:1px solid var(--ok);color:var(--ok)}
  .s-err{background:rgba(255,107,107,.12);border:1px solid var(--err);color:var(--err)}
  .s-run{background:rgba(155,0,0,.12);border:1px solid var(--accent);color:#e08585}
  .hint{color:var(--mut);font-size:12px;margin-top:8px;line-height:1.5}
  a.link{color:var(--accent);text-decoration:none}
  .badge{display:inline-block;background:#381518;border:1px solid #4a2024;border-radius:999px;padding:2px 10px;font-size:12px;color:var(--mut);margin-left:8px}
  .modes{display:flex;gap:10px;flex-wrap:wrap;margin-top:10px}
  .mode{flex:1;min-width:170px;background:#180b0d;border:1px solid #4a2024;border-radius:8px;padding:10px 12px;cursor:pointer}
  .mode:hover{border-color:#6b2a2f}
  .mode.sel{border-color:var(--accent);background:rgba(155,0,0,.10)}
  .mode b{font-size:13px;display:block;margin-bottom:2px}
  .mode span{font-size:11px;color:var(--mut);line-height:1.4}
  #overlay,#selPreview{max-width:100%;border:1px solid #4a2024;border-radius:8px;margin-top:10px;display:none}
  .legend{font-size:11px;color:var(--mut);margin-top:6px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 10px;vertical-align:-1px}
  /* ---- 左栏对话 ---- */
  #chatlog{height:calc(100vh - 300px);min-height:340px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;
           padding:10px;background:#180b0d;border:1px solid #4a2024;border-radius:8px}
  .msg{max-width:88%;padding:8px 12px;border-radius:10px;font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-word}
  .msg.user{align-self:flex-end;background:rgba(155,0,0,.16);border:1px solid #6e2b2b}
  .msg.bot{align-self:flex-start;background:#2b1114;border:1px solid #4a2024}
  .msg.err{border-color:var(--err);color:var(--err)}
  .chatrow{display:flex;gap:8px;margin-top:10px;align-items:flex-end}
  .chatrow textarea{flex:1}
  .chatbtns{display:flex;flex-direction:column;gap:6px}
  /* ---- 线段选择 ---- */
  .sellist{max-height:230px;overflow-y:auto;margin-top:10px;background:#180b0d;border:1px solid #4a2024;border-radius:8px;padding:6px}
  .selitem{display:flex;gap:8px;align-items:center;padding:6px 8px;border-radius:6px;font-size:12.5px;color:var(--txt);cursor:pointer;margin:0}
  .selitem:hover{background:#221013}
  .selitem input{margin:0}
  .selitem .no{color:var(--mut);min-width:34px}
  .selitem .ty{display:inline-block;min-width:44px;color:var(--accent);font-weight:600}
  .checkrow{display:flex;gap:14px;align-items:center;margin-top:10px;flex-wrap:wrap;font-size:13px;color:var(--mut)}
  .checkrow label{display:flex;gap:6px;align-items:center;margin:0;color:var(--mut);cursor:pointer}
  /* ---- 尺寸栏（直线长度/圆半径/n边形边长） ---- */
  #dimRow input{background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;
                padding:4px 6px;font-size:13px}
  #dimRow input[type=checkbox]{width:auto}
  /* ---- 中心背景水印（JOZZBOT） ---- */
  #bgmark{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);
          width:min(60vmin,540px);opacity:.13;pointer-events:none;z-index:0;
          filter:drop-shadow(0 0 22px rgba(255,255,255,.18));user-select:none}
  .wrap{position:relative;z-index:1}
  /* ---- 右上角看板徽章 ---- */
  #corner{position:fixed;right:16px;top:16px;width:128px;height:128px;z-index:5;
          border-radius:50%;object-fit:cover;pointer-events:none;user-select:none;
          border:2px solid #4a2024;background:#1c0d0f;
          filter:drop-shadow(0 8px 20px rgba(0,0,0,.5));
          animation:cornerFloat 5s ease-in-out infinite}
  @keyframes cornerFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
  @media (max-width:1020px){#corner{width:84px;height:84px;right:10px;top:10px}}
  /* ---- 图层系统 ---- */
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  .chip{display:inline-flex;align-items:center;gap:6px;background:#180b0d;border:1px solid #4a2024;
        border-radius:999px;padding:5px 12px;font-size:13px;color:var(--txt);cursor:pointer}
  .chip:hover{border-color:#6b2a2f}
  .chip.sel{border-color:var(--accent);background:rgba(155,0,0,.14);font-weight:600}
  .chip .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
  .chip .cnt{color:var(--mut);font-size:11px}
  .chip.add{border-style:dashed;color:var(--mut)}
  /* ---- 内联文本输入框（MS 画图风格） ---- */
  #canvasWrap{position:relative;margin-top:10px}
  .textedit{position:absolute;z-index:8;background:rgba(24,11,13,.92);border:1px solid var(--accent);
            border-radius:6px;padding:4px 6px;display:none}
  .textedit input{background:#180b0d;border:none;color:#fff;font-family:system-ui,"Microsoft YaHei",sans-serif;
                  outline:none;min-width:80px}
  .combinebox{background:#180b0d;border:1px solid #4a2024;border-radius:8px;padding:10px;margin-top:10px}
  /* ---- 编辑器横向布局：工具栏在左，画布在右 ---- */
  #editorLayout{display:flex;flex-direction:column;gap:12px}
  #toolPanel{display:flex;flex-direction:column;gap:10px;width:100%}
  #toolPanel #toolbar{display:flex;flex-direction:row;flex-wrap:wrap;gap:6px;margin-top:0}
  #toolPanel #toolbar .mode{flex:1 1 132px;min-width:0;padding:8px 10px}
  #toolPanel #toolbar .mode b{display:block;margin-bottom:2px}
  #canvasArea{flex:0 0 auto;min-width:0;display:flex;flex-direction:column}
  #canvasArea #editor{width:100%;height:62vh;display:block}
  /* 全屏（页内最大化）态：编辑卡铺满视口，工具栏在上、画布在下占满宽度 */
  #editCard.editor-fs{position:fixed;inset:0;z-index:9999;background:var(--bg);
                      padding:14px;margin:0;display:flex!important;flex-direction:column;gap:10px}
  #editCard.editor-fs > label{display:none}
  #editCard.editor-fs #editorLayout{flex:1 1 auto;min-height:0;width:100%;display:flex;flex-direction:column;gap:12px}
  #editCard.editor-fs #toolPanel{flex:0 0 auto;max-height:none;overflow:visible;width:100%}
  #editCard.editor-fs #canvasArea{flex:1 1 auto;min-height:0;display:flex;flex-direction:column}
  #editCard.editor-fs #canvasWrap{flex:1 1 auto;min-height:0}
  #editCard.editor-fs #editor{width:100%;height:100%}
  #editCard.editor-fs .hint,#editCard.editor-fs #openRow2{display:none}
  #selInfo{font-size:12px;color:var(--mut);margin-top:6px;line-height:1.5}
  /* ---- 环境异常横幅（解释器缺 numpy/cv2 时才出现） ---- */
  .envbar{display:flex;gap:10px;align-items:flex-start;margin:0 0 12px;padding:10px 12px;
          background:#2b1418;border:1px solid #7a3038;border-left:4px solid #e0535f;border-radius:8px;
          font-size:13px;line-height:1.6;color:#ffb8bd}
  .envbar b{color:#ff8a95}
  .envbar code{background:#180b0d;border:1px solid #4a2024;border-radius:4px;padding:1px 5px;
               font-family:Consolas,monospace;color:#ffd9dc;word-break:break-all}
  .envbar .x{margin-left:auto;cursor:pointer;color:#c98a90;background:none;border:none;
             font-size:16px;line-height:1;padding:0 2px}
  .envbar .x:hover{color:#fff}
</style>
</head>
<body>
<img id="bgmark" src="/assets/joZZbot.png" alt="" aria-hidden="true">
<img id="corner" src="/assets/chibi.png" alt="看板" aria-hidden="true">
<div class="wrap">
  <div id="envBanner" class="envbar" style="display:none"></div>
  <h1>👻 J022 Ghoul <span class="badge" id="modelBadge">本地模型</span></h1>
  <div class="sub">纯本地运行 · 断网可用 ｜ 左：AI 对话 · 右：指令/草图 → DXF · 下：CAD 编辑工作台（工具在左，画布在右）</div>

  <div class="cols">
    <!-- ============ 左栏：AI 对话 ============ -->
    <div class="card">
      <label>💬 AI 对话（本地模型，离线答疑 · 可上传图片让 AI 看图/识文字）</label>
      <div id="chatlog"></div>
      <div id="chatImgBar" style="display:flex;gap:8px;align-items:center;margin-top:8px">
        <button id="chatImgBtn" class="ghost sm" type="button">📎 图片</button>
        <input id="chatImg" type="file" accept="image/*" style="display:none">
        <button id="ollamaFix" class="ghost sm" type="button" title="模型转圈/卡住时点这里：卸载显存里的旧模型并重新预热">🔧 模型卡住？</button>
        <div id="chatImgPrev" style="display:none;align-items:center;gap:6px;background:#180b0d;border:1px solid #4a2024;border-radius:8px;padding:4px 6px">
          <img id="chatImgThumb" alt="已选图片" style="max-width:42px;max-height:42px;border-radius:4px;display:block">
          <span id="chatImgName" style="font-size:12px;color:var(--mut);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
          <button id="chatImgDel" type="button" class="ghost sm" style="padding:2px 8px">✕</button>
        </div>
      </div>
      <div class="chatrow">
        <textarea id="chatin" placeholder="向 AI 提问…（Enter 发送，Shift+Enter 换行）&#10;例：DXF 里 LINE 的坐标是什么格式？/ 帮我写一个画法兰的绘图指令"></textarea>
        <div class="chatbtns">
          <button id="chatsend">发送</button>
          <button id="chatclear" class="ghost sm">清空</button>
        </div>
      </div>
      <div class="hint">💡 点「📎 图片」可上传一张图，再输入问题让 AI 看图/识别图中文字后回答（看图时会自动切换视觉模型）。对话与右侧绘图相互独立。<br>模型转圈不动/一直不回话时，点「🔧 模型卡住？」：会卸掉显存里的旧模型并重新预热（模型现在会常驻 30 分钟，不再反复冷加载）。</div>
    </div>

    <!-- ============ 右栏：CAD 绘制 ============ -->
    <div>
      <div class="card">
        <label>画图指令（自然语言，可留空仅用图片）</label>
        <textarea id="instr" placeholder="例如：画一个五角星，外接圆半径50；画一个长60宽30的矩形；或上传草图后留空&#10;💡 五角星/多边形/菱形/箭头/心形等由图形库精确生成；圆+矩形等简单组合也支持（如：画一个圆和一个正方形）"></textarea>
        <div class="row">
          <div>
            <label style="margin:0">上传参考图 / 草图（可选，启用视觉识别）</label>
            <input type="file" id="img" accept="image/*">
          </div>
          <div>
            <label style="margin:0">输出格式</label>
            <select id="fmtSel" title="DWG 由 ODA File Converter 离线转换（R2000~R2018，缺 ODA 时回退 LibreDWG R2000）">
              <option value="dxf">DXF（默认）</option>
              <option value="dwg">DWG</option>
              <option value="both">DXF + DWG</option>
            </select>
          </div>
          <button id="gen">生成 DXF</button>
        </div>
        <img id="preview" alt="预览">
        <div class="checkrow">
          <label><input type="checkbox" id="selStep" checked> 识别后先勾选保留线段（仅图片 + 自动/线条模式）</label>
          <label><input type="checkbox" id="ocrText" checked> 🔤 同时识别图中文字，并把文字作为文本图元写进 CAD</label>
          <label><input type="checkbox" id="multiChk" checked> 🔁 多次识别（多遍 OpenCV 融合，曲线图更全；取消则只跑 1 遍，更快）</label>
          <label><input type="checkbox" id="bwChk" checked> 🖨️ 先转纯黑白图（对比度拉满的白底黑线，抗阴影/彩线/灰阶淡线）</label>
          <label style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">🔷 Homotope 辅助
            <select id="htSel" style="padding:2px 4px">
              <option value="auto" selected>自动（CV 失败/结果弱时接手）</option>
              <option value="on">只用 Homotope（拓扑分割+中线描摹）</option>
              <option value="off">关闭</option>
            </select>
            <span class="hint" id="htHint" style="margin:0"></span>
          </label>
        </div>
        <div class="hint" id="ocrHint" style="margin-top:4px">勾选文字识别后：用 OpenCV 定位每一行文字的位置，再让本地视觉模型读内容（首次约 10–40 秒，之后模型常驻会快很多）。识别出的文字可直接在下方编辑台选中、移动、删除，或逐条取消勾选。</div>
        <div class="modes" id="modes">
          <div class="mode sel" data-mode="auto"><b>⚡ 自动（推荐）</b><span>先走经典视觉精确提线（秒级），失败自动回退 AI。支持按颜色过滤，如"去掉红色标注只留黑色线条"</span></div>
          <div class="mode" data-mode="lines"><b>📐 线条精确</b><span>强制经典视觉（OpenCV）：直线/圆/矩形像素级提取，支持按颜色过滤</span></div>
          <div class="mode" data-mode="vlm"><b>🧠 AI 理解</b><span>只用本地 7B 视觉大模型，适合语义化图形；颜色过滤由服务器按原图颜色强制兜底校验（投影视图如俯视/左视仍建议手动）</span></div>
        </div>
        <div class="hint">支持 PNG/JPG。上传草图后优先用经典视觉识别——线条坐标是算出来的，不是猜的。<br>🔤 勾选「识别图中文字」后，图上标注/尺寸文字会被读出来，作为可编辑的 TEXT 一起写进 DXF 和下方编辑台。<br>🧩 纯文字指令已内置 <b>54 个常见物体</b>（葫芦 / 杯子 / 雨伞 / 台灯 / 猫 / 鱼 / 汽车 / 齿轮 / 房子 …），直接写物体名就会按真实形状精确出图，还能带尺寸：<b>画一个葫芦，高150宽90</b>。<br>💡 颜色过滤指令示例：<b>去掉红色标注，只保留黑色线条</b> / <b>删掉蓝色中心线</b> / <b>只保留红色部分</b>。</div>
      </div>

      <div class="card" id="layerCard" style="display:none">
        <label>🗂️ 图层管理 — 每次生成的 DXF 可分配到不同图层，最后合并成一个文件</label>
        <div class="hint" style="margin:0 0 8px">先选好「当前图层」，再点生成 / 绘制；不同图层用不同颜色区分。全部画完后点「合成 DXF」把它们合并导出。</div>
        <div id="layerChips" class="chips"></div>
        <div class="row" style="margin-top:12px">
          <button id="combineBtn">🧩 合成 DXF</button>
          <button id="addLayer" class="ghost sm">＋ 新建图层</button>
          <button id="resetLayers" class="ghost sm">重置图层</button>
          <button id="importDxf" class="ghost sm">📥 导入 DXF</button>
          <input id="importDxfFile" type="file" accept=".dxf" style="display:none">
          <span id="layerHint" class="hint" style="margin:0 0 0 auto"></span>
        </div>
        <div id="importPanel" class="combinebox" style="display:none">
          <div class="hint" id="importInfo" style="margin:0 0 8px">选择 DXF 文件后，可将其图元导入到指定图层</div>
          <div class="row" style="gap:10px;align-items:center;flex-wrap:wrap">
            <span style="font-size:13px;color:var(--mut)">导入到图层</span>
            <input id="importLayer" type="number" min="1" value="1" style="width:64px;background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:4px 6px">
            <button id="importGo" class="sm">导入</button>
            <button id="importCancel" class="ghost sm">取消</button>
          </div>
        </div>
        <div id="combinePanel" class="combinebox" style="display:none">
          <div class="hint" style="margin:0 0 8px">勾选要合并进最终 DXF 的图层（默认全部）：</div>
          <div id="combineLayers" class="chips"></div>
          <div class="checkrow" style="margin-top:10px">
            <label><input type="checkbox" id="combineFlatten" checked> 合并为单层（推荐：1,2,3…全部归到 0 层）</label>
            <label><input type="checkbox" id="combineKeep"> 保留分层（每层成为 DXF 里的独立图层）</label>
            <label><input type="checkbox" id="combineDwg"> 同时输出 DWG（+1–3 秒）</label>
          </div>
          <div class="row" style="margin-top:10px">
            <span style="font-size:13px;color:var(--mut)">文件名</span>
            <input id="combineName" type="text" value="合并图" style="background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:4px 8px;font-size:13px;width:160px">
            <button id="combineGo" class="sm">生成并打开</button>
            <button id="combineCancel" class="ghost sm">取消</button>
          </div>
        </div>
      </div>

      <div class="card" id="selCard" style="display:none">
        <label id="selTitle">识别结果——勾选要保留的线段</label>
        <img id="selPreview" alt="编号预览图">
        <div id="selList" class="sellist"></div>
        <div class="row">
          <button id="selBuild">✅ 生成 DXF（保留勾选项）</button>
          <button id="selAll" class="ghost">全选</button>
          <button id="selNone" class="ghost">全不选</button>
          <button id="selSkip" class="ghost">跳过勾选，全部生成</button>
          <button id="selCancel" class="ghost">取消</button>
        </div>
        <div class="hint">编号与预览图上的白色圆圈标号一一对应；不想要的杂线/噪点取消勾选即可。</div>
      </div>

      <div class="card">
        <div id="status"></div>
        <img id="overlay" alt="识别叠加图">
        <div class="legend" id="overlayLegend" style="display:none">
          识别叠加图（机器看到的结果）：
          <i style="background:#ff0000"></i>矩形
          <i style="background:#0000ff"></i>圆
          <i style="background:#00cccc"></i>圆弧
          <i style="background:#00b400"></i>直线
          <i style="background:#ff8c00"></i>多段线
          <i style="background:#ff00ff"></i>文本
        </div>
        <div id="openRow" style="display:none;margin-top:10px">
          <a class="link" id="openLink" target="_blank">🔗 用 LibreCAD 打开此 DXF</a>
          <a class="link" id="openDwg" target="_blank" style="display:none;margin-left:16px">📂 打开 DWG 所在文件夹</a>
        </div>
      </div>

      <!-- 编辑器已移至底部「CAD 编辑工作台」区块 -->
    </div>
  </div>

  <!-- ============ 底部：CAD 编辑工作台（工具在左，画布在右） ============ -->
  <div class="card" id="editCard" style="display:none">
    <label>🖊️ 编辑视图 — 左侧选工具，右侧画布点击选中 / 拖动移动 / Delete 删除；工具可额外绘制线条</label>
    <div id="editorLayout">
      <div id="toolPanel">
        <div id="toolbar" class="modes">
          <div class="mode sel" data-tool="select"><b>↖ 选择/移动</b><span>点线选中，拖动移动；Shift 点击加选/取消；空白处拖框多选</span></div>
          <div class="mode" data-tool="rotate"><b>⟳ 旋转</b><span>拖动自由旋转；或填「与水平夹角」点应用（0°=水平，逆时针为正）</span></div>
          <div class="mode" data-tool="line"><b>／ 直线</b><span>按下起点松开终点；尺寸栏定长</span></div>
          <div class="mode" data-tool="pen"><b>✎ 画笔</b><span>按住拖动随手画（多段线）</span></div>
          <div class="mode" data-tool="curve"><b>〜 曲线</b><span>点起点→点终点→移动调曲率再点确认（Esc 取消）</span></div>
          <div class="mode" data-tool="circle"><b>◯ 圆</b><span>按下圆心松开定半径；尺寸栏定半径</span></div>
          <div class="mode" data-tool="poly"><b>⬡ n边形</b><span>点画布放置（边数/边长见尺寸栏）</span></div>
          <div class="mode" data-tool="erase"><b>🧽 橡皮擦</b><span>按住拖动擦除触碰线条</span></div>
          <div class="mode" data-tool="text"><b>🅣 文本</b><span>尺寸栏输入文字，点画布放置</span></div>
          <div class="mode" data-tool="pan"><b>✋ 平移</b><span>拖动平移视图</span></div>
        </div>
        <div class="row" id="dimRow" style="display:none;margin-top:8px;font-size:13px;color:var(--mut);align-items:center;gap:10px;flex-wrap:wrap">
          <b style="color:var(--txt)">尺寸：</b>
          <span id="dimLine" style="display:none">长度 <input id="dimLen" type="number" min="0.1" step="any" style="width:84px"> mm（留空自由绘制）</span>
          <span id="dimCircle" style="display:none">半径 <input id="dimRad" type="number" min="0.1" step="any" style="width:84px"> mm（留空自由拖动）</span>
          <span id="dimText" style="display:none">字号 <input id="dimTextH" type="number" min="0.5" step="any" value="5" style="width:64px"> mm　文字 <input id="dimTextStr" type="text" placeholder="在此输入，点画布放置" style="width:150px"></span>
          <span id="dimPoly" style="display:none">边数 <input id="dimN" type="number" min="3" max="24" value="6" style="width:52px">
            <label style="display:inline;margin:0 0 0 4px"><input type="checkbox" id="dimRegular" checked> 正多边形</label>
            <span id="dimRegBox">边长 <input id="dimSide" type="number" min="0.1" step="any" style="width:84px"> mm</span>
            <span id="dimIrrBox" style="display:none">各边长(逗号) <input id="dimSides" type="text" placeholder="30,40,50,25" style="width:110px"></span>
          </span>
          <span id="dimRotate" style="display:none">与水平夹角 <input id="rotAngle" type="number" step="any" value="90" style="width:70px"> °
            <button id="applyRot" class="ghost sm" style="margin-left:4px">应用旋转</button>
            <span style="color:var(--mut)">（0°=水平，逆时针为正；填 60° 即与水平成 60°；绕选中图元质心）</span>
          </span>
        </div>
        <div id="editorActions" style="display:flex;flex-direction:row;flex-wrap:wrap;gap:8px;margin-top:12px">
          <button id="fsBtn" class="ghost sm">⛶ 全屏</button>
          <button id="undoBtn" class="ghost sm" disabled>↶ 撤回</button>
          <button id="redoBtn" class="ghost sm" disabled>↷ 恢复</button>
          <button id="delSel" class="ghost sm">删除选中</button>
          <button id="fitView" class="ghost sm">适配视图</button>
          <button id="clearAll" class="ghost sm">清空</button>
          <button id="exportDxf" class="ghost sm">导出 DXF</button>
          <button id="exportDwg" class="ghost sm">导出 DWG</button>
          <span style="display:inline-flex;gap:6px;align-items:center;margin-left:6px"><b style="color:var(--txt);font-size:12px">线型:</b><button id="dashSolid" class="ghost sm">━ 实线</button><button id="dashDashed" class="ghost sm">⠿ 虚线</button><span style="font-size:12px;color:var(--mut)">段长</span><input id="dashLenIn" type="number" min="0.5" step="0.5" value="5" style="width:52px;background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:3px 5px"><span style="font-size:12px;color:var(--mut)">间隔</span><input id="dashGapIn" type="number" min="0.5" step="0.5" value="3" style="width:52px;background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:3px 5px"></span>
          <span style="display:inline-flex;gap:6px;align-items:center;margin-left:6px"><b style="color:var(--txt);font-size:12px">箭头:</b><button id="arrNone" class="ghost sm">无</button><button id="arrStart" class="ghost sm">起点</button><button id="arrEnd" class="ghost sm">终点</button><button id="arrBoth" class="ghost sm">两端</button><span style="font-size:12px;color:var(--mut)">大小</span><input id="arrSizeIn" type="number" min="1" step="0.5" value="4" style="width:52px;background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:3px 5px">mm</span>
          <span style="display:inline-flex;gap:6px;align-items:center;margin-left:6px"><b style="color:var(--txt);font-size:12px">移至图层:</b><select id="moveLayerSel" style="background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:3px 5px;font-size:12px"></select><button id="moveLayerBtn" class="ghost sm">移动</button></span>
          <span style="display:inline-flex;gap:6px;align-items:center;margin-left:6px"><b style="color:var(--txt);font-size:12px">🧲 吸附:</b><label style="display:inline;margin:0"><input type="checkbox" id="snapChk" checked></label><span style="font-size:12px;color:var(--mut)">距离</span><input id="snapPxIn" type="number" min="0" max="60" step="1" value="8" style="width:52px;background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:3px 5px"><span style="font-size:12px;color:var(--mut)">px</span><span style="font-size:12px;color:var(--mut)">角度</span><input id="snapDegIn" type="number" min="0" max="45" step="1" value="6" style="width:52px;background:#180b0d;border:1px solid #4a2024;color:var(--txt);border-radius:6px;padding:3px 5px"><span style="font-size:12px;color:var(--mut)">°</span></span>
          <label style="font-size:12px;color:var(--mut);display:flex;align-items:center;gap:6px;margin-left:6px">🆕 新线 <input id="newDash" type="checkbox">虚线</label>
          <label style="font-size:12px;color:var(--mut);display:flex;align-items:center;gap:6px">笔刷
            <input id="eraserSize" type="range" min="4" max="48" value="14" style="flex:1">
            <span id="eraserSizeVal">14</span>px
          </label>
        </div>
      </div>
      <div id="canvasArea">
        <div id="canvasWrap" style="position:relative">
          <canvas id="editor" width="860" height="520"
                  style="width:100%;height:auto;border:1px solid #4a2024;border-radius:8px;background:#180b0d;cursor:crosshair;touch-action:none;display:block"></canvas>
        </div>
        <div class="hint">滚轮缩放；选择工具下点线条选中、拖动整体移动；Shift 点击可加选/取消，空白处拖框可框选多根线条；Delete 删除选中；绘制工具下在画布上成图，直线/圆/n边形可在尺寸栏自定义大小。导出后自动用 LibreCAD 打开。</div>
        <div id="selInfo">未选中（点线条选中，Shift 加选，空白处拖框多选）</div>
        <div id="openRow2" style="display:none;margin-top:10px">
          <a class="link" id="openLink2" target="_blank">🔗 用 LibreCAD 打开此 DXF</a>
          <a class="link" id="openDwg2" target="_blank" style="display:none;margin-left:16px">📂 打开 DWG 所在文件夹</a>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
const instr=$('#instr'), img=$('#img'), gen=$('#gen'), status=$('#status'),
      preview=$('#preview'), openRow=$('#openRow'), openLink=$('#openLink'),
      overlay=$('#overlay'), overlayLegend=$('#overlayLegend'),
      chatlog=$('#chatlog'), chatin=$('#chatin'), chatsend=$('#chatsend'),
      chatclear=$('#chatclear'), selCard=$('#selCard'), selTitle=$('#selTitle'),
      selPreview=$('#selPreview'), selList=$('#selList'),
      selBuild=$('#selBuild'), selAll=$('#selAll'), selNone=$('#selNone'),
      selSkip=$('#selSkip'), selCancel=$('#selCancel'), selStep=$('#selStep'),
      ocrChk=$('#ocrText'),
      multiChk=$('#multiChk'),
      bwChk=$('#bwChk'),
      htSel=$('#htSel'), htHint=$('#htHint'),
      fmtSel=$('#fmtSel'), openDwg=$('#openDwg'), openDwg2=$('#openDwg2');
let mode = 'auto';
let job = null;   // {job_id, entities, total}

/* ================= 输出格式（DXF / DWG） ================= */
// DWG 靠外部转换器（包内 LibreDWG）。启动时问一下后端有没有，没有就把 DWG 选项
// 灰掉并说明原因 —— 总比让用户点了以后拿到一堆"DWG 输出失败"好。
async function probeDwg(){
  try{
    const r = await fetch('/api/dwg_status', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json();
    if(!d.ok){
      for(const opt of fmtSel.options){
        if(opt.value !== 'dxf'){ opt.disabled = true; opt.text = opt.text.replace('DWG','DWG（不可用）'); }
      }
      fmtSel.value = 'dxf';
      fmtSel.title = d.note || 'DWG 转换器不可用';
    }else{
      fmtSel.title = d.note || 'DWG 输出可用';
    }
  }catch(e){ /* 探测失败不影响 DXF */ }
}
// 统一处理"这次导出有没有 DWG"，成功就把第二个链接亮出来
function showDwgResult(data, linkEl, rowEl){
  if(data && data.dwg){
    if(linkEl){ linkEl.href = '/open?file=' + encodeURIComponent(data.dwg); linkEl.style.display='inline'; }
    return '｜DWG：' + data.dwg;
  }
  if(linkEl) linkEl.style.display='none';
  return (data && data.dwg_error) ? ('｜DWG 未生成：' + data.dwg_error) : '';
}
// 按钮文案跟着输出格式走：不然选 DWG 却点"生成 DXF"，用户会以为没生效
function syncGenLabel(){
  gen.textContent = fmtSel.value === 'dwg' ? '生成 DWG'
                  : (fmtSel.value === 'both' ? '生成 DXF+DWG' : '生成 DXF');
}
fmtSel.onchange = syncGenLabel;
syncGenLabel();
probeDwg();

/* ================= 图层配色（编辑器与 DXF 图层颜色对应） ================= */
function layerColorCss(layer){
  let n = parseInt(String(layer), 10);
  if(isNaN(n) || n < 1) return '#e6e9ef';   // 0 层 / 非法层名 → 默认浅色
  const PAL = ['#ff5b5b','#ffe14d','#5bff7a','#5bd6ff','#5b7bff','#ff6bff',
               '#ff9e3d','#ff8a8a','#3dffc4','#c46bff','#8aff5b','#ff5bd6',
               '#9bff5b','#5bffea','#ff5b8a','#7affff'];
  return PAL[(n - 1) % PAL.length];
}

/* ================= 模式选择 ================= */
document.querySelectorAll('#modes .mode').forEach(el=>{
  el.onclick = ()=>{
    document.querySelectorAll('.mode').forEach(x=>x.classList.remove('sel'));
    el.classList.add('sel');
    mode = el.dataset.mode;
  };
});

img.onchange = () => {
  const f = img.files[0];
  if(!f){ preview.style.display='none'; return; }
  const r = new FileReader();
  r.onload = e => { preview.src=e.target.result; preview.style.display='block'; };
  r.readAsDataURL(f);
};

function setStatus(msg, cls){
  status.textContent = msg;
  status.className = cls ? ('s-'+cls) : '';
  status.style.display = 'block';
}

async function readImageB64(){
  const f = img.files[0];
  if(!f) return '';
  return await new Promise(res=>{const r=new FileReader();r.onload=e=>res(r.result);r.readAsDataURL(f);});
}

/* ================= 生成主流程 ================= */
gen.onclick = async () => {
  const b64 = await readImageB64();
  if(!instr.value.trim() && !b64){
    setStatus('❌ 请输入画图指令，或上传一张参考图片', 'err');
    return;
  }
  if(b64 && mode!=='vlm' && selStep.checked){
    await doExtract(instr.value, b64);
  }else{
    await doGenerate(instr.value, b64);
  }
};

async function doGenerate(instruction, b64){
  gen.disabled = true;
  selCard.style.display='none';
  openRow.style.display='none';
  overlay.style.display='none';
  overlayLegend.style.display='none';
  const wantOcr = !!(ocrChk && ocrChk.checked && b64);
  setStatus('⏳ 正在生成 DXF…（经典视觉约 1–3 秒' +
            (wantOcr ? '；文字识别首次需加载视觉模型，约 10–40 秒' : '；AI 模式约 10–40 秒') + '）', 'run');
  try{
    const resp = await fetch('/api/generate', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({instruction: instruction, image_base64: b64, mode: mode,
                            ocr_text: wantOcr, fmt: fmtSel.value,
                            multipass: !!(multiChk && multiChk.checked),
                            homotope: htSel ? htSel.value : 'auto',
                            bw_first: !!(bwChk && bwChk.checked)})
    });
    const data = await resp.json();
    if(data.ok){
      const how = data.method === 'cv-lines' ? '🔬 经典视觉(OpenCV)' :
                  (data.method === 'homotope' ? '🔷 Homotope 引擎(中线矢量化)' :
                   (data.method === 'shapes' ? '📐 图形库(确定性几何)' :
                   (data.method === 'concepts' ? '🧩 物体概念库(确定性几何)' : '🧠 大模型')));
      const dwgNote = showDwgResult(data, openDwg, openRow);
      setStatus('✅ 已生成：' + data.dxf + '（' + data.entities + ' 个图元' +
                (data.text_count ? '，含 ' + data.text_count + ' 处文字' : '') + '）' + dwgNote +
                '\\n识别方式：' + how +
                (data.counts_text ? '｜' + data.counts_text : '') +
                (data.message ? '\\n' + data.message : ''), 'ok');
      if(data.debug_image){
        overlay.src = data.debug_image;
        overlay.style.display = 'block';
        overlayLegend.style.display = 'block';
      }
      openLink.href = '/open?file=' + encodeURIComponent(data.dxf);
      openRow.style.display='block';
      if(data.entities_list && data.entities_list.length){
        showInEditor(data.entities_list);
      }
    } else {
      setStatus('❌ 生成失败：' + (data.error||'未知错误'), 'err');
    }
  }catch(e){
    setStatus('❌ 无法连接本地服务：' + e.message + '\\n请确认启动器已运行', 'err');
  }finally{
    gen.disabled = false;
  }
}

/* ================= 线段识别 + 勾选 ================= */
async function doExtract(instruction, b64){
  gen.disabled = true;
  selCard.style.display='none';
  openRow.style.display='none';
  overlay.style.display='none';
  overlayLegend.style.display='none';
  const wantOcr = !!(ocrChk && ocrChk.checked);
  setStatus('⏳ 正在识别线段' + (wantOcr ? '与图中文字（首次加载视觉模型约 10–40 秒）' : '') +
            '…（经典视觉约 1–3 秒）', 'run');
  try{
    const resp = await fetch('/api/extract', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({instruction: instruction, image_base64: b64, ocr_text: wantOcr,
                            multipass: !!(multiChk && multiChk.checked),
                            homotope: htSel ? htSel.value : 'auto',
                            bw_first: !!(bwChk && bwChk.checked)})
    });
    const data = await resp.json();
    if(!data.ok){
      setStatus('❌ 识别失败：' + (data.error||'未知错误') +
                (data.hint ? '\\n💡 ' + data.hint : ''), 'err');
      return;
    }
    job = {job_id: data.job_id, total: data.total};
    selTitle.textContent = '识别到 ' + data.total + ' 个图元' +
      (data.text_count ? '（其中文字 ' + data.text_count + ' 处）' : '') +
      (data.color_filter ? '（颜色过滤：' + data.color_filter + '）' : '') +
      (data.passes ? '（OpenCV ' + data.passes + ' 遍融合）' : '') +
      '——勾选要保留的图元：';
    selPreview.src = data.preview;
    selPreview.style.display = 'block';
    selList.innerHTML = '';
    data.entities.forEach(en=>{
      const lab = document.createElement('label');
      lab.className = 'selitem';
      const cb = document.createElement('input');
      cb.type = 'checkbox'; cb.checked = true; cb.dataset.id = en.id;
      const no = document.createElement('span'); no.className='no'; no.textContent = '#'+(en.id+1);
      const ty = document.createElement('span'); ty.className='ty'; ty.textContent = en.type_name;
      const de = document.createElement('span'); de.textContent = en.desc;
      lab.append(cb, no, ty, de);
      selList.appendChild(lab);
    });
    selCard.style.display = 'block';
    setStatus('🔍 已识别 ' + data.total + ' 个图元' +
              (data.ocr_note ? '\\n' + data.ocr_note : '') +
              '\\n请勾选要保留的图元后点"生成 DXF"', 'ok');
  }catch(e){
    setStatus('❌ 无法连接本地服务：' + e.message, 'err');
  }finally{
    gen.disabled = false;
  }
}

function collectIds(){
  const ids = [];
  selList.querySelectorAll('input[type=checkbox]').forEach(cb=>{
    if(cb.checked) ids.push(parseInt(cb.dataset.id, 10));
  });
  return ids;
}

async function doBuild(keepIds){
  if(!job){ setStatus('❌ 没有可用的识别结果，请重新上传图片识别', 'err'); return; }
  if(!keepIds.length){ setStatus('❌ 请至少勾选一个图元（或点"跳过勾选，全部生成"）', 'err'); return; }
  selBuild.disabled = true; selSkip.disabled = true;
  setStatus('⏳ 正在按勾选生成 DXF…', 'run');
  try{
    const resp = await fetch('/api/build', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({job_id: job.job_id, keep_ids: keepIds, fmt: fmtSel.value})
    });
    const data = await resp.json();
    if(data.ok){
      const dwgNote = showDwgResult(data, openDwg, openRow);
      setStatus('✅ 已生成：' + data.dxf + '（' + data.entities + ' 个图元）' + dwgNote + '\\n' +
                (data.message || ''), 'ok');
      openLink.href = '/open?file=' + encodeURIComponent(data.dxf);
      openRow.style.display='block';
      if(data.entities_list && data.entities_list.length){
        showInEditor(data.entities_list);
      }
    } else {
      setStatus('❌ 生成失败：' + (data.error||'未知错误'), 'err');
    }
  }catch(e){
    setStatus('❌ 无法连接本地服务：' + e.message, 'err');
  }finally{
    selBuild.disabled = false; selSkip.disabled = false;
  }
}

selBuild.onclick = () => doBuild(collectIds());
selSkip.onclick  = () => {
  selList.querySelectorAll('input[type=checkbox]').forEach(cb=>cb.checked=true);
  doBuild(collectIds());
};
selAll.onclick   = () => selList.querySelectorAll('input[type=checkbox]').forEach(cb=>cb.checked=true);
selNone.onclick  = () => selList.querySelectorAll('input[type=checkbox]').forEach(cb=>cb.checked=false);
selCancel.onclick = () => { selCard.style.display='none'; setStatus('', ''); job = null; };

/* ================= 左栏 AI 对话 ================= */
let history = [];

function addMsg(role, text, imgDataUrl){
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if(imgDataUrl){
    const im = document.createElement('img');
    im.src = imgDataUrl;
    im.style.cssText = 'display:block;max-width:200px;max-height:160px;border-radius:8px;margin-bottom:6px;border:1px solid #4a2024';
    div.appendChild(im);
  }
  const span = document.createElement('span');
  span.textContent = text;
  div.appendChild(span);
  chatlog.appendChild(div);
  chatlog.scrollTop = chatlog.scrollHeight;
  return div;
}

// ---- 对话图片上传（看图问答 / OCR）----
let chatImgData = null;   // data URL（含前缀），随本次请求发出；不进历史
const chatImgBtn = $('#chatImgBtn'), chatImg = $('#chatImg'),
      chatImgPrev = $('#chatImgPrev'), chatImgThumb = $('#chatImgThumb'),
      chatImgName = $('#chatImgName'), chatImgDel = $('#chatImgDel');
chatImgBtn.onclick = () => chatImg.click();
chatImg.onchange = () => {
  const f = chatImg.files && chatImg.files[0];
  if(!f){ return; }
  if(f.type.indexOf('image/') !== 0){ alert('请选择图片文件（PNG/JPG 等）'); chatImg.value=''; return; }
  const r = new FileReader();
  r.onload = ev => {
    chatImgData = ev.target.result;          // data:image/...;base64,...
    chatImgThumb.src = chatImgData;
    chatImgName.textContent = f.name;
    chatImgPrev.style.display = 'inline-flex';
  };
  r.readAsDataURL(f);
};
chatImgDel.onclick = () => {
  chatImgData = null;
  chatImg.value = '';
  chatImgPrev.style.display = 'none';
  chatImgThumb.src = '';
};

async function sendChat(){
  const text = chatin.value.trim();
  if((!text && !chatImgData) || chatsend.disabled) return;
  const sentImg = chatImgData;
  chatin.value = '';
  addMsg('user', text || '（请看图）', sentImg || undefined);
  history.push({role:'user', content:(text || '（请看这张图）')});
  if(history.length > 16) history = history.slice(-16);
  const wait = addMsg('bot', '…思考中（本地模型，稍等几秒）');
  chatsend.disabled = true;
  try{
    const resp = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages: history, image_base64: sentImg || ''})
    });
    const data = await resp.json();
    if(data.ok){
      wait.textContent = data.reply || '（空回复）';
      history.push({role:'assistant', content:data.reply});
      if(history.length > 16) history = history.slice(-16);
      chatImgDel.onclick();   // 发送成功后清空已选图片，避免后续纯文字消息误带旧图
    }else{
      wait.textContent = '❌ ' + (data.error || '未知错误');
      wait.className = 'msg bot err';
    }
  }catch(e){
    wait.textContent = '❌ 无法连接本地服务：' + e.message;
    wait.className = 'msg bot err';
  }finally{
    chatsend.disabled = false;
    chatlog.scrollTop = chatlog.scrollHeight;
  }
}

chatsend.onclick = sendChat;
chatclear.onclick = () => { history = []; chatlog.innerHTML = ''; chatImgDel.onclick(); addMsg('bot','对话已清空。有什么可以帮你？'); };
chatin.addEventListener('keydown', e=>{
  if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendChat(); }
});

// ---- 模型卡住时的一键自愈：卸载占显存的模型 + 重新预热 ----
const ollamaFix = $('#ollamaFix');
if(ollamaFix) ollamaFix.onclick = async () => {
  ollamaFix.disabled = true;
  const wait = addMsg('bot', '🔧 正在清理本地模型（卸载显存里的旧模型 → 重新预热）…');
  try{
    const resp = await fetch('/api/ollama/repair', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({warm: true})
    });
    const d = await resp.json();
    if(d.ok){
      wait.textContent = '✅ ' + (d.note || '已清理') + '\\n现在再试一次刚才的操作就行。';
    }else{
      wait.textContent = '❌ ' + (d.error || '修复失败') + (d.note ? '\\n' + d.note : '');
      wait.className = 'msg bot err';
    }
  }catch(e){
    wait.textContent = '❌ 无法连接本地服务：' + e.message;
    wait.className = 'msg bot err';
  }finally{
    ollamaFix.disabled = false;
    chatlog.scrollTop = chatlog.scrollHeight;
  }
};

/* ================= 交互式编辑器（选中/移动/删除 + 画笔/直线/曲线/圆） ================= */
const Editor = (() => {
  const canvas = $('#editor');
  const ctx = canvas.getContext('2d');
  const editCard = $('#editCard'), openLink2 = $('#openLink2'), openRow2 = $('#openRow2'),
        canvasArea = $('#canvasArea');
  let entities = [], selSet = new Set(), tool = 'select';
  let cursor = null, eraserSize = 14;
  let view = {cx:0, cy:0, scale:10}, drag = null, draw = null;
  let eraserBase = null, eraserTrail = null;   // 橡皮擦当前笔迹（一次拖动）
  let activeLayer = 1;                          // 当前图层：新生成的图元归入此层
  let newDash = false;                          // 之后绘制的线条是否默认虚线
  let newDashLen = 5, newDashGap = 3;           // 新虚线的段长/间隔（mm）
  let newArrow = 0, newArrowSize = 4;           // 新线箭头：0无 1起点 2终点 3两端；大小 mm
  let onChangeCb = null;                        // 图元变化时通知外层（刷新图层计数）
  // 文本工具：文字在尺寸栏输入，点画布放置（见 placeText）
  const HIT = 6;

  /* ---- 多选辅助 ---- */
  function updateSelInfo(){
    const n=selSet.size, el=$('#selInfo');
    if(!el) return;
    if(!n){
      el.textContent = '未选中（点线条选中，Shift 加选，空白处拖框多选）';
      return;
    }
    let msg='已选中 '+n+' 个图元';
    if(n===1){
      const e=[...selSet][0], a=entAngleDeg(e);
      if(e.type==='line'){
        const L=Math.hypot(e.p2[0]-e.p1[0], e.p2[1]-e.p1[1]);
        msg+='（直线，长 '+L.toFixed(2)+' mm）· 与水平夹角 '+a.toFixed(1)+'°'+angHint(a);
      }else if(e.type==='circle'){
        msg+='（圆，R '+Number(e.radius).toFixed(2)+' mm）';
      }else if(e.type==='arc'){
        msg+='（圆弧，R '+Number(e.radius).toFixed(2)+' mm，'+Number(e.start).toFixed(0)+'°→'+Number(e.end).toFixed(0)+'°）';
      }else if(a!==null){
        const nm={rect:'矩形',polyline:'多段线',spline:'样条',ellipse:'椭圆'}[e.type]||e.type;
        msg+='（'+nm+'）· 与水平夹角 '+a.toFixed(1)+'°'+angHint(a);
      }else if(e.type==='text'){
        msg+='（文本：「'+(e.value||'')+'」）';
      }
    }else{
      const a=selAngleDeg(selSet);
      if(a!==null) msg+=' · 代表方向与水平夹角 '+a.toFixed(1)+'°'+angHint(a);
    }
    el.textContent = msg + '（Shift 点击加选/取消；空白处拖框多选；Delete 删除）';
  }
  function entityScreenBBox(ent){
    if(ent.type==='text'){const b=textBoxScreen(ent);return [b.x0,b.y0,b.x1,b.y1];}
    if(ent.type==='circle'){const c=w2s(ent.center),r=ent.radius*view.scale;return [c[0]-r,c[1]-r,c[0]+r,c[1]+r];}
    const pts=entityPoints(ent).map(w2s);
    let xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
    return [Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];
  }
  function rectsOverlap(a,b){return !(a[2]<b[0]||a[0]>b[2]||a[3]<b[1]||a[1]>b[3]);}
  function applyMarquee(s,c,add){
    const x0=Math.min(s[0],c[0]), y0=Math.min(s[1],c[1]),
          x1=Math.max(s[0],c[0]), y1=Math.max(s[1],c[1]);
    if(Math.abs(x1-x0)<4 && Math.abs(y1-y0)<4){ updateSelInfo(); render(); return; }
    const box=[x0,y0,x1,y1];
    for(const ent of entities){ if(rectsOverlap(entityScreenBBox(ent),box)) selSet.add(ent); }
    updateSelInfo(); render();
  }

  /* ---- 撤回 / 恢复 ---- */
  let undoStack = [], redoStack = [], gesturePre = null;
  function snapshot(){ return JSON.parse(JSON.stringify(entities)); }
  function pushUndo(pre){
    undoStack.push(pre!==undefined ? pre : snapshot());
    if(undoStack.length>200) undoStack.shift();
    redoStack.length = 0; updateUndoRedoBtns();
  }
  function undo(){
    if(!undoStack.length) return;
    redoStack.push(snapshot());
    entities = undoStack.pop(); selSet.clear(); updateSelInfo(); render(); fireChange(); updateUndoRedoBtns();
  }
  function redo(){
    if(!redoStack.length) return;
    undoStack.push(snapshot());
    entities = redoStack.pop(); selSet.clear(); updateSelInfo(); render(); fireChange(); updateUndoRedoBtns();
  }
  function updateUndoRedoBtns(){
    const u=$('#undoBtn'), r=$('#redoBtn');
    if(u) u.disabled = !undoStack.length;
    if(r) r.disabled = !redoStack.length;
  }

  /* ---- 尺寸栏：直线长度 / 圆半径 / n边形边数与边长 ---- */
  const dimRow=$('#dimRow'), dimLine=$('#dimLine'), dimCircle=$('#dimCircle'), dimPoly=$('#dimPoly'), dimRotate=$('#dimRotate');
  const dimLen=$('#dimLen'), dimRad=$('#dimRad'), dimN=$('#dimN'),
        dimRegular=$('#dimRegular'), dimSide=$('#dimSide'),
        dimRegBox=$('#dimRegBox'), dimIrrBox=$('#dimIrrBox'), dimSides=$('#dimSides'),
        dimText=$('#dimText'), dimTextH=$('#dimTextH'), dimTextStr=$('#dimTextStr'),
        rotAngle=$('#rotAngle'), applyRot=$('#applyRot'),
        snapChk=$('#snapChk'), snapPxIn=$('#snapPxIn'), snapDegIn=$('#snapDegIn');
  // 吸附设置（每次都现读控件，改完立刻生效，不用重启页面）
  function snapCfg(){
    const on = !!(snapChk && snapChk.checked);
    const px = Math.max(0, Math.min(60, parseFloat(snapPxIn && snapPxIn.value) || 0));
    const deg = Math.max(0, Math.min(45, parseFloat(snapDegIn && snapDegIn.value) || 0));
    return {on:on, px:px, deg:deg, w:(on ? px/Math.max(view.scale,1e-6) : 0)};
  }
  function updDimRow(){
    dimLine.style.display   = tool==='line'   ? 'inline' : 'none';
    dimCircle.style.display = tool==='circle' ? 'inline' : 'none';
    dimPoly.style.display   = tool==='poly'   ? 'inline' : 'none';
    dimText.style.display   = tool==='text'   ? 'inline' : 'none';
    dimRotate.style.display = tool==='rotate' ? 'inline' : 'none';
    dimRow.style.display    = (tool==='line'||tool==='circle'||tool==='poly'||tool==='text'||tool==='rotate') ? 'flex' : 'none';
  }
  dimRegular.onchange = ()=>{
    dimRegBox.style.display = dimRegular.checked ? 'inline' : 'none';
    dimIrrBox.style.display = dimRegular.checked ? 'none'   : 'inline';
  };
  function placePoly(c){
    const n=Math.max(3,Math.min(24,parseInt(dimN.value,10)||6));
    let sides=null;
    if(dimRegular.checked){
      const s=parseFloat(dimSide.value);
      if(!(s>0)){ alert('请先在尺寸栏填写边长（mm）'); return; }
      sides=Array(n).fill(s);
    }else{
      const parts=dimSides.value.split(/[,，;；\\s]+/).filter(x=>x.length);
      const vals=parts.map(Number);
      if(vals.length<3 || vals.some(v=>!(v>0))){
        alert('非正多边形请填写至少 3 个用逗号分隔的正数边长（mm）'); return;
      }
      const total=vals.reduce((a,b)=>a+b,0);
      const bad=vals.find(L=>L>=total-L);
      if(bad!==undefined){
        alert('警告：边长不合理——边 '+bad+' 大于或等于其余各边之和，无法构成闭合多边形。\\n请调整各边长后重试，或勾选"正多边形"只填一个边长。');
        return;
      }
      sides=vals;
    }
    entities.push({type:'polyline',points:makePolyPoints(c,sides).map(rnd),closed:true,layer:String(activeLayer),dash:newDash,dashLen:newDashLen,dashGap:newDashGap});
    render(); fireChange();
  }

  function ev2canvas(e){
    const r = canvas.getBoundingClientRect();
    return [ (e.clientX - r.left) * (canvas.width / r.width),
             (e.clientY - r.top) * (canvas.height / r.height) ];
  }
  function arcPoints(e){
    // 圆弧：服务端会带 points 细分点；万一只有 center/radius/start/end（手写数据），
    // 这里现算一遍，避免图元"选中了却看不见"。
    if(e.points && e.points.length>=2) return e.points;
    const c=e.center||[0,0], r=e.radius||0;
    let s=Number(e.start||0), t=Number(e.end||0);
    while(t<=s) t+=360;
    const n=Math.max(8, Math.round((t-s)/6)), out=[];
    for(let k=0;k<=n;k++){
      const a=(s+(t-s)*k/n)*Math.PI/180;
      out.push([c[0]+r*Math.cos(a), c[1]+r*Math.sin(a)]);
    }
    return out;
  }
  function entityPoints(e){
    if(e.type==='line') return [e.p1, e.p2];
    if(e.type==='rect') return [[e.x,e.y],[e.x+e.w,e.y],[e.x+e.w,e.y+e.h],[e.x,e.y+e.h]];
    if(e.type==='circle'){const c=e.center,r=e.radius;return [[c[0]-r,c[1]],[c[0]+r,c[1]],[c[0],c[1]-r],[c[0],c[1]+r]];}
    if(e.type==='polyline') return e.points||[];
    if(e.type==='spline'||e.type==='ellipse') return e.points||[];
    if(e.type==='arc') return arcPoints(e);
    if(e.type==='text') return [e.p];
    return [];
  }
  function worldBounds(){
    let xs=[],ys=[];
    for(const e of entities){
      if(e.type==='text'){
        const hpx=Math.max(8,(e.height||5)*view.scale);
        ctx.font=hpx+'px system-ui,"Microsoft YaHei",sans-serif';
        const wWorld=ctx.measureText(e.value||'').width/view.scale;
        xs.push(e.p[0], e.p[0]+wWorld); ys.push(e.p[1]-(e.height||5), e.p[1]);
      } else {
        for(const p of entityPoints(e)){xs.push(p[0]);ys.push(p[1]);}
      }
    }
    if(!xs.length) return null;
    return [Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];
  }
  function fitView(){
    const b=worldBounds(), W=canvas.width,H=canvas.height;
    if(!b){view={cx:0,cy:0,scale:10};return;}
    const [x0,y0,x1,y1]=b, pad=30;
    const w=Math.max(x1-x0,1e-3), h=Math.max(y1-y0,1e-3);
    view.scale=Math.min((W-2*pad)/w,(H-2*pad)/h);
    view.cx=(x0+x1)/2; view.cy=(y0+y1)/2;
  }
  function w2s(p){const W=canvas.width,H=canvas.height;return [(p[0]-view.cx)*view.scale+W/2, H/2-(p[1]-view.cy)*view.scale];}
  function s2w(p){const W=canvas.width,H=canvas.height;return [(p[0]-W/2)/view.scale+view.cx,(H/2-p[1])/view.scale+view.cy];}
  function dist2seg(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1];const L2=dx*dx+dy*dy;let t=L2?((p[0]-a[0])*dx+(p[1]-a[1])*dy)/L2:0;t=Math.max(0,Math.min(1,t));const x=a[0]+t*dx,y=a[1]+t*dy;return Math.hypot(p[0]-x,p[1]-y);}
  function textBoxScreen(e){
    const p=w2s(e.p);
    const hpx=Math.max(8,(e.height||5)*view.scale);
    ctx.font=hpx+'px system-ui,"Microsoft YaHei",sans-serif';
    const m=ctx.measureText(e.value||'');
    const asc=m.actualBoundingBoxAscent||hpx*0.8;
    const desc=m.actualBoundingBoxDescent||hpx*0.2;
    return {x0:p[0], y0:p[1]-asc, x1:p[0]+m.width, y1:p[1]+desc};
  }
  function hitTest(sp){
    for(let i=entities.length-1;i>=0;i--){
      const e=entities[i];
      if(e.type==='text'){const b=textBoxScreen(e),t=HIT;
        if(sp[0]>=b.x0-t&&sp[0]<=b.x1+t&&sp[1]>=b.y0-t&&sp[1]<=b.y1+t)return i;continue;}
      if(e.type==='circle'){const c=w2s(e.center),r=e.radius*view.scale;if(Math.abs(Math.hypot(sp[0]-c[0],sp[1]-c[1])-r)<=HIT)return i;continue;}
      if(e.type==='rect'){const p=entityPoints(e);const inside=(sp[0]>=Math.min(p[0][0],p[1][0])&&sp[0]<=Math.max(p[0][0],p[1][0])&&sp[1]>=Math.min(p[0][1],p[2][1])&&sp[1]<=Math.max(p[0][1],p[2][1]));if(inside)return i;}
      const pts=entityPoints(e).map(w2s);
      const segs = e.type==='rect'?[[0,1],[1,2],[2,3],[3,0]]:
                   e.closed?pts.map((_,k)=>[k,(k+1)%pts.length]):
                   pts.slice(0,-1).map((_,k)=>[k,k+1]);
      for(const [a,b] of segs){ if(dist2seg(sp,pts[a],pts[b])<=HIT) return i; }
    }
    return -1;
  }
  function rnd(p){return [Math.round(p[0]*1000)/1000,Math.round(p[1]*1000)/1000];}
  /* ================= 与水平夹角（世界坐标 y 向上，逆时针为正，折到 0~180°）=================
     约定：0° = 水平（+X），90° = 竖直（+Y）。方向无正反之分，所以 a 与 a+180 视为同一角度。 */
  function dirAngleDeg(a,b){
    let d=Math.atan2(b[1]-a[1], b[0]-a[0])*180/Math.PI;
    d=((d%360)+360)%360;
    return d>=180 ? d-180 : d;
  }
  function entAngleDeg(e){
    if(!e) return null;
    if(e.type==='line') return dirAngleDeg(e.p1,e.p2);
    if(e.type==='rect') return 0;                                  // 轴对齐矩形按 0° 看
    if(e.type==='ellipse') return (((Number(e.rotation||0))%180)+180)%180;
    if(e.type==='polyline'||e.type==='spline'||e.type==='arc'){
      const p=entityPoints(e); if(!p||p.length<2) return null;
      const n=(e.closed?p.length:p.length-1);
      let best=0,bL=-1;
      for(let k=0;k<n;k++){
        const a=p[k], b=p[(k+1)%p.length];
        const L=Math.hypot(b[0]-a[0],b[1]-a[1]);
        if(L>bL){bL=L;best=dirAngleDeg(a,b);}
      }
      return bL>1e-9 ? best : null;
    }
    return null;                                                   // 圆 / 文本没有方向
  }
  // 选中集的"代表角"：取最长的那条有方向的图元
  function selAngleDeg(set){
    let best=null,bL=-1;
    for(const e of set){
      const a=entAngleDeg(e);
      if(a===null) continue;
      let L=0;
      if(e.type==='line') L=Math.hypot(e.p2[0]-e.p1[0],e.p2[1]-e.p1[1]);
      else{ const p=entityPoints(e); for(let k=1;k<p.length;k++) L+=Math.hypot(p[k][0]-p[k-1][0],p[k][1]-p[k-1][1]); }
      if(L>bL){bL=L;best=a;}
    }
    return best;
  }
  function angHint(a){
    if(Math.abs(a)<0.05) return '（水平）';
    if(Math.abs(a-90)<0.05) return '（竖直）';
    return a<90 ? '（斜向右上）' : '（斜向左上）';
  }
  /* ================= 吸附 =================
     三档：位移正交吸附 → 线条扶正 → 顶点/线段吸附。强度由工具栏「🧲 吸附」的距离(px)
     与角度(°) 控制，拖动过程中改数字立刻生效。 */
  // 图元的"锚点"：拖动时参与吸附对齐的代表点（对齐到别的图元上）
  function anchorPtsOf(e){
    if(e.type==='circle'){ const c=e.center,r=e.radius;
      return [c,[c[0]-r,c[1]],[c[0]+r,c[1]],[c[0],c[1]-r],[c[0],c[1]+r]]; }
    if(e.type==='rect'){ const p=entityPoints(e); return p.concat([[e.x+e.w/2,e.y+e.h/2]]); }
    if(e.type==='arc'){ const p=entityPoints(e); return p.length?[p[0],p[p.length-1],e.center]:[]; }
    if(e.type==='line'){ return [e.p1,e.p2,[(e.p1[0]+e.p2[0])/2,(e.p1[1]+e.p2[1])/2]]; }
    const p=entityPoints(e);
    if(p.length<=12) return p.slice();
    const step=Math.max(1,Math.floor(p.length/6)), pick=[];
    for(let k=0;k<p.length;k+=step) pick.push(p[k]);
    if(pick[pick.length-1]!==p[p.length-1]) pick.push(p[p.length-1]);
    return pick;
  }
  // 拖拽开始时算一次（未选中的图元在拖拽中不动，所以候选点/线段可以缓存）
  function collectSnapTargets(exclude){
    const pts=[], segs=[];
    for(const e of entities){
      if(exclude.has(e)) continue;
      for(const q of anchorPtsOf(e)) pts.push(q);
      if(e.type==='circle'){
        const c=e.center,r=e.radius,N=24,poly=[];
        for(let k=0;k<N;k++){const th=k*2*Math.PI/N;poly.push([c[0]+r*Math.cos(th),c[1]+r*Math.sin(th)]);}
        for(let k=0;k<N;k++) segs.push([poly[k],poly[(k+1)%N]]);
        continue;
      }
      const p=entityPoints(e);
      if(p.length>=2 && p.length<=240){
        const n=(e.closed||e.type==='rect')?p.length:p.length-1;
        for(let k=0;k<n;k++) segs.push([p[k],p[(k+1)%p.length]]);
      }
    }
    return {pts:pts, segs:segs};
  }
  function projectOnSeg(p,a,b){
    const dx=b[0]-a[0],dy=b[1]-a[1],L2=dx*dx+dy*dy;
    let t=L2?((p[0]-a[0])*dx+(p[1]-a[1])*dy)/L2:0;
    t=Math.max(0,Math.min(1,t));
    return [a[0]+t*dx, a[1]+t*dy];
  }
  // 线条扶正：与水平/竖直夹角小于容差时改成精确 0°/90°（保长度，只挪 p2）
  function straightenSelLines(list,tolDeg){
    if(!(tolDeg>0)) return 0;
    let n=0;
    for(const e of list){
      if(!e||e.type!=='line') continue;
      const L=Math.hypot(e.p2[0]-e.p1[0],e.p2[1]-e.p1[1]);
      if(L<1e-6) continue;
      const a=dirAngleDeg(e.p1,e.p2);
      const dH=Math.min(a,180-a), dV=Math.abs(a-90);
      if(dH<=tolDeg && dH<=dV){
        const s=(e.p2[0]>=e.p1[0])?1:-1;
        e.p2=[Math.round((e.p1[0]+s*L)*1e6)/1e6, e.p1[1]]; n++;
      } else if(dV<=tolDeg){
        const s=(e.p2[1]>=e.p1[1])?1:-1;
        e.p2=[e.p1[0], Math.round((e.p1[1]+s*L)*1e6)/1e6]; n++;
      }
    }
    return n;
  }
  // 按"基准快照 + 位移"重画选中集，再扶正线（每帧从头算，避免增量累积漂移）
  function applyMoveFromBases(bases,dx,dy,cfg){
    bases.forEach(([en,base])=>{
      const t=JSON.parse(JSON.stringify(base));
      moveEntity(t,dx,dy);
      Object.assign(en,t);
    });
    if(cfg&&cfg.on&&cfg.deg>0) straightenSelLines(bases.map(b=>b[0]),cfg.deg);
  }
  // 顶点优先、线段其次的吸附修正量
  function snapAnchorOffset(drag,cfg){
    let bestD=cfg.w, bestOff=null, bestP=null, bestKind='pt';
    for(const [en] of drag.bases){
      for(const q of anchorPtsOf(en)){
        for(const t of drag.targets.pts){
          const d=Math.hypot(t[0]-q[0],t[1]-q[1]);
          if(d<bestD){bestD=d;bestOff=[t[0]-q[0],t[1]-q[1]];bestP=t;bestKind='pt';}
        }
      }
    }
    if(!bestOff){
      for(const [en] of drag.bases){
        for(const q of anchorPtsOf(en)){
          for(const s of drag.targets.segs){
            if(q[0]<Math.min(s[0][0],s[1][0])-cfg.w || q[0]>Math.max(s[0][0],s[1][0])+cfg.w) continue;
            if(q[1]<Math.min(s[0][1],s[1][1])-cfg.w || q[1]>Math.max(s[0][1],s[1][1])+cfg.w) continue;
            const pr=projectOnSeg(q,s[0],s[1]);
            const d=Math.hypot(pr[0]-q[0],pr[1]-q[1]);
            if(d<bestD){bestD=d;bestOff=[pr[0]-q[0],pr[1]-q[1]];bestP=pr;bestKind='seg';}
          }
        }
      }
    }
    return bestOff ? {d:bestOff, hint:{p:bestP,kind:bestKind}} : null;
  }
  // 注意：ref 既可以是图元对象（selSet 里存的就是对象），也可以是数组下标 —— 两种都认，
  // 免得调用方传对象时走到 entities[object] → undefined 直接抛异常把整条拖拽打断。
  function moveEntity(ref,dx,dy){
    const e = (typeof ref === 'number') ? entities[ref] : ref;
    if(!e || !e.type) return;
    if(e.type==='line'){e.p1=[e.p1[0]+dx,e.p1[1]+dy];e.p2=[e.p2[0]+dx,e.p2[1]+dy];}
    else if(e.type==='rect'){e.x+=dx;e.y+=dy;}
    else if(e.type==='circle'){e.center=[e.center[0]+dx,e.center[1]+dy];}
    else if(e.type==='polyline'){e.points=e.points.map(p=>[p[0]+dx,p[1]+dy]);}
    else if(e.type==='spline'||e.type==='ellipse'){
      e.points=(e.points||[]).map(p=>[p[0]+dx,p[1]+dy]);
      if(e.center) e.center=[e.center[0]+dx,e.center[1]+dy];
    }
    else if(e.type==='arc'){
      e.points=arcPoints(e).map(p=>[p[0]+dx,p[1]+dy]);
      if(e.center) e.center=[e.center[0]+dx,e.center[1]+dy];
    }
    else if(e.type==='text'){e.p=[e.p[0]+dx,e.p[1]+dy];}
  }
  /* ---- 旋转：绕中心点旋转图元（矩形旋转后转为闭合多段线） ---- */
  function rotatePts(pts,cx,cy,a){
    const co=Math.cos(a),si=Math.sin(a);
    return pts.map(p=>[cx+(p[0]-cx)*co-(p[1]-cy)*si, cy+(p[0]-cx)*si+(p[1]-cy)*co]);
  }
  function centroid(e){
    const p=entityPoints(e);
    if(!p.length) return [0,0];
    return [p.reduce((s,q)=>s+q[0],0)/p.length, p.reduce((s,q)=>s+q[1],0)/p.length];
  }
  function combinedCentroid(set){
    let xs=[],ys=[];
    set.forEach(en=>{ for(const p of entityPoints(en)){ xs.push(p[0]); ys.push(p[1]); } });
    if(!xs.length) return [0,0];
    return [xs.reduce((a,b)=>a+b,0)/xs.length, ys.reduce((a,b)=>a+b,0)/ys.length];
  }
  function rotateEntity(e,c,a){
    const st={dash:e.dash,dashLen:e.dashLen,dashGap:e.dashGap,arrow:e.arrow,arrowSize:e.arrowSize};
    const withSt=o=>Object.assign(o,st);
    if(e.type==='line'){const p=rotatePts([e.p1,e.p2],c[0],c[1],a);
      return withSt({type:'line',p1:rnd(p[0]),p2:rnd(p[1]),layer:e.layer||'0'});}
    if(e.type==='polyline'){return withSt({type:'polyline',points:rotatePts(e.points||[],c[0],c[1],a).map(rnd),closed:e.closed,layer:e.layer||'0'});}
    if(e.type==='spline'||e.type==='ellipse'){
      const out=withSt({type:e.type,points:rotatePts(e.points||[],c[0],c[1],a).map(rnd),
                 closed:e.closed,layer:e.layer||'0'});
      if(e.type==='ellipse'){
        const cc=rotatePts([e.center||[0,0]],c[0],c[1],a)[0];
        out.center=rnd(cc); out.rx=e.rx; out.ry=e.ry;
        out.rotation=(e.rotation||0)+a*180/Math.PI;
        out.start=e.start; out.end=e.end;
      }
      return out;
    }
    if(e.type==='arc'){
      const cc=rotatePts([e.center||[0,0]],c[0],c[1],a)[0];
      const d=a*180/Math.PI;
      return withSt({type:'arc',center:rnd(cc),radius:e.radius,
              start:((Number(e.start||0)+d)%360+360)%360,
              end:((Number(e.end||0)+d)%360+360)%360,
              points:rotatePts(arcPoints(e),c[0],c[1],a).map(rnd),
              layer:e.layer||'0'});
    }
    if(e.type==='rect'){const cor=[[e.x,e.y],[e.x+e.w,e.y],[e.x+e.w,e.y+e.h],[e.x,e.y+e.h]];
      return withSt({type:'polyline',points:rotatePts(cor,c[0],c[1],a).map(rnd),closed:true,layer:e.layer||'0'});}
    if(e.type==='circle'){const p=rotatePts([e.center],c[0],c[1],a)[0];
      return withSt({type:'circle',center:rnd(p),radius:e.radius,layer:e.layer||'0'});}
    return e;
  }
  /* ---- n边形：按边长构造圆内接多边形（支持非等边，需满足多边形存在条件） ----
     原理：任意一组满足"每边 < 其余边之和"的边长都存在圆内接多边形。
     二分外接圆半径 R，使各边所对圆心角之和 = 2π；最长边在钝角情形走优弧。 */
  function makePolyPoints(center,sides){
    const m=Math.max(...sides);
    const ang=(R,L)=>2*Math.asin(Math.min(1,L/(2*R)));
    let R,thetas;
    const sumAll=r=>sides.reduce((s,L)=>s+ang(r,L),0);
    const R0=m/2*1.0000001;
    if(sumAll(R0)>=2*Math.PI){          // 各边走劣弧即可闭合
      let lo=R0,hi=R0;
      while(sumAll(hi)>2*Math.PI) hi*=2;
      for(let k=0;k<80;k++){const mid=(lo+hi)/2; if(sumAll(mid)>2*Math.PI) lo=mid; else hi=mid;}
      R=(lo+hi)/2;
      thetas=sides.map(L=>ang(R,L));
    }else{                              // 最长边走优弧（圆心在形外的钝角情形）
      const iM=sides.indexOf(m);
      const rest=sides.filter((_,i)=>i!==iM);
      const h=r=>rest.reduce((s,L)=>s+ang(r,L),0)-ang(r,m);
      let lo=R0,hi=R0;
      while(h(hi)<0) hi*=2;
      for(let k=0;k<80;k++){const mid=(lo+hi)/2; if(h(mid)<0) lo=mid; else hi=mid;}
      R=(lo+hi)/2;
      thetas=sides.map((L,i)=> i===iM ? (2*Math.PI-ang(R,L)) : ang(R,L));
    }
    const pts=[]; let a0=0;
    for(const th of thetas){ pts.push([center[0]+R*Math.cos(a0), center[1]+R*Math.sin(a0)]); a0+=th; }
    return pts;
  }
  // 橡皮擦：把任意图元密集采样成点序列（世界坐标），笔刷圆（屏幕坐标）命中的
  // 采样点即被擦除，其余连续片段保留为 polyline —— 效果等同微软画图橡皮
  // （只擦掉接触到的部分，不整条删除；直线断成两截、圆/矩形被咬掉一块变成开口）。
  function flattenEntity(e){
    if(e.type==='line'){
      const a=e.p1, b=e.p2, sa=w2s(a), sb=w2s(b);
      const len=Math.hypot(sa[0]-sb[0], sa[1]-sb[1]);
      const N=Math.max(2, Math.ceil(len/2)); const pts=[];
      for(let k=0;k<=N;k++){const t=k/N; pts.push([a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]);}
      return {pts, closed:false};
    }
    if(e.type==='polyline'||e.type==='spline'||e.type==='ellipse'||e.type==='arc'){
      const src=entityPoints(e); if(src.length<2) return {pts:[], closed:false};
      const segs=e.closed?src.length:src.length-1; const out=[];
      for(let s=0;s<segs;s++){
        const a=src[s], b=src[(s+1)%src.length];
        const sa=w2s(a), sb=w2s(b);
        const len=Math.hypot(sa[0]-sb[0], sa[1]-sb[1]);
        const N=Math.max(1, Math.ceil(len/2));
        for(let k=0;k<N;k++){const t=k/N; out.push([a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]);}
      }
      out.push(src[src.length-1]); if(e.closed) out.push(src[0]);
      return {pts:out, closed:e.closed};
    }
    if(e.type==='rect'){
      const c=[[e.x,e.y],[e.x+e.w,e.y],[e.x+e.w,e.y+e.h],[e.x,e.y+e.h]];
      const out=[];
      for(let s=0;s<4;s++){
        const a=c[s], b=c[(s+1)%4];
        const sa=w2s(a), sb=w2s(b);
        const len=Math.hypot(sa[0]-sb[0], sa[1]-sb[1]);
        const N=Math.max(1, Math.ceil(len/2));
        for(let k=0;k<N;k++){const t=k/N; out.push([a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]);}
      }
      out.push(c[0]);
      return {pts:out, closed:true};
    }
    if(e.type==='circle'){
      const c=e.center, r=e.radius, rscr=r*view.scale;
      const N=Math.max(16, Math.ceil(2*Math.PI*rscr/2)); const pts=[];
      for(let k=0;k<=N;k++){const ang=2*Math.PI*k/N; pts.push([c[0]+r*Math.cos(ang), c[1]+r*Math.sin(ang)]);}
      return {pts, closed:true};
    }
    return {pts:[], closed:false};
  }
  function applyEraser(){
    if(!eraserBase) return;
    const R = eraserSize/2;
    const frags=[];
    for(const e of eraserBase){
      const f=flattenEntity(e);
      if(f.pts.length<2){ frags.push(e); continue; }
      const scr=f.pts.map(w2s);
      const erased=scr.map(p=>{
        for(const c of eraserTrail){ if(Math.hypot(p[0]-c[0], p[1]-c[1])<=R) return true; }
        return false;
      });
      if(!erased.some(Boolean)){ frags.push(e); continue; }   // 没碰到 → 原样保留
      let run=[];
      for(let k=0;k<scr.length;k++){
        if(erased[k]){ if(run.length>=2) frags.push({pts:run.map(i=>f.pts[i]), src:e}); run=[]; }
        else run.push(k);
      }
      if(run.length>=2) frags.push({pts:run.map(i=>f.pts[i]), src:e});
    }
    // 碎片继承原图元的线型（虚线段长/间隔）；箭头不继承——碎片是新的端点几何
    entities = frags.map(x => (x && x.pts)
      ? {type:'polyline', points:x.pts.map(rnd), closed:false, layer:String(activeLayer),
         dash:x.src.dash?true:false, dashLen:x.src.dashLen, dashGap:x.src.dashGap}
      : x);
    fireChange();
  }
  /* ---- 虚线像素换算 + 箭头三角形 ---- */
  function dashPx(e){
    const d=Math.max(0.5,Number(e&&e.dashLen)||newDashLen), g=Math.max(0.5,Number(e&&e.dashGap)||newDashGap);
    const cl=v=>Math.max(2,Math.min(90,v));
    return [cl(d*view.scale), cl(g*view.scale)];
  }
  function isArrowable(e){ return e.type==='line' || ((e.type==='polyline'||e.type==='spline') && !e.closed); }
  function arrowTris(e){
    if(!e.arrow || !isArrowable(e)) return [];
    let pts=null;
    if(e.type==='line') pts=[e.p1,e.p2];
    else if(e.type==='polyline') pts=e.points;
    else if(e.type==='spline') pts=(e.points&&e.points.length>=2)?e.points:entityPoints(e);
    if(!pts||pts.length<2) return [];
    const size=Math.max(0.5,Number(e.arrowSize)||newArrowSize), tris=[];
    const mk=(tip,prev)=>{const dx=tip[0]-prev[0],dy=tip[1]-prev[1];const L=Math.hypot(dx,dy);if(L<1e-9)return;
      const ux=dx/L,uy=dy/L,px=-uy,py=ux,w=size*0.3,bx=tip[0]-ux*size,by=tip[1]-uy*size;
      tris.push([[tip[0],tip[1]],[bx+px*w,by+py*w],[bx-px*w,by-py*w]]);};
    if(e.arrow&1) mk(pts[0],pts[1]);
    if(e.arrow&2) mk(pts[pts.length-1],pts[pts.length-2]);
    return tris;
  }
  function drawArrowTris(e,sel){
    const tris=arrowTris(e); if(!tris.length) return;
    ctx.setLineDash([]); ctx.fillStyle=sel?'#9b0000':layerColorCss(e.layer);
    for(const t of tris){const s=t.map(w2s);
      ctx.beginPath();ctx.moveTo(s[0][0],s[0][1]);ctx.lineTo(s[1][0],s[1][1]);ctx.lineTo(s[2][0],s[2][1]);
      ctx.closePath();ctx.fill();}
  }
  function drawEntity(e,sel){
    if(e.type==='text'){
      const p=w2s(e.p); const hpx=Math.max(8,(e.height||5)*view.scale);
      ctx.save();
      ctx.fillStyle = sel ? '#9b0000' : layerColorCss(e.layer);
      ctx.font = hpx + 'px system-ui,"Microsoft YaHei",sans-serif';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(e.value || '', p[0], p[1]);
      if(sel){
        const b=textBoxScreen(e);
        ctx.strokeStyle='#9b0000'; ctx.setLineDash([4,3]); ctx.lineWidth=1.2;
        ctx.strokeRect(b.x0-2,b.y0-2,(b.x1-b.x0)+4,(b.y1-b.y0)+4);
      }
      ctx.restore();
      return;
    }
    ctx.lineWidth=sel?3:1.6; ctx.strokeStyle=sel?'#9b0000':layerColorCss(e.layer);
    ctx.setLineDash(e.dash?dashPx(e):[]);
    if(e.type==='line'){const a=w2s(e.p1),b=w2s(e.p2);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
    else if(e.type==='rect'){const p=entityPoints(e).map(w2s);ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);for(let k=1;k<4;k++)ctx.lineTo(p[k][0],p[k][1]);ctx.closePath();ctx.stroke();}
    else if(e.type==='circle'){const c=w2s(e.center),r=e.radius*view.scale;ctx.beginPath();ctx.arc(c[0],c[1],r,0,2*Math.PI);ctx.stroke();}
    else if(e.type==='polyline'||e.type==='spline'||e.type==='ellipse'||e.type==='arc'){const p=entityPoints(e).map(w2s);if(!p.length)return;ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);for(let k=1;k<p.length;k++)ctx.lineTo(p[k][0],p[k][1]);if(e.closed)ctx.closePath();ctx.stroke();}
    drawArrowTris(e,sel);
    ctx.setLineDash([]);
  }
  function drawPreview(){
    ctx.lineWidth=1.6; ctx.strokeStyle='#36c275';
    if(!draw) return;
    if(draw.kind==='line'){const a=w2s(draw.a),b=w2s(draw.b);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.setLineDash(newDash?dashPx(null):[]);ctx.stroke();ctx.setLineDash([]);
      if(newArrow){ctx.fillStyle='#36c275';
        for(const tr of arrowTris({type:'line',p1:draw.a,p2:draw.b,arrow:newArrow,arrowSize:newArrowSize})){
          const s=tr.map(w2s);ctx.beginPath();ctx.moveTo(s[0][0],s[0][1]);ctx.lineTo(s[1][0],s[1][1]);ctx.lineTo(s[2][0],s[2][1]);ctx.closePath();ctx.fill();}}
    }
    else if(draw.kind==='circle'){const c=w2s(draw.c),r=draw.r*view.scale;ctx.beginPath();ctx.arc(c[0],c[1],r,0,2*Math.PI);ctx.stroke();}
    else if(draw.kind==='pen'){const p=draw.pts.map(w2s);ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);for(let k=1;k<p.length;k++)ctx.lineTo(p[k][0],p[k][1]);ctx.stroke();}
    else if(draw.kind==='curve'){
      const A=w2s(draw.a),B=w2s(draw.b);
      const ctrl = draw.ctrl ? w2s(draw.ctrl) : [(A[0]+B[0])/2,(A[1]+B[1])/2];
      ctx.beginPath();ctx.moveTo(A[0],A[1]);
      for(let t=0;t<=1;t+=0.05){const x=(1-t)*(1-t)*A[0]+2*(1-t)*t*ctrl[0]+t*t*B[0];const y=(1-t)*(1-t)*A[1]+2*(1-t)*t*ctrl[1]+t*t*B[1];ctx.lineTo(x,y);}
      ctx.stroke();
    }
  }
  function render(){
    const W=canvas.width,H=canvas.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle='#180b0d';ctx.fillRect(0,0,W,H);
    ctx.strokeStyle='#261014';ctx.lineWidth=1;
    for(let gx=0;gx<=W;gx+=24){ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke();}
    for(let gy=0;gy<=H;gy+=24){ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke();}
    for(let i=0;i<entities.length;i++) drawEntity(entities[i], selSet.has(entities[i]));
    if(draw) drawPreview();
    if(drag && drag.hint && drag.hint.p){
      const s=w2s(drag.hint.p), r=7;
      ctx.save();
      ctx.strokeStyle='#ffd34d'; ctx.lineWidth=1.8;
      if(drag.hint.kind==='pt'){                 // 顶点吸附：方框 + 十字
        ctx.beginPath(); ctx.strokeRect(s[0]-r,s[1]-r,2*r,2*r);
        ctx.beginPath();
        ctx.moveTo(s[0]-r-4,s[1]); ctx.lineTo(s[0]+r+4,s[1]);
        ctx.moveTo(s[0],s[1]-r-4); ctx.lineTo(s[0],s[1]+r+4);
        ctx.stroke();
      }else{                                     // 线段吸附：三角
        ctx.beginPath();
        ctx.moveTo(s[0],s[1]-r); ctx.lineTo(s[0]+r,s[1]+r*0.9);
        ctx.lineTo(s[0]-r,s[1]+r*0.9); ctx.closePath(); ctx.stroke();
      }
      ctx.restore();
    }
    if(drag && drag.mode==='marquee'){
      const s=drag.start, c=drag.cur;
      ctx.save();
      ctx.fillStyle='rgba(54,194,117,0.12)';
      ctx.strokeStyle='rgba(54,194,117,0.9)';
      ctx.lineWidth=1; ctx.setLineDash([5,3]);
      const x=Math.min(s[0],c[0]), y=Math.min(s[1],c[1]),
            w=Math.abs(c[0]-s[0]), h=Math.abs(c[1]-s[1]);
      ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
      ctx.restore();
    }
    if(tool==='erase' && cursor && cursor[0]>=0 && cursor[0]<=W && cursor[1]>=0 && cursor[1]<=H){
      ctx.save();
      ctx.beginPath();
      ctx.arc(cursor[0], cursor[1], eraserSize, 0, 2*Math.PI);
      ctx.fillStyle='rgba(255,120,120,0.18)';
      ctx.fill();
      ctx.strokeStyle='rgba(255,150,150,0.9)';
      ctx.lineWidth=1.5;
      ctx.stroke();
      ctx.restore();
    }
  }
  function commitCurve(){
    const A=draw.a,B=draw.b,C=draw.ctrl||[(A[0]+B[0])/2,(A[1]+B[1])/2];
    if(Math.hypot(A[0]-B[0],A[1]-B[1])<1e-3){ draw=null; render(); return; }  // 起点≈终点，放弃
    // 二次贝塞尔 → 采 9 个在曲线上的点 → spline 实体。
    // 不再存成 21 点折线：那样放大看全是小直线，导出 DXF 也只是多段线。
    const pts=[];
    for(let k=0;k<=8;k++){
      const t=k/8, x=(1-t)*(1-t)*A[0]+2*(1-t)*t*C[0]+t*t*B[0],
                    y=(1-t)*(1-t)*A[1]+2*(1-t)*t*C[1]+t*t*B[1];
      pts.push([Math.round(x*1000)/1000,Math.round(y*1000)/1000]);
    }
    entities.push({type:'spline',points:pts,closed:false,layer:String(activeLayer),dash:newDash,dashLen:newDashLen,dashGap:newDashGap,arrow:newArrow,arrowSize:newArrowSize});
    draw=null; render(); fireChange();
  }
  function endDrag(e){
    if(drag && drag.mode==='erase'){ eraserBase=null; eraserTrail=null; drag=null; render(); return; }
    const sp=ev2canvas(e);
    if(draw){
      if(draw.kind==='line'){
        let b=draw.b;
        const Ldim=parseFloat(dimLen.value);
        if(Ldim>0){
          const dx=b[0]-draw.a[0], dy=b[1]-draw.a[1], L=Math.hypot(dx,dy);
          b = L>1e-6 ? [draw.a[0]+dx/L*Ldim, draw.a[1]+dy/L*Ldim]   // 沿拖动方向定长
                     : [draw.a[0]+Ldim, draw.a[1]];                  // 原地点击→水平定长
        }
        if(Math.hypot(draw.a[0]-b[0],draw.a[1]-b[1])>1e-3)
          entities.push({type:'line',p1:rnd(draw.a),p2:rnd(b),layer:String(activeLayer),dash:newDash,dashLen:newDashLen,dashGap:newDashGap,arrow:newArrow,arrowSize:newArrowSize});
        draw=null;
      }
      else if(draw.kind==='circle'){ if(draw.r>0.5) entities.push({type:'circle',center:rnd(draw.c),radius:Math.round(draw.r*1000)/1000,layer:String(activeLayer),dash:newDash,dashLen:newDashLen,dashGap:newDashGap}); draw=null; }
      else if(draw.kind==='pen'){ if(draw.pts.length>=2) entities.push({type:'polyline',points:draw.pts.map(rnd),closed:false,layer:String(activeLayer),dash:newDash,dashLen:newDashLen,dashGap:newDashGap,arrow:newArrow,arrowSize:newArrowSize}); draw=null; }
    }
    drag=null; render();
  }
  canvas.addEventListener('mousedown',e=>{
    const sp=ev2canvas(e);
    gesturePre = snapshot();   // 记录手势前状态，供撤回（仅当本次确实改变图元时才入栈）
    if(tool==='select'){
      const hit=hitTest(sp);
      if(hit>=0){
        const ent=entities[hit];
        if(e.shiftKey){
          if(selSet.has(ent)) selSet.delete(ent); else selSet.add(ent);
          drag={mode:'none'}; updateSelInfo(); render();
        } else {
          if(!selSet.has(ent)){ selSet.clear(); selSet.add(ent); }
          const w0=s2w(sp), bases=[];
          selSet.forEach(en=>bases.push([en, JSON.parse(JSON.stringify(en))]));
          const cfg0=snapCfg();
          drag={mode:'move',last:w0,start:w0,bases:bases,hint:null,
                targets:cfg0.on?collectSnapTargets(selSet):{pts:[],segs:[]}};
        }
      } else {
        if(!e.shiftKey) selSet.clear();
        drag={mode:'marquee',add:e.shiftKey,start:sp,cur:sp};
        updateSelInfo();
      }
    } else if(tool==='rotate'){
      if(selSet.size===0){
        const hit=hitTest(sp);
        if(hit>=0){ selSet.add(entities[hit]); updateSelInfo(); }
      }
      if(selSet.size>0){
        const c=combinedCentroid(selSet);
        const bases=[];
        selSet.forEach(en=>bases.push([en, JSON.parse(JSON.stringify(en))]));
        const w=s2w(sp);
        drag={mode:'rotate',c:c,bases:bases,a0:Math.atan2(w[1]-c[1],w[0]-c[0]),selA:selAngleDeg(selSet)};
      }
    } else if(tool==='pan'){ drag={mode:'pan',last:sp}; }
    else if(tool==='line'){ draw={kind:'line',a:s2w(sp),b:s2w(sp)}; }
    else if(tool==='circle'){
      const Rdim=parseFloat(dimRad.value);
      draw={kind:'circle',c:s2w(sp),r:(Rdim>0?Rdim:0),fixed:(Rdim>0)};
    }
    else if(tool==='poly'){ placePoly(s2w(sp)); }
    else if(tool==='text'){ placeText(s2w(sp)); }
    else if(tool==='pen'){ draw={kind:'pen',pts:[s2w(sp)]}; }
    else if(tool==='erase'){ drag={mode:'erase'}; eraserBase=entities.map(e=>({...e})); eraserTrail=[sp]; applyEraser(); }
    else if(tool==='curve'){
      if(!draw) draw={kind:'curve',step:0,a:s2w(sp),b:s2w(sp),ctrl:null};
      else if(draw.step===0){ draw.b=s2w(sp); draw.step=1; }
      else { draw.ctrl=s2w(sp); commitCurve(); }
    }
    render();
  });
  window.addEventListener('mousemove',e=>{
    const sp=ev2canvas(e);
    if(tool==='erase'){ cursor=sp; }
    if(!drag&&!draw){ if(tool==='erase') render(); return; }
    if(drag&&drag.mode==='move'){
      const w=s2w(sp);
      let dx=w[0]-drag.start[0], dy=w[1]-drag.start[1];
      const cfg=snapCfg();
      if(cfg.on && cfg.w>0){
        // ① 位移正交吸附：拖动方向几乎纯水平/竖直时，吃掉另一轴的抖动
        const ax=Math.abs(dx), ay=Math.abs(dy);
        if(ax<cfg.w && ay>2*cfg.w) dx=0;
        else if(ay<cfg.w && ax>2*cfg.w) dy=0;
      }
      applyMoveFromBases(drag.bases,dx,dy,cfg);   // 含 ② 线条扶正（近水平/竖直→精确 0°/90°）
      drag.hint=null;
      if(cfg.on && cfg.w>0){
        // ③ 顶点/线段吸附：整体平移补齐到最近的目标点
        const off=snapAnchorOffset(drag,cfg);
        if(off){ dx+=off.d[0]; dy+=off.d[1]; applyMoveFromBases(drag.bases,dx,dy,cfg); drag.hint=off.hint; }
      }
      drag.last=w;
      updateSelInfo();
    }
    else if(drag&&drag.mode==='rotate'){
      const w=s2w(sp);
      const a=Math.atan2(w[1]-drag.c[1],w[0]-drag.c[0]);
      let delta=a-drag.a0;
      const cfg=snapCfg();
      if(cfg.on && cfg.deg>0 && drag.selA!==null && drag.selA!==undefined){
        const mod=(((drag.selA+delta*180/Math.PI)%180)+180)%180;
        let tgt=null;
        if(mod<=cfg.deg) tgt=0;
        else if(mod>=180-cfg.deg) tgt=180;
        else if(Math.abs(mod-90)<=cfg.deg) tgt=90;
        if(tgt!==null) delta=(tgt-drag.selA)*Math.PI/180;
      }
      drag.bases.forEach(([en,base])=>{Object.assign(en, rotateEntity(base, drag.c, delta));});
      drag.lastDeg = ((drag.selA===null||drag.selA===undefined) ? null : drag.selA + delta*180/Math.PI);
      updateSelInfo();
    }
    else if(drag&&drag.mode==='marquee'){drag.cur=sp;}
    else if(drag&&drag.mode==='pan'){const dx=(sp[0]-drag.last[0])/view.scale,dy=-(sp[1]-drag.last[1])/view.scale;view.cx-=dx;view.cy-=dy;drag.last=sp;}
    else if(drag&&drag.mode==='erase'){ const last=eraserTrail[eraserTrail.length-1]; if(!last||Math.hypot(sp[0]-last[0], sp[1]-last[1])>=1) eraserTrail.push(sp); applyEraser(); }
    else if(draw){
      if(draw.kind==='line')draw.b=s2w(sp);
      else if(draw.kind==='circle'){ if(!draw.fixed){const c=w2s(draw.c);draw.r=Math.hypot(sp[0]-c[0],sp[1]-c[1])/view.scale;} }
      else if(draw.kind==='pen'){const w=s2w(sp);const last=draw.pts[draw.pts.length-1];if(Math.hypot(w[0]-last[0],w[1]-last[1])*view.scale>3)draw.pts.push(w);}
      else if(draw.kind==='curve'){ if(draw.step===0)draw.b=s2w(sp); else draw.ctrl=s2w(sp); }
    }
    render();
  });
  window.addEventListener('mouseup',e=>{
    if(drag&&drag.mode==='marquee'){ applyMarquee(drag.start,drag.cur,drag.add); }
    if(drag) endDrag(e);                       // 拖拽类（直线/圆/画笔）抬起时提交
    else if(draw && draw.kind!=='curve') endDrag(e);  // 曲线用分步点击，不在此提交/清除
    if(gesturePre!==null){
      if(JSON.stringify(gesturePre)!==JSON.stringify(entities)) pushUndo(gesturePre);
      gesturePre=null;
    }
    drag=null;
    if(!(draw && draw.kind==='curve')) draw=null;   // 曲线保持跨多次点击的状态，直到第三步确认
  });
  canvas.addEventListener('wheel',e=>{
    e.preventDefault();
    const sp=ev2canvas(e); const before=s2w(sp);
    const f=e.deltaY<0?1.1:1/1.1;
    view.scale=Math.max(0.5,Math.min(2000,view.scale*f));
    const after=s2w(sp);
    view.cx+=before[0]-after[0]; view.cy+=before[1]-after[1];
    render();
  },{passive:false});
  document.addEventListener('keydown',e=>{
    const ae=document.activeElement;
    const inField = ae && (ae.tagName==='INPUT' || ae.tagName==='TEXTAREA');
    if((e.key==='Delete'||e.key==='Backspace') && selSet.size>0 && !inField){
      pushUndo();
      entities = entities.filter(en=>!selSet.has(en));
      selSet.clear(); updateSelInfo(); render(); fireChange();
    }
  });
  document.querySelectorAll('#toolbar .mode').forEach(el=>{
    el.onclick=()=>{document.querySelectorAll('#toolbar .mode').forEach(x=>x.classList.remove('sel'));el.classList.add('sel');tool=el.dataset.tool;draw=null;selSet.clear();updateSelInfo();cursor=null;updDimRow();render();};
  });
  canvas.addEventListener('mouseleave',()=>{ if(tool==='erase'){ cursor=null; render(); } });
  $('#eraserSize').addEventListener('input',e=>{ eraserSize=parseInt(e.target.value,10)||14; $('#eraserSizeVal').textContent=eraserSize; });
  $('#delSel').onclick=()=>{ if(selSet.size){pushUndo();entities=entities.filter(en=>!selSet.has(en));selSet.clear();updateSelInfo();render();fireChange();} };
  /* ---- 线型：把选中的线条在虚线 / 实线间切换（段长/间隔可自定义） ---- */
  function isStrokable(e){ return ['line','rect','circle','arc','polyline','spline','ellipse'].includes(e.type); }
  function readDashInputs(){
    newDashLen=Math.max(0.5,parseFloat($('#dashLenIn').value)||5);
    newDashGap=Math.max(0.5,parseFloat($('#dashGapIn').value)||3);
    $('#dashLenIn').value=newDashLen; $('#dashGapIn').value=newDashGap;
  }
  function applyDash(on,useInputs){
    if(!selSet.size){ setStatus('⚠️ 请先选中要设置线型的线条（选择 / 框选）','err'); return; }
    if(useInputs) readDashInputs();
    pushUndo();
    let cnt=0;
    selSet.forEach(en=>{ if(isStrokable(en)){ en.dash=on; if(on){en.dashLen=newDashLen; en.dashGap=newDashGap;} cnt++; } });
    updateSelInfo(); render(); fireChange();
    setStatus(on?('✅ 已将 '+cnt+' 条选中线条设为虚线（段长 '+newDashLen+' / 间隔 '+newDashGap+' mm）')
                :('✅ 已将 '+cnt+' 条选中线条设为实线'),'ok');
  }
  $('#dashSolid').onclick=()=>applyDash(false);
  $('#dashDashed').onclick=()=>applyDash(true,true);
  $('#dashLenIn').addEventListener('change',()=>{
    readDashInputs();
    if(selSet.size){ pushUndo(); let cnt=0;
      selSet.forEach(en=>{ if(isStrokable(en)&&en.dash){ en.dashLen=newDashLen; en.dashGap=newDashGap; cnt++; } });
      updateSelInfo(); render(); fireChange();
      setStatus(cnt?('✅ 已更新 '+cnt+' 条选中虚线：段长 '+newDashLen+' / 间隔 '+newDashGap+' mm')
                   :('🆕 新虚线默认：段长 '+newDashLen+' / 间隔 '+newDashGap+' mm（选中虚线后修改会直接生效）'),'ok');
    } else setStatus('🆕 新虚线默认：段长 '+newDashLen+' / 间隔 '+newDashGap+' mm','ok');
  });
  $('#dashGapIn').addEventListener('change',()=>$('#dashLenIn').dispatchEvent(new Event('change')));
  /* ---- 箭头：给选中线条的起点/终点加箭头，大小可自定义 ---- */
  function readArrowSize(){ newArrowSize=Math.max(1,parseFloat($('#arrSizeIn').value)||4); $('#arrSizeIn').value=newArrowSize; }
  function applyArrow(mode){
    if(selSet.size){
      readArrowSize(); pushUndo(); let cnt=0;
      selSet.forEach(en=>{ if(isArrowable(en)){ en.arrow=mode; en.arrowSize=newArrowSize; cnt++; } });
      updateSelInfo(); render(); fireChange();
      if(cnt) setStatus('✅ 已为 '+cnt+' 条选中线条'+['去除','加起点','加终点','加两端'][mode]+'箭头（大小 '+newArrowSize+' mm）','ok');
      else setStatus('⚠️ 选中对象里没有可加箭头的线条（只有直线 / 开多段线 / 开样条可以）','err');
    } else {
      newArrow=mode;
      const names=['无箭头','起点箭头','终点箭头','两端箭头'];
      setStatus('🆕 之后绘制的线条默认'+names[mode]+'（选中线条后点击按钮可直接修改）','ok');
    }
  }
  $('#arrNone').onclick=()=>applyArrow(0);
  $('#arrStart').onclick=()=>applyArrow(1);
  $('#arrEnd').onclick=()=>applyArrow(2);
  $('#arrBoth').onclick=()=>applyArrow(3);
  $('#arrSizeIn').addEventListener('change',()=>{
    readArrowSize();
    if(selSet.size){ pushUndo(); let cnt=0;
      selSet.forEach(en=>{ if(isArrowable(en)&&en.arrow){ en.arrowSize=newArrowSize; cnt++; } });
      updateSelInfo(); render(); fireChange();
      if(cnt) setStatus('✅ 已更新 '+cnt+' 条选中线条的箭头大小：'+newArrowSize+' mm','ok');
    } else setStatus('🆕 新线箭头默认大小：'+newArrowSize+' mm','ok');
  });
  $('#newDash').onchange=()=>{ newDash = $('#newDash').checked; if(newDash){ readDashInputs(); setStatus('🆕 之后绘制的线条默认为虚线（段长 '+newDashLen+' / 间隔 '+newDashGap+' mm；选中后可再切回实线）','ok'); } };

  /* ---- 图层：把选中的图元整体移到指定图层 ---- */
  function applyLayer(n){
    n = (n|0) || 1;
    if(!selSet.size){ setStatus('⚠️ 请先选中要移动的线条（选择 / 框选）','err'); return; }
    pushUndo();
    let cnt=0;
    selSet.forEach(en=>{ en.layer=String(n); cnt++; });
    updateSelInfo(); render(); fireChange();
    setStatus('✅ 已将 '+cnt+' 个图元移至图层 '+n,'ok');
  }
  $('#moveLayerBtn').onclick=()=>applyLayer(parseInt($('#moveLayerSel').value,10));
  $('#undoBtn').onclick=()=>undo();
  $('#redoBtn').onclick=()=>redo();
  updateUndoRedoBtns();
  updateSelInfo();
  document.addEventListener('keydown',e=>{
    const ae=document.activeElement;
    if(ae&&(ae.tagName==='INPUT'||ae.tagName==='TEXTAREA')) return;   // 在输入框里不拦截快捷键
    if(e.key==='Escape'){ if(draw||drag){ draw=null; drag=null; gesturePre=null; render(); } return; }
    const k=e.key.toLowerCase();
    if((e.ctrlKey||e.metaKey)&&!e.shiftKey&&k==='z'){ e.preventDefault(); undo(); }
    else if((e.ctrlKey||e.metaKey)&&(k==='y'||(e.shiftKey&&k==='z'))){ e.preventDefault(); redo(); }
  });
  $('#fitView').onclick=()=>{ fitView(); render(); };
  /* ---- 旋转：把选中图元旋到「与水平夹角 = 填写值」（0°=水平，逆时针为正）---- */
  $('#applyRot').onclick=()=>{
    if(selSet.size===0){ alert('请先用「旋转」或「选择」工具选中要旋转的图元（可 Shift 多选 / 框选）'); return; }
    const deg=parseFloat(rotAngle.value);
    if(!isFinite(deg)){ alert('请输入有效的角度（与水平夹角，度）'); return; }
    const cur=selAngleDeg(selSet);
    let delta;
    if(cur===null){
      if(!confirm('选中的图元（圆/弧/文本）没有可判定的方向，将按【相对角度】旋转 '+deg+'°。继续？')) return;
      delta=deg;
    }else{
      delta=deg-cur;
    }
    pushUndo();
    const c=combinedCentroid(selSet);
    const a=delta*Math.PI/180;
    selSet.forEach(en=>{ Object.assign(en, rotateEntity(en, c, a)); });
    render(); fireChange(); updateSelInfo();
    const now=selAngleDeg(selSet);
    if(now!==null) setStatus('已旋转 '+delta.toFixed(1)+'° → 与水平夹角 '+now.toFixed(1)+'°','ok');
    else setStatus('已按相对角度旋转 '+delta.toFixed(1)+'°','ok');
  };
  /* ---- 全屏 / 退出全屏（页内最大化） ---- */
  $('#fsBtn').onclick=()=>{
    const fs=editCard.classList.toggle('editor-fs');
    $('#fsBtn').textContent = fs ? '⤡ 退出全屏' : '⛶ 全屏';
    syncFsCanvas();
  };
  // 把画布后端分辨率对齐到它的显示尺寸，避免拉伸模糊；refit=true 时重新适配视图
  function resizeCanvasToDisplay(refit){
    const w=Math.max(320,Math.round(canvas.clientWidth||860));
    const h=Math.max(240,Math.round(canvas.clientHeight||520));
    const changed=(canvas.width!==w||canvas.height!==h);
    if(changed){ canvas.width=w; canvas.height=h; }
    if(refit||changed) render();
    if(refit) fitView();
  }
  function syncFsCanvas(){ requestAnimationFrame(()=>resizeCanvasToDisplay(true)); }
  window.addEventListener('resize',()=>{ resizeCanvasToDisplay(false); });
  $('#clearAll').onclick=()=>{ if(confirm('清空当前画布上的所有线条？')){pushUndo();entities=[];selSet.clear();updateSelInfo();render();fireChange();} };
  // 导出画布：fmt='dxf' 只出 DXF，'dwg' 额外转一份 DWG（转失败 DXF 照常落地）
  async function exportCanvas(fmt, btn){
    if(!entities.length){ alert('画布为空，先生成或绘制一些线条吧'); return; }
    if(btn) btn.disabled=true;
    try{
      const resp=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({entities:entities,name:'editor_drawing',fmt:fmt})});
      const d=await resp.json();
      if(d.ok){
        openLink2.href='/open?file='+encodeURIComponent(d.dxf);
        openRow2.style.display='block';
        const dwgNote = showDwgResult(d, openDwg2, openRow2);
        setStatus('✅ 已导出：'+(d.main||d.dxf)+'（'+d.entities+' 个图元）'+dwgNote,'ok');
      }
      else setStatus('❌ 导出失败：'+(d.error||''),'err');
    }catch(err){ setStatus('❌ 无法连接本地服务：'+err.message,'err'); }
    finally{ if(btn) btn.disabled=false; }
  }
  $('#exportDxf').onclick=()=>exportCanvas('dxf', $('#exportDxf'));
  $('#exportDwg').onclick=()=>exportCanvas('dwg', $('#exportDwg'));
  /* ---- 文本工具：在尺寸栏输入文字，点击画布放置（所见即所得） ---- */
  function placeText(w){
    const v = (dimTextStr.value || '').trim();
    if(!v){ setStatus('⚠️ 请先在尺寸栏的「文字」框里输入内容，再点画布放置', 'err'); return; }
    const h = parseFloat(dimTextH.value) || 5;
    entities.push({
      type:'text',
      p:[Math.round(w[0]*1000)/1000, Math.round(w[1]*1000)/1000],
      height:h, value:v, rotation:0, layer:String(activeLayer)
    });
    render(); fireChange();
  }

  function fireChange(){ if(onChangeCb) onChangeCb(); }
  function setEntities(list){ pushUndo(); entities=(list||[]).map(e=>({...e})); selSet.clear(); updateSelInfo(); fitView(); render(); fireChange(); }
  function appendEntities(list){
    pushUndo();
    const wasEmpty = entities.length === 0;
    (list||[]).forEach(e=>{ const c={...e}; c.layer=String(activeLayer); entities.push(c); });
    if(wasEmpty && entities.length) fitView();
    selSet.clear(); updateSelInfo(); render(); fireChange();
  }
  function getEntities(){ return entities.map(e=>({...e})); }
  function setActiveLayer(n){ activeLayer = n|0 || 1; }
  function getActiveLayer(){ return activeLayer; }
  function setOnChange(cb){ onChangeCb = cb; }
  function clearEntities(){ pushUndo(); entities=[]; selSet.clear(); updateSelInfo(); render(); fireChange(); }
  function show(noScroll){ editCard.style.display='block'; if(!noScroll) editCard.scrollIntoView({behavior:'smooth',block:'nearest'}); requestAnimationFrame(()=>resizeCanvasToDisplay(true)); }
  return {setEntities, appendEntities, getEntities, setActiveLayer,
          getActiveLayer, setOnChange, clearEntities, show};
})();

function showInEditor(list){ if(list&&list.length){ Editor.appendEntities(list); Editor.show(); revealLayerPanel(); } }

/* ================= 图层面板 + 合成 ================= */
let maxLayer = 1;
const layerCard=$('#layerCard'), layerChips=$('#layerChips'), layerHint=$('#layerHint');
function renderLayerChips(){
  const counts={};
  Editor.getEntities().forEach(e=>{ const k=String(e.layer||'0'); counts[k]=(counts[k]||0)+1; });
  let html='';
  for(let n=1;n<=maxLayer;n++){
    const c=counts[String(n)]||0;
    html += '<span class="chip '+(n===Editor.getActiveLayer()?'sel':'')+'" data-layer="'+n+'">'
          + '<span class="dot" style="background:'+layerColorCss(n)+'"></span>图层 '+n
          + '<span class="cnt">'+c+'</span></span>';
  }
  layerChips.innerHTML = html;
  // 同步「移至图层」下拉框选项（工具栏：把选中线条移到指定图层）
  const sel=$('#moveLayerSel');
  if(sel){
    const cur=parseInt(sel.value,10)||Editor.getActiveLayer();
    let opts='';
    for(let n=1;n<=maxLayer;n++)
      opts += '<option value="'+n+'"'+(n===cur?' selected':'')+'>图层 '+n+'</option>';
    sel.innerHTML=opts;
  }
}
function revealLayerPanel(){
  layerCard.style.display='block';
  renderLayerChips();
}
layerChips.addEventListener('click', ev=>{
  const chip=ev.target.closest('.chip'); if(!chip) return;
  const n=parseInt(chip.dataset.layer,10); if(!n) return;
  Editor.setActiveLayer(n); renderLayerChips();
  layerHint.textContent='已选图层 '+n+'：接下来生成 / 绘制的图元会归入此层';
});
$('#addLayer').onclick=()=>{ maxLayer++; Editor.setActiveLayer(maxLayer); renderLayerChips(); layerHint.textContent='已新建并选中图层 '+maxLayer; };
$('#resetLayers').onclick=()=>{
  if(!confirm('重置图层会清空画布上所有线条与文本，确定？')) return;
  Editor.clearEntities(); maxLayer=1; Editor.setActiveLayer(1); renderLayerChips(); layerHint.textContent='';
};
/* ---- 合成弹窗 ---- */
$('#combineBtn').onclick=()=>{
  const ents=Editor.getEntities();
  if(!ents.length){ alert('还没有任何图元，先生成或绘制一些内容'); return; }
  const counts={}; ents.forEach(e=>{ const k=String(e.layer||'0'); counts[k]=(counts[k]||0)+1; });
  const nums=Object.keys(counts).map(Number).filter(n=>n>=1).sort((a,b)=>a-b);
  if(!nums.length){ alert('当前内容未分配到任何图层（请先用图层 1 及以上生成 / 绘制）'); return; }
  let html='';
  nums.forEach(n=>{ html += '<label class="selitem" style="cursor:pointer">'
    + '<input type="checkbox" class="cl" value="'+n+'" checked> 图层 '+n+'（'+(counts[String(n)])+'）</label>'; });
  $('#combineLayers').innerHTML=html;
  $('#combinePanel').style.display='block';
};
$('#combineFlatten').onchange=()=>{ if($('#combineFlatten').checked) $('#combineKeep').checked=false; };
$('#combineKeep').onchange=()=>{ if($('#combineKeep').checked) $('#combineFlatten').checked=false; };
$('#combineCancel').onclick=()=>{ $('#combinePanel').style.display='none'; };
$('#combineGo').onclick=async()=>{
  const checked=[...document.querySelectorAll('#combineLayers .cl')].filter(c=>c.checked).map(c=>c.value);
  if(!checked.length){ alert('至少勾选一个图层'); return; }
  const flatten = $('#combineFlatten').checked && !$('#combineKeep').checked;
  const ents=Editor.getEntities().map(e=>({...e}));
  const name=($('#combineName').value||'合并图').slice(0,60);
  $('#combineGo').disabled=true;
  try{
    const resp=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({entities:ents, name:name, flatten:flatten, layers:checked,
                            fmt:$('#combineDwg').checked?'both':'dxf'})});
    const d=await resp.json();
    if(d.ok){ $('#openLink2').href='/open?file='+encodeURIComponent(d.dxf); $('#openRow2').style.display='block';
      const dwgNote = showDwgResult(d, openDwg2, openRow2);
      setStatus('✅ 已合成并导出：'+d.dxf+'（'+d.entities+' 个图元）'+(flatten?'｜已合并为单层 0':'｜保留分层')+dwgNote,'ok');
      $('#combinePanel').style.display='none'; }
    else setStatus('❌ 合成失败：'+(d.error||''),'err');
  }catch(err){ setStatus('❌ 无法连接本地服务：'+err.message,'err'); }
  finally{ $('#combineGo').disabled=false; }
};
// 编辑器图元变化时自动刷新图层计数
Editor.setOnChange(renderLayerChips);

/* 画布与图层面板常驻显示：进入页面即可直接画图 / 导入 DXF，
   不再要等生成或导出过 DXF/DWG 之后才出现（旧行为是生成回调里才 show）。 */
(function revealOnLoad(){ Editor.show(true); revealLayerPanel(); })();

/* Homotope 引擎自检：不可用就把下拉禁用并说明原因（缺 Node 最常见） */
(async function homotopeProbe(){
  try{
    if(!htSel) return;
    const r = await (await fetch('/api/homotope_status', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})).json();
    if(r && r.available){
      if(htHint) htHint.textContent = '（引擎就绪：' + (r.preset||'centerline') + '）';
    }else{
      if(htHint) htHint.textContent = '（不可用：' + (r && r.reason || '未安装') + '）';
      htSel.value = 'off';
      htSel.disabled = true;
    }
  }catch(e){ /* 探针失败不影响主流程 */ }
})();

/* 运行环境自检：解释器缺 numpy/cv2 时，在页面顶部给一条红色横幅。
   旧行为只在启动控制台里打印一行，用户根本看不到，直到点「生成/导入」才莫名失败。 */
(async function envProbe(){
  try{
    const bar = document.getElementById('envBanner');
    if(!bar) return;
    const r = await (await fetch('/api/env', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:'{}'})).json();
    const miss = (r && r.missing) || [];
    if(!miss.length) return;
    const box = document.createElement('div');
    box.innerHTML = '<b>⚠ 图片识别不可用</b>：当前 Python 解释器缺少 <code>'
      + miss.join('</code> <code>') + '</code>。'
      + '<br>正在使用：<code>' + (r.python || '未知') + '</code>'
      + '<br>解决办法：关闭本页，改用包内 <b>启动 Ghoul.bat</b> 启动（会自动使用自带 Python）；'
      + '或在命令行执行 <code>python -m pip install numpy opencv-python</code>';
    const x = document.createElement('button');
    x.className = 'x'; x.type = 'button'; x.textContent = '✕'; x.title = '关闭';
    x.onclick = function(){ bar.style.display = 'none'; };
    bar.appendChild(box);
    bar.appendChild(x);
    bar.style.display = 'flex';
  }catch(e){ /* 自检失败不影响主流程 */ }
})();

/* ================= 图层处：导入已有 DXF ================= */
let pendingImport = null;
$('#importDxf').onclick=()=>{ $('#importDxfFile').click(); };
$('#importDxfFile').onchange=async ev=>{
  const f=ev.target.files && ev.target.files[0];
  if(!f){ return; }
  if(!/[.]dxf$/i.test(f.name)){ alert('请选择 .dxf 文件'); ev.target.value=''; return; }
  $('#importInfo').textContent='正在解析 '+f.name+' …';
  $('#importPanel').style.display='block';
  try{
    const resp=await fetch('/api/import_dxf',{method:'POST',body:f});
    const j=await resp.json();
    if(!j.ok){ alert('导入失败：'+(j.error||'未知错误')); $('#importPanel').style.display='none'; pendingImport=null; return; }
    pendingImport=j.entities;
    $('#importInfo').textContent='共解析出 '+j.count+' 个图元 — 选择要导入到的图层（默认当前图层 '+Editor.getActiveLayer()+'）：';
    $('#importLayer').value=String(Editor.getActiveLayer());
  }catch(err){ alert('导入出错：'+err.message); $('#importPanel').style.display='none'; pendingImport=null; }
  ev.target.value='';
};
$('#importGo').onclick=()=>{
  if(!pendingImport){ return; }
  const ln=Math.max(1, parseInt($('#importLayer').value,10)||1);
  while(maxLayer<ln){ maxLayer++; }
  pendingImport.forEach(e=>{ e.layer=String(ln); });
  Editor.appendEntities(pendingImport);
  Editor.setActiveLayer(ln);
  Editor.show();
  renderLayerChips();
  layerHint.textContent='已从 DXF 导入 '+pendingImport.length+' 个图元到图层 '+ln;
  pendingImport=null;
  $('#importPanel').style.display='none';
};
$('#importCancel').onclick=()=>{ pendingImport=null; $('#importPanel').style.display='none'; };

addMsg('bot', '你好！我是 J022 Ghoul——完全离线的本地 CAD 助手。\\n可以问我制图知识、DXF 格式、几何计算，或让我帮你写绘图指令（写好复制到右侧执行）。');
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _read_raw(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        p = urlparse(self.path).path
        if p.startswith("/assets/"):
            self._serve_asset(p)
        elif self.path.startswith("/open"):
            self._do_open()
        else:
            self._serve_html()

    _ASSETS = {"joZZbot.png": "image/png", "chibi.png": "image/png",
               "joZZbot.ico": "image/x-icon"}

    def _serve_asset(self, path):
        name = os.path.basename(path)
        ctype = self._ASSETS.get(name)
        if not ctype:
            self._json({"ok": False, "error": "未知资源"}, code=404)
            return
        f = os.path.join(ASSETS_DIR, name)
        if not os.path.exists(f):
            self._json({"ok": False, "error": "资源不存在"}, code=404)
            return
        with open(f, "rb") as fp:
            body = fp.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _do_open(self):
        q = parse_qs(urlparse(self.path).query)
        f = q.get("file", [""])[0]
        if f and os.path.exists(f):
            try:
                if f.lower().endswith(".dwg"):
                    # 便携版 LibreCAD 读不了 DWG（会弹"DWG 支持尚未完善"），
                    # 所以给它一个"在文件夹中定位"，用户拿去别的 CAD 打开。
                    subprocess.Popen(["explorer", "/select,", os.path.normpath(f)])
                else:
                    ensure_librecad_text_visible()   # 别让草稿模式把文字画成方框
                    subprocess.Popen([LIBRECAD, f])
                self._json({"ok": True})
            except Exception as e:  # noqa
                self._json({"ok": False, "error": str(e)})
        else:
            self._json({"ok": False, "error": "文件不存在: " + f})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/import_dxf":
            self._api_import_dxf()
            return
        routes = {
            "/api/generate": self._api_generate,
            "/api/chat": self._api_chat,
            "/api/extract": self._api_extract,
            "/api/build": self._api_build,
            "/api/save": self._api_save,
            "/api/ollama/repair": self._api_ollama_repair,
            "/api/ollama/status": self._api_ollama_status,
            "/api/dwg_status": self._api_dwg_status,
            "/api/env": self._api_env,
            "/api/homotope_status": self._api_homotope_status,
        }
        fn = routes.get(path)
        if fn is None:
            self.send_error(404)
            return
        try:
            data = self._read_json()
        except Exception as e:  # noqa
            self._json({"ok": False, "error": "请求体解析失败: %s" % e})
            return
        try:
            fn(data)
        except Exception as e:  # noqa
            self._json({"ok": False, "error": str(e)})

    # ------------------------------------------------ /api/homotope_status
    def _api_homotope_status(self, data: dict):
        """Homotope 引擎可用性（页面加载时查一次，用来提示要不要装 Node）。"""
        try:
            import homotope_bridge as _hb
            ok = _hb.available()
            self._json({"ok": True, "available": ok,
                        "node": _hb.find_node(),
                        "preset": _hb.DEFAULT_PRESET,
                        "reason": "" if ok else _hb.reason_unavailable()})
        except Exception as e:                            # noqa: BLE001
            self._json({"ok": True, "available": False, "reason": str(e)[:120]})

    # ------------------------------------------------ /api/generate（原有行为）
    def _api_generate(self, data: dict):
        instruction = (data.get("instruction") or "").strip()
        img_b64 = data.get("image_base64") or ""
        mode = (data.get("mode") or "auto").strip()
        want_ocr = bool(data.get("ocr_text"))
        fmt = (data.get("fmt") or "dxf").strip().lower()
        if fmt not in ("dxf", "dwg", "both"):
            fmt = "dxf"
        if mode not in ("auto", "lines", "vlm"):
            mode = "auto"
        if not instruction and not img_b64:
            self._json({"ok": False,
                        "error": "请输入画图指令，或上传一张参考图片"})
            return
        if not _model_slot():
            self._json({"ok": False,
                        "error": "本地模型正忙（另一个生成/识图任务还在跑），等它结束再试一次~"})
            return
        try:
            self._generate_locked(instruction, img_b64, mode, want_ocr, fmt,
                                  multipass=bool(data.get("multipass", True)),
                                  bw_first=bool(data.get("bw_first", True)),
                                  homotope=str(data.get("homotope", "auto") or "auto"))
        finally:
            _MODEL_LOCK.release()

    def _generate_locked(self, instruction: str, img_b64: str, mode: str,
                         want_ocr: bool, fmt: str = "dxf", multipass: bool = True,
                         bw_first: bool = True, homotope: str = "auto"):
        img_path = None
        tmp = None
        if img_b64:
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            blob = base64.b64decode(img_b64)
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", dir=CAD_DIR)
            tmp.write(blob)
            tmp.close()
            img_path = tmp.name

        try:
            out = cad_agent.generate_dxf_detailed(
                instruction,
                cad_agent.DEFAULT_MODEL,
                cad_agent.OLLAMA_HOST,
                CAD_DIR,
                open_libre=True,   # 生成后用 LibreCAD 自动打开
                image=img_path,
                mode=mode,
                ocr_text=want_ocr,
                fmt=fmt,
                multipass=multipass,
                bw_first=bw_first,
                homotope=homotope,
            )
            debug_image = None
            if out.get("debug_path") and os.path.exists(out["debug_path"]):
                try:
                    with open(out["debug_path"], "rb") as fp:
                        debug_image = ("data:image/png;base64,"
                                       + base64.b64encode(fp.read()).decode())
                except OSError:
                    pass
            text_count = sum(1 for e in (out.get("entities_list") or [])
                             if str(e.get("type")) == "text")
            self._json({"ok": True, "dxf": out["path"],
                        "main": out.get("main") or out["path"],
                        "dwg": out.get("dwg"),
                        "dwg_error": out.get("dwg_error"),
                        "entities": out["entities"],
                        "entities_list": out.get("entities_list", []),
                        "method": out["method"],
                        "text_count": text_count,
                        "counts_text": out.get("message", ""),
                        "debug_image": debug_image})
        finally:
            if tmp and os.path.exists(tmp.name):
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass

    # ------------------------------------------------ /api/chat（左栏对话）
    def _api_chat(self, data: dict):
        msgs_in = data.get("messages") or []
        clean = []
        for m in msgs_in:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = str(m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                clean.append({"role": role, "content": content})
        if not clean or clean[-1]["role"] != "user":
            self._json({"ok": False, "error": "请输入对话内容"})
            return
        clean = clean[-16:]
        # ---- 图片（看图问答 / OCR）----
        img_b64 = (data.get("image_base64") or "").strip()
        if img_b64 and "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]   # 去掉 data:image/...;base64, 前缀
        has_img = bool(img_b64)
        if has_img:
            model = cad_agent.VISION_MODEL          # qwen2.5vl:7b 视觉大模型
            sys_prompt = CHAT_VISION_SYSTEM_PROMPT
        else:
            model = CHAT_MODEL
            sys_prompt = CHAT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": sys_prompt}] + clean
        # 有图时把图片挂到最后一个 user 消息上（Ollama /api/chat 的 images 字段）
        if has_img:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["images"] = [img_b64]
                    break
        # 流式请求 + 三道时限保护（首字/停滞/总时长），避免慢模型把浏览器请求
        # 挂死（旧实现同步阻塞最长 300 秒）。复用 ollama_stream 的健壮读取。
        if not _model_slot():
            self._json({"ok": False,
                        "error": "本地模型正忙（另一个生成/识图任务还在跑），等它结束再试一次~"})
            return
        try:
            # 模型已在显存 → 首字通常 1~2 秒，收紧时限早报错；冷加载才放宽。
            loaded = ollama_stream.is_loaded(cad_agent.OLLAMA_HOST, model)
            first = 70.0 if loaded else 200.0
            text, err = stream_ollama_chat(
                messages, model=model, host=cad_agent.OLLAMA_HOST,
                temperature=0.7, num_predict=1024,
                first_token=first, stall=35.0, total=240.0,
                keep_alive=ollama_stream.DEFAULT_KEEP_ALIVE)
        finally:
            _MODEL_LOCK.release()
        if err:
            self._json({"ok": False, "error": err})
            return
        # 清洗：LaTeX → 纯文本；emoji → 颜文字（助手说话不用表情符号）
        reply = demojify(sanitize_math(text))
        if not reply:
            self._json({"ok": False, "error": "模型返回了空回复，请重试"})
            return
        self._json({"ok": True, "reply": reply, "model": model, "vision": has_img})

    # ------------------------------------------------ /api/extract（识别+编号预览）
    def _api_extract(self, data: dict):
        instruction = (data.get("instruction") or "").strip()
        img_b64 = data.get("image_base64") or ""
        want_ocr = bool(data.get("ocr_text"))
        if not img_b64:
            self._json({"ok": False, "error": "线段勾选流程需要上传图片",
                        "hint": "纯文本指令请直接点生成（无需勾选）"})
            return
        if not line_extractor.available():
            self._json({"ok": False, "error": "本机缺少 OpenCV，无法使用经典视觉识别"})
            return
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        img = line_extractor.load_image_b64(img_b64)
        if img is None:
            self._json({"ok": False, "error": "图片解码失败，请换一张 PNG/JPG"})
            return

        bw_path = None
        if data.get("bw_first", True):
            # 与 /api/generate 的 CV 路线同规则：有色过滤指令时**不能**先黑白化。
            # bw_photo 会把白线/青线/绿线一律压成黑线，之后再按颜色过滤必然错位
            # （暗底白线图上"只留白"→ 整片背景被判成目标色 → 0 图元）。
            try:
                from line_extractor import parse_color_filter as _pcf
                _kc, _dc = _pcf(str(data.get("instruction") or ""))
                if _kc or _dc:
                    data["bw_first"] = False
            except Exception:
                pass
        if data.get("bw_first", True):
            # 先把图压成高对比纯黑白（白底黑线）再识别——与 /api/generate 的 CV 路线一致
            try:
                bwimg = line_extractor.bw_photo(img)
                if bwimg is not None:
                    img = bwimg
                try:
                    import cv2
                    dbg_dir = os.path.join(CAD_DIR, "_debug")
                    os.makedirs(dbg_dir, exist_ok=True)
                    bw_path = os.path.join(dbg_dir, "bw_%d.png" % (time.time() * 1000))
                    okw, buf = cv2.imencode(".png", bwimg)
                    if okw:
                        buf.tofile(bw_path)
                    else:
                        bw_path = None
                except Exception:
                    bw_path = None
            except Exception:
                bw_path = None

        spec = line_extractor.extract_spec(
            img, instruction=instruction,
            passes=None if data.get("multipass", True) else 1)
        meta = spec.get("_meta") or {}
        ents = spec.get("entities") or []
        # ---- Homotope 辅助：CV 失败 / 结果很弱（碎直线一堆）时接手 ----
        ht_mode = str(data.get("homotope", "auto") or "auto").strip().lower()
        if ht_mode != "off" and (not meta.get("ok") or not ents
                                 or cad_agent._spec_is_weak(spec)):
            # 引擎吃文件路径：把内存里的图落一份临时 PNG
            ht_path = None
            try:
                okh, hbuf = cv2.imencode(".png", img)
                if okh:
                    ht_path = os.path.join(CAD_DIR, "_ht_%d.png" % (time.time() * 1000))
                    hbuf.tofile(ht_path)
                    hspec, herr = cad_agent._homotope_spec_from_image(
                        ht_path, instruction, "centerline")
                    if hspec and len(hspec.get("entities") or ()) > len(ents):
                        spec, meta, ents = hspec, hspec.get("_meta") or {}, \
                            hspec.get("entities") or []
                        meta["homotope_used"] = True
            except Exception:                            # noqa: BLE001
                pass
            finally:
                if ht_path and os.path.exists(ht_path):
                    try:
                        os.remove(ht_path)
                    except Exception:                    # noqa: BLE001
                        pass
        if not meta.get("ok") or not ents:
            # 几何太少时，若用户勾了识别文字，仍尝试只出文字
            if not (want_ocr and ocr_text.available()):
                if meta.get("reason") == "not-a-line-drawing":
                    self._json({"ok": False,
                                "error": "这张图不像线稿（墨迹占比 %s%%，接近照片/纹理/噪点）"
                                         % round(float(meta.get("ink_ratio") or 0) * 100),
                                "hint": "线条识别需要白底黑线或黑底亮线的清晰线稿；"
                                        "这类图片请切换到 🧠AI 理解 模式"})
                    return
                self._json({"ok": False,
                            "error": "经典视觉未识别出足够的几何（%s）" %
                                     json.dumps(meta.get("counts") or {}, ensure_ascii=False),
                            "hint": "可切换到 🧠AI 理解 模式直接生成，"
                                    "或提高图片对比度/线条清晰度后重试"})
                return

        # ---- 文字识别（可选）：定位来自 OpenCV，内容来自视觉模型 ----
        ocr_note = ""
        if want_ocr:
            if not _model_slot():
                self._json({"ok": False,
                            "error": "本地模型正忙（另一个生成/识图任务还在跑），等它结束再试一次~"})
                return
            try:
                oinfo = ocr_text.enrich_spec_with_text(
                    spec, img, model=cad_agent.VISION_MODEL,
                    host=cad_agent.OLLAMA_HOST)
            finally:
                _MODEL_LOCK.release()
            ocr_note = ocr_text.describe(oinfo)
            meta = spec.get("_meta") or {}
            ents = spec.get("entities") or []
            if not ents:
                self._json({"ok": False,
                            "error": "既没识别出几何，也没读出文字",
                            "hint": ocr_note})
                return

        jid = _job_store(spec)
        ent_list = [{"id": i, "type": str(e.get("type")),
                     "type_name": _TYPE_NAME.get(str(e.get("type")), str(e.get("type"))),
                     "desc": _entity_desc(e)}
                    for i, e in enumerate(ents)]
        preview = _annotate_preview(img, spec)
        text_count = sum(1 for e in ents if str(e.get("type")) == "text")
        self._json({"ok": True, "job_id": jid, "total": len(ents),
                    "entities": ent_list, "preview": preview,
                    "text_count": text_count, "ocr_note": ocr_note,
                    "color_filter": meta.get("color_filter"),
                    "passes": (meta.get("passes") or {}).get("passes"),
                    "counts": meta.get("counts"), "bw": bw_path})

    # ------------------------------------------------ /api/build（按勾选生成）
    def _api_build(self, data: dict):
        jid = str(data.get("job_id") or "")
        keep_ids = data.get("keep_ids")
        fmt = str(data.get("fmt") or "dxf").strip().lower()
        if fmt not in ("dxf", "dwg", "both"):
            fmt = "dxf"
        with _JOBS_LOCK:
            job = JOBS.get(jid)
        if not job:
            self._json({"ok": False,
                        "error": "识别结果已过期或不存在，请重新上传图片识别"})
            return
        if not isinstance(keep_ids, list):
            self._json({"ok": False, "error": "keep_ids 参数缺失"})
            return
        keep = set()
        for v in keep_ids:
            try:
                keep.add(int(v))
            except (TypeError, ValueError):
                continue
        src = job["spec"]
        ents = src.get("entities") or []
        if not keep:
            self._json({"ok": False, "error": "至少保留一个图元"})
            return
        filtered = [e for i, e in enumerate(ents) if i in keep]
        counts = {"line": 0, "rect": 0, "circle": 0, "polyline": 0, "text": 0,
                  "spline": 0, "ellipse": 0, "arc": 0}
        for e in filtered:
            t = str(e.get("type"))
            if t in counts:
                counts[t] += 1
        meta = dict(src.get("_meta") or {})
        meta["counts"] = counts
        spec2 = {"name": src.get("name", "sketch"), "unit": src.get("unit", "mm"),
                 "entities": filtered, "_meta": meta}
        note = "已按勾选保留 %d/%d 个图元；" % (len(filtered), len(ents))
        out = cad_agent._build_dxf_from_spec(
            spec2, CAD_DIR, open_libre=True, method="cv-lines", note=note,
            fmt=fmt)
        self._json({"ok": True, "dxf": out["path"],
                    "main": out.get("main") or out["path"],
                    "dwg": out.get("dwg"), "dwg_error": out.get("dwg_error"),
                    "entities": out["entities"],
                    "entities_list": filtered,
                    "message": out.get("message", "")})

    # ------------------------------------------------ /api/save（编辑器/合成导出）
    def _api_save(self, data: dict):
        ents = data.get("entities")
        if not isinstance(ents, list) or not ents:
            self._json({"ok": False, "error": "没有可保存的图元"})
            return
        # 仅保留受支持的图元类型（含 text 文本）
        ALLOWED = ("line", "rect", "circle", "polyline", "text", "arc", "point",
                   "spline", "ellipse")
        clean = []
        for e in ents:
            if not isinstance(e, dict):
                continue
            t = str(e.get("type"))
            if t in ALLOWED:
                clean.append(e)
        if not clean:
            self._json({"ok": False, "error": "图元格式不受支持"})
            return
        # 按图层筛选（合成弹窗里勾选了部分图层时）
        layers = data.get("layers")
        if isinstance(layers, list) and layers:
            want = set(str(x) for x in layers)
            clean = [e for e in clean if str(e.get("layer", "0")) in want]
        if not clean:
            self._json({"ok": False, "error": "所选图层内没有可保存的图元"})
            return
        # 合并为单层（"合成进一个图层"）时把全部实体归到 0 层
        flatten = bool(data.get("flatten"))
        name = str(data.get("name") or "editor_drawing")[:60]
        fmt = str(data.get("fmt") or "dxf").strip().lower()
        if fmt not in ("dxf", "dwg", "both"):
            fmt = "dxf"
        spec = {"name": name, "unit": "mm", "entities": clean}
        try:
            out = cad_agent._build_dxf_from_spec(
                spec, CAD_DIR, open_libre=True, method="editor",
                flatten_layer="0" if flatten else None, fmt=fmt)
        except Exception as e:  # noqa
            self._json({"ok": False, "error": str(e)})
            return
        self._json({"ok": True, "dxf": out["path"],
                    "main": out.get("main") or out["path"],
                    "dwg": out.get("dwg"), "dwg_error": out.get("dwg_error"),
                    "entities": out["entities"]})

    # ------------------------------------------------ /api/env（运行环境自检）
    def _api_env(self, data: dict):
        """回报「到底是谁在跑这个服务」——排查「未安装 ezdxf」这类幽灵问题用。

        典型坑：用系统自带的干净 Python 启动时界面正常，只有生成/导入才失败；
        这里能一眼看到解释器路径与缺失的库。
        """
        info = {"python": sys.executable, "version": sys.version.split()[0],
                "deps": {}, "bundled_python": None, "relaunched": False}
        try:
            import runtime_guard
            info.update(runtime_guard.describe())
        except Exception:                              # noqa: BLE001
            pass
        for m in ("numpy", "cv2", "ezdxf"):
            try:
                mod = __import__(m)
                info["deps"][m] = str(getattr(mod, "__version__", "ok"))
            except Exception as e:                      # noqa: BLE001
                info["deps"][m] = "缺失（%s）" % type(e).__name__
        try:
            import dxf_reader  # noqa: F401 - 内置 DXF 读取器，永远可用
            info["dxf_reader"] = True
        except Exception:
            info["dxf_reader"] = False
        self._json(dict(info, ok=not info.get("missing")))


    # ------------------------------------------------ /api/dwg_status（DWG 可用性）
    def _api_dwg_status(self, data: dict):
        """告诉前端这台机器有没有 DWG 转换器（没有就把选项灰掉并说明原因）。"""
        try:
            import dwg_export
            st = dwg_export.status()
            ok = bool(st.get("ok"))
            self._json({"ok": ok,
                        "name": st.get("name"),
                        "versions": st.get("versions") or [],
                        "note": st.get("note") or ""})
        except Exception as e:                          # noqa: BLE001
            self._json({"ok": False, "name": None, "versions": [],
                        "note": "DWG 模块不可用：%s" % e})

    # ------------------------------------------------ /api/import_dxf（导入已有 DXF）
    def _api_import_dxf(self):
        """接收一个 DXF 文件的原始字节体，解析为编辑器图元后返回。"""
        raw = self._read_raw()
        if not raw:
            self._json({"ok": False, "error": "未收到文件内容"})
            return
        # 宽松校验：DXF 文本通常含 SECTION / 0 ... 行；二进制 DXF 以 AC 开头
        head = raw[:32]
        is_text = (b"SECTION" in raw[:400]) or (b"0\n" in raw[:200]) or (b"0\r\n" in raw[:200])
        is_binary = head.startswith(b"AutoCAD Binary DXF") or head[:2] == b"AC"
        if not (is_text or is_binary):
            self._json({"ok": False, "error": "文件看起来不是 DXF（缺少 SECTION/实体标记）"})
            return
        tmp = os.path.join(CAD_DIR, "_import_%s.dxf" % uuid.uuid4().hex)
        try:
            with open(tmp, "wb") as f:
                f.write(raw)
            ents = dxf_writer.parse_dxf_to_entities(tmp)
        except Exception as e:  # noqa
            self._json({"ok": False, "error": "DXF 解析失败: %s" % e})
            return
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if not ents:
            self._json({"ok": False, "error": "未能解析出可识别的图元（仅支持 LINE/LWPOLYLINE/POLYLINE/CIRCLE/ARC/TEXT/MTEXT）"})
            return
        self._json({"ok": True, "entities": ents, "count": len(ents)})

    # ------------------------------------------------ /api/ollama/*（模型自愈）
    def _api_ollama_status(self, data: dict):
        host = cad_agent.OLLAMA_HOST
        loaded = ollama_stream.ps_models(host)
        installed = ollama_stream.installed_models(host)
        base = cad_agent.VISION_MODEL.split(":")[0]
        vision_ready = any(m == cad_agent.VISION_MODEL or m.split(":")[0] == base
                           for m in installed)
        self._json({"ok": bool(installed) or bool(loaded), "host": host,
                    "loaded": loaded, "installed": installed,
                    "chat_model": CHAT_MODEL,
                    "vision_model": cad_agent.VISION_MODEL,
                    "vision_ready": vision_ready})

    def _api_ollama_repair(self, data: dict):
        """模型卡住时的一键自愈：先卸掉占着显存的模型，再预热文本模型。"""
        if not _model_slot(wait=1.0):
            self._json({"ok": False,
                        "error": "有任务正在使用模型，先等它结束（或重启程序）再来"})
            return
        try:
            warm = CHAT_MODEL if data.get("warm", True) else ""
            info = ollama_stream.repair(cad_agent.OLLAMA_HOST, warm=warm)
        finally:
            _MODEL_LOCK.release()
        self._json(dict(info, host=cad_agent.OLLAMA_HOST))

    def log_message(self, *args):
        pass  # 静默日志


def _count_entities(dxf_path: str) -> int:
    try:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        return sum(1 for _ in doc.modelspace())
    except Exception:
        return 0


def main():
    # 自检：如果当前解释器缺 numpy/cv2（网页能开、一点生成就失败），
    # 自动换成包内自带的 Python 重启一遍。没问题时这一步什么都不做。
    try:
        import runtime_guard
        runtime_guard.ensure()
    except SystemExit:
        raise
    except Exception:                                  # noqa: BLE001 - 自检失败不影响启动
        pass
    os.makedirs(CAD_DIR, exist_ok=True)
    _warmup_async()          # 后台预热文本模型（失败无妨，只是慢一点）
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"J022 Ghoul Web 界面已启动 →  http://{HOST}:{PORT}")
    print("（按 Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
