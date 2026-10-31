"""ocr_text.py — 从草图/工程图里读文字，并变成可编辑的文本图元。

分工（两层各做自己擅长的部分）：
  1) OpenCV 负责「位置」：形态学横向膨胀把字符连成行，再按尺寸 / 长宽比 /
     填充率过滤掉直线、矩形框、圆这类几何，得到像素级准确的文字行包围盒。
  2) 视觉大模型负责「内容」：在图上给每个包围盒画编号框，一次性问模型
     「每个编号框里是什么字」，返回 JSON。位置来自 CV、内容来自模型，
     模型再也不用瞎猜坐标（7B 视觉模型猜坐标本来就不可靠）。
  3) 用 spec._meta.frame 把像素框映射回毫米坐标，生成
     {"type":"text","p":[x,y],"height":h,"value":"..."} 图元，
     顺手删掉落在文字框内的碎线段，避免笔画被当成线条重复画一遍。

对外只有两个入口：
  available()                        —— 本机 OpenCV 是否可用
  enrich_spec_with_text(spec, img)   —— 就地给 spec 补文本，返回结果信息
两者都不抛异常（OCR 失败绝不能拖垮几何识别）。
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # OpenCV 是可选的：缺了就只是没有文字识别，几何识别照常
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    HAS_CV = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore
    np = None  # type: ignore
    HAS_CV = False

import ollama_stream


def available() -> bool:
    return HAS_CV


def load_image(path: str):
    """按路径读图（用 imdecode，兼容中文/空格路径）。"""
    if not HAS_CV or not path:
        return None
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


def describe(info: dict) -> str:
    """把 enrich 的结果说成一句话（给网页状态/控制台用）。"""
    if not isinstance(info, dict):
        return ""
    if info.get("ok"):
        n = info.get("count", 0)
        extra = "（同时剔除了 %d 根文字笔画碎线）" % info["dropped"] \
            if info.get("dropped") else ""
        return "🔤 已识别 %d 处文字并写入 CAD%s" % (n, extra)
    err = info.get("error") or ""
    return "（文字识别未生效：%s）" % err if err else ""


# --------------------------------------------------------------------------
# 1) 文字行包围盒检测
# --------------------------------------------------------------------------
def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def detect_text_boxes(img, max_boxes: int = 40) -> List[Tuple[int, int, int, int]]:
    """返回文字行包围盒 [(x, y, w, h), ...]（像素坐标，左上原点）。

    判据（都来自工程图的通常特征）：
      * 字高在图像对角线的 0.6%~10% 之间；
      * 长宽比不宜过极端（单字 ~0.6，一行字可以很宽）；
      * 框内「墨迹填充率」中等（0.06~0.85）：直线框填充率极低，
        实心色块接近 1，两者都被排除。
    """
    if not HAS_CV or img is None:
        return []
    H, W = img.shape[:2]
    if H < 24 or W < 24:
        return []
    D = math.hypot(W, H)
    gray = _to_gray(img)
    # 暗底白字反相，统一成「浅底深字」
    if float(np.median(gray)) < 110:
        gray = 255 - gray
    bw = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 横向连字成行（核宽约为字高的量级），纵向不膨，避免把上下两行粘一起
    kx = max(3, int(round(0.010 * W)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    conn = cv2.dilate(bw, kernel, iterations=1)

    cnts = cv2.findContours(conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    h_min, h_max = max(7.0, 0.006 * D), max(12.0, 0.10 * H)
    boxes: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h < h_min or h > h_max:
            continue
        if w < 6 or w * h < 40:
            continue
        ar = w / float(h)
        if ar < 0.12 or ar > 40.0:
            continue
        if (w * h) / float(W * H) > 0.22:      # 大色块/整张图的外框
            continue
        patch = bw[y:y + h, x:x + w]
        fill = float(cv2.countNonZero(patch)) / float(max(1, w * h))
        if fill < 0.06 or fill > 0.85:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    boxes = _merge_rows(boxes)
    # 去掉被大框完全包含的小框（重叠检测造成的重复）
    boxes = _drop_contained(boxes)
    boxes.sort(key=lambda b: (b[1] // max(1, int(h_min)), b[0]))
    if len(boxes) > max_boxes:                  # 太多就取面积最大的那批
        boxes = sorted(boxes, key=lambda b: -b[2] * b[3])[:max_boxes]
        boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _merge_rows(boxes: Sequence[Sequence[int]]) -> List[Tuple[int, int, int, int]]:
    """把同一行的相邻框（词与词之间有空隙）合并成整行。"""
    work = [list(map(int, b)) for b in boxes]
    while True:
        merged_any = False
        out: List[List[int]] = []
        while work:
            a = work.pop()
            for i, b in enumerate(work):
                if _same_row(a, b):
                    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
                    x2 = max(a[0] + a[2], b[0] + b[2])
                    y2 = max(a[1] + a[3], b[1] + b[3])
                    work[i] = [x1, y1, x2 - x1, y2 - y1]
                    merged_any = True
                    break
            else:
                out.append(a)
        work = out
        if not merged_any:
            break
    return [tuple(b) for b in work]


def _same_row(a: Sequence[int], b: Sequence[int]) -> bool:
    ay1, ay2 = a[1], a[1] + a[3]
    by1, by2 = b[1], b[1] + b[3]
    overlap = min(ay2, by2) - max(ay1, by1)
    if overlap < 0.45 * min(a[3], b[3]):
        return False
    gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
    return gap <= 1.0 * max(a[3], b[3])


def _drop_contained(boxes: Sequence[Sequence[int]]):
    out = []
    for i, a in enumerate(boxes):
        ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        inside = False
        for j, b in enumerate(boxes):
            if i == j:
                continue
            bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            if (bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2
                    and (b[2] * b[3]) > (a[2] * a[3]) * 1.15):
                inside = True
                break
        if not inside:
            out.append(tuple(a))
    return out


# --------------------------------------------------------------------------
# 2) 让视觉模型读内容
# --------------------------------------------------------------------------
_BOX_PROMPT = (
    "你是一个 OCR 引擎。图中用绿色矩形标出了若干文字区域，每个框的左上角有"
    "白底黑字的编号。请逐个读出每个编号框内的文字，要求：\n"
    "1. 原样转写，保留数字、字母大小写、单位、符号；不要翻译、不要解释、不要补全；\n"
    "2. 看不清或框里没有文字的，text 填空字符串；\n"
    "3. 只输出一个 JSON 数组，不要代码块、不要多余文字，格式：\n"
    '[{"id":1,"text":"M10"},{"id":2,"text":"长度 50"}]\n'
)

_FREE_PROMPT = (
    "这是一张草图或工程图。请找出图中所有文字（尺寸标注、编号、说明、标题等），"
    "按从上到下、从左到右的顺序返回。只输出一个 JSON 数组，不要代码块、不要解释：\n"
    '[{"text":"M10","x":0.32,"y":0.68,"h":0.03}]\n'
    "x、y 是文字左上角在图片中的相对坐标（0~1），h 是字高占图片高度的比例。"
    "看不清的跳过，不要编造。"
)


def _b64_png(img) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    import base64
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _annotate_boxes(img, boxes: Sequence[Sequence[int]]):
    """给每个文字框画绿框 + 白底编号，返回标注后的图（超宽会等比缩小）。"""
    vis = img.copy()
    for i, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = str(i)
        scale = max(0.5, min(1.2, h / 26.0))
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        ty = max(0, y - th - 6)
        cv2.rectangle(vis, (x, ty), (x + tw + 8, ty + th + 6), (255, 255, 255), -1)
        cv2.putText(vis, label, (x + 4, ty + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return _limit_size(vis)


def _limit_size(img, max_dim: int = 1600):
    H, W = img.shape[:2]
    m = max(H, W)
    if m <= max_dim:
        return img
    f = max_dim / float(m)
    return cv2.resize(img, (max(1, int(W * f)), max(1, int(H * f))),
                      interpolation=cv2.INTER_AREA)


def _parse_json_array(text: str) -> List[dict]:
    """从模型输出里抠出第一个完整的 JSON 数组（容忍代码块/前后废话）。"""
    if not text:
        return []
    s = text.replace("```json", "```").replace("```", "")
    start = s.find("[")
    if start < 0:
        return []
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    val = json.loads(s[start:i + 1])
                except Exception:  # noqa: BLE001
                    return []
                return val if isinstance(val, list) else []
    return []


def _ask_vision(prompt: str, image_b64: str, model: str, host: str,
                first_token: float = 180.0, stall: float = 40.0,
                total: float = 240.0) -> Tuple[str, Optional[str]]:
    msgs = [{"role": "user", "content": prompt, "images": [image_b64]}]
    return ollama_stream.stream_ollama_chat(
        msgs, model=model, host=host, temperature=0.1, num_predict=800,
        first_token=first_token, stall=stall, total=total)


# --------------------------------------------------------------------------
# 3) 像素 → 毫米 映射
# --------------------------------------------------------------------------
class _Mapper:
    """把像素坐标映射到 spec 的毫米坐标系。

    CV 路线有 _meta.frame = [cx, cy, scale]（px = cx + x/scale, py = cy - y/scale）；
    VLM 路线没有 frame，就用「几何包围盒 ≈ 图片内容」近似推一个，
    位置不会像素级精确，但文字至少落在该在的零件附近。
    """

    def __init__(self, cx: float, cy: float, scale: float,
                 ox: float = 0.0, oy: float = 0.0):
        self.cx, self.cy = float(cx), float(cy)
        self.scale = float(scale) if scale else 1.0
        self.ox, self.oy = float(ox), float(oy)

    def to_mm(self, px: float, py: float) -> Tuple[float, float]:
        return ((px - self.cx) * self.scale + self.ox,
                (self.cy - py) * self.scale + self.oy)

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        return (self.cx + (x - self.ox) / self.scale,
                self.cy - (y - self.oy) / self.scale)


def _make_mapper(spec: dict, img) -> _Mapper:
    H, W = img.shape[:2]
    frame = (spec.get("_meta") or {}).get("frame")
    if frame and len(frame) == 3 and frame[2]:
        return _Mapper(frame[0], frame[1], frame[2])
    # 无 frame：用几何包围盒近似
    xs: List[float] = []
    ys: List[float] = []
    for e in spec.get("entities") or []:
        for key in ("p", "p1", "p2", "center"):
            v = e.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    xs.append(float(v[0]))
                    ys.append(float(v[1]))
                except (TypeError, ValueError):
                    pass
        for v in (e.get("points") or []):
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    xs.append(float(v[0]))
                    ys.append(float(v[1]))
                except (TypeError, ValueError):
                    pass
        for k in ("x", "y"):
            if isinstance(e.get(k), (int, float)):
                (xs if k == "x" else ys).append(float(e[k]))
        if isinstance(e.get("w"), (int, float)):
            xs.append(float(e.get("x", 0)) + float(e["w"]))
        if isinstance(e.get("h"), (int, float)):
            ys.append(float(e.get("y", 0)) + float(e["h"]))
    if len(xs) < 2 or len(ys) < 2:
        return _Mapper(W / 2.0, H / 2.0, max(W, H) / 100.0)
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    if bw <= 0 or bh <= 0:
        return _Mapper(W / 2.0, H / 2.0, max(W, H) / 100.0)
    scale = max(bw / (0.92 * W), bh / (0.92 * H))
    return _Mapper(W / 2.0, H / 2.0, scale,
                   (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


# --------------------------------------------------------------------------
# 4) 主入口
# --------------------------------------------------------------------------
def _text_entity(mapper: _Mapper, x: int, y: int, w: int, h: int,
                 value: str, image_h: int) -> Optional[Dict[str, Any]]:
    value = " ".join(str(value).split())
    if not value:
        return None
    if len(value) > 120:
        value = value[:120]
    x_mm, y_mm = mapper.to_mm(x, y + h)          # 左下角，DXF 文本锚点
    height = max(2.0, min(60.0, h * mapper.scale * 0.92))
    return {"type": "text",
            "p": [round(x_mm, 3), round(y_mm, 3)],
            "height": round(height, 3),
            "value": value,
            "rotation": 0}


def _drop_strokes_inside(spec: dict, mapper: _Mapper,
                         boxes: Sequence[Sequence[int]]) -> int:
    """删掉完全落在文字框内的 line / polyline —— 它们是字的笔画，不是线条。"""
    if not boxes:
        return 0
    keep = []
    dropped = 0
    for e in spec.get("entities") or []:
        t = str(e.get("type"))
        if t not in ("line", "polyline"):
            keep.append(e)
            continue
        pts = []
        if t == "line":
            for key in ("p1", "p2"):
                v = e.get(key)
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    pts.append(v)
        else:
            pts = [p for p in (e.get("points") or [])
                   if isinstance(p, (list, tuple)) and len(p) >= 2]
        if len(pts) < 2:
            keep.append(e)
            continue
        inside_all = True
        for p in pts:
            try:
                px, py = mapper.to_px(float(p[0]), float(p[1]))
            except (TypeError, ValueError):
                inside_all = False
                break
            hit = False
            for (bx, by, bw, bh) in boxes:
                if bx - 2 <= px <= bx + bw + 2 and by - 2 <= py <= by + bh + 2:
                    hit = True
                    break
            if not hit:
                inside_all = False
                break
        if inside_all:
            dropped += 1
        else:
            keep.append(e)
    if dropped:
        spec["entities"] = keep
    return dropped


def enrich_spec_with_text(spec: dict, img, model: str = "qwen2.5vl:7b",
                          host: str = "http://localhost:11434",
                          max_boxes: int = 40,
                          first_token: float = 180.0) -> dict:
    """就地把识别到的文字加到 spec（text 图元），返回结果信息。

    返回 {"ok":bool, "count":int, "note":str, "error":str, "dropped":int}
    任何一步失败都只返回 error，不会抛异常，也不会动几何图元。
    """
    info: Dict[str, Any] = {"ok": False, "count": 0, "note": "", "error": "",
                            "dropped": 0}
    if not HAS_CV or img is None:
        info["error"] = "缺少 OpenCV，无法定位文字"
        return info
    try:
        boxes = detect_text_boxes(img, max_boxes=max_boxes)
        mapper = _make_mapper(spec, img)
        H, W = img.shape[:2]
        texts: List[Dict[str, Any]] = []

        if boxes:
            annotated = _annotate_boxes(img, boxes)
            raw, err = _ask_vision(_BOX_PROMPT, _b64_png(annotated),
                                   model, host, first_token=first_token)
            if err and not raw:
                info["error"] = err
                return info
            got = {}
            for item in _parse_json_array(raw):
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("id"))
                except (TypeError, ValueError):
                    continue
                txt = str(item.get("text") or "").strip()
                if txt:
                    got[idx] = txt
            for i, (x, y, w, h) in enumerate(boxes, 1):
                ent = _text_entity(mapper, x, y, w, h, got.get(i, ""), H)
                if ent:
                    texts.append(ent)
            if not texts:
                info["error"] = "模型没有读出任何文字（可换更清晰的图再试）"
                return info
            info["dropped"] = _drop_strokes_inside(spec, mapper, boxes)
        else:
            # 没找到规整文字框：让模型自己给相对坐标兜底
            raw, err = _ask_vision(_FREE_PROMPT, _b64_png(_limit_size(img)),
                                   model, host, first_token=first_token)
            if err and not raw:
                info["error"] = err
                return info
            for item in _parse_json_array(raw):
                if not isinstance(item, dict):
                    continue
                txt = str(item.get("text") or "").strip()
                if not txt:
                    continue
                try:
                    rx = float(item.get("x", 0.0))
                    ry = float(item.get("y", 0.0))
                    rh = float(item.get("h", 0.03))
                except (TypeError, ValueError):
                    continue
                rx = min(max(rx, 0.0), 1.0)
                ry = min(max(ry, 0.0), 1.0)
                px, py = rx * W, ry * H
                box_h = max(6.0, min(H * 0.2, rh * H))
                ent = _text_entity(mapper, int(px), int(py),
                                   int(max(10.0, box_h * 6)), int(box_h), txt, H)
                if ent:
                    texts.append(ent)
            if not texts:
                info["error"] = "模型没有读出任何文字（可换更清晰的图再试）"
                return info

        spec.setdefault("entities", []).extend(texts)
        meta = spec.setdefault("_meta", {})
        counts = meta.get("counts")
        if not isinstance(counts, dict):
            counts = {}
            meta["counts"] = counts
        counts["text"] = counts.get("text", 0) + len(texts)
        meta["ocr"] = {"count": len(texts),
                       "boxes": len(boxes),
                       "model": model,
                       "dropped_strokes": info["dropped"]}
        info["ok"] = True
        info["count"] = len(texts)
        info["texts"] = [t["value"] for t in texts]
        return info
    except Exception as e:  # noqa: BLE001  兜底：OCR 出问题也不能影响几何
        info["error"] = str(e)[:160]
        return info
