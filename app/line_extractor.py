"""
line_extractor.py — 经典计算机视觉（OpenCV）几何提取器

为什么不堆大模型来"识线"：
  本地 7B 视觉模型靠语义先验"猜"形状，坐标误差常达几十像素，且每次输出都可能不同。
  而霍夫直线变换 / 轮廓检测是确定性算法：草图上的直线被精确提取，圆心半径分毫不差，
  结果 100% 可复现。对"只需要识别线条然后画"这类任务，经典 CV 的精度实际超过 7B VLM。

管线：
  读图 → 降采样(长边>1400) → **颜色无关墨迹场(Lab 色差)** → 二值化 → 多遍直线检测
     → 6 套「预处理+CLAHE+Canny+Hough」并行跑：基线/灵敏/严格/断线桥接/去噪/低对比
     → 彩色图再加 2 遍"只看色度"的通道：淡红/淡蓝/淡青标注线即使亮度接近纸色也能抓出来
     → 跨遍投票融合（票数 + 墨迹支撑度裁决，PCA 重拟合中心线） → 共线合并(吃下虚线/粗线双边缘)
     → 文字区抹白 + 字形框剔除（尺寸数值/R 标注的笔画不当成几何）
     → 轴对齐吸附 + 端点吸附
     → 矩形识别(轮廓+直线配对，带边/角支撑度验证) → 圆识别(HoughCircles+支持度)
     → 剔除贴圆切线的弧线碎片 → 剩余碎片串多段线
     → 坐标翻转(y-down→y-up) → 归一化到毫米 → 标准几何 spec

为什么跑 6 遍（彩色图 8 遍）而不是调一套"最优参数"：
  单套阈值不可能兼顾淡线与粗线、虚线与噪点、红线与黑线。多遍各自去找，再由投票决定去留——
  召回靠"任一 profile 命中即入池"，特异性靠"多遍独立命中 / 线身有墨迹"双重背书，
  假线（阴影边、网格、压缩噪点）几乎不可能同时骗过 3 套以上互相矛盾的参数。

仅依赖 opencv + numpy。无 OpenCV 时本模块整体不可用（调用方应兜底到 VLM 路线）。
"""

from __future__ import annotations
import base64
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    HAS_CV = True
except Exception:  # ImportError 或 DLL 缺失等
    HAS_CV = False


# ---------------------------------------------------------------- 基础 IO

def available() -> bool:
    return HAS_CV


def load_image(path: str):
    """读取图片；cv2.imread 不支持中文路径，用 np.fromfile + imdecode 兜底。"""
    if not HAS_CV:
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_image_b64(b64: str):
    """从 base64 字符串解码图片（Web 上传用）。"""
    if not HAS_CV or not b64:
        return None
    try:
        if "," in b64[:64]:                 # data:image/png;base64,....
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64 + "=" * (-len(b64) % 4))
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------------------------------------------- 二值化

def _binarize(gray):
    """自适应二值化：输出线条=255、背景=0。自动处理暗底亮线/亮底暗线两种情况。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 0).mean() > 0.5:          # 白色占多数 → 线条被当成了背景，反转
        bw = 255 - bw
    ratio = (bw > 0).mean()
    if ratio < 0.002 or ratio > 0.6:   # Otsu 失效（近均匀图），退化到自适应阈值
        bw = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        if (bw > 0).mean() > 0.5:
            bw = 255 - bw
    return bw


# ---------------------------------------------------------------- 颜色无关墨迹场
#
# 为什么不能直接转灰度：
#   工程图里除了黑色轮廓，还有红色尺寸线、蓝/青色中心线、浅灰引线、以及扫描件的
#   灰阶淡线。转灰度后这些颜色只剩"亮度差"这一维信息——淡红外加 JPEG 压缩，
#   红线的饱和度会掉到 20~60、亮度掉到 210~230，Otsu 直接把它和白纸划进同一类，
#   一个像素都进不了墨迹（实测：一张 638×725 的标注图，红线全丢，只认出黑轮廓）。
#
# 做法：在 Lab 色彩空间里量"每个像素离纸色有多远"（ΔE），而不是只看亮度。
#   纸色用众数色估计（线稿图背景占 60%~95%，众数稳如磐石）；
#   照片/渐变纸这类没有主导色的图，退化成"局部背景"估计（下采样 + 中值 + 放大）。
#   得到的 dist 再按噪声底 + 99.5 分位归一化并做 gamma 提亮，让"淡红线"与
#   "黑轮廓"落在同一张可比较的灰阶上——后续 6 遍 Canny/Hough 才能一视同仁。

def _bg_lab_modal(lab):
    """用粗量化直方图的众数估计纸色（返回 (bg_lab, 占比)）。"""
    L = (lab[..., 0].astype(np.int32) >> 3)
    A = (lab[..., 1].astype(np.int32) >> 3)
    B = (lab[..., 2].astype(np.int32) >> 3)
    idx = (L << 10) | (A << 5) | B
    flat = idx.ravel()
    cnt = np.bincount(flat, minlength=1 << 15)
    k = int(cnt.argmax())
    sel = flat == k
    frac = float(sel.mean())
    if sel.sum() < 16:
        return np.median(lab.reshape(-1, 3), axis=0), 0.0
    return np.median(lab.reshape(-1, 3)[sel], axis=0), frac


def _bg_lab_local(lab):
    """局部背景估计：缩小 → 中值 → 放大，得到"没有线条的纸"。
    用于照片/渐变背景：阴影、暖光、纸张不匀都能被吃掉，同时细线不影响 1/8 尺度上的中值。"""
    H, W = lab.shape[:2]
    s = 8 if max(H, W) > 400 else 4
    small = cv2.resize(lab, (max(4, W // s), max(4, H // s)), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def _normalize_ink(d):
    """色差图 → 墨迹灰阶。返回 (gray, noise, noise_raw, ref)。"""
    noise_raw = float(np.median(d))
    p995 = float(np.percentile(d, 99.5))
    ref = max(p995 - noise_raw, 1.0)
    noise = min(noise_raw, 0.15 * ref)           # 极端图（大面积墨迹）时别把噪声底抬高
    lvl = np.clip((d - noise) / ref, 0.0, 1.0)
    ink = np.power(lvl, 0.62)                    # gamma<1：把淡线（0.2~0.4）提到 0.35~0.55
    gray = np.clip(255.0 * (1.0 - ink), 0, 255).astype(np.uint8)
    return gray, noise, noise_raw, ref


def _field_from_dist(d):
    """只按色差图重建一个"纯墨迹场"（无彩色通道），给颜色过滤后的图用。"""
    gray, noise, noise_raw, ref = _normalize_ink(d)
    return {"gray": gray, "dist": d.astype(np.float32), "noise": noise,
            "noise_raw": noise_raw, "ref": ref, "chroma": gray,
            "chroma_raw": None, "chroma_cov": 0.0, "colorful": False}


def _dist_map(image):
    """原图 → (色差图 d, 色度图 chroma, 彩色占比, 背景色)。"""
    if image.ndim == 2:
        g = image.astype(np.float32)
        bgv = float(np.median(g))
        d = np.abs(g - bgv)
        return d, np.zeros_like(d), 0.0, bgv
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg, frac = _bg_lab_modal(lab)
    if frac < 0.55:
        bg = _bg_lab_local(lab)                  # 无主导背景色 → 走局部估计
    d = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    # 色度：与"中性灰轴"的距离（Lab 的 a/b 都偏离 128 才算有色）
    chroma = np.sqrt((lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2)
    return d, chroma, float((chroma > 8.0).mean()), bg


def ink_field(image):
    """把彩图统一成"墨迹强度场"：返回 dict。

    gray   : uint8，纸=255、墨=0（可直接喂 Canny / 二值化）
    dist   : float32，每像素与纸色的 Lab 距离（ΔE 近似）
    noise  : 背景噪声底（中位数），用于定阈值
    ref    : 归一化参考（99.5 分位 - 噪声）
    chroma : uint8，只看色度的墨迹强度（黑线=255 白底，彩色线变暗），供色度专用遍使用
    colorful/chroma_cov : 是否真有彩色墨迹、彩色像素占比
    """
    if image is None:
        return None
    d, chroma, chroma_cov, _ = _dist_map(image)
    noise_raw = float(np.median(d))
    # 纸纹/传感器噪声是高频的，线条不是：噪声底超过最强墨迹的 1/10 时，
    # 先做一次 3×3 轻模糊重算色差。淡铅笔稿上这一步把纸纹从 σ≈3 压到 σ≈1.3
    # （实测：700×700 淡稿不模糊吐 290 条假线，模糊后 3 条）；
    # 干净的数字图/截图噪声底≈0，完全不受影响（保住 1px 细线的锐度）。
    if noise_raw > 0.10 * max(float(np.percentile(d, 99.5)) - noise_raw, 1.0) \
            and min(image.shape[:2]) >= 16:
        d, chroma, chroma_cov, _ = _dist_map(cv2.GaussianBlur(image, (3, 3), 0))

    gray, noise, noise_raw, ref = _normalize_ink(d)

    if image.ndim == 2:
        ch_gray = gray
        chroma_cov = 0.0
    else:
        cref = max(float(np.percentile(chroma, 99.5)), 8.0)
        cl = np.power(np.clip(chroma / cref, 0.0, 1.0), 0.55)
        ch_gray = np.clip(255.0 * (1.0 - cl), 0, 255).astype(np.uint8)

    return {"gray": gray, "dist": d, "noise": noise, "noise_raw": noise_raw, "ref": ref,
            "chroma": ch_gray, "chroma_raw": chroma.astype(np.float32),
            "chroma_cov": chroma_cov, "colorful": chroma_cov > 0.0015}


def _line_structure(bw, samples=0):
    """墨迹"成线程度"：用 1×3 / 3×1 的方向开运算，看墨迹像素里有多少能连成 3 个以上。

    真实线条（哪怕只有 1px 宽）几乎都能通过；纸纹、JPEG 噪点、照片纹理这些
    "满天星"式墨迹通过率很低（实测：线稿 0.98~1.00，纸纹噪声 0.24~0.48）。
    这是"这张图到底是不是线稿"最稳的判据，比单纯看墨迹占比靠谱。
    """
    n = int((bw > 0).sum())
    if n <= 0:
        return 0.0
    r = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8))
    c = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
    return float(((r | c) > 0).sum()) / n


def _binarize_dist(field, deep=None):
    """按"离纸色多远"二值化。不用 Otsu：线稿图的直方图是"白纸上几根针"，
    Otsu 会把阈值推到接近纸色，JPEG 振铃/纸纹全被当成墨迹；改用噪声底 + 相对量。"""
    d = field["dist"]
    thr = field["noise"] + max(5.0, float(deep if deep else _INK_DEEP) * field["ref"])
    bw = (d > thr).astype(np.uint8) * 255
    return bw, thr


def bw_photo(image, deep: float = None):
    """把任意照片压成高对比纯黑白：纸=255（白）、墨=0（黑）。返回 uint8 BGR（或灰度）。

    供"生成 DXF 前先转纯黑白图"用。关键点：
    - 判墨走颜色无关墨迹场（Lab 色差），不是灰度 Otsu——淡红尺寸线、浅灰引线
      与黑轮廓一视同仁；暗底亮线照片也会被归一成白底黑线。
    - 阈值与主管线同款（噪声底 + deep×最强墨迹），天然抗 JPEG 振铃与纸纹阴影；
      背景直接钉死成纯白，对比度拉满。
    - 只删"孤立的小噪点连通域"（振铃残点），不碰任何贴着线条的结构。
    """
    if image is None or not HAS_CV:
        return None
    field = ink_field(image)
    d = field["dist"]
    thr = field["noise"] + max(5.0, float(_INK_DEEP if deep is None else deep) * field["ref"])
    ink = d > thr
    # 孤立小噪点：面积 ≤ max(2, 图幅百万分之四) 的连通域删除（虚线圆点通常 ≥4px 且成串）
    try:
        nlab, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        min_area = max(2.0, 4e-6 * ink.size)
        for i in range(1, nlab):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                ink[lab == i] = False
    except Exception:                      # noqa: BLE001 - 去噪失败不影响主流程
        pass
    out = np.where(ink, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out


# ---------------------------------------------------------------- 直线工具

def _seg_len(s) -> float:
    return math.hypot(s[2] - s[0], s[3] - s[1])


def _seg_angle(s) -> float:
    """无向角度，归一化到 [0,180)。"""
    a = math.degrees(math.atan2(s[3] - s[1], s[2] - s[0]))
    return a % 180.0


def _ang_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _perp_dist(px, py, s) -> float:
    """点到直线(无限延长)的垂直距离。"""
    x1, y1, x2, y2 = s
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return math.hypot(px - x1, py - y1)
    return abs(dy * (px - x1) - dx * (py - y1)) / L


def _try_extend(base, s, dist_tol, max_bridge):
    """若线段 s 与 base 共线（两端垂直距离≤tol）且纵向间隙≤max_bridge，
    返回延长后的 base；否则 None。max_bridge 防止把中间隔着一段的两条独立线误并。"""
    ax, ay, bx, by = s[0], s[1], s[2], s[3]
    if _perp_dist(ax, ay, base) > dist_tol or _perp_dist(bx, by, base) > dist_tol:
        return None
    dx, dy = base[2] - base[0], base[3] - base[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return None
    ux, uy = dx / L, dy / L
    t1 = (ax - base[0]) * ux + (ay - base[1]) * uy
    t2 = (bx - base[0]) * ux + (by - base[1]) * uy
    lo, hi = min(t1, t2), max(t1, t2)
    if lo > L:                      # 整段在 base 前方
        if lo - L > max_bridge:
            return None
    elif hi < 0:                    # 整段在 base 后方
        if -hi > max_bridge:
            return None
    return (base[0] + ux * min(0.0, lo), base[1] + uy * min(0.0, lo),
            base[0] + ux * max(L, hi), base[1] + uy * max(L, hi))


def _merge_collinear(segs, angle_tol=6.0, dist_tol=5.0, max_bridge=48.0):
    """合并共线碎片：长段优先当主干，逐步吸收同向且贴合的小段。
    可把粗线的双边缘、虚线碎片、小断裂缝合成一条中心线；
    max_bridge 限制能跨过的间隙（虚线约 15px，独立线段间距通常更大）。"""
    merged: List[list] = []
    used = [False] * len(segs)
    order = sorted(range(len(segs)), key=lambda i: -_seg_len(segs[i]))
    for i in order:
        if used[i]:
            continue
        base = list(segs[i])
        used[i] = True
        changed = True
        while changed:                      # 迭代直到吸不动为止
            changed = False
            for j in order:
                if used[j]:
                    continue
                if _ang_diff(_seg_angle(base), _seg_angle(segs[j])) > angle_tol:
                    continue
                nb = _try_extend(base, segs[j], dist_tol, max_bridge)
                if nb is not None:
                    base = list(nb)
                    used[j] = True
                    changed = True
        merged.append(base)
    return [tuple(s) for s in merged]


def _snap_axis(s, tol_deg=1.5):
    """近水平/近垂直的线吸附到精确 0°/90°（CAD 里看起来才干净）。"""
    x1, y1, x2, y2 = s
    a = _seg_angle(s)
    if a <= tol_deg or a >= 180.0 - tol_deg:
        y = (y1 + y2) / 2.0
        return (x1, y, x2, y)
    if abs(a - 90.0) <= tol_deg:
        x = (x1 + x2) / 2.0
        return (x, y1, x, y2)
    return s


def _dedup_lines(segs, angle_tol=3.0, dist_tol=3.0):
    """去掉重复/近重复线段（保留更长者）。"""
    kept: List[tuple] = []
    for s in sorted(segs, key=_seg_len, reverse=True):
        dup = False
        for k in kept:
            if _ang_diff(_seg_angle(s), _seg_angle(k)) > angle_tol:
                continue
            if _perp_dist((s[0] + s[2]) / 2, (s[1] + s[3]) / 2, k) > dist_tol:
                continue
            # 投影重叠判定：任一端点落在对方的延长范围内
            ux, uy = (k[2] - k[0]) / max(_seg_len(k), 1e-9), (k[3] - k[1]) / max(_seg_len(k), 1e-9)
            t1 = (s[0] - k[0]) * ux + (s[1] - k[1]) * uy
            t2 = (s[2] - k[0]) * ux + (s[3] - k[1]) * uy
            Lk = _seg_len(k)
            if max(t1, t2) >= -dist_tol and min(t1, t2) <= Lk + dist_tol:
                dup = True
                break
        if not dup:
            kept.append(s)
    return kept


def _chain_segments(segs, tol, return_used=False):
    """把首尾相接（夹角<60°）的碎片串成多段线，输出更干净的 CAD 曲线。

    return_used=True 时额外返回"已被串进多段线的线段下标"，
    调用方要把这些线段从直线列表里去掉，否则同一条边会被画两遍。
    """
    used_idx = set()
    segs = [list(s) for s in segs]
    used = [False] * len(segs)
    chains: List[List[Tuple[float, float]]] = []
    order = sorted(range(len(segs)), key=lambda i: -_seg_len(segs[i]))
    for i in order:
        if used[i]:
            continue
        used[i] = True
        members = [i]
        chain = [(segs[i][0], segs[i][1]), (segs[i][2], segs[i][3])]
        growing = True
        while growing:
            growing = False
            tail = chain[-1]
            best, best_score = None, 1e18
            for j in order:
                if used[j]:
                    continue
                # 两个端点都试，选更优接入
                for end_idx in (0, 2):
                    px, py = segs[j][end_idx], segs[j][end_idx + 1]
                    d = math.hypot(px - tail[0], py - tail[1])
                    if d > tol:
                        continue
                    ox, oy = segs[j][2] - segs[j][0], segs[j][3] - segs[j][1]
                    if end_idx == 2:
                        ox, oy = -ox, -oy
                    score = d
                    if len(chain) >= 2:
                        vx, vy = tail[0] - chain[-2][0], tail[1] - chain[-2][1]
                        va, vo = math.hypot(vx, vy), math.hypot(ox, oy)
                        if va > 1e-6 and vo > 1e-6:
                            cosv = max(-1.0, min(1.0, (vx * ox + vy * oy) / (va * vo)))
                            turn = math.degrees(math.acos(cosv))
                            if turn > 60.0:
                                continue
                            score = d + turn * 2.0
                    if score < best_score:
                        best_score, best = score, j
            if best is not None:
                used[best] = True
                members.append(best)
                s = segs[best]
                other = 2 if (s[0], s[1]) == (chain[-1][0], chain[-1][1]) else 0
                # 找到 s 上离 tail 最近的点，接上另一端
                if math.hypot(s[0] - tail[0], s[1] - tail[1]) <= math.hypot(s[2] - tail[0], s[3] - tail[1]):
                    chain.append((s[2], s[3]))
                else:
                    chain.append((s[0], s[1]))
                growing = True
        if len(chain) >= 3:
            chains.append(chain)
            used_idx.update(members)
    if return_used:
        return chains, used_idx
    return chains


# ---------------------------------------------------------------- 短弦串弧
#
# 霍夫只认直线：挂钩 / 法兰 / 手柄这类"直线 + 切线圆弧"的工程轮廓，弧线会被
# 切成一串首尾相接的短弦（实测挂钩图 173 条线、中位 8.9px，一条 R29 弧贡献
# 几十段）。HoughCircles 那条路又接不住部分弧（覆盖率/支持度判据是为整圆调的）。
# 这里把"首尾相接、同向缓弯、能共圆"的弦链升级成 ARC 实体：
#   · 只串短弦（长直线不吞），岔口即停（唯一候选才前进）；
#   · 每接一段就用整条链做一次圆拟合，偏差超限立刻断链 —— 保证切点处的
#     直线不会被误吸进弧里；
#   · 接受前还要在膨胀墨迹上量弧带的覆盖率，杜绝"交叉直线凑出来的假圆弧"。

def _fit_circle_pts(pts):
    """Kåsa 代数最小二乘圆拟合。返回 (cx, cy, r, max_radial_dev) 或 None。"""
    n = len(pts)
    if n < 5:
        return None
    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)
    A = np.column_stack([x, y, np.ones(n)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:                          # noqa: BLE001
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 4.0:
        return None
    r = math.sqrt(r2)
    dev = max(abs(math.hypot(px - cx, py - cy) - r) for px, py in pts)
    return cx, cy, r, dev


def _arc_ink_cover(dil, cx, cy, r, a0, a1, step_deg=3.0):
    """弧带墨迹覆盖率：沿弧每 ~3° 采样一次，落在膨胀墨迹上的比例。"""
    if dil is None:
        return 1.0
    span = (a1 - a0) % 360.0
    if span < 1e-3:
        span = 360.0
    n = max(16, int(span / step_deg))
    H, W = dil.shape[:2]
    hit = 0
    for k in range(n + 1):
        a = math.radians(a0 + span * k / n)
        ix, iy = int(cx + r * math.cos(a)), int(cy + r * math.sin(a))
        if 0 <= iy < H and 0 <= ix < W and dil[iy, ix] > 0:
            hit += 1
    return hit / (n + 1)


def _circumcircle(p1, p2, p3):
    """三点定圆。返回 (cx, cy, r) 或 None（三点近共线以外，任何三点都能定圆）。"""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    r = math.hypot(ax - ux, ay - uy)
    if r < 1e-6:
        return None
    return ux, uy, r


def _chords_to_arcs(segs, D, dil=None, circles=()):
    """把霍夫切成短弦的圆弧重新串成弧实体。返回 (剩余 segs, 新增 arcs)。

    弧元组与 _detect_circles 一致：[(cx, cy, r, a0, a1)]，角度按图像坐标
    （y 向下），沿 a0→a1（模 360 递增）覆盖弧段，交给统一的角度翻转逻辑出 DXF。

    算法：共圆生长（不依赖端点精确共点 —— 霍夫碎段的端点漂移好几个像素，
    "端点串链"会处处断）。
      1) 种子：相邻（端点距 ≤ 0.05×对角）的短弦对，三点定圆；
      2) 生长：凡"两端都贴圆 + 走向近似切向"的未收编短弦全部收编；
      3) 精修：Kåsa 拟合 → 按弦偏差修剪离群（掐掉贴着圆周混进来的直线弦）
         → 重拟合，迭代两轮；
      4) 接受：支撑弦 ≥4、角跨度 25°~345°、墨迹弧带覆盖率 ≥0.55（5×5 膨胀带）、
         半径 ≤ 0.55×对角；不与已检出整圆重合；
      5) 全部种子评完后按"覆盖角×覆盖率"贪心收编（支撑弦大多空闲才收）。
    """
    segs = [list(map(float, s)) for s in segs]
    if len(segs) < 4:
        return segs, []
    tol_grow = max(4.0, 0.007 * D)
    max_chord = max(28.0, 0.10 * D)
    seed_dist = max(20.0, 0.05 * D)
    r_max = 0.55 * D
    r_min = max(6.0, 0.010 * D)
    dil5 = (cv2.dilate(bw_for_cover, np.ones((5, 5), np.uint8))
            if (bw_for_cover := dil) is not None else None)

    cand = [i for i, s in enumerate(segs) if _seg_len(s) <= max_chord]
    if len(cand) < 4:
        return segs, []

    def _ends(i):
        s = segs[i]
        return (s[0], s[1]), (s[2], s[3])

    def _mid(i):
        s = segs[i]
        return ((s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0)

    def _chord_dir(i):
        s = segs[i]
        L = _seg_len(s) or 1.0
        return (s[2] - s[0]) / L, (s[3] - s[1]) / L

    def _chord_dev(i, cx, cy, r):
        """弦偏离圆的程度：两端点径向偏差的较大值。"""
        (x0, y0), (x1, y1) = _ends(i)
        return max(abs(math.hypot(x0 - cx, y0 - cy) - r),
                   abs(math.hypot(x1 - cx, y1 - cy) - r))

    def _tangent_ok(i, cx, cy):
        """弦走向是否近似切向（短弦的圆弧弦与切线夹角 = asin(L/2r)，很小；
        贴着圆周混进来的直线弦通常以浅角度斜穿，方向差得远）。"""
        mx, my = _mid(i)
        dx, dy = mx - cx, my - cy
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return False
        tx, ty = -dy / n, dx / n                 # 切向（两个朝向等价，取 abs 点积）
        cdx, cdy = _chord_dir(i)
        if abs(tx * cdx + ty * cdy) < 0.5:       # 夹角 > 60° → 不是弦
            return False
        limit = math.degrees(math.asin(min(1.0, 0.5 * _seg_len(segs[i]) / max(n, 1e-6))))
        ang = math.degrees(math.asin(min(1.0, abs(tx * cdy - ty * cdx))))
        return ang <= limit + 18.0

    def _support(cx, cy, r):
        out = []
        for i in cand:
            if used[i]:
                continue
            if _chord_dev(i, cx, cy, r) <= tol_grow and _tangent_ok(i, cx, cy):
                out.append(i)
        return out

    def _refit(members):
        """拟合 + 离群修剪，迭代两轮。返回 (cx, cy, r, members) 或 None。"""
        cur = list(members)
        for _ in range(3):
            pts = []
            for i in cur:
                s = segs[i]
                pts += [(s[0], s[1]), ((s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0),
                        (s[2], s[3])]
            fit = _fit_circle_pts(pts)
            if fit is None:
                return None
            fcx, fcy, fr, _dev = fit
            devs = sorted(_chord_dev(i, fcx, fcy, fr) for i in cur)
            med = devs[len(devs) // 2] if devs else 0.0
            cut = max(2.5, 1.8 * med)
            cur = [i for i in cur if _chord_dev(i, fcx, fcy, fr) <= cut]
            if len(cur) < 4:
                return None
        if not (r_min <= fr <= r_max):
            return None
        return fcx, fcy, fr, cur

    used = [False] * len(segs)
    found = []

    # ---- 种子枚举：相邻短弦对 ----
    seen_seeds = set()
    for a_i in range(len(cand)):
        i = cand[a_i]
        (ix0, iy0), (ix1, iy1) = _ends(i)
        for b_i in range(a_i + 1, len(cand)):
            j = cand[b_i]
            (jx0, jy0), (jx1, jy1) = _ends(j)
            ds = [math.hypot(ix0 - jx0, iy0 - jy0), math.hypot(ix0 - jx1, iy0 - jy1),
                  math.hypot(ix1 - jx0, iy1 - jy0), math.hypot(ix1 - jx1, iy1 - jy1)]
            if min(ds) > seed_dist:
                continue
            for (px, py) in (((jx0, jy0), (jx1, jy1)) if ds[0] + ds[1] <= ds[2] + ds[3]
                             else ((jx1, jy1), (jx0, jy0))):
                cc = _circumcircle((ix0, iy0), (ix1, iy1), (px, py))
                if cc is None:
                    continue
                scx, scy, sr = cc
                if not (r_min <= sr <= r_max):
                    continue
                key = (round(scx / 6), round(scy / 6), round(sr / 6))
                if key in seen_seeds:
                    continue
                seen_seeds.add(key)
                members = _support(scx, scy, sr)
                if len(members) < 4:
                    continue
                res = _refit(members)
                if res is None:
                    continue
                fcx, fcy, fr, members = res
                if len(members) < 4:
                    continue
                angs = sorted(math.degrees(math.atan2(_mid(k)[1] - fcy, _mid(k)[0] - fcx)) % 360.0
                              for k in members)
                gaps = [(angs[(k + 1) % len(angs)] - angs[k]) % 360.0
                        for k in range(len(angs))]
                gmax_i = max(range(len(angs)), key=lambda k: gaps[k])
                span = 360.0 - gaps[gmax_i]
                if span < 25.0:
                    continue
                a0 = angs[(gmax_i + 1) % len(angs)]
                a1 = angs[gmax_i]
                # 端角取支撑弦的"外端点"角（中点角会让弧端缩进半条弦，
                # 端点落到墨迹外）：全部端点角里离缺口中心最近/最远的就是两端
                gmid = (angs[gmax_i] + gaps[gmax_i] / 2.0) % 360.0
                eangs, eds = [], []
                for k in members:
                    for (ex, ey) in _ends(k):
                        ea = math.degrees(math.atan2(ey - fcy, ex - fcx)) % 360.0
                        eangs.append(ea)
                        eds.append((ea - gmid) % 360.0)
                a0 = eangs[eds.index(min(eds))]
                a1 = eangs[eds.index(max(eds))]
                cover = _arc_ink_cover(dil5, fcx, fcy, fr, a0, a1)
                if cover < 0.55:
                    continue
                if any(math.hypot(fcx - qc, fcy - qy) < max(fr, qr) * 0.25
                       and abs(fr - qr) < max(fr, qr) * 0.25 for (qc, qy, qr) in circles):
                    continue
                found.append((span * cover, span, cover, fcx, fcy, fr, a0, a1,
                              list(members)))

    if not found:
        return segs, []
    found.sort(key=lambda t: -t[0])
    kept = []
    for (_score, span, cover, scx, scy, sr, a0, a1, members) in found:
        free = [m for m in members if not used[m]]
        if len(free) < max(4, int(0.7 * len(members))):
            continue
        # 与已接受弧共圆且角区间重叠 → 跳过（同一道弧）
        dup = False
        for (_sp2, _cv2, bx, by, br, b0, b1) in kept:
            if math.hypot(scx - bx, scy - by) < max(sr, br) * 0.25 \
                    and abs(sr - br) < max(sr, br) * 0.25:
                ov = max((a1 - b0) % 360.0, (b1 - a0) % 360.0)
                if ov > 20.0:
                    dup = True
                    break
        if dup:
            continue
        kept.append((span, cover, scx, scy, sr, a0, a1))
        for m in members:
            used[m] = True
    if not kept:
        return segs, []
    out_segs = [s for i, s in enumerate(segs) if not used[i]]
    return out_segs, [(k[2], k[3], k[4], k[5], k[6]) for k in kept]




# ---------------------------------------------------------------- 支撑度验证

def _edge_support(edges, p1, p2, samples=12, band=4.0):
    """线段附近有边缘像素的比例（用于验证矩形边是否真实存在）。"""
    H, W = edges.shape
    hit = 0
    for k in range(samples + 1):
        t = k / samples
        x, y = p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t
        x0, x1 = max(0, int(x - band)), min(W, int(x + band) + 1)
        y0, y1 = max(0, int(y - band)), min(H, int(y + band) + 1)
        if x1 > x0 and y1 > y0 and edges[y0:y1, x0:x1].max() > 0:
            hit += 1
    return hit / (samples + 1)


def _corner_support(edges, x, y, win=8):
    H, W = edges.shape
    x0, x1 = max(0, int(x) - win), min(W, int(x) + win + 1)
    y0, y1 = max(0, int(y) - win), min(H, int(y) + win + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    return bool(edges[y0:y1, x0:x1].max() > 0)


# ---------------------------------------------------------------- 多遍识别（集成投票）
#
# 单套参数一定会漏：淡线要低阈值、粗线要高阈值、虚线要大 gap、噪点要形态学、
# 低对比扫描件要 CLAHE。所以同一张图跑 6 遍「预处理 + Canny + Hough」，彩色图再加 2 遍色度，
# 再按「跨遍票数 + 墨迹支撑度」裁决：
#   · 召回（强度）：任一 profile 抓到就进候选池，弱线/断线/粗线各自有人管；
#   · 特异性（准确）：只有个别遍命中、且线身上找不到墨迹的候选一律判为假线剔除；
#   · 亚像素稳定：簇内所有端点做总体最小二乘（PCA）重拟合，比"取某一条"更贴中心线，
#     顺手解决粗笔双边缘被当成两条平行线的问题。
_PASS_PROFILES = [
    # name           blur         clahe  lo   hi   thr    minlen  gap    morph   通道
    ("baseline",     "gauss",     False, 50,  150, 0.040, 0.025, 0.020, None,    "ink"),
    ("sensitive",    "gauss",     True,  20,  60,  0.022, 0.012, 0.030, None,    "ink"),
    ("strict",       "gauss",     False, 90,  200, 0.070, 0.030, 0.012, None,    "ink"),
    ("bridge",       "gauss",     False, 40,  120, 0.030, 0.014, 0.022, "close", "ink"),
    ("denoise",      "bilateral", True,  30,  100, 0.026, 0.015, 0.025, "open",  "ink"),
    ("lowcontrast",  "median",    True,  15,  50,  0.030, 0.012, 0.035, None,    "ink"),
    # 彩色专用两遍：色度通道把"淡红/淡蓝/淡青"从中性底色里单独拎出来，
    # 不受亮度影响（黑轮廓在色度通道里几乎消失，互不干扰 → 天然是另一套独立判据）
    ("chroma",       "gauss",     True,  18,  55,  0.024, 0.012, 0.030, None,    "chroma"),
    ("chroma-strict","median",    False, 45,  110, 0.040, 0.016, 0.020, None,    "chroma"),
]
DEFAULT_PASSES = len(_PASS_PROFILES)      # 6 遍 + 彩色图 2 遍色度（需求：至少 5 遍）

_MAX_SIDE = 1400          # 超过就先降采样：既提速又压掉高频噪点，对线条识别无损
_INK_DEEP = 0.07          # 墨迹阈值：占"最强墨迹"的比例（太低会吃进 JPEG 振铃，太高会丢淡线）
_MAX_SEGS_PER_PASS = 420  # 单遍进入共线合并的线段上限，防止噪点图把 O(n²) 拖爆


def _downscale(img):
    """长边超过 _MAX_SIDE 时等比缩小。返回 (图, 缩放比)。"""
    H, W = img.shape[:2]
    m = max(H, W)
    if m <= _MAX_SIDE:
        return img, 1.0
    s = _MAX_SIDE / float(m)
    out = cv2.resize(img, (max(1, int(round(W * s))), max(1, int(round(H * s)))),
                     interpolation=cv2.INTER_AREA)
    return out, s


def _pass_edges(chans, prof):
    """按 profile 做"预处理 → Canny → 形态学"，得到该遍的边缘图。
    chans 是 {"ink": 墨迹场灰阶, "chroma": 色度灰阶}；彩色图才有多通道。"""
    _name, blur, use_clahe, lo, hi, _thr, _ml, _gap, morph, chan = prof
    g = chans.get(chan, chans["ink"])
    if use_clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    if blur == "bilateral":
        g = cv2.bilateralFilter(g, 5, 45, 45)
    elif blur == "median":
        g = cv2.medianBlur(g, 3)
    else:
        g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, int(lo), int(hi))
    if morph == "close":       # 接断线：虚线/毛边先连起来再找直线
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        e = cv2.dilate(e, np.ones((2, 2), np.uint8), iterations=1)
    elif morph == "open":      # 去噪点：先抹掉孤立毛刺，再找主线
        e = cv2.morphologyEx(e, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return e


def _hough_pass(edges, D, prof):
    """按 profile 跑一次概率霍夫直线，返回该遍的原始线段。"""
    _name, _b, _c, _l, _h, thr, minlen, gap = prof[:8]
    try:
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=math.pi / 180.0,
            threshold=max(14, int(thr * D)),
            minLineLength=max(9, int(minlen * D)),
            maxLineGap=max(4, int(gap * D)))
    except cv2.error:
        return []
    if lines is None:
        return []
    segs = [tuple(float(v) for v in l[0]) for l in lines[:2000]]
    segs.sort(key=_seg_len, reverse=True)
    return segs[:_MAX_SEGS_PER_PASS]


def _proj_overlap(a, b, bridge) -> bool:
    """线段 b 投影到 a 的方向上是否与 a 有（或近乎有）重叠。"""
    dx, dy = a[2] - a[0], a[3] - a[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return True
    ux, uy = dx / L, dy / L
    t1 = (b[0] - a[0]) * ux + (b[1] - a[1]) * uy
    t2 = (b[2] - a[0]) * ux + (b[3] - a[1]) * uy
    lo, hi = min(t1, t2), max(t1, t2)
    return not (hi < -bridge or lo > L + bridge)


def _fit_line_pca(pts, fallback):
    """总体最小二乘重拟合：质心 + 主轴，端点取簇内投影极值。"""
    if len(pts) < 2:
        return fallback
    arr = np.asarray(pts, dtype=np.float64)
    c = arr.mean(axis=0)
    try:
        _u, _s, vt = np.linalg.svd(arr - c, full_matrices=False)
    except np.linalg.LinAlgError:
        return fallback
    d = vt[0]
    t = (arr - c) @ d
    t0, t1 = float(t.min()), float(t.max())
    if t1 - t0 < 1e-6:
        return fallback
    return (float(c[0] + d[0] * t0), float(c[1] + d[1] * t0),
            float(c[0] + d[0] * t1), float(c[1] + d[1] * t1))


def _ink_support(ink, s, samples=24, band=2):
    """线条落在墨迹上的比例：沿线的墨迹覆盖率。
    这是特异性裁决的硬证据——真实线条处处有墨，噪点/阴影凑出来的假线只是偶合。"""
    H, W = ink.shape
    hit = 0
    for k in range(samples + 1):
        t = k / samples
        x = s[0] + (s[2] - s[0]) * t
        y = s[1] + (s[3] - s[1]) * t
        x0, x1 = max(0, int(x - band)), min(W, int(x + band) + 1)
        y0, y1 = max(0, int(y - band)), min(H, int(y + band) + 1)
        if x1 > x0 and y1 > y0 and ink[y0:y1, x0:x1].max() > 0:
            hit += 1
    return hit / (samples + 1)


def _ink_evidence(ink, s, samples=40, band=2):
    """返回 (覆盖率, 最长连续空白占比)。虚线是"覆盖率中等但缺口均匀"，假线是"整段空白"。
    只看覆盖率会把虚线误杀；加上最长缺口就能区分：
      实线  → (1.00, 0.00)
      虚线  → (0.76, 0.02)
      假线  → (0.05, 0.95)
    """
    H, W = ink.shape
    hits = []
    for k in range(samples + 1):
        t = k / samples
        x = s[0] + (s[2] - s[0]) * t
        y = s[1] + (s[3] - s[1]) * t
        x0, x1 = max(0, int(x - band)), min(W, int(x + band) + 1)
        y0, y1 = max(0, int(y - band)), min(H, int(y + band) + 1)
        hits.append(1 if (x1 > x0 and y1 > y0 and ink[y0:y1, x0:x1].max() > 0) else 0)
    n = len(hits)
    cov = sum(hits) / n
    doubled = hits + hits
    best = cur = 0
    for v in doubled:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return cov, min(best, n) / n


def _chroma_evidence(chroma_raw, s, samples=24, band=1, floor=11.0):
    """沿线段采样"真彩色"证据：返回落在色度 floor 以上的采样点比例。

    为什么需要：色度通道对淡彩线极敏感，但 JPEG 的 4:2:0 子采样会把 8×8 块
    做得花花绿绿，Canny 在这些块边界上同样能连出"直线"。用原始色度做二次校验，
    "一条线沿线都是红的/蓝的"才算数，块状伪影只是零散的点，过不了这一关。
    """
    if chroma_raw is None:
        return 1.0
    H, W = chroma_raw.shape
    hit = 0
    for k in range(samples + 1):
        t = k / samples
        x = s[0] + (s[2] - s[0]) * t
        y = s[1] + (s[3] - s[1]) * t
        x0, x1 = max(0, int(x - band)), min(W, int(x + band) + 1)
        y0, y1 = max(0, int(y - band)), min(H, int(y + band) + 1)
        if x1 > x0 and y1 > y0 and float(chroma_raw[y0:y1, x0:x1].max()) >= floor:
            hit += 1
    return hit / (samples + 1)


def _glyph_boxes(bw, D, max_boxes=64, return_counts=False):
    """找出"文字/数字"的包围盒（工程图上的尺寸数值、R 标注、φ 标注）。

    思路：线稿里真正的几何几乎都连着（一张图往往只有一两个大连通域），
    而文字是**孤立的小连通域**。于是取"面积够、尺寸小"的连通域当字形，
    再横向把挨得近的字连成一行（"R24" 三个字 → 一个框）。
    这些框里的短线段要被剔除，否则数字笔画会被当成一堆碎线画进 CAD。
    """
    if bw is None:
        return []
    limit = max(16.0, min(0.05 * D, 52.0))     # 单字最大边长（相对 + 绝对双限）
    n, _lab, stats, _cen = cv2.connectedComponentsWithStats(bw, 8)
    cells: List[Tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i][:5])
        if a < 10 or max(w, h) > limit or min(w, h) < 3:
            continue
        ar = w / float(h)
        if ar < 0.22 or ar > 4.5:              # 细长条（虚线段、短线头）不是字
            continue
        fill = a / float(w * h)
        if fill < 0.08 or fill > 0.80:         # 太空（框线）/ 太实（色块）都不是字
            continue
        cells.append((x, y, w, h))
    if not cells or len(cells) > 400:
        return []
    # 横向连字：同一行、水平间距小于一个字高的算一行（"R24" 三个字 → 一个框）
    cells.sort(key=lambda b: (b[1], b[0]))
    merged = [list(c) + [1] for c in cells]          # 第 5 位记"由几个字形拼成"
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                ah = max(a[3], b[3])
                vgap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
                hgap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
                if vgap <= 0.45 * ah and hgap <= 0.9 * ah:
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    if (x1 - x0) > 5.0 * limit or (y1 - y0) > 1.9 * limit:
                        continue                   # 连成一大片 → 不是一行字
                    merged[i] = [x0, y0, x1 - x0, y1 - y0,
                                 merged[i][4] + merged[j][4]]
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    merged = merged[:max_boxes]
    if return_counts:
        return [(int(x), int(y), int(w), int(h), int(c)) for (x, y, w, h, c) in merged]
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h, c) in merged]


def _in_glyph_boxes(s, boxes, pad=2.0):
    """线段两端都落在某个文字框内（且本身不长）→ 认为它是文字笔画。"""
    x1, y1, x2, y2 = s[0], s[1], s[2], s[3]
    L = _seg_len(s)
    for (bx, by, bw_, bh_) in boxes:
        if L > 1.9 * max(bw_, bh_) + 4:
            continue
        if (bx - pad <= x1 <= bx + bw_ + pad and by - pad <= y1 <= by + bh_ + pad and
                bx - pad <= x2 <= bx + bw_ + pad and by - pad <= y2 <= by + bh_ + pad):
            return True
    return False


def _rect_in_glyph_boxes(r, boxes, pad=2.0):
    """判断矩形是不是文字凑出来的假矩形。

    两条判据（任一命中即算）：
      1) 矩形整个落在文字框里；
      2) 矩形中心落在文字框里，且尺寸与文字框同量级（说明就是这几个字的外接框）。
    第 2 条是实测补上的：数字 "5" 的笔画能配出一个 9×7 的小方框，
    它的外接框比字形框大一圈，光用"完全包含"抓不住。
    """
    x, y, w, h = r
    cx, cy = x + w / 2.0, y + h / 2.0
    for (bx, by, bw_, bh_) in boxes:
        if (x >= bx - pad and y >= by - pad and
                x + w <= bx + bw_ + pad and y + h <= by + bh_ + pad):
            return True
        if (bx - pad <= cx <= bx + bw_ + pad and by - pad <= cy <= by + bh_ + pad
                and w <= 1.8 * bw_ + 6 and h <= 1.8 * bh_ + 6):
            return True
    return False


def detect_lines_multipass(gray, ink, D, passes=None, chroma=None, chroma_raw=None):
    """多遍直线检测：每遍换一套预处理/阈值，返回 [[seg, ...], ...]（按遍分组）。

    chroma 非空时追加两遍"只看色度"的检测（彩色标注线的召回主要靠它们）；
    灰度图/纯黑白图传 None，自动跳过，不浪费算力。
    """
    profs = _PASS_PROFILES[:max(1, passes)] if passes else list(_PASS_PROFILES)
    if chroma is None:
        profs = [p for p in profs if p[9] == "ink"]
    chans = {"ink": gray, "chroma": chroma if chroma is not None else gray}
    per_pass: List[List[tuple]] = []
    for prof in profs:
        edges_p = _pass_edges(chans, prof)
        raw = _hough_pass(edges_p, D, prof)
        if raw and prof[9] == "chroma" and chroma_raw is not None:
            # 色度遍只保留"沿线确实有颜色"的段，挡掉 JPEG 块状伪影连出的假线
            raw = [t for t in raw
                   if _chroma_evidence(chroma_raw, t) >= 0.70] or []
        if not raw:
            per_pass.append([])
            continue
        # 遍内先合并共线碎片（虚线/粗线双边缘），再交出去投票
        merged = _merge_collinear(raw, angle_tol=6.0,
                                 dist_tol=max(4.0, 0.006 * D),
                                 max_bridge=max(10.0, 0.035 * D))
        per_pass.append(merged)
    return per_pass


def fuse_line_passes(per_pass, ink, D, min_votes=2, max_out=400):
    """跨遍投票融合：把各遍的线段聚成簇，按票数 + 墨迹支撑度决定去留。

    返回 ([(seg, votes, support), ...], stats)：
      · votes ≥ min_votes             → 直接认（多套参数独立命中，假线概率极低）
      · votes == 1 但墨迹覆盖率 ≥0.9 且足够长 → 认（弱线只在灵敏遍出现也留）
      · votes == 1 且墨迹覆盖率低      → 丢（特异性过滤：阴影/网格/噪点）
    """
    n_pass = max(1, len(per_pass))
    pos_tol = max(4.0, 0.007 * D)
    ang_tol = 6.0
    bridge = max(12.0, 0.03 * D)
    min_len = max(8.0, 0.006 * D)

    clusters: List[Dict[str, Any]] = []
    for pid, segs in enumerate(per_pass):
        for s in sorted(segs, key=_seg_len, reverse=True):
            if _seg_len(s) < min_len:
                continue
            hit = None
            for cl in clusters:
                if pid in cl["passes"]:
                    continue                 # 同一遍只投一票
                rep = cl["rep"]
                if _ang_diff(_seg_angle(s), _seg_angle(rep)) > ang_tol:
                    continue
                if _perp_dist(s[0], s[1], rep) > pos_tol or \
                        _perp_dist(s[2], s[3], rep) > pos_tol:
                    continue
                if not _proj_overlap(rep, s, bridge):
                    continue
                hit = cl
                break
            if hit is None:
                clusters.append({"rep": s, "pts": [(s[0], s[1]), (s[2], s[3])],
                                 "passes": {pid}})
            else:
                hit["pts"].append((s[0], s[1]))
                hit["pts"].append((s[2], s[3]))
                hit["passes"].add(pid)
                hit["rep"] = _fit_line_pca(hit["pts"], hit["rep"])

    out: List[Tuple[tuple, int, float]] = []
    votes_hist: Dict[str, int] = {}
    weak_dropped = 0
    for cl in clusters:
        rep = cl["rep"]
        v = len(cl["passes"])
        L_ = _seg_len(rep)
        cov, gap = _ink_evidence(ink, rep)
        votes_hist[str(v)] = votes_hist.get(str(v), 0) + 1
        # 墨迹背书：要么覆盖率高，要么中等覆盖但缺口均匀（虚线）
        solid = cov >= 0.70 or (cov >= 0.50 and gap <= 0.25)
        if v >= min_votes and solid:
            out.append((rep, v, cov))
        elif cov >= 0.90 and L_ >= max(18.0, 0.02 * D):
            out.append((rep, v, cov))
        else:
            # 多票但线身上大段空白 → 圆心对侧连线、纹理交线之类的伪线
            weak_dropped += 1

    out.sort(key=lambda t: (-t[1], -_seg_len(t[0])))
    if len(out) > max_out:
        weak_dropped += len(out) - max_out
        out = out[:max_out]
    stats = {"passes": n_pass, "candidates": len(clusters), "kept": len(out),
             "weak_dropped": weak_dropped, "votes": votes_hist}
    return out, stats


# ---------------------------------------------------------------- 形状识别

def _detect_rects_from_contours(bw, edges, min_area_ratio=0.008):
    """从闭合轮廓找矩形（允许旋转），要求四边都有真实边缘支撑。"""
    H, W = bw.shape
    rects = []
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < min_area_ratio * W * H:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 60:
            continue
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(float)
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        w, h = xs[-1] - xs[0], ys[-1] - ys[0]
        if w < 15 or h < 15:
            continue
        # 直角校验（允许 ±15°）
        ok_ang = True
        for k in range(4):
            a, b, c = pts[k], pts[(k + 1) % 4], pts[(k + 2) % 4]
            v1 = (a[0] - b[0], a[1] - b[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 < 1e-6 or n2 < 1e-6:
                ok_ang = False
                break
            cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            if abs(cosv) > 0.26:
                ok_ang = False
                break
        if not ok_ang:
            continue
        # 四边支撑度：挡住"六边形被近似成四边形"的误检
        supports = [_edge_support(edges, pts[k], pts[(k + 1) % 4]) for k in range(4)]
        if min(supports) < 0.75:
            continue
        rects.append((xs[0], ys[0], w, h))
    return rects


def _detect_rects_from_lines(segs, edges, D, min_side=None, axis_tol=10.0):
    """草图矩形常画成 4 条不相接的线（轮廓法失效）——用直线配对补漏。
    要求：两横两竖、四条边都真实支撑、四角有边缘、横边完整跨越两竖边。"""
    if min_side is None:
        # 太窄的"矩形"多半是两条几乎重合的平行线被误配对（多遍检测后更易出现）
        min_side = max(20.0, 0.03 * D)
    tol = max(6.0, 0.008 * D)
    hs = [s for s in segs if _seg_angle(s) <= axis_tol or _seg_angle(s) >= 180 - axis_tol]
    vs = [s for s in segs if abs(_seg_angle(s) - 90.0) <= axis_tol]
    hs.sort(key=_seg_len, reverse=True)
    vs.sort(key=_seg_len, reverse=True)
    hs, vs = hs[:24], vs[:24]
    rects = []
    for i1 in range(len(hs)):
        for i2 in range(i1 + 1, len(hs)):
            h1, h2 = hs[i1], hs[i2]
            y1, y2 = (h1[1] + h1[3]) / 2, (h2[1] + h2[3]) / 2
            if abs(y1 - y2) < min_side:
                continue
            for j1 in range(len(vs)):
                for j2 in range(j1 + 1, len(vs)):
                    if len(rects) >= 8:
                        return rects
                    v1, v2 = vs[j1], vs[j2]
                    x1, x2 = (v1[0] + v1[2]) / 2, (v2[0] + v2[2]) / 2
                    if abs(x1 - x2) < min_side:
                        continue
                    xlo, xhi = min(x1, x2), max(x1, x2)
                    ylo, yhi = min(y1, y2), max(y1, y2)
                    # 横边须基本完整跨越两竖边之间（容忍手抖/端点内缩，
                    # 但挡不住"竖板右边+横板上下边"这种根本不搭边的误配对）
                    span_tol = max(15.0, 0.02 * D)
                    ok_h = True
                    for hh in (h1, h2):
                        hx_lo, hx_hi = min(hh[0], hh[2]), max(hh[0], hh[2])
                        if hx_lo > xlo + span_tol or hx_hi < xhi - span_tol:
                            ok_h = False
                            break
                    if not ok_h:
                        continue
                    ok_v = True
                    for vv in (v1, v2):
                        vy_lo, vy_hi = min(vv[1], vv[3]), max(vv[1], vv[3])
                        if vy_lo > ylo + span_tol or vy_hi < yhi - span_tol:
                            ok_v = False
                            break
                    if not ok_v:
                        continue
                    # 四角边缘支撑（挡住"圆的外接矩形"这类误检）
                    corners = [(xlo, ylo), (xhi, ylo), (xlo, yhi), (xhi, yhi)]
                    if not all(_corner_support(edges, cx, cy) for cx, cy in corners):
                        continue
                    # 四边支撑（虚线矩形也能过：阈值放低）
                    sides = [((xlo, ylo), (xhi, ylo)), ((xhi, ylo), (xhi, yhi)),
                             ((xhi, yhi), (xlo, yhi)), ((xlo, yhi), (xlo, ylo))]
                    if min(_edge_support(edges, a, b) for a, b in sides) < 0.5:
                        continue
                    rects.append((xlo, ylo, xhi - xlo, yhi - ylo))
    return rects


def _dedup_rects(rects, overlap=0.6):
    """重复矩形去重：位置重叠且尺寸相近才算重复（轮廓法与直线配对法常各检出一次）。
    注意：嵌套但尺寸不同的小矩形（如墙上的门）是真实特征，不能去重。"""
    kept: List[tuple] = []
    for r in sorted(rects, key=lambda r: -(r[2] * r[3])):
        x, y, w, h = r
        dup = False
        for k in kept:
            kx, ky, kw, kh = k
            ox = max(0.0, min(x + w, kx + kw) - max(x, kx))
            oy = max(0.0, min(y + h, ky + kh) - max(y, ky))
            inter = ox * oy
            dim_similar = (abs(w - kw) < 0.25 * max(w, kw)
                           and abs(h - kh) < 0.25 * max(h, kh))
            if dim_similar and inter > overlap * min(w * h, kw * kh):
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept


def _drop_split_rects(rects, tol=4.0):
    """去掉"被另一个矩形切出来的子块"。

    一个大框里画十字/分隔线时，直线配对法会把每个分区都配成矩形，
    于是一个框变成四块（DXF 里动一个框就散架）。判据：
      r 被 k 完整包含，且 r 与 k 共享 ≥2 条共线边 → r 是切分产物。
    真实的内嵌特征（墙上的门）通常只有 1 条边与外墙共线，不会被误删。
    """
    kept: List[tuple] = []
    for i, (x, y, w, h) in enumerate(rects):
        split = False
        for j, (kx, ky, kw, kh) in enumerate(rects):
            if i == j:
                continue
            if x < kx - tol or y < ky - tol or \
                    x + w > kx + kw + tol or y + h > ky + kh + tol:
                continue                     # r 没被 k 包住
            shared = 0
            if abs(x - kx) <= tol:
                shared += 1
            if abs((x + w) - (kx + kw)) <= tol:
                shared += 1
            if abs(y - ky) <= tol:
                shared += 1
            if abs((y + h) - (ky + kh)) <= tol:
                shared += 1
            if shared >= 2:
                split = True
                break
        if not split:
            kept.append((x, y, w, h))
    return kept


def _circle_support(edges, cx, cy, r, samples=64, band=3.0):
    """圆周边缘支持度：圆周上采样点附近有边缘的比例，用于过滤霍夫圆误报。"""
    H, W = edges.shape
    hit = 0
    for k in range(samples):
        a = 2 * math.pi * k / samples
        x, y = int(round(cx + r * math.cos(a))), int(round(cy + r * math.sin(a)))
        x0, x1 = max(0, x - int(band)), min(W, x + int(band) + 1)
        y0, y1 = max(0, y - int(band)), min(H, y + int(band) + 1)
        if x1 > x0 and y1 > y0 and edges[y0:y1, x0:x1].max() > 0:
            hit += 1
    return hit / samples


def _circle_longest_arc(edges, cx, cy, r, samples=72, band=2):
    """圆周上"最长连续命中弧"占整圈的比例。

    这是抗网格/抗噪点的关键判据：真圆的圆周是连续墨迹，最长连续弧接近 1.0；
    坐标纸格子、纹理、线条交叉处凑出来的假圆只在零星角度碰巧压到边缘，
    最长连续弧通常 < 0.1。逐点支持度对这类假圆无效（到处都有边缘），
    但"连续"二字能把它们区分开。"""
    H, W = edges.shape
    hit = bytearray(samples)
    for k in range(samples):
        a = 2 * math.pi * k / samples
        x, y = int(round(cx + r * math.cos(a))), int(round(cy + r * math.sin(a)))
        x0, x1 = max(0, x - int(band)), min(W, x + int(band) + 1)
        y0, y1 = max(0, y - int(band)), min(H, y + int(band) + 1)
        if x1 > x0 and y1 > y0 and edges[y0:y1, x0:x1].max() > 0:
            hit[k] = 1
    n = sum(hit)
    if n == 0:
        return 0.0
    # 环形最长连续段（允许跨 0 点）
    doubled = list(hit) + list(hit)
    best = cur = 0
    for v in doubled:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return min(best, samples) / samples


def _refine_circle_radius(edges, cx, cy, r, search=10.0, angles=48):
    """用圆周附近的边缘像素距离中位数精修半径。
    霍夫圆对粗线圆往往估偏（如 5px 线宽的 r=60 圆估成 55），精修后回到线宽中心。"""
    H, W = edges.shape
    per_ray = []
    for k in range(angles):
        a = 2 * math.pi * k / angles
        ca, sa = math.cos(a), math.sin(a)
        ds = []
        for d in range(-int(search), int(search) + 1):
            x = int(round(cx + (r + d) * ca))
            y = int(round(cy + (r + d) * sa))
            if 0 <= x < W and 0 <= y < H and edges[y, x] > 0:
                ds.append(r + d)
        if ds:
            ds.sort()
            per_ray.append(ds[len(ds) // 2])      # 每射线取中值，抗线条穿越干扰
    if not per_ray:
        return r
    per_ray.sort()
    return per_ray[len(per_ray) // 2]


def _circle_radial_consistency(edges, cx, cy, r, angles=36, search=12.0):
    """径向一致性：真圆在每个角度上到边缘的距离都≈r（标准差很小）；
    内切于多边形的假圆则时远时近。返回标准差占半径的比例。"""
    H, W = edges.shape
    ds = []
    for k in range(angles):
        a = 2 * math.pi * k / angles
        ca, sa = math.cos(a), math.sin(a)
        best = None
        for d in range(-int(search), int(search) + 1):
            x = int(round(cx + (r + d) * ca))
            y = int(round(cy + (r + d) * sa))
            if 0 <= x < W and 0 <= y < H and edges[y, x] > 0:
                if best is None or abs(d) < abs(best):
                    best = d
        if best is not None:
            ds.append(best)
    if len(ds) < angles * 0.6:
        return 1.0                        # 边缘太少，不可信
    mean = sum(ds) / len(ds)
    var = sum((d - mean) ** 2 for d in ds) / len(ds)
    return math.sqrt(var) / max(r, 1.0)


def _circle_tangential(edges, cx, cy, r, samples=48, step=6.0, band=3.0):
    """切向连续性：真圆的边缘沿切线方向连续延伸；多边形顶点/线条交叉处则不然。
    用于挡掉"内切于多边形"的假圆（如五角星中心区域）。"""
    H, W = edges.shape
    good = 0
    for k in range(samples):
        a = 2 * math.pi * k / samples
        ca, sa = math.cos(a), math.sin(a)
        tx, ty = -sa, ca
        bx, by = cx + r * ca, cy + r * sa

        def probe(px, py):
            x0, x1 = max(0, int(px - band)), min(W, int(px + band) + 1)
            y0, y1 = max(0, int(py - band)), min(H, int(py + band) + 1)
            if x1 <= x0 or y1 <= y0:
                return False
            return bool(edges[y0:y1, x0:x1].max() > 0)

        c0 = probe(bx, by)
        c1 = probe(bx + tx * step, by + ty * step)
        c2 = probe(bx - tx * step, by - ty * step)
        if c0 and (c1 or c2):
            good += 1
    return good / samples


# 圆弧判据（在墨迹图上量，不是边缘图）
_ARC_FULL_SPAN = 0.90        # 最长连续弧 ≥90% 整圈才算"整圆"，否则它是圆弧
_ARC_MIN_SPAN = 0.17         # 连续弧不足 17%（约 60°）当噪声扔掉，不值得画
_ARC_MAX_SCATTER = 0.30      # (覆盖比例 - 最长连续弧) 超过它说明命中是散的，不是一段弧


def _fill_small_gaps(hit, gap_tol):
    """把长度 ≤ gap_tol 的空洞补上（环状，允许跨 0 点）。

    中心线/尺寸线压过圆弧、JPEG 断层都会在圆周上打几个洞，不补的话一条完整的
    圆弧会被切成两段，"连续弧占比"直接腰斩。真缺口（弧真的在这儿结束）远长于
    gap_tol，不会被补掉。
    """
    n = len(hit)
    if n == 0 or gap_tol <= 0:
        return bytearray(hit)
    out = bytearray(hit)
    if all(out):                       # 整圈都命中，无洞可补
        return out
    # 从一个空点出发绕一圈，逐段处理
    start = out.index(0)
    k = 0
    while k < n:
        i = (start + k) % n
        if out[i] == 0:
            j = 0
            while j < n and out[(i + j) % n] == 0:
                j += 1
            if j <= gap_tol and j < n:
                for t in range(j):
                    out[(i + t) % n] = 1
            k += j
        else:
            k += 1
    return out


def _longest_run(hit):
    """环状最长连续 1 段 → (起始下标, 长度)。"""
    n = len(hit)
    if n == 0:
        return 0, 0
    if all(hit):
        return 0, n
    best_s = best_l = 0
    for s in range(n):
        if hit[s] and not hit[(s - 1) % n]:        # 只从每段的起点开始数
            l = 0
            while l < n and hit[(s + l) % n]:
                l += 1
            if l > best_l:
                best_s, best_l = s, l
    return best_s, best_l


def _ink_ray_run(dil, cx, cy, r, ca, sa, scan=10):
    """沿半径方向从 r-scan 扫到 r+scan，找最靠近 r 的那段墨迹。

    dil 是已经做过 3×3 膨胀的墨迹图（只查单点，不再逐点切 numpy 窗口 —— 霍夫一张
    图会给出上千个候选，逐点 slicing 会让整张图从 2 秒变成 40 秒）。

    返回 (中心偏移, 厚度) 或 None。这是"圆到底压没压在一条细线上"的判据：
    真圆弧的墨迹在半径方向只有线宽那么厚（且中心就落在 r 上）；文字、交叉线、
    色块凑出来的假圆，射线扫过去是一大片甚至整段都是墨迹 —— 厚度会爆。
    """
    H, W = dil.shape[:2]
    vals = []
    for d in range(-scan, scan + 1):
        rr = r + d
        x = int(math.floor(cx + rr * ca + 0.5))
        y = int(math.floor(cy + rr * sa + 0.5))
        vals.append(1 if (0 <= x < W and 0 <= y < H and dil[y, x]) else 0)
    runs = []
    i, n = 0, len(vals)
    while i < n:
        if vals[i]:
            j = i
            while j < n and vals[j]:
                j += 1
            runs.append((i - scan, j - 1 - scan))
            i = j
        else:
            i += 1
    if not runs:
        return None
    for (a, b) in runs:                     # 优先取跨过 0 的那段
        if a <= 0 <= b:
            return ((a + b) / 2.0, b - a + 1)
    best = min(runs, key=lambda ab: abs((ab[0] + ab[1]) / 2.0))
    return ((best[0] + best[1]) / 2.0, best[1] - best[0] + 1)


def _circle_ink_profile(dil, edges, cx, cy, r, samples=180, band=3, gap_tol=3,
                        step=6.0, max_off=3.0, scan=10):
    """沿圆周采样，量出"这段圆周上到底有哪一段真的有墨迹"，并给出弧的起止角。

    返回 None 或 dict：
      cover / span / start / end（角度按图像坐标，y 向下）/ scatter / off

    为什么不用现成的 _circle_longest_arc：那个跑在 Canny 边缘（1px）上，只够回答
    "像不像圆"；判断"整圆还是圆弧"必须在清洗后的粗墨迹上量，否则线宽、JPEG 断层、
    尺寸线压线都会把整圆误判成圆弧。

    四道闸门：
      1) 每个采样点必须在**半径方向**压到一段"细"墨迹（厚度 ≤ 线宽量级）且中心
         落在 r±max_off 内 —— 这是挡"圈在文字/交叉处"的关键，只看到"附近有墨迹"
         会把整张图的密集区都圈一遍；
      2) 弧段长度 ≥ _ARC_MIN_SPAN，否则是零碎命中；
      3) scatter（覆盖比例 − 最长连续弧）≤ _ARC_MAX_SCATTER：真弧是"一段连着的墨迹"；
      4) 切向连续 + 径向一致：真弧沿切线方向延伸、到圆心距离稳定。
    """
    H, W = dil.shape[:2]
    if r < 3:
        return None
    hit = bytearray(samples)
    offs = []
    t_max = max(8.0, 0.10 * r)              # 允许的墨迹厚度（3x3 膨胀会胖 2px）
    for k in range(samples):
        a = 2 * math.pi * k / samples
        run = _ink_ray_run(dil, cx, cy, r, math.cos(a), math.sin(a), scan=scan)
        if run is None:
            continue
        off, thick = run
        if abs(off) <= max_off and thick <= t_max:
            hit[k] = 1
            offs.append(off)
    if not any(hit):
        return None
    cover = sum(hit) / samples
    filled = _fill_small_gaps(hit, gap_tol)
    cover_f = sum(filled) / samples
    s_idx, length = _longest_run(filled)
    span = length / samples
    if span < _ARC_MIN_SPAN:
        return None
    if (cover_f - span) > _ARC_MAX_SCATTER:
        return None                     # 命中是散的，不是一段弧

    # ---- 只在弧段内验证几何质量 ----
    good = tot = 0
    ds = []
    for t in range(length):
        k = (s_idx + t) % samples
        a = 2 * math.pi * k / samples
        ca, sa = math.cos(a), math.sin(a)
        bx, by = cx + r * ca, cy + r * sa

        def probe(px, py):
            x0, x1 = max(0, int(px - band)), min(W, int(px + band) + 1)
            y0, y1 = max(0, int(py - band)), min(H, int(py + band) + 1)
            if x1 <= x0 or y1 <= y0:
                return False
            return bool(edges[y0:y1, x0:x1].max() > 0)

        tot += 1
        if probe(bx, by) and (probe(bx - sa * step, by + ca * step)
                              or probe(bx + sa * step, by - ca * step)):
            good += 1                    # 沿切线前后还有墨迹 → 真弧
        # 径向偏移：在弧段内找最近的边缘点
        best = None
        for d in range(-9, 10):
            x = int(round(cx + (r + d) * ca))
            y = int(round(cy + (r + d) * sa))
            if 0 <= x < W and 0 <= y < H and edges[y, x] > 0:
                if best is None or abs(d) < abs(best):
                    best = d
        if best is not None:
            ds.append(best)
    if tot >= 6 and good / tot < 0.55:
        return None
    if len(ds) >= max(5, int(tot * 0.5)):
        mean = sum(ds) / len(ds)
        sd = math.sqrt(sum((d - mean) ** 2 for d in ds) / len(ds))
        if sd / max(r, 1.0) > 0.12:
            return None

    a_start = 360.0 * s_idx / samples
    a_end = 360.0 * (s_idx + length - 1) / samples
    med = 0.0
    if offs:
        offs.sort()
        med = abs(offs[len(offs) // 2])
    return {"cover": cover, "span": span, "start": a_start, "end": a_end,
            "scatter": cover_f - span, "off": med}



def _dedup_by_circle(items):
    """同心同径的去重（items 是 (cx,cy,r,...) 元组，按半径从大到小保留）。"""
    items = sorted(items, key=lambda c: -c[2])
    kept = []
    for c in items:
        dup = False
        for k in kept:
            d = math.hypot(c[0] - k[0], c[1] - k[1])
            big = max(c[2], k[2])
            if d < big * 0.3 and abs(c[2] - k[2]) < big * 0.3:
                dup = True
                break
            if d < 0.30 * big and abs(c[2] - k[2]) < 0.30 * big:
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


def _refine_radius_ink(dil, cx, cy, r, angles=72, scan=14):
    """用墨迹在半径方向上"找中线"来精修半径。

    为什么不用 _refine_circle_radius（边缘版）：粗线（3~5px）会产生内、外两条
    Canny 边缘，那个函数取的是中位元素 —— 两条时等于取外侧那条，半径稳定偏大
    半个线宽。半径偏 2~3px，"这一圈到底覆盖多少角度"就会从 360° 掉到 230°，
    整圆被误判成圆弧。这里改成：每条射线找包含/最靠近当前半径的那段墨迹，取
    段中心，再取全体中位数。

    返回 (radius, support)，support 是有效射线的比例（太低说明这个候选不靠谱）。
    """
    centers = []
    for k in range(angles):
        a = 2 * math.pi * k / angles
        run = _ink_ray_run(dil, cx, cy, r, math.cos(a), math.sin(a), scan=scan)
        if run is None:
            continue
        off, thick = run
        if thick <= max(8.0, 0.12 * r) and abs(off) <= scan * 0.6:
            centers.append(off)
    if len(centers) < angles * 0.25:
        return r, 0.0
    centers.sort()
    med = centers[len(centers) // 2]
    keep = sorted(c for c in centers if abs(c - med) <= 4.5)
    if not keep:
        return r, 0.0
    return r + keep[len(keep) // 2], len(keep) / angles


def _detect_circles(gray, D, edges=None, ink=None):
    """霍夫圆（分半径波段迭代 + 已检出圆环带掩蔽，解决同心圆被 minDist 压制的问题）
    + 墨迹弧段验证 + 切向/径向验证 + 半径精修 + 去重。

    返回 (circles, arcs)：
      circles = [(cx, cy, r)]                      整圆
      arcs    = [(cx, cy, r, a0_deg, a1_deg)]      圆弧（角度按图像坐标，顺时针增）

    早先这里只有"圆"一种产物，于是被霍夫拟合成圆的一段圆弧（吊钩的 R29、法兰的
    缺口弧）会被当成整圆画出来 —— 图上凭空多出一个不合法的整圆。现在在墨迹图上量
    圆周的连续覆盖：≥90% 才是整圆，否则输出圆弧实体（DXF ARC），不足 60° 直接丢。
    """
    circles: List[Tuple[float, float, float]] = []
    arcs: List[Tuple[float, float, float, float, float]] = []
    H, W = gray.shape[:2]
    if edges is None:
        edges = cv2.Canny(gray, 50, 150)
    # 墨迹做一次 3×3 膨胀，后面所有"沿半径找墨迹"都查这张图（单点查，不切片）：
    # 霍夫一张图能吐上千个候选，逐点切 numpy 窗口会让 2 秒变成 40 秒。
    dil = cv2.dilate(ink, np.ones((3, 3), np.uint8)) if ink is not None else None
    rmin = max(22, int(0.02 * D))
    bands = [(int(0.15 * D), int(0.45 * D)),
             (rmin, int(0.16 * D)),
             (rmin, int(0.08 * D))]
    for (lo, hi) in bands:
        if hi <= lo:
            continue
        work = edges.copy()
        for (cx, cy, r) in circles:                 # 只掩已检出圆的环带，不毁圆内结构
            cv2.circle(work, (int(cx), int(cy)), int(r), 0, 18)
        try:
            found = cv2.HoughCircles(
                work, cv2.HOUGH_GRADIENT, dp=1.5,
                minDist=max(15, int(0.02 * D)),
                param1=120, param2=25,
                minRadius=lo, maxRadius=hi)
        except cv2.error:
            continue
        if found is None:
            continue
        for (cx, cy, r) in found[0]:
            cx, cy, r = float(cx), float(cy), float(r)
            # ---- 第一关：够不够格（原有四道边缘判据，几百张图调出来的，不动）----
            strict = (
                _circle_support(edges, cx, cy, r) >= 0.75
                and _circle_longest_arc(edges, cx, cy, r) >= 0.28
                and _circle_tangential(edges, cx, cy, r) >= 0.6
                and _circle_radial_consistency(edges, cx, cy, r) <= 0.10)
            if ink is None:                      # 没有墨迹图：只认整圆，行为同旧版
                if not strict:
                    continue
                r2 = _refine_circle_radius(edges, cx, cy, r)
                if rmin * 0.8 <= r2 <= 0.5 * D:
                    circles.append((cx, cy, r2))
                continue
            if not strict:
                # 便宜的预筛（少采样、原半径）：挡掉图上根本不存在的候选，别让上千个
                # 假候选都去做精细测量。
                quick = _circle_ink_profile(dil, edges, cx, cy, r, samples=72)
                if quick is None or quick["span"] < 0.22:
                    continue
            # ---- 半径修准：霍夫给的半径常偏 3~6%，半径偏 3px 就能把整圆量成 230°，
            #      于是整圆被误判成圆弧。墨迹版取"墨迹中线"，比边缘版准（边缘版在粗线
            #      上稳定偏大半个线宽）。量不出东西时退回原始半径。----
            r2, _sup = _refine_radius_ink(dil, cx, cy, r)
            if not (rmin * 0.8 <= r2 <= 0.5 * D):
                continue
            prof = _circle_ink_profile(dil, edges, cx, cy, r2)
            if prof is None and abs(r2 - r) > 0.5:
                prof = _circle_ink_profile(dil, edges, cx, cy, r)
                r2 = r
            if prof is None:
                continue
            # ---- 第二关：整圆还是圆弧 ----
            if prof["span"] >= _ARC_FULL_SPAN and prof["cover"] >= 0.85:
                circles.append((cx, cy, r2))
            elif (strict and prof["span"] >= 0.20) \
                    or (prof["span"] >= 0.28 and prof["scatter"] <= 0.12
                        and prof["off"] <= 1.2):
                # 第二档只为"圆弧"开门：干净图纸上的圆弧只有 200~270° 有边缘，边缘
                # 支持度天然到不了 0.75，strict 会把它整个拒掉；但既然它确实是一段
                # 压在细线上的连续墨迹（够长、不散、正中），就该画成圆弧。
                # 不是整圆就落到这里 —— 绝不允许"不确定就画一个整圆"。
                arcs.append((cx, cy, r2, prof["start"], prof["end"]))
    # 去重：同心同径的重复检出只留一个；但同心不同半径（齿轮/轴孔）必须保留
    kept = _dedup_by_circle(circles)
    kept_arcs = _dedup_by_circle(arcs)
    # 与已确认的整圆重合的弧丢掉（同一段轮廓不该同时有圆和弧）
    out_arcs = []
    for a in kept_arcs:
        if any(math.hypot(a[0] - c[0], a[1] - c[1]) < max(a[2], c[2]) * 0.25
               and abs(a[2] - c[2]) < max(a[2], c[2]) * 0.25 for c in kept):
            continue
        out_arcs.append(a)
    return kept, out_arcs


def _circle_iou(d, r1, r2):
    """两圆相交面积 / 并集面积（用来判断是不是同一段弧被重复拟合）。"""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        small, big = min(r1, r2), max(r1, r2)
        return (small * small) / (big * big)
    a1 = math.acos(max(-1.0, min(1.0, (d * d + r1 * r1 - r2 * r2) / (2 * d * r1))))
    a2 = math.acos(max(-1.0, min(1.0, (d * d + r2 * r2 - r1 * r1) / (2 * d * r2))))
    inter = r1 * r1 * (a1 - math.sin(2 * a1) / 2) + r2 * r2 * (a2 - math.sin(2 * a2) / 2)
    union = math.pi * (r1 * r1 + r2 * r2) - inter
    return inter / union if union > 1e-9 else 0.0


def _drop_arc_fragments(segs, circles, D):
    """剔除"其实是圆弧一部分"的直线碎片。

    多遍低阈值会把圆弧切成一段段弦（草图上很常见），它们在 DXF 里表现为
    绕着圆心的杂乱短直线。旧判定要求"两端都贴圆 + 线垂直于半径"，大跨度弦
    （弦切角 < 90°）会漏网，于是改用沿线密采样：
      · 采样点里 ≥80% 落在圆环带内  → 整段都贴着圆周，是弧碎片
      · 径向辐条只有 1~2 个采样点贴圆 → 不会被误删
      · 贯穿圆的弦大部分采样点远离圆周 → 保留
    """
    if not circles:
        return segs
    band = max(8.0, 0.012 * D)
    kept = []
    samples = 8
    for s in segs:
        L = _seg_len(s)
        drop = False
        for (cx, cy, r) in circles:
            if L > 2 * math.pi * r * 0.45:      # 太长的不可能是某段圆弧
                continue
            near = 0
            for k in range(samples + 1):
                t = k / samples
                x = s[0] + (s[2] - s[0]) * t
                y = s[1] + (s[3] - s[1]) * t
                if abs(math.hypot(x - cx, y - cy) - r) <= band:
                    near += 1
            if near / (samples + 1) >= 0.8:
                drop = True
                break
            # 大跨度弦（圆心角 40°~90°）的中点明显向内偏，采样阈值压不住，
            # 用"两端贴圆 + 中点内凹"补一刀
            d1 = math.hypot(s[0] - cx, s[1] - cy)
            d2 = math.hypot(s[2] - cx, s[3] - cy)
            dm = math.hypot((s[0] + s[2]) / 2 - cx, (s[1] + s[3]) / 2 - cy)
            if abs(d1 - r) <= band and abs(d2 - r) <= band \
                    and (r - band * 2.2) <= dm <= (r - 1.0):
                drop = True
                break
            # "圆周上取对侧两点连成的假直径"：两端贴圆、长度≈2r、中点落在圆心附近。
            # 真辐条会一路穿到外圆（长度远大于 2r），不受影响。
            if abs(d1 - r) <= band and abs(d2 - r) <= band and L <= 2 * r + band:
                drop = True
                break
        if not drop:
            kept.append(s)
    return kept


def _drop_hanging_chords(segs, bw, D):
    """剔除"中点悬空"的桥接弦（假直线）。

    霍夫的共线投票会把缓弯中段"抄近路"：两端压在墨迹上、中间穿过弯曲区
    （墨迹在那里已经弯走了），DXF 里表现为一条斜穿曲线的假长直线。
    真直边（实线）的中点必然压墨——霍夫线段本来就是沿墨迹投出来的；
    虚线的每一段也各自压墨。所以"长弦中点 5×5 邻域无墨"即可判假。
    只查长弦（> 0.05×图幅），短段没有桥接价值，不动。
    """
    if bw is None or not hasattr(bw, "shape") or not segs:
        return segs
    dil = cv2.dilate((bw > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    H, W = dil.shape[:2]
    min_len = max(20.0, 0.05 * D)
    out = []
    for s in segs:
        if _seg_len(s) > min_len:
            mx, my = int(round((s[0] + s[2]) / 2)), int(round((s[1] + s[3]) / 2))
            if 0 <= mx < W and 0 <= my < H:
                if not dil[max(0, my - 2):my + 3, max(0, mx - 2):mx + 3].any():
                    continue                    # 中点悬空 → 桥接弦，删
        out.append(s)
    return out


# ---------------------------------------------------------------- 坐标变换

def _parse_scale_from_text(text: str) -> Optional[float]:
    """从指令里解析显式尺寸（如 '100mm'、'边长50'），用作比例尺。"""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|毫米|厘米|cm)\b", text, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        if "cm" in m.group(0).lower() or "厘米" in m.group(0):
            v *= 10.0
        return v
    m = re.search(r"(?:边长|直径|半径|宽度|高度|长|宽|高)\s*[为是]?\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------- 颜色过滤

# OpenCV HSV：H ∈ [0,180)，S,V ∈ [0,255]
# 每个颜色 = 若干 (H下界,H上界) 区间；特殊颜色（黑/白/灰）按 S/V 判定
_COLOR_RANGES = {
    "红": [(0, 10), (170, 180)],
    "橙": [(8, 22)],
    "黄": [(22, 38)],
    "绿": [(38, 85)],
    "青": [(85, 100)],
    "蓝": [(100, 130)],
    "紫": [(130, 160)],
    "粉": [(160, 172)],
    "棕": [(8, 25)],
}
_COLOR_ALIASES = {
    "红色": "红", "red": "红", "Red": "红", "RED": "红",
    "橙色": "橙", "orange": "橙",
    "黄色": "黄", "yellow": "黄",
    "绿色": "绿", "green": "绿",
    "青色": "青", "cyan": "青",
    "蓝色": "蓝", "blue": "蓝", "Blue": "蓝", "BLUE": "蓝",
    "紫色": "紫", "purple": "紫",
    "粉色": "粉", "pink": "粉",
    "棕色": "棕", "brown": "棕",
    "黑色": "黑", "black": "黑", "Black": "黑", "BLACK": "黑",
    "白色": "白", "white": "白", "White": "白", "WHITE": "白",
    "灰色": "灰", "gray": "灰", "grey": "灰",
}

# 排除类动词 + 保留类动词（单字"去/留"也认，覆盖"去红留黑"这类简写）
_DROP_WORDS = ["去掉", "去除", "删除", "剔除", "删掉", "滤掉", "滤除",
               "排除", "忽略", "擦除", "涂掉", "遮住", "遮挡", "不要", "清掉",
               "抹掉", "过滤掉", "去"]
_KEEP_WORDS = ["保留", "留下", "只留", "仅留", "仅保留", "只要", "留着", "保留住",
               "只输出", "仅输出", "只画", "仅画", "只生成", "仅生成", "保留着", "留"]

# 全部颜色词（长词优先，避免"红色"被单字"红"抢先匹配后语境丢失）
_COLOR_WORDS = ["红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "粉色",
                "棕色", "黑色", "白色", "灰色",
                "red", "orange", "yellow", "green", "cyan", "blue", "purple",
                "pink", "brown", "black", "white", "gray", "grey",
                "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple",
                "Pink", "Brown", "Black", "White", "Gray",
                "RED", "ORANGE", "YELLOW", "GREEN", "CYAN", "BLUE", "PURPLE",
                "PINK", "BROWN", "BLACK", "WHITE", "GRAY",
                "红", "橙", "黄", "绿", "青", "蓝", "紫", "粉", "棕", "黑", "白", "灰"]

_ALL_COLORS = set(_COLOR_RANGES.keys()) | {"黑", "白", "灰"}


def _canon_color(word: str):
    return _COLOR_ALIASES.get(word, word if word in _ALL_COLORS else None)


def parse_color_filter(instruction: str):
    """从自然语言指令解析颜色过滤意图。

    返回 (keep, drop)：
      keep：只保留这些颜色（用户说"保留/只留 X"）
      drop：去掉这些颜色（用户说"去掉/不要 X"）
    两者可同时出现（"去掉红色，只留黑色" → keep={'黑'}, drop={'红'}）。

    关键规则：颜色词归属"离它最近"的动词——"去掉红色只留黑色"中"黑"
    离"只留"更近，判保留，不会被句首的"去掉"误伤；
    同时识别后置动词（"把红色去掉"）与"除了X都去掉"句式。
    """
    if not instruction:
        return set(), set()
    keep, drop = set(), set()
    # "除了X（的）…" → 保留 X（常配合"（其他）都去掉"）
    except_keep = set()
    for w in _COLOR_WORDS:
        c = _canon_color(w)
        if not c:
            continue
        for m in re.finditer(r"除了[色的]{0,3}" + re.escape(w) + r"(?![a-zA-Z])",
                             instruction):
            except_keep.add(c)
    keep |= except_keep
    if except_keep and re.search(
            r"(都|全)[^。；;，,]{0,4}(去掉|去除|删除|删掉|剔除|不要)", instruction):
        drop |= (_ALL_COLORS - except_keep)
    # 按标点分句；每句内的颜色词归属"离它最近"的动词（动词可在前也可在后）。
    # 这样"画一个红色圆圈，去掉蓝色辅助线"里的"红"不会被后半句的"去掉"误伤，
    # "去掉红色只留黑色"里的"黑"也只会归属更近的"只留"。
    for clause in re.split(r"[。；;，,．.！!？?\n\r\t]+", instruction):
        if not clause:
            continue
        occ = []          # [(起点, 终点, 颜色)]
        for w in _COLOR_WORDS:
            c = _canon_color(w)
            if not c or any(c == cc for _, _, cc in occ):
                continue
            idx = clause.find(w)
            if idx != -1:
                occ.append((idx, idx + len(w), c))
        if not occ:
            continue
        verbs = []        # [(起点, 终点, drop/keep)]
        for words, kind in ((_DROP_WORDS, "drop"), (_KEEP_WORDS, "keep")):
            for vw in words:
                j = clause.find(vw)
                while j != -1:
                    verbs.append((j, j + len(vw), kind))
                    j = clause.find(vw, j + 1)
        if not verbs:
            continue
        for pos, end, c in occ:
            if c in except_keep:
                continue
            best = None
            for vs, ve, kind in verbs:
                dist = pos - ve if ve <= pos else vs - end
                if dist < 0:
                    dist = 0
                if best is None or dist < best[0]:
                    best = (dist, kind)
            if best[1] == "drop":
                drop.add(c)
            else:
                keep.add(c)
    return keep, drop


def _color_hit(hsv_px: int, s_px: int, v_px: int, color: str) -> bool:
    """判断单个像素（H,S,V）是否属于某颜色。"""
    if color in _COLOR_RANGES:
        return any(lo <= hsv_px <= hi for lo, hi in _COLOR_RANGES[color])
    if color == "黑":
        return v_px < 90 and s_px < 170            # 暗像素（允许低饱和的深色）
    if color == "白":
        return s_px < 60 and v_px > 170
    if color == "灰":
        return s_px < 60 and 90 <= v_px <= 170
    return False


def _edge_halo(v, bg_v, dark=False, iters=2):
    """暗线条两侧的抗锯齿/压缩过渡带（比线本身宽 1~2px 的那圈"中间调"）。

    用于"只留灰"这类过滤：过渡带亮度恰好落在灰区间里，不挖掉就会把黑轮廓
    复制成一圈浅灰线。
    """
    core = cv2.inRange(v, np.array(0, np.uint8),
                       np.array(int(max(30, 0.45 * bg_v)), np.uint8)) if not dark else \
        cv2.inRange(v, np.array(int(min(240, 1.35 * bg_v)), np.uint8), np.array(255, np.uint8))
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(core, k, iterations=iters)


def color_filter_gray(bgr, keep, drop):
    """按颜色过滤线条：把非目标颜色的内容涂成背景(白)，目标颜色保留为深色。

    返回与输入同尺寸的灰度图（0=目标线条，255=背景），可直接喂给 _binarize/Canny。
    keep/drop 均为空时返回 None（不过滤）。
    规则：目标 = keep（若非空，否则全部颜色） - drop。

    两个容易踩的坑（都按"扫描件/截图的真实像素"处理，不按理想色处理）：
      1) 淡彩线：JPEG 压缩 + 抗锯齿会把红线压成 (216,216,240) 这种"淡粉"，
         饱和度只有 20~60。用 S≥60 的门槛会把它们整体丢掉 → 这里放宽到 S≥18，
         再用色相区间判定，仍能把中性灰（S≈0）排除在外。
      2) 只给 drop（"去掉红色"）时，目标集合会包含"白"——而白就是纸，
         整张背景会变成"线条"（二值化后 97% 是墨迹，直接判非线稿失败）。
         亮底图必须把"白"从目标里摘掉；暗底图才保留（那时白=亮色线条）。
    """
    if not keep and not drop:
        return None
    if bgr is None or bgr.ndim != 3:
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    bg_v = float(np.median(v))
    dark_bg = bg_v < 128.0        # 暗底图（如 LibreCAD 黑底截图）
    # 消色图：彩度像素占比 <0.2%，说明这张图已经没有颜色信息了 ——
    # 可能是 bw_photo 压过的纯黑白图，也可能本来就是灰度扫描件。
    # 这种图上"白"只可能是纸（背景），绝不能当"白线"：否则（典型场景：
    # 暗底白线图先被 bw_photo 翻成白底黑线）整片背景会被判成目标色、
    # 图像反转 → 二值化后 0 墨迹 → 一条线都认不出来。
    achromatic = float((s >= 18).mean()) < 0.002

    all_colors = set(_COLOR_RANGES.keys()) | {"黑", "白", "灰"}

    def _canon_set(cs):
        """英文/大小写别名统一成内部色名（网页端可能传 red / Red / 红）。"""
        out = set()
        for c in cs or ():
            out.add(_canon_color(str(c)) or str(c))
        return out

    keep = _canon_set(keep)
    drop = _canon_set(drop)
    colors = set(keep) if keep else set(all_colors)
    if not keep and not dark_bg:
        colors.discard("白")      # 亮底：白是纸，不是线
    if achromatic:
        colors.discard("白")      # 灰度图里没有"白线"，白就是纸
    colors -= set(drop)
    if not colors:
        # 目标集合被掏空（典型：黑白图上"只留白"）→ 不过滤，直接用原图识别。
        # 这比"整张反转导致 0 图元"正确得多：bw_photo 已经把白线转成黑线了。
        return None

    def _mask(color):
        if color in _COLOR_RANGES:
            m = np.zeros(h.shape, dtype=np.uint8)
            for lo, hi in _COLOR_RANGES[color]:
                m |= cv2.inRange(h, np.array(lo, np.uint8), np.array(hi, np.uint8))
            # 饱和度门槛放到 18：专收"淡彩线"（实测淡红线 S≈26），中性灰仍被排除
            m &= cv2.inRange(s, np.array(18, np.uint8), np.array(255, np.uint8))
            m &= cv2.inRange(v, np.array(38, np.uint8), np.array(255, np.uint8))
            return m
        if color == "黑":
            if dark_bg:
                # 暗底下"黑线"与背景无法区分——放宽为"消色且亮度明显偏离背景"的线条
                achroma = cv2.inRange(s, np.array(0, np.uint8), np.array(60, np.uint8))
                diff = cv2.absdiff(v, np.full_like(v, int(round(bg_v))))
                return achroma & cv2.inRange(diff, np.array(45, np.uint8),
                                             np.array(255, np.uint8))
            # 亮底：按"相对纸色有多暗"判定，比固定 v<90 更适应灰底扫描件
            lim = int(max(60.0, min(150.0, 0.62 * bg_v)))
            return cv2.inRange(v, np.array(0, np.uint8), np.array(lim, np.uint8))
        if color == "白":
            return cv2.inRange(s, np.array(0, np.uint8), np.array(45, np.uint8)) & \
                cv2.inRange(v, np.array(max(140, int(0.9 * bg_v)), np.uint8),
                            np.array(255, np.uint8))
        if color == "灰":
            # 灰 = 消色 + 明显暗于纸但不至于黑。用相对区间，浅灰引线(216/240)才算得进来。
            # 必须再挖掉"黑线的抗锯齿过渡带"：黑线两侧 1~2px 的像素亮度正好落在灰区间里，
            # 不挖的话"只留灰"会把整圈黑轮廓复制成两圈浅灰（实测 43 条线变 48 条且全乱）。
            if dark_bg:
                m = cv2.inRange(s, np.array(0, np.uint8), np.array(60, np.uint8)) & \
                    cv2.inRange(v, np.array(90, np.uint8), np.array(230, np.uint8))
                return m & cv2.bitwise_not(_edge_halo(v, bg_v, dark=True))
            lo = int(max(40.0, 0.30 * bg_v))
            hi = int(max(lo + 10, 0.94 * bg_v))
            m = cv2.inRange(s, np.array(0, np.uint8), np.array(45, np.uint8)) & \
                cv2.inRange(v, np.array(lo, np.uint8), np.array(hi, np.uint8))
            return m & cv2.bitwise_not(_edge_halo(v, bg_v, dark=False))
        return np.zeros(h.shape, dtype=np.uint8)

    target = np.zeros(h.shape, dtype=np.uint8)
    for color in colors:
        target |= _mask(color)
    if drop:
        # 剔除色优先：目标掩码再减掉 drop 色（挡住"黑线中混入的暗红像素"等）
        drop_mask = np.zeros(h.shape, dtype=np.uint8)
        for color in set(drop) & (set(_COLOR_RANGES) | {"黑", "白", "灰"}):
            drop_mask |= _mask(color)
        target &= cv2.bitwise_not(drop_mask)
    # 目标=黑(线条)，其余=白(背景)
    return np.where(target > 0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 模型输出颜色兜底

_ENT_TYPE_MAP = {
    "line": "line", "线段": "line", "直线": "line", "linesegment": "line",
    "circle": "circle", "圆": "circle", "圆形": "circle",
    "arc": "arc", "圆弧": "arc",
    "rect": "rect", "rectangle": "rect", "矩形": "rect", "方形": "rect",
    "polyline": "polyline", "多段线": "polyline", "多边形": "polyline",
    "spline": "spline", "bspline": "spline", "nurbs": "spline", "curve": "spline",
    "bezier": "spline", "曲线": "spline", "样条": "spline", "样条曲线": "spline",
    "自由曲线": "spline",
    "ellipse": "ellipse", "oval": "ellipse", "椭圆": "ellipse", "椭圆形": "ellipse",
    "text": "text", "文字": "text", "标注": "text", "label": "text",
    "point": "point", "点": "point",
}


def _pt2(v):
    """宽松取二维点；失败返回 None。"""
    if isinstance(v, dict):
        v = v.get("p") or v.get("point") or [v.get("x", 0), v.get("y", 0)]
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            return None
    return None


def _num2(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _entity_points_mm(ent) -> List[Tuple[float, float]]:
    """把图元采样成 mm 坐标点列（用于与 CV 参考几何做跨坐标系匹配）。
    字段别名与 dxf_writer.add_entity 保持一致；无法识别的图元返回空列表。"""
    if not isinstance(ent, dict):
        return []
    t = _ENT_TYPE_MAP.get(str(ent.get("type") or ent.get("kind") or "").strip().lower())
    pts: List[Tuple[float, float]] = []

    def seg(a, b, n=24):
        if not a or not b:
            return
        for k in range(n + 1):
            f = k / n
            pts.append((a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))

    def rect_edges(x, y, w, h):
        if None in (x, y, w, h):
            return
        for a, b in (((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                     ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))):
            seg(a, b, 16)

    if t == "line":
        p1 = _pt2(ent.get("p1", ent.get("start", ent.get("from", ent.get("a")))))
        p2 = _pt2(ent.get("p2", ent.get("end", ent.get("to", ent.get("b")))))
        seg(p1, p2)
    elif t == "rect":
        if all(k in ent for k in ("x", "y", "w", "h")):
            rect_edges(_num2(ent["x"]), _num2(ent["y"]),
                       _num2(ent["w"]), _num2(ent["h"]))
        else:
            p1, p2 = _pt2(ent.get("p1")), _pt2(ent.get("p2"))
            if p1 and p2:
                rect_edges(min(p1[0], p2[0]), min(p1[1], p2[1]),
                           abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
    elif t == "circle":
        c = _pt2(ent.get("center", ent.get("c", ent.get("p"))))
        r = _num2(ent.get("radius", ent.get("r")))
        if r is None and ent.get("diameter") is not None:
            r = _num2(ent.get("diameter"))
            r = None if r is None else r / 2.0
        if c and r and r > 0:
            for k in range(36):
                a = 2 * math.pi * k / 36
                pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    elif t == "arc":
        c = _pt2(ent.get("center", ent.get("c")))
        r = _num2(ent.get("radius", ent.get("r")))
        s_a = _num2(ent.get("start", ent.get("start_angle", ent.get("begin"))))
        e_a = _num2(ent.get("end", ent.get("end_angle", ent.get("finish"))))
        if c and r and r > 0 and s_a is not None and e_a is not None:
            s_r, e_r = math.radians(s_a), math.radians(e_a)
            while e_r <= s_r:
                e_r += 2 * math.pi
            for k in range(24):
                a = s_r + (e_r - s_r) * k / 24
                pts.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    elif t == "polyline":
        raw = ent.get("points") or ent.get("vertices") or ent.get("pts") or []
        vs = [p for p in (_pt2(p) for p in raw if isinstance(p, (list, tuple, dict)))
              if p]
        for i in range(len(vs) - 1):
            seg(vs[i], vs[i + 1], 12)
        if len(vs) > 2 and ent.get("closed", ent.get("close")) in (
                True, 1, "true", "True"):
            seg(vs[-1], vs[0], 12)
    elif t == "spline":
        # 曲线：points 已经是细分点就直接用；只有少量控制点时补足采样
        raw = (ent.get("points") or ent.get("fit_points") or ent.get("fit")
               or ent.get("vertices") or [])
        vs = [p for p in (_pt2(p) for p in raw if isinstance(p, (list, tuple, dict)))
              if p]
        closed = bool(ent.get("closed", ent.get("close")))
        if len(vs) >= 8:
            pts.extend(vs)
            if closed:
                pts.append(vs[0])
        elif len(vs) >= 3:
            try:
                import curves
                pts.extend(curves.spline_entity(vs, closed)["points"])
            except Exception:                        # noqa: BLE001
                pts.extend(vs)
        elif vs:
            pts.extend(vs)
    elif t == "ellipse":
        c = _pt2(ent.get("center", ent.get("c")))
        rx = _num2(ent.get("rx", ent.get("a", ent.get("major"))))
        ry = _num2(ent.get("ry", ent.get("b", ent.get("minor"))))
        if rx is None or ry is None:
            r = _num2(ent.get("radius"))
            rx = rx if rx is not None else r
            ry = ry if ry is not None else r
        if c and rx and ry:
            try:
                import curves
                pts.extend(curves.ellipse_points(
                    c, rx, ry, _num2(ent.get("rotation")) or 0.0,
                    _num2(ent.get("start")) or 0.0, _num2(ent.get("end")) or 360.0))
            except Exception:                        # noqa: BLE001
                pass
    elif t in ("text", "point"):
        p = _pt2(ent.get("p", ent.get("position", ent.get("at", ent.get("point")))))
        if p:
            if t == "text":
                # 文本用「估算包围盒」参与预览/匹配，否则叠加图上只有一个点、
                # 编号会盖住字，用户看不出这是文本图元
                h = _num2(ent.get("height")) or 5.0
                s = str(ent.get("value", ent.get("text",
                          ent.get("content", ent.get("label", ""))) or ""))
                w = max(h * 0.62 * max(1, len(s)), h * 0.62)
                pts.extend([(p[0], p[1]), (p[0] + w, p[1]),
                            (p[0] + w, p[1] + h), (p[0], p[1] + h)])
            else:
                pts.append(p)
    return pts


def _spec_pixels(spec):
    """把 CV spec 的图元变换回像素坐标（用 spec meta 里的 frame）。
    返回 (points, ok)。"""
    meta = spec.get("_meta") or {}
    frame = meta.get("frame")
    if not frame or len(frame) != 3:
        return [], False
    cx, cy, scale = frame
    if not scale or scale <= 0:
        return [], False
    out = []
    for e in spec.get("entities") or []:
        for (x, y) in _entity_points_mm(e):
            out.append((cx + x / scale, cy - y / scale))
    return out, bool(out)


def _match_ratio_px(sample_pts, ref_arr, tol):
    """sample_pts 中落入 ref_arr 任一点 tol 像素范围内的比例。"""
    if not sample_pts or ref_arr is None or len(ref_arr) == 0:
        return 0.0
    S = np.asarray(sample_pts, dtype=np.float64)
    d = np.sqrt(((S[:, None, :] - ref_arr[None, :, :]) ** 2).sum(-1))
    return float((d.min(axis=1) <= tol).mean())


def _vlm_mapping_candidates(pts_mm, full_px, W, H):
    """VLM 输出(mm) → 像素 的候选映射。
    模型不保证遵守"100mm 宽、左下原点"的约定，因此给出多个候选，按贴合度择优：
    canonical=按提示词约定；bbox=按整体外接框拟合（容忍任意缩放/原点）；
    center=按中心对齐+统一缩放。"""
    cands = [("canonical", lambda p: (p[0] * W / 100.0, H - p[1] * W / 100.0))]
    if pts_mm and full_px is not None and len(full_px) > 1:
        xs = [p[0] for p in pts_mm]
        ys = [p[1] for p in pts_mm]
        bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
        fx0, fy0 = float(full_px[:, 0].min()), float(full_px[:, 1].min())
        fx1, fy1 = float(full_px[:, 0].max()), float(full_px[:, 1].max())
        bw, bh = max(bx1 - bx0, 1e-6), max(by1 - by0, 1e-6)
        fw, fh = max(fx1 - fx0, 1e-6), max(fy1 - fy0, 1e-6)
        cands.append(("bbox", lambda p: (
            fx0 + (p[0] - bx0) * fw / bw, fy1 - (p[1] - by0) * fh / bh)))
        ccx, ccy = (bx0 + bx1) / 2, (by0 + by1) / 2
        fccx, fccy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        su = max(fw, fh) / max(bw, bh)
        cands.append(("center", lambda p: (
            fccx + (p[0] - ccx) * su, fccy - (p[1] - ccy) * su)))
    return cands


def _target_pixel_fraction(img, keep, drop):
    """目标颜色线条像素占全部线条像素的比例（0~1）；异常时返回 1（不删）。"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        n_full = int((_binarize(gray) > 0).sum())
        fg = color_filter_gray(img, keep, drop)
        if fg is None or n_full == 0:
            return 1.0
        n_keep = int((_binarize(fg) > 0).sum())
        return n_keep / n_full
    except Exception:
        return 1.0


def _load_image_any(image):
    """image：ndarray / 文件路径 / 裸 base64 / data: URL → ndarray 或 None。"""
    if image is None:
        return None
    if hasattr(image, "shape") and hasattr(image, "ndim"):
        return image
    if isinstance(image, str):
        s = image.strip()
        if s.startswith("data:"):
            return load_image_b64(s)
        if len(s) > 300 and "," not in s[:64]:
            return load_image_b64(s)
        if os.path.exists(s):
            return load_image(s)
    return None


def filter_spec_by_color(spec, image, keep, drop):
    """视觉大模型输出的 spec 颜色兜底过滤（确定性，不依赖模型是否听话）。

    本地 7B 视觉模型经常无视"去掉红色只留黑色"这类指令。这里用经典视觉做裁判：
    对原图分别做「全量 CV 提取」与「带颜色过滤的 CV 提取」，得到两套参考几何；
    再把模型输出的每个图元（经坐标系候选映射到像素）与两套几何比对：
      - 图元采样点 ≥60% 落在目标色几何上        → 保留
      - 图元采样点 ≥60% 落在全量几何但非目标色上 → 剔除（线真实存在但颜色不对）
      - 两套都贴不上（坐标偏差大/凭空想象）      → 保守保留
    返回 (新 spec 或原 spec, info)。info={"enabled","dropped","desc","reason"}。
    """
    info = {"enabled": False, "dropped": 0, "desc": "", "reason": ""}
    if not (keep or drop) or not isinstance(spec, dict):
        return spec, info
    ents = spec.get("entities") or []
    if not ents:
        return spec, info
    img = _load_image_any(image)
    if img is None or not HAS_CV:
        info["reason"] = "无法读取图片或无 OpenCV"
        return spec, info

    # 这里只为拿"参考几何"做坐标校验，用 3 遍即可（全 6 遍会重复三倍耗时）
    full = extract_spec(img, instruction="", passes=3)
    full_px, ok = _spec_pixels(full)
    if not ok:
        info["reason"] = "全量 CV 提取失败，无法校验"
        return spec, info
    full_arr = np.asarray(full_px, dtype=np.float64)

    kept = extract_spec(img, keep_colors=set(keep) or None,
                        drop_colors=set(drop) or None, passes=3)
    keep_px, ok2 = _spec_pixels(kept)
    if not ok2 and keep:
        # "黑"掩码对浅灰线不友好 → 放宽为消色再试一次。
        # 亮底图不能含"白"（白=背景本身，会把整张图误判成线条）；暗底图才加"白"。
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dark_bg = float(np.median(hsv[:, :, 2])) < 128.0
        achroma = ({"黑", "白", "灰"} if dark_bg else {"黑", "灰"}) - set(drop)
        keep_px, ok2 = _spec_pixels(extract_spec(
            img, keep_colors=achroma or None, drop_colors=set(drop) or None, passes=3))
    if ok2:
        keep_arr = np.asarray(keep_px, dtype=np.float64)
    else:
        # CV 在过滤图上没提取到几何：目标色像素是否几乎不存在？
        if _target_pixel_fraction(img, keep, drop) >= 0.005:
            info["reason"] = "目标色有线但 CV 未提取到，跳过兜底"
            return spec, info
        keep_arr = np.zeros((0, 2), dtype=np.float64)   # 目标色不存在 → 应全剔除

    H, W = img.shape[:2]
    tol = max(8.0, 0.018 * math.hypot(W, H))
    per_ent = [(e, _entity_points_mm(e)) for e in ents]
    if not any(p for _, p in per_ent):
        return spec, info
    all_mm = [p for _, pts in per_ent for p in pts]
    # 坐标系候选映射择优（按与全量几何的平均贴合度）
    best_fn, best_score = None, -1.0
    for _name, fn in _vlm_mapping_candidates(all_mm, full_arr, W, H):
        score = sum(_match_ratio_px([fn(p) for p in pts], full_arr, tol)
                    for _, pts in per_ent if pts) / max(
                        sum(1 for _, p in per_ent if p), 1)
        if score > best_score:
            best_fn, best_score = fn, score
    if best_fn is None or best_score < 0.15:
        info["reason"] = "无法把模型坐标对应到图片（跳过兜底）"
        return spec, info

    new_ents, dropped = [], 0
    for e, pts in per_ent:
        if not pts:
            new_ents.append(e)
            continue
        px = [best_fn(p) for p in pts]
        r_keep = _match_ratio_px(px, keep_arr, tol) if len(keep_arr) else 0.0
        r_full = _match_ratio_px(px, full_arr, tol)
        if r_keep >= 0.6:
            new_ents.append(e)
        elif r_full >= 0.6:
            dropped += 1
        else:
            new_ents.append(e)   # 无证据，保守保留
    if not dropped:
        return spec, info
    new_spec = dict(spec)
    new_spec["entities"] = new_ents
    desc = (("只保留" + "/".join(sorted(keep)) + "，") if keep else "") + \
           (("去掉" + "/".join(sorted(drop))) if drop else "")
    info.update({"enabled": True, "dropped": dropped, "desc": desc.strip("，")})
    return new_spec, info


# ---------------------------------------------------------------- 主入口

def extract_spec(image, instruction: str = "", name: Optional[str] = None,
                 max_dim: float = 100.0, debug_path: Optional[str] = None,
                 keep_colors=None, drop_colors=None, passes=None):
    """从图片提取几何 spec（与 dxf_writer.build_from_entities 兼容）。

    keep_colors/drop_colors：由调用方直接指定颜色集合（跳过指令解析），
    供颜色兜底过滤精确复用同一套颜色语义；为 None 时按 instruction 解析。
    passes：霍夫直线跑几遍（默认 6 遍灰度 + 彩色图追加 2 遍色度；1 表示退回单遍快跑）。

    返回 dict：
      {"name","unit","entities", ...,
       "_meta": {"method","counts","scale","debug_path","frame","passes"}}
    若提取失败/成果太少，"_meta.ok" 为 False（调用方应兜底到 VLM）。
    """
    if not HAS_CV or image is None:
        return {"_meta": {"ok": False, "reason": "no-cv-or-image"}}

    image = _load_image_any(image)
    if image is None:
        return {"_meta": {"ok": False, "reason": "no-cv-or-image"}}
    H_orig, W_orig = image.shape[:2]

    # ---- 0-0) 过大图片先降采样（提速 + 压高频噪点，坐标最后换算回原图）----
    image, rz = _downscale(image)

    # ---- 0) 颜色过滤（指令如"去掉红色只留黑色"；或调用方直接指定）----
    if keep_colors is None and drop_colors is None:
        keep_c, drop_c = parse_color_filter(instruction)
    else:
        keep_c = set(keep_colors or ())
        drop_c = set(drop_colors or ())
    color_note = ""
    # ---- 0) 颜色无关墨迹场：红色尺寸线 / 浅灰引线 / 蓝青中心线 与 黑轮廓同等对待 ----
    field = ink_field(image)
    filtered = None
    if keep_c or drop_c:
        filtered = color_filter_gray(image, keep_c, drop_c)
    if filtered is not None:
        # 只把"目标色像素"的原始色差留下，其余置 0。
        # 不直接拿 0/255 掩码当灰度图：掩码丢掉了抗锯齿/压缩造成的软边，
        # Canny 在硬边上会出双线，细线也会断成点。
        mask = filtered < 128
        field = _field_from_dist(np.where(mask, field["dist"], 0.0))
        color_note = "｜" + \
            (("只留" + "/".join(sorted(keep_c)) + " ") if keep_c else "") + \
            (("去掉" + "/".join(sorted(drop_c))) if drop_c else "")
    gray = field["gray"]

    H, W = image.shape[:2]
    D = math.hypot(W, H)

    # ---- 1) 二值化 + 边缘 ----
    bw, ink_thr = _binarize_dist(field)
    # 灰阶图清洗：把"离纸色不到阈值"的像素直接归为纸色。
    # 不洗的话，JPEG 压缩会在每根线周围留一圈 2~3px 的淡色振铃，Canny 会把振铃的
    # 内外边界各算一条边 → 一根线变两根、还会在标注密集区"凑"出成片的假斜线。
    if "dist" in field:
        gray = field["gray"].copy()
        gray[field["dist"] <= ink_thr] = 255
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if field.get("colorful"):
        field["chroma"] = cv2.medianBlur(field["chroma"], 3)
    ink_ratio = float((bw > 0).mean())
    # 纸纹/噪声图：每个像素都"偏离纸色"一点点，中位色差本身就占了最强墨迹的 1/3 以上。
    # 线稿图的中位色差几乎恒为 0（背景是同一块纸），两者一眼可分。
    structure = _line_structure(bw)
    noisy_scan = ink_ratio > 0.10 and structure < 0.70
    if ink_ratio > 0.42 or noisy_scan:
        # 墨迹占了小半张图：这不是线稿（照片/纹理/纯噪点），
        # 经典 CV 只会在每个像素团上"找到"线，结果全是垃圾 → 直接交给 VLM 路线
        return {"_meta": {"ok": False, "reason": "not-a-line-drawing",
                          "ink_ratio": round(ink_ratio, 3),
                          "structure": round(structure, 3)}}
    edges = cv2.Canny(bw, 50, 150)

    # ---- 2) 文字区先抹白：尺寸数值/R 标注的笔画一旦进入霍夫，会被"共线桥接"
    #         合成一条条穿过文字的假斜线（实测左下角 3 条引线被凑成 10 多条）。
    #         抹白只影响"检测用的灰度/色度图"，二值图 bw 保留原样供墨迹证据和轮廓用。
    glyph_all = _glyph_boxes(bw, D, return_counts=True)
    glyphs = [(x, y, w, h) for (x, y, w, h, _c) in glyph_all]
    # 只抹"多字组成"的文字行（R24 / 2×45° 这类）。单字框可能是数字，也可能恰好是
    # 图形的一个拐角（虚线矩形的角就是 22×22 的 L 形），抹掉会毁掉真实几何，
    # 所以单字只做线段级过滤，不动像素。
    for (gx, gy, gw, gh) in [t for (x, y, w, h, c) in glyph_all if c >= 2
                             for t in ((x, y, w, h),)]:
        x0, y0 = max(0, gx - 2), max(0, gy - 2)
        x1, y1 = min(W, gx + gw + 2), min(H, gy + gh + 2)
        gray[y0:y1, x0:x1] = 255
        if field.get("colorful") and field.get("chroma") is not None:
            field["chroma"][y0:y1, x0:x1] = 255

    # ---- 2-3) 多遍霍夫直线（默认 6 遍 + 彩色图再加 2 遍色度）→ 跨遍投票融合 ----
    n_pass = max(1, int(passes)) if passes else DEFAULT_PASSES
    chroma_img = field.get("chroma") if field.get("colorful") else None
    per_pass = detect_lines_multipass(gray, bw, D, passes=n_pass,
                                      chroma=chroma_img,
                                      chroma_raw=field.get("chroma_raw"))
    fused, line_stats = fuse_line_passes(per_pass, bw, D)
    dist_tol = max(4.0, 0.008 * D)
    # 融合后再走一遍共线合并：把"票数一致、但端点差几像素"的平行碎段吸回同一条主干，
    # 否则同一根线会被拆成多条首尾相接的线段写进 DXF
    segs = [_snap_axis(t[0]) for t in fused]
    segs = _merge_collinear(segs, angle_tol=5.0, dist_tol=dist_tol,
                            max_bridge=max(12.0, 0.03 * D))
    segs = [_snap_axis(s) for s in segs]
    segs = _dedup_lines(segs)
    segs = [s for s in segs if _seg_len(s) > max(12.0, 0.012 * D)]

    # ---- 3-5) 再兜一层：跨到文字框外的残留碎线也剔掉 ----
    n_glyph_seg = 0
    if glyphs:
        before_n = len(segs)
        segs = [s for s in segs if not _in_glyph_boxes(s, glyphs)]
        n_glyph_seg = before_n - len(segs)
    # ---- 3-6) 剔除"中点悬空"的桥接弦（尽可能用曲线：这些假直线是霍夫
    #           把缓弯抄近路连出来的，真曲线应由串弧/样条表达）----
    segs = _drop_hanging_chords(segs, bw, D)

    # ---- 4) 矩形（轮廓 + 直线配对，各自验证后合并去重）----
    rects = _detect_rects_from_contours(bw, edges)
    rects_lines = _detect_rects_from_lines(segs, edges, D)
    all_rects = _dedup_rects(rects + rects_lines, overlap=0.5)
    all_rects = _drop_split_rects(all_rects)
    if glyphs:
        all_rects = [r for r in all_rects if not _rect_in_glyph_boxes(r, glyphs)]

    # ---- 5) 圆 / 圆弧 ----
    circles, arcs = _detect_circles(gray, D, edges, ink=bw)

    # ---- 6) 剔除贴圆切线的弧线碎片（圆弧同样要剔，否则一条弧会被画两遍）----
    segs = _drop_arc_fragments(segs, list(circles) + [(a[0], a[1], a[2]) for a in arcs], D)

    # ---- 6.5) 短弦串弧：挂钩/法兰这类切线圆弧被霍夫切成一串短弦，
    #       把"首尾相接、同向缓弯、能共圆"的弦链升级成 ARC 实体。
    #       跑两遍：第一遍收编后释放的碎弦可能凑成新的种子圆 ----
    try:
        _dil6 = cv2.dilate(bw, np.ones((3, 3), np.uint8)) if bw is not None else None
        _dil65 = cv2.dilate(bw, np.ones((5, 5), np.uint8)) if bw is not None else None
        _n_ca = 0
        for _ in range(2):
            segs, chord_arcs = _chords_to_arcs(segs, D, dil=_dil6, circles=circles)
            if not chord_arcs:
                break
            arcs = list(arcs) + chord_arcs
            arcs = _dedup_by_circle(arcs)
            _n_ca += len(chord_arcs)
        if _n_ca:
            print("    短弦串弧：%d 道弧由碎弦合成" % _n_ca)
            # 串出的新弧回头再剔一遍贴弧碎弦：步骤 6 用的是检出圆/弧，
            # 串出来的弧当时还不在名单里，它们的残弦会跟弧重复绘制
            segs = _drop_arc_fragments(segs, [(a[0], a[1], a[2]) for a in arcs], D)
    except Exception as _e:                        # noqa: BLE001
        print(f"    [短弦串弧跳过：{_e}]")

    # ---- 7) 吃掉矩形边上的线段，避免重复输出 ----
    def _on_rect_edge(s, r):
        x, y, w, h = r
        mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        for (ex1, ey1, ex2, ey2) in ((x, y, x + w, y), (x, y + h, x + w, y + h),
                                     (x, y, x, y + h), (x + w, y, x + w, y + h)):
            if _perp_dist(mx, my, (ex1, ey1, ex2, ey2)) < dist_tol \
                    and _seg_len(s) < max(w, h) * 1.05:
                return True
        return False

    segs = [s for s in segs if not any(_on_rect_edge(s, r) for r in all_rects)]

    # ---- 8) 剩余碎片串多段线 ----
    chains, chain_used = _chain_segments(segs, tol=max(3.0, 0.004 * D), return_used=True)
    if chain_used:      # 已串进多段线的线段不要再画一遍直线，否则同一条边出现两次
        segs = [s for i, s in enumerate(segs) if i not in chain_used]

    # ---- 8.5) 有机线稿（螺旋/云纹/手绘曲线）：霍夫只能把一条曲线切成几十段短弦，
    #           这些短弦还是互相交叉重叠的（端点邻域里挤着十几个邻居），端点串链
    #           根本连不起来。这类图改用"墨迹骨架 → 有序路径 → 样条"矢量化，
    #           直接得到真曲线。工程图（边很长）不会走到这里。
    organic_res = None
    # 判据：霍夫把线条切得稀碎（段数多、段长普遍很短）才是有机线稿（螺旋/云纹/
    # 手绘曲线）。工程图的边是一条一段，段数少、段长大 → 保持霍夫的精确结果。
    organic_res = None
    try:
        import organic as _organic
        # 严格预检不过时走宽松二次预检：多遍融合会把短弦拼成中段，"短段占比"
        # 差一点的弯曲图（螺旋/云纹）会漏网；宽松预检只看"段多 + 中位段短"，
        # 最终由 vectorize 的"弯曲墨水占比"闸门裁决，工程图进去也会被弹回来。
        if _organic.looks_organic(segs, D) or _organic.looks_organic_loose(segs, D):
            organic_res = _organic.vectorize(bw, D, blank_boxes=glyphs)
        if organic_res and organic_res.get("entities"):
            segs, chains, all_rects, circles, arcs = [], [], [], [], []
            chain_used = set()
            print("    有机线稿：骨架矢量化 %d 条路径（弯曲墨水占 %.0f%%）"
                  % (organic_res["paths"], 100 * organic_res["curved_share"]))
        else:
            organic_res = None
    except Exception as e:                          # noqa: BLE001
        organic_res = None
        print(f"    [有机线稿矢量化跳过：{e}]")

    # ---- 9) 统计与质量门 ----
    n_line = len(segs)
    n_circ = len(circles)
    n_arc = len(arcs)
    n_rect = len(all_rects)
    n_chain = len(chains)
    total = n_line + n_circ + n_arc + n_rect + n_chain
    total_len = sum(_seg_len(s) for s in segs)
    if organic_res:
        oe = organic_res["entities"]
        n_line = sum(1 for e in oe if e["type"] == "line")
        n_circ = sum(1 for e in oe if e["type"] == "circle")
        n_arc = 0
        n_rect = 0
        n_chain = sum(1 for e in oe if e["type"] in ("spline", "polyline"))
        total = len(oe) + len(all_rects) + len(circles)
        total_len = float(organic_res["total_len"])
    ok = total >= 2 and (total_len > 0.08 * D or n_circ + n_arc + n_rect >= 1)
    if not ok and (keep_c or drop_c) and total >= 1 \
            and (n_circ + n_arc + n_rect >= 1 or total_len > 0.05 * D):
        # 颜色过滤后常只剩一两个形状，放宽质量门，避免误判失败而回退 VLM
        ok = True

    # ---- 10) 坐标变换：y 翻转 + 归一化到毫米 ----
    all_pts = []
    for s in segs:
        all_pts += [(s[0], s[1]), (s[2], s[3])]
    for c in chains:
        all_pts += c
    for (x, y, w, h) in all_rects:
        all_pts += [(x, y), (x + w, y + h)]
    for (cx, cy, r) in circles:
        all_pts += [(cx - r, cy - r), (cx + r, cy + r)]
    for (cx, cy, r, _a0, _a1) in arcs:
        all_pts += [(cx - r, cy - r), (cx + r, cy + r)]
    if organic_res:
        for e in organic_res["entities"]:
            for p in (e.get("points") or ()):
                all_pts.append((p[0], p[1]))
            if e["type"] == "line":
                all_pts += [tuple(e["p1"]), tuple(e["p2"])]
            elif e["type"] == "circle":
                cx, cy = e["center"]
                r = e["radius"]
                all_pts += [(cx - r, cy - r), (cx + r, cy + r)]
    if not all_pts:
        return {"_meta": {"ok": False, "reason": "no-geometry"}}
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    bw_, bh_ = max(xs) - min(xs), max(ys) - min(ys)
    scale = _parse_scale_from_text(instruction)
    if not scale or scale <= 0:
        scale = max_dim / max(bw_, bh_, 1.0)
    elif rz != 1.0:
        # 显式比例尺（"宽 100mm"）按原图像素定义，降采样后要按比例补偿
        scale = scale / rz

    def T(p):
        """像素坐标 → DXF 毫米坐标（y 翻转 + 居中 + 缩放）。"""
        return (round((p[0] - (min(xs) + max(xs)) / 2) * scale, 3),
                round(((H - p[1]) - (H - (min(ys) + max(ys)) / 2)) * scale, 3))

    # ---- 11) 组装 entities（rect 用 DXF 坐标系左下角）----
    entities: List[Dict[str, Any]] = []
    if organic_res:
        # 骨架矢量化的结果：保留 line / circle / spline 三种，坐标统一过 T()
        for e in organic_res["entities"]:
            t = e.get("type")
            if t == "line":
                entities.append({"type": "line",
                                 "p1": list(T(tuple(e["p1"]))),
                                 "p2": list(T(tuple(e["p2"])))})
            elif t == "circle":
                c = T(tuple(e["center"]))
                entities.append({"type": "circle", "center": [c[0], c[1]],
                                 "radius": round(e["radius"] * scale, 3)})
            elif t in ("spline", "polyline"):
                pts = [list(T(tuple(p))) for p in e.get("points") or ()]
                if len(pts) < 2:
                    continue
                ent = {"type": t, "points": pts, "closed": bool(e.get("closed"))}
                entities.append(ent)
    for (x, y, w, h) in all_rects:
        # 像素坐标 y 向下；翻转到 DXF（y 向上）后，图像上的 (x,y) 变成左上角
        bl = T((x, y + h))          # 图像左下角 = DXF 左下角
        tr = T((x + w, y))          # 图像右上角 = DXF 右上角
        entities.append({"type": "rect", "x": bl[0], "y": bl[1],
                         "w": round(tr[0] - bl[0], 3), "h": round(tr[1] - bl[1], 3)})
    for (cx, cy, r) in circles:
        c = T((cx, cy))
        entities.append({"type": "circle", "center": [c[0], c[1]], "radius": round(r * scale, 3)})
    for (cx, cy, r, a0, a1) in arcs:
        # 图像角度（y 向下）→ DXF 角度（y 向上，逆时针）：θ_world = -θ_image，
        # 且起止要对调才能保持 DXF 的逆时针走向。
        c = T((cx, cy))
        entities.append({"type": "arc", "center": [c[0], c[1]],
                         "radius": round(r * scale, 3),
                         "start": round((-a1) % 360.0, 2),
                         "end": round((-a0) % 360.0, 2)})
    for s in segs:
        entities.append({"type": "line", "p1": list(T((s[0], s[1]))), "p2": list(T((s[2], s[3])))})
    for ch in chains:
        entities.append({"type": "polyline",
                         "points": [list(T(p)) for p in ch], "closed": False})

    # mm→像素 反变换参数：px = frame[0] + x/scale, py = frame[1] - y/scale
    # 若做过降采样，把 frame 换算回原图像素，保证前端叠加图不错位
    fcx = (min(xs) + max(xs)) / 2.0
    fcy = (min(ys) + max(ys)) / 2.0
    fscale = scale
    if rz != 1.0:
        fcx, fcy, fscale = fcx / rz, fcy / rz, scale * rz

    n_curve = 0
    cnt = {"line": 0, "rect": 0, "circle": 0, "arc": 0, "polyline": 0,
           "spline": 0, "text": 0}
    for e in entities:
        t = e.get("type")
        cnt[t] = cnt.get(t, 0) + 1
        if t in ("spline", "ellipse", "arc"):
            n_curve += 1
    meta = {"ok": ok, "method": "cv-lines", "counts": cnt,
            "organic": bool(organic_res), "curves": n_curve,
            "arcs": len(arcs),
            "scale": round(scale, 4), "image_size": [W_orig, H_orig],
            "passes": line_stats, "downscale": round(rz, 4),
            "color_filter": color_note.strip("｜") or None,
            "frame": [round(fcx, 2), round(fcy, 2), round(fscale, 6)],
            "debug_path": debug_path}

    # ---- 12) 调试叠加图（让用户亲眼看到 CV 提取了什么）----
    if debug_path:
        try:
            vis = image.copy()
            for (x, y, w, h) in all_rects:
                cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            for (cx, cy, r) in circles:
                cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
                cv2.circle(vis, (int(cx), int(cy)), 2, (255, 0, 0), -1)
            for (cx, cy, r, a0, a1) in arcs:        # 圆弧：青色，只画覆盖的那一段
                cv2.ellipse(vis, (int(cx), int(cy)), (int(r), int(r)), 0,
                            a0, a0 + (a1 - a0) % 360.0, (255, 255, 0), 2)
                cv2.circle(vis, (int(cx), int(cy)), 2, (255, 255, 0), -1)
            for s in segs:
                cv2.line(vis, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 180, 0), 2)
            for ch in chains:
                for k in range(len(ch) - 1):
                    cv2.line(vis, (int(ch[k][0]), int(ch[k][1])),
                             (int(ch[k + 1][0]), int(ch[k + 1][1])), (0, 140, 255), 2)
            if organic_res:                    # 有机线稿：绿=直线 红=圆 橙=样条
                for e in organic_res["entities"]:
                    t = e.get("type")
                    if t == "line":
                        cv2.line(vis, (int(e["p1"][0]), int(e["p1"][1])),
                                 (int(e["p2"][0]), int(e["p2"][1])), (0, 180, 0), 2)
                    elif t == "circle":
                        cv2.circle(vis, (int(e["center"][0]), int(e["center"][1])),
                                   int(e["radius"]), (0, 0, 255), 2)
                    else:
                        pts = [(int(p[0]), int(p[1])) for p in e.get("points") or ()]
                        for k in range(len(pts) - 1):
                            cv2.line(vis, pts[k], pts[k + 1], (0, 140, 255), 2)
                        if e.get("closed") and len(pts) > 2:
                            cv2.line(vis, pts[-1], pts[0], (0, 140, 255), 2)
            cv2.imwrite(debug_path, vis)
        except Exception:
            pass

    spec = {"name": name or "cv_drawing", "unit": "mm", "entities": entities, "_meta": meta}
    return spec
