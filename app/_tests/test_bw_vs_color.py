# -*- coding: utf-8 -*-
"""黑白化(bw_photo) 与 颜色过滤 的执行顺序回归。

真实 bug（暗底 CAD 截图，黑背景 + 白轮廓 + 青色标注 + 绿色中心线）：
  · bw_photo 先把图压成"白底黑线"（白线/青线/绿线一律变黑，颜色信息全毁）
  · 之后 color_filter_gray 再判"只留白" → 背景 92% 被判成目标色 → 整张反转
    → 二值化后 0 墨迹 → **一条线都识别不到**
  · "只留黑"则等于全保留 → 52 条杂线（青色尺寸线、绿色中心线全进来）
修法两层：①有色过滤指令时 _cv_spec_from_image 跳过 bw_photo；
          ②color_filter_gray 遇消色图（已黑白化）把"白"当纸，集合空则不过滤。
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


def _near(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def main():
    import numpy as np
    import cv2
    from line_extractor import (bw_photo, color_filter_gray, _binarize,
                                parse_color_filter)

    print("=== 黑白化 vs 颜色过滤 ===\n[构造素材]")
    H, W = 400, 600
    img = np.zeros((H, W, 3), np.uint8)              # 纯黑背景（暗底）
    cv2.rectangle(img, (40, 60), (300, 340), (255, 255, 255), 3)   # 白：主轮廓
    cv2.circle(img, (450, 200), 70, (255, 255, 255), 3)            # 白：圆
    # 注意 OpenCV 是 BGR：青 = (255,255,0)，绿 = (0,255,0)
    cv2.line(img, (40, 370), (300, 370), (255, 255, 0), 2)         # 青：尺寸线
    cv2.line(img, (40, 200), (300, 200), (0, 255, 0), 2)           # 绿：中心线
    img = cv2.GaussianBlur(img, (3, 3), 0)           # 抗锯齿：避免纯理想像素

    def ink_ratio(g):
        if g is None:
            return -1.0
        gg = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) if g.ndim == 3 else g
        return float((_binarize(gg) > 0).mean())

    print("\n[1] 原图过滤：各色只留自己")
    g_white = color_filter_gray(img, {"白"}, set())
    g_black = color_filter_gray(img, {"黑"}, set())
    g_cyan = color_filter_gray(img, {"青"}, set())
    check("原图 keep=白 → 有目标像素且不吞掉全图",
          0.002 < ink_ratio(g_white) < 0.25, ink_ratio(g_white))
    check("原图 keep=青 → 只留青线（比白线少）",
          0.0 < ink_ratio(g_cyan) < ink_ratio(g_white) + 1e-9,
          "青=%.4f 白=%.4f" % (ink_ratio(g_cyan), ink_ratio(g_white)))
    check("原图 keep=黑（暗底）→ 放宽为消色偏离，能出东西",
          ink_ratio(g_black) > 0.0, ink_ratio(g_black))

    print("\n[2] 关键回归：消色图（bw_photo 之后）不得把背景当'白线'")
    bw = bw_photo(img)
    hsv = cv2.cvtColor(bw, cv2.COLOR_BGR2HSV)
    check("bw_photo 输出确实是白底黑线", float(np.median(hsv[:, :, 2])) > 200.0)
    g_bw_white = color_filter_gray(bw, {"白"}, set())
    check("bw 图 keep=白 → 返回 None（不过滤，防止整张反转）",
          g_bw_white is None, type(g_bw_white).__name__)
    # 即便没走 None 分支，也不能出现"目标=整片背景"的反转
    if g_bw_white is not None:
        check("bw 图 keep=白 → 目标占比不是整张图",
              ink_ratio(g_bw_white) < 0.5, ink_ratio(g_bw_white))

    print("\n[3] 消色图不能误伤正常过滤")
    gray_scan = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_scan = cv2.cvtColor(gray_scan, cv2.COLOR_GRAY2BGR)   # 灰度扫描件（消色）
    g_gs_black = color_filter_gray(gray_scan, {"黑"}, set())
    check("灰度图 keep=黑 → 仍正常过滤（不被 achromatic 误杀）",
          g_gs_black is not None and ink_ratio(g_gs_black) > 0.0,
          ink_ratio(g_gs_black))

    print("\n[4] 亮底彩图不受 achromatic 分支影响")
    light = np.full((H, W, 3), 250, np.uint8)
    cv2.line(light, (60, 100), (500, 100), (40, 40, 220), 3)     # 红线
    cv2.line(light, (60, 200), (500, 200), (30, 30, 30), 3)      # 黑线
    light = cv2.GaussianBlur(light, (3, 3), 0)
    g_red = color_filter_gray(light, {"红"}, set())
    check("亮底彩图 keep=红 → 正常出目标", g_red is not None
          and ink_ratio(g_red) > 0.0, ink_ratio(g_red))

    print("\n[5] 指令解析 + 端到端：有色过滤指令时跳过黑白化")
    k, d = parse_color_filter("只保留白色线条")
    check("解析 '只保留白色线条' → keep={'白'}", k == {"白"} and not d, (k, d))
    try:
        import cad_agent
        out = tempfile.mkdtemp(prefix="bwcf_")
        tmp = os.path.join(tempfile.gettempdir(), "_bwcf_case.png")
        cv2.imencode(".png", img)[1].tofile(tmp)
        spec, dbg = cad_agent._cv_spec_from_image(
            tmp, "只保留白色线条", out, bw_first=True)
        ok = bool(spec and spec.get("_meta", {}).get("ok"))
        n = len(spec.get("entities", [])) if spec else 0
        check("端到端：暗底图 + 只留白 → 有产出", ok and n >= 2, "图元=%d" % n)
        check("端到端：跳过了黑白化（_meta 无 bw_path）",
              not (spec or {}).get("_meta", {}).get("bw_path"),
              (spec or {}).get("_meta", {}).get("bw_path"))
    except Exception as e:                           # noqa: BLE001
        check("端到端：暗底图 + 只留白 → 有产出", False, str(e)[:80])
        check("端到端：跳过了黑白化（_meta 无 bw_path）", False, str(e)[:80])

    total = 12
    print("\n===== 黑白化/颜色过滤测试：%d 通过 / %d 失败 =====" % (total - len(FAILS), len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("   FAILED:", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
