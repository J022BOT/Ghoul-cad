"""彩色线条识别（颜色无关墨迹场 + 色度专用遍）离线自测。

背景：工程图里除了黑轮廓，还有红色尺寸线、蓝/青色中心线、浅灰引线。
       旧管线先转灰度再 Otsu——淡红线的灰度接近纸色（RGB 216,216,240 → 灰 220），
       会被当成背景整根丢掉。本测试盯的正是这类"颜色不同但必须识别"的线。

验证六件事：
  1) 红/蓝/绿/黄/浅灰/深灰/黑 七种颜色的线都能被检出（不偏科）；
  2) 极淡的彩线（饱和度 ~25、亮度接近纸）也能抓到——靠色度通道；
  3) 暗底（CAD 截图）上的白/红/蓝/绿/青线同样能识别；
  4) 淡铅笔稿 + 纸纹 + 光照不均：只出真实几何，不吐假线；
  5) 颜色过滤（只留红 / 去掉红 / 只留灰）语义正确，且不把白纸当线条；
  6) 纸纹/噪点图仍被判定"非线稿"拒绝。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import line_extractor as L

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  OK   %-40s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-40s %s" % (name, detail))


def blank(w=700, h=700, val=255):
    return np.full((h, w, 3), val, np.uint8)


# ---------------------------------------------------------------- 1) 七色还原
print("[1] 七种颜色的线条都要能识别")
img = blank()
cv2.rectangle(img, (60, 60), (300, 240), (10, 10, 10), 2)        # 黑
cv2.circle(img, (500, 150), 80, (0, 0, 230), 2)                  # 红
cv2.line(img, (60, 320), (300, 320), (230, 0, 0), 2)             # 蓝
cv2.line(img, (60, 380), (300, 380), (0, 160, 0), 2)             # 绿
cv2.rectangle(img, (400, 300), (650, 450), (205, 205, 205), 2)   # 浅灰
cv2.circle(img, (150, 550), 70, (0, 200, 230), 2)                # 黄
cv2.line(img, (400, 550), (650, 550), (120, 120, 120), 2)        # 深灰

s = L.extract_spec(img)
c = s["_meta"]["counts"]
check("全量识别：3 线 + 2 矩形 + 2 圆",
      c["line"] >= 3 and c["rect"] >= 2 and c["circle"] >= 2, str(c))
check("彩色图自动追加色度遍（8 遍）",
      s["_meta"]["passes"]["passes"] == 8, "passes=%d" % s["_meta"]["passes"]["passes"])

# ---------------------------------------------------------------- 2) 极淡彩线
print("[2] 极淡彩线（亮度≈纸色、只靠颜色区分）")
faint = blank()
pale_red = (216, 216, 240)      # BGR：淡粉，灰度 220，与白纸几乎同亮度
pale_blue = (240, 216, 216)     # 淡蓝
cv2.line(faint, (80, 200), (620, 200), pale_red, 1)
cv2.line(faint, (80, 400), (620, 400), pale_blue, 1)
sf = L.extract_spec(faint)
# 淡线在灰度图上几乎是白的：旧管线这里会返回 0 条线
check("淡红线 / 淡蓝线各检出 1 条以上",
      sf["_meta"]["counts"]["line"] >= 2, str(sf["_meta"]["counts"]))

gray_only = cv2.cvtColor(faint, cv2.COLOR_BGR2GRAY)
old_gray = L._binarize(gray_only)
check("对照：同图转灰度后墨迹几乎为零（说明必须靠颜色）",
      float((old_gray > 0).mean()) < 0.02, "灰阶墨迹占比=%.4f" % (old_gray > 0).mean())

# ---------------------------------------------------------------- 3) 暗底截图
print("[3] 暗底（CAD 截图风）彩色线")
dark = blank(val=0)
cv2.rectangle(dark, (60, 60), (300, 240), (255, 255, 255), 2)     # 白
cv2.circle(dark, (500, 150), 80, (0, 0, 255), 2)                  # 红
cv2.line(dark, (60, 320), (300, 320), (255, 128, 0), 2)           # 蓝
cv2.line(dark, (60, 380), (300, 380), (0, 255, 0), 2)             # 绿
cv2.rectangle(dark, (400, 300), (650, 450), (200, 200, 0), 2)     # 青
cv2.circle(dark, (150, 550), 70, (0, 255, 255), 2)               # 黄
dark = np.clip(dark.astype(np.int16) +
               np.random.default_rng(4).normal(0, 4, dark.shape), 0, 255).astype(np.uint8)
sd = L.extract_spec(dark)
cd = sd["_meta"]["counts"]
check("暗底：2 矩形 + 2 圆 + 2 线以上",
      cd["rect"] >= 2 and cd["circle"] >= 2 and cd["line"] >= 2, str(cd))

# ---------------------------------------------------------------- 4) 淡铅笔稿
print("[4] 淡铅笔稿 + 纸纹 + 光照不均")
rng = np.random.default_rng(9)
yy, xx = np.mgrid[0:700, 0:700]
paper = 232 + 18 * (xx / 700) - 10 * (yy / 700)          # 光照渐变
paper = np.clip(paper + rng.normal(0, 3.5, (700, 700)), 0, 255).astype(np.uint8)
pen = cv2.merge([paper, paper, paper])
cv2.rectangle(pen, (80, 80), (320, 250), (150, 150, 150), 1)
cv2.circle(pen, (520, 170), 85, (150, 150, 150), 1)
cv2.line(pen, (80, 340), (320, 340), (150, 150, 150), 1)
pen = np.clip(pen.astype(np.int16) + rng.normal(0, 2, pen.shape), 0, 255).astype(np.uint8)
sp = L.extract_spec(pen)
cp = sp["_meta"]["counts"]
check("淡稿：1 矩形 + 1 圆 + 1 线",
      cp["rect"] == 1 and cp["circle"] == 1 and cp["line"] <= 2, str(cp))
check("淡稿不吐垃圾（总图元 ≤ 5）", sum(cp.values()) <= 5, str(cp))

# ---------------------------------------------------------------- 5) 颜色过滤
print("[5] 颜色过滤语义（只留 / 去掉）")
s_red = L.extract_spec(img, keep_colors={"红"})
check("只留红 → 仅 1 个圆（别的颜色全滤掉）",
      s_red["_meta"]["ok"] and s_red["_meta"]["counts"]["circle"] == 1
      and s_red["_meta"]["counts"]["line"] == 0, str(s_red["_meta"]["counts"]))

s_blue = L.extract_spec(img, keep_colors={"blue"})      # 英文别名同样认
check("只留蓝（英文别名）→ 1 条线",
      s_blue["_meta"]["ok"] and s_blue["_meta"]["counts"]["line"] == 1,
      str(s_blue["_meta"]["counts"]))

s_gray = L.extract_spec(img, keep_colors={"灰"})
cg = s_gray["_meta"]["counts"]
check("只留灰 → 浅灰矩形 + 深灰线",
      s_gray["_meta"]["ok"] and cg["rect"] == 1 and cg["line"] == 1, str(cg))

s_nored = L.extract_spec(img, drop_colors={"红"})
# 图上只有两个圆：红圆 (500,150) 与黄圆 (150,550)。去掉红后应当只剩那个黄圆
check("去掉红 → 红圆被剔除、黄圆保留",
      s_nored["_meta"]["ok"] and s_nored["_meta"]["counts"]["circle"] == 1
      and s_nored["_meta"]["counts"]["rect"] >= 2,
      str(s_nored["_meta"]["counts"]))

fg = L.color_filter_gray(img, set(), {"红"})
check("去掉红时白纸不会被当成线条",
      fg is not None and float((fg < 128).mean()) < 0.20,
      "墨迹占比=%.4f" % ((fg < 128).mean() if fg is not None else -1))

# ---------------------------------------------------------------- 6) 非线稿拒绝
print("[6] 非线稿拒绝（不能因为换算法就放行）")
rng = np.random.default_rng(11)
g1 = np.clip(np.full((600, 600), 255, np.int16) + rng.normal(0, 18, (600, 600)),
             0, 255).astype(np.uint8)
grain = cv2.merge([g1, g1, g1])
sg = L.extract_spec(grain)
check("纸纹噪声图被判非线稿", not sg["_meta"].get("ok"),
      "reason=%s structure=%s" % (sg["_meta"].get("reason"),
                                  sg["_meta"].get("structure")))

noise = rng.integers(0, 255, (600, 600, 3), dtype=np.uint8)
sn = L.extract_spec(noise)
check("纯噪点图被判非线稿", not sn["_meta"].get("ok"),
      "reason=%s" % sn["_meta"].get("reason"))

print("\n总结：%d 通过 / %d 失败" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
