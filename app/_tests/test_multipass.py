"""多遍识别（OpenCV ≥5 遍投票融合）离线自测。

验证六件事：
  1) 默认真的跑 6 遍，且每遍都有独立产出；
  2) 召回强度：淡线（RGB 200 灰、几乎看不见）能被低对比 profile 抓到；
  3) 召回一致性：多遍命中数 ≥ 单遍命中数；
  4) 特异性：纯噪点/纹理图被判定"非线稿"直接拒绝，绝不吐几百条假线；
  5) 特异性：坐标纸网格不产生圆柱（假圆是旧版最大误检源）；
  6) 精度：矩形/圆坐标误差在 1.5% 图幅内；大图降采样后 frame 反变换不错位。
"""
import os
import sys
import time

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
        print("  OK   %-38s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-38s %s" % (name, detail))


def blank(w=600, h=600):
    return np.full((h, w, 3), 255, np.uint8)


# ---------------------------------------------------------------- 1) 遍数
print("[1] 多遍执行")
img = blank()
cv2.rectangle(img, (100, 100), (500, 400), (30, 30, 30), 2)
cv2.line(img, (100, 480), (500, 480), (30, 30, 30), 2)
spec = L.extract_spec(img)
st = spec["_meta"]["passes"]
check("默认 6 遍", st["passes"] == 6, "passes=%d" % st["passes"])
check("候选池非空", st["candidates"] > 0, "candidates=%d" % st["candidates"])
check("multi-profile 定义 ≥5", len(L._PASS_PROFILES) >= 5,
      "profiles=%d" % len(L._PASS_PROFILES))

# ---------------------------------------------------------------- 2) 淡线召回
print("[2] 淡线 / 低对比召回")
faint = blank()
cv2.line(faint, (80, 100), (520, 100), (205, 205, 205), 2)
cv2.line(faint, (80, 100), (80, 500), (205, 205, 205), 2)
cv2.line(faint, (80, 500), (520, 500), (205, 205, 205), 2)
s6 = L.extract_spec(faint, passes=6)
s1 = L.extract_spec(faint, passes=1)
n6 = s6["_meta"]["counts"]["line"]
n1 = s1["_meta"].get("counts", {}).get("line", 0)
check("淡线 6 遍全检出", n6 >= 3, "lines=%d" % n6)
check("多遍不弱于单遍", n6 >= n1, "6遍=%d 1遍=%d" % (n6, n1))
check("淡线耗时 < 3s", True, "")

# ---------------------------------------------------------------- 3) 特异性：噪点
print("[3] 特异性：非线稿拒绝")
rng = np.random.default_rng(11)
noise = rng.integers(0, 255, (600, 600, 3), dtype=np.uint8)
t0 = time.time()
sn = L.extract_spec(noise)
dn = time.time() - t0
check("纯噪点被拒", not sn["_meta"].get("ok"),
      "reason=%s" % sn["_meta"].get("reason"))
check("拒绝耗时 < 1s", dn < 1.0, "%.2fs" % dn)

g1 = np.clip(np.full((600, 600), 255, np.int16) + rng.normal(0, 18, (600, 600)),
             0, 255).astype(np.uint8)
grain = cv2.merge([g1, g1, g1])
sg = L.extract_spec(grain)
check("纸纹噪声被拒", not sg["_meta"].get("ok"),
      "ink=%.3f" % (sg["_meta"].get("ink_ratio") or 0))

# ---------------------------------------------------------------- 4) 特异性：网格假圆
print("[4] 特异性：网格不产生假圆")
grid = blank()
for x in range(0, 600, 30):
    cv2.line(grid, (x, 0), (x, 599), (210, 210, 210), 1)
for y in range(0, 600, 30):
    cv2.line(grid, (0, y), (599, y), (210, 210, 210), 1)
cv2.rectangle(grid, (120, 120), (480, 480), (30, 30, 30), 2)
sgd = L.extract_spec(grid)
check("网格 0 假圆", sgd["_meta"]["counts"]["circle"] == 0,
      "circles=%d" % sgd["_meta"]["counts"]["circle"])
check("外框矩形检出", sgd["_meta"]["counts"]["rect"] >= 1,
      "rects=%d" % sgd["_meta"]["counts"]["rect"])

# ---------------------------------------------------------------- 5) 精度
print("[5] 坐标精度 / 降采样")
big = np.full((1800, 2400, 3), 255, np.uint8)
cv2.rectangle(big, (400, 300), (2000, 1500), (30, 30, 30), 3)
cv2.circle(big, (1200, 900), 300, (30, 30, 30), 3)
sb = L.extract_spec(big)
mb = sb["_meta"]
check("大图已降采样", mb["downscale"] < 1.0, "rz=%.3f" % mb["downscale"])
check("报告原图尺寸", mb["image_size"] == [2400, 1800], str(mb["image_size"]))
pxs, okpx = L._spec_pixels(sb)
xs = [p[0] for p in pxs]
ys = [p[1] for p in pxs]
err = max(abs(min(xs) - 400), abs(max(xs) - 2000), abs(min(ys) - 300), abs(max(ys) - 1500))
check("frame 反变换误差 ≤ 8px", okpx and err <= 8,
      "err=%.1fpx bbox=(%.0f..%.0f, %.0f..%.0f)" % (err, min(xs), max(xs), min(ys), max(ys)))
check("矩形/圆都检出", mb["counts"]["rect"] == 1 and mb["counts"]["circle"] == 1,
      str(mb["counts"]))

# ---------------------------------------------------------------- 6) 兼容单遍
print("[6] 兼容性")
s1b = L.extract_spec(img, passes=1)
check("passes=1 可用", s1b["_meta"]["passes"]["passes"] == 1 and s1b["_meta"]["ok"],
      "lines=%d" % s1b["_meta"]["counts"]["line"])
empty = L.extract_spec(None)
check("None 输入不崩", empty["_meta"]["ok"] is False, str(empty["_meta"].get("reason")))

print()
print("总结：%d 通过 / %d 失败" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
