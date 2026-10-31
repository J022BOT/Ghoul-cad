# -*- coding: utf-8 -*-
"""detect_skeleton.py — 从用户标注的骨骼图里自动提取关节坐标。

用法：
  1. 在角色原图上用画笔标注：
       绿线 + 绿圈 = 手臂（绿圈 = 肩关节）
       黄线 + 黄圈 = 腿（黄圈 = 髋关节）
       紫圈     = 眼睛
  2. 存成 assets/pet_skeleton_ref.png
  3. 运行本脚本 → 打印 SKELETON 字典，粘回 make_pet_frames.py

原理：按颜色分离三组标注 → Canny 边缘上做 Hough 圆检测找关节圆圈；
线条末端（手/脚）取该色直线的极值点。
"""
import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "assets", "pet_skeleton_ref.png")

# 标注颜色范围（HSV）
COLOR_RANGES = {
    "green": ((35, 90, 90), (85, 255, 255)),    # 手臂
    "yellow": ((18, 90, 120), (38, 255, 255)),   # 腿
    "purple": ((135, 60, 120), (170, 255, 255)),  # 眼睛
}


def detect():
    if not os.path.exists(REF):
        raise FileNotFoundError(f"缺少标注图 {REF}")
    img = Image.open(REF).convert("RGB")
    a = np.array(img)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    out = {}
    for name, (lo, hi) in COLOR_RANGES.items():
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        edges = cv2.Canny(m, 50, 150)
        circles = []
        for p2 in (30, 40, 50):
            c = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.0, minDist=50,
                                 param1=100, param2=p2, minRadius=16, maxRadius=44)
            if c is None:
                continue
            for x, y, r in c[0].round(0).astype(int):
                if not any(abs(x - u[0]) < 14 and abs(y - u[1]) < 14
                           for u in circles):
                    circles.append((int(x), int(y), int(r)))
        ys, xs = np.where(m > 0)
        out[name] = {"circles": circles,
                      "line_bbox": (int(xs.min()), int(ys.min()),
                                    int(xs.max()), int(ys.max())) if len(xs)
                                    else None}
    return out


def main():
    det = detect()
    for name, d in det.items():
        print(f"[{name}] 关节圆圈: {d['circles']}  线条范围: {d['line_bbox']}")

    # 组装 SKELETON：绿圈=肩、黄圈=髋、紫圈=眼；线末端=手/脚
    g = sorted(det["green"]["circles"], key=lambda c: c[0])
    y = sorted(det["yellow"]["circles"], key=lambda c: c[0])
    p = sorted(det["purple"]["circles"], key=lambda c: c[0])
    if len(g) < 2 or len(y) < 2 or len(p) < 2:
        print("标注识别不完整，请检查标注图")
        return
    gl, gr = g[0], g[-1]
    yl, yr = y[0], y[-1]
    pl, pr = p[0], p[-1]
    gb = det["green"]["line_bbox"]
    yb = det["yellow"]["line_bbox"]
    sk = {
        "shoulder_l": (gl[0], gl[1]),
        "hand_l": (gb[0], gl[1]),
        "shoulder_r": (gr[0], gr[1]),
        "hand_r": (gb[2], gr[1]),
        "hip_l": (yl[0], yl[1]),
        "foot_l": (yl[0], yb[3]),
        "hip_r": (yr[0], yr[1]),
        "foot_r": (yr[0], yb[3]),
        "eye_l": (pl[0], pl[1]),
        "eye_r": (pr[0], pr[1]),
    }
    print("\nSKELETON = {")
    for k, v in sk.items():
        print(f'    "{k}": {v},')
    print("}")


if __name__ == "__main__":
    main()
