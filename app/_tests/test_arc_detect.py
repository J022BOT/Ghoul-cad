"""圆弧识别自测（line_extractor 圆/弧判定 + dxf_writer ARC 往返）。

这次修的 bug：一段圆弧被霍夫拟合成圆后，被当成**完整的圆**画出来 —— 吊钩图里
凭空多出两个整圆，其中两个还悬在标注文字上。修法是：识别阶段除了"像不像圆"，
再在墨迹图上量"这一圈到底有多少角度真的有墨迹"，≥90% 才算整圆，否则输出
圆弧（DXF ARC），量不出连续弧的候选直接丢掉。

验证九件事：
  1) 合成整圆 → circle，不能变成 arc（这条一旦错，圆全变弧，比原 bug 还糟）；
  2) 合成 240° 圆弧 → arc，且弧长与真实值相差 ≤ 25°（以前是 circle）；
  3) 合成 120° 圆弧 → **不能**出现 circle（宁可不画，也不能画成整圆）；
  4) 吊钩真实图 → 一个圆都没有，至少 2 条圆弧，且有一条半径≈R29（44.6mm ±20%）；
  5) 圆弧角度方向：图像上半的圆弧，导出后必须在世界坐标上半（验证 y 翻转没搞反）；
  6) 圆弧端点必须落在墨迹上（角度算错会落到空白处，一眼就能查出来）；
  7) DXF 往返：arc → 写出 ARC → 读回仍是 arc，圆心/半径/弧长误差 < 1%；
  8) 退化输入不崩：半径 0、空图、全白图；
  9) 耗时：吊钩图整条管线 < 8 秒（墨迹判定加进来后不能拖垮性能）。
"""
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import dxf_writer
import line_extractor as LE

PASS = 0
FAIL = 0
HOOK = r"C:\Users\ASUS\Desktop\AI饲料\v2-9ac98fc27e3124daeae882f54d4f2e01_r.jpg"


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  OK   %-42s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-42s %s" % (name, detail))


def render(draw, size=420):
    img = np.full((size, size, 3), 255, np.uint8)
    draw(img)
    p = os.path.join(tempfile.mkdtemp(prefix="arc_"), "syn.png")
    cv2.imwrite(p, img)
    return p


def scene(img):
    """两条横线当"图框"：让质量门（图元数/总长）能过，不影响圆判定。"""
    cv2.line(img, (20, 20), (400, 20), (0, 0, 0), 3)
    cv2.line(img, (20, 400), (400, 400), (0, 0, 0), 3)


def counts(spec):
    out = {}
    for e in spec.get("entities") or []:
        t = e.get("type")
        out[t] = out.get(t, 0) + 1
    return out


def arcs_of(spec):
    return [e for e in spec.get("entities") or [] if e.get("type") == "arc"]


def span(e):
    return (float(e["end"]) - float(e["start"])) % 360.0 or 360.0


def main():
    print("[1] 合成整圆：必须是 circle，不能被判成圆弧")
    p = render(lambda im: (cv2.circle(im, (210, 210), 120, (0, 0, 0), 5), scene(im)))
    spec = LE.extract_spec(p, max_dim=150)
    c = counts(spec)
    check("整圆 → 1 个 circle", c.get("circle", 0) == 1, str(c))
    check("整圆没有变成 arc", c.get("arc", 0) == 0, str(c))

    print("\n[2] 合成 240° 圆弧：必须是 arc（修 bug 的主场景）")
    p = render(lambda im: (cv2.ellipse(im, (210, 210), (120, 120), 0, 200, 440,
                                       (0, 0, 0), 5), scene(im)))
    spec = LE.extract_spec(p, max_dim=150)
    c = counts(spec)
    a = arcs_of(spec)
    check("240° 弧不再被当成整圆", c.get("circle", 0) == 0, str(c))
    check("240° 弧 → 至少 1 条 arc", len(a) >= 1, str(c))
    if a:
        s = span(a[0])
        check("弧长接近 240°（±25°）", abs(s - 240) <= 25, "span=%.0f°" % s)
        check("半径接近 46.8mm（±8%）", abs(a[0]["radius"] - 46.8) <= 46.8 * 0.08,
              "r=%.2f" % a[0]["radius"])

    print("\n[3] 合成 120° 圆弧：绝不能出现 circle")
    p = render(lambda im: (cv2.ellipse(im, (210, 210), (120, 120), 0, 200, 320,
                                       (0, 0, 0), 5), scene(im)))
    spec = LE.extract_spec(p, max_dim=150)
    c = counts(spec)
    check("短弧不吐整圆", c.get("circle", 0) == 0, str(c))
    if c.get("arc"):
        s = span(arcs_of(spec)[0])
        check("检出的弧长不夸张（≤160°）", s <= 160, "span=%.0f°" % s)

    print("\n[4] 吊钩真实图：不该有整圆，应有圆弧")
    if not os.path.exists(HOOK):
        print("  [跳过] 找不到测试图 %s" % HOOK)
    else:
        t0 = time.time()
        spec = LE.extract_spec(HOOK, max_dim=180)
        dt = time.time() - t0
        c = counts(spec)
        a = arcs_of(spec)
        check("吊钩图 0 个整圆（原来是 4 个）", c.get("circle", 0) == 0, str(c))
        check("至少 2 条圆弧", len(a) >= 2, "%d 条" % len(a))
        big = [e for e in a if e["radius"] >= 30]
        check("存在 R29 级别的大圆弧（r≥30mm）", bool(big),
              "、".join("r=%.1f" % e["radius"] for e in a[:4]))
        if big:
            check("大圆弧半径落在 35~55mm", 35 <= big[0]["radius"] <= 55,
                  "r=%.2f" % big[0]["radius"])
        check("耗时 < 8 秒", dt < 8.0, "%.2fs" % dt)

        print("\n[5] 角度方向：上半圆的弧必须落在世界坐标上半")
        # 图像坐标里 190°~350° 是"上半"（y 向下），翻转到世界坐标后仍应是上半（y>中心）
        p2 = render(lambda im: (cv2.ellipse(im, (210, 210), (120, 120), 0, 190, 350,
                                            (0, 0, 0), 5), scene(im)))
        s2 = LE.extract_spec(p2, max_dim=150)
        a2 = arcs_of(s2)
        if not a2:
            print("  [跳过] 这张合成图没检出圆弧")
        else:
            e = a2[0]
            cy = e["center"][1]
            ys = []
            for ang in (e["start"], e["end"]):
                r = math.radians(ang)
                ys.append(cy + e["radius"] * math.sin(r))
            check("弧端点都在世界坐标上半（y > 中心）",
                  all(y > cy for y in ys), "cy=%.1f ys=%s" % (cy, [round(v, 1) for v in ys]))

        print("\n[6] 端点必须压在墨迹上（角度算错会落到空白）")
        img = cv2.imdecode(np.fromfile(HOOK, dtype=np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        field = LE.ink_field(gray)
        bw, _thr = LE._binarize_dist(field)
        fcx, fcy, fs = spec["_meta"]["frame"]
        bad = 0
        for e in a:
            for ang in (e["start"], e["end"]):
                r = math.radians(ang)
                x = e["center"][0] + e["radius"] * math.cos(r)
                y = e["center"][1] + e["radius"] * math.sin(r)
                px, py = int(round(fcx + x / fs)), int(round(fcy - y / fs))
                if not (0 <= px < bw.shape[1] and 0 <= py < bw.shape[0]):
                    bad += 1
                    continue
                win = bw[max(0, py - 3):py + 4, max(0, px - 3):px + 4]
                if win.size == 0 or win.max() <= 127:
                    bad += 1
        check("所有弧端点都压在墨迹上", bad == 0, "偏离 %d 个" % bad)

        print("\n[7] DXF 往返：arc 写出 ARC 后能读回成 arc")
        out = os.path.join(tempfile.mkdtemp(prefix="arcdxf_"), "arc.dxf")
        dxf = dxf_writer.build_from_entities(spec["entities"], unit="mm")
        dxf.save(out)
        back = dxf_writer.parse_dxf_to_entities(out)
        back_arcs = [e for e in back if e.get("type") == "arc"]
        check("写盘无异常图元", dxf.skipped == 0, "skipped=%d" % dxf.skipped)
        check("读回仍为 arc 类型", len(back_arcs) == len(a),
              "写 %d 读 %d" % (len(a), len(back_arcs)))
        if back_arcs:
            o, b = a[0], back_arcs[0]
            dr = abs(o["radius"] - b["radius"]) / max(o["radius"], 1e-9)
            dspan = abs(span(o) - span(b))
            check("半径误差 < 1%", dr < 0.01, "%.4f" % dr)
            check("弧长误差 < 2°", dspan < 2.0, "%.2f°" % dspan)
            dc = math.hypot(o["center"][0] - b["center"][0],
                            o["center"][1] - b["center"][1])
            check("圆心误差 < 0.1mm", dc < 0.1, "%.4f" % dc)

    print("\n[8] 退化输入不崩")
    try:
        LE._circle_ink_profile(np.zeros((10, 10), np.uint8),
                               np.zeros((10, 10), np.uint8), 5, 5, 0)
        LE._circle_ink_profile(np.zeros((10, 10), np.uint8),
                               np.zeros((10, 10), np.uint8), 5, 5, 3)
        blank = render(lambda im: None)
        s = LE.extract_spec(blank, max_dim=100)
        check("空白图不崩且无图元", not (s.get("entities") or []),
              str((s.get("_meta") or {}).get("reason")))
    except Exception as e:                                  # noqa: BLE001
        check("空白图不崩且无图元", False, repr(e))
    check("空 hit 不崩", LE._circle_ink_profile(
        np.zeros((8, 8), np.uint8), np.zeros((8, 8), np.uint8), 4, 4, 1) is None)

    # ------------------------------------- 10) 短弦串弧：碎弦 → ARC
    print("\n[10] _chords_to_arcs：共圆生长")
    rng = np.random.default_rng(7)
    # 正例：140° 圆弧（r=120，中心 (250,250)）切成 15 段带抖动的短弦
    cx_t, cy_t, r_t = 250.0, 250.0, 120.0
    a_start, a_end = 20.0, 160.0                      # 图像坐标角度
    n_ch = 15
    ang = np.linspace(math.radians(a_start), math.radians(a_end), n_ch + 1)
    chord_segs = []
    for k in range(n_ch):
        x0 = cx_t + r_t * math.cos(ang[k]);  y0 = cy_t + r_t * math.sin(ang[k])
        x1 = cx_t + r_t * math.cos(ang[k + 1]); y1 = cy_t + r_t * math.sin(ang[k + 1])
        jitter = rng.uniform(-1.5, 1.5, 4)
        chord_segs.append([x0 + jitter[0], y0 + jitter[1],
                           x1 + jitter[2], y1 + jitter[3]])
    # 顺便放几条不相干的直线，验证不会被误收
    chord_segs += [[10, 10, 200, 12], [12, 14, 210, 16], [300, 400, 480, 402]]
    arc_img = np.zeros((500, 500), np.uint8)
    cv2.ellipse(arc_img, (250, 250), (120, 120), 0, a_start, a_end, 255, 3)
    dil = cv2.dilate(arc_img, np.ones((5, 5), np.uint8))
    rest, got = LE._chords_to_arcs(chord_segs, 700.0, dil=dil)
    check("碎弦合成出 1 道弧", len(got) == 1, str(len(got)))
    if got:
        gcx, gcy, gr, ga0, ga1 = got[0]
        check("半径误差 ≤6%", abs(gr - r_t) / r_t <= 0.06, "r=%.1f" % gr)
        gspan = (ga1 - ga0) % 360.0
        check("角跨度 ≈140°（±25°）", abs(gspan - 140.0) <= 25.0,
              "span=%.0f" % gspan)
        check("圆心误差 ≤6px", math.hypot(gcx - cx_t, gcy - cy_t) <= 6.0,
              "c=(%.0f,%.0f)" % (gcx, gcy))
    n_eat = 15 - sum(1 for i in range(15)
                     if any(math.hypot(rest_q[0] - chord_segs[i][0],
                                       rest_q[1] - chord_segs[i][1]) < 0.5
                            and len(rest_q) == 4 for rest_q in rest))
    check("大部分弦被收编（剩余直线 ≤12 条）", len(rest) <= 12, str(len(rest)))
    # 反例：星形交叉直线不应产生弧
    cross = []
    for k in range(8):
        a = math.pi * k / 8.0
        cross.append([250 - 200 * math.cos(a), 250 - 200 * math.sin(a),
                      250 + 200 * math.cos(a), 250 + 200 * math.sin(a)])
    cross_img = np.zeros((500, 500), np.uint8)
    for s in cross:
        cv2.line(cross_img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), 255, 3)
    _, got2 = LE._chords_to_arcs([list(map(float, s)) for s in cross],
                                 700.0, dil=cv2.dilate(cross_img, np.ones((5, 5), np.uint8)))
    check("交叉直线不产生假弧", len(got2) == 0, str(len(got2)))

    print("\n%d 通过 / %d 失败" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
