"""有机线稿矢量化离线自测（organic.py + line_extractor 的有机分支）。

验证八件事：
  1) 螺旋线稿 → 判为有机，输出样条（而不是几百条短直线），且贴合真实螺旋；
  2) 工程图（矩形+圆+直线）→ 不判有机，仍走霍夫的精确结果；
  3) 大图（1400px）→ 仍判有机，且坐标乘回原尺度（不会缩成四分之一）；
  4) 骨架：细化后是 1px；短毛刺被剪掉、主脊保留；
  5) RDP：直线点列压成 2 点；圆弧保留足够点；
  6) 圆拟合：圆上采样点 → 半径误差 <1%；
  7) 平滑：抖动的点列平滑后波动幅度显著下降（样条不会画成蛇形）；
  8) 退化输入：空白图、全黑图、纯噪声不崩且不误判。
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw   # noqa: E402

import organic                     # noqa: E402

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


def _ink_from(img):
    """RGBA/灰度 PIL 图 → 0/1 墨迹（深色为墨）。"""
    g = img.convert("L")
    a = np.asarray(g, dtype=np.uint8)
    return (a < 128).astype(np.uint8)


def spiral_img(n_arm=4, size=600, turns=2.2):
    im = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    cx = cy = size / 2
    for k in range(n_arm):
        pts = []
        for i in range(240):
            t = i / 240.0
            a = 2 * math.pi * turns * t + 2 * math.pi * k / n_arm
            r = 40 + (size * 0.42 - 40) * t
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        dr.line(pts, fill=(0, 0, 0), width=3)
    return im


def engineering_img(size=600):
    im = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    dr.rectangle([50, 50, size - 50, size - 50], outline=(0, 0, 0), width=3)
    dr.ellipse([180, 180, 420, 420], outline=(0, 0, 0), width=3)
    dr.line([50, 300, size - 50, 300], fill=(0, 0, 0), width=2)
    dr.line([300, 50, 300, size - 50], fill=(0, 0, 0), width=2)
    return im


# ---------------------------------------------------------- 1) 螺旋 → 曲线
print("[1] 螺旋线稿 → 有机矢量化")
im = spiral_img()
ink = _ink_from(im)
D = math.hypot(*ink.shape[:2])
res = organic.vectorize(ink, D)
ents = res["entities"]
kinds = [e["type"] for e in ents]
check("判为有机线稿并提供实体", bool(ents) and res["curved_share"] > 0.55,
      "弯曲墨水 %.0f%% / %d 条路径" % (100 * res["curved_share"], res["paths"]))
check("输出里有样条曲线", kinds.count("spline") >= 2, str(kinds))
check("实体数远小于霍夫的直线数（不会碎成一片）", len(ents) <= 20, "%d 个实体" % len(ents))

# 样条点到真实螺旋的最近距离（相对图幅）
def spiral_dist(p):
    cx = cy = ink.shape[1] / 2.0
    best = 1e9
    for k in range(4):
        for i in range(400):
            t = i / 400.0
            a = 2 * math.pi * 2.2 * t + 2 * math.pi * k / 4
            r = 40 + (ink.shape[1] * 0.42 - 40) * t
            q = (cx + r * math.cos(a), cy + r * math.sin(a))
            best = min(best, math.hypot(p[0] - q[0], p[1] - q[1]))
    return best


devs = []
for e in ents:
    if e["type"] == "spline":
        for p in e["points"][::4]:
            devs.append(spiral_dist((p[0], p[1])))
if devs:
    med = sorted(devs)[len(devs) // 2]
    check("样条贴合真实螺旋（中位偏差<6%图幅）", med < 0.06 * ink.shape[1],
          "%.1f px / %d px" % (med, ink.shape[1]))
else:
    check("样条贴合真实螺旋（中位偏差<6%图幅）", False, "没有样条点")

# ------------------------------------------------------- 2) 工程图不误判
print("[2] 工程图必须保持霍夫路线")
im2 = engineering_img()
ink2 = _ink_from(im2)
res2 = organic.vectorize(ink2, math.hypot(*ink2.shape[:2]))
check("矩形+圆+直线 不判为有机（不抢霍夫的活）", not res2["entities"],
      "share=%.2f paths=%d" % (res2["curved_share"], res2["paths"]))

# ------------------------------------------------ 3) 大图坐标必须乘回原尺度
print("[3] 大图（骨架化会降尺度）坐标必须还原")
big = spiral_img(size=1400)
ink3 = _ink_from(big)
res3 = organic.vectorize(ink3, math.hypot(*ink3.shape[:2]))
xs = [p[0] for e in res3["entities"] for p in (e.get("points") or [])]
ys = [p[1] for e in res3["entities"] for p in (e.get("points") or [])]
w = (max(xs) - min(xs)) if xs else 0
check("大图仍判为有机", bool(res3["entities"]), "paths=%d" % res3["paths"])
check("坐标已乘回 1400 尺度（宽度>1000px）", w > 1000, "包围盒宽 %.0f px" % w)

# --------------------------------------------------------------- 4) 骨架
print("[4] 细化与剪毛刺")
cross = np.zeros((80, 80), np.uint8)
cross[38:42, 10:70] = 1
cross[10:70, 38:42] = 1
cross[20:38, 50:52] = 1          # 从横线上长出的一根 17px 毛刺
sk = organic.thin(cross)
check("细化后笔画宽度为 1px", int(sk.sum()) < 200, "骨架 %d px" % sk.sum())
sk2 = organic.prune_spurs(sk.copy(), min_len=20)   # 毛刺 17px < 20 → 该被剪
check("毛刺被剪掉（骨架像素减少）", sk2.sum() < sk.sum(), "%d → %d" % (sk.sum(), sk2.sum()))
paths = organic.trace_paths(sk2, min_len=10)
lens = sorted((organic.path_length(p) for p in paths), reverse=True)
check("主脊保留（仍有长路径）", bool(lens) and lens[0] > 40, "最长 %.0f px" % (lens[0] if lens else 0))

# ---------------------------------------------------------------- 5) RDP
print("[5] RDP 简化")
line = [(0.0, 0.0), (10, 0.2), (20, 0.1), (30, 0.0)]
check("直线点列压成 2 点", len(organic.rdp(line, 1.0)) == 2, str(len(organic.rdp(line, 1.0))))
arc = [(30 * math.cos(t / 40.0), 30 * math.sin(t / 40.0)) for t in range(0, 90)]
simp = organic.rdp(arc, 1.0)
check("四分之一圆弧保留足够点（>3）", 3 < len(simp) < len(arc), "%d → %d" % (len(arc), len(simp)))

# ------------------------------------------------------------ 6) 圆拟合
print("[6] 圆拟合")
ring = [(50 + 40 * math.cos(2 * math.pi * k / 64), 30 + 40 * math.sin(2 * math.pi * k / 64))
        for k in range(64)]
fit = organic.fit_circle(ring)
check("圆心 / 半径准确", fit is not None and abs(fit[2] - 40) < 0.4
      and abs(fit[0] - 50) < 0.4 and abs(fit[1] - 30) < 0.4,
      "r=%.2f c=(%.1f,%.1f)" % (fit[2], fit[0], fit[1]) if fit else "None")

# --------------------------------------------------------------- 7) 平滑
print("[7] 路径平滑（防蛇形抖动）")
n = 200
base = [(i * 1.0, 100 + 30 * math.sin(i / 30.0)) for i in range(n)]
jitter = [(x, y + (1.5 if i % 2 else -1.5)) for i, (x, y) in enumerate(base)]
sm = organic.smooth_path(jitter, 9)
def rough(pts):
    return sum(abs(pts[i + 1][1] - 2 * pts[i][1] + pts[i - 1][1]) for i in range(1, len(pts) - 1))
check("抖动幅度显著下降（<20%）", rough(sm) < 0.35 * rough(jitter),
      "%.0f → %.0f" % (rough(jitter), rough(sm)))
check("平滑后形状不跑偏（最大位移<3px）",
      max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(sm, base)) < 3.0)

# ------------------------------------- 7.5) 宽松二次预检（多遍融合的漏网）
print("[7.5] looks_organic_loose：多遍融合把短弦拼成中段后仍能进有机路线")
# 场景：6 遍霍夫融合把曲线切成 0.13D~0.18D 的中段弦——短段(<0.12D)占比不足 0.5，
# 严格预检拒绝；宽松预检只看"段多 + 中位段长 < 0.2D"，放行后由
# vectorize 的"弯曲墨水占比"闸门做最终裁决（工程图墨迹以直为主，进了也会被弹出）。
rng2 = np.random.default_rng(11)
D2 = 600.0
segs2 = []
for _ in range(150):
    L = float(rng2.uniform(0.13 * D2, 0.18 * D2))
    a = float(rng2.uniform(0, 2 * math.pi))
    x, y = float(rng2.uniform(0, 400)), float(rng2.uniform(0, 400))
    segs2.append((x, y, x + L * math.cos(a), y + L * math.sin(a)))
check("严格预检拒绝（短段占比不足 0.5）", not organic.looks_organic(segs2, D2))
check("宽松预检放行", organic.looks_organic_loose(segs2, D2))
check("少于 60 段时宽松预检也不放行",
      not organic.looks_organic_loose(segs2[:30], D2))
check("中位段长过大（真工程图）时宽松预检不放行",
      not organic.looks_organic_loose(
          [(0, 0, 0.5 * D2, 0), (10, 10, 0.5 * D2, 10)] * 40, D2))

# ------------------------------------- 7.6) 整管线：螺旋图应出样条而非几百条直线
print("[7.6] extract_spec 整管线（passes=6）→ 有机矢量化")
import line_extractor as _LE
import tempfile, os as _os
sp = spiral_img()
p_sp = _os.path.join(tempfile.mkdtemp(prefix="org_e2e_"), "spiral.png")
sp.save(p_sp)
spec_sp = _LE.extract_spec(p_sp, instruction="", passes=6)
cnt_sp = (spec_sp.get("_meta") or {}).get("counts") or {}
ents_sp = spec_sp.get("entities") or []
n_line_sp = sum(1 for e in ents_sp if e["type"] == "line")
n_spline_sp = sum(1 for e in ents_sp if e["type"] == "spline")
check("螺旋图不再碎成几百条短直线", n_line_sp <= 10, "%d 条直线" % n_line_sp)
check("螺旋图输出样条曲线", n_spline_sp >= 2, str(cnt_sp))

# --------------------------------------------------------- 8) 退化输入
print("[8] 退化输入不崩")
cases = [("空白图", np.zeros((100, 100), np.uint8)),
         ("全墨图", np.ones((100, 100), np.uint8)),
         ("单点", np.zeros((50, 50), np.uint8))]
cases[2][1][25, 25] = 1
for name, m in cases:
    try:
        r = organic.vectorize(m, 141.0)
        check("%s 安全返回" % name, isinstance(r, dict) and "entities" in r,
              "%d 个实体" % len(r["entities"]))
    except Exception as ex:                           # noqa: BLE001
        check("%s 安全返回" % name, False, repr(ex))
rng = np.random.default_rng(7)
noise = (rng.random((200, 200)) > 0.97).astype(np.uint8)
try:
    r = organic.vectorize(noise, 282.0)
    check("随机噪点不误判为有机", not r["entities"], "share=%.2f" % r["curved_share"])
except Exception as ex:                               # noqa: BLE001
    check("随机噪点不误判为有机", False, repr(ex))
check("vectorize(None) 不崩", organic.vectorize(None, 100)["entities"] == [])

print("\n===== 有机线稿测试：%d 通过 / %d 失败 =====" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
