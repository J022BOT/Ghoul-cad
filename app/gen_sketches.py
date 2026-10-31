"""生成测试草图（几何参数已知，用于验证 CV 提取精度）。"""
from PIL import Image, ImageDraw
import math, os, random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sketches")
os.makedirs(OUT, exist_ok=True)

def new(w=800, h=600):
    im = Image.new("RGB", (w, h), "white")
    return im, ImageDraw.Draw(im)

def jitter_pt(p, amp, rng):
    return (p[0] + rng.uniform(-amp, amp), p[1] + rng.uniform(-amp, amp))

def jline(d, a, b, w, amp, rng):
    d.line([*jitter_pt(a, amp, rng), *jitter_pt(b, amp, rng)], fill="black", width=w)

def jrect(d, box, w, amp, rng):
    x0, y0, x1, y1 = box
    jline(d, (x0, y0), (x1, y0), w, amp, rng)
    jline(d, (x1, y0), (x1, y1), w, amp, rng)
    jline(d, (x1, y1), (x0, y1), w, amp, rng)
    jline(d, (x0, y1), (x0, y0), w, amp, rng)

def jcircle(d, cx, cy, r, w, amp, rng):
    pts = []
    for k in range(48):
        a = 2 * math.pi * k / 48
        rr = r + rng.uniform(-amp, amp)
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    pts.append(pts[0])
    d.line(pts, fill="black", width=w, joint="curve")

# ---------- 1. 房子（干净版 + 手抖版）----------
def house(path, amp=0.0, seed=1):
    rng = random.Random(seed)
    im, d = new()
    jrect(d, [200, 300, 500, 550], 4, amp, rng)            # 墙体
    jline(d, (180, 300), (350, 180), 4, amp, rng)          # 屋顶左
    jline(d, (350, 180), (520, 300), 4, amp, rng)          # 屋顶右
    jrect(d, [280, 450, 360, 550], 4, amp, rng)            # 门
    if amp == 0:
        d.ellipse([390, 340, 470, 420], outline="black", width=4)   # 窗
    else:
        jcircle(d, 430, 380, 40, 4, amp, rng)
    im.save(path)

house(f"{OUT}/sketch_house.png", 0.0)
house(f"{OUT}/sketch_house_shaky.png", 4.0)

# ---------- 2. L 支架（两矩形 + 两孔）----------
def bracket(path, amp=0.0, seed=2):
    rng = random.Random(seed)
    im, d = new()
    jrect(d, [150, 100, 350, 500], 4, amp, rng)            # 竖板
    jrect(d, [380, 380, 650, 500], 4, amp, rng)            # 横板
    if amp == 0:
        d.ellipse([220, 180, 280, 240], outline="black", width=4)   # 孔1 圆心(250,210) r30
        d.ellipse([480, 420, 540, 480], outline="black", width=4)   # 孔2 圆心(510,450) r30
    else:
        jcircle(d, 250, 210, 30, 4, amp, rng)
        jcircle(d, 510, 450, 30, 4, amp, rng)
    im.save(path)

bracket(f"{OUT}/sketch_bracket.png", 0.0)
bracket(f"{OUT}/sketch_bracket_shaky.png", 3.5)

# ---------- 3. 齿轮（2 圆 + 8 辐条）----------
def gear(path, amp=0.0, seed=3):
    rng = random.Random(seed)
    im, d = new(800, 800)
    cx, cy, R, r_in = 400, 400, 250, 60
    if amp == 0:
        d.ellipse([cx - R, cy - R, cx + R, cy + R], outline="black", width=5)
        d.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], outline="black", width=5)
    else:
        jcircle(d, cx, cy, R, 5, amp, rng)
        jcircle(d, cx, cy, r_in, 5, amp, rng)
    for k in range(8):
        a = 2 * math.pi * k / 8
        jline(d, (cx + r_in * math.cos(a), cy + r_in * math.sin(a)),
              (cx + R * math.cos(a), cy + R * math.sin(a)), 4, amp, rng)
    im.save(path)

gear(f"{OUT}/sketch_gear.png", 0.0)
gear(f"{OUT}/sketch_gear_shaky.png", 3.0)

# ---------- 4. 虚线矩形 + 实心圆 ----------
def dashed(path):
    im, d = new()
    x0, y0, x1, y1 = 100, 100, 600, 500
    dash, gap = 20, 14
    x = x0
    while x < x1:
        d.line([x, y0, min(x + dash, x1), y0], fill="black", width=4)
        d.line([x, y1, min(x + dash, x1), y1], fill="black", width=4)
        x += dash + gap
    y = y0
    while y < y1:
        d.line([x0, y, x0, min(y + dash, y1)], fill="black", width=4)
        d.line([x1, y, x1, min(y + dash, y1)], fill="black", width=4)
        y += dash + gap
    d.ellipse([250, 200, 450, 400], outline="black", width=5)   # 圆心(350,300) r100
    im.save(path)

dashed(f"{OUT}/sketch_dashed.png")

# ---------- 5. 五角星（多段线串接测试）----------
def star(path):
    im, d = new(800, 800)
    cx, cy, R = 400, 400, 300
    pts = [(cx + R * math.cos(-math.pi / 2 + 2 * math.pi * k / 5),
            cy + R * math.sin(-math.pi / 2 + 2 * math.pi * k / 5)) for k in range(5)]
    order = [0, 2, 4, 1, 3]
    for i in range(5):
        a, b = pts[order[i]], pts[order[(i + 1) % 5]]
        d.line([*a, *b], fill="black", width=4)
    im.save(path)

star(f"{OUT}/sketch_star.png")

print("生成完毕：")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
