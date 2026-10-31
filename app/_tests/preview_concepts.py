"""把概念库里的每个物体画成预览格子图，用来肉眼检查形状是否可辨认。

用法：
    python _tests/preview_concepts.py            # 生成 3 张图
    python _tests/preview_concepts.py 0 18       # 只画第 0~17 个
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont   # noqa: E402

import concepts as C   # noqa: E402

CELL_W, CELL_H = 250, 250
COLS, ROWS = 6, 3
MARGIN = 18
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"


def _font(size=16):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def _draw_entities(dr, ents, box, font):
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = C._bbox(ents)
    bw, bh = max(1e-6, bx1 - bx0), max(1e-6, by1 - by0)
    s = min((x1 - x0) / bw, (y1 - y0) / bh)

    def m(x, y):
        # 世界坐标（y 向上）→ 屏幕坐标（y 向下），居中
        px = (x0 + x1) / 2 + (x - (bx0 + bx1) / 2) * s
        py = (y0 + y1) / 2 - (y - (by0 + by1) / 2) * s
        return (px, py)

    for e in ents:
        t = e["type"]
        if t == "line":
            dr.line([m(*e["p1"]), m(*e["p2"])], fill=(25, 25, 25), width=2)
        elif t == "circle":
            cx, cy = m(*e["center"])
            r = e["radius"] * s
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(25, 25, 25), width=2)
        elif t == "rect":
            a = m(e["x"], e["y"])
            b = m(e["x"] + e["w"], e["y"] + e["h"])
            dr.rectangle([min(a[0], b[0]), min(a[1], b[1]),
                          max(a[0], b[0]), max(a[1], b[1])],
                         outline=(25, 25, 25), width=2)
        elif t == "polyline":
            pts = [m(*p) for p in e["points"]]
            if len(pts) >= 2:
                if e.get("closed"):
                    pts = pts + [pts[0]]
                dr.line(pts, fill=(25, 25, 25), width=2, joint="curve")


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(C.CONCEPTS)
    subset = C.CONCEPTS[start:end]

    per_img = COLS * ROWS
    imgs = []
    for page in range((len(subset) + per_img - 1) // per_img):
        chunk = subset[page * per_img:(page + 1) * per_img]
        img = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), "white")
        dr = ImageDraw.Draw(img)
        font = _font(16)
        for i, c in enumerate(chunk):
            r, col = divmod(i, COLS)
            cx0, cy0 = col * CELL_W, r * CELL_H
            # 网格
            dr.rectangle([cx0 + 2, cy0 + 2, cx0 + CELL_W - 3, cy0 + CELL_H - 3],
                         outline=(225, 225, 225))
            w, h = c["size"]
            ents = C._fit(C._center(c["recipe"](w, h)), w, h)
            _draw_entities(dr, ents, (cx0 + MARGIN, cy0 + MARGIN + 12,
                                      cx0 + CELL_W - MARGIN, cy0 + CELL_H - MARGIN - 12),
                           font)
            dr.text((cx0 + 8, cy0 + 6), "%s  %g×%g" % (c["cn"], w, h),
                    fill=(20, 20, 20), font=font)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_concepts_%d.png" % (start + page * per_img))
        img.save(out)
        imgs.append(out)
        print("写出", out)
    return imgs


if __name__ == "__main__":
    main()
