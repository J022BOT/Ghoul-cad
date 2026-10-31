# -*- coding: utf-8 -*-
"""make_pet_frames.py — 桌宠骨骼绑定与动画帧生成。

输入：
  assets/pet_clean.png        干净透明图（1280x1280，带 alpha）
  assets/pet_skeleton_ref.png 用户标注了骨骼的参考图（用于自检）

骨骼：用户在 22带骨骼.png 上用线条和圆圈标出
  绿线/绿圈 = 手臂（肩关节 → 手）
  黄线/黄圈 = 腿（髋关节 → 脚）
  紫圈     = 眼睛
下面 SKELETON 常量即从该图自动检测得到的坐标（1280x1280 源图系）。

肢体分离：骨线走廊 + 线稿吸附
  ① 用骨线构造"肢体走廊"，限定候选区域
  ② 把深色描边当作边界（线稿），在"非描边"像素上做连通域
  ③ 走廊内占比高的连通域 = 该肢体（袖子/手/腿），头发等因有描边被自动排除
  ④ 肢体旋转 = 绕肩/髋关节旋转，合成回底图

输出：
  assets/pet_base.png    去掉四肢的底图
  assets/pet_eyes.json   眼睛坐标
  assets/frames/*.png    全套动画帧

用法：python make_pet_frames.py
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_CLEAN = os.path.join(HERE, "assets", "pet_clean.png")
SRC_SKEL = os.path.join(HERE, "assets", "pet_skeleton_ref.png")
BASE_OUT = os.path.join(HERE, "assets", "pet_base.png")
EYES_OUT = os.path.join(HERE, "assets", "pet_eyes.json")
FRAMES_DIR = os.path.join(HERE, "assets", "frames")
DEBUG_DIR = os.path.join(HERE, "_debug")

# 桌宠角色目标高度（屏幕像素）
PET_H = 300
PAD_TOP, PAD_BOTTOM = 56, 8
PAD_X = 48

# ---------------------------------------------------------------- 骨骼坐标
# 来自用户标注图 assets/pet_skeleton_ref.png（1280x1280 源图坐标系）
# 圆圈 = 关节（旋转中心），线的另一端 = 肢体末端
SKELETON = {
    "shoulder_l": (530, 580),   # 左肩（画面左）
    "hand_l": (216, 554),
    "shoulder_r": (724, 580),   # 右肩（画面右）
    "hand_r": (1050, 596),
    "hip_l": (566, 842),
    "foot_l": (566, 1234),
    "hip_r": (690, 842),
    "foot_r": (690, 1232),
    "eye_l": (544, 400),
    "eye_r": (708, 392),
}

# 肢体走廊参数
ARM_DEPTH = 185        # 手臂主体在骨线下方的深度
ARM_TOP = 30           # 骨线上方保留的余量
ARM_EXTEND = 90        # 骨线两端延长（把整只手/脚纳入走廊）
LEG_HALF = 46          # 腿的半宽
LEG_EXTEND = 40        # 腿的骨线两端延长
LINEART_THR = 125      # 描边判定阈值（亮度低于此值算描边）
POLY_DILATE = 3        # 手臂多边形外扩（把描边残影一并切掉）

# 静止姿态：手臂从原画的 T 字型旋到垂臂（绕用户标注的肩关节）
# PIL 的 rotate() 正值 = 逆时针；左臂逆时针 62°、右臂顺时针 62° 即下垂。
ARM_REST_L = 62
ARM_REST_R = -62

# ---------------------------------------------------------------- 轮廓多边形
# 全部来自用户标注图 assets/pet_skeleton_ref.png（1280x1280 源图坐标）：
#   绿色闭合轮廓 = 整条手臂（袖子+手）
#   黄色闭合轮廓 = 整条腿（腿+鞋）
# 由 _debug/extract_limb_polys.py 提取。
ARM_POLY_L = [
    (556, 576), (550, 563), (541, 556), (526, 554), (515, 559), (492, 557),
    (463, 538), (459, 546), (456, 567), (423, 567), (405, 559), (404, 570),
    (411, 582), (410, 585), (384, 580), (376, 575), (352, 584), (324, 573),
    (274, 571), (267, 584), (251, 584), (250, 588), (236, 580), (227, 579),
    (225, 594), (229, 597), (228, 600), (179, 599), (174, 607), (174, 624),
    (185, 639), (197, 648), (236, 655), (244, 659), (245, 684), (260, 705),
    (260, 733), (267, 734), (314, 720), (411, 676), (428, 671), (442, 662),
    (472, 655), (504, 641), (525, 638), (552, 604), (550, 597), (555, 589),]

ARM_POLY_R = [
    (698, 575), (702, 595), (720, 609), (730, 623), (732, 635), (757, 649),
    (770, 664), (786, 665), (796, 653), (855, 675), (904, 699), (911, 709),
    (953, 719), (994, 736), (1001, 735), (1000, 710), (1015, 687),
    (1013, 658), (1029, 652), (1049, 652), (1067, 647), (1081, 639),
    (1096, 623), (1095, 615), (1070, 597), (1048, 595), (1039, 598),
    (1038, 579), (1003, 586), (988, 572), (958, 573), (929, 582),
    (903, 582), (895, 574), (882, 574), (868, 583), (856, 584), (855, 581),
    (863, 566), (862, 562), (832, 569), (816, 569), (811, 565), (807, 555),
    (805, 535), (791, 541), (782, 549), (771, 533), (757, 548), (746, 548),
    (736, 553), (716, 553), (703, 563),]

LEG_POLY_L = [
    (560, 805), (537, 814), (522, 835), (506, 891), (505, 924),
    (530, 1050), (530, 1081), (526, 1095), (527, 1128), (537, 1181),
    (547, 1209), (549, 1231), (562, 1245), (600, 1245), (609, 1236),
    (606, 1126), (610, 1110), (610, 1037), (615, 984), (615, 929),
    (607, 885), (604, 854), (606, 836), (603, 827), (588, 811), (575, 806),]

LEG_POLY_R = [
    (696, 805), (674, 808), (656, 821), (649, 837), (649, 852), (641, 864),
    (633, 892), (639, 1025), (644, 1048), (647, 1090), (652, 1106),
    (654, 1152), (659, 1182), (659, 1226), (654, 1237), (660, 1248),
    (671, 1251), (693, 1248), (720, 1236), (736, 1150), (736, 1081),
    (727, 1052), (727, 1028), (743, 943), (741, 872), (733, 832),
    (727, 821), (717, 812),]


# 身体保留区域（去掉两侧展开的长发）：
#   由用户在桌宠截图上勾画后，_debug/calibrate_body.py 反算回源图坐标，
#   结果存成掩膜图 assets/pet_body_mask.png（255=保留）。
#   注意：不能用多边形填充 —— 用户勾的线是**开放路径**且在手臂处自交，
#   fillPoly 的奇偶规则会漏掉躯干（出现空心）；掩膜是泛洪出来的，无此问题。
BODY_MASK_PATH = os.path.join(HERE, "assets", "pet_body_mask.png")



# ---------------------------------------------------------------- 基础工具
def load_source():
    """载入源图（透明 RGBA）。"""
    if not os.path.exists(SRC_CLEAN):
        raise FileNotFoundError(f"缺少源图 {SRC_CLEAN}")
    img = Image.open(SRC_CLEAN).convert("RGBA")
    print(f"源图：{SRC_CLEAN}  {img.size}")
    return img


def lineart_mask(arr, alpha, thr=LINEART_THR, shell=2.0):
    """轮廓边界掩膜（用作肢体分割的边界）。

    两种边界取并集：
      ① "细描边"：暗色区域只取外沿 shell 像素的壳。
         —— 若直接用亮度阈值，手/袖口这类大块深色会被整块当成描边，
            导致肢体分离不出来；只留外沿后，块内部仍是完整区域。
      ② Canny 轮廓线：补上浅色区域的细边界。
    """
    lum = arr[..., :3].mean(axis=2)
    dark = ((lum < thr) & alpha).astype(np.uint8)
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 3)
    thin = (dark > 0) & (dist <= shell)
    gray = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 130)
    return (thin | (edges > 0)) & alpha


def corridor_below(shape, a, b, depth, top=ARM_TOP, extend=0):
    """骨线以下的走廊：肢体主体在骨线下方。

    extend：沿骨线方向向两端延长，用于把末端（手/脚）完整纳入走廊。
    """
    H, W = shape
    ax, ay = a
    bx, by = b
    if extend:
        dx, dy = bx - ax, by - ay
        ln = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / ln, dy / ln
        ax, ay = ax - ux * extend, ay - uy * extend
        bx, by = bx + ux * extend, by + uy * extend
    poly = np.array([[ax, ay - top], [bx, by - top],
                     [bx, by + depth], [ax, ay + depth]], dtype=np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def corridor_band(shape, a, b, half_w, extend=20):
    """沿骨线的带状走廊（用于腿这种细长肢体）。"""
    H, W = shape
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    ln = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / ln, dy / ln
    ax, ay = ax - ux * extend, ay - uy * extend
    bx, by = bx + ux * extend, by + uy * extend
    nx, ny = -uy, ux
    poly = np.array([[ax + nx * half_w, ay + ny * half_w],
                     [bx + nx * half_w, by + ny * half_w],
                     [bx - nx * half_w, by - ny * half_w],
                     [ax - nx * half_w, ay - ny * half_w]], dtype=np.int32)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def poly_mask(shape, poly):
    """多边形掩膜。"""
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, dtype=np.int32)], 1)
    return m.astype(bool)


def separate_by_poly(alpha, boundary, poly, ratio=0.5):
    """按多边形切分肢体。

    多边形已沿画稿轮廓手工标定，直接取 多边形 ∩ 角色像素 即可，
    再把紧贴的轮廓线补上，避免边缘出现白边。

    注意 alpha 阈值要放宽：源图是调色板 PNG，画师的手套是"半透明深色"
    （alpha 仅 35~135），用 alpha>200 会把手整块漏掉。
    solid=True 时把部件内的 alpha 固化为 255，让半透明手套恢复成实心。
    """
    pm = poly_mask(alpha.shape, poly)
    # 外扩：把紧贴手臂的描边线也纳入，避免原 T 字型手臂留下残影
    pm = cv2.dilate(pm.astype(np.uint8), np.ones((5, 5), np.uint8),
                    iterations=POLY_DILATE) > 0
    piece = pm & alpha
    # 补上紧贴肢体边缘的描边线（避免切出白边）
    dil = cv2.dilate(piece.astype(np.uint8), np.ones((5, 5), np.uint8), 1)
    piece |= boundary & (dil > 0)
    piece &= (cv2.dilate(pm.astype(np.uint8),
                         np.ones((5, 5), np.uint8), 1) > 0)
    return piece


def inpaint_hole(arr, hole, radius=7):
    """把移除肢体留下的空洞用周围像素修补（cv2.inpaint）。

    只修补 RGB，alpha 保持原样（空洞处仍为透明，靠旋转回来的肢体遮住）。
    """
    if not hole.any():
        return arr
    src = arr[..., :3].astype(np.uint8)
    mask = (hole & (arr[..., 3] > 20)).astype(np.uint8) * 255
    out = cv2.inpaint(src, mask, radius, cv2.INPAINT_TELEA)
    res = arr.copy()
    res[..., :3] = out
    return res


def separate_limb(alpha, boundary, corr, ratio=0.5, min_area=60):
    """走廊 + 轮廓吸附，分离出一段肢体。返回布尔掩膜。

    步骤：
      ① 在"非轮廓"像素上求连通域（袖子/手/袖口各自成块）
      ② 走廊内占比高的连通域 = 该肢体
      ③ 把与肢体块相邻的轮廓线也带上（轮廓属于肢体自身轮廓）
    """
    free = (alpha & ~boundary).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free, 8)
    piece = np.zeros_like(alpha)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp = (lab == i)
        if int((comp & corr).sum()) > ratio * area:
            piece |= comp
    # 带上与肢体块直接相邻的轮廓线
    dil = cv2.dilate(piece.astype(np.uint8), np.ones((5, 5), np.uint8), 1)
    piece |= boundary & (dil > 0)
    # 再把肢体块向外扩一圈，把紧贴的轮廓线补进来
    dil2 = cv2.dilate(piece.astype(np.uint8), np.ones((7, 7), np.uint8), 1)
    piece |= (dil2 > 0) & boundary
    # 限制在走廊邻域内，避免误拉远处描边
    near = cv2.dilate(corr.astype(np.uint8), np.ones((17, 17), np.uint8), 1)
    return piece & near.astype(bool)


# ---------------------------------------------------------------- 装配
def _mirror_fill_leg(arr, m_leg_l, m_leg_r, S, LEG_POLY_R):
    """用镜像左腿补全右腿的局部缺失。

    返回 (新右腿掩膜, 新填充像素掩膜, 镜像左腿颜色表)。

    做法：把左腿绕角色对称轴（两髋中点）水平镜像，得到"理想右腿"；
    在右腿预期多边形(膨胀)范围内，从现有右腿边缘向内生长，只接纳
    "镜像左腿在此处有像素"的透明点，从而把右腿缺口补成与左腿对称、
    且不会溢出到两腿之间的背景。
    """
    H, W = arr.shape[:2]
    AXIS = (S["hip_l"][0] + S["hip_r"][0]) // 2
    # 镜像左腿：源像素 (x,y) → (2*AXIS - x, y)
    yl, xl = np.where(m_leg_l)
    mx = (2 * AXIS - xl)
    mir = {}
    ok = (mx >= 0) & (mx < W)
    for yy, xx, mxx in zip(yl[ok], xl[ok], mx[ok]):
        mir[(int(mxx), int(yy))] = arr[yy, xx]
    # 可填充区域 = 镜像左腿落在右腿预期多边形(膨胀)内
    polyR = cv2.dilate(poly_mask((H, W), LEG_POLY_R).astype(np.uint8),
                       np.ones((5, 5), np.uint8), 3) > 0
    fillable = np.zeros((H, W), bool)
    for (xx, yy), c in mir.items():
        if polyR[yy, xx]:
            fillable[yy, xx] = True
    cur = m_leg_r.copy()
    orig = m_leg_r.copy()
    for _ in range(60):
        grown = cv2.dilate(cur.astype(np.uint8), np.ones((3, 3), np.uint8), 1) > 0
        grown &= ~orig        # 只接纳尚未填充的区域
        grown &= fillable
        if not grown.any():
            break
        cur |= grown
    newmask = cur & ~orig
    return cur, newmask, mir


def body_keep_mask(alpha):
    """身体保留区：躯干+裙+腿（掩膜） ∪ 肩线以上（头部+头发）。

    掩膜外的两侧长发光翼被剔除 —— 这是用户在桌宠截图上亲手勾画的。
    """
    H, W = alpha.shape
    m = np.zeros((H, W), bool)
    if os.path.exists(BODY_MASK_PATH):
        mk = np.array(Image.open(BODY_MASK_PATH).convert("L")) > 127
        if mk.shape == (H, W):
            m |= alpha & mk
            # 掩膜顶端以上（头部与头发）整体保留
            rows = np.where(mk.any(axis=1))[0]
            if len(rows):
                m |= alpha & (np.arange(H)[:, None] < rows.min())
    return m


def build_parts(src):
    """切分出四肢。返回 (base, {名称: (图片, 掩膜, 旋转中心)})。"""
    arr = np.array(src).astype(int)
    alpha = arr[..., 3] > 200
    soft = arr[..., 3] > 8
    boundary = lineart_mask(arr, alpha)

    S = SKELETON
    # 四肢：用户标注的闭合轮廓多边形（绿色=手臂，黄色=腿）
    m_arm_l = separate_by_poly(soft, boundary, ARM_POLY_L)
    m_arm_r = separate_by_poly(soft, boundary, ARM_POLY_R)
    m_leg_l = separate_by_poly(alpha, boundary, LEG_POLY_L)
    m_leg_r = separate_by_poly(alpha, boundary, LEG_POLY_R)

    # 修复右腿残缺：源图右腿存在一块局部缺失（半透明/透明），
    # 静止姿态下曾用镜像左腿补全，但行走时右腿会绕髋旋转，固定画布的镜像
    # 补全会错位。这里在"源级"用镜像左腿补全右腿缺口——只填右腿预期区域内、
    # 右腿自身缺失而镜像左腿对应位置有像素的透明点。这样右腿在任意姿态
    # （含行走旋转）下都完整，且不破坏左腿或其它部位。
    m_leg_r, leg_r_newmask, leg_r_mirror = _mirror_fill_leg(
        arr, m_leg_l, m_leg_r, S, LEG_POLY_R)

    parts = {}
    for name, mask, pivot, solid in (
            ("arm_l", m_arm_l, S["shoulder_l"], True),
            ("arm_r", m_arm_r, S["shoulder_r"], True),
            ("leg_l", m_leg_l, S["hip_l"], False),
            ("leg_r", m_leg_r, S["hip_r"], False)):
        img = arr.copy()
        img[~mask] = 0
        if solid:
            # 半透明手套恢复实心
            img[mask, 3] = 255
        parts[name] = (Image.fromarray(img.astype(np.uint8), "RGBA").copy(), mask, pivot)
    # 右腿新填充的像素：源右腿此处透明，必须用镜像左腿的颜色覆盖
    if leg_r_newmask.any():
        yy, xx = np.where(leg_r_newmask)
        px = parts["leg_r"][0].load()
        for yi, xi in zip(yy.tolist(), xx.tolist()):
            c = leg_r_mirror.get((xi, yi))
            if c is not None:
                px[xi, yi] = (int(c[0]), int(c[1]), int(c[2]), 255)

    # 底图 = 原图去掉四肢，再套上"身体保留区"（剔除两侧展开的长发）
    keep = body_keep_mask(alpha)
    hole = m_arm_l | m_arm_r | m_leg_l | m_leg_r
    base_arr = inpaint_hole(arr, hole)
    base_arr[~keep] = 0            # 轮廓外（长发光翼）整体去掉
    base_arr[hole & keep] = 0      # 四肢位置留空，由旋转回来的部件补上
    base_img = Image.fromarray(base_arr.astype(np.uint8), "RGBA")
    print(f"  分离：臂L {int(m_arm_l.sum())}px  臂R {int(m_arm_r.sum())}px  "
          f"腿L {int(m_leg_l.sum())}px  腿R {int(m_leg_r.sum())}px  "
          f"保留区 {int(keep.sum())}px")
    return base_img, parts


def pose_parts(base, parts, arm_l=0.0, arm_r=0.0, leg_l=0.0, leg_r=0.0):
    """按角度合成姿态。角度单位：度。

    左臂绕左肩逆时针为正（向左下摆），右臂顺时针为正（向右下摆）。
    腿同理：正值向外/向前摆。
    """
    canvas = base.copy()
    order = ["leg_l", "leg_r", "arm_l", "arm_r"]
    angles = {"arm_l": arm_l, "arm_r": arm_r, "leg_l": leg_l, "leg_r": leg_r}
    for name in order:
        img, mask, pivot = parts[name]
        ang = angles[name]
        if not ang:
            canvas.alpha_composite(img)
            continue
        rot = img.rotate(ang, resample=Image.BICUBIC, center=pivot, expand=False)
        canvas.alpha_composite(rot)
    return canvas


# ---------------------------------------------------------------- 眨眼
def skin_color(arr, cx, cy, w, h):
    """取眼睛下方脸颊区域的肤色中位值。"""
    x0, x1 = max(0, cx - w), min(arr.shape[1], cx + w)
    y0, y1 = min(arr.shape[0], cy + h), min(arr.shape[0], cy + h * 2)
    roi = arr[y0:y1, x0:x1].reshape(-1, 3)
    if not len(roi):
        return (246, 240, 242)
    return tuple(int(v) for v in np.median(roi, axis=0))


def eye_geometry(arr, alpha, cx, cy, w, h):
    """求一只眼的：虹膜红色范围、上下睫毛线位置、睫毛色、肤色。"""
    hsv = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    Hh, Ss, Vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((Hh < 14) | (Hh > 168)) & (Ss > 80) & (Vv > 70)
    px0, px1 = max(0, cx - 14), min(arr.shape[1], cx + 14)
    rows = [yy for yy in range(max(0, cy - 45), min(arr.shape[0], cy + 45))
            if red[yy, px0:px1].sum() >= (px1 - px0) * 0.25]
    if len(rows) >= 4:
        r_top, r_bot = min(rows), max(rows)
    else:
        r_top, r_bot = cy - h // 2, cy + h // 2
    iris_h = max(10, r_bot - r_top)
    lash_top = max(0, r_top - int(round(iris_h * 0.55)))
    lash_bot = min(arr.shape[0] - 1, r_bot + int(round(iris_h * 0.25)))

    band_x0, band_x1 = max(0, cx - w), min(arr.shape[1], cx + w)
    band = arr[lash_top:lash_top + 4, band_x0:band_x1].reshape(-1, 3)
    if not len(band):
        band = arr[max(0, lash_top - 4):lash_top + 4,
                   band_x0:band_x1].reshape(-1, 3)
    if len(band):
        bl = band.mean(axis=1)
        dark = band[bl <= np.percentile(bl, 45)]
        lash_col = tuple(int(v) for v in np.median(dark, axis=0)) if len(dark) \
            else (40, 25, 30)
    else:
        lash_col = (40, 25, 30)

    sk_y0 = min(arr.shape[0] - 1, lash_bot + 5)
    sk_y1 = min(arr.shape[0], sk_y0 + max(6, iris_h // 3))
    skin_roi = arr[sk_y0:sk_y1, band_x0:band_x1].reshape(-1, 3)
    skin_col = tuple(int(v) for v in np.median(skin_roi, axis=0)) \
        if len(skin_roi) else (245, 235, 238)
    return {
        "iris": (cx, cy, w, iris_h),
        "lash_top": lash_top, "lash_bot": lash_bot,
        "lash_col": lash_col, "skin_col": skin_col,
    }


def make_blink(posed, eye_info, close=1.0):
    """闭眼帧：羽化蒙版覆盖整只眼 + 上睫毛处下弯弧线。

    close=1.0 全闭；close<1 半闭（眼皮从上往下落，保留下方虹膜）。
    """
    img = posed.copy()
    for info in eye_info:
        cx, cy, w, h = info["iris"]
        lash_col = info["lash_col"]
        skin_col = info["skin_col"]
        # 眼睛外扩，覆盖睫毛与眼白
        ew = int(w * 1.9)
        eh = int(h * 2.1)
        mcx, mcy = cx, cy
        mx, my = int(ew * 0.58), int(eh * 0.58)
        mask = Image.new("L", (mx * 2, my * 2), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, mx * 2 - 1, my * 2 - 1], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(1, mx // 7)))
        patch = Image.new("RGBA", (mx * 2, my * 2), skin_col + (255,))
        d = ImageDraw.Draw(img)
        if close >= 0.999:
            img.paste(patch, (mcx - mx, mcy - my), mask)
            lw = max(3, eh // 7)
            # 闭眼缝：横向更长、更明显的"笑眼"，但保持细弯缝感
            # （用户反馈原眼缝太短，向左右延伸一点显得眼睛大）
            rw = int(ew * 0.44)
            rh = max(4, int(ew * 0.16))
            arc_y = info["lash_top"] + int(
                (info["lash_bot"] - info["lash_top"]) * 0.35)
            d.arc([mcx - rw, arc_y - rh, mcx + rw, arc_y + rh],
                  start=12, end=168, fill=lash_col, width=lw)
        else:
            half_h = max(1, int(my * 2 * close))
            sub = mask.crop((0, 0, mx * 2, half_h))
            sub_patch = patch.crop((0, 0, mx * 2, half_h))
            img.paste(sub_patch, (mcx - mx, mcy - my), sub)
            eye_y = mcy - my + half_h
            lw = max(2, eh // 9)
            # 半闭眼线也相应加长，与全闭眼缝保持一致
            d.line([(mcx - int(mx * 0.82), eye_y),
                    (mcx + int(mx * 0.82), eye_y)], fill=lash_col, width=lw)
    return img


# ---------------------------------------------------------------- 帧生成
CANVAS_W, CANVAS_H = 376, 364


def rest_mapping(base, parts):
    """静止姿态的裁剪/缩放/粘贴参数（作为身体与腿的不动基准）。

    挥手等抬起手臂的姿态会让整体 bbox 变宽，通用 transform() 会按新宽度重新
    居中，导致身体相对 idle 横向偏移；抬手帧需要复用本函数的左/上/下边界与
    粘贴 x，才能与 idle 帧像素级对齐。
    """
    rest = pose_parts(base, parts, ARM_REST_L, ARM_REST_R)
    L, T, R, B = rest.getbbox()
    bscale = PET_H / (B - T)
    rest_w = round((R - L) * bscale)
    paste_x = (CANVAS_W - rest_w) // 2
    return L, T, B, R, bscale, paste_x


def transform(img, scale_x=1.0, scale_y=1.0, angle=0.0, dy=0.0):
    """裁紧 → 缩放到目标高度 → 缩放/旋转/平移 → 贴画布，底部锚定。"""
    img = img.crop(img.getbbox())
    w, h = img.size
    base_scale = PET_H / h
    img = img.resize((max(1, round(w * base_scale * scale_x)),
                      max(1, round(PET_H * scale_y))), Image.LANCZOS)
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    canvas.alpha_composite(img, ((CANVAS_W - img.size[0]) // 2,
                                 CANVAS_H - img.size[1] - PAD_BOTTOM + int(dy)))
    return canvas


def gen_frames(base, parts, eye_info):
    os.makedirs(FRAMES_DIR, exist_ok=True)
    frames = {}

    def posed(al=0.0, ar=0.0, ll=0.0, lr=0.0, blink=0.0):
        img = pose_parts(base, parts, al, ar, ll, lr)
        if blink:
            img = make_blink(img, eye_info, close=blink)
        return img

    def rest(dl=0.0, dr=0.0):
        """静止姿态 + 增量（dl/dr 为相对垂臂的角度偏移）。"""
        return ARM_REST_L + dl, ARM_REST_R + dr

    # 待机：呼吸 + 手臂微摆（4 帧循环）
    for i, (sy, dy, dl, dr) in enumerate([
            (1.000, 0, 3, -3),
            (1.018, -2, 1, -1),
            (1.030, -4, 0, 0),
            (1.018, -2, 1, -1)]):
        al, ar = rest(dl, dr)
        frames[f"idle_{i}"] = transform(posed(al, ar), scale_y=sy, dy=dy)

    # 眨眼：半闭 → 全闭 → 半闭
    for i, close in enumerate([0.55, 1.0, 0.55], start=1):
        al, ar = rest()
        frames[f"blink_{i}"] = transform(posed(al, ar, blink=close))

    # 行走：身体摇摆 + 双腿前后摆 + 双臂反相摆动（4 帧）
    for i, (ang, dy, ll, lr, dl, dr) in enumerate([
            (2.5, 0, 13, -11, -7, 7),
            (0, -3, 0, 0, 0, 0),
            (-2.5, 0, -11, 13, 7, -7),
            (0, -3, 0, 0, 0, 0)]):
        al, ar = rest(dl, dr)
        frames[f"walk_{i}"] = transform(posed(al, ar, ll, lr), angle=ang, dy=dy)

    # 跳跃：起手压缩 + 空中伸展
    al, ar = rest(5, -5)
    frames["squash"] = transform(posed(al, ar, 7, -7),
                                 scale_x=1.08, scale_y=0.86, dy=12)
    al, ar = rest(-7, 7)
    frames["stretch"] = transform(posed(al, ar, -5, 5),
                                  scale_x=0.92, scale_y=1.08, dy=-8)

    # 睡觉：手臂放松下垂 + 闭眼（休息时眼睛闭上，醒来才睁开）
    al, ar = rest(2, -2)
    frames["sleep"] = transform(posed(al, ar, blink=1.0), scale_y=0.97, dy=6)

    # 对话：轻微前倾，单手微抬
    al, ar = rest(-6, 10)
    frames["talk"] = transform(posed(al, ar), scale_y=1.02, dy=-2)

    # 挥手打招呼：右臂抬起（约 50°→72°）并前后摆动（4 帧循环）
    # 注意：挥手时抬起的右臂会让整体 bbox 变宽，若仍用通用 transform()，
    # 横向重新居中会使身体/腿整体偏移（相对 idle 错位 → 出现"多一条腿"）。
    # 这里沿用静止姿态的裁剪左/上/下边界，只把右边扩展以容纳手臂，
    # 并用与 idle 相同的粘贴 x，保证身体与腿部像素完全对齐。
    L, T, B, RST_R, bscale, pxs = rest_mapping(base, parts)
    _cr = max(pose_parts(base, parts, ARM_REST_L, a).getbbox()[2]
              for a in (50, 72)) + 6
    for i, ar in enumerate([50, 72, 50, 72]):
        img = pose_parts(base, parts, ARM_REST_L, ar).crop((L, T, _cr, B))
        img = img.resize((round((_cr - L) * bscale), PET_H), Image.LANCZOS)
        cv = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        cv.alpha_composite(img, (pxs, CANVAS_H - PET_H - PAD_BOTTOM))
        frames[f"wave_{i}"] = cv

    for name, img in frames.items():
        img.save(os.path.join(FRAMES_DIR, f"{name}.png"))
    return sorted(frames)


def checkerboard(img, cell=10):
    """棋盘格背景预览，验证透明区域。"""
    w, h = img.size
    bg = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(bg)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (x // cell + y // cell) % 2 == 0:
                d.rectangle([x, y, x + cell, y + cell], fill=(200, 200, 200, 255))
    out = bg.copy()
    out.alpha_composite(img)
    return out.convert("RGB")


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(BASE_OUT), exist_ok=True)

    src = load_source()
    base, parts = build_parts(src)
    base.save(BASE_OUT)
    print(f"底图：{base.size} → {BASE_OUT}")
    checkerboard(base).save(os.path.join(DEBUG_DIR, "pet_checkerboard.png"))

    # 姿态预览：手臂垂下
    rest = pose_parts(base, parts, arm_l=ARM_REST_L, arm_r=ARM_REST_R)
    rest.save(os.path.join(DEBUG_DIR, "pose_rest.png"))
    checkerboard(rest).save(os.path.join(DEBUG_DIR, "pose_rest_cb.png"))

    # 眼睛：直接用用户标注的圈中心
    arr = np.array(src).astype(int)
    alpha = arr[..., 3] > 200
    S = SKELETON
    eye_info = []
    for key in ("eye_l", "eye_r"):
        cx, cy = S[key]
        info = eye_geometry(arr, alpha, cx, cy, 30, 16)
        eye_info.append(info)
        print(f"  {key}: 睫毛线 y={info['lash_top']}..{info['lash_bot']} "
              f"弧色={info['lash_col']} 肤色={info['skin_col']}")

    # 画布系眼睛坐标（供运行时参考）
    posed = pose_parts(base, parts, arm_l=ARM_REST_L, arm_r=ARM_REST_R)
    bbox = posed.getbbox()
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    scale = PET_H / h
    canvas_size = (round((bbox[2] - bbox[0]) * scale + PAD_X * 2),
                   round(PET_H + PAD_TOP + PAD_BOTTOM))
    globals()["CANVAS_W"], globals()["CANVAS_H"] = canvas_size
    eyes = [(round((info["iris"][0] - bbox[0]) * scale + PAD_X),
             round((info["iris"][1] - bbox[1]) * scale + PAD_TOP),
             max(3, round(info["iris"][2] * scale)),
             max(3, round(info["iris"][3] * scale)))
            for info in eye_info]
    json.dump(eyes, open(EYES_OUT, "w"))
    print(f"眼睛坐标（画布系）：{eyes}  画布 {canvas_size}")

    names = gen_frames(base, parts, eye_info)
    print(f"动画帧 {len(names)} 张 → {FRAMES_DIR}")
    print("OK")


if __name__ == "__main__":
    main()
