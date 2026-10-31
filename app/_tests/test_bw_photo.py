"""bw_photo 纯黑白预处理回归测试（line_extractor + cad_agent CV 路线）。

功能：生成 DXF 前先把图片压成高对比纯黑白（纸=255 白、墨=0 黑）再识别。
关键契约：
  1) 输出像素值只有 0/255（真二值，不是灰阶拉伸）；
  2) 颜色无关：淡红线 / 浅灰线与黑轮廓一样进墨迹（不走灰度 Otsu）；
  3) 暗底亮线照片被归一成白底黑线；
  4) 阈值与主管线同款（噪声底 + 0.07×最强墨迹），抗 JPEG 振铃与纸纹阴影；
  5) cad_agent._cv_spec_from_image(bw_first=True) 落盘黑白图并让识别吃这张图。

跑法：python _tests/test_bw_photo.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import line_extractor  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  OK   %-46s %s" % (name, detail))
    else:
        FAIL += 1
        print("  FAIL %-46s %s" % (name, detail))


def main():
    cv = line_extractor.cv2
    if not line_extractor.HAS_CV:
        print("SKIP 未装 OpenCV")
        return 0
    tmp = tempfile.mkdtemp(prefix="bw_")

    # ------------------------------------------------ 1) 基本二值性 + 颜色无关
    print("\n[1] 彩色淡线图 → 纯黑白（黑轮廓 + 淡红 + 浅灰 全保留）")
    W, H = 640, 480
    img = np.full((H, W, 3), 245, np.uint8)
    # 轻微纸纹阴影（低频渐变）
    gx = np.linspace(0, 12, W, dtype=np.float32)
    img += gx[None, :, None].astype(np.uint8)
    # 黑色矩形轮廓
    cv.rectangle(img, (60, 60), (300, 260), (20, 20, 20), 3)
    # 淡红尺寸线（灰度上几乎与纸同亮度）
    cv.line(img, (60, 320), (300, 320), (200, 200, 240), 2)
    # 浅灰引线
    cv.line(img, (350, 80), (580, 240), (200, 200, 200), 2)
    # JPEG 压缩制造振铃
    okj, enc = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, 80])
    img_jpg = cv.imdecode(enc, cv.IMREAD_COLOR) if okj else img
    bw = line_extractor.bw_photo(img_jpg)
    check("bw_photo 返回非空", bw is not None)
    vals = np.unique(bw)
    check("输出像素只有 0/255（真二值）", set(vals.tolist()) <= {0, 255}, str(vals))
    ink = bw[..., 0] == 0
    check("黑轮廓进墨迹", ink[60:260, 60:300].sum() > 300)
    check("淡红线进墨迹（颜色无关）", ink[314:326, 60:300].sum() > 200,
          str(int(ink[314:326, 60:300].sum())))
    check("浅灰线进墨迹", ink[80:240, 350:580].sum() > 150)
    check("背景是纯白", float(bw[400:, 40:60].mean()) == 255.0)

    # ------------------------------------------------ 2) 暗底亮线 → 归一成白底黑线
    print("\n[2] 暗底亮线照片 → 白底黑线")
    dark = np.full((H, W, 3), 30, np.uint8)
    cv.line(dark, (100, 100), (500, 380), (230, 230, 230), 3)
    bw2 = line_extractor.bw_photo(dark)
    check("暗底图输出也只有 0/255", set(np.unique(bw2).tolist()) <= {0, 255})
    check("暗底被归一成白底", float(bw2[0:50, 0:50].mean()) == 255.0)
    check("亮线变黑线", (bw2[..., 0] == 0).sum() > 300,
          str(int((bw2[..., 0] == 0).sum())))

    # ------------------------------------------------ 3) 喂给主管线仍能识别
    print("\n[3] 黑白图直接进 extract_spec")
    p_bw = os.path.join(tmp, "bw.png")
    cv.imencode(".png", bw)[1].tofile(p_bw)
    spec = line_extractor.extract_spec(p_bw, instruction="")
    meta = spec.get("_meta") or {}
    check("识别 ok", bool(meta.get("ok")), str(meta.get("reason")))
    check("检出图元 > 0", len(spec.get("entities") or []) > 0,
          str(meta.get("counts")))

    # ------------------------------------------------ 4) cad_agent CV 路线接线
    print("\n[4] cad_agent._cv_spec_from_image(bw_first=True)")
    import cad_agent
    if not cad_agent or not line_extractor.available():
        print("  SKIP")
    else:
        p_in = os.path.join(tmp, "in.png")
        cv.imencode(".png", img_jpg)[1].tofile(p_in)
        spec2, dbg = cad_agent._cv_spec_from_image(
            p_in, "", tmp, passes=1, bw_first=True)
        check("CV 路线识别成功", spec2 is not None)
        bw_path = (spec2.get("_meta") or {}).get("bw_path") if spec2 else None
        check("spec 带黑白图路径", bool(bw_path) and os.path.isfile(bw_path),
              str(bw_path))
        if bw_path and os.path.isfile(bw_path):
            saved = line_extractor.load_image(bw_path)
            check("落盘黑白图也是纯 0/255",
                  set(np.unique(saved).tolist()) <= {0, 255})
        # 关掉开关 → 不落黑白图
        spec3, _ = cad_agent._cv_spec_from_image(
            p_in, "", tmp, passes=1, bw_first=False)
        if spec3:
            check("bw_first=False 时不做预处理",
                  not (spec3.get("_meta") or {}).get("bw_path"))
        # 整管线 generate_dxf_detailed 也接了参数（只查签名，不跑模型）
        import inspect
        sig = inspect.signature(cad_agent.generate_dxf_detailed)
        check("generate_dxf_detailed 有 bw_first 参数", "bw_first" in sig.parameters)

    print("\n%d pass, %d fail" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
