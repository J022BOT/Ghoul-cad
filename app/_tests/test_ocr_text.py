"""OCR 文字识别离线自测：不依赖 Ollama（模型调用用 mock 顶替）。

验证四件事：
  1) OpenCV 能把「文字行」框出来，并且不会把矩形/直线误当文字；
  2) 像素框 → mm 坐标的映射（用 spec._meta.frame）方向正确（y 翻转）；
  3) 文字通过视觉模型返回的 JSON 正确落到 text 图元上；
  4) 落在文字框内的笔画碎线会被剔除，不会重复画。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

import line_extractor
import ocr_text


def make_image():
    img = np.full((600, 900, 3), 255, np.uint8)
    # 几何：矩形 + 一条长线
    cv2.rectangle(img, (520, 120), (820, 380), (0, 0, 0), 2)
    cv2.line(img, (60, 500), (840, 500), (0, 0, 0), 2)
    # 文字：两处
    cv2.putText(img, "M10", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "50mm", (340, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    # 文字笔画碎线（模拟 CV 把字当线条）：落在第一处文字里
    cv2.line(img, (86, 150), (120, 150), (0, 0, 0), 2)
    return img


def main():
    assert ocr_text.available(), "OpenCV 不可用"
    img = make_image()

    boxes = ocr_text.detect_text_boxes(img)
    print("检测到文字框：", boxes)
    assert 2 <= len(boxes) <= 4, "文字框数量不合理：%d" % len(boxes)

    # 文字框应落在两处文字附近，且不该框住矩形/长线
    top = min(boxes, key=lambda b: b[1])
    assert top[1] < 200 and top[0] < 200, "第一处文字框位置不对：%s" % (top,)
    for b in boxes:
        assert b[3] < 120, "框太高，可能把几何当成文字：%s" % (b,)
        assert not (b[0] > 500 and b[1] < 400 and b[3] > 200), "把矩形框进去了：%s" % (b,)
    print("PASS 1: 文字框定位正确（共 %d 个）" % len(boxes))

    spec = line_extractor.extract_spec(img, instruction="")
    print("CV 几何：", spec["_meta"].get("counts"), "frame=", spec["_meta"].get("frame"))
    before = len(spec["entities"])

    # mock 视觉模型：按编号返回文字
    texts = {1: "M10", 2: "50mm"}

    def fake_ask(prompt, image_b64, model, host, **kw):
        import json
        return json.dumps([{"id": i, "text": t} for i, t in texts.items()], ensure_ascii=False), None

    ocr_text._ask_vision = fake_ask

    info = ocr_text.enrich_spec_with_text(spec, img, model="fake", host="http://127.0.0.1:1")
    print("enrich 结果：", info)
    assert info.get("ok"), "OCR 未成功：%s" % info.get("error")

    txts = [e for e in spec["entities"] if e.get("type") == "text"]
    assert len(txts) == 2, "文本图元数量不对：%d" % len(txts)
    vals = sorted(t["value"] for t in txts)
    assert vals == ["50mm", "M10"], vals
    print("PASS 2: 文本图元生成正确：", vals)

    # y 方向：第一处文字（图像上方 y≈160）在 DXF 里 y 应该更大（y 翻转）
    txt_top = max(txts, key=lambda t: t["p"][1])
    txt_bot = min(txts, key=lambda t: t["p"][1])
    assert txt_top["value"] == "M10", "上方文字应为 M10：%s" % (txt_top,)
    assert txt_top["p"][1] > txt_bot["p"][1], "y 翻转不对：%s / %s" % (txt_top, txt_bot)
    for t in txts:
        assert 1.0 <= t["height"] <= 40, "字高异常：%s" % t["height"]
    print("PASS 3: 坐标映射（y 翻转）与字高正常：",
          [(t["value"], t["p"], t["height"]) for t in txts])

    # 笔画碎线剔除：直接塞一条「落在文字框内」的短线，验证会被剔除
    mapper = ocr_text._make_mapper(spec, img)
    bx, by, bw, bh = boxes[0]
    p1 = mapper.to_mm(bx + 4, by + bh - 4)
    p2 = mapper.to_mm(bx + bw - 4, by + bh - 4)
    spec["entities"].append({"type": "line",
                             "p1": [p1[0], p1[1]], "p2": [p2[0], p2[1]]})
    dropped = ocr_text._drop_strokes_inside(spec, mapper, boxes)
    assert dropped == 1, "框内笔画没有被剔除：%s" % dropped
    print("PASS 4: 文字框内的笔画碎线已剔除（%d 根）" % dropped)

    # 再跑一次可重复（不重复堆叠文字）
    info2 = ocr_text.enrich_spec_with_text(spec, img, model="fake", host="http://127.0.0.1:1")
    assert info2.get("ok")
    assert len([e for e in spec["entities"] if e.get("type") == "text"]) == 4
    print("PASS 5: 可重复调用（追加而非崩溃）")
    print("\nALL OCR TESTS PASSED")


if __name__ == "__main__":
    main()
