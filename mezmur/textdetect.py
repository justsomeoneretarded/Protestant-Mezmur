"""'Does this picture contain written text?' — PP-OCRv3 text detector (DB) from the OpenCV model zoo.

Used to keep title cards, watermarks and cover-art lettering out of the video backgrounds,
where they would fight with the lyrics. Text *detection* is script-agnostic enough for
Ge'ez and Latin lettering alike. If the model cannot be loaded every image counts as clean.
"""
import cv2
import numpy as np

_model = None
_failed = False


def _load():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    try:
        from .thumbnail import model_file
        path = model_file("ppocr_det")
        if path is None:
            raise FileNotFoundError("text detector download failed")
        m = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(path)))
        m.setBinaryThreshold(0.3)
        m.setPolygonThreshold(0.5)
        m.setUnclipRatio(2.0)
        m.setMaxCandidates(200)
        m.setInputParams(1.0 / 255.0, (736, 736), (122.67891434, 116.66876762, 104.00698793))
        _model = m
    except Exception as error:  # noqa: BLE001 - optional quality filter
        print(f"  text detector unavailable: {error}", flush=True)
        _failed = True
    return _model


def text_boxes(img):
    """Axis-aligned boxes (x, y, w, h) of detected text lines."""
    if img is None:
        return []
    model = _load()
    if model is None:
        return []
    h, w = img.shape[:2]
    small = cv2.resize(img, (736, 736), interpolation=cv2.INTER_AREA)
    try:
        polys, _ = model.detect(small)
    except cv2.error:
        return []
    out = []
    for p in polys:
        p = np.array(p, dtype=np.float32)
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        out.append((int(x0 * w / 736), int(y0 * h / 736), int((x1 - x0) * w / 736), int((y1 - y0) * h / 736)))
    return out


def text_score(img):
    """Share of the picture covered by readable-size text (0 = clean)."""
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    area = 0.0
    for x, y, bw, bh in text_boxes(img):
        if bh >= 0.025 * h and bw >= 0.06 * w:     # ignore tiny logos / distant signage
            area += bw * bh
    return float(area / (w * h))


def has_text(img, threshold=0.012):
    return text_score(img) >= threshold
