"""Artist photo selection (face detection + identity consensus), person cutout and thumbnails."""
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .textdraw import Mixed, fit, gradient_text, shadow_text

MODELS = Path(os.environ.get("MEZMUR_MODEL_CACHE", ".model-cache")) / "vision"
URLS = {
    "yunet": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "u2net_human_seg": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx",
}
W, H = 1280, 720


def model_file(name):
    path = MODELS / Path(URLS[name]).name
    if not path.is_file() or path.stat().st_size < 10000:
        from .web import get
        MODELS.mkdir(parents=True, exist_ok=True)
        r = get(URLS[name], timeout=120, max_bytes=400_000_000)
        if r is None or r.status_code != 200 or len(r.content) < 10000:
            return None
        path.write_bytes(r.content)
    return path


# ---------------------------------------------------------------- faces
_detector = None
_recognizer = None


def detect_faces(img):
    """Return faces as dicts: box (x, y, w, h), score, landmarks (5x2) — YuNet, Haar fallback."""
    global _detector
    h, w = img.shape[:2]
    scale = min(1.0, 1280 / max(h, w))
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img
    faces = []
    path = model_file("yunet") if _detector is None else True
    if path and hasattr(cv2, "FaceDetectorYN"):
        if _detector is None:
            _detector = cv2.FaceDetectorYN.create(str(path), "", (small.shape[1], small.shape[0]), 0.75, 0.3, 50)
        _detector.setInputSize((small.shape[1], small.shape[0]))
        _, found = _detector.detect(small)
        for row in (found if found is not None else []):
            x, y, fw, fh = (row[:4] / scale).tolist()
            faces.append({"box": (x, y, fw, fh), "score": float(row[14]), "landmarks": (row[4:14].reshape(5, 2) / scale),
                          "raw": row, "scale": scale})
    elif hasattr(cv2, "CascadeClassifier"):
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for (x, y, fw, fh) in cascade.detectMultiScale(gray, 1.1, 6, minSize=(40, 40)):
            faces.append({"box": (x / scale, y / scale, fw / scale, fh / scale), "score": 0.8, "landmarks": None,
                          "raw": None, "scale": scale})
    return sorted(faces, key=lambda f: -f["box"][2] * f["box"][3])


def face_embedding(img, face):
    global _recognizer
    if face.get("raw") is None:
        return None
    if _recognizer is None:
        path = model_file("sface")
        if not path or not hasattr(cv2, "FaceRecognizerSF"):
            return None
        _recognizer = cv2.FaceRecognizerSF.create(str(path), "")
    scale = face["scale"]
    small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))) if scale < 1 else img
    aligned = _recognizer.alignCrop(small, face["raw"])
    feat = _recognizer.feature(aligned).flatten()
    return feat / (np.linalg.norm(feat) + 1e-9)


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def rank_photos(candidates, log=print):
    """Pick images that show the artist.

    candidates: [{"path", "source", "weight"}]. The identity that appears across the most
    (weighted) sources is taken as the artist; images are then ranked by face size, quality and
    framing. Returns ranked list of dicts with the chosen face.
    """
    items = []
    for c in candidates:
        img = cv2.imread(str(c["path"]))
        if img is None or min(img.shape[:2]) < 200:
            continue
        faces = [f for f in detect_faces(img) if f["score"] >= 0.75 and f["box"][2] >= 0.06 * img.shape[1]]
        if not faces:
            continue
        face = faces[0]
        emb = face_embedding(img, face)
        items.append({**c, "img_shape": img.shape[:2], "face": face, "emb": emb, "n_faces": len(faces),
                      "sharp": sharpness(img)})
    if not items:
        return []
    # Identity consensus.
    with_emb = [i for i in items if i["emb"] is not None]
    if len(with_emb) >= 2:
        for i in with_emb:
            i["support"] = sum(j["weight"] for j in with_emb if j is not i and float(i["emb"] @ j["emb"]) > 0.40)
        best_support = max(i["support"] for i in with_emb)
        anchor = max(with_emb, key=lambda i: (i["support"], i["weight"]))
        for i in items:
            i["same_person"] = i["emb"] is not None and (i is anchor or float(i["emb"] @ anchor["emb"]) > 0.40)
        if best_support == 0:
            for i in items:
                i["same_person"] = i["weight"] >= 2
    else:
        for i in items:
            i["same_person"] = i["weight"] >= 2 or len(items) == 1
    for i in items:
        h, w = i["img_shape"]
        x, y, fw, fh = i["face"]["box"]
        rel = fh / h
        framing = 1.0 if 0.12 <= rel <= 0.42 else 0.6  # room for shoulders / cutout
        edge = 0.5 if x < 2 or y < 2 or x + fw > w - 2 else 1.0
        solo = 1.0 if i["n_faces"] == 1 else 0.7
        res = min(1.0, min(h, w) / 700)
        i["score"] = (2.0 if i["same_person"] else 0.3) * i["weight"] * framing * edge * solo * (0.6 + 0.4 * res) \
            * (0.7 + 0.3 * min(1.0, i["sharp"] / 300))
    ranked = sorted(items, key=lambda i: -i["score"])
    for i in ranked[:6]:
        log(f"  photo {i['source']}: score {i['score']:.2f} same={i['same_person']} faces={i['n_faces']}", flush=True)
    return ranked


# ---------------------------------------------------------------- cutout
_seg = None


def person_mask(img):
    """Soft person mask (0..1) using U^2-Net human segmentation via onnxruntime."""
    global _seg
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if _seg is None:
        path = model_file("u2net_human_seg")
        if not path:
            return None
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() or 2)
        _seg = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    h, w = img.shape[:2]
    x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    pred = _seg.run(None, {_seg.get_inputs()[0].name: x})[0][0, 0]
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
    mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.clip((mask - 0.25) / 0.5, 0, 1)
    # Keep the largest connected component.
    binary = (mask > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = cv2.dilate((labels == keep).astype(np.uint8), np.ones((25, 25), np.uint8))
        mask = mask * region
    return cv2.GaussianBlur(mask, (0, 0), 1.2)


def cutout(img, face=None):
    """RGBA PIL image of the person cropped to the mask, or None if the cutout looks wrong."""
    mask = person_mask(img)
    if mask is None:
        return None
    area = float((mask > 0.5).mean())
    if not 0.04 <= area <= 0.85:
        return None
    if face is not None:
        x, y, fw, fh = [int(v) for v in face["box"]]
        cx, cy = min(img.shape[1] - 1, x + fw // 2), min(img.shape[0] - 1, y + fh // 2)
        if mask[cy, cx] < 0.5:
            return None
    ys, xs = np.where(mask > 0.3)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (mask * 255).astype(np.uint8)])[y0:y1, x0:x1]
    out = Image.fromarray(rgba, "RGBA")
    out.info["touches_bottom"] = y1 >= img.shape[0] - 3
    if face is not None:
        out.info["face"] = (face["box"][0] - x0, face["box"][1] - y0, face["box"][2], face["box"][3])
    return out


# ---------------------------------------------------------------- composition
def _bg(img_bgr, warm=True):
    if img_bgr is None:
        base = Image.new("RGB", (W, H), (30, 18, 10))
    else:
        h, w = img_bgr.shape[:2]
        s = max(W / w, H / h)
        im = cv2.resize(img_bgr, (int(w * s) + 1, int(h * s) + 1), interpolation=cv2.INTER_AREA)
        y, x = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2
        base = Image.fromarray(cv2.cvtColor(im[y:y + H, x:x + W], cv2.COLOR_BGR2RGB))
    base = ImageEnhance.Color(base).enhance(1.15)
    base = ImageEnhance.Contrast(base).enhance(1.08)
    tint = Image.new("RGB", (W, H), (60, 30, 5) if warm else (5, 30, 50))
    return Image.blend(base, tint, 0.18).convert("RGBA")


def _side_shade(canvas, side="left", strength=215):
    shade = Image.new("L", (W, 1))
    for x in range(W):
        k = (1 - x / (W * 0.72)) if side == "left" else (x - W * 0.28) / (W * 0.72)
        shade.putpixel((x, 0), int(max(0, min(1, k)) ** 1.25 * strength))
    shade = shade.resize((W, H))
    dark = Image.new("RGBA", (W, H), (6, 4, 2, 255))
    dark.putalpha(shade)
    canvas.alpha_composite(dark)


def _glow_behind(canvas, center, radius, color):
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(glow)
    cx, cy = center
    for k in range(8, 0, -1):
        r = radius * k / 8
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color + (int(18 + 10 * (8 - k)),))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius * 0.25)))


def _rays(canvas, origin, color, n=9, seed=0):
    rng = np.random.default_rng(seed)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ox, oy = origin
    for k in range(n):
        ang = -math.pi / 2 + rng.uniform(-1.1, 1.1)
        spread = rng.uniform(0.03, 0.08)
        length = 1500
        p1 = (ox + length * math.cos(ang - spread), oy + length * math.sin(ang - spread))
        p2 = (ox + length * math.cos(ang + spread), oy + length * math.sin(ang + spread))
        d.polygon([origin, p1, p2], fill=color + (int(rng.uniform(16, 34)),))
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(14)))


def _person(canvas, person, side, accent):
    target_h = int(H * 0.98)
    face = person.info.get("face")
    s = target_h / person.height
    if face:  # keep the face a strong size even for tight crops
        s = min(s * 1.0, (H * 0.36) / max(1, face[3])) if face[3] * s > H * 0.36 else s
    pw, ph = int(person.width * s), int(person.height * s)
    p = person.resize((pw, ph), Image.LANCZOS)
    cx = int(W * (0.74 if side == "right" else 0.26))
    if face:
        fx = int((face[0] + face[2] / 2) * s)
        x = cx - fx
    else:
        x = cx - pw // 2
    x = max(-pw // 3, min(W - 2 * pw // 3, x))
    y = H - ph if person.info.get("touches_bottom", True) or ph >= H * 0.9 else int(H * 0.5 - ph / 2 + H * 0.05)
    alpha = p.getchannel("A")
    glow = Image.new("RGBA", p.size, accent + (0,))
    glow.putalpha(alpha.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(18)).point(lambda v: int(v * 0.75)))
    canvas.alpha_composite(glow, (x, y))
    shadow = Image.new("RGBA", p.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.55)))
    canvas.alpha_composite(shadow, (x + 10, y + 8))
    rim = ImageEnhance.Brightness(p).enhance(1.05)
    canvas.alpha_composite(rim, (x, y))
    box = alpha.point(lambda v: 255 if v > 90 else 0).getbbox() or (0, 0, pw, ph)
    return x + box[0], x + box[2]


def _wrap(text, mixed, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if mixed.length(test) <= width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _text_block(canvas, meta, x, width, align, accent, accent_dark):
    title = meta.get("title_am") or meta.get("title_latin", "")
    size = 170
    while size > 70:
        f = Mixed("black", size)
        lines = _wrap(title, f, width)
        if len(lines) <= 2 and all(f.length(l) <= width for l in lines):
            break
        size -= 6
    anchor = "la" if align == "left" else ("ma" if align == "center" else "ra")
    ax = x if align == "left" else (x + width // 2 if align == "center" else x + width)
    roman = (meta.get("title_latin") or "").upper() if meta.get("title_am") else ""
    artist_am = meta.get("artist_am") or ""
    artist_lat = meta.get("artist_latin") or ""
    rf = fit("black", roman, 64, width) if roman else None
    block_h = 64 + len(lines) * int(size * 1.08) + (int(rf.size * 1.3) if rf else 0) + 120
    y = max(36, (H - block_h) // 2 - 10)
    # Pill.
    tag = "ግጥም • LYRICS"
    pf = Mixed("black", 30)
    tw = pf.length(tag)
    px = ax if align == "left" else (ax - tw / 2 if align == "center" else ax - tw)
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle((px - 22, y, px + tw + 22, y + 52), radius=26, fill=accent + (255,))
    pf.draw(d, (px + tw / 2, y + 26), tag, (25, 14, 4, 255), anchor="mm")
    y += 74
    for line in lines:
        gradient_text(canvas, (ax, y), line, Mixed("black", size), (255, 244, 200), accent_dark,
                      anchor=anchor, stroke=max(4, size // 22), offset=6, blur=12)
        y += int(size * 1.08)
    if rf:
        shadow_text(canvas, (ax, y + 4), roman, rf, (255, 255, 255, 255), anchor=anchor, blur=8, offset=4,
                    stroke=3, stroke_fill=(10, 6, 2, 255))
        y += int(rf.size * 1.35)
    if artist_am or artist_lat:
        af = fit("bold", artist_am or artist_lat, 50, width)
        shadow_text(canvas, (ax, y + 12), artist_am or artist_lat, af, (255, 255, 255, 250), anchor=anchor, blur=6,
                    stroke=2, stroke_fill=(10, 6, 2, 255))
        y += int(af.size * 1.25) + 12
        if artist_am and artist_lat:
            lf = fit("bold", artist_lat, 36, width)
            shadow_text(canvas, (ax, y + 6), artist_lat, lf, accent + (255,), anchor=anchor, blur=5,
                        stroke=2, stroke_fill=(10, 6, 2, 255))


PALETTES = {
    "gold": ((255, 200, 90), (225, 120, 30), True),
    "sky": ((120, 220, 255), (40, 120, 220), False),
    "rose": ((255, 170, 150), (205, 70, 90), True),
}


def compose(meta, background=None, person=None, photo=None, variant="a", seed=0):
    """Build one 1280x720 thumbnail.

    variant a: cutout right, text left (gold) · b: cutout left, text right (sky) ·
    c: full-bleed photo with bottom title (rose) · fallback: centered text.
    """
    accent, accent_dark, warm = PALETTES[{"a": "gold", "b": "sky", "c": "rose"}.get(variant, "gold")]
    if variant == "c" and (photo is not None or background is not None):
        source = photo if photo is not None else background
        blurred = cv2.GaussianBlur(source, (0, 0), 18)
        canvas = _bg(blurred, warm)
        canvas.alpha_composite(Image.new("RGBA", (W, H), (8, 5, 3, 90)))
        _rays(canvas, (int(W * 0.5), -60), accent, n=12, seed=seed + 1)
        if person is not None:
            _glow_behind(canvas, (int(W * 0.5), int(H * 0.40)), 280, accent)
            small = person.copy()
            scale = (H * 0.78) / small.height
            small = small.resize((int(small.width * scale), int(small.height * scale)), Image.LANCZOS)
            small.info.update(person.info)
            if "face" in person.info:
                fx, fy, fw, fh = person.info["face"]
                small.info["face"] = (fx * scale, fy * scale, fw * scale, fh * scale)
            px = (W - small.width) // 2
            alpha = small.getchannel("A")
            glow = Image.new("RGBA", small.size, accent + (0,))
            glow.putalpha(alpha.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(18)).point(lambda v: int(v * 0.7)))
            canvas.alpha_composite(glow, (px, 0))
            canvas.alpha_composite(small, (px, 0))
        grad = Image.new("L", (1, H))
        for yy in range(H):
            grad.putpixel((0, yy), int(255 * max(0, (yy / H - 0.40) / 0.60) ** 1.3))
        dark = Image.new("RGBA", (W, H), (5, 3, 2, 255))
        dark.putalpha(grad.resize((W, H)))
        canvas.alpha_composite(dark)
        title = meta.get("title_am") or meta.get("title_latin", "")
        f = fit("black", title, 150, int(W * 0.9))
        gradient_text(canvas, (W // 2, int(H * 0.72)), title, f, (255, 244, 200), accent_dark, anchor="mm",
                      stroke=max(4, f.size // 22), offset=6, blur=12)
        sub = "  •  ".join(p for p in [(meta.get("title_latin") or "").upper() if meta.get("title_am") else "",
                                       meta.get("artist_am") or meta.get("artist_latin")] if p)
        shadow_text(canvas, (W // 2, int(H * 0.88)), sub, fit("bold", sub, 46, int(W * 0.9)), (255, 255, 255, 255),
                    anchor="mm", blur=6, stroke=2, stroke_fill=(10, 6, 2, 255))
        tag = "ግጥም • LYRICS"
        pf = Mixed("black", 30)
        d = ImageDraw.Draw(canvas)
        tw = pf.length(tag)
        d.rounded_rectangle((40, 36, 40 + tw + 44, 88), radius=26, fill=accent + (255,))
        pf.draw(d, (62 + tw / 2, 62), tag, (25, 14, 4, 255), anchor="mm")
        d.rectangle((10, 10, W - 11, H - 11), outline=accent + (160,), width=3)
        return canvas.convert("RGB")
    canvas = _bg(background, warm)
    side = "right" if variant == "a" else "left"
    if person is not None:
        _side_shade(canvas, "left" if side == "right" else "right")
        _rays(canvas, (int(W * (0.74 if side == "right" else 0.26)), -40), accent, seed=seed)
        _glow_behind(canvas, (int(W * (0.74 if side == "right" else 0.26)), int(H * 0.45)), 300, accent)
        left, right = _person(canvas, person, side, accent)
        if side == "right":
            width = int(min(W * 0.58, max(W * 0.42, left - 40 - 54)))
            _text_block(canvas, meta, 54, width, "left", accent, accent_dark)
        else:
            x0 = int(max(W * 0.40, min(W * 0.58, right + 40)))
            _text_block(canvas, meta, x0, W - 54 - x0, "right", accent, accent_dark)
    else:
        shade = Image.new("RGBA", (W, H), (5, 3, 2, 120))
        canvas.alpha_composite(shade)
        _rays(canvas, (W // 2, -60), accent, n=12, seed=seed)
        _text_block(canvas, meta, 80, W - 160, "center", accent, accent_dark)
    # Thin frame for a finished look.
    d = ImageDraw.Draw(canvas)
    d.rectangle((10, 10, W - 11, H - 11), outline=accent + (160,), width=3)
    return canvas.convert("RGB")


def save(img, path):
    img.save(path, "JPEG", quality=92, optimize=True, progressive=True)
    if Path(path).stat().st_size > 1_900_000:
        img.save(path, "JPEG", quality=82, optimize=True)
