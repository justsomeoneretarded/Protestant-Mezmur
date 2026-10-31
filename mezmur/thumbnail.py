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
    "ppocr_det": "https://github.com/opencv/opencv_zoo/raw/main/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx",
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
    # Who is the singer? Decided only by trusted photos (this song's cover, post, official uploads, your
    # artists/ folder). Name searches can be flooded with a more famous namesake, so they never decide;
    # they are used only when their face matches the trusted face.
    SAME = 0.45
    trusted = [i for i in items if i.get("trusted") and i["emb"] is not None]
    anchor = None
    if trusted:
        for i in trusted:
            i["support"] = i["weight"] + sum(j["weight"] for j in trusted
                                             if j is not i and float(i["emb"] @ j["emb"]) > SAME)
        anchor = max(trusted, key=lambda i: (i["support"], i["weight"]))
    else:
        strict = [i for i in items if i.get("strict_name") and i["emb"] is not None]
        named = [i for i in items if i.get("named") and i["emb"] is not None]
        if strict:  # the singer's own YouTube channel (its name contains the singer's full name)
            anchor = max(strict, key=lambda i: i["weight"])
        elif len(named) >= 3:  # web pages that name the singer in full, and at least 3 agree on the face
            for i in named:
                i["support"] = sum(1 for j in named if j is not i and float(i["emb"] @ j["emb"]) > SAME)
            best = max(named, key=lambda i: i["support"])
            if best["support"] >= 2:
                anchor = best
    for i in items:
        i["same_person"] = bool(anchor is not None and i["emb"] is not None
                                and (i is anchor or float(i["emb"] @ anchor["emb"]) > SAME))
    if anchor is None:
        log("  no photo can be tied to this singer with confidence; thumbnails use the text design", flush=True)
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


def _guided_filter(guide, src, radius, eps):
    """He et al. guided filter (box-filter form): snaps a coarse mask to the photo's real edges."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, k)
    mean_p = cv2.boxFilter(src, -1, k)
    corr_ip = cv2.boxFilter(guide * src, -1, k)
    var_i = cv2.boxFilter(guide * guide, -1, k) - mean_i * mean_i
    a = (corr_ip - mean_i * mean_p) / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, k) * guide + cv2.boxFilter(b, -1, k)


def person_mask(img):
    """Soft person mask (0..1): U^2-Net human segmentation, refined to full-resolution edges."""
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
    pred = _seg.run(None, {_seg.get_inputs()[0].name: x})[0][0, 0].astype(np.float32)
    # The network already outputs 0..1; stretching weak responses is what let backgrounds leak in.
    coarse = cv2.resize(np.clip(pred, 0, 1), (w, h), interpolation=cv2.INTER_CUBIC)
    binary = (coarse > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n > 2:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = cv2.dilate((labels == keep).astype(np.uint8), np.ones((25, 25), np.uint8))
        coarse = coarse * region
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    r = max(4, int(min(h, w) * 0.012))
    fine = _guided_filter(gray, coarse.astype(np.float32), r, 1e-3)
    mask = np.clip((fine - 0.2) / 0.6, 0, 1)
    mask = mask * mask * (3 - 2 * mask)  # smoothstep: crisp but anti-aliased edge
    return mask.astype(np.float32)


def _decontaminate(rgb, alpha):
    """Replace the old background's colour bleeding into soft edges with the person's own colours."""
    a = alpha[..., None].astype(np.float32)
    inner = (a > 0.95).astype(np.float32)
    k = (0, 0)
    num = cv2.GaussianBlur(rgb.astype(np.float32) * inner, k, 6)
    den = cv2.GaussianBlur(inner, k, 6)[..., None] if inner.ndim == 3 else cv2.GaussianBlur(inner[..., 0], k, 6)[..., None]
    est = num / np.maximum(den, 1e-3)
    w = np.clip((0.95 - a) / 0.95, 0, 1) * (den > 0.02)
    return np.clip(rgb * (1 - w) + est * w, 0, 255).astype(np.uint8)


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
    h, w = mask.shape
    # Where the photo itself cut the person off (frame edges), fade out instead of a hard line.
    ramp = max(8, int(min(h, w) * 0.10))
    edge = np.ones_like(mask)
    for side, strip in (("left", mask[:, :3]), ("right", mask[:, -3:]), ("bottom", mask[-3:, :])):
        if (strip > 0.5).mean() > 0.02:
            lin = np.linspace(0, 1, ramp, dtype=np.float32)
            if side == "left":
                edge[:, :ramp] *= lin[None, :]
            elif side == "right":
                edge[:, -ramp:] *= lin[::-1][None, :]
            else:
                edge[-ramp:, :] *= lin[::-1][:, None]
    touches_bottom = (mask[-3:, :] > 0.5).mean() > 0.02
    mask = mask * edge
    ys, xs = np.where(mask > 0.05)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = _decontaminate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask)
    rgba = np.dstack([rgb, (mask * 255).astype(np.uint8)])[y0:y1, x0:x1]
    out = Image.fromarray(rgba, "RGBA")
    out.info["touches_bottom"] = touches_bottom
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


def _harmonize(person_rgb, alpha, scene_rgb, amount=0.35):
    """Pull the person's brightness/colour cast part-way toward the scene behind them (LAB stats)."""
    a = alpha > 0.5
    if a.sum() < 100:
        return person_rgb
    p = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
    pm, ps = p[a].mean(0), p[a].std(0) + 1e-3
    bm, bs = b.mean(0), b.std(0) + 1e-3
    # Lightness: only a gentle shift (faces must stay readable); colour: match the scene's cast.
    target_m = pm + np.array([0.25, 1.0, 1.0]) * amount * (bm - pm)
    target_s = ps * (1 + np.array([0.0, 0.5, 0.5]) * amount * (bs / ps - 1))
    q = (p - pm) / ps * target_s + target_m
    return cv2.cvtColor(np.clip(q, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _person(canvas, person, side, accent):
    target_h = int(H * 0.98)
    face = person.info.get("face")
    s = target_h / person.height
    if face and face[3] * s > H * 0.36:  # keep the face a strong but not huge size
        s = (H * 0.36) / max(1, face[3])
    pw, ph = int(person.width * s), int(person.height * s)
    p = person.resize((pw, ph), Image.LANCZOS)
    cx = int(W * (0.74 if side == "right" else 0.26))
    x = cx - (int((face[0] + face[2] / 2) * s) if face else pw // 2)
    x = max(-pw // 3, min(W - 2 * pw // 3, x))
    y = H - ph if person.info.get("touches_bottom", True) or ph >= H * 0.9 else int(H * 0.5 - ph / 2 + H * 0.05)

    # Scene directly behind the person (for colour matching and light wrap).
    scene = np.array(canvas.convert("RGB"))
    sx0, sy0, sx1, sy1 = max(0, x), max(0, y), min(W, x + pw), min(H, y + ph)
    region = scene[sy0:sy1, sx0:sx1]
    arr = np.array(p)
    alpha = arr[..., 3].astype(np.float32) / 255
    rgb = arr[..., :3]
    if region.size:
        rgb = _harmonize(rgb, alpha, region)

    # 1) Soft backlight behind head and shoulders (not a traced outline).
    fx = x + (int((face[0] + face[2] / 2) * s) if face else pw // 2)
    fy = y + (int((face[1] + face[3] / 2) * s) if face else ph // 3)
    back = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(back)
    rad = int(max(pw, ph) * 0.42)
    d.ellipse((fx - rad, fy - rad * 0.8, fx + rad, fy + rad * 1.2), fill=accent + (70,))
    canvas.alpha_composite(back.filter(ImageFilter.GaussianBlur(rad * 0.45)))

    # 2) Light wrap: the scene's light spills a little over the person's edges.
    full = np.zeros((H, W, 3), np.float32)
    blurred = cv2.GaussianBlur(scene.astype(np.float32), (0, 0), 18)
    a_full = np.zeros((H, W), np.float32)
    rgb_full = np.zeros((H, W, 3), np.float32)
    px0, py0 = sx0 - x, sy0 - y
    a_full[sy0:sy1, sx0:sx1] = alpha[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)]
    rgb_full[sy0:sy1, sx0:sx1] = rgb[py0:py0 + (sy1 - sy0), px0:px0 + (sx1 - sx0)]
    inner = cv2.GaussianBlur(a_full, (0, 0), 7)
    wrap = np.clip(a_full - inner * 0.85, 0, 1) * 0.9 + (1 - inner) * a_full * 0.5
    wrap = np.clip(wrap, 0, 0.55)[..., None]
    rgb_full = rgb_full * (1 - wrap) + blurred * wrap
    # 3) Bottom of the body dissolves into the dark lower edge of the frame.
    fade = np.ones(H, np.float32)
    fade[int(H * 0.80):] = np.linspace(1, 0.35, H - int(H * 0.80))
    rgb_full *= np.minimum(1, 0.55 + 0.45 * fade)[:, None, None]
    # 4) Contact shadow so the person sits in the scene rather than floats.
    shadow = cv2.GaussianBlur(a_full, (0, 0), 14)
    canvas_arr = np.array(canvas.convert("RGB")).astype(np.float32)
    canvas_arr *= (1 - 0.35 * np.roll(np.roll(shadow, 10, 1), 8, 0))[..., None]
    out = canvas_arr * (1 - a_full[..., None]) + rgb_full * a_full[..., None]
    canvas.paste(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).convert("RGBA"))
    box = Image.fromarray((alpha * 255).astype(np.uint8)).point(lambda v: 255 if v > 90 else 0).getbbox() or (0, 0, pw, ph)
    return x + box[0], x + box[2]


def _wrap(text, mixed, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if mixed.length(w) > width:
            if cur:
                lines.append(cur)
            cur = ""
            for char in w:
                if cur and mixed.length(cur + char) > width:
                    lines.append(cur)
                    cur = ""
                cur += char
            continue
        test = (cur + " " + w).strip()
        if mixed.length(test) <= width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _text_block(canvas, meta, accent, accent_dark, top=54, bottom=666, compact=False):
    """Fit the complete centered bilingual block into its assigned vertical band."""
    width = W - 160
    title = meta.get("title_am") or meta.get("title_latin", "")
    roman = (meta.get("title_latin") or "").upper() if meta.get("title_am") else ""
    artist_am = meta.get("artist_am") or ""
    artist_lat = meta.get("artist_latin") or ""
    for size in range(60 if compact else 132, 7, -2):
        f = Mixed("black", size)
        rows = [("MEZMUR / LYRICS", Mixed("semibold", min(18 if compact else 24, max(10, size // 3))), "accent")]
        rows += [(line, f, "title") for line in _wrap(title, f, width)]
        for text, weight, maximum, color in (
            (roman, "semibold", max(12, round(size * 0.43)), "white"),
            (artist_am or artist_lat, "bold", min(30 if compact else 44, max(12, size // 2)), "white"),
            (artist_lat if artist_am else "", "medium", min(22 if compact else 32, max(10, round(size * 0.4))), "accent"),
        ):
            if text:
                font = fit(weight, text, maximum, width, min_size=8)
                rows += [(line, font, color) for line in _wrap(text, font, width)]
        heights = [sum(font.metrics()) for _, font, _ in rows]
        gap = max(4, round(size * 0.05))
        block_h = sum(heights) + gap * (len(rows) - 1)
        if block_h <= bottom - top:
            break
    y = top + max(0, (bottom - top - block_h) // 2)
    for (text, font, kind), line_h in zip(rows, heights):
        if kind == "title":
            gradient_text(canvas, (W // 2, y), text, font, (255, 244, 200), accent_dark,
                          anchor="ma", stroke=max(1, font.size // 32), offset=3, blur=6)
        else:
            fill = accent if kind == "accent" else (255, 255, 255)
            shadow_text(canvas, (W // 2, y), text, font, fill + (255,), anchor="ma", blur=3,
                        offset=2, stroke=1, stroke_fill=(10, 6, 2, 255))
        y += line_h + gap


PALETTES = {
    "gold": ((255, 200, 90), (225, 120, 30), True),
    "sky": ((120, 220, 255), (40, 120, 220), False),
    "rose": ((255, 170, 150), (205, 70, 90), True),
}


def compose(meta, background=None, person=None, photo=None, variant="a", seed=0):
    """Build one 1280x720 thumbnail.

    a/c: centered text only, with/without the outer frame.
    b/d: the video's circular portrait above smaller centered text, with/without
    the outer frame. Missing portraits fall back to text, never a guessed face.
    The legacy person argument is accepted but cutouts are no longer drawn.
    """
    if variant not in ("a", "b", "c", "d"):
        raise ValueError("Unknown thumbnail variant")
    accent, accent_dark, warm = PALETTES["gold" if variant in ("a", "c") else "sky"]
    canvas = _bg(background, warm)
    canvas.alpha_composite(Image.new("RGBA", (W, H), (5, 5, 8, 110)))
    if variant in ("b", "d") and photo is not None:
        from .visuals import circle_photo
        portrait = circle_photo(photo, 316, ring=accent, ring_w=4)
        canvas.alpha_composite(portrait, ((W - portrait.width) // 2, 80))
        _text_block(canvas, meta, accent, accent_dark, top=428, compact=True)
    else:
        _text_block(canvas, meta, accent, accent_dark)
    if variant in ("a", "b"):
        ImageDraw.Draw(canvas).rectangle((10, 10, W - 11, H - 11), outline=accent + (160,), width=3)
    return canvas.convert("RGB")


def save(img, path):
    img.save(path, "JPEG", quality=92, optimize=True, progressive=True)
    if Path(path).stat().st_size > 1_900_000:
        img.save(path, "JPEG", quality=82, optimize=True)
