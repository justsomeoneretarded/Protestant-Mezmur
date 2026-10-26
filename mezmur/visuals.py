"""Music-reactive background compositor. Frames are piped to FFmpeg, which adds lyrics (ASS)."""
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .features import Features, snap
from .textdraw import Mixed, fit, shadow_text, gradient_text

FONTS = Path(__file__).parent / "assets" / "fonts"
GOLD = (255, 214, 140)
WHITE = (255, 255, 255)


@dataclass
class Source:
    kind: str                 # "video" | "image" | "procedural"
    path: str = ""
    label: str = ""
    credit: str = ""
    blur: int = 0             # background-blur strength for photos of people / covers


@dataclass
class Scene:
    start: float
    end: float
    source: Source
    seed: int
    reader: object = None
    still: np.ndarray = None


def cover_resize(img, w, h):
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    img = cv2.resize(img, (max(w, int(round(iw * s))), max(h, int(round(ih * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


class VideoReader:
    def __init__(self, path, w, h, fps):
        self.path, self.w, self.h, self.fps = path, w, h, fps
        self.cap = cv2.VideoCapture(path)
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self.step = src_fps / fps
        self.pos = 0.0
        self.last = None

    def read(self):
        target = int(self.pos)
        self.pos += self.step
        frame = None
        while True:
            current = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            if current > target and self.last is not None:
                return self.last
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.pos = self.step
                ok, frame = self.cap.read()
                if not ok:
                    return self.last if self.last is not None else np.zeros((self.h, self.w, 3), np.uint8)
                break
            if current >= target:
                break
        self.last = cover_resize(frame, self.w, self.h)
        return self.last


PALETTES = [  # (deep background, glow 1, glow 2, light) in RGB — warm, worshipful colour schemes
    [(18, 10, 38), (214, 120, 48), (120, 40, 110), (255, 220, 160)],   # dusk gold / plum
    [(8, 20, 42), (40, 110, 190), (230, 170, 80), (255, 236, 190)],    # night blue / sunrise
    [(26, 12, 20), (190, 70, 90), (240, 160, 70), (255, 214, 170)],    # crimson / amber
    [(6, 26, 30), (40, 160, 150), (110, 70, 170), (210, 255, 240)],    # teal aurora
    [(30, 22, 10), (200, 150, 60), (90, 60, 30), (255, 240, 200)],     # golden glory
    [(12, 14, 34), (90, 80, 200), (220, 120, 160), (240, 225, 255)],   # violet dawn
]


def procedural(w, h, t, seed, energy):
    """Generated background (always available): drifting colour glows, soft light rays from
    above and slow breathing with the music. One palette per song seed."""
    rng = np.random.default_rng(seed)
    bg, c1, c2, light = (np.array(c, np.float32) for c in PALETTES[int(rng.integers(len(PALETTES)))])
    sw, sh = 192, 108
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    xx /= sw
    yy /= sh
    img = np.broadcast_to(bg, (sh, sw, 3)).copy()
    phases = rng.uniform(0, 2 * np.pi, 6)
    for k, (col, size) in enumerate([(c1, 0.55), (c2, 0.45), (c1 * 0.6 + c2 * 0.4, 0.35)]):
        cx = 0.5 + 0.32 * math.sin(t * (0.045 + 0.013 * k) + phases[k])
        cy = 0.5 + 0.25 * math.cos(t * (0.037 + 0.011 * k) + phases[k + 3])
        d2 = (xx - cx) ** 2 + ((yy - cy) * 0.75) ** 2
        g = np.exp(-d2 / (2 * (size * (0.9 + 0.25 * energy)) ** 2 * 0.25))[..., None]
        img = img * (1 - 0.85 * g) + col * 0.85 * g
    # Soft rays fanning down from a light source above the frame.
    ox = 0.5 + 0.18 * math.sin(t * 0.03 + phases[0])
    ang = np.arctan2(xx - ox, yy + 0.35)
    rays = (0.5 + 0.5 * np.sin(ang * 14 + t * 0.15 + phases[1])) ** 6
    rays *= np.clip(1.15 - yy * 1.1, 0, 1) * (0.22 + 0.18 * energy)
    img = img + light * rays[..., None] * 0.55
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.2)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img[..., ::-1].copy()


def rounded_rect(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def circle_photo(img_bgr, size, ring=(255, 214, 140), ring_w=6):
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    side = min(img.size)
    img = img.crop(((img.width - side) // 2, 0, (img.width - side) // 2 + side, side)).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    out = Image.new("RGBA", (size + 2 * ring_w, size + 2 * ring_w), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    d.ellipse((0, 0, out.width - 1, out.height - 1), fill=ring + (255,))
    out.paste(img, (ring_w, ring_w), mask)
    return out


def to_rgba_array(img):
    a = np.asarray(img).astype(np.float32)
    return np.ascontiguousarray(a[..., [2, 1, 0]]), a[..., 3:4] / 255.0


def blend(frame, rgb, alpha, x=0, y=0, opacity=1.0):
    h, w = rgb.shape[:2]
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1 or opacity <= 0.003:
        return
    sub = frame[y0:y1, x0:x1].astype(np.float32)
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
    sub = sub * (1 - a) + rgb[y0 - y:y1 - y, x0 - x:x1 - x] * a
    frame[y0:y1, x0:x1] = np.clip(sub, 0, 255).astype(np.uint8)


@dataclass
class Overlays:
    badge: tuple = None        # (rgb, alpha, x, y)
    intro: tuple = None        # (rgb, alpha)
    outro: tuple = None
    labels: dict = field(default_factory=dict)


class Renderer:
    def __init__(self, width, height, fps, duration, features: Features, sources, cues, meta,
                 artist_photo=None, seed=0, end_screen=20.0):
        self.W, self.H, self.fps, self.duration = width, height, fps, duration
        self.f = features
        self.cues = cues
        self.meta = meta
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.first_vocal = cues[0]["start"] if cues else duration
        last_vocal = cues[-1]["end"] if cues else 0
        self.intro_end = float(np.clip(self.first_vocal - 0.4, 3.5, 9.0)) if self.first_vocal > 2.0 else 0.0
        self.outro_start = max(last_vocal + 1.0, duration - end_screen) if duration - last_vocal > 6 else duration
        self.scenes = self._schedule(sources)
        self.artist_photo = artist_photo
        self.overlays = self._overlays()
        self._vignette = self._make_vignette()
        self._init_particles()
        self.bars = np.zeros(features.bands.shape[1], np.float32)
        self._clock_cache = {}

    # ---------- planning ----------
    def _schedule(self, sources):
        sources = [s for s in sources if s] or [Source("procedural", label="light")]
        bars = self.f.downbeats
        target = 13.0
        times = [0.0]
        t = 0.0
        while t + target < self.duration - 4:
            t = snap(t + target, bars, 2.0)
            times.append(t)
        times.append(self.duration)
        order = list(range(len(sources)))
        self.rng.shuffle(order)
        scenes = []
        for i in range(len(times) - 1):
            src = sources[order[i % len(order)]]
            scenes.append(Scene(times[i], times[i + 1], src, seed=int(self.rng.integers(1e9))))
        return scenes

    def _make_vignette(self):
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        nx, ny = (xx / self.W - 0.5) * 2, (yy / self.H - 0.5) * 2
        vig = 1.0 - 0.42 * np.clip(nx ** 2 * 0.55 + ny ** 2 * 0.8, 0, 1.3)
        band = 1.0 - 0.38 * np.exp(-((yy / self.H - 0.56) / 0.17) ** 4)  # darken lyric zone
        m = np.clip(vig * band, 0, 1)
        return (m[..., None] * 255).astype(np.uint8).repeat(3, axis=2)

    def _init_particles(self, n=70):
        r = self.rng
        self.p_pos = np.stack([r.uniform(0, 1, n), r.uniform(0, 1, n)], 1)
        self.p_size = r.uniform(0.8, 3.4, n)
        self.p_speed = r.uniform(0.01, 0.04, n)
        self.p_phase = r.uniform(0, 2 * np.pi, n)
        self.p_warm = r.uniform(0, 1, n) < 0.6

    def _overlays(self):
        o = Overlays()
        m = self.meta
        # Top-left badge: circular artist photo + names.
        badge = Image.new("RGBA", (900, 150), (0, 0, 0, 0))
        x = 0
        if self.artist_photo is not None:
            photo = circle_photo(self.artist_photo, 104, ring_w=4)
            badge.alpha_composite(photo, (0, 12))
            x = photo.width + 18
        name_am = m.get("artist_am") or m.get("artist_latin") or ""
        name_lat = m.get("artist_latin") if m.get("artist_am") else ""
        shadow_text(badge, (x, 22), name_am, fit("bold", name_am, 38, 640), WHITE + (240,), blur=4)
        sub = "  •  ".join(p for p in [name_lat, m.get("title_am", ""), m.get("title_latin", "")] if p)
        shadow_text(badge, (x + 2, 80), sub, fit("semibold", sub, 25, 720), GOLD + (235,), blur=4)
        o.badge = (*to_rgba_array(badge), 48, 34)
        o.intro = to_rgba_array(self._title_card())
        o.outro = to_rgba_array(self._outro_card())
        return o

    def _title_card(self):
        W, H, m = self.W, self.H, self.meta
        card = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(shade).rectangle((0, 0, W, H), fill=(8, 6, 4, 120))
        card.alpha_composite(shade)
        cy = int(H * 0.30)
        if self.artist_photo is not None:
            photo = circle_photo(self.artist_photo, int(H * 0.30), ring_w=8)
            glow = Image.new("RGBA", (photo.width + 120, photo.height + 120), (0, 0, 0, 0))
            ImageDraw.Draw(glow).ellipse((40, 40, glow.width - 40, glow.height - 40), fill=GOLD + (150,))
            glow = glow.filter(ImageFilter.GaussianBlur(28))
            card.alpha_composite(glow, ((W - glow.width) // 2, cy - glow.height // 2))
            card.alpha_composite(photo, ((W - photo.width) // 2, cy - photo.height // 2))
            ty = cy + photo.height // 2 + int(H * 0.07)
        else:
            ty = int(H * 0.38)
        title = m.get("title_am") or m.get("title_latin", "")
        f = fit("black", title, int(H * 0.11), int(W * 0.85))
        gradient_text(card, (W // 2, ty), title, f, (255, 236, 180), (232, 160, 60), anchor="mm", stroke=3)
        ty += int(f.size * 0.95)
        if m.get("title_latin") and m.get("title_am"):
            f = fit("bold", m["title_latin"].upper(), int(H * 0.045), int(W * 0.8))
            shadow_text(card, (W // 2, ty), m["title_latin"].upper(), f, WHITE + (245,), anchor="mm", blur=6)
            ty += int(f.size * 1.6)
        artist = "  •  ".join(p for p in [m.get("artist_am"), m.get("artist_latin")] if p)
        if artist:
            f = fit("semibold", artist, int(H * 0.04), int(W * 0.8))
            shadow_text(card, (W // 2, ty), artist, f, (235, 235, 235, 240), anchor="mm", blur=6)
            ty += int(f.size * 1.7)
        tag = "ግጥም  •  LYRICS"
        f = Mixed("bold", int(H * 0.026))
        tw = f.length(tag)
        th = f.size
        d = ImageDraw.Draw(card)
        rounded_rect(d, (W // 2 - tw // 2 - 26, ty - th // 2 - 14, W // 2 + tw // 2 + 26, ty + th // 2 + 14), 30, GOLD + (235,))
        f.draw(d, (W // 2, ty), tag, (30, 20, 8, 255), anchor="mm")
        return card

    def _outro_card(self):
        W, H, m = self.W, self.H, self.meta
        card = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(card).rectangle((0, 0, W, H), fill=(5, 4, 3, 130))
        gradient_text(card, (W // 2, int(H * 0.13)), "ተባረኩ  •  Be blessed", Mixed("black", int(H * 0.07)),
                      (255, 236, 180), (232, 160, 60), anchor="mm", stroke=2)
        title = "  •  ".join(p for p in [m.get("title_am"), m.get("title_latin")] if p)
        shadow_text(card, (W // 2, int(H * 0.235)), title, fit("semibold", title, int(H * 0.04), int(W * 0.7)),
                    WHITE + (230,), anchor="mm", blur=5)
        shadow_text(card, (W // 2, int(H * 0.9)), "Subscribe for more Amharic mezmur lyrics  •  ለተጨማሪ መዝሙሮች ሰብስክራይብ ያድርጉ",
                    Mixed("semibold", int(H * 0.028)), (240, 240, 240, 220), anchor="mm", blur=5)
        return card

    # ---------- per-frame ----------
    def _scene_frame(self, scene, t):
        local = t - scene.start
        length = max(1e-3, scene.end - scene.start)
        rng = np.random.default_rng(scene.seed)
        zoom_from, zoom_to = (1.0, 1.10) if rng.random() < 0.5 else (1.10, 1.0)
        pan = rng.uniform(-0.04, 0.04, 2)
        p = local / length
        p = p * p * (3 - 2 * p)
        zoom = zoom_from + (zoom_to - zoom_from) * p
        if scene.source.kind == "video":
            if scene.reader is None:
                scene.reader = VideoReader(scene.source.path, self.W, self.H, self.fps)
            img = scene.reader.read()
        elif scene.source.kind == "image":
            if scene.still is None:
                img = cv2.imread(scene.source.path)
                if img is None:
                    img = procedural(self.W, self.H, t, scene.seed, 0.5)
                img = cover_resize(img, self.W, self.H)
                if scene.source.blur:
                    k = scene.source.blur | 1
                    img = cv2.GaussianBlur(img, (k, k), 0)
                    img = cv2.convertScaleAbs(img, alpha=0.8, beta=0)
                scene.still = img
            img = scene.still
        else:
            img = procedural(self.W, self.H, t, scene.seed, float(self.f.rms[self.f.at(t)]))
        pulse = 0.018 * float(self.f.beat_env[self.f.at(t)]) * (0.5 + float(self.f.rms[self.f.at(t)]))
        z = zoom + pulse
        cx, cy = self.W / 2 + pan[0] * self.W * (p - 0.5), self.H / 2 + pan[1] * self.H * (p - 0.5)
        M = np.float32([[z, 0, (1 - z) * cx], [0, z, (1 - z) * cy]])
        return cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def background(self, t):
        idx = next((i for i, s in enumerate(self.scenes) if s.start <= t < s.end), len(self.scenes) - 1)
        scene = self.scenes[idx]
        frame = self._scene_frame(scene, t)
        fade = 1.2
        if idx + 1 < len(self.scenes) and t > scene.end - fade:
            nxt = self._scene_frame(self.scenes[idx + 1], t)
            a = (t - (scene.end - fade)) / fade
            a = a * a * (3 - 2 * a)
            frame = cv2.addWeighted(frame, 1 - a, nxt, a, 0)
        return frame

    def _grade(self, frame, t):
        e = float(self.f.rms[self.f.at(t)])
        b = float(self.f.beat_env[self.f.at(t)])
        gain = 0.78 + 0.22 * e + 0.05 * b
        x = np.arange(256, dtype=np.float32)
        lut_b = np.clip(x * gain * 0.94, 0, 255)
        lut_g = np.clip(x * gain * 1.0, 0, 255)
        lut_r = np.clip(x * gain * 1.06 + 4, 0, 255)
        lut = np.stack([lut_b, lut_g, lut_r], 1).astype(np.uint8)[:, None, :]
        frame = cv2.LUT(frame, lut)
        return cv2.multiply(frame, self._vignette, scale=1 / 255.0)

    def _particles(self, frame, t, dt):
        e = float(self.f.rms[self.f.at(t)])
        b = float(self.f.beat_env[self.f.at(t)])
        self.p_pos[:, 1] -= self.p_speed * dt * (0.6 + 2.2 * e)
        self.p_pos[:, 0] += 0.004 * dt * np.sin(t * 0.8 + self.p_phase)
        wrap = self.p_pos[:, 1] < -0.05
        self.p_pos[wrap, 1] = 1.05
        self.p_pos[wrap, 0] = self.rng.uniform(0, 1, wrap.sum())
        sw, sh = self.W // 3, self.H // 3
        layer = np.zeros((sh, sw, 3), np.uint8)
        twinkle = 0.55 + 0.45 * np.sin(t * 2.0 + self.p_phase)
        for (x, y), s, tw, warm in zip(self.p_pos, self.p_size, twinkle, self.p_warm):
            band = 0.35 if 0.38 < y < 0.75 else 1.0
            level = int(min(255, (40 + 120 * tw) * (0.5 + 0.7 * e + 0.5 * b) * band))
            color = (int(level * 0.55), int(level * 0.82), level) if warm else (level, level, level)
            cv2.circle(layer, (int(x * sw), int(y * sh)), max(1, int(s * (1 + 0.4 * b))), color, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.6)
        layer = cv2.resize(layer, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return cv2.add(frame, layer)

    def _spectrum(self, frame, t):
        i = self.f.at(t)
        target = self.f.bands[i]
        self.bars = np.where(target > self.bars, self.bars + 0.6 * (target - self.bars), self.bars * 0.88)
        n = len(self.bars)
        width = int(self.W * 0.62)
        x0 = (self.W - width) // 2
        base = self.H - 70
        maxh = int(self.H * 0.085)
        strip_y0 = base - maxh - 4
        strip = frame[strip_y0:base + maxh // 3 + 4].copy()
        bw = width / (2 * n)
        values = np.concatenate([self.bars[::-1], self.bars])
        for k, v in enumerate(values):
            h = int(4 + v * maxh)
            xa = int(x0 + k * bw + bw * 0.2)
            xb = int(x0 + (k + 1) * bw - bw * 0.2)
            warm = 0.5 + 0.5 * v
            color = (int(140 * warm), int(214 * warm + 30), 255)
            cv2.rectangle(strip, (xa, base - h - strip_y0), (xb, base - strip_y0), color, -1)
            cv2.rectangle(strip, (xa, base - strip_y0 + 3), (xb, base - strip_y0 + 3 + h // 3), tuple(int(c * 0.35) for c in color), -1)
        frame[strip_y0:base + maxh // 3 + 4] = cv2.addWeighted(frame[strip_y0:base + maxh // 3 + 4], 0.42, strip, 0.58, 0)
        return frame

    def _progress(self, frame, t):
        y = self.H - 22
        x0, x1 = 48, self.W - 48
        cv2.line(frame, (x0, y), (x1, y), (90, 90, 90), 3, cv2.LINE_AA)
        xp = int(x0 + (x1 - x0) * min(1.0, t / self.duration))
        cv2.line(frame, (x0, y), (xp, y), (140, 214, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, (xp, y), 6, (200, 235, 255), -1, cv2.LINE_AA)
        sec = int(t)
        if sec not in self._clock_cache:
            txt = f"{sec // 60}:{sec % 60:02d} / {int(self.duration) // 60}:{int(self.duration) % 60:02d}"
            img = Image.new("RGBA", (260, 40), (0, 0, 0, 0))
            shadow_text(img, (258, 20), txt, Mixed("semibold", 22), (230, 230, 230, 230), anchor="rm", blur=3, offset=2)
            self._clock_cache = {sec: to_rgba_array(img)}
        rgb, a = self._clock_cache[sec]
        blend(frame, rgb, a, self.W - 48 - 260, self.H - 66)
        return frame

    def _countdown(self, frame, t):
        """Karaoke-style dots before vocals return after a long break."""
        upcoming = next((c for c in self.cues if c["start"] > t), None)
        if upcoming is None:
            return frame
        prev_end = max((c["end"] for c in self.cues if c["end"] <= t), default=0.0)
        if upcoming["start"] - prev_end < 5.0 or t < self.intro_end - 0.5:
            return frame
        beat = 60.0 / self.f.tempo if self.f.tempo else 0.5
        remaining = upcoming["start"] - t
        lead = 3 * beat
        if remaining > lead + 0.6:
            return frame
        n = int(math.ceil(remaining / beat))
        n = max(0, min(3, n))
        cy = int(self.H * 0.44)
        for k in range(3):
            on = k < n
            cx = self.W // 2 + (k - 1) * 46
            color = (140, 214, 255) if on else (80, 80, 80)
            r = 13 if on else 9
            cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)
        return frame

    def frame(self, t, dt):
        frame = self.background(t)
        frame = self._grade(frame, t)
        frame = self._particles(frame, t, dt)
        frame = self._spectrum(frame, t)
        o = self.overlays
        if t < self.intro_end:
            a = min(1.0, t / 0.6, (self.intro_end - t) / 0.8)
            blend(frame, *o.intro, 0, 0, opacity=max(0.0, a))
        elif t < self.outro_start:
            a = min(1.0, (t - self.intro_end) / 1.0, (self.outro_start - t) / 1.0)
            blend(frame, o.badge[0], o.badge[1], o.badge[2], o.badge[3], opacity=max(0.0, a))
        if t >= self.outro_start:
            blend(frame, *o.outro, 0, 0, opacity=min(1.0, (t - self.outro_start) / 1.2))
        frame = self._countdown(frame, t)
        return self._progress(frame, t)


def render_video(renderer: Renderer, audio, ass_path, out_path, crf=18, preset="medium", log=print):
    W, H, fps = renderer.W, renderer.H, renderer.fps
    total = int(math.ceil(renderer.duration * fps))
    ass = str(Path(ass_path).resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    fontsdir = str(FONTS.resolve()).replace(":", r"\:")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0",
           "-i", str(audio),
           "-vf", f"ass='{ass}':fontsdir='{fontsdir}'",
           "-map", "0:v", "-map", "1:a", "-t", f"{renderer.duration:.3f}",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-g", str(int(fps * 2)), "-bf", "2",
           "-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dt = 1.0 / fps
    try:
        for i in range(total):
            frame = renderer.frame(i * dt, dt)
            proc.stdin.write(frame.tobytes())
            if i % int(fps * 30) == 0:
                log(f"  render {i / fps:.0f}/{renderer.duration:.0f}s", flush=True)
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code:
        raise RuntimeError(f"FFmpeg render failed with exit code {code}")
