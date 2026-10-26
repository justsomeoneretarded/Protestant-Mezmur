"""Mixed Ge'ez/Latin text drawing with Pillow (each script uses its own Noto font)."""
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path(__file__).parent / "assets" / "fonts"
WEIGHTS = {
    "black": ("NotoSansEthiopic-Black.ttf", "NotoSans-Black.ttf"),
    "bold": ("NotoSansEthiopic-Bold.ttf", "NotoSans-Bold.ttf"),
    "semibold": ("NotoSansEthiopic-SemiBold.ttf", "NotoSans-SemiBold.ttf"),
    "medium": ("NotoSansEthiopic-Medium.ttf", "NotoSans-SemiBold.ttf"),
    "italic": ("NotoSansEthiopic-Medium.ttf", "NotoSans-MediumItalic.ttf"),
}
RUN = re.compile(r"([ሀ-᎟ⶀ-⷟꬀-꬯፡።፣፤፥፦፧፨]+(?:\s+[ሀ-᎟ⶀ-⷟꬀-꬯፡።፣፤፥፦፧፨]+)*)")


@lru_cache(maxsize=256)
def _font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


class Mixed:
    def __init__(self, weight, size):
        self.size = size
        self.eth = _font(WEIGHTS[weight][0], size)
        self.lat = _font(WEIGHTS[weight][1], size)

    def runs(self, text):
        out = []
        for part in RUN.split(text):
            if part:
                out.append((part, self.eth if RUN.fullmatch(part) else self.lat))
        return out

    def length(self, text):
        return sum(f.getlength(t) for t, f in self.runs(text))

    def metrics(self):
        a1, d1 = self.eth.getmetrics()
        a2, d2 = self.lat.getmetrics()
        return max(a1, a2), max(d1, d2)

    def draw(self, draw, xy, text, fill, anchor="la", stroke=0, stroke_fill=None):
        x, y = xy
        width = self.length(text)
        ascent, descent = self.metrics()
        if anchor[0] == "m":
            x -= width / 2
        elif anchor[0] == "r":
            x -= width
        vertical = anchor[1]
        if vertical == "m":
            y = y + (ascent - descent) / 2 - descent * 0.15
        elif vertical in ("a", "t"):
            y = y + ascent
        for part, fnt in self.runs(text):
            draw.text((x, y), part, font=fnt, fill=fill, anchor="ls", stroke_width=stroke,
                      stroke_fill=stroke_fill or fill)
            x += fnt.getlength(part)
        return width


def fit(weight, text, max_size, max_width, min_size=16):
    size = max_size
    while size > min_size and Mixed(weight, size).length(text) > max_width:
        size -= 2
    return Mixed(weight, size)


def shadow_text(canvas, xy, text, mixed, fill, anchor="la", shadow=(0, 0, 0, 170), offset=3, blur=6,
                stroke=0, stroke_fill=(20, 12, 4, 255), glow=None):
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if glow:
        mixed.draw(d, xy, text, glow, anchor, stroke=max(stroke, 6), stroke_fill=glow)
        layer = layer.filter(ImageFilter.GaussianBlur(blur * 2))
        canvas.alpha_composite(layer)
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
    mixed.draw(d, (xy[0] + offset, xy[1] + offset), text, shadow, anchor, stroke=stroke, stroke_fill=shadow)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    canvas.alpha_composite(layer)
    mixed.draw(ImageDraw.Draw(canvas), xy, text, fill, anchor, stroke=stroke, stroke_fill=stroke_fill)


def gradient_text(canvas, xy, text, mixed, top, bottom, anchor="la", stroke=0, stroke_fill=(25, 14, 4, 255),
                  shadow=(0, 0, 0, 200), offset=5, blur=10):
    """Text filled with a vertical gradient (e.g. gold) plus stroke and drop shadow."""
    w, h = canvas.size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    mixed.draw(d, (xy[0] + offset, xy[1] + offset), text, shadow, anchor, stroke=stroke, stroke_fill=shadow)
    canvas.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))
    if stroke:
        outline = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mixed.draw(ImageDraw.Draw(outline), xy, text, stroke_fill, anchor, stroke=stroke, stroke_fill=stroke_fill)
        canvas.alpha_composite(outline)
    mask = Image.new("L", (w, h), 0)
    mixed.draw(ImageDraw.Draw(mask), xy, text, 255, anchor)
    box = mask.getbbox()
    if not box:
        return
    grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    y0, y1 = box[1], box[3]
    for y in range(y0, y1 + 1):
        k = (y - y0) / max(1, y1 - y0)
        c = tuple(int(top[i] + (bottom[i] - top[i]) * k) for i in range(3)) + (255,)
        gd.line([(box[0], y), (box[2], y)], fill=c)
    canvas.paste(grad, (0, 0), mask)
