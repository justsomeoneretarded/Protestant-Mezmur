"""Background footage: themes from the lyrics, free stock sources, local fallbacks."""
import os
import random
import subprocess
from pathlib import Path

import cv2

from . import web
from .geez import fold
from .visuals import Source

# Amharic word stems (folded) -> English footage searches.
THEMES = [
    ("ብርሃን", "golden light rays"), ("ፀሀይ", "sunrise timelapse"), ("ጸሀይ", "sunrise timelapse"),
    ("ሰማይ", "sky clouds timelapse"), ("ደመና", "clouds timelapse"), ("ተራራ", "mountains aerial"),
    ("ባህር", "ocean waves sunset"), ("ወንዝ", "river nature"), ("ውሀ", "water stream nature"), ("ዝናብ", "rain nature"),
    ("ኮከብ", "starry night sky"), ("ሌሊት", "night sky stars"), ("ንጋት", "dawn sunrise"), ("መስቀል", "cross silhouette sunset"),
    ("ሻማ", "candle flame"), ("እሳት", "candle flame"), ("አበባ", "flowers field"), ("ዛፍ", "forest trees sunlight"),
    ("በረሀ", "desert dunes"), ("ምድረ በዳ", "desert dunes"), ("ሰላም", "calm lake sunrise"), ("ክብር", "golden light sky"),
    ("ዙፋን", "majestic clouds"), ("ንጉስ", "majestic sky sunset"), ("እረኛ", "sheep meadow"), ("በግ", "sheep meadow"),
    ("ርግብ", "dove flying"), ("ንስር", "eagle flying"), ("ክንፍ", "birds flying sky"), ("መንገድ", "road journey nature"),
    ("ጸሎት", "praying hands light"), ("አምልኮ", "worship hands raised"), ("ደስታ", "joyful sunrise"),
    ("ማእበል", "storm clouds"), ("ጨለማ", "light in darkness"), ("ህይወት", "spring nature"), ("ፍቅር", "warm sunset"),
]
DEFAULT = ["sunrise clouds timelapse", "golden hour nature", "mountains mist", "light rays forest", "ocean sunset",
           "sky clouds timelapse", "flowers field sunlight", "calm lake sunrise"]


def themes_from_lyrics(text, limit=6, seed=0):
    key = fold(text)
    found = []
    for stem, query in THEMES:
        count = key.count(fold(stem))
        if count:
            found.append((count, query))
    found.sort(key=lambda x: -x[0])
    queries = []
    for _, q in found:
        if q not in queries:
            queries.append(q)
    rnd = random.Random(seed)
    extra = DEFAULT[:]
    rnd.shuffle(extra)
    for q in extra:
        if len(queries) >= limit:
            break
        if q not in queries:
            queries.append(q)
    return queries[:limit]


def _pexels(query, key, used):
    r = web.get("https://api.pexels.com/videos/search", params={"query": query, "orientation": "landscape",
                "size": "large", "per_page": 15}, headers={"Authorization": key})
    if r is None or r.status_code != 200:
        return []
    out = []
    for v in r.json().get("videos", []):
        vid = f"pexels:{v['id']}"
        if vid in used or not 8 <= v.get("duration", 0) <= 90:
            continue
        files = sorted([f for f in v.get("video_files", []) if f.get("width") and f["width"] >= 1280 and
                        f.get("file_type") == "video/mp4"], key=lambda f: abs(f["width"] - 1920))
        if files:
            out.append({"id": vid, "url": files[0]["link"], "credit": f"Video by {v.get('user', {}).get('name', 'Pexels')} on Pexels",
                        "page": v.get("url")})
    return out


def _pixabay(query, key, used):
    r = web.get("https://pixabay.com/api/videos/", params={"key": key, "q": query, "per_page": 20, "safesearch": "true"})
    if r is None or r.status_code != 200:
        return []
    out = []
    for v in r.json().get("hits", []):
        vid = f"pixabay:{v['id']}"
        if vid in used or not 8 <= v.get("duration", 0) <= 90:
            continue
        f = v.get("videos", {}).get("large") or v.get("videos", {}).get("medium") or {}
        if f.get("url") and f.get("width", 0) >= 1280:
            out.append({"id": vid, "url": f["url"], "credit": f"Video by {v.get('user', 'Pixabay')} on Pixabay",
                        "page": v.get("pageURL")})
    return out


def _commons(query, used):
    r = web.get("https://commons.wikimedia.org/w/api.php", params={
        "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:video", "gsrnamespace": 6,
        "gsrlimit": 20, "prop": "videoinfo", "viprop": "url|size|derivatives|extmetadata|mediatype", "format": "json"})
    if r is None or r.status_code != 200:
        return []
    out = []
    for page in (r.json().get("query") or {}).get("pages", {}).values():
        info = (page.get("videoinfo") or [{}])[0]
        vid = f"commons:{page.get('pageid')}"
        if vid in used or not info:
            continue
        if (info.get("width") or 0) < 1280 or not 8 <= float(info.get("duration") or 0) <= 600:
            continue
        derivs = [d for d in info.get("derivatives", []) if d.get("height") in (1080, 720) and "webm" in d.get("type", "")]
        url = sorted(derivs, key=lambda d: -d.get("height", 0))[0]["src"] if derivs else info.get("url")
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not any(x in lic for x in ["CC0", "Public domain", "CC BY", "CC-BY"]):
            continue
        artist = (meta.get("Artist") or {}).get("value", "")
        import re
        artist = re.sub(r"<[^>]+>", "", artist).strip()[:80]
        out.append({"id": vid, "url": url, "credit": f"{page.get('title', '').replace('File:', '')} — {artist or 'Wikimedia Commons'} ({lic})",
                    "page": info.get("descriptionurl")})
    return out


def _prepare(url, target, seconds=24):
    raw = target.with_suffix(".src")
    r = web.get(url, timeout=120, max_bytes=120_000_000)
    if r is None or r.status_code != 200 or len(r.content) < 200_000:
        return False
    raw.write_bytes(r.content)
    code = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "1", "-t", str(seconds), "-i", str(raw),
                           "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,format=yuv420p",
                           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", str(target)]).returncode
    raw.unlink(missing_ok=True)
    if code or not target.is_file():
        return False
    cap = cv2.VideoCapture(str(target))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 60
    cap.release()
    return ok


def footage(lyrics_text, folder, used=(), wanted=6, seed=0, log=print):
    """Download and normalize stock clips. Returns (sources, records)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    used = set(used)
    pexels, pixabay = os.environ.get("PEXELS_API_KEY", "").strip(), os.environ.get("PIXABAY_API_KEY", "").strip()
    queries = themes_from_lyrics(lyrics_text, limit=wanted, seed=seed)
    sources, records = [], []
    rnd = random.Random(seed)
    for q in queries:
        if len(sources) >= wanted:
            break
        options = []
        try:
            if pexels:
                options += _pexels(q, pexels, used)
            if pixabay and len(options) < 3:
                options += _pixabay(q, pixabay, used)
            if not options:
                options += _commons(q, used)
        except Exception as error:
            log(f"  footage search {q!r} failed: {error}", flush=True)
        rnd.shuffle(options)
        for opt in options[:3]:
            target = folder / f"clip{len(sources)}.mp4"
            if _prepare(opt["url"], target):
                used.add(opt["id"])
                sources.append(Source("video", str(target), label=q, credit=opt["credit"]))
                records.append({**opt, "query": q})
                log(f"  footage: {q!r} <- {opt['id']}", flush=True)
                break
    return sources, records


def local_scenes(folder):
    """The four bundled scenic paintings as separate image sources."""
    atlas = cv2.imread(str(Path(__file__).parent / "assets" / "scenery.png"))
    if atlas is None:
        return []
    h, w = atlas.shape[:2]
    out = []
    for i, (y, x) in enumerate([(0, 0), (0, w // 2), (h // 2, 0), (h // 2, w // 2)]):
        path = Path(folder) / f"scene{i}.png"
        cv2.imwrite(str(path), atlas[y:y + h // 2, x:x + w // 2])
        out.append(Source("image", str(path), label="bundled scene"))
    return out
