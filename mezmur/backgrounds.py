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
        title = page.get("title", "").lower()
        if any(x in title for x in ("documentary", "interview", "lecture", "news", "trailer", "presentation", "talk", "speech")):
            continue
        url = sorted(derivs, key=lambda d: (d.get("height") != 1080, d.get("height", 0)))[0]["src"] if derivs else info.get("url")
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


def _probe_seconds(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def _text_free(target, log=print):
    """Reject clips with burned-in titles, captions or watermarks (checked at 3 points)."""
    from .textdetect import text_score
    cap = cv2.VideoCapture(str(target))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    worst = 0.0
    for f in (0.15, 0.5, 0.85):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f))
        ok, frame = cap.read()
        if ok:
            worst = max(worst, text_score(frame))
    cap.release()
    if worst >= 0.012:
        log(f"  footage rejected (text on screen, score {worst:.3f}): {target.name}", flush=True)
        return False
    return True


def _prepare(url, target, seconds=24, log=print):
    raw = target.with_suffix(".src")
    r = web.get(url, timeout=90, max_bytes=80_000_000)
    if r is None or r.status_code != 200 or len(r.content) < 200_000:
        return False
    raw.write_bytes(r.content)
    total = _probe_seconds(raw)
    # Take the middle of the clip: title cards and end credits live at the edges.
    start = max(1.0, min(total * 0.35, total - seconds - 1.0)) if total > seconds + 2 else 0.5
    code = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", str(seconds), "-i", str(raw),
                           "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,format=yuv420p",
                           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", str(target)]).returncode
    raw.unlink(missing_ok=True)
    if code or not target.is_file():
        return False
    cap = cv2.VideoCapture(str(target))
    ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 60
    cap.release()
    return ok and _text_free(target, log)


def footage(lyrics_text, folder, used=(), wanted=6, seed=0, log=print):
    """Find, download and normalize stock clips (in parallel). Returns (sources, records)."""
    from concurrent.futures import ThreadPoolExecutor
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    used = set(used)
    pexels, pixabay = os.environ.get("PEXELS_API_KEY", "").strip(), os.environ.get("PIXABAY_API_KEY", "").strip()
    queries = themes_from_lyrics(lyrics_text, limit=wanted + 2, seed=seed)
    rnd = random.Random(seed)
    plan = []  # (query, [options...]) in priority order
    for q in queries:
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
        if options:
            plan.append((q, options[:3]))
    if not pexels and not pixabay:
        log("  tip: add free PEXELS_API_KEY / PIXABAY_API_KEY secrets for more and better footage", flush=True)

    def attempt(job):
        index, (q, options) = job
        for k, opt in enumerate(options):
            if opt["id"] in used:
                continue
            target = folder / f"clip{index}_{k}.mp4"
            if _prepare(opt["url"], target, log=log):
                return index, q, opt, target
        return index, q, None, None

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, q, opt, target in pool.map(attempt, list(enumerate(plan))):
            if opt is not None and opt["id"] not in used:
                used.add(opt["id"])
                results.append((index, q, opt, target))
                log(f"  footage: {q!r} <- {opt['id']}", flush=True)
    results.sort(key=lambda x: x[0])
    results = results[:wanted]
    sources = [Source("video", str(t), label=q, credit=o["credit"]) for _, q, o, t in results]
    records = [{**o, "query": q} for _, q, o, _ in results]
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
