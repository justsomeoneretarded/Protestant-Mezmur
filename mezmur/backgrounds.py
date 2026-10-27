"""Background footage: themes from the lyrics, free stock sources, local fallbacks."""
import os
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

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


CALM = ["waterfall", "starry night sky", "milky way timelapse", "calm lake reflection", "clouds timelapse",
        "ocean waves", "mountain mist", "sunrise clouds", "waterfall forest", "night sky stars"]


def calm_themes(limit=4, seed=0):
    """Neat, slow subjects only (waterfalls, sky, space, calm water); a different order per song."""
    rnd = random.Random(seed)
    out = CALM[:]
    rnd.shuffle(out)
    return out[:limit]


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
        "gsrlimit": 20, "prop": "videoinfo", "viprop": "url|size|derivatives|extmetadata|mediatype", "format": "json"},
        headers=BOT_HEADERS, timeout=20)
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
            continue  # free licences only; CC BY/BY-SA ones are credited in the description
        artist = (meta.get("Artist") or {}).get("value", "")
        import re
        artist = re.sub(r"<[^>]+>", "", artist).strip()[:80]
        out.append({"id": vid, "url": url, "credit": f"{page.get('title', '').replace('File:', '')} — {artist or 'Wikimedia Commons'} ({lic})",
                    "page": info.get("descriptionurl")})
    return out


BOT_HEADERS = {"User-Agent": "protestant-mezmur/0.2 (https://github.com/justsomeoneretarded/Protestant-Mezmur; lyric videos)",
               "Api-User-Agent": "protestant-mezmur/0.2 (https://github.com/justsomeoneretarded/Protestant-Mezmur)"}

# NASA's library is mostly sky and space: these suit worship themes (heavens, light, glory).
NASA_QUERIES = ["earth from space clouds", "sunrise over earth", "aurora from space", "nebula", "milky way",
                "earth night lights", "clouds over ocean from space", "galaxy hubble"]
NASA_SKIP = ("interview", "briefing", "conference", "press", "launch", "rocket", "astronaut", "spacewalk",
             "podcast", "talk", "lecture", "news", "update", "anniversary", "this week")


def _nasa(query, used):
    """NASA Image and Video Library: public domain, no key."""
    r = web.get("https://images-api.nasa.gov/search", params={"q": query, "media_type": "video"}, timeout=20)
    if r is None or r.status_code != 200:
        return []
    out = []
    for item in (r.json().get("collection") or {}).get("items", [])[:25]:
        data = (item.get("data") or [{}])[0]
        vid = f"nasa:{data.get('nasa_id')}"
        title = (data.get("title") or "").lower()
        if vid in used or any(w in title for w in NASA_SKIP):
            continue
        out.append({"id": vid, "collection": item.get("href"), "credit": f"{data.get('title', 'NASA video')} — NASA (public domain)",
                    "page": f"https://images.nasa.gov/details/{data.get('nasa_id')}"})
        if len(out) >= 6:
            break
    return out


def _nasa_file(collection_url):
    r = web.get(collection_url, timeout=20)
    try:
        files = r.json() if r is not None and r.status_code == 200 else []
    except ValueError:
        return None
    mp4 = [f for f in files if isinstance(f, str) and f.lower().endswith(".mp4")]
    for pref in ("~medium.mp4", "~large.mp4", "~orig.mp4", "~mobile.mp4"):
        for f in mp4:
            if f.endswith(pref):
                return f.replace("http://", "https://")
    return mp4[0].replace("http://", "https://") if mp4 else None


MIXKIT_TOPICS = [("waterfall", "waterfall"), ("milky", "stars"), ("mist", "fog"), ("lake", "lake"), ("sunrise", "sunrise"), ("sunset", "sunset"), ("cloud", "clouds"), ("sky", "sky"), ("mountain", "mountain"),
                 ("ocean", "ocean"), ("wave", "ocean"), ("river", "river"), ("water", "water"), ("rain", "rain"),
                 ("star", "stars"), ("night", "night"), ("candle", "candle"), ("flame", "fire"), ("flower", "flowers"),
                 ("forest", "forest"), ("tree", "forest"), ("desert", "desert"), ("light", "light"), ("bird", "birds"),
                 ("dove", "birds"), ("eagle", "birds"), ("lake", "lake"), ("road", "road"), ("hands", "hands")]


def _mixkit(query, used):
    """Mixkit free stock videos (free licence incl. YouTube); read from its public topic pages."""
    import re as _re
    topic = next((slug for key, slug in MIXKIT_TOPICS if key in query), "nature")
    r = web.get(f"https://mixkit.co/free-stock-video/{topic}/", timeout=20)
    if r is None or r.status_code != 200:
        return []
    urls = []
    for u in _re.findall(r"https://assets\.mixkit\.co/[^\"'\s>]+?\.mp4", r.text):
        if u not in urls:
            urls.append(u)
    # Prefer HD renditions; previews are tiny.
    hd = [u for u in urls if any(k in u for k in ("-1080", "-720", "large", "hd"))] or urls
    out = []
    for u in hd:
        vid = "mixkit:" + u.rsplit("/", 1)[-1].split(".")[0]
        if vid in used:
            continue
        out.append({"id": vid, "url": u, "credit": "Video from Mixkit (mixkit.co, free licence)", "page": r.url})
        if len(out) >= 6:
            break
    return out


def _probe_seconds(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def _overlay_reason(target):
    """Detect things laid over footage: watermarks (logos or small corner text) and player bars.

    An overlay stays perfectly still while the picture under it moves, and sits near the edges.
    Returns a reason string, or None when the clip is clean.
    """
    cap = cv2.VideoCapture(str(target))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for k in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (0.05 + 0.09 * k)))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.resize(f, (480, 270), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(frames) < 6:
        return None
    H, W = 270, 480
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    raw_motion = gray.std(0)
    motion = cv2.GaussianBlur(raw_motion, (0, 0), 2)
    edges = np.stack([cv2.Canny(g.astype(np.uint8), 60, 150) > 0 for g in gray]).mean(0)
    steady = edges >= 0.9
    # Player bar / timeline: a straight line, still in every frame, across the top or bottom.
    for r in list(range(0, int(H * 0.08))) + list(range(int(H * 0.88), H)):
        if steady[max(0, r - 1):r + 2].any(0).mean() > 0.5:
            return "player bar or frame line"
    # Logo watermark: a still, edged blob near the edges while the picture around it moves.
    zone = np.zeros((H, W), bool)
    zone[:int(H * 0.22), :] = zone[-int(H * 0.22):, :] = True
    zone[:, :int(W * 0.22)] = zone[:, -int(W * 0.22):] = True
    blobs = cv2.dilate((steady & zone).astype(np.uint8), np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blobs)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 40 or w < 8 or h < 5 or w > W * 0.45 or h > H * 0.3:
            continue
        inside = labels == i
        ring = cv2.dilate(inside.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~cv2.dilate(
            inside.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        strokes = inside & steady
        ring_motion = motion[ring].mean()
        if strokes.sum() >= 20 and ring_motion > 6.0 and np.median(raw_motion[strokes]) < 0.6 * ring_motion:
            return "watermark/logo"
    # Small watermark text in a corner (the main text check ignores tiny text).
    from .textdetect import text_boxes
    hits = 0
    for f in frames[1::4]:
        fh, fw = f.shape[:2]
        for bx, by, bw, bh in text_boxes(f):
            corner = (bx < fw * 0.3 or bx + bw > fw * 0.7) and (by < fh * 0.2 or by + bh > fh * 0.8)
            if corner and bh >= fh * 0.012:
                hits += 1
                break
    if hits >= 2:
        return "watermark text"
    return None


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
    reason = _overlay_reason(target)
    if reason:
        log(f"  footage rejected ({reason}): {target.name}", flush=True)
        return False
    return True


def _prepare(url, target, seconds=40, log=print):
    raw = target.with_suffix(".src")
    r = web.get(url, timeout=60, max_bytes=80_000_000)
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


def footage(lyrics_text, folder, used=(), wanted=6, seed=0, log=print, budget=300):
    """Find, download and normalize free clips: NASA, then Wikimedia Commons, then Mixkit
    (Pexels/Pixabay first when keys exist). Stops after ``budget`` seconds whatever happens."""
    import time as _time
    from concurrent.futures import ThreadPoolExecutor
    deadline = _time.time() + budget
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    used = set(used)
    pexels, pixabay = os.environ.get("PEXELS_API_KEY", "").strip(), os.environ.get("PIXABAY_API_KEY", "").strip()
    rnd = random.Random(seed)
    plan = []  # (label, [options]) — options may need a NASA file lookup

    def search(label, fn, *args):
        if _time.time() > deadline:
            return []
        try:
            return fn(*args)
        except Exception as error:  # noqa: BLE001 - one bad source must not stop the others
            log(f"  footage search {label!r} failed: {error.__class__.__name__}", flush=True)
            return []

    themes = calm_themes(limit=4, seed=seed)
    if pexels or pixabay:
        for q in themes:
            opts = (search(q, _pexels, q, pexels, used) if pexels else []) + \
                   (search(q, _pixabay, q, pixabay, used) if pixabay else [])
            if opts:
                plan.append((q, opts[:3]))
    nasa_qs = NASA_QUERIES[:]
    rnd.shuffle(nasa_qs)
    for q in nasa_qs[:3]:                      # 1. NASA (sky / space)
        opts = search(q, _nasa, q, used)
        rnd.shuffle(opts)
        if opts:
            plan.append((q, opts[:3]))
    for q in themes:                           # 2. Wikimedia, 3. Mixkit (nature themes)
        if len(plan) >= wanted + 3 or _time.time() > deadline:
            break
        opts = search(q, _commons, q, used)
        if len(opts) < 2:
            opts += search(q, _mixkit, q, used)
        rnd.shuffle(opts)
        if opts:
            plan.append((q, opts[:3]))
    log(f"  footage plan: {len(plan)} theme(s) with candidates", flush=True)

    def attempt(job):
        index, (q, options) = job
        for k, opt in enumerate(options):
            if _time.time() > deadline:
                break
            if opt["id"] in used:
                continue
            url = opt.get("url") or (_nasa_file(opt["collection"]) if opt.get("collection") else None)
            target = folder / f"clip{index}_{k}.mp4"
            if url and _prepare(url, target, log=log):
                return index, q, {**opt, "url": url}, target
        return index, q, None, None

    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for index, q, opt, target in pool.map(attempt, list(enumerate(plan))[:wanted + 2]):
            if opt is not None and opt["id"] not in used:
                used.add(opt["id"])
                results.append((index, q, opt, target))
                log(f"  footage: {q!r} <- {opt['id']}", flush=True)
    results.sort(key=lambda x: x[0])
    # Mix sources: alternate sky/space and nature clips rather than grouping them.
    results = results[:wanted]
    sources = [Source("video", str(t), label=q, credit=o["credit"]) for _, q, o, t in results]
    records = [{k: v for k, v in o.items() if k != "collection"} | {"query": q} for _, q, o, _ in results]
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


# ------------------------------------------------------------------------------------------------
# One calm nature photograph per song (Wikimedia Commons "Quality images": sharp, well composed).
# Nothing man-made, no people, no space imagery, no graphics, no text.
# ------------------------------------------------------------------------------------------------
NATURE_QUERIES = ["waterfall", "mountain lake reflection", "misty mountains", "misty forest", "lake sunrise",
                  "mountains sunrise clouds", "forest stream", "river valley", "ocean waves", "sea sunset",
                  "sunset sky clouds", "clouds sky", "milky way", "starry night sky", "city skyline night",
                  "city skyline sunset", "mountain road landscape", "canyon", "desert dunes", "snow mountains"]
DAY_QUERIES = ["waterfall sunny", "blue sky clouds", "green hills", "mountain lake summer", "beach sunny",
               "meadow flowers", "forest sunlight", "snow mountains blue sky", "river valley summer", "rolling hills",
               "lake reflection daytime", "alpine meadow", "turquoise sea", "palm beach", "sunflower field",
               "green valley mountains", "cliffs sea sunny", "spring landscape"]


def day_look(seed):
    """About 7 songs in 10 get a bright daytime picture; the rest a calmer evening one."""
    return int(seed) % 10 < 7


def brightness(img):
    """Mean brightness (0-255) of the band where the lyrics are drawn."""
    if img is None:
        return 128.0
    h = img.shape[0]
    band = cv2.cvtColor(img[int(h * 0.42):int(h * 0.80)], cv2.COLOR_BGR2HSV)[..., 2]
    return float(band.mean())


# What must not be *on* the picture: overlays, graphics, people. Scenery itself may be anything.
BANNED = ("flag", " sign ", "signs", "signpost", "map ", " maps", "diagram", "arrow", "text", "logo", "poster",
          "caption", "label", "advert", "billboard", "graffiti", "screenshot", "chart", "graph", "infographic",
          "people", "person", "portrait", "selfie", "tourist", "crowd", "man ", "woman", "child", "girl", "boy",
          "rendering", "illustration", "drawing", "painting", "artwork", "cgi", "collage", "montage", "composite")


def _banned(text):
    t = f" {text.lower()} "
    return next((b for b in BANNED if b in t), None)


def _nature_candidates(query, used):
    r = web.get("https://commons.wikimedia.org/w/api.php", headers=BOT_HEADERS, timeout=20, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6, "gsrlimit": 30,
        "gsrsearch": f'incategory:"Quality_images" {query} filetype:bitmap',
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 2560})
    if r is None or r.status_code != 200:
        return []
    out = []
    for page in ((r.json().get("query") or {}).get("pages") or {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        pid = f"commons-img:{page.get('pageid')}"
        w, h = info.get("width") or 0, info.get("height") or 0
        if pid in used or w < 1920 or not 1.3 <= w / max(1, h) <= 2.2:
            continue
        meta = info.get("extmetadata") or {}
        lic = (meta.get("LicenseShortName") or {}).get("value", "")
        if not any(x in lic for x in ["CC0", "Public domain", "CC BY", "CC-BY"]):
            continue
        import re as _re
        words = " ".join([page.get("title", ""), (meta.get("Categories") or {}).get("value", "").replace("|", " "),
                          _re.sub(r"<[^>]+>", " ", (meta.get("ImageDescription") or {}).get("value", ""))[:600]])
        if _banned(words):
            continue
        artist = _re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "")).strip()[:80]
        out.append({"id": pid, "url": info.get("thumburl") or info.get("url"),
                    "credit": f"{page.get('title', '').replace('File:', '')} — {artist or 'Wikimedia Commons'} ({lic})",
                    "page": info.get("descriptionurl"), "query": query})
    return out


def picture_problem(img, day=False):
    """Reasons a picture is not a clean background (None when it is fine): nothing may be drawn or written on it."""
    from .textdetect import text_score
    if img is None:
        return "unreadable"
    small = cv2.resize(img, (960, int(960 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = float(hsv[..., 2].mean())
    if not (120 if day else 55) <= v <= 215:
        return f"{'not bright enough for a daytime look' if day else 'too dark or too bright'} ({v:.0f})"
    if text_score(img) >= 0.004:
        return "text in the picture"
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    local_std = cv2.blur(gray.astype(np.float32) ** 2, (5, 5)) - cv2.blur(gray.astype(np.float32), (5, 5)) ** 2
    flat = local_std < 4
    hue = hsv[..., 0]
    graphic = flat & (hsv[..., 1] > 210) & ~((hue >= 90) & (hue <= 130))  # flat, vivid, not sky blue (flags, signs)
    if graphic.mean() > 0.04:
        return "graphic / flat colour areas"
    return None


def nature_still(folder, used=(), seed=0, log=print, budget=150, slot=0, slots=1, day=None):
    """Return (Source, record) for one checked photograph never used before, or (None, None).

    Songs of one batch render at the same time, so each takes only candidates whose id falls in its
    own slot (id mod slots); together with the remembered ``used`` ids no two videos share a picture."""
    import time as _time
    deadline = _time.time() + budget
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    used = set(used)
    rnd = random.Random(seed)
    day = day_look(seed) if day is None else day
    queries = (DAY_QUERIES if day else NATURE_QUERIES)[:]
    rnd.shuffle(queries)
    tried = 0
    for q in queries:
        if _time.time() > deadline:
            break
        try:
            cands = _nature_candidates(q, used)
        except Exception as error:  # noqa: BLE001
            log(f"  nature photo search {q!r} failed: {error.__class__.__name__}", flush=True)
            continue
        cands = [c for c in cands if int(c["id"].split(":")[1]) % max(1, slots) == slot]
        rnd.shuffle(cands)
        for c in cands[:4]:
            if _time.time() > deadline or tried >= 12:
                break
            tried += 1
            r = web.get(c["url"], headers=BOT_HEADERS, timeout=40, max_bytes=25_000_000)
            if r is None or r.status_code != 200:
                continue
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            problem = picture_problem(img, day=day)
            if problem:
                log(f"  background rejected ({problem}): {c['credit'][:70]}", flush=True)
                continue
            path = folder / "nature.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
            log(f"  background: {c['credit'][:90]}", flush=True)
            return Source("image", str(path), label="nature", credit=c["credit"]), c
    return None, None


def fallback_scene(folder, used=(), seed=0):
    """The four bundled pictures (mountains, skyline, road, desert), each used for one video only."""
    scenes = local_scenes(folder)
    for i in sorted(range(len(scenes)), key=lambda i: (i + seed) % max(1, len(scenes))):
        sid = f"bundled:{i}"
        if i < len(scenes) and sid not in used:
            return scenes[i], {"id": sid, "credit": "", "query": "bundled"}
    return None, None
