"""Keyless, free web lookups: search engines, YouTube search pages, LRCLIB, Wikimedia."""
import json
import re
import time
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/130.0 Safari/537.36")
_session = None


def session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9,am;q=0.8"})
        _session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))
    return _session


def get(url, timeout=25, max_bytes=3_000_000, **kwargs):
    try:
        r = session().get(url, timeout=timeout, stream=True, **kwargs)
        chunks, size = [], 0
        for chunk in r.iter_content(65536):
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                break
        r._content = b"".join(chunks)
        return r
    except requests.RequestException as error:
        print(f"web: {url[:90]} failed: {error.__class__.__name__}", flush=True)
        return None


def _searxng(query, max_results=10, category="general"):
    """The workflow starts a private SearXNG container (Google/Bing/Brave/Mojeek/... in one call)."""
    import os
    base = os.environ.get("SEARXNG_URL", "").rstrip("/")
    if not base:
        return []
    try:
        r = session().get(f"{base}/search", params={"q": query, "format": "json", "categories": category,
                                                    "language": "all", "safesearch": 0}, timeout=25)
        data = r.json() if r.status_code == 200 else {}
    except (requests.RequestException, ValueError) as error:
        print(f"searxng {query[:50]!r}: {error.__class__.__name__}", flush=True)
        return []
    return data.get("results", [])[:max_results * 2]


def search(query, max_results=10):
    """Text web search: private SearXNG first, then the `ddgs` metasearch library."""
    results, seen = [], set()
    for item in _searxng(query, max_results):
        href = item.get("url") or ""
        if href and href not in seen:
            seen.add(href)
            results.append({"title": item.get("title", ""), "url": href, "snippet": item.get("content", "")})
    # Always also ask ddgs: the engines find different lyrics pages (song 5's came only from ddgs).
    try:
        from ddgs import DDGS
    except ImportError:
        return results
    for backend in ["auto", "yahoo", "duckduckgo"]:
        try:
            for item in DDGS(timeout=20).text(query, max_results=max_results, backend=backend):
                href = item.get("href") or ""
                if href and href not in seen:
                    seen.add(href)
                    results.append({"title": item.get("title", ""), "url": href, "snippet": item.get("body", "")})
        except Exception as error:  # rate limits, no results
            print(f"search[{backend}] {query[:50]!r}: {error.__class__.__name__}", flush=True)
        if len(results) >= 2 * max_results:
            break
        time.sleep(0.8)
    return results


def image_search(query, max_results=12):
    found = [{"url": x.get("img_src"), "thumb": x.get("thumbnail_src"), "title": x.get("title", ""),
              "width": None, "height": None, "source": x.get("url")}
             for x in _searxng(query, max_results, category="images") if x.get("img_src")]
    if len(found) >= 4:
        return found[:max_results]
    try:
        from ddgs import DDGS
        return [{"url": x.get("image"), "thumb": x.get("thumbnail"), "title": x.get("title", ""),
                 "width": x.get("width"), "height": x.get("height"), "source": x.get("url")}
                for x in DDGS(timeout=20).images(query, max_results=max_results)]
    except Exception as error:
        print(f"image search {query[:50]!r}: {error.__class__.__name__}", flush=True)
        return []


def _initial_data(markup):
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", markup)
    return json.loads(m.group(1)) if m else None


def _walk(node, key):
    if isinstance(node, dict):
        if key in node:
            yield node[key]
        for value in node.values():
            yield from _walk(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value, key)


def _runs(obj):
    if not obj:
        return ""
    return obj.get("simpleText") or "".join(r.get("text", "") for r in obj.get("runs", []))


def youtube_search(query, channels=False):
    """Scrape the public YouTube results page (no API key). Returns videos or channels."""
    params = {"search_query": query, "hl": "en"}
    if channels:
        params["sp"] = "EgIQAg=="
    r = get("https://www.youtube.com/results", params=params)
    if r is None or r.status_code != 200:
        return []
    data = _initial_data(r.text)
    if not data:
        return []
    out = []
    if channels:
        for c in _walk(data, "channelRenderer"):
            thumbs = c.get("thumbnail", {}).get("thumbnails") or []
            url = thumbs[-1]["url"] if thumbs else ""
            url = ("https:" + url) if url.startswith("//") else url
            out.append({"channel_id": c.get("channelId"), "title": _runs(c.get("title")),
                        "avatar": re.sub(r"=s\d+-", "=s900-", url),
                        "subscribers": _runs(c.get("videoCountText")) or _runs(c.get("subscriberCountText")),
                        "verified": "BADGE_STYLE_TYPE_VERIFIED" in json.dumps(c.get("ownerBadges", []))})
    else:
        for v in _walk(data, "videoRenderer"):
            out.append({"video_id": v.get("videoId"), "title": _runs(v.get("title")),
                        "channel": _runs(v.get("ownerText")),
                        "channel_id": (v.get("ownerText", {}).get("runs") or [{}])[0]
                        .get("navigationEndpoint", {}).get("browseEndpoint", {}).get("browseId"),
                        "snippet": " ".join(_runs(s.get("snippetText")) for s in v.get("detailedMetadataSnippets", [])),
                        "views": _runs(v.get("viewCountText")), "length": _runs(v.get("lengthText")),
                        "verified": "VERIFIED" in json.dumps(v.get("ownerBadges", []))})
    return out


def youtube_watch(video_id):
    """Description and caption tracks of a public video (one page request)."""
    r = get("https://www.youtube.com/watch", params={"v": video_id, "hl": "en"}, max_bytes=4_000_000)
    if r is None or r.status_code != 200:
        return {"description": "", "captions": []}
    r.encoding = "utf-8"
    player = {}
    m = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\});\s*(?:var |</script>)", r.text, re.S)
    if m:
        try:
            player = json.loads(m.group(1))
        except ValueError:
            player = {}
    description = (player.get("videoDetails") or {}).get("shortDescription", "") or ""
    if not description:
        m = re.search(r'"shortDescription":"((?:[^"\\]|\\.)*)"', r.text)
        if m:
            try:
                description = json.loads(f'"{m.group(1)}"')
            except ValueError:
                description = ""
    tracks = (((player.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}).get("captionTracks")) or []
    captions = [{"url": t.get("baseUrl", ""), "lang": t.get("languageCode", ""), "auto": t.get("kind") == "asr",
                 "name": _runs(t.get("name"))} for t in tracks if t.get("baseUrl")]
    return {"description": description, "captions": captions}


def youtube_description(video_id):
    return youtube_watch(video_id)["description"]


def youtube_caption_text(url):
    """Text of one caption track, one caption per line (lyric videos usually caption one line at a time)."""
    import html as _html
    r = get(url, timeout=20)
    if r is None or r.status_code != 200 or not r.content:
        return ""
    r.encoding = "utf-8"
    parts = re.findall(r"<text[^>]*>(.*?)</text>", r.text, re.S) or re.findall(r"<p [^>]*>(.*?)</p>", r.text, re.S)
    lines = []
    for part in parts:
        text = _html.unescape(re.sub(r"<[^>]+>", "", _html.unescape(part))).strip()
        for line in text.splitlines():
            if line.strip():
                lines.append(line.strip())
    return "\n".join(lines)


def telegram_public_search(channel, query, limit=8):
    """Search a public Telegram channel through its web preview (t.me/s/...) — no login needed."""
    import html as _html
    r = get(f"https://t.me/s/{channel}", params={"q": query}, timeout=20)
    if r is None or r.status_code != 200:
        return []
    r.encoding = "utf-8"
    out = []
    for post, body in re.findall(r'data-post="([^"]+)".*?<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
                                 r.text, re.S)[:limit]:
        text = re.sub(r"<br\s*/?>", "\n", body)
        text = _html.unescape(re.sub(r"<[^>]+>", "", text))
        out.append({"post": post, "text": text, "link": f"https://t.me/{post}"})
    return out


def youtube_thumbnail(video_id):
    for name in ["maxresdefault", "sddefault", "hqdefault"]:
        r = get(f"https://i.ytimg.com/vi/{video_id}/{name}.jpg")
        if r is not None and r.status_code == 200 and len(r.content) > 5000:
            return r.content
    return None


def lrclib(track, artist="", duration=None):
    """LRCLIB public lyrics database (free, no key). Tries field search, then free text."""
    headers = {"Lrclib-Client": "protestant-mezmur (github.com/justsomeoneretarded/Protestant-Mezmur)"}
    attempts = [{"track_name": track, "artist_name": artist}] if artist else []
    attempts.append({"q": f"{track} {artist}".strip()})
    items = []
    for params in attempts:
        r = get("https://lrclib.net/api/search", params=params, headers=headers)
        try:
            data = r.json() if r is not None and r.status_code == 200 else []
        except ValueError:
            data = []
        if isinstance(data, list) and data:
            items = data
            break
    if duration:
        items.sort(key=lambda x: abs((x.get("duration") or 0) - duration))
    return items[:6]


def commons_images(query, limit=8):
    from .backgrounds import BOT_HEADERS
    r = get("https://commons.wikimedia.org/w/api.php", headers=BOT_HEADERS, timeout=20, params={
        "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
        "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1600,
        "format": "json"})
    if r is None or r.status_code != 200:
        return []
    pages = (r.json().get("query") or {}).get("pages", {})
    out = []
    for page in pages.values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        out.append({"url": info.get("thumburl") or info.get("url"), "title": page.get("title", ""),
                    "license": (meta.get("LicenseShortName") or {}).get("value", ""),
                    "artist": re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value", "")),
                    "page": info.get("descriptionurl")})
    return out


def domain(url):
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
