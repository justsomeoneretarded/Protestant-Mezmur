"""Collect lyric candidates for one song from every free source we know."""
import re
from pathlib import Path

from . import web
from .geez import fold, has_ethiopic
from .lyrics_text import extract_blocks, html_to_text, parse_lyrics, script_texts, stanzas_text, title_overlap

LYRICS_DOMAINS = {
    "wikimezmur.org": 3, "mezmuroch.com": 3, "legeta316.com": 3, "mezmurdebter.vercel.app": 3,
    "lrclib.net": 2, "lyricstranslate.com": 2, "genius.com": 1, "musixmatch.com": 1, "blogspot.com": 1,
}
SKIP_DOMAINS = {"youtube.com", "m.youtube.com", "facebook.com", "m.facebook.com", "tiktok.com", "instagram.com",
                "x.com", "twitter.com", "spotify.com", "music.apple.com", "zemare.org"}


def read_correction(folder, message_id):
    """corrections/<id>.txt — optional '---' front matter with title/artist overrides."""
    path = Path(folder) / f"{message_id}.txt"
    if not path.is_file():
        return None, {}
    text = path.read_text(encoding="utf-8")
    meta = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]
    return parse_lyrics(text), meta


def _candidate(source, url, stanzas, title="", trusted=False, extra=None):
    return {"source": source, "url": url, "title": title, "stanzas": stanzas, "trusted": trusted,
            "text": stanzas_text(stanzas), **(extra or {})}


def from_telegram(data):
    out = []
    for item in (data or {}).get("texts", []):
        for stanzas in extract_blocks(item.get("text", ""), min_lines=3):
            out.append(_candidate(f"telegram:{item.get('where', '')}", item.get("link", ""), stanzas,
                                  title=item.get("text", "")[:80]))
    return out


def from_lrclib(record, duration, log=print):
    out, seen = [], set()
    queries = [(record.get("title_latin"), record.get("artist_latin")), (record.get("title_am"), record.get("artist_am")),
               (record.get("title_am"), "")]
    for track, artist in queries:
        if not track:
            continue
        for item in web.lrclib(track, artist or "", duration):
            if item.get("id") in seen:
                continue
            seen.add(item.get("id"))
            text = item.get("plainLyrics") or ""
            if not has_ethiopic(text, 20):
                continue
            stanzas = parse_lyrics(text)
            if stanzas:
                out.append(_candidate("lrclib", f"https://lrclib.net/api/get/{item.get('id')}", stanzas,
                                      title=f"{item.get('trackName')} — {item.get('artistName')}",
                                      extra={"synced": item.get("syncedLyrics"), "lrclib_duration": item.get("duration")}))
    log(f"  lrclib: {len(out)} candidate(s)", flush=True)
    return out


def _queries(record):
    t_am, t_lat = record.get("title_am", ""), record.get("title_latin", "")
    a_am, a_lat = record.get("artist_am", ""), record.get("artist_latin", "")
    qs = []
    if t_am:
        qs += [f"{t_am} {a_am} ግጥም".strip(), f"{t_am} {a_am or a_lat} lyrics".strip(), f"\"{t_am}\" መዝሙር ግጥም"]
    if t_lat:
        qs += [f"{t_lat} {a_lat} lyrics".strip()]
    qs += [f"{t_am or t_lat} wikimezmur", f"{t_am or t_lat} {a_lat} mezmuroch"]
    out = []
    for q in qs:
        q = " ".join(q.split())
        if q and q not in out:
            out.append(q)
    return out


def _page_candidates(url, html, record):
    text = html_to_text(html)
    blocks = extract_blocks(text, min_lines=4)
    for embedded in script_texts(html):
        blocks += extract_blocks(html_to_text(embedded.replace("\n", "<br>")), min_lines=4)
    title_key = record.get("title_am") or record.get("title_latin") or ""
    out = []
    for stanzas in blocks:
        body = stanzas_text(stanzas)
        overlap = title_overlap(title_key, body + " " + text[:3000]) if title_key else 0
        out.append((overlap, len(body), stanzas))
    out.sort(key=lambda x: (-x[0], -x[1]))
    return [s for _, _, s in out[:2]]


def from_web(record, log=print, max_pages=10):
    results, seen = [], set()
    titles = []
    for q in _queries(record):
        for r in web.search(q, max_results=10):
            titles.append(r["title"])
            d = web.domain(r["url"])
            if r["url"] in seen or any(d == s or d.endswith("." + s) for s in SKIP_DOMAINS):
                continue
            seen.add(r["url"])
            rank = max([v for k, v in LYRICS_DOMAINS.items() if d == k or d.endswith("." + k)] or [0])
            title_hit = title_overlap(record.get("title_am") or record.get("title_latin", ""), r["title"] + " " + r["snippet"])
            results.append((rank + 2 * title_hit, r))
    results.sort(key=lambda x: -x[0])
    out = []
    for score, r in results[:max_pages]:
        page = web.get(r["url"])
        if page is None or page.status_code != 200 or "html" not in page.headers.get("content-type", "html"):
            continue
        page.encoding = page.encoding if page.encoding and page.encoding.lower() != "iso-8859-1" else "utf-8"
        for stanzas in _page_candidates(r["url"], page.text, record):
            out.append(_candidate(f"web:{web.domain(r['url'])}", r["url"], stanzas, title=r["title"]))
    log(f"  web: {len(results)} result(s), {len(out)} lyric candidate(s)", flush=True)
    return out, titles


def _anchors(html):
    out = []
    for href, inner in re.findall(r'<a[^>]+href="([^"#]+)"[^>]*>(.*?)</a>', html, re.S | re.I):
        text = " ".join(html_to_text(inner).split())
        if text:
            out.append((href, text))
    return out


def title_match(text, record):
    """How well a link text names this song (0..1). Handles Ge'ez and romanized spellings."""
    from .names import similar
    best = 0.0
    t_am, t_lat = record.get("title_am", ""), record.get("title_latin", "")
    if t_am:
        for run in re.findall(r"[\u1200-\u137F][\u1200-\u137F\s፡]*[\u1200-\u137F]|[\u1200-\u137F]", text):
            run = " ".join(run.replace("፡", " ").split())
            best = max(best, 1.0 if fold(run) == fold(t_am) else similar(run, t_am))
    if t_lat:
        words = re.findall(r"[A-Za-z']+", text)
        n = len(t_lat.split())
        for i in range(0, max(1, len(words) - n + 1)):
            best = max(best, similar(" ".join(words[i:i + n]), t_lat) * 0.97)
    return best


def _follow(page_url, html, record, source, base, artist_check=False, minimum=0.8):
    from .names import similar
    scored = []
    for href, text in _anchors(html):
        if "/lyrics/" not in href and source != "mezmurdebter":
            continue
        score = title_match(text, record)
        if artist_check and record.get("artist_latin") and score >= minimum:
            if not any(similar(record["artist_latin"], " ".join(text.split()[i:i + len(record["artist_latin"].split())])) > 0.7
                       for i in range(len(text.split()))) and fold(record.get("artist_am", "x")) not in fold(text):
                score *= 0.7
        if score >= minimum:
            scored.append((score, href if href.startswith("http") else base + href, text))
    scored.sort(key=lambda x: -x[0])
    out = []
    for score, url, text in scored[:2]:
        page = web.get(url)
        if page is None or page.status_code != 200:
            continue
        page.encoding = "utf-8"
        for stanzas in _page_candidates(url, page.text, record):
            out.append(_candidate(f"site:{source}", url, stanzas, title=text[:100], extra={"link_score": round(score, 2)}))
    return out


def from_sites(record, log=print):
    """Look the song up directly on mezmur lyrics sites (no search engine needed)."""
    out = []
    artist = record.get("artist_latin", "")
    try:
        if artist:
            url = "https://mezmuroch.com/artists/" + "_".join(w[:1].upper() + w[1:] for w in artist.split())
            page = web.get(url)
            if page is not None and page.status_code == 200:
                page.encoding = "utf-8"
                out += _follow(url, page.text, record, "mezmuroch.com", "https://mezmuroch.com")
    except Exception as error:
        log(f"  mezmuroch lookup failed: {error}", flush=True)
    try:
        for q in [record.get("title_am"), record.get("title_latin")]:
            if not q:
                continue
            page = web.get("https://legeta316.com/search", params={"q": q})
            if page is not None and page.status_code == 200:
                page.encoding = "utf-8"
                found = _follow(page.url, page.text, record, "legeta316.com", "https://legeta316.com", artist_check=True)
                out += found
                if found:
                    break
    except Exception as error:
        log(f"  legeta316 lookup failed: {error}", flush=True)
    try:
        if artist:
            slug = re.sub(r"[^a-z0-9]+", "-", artist.lower()).strip("-")
            url = f"https://mezmurdebter.vercel.app/{slug}"
            page = web.get(url)
            if page is not None and page.status_code == 200:
                page.encoding = "utf-8"
                out += _follow(url, page.text, record, "mezmurdebter", "https://mezmurdebter.vercel.app")
    except Exception as error:
        log(f"  mezmurdebter lookup failed: {error}", flush=True)
    log(f"  lyrics sites: {len(out)} candidate(s)", flush=True)
    return out


def youtube_titles(record):
    q = " ".join(p for p in [record.get("title_am") or record.get("title_latin"), record.get("artist_latin") or record.get("artist_am")] if p)
    videos = web.youtube_search(q) if q else []
    return videos


def dedupe(cands):
    out = []
    for c in cands:
        key = fold(c["text"])[:400]
        if not key:
            continue
        if any(fold(o["text"])[:400] == key for o in out):
            continue
        out.append(c)
    return out


def gather(record, message_id, telegram_data, corrections_dir, duration, log=print):
    cands, titles, notes = [], [], {}
    corrected, overrides = read_correction(corrections_dir, message_id)
    if corrected:
        cands.append(_candidate("correction", f"corrections/{message_id}.txt", corrected, trusted=True))
        log("  using your corrected lyrics", flush=True)
    cands += from_telegram(telegram_data)
    try:
        cands += from_lrclib(record, duration, log)
    except Exception as error:
        log(f"  lrclib failed: {error}", flush=True)
    try:
        cands += from_sites(record, log)
    except Exception as error:
        log(f"  lyrics sites failed: {error}", flush=True)
    try:
        web_cands, titles = from_web(record, log)
        cands += web_cands
    except Exception as error:
        log(f"  web search failed: {error}", flush=True)
    return dedupe(cands), titles, overrides, notes
