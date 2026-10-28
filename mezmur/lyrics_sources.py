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


def latin_variants(text, amharic=""):
    """Common Latin spellings of a name: as given, doubled letters single/double, and our own
    romanization of the Amharic ('Agegnehu' / 'Agegnew', 'Addis' / 'Adis', ...)."""
    import re as _re
    from .geez import romanize
    out = []

    def add(v):
        v = " ".join(str(v or "").split())
        if v and v.lower() not in {o.lower() for o in out}:
            out.append(v)

    add(text)
    add(_re.sub(r"([a-z])\1", r"\1", text or "", flags=_re.I))           # Addis -> Adis
    if amharic:
        add(romanize(amharic, title_case=True))
        add(_re.sub(r"(?i)nye|gnie", "gne", romanize(amharic, title_case=True)))
    return out[:4]


def _queries(record):
    t_am, a_am = record.get("title_am", ""), record.get("artist_am", "")
    t_lats = latin_variants(record.get("title_latin", ""), t_am)
    a_lats = latin_variants(record.get("artist_latin", ""), a_am)
    a_lat = a_lats[0] if a_lats else ""
    qs = []
    if t_am:
        qs += [f"{t_am} {a_am} ግጥም", f"{t_am} {a_am or a_lat} lyrics", f"\"{t_am}\" መዝሙር ግጥም"]
    for t in t_lats[:3]:
        qs.append(f"{t} {a_lat} lyrics")
    for a in a_lats[1:2]:
        if t_lats:
            qs.append(f"{t_lats[0]} {a} lyrics")
    qs += [f"{t_am or (t_lats[0] if t_lats else '')} wikimezmur", f"{t_am or (t_lats[0] if t_lats else '')} {a_lat} mezmuroch"]
    out = []
    for q in qs:
        q = " ".join(q.split())
        if q and q not in out:
            out.append(q)
    return out[:9]


COMMON = {"ነው", "ነኝ", "ነህ", "ነሽ", "ነን", "እና", "ላይ", "ውስጥ", "የኔ", "እኔ", "አንተ", "አንቺ", "ሁሉ", "ሆይ", "ግን",
          "እስከ", "ከዚህ", "ይህ", "ያ", "ወደ", "ስለ", "እንደ", "አለ", "የለም", "ዛሬ", "ዛሬም", "አሁን", "ደግሞ", "በጣም"}


SPELLING = {  # how the acoustic model often spells words that lyrics write differently
    "እግዚያብሔር": "እግዚአብሔር", "እግዚብሔር": "እግዚአብሔር", "እግዚያብሄር": "እግዚአብሔር", "እግዚአብሄር": "እግዚአብሔር",
    "እየሱስ": "ኢየሱስ", "ሀሌሉያ": "ሃሌሉያ", "ሐሌሉያ": "ሃሌሉያ", "መንፈስቅዱስ": "መንፈስ ቅዱስ", "አዝኖል": "አዝኗል",
    "ወቶአል": "ወጥቷል", "ሆኖአል": "ሆኗል", "ሆኖል": "ሆኗል", "ቆሞአል": "ቆሟል", "ፀጋ": "ጸጋ",
}


def _spell(word):
    return SPELLING.get(word, word)


def phrases_from_draft(draft, limit=3):
    """Distinctive sung phrases from the automatic transcription: clearly heard word pairs, repeated (the
    chorus) and not made only of common words. Lyrics posts nearly always contain the chorus, even when the
    song's title is wrong or spelled differently."""
    from collections import Counter
    good = [w for w in draft if w.get("probability", 1) >= 0.55 and has_ethiopic(w["word"], 2)]
    words = [_spell(w["word"]) for w in good]
    pairs = Counter(f"{a} {b}" for a, b in zip(words, words[1:]) if a != b)

    def distinctive(p):
        ws = p.split()
        return any(len(w) >= 3 and w not in COMMON for w in ws) and not all(w in COMMON for w in ws)

    ranked = sorted((p for p in pairs if distinctive(p)),
                    key=lambda p: (-pairs[p], -sum(len(w) for w in p.split())))
    out = [p for p in ranked if pairs[p] >= 2][:limit]
    if len(out) < limit:
        for p in ranked:
            if p not in out:
                out.append(p)
            if len(out) >= limit:
                break
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


LYRIC_WORDS = ("lyric", "ግጥም", "lyrics", "with lyrics", "karaoke")


def from_youtube(record, videos=(), log=print, max_videos=15, phrases=(), report=None):
    """Lyric videos of the song: their descriptions and their caption tracks (lyric uploads usually caption
    every line). Videos whose title says 'lyrics'/'ግጥም' are read first."""
    found, seen = [], set()

    def take(vs, origin):
        for v in vs:
            if v.get("video_id") and v["video_id"] not in seen:
                seen.add(v["video_id"])
                found.append({**v, "origin": origin})

    take(videos or [], "title")
    t_am, a_am = record.get("title_am", ""), record.get("artist_am", "")
    t_lats = latin_variants(record.get("title_latin", ""), t_am)
    a_lat = (latin_variants(record.get("artist_latin", ""), a_am) or [""])[0]
    queries = [f"{t_am} {a_am} ግጥም", f"{t_am} {a_am} lyrics"] + [f"{a_lat} {t} lyrics" for t in t_lats[:2]]
    for q in queries:
        q = " ".join(q.split())
        if len(q) >= 4:
            take(web.youtube_search(q), "title")
    for p in phrases:
        take(web.youtube_search(f"{p} ግጥም"), "phrase")

    def is_lyric(v):
        t = v.get("title", "").lower()
        return any(w in t for w in LYRIC_WORDS)

    usable = [v for v in found if title_match(v.get("title", ""), record) >= 0.6
              or (v["origin"] == "phrase" and is_lyric(v))]
    usable.sort(key=lambda v: (not is_lyric(v), -title_match(v.get("title", ""), record)))
    out, n_desc, n_cap = [], 0, 0
    for v in usable[:max_videos]:
        url = f"https://www.youtube.com/watch?v={v['video_id']}"
        page = web.youtube_watch(v["video_id"])
        for stanzas in extract_blocks(page["description"], min_lines=4):
            out.append(_candidate("youtube description", url, stanzas, title=v.get("title", "")[:100]))
            n_desc += 1
        tracks = sorted(page["captions"], key=lambda t: (t["auto"], t["lang"] not in ("am", "am-ET")))
        for t in tracks[:2]:
            if t["auto"]:
                continue  # YouTube's automatic captions of sung Amharic are not lyrics
            text = web.youtube_caption_text(t["url"])
            if has_ethiopic(text, 40):
                stanzas = parse_lyrics(text)
                if stanzas:
                    out.append(_candidate("youtube captions", url, stanzas, title=v.get("title", "")[:100]))
                    n_cap += 1
                    break
    if report is not None:
        report["youtube videos checked"] = min(len(usable), max_videos)
        report["youtube descriptions"] = n_desc
        report["youtube captions"] = n_cap
    log(f"  youtube: {min(len(usable), max_videos)} video(s) checked, {n_desc} description(s), {n_cap} caption track(s)",
        flush=True)
    return out


def from_phrases(record, phrases, channels, log=print, report=None):
    """Search by sung phrases (not the title): public Telegram lyrics channels and the web."""
    out, seen = [], set()
    n_tg = n_web = 0
    for p in phrases:
        for ch in channels:
            for item in web.telegram_public_search(ch, p, limit=4):
                if item["link"] in seen:
                    continue
                seen.add(item["link"])
                for stanzas in extract_blocks(item["text"], min_lines=3):
                    out.append(_candidate(f"telegram:@{ch} (phrase)", item["link"], stanzas))
                    n_tg += 1
        for r in web.search(f"\"{p}\" ግጥም", max_results=8)[:5]:
            if r["url"] in seen or web.domain(r["url"]) in SKIP_DOMAINS:
                continue
            seen.add(r["url"])
            page = web.get(r["url"])
            if page is None or page.status_code != 200:
                continue
            page.encoding = page.encoding if page.encoding and page.encoding.lower() != "iso-8859-1" else "utf-8"
            for stanzas in _page_candidates(r["url"], page.text, record):
                out.append(_candidate(f"web:{web.domain(r['url'])} (phrase)", r["url"], stanzas, title=r["title"]))
                n_web += 1
    if report is not None:
        report["phrases searched"] = list(phrases)
        report["telegram by phrase"] = n_tg
        report["web by phrase"] = n_web
    log(f"  phrase search {phrases}: telegram {n_tg}, web {n_web} candidate(s)", flush=True)
    return out


def from_library(record, draft_text, folder, report=None):
    """Lyrics verified for earlier songs (saved automatically) — re-posts and remakes find them instantly."""
    from .lyrics_text import similarity
    from .names import same_person_name
    folder = Path(folder) if folder else None
    out = []
    if folder and folder.is_dir():
        for path in folder.rglob("*.txt"):
            stanzas, meta = read_correction(path.parent, path.stem)
            if not stanzas:
                continue
            text = stanzas_text(stanzas)
            named = (title_match(" ".join([meta.get("title_am", ""), meta.get("title_latin", "")]), record) >= 0.8
                     and (not record.get("artist_latin") and not record.get("artist_am")
                          or same_person_name(" ".join([meta.get("artist_am", ""), meta.get("artist_latin", "")]),
                                              record.get("artist_latin", ""), record.get("artist_am", ""))))
            if named or (draft_text and similarity(text, draft_text)["score"] >= 0.25):
                out.append(_candidate("library", f"lyrics library: {path.parent.name}/{path.name}", stanzas,
                                      title=meta.get("title_am") or meta.get("title_latin", "")))
    if report is not None:
        report["library"] = len(out)
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


def gather(record, message_id, telegram_data, corrections_dir, duration, log=print, videos=(), draft=(),
           channels=(), library=None):
    """All lyric candidates for one song, plus a report of what each source returned."""
    cands, titles = [], []
    report = {"searched title": " / ".join(x for x in [record.get("title_am"), record.get("title_latin")] if x),
              "searched singer": " / ".join(x for x in [record.get("artist_am"), record.get("artist_latin")] if x)}
    corrected, overrides = read_correction(corrections_dir, message_id)
    if corrected:
        cands.append(_candidate("correction", f"corrections/{message_id}.txt", corrected, trusted=True))
        log("  using your corrected lyrics", flush=True)
    draft_text = " ".join(w["word"] for w in draft)

    def run(name, fn, *a, **k):
        try:
            got = fn(*a, **k)
        except Exception as error:  # one source failing never stops the others
            log(f"  {name} failed: {error.__class__.__name__}: {error}", flush=True)
            got = []
        return got

    tg = from_telegram(telegram_data)
    report["telegram (channel posts, lyrics channels, joined chats)"] = len(tg)
    cands += tg
    cands += run("library", from_library, record, draft_text, library, report=report)
    got = run("lrclib", from_lrclib, record, duration, log)
    report["lrclib"] = len(got)
    cands += got
    got = run("lyrics sites", from_sites, record, log)
    report["lyrics sites (mezmuroch, legeta316, mezmurdebter)"] = len(got)
    cands += got
    phrases = phrases_from_draft(list(draft)) if draft else []
    cands += run("youtube", from_youtube, record, videos, log, phrases=phrases, report=report)
    if phrases:
        cands += run("phrase search", from_phrases, record, phrases, channels, log, report=report)
    try:
        web_cands, titles = from_web(record, log)
    except Exception as error:
        log(f"  web search failed: {error}", flush=True)
        web_cands = []
    report["web search (title)"] = len(web_cands)
    cands += web_cands
    from .lyrics_text import strip_header
    for c in cands:
        if not c["trusted"]:
            c["stanzas"] = strip_header(c["stanzas"], record)
            c["text"] = stanzas_text(c["stanzas"])
    cands = [c for c in cands if c["stanzas"]]
    cands = dedupe(cands)
    report["distinct candidates checked"] = len(cands)
    return cands, titles, overrides, report
