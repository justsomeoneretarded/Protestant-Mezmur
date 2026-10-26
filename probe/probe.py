"""One-off data probe. Writes small JSON/images into out/ for inspection."""
import asyncio, json, os, re, subprocess, time, traceback, urllib.parse
from pathlib import Path

import requests

OUT = Path("out"); OUT.mkdir(exist_ok=True)
ETH = re.compile(r"[ሀ-᎟ⶀ-⷟꬀-꬯]")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9,am;q=0.8"}
report = {}


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def eth_lines(text):
    return sum(1 for l in (text or "").splitlines() if len(ETH.findall(l)) >= 4)


async def telegram():
    from telethon import TelegramClient
    from telethon.sessions import StringSession
    c = TelegramClient(StringSession(os.environ["TELEGRAM_SESSION"]), int(os.environ["TELEGRAM_API_ID"]), os.environ["TELEGRAM_API_HASH"])
    await c.connect()
    ent = await c.get_entity(os.environ.get("TELEGRAM_CHANNEL") or "@amharicprotestantmezmurss")
    info = {"id": ent.id, "title": getattr(ent, "title", ""), "username": getattr(ent, "username", "")}
    rows = []
    async def dump(it, tag):
        async for m in it:
            r = {"tag": tag, "id": m.id, "date": str(m.date), "grouped": m.grouped_id, "text": (m.raw_text or "")[:1500],
                 "eth_lines": eth_lines(m.raw_text), "photo": bool(m.photo), "media": type(m.media).__name__ if m.media else None}
            if m.document:
                d = m.document
                r["mime"] = d.mime_type; r["size"] = d.size; r["thumbs"] = len(d.thumbs or [])
                for a in d.attributes:
                    n = type(a).__name__
                    if n == "DocumentAttributeAudio":
                        r.update(title=a.title, performer=a.performer, duration=a.duration, voice=a.voice)
                    if n == "DocumentAttributeFilename":
                        r["filename"] = a.file_name
            if m.forward:
                r["fwd_from"] = str(getattr(m.forward, "chat_id", None) or getattr(m.fwd_from, "from_name", ""))
            rows.append(r)
    await dump(c.iter_messages(ent, limit=80, reverse=True), "oldest")
    await dump(c.iter_messages(ent, limit=80), "newest")
    info["total"] = (await c.get_messages(ent, limit=0)).total
    save("telegram_messages.json", rows); save("telegram_channel.json", info)
    # Download some media samples: 3 audio thumbs + cover art, 3 photos
    audios = [r for r in rows if r.get("mime", "").startswith("audio")][:4]
    photos = [r for r in rows if r["photo"]][:4]
    for r in audios:
        m = await c.get_messages(ent, ids=r["id"])
        try:
            p = await c.download_media(m, file=str(OUT / f"thumb-{r['id']}.jpg"), thumb=-1)
            r["thumb_saved"] = bool(p)
        except Exception as e:
            r["thumb_err"] = repr(e)
        if r["size"] < 15_000_000:
            a = await c.download_media(m, file=f"/tmp/a-{r['id']}")
            pr = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", a], capture_output=True, text=True)
            r["ffprobe"] = json.loads(pr.stdout or "{}")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-an", "-frames:v", "1", str(OUT / f"cover-{r['id']}.jpg")])
            r["audio_path"] = a
    for r in photos:
        m = await c.get_messages(ent, ids=r["id"])
        await c.download_media(m, file=str(OUT / f"photo-{r['id']}.jpg"), thumb=-1)
    save("telegram_samples.json", audios)
    # Search tests
    found = {}
    for q in ["ግጥም", "lyrics", "አዲስዓለም"]:
        try:
            ms = await c.get_messages(ent, search=q, limit=5)
            found[q] = [{"id": m.id, "text": (m.raw_text or "")[:200]} for m in ms]
        except Exception as e:
            found[q] = repr(e)
    try:
        from telethon.tl.functions.contacts import SearchRequest
        res = await c(SearchRequest(q="mezmur lyrics", limit=20))
        found["contacts.search"] = [{"id": ch.id, "title": getattr(ch, "title", ""), "username": getattr(ch, "username", ""), "broadcast": getattr(ch, "broadcast", None)} for ch in res.chats]
        res = await c(SearchRequest(q="መዝሙር ግጥም", limit=20))
        found["contacts.search.am"] = [{"id": ch.id, "title": getattr(ch, "title", ""), "username": getattr(ch, "username", ""), "broadcast": getattr(ch, "broadcast", None)} for ch in res.chats]
    except Exception as e:
        found["contacts.search"] = repr(e)
    try:
        from telethon.tl.functions.channels import SearchPostsRequest
        res = await c(SearchPostsRequest(query="ሲወድ ጌታ", offset_rate=0, offset_peer="me", offset_id=0, limit=10))
        found["searchPosts"] = [(getattr(m, "message", "") or "")[:200] for m in res.messages]
    except Exception as e:
        found["searchPosts"] = repr(e)[:300]
    save("telegram_search.json", found)
    await c.disconnect()
    return audios


def http(name, fn):
    t = time.time()
    try:
        report[name] = fn()
    except Exception as e:
        report[name] = {"error": repr(e)[:400]}
    report[name] = {"t": round(time.time() - t, 1), "r": report[name]}


def get(url, **kw):
    r = requests.get(url, headers=UA, timeout=25, **kw)
    return r


def ddg():
    r = requests.post("https://html.duckduckgo.com/html/", data={"q": "ሲወድ ጌታ አዲስዓለም አሰፋ lyrics"}, headers=UA, timeout=25)
    links = re.findall(r'class="result__a" href="([^"]+)"', r.text)
    return {"status": r.status_code, "n": len(links), "links": [urllib.parse.unquote(l)[:200] for l in links[:10]]}


def ddgs_pkg():
    from ddgs import DDGS
    out = {}
    for backend in ["auto", "bing", "brave", "google", "mojeek", "yahoo", "duckduckgo", "wikipedia", "yandex"]:
        try:
            res = DDGS().text("ሲወድ ጌታ አዲስዓለም አሰፋ ግጥም", max_results=8, backend=backend)
            out[backend] = [(x.get("title", "")[:80], x.get("href", "")[:120]) for x in res]
        except Exception as e:
            out[backend] = repr(e)[:200]
    try:
        out["images"] = [(x.get("title", "")[:60], x.get("image", "")[:150], x.get("width"), x.get("height")) for x in DDGS().images("Addisalem Assefa singer", max_results=8)]
    except Exception as e:
        out["images"] = repr(e)[:200]
    try:
        out["videos"] = [(x.get("title", "")[:80], x.get("content", "")[:120], (x.get("description") or "")[:200]) for x in DDGS().videos("ሲወድ ጌታ አዲስዓለም አሰፋ", max_results=8)]
    except Exception as e:
        out["videos"] = repr(e)[:200]
    return out


def yt_search():
    r = get("https://www.youtube.com/results", params={"search_query": "ሲወድ ጌታ አዲስዓለም አሰፋ lyrics", "hl": "en"})
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", r.text)
    vids = []
    if m:
        data = json.loads(m.group(1))
        def walk(o):
            if isinstance(o, dict):
                if "videoRenderer" in o:
                    v = o["videoRenderer"]
                    vids.append({"id": v.get("videoId"), "title": "".join(x.get("text", "") for x in v.get("title", {}).get("runs", [])),
                                 "owner": "".join(x.get("text", "") for x in v.get("ownerText", {}).get("runs", [])),
                                 "snippet": "".join(x.get("text", "") for s in v.get("detailedMetadataSnippets", []) for x in s.get("snippetText", {}).get("runs", []))})
                for x in o.values(): walk(x)
            elif isinstance(o, list):
                for x in o: walk(x)
        walk(data)
    return {"status": r.status_code, "len": len(r.text), "has_initial": bool(m), "videos": vids[:10]}


def yt_watch():
    r = get("https://www.youtube.com/watch", params={"v": "sMX8svAk9aM", "hl": "en"})
    m = re.search(r"var ytInitialPlayerResponse = (\{.*?\});(?:var|</script>)", r.text)
    out = {"status": r.status_code, "len": len(r.text), "has_player": bool(m), "bot": "confirm you" in r.text}
    if m:
        p = json.loads(m.group(1))
        vd = p.get("videoDetails", {})
        out.update(title=vd.get("title"), author=vd.get("author"), channelId=vd.get("channelId"), desc=(vd.get("shortDescription") or "")[:3000],
                   playability=p.get("playabilityStatus", {}).get("status"),
                   captions=[t.get("languageCode") for t in p.get("captions", {}).get("playerCaptionsTracklistRenderer", {}).get("captionTracks", [])])
    return out


def yt_channel_avatar():
    r = get("https://www.youtube.com/results", params={"search_query": "Addisalem Assefa", "sp": "EgIQAg%3D%3D"})
    m = re.search(r"var ytInitialData = (\{.*?\});</script>", r.text)
    chans = []
    if m:
        def walk(o):
            if isinstance(o, dict):
                if "channelRenderer" in o:
                    v = o["channelRenderer"]
                    chans.append({"id": v.get("channelId"), "title": v.get("title", {}).get("simpleText"),
                                  "thumb": (v.get("thumbnail", {}).get("thumbnails") or [{}])[-1].get("url"),
                                  "subs": v.get("videoCountText", {}).get("simpleText") or v.get("subscriberCountText", {}).get("simpleText")})
                for x in o.values(): walk(x)
            elif isinstance(o, list):
                for x in o: walk(x)
        walk(json.loads(m.group(1)))
    for c in chans[:1]:
        if c["thumb"]:
            u = c["thumb"] if c["thumb"].startswith("http") else "https:" + c["thumb"]
            u = re.sub(r"=s\d+", "=s800", u)
            img = get(u)
            (OUT / "yt-avatar.jpg").write_bytes(img.content)
            c["dl"] = img.status_code
    t = get("https://i.ytimg.com/vi/sMX8svAk9aM/maxresdefault.jpg")
    (OUT / "yt-thumb.jpg").write_bytes(t.content)
    return {"status": r.status_code, "channels": chans[:5], "thumb": t.status_code}


def commons():
    r = get("https://commons.wikimedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": "Tesfaye Gabiso", "srnamespace": 6, "format": "json"})
    return r.json()


def wiki():
    r = get("https://en.wikipedia.org/w/api.php", params={"action": "query", "titles": "Tesfaye Gabiso", "prop": "pageimages", "piprop": "original", "format": "json"})
    return r.json()


def lrclib():
    r = get("https://lrclib.net/api/search", params={"q": "Addisalem Assefa"})
    return {"status": r.status_code, "n": len(r.json()), "items": [(x["trackName"], x["artistName"], bool(x.get("syncedLyrics"))) for x in r.json()[:10]]}


def pexels():
    r = get("https://api.pexels.com/videos/search?query=sunrise")
    return {"status": r.status_code}


def pixabay():
    r = get("https://pixabay.com/api/videos/?q=sunrise")
    return {"status": r.status_code, "body": r.text[:200]}


def invidious():
    out = {}
    for inst in ["https://inv.nadeko.net", "https://yewtu.be", "https://invidious.nerdvpn.de", "https://iv.ggtyler.dev"]:
        try:
            r = get(inst + "/api/v1/search", params={"q": "Addisalem Assefa siwed geta"})
            out[inst] = [r.status_code, r.text[:150]]
        except Exception as e:
            out[inst] = repr(e)[:150]
    return out


def zemare():
    out = {}
    for u in ["https://zemare.org", "https://zemare.org/robots.txt"]:
        try:
            r = get(u); out[u] = [r.status_code, r.text[:600]]
        except Exception as e:
            out[u] = repr(e)[:200]
    return out


def bing():
    r = get("https://www.bing.com/search", params={"q": "ሲወድ ጌታ አዲስዓለም አሰፋ ግጥም"})
    links = re.findall(r'<h2><a href="(http[^"]+)"', r.text)
    return {"status": r.status_code, "links": links[:10]}


def vocab():
    from transformers import AutoProcessor, AutoConfig
    p = AutoProcessor.from_pretrained("badrex/Ethio-ASR-amharic", cache_dir=".model-cache/ethio-asr")
    tok = p.tokenizer
    v = tok.get_vocab()
    cfg = AutoConfig.from_pretrained("badrex/Ethio-ASR-amharic", cache_dir=".model-cache/ethio-asr")
    return {"size": len(v), "pad": tok.pad_token, "pad_id": tok.pad_token_id, "word_delim": getattr(tok, "word_delimiter_token", None),
            "unk": tok.unk_token, "tokens": sorted(v, key=v.get)[:400], "class": type(tok).__name__,
            "fe": type(p.feature_extractor).__name__, "arch": cfg.architectures, "cfg": {k: getattr(cfg, k, None) for k in ["num_adapter_layers", "adapter_stride", "add_adapter", "conv_stride", "feature_projection_input_dim"]}}


async def main():
    try:
        audios = await telegram()
    except Exception:
        report["telegram_error"] = traceback.format_exc()[-2000:]
    for name, fn in [("ddg_html", ddg), ("ddgs", ddgs_pkg), ("yt_search", yt_search), ("yt_watch", yt_watch), ("yt_channel", yt_channel_avatar),
                     ("commons", commons), ("wiki", wiki), ("lrclib", lrclib), ("pexels", pexels), ("pixabay", pixabay),
                     ("invidious", invidious), ("zemare", zemare), ("bing", bing), ("vocab", vocab)]:
        http(name, fn)
        save("report.json", report)


asyncio.run(main())
