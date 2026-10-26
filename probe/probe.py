"""Probe 2: lyrics site structures, LRCLIB, Telegram lyrics channels, separator timing."""
import asyncio, json, os, re, subprocess, time, traceback
from pathlib import Path

import requests

OUT = Path("out"); OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
report = {}


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def step(name, fn):
    t = time.time()
    try:
        r = fn()
    except Exception:
        r = {"error": traceback.format_exc()[-1500:]}
    report[name] = {"t": round(time.time() - t, 1), "r": r}
    save("report.json", report)


def page(name, url, **kw):
    r = requests.get(url, headers=UA, timeout=30, **kw)
    (OUT / f"{name}.html").write_text(r.text[:400000], encoding="utf-8")
    return {"status": r.status_code, "url": r.url, "len": len(r.text), "ctype": r.headers.get("content-type")}


def sites():
    out = {}
    out["wm_page"] = page("wikimezmur_page", "https://wikimezmur.org/am/Addisalem_Assefa/Sebariw_Gieta_Kefitie_Wettual/Siwed_Gieta")
    for api in ["https://wikimezmur.org/api.php", "https://wikimezmur.org/w/api.php", "https://wikimezmur.org/am/api.php"]:
        try:
            r = requests.get(api, params={"action": "query", "list": "search", "srsearch": "ሲወድ ጌታ", "format": "json"}, headers=UA, timeout=20)
            out["wm_api " + api] = [r.status_code, r.text[:600]]
        except Exception as e:
            out["wm_api " + api] = repr(e)[:200]
    out["legeta"] = page("legeta", "https://legeta316.com/lyrics/siwed-gieta--69a1ecb4cb1454b2c3a8ac5f")
    out["legeta_search"] = page("legeta_search", "https://legeta316.com/search", params={"q": "ሲወድ ጌታ"})
    out["debter"] = page("debter", "https://mezmurdebter.vercel.app/66/66-2/1310")
    out["mezmuroch"] = page("mezmuroch", "https://mezmuroch.com/artists/Addisalem_Assefa")
    for u in ["https://wikimezmur.org/robots.txt", "https://legeta316.com/robots.txt", "https://mezmuroch.com/robots.txt", "https://mezmurdebter.vercel.app/robots.txt"]:
        try:
            r = requests.get(u, headers=UA, timeout=20); out[u] = [r.status_code, r.text[:500]]
        except Exception as e:
            out[u] = repr(e)[:200]
    return out


def lrclib():
    r = requests.get("https://lrclib.net/api/search", params={"artist_name": "Addisalem Assefa", "track_name": "Siwod Geta"}, headers=UA, timeout=30).json()
    r2 = requests.get("https://lrclib.net/api/search", params={"q": "Yishak Sedik"}, headers=UA, timeout=30).json()
    return {"siwod": [{k: x.get(k) for k in ["id", "trackName", "artistName", "albumName", "duration", "plainLyrics", "syncedLyrics"]} for x in r[:2]],
            "yishak": [(x["trackName"], x["albumName"], x["duration"], bool(x.get("syncedLyrics")), (x.get("plainLyrics") or "")[:80]) for x in r2[:15]]}


def ddgs_queries():
    from ddgs import DDGS
    out = {}
    for q in ["ተመስገን ይስሐቅ ሰድቅ ግጥም", "Temesgen Yishak Sedik lyrics", "ክቡር አደራ ይስሐቅ ሰድቅ", "Sebariw Geta Kefite Wetoal lyrics Addisalem"]:
        for backend in ["auto", "yahoo", "duckduckgo"]:
            try:
                out[f"{q} [{backend}]"] = [(x["title"][:90], x["href"][:140]) for x in DDGS().text(q, max_results=10, backend=backend)]
            except Exception as e:
                out[f"{q} [{backend}]"] = repr(e)[:160]
    return out


async def telegram():
    from telethon import TelegramClient
    from telethon.sessions import StringSession
    c = TelegramClient(StringSession(os.environ["TELEGRAM_SESSION"]), int(os.environ["TELEGRAM_API_ID"]), os.environ["TELEGRAM_API_HASH"])
    await c.connect()
    out = {}
    chans = ["MezmurLyrics0", "mezmur1", "Amharic_spiritual_song_lyrics", "SongLegetaLyrics", "mezmurlyricsbb", "mezmur_lyrics", "lyrics_mezmur", "mezmur_lyrics2", "mezmur_gitem", "eyesusyiwedenal", "mezmuroch1"]
    for ch in chans:
        try:
            ent = await c.get_entity(ch)
            total = (await c.get_messages(ent, limit=0)).total
            res = {"total": total}
            for q in ["ሲወድ ጌታ", "ተመስገን", "ይስሐቅ", "አዲስዓለም"]:
                ms = await c.get_messages(ent, search=q, limit=3)
                res[q] = [{"id": m.id, "text": (m.raw_text or "")[:700]} for m in ms]
            out[ch] = res
        except Exception as e:
            out[ch] = repr(e)[:200]
    # global search over joined dialogs
    try:
        from telethon.tl.functions.messages import SearchGlobalRequest
        from telethon.tl.types import InputMessagesFilterEmpty, InputPeerEmpty
        r = await c(SearchGlobalRequest(q="ሲወድ ጌታ", filter=InputMessagesFilterEmpty(), min_date=None, max_date=None, offset_rate=0, offset_peer=InputPeerEmpty(), offset_id=0, limit=10))
        out["global"] = [(getattr(m, "message", "") or "")[:200] for m in r.messages]
    except Exception as e:
        out["global"] = repr(e)[:300]
    # download one song for timing tests
    ent = await c.get_entity(os.environ.get("TELEGRAM_CHANNEL") or "@amharicprotestantmezmurss")
    m = await c.get_messages(ent, ids=4)
    await c.download_media(m, file="/tmp/song4.m4a")
    await c.disconnect()
    return out


def separators():
    res = {}
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "60", "-t", "60", "-i", "/tmp/song4.m4a", "-ar", "44100", "-ac", "2", "/tmp/clip.wav"], check=True)
    for model in ["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "vocals_mel_band_roformer.ckpt", "htdemucs_ft.yaml", "UVR-MDX-NET-Voc_FT.onnx"]:
        t = time.time()
        p = subprocess.run(["audio-separator", "/tmp/clip.wav", "-m", model, "--output_dir", f"/tmp/sep-{model}", "--single_stem", "Vocals", "--model_file_dir", "/tmp/models"], capture_output=True, text=True)
        res[model] = {"secs_for_60s": round(time.time() - t, 1), "rc": p.returncode, "tail": (p.stdout + p.stderr)[-600:], "files": os.listdir(f"/tmp/sep-{model}") if os.path.isdir(f"/tmp/sep-{model}") else []}
        save("report.json", report | {"sep_partial": res})
    return res


async def main():
    step("sites", sites)
    step("lrclib", lrclib)
    step("ddgs", ddgs_queries)
    t = time.time()
    try:
        report["telegram"] = {"r": await telegram(), "t": round(time.time() - t, 1)}
    except Exception:
        report["telegram"] = traceback.format_exc()[-1500:]
    save("report.json", report)
    step("separators", separators)


asyncio.run(main())
