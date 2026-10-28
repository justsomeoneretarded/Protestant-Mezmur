"""Produce one finished song folder from a fetched input folder.

input folder (written by `mezmur.fetch`):  audio.*, info.json, telegram.json, photos/
output folder: video.mp4, thumbnails, captions, lyrics, YouTube upload kit, diagnostics.
"""
import argparse
import json
import re
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import asr, audio, backgrounds, names, thumbnail, web
from .align import Lyrics, align_lyrics, build_target, complete_with_lyrics, smooth_words
from .common import duration as common_duration, write_json
from .features import analyze
from .geez import fold, romanize
from .lyrics_sources import gather
from .lyrics_text import all_lines, line_repeats, line_stanzas, similarity, stanzas_text
from .separate import separate
from .subtitles import build_cues, write_ass, write_srt
from .visuals import Renderer, Source, render_video
from .youtube import write_kit

# Strict: a song is only made when found lyrics clearly explain the singing. Measured on real songs, the
# right lyrics explain 81-87 % of the singing (gain 0.21-0.25, 74-97 % of lines sung); wrong lyrics <= 36 %.
ACCEPT_SHARE = 0.70       # share of vocal frames inside aligned lyric lines
ACCEPT_GAIN = 0.18        # per-vocal-frame log-likelihood gain over "unscripted singing"
ACCEPT_LINES = 0.70       # share of the found lines that were actually sung
MAX_ESTIMATED = 0.35      # share of word timings that may be estimated rather than heard
MIN_TEXT_SIM = 0.06       # cheap pre-filter against the automatic draft

def notice(mid, text, level="notice"):
    """A line on the Actions run page (readable later, unlike the step log)."""
    print(f"::{level} title=Song {mid}::{text}", flush=True)


def skip(out, mid, record, cand_reports, timings, reason):
    """Lyrics are not certain: make nothing for this song, say why, keep diagnostics."""
    from .youtube import titles
    meta = {k: record.get(k, "") for k in ["title_am", "title_latin", "artist_am", "artist_latin"]}
    ts = titles(meta)
    (out / "title.txt").write_text((ts[0] if ts else f"Telegram message {mid}") + "\n", encoding="utf-8")
    write_json(out / "skipped.json", {"message_id": mid, "reason": reason,
                                      "fix": f"Put the correct lyrics in corrections/{mid}.txt and commit it to main.",
                                      "candidates": cand_reports, "timings_seconds": timings})
    notice(mid, f"Skipped — {reason} Add corrections/{mid}.txt to make it.", level="warning")
    log(f"[{mid}] skipped: {reason}")
    return {"message_id": mid, "mode": "skipped", "reason": reason}


def log(*a, **k):
    print(*a, **{"flush": True, **k})


def _audio_file(folder):
    files = [p for p in Path(folder).glob("audio.*")]
    if not files:
        raise FileNotFoundError(f"No audio.* in {folder}")
    return files[0]


def chorus_lines_of(stanzas, aligned_words):
    marked = set()
    index = 0
    for s in stanzas:
        for _ in s["lines"]:
            if s.get("chorus"):
                marked.add(index)
            index += 1
    if marked:
        return marked
    counts = {}
    last = None
    for w in aligned_words:
        key = (w["line"], w["phrase"])
        if key != last:
            counts[w["line"]] = counts.get(w["line"], 0) + 1
            last = key
    if not counts:
        return set()
    repeated = {l for l, c in counts.items() if c >= 2}
    return repeated if len(repeated) < 0.8 * len(counts) else set()


def choose_lyrics(cands, draft_text, logp, frame_s, vocab, energy, max_aligned=6):
    """Verify candidates against the audio itself; return (best, report list)."""
    for c in cands:
        c["similarity"] = similarity(c["text"], draft_text) if draft_text else {"score": 0.0}
    ordered = sorted(cands, key=lambda c: (not c["trusted"], -c["similarity"]["score"]))
    reports, best = [], None
    aligned = 0
    for c in ordered:
        if not c["trusted"] and (c["similarity"]["score"] < MIN_TEXT_SIM or aligned >= max_aligned):
            reports.append({"source": c["source"], "url": c["url"], "similarity": c["similarity"],
                            "rejected": "text mismatch" if c["similarity"]["score"] < MIN_TEXT_SIM else "not checked"})
            continue
        aligned += 1
        lines = [l.split() for l in all_lines(c["stanzas"])]
        target = build_target(Lyrics(lines), vocab["char_to_id"])
        t0 = time.time()
        words, rep = align_lyrics(logp, frame_s, target, vocab["blank"], vocab["delimiter"], energy=energy,
                                  repeats=line_repeats(c["stanzas"]))
        lines_share = rep["lines_used"] / max(1, rep["lines_total"])
        entry = {"source": c["source"], "url": c["url"], "similarity": c["similarity"],
                 **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items() if k != "phrase_details"},
                 "lines_share": round(lines_share, 3), "seconds": round(time.time() - t0, 1)}
        ok = c["trusted"] or (rep["matched_share"] >= ACCEPT_SHARE and rep["gain_per_vocal_frame"] >= ACCEPT_GAIN
                              and lines_share >= ACCEPT_LINES)
        entry["accepted"] = bool(ok)
        reports.append(entry)
        log(f"  candidate {c['source']}: sim={c['similarity']['score']} share={rep['matched_share']:.2f} "
            f"gain={rep['gain_per_vocal_frame']:.2f} lines={rep['lines_used']}/{rep['lines_total']} "
            f"complete={rep['complete_share']:.2f} -> {'OK' if ok else 'no'}")
        if ok:
            score = rep["matched_share"] + rep["gain_per_vocal_frame"] + 0.3 * rep["complete_share"] + (10 if c["trusted"] else 0)
            if best is None or score > best[0]:
                best = (score, c, words, rep, lines)
            if c["trusted"]:
                break
    return best, reports


def lrc_check(cues, candidates, duration):
    """Compare our line starts with human-made synced lyrics (LRCLIB), when available."""
    import re
    for c in candidates:
        lrc = c.get("synced")
        if not lrc or abs((c.get("lrclib_duration") or 0) - duration) > 2.5:
            continue
        marks = []
        for line in lrc.splitlines():
            m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.+)", line)
            if m:
                marks.append((int(m.group(1)) * 60 + float(m.group(2)), fold(m.group(3))))
        diffs, used = [], set()
        for cue in cues:
            key = fold(cue["text"])
            best = min((abs(t - cue["start"]), i) for i, (t, text) in enumerate(marks)
                       if i not in used and text and (text in key or key in text)) if marks else None
            if best and best[0] < 8:
                used.add(best[1])
                diffs.append(best[0])
        if len(diffs) >= 4:
            return {"source": c["url"], "lines_compared": len(diffs), "median_seconds": round(float(np.median(diffs)), 2)}
    return None


def face_square(img, face, margin=2.4):
    x, y, w, h = face["box"]
    cx, cy = x + w / 2, y + h / 2
    side = int(max(w, h) * margin)
    x0 = int(max(0, min(img.shape[1] - side, cx - side / 2)))
    y0 = int(max(0, min(img.shape[0] - side, cy - side / 2 + h * 0.15)))
    side = min(side, img.shape[1] - x0, img.shape[0] - y0)
    return img[y0:y0 + side, x0:x0 + side]


def photo_candidates(work, inp, record, videos):
    """Photos of the singer. 'trusted' ones belong to this song (your artists/ folder, the song's cover art,
    its Telegram post, official uploads of this exact song) and decide who the singer is; name searches
    (YouTube channel, Wikimedia, web) only add photos of that same face."""
    cands = []
    pdir = work / "photos"
    pdir.mkdir(exist_ok=True)
    names_all = [n for n in [record.get("artist_latin"), record.get("artist_am")] if n]
    root = Path(__file__).resolve().parent.parent / "artists"
    for name in names_all:
        if root.is_dir():
            for p in root.glob(f"{names.slug(name)}*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    cands.append({"path": p, "source": f"artists/{p.name}", "weight": 6, "trusted": True})
    for p in sorted((inp / "photos").glob("*")) if (inp / "photos").is_dir() else []:
        cands.append({"path": p, "source": f"telegram:{p.name}", "weight": 2, "trusted": True})
    cover = audio.cover_art(_audio_file(inp), pdir / "cover.jpg")
    if cover:
        cands.append({"path": cover, "source": "embedded cover art", "weight": 3, "trusted": True})
    artist = record.get("artist_latin") or record.get("artist_am")
    if not artist:
        return cands
    from .lyrics_sources import title_match
    try:
        # Official uploads of this very song: title names the song AND the channel or title names the singer.
        picked = 0
        for v in videos:
            if picked >= 3:
                break
            if title_match(v.get("title", ""), record) >= 0.8 and (
                    names.same_person_name(v.get("channel", ""), *names_all)
                    or any(names.same_person_name(part, *names_all)
                           for part in re.split(r"[|/\-–—()]+", v.get("title", "")))):
                data = web.youtube_thumbnail(v["video_id"])
                if data:
                    path = pdir / f"yt-video-{v['video_id']}.jpg"
                    path.write_bytes(data)
                    cands.append({"path": path, "source": f"youtube video {v['video_id']}", "weight": 2, "trusted": True})
                    picked += 1
        chans = [c for c in web.youtube_search(artist, channels=True)
                 if names.same_person_name(c["title"], *names_all)]
        chans.sort(key=lambda c: (" - Topic" in c["title"], not c["verified"], -_subs(c["subscribers"])))
        for i, c in enumerate(chans[:2]):
            r = web.get(c["avatar"])
            if r is not None and r.status_code == 200 and len(r.content) > 5000:
                path = pdir / f"yt-channel-{i}.jpg"
                path.write_bytes(r.content)
                cands.append({"path": path, "source": f"youtube channel {c['title']}", "weight": 2,
                              "trusted": False, "strict_name": True})
    except Exception as error:
        log(f"  youtube photos failed: {error}")
    try:
        for i, im in enumerate(web.commons_images(artist, 4)):
            if not names.same_person_name(im.get("title", "").replace("File:", "").rsplit(".", 1)[0], *names_all):
                continue
            r = web.get(im["url"]) if im.get("url") else None
            if r is not None and r.status_code == 200:
                path = pdir / f"commons-{i}.jpg"
                path.write_bytes(r.content)
                cands.append({"path": path, "source": f"wikimedia {im['title']} ({im['license']})", "weight": 1,
                              "trusted": False, "credit": f"{im['title']} — {im['artist']} ({im['license']})"})
    except Exception as error:
        log(f"  commons photos failed: {error}")
    try:
        query = f'"{artist}" {"mezmur" if record.get("artist_latin") else "መዝሙር"}'
        for i, im in enumerate(web.image_search(query, 10)[:8]):
            r = web.get(im["url"], timeout=15) if im.get("url") else None
            if r is not None and r.status_code == 200 and len(r.content) > 15000:
                path = pdir / f"web-{i}.jpg"
                path.write_bytes(r.content)
                cands.append({"path": path, "source": f"web image {web.domain(im.get('source') or im['url'])}",
                              "weight": 1, "trusted": False,
                              "named": names.same_person_name(im.get("title", ""), *names_all)})
    except Exception as error:
        log(f"  web photos failed: {error}")
    return cands


def _subs(text):
    import re
    m = re.search(r"([\d.]+)\s*([KM]?)", text or "")
    if not m:
        return 0
    return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6}[m.group(2)]


def produce(inp, out, work, separator=None, height=1080, fps=30, preset="medium", corrections="corrections"):
    inp, out, work = Path(inp), Path(out), Path(work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    info = json.loads((inp / "info.json").read_text(encoding="utf-8"))
    tg = json.loads((inp / "telegram.json").read_text(encoding="utf-8")) if (inp / "telegram.json").is_file() else {}
    used = json.loads((inp / "used_backgrounds.json").read_text()) if (inp / "used_backgrounds.json").is_file() else []
    source_audio = _audio_file(inp)
    mid = info["message_id"]
    timings = {}
    t = time.time()

    # 1. Names.
    file_tags = audio.tags(source_audio)
    record = names.resolve({**info, "tags": file_tags})
    record["album"] = file_tags.get("album", "")
    log(f"[{mid}] {record.get('title_am') or record.get('title_latin')} — {record.get('artist_am') or record.get('artist_latin')}")

    # 2. Audio.
    master = work / "master.wav"
    loudness = audio.master(source_audio, master)
    duration = common_duration(master)
    timings["master"] = round(time.time() - t, 1)

    # 3. Vocals + acoustic model (load the model first so a broken install fails fast).
    vocab = asr.vocabulary()
    t = time.time()
    vocals = separate(source_audio, work, model=separator, log=log)
    timings["separation"] = round(time.time() - t, 1)
    t = time.time()
    logp, frame_s = asr.frame_log_probs(vocals, log=log)
    energy = asr.vocal_energy(vocals, frame_s, len(logp))
    draft = asr.greedy_words(logp, frame_s, vocab)
    draft_text = " ".join(w["word"] for w in draft)
    timings["acoustic_model"] = round(time.time() - t, 1)

    # 4. Names from YouTube titles, then lyrics search.
    t = time.time()
    videos = []
    try:
        q = " ".join(p for p in [record.get("title_am") or record.get("title_latin"),
                                 record.get("artist_latin") or record.get("artist_am")] if p)
        videos = web.youtube_search(q) if q else []
    except Exception as error:
        log(f"  youtube search failed: {error}")
    record = names.learn_from_titles(record, [v["title"] for v in videos])
    channels = []
    plan_file = inp.parent / "plan.json"
    learned = []
    if plan_file.is_file():
        learned = json.loads(plan_file.read_text()).get("snapshot", {}).get("state", {}).get("lyrics_channels", [])
    from .fetch import DEFAULT_LYRICS_CHANNELS
    for c in (os.environ.get("LYRICS_CHANNELS") or DEFAULT_LYRICS_CHANNELS).split(",") + list(learned):
        c = c.strip().lstrip("@")
        if c and c not in channels:
            channels.append(c)
    cands, titles, overrides, search_report = gather(record, mid, tg, corrections, duration, log=log, videos=videos,
                                                     draft=draft, channels=channels,
                                                     library=os.environ.get("MEZMUR_LIBRARY", "library/lyrics"))
    notice(mid, "Lyrics search — " + "; ".join(
        f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in search_report.items()))
    record = names.learn_from_titles(record, titles)
    for k in ["title_am", "title_latin", "artist_am", "artist_latin"]:
        if overrides.get(k):
            record[k] = overrides[k]
    record = names.finalize(record)
    timings["lyrics_search"] = round(time.time() - t, 1)
    if os.environ.get("MEZMUR_DEBUG"):
        dbg = out / "debug"
        dbg.mkdir(exist_ok=True)
        np.savez_compressed(dbg / "acoustic.npz", logp=logp.astype(np.float16), frame_s=frame_s, energy=energy)
        write_json(dbg / "vocab.json", {"id_to_char": {str(k): v for k, v in vocab["id_to_char"].items()},
                                        "blank": vocab["blank"], "delimiter": vocab["delimiter"]})
        write_json(dbg / "candidates.json", [{k: v for k, v in c.items() if k != "synced"} | {"synced": c.get("synced")}
                                             for c in cands])
        write_json(dbg / "record.json", {"record": record, "draft": draft, "duration": duration})
        if os.environ.get("MEZMUR_DEBUG") == "acoustic":
            log(f"[{mid}] debug dump written; stopping before alignment/render")
            return {"message_id": mid, "mode": "debug"}

    # 5. Verify + align.
    t = time.time()
    best, cand_reports = choose_lyrics(cands, draft_text, logp, frame_s, vocab, energy)
    review = {"needs_review": False, "reasons": []}
    if best:
        _, chosen, aligned, align_report, lines = best
        stanzas = chosen["stanzas"]
        flat = [word for line in lines for word in line]
        aligned, filled = complete_with_lyrics(aligned, lines, energy, frame_s, duration, logp=logp,
                                               char_to_id=vocab["char_to_id"], blank=vocab["blank"],
                                               delimiter=vocab["delimiter"], repeats=line_repeats(stanzas),
                                               stanza_of=line_stanzas(stanzas))
        align_report["completion"] = filled
        if os.environ.get("MEZMUR_DEBUG"):
            dbg = out / "debug"
            dbg.mkdir(exist_ok=True)
            write_json(dbg / "alignment.json", {
                "lyrics_source": chosen["source"], "url": chosen["url"], "lines": [" ".join(l) for l in lines],
                "repeats": line_repeats(stanzas),
                "first_pass": [{k: v for k, v in d.items()} for d in align_report.get("phrase_details", [])],
                "gap_decisions": filled.get("gaps", []),
                "final": [{"line": w["line"], "word": flat[w["word_index"]], "start": round(w["start"], 2),
                           "end": round(w["end"], 2), "heard": bool(w.get("recognised", not w.get("estimated")))}
                          for w in sorted(aligned, key=lambda w: w["start"])]})
        words = [{**w, "word": flat[w["word_index"]]} for w in aligned]
        words = smooth_words(words)
        est = filled["estimated_words"] / max(1, len(words))
        if est > 0.25:
            review["reasons"].append(f"{est:.0%} of the word timings were estimated (the singing there was hard to "
                                     "hear, e.g. choir parts). Spot-check those lines in the video.")
        chorus = chorus_lines_of(stanzas, aligned)
        lyrics_source = f"{chosen['source']} {chosen['url']}".strip()
        if not chosen["trusted"] and align_report["matched_share"] < 0.8:
            review["reasons"].append(f"Only {align_report['matched_share']:.0%} of the singing matched the found lyrics; "
                                     "watch for missing lines.")
        mode = "lyrics-aligned"
    else:
        checked = [r for r in cand_reports if "matched_share" in r]
        if not checked:
            reason = "No lyrics for this song were found in any source."
        else:
            r = max(checked, key=lambda r: r["matched_share"] + r.get("gain_per_vocal_frame", 0))
            fails = []
            if r["matched_share"] < ACCEPT_SHARE:
                fails.append(f"they explain only {r['matched_share']:.0%} of the singing ({ACCEPT_SHARE:.0%} needed)")
            if r.get("lines_share", 1) < ACCEPT_LINES:
                fails.append(f"only {r.get('lines_used')} of their {r.get('lines_total')} lines are sung in this song "
                             f"({ACCEPT_LINES:.0%} needed — the page may hold a different version or extra lines)")
            if r.get("gain_per_vocal_frame", 1) < ACCEPT_GAIN:
                fails.append(f"the words match the sound only weakly (score {r['gain_per_vocal_frame']:.2f}, "
                             f"{ACCEPT_GAIN} needed)")
            reason = (f"The best lyrics found ({r['source']} {r.get('url', '')}) did not pass: "
                      + "; ".join(fails or ["checks failed"]) + ".")
        return skip(out, mid, record, cand_reports, timings, reason)
    if not chosen["trusted"] and est > MAX_ESTIMATED:
        return skip(out, mid, record, cand_reports, timings,
                    f"The lyrics were found, but {est:.0%} of the word timings could only be estimated "
                    f"(limit {MAX_ESTIMATED:.0%}), so the karaoke would not be reliable.")
    sure = "your corrected lyrics" if chosen["trusted"] else f"{align_report['matched_share']:.0%} of the singing matched"
    notice(mid, f"Lyrics verified ({sure}) — source: {lyrics_source}")
    if not chosen["source"].startswith("library"):
        from .names import slug
        learned_dir = out / "learned"
        learned_dir.mkdir(exist_ok=True)
        head = "\n".join(f"{k}: {record.get(k, '')}" for k in ["title_am", "title_latin", "artist_am", "artist_latin"])
        (learned_dir / "lyrics.txt").write_text(
            f"---\n{head}\nsource: {chosen['source']} {chosen['url']}\nmessage_id: {mid}\n---\n"
            + stanzas_text(stanzas), encoding="utf-8")
        channel = re.match(r"telegram:(?:channel )?@([A-Za-z0-9_]+)", chosen["source"])
        write_json(learned_dir / "learned.json", {
            "path": f"lyrics/{slug(record.get('artist_latin') or record.get('artist_am') or 'unknown')}/"
                    f"{slug(record.get('title_latin') or record.get('title_am') or str(mid))}.txt",
            "channel": channel.group(1) if channel else None})
    if not words:
        raise RuntimeError("No timed lyrics could be produced")
    cues = build_cues(words, chorus_lines=chorus)
    timings["alignment"] = round(time.time() - t, 1)
    lrc = lrc_check(cues, cands, duration)
    if lrc and lrc["median_seconds"] > 1.0:
        review["reasons"].append(f"Line timing differs from LRCLIB's synced lyrics by {lrc['median_seconds']} s (median); "
                                 "spot-check the karaoke timing.")

    # 6. Subtitles & lyrics files.
    (out / "captions").mkdir(exist_ok=True)
    write_srt(cues, out / "captions" / "am.srt")
    write_srt(cues, out / "captions" / "am-Latn.srt", romanized=True)
    text = stanzas_text(stanzas) if stanzas else "\n".join(c["text"] for c in cues) + "\n"
    (out / "lyrics.txt").write_text(text, encoding="utf-8")
    (out / "lyrics.romanized.txt").write_text("\n".join(romanize(l) if l.strip() else "" for l in text.splitlines()) + "\n",
                                               encoding="utf-8")
    write_json(out / "words.json", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items()} for w in words])

    # 7. Artist photo.
    t = time.time()
    photo_cands = photo_candidates(work, inp, record, videos)
    ranked = thumbnail.rank_photos(photo_cands, log=log) if photo_cands else []
    person, badge_photo, photo_img, photo_source = None, None, None, None
    photo_credit = None
    confirmed = [i for i in ranked if i.get("same_person")]
    for item in confirmed[:4]:
        img = cv2.imread(str(item["path"]))
        cut = thumbnail.cutout(img, item["face"])
        if cut is not None:
            person, photo_img, photo_source = cut, img, item["source"]
            photo_credit = item.get("credit")
            badge_photo = face_square(img, item["face"])
            break
    if person is None and confirmed:
        img = cv2.imread(str(confirmed[0]["path"]))
        photo_img, badge_photo, photo_source = img, face_square(img, confirmed[0]["face"]), confirmed[0]["source"]
    if badge_photo is None:
        review["reasons"].append("No photo could be confirmed as this singer; thumbnails use the text design.")
    timings["photos"] = round(time.time() - t, 1)

    # 8. Backgrounds.
    t = time.time()
    # One calm, clean nature photograph for the whole song (and its thumbnails).
    slot, slots = 0, 1
    plan_file = inp.parent / "plan.json"
    if plan_file.is_file():  # songs of one batch must not pick the same picture
        ids = [int(x) for x in json.loads(plan_file.read_text()).get("ids", [])]
        if mid in ids:
            slot, slots = ids.index(mid), len(ids)
    still, still_record = backgrounds.nature_still(work / "background", used=used, seed=mid, log=log,
                                                   slot=slot, slots=slots)
    if still is None:
        still, still_record = backgrounds.fallback_scene(work, used=used, seed=mid)
        log("  no clean photo found online; using a bundled picture" if still else
            "  no unused picture left; using a generated light background")
    clip_records = [still_record] if still_record else []
    sources = [still] if still else [Source("procedural", label="generated light")]
    # Lyric style for this song: varies by song; dark letters on bright daytime pictures.
    from .backgrounds import brightness
    from .subtitles import choose_theme
    still_img = cv2.imread(still.path) if still is not None and still.path else None
    bright = brightness(still_img) >= 140
    theme = choose_theme(seed=mid // 10 + mid, bright_background=bright)
    write_ass(cues, work / "lyrics.ass", theme=theme)
    shutil.copy2(work / "lyrics.ass", out / "lyrics.ass")
    log(f"  look: {'bright daytime' if bright else 'darker'} picture, lyric style '{theme['name']}'")
    timings["footage"] = round(time.time() - t, 1)

    # 9. Thumbnails.
    meta = {k: record.get(k, "") for k in ["title_am", "title_latin", "artist_am", "artist_latin"]}
    bg_img = cv2.imread(still.path) if still is not None and still.path else None
    for variant, name in [("a", "thumbnail.jpg"), ("b", "thumbnail-b.jpg"), ("c", "thumbnail-c.jpg")]:
        img = thumbnail.compose(meta, background=bg_img, person=person, photo=photo_img, variant=variant, seed=mid)
        thumbnail.save(img, out / name)

    # 10. Video.
    t = time.time()
    width = height * 16 // 9
    feats = analyze(master, fps=fps)
    accent = tuple(int(theme["sung"][i:i + 2], 16) for i in (1, 3, 5))
    renderer = Renderer(width, height, fps, duration, feats, sources, cues, meta, artist_photo=badge_photo, seed=mid,
                        bright=bright, accent=accent)
    render_video(renderer, master, work / "lyrics.ass", out / "video.mp4", preset=preset, log=log)
    timings["render"] = round(time.time() - t, 1)

    if mode == "transcription":
        review["reasons"].append(f"To fix the words: copy the Amharic lyrics from description.txt into "
                                 f"corrections/{mid}.txt, correct them and commit — the video is re-made "
                                 "with the right words and timing.")
    # 11. Upload kit + diagnostics.
    credits = [c["credit"] for c in clip_records if c.get("credit")]
    if photo_credit:
        credits.insert(0, f"Photo: {photo_credit}")
    review["needs_review"] = bool(review["reasons"]) or mode != "lyrics-aligned"
    if record.get("title_latin_auto") or record.get("artist_latin_auto"):
        review["reasons"].append("The romanized title/artist was generated automatically; check spelling in title.txt.")
    write_kit(out, {**meta, "album": record.get("album", "")}, stanzas, cues, duration, chorus, credits,
              lyrics_source, renderer.outro_start, review)
    write_json(out / "metadata.json", {
        "message_id": mid, "names": record, "duration_seconds": round(duration, 2), "loudness_before": loudness,
        "lyrics_mode": mode, "lyrics_source": lyrics_source, "artist_photo": photo_source,
        "backgrounds": clip_records, "tempo_bpm": round(feats.tempo, 1), "timings_seconds": timings,
        "review": review, "model": asr.MODEL})
    write_json(out / "quality.json", {"mode": mode, "candidates": cand_reports, "lrc_check": lrc, "alignment": {
        k: v for k, v in align_report.items() if k != "phrase_details"}, "phrases": align_report.get("phrase_details", []),
        "draft_text": draft_text, "draft_word_count": len(draft), "review": review})
    write_json(out / "backgrounds.json", [c["id"] for c in clip_records])
    log(f"[{mid}] done in {sum(timings.values()):.0f}s · {mode} · {timings}")
    return {"message_id": mid, "mode": mode, "review": review, "title": meta}


def main():
    p = argparse.ArgumentParser(description="Render one fetched song")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--work", default=".runtime/work")
    p.add_argument("--separator", default=None, help="audio-separator model, 'htdemucs' or 'none'")
    p.add_argument("--height", type=int, default=1080, choices=[720, 1080])
    p.add_argument("--preset", default="medium")
    a = p.parse_args()
    produce(a.input, a.output, a.work, a.separator, a.height, preset=a.preset)


if __name__ == "__main__":
    main()
