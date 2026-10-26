"""Produce one finished song folder from a fetched input folder.

input folder (written by `mezmur.fetch`):  audio.*, info.json, telegram.json, photos/
output folder: video.mp4, thumbnails, captions, lyrics, YouTube upload kit, diagnostics.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import asr, audio, backgrounds, names, thumbnail, web
from .align import Lyrics, align_lyrics, build_target, smooth_words
from .common import duration as common_duration, write_json
from .features import analyze
from .geez import fold, romanize
from .lyrics_sources import gather
from .lyrics_text import all_lines, similarity, stanzas_text
from .separate import separate
from .subtitles import build_cues, write_ass, write_srt
from .visuals import Renderer, Source, render_video
from .youtube import write_kit

ACCEPT_SHARE = 0.55       # share of vocal time explained by the lyrics
ACCEPT_DEFICIT = 1.35     # mean per-frame log-prob gap to the unconstrained path


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


def choose_lyrics(cands, draft_text, logp, frame_s, vocab, energy):
    """Verify candidates against the audio; return (best, report list)."""
    scored = []
    for c in cands:
        sim = similarity(c["text"], draft_text) if draft_text else {"score": 0.0}
        c["similarity"] = sim
        scored.append(c)
    scored.sort(key=lambda c: (not c["trusted"], -c["similarity"]["score"]))
    reports, best = [], None
    for c in scored[:4]:
        if not c["trusted"] and c["similarity"]["score"] < 0.18:
            reports.append({"source": c["source"], "url": c["url"], "similarity": c["similarity"], "rejected": "text mismatch"})
            continue
        lines = [l.split() for l in all_lines(c["stanzas"])]
        target = build_target(Lyrics(lines), vocab["char_to_id"])
        t0 = time.time()
        words, rep = align_lyrics(logp, frame_s, target, vocab["blank"], vocab["delimiter"], energy=energy)
        entry = {"source": c["source"], "url": c["url"], "similarity": c["similarity"],
                 "matched_share": round(rep["matched_share"], 3), "mean_deficit": rep["mean_deficit"],
                 "phrases": rep["phrases"], "null_phrases": rep["null"], "seconds": round(time.time() - t0, 1)}
        ok = c["trusted"] or (rep["matched_share"] >= ACCEPT_SHARE and (rep["mean_deficit"] or 9) <= ACCEPT_DEFICIT)
        entry["accepted"] = bool(ok)
        reports.append(entry)
        log(f"  candidate {c['source']}: sim={c['similarity']['score']} share={entry['matched_share']} "
            f"deficit={entry['mean_deficit']} -> {'OK' if ok else 'no'}")
        if ok:
            score = rep["matched_share"] - 0.2 * (rep["mean_deficit"] or 0) + (5 if c["trusted"] else 0)
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
    cands = []
    pdir = work / "photos"
    pdir.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parent.parent / "artists"
    for name in [record.get("artist_latin"), record.get("artist_am")]:
        if name and root.is_dir():
            for p in root.glob(f"{names.slug(name)}*"):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    cands.append({"path": p, "source": f"artists/{p.name}", "weight": 6})
    for p in sorted((inp / "photos").glob("*")) if (inp / "photos").is_dir() else []:
        cands.append({"path": p, "source": f"telegram:{p.name}", "weight": 2})
    cover = audio.cover_art(_audio_file(inp), pdir / "cover.jpg")
    if cover:
        cands.append({"path": cover, "source": "embedded cover art", "weight": 2})
    artist = record.get("artist_latin") or record.get("artist_am")
    if artist:
        try:
            chans = web.youtube_search(artist, channels=True)
            chans = [c for c in chans if names.similar(c["title"].replace("Official", ""), artist) >= 0.6
                     or names.similar(c["title"], record.get("artist_am", "") or artist) >= 0.6]
            chans.sort(key=lambda c: (" - Topic" in c["title"], not c["verified"], -_subs(c["subscribers"])))
            for i, c in enumerate(chans[:2]):
                r = web.get(c["avatar"])
                if r is not None and r.status_code == 200 and len(r.content) > 5000:
                    path = pdir / f"yt-channel-{i}.jpg"
                    path.write_bytes(r.content)
                    cands.append({"path": path, "source": f"youtube channel {c['title']}", "weight": 3 if i == 0 else 2})
            official = {c["channel_id"] for c in chans[:2]}
            picked = 0
            for v in videos:
                if picked >= 3:
                    break
                if v.get("channel_id") in official or names.similar(v.get("channel", ""), artist) >= 0.7:
                    data = web.youtube_thumbnail(v["video_id"])
                    if data:
                        path = pdir / f"yt-video-{v['video_id']}.jpg"
                        path.write_bytes(data)
                        cands.append({"path": path, "source": f"youtube video {v['video_id']}", "weight": 2})
                        picked += 1
        except Exception as error:
            log(f"  youtube photos failed: {error}")
        try:
            for i, im in enumerate(web.commons_images(artist, 4)):
                r = web.get(im["url"]) if im.get("url") else None
                if r is not None and r.status_code == 200:
                    path = pdir / f"commons-{i}.jpg"
                    path.write_bytes(r.content)
                    cands.append({"path": path, "source": f"wikimedia {im['title']} ({im['license']})", "weight": 2,
                                  "credit": f"{im['title']} — {im['artist']} ({im['license']})"})
        except Exception as error:
            log(f"  commons photos failed: {error}")
        try:
            for i, im in enumerate(web.image_search(f"{artist} {'singer' if record.get('artist_latin') else 'ዘማሪ'}", 8)[:6]):
                r = web.get(im["url"], timeout=15) if im.get("url") else None
                if r is not None and r.status_code == 200 and len(r.content) > 15000:
                    path = pdir / f"web-{i}.jpg"
                    path.write_bytes(r.content)
                    cands.append({"path": path, "source": f"web image {web.domain(im.get('source') or im['url'])}", "weight": 1})
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
    cands, titles, overrides, _ = gather(record, mid, tg, corrections, duration, log=log)
    record = names.learn_from_titles(record, titles)
    for k in ["title_am", "title_latin", "artist_am", "artist_latin"]:
        if overrides.get(k):
            record[k] = overrides[k]
    record = names.finalize(record)
    timings["lyrics_search"] = round(time.time() - t, 1)

    # 5. Verify + align.
    t = time.time()
    best, cand_reports = choose_lyrics(cands, draft_text, logp, frame_s, vocab, energy)
    review = {"needs_review": False, "reasons": []}
    if best:
        _, chosen, aligned, align_report, lines = best
        stanzas = chosen["stanzas"]
        flat = [word for line in lines for word in line]
        words = [{**w, "word": flat[w["word_index"]]} for w in aligned]
        words = smooth_words(words)
        chorus = chorus_lines_of(stanzas, aligned)
        lyrics_source = f"{chosen['source']} {chosen['url']}".strip()
        if not chosen["trusted"] and align_report["matched_share"] < 0.75:
            review["reasons"].append(f"Only {align_report['matched_share']:.0%} of the singing matched the found lyrics; "
                                     "watch for missing lines.")
        mode = "lyrics-aligned"
    else:
        stanzas, chorus, align_report = [], set(), {}
        words = [{**w, "line": None} for w in draft]
        lyrics_source = "automatic transcription (no verified lyrics found)"
        review["reasons"].append("No published lyrics could be verified for this song — the words are an automatic "
                                 "transcription. Fix them in corrections/" + str(mid) + ".txt and re-run.")
        mode = "transcription"
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
    write_ass(cues, work / "lyrics.ass")
    write_srt(cues, out / "captions" / "am.srt")
    write_srt(cues, out / "captions" / "am-Latn.srt", romanized=True)
    shutil.copy2(work / "lyrics.ass", out / "lyrics.ass")
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
    for item in ranked[:4]:
        if not item.get("same_person", True):
            continue
        img = cv2.imread(str(item["path"]))
        cut = thumbnail.cutout(img, item["face"])
        if cut is not None:
            person, photo_img, photo_source = cut, img, item["source"]
            badge_photo = face_square(img, item["face"])
            break
    if person is None and ranked:
        img = cv2.imread(str(ranked[0]["path"]))
        photo_img, badge_photo, photo_source = img, face_square(img, ranked[0]["face"]), ranked[0]["source"]
    if badge_photo is None:
        review["reasons"].append("No photo of the singer was found; thumbnails use a text design. Add one under "
                                 "artists/<name>.jpg to always use your own picture.")
    timings["photos"] = round(time.time() - t, 1)

    # 8. Backgrounds.
    t = time.time()
    lyric_text = text + " " + draft_text
    clips, clip_records = backgrounds.footage(lyric_text, work / "footage", used=used, wanted=6, seed=mid, log=log)
    scenes = backgrounds.local_scenes(work)
    sources = clips + (scenes[:2] if len(clips) >= 3 else scenes)
    if photo_img is not None:
        p = work / "photo_bg.jpg"
        cv2.imwrite(str(p), photo_img)
        sources.append(Source("image", str(p), label="artist", blur=41))
    timings["footage"] = round(time.time() - t, 1)

    # 9. Thumbnails.
    meta = {k: record.get(k, "") for k in ["title_am", "title_latin", "artist_am", "artist_latin"]}
    bg_img = None
    if clips:
        cap = cv2.VideoCapture(clips[0].path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4))
        ok, bg_img = cap.read()
        cap.release()
        bg_img = bg_img if ok else None
    if bg_img is None:
        bg_img = cv2.imread(scenes[mid % len(scenes)].path) if scenes else None
    for variant, name in [("a", "thumbnail.jpg"), ("b", "thumbnail-b.jpg"), ("c", "thumbnail-c.jpg")]:
        img = thumbnail.compose(meta, background=bg_img, person=person, photo=photo_img, variant=variant, seed=mid)
        thumbnail.save(img, out / name)

    # 10. Video.
    t = time.time()
    width = height * 16 // 9
    feats = analyze(master, fps=fps)
    renderer = Renderer(width, height, fps, duration, feats, sources, cues, meta, artist_photo=badge_photo, seed=mid)
    render_video(renderer, master, work / "lyrics.ass", out / "video.mp4", preset=preset, log=log)
    timings["render"] = round(time.time() - t, 1)

    # 11. Upload kit + diagnostics.
    credits = [c["credit"] for c in clip_records]
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
