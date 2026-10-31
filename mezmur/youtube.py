"""YouTube upload kit: titles, description with chapters, tags, checklist."""
import json
import re
from pathlib import Path

from .geez import romanize

GENERIC_TAGS = ["amharic mezmur", "protestant mezmur", "ethiopian gospel song", "amharic worship song",
                "mezmur lyrics", "amharic mezmur lyrics", "new amharic mezmur", "ethiopian protestant mezmur",
                "የፕሮቴስታንት መዝሙር", "አዲስ መዝሙር", "የአምልኮ መዝሙር", "መዝሙር ግጥም"]


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _fit(parts, limit=100):
    """Join title parts, dropping optional tail parts until it fits."""
    parts = [p for p in parts if p]
    while parts:
        text = " ".join(" ".join(parts).split())
        if len(text) <= limit:
            return text
        parts = parts[:-1]
    return ""


def titles(m):
    """First option follows the common search pattern on YouTube:
    'Addisalem Assefa siwed geta ሲወድ ጌታ አዲስዓለም አሰፋ lyrics'."""
    t_am, t_lat, a_am, a_lat = m.get("title_am", ""), m.get("title_latin", ""), m.get("artist_am", ""), m.get("artist_latin", "")
    opts = [_fit([a_lat, t_lat.lower() if t_lat else "", t_am, a_am, "lyrics", "| Amharic Mezmur"]),
            _fit([f"{t_am} - {a_am}" if t_am and a_am else (t_am or t_lat),
                  f"| {t_lat} - {a_lat}" if t_lat and a_lat else "", "| Mezmur Lyrics ግጥም"]),
            _fit([f"{a_lat} - {t_lat}" if a_lat and t_lat else (t_lat or t_am), f"({t_am})" if t_am and t_lat else "",
                  a_am, "| New Amharic Protestant Mezmur Lyrics"])]
    out = []
    for o in opts:
        if o and o not in out:
            out.append(o)
    return out


def sections(cues, duration, chorus_lines, max_chapters=10):
    """Chapters from the sung structure. Same verse sung twice keeps its number; back-to-back
    blocks with the same label are merged; YouTube needs 0:00 first, >= 3 chapters, each >= 10 s."""
    if not cues:
        return []
    chorus_lines = set(chorus_lines or ())
    cues = [{**c, "chorus": bool(c.get("chorus")) or c.get("line") in chorus_lines} for c in cues]
    blocks = []
    for cue in cues:
        if blocks and cue["start"] - blocks[-1][-1]["end"] <= 3.5:
            blocks[-1].append(cue)
        else:
            blocks.append([cue])
    # Split blocks where the chorus starts or stops inside them.
    parts = []
    for block in blocks:
        cur = [block[0]]
        for cue in block[1:]:
            if bool(cue.get("chorus")) != bool(cur[-1].get("chorus")) and len(cur) >= 2:
                parts.append(cur)
                cur = [cue]
            else:
                cur.append(cue)
        parts.append(cur)
    verses = {}
    raw = [{"start": 0.0, "label": "Intro"}]
    prev_end = 0.0
    for part in parts:
        if part[0]["start"] - prev_end > 12 and raw[-1]["label"] != "Intro":
            raw.append({"start": prev_end + 0.5, "label": "Instrumental • ሙዚቃ"})
        if sum(1 for c in part if c.get("chorus")) > len(part) / 2:
            label = "Chorus • አዝማች"
        else:
            key = frozenset(c.get("line") for c in part if c.get("line") is not None) or frozenset([id(part)])
            match = next((n for k, n in verses.items() if len(k & key) >= max(1, len(key) // 2)), None)
            if match is None:
                match = len(verses) + 1
                verses[key] = match
            label = f"Verse {match} • ስንኝ {match}"
        if raw[-1]["label"] != label:
            raw.append({"start": max(prev_end, part[0]["start"] - 0.8), "label": label})
        prev_end = part[-1]["end"]
    if duration - prev_end > 12:
        raw.append({"start": prev_end + 0.5, "label": "Outro"})
    merged = []
    for c in raw:
        if merged and (c["start"] - merged[-1]["start"] < 10 or c["label"] == merged[-1]["label"]):
            continue
        merged.append(c)
    while len(merged) > 1 and duration - merged[-1]["start"] < 10:
        merged.pop()
    while len(merged) > max_chapters:  # drop the shortest chapter (never the first)
        lengths = [(merged[i + 1]["start"] if i + 1 < len(merged) else duration) - merged[i]["start"]
                   for i in range(len(merged))]
        i = min(range(1, len(merged)), key=lambda k: lengths[k])
        merged.pop(i)
        merged = [c for j, c in enumerate(merged) if j == 0 or c["label"] != merged[j - 1]["label"]]
    if not merged:
        return []
    renumber = {}
    for c in merged:  # number the verses that survived, in order of first appearance
        m = re.match(r"Verse (\d+)", c["label"])
        if m:
            n = renumber.setdefault(m.group(1), len(renumber) + 1)
            c["label"] = f"Verse {n} • ስንኝ {n}"
    merged[0]["start"] = 0.0
    return merged if len(merged) >= 3 else []


def tags(m, album=""):
    t_am, t_lat, a_am, a_lat = m.get("title_am", ""), m.get("title_latin", ""), m.get("artist_am", ""), m.get("artist_latin", "")
    specific = [t_am, t_lat, a_am, a_lat, f"{t_lat} lyrics" if t_lat else "", f"{t_am} ግጥም" if t_am else "",
                f"{a_lat} {t_lat}" if a_lat and t_lat else "", f"{a_am} {t_am}" if a_am and t_am else "",
                f"{a_lat} mezmur" if a_lat else "", f"{a_lat} new song" if a_lat else "", f"{a_am} መዝሙር" if a_am else "",
                album, f"{t_lat} {a_lat} lyrics" if t_lat and a_lat else ""]
    out, total = [], 0
    for tag in specific + GENERIC_TAGS:
        tag = " ".join(re.sub(r"[<>,\"]", " ", tag or "").split())
        if not tag or tag.lower() in {t.lower() for t in out}:
            continue
        cost = len(tag) + (2 if " " in tag else 0) + (1 if out else 0)
        if total + cost > 480:
            continue
        out.append(tag)
        total += cost
    return out


def hashtag(text):
    """#NameLikeThis from a singer or song name, without extra words ('Official', 'Lyrics', 'Mezmur'...)."""
    from .names import NAME_FILLER
    words = [w for w in re.findall(r"[A-Za-z0-9ሀ-፿]+", text or "") if w.lower() not in NAME_FILLER]
    return "#" + "".join(w[:1].upper() + w[1:] for w in words) if words else ""


def description(m, stanzas, cues, chapters, credits, lyrics_source):
    """YouTube description: title in both scripts, lyrics (Ge'ez + Latin letters), credits, hashtags.
    No chapters; credits are the song, the singer, and the photo/footage used.)"""
    t_am, t_lat, a_am, a_lat = m.get("title_am", ""), m.get("title_latin", ""), m.get("artist_am", ""), m.get("artist_latin", "")
    head = " | ".join(p for p in [" - ".join(x for x in [t_am, a_am] if x), " - ".join(x for x in [t_lat, a_lat] if x)] if p)
    lines = [f"{head} (Lyrics • ግጥም)", ""]
    lines.append(f"🎵 {t_am or t_lat}{f' ({t_lat})' if t_am and t_lat else ''} by {a_am or a_lat}"
                 f"{f' ({a_lat})' if a_am and a_lat else ''}")
    if m.get("album"):
        lines.append(f"💿 Album: {m['album']}")
    lines.append("")
    am_text = "\n\n".join("\n".join(s["lines"]) for s in stanzas) if stanzas else "\n".join(c["text"] for c in cues)
    rom_text = "\n\n".join("\n".join(romanize(l) for l in s["lines"]) for s in stanzas) if stanzas else \
        "\n".join(c["romanized"] for c in cues)
    lines += ["📝 ግጥም — Amharic lyrics", am_text, ""]
    rom_block = ["🔤 Lyrics", rom_text, ""]
    tail = ["🙏 Credits",
            f"Song: {t_am or t_lat}{f' / {t_lat}' if t_am and t_lat else ''}",
            f"Singer: {a_am or a_lat}{f' / {a_lat}' if a_am and a_lat else ''}"]
    tail += [c if c.startswith("Photo:") else f"Background: {c}" for c in credits[:8]]
    tail.append("")
    tags = ["#AmharicMezmur", "#ProtestantMezmur", "#MezmurLyrics", hashtag(a_lat or a_am), hashtag(t_lat or t_am)]
    tags_line = " ".join(dict.fromkeys(t for t in tags if t and t != "#"))
    for with_roman in (True, False):
        body = lines + (rom_block if with_roman else []) + tail + [tags_line]
        text = "\n".join(body)
        if len(text) <= 4900:
            return text
    return text[:4900]


def write_kit(folder, m, stanzas, cues, duration, chorus_lines, credits, lyrics_source, outro_start, review):
    folder = Path(folder)
    chapters = sections(cues, duration, chorus_lines)
    ts = titles(m)
    tg = tags(m, m.get("album", ""))
    desc = description(m, stanzas, cues, chapters, credits, lyrics_source)
    (folder / "description.txt").write_text(desc + "\n", encoding="utf-8")
    (folder / "title.txt").write_text("\n".join(ts) + "\n", encoding="utf-8")
    (folder / "tags.txt").write_text(", ".join(tg) + "\n", encoding="utf-8")
    name = m.get("title_am") or m.get("title_latin", "")
    singer = m.get("artist_am") or m.get("artist_latin", "")
    (folder / "pinned_comment.txt").write_text(
        f"🙏 «{name}» — {singer}\n"
        "ከዚህ መዝሙር የትኛው ስንኝ ልብዎን ነካው? በኮሜንት ይጻፉልን 👇\n"
        "Which line touched you the most? Write it below 👇\n"
        "ለተጨማሪ የመዝሙር ግጥሞች ሰብስክራይብ ያድርጉ 🔔\n", encoding="utf-8")
    kit = {"titles": ts, "recommended_title": ts[0] if ts else "", "description_file": "description.txt",
           "tags": tg, "category": "Music (10)", "language": "Amharic", "chapters": chapters,
           "thumbnails": ["thumbnail.jpg", "thumbnail-b.jpg", "thumbnail-c.jpg", "thumbnail-d.jpg"],
           "captions": [{"file": "captions/am.srt", "language": "Amharic"},
                        {"file": "captions/am-Latn.srt", "language": "Amharic (romanized, optional)"}],
           "end_screen_from": clock(outro_start) if outro_start < duration - 5 else None,
           "made_for_kids": False, "review": review}
    (folder / "youtube.json").write_text(json.dumps(kit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checklist = [f"# Upload: {ts[0] if ts else ''}", "",
                 "1. Upload `video.mp4` in YouTube Studio.",
                 "2. Title — pick one from `title.txt` (first is recommended).",
                 "3. Description — paste `description.txt`.",
                 "4. Thumbnail — four options: `thumbnail.jpg`, `thumbnail-b.jpg`, `thumbnail-c.jpg`, `thumbnail-d.jpg`.",
                 "   A: centered text + frame; B: circular portrait + frame; C: text without frame; D: portrait without frame.",
                 "   Choose up to three for YouTube's *Test & Compare*.",
                 "5. Tags — paste `tags.txt` (Show more → Tags).",
                 "6. Category: Music · Video language: Amharic · Not made for kids.",
                 "   After publishing, post and pin `pinned_comment.txt`.",
                 "7. Subtitles → Upload file → With timing → `captions/am.srt` (Amharic).",
                 f"8. End screen: add elements from {clock(outro_start)} (the video leaves that area clean)."
                 if kit["end_screen_from"] else "8. End screen: the song ends without a long outro; use a 5–10 s end screen.",
                 ""]
    if review.get("needs_review"):
        checklist += ["## ⚠ Check before publishing", *[f"- {r}" for r in review.get("reasons", [])], ""]
    (folder / "UPLOAD.md").write_text("\n".join(checklist), encoding="utf-8")
    return kit
