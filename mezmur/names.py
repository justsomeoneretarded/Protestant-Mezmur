"""Resolve song title and artist in both Ge'ez script and romanized Latin."""
import re
from difflib import SequenceMatcher
from pathlib import Path

import yaml

from .geez import ETHIOPIC, ethiopic_share, fold, has_ethiopic, latin_key, romanize, skeleton

SEPARATORS = re.compile(r"\s*(?://+|\|+|\s[-–—]\s|_{1,}|\s*[•·]\s*|\[|\]|\(|\)|\{|\}|:)\s*")
JUNK = re.compile(
    r"\b(official|music|video|audio|lyrics?|lyric|new|mezmur|mezmure|song|songs|protestant|amharic|ethiopian|"
    r"gospel|worship|live|hd|4k|mp3|m4a|full|album|cover|clip|retouched|by|remix|instrumental|karaoke|"
    r"\d{2,3}k?bps|\(\d+\)|official\s+channel)\b", re.I)
AM_JUNK = re.compile(r"(አዲስ\s+መዝሙር|መዝሙር|ግጥም|ዘማሪት?|ዘማሪ|ሙሉ\s+አልበም|አልበም|ቪዲዮ|ኦፊሻል)")
LEADING_NUMBER = re.compile(r"^\s*\d{1,3}\s*[.)\-_]\s*")
YEAR = re.compile(r"\b(19|20)\d{2}\b|\b\d{4}\b")


def similar(a, b):
    """Similarity of two names that may be in different scripts or spellings (0..1)."""
    ka, kb = latin_key(a), latin_key(b)
    if not ka or not kb:
        return 0.0
    full = SequenceMatcher(None, ka, kb).ratio()
    sa, sb = skeleton(a), skeleton(b)
    skel = SequenceMatcher(None, sa, sb).ratio() if sa and sb else 0.0
    return max(full, 0.9 * skel)


NAME_FILLER = {"official", "music", "mezmur", "mezmurs", "gospel", "worship", "singer", "tv", "channel", "media",
               "production", "productions", "records", "entertainment", "ministry", "ministries", "lyrics", "topic",
               "song", "songs", "new", "hd", "the", "and", "&", "-", "|", "ዘማሪ", "ዘማሪት", "መዝሙር", "መዝሙሮች"}


def _word_match(a, b):
    """Same name word, allowing spelling variants and short forms (Dagi ~ Dagmawi, Yishak ~ Yisihak)."""
    if SequenceMatcher(None, a, b).ratio() >= 0.8:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 3 and long_.startswith(short[:3]) and len(short) <= 5


def same_person_name(candidate, *names):
    """True when ``candidate`` mentions this artist: every word of one of ``names`` appears in it (any
    script, spelling variants and short forms allowed). Extra words are fine ('Dagmawi Tilahun Official',
    'Dagi Tilahun - Topic', 'Dagi Tilahun | New Mezmur'); a different person sharing one name is not
    ('Tilahun Gessesse' is not 'Dagi Tilahun')."""
    cand_words = [w for w in re.split(r"[\s/|,()\-–—:]+", str(candidate)) if w]
    cand_keys = [latin_key(w) for w in cand_words if w.lower() not in NAME_FILLER and latin_key(w)]
    for name in names:
        words = [latin_key(w) for w in str(name or "").split() if latin_key(w) and w.lower() not in NAME_FILLER]
        if not words:
            continue
        used, ok = set(), True
        for w in words:
            hit = next((i for i, c in enumerate(cand_keys) if i not in used and _word_match(w, c)), None)
            if hit is None:
                ok = False
                break
            used.add(hit)
        if ok:
            return True
    return False


def strip_junk(text):
    text = re.sub(r"https?://\S+|t\.me/\S+|@\w+|#\w+", " ", str(text))
    text = re.sub(r"(?i)\b(join|subscribe|follow)\b.*$", " ", text)  # 'Join us ...' tails in captions
    text = re.sub(r"(ቻናላችንን|ቴሌግራም|ይቀላቀሉ|ተቀላቀሉ|ሰብስክራይብ).*$", " ", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿️]", " ", text)
    text = LEADING_NUMBER.sub("", text)
    text = re.sub(r"\.(mp3|m4a|ogg|opus|wav|flac|aac)$", "", text, flags=re.I)
    text = YEAR.sub(" ", text)
    if has_ethiopic(text):
        text = AM_JUNK.sub(" ", text)
    text = JUNK.sub(" ", text)
    return " ".join(text.replace("_", " ").split()).strip(" -–—.,/|:")


def pieces(*texts):
    out = []
    for text in texts:
        if not text:
            continue
        for part in SEPARATORS.split(str(text)):
            part = strip_junk(part)
            if (len(re.findall(r"[^\W\d_]", part)) >= 2
                    and part.lower() not in {p.lower() for p in out}):
                out.append(part)
    return out


def load_artists(path=None):
    path = Path(path or Path(__file__).resolve().parent.parent / "data" / "artists.yaml")
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("artists", [])


def known_artist(name, artists):
    best, score = None, 0.0
    for artist in artists:
        for alias in [artist.get("latin", ""), artist.get("amharic", "")] + artist.get("aliases", []):
            if alias:
                s = 1.0 if fold(alias) == fold(name) else similar(alias, name)
                if s > score:
                    best, score = artist, s
    return best if score >= 0.86 else None


def resolve(info, artists=None):
    """Combine Telegram/ID3 fields into a structured name record.

    info keys used: title, performer, caption, filename, tags (dict of ID3/MP4 tags).
    """
    artists = load_artists() if artists is None else artists
    tags = info.get("tags") or {}
    caption_lines = [l for l in (info.get("caption") or "").splitlines() if l.strip()]
    singer_hint = ""
    for line in caption_lines:
        m = re.search(r"ዘማሪት?\s*[:፡]?\s*([ሀ-፿ ]{3,40})", line)
        if m:
            singer_hint = strip_junk(m.group(1))
    performer = strip_junk(info.get("performer") or tags.get("artist") or tags.get("album_artist") or "")
    parts = pieces(info.get("title"), info.get("filename"), *caption_lines[:3])
    record = {"title_am": "", "title_latin": "", "artist_am": "", "artist_latin": "", "evidence": parts}

    artist_names = [n for n in [performer, singer_hint] if n]
    if not artist_names:
        artist_names = [p for p in parts if known_artist(p, artists)][:1]
    for part in parts:
        for name in list(artist_names):
            if part not in artist_names and similar(part, name) >= 0.72:
                artist_names.append(part)
    for name in artist_names:
        if has_ethiopic(name) and ethiopic_share(name) > 0.6:
            record["artist_am"] = record["artist_am"] or name
        elif name:
            record["artist_latin"] = record["artist_latin"] or name
    title_parts = [p for p in parts if p not in artist_names and not any(similar(p, a) >= 0.72 for a in artist_names)]
    am_titles = [p for p in title_parts if ethiopic_share(p) > 0.6]
    latin_titles = [p for p in title_parts if ethiopic_share(p) < 0.2]
    if am_titles:
        record["title_am"] = am_titles[0]
        match = max(latin_titles, key=lambda p: similar(p, am_titles[0]), default="")
        if match and similar(match, am_titles[0]) >= 0.5:
            record["title_latin"] = match
    elif latin_titles:
        record["title_latin"] = latin_titles[0]
    known = known_artist(record["artist_am"] or record["artist_latin"], artists) if artist_names else None
    if known:
        record["artist_am"] = known.get("amharic") or record["artist_am"]
        record["artist_latin"] = known.get("latin") or record["artist_latin"]
        record["artist_known"] = True
    return finalize(record)


def tidy_name(value):
    """Drop separators and stray words left at either end of a name ('Agegnehu Yideg | በ', '- Title')."""
    value = str(value or "")
    for _ in range(3):
        before = value
        value = re.sub(r"\s*[|/\\\-–—:,•·]+\s*\S{0,2}\s*$", "", value)          # '... | በ', '... -'
        value = re.sub(r"^\s*\S{0,2}\s*[|/\\\-–—:,•·]+\s*", "", value)          # 'በ | ...'
        value = re.sub(r"\s+(በ|by|By|ft\.?|feat\.?)\s*$", "", value)              # trailing 'by'
        if value == before:
            break
    return " ".join(value.split())


def finalize(record):
    """Fill missing romanized forms from the Ge'ez spelling; title-case Latin; tidy every name."""
    for key in ["title_am", "title_latin", "artist_am", "artist_latin"]:
        if record.get(key):
            record[key] = tidy_name(record[key])
    if record.get("title_am") and not record.get("title_latin"):
        record["title_latin"] = romanize(record["title_am"], title_case=True)
        record["title_latin_auto"] = True
    if record.get("artist_am") and not record.get("artist_latin"):
        record["artist_latin"] = romanize(record["artist_am"], title_case=True)
        record["artist_latin_auto"] = True
    for key in ["title_latin", "artist_latin"]:
        value = record.get(key, "")
        if value:
            record[key] = " ".join(w[:1].upper() + w[1:] for w in value.split())
    return record


def learn_from_titles(record, titles):
    """Fill a missing Ge'ez title/artist from titles that show both scripts (YouTube, lyrics sites)."""
    want = []
    if not record.get("title_am") and record.get("title_latin"):
        want.append(("title_am", record["title_latin"]))
    if not record.get("artist_am") and record.get("artist_latin"):
        want.append(("artist_am", record["artist_latin"]))
    if not want:
        return record
    runs = []
    for title in titles:
        for part in SEPARATORS.split(title or ""):
            part = strip_junk(part)
            for run in re.findall(r"[ሀ-፿][ሀ-፿\s፡]*[ሀ-፿]", part):
                run = " ".join(run.replace("፡", " ").split())
                if len(run) >= 2:
                    runs.append(run)
    for key, latin in want:
        scored = {}
        for run in runs:
            s = similar(run, latin)
            if s >= 0.62:
                k = fold(run)
                scored.setdefault(k, [run, 0.0, 0])
                scored[k][1] = max(scored[k][1], s)
                scored[k][2] += 1
        if scored:
            run, s, count = max(scored.values(), key=lambda x: (x[1] + 0.03 * min(x[2], 5)))
            record[key] = run
            record.setdefault("learned", {})[key] = {"from": "titles", "similarity": round(s, 2), "count": count}
    return record


def slug(text):
    text = latin_key(text) or fold(text)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"
