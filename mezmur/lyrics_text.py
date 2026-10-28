"""Parse, clean and compare lyric texts from heterogeneous sources."""
import html
import re
from html.parser import HTMLParser

from .geez import ETHIOPIC, LATIN, clean, ethiopic_share, fold

CHORUS = re.compile(r"^\s*[\(\[]?\s*(አዝ(ማች)?|አዝማቹ|ዝማሬ|chorus|ref(rain)?|ኮረስ)\s*[\)\]:.\-–—፦፥፡]*\s*", re.I)
REPEAT = re.compile(r"[\(\[]?\s*(?:[x×*]\s*[\d፩-፼]+|[\d፩-፼]+\s*[x×*]|[\d፩-፼]+\s*ጊዜ|ሁለት ጊዜ|ሶስት ጊዜ|ሦስት ጊዜ|"
                    r"/+\s*[\d፩-፼]+|//+|\\\\+)\s*[\)\]]?\s*$", re.I)
GEEZ_DIGITS = {"፩": 1, "፪": 2, "፫": 3, "፬": 4, "፭": 5, "፮": 6, "፯": 7, "፰": 8, "፱": 9}


def repeat_count(marker):
    """How many times a line is sung, from its trailing marker ('/3', 'x2', '3 ጊዜ', '//', ...)."""
    if not marker:
        return 1
    m = re.search(r"\d+", marker)
    if m:
        return max(1, min(8, int(m.group(0))))
    for ch in marker:
        if ch in GEEZ_DIGITS:
            return GEEZ_DIGITS[ch]
    if "ሶስት" in marker or "ሦስት" in marker:
        return 3
    return 2  # '//', '\\', 'ሁለት ጊዜ'

NOISE = re.compile(r"(https?://\S+|www\.\S+|@\w+|#\S+|t\.me/\S+)", re.I)
JUNK_LINE = re.compile(
    r"(join|subscribe|channel|telegram|youtube|facebook|tiktok|instagram|whats ?app|twitter|follow|share|like|"
    r"comment|inbox|contact|link|lyrics by|credit|copyright|©|all rights|for more|visit|click|download|"
    r"ጆይን|ሰብስክራይብ|ሰብስክራይፕ|ቻናል|ቴሌግራም|ቴሌግራማችን|ዩቲዩብ|ዩትዩብ|ፌስቡክ|ቲክቶክ|ኢንስታግራም|ሊንክ|ሼር|ላይክ|ኮሜንት|"
    r"ፎሎው|ተቀላቀሉ|ይቀላቀሉ|ተቀላቀሉን|ይቀላቀሉን|ይጎብኙ|ለተጨማሪ|ለበለጠ|ይከታተሉ|ተከታተሉ|ያጋሩ|አጋሩ|"
    r"ግጥም\s*[:፡]|ዜማ\s*[:፡]|ዘማሪ\s*[:፡]|ዘማሪት\s*[:፡]|ቅንብር|ሙዚቃ\s*[:፡]|ስቱዲዮ)", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
VERSE_NO = re.compile(r"^\s*(\d+|[፩-፼]+)\s*[.)\-:፡]\s*")


def balance_brackets(line):
    """Drop unmatched ( ) [ ] — sources often lose one half of an echo marker like '(በተራዬ'."""
    out, stack = list(line), []
    pairs = {")": "(", "]": "["}
    for i, ch in enumerate(line):
        if ch in "([":
            stack.append(i)
        elif ch in ")]":
            if stack and line[stack[-1]] == pairs[ch]:
                stack.pop()
            else:
                out[i] = ""
    for i in stack:
        out[i] = ""
    return re.sub(r"\(\s*\)|\[\s*\]", "", "".join(out))


def clean_line(line):
    line = html.unescape(line)
    line = EMOJI.sub(" ", line)
    line = NOISE.sub(" ", line)
    line = VERSE_NO.sub("", line)
    line = line.replace("፡", " ").replace("…", " ").replace("...", " ")
    line = re.sub(r"[\"“”«»]", "", line)
    line = balance_brackets(line)
    line = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", line))
    line = re.sub(r"[ \t]+", " ", line).strip(" -–—*•·|~_=")
    return clean(line)


def is_lyric_line(line):
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 2 and ethiopic_share(line) >= 0.6 and not JUNK_LINE.search(line)


def parse_lyrics(text, keep_markers=False):
    """Return stanzas: list of {"lines": [...], "repeats": [...], "chorus": bool}.

    Repeat markers ('/3', 'x2', '3 ጊዜ', '//') are removed from the text and kept as counts."""
    stanzas, current, counts, chorus = [], [], [], False

    def close():
        nonlocal current, counts
        if current:
            stanzas.append({"lines": current, "repeats": counts, "chorus": chorus})
        current, counts = [], []

    for raw in str(text).replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            close()
            chorus = False
            continue
        if CHORUS.match(line):
            rest = CHORUS.sub("", line).strip()
            close()
            chorus = True
            line = rest
            if not line:
                continue
        m = REPEAT.search(line)
        count = repeat_count(m.group(0)) if m else 1
        line = REPEAT.sub("", line).strip()
        line = clean_line(line)
        if not is_lyric_line(line):
            continue
        if current and fold(current[-1]) == fold(line):
            counts[-1] += 1  # the same line written twice in a row means it is sung twice
            continue
        current.append(line)
        counts.append(count)
    close()
    return [s for s in stanzas if s["lines"]]


def stanzas_text(stanzas):
    def show(s, i):
        n = (s.get("repeats") or [1] * len(s["lines"]))[i]
        return s["lines"][i] + (f" ×{n}" if n > 1 else "")
    return "\n\n".join("\n".join(show(s, i) for i in range(len(s["lines"]))) for s in stanzas) + "\n"


def line_repeats(stanzas):
    return [n for s in stanzas for n in (s.get("repeats") or [1] * len(s["lines"]))]


def line_stanzas(stanzas):
    return [k for k, s in enumerate(stanzas) for _ in s["lines"]]


def all_lines(stanzas):
    return [line for s in stanzas for line in s["lines"]]


class _TextExtractor(HTMLParser):
    BREAK = {"br", "p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article", "td"}
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "button", "select"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BREAK:
            self.parts.append("\n")
            if tag in {"p", "div", "pre", "section", "article"}:
                self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(markup):
    parser = _TextExtractor()
    try:
        parser.feed(markup)
    except Exception:
        return re.sub(r"<[^>]+>", "\n", markup)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n[ \t]+", "\n", text)


def script_texts(markup, min_ethiopic=40):
    """Decode JSON/JS string literals that carry Ge'ez text (Next.js flight data, __NEXT_DATA__, Nuxt...)."""
    import json
    out = []
    for script in re.findall(r"<script[^>]*>(.*?)</script>", markup, re.S | re.I):
        if len(ETHIOPIC.findall(script)) < min_ethiopic and "\\u12" not in script:
            continue
        for literal in re.findall(r'"((?:[^"\\]|\\.){60,})"', script):
            if len(ETHIOPIC.findall(literal)) < min_ethiopic and literal.count("\\u12") < min_ethiopic:
                continue
            text = literal
            for _ in range(3):  # strings may be JSON-encoded inside other JSON strings
                try:
                    decoded = json.loads(f'"{text}"')
                except ValueError:
                    break
                if decoded == text:
                    break
                text = decoded
            if text.count("\\n") > 3 and "\n" not in text:
                text = text.replace("\\n", "\n")
            out.append(text)
    return out


def extract_blocks(text, min_lines=4):
    """Find runs of lyric-looking lines inside arbitrary text (pages, posts, descriptions)."""
    blocks, current, blank_run = [], [], 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            blank_run += 1
            if blank_run >= 4 and current:
                blocks.append(current)
                current = []
            elif current:
                current.append("")
            continue
        blank_run = 0
        candidate = clean_line(REPEAT.sub("", CHORUS.sub("", line)))
        if is_lyric_line(candidate) or (CHORUS.match(line) and current):
            current.append(line)
        else:
            if sum(1 for l in current if l) >= min_lines:
                blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    out = []
    for block in blocks:
        body = "\n".join(block).strip()
        stanzas = parse_lyrics(body)
        # A lone short first line is usually the page heading (song title), not a lyric.
        if len(stanzas) > 1 and len(stanzas[0]["lines"]) == 1 and len(stanzas[0]["lines"][0].split()) <= 4:
            stanzas = stanzas[1:]
        if sum(len(s["lines"]) for s in stanzas) >= min_lines:
            out.append(stanzas)
    return out


def trigrams(text):
    t = fold(text).replace(" ", "_")
    return {t[i:i + 3] for i in range(len(t) - 2)}


def word_set(text):
    return [w for w in fold(text).split() if len(w) >= 2]


def edit_distance(a, b, limit=None):
    if abs(len(a) - len(b)) > (limit if limit is not None else 1e9):
        return (limit or 0) + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(candidate_text, draft_text):
    """How well a candidate lyric explains the (noisy) recognizer draft. 0..1 scores."""
    c3, d3 = trigrams(candidate_text), trigrams(draft_text)
    if not c3 or not d3:
        return {"trigram_recall": 0.0, "trigram_precision": 0.0, "word_recall": 0.0, "score": 0.0}
    inter = len(c3 & d3)
    recall = inter / len(d3)
    precision = inter / len(c3)
    cand_words = set(word_set(candidate_text))
    draft_words = [w for w in word_set(draft_text) if len(w) >= 3]
    hits = 0
    for w in draft_words:
        if w in cand_words or any(abs(len(w) - len(c)) <= 1 and edit_distance(w, c, 1) <= 1 for c in cand_words):
            hits += 1
    word_recall = hits / len(draft_words) if draft_words else 0.0
    score = 0.45 * recall + 0.35 * word_recall + 0.20 * precision
    return {"trigram_recall": round(recall, 3), "trigram_precision": round(precision, 3),
            "word_recall": round(word_recall, 3), "score": round(score, 3)}


def title_overlap(title, text):
    t = set(word_set(title))
    if not t:
        return 0.0
    words = set(word_set(text))
    return sum(1 for w in t if w in words) / len(t)


def has_latin_lyrics(text):
    lines = [l for l in text.splitlines() if l.strip()]
    latin = [l for l in lines if len(LATIN.findall(l)) > 8 and not ETHIOPIC.search(l)]
    return len(latin) >= max(4, len(lines) // 2)


def strip_header(stanzas, record):
    """Remove leading 'Title — Singer' / 'ዘማሪ ...' header lines that lyric pages put above the text."""
    import re as _re
    names = [record.get(k, "") for k in ("artist_am", "artist_latin")]
    title = fold(record.get("title_am", "") or "")
    keys = [fold(n) for n in names if n]
    # Ethiopic spellings vary (ይስሐቅ/ይስሃቅ, ሰድቅ/ሰዲቅ): compare the first 2 letters of each name word.
    stems = [w[:2] for k in keys for w in k.split() if len(w) >= 2]
    out = [dict(st, lines=list(st["lines"]), repeats=list(st.get("repeats") or [1] * len(st["lines"])))
           for st in stanzas]
    if not out:
        return out
    first = out[0]["lines"]
    while first:
        line = first[0]
        f = fold(line)
        words = f.split()
        named = sum(1 for st in stems if any(w.startswith(st) for w in words)) >= 2
        sep = bool(_re.search(r"-{2,}|—|–|\||//", line))
        only_title = title and f == title
        if (named and (sep or len(words) <= 6)) or (sep and title and title in f) or only_title and len(first) > 4:
            first.pop(0)
            out[0]["repeats"].pop(0)
        else:
            break
    return [st for st in out if st["lines"]]

