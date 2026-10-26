"""Parse, clean and compare lyric texts from heterogeneous sources."""
import html
import re
from html.parser import HTMLParser

from .geez import ETHIOPIC, LATIN, clean, ethiopic_share, fold

CHORUS = re.compile(r"^\s*[\(\[]?\s*(አዝ(ማች)?|አዝማቹ|ዝማሬ|chorus|ref(rain)?|ኮረስ)\s*[\)\]:.\-–—፦፥፡]*\s*", re.I)
REPEAT = re.compile(r"[\(\[]?\s*(?:[x×*]\s*[\d፩-፼]+|[\d፩-፼]+\s*[x×*]|[\d፩-፼]+\s*ጊዜ|ሁለት ጊዜ|//+|\\\\+)\s*[\)\]]?\s*$", re.I)
NOISE = re.compile(r"(https?://\S+|www\.\S+|@\w+|#\S+|t\.me/\S+)", re.I)
JUNK_LINE = re.compile(r"(join|subscribe|ሰብስክራይብ|ቻናል|channel|telegram|youtube|lyrics by|ግጥም\s*[:፡]|ዜማ\s*[:፡]|"
                       r"ዘማሪ\s*[:፡]|ቅንብር|copyright|©|share|ሼር|like|ላይክ|follow|ፎሎው|ተቀላቀሉ|ይቀላቀሉ)", re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")
VERSE_NO = re.compile(r"^\s*(\d+|[፩-፼]+)\s*[.)\-:፡]\s*")


def clean_line(line):
    line = html.unescape(line)
    line = EMOJI.sub(" ", line)
    line = NOISE.sub(" ", line)
    line = VERSE_NO.sub("", line)
    line = line.replace("፡", " ").replace("…", " ").replace("...", " ")
    line = re.sub(r"[\"“”«»]", "", line)
    line = re.sub(r"[ \t]+", " ", line).strip(" -–—*•·|~_=")
    return clean(line)


def is_lyric_line(line):
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 2 and ethiopic_share(line) >= 0.6 and not JUNK_LINE.search(line)


def parse_lyrics(text, keep_markers=False):
    """Return stanzas: list of {"lines": [...], "chorus": bool}. Repeat markers are removed."""
    stanzas, current, chorus = [], [], False
    for raw in str(text).replace("\r", "").split("\n"):
        line = raw.strip()
        if not line:
            if current:
                stanzas.append({"lines": current, "chorus": chorus})
                current, chorus = [], False
            continue
        if CHORUS.match(line):
            rest = CHORUS.sub("", line).strip()
            if current:
                stanzas.append({"lines": current, "chorus": chorus})
                current = []
            chorus = True
            line = rest
            if not line:
                continue
        line = REPEAT.sub("", line).strip()
        line = clean_line(line)
        if not is_lyric_line(line):
            continue
        if current and fold(current[-1]) == fold(line):
            continue
        current.append(line)
    if current:
        stanzas.append({"lines": current, "chorus": chorus})
    return [s for s in stanzas if s["lines"]]


def stanzas_text(stanzas):
    return "\n\n".join("\n".join(s["lines"]) for s in stanzas) + "\n"


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
