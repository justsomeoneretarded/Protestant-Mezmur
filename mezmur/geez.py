"""Ge'ez (Ethiopic) script helpers: detection, spelling-variant folding and romanization.

Romanization follows the informal style used in Amharic YouTube titles
("ሲወድ ጌታ" -> "siwed geta", "ይስሐቅ ሰድቅ" -> "yishak sedik"), not a scholarly scheme.
"""
import re
import unicodedata

BASE = 0x1200
# One consonant per 8-syllable row, starting at U+1200.
ROWS = [
    "h", "l", "h", "m", "s", "r", "s", "sh", "k", "kW", "k", "kW", "b", "v", "t", "ch",
    "h", "hW", "n", "gn", "", "k", "kW", "h", "hW", "w", "", "z", "zh", "y", "d", "d",
    "j", "g", "gW", "g", "t", "ch", "p", "ts", "ts", "f", "p",
]
VOWELS = ["e", "u", "i", "a", "e", "", "o", "wa"]
LABIAL_VOWELS = {0: "we", 2: "wi", 3: "wa", 4: "we", 5: "u"}
GLOTTAL_VOWELS = ["a", "u", "i", "a", "e", "e", "o", "e"]
EXTRA = {0x1358: "mya", 0x1359: "rya", 0x135A: "fya"}
PUNCT = {"፡": " ", "።": ".", "፣": ",", "፤": ";", "፥": ":", "፦": ":", "፧": "?", "፨": " ", "፠": " "}
DIGITS = {chr(0x1369 + i): str(i + 1) for i in range(9)}
ETHIOPIC = re.compile(r"[ሀ-፿ᎀ-᎟ⶀ-⷟꬀-꬯]")
LATIN = re.compile(r"[A-Za-z]")

# Rows that are pronounced identically in modern Amharic: fold to one spelling for matching.
FOLD_ROWS = {2: 0, 16: 0, 23: 0, 4: 6, 26: 20, 40: 39}


def is_syllable(ch):
    return BASE <= ord(ch) < BASE + 8 * len(ROWS)


def split(ch):
    """Return (row, order) for an Ethiopic syllable."""
    offset = ord(ch) - BASE
    return offset // 8, offset % 8


def ethiopic_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum(bool(ETHIOPIC.match(c)) for c in letters) / len(letters) if letters else 0.0


def has_ethiopic(text, minimum=2):
    return len(ETHIOPIC.findall(text or "")) >= minimum


def fold_char(ch):
    if not is_syllable(ch):
        return ch
    row, order = split(ch)
    row = FOLD_ROWS.get(row, row)
    if row in (0, 20) and order == 3:  # ሃ == ሀ, ኣ == አ
        order = 0
    if row == 20 and order == 7:  # ኧ
        order = 0
    return chr(BASE + row * 8 + order)


def clean(text):
    """Normalize punctuation/spacing of lyric text without changing spelling."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("፡", " ").replace("​", "").replace("﻿", "")
    return re.sub(r"[ \t ]+", " ", text).strip()


def fold(text):
    """Matching key: folded homophones, no punctuation, single spaces, lowercase Latin."""
    text = clean(text).lower()
    out = []
    for ch in text:
        if ch in PUNCT:
            out.append(" ")
        elif is_syllable(ch):
            out.append(fold_char(ch))
        elif ch.isalnum():
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def _syllable(ch):
    code = ord(ch)
    if code in EXTRA:
        return "C", EXTRA[code]
    row, order = split(ch)
    consonant = ROWS[row]
    if row in (20, 26):
        return ("V5" if order == 5 else "V"), GLOTTAL_VOWELS[order]
    if consonant.endswith("W"):
        base = consonant[:-1]
        return ("C5" if order == 5 else "C"), base + LABIAL_VOWELS.get(order, "wa")
    if order == 5:
        return "C5", consonant
    if consonant == "h" and order == 0:  # ሀ/ሐ/ኀ are pronounced "ha"
        return "C", "ha"
    return "C", consonant + VOWELS[order]


def romanize_word(word):
    parts = []
    chars = [c for c in word if is_syllable(c) or ord(c) in EXTRA]
    for index, ch in enumerate(chars):
        kind, sound = _syllable(ch)
        if kind == "C5":
            final = index == len(chars) - 1
            following_final_sixth = (
                index == len(chars) - 2 and _syllable(chars[index + 1])[0] == "C5"
            )
            if index == 0 and not final:
                sound += "u" if sound == "w" else "i"
            elif following_final_sixth:
                sound += "i"
        parts.append(sound)
    text = "".join(parts)
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def romanize(text, title_case=False):
    """Romanize Ethiopic runs; Latin text and digits pass through unchanged."""
    text = clean(text)
    for k, v in PUNCT.items():
        text = text.replace(k, v)
    for k, v in DIGITS.items():
        text = text.replace(k, v)
    tokens = re.split(r"(\s+)", text)
    out = []
    for token in tokens:
        if not token or token.isspace():
            out.append(token)
            continue
        pieces = re.split(r"([ሀ-፿]+)", token)
        word = "".join(romanize_word(p) if p and ETHIOPIC.match(p[0]) else p for p in pieces)
        out.append(word)
    result = re.sub(r"\s+", " ", "".join(out)).strip()
    result = re.sub(r"\s+([.,;:?!])", r"\1", result)
    if title_case:
        result = " ".join(w[:1].upper() + w[1:] for w in result.split(" "))
    return result


def latin_key(text):
    """Loose phonetic key for comparing romanized Amharic spellings (Kebur ~ kibur, Gieta ~ geta)."""
    text = romanize(text).lower()
    text = re.sub(r"[^a-z ]", "", text)
    for a, b in [("ph", "f"), ("q", "k"), ("ie", "e"), ("ee", "e"), ("oo", "u"), ("c", "ch"),
                 ("chh", "ch"), ("x", "ks"), ("v", "b")]:
        text = text.replace(a, b)
    text = re.sub(r"(.)\1+", r"\1", text)
    return " ".join(text.split())


def skeleton(text):
    """Consonant skeleton (vowels removed) of the Latin key."""
    return " ".join(w for w in (re.sub(r"[aeiou]", "", w) for w in latin_key(text).split()) if w)
