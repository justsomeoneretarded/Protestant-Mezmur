"""Conservative failure checks, not a claim that the remaining lyrics are correct."""
from collections import Counter


class TranscriptionQualityError(ValueError):
    pass


def repeated_character_word(text):
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and Counter(letters).most_common(1)[0][1] / len(letters) >= 0.75


def assess(words):
    tokens = [w["word"].strip() for w in words if w["word"].strip()]
    letters = [c for token in tokens for c in token if c.isalpha()]
    reasons = []
    character_share = Counter(letters).most_common(1)[0][1] / len(letters) if letters else 1.0
    word_share = Counter(tokens).most_common(1)[0][1] / len(tokens) if tokens else 1.0
    if not tokens:
        reasons.append("No clear, confident vocal words were detected")
    if len(letters) >= 20 and character_share > 0.45:
        reasons.append("Transcript collapsed into a repeated character")
    if len(tokens) >= 20 and word_share > 0.65:
        reasons.append("Transcript collapsed into one repeated word")
    if len(tokens) >= 10 and sum(repeated_character_word(t) for t in tokens) / len(tokens) > 0.2:
        reasons.append("Too many words contain repeated-character noise")
    return {"passed": not reasons, "reasons": reasons, "word_count": len(tokens),
            "dominant_character_share": character_share, "dominant_word_share": word_share,
            "first_lyric_seconds": words[0]["start"] if words else None,
            "human_review_required": True}
