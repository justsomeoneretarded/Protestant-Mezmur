"""Lyrics-constrained CTC alignment.

Given per-frame CTC log-probabilities for the whole song and the known lyrics, find
which words are sung in each vocal phrase and exactly when. Lyrics are treated as an
ordered list of words; each phrase may continue where the previous one stopped or
jump to the start of any line (repeated choruses, skipped or re-ordered verses). A
phrase may also match nothing (ad-libs, humming, instrumental noise).
"""
from dataclasses import dataclass, field

import numpy as np

NEG = -1e30


@dataclass
class Lyrics:
    lines: list  # list[list[str]] words per line
    stanza_of_line: list = field(default_factory=list)

    @property
    def words(self):
        return [w for line in self.lines for w in line]


@dataclass
class Target:
    tokens: np.ndarray          # token ids (no blanks, no delimiters)
    token_word: np.ndarray      # word index of each token
    word_first_token: np.ndarray  # first token of each word (-1 if word has no tokens)
    word_last_token: np.ndarray
    word_line: np.ndarray
    line_first_word: np.ndarray


def build_target(lyrics: Lyrics, char_to_id: dict) -> Target:
    tokens, token_word, word_line = [], [], []
    first, last = [], []
    line_first_word = []
    w = 0
    for li, line in enumerate(lyrics.lines):
        line_first_word.append(w)
        for word in line:
            ids = [char_to_id[c] for c in word if c in char_to_id]
            first.append(len(tokens) if ids else -1)
            tokens.extend(ids)
            token_word.extend([w] * len(ids))
            last.append(len(tokens) - 1 if ids else -1)
            word_line.append(li)
            w += 1
    return Target(np.array(tokens, dtype=np.int64), np.array(token_word, dtype=np.int64),
                  np.array(first, dtype=np.int64), np.array(last, dtype=np.int64),
                  np.array(word_line, dtype=np.int64), np.array(line_first_word, dtype=np.int64))


def ctc_free_end(logp: np.ndarray, tokens: np.ndarray, blank: int, backtrace=False):
    """Viterbi CTC alignment that must start at tokens[0] but may end after any token.

    logp: [T, V] log probabilities (blank already merged with word delimiter).
    Returns per-token best final scores ``end_scores[k]`` (path ending right after token k)
    and, if requested, a function to recover token frame spans for a chosen end.
    """
    T = logp.shape[0]
    L = len(tokens)
    S = 2 * L + 1
    labels = np.full(S, blank, dtype=np.int64)
    labels[1::2] = tokens
    emit = logp[:, labels]  # [T, S]
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[3::2] = tokens[1:] != tokens[:-1]
    alpha = np.full(S, NEG)
    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]
    choice = np.zeros((T, S), dtype=np.int8) if backtrace else None
    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([NEG], alpha[:-1]))
        skip = np.concatenate(([NEG, NEG], alpha[:-2]))
        skip = np.where(skip_ok, skip, NEG)
        best = np.maximum(np.maximum(stay, step), skip)
        if backtrace:
            c = np.zeros(S, dtype=np.int8)
            c[step > stay] = 1
            c[(skip > stay) & (skip > step)] = 2
            choice[t] = c
        alpha = best + emit[t]
    token_state = np.arange(L) * 2 + 1
    end_scores = np.maximum(alpha[token_state], alpha[token_state + 1])
    end_state = np.where(alpha[token_state] >= alpha[token_state + 1], token_state, token_state + 1)

    def spans(k):
        s = int(end_state[k])
        frames = np.zeros(T, dtype=np.int64)
        for t in range(T - 1, -1, -1):
            frames[t] = s
            if t:
                s -= int(choice[t, s])
        token_spans = {}
        for t, s in enumerate(frames):
            if s % 2 == 1:
                i = s // 2
                a, b = token_spans.get(i, (t, t))
                token_spans[i] = (min(a, t), max(b, t))
        return token_spans

    return end_scores, (spans if backtrace else None)


def merge_blank(logp: np.ndarray, blank: int, delimiter):
    """Treat the word delimiter as silence: lyrics carry explicit word boundaries."""
    if delimiter is None:
        return logp
    out = logp.copy()
    out[:, blank] = np.logaddexp(logp[:, blank], logp[:, delimiter])
    return out


def phrases_from_activity(active: np.ndarray, frame_seconds: float, min_gap=0.45,
                          min_len=0.1, max_len=13.0, pad=0.12):
    """Split frame activity into phrases (start, end) frame indices."""
    T = len(active)
    gap = max(1, int(round(min_gap / frame_seconds)))
    runs, start = [], None
    silence = 0
    for t in range(T):
        if active[t]:
            if start is None:
                start = t
            silence = 0
            last = t
        elif start is not None:
            silence += 1
            if silence >= gap:
                runs.append((start, last + 1))
                start = None
    if start is not None:
        runs.append((start, last + 1))
    minimum = int(min_len / frame_seconds)
    runs = [r for r in runs if r[1] - r[0] >= minimum]
    out = []
    limit = int(max_len / frame_seconds)

    def split(a, b):
        if b - a <= limit:
            out.append((a, b))
            return
        # Cut at the quietest stretch near the middle third.
        seg = active[a:b].astype(float)
        kernel = np.ones(max(3, int(0.25 / frame_seconds)))
        density = np.convolve(seg, kernel, "same")
        lo, hi = (b - a) // 3, 2 * (b - a) // 3
        cut = a + lo + int(np.argmin(density[lo:hi]))
        split(a, cut)
        split(cut, b)

    for a, b in runs:
        split(a, b)
    p = int(round(pad / frame_seconds))
    return [(max(0, a - p), min(T, b + p)) for a, b in out]


def frame_activity(logp: np.ndarray, blank: int, delimiter, energy=None, threshold=0.5):
    """A frame is vocal when the model places real probability on any character nearby."""
    probs = np.exp(logp)
    nonblank = 1.0 - probs[:, blank] - (probs[:, delimiter] if delimiter is not None else 0)
    k = 5
    peak = np.array([nonblank[max(0, t - k):t + k + 1].max() for t in range(len(nonblank))])
    active = peak > threshold
    if energy is not None:
        active &= energy
    return active


@dataclass
class Hyp:
    score: float
    end_word: int          # next word to be sung on continuation
    trail: tuple           # ((phrase_index, start_word, end_word_inclusive) | (phrase_index, None, None), ...)


def align_lyrics(logp, frame_seconds, target: Target, blank, delimiter, energy=None,
                 jump_cost=6.0, null_cost_per_frame=1.2, beam=10, max_words=40, log=print):
    """Return (words, report). Words carry start/end seconds, line index and confidence."""
    logp = merge_blank(logp, blank, delimiter)
    active = frame_activity(logp, blank, None, energy)
    phrases = phrases_from_activity(active, frame_seconds)
    n_words = len(target.word_line)
    voiced = [w for w in range(n_words) if target.word_first_token[w] >= 0]
    line_starts = sorted({int(next((w for w in range(f, n_words) if target.word_first_token[w] >= 0), n_words))
                          for f in target.line_first_word})
    line_starts = [w for w in line_starts if w < n_words]
    greedy = logp.max(-1)
    hyps = [Hyp(0.0, voiced[0] if voiced else 0, ())]
    cache = {}
    for pi, (a, b) in enumerate(phrases):
        seg = logp[a:b]
        best_free = float(greedy[a:b].sum())
        candidates = {}  # start_word -> (prior score, hyp)
        for h in hyps:
            starts = {h.end_word: 0.0}
            for s in line_starts:
                if s != h.end_word:
                    starts[s] = jump_cost
            for s, cost in starts.items():
                prior = h.score - cost
                if s not in candidates or candidates[s][0] < prior:
                    candidates[s] = (prior, h)
        new = {}
        for s, (prior, h) in candidates.items():
            if s >= n_words:
                continue
            key = (pi, s)
            if key not in cache:
                last_word = min(n_words - 1, s + max_words)
                t0 = target.word_first_token[s]
                t1 = max(target.word_last_token[s:last_word + 1].max(), t0)
                toks = target.tokens[t0:t1 + 1]
                if len(toks) == 0 or 2 * len(toks) + 1 > 6 * (b - a):
                    toks = toks[:max(1, (b - a) // 2)]
                scores, _ = ctc_free_end(seg, toks, blank)
                ends = {}
                for k, sc in enumerate(scores):
                    w = int(target.token_word[t0 + k])
                    if target.word_last_token[w] == t0 + k:
                        ends[w] = float(sc)
                cache[key] = ends
            for w, sc in cache[key].items():
                total = prior + sc
                nxt = next((x for x in range(w + 1, n_words) if target.word_first_token[x] >= 0), n_words)
                if nxt not in new or new[nxt].score < total:
                    new[nxt] = Hyp(total, nxt, h.trail + ((pi, s, w),))
        null_score = best_free - null_cost_per_frame * (b - a)
        for h in hyps:
            total = h.score + null_score
            k = ("null", h.end_word)
            if k not in new or new[k].score < total:
                new[k] = Hyp(total, h.end_word, h.trail + ((pi, None, None),))
        hyps = sorted(new.values(), key=lambda x: -x.score)[:beam]
    best = hyps[0]
    words, report = [], {"phrases": len(phrases), "matched": 0, "null": 0, "phrase_details": []}
    for (pi, s, w) in best.trail:
        a, b = phrases[pi]
        if s is None:
            report["null"] += 1
            report["phrase_details"].append({"start": a * frame_seconds, "end": b * frame_seconds, "match": None})
            continue
        report["matched"] += 1
        t0 = target.word_first_token[s]
        t1 = target.word_last_token[w]
        toks = target.tokens[t0:t1 + 1]
        scores, spans_fn = ctc_free_end(logp[a:b], toks, blank, backtrace=True)
        token_spans = spans_fn(len(toks) - 1)
        seg_greedy = float(greedy[a:b].sum())
        deficit = (seg_greedy - float(scores[-1])) / max(1, b - a)
        confidence = float(np.exp(-deficit))
        report["phrase_details"].append({"start": a * frame_seconds, "end": b * frame_seconds,
                                         "words": [s, w], "deficit_per_frame": round(deficit, 3)})
        for word in range(s, w + 1):
            f, l = target.word_first_token[word], target.word_last_token[word]
            if f < 0:
                continue
            fs = [token_spans[k - t0] for k in range(f, l + 1) if (k - t0) in token_spans]
            if not fs:
                continue
            start = (a + min(x[0] for x in fs)) * frame_seconds
            end = (a + max(x[1] for x in fs) + 1) * frame_seconds
            words.append({"word_index": int(word), "line": int(target.word_line[word]), "start": start,
                          "end": end, "phrase": pi, "confidence": confidence})
    words.sort(key=lambda x: x["start"])
    report["score"] = best.score
    deficits = [d["deficit_per_frame"] for d in report["phrase_details"] if d.get("words")]
    report["mean_deficit"] = float(np.mean(deficits)) if deficits else None
    voiced_frames = sum(b - a for a, b in phrases)
    matched_frames = sum(int((d["end"] - d["start"]) / frame_seconds) for d in report["phrase_details"] if d.get("words"))
    report["matched_share"] = matched_frames / voiced_frames if voiced_frames else 0.0
    return words, report


def smooth_words(words, gap_fill=0.3, min_duration=0.12):
    """Close tiny gaps inside a line so the karaoke fill flows; enforce minimal durations."""
    out = [dict(w) for w in words]
    for i, w in enumerate(out):
        w["end"] = max(w["end"], w["start"] + min_duration)
        if i + 1 < len(out):
            nxt = out[i + 1]
            if nxt["line"] == w["line"] and 0 <= nxt["start"] - w["end"] <= gap_fill:
                w["end"] = nxt["start"]
            w["end"] = min(w["end"], max(nxt["start"], w["start"] + 0.05))
    return out
