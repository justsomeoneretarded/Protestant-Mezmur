"""Whole-song, lyrics-constrained CTC alignment.

The song is one path through a graph built from the lyric lines:

* inside a line the path follows normal CTC rules (blank / character / blank ...);
* after finishing a line it may go straight into another line, or wait in a
  "between lines" state that absorbs instrumentals, breaths and unscripted singing;
* the next line costs nothing, repeating a line or going back to an earlier one
  (choruses) costs a little, skipping ahead costs more;
* starting or stopping in the middle of a line is allowed at a price (echoes, ad-libs).

Unscripted singing is not free: in the between-lines state every frame where the
acoustic model hears a character pays ``garbage_penalty``. Without that, a weak
acoustic model on singing makes "say nothing" the cheapest explanation, and lyrics
flash on screen for an instant and disappear.

Character posteriors are divided by their song-level priors (``prior_weight``), the
standard correction for CTC's blank-dominated, peaky output on long audio.
"""
from dataclasses import dataclass, field

import numpy as np

NEG = -1e9


@dataclass
class Lyrics:
    lines: list  # list[list[str]] words per line
    stanza_of_line: list = field(default_factory=list)

    @property
    def words(self):
        return [w for line in self.lines for w in line]


@dataclass
class Target:
    tokens: np.ndarray            # token ids of every line, concatenated (no blanks)
    token_word: np.ndarray        # global word index of each token
    word_first_token: np.ndarray  # first token of each word (-1 if the word has no known characters)
    word_last_token: np.ndarray
    word_line: np.ndarray
    line_first_word: np.ndarray


def build_target(lyrics: Lyrics, char_to_id: dict) -> Target:
    tokens, token_word, word_line, first, last, line_first_word = [], [], [], [], [], []
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
    arr = lambda x: np.array(x, dtype=np.int64)  # noqa: E731
    return Target(arr(tokens), arr(token_word), arr(first), arr(last), arr(word_line), arr(line_first_word))


def merge_blank(logp: np.ndarray, blank: int, delimiter):
    """Treat the word delimiter as silence: the lyrics carry their own word boundaries."""
    if delimiter is None:
        return logp
    out = logp.copy()
    out[:, blank] = np.logaddexp(logp[:, blank], logp[:, delimiter])
    out[:, delimiter] = NEG
    return out


def frame_activity(logp: np.ndarray, blank: int, energy=None, threshold=0.5, radius=5):
    """A frame is 'vocal' when the model puts real probability on some character nearby."""
    nonblank = 1.0 - np.exp(logp[:, blank])
    padded = np.pad(nonblank, radius, mode="edge")
    peak = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1).max(-1)[:len(nonblank)]
    active = peak > threshold
    if energy is not None and len(energy) == len(active):
        active &= energy
    return active


@dataclass
class _Line:
    index: int
    offset: int        # first global state
    n_states: int
    words: list        # global word indices with tokens, in order
    word_start_state: list  # local state of each word's first token
    word_end_state: list    # local state of each word's last token


def _build_graph(target: Target, blank: int):
    lines, labels, skip_ok = [], [], []
    n_lines = len(target.line_first_word)
    n_words = len(target.word_line)
    for li in range(n_lines):
        w0 = target.line_first_word[li]
        w1 = target.line_first_word[li + 1] if li + 1 < n_lines else n_words
        words = [w for w in range(w0, w1) if target.word_first_token[w] >= 0]
        if not words:
            lines.append(None)
            continue
        t0, t1 = target.word_first_token[words[0]], target.word_last_token[words[-1]]
        toks = target.tokens[t0:t1 + 1]
        S = 2 * len(toks) + 1
        lab = np.full(S, blank, dtype=np.int64)
        lab[1::2] = toks
        sk = np.zeros(S, dtype=bool)
        sk[3::2] = toks[1:] != toks[:-1]
        line = _Line(li, len(labels), S, words,
                     [2 * (target.word_first_token[w] - t0) + 1 for w in words],
                     [2 * (target.word_last_token[w] - t0) + 1 for w in words])
        lines.append(line)
        labels.extend(lab.tolist())
        skip_ok.extend(sk.tolist())
    return lines, np.array(labels, dtype=np.int64), np.array(skip_ok, dtype=bool)


def _jump_cost_matrix(n, repeat_same, go_back, skip_per_line, start_cost):
    """cost[k + 1, j]: from 'just finished line k' (k = -1: song start) into line j."""
    c = np.zeros((n + 1, n))
    for k in range(-1, n):
        for j in range(n):
            if k == -1:
                c[0, j] = 0.0 if j == 0 else start_cost
            elif j == k + 1:
                c[k + 1, j] = 0.0
            elif j == k:
                c[k + 1, j] = repeat_same
            elif j < k:
                c[k + 1, j] = go_back
            else:
                c[k + 1, j] = skip_per_line * (j - k - 1)
    return c


def align_lyrics(logp, frame_seconds, target: Target, blank, delimiter, energy=None,
                 prior_weight=0.45, garbage_penalty=2.2, repeat_same=1.0, go_back=1.5,
                 skip_per_line=2.5, start_cost=1.5, partial_cost=3.5, inline_wait=0.03, repeats=None,
                 log=print, **_ignored):
    """Return (words, report).

    words: [{"word_index", "line", "start", "end", "phrase", "confidence"}] where "phrase"
    numbers each sung occurrence of a line.
    """
    merged = merge_blank(np.asarray(logp, dtype=np.float64), blank, delimiter)
    T = merged.shape[0]
    probs = np.exp(merged)
    prior = probs.mean(0) + 1e-6
    adj = merged - prior_weight * np.log(prior)[None, :]
    adj[:, blank] = merged[:, blank] - prior_weight * np.log(prior[blank])
    chars = np.ones(merged.shape[1], dtype=bool)
    chars[blank] = False
    if delimiter is not None:
        chars[delimiter] = False
    best_char = adj[:, chars].max(1)
    garbage = np.maximum(adj[:, blank], best_char - garbage_penalty)

    lines, labels, skip_ok = _build_graph(target, blank)
    real = [l for l in lines if l is not None]
    n = len(lines)
    if not real or T == 0:
        return [], {"phrases": 0, "matched": 0, "null": 0, "matched_share": 0.0, "mean_deficit": None,
                    "gain_per_vocal_frame": 0.0, "phrase_details": []}
    S = len(labels)
    emit = adj[:, labels]                      # [T, S]
    # Waiting silently *inside* a line costs a little more than waiting between lines, so a
    # line is not stretched across an instrumental intro/outro just because it began there.
    emit[:, labels == blank] -= inline_wait
    cost = _jump_cost_matrix(n, repeat_same, go_back, skip_per_line, start_cost)
    if repeats is not None:  # lines marked '/3', 'x2', ... are expected to be sung again right away
        for k in range(min(n, len(repeats))):
            if repeats[k] > 1:
                cost[k + 1, k] = 0.0
    missing = np.array([l is None for l in lines])
    cost[:, missing] = 1e6

    # Index helpers.
    line_of_state = np.empty(S, dtype=np.int64)
    first_state = np.zeros(S, dtype=bool)      # leading blank or first token of a line
    for l in real:
        line_of_state[l.offset:l.offset + l.n_states] = l.index
        first_state[l.offset] = first_state[l.offset + 1] = True
    full_end = {l.index: np.array([l.offset + l.n_states - 2, l.offset + l.n_states - 1]) for l in real}
    part_end = {l.index: np.array([l.offset + s for s in l.word_end_state[:-1]] +
                                  [l.offset + s + 1 for s in l.word_end_state[:-1]], dtype=np.int64) for l in real}
    part_start = {l.index: np.array([l.offset + s for s in l.word_start_state[1:]] +
                                    [l.offset + s - 1 for s in l.word_start_state[1:]], dtype=np.int64) for l in real}
    step_ok = np.ones(S, dtype=bool)
    for l in real:
        step_ok[l.offset] = False              # nothing steps into a line's first state from the previous line
    skip_ok_g = skip_ok.copy()
    for l in real:
        skip_ok_g[l.offset:l.offset + 2] = False

    # Backpointers.
    choice = np.zeros((T, S), dtype=np.int8)          # 0 stay, 1 step, 2 skip, 3 full entry, 4 partial entry
    entry_src = np.full((T, n), -2, dtype=np.int16)   # k (finished line) or -1 (start)
    entry_via = np.zeros((T, n), dtype=np.int8)       # 0 from hub, 1 direct from the line's exit
    hub_from = np.zeros((T, n + 1), dtype=np.int8)    # 0 stay, 1 came from the line's exit
    come_state = np.full((T, n), -1, dtype=np.int32)  # which state the exit left from

    hub = np.full(n + 1, NEG)                         # hub[0] = song start, hub[k+1] = after line k
    hub[0] = garbage[0]
    alpha = np.full(S, NEG)
    # t = 0: enter from the start hub.
    for l in real:
        c = cost[0, l.index]
        alpha[l.offset] = max(alpha[l.offset], -c + emit[0, l.offset])
        alpha[l.offset + 1] = max(alpha[l.offset + 1], -c + emit[0, l.offset + 1])
        choice[0, l.offset] = choice[0, l.offset + 1] = 3
        entry_src[0, l.index] = -1
        for s in part_start[l.index]:
            v = -c - partial_cost + emit[0, s]
            if v > alpha[s]:
                alpha[s], choice[0, s] = v, 4

    for t in range(1, T):
        # Exits from lines at t-1.
        come = np.full(n, NEG)
        for l in real:
            fe = full_end[l.index]
            i = int(np.argmax(alpha[fe]))
            best, state = alpha[fe[i]], fe[i]
            pe = part_end[l.index]
            if len(pe):
                j = int(np.argmax(alpha[pe]))
                if alpha[pe[j]] - partial_cost > best:
                    best, state = alpha[pe[j]] - partial_cost, pe[j]
            come[l.index] = best
            come_state[t, l.index] = state
        # Hubs.
        stay = hub[1:]
        from_line = come > stay
        new_hub = np.empty(n + 1)
        new_hub[0] = hub[0] + garbage[t]
        new_hub[1:] = np.where(from_line, come, stay) + garbage[t]
        hub_from[t, 1:] = from_line
        # Entries: best predecessor per line (hub or direct exit).
        src_val = np.concatenate(([hub[0]], np.maximum(hub[1:], come)))
        via_direct = np.concatenate(([False], come > hub[1:]))
        scores = src_val[:, None] - cost                 # [n+1, n]
        k_best = scores.argmax(0)
        entry = scores[k_best, np.arange(n)]
        entry_src[t] = k_best - 1
        entry_via[t] = via_direct[k_best]
        # In-line CTC transitions.
        prev = alpha
        step = np.concatenate(([NEG], prev[:-1]))
        step[~step_ok] = NEG
        skip = np.concatenate(([NEG, NEG], prev[:-2]))
        skip[~skip_ok_g] = NEG
        best = prev.copy()
        ch = np.zeros(S, dtype=np.int8)
        m = step > best
        best[m], ch[m] = step[m], 1
        m = skip > best
        best[m], ch[m] = skip[m], 2
        # Line entries.
        for l in real:
            e = entry[l.index]
            for s in (l.offset, l.offset + 1):
                if e > best[s]:
                    best[s], ch[s] = e, 3
            ep = e - partial_cost
            for s in part_start[l.index]:
                if ep > best[s]:
                    best[s], ch[s] = ep, 4
        alpha = best + emit[t]
        choice[t] = ch
        hub = new_hub

    # Final: best of any hub or any line end.
    finals = [(hub[k], ("hub", k)) for k in range(n + 1)]
    for l in real:
        for s in full_end[l.index]:
            finals.append((alpha[s], ("state", int(s))))
    total, where = max(finals, key=lambda x: x[0])

    # Backtrace -> per-frame global state (or -1 for hub).
    path = np.full(T, -1, dtype=np.int64)
    entered = np.zeros(T, dtype=bool)
    kind, val = where
    t = T - 1
    while t >= 0:
        if kind == "state":
            s = val
            path[t] = s
            c = choice[t, s]
            if c == 0:
                pass
            elif c == 1:
                s -= 1
            elif c == 2:
                s -= 2
            else:
                entered[t] = True
                j = int(line_of_state[s])
                k = int(entry_src[t, j])
                if t == 0:
                    break
                if k >= 0 and entry_via[t, j]:
                    kind, val = "state", int(come_state[t, k])
                else:
                    kind, val = "hub", k + 1
                t -= 1
                continue
            val = s
        else:
            k = val
            if t == 0:
                break
            if k >= 1 and hub_from[t, k]:
                kind, val = "state", int(come_state[t, k - 1])
            t -= 1
            continue
        t -= 1

    # Occurrences ("phrases") of lines and their word timings.
    words, details = [], []
    active = frame_activity(merged, blank, energy)
    occ_frames = np.zeros(T, dtype=bool)
    t = 0
    phrase = -1
    while t < T:
        if path[t] < 0:
            t += 1
            continue
        start = t
        li = int(line_of_state[path[t]])
        t += 1
        while t < T and path[t] >= 0 and not entered[t] and line_of_state[path[t]] == li:
            t += 1
        end = t
        phrase += 1
        seg = path[start:end]
        local = seg - lines[li].offset
        token_frames = {}
        for f, s in enumerate(local):
            if s % 2 == 1:
                token_frames.setdefault(int(s // 2), []).append(start + f)
        line = lines[li]
        t0 = target.word_first_token[line.words[0]]
        chosen = merged[np.arange(start, end), labels[seg]]
        deficit = float(np.mean(merged[start:end].max(1) - chosen)) if end > start else 9.0
        conf = float(np.exp(-deficit))
        occ_words = []
        for w in line.words:
            f, last = target.word_first_token[w] - t0, target.word_last_token[w] - t0
            frames = [x for k in range(f, last + 1) for x in token_frames.get(int(k), [])]
            if not frames:
                continue
            occ_words.append({"word_index": int(w), "line": int(li), "start": min(frames) * frame_seconds,
                              "end": (max(frames) + 1) * frame_seconds, "phrase": phrase, "confidence": conf})
        if not occ_words:
            continue
        span = occ_words[-1]["end"] - occ_words[0]["start"]
        if len(occ_words) < len(line.words) and (span < 0.6 or len(occ_words) < 2 and deficit > 0.8):
            continue  # stray fragment between real lines
        occ_frames[int(occ_words[0]["start"] / frame_seconds):int(occ_words[-1]["end"] / frame_seconds)] = True
        words += occ_words
        details.append({"start": round(occ_words[0]["start"], 2), "end": round(occ_words[-1]["end"], 2),
                        "line": int(li), "words": [occ_words[0]["word_index"], occ_words[-1]["word_index"]],
                        "complete": len(occ_words) == len(line.words), "deficit_per_frame": round(deficit, 3)})

    # Vocal stretches nobody claimed.
    unclaimed = active & ~occ_frames
    null = 0
    run = 0
    for v in np.append(unclaimed, False):
        if v:
            run += 1
        else:
            if run * frame_seconds >= 0.5:
                null += 1
            run = 0
    vocal = int(active.sum())
    report = {
        "phrases": len(details), "matched": len(details), "null": null,
        "matched_share": float((active & occ_frames).sum() / vocal) if vocal else 0.0,
        "mean_deficit": float(np.mean([d["deficit_per_frame"] for d in details])) if details else None,
        "complete_share": float(np.mean([d["complete"] for d in details])) if details else 0.0,
        "gain_per_vocal_frame": float((total - garbage.sum()) / max(1, vocal)),
        "score": float(total), "phrase_details": details,
        "lines_used": len({d["line"] for d in details}), "lines_total": len(real),
    }
    words.sort(key=lambda x: x["start"])
    words = _merge_split_lines(words)
    return words, report


def _merge_split_lines(words, max_gap=4.0):
    """A singer pausing mid-line yields two partial occurrences; show them as one line."""
    out = [dict(w) for w in words]
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        if (b["phrase"] != a["phrase"] and b["line"] == a["line"] and b["word_index"] > a["word_index"]
                and b["start"] - a["end"] <= max_gap):
            old = b["phrase"]
            for w in out[i:]:
                if w["phrase"] == old:
                    w["phrase"] = a["phrase"]
    return out


def smooth_words(words, gap_fill=0.3, min_duration=0.12):
    """Close tiny gaps inside a sung line so the karaoke fill flows; enforce minimal durations."""
    out = [dict(w) for w in words]
    for i, w in enumerate(out):
        w["end"] = max(w["end"], w["start"] + min_duration)
        if i + 1 < len(out):
            nxt = out[i + 1]
            same = nxt.get("phrase", nxt.get("line")) == w.get("phrase", w.get("line"))
            if same and 0 <= nxt["start"] - w["end"] <= gap_fill:
                w["end"] = nxt["start"]
            w["end"] = min(w["end"], max(nxt["start"], w["start"] + 0.05))
    return out


# ---------------------------------------------------------------------------------------------
# Completing the alignment with the (verified) lyrics.
#
# Choirs, harmonies and heavy echo can make the acoustic model nearly deaf for a whole chorus
# even though the vocal stem is clearly singing. When the lyrics have been verified, the lines
# that are missing between two confidently placed lines must have been sung in that gap. They
# are laid out over the gap's actual singing (vocal loudness), anchored on any fragments that
# were recognised, and their words are marked ``estimated``.
# ---------------------------------------------------------------------------------------------

def _vocal_phrases(energy, frame_seconds, a, b, min_pause=0.35, min_len=0.4):
    fa, fb = max(0, int(a / frame_seconds)), min(len(energy), int(b / frame_seconds))
    seg = np.asarray(energy[fa:fb], dtype=bool)
    out, start, quiet = [], None, 0
    pause = max(1, int(min_pause / frame_seconds))
    for i, v in enumerate(np.append(seg, [False] * (pause + 1))):
        if v:
            if start is None:
                start = i
            quiet = 0
            last = i
        elif start is not None:
            quiet += 1
            if quiet >= pause:
                if (last + 1 - start) * frame_seconds >= min_len:
                    out.append(((fa + start) * frame_seconds, (fa + last + 1) * frame_seconds))
                start = None
    return out


def _lay_out(line_ids, lines, first_word, phrases, sec_per_char, phrase_id):
    """Spread whole lines over vocal phrases in proportion to their length; returns words."""
    lengths = [max(1, sum(len(w) for w in lines[l])) for l in line_ids]
    total_chars = sum(lengths)
    active = sum(b - a for a, b in phrases)
    if not phrases or active <= 0:
        return []

    def at(fraction):  # map a fraction of the sung time to a clock time
        target = fraction * active
        for a, b in phrases:
            if target <= b - a:
                return a + target
            target -= b - a
        return phrases[-1][1]

    starts = [p[0] for p in phrases]
    out, done = [], 0
    for k, l in enumerate(line_ids):
        t0, t1 = at(done / total_chars), at((done + lengths[k]) / total_chars)
        near = min(starts, key=lambda s: abs(s - t0))
        if abs(near - t0) < 1.5 and near < t1 - 0.5:
            t0 = near  # lines start where singing starts
        t1 = min(t1, t0 + lengths[k] * sec_per_char * 1.6)
        done += lengths[k]
        cursor, span = t0, max(0.3, t1 - t0)
        chars = lengths[k]
        for j, word in enumerate(lines[l]):
            d = span * max(1, len(word)) / chars
            out.append({"word_index": int(first_word[l] + j), "line": int(l), "start": cursor,
                        "end": cursor + d * 0.92, "phrase": phrase_id + k, "confidence": 0.0, "estimated": True})
            cursor += d
    return out


def _force_in_gap(emit_all, energy, fa, fb, seq, lines, first_word, char_to_id, blank, anchors, layout,
                  frame_seconds, phrase_id, silence_cost=3.0, spread_bonus=0.15, weak=-4.0):
    """Second recognition pass inside a gap: every word of ``seq`` (line ids) must appear, in order.

    emit_all: prior-normalised log-probabilities for the whole song [T, V].
    Characters placed on frames without vocal energy pay ``silence_cost``; blank on sung frames pays
    ``spread_bonus`` (so words follow the singing instead of bunching). Recognised fragments
    (``anchors``: list of (word position in seq, start, end)) pin their words in time.
    Returns words, or None if there are not enough frames.
    """
    # Token sequence with word bookkeeping.
    toks, tok_word = [], []
    word_meta = []  # (seq position, line id, word idx in line)
    for k, l in enumerate(seq):
        for j, word in enumerate(lines[l]):
            ids = [char_to_id[c] for c in word if c in char_to_id]
            wpos = len(word_meta)
            word_meta.append((k, l, j))
            toks.extend(ids)
            tok_word.extend([wpos] * len(ids))
    T = fb - fa
    L = len(toks)
    if L == 0 or T < L + 2:
        return None
    S = 2 * L + 1
    labels = np.full(S, blank, dtype=np.int64)
    labels[1::2] = toks
    emit = emit_all[fa:fb][:, labels].astype(np.float64).copy()
    voiced = np.asarray(energy[fa:fb], dtype=bool)
    emit[np.ix_(~voiced, np.arange(1, S, 2))] -= silence_cost
    emit[np.ix_(voiced, np.arange(0, S, 2))] -= spread_bonus
    # Anchors: a recognised word's characters may only sit inside its recognised span (+-0.3 s).
    tok_word = np.array(tok_word)
    for wpos, a, b in anchors:
        f0 = max(0, int((a - 0.3) / frame_seconds) - fa)
        f1 = min(T, int((b + 0.3) / frame_seconds) - fa)
        states = 1 + 2 * np.where(tok_word == wpos)[0]
        mask = np.ones(T, dtype=bool)
        mask[f0:f1] = False
        emit[np.ix_(mask, states)] = NEG
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[3::2] = np.array(toks[1:]) != np.array(toks[:-1])
    alpha = np.full(S, NEG)
    alpha[0], alpha[1] = emit[0, 0], emit[0, 1]
    choice = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([NEG], alpha[:-1]))
        skip = np.where(skip_ok, np.concatenate(([NEG, NEG], alpha[:-2])), NEG)
        best = stay.copy()
        c = np.zeros(S, dtype=np.int8)
        m = step > best
        best[m], c[m] = step[m], 1
        m = skip > best
        best[m], c[m] = skip[m], 2
        choice[t] = c
        alpha = best + emit[t]
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[s] <= NEG / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(choice[t, s])
    spans = {}
    for t, st in enumerate(path):
        if st % 2 == 1:
            w = int(tok_word[st // 2])
            a, b = spans.get(w, (t, t))
            spans[w] = (min(a, t), max(b, t))
    raw = emit_all[fa:fb]
    out = []
    for wpos, (k, l, j) in enumerate(word_meta):
        if wpos not in spans:
            continue
        a, b = spans[wpos]
        idx = np.where(tok_word == wpos)[0]
        # Evidence: best posterior of this word's characters inside its span.
        ev = float(np.mean([raw[a:b + 1, toks[i]].max() for i in idx])) if len(idx) else weak - 1
        anchored = any(x[0] == wpos for x in anchors)
        out.append({"word_index": int(first_word[l] + j), "line": int(l), "start": (fa + a) * frame_seconds,
                    "end": (fa + b + 1) * frame_seconds, "phrase": phrase_id + k, "confidence": float(np.exp(min(0, ev))),
                    "estimated": not anchored and ev < weak, "recognised": anchored or ev >= weak})
    return out


def complete_with_lyrics(words, lines, energy, frame_seconds, duration, max_repeat=4,
                         logp=None, char_to_id=None, blank=None, delimiter=None, repeats=None, stanza_of=None,
                         min_heard=0.2):
    """Fill lines the acoustic model missed and finish partially matched lines.

    words: aligned words (word_index, line, start, end, phrase); lines: list of word lists.
    Returns (words, info) where info counts what was estimated.
    """
    if not words or energy is None:
        return words, {"filled_lines": 0, "completed_lines": 0, "estimated_words": 0}
    first_word = np.concatenate(([0], np.cumsum([len(l) for l in lines])))[:-1]
    n_lines = len(lines)
    occ = {}
    for w in words:
        occ.setdefault(w["phrase"], []).append(w)
    occs = sorted(occ.values(), key=lambda ws: ws[0]["start"])
    full = [ws for ws in occs if len(ws) == len(lines[ws[0]["line"]])]
    rates = [(ws[-1]["end"] - ws[0]["start"]) / max(1, sum(len(w) for w in lines[ws[0]["line"]])) for ws in full]
    spc = float(np.median(rates)) if rates else 0.12
    next_phrase = max(w["phrase"] for w in words) + 1
    emit_all = None
    if logp is not None and char_to_id is not None and blank is not None:
        merged = merge_blank(np.asarray(logp, dtype=np.float64), blank, delimiter)
        prior = np.exp(merged).mean(0) + 1e-6
        emit_all = merged - 0.45 * np.log(prior)[None, :]
    out = [dict(w) for w in words]
    info = {"filled_lines": 0, "completed_lines": 0, "estimated_words": 0, "gaps": []}

    def note(**kw):
        info["gaps"].append({k: (round(v, 2) if isinstance(v, float) else v) for k, v in kw.items()})

    # 1) Gaps between confidently placed (complete) lines.
    bounds = [(None, 0.0)] + [(ws[0]["line"], ws[-1]["end"]) for ws in full]
    starts = [ws[0]["start"] for ws in full] + [duration]
    lines_after = [(ws[0]["line"]) for ws in full] + [None]
    for k in range(len(bounds)):
        prev_line, gap_a = bounds[k]
        next_line, gap_b = lines_after[k], starts[k]
        if gap_b - gap_a < 3.0:
            continue
        note(start=gap_a, end=gap_b, after_line=prev_line, before_line=next_line, action="examined")
        first = 0 if prev_line is None else prev_line + 1
        if next_line is not None and next_line <= first:
            # The next clearly heard line is not new (a chorus coming back, or a line written once with
            # '/3' that was only heard on its last repetition): the gap holds the start of its stanza.
            if stanza_of is None:
                continue
            st = stanza_of[next_line]
            s0 = min(i for i in range(n_lines) if stanza_of[i] == st)
            s1 = max(i for i in range(n_lines) if stanza_of[i] == st)
            missing = list(range(s0, next_line)) or list(range(s0, s1 + 1))
        else:
            last = (next_line - 1) if next_line is not None else n_lines - 1
            missing = list(range(first, last + 1))
        rep = repeats or [1] * n_lines
        certain = not (next_line is not None and next_line <= first)
        # Repetitions of the line just before the gap that were not heard yet ('ዛሬም ይሰራል /3' heard once).
        carry = []
        if prev_line is not None and rep[prev_line] > 1:
            run = 0
            for ws in reversed(full[:k]):
                if ws[0]["line"] != prev_line:
                    break
                run += 1
            carry = [prev_line] * max(0, rep[prev_line] - run)
        if not missing and not carry:
            continue
        unit = carry + [l for l in missing for _ in range(max(1, rep[l]))]
        missing = sorted(set(unit))
        phrases = _vocal_phrases(energy, frame_seconds, gap_a + 0.15, gap_b - 0.15)
        active = sum(b - a for a, b in phrases)
        need = sum(len(w) for l in unit for w in lines[l]) * spc
        if active < 1.0 or need <= 0:
            note(start=gap_a, end=gap_b, lines=unit, singing=active, action="skipped: no singing")
            continue
        ratio = active / need
        if ratio < 0.45 or ratio > max_repeat + 0.6:
            note(start=gap_a, end=gap_b, lines=unit, singing=active, needed=need, ratio=ratio,
                 action="skipped: singing does not fit these lines")
            continue  # far too little or far too much singing for these lines
        reps = 1 if ratio < 1.7 else int(min(max_repeat, round(ratio)))
        texts = {" ".join(lines[l]) for l in missing}
        in_gap = [w for w in out if gap_a <= w["start"] < gap_b and " ".join(lines[w["line"]]) in texts]
        if (reps == 1 and len(unit) == len(missing) and {w["line"] for w in in_gap} >= set(missing)
                and all(" ".join(lines[w["line"]]) in texts for w in out if gap_a <= w["start"] < gap_b)):
            continue  # every missing line was partly recognised: keep real timings, finish them below
        seq = unit * reps
        # Everything heard inside the gap is replaced by the fill; matching fragments become anchors and
        # fragments of other lines (contradicting the lyrics' order) are dropped.
        fragments = [w for w in out if gap_a <= w["start"] < gap_b and " ".join(lines[w["line"]]) in texts]
        stray = [w for w in out if gap_a <= w["start"] < gap_b and " ".join(lines[w["line"]]) not in texts]
        out = [w for w in out if not (gap_a <= w["start"] < gap_b)]
        layout = _lay_out(seq, lines, first_word, phrases, spc, next_phrase)
        added = None
        if emit_all is not None:
            # Anchor each recognised fragment word to the nearest copy of that word in the sequence.
            positions = []  # (wpos, line text, word idx in line, layout start)
            wpos = 0
            lay_by_k = {}
            for w in layout:
                lay_by_k.setdefault(w["phrase"] - next_phrase, []).append(w)
            for k, l in enumerate(seq):
                for j in range(len(lines[l])):
                    lw = lay_by_k.get(k, [])
                    positions.append((wpos, " ".join(lines[l]), j, lw[j]["start"] if j < len(lw) else gap_a))
                    wpos += 1
            anchors, taken = [], set()
            for fw in fragments:
                key = " ".join(lines[fw["line"]])
                j = fw["word_index"] - first_word[fw["line"]]
                cand = [p for p in positions if p[1] == key and p[2] == j and p[0] not in taken]
                if cand:
                    best = min(cand, key=lambda p: abs(p[3] - fw["start"]))
                    taken.add(best[0])
                    anchors.append((best[0], fw["start"], fw["end"]))
            fa, fb = int((gap_a + 0.05) / frame_seconds), int((gap_b - 0.05) / frame_seconds)
            try:
                added = _force_in_gap(emit_all, energy, fa, fb, seq, lines, first_word, char_to_id, blank, anchors,
                                      layout, frame_seconds, next_phrase)
            except Exception:  # noqa: BLE001 - fall back to the layout
                added = None
        if not added:
            added = layout
        heard = sum(1 for w in added if w.get("recognised")) / max(1, len(added))
        if emit_all is not None and heard < min_heard and not certain:
            out += fragments + stray  # uncertain structure and nothing heard: leave the gap as it was
            note(start=gap_a, end=gap_b, lines=seq, heard=heard, action="skipped: nothing heard, order uncertain")
            continue
        note(start=gap_a, end=gap_b, lines=seq, singing=active, needed=need, ratio=ratio, heard=heard,
             action="filled")
        next_phrase += len(seq)
        out += added
        info["filled_lines"] += len(seq)
        info["estimated_words"] += sum(1 for w in added if w.get("estimated"))
        info["recognised_in_gaps"] = info.get("recognised_in_gaps", 0) + sum(1 for w in added if w.get("recognised"))

    # 2) Finish partially matched lines (outside filled gaps).
    out.sort(key=lambda w: w["start"])
    groups = {}
    for w in out:
        groups.setdefault(w["phrase"], []).append(w)
    ordered = sorted(groups.values(), key=lambda ws: ws[0]["start"])
    for idx, ws in enumerate(ordered):
        l = ws[0]["line"]
        have = {w["word_index"] - first_word[l] for w in ws}
        if len(have) == len(lines[l]) or any(w.get("estimated") for w in ws):
            continue
        prev_end = ordered[idx - 1][-1]["end"] if idx else 0.0
        next_start = ordered[idx + 1][0]["start"] if idx + 1 < len(ordered) else duration
        lo, hi = min(have), max(have)
        added = []
        cursor = ws[0]["start"]
        for j in range(lo - 1, -1, -1):  # leading words, backwards
            d = max(0.15, len(lines[l][j]) * spc)
            if cursor - d < prev_end + 0.05:
                break
            cursor -= d
            added.append({"word_index": int(first_word[l] + j), "line": l, "start": cursor, "end": cursor + d * 0.92,
                          "phrase": ws[0]["phrase"], "confidence": 0.0, "estimated": True})
        cursor = ws[-1]["end"]
        for j in range(hi + 1, len(lines[l])):  # trailing words
            d = max(0.15, len(lines[l][j]) * spc)
            if cursor + d > next_start - 0.05:
                break
            added.append({"word_index": int(first_word[l] + j), "line": l, "start": cursor + 0.04,
                          "end": cursor + d * 0.92, "phrase": ws[0]["phrase"], "confidence": 0.0, "estimated": True})
            cursor += d
        # Inner holes: interpolate between neighbours.
        known = sorted(ws, key=lambda w: w["word_index"])
        for x, y in zip(known, known[1:]):
            hole = list(range(x["word_index"] + 1, y["word_index"]))
            if not hole:
                continue
            span_a, span_b = x["end"], y["start"]
            if span_b - span_a < 0.1 * len(hole):
                span_a, span_b = x["start"], y["end"]
            step = (span_b - span_a) / (len(hole) + 1)
            for m, gw in enumerate(hole, 1):
                added.append({"word_index": int(gw), "line": l, "start": span_a + step * (m - 0.5),
                              "end": span_a + step * (m + 0.4), "phrase": ws[0]["phrase"], "confidence": 0.0,
                              "estimated": True})
        if added:
            out += added
            info["completed_lines"] += 1
            info["estimated_words"] += len(added)
    out.sort(key=lambda w: (w["start"], w["word_index"]))
    return out, info
