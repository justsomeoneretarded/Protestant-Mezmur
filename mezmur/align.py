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
                 skip_per_line=2.5, start_cost=1.5, partial_cost=3.5, inline_wait=0.03, log=print, **_ignored):
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
        if (b["phrase"] != a["phrase"] and b["line"] == a["line"] and b["word_index"] == a["word_index"] + 1
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
