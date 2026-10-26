import numpy as np

from mezmur.align import Lyrics, align_lyrics, build_target


def synth(lines, plan, vocab, blank, seed=0):
    rng = np.random.default_rng(seed)
    V = blank + 1
    frames, truth = [], []

    def emit(c, n, p=0.75):
        for _ in range(n):
            v = np.full(V, (1 - p) / (V - 1)); v[c] = p
            v = v * rng.uniform(0.6, 1.4, V); frames.append(v / v.sum())

    def silence(n):
        for _ in range(n):
            v = np.full(V, 0.02 / (V - 1)); v[blank] = 0.98; frames.append(v / v.sum())

    silence(100)
    for phrase in plan:
        if phrase == "adlib":
            for c in "ዘየደገ":
                emit(vocab[c], 6)
            silence(60)
            continue
        for li, wi in phrase:
            start = len(frames)
            for c in lines[li][wi]:
                emit(vocab[c], int(rng.integers(3, 9))); silence(int(rng.integers(0, 3)))
            truth.append((li, wi, start * 0.02))
            silence(4)
        silence(int(rng.integers(35, 60)))
    return np.log(np.array(frames)), truth


def test_repeated_chorus_split_line_and_adlib():
    lines = [["ሲወድ", "ጌታ", "ክብር"], ["ምስጋና", "ለአምላክ", "ነው"], ["ሀሌሉያ", "ክብር", "ሲወድ"], ["በረከት", "ሰላም", "ደስታ", "ተስፋ"]]
    chars = sorted({c for l in lines for w in l for c in w} | set("ዘየደገ"))
    vocab = {c: i for i, c in enumerate(chars)}
    delim, blank = len(chars), len(chars) + 1
    plan = [[(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1)], [(1, 2)], [(2, 0), (2, 1), (2, 2)],
            [(3, 0), (3, 1), (3, 2), (3, 3)], [(2, 0), (2, 1), (2, 2)], "adlib", [(2, 0), (2, 1), (2, 2)]]
    logp, truth = synth(lines, plan, vocab, blank)
    target = build_target(Lyrics(lines), vocab)
    words, report = align_lyrics(logp, 0.02, target, blank, delim)
    flat = [w for l in lines for w in l]
    got = [(flat[w["word_index"]], w["line"], round(w["start"], 1)) for w in words]
    want = [(lines[l][i], l, round(s, 1)) for l, i, s in truth]
    assert [g[:2] for g in got] == [w[:2] for w in want]
    assert all(abs(g[2] - w[2]) <= 0.1 for g, w in zip(got, want))
    assert report["null"] == 1
