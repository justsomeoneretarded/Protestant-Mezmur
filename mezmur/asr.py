"""Amharic CTC acoustic model: whole-song frame log-probabilities and a greedy draft."""
import math
import os
from functools import lru_cache

import numpy as np

from .audio import read_audio

MODEL = "badrex/Ethio-ASR-amharic"
SAMPLE_RATE = 16000
WINDOW = 24.0
CONTEXT = 3.0
CACHE = os.environ.get("MEZMUR_MODEL_CACHE", ".model-cache")


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    torch.set_num_threads(max(1, os.cpu_count() or 2))
    processor = AutoProcessor.from_pretrained(MODEL, cache_dir=f"{CACHE}/ethio-asr")
    model = AutoModelForCTC.from_pretrained(MODEL, cache_dir=f"{CACHE}/ethio-asr", use_safetensors=True).eval()
    return processor, model


def vocabulary():
    processor, _ = load_model()
    tok = processor.tokenizer
    vocab = tok.get_vocab()
    return {
        "char_to_id": {k: v for k, v in vocab.items() if len(k) == 1},
        "id_to_char": {v: k for k, v in vocab.items()},
        "blank": tok.pad_token_id,
        "delimiter": vocab.get(getattr(tok, "word_delimiter_token", "|") or "|"),
    }


def frame_log_probs(path, log=print):
    """Run the model over overlapping windows and stitch per-frame log-probabilities."""
    import torch

    processor, model = load_model()
    samples = read_audio(path, SAMPLE_RATE)
    seconds = len(samples) / SAMPLE_RATE
    pieces, frame_seconds = [], None
    starts = list(np.arange(0, max(seconds, 1e-3), WINDOW))
    for index, core_start in enumerate(starts):
        core_end = min(seconds, core_start + WINDOW)
        a = max(0.0, core_start - CONTEXT)
        b = min(seconds, core_end + CONTEXT)
        chunk = samples[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE * 0.5:
            chunk = np.pad(chunk, (0, int(SAMPLE_RATE * 0.5) - len(chunk)))
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits[0].float()
        logp = torch.log_softmax(logits, -1).numpy()
        step = (len(chunk) / SAMPLE_RATE) / logp.shape[0]
        frame_seconds = frame_seconds or step
        lo = int(round((core_start - a) / step))
        hi = int(round((core_end - a) / step))
        pieces.append(logp[lo:hi])
        if index % 4 == 0:
            log(f"  acoustic model {core_end:.0f}/{seconds:.0f}s", flush=True)
    logp = np.concatenate(pieces).astype(np.float32)
    # Re-derive the frame duration from the stitched length to avoid drift.
    frame_seconds = seconds / len(logp)
    return logp, frame_seconds


def greedy_words(logp, frame_seconds, vocab, min_confidence=0.35):
    """Unconstrained draft: collapse the best CTC path into timed words with confidence."""
    blank, delim, id_to_char = vocab["blank"], vocab["delimiter"], vocab["id_to_char"]
    ids = logp.argmax(-1)
    probs = np.exp(logp.max(-1))
    words, current = [], None  # current: {"chars": [], "start": f, "end": f, "scores": []}
    prev, gap = None, 0

    def flush():
        nonlocal current
        if current and current["chars"]:
            text = "".join(current["chars"]).strip()
            if text:
                words.append({"word": text, "start": current["start"] * frame_seconds,
                              "end": (current["end"] + 1) * frame_seconds,
                              "probability": float(np.mean(current["scores"]))})
        current = None

    for t, i in enumerate(ids):
        i = int(i)
        if i == blank:
            gap += 1
            if gap * frame_seconds > 0.35:
                flush()
        elif i == delim:
            flush()
            gap = 0
        else:
            gap = 0
            if current is None:
                current = {"chars": [], "start": t, "end": t, "scores": []}
            if i != prev:
                current["chars"].append(id_to_char.get(i, ""))
            current["end"] = t
            current["scores"].append(float(probs[t]))
        prev = i
    flush()
    return [w for w in words if w["probability"] >= min_confidence]


def vocal_energy(path, frame_seconds, n_frames, floor_db=-42.0):
    """Boolean per-frame gate from the separated vocal stem's loudness."""
    samples = read_audio(path, SAMPLE_RATE)
    hop = frame_seconds * SAMPLE_RATE
    out = np.zeros(n_frames, dtype=bool)
    peak = float(np.max(np.abs(samples))) or 1.0
    for t in range(n_frames):
        a = int(t * hop)
        b = int(a + max(hop, SAMPLE_RATE * 0.04))
        seg = samples[a:b]
        if len(seg):
            rms = math.sqrt(float(np.mean(seg * seg))) / peak
            out[t] = 20 * math.log10(rms + 1e-9) > floor_db
    # Dilate so consonant onsets are not cut.
    k = max(1, int(0.12 / frame_seconds))
    padded = np.convolve(out.astype(int), np.ones(2 * k + 1, dtype=int), "same") > 0
    return padded
