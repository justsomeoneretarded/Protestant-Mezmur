"""Transcribe short overlapping chunks; all timestamps stay on the source timeline."""
import math
from functools import lru_cache
from pathlib import Path

from .audio import read_audio
from .common import run, write_json
from .quality import TranscriptionQualityError, assess, repeated_character_word

MODEL = "badrex/Ethio-ASR-amharic"
CORE_SECONDS = 20
OVERLAP_SECONDS = 2
SAMPLE_RATE = 16000
MIN_WORD_CONFIDENCE = 0.35


def chunks(seconds):
    for core_start in range(0, math.ceil(seconds), CORE_SECONDS):
        core_end = min(seconds, core_start + CORE_SECONDS)
        start = max(0, core_start - OVERLAP_SECONDS)
        end = min(seconds, core_end + OVERLAP_SECONDS)
        yield core_start, core_end, start, end


def merge_words(results, seconds):
    words = []
    for result in results:
        for item in result["response"].get("words", []):
            text = str(item.get("word", "")).strip()
            start = float(item["start"]) + result["offset"]
            end = float(item["end"]) + result["offset"]
            if not all(math.isfinite(x) for x in [start, end]) or end <= start or not text:
                continue
            # Each word belongs to one core interval, even when audio chunks overlap.
            midpoint = (start + end) / 2
            if result["core_start"] <= midpoint < result["core_end"]:
                word = {"word": text, "start": max(0, start), "end": min(seconds, end)}
                if "probability" in item:
                    word["probability"] = float(item["probability"])
                words.append(word)
    words.sort(key=lambda word: word["start"])
    clean = []
    for word in words:
        if clean:
            word["start"] = max(word["start"], clean[-1]["end"])
        if word["end"] > word["start"]:
            clean.append(word)
    if not clean:
        raise ValueError("No timed lyrics returned. Refusing to create an empty lyric video.")
    return clean


@lru_cache(maxsize=1)
def load_model():
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(MODEL, cache_dir=".model-cache/ethio-asr")
    model = AutoModelForCTC.from_pretrained(
        MODEL, cache_dir=".model-cache/ethio-asr", use_safetensors=True,
    ).eval()
    return processor, model


def decode_words(processor, logits, seconds):
    """Native CTC frame offsets locate words; blank frames leave music uncaptioned."""
    probabilities = logits.softmax(-1)
    ids = logits.argmax(-1)
    decoded = processor.tokenizer.decode(ids.tolist(), output_word_offsets=True)
    step = seconds / len(ids)
    words, rejected = [], []
    blank = processor.tokenizer.pad_token_id
    for item in decoded.word_offsets:
        start, end = int(item["start_offset"]), int(item["end_offset"])
        active = ids[start:end] != blank
        scores = probabilities[start:end].max(-1).values[active]
        confidence = float(scores.mean()) if len(scores) else 0.0
        word = {"word": item["word"].strip(), "start": start * step,
                "end": min(seconds, end * step), "probability": confidence}
        # A long hiss in the instrumental intro must not become a lyric.
        if confidence < MIN_WORD_CONFIDENCE or repeated_character_word(word["word"]):
            rejected.append(word)
        elif word["word"]:
            words.append(word)
    return {"language": "am", "text": decoded.text, "words": words,
            "rejected_words": rejected, "frame_seconds": step}


def transcribe(audio: Path, seconds: float, folder: Path):
    import torch

    processor, model = load_model()
    samples = read_audio(audio, SAMPLE_RATE)
    results = []
    for index, (core_start, core_end, start, end) in enumerate(chunks(seconds)):
        chunk = samples[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)]
        print(f"Transcribing chunk {index + 1} locally ({core_start:.0f}-{core_end:.0f}s)", flush=True)
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        response = decode_words(processor, logits, len(chunk) / SAMPLE_RATE)
        results.append({"offset": start, "core_start": core_start, "core_end": core_end,
                        "response": response})
    write_json(folder / "transcription.raw.json", results)
    try:
        words = merge_words(results, seconds)
    except ValueError:
        words = []
    quality = assess(words)
    if seconds >= 60 and len(words) < max(5, int(seconds / 30)):
        quality["passed"] = False
        quality["reasons"].append("Too few clear words for the song duration; review the source and vocals")
    quality["low_confidence_word_count"] = sum(w.get("probability", 1) < 0.60 for w in words)
    quality["discarded_word_count"] = sum(len(r["response"].get("rejected_words", [])) for r in results)
    quality["timing_method"] = "Native CTC acoustic frame offsets"
    write_json(folder / "quality.json", quality)
    if not quality["passed"]:
        raise TranscriptionQualityError("; ".join(quality["reasons"]))
    return words


def separate(audio: Path, folder: Path) -> Path:
    import sys

    run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
         "-d", "cpu", "--shifts", "0", "-j", "2", "-o", str(folder / "stems"), str(audio)])
    vocals = folder / "stems" / "htdemucs" / audio.stem / "vocals.wav"
    if not vocals.is_file():
        raise RuntimeError("Vocal separation did not produce vocals.wav")
    return vocals
