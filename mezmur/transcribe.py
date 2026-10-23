"""Transcribe short overlapping chunks; all timestamps stay on the source timeline."""
import math
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .common import run, write_json

MODEL = "large-v3"
CORE_SECONDS = 240
OVERLAP_SECONDS = 4


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
                words.append({"word": text, "start": max(0, start), "end": min(seconds, end)})
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
    from faster_whisper import WhisperModel

    return WhisperModel(
        MODEL, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1,
        download_root=".model-cache/whisper",
    )


def transcribe(audio: Path, seconds: float, folder: Path):
    model = load_model()
    results = []
    for index, (core_start, core_end, start, end) in enumerate(chunks(seconds)):
        chunk = folder / f"chunk-{index}.wav"
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", str(audio),
             "-t", str(end - start), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(chunk)])
        print(f"Transcribing chunk {index + 1} locally ({core_start:.0f}-{core_end:.0f}s)", flush=True)
        try:
            segments, info = model.transcribe(
                str(chunk), language="am", task="transcribe", beam_size=5,
                temperature=0, word_timestamps=True, condition_on_previous_text=False,
                # Speech VAD can discard held notes and soft singing.
                vad_filter=False,
            )
            # Inference is lazy: consume the iterator before deleting its audio.
            segments = [asdict(segment) for segment in segments]
            response = {
                "language": info.language, "language_probability": info.language_probability,
                "segments": segments,
                "words": [word for segment in segments for word in (segment.get("words") or [])],
            }
            results.append({"offset": start, "core_start": core_start, "core_end": core_end,
                            "response": response})
        finally:
            chunk.unlink(missing_ok=True)
    write_json(folder / "transcription.raw.json", results)
    return merge_words(results, seconds)


def separate(audio: Path, folder: Path) -> Path:
    import sys

    run([sys.executable, "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
         "-d", "cpu", "--shifts", "0", "-j", "2", "-o", str(folder / "stems"), str(audio)])
    vocals = folder / "stems" / "htdemucs" / audio.stem / "vocals.wav"
    if not vocals.is_file():
        raise RuntimeError("Vocal separation did not produce vocals.wav")
    return vocals
