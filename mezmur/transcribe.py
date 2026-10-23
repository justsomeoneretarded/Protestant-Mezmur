"""Transcribe short overlapping chunks; all timestamps stay on the source timeline."""
import math
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import required, run, write_json

MODEL = "whisper-large-v3"
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


def transcribe(audio: Path, seconds: float, folder: Path):
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {required('GROQ_API_KEY')}"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"], respect_retry_after_header=True,
    )))
    results = []
    for index, (core_start, core_end, start, end) in enumerate(chunks(seconds)):
        chunk = folder / f"chunk-{index}.wav"
        run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(start), "-i", str(audio),
             "-t", str(end - start), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(chunk)])
        with chunk.open("rb") as handle:
            response = session.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                files={"file": (chunk.name, handle, "audio/wav")},
                data=[("model", MODEL), ("language", "am"), ("temperature", "0"),
                      ("response_format", "verbose_json"),
                      ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")],
                timeout=(30, 300),
            )
        # Do not log response bodies, which can contain private source content.
        response.raise_for_status()
        results.append({"offset": start, "core_start": core_start, "core_end": core_end,
                        "response": response.json()})
        chunk.unlink()
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
