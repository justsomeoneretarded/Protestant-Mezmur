from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mezmur import transcribe


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass
class Segment:
    words: list[Word]


def test_real_audio_decoder_reads_pcm(tmp_path):
    import wave
    from faster_whisper.audio import decode_audio

    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)
    samples = decode_audio(str(audio), sampling_rate=16000)
    assert samples.shape == (1600,)
    assert not samples.any()


def test_local_model_uses_cpu_int8_and_is_reused(monkeypatch):
    calls = []

    def model(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=model))
    transcribe.load_model.cache_clear()
    try:
        assert transcribe.load_model() is transcribe.load_model()
        assert calls == [(("large-v3",), {
            "device": "cpu", "compute_type": "int8", "cpu_threads": 2,
            "num_workers": 1, "download_root": ".model-cache/whisper",
        })]
    finally:
        transcribe.load_model.cache_clear()


def test_local_amharic_words_keep_original_timeline(tmp_path, monkeypatch):
    calls = []

    def fake_ffmpeg(args):
        Path(args[-1]).write_bytes(b"fake wav bytes")

    class Model:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)

            def segments():
                assert Path(audio).exists()
                yield Segment([Word("ሃሌሉያ", 6, 7)])

            return segments(), SimpleNamespace(language="am", language_probability=1.0)

    monkeypatch.setattr(transcribe, "run", fake_ffmpeg)
    monkeypatch.setattr(transcribe, "load_model", lambda: Model())
    words = transcribe.transcribe(tmp_path / "audio.wav", 300, tmp_path)
    assert [word["start"] for word in words] == [6, 242]
    assert len(calls) == 2
    assert all(call["language"] == "am" and call["word_timestamps"] is True for call in calls)
    assert all(call["vad_filter"] is False and call["condition_on_previous_text"] is False for call in calls)
    assert not list(tmp_path.glob("chunk-*.wav"))
    assert len(json.loads((tmp_path / "transcription.raw.json").read_text())) == 2


def test_failed_inference_removes_chunk(tmp_path, monkeypatch):
    def fake_ffmpeg(args):
        Path(args[-1]).write_bytes(b"fake wav bytes")

    class Model:
        def transcribe(self, *args, **kwargs):
            def segments():
                raise RuntimeError("inference failed")
                yield

            return segments(), SimpleNamespace(language="am", language_probability=1.0)

    monkeypatch.setattr(transcribe, "run", fake_ffmpeg)
    monkeypatch.setattr(transcribe, "load_model", lambda: Model())
    with pytest.raises(RuntimeError, match="inference failed"):
        transcribe.transcribe(tmp_path / "audio.wav", 10, tmp_path)
    assert not list(tmp_path.glob("chunk-*.wav"))
