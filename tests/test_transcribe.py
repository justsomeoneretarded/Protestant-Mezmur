import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import wave

import numpy as np
import pytest
import torch

from mezmur import transcribe
from mezmur.audio import read_audio
from mezmur.quality import TranscriptionQualityError


def test_real_audio_decoder_reads_pcm(tmp_path):
    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)
    samples = read_audio(audio)
    assert samples.shape == (1600,)
    assert not samples.any()


def test_local_amharic_model_is_loaded_once(monkeypatch):
    calls = []
    model = NS(eval=lambda: "model")
    processor_factory = NS(from_pretrained=lambda *a, **kw: calls.append(("processor", a, kw)) or "processor")
    model_factory = NS(from_pretrained=lambda *a, **kw: calls.append(("model", a, kw)) or model)
    monkeypatch.setitem(sys.modules, "transformers", NS(AutoProcessor=processor_factory, AutoModelForCTC=model_factory))
    transcribe.load_model.cache_clear()
    try:
        assert transcribe.load_model() == ("processor", "model")
        assert transcribe.load_model() == ("processor", "model")
        assert len(calls) == 2
        assert calls[1][1] == ("badrex/Ethio-ASR-amharic",)
        assert calls[1][2]["use_safetensors"] is True
    finally:
        transcribe.load_model.cache_clear()


def test_blank_intro_is_not_assigned_lyrics():
    class Tokenizer:
        pad_token_id = 0

        def decode(self, ids, output_word_offsets):
            assert ids[:8] == [0] * 8
            return NS(text="አሜን", word_offsets=[{"word": "አሜን", "start_offset": 8, "end_offset": 13}])

    ids = [0] * 8 + [1, 1, 0, 2, 2, 0, 0]
    logits = torch.full((15, 4), -10.0)
    for index, token in enumerate(ids):
        logits[index, token] = 10
    result = transcribe.decode_words(NS(tokenizer=Tokenizer()), logits, 15)
    assert result["words"][0]["start"] == 8
    assert result["words"][0]["end"] == 13
    assert result["words"][0]["probability"] > 0.99


@pytest.mark.parametrize("text,score", [("አሜን", 0), ("ስስስስስ", 10)])
def test_weak_or_hissing_words_are_rejected(text, score):
    tokenizer = NS(pad_token_id=0, decode=lambda *a, **kw: NS(
        text=text, word_offsets=[{"word": text, "start_offset": 0, "end_offset": 3}]))
    logits = torch.zeros((4, 4))
    logits[:, 1] = score
    result = transcribe.decode_words(NS(tokenizer=tokenizer), logits, 4)
    assert not result["words"]
    assert len(result["rejected_words"]) == 1


def fake_model(monkeypatch, seconds):
    calls = []

    def processor(audio, **kwargs):
        calls.append((len(audio), kwargs))
        return {}

    monkeypatch.setattr(transcribe, "read_audio", lambda *a: np.zeros(seconds * 16000))
    monkeypatch.setattr(transcribe, "load_model", lambda: (
        processor, lambda **kw: NS(logits=torch.zeros((1, 10, 4)))))
    return calls


def test_chunks_preserve_native_absolute_timing(tmp_path, monkeypatch):
    calls = fake_model(monkeypatch, 44)
    monkeypatch.setattr(transcribe, "decode_words", lambda *a: {
        "words": [{"word": "አሜን", "start": 3, "end": 4}], "rejected_words": []})
    words = transcribe.transcribe(tmp_path / "source.wav", 44, tmp_path)
    assert [w["start"] for w in words] == [3, 21, 41]
    assert [size for size, _ in calls] == [22 * 16000, 24 * 16000, 6 * 16000]
    assert all(kwargs["sampling_rate"] == 16000 for _, kwargs in calls)
    assert json.loads((tmp_path / "quality.json").read_text())["passed"]


def test_character_collapse_fails_before_rendering(tmp_path, monkeypatch):
    fake_model(monkeypatch, 10)
    monkeypatch.setattr(transcribe, "decode_words", lambda *a: {
        "words": [{"word": "ስስ", "start": i / 10, "end": i / 10 + 0.05} for i in range(25)],
        "rejected_words": [],
    })
    with pytest.raises(TranscriptionQualityError, match="repeated character"):
        transcribe.transcribe(tmp_path / "source.wav", 10, tmp_path)
    assert not json.loads((tmp_path / "quality.json").read_text())["passed"]
    assert (tmp_path / "transcription.raw.json").exists()
