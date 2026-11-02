"""Recognize and render from the original source; low acoustic confidence is not a sole veto."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mezmur import audio, song


@pytest.fixture
def clean_pipeline(monkeypatch, tmp_path):
    """Exercise produce() through its real acceptance gate and subtitle writers, without network/model downloads."""
    inp, out, work = [tmp_path / p for p in ("input", "output", "work")]
    inp.mkdir()
    source = inp / "audio.wav"
    source.touch()
    (inp / "info.json").write_text(json.dumps({"message_id": 22}))
    vocals = tmp_path / "vocals.wav"
    calls = {"events": [], "input": inp, "output": out, "work": work, "source": source,
             "record": {"title_latin": "Test"}, "similarity": 0.45}
    report = {"matched_share": 0.86, "gain_per_vocal_frame": 0.23,
              "lines_used": 2, "lines_total": 2, "complete_share": 0.96}
    filled = {"estimated_words": 0}
    aligned = [{"word_index": 0, "line": 0, "phrase": 0, "start": 0.5, "end": 0.75},
               {"word_index": 1, "line": 1, "phrase": 1, "start": 2.5, "end": 3.0}]
    candidate = {"text": "one\ntwo", "stanzas": [{"lines": ["one", "two"], "chorus": True}],
                 "trusted": False, "source": "library:test", "url": ""}
    calls.update(report=report, filled=filled, candidates=[candidate])

    def duration(path):
        assert path == source
        return 4

    def separate(path, folder, **kwargs):
        assert path == source and folder == work
        calls["events"].append("separate-original")
        return vocals

    def infer(path, **kwargs):
        assert path == vocals
        calls["events"].append("recognise-original")
        return np.zeros((200, 2)), 0.02

    def align(*args, **kwargs):
        calls["events"].append("verify-original")
        return copy.deepcopy(aligned), dict(report)

    def lrc_check(cues, candidates, seconds):
        assert seconds == 4 and cues[0]["start"] == 0.5
        calls["events"].append("lrc-original")
        return {"median_seconds": 0}

    def write_ass(cues, path, **kwargs):
        calls["ass_cues"] = copy.deepcopy(cues)
        path.write_text("test subtitles")

    def renderer(width, height, fps, seconds, feats, sources, cues, meta, **kwargs):
        calls["video_cues"], calls["video_duration"] = copy.deepcopy(cues), seconds
        return SimpleNamespace(outro_start=seconds - 1)

    def render_video(renderer, track, *args, **kwargs):
        assert track == source
        calls["events"].append("render")

    def kit(out, meta, stanzas, cues, seconds, *args):
        calls["kit_cues"], calls["kit_duration"] = copy.deepcopy(cues), seconds

    monkeypatch.delenv("MEZMUR_DEBUG", raising=False)
    monkeypatch.setattr(audio, "tags", lambda _: {})
    monkeypatch.setattr(song, "common_duration", duration)
    monkeypatch.setattr(song, "separate", separate)
    monkeypatch.setattr(song.asr, "vocabulary", lambda: {"char_to_id": {}, "blank": 0, "delimiter": 1})
    monkeypatch.setattr(song.asr, "frame_log_probs", infer)
    monkeypatch.setattr(song.asr, "vocal_energy", lambda *a: np.ones(200))
    monkeypatch.setattr(song.asr, "greedy_words", lambda *a: [{"word": "one"}, {"word": "two"}])
    monkeypatch.setattr(song.names, "resolve", lambda _: dict(calls["record"]))
    monkeypatch.setattr(song.names, "learn_from_titles", lambda record, _: record)
    monkeypatch.setattr(song.names, "finalize", lambda record: record)
    monkeypatch.setattr(song.web, "youtube_search", lambda _: [])
    monkeypatch.setattr(song, "gather", lambda *a, **k: (calls["candidates"], [], {}, {}))
    monkeypatch.setattr(song, "similarity", lambda *a: {"score": calls["similarity"]})
    monkeypatch.setattr(song, "build_target", lambda *a: None)
    monkeypatch.setattr(song, "align_lyrics", align)

    def complete(words, *args, **kwargs):
        calls["events"].append("force-complete")
        return words, dict(filled)

    monkeypatch.setattr(song, "complete_with_lyrics", complete)
    monkeypatch.setattr(song, "lrc_check", lrc_check)
    monkeypatch.setattr(song, "photo_candidates", lambda *a: [])
    monkeypatch.setattr(song.backgrounds, "nature_still", lambda *a, **k: (None, None))
    monkeypatch.setattr(song.backgrounds, "fallback_scene", lambda *a, **k: (None, None))
    monkeypatch.setattr(song.backgrounds, "brightness", lambda _: 100)
    monkeypatch.setattr(song, "write_ass", write_ass)
    monkeypatch.setattr(song.thumbnail, "compose", lambda *a, **k: np.zeros((10, 10, 3), dtype=np.uint8))
    monkeypatch.setattr(song.thumbnail, "save", lambda *a: None)
    monkeypatch.setattr(song, "analyze", lambda *a, **k: SimpleNamespace(tempo=100))
    monkeypatch.setattr(song, "Renderer", renderer)
    monkeypatch.setattr(song, "render_video", render_video)
    monkeypatch.setattr(song, "write_kit", kit)
    return calls


def test_original_source_drives_every_output_without_effects(clean_pipeline):
    calls = clean_pipeline
    result = song.produce(calls["input"], calls["output"], calls["work"])
    out = calls["output"]
    assert result["mode"] == "lyrics-aligned"
    assert calls["events"] == ["separate-original", "recognise-original", "verify-original",
                               "force-complete", "lrc-original", "render"]
    words = json.loads((out / "words.json").read_text())
    assert words[0]["start"] == 0.5
    for cues in (calls["ass_cues"], calls["video_cues"], calls["kit_cues"]):
        assert cues[0]["start"] == 0.5 and all(c["chorus"] for c in cues)
    for name in ("am.srt", "am-Latn.srt"):
        assert "00:00:00,500" in (out / "captions" / name).read_text()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["audio_processing"]["source"] == "original"
    assert meta["audio_processing"]["effects_applied"] is False
    assert meta["audio_processing"]["loudness_normalization"] is False
    assert meta["lyrics_verification"]["method"] == "acoustic"
    assert not (out / "augmentation.json").exists()
    assert not (calls["work"] / "master.wav").exists()


def test_source_supported_011_is_force_aligned_even_when_completion_estimates_some_words(clean_pipeline):
    calls = clean_pipeline
    calls["record"] = {"title_latin": "Nefsen Befith", "artist_latin": "Kalkidan Lily Tilahun"}
    calls["candidates"][0].update(source="web:mezmuroch.com",
        url="https://mezmuroch.com/lyrics/kalkidan-tilahun/nefsen-befiteh", title="Nefsen Befiteh")
    calls["report"].update(matched_share=0.73, gain_per_vocal_frame=0.11, complete_share=0.57)
    calls["similarity"] = 0.217
    calls["filled"]["estimated_words"] = 1
    result = song.produce(calls["input"], calls["output"], calls["work"])
    assert result["mode"] == "lyrics-aligned"
    assert "force-complete" in calls["events"] and "render" in calls["events"]
    meta = json.loads((calls["output"] / "metadata.json").read_text())
    assert meta["lyrics_verification"]["method"] == "source-supported"
    assert meta["review"]["needs_review"]
    assert "estimated" in " ".join(meta["review"]["reasons"])
    quality = json.loads((calls["output"] / "quality.json").read_text())
    assert quality["candidates"][0]["accepted"]
    assert quality["alignment"]["gain_per_vocal_frame"] == 0.11


@pytest.mark.parametrize("failure", ["weak-unknown-source", "missing-lyrics", "estimated-unknown-source"])
def test_unsupported_songs_still_skip_before_rendering(clean_pipeline, failure):
    calls = clean_pipeline
    if failure == "weak-unknown-source":
        calls["report"]["gain_per_vocal_frame"] = 0.10
    elif failure == "missing-lyrics":
        calls["candidates"].clear()
    else:
        calls["filled"]["estimated_words"] = 2
    result = song.produce(calls["input"], calls["output"], calls["work"])
    assert result["mode"] == "skipped"
    assert "render" not in calls["events"]
    assert (calls["output"] / "skipped.json").is_file()


def test_song_cli_has_no_audio_effect_options(monkeypatch):
    monkeypatch.setattr(__import__("sys"), "argv", ["mezmur-song", "--input", "in", "--output", "out"])
    calls = []
    monkeypatch.setattr(song, "produce", lambda *a, **k: calls.append(k))
    song.main()
    assert not any("augment" in key or "noise" in key or "space" in key for key in calls[0])
