"""Keep recognition on clean audio and carry only verified timings into render effects."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mezmur import audio, song


def timing_map(output_frames=(0, 1200, 2100, 4500)):
    return {"enabled": True, "sample_rate": 1000,
            "input_frames": [0, 1000, 2000, 4000], "output_frames": list(output_frames)}


def test_retime_words_uses_local_anchors_and_preserves_annotations():
    words = [{"word": "one", "start": 0.5, "end": 0.75, "line": 0, "estimated": False},
             {"word": "two", "start": 2.5, "end": 3.0, "line": 1, "confidence": 0.9}]
    original = copy.deepcopy(words)
    mapped = audio.retime_words(words, timing_map(), 4, 4.5)
    np.testing.assert_allclose([(w["start"], w["end"]) for w in mapped], [(0.6, 0.9), (2.7, 3.3)])
    assert mapped[0]["estimated"] is False and mapped[1]["confidence"] == 0.9
    assert mapped[0]["line"] == 0 and mapped[1]["word"] == "two"
    assert words == original and mapped[0] is not words[0]


@pytest.mark.parametrize("end_frame", [3800, 4000, 4200])
def test_local_retiming_applies_even_when_total_duration_is_unchanged(end_frame):
    metadata = timing_map((0, 1100, 2000, end_frame))
    mapped = audio.retime_words([{"start": 1.0, "end": 2.0}], metadata, 4, end_frame / 1000)
    assert mapped[0] == {"start": 1.1, "end": 2.0}


def test_retime_caps_smoothed_last_word_at_the_render_end():
    mapped = audio.retime_words([{"start": 3.95, "end": 4.07}], timing_map(), 4, 4.5)
    assert mapped[0]["end"] == 4.5
    assert 0 < mapped[0]["start"] < 4.5


def test_retime_empty_annotations():
    assert audio.retime_words([], timing_map(), 4, 4.5) == []


@pytest.mark.parametrize("patch", [
    {"sample_rate": 0}, {"sample_rate": float("nan")}, {"sample_rate": None},
    {"input_frames": []}, {"input_frames": [0]}, {"input_frames": [[0, 4000]]},
    {"input_frames": [0, 1000, 1000, 4000]}, {"input_frames": [0, 1000, 900, 4000]},
    {"input_frames": [1, 1000, 2000, 4000]}, {"output_frames": [0, 4500]},
    {"output_frames": [0, 1200, float("inf"), 4500]}, {"output_frames": [1, 1200, 2100, 4500]},
    {"input_frames": [0, 1000, 2000, 4100]}, {"output_frames": [0, 1200, 2100, 4600]},
])
def test_retime_rejects_invalid_or_inconsistent_anchors(patch):
    with pytest.raises(ValueError):
        audio.retime_words([{"start": 0.5, "end": 1}], timing_map() | patch, 4, 4.5)


@pytest.mark.parametrize("start,end", [
    (-0.1, 1), (1, 1), (2, 1), (float("nan"), 1), (0, float("inf")), (4, 4.1),
])
def test_retime_rejects_invalid_word_times(start, end):
    with pytest.raises(ValueError):
        audio.retime_words([{"start": start, "end": end}], timing_map(), 4, 4.5)


@pytest.mark.parametrize("source_duration,output_duration", [(0, 4.5), (4, -1), (4, float("nan"))])
def test_retime_rejects_invalid_durations(source_duration, output_duration):
    with pytest.raises(ValueError):
        audio.retime_words([], timing_map(), source_duration, output_duration)


def test_saved_time_map_agrees_with_native_augmentation_mapping():
    from mezmur.augmentation import augment_audio

    sr = 16000
    samples = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    result = augment_audio(samples, sr, seed=22, tempo_factor=0.96, preserve_duration=False)
    words = [{"start": 0.3, "end": 0.9}, {"start": 1.2, "end": 1.8}]
    mapped = audio.retime_words(words, result.metadata(), 2, len(result.audio) / sr)
    expected = result.map_times(np.array([[0.3, 0.9], [1.2, 1.8]]))
    np.testing.assert_allclose([[w["start"], w["end"]] for w in mapped], expected)


@pytest.fixture
def clean_pipeline(monkeypatch, tmp_path):
    """Exercise produce() through its real acceptance gate and subtitle writers, without network/model downloads."""
    inp, out, work = [tmp_path / p for p in ("input", "output", "work")]
    inp.mkdir()
    source = inp / "audio.wav"
    source.touch()
    (inp / "info.json").write_text(json.dumps({"message_id": 22}))
    augmented, vocals = tmp_path / "augmented.wav", tmp_path / "vocals.wav"
    calls = {"events": [], "masters": [], "input": inp, "output": out, "work": work}
    report = {"matched_share": 0.86, "gain_per_vocal_frame": 0.23,
              "lines_used": 2, "lines_total": 2, "complete_share": 0.96}
    filled = {"estimated_words": 0}
    aligned = [{"word_index": 0, "line": 0, "phrase": 0, "start": 0.5, "end": 0.75},
               {"word_index": 1, "line": 1, "phrase": 1, "start": 2.5, "end": 3.0}]
    candidate = {"text": "one\ntwo", "stanzas": [{"lines": ["one", "two"], "chorus": True}],
                 "trusted": False, "source": "library:test", "url": ""}
    calls.update(report=report, filled=filled, candidates=[candidate])

    def master(path, target):
        assert target == work / "master.wav"
        assert path in (source, augmented)
        calls["masters"].append(path)
        calls["events"].append("master-original" if path == source else "master-render")
        return {}

    def duration(path):
        assert path == work / "master.wav"
        return calls.get("render_duration", 4) if calls["masters"][-1] == augmented else 4

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

    def prepare(path, folder, *, augment, seed, style, noise_profile, space_effect):
        assert path == source and folder == work and augment and seed == 22
        assert calls["events"][-1] == "lrc-original"
        assert noise_profile == "hiss" and space_effect == "room"
        calls["events"].append("augment")
        calls["render_duration"] = 4.5 if style == "musical" else 4
        frames = (0, 1200, 2100, 4500) if style == "musical" else (0, 1100, 2000, 4000)
        return augmented, timing_map(frames) | {"seed": seed, "style": style}

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
        assert track == work / "master.wav"
        calls["events"].append("render")

    def kit(out, meta, stanzas, cues, seconds, *args):
        calls["kit_cues"], calls["kit_duration"] = copy.deepcopy(cues), seconds

    monkeypatch.delenv("MEZMUR_DEBUG", raising=False)
    monkeypatch.setattr(audio, "tags", lambda _: {})
    monkeypatch.setattr(audio, "master", master)
    monkeypatch.setattr(audio, "prepare_source", prepare)
    monkeypatch.setattr(song, "common_duration", duration)
    monkeypatch.setattr(song, "separate", separate)
    monkeypatch.setattr(song.asr, "vocabulary", lambda: {"char_to_id": {}, "blank": 0, "delimiter": 1})
    monkeypatch.setattr(song.asr, "frame_log_probs", infer)
    monkeypatch.setattr(song.asr, "vocal_energy", lambda *a: np.ones(200))
    monkeypatch.setattr(song.asr, "greedy_words", lambda *a: [{"word": "one"}, {"word": "two"}])
    monkeypatch.setattr(song.names, "resolve", lambda _: {"title_latin": "Test"})
    monkeypatch.setattr(song.names, "learn_from_titles", lambda record, _: record)
    monkeypatch.setattr(song.names, "finalize", lambda record: record)
    monkeypatch.setattr(song.web, "youtube_search", lambda _: [])
    monkeypatch.setattr(song, "gather", lambda *a, **k: (calls["candidates"], [], {}, {}))
    monkeypatch.setattr(song, "similarity", lambda *a: {"score": 0.45})
    monkeypatch.setattr(song, "build_target", lambda *a: None)
    monkeypatch.setattr(song, "align_lyrics", align)
    monkeypatch.setattr(song, "complete_with_lyrics", lambda words, *a, **k: (words, dict(filled)))
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


@pytest.mark.parametrize("enabled,style", [(False, "musical"), (True, "musical"), (True, "subtle")])
def test_verified_clean_lyrics_drive_every_final_output(clean_pipeline, enabled, style):
    calls = clean_pipeline
    result = song.produce(calls["input"], calls["output"], calls["work"],
                          augment_audio=enabled, augmentation_style=style)
    out = calls["output"]
    assert result["mode"] == "lyrics-aligned"
    assert song.ACCEPT_GAIN == 0.18
    expected_events = ["master-original", "separate-original", "recognise-original",
                       "verify-original", "lrc-original"]
    if enabled:
        expected_events += ["augment", "master-render"]
    assert calls["events"] == expected_events + ["render"]
    expected_start = 0.6 if enabled and style == "musical" else 0.55 if enabled else 0.5
    words = json.loads((out / "words.json").read_text())
    assert words[0]["start"] == pytest.approx(expected_start)
    for cues in (calls["ass_cues"], calls["video_cues"], calls["kit_cues"]):
        assert cues[0]["start"] == pytest.approx(expected_start)
        assert all(c["chorus"] for c in cues)
        assert cues[0]["words"][0]["start"] == pytest.approx(expected_start)
    timestamp = f"00:00:00,{round(expected_start * 1000):03d}"
    for name in ("am.srt", "am-Latn.srt"):
        assert timestamp in (out / "captions" / name).read_text()
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["original_duration_seconds"] == 4
    assert metadata["audio_augmentation"]["enabled"] is enabled
    expected_duration = calls.get("render_duration", 4)
    assert metadata["duration_seconds"] == calls["video_duration"] == calls["kit_duration"] == expected_duration
    quality = json.loads((out / "quality.json").read_text())
    assert quality["alignment_timeline"] == "original"
    assert quality["candidates"][0]["gain_per_vocal_frame"] == 0.23
    assert quality["candidates"][0]["accepted"]
    assert (out / "augmentation.json").exists() is enabled
    if enabled:
        assert json.loads((out / "augmentation.json").read_text())["lyrics_timing"] == "mapped_from_original"
    else:
        assert metadata["audio_augmentation"]["lyrics_timing"] == "original"


@pytest.mark.parametrize("failure", ["weak-match", "missing-lyrics", "estimated-timings"])
def test_unverified_songs_skip_before_effects(clean_pipeline, failure):
    calls = clean_pipeline
    if failure == "weak-match":
        calls["report"]["gain_per_vocal_frame"] = 0.16
    elif failure == "missing-lyrics":
        calls["candidates"].clear()
    else:
        calls["filled"]["estimated_words"] = 2
    result = song.produce(calls["input"], calls["output"], calls["work"], augment_audio=True)
    assert result["mode"] == "skipped"
    assert "augment" not in calls["events"] and "render" not in calls["events"]
    assert not (calls["output"] / "augmentation.json").exists()
    assert not (calls["output"] / "captions").exists()
    assert (calls["output"] / "skipped.json").is_file()
