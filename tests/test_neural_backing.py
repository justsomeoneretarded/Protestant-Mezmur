"""Source-conditioned AI generation contracts without network, model downloads or video uploads."""
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mezmur import arrangement, neural_backing
from mezmur import separate as separation
from mezmur.harmony import ArrangementError


@pytest.mark.parametrize("duration", [4.913, 10, 16.713, 71, 72, 73, 143, 383, 901])
def test_chunk_windows_cover_exact_source_without_tiny_tails_or_new_bars(duration):
    windows = neural_backing.chunk_windows(duration, cues=[{"end": 67.3}, {"end": float("nan")}, {}])
    assert windows[0][0] == 0 and windows[-1][1] == duration
    assert all(0 <= a < b <= duration for a, b in windows)
    assert all(a1 < a2 < b1 <= b2 for (a1, b1), (a2, b2) in zip(windows, windows[1:]))
    assert all(b - a >= min(10, duration) for a, b in windows)


@pytest.mark.parametrize("values", [(0, 72, 4), (float("nan"), 72, 4), (30, float("inf"), 4),
                                  (30, 15, 4), (30, 72, 30), (30, 72, -1)])
def test_invalid_ai_windows_are_rejected(values):
    with pytest.raises(ArrangementError):
        neural_backing.chunk_windows(values[0], maximum=values[1], overlap=values[2])


def test_missing_engine_is_actionable_without_requesting_a_paid_api(tmp_path, monkeypatch):
    monkeypatch.setenv("MEZMUR_AI_ROOT", str(tmp_path))
    with pytest.raises(ArrangementError, match="setup_ai_backing.py"):
        neural_backing.engine_root()


@pytest.fixture
def fake_ai(monkeypatch, tmp_path):
    sr, seconds = 16000, 32
    reference = tmp_path / "reference.wav"
    t = np.arange(sr * seconds) / sr
    signal = np.column_stack([0.1 * np.sin(2 * np.pi * 220 * t)] * 2)
    sf.write(reference, signal, sr, subtype="FLOAT")
    root = tmp_path / "engine"
    root.mkdir()
    manifest = {"engine_commit": neural_backing.ENGINE_COMMIT, "model_revision": "tested"}
    calls = {"requests": [], "source": [], "cleanup": []}
    monkeypatch.setattr(neural_backing, "engine_root", lambda: (root, manifest))
    monkeypatch.setattr(neural_backing, "chunk_windows", lambda *a: [(0, 20), (16, 32)])

    def synth(command, **kwargs):
        request = Path(command[command.index("--request") + 1])
        data = json.loads(request.read_text())
        source = Path(command[command.index("--src-audio") + 1])
        calls["requests"].append(data)
        calls["source"].append(source)
        n = sf.info(source).frames
        value = 0.1 if data["audio_cover_strength"] == 0.8 else 0.2
        sf.write(request.with_name(request.stem + "0.wav"), np.full((n, 2), value, dtype=np.float32),
                 sr, subtype="FLOAT")

    monkeypatch.setattr(neural_backing.subprocess, "run", synth)
    monkeypatch.setattr(neural_backing, "assess_candidate", lambda audio, *a, **k: {
        "source_harmony_affinity": 0.8, "obvious_harmonic_drift": False,
        "selection_score": float(audio.mean()) * 2, "score_is_musical_quality": False})

    def clean_voice(source, work, **kwargs):
        assert work.is_dir(), "The native separator requires its working directory to exist"
        calls["cleanup"].append((source, kwargs))
        return source

    monkeypatch.setattr(separation, "separate", clean_voice)

    def instrument(source, target):
        shutil.copy2(source, target)
        return target

    monkeypatch.setattr(separation, "instrumental_stem", instrument)
    return {"sr": sr, "reference": reference, "target": tmp_path / "instrumental.wav",
            "work": tmp_path / "work", "calls": calls}


def test_ai_stitches_on_original_clock_selects_candidates_and_never_receives_real_voice(fake_ai):
    case = fake_ai
    before = hashlib.sha256(case["reference"].read_bytes()).hexdigest()
    cues = [{"start": 1, "end": 4, "chorus": True}]
    saved = copy.deepcopy(cues)
    report = neural_backing.compose(case["reference"], case["target"], case["work"], [],
                                     {"tempo_bpm": 92}, case["sr"], 32, cues=cues, seed=40)
    audio, sr = sf.read(case["target"], always_2d=True)
    assert len(audio) == 32 * sr and np.allclose(audio, 0.2, atol=1e-6)
    assert cues == saved and hashlib.sha256(case["reference"].read_bytes()).hexdigest() == before
    assert len(case["calls"]["requests"]) == 4
    for data in case["calls"]["requests"]:
        assert data["task_type"] == "cover-nofsq" and data["lyrics"] == "[Instrumental]"
        assert data["bpm"] == 92 and data["peak_clip"] == 0
        assert data["synth_model"] == neural_backing.MODEL_NAME + ".gguf"
        assert "bass" in data["caption"] and "used sparingly" in data["caption"]
        assert "strumming" not in data["caption"]
    assert report["vocal_waveform_enters_model"] is False and report["generated_vocal_cleanup"]
    assert report["no_midi_loop_fallback"] and report["phrase_count"] == 2
    assert case["calls"]["cleanup"][0][0] == case["target"]
    assert case["calls"]["cleanup"][0][1]["require_vocals"] is True
    diagnostic = json.loads((case["target"].parent / "ai-backing/candidate-report.json").read_text())
    assert len([r for r in diagnostic["candidates"] if "selected_candidate" in r]) == 2


def test_failed_ai_candidates_do_not_fall_back_to_original_music_or_a_midi_loop(fake_ai, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, ["ace-synth"])

    monkeypatch.setattr(neural_backing.subprocess, "run", fail)
    with pytest.raises(ArrangementError, match="Neither AI candidate"):
        neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                               fake_ai["sr"], 32)
    assert not fake_ai["target"].exists() and not fake_ai["calls"]["cleanup"]


def test_ai_duration_mismatch_is_not_hidden_by_length_padding(fake_ai, monkeypatch):
    def synth(command, **kwargs):
        request = Path(command[command.index("--request") + 1])
        sf.write(request.with_name(request.stem + "0.wav"), np.ones((fake_ai["sr"] * 5, 2)),
                 fake_ai["sr"], subtype="FLOAT")

    monkeypatch.setattr(neural_backing.subprocess, "run", synth)
    with pytest.raises(ArrangementError, match="Neither AI candidate"):
        neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                               fake_ai["sr"], 32)
    diagnostic = json.loads((fake_ai["target"].parent / "ai-backing/candidate-report.json").read_text())
    assert all("reference duration" in record["reason"] for record in diagnostic["candidates"])


def test_obvious_harmonic_drift_cannot_be_marked_as_success(fake_ai, monkeypatch):
    monkeypatch.setattr(neural_backing, "assess_candidate", lambda *a, **k: {
        "source_harmony_affinity": 0.9, "obvious_harmonic_drift": True, "selection_score": 0.9})
    with pytest.raises(ArrangementError, match="Neither AI candidate"):
        neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                               fake_ai["sr"], 32)


def test_native_candidate_metrics_are_not_called_musical_quality_or_lyric_accuracy():
    sr = 22050
    t = np.arange(sr * 6) / sr
    mono = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 330 * t)
    stereo = np.column_stack([mono, mono]).astype(np.float32)
    report = neural_backing.assess_candidate(stereo, stereo, sr)
    assert report["source_harmony_affinity"] > 0.99
    assert not report["score_is_musical_quality"] and not report["obvious_harmonic_drift"]
    with pytest.raises(ArrangementError, match="silent"):
        neural_backing.assess_candidate(np.zeros_like(stereo), stereo, sr)


def test_ai_rebuild_uses_no_soundfont_and_keeps_original_voice_and_fractional_duration(tmp_path, monkeypatch):
    sr, seconds = 24000, 5.137
    n = round(sr * seconds)
    signal = (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    original, voice, reference = [tmp_path / name for name in ("source.wav", "voice.wav", "reference.wav")]
    for path in (original, voice, reference):
        sf.write(path, np.column_stack([signal, signal]), sr, subtype="FLOAT")
    voice_hash = hashlib.sha256(voice.read_bytes()).hexdigest()
    monkeypatch.setattr(arrangement, "soundfont_path", lambda: pytest.fail("AI used the old SoundFont engine"))
    monkeypatch.setattr(arrangement, "harmony_plan", lambda *a, **k: ([], {"tuning_cents": 0}))

    def compose(source, target, work, plan, analysis, rate, duration, **kwargs):
        assert source == reference and duration == n / sr
        assert kwargs["separator_model"] == "chosen.onnx"
        sf.write(target, np.full((n, 2), 0.02, dtype=np.float32), rate, subtype="FLOAT")
        return {"engine": "ACE-Step 1.5 / acestep.cpp", "arrangement_style": "source-conditioned-ai-worship-band"}

    monkeypatch.setattr(neural_backing, "compose", compose)
    target = tmp_path / "output/master.wav"
    report = arrangement.rebuild(original, voice, target, tmp_path / "work", instrumental=reference,
                                 engine="ace-step", separator_model="chosen.onnx")
    assert sf.info(target).frames == n and report["duration_locked"] and report["soundfont"] is None
    assert report["vocal_offset_samples"] == report["vocal_pitch_shift_semitones"] == 0
    assert report["vocal_tempo_factor"] == 1 and report["true_peak_dbfs"] <= -0.999
    assert report["backing_target_db_relative_to_vocal"] == -7.0
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == voice_hash


def test_overlap_is_short_beat_aligned_blend_not_four_seconds_of_double_drums():
    windows = [(0, 20), (16, 32)]
    sr = 1000
    left = neural_backing.blend_weights(windows, 0, sr, beats=[17.8, 18.3], bpm=120)
    right = neural_backing.blend_weights(windows, 1, sr, beats=[17.8, 18.3], bpm=120)
    assert np.allclose(left[16 * sr:] + right[:4 * sr], 1)
    mixed = (left[16 * sr:] > 0.01) & (right[:4 * sr] > 0.01)
    assert mixed.sum() / sr <= 0.35
    assert left[17 * sr] == 1 and right[3 * sr] == 1


def test_ai_cannot_overwrite_the_actual_reference_even_before_model_loading(tmp_path, monkeypatch):
    source = tmp_path / "reference.wav"
    source.write_bytes(b"original")
    monkeypatch.setattr(neural_backing, "engine_root", lambda: pytest.fail("Overwrite check was too late"))
    with pytest.raises(ArrangementError, match="must not overwrite"):
        neural_backing.compose(source, source, tmp_path / "work", [], {}, 16000, 10)
    assert source.read_bytes() == b"original"


def test_style_freedom_wins_over_near_copy_when_harmony_scores_are_close(fake_ai, monkeypatch):
    monkeypatch.setattr(neural_backing, "assess_candidate", lambda audio, *a, **k: {
        "source_harmony_affinity": 0.90 if audio.mean() < 0.15 else 0.95,
        "vocal_clash_proxy": 0.01, "obvious_harmonic_drift": False,
        "selection_score": 0.90 if audio.mean() < 0.15 else 0.95})
    neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                           fake_ai["sr"], 32)
    audio = sf.read(fake_ai["target"])[0]
    assert np.allclose(audio, 0.1, atol=1e-6)


def test_conflicting_source_and_vocal_rankings_never_leave_an_empty_candidate_set(fake_ai, monkeypatch):
    monkeypatch.setattr(neural_backing, "assess_candidate", lambda audio, *a, **k: {
        "source_harmony_affinity": 0.9 if audio.mean() < 0.15 else 0.6,
        "vocal_clash_proxy": 0.2 if audio.mean() < 0.15 else 0.0,
        "obvious_harmonic_drift": False,
        "selection_score": 0.9 if audio.mean() < 0.15 else 0.6})
    report = neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                                    fake_ai["sr"], 32)
    assert report["phrase_count"] == 2
    assert len(sf.read(fake_ai["target"])[0]) == 32 * fake_ai["sr"]


def test_success_exit_without_a_readable_wav_is_a_rejected_candidate_not_a_crash(fake_ai, monkeypatch):
    monkeypatch.setattr(neural_backing.subprocess, "run", lambda *a, **k: None)
    with pytest.raises(ArrangementError, match="Neither AI candidate"):
        neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                               fake_ai["sr"], 32)
    assert not fake_ai["target"].exists()
    records = json.loads((fake_ai["target"].parent / "ai-backing/candidate-report.json").read_text())["candidates"]
    assert len(records) == 2 and all(record["rejected"] for record in records)


def test_single_strength_true_cover_outputs_that_version_without_auto_selection(fake_ai):
    report = neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                                    fake_ai["sr"], 32, task_type="cover", cover_strengths=(0.5,),
                                    keep_instrumentals=True)
    assert len(fake_ai["calls"]["requests"]) == 2
    assert all(request["task_type"] == "cover" and request["audio_cover_strength"] == 0.5
               for request in fake_ai["calls"]["requests"])
    assert report["cover_strengths"] == [0.5] and report["candidates_per_phrase"] == 1
    assert report["source_fit_checks_enforced"]
    assert (fake_ai["target"].parent / "ai-backing/before-vocal-cleanup.wav").is_file()
    assert (fake_ai["target"].parent / "ai-backing/after-vocal-cleanup.wav").is_file()


def test_full_song_listening_mode_flags_fit_concerns_without_rejecting_requested_version(fake_ai, monkeypatch):
    monkeypatch.setattr(neural_backing, "assess_candidate", lambda *a, **k: {
        "source_harmony_affinity": 0.25, "obvious_harmonic_drift": True, "selection_score": 0.25})
    report = neural_backing.compose(fake_ai["reference"], fake_ai["target"], fake_ai["work"], [], {},
                                    fake_ai["sr"], 32, task_type="cover", cover_strengths=(0.35,),
                                    enforce_source_fit=False)
    assert not report["source_fit_checks_enforced"]
    assert report["cleanup_harmony_metrics"]["obvious_harmonic_drift"]
    assert sf.info(fake_ai["target"]).frames == 32 * fake_ai["sr"]


@pytest.mark.parametrize("settings", [
    {"task_type": "text2music"}, {"cover_strengths": ()}, {"cover_strengths": (0,)},
    {"cover_strengths": (1.1,)}, {"cover_strengths": (float("nan"),)},
    {"cover_strengths": (0.5, 0.4, 0.3)},
])
def test_invalid_experimental_settings_fail_before_engine_loading(tmp_path, monkeypatch, settings):
    monkeypatch.setattr(neural_backing, "engine_root", lambda: pytest.fail("Invalid settings reached model loading"))
    with pytest.raises(ArrangementError):
        neural_backing.compose(tmp_path / "reference.wav", tmp_path / "output.wav",
                               tmp_path / "work", [], {}, 44100, 30, **settings)
