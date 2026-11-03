"""Native audio tests for the explicitly requested, CPU-only backing rebuild."""
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import yaml

from mezmur import arrangement
from mezmur import separate as separator

SR = 44100


def stereo_tone(seconds=6, pitch=220, amplitude=0.12):
    t = np.arange(int(SR * seconds)) / SR
    wave = (amplitude * np.sin(2 * np.pi * pitch * t)).astype(np.float32)
    return np.column_stack([wave, wave])


def test_rebuild_checkbox_is_typed_opt_in_and_passed_only_when_selected():
    doc = yaml.safe_load(Path(".github/workflows/generate.yml").read_text())
    events = doc.get("on", doc.get(True))
    for trigger in ("workflow_dispatch", "workflow_call"):
        assert events[trigger]["inputs"]["rebuild_music"]["type"] == "boolean"
        assert events[trigger]["inputs"]["rebuild_music"]["default"] is False
    job = doc["jobs"]["song"]
    install = next(s for s in job["steps"] if s.get("name", "").startswith("Install backing arranger"))
    assert install["if"] == "inputs.rebuild_music == true"
    assert "fluid-soundfont-gm" in install["run"] and ".[arrange]" in install["run"]
    render = next(s for s in job["steps"] if s.get("name", "").startswith("Make video"))
    assert render["env"]["REBUILD_MUSIC"] == "${{ inputs.rebuild_music && 'true' || 'false' }}"
    assert 'if [ "$REBUILD_MUSIC" = "true" ]; then args+=(--rebuild-music); fi' in render["run"]
    assert '"${args[@]}"' in render["run"]
    preflight = next(s for s in doc["jobs"]["fetch"]["steps"] if s.get("name") == "Validate music rebuild settings")
    assert preflight["if"] == "inputs.rebuild_music == true && inputs.separator == 'none'"


def test_required_stem_cannot_disable_separation(monkeypatch, tmp_path):
    monkeypatch.setattr(separator.subprocess, "run", lambda *a, **k: pytest.fail("Should fail before ffmpeg"))
    with pytest.raises(ValueError, match="requires vocal isolation"):
        separator.separate(tmp_path / "source.wav", tmp_path, model="none", require_vocals=True)


@pytest.mark.parametrize("required", [False, True])
def test_failed_separation_never_masquerades_as_vocals_in_rebuild_mode(monkeypatch, tmp_path, required):
    monkeypatch.setattr(separator.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(separator, "_run", lambda *a, **k: None)
    if required:
        with pytest.raises(RuntimeError, match="refusing to rebuild"):
            separator.separate(tmp_path / "source.wav", tmp_path, require_vocals=True)
    else:
        assert separator.separate(tmp_path / "source.wav", tmp_path) == tmp_path / "sep_input.wav"


def test_mix_preserves_vocal_waveform_and_zero_offset_with_constant_headroom_gain():
    vocals = stereo_tone(amplitude=0.95)
    backing = stereo_tone(pitch=330, amplitude=0.7)
    original = vocals.copy()
    mixed, report = arrangement.mix_tracks(vocals, backing, SR)
    expected = vocals.astype(float) * report["vocal_gain"] + backing.astype(float) * report["backing_gain"]
    np.testing.assert_allclose(mixed, expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(vocals, original)
    recovered = mixed.astype(float) - backing * report["backing_gain"]
    np.testing.assert_allclose(recovered, vocals * report["vocal_gain"], atol=2e-7)
    assert mixed.shape == vocals.shape and mixed.dtype == np.float32
    assert report["vocal_offset_samples"] == report["vocal_pitch_shift_semitones"] == 0
    assert report["vocal_tempo_factor"] == 1.0
    assert report["true_peak_dbfs"] <= -0.9999
    assert arrangement.true_peak(mixed, SR) <= arrangement.PEAK_LIMIT + 1e-6


@pytest.mark.parametrize("bad", ["silent-voice", "silent-music", "nan", "wrong-length"])
def test_mix_refuses_invalid_or_silent_tracks(bad):
    vocals, backing = stereo_tone(), stereo_tone(pitch=330)
    if bad == "silent-voice":
        vocals[:] = 0
    elif bad == "silent-music":
        backing[:] = 0
    elif bad == "nan":
        backing[0, 0] = np.nan
    else:
        backing = backing[:-1]
    with pytest.raises(arrangement.ArrangementError):
        arrangement.mix_tracks(vocals, backing, SR)


@pytest.mark.parametrize("samples", [
    np.zeros(22050 * 5, dtype=np.float32),
    np.ones(22050 * 3, dtype=np.float32),
    np.full(22050 * 5, np.nan, dtype=np.float32),
])
def test_invalid_music_analysis_fails_explicitly(samples):
    with pytest.raises(arrangement.ArrangementError):
        arrangement.harmony_plan(samples)


def test_midi_is_new_notes_on_absolute_timeline_and_skips_unreliable_harmony(tmp_path):
    import pretty_midi

    plan = [{"start": 0, "end": 2, "root": 0, "minor": False, "score": 0.95, "reliable": True},
            {"start": 2, "end": 4, "root": 1, "minor": False, "score": 0.4, "reliable": False},
            {"start": 4, "end": 6, "root": 9, "minor": True, "score": 0.9, "reliable": True}]
    target = tmp_path / "notes.mid"
    arrangement.write_midi(plan, target, seed=29)
    midi = pretty_midi.PrettyMIDI(str(target))
    assert [i.program for i in midi.instruments] == [0, 32, 89]
    for instrument in midi.instruments:
        assert instrument.notes
        assert all(0 <= n.start < n.end <= 6.001 for n in instrument.notes)
        assert not any(2.001 < n.start < 3.999 for n in instrument.notes)
    assert {n.pitch for n in midi.instruments[1].notes} == {36, 45}


def test_synthesizer_disables_added_room_chorus_and_uses_float_output(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(arrangement.shutil, "which", lambda _: "/usr/bin/fluidsynth")
    monkeypatch.setattr(arrangement.subprocess, "run", lambda command, **kwargs: commands.append(command))
    arrangement.render_instrumental("notes.mid", tmp_path / "music.wav", "font.sf2", SR)
    command = commands[0]
    assert command[command.index("-R") + 1] == command[command.index("-C") + 1] == "0"
    assert command[command.index("-O") + 1] == "float"
    assert command[command.index("-r") + 1] == str(SR)


def test_rebuild_cannot_overwrite_original_or_stem(tmp_path):
    for target in (tmp_path / "original.wav", tmp_path / "vocals.wav"):
        with pytest.raises(arrangement.ArrangementError, match="overwrite"):
            arrangement.rebuild(tmp_path / "original.wav", tmp_path / "vocals.wav", target, tmp_path / "work")


def test_duration_mismatch_fails_before_analysis_or_synthesis(monkeypatch, tmp_path):
    source, vocals = tmp_path / "original.wav", tmp_path / "vocals.wav"
    sf.write(source, stereo_tone(seconds=6), SR)
    sf.write(vocals, stereo_tone(seconds=5), SR)
    monkeypatch.setattr(arrangement, "soundfont_path", lambda: tmp_path / "font.sf2")
    monkeypatch.setattr(arrangement, "harmony_plan", lambda *a: pytest.fail("Bad timing reached composition"))
    with pytest.raises(arrangement.ArrangementError, match="duration"):
        arrangement.rebuild(source, vocals, tmp_path / "output.wav", tmp_path / "work")


def test_native_soundfont_rebuild_writes_lossless_master_without_modifying_source(tmp_path):
    import pretty_midi

    seconds = 16
    t = np.arange(SR * seconds) / SR
    source = np.zeros(len(t), dtype=np.float32)
    for index, pitches in enumerate(((48, 52, 55), (53, 57, 60), (55, 59, 62), (48, 52, 55))):
        a, b = index * SR * 4, (index + 1) * SR * 4
        for pitch in pitches:
            source[a:b] += 0.07 * np.sin(2 * np.pi * pretty_midi.note_number_to_hz(pitch) * t[a:b])
    for tick in np.arange(0.2, seconds - 0.1, 0.5):
        a = int(tick * SR)
        pulse = np.arange(int(SR * 0.035)) / SR
        source[a:a + len(pulse)] += (0.25 * np.sin(2 * np.pi * 1800 * pulse) * np.exp(-pulse * 160))
    original, stem = tmp_path / "source.wav", tmp_path / "vocals.wav"
    sf.write(original, np.column_stack([source, source]), SR, subtype="PCM_24")
    voice = stereo_tone(seconds=seconds, pitch=261.6256, amplitude=0.1)
    sf.write(stem, voice, SR, subtype="FLOAT")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    target = tmp_path / "out" / "rebuilt-audio.wav"
    report = arrangement.rebuild(original, stem, target, tmp_path / "work", seed=29)
    mixed, sr = sf.read(target, always_2d=True)
    assert sf.info(target).subtype == "FLOAT" and sr == SR
    assert mixed.shape == voice.shape and np.isfinite(mixed).all()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert report["original_recording_unchanged"] and not report["original_instrumental_reused"]
    assert report["beat_count"] >= 4 and report["reliable_harmony_share"] >= 0.55
    assert report["duration_seconds"] == seconds and report["vocal_offset_samples"] == 0
    assert report["true_peak_dbfs"] <= -0.9999
    assert (target.parent / "arrangement.mid").is_file()
    assert not np.allclose(mixed, voice)
