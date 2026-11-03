"""Native audio tests for the explicitly requested, CPU-only backing rebuild."""
import hashlib
import json
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
    assert "Guitar-led acoustic" in events["workflow_dispatch"]["inputs"]["rebuild_music"]["description"]
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
    assert [i.program for i in midi.instruments] == [25, 32]
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
    sf.write(vocals, stereo_tone(seconds=5), SR, subtype="FLOAT")
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
    reference = tmp_path / "instrumental-reference.wav"
    sf.write(reference, np.column_stack([source, source]), SR, subtype="FLOAT")
    report = arrangement.rebuild(original, stem, target, tmp_path / "work", seed=29, instrumental=reference)
    mixed, sr = sf.read(target, always_2d=True)
    assert sf.info(target).subtype == "FLOAT" and sr == SR
    assert mixed.shape == voice.shape and np.isfinite(mixed).all()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
    assert report["original_recording_unchanged"] and not report["original_instrumental_reused"]
    assert report["beat_count"] >= 4 and report["supported_accompaniment_share"] > 0.5
    assert report["method"] == "instrumental-led-vocal-constrained"
    assert report["vocal_melody_stable_frames"] > 0 and report["vocal_conflicts_checked"]
    assert report["analysis_instrumental"] == "instrumental-reference.wav"
    assert (target.parent / "arrangement-plan.json").is_file()
    sections = json.loads((target.parent / "arrangement-plan.json").read_text())["segments"]
    for second, expected_root in ((2, 0), (6, 5), (10, 7), (14, 0)):
        section = next(part for part in sections if part["start"] <= second < part["end"])
        assert section["root"] == expected_root, section
    assert report["duration_seconds"] == seconds and report["vocal_offset_samples"] == 0
    assert report["vocal_stem_format"] == "FLOAT"
    assert report["true_peak_dbfs"] <= -0.9999
    assert (target.parent / "arrangement.mid").is_file()
    assert not np.allclose(mixed, voice)


def test_required_vocal_path_requests_float_decode_and_soundfile_export(monkeypatch, tmp_path):
    commands, precision = [], []
    monkeypatch.setattr(separator.subprocess, "run", lambda command, **kwargs: commands.append(command))

    def isolate(wav, out, model, log, high_precision=False):
        precision.append(high_precision)
        return out / "vocals.wav"

    monkeypatch.setattr(separator, "_run", isolate)
    separator.separate(tmp_path / "source.mp3", tmp_path, require_vocals=True)
    assert commands[0][commands[0].index("-c:a") + 1] == "pcm_f32le"
    assert precision == [True]


def test_high_precision_separator_command_avoids_pydub_quantization(monkeypatch, tmp_path):
    commands = []
    (tmp_path / "Vocals.wav").touch()
    monkeypatch.setattr(separator.subprocess, "run",
                        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    separator._run(tmp_path / "source.wav", tmp_path, separator.DEFAULT_MODEL, lambda *a, **k: None,
                   high_precision=True)
    assert "--use_soundfile" in commands[0]
    assert commands[0][commands[0].index("--normalization") + 1] == "1.0"
    assert commands[0][commands[0].index("--amplification") + 1] == "0.0"


def test_low_precision_stem_is_not_silently_presented_as_a_lossless_rebuild(monkeypatch, tmp_path):
    source, stem = tmp_path / "source.wav", tmp_path / "stem.wav"
    sf.write(source, stereo_tone(), SR)
    sf.write(stem, stereo_tone(), SR, subtype="PCM_16")
    monkeypatch.setattr(arrangement, "soundfont_path", lambda: tmp_path / "font.sf2")
    with pytest.raises(arrangement.ArrangementError, match="high-precision"):
        arrangement.rebuild(source, stem, tmp_path / "mix.wav", tmp_path / "work")


def test_required_stem_honors_selected_model_without_quality_fallback(monkeypatch, tmp_path):
    models = []
    monkeypatch.setattr(separator.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))

    def failed(wav, out, model, log, **kwargs):
        models.append(model)
        return None

    monkeypatch.setattr(separator, "_run", failed)
    with pytest.raises(RuntimeError):
        separator.separate(tmp_path / "source.wav", tmp_path, model="selected.ckpt", require_vocals=True)
    assert models == ["selected.ckpt"]


def test_roformer_keeps_quality_overlap_in_rebuild_mode(monkeypatch, tmp_path):
    commands = []
    (tmp_path / "Vocals.wav").touch()
    monkeypatch.setattr(separator.subprocess, "run",
                        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    separator._run(tmp_path / "source.wav", tmp_path, "vocals_mel_band_roformer.ckpt",
                   lambda *a, **k: None, high_precision=True)
    assert "--mdxc_overlap" not in commands[0]


def test_checked_separator_keeps_both_stems_but_unchecked_stays_vocals_only(monkeypatch, tmp_path):
    commands = []
    (tmp_path / "track_(Vocals)_model.wav").touch()
    (tmp_path / "track_(Instrumental)_model.wav").touch()
    monkeypatch.setattr(separator.subprocess, "run",
                        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    for checked in (False, True):
        path = separator._run(tmp_path / "source.wav", tmp_path, "model.onnx", lambda *a, **k: None,
                              high_precision=checked)
        assert path.name == "track_(Vocals)_model.wav"
    assert "--single_stem" in commands[0] and "--single_stem" not in commands[1]


def test_no_vocals_stem_is_not_mistaken_for_singer(monkeypatch, tmp_path):
    (tmp_path / "track_(No_Vocals)_model.wav").touch()
    (tmp_path / "track_(Vocals)_model.wav").touch()
    monkeypatch.setattr(separator.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    result = separator._run(tmp_path / "source.wav", tmp_path, "model.onnx", lambda *a, **k: None,
                            high_precision=True)
    assert result.name == "track_(Vocals)_model.wav"


@pytest.mark.parametrize("split", [False, True])
def test_instrumental_reference_uses_actual_nonvocal_stems_in_float_without_changing_voice(tmp_path, split):
    voice = tmp_path / "track_(Vocals)_model.wav"
    sf.write(voice, stereo_tone(), SR, subtype="FLOAT")
    before = hashlib.sha256(voice.read_bytes()).hexdigest()
    reference = stereo_tone(pitch=330)
    if split:
        for role in ("Drums", "Bass", "Other"):
            sf.write(tmp_path / f"track_({role})_model.wav", reference / 3, SR, subtype="FLOAT")
    else:
        sf.write(tmp_path / "track_(Instrumental)_model.wav", reference, SR, subtype="FLOAT")
    output = separator.instrumental_stem(voice, tmp_path / "reference.wav")
    actual, rate = sf.read(output, always_2d=True)
    np.testing.assert_allclose(actual, reference, atol=2e-8)
    assert rate == SR and sf.info(output).subtype == "FLOAT"
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == before


def test_missing_instrumental_is_not_replaced_with_original_mix(tmp_path):
    voice = tmp_path / "Vocals.wav"
    sf.write(voice, stereo_tone(), SR, subtype="FLOAT")
    with pytest.raises(RuntimeError, match="paired instrumental"):
        separator.instrumental_stem(voice, tmp_path / "reference.wav")


def test_instrumental_with_wrong_duration_is_rejected(tmp_path):
    voice, music = tmp_path / "Vocals.wav", tmp_path / "Instrumental.wav"
    sf.write(voice, stereo_tone(), SR, subtype="FLOAT")
    sf.write(music, stereo_tone(seconds=4), SR, subtype="FLOAT")
    with pytest.raises(RuntimeError, match="timelines"):
        separator.instrumental_stem(voice, tmp_path / "reference.wav")


def test_rebuild_never_analyzes_the_full_mix_for_harmony(monkeypatch, tmp_path):
    source, voice, reference = [tmp_path / name for name in ("source.wav", "voice.wav", "reference.wav")]
    for path in (source, voice, reference):
        sf.write(path, stereo_tone(), SR, subtype="FLOAT")
    monkeypatch.setattr(arrangement, "soundfont_path", lambda: tmp_path / "font.sf2")
    read = []

    def read_audio(path, rate):
        read.append(path)
        assert path != source
        return np.zeros(rate * 6)

    monkeypatch.setattr(arrangement, "read_audio", read_audio)
    monkeypatch.setattr(arrangement, "harmony_plan",
                        lambda *a, **k: (_ for _ in ()).throw(arrangement.ArrangementError("analysis reached")))
    with pytest.raises(arrangement.ArrangementError, match="analysis reached"):
        arrangement.rebuild(source, voice, tmp_path / "mix.wav", tmp_path / "work", instrumental=reference)
    assert read == [reference, voice]
    with pytest.raises(arrangement.ArrangementError, match="separate instrumental"):
        arrangement.rebuild(source, voice, tmp_path / "mix.wav", tmp_path / "work", instrumental=source)


def test_midi_supports_sparse_single_notes_percussion_and_instrument_tuning(tmp_path):
    import pretty_midi

    plan = [{"start": 0, "end": 2, "root": 0, "minor": False, "score": 0.3,
             "reliable": True, "intervals": [0], "beats": [0.2, 1.2]},
            {"start": 2, "end": 4, "root": 0, "minor": False, "score": 0,
             "reliable": False, "intervals": [], "beats": []}]
    target = tmp_path / "sparse.mid"
    arrangement.write_midi(plan, target, tuning_cents=-25)
    midi = pretty_midi.PrettyMIDI(str(target))
    assert [i.program for i in midi.instruments] == [25, 32]
    assert not any(i.is_drum for i in midi.instruments)
    for instrument in midi.instruments:
        assert instrument.pitch_bends[0].pitch == -1024
        assert all(note.pitch % 12 == 0 and note.end <= 2.01 for note in instrument.notes)


def test_synthesis_work_path_cannot_overwrite_an_input_stem(monkeypatch, tmp_path):
    source, voice, reference = [tmp_path / name for name in ("instrumental.wav", "voice.wav", "reference.wav")]
    for path in (source, voice, reference):
        sf.write(path, stereo_tone(), SR, subtype="FLOAT")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(arrangement, "soundfont_path", lambda: tmp_path / "font.sf2")
    with pytest.raises(arrangement.ArrangementError, match="Synthesis output must not overwrite"):
        arrangement.rebuild(source, voice, tmp_path / "master.wav", tmp_path, instrumental=reference)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
