"""Acoustic performance, musical structure and strict source-clock boundary regressions."""
import copy
import json

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from mezmur import acoustic, arrangement
from mezmur.harmony import ArrangementError


def section(start=0, end=2, root=0, energy=0.5, chorus=False):
    return {"start": start, "end": end, "root": root, "minor": False, "score": 0.9,
            "reliable": True, "reference_confident": True, "intervals": [0, 4, 7],
            "beats": [start, start + (end - start) / 2], "energy": energy,
            "percussive_strength": 0.0, "vocal_activity": 1.0}


def test_default_accompaniment_is_guitar_and_bass_without_always_on_full_band(tmp_path):
    path = tmp_path / "acoustic.mid"
    report = acoustic.write_midi([section()], path, seed=40)
    midi = pretty_midi.PrettyMIDI(str(path))
    assert [track.program for track in midi.instruments] == [25, 32]
    assert not any(track.is_drum for track in midi.instruments)
    assert report["arrangement_style"] == "adaptive-guitar-and-bass"


def test_dynamics_and_lyric_structure_change_playing_style_without_random_chord_changes(tmp_path):
    plan = [section(i * 2, (i + 1) * 2, energy=0.25 if i < 4 else 0.95) for i in range(8)]
    cues = [{"start": 0, "end": 8, "chorus": False}, {"start": 8, "end": 16, "chorus": True}]
    first = acoustic.write_midi(plan, tmp_path / "first.mid", cues=cues, seed=1)
    second = acoustic.write_midi(plan, tmp_path / "second.mid", cues=cues, seed=99)
    assert first["segments"][1]["playing_style"] == "fingerpick"
    assert first["segments"][6]["playing_style"] == "strum"
    assert first["segments"][6]["section"] == "chorus"
    assert [(s["root"], s["intervals"], s["playing_style"]) for s in first["segments"]] == [
        (s["root"], s["intervals"], s["playing_style"]) for s in second["segments"]]
    assert {s["root"] for s in first["segments"]} == {0}


@pytest.mark.parametrize("root,intervals", [(0, [0, 4, 7]), (9, [0, 3, 7]), (7, [0, 2, 7]),
                                         (11, [0, 4, 7, 11]), (6, [0]), (2, [0, 7])])
def test_guitar_voicings_obey_string_range_fret_span_and_allowed_pitch_classes(root, intervals):
    voice = acoustic.guitar_voicing(root, intervals)
    assert voice
    allowed = {(root + interval) % 12 for interval in intervals}
    pressed = []
    for string, fret, pitch in voice:
        assert 0 <= string < 6 and 0 <= fret <= 12
        assert pitch == acoustic.STRINGS[string] + fret and pitch % 12 in allowed
        if fret:
            pressed.append(fret)
    assert len({string for string, _, _ in voice}) == len(voice)
    assert not pressed or max(pressed) - min(pressed) <= 4


def test_changes_follow_supplied_chords_and_repeated_sections_reuse_the_same_motif(tmp_path):
    plan = [section(i * 2, (i + 1) * 2, root=root) for i, root in enumerate((0, 5, 7, 0, 5, 7))]
    path = tmp_path / "progression.mid"
    acoustic.write_midi(plan, path)
    guitar = pretty_midi.PrettyMIDI(str(path)).instruments[0]
    for part in plan:
        allowed = {(part["root"] + interval) % 12 for interval in part["intervals"]}
        assert all(note.pitch % 12 in allowed for note in guitar.notes
                   if part["start"] <= note.start < part["end"] - 0.001)
    for a, b in ((0, 3), (1, 4), (2, 5)):
        left = [round(note.start - plan[a]["start"], 3) for note in guitar.notes
                if plan[a]["start"] <= note.start < plan[a]["end"]]
        right = [round(note.start - plan[b]["start"], 3) for note in guitar.notes
                 if plan[b]["start"] <= note.start < plan[b]["end"]]
        assert left == right


def test_optional_colors_require_supported_context_not_merely_beat_timestamps(tmp_path):
    quiet = [section(i * 2, (i + 1) * 2, energy=0.3) for i in range(6)]
    report = acoustic.write_midi(quiet, tmp_path / "quiet.mid")
    assert len(report["instruments"]) == 2
    strong = [section(i * 2, (i + 1) * 2, energy=0.95) for i in range(6)]
    for part in strong:
        part["percussive_strength"] = 0.7
    acoustic.write_midi(strong, tmp_path / "strong.mid",
                       cues=[{"start": 0, "end": 12, "chorus": True}])
    tracks = pretty_midi.PrettyMIDI(str(tmp_path / "strong.mid")).instruments
    assert any(track.is_drum for track in tracks) and any(track.program == 89 for track in tracks)
    assert not any(track.program == 0 and not track.is_drum for track in tracks)  # no piano over the singer


def test_piano_accents_only_fill_confirmed_instrumental_space(tmp_path):
    plan = [section(i * 2, (i + 1) * 2, energy=0.9) for i in range(4)]
    for part in plan:
        part["vocal_activity"] = 0
    acoustic.write_midi(plan, tmp_path / "break.mid")
    tracks = pretty_midi.PrettyMIDI(str(tmp_path / "break.mid")).instruments
    assert any(track.program == 0 and not track.is_drum for track in tracks)


@pytest.mark.parametrize("patch", [
    {"start": -1}, {"end": float("nan")}, {"end": float("inf")},
    {"end": 0}, {"root": 0.5}, {"root": float("nan")}, {"root": 12},
    {"beats": [float("nan")]}, {"energy": float("inf")}, {"intervals": [99]},
])
def test_invalid_plan_fails_before_writing_midi(tmp_path, patch):
    path = tmp_path / "bad.mid"
    with pytest.raises(ArrangementError):
        acoustic.write_midi([{**section(), **patch}], path, duration=2)
    assert not path.exists()


def test_unsorted_overlapping_and_out_of_range_sections_are_rejected():
    for plan in ([section(2, 4), section(0, 2)], [section(0, 3), section(2, 4)], [section(0, 5)]):
        with pytest.raises(ArrangementError):
            acoustic.prepare_plan(plan, duration=4)


def test_empty_or_all_rest_plan_is_not_falsely_reported_as_an_arrangement(tmp_path):
    for plan in ([], [{**section(), "reliable": False}]):
        with pytest.raises(ArrangementError):
            acoustic.write_midi(plan, tmp_path / "none.mid")


def test_irregular_beats_missing_beats_and_short_last_section_never_extend_timeline(tmp_path):
    plan = [section(0, 1.7), section(1.7, 3.333), section(3.333, 3.38)]
    plan[0]["beats"] = [0.02, 0.38, 1.1, 1.1, 1.69]
    plan[1]["beats"] = []
    original = copy.deepcopy(plan)
    target = tmp_path / "timing.mid"
    acoustic.write_midi(plan, target, duration=3.36, tuning_cents=-17)
    midi = pretty_midi.PrettyMIDI(str(target))
    assert midi.get_end_time() <= 3.36
    assert plan == original
    for track in midi.instruments:
        assert all(0 <= note.start < note.end <= 3.36 for note in track.notes)
        for pitch in {note.pitch for note in track.notes}:
            notes = sorted((n for n in track.notes if n.pitch == pitch), key=lambda n: n.start)
            assert all(a.end <= b.start + 0.001 for a, b in zip(notes, notes[1:]))


@pytest.mark.parametrize("tail_delta", [-24000, 96000])
def test_synthesis_tail_is_trimmed_or_padded_without_moving_vocals(monkeypatch, tmp_path, tail_delta):
    sr, seconds = 24000, 5
    source, voice, reference = [tmp_path / n for n in ("source.wav", "voice.wav", "reference.wav")]
    values = (0.12 * np.sin(2 * np.pi * 220 * np.arange(sr * seconds) / sr)).astype(np.float32)
    stereo = np.column_stack([values, values])
    for path in (source, voice, reference):
        sf.write(path, stereo, sr, subtype="FLOAT")
    monkeypatch.setattr(arrangement, "soundfont_path", lambda: tmp_path / "font.sf2")
    monkeypatch.setattr(arrangement, "harmony_plan", lambda *a, **k: (
        [section(0, seconds)], {"tuning_cents": 0}))
    def synth(midi, target, font, rate):
        n = sr * seconds + tail_delta
        music = 0.05 * np.sin(2 * np.pi * 330 * np.arange(n) / sr)
        sf.write(target, np.column_stack([music, music]), rate, subtype="FLOAT")
    monkeypatch.setattr(arrangement, "render_instrumental", synth)
    target = tmp_path / "out" / "master.wav"
    report = arrangement.rebuild(source, voice, target, tmp_path / "work", instrumental=reference)
    assert sf.info(target).frames == sr * seconds and report["duration_locked"]
    assert report["trimmed_backing_samples"] == max(0, tail_delta)
    assert report["padded_backing_samples"] == max(0, -tail_delta)
    assert report["vocal_offset_samples"] == report["vocal_pitch_shift_semitones"] == 0
    assert report["vocal_tempo_factor"] == 1 and report["true_peak_dbfs"] <= -0.999
