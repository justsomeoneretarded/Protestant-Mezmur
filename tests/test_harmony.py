"""Hybrid harmony evidence and conservative accompaniment, independent of lyric recognition."""
import numpy as np
import pytest

from mezmur.harmony import CHORDS, compatible_notes, select_voicing, vocal_melody


def histogram(*pitches):
    values = np.zeros(12)
    for pitch in pitches:
        values[pitch] += 1 / len(pitches)
    return values


def test_sustained_vocal_notes_exclude_conflicting_accompaniment_pitches():
    safe = compatible_notes(range(12), histogram(11))
    assert not ({0, 5, 10} & set(safe))
    assert {4, 7, 11} <= set(safe)


def test_instrumental_chord_is_retained_when_it_supports_the_singer():
    reference = histogram(0, 4, 7)
    root, intervals, mode = select_voicing(0, "major", 0.95, reference, histogram(0), histogram(4))
    assert root == 0 and intervals == list(CHORDS["major"]) and mode == "chord"


def test_wrong_third_or_root_is_not_forced_just_because_instrumental_score_is_high():
    root, intervals, mode = select_voicing(0, "major", 0.95,
        histogram(0, 4, 7), histogram(0), histogram(11))
    notes = {(root + interval) % 12 for interval in intervals}
    assert mode != "chord" and notes
    assert not ({0, 5, 10} & notes)


@pytest.mark.parametrize("fifth,expected", [(True, "open"), (False, "single")])
def test_weak_harmony_uses_only_evidenced_open_or_single_note_voicings(fifth, expected):
    reference = histogram(9, 4) if fifth else histogram(9)
    root, intervals, mode = select_voicing(9, "minor", 0.4, reference, histogram(9), histogram(9))
    assert root == 9 and mode == expected
    assert intervals == ([0, 7] if fifth else [0])


def test_no_pitch_evidence_produces_a_rest_not_an_arbitrary_chord():
    _, intervals, mode = select_voicing(0, "major", 0.1, np.zeros(12), np.zeros(12), np.zeros(12))
    assert mode == "rest" and intervals == []


def test_suspended_harmony_does_not_require_inventing_a_major_or_minor_third():
    root, intervals, mode = select_voicing(0, "sus2", 0.9,
        histogram(0, 2, 7), histogram(0), histogram(2))
    assert root == 0 and intervals == [0, 2, 7] and mode == "chord"


def test_low_weight_passing_pitch_does_not_override_a_supported_chord():
    sung = np.zeros(12)
    sung[4], sung[5] = 0.9, 0.1
    assert compatible_notes([0, 4, 7], sung) == [0, 4, 7]


def test_native_melody_extraction_finds_sung_pitch_and_does_not_voice_silence():
    sr = 22050
    time = np.arange(sr * 4) / sr
    values = np.r_[0.15 * np.sin(2 * np.pi * 440 * time), np.zeros(sr * 4)].astype(np.float32)
    before = values.copy()
    melody = vocal_melody(values, sr)
    reliable = melody["weights"] > 0
    assert reliable.any() and melody["stable_frames"] > 30
    assert np.mean(melody["pitch_classes"][reliable] == 9) > 0.9
    assert not np.any(reliable[melody["times"] > 4.2])
    np.testing.assert_array_equal(values, before)


def test_single_pitch_does_not_masquerade_as_a_fully_observed_power_chord():
    root, intervals, mode = select_voicing(0, "fifth", 0.9, histogram(0), histogram(0), histogram(0))
    assert root == 0 and intervals == [0] and mode == "single"
