"""Reported song-29 evidence may reach forced alignment without weakening wrong-song checks."""
import numpy as np
import pytest

from mezmur import song
from mezmur.lyrics_sources import identity_evidence

RECORD = {"title_latin": "Nefsen Befith", "artist_latin": "Kalkidan Lily Tilahun"}
URL = "https://mezmuroch.com/lyrics/kalkidan-tilahun/nefsen-befiteh"


def candidate(url=URL, trusted=False, first="one"):
    return {"source": "web:mezmuroch.com", "url": url, "title": "Nefsen Befiteh",
            "trusted": trusted, "text": first, "stanzas": [{"lines": [first, "two"]}]}


def choose(monkeypatch, candidates=None, report=None, sim=0.217, record=RECORD):
    report = {"matched_share": 0.73, "gain_per_vocal_frame": 0.11,
              "lines_used": 14, "lines_total": 20, "complete_share": 0.57, **(report or {})}
    monkeypatch.setattr(song, "similarity", lambda *a: {"score": sim})
    monkeypatch.setattr(song, "build_target", lambda *a: None)
    monkeypatch.setattr(song, "align_lyrics", lambda *a, **k: (
        [{"word_index": 0, "line": 0, "phrase": 0, "start": 1, "end": 2}], dict(report)))
    return song.choose_lyrics(candidates or [candidate()], "draft", np.zeros((10, 2)), 0.02,
                              {"char_to_id": {}, "blank": 0, "delimiter": 1}, np.ones(10), record=record)


def test_song_29_reported_011_metrics_are_accepted_without_changing_strict_gain(monkeypatch):
    best, reports = choose(monkeypatch)
    assert song.ACCEPT_GAIN == 0.18
    assert best is not None and reports[0]["accepted"]
    assert best[3]["verification"]["method"] == "source-supported"
    assert reports[0]["gain_per_vocal_frame"] == 0.11


@pytest.mark.parametrize("patch", [
    {"matched_share": 0.51}, {"lines_used": 13}, {"gain_per_vocal_frame": 0},
    {"gain_per_vocal_frame": -0.1}, {"complete_share": 0.49},
    {"gain_per_vocal_frame": float("nan")}, {"gain_per_vocal_frame": float("inf")},
])
def test_low_score_path_still_requires_real_audio_and_line_evidence(monkeypatch, patch):
    best, reports = choose(monkeypatch, report=patch)
    assert best is None and not reports[0]["accepted"]


@pytest.mark.parametrize("sim", [0, 0.06, 0.14])
def test_title_alone_cannot_override_a_mismatching_draft(monkeypatch, sim):
    assert choose(monkeypatch, sim=sim)[0] is None


@pytest.mark.parametrize("url", [
    "https://mezmuroch.com/lyrics/other-tilahun/nefsen-befiteh",
    "https://mezmuroch.com/lyrics/kalkidan-tilahun/different-song",
    "https://mezmuroch.com.evil.invalid/lyrics/kalkidan-tilahun/nefsen-befiteh",
    "http://mezmuroch.com/lyrics/kalkidan-tilahun/nefsen-befiteh",
])
def test_wrong_identity_or_spoofed_source_does_not_gain_automatic_trust(monkeypatch, url):
    best, reports = choose(monkeypatch, [candidate(url)])
    assert best is None
    assert not reports[0]["verification"]["identity"]["matched"]


def test_shorter_artist_slug_matches_first_and_last_names_not_just_shared_surname():
    evidence = identity_evidence(candidate(), RECORD)
    assert evidence["matched"] and evidence["short_artist_name"]
    assert not identity_evidence(candidate(), RECORD | {"artist_latin": "Other Lily Tilahun"})["matched"]


def test_missing_song_metadata_does_not_enable_the_support_path(monkeypatch):
    assert choose(monkeypatch, record=None)[0] is None


def test_high_acoustic_evidence_keeps_existing_acceptance_without_site_identity(monkeypatch):
    best, reports = choose(monkeypatch, [candidate("https://unknown.invalid/song")],
                           report={"gain_per_vocal_frame": 0.23})
    assert best is not None and reports[0]["verification"]["method"] == "acoustic"


def test_manual_corrections_keep_their_explicit_override(monkeypatch):
    best, _ = choose(monkeypatch, [candidate(trusted=True)],
                     report={"gain_per_vocal_frame": 0, "matched_share": 0.1, "lines_used": 1}, sim=0)
    assert best is not None and best[3]["verification"]["method"] == "correction"


def test_saved_library_identity_is_supporting_evidence_not_a_trusted_correction():
    c = candidate()
    c.update(source="library", url="lyrics library: artist/song.txt", identity_match=True)
    evidence = identity_evidence(c, RECORD)
    assert evidence["matched"] and not c["trusted"]


def test_strict_acoustic_candidate_is_preferred_over_source_supported_candidate(monkeypatch):
    low, high = candidate(first="low"), candidate("https://unknown.invalid/lyrics", first="high")
    monkeypatch.setattr(song, "similarity", lambda *a: {"score": 0.3})
    monkeypatch.setattr(song, "build_target", lambda lyrics, *a: lyrics.lines[0][0])

    def align(logp, frame_s, target, *args, **kwargs):
        strong = target == "high"
        return [{"word_index": 0, "line": 0, "phrase": 0, "start": 1, "end": 2}], {
            "matched_share": 0.70 if strong else 0.99, "gain_per_vocal_frame": 0.18 if strong else 0.11,
            "lines_used": 14, "lines_total": 20, "complete_share": 0.5 if strong else 0.99}

    monkeypatch.setattr(song, "align_lyrics", align)
    best, _ = song.choose_lyrics([low, high], "draft", np.zeros((10, 2)), 0.02,
                                {"char_to_id": {}, "blank": 0, "delimiter": 1}, np.ones(10), record=RECORD)
    assert best[1] is high
