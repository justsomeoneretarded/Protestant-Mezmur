"""Reported songs pass the requested coverage/gain policy without trusting a source blindly."""
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


def test_song_29_reported_011_metrics_are_accepted(monkeypatch):
    best, reports = choose(monkeypatch)
    assert song.ACCEPT_GAIN == 0.07 and song.ACCEPT_SHARE == song.ACCEPT_LINES == 0.40
    assert best is not None and reports[0]["accepted"]
    assert best[3]["verification"]["method"] == "acoustic"
    assert reports[0]["gain_per_vocal_frame"] == 0.11


@pytest.mark.parametrize("patch", [
    {"matched_share": 0.38}, {"lines_used": 7}, {"gain_per_vocal_frame": 0.06},
    {"gain_per_vocal_frame": 0},
    {"gain_per_vocal_frame": -0.1}, {"complete_share": float("nan")},
    {"gain_per_vocal_frame": float("nan")}, {"gain_per_vocal_frame": float("inf")},
])
def test_low_score_path_still_requires_real_audio_and_line_evidence(monkeypatch, patch):
    best, reports = choose(monkeypatch, report=patch)
    assert best is None and not reports[0]["accepted"]


@pytest.mark.parametrize("sim", [0, 0.01, 0.059])
def test_title_alone_cannot_override_a_mismatching_draft(monkeypatch, sim):
    assert choose(monkeypatch, sim=sim)[0] is None


@pytest.mark.parametrize("url", [
    "https://mezmuroch.com/lyrics/other-tilahun/nefsen-befiteh",
    "https://mezmuroch.com/lyrics/kalkidan-tilahun/different-song",
    "https://mezmuroch.com.evil.invalid/lyrics/kalkidan-tilahun/nefsen-befiteh",
    "http://mezmuroch.com/lyrics/kalkidan-tilahun/nefsen-befiteh",
])
def test_wrong_identity_or_spoofed_source_does_not_gain_automatic_trust(monkeypatch, url):
    best, reports = choose(monkeypatch, [candidate(url)], report={"gain_per_vocal_frame": 0.06})
    assert best is None
    assert not reports[0]["verification"]["identity"]["matched"]


def test_shorter_artist_slug_matches_first_and_last_names_not_just_shared_surname():
    evidence = identity_evidence(candidate(), RECORD)
    assert evidence["matched"] and evidence["short_artist_name"]
    assert not identity_evidence(candidate(), RECORD | {"artist_latin": "Other Lily Tilahun"})["matched"]


def test_source_identity_is_not_required_when_acoustic_policy_passes(monkeypatch):
    assert choose(monkeypatch, record=None)[0] is not None
    assert choose(monkeypatch, report={"gain_per_vocal_frame": 0.06}, record=None)[0] is None


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


def test_more_complete_acoustic_candidate_can_win_without_a_special_source_bonus(monkeypatch):
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
    assert best[1] is low


@pytest.mark.parametrize("share,gain,used,total,complete,sim", [
    (0.48, 0.07, 8, 8, 0.87, 0.19),       # song 35, Telegram caption
    (0.40, 0.09, 32, 32, 0.45, 0.317),    # song 37, complete Telegram candidate
])
def test_reported_telegram_songs_are_accepted(monkeypatch, share, gain, used, total, complete, sim):
    c = candidate("https://t.me/bestprotestant_songs/1242")
    c["source"] = "telegram:channel @bestprotestant_songs"
    best, reports = choose(monkeypatch, [c], report={
        "matched_share": share, "gain_per_vocal_frame": gain,
        "lines_used": used, "lines_total": total, "complete_share": complete}, sim=sim)
    assert best is not None and reports[0]["accepted"]
    assert best[3]["verification"]["method"] == "acoustic"


@pytest.mark.parametrize("share,gain,accepted", [
    (0.40, 0.07, True), (0.399, 0.069, True),
    (0.394, 0.07, False), (0.40, 0.064, False),
])
def test_admission_uses_display_precision_and_keeps_raw_evidence(monkeypatch, share, gain, accepted):
    best, reports = choose(monkeypatch, report={"matched_share": share, "gain_per_vocal_frame": gain})
    assert (best is not None) is accepted
    evidence = reports[0]["verification"]
    assert evidence["score_decimals"] == 2
    assert evidence["decision_scores"]["matched_share"] == round(share, 2)
    assert evidence["decision_scores"]["gain_per_vocal_frame"] == round(gain, 2)
    assert evidence["thresholds"]["gain_per_vocal_frame"] == 0.07
    assert reports[0]["gain_per_vocal_frame"] == gain


def test_40_percent_line_coverage_is_enough_and_low_complete_phrase_share_is_not_an_extra_veto(monkeypatch):
    best, _ = choose(monkeypatch, report={"lines_used": 8, "lines_total": 20, "complete_share": 0.1}, sim=0.06)
    assert best is not None
