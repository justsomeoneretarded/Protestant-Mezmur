import re

import pytest

from mezmur.subtitles import (LYRIC_MAX_SIZE, MAX_WIDTH, _measure, build_cues,
                              choose_theme, fit_size, write_ass, write_srt)


def words():
    out, t = [], 1.0
    for li, line in enumerate([["ሲወድ", "ጌታ", "እስከ", "መስቀል", "ነው"], ["ሃሌሉያ", "ክብር", "ለስሙ"]] * 2):
        for wi, w in enumerate(line):
            out.append({"word": w, "start": t, "end": t + 0.4, "line": li % 2, "word_index": (li % 2) * 10 + wi})
            t += 0.5
        t += 1.0
    return out


def test_cues_keep_romanization_but_default_burned_lyrics_are_amharic_only(tmp_path):
    cues = build_cues(words(), chorus_lines={1})
    assert [c["text"] for c in cues] == ["ሲወድ ጌታ እስከ መስቀል ነው", "ሃሌሉያ ክብር ለስሙ"] * 2
    assert cues[0]["romanized"] == "siwed geta eske meskel new"
    write_ass(cues, tmp_path / "x.ass")
    ass = (tmp_path / "x.ass").read_text(encoding="utf-8")
    assert "\\kf" in ass and ",Roman," not in ass and ",Next," in ass and "CHORUS" in ass
    assert cues[0]["romanized"] not in ass
    write_srt(cues, tmp_path / "x.srt")
    assert (tmp_path / "x.srt").read_text(encoding="utf-8").startswith("1\n00:00:01,000 --> ")


def test_unlabelled_words_split_at_pauses():
    w = [{"word": f"ቃል{i}", "start": i * 0.5 + (3 if i >= 4 else 0), "end": i * 0.5 + 0.4 + (3 if i >= 4 else 0)}
         for i in range(8)]
    assert [len(c["words"]) for c in build_cues(w)] == [4, 4]


def test_upload_srt_matches_burned_main_lyric_visibility_and_source_end(tmp_path):
    import srt

    cues = build_cues(words())
    duration = cues[-1]["end"] + 0.127
    write_ass(cues, tmp_path / "x.ass", duration=duration)
    write_srt(cues, tmp_path / "subtitle.srt", display_timing=True, duration=duration)
    tracks = list(srt.parse((tmp_path / "subtitle.srt").read_text(encoding="utf-8")))
    events = [line.split(",", 9) for line in (tmp_path / "x.ass").read_text(encoding="utf-8").splitlines()
              if line.startswith("Dialogue:") and ",Lyric," in line]

    def seconds(value):
        h, m, sec = value.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)

    assert [c.content for c in tracks] == [c["text"] for c in cues]
    assert len(events) == len(tracks)
    for caption, event in zip(tracks, events):
        assert abs(caption.start.total_seconds() - seconds(event[1])) < 0.0001
        assert abs(caption.end.total_seconds() - seconds(event[2])) < 0.0001
        assert 0 <= caption.start.total_seconds() < caption.end.total_seconds() <= duration
    assert tracks[0].start.total_seconds() == 0.55
    assert "{" not in (tmp_path / "subtitle.srt").read_text()


def test_display_subtitles_skip_cues_outside_video_without_extending_it(tmp_path):
    import srt

    cues = build_cues(words())
    write_srt(cues, tmp_path / "subtitle.srt", display_timing=True, duration=1.913)
    captions = list(srt.parse((tmp_path / "subtitle.srt").read_text(encoding="utf-8")))
    assert len(captions) == 1
    assert captions[-1].end.total_seconds() <= 1.913


def test_upload_subtitle_writer_is_amharic_script_only_not_romanized(tmp_path):
    import pytest

    cues = build_cues(words())
    target = tmp_path / "subtitle.srt"
    write_srt(cues, target, display_timing=True, amharic_only=True)
    text = target.read_text(encoding="utf-8")
    assert "ሲወድ ጌታ" in text and "siwed" not in text and "haleluya" not in text
    with pytest.raises(ValueError, match="Amharic script only"):
        write_srt(cues, tmp_path / "roman.srt", romanized=True, amharic_only=True)
    assert not (tmp_path / "roman.srt").exists()


def test_romanized_ass_is_still_an_explicit_development_opt_in(tmp_path):
    cues = build_cues(words())
    path = tmp_path / "optional-romanized.ass"
    write_ass(cues, path, romanized=True)
    assert ",Roman," in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("seed,bright", [(0, False), (2, False), (0, True), (3, True)])
def test_larger_main_lyrics_fit_their_band_and_leave_room_for_preview(tmp_path, seed, bright):
    cues = build_cues(words())
    theme = choose_theme(seed, bright_background=bright)
    target = tmp_path / "larger.ass"
    write_ass(cues, target, theme=theme)
    text = target.read_text(encoding="utf-8")
    assert LYRIC_MAX_SIZE == 112
    events = [line.split(",", 9) for line in text.splitlines() if line.startswith("Dialogue:")]
    for event in events:
        if event[3] == "Lyric":
            size = int(re.search(r"\\fs(\d+)", event[9])[1])
            assert size > 92 and _measure(theme["file"], size, cues[0]["text"]) <= MAX_WIDTH
        elif event[3] == "Next":
            assert r"\pos(960,700)" in event[9]
    assert all(event[3] != "Roman" for event in events)


@pytest.mark.parametrize("text", ["ሃሌሉያ ክብር ለስሙ " * 20, "አ" * 120])
def test_long_lyric_text_never_overflows_the_available_width(text):
    size = fit_size(text)
    assert 1 <= size < LYRIC_MAX_SIZE
    assert _measure("NotoSansEthiopic-Bold.ttf", size, text) <= MAX_WIDTH


def test_amharic_subtitle_drops_latin_words_instead_of_failing(tmp_path):
    from mezmur.lyrics import is_amharic_text
    from mezmur.subtitles import write_srt
    cues = [{"start": 1.0, "end": 2.0, "text": "ኧረ ስንቱ x2", "romanized": ""},
            {"start": 3.0, "end": 4.0, "text": "Chorus", "romanized": ""},
            {"start": 5.0, "end": 6.0, "text": "ሁሉ ለፍኩብህ (Repeat)", "romanized": ""}]
    write_srt(cues, tmp_path / "s.srt", amharic_only=True)
    blocks = (tmp_path / "s.srt").read_text(encoding="utf-8").strip().split("\n\n")
    texts = [b.splitlines()[2] for b in blocks]
    assert texts == ["ኧረ ስንቱ", "ሁሉ ለፍኩብህ"]
    assert all(is_amharic_text(t) for t in texts)
