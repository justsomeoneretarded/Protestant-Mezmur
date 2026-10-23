from mezmur.lyrics import make_cues, safe_ass, timestamp, write_lyrics
from mezmur.transcribe import chunks, merge_words


def test_chunk_overlap_keeps_one_owner_and_absolute_times():
    results = [
        {"offset": 0, "core_start": 0, "core_end": 240,
         "response": {"words": [{"word": "first", "start": 238, "end": 239},
                                  {"word": "next", "start": 241, "end": 242}]}},
        {"offset": 236, "core_start": 240, "core_end": 300,
         "response": {"words": [{"word": "first", "start": 2, "end": 3},
                                  {"word": "next", "start": 5, "end": 6}]}},
    ]
    words = merge_words(results, 300)
    assert [w["word"] for w in words] == ["first", "next"]
    assert words[1]["start"] == 241
    assert list(chunks(300)) == [(0, 240, 0, 244), (240, 300, 236, 300)]


def test_lyrics_do_not_cover_instrumental_gap(tmp_path):
    words = [{"word": "ሃሌሉያ", "start": 1.5, "end": 3},
             {"word": "አሜን", "start": 13, "end": 14}]
    cues = make_cues(words)
    assert len(cues) == 2
    assert cues[0].end == 3
    write_lyrics(tmp_path, cues, "መዝሙር", 15)
    assert "00:00:01,500 --> 00:00:03,000" in (tmp_path / "lyrics.srt").read_text()
    assert "\\kf150" in (tmp_path / "lyrics.ass").read_text()


def test_subtitle_injection_and_timestamp_rounding():
    assert safe_ass("{\\pos(0,0)}\nTitle") == "pos(0,0) Title"
    assert timestamp(59.9999) == "00:01:00,000"
    assert timestamp(59.9999, True) == "0:01:00.00"


def test_no_timed_words_is_error():
    import pytest
    with pytest.raises(ValueError, match="No timed lyrics"):
        merge_words([], 10)
