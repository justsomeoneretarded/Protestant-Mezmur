from mezmur.quality import assess


def test_reported_repeated_ethiopic_character_is_rejected():
    words = [{"word": "ስስ", "start": i, "end": i + 0.5} for i in range(80)]
    report = assess(words)
    assert not report["passed"]
    assert "repeated character" in " ".join(report["reasons"])


def test_repeated_chorus_with_real_words_is_allowed():
    chorus = ["አሜን", "ሃሌሉያ", "ጌታ", "ይመስገን", "እግዚአብሔር"]
    words = [{"word": text, "start": i, "end": i + 0.5} for i, text in enumerate(chorus * 12)]
    assert assess(words)["passed"]


def test_no_clear_words_is_a_quality_failure():
    report = assess([])
    assert not report["passed"]
    assert report["first_lyric_seconds"] is None
