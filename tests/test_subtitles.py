from mezmur.subtitles import build_cues, write_ass, write_srt


def words():
    out, t = [], 1.0
    for li, line in enumerate([["ሲወድ", "ጌታ", "እስከ", "መስቀል", "ነው"], ["ሃሌሉያ", "ክብር", "ለስሙ"]] * 2):
        for wi, w in enumerate(line):
            out.append({"word": w, "start": t, "end": t + 0.4, "line": li % 2, "word_index": (li % 2) * 10 + wi})
            t += 0.5
        t += 1.0
    return out


def test_one_cue_per_sung_line_with_romanization(tmp_path):
    cues = build_cues(words(), chorus_lines={1})
    assert [c["text"] for c in cues] == ["ሲወድ ጌታ እስከ መስቀል ነው", "ሃሌሉያ ክብር ለስሙ"] * 2
    assert cues[0]["romanized"] == "siwed geta eske meskel new"
    write_ass(cues, tmp_path / "x.ass")
    ass = (tmp_path / "x.ass").read_text(encoding="utf-8")
    assert "\\kf" in ass and ",Roman," in ass and ",Next," in ass and "CHORUS" in ass
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
