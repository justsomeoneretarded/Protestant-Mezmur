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
