from mezmur.youtube import sections, tags, titles

M = {"title_am": "ሰባሪው ጌታ ከፊቴ ወቷል", "title_latin": "Sebariw Geta Kefite Wetoal",
     "artist_am": "አዲስዓለም አሰፋ", "artist_latin": "Addisalem Assefa"}


def test_titles_fit_youtube_limit_and_carry_both_scripts():
    ts = titles(M)
    assert ts and all(len(t) <= 100 for t in ts)
    assert "ሰባሪው ጌታ" in ts[0] and "Addisalem Assefa" in ts[0]


def test_tags_under_500_characters():
    tg = tags(M)
    assert sum(len(t) + (2 if " " in t else 0) for t in tg) + len(tg) - 1 <= 500
    assert "Addisalem Assefa" in tg and "ሰባሪው ጌታ ከፊቴ ወቷል" in tg


def test_chapters_follow_youtube_rules():
    cues = []
    for block, start in [(0, 15), (1, 45), (0, 75), (1, 120), (0, 150)]:
        for k in range(4):
            cues.append({"start": start + 5 * k, "end": start + 5 * k + 4, "line": block * 10 + k})
    ch = sections(cues, 200, chorus_lines={10, 11, 12, 13})
    assert ch[0]["start"] == 0 and len(ch) >= 3
    assert all(b["start"] - a["start"] >= 10 for a, b in zip(ch, ch[1:]))
    assert any("Chorus" in c["label"] for c in ch)
