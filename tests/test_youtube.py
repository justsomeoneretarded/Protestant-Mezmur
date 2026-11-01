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



def test_description_fits_api_byte_limit_and_keeps_credits_and_hashtags():
    from mezmur.youtube import description

    stanzas = [{"lines": ["\u12a0\u121d\u120b\u12ad " * 150 for _ in range(12)]}]
    text = description(M, stanzas, [], [], ["Photo: Example credit"], "library")
    assert len(text.encode("utf-8")) <= 4900
    assert "\ufffd" not in text
    assert "Photo: Example credit" in text
    assert "#AmharicMezmur" in text and "#ProtestantMezmur" in text



def test_very_long_generated_names_still_produce_nonempty_valid_titles():
    long = {key: "worship " * 100 for key in M}
    assert all(0 < len(title) <= 100 for title in titles(long))


def test_title_limiter_shortens_and_cleans_invalid_characters():
    from mezmur.youtube import limit_title

    result = limit_title("  <Worship>\n" + "A" * 200)
    assert result.startswith("Worship ") and len(result) == 100
    assert "<" not in result and ">" not in result and "\n" not in result


def test_description_limiter_does_not_split_multibyte_characters():
    from mezmur.youtube import limit_description

    result = limit_description("\u12a0" * 2000)
    assert len(result.encode("utf-8")) <= 5000 and len(result) == 1666
    assert "\ufffd" not in result


def test_tag_limiter_counts_implicit_quotes_commas_and_duplicates():
    from mezmur.youtube import limit_tags

    result = limit_tags(["worship song", "WORSHIP SONG", "a" * 600, "gospel"] * 100)
    cost = sum(len(t) + (2 if " " in t else 0) for t in result) + len(result) - 1
    assert result == ["worship song", "gospel"] and cost <= 500
