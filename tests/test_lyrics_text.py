from mezmur.lyrics_text import extract_blocks, html_to_text, parse_lyrics, similarity

TEXT = """ሲወድ ፡ ጌታ ፡ እስከ ፡ መስቀል ፡ ነው (x2)
ፍቅሩ ፡ አይለካም ፡ ከቶ

አዝ፦ ሃሌሉያ ፡ ክብር ፡ ለስሙ
ሃሌሉያ ፡ ክብር ፡ ለስሙ
ዘመኔን ፡ ሁሉ ፡ አመሰግናለሁ
Join us @mezmuroch1 👉
"""


def test_parse_markers_repeats_and_junk():
    stanzas = parse_lyrics(TEXT)
    assert stanzas[0] == {"lines": ["ሲወድ ጌታ እስከ መስቀል ነው", "ፍቅሩ አይለካም ከቶ"], "repeats": [2, 1], "chorus": False}
    assert stanzas[1]["chorus"] is True
    assert stanzas[1]["lines"] == ["ሃሌሉያ ክብር ለስሙ", "ዘመኔን ሁሉ አመሰግናለሁ"]
    assert stanzas[1]["repeats"] == [2, 1]  # written twice in a row = sung twice


def test_slash_repeat_marker():
    from mezmur.lyrics_text import line_repeats, stanzas_text
    stanzas = parse_lyrics("ዛሬም ይሰራል/3\nዛሬም እየሱስ ይሰራል\n")
    assert stanzas[0]["lines"] == ["ዛሬም ይሰራል", "ዛሬም እየሱስ ይሰራል"]
    assert line_repeats(stanzas) == [3, 1]
    assert stanzas_text(stanzas).startswith("ዛሬም ይሰራል ×3\n")


def test_extract_from_html_page():
    html = ("<html><head><title>x</title><script>var a=1</script></head><body><nav>Home</nav>"
            "<h1>ሲወድ ጌታ</h1><p>" + TEXT.replace("\n", "<br>") + "</p><footer>© site</footer></body></html>")
    blocks = extract_blocks(html_to_text(html))
    assert blocks and blocks[0][0]["lines"][0] == "ሲወድ ጌታ እስከ መስቀል ነው"


def test_similarity_prefers_the_right_song():
    draft = "ሲወድ ጌታ እስከ መስቀል ነው ፍቅሩ አይለካም ከቶ ሃሌሉያ ክብር ለስሙ"
    right = similarity("ሲወድ ጌታ እስከ መስቀል ነው\nፍቅሩ አይለካም ከቶ\nሃሌሉያ ክብር ለስሙ", draft)["score"]
    wrong = similarity("ተመስገን ጌታ ስላደረግህልኝ\nበረከትህ ብዙ ነው\nልቤ ደስ ይለዋል", draft)["score"]
    assert right > 0.7 > 0.3 > wrong


def test_header_and_brackets_are_cleaned():
    from mezmur.lyrics_text import balance_brackets, parse_lyrics, strip_header
    record = {"title_am": "ክቡር አደራ", "artist_am": "ይስሐቅ ሰድቅ", "artist_latin": "Yishak Sedik"}
    stanzas = parse_lyrics("ክቡር አደራ --- ይስሃቅ ሰዲቅ\nከጨለምተኝነትና ከጠመሙት ጋራ\nአትዛመዱ ከቶ አትላመዱ\nስትሰሩ ስትከናወኑ\n")
    cleaned = strip_header(stanzas, record)
    assert cleaned[0]["lines"][0] == "ከጨለምተኝነትና ከጠመሙት ጋራ"
    assert balance_brackets("ታብሷል እምባዬ (ታብሷል") == "ታብሷል እምባዬ ታብሷል"
    assert balance_brackets("(ይዤ ልምጣ) እኔም") == "(ይዤ ልምጣ) እኔም"
