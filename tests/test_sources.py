from pathlib import Path

from mezmur import web
from mezmur.lyrics_sources import from_library, latin_variants, phrases_from_draft


def test_phrases_prefer_repeated_distinctive_pairs():
    words = "ኤልሻዳይ ነው በስራው ትልቅ ሌላ ቃል ኤልሻዳይ ነው በስራው ትልቅ ነው ነው".split()
    draft = [{"word": w, "probability": 0.9} for w in words]
    phrases = phrases_from_draft(draft)
    assert phrases[0] in {"ኤልሻዳይ ነው", "ነው በስራው", "በስራው ትልቅ"}
    assert "ነው ነው" not in phrases


def test_latin_variants():
    assert latin_variants("Addis Kidan", "አዲስ ኪዳን")[:2] == ["Addis Kidan", "Adis Kidan"]


def test_library_finds_song_by_title_and_singer(tmp_path):
    d = tmp_path / "lyrics" / "yishak-sedik"
    d.mkdir(parents=True)
    (d / "el-shadai.txt").write_text("---\ntitle_am: ኤልሻዳይ\nartist_latin: Yishak Sedik\n---\nዛሬም ይሰራል ×3\n"
                                     "ዛሬም እየሱስ ይሰራል\nአምላኬማ አባቴማ\n", encoding="utf-8")
    got = from_library({"title_am": "ኤልሻዳይ", "artist_latin": "Yishak Sedik"}, "", tmp_path / "lyrics")
    assert got and got[0]["stanzas"][0]["repeats"][0] == 3


def test_public_telegram_and_caption_parsers(monkeypatch):
    class Page:
        status_code, encoding, content = 200, "utf-8", b"x"
        text = ('<div class="tgme_widget_message" data-post="Ch/7"><div class="tgme_widget_message_text js-message_text"'
                ' dir="auto">ዛሬም ይሰራል<br/>ዛሬም እየሱስ ይሰራል</div></div>')
    monkeypatch.setattr(web, "get", lambda *a, **k: Page())
    assert web.telegram_public_search("Ch", "ዛሬም")[0]["text"] == "ዛሬም ይሰራል\nዛሬም እየሱስ ይሰራል"
    Page.text = '<transcript><text start="1" dur="2">ዛሬም ይሰራል</text><text start="3" dur="2">አምላኬ</text></transcript>'
    assert web.youtube_caption_text("u") == "ዛሬም ይሰራል\nአምላኬ"
