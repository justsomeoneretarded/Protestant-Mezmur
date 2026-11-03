from pathlib import Path

import pytest

from mezmur import fetch, lyrics_sources, web
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


REQUESTED_CHANNELS = {"yesemayzega12", "protestant_Song_lyrics1", "bestprotestant_songs"}
SEARCH_RECORDS = [
    {"title_am": "ነፍሴን በፊትህ", "artist_am": "ቃል ኪዳን ሊሊ ጥላሁን",
     "title_latin": "Nefsen Befith", "artist_latin": "Kalkidan Lily Tilahun"},
    {"title_am": "ነፍሴን በፊትህ", "artist_am": "ቃል ኪዳን ሊሊ ጥላሁን"},
    {"title_latin": "Nefsen Befith", "artist_latin": "Kalkidan Lily Tilahun"},
]


def assert_english_lyric_queries(queries):
    assert queries
    assert all(query.split()[-1] == "lyrics" and "ግጥም" not in query for query in queries)


@pytest.mark.parametrize("record", SEARCH_RECORDS)
def test_web_title_search_uses_english_lyrics_keyword(record):
    queries = lyrics_sources._queries(record)
    assert_english_lyric_queries(queries)
    assert len(queries) == len(set(queries)) <= 9
    assert any("wikimezmur" in query for query in queries)
    assert any("mezmuroch" in query for query in queries)


@pytest.mark.parametrize("record", SEARCH_RECORDS)
def test_youtube_title_and_phrase_search_use_english_lyrics_keyword(monkeypatch, record):
    queries = []
    monkeypatch.setattr(web, "youtube_search", lambda query: queries.append(query) or [])
    lyrics_sources.youtube_titles(record)
    lyrics_sources.from_youtube(record, phrases=["ነፍሴን በፊትህ"], log=lambda *a, **k: None)
    assert_english_lyric_queries(queries)
    assert "ነፍሴን በፊትህ lyrics" in queries


def test_missing_song_identity_does_not_search_youtube_for_lyrics_alone(monkeypatch):
    queries = []
    monkeypatch.setattr(web, "youtube_search", lambda query: queries.append(query) or [])
    assert lyrics_sources.youtube_titles({}) == []
    assert queries == []


def test_builtin_lyrics_channels_include_requested_sources(monkeypatch):
    monkeypatch.delenv("LYRICS_CHANNELS", raising=False)
    channels = fetch.lyrics_channels()
    assert REQUESTED_CHANNELS <= set(channels)
    assert {"MezmurLyrics0", "mezmuroch1"} <= set(channels)


def test_custom_and_learned_channels_extend_defaults_without_duplicates(monkeypatch):
    monkeypatch.setenv("LYRICS_CHANNELS", " @YESEMAYZEGA12, @CustomLyrics, CustomLyrics, , ")
    channels = fetch.lyrics_channels(["@customlyrics", "@VerifiedLyrics", ""])
    assert REQUESTED_CHANNELS <= set(channels)
    assert {"CustomLyrics", "VerifiedLyrics"} <= set(channels)
    assert len(channels) == len({channel.lower() for channel in channels})
    assert all(channel and not channel.startswith("@") for channel in channels)


def test_phrase_search_uses_english_web_keyword_and_all_telegram_sources(monkeypatch):
    queries, searched_channels = [], []
    monkeypatch.setenv("LYRICS_CHANNELS", "@CustomLyrics")
    channels = fetch.lyrics_channels()
    phrase = "ነፍሴን በፊትህ"

    def public_search(channel, query, **kwargs):
        assert query == phrase
        searched_channels.append(channel)
        return []

    monkeypatch.setattr(web, "telegram_public_search", public_search)
    monkeypatch.setattr(web, "search", lambda query, **kwargs: queries.append(query) or [])
    lyrics_sources.from_phrases(SEARCH_RECORDS[0], [phrase], channels, log=lambda *a, **k: None)
    assert_english_lyric_queries(queries)
    assert queries == [f'"{phrase}" lyrics']
    assert searched_channels == channels
    assert REQUESTED_CHANNELS | {"CustomLyrics"} <= set(searched_channels)


@pytest.mark.parametrize("record,artist", [
    ({"title_latin": "Yedestayaye Eleleta | Aster Abebe Selam Official Lyrics",
      "artist_am": "አስቴር አበበ", "artist_latin": "Yedestayaye Eleleta"}, "Aster Abebe"),
    ({"title_am": "መደምደሚያዬ", "artist_am": "አገኘሁ ይደግ",
      "artist_latin": "<unknown>"}, "Agegnehu Yideg"),
])
def test_all_title_query_paths_clean_channel_brands_and_unknowns(monkeypatch, record, artist):
    queries = []
    monkeypatch.setattr(web, "youtube_search", lambda q: queries.append(q) or [])
    monkeypatch.setattr(web, "search", lambda q, **k: queries.append(q) or [])
    lyrics_sources.youtube_titles(record)
    lyrics_sources.from_youtube(record, log=lambda *a, **k: None)
    lyrics_sources.from_web(record, log=lambda *a, **k: None)
    telegram_queries = fetch._title_queries(record)
    assert queries and telegram_queries
    joined = " ".join(queries + telegram_queries).casefold()
    assert "yedest" not in joined and "eleleta" not in joined and "unknown" not in joined
    assert artist.casefold() in joined
    assert all(q.count(artist) <= 1 for q in queries)
    assert_english_lyric_queries(queries)


def test_missing_song_title_does_not_make_broad_unusable_queries(monkeypatch):
    queries = []
    monkeypatch.setattr(web, "youtube_search", lambda q: queries.append(q) or [])
    r = {"title_latin": "<unknown>", "artist_latin": "Agegnehu Yideg"}
    assert lyrics_sources._queries(r) == []
    assert lyrics_sources.youtube_titles(r) == []
    lyrics_sources.from_youtube(r, log=lambda *a, **k: None)
    assert queries == []


def test_structured_lyrics_apis_and_direct_site_lookups_use_clean_identity(monkeypatch):
    tracks, requests = [], []
    record = {"title_latin": "Yedestayaye Eleleta | Aster Abebe Selam Official Lyrics",
              "artist_am": "አስቴር አበበ", "artist_latin": "<unknown>"}
    monkeypatch.setattr(web, "lrclib", lambda title, artist, seconds: tracks.append((title, artist)) or [])
    monkeypatch.setattr(web, "get", lambda url, **kwargs: requests.append((url, kwargs)) or None)
    lyrics_sources.from_lrclib(record, 180, log=lambda *a, **k: None)
    lyrics_sources.from_sites(record, log=lambda *a, **k: None)
    assert tracks == [("Selam", "Aster Abebe")]
    assert requests and any("Aster_Abebe" in url for url, _ in requests)
    assert any(kwargs.get("params") == {"q": "Selam"} for _, kwargs in requests)
    assert not any("unknown" in str(item).lower() or "Yedest" in str(item) for item in requests)
