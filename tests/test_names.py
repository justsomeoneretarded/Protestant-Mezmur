import pytest

from mezmur.names import clean_name, finalize, learn_from_titles, resolve, search_record

ARTISTS = [{"latin": "Yishak Sedik", "amharic": "ይስሐቅ ሰድቅ", "aliases": ["Yishak Sedik Official"]},
           {"latin": "Addisalem Assefa", "amharic": "አዲስዓለም አሰፋ"}]


def test_double_script_telegram_title():
    r = resolve({"title": "የሌለው አቻ//Yelelew acha//ይስሐቅ ሰድቅ//Yishak Sedik", "performer": "Yishak Sedik Official",
                 "caption": "🎹 ዘማሪ ይስሐቅ ሰዲቅ 🎹\n አቻ የሌለው \nJoin us @mezmuroch1"}, ARTISTS)
    assert (r["title_am"], r["title_latin"], r["artist_am"], r["artist_latin"]) == \
        ("የሌለው አቻ", "Yelelew Acha", "ይስሐቅ ሰድቅ", "Yishak Sedik")


def test_latin_only_tags_learn_geez_title_from_youtube_titles():
    r = resolve({"title": "Sebariw Geta Kefite Wetoal", "performer": "Addisalem Assefa",
                 "filename": "1. Addisalem Assefa - Sebariw Geta Kefite Wetoal (128).mp3"}, ARTISTS)
    assert r["title_am"] == "" and r["artist_am"] == "አዲስዓለም አሰፋ"
    r = learn_from_titles(r, ["አዲስአለም አሰፋ // ሰባሪው ጌታ ከፊቴ ወቷል //Addisalem Assefa //Sebariw Geta Kefite wetal// 2025"])
    assert r["title_am"] == "ሰባሪው ጌታ ከፊቴ ወቷል"


def test_caption_only_with_known_artist():
    r = resolve({"caption": "ሲወድ ጌታ - አዲስዓለም አሰፋ"}, ARTISTS)
    assert r["title_am"] == "ሲወድ ጌታ" and r["artist_latin"] == "Addisalem Assefa" and r["title_latin"] == "Siwed Geta"


@pytest.mark.parametrize("placeholder", ["<unknown>", "[Unknown Artist]", "Unknown", "unknow", "N/A"])
def test_missing_performer_does_not_hide_useful_id3_artist(placeholder):
    r = resolve({"title": "Siwed Geta", "performer": placeholder,
                 "tags": {"artist": "Addisalem Assefa"}}, ARTISTS)
    assert r["artist_latin"] == "Addisalem Assefa"
    assert r["title_latin"] == "Siwed Geta"


def test_song_32_uses_known_singer_instead_of_literal_unknown():
    r = finalize({"title_am": "መደምደሚያዬ", "title_latin": "Medmdemiyaye",
                  "artist_am": "አገኘሁ ይደግ", "artist_latin": "<unknown>", "album": "<unknown>"})
    assert r["artist_latin"] == "Agegnehu Yideg" and not r["album"]
    q = search_record(r)
    assert q["artist_latin"] == "Agegnehu Yideg"


@pytest.mark.parametrize("brand", ["Yedestayaye Eleleta", "Yedestayeeye Eleleta",
                                  "Yedestayaeye Eleleta", "የደስታዬ እልልታ",
                                  "ተወዳጁ የደስታዬ እልልታ", "Tewedaju Yedestaye Ellta"])
def test_channel_brand_is_not_aster_abebes_name_or_song_title(brand):
    r = resolve({"title": f"{brand} | Aster Abebe - Selam",
                 "performer": brand, "filename": f"{brand}_Aster Abebe_Selam.mp3"})
    assert r["artist_latin"] == "Aster Abebe"
    assert r["artist_am"] == "አስቴር አበበ"
    assert r["title_latin"] == "Selam"
    assert brand.casefold() not in " ".join(search_record(r)[k] for k in
        ("title_am", "title_latin", "artist_am", "artist_latin")).casefold()


def test_embedded_singer_is_removed_from_search_title_without_losing_song_words():
    r = resolve({"title": "Aster Abebe Selam", "performer": "Yedestayaye Eleleta"})
    assert r["title_latin"] == "Selam" and r["artist_latin"] == "Aster Abebe"


def test_only_a_singer_does_not_invent_a_song_title():
    r = resolve({"title": "<unknown>", "performer": "Agegnehu Yideg"})
    assert r["artist_latin"] == "Agegnehu Yideg"
    assert r["title_latin"] == r["title_am"] == ""


@pytest.mark.parametrize("title", ["Unknown Road", "The Unknown", "Unknown Love", "ያልታወቀ"])
def test_real_titles_are_not_erased_as_missing_metadata(title):
    assert clean_name(title) == title


def test_search_record_cleans_legacy_names_without_mutating_input():
    original = {"title_latin": "Yedestayaye Eleleta | Aster Abebe Selam Official Lyrics",
                "artist_am": "አስቴር አበበ", "artist_latin": "Yedestayaye Eleleta"}
    r = search_record(original)
    assert r["title_latin"] == "Selam"
    assert r["artist_latin"] == "Aster Abebe"
    assert original["artist_latin"] == "Yedestayaye Eleleta"


def test_placeholders_do_not_prevent_learning_geez_from_search_results():
    r = learn_from_titles({"title_am": "<unknown>", "title_latin": "Siwed Geta"},
                          ["ሲወድ ጌታ // Siwed Geta"])
    assert r["title_am"] == "ሲወድ ጌታ"


@pytest.mark.parametrize("info,artist,title", [
    ({"title": "ራስህን አኗሪ - Addisalem Assefa", "performer": "ተወዳጁ የደስታዬ እልልታ",
      "filename": "Addisalem_assefa_አዲስአለም_አሠፋ_Rashn_anuri_ራስህን_አኗሪ_New_album_2025M.mp3",
      "caption": "ራስህን አኗሪ\nአዲስዓለም አሰፋ - New Album\n▷ ▸ Albastros.com ◂ ◁"},
     "Addisalem Assefa", "Rashn Anuri"),
    ({"title": "ስምህን ጠራሁ - Aster Abebe", "performer": "ተወዳጁ የደስታዬ እልልታ",
      "filename": "_SIMIHIN_TERAHU_ስምህን_ጠራሁ_By_Aster_Abebe_Official_Gospel_Song_202.mp3",
      "caption": "ስምህን ጠራሁ\nአስቴር አበበ - New Clip\n▷ ▸ Albastros.com ◂ ◁"},
     "Aster Abebe", "SIMIHIN TERAHU"),
])
def test_actual_songs_39_and_40_keep_full_filename_titles_and_real_singers(info, artist, title):
    r = resolve(info)
    assert r["artist_latin"] == artist and r["title_latin"] == title
    assert not any("Albastros" in part for part in r["evidence"])
