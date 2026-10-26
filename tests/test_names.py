from mezmur.names import learn_from_titles, resolve

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
