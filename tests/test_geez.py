from mezmur.geez import fold, latin_key, romanize, skeleton


def test_youtube_style_romanization():
    cases = {
        "ሲወድ ጌታ": "siwed geta", "ይስሐቅ ሰድቅ": "yishak sedik", "የሌለው አቻ": "yelelew acha",
        "ተመስገን": "temesgen", "እግዚአብሔር": "egziabher", "ኢየሱስ": "iyesus", "ክብር": "kibir",
        "መንፈስ ቅዱስ": "menfes kidus", "ምስጋና": "misgana", "ሃሌሉያ": "haleluya",
    }
    for geez, latin in cases.items():
        assert romanize(geez) == latin


def test_title_case_and_punctuation():
    assert romanize("ሲወድ ፡ ጌታ ።", title_case=True) == "Siwed Geta."


def test_homophones_fold_to_one_key():
    assert fold("ሐሌሉያ") == fold("ሃሌሉያ") == fold("ሀሌሉያ")
    assert fold("ፀሐይ") == fold("ጸሀይ")
    assert fold("ዓለም") == fold("አለም")
    assert fold("ሠላም") == fold("ሰላም")


def test_cross_script_keys():
    assert latin_key("ሲወድ ጌታ") == latin_key("Siwed Geta")
    assert skeleton("ሰባሪው ጌታ") == skeleton("Sebariw Geta")
