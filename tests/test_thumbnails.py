from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from mezmur import thumbnail
from mezmur.youtube import write_kit


META = {"title_am": "ሲወድ ጌታ", "title_latin": "Siwed Geta",
        "artist_am": "አዲስዓለም አሰፋ", "artist_latin": "Addisalem Assefa"}
BACKGROUND = np.full((720, 1280, 3), (60, 80, 90), dtype=np.uint8)
PORTRAIT = np.full((400, 400, 3), (40, 160, 70), dtype=np.uint8)


@pytest.mark.parametrize("variant", ["a", "c"])
def test_text_only_variants_never_draw_a_person(variant):
    plain = thumbnail.compose(META, background=BACKGROUND, variant=variant)
    supplied = thumbnail.compose(META, background=BACKGROUND, photo=PORTRAIT,
                                  person=Image.new("RGBA", (300, 500), "red"), variant=variant)
    np.testing.assert_array_equal(np.array(plain), np.array(supplied))
    assert plain.size == (1280, 720)


@pytest.mark.parametrize("variant", ["b", "d"])
def test_portrait_is_large_circular_and_centered(variant):
    image = np.array(thumbnail.compose(META, background=BACKGROUND, photo=PORTRAIT, variant=variant))
    np.testing.assert_array_equal(image[242, 640], PORTRAIT[200, 200][::-1])
    assert not np.array_equal(image[81, 479], PORTRAIT[200, 200][::-1])
    np.testing.assert_array_equal(image[242, 485], image[242, 794])


@pytest.mark.parametrize("pair", [("a", "c"), ("b", "d")])
def test_unframed_versions_differ_only_at_the_outer_rectangle(pair):
    a, b = [np.array(thumbnail.compose(META, background=BACKGROUND, photo=PORTRAIT, variant=v)) for v in pair]
    changed = np.any(a != b, axis=2)
    assert changed[10, 10] and changed.sum() > 5000
    frame = np.zeros(changed.shape, dtype=bool)
    frame[10:13, 10:1270] = frame[707:710, 10:1270] = True
    frame[10:710, 10:13] = frame[10:710, 1267:1270] = True
    assert not changed[~frame].any()


@pytest.mark.parametrize("variant", ["b", "d"])
def test_missing_portrait_uses_text_without_inventing_a_face(variant):
    image = thumbnail.compose(META, background=BACKGROUND, photo=None, variant=variant)
    assert image.size == (1280, 720)


@pytest.mark.parametrize("compact", [False, True])
def test_long_text_stays_centered_inside_its_assigned_band(monkeypatch, compact):
    calls = []

    def capture(canvas, xy, text, font, *args, **kwargs):
        assert xy[0] == 640 and kwargs["anchor"] == "ma"
        assert font.length(text) <= 1120
        calls.append((xy[1], sum(font.metrics()), font.size))

    monkeypatch.setattr(thumbnail, "gradient_text", capture)
    monkeypatch.setattr(thumbnail, "shadow_text", capture)
    long = {**META, "title_am": "ስለማይለወጠው የጌታ ፍቅር እና ምሕረት " * 4,
            "title_latin": "The Love And Mercy Of The Lord " * 4}
    top = 428 if compact else 54
    thumbnail._text_block(Image.new("RGBA", (1280, 720)), long, (255, 200, 90), (225, 120, 30),
                          top=top, compact=compact)
    assert calls[0][0] >= top and calls[-1][0] + calls[-1][1] <= 666
    assert all(b[0] >= a[0] + a[1] for a, b in zip(calls, calls[1:]))
    assert max(c[2] for c in calls) <= (60 if compact else 132)


def test_four_thumbnail_files_are_in_the_upload_kit_and_artifact(tmp_path):
    kit = write_kit(tmp_path, META, [], [], 200, set(), [], "test", 180, {})
    expected = ["thumbnail.jpg", "thumbnail-b.jpg", "thumbnail-c.jpg", "thumbnail-d.jpg"]
    assert kit["thumbnails"] == expected
    workflow = yaml.load((Path(__file__).parents[1] / ".github/workflows/generate.yml").read_text(), Loader=yaml.BaseLoader)
    upload = next(s for s in workflow["jobs"]["song"]["steps"] if s.get("name", "").startswith("Upload the YouTube kit"))
    assert all(name in upload["with"]["path"] for name in expected)
    assert "thumbnail-d.jpg" in (tmp_path / "UPLOAD.md").read_text()
