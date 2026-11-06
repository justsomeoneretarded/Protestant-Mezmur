import json
import subprocess

import numpy as np
import pytest
import scipy.io.wavfile as wav
from PIL import Image

from mezmur.features import analyze
from mezmur.subtitles import build_cues, write_ass
from mezmur.thumbnail import compose
from mezmur.visuals import Renderer, Source, render_video

META = {"title_am": "ሲወድ ጌታ", "title_latin": "Siwed Geta", "artist_am": "አዲስዓለም አሰፋ", "artist_latin": "Addisalem Assefa"}


def test_text_only_thumbnail(tmp_path):
    img = compose(META, background=None, person=None, variant="a")
    assert img.size == (1280, 720)


def test_small_render_with_burned_lyrics(tmp_path):
    sr = 22050
    x = (0.2 * np.sin(2 * np.pi * 330 * np.arange(sr * 6) / sr) * 20000).astype(np.int16)
    wav.write(tmp_path / "a.wav", sr, x)
    words = [{"word": w, "start": 3.0 + i * 0.5, "end": 3.4 + i * 0.5, "line": 0, "word_index": i}
             for i, w in enumerate(["ሲወድ", "ጌታ", "ነው"])]
    cues = build_cues(words)
    write_ass(cues, tmp_path / "l.ass")
    feats = analyze(tmp_path / "a.wav", fps=12)
    r = Renderer(640, 360, 12, 6.0, feats, [Source("procedural")], cues, META, seed=1)
    render_video(r, tmp_path / "a.wav", tmp_path / "l.ass", tmp_path / "v.mp4", preset="ultrafast")
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(tmp_path / "v.mp4")],
                                     capture_output=True, text=True, check=True).stdout)
    kinds = {s["codec_type"] for s in info["streams"]}
    assert kinds == {"video", "audio"}


@pytest.mark.parametrize("height", [720, 1080])
def test_native_lyrics_are_larger_with_no_romanized_row_at_both_resolutions(tmp_path, height):
    from mezmur.subtitles import FONTS

    cues = build_cues([{"word": word, "start": 0.8 + index * 0.3,
                       "end": 1.0 + index * 0.3, "line": 0}
                      for index, word in enumerate(["ሃሌሉያ", "ክብር", "ለስሙ"])])
    width = height * 16 // 9

    def frame(romanized):
        ass = tmp_path / ("comparison.ass" if romanized else "amharic-only.ass")
        write_ass(cues, ass, romanized=romanized, preview=False)
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r=10:d=2",
            "-vf", f"ass='{ass}':fontsdir='{FONTS}'", "-ss", "1.2", "-frames:v", "1",
            "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ], capture_output=True, check=True)
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)

    current, comparison = frame(False), frame(True)
    current_mask = current.max(axis=2) > 32
    comparison_mask = comparison.max(axis=2) > 32
    main_band = slice(round(height * 0.40), round(height * 0.58))

    def ink_height(mask):
        rows = np.flatnonzero(mask[main_band].any(axis=1))
        assert rows.size > 0
        return rows[-1] - rows[0] + 1

    assert ink_height(current_mask) > ink_height(comparison_mask) * 1.10
    roman_band = slice(round(height * 0.59), round(height * 0.69))
    assert not current_mask[roman_band].any()
    assert comparison_mask[roman_band].sum() > 20
    margin = round(width * 0.04)
    assert not current_mask[:, :margin].any() and not current_mask[:, -margin:].any()
