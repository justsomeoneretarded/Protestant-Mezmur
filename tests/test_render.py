import json
import subprocess

import numpy as np
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
