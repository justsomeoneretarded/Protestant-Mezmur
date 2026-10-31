import numpy as np
import pytest

from mezmur.features import Features
from mezmur.visuals import Renderer, Source, WATERMARK_TEXT


def renderer(height, bright=False):
    features = Features(
        fps=1, rms=np.zeros(60), bands=np.zeros((60, 8)), onset=np.zeros(60),
        beats=np.array([]), tempo=120, beat_env=np.zeros(60), downbeats=np.array([]),
    )
    cues = [{"start": 8, "end": 30, "text": "Mezmur"}]
    return Renderer(height * 16 // 9, height, 1, 60, features, [Source("procedural")],
                    cues, {"title_latin": "Song", "artist_latin": "Singer"}, bright=bright)


@pytest.mark.parametrize("height", [720, 1080])
@pytest.mark.parametrize("bright", [False, True])
def test_watermark_is_small_faded_and_inside_top_right(height, bright):
    r = renderer(height, bright)
    rgb, alpha, x, y = r.overlays.watermark
    h, w = rgb.shape[:2]
    assert WATERMARK_TEXT == "Rehoboth Mezmur"
    assert 0 < w < r.W * 0.20 and 0 < h < r.H * 0.06
    assert x > r.W * 0.75 and y > 0
    assert x + w == r.W - round(r.W * 0.025)
    assert y + h < r.H * 0.10
    assert 0.2 < float(alpha.max()) <= 0.4
    assert np.count_nonzero(alpha) > 50


@pytest.mark.parametrize("t", [1, 20, 55])
def test_watermark_is_present_in_intro_song_and_outro(t):
    r = renderer(720)
    assert 1 < r.intro_end < 20 < r.outro_start < 55
    r._ink = None
    # Remove changing backgrounds and spectrum state so only the watermark can
    # differ between the two frames, while retaining all intro/outro overlays.
    r.background = lambda _: np.full((r.H, r.W, 3), 100, dtype=np.uint8)
    r._grade = lambda frame, _: frame
    r._spectrum = lambda frame, _: frame
    r._countdown = lambda frame, _: frame
    stamp = r.overlays.watermark
    with_stamp = r.frame(t, 1)
    r.overlays.watermark = None
    without_stamp = r.frame(t, 1)
    changed = np.any(with_stamp != without_stamp, axis=2)
    rgb, _, x, y = stamp
    h, w = rgb.shape[:2]
    assert np.count_nonzero(changed[y:y + h, x:x + w]) > 50
    changed[y:y + h, x:x + w] = False
    assert not changed.any()
