"""Bounds, audio-driven motion and color-independent geometry for the bottom spectrum."""
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from mezmur.features import Features
from mezmur.subtitles import write_ass
from mezmur.visuals import Renderer, Source, render_video


def renderer(height=360, level=0.6, pulse=0.0, fps=30, accent=(255, 180, 70), bright=False, bands=None):
    if bands is None:
        bands = np.full((3, 32), level, dtype=np.float32)
    bands = np.asarray(bands, dtype=np.float32)
    features = Features(
        fps=fps, rms=np.ones(len(bands)), bands=bands, onset=np.zeros(len(bands)),
        beats=np.array([]), tempo=120, beat_env=np.full(len(bands), pulse), downbeats=np.array([]),
    )
    return Renderer(height * 16 // 9, height, fps, 60, features, [Source("procedural")],
                    [{"start": 8, "end": 30, "text": "Mezmur"}],
                    {"title_latin": "Song", "artist_latin": "Singer"}, accent=accent, bright=bright)


def settled(r, value=0, count=12):
    background = np.full((r.H, r.W, 3), value, dtype=np.uint8)
    for _ in range(count):
        output = r._spectrum(background.copy(), 0)
    return background, output


@pytest.mark.parametrize("height", [360, 720, 1080])
@pytest.mark.parametrize("bright", [False, True])
def test_tall_spectrum_stays_below_lyrics_and_inside_the_frame(height, bright):
    r = renderer(height, level=1, bright=bright)
    before, after = settled(r, value=235 if bright else 12)
    changed = np.any(before != after, axis=2)
    rows = np.flatnonzero(changed.any(axis=1))
    assert rows[0] >= round(height * 0.71)
    assert rows[-1] < round(height * 0.985)
    assert not changed[:round(height * 0.70)].any()
    above_baseline = changed[:round(height * 0.94)].any(axis=1)
    assert above_baseline.sum() >= height * 0.21
    assert after.shape == before.shape and after.dtype == np.uint8


def test_color_components_cannot_change_bar_heights():
    masks = []
    for accent in ((255, 40, 30), (40, 255, 60), (20, 80, 255)):
        r = renderer(accent=accent)
        _, frame = settled(r)
        masks.append(frame.max(axis=2) > 0)
    np.testing.assert_array_equal(masks[0], masks[1])
    np.testing.assert_array_equal(masks[0], masks[2])


def test_spectrum_is_mirrored_around_the_center():
    r = renderer(bands=np.tile(np.linspace(0.05, 1, 8), (3, 1)))
    _, frame = settled(r)
    mask = frame.max(axis=2) > 12
    width = round(r.W * 0.80)
    band_width = width / 16
    x0, heights = (r.W - width) // 2, []
    for k in range(16):
        left = int(x0 + k * band_width + band_width * 0.18)
        right = max(left + 1, int(x0 + (k + 1) * band_width - band_width * 0.18))
        rows = np.flatnonzero(mask[:round(r.H * 0.94), (left + right) // 2])
        heights.append(round(r.H * 0.94) - rows[0])
    np.testing.assert_allclose(heights, heights[::-1], atol=1)


def test_beat_pulse_strengthens_the_visual_without_touching_audio_features():
    quiet, beat = renderer(level=0.25), renderer(level=0.25, pulse=1)
    original = beat.f.bands.copy()
    settled(quiet)
    settled(beat)
    assert beat.bars.mean() > quiet.bars.mean() * 1.10
    np.testing.assert_array_equal(beat.f.bands, original)


def test_bars_and_peak_caps_fall_after_sound_stops():
    r = renderer(level=1)
    settled(r)
    peak = r.bars.copy()
    r.f.bands[:] = 0
    settled(r, count=1)
    assert np.all(r.bars < peak) and np.all(r._spectrum_peaks > r.bars)
    settled(r, count=60)
    assert r.bars.max() < 0.001 and r._spectrum_peaks.max() < 0.02


@pytest.mark.parametrize("pulse", [0.0, 1.0])
def test_silence_has_only_a_small_still_baseline_not_fake_motion(pulse):
    r = renderer(level=0, pulse=pulse)
    _, first = settled(r, count=1)
    _, last = settled(r, count=15)
    np.testing.assert_array_equal(first, last)
    assert not r.bars.any() and not r._spectrum_peaks.any()
    rows = np.flatnonzero((last.max(axis=2) > 12).any(axis=1))
    assert rows.size < r.H * 0.06


def test_attack_timing_is_consistent_across_frame_rates():
    results = []
    for fps in (15, 30, 60):
        r = renderer(level=0.4, fps=fps)
        settled(r, count=fps)
        results.append(r.bars)
    np.testing.assert_allclose(results[0], results[1], atol=1e-5)
    np.testing.assert_allclose(results[0], results[2], atol=1e-5)


def test_empty_or_nonfinite_visual_features_cannot_break_rendering():
    empty = renderer(bands=np.empty((3, 0)))
    before, after = settled(empty)
    np.testing.assert_array_equal(before, after)
    r = renderer(bands=np.tile([np.nan, np.inf, -np.inf, -1, 2], (3, 1)))
    before_features = r.f.bands.copy()
    _, output = settled(r)
    assert np.isfinite(r.bars).all() and 0 <= r.bars.min() <= r.bars.max() <= 1
    np.testing.assert_array_equal(r.f.bands, before_features)
    assert output.dtype == np.uint8


def test_native_encoded_spectrum_moves_with_signal_and_preserves_duration(tmp_path):
    fps, seconds, height = 12, 2, 360
    bands = np.zeros((fps * seconds, 32), np.float32)
    bands[fps:] = 0.9
    r = renderer(height, fps=fps, bands=bands)
    clip = SimpleNamespace(W=r.W, H=r.H, fps=fps, duration=seconds,
                           frame=lambda t, dt: r._spectrum(np.zeros((r.H, r.W, 3), np.uint8), t))
    source, ass, out = tmp_path / "audio.wav", tmp_path / "lyrics.ass", tmp_path / "video.mp4"
    sr = 22050
    sf.write(source, 0.1 * np.sin(2 * np.pi * 220 * np.arange(sr * seconds) / sr), sr)
    write_ass([], ass)
    render_video(clip, source, ass, out, preset="ultrafast", log=lambda *a, **k: None)

    def ink_rows(t):
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(out), "-ss", str(t), "-frames:v", "1",
            "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ], capture_output=True, check=True)
        frame = np.frombuffer(result.stdout, dtype=np.uint8).reshape(r.H, r.W, 3)
        assert frame[:round(height * 0.65)].max() < 12
        return np.count_nonzero((frame.max(axis=2) > 20).any(axis=1))

    assert ink_rows(1.7) > ink_rows(0.4) * 2
    result = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-of", "json", str(out)],
                            capture_output=True, text=True, check=True)
    assert abs(float(json.loads(result.stdout)["format"]["duration"]) - seconds) < 0.07
