import numpy as np
import scipy.io.wavfile as wav

from mezmur.features import analyze


def test_beats_and_tempo_on_click_track(tmp_path):
    sr, seconds = 22050, 20
    x = 0.1 * np.sin(2 * np.pi * 220 * np.arange(sr * seconds) / sr)
    rng = np.random.default_rng(0)
    for b in np.arange(0.5, seconds, 0.5):
        i = int(b * sr)
        x[i:i + 600] += rng.standard_normal(600) * np.exp(-np.arange(600) / 120)
    wav.write(tmp_path / "click.wav", sr, (x * 12000).astype(np.int16))
    f = analyze(tmp_path / "click.wav", fps=30)
    assert abs(f.tempo - 120) < 3
    inner = f.beats[(f.beats > 1) & (f.beats < seconds - 1)]
    assert np.all(np.abs(inner * 2 - np.round(inner * 2)) < 0.1)
