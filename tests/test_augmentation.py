import json

import numpy as np
import pytest
import soundfile as sf

from mezmur.augmentation import _time_map, augment_audio, augment_file
from mezmur import augmentation


SR = 16000


def tone(seconds=2, frequency=220):
    t = np.arange(round(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


@pytest.mark.parametrize("samples", [
    np.array([], dtype=float), np.zeros((4, 0)), np.zeros((2, 3, 4)),
    np.array([np.nan]), np.array([np.inf]), np.array([1.01]), np.zeros((2, 33)),
])
def test_reject_invalid_audio(samples):
    with pytest.raises(ValueError):
        augment_audio(samples, SR)


def test_integer_pcm_requires_explicit_conversion():
    with pytest.raises(TypeError, match="floating"):
        augment_audio(np.zeros(100, dtype=np.int16), SR)


@pytest.mark.parametrize("sr", [True, 16000.0, 7999, 192001])
def test_reject_invalid_sample_rate(sr):
    with pytest.raises(ValueError):
        augment_audio(tone(), sr)


@pytest.mark.parametrize("kwargs", [
    {"max_speed_change": -0.01}, {"max_speed_change": 0.051},
    {"max_speed_change": np.nan}, {"max_formant_semitones": 1.01},
    {"control_seconds": 0.1}, {"control_seconds": np.inf},
])
def test_reject_unsubtle_or_nonfinite_settings(kwargs):
    with pytest.raises(ValueError):
        augment_audio(tone(), SR, **kwargs)


def test_no_effects_is_an_independent_exact_copy():
    x = tone()
    result = augment_audio(x, SR, seed=5, max_speed_change=0, max_formant_semitones=0)
    np.testing.assert_array_equal(result.audio, x)
    assert not np.shares_memory(result.audio, x)
    assert result.metadata()["boundary_context_seconds"] == 0


@pytest.mark.parametrize("preserve", [True, False])
def test_time_map_is_dynamic_monotonic_and_speed_bounded(preserve):
    n = SR * 8
    src, dst = _time_map(n, SR, np.random.default_rng(7), 0.015, 1, preserve)
    assert src[0] == dst[0] == 0
    assert src[-1] == n
    assert np.all(np.diff(src) > 0) and np.all(np.diff(dst) > 0)
    speeds = np.diff(src) / np.diff(dst)
    rounding_tolerance = 2 / np.min(np.diff(src))
    assert np.max(np.abs(speeds - 1)) <= 0.015 + rounding_tolerance
    assert np.ptp(speeds) > 0.001
    if preserve:
        assert dst[-1] == n


def test_native_engine_preserves_shape_duration_and_stereo_coherence():
    x = np.column_stack([tone(), tone()])
    original = x.copy()
    result = augment_audio(x, SR, seed=3)
    assert result.audio.shape == x.shape
    assert result.audio.dtype == np.float32
    assert result.metadata()["boundary_context_seconds"] == 0.25
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio)) <= 0.999001
    np.testing.assert_allclose(result.audio[:, 0], result.audio[:, 1], atol=1e-6)
    np.testing.assert_array_equal(x, original)


def test_same_seed_same_result_and_new_seed_changes_audio():
    x = tone()
    first = augment_audio(x, SR, seed=4)
    again = augment_audio(x, SR, seed=4)
    other = augment_audio(x, SR, seed=5)
    np.testing.assert_array_equal(first.audio, again.audio)
    assert not np.array_equal(first.audio, other.audio)
    assert abs(first.formant_semitones) <= 0.3


def test_no_intentional_pitch_shift():
    result = augment_audio(tone(3, frequency=440), SR, seed=8)
    middle = result.audio[SR:2 * SR]
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
    dominant_hz = np.fft.rfftfreq(len(middle), 1 / SR)[np.argmax(spectrum)]
    assert abs(dominant_hz - 440) <= 2


def test_silence_stays_silent():
    result = augment_audio(np.zeros(SR, dtype=np.float32), SR, seed=0)
    assert len(result.audio) == SR
    assert np.max(np.abs(result.audio)) == 0


def test_native_boundary_shortfall_does_not_remove_song_samples(monkeypatch):
    x = np.linspace(-0.4, 0.4, SR, dtype=np.float32)

    def finish_short(planar, sr, src, dst, formants):
        assert planar.shape[1] == len(x) + round(sr * 0.25)
        # The real song-23 failure: the native engine returned 563 fewer frames.
        return planar[:, :-563].T.copy()

    monkeypatch.setattr(augmentation, "_render_native", finish_short)
    result = augment_audio(x, SR, seed=23, max_speed_change=0)
    np.testing.assert_array_equal(result.audio, x)
    assert result.audio[-1] == x[-1]


def test_substantial_native_truncation_still_fails(monkeypatch):
    def truncate(planar, *args):
        return planar[:, :SR // 2].T.copy()

    monkeypatch.setattr(augmentation, "_render_native", truncate)
    with pytest.raises(RuntimeError, match="boundary context"):
        augment_audio(tone(), SR, seed=23)


def test_invalid_native_samples_still_fail(monkeypatch):
    def invalid(planar, *args):
        output = planar.T.copy()
        output[-1, 0] = np.nan
        return output

    monkeypatch.setattr(augmentation, "_render_native", invalid)
    with pytest.raises(RuntimeError, match="non-finite"):
        augment_audio(tone(), SR, seed=23)


def test_native_longer_stereo_clip_keeps_every_frame():
    sr = 48000
    t = np.arange(sr * 20 + 700) / sr
    signal = (0.25 * np.sin(2 * np.pi * 220 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 3 * t)))
    x = np.column_stack([signal, signal]).astype(np.float32)
    result = augment_audio(x, sr, seed=23)
    assert result.audio.shape == x.shape
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio[-sr // 20:])) > 0.05


def test_short_input_and_multichannel():
    for x in [tone()[:1], tone()[:17], tone()[:63], np.column_stack([tone(0.2)] * 3)]:
        result = augment_audio(x, SR, seed=10)
        assert result.audio.shape == x.shape
        assert np.isfinite(result.audio).all()


def test_formant_shifting_actually_changes_the_spectral_envelope():
    sr = 48000
    t = np.arange(sr * 3) / sr
    frequencies = np.arange(1, 100) * 100
    weights = (np.exp(-0.5 * ((frequencies - 700) / 150) ** 2)
               + 0.5 * np.exp(-0.5 * ((frequencies - 1800) / 200) ** 2))
    x = np.sum(weights[:, None] * np.sin(2 * np.pi * frequencies[:, None] * t), axis=0)
    x = (x / np.max(np.abs(x)) * 0.5).astype(np.float32)
    result = augment_audio(x, sr, seed=7, max_speed_change=0, max_formant_semitones=1)

    def centroid(y):
        spectrum = np.abs(np.fft.rfft(y[sr:2 * sr] * np.hanning(sr)))
        return np.sum(np.fft.rfftfreq(sr, 1 / sr) * spectrum) / np.sum(spectrum)

    assert result.formant_semitones < 0
    assert centroid(result.audio) < centroid(x) - 10


def test_optional_duration_change_uses_reported_length():
    result = augment_audio(tone(), SR, seed=2, preserve_duration=False)
    assert len(result.audio) == result.output_frames[-1]
    assert len(result.audio) != len(tone())


def test_annotation_mapping_and_range_check():
    result = augment_audio(tone(), SR, seed=1)
    times = np.array([0.0, 0.5, 1.0, 2.0])
    mapped = result.map_times(times)
    assert mapped[0] == 0 and mapped[-1] == 2
    assert np.all(np.diff(mapped) > 0)
    with pytest.raises(ValueError):
        result.map_times(np.array([np.nan]))
    with pytest.raises(ValueError):
        result.map_times(np.array([2.01]))


def test_percussive_events_remain_near_planned_anchors():
    x = np.zeros(SR * 4, dtype=np.float32)
    times = np.array([0.5, 1.2, 2.1, 3.2])
    pulse = np.hanning(81).astype(np.float32) * 0.6
    for t in times:
        start = round(t * SR) - 40
        x[start:start + len(pulse)] = pulse
    result = augment_audio(x, SR, seed=3, max_speed_change=0.05, max_formant_semitones=0)
    for expected in result.map_times(times):
        start = round((expected - 0.06) * SR)
        window = result.audio[start:start + round(0.12 * SR)]
        observed = (start + np.argmax(np.abs(window))) / SR
        assert abs(observed - expected) < 0.04


@pytest.mark.parametrize("suffix", [".wav", ".flac"])
def test_lossless_file_output_and_no_overwrite(tmp_path, suffix):
    source = tmp_path / "source.wav"
    target = tmp_path / ("new" + suffix)
    sf.write(source, tone(), SR, subtype="PCM_24")
    original = source.read_bytes()
    meta = augment_file(source, target, seed=3)
    y, sr = sf.read(target)
    assert sr == SR and len(y) == SR * 2
    assert source.read_bytes() == original
    assert sf.info(target).subtype == "PCM_24"
    sidecar = target.with_suffix(suffix + ".augmentation.json")
    assert json.loads(sidecar.read_text()) == meta
    assert meta["seed"] == 3 and meta["privacy_guaranteed"] is False
    assert str(source) not in sidecar.read_text()
    with pytest.raises(FileExistsError):
        augment_file(source, target)
    with pytest.raises(FileExistsError):
        augment_file(source, source)
