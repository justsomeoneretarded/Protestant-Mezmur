import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from mezmur import audio, song


def test_disabled_augmentation_does_not_load_the_native_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mezmur.augmentation", None)
    source = tmp_path / "original.m4a"
    work = tmp_path / "unused"
    path, metadata = audio.prepare_source(source, work)
    assert path == source
    assert metadata == {"enabled": False}
    assert not work.exists()


def test_prepare_source_supports_m4a_and_does_not_overwrite_original(tmp_path):
    sr = 44100
    t = np.arange(sr * 2) / sr
    wave = tmp_path / "source.wav"
    source = tmp_path / "audio.m4a"
    sf.write(wave, 0.3 * np.sin(2 * np.pi * 220 * t), sr)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wave),
                    "-c:a", "aac", str(source)], check=True)
    before = source.read_bytes()
    prepared, metadata = audio.prepare_source(source, tmp_path / "work", augment=True, seed=21)
    result, result_sr = sf.read(prepared, always_2d=True)
    assert source.read_bytes() == before
    assert metadata["enabled"] is True and metadata["seed"] == 21
    assert metadata["privacy_guaranteed"] is False
    assert result_sr == 48000 and result.shape[1] == 2
    assert len(result) == metadata["output_frames"][-1] == metadata["input_frames"][-1]
    assert abs(len(result) / result_sr - 2) < 0.05
    # Replays may reuse a working directory without overwriting the original or
    # failing on a previous augmentation output. The seed still stays stable.
    replay, _ = audio.prepare_source(source, tmp_path / "work", augment=True, seed=21)
    assert replay != prepared
    np.testing.assert_array_equal(sf.read(replay)[0], sf.read(prepared)[0])


def test_song_aligns_to_the_same_processed_source_it_masters(monkeypatch, tmp_path):
    inp, out, work = [tmp_path / name for name in ("input", "output", "work")]
    inp.mkdir()
    original = inp / "audio.wav"
    original.touch()
    (inp / "info.json").write_text(json.dumps({"message_id": 21}))
    augmented = tmp_path / "augmented.wav"
    metadata = {"enabled": True, "formant_semitones": 0.1, "seed": 21}
    calls = []

    def tags(path):
        assert path == original
        return {}

    def prepare(source, folder, *, augment, seed):
        assert source == original and folder == work
        assert augment is True and seed == 21
        calls.append("prepare")
        return augmented, metadata

    def master(source, target):
        assert source == augmented and target == work / "master.wav"
        calls.append("master")
        return {}

    class StopBeforeInference(Exception):
        pass

    def separate(source, folder, **kwargs):
        assert source == augmented and folder == work
        assert calls == ["prepare", "master"]
        assert json.loads((out / "augmentation.json").read_text()) == metadata
        raise StopBeforeInference

    monkeypatch.setattr(audio, "tags", tags)
    monkeypatch.setattr(audio, "prepare_source", prepare)
    monkeypatch.setattr(audio, "master", master)
    monkeypatch.setattr(song.names, "resolve", lambda _: {})
    monkeypatch.setattr(song, "common_duration", lambda _: 2)
    monkeypatch.setattr(song.asr, "vocabulary", lambda: {})
    monkeypatch.setattr(song, "separate", separate)
    with pytest.raises(StopBeforeInference):
        song.produce(inp, out, work, augment_audio=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_song_cli_passes_augmentation_flag(monkeypatch, enabled):
    argv = ["mezmur-song", "--input", "in", "--output", "out"]
    if enabled:
        argv.append("--augment-audio")
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    monkeypatch.setattr(song, "produce", lambda *a, **k: calls.append(k))
    song.main()
    assert calls[0]["augment_audio"] is enabled
