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
    assert metadata["style"] == "subtle" and metadata["noise_snr_db"] == 50
    assert metadata["privacy_guaranteed"] is False
    assert result_sr == 48000 and result.shape[1] == 2
    assert len(result) == metadata["output_frames"][-1] == metadata["input_frames"][-1]
    assert abs(len(result) / result_sr - 2) < 0.05
    # Replays may reuse a working directory without overwriting the original or
    # failing on a previous augmentation output. The seed still stays stable.
    replay, _ = audio.prepare_source(source, tmp_path / "work", augment=True, seed=21)
    assert replay != prepared
    np.testing.assert_array_equal(sf.read(replay)[0], sf.read(prepared)[0])


def test_song_separates_original_before_augmentation(monkeypatch, tmp_path):
    inp, out, work = [tmp_path / name for name in ("input", "output", "work")]
    inp.mkdir()
    original = inp / "audio.wav"
    original.touch()
    (inp / "info.json").write_text(json.dumps({"message_id": 21}))
    calls = []

    def prepare(*args, **kwargs):
        pytest.fail("Effects must not run before the original lyrics pass verification")

    def master(source, target):
        assert source == original and target == work / "master.wav"
        calls.append("master-original")
        return {}

    class StopBeforeInference(Exception):
        pass

    def separate(source, folder, **kwargs):
        assert source == original and folder == work
        assert calls == ["master-original"]
        assert not (out / "augmentation.json").exists()
        raise StopBeforeInference

    monkeypatch.setattr(audio, "tags", lambda path: {})
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
    assert calls[0]["augmentation_style"] == "musical"
    assert calls[0]["noise_profile"] == "hiss" and calls[0]["space_effect"] == "room"


def test_musical_preset_applies_one_key_and_tempo_to_the_mix(tmp_path):
    source = tmp_path / "source.wav"
    sr = 16000
    sf.write(source, 0.3 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr), sr)
    prepared, metadata = audio.prepare_source(source, tmp_path / "work", augment=True, seed=23, style="musical")
    assert metadata["style"] == "musical"
    assert abs(metadata["pitch_semitones"]) == 2
    assert metadata["tempo_factor"] in (0.96, 1.04)
    assert metadata["noise_snr_db"] == 50
    assert sf.info(prepared).frames != metadata["input_frames"][-1]


def test_lyrics_lookup_uses_original_before_any_tempo_change(monkeypatch, tmp_path):
    inp = tmp_path / "input"
    inp.mkdir()
    source = inp / "audio.wav"
    source.touch()
    (inp / "info.json").write_text(json.dumps({"message_id": 23}))
    monkeypatch.setattr(audio, "tags", lambda _: {})
    monkeypatch.setattr(audio, "prepare_source", lambda *a, **k: pytest.fail("Premature audio effects"))
    monkeypatch.setattr(audio, "master", lambda *a: {})
    monkeypatch.setattr(song, "common_duration", lambda path: 2)
    monkeypatch.setattr(song.names, "resolve", lambda _: {})
    monkeypatch.setattr(song.names, "learn_from_titles", lambda record, _: record)
    monkeypatch.setattr(song.asr, "vocabulary", lambda: {})
    monkeypatch.setattr(song, "separate", lambda path, *a, **k: path)
    monkeypatch.setattr(song.asr, "frame_log_probs", lambda *a, **k: (np.zeros((1, 1)), 0.02))
    monkeypatch.setattr(song.asr, "vocal_energy", lambda *a: np.ones(1))
    monkeypatch.setattr(song.asr, "greedy_words", lambda *a: [])

    class StopAtLookup(Exception):
        pass

    def gather(record, mid, tg, corrections, duration, **kwargs):
        assert duration == 2
        raise StopAtLookup

    monkeypatch.setattr(song, "gather", gather)
    with pytest.raises(StopAtLookup):
        song.produce(inp, tmp_path / "output", tmp_path / "work", augment_audio=True)
