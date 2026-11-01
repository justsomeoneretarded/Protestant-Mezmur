"""Final audio is the source recording, not the transcription stem or a processed mix."""
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mezmur import audio, visuals
from mezmur.common import duration
from mezmur.subtitles import write_ass


class TinyRenderer:
    W, H, fps = 160, 90, 5

    def __init__(self, seconds):
        self.duration = seconds

    def frame(self, *args):
        return np.full((self.H, self.W, 3), 80, dtype=np.uint8)


def tone(path):
    sr = 44100
    values = (0.18 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    sf.write(path, values, sr, subtype="PCM_16")
    return sr


def packets(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_packets",
        "-show_data_hash", "sha256", "-show_entries", "packet=data_hash", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    return [packet["data_hash"] for packet in json.loads(result.stdout)["packets"]]


@pytest.mark.parametrize("suffix,codec", [("m4a", "aac"), ("mp3", "libmp3lame")])
def test_native_render_preserves_original_compressed_audio_packets(tmp_path, suffix, codec):
    wave, source = tmp_path / "tone.wav", tmp_path / f"original.{suffix}"
    tone(wave)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wave), "-c:a", codec,
                    "-b:a", "192k", str(source)], check=True)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    ass, out = tmp_path / "lyrics.ass", tmp_path / "video.mp4"
    write_ass([], ass)
    visuals.render_video(TinyRenderer(duration(source)), source, ass, out, preset="ultrafast")
    original, copied = packets(source), packets(out)
    assert copied and len(copied) >= len(original) - 2
    assert copied == original[:len(copied)]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert audio.mp4_audio_options(source) == ["-c:a", "copy"]


def test_pcm_fallback_only_encodes_aac_without_speed_pitch_or_gain_changes(tmp_path):
    source, out, ass = tmp_path / "original.wav", tmp_path / "video.mp4", tmp_path / "lyrics.ass"
    sr = tone(source)
    write_ass([], ass)
    visuals.render_video(TinyRenderer(duration(source)), source, ass, out, preset="ultrafast")
    original, final = audio.read_audio(source, sr), audio.read_audio(out, sr)
    a, b = original[sr // 2:sr * 3 // 2], final[sr // 2:sr * 3 // 2]
    assert len(a) == len(b)
    assert 0.98 < np.sqrt(np.mean(b ** 2) / np.mean(a ** 2)) < 1.02
    assert np.corrcoef(a, b)[0, 1] > 0.98
    frequency = np.fft.rfftfreq(len(b), 1 / sr)[np.argmax(np.abs(np.fft.rfft(b)))]
    assert abs(frequency - 220) <= 1
    stream = next(s for s in audio.probe(out)["streams"] if s["codec_type"] == "audio")
    assert stream["codec_name"] == "aac" and int(stream["sample_rate"]) == sr
    assert abs(float(stream["duration"]) - 2) < 0.06


def test_render_command_has_no_audio_filters_resampling_or_channel_remix(monkeypatch, tmp_path):
    captured = []

    class Process:
        def __init__(self):
            self.stdin = io.BytesIO()

        def wait(self):
            return 0

    monkeypatch.setattr(visuals.subprocess, "Popen", lambda command, **kwargs: captured.append(command) or Process())
    monkeypatch.setattr(visuals, "mp4_audio_options", lambda _: ["-c:a", "copy"])
    source = tmp_path / "original.mp3"
    visuals.render_video(TinyRenderer(0.2), source, tmp_path / "lyrics.ass", tmp_path / "video.mp4")
    command = captured[0]
    assert str(source) in command
    assert not any(option in command for option in ("-af", "-filter:a", "-ar", "-ac"))
    assert command[command.index("-c:a") + 1] == "copy"


@pytest.mark.parametrize("codec,copy", [("aac", True), ("mp3", True), ("flac", False), ("vorbis", False)])
def test_audio_codec_policy_does_not_add_effects(monkeypatch, codec, copy):
    monkeypatch.setattr(audio, "probe", lambda _: {"streams": [
        {"codec_type": "video", "codec_name": "mjpeg"}, {"codec_type": "audio", "codec_name": codec}]})
    options = audio.mp4_audio_options("source")
    assert options == (["-c:a", "copy"] if copy else ["-c:a", "aac", "-b:a", "320k"])


def test_missing_audio_stream_fails_explicitly(monkeypatch):
    monkeypatch.setattr(audio, "probe", lambda _: {"streams": []})
    with pytest.raises(ValueError, match="no audio"):
        audio.mp4_audio_options("source")


def test_workflows_and_public_api_cannot_enable_sound_effects():
    doc = yaml.safe_load(Path(".github/workflows/generate.yml").read_text())
    events = doc.get("on", doc.get(True))
    inputs = events["workflow_dispatch"]["inputs"]
    assert not {"augment_audio", "augmentation_style", "noise_profile", "space_effect"} & inputs.keys()
    assert "Rubber Band" not in Path(".github/workflows/generate.yml").read_text()
    assert not hasattr(audio, "prepare_source") and not hasattr(audio, "master")
    assert not Path("mezmur/augmentation.py").exists()
