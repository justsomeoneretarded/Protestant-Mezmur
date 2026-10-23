import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mezmur import pipeline
from mezmur.common import write_json


@pytest.fixture
def setup_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("TELEGRAM_CHANNEL", "@test")
    write_json(Path(".runtime/snapshot.json"), {"head": None, "state": {"version": 1, "channels": {}}})

    class Client:
        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def is_user_authorized(self):
            return True

    async def resolve(*args):
        return NS(id=42, broadcast=True)

    async def select(*args):
        return [NS(id=10), NS(id=20)]

    async def download(client, message, target):
        target.write_bytes(b"source")
        return target

    def write_lyrics(folder, cues, title, seconds):
        for name in ["lyrics.txt", "lyrics.srt", "lyrics.ass"]:
            (folder / name).write_text("test lyrics")
        return "test lyrics"

    def transcribe(audio, seconds, folder):
        write_json(folder / "transcription.raw.json", [])
        return [{"word": "test", "start": 1, "end": 2}]

    def render(audio, folder, seconds, height):
        (folder / "video.mp4").write_bytes(b"video")

    monkeypatch.setattr(pipeline, "client", Client)
    monkeypatch.setattr(pipeline, "resolve_channel", resolve)
    monkeypatch.setattr(pipeline, "select", select)
    monkeypatch.setattr(pipeline, "download", download)
    monkeypatch.setattr(pipeline, "song_info", lambda m: {"title": "test", "artist": "", "message_id": m.id})
    monkeypatch.setattr(pipeline, "duration", lambda path: 3)
    monkeypatch.setattr(pipeline, "transcribe", transcribe)
    monkeypatch.setattr(pipeline, "write_lyrics", write_lyrics)
    monkeypatch.setattr(pipeline, "render", render)
    return argparse.Namespace(batch_size=3, max_duration=1800, separate_vocals=False, height=720)


def test_success_creates_candidate_but_does_not_commit_progress(setup_pipeline):
    asyncio.run(pipeline.process(setup_pipeline))
    assert json.loads(Path(".runtime/pending.json").read_text())["channels"]["42"]["last_message_id"] == 20
    assert json.loads(Path(".runtime/snapshot.json").read_text())["state"]["channels"] == {}
    files = {path.name for path in Path("output/song-10").iterdir()}
    assert files == {"lyrics.txt", "lyrics.srt", "lyrics.ass", "words.json", "video.mp4",
                     "description.txt", "transcription.raw.json", "metadata.json"}
    assert "Keywords:" in Path("output/song-10/description.txt").read_text()
    assert not list(Path(".runtime").glob("song-*"))


def test_failure_keeps_completed_output_and_no_candidate(setup_pipeline, monkeypatch):
    write_json(Path(".runtime/pending.json"), {"stale": True})
    original = pipeline.render

    def fail_on_second(audio, folder, seconds, height):
        if folder.name == "song-20":
            raise RuntimeError("Render failed")
        original(audio, folder, seconds, height)

    monkeypatch.setattr(pipeline, "render", fail_on_second)
    with pytest.raises(RuntimeError, match="Render failed"):
        asyncio.run(pipeline.process(setup_pipeline))
    assert Path("output/song-10/video.mp4").exists()
    assert not Path(".runtime/pending.json").exists()
