import json
import shutil
import subprocess

import pytest
from PIL import Image, ImageChops, ImageStat

from mezmur.common import duration, run
from mezmur.lyrics import make_cues, write_lyrics
from mezmur.render import render


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
def test_video_has_audio_motion_and_ethiopic_lyrics(tmp_path):
    audio = tmp_path / "tone.wav"
    run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
         "-c:a", "pcm_s16le", str(audio)])
    words = [{"word": "ሃሌሉያ", "start": 0.4, "end": 1.2},
             {"word": "አሜን", "start": 1.3, "end": 2.4}]
    write_lyrics(tmp_path, make_cues(words), "የአምልኮ መዝሙር", 3)
    render(audio, tmp_path, 3, 720)
    assert abs(duration(tmp_path / "video.mp4") - 3) < 0.2
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                                     str(tmp_path / "video.mp4")], text=True)
    streams = json.loads(probe)["streams"]
    assert {s["codec_type"] for s in streams} == {"audio", "video"}
    video = next(s for s in streams if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1280, 720)
    images = []
    for index, time in enumerate([0.2, 1.5]):
        frame = tmp_path / f"frame-{index}.png"
        run(["ffmpeg", "-v", "error", "-ss", str(time), "-i", str(tmp_path / "video.mp4"),
             "-frames:v", "1", str(frame)])
        images.append(Image.open(frame).convert("RGB"))
    assert sum(ImageStat.Stat(ImageChops.difference(*images)).mean) > 2
    # Lyric pixels are bright in the center only while the words are being sung.
    before = images[0].crop((100, 290, 1180, 430)).convert("L")
    during = images[1].crop((100, 290, 1180, 430)).convert("L")
    assert sum(p > 150 for p in during.getdata()) > sum(p > 150 for p in before.getdata()) + 50
