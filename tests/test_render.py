import json
import shutil
import subprocess

import pytest
from PIL import Image, ImageChops, ImageStat

from mezmur.common import duration, run
from mezmur.lyrics import make_cues, write_lyrics
from mezmur.render import render


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")
@pytest.mark.parametrize("height", [720, 1080])
def test_video_has_audio_motion_and_ethiopic_lyrics(tmp_path, height):
    audio = tmp_path / "tone.wav"
    run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
         "-c:a", "pcm_s16le", str(audio)])
    words = [{"word": "ሃሌሉያ", "start": 0.4, "end": 1.2},
             {"word": "አሜን", "start": 1.3, "end": 2.4}]
    write_lyrics(tmp_path, make_cues(words), "የአምልኮ መዝሙር", 3)
    render(audio, tmp_path, 3, height)
    assert abs(duration(tmp_path / "video.mp4") - 3) < 0.2
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                                     str(tmp_path / "video.mp4")], text=True)
    streams = json.loads(probe)["streams"]
    assert {s["codec_type"] for s in streams} == {"audio", "video"}
    video = next(s for s in streams if s["codec_type"] == "video")
    width = height * 16 // 9
    assert (video["width"], video["height"]) == (width, height)
    images = []
    for index, time in enumerate([0.2, 1.5]):
        frame = tmp_path / f"frame-{index}.png"
        run(["ffmpeg", "-v", "error", "-ss", str(time), "-i", str(tmp_path / "video.mp4"),
             "-frames:v", "1", str(frame)])
        images.append(Image.open(frame).convert("RGB"))
    assert sum(ImageStat.Stat(ImageChops.difference(*images)).mean) > 2
    # Lyric pixels are bright in the center only while the words are being sung.
    box = (width // 12, height * 2 // 5, width * 11 // 12, height * 3 // 5)
    before = images[0].crop(box).convert("L")
    during = images[1].crop(box).convert("L")
    assert sum(p > 150 for p in during.getdata()) > sum(p > 150 for p in before.getdata()) + 50
    wave = images[1].crop((0, height * 3 // 4, width, height * 11 // 12))
    assert sum(g > 90 and g > r * 1.15 for r, g, b in wave.getdata()) > 100
