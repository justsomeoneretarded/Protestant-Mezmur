"""Read-only, real-song technical replay; never publishes or modifies song state."""
import base64
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from mezmur import arrangement, thumbnail
from mezmur.common import write_json
from mezmur.separate import separate

inp = Path("replay-inputs/song-30")
source = next(p for p in inp.glob("audio.*"))
kit = Path("replay-kit")
metadata = json.loads((kit / "metadata.json").read_text())
out = Path(".runtime/arrangement-preview")
work = Path(".runtime/arrangement-replay-work")
out.mkdir(parents=True, exist_ok=True)
work.mkdir(parents=True, exist_ok=True)
clip = work / "original-clip.wav"
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", "15", "-i", str(source),
                "-t", "40", "-ac", "2", "-ar", "44100", str(clip)], check=True)
vocals = separate(clip, work, require_vocals=True)
target = out / "rebuilt-audio.wav"
report = arrangement.rebuild(clip, vocals, target, work / "new-backing", seed=30)
portrait = None
if metadata.get("artist_photo"):
    old = cv2.imread(str(kit / "thumbnail-b.jpg"))
    if old is not None:
        portrait = old[132:352, 530:750].copy()
background = cv2.imread("mezmur/assets/scenery.png")
background = background[:background.shape[0] // 2, :background.shape[1] // 2].copy()
names = metadata["names"]
for variant in ("b", "d"):
    thumbnail.save(thumbnail.compose(names, background=background, photo=portrait, variant=variant),
                   out / f"thumbnail-{variant}.jpg")
movie = out / "preview.mp4"
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1",
                "-i", str(out / "thumbnail-d.jpg"), "-i", str(target), "-t", "40", "-r", "10",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", str(movie)], check=True)
result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(movie), "-vn",
                         "-f", "f32le", "pipe:1"], check=True, capture_output=True)
decoded = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)
master, sr = sf.read(target, dtype="float32", always_2d=True)
length = min(len(master), len(decoded))
correlation = float(np.corrcoef(master[:length].ravel(), decoded[:length].ravel())[0, 1])
assert abs(length / sr - 40) < 0.06
assert np.isfinite(decoded).all() and np.max(np.abs(decoded)) < 1
assert correlation > 0.97
report.update(real_audio_message_id=30, source_clip_start_seconds=15,
              portrait_available=portrait is not None,
              encoded_sample_peak=float(np.max(np.abs(decoded))), encoded_master_correlation=correlation,
              preview_is_not_a_full_lyric_video=True, publishing_performed=False)
write_json(out / "arrangement.json", report)
print("ARRANGEMENT_REPLAY_RESULT=" + json.dumps(report, ensure_ascii=True))
for variant in ("b", "d"):
    print(f"THUMB_{variant.upper()}_PREVIEW_BASE64=" +
          base64.b64encode((out / f"thumbnail-{variant}.jpg").read_bytes()).decode("ascii"))
