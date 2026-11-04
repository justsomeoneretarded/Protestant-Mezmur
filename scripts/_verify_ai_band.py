"""Read-only native verification of the production AI backing path on a real song excerpt."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from mezmur import arrangement
from mezmur.common import write_json
from mezmur.subtitles import build_cues, write_ass, write_srt

out, work = Path("verification/production-ai-band"), Path(".runtime/production-ai-band")
out.mkdir(parents=True, exist_ok=True)
work.mkdir(parents=True, exist_ok=True)
source, reference = out / "comparison-original.wav", work / "original-instrumental.wav"
shutil.copy2(next(Path("probe-source").rglob("comparison-original.wav")), source)
shutil.copy2(next(Path("probe-source").rglob("reference.wav")), reference)
audio, sr = sf.read(source, dtype="float32", always_2d=True)
music, music_sr = sf.read(reference, dtype="float32", always_2d=True)
assert sr == music_sr and audio.shape == music.shape
voice = work / "original-vocal-complement.wav"
sf.write(voice, audio - music, sr, subtype="FLOAT")
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, voice, reference)}
# Cue preservation is verified separately from recognition; this test does not claim these are song lyrics.
cues = build_cues([{"word": "\u1203\u120c\u1209\u12eb", "start": 3.0, "end": 4.2},
                   {"word": "\u12a0\u121c\u1295", "start": 5.0, "end": 6.2}])
captions = out / "subtitle.srt"
write_srt(cues, captions, display_timing=True, duration=len(audio) / sr, amharic_only=True)
caption_hash = hashlib.sha256(captions.read_bytes()).hexdigest()
started = time.monotonic()
master = out / "comparison-production-ai.wav"
report = arrangement.rebuild(source, voice, master, work / "arrangement", seed=40,
                             instrumental=reference, cues=cues, engine="ace-step")
assert sf.info(master).frames == len(audio)
assert report["duration_locked"] and report["vocal_offset_samples"] == report["vocal_pitch_shift_semitones"] == 0
assert report["vocal_tempo_factor"] == 1 and report["true_peak_dbfs"] <= -0.999
assert report["arrangement_style"] == "source-conditioned-ai-worship-band"
assert report["vocal_waveform_enters_model"] is False and report["generated_vocal_cleanup"]
assert report["original_instrumental_reused"] is False
assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, voice, reference)} == before
assert hashlib.sha256(captions.read_bytes()).hexdigest() == caption_hash
write_json(out / "arrangement.json", report)
for name in ("comparison-original", "comparison-production-ai"):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(out / (name + ".wav")),
                    "-c:a", "libmp3lame", "-b:a", "192k", str(out / (name + ".mp3"))], check=True)
# A native mux check, not a replacement for the normal scenic renderer.
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
                "color=c=black:s=640x360:r=30", "-i", str(master), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-b:a", "192k",
                "-t", str(len(audio) / sr), str(out / "duration-check.mp4")], check=True)
from mezmur.audio import probe
video_seconds = float(probe(out / "duration-check.mp4")["format"]["duration"])
assert abs(video_seconds - len(audio) / sr) <= 0.10
verification = {"sample_count": len(audio), "sample_rate": sr, "source_seconds": len(audio) / sr,
                "video_seconds": video_seconds, "elapsed_seconds": time.monotonic() - started,
                "vocal_source": "original recording minus its UVR instrumental (no AI singer)",
                "caption_hash_unchanged": True, "test_captions_are_fixture_not_song_lyrics": True,
                "youtube_uploaded": False}
write_json(out / "verification.json", verification)
print("AI_PRODUCTION_NATIVE_PASSED " + json.dumps({"report": report, "verification": verification}), flush=True)
