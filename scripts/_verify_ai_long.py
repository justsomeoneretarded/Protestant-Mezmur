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
from mezmur.separate import separate, instrumental_stem
source = out / "comparison-original.wav"
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i",
                str(next(Path("old-kit").rglob("comparison-original.mp3"))), "-t", "44.913",
                "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le", str(source)], check=True)
separation_work = work / "original-separation"
separation_work.mkdir(parents=True, exist_ok=True)
voice = separate(source, separation_work, require_vocals=True)
reference = instrumental_stem(voice, work / "original-instrumental.wav")
audio, sr = sf.read(source, dtype="float32", always_2d=True)
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, voice, reference)}
import srt
old_captions = list(srt.parse(next(Path("old-kit").rglob("captions/am.srt")).read_text(encoding="utf-8")))
offset = max(0, old_captions[0].start.total_seconds() - 3) if old_captions else 30
cues = [{"start": max(0, cue.start.total_seconds() - offset),
         "end": min(len(audio) / sr, cue.end.total_seconds() - offset), "text": cue.content}
        for cue in old_captions if cue.end.total_seconds() > offset
        and cue.start.total_seconds() < offset + len(audio) / sr]
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
# Reuse real earlier video frames, replacing its old audio with the new master for the same excerpt.
old_video = next(Path("old-kit").rglob("acoustic-timing-check.mp4"))
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(offset), "-i", str(old_video),
                "-i", str(master), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "26", "-c:a", "aac", "-b:a", "192k",
                "-t", str(len(audio) / sr), str(out / "duration-check.mp4")], check=True)
from mezmur.audio import probe
video_seconds = float(probe(out / "duration-check.mp4")["format"]["duration"])
assert abs(video_seconds - len(audio) / sr) <= 0.10
verification = {"sample_count": len(audio), "sample_rate": sr, "source_seconds": len(audio) / sr,
                "video_seconds": video_seconds, "elapsed_seconds": time.monotonic() - started,
                "vocal_source": "actual float UVR vocal stem from the original recording (no AI singer)",
                "caption_hash_unchanged": True, "listening_excerpt_offset_seconds": offset,
                "video_uses_earlier_frames": True,
                "youtube_uploaded": False}
write_json(out / "verification.json", verification)
print("AI_PRODUCTION_NATIVE_PASSED " + json.dumps({"report": report, "verification": verification}), flush=True)
