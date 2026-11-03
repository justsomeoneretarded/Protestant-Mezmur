"""Read-only actual-song acoustic rebuild and native full-video timing verification."""
import bisect
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import srt

from mezmur import arrangement, separate
from mezmur.audio import probe
from mezmur.common import duration, write_json
from mezmur.harmony import compatible_notes

out, work = Path("verification/song-40"), Path(".runtime/acoustic-40")
out.mkdir(parents=True, exist_ok=True)
work.mkdir(parents=True, exist_ok=True)
source = next(Path("inputs/song-40").glob("audio.*"))
before = hashlib.sha256(source.read_bytes()).hexdigest()
started = time.monotonic()
voice = separate.separate(source, work, require_vocals=True)
reference = separate.instrumental_stem(voice, work / "instrumental-reference.wav")
voice_hash = hashlib.sha256(voice.read_bytes()).hexdigest()
print("PAIRED_STEMS_READY", flush=True)

shutil.copytree(Path("old-kit/captions"), out / "captions", dirs_exist_ok=True)
captions_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "captions").glob("*.srt")}
for name in ("lyrics.txt", "lyrics.romanized.txt"):
    if (Path("old-kit") / name).exists():
        shutil.copy2(Path("old-kit") / name, out / name)
chapters = json.loads(Path("old-kit/youtube.json").read_text()).get("chapters", [])
cues = []
for cue in srt.parse((out / "captions/am.srt").read_text()):
    start = cue.start.total_seconds()
    chapter = next((c for c in reversed(chapters) if c["start"] <= start), {})
    cues.append({"start": start, "end": cue.end.total_seconds(),
                 "chorus": chapter.get("label", "").startswith("Chorus")})

master = out / "rebuilt-audio.wav"
report = arrangement.rebuild(source, voice, master, work / "synthesis",
                             instrumental=reference, cues=cues, seed=40)
write_json(out / "arrangement.json", report)
info = sf.info(master)
assert info.frames == report["samples"] and report["duration_locked"]
assert report["vocal_offset_samples"] == report["vocal_pitch_shift_semitones"] == 0
assert report["vocal_tempo_factor"] == 1 and report["true_peak_dbfs"] <= -0.999
assert report["arrangement_style"] == "adaptive-guitar-and-bass"
assert hashlib.sha256(source.read_bytes()).hexdigest() == before
assert hashlib.sha256(voice.read_bytes()).hexdigest() == voice_hash
assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "captions").glob("*.srt")} == captions_before

sections = json.loads((out / "arrangement-plan.json").read_text())["segments"]
starts = [p["start"] for p in sections]
midi = pretty_midi.PrettyMIDI(str(out / "arrangement.mid"))
assert midi.instruments[0].program == 25 and any(i.program == 32 for i in midi.instruments)
assert midi.get_end_time() <= report["duration_seconds"] + 0.001
for instrument in midi.instruments:
    for note in instrument.notes:
        assert 0 <= note.start < note.end <= report["duration_seconds"] + 0.001
        if instrument.is_drum:
            continue
        part = sections[max(0, bisect.bisect_right(starts, note.start + 0.0008) - 1)]
        allowed = {(part["root"] + interval) % 12 for interval in part["intervals"]}
        assert note.pitch % 12 in allowed, (instrument.name, note.pitch, part)
        assert compatible_notes([note.pitch % 12], np.asarray(part["vocal_pitch_weights"])), part
assert np.isfinite(sf.read(master)[0]).all()

# Reuse the earlier video's video track only: new master replaces, never layers over, its old audio.
video = out / "acoustic-timing-check.mp4"
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", "old-kit/video.mp4",
                "-i", str(master), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "320k", "-t", f"{report['duration_seconds']:.9f}",
                "-movflags", "+faststart", str(video)], check=True)
old_info, new_info = probe(Path("old-kit/video.mp4")), probe(video)
original_duration = float(old_info["format"]["duration"])
actual_duration = float(new_info["format"]["duration"])
assert abs(actual_duration - original_duration) <= 0.10, (actual_duration, original_duration)
assert abs(actual_duration - duration(source)) <= 0.10
assert len([s for s in new_info["streams"] if s["codec_type"] == "audio"]) == 1

offset = max(0, cues[0]["start"] - 3) if cues else 30
for name, audio in (("comparison-original", source), ("comparison-acoustic", master)):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(offset), "-i", str(audio),
                    "-t", "45", "-vn", "-c:a", "libmp3lame", "-b:a", "192k",
                    str(out / f"{name}.mp3")], check=True)
verification = {"original_video_duration": original_duration, "rebuilt_video_duration": actual_duration,
                "caption_sha256": captions_before, "source_sha256": before, "elapsed_seconds": time.monotonic() - started,
                "youtube_uploaded": False, "timing_video_uses_earlier_video_frames": True}
write_json(out / "verification.json", verification)
print("ACOUSTIC_NATIVE_PASSED", json.dumps({"report": report, "verification": verification}), flush=True)
