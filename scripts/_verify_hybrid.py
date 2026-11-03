"""Read-only native regression on the recording that failed the previous arranger."""
import ast
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import srt

from mezmur import arrangement, separate
from mezmur.audio import read_audio
from mezmur.common import write_json

out, work = Path("verification/song-40"), Path(".runtime/hybrid-40")
out.mkdir(parents=True, exist_ok=True)
work.mkdir(parents=True, exist_ok=True)
source = next(Path("inputs/song-40").glob("audio.*"))
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
timings = {}
start = time.monotonic()
voice = separate.separate(source, work, require_vocals=True)
reference = separate.instrumental_stem(voice, work / "instrumental-reference.wav")
voice_hash = hashlib.sha256(voice.read_bytes()).hexdigest()
timings["separation"] = time.monotonic() - start
print("PAIRED_STEMS_READY", voice, reference, flush=True)

# Re-run the exact legacy harmony function on the full mix for an honest comparison.
base = subprocess.run(["git", "show", "a1a10f4e175e36ccd084de657007530474e65ec2:mezmur/arrangement.py"],
                       check=True, capture_output=True, text=True).stdout
function = next(node for node in ast.parse(base).body if isinstance(node, ast.FunctionDef) and node.name == "harmony_plan")
legacy = {"np": np, "ANALYSIS_SR": 22050, "HOP": 512, "ArrangementError": arrangement.ArrangementError}
exec(compile(ast.Module(body=[function], type_ignores=[]), "legacy_harmony", "exec"), legacy)
try:
    _, old_report = legacy["harmony_plan"](read_audio(source, 22050))
    baseline = {"accepted": True, "report": old_report}
except arrangement.ArrangementError as error:
    baseline = {"accepted": False, "reason": str(error)}
print("LEGACY_BASELINE", json.dumps(baseline), flush=True)
write_json(out / "legacy-baseline.json", baseline)

for name in ("captions", "lyrics.txt", "lyrics.romanized.txt"):
    old = Path("old-kit") / name
    if old.is_dir():
        shutil.copytree(old, out / name, dirs_exist_ok=True)
    elif old.is_file():
        shutil.copy2(old, out / name)
caption_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "captions").glob("*.srt")}

start = time.monotonic()
master = out / "rebuilt-audio.wav"
report = arrangement.rebuild(source, voice, master, work / "synthesis", instrumental=reference, seed=40)
timings["hybrid_rebuild"] = time.monotonic() - start
write_json(out / "arrangement.json", report)
info = sf.info(master)
assert info.subtype == "FLOAT" and info.channels == 2
assert info.frames == report["samples"] and np.isfinite(sf.read(master)[0]).all()
assert report["method"] == "instrumental-led-vocal-constrained"
assert report["vocal_offset_samples"] == report["vocal_pitch_shift_semitones"] == 0
assert report["vocal_tempo_factor"] == 1 and report["true_peak_dbfs"] <= -0.999
assert report["vocal_analysis_ran"] and report["vocal_conflicts_checked"], report
assert report["vocal_melody_stable_frames"] > 0
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
assert hashlib.sha256(voice.read_bytes()).hexdigest() == voice_hash
assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "captions").glob("*.srt")} == caption_hashes

cues = list(srt.parse((out / "captions/am.srt").read_text(encoding="utf-8")))
offset = max(0, cues[0].start.total_seconds() - 3) if cues else 30
for name, audio in (("comparison-original", source), ("comparison-rebuilt", master),
                    ("comparison-vocals", voice), ("comparison-new-instrumental", work / "synthesis/instrumental.wav")):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(offset), "-i", str(audio),
                    "-t", "45", "-vn", "-c:a", "libmp3lame", "-b:a", "192k", str(out / f"{name}.mp3")], check=True)
write_json(out / "verification.json", {"song": 40, "source_sha256": source_hash, "timings_seconds": timings,
           "caption_sha256": caption_hashes, "preview_start_seconds": offset,
           "original_vocal_waveform_unchanged": True, "youtube_uploaded": False})
print("HYBRID_NATIVE_PASSED", json.dumps({"report": report, "timings": timings}), flush=True)
