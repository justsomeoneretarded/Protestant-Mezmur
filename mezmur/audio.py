"""Audio I/O, metadata, cover art and loudness mastering (all via FFmpeg)."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def read_audio(path: Path, sample_rate=16000):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-f", "f32le", "pipe:1"],
        check=True, capture_output=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def tags(path):
    info = probe(path)
    out = {}
    for source in [info.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in info.get("streams", [])]:
        for k, v in (source or {}).items():
            out.setdefault(k.lower(), v)
    return out


def cover_art(path, target):
    """Extract an embedded cover image, if any. Returns the target path or None."""
    info = probe(path)
    if not any(s.get("codec_type") == "video" for s in info.get("streams", [])):
        return None
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-an", "-map", "0:v:0",
                    "-frames:v", "1", str(target)], check=False)
    return target if Path(target).is_file() and Path(target).stat().st_size > 2000 else None


def master(source, target, lufs=-14.0, true_peak=-1.0):
    """Two-pass EBU R128 loudness normalization to YouTube's reference, 48 kHz stereo WAV."""
    measure = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(source), "-vn",
         "-af", f"loudnorm=I={lufs}:TP={true_peak}:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", measure.stderr, re.S)
    if m:
        s = json.loads(m.group(0))
        flt = (f"loudnorm=I={lufs}:TP={true_peak}:LRA=11:measured_I={s['input_i']}:measured_TP={s['input_tp']}:"
               f"measured_LRA={s['input_lra']}:measured_thresh={s['input_thresh']}:offset={s['target_offset']}:"
               "linear=true:print_format=summary")
        stats = {k: float(s[k]) for k in ["input_i", "input_tp", "input_lra"]}
    else:
        flt, stats = f"loudnorm=I={lufs}:TP={true_peak}:LRA=11", {}
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-vn", "-af", flt,
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(target)], check=True)
    return stats
