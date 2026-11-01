"""Read audio and metadata without modifying the original recording."""
import json
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


def mp4_audio_options(source):
    """Copy compatible audio packets; otherwise encode AAC without applying sound effects."""
    streams = [stream for stream in probe(source).get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        raise ValueError("The source has no audio stream")
    if streams[0].get("codec_name") in ("aac", "mp3"):
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "320k"]
