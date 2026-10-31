"""Audio I/O, metadata, cover art and loudness mastering (all via FFmpeg)."""
import json
import re
import subprocess
import tempfile
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


def prepare_source(source, work, *, augment=False, seed=None, style="subtle",
                   noise_profile="hiss", space_effect="none"):
    """Optionally augment a decoded, peak-safe source for both rendering and ASR."""
    source = Path(source)
    if not augment:
        return source, {"enabled": False}
    if style not in ("subtle", "musical"):
        raise ValueError("Unknown augmentation style")
    if noise_profile not in ("off", "hiss", "vinyl", "ambient") or space_effect not in ("none", "room", "echo"):
        raise ValueError("Unknown noise profile or space effect")
    from .augmentation import augment_file

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="augmentation-", dir=work))
    # FFmpeg accepts all Telegram audio formats, including M4A. Pre-mastering
    # also prevents decoder overshoots from entering the normalized-float API.
    decoded = folder / "input.wav"
    before = master(source, decoded)
    target = folder / "augmented.wav"
    settings = {"seed": seed, "noise_snr_db": None if noise_profile == "off" else 50.0,
                "noise_profile": noise_profile,
                "space_effect": space_effect}
    if style == "musical":
        rng = np.random.default_rng(seed)
        settings.update({"pitch_semitones": float(rng.choice([-2, 2])),
                         "tempo_factor": float(rng.choice([0.96, 1.04])),
                         "max_speed_change": 0.01, "max_formant_semitones": 0.15,
                         "preserve_duration": False})
    metadata = augment_file(decoded, target, **settings)
    metadata.update({"enabled": True, "style": style, "loudness_before_augmentation": before})
    return target, metadata


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
