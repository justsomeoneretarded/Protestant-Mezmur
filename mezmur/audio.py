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
    """Optionally augment a decoded, peak-safe source for final rendering only."""
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


def retime_words(words, metadata, source_duration, output_duration):
    """Copy clean-audio word timings onto the processed track's planned time map.

    Anchors describe the DSP's intended timing, not sample-accurate alignment.
    Reject inconsistent maps rather than silently putting lyrics on the wrong clock.
    """
    try:
        sr = float(metadata["sample_rate"])
        source_duration, output_duration = float(source_duration), float(output_duration)
        src = np.asarray(metadata["input_frames"], dtype=np.float64)
        dst = np.asarray(metadata["output_frames"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Augmentation timing metadata is missing or invalid") from error
    if not np.isfinite([sr, source_duration, output_duration]).all() or min(sr, source_duration, output_duration) <= 0:
        raise ValueError("Timing sample rate and durations must be finite and positive")
    if (src.ndim != 1 or dst.ndim != 1 or len(src) < 2 or src.shape != dst.shape
            or not np.isfinite(src).all() or not np.isfinite(dst).all()
            or src[0] != 0 or dst[0] != 0 or np.any(np.diff(src) <= 0) or np.any(np.diff(dst) <= 0)):
        raise ValueError("Augmentation timing anchors must be finite, increasing and start at zero")
    src, dst = src / sr, dst / sr
    # Decoding/resampling may round the last sample, but must not hide a wrong or truncated track.
    tolerance = max(0.02, 2 / sr)
    if abs(src[-1] - source_duration) > tolerance or abs(dst[-1] - output_duration) > tolerance:
        raise ValueError("Augmentation timing map does not match the source and rendered durations")
    result = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if not np.isfinite([start, end]).all() or start < 0 or start >= source_duration or end <= start:
            raise ValueError("Word timings must be finite, ordered and within the original recording")
        # Smoothing can extend the last word slightly beyond the recording; cap it at the song end.
        mapped = np.interp([start, min(end, source_duration)], src, dst)
        mapped = np.clip(mapped, 0, output_duration)
        if mapped[1] <= mapped[0]:
            raise ValueError("Augmentation timing map collapsed a word's duration")
        result.append({**word, "start": float(mapped[0]), "end": float(mapped[1])})
    return result


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
