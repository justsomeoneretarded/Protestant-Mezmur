"""Optional CPU-only backing arrangement; vocals keep their original sample timeline."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .acoustic import write_midi
from .audio import read_audio
from .common import duration as audio_duration, write_json
from .harmony import ANALYSIS_SR, ArrangementError, harmony_plan

PEAK_LIMIT = 10 ** (-1.0 / 20)
DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")


def soundfont_path():
    path = Path(os.environ.get("MEZMUR_SOUNDFONT") or DEFAULT_SOUNDFONT)
    if not path.is_file():
        raise ArrangementError("Backing rebuild needs FluidSynth and a SoundFont; install fluid-soundfont-gm")
    return path


def render_instrumental(midi, target, font, sr):
    if not shutil.which("fluidsynth"):
        raise ArrangementError("Install FluidSynth to rebuild backing music")
    subprocess.run(["fluidsynth", "-ni", "-q", "-R", "0", "-C", "0", "-g", "0.5",
                    "-r", str(sr), "-T", "wav", "-O", "float", "-F", str(target),
                    str(font), str(midi)], check=True, timeout=1200)


def true_peak(samples, sr):
    peak = 0.0
    # Overlapped blocks bound memory while checking inter-sample peaks at four times the output rate.
    for start in range(0, len(samples), sr * 4):
        block = samples[max(0, start - 64):min(len(samples), start + sr * 4 + 64)]
        peak = max(peak, float(np.max(np.abs(resample_poly(block, 4, 1, axis=0)))))
    return peak


def mix_tracks(vocals, backing, sr):
    vocals, backing = np.asarray(vocals), np.asarray(backing)
    if (vocals.ndim != 2 or backing.ndim != 2 or vocals.shape != backing.shape or not len(vocals)
            or not np.isfinite(vocals).all() or not np.isfinite(backing).all()):
        raise ArrangementError("Vocal and backing samples must be finite, equally sized audio buffers")
    vrms = float(np.sqrt(np.mean(vocals.astype(float) ** 2)))
    mrms = float(np.sqrt(np.mean(backing.astype(float) ** 2)))
    if min(vrms, mrms) < 1e-6:
        raise ArrangementError("A vocal or synthesized backing track is silent")
    music_gain = min(4.0, (vrms / mrms) * 10 ** (-10 / 20))
    combined = vocals.astype(float) + backing.astype(float) * music_gain
    peak = true_peak(combined, sr)
    common_gain = min(1.0, PEAK_LIMIT / max(peak, 1e-12))
    mixed = (combined * common_gain).astype(np.float32)
    return mixed, {"vocal_gain": common_gain, "backing_gain": music_gain * common_gain,
                   "true_peak_dbfs": round(20 * np.log10(max(peak * common_gain, 1e-12)), 4),
                   "sample_peak_dbfs": round(20 * np.log10(max(float(np.max(np.abs(mixed))), 1e-12)), 4),
                   "vocal_offset_samples": 0, "vocal_pitch_shift_semitones": 0, "vocal_tempo_factor": 1.0}


def rebuild(source, vocals, target, work, seed=0, instrumental=None, cues=()):
    source, vocals, target, work = map(Path, (source, vocals, target, work))
    if target.resolve() in (source.resolve(), vocals.resolve()):
        raise ArrangementError("Rebuilt audio must not overwrite the source recording or vocal stem")
    work.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    font = soundfont_path()
    with source.open("rb") as handle:
        before = hashlib.file_digest(handle, "sha256").hexdigest()
    stem_format = sf.info(vocals).subtype
    if stem_format not in ("FLOAT", "DOUBLE", "PCM_24", "PCM_32"):
        raise ArrangementError("Backing rebuild needs a high-precision vocal stem; update the separator and use float WAV export")
    voice, sr = sf.read(vocals, dtype="float32", always_2d=True)
    if sr < 16000 or voice.shape[1] not in (1, 2) or not np.isfinite(voice).all():
        raise ArrangementError("The separated vocal stem has an unsupported format or invalid samples")
    expected = int(round(audio_duration(source) * sr))
    if expected < sr * 4 or abs(len(voice) - expected) > sr * 0.15:
        raise ArrangementError("Vocal duration does not match the original; refusing to move lyric timings")
    voice = np.pad(voice[:expected], ((0, max(0, expected - len(voice))), (0, 0)))
    if voice.shape[1] == 1:
        voice = np.repeat(voice, 2, axis=1)
    if instrumental is None:
        raise ArrangementError("A separated instrumental reference is required for hybrid rebuilding")
    reference_path = Path(instrumental)
    if reference_path.resolve() in (source.resolve(), vocals.resolve(), target.resolve()):
        raise ArrangementError("Harmony analysis needs a separate instrumental stem, not the original mix or vocals")
    reference_info = sf.info(reference_path)
    if reference_info.subtype not in ("FLOAT", "DOUBLE", "PCM_24", "PCM_32"):
        raise ArrangementError("Hybrid rebuilding needs a high-precision instrumental stem")
    synthesized = work / "instrumental.wav"
    if synthesized.resolve() in {p.resolve() for p in (source, vocals, reference_path, target)}:
        raise ArrangementError("Synthesis output must not overwrite an input stem or the final master")
    reference = read_audio(reference_path, ANALYSIS_SR)
    melody = read_audio(vocals, ANALYSIS_SR)
    plan, report = harmony_plan(reference, vocals=melody)
    midi = target.parent / "arrangement.mid"
    performance = write_midi(plan, midi, seed=seed, tuning_cents=report["tuning_cents"],
                             duration=expected / sr, cues=cues)
    sections = performance.pop("segments")
    write_json(target.parent / "arrangement-plan.json", {
        "analysis_sources": {"rhythm": "separated instrumental", "harmony": "separated instrumental",
                             "melody": "separated vocals"}, "segments": sections,
        "report": {**report, **performance}})
    render_instrumental(midi, synthesized, font, sr)
    backing, backing_sr = sf.read(synthesized, dtype="float32", always_2d=True)
    if backing_sr != sr or backing.shape[1] != 2 or not np.isfinite(backing).all():
        raise ArrangementError("The synthesizer returned an unexpected audio format")
    rendered_samples = len(backing)
    if rendered_samples == 0:
        raise ArrangementError("The synthesized backing is empty")
    backing = np.pad(backing[:expected], ((0, max(0, expected - len(backing))), (0, 0)))
    fade = min(sr * 2, expected // 2)
    backing[:fade] *= np.linspace(0, 1, fade)[:, None]
    backing[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mixed, levels = mix_tracks(voice, backing, sr)
    if len(mixed) != expected:
        raise ArrangementError("Rebuilt audio failed the original-duration lock")
    sf.write(target, mixed, sr, subtype="FLOAT")
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != before:
            raise ArrangementError("The original source changed while rebuilding music")
    license_file = Path("/usr/share/doc/fluid-soundfont-gm/copyright")
    if font == DEFAULT_SOUNDFONT and license_file.is_file():
        shutil.copy2(license_file, target.parent / "arrangement-license.txt")
    return {**report, **performance, **levels, "mode": "rebuilt-accompaniment", "experimental": True,
            "sample_rate": sr, "vocal_stem_format": stem_format, "samples": expected, "duration_seconds": expected / sr,
            "soundfont": font.name, "master_format": "32-bit float WAV",
            "original_recording_unchanged": True, "original_instrumental_reused": False,
            "analysis_sources": {"rhythm": "separated instrumental", "harmony": "separated instrumental",
                                 "melody": "separated vocals"},
            "analysis_instrumental": reference_path.name,
            "duration_locked": True, "source_duration_seconds": expected / sr,
            "trimmed_backing_samples": max(0, rendered_samples - expected),
            "padded_backing_samples": max(0, expected - rendered_samples),
            "warnings": ["Vocal separation can leave artifacts or instrumental bleed.",
                         "Automatic harmony is an estimate and requires listening review.",
                         "A rebuilt arrangement does not guarantee a different acoustic fingerprint or grant music rights."]}
