"""Optional CPU-only backing arrangement; vocals keep their original sample timeline."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import read_audio
from .common import duration as audio_duration

ANALYSIS_SR = 22050
HOP = 512
PEAK_LIMIT = 10 ** (-1.0 / 20)
DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")


class ArrangementError(RuntimeError):
    """The requested arrangement cannot be produced without violating a technical quality check."""


def soundfont_path():
    path = Path(os.environ.get("MEZMUR_SOUNDFONT") or DEFAULT_SOUNDFONT)
    if not path.is_file():
        raise ArrangementError("Backing rebuild needs FluidSynth and a SoundFont; install fluid-soundfont-gm")
    return path


def harmony_plan(samples, sr=ANALYSIS_SR):
    import librosa

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) < sr * 4:
        raise ArrangementError("Backing analysis needs at least four seconds of finite mono audio")
    if np.sqrt(np.mean(samples ** 2)) < 1e-5:
        raise ArrangementError("The recording is silent; cannot infer a backing arrangement")
    harmonic, percussive = librosa.effects.hpss(samples)
    onset = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=HOP,
                                          trim=False, units="time")
    tempo = float(np.asarray(tempo).ravel()[0])
    seconds = len(samples) / sr
    beats = np.asarray(beats, dtype=float)
    beats = beats[(beats > 0) & (beats < seconds)]
    if not np.isfinite(tempo) or not 35 <= tempo <= 240 or len(beats) < 4:
        raise ArrangementError("No usable beat timeline was detected; the original vocals will not be time-stretched")
    bounds = np.unique(np.r_[0.0, beats[::2], seconds])
    chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr, hop_length=HOP, n_fft=4096, norm=2)
    levels = librosa.feature.rms(y=harmonic, hop_length=HOP, frame_length=4096)[0]
    frames = min(chroma.shape[1], len(levels))
    chroma, levels = chroma[:, :frames], levels[:frames]
    observations, active = [], []
    for start, end in zip(bounds, bounds[1:]):
        a = min(frames - 1, int(start * sr / HOP))
        b = max(a + 1, min(frames, int(np.ceil(end * sr / HOP))))
        weights = levels[a:b]
        vector = np.average(chroma[:, a:b], axis=1, weights=weights + 1e-12)
        observations.append(vector / max(np.linalg.norm(vector), 1e-12))
        active.append(float(weights.mean()) > max(1e-5, float(np.percentile(levels, 90)) * 0.04))
    observations = np.asarray(observations).T
    templates = np.zeros((24, 12))
    for state in range(24):
        root, minor = state % 12, state >= 12
        templates[state, [(root + interval) % 12 for interval in (0, 3 if minor else 4, 7)]] = 1 / np.sqrt(3)
    scores = templates @ observations
    # The library decoder discourages implausibly rapid chord switching; weak sections stay unscored.
    probabilities = np.exp(10 * (scores - scores.max(axis=0, keepdims=True)))
    transition = np.full((24, 24), 0.18 / 23)
    np.fill_diagonal(transition, 0.82)
    states = librosa.sequence.viterbi(probabilities, transition)
    chosen = scores[states, np.arange(len(states))]
    active = np.asarray(active, dtype=bool)
    reliable = active & (chosen >= 0.60)
    lengths = np.diff(bounds)
    share = float(lengths[reliable].sum() / max(lengths[active].sum(), 1e-9))
    if not active.any() or share < 0.55:
        raise ArrangementError("Too little stable harmonic evidence for a backing rebuild; use the original-audio mode")
    plan = [{"start": float(start), "end": float(end), "root": int(state % 12),
             "minor": bool(state >= 12), "score": float(score), "reliable": bool(ok)}
            for start, end, state, score, ok in zip(bounds, bounds[1:], states, chosen, reliable)]
    return plan, {"tempo_bpm": round(tempo, 2), "beat_count": len(beats),
                  "reliable_harmony_share": round(share, 4),
                  "harmony_score_is_accuracy": False}


def write_midi(plan, target, seed=0):
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=960)
    piano = pretty_midi.Instrument(program=0, name="New piano arrangement")
    bass = pretty_midi.Instrument(program=32, name="New acoustic bass")
    pad = pretty_midi.Instrument(program=89, name="New warm pad")
    rng = np.random.default_rng(seed)
    for segment in plan:
        start, end = segment["start"], segment["end"]
        if not segment["reliable"] or end - start < 0.12:
            continue
        root = segment["root"]
        intervals = (0, 3 if segment["minor"] else 4, 7) if segment["score"] >= 0.78 else (0, 7)
        pitches = [48 + root + interval for interval in intervals]
        pattern = [0, len(pitches) - 1, 1, len(pitches) - 1]
        if rng.integers(2):
            pattern = pattern[::-1]
        for index, degree in enumerate(pattern):
            t = start + index * (end - start) / 4
            stop = min(end, t + min(0.65, (end - start) * 0.40))
            piano.notes.append(pretty_midi.Note(velocity=int(rng.integers(58, 72)),
                                                pitch=pitches[degree], start=t, end=stop))
        bass.notes.append(pretty_midi.Note(velocity=52, pitch=36 + root, start=start, end=end))
        for pitch in pitches:
            pad.notes.append(pretty_midi.Note(velocity=30, pitch=pitch + 12, start=start, end=end))
    midi.instruments = [piano, bass, pad]
    if not piano.notes:
        raise ArrangementError("No reliable notes were available to synthesize")
    midi.write(str(target))


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


def rebuild(source, vocals, target, work, seed=0):
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
    plan, report = harmony_plan(read_audio(source, ANALYSIS_SR))
    midi = target.parent / "arrangement.mid"
    write_midi(plan, midi, seed=seed)
    instrumental = work / "instrumental.wav"
    render_instrumental(midi, instrumental, font, sr)
    backing, backing_sr = sf.read(instrumental, dtype="float32", always_2d=True)
    if backing_sr != sr or backing.shape[1] != 2 or not np.isfinite(backing).all():
        raise ArrangementError("The synthesizer returned an unexpected audio format")
    backing = np.pad(backing[:expected], ((0, max(0, expected - len(backing))), (0, 0)))
    fade = min(sr * 2, expected // 2)
    backing[:fade] *= np.linspace(0, 1, fade)[:, None]
    backing[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mixed, levels = mix_tracks(voice, backing, sr)
    sf.write(target, mixed, sr, subtype="FLOAT")
    with source.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != before:
            raise ArrangementError("The original source changed while rebuilding music")
    license_file = Path("/usr/share/doc/fluid-soundfont-gm/copyright")
    if font == DEFAULT_SOUNDFONT and license_file.is_file():
        shutil.copy2(license_file, target.parent / "arrangement-license.txt")
    return {**report, **levels, "mode": "rebuilt-accompaniment", "experimental": True,
            "sample_rate": sr, "vocal_stem_format": stem_format, "samples": expected, "duration_seconds": expected / sr,
            "soundfont": font.name, "master_format": "32-bit float WAV",
            "original_recording_unchanged": True, "original_instrumental_reused": False,
            "instruments": ["piano", "acoustic bass", "warm pad"],
            "warnings": ["Vocal separation can leave artifacts or instrumental bleed.",
                         "Automatic harmony is an estimate and requires listening review.",
                         "A rebuilt arrangement does not guarantee a different acoustic fingerprint or grant music rights."]}
