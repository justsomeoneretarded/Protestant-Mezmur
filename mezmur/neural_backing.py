"""Source-conditioned local AI instrumental generation; original vocals never enter the model."""
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .common import write_json
from .harmony import ANALYSIS_SR, ArrangementError, ROUGH_INTERVALS

MODEL_NAME = "acestep-v15-turbo-Q8_0"
ENGINE_COMMIT = "de858d8e1a94f0e1939cf9bdd8ff7d9631953077"
MODEL_FILES = ("Qwen3-Embedding-0.6B-Q8_0.gguf", MODEL_NAME + ".gguf", "vae-BF16.gguf")
DEFAULT_STYLE = (
    "Instrumental Ethiopian gospel worship band. Expressive melodic bass guitar, soulful piano and warm electric "
    "piano, restrained live drums with tasteful phrase-end fills, spacious pads, occasional electric guitar answers. "
    "Follow the source harmony, groove, phrasing and section dynamics. Sparse verses, wider choruses, breathing "
    "instrumental breaks. No vocals, humming or choir. No continuous acoustic strumming, rigid looping or lead doubling."
)


def engine_root():
    root = Path(os.environ.get("MEZMUR_AI_ROOT", ".ai-backing")).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ArrangementError("AI backing is not installed; run scripts/setup_ai_backing.py") from None
    if manifest.get("engine_commit") != ENGINE_COMMIT:
        raise ArrangementError("AI engine version differs from the tested build; rerun scripts/setup_ai_backing.py")
    if not (root / "source/build/ace-synth").is_file() or any(not (root / "models" / name).is_file() for name in MODEL_FILES):
        raise ArrangementError("AI backing installation is incomplete; rerun scripts/setup_ai_backing.py")
    return root, manifest


def chunk_windows(duration, cues=(), maximum=72.0, overlap=4.0):
    """Overlap on the original clock, never concatenate extra bars or restart every chord."""
    if not all(math.isfinite(value) for value in (duration, maximum, overlap)) or duration <= 0 or maximum < 20 or not 0 < overlap < maximum / 3:
        raise ArrangementError("Invalid AI arrangement duration/chunk settings")
    ends = []
    for cue in cues or ():
        try:
            point = float(cue["end"]) + 0.25
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(point) and 0 < point < duration:
            ends.append(point)
    result, start = [], 0.0
    while start < duration:
        end = min(start + maximum, duration)
        if end < duration and duration - end < 10:
            end = duration
        elif end < duration:
            phrase_ends = [point for point in ends if end - 6 <= point <= end]
            if phrase_ends:
                end = max(phrase_ends)
        result.append((start, end))
        if end >= duration:
            break
        start = end - overlap
    return result


def _mono(audio, sr):
    y = np.asarray(audio, dtype=np.float32).mean(axis=1)
    if sr != ANALYSIS_SR:
        divisor = math.gcd(int(sr), ANALYSIS_SR)
        y = resample_poly(y, ANALYSIS_SR // divisor, int(sr) // divisor)
    return y


def assess_candidate(audio, reference, sr, plan=(), offset=0.0):
    """Diagnostics/ranking, not a claimed musical-quality or note-accuracy percentage."""
    import librosa

    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all() or len(audio) < sr * 4:
        raise ArrangementError("AI returned invalid or very short stereo audio")
    if float(np.sqrt(np.mean(audio.astype(float) ** 2))) < 1e-5:
        raise ArrangementError("AI returned silent accompaniment")
    y, ref = _mono(audio, sr), _mono(reference, sr)
    hop = 1024
    generated = librosa.feature.chroma_stft(y=y, sr=ANALYSIS_SR, n_fft=4096, hop_length=hop, norm=2)
    original = librosa.feature.chroma_stft(y=ref, sr=ANALYSIS_SR, n_fft=4096, hop_length=hop, norm=2)
    n = min(generated.shape[1], original.shape[1])
    generated, original = generated[:, :n], original[:, :n]
    levels = librosa.feature.rms(y=ref, hop_length=hop, frame_length=4096)[0][:n]
    active = levels > max(1e-5, float(np.percentile(levels, 90)) * 0.06)
    if active.sum() < 4:
        raise ArrangementError("The instrumental reference has too little usable music")

    def similarity(shift):
        a, b = max(0, shift), n + min(0, shift)
        x, z = generated[:, a:b], original[:, a - shift:b - shift]
        weight = active[a - shift:b - shift]
        if not weight.any():
            return 0.0
        cosine = np.sum(x * z, axis=0) / np.maximum(np.linalg.norm(x, axis=0) * np.linalg.norm(z, axis=0), 1e-9)
        return float(np.mean(cosine[weight]))

    zero = similarity(0)
    shifts = range(-min(40, n // 4), min(40, n // 4) + 1)
    best = max(shifts, key=similarity)
    drift = best * hop / ANALYSIS_SR
    improvement = similarity(best) - zero
    # Only reject an identifiable displaced harmonic sequence, not ambiguous repeated chords.
    drift_detected = abs(drift) > 0.45 and improvement > 0.12
    clash, weight = 0.0, 0.0
    for part in plan:
        hist = np.asarray(part.get("vocal_pitch_weights", []), dtype=float)
        if hist.shape != (12,) or hist.sum() <= 0:
            continue
        a = max(0, int((part["start"] - offset) * ANALYSIS_SR / hop))
        b = min(n, int(np.ceil((part["end"] - offset) * ANALYSIS_SR / hop)))
        if b <= a:
            continue
        chroma = generated[:, a:b].mean(axis=1)
        chroma /= max(float(chroma.sum()), 1e-9)
        dissonance = sum(hist[p] * sum(chroma[(p - interval) % 12] for interval in ROUGH_INTERVALS)
                        for p in range(12))
        clash += float(dissonance) * (b - a)
        weight += b - a
    vocal_clash = clash / max(weight, 1)
    return {"source_harmony_affinity": round(zero, 5), "vocal_clash_proxy": round(vocal_clash, 5),
            "harmonic_drift_seconds": round(drift, 4), "drift_improvement": round(improvement, 5),
            "obvious_harmonic_drift": bool(drift_detected),
            "selection_score": round(zero - 0.20 * vocal_clash, 5),
            "score_is_musical_quality": False}


def compose(reference_path, target, work, plan, analysis, sr, duration, cues=(), seed=0, diagnostics=None,
            separator_model=None):
    """Generate a fresh instrumental in coherent phrases, crossfaded at original-time positions."""
    root, manifest = engine_root()
    reference_path, target, work = map(Path, (reference_path, target, work))
    work.mkdir(parents=True, exist_ok=True)
    diagnostics = Path(diagnostics) if diagnostics is not None else target.parent / "ai-backing"
    diagnostics.mkdir(parents=True, exist_ok=True)
    reference, ref_sr = sf.read(reference_path, dtype="float32", always_2d=True)
    if ref_sr != sr:
        divisor = math.gcd(ref_sr, sr)
        reference = resample_poly(reference, sr // divisor, ref_sr // divisor)
    expected = int(round(duration * sr))
    if reference.shape[1] == 1:
        reference = np.repeat(reference, 2, axis=1)
    if reference.shape[1] != 2 or abs(len(reference) - expected) > sr * 0.15 or not np.isfinite(reference).all():
        raise ArrangementError("The AI reference must match the original instrumental timeline")
    reference = np.pad(reference[:expected], ((0, max(0, expected - len(reference))), (0, 0)))
    result, weights = np.zeros((expected, 2), dtype=np.float64), np.zeros(expected, dtype=np.float64)
    windows = chunk_windows(duration, cues)
    records, started = [], time.monotonic()
    style = os.environ.get("MEZMUR_BACKING_STYLE", "").strip() or DEFAULT_STYLE
    if len(style) > 1500:
        raise ArrangementError("MEZMUR_BACKING_STYLE must be at most 1500 characters")
    for index, (start, end) in enumerate(windows):
        a, b = int(round(start * sr)), min(expected, int(round(end * sr)))
        source = reference[a:b]
        source_path = work / f"source-{index:03}.wav"
        # The model's short-clip minimum is handled on the instrumental only, then discarded.
        padded = np.pad(source, ((0, max(0, sr * 10 - len(source))), (0, 0)))
        sf.write(source_path, padded, sr, subtype="FLOAT")
        candidates = []
        for attempt in range(2):
            if time.monotonic() - started > 120 * 60:
                raise ArrangementError("AI composition exceeded its two-hour budget; no MIDI loop fallback was used")
            request = diagnostics / f"chunk-{index:03}-candidate-{attempt}.json"
            params = {"task_type": "cover-nofsq", "lyrics": "[Instrumental]", "caption": style,
                      "bpm": int(round(analysis.get("tempo_bpm", 0))), "duration": len(padded) / sr,
                      "seed": (int(seed) + attempt * 104729) % 2**32, "inference_steps": 8,
                      "synth_batch_size": 1, "audio_cover_strength": 0.80 if attempt == 0 else 1.0,
                      "cover_noise_strength": 0.0, "output_format": "wav32", "peak_clip": 0,
                      "synth_model": MODEL_NAME + ".gguf"}
            write_json(request, params)
            generated = request.with_name(request.stem + "0.wav")
            generated.unlink(missing_ok=True)
            log_path = diagnostics / f"chunk-{index:03}-candidate-{attempt}.log"
            print(f"AI band: phrase {index + 1}/{len(windows)}, candidate {attempt + 1}", flush=True)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run([str(root / "source/build/ace-synth"), "--models", str(root / "models"),
                                    "--device", "CPU", "--src-audio", str(source_path), "--request", str(request)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=45 * 60)
                audio, rate = sf.read(generated, dtype="float32", always_2d=True)
                if abs(len(audio) / rate - len(padded) / sr) > 0.75:
                    raise ArrangementError("AI changed the reference duration; refusing to hide missing musical bars")
                divisor = math.gcd(rate, sr)
                if rate != sr:
                    audio = resample_poly(audio, sr // divisor, rate // divisor)
                audio = np.pad(audio[:b - a], ((0, max(0, b - a - len(audio))), (0, 0)))
                metrics = assess_candidate(audio, source, sr, plan, offset=start)
                if metrics["obvious_harmonic_drift"] or metrics["source_harmony_affinity"] < 0.30:
                    raise ArrangementError("AI accompaniment did not retain enough source harmony/timing")
                candidates.append((metrics["selection_score"], audio, metrics, attempt))
                records.append({"chunk": index, "candidate": attempt, "rejected": False, **metrics})
            except (subprocess.SubprocessError, OSError, ArrangementError) as error:
                records.append({"chunk": index, "candidate": attempt, "rejected": True,
                                "reason": type(error).__name__ + ": " + str(error)})
            finally:
                generated.unlink(missing_ok=True)
        if not candidates:
            write_json(diagnostics / "candidate-report.json", {"candidates": records})
            raise ArrangementError(f"Neither AI candidate passed phrase {index + 1}; inspect ai-backing logs, "
                                   "no original music or repeating MIDI loop was substituted")
        _, selected, metrics, attempt = max(candidates, key=lambda item: item[0])
        envelope = np.ones(b - a)
        if index:
            overlap = min(len(envelope), int(round((windows[index - 1][1] - start) * sr)))
            envelope[:overlap] = np.linspace(0, 1, overlap, endpoint=False)
        if index + 1 < len(windows):
            overlap = min(len(envelope), int(round((end - windows[index + 1][0]) * sr)))
            envelope[-overlap:] = np.linspace(1, 0, overlap, endpoint=False)
        result[a:b] += selected * envelope[:, None]
        weights[a:b] += envelope
        records.append({"chunk": index, "start": start, "end": end, "selected_candidate": attempt, **metrics})
        source_path.unlink(missing_ok=True)
    if np.any(weights <= 0):
        raise ArrangementError("AI phrase stitching left an uncovered point on the original timeline")
    result = (result / weights[:, None]).astype(np.float32)
    if len(result) != expected or not np.isfinite(result).all():
        raise ArrangementError("AI phrase stitching failed the original-duration lock")
    sf.write(target, result, sr, subtype="FLOAT")
    # Remove any synthesized singing/bleed from the new backing only; never reprocess the real singer.
    from .separate import instrumental_stem, separate
    generated_voice = separate(target, work / "vocal-cleanup", model=separator_model, require_vocals=True)
    cleaned = instrumental_stem(generated_voice, work / "cleaned-instrumental.wav")
    clean_audio, clean_sr = sf.read(cleaned, dtype="float32", always_2d=True)
    if clean_sr != sr or clean_audio.shape[1] != 2 or abs(len(clean_audio) - expected) > sr * 0.15:
        raise ArrangementError("AI instrumental cleanup changed the original timeline or sample format")
    clean_audio = np.pad(clean_audio[:expected], ((0, max(0, expected - len(clean_audio))), (0, 0)))
    if not np.isfinite(clean_audio).all() or float(np.sqrt(np.mean(clean_audio ** 2))) < 1e-5:
        raise ArrangementError("AI instrumental cleanup produced invalid/silent music")
    sf.write(target, clean_audio, sr, subtype="FLOAT")
    write_json(diagnostics / "candidate-report.json", {"candidates": records})
    write_json(diagnostics / "model-manifest.json", manifest)
    if (root / "model-license.txt").is_file():
        shutil.copy2(root / "model-license.txt", diagnostics / "model-license.txt")
    return {"arrangement_style": "source-conditioned-ai-worship-band", "engine": "ACE-Step 1.5 / acestep.cpp",
            "engine_commit": manifest["engine_commit"], "model_revision": manifest["model_revision"],
            "model_name": MODEL_NAME, "instruments": ["AI worship band (prompt-guided, not guaranteed isolated stems)"],
            "source_conditioning": "separated original instrumental, raw VAE cover context",
            "vocal_waveform_enters_model": False, "generated_vocal_cleanup": True, "phrase_count": len(windows), "candidates_per_phrase": 2,
            "ai_generation_seconds": round(time.monotonic() - started, 2), "instrumental_prompt": style,
            "candidate_metrics_are_accuracy": False, "no_midi_loop_fallback": True}
