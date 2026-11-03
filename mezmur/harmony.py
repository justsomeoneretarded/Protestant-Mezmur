"""Instrumental-led harmony with conservative checks against a separated vocal melody."""
import numpy as np

ANALYSIS_SR = 22050
HOP = 512
CHORDS = {
    "major": (0, 4, 7), "minor": (0, 3, 7),
    "dominant7": (0, 4, 7, 10), "minor7": (0, 3, 7, 10), "major7": (0, 4, 7, 11),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "fifth": (0, 7),
}
ROUGH_INTERVALS = (1, 6, 11)


class ArrangementError(RuntimeError):
    """A requested arrangement cannot satisfy the technical input/output checks."""


def _samples(values, sr, label):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < sr * 4:
        raise ArrangementError(f"{label} needs at least four seconds of finite mono audio")
    return values


def vocal_melody(samples, sr, tuning=0.0):
    """pYIN estimates analysis-only pitches; the full-resolution vocal waveform is never changed."""
    import librosa

    pitch_sr, pitch_hop = 8000, 256
    y = librosa.resample(samples, orig_sr=sr, target_sr=pitch_sr)
    f0, voiced, probability = librosa.pyin(
        y, sr=pitch_sr, fmin=65.4, fmax=1046.5, frame_length=1024, hop_length=pitch_hop,
        n_thresholds=20, resolution=0.25, fill_na=np.nan)
    level = librosa.feature.rms(y=y, frame_length=1024, hop_length=pitch_hop)[0]
    n = min(len(f0), len(level))
    f0, voiced, probability, level = f0[:n], voiced[:n], probability[:n], level[:n]
    midi = np.full(n, np.nan)
    valid = np.isfinite(f0) & (f0 > 0)
    midi[valid] = librosa.hz_to_midi(f0[valid]) - tuning
    nearest = np.zeros(n, dtype=int)
    nearest[valid] = np.rint(midi[valid]).astype(int)
    stable = (valid & voiced & (probability >= 0.65)
              & (level > max(1e-5, float(np.percentile(level, 90)) * 0.04))
              & (np.abs(midi - nearest) <= 0.35))
    return {"times": np.arange(n) * pitch_hop / pitch_sr, "pitch_classes": nearest % 12,
            "weights": np.where(stable, probability, 0.0),
            "stable_frames": int(stable.sum()), "frames": n}


def _mean_frames(values, levels, start, end, sr):
    a = min(values.shape[1] - 1, max(0, int(start * sr / HOP)))
    b = max(a + 1, min(values.shape[1], int(np.ceil(end * sr / HOP))))
    vector = np.average(values[:, a:b], axis=1, weights=levels[a:b] + 1e-12)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _vocal_histogram(melody, start, end):
    if melody is None:
        return np.zeros(12)
    a, b = np.searchsorted(melody["times"], [start, end])
    hist = np.bincount(melody["pitch_classes"][a:b], weights=melody["weights"][a:b], minlength=12)
    if hist.sum():
        hist /= hist.sum()
    return hist


def compatible_notes(notes, vocal_histogram, limit=0.25):
    """Avoid sustained semitone/tritone clashes; passing notes are not treated as ground truth."""
    histogram = np.asarray(vocal_histogram)
    strong = np.where(histogram >= 0.20, histogram, 0.0)
    if not strong.sum():
        return list(notes)
    strong /= strong.sum()
    return [int(note) for note in notes if sum(
        strong[pitch] for pitch in range(12) if (pitch - note) % 12 in ROUGH_INTERVALS) <= limit]


def select_voicing(root, kind, score, reference, bass, vocal_histogram):
    """Use identified chords where supported; uncertain regions get evidenced open/single-note voicings."""
    chord = [(root + interval) % 12 for interval in CHORDS[kind]]
    safe = compatible_notes(chord, vocal_histogram)
    floor = max(0.08, float(np.max(reference)) * 0.20)
    supported = all(reference[note] >= floor or vocal_histogram[note] >= 0.20 for note in chord)
    if score >= 0.68 and supported and len(safe) == len(chord):
        return root, list(CHORDS[kind]), "chord"

    support = np.asarray(reference) + 0.35 * np.asarray(bass) + 0.65 * np.asarray(vocal_histogram)
    evidenced = [pitch for pitch in range(12) if (
        reference[pitch] >= max(0.16, float(np.max(reference)) * 0.45)
        or bass[pitch] >= 0.45 or vocal_histogram[pitch] >= 0.20)]
    safe = compatible_notes(evidenced, vocal_histogram)
    if not safe:
        return root, [], "rest"
    chosen = root if root in safe else max(safe, key=lambda pitch: support[pitch])
    fifth = (chosen + 7) % 12
    intervals = [0, 7] if fifth in safe else [0]
    return chosen, intervals, "open" if len(intervals) == 2 else "single"


def harmony_plan(samples, sr=ANALYSIS_SR, vocals=None):
    """Infer accompaniment from a real instrumental stem, constrained by confident sung pitches."""
    import librosa

    samples = _samples(samples, sr, "Backing analysis")
    if np.sqrt(np.mean(samples ** 2)) < 1e-5:
        raise ArrangementError("The instrumental is silent; cannot infer a backing arrangement")
    if vocals is not None:
        vocals = _samples(vocals, sr, "Vocal melody analysis")
        if abs(len(vocals) - len(samples)) > sr * 0.15:
            raise ArrangementError("Instrumental and vocal analysis timelines do not match")

    harmonic, percussive = librosa.effects.hpss(samples)
    onset = librosa.onset.onset_strength(y=percussive, sr=sr, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=HOP,
                                          trim=False, units="time")
    tempo = float(np.asarray(tempo).ravel()[0])
    seconds = len(samples) / sr
    beats = np.asarray(beats, dtype=float)
    beats = beats[(beats > 0) & (beats < seconds)]
    if not np.isfinite(tempo) or not 35 <= tempo <= 240 or len(beats) < 4:
        raise ArrangementError("No usable instrumental beat timeline; vocals will not be time-stretched")

    tuning = float(librosa.estimate_tuning(y=harmonic, sr=sr))
    if not np.isfinite(tuning):
        tuning = 0.0
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=HOP, tuning=tuning, norm=2)
    spectrum = np.abs(librosa.stft(harmonic, n_fft=8192, hop_length=HOP))
    frequency = librosa.fft_frequencies(sr=sr, n_fft=8192)
    spectrum[(frequency < 40) | (frequency > 330)] = 0
    bass_chroma = librosa.feature.chroma_stft(S=spectrum ** 2, sr=sr, n_fft=8192,
                                            hop_length=HOP, tuning=tuning, norm=2)
    levels = librosa.feature.rms(y=samples, hop_length=HOP, frame_length=4096)[0]
    frames = min(chroma.shape[1], bass_chroma.shape[1], len(levels))
    chroma, bass_chroma, levels = chroma[:, :frames], bass_chroma[:, :frames], levels[:frames]
    melody = vocal_melody(vocals, sr, tuning) if vocals is not None else None
    bounds = np.unique(np.r_[0.0, beats[::2], seconds])
    reference, bass, sung, active = [], [], [], []
    floor = max(1e-5, float(np.percentile(levels, 90)) * 0.04)
    for start, end in zip(bounds, bounds[1:]):
        reference.append(_mean_frames(chroma, levels, start, end, sr))
        bass.append(_mean_frames(bass_chroma, levels, start, end, sr))
        sung.append(_vocal_histogram(melody, start, end))
        a, b = max(0, int(start * sr / HOP)), min(frames, int(np.ceil(end * sr / HOP)))
        active.append(bool(b > a and np.mean(levels[a:b]) > floor))
    reference, bass, sung = np.asarray(reference), np.asarray(bass), np.asarray(sung)
    states = [(root, kind) for kind in CHORDS for root in range(12)]
    templates = np.zeros((len(states), 12))
    for index, (root, kind) in enumerate(states):
        notes = [(root + interval) % 12 for interval in CHORDS[kind]]
        templates[index, notes] = 1 / np.sqrt(len(notes))
    scores = templates @ reference.T
    evidence = scores * 12.0
    for index, (root, kind) in enumerate(states):
        notes = [(root + interval) % 12 for interval in CHORDS[kind]]
        clash = np.mean([sung[:, [(note + i) % 12 for i in ROUGH_INTERVALS]].sum(axis=1)
                         for note in notes], axis=0)
        evidence[index] += 0.70 * bass[:, root] - 2.5 * clash
    probabilities = np.exp(evidence - evidence.max(axis=0, keepdims=True))
    transition = np.full((len(states), len(states)), 0.30 / (len(states) - 1))
    np.fill_diagonal(transition, 0.70)
    decoded = librosa.sequence.viterbi(probabilities, transition)
    plan = []
    for index, (start, end, state) in enumerate(zip(bounds, bounds[1:], decoded)):
        root, kind = states[state]
        score = float(scores[state, index])
        contrast = float(reference[index].max() - np.median(reference[index]))
        if active[index] and (contrast >= 0.06 or sung[index].sum() > 0):
            root, intervals, mode = select_voicing(root, kind, score, reference[index], bass[index], sung[index])
        else:
            intervals, mode = [], "rest"
        local_beats = beats[(beats >= start) & (beats < end)] if active[index] else []
        plan.append({"start": float(start), "end": float(end), "root": int(root),
                     "minor": mode == "chord" and kind in ("minor", "minor7"),
                     "kind": kind if mode == "chord" else mode, "score": score,
                     "intervals": intervals, "mode": mode, "reliable": bool(intervals),
                     "reference_confident": mode == "chord", "beats": [float(t) for t in local_beats],
                     "vocal_pitch_classes": [int(p) for p in np.flatnonzero(sung[index] >= 0.20)],
                     "vocal_pitch_weights": [float(weight) for weight in sung[index]]})
    if not any(segment["reliable"] for segment in plan):
        raise ArrangementError("No supported, vocal-compatible pitched accompaniment could be inferred")
    active_seconds = sum(s["end"] - s["start"] for s, on in zip(plan, active) if on)
    confident_seconds = sum(s["end"] - s["start"] for s in plan if s["reference_confident"])
    supported_seconds = sum(s["end"] - s["start"] for s in plan if s["reliable"])
    return plan, {
        "method": "instrumental-led-vocal-constrained",
        "tempo_bpm": round(tempo, 2), "beat_count": len(beats), "tuning_cents": round(tuning * 100, 2),
        "chord_templates": list(CHORDS), "reliable_harmony_share": round(confident_seconds / max(active_seconds, 1e-9), 4),
        "supported_accompaniment_share": round(supported_seconds / max(active_seconds, 1e-9), 4),
        "voicing_segments": {mode: sum(s["mode"] == mode for s in plan) for mode in ("chord", "open", "single", "rest")},
        "vocal_melody_stable_frames": melody["stable_frames"] if melody is not None else 0,
        "vocal_melody_total_frames": melody["frames"] if melody is not None else 0,
        "harmony_score_is_accuracy": False, "vocal_analysis_ran": vocals is not None,
        "vocal_conflicts_checked": bool(melody is not None and melody["stable_frames"] > 0),
        "weak_harmony_policy": "Supported open/single-note voicings, otherwise rests; never force a full chord",
    }
