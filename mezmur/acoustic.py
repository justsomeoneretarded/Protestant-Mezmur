"""Guitar-led acoustic performance on the source clock, with source-driven dynamics and structure."""
from functools import lru_cache
from itertools import product

import numpy as np

from .harmony import ArrangementError

STRINGS = (40, 45, 50, 55, 59, 64)  # Standard EADGBE, low to high.
GUITAR_PROGRAM = 25  # General MIDI acoustic steel-string guitar (zero-based).


def prepare_plan(plan, duration=None, cues=()):
    if not plan:
        raise ArrangementError("No harmony sections were supplied")
    if duration is None:
        duration = float(plan[-1]["end"])
    if not np.isfinite(duration) or duration <= 0:
        raise ArrangementError("The arrangement duration must be finite and positive")
    out, previous_end = [], 0.0
    for section in plan:
        try:
            start, end = float(section["start"]), float(section["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArrangementError("Harmony sections need numeric start/end timestamps") from error
        if (not np.isfinite([start, end]).all() or start < -1e-6 or end <= start
                or start < previous_end - 1e-6 or end > duration + 0.15):
            raise ArrangementError("Harmony sections must be ordered, non-overlapping and within the source duration")
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        try:
            root_value = float(section["root"])
            root = int(root_value)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ArrangementError("Harmony sections need a valid root pitch class") from error
        if root != root_value:
            raise ArrangementError("Root pitch classes must be integers")
        intervals = section.get("intervals")
        if intervals is None:
            intervals = [0, 3 if section.get("minor") else 4, 7]
        if root not in range(12) or any(not isinstance(n, (int, np.integer)) or not 0 <= n < 12 for n in intervals):
            raise ArrangementError("Harmony sections contain invalid pitch classes")
        raw_beats = np.asarray(section.get("beats", []), dtype=float)
        if raw_beats.ndim != 1 or not np.isfinite(raw_beats).all():
            raise ArrangementError("Beat timestamps must be finite")
        beats = sorted(set(float(t) for t in raw_beats if start <= t < end))
        energy = float(section.get("energy", 0.5))
        percussion = float(section.get("percussive_strength", 0.0))
        activity = float(section.get("vocal_activity", 1.0))
        if not np.isfinite([energy, percussion, activity]).all():
            raise ArrangementError("Arrangement dynamics must be finite")
        out.append({**section, "start": start, "end": end, "root": root, "intervals": list(intervals),
                    "beats": beats, "energy": float(np.clip(energy, 0, 1)),
                    "percussive_strength": float(np.clip(percussion, 0, 1)),
                    "vocal_activity": float(np.clip(activity, 0, 1))})
        previous_end = end
    if not out:
        raise ArrangementError("No harmony section fits the source timeline")
    valid_cues = []
    for cue in cues or ():
        try:
            a, b = float(cue["start"]), float(cue["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite([a, b]).all() and 0 <= a < b <= duration + 0.15:
            valid_cues.append((a, min(b, duration), cue.get("chorus") is True))
    levels = np.asarray([s["energy"] for s in out])
    for i, section in enumerate(out):
        start, end = section["start"], section["end"]
        overlap = [cue for cue in valid_cues if cue[0] < end and cue[1] > start]
        singing = bool(overlap) or section["vocal_activity"] > 0.15
        chorus = any(cue[2] and min(end, cue[1]) - max(start, cue[0]) >= (end - start) * 0.20 for cue in overlap)
        intensity = float(np.median(levels[max(0, i - 2):i + 3]))
        label = "chorus" if chorus else "verse" if singing else "instrumental"
        style = "strum" if ((chorus and intensity >= 0.50) or intensity >= 0.78) else "fingerpick"
        if not section.get("reliable") or not section["intervals"]:
            style = "rest"
        section.update(section=label, playing_style=style, intensity=intensity, vocal_active=singing)
    return out


@lru_cache(maxsize=256)
def _voicings(root, pitch_classes):
    allowed = set(pitch_classes)
    choices = [(-1,) + tuple(fret for fret in range(13) if (base + fret) % 12 in allowed) for base in STRINGS]
    best = []
    for frets in product(*choices):
        notes = tuple((string, fret, STRINGS[string] + fret) for string, fret in enumerate(frets) if fret >= 0)
        if not notes:
            continue
        pitches = [note[2] for note in notes]
        if pitches != sorted(set(pitches)) or root not in {pitch % 12 for pitch in pitches}:
            continue
        pressed = [fret for fret in frets if fret > 0]
        span = max(pressed) - min(pressed) if pressed else 0
        if span > 4 or len(set(pressed)) > 4:
            continue
        covered = {pitch % 12 for pitch in pitches}
        target_strings = 2 if len(allowed) == 1 else min(5, len(allowed) + 1)
        cost = (40 * len(allowed - covered) + 2 * abs(len(notes) - target_strings) + span
                + sum(max(0, fret) for fret in frets) * 0.12 + (0 if pitches[0] % 12 == root else 3))
        best.append((cost, notes))
    if not best:
        raise ArrangementError("No playable guitar voicing supports the inferred notes")
    return tuple(notes for _, notes in sorted(best, key=lambda entry: entry[0])[:12])


def guitar_voicing(root, intervals, previous=()):
    pitch_classes = tuple(sorted({(int(root) + int(interval)) % 12 for interval in intervals}))
    if not pitch_classes:
        return ()
    candidates = _voicings(int(root), pitch_classes)
    if not previous:
        return candidates[0]
    old = {string: pitch for string, _, pitch in previous}
    def distance(candidate):
        motion = sum(abs(pitch - old.get(string, pitch)) for string, _, pitch in candidate)
        return motion + 0.5 * abs(len(candidate) - len(previous))
    # Candidates already favor playable, complete chords; use voice leading only among these.
    return min(candidates[:6], key=distance)


def write_midi(plan, target, seed=0, tuning_cents=0.0, duration=None, cues=()):
    import pretty_midi

    if not np.isfinite(tuning_cents) or abs(tuning_cents) > 100:
        raise ArrangementError("Instrument tuning must be finite and within one semitone")
    sections = prepare_plan(plan, duration, cues)
    duration = float(duration if duration is not None else sections[-1]["end"])
    midi = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=960)
    guitar = pretty_midi.Instrument(GUITAR_PROGRAM, name="Acoustic steel-string guitar")
    bass = pretty_midi.Instrument(32, name="Acoustic bass")
    piano = pretty_midi.Instrument(0, name="Sparse piano accents")
    pad = pretty_midi.Instrument(89, name="Quiet chorus pad")
    drums = pretty_midi.Instrument(0, is_drum=True, name="Source-supported light percussion")
    rng = np.random.default_rng(seed)
    previous = ()

    def add(instrument, pitch, start, end, velocity, boundary):
        end = min(float(end), float(boundary) - 0.001, duration - 0.001)
        start = max(0.0, float(start))
        if end - start < 0.008:
            return
        instrument.notes.append(pretty_midi.Note(
            velocity=int(np.clip(round(velocity), 1, 100)), pitch=int(pitch), start=start, end=end))

    for section in sections:
        start, end = section["start"], section["end"]
        if section["playing_style"] == "rest":
            previous = ()
            section["guitar_voicing"] = []
            continue
        voice = guitar_voicing(section["root"], section["intervals"], previous)
        previous = voice
        section["guitar_voicing"] = [{"string": string + 1, "fret": fret, "pitch": pitch}
                                    for string, fret, pitch in voice]
        pitches = [note[2] for note in voice]
        beats = section["beats"] or [start]
        bounds = sorted(set([start, *beats, end]))
        pulses = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 0.08]
        level = section["intensity"]
        velocity = 48 + 24 * level
        for index, (a, b) in enumerate(pulses):
            span = b - a
            if section["playing_style"] == "strum" and len(pitches) >= 2:
                for up, moment in ((False, a), (True, a + span * 0.55)):
                    stroke = list(reversed(pitches)) if up else pitches
                    if up:
                        stroke = stroke[:max(2, len(stroke) - 1)]
                    stagger = min(0.013, span / max(8, len(stroke) * 4))
                    for string, pitch in enumerate(stroke):
                        onset = moment + string * stagger
                        add(guitar, pitch, onset, onset + min(1.1, span * 0.90),
                            velocity - (9 if up else 0) + rng.integers(-2, 3), end)
            else:
                # The same thumb/upper-string motif returns in repeated quiet passages.
                upper = pitches[-1] if index % 2 == 0 else pitches[len(pitches) // 2]
                for moment, pitch in ((a, pitches[0]), (a + span * 0.5, upper)):
                    add(guitar, pitch, moment, moment + min(1.3, span * 0.85),
                        velocity + rng.integers(-2, 3), end)
            root = 36 + section["root"]
            bass_pitch = root + 7 if (index % 2 and level >= 0.65 and 7 in section["intervals"]) else root
            add(bass, bass_pitch, a, min(b, a + span * 0.90), 38 + 18 * level, end)
        # Additional colors follow observed dynamics, never an always-on full-band loop.
        if not section["vocal_active"] and section.get("reference_confident") and level >= 0.65:
            for interval in section["intervals"][:3]:
                add(piano, 60 + section["root"] + interval, start, min(end, start + 0.7), 32 + 9 * level, end)
        if section["section"] == "chorus" and level >= 0.65 and len(section["intervals"]) >= 3:
            for interval in section["intervals"]:
                add(pad, 60 + section["root"] + interval, start, end, 18 + 7 * level, end)
        if section["percussive_strength"] >= 0.45 and level >= 0.65 and len(section["beats"]) >= 2:
            for index, beat in enumerate(section["beats"]):
                add(drums, 42, beat, beat + 0.045, 26 + 8 * level, end)
                if index == 0:
                    add(drums, 36, beat, beat + 0.07, 32 + 8 * level, end)
    instruments = [instrument for instrument in (guitar, bass, piano, pad, drums) if instrument.notes]
    if not guitar.notes:
        raise ArrangementError("No supported guitar notes were available to synthesize")
    bend = int(np.clip(round(float(tuning_cents) / 200 * 8192), -8192, 8191))
    for instrument in instruments:
        if instrument.is_drum:
            continue
        for number, value in ((101, 0), (100, 0), (6, 2), (38, 0), (101, 127), (100, 127)):
            instrument.control_changes.append(pretty_midi.ControlChange(number, value, 0))
        instrument.pitch_bends.append(pretty_midi.PitchBend(bend, 0))
    for instrument in instruments:
        for pitch in {note.pitch for note in instrument.notes}:
            repeated = sorted((note for note in instrument.notes if note.pitch == pitch), key=lambda n: n.start)
            for note, following in zip(repeated, repeated[1:]):
                note.end = min(note.end, max(note.start + 0.001, following.start - 0.003))
    midi.instruments = instruments
    midi.write(str(target))
    return {"arrangement_style": "adaptive-guitar-and-bass", "instruments": [i.name for i in instruments],
            "playing_styles": {style: sum(s["playing_style"] == style for s in sections)
                               for style in ("fingerpick", "strum", "rest")},
            "note_count": sum(len(i.notes) for i in instruments), "midi_end_seconds": midi.get_end_time(),
            "duration_limit_seconds": duration, "segments": sections}
