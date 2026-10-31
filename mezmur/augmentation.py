"""Coherent audio augmentation using Rubber Band R3, not a privacy guarantee."""
from __future__ import annotations

import argparse
import ctypes as ct
import ctypes.util
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf


MAX_CHANNELS = 32
MIN_SAMPLE_RATE, MAX_SAMPLE_RATE = 8000, 192000
FLOAT_PTR = ct.POINTER(ct.c_float)
FLOAT_PTR_PTR = ct.POINTER(FLOAT_PTR)
UINT_PTR = ct.POINTER(ct.c_uint)


@lru_cache(maxsize=1)
def _rubberband():
    """Bind the documented C API; the DSP itself is provided by Rubber Band."""
    path = os.environ.get("RUBBERBAND_LIBRARY") or ctypes.util.find_library("rubberband")
    if not path:
        raise RuntimeError("Install Rubber Band >= 3.0 (e.g. apt install librubberband2)")
    lib = ct.CDLL(path)
    signatures = {
        "new": (ct.c_void_p, [ct.c_uint, ct.c_uint, ct.c_int, ct.c_double, ct.c_double]),
        "delete": (None, [ct.c_void_p]),
        "get_engine_version": (ct.c_int, [ct.c_void_p]),
        "set_formant_scale": (None, [ct.c_void_p, ct.c_double]),
        "set_expected_input_duration": (None, [ct.c_void_p, ct.c_uint]),
        "set_max_process_size": (None, [ct.c_void_p, ct.c_uint]),
        "set_key_frame_map": (None, [ct.c_void_p, ct.c_uint, UINT_PTR, UINT_PTR]),
        "study": (None, [ct.c_void_p, FLOAT_PTR_PTR, ct.c_uint, ct.c_int]),
        "process": (None, [ct.c_void_p, FLOAT_PTR_PTR, ct.c_uint, ct.c_int]),
        "available": (ct.c_int, [ct.c_void_p]),
        "retrieve": (ct.c_uint, [ct.c_void_p, FLOAT_PTR_PTR, ct.c_uint]),
    }
    try:
        for name, (restype, argtypes) in signatures.items():
            fn = getattr(lib, "rubberband_" + name)
            fn.restype, fn.argtypes = restype, argtypes
    except AttributeError as error:
        raise RuntimeError("Rubber Band >= 3.0 with the R3 engine is required") from error
    return lib


def _pointers(planar):
    return (FLOAT_PTR * len(planar))(*(row.ctypes.data_as(FLOAT_PTR) for row in planar))


def _render(planar, sr, src, dst, formants, pitch_semitones=0.0):
    _, n = planar.shape
    # R3 may finish a few synthesis frames short when a variable time map is
    # used. Supply trailing context so that only disposable silence is affected,
    # rather than padding over missing song samples or rejecting valid audio.
    tail = round(sr * 0.25)
    lead = tail if n < 64 else 0
    if max(int(src[-1]), int(dst[-1])) + lead + tail > np.iinfo(np.uint32).max:
        raise ValueError("Audio plus boundary context exceeds the native C API's frame limit")
    padded = np.pad(planar, ((0, 0), (lead, tail)))
    padded_src = np.r_[src + lead, src[-1] + lead + tail]
    padded_dst = np.r_[dst + lead, dst[-1] + lead + tail]
    if lead:
        padded_src, padded_dst = np.r_[0, padded_src], np.r_[0, padded_dst]
    rendered = _render_native(padded, sr, padded_src, padded_dst, formants, pitch_semitones)
    end = lead + int(dst[-1])
    if not np.isfinite(rendered).all():
        raise RuntimeError("Rubber Band returned non-finite audio samples")
    if len(rendered) < end:
        raise RuntimeError(f"Rubber Band returned {len(rendered)} frames; at least {end} "
                           "are required even after adding boundary context")
    return rendered[lead:end]


def _render_native(planar, sr, src, dst, formants, pitch_semitones=0.0):
    channels, n = planar.shape
    lib = _rubberband()
    # R3/Finer, synchronized stereo, formant processing, and one thread.
    # Explicit FORMANT_PRESERVED enables envelope processing on older R3 builds.
    options = 0x20000000 | 0x10000000 | 0x01000000 | 0x00010000
    pitch_scale = 2.0 ** (pitch_semitones / 12.0)
    handle = lib.rubberband_new(sr, channels, options, int(dst[-1]) / n, pitch_scale)
    if not handle:
        raise RuntimeError("Could not allocate a Rubber Band stretcher")
    try:
        if lib.rubberband_get_engine_version(handle) != 3:
            raise RuntimeError("Rubber Band R3 is required for formant shifting")
        # Compensate the whole-mix transposition to keep the independent formant
        # control absolute, rather than shifting voices by the pitch interval too.
        lib.rubberband_set_formant_scale(handle, 2.0 ** (formants / 12.0) / pitch_scale)
        lib.rubberband_set_expected_input_duration(handle, n)
        block_size = 8192
        lib.rubberband_set_max_process_size(handle, block_size)
        # Offline processing first analyzes the complete input, then renders it.
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            lib.rubberband_study(handle, _pointers(planar[:, start:end]), end - start, end == n)
        # R3 computes an initial ratio from the first anchor; omit (0, 0) to
        # avoid a 0/0 calculation. Install the map after the study pass.
        source = (ct.c_uint * (len(src) - 1))(*src[1:].tolist())
        target = (ct.c_uint * (len(dst) - 1))(*dst[1:].tolist())
        lib.rubberband_set_key_frame_map(handle, len(src) - 1, source, target)
        blocks = []

        def drain():
            while (available := lib.rubberband_available(handle)) > 0:
                block = np.empty((channels, available), dtype=np.float32)
                got = lib.rubberband_retrieve(handle, _pointers(block), available)
                if not got:
                    raise RuntimeError("Rubber Band returned no available samples")
                blocks.append(block[:, :got])

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            lib.rubberband_process(handle, _pointers(planar[:, start:end]), end - start, end == n)
            drain()
        if lib.rubberband_available(handle) != -1 or not blocks:
            raise RuntimeError("Rubber Band did not finish producing the audio")
        return np.concatenate(blocks, axis=1).T.copy()
    finally:
        lib.rubberband_delete(handle)


@dataclass(frozen=True)
class AugmentationResult:
    audio: np.ndarray
    sample_rate: int
    input_frames: np.ndarray
    output_frames: np.ndarray
    formant_semitones: float
    peak_gain: float
    boundary_context_seconds: float = 0.0
    pitch_semitones: float = 0.0
    tempo_factor: float = 1.0
    noise_snr_db: float | None = None
    noise_profile: str | None = None
    space_effect: str = "none"

    def map_times(self, input_seconds: np.ndarray) -> np.ndarray:
        """Map event times using planned anchors; allow for DSP timing tolerance."""
        times = np.asarray(input_seconds, dtype=np.float64)
        duration = self.input_frames[-1] / self.sample_rate
        if not np.isfinite(times).all() or np.any((times < 0) | (times > duration)):
            raise ValueError("Annotation times must be finite and within the input duration")
        return np.interp(times, self.input_frames / self.sample_rate,
                         self.output_frames / self.sample_rate)

    def metadata(self) -> dict:
        return {
            "engine": "Rubber Band R3 (native C API)",
            "sample_rate": self.sample_rate,
            "input_frames": self.input_frames.tolist(),
            "output_frames": self.output_frames.tolist(),
            "formant_semitones": self.formant_semitones,
            "peak_gain": self.peak_gain,
            "boundary_context_seconds": self.boundary_context_seconds,
            "pitch_semitones": self.pitch_semitones,
            "tempo_factor": self.tempo_factor,
            "noise_snr_db": self.noise_snr_db,
            "noise_profile": self.noise_profile,
            "space_effect": self.space_effect,
            "timing_note": "Planned anchors; verify fine-grained labels after processing",
            "privacy_guaranteed": False,
        }


def _time_map(n, sr, rng, strength, control_seconds, preserve_duration):
    # Interpolate slow random controls, then integrate duration ratios into a
    # monotonic time map. No slicing/rejoining of independently stretched clips.
    count = max(1, math.ceil(n / (sr * 0.05)))
    src = np.rint(np.linspace(0, n, count + 1)).astype(np.int64)
    widths = np.diff(src)
    controls = np.linspace(0, n, max(1, math.ceil(n / (sr * control_seconds))) + 1)
    values = rng.uniform(-1.0, 1.0, len(controls))
    midpoints = (src[:-1] + src[1:]) / 2
    index = np.searchsorted(controls, midpoints, side="right") - 1
    fraction = (midpoints - controls[index]) / np.diff(controls)[index]
    weight = 0.5 - 0.5 * np.cos(np.pi * fraction)
    curve = values[index] * (1 - weight) + values[index + 1] * weight
    if preserve_duration:
        curve -= np.average(curve, weights=widths)
    curve /= max(1.0, float(np.max(np.abs(curve))))
    # This bound on duration ratio keeps playback speed within 1 +/- strength
    # before sample rounding, even when preserving the total duration.
    ratios = 1 + (strength / (1 + strength)) * curve
    dst = np.rint(np.r_[0.0, np.cumsum(widths * ratios)]).astype(np.int64)
    if preserve_duration:
        dst[-1] = n
    if np.any(np.diff(dst) <= 0):
        raise RuntimeError("Generated a non-monotonic time map")
    return src, dst


def _mix_noise(samples, sr, rng, snr_db, profile):
    """Add band-limited mono noise at a measured SNR, gated out of silent gaps."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import butter, sosfilt

    signal_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if signal_rms == 0:
        return samples
    power = np.mean(samples ** 2, axis=1)
    envelope = np.sqrt(np.maximum(0, uniform_filter1d(power, max(1, round(sr * 0.02)))))
    floor = max(1e-6, signal_rms * 0.001)
    gate = np.clip((envelope - floor) / (floor * 3), 0, 1)
    gate *= np.any(samples != 0, axis=1)
    high = 1200 if profile == "ambient" else 8000
    filt = butter(2, [80, min(high, sr * 0.4)], btype="bandpass", fs=sr, output="sos")
    noise = sosfilt(filt, rng.standard_normal(len(samples))).astype(np.float32)
    if profile == "vinyl":
        # Sparse, band-limited synthetic pops over a quieter steady noise bed.
        impulses = np.zeros(len(samples), dtype=np.float32)
        count = int(rng.poisson(1.5 * len(samples) / sr))
        positions = rng.integers(0, len(samples), count)
        np.add.at(impulses, positions, rng.uniform(-1, 1, count))
        noise = noise * 0.35 + sosfilt(filt, impulses).astype(np.float32) * 12
    noise *= gate
    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    if noise_rms > 0:
        noise *= signal_rms / (10 ** (snr_db / 20) * noise_rms)
        samples = samples + noise[:, None]
    return samples


def _apply_space(samples, sr, effect):
    """Use established delay/reverb engines, retaining the original frame count."""
    from pedalboard import Delay, Pedalboard, Reverb

    output = np.empty_like(samples)
    # Process stereo pairs (or a final mono channel) for multichannel inputs.
    for start in range(0, samples.shape[1], 2):
        plugin = (Reverb(room_size=0.2, damping=0.7, wet_level=0.08, dry_level=1.0, width=0.7)
                  if effect == "room" else Delay(delay_seconds=0.08, feedback=0.12, mix=0.12))
        board = Pedalboard([plugin])
        group = np.ascontiguousarray(samples[:, start:start + 2].T)
        if len(samples) < 64:
            group = np.pad(group, ((0, 0), (0, 64 - len(samples))))
        output[:, start:start + 2] = board(group, sr, reset=True).T[:len(samples)]
    # Blend the effect back to dry at EOF instead of ending on a cut-off echo.
    fade = min(len(output), max(1, round(sr * 0.05)))
    weight = np.linspace(1, 0, fade + 1, dtype=np.float32)[1:, None]
    output[-fade:] = output[-fade:] * weight + samples[-fade:] * (1 - weight)
    return output


def augment_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    seed: int | None = None,
    max_speed_change: float = 0.015,
    max_formant_semitones: float = 0.3,
    control_seconds: float = 2.0,
    preserve_duration: bool = True,
    pitch_semitones: float = 0.0,
    tempo_factor: float = 1.0,
    noise_snr_db: float | None = None,
    noise_profile: str = "hiss",
    space_effect: str = "none",
) -> AugmentationResult:
    """Augment speech, music, percussion, or mixed audio with coherent transforms.

    Args:
        samples: Floating-point audio in [-1, 1], shaped (frames,) or
            (frames, channels). Convert integer PCM before calling.
        sample_rate: Unchanged sample rate, 8000..192000 Hz.
        seed: Seed for reproducible augmentation parameters.
        max_speed_change: Local playback-speed deviation, e.g. 0.015 for 1.5%.
            Applied via 50 ms time-map anchors, with sample-rounding tolerance.
        max_formant_semitones: Random spectral-envelope shift within +/- this
            value. This changes timbre, not the intended musical pitch. For
            non-vocal material, "formants" are not a speaker-identity property.
        control_seconds: Spacing of slow, cosine-interpolated random controls.
        preserve_duration: Keep the original number of frames; internal event
            positions still change. False permits a small duration change.
        pitch_semitones: Whole-mix transposition; every pitched part moves by the
            same interval, preserving their musical relationships (default 0).
        tempo_factor: Nominal playback speed, independent of pitch (default 1).
            Non-unity values require preserve_duration=False.
        noise_snr_db: Optional band-limited noise at this signal-to-noise ratio.
            Higher is quieter; 50 dB is conservative, not guaranteed inaudible.
        noise_profile: hiss, synthetic vinyl crackle, synthetic ambient room tone, or off.
        space_effect: none, small-room reverb, or a short slapback echo.

    Returns:
        Float32 audio with the original channel layout, plus timing metadata.

    These transformations do not guarantee anonymity, fingerprint removal, or
    intelligibility. Validate quality and label integrity on your own dataset.
    """
    x = np.asarray(samples)
    if x.ndim not in (1, 2) or x.shape[0] == 0:
        raise ValueError("Audio must be nonempty (frames,) or (frames, channels)")
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("Use normalized floating-point samples, not integer PCM")
    channels = 1 if x.ndim == 1 else x.shape[1]
    if not 1 <= channels <= MAX_CHANNELS:
        raise ValueError(f"Expected 1..{MAX_CHANNELS} channels")
    if not np.isfinite(x).all() or np.max(np.abs(x)) > 1:
        raise ValueError("Samples must be finite and normalized to [-1, 1]")
    if (isinstance(sample_rate, (bool, np.bool_))
            or not isinstance(sample_rate, (int, np.integer))
            or not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE):
        raise ValueError("sample_rate must be an integer in 8000..192000 Hz")
    for name, value, lower, upper in (
        ("max_speed_change", max_speed_change, 0, 0.05),
        ("max_formant_semitones", max_formant_semitones, 0, 1),
        ("control_seconds", control_seconds, 0.25, 60),
        ("pitch_semitones", pitch_semitones, -3, 3),
        ("tempo_factor", tempo_factor, 0.9, 1.1),
    ):
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    if not isinstance(preserve_duration, (bool, np.bool_)):
        raise TypeError("preserve_duration must be boolean")
    if preserve_duration and tempo_factor != 1:
        raise ValueError("Changing tempo requires preserve_duration=False")
    if noise_snr_db is not None and (not np.isfinite(noise_snr_db) or not 20 <= noise_snr_db <= 80):
        raise ValueError("noise_snr_db must be None or finite and in [20, 80]")
    if noise_profile not in ("hiss", "vinyl", "ambient", "off"):
        raise ValueError("noise_profile must be hiss, vinyl, ambient, or off")
    if noise_profile == "off":
        noise_snr_db = None
    if space_effect not in ("none", "room", "echo"):
        raise ValueError("space_effect must be none, room, or echo")

    n, sr = len(x), int(sample_rate)
    if n > np.iinfo(np.uint32).max:
        raise ValueError("Input exceeds the native C API's 32-bit frame limit")
    rng = np.random.default_rng(seed)
    src, dst = _time_map(n, sr, rng, max_speed_change, control_seconds, preserve_duration)
    dst = np.rint(dst / tempo_factor).astype(np.int64)
    if dst[-1] > np.iinfo(np.uint32).max:
        raise ValueError("Output exceeds the native C API's 32-bit frame limit")
    formants = float(rng.uniform(-max_formant_semitones, max_formant_semitones))
    # A disabled augmentation is an exact float32 copy, not a DSP round trip.
    needs_dsp = any((max_speed_change, max_formant_semitones, pitch_semitones, tempo_factor != 1))
    if not needs_dsp and noise_snr_db is None and space_effect == "none":
        return AugmentationResult(x.astype(np.float32, copy=True), sr, src, dst, 0, 1)

    # Rubber Band uses planar channel-first float32 arrays; callers use SoundFile
    # convention. All channels pass through the same engine and time map.
    if needs_dsp:
        planar = np.ascontiguousarray(x.reshape(n, channels).T, dtype=np.float32)
        y = _render(planar, sr, src, dst, formants, pitch_semitones)
    else:
        y = x.reshape(n, channels).astype(np.float32, copy=True)
    if not np.isfinite(y).all():
        raise RuntimeError("Rubber Band returned non-finite audio samples")
    if len(y) != dst[-1]:
        raise RuntimeError(f"Unexpected output length: got {len(y)} frames, expected {dst[-1]}")
    if space_effect != "none":
        y = _apply_space(y, sr, space_effect)
    if noise_snr_db is not None:
        y = _mix_noise(y, sr, rng, noise_snr_db, noise_profile)
    if not np.isfinite(y).all():
        raise RuntimeError("Audio effects returned non-finite samples")
    # Only attenuate overshoots; do not normalize silence or quiet recordings up.
    peak = float(np.max(np.abs(y)))
    gain = 0.999 / peak if peak > 0.999 else 1.0
    y *= gain
    if x.ndim == 1:
        y = y[:, 0]
    return AugmentationResult(y, sr, src, dst, formants, gain, 0.25 if needs_dsp else 0,
                              pitch_semitones, tempo_factor, noise_snr_db,
                              noise_profile if noise_snr_db is not None else None, space_effect)


def augment_file(input_path: str | Path, output_path: str | Path, **kwargs) -> dict:
    """Write new WAV/FLAC audio and a JSON time-map sidecar; never overwrite files.

    File decoding/encoding supports the formats provided by your libsndfile build.
    Source tags and filenames are not copied into the output or sidecar.
    """
    source, target = Path(input_path), Path(output_path)
    sidecar = target.with_suffix(target.suffix + ".augmentation.json")
    if source.resolve() == target.resolve() or target.exists() or sidecar.exists():
        raise FileExistsError("Choose a new output path; originals are never overwritten")
    if target.suffix.lower() not in (".wav", ".flac"):
        raise ValueError("Output must be .wav or .flac (lossless)")
    samples, sr = sf.read(source, dtype="float32", always_2d=True)
    result = augment_audio(samples, sr, **kwargs)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, result.audio, sr, subtype="PCM_24")
    metadata = result.metadata()
    metadata["seed"] = kwargs.get("seed")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--speed", type=float, default=0.015,
                        help="Maximum fractional playback-speed change (default 0.015)")
    parser.add_argument("--formant", type=float, default=0.3,
                        help="Maximum absolute formant shift in semitones (default 0.3)")
    parser.add_argument("--control-seconds", type=float, default=2.0)
    parser.add_argument("--allow-duration-change", action="store_true")
    parser.add_argument("--pitch", type=float, default=0, help="Whole-mix transposition in semitones")
    parser.add_argument("--tempo", type=float, default=1, help="Nominal tempo factor; 1.04 is 4 percent faster")
    parser.add_argument("--noise-snr", type=float, default=None, help="Optional noise SNR in dB; higher is quieter")
    parser.add_argument("--noise-profile", choices=["hiss", "vinyl", "ambient", "off"], default="hiss")
    parser.add_argument("--space-effect", choices=["none", "room", "echo"], default="none")
    args = parser.parse_args()
    metadata = augment_file(args.input, args.output, seed=args.seed,
                            max_speed_change=args.speed, max_formant_semitones=args.formant,
                            control_seconds=args.control_seconds,
                            preserve_duration=not args.allow_duration_change,
                            pitch_semitones=args.pitch, tempo_factor=args.tempo, noise_snr_db=args.noise_snr,
                            noise_profile=args.noise_profile, space_effect=args.space_effect)
    print(f"Wrote {args.output}; formant shift: {metadata['formant_semitones']:+.3f} semitones")


if __name__ == "__main__":
    main()
