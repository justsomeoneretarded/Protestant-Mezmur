"""Subtle audio augmentation using Rubber Band R3, not a privacy guarantee."""
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


def _render(planar, sr, src, dst, formants):
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
    rendered = _render_native(padded, sr, padded_src, padded_dst, formants)
    end = lead + int(dst[-1])
    if not np.isfinite(rendered).all():
        raise RuntimeError("Rubber Band returned non-finite audio samples")
    if len(rendered) < end:
        raise RuntimeError(f"Rubber Band returned {len(rendered)} frames; at least {end} "
                           "are required even after adding boundary context")
    return rendered[lead:end]


def _render_native(planar, sr, src, dst, formants):
    channels, n = planar.shape
    lib = _rubberband()
    # R3/Finer, synchronized stereo, formant processing, and one thread.
    # Explicit FORMANT_PRESERVED enables envelope processing on older R3 builds.
    options = 0x20000000 | 0x10000000 | 0x01000000 | 0x00010000
    handle = lib.rubberband_new(sr, channels, options, int(dst[-1]) / n, 1.0)
    if not handle:
        raise RuntimeError("Could not allocate a Rubber Band stretcher")
    try:
        if lib.rubberband_get_engine_version(handle) != 3:
            raise RuntimeError("Rubber Band R3 is required for formant shifting")
        lib.rubberband_set_formant_scale(handle, 2.0 ** (formants / 12.0))
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


def augment_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    seed: int | None = None,
    max_speed_change: float = 0.015,
    max_formant_semitones: float = 0.3,
    control_seconds: float = 2.0,
    preserve_duration: bool = True,
) -> AugmentationResult:
    """Augment speech, music, percussion, or mixed audio without intentional pitch shift.

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
    ):
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    if not isinstance(preserve_duration, (bool, np.bool_)):
        raise TypeError("preserve_duration must be boolean")

    n, sr = len(x), int(sample_rate)
    if n > np.iinfo(np.uint32).max:
        raise ValueError("Input exceeds the native C API's 32-bit frame limit")
    rng = np.random.default_rng(seed)
    src, dst = _time_map(n, sr, rng, max_speed_change, control_seconds, preserve_duration)
    if dst[-1] > np.iinfo(np.uint32).max:
        raise ValueError("Output exceeds the native C API's 32-bit frame limit")
    formants = float(rng.uniform(-max_formant_semitones, max_formant_semitones))
    # A disabled augmentation is an exact float32 copy, not a DSP round trip.
    if max_speed_change == 0 and max_formant_semitones == 0:
        return AugmentationResult(x.astype(np.float32, copy=True), sr, src, dst, 0, 1)

    # Rubber Band uses planar channel-first float32 arrays; callers use SoundFile
    # convention. All channels pass through the same engine and time map.
    planar = np.ascontiguousarray(x.reshape(n, channels).T, dtype=np.float32)
    y = _render(planar, sr, src, dst, formants)
    if not np.isfinite(y).all():
        raise RuntimeError("Rubber Band returned non-finite audio samples")
    if len(y) != dst[-1]:
        raise RuntimeError(f"Unexpected output length: got {len(y)} frames, expected {dst[-1]}")
    # Only attenuate overshoots; do not normalize silence or quiet recordings up.
    peak = float(np.max(np.abs(y)))
    gain = 0.999 / peak if peak > 0.999 else 1.0
    y *= gain
    if x.ndim == 1:
        y = y[:, 0]
    return AugmentationResult(y, sr, src, dst, formants, gain, 0.25)


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
    args = parser.parse_args()
    metadata = augment_file(args.input, args.output, seed=args.seed,
                            max_speed_change=args.speed, max_formant_semitones=args.formant,
                            control_seconds=args.control_seconds,
                            preserve_duration=not args.allow_duration_change)
    print(f"Wrote {args.output}; formant shift: {metadata['formant_semitones']:+.3f} semitones")


if __name__ == "__main__":
    main()
