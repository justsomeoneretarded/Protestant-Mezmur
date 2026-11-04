"""Listening-only experiment: current preservation versus stronger source-conditioned rearrangement."""
import argparse
import hashlib
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

from mezmur.common import write_json
from mezmur.neural_backing import DEFAULT_STYLE, assess_candidate

PROFILES = {
    "01-preserve": {"task_type": "cover-nofsq", "strength": 0.80, "caption": DEFAULT_STYLE},
    "02-cover-050": {"task_type": "cover", "strength": 0.50},
    "03-cover-035": {"task_type": "cover", "strength": 0.35},
}
NEW_STYLE = (
    "Instrumental live gospel-soul quartet. Prominent expressive bass guitar with melodic connecting phrases, "
    "warm Rhodes electric piano with rich chord inversions and answering fills, Hammond organ swells, "
    "a light acoustic drum kit with brushed snare and tasteful tom fills. A spacious, intimate live performance "
    "with clear phrase endings, rests and dynamic movement. Follow the source harmonic movement and groove. Instrumental."
)


def run_profile(profile):
    configuration = PROFILES[profile]
    root = Path(os.environ["MEZMUR_AI_ROOT"])
    out = Path("trial") / profile
    out.mkdir(parents=True, exist_ok=True)
    source_path = next(Path("source-kit").rglob("reference.wav"))
    original, sr = sf.read(source_path, dtype="float32", always_2d=True)
    assert original.shape[1] == 2 and np.isfinite(original).all()
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    shutil.copy2(source_path, out / "00-original-instrumental.wav")
    params = {"task_type": configuration["task_type"], "audio_cover_strength": configuration["strength"],
              "caption": configuration.get("caption", NEW_STYLE), "lyrics": "[Instrumental]",
              "duration": len(original) / sr, "seed": 40, "inference_steps": 8,
              "synth_batch_size": 1, "cover_noise_strength": 0.0, "peak_clip": 0,
              "output_format": "wav32", "synth_model": "acestep-v15-turbo-Q8_0.gguf"}
    request = out / "request.json"
    write_json(request, params)
    started = time.monotonic()
    log_path = out / "engine.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            subprocess.run([str(root / "source/build/ace-synth"), "--models", str(root / "models"),
                            "--device", "CPU", "--src-audio", str(source_path), "--request", str(request)],
                           stdout=log, stderr=subprocess.STDOUT, timeout=30 * 60, check=True)
        except subprocess.SubprocessError:
            log.flush()
            print(log_path.read_text()[-20000:], flush=True)
            raise
    generated = out / "request0.wav"
    values, rate = sf.read(generated, dtype="float32", always_2d=True)
    assert values.shape[1] == 2 and np.isfinite(values).all()
    if abs(len(values) / rate - len(original) / sr) > 0.75:
        raise RuntimeError("Generated duration changed materially")
    if rate != sr:
        d = math.gcd(rate, sr)
        values = resample_poly(values, sr // d, rate // d)
    n = len(original)
    values = np.pad(values[:n], ((0, max(0, n - len(values))), (0, 0)))
    sf.write(out / (profile + "-raw.wav"), values, sr, subtype="FLOAT")
    generated.unlink()
    report = {"profile": profile, "request": params, "sample_rate": sr, "samples": n,
              "duration_seconds": n / sr, "elapsed_seconds": time.monotonic() - started,
              "source_sha256": before, "raw_metrics": assess_candidate(values, original, sr),
              "musical_quality_certified": False, "vocal_compatibility_certified": False}
    write_json(out / "report.json", report)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
    print("REARRANGEMENT_CANDIDATE_READY " + json.dumps(report), flush=True)


def measure(path):
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(path), "-af",
         "loudnorm=I=-18:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
        check=True, capture_output=True, text=True)
    data = json.loads(process.stderr[process.stderr.rfind("{"):])
    result = {"lufs": float(data["input_i"]), "true_peak_db": float(data["input_tp"])}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Cannot level-match a silent/nonfinite listening sample")
    return result


def package():
    from mezmur.separate import separate, instrumental_stem

    out = Path("listening-pack")
    out.mkdir(parents=True, exist_ok=True)
    raw, clean = out / "raw", out / "cleaned"
    raw.mkdir(exist_ok=True)
    clean.mkdir(exist_ok=True)
    original_path = next(Path("candidates").rglob("00-original-instrumental.wav"))
    original, sr = sf.read(original_path, dtype="float32", always_2d=True)
    n = len(original)
    sf.write(raw / "00-original-instrumental.wav", original, sr, subtype="FLOAT")
    sf.write(clean / "00-original-instrumental.wav", original, sr, subtype="FLOAT")
    reports = {}
    for profile in PROFILES:
        path = next(Path("candidates").rglob(profile + "-raw.wav"))
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == sr and values.shape == original.shape
        shutil.copy2(path, raw / (profile + ".wav"))
        folder = Path(".runtime") / ("clean-" + profile)
        folder.mkdir(parents=True, exist_ok=True)
        voice = separate(path, folder, require_vocals=True)
        cleaned_path = instrumental_stem(voice, folder / "cleaned.wav")
        cleaned, cleaned_sr = sf.read(cleaned_path, dtype="float32", always_2d=True)
        assert cleaned_sr == sr and abs(len(cleaned) - n) <= sr * 0.15
        cleaned = np.pad(cleaned[:n], ((0, max(0, n - len(cleaned))), (0, 0)))
        sf.write(clean / (profile + ".wav"), cleaned, sr, subtype="FLOAT")
        source_report = path.parent / "report.json"
        reports[profile] = json.loads(source_report.read_text())
        reports[profile]["cleaned_metrics"] = assess_candidate(cleaned, original, sr)
        reports[profile]["raw_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copy2(path.parent / "request.json", out / (profile + "-request.json"))
    paths = sorted(raw.glob("*.wav")) + sorted(clean.glob("*.wav"))
    levels = {str(path): measure(path) for path in paths}
    target = min(-18.0, *(value["lufs"] - 1.5 - value["true_peak_db"] for value in levels.values()))
    final_values = {}
    for path in paths:
        values, rate = sf.read(path, dtype="float32", always_2d=True)
        gain_db = target - levels[str(path)]["lufs"]
        matched = (values * 10 ** (gain_db / 20)).astype(np.float32)
        assert np.isfinite(matched).all() and float(np.max(np.abs(matched))) < 1
        sf.write(path, matched, rate, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                        "-c:a", "libmp3lame", "-b:a", "192k", str(path.with_suffix(".mp3"))], check=True)
        levels[str(path)]["constant_gain_db"] = gain_db
        levels[str(path)]["matched_lufs"] = measure(path)["lufs"]
        if path.parent == clean:
            shutil.copy2(path.with_suffix(".mp3"), out / path.with_suffix(".mp3").name)
            final_values[path.stem] = matched
    # One file lets the listener compare without switching players. Gaps belong only to this demonstration.
    gap = np.zeros((sr, 2), dtype=np.float32)
    order = ["00-original-instrumental", *PROFILES]
    sequence = np.concatenate([part for name in order for part in (final_values[name], gap)])
    sequence_path = out / "listen-in-order.wav"
    sf.write(sequence_path, sequence, sr, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(sequence_path),
                    "-c:a", "libmp3lame", "-b:a", "192k", str(sequence_path.with_suffix(".mp3"))], check=True)
    sequence_path.unlink()
    manifest = {"main_workflow_changed": False, "youtube_uploaded": False, "singing_in_comparison": False,
                "source_samples": n, "sample_rate": sr, "source_seconds": n / sr,
                "common_target_lufs": target, "matching_method": "constant gain only, with true-peak headroom",
                "levels": levels, "profiles": reports,
                "listening_order": [{"file": name + ".mp3", "starts_at_seconds": i * (n / sr + 1)}
                                    for i, name in enumerate(order)],
                "no_automatic_winner": True,
                "note": "Waveform and harmonic metrics do not certify audible rearrangement or compatibility with a singer."}
    write_json(out / "comparison-report.json", manifest)
    (out / "README.txt").write_text(
        "INSTRUMENTAL-ONLY LISTENING TEST; NO LIVE WORKFLOW CHANGE\n\n"
        "00-original-instrumental.mp3: original UVR-separated accompaniment.\n"
        "01-preserve.mp3: current cover-nofsq mode at 0.80 source conditioning.\n"
        "02-cover-050.mp3: true cover mode at 0.50, new gospel-soul quartet prompt.\n"
        "03-cover-035.mp3: same new prompt, true cover mode at 0.35.\n\n"
        "listen-in-order.mp3 plays those four clips with one-second gaps.\n"
        "Every individual clip uses the same source timeline and constant-gain loudness matching.\n"
        "raw/ contains model outputs before vocal cleanup; cleaned/ contains the cleanup versions.\n"
        "The root MP3s are the cleaned versions. Faint vocal bleed can remain in separated audio.\n"
        "Equal MP3 sizes are expected from equal duration and bitrate; they do not mean equal audio.\n"
        "These are experiments, not approved replacements. Judge the instruments, bass movement, harmony and groove.\n"
        "No human musical-quality verdict or perfect singer alignment is claimed.\n", encoding="utf-8")
    print("INSTRUMENTAL_TRIAL_PACK_READY " + json.dumps({
        "seconds_per_clip": n / sr, "target_lufs": target,
        "profiles": {name: {"mode": report["request"]["task_type"],
                           "source_strength": report["request"]["audio_cover_strength"],
                           "raw_affinity": report["raw_metrics"]["source_harmony_affinity"],
                           "cleaned_affinity": report["cleaned_metrics"]["source_harmony_affinity"]}
                     for name, report in reports.items()},
        "main_workflow_changed": False}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    if args.package:
        package()
    elif args.profile:
        run_profile(args.profile)
    else:
        parser.error("Choose --profile or --package")
