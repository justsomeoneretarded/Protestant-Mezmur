"""Vocal isolation for recognition and the explicitly enabled backing-arrangement mode."""
import os
import re
import subprocess
from pathlib import Path

DEFAULT_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
FALLBACKS = ["UVR_MDXNET_KARA_2.onnx", "htdemucs_ft.yaml"]


def _stem_role(path):
    name = path.stem.casefold()
    if re.search(r"instrumental|accompaniment|no[\W_]*vocals", name):
        return "instrumental"
    for role in ("vocals", "drums", "bass", "other", "guitar", "piano", "strings", "synth"):
        if re.search(r"(?<![a-z])" + role + r"(?![a-z])", name):
            return role
    return ""


def instrumental_stem(vocals, target):
    """Keep the model's paired instrumental, or combine its non-vocal Demucs stems, in float WAV."""
    import numpy as np
    import soundfile as sf

    vocals, target = Path(vocals), Path(target)
    if target.resolve() == vocals.resolve():
        raise ValueError("Instrumental reference must not overwrite the vocal stem")
    candidates = [p for p in sorted(vocals.parent.glob("*.wav"))
                  if p.resolve() not in (vocals.resolve(), target.resolve())
                  and _stem_role(p) in ("instrumental", "drums", "bass", "other", "guitar", "piano", "strings", "synth")]
    paired = [p for p in candidates if _stem_role(p) == "instrumental"]
    candidates = paired[:1] if paired else candidates
    if not candidates:
        raise RuntimeError("The separator did not produce a paired instrumental or non-vocal stems")
    vocal_info = sf.info(vocals)
    mixed = None
    for path in candidates:
        info = sf.info(path)
        if info.subtype not in ("FLOAT", "DOUBLE", "PCM_24", "PCM_32"):
            raise RuntimeError("Instrumental separation needs high-precision stem export")
        if info.samplerate != vocal_info.samplerate or abs(info.frames - vocal_info.frames) > info.samplerate * 0.15:
            raise RuntimeError("Instrumental and vocal stem timelines do not match")
        values, _ = sf.read(path, dtype="float32", always_2d=True)
        if values.shape[1] not in (1, 2) or not np.isfinite(values).all():
            raise RuntimeError("The separated instrumental has invalid audio samples")
        if values.shape[1] == 1:
            values = np.repeat(values, 2, axis=1)
        values = np.pad(values[:vocal_info.frames],
                        ((0, max(0, vocal_info.frames - len(values))), (0, 0)))
        mixed = values.astype(np.float64) if mixed is None else mixed + values
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(target, mixed.astype(np.float32), vocal_info.samplerate, subtype="FLOAT")
    return target


def _run(wav, out, model, log, high_precision=False):
    cmd = [os.environ.get("AUDIO_SEPARATOR_BIN", "audio-separator"), str(wav), "-m", model, "--output_dir", str(out),
           "--model_file_dir", os.environ.get("MEZMUR_MODEL_CACHE", ".model-cache") + "/separator",
           "--output_format", "WAV"]
    if not high_precision:
        cmd += ["--single_stem", "Vocals"]
    if high_precision:
        cmd += ["--use_soundfile", "--normalization", "1.0", "--amplification", "0.0"]
    if "roformer" in model.lower() and not high_precision:
        cmd += ["--mdxc_overlap", "2"]  # default 8 is ~4x slower on CPU for a tiny quality gain
    code = subprocess.run(cmd).returncode
    vocals = sorted(p for p in out.glob("*.wav") if _stem_role(p) == "vocals")
    if code == 0 and vocals:
        log(f"  vocals isolated with {model}", flush=True)
        return vocals[0]
    log(f"  separator {model} failed (exit {code})", flush=True)
    return None


def separate(audio, folder, model=None, log=print, require_vocals=False):
    """Return vocals, allowing full-mix fallback only when a separated stem is not required."""
    model = model or os.environ.get("MEZMUR_SEPARATOR") or DEFAULT_MODEL
    if require_vocals and model == "none":
        raise ValueError("Music rebuilding requires vocal isolation; choose a separator other than none")
    folder = Path(folder)
    wav = folder / "sep_input.wav"
    codec = ["-c:a", "pcm_f32le"] if require_vocals else []
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio), "-ar", "44100", "-ac", "2",
                    *codec, str(wav)], check=True)
    if model == "none":
        return wav
    models = [model] if require_vocals else [model] + [m for m in FALLBACKS if m != model]
    for i, name in enumerate(models):
        out = folder / f"stems{i}"
        out.mkdir(parents=True, exist_ok=True)
        vocals = _run(wav, out, name, log, high_precision=require_vocals)
        if vocals:
            return vocals
    if require_vocals:
        raise RuntimeError("Vocal isolation failed; refusing to rebuild music over the original full mix")
    log("  WARNING: vocal separation unavailable; using the full mix", flush=True)
    return wav
