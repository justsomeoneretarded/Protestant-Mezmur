"""Vocal isolation for the acoustic model. The final video always uses the original mix."""
import os
import subprocess
from pathlib import Path

DEFAULT_MODEL = "vocals_mel_band_roformer.ckpt"
FALLBACKS = ["UVR-MDX-NET-Voc_FT.onnx", "htdemucs_ft.yaml"]


def _run(wav, out, model, log):
    cmd = ["audio-separator", str(wav), "-m", model, "--output_dir", str(out), "--single_stem", "Vocals",
           "--model_file_dir", os.environ.get("MEZMUR_MODEL_CACHE", ".model-cache") + "/separator",
           "--output_format", "WAV"]
    if "roformer" in model.lower():
        cmd += ["--mdxc_overlap", "2"]  # default 8 is ~4x slower on CPU for a tiny quality gain
    code = subprocess.run(cmd).returncode
    vocals = sorted(out.glob("*[Vv]ocals*.wav"))
    if code == 0 and vocals:
        log(f"  vocals isolated with {model}", flush=True)
        return vocals[0]
    log(f"  separator {model} failed (exit {code})", flush=True)
    return None


def separate(audio, folder, model=None, log=print):
    """Return a vocal stem path; falls back to faster models, then to the full mix."""
    model = model or os.environ.get("MEZMUR_SEPARATOR") or DEFAULT_MODEL
    folder = Path(folder)
    wav = folder / "sep_input.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio), "-ar", "44100", "-ac", "2", str(wav)],
                   check=True)
    if model == "none":
        return wav
    for i, name in enumerate([model] + [m for m in FALLBACKS if m != model]):
        out = folder / f"stems{i}"
        out.mkdir(parents=True, exist_ok=True)
        vocals = _run(wav, out, name, log)
        if vocals:
            return vocals
    log("  WARNING: vocal separation unavailable; using the full mix", flush=True)
    return wav
