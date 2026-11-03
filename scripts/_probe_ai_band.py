"""Real-song CPU generation probe; never accesses publishing secrets or progress state."""
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from mezmur import separate
from mezmur.arrangement import mix_tracks
from mezmur.common import write_json

out, work = Path("verification/ai-band"), Path(".runtime/ai-probe")
out.mkdir(parents=True, exist_ok=True)
work.mkdir(parents=True, exist_ok=True)
root = Path(os.environ["MEZMUR_AI_ROOT"])
source = out / "comparison-original.wav"
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(next(Path("old-kit").rglob("comparison-original.mp3"))),
                "-t", "16.713", "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le", str(source)], check=True)
voice = separate.separate(source, work, require_vocals=True)
reference = separate.instrumental_stem(voice, out / "reference.wav")
voice_hash = hashlib.sha256(voice.read_bytes()).hexdigest()
for strength in (0.65, 0.90):
    request = out / f"cover-{strength}.json"
    write_json(request, {"task_type": "cover-nofsq", "lyrics": "[Instrumental]",
        "caption": "Instrumental Ethiopian gospel worship band, expressive warm melodic bass guitar, soulful piano, "
                   "tasteful electric guitar answers, warm spacious pads, tight restrained acoustic drums. "
                   "Follow the source chord movement and groove, breathing phrases and natural transitions. "
                   "No singing, no humming, no constant acoustic guitar strumming, no metronomic loop.",
        "duration": 16.713, "seed": 40, "inference_steps": 8, "synth_batch_size": 1,
        "audio_cover_strength": strength, "cover_noise_strength": 0.0,
        "output_format": "wav32", "peak_clip": 0, "synth_model": "acestep-v15-turbo-Q8_0"})
    started = time.monotonic()
    command = [str(root / "source/build/ace-synth"), "--models", str(root / "models"),
               "--device", "CPU", "--src-audio", str(reference), "--request", str(request)]
    with (out / f"engine-{strength}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    generated = request.with_name(request.stem + "0.wav")
    audio, sr = sf.read(generated, dtype="float32", always_2d=True)
    assert sr == 48000 and audio.shape[1] == 2 and np.isfinite(audio).all()
    from scipy.signal import resample_poly
    vocals, voice_sr = sf.read(voice, dtype="float32", always_2d=True)
    n = sf.info(source).frames
    backing = resample_poly(audio, voice_sr, sr)
    backing = np.pad(backing[:n], ((0, max(0, n - len(backing))), (0, 0)))
    vocals = vocals[:n]
    mixed, levels = mix_tracks(vocals, backing, voice_sr)
    master = out / f"comparison-ai-{strength}.wav"
    sf.write(master, mixed, voice_sr, subtype="FLOAT")
    assert sf.info(master).frames == n
    report = {"strength": strength, "generated_seconds": len(audio) / sr,
              "duration_seconds": n / voice_sr, "elapsed_seconds": time.monotonic() - started,
              "peak_memory_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              **levels}
    write_json(out / f"probe-{strength}.json", report)
    print("AI_REAL_CPU_PASSED " + json.dumps(report), flush=True)
assert hashlib.sha256(voice.read_bytes()).hexdigest() == voice_hash
