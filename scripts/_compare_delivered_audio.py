"""Read-only comparison of the exact delivered listening files; no generation or publishing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path("kit")
paths = {name: next(root.rglob(name)) for name in
         ("comparison-original.wav", "comparison-production-ai.wav",
          "comparison-original.mp3", "comparison-production-ai.mp3")}
files = {}
for name, path in paths.items():
    files[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
x, sx = sf.read(paths["comparison-original.wav"], dtype="float64", always_2d=True)
y, sy = sf.read(paths["comparison-production-ai.wav"], dtype="float64", always_2d=True)
assert sx == sy and x.shape == y.shape
flat_x, flat_y = x.ravel(), y.ravel()
alpha = float(np.dot(flat_x, flat_y) / np.dot(flat_x, flat_x))
residual = y - alpha * x
rms = lambda a: float(np.sqrt(np.mean(a * a)))
report = {"files": files, "samplerate": sx, "frames": len(x), "channels": x.shape[1],
          "pcm_samples_identical": bool(np.array_equal(x, y)),
          "pearson_correlation": float(np.corrcoef(flat_x, flat_y)[0, 1]),
          "best_constant_gain": alpha,
          "direct_difference_rms": rms(y - x),
          "gain_matched_residual_db_relative_to_ai_mix": float(20 * np.log10(max(rms(residual), 1e-15) / rms(y))),
          "largest_absolute_sample_difference": float(np.max(np.abs(y - x)))}
candidate_report = next(root.rglob("candidate-report.json"))
report["candidate_selection"] = json.loads(candidate_report.read_text())
print("AUDIO_COMPARISON_REPORT " + json.dumps(report), flush=True)
Path("comparison-report.json").write_text(json.dumps(report, indent=2) + "\n")
