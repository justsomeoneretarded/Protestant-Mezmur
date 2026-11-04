"""Install a pinned, CPU-capable local ACE-Step engine and checksum-verified public weights."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests

ENGINE_REPO = "https://github.com/ServeurpersoCom/acestep.cpp.git"
ENGINE_COMMIT = "de858d8e1a94f0e1939cf9bdd8ff7d9631953077"
MODEL_REPO = "Serveurperso/ACE-Step-1.5-GGUF"
MODEL_REVISION = "666ac70204440867d8c01ba4b119cc79c95b370a"
MODEL_FILES = {
    "Qwen3-Embedding-0.6B-Q8_0.gguf": "972f23255e46adfe744a0eb9a0039f3c63988f65753b0968d776e8b27168c321",
    "acestep-v15-turbo-Q8_0.gguf": "288f708a61cfc241013a98a62f98ba331f83fe34d0d3559acdd9b0f6a2f7cd6b",
    "vae-BF16.gguf": "0599862ac5d15cd308e1d2e368373aea6c02e25ebd1737ad4a4562a0901b0ef8",
}


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def install(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    source, models = root / "source", root / "models"
    models.mkdir(exist_ok=True)
    for program in ("git", "cmake", "g++"):
        if not shutil.which(program):
            raise RuntimeError(f"AI backing setup requires {program}")
    if not (source / ".git").is_dir():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", ENGINE_REPO, str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", ENGINE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", ENGINE_COMMIT], check=True)
    subprocess.run(["git", "-C", str(source), "submodule", "update", "--init", "--depth=1"], check=True)
    build = source / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_BLAS=ON", "-DGGML_NATIVE=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "ace-synth", "-j",
                    str(min(4, os.cpu_count() or 2))], check=True)
    manifest = {"engine_commit": ENGINE_COMMIT, "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION, "files": {}}
    for name, expected in MODEL_FILES.items():
        path = models / name
        if not path.is_file() or sha256(path) != expected:
            for attempt in range(3):
                partial = path.with_suffix(".partial")
                try:
                    with requests.get(f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}",
                                      stream=True, timeout=(30, 300)) as download:
                        download.raise_for_status()
                        with partial.open("wb") as handle:
                            for chunk in download.iter_content(8 * 1024 * 1024):
                                handle.write(chunk)
                    if sha256(partial) != expected:
                        raise RuntimeError(f"Model checksum failed: {name}")
                    partial.replace(path)
                    break
                except (requests.RequestException, OSError, RuntimeError):
                    partial.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
        manifest["files"][name] = {"sha256": expected, "bytes": path.stat().st_size}
        print(f"Verified AI weights: {name}", flush=True)
    license_response = requests.get(
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/LICENSE", timeout=60)
    license_response.raise_for_status()
    (root / "model-license.txt").write_text(license_response.text, encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("AI_ENGINE_READY " + json.dumps(manifest), flush=True)
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as handle:
            handle.write(f"MEZMUR_AI_ROOT={root}\n")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".ai-backing"))
    install(parser.parse_args().root)
