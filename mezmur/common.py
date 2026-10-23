import json
import math
import os
from pathlib import Path
import subprocess


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: list[str], **kwargs) -> None:
    subprocess.run(args, check=True, **kwargs)


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    seconds = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Audio has invalid duration")
    return seconds
