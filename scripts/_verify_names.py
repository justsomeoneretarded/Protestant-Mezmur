import json
from pathlib import Path
from mezmur import audio, names
from mezmur.lyrics_sources import _queries

for folder in sorted(Path("latest-inputs").glob("song-*")):
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    tags = audio.tags(next(folder.glob("audio.*")))
    record = names.resolve({**info, "tags": tags})
    print("ACTUAL_NAMES", json.dumps({"info": info, "tags": tags, "resolved": record,
          "queries": _queries(record)}, ensure_ascii=False), flush=True)
