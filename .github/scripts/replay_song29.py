"""Temporary CI replay: real song 29 audio, real source page and model, no publishing."""
import json
import os
from pathlib import Path

from mezmur import names, song, web
from mezmur.common import write_json
from mezmur.lyrics_sources import _candidate, _page_candidates, dedupe
from mezmur.lyrics_text import strip_header

OUT = Path(".runtime/song29-check")
SOURCE = "https://mezmuroch.com/lyrics/kalkidan-tilahun/nefsen-befiteh"
captured = {}
choose = song.choose_lyrics
complete = song.complete_with_lyrics


class CaptionsReady(Exception):
    pass


def candidates(record, mid, *args, **kwargs):
    assert mid == 29
    response = web.get(SOURCE)
    if response is None or response.status_code != 200:
        raise RuntimeError("The original lyric source could not be fetched with TLS verification")
    response.encoding = "utf-8"
    rows = [_candidate("web:mezmuroch.com", SOURCE, strip_header(stanzas, record))
            for stanzas in _page_candidates(SOURCE, response.text, record)]
    rows = dedupe([row for row in rows if row["stanzas"]])
    if not rows:
        raise RuntimeError("No lyric block could be parsed from the original source")
    return rows, [], {}, {"replay_source": SOURCE, "candidates": len(rows)}


def capture_choose(*args, **kwargs):
    result = choose(*args, **kwargs)
    captured["candidates"] = result[1]
    if result[0]:
        captured["verification"] = result[0][3]["verification"]
    return result


def capture_complete(*args, **kwargs):
    result = complete(*args, **kwargs)
    captured["completion"] = result[1]
    return result


def stop_before_photos(*args, **kwargs):
    raise CaptionsReady


song.gather = candidates
song.choose_lyrics = capture_choose
song.complete_with_lyrics = capture_complete
song.photo_candidates = stop_before_photos
song.web.youtube_search = lambda *args, **kwargs: []
os.environ.pop("MEZMUR_DEBUG", None)
OUT.mkdir(parents=True, exist_ok=True)
try:
    result = song.produce("inputs/song-29", OUT, ".runtime/work",
                          separator="UVR-MDX-NET-Voc_FT.onnx", height=720)
except CaptionsReady:
    result = {"mode": "captions-ready"}
captured["result"] = result
if (OUT / "words.json").is_file():
    words = json.loads((OUT / "words.json").read_text())
    captured["timed_words"] = len(words)
    captured["first_word_seconds"] = words[0]["start"]
    captured["last_word_seconds"] = words[-1]["end"]
write_json(OUT / "alignment-report.json", captured)
print(json.dumps({key: value for key, value in captured.items() if key != "completion"}, indent=2), flush=True)
assert result["mode"] == "captions-ready", result
assert captured["timed_words"] > 0
assert (OUT / "captions/am.srt").stat().st_size > 0
