"""Temporary read-only regression replay against the original saved Telegram inputs."""
import json
import sys
from pathlib import Path

from mezmur import audio, names, song
from mezmur.common import write_json
from mezmur.lyrics_sources import _queries, dedupe, from_telegram
from mezmur.lyrics_text import stanzas_text, strip_header

mid = int(sys.argv[1])
root = Path("inputs")
out = Path("verification") / f"song-{mid}"
out.mkdir(parents=True, exist_ok=True)

for folder in sorted(root.glob("song-*")):
    info = json.loads((folder / "info.json").read_text(encoding="utf-8"))
    source = next(folder.glob("audio.*"))
    record = names.resolve({**info, "tags": audio.tags(source)})
    print("NAME_CHECK", json.dumps({"id": info["message_id"], "raw_title": info.get("title"),
          "raw_performer": info.get("performer"), "record": record, "queries": _queries(record)},
          ensure_ascii=False), flush=True)
    if info["message_id"] == 32:
        assert record["artist_latin"] == "Agegnehu Yideg", record

evidence = {}
original_choose = song.choose_lyrics
original_complete = song.complete_with_lyrics


def saved_candidates(record, message_id, telegram_data, *args, **kwargs):
    candidates = from_telegram(telegram_data)
    for candidate in candidates:
        candidate["stanzas"] = strip_header(candidate["stanzas"], record)
        candidate["text"] = stanzas_text(candidate["stanzas"])
    candidates = dedupe([c for c in candidates if c["stanzas"]])
    return candidates, [], {}, {"saved Telegram candidates": len(candidates)}


def choose(*args, **kwargs):
    best, reports = original_choose(*args, **kwargs)
    evidence["candidates"] = reports
    if best:
        evidence.update(source=best[1]["source"], url=best[1]["url"],
                        alignment={k: v for k, v in best[3].items() if k != "phrase_details"})
    write_json(out / "evidence.json", evidence)
    return best, reports


def complete(*args, **kwargs):
    words, report = original_complete(*args, **kwargs)
    evidence["completion"] = report
    evidence["estimated_share"] = report["estimated_words"] / max(1, len(words))
    write_json(out / "evidence.json", evidence)
    return words, report


class CaptionsReady(Exception):
    pass


def stop_before_render(*args, **kwargs):
    assert (out / "captions" / "am.srt").stat().st_size > 0
    words = json.loads((out / "words.json").read_text())
    assert words and all(w["start"] >= 0 and w["end"] > w["start"] for w in words)
    evidence["timed_word_count"] = len(words)
    write_json(out / "evidence.json", evidence)
    raise CaptionsReady


song.gather = saved_candidates
song.youtube_titles = lambda *args: []
song.choose_lyrics = choose
song.complete_with_lyrics = complete
song.photo_candidates = stop_before_render

try:
    result = song.produce(root / f"song-{mid}", out, Path(".runtime") / f"verify-{mid}",
                          separator="UVR-MDX-NET-Voc_FT.onnx")
except CaptionsReady:
    print("CAPTIONS_READY", mid, json.dumps(evidence, ensure_ascii=False), flush=True)
else:
    raise AssertionError(f"Song {mid} did not reach subtitles: {result}")
