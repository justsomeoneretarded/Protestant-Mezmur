"""Markdown summary of one finished song for the Actions run page."""
import json
import sys
from pathlib import Path


def main():
    folder = Path(sys.argv[1])
    meta = json.loads((folder / "metadata.json").read_text()) if (folder / "metadata.json").is_file() else {}
    quality = json.loads((folder / "quality.json").read_text()) if (folder / "quality.json").is_file() else {}
    title = (folder / "title.txt").read_text().splitlines()[0] if (folder / "title.txt").is_file() else folder.name
    print(f"### {title}")
    skipped = folder / "skipped.json"
    if skipped.is_file():
        info = json.loads(skipped.read_text(encoding="utf-8"))
        print(f"- ⏭ **Not made:** {info.get('reason')}")
        print(f"- To make it: {info.get('fix')}")
        print()
        return
    if not meta:
        print("❌ This song failed — see the job log above.")
        return
    mode = meta.get("lyrics_mode")
    icon = "✅" if mode == "lyrics-aligned" else "⚠️"
    print(f"- {icon} Lyrics: **{mode}** — {meta.get('lyrics_source')}")
    align = quality.get("alignment") or {}
    if align:
        print(f"- Singing matched to lyrics: {align.get('matched_share', 0):.0%} of vocal time")
    print(f"- Singer photo: {meta.get('artist_photo') or 'none found'}")
    print(f"- Footage: {len(meta.get('backgrounds', []))} clip(s); tempo {meta.get('tempo_bpm')} BPM")
    print(f"- Timings (s): {meta.get('timings_seconds')}")
    for reason in (meta.get("review") or {}).get("reasons", []):
        print(f"- ⚠️ {reason}")
    print()


if __name__ == "__main__":
    main()
