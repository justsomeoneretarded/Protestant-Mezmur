"""Save the channel cursor (and used footage) only after every song in the batch succeeded."""
import argparse
import json
from pathlib import Path

from .state import Store

MAX_USED = 100000  # never reuse a background (a few MB of state at most)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="inputs/plan.json")
    p.add_argument("--backgrounds", default="bg", help="folder with per-song backgrounds.json files")
    p.add_argument("--learned", default="learned", help="folder with per-song learned lyrics")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    state = plan["pending_state"] if not plan["replay"] else plan["snapshot"]["state"]
    used = list(state.get("used_backgrounds", []))
    for f in sorted(Path(args.backgrounds).rglob("backgrounds.json")):
        for item in json.loads(f.read_text()):
            if item not in used:
                used.append(item)
    state["used_backgrounds"] = used[-MAX_USED:]
    # Lyrics verified in this batch join the library; channels that supplied them join the search list.
    files, channels = {}, list(state.get("lyrics_channels", []))
    for meta_file in sorted(Path(args.learned).rglob("learned.json")) if args.learned else []:
        meta = json.loads(meta_file.read_text())
        lyrics = meta_file.parent / "lyrics.txt"
        if lyrics.is_file() and meta.get("path"):
            files[meta["path"]] = lyrics.read_text(encoding="utf-8")
        if meta.get("channel") and meta["channel"] not in channels:
            channels.append(meta["channel"])
    state["lyrics_channels"] = channels[-200:]
    Store().save(plan["snapshot"], state, files)
    print(f"Lyrics library: +{len(files)} song(s); learned channels: {len(channels)}", flush=True)
    print(f"Saved progress: {len(plan['ids'])} song(s), replay={plan['replay']}, {len(used)} used clips", flush=True)


if __name__ == "__main__":
    main()
