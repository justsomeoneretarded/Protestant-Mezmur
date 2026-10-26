"""Save the channel cursor (and used footage) only after every song in the batch succeeded."""
import argparse
import json
from pathlib import Path

from .state import Store

MAX_USED = 4000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", default="inputs/plan.json")
    p.add_argument("--backgrounds", default="bg", help="folder with per-song backgrounds.json files")
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    state = plan["pending_state"] if not plan["replay"] else plan["snapshot"]["state"]
    used = list(state.get("used_backgrounds", []))
    for f in sorted(Path(args.backgrounds).rglob("backgrounds.json")):
        for item in json.loads(f.read_text()):
            if item not in used:
                used.append(item)
    state["used_backgrounds"] = used[-MAX_USED:]
    Store().save(plan["snapshot"], state)
    print(f"Saved progress: {len(plan['ids'])} song(s), replay={plan['replay']}, {len(used)} used clips", flush=True)


if __name__ == "__main__":
    main()
