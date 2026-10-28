"""Durable cursor on a separate branch; optimistic writes prevent lost progress."""
import argparse
import base64
import json
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import required, write_json

BRANCH = "mezmur-state"


class Store:
    def __init__(self):
        self.root = f"https://api.github.com/repos/{required('GITHUB_REPOSITORY')}"
        self.http = requests.Session()
        self.http.headers.update({
            "Authorization": f"Bearer {required('GH_TOKEN')}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.http.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )))

    def call(self, method, path, **kwargs):
        response = self.http.request(method, self.root + path, timeout=60, **kwargs)
        if response.status_code != 404:
            response.raise_for_status()
        return response

    def load(self):
        branch = self.call("GET", f"/git/ref/heads/{BRANCH}")
        if branch.status_code == 404:
            # Distinguish a missing branch from inaccessible repository credentials.
            self.call("GET", "").raise_for_status()
            return {"head": None, "state": {"version": 1, "channels": {}}}
        head = branch.json()["object"]["sha"]
        response = self.call("GET", "/contents/state.json", params={"ref": head})
        response.raise_for_status()
        state = json.loads(base64.b64decode(response.json()["content"]))
        if state.get("version") != 1 or not isinstance(state.get("channels"), dict):
            raise ValueError("Unsupported or corrupt saved state; refusing to reset progress")
        return {"head": head, "state": state}

    def save(self, snapshot, state, files=None):
        """Write state.json (and any extra text files, e.g. the lyrics library) on top of the branch."""
        current = self.load()
        if current["head"] != snapshot["head"]:
            raise RuntimeError("Progress changed during this run; refusing to overwrite it")
        entries = [{"path": "state.json", "mode": "100644", "type": "blob",
                    "content": json.dumps(state, ensure_ascii=False, indent=2) + "\n"}]
        for path, content in (files or {}).items():
            entries.append({"path": path, "mode": "100644", "type": "blob", "content": content})
        body = {"tree": entries}
        if snapshot["head"]:  # keep every file already on the branch (the library grows over time)
            body["base_tree"] = self.call("GET", f"/git/commits/{snapshot['head']}").json()["tree"]["sha"]
        tree = self.call("POST", "/git/trees", json=body).json()["sha"]
        parents = [snapshot["head"]] if snapshot["head"] else []
        commit = self.call("POST", "/git/commits", json={
            "message": f"Record completed mezmur batch {os.environ.get('GITHUB_RUN_ID', 'local')}",
            "tree": tree, "parents": parents,
        }).json()["sha"]
        if parents:
            # GitHub rejects a non-fast-forward if another writer advanced the branch.
            response = self.call("PATCH", f"/git/refs/heads/{BRANCH}", json={"sha": commit, "force": False})
        else:
            response = self.call("POST", "/git/refs", json={"ref": f"refs/heads/{BRANCH}", "sha": commit})
        response.raise_for_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["load", "save"])
    args = parser.parse_args()
    folder = Path(".runtime")
    store = Store()
    if args.operation == "load":
        write_json(folder / "snapshot.json", store.load())
    else:
        pending = folder / "pending.json"
        if pending.exists():
            store.save(json.loads((folder / "snapshot.json").read_text()), json.loads(pending.read_text()))


if __name__ == "__main__":
    main()
