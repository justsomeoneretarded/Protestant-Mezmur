import base64
import json

import pytest
import responses

from mezmur.state import Store

ROOT = "https://api.github.com/repos/owner/repo"


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    return Store()


@responses.activate
def test_new_branch_is_empty_state_only_if_repository_accessible(store):
    responses.get(ROOT + "/git/ref/heads/mezmur-state", status=404)
    responses.get(ROOT, json={"id": 123})
    assert store.load() == {"head": None, "state": {"version": 1, "channels": {}}}


@responses.activate
def test_missing_auth_does_not_reset_progress(store):
    responses.get(ROOT + "/git/ref/heads/mezmur-state", status=404)
    responses.get(ROOT, status=404)
    with pytest.raises(Exception):
        store.load()


def saved_state(head="head-one"):
    responses.get(ROOT + "/git/ref/heads/mezmur-state", json={"object": {"sha": head}})
    state = {"version": 1, "channels": {"42": {"last_message_id": 12}}}
    responses.get(ROOT + "/contents/state.json", json={
        "content": base64.b64encode(json.dumps(state).encode()).decode(),
    })
    return state


@responses.activate
def test_changed_head_refuses_to_overwrite(store):
    state = saved_state("someone-else")
    with pytest.raises(RuntimeError, match="Progress changed"):
        store.save({"head": "old", "state": state}, state)
    assert len(responses.calls) == 2


@responses.activate
def test_saved_state_moves_branch_without_force(store):
    state = saved_state()
    responses.post(ROOT + "/git/trees", json={"sha": "tree"})
    responses.post(ROOT + "/git/commits", json={"sha": "new"})
    responses.patch(ROOT + "/git/refs/heads/mezmur-state", json={"object": {"sha": "new"}})
    store.save({"head": "head-one", "state": state}, state)
    commit = json.loads(responses.calls[-2].request.body)
    update = json.loads(responses.calls[-1].request.body)
    assert commit["parents"] == ["head-one"]
    assert update == {"sha": "new", "force": False}
