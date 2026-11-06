"""External dispatch defaults, shared locks and trusted-source publishing."""
import json
from pathlib import Path

import pytest
import yaml


def test_external_and_manual_generation_default_to_one_song_and_share_the_lock():
    document = yaml.safe_load(Path(".github/workflows/generate.yml").read_text())
    events = document.get("on", document.get(True))
    assert "schedule" not in events
    for trigger in ("workflow_dispatch", "workflow_call"):
        assert events[trigger]["inputs"]["batch_size"]["default"] == "1"
        assert events[trigger]["inputs"]["upload_youtube"]["default"] is True
        assert "rebuild_music" not in events[trigger]["inputs"]
    assert events["workflow_dispatch"]["inputs"]["batch_size"]["options"][0] == "1"
    fetch = next(step for step in document["jobs"]["fetch"]["steps"] if step.get("id") == "fetch")
    assert "${{ inputs.batch_size || '1' }}" in fetch["run"]
    assert document["concurrency"] == {"group": "mezmur-generate", "cancel-in-progress": False}
    publisher = yaml.safe_load(Path(".github/workflows/publish.yml").read_text())
    assert "mezmur-generate" in publisher["concurrency"]["group"]
    assert publisher["concurrency"]["cancel-in-progress"] is False


def test_fetch_cli_defaults_to_one_song(monkeypatch):
    from mezmur import fetch

    calls = []

    async def capture(args):
        calls.append(args)

    monkeypatch.setattr("sys.argv", ["mezmur-fetch"])
    monkeypatch.setattr(fetch, "run", capture)
    fetch.main()
    assert calls[0].batch_size == 1 and calls[0].message_ids == ""


def validate_publish_source(tmp_path, monkeypatch, event, from_generation=True, **changes):
    repository = "justsomeoneretarded/Protestant-Mezmur"
    source = {"head_repository": {"full_name": repository}, "head_branch": "main",
              "path": ".github/workflows/generate.yml", "event": event, "status": "completed"}
    source.update(changes)
    (tmp_path / "source-run.json").write_text(json.dumps(source))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", repository)
    monkeypatch.setenv("FROM_GENERATION", "true" if from_generation else "false")
    document = yaml.safe_load(Path(".github/workflows/publish.yml").read_text())
    step = next(item for item in document["jobs"]["publish"]["steps"]
                if item.get("name") == "Validate source run and requested song")
    _, delimiter, code = step["run"].partition("python - <<'PY'\n")
    assert delimiter
    code, delimiter, _ = code.partition("\nPY")
    assert delimiter
    exec(compile(code, "publish-workflow-source-validation", "exec"), {})


@pytest.mark.parametrize("event", ["workflow_dispatch", "push", "workflow_call"])
@pytest.mark.parametrize("from_generation", [False, True])
def test_publisher_accepts_trusted_generation_events(tmp_path, monkeypatch, event, from_generation):
    validate_publish_source(tmp_path, monkeypatch, event, from_generation)


@pytest.mark.parametrize("changes", [
    {"head_repository": {"full_name": "other/repository"}},
    {"head_repository": None},
    {"head_branch": "untrusted-branch"},
    {"path": ".github/workflows/_full-song-trial.yml"},
    {"event": "pull_request"},
])
def test_publisher_still_rejects_untrusted_or_experimental_runs(tmp_path, monkeypatch, changes):
    event = changes.get("event", "workflow_dispatch")
    with pytest.raises(SystemExit, match="trusted main workflow"):
        validate_publish_source(tmp_path, monkeypatch, event,
                                **{key: value for key, value in changes.items() if key != "event"})


def test_only_parent_generation_can_publish_an_in_progress_external_run(tmp_path, monkeypatch):
    validate_publish_source(tmp_path, monkeypatch, "workflow_dispatch", status="in_progress")
    with pytest.raises(SystemExit, match="trusted main workflow"):
        validate_publish_source(tmp_path, monkeypatch, "workflow_dispatch", from_generation=False, status="in_progress")


def test_external_request_uses_main_one_new_song_and_upload_enabled():
    payload = json.loads(Path("docs/cron-dispatch.json").read_text())
    assert payload == {"ref": "main", "inputs": {"batch_size": "1", "message_ids": "", "upload_youtube": "true"}}
    document = yaml.safe_load(Path(".github/workflows/generate.yml").read_text())
    events = document.get("on", document.get(True))
    assert set(payload["inputs"]) <= set(events["workflow_dispatch"]["inputs"])
