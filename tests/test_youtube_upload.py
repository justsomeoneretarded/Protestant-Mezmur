import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from mezmur import upload
from mezmur.common import write_json

CHANNEL = "UC" + "a" * 22
VIDEO = "abcde123456"


class MemoryStore:
    def __init__(self):
        self.head, self.writes = 0, 0
        self.state = {"version": 1, "channels": {"42": {"last_message_id": 22}},
                      "used_backgrounds": ["old"], "lyrics_channels": ["saved"]}
        self.fail_write = None

    def load(self):
        return {"head": self.head, "state": copy.deepcopy(self.state)}

    def save(self, snapshot, state):
        self.writes += 1
        if self.writes == self.fail_write:
            raise RuntimeError("Simulated unavailable GitHub checkpoint")
        assert snapshot["head"] == self.head
        self.state = copy.deepcopy(state)
        self.head += 1


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "song-22"
    root.mkdir()
    write_json(root / "metadata.json", {"message_id": 22, "telegram_channel_id": "42",
               "lyrics_mode": "lyrics-aligned", "review": {"needs_review": False, "reasons": []}})
    write_json(root / "youtube.json", {"recommended_title": "A worship song",
               "tags": ["amharic mezmur", "worship"], "review": {"needs_review": False, "reasons": []}})
    (root / "description.txt").write_text("Worship lyrics\n#AmharicMezmur", encoding="utf-8")
    (root / "video.mp4").write_bytes(b"video-payload")
    for name in upload.THUMBNAILS.values():
        (root / name).write_bytes(b"jpeg-data")
    (root / "captions").mkdir()
    for relative, _ in upload.CAPTIONS:
        (root / relative).write_text("1\n00:00:01,000 --> 00:00:02,000\nLyrics\n")
    return root


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(upload, "media", lambda path, mime, resumable=False: (path.name, mime, resumable))
    api = MagicMock()
    state = {"video": None, "tracks": [], "insert_count": 0, "next_error": None, "caption_error": False}

    def execute(value):
        request = MagicMock()
        request.execute.return_value = copy.deepcopy(value)
        return request

    def insert(**kwargs):
        state["insert_count"] += 1
        state["insert_args"] = kwargs
        state["video"] = {"id": VIDEO, "snippet": dict(kwargs["body"]["snippet"], channelId=CHANNEL),
                          "status": copy.deepcopy(kwargs["body"]["status"])}
        request = MagicMock()

        def next_chunk(**kwargs):
            if state["next_error"]:
                raise state["next_error"]
            return None, {"id": VIDEO}
        request.next_chunk.side_effect = next_chunk
        return request

    def list_videos(**kwargs):
        return execute({"items": [state["video"]] if state["video"] else []})

    def update(**kwargs):
        state["video"]["status"].update(kwargs["body"]["status"])
        return execute(state["video"])

    def insert_caption(**kwargs):
        request = MagicMock()

        def run(**kwargs_execute):
            if state["caption_error"]:
                raise OSError("Simulated caption transport failure")
            caption = {"id": "caption-" + str(len(state["tracks"])), **kwargs["body"]}
            state["tracks"].append(caption)
            return copy.deepcopy(caption)
        request.execute.side_effect = run
        return request

    api.channels.return_value.list.return_value = execute({"items": [{"id": CHANNEL}]})
    api.videos.return_value.insert.side_effect = insert
    api.videos.return_value.list.side_effect = list_videos
    api.videos.return_value.update.side_effect = update
    api.thumbnails.return_value.set.side_effect = lambda **k: execute({"items": [{}]})
    api.captions.return_value.list.side_effect = lambda **k: execute({"items": state["tracks"]})
    api.captions.return_value.insert.side_effect = insert_caption
    api.test_state = state
    return api


def test_private_upload_sets_metadata_thumbnail_and_both_caption_tracks(folder, service):
    store = MemoryStore()
    result = upload.upload_song(upload.load_kit(folder, "d"), service, store, CHANNEL)
    body = service.test_state["insert_args"]["body"]
    assert body["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False}
    assert body["snippet"]["categoryId"] == "10"
    assert body["snippet"]["defaultLanguage"] == body["snippet"]["defaultAudioLanguage"] == "am"
    assert service.test_state["insert_args"]["notifySubscribers"] is False
    assert service.test_state["insert_args"]["media_body"][2] is True
    assert service.thumbnails.return_value.set.call_args.kwargs["media_body"][0] == "thumbnail-d.jpg"
    assert {c["snippet"]["name"] for c in service.test_state["tracks"]} == {"Amharic", "Amharic (Latin letters)"}
    assert all(c["snippet"]["language"] == "am" for c in service.test_state["tracks"])
    assert result["status"] == "complete" and result["privacy"] == "private"
    assert json.loads((folder / "youtube-upload.json").read_text())["video_id"] == VIDEO
    assert store.state["channels"]["42"]["last_message_id"] == 22
    assert store.state["used_backgrounds"] == ["old"] and store.state["lyrics_channels"] == ["saved"]
    assert result["source_marker"] in body["snippet"]["tags"]


@pytest.mark.parametrize("visibility", ["unlisted", "public"])
def test_visibility_changes_only_after_assets_are_uploaded(folder, service, visibility):
    result = upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL, visibility)
    assert result["privacy"] == visibility
    assert len(service.test_state["tracks"]) == 2
    assert service.test_state["insert_args"]["body"]["status"]["privacyStatus"] == "private"
    service.videos.return_value.update.assert_called_once()


def test_review_warnings_hold_public_request_private_even_if_boolean_is_stale(folder, service):
    meta = json.loads((folder / "metadata.json").read_text())
    meta["review"]["reasons"].append("Check the artist")
    write_json(folder / "metadata.json", meta)
    result = upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL, "public")
    assert result["held_for_review"] and result["privacy"] == "private"
    service.videos.return_value.update.assert_not_called()


def test_completed_song_is_never_inserted_again_on_a_replay(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    first = upload.upload_song(kit, service, store, CHANNEL)
    second = upload.upload_song(kit, service, store, CHANNEL)
    assert second["video_id"] == first["video_id"]
    assert service.test_state["insert_count"] == 1
    assert service.captions.return_value.insert.call_count == 2


def test_partial_failure_resumes_the_same_video_and_finishes_assets(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["caption_error"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    pending = next(iter(store.state["youtube_uploads"].values()))
    assert pending["status"] == "uploaded" and pending["thumbnail_done"]
    service.test_state["caption_error"] = False
    result = upload.upload_song(kit, service, store, CHANNEL, visibility="public")
    assert result["status"] == "complete" and result["privacy"] == "private"
    assert service.test_state["insert_count"] == 1
    assert service.thumbnails.return_value.set.call_count == 1


def test_unknown_upload_outcome_blocks_a_fresh_insert(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["next_error"] = OSError("Network disconnected")
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    assert next(iter(store.state["youtube_uploads"].values()))["status"] == "uploading"
    with pytest.raises(upload.UploadError, match="uncertain"):
        upload.upload_song(kit, service, store, CHANNEL)
    assert service.test_state["insert_count"] == 1


def test_recover_returned_id_after_checkpoint_failure_without_reuploading(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    store.fail_write = 2
    with pytest.raises(RuntimeError):
        upload.upload_song(kit, service, store, CHANNEL)
    assert json.loads((folder / "youtube-upload.json").read_text())["video_id"] == VIDEO
    result = upload.upload_song(kit, service, store, CHANNEL, recover_video_id=VIDEO)
    assert result["status"] == "complete"
    assert service.test_state["insert_count"] == 1


def test_recovery_rejects_a_video_from_another_source(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["next_error"] = OSError()
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    service.test_state["video"]["snippet"]["tags"] = ["wrong-song"]
    with pytest.raises(upload.UploadError, match="does not belong"):
        upload.upload_song(kit, service, store, CHANNEL, recover_video_id=VIDEO)


def test_unfinished_upload_rejects_regenerated_video_bytes(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["caption_error"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    (folder / "video.mp4").write_bytes(b"different-render")
    with pytest.raises(upload.UploadError, match="different video bytes"):
        upload.upload_song(kit, service, store, CHANNEL)
    assert service.test_state["insert_count"] == 1


def test_existing_caption_is_reconciled_after_an_uncertain_caption_response(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    # The caption POST succeeds but its following GitHub checkpoint fails.
    store.fail_write = 4
    with pytest.raises(RuntimeError):
        upload.upload_song(kit, service, store, CHANNEL)
    assert len(service.test_state["tracks"]) == 1
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["status"] == "complete" and len(service.test_state["tracks"]) == 2
    assert service.captions.return_value.insert.call_count == 2


def test_oauth_channel_mismatch_prevents_upload(service):
    with pytest.raises(upload.UploadError, match="different"):
        upload.verify_channel(service, "UC" + "b" * 22)
    service.videos.return_value.insert.assert_not_called()


@pytest.mark.parametrize("value", ["@handle", "", "UCshort"])
def test_invalid_destination_channel_id_is_rejected(service, value):
    with pytest.raises(upload.UploadError, match="UC"):
        upload.verify_channel(service, value)


@pytest.mark.parametrize("missing", ["youtube.json", "metadata.json", "video.mp4", "captions/am.srt", "thumbnail.jpg"])
def test_missing_or_legacy_kits_fail_before_upload(folder, missing):
    (folder / missing).unlink()
    with pytest.raises(upload.UploadError):
        upload.load_kit(folder)


@pytest.mark.parametrize("change", [
    {"telegram_channel_id": None}, {"message_id": 0}, {"lyrics_mode": "transcription"}, {"review": None},
])
def test_invalid_song_identity_or_unverified_lyrics_are_rejected(folder, change):
    meta = json.loads((folder / "metadata.json").read_text())
    write_json(folder / "metadata.json", meta | change)
    with pytest.raises(upload.UploadError):
        upload.load_kit(folder)


def test_metadata_limits_use_utf8_bytes_not_characters(folder):
    (folder / "description.txt").write_text("\u12a0" * 2000, encoding="utf-8")
    with pytest.raises(upload.UploadError, match="UTF-8"):
        upload.load_kit(folder)


def test_marker_and_keyword_tags_always_fit_the_api_limit():
    result = upload.upload_tags(["a long keyword phrase " + str(i) for i in range(100)], "marker")
    cost = sum(len(t) + (2 if " " in t else 0) for t in result) + len(result) - 1
    assert result[0] == "marker" and cost <= 500


def test_checkpoint_refuses_to_overwrite_another_writer():
    store = MemoryStore()
    store.state["youtube_uploads"] = {"key": {"video_id": "other"}}
    with pytest.raises(upload.UploadError, match="concurrently"):
        upload.save_record(store, "key", {"video_id": VIDEO}, None)
    assert store.writes == 0


def test_safe_error_never_exposes_http_session_urls():
    error = RuntimeError("https://upload.example/?upload_id=TOP_SECRET")
    assert "TOP_SECRET" not in upload.safe_error(error)
    assert "https://" not in upload.safe_error(error)


def test_cli_dry_run_needs_no_google_secret_or_github_credentials(folder, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["mezmur-upload", "--folder", str(folder), "--dry-run"])
    monkeypatch.setattr(upload, "youtube_client", lambda: pytest.fail("Dry run tried OAuth"))
    monkeypatch.setattr(upload, "Store", lambda: pytest.fail("Dry run tried GitHub"))
    upload.main()
    assert not (folder / "youtube-upload.json").exists()


def test_cli_records_the_video_link_in_actions_summary(folder, service, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("YOUTUBE_CHANNEL_ID", CHANNEL)
    monkeypatch.setattr(sys, "argv", ["mezmur-upload", "--root", str(folder.parent)])
    monkeypatch.setattr(upload, "youtube_client", lambda: service)
    monkeypatch.setattr(upload, "Store", MemoryStore)
    upload.main()
    assert f"https://youtu.be/{VIDEO}" in summary.read_text()


def test_generation_upload_is_opt_in_and_after_progress():
    from pathlib import Path

    doc = yaml.safe_load(Path(".github/workflows/generate.yml").read_text())
    events = doc.get("on", doc.get(True))
    inputs = events["workflow_dispatch"]["inputs"]
    assert len(inputs) <= 10
    assert inputs["upload_youtube"]["default"] is False
    assert doc["jobs"]["publish"]["needs"] == ["fetch", "song", "finalize"]
    assert "upload_youtube" in doc["jobs"]["publish"]["if"]
    assert "YOUTUBE_OAUTH_JSON" not in str(doc["jobs"]["song"].get("env", {}))
    artifacts = next(s for s in doc["jobs"]["song"]["steps"] if s.get("name", "").startswith("Upload the YouTube kit"))
    assert "/metadata.json" in artifacts["with"]["path"] and "/youtube.json" in artifacts["with"]["path"]


def test_publish_workflow_guards_artifacts_and_keeps_secrets_out_of_receipts():
    from pathlib import Path

    doc = yaml.safe_load(Path(".github/workflows/publish.yml").read_text())
    job = doc["jobs"]["publish"]
    assert "refs/heads/main" in job["if"]
    assert "from_generation" in doc["concurrency"]["group"]
    checks = next(s for s in job["steps"] if s.get("name") == "Validate source run and requested song")
    assert "head_repository" in checks["run"] and "head_branch" in checks["run"]
    assert ".github/workflows/generate.yml" in checks["run"]
    receipt = job["steps"][-1]
    assert receipt["with"]["path"] == "uploads/**/youtube-upload*.json"
    assert receipt["if"] == "always()"


def test_actual_google_sdk_accepts_upload_and_caption_request_shapes(folder):
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    api = build("youtube", "v3", developerKey="not-used", cache_discovery=False, static_discovery=True)
    request = api.videos().insert(part="snippet,status",
        body={"snippet": {"title": "Test", "categoryId": "10"}, "status": {"privacyStatus": "private"}},
        notifySubscribers=False, media_body=MediaFileUpload(str(folder / "video.mp4"), resumable=True))
    assert request.resumable is not None
    caption = api.captions().insert(part="snippet",
        body={"snippet": {"videoId": VIDEO, "language": "am", "name": "Amharic (Latin letters)", "isDraft": False}},
        media_body=MediaFileUpload(str(folder / "captions/am-Latn.srt"), mimetype="application/octet-stream"))
    assert "uploadType=media" in caption.uri or "uploadType=multipart" in caption.uri
    api.close()


def test_authorization_script_writes_only_a_private_refresh_credential_file(tmp_path, monkeypatch, capsys):
    import runpy
    from google_auth_oauthlib.flow import InstalledAppFlow
    import googleapiclient.discovery

    config = tmp_path / "client_secret_test.json"
    config.write_text(json.dumps({"installed": {"client_id": "client"}}))
    output = tmp_path / "credentials.json"
    credentials = SimpleNamespace(refresh_token="REFRESH_PRIVATE", client_id="client",
                                  client_secret="CLIENT_PRIVATE", token="ACCESS_PRIVATE")
    flow = MagicMock()
    flow.run_local_server.return_value = credentials
    make_flow = MagicMock(return_value=flow)
    monkeypatch.setattr(InstalledAppFlow, "from_client_config", make_flow)
    service = MagicMock()
    service.channels.return_value.list.return_value.execute.return_value = {
        "items": [{"id": CHANNEL, "snippet": {"title": "Test channel"}}]}
    monkeypatch.setattr(googleapiclient.discovery, "build", lambda *a, **k: service)
    monkeypatch.setattr(sys, "argv", ["youtube_authorize.py", "--client-secrets", str(config), "--output", str(output)])
    runpy.run_path("scripts/youtube_authorize.py", run_name="__main__")
    assert output.stat().st_mode & 0o777 == 0o600
    data = json.loads(output.read_text())
    assert data["refresh_token"] == "REFRESH_PRIVATE" and "token" not in data
    stdout = capsys.readouterr().out
    assert "REFRESH_PRIVATE" not in stdout and "CLIENT_PRIVATE" not in stdout and "ACCESS_PRIVATE" not in stdout
    assert flow.run_local_server.call_args.kwargs["host"] == "127.0.0.1"
    assert make_flow.call_args.kwargs["autogenerate_code_verifier"] is True


@pytest.mark.parametrize("raw", ["{}", "invalid JSON", '{"client_id": "x"}'])
def test_bad_oauth_configuration_has_a_useful_non_secret_error(monkeypatch, raw):
    monkeypatch.setenv("YOUTUBE_OAUTH_JSON", raw)
    with pytest.raises(upload.UploadError, match="YOUTUBE_OAUTH_JSON"):
        upload.youtube_client()
