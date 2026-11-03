import copy
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from mezmur import upload
from mezmur.common import write_json

CHANNEL = "UC" + "a" * 22
VIDEO = "abcde123456"
PLAYLIST = "PL" + "b" * 32


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
        if snapshot["head"] != self.head:
            raise RuntimeError("Progress changed during this run")
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
    return root


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(upload, "media", lambda path, mime, resumable=False: (path.name, mime, resumable))
    api = MagicMock()
    monkeypatch.setattr(upload, "utc_now", lambda: datetime(2026, 11, 1, 12, tzinfo=timezone.utc))
    state = {"video": None, "videos": {}, "insertions": [], "insert_count": 0, "next_error": None,
             "thumbnail_error": False, "schedule_error": False, "reject_schedule": False,
             "playlists": [{"id": PLAYLIST, "snippet": {"title": "Mezmurs", "channelId": CHANNEL}}],
             "playlist_items": {}, "playlist_insert_count": 0, "playlist_error": False,
             "playlist_lost_response": False}

    def execute(value):
        request = MagicMock()
        request.execute.return_value = copy.deepcopy(value)
        return request

    def insert(**kwargs):
        state["insert_count"] += 1
        state["insert_args"] = kwargs
        state["insertions"].append(copy.deepcopy(kwargs))
        video_id = VIDEO if state["insert_count"] == 1 else f"video{state['insert_count']:06d}"
        state["video"] = {"id": video_id, "snippet": dict(kwargs["body"]["snippet"], channelId=CHANNEL),
                          "status": copy.deepcopy(kwargs["body"]["status"])}
        state["videos"][video_id] = state["video"]
        request = MagicMock()

        def next_chunk(**kwargs):
            if state["next_error"]:
                raise state["next_error"]
            return None, {"id": video_id}
        request.next_chunk.side_effect = next_chunk
        return request

    def list_videos(**kwargs):
        item = state["videos"].get(kwargs["id"])
        return execute({"items": [item] if item else []})

    def update(**kwargs):
        if state["schedule_error"]:
            raise OSError("Simulated schedule transport failure")
        state["video"] = state["videos"][kwargs["body"]["id"]]
        if not state["reject_schedule"]:
            state["video"]["status"].update(kwargs["body"]["status"])
        return execute(state["video"])

    def thumbnail(**kwargs):
        if state["thumbnail_error"]:
            raise OSError("Simulated thumbnail transport failure")
        return execute({"items": [{}]})


    def list_playlists(**kwargs):
        assert kwargs["mine"] is True and kwargs["maxResults"] == 50
        return execute({"items": state["playlists"]})

    def list_playlist_items(**kwargs):
        key = (kwargs["playlistId"], kwargs["videoId"])
        item_id = state["playlist_items"].get(key)
        return execute({"items": [{"id": item_id}] if item_id else []})

    def insert_playlist_item(**kwargs):
        request = MagicMock()

        def insert_item(num_retries):
            assert num_retries == 0
            if state["playlist_error"]:
                raise OSError("Simulated playlist transport failure")
            snippet = kwargs["body"]["snippet"]
            assert snippet["resourceId"]["kind"] == "youtube#video"
            key = (snippet["playlistId"], snippet["resourceId"]["videoId"])
            state["playlist_insert_count"] += 1
            item_id = f"playlist-item-{state['playlist_insert_count']}"
            state["playlist_items"][key] = item_id
            if state["playlist_lost_response"]:
                raise OSError("Playlist insert accepted but response lost")
            return {"id": item_id}

        request.execute.side_effect = insert_item
        return request

    api.channels.return_value.list.return_value = execute({"items": [{"id": CHANNEL}]})
    api.videos.return_value.insert.side_effect = insert
    api.videos.return_value.list.side_effect = list_videos
    api.videos.return_value.update.side_effect = update
    api.thumbnails.return_value.set.side_effect = thumbnail
    api.playlists.return_value.list.side_effect = list_playlists
    api.playlistItems.return_value.list.side_effect = list_playlist_items
    api.playlistItems.return_value.insert.side_effect = insert_playlist_item
    api.captions.side_effect = AssertionError("Subtitle tracks must never be uploaded")
    api.endScreens.side_effect = AssertionError("End screens require YouTube Studio")
    api.test_state = state
    return api


def test_default_upload_schedules_after_four_hours_without_subtitle_tracks(folder, service):
    store = MemoryStore()
    result = upload.upload_song(upload.load_kit(folder, "d"), service, store, CHANNEL)
    body = service.test_state["insert_args"]["body"]
    assert body["status"] == {"privacyStatus": "private", "selfDeclaredMadeForKids": False}
    assert body["snippet"]["categoryId"] == "10"
    assert body["snippet"]["defaultLanguage"] == body["snippet"]["defaultAudioLanguage"] == "am"
    assert service.test_state["insert_args"]["notifySubscribers"] is False
    assert service.test_state["insert_args"]["media_body"][2] is True
    assert service.thumbnails.return_value.set.call_args.kwargs["media_body"][0] == "thumbnail-d.jpg"
    assert result["publish_at"] == "2026-11-01T16:00:00Z"
    schedule = service.videos.return_value.update.call_args.kwargs["body"]["status"]
    assert schedule["privacyStatus"] == "private" and schedule["publishAt"] == result["publish_at"]
    service.captions.assert_not_called()
    assert result["status"] == "complete" and result["privacy"] == "private"
    assert json.loads((folder / "youtube-upload.json").read_text())["video_id"] == VIDEO
    assert store.state["channels"]["42"]["last_message_id"] == 22
    assert store.state["used_backgrounds"] == ["old"] and store.state["lyrics_channels"] == ["saved"]
    assert result["source_marker"] in body["snippet"]["tags"]


@pytest.mark.parametrize("visibility", ["private", "unlisted"])
def test_local_visibility_overrides_do_not_publish_or_upload_subtitles(folder, service, visibility):
    result = upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL, visibility)
    assert result["privacy"] == visibility and "publish_at" not in result
    assert service.test_state["insert_args"]["body"]["status"]["privacyStatus"] == "private"
    service.captions.assert_not_called()


def test_review_notes_do_not_prevent_requested_automatic_scheduling(folder, service):
    meta = json.loads((folder / "metadata.json").read_text())
    meta["review"]["reasons"].append("Check the artist")
    write_json(folder / "metadata.json", meta)
    result = upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL)
    assert result["review_warnings"] and not result["held_for_review"]
    assert result["publish_at"] == "2026-11-01T16:00:00Z"
    assert result["privacy"] == "private"


def test_completed_song_is_never_inserted_again_on_a_replay(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    first = upload.upload_song(kit, service, store, CHANNEL)
    second = upload.upload_song(kit, service, store, CHANNEL)
    assert second["video_id"] == first["video_id"]
    assert service.test_state["insert_count"] == 1
    service.captions.assert_not_called()


def test_partial_failure_resumes_the_same_video_and_finishes_assets(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["schedule_error"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    pending = next(iter(store.state["youtube_uploads"].values()))
    assert pending["status"] == "uploaded" and pending["thumbnail_done"]
    deadline = pending["publish_at"]
    service.test_state["schedule_error"] = False
    result = upload.upload_song(kit, service, store, CHANNEL, visibility="public")
    assert result["status"] == "complete" and result["privacy"] == "private"
    assert result["publish_at"] == deadline
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
    with pytest.raises(upload.UploadError, match="not marked as this Telegram song"):
        upload.upload_song(kit, service, store, CHANNEL, recover_video_id=VIDEO)


def test_unfinished_upload_rejects_regenerated_video_bytes(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["thumbnail_error"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    (folder / "video.mp4").write_bytes(b"different-render")
    with pytest.raises(upload.UploadError, match="different video bytes"):
        upload.upload_song(kit, service, store, CHANNEL)
    assert service.test_state["insert_count"] == 1


def test_accepted_schedule_is_reconciled_after_final_checkpoint_failure(folder, service, monkeypatch):
    store, kit = MemoryStore(), upload.load_kit(folder)
    # Reserve, video ID, thumbnail, playlist, deadline, then fail final completion.
    store.fail_write = 6
    with pytest.raises(RuntimeError):
        upload.upload_song(kit, service, store, CHANNEL)
    deadline = service.test_state["video"]["status"]["publishAt"]
    monkeypatch.setattr(upload, "utc_now", lambda: datetime(2026, 11, 1, 13, tzinfo=timezone.utc))
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["status"] == "complete" and result["publish_at"] == deadline
    assert service.test_state["insert_count"] == 1
    service.videos.return_value.update.assert_called_once()


def test_past_failed_schedule_gets_a_fresh_four_hour_delay_not_immediate_publication(folder, service, monkeypatch):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["schedule_error"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    monkeypatch.setattr(upload, "utc_now", lambda: datetime(2026, 11, 2, 8, tzinfo=timezone.utc))
    service.test_state["schedule_error"] = False
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["publish_at"] == "2026-11-02T12:00:00Z" and result["privacy"] == "private"
    assert service.test_state["insert_count"] == 1


def test_already_published_schedule_is_not_made_private_or_uploaded_again(folder, service, monkeypatch):
    store, kit = MemoryStore(), upload.load_kit(folder)
    store.fail_write = 6
    with pytest.raises(RuntimeError):
        upload.upload_song(kit, service, store, CHANNEL)
    monkeypatch.setattr(upload, "utc_now", lambda: datetime(2026, 11, 2, 8, tzinfo=timezone.utc))
    service.test_state["video"]["status"]["privacyStatus"] = "public"
    service.test_state["video"]["status"].pop("publishAt")
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["privacy"] == "public" and result["status"] == "complete"
    assert service.test_state["insert_count"] == 1
    service.videos.return_value.update.assert_called_once()


def test_rejected_schedule_is_not_misreported_as_success_or_published_immediately(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["reject_schedule"] = True
    with pytest.raises(upload.UploadError, match="did not accept"):
        upload.upload_song(kit, service, store, CHANNEL)
    assert next(iter(store.state["youtube_uploads"].values()))["status"] == "uploaded"
    assert service.test_state["video"]["status"]["privacyStatus"] == "private"


def test_oauth_channel_mismatch_prevents_upload(service):
    with pytest.raises(upload.UploadError, match="different"):
        upload.verify_channel(service, "UC" + "b" * 22)
    service.videos.return_value.insert.assert_not_called()


@pytest.mark.parametrize("value", ["@handle", "", "UCshort"])
def test_invalid_destination_channel_id_is_rejected(service, value):
    with pytest.raises(upload.UploadError, match="UC"):
        upload.verify_channel(service, value)


@pytest.mark.parametrize("missing", ["youtube.json", "metadata.json", "video.mp4", "thumbnail-b.jpg"])
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


def test_metadata_is_limited_before_upload_instead_of_rejecting_long_text(folder, service):
    kit = json.loads((folder / "youtube.json").read_text())
    kit["recommended_title"] = "A" * 150
    kit["tags"] = ["worship music " + str(i) for i in range(100)]
    write_json(folder / "youtube.json", kit)
    (folder / "description.txt").write_text("\u12a0" * 2000, encoding="utf-8")
    validated = upload.load_kit(folder)
    assert len(validated["title"]) == 100 and len(validated["description"].encode("utf-8")) <= 5000
    assert set(validated["metadata_adjusted"]) == {"title", "description", "tags"}
    upload.upload_song(validated, service, MemoryStore(), CHANNEL)
    snippet = service.test_state["insert_args"]["body"]["snippet"]
    assert len(snippet["title"]) <= 100 and len(snippet["description"].encode("utf-8")) <= 5000
    cost = sum(len(t) + (2 if " " in t else 0) for t in snippet["tags"]) + len(snippet["tags"]) - 1
    assert cost <= 500
    service.captions.assert_not_called()


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


def test_generation_upload_is_default_on_and_after_progress():
    from pathlib import Path

    doc = yaml.safe_load(Path(".github/workflows/generate.yml").read_text())
    events = doc.get("on", doc.get(True))
    inputs = events["workflow_dispatch"]["inputs"]
    assert len(inputs) <= 10
    assert inputs["upload_youtube"]["type"] == "boolean"
    assert inputs["upload_youtube"]["default"] is True
    assert events["workflow_call"]["inputs"]["upload_youtube"] == {"type": "boolean", "default": True}
    assert doc["jobs"]["publish"]["needs"] == ["fetch", "song", "finalize"]
    # Explicit false must still disable publishing and its OAuth preflight.
    assert doc["jobs"]["publish"]["if"] == "success() && inputs.upload_youtube == true"
    preflight = next(s for s in doc["jobs"]["fetch"]["steps"] if s.get("name", "").startswith("Check YouTube configuration"))
    assert preflight["if"] == "inputs.upload_youtube == true"
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
    step = next(s for s in job["steps"] if s.get("name", "").startswith("Upload videos and thumbnail"))
    assert "--visibility public" in step["run"]
    assert "YOUTUBE_PRIVACY" not in step["env"]
    assert "== 'd'" in step["env"]["YOUTUBE_THUMBNAIL"]
    assert "|| 'b'" in step["env"]["YOUTUBE_THUMBNAIL"]


def test_actual_google_sdk_accepts_upload_and_native_schedule_shapes(folder):
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    api = build("youtube", "v3", developerKey="not-used", cache_discovery=False, static_discovery=True)
    request = api.videos().insert(part="snippet,status",
        body={"snippet": {"title": "Test", "categoryId": "10"}, "status": {"privacyStatus": "private"}},
        notifySubscribers=False, media_body=MediaFileUpload(str(folder / "video.mp4"), resumable=True))
    assert request.resumable is not None
    schedule = api.videos().update(part="status", body={"id": VIDEO, "status": {
        "privacyStatus": "private", "publishAt": "2030-01-01T12:00:00Z"}})
    assert json.loads(schedule.body)["status"]["publishAt"] == "2030-01-01T12:00:00Z"
    lookup = api.playlists().list(part="id,snippet", mine=True, maxResults=50)
    assert "mine=true" in lookup.uri
    membership = api.playlistItems().list(part="id", playlistId=PLAYLIST, videoId=VIDEO, maxResults=50)
    assert f"playlistId={PLAYLIST}" in membership.uri and f"videoId={VIDEO}" in membership.uri
    insert_item = api.playlistItems().insert(part="snippet", body={"snippet": {
        "playlistId": PLAYLIST, "resourceId": {"kind": "youtube#video", "videoId": VIDEO}}})
    assert json.loads(insert_item.body)["snippet"]["resourceId"]["videoId"] == VIDEO
    with pytest.raises(AttributeError):
        api.endScreens()
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



def test_oauth_refresh_uses_google_endpoint_and_suppresses_sensitive_retry_logs(monkeypatch):
    import logging
    import google.oauth2.credentials
    import googleapiclient.discovery

    logger = logging.getLogger("googleapiclient.http")
    monkeypatch.setattr(logger, "disabled", False)
    credentials = MagicMock()
    constructor = MagicMock(return_value=credentials)
    monkeypatch.setattr(google.oauth2.credentials, "Credentials", constructor)
    built = MagicMock()
    monkeypatch.setattr(googleapiclient.discovery, "build", lambda *a, **k: built)
    monkeypatch.setenv("YOUTUBE_OAUTH_JSON", json.dumps({
        "refresh_token": "PRIVATE_REFRESH", "client_id": "CLIENT", "client_secret": "PRIVATE_CLIENT",
        "token_uri": "https://not-google.invalid/token"}))
    assert upload.youtube_client() is built
    credentials.refresh.assert_called_once()
    assert constructor.call_args.kwargs["token_uri"] == "https://oauth2.googleapis.com/token"
    assert constructor.call_args.kwargs["scopes"] == upload.SCOPES
    assert logger.disabled



@pytest.mark.parametrize("value", ["2030-01-01T12:00:00", "not-a-time", None])
def test_schedule_rejects_missing_timezone_or_invalid_saved_times(value):
    with pytest.raises(upload.UploadError, match="publication time"):
        upload.publish_time(value)


def test_no_subtitle_files_are_required_or_sent(folder, service):
    assert not (folder / "captions").exists()
    result = upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL)
    assert result["status"] == "complete"
    service.captions.assert_not_called()


def test_default_schedule_is_exactly_four_hours_in_the_future(folder, service):
    result = upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL)
    assert upload.publish_time(result["publish_at"]) - upload.utc_now() == timedelta(hours=4)
    assert service.test_state["video"]["status"]["privacyStatus"] == "private"


def test_old_privacy_variable_cannot_silently_disable_workflow_scheduling(folder, service, monkeypatch):
    monkeypatch.setenv("YOUTUBE_PRIVACY", "private")
    monkeypatch.setenv("YOUTUBE_CHANNEL_ID", CHANNEL)
    monkeypatch.setattr(sys, "argv", ["mezmur-upload", "--folder", str(folder)])
    monkeypatch.setattr(upload, "youtube_client", lambda: service)
    monkeypatch.setattr(upload, "Store", MemoryStore)
    upload.main()
    assert service.test_state["video"]["status"]["publishAt"] == "2026-11-01T16:00:00Z"



def make_batch(folder, ids=(22, 23, 24, 25)):
    kits = []
    for mid in ids:
        target = folder.parent / f"song-{mid}"
        if target != folder:
            shutil.copytree(folder, target)
        meta = json.loads((target / "metadata.json").read_text())
        write_json(target / "metadata.json", meta | {"message_id": mid})
        kits.append(upload.load_kit(target))
    return kits


def test_batch_randomizes_once_and_reserves_four_hour_slots_before_upload(folder, service, monkeypatch):
    kits, store = make_batch(folder), MemoryStore()
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    planned = upload.plan_batch(kits, store, CHANNEL, "generation-123")
    assert [kit["message_id"] for kit in planned] == [25, 24, 23, 22]
    records = store.state["youtube_uploads"]
    assert len(records) == 4 and store.writes == 1
    for index, kit in enumerate(planned, 1):
        record = records[upload.source_key(kit, CHANNEL)]
        assert record["status"] == "queued" and record["batch_id"] == "generation-123"
        assert upload.publish_time(record["publish_at"]) - upload.utc_now() == timedelta(hours=4 * index)
        assert record["publish_at"] == kit["planned_publish_at"]
    service.videos.return_value.insert.assert_not_called()
    assert store.state["channels"]["42"]["last_message_id"] == 22
    assert store.state["used_backgrounds"] == ["old"]


def test_retry_and_single_song_retry_keep_original_random_order_and_times(folder, service, monkeypatch):
    kits, store = make_batch(folder), MemoryStore()
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    first = upload.plan_batch(kits, store, CHANNEL, "original-run")
    saved = copy.deepcopy(store.state)
    monkeypatch.setattr(upload.random, "shuffle", lambda _: pytest.fail("Retry reshuffled saved songs"))
    monkeypatch.setattr(upload, "utc_now", lambda: datetime(2026, 11, 1, 13, tzinfo=timezone.utc))
    retry = upload.plan_batch(list(reversed(kits)), store, CHANNEL, "new-attempt")
    assert [kit["message_id"] for kit in retry] == [kit["message_id"] for kit in first]
    subset = upload.plan_batch([kits[0]], store, CHANNEL, "subset-attempt")
    assert subset[0]["planned_publish_at"] == first[-1]["planned_publish_at"]
    assert store.state == saved and store.writes == 1


def test_new_batch_appends_after_existing_future_slots(folder, service, monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    first = upload.plan_batch(make_batch(folder), store, CHANNEL, "first")
    second = upload.plan_batch(make_batch(folder, (26, 27)), store, CHANNEL, "second")
    times = [upload.publish_time(kit["planned_publish_at"]) for kit in first + second]
    assert [(value - upload.utc_now()).total_seconds() / 3600 for value in times] == [4, 8, 12, 16, 20, 24]
    assert all(b - a == timedelta(hours=4) for a, b in zip(times, times[1:]))


def test_queue_is_isolated_between_youtube_channels(folder, service):
    store = MemoryStore()
    upload.plan_batch(make_batch(folder), store, CHANNEL, "first-channel")
    other = "UC" + "b" * 22
    second = upload.plan_batch(make_batch(folder, (26, 27)), store, other, "second-channel")
    assert upload.publish_time(second[0]["planned_publish_at"]) - upload.utc_now() == timedelta(hours=4)


def test_completed_replay_does_not_consume_an_additional_slot(folder, service, monkeypatch):
    store = MemoryStore()
    first = upload.load_kit(folder)
    completed = upload.upload_song(first, service, store, CHANNEL)
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    planned = upload.plan_batch([first, *make_batch(folder, (23,))], store, CHANNEL, "new-run")
    saved = store.state["youtube_uploads"][upload.source_key(first, CHANNEL)]
    assert saved["publish_at"] == completed["publish_at"] and saved["status"] == "complete"
    new = next(kit for kit in planned if kit["message_id"] == 23)
    assert upload.publish_time(new["planned_publish_at"]) - upload.utc_now() == timedelta(hours=8)


def test_expired_unstarted_slots_move_after_future_queue_without_reshuffling(folder, service, monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    first = upload.plan_batch(make_batch(folder), store, CHANNEL, "first")
    future = {kit["message_id"]: kit["planned_publish_at"] for kit in first[1:]}
    base = upload.utc_now()
    monkeypatch.setattr(upload, "utc_now", lambda: base + timedelta(hours=5))
    monkeypatch.setattr(upload.random, "shuffle", lambda _: pytest.fail("Existing queue was reshuffled"))
    retry = upload.plan_batch(first, store, CHANNEL, "retry")
    moved = next(kit for kit in retry if kit["message_id"] == first[0]["message_id"])
    assert upload.publish_time(moved["planned_publish_at"]) - base == timedelta(hours=20)
    assert all(kit["planned_publish_at"] == future[kit["message_id"]] for kit in retry
               if kit["message_id"] in future)


def test_partial_upload_retry_never_collides_with_later_reserved_videos(folder, service, monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    first = upload.plan_batch(make_batch(folder), store, CHANNEL, "first")
    kit, base = first[0], upload.utc_now()
    service.test_state["schedule_error"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    service.test_state["schedule_error"] = False
    monkeypatch.setattr(upload, "utc_now", lambda: base + timedelta(hours=5))
    # The uploaded video's outcome must be checked before moving its old deadline.
    upload.plan_batch([kit], store, CHANNEL, "retry")
    old = store.state["youtube_uploads"][upload.source_key(kit, CHANNEL)]
    assert old["publish_at"] == kit["planned_publish_at"]
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert upload.publish_time(result["publish_at"]) - base == timedelta(hours=20)
    assert service.test_state["insert_count"] == 1
    assert len({r["publish_at"] for r in store.state["youtube_uploads"].values()}) == 4


def test_uncertain_upload_keeps_its_slot_and_blocks_another_insert(folder, service):
    store = MemoryStore()
    kit = upload.plan_batch([upload.load_kit(folder)], store, CHANNEL, "original")[0]
    service.test_state["next_error"] = OSError()
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    before = copy.deepcopy(store.state)
    upload.plan_batch([kit], store, CHANNEL, "retry")
    assert store.state == before
    with pytest.raises(upload.UploadError, match="uncertain"):
        upload.upload_song(kit, service, store, CHANNEL)
    assert service.test_state["insert_count"] == 1


def test_slot_checkpoint_rejects_an_advanced_snapshot():
    store = MemoryStore()
    snapshot = store.load()
    store.head += 1
    with pytest.raises(RuntimeError, match="Progress changed"):
        upload.save_record(store, "new", {"status": "queued"}, None, snapshot=snapshot)
    assert "youtube_uploads" not in store.state


def test_duplicate_kits_are_rejected_without_reserving_two_slots(folder, service):
    kit, store = upload.load_kit(folder), MemoryStore()
    with pytest.raises(upload.UploadError, match="same Telegram song"):
        upload.plan_batch([kit, kit], store, CHANNEL, "batch")
    assert store.writes == 0


@pytest.mark.parametrize("record", [None, [], {}, {"status": "invalid"}])
def test_corrupt_upload_record_cannot_silently_trigger_a_duplicate(folder, service, record):
    store, kit = MemoryStore(), upload.load_kit(folder)
    store.state["youtube_uploads"] = {upload.source_key(kit, CHANNEL): record}
    with pytest.raises(upload.UploadError):
        upload.upload_song(kit, service, store, CHANNEL)
    service.videos.return_value.insert.assert_not_called()


def test_all_four_uploads_use_youtube_scheduler_and_enforce_metadata_limits(folder, service, monkeypatch):
    store, kits = MemoryStore(), make_batch(folder)
    for kit in kits:
        root = kit["folder"]
        data = json.loads((root / "youtube.json").read_text())
        write_json(root / "youtube.json", data | {
            "recommended_title": "Worship title " * 50,
            "tags": ["amharic worship song " + str(i) for i in range(100)]})
        (root / "description.txt").write_text("\u12a0" * 3000, encoding="utf-8")
    kits = [upload.load_kit(kit["folder"]) for kit in kits]
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    planned = upload.plan_batch(kits, store, CHANNEL, "batch")
    results = [upload.upload_song(kit, service, store, CHANNEL) for kit in planned]
    assert len({r["video_id"] for r in results}) == 4
    assert [upload.publish_time(r["publish_at"]) - upload.utc_now() for r in results] == [
        timedelta(hours=h) for h in (4, 8, 12, 16)]
    for video in service.test_state["videos"].values():
        snippet, status = video["snippet"], video["status"]
        assert len(snippet["title"]) <= 100 and len(snippet["description"].encode("utf-8")) <= 5000
        cost = sum(len(t) + (2 if " " in t else 0) for t in snippet["tags"]) + len(snippet["tags"]) - 1
        assert cost <= 500
        assert status["privacyStatus"] == "private" and status["publishAt"]
    service.captions.assert_not_called()


def test_cli_summary_displays_publication_time_in_east_africa_time(folder, service, monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("YOUTUBE_CHANNEL_ID", CHANNEL)
    monkeypatch.setattr(sys, "argv", ["mezmur-upload", "--folder", str(folder)])
    monkeypatch.setattr(upload, "youtube_client", lambda: service)
    monkeypatch.setattr(upload, "Store", MemoryStore)
    upload.main()
    assert "2026-11-01 19:00 EAT" in summary.read_text()



def test_automatic_thumbnail_defaults_to_the_circular_portrait_layout(folder, service, monkeypatch):
    monkeypatch.delenv("YOUTUBE_THUMBNAIL", raising=False)
    monkeypatch.setenv("YOUTUBE_CHANNEL_ID", CHANNEL)
    monkeypatch.setattr(sys, "argv", ["mezmur-upload", "--folder", str(folder)])
    monkeypatch.setattr(upload, "youtube_client", lambda: service)
    monkeypatch.setattr(upload, "Store", MemoryStore)
    assert upload.load_kit(folder)["thumbnail"] == "thumbnail-b.jpg"
    upload.main()
    assert service.thumbnails.return_value.set.call_args.kwargs["media_body"][0] == "thumbnail-b.jpg"


def test_own_upload_is_accepted_even_if_youtube_drops_the_marker_tag(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    real_list = service.videos.return_value.list.side_effect

    def without_tags(**kwargs):
        for item in service.test_state["videos"].values():
            item["snippet"]["tags"] = []   # YouTube dropped or rewrote the tags
        return real_list(**kwargs)

    service.videos.return_value.list.side_effect = without_tags
    record = upload.upload_song(kit, service, store, CHANNEL)
    assert record["video_id"] == VIDEO


def test_upload_adds_original_video_to_mezmurs_and_records_manual_end_screen(folder, service):
    store = MemoryStore()
    record = upload.upload_song(upload.load_kit(folder), service, store, CHANNEL)
    request = service.playlistItems.return_value.insert.call_args.kwargs
    assert request == {"part": "snippet", "body": {"snippet": {
        "playlistId": PLAYLIST, "resourceId": {"kind": "youtube#video", "videoId": VIDEO}}}}
    assert service.test_state["playlist_insert_count"] == 1
    assert record["playlist_id"] == PLAYLIST and record["playlist_item_id"] == "playlist-item-1"
    assert record["playlist_title"] == "Mezmurs"
    assert record["playlist_url"] == f"https://www.youtube.com/playlist?list={PLAYLIST}"
    assert record["end_screen"] == {
        "status": "manual_required", "elements": ["best_for_viewer", "most_recent_upload"],
        "studio_url": f"https://studio.youtube.com/video/{VIDEO}/editor"}
    assert json.loads((folder / "youtube-upload.json").read_text()) == record
    assert store.state["youtube_uploads"][upload.source_key(upload.load_kit(folder), CHANNEL)] == record
    service.endScreens.assert_not_called()


def test_playlist_lookup_paginates_matches_name_and_checks_owner(service):
    def page(**kwargs):
        request = MagicMock()
        if kwargs.get("pageToken") == "page-2":
            request.execute.return_value = {"items": [
                {"id": PLAYLIST, "snippet": {"title": " mEzMuRs ", "channelId": CHANNEL}}]}
        else:
            request.execute.return_value = {"items": [
                {"id": "other", "snippet": {"title": "Mezmurs", "channelId": "UC" + "c" * 22}},
                {"id": "another", "snippet": {"title": "Other songs", "channelId": CHANNEL}}],
                "nextPageToken": "page-2"}
        return request

    service.playlists.return_value.list.side_effect = page
    found = upload.find_playlist(service, CHANNEL)
    assert found == {"id": PLAYLIST, "title": "mEzMuRs", "channel_id": CHANNEL}
    assert service.playlists.return_value.list.call_count == 2
    assert service.playlists.return_value.list.call_args.kwargs["pageToken"] == "page-2"


@pytest.mark.parametrize("playlists", [
    [],
    [{"id": "other", "snippet": {"title": "Other songs", "channelId": CHANNEL}}],
    [{"id": PLAYLIST, "snippet": {"title": "Mezmurs", "channelId": "UC" + "c" * 22}}],
    [{"id": PLAYLIST, "snippet": {"title": "Mezmurs", "channelId": CHANNEL}},
     {"id": "duplicate", "snippet": {"title": "Mezmurs", "channelId": CHANNEL}}],
])
def test_missing_wrong_owner_or_ambiguous_playlist_prevents_new_upload(folder, service, playlists):
    service.test_state["playlists"] = playlists
    store = MemoryStore()
    with pytest.raises(upload.UploadError, match="playlist"):
        upload.upload_song(upload.load_kit(folder), service, store, CHANNEL)
    service.videos.return_value.insert.assert_not_called()
    assert store.writes == 0


def test_supplied_playlist_from_another_channel_is_rejected(folder, service):
    with pytest.raises(upload.UploadError, match="belong"):
        upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL,
                           playlist={"id": PLAYLIST, "title": "Mezmurs", "channel_id": "other"})
    service.videos.return_value.insert.assert_not_called()


def test_existing_playlist_membership_is_reused_without_another_insert(folder, service):
    service.test_state["playlist_items"][(PLAYLIST, VIDEO)] = "existing-item"
    record = upload.upload_song(upload.load_kit(folder), service, MemoryStore(), CHANNEL)
    assert record["playlist_item_id"] == "existing-item"
    service.playlistItems.return_value.insert.assert_not_called()


def test_failed_playlist_addition_resumes_same_private_video_without_moving_slot(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    kit = upload.plan_batch([kit], store, CHANNEL, "batch")[0]
    service.test_state["playlist_error"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    saved = store.state["youtube_uploads"][upload.source_key(kit, CHANNEL)]
    assert saved["status"] == "uploaded" and saved["thumbnail_done"]
    assert service.test_state["video"]["status"]["privacyStatus"] == "private"
    service.videos.return_value.update.assert_not_called()
    service.test_state["playlist_error"] = False
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["status"] == "complete" and result["publish_at"] == kit["planned_publish_at"]
    assert service.test_state["insert_count"] == service.test_state["playlist_insert_count"] == 1
    assert service.thumbnails.return_value.set.call_count == 1


def test_lost_playlist_insert_response_is_reconciled_before_retrying(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    service.test_state["playlist_lost_response"] = True
    with pytest.raises(OSError):
        upload.upload_song(kit, service, store, CHANNEL)
    service.test_state["playlist_lost_response"] = False
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["status"] == "complete" and result["playlist_item_id"] == "playlist-item-1"
    assert service.test_state["insert_count"] == service.test_state["playlist_insert_count"] == 1


def test_playlist_checkpoint_failure_reuses_membership_on_retry(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    store.fail_write = 4
    with pytest.raises(RuntimeError):
        upload.upload_song(kit, service, store, CHANNEL)
    saved = store.state["youtube_uploads"][upload.source_key(kit, CHANNEL)]
    assert saved["status"] == "uploaded" and not saved.get("playlist_item_id")
    assert service.test_state["playlist_insert_count"] == 1
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["status"] == "complete" and result["playlist_item_id"] == "playlist-item-1"
    assert service.test_state["playlist_insert_count"] == service.test_state["insert_count"] == 1


def test_empty_playlist_insert_response_does_not_mark_upload_complete(folder, service):
    request = MagicMock()
    request.execute.return_value = {}
    service.playlistItems.return_value.insert.side_effect = lambda **kwargs: request
    store, kit = MemoryStore(), upload.load_kit(folder)
    with pytest.raises(upload.UploadError, match="playlist item ID"):
        upload.upload_song(kit, service, store, CHANNEL)
    assert store.state["youtube_uploads"][upload.source_key(kit, CHANNEL)]["status"] == "uploaded"
    assert service.test_state["video"]["status"]["privacyStatus"] == "private"


def test_completed_uploads_without_playlist_can_be_backfilled_without_touching_schedule(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    original = upload.upload_song(kit, service, store, CHANNEL)
    key = upload.source_key(kit, CHANNEL)
    store.state["youtube_uploads"][key] = {name: value for name, value in original.items()
                                           if not name.startswith("playlist_") and name != "end_screen"}
    service.test_state["playlist_items"].clear()
    updates = service.videos.return_value.update.call_count
    result = upload.upload_song(kit, service, store, CHANNEL)
    assert result["status"] == "complete" and result["playlist_id"] == PLAYLIST
    assert result["publish_at"] == original["publish_at"]
    assert service.test_state["insert_count"] == 1
    assert service.videos.return_value.update.call_count == updates
    assert service.thumbnails.return_value.set.call_count == 1


def test_completed_playlist_checkpoint_avoids_all_repeat_insertions(folder, service):
    store, kit = MemoryStore(), upload.load_kit(folder)
    first = upload.upload_song(kit, service, store, CHANNEL)
    requests = service.playlistItems.return_value.list.call_count
    writes = store.writes
    replay = upload.upload_song(kit, service, store, CHANNEL)
    assert replay == first and store.writes == writes
    assert service.playlistItems.return_value.list.call_count == requests
    assert service.test_state["playlist_insert_count"] == service.test_state["insert_count"] == 1


def test_cli_resolves_playlist_once_for_batch_and_reports_manual_end_screen(folder, service, monkeypatch, tmp_path):
    summary, store = tmp_path / "summary.md", MemoryStore()
    make_batch(folder, (22, 23))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("YOUTUBE_CHANNEL_ID", CHANNEL)
    monkeypatch.setattr(sys, "argv", ["mezmur-upload", "--root", str(folder.parent)])
    monkeypatch.setattr(upload, "youtube_client", lambda: service)
    monkeypatch.setattr(upload, "Store", lambda: store)
    upload.main()
    assert service.playlists.return_value.list.call_count == 1
    assert service.test_state["playlist_insert_count"] == service.test_state["insert_count"] == 2
    text = summary.read_text()
    assert f"https://www.youtube.com/playlist?list={PLAYLIST}" in text
    assert f"https://studio.youtube.com/video/{VIDEO}/editor" in text
    assert "Best for viewer" in text and "Most recent upload" in text
    assert "Manually" in text and "cannot apply end screens" in text


def test_cli_missing_playlist_fails_before_queue_planning_or_upload(folder, service, monkeypatch):
    service.test_state["playlists"] = []
    monkeypatch.setenv("YOUTUBE_CHANNEL_ID", CHANNEL)
    monkeypatch.setattr(sys, "argv", ["mezmur-upload", "--folder", str(folder)])
    monkeypatch.setattr(upload, "youtube_client", lambda: service)
    monkeypatch.setattr(upload, "Store", lambda: pytest.fail("Missing playlist must fail before state writes"))
    with pytest.raises(SystemExit) as stopped:
        upload.main()
    assert stopped.value.code == 1
    service.videos.return_value.insert.assert_not_called()


@pytest.mark.parametrize("previous_delay,new_hours", [(2, (12, 16)), (5, (24, 28))])
def test_legacy_reservations_stay_fixed_and_new_songs_follow_after_four_hours(
        folder, service, monkeypatch, previous_delay, new_hours):
    kits, store = make_batch(folder), MemoryStore()
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    with monkeypatch.context() as legacy:
        legacy.setattr(upload, "PUBLISH_DELAY", timedelta(hours=previous_delay))
        original = upload.plan_batch(kits, store, CHANNEL, "legacy-batch")
    saved = copy.deepcopy(store.state["youtube_uploads"])
    retry = upload.plan_batch(list(reversed(kits)), store, CHANNEL, "retry")
    assert retry == original
    assert store.state["youtube_uploads"] == saved
    appended = upload.plan_batch(make_batch(folder, (26, 27)), store, CHANNEL, "new-batch")
    assert [upload.publish_time(kit["planned_publish_at"]) - upload.utc_now() for kit in appended] == [
        timedelta(hours=hours) for hours in new_hours]
    assert all(store.state["youtube_uploads"][key] == record for key, record in saved.items())
    completed = upload.upload_song(original[0], service, store, CHANNEL)
    assert completed["publish_at"] == original[0]["planned_publish_at"]
    assert service.test_state["video"]["status"]["publishAt"] == original[0]["planned_publish_at"]


def test_multiple_expired_slots_move_after_remaining_queue_at_four_hour_intervals(folder, service, monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(upload.random, "shuffle", lambda items: items.reverse())
    first = upload.plan_batch(make_batch(folder), store, CHANNEL, "batch")
    base = upload.utc_now()
    future = first[-1]["planned_publish_at"]
    monkeypatch.setattr(upload, "utc_now", lambda: base + timedelta(hours=13))
    monkeypatch.setattr(upload.random, "shuffle", lambda _: pytest.fail("Retry reshuffled the queue"))
    retry = upload.plan_batch(first, store, CHANNEL, "retry")
    assert [upload.publish_time(kit["planned_publish_at"]) - base for kit in retry] == [
        timedelta(hours=h) for h in (16, 20, 24, 28)]
    assert retry[0]["planned_publish_at"] == future
