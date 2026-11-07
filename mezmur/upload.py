"""YouTube publishing with unlisted-first uploads, native scheduling and duplicate protection."""
import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .common import required, write_json
from .lyrics import is_amharic_text
from .state import Store
from .youtube import limit_description, limit_tags, limit_title

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
PRIVACIES = ("private", "unlisted", "public")
THUMBNAILS = {"a": "thumbnail.jpg", "b": "thumbnail-b.jpg",
              "c": "thumbnail-c.jpg", "d": "thumbnail-d.jpg"}
PUBLISH_DELAY = timedelta(hours=4)
PLAYLIST_NAME = "Mezmurs"
CAPTION_LANGUAGE = "am"
CAPTION_NAME = "Amharic lyrics (Rehoboth Mezmur)"
CAPTION_MAX_BYTES = 100 * 1024 * 1024
CAPTION_ACCEPTED = ("uploaded", "serving", "syncing")


class UploadError(RuntimeError):
    """A user-actionable error whose text is safe to include in public job logs."""


def youtube_client():
    """Refresh only in memory; never write credentials into an artifact or state branch."""
    # SDK retry warnings can include bearer-like resumable-session URLs.
    logging.getLogger("googleapiclient.http").disabled = True
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    try:
        info = json.loads(required("YOUTUBE_OAUTH_JSON"))
        values = {name: info[name] for name in ("refresh_token", "client_id", "client_secret")}
        if not all(isinstance(v, str) and v.strip() for v in values.values()):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise UploadError("Set YOUTUBE_OAUTH_JSON using scripts/youtube_authorize.py; do not use an API key") from None
    credentials = Credentials(token=None, token_uri="https://oauth2.googleapis.com/token", scopes=SCOPES, **values)
    credentials.refresh(Request())
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def verify_channel(service, expected):
    if not re.fullmatch(r"UC[A-Za-z0-9_-]{22}", expected):
        raise UploadError("YOUTUBE_CHANNEL_ID must be your UC... channel ID, not its @handle")
    items = service.channels().list(part="id", mine=True, maxResults=50).execute(num_retries=3).get("items", [])
    if [item["id"] for item in items] != [expected]:
        raise UploadError("OAuth authorized a different YouTube channel; authorize the intended channel or Brand Account")



def find_playlist(service, channel):
    """Find the uniquely named playlist owned by the authorized channel."""
    matches, token = {}, None
    while True:
        params = {"part": "id,snippet", "mine": True, "maxResults": 50}
        if token:
            params["pageToken"] = token
        page = service.playlists().list(**params).execute(num_retries=3)
        for item in page.get("items", []):
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            playlist_id = item.get("id")
            if (isinstance(title, str) and title.strip().casefold() == PLAYLIST_NAME.casefold()
                    and snippet.get("channelId") == channel and isinstance(playlist_id, str) and playlist_id):
                matches[playlist_id] = {"id": playlist_id, "title": title.strip(), "channel_id": channel}
        token = page.get("nextPageToken")
        if not token:
            break
    if not matches:
        raise UploadError("The connected channel has no playlist named Mezmurs; create or rename it in Studio before retrying")
    if len(matches) != 1:
        raise UploadError("More than one playlist is named Mezmurs; rename the duplicate playlists before retrying")
    return next(iter(matches.values()))


def playlist_membership(service, playlist, video_id):
    """Check membership before inserting; reconcile an accepted-but-uncheckpointed insert on retry."""
    items = service.playlistItems().list(part="id", playlistId=playlist["id"], videoId=video_id,
                                         maxResults=50).execute(num_retries=3).get("items", [])
    if items:
        item_id = items[0].get("id")
    else:
        # Inserts are not idempotent. Never automatically retry a potentially accepted POST.
        response = service.playlistItems().insert(part="snippet", body={"snippet": {
            "playlistId": playlist["id"], "resourceId": {"kind": "youtube#video", "videoId": video_id}
        }}).execute(num_retries=0)
        item_id = response.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise UploadError("YouTube returned no playlist item ID; retry the existing video, not a new upload")
    return {"playlist_id": playlist["id"], "playlist_title": playlist["title"], "playlist_item_id": item_id,
            "playlist_url": f"https://www.youtube.com/playlist?list={playlist['id']}",
            "end_screen": {"status": "manual_required",
                           "elements": ["best_for_viewer", "most_recent_upload"],
                           "studio_url": f"https://studio.youtube.com/video/{video_id}/editor"}}


def media(path, mime, resumable=False):
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(str(path), mimetype=mime, chunksize=8 * 1024 * 1024, resumable=resumable)


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise UploadError(f"Missing or invalid {path.name}; use a newly generated complete YouTube kit") from None


def load_subtitle(folder, duration=None):
    """Validate UTF-8 timed SRT before any OAuth, queue reservation or video upload."""
    import srt

    path = folder / "subtitle.srt"
    if not path.exists():
        path = folder / "captions" / "am.srt"  # Older complete artifacts already contain aligned SRT.
    try:
        if not path.is_file() or not 0 < path.stat().st_size <= CAPTION_MAX_BYTES:
            raise ValueError
        data = path.read_bytes()
        text = data.decode("utf-8-sig")
        cues = list(srt.parse(text))
        if not cues:
            raise ValueError
        previous = -1.0
        for cue in cues:
            start, end = cue.start.total_seconds(), cue.end.total_seconds()
            if start < 0 or end <= start or start < previous or not cue.content.strip():
                raise ValueError
            previous = start
        if duration is not None:
            seconds = float(duration)
            if not math.isfinite(seconds) or seconds <= 0 or max(c.end.total_seconds() for c in cues) > seconds + 0.5:
                raise ValueError
    except (OSError, UnicodeError, TypeError, ValueError, OverflowError, srt.SRTParseError):
        raise UploadError("A valid UTF-8 timed subtitle.srt (or legacy captions/am.srt) is required; "
                          "use the original complete generation artifact") from None
    if not all(is_amharic_text(cue.content) for cue in cues):
        raise UploadError("subtitle.srt must contain Amharic script only; no English or romanized Amharic")
    return {"subtitle": path.relative_to(folder).as_posix(), "subtitle_sha256": hashlib.sha256(data).hexdigest(),
            "subtitle_cues": len(cues)}


def load_kit(folder, thumbnail="d"):
    folder = Path(folder)
    kit, meta = _json(folder / "youtube.json"), _json(folder / "metadata.json")
    if not isinstance(kit, dict) or not isinstance(meta, dict):
        raise UploadError("Upload kit and metadata must be JSON objects")
    mid, channel = str(meta.get("message_id", "")), str(meta.get("telegram_channel_id", ""))
    if not mid.isdigit() or int(mid) <= 0 or not re.fullmatch(r"-?[0-9]+", channel) or int(channel) == 0:
        raise UploadError("Kit is missing its Telegram channel/message identity; generate it again with the new workflow")
    if meta.get("lyrics_mode") != "lyrics-aligned":
        raise UploadError("Only videos with verified, aligned lyrics may be uploaded")
    if thumbnail not in THUMBNAILS:
        raise UploadError("Choose thumbnail a, b, c or d")
    title = kit.get("recommended_title", "")
    if not isinstance(title, str):
        raise UploadError("The recommended title must be text")
    original_title = title
    title = limit_title(title)
    if not title:
        raise UploadError("The recommended title must contain text")
    try:
        description = (folder / "description.txt").read_text(encoding="utf-8").strip()
    except OSError:
        raise UploadError("description.txt is missing") from None
    original_description = description
    description = limit_description(description)
    if not description:
        raise UploadError("Description must contain text")
    if not isinstance(kit.get("tags"), list) or not all(isinstance(t, str) for t in kit["tags"]):
        raise UploadError("youtube.json must contain a list of text tags")
    for relative in ["video.mp4", THUMBNAILS[thumbnail]]:
        path = folder / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise UploadError(f"Required upload file is missing or empty: {relative}")
    for review in (kit.get("review"), meta.get("review")):
        if not isinstance(review, dict):
            raise UploadError("Kit is missing its publishing review status")
    review_needed = any(r.get("needs_review") or r.get("reasons")
                        for r in (kit["review"], meta["review"]))
    subtitle = load_subtitle(folder, meta.get("duration_seconds"))
    tags = limit_tags(kit["tags"])
    adjusted = [name for name, before, after in (
        ("title", original_title, title), ("description", original_description, description),
        ("tags", kit["tags"], tags)) if before != after]
    return {"folder": folder, "message_id": int(mid), "telegram_channel_id": str(int(channel)),
            "title": title, "description": description, "tags": tags, "metadata_adjusted": adjusted,
            "review_needed": bool(review_needed), "thumbnail": THUMBNAILS[thumbnail], **subtitle}


def source_key(kit, channel):
    return f"{channel}:{kit['telegram_channel_id']}:{kit['message_id']}"


def source_marker(key):
    return "mezmur-source-" + hashlib.sha256(key.encode()).hexdigest()[:24]


def upload_tags(tags, marker):
    return limit_tags([marker, *tags])


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def publish_time(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError
        return result.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError):
        raise UploadError("Saved publication time is invalid; inspect the existing video's schedule") from None


def next_publish_slot(uploads, channel, now):
    """Append after all recorded future slots for this channel, including unfinished uploads."""
    if not isinstance(uploads, dict):
        raise UploadError("Saved YouTube upload state is corrupt")
    tail = now
    for record in uploads.values():
        if not isinstance(record, dict):
            raise UploadError("Saved YouTube upload record is corrupt")
        if record.get("youtube_channel_id") == channel and record.get("publish_at"):
            tail = max(tail, publish_time(record["publish_at"]))
    return tail + PUBLISH_DELAY


def plan_batch(kits, store, channel, batch_id):
    """Persist one randomized order and distinct deadlines before sending any video."""
    snapshot = store.load()
    state = copy.deepcopy(snapshot["state"])
    uploads = state.setdefault("youtube_uploads", {})
    if not isinstance(uploads, dict):
        raise UploadError("Saved YouTube upload state is corrupt")
    keys = [source_key(kit, channel) for kit in kits]
    if len(set(keys)) != len(keys):
        raise UploadError("Multiple input kits identify the same Telegram song")
    now = utc_now()
    new, expired = [], []
    for kit, key in zip(kits, keys):
        record = uploads.get(key)
        if key not in uploads:
            new.append((kit, key))
        elif not isinstance(record, dict):
            raise UploadError("Saved YouTube upload record is corrupt")
        elif record.get("status") == "queued":
            if not record.get("publish_at"):
                raise UploadError("Queued video has no saved publication time")
            if publish_time(record["publish_at"]) <= now:
                expired.append((kit, key))
    # Only new songs are shuffled. Existing reservations and subset retries retain their slots.
    if new:
        random.shuffle(new)
    expired.sort(key=lambda item: publish_time(uploads[item[1]]["publish_at"]))
    for kit, key in expired + new:
        previous = uploads.get(key)
        target = next_publish_slot(uploads, channel, now)
        record = copy.deepcopy(previous) if previous else {
            "status": "queued", "message_id": kit["message_id"],
            "telegram_channel_id": kit["telegram_channel_id"], "youtube_channel_id": channel,
            "requested_visibility": "public", "held_for_review": False,
            "batch_id": str(batch_id), "queued_at": now.isoformat()}
        record["publish_at"] = target.isoformat().replace("+00:00", "Z")
        if previous:
            record["rescheduled_at"] = now.isoformat()
        uploads[key] = record
    if expired or new:
        # Store.save checks this exact head; another writer cannot silently steal a slot.
        store.save(snapshot, state)
    earliest = datetime.min.replace(tzinfo=timezone.utc)
    ordered = sorted(kits, key=lambda kit: (
        publish_time(uploads[source_key(kit, channel)]["publish_at"])
        if uploads[source_key(kit, channel)].get("publish_at") else earliest,
        source_key(kit, channel)))
    return [{**kit, "planned_publish_at": uploads[source_key(kit, channel)].get("publish_at")} for kit in ordered]


def save_record(store, key, record, previous, snapshot=None):
    """Checkpoint one record without losing cursor/library data or another writer's reservations."""
    snapshot = store.load() if snapshot is None else snapshot
    state = copy.deepcopy(snapshot["state"])
    uploads = state.setdefault("youtube_uploads", {})
    if not isinstance(uploads, dict) or uploads.get(key) != previous:
        raise UploadError("Upload state changed concurrently; refusing to overwrite it")
    uploads[key] = copy.deepcopy(record)
    store.save(snapshot, state)


def _video(service, video_id, channel, marker, require_marker=True):
    """Fetch a video and make sure it is ours.

    require_marker=False is used for a video ID that YouTube itself returned to this pipeline's upload
    (so it is certainly this song); YouTube may drop or rewrite tags, so the hidden marker tag is only
    required when someone hands in a video ID for recovery."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise UploadError("Invalid YouTube video ID")
    items = service.videos().list(part="snippet,status", id=video_id).execute(num_retries=3).get("items", [])
    if len(items) != 1:
        raise UploadError(f"YouTube could not find video {video_id} (deleted, or not visible to this account)")
    owner = items[0]["snippet"].get("channelId")
    if owner != channel:
        raise UploadError(f"Video {video_id} belongs to channel {owner}, not the connected channel {channel}")
    if require_marker and marker not in items[0]["snippet"].get("tags", []):
        raise UploadError(f"Video {video_id} is on this channel but is not marked as this Telegram song")
    return items[0]


def ensure_captions(kit, service, store, key, record, previous):
    """Checkpoint caption intent and reconcile an accepted insert before attempting another POST."""
    digest = kit["subtitle_sha256"]
    if record.get("caption_sha256") and record["caption_sha256"] != digest:
        raise UploadError("An existing upload uses different subtitle bytes; retry with the original run's artifact")
    if record.get("caption_id") and record.get("caption_status") in CAPTION_ACCEPTED:
        return previous
    path = kit["folder"] / kit["subtitle"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise UploadError("Subtitle bytes changed after validation; reload the original artifact before retrying")
    if not record.get("caption_sha256"):
        record.update(caption_file=kit["subtitle"], caption_sha256=digest, caption_language=CAPTION_LANGUAGE,
                      caption_name=CAPTION_NAME, caption_status="pending")
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)
    tracks = service.captions().list(part="id,snippet", videoId=record["video_id"]).execute(
        num_retries=3).get("items", [])
    matches = [track for track in tracks if track.get("snippet", {}).get("language") == CAPTION_LANGUAGE
               and track.get("snippet", {}).get("name") == CAPTION_NAME]
    if len(matches) > 1:
        raise UploadError("Multiple managed Amharic caption tracks exist; inspect the existing video in Studio")
    if matches:
        track = matches[0]
    else:
        # A timed SRT needs no deprecated auto-sync parameter. Never retry a possibly accepted insert.
        track = service.captions().insert(part="snippet", body={"snippet": {
            "videoId": record["video_id"], "language": CAPTION_LANGUAGE, "name": CAPTION_NAME, "isDraft": False}},
            media_body=media(path, "application/octet-stream")).execute(num_retries=0)
    caption_id = track.get("id")
    snippet = track.get("snippet", {})
    if not isinstance(caption_id, str) or not caption_id:
        raise UploadError("YouTube returned no caption ID; retry the existing video, not a new upload")
    record.update(caption_id=caption_id, caption_status="draft" if snippet.get("isDraft") else snippet.get("status", "uploaded"))
    save_record(store, key, record, previous)
    previous = copy.deepcopy(record)
    if snippet.get("status") == "failed" or snippet.get("isDraft") is True:
        raise UploadError("YouTube could not publish the Amharic caption track; inspect Subtitles in Studio "
                          "and retry the existing video after correcting the track")
    return previous


def upload_song(kit, service, store, channel, visibility="public", recover_video_id=None, playlist=None):
    if visibility not in PRIVACIES:
        raise UploadError("Visibility must be private, unlisted or public")
    folder, key = kit["folder"], source_key(kit, channel)
    marker = source_marker(key)
    uploads = store.load()["state"].get("youtube_uploads", {})
    if not isinstance(uploads, dict):
        raise UploadError("Saved YouTube upload state is corrupt")
    previous = uploads.get(key)
    if key in uploads and not isinstance(previous, dict):
        raise UploadError("Saved YouTube upload record is corrupt")
    record = copy.deepcopy(previous) if key in uploads else None
    if record is not None and record.get("status") not in ("queued", "uploading", "uploaded", "complete"):
        raise UploadError("Unknown upload status; inspect the state branch before retrying")
    playlist = find_playlist(service, channel) if playlist is None else playlist
    if playlist.get("channel_id") != channel or not playlist.get("id"):
        raise UploadError("The destination playlist must belong to the connected YouTube channel")
    if record and record.get("status") == "complete":
        needs_captions = not record.get("caption_id") or record.get("caption_status") not in CAPTION_ACCEPTED
        needs_playlist = record.get("playlist_id") != playlist["id"] or not record.get("playlist_item_id")
        if needs_captions:
            with (folder / "video.mp4").open("rb") as fh:
                digest = hashlib.file_digest(fh, "sha256").hexdigest()
            if record.get("video_sha256") != digest:
                raise UploadError("Caption backfill requires the original artifact's video bytes, not a regenerated video")
        if needs_playlist or needs_captions:
            _video(service, record["video_id"], channel, marker, require_marker=False)
        if needs_playlist:
            record.update(playlist_membership(service, playlist, record["video_id"]))
            save_record(store, key, record, previous)
            previous = copy.deepcopy(record)
        if needs_captions:
            ensure_captions(kit, service, store, key, record, previous)
        write_json(folder / "youtube-upload.json", record)
        print(f"Song {kit['message_id']}: already uploaded, https://youtu.be/{record['video_id']}", flush=True)
        return record
    with (folder / "video.mp4").open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    if record and record.get("status") != "queued" and record.get("video_sha256") != digest:
        raise UploadError("An unfinished upload uses different video bytes; retry Publish Generated Videos with the original run's artifact")
    if record and record.get("caption_sha256") and record["caption_sha256"] != kit["subtitle_sha256"]:
        raise UploadError("An unfinished upload uses different subtitle bytes; retry the original run's artifact")
    if record and record["status"] == "uploading":
        if not recover_video_id:
            raise UploadError("An earlier upload has an uncertain outcome. Check YouTube Studio, then use recover_video_id; no duplicate will be created")
        _video(service, recover_video_id, channel, marker)
        record.update(status="uploaded", video_id=recover_video_id, url=f"https://youtu.be/{recover_video_id}")
        write_json(folder / "youtube-upload.json", record)
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)
    elif recover_video_id:
        raise UploadError("Recovery is only allowed for an uncertain existing upload reservation")
    if record is None or record["status"] == "queued":
        record = {**(record or {}), "status": "uploading", "message_id": kit["message_id"],
                  "telegram_channel_id": kit["telegram_channel_id"], "youtube_channel_id": channel,
                  "video_sha256": digest, "source_marker": marker,
                  "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                  "created_at": utc_now().isoformat(),
                  "requested_visibility": record.get("requested_visibility", visibility) if record else visibility,
                  "held_for_review": False, "review_warnings": kit["review_needed"],
                  "thumbnail_done": False, "metadata_adjusted": kit.get("metadata_adjusted", []),
                  "caption_file": kit["subtitle"], "caption_sha256": kit["subtitle_sha256"],
                  "caption_language": CAPTION_LANGUAGE, "caption_name": CAPTION_NAME, "caption_status": "pending"}
        record["initial_privacy"] = "private" if record["requested_visibility"] == "private" else "unlisted"
        # A durable reservation precedes all upload requests. Unknown outcomes never trigger a fresh insert.
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)
        write_json(folder / "youtube-upload.json", record)
        body = {"snippet": {"title": kit["title"], "description": kit["description"],
                            "tags": upload_tags(kit["tags"], marker), "categoryId": "10",
                            "defaultLanguage": "am", "defaultAudioLanguage": "am"},
                "status": {"privacyStatus": record["initial_privacy"], "selfDeclaredMadeForKids": False}}
        request = service.videos().insert(part="snippet,status", body=body, notifySubscribers=False,
                                          media_body=media(folder / "video.mp4", "video/mp4", resumable=True))
        response = None
        while response is None:
            progress, response = request.next_chunk(num_retries=3)
            if progress:
                print(f"Song {kit['message_id']}: upload {progress.progress():.0%}", flush=True)
        video_id = response.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise UploadError("YouTube returned no valid video ID; inspect Studio before recovering this reservation")
        record.update(status="uploaded", video_id=video_id, url=f"https://youtu.be/{video_id}")
        print(f"YouTube accepted song {kit['message_id']}: {record['url']}", flush=True)
        write_json(folder / "youtube-upload.json", record)
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)

    video_id = record["video_id"]
    # This ID came from our own upload (or an already verified recovery): check the channel only.
    current = _video(service, video_id, channel, marker, require_marker=False)
    if not record.get("thumbnail_done"):
        service.thumbnails().set(videoId=video_id,
                                 media_body=media(folder / kit["thumbnail"], "image/jpeg")).execute(num_retries=3)
        record["thumbnail_done"] = True
        record["thumbnail_file"] = kit["thumbnail"]
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)
    if record.get("playlist_id") != playlist["id"] or not record.get("playlist_item_id"):
        record.update(playlist_membership(service, playlist, video_id))
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)
        write_json(folder / "youtube-upload.json", record)
    previous = ensure_captions(kit, service, store, key, record, previous)
    write_json(folder / "youtube-upload.json", record)
    # Scheduling is private -> public; YouTube does not support publishAt on unlisted videos.
    # Persist the deadline before requesting it so retries cannot slide an accepted schedule.
    requested = record["requested_visibility"]
    actual = "private" if record.get("held_for_review") else requested
    status = current.get("status", {})
    if actual == "public":
        now = utc_now()
        target = publish_time(record["publish_at"]) if record.get("publish_at") else None
        if status.get("privacyStatus") == "public":
            if target is None or target > now:
                raise UploadError("The existing video is already public unexpectedly; inspect Studio before retrying")
        else:
            # A late retry must never send a past date: YouTube would publish it immediately.
            if target is None or target <= now:
                snapshot = store.load()
                target = next_publish_slot(snapshot["state"].get("youtube_uploads", {}), channel, now)
                record["publish_at"] = target.isoformat().replace("+00:00", "Z")
                save_record(store, key, record, previous, snapshot=snapshot)
                previous = copy.deepcopy(record)
            actual = "private"
            scheduled = publish_time(status["publishAt"]) if status.get("publishAt") else None
            if status.get("privacyStatus") != "private" or scheduled != target:
                result = service.videos().update(part="status", body={
                    "id": video_id, "status": {"privacyStatus": "private",
                                               "publishAt": record["publish_at"],
                                               "selfDeclaredMadeForKids": False}}).execute(num_retries=3)
                accepted = result.get("status", {})
                if (accepted.get("privacyStatus") != "private" or not accepted.get("publishAt")
                        or publish_time(accepted["publishAt"]) != target):
                    raise UploadError("YouTube did not accept the saved publication schedule; retry the existing video, not a new upload")
    elif status.get("privacyStatus") != actual:
        result = service.videos().update(part="status", body={
            "id": video_id, "status": {"privacyStatus": actual, "selfDeclaredMadeForKids": False}}).execute(num_retries=3)
        if result.get("status", {}).get("privacyStatus") != actual:
            raise UploadError("YouTube did not apply the requested visibility; inspect the existing video, do not re-upload")
    record.update(status="complete", privacy=actual, completed_at=utc_now().isoformat())
    save_record(store, key, record, previous)
    write_json(folder / "youtube-upload.json", record)
    when = f"; public at {record['publish_at']}" if record.get("publish_at") else ""
    print(f"Song {kit['message_id']}: {record['url']} ({actual}{when}); playlist={record['playlist_title']}; "
          f"captions={record['caption_id']}; end screen requires Studio", flush=True)
    return record


def safe_error(error):
    if isinstance(error, UploadError):
        return str(error)
    # HttpError.__str__ can contain a resumable session URL. Never emit it, tokens or raw responses.
    status = getattr(getattr(error, "resp", None), "status", None)
    return f"{type(error).__name__}" + (f" (HTTP {status})" if isinstance(status, int) else "") + \
        "; check API access, quota and OAuth authorization. See docs/youtube-upload.md."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--folder", type=Path)
    where.add_argument("--root", type=Path, help="Downloaded artifact folders containing video.mp4")
    parser.add_argument("--visibility", choices=PRIVACIES, default="public",
                        help="Public queues randomized uploads at four-hour intervals; private/unlisted are local-only overrides")
    parser.add_argument("--thumbnail", choices=tuple(THUMBNAILS), default=os.environ.get("YOUTUBE_THUMBNAIL") or "d")
    parser.add_argument("--recover-video-id")
    parser.add_argument("--dry-run", action="store_true", help="Validate kits without OAuth, network access or state writes")
    args = parser.parse_args()
    folders = [args.folder] if args.folder else sorted(p.parent for p in args.root.rglob("video.mp4"))
    if not folders:
        print("No finished videos found; nothing to upload.")
        return
    try:
        kits = [load_kit(folder, args.thumbnail) for folder in folders]
        if args.recover_video_id and len(kits) != 1:
            raise UploadError("Recover one song at a time using message_id in the publish workflow")
        if args.visibility not in PRIVACIES:
            raise UploadError("Visibility must be private, unlisted or public")
        if args.dry_run:
            for kit in kits:
                print(f"Validated song {kit['message_id']}; metadata adjusted={kit['metadata_adjusted']}; no upload performed")
            return
        channel = required("YOUTUBE_CHANNEL_ID")
        service = youtube_client()
        verify_channel(service, channel)
        playlist = find_playlist(service, channel)
        store = Store()
        if args.visibility == "public":
            batch_id = os.environ.get("SOURCE_RUN_ID") or os.environ.get("GITHUB_RUN_ID", "local")
            kits = plan_batch(kits, store, channel, batch_id)
            for kit in kits:
                print(f"Song {kit['message_id']}: saved publication slot {kit['planned_publish_at'] or 'existing upload'}", flush=True)
    except Exception as error:
        print("::error::" + safe_error(error), flush=True)
        raise SystemExit(1) from None
    failed = False
    for kit in kits:
        try:
            record = upload_song(kit, service, store, channel, args.visibility, args.recover_video_id, playlist=playlist)
            line = f"- Song {kit['message_id']}: [YouTube video]({record['url']}) ({record['privacy']})"
            if record.get("publish_at"):
                local = publish_time(record["publish_at"]).astimezone(timezone(timedelta(hours=3), "EAT"))
                line += f" - public publication scheduled for {local:%Y-%m-%d %H:%M %Z}"
            if record.get("review_warnings"):
                line += " - review notes remain in the kit"
            line += f" - playlist: [Mezmurs]({record['playlist_url']})"
            line += f" - Amharic timed captions uploaded ({record['caption_status']})"
            line += (f"\n- Song {kit['message_id']} end screen: "
                     f"[Open YouTube Studio]({record['end_screen']['studio_url']}). "
                     "Manually add two Video elements: **Best for viewer** and **Most recent upload**, then Save. "
                     "The YouTube Data API cannot apply end screens.")
        except Exception as error:
            failed = True
            message = safe_error(error)
            write_json(kit["folder"] / "youtube-upload-error.json", {"error": message})
            line = f"- Song {kit['message_id']}: upload incomplete. {message}"
            print("::error::" + line, flush=True)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
