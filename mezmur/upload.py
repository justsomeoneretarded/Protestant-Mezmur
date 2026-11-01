"""Opt-in YouTube publishing with private-first uploads and durable duplicate protection."""
import argparse
import copy
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .common import required, write_json
from .state import Store

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
PRIVACIES = ("private", "unlisted", "public")
THUMBNAILS = {"a": "thumbnail.jpg", "b": "thumbnail-b.jpg",
              "c": "thumbnail-c.jpg", "d": "thumbnail-d.jpg"}
CAPTIONS = [("captions/am.srt", "Amharic"), ("captions/am-Latn.srt", "Amharic (Latin letters)")]


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


def media(path, mime, resumable=False):
    from googleapiclient.http import MediaFileUpload

    return MediaFileUpload(str(path), mimetype=mime, chunksize=8 * 1024 * 1024, resumable=resumable)


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise UploadError(f"Missing or invalid {path.name}; use a newly generated complete YouTube kit") from None


def load_kit(folder, thumbnail="a"):
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
    title = title.strip()
    if not title or len(title) > 100 or any(c in title for c in "<>"):
        raise UploadError("The recommended title must be 1..100 characters without < or >")
    try:
        description = (folder / "description.txt").read_text(encoding="utf-8").strip()
    except OSError:
        raise UploadError("description.txt is missing") from None
    if not description or len(description.encode("utf-8")) > 5000 or any(c in description for c in "<>"):
        raise UploadError("Description must be nonempty, at most 5000 UTF-8 bytes, and contain no < or >")
    if not isinstance(kit.get("tags"), list) or not all(isinstance(t, str) for t in kit["tags"]):
        raise UploadError("youtube.json must contain a list of text tags")
    for relative in ["video.mp4", THUMBNAILS[thumbnail], *[p for p, _ in CAPTIONS]]:
        path = folder / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise UploadError(f"Required upload file is missing or empty: {relative}")
    for review in (kit.get("review"), meta.get("review")):
        if not isinstance(review, dict):
            raise UploadError("Kit is missing its publishing review status")
    review_needed = any(r.get("needs_review") or r.get("reasons")
                        for r in (kit["review"], meta["review"]))
    return {"folder": folder, "message_id": int(mid), "telegram_channel_id": str(int(channel)),
            "title": title, "description": description, "tags": kit["tags"],
            "review_needed": bool(review_needed), "thumbnail": THUMBNAILS[thumbnail]}


def source_key(kit, channel):
    return f"{channel}:{kit['telegram_channel_id']}:{kit['message_id']}"


def source_marker(key):
    return "mezmur-source-" + hashlib.sha256(key.encode()).hexdigest()[:24]


def upload_tags(tags, marker):
    out, cost = [marker], len(marker)
    for value in tags:
        tag = " ".join(re.sub(r'[<>,"\r\n]', " ", value).split())
        if not tag or tag.casefold() in {t.casefold() for t in out}:
            continue
        extra = len(tag) + (2 if " " in tag else 0) + 1
        if cost + extra <= 500:
            out.append(tag)
            cost += extra
    return out


def save_record(store, key, record, previous):
    """Re-read before every checkpoint, preserving cursor/library and refusing concurrent edits."""
    snapshot = store.load()
    state = copy.deepcopy(snapshot["state"])
    uploads = state.setdefault("youtube_uploads", {})
    if not isinstance(uploads, dict) or uploads.get(key) != previous:
        raise UploadError("Upload state changed concurrently; refusing to overwrite it")
    uploads[key] = copy.deepcopy(record)
    store.save(snapshot, state)


def _video(service, video_id, channel, marker):
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise UploadError("Invalid YouTube video ID")
    items = service.videos().list(part="snippet,status", id=video_id).execute(num_retries=3).get("items", [])
    if (len(items) != 1 or items[0]["snippet"].get("channelId") != channel
            or marker not in items[0]["snippet"].get("tags", [])):
        raise UploadError("The existing video does not belong to this channel and Telegram song")
    return items[0]


def upload_song(kit, service, store, channel, visibility="private", recover_video_id=None):
    if visibility not in PRIVACIES:
        raise UploadError("Visibility must be private, unlisted or public")
    folder, key = kit["folder"], source_key(kit, channel)
    marker = source_marker(key)
    uploads = store.load()["state"].get("youtube_uploads", {})
    if not isinstance(uploads, dict):
        raise UploadError("Saved YouTube upload state is corrupt")
    previous = uploads.get(key)
    record = copy.deepcopy(previous) if previous else None
    if record and record.get("status") == "complete":
        write_json(folder / "youtube-upload.json", record)
        print(f"Song {kit['message_id']}: already uploaded, https://youtu.be/{record['video_id']}", flush=True)
        return record
    with (folder / "video.mp4").open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    if record and record.get("video_sha256") != digest:
        raise UploadError("An unfinished upload uses different video bytes; retry Publish Generated Videos with the original run's artifact")
    if record and record.get("status") not in ("uploading", "uploaded"):
        raise UploadError("Unknown upload status; inspect the state branch before retrying")
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
    if record is None:
        record = {"status": "uploading", "message_id": kit["message_id"],
                  "telegram_channel_id": kit["telegram_channel_id"], "youtube_channel_id": channel,
                  "video_sha256": digest, "source_marker": marker,
                  "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                  "created_at": datetime.now(timezone.utc).isoformat(), "requested_visibility": visibility,
                  "held_for_review": kit["review_needed"], "thumbnail_done": False, "captions": {}}
        # A durable reservation precedes all upload requests. Unknown outcomes never trigger a fresh insert.
        save_record(store, key, record, None)
        previous = copy.deepcopy(record)
        write_json(folder / "youtube-upload.json", record)
        body = {"snippet": {"title": kit["title"], "description": kit["description"],
                            "tags": upload_tags(kit["tags"], marker), "categoryId": "10",
                            "defaultLanguage": "am", "defaultAudioLanguage": "am"},
                "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False}}
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
    current = _video(service, video_id, channel, marker)
    if not record.get("thumbnail_done"):
        service.thumbnails().set(videoId=video_id,
                                 media_body=media(folder / kit["thumbnail"], "image/jpeg")).execute(num_retries=3)
        record["thumbnail_done"] = True
        record["thumbnail_file"] = kit["thumbnail"]
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)
    existing = service.captions().list(part="snippet", videoId=video_id).execute(num_retries=3).get("items", [])
    for relative, name in CAPTIONS:
        if relative in record["captions"]:
            continue
        matches = [c for c in existing if c["snippet"].get("language") == "am" and c["snippet"].get("name") == name]
        if matches:
            caption_id = matches[0]["id"]
        else:
            result = service.captions().insert(
                part="snippet", body={"snippet": {"videoId": video_id, "language": "am", "name": name, "isDraft": False}},
                media_body=media(folder / relative, "application/octet-stream")).execute(num_retries=0)
            caption_id = result["id"]
        record["captions"][relative] = caption_id
        save_record(store, key, record, previous)
        previous = copy.deepcopy(record)
    # Use the reservation's policy on retries; never accidentally publish by changing a retry input.
    requested = record["requested_visibility"]
    actual = "private" if record["held_for_review"] else requested
    if current.get("status", {}).get("privacyStatus") != actual:
        result = service.videos().update(part="status", body={
            "id": video_id, "status": {"privacyStatus": actual, "selfDeclaredMadeForKids": False}}).execute(num_retries=3)
        if result.get("status", {}).get("privacyStatus") != actual:
            raise UploadError("YouTube did not apply the requested visibility; inspect the existing video, do not re-upload")
    record.update(status="complete", privacy=actual, completed_at=datetime.now(timezone.utc).isoformat())
    save_record(store, key, record, previous)
    write_json(folder / "youtube-upload.json", record)
    print(f"Song {kit['message_id']}: {record['url']} ({actual})", flush=True)
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
    parser.add_argument("--visibility", choices=PRIVACIES, default=os.environ.get("YOUTUBE_PRIVACY") or "private")
    parser.add_argument("--thumbnail", choices=tuple(THUMBNAILS), default=os.environ.get("YOUTUBE_THUMBNAIL") or "a")
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
            raise UploadError("YOUTUBE_PRIVACY must be private, unlisted or public")
        if args.dry_run:
            for kit in kits:
                print(f"Validated song {kit['message_id']}; review hold={kit['review_needed']}; no upload performed")
            return
        channel = required("YOUTUBE_CHANNEL_ID")
        service = youtube_client()
        verify_channel(service, channel)
        store = Store()
    except Exception as error:
        print("::error::" + safe_error(error), flush=True)
        raise SystemExit(1) from None
    failed = False
    for kit in kits:
        try:
            record = upload_song(kit, service, store, channel, args.visibility, args.recover_video_id)
            line = f"- Song {kit['message_id']}: [YouTube video]({record['url']}) ({record['privacy']})"
            if record.get("held_for_review"):
                line += " - kept private for review"
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
