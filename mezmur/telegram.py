from pathlib import Path

from telethon import TelegramClient
from telethon.sessions import StringSession
from telethon.utils import get_peer_id

from .common import required


def client():
    return TelegramClient(
        StringSession(required("TELEGRAM_SESSION")),
        int(required("TELEGRAM_API_ID")), required("TELEGRAM_API_HASH"),
        flood_sleep_threshold=120, request_retries=5,
    )


class PeerCache:
    """Find channels with as few Telegram username lookups as possible.

    Telegram rate-limits username lookups (ResolveUsername) per account and answers with long waits
    when a few runs do many of them. Channels the account has joined come from one dialog listing
    (no lookups); every other channel is looked up once and remembered in the saved state
    (id + access hash), so later runs need no lookup at all. After a rate-limit, no further lookups
    are attempted in this run."""

    def __init__(self, client, saved=None):
        self.client = client
        self.saved = saved if isinstance(saved, dict) else {}
        self.joined = None
        self.blocked = False

    async def _load_joined(self):
        if self.joined is not None:
            return
        self.joined = {}
        try:
            async for dialog in self.client.iter_dialogs():
                ent = dialog.entity
                name = (getattr(ent, "username", None) or "").lower()
                if name:
                    self.joined[name] = ent
                self.joined[str(ent.id)] = ent
                self.joined[str(get_peer_id(ent))] = ent
        except Exception as error:  # noqa: BLE001 - fall back to lookups
            print(f"  could not list joined chats: {error.__class__.__name__}", flush=True)

    async def get(self, name):
        from telethon.errors import FloodWaitError
        from telethon.tl.types import InputPeerChannel
        key = str(name).strip().lstrip("@").lower()
        await self._load_joined()
        if key in self.joined:
            return self.joined[key]
        if key.lstrip("-").isdigit():
            raise ValueError("Private channel ID not found among this Telegram account's dialogs")
        hit = self.saved.get(key)
        if isinstance(hit, list) and len(hit) == 2:
            return InputPeerChannel(int(hit[0]), int(hit[1]))
        if self.blocked:
            raise RuntimeError("Telegram username lookups are rate-limited for now")
        try:
            ent = await self.client.get_entity(key)
        except FloodWaitError as error:
            self.blocked = True
            print(f"  Telegram asked to wait {error.seconds // 60} min before more channel lookups; "
                  "skipping the remaining lookups this run", flush=True)
            raise
        if getattr(ent, "access_hash", None) is not None:
            self.saved[key] = [ent.id, ent.access_hash]
        return ent


async def resolve_channel(client, channel, cache=None):
    cache = cache or PeerCache(client)
    try:
        return await cache.get(channel)
    except Exception as error:
        from telethon.errors import FloodWaitError
        if isinstance(error, FloodWaitError):
            raise ValueError(f"Telegram temporarily blocked looking up {channel} (wait about "
                             f"{error.seconds // 60} minutes, then run again). Joining that channel with "
                             "this Telegram account avoids the lookup completely.") from None
        raise


def is_song(message) -> bool:
    document = getattr(message, "document", None)
    return bool(document and not getattr(message, "voice", False) and (
        getattr(message, "audio", None) or (document.mime_type or "").startswith("audio/")
    ))


async def select(client, entity, after: int, count: int):
    selected = []
    async for message in client.iter_messages(entity, min_id=after, reverse=True):
        if is_song(message):
            selected.append(message)
            if len(selected) == count:
                break
    return selected


def song_info(message):
    """Raw Telegram fields; names.resolve() turns them into clean titles/artists."""
    title, artist, filename, seconds = "", "", "", 0
    for attribute in message.document.attributes:
        title = getattr(attribute, "title", "") or title
        artist = getattr(attribute, "performer", "") or artist
        filename = getattr(attribute, "file_name", "") or filename
        seconds = getattr(attribute, "duration", 0) or seconds
    return {
        "message_id": message.id,
        "title": (title or "")[:200],
        "performer": (artist or "")[:160],
        "filename": filename[:200],
        "duration": seconds,
    }


async def download(client, message, target: Path):
    if message.document.size > 250 * 1024 * 1024:
        raise ValueError(f"Message {message.id} exceeds the 250 MiB download limit")
    path = await client.download_media(message, file=str(target))
    if not path:
        raise RuntimeError(f"No audio downloaded for message {message.id}")
    return Path(path)
