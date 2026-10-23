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


async def resolve_channel(client, channel):
    if channel.lstrip("-").isdigit():
        target = int(channel)
        # StringSession does not persist the entity/access-hash cache.
        async for dialog in client.iter_dialogs():
            if dialog.entity.id == target or get_peer_id(dialog.entity) == target:
                return dialog.entity
        raise ValueError("Private channel ID not found among this Telegram account's dialogs")
    return await client.get_entity(channel)


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
    title, artist = "", ""
    for attribute in message.document.attributes:
        title = getattr(attribute, "title", "") or title
        artist = getattr(attribute, "performer", "") or artist
    if not title:
        title = ((message.raw_text or "").splitlines() or [""])[0].strip()
    return {
        "message_id": message.id,
        "title": (title or f"Mezmur {message.id}")[:160],
        "artist": artist[:120],
    }


async def download(client, message, target: Path):
    if message.document.size > 250 * 1024 * 1024:
        raise ValueError(f"Message {message.id} exceeds the 250 MiB download limit")
    path = await client.download_media(message, file=str(target))
    if not path:
        raise RuntimeError(f"No audio downloaded for message {message.id}")
    return Path(path)
