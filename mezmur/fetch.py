"""Telegram step: select songs, download audio + nearby texts/photos, search lyrics channels.

Only this step uses the Telegram session, so per-song jobs can run in parallel safely.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from .common import required, write_json
from .geez import has_ethiopic
from .names import resolve
from .telegram import client, download, is_song, resolve_channel, select, song_info

DEFAULT_LYRICS_CHANNELS = ("MezmurLyrics0,mezmur1,Amharic_spiritual_song_lyrics,SongLegetaLyrics,mezmurlyricsbb,"
                           "mezmur_lyrics,lyrics_mezmur,mezmur_lyrics2,mezmur_gitem,eyesusyiwedenal,mezmuroch1")


def _ext(message):
    name = song_info(message).get("filename") or ""
    if "." in name:
        return name.rsplit(".", 1)[1].lower()[:5]
    mime = (message.document.mime_type or "").split("/")[-1]
    return {"mpeg": "mp3", "mp4": "m4a", "x-m4a": "m4a", "ogg": "ogg", "flac": "flac", "wav": "wav"}.get(mime, "mp3")


async def neighbours(tg, entity, message, folder, span=6):
    texts, photos = [], []
    caption = message.raw_text or ""
    if caption.strip():
        texts.append({"where": "caption", "text": caption, "link": ""})
    ids = [i for i in range(message.id - span, message.id + span + 1) if i > 0 and i != message.id]
    nearby = [m for m in await tg.get_messages(entity, ids=ids) if m]
    for m in nearby:
        if is_song(m):
            continue
        replies = getattr(m, "reply_to_msg_id", None) == message.id
        grouped = message.grouped_id and m.grouped_id == message.grouped_id
        if m.raw_text and has_ethiopic(m.raw_text, 20):
            texts.append({"where": "reply" if replies else ("album" if grouped else f"nearby {m.id - message.id:+d}"),
                          "text": m.raw_text, "link": ""})
        if m.photo and len(photos) < 3 and (abs(m.id - message.id) <= 2 or replies or grouped):
            path = await tg.download_media(m, file=str(folder / "photos" / f"tg-photo-{m.id}.jpg"))
            if path:
                photos.append(Path(path).name)
    if message.document and message.document.thumbs:
        try:
            path = await tg.download_media(message, file=str(folder / "photos" / "tg-thumb.jpg"), thumb=-1)
            if path:
                photos.append(Path(path).name)
        except Exception as error:
            print(f"  thumb failed: {error}", flush=True)
    return texts, photos


def _title_queries(record):
    qs = []
    t_am, t_lat = record.get("title_am", ""), record.get("title_latin", "")
    if t_am:
        qs.append(t_am)
        words = t_am.split()
        if len(words) == 2:
            qs.append(f"{words[1]} {words[0]}")      # channels often flip two-word titles
        longest = max(words, key=len)
        if len(words) > 1 and len(longest) >= 4:
            qs.append(longest)
    if t_lat:
        qs.append(t_lat)
    out = []
    for q in qs:
        if q and q not in out:
            out.append(q)
    return out[:4]


async def search_channels(tg, record, channels):
    """Search known lyrics channels, then every chat this Telegram account has joined."""
    queries = _title_queries(record)
    out, seen = [], set()

    def keep(m, where, link):
        if m.raw_text and has_ethiopic(m.raw_text, 30) and (where, m.id) not in seen:
            seen.add((where, m.id))
            out.append({"where": where, "text": m.raw_text, "link": link})

    for name in channels:
        try:
            entity = await tg.get_entity(name)
        except Exception as error:
            print(f"  lyrics channel {name}: {error.__class__.__name__}", flush=True)
            continue
        for q in queries:
            try:
                for m in await tg.get_messages(entity, search=q, limit=5):
                    keep(m, f"channel @{name}", f"https://t.me/{name}/{m.id}")
            except Exception as error:
                print(f"  search {name} {q!r}: {error.__class__.__name__}", flush=True)
            await asyncio.sleep(0.4)
    for q in queries[:2]:
        try:  # global search: all channels/groups the account is a member of
            async for m in tg.iter_messages(None, search=q, limit=15):
                chat = getattr(m.chat, "username", None) or getattr(m.chat, "title", "chat")
                link = f"https://t.me/{chat}/{m.id}" if getattr(m.chat, "username", None) else ""
                keep(m, f"joined chat {chat}", link)
        except Exception as error:
            print(f"  global search {q!r}: {error.__class__.__name__}", flush=True)
        await asyncio.sleep(0.6)
    return out


async def run(args):
    snapshot = json.loads(Path(".runtime/snapshot.json").read_text())
    state = snapshot["state"]
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    tg = client()
    await tg.connect()
    try:
        if not await tg.is_user_authorized():
            raise ValueError("Telegram session expired; generate a new TELEGRAM_SESSION locally")
        entity = await resolve_channel(tg, required("TELEGRAM_CHANNEL"))
        if not getattr(entity, "broadcast", False):
            raise ValueError("TELEGRAM_CHANNEL must resolve to a Telegram broadcast channel")
        key = str(entity.id)
        after = state["channels"].get(key, {}).get("last_message_id", 0)
        replay = [int(x) for x in str(args.message_ids or "").replace(" ", "").split(",") if x.strip().isdigit() and int(x)]
        if replay:
            messages = [m for m in await tg.get_messages(entity, ids=replay) if m and is_song(m)]
            if len(messages) != len(replay):
                raise ValueError("A replay message ID is not an accessible audio post")
        else:
            messages = await select(tg, entity, after, args.batch_size)
        print(f"Selected {len(messages)} song(s) after message {after}", flush=True)
        channels = []
        for c in (os.environ.get("LYRICS_CHANNELS") or DEFAULT_LYRICS_CHANNELS).split(",") + state.get("lyrics_channels", []):
            c = c.strip().lstrip("@")
            if c and c not in channels:
                channels.append(c)  # channels that supplied verified lyrics before are searched too
        used = state.get("used_backgrounds", [])
        ids = []
        for message in messages:
            folder = root / f"song-{message.id}"
            (folder / "photos").mkdir(parents=True, exist_ok=True)
            info = song_info(message)
            info.update({"caption": message.raw_text or "", "date": str(message.date),
                         "channel": getattr(entity, "username", "") or key,
                         "link": f"https://t.me/{entity.username}/{message.id}" if getattr(entity, "username", None) else ""})
            await download(tg, message, folder / f"audio.{_ext(message)}")
            texts, photos = await neighbours(tg, entity, message, folder)
            record = resolve(info)
            texts += await search_channels(tg, record, channels)
            write_json(folder / "info.json", info)
            write_json(folder / "telegram.json", {"texts": texts, "photos": photos})
            write_json(folder / "used_backgrounds.json", used)
            print(f"  {message.id}: {record.get('title_am') or record.get('title_latin')} — {len(texts)} text(s), "
                  f"{len(photos)} photo(s)", flush=True)
            ids.append(message.id)
        if ids and not replay:
            state["channels"][key] = {"last_message_id": messages[-1].id, "run_id": os.environ.get("GITHUB_RUN_ID", "local")}
        write_json(root / "plan.json", {"ids": ids, "channel_id": key, "replay": bool(replay), "snapshot": snapshot,
                                        "pending_state": state})
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
                fh.write(f"ids={json.dumps(ids)}\nhas_songs={'true' if ids else 'false'}\n")
    finally:
        await tg.disconnect()


def main():
    p = argparse.ArgumentParser(description="Fetch the next songs from Telegram")
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--message-ids", default="", help="Comma-separated message IDs to replay (cursor unchanged)")
    p.add_argument("--output", default="inputs")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
