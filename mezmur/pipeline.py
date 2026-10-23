import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil

from .common import duration, required, write_json
from .lyrics import make_cues, write_lyrics
from .render import render
from .telegram import client, download, resolve_channel, select, song_info
from .transcribe import MODEL, separate, transcribe

KEYWORDS = ["Protestant mezmur", "Amharic mezmur lyrics", "Ethiopian gospel music",
            "Amharic worship", "የፕሮቴስታንት መዝሙር", "የአምልኮ መዝሙር", "መዝሙር ግጥም"]


def description(info, lyrics):
    artist = f"Artist: {info['artist']}\n" if info["artist"] else ""
    return (f"{info['title']} | Amharic Protestant Mezmur Lyrics\n\n{artist}\n"
            f"Lyrics\n{lyrics}\nKeywords: {', '.join(KEYWORDS)}\n\n"
            "#ProtestantMezmur #AmharicMezmur #EthiopianGospel #Worship\n")


async def process(args):
    # A failed/no-op local rerun must not leave an old candidate available to save.
    Path(".runtime/pending.json").unlink(missing_ok=True)
    required("GROQ_API_KEY")
    snapshot = json.loads(Path(".runtime/snapshot.json").read_text())
    state = snapshot["state"]
    output = Path("output")
    output.mkdir(exist_ok=True)
    telegram = client()
    await telegram.connect()
    try:
        if not await telegram.is_user_authorized():
            raise ValueError("Telegram session expired; generate a new TELEGRAM_SESSION locally")
        channel = required("TELEGRAM_CHANNEL")
        entity = await resolve_channel(telegram, channel)
        if not getattr(entity, "broadcast", False):
            raise ValueError("TELEGRAM_CHANNEL must resolve to a Telegram broadcast channel")
        key = str(entity.id)
        after = state["channels"].get(key, {}).get("last_message_id", 0)
        if not isinstance(after, int) or after < 0:
            raise ValueError("Invalid saved Telegram cursor")
        messages = await select(telegram, entity, after, args.batch_size)
        print(f"Selected {len(messages)} song(s) after message {after}", flush=True)
        completed = []
        for message in messages:
            info = song_info(message)
            print(f"Processing message {message.id}", flush=True)
            scratch = Path(".runtime") / f"song-{message.id}"
            scratch.mkdir(parents=True, exist_ok=True)
            audio = await download(telegram, message, scratch / "source.audio")
            seconds = duration(audio)
            if seconds > args.max_duration:
                raise ValueError(f"Message {message.id} exceeds --max-duration; cursor unchanged")
            vocal_audio = separate(audio, scratch) if args.separate_vocals else audio
            words = transcribe(vocal_audio, seconds, scratch)
            cues = make_cues(words)
            lyrics = write_lyrics(scratch, cues, info["title"], seconds)
            render(audio, scratch, seconds, args.height)
            info.update({
                "duration_seconds": seconds, "model": MODEL, "language": "am",
                "vocal_separation": args.separate_vocals, "keywords": KEYWORDS,
                "review_required": True,
                "review_note": "Automatic singing transcription and word timings need human review.",
            })
            write_json(scratch / "metadata.json", info)
            write_json(scratch / "words.json", words)
            (scratch / "description.txt").write_text(description(info, lyrics), encoding="utf-8")
            destination = output / f"song-{message.id}"
            destination.mkdir(exist_ok=False)
            for name in ["video.mp4", "lyrics.txt", "lyrics.srt", "lyrics.ass", "words.json",
                         "transcription.raw.json", "metadata.json", "description.txt"]:
                shutil.copy2(scratch / name, destination / name)
            completed.append(info)
            # Stems and original channel media are temporary, never uploaded as artifacts.
            shutil.rmtree(scratch)
        if completed:
            state["channels"][key] = {
                "last_message_id": messages[-1].id,
                "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
                "artifact_name": f"mezmur-{os.environ.get('GITHUB_RUN_ID', 'local')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}",
            }
            write_json(output / "manifest.json", {"songs": completed, "channel_id": key})
            # The workflow commits this candidate only AFTER a successful artifact upload.
            write_json(Path(".runtime/pending.json"), state)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
                handle.write(f"has_output={'true' if completed else 'false'}\n")
    finally:
        await telegram.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Generate the next batch of Amharic lyric videos")
    parser.add_argument("--batch-size", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--height", type=int, default=1080, choices=[720, 1080])
    parser.add_argument("--max-duration", type=int, default=1800)
    parser.add_argument("--separate-vocals", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 1 <= args.max_duration <= 7200:
        parser.error("--max-duration must be between 1 and 7200 seconds")
    asyncio.run(process(args))


if __name__ == "__main__":
    main()
