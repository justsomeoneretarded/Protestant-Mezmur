"""Tell a Telegram channel/chat that a song's video is ready (or failed).

Needs the repository secret TELEGRAM_BOT_TOKEN; the bot must be an admin of the channel.
The chat comes from TELEGRAM_NOTIFY_CHAT (e.g. @MyApplication_aman). Never fails the job.
"""
import argparse
import html
import json
import os
from pathlib import Path

import requests


def message(folder, run_url, ok, retention_days, message_id):
    folder = Path(folder)
    title_file = folder / "title.txt"
    title = title_file.read_text(encoding="utf-8").splitlines()[0].strip() if title_file.is_file() else ""
    title = title or f"Telegram message {message_id}"
    if not ok:
        return (f"❌ <b>Video failed</b>\n\nSong: {html.escape(title)}\n\n"
                f"Details: <a href=\"{run_url}\">GitHub run link</a>")
    lines = ["✅ <b>Your video is ready!</b>", "", f"Title: {html.escape(title)}"]
    meta_file = folder / "metadata.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        reasons = (meta.get("review") or {}).get("reasons", [])
        if meta.get("lyrics_mode") != "lyrics-aligned":
            lines.append("⚠️ Lyrics are an automatic transcription — check them before publishing.")
        elif reasons:
            lines.append(f"⚠️ {len(reasons)} thing(s) to check — see the run summary.")
    days = f"{retention_days} day" + ("s" if retention_days != 1 else "")
    lines += ["", f"Download: <a href=\"{run_url}\">GitHub run link</a>", "",
              f"Open Artifacts to download. Available for {days}."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--message-id", default="")
    p.add_argument("--status", choices=["success", "failure"], default="success")
    p.add_argument("--retention-days", type=int, default=3)
    a = p.parse_args()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_NOTIFY_CHAT", "").strip()
    if not token or not chat:
        print("Telegram notification skipped (TELEGRAM_BOT_TOKEN secret or chat not set).")
        return
    run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}" \
              f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    text = message(a.folder, run_url, a.status == "success", a.retention_days, a.message_id)
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=30, json={
            "chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True})
        ok = r.status_code == 200 and r.json().get("ok")
        print("Telegram notification sent." if ok else f"Telegram notification failed: {r.status_code} {r.text[:200]}")
    except requests.RequestException as error:
        print(f"Telegram notification failed: {error.__class__.__name__}")


if __name__ == "__main__":
    main()
