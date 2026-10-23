"""Run locally once. Treat the resulting session string like a password."""
import asyncio
from getpass import getpass

from telethon import TelegramClient
from telethon.sessions import StringSession


async def main():
    api_id = int(input("Telegram API ID (my.telegram.org): "))
    api_hash = getpass("Telegram API hash: ")
    client = TelegramClient(StringSession(), api_id, api_hash)
    try:
        await client.start(phone=lambda: input("Phone number with country code: "))
        print("\nSave the following as the TELEGRAM_SESSION repository secret:\n")
        print(client.session.save())
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
