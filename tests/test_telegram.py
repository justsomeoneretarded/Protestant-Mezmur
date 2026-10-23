import asyncio
from types import SimpleNamespace as NS

from mezmur.telegram import select


def test_oldest_three_audio_messages_then_resume():
    messages = [NS(id=i, document=NS(mime_type=mime), audio=None, voice=voice)
                for i, mime, voice in [(1, "image/jpeg", False), (2, "audio/mpeg", False),
                                      (3, "audio/ogg", True), (4, "audio/mpeg", False),
                                      (5, "audio/flac", False), (9, "audio/mpeg", False)]]

    class Client:
        async def iter_messages(self, entity, min_id, reverse):
            assert reverse is True
            for message in messages:
                if message.id > min_id:
                    yield message

    first = asyncio.run(select(Client(), object(), 0, 3))
    second = asyncio.run(select(Client(), object(), first[-1].id, 3))
    assert [m.id for m in first] == [2, 4, 5]
    assert [m.id for m in second] == [9]
