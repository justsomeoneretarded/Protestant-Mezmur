import json

import responses

from mezmur import transcribe


@responses.activate
def test_api_requests_amharic_words_and_returns_original_timeline(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")

    def fake_ffmpeg(args):
        from pathlib import Path
        Path(args[-1]).write_bytes(b"fake wav bytes")

    monkeypatch.setattr(transcribe, "run", fake_ffmpeg)
    endpoint = "https://api.groq.com/openai/v1/audio/transcriptions"
    responses.post(endpoint, json={"words": [{"word": "ሃሌሉያ", "start": 2, "end": 3}]})
    responses.post(endpoint, json={"words": [{"word": "አሜን", "start": 6, "end": 7}]})
    words = transcribe.transcribe(tmp_path / "audio.wav", 300, tmp_path)
    assert [word["start"] for word in words] == [2, 242]
    body = responses.calls[0].request.body.decode("utf-8")
    for expected in ['name="language"\r\n\r\nam', 'name="model"\r\n\r\nwhisper-large-v3',
                     'name="timestamp_granularities[]"\r\n\r\nword']:
        assert expected in body
    assert not list(tmp_path.glob("chunk-*.wav"))
    assert len(json.loads((tmp_path / "transcription.raw.json").read_text())) == 2
