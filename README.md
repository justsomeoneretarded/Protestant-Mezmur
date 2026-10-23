# Protestant Mezmur

Manually turn the next three songs from a Telegram channel into Amharic lyric
videos using GitHub Actions. Each run starts after the last completed batch.
The first run begins with the oldest available audio messages in channel history.
Transcription runs locally with open-source Whisper weights and faster-whisper.
No hosted speech service, paid transcription API, or transcription API key is used.

## What you download

Under **Actions > Generate Mezmur Videos > a run > Artifacts**, download
`mezmur-RUN_ID-ATTEMPT`. Each `song-MESSAGE_ID/` contains:

| File | Contents |
| --- | --- |
| `video.mp4` | 1080p or 720p, moving color field, audio waveform, Amharic karaoke lyrics, original music mix |
| `lyrics.txt` | Plain UTF-8 lyrics |
| `lyrics.srt` | Timed lyrics for editors and video platforms |
| `lyrics.ass` | Styled lyrics with word highlighting |
| `description.txt` | Song title, known artist, full lyrics, SEO keywords and hashtags |
| `words.json` | Word timestamps in seconds relative to the original audio |
| `transcription.raw.json` | Original recognizer results for reviewing transcription |
| `metadata.json` | Source message ID, title, model, keywords and review status |

The artifact also contains `manifest.json`. Downloads expire after 30 days;
download them before expiration. Videos are not automatically published.

## One-time setup

1. Create a Telegram application at <https://my.telegram.org/apps> to obtain
   your API ID and API hash. Use a Telegram account that can read the source
   channel's history. For a private channel, join it first.
2. On your own computer, install Python 3.11+, clone this repository, then run:

   ```sh
   python -m venv .venv
   # Linux/macOS:
   source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -e .
   python scripts/telegram_session.py
   ```

   Enter your Telegram phone number, login code and two-factor password when
   prompted. Save the printed session string directly into GitHub Secrets.
   Run this helper locally, never in an Actions log. A session grants access to
   your Telegram account; do not commit or share it. Telegram's Devices settings
   can revoke it.
3. Under **Settings > Secrets and variables > Actions**, add these four repository secrets:

   | Secret | Value |
   | --- | --- |
   | `TELEGRAM_API_ID` | Telegram application numeric ID |
   | `TELEGRAM_API_HASH` | Telegram application hash |
   | `TELEGRAM_SESSION` | String printed by the local helper |
   | `TELEGRAM_CHANNEL` | Public `@channelusername` or joined private channel ID such as `-1001234567890` |

4. Allow GitHub Actions in repository settings. The workflow requests
   `contents: write` so its built-in `GITHUB_TOKEN` can save progress. An
   organization policy restricting writes or branch creation must permit
   the `mezmur-state` branch. No personal GitHub token is needed.
5. Open **Actions > Generate Mezmur Videos > Run workflow** on the default branch.
   Start with one song to check recognition quality and CPU processing time,
   then use the default batch of three.

## Transcription and synchronization

The recognizer is **Whisper large-v3**, executed locally through the open-source
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) library and CTranslate2.
Whisper supports Amharic (`am`); we force that language and compute word and
segment timestamps. CPU inference uses INT8 quantization, two threads and one
model instance reused across songs. No model has been benchmarked on your channel
yet, so this is a practical multilingual baseline rather than a measured claim
about accuracy on sung mezmur.

The public model weights download from Hugging Face on first use (several GB).
GitHub Actions caches them in `.model-cache/whisper` and caches Demucs weights
separately. Cache eviction can cause a fresh download. Audio stays on the runner
during recognition; downloading weights does not send your songs to a speech API.

Demucs `htdemucs` separates vocals on the GitHub CPU runner before transcription
by default. The original music, not the separated stem, goes into the final video.
Separation is timeline-preserving. You can turn it off in the Run workflow form.
Audio is transcribed in four-minute chunks with four-second overlaps. Each chunk
is mono 16 kHz PCM. Speech VAD is disabled to avoid dropping sustained or soft
singing, and previous-text conditioning is disabled to reduce repetition loops.
Word timestamps are shifted back onto the full song timeline, then grouped into
short lyric lines.
Lyrics disappear during detected gaps instead of remaining across instrumental
sections. Background motion and waveform are generated locally by FFmpeg;
Noto Sans Ethiopic supplies the glyphs.

Singing, choirs, strong reverb and accompaniment can produce incorrect words,
hallucinations or inaccurate timings, including after separation. These are
automatic draft lyrics: review the Amharic and timing before publishing. The
pipeline fails if no timed words are returned; it does not invent lyric timings
from an untimed text response. Edit `lyrics.ass` to correct the burned-in lyrics
and rerender locally from the original song audio:

```sh
python -m mezmur.render --audio original-song.mp3 --folder output/song-MESSAGE_ID
```

This replaces `video.mp4` in that folder. Use the original audio rather than the
already subtitled MP4. `lyrics.srt` is also available for your editor. Editing the
ASS file does not automatically update the separate TXT/SRT/description files.

References: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[Whisper languages](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py),
[Demucs](https://github.com/facebookresearch/demucs),
[Telethon history](https://docs.telethon.dev/en/stable/modules/client.html#telethon.client.messages.MessageMethods.iter_messages).

## How progress and retries work

- Audio documents/music are read in ascending Telegram message ID order.
  Images, text, videos and voice notes are ignored. Titles and artists use
  Telegram audio attributes, falling back to the caption and message ID.
- Progress is keyed by the resolved channel ID on the separate `mezmur-state`
  branch. Changing the channel starts that channel from its oldest audio;
  returning to a previous channel resumes its saved position.
- The entire batch must finish and the artifact upload must succeed before
  progress advances. A single concurrency group serializes generation across
  branches. State writes also reject concurrent changes.
- If a batch fails, completed songs are uploaded when possible, but the whole
  batch is retried next time. This deliberately allows duplicates rather than
  skipping a failed song. A cancellation after upload but before the state
  commit can likewise produce a duplicate batch. Exactly-once delivery across
  separate GitHub services is not promised.
- A state-save error leaves the artifact available; fix the write permissions
  before rerunning. Do not delete the state branch unless you intend to restart
  all channels. Never store sessions or keys on that branch.
- Deleted or inaccessible older messages cannot be recovered. New edits to
  already processed messages do not cause reprocessing. A repost with a new
  message ID counts as a new song. No-new-songs runs finish without an artifact.

## Limits and troubleshooting

GitHub hosted runners use CPU. Full large-v3 inference, Demucs and 1080p rendering
may take substantial time, potentially longer than the audio duration. There is
no GPU or fast turnaround guarantee. The job timeout is 330 minutes. Choose 720p,
a smaller batch, or disable separation to reduce runtime. Default limits are
250 MiB and 30 minutes per song; an oversized song stops the batch and leaves
progress unchanged. Adjust `--max-duration` or the download limit deliberately
if your channel has longer recordings. The pipeline processes one song at a time
and removes stems and source files after rendering to limit disk use.

There are no transcription service charges or speech API quotas. GitHub Actions
minutes, artifact storage and model cache storage still follow your GitHub plan's
limits and pricing. Model download or inference failures stop the batch without
advancing progress. Telegram still requires its free API ID, hash and user session
to access channel history; replacing the speech API does not replace Telegram
authentication. A public channel URL is not a substitute for Telegram login:
the Bot API cannot read old channel
history. Numeric private channel IDs must belong to the logged-in user's dialogs.
Only process music you have permission to republish.

## Development

Install FFmpeg, Fontconfig and Noto Sans Ethiopic (`fonts-noto-core` on Ubuntu), then:

```sh
pip install -e '.[test]'
pytest -q
```

Tests cover chronological selection/resume, subtitle timing, chunk boundaries,
state conflicts, and a real FFmpeg render with Amharic text, audio and moving
pixels. Tests use synthetic audio and mocked services, not a live Telegram
account or a claim of measured Amharic transcription accuracy.

For a local production run, export the four Telegram secrets above plus
`GITHUB_REPOSITORY=owner/repo` and `GH_TOKEN` with repository write permission:

```sh
python -m mezmur.state load
mezmur --batch-size 3 --no-separate-vocals
# Verify and back up output/ before saving progress locally:
python -m mezmur.state save
```

For local separation install matching CPU PyTorch/Torchaudio 2.5.1 and
`pip install -e '.[vocals]'`, as in the workflow. Never upload `.runtime/`.
