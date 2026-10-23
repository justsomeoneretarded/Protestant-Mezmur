# Protestant Mezmur

Manually turn the next three songs from a Telegram channel into Amharic lyric
videos using GitHub Actions. Each run starts after the last completed batch.
The first run begins with the oldest available audio messages in channel history.
The configured source is [@amharicprotestantmezmurss](https://t.me/amharicprotestantmezmurss).
Transcription runs locally with the Amharic-specific Ethio-ASR model.
No hosted speech service, paid transcription API, or transcription API key is used.

## What you download

Under **Actions > Generate Mezmur Videos > a run > Artifacts**, download
`mezmur-RUN_ID-ATTEMPT`. Each `song-MESSAGE_ID/` contains:

| File | Contents |
| --- | --- |
| `video.mp4` | 1080p or 720p, moving scenic backgrounds, crossfades, reactive spectrum, Amharic karaoke lyrics, original music mix |
| `lyrics.txt` | Plain UTF-8 lyrics |
| `lyrics.srt` | Timed lyrics for editors and video platforms |
| `lyrics.ass` | Styled lyrics with word highlighting |
| `description.txt` | Song title, known artist, full lyrics, SEO keywords and hashtags |
| `words.json` | Word timestamps in seconds relative to the original audio |
| `transcription.raw.json` | Original recognizer results for reviewing transcription |
| `metadata.json` | Source message ID, title, model, keywords and review status |
| `quality.json` | Repetition checks, uncertain-word count, first detected lyric time and review status |

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
   pip install Telethon==1.42.0
   python scripts/telegram_session.py
   ```

   Enter your Telegram phone number, login code and two-factor password when
   prompted. Save the printed session string directly into GitHub Secrets.
   Run this helper locally, never in an Actions log. A session grants access to
   your Telegram account; do not commit or share it. Telegram's Devices settings
   can revoke it.
3. Under **Settings > Secrets and variables > Actions > Secrets**, add these three repository secrets:

   | Secret | Value |
   | --- | --- |
   | `TELEGRAM_API_ID` | Telegram application numeric ID |
   | `TELEGRAM_API_HASH` | Telegram application hash |
   | `TELEGRAM_SESSION` | String printed by the local helper |

   The channel is public configuration, not a secret. The workflow defaults to
   `@amharicprotestantmezmurss`; no channel setting is required for this source.
   To change it, use the **Variables** tab to create a repository variable named
   `TELEGRAM_CHANNEL`, containing a public `@channelusername` or joined private
   channel ID such as `-1001234567890`.

4. Allow GitHub Actions in repository settings. The workflow requests
   `contents: write` so its built-in `GITHUB_TOKEN` can save progress. An
   organization policy restricting writes or branch creation must permit
   the `mezmur-state` branch. No personal GitHub token is needed.
5. Open **Actions > Generate Mezmur Videos > Run workflow** on the default branch.
   Start with one song to check recognition quality and CPU processing time,
   then use the default batch of three.

## Transcription and synchronization

The recognizer is **[Ethio-ASR Amharic](https://huggingface.co/badrex/Ethio-ASR-amharic)**,
a 600M-parameter Wav2Vec2-BERT CTC model specialized for Amharic. Its authors report
22.37% word error rate on the WAXAL Amharic speech test set. That is a speech
benchmark, not a mezmur/singing accuracy guarantee. The model is licensed
CC-BY-4.0 by the Ethio-ASR team; see their model card for attribution and the
[research paper](https://arxiv.org/abs/2603.23654).

The previous generic Whisper large-v3 implementation produced repeated-character
hallucinations on the first real channel song and has been replaced. The new
model was probed on that actual recording, including its instrumental intro;
it produces varied Amharic words, but some words still require correction.
Recognition uses CPU PyTorch, two threads, float32 and one cached model instance.

The public model weights download from Hugging Face on first use (several GB).
GitHub Actions caches them in `.model-cache/ethio-asr` and caches Demucs weights
separately. Cache eviction can cause a fresh download. Audio stays on the runner
during recognition; downloading weights does not send your songs to a speech API.

Demucs `htdemucs` separates vocals on the GitHub CPU runner before transcription
by default. The original music, not the separated stem, goes into the final video.
Separation is timeline-preserving. You can turn it off in the Run workflow form.
Audio is transcribed in 20-second chunks with two-second overlaps. Each chunk is
mono 16 kHz PCM. The CTC tokenizer supplies acoustic frame offsets for words;
we convert those positions to time and restore the full-song offset. Blank
frames stay uncaptioned. Low-confidence words and obvious repeated-character
hissing are rejected, rather than assigning the intro artificial lyric timings.
First detected lyric time is recorded in `quality.json` and `metadata.json`.
Confidence values are model scores, not a promise that a word is correct.

Obvious character/word collapse fails the batch before rendering or saving
progress. Failure artifacts contain `failed-song-ID/quality.json` and raw
transcription diagnostics. Checks are deliberately narrow; plausible-looking
wrong words can still pass, so every output requires lyric review.

The bundled, generated scenic artwork contains highland mountains, a skyscraper
skyline, a winding road through red earth and a desert. Continuous camera pans,
20-second scene changes and two-second crossfades animate it locally in FFmpeg;
the spectrum and waveform react to the actual music. These are generated still
scenes with camera motion, not live footage or an interactive video. No image API
is used during workflow runs. The center is shaded for contrast, with bold Noto
Serif Ethiopic lyrics, white upcoming words, warm-gold karaoke highlighting and
short fades. Lyrics stay absent until the first detected word.

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

References: [Ethio-ASR model](https://huggingface.co/badrex/Ethio-ASR-amharic),
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
- To regenerate a previously processed song, enter its Telegram message ID in
  **Run workflow > message_id**. For the first failed transcription, use `2`.
  Replay generates a new artifact and does not change any channel's bookmark.
- Deleted or inaccessible older messages cannot be recovered. New edits to
  already processed messages do not cause reprocessing. A repost with a new
  message ID counts as a new song. No-new-songs runs finish without an artifact.

## Limits and troubleshooting

GitHub hosted runners use CPU. Ethio-ASR inference, Demucs and 1080p rendering
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

Install FFmpeg, Fontconfig and Noto Serif/Sans Ethiopic (`fonts-noto-core` on Ubuntu), then:

```sh
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
pytest -q
```

Tests cover chronological selection/resume, subtitle timing, chunk boundaries,
state conflicts, replay without cursor changes, repeated-character rejection,
instrumental blank-frame timing and a real scenic FFmpeg render with Amharic
text and audio. Unit tests use synthetic audio and mocked inference; the manual
model probe used the real first channel recording. Neither is a measured singing
accuracy benchmark.

For a local production run, export the three Telegram secrets above plus
`TELEGRAM_CHANNEL=@amharicprotestantmezmurss`,
`GITHUB_REPOSITORY=owner/repo` and `GH_TOKEN` with repository write permission:

```sh
python -m mezmur.state load
mezmur --batch-size 3 --no-separate-vocals
# Verify and back up output/ before saving progress locally:
python -m mezmur.state save
```

For local separation install matching CPU PyTorch/Torchaudio 2.5.1 and
`pip install -e '.[vocals]'`, as in the workflow. Never upload `.runtime/`.
