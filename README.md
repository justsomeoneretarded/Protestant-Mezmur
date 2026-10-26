# Protestant Mezmur

Turn the audio posts of a Telegram channel into finished, YouTube-ready Amharic
lyric videos — entirely on free GitHub Actions. For every song you download one
folder with the video, three thumbnails, titles, description, tags, captions and
an upload checklist, then upload it to YouTube yourself.

Source channel: [@amharicprotestantmezmurss](https://t.me/amharicprotestantmezmurss)
(change with the `TELEGRAM_CHANNEL` repository variable).

## What each song gets

| File | What it is |
| --- | --- |
| `video.mp4` | 1080p30 H.264, AAC 320k, loudness-mastered to YouTube's −14 LUFS |
| `thumbnail.jpg`, `-b`, `-c` | 1280×720 thumbnails with the singer cut out, Amharic + romanized title — three designs for YouTube *Test & Compare* |
| `title.txt` | Three title options ≤100 chars, Amharic + romanized (e.g. `ሲወድ ጌታ - አዲስዓለም አሰፋ \| Siwed Geta - Addisalem Assefa \| Amharic Mezmur Lyrics`) |
| `description.txt` | Description with chapters, Amharic lyrics, romanized lyrics, credits, hashtags |
| `tags.txt` | Song-specific + genre tags within YouTube's 500-character limit |
| `captions/am.srt`, `captions/am-Latn.srt` | Timed Amharic and romanized captions |
| `UPLOAD.md` | Step-by-step upload checklist (incl. where the end screen starts) and anything to double-check |
| `lyrics.txt`, `lyrics.romanized.txt`, `lyrics.ass` | Lyrics as text and the styled karaoke subtitles burned into the video |
| `metadata.json`, `quality.json` | Where lyrics/photos/footage came from, alignment statistics, timings |

### The video

* **Intro card** — singer photo, Amharic title, romanized title, singer name.
* **Karaoke lyrics** — each sung line appears just before it is sung; a gold fill
  sweeps across each word *as it is sung*, the romanized line fills in sync below,
  and the next line is previewed. Choruses get a small `አዝማች • CHORUS` tag.
* **Music-reactive background** — HD footage (or bundled scenes) with slow camera
  moves, a zoom pulse on every beat, brightness that follows the song's energy,
  rising light particles, a live spectrum, scene changes on the downbeat.
* **Countdown dots** before vocals return after an instrumental break, a singer
  badge and a progress bar, and a clean outro leaving room for YouTube end screens.

## How lyrics are made accurate

Speech recognition alone gets sung Amharic wrong too often, so the pipeline looks
for the **real lyrics** and only uses the speech model to *time* them.

1. **Collect candidates** from every free source:
   your `corrections/<id>.txt` → the Telegram post's caption, nearby posts and
   replies → public Telegram lyrics channels (`LYRICS_CHANNELS`) → [LRCLIB](https://lrclib.net)
   → web search across WikiMezmur, mezmuroch.com, legeta316.com, Mezmur Debter and any
   other page that carries the lyrics.
2. **Isolate the vocals** (MDX-Net Voc-FT via `audio-separator`; Mel-Band RoFormer optional, HTDemucs as fallback).
3. **Listen** with [Ethio-ASR Amharic](https://huggingface.co/badrex/Ethio-ASR-amharic)
   (Wav2Vec2-BERT CTC, CC BY 4.0), producing per-frame character probabilities.
4. **Verify** each candidate against what is actually sung (spelling variants such as
   ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ are treated as equal), and reject lyrics of a different song.
5. **Align**: a lyrics-constrained CTC search decides, for every vocal phrase, which
   words of the lyrics explain the audio — choruses repeated any number of times, lines
   split across breaths, ad-libs skipped — and takes word timings from the acoustic path.
6. If no lyrics can be verified, the automatic transcription is used and the song is
   flagged in `UPLOAD.md` for review.

**Fixing lyrics** never needs your computer: commit `corrections/<message id>.txt`
(see [corrections/README.md](corrections/README.md)) and the workflow re-makes that song,
re-timing your text automatically.

## Names, romanization, photos

* Titles like `የሌለው አቻ//Yelelew acha//ይስሐቅ ሰድቅ//Yishak Sedik`, ID3/MP4 tags, captions
  and file names are split into title/artist in both scripts. Missing Ge'ez spellings are
  learned from YouTube and lyrics-site titles; missing Latin spellings are romanized in the
  usual YouTube style (`ሲወድ ጌታ` → `Siwed Geta`). Fix spellings once in
  [data/artists.yaml](data/artists.yaml).
* Singer photos are gathered from `artists/` (your own, always preferred), the song's
  embedded cover art, Telegram photos near the post, the artist's official YouTube channel
  and videos, Wikimedia Commons and image search. Faces are detected (YuNet), the identity
  that appears across most sources is taken as the singer (SFace), and the best shot is cut
  out with U²-Net human segmentation for the thumbnails.

## One-time setup

1. Create a Telegram application at <https://my.telegram.org/apps> (API ID + hash).
2. On your computer (Python 3.11+): `pip install Telethon==1.42.0` then
   `python scripts/telegram_session.py`, and log in. Copy the printed session string.
3. **Settings → Secrets and variables → Actions → Secrets**:

   | Secret | Value |
   | --- | --- |
   | `TELEGRAM_API_ID` | numeric app ID |
   | `TELEGRAM_API_HASH` | app hash |
   | `TELEGRAM_SESSION` | the session string (treat like a password) |
   | `PEXELS_API_KEY` *(optional, recommended)* | free key from <https://www.pexels.com/api/> — real HD stock footage matched to each song |
   | `PIXABAY_API_KEY` *(optional)* | free key from <https://pixabay.com/api/docs/> |

   Without footage keys, videos use Wikimedia Commons footage (free licences, credited
   in the description) and the bundled scenes.
4. Optional **Variables**: `TELEGRAM_CHANNEL` (another channel), `LYRICS_CHANNELS`
   (comma-separated Telegram lyrics channels to search; a good default list is built in).

## Running

**Actions → Generate Mezmur Videos → Run workflow**

* `batch_size` — how many new songs (oldest first). Each song renders on its own
  runner in parallel.
* `message_ids` — re-make specific posts (e.g. `2,5`) without moving the bookmark.
* `separator` — vocal isolation model (MDX-Net Voc-FT is the fast default; Mel-Band RoFormer is cleaner but several times slower).

When it finishes, open the run and download **Artifacts → `mezmur-<id>-run…`**.
The run summary lists each song's lyric source and anything to check.

Progress is stored on the `mezmur-state` branch and only advances after every song in
the batch succeeded. Failed batches are retried next time; nothing is skipped.

## Costs and limits

Everything is free for a public repository: GitHub-hosted runners (4 vCPU / 16 GB),
open models (Ethio-ASR, MDX-Net, YuNet, SFace, U²-Net), keyless web sources and
free API keys. A 5-minute song takes roughly 20–40 minutes of runner time, mostly
vocal separation and rendering. Artifacts are kept 30 days.

Only publish songs you have permission to share. Lyrics and photos found online belong
to their owners; the description credits the singer and the lyric/footage sources.

## Development

```sh
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[render,test]'
pytest -q
python -m mezmur.song --input inputs/song-2 --output output/song-2   # after a fetch
```

FFmpeg with libass is required (the workflows download a static build).
