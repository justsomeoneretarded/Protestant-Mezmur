# Protestant Mezmur

Turns songs from a Telegram channel into **YouTube-ready Amharic lyric videos**, entirely
inside GitHub Actions, with free tools only. You download the finished folder and upload it
yourself.

For every song you get:

| File | What it is |
| --- | --- |
| `video.mp4` | 1080p lyric video: karaoke Amharic lyrics timed word-by-word to the real singing, romanized line underneath, next line preview, one calm photograph for the whole song (waterfall, mountains, sea, sky, stars, skyline…) with nothing drawn or written on it, never reused for another video, a smooth music spectrum, title card and a clean end-screen area |
| `thumbnail.jpg`, `thumbnail-b.jpg`, `thumbnail-c.jpg` | 1280×720 thumbnails with the singer's photo (cut out), the song name in Amharic and Latin letters and the singer's name. Three designs for YouTube's *Test & Compare* |
| `title.txt` | Title options, first one in the common search style: `Addisalem Assefa siwed geta ሲወድ ጌታ አዲስዓለም አሰፋ lyrics` |
| `description.txt` | Ready to paste: both scripts, full Amharic lyrics, lyrics in Latin letters, credits, hashtags |
| `tags.txt` | Song-specific + general tags, under YouTube's 500-character limit |
| `captions/am.srt`, `captions/am-Latn.srt` | Caption files to upload in YouTube Studio: Amharic, and Amharic in Latin letters (name that track "Amharic (Latin letters)") |
| `pinned_comment.txt` | A comment to post and pin |

## How the lyrics are made accurate

1. **Find the real lyrics.** For each song the pipeline searches: your `corrections/<message id>.txt`
   → its own lyrics library (every lyric verified before is saved on the `mezmur-state` branch) →
   posts around the song in the source channel → Amharic lyrics channels on Telegram (plus every chat
   your account has joined, plus any channel that supplied verified lyrics before) → LRCLIB → mezmur
   lyrics sites (mezmuroch.com, legeta316.com, mezmurdebter) → up to 15 YouTube lyric videos, reading
   both their descriptions and their caption tracks → web search (private SearXNG + a second engine).
   Titles are searched in Amharic and in several Latin spellings, and a few clearly sung phrases from
   the song (usually its chorus) are searched too, so lyrics are found even when the title is wrong.
   Each song's run page lists what every source returned.
2. **Listen to the song.** The vocals are isolated (UVR-MDX-NET) and the Amharic acoustic model
   Ethio-ASR turns them into per-moment character probabilities.
3. **Check every candidate against the audio.** Found lyrics are aligned to the whole song. A
   candidate is used only if it explains most of the singing; lyrics for a different song with
   the same title fail this check.
4. **Align word by word.** The aligner follows the song through the lyric lines: choruses can
   repeat, lines can be skipped or start halfway, instrumentals and ad-libs are allowed, but
   singing that the lyrics do not explain costs extra, so lines stay on screen for exactly as
   long as they are sung.
5. **No verified lyrics?** You still get a video from an automatic transcription, clearly marked
   in `UPLOAD.md` as needing review.

### Fixing lyrics (from your phone, no computer needed)

Create or edit `corrections/<message id>.txt` in GitHub's web editor and commit it to `main`:

```text
---
title_am: ሲወድ ጌታ
title_latin: Siwed Geta
artist_am: አዲስዓለም አሰፋ
artist_latin: Addisalem Assefa
---
ሲወድ ጌታ እስከ መስቀል ነው
...

አዝ፦
ሀሌሉያ ክብር ለስሙ
...
```

The header is optional. Write each stanza once and mark the chorus with `አዝ፦` (or `Chorus:`);
repeats are found automatically. The commit starts the workflow for that song by itself. Your
text is treated as correct, the timing is redone, and the video, subtitles, description and
thumbnails are rebuilt.

### Typing lyrics straight into the workflow

If you have a song's lyrics at hand, you can skip the file: in **Run workflow**, enter that song's
**message_ids** (one ID), untick **No lyrics input**, and type the lyrics in the **Lyrics** box.
The box is one line, so put `|` between lines and `||` between verses, for example:

`ዛሬም ይሰራል /3 | ዛሬም እየሱስ ይሰራል || አዝ፦ ኤልሻዳይ ነው ዛሬም ይሰራል | ኤልሻዳይ ነው ድንቅን ያደርጋል`

Typed lyrics are trusted like a correction file and saved to the lyrics library.

### Your own singer photos

Put a good photo in `artists/` named after the singer, for example
`artists/addisalem-assefa.jpg`. It is then always used for that singer's thumbnails. Without
one, the pipeline picks the best photo from the song's cover art, the Telegram post, the singer's
YouTube channel and the web, checking that the faces agree.

## One-time setup

1. Create a Telegram application at <https://my.telegram.org/apps> (API ID and API hash) with
   an account that can read the source channel.
2. On your computer (Python 3.11+): `pip install Telethon==1.42.0`, then
   `python scripts/telegram_session.py`. Log in when asked and copy the printed session string.
   It grants access to your Telegram account, so keep it secret.
3. **Settings → Secrets and variables → Actions → Secrets:**

   | Secret | Required | Value |
   | --- | --- | --- |
   | `TELEGRAM_API_ID` | yes | numeric app ID |
   | `TELEGRAM_API_HASH` | yes | app hash |
   | `TELEGRAM_SESSION` | yes | the session string |
   | `PEXELS_API_KEY` | recommended | free key from <https://www.pexels.com/api/> — real HD nature/sky footage chosen from the lyrics' themes |
   | `TELEGRAM_BOT_TOKEN` | optional | your bot's token from @BotFather — the bot posts "✅ Your video is ready!" (title + run link) when each song finishes. Make the bot an admin of the channel |
   | `PIXABAY_API_KEY` | optional | free key from <https://pixabay.com/api/docs/> — more footage |

   Without footage keys the backgrounds come from free sources that need no key, in this order:
   NASA's public-domain video library (sky, Earth, stars), Wikimedia Commons, and Mixkit. Clips
   with burned-in text are rejected automatically, and footage search stops after 5 minutes. If
   every site fails, the video still gets the bundled scenery plus generated light backgrounds.
4. Optional **Variables:** `TELEGRAM_CHANNEL` (default `@amharicprotestantmezmurss`) and
   `TELEGRAM_NOTIFY_CHAT` (where the bot posts; default `@MyApplication_aman`) and
   `LYRICS_CHANNELS` (comma-separated Telegram lyrics channels to search; a good default list is
   built in). Joining extra lyrics channels with your Telegram account also helps, because the
   pipeline searches every chat the account is in.

## Running it

**Actions → Generate Mezmur Videos → Run workflow**

- `batch_size`: how many new songs (oldest unprocessed first). Each song renders on its own free
  runner at the same time, so 5 songs take about as long as 1 (≈ 25–35 minutes).
- `message_ids`: remake specific songs (e.g. `2,3`) without moving your place in the channel.
- `separator`: vocal-isolation model (default is the fast, accurate UVR-MDX-NET).

When it finishes, open the run and download each `mezmur-<message id>-run…` artifact
(kept 3 days — download it before then). It holds only what you upload. Anything to check before publishing (for example
unverified lyrics) is listed per song on the run's summary page. If a song fails, a `log-…`
artifact with its log is added instead.

### Progress

Your place in the channel is stored on the `mezmur-state` branch and only moves forward after
every song in the batch succeeded. A failed batch is retried next time, so no song is skipped.
Replays and corrections never move it. Footage clips already used are remembered so videos
don't repeat backgrounds.

## Costs and limits

Everything is free: the repository is public, so GitHub Actions minutes are free on 4-CPU
runners; all models (Ethio-ASR, UVR-MDX-NET, YuNet/SFace faces, U²-Net cutout, PP-OCR text
detection) are open and run on the runner; Telegram, LRCLIB, Wikimedia, Pexels and Pixabay are free.
Model files are cached between runs (Actions cache, 10 GB).

Public repositories' artifacts can be downloaded by any signed-in GitHub user. Make sure you
have permission to republish the songs; Content ID claims and strikes apply to lyric videos too.

## Development

```sh
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[render,test]'
pytest -q
python -m mezmur.song --input inputs/song-2 --output output/song-2   # one fetched song
```

Model credits: [Ethio-ASR Amharic](https://huggingface.co/badrex/Ethio-ASR-amharic) (CC BY 4.0),
[audio-separator / UVR](https://github.com/nomadkaraoke/python-audio-separator),
[OpenCV model zoo](https://github.com/opencv/opencv_zoo), [rembg U²-Net](https://github.com/danielgatis/rembg).
