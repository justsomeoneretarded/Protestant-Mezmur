# Automatic YouTube Uploads

The normal workflow can generate videos and upload them to your own YouTube
channel. Uploading is opt-in and defaults to **private**. Google OAuth is required
for uploading; open-source transcription and rendering still run on GitHub Actions.
An API key or service account cannot replace channel-owner authorization.

## One-Time Setup

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project
   and enable **YouTube Data API v3**.
2. Configure the OAuth consent screen / Google Auth Platform audience. For initial
   testing, add your own Google account as a test user. Follow Google's consent
   and verification requirements for your intended use.
3. Create an OAuth client of type **Desktop app**, then download its JSON.
   Keep it outside Git or name it with the ignored pattern `client_secret*.json`.
4. On your own computer, inside this repository:

   ```bash
   python -m pip install -e '.[upload]'
   python scripts/youtube_authorize.py --client-secrets /path/to/client_secret.json
   ```

5. Complete Google's browser authorization, selecting the **correct YouTube
   channel or Brand Account**. The script uses a loopback callback bound to
   `127.0.0.1`, requests offline access with PKCE, and writes a private
   `.runtime/youtube-oauth.json` file. It prints your public `UC...` channel ID,
   never the refresh token or client secret.
6. In **Settings > Secrets and variables > Actions**, add:
   - **Secret** `YOUTUBE_OAUTH_JSON`: the full contents of that private file.
   - **Variable** `YOUTUBE_CHANNEL_ID`: the exact `UC...` ID printed by the script.

   With GitHub CLI, from the repository:

   ```bash
   gh secret set YOUTUBE_OAUTH_JSON < .runtime/youtube-oauth.json
   gh variable set YOUTUBE_CHANNEL_ID --body UC_REPLACE_WITH_YOUR_CHANNEL_ID
   ```

   Never send credentials in chat, commit them, or add them to artifacts. OAuth
   can manage this channel's videos and caption tracks. Revoke the grant in your
   Google account if you stop using the uploader.

A refresh token is not guaranteed to last forever. Testing-mode grants can
expire quickly; Google may revoke grants or require reauthorization. For unattended
use, configure your app's publishing status appropriately and complete any required
verification. A refresh failure requires running the authorization script again
with a new output path and updating the secret.

The scope is `youtube.force-ssl`, not upload-only, because
[caption uploads require it](https://developers.google.com/youtube/v3/docs/captions/insert#authorization).
No Google password is stored. API quotas, channel upload limits, account restrictions
and copyright enforcement still apply. Only process/publish audio you have rights to use.

## Generate and Upload

In **Actions > Generate Mezmur Videos > Run workflow**, select `main` and check
**Automatically upload finished videos to YouTube** (`upload_youtube`).

The workflow checks configuration before downloading/rendering songs. Once all
song jobs and progress finalization succeed, it calls the publisher. Existing
video/lyrics/thumbnail artifacts remain available if a later upload fails.
Unchecked means no YouTube authorization or upload occurs.

Repository **Variables** control optional settings:

| Variable | Default | Values |
| --- | --- | --- |
| `YOUTUBE_PRIVACY` | `private` | `private`, `unlisted`, `public` |
| `YOUTUBE_THUMBNAIL` | `a` | `a`, `b`, `c`, `d` |

Every new video initially uploads as **private**, with the recommended title,
description, tags, Music category, Amharic metadata/audio language and the existing
not-made-for-kids setting. One selected thumbnail and both timed caption tracks
are uploaded. Amharic and Amharic (Latin letters) use language `am` with distinct
track names. Existing SRT timing is retained; deprecated automatic caption
synchronization is not requested.

Only after those steps succeed may the uploader apply `unlisted` or `public`.
**Any review warning keeps the video private**, even if public was requested.
Approve those videos yourself in YouTube Studio. Posting/pinning comments, end
screens and thumbnail Test & Compare remain manual Studio tasks.

The description respects YouTube's **5000-byte UTF-8 limit**, not just 5000
characters. When necessary, the optional romanized block is omitted and long
lyrics are shortened while credits/hashtags are retained. Complete lyrics in
both scripts remain separate artifact files.

## Retry Without Rendering

Use **Actions > Publish Generated Videos > Run workflow**, on `main`:

- `run_id`: the number in the generation run's URL.
- `message_id`: optional; one Telegram message ID, or blank for all finished songs.
- `recover_video_id`: normally blank; only for the recovery described below.

The source run must have completed on this repository's `main` and still have
its artifacts (currently kept three days). Failed generation runs can publish
their successfully rendered song artifacts this way. Older artifacts lacking
`youtube.json`, `metadata.json` and Telegram source identity must be generated
again first.

A `youtube-upload-...` receipt artifact and YouTube links appear in the Actions
summary. Private videos are not viewable by everyone who has their URL.

## Duplicate Protection

A `youtube_uploads` map on **mezmur-state** records each destination YouTube
channel + numeric Telegram channel ID + message ID. State writes preserve the
song cursor and lyrics library. Both workflows serialize state writes; generation
finalizes progress *before* publishing, so upload checkpoints cannot invalidate
the generation snapshot.

- A durable reservation is saved **before** any video-upload request.
- The accepted video ID is saved before thumbnail/caption work.
- Completed songs are skipped on replays, even when rendered again.
  Existing videos are never automatically replaced or deleted.
- Partial uploads with known IDs resume thumbnail/caption work on the same video.
  Existing named caption tracks are reconciled rather than duplicated.
- Retry using the **original artifact's video bytes**, not a regenerated video
  whose timing files could mismatch the existing upload.
- The original visibility/review policy stays in force on retries. Changing retry
  settings cannot unexpectedly publish an unfinished private upload.

Exactly-once posting across two services cannot be guaranteed if a connection
dies just as YouTube accepts a video. If the response/checkpoint is lost, the
reservation stays `uploading` and another video insert is blocked.

Check YouTube Studio and the previous receipt/log for the accepted video ID.
Run **Publish Generated Videos** with the same original `run_id`, one `message_id`
and that `recover_video_id`. Ownership and a deterministic source tag are checked
before resuming. If Studio confirms **no video was created**, an administrator
may remove only that song's incomplete reservation from `state.json` on
`mezmur-state`, then retry the original artifact. Never clear reservations
blindly: that can create duplicates.

This repository is public. State and receipt artifacts expose video/source IDs
and links, including **unlisted** links. Use private videos for access control.
OAuth tokens and resumable session URLs are never stored there. Raw SDK errors
are not printed because they can contain sensitive URLs.

## Local Validation

```bash
python -m mezmur.upload --folder output/song-22 --dry-run
# With YOUTUBE_OAUTH_JSON, YOUTUBE_CHANNEL_ID, GITHUB_REPOSITORY and GH_TOKEN configured:
python -m mezmur.upload --folder output/song-22 --visibility private --thumbnail b
```

Dry run validates the kit without OAuth refresh, network requests, state writes
or uploads. Tests mock network calls and exercise the official SDK's request
construction. Live uploads require your own channel authorization and quota.

Official references: [video upload](https://developers.google.com/youtube/v3/docs/videos/insert),
[metadata limits](https://developers.google.com/youtube/v3/docs/videos#snippet.description),
[caption uploads](https://developers.google.com/youtube/v3/docs/captions/insert),
[Desktop app OAuth](https://developers.google.com/identity/protocols/oauth2/native-app).
