# Automatic YouTube Uploads

The normal workflow can generate videos and upload them to your own YouTube
channel. Uploading is opt-in: videos start **private** and are scheduled to become
**public in randomized order at five-hour intervals**. Google OAuth is required
for uploading; open-source transcription and rendering still run on GitHub Actions.
An API key or service account cannot replace channel-owner authorization.

## Browser-Only One-Time Setup

Use the Google account that manages the intended YouTube channel. Never paste
real credentials into chat, source files, repository variables or public artifacts.

1. Create or select **Rehoboth Mezmur** in [Google Cloud](https://console.cloud.google.com/projectcreate).
   If you already created it, reuse it; do not create another project.
2. Enable [YouTube Data API v3](https://console.cloud.google.com/apis/library/youtube.googleapis.com)
   in that project. The button becomes Manage when enabled.
3. Open [Google Auth Platform](https://console.cloud.google.com/auth/overview),
   click Get started, and enter the app name **Rehoboth Mezmur** and your support email.
4. Choose **External** audience, enter your contact email, accept the terms and Create.
   In Audience, add your own channel account under Test users if the app is in Testing.
5. Complete the app name, support email, homepage URL and privacy URL in Branding.
   [The exact values and free Pages setup are listed here](oauth-branding.md).
   For unattended use, set the app's publishing status to **In production** before
   issuing the long-term token; complete verification if Google requires it.
   Testing-mode grants can expire quickly and need reauthorization.
6. Under Clients, create an OAuth client of type **Web application**.
   Name it **Rehoboth GitHub Uploader**. Leave JavaScript origins empty and add
   this exact Authorized redirect URI (no trailing slash):

   ```text
   https://developers.google.com/oauthplayground
   ```

7. Create the client and keep its Client ID and Client secret private.
8. Open [Google OAuth Playground](https://developers.google.com/oauthplayground/).
   Open its gear/settings, keep Google endpoints and Server-side flow, choose
   **Access type: Offline**, **Force prompt: Consent Screen**, and check
   **Use your own OAuth credentials**. Enter your own Client ID and secret.
   Without this setting, Playground tokens are revoked after 24 hours.
9. In Step 1, enter this scope in the custom scopes field and click Authorize APIs:

   ```text
   https://www.googleapis.com/auth/youtube.force-ssl
   ```

10. Sign in, select the intended YouTube channel / Brand Account when prompted,
    and approve access to your own application.
11. In Playground Step 2, click **Exchange authorization code for tokens**.
    Use the **Refresh token**, not the Access token. Never share a credential
    link containing tokens or send the values in chat.
12. In this repository's Settings > Secrets and variables > Actions > Secrets,
    create **YOUTUBE_OAUTH_JSON** with this JSON, replacing all three values
    directly in GitHub's secret editor:

    ```json
    {
      "client_id": "YOUR_CLIENT_ID",
      "client_secret": "YOUR_CLIENT_SECRET",
      "refresh_token": "YOUR_REFRESH_TOKEN"
    }
    ```

13. Switch YouTube to the intended channel and open
    [Advanced settings](https://www.youtube.com/account_advanced) for its **UC... Channel ID**.
14. In the repository's Actions **Variables** tab, create **YOUTUBE_CHANNEL_ID**
    containing that ID, not the channel name or @handle.
15. Optionally set **YOUTUBE_THUMBNAIL** to a, b, c or d (default a).
    Keep your existing Telegram secrets unchanged.
16. Run Generate Mezmur Videos on main and check Automatically upload finished
    videos to YouTube. For an initial test, use one song.

## Local Authorization Alternative

The existing helper is optional, for a computer with Python. Unlike the browser
method above, it needs a **Desktop app** OAuth client JSON:

```bash
python -m pip install -e '.[upload]'
python scripts/youtube_authorize.py --client-secrets /path/to/client_secret.json
gh secret set YOUTUBE_OAUTH_JSON < .runtime/youtube-oauth.json
```

It binds its browser callback to 127.0.0.1, uses PKCE/offline access and creates
a mode-0600 .runtime/youtube-oauth.json file. Never commit it. The printed UC...
channel ID is public; the client secret and refresh token must remain private.
A refresh token is not guaranteed to last forever. Testing-mode grants can
expire quickly; Google may revoke grants or require reauthorization. For unattended
use, configure your app's publishing status appropriately and complete any required
verification. A refresh failure requires running the authorization script again
with a new output path and updating the secret.

The scope is `youtube.force-ssl` for channel ownership checks and managing the
uploaded video's publication status. No subtitle API calls are made.
No Google password is stored. API quotas, channel upload limits, account restrictions
and copyright enforcement still apply. Only process/publish audio you have rights to use.

## Generate and Upload

In **Actions > Generate Mezmur Videos > Run workflow**, select `main` and check
**Automatically upload finished videos to YouTube** (`upload_youtube`).

The workflow checks configuration before downloading/rendering songs. Once all
song jobs and progress finalization succeed, it calls the publisher. Existing
video/lyrics/thumbnail artifacts remain available if a later upload fails.
Unchecked means no YouTube authorization or upload occurs.

Only **YOUTUBE_THUMBNAIL** (a, b, c or d; default a) changes the upload layout.
The workflow always requests the five-hour-interval public queue. A previously configured
YOUTUBE_PRIVACY variable no longer changes workflow behavior.

Every new video initially uploads as **private**, with the recommended title,
description, tags, Music category, Amharic metadata/audio language and the existing
not-made-for-kids setting. It uploads one selected thumbnail and **never uploads
subtitle/caption tracks**. Burned-in video lyrics remain unchanged; SRT files
can remain in the download kit for your own use.

Before the first upload, the uploader shuffles newly selected songs once and saves
every UTC publishAt slot on mezmur-state with status queued. With an empty queue,
four selected videos get slots at **+5, +10, +15 and +20 hours** from queue planning.
The randomized order and timestamps are durable, including on a subset retry.

A new batch appends after this channel's last recorded future slot, keeping a
five-hour gap across batches. Other destination channels have independent queues.
Completed replays do not consume another slot. The queue only tracks this app's
saved records; manual changes in Studio or videos scheduled outside the app are
not automatically synchronized.

Each video and its thumbnail are uploaded and its reserved publishAt is applied
through YouTube's built-in scheduler. YouTube handles publication; no five-hour
runner wait or GitHub cron is needed. The Actions summary displays times in EAT
(Africa/Nairobi); machine-readable receipts keep UTC timestamps.
[YouTube requires private status for scheduling](https://developers.google.com/youtube/v3/docs/videos#status.publishAt),
so an unlisted video cannot simultaneously carry this native schedule.

Future deadlines remain fixed on retries: songs are not reshuffled and accepted
schedules are not moved. An expired, unfinished slot moves behind the remaining
future queue (or at least five hours ahead when none remains), never to a past
time that could publish immediately. This can leave gaps when an upload fails.
For an uploaded video, the app checks YouTube before moving a missed slot: if
YouTube already published it, recovery records that outcome without making the
video private again. Completed uploads are not changed.

Only verified, aligned lyric videos are eligible. Review notes remain in the kit
but no longer silently cancel automatic scheduling. Older unfinished reservations
retain their saved visibility/review policy. Posting/pinning comments, end screens
and thumbnail Test & Compare remain manual Studio tasks.

Title, description and tags are sanitized and shortened **before any upload**:

- Title: at most **100 characters**.
- Description: at most **5000 UTF-8 bytes**, without splitting Amharic characters.
- Tags: at most **500 total characters**, counting commas and implicit quotes.
  The duplicate-protection source tag is included in this same budget.

Newly generated descriptions reserve room for credits/hashtags and omit the
optional romanized block first when necessary. Complete lyrics in both scripts
remain separate artifact files. Upload-time corrections are listed in the receipt
as metadata_adjusted; overlong metadata does not cause an avoidable API rejection.

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
channel + numeric Telegram channel ID + message ID, including queued slots and
their source batch IDs. State writes preserve the
song cursor and lyrics library. Both workflows serialize state writes; generation
finalizes progress *before* publishing, so upload checkpoints cannot invalidate
the generation snapshot.

- A randomized batch plan with distinct five-hour slots is saved **before** any video-upload request.
- A queued record is changed to an uploading reservation before a new video insert.
- The accepted video ID is saved before thumbnail/scheduling work.
- Completed songs are skipped on replays, even when rendered again.
  Existing videos are never automatically replaced or deleted.
- Partial uploads with known IDs resume thumbnail/scheduling work on the same video.
  No caption tracks are inserted, updated or deleted, including during retries.
- Retry using the **original artifact's video bytes**, not a regenerated video
  whose timing files could mismatch the existing upload.
- The reserved visibility and saved schedule stay in force on retries. Changing
  retry inputs cannot trigger an immediate publication.

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
python -m mezmur.upload --folder output/song-22 --visibility public --thumbnail b
```

Dry run validates the kit without OAuth refresh, network requests, state writes
or uploads. Tests mock network calls and exercise the official SDK's request
construction. Live uploads require your own channel authorization and quota.

Official references: [video upload](https://developers.google.com/youtube/v3/docs/videos/insert),
[metadata limits](https://developers.google.com/youtube/v3/docs/videos#snippet.description),
[native scheduling](https://developers.google.com/youtube/v3/docs/videos#status.publishAt),
[OAuth Playground](https://developers.google.com/oauthplayground/),
[Desktop app OAuth](https://developers.google.com/identity/protocols/oauth2/native-app).
