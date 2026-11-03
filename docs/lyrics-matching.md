# Lyrics Acceptance and Alignment

## Search Names

Telegram audio tags and filenames can contain channel branding, not the singer.
Name resolution drops missing-value placeholders such as `<unknown>`, `unknow`
and `Unknown Artist`, and the reported Yedestayaye Eleleta channel prefixes.
Known singer aliases embedded in filenames are separated from the song title.
The artist catalog includes Agegnehu Yideg and Aster Abebe in both scripts.

All title-query paths (Telegram, YouTube, web, direct lyrics sites and LRCLIB)
use cleaned metadata. Queries do not duplicate the singer inside the title and
use the English keyword **lyrics**, never an added Amharic keyword.
Missing song titles do not trigger artist-only YouTube/title searches.
Sung-phrase searches remain available when the title is missing or wrong.
Raw source fields remain available as diagnostic evidence.

## Acceptance

Automatic candidates must meet all of these checks:

- At least **40%** active vocal-frame coverage.
- At least **40%** lyric-line coverage.
- Acoustic gain at least **0.07** per vocal frame.
- Finite alignment scores and draft-text similarity at least 0.06.

Coverage and gain decisions use two decimal places, matching the displayed
policy. Thus raw coverage 0.399 and gain 0.069 are evaluated as 0.40 and 0.07;
0.394 coverage or 0.064 gain still fail. Raw values and decision values are both
recorded so this tolerance is visible, not an unexplained score discrepancy.
Acoustic gain is a relative log-likelihood measure, not a percentage of correct
words. Lower thresholds increase the risk of accepting an incorrect version.

A recognized website or saved library identity is diagnostic evidence, not an
override for inadequate audio evidence. Web/Telegram/library candidates are
never silently marked as trusted corrections. User-supplied corrections retain
their explicit manual override.

Reported song 35 (48% coverage, gain 0.07, 8/8 lines) and song 37 (40% coverage,
gain 0.09, 32/32 lines) can proceed without a song-ID exception. Candidates are
ranked by coverage, gain and phrase completeness.

## Timing and Review

Accepted candidates reach forced alignment/completion on the **original vocal
timeline**. Heard fragments anchor missing words; vocal activity and
stanza/repeat structure guide remaining placements. High estimated-word shares
produce review warnings, not a second rejection after lyrics were accepted.
This does not guarantee perfect word-level synchronization. Review estimated
sections, especially for live/studio versions and choir parts.

By default the video soundtrack remains the original recording without pitch,
tempo, noise, echo, reverb or loudness processing. Compatible AAC/MP3 streams
are copied; other formats receive AAC encoding only. The optional experimental
backing arranger runs only after the original lyric alignment and never changes
caption timestamps.

## Diagnostics

- Logs name the acceptance method and show raw coverage/gain to four decimals.
- `quality.json` includes identity evidence, raw and decision scores, policy
  thresholds, draft similarity and completion details.
- `metadata.json` includes `lyrics_verification` and review reasons.
- Debug timestamps refer to the original track.
