# Lyrics Acceptance and Alignment

A found lyric is not rejected solely because its acoustic gain is below 0.18.
That score is a relative log-likelihood measure, not a percentage of correctly
transcribed words. Speech models can score sung Amharic weakly even when a lyric
page names the correct song.

## Acceptance Paths

1. A user-supplied correction remains an explicit manual override.
2. A strong acoustic match uses the existing 70% vocal/line coverage and 0.18
   gain thresholds.
3. A **source-supported** match can proceed with a lower positive gain when all
   of the following are true:

- A recognized HTTPS lyrics source identifies the title and artist, using the
  existing spelling-matching helpers and structured artist/song URLs where available.
- At least 70% of active vocal frames are covered by aligned lyric sections.
- At least 70% of lyric lines are matched, even if some lines are only partial.
- Candidate/draft text similarity is at least 0.15.
- At least half of matched phrase occurrences are complete.
- Acoustic gain is positive and all measured scores are finite.

A matching title by itself does not bypass the checks. Unknown or spoofed
domains, a different artist, poor coverage and zero/negative acoustic evidence
are not automatically accepted. Strong acoustic candidates are preferred to
source-supported alternatives. Web/library candidates are never silently marked
as trusted corrections.

The reported song-29 values (coverage 0.73, 14/20 lines, gain 0.11, complete
phrases 0.57 and draft similarity 0.217) can use this path when the original
Mezmuroch page identifies the artist and song. This is a generic rule, not a
message-ID exception or a claim that a website guarantees lyric correctness.

## Timing

Accepted candidates reach the existing forced-alignment/completion pass on the
**original recording's vocal stem**. Heard fragments anchor missing words;
vocal activity and stanza/repeat structure guide the remaining placements.
The sound used in the final video is still the original track, never the stem.

Source-supported matches continue when more than 35% of word timings must be
estimated, with explicit review notes instead of another automatic rejection.
Unidentified candidates retain the existing estimated-timing guard. Review
estimated sections: this does not guarantee perfect word-level synchronization
or distinguish every different live/studio version of a song.

## Diagnostics

- Candidate logs name the acceptance path: acoustic, source-supported, correction
  or rejected.
- quality.json records identity evidence, coverage, draft agreement, completion
  and acoustic scores without pretending that gain is a percentage.
- metadata.json includes lyrics_verification and review reasons.
- Debug alignment timestamps refer to the original track.

No pitch/tempo changes, added noise, echo, reverb, formant processing or loudness
normalization run in the rendering pipeline. The original AAC/MP3 audio packets
are copied when compatible with MP4; other codecs receive AAC encoding only.
