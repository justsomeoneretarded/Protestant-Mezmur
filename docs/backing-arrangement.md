# Optional Guitar-Led Acoustic Arrangement

Check **Guitar-led acoustic backing** in Generate Mezmur Videos to enable this mode.
It remains off by default. Unchecked runs use the original soundtrack and do
not invoke separation of a paired instrumental or the backing arranger.

## Processing Order

1. Isolate vocals and retain the actual paired instrumental. Multi-stem Demucs
   outputs are combined without vocals when no paired instrumental is provided.
2. Search, verify and complete lyrics against the original vocal timeline.
   Existing word/line gap-filling and subtitle timestamps are unchanged.
3. Analyze the separated instrumental for beats, tuned chroma and bass evidence.
4. Extract confident, stable vocal pitches with pYIN. Use these as conservative
   constraints on accompaniment, not as a replacement for the instrumental map.
5. Generate guitar-led acoustic accompaniment and bass on the original beat grid.
   Quiet passages use fingerpicking; supported stronger passages use strumming.
   Lyric verse/chorus cues, source dynamics and vocal activity guide the performance.
   Repeated sections reuse their motifs instead of random pattern changes.
   Guitar voicings use the standard six-string range and bounded fret spans.
   Piano accents are reserved for confirmed vocal gaps; pads and light percussion
   enter only in supported energetic sections. Guitar and bass remain the foundation.
   Tune new instruments to the estimated recording tuning, never the singer.
6. Mix the new instrumental with the full-resolution vocal stem and render video.
   Every MIDI note is bounded by the source endpoint. Synthesizer release tails
   are trimmed, or short backing renders end-padded, to the original sample count.
   Vocals are never padded to match an extended instrumental.

The original instrumental is used for analysis only, never mixed into the final
soundtrack. Separation can still leave instrumental bleed in the vocal stem.
No paid API, GPU or new model weights are needed.

## Harmony and Uncertainty

Instrumental CQT chroma and bass evidence propose major, minor, seventh,
suspended and open-fifth harmonies. Library Viterbi decoding smooths the
progression. Confident sustained vocal notes discourage semitone/tritone
conflicts; low-confidence frames and passing pitches do not automatically
override the instrumental.

Full chords are used only where supported and compatible with the observed
melody. Weak regions use supported open fifths or single-note accompaniment;
regions without usable evidence stay unpitched. There is no longer a global
55% major/minor-template cutoff that rejects an otherwise usable arrangement.
This is a different uncertainty policy, not a claim of improved calibrated
accuracy or guaranteed pleasant music.

The arrangement is adaptive, not an unchanged guitar loop. A source with repeated
harmony can retain those repetitions; variation is driven by sections and energy,
not arbitrary chord changes. Guitar is synthesized with the existing FluidR3
steel-string patch, not a newly recorded human guitarist.

The system does not infer every musical intention, chord inversion or meter.
pYIN can struggle with choirs, bleed and ornamented/microtonal singing. Pitch
checks are conservative heuristics; intentional dissonance may be simplified.
The report records stable vocal-frame counts and whether vocal constraints had
usable evidence. Listen before publishing.

## Technical Safeguards

- Checked mode retains both stems and refuses a missing instrumental; the full
  original mix cannot silently masquerade as an isolated instrumental.
- The selected separator is honored without automatic model-quality fallbacks.
  RoFormer retains its default overlap in rebuild mode.
- Floating-point stem export is used. Unsupported/low-precision stems,
  invalid/silent audio and mismatched stem timelines fail explicitly.
- Original vocal pitch and timing are retained. Mixing uses constant headroom
  gain only; no vocal EQ, added hiss, reverb, echo or dynamic compression.
- The new backing stays below the singer and fades at track edges.
- Four-times oversampled peak checks target at least 1 dB of PCM headroom.
- The master is float WAV with no lossy intermediate encodes. MP4 encodes AAC
  once; YouTube may re-encode afterward.
- Ordered, finite MIDI sections and notes are required. Invalid/out-of-range
  timing data fail before synthesis instead of silently extending the song.
- The lossless master is locked to the original sample count. MP4 frame/AAC
  packet quantization can differ by a small fraction of a second, not added
  musical bars or a synthesizer tail. Both paths have native duration tests.
- Original recording integrity is checked. Captions remain on the original
  timeline even when the new backing has different instrumental patterns.

Lossless export cannot fix separation artifacts, wrong musical inference or
unnatural instrument performance. These are technical safeguards, not a
guarantee of studio quality. Review warnings do not disable automatic YouTube
uploads. For the first trial, select one song and uncheck uploading.

## Outputs

- `video.mp4`: lyric video with the selected new soundtrack.
- `rebuilt-audio.wav`: float WAV master for listening.
- `arrangement.mid`: generated instrumental notes and tuning.
- `arrangement.json`: rhythm/harmony/melody evidence, mix gains, zero vocal
  offset/pitch shift, sample count and peak checks.
- `arrangement-plan.json`: per-section voicings, confidence mode, observed vocal
  pitch classes, guitar string/fret voicings, playing styles, dynamics and original beat timestamps.
- `arrangement-license.txt`: installed FluidR3 copyright notice.

The implementation uses librosa 0.11 (HPSS, CQT, pYIN and Viterbi), pretty_midi,
FluidSynth and the MIT-licensed FluidR3 GM SoundFont. The SF2 can be changed
locally using `MEZMUR_SOUNDFONT`; respect its license.

Use only recordings and songs you have permission to rearrange and publish.
Changing the backing does not remove rights in the vocal recording, melody or
lyrics, or guarantee avoiding acoustic fingerprinting / Content ID.
The upload ledger still prevents duplicate posting of the same Telegram song.

The workflow input key remains `rebuild_music` for saved/reusable workflow compatibility,
but its visible checkbox now selects this acoustic arranger, not the previous
piano-led pattern generator. Start a fresh workflow on `main`; historical reruns
use their original code. The independent `upload_youtube` checkbox remains checked
by default and can be unchecked for test generations.
