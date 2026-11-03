# Optional Hybrid Backing Arrangement

Check **Rebuild backing music** in Generate Mezmur Videos to enable this mode.
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
5. Generate new piano, acoustic bass, pad and light beat-following percussion.
   Tune the new instruments to the estimated recording tuning; never retune or
   time-stretch the singer.
6. Mix the new instrumental with the full-resolution vocal stem and render video.

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
  pitch classes and original beat timestamps.
- `arrangement-license.txt`: installed FluidR3 copyright notice.

The implementation uses librosa 0.11 (HPSS, CQT, pYIN and Viterbi), pretty_midi,
FluidSynth and the MIT-licensed FluidR3 GM SoundFont. The SF2 can be changed
locally using `MEZMUR_SOUNDFONT`; respect its license.

Use only recordings and songs you have permission to rearrange and publish.
Changing the backing does not remove rights in the vocal recording, melody or
lyrics, or guarantee avoiding acoustic fingerprinting / Content ID.
The upload ledger still prevents duplicate posting of the same Telegram song.
