# Optional Backing Arrangement

In Generate Mezmur Videos, check **Rebuild backing music** to enable this mode.
It is off by default for manual and reusable workflows. Unchecked runs retain
the original soundtrack and do not install or invoke the backing arranger.

## What It Builds

The existing lyric search, acoustic verification and forced alignment finish on
the original recording first. A genuinely separated vocal stem is then combined
with newly synthesized piano, acoustic bass and warm-pad notes. The original
instrumental waveform is not mixed back in, except for unavoidable bleed that
may remain in the vocal extraction. This is an automatic new arrangement guided
by estimated harmony, not a newly recorded singer or a guaranteed new composition.

The CPU implementation uses librosa for onset/beat/chroma analysis and Viterbi
harmonic decoding, pretty_midi to write notes in absolute seconds, and FluidSynth
with the MIT-licensed FluidR3 GM SoundFont for rendering. There is no paid API,
GPU requirement, large generative model, or noncommercial-only model weight.
Original timing drives the new notes; vocals are not time-stretched or pitch-shifted.

## Quality Safeguards

- A failed separator or separator=none cannot silently use the full original mix
  as a supposed isolated vocal. Such a rebuild fails explicitly.
- The stem must match the original duration to within 150 ms. Small end padding
  is permitted; the start, sung words and internal vocal timeline are not moved.
- Silent, non-finite, unsupported or unreasonably short inputs are rejected.
- Beat/harmony evidence must be sufficient. Weak harmonic segments are omitted;
  the harmony score is an uncalibrated heuristic, not a word or musical accuracy percentage.
- Mixing retains the separated vocal waveform, with only one constant headroom
  gain when necessary. There is no voice EQ, added hiss, echo, room effect, pitch
  change, tempo change or dynamic vocal compression.
- The new backing is balanced below the vocals and faded at the track edges.
  Four-times oversampled peak checks leave at least 1 dB of PCM true-peak headroom.
- A 32-bit float WAV master is kept without lossy intermediate encodes. MP4 uses
  the existing 320 kbps AAC encoding once. Platform re-encoding is outside this tool.
- Failed technical checks stop the job rather than falsely label the old mix as rebuilt.

These checks cannot guarantee artifact-free separation, correct chord inference,
natural instrument performance, pleasant arranging, or studio-master quality.
Pentatonic, microtonal, live/rubato and unusual-meter songs are especially likely
to need review. High-confidence sections use inferred triads; less certain ones
use simpler root/fifth voicings. Automated review notes do not disable the existing
YouTube upload checkbox. For a first trial, select one song and uncheck uploading.

## Outputs

- video.mp4: the lyric video with the selected new soundtrack.
- rebuilt-audio.wav: lossless floating-point mix for listening and archiving.
- arrangement.mid: the new piano/bass/pad notes.
- arrangement.json: detected beat/harmony evidence, mixing gains, sample count,
  zero vocal offset/pitch change and peak measurements.
- arrangement-license.txt: installed FluidR3 copyright notice when available.

Use only songs and vocal recordings you have permission to rearrange and publish.
Changing the backing does not remove rights in a vocal recording, melody or lyrics,
and there is no guarantee of avoiding audio fingerprinting or Content ID matches.

The existing upload ledger still prevents posting the same Telegram song twice.
Selecting this checkbox does not replace or delete a previously uploaded video.

## Local Run

Install system packages fluidsynth and fluid-soundfont-gm, then:

```sh
pip install -e '.[render,arrange]'
python -m mezmur.song --input inputs/song-29 --output output/song-29 --rebuild-music
```

MEZMUR_SOUNDFONT can point to a different licensed SF2 file for local use. You
must supply the appropriate rights/credits for that file. The workflow uses
/usr/share/sounds/sf2/FluidR3_GM.sf2 without any extra secret or setup.

References:
- https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html
- https://librosa.org/doc/0.11.0/generated/librosa.sequence.viterbi.html
- https://www.fluidsynth.org/wiki/ExampleCommandLines/
- https://sources.debian.org/src/fluid-soundfont/3.1-6/debian/copyright/
