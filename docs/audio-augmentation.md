# Audio Augmentation

A standalone Python utility for speech, music, percussion, and mixed recordings.
Rubber Band's established R3 engine handles the signal processing. Use it as a
standalone utility or enable it explicitly for a mezmur video workflow run.

## Install

Tested on Python 3.12 and Linux with Rubber Band 3.3.0. Requires Python 3.11+
and Rubber Band 3.0+ with its native shared library.

```bash
# Debian/Ubuntu
sudo apt-get install librubberband2
python -m pip install -e '.[augmentation]'

# macOS: install the native library with `brew install rubberband` instead.
```

If the shared library is outside the system search path, set
`RUBBERBAND_LIBRARY` to its absolute path. A command-line executable alone is
not sufficient: this script calls the library, not a shell command.

## Use

```python
import soundfile as sf
from mezmur.augmentation import augment_audio, augment_file

# File API: writes a new lossless file and an augmentation JSON sidecar.
augment_file("input.wav", "augmented.wav", seed=42)

# Array API: SoundFile layout is (frames, channels), not (channels, frames).
samples, sample_rate = sf.read("input.wav", dtype="float32", always_2d=True)
result = augment_audio(
    samples,
    sample_rate,
    seed=42,
    max_speed_change=0.015,
    max_formant_semitones=0.3,
    control_seconds=2.0,
    preserve_duration=True,
)
sf.write("another.wav", result.audio, sample_rate, subtype="PCM_24")
new_event_times = result.map_times([0.1, 0.4])
```

```bash
python -m mezmur.augmentation input.wav augmented.wav --seed 42
# Or use the installed entry point:
mezmur-augment input.wav augmented.wav --seed 42
# A stronger, coherent change to the whole mix:
mezmur-augment input.wav musical.wav --seed 42 --pitch 2 --tempo 1.04 --noise-snr 50 --allow-duration-change
# Vinyl-style texture and small-room space:
mezmur-augment input.wav textured.wav --noise-snr 50 --noise-profile vinyl --space-effect room --seed 42
```

## GitHub Actions Integration

In **Actions > Generate Mezmur Videos > Run workflow**, check **Audio
augmentation** (`augment_audio`) to enable it. It is off by default: ordinary
runs retain the original audio-processing path. Reusable workflows can pass
`augment_audio: true` as well.

Choose `augmentation_style`:

- **musical** (default when augmentation is enabled): the entire mix moves by
  +2 or -2 semitones and its nominal tempo changes by about 4%. Every instrument
  and voice receives the same transposition; no random notes or replacement beats
  are introduced. Smaller local timing/formant variations and 50 dB SNR noise are
  also applied. Lyrics are verified on the original recording before any effects;
  their timestamps are then mapped to the changed duration.
- **subtle**: keeps the original key and duration, applies the original small
  timing/formant variations, and adds the same quiet 50 dB SNR noise.

`noise_profile` selects **hiss**, **vinyl** (synthetic sparse crackle over hiss),
**ambient** (soft synthetic room tone, not a field recording), or **off**. All
enabled overlays are mixed at 50 dB SNR in the video pipeline. The standalone
API/CLI accepts 20..80 dB when a different training strength is wanted.

`space_effect` selects **room** (default; a small, damped room with a quiet wet
signal), **echo** (80 ms slapback with low feedback), or **none**. Pedalboard's
existing reverb/delay engines process stereo pairs. Effects keep the planned
frame count and blend back to dry at the end; they do not append a long tail.
Reverb and echo deliberately add decay into pauses, while the noise gate keeps
otherwise silent samples clean. Captions use the saved time map, including local
speed variations even when the total duration stays unchanged.

These are changes to the original recording, not promises of improved sound or
anonymity. The standalone API retains its conservative defaults: no intentional
key/nominal-tempo change and no noise unless explicitly requested.

The runner installs the native library automatically when enabled. Each song
uses its Telegram message ID as the seed, keeping replays reproducible. The
source is decoded and pre-mastered through FFmpeg so Telegram's M4A, Ogg, MP3,
and other supported formats are accepted without decoder clipping. The
original recording is used for vocal separation, recognition, lyric search,
verification, word alignment and synced-lyrics cross-checks. The same strict
acceptance thresholds apply whether augmentation is enabled or not. Songs that
fail verification are skipped before effects run.

Only verified songs are augmented and finally mastered for the video. Every
word's start and end are remapped through the saved input/output frame anchors;
SRT, ASS karaoke, on-screen lyrics and upload-kit chapters use the remapped
timeline. This is not just a global tempo adjustment: subtle mode can move words
within a track without changing its total length. Anchors describe planned DSP
timing, so small transient/word-boundary deviations still need listening checks.
Inconsistent or missing maps fail explicitly instead of silently mis-syncing
lyrics. Original source tags and cover art remain available for the upload kit.

Enabled runs include `augmentation.json` in the song's artifact with the seed,
parameters, planned timing anchors and original/render duration provenance.
`quality.json` and debug acoustic/alignment diagnostics refer to the original
recording; `words.json`, subtitles and the video refer to the final audio.
This is diagnostic metadata, not a file to upload to YouTube. Local rendering
supports the same option:

```bash
python -m mezmur.song --input inputs/song-21 --output output/song-21 --augment-audio
# Keep the earlier, quieter transformation instead:
python -m mezmur.song --input inputs/song-21 --output output/song-21 --augment-audio --augmentation-style subtle
```

The integration is not a copyright, Content ID, or privacy workaround. Use audio
you have permission to process and publish.

## Behavior

- Cosine-interpolated random controls generate a slowly varying time map.
  The planned local speed differs by at most approximately 1.5% by default,
  subject to sample rounding. The native engine may deviate around transients.
- A small independent spectral-envelope shift changes timbre without an
  intentional musical-pitch shift by default. Explicit transposition moves all
  pitched parts together and compensates the formant envelope independently.
- Optional band-limited noise is added at a measured signal-to-noise ratio.
  Higher dB is quieter. Noise is the same across channels to preserve stereo
  coherence, and exact digital silence remains silent. Quiet noise may be removed
  by final AAC compression; it does not reliably hide a song's identity.
- Identical settings and seed produce reproducible parameters. Exact output
  samples may differ across native library versions and platforms.
- The default array API preserves the original frame count. Internal word, beat, and event
  positions can still change. Use the planned mapping for approximate labels and
  validate any sample-accurate or word-level ground truth after transformation.
- Mono, stereo, and up to 32 channels use the same time map; synchronized stereo
  processing preserves mono compatibility. Input is finite floating-point PCM
  normalized to [-1, 1], at 8000..192000 Hz. Integer PCM is rejected explicitly.
- All inputs receive 250 ms of temporary trailing analysis context, removed from
  the returned result. This absorbs small native synthesis-boundary shortfalls
  without inserting silence over missing song samples. Extremely short hits also
  receive leading context. There may be little meaningful augmentation in those
  clips. Non-finite samples and substantial truncation still fail explicitly.
- Peaks above 0.999 are attenuated uniformly, never hard-clipped. Quiet audio is
  not boosted. If all effects are disabled, the array API returns an exact float32
  copy without native processing or gain changes.
- Input decoding depends on libsndfile's codecs. Output is 24-bit WAV or FLAC.
  The file API refuses existing output paths and never changes the original.
- Processing is offline and in memory. For long recordings, budget RAM accordingly.

## Privacy and Quality Limits

This is a training augmentation tool, **not anonymization, encryption, or a
guarantee of privacy**. It does not guarantee removal of speaker identity,
audio fingerprints, spoken personal information, or recognizability. No subtle
transformation can promise perfect intelligibility for every recording.

Formants primarily describe resonances in voiced speech; on music, percussion,
noise, and mixtures the same operation is a spectral-envelope/timbre change, not
a universal privacy mechanism. Perform listening evaluations and task-specific
ASR, speaker-identification, and label-integrity checks on your dataset. Use
consented data and an appropriate threat model when privacy is a requirement.

The output file and JSON do not copy source tags or source filenames. The JSON
contains the seed and time-map parameters for reproducibility; keep that sidecar
internal if your research requires the augmentation parameters not to be shared.

Rubber Band is GPL-licensed, with commercial licensing also available. Room and
echo effects use [Pedalboard](https://github.com/spotify/pedalboard), which is
GPL-licensed as well. Review their terms before redistributing integrated software.

## Test

```bash
python -m pip install -e '.[test,augmentation]'
python -m pytest -q tests/test_augmentation.py
```

Tests exercise the actual native engine, not just mocks: seed reproducibility,
pitch stability, coherent transposition, tempo changes, measured noise SNR,
noise profiles, room/echo effects, formant-envelope change, stereo coherence, percussion timing,
silence, short hits, invalid inputs, lossless export, and original-file protection.
They do not certify perceptual quality or anonymity on real-world datasets.
