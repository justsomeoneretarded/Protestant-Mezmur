# Subtle Audio Augmentation

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
```

## GitHub Actions Integration

In **Actions > Generate Mezmur Videos > Run workflow**, check **Subtle audio
augmentation** (`augment_audio`) to enable it. It is off by default: ordinary
runs retain the original audio-processing path. Reusable workflows can pass
`augment_audio: true` as well.

The runner installs the native library automatically when enabled. Each song
uses its Telegram message ID as the seed, keeping replays reproducible. The
source is decoded and pre-mastered through FFmpeg so Telegram's M4A, Ogg, MP3,
and other supported formats are accepted without decoder clipping. The
augmented track then goes through final mastering for the video; vocal
separation and word alignment use that same augmented source, not old timings.
Original source tags and cover art remain available for the upload kit.

Enabled runs include `augmentation.json` in the song's artifact with the seed,
parameters and planned timing anchors. This is diagnostic metadata, not a file
to upload to YouTube. Local rendering supports the same option:

```bash
python -m mezmur.song --input inputs/song-21 --output output/song-21 --augment-audio
```

The integration is not a copyright, Content ID, or privacy workaround. Use audio
you have permission to process and publish.

## Behavior

- Cosine-interpolated random controls generate a slowly varying time map.
  The planned local speed differs by at most approximately 1.5% by default,
  subject to sample rounding. The native engine may deviate around transients.
- A small independent spectral-envelope shift changes timbre without an
  intentional musical-pitch shift. This is not simply resampling or pitch shifting.
- Identical settings and seed produce reproducible parameters. Exact output
  samples may differ across native library versions and platforms.
- The default preserves the original frame count. Internal word, beat, and event
  positions can still change. Use the planned mapping for approximate labels and
  validate any sample-accurate or word-level ground truth after transformation.
- Mono, stereo, and up to 32 channels use the same time map; synchronized stereo
  processing preserves mono compatibility. Input is finite floating-point PCM
  normalized to [-1, 1], at 8000..192000 Hz. Integer PCM is rejected explicitly.
- Extremely short hits receive temporary analysis context, removed from the
  returned result. There may be little meaningful augmentation in such clips.
- Peaks above 0.999 are attenuated uniformly, never hard-clipped. Quiet audio is
  not boosted. If both strengths are zero, the array API returns an exact float32
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

Rubber Band is GPL-licensed, with commercial licensing also available. Review its
licensing terms before redistributing software that integrates the library.

## Test

```bash
python -m pip install -e '.[test,augmentation]'
python -m pytest -q tests/test_augmentation.py
```

Tests exercise the actual native engine, not just mocks: seed reproducibility,
pitch stability, formant-envelope change, stereo coherence, percussion timing,
silence, short hits, invalid inputs, lossless export, and original-file protection.
They do not certify perceptual quality or anonymity on real-world datasets.
