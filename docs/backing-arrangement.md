# Optional AI Worship Band Backing

Check **AI worship band backing** in Generate Mezmur Videos. The saved input key is
still `rebuild_music`, off by default. Unchecked runs retain the original soundtrack.
The checked workflow now uses **ACE-Step 1.5 through acestep.cpp**, not the old
repeating MIDI/SoundFont guitar performer. No paid generation service, API key or
GPU is required. CPU generation is substantially slower than GPU generation.

## What Changed Musically

The old performer restarted a small guitar pattern at each two-beat harmony section.
A stronger section could therefore become continuous down/up strumming, while bass
mostly repeated roots and fifths. Adding instruments to that loop did not make it
a skilled composer.

The new model generates an audio-conditioned instrumental arrangement: expressive
bass guitar, piano/electric piano, restrained live drums, pads and occasional guitar
responses. The prompt asks for breathing verses, wider choruses, breaks and fills,
not constant strumming. These are generation instructions, not guaranteed isolated
instrument stems or a promise of Suno-equivalent quality.

[The Muser](https://github.com/noah-chelednik/the-muser) orchestrates multiple models;
it is not itself a replacement audio model. Its ACE-Step route is relevant, but its
full LLM/voice stack is unnecessary here. We use the smaller native ACE-Step runtime
directly, without adding a singer generator, voice conversion or a generic text-only
song underneath an existing performance.

## Processing Order

1. Separate the original singer and paired instrumental in float WAV.
2. Search, verify and complete lyrics on the original performance. The burned-in
   lyrics and Amharic-only `subtitle.srt` retain that same clock.
3. Analyze the actual instrumental for rhythm/harmony and the singer for stable
   pitch evidence. This analysis also supplies the BPM and candidate-ranking checks.
4. Pass **only the separated original instrumental** to ACE-Step's raw-source
   `cover-nofsq` mode. The real singer is never sent through the generation model.
   Source audio supplies the musical context; this is not an unrelated generated
   chord progression or guitar layered over the original full mix.
5. Generate two candidates for each long coherent phrase, normally at most
   72 seconds, rather than restarting at every chord. Source context strength is
   0.8 and 1.0. Both use the same instrumental-only style and deterministic song seed.
6. Compare source harmonic affinity and a conservative vocal-clash proxy. Reject
   invalid/silent outputs, material duration changes, very weak harmonic retention
   and identifiable large harmonic displacement. These metrics are **not**
   musical-taste scores, pitch accuracy percentages or a guarantee of a good arrangement.
7. Place selected phrases at their original absolute times. Four seconds of shared
   context help continuity, but only a short, original-beat-aligned blend (at most
   0.35 seconds) mixes two backings, avoiding four seconds of doubled drum grooves.
   Short final tails remain part of a longer phrase. A sub-ten-second
   input is padded only for the model and trimmed back afterward; no bars are appended.
8. Separate any synthesized singing/bleed out of the assembled new backing using
   the selected separator. This can also affect instrumental timbre and requires review.
   The real singer is not reprocessed here.
9. Mix the untouched original vocal stem with the new instrumental at a target
   7 dB lower whole-track RMS. Apply constant master headroom gain if necessary.
   Never shift/stretch/pitch-correct the singer. The original instrumental is not
   copied into the final mix; separation can still leave bleed inside the vocal stem.

The generator can still make poor musical choices, drift, ignore an instrument
request or produce separation artifacts. Start with **one song and Upload unticked**.
Listen to the result before scheduling it. A failed AI pass does **not** silently
substitute the original accompaniment or the previous MIDI loop.

## Actions Runtime and Reproducibility

The workflow builds the pinned MIT-licensed
[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) revision
`de858d8e1a94f0e1939cf9bdd8ff7d9631953077` with CPU/BLAS support.

It downloads only three public
[ACE-Step GGUF weights](https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF):
Q8 text encoder, Q8 2B turbo DiT, and BF16 VAE, approximately **3.7 GB combined**.
The model snapshot and each SHA-256 are pinned and checked. There is no LLM,
RVC voice conversion, external inference service, or model training. Files are
cached in `.ai-backing/`; the cache is ignored by Git.

CPU work can add significant time per song; generation has a two-hour budget,
with a 45-minute per-candidate subprocess timeout. These are failure limits, not
runtime or quality guarantees. AI batches run at most three songs in parallel.
The existing video job remains bounded by its overall Actions timeout.

To customize the sound, optionally set repository variable `MEZMUR_BACKING_STYLE`
(up to 1500 characters), for example a more acoustic worship-band description.
The source reference and instrumental-only lyrics token remain in force.
An empty variable uses the band prompt above. No secret or extra setup is needed.

For local use, install the project's `arrange` dependencies and system
`cmake`, `g++`, OpenBLAS and FFmpeg, then run:

```bash
python scripts/setup_ai_backing.py
```

The old MIDI helper remains callable by Python for regression comparisons;
the workflow checkbox no longer selects it.

## Technical Safeguards and Outputs

- Float stems and a float WAV master; no lossy intermediate soundtrack.
- Original vocal offset zero, pitch shift zero, tempo factor one.
- Constant headroom adjustment only on the singer, with four-times oversampled
  peak checks targeting at least 1 dB of PCM headroom.
- Exact original output sample count. AI padding, overlapping phrases and release
  tails never extend the video. MP4 can differ by ordinary frame/AAC quantization.
- Only the final master is encoded once to AAC for MP4; YouTube can re-encode again.
- `rebuilt-audio.wav`: final listening master.
- `arrangement.json`: model/engine revisions, prompt, analysis, zero vocal movement,
  duration lock, mix gains and peak checks.
- `arrangement-plan.json`: source rhythm/harmony and vocal pitch evidence, not a
  claim that every model-generated note follows that plan.
- `ai-backing/`: candidate requests, native logs, selection diagnostics, pinned
  model manifest and license. No MIDI is fabricated for an audio-generation model.

The original audio and lyric timestamps are not rewritten. Upload remains an
independent checkbox; when enabled, the existing caption upload, playlist placement
and four-hour native YouTube queue still apply. Historical workflow reruns retain
their old code; start a fresh run from `main`.

Use only recordings and songs you have permission to rearrange and publish.
Open model licensing does not grant rights in an existing singer's recording,
melody or lyrics. A replacement backing does not guarantee a different acoustic
fingerprint, Content ID avoidance or monetization.
