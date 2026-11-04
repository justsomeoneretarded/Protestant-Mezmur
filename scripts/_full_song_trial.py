"""Prepare one complete song and render two isolated experimental cover strengths; never publish."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import soundfile as sf

from mezmur import arrangement
from mezmur.audio import probe
from mezmur.common import duration, write_json
from mezmur.separate import instrumental_stem, separate
from mezmur.subtitles import write_srt

SONG_ID = 40


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare():
    import srt

    out, work = Path("prepared"), Path(".runtime/full-prepare")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    source = next(Path(f"inputs/song-{SONG_ID}").glob("audio.*"))
    saved_source = out / ("source" + source.suffix)
    shutil.copy2(source, saved_source)
    vocals = separate(saved_source, work, require_vocals=True)
    reference = instrumental_stem(vocals, out / "instrumental-reference.wav")
    saved_voice = out / "vocals.wav"
    shutil.copy2(vocals, saved_voice)
    seconds = duration(saved_source)
    sr = sf.info(saved_voice).samplerate
    expected = round(seconds * sr)
    assert abs(sf.info(saved_voice).frames - expected) <= sr * 0.15
    old = Path("old-kit")
    old_video = next(old.rglob("video.mp4"))
    shutil.copy2(old_video, out / "original-lyric-video.mp4")
    old_srt = next(old.rglob("captions/am.srt"))
    kit_path = next(old.rglob("youtube.json"))
    kit = json.loads(kit_path.read_text(encoding="utf-8"))
    chapters = kit.get("chapters", [])
    cues = []
    for caption in srt.parse(old_srt.read_text(encoding="utf-8")):
        start = caption.start.total_seconds()
        if start >= seconds:
            continue
        # The old sung-line SRT added 0.3s; recover the original cue for display-window export.
        end = min(seconds, max(start + 0.01, caption.end.total_seconds() - 0.3))
        chapter = next((item for item in reversed(chapters) if item["start"] <= start), {})
        cues.append({"start": start, "end": end, "text": caption.content,
                     "chorus": chapter.get("label", "").startswith("Chorus")})
    write_json(out / "cues.json", cues)
    write_srt(cues, out / "subtitle.srt", display_timing=True, duration=seconds, amharic_only=True)
    for name in ("lyrics.txt", "description.txt", "tags.txt", "title.txt"):
        matches = list(old.rglob(name))
        if matches:
            shutil.copy2(matches[0], out / name)
    info = {"song_id": SONG_ID, "source_file": saved_source.name, "source_seconds": seconds,
            "expected_samples": expected, "sample_rate": sr, "source_sha256": digest(saved_source),
            "vocal_sha256": digest(saved_voice), "instrumental_sha256": digest(reference),
            "subtitle_sha256": digest(out / "subtitle.srt"),
            "source_title": kit.get("recommended_title", ""), "youtube_uploaded": False,
            "telegram_progress_changed": False}
    write_json(out / "source-info.json", info)
    print("FULL_SONG_PREPARED " + json.dumps(info), flush=True)


def encode_mp3(source, target, bitrate="320k", headroom=False):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source)]
    if headroom:
        cmd += ["-af", "volume=-2dB"]
    subprocess.run([*cmd, "-c:a", "libmp3lame", "-b:a", bitrate, str(target)], check=True)


def render(strength):
    started = time.monotonic()
    root = Path("prepared")
    info = json.loads((root / "source-info.json").read_text())
    source, voice, reference = root / info["source_file"], root / "vocals.wav", root / "instrumental-reference.wav"
    for path, key in ((source, "source_sha256"), (voice, "vocal_sha256"), (reference, "instrumental_sha256")):
        assert digest(path) == info[key], f"Prepared input changed: {path.name}"
    cues = json.loads((root / "cues.json").read_text())
    out, work = Path("output") / f"full-song-{strength}", Path(".runtime") / f"full-song-{strength}"
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    master = out / "master-float.wav"
    value = int(strength) / 100
    report = arrangement.rebuild(source, voice, master, work, seed=40, instrumental=reference, cues=cues,
                                 engine="ace-step", ai_task_type="cover", ai_cover_strengths=(value,),
                                 ai_enforce_source_fit=False, keep_instrumentals=True)
    assert report["cover_strengths"] == [value] and report["task_type"] == "cover"
    assert report["candidates_per_phrase"] == 1 and not report["source_fit_checks_enforced"]
    assert sf.info(master).frames == info["expected_samples"] and report["duration_locked"]
    assert report["vocal_offset_samples"] == report["vocal_pitch_shift_semitones"] == 0
    assert report["vocal_tempo_factor"] == 1 and report["true_peak_dbfs"] <= -0.999
    assert digest(voice) == info["vocal_sha256"] and digest(source) == info["source_sha256"]
    assert digest(reference) == info["instrumental_sha256"]
    write_json(out / "arrangement.json", report)
    for name in ("subtitle.srt", "lyrics.txt", "description.txt", "tags.txt", "title.txt"):
        if (root / name).exists():
            shutil.copy2(root / name, out / name)
    assert digest(out / "subtitle.srt") == info["subtitle_sha256"]
    encode_mp3(master, out / f"complete-song-{strength}.mp3")
    values, sr = sf.read(master, dtype="float32", always_2d=True)
    sf.write(out / f"complete-song-{strength}.flac", values, sr, subtype="PCM_24")
    assert sf.info(out / f"complete-song-{strength}.flac").frames == info["expected_samples"]
    for name in ("before-vocal-cleanup", "after-vocal-cleanup"):
        raw = out / "ai-backing" / (name + ".wav")
        encode_mp3(raw, out / (name + ".mp3"), bitrate="192k", headroom=True)
    # Keep all existing video frames/lyrics, but replace its audio with exactly this new vocal mix.
    video = out / "video.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(root / "original-lyric-video.mp4"),
                    "-i", str(master), "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "-1",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-t", f"{report['duration_seconds']:.9f}",
                    "-movflags", "+faststart", str(video)], check=True)
    video_info = probe(video)
    video_seconds = float(video_info["format"]["duration"])
    assert abs(video_seconds - info["source_seconds"]) <= 0.10
    assert len([stream for stream in video_info["streams"] if stream["codec_type"] == "audio"]) == 1
    verification = {"source": info, "strength": value, "source_samples": info["expected_samples"],
                    "master_samples": sf.info(master).frames, "video_seconds": video_seconds,
                    "vocal_and_subtitle_hashes_unchanged": True, "original_video_frames_reused": True,
                    "youtube_uploaded": False, "telegram_progress_changed": False,
                    "elapsed_seconds": time.monotonic() - started,
                    "experimental": True, "musical_fit_approved": False,
                    "note": "Fit concerns are recorded, not vetoed, for this requested full-song listening test."}
    write_json(out / "verification.json", verification)
    (out / "README.txt").write_text(
        f"FULL SONG {SONG_ID}: TRUE COVER, {value:.0%} SOURCE CONDITIONING\n\n"
        f"complete-song-{strength}.mp3: complete original singer plus the new accompaniment.\n"
        f"complete-song-{strength}.flac: 24-bit FLAC listening master.\n"
        "video.mp4: existing lyric-video frames with the new complete soundtrack; no new visual render.\n"
        "subtitle.srt: Amharic-script-only captions on the original vocal timeline.\n"
        "before-vocal-cleanup.mp3 and after-vocal-cleanup.mp3: full instrumental previews with 2dB headroom.\n\n"
        "This is a listening experiment, not an approved replacement. Harmony/rhythm fit concerns remain in diagnostics.\n"
        "The singer's pitch and tempo were not changed. Exact source sample count and caption hashes are checked.\n"
        "Constant mix/headroom gain is allowed. No original accompaniment is layered into the new mix.\n"
        "No YouTube upload or Telegram progress update was performed.\n"
        "Similar file sizes are expected from equal duration and bitrate. They do not mean equal audio.\n",
        encoding="utf-8")
    print("FULL_SONG_VERSION_READY " + json.dumps({"verification": verification,
          "generation_seconds": report["ai_generation_seconds"], "phrases": report["phrase_count"],
          "true_peak_dbfs": report["true_peak_dbfs"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--strength", choices=("050", "035"))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.strength:
        render(args.strength)
    else:
        parser.error("Choose --prepare or --strength")
