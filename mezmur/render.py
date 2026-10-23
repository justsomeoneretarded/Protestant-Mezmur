from pathlib import Path

from .common import run


def render(audio: Path, folder: Path, seconds: float, height=1080):
    width = height * 16 // 9
    # Calculate the moving color field at low resolution, then upscale smoothly.
    background = (
        "nullsrc=s=320x180:r=24,geq="
        "r='19+9*sin(X/78+T/4)+5*sin(Y/45-T/3)':"
        "g='26+10*sin(X/90+T/5)+6*cos(Y/60+T/4)':"
        "b='30+10*cos(X/100-T/6)+8*sin(Y/55+T/5)',"
        f"scale={width}:{height}:flags=bilinear,format=yuv420p[bg];"
    )
    filters = (
        background
        + f"[0:a]aformat=channel_layouts=mono,showwaves=s={width}x{height // 6}:"
        "mode=cline:rate=24:colors=0x94d6b0:scale=sqrt:draw=full,format=rgba,colorchannelmixer=aa=0.8[wave];"
        + f"[bg][wave]overlay=0:H-h-{height // 12}:shortest=1,"
        "ass=lyrics.ass[v]"
    )
    run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio.resolve()),
        "-filter_complex_threads", "2", "-filter_complex", filters,
        "-map", "[v]", "-map", "0:a:0", "-t", str(seconds),
        "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", "video.mp4",
    ], cwd=folder)


if __name__ == "__main__":
    import argparse
    from .common import duration

    parser = argparse.ArgumentParser(description="Rerender an edited lyrics.ass using original audio")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--height", type=int, choices=[720, 1080], default=1080)
    args = parser.parse_args()
    if not (args.folder / "lyrics.ass").is_file():
        parser.error("--folder must contain lyrics.ass")
    render(args.audio, args.folder, duration(args.audio), args.height)
