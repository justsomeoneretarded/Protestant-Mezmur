from pathlib import Path

from .common import run


def render(audio: Path, folder: Path, seconds: float, height=1080):
    output_width = height * 16 // 9
    output_height = height
    # The scenic atlas is photographic; blend its layers at a practical CPU resolution.
    # Upscale once before ASS so the lyric glyphs still render at the full output size.
    height = min(height, 540)
    width = height * 16 // 9
    atlas = Path(__file__).parent / "assets" / "scenery.png"
    # Two moving crops of the atlas crossfade over the final two seconds of each scene.
    # Both cameras share a continuous clock, keeping the four-scene loop seamless.
    cameras = []
    for index, name in enumerate(["current", "next"]):
        scene = f"floor(t/20)+{index}"
        cameras.append(
            f"[{name}]crop=w=trunc(iw/4)*2:h=trunc(ih/4)*2:"
            f"x='mod({scene},2)*out_w':y='floor(mod({scene},4)/2)*out_h',"
            f"scale={width * 6 // 5 // 2 * 2}:{height * 6 // 5 // 2 * 2},"
            f"crop={width}:{height}:x='(iw-ow)*(0.5+0.45*sin(t/17))':"
            "y='(ih-oh)*(0.5+0.35*cos(t/23))',setsar=1[scene" + str(index) + "];"
        )
    filters = (
        "[1:v]fps=24,split=2[current][next];" + "".join(cameras)
        + "[scene0][scene1]blend=all_expr='A*(1-max(0,(mod(T,20)-18)/2))+B*max(0,(mod(T,20)-18)/2)':"
        "enable='gte(mod(t,20),18)',"
        "eq=saturation=0.90:brightness=-0.05[scenery];"
        "color=c=black:s=320x180:r=24,format=rgba,"
        "geq=r=0:g=0:b=0:a='120*exp(-pow((Y-H*0.51)/(H*0.20),4))',"
        f"scale={width}:{height}[shade];"
        "[scenery][shade]overlay=0:0[bg];"
        "[0:a]aformat=channel_layouts=mono,asplit=2[wa][sa];"
        f"[wa]showwaves=s={width}x{height // 10}:mode=cline:rate=24:"
        "colors=0xffe4a3:scale=sqrt:draw=full,format=rgba,colorchannelmixer=aa=0.65[wave];"
        f"[sa]showfreqs=s={width}x{height // 8}:mode=bar:ascale=sqrt:fscale=log:"
        "win_size=2048:colors=0x9be7d5:rate=24,format=rgba,colorchannelmixer=aa=0.45[spectrum];"
        f"[bg][spectrum]overlay=0:H-h-{height // 12}:shortest=1[withspectrum];"
        f"[withspectrum][wave]overlay=0:H-h-{height // 40}:shortest=1,"
        f"scale={output_width}:{output_height}:flags=lanczos,ass=lyrics.ass[v]"
    )
    run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(audio.resolve()),
        "-loop", "1", "-framerate", "1", "-i", str(atlas.resolve()),
        "-filter_complex_threads", "2", "-filter_complex", filters,
        "-map", "[v]", "-map", "0:a:0", "-t", str(seconds),
        "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", "-progress", "render.progress", "video.mp4",
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
