from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFont


@dataclass
class Cue:
    start: float
    end: float
    words: list[dict]

    @property
    def text(self):
        return " ".join(word["word"] for word in self.words)


def make_cues(words):
    cues, group = [], []
    for word in words:
        if group and (
            len(group) >= 7
            or len(" ".join(w["word"] for w in group + [word])) > 42
            or word["end"] - group[0]["start"] > 6
            or word["start"] - group[-1]["end"] > 0.8
        ):
            cues.append(Cue(group[0]["start"], group[-1]["end"], group))
            group = []
        group.append(word)
    if group:
        cues.append(Cue(group[0]["start"], group[-1]["end"], group))
    return cues


def timestamp(seconds, ass=False):
    precision = 100 if ass else 1000
    ticks = round(seconds * precision)
    hours, ticks = divmod(ticks, 3600 * precision)
    minutes, ticks = divmod(ticks, 60 * precision)
    whole, fraction = divmod(ticks, precision)
    return (f"{hours}:{minutes:02}:{whole:02}.{fraction:02}" if ass
            else f"{hours:02}:{minutes:02}:{whole:02},{fraction:03}")


def safe_ass(text):
    # Telegram titles and recognizer output must never become ASS control tags.
    return " ".join(str(text).replace("\\", "").replace("{", "").replace("}", "").split())


def font_path():
    import subprocess

    path = subprocess.check_output(
        ["fc-match", "-f", "%{file}", "Noto Sans Ethiopic"], text=True,
    ).strip()
    if "Ethiopic" not in path:
        raise RuntimeError("Install fonts-noto-core: Noto Sans Ethiopic is required")
    return path


def fitted_size(text, font, maximum, width):
    size = maximum
    while size > 8 and ImageFont.truetype(font, size).getlength(text) > width:
        size -= 1
    if ImageFont.truetype(font, size).getlength(text) > width:
        raise ValueError("A lyric/title is too long to fit the video")
    return size


def write_lyrics(folder: Path, cues, title: str, seconds: float):
    lyrics = "\n".join(cue.text for cue in cues) + "\n"
    (folder / "lyrics.txt").write_text(lyrics, encoding="utf-8")
    srt = "\n\n".join(
        f"{index}\n{timestamp(cue.start)} --> {timestamp(cue.end)}\n{cue.text}"
        for index, cue in enumerate(cues, 1)
    )
    (folder / "lyrics.srt").write_text(srt + "\n", encoding="utf-8")
    font = font_path()
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,Noto Sans Ethiopic,76,&H009AE8A7,&H00FFFFFF,&H00171413,&H88000000,0,0,0,0,100,100,0,0,1,2,1,5,130,130,60,1
Style: Title,Noto Sans Ethiopic,42,&H00DDDAD9,&H00DDDAD9,&H00171413,&H88000000,0,0,0,0,100,100,0,0,1,1,0,8,160,160,110,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    title = safe_ass(title)
    title_size = fitted_size(title, font, 42, 1580)
    events = [f"Dialogue: 0,0:00:00.00,{timestamp(seconds, True)},Title,,0,0,0,,{{\\fs{title_size}}}{title}"]
    for cue in cues:
        size = fitted_size(safe_ass(cue.text), font, 76, 1640)
        parts = [f"{{\\fs{size}}}"]
        cursor = round(cue.start * 100)
        for word in cue.words:
            start, end = round(word["start"] * 100), round(word["end"] * 100)
            if start > cursor:
                parts.append(f"{{\\k{start - cursor}}}")
            parts.append(f"{{\\kf{max(1, end - start)}}}{safe_ass(word['word'])} ")
            cursor = end
        events.append(f"Dialogue: 1,{timestamp(cue.start, True)},{timestamp(cue.end, True)},Lyrics,,0,0,0,,{''.join(parts).rstrip()}")
    (folder / "lyrics.ass").write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return lyrics
