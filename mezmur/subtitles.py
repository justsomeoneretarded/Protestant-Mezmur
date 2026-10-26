"""Cue construction and subtitle writers (ASS karaoke, SRT, plain text)."""
from pathlib import Path

from PIL import ImageFont

from .geez import romanize
from .lyrics import safe_ass, timestamp

FONTS = Path(__file__).parent / "assets" / "fonts"
LEAD = 0.45          # line appears before its first word
MAX_WIDTH = 1700     # px at PlayResX 1920


def _measure(font_file, size, text):
    return ImageFont.truetype(str(FONTS / font_file), size).getlength(text)


def fit_size(text, font_file="NotoSansEthiopic-Bold.ttf", maximum=92, minimum=56, width=MAX_WIDTH):
    size = maximum
    while size > minimum and _measure(font_file, size, text) > width:
        size -= 2
    return size


def build_cues(words, line_texts=None, chorus_lines=frozenset(), max_gap=1.2):
    """Group timed words into display cues.

    words: [{"word", "start", "end", "line" (optional)}]. With lyric line ids, a cue is one
    continuous sung occurrence of a line; otherwise phrases are cut at pauses / length.
    """
    cues, group = [], []

    def close():
        if group:
            line = group[0].get("line")
            cues.append({"start": group[0]["start"], "end": group[-1]["end"], "words": list(group), "line": line,
                         "chorus": line in chorus_lines if line is not None else False})
            group.clear()

    for w in words:
        if group:
            same_line = w.get("line") is not None and w.get("line") == group[-1].get("line")
            gap = w["start"] - group[-1]["end"]
            new_occurrence = same_line and w.get("word_index", 0) <= group[-1].get("word_index", -1)
            if w.get("line") is None:
                text = " ".join(x["word"] for x in group + [w])
                if gap > 0.6 or len(group) >= 7 or len(text) > 40:
                    close()
            elif not same_line or new_occurrence or gap > max_gap:
                close()
        group.append(w)
    close()
    # Split cues that cannot fit on one line at the minimum size.
    out = []
    for cue in cues:
        text = " ".join(w["word"] for w in cue["words"])
        if len(cue["words"]) > 3 and _measure("NotoSansEthiopic-Bold.ttf", 56, text) > MAX_WIDTH:
            half = len(cue["words"]) // 2
            for part in (cue["words"][:half], cue["words"][half:]):
                out.append({**cue, "start": part[0]["start"], "end": part[-1]["end"], "words": part})
        else:
            out.append(cue)
    for cue in out:
        cue["text"] = " ".join(w["word"] for w in cue["words"])
        cue["romanized"] = " ".join(romanize(w["word"]) for w in cue["words"])
    return out


def _karaoke(words, event_start, roman=False):
    parts, cursor = [], round(event_start * 100)
    for w in words:
        start, end = round(w["start"] * 100), round(w["end"] * 100)
        if start > cursor:
            parts.append(f"{{\\k{start - cursor}}}")
            cursor = start
        text = romanize(w["word"]) if roman else w["word"]
        parts.append(f"{{\\kf{max(1, end - cursor)}}}{safe_ass(text)} ")
        cursor = max(cursor, end)
    return "".join(parts).rstrip()


HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyric,Noto Sans Ethiopic,92,&H008CD6FF,&H00FFFFFF,&H00100A04,&H96000000,-1,0,0,0,100,100,1,0,1,4.5,3,5,60,60,0,1
Style: Roman,Noto Sans,44,&H00A8DEFF,&H00E6E6E6,&H00100A04,&H96000000,0,-1,0,0,100,100,1,0,1,2.6,2,5,60,60,0,1
Style: Next,Noto Sans Ethiopic,46,&H50FFFFFF,&H50FFFFFF,&H70100A04,&HC8000000,0,0,0,0,100,100,1,0,1,2,1,5,60,60,0,1
Style: Section,Noto Sans,30,&H008CD6FF,&H008CD6FF,&H00100A04,&H96000000,-1,0,0,0,100,100,6,0,1,2,1,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def write_ass(cues, path, romanized=True, preview=True):
    events = []
    for i, cue in enumerate(cues):
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        prev_end = cues[i - 1]["end"] if i else 0.0
        start = max(cue["start"] - LEAD, prev_end + 0.05 if i else 0.0, 0.0)
        end = cue["end"] + 0.5
        if nxt:
            end = min(end, max(cue["end"] + 0.05, nxt["start"] - LEAD - 0.02))
        size = fit_size(cue["text"])
        fade = "\\fad(180,220)"
        move = f"\\move(960,{578},960,560,0,260)"
        events.append(f"Dialogue: 2,{timestamp(start, True)},{timestamp(end, True)},Lyric,,0,0,0,,"
                      f"{{{move}{fade}\\fs{size}}}{_karaoke(cue['words'], start)}")
        if romanized and cue.get("romanized"):
            rsize = 44 if _measure("NotoSans-MediumItalic.ttf", 44, cue["romanized"]) < MAX_WIDTH else 34
            events.append(f"Dialogue: 2,{timestamp(start, True)},{timestamp(end, True)},Roman,,0,0,0,,"
                          f"{{\\move(960,{672},960,656,0,260){fade}\\fs{rsize}}}{_karaoke(cue['words'], start, roman=True)}")
        if preview and nxt and nxt["start"] - cue["end"] < 6.0:
            nsize = min(46, fit_size(nxt["text"], maximum=46, minimum=30))
            events.append(f"Dialogue: 1,{timestamp(start, True)},{timestamp(end, True)},Next,,0,0,0,,"
                          f"{{\\pos(960,756){fade}\\fs{nsize}}}{safe_ass(nxt['text'])}")
        first_of_chorus = cue.get("chorus") and not (i and cues[i - 1].get("chorus") and cue["start"] - cues[i - 1]["end"] < 3)
        if first_of_chorus:
            events.append(f"Dialogue: 3,{timestamp(start, True)},{timestamp(min(end, start + 3.5), True)},Section,,0,0,0,,"
                          f"{{\\pos(960,452)\\fad(250,400)}}አዝማች  •  CHORUS")
    Path(path).write_text(HEADER + "\n".join(events) + "\n", encoding="utf-8")


def write_srt(cues, path, romanized=False):
    blocks = []
    for i, cue in enumerate(cues, 1):
        text = cue["romanized"] if romanized else cue["text"]
        blocks.append(f"{i}\n{timestamp(cue['start'])} --> {timestamp(cue['end'] + 0.3)}\n{text}")
    Path(path).write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
