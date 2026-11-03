"""Small subtitle formatting helpers shared by the writers."""
import unicodedata


def is_amharic_text(text):
    """Require Ethiopic-script lyric letters, allowing normal punctuation, spaces and numbers."""
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and all(unicodedata.name(char, "").startswith("ETHIOPIC ") for char in letters)


def timestamp(seconds, ass=False):
    precision = 100 if ass else 1000
    ticks = round(max(0.0, seconds) * precision)
    hours, ticks = divmod(ticks, 3600 * precision)
    minutes, ticks = divmod(ticks, 60 * precision)
    whole, fraction = divmod(ticks, precision)
    return (f"{hours}:{minutes:02}:{whole:02}.{fraction:02}" if ass
            else f"{hours:02}:{minutes:02}:{whole:02},{fraction:03}")


def safe_ass(text):
    # Telegram titles and lyric text must never become ASS control tags.
    return " ".join(str(text).replace("\\", "").replace("{", "").replace("}", "").split())
