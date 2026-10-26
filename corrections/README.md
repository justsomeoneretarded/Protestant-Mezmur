# Lyric corrections

To fix the lyrics of a song, create `corrections/<telegram message id>.txt` here
(GitHub web editor works fine, also on a phone) and commit it to `main`.
The **Generate Mezmur Videos** workflow starts automatically for that song and
re-times your text against the audio — you only fix words, never timings.

```
---
title_am: ሲወድ ጌታ
title_latin: Siwed Geta
artist_am: አዲስዓለም አሰፋ
artist_latin: Addisalem Assefa
---
ሲወድ ጌታ እስከ መስቀል ነው
ፍቅሩ አይለካም ከቶ

አዝ፦ ሃሌሉያ ክብር ለስሙ
ዘመኔን ሁሉ አመሰግናለሁ
```

* The `---` block is optional; use it only to fix the title/artist spelling.
* One sung line per text line; a blank line between verses.
* Mark the chorus with `አዝ፦` (or `Chorus:`) once — repeats are found in the audio.
* Easiest start: copy `lyrics.txt` from the song's artifact and edit it.
