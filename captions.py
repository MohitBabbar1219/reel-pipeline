"""Builds karaoke-style ASS captions from tts.py's word_timings.

Words are chunked into short on-screen groups (~3-5 words, breaking
preferentially at punctuation). Each chunk becomes one ASS Dialogue line
using per-word \\k karaoke tags - no smooth sweep, an instant per-word
switch (see IMPLEMENTATION.md §3.3).

Color mapping note (verified empirically, not from spec-reading): libass's
\\k renders the *not-yet-reached* portion of the line in SecondaryColour
and the *already-reached* portion (from the start of the currently-active
word onward) in PrimaryColour - the opposite of what the tag names
suggest. So PrimaryColour is set to yellow and SecondaryColour to white
below, to get "turns yellow as it's read" rather than the inverse.

Gap-timing note (found 2026-09-28 from a user-reported sync issue): each
\\k duration must span to the START of the NEXT word, not to this word's
own end - see the comment in build_ass() for why.

Wrap/margin note (found 2026-09-28 from a user report of cropped
captions): the original WrapStyle: 2 means "no wrapping, one line no
matter what" in ASS - a wide 5-word chunk at this Fontsize (76) simply ran
past the frame edge and got clipped instead of dropping to a second line.
Changed to WrapStyle: 0 (smart/balanced 2-line wrapping) and widened
MarginL/MarginR from 60 to 90 for extra horizontal breathing room.
"""

VIDEO_W = 1080
VIDEO_H = 1920

MIN_CHUNK = 3
MAX_CHUNK = 5

ASS_HEADER = f"""[Script Info]
Title: Reel Captions
ScriptType: v4.00+
PlayResX: {VIDEO_W}
PlayResY: {VIDEO_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,Arial Black,76,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,5,2,2,90,90,320,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_ts(seconds):
    seconds = max(0.0, seconds)
    cs_total = round(seconds * 100)
    h, rem = divmod(cs_total, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _chunk_words(word_timings, min_size=MIN_CHUNK, max_size=MAX_CHUNK):
    chunks = []
    current = []
    for wt in word_timings:
        current.append(wt)
        word = wt[0]
        ends_with_punct = bool(word) and word[-1] in ".,!?;:"
        if len(current) >= max_size or (len(current) >= min_size and ends_with_punct):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def build_ass(word_timings, out_path):
    """Write an ASS subtitle file with karaoke word-highlight timing.

    word_timings: [(word, start_seconds, end_seconds), ...] from
    tts.synthesize(). Returns out_path.
    """
    chunks = _chunk_words(word_timings)
    lines = []
    for chunk in chunks:
        start = chunk[0][1]
        end = chunk[-1][2]
        text_parts = []
        for i, (word, wstart, wend) in enumerate(chunk):
            # \k duration must run to the START of the NEXT word, not to this
            # word's own end - ASS karaoke timing is cumulative with no way to
            # encode a silent gap between \k segments, so the inter-word gap
            # has to be absorbed into the current word's duration (highlight
            # holds through the pause until the next word actually starts).
            # Using (wend-wstart) alone drops every inter-word gap, so the
            # highlight races ahead of the audio within each chunk, resetting
            # at the next chunk's Dialogue line - a real, audible/visible
            # sawtooth desync found 2026-09-28 from a user report.
            next_start = chunk[i + 1][1] if i + 1 < len(chunk) else end
            dur_cs = max(1, round((next_start - wstart) * 100))
            escaped = word.replace("{", "(").replace("}", ")")
            text_parts.append(f"{{\\k{dur_cs}}}{escaped} ")
        text = "".join(text_parts).rstrip()
        lines.append(
            f"Dialogue: 0,{_fmt_ts(start)},{_fmt_ts(end)},Karaoke,,0,0,0,,{text}"
        )

    out_path = str(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER)
        f.write("\n".join(lines))
        f.write("\n")
    return out_path
