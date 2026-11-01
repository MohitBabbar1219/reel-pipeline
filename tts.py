"""Kokoro TTS synthesis with word-level timing.

Uses Kokoro.create_timed() instead of create() - it returns per-phoneme
timing alongside the audio in one call, with word boundaries marked by
literal phoneme=' ' entries. Grouping on those boundaries and zipping
positionally against text.split() gives exact word-level timestamps with
no separate alignment step or STT dependency (see IMPLEMENTATION.md §3.2).
"""

from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro

HERE = Path(__file__).parent
MODEL_PATH = HERE / "models" / "kokoro-v1.0.onnx"
VOICES_PATH = HERE / "models" / "voices-v1.0.bin"

DEFAULT_VOICE = "am_adam"
DEFAULT_SPEED = 1.0

_kokoro = None


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    return _kokoro


_PUNCT = set(".,!?;:")


def _group_words(timings):
    groups = []
    current = []
    for t in timings:
        if t.phoneme == " ":
            if current:
                groups.append(current)
                current = []
            continue
        current.append(t)
    if current:
        groups.append(current)
    return groups


def _split_merged_group(group, n_words):
    """Split one phoneme group that actually spans n_words (>1) real words.

    First tries internal (non-final) punctuation phonemes as split points -
    punctuation always trails its own word in the source text, so this
    exactly locates the boundary (handles the common case: two words
    merged because create_timed() omitted the space entry after a
    sentence/clause-ending mark). Falls back to an even phoneme-count split
    for the rarer case of a merge with no punctuation cue at all.
    """
    phonemes = [t.phoneme for t in group]
    punct_positions = [i for i, p in enumerate(phonemes[:-1]) if p in _PUNCT]
    if len(punct_positions) >= n_words - 1:
        cuts = punct_positions[: n_words - 1]
        spans, start_i = [], 0
        for c in cuts:
            spans.append(group[start_i : c + 1])
            start_i = c + 1
        spans.append(group[start_i:])
        return spans
    per = max(1, len(group) // n_words)
    spans = [group[i * per : (i + 1) * per] for i in range(n_words - 1)]
    spans.append(group[(n_words - 1) * per :])
    return [s for s in spans if s] or [group]


def _align_groups_to_words(raw_groups, n_words):
    """Reconcile raw_groups (split purely on literal space-phoneme
    boundaries) against the known word count, splitting any group that
    actually represents multiple merged words. Returns None if it can't
    reconcile the counts (caller falls back to flat interpolation - see
    IMPLEMENTATION.md §7.2 on the phoneme->word grouping gotcha).

    Found 2026-09-28 from a user-reported caption sync bug: create_timed()
    sometimes omits the space-phoneme entry between two words entirely
    (not a zero-duration space - no entry at all) when its prosody model
    picks a near-zero pause there, e.g. "over." immediately followed by
    "but" with nothing between them. That silently merges 2+ words into
    one group. The OLD behavior treated any count mismatch as fatal and
    fell back to evenly dividing the ENTIRE clip's duration across every
    word regardless of actual content - on a real 158-word/45.9s script,
    3 merged pairs (out of 157 word-to-word boundaries, ~2%) caused ALL 158
    words to lose real timing, which is what actually broke the karaoke
    sync, not a small per-chunk drift.
    """
    if len(raw_groups) == n_words:
        return raw_groups
    if len(raw_groups) > n_words:
        return None  # more groups than words - unexpected, don't guess

    deficit = n_words - len(raw_groups)
    result = []
    for group in raw_groups:
        phonemes = [t.phoneme for t in group]
        internal_punct = sum(1 for p in phonemes[:-1] if p in _PUNCT)
        take = min(deficit, internal_punct) if deficit > 0 else 0
        if take > 0:
            result.extend(_split_merged_group(group, take + 1))
            deficit -= take
        else:
            result.append(group)
    if deficit > 0:
        # still short with no punctuation cues left - split the largest
        # remaining groups evenly by phoneme count until counts reconcile.
        while deficit > 0:
            idx = max(range(len(result)), key=lambda i: len(result[i]))
            a, b = _split_merged_group(result[idx], 2)
            result[idx : idx + 1] = [a, b]
            deficit -= 1
    return result if len(result) == n_words else None


def synthesize(text, out_path, voice=DEFAULT_VOICE, speed=DEFAULT_SPEED):
    """Synthesize `text` to `out_path` (.wav).

    Returns (wav_path, duration_seconds, word_timings), where word_timings
    is [(word, start_seconds, end_seconds), ...] - one entry per word in
    text.split(), in order, with punctuation attached to the word exactly
    as it appears in the source text (so captions can display it as-is).
    """
    kokoro = _get_kokoro()
    try:
        samples, sample_rate, timings = kokoro.create_timed(
            text, voice=voice, speed=speed, lang="en-us"
        )
    except Exception:
        # mirrors the known rare onnxruntime thread-pool init crash
        # mitigation used elsewhere for this same engine
        samples, sample_rate, timings = kokoro.create_timed(
            text, voice=voice, speed=speed, lang="en-us"
        )

    out_path = str(out_path)
    sf.write(out_path, samples, sample_rate)
    duration_seconds = len(samples) / sample_rate

    words = text.split()
    raw_groups = _group_words(timings)
    groups = _align_groups_to_words(raw_groups, len(words))
    if groups is not None:
        word_timings = [(w, g[0].start, g[-1].end) for w, g in zip(words, groups)]
    else:
        # Couldn't reconcile group/word counts even after trying to split
        # merged groups (see _align_groups_to_words) - fall back to even
        # interpolation across the clip rather than crash the run. Should
        # be rare; if this fires often, check what's unusual about the
        # input text (see IMPLEMENTATION.md §7.2).
        step = duration_seconds / max(len(words), 1)
        word_timings = [(w, i * step, (i + 1) * step) for i, w in enumerate(words)]

    return out_path, duration_seconds, word_timings


def list_voices():
    return _get_kokoro().get_voices()
