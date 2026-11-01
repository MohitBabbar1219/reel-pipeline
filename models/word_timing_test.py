import dataclasses
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro

HERE = Path(__file__).parent
kokoro = Kokoro(str(HERE / "kokoro-v1.0.onnx"), str(HERE / "voices-v1.0.bin"))

TEXT = "The quick brown fox jumps over the lazy dog."
VOICE = "af_bella"

print("=== 1. Raw Timing dataclass shape ===")
print("dataclasses.fields(Timing):",
      [f.name for f in dataclasses.fields(type(
          kokoro.create_timed(TEXT, voice=VOICE)[2][0]
      ))])

samples, sample_rate, timings = kokoro.create_timed(TEXT, voice=VOICE, speed=1.0, lang="en-us")

print(f"\ntotal phoneme-level timing entries: {len(timings)}")
print("first 10 raw entries:")
for t in timings[:10]:
    print(f"  {t!r}")

print(f"\naudio: {len(samples)} samples @ {sample_rate}Hz = {len(samples)/sample_rate:.3f}s")

print("\n=== 2. Group phoneme entries into words on space-phoneme boundaries ===")


def group_words(timings):
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


groups = group_words(timings)
print(f"phoneme groups found: {len(groups)}")
for i, g in enumerate(groups):
    phon = "".join(t.phoneme for t in g)
    print(f"  group[{i}]: phonemes={phon!r} start={g[0].start:.3f} end={g[-1].end:.3f}")

words = TEXT.rstrip(".!?").split()
print(f"\nwords in source text (punctuation stripped): {len(words)} -> {words}")

print("\n=== word-level (word, start, end) tuples, zipped positionally ===")
if len(groups) != len(words):
    print(f"!!! MISMATCH: {len(groups)} phoneme groups vs {len(words)} words - inspect above before zipping")
word_timings = list(zip(words, groups))
result = [(w, g[0].start, g[-1].end) for w, g in word_timings]
for w, start, end in result:
    print(f"  {w!r:12s} start={start:.3f} end={end:.3f}")

print("\n=== 3. Sanity checks ===")
print(f"word count matches text word count: {len(result) == len(words)} ({len(result)} vs {len(words)})")
starts = [r[1] for r in result]
print(f"start times strictly increasing: {all(a < b for a, b in zip(starts, starts[1:]))}")
print(f"first word start: {result[0][1]:.3f}s, last word end: {result[-1][2]:.3f}s, total audio: {len(samples)/sample_rate:.3f}s")

print("\n=== 4. Does create_timed also return usable audio samples? ===")
out_wav = HERE / "word_timing_test_output.wav"
sf.write(str(out_wav), samples, sample_rate)
print(f"wrote {out_wav} ({out_wav.stat().st_size} bytes) directly from create_timed's samples - no separate create() call made")

# compare against a plain create() call on the same text/voice to confirm equivalence
plain_samples, plain_sr = kokoro.create(TEXT, voice=VOICE, speed=1.0, lang="en-us")
print(f"plain create(): {len(plain_samples)} samples @ {plain_sr}Hz = {len(plain_samples)/plain_sr:.3f}s")
print(f"create_timed(): {len(samples)} samples @ {sample_rate}Hz = {len(samples)/sample_rate:.3f}s")
print(f"sample arrays identical: {plain_samples.shape == samples.shape and bool((plain_samples == samples).all())}")
