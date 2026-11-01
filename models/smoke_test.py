import time
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro

HERE = Path(__file__).parent
MODEL_PATH = HERE / "kokoro-v1.0.onnx"
VOICES_PATH = HERE / "voices-v1.0.bin"

t0 = time.time()
kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
print(f"Kokoro() load time: {time.time() - t0:.1f}s")

voices = kokoro.get_voices()
print(f"\n{len(voices)} voices available:")
print(voices)

voice = "af_bella" if "af_bella" in voices else voices[0]
print(f"\nUsing voice: {voice}")

text = "This is a test of the Kokoro text to speech engine."

t0 = time.time()
samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
print(f"create() time: {time.time() - t0:.1f}s")
duration = len(samples) / sample_rate
print(f"sample_rate={sample_rate}, duration={duration:.2f}s, samples={len(samples)}")

out_wav = HERE / "test_output.wav"
sf.write(str(out_wav), samples, sample_rate)
print(f"Wrote {out_wav} ({out_wav.stat().st_size} bytes)")

print("\n--- create_timed test ---")
t0 = time.time()
timed_samples, timed_sr, timings = kokoro.create_timed(text, voice=voice, speed=1.0, lang="en-us")
print(f"create_timed() time: {time.time() - t0:.1f}s")
print(f"timings returned: {len(timings)} entries")
if timings:
    for t in timings[:15]:
        print(f"  phoneme={t.phoneme!r} start={t.start:.3f} end={t.end:.3f}")
    if len(timings) > 15:
        print(f"  ... ({len(timings) - 15} more)")
else:
    print("  EMPTY - this model/version does not expose duration output.")
