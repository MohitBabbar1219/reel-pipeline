# reel-pipeline

Generates a vertical (1080x1920, 30fps) Instagram-style reel from a one-line idea.

```
idea -> script + 8 image prompts + music pick   (OpenRouter LLM, default z-ai/glm-5.3)
     -> narration + word-level timing           (Kokoro TTS, runs locally)
     -> 8 images, 9:16                          (OpenRouter, bytedance-seed/seedream-4.5)
     -> karaoke captions                        (.ass, burned in by ffmpeg)
     -> reel.mp4                                (ffmpeg: zoom/crossfade slideshow + narration + ducked music)
```

## Requirements

| | |
|---|---|
| Python | **3.12 or 3.13.** `numpy==2.5.3` needs 3.12+, and `kokoro-onnx==0.6.1` needs `<3.14`. If your system `python3` is newer or older, call `python3.13` (or `python3.12`) explicitly. |
| ffmpeg + ffprobe | Must be built with **libass** (for the `subtitles` filter). The plain Homebrew `ffmpeg` formula isn't; see [step 4](#4-point-the-pipeline-at-your-ffmpeg). |
| OpenRouter account | An API key with credit. Script and image generation are paid calls. |
| Disk / network | ~350 MB for the Kokoro model files, downloaded once. |

## Setup

### 1. Get the code and create a virtualenv

```bash
cd reel-pipeline
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the Kokoro model files

These are not in `requirements.txt` and must sit in `models/`:

```bash
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

(`kokoro-v1.0.onnx` is ~326 MB, `voices-v1.0.bin` ~28 MB.)

### 3. Add your OpenRouter key

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=...
```

Optionally add `OPENROUTER_MODEL=<slug>` to `.env` to change the script model. The `--model` flag overrides it.

### 4. Point the pipeline at your ffmpeg

`assemble.py` hardcodes the ffmpeg location (lines 10-11):

```python
FFMPEG_BIN = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
FFPROBE_BIN = "/opt/homebrew/opt/ffmpeg-full/bin/ffprobe"
```

That path is correct for **Apple Silicon macOS** after `brew install ffmpeg-full`. On any other system, install an ffmpeg with libass and edit those two lines:

| System | Install | Typical path |
|---|---|---|
| Apple Silicon macOS | `brew install ffmpeg-full` | `/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg` (no edit needed) |
| Intel macOS | `brew install ffmpeg-full` | `/usr/local/opt/ffmpeg-full/bin/ffmpeg` |
| Debian/Ubuntu | `sudo apt install ffmpeg` | `/usr/bin/ffmpeg` |

Check that your build has the filters the pipeline uses (all three lines should print):

```bash
/path/to/ffmpeg -hide_banner -filters | grep -E " subtitles | xfade | zoompan "
```

### 5. Background music (optional)

The LLM picks a track from `music/list.json`, a list of `{"file_ref": "<file>.mp3", "description": "<mood/tempo>"}` entries, and `file_ref` is the filename inside `music/`. The `.mp3` files themselves are not generated; copy them over from the original machine or add your own tracks and list entries.

- No `music/list.json`: the reel is made without music.
- A picked `file_ref` that isn't in `music/`: a warning is printed and the music is skipped.

### 6. Check TTS works

```bash
python models/smoke_test.py
```

This loads the Kokoro model and prints the available voices. If it fails here, fix it before running the full pipeline.

## Usage

```bash
source .venv/bin/activate
python generate.py "a cat learning to skateboard"
python generate.py "a cat learning to skateboard" --model <openrouter-model-slug>
```

Each run writes to `output/<timestamp>-<slug>/`:

| File | What |
|---|---|
| `reel.mp4` | The final video |
| `script.txt` | Narration script |
| `audio.wav` | Narration |
| `img_0.png` ... `img_7.png` | The 8 generated images |
| `captions.ass` | Karaoke captions |

Every run also appends its cost to `costs.jsonl`. Earlier runs came to about $0.32 each, almost all of it image generation.

## Project layout

| Path | Role |
|---|---|
| `generate.py` | Entry point; runs the six steps in order |
| `llm.py` | OpenRouter chat call: script, 8 image prompts, music pick |
| `tts.py` | Kokoro narration + per-word timestamps |
| `images.py` | OpenRouter image generation (8 in parallel) |
| `captions.py` | Builds the `.ass` karaoke subtitles |
| `assemble.py` | ffmpeg assembly (slideshow, captions, audio mix) |
| `models/` | Kokoro model files, plus `smoke_test.py` and `word_timing_test.py` |
| `music/` | Background tracks + `list.json` |

## Troubleshooting

- **`OPENROUTER_API_KEY is not set`**: `.env` is missing or empty. It is read from the project directory, not your current directory.
- **`pip install` fails resolving `numpy==2.5.3` or `kokoro-onnx`**: wrong Python version; see Requirements.
- **`ffmpeg failed` mentioning `subtitles` or `No such filter`**: your ffmpeg lacks libass; use the filter check in step 4.
- **Captions render in a different font**: `captions.py` uses Arial Black. If that font isn't installed (common on Linux), libass substitutes another one. Install the font or change `Fontname` in `ASS_HEADER`.
- **Image request returns 403 "not available in your region"**: some OpenRouter image models are geo-gated, which is why the default is `bytedance-seed/seedream-4.5`. Change `DEFAULT_MODEL` in `images.py` if your account can't use it.
