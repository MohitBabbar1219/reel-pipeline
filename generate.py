#!/usr/bin/env python3
"""Entry point: python generate.py "<prompt>" [--model <openrouter-slug>]

Runs the full pipeline start to finish: script+prompts (OpenRouter LLM) ->
narration+word timing (Kokoro, local) -> 8 images (OpenRouter Seedream 4.5) ->
captions (ASS karaoke) -> ffmpeg assembly -> reel.mp4. See IMPLEMENTATION.md.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

import assemble
import captions
import images
import llm
import tts

DEFAULT_SCRIPT_MODEL = "z-ai/glm-5.3"


def _slugify(text, max_len=40):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "reel"


def _load_music_list():
    path = HERE / "music" / "list.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def run(prompt, model=None):
    model = model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_SCRIPT_MODEL

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slugify(prompt)}"
    out_dir = HERE / "output" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/6] run_id={run_id}  model={model}")

    music_list = _load_music_list()

    print("[2/6] generating script + image prompts + music pick...")
    payload, script_cost = llm.generate_script(prompt, music_list=music_list, model=model)
    script = payload["script"]
    image_prompts = payload["image_prompts"]
    music_ref = payload["music_file_ref"]
    (out_dir / "script.txt").write_text(script)
    print(f"    script ({len(script.split())} words): {script[:80]}...")
    print(f"    music: {music_ref}")

    print("[3/6] synthesizing narration (Kokoro, af_sky)...")
    wav_path, duration, word_timings = tts.synthesize(script, out_dir / "audio.wav")
    print(f"    {duration:.1f}s narration, {len(word_timings)} words timed")

    print("[4/6] generating 8 images (OpenRouter Seedream 4.5)...")
    image_results = images.generate_images(image_prompts, str(out_dir))
    image_paths = [p for p, _ in image_results]
    image_cost = sum(c for _, c in image_results)
    print(f"    {len(image_paths)} images written")

    print("[5/6] building captions...")
    ass_path = captions.build_ass(word_timings, out_dir / "captions.ass")

    music_path = None
    if music_ref:
        candidate = HERE / "music" / music_ref
        if candidate.exists():
            music_path = str(candidate)
        else:
            print(f"    warning: music_file_ref {music_ref!r} picked but file not found, skipping music")

    print("[6/6] assembling reel.mp4 (ffmpeg)...")
    reel_path = assemble.assemble(
        image_paths, wav_path, duration, ass_path, out_dir / "reel.mp4", music_path=music_path
    )

    total_cost = script_cost + image_cost
    print(f"\nDone: {reel_path}")
    print(f"Cost: ${total_cost:.4f} (script ${script_cost:.4f}, images ${image_cost:.4f})")

    cost_record = {
        "run_id": run_id,
        "prompt": prompt,
        "script_cost": script_cost,
        "image_cost": image_cost,
        "total_cost": total_cost,
    }
    with open(HERE / "costs.jsonl", "a") as f:
        f.write(json.dumps(cost_record) + "\n")

    return reel_path


def main():
    parser = argparse.ArgumentParser(description="Generate an AI Instagram reel from a prompt.")
    parser.add_argument("prompt", help="The idea/prompt for the reel.")
    parser.add_argument(
        "--model", default=None, help="OpenRouter model slug for script generation (default: z-ai/glm-5.3)."
    )
    args = parser.parse_args()

    try:
        run(args.prompt, model=args.model)
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
