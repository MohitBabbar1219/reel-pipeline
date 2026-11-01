"""OpenRouter image generation calls - one image per prompt, fired concurrently.

See IMPLEMENTATION.md §3.4. Synchronous endpoint, base64 response - no
polling, no second HTTP round-trip to fetch a hosted URL.
"""

import base64
import concurrent.futures
import os

import requests

OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
# meta/muse-image (the doc's chosen default) returns 403 "not available in
# your region" for this account - confirmed geo-gated by OpenRouter itself,
# not a code issue. Using the doc's documented fallback until this is
# resolved with the user - see IMPLEMENTATION.md for the pending decision.
DEFAULT_MODEL = "bytedance-seed/seedream-4.5"


def _api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to reel-pipeline/.env "
            "(see .env.example) before generating images."
        )
    return key


def _generate_one(prompt, out_path, model, aspect_ratio):
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "usage": {"include": True},
    }

    def _attempt():
        resp = requests.post(OPENROUTER_IMAGES_URL, headers=headers, json=body, timeout=180)
        resp.raise_for_status()
        return resp.json()

    try:
        data = _attempt()
    except Exception:
        data = _attempt()  # 1 retry per failed image, per doc's failure policy

    image_b64 = data["data"][0]["b64_json"]
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(image_b64))

    cost = data.get("usage", {}).get("cost", 0.0) or 0.0
    return str(out_path), cost


def generate_images(prompts, out_dir, model=DEFAULT_MODEL, aspect_ratio="9:16"):
    """Fire all prompts concurrently. Returns [(path, cost), ...] in prompt order.

    Any image that still fails after its retry propagates the exception,
    which aborts the whole run - matches the doc's v1 failure policy (no
    placeholder-image fallback yet).
    """
    results = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        futures = {
            ex.submit(_generate_one, p, f"{out_dir}/img_{i}.png", model, aspect_ratio): i
            for i, p in enumerate(prompts)
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    return results
