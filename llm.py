"""OpenRouter chat-completions call for script + 8 image prompts + music pick.

See IMPLEMENTATION.md §3.1. One call produces everything creative in one
shot so the script and image prompts stay thematically consistent.
"""

import json
import os

import requests

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "z-ai/glm-5.3"

SYSTEM_PROMPT = """You are a scriptwriter for short-form vertical video reels (Instagram Reels style).

Given an idea, produce:
1. A narration script, 90-160 words, written to be read aloud: a hook in the very first line, short punchy sentences, second-person/direct address where it fits naturally. Write ONLY what should be spoken aloud - no stage directions, no scene headings.
2. Exactly 8 image prompts for an AI image generator, one per rough eighth of the narration in order. Repeat a consistent setting/subject/style description across all 8 prompts so they read as one coherent visual world, not 8 unrelated images.
3. A music pick, only if a music list is given to you in the user message.

Respond with STRICT JSON only - no markdown code fences, no commentary before or after - matching exactly this shape:
{"script": "...", "image_prompts": ["...", "...", "...", "...", "...", "...", "...", "..."], "music_file_ref": "..." or null}
"""


def _build_user_prompt(idea, music_list):
    lines = [f"Idea: {idea}", "", "Target script length: 90-160 words."]
    if music_list:
        lines.append("")
        lines.append(
            "Pick ONE music_file_ref from this list if its mood fits the idea, "
            "else use null:"
        )
        for m in music_list:
            lines.append(f"- {m['file_ref']}: {m['description']}")
    else:
        lines.append("")
        lines.append("No music list is available - music_file_ref must be null.")
    return "\n".join(lines)


def validate(payload, music_list):
    """Returns a list of error strings; empty list means valid."""
    if not isinstance(payload, dict):
        return ["response is not a JSON object"]

    errors = []
    for key in ("script", "image_prompts", "music_file_ref"):
        if key not in payload:
            errors.append(f"missing key: {key}")
    if errors:
        return errors

    if not isinstance(payload["script"], str) or not payload["script"].strip():
        errors.append("script must be a non-empty string")

    prompts = payload["image_prompts"]
    if (
        not isinstance(prompts, list)
        or len(prompts) != 8
        or not all(isinstance(p, str) and p.strip() for p in prompts)
    ):
        errors.append("image_prompts must be a list of exactly 8 non-empty strings")

    ref = payload["music_file_ref"]
    valid_refs = {m["file_ref"] for m in music_list}
    if ref is not None and ref not in valid_refs:
        errors.append(f"music_file_ref {ref!r} does not match any provided file_ref")

    return errors


def _call(messages, model, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
    }
    resp = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()


def generate_script(idea, music_list=None, model=DEFAULT_MODEL, api_key=None):
    """Returns (payload, cost_usd). Raises RuntimeError after 2 failed attempts."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to reel-pipeline/.env "
            "(see .env.example)."
        )
    music_list = music_list or []

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(idea, music_list)},
    ]

    total_cost = 0.0
    errors = []
    for attempt in range(2):
        data = _call(messages, model, api_key)
        total_cost += data.get("usage", {}).get("cost", 0.0) or 0.0
        raw = data["choices"][0]["message"]["content"]

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            payload = None
            errors = [f"invalid JSON: {e}"]
        else:
            errors = validate(payload, music_list)

        if not errors:
            word_count = len(payload["script"].split())
            if not (90 <= word_count <= 160):
                print(f"warning: script word count {word_count} outside target range 90-160")
            return payload, total_cost

        if attempt == 0:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last response didn't match the required format "
                        f"({'; '.join(errors)}). Resend as strict JSON matching "
                        'exactly: {"script": "...", "image_prompts": [8 non-empty '
                        'strings], "music_file_ref": "..." or null}. No markdown '
                        "fences, no commentary."
                    ),
                }
            )

    raise RuntimeError(f"LLM failed to produce valid JSON after 2 attempts: {errors}")
