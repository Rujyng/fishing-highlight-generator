from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ..schema import build_analysis_result, rank_candidates
from ..secrets import load_secret
from ..settings import AnalysisSettings
from ..video import build_windows, clamp, get_video_duration


INTERESTING_EVENTS = {
    "fish_first_visible",
    "fish_shown_on_camera",
}

PROMPT = """
Analyze this fishing video clip.

Your job is to find ONLY close-up or clearly visible fish moments.

Mark the clip as interesting ONLY if a real fish is visually visible in the frame.

Do NOT mark the clip as interesting for:
- fishing rods
- fishing line
- casting
- reeling
- fighting the fish
- people talking
- people pointing
- people reacting
- water splashing
- boat/water/scenery
- someone saying there is a fish
- a fish that is too tiny, blurry, hidden, or uncertain

A clip is interesting ONLY when one of these is true:
- a fish is clearly visible in the frame
- a person is holding a visible fish
- a person is showing/presenting a visible fish to the camera
- a visible fish is being measured, stored, released, or handled

Return JSON only.

Choose event_type from:
- fish_shown_on_camera
- low_interest

Return:
{
  "has_fish": boolean,
  "interesting_score": integer 0-100,
  "event_type": string,
  "reason": string,
  "start_offset": number,
  "end_offset": number
}

Scoring rules:
- 0-40: No visible fish.
- 41-59: Possible fish, but too unclear, too small, blurry, hidden, or uncertain.
- 60-74: Fish is visible, but not a strong highlight.
- 75-84: Fish is clearly visible.
- 85-100: Fish is clearly shown, held, presented, measured, stored, or released.

Strict rules:
- If you are not sure a fish is visible, set has_fish=false.
- If has_fish=false, interesting_score must be below 60.
- If event_type is fish_shown_on_camera, has_fish must be true.
- Do not use fish_shown_on_camera for reeling, casting, water splash, or talking.
- Select only the tight time range where the fish is visible.
- Do not use the full clip unless the fish is visible for nearly the full clip.
- start_offset and end_offset must be in seconds relative to this clip.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "has_fish": {"type": "boolean"},
        "interesting_score": {"type": "integer"},
        "event_type": {"type": "string"},
        "reason": {"type": "string"},
        "start_offset": {"type": "number"},
        "end_offset": {"type": "number"},
    },
    "required": [
        "has_fish",
        "interesting_score",
        "event_type",
        "reason",
        "start_offset",
        "end_offset",
    ],
}


class GeminiFileProcessingError(RuntimeError):
    pass


def cut_clip(video_path: Path, out_path: Path, start: float, duration: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
            "-t",
            str(duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(out_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def wait_for_file(client, uploaded_file, timeout_seconds=120, poll_seconds=2):
    started_at = time.monotonic()

    while True:
        state = getattr(uploaded_file, "state", None)
        state_name = str(getattr(state, "name", state or "")).upper()

        if state_name == "ACTIVE":
            return uploaded_file

        if state_name == "FAILED":
            file_name = getattr(uploaded_file, "name", "unknown")
            raise GeminiFileProcessingError(
                f"Gemini file processing failed for {file_name} with state={state_name}"
            )

        if time.monotonic() - started_at > timeout_seconds:
            file_name = getattr(uploaded_file, "name", "unknown")
            raise TimeoutError(
                f"Timed out waiting for Gemini file {file_name} to become ACTIVE; last state={state_name}"
            )

        time.sleep(poll_seconds)
        uploaded_file = client.files.get(name=uploaded_file.name)


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def analyze_clip(client, clip_path: Path, model_name: str):
    from google.genai import types

    uploaded = client.files.upload(file=str(clip_path))
    uploaded = wait_for_file(client, uploaded)

    response = client.models.generate_content(
        model=model_name,
        contents=[uploaded, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
            temperature=0.0,
        ),
    )

    return parse_json(response.text)


def is_retryable_error(error):
    text = str(error).lower()
    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "resource_exhausted",
        "unavailable",
        "internal",
        "deadline",
        "timeout",
        "timed out",
        "temporarily",
        "high demand",
        "file processing failed",
    )
    return any(marker in text for marker in retryable_markers)


def analyze_clip_with_retries(client, clip_path, model_name, retries, sleep_seconds):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            return analyze_clip(client, clip_path, model_name)
        except Exception as error:
            last_error = error
            if attempt >= retries or not is_retryable_error(error):
                raise

            wait_seconds = sleep_seconds * attempt
            print(f"Temporary Gemini error on attempt {attempt}/{retries}: {error}")
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    raise last_error


def normalize_result(data, start, end, fish_seen, max_candidate_duration):
    score = int(clamp(int(data.get("interesting_score", 0)), 0, 100))
    has_fish = bool(data.get("has_fish", False))

    if has_fish:
        score = max(score, 70)
        event_type = "fish_first_visible" if not fish_seen else "fish_shown_on_camera"
    else:
        score = min(score, 59)
        event_type = "low_interest"

    offset_start = float(data.get("start_offset", 0))
    offset_end = float(data.get("end_offset", end - start))
    offset_start = clamp(offset_start, 0, end - start)
    offset_end = clamp(offset_end, 0, end - start)
    if offset_end <= offset_start:
        offset_start, offset_end = 0, end - start

    if offset_end - offset_start > max_candidate_duration:
        center = (offset_start + offset_end) / 2
        half = max_candidate_duration / 2
        offset_start = clamp(center - half, 0, end - start)
        offset_end = clamp(center + half, 0, end - start)

        if offset_end - offset_start > max_candidate_duration:
            offset_end = offset_start + max_candidate_duration

    return {
        "start_sec": round(start + offset_start, 2),
        "end_sec": round(start + offset_end, 2),
        "score": score,
        "event_type": event_type,
        "reasons": [str(data.get("reason", ""))],
    }


def analyze_video(video_path: Path, settings: AnalysisSettings):
    from google import genai

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or load_secret("geminiAPI.key")
    )
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or add secrets/geminiAPI.key first")

    os.environ.setdefault("GEMINI_API_KEY", api_key)

    duration = get_video_duration(video_path)
    client = genai.Client()
    windows = build_windows(duration, settings.window_seconds, settings.stride_seconds)
    if settings.max_windows:
        windows = windows[:settings.max_windows]

    scored = []
    candidates = []
    fish_seen = False

    with tempfile.TemporaryDirectory() as tmp:
        for index, (start, end) in enumerate(windows, start=1):
            print(f"Analyzing {start:.1f}s-{end:.1f}s")
            clip_path = Path(tmp) / f"window_{index:04d}.mp4"
            cut_clip(video_path, clip_path, start, end - start)

            try:
                data = analyze_clip_with_retries(
                    client=client,
                    clip_path=clip_path,
                    model_name=settings.model_name,
                    retries=settings.retries,
                    sleep_seconds=settings.sleep_seconds,
                )
            except Exception as error:
                print(f"Failed: {error}")
                continue

            item = normalize_result(
                data=data,
                start=start,
                end=end,
                fish_seen=fish_seen,
                max_candidate_duration=settings.max_candidate_duration,
            )

            if item["event_type"] in INTERESTING_EVENTS:
                fish_seen = True

            scored.append(item)

            if item["score"] >= settings.min_score and item["event_type"] in INTERESTING_EVENTS:
                candidates.append(item.copy())

            print(f"score={item['score']} event={item['event_type']} reason={item['reasons'][0]}")

    rank_candidates(candidates)
    if windows and not scored:
        raise RuntimeError("Gemini analysis produced no scored windows; all requested windows failed.")

    return build_analysis_result(video_path, candidates, scored)
