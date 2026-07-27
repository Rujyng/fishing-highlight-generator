import argparse
import json
import os
import subprocess
import time
from pathlib import Path


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


JSON_SCHEMA = {
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


def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def get_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def upload_asset(client, video_path, video_url=None):
    if video_url:
        asset = client.assets.create(
            method="url",
            url=video_url,
            filename=video_path.name,
        )
    else:
        with open(video_path, "rb") as file:
            asset = client.assets.create(
                method="direct",
                file=file,
            )

    asset_id = get_attr(asset, "id") or get_attr(asset, "_id")
    if not asset_id:
        raise RuntimeError(f"Could not get TwelveLabs asset id: {asset}")

    while True:
        asset = client.assets.retrieve(asset_id)
        status = str(get_attr(asset, "status", "")).lower()

        if status == "ready":
            return asset_id

        if status == "failed":
            raise RuntimeError("TwelveLabs asset processing failed")

        print("Waiting for TwelveLabs asset...")
        time.sleep(5)


def build_windows(video_duration, window_seconds, stride_seconds):
    windows = []
    start = 0.0

    while start < video_duration:
        end = min(start + window_seconds, video_duration)

        if end - start >= 4:
            windows.append((round(start, 2), round(end, 2)))

        start += stride_seconds

    return windows


def parse_json(value):
    if isinstance(value, dict):
        return value

    text = str(value).strip()

    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        )

    return json.loads(text)


def analyze_window(client, asset_id, start, end, model_name, temperature):
    from twelvelabs.types import AnalyzePromptV2, SyncResponseFormat, VideoContext_AssetId

    prompt = PROMPT + f"\nAnalyze only {start:.2f}s to {end:.2f}s of the original video."

    response = client.analyze(
        model_name=model_name,
        video=VideoContext_AssetId(asset_id=asset_id),
        prompt_v_2=AnalyzePromptV2(input_text=prompt),
        response_format=SyncResponseFormat(
            type="json_schema",
            json_schema=JSON_SCHEMA,
        ),
        temperature=temperature,
        max_tokens=512,
        start_time=start,
        end_time=end,
    )

    return parse_json(get_attr(response, "data"))


def clamp(value, low, high):
    return max(low, min(high, value))


def to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "yes", "1"}


def normalize_result(data, window_start, window_end, fish_seen, max_candidate_duration):
    window_duration = window_end - window_start

    has_fish = to_bool(data.get("has_fish", False))
    score = int(clamp(int(data.get("interesting_score", 0)), 0, 100))

    if has_fish:
        score = max(score, 75)
        event_type = "fish_first_visible" if not fish_seen else "fish_shown_on_camera"
    else:
        score = min(score, 59)
        event_type = "low_interest"

    offset_start = float(data.get("start_offset", 0))
    offset_end = float(data.get("end_offset", window_duration))

    offset_start = clamp(offset_start, 0, window_duration)
    offset_end = clamp(offset_end, 0, window_duration)

    if offset_end <= offset_start:
        offset_start = 0
        offset_end = window_duration

    if offset_end - offset_start > max_candidate_duration:
        center = (offset_start + offset_end) / 2
        half = max_candidate_duration / 2
        offset_start = clamp(center - half, 0, window_duration)
        offset_end = clamp(center + half, 0, window_duration)

        if offset_end - offset_start > max_candidate_duration:
            offset_end = offset_start + max_candidate_duration

    return {
        "start_sec": round(window_start + offset_start, 2),
        "end_sec": round(window_start + offset_end, 2),
        "score": score,
        "event_type": event_type,
        "reasons": [str(data.get("reason", "")).strip()],
        "ai_analysis": data,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--video-url", default=None)
    parser.add_argument("--model-name", default="pegasus1.5")
    parser.add_argument("--window-seconds", type=float, default=30)
    parser.add_argument("--stride-seconds", type=float, default=15)
    parser.add_argument("--min-score", type=int, default=75)
    parser.add_argument("--max-candidate-duration", type=float, default=20)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    api_key = os.getenv("TWELVELABS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TWELVELABS_API_KEY")

    from twelvelabs import TwelveLabs

    client = TwelveLabs(api_key=api_key)

    video_duration = get_duration(video_path)
    asset_id = args.asset_id or upload_asset(client, video_path, args.video_url)
    output = args.output or f"{video_path.stem}_pegasus_fish_highlights.json"

    windows = build_windows(
        video_duration=video_duration,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    if args.max_windows:
        windows = windows[:args.max_windows]

    scored = []
    candidates = []
    fish_seen = False

    for start, end in windows:
        print(f"Analyzing {start}s-{end}s")

        try:
            data = analyze_window(
                client=client,
                asset_id=asset_id,
                start=start,
                end=end,
                model_name=args.model_name,
                temperature=args.temperature,
            )
        except Exception as error:
            if "too_many_requests" in str(error) or "status_code: 429" in str(error):
                raise RuntimeError(f"TwelveLabs rate limit hit. Stop and retry later.\n{error}")

            print(f"Failed {start}s-{end}s: {error}")
            continue

        item = normalize_result(
            data=data,
            window_start=start,
            window_end=end,
            fish_seen=fish_seen,
            max_candidate_duration=args.max_candidate_duration,
        )

        if item["ai_analysis"].get("has_fish", False):
            fish_seen = True

        scored.append(item)

        if item["score"] >= args.min_score and item["event_type"] in INTERESTING_EVENTS:
            candidates.append(item.copy())

        print(
            f"score={item['score']} "
            f"event={item['event_type']} "
            f"selected={item['start_sec']}s-{item['end_sec']}s "
            f"reason={item['reasons'][0]}"
        )

        time.sleep(args.sleep_seconds)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        item["duration"] = round(item["end_sec"] - item["start_sec"], 2)

    result = {
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "candidate_highlights": candidates,
        "all_scored_windows": scored,
    }

    with open(output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Saved {output}")
    print(f"Candidates: {len(candidates)}")


if __name__ == "__main__":
    main()