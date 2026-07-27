import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from google import genai
from google.genai import types

INTERESTING_EVENTS = {
    "fish_first_visible",
    "fish_shown_on_camera",
    # "reeling_or_casting",
    # "fish_storage_or_release",
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


def cut_clip(video_path, out_path, start, duration):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "26",
            "-c:a", "aac",
            "-b:a", "64k",
            str(out_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def wait_for_file(client, uploaded_file):
    while not uploaded_file.state or uploaded_file.state.name != "ACTIVE":
        if uploaded_file.state and uploaded_file.state.name == "FAILED":
            raise RuntimeError("Gemini file processing failed")
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
    return uploaded_file


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def analyze_clip(client, clip_path, model):
    uploaded = client.files.upload(file=str(clip_path))
    uploaded = wait_for_file(client, uploaded)

    response = client.models.generate_content(
        model=model,
        contents=[uploaded, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=SCHEMA,
            temperature=0.0,
        ),
    )

    return parse_json(response.text)


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--window-seconds", type=float, default=30)
    parser.add_argument("--stride-seconds", type=float, default=15)
    parser.add_argument("--min-score", type=int, default=75)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-candidate-duration", type=float, default=20)
    args = parser.parse_args()

    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GEMINI_API_KEY first")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output = args.output or f"{video_path.stem}_gemini_result.json"
    duration = get_duration(video_path)
    client = genai.Client()

    scored = []
    candidates = []
    fish_seen = False

    windows = []
    start = 0.0
    while start < duration:
        end = min(start + args.window_seconds, duration)
        if end - start >= 4:
            windows.append((start, end))
        start += args.stride_seconds

    if args.max_windows:
        windows = windows[:args.max_windows]

    with tempfile.TemporaryDirectory() as tmp:
        for i, (start, end) in enumerate(windows, start=1):
            print(f"Analyzing {start:.1f}s-{end:.1f}s")
            clip_path = Path(tmp) / f"window_{i:04d}.mp4"
            cut_clip(video_path, clip_path, start, end - start)

            try:
                data = analyze_clip(client, clip_path, args.model)
            except Exception as error:
                print(f"Failed: {error}")
                continue

            score = int(clamp(int(data.get("interesting_score", 0)), 0, 100))
            event_type = str(data.get("event_type", "low_interest"))

            has_fish = bool(data.get("has_fish", False))

            if has_fish:
                score = max(score, 70)
                event_type = "fish_first_visible" if not fish_seen else "fish_shown_on_camera"
                fish_seen = True
            else:
                score = min(score, 59)
                event_type = "low_interest"

            offset_start = float(data.get("start_offset", 0))
            offset_end = float(data.get("end_offset", end - start))
            offset_start = clamp(offset_start, 0, end - start)
            offset_end = clamp(offset_end, 0, end - start)
            if offset_end <= offset_start:
                offset_start, offset_end = 0, end - start

            if offset_end - offset_start > args.max_candidate_duration:
                center = (offset_start + offset_end) / 2
                half = args.max_candidate_duration / 2
                offset_start = clamp(center - half, 0, end - start)
                offset_end = clamp(center + half, 0, end - start)

                if offset_end - offset_start > args.max_candidate_duration:
                    offset_end = offset_start + args.max_candidate_duration

            item = {
                "start_sec": round(start + offset_start, 2),
                "end_sec": round(start + offset_end, 2),
                "score": score,
                "event_type": event_type,
                "reasons": [str(data.get("reason", ""))],
            }
            scored.append(item)

            if score >= args.min_score and event_type in INTERESTING_EVENTS:
                candidates.append(item.copy())

            print(f"score={score} event={event_type} reason={data.get('reason', '')}")

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

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved {output}")
    print(f"Candidates: {len(candidates)}")


if __name__ == "__main__":
    main()
