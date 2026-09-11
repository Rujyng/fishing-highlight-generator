from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ..config import JSON_RESULTS_DIR, LOCAL_HIGHLIGHTS_DIR
from ..schema import INTERESTING_EVENTS
from ..settings import TrimSettings
from ..video import get_video_duration


def load_interesting_windows_from_result(data, min_score):
    windows = []
    for item in data.get("candidate_highlights", []):
        if item.get("event_type") in INTERESTING_EVENTS and item.get("score", 0) >= min_score:
            windows.append(
                {
                    "start": float(item["start_sec"]),
                    "end": float(item["end_sec"]),
                    "score": item.get("score", 0),
                    "event_type": item.get("event_type"),
                }
            )
    return sorted(windows, key=lambda item: item["start"])


def load_interesting_windows(json_path, min_score):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return load_interesting_windows_from_result(data, min_score)


def resolve_json_path(json_arg):
    json_path = Path(json_arg)
    if json_path.exists():
        return json_path

    fallback = JSON_RESULTS_DIR / json_path.name
    if fallback.exists():
        return fallback

    return json_path


def merge_windows(windows, video_duration, pad=10, merge_gap=80, max_duration=90):
    if not windows:
        return []

    padded = []
    for window in windows:
        start = max(0, window["start"] - pad)
        end = min(video_duration, window["end"] + pad)

        padded.append(
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "score": window["score"],
                "event_types": {window["event_type"]},
            }
        )

    padded.sort(key=lambda item: item["start"])

    no_overlap = []
    current = padded[0]

    for segment in padded[1:]:
        if segment["start"] <= current["end"]:
            current["end"] = max(current["end"], segment["end"])
            current["duration"] = current["end"] - current["start"]
            current["score"] += segment["score"]
            current["event_types"].update(segment["event_types"])
        else:
            no_overlap.append(current)
            current = segment

    no_overlap.append(current)

    clips = []
    current_segments = [no_overlap[0]]
    current_score = no_overlap[0]["score"]
    current_event_types = set(no_overlap[0]["event_types"])

    def total_duration(segments):
        return sum(segment["duration"] for segment in segments)

    def append_clip():
        clips.append(
            {
                "start": round(current_segments[0]["start"], 2),
                "end": round(current_segments[-1]["end"], 2),
                "segments": [
                    {
                        "start": round(segment["start"], 2),
                        "end": round(segment["end"], 2),
                        "duration": round(segment["duration"], 2),
                    }
                    for segment in current_segments
                ],
                "duration": round(total_duration(current_segments), 2),
                "score": current_score,
                "event_types": sorted(current_event_types),
            }
        )

    for segment in no_overlap[1:]:
        gap = segment["start"] - current_segments[-1]["end"]
        new_duration = total_duration(current_segments) + segment["duration"]

        if gap <= merge_gap and new_duration <= max_duration:
            current_segments.append(segment)
            current_score += segment["score"]
            current_event_types.update(segment["event_types"])
        else:
            append_clip()
            current_segments = [segment]
            current_score = segment["score"]
            current_event_types = set(segment["event_types"])

    append_clip()
    return clips


def trim_video(video_path, output_path, segments):
    temp_files = []

    for index, segment in enumerate(segments, start=1):
        temp_path = output_path.parent / f"temp_segment_{index:02d}.mp4"
        temp_files.append(temp_path)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(segment["start"]),
                "-i",
                str(video_path),
                "-t",
                str(segment["duration"]),
                "-c",
                "copy",
                str(temp_path),
            ],
            check=True,
        )

    concat_file = output_path.parent / "concat_list.txt"

    with open(concat_file, "w", encoding="utf-8") as file:
        for temp_file in temp_files:
            file.write(f"file '{temp_file.name}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file.name),
            "-c",
            "copy",
            str(output_path.name),
        ],
        cwd=output_path.parent,
        check=True,
    )

    for temp_file in temp_files:
        temp_file.unlink()

    concat_file.unlink()


def create_highlights_from_result(video_path, result, output_dir=None, settings=None):
    settings = settings or TrimSettings()
    video_path = Path(video_path)
    output_dir = Path(output_dir) if output_dir else LOCAL_HIGHLIGHTS_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    video_duration = get_video_duration(video_path)
    windows = load_interesting_windows_from_result(result, settings.min_score)

    if not windows:
        print("No interesting windows found.")
        return {"clips": [], "output_dir": str(output_dir)}

    clips = merge_windows(
        windows,
        video_duration=video_duration,
        pad=settings.pad,
        merge_gap=settings.merge_gap,
        max_duration=settings.max_duration,
    )

    for index, clip in enumerate(clips, start=1):
        event = "_and_".join(clip["event_types"])
        output_path = output_dir / (
            f"highlight_{index:02d}_"
            f"{event}_"
            f"{int(clip['start'])}s_to_{int(clip['end'])}s.mp4"
        )

        print(f"Creating {output_path}")
        trim_video(video_path, output_path, segments=clip["segments"])
        clip["output_file"] = str(output_path)

    summary = {"clips": clips, "output_dir": str(output_dir)}
    summary_path = output_dir / "highlight_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump({"clips": clips}, file, indent=2)

    print(f"\nCreated {len(clips)} clip(s).")
    print(f"Saved to: {output_dir}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--min-score", type=float, default=60)
    parser.add_argument("--pad", type=float, default=10)
    parser.add_argument("--merge-gap", type=float, default=60)
    parser.add_argument("--max-duration", type=float, default=90)

    args = parser.parse_args(argv)

    json_path = resolve_json_path(args.json)
    video_path = Path(args.video)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with open(json_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    create_highlights_from_result(
        video_path=video_path,
        result=result,
        settings=TrimSettings(
            min_score=args.min_score,
            pad=args.pad,
            merge_gap=args.merge_gap,
            max_duration=args.max_duration,
        ),
    )
