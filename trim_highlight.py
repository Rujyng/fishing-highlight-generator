import argparse
import json
import subprocess
from pathlib import Path


INTERESTING_EVENTS = {
    "fish_first_visible",
    "fish_shown_on_camera",
    "reeling_or_casting",
    "fish_storage_or_release",
}


def get_video_duration(video_path):
    # ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 fishing_video.mp4
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error", # only show errors, not extra logs
            "-show_entries", "format=duration", # return only the duration
            "-of", "default=noprint_wrappers=1:nokey=1", # format to only print the number
            str(video_path),
        ],
        capture_output=True, # stores result to variable 
        text=True, # string instead of bytes
        check=True, # python throws error if ffprobe fails
    )

    return float(result.stdout.strip())


def load_interesting_windows(json_path, min_score):
    '''
    get interesting windows (>= min_score)
    '''
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    windows = []
    for item in data.get("candidate_highlights", []):
        # if event is interesting with score >= min_score
        if (item.get("event_type") in INTERESTING_EVENTS and item.get("score", 0) >= min_score):
            windows.append({
                "start": float(item["start_sec"]),
                "end": float(item["end_sec"]),
                "score": item.get("score", 0),
                "event_type": item.get("event_type"),
            })
    return sorted(windows, key=lambda x: x["start"])


def merge_windows(windows, video_duration, pad=10, merge_gap=80, max_duration=90):
    '''
    add padding around each highlight (10 seconds)
    merge overlapping segments
    stitch nearby segments into one clip if possible
        (stitch happens when the segments have lower gap than merge_gap, 
            and combined duration does not exceed max_duration)
    '''
    if not windows:
        return []

    # Step 1: apply padding to each window
    padded = []

    for window in windows:
        start = max(0, window["start"] - pad)
        end = min(video_duration, window["end"] + pad)

        padded.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "score": window["score"],
            "event_types": {window["event_type"]},
        })

    padded.sort(key=lambda x: x["start"])

    # Step 2: merge overlapping padded segments
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

    # Step 3: group nearby segments into one stitched highlight
    clips = []
    current_segments = [no_overlap[0]]
    current_score = no_overlap[0]["score"]
    current_event_types = set(no_overlap[0]["event_types"])

    def total_duration(segments):
        return sum(segment["duration"] for segment in segments)

    for segment in no_overlap[1:]:
        gap = segment["start"] - current_segments[-1]["end"]
        new_duration = total_duration(current_segments) + segment["duration"]

        # if gap between 2 highlights are too small and combining them doesn't make
        #   the duration too long, then combine them
        if gap <= merge_gap and new_duration <= max_duration:
            current_segments.append(segment)
            current_score += segment["score"]
            current_event_types.update(segment["event_types"])
        else: # if not then save the highlight and move on
            clips.append({
                "start": round(current_segments[0]["start"], 2),
                "end": round(current_segments[-1]["end"], 2),
                "segments": [
                    {
                        "start": round(s["start"], 2),
                        "end": round(s["end"], 2),
                        "duration": round(s["duration"], 2),
                    }
                    for s in current_segments
                ],
                "duration": round(total_duration(current_segments), 2),
                "score": current_score,
                "event_types": sorted(current_event_types),
            })

            current_segments = [segment]
            current_score = segment["score"]
            current_event_types = set(segment["event_types"])

    clips.append({
        "start": round(current_segments[0]["start"], 2),
        "end": round(current_segments[-1]["end"], 2),
        "segments": [
            {
                "start": round(s["start"], 2),
                "end": round(s["end"], 2),
                "duration": round(s["duration"], 2),
            }
            for s in current_segments
        ],
        "duration": round(total_duration(current_segments), 2),
        "score": current_score,
        "event_types": sorted(current_event_types),
    })

    return clips


def trim_video(video_path, output_path, segments):
    temp_files = []

    for i, segment in enumerate(segments, start=1):
        temp_path = output_path.parent / f"temp_segment_{i:02d}.mp4"
        temp_files.append(temp_path)

        # ffmpeg -y -ss 100 -i live_fishing.mp4 -t 25 -c copy temp_segment_01.mp4
        subprocess.run(
            [
                "ffmpeg",
                "-y",                           # overwrite output file if it already exists
                "-ss", str(segment["start"]),   # start cutting at __ seconds
                "-i", str(video_path),          # input video
                "-t", str(segment["duration"]), # duration
                "-c", "copy",                   # copy original video/audio
                str(temp_path),                 # output temp file
            ],
            check=True,
        )

    concat_file = output_path.parent / "concat_list.txt"

    with open(concat_file, "w", encoding="utf-8") as file:
        for temp_file in temp_files:
            file.write(f"file '{temp_file.name}'\n")

    # ffmpeg -y -f concat -safe 0 -i concat_list.txt -c copy highlight_01.mp4
    subprocess.run(
        [
            "ffmpeg",
            "-y",                           # overwrite output file if it already exists
            "-f", "concat",                 # use FFmpeg concat mode
            "-safe", "0",                   # allow file paths from the concat list
            "-i", str(concat_file.name),    # input list of clips
            "-c", "copy",                   # stitch without re-encoding
            str(output_path.name),          # final output video
        ],
        cwd=output_path.parent,
        check=True,
    )

    # Deletes the temp segment files after the final video is created
    for temp_file in temp_files:
        temp_file.unlink()

    # Deletes concat_list.txt
    concat_file.unlink()

def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--min-score", type=float, default=60)
    parser.add_argument("--pad", type=float, default=10)
    parser.add_argument("--merge-gap", type=float, default=60)
    parser.add_argument("--max-duration", type=float, default=90)

    args = parser.parse_args()

    json_path = Path(args.json)
    video_path = Path(args.video)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir = Path("highlights") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # get video duration in seconds
    video_duration = get_video_duration(video_path)
    # get interesting windows
    windows = load_interesting_windows(json_path, args.min_score)

    if not windows:
        print("No interesting windows found.")
        return

    # create highlights
    clips = merge_windows(
        windows,
        video_duration=video_duration,
        pad=args.pad,
        merge_gap=args.merge_gap,
        max_duration=args.max_duration,
    )

    for i, clip in enumerate(clips, start=1):
        event = "_and_".join(clip["event_types"])

        output_path = output_dir / (
            f"highlight_{i:02d}_"
            f"{event}_"
            f"{int(clip['start'])}s_to_{int(clip['end'])}s.mp4"
        ) # name of the highlight file

        print(f"Creating {output_path}")

        trim_video(
            video_path,                # original video
            output_path,               # path to put the highlight in
            segments=clip["segments"], # exact video parts to cut and stitch
        )

        clip["output_file"] = str(output_path)

    summary_path = output_dir / "highlight_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump({"clips": clips}, file, indent=2)

    print(f"\nCreated {len(clips)} clip(s).")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()