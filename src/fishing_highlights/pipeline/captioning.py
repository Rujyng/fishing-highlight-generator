from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def srt_time(seconds):
    ms = int((seconds % 1) * 1000)
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extend_caption_times(captions, hold_seconds):
    extended = []

    for index, (start, end, text) in enumerate(captions):
        new_end = end + hold_seconds

        if index + 1 < len(captions):
            next_start = captions[index + 1][0]
            new_end = min(new_end, next_start - 0.05)

        if new_end <= start:
            new_end = end

        extended.append((start, new_end, text))

    return extended


def write_srt(segments, srt_path, hold_seconds=0.5, max_words=7, max_duration=2.5):
    captions = []

    for segment in segments:
        if segment.get("no_speech_prob", 0) > 0.75:
            continue

        words = segment.get("words", [])

        if not words:
            text = segment["text"].strip()
            if text:
                captions.append((segment["start"], segment["end"], text))
            continue

        current_words = []
        start_time = None

        for word in words:
            text = word.get("word", "").strip()

            if not text:
                continue

            if start_time is None:
                start_time = word["start"]

            current_words.append(word)

            duration = word["end"] - start_time
            should_end_caption = (
                len(current_words) >= max_words
                or duration >= max_duration
                or text.endswith((".", "?", "!"))
            )

            if should_end_caption:
                caption_text = " ".join(item["word"].strip() for item in current_words)
                captions.append((start_time, word["end"], caption_text))
                current_words = []
                start_time = None

        if current_words:
            captions.append(
                (
                    start_time,
                    current_words[-1]["end"],
                    " ".join(word["word"].strip() for word in current_words),
                )
            )

    if not captions:
        return False

    captions = extend_caption_times(captions, hold_seconds)

    with open(srt_path, "w", encoding="utf-8") as file:
        for index, (start, end, text) in enumerate(captions, start=1):
            file.write(f"{index}\n")
            file.write(f"{srt_time(start)} --> {srt_time(end)}\n")
            file.write(f"{text}\n\n")

    return True


def burn_captions(video_path, srt_path, output_path, margin_v):
    output_rel = f"captioned/{output_path.name}"

    subtitle_filter = (
        f"subtitles=filename={srt_path.name}:"
        f"force_style=Alignment=2\\,MarginV={margin_v}\\,Fontsize=22"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path.name,
        "-vf",
        subtitle_filter,
        "-c:a",
        "copy",
        output_rel,
    ]

    subprocess.run(command, cwd=video_path.parent, check=True)


def caption_directory(input_dir, model_name="base", hold_seconds=0.5, margin_v=30):
    import whisper

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder not found: {input_dir}")

    output_dir = input_dir / "captioned"
    output_dir.mkdir(exist_ok=True)

    model = whisper.load_model(model_name)
    videos = sorted(input_dir.glob("*.mp4"))

    if not videos:
        print("No .mp4 highlight files found.")
        return output_dir

    for video_path in videos:
        print(f"Transcribing: {video_path.name}")

        result = model.transcribe(str(video_path), word_timestamps=True)

        srt_path = video_path.with_suffix(".srt")
        has_speech = write_srt(
            result["segments"],
            srt_path,
            hold_seconds=hold_seconds,
        )

        if not has_speech:
            print(f"No clear speech found: {video_path.name}")
            continue

        output_path = output_dir / f"{video_path.stem}_captioned.mp4"

        print(f"Creating captioned video: {output_path}")

        burn_captions(
            video_path=video_path,
            srt_path=srt_path,
            output_path=output_path,
            margin_v=margin_v,
        )

        srt_path.unlink()

    print(f"\nDone. Captioned videos saved in: {output_dir}")
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--model", default="base")
    parser.add_argument("--hold", type=float, default=0.5)
    parser.add_argument("--margin-v", type=int, default=30)

    args = parser.parse_args(argv)
    caption_directory(
        input_dir=args.input_dir,
        model_name=args.model,
        hold_seconds=args.hold,
        margin_v=args.margin_v,
    )
