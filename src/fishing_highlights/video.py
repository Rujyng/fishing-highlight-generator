from __future__ import annotations

import subprocess
from pathlib import Path


def clamp(value, low, high):
    return max(low, min(high, value))


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def build_windows(video_duration: float, window_seconds: float, stride_seconds: float):
    windows = []
    start = 0.0

    while start < video_duration:
        end = min(start + window_seconds, video_duration)
        if end - start >= 4:
            windows.append((round(start, 2), round(end, 2)))
        start += stride_seconds

    return windows
