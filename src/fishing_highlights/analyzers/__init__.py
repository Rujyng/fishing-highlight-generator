from __future__ import annotations

from pathlib import Path

from ..settings import AnalysisSettings


def analyze_video(video_path: Path, settings: AnalysisSettings):
    if settings.analyzer == "gemini":
        from .gemini import analyze_video as run

        return run(video_path, settings)

    if settings.analyzer == "rekognition":
        from .rekognition import analyze_video as run

        return run(video_path, settings)

    if settings.analyzer == "twelvelabs":
        from .twelvelabs import analyze_video as run

        return run(video_path, settings)

    raise RuntimeError(f"Unknown analyzer: {settings.analyzer}")
