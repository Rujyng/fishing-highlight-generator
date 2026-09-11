from __future__ import annotations

from pathlib import Path

from ..analyzers import analyze_video as run_analyzer
from ..config import apply_analyzer_defaults
from ..settings import AnalysisSettings


def analyze_video(video_path: Path, settings: AnalysisSettings):
    apply_analyzer_defaults(settings)
    return run_analyzer(video_path, settings)
