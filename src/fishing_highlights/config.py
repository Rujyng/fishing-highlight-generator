from __future__ import annotations

import os
from pathlib import Path

from .settings import AnalysisSettings


JSON_RESULTS_DIR = Path("json_results")
SECRETS_DIR = Path("secrets")
LOCAL_HIGHLIGHTS_DIR = Path("highlights")

DEFAULT_PRODUCT_ANALYZER = os.getenv("FISHING_HIGHLIGHTS_ANALYZER", "gemini")

ANALYZER_ALIASES = {
    "rekog": "rekognition",
    "pegasus": "twelvelabs",
}

ANALYZER_OUTPUT_SUFFIXES = {
    "gemini": "gemini_result",
    "rekognition": "rekognition_result",
    "twelvelabs": "pegasus_fish_highlights",
}

ANALYZER_MODEL_DEFAULTS = {
    "gemini": "gemini-3.1-flash-lite",
    "twelvelabs": "pegasus1.5",
}


def normalize_analyzer(value: str | None) -> str:
    analyzer = value or DEFAULT_PRODUCT_ANALYZER
    return ANALYZER_ALIASES.get(analyzer, analyzer)


def default_output_path(video_path: Path, analyzer: str) -> Path:
    normalized = normalize_analyzer(analyzer)
    suffix = ANALYZER_OUTPUT_SUFFIXES[normalized]
    return JSON_RESULTS_DIR / f"{video_path.stem}_{suffix}.json"


def apply_analyzer_defaults(settings: AnalysisSettings) -> AnalysisSettings:
    settings.analyzer = normalize_analyzer(settings.analyzer)

    if settings.analyzer == "gemini":
        settings.window_seconds = 30 if settings.window_seconds is None else settings.window_seconds
        settings.stride_seconds = 15 if settings.stride_seconds is None else settings.stride_seconds
        settings.min_score = 75 if settings.min_score is None else settings.min_score
        settings.model_name = settings.model_name or ANALYZER_MODEL_DEFAULTS["gemini"]
        return settings

    if settings.analyzer == "rekognition":
        settings.window_seconds = 5 if settings.window_seconds is None else settings.window_seconds
        settings.stride_seconds = (
            settings.window_seconds if settings.stride_seconds is None else settings.stride_seconds
        )
        settings.min_score = 65 if settings.min_score is None else settings.min_score
        return settings

    if settings.analyzer == "twelvelabs":
        settings.window_seconds = 30 if settings.window_seconds is None else settings.window_seconds
        settings.stride_seconds = 15 if settings.stride_seconds is None else settings.stride_seconds
        settings.min_score = 75 if settings.min_score is None else settings.min_score
        settings.model_name = settings.model_name or ANALYZER_MODEL_DEFAULTS["twelvelabs"]
        return settings

    raise RuntimeError(f"Unknown analyzer: {settings.analyzer}")
