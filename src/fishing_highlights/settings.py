from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnalysisSettings:
    analyzer: str
    model_name: str | None = None
    window_seconds: float | None = None
    stride_seconds: float | None = None
    min_score: int | None = None
    max_windows: int | None = None
    max_candidate_duration: float = 20
    bucket: str | None = None
    region: str = "us-east-2"
    min_confidence: float = 45
    pad_seconds: int = 5
    asset_id: str | None = None
    video_url: str | None = None
    temperature: float = 0
    sleep_seconds: float = 1
    retries: int = 3


@dataclass
class TrimSettings:
    min_score: float = 60
    pad: float = 10
    merge_gap: float = 60
    max_duration: float = 90


@dataclass
class CaptionSettings:
    model_name: str = "base"
    hold_seconds: float = 0.5
    margin_v: int = 30
