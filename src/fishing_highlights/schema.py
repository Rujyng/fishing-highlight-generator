from __future__ import annotations

import json
from pathlib import Path
from typing import Any


INTERESTING_EVENTS = {
    "fish_first_visible",
    "fish_shown_on_camera",
    "reeling_or_casting",
    "fish_storage_or_release",
}


def rank_candidates(candidates: list[dict[str, Any]]) -> None:
    candidates.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        item["duration"] = round(item["end_sec"] - item["start_sec"], 2)


def build_analysis_result(
    video_path: Path,
    candidate_highlights: list[dict[str, Any]],
    all_scored_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "candidate_highlights": candidate_highlights,
        "all_scored_windows": all_scored_windows,
    }


def write_result(output_path: Path, result: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
