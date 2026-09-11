from __future__ import annotations

from pathlib import Path

from ..config import default_output_path
from ..schema import write_result
from ..settings import AnalysisSettings, CaptionSettings, TrimSettings
from .analyze import analyze_video
from .captioning import caption_directory
from .trimming import create_highlights_from_result


def run_local_job(
    video_path: Path,
    analysis_settings: AnalysisSettings,
    output_path: Path | None = None,
    create_highlights: bool = False,
    include_captions: bool = False,
    trim_settings: TrimSettings | None = None,
    caption_settings: CaptionSettings | None = None,
):
    video_path = Path(video_path)
    result = analyze_video(video_path, analysis_settings)

    output_path = output_path or default_output_path(video_path, analysis_settings.analyzer)
    write_result(output_path, result)

    job_result = {
        "analysis_result": result,
        "analysis_output": str(output_path),
        "highlight_summary": None,
        "caption_output_dir": None,
    }

    if include_captions:
        create_highlights = True

    if create_highlights:
        highlight_summary = create_highlights_from_result(
            video_path=video_path,
            result=result,
            settings=trim_settings or TrimSettings(),
        )
        job_result["highlight_summary"] = highlight_summary

        if include_captions and highlight_summary["clips"]:
            caption_settings = caption_settings or CaptionSettings()
            caption_output_dir = caption_directory(
                input_dir=highlight_summary["output_dir"],
                model_name=caption_settings.model_name,
                hold_seconds=caption_settings.hold_seconds,
                margin_v=caption_settings.margin_v,
            )
            job_result["caption_output_dir"] = str(caption_output_dir)

    return job_result
