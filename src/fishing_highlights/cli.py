from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import DEFAULT_PRODUCT_ANALYZER, default_output_path, normalize_analyzer
from .schema import write_result
from .settings import AnalysisSettings, CaptionSettings, TrimSettings
from .pipeline.analyze import analyze_video
from .pipeline.captioning import caption_directory
from .pipeline.trimming import create_highlights_from_result


ANALYZER_CHOICES = {"gemini", "rekognition", "rekog", "twelvelabs", "pegasus"}


def parse_args(argv=None, default_analyzer=None):
    parser = argparse.ArgumentParser(
        description="Analyze fishing videos and prepare highlight candidates."
    )
    parser.add_argument("--analyzer", choices=sorted(ANALYZER_CHOICES), default="gemini")
    parser.add_argument("--model", default=None)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--min-score", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--max-candidate-duration", type=float, default=20)

    parser.add_argument(
        "--bucket",
        default=os.getenv("REKOGNITION_BUCKET"),
        help="S3 bucket for Rekognition input video upload.",
    )
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-2"))
    parser.add_argument("--min-confidence", type=float, default=45)
    parser.add_argument("--pad-seconds", type=int, default=5)

    parser.add_argument("--asset-id", default=None, help="Existing TwelveLabs asset id.")
    parser.add_argument("--video-url", default=None, help="Video URL for TwelveLabs asset creation.")
    parser.add_argument("--model-name", default=None, help="Provider model name.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=1)
    parser.add_argument("--retries", type=int, default=3)

    parser.add_argument("--create-highlights", action="store_true")
    parser.add_argument("--include-captions", action="store_true")
    parser.add_argument("--trim-min-score", type=float, default=60)
    parser.add_argument("--trim-pad", type=float, default=10)
    parser.add_argument("--trim-merge-gap", type=float, default=60)
    parser.add_argument("--trim-max-duration", type=float, default=90)
    parser.add_argument("--caption-model", default="base")
    parser.add_argument("--caption-hold", type=float, default=0.5)
    parser.add_argument("--caption-margin-v", type=int, default=30)

    args = parser.parse_args(argv)

    analyzer = args.analyzer or default_analyzer
    if not analyzer and args.model in ANALYZER_CHOICES:
        analyzer = args.model
    if not analyzer:
        analyzer = DEFAULT_PRODUCT_ANALYZER

    if default_analyzer and args.model and args.model not in ANALYZER_CHOICES and not args.model_name:
        args.model_name = args.model

    args.analyzer = normalize_analyzer(analyzer)
    return args


def build_analysis_settings(args):
    return AnalysisSettings(
        analyzer=args.analyzer,
        model_name=args.model_name,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_score=args.min_score,
        max_windows=args.max_windows,
        max_candidate_duration=args.max_candidate_duration,
        bucket=args.bucket,
        region=args.region,
        min_confidence=args.min_confidence,
        pad_seconds=args.pad_seconds,
        asset_id=args.asset_id,
        video_url=args.video_url,
        temperature=args.temperature,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
    )


def print_top_candidates(candidates, limit=10):
    if not candidates:
        return

    print("\nTop candidates:")
    for candidate in candidates[:limit]:
        print(
            f"Rank {candidate.get('rank')}: "
            f"{candidate.get('start_sec')}s - {candidate.get('end_sec')}s | "
            f"score={candidate.get('score')} | "
            f"{candidate.get('event_type')}"
        )


def main(argv=None, default_analyzer=None):
    args = parse_args(argv=argv, default_analyzer=default_analyzer)
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    settings = build_analysis_settings(args)
    result = analyze_video(video_path, settings)

    output_path = Path(args.output) if args.output else default_output_path(video_path, settings.analyzer)
    write_result(output_path, result)

    print(f"Saved {output_path}")
    print(f"Candidates: {len(result.get('candidate_highlights', []))}")
    print_top_candidates(result["candidate_highlights"])

    if args.include_captions:
        args.create_highlights = True

    if args.create_highlights:
        highlight_summary = create_highlights_from_result(
            video_path=video_path,
            result=result,
            settings=TrimSettings(
                min_score=args.trim_min_score,
                pad=args.trim_pad,
                merge_gap=args.trim_merge_gap,
                max_duration=args.trim_max_duration,
            ),
        )

        if args.include_captions and highlight_summary["clips"]:
            caption_directory(
                input_dir=highlight_summary["output_dir"],
                model_name=args.caption_model,
                hold_seconds=args.caption_hold,
                margin_v=args.caption_margin_v,
            )
