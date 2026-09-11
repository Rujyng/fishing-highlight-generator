import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fishing_highlights.config import default_output_path, normalize_analyzer
from fishing_highlights.cli import parse_args
from fishing_highlights.analyzers.gemini import is_retryable_error as is_retryable_gemini_error
from fishing_highlights.pipeline.trimming import load_interesting_windows_from_result, merge_windows
from fishing_highlights.schema import rank_candidates
from fishing_highlights.video import build_windows


class CoreBehaviorTests(unittest.TestCase):
    def test_build_windows_supports_overlap(self):
        self.assertEqual(
            build_windows(video_duration=45, window_seconds=20, stride_seconds=10),
            [(0.0, 20.0), (10.0, 30.0), (20.0, 40.0), (30.0, 45), (40.0, 45)],
        )

    def test_short_final_window_is_skipped(self):
        self.assertEqual(
            build_windows(video_duration=43, window_seconds=20, stride_seconds=10),
            [(0.0, 20.0), (10.0, 30.0), (20.0, 40.0), (30.0, 43)],
        )

    def test_candidate_ranking_adds_rank_and_duration(self):
        candidates = [
            {"start_sec": 10, "end_sec": 20, "score": 80},
            {"start_sec": 30, "end_sec": 34, "score": 90},
        ]

        rank_candidates(candidates)

        self.assertEqual(candidates[0]["score"], 90)
        self.assertEqual(candidates[0]["rank"], 1)
        self.assertEqual(candidates[0]["duration"], 4)

    def test_trim_loader_accepts_shared_result_contract(self):
        result = {
            "candidate_highlights": [
                {
                    "start_sec": 1,
                    "end_sec": 6,
                    "score": 75,
                    "event_type": "fish_shown_on_camera",
                },
                {
                    "start_sec": 7,
                    "end_sec": 8,
                    "score": 99,
                    "event_type": "low_interest",
                },
            ]
        }

        windows = load_interesting_windows_from_result(result, min_score=60)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start"], 1)

    def test_merge_windows_combines_overlapping_segments(self):
        clips = merge_windows(
            [
                {"start": 10, "end": 20, "score": 80, "event_type": "fish_shown_on_camera"},
                {"start": 18, "end": 25, "score": 90, "event_type": "fish_shown_on_camera"},
            ],
            video_duration=60,
            pad=0,
            merge_gap=10,
            max_duration=90,
        )

        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["start"], 10)
        self.assertEqual(clips[0]["end"], 25)

    def test_analyzer_aliases_and_output_paths(self):
        self.assertEqual(normalize_analyzer("rekog"), "rekognition")
        self.assertEqual(normalize_analyzer("pegasus"), "twelvelabs")
        self.assertEqual(
            default_output_path(Path("vids/live_fishing.mp4"), "gemini"),
            Path("json_results/live_fishing_gemini_result.json"),
        )

    def test_model_argument_preserves_legacy_wrapper_behavior(self):
        unified_args = parse_args(["--video", "video.mp4", "--model", "gemini"])
        gemini_args = parse_args(
            ["--video", "video.mp4", "--model", "gemini-3.1-flash-lite"],
            default_analyzer="gemini",
        )

        self.assertEqual(unified_args.analyzer, "gemini")
        self.assertEqual(gemini_args.analyzer, "gemini")
        self.assertEqual(gemini_args.model_name, "gemini-3.1-flash-lite")

    def test_gemini_file_processing_failure_is_retryable(self):
        self.assertTrue(is_retryable_gemini_error(RuntimeError("Gemini file processing failed")))


if __name__ == "__main__":
    unittest.main()
