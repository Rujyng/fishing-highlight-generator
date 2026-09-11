from __future__ import annotations

import time
import uuid
from collections import defaultdict
from pathlib import Path

from ..schema import build_analysis_result
from ..settings import AnalysisSettings


FISH_TERMS = {"fish", "bass", "trout", "salmon", "tuna"}
PERSON_TERMS = {"person", "human", "people", "face"}
CONTEXT_TERMS = {"water", "sea", "ocean", "lake", "river", "boat", "outdoors"}
FISHING_TERMS = {"fishing", "fishing rod", "rod", "bait", "hook", "reel"}

FISH_THRESHOLD = 65
OTHER_THRESHOLD = 80


def start_rekognition_job(rekog, bucket, key, min_confidence):
    response = rekog.start_label_detection(
        Video={
            "S3Object": {
                "Bucket": bucket,
                "Name": key,
            }
        },
        MinConfidence=min_confidence,
        Features=["GENERAL_LABELS"],
    )
    return response["JobId"]


def wait_for_job(rekog, job_id):
    print(f"Waiting for Rekognition job: {job_id}")

    while True:
        response = rekog.get_label_detection(
            JobId=job_id,
            MaxResults=1,
            SortBy="TIMESTAMP",
        )

        status = response["JobStatus"]
        print(f"Status: {status}")

        if status == "SUCCEEDED":
            return

        if status == "FAILED":
            raise RuntimeError("Rekognition job failed.")

        time.sleep(10)


def fetch_labels(rekog, job_id):
    labels = []
    next_token = None

    while True:
        params = {
            "JobId": job_id,
            "MaxResults": 1000,
            "SortBy": "TIMESTAMP",
            "AggregateBy": "TIMESTAMPS",
        }

        if next_token:
            params["NextToken"] = next_token

        response = rekog.get_label_detection(**params)
        labels.extend(response.get("Labels", []))
        next_token = response.get("NextToken")

        if not next_token:
            break

    return labels


def extract_label_names(label_item):
    label = label_item.get("Label", {})
    names = set()

    if "Name" in label:
        names.add(label["Name"].lower())

    for parent in label.get("Parents", []):
        names.add(parent["Name"].lower())

    for alias in label.get("Aliases", []):
        names.add(alias["Name"].lower())

    for category in label.get("Categories", []):
        names.add(category["Name"].lower())

    return names


def group_by_time_window(labels, window_seconds):
    windows = defaultdict(lambda: defaultdict(float))

    for item in labels:
        timestamp_sec = item.get("Timestamp", 0) / 1000
        window_index = int(timestamp_sec // window_seconds)

        confidence = item.get("Label", {}).get("Confidence", 0)
        names = extract_label_names(item)

        for name in names:
            windows[window_index][name] = max(windows[window_index][name], confidence)

    return windows


def get_matches(label_map, terms, threshold):
    matches = {}

    for term in terms:
        confidence = label_map.get(term, 0)
        if confidence >= threshold:
            matches[term] = confidence

    return matches


def format_matches(matches):
    return [
        {"label": name, "confidence": round(confidence, 2)}
        for name, confidence in sorted(matches.items(), key=lambda item: item[1], reverse=True)
    ]


def score_windows(windows, window_seconds):
    scored = []
    fish_seen_before = False

    for window_index in sorted(windows.keys()):
        labels = windows[window_index]

        fish_matches = get_matches(labels, FISH_TERMS, FISH_THRESHOLD)
        person_matches = get_matches(labels, PERSON_TERMS, OTHER_THRESHOLD)
        context_matches = get_matches(labels, CONTEXT_TERMS, OTHER_THRESHOLD)
        fishing_matches = get_matches(labels, FISHING_TERMS, OTHER_THRESHOLD)

        has_fish = bool(fish_matches)
        has_person = bool(person_matches)
        has_context = bool(context_matches)
        has_fishing = bool(fishing_matches)

        top_labels = sorted(labels.items(), key=lambda item: item[1], reverse=True)[:16]

        score = 0
        reasons = []

        if has_fish:
            score += 50
            reasons.append("fish detected")

        if has_fish and not fish_seen_before:
            score += 40
            reasons.append("fish first visible")

        if has_fish and has_person:
            score += 40
            reasons.append("person and fish together; possible fish shown to camera")

        if has_person and has_context:
            score += 20
            reasons.append("person in fishing/water context")

        if has_fishing:
            score += 20
            reasons.append("fishing-related label detected")

        if not has_person and not has_fish:
            score -= 20
            reasons.append("low-interest: no person or fish detected")

        event_type = "low_interest"
        if has_fish and not fish_seen_before:
            event_type = "fish_first_visible"
        elif has_fish and has_person:
            event_type = "fish_shown_on_camera"
        elif has_person and has_fishing:
            event_type = "reeling_or_casting"
        elif has_fish and has_context:
            event_type = "fish_storage_or_release"

        scored.append(
            {
                "start_sec": window_index * window_seconds,
                "end_sec": (window_index + 1) * window_seconds,
                "score": score,
                "event_type": event_type,
                "matched_labels": {
                    "fish": format_matches(fish_matches),
                    "person": format_matches(person_matches),
                    "context": format_matches(context_matches),
                    "fishing": format_matches(fishing_matches),
                },
                "reasons": reasons,
                "top_labels": [
                    {"label": name, "confidence": round(confidence, 2)}
                    for name, confidence in top_labels
                ],
            }
        )

        if has_fish:
            fish_seen_before = True

    return scored


def make_candidates(scored_windows, min_score, pad_seconds):
    candidates = []
    for item in scored_windows:
        if item["score"] >= min_score:
            start = max(0, item["start_sec"] - pad_seconds)
            end = item["end_sec"] + pad_seconds
            candidates.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "duration": end - start,
                    "score": item["score"],
                    "event_type": item["event_type"],
                    "matched_labels": item["matched_labels"],
                    "reasons": item["reasons"],
                    "top_labels": item["top_labels"],
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank

    return candidates


def analyze_video(video_path: Path, settings: AnalysisSettings):
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    if not settings.bucket:
        raise RuntimeError("--bucket is required when using Rekognition")

    s3 = boto3.client(
        "s3",
        region_name=settings.region,
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=60,
            read_timeout=300,
        ),
    )
    rekog = boto3.client("rekognition", region_name=settings.region)
    s3_key = f"input/{uuid.uuid4().hex}-{video_path.name}"

    print(f"Uploading video to s3://{settings.bucket}/{s3_key}")
    s3.upload_file(
        str(video_path),
        settings.bucket,
        s3_key,
        Config=TransferConfig(
            multipart_threshold=64 * 1024 * 1024,
            multipart_chunksize=64 * 1024 * 1024,
            max_concurrency=1,
            use_threads=False,
        ),
    )
    print("Upload complete.")

    job_id = start_rekognition_job(
        rekog=rekog,
        bucket=settings.bucket,
        key=s3_key,
        min_confidence=settings.min_confidence,
    )
    print(f"Started Rekognition job: {job_id}")

    wait_for_job(rekog, job_id)
    labels = fetch_labels(rekog, job_id)
    print(f"Fetched {len(labels)} label detections.")

    window_seconds = int(settings.window_seconds)
    windows = group_by_time_window(labels, window_seconds)
    scored_windows = score_windows(windows, window_seconds)
    candidates = make_candidates(
        scored_windows=scored_windows,
        min_score=settings.min_score,
        pad_seconds=settings.pad_seconds,
    )

    return build_analysis_result(video_path, candidates, scored_windows)
