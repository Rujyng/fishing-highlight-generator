import argparse
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

from boto3.s3.transfer import TransferConfig
from botocore.config import Config
import boto3


FISH_TERMS = {
    "fish", "bass", "trout", "salmon", "tuna"
}

PERSON_TERMS = {
    "person", "human", "people", "face"
}

CONTEXT_TERMS = {
    "water", "sea", "ocean", "lake", "river", "boat", "outdoors"
}

FISHING_TERMS = {
    "fishing", "fishing rod", "rod", "bait", "hook", "reel"
}


FISH_THRESHOLD = 65
OTHER_THRESHOLD = 80


def upload_video(s3, video_path, bucket, key):
    print(f"Uploading video to s3://{bucket}/{key}")

    config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False
    )

    s3.upload_file(
        str(video_path),
        bucket,
        key,
        Config=config
    )

    print("Upload complete.")


def start_rekognition_job(rekog, bucket, key, min_confidence):
    '''
    starts a rekognition job given the video, confidence level and all general labels 
    '''
    response = rekog.start_label_detection(
        Video={
            "S3Object": {
                "Bucket": bucket,
                "Name": key
            }
        },
        MinConfidence=min_confidence,
        Features=["GENERAL_LABELS"]
    )
    return response["JobId"]


def wait_for_job(rekog, job_id):
    '''
    check in on the rekognition job every 10 secs so we are not staring at a blank screen not knowing its status
    '''
    print(f"Waiting for Rekognition job: {job_id}")

    while True:
        response = rekog.get_label_detection(
            JobId=job_id,
            MaxResults=1,
            SortBy="TIMESTAMP"
        )

        status = response["JobStatus"]
        print(f"Status: {status}")

        if status == "SUCCEEDED":
            return

        if status == "FAILED":
            raise RuntimeError("Rekognition job failed.")

        time.sleep(10)


def fetch_labels(rekog, job_id):
    '''
    get all labels with associated timestamp, page by page
    '''
    labels = []
    next_token = None

    while True:
        params = {
            "JobId": job_id,
            "MaxResults": 1000,
            "SortBy": "TIMESTAMP",
            "AggregateBy": "TIMESTAMPS"
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
    '''
    extract all related/parent labels for label_item
    '''
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
    '''
    fill the windows map with all the main and related labels for each window
    each related label will inherit the confidence of the main label
    (may need to fine tune this part more in the future with more examples)
    '''
    windows = defaultdict(lambda: defaultdict(float))

    for item in labels:
        timestamp_sec = item.get("Timestamp", 0) / 1000
        window_index = int(timestamp_sec // window_seconds)

        confidence = item.get("Label", {}).get("Confidence", 0)
        names = extract_label_names(item)

        for name in names:
            windows[window_index][name] = max(
                windows[window_index][name],
                confidence
            )

    return windows


def get_matches(label_map, terms, threshold):
    '''
    find labels we care about that pass the confidence threshold
    '''
    matches = {}

    for term in terms:
        confidence = label_map.get(term, 0)

        if confidence >= threshold:
            matches[term] = confidence

    return matches


def format_matches(matches):
    '''
    return json format of the matched labels
    '''
    return [
        {"label": name, "confidence": round(confidence, 2)}
        for name, confidence in sorted(
            matches.items(),
            key=lambda item: item[1],
            reverse=True
        )
    ]


def score_windows(windows, window_seconds):
    '''
    takes in the windows and their size to score each window to identify highlights later
    '''
    scored = []
    fish_seen_before = False

    for window_index in sorted(windows.keys()):
        # labels of the current window
        labels = windows[window_index]

        # find if there are labels/terms we are looking for within the window
        fish_matches = get_matches(labels, FISH_TERMS, FISH_THRESHOLD)
        person_matches = get_matches(labels, PERSON_TERMS, OTHER_THRESHOLD)
        context_matches = get_matches(labels, CONTEXT_TERMS, OTHER_THRESHOLD)
        fishing_matches = get_matches(labels, FISHING_TERMS, OTHER_THRESHOLD)

        has_fish = bool(fish_matches)
        has_person = bool(person_matches)
        has_context = bool(context_matches)
        has_fishing = bool(fishing_matches)

        # sort all labels by confidence and keep top 16
        # can be fine tuned here
        top_labels = sorted(
            labels.items(),
            key=lambda item: item[1],
            reverse=True
        )[:16]

        score = 0
        reasons = []

        # calculate the interest score for this window
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

        # identify the event
        event_type = "low_interest"

        if has_fish and not fish_seen_before:
            event_type = "fish_first_visible"
        elif has_fish and has_person:
            event_type = "fish_shown_on_camera"
        elif has_person and has_fishing:
            event_type = "reeling_or_casting"
        elif has_fish and has_context:
            event_type = "fish_storage_or_release"

        # return the windows with interest score
        scored.append({
            "start_sec": window_index * window_seconds,
            "end_sec": (window_index + 1) * window_seconds,
            "score": score,
            "event_type": event_type,
            "matched_labels": {
                "fish": format_matches(fish_matches),
                "person": format_matches(person_matches),
                "context": format_matches(context_matches),
                "fishing": format_matches(fishing_matches)
            },
            "reasons": reasons,
            "top_labels": [
                {"label": name, "confidence": round(conf, 2)}
                for name, conf in top_labels
            ]
        })

        if has_fish:
            fish_seen_before = True

    return scored


def make_candidates(scored_windows, min_score, pad_seconds):
    '''
    takes all scored windows, keep only the windows with interesting score of min_score or above
    sort by interesting score
    '''
    candidates = []
    for item in scored_windows:
        if item["score"] >= min_score:
            start = max(0, item["start_sec"] - pad_seconds)
            end = item["end_sec"] + pad_seconds
            candidates.append({
                "start_sec": start,
                "end_sec": end,
                "duration": end - start,
                "score": item["score"],
                "event_type": item["event_type"],
                "matched_labels": item["matched_labels"],
                "reasons": item["reasons"],
                "top_labels": item["top_labels"]
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    for index, item in enumerate(candidates, start=1):
        item["rank"] = index

    return candidates


def main():
    # check arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--min-confidence", type=float, default=45)
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument("--min-score", type=int, default=65)
    parser.add_argument("--pad-seconds", type=int, default=5)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    # retrieve video path
    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # default output file name to be {video_name}_rekognition_result.json
    if args.output is None:
        args.output = f"{video_path.stem}_rekognition_result.json"

    # set up aws s3 and rekognition
    s3 = boto3.client(
        "s3",
        region_name=args.region,
        config=Config(
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=60,
            read_timeout=300
        )
    )
    rekog = boto3.client("rekognition", region_name=args.region)
    s3_key = f"input/{uuid.uuid4().hex}-{video_path.name}"

    # upload video to s3
    upload_video(s3, video_path, args.bucket, s3_key)

    # start rekognition video analysis job
    job_id = start_rekognition_job(
        rekog=rekog, # boto3 
        bucket=args.bucket, # bucket where the video is
        key=s3_key, # exact s3 object path of the video 
        min_confidence=args.min_confidence # minimum confidence to accept any label
    )

    print(f"Started Rekognition job: {job_id}")

    # periodically check in the job to see its status
    wait_for_job(rekog, job_id)

    # get all labels the rekognition returned (a lot)
    labels = fetch_labels(rekog, job_id)
    print(f"Fetched {len(labels)} label detections.")

    # each row of 'labels' is a single label at a specific time
    # we need to turn it into a shorter list and easier to work with
    # turn into map/dictionary where key is the 5 second window of the video, 
    # and the value is the group of labels within that 5 second window
    windows = group_by_time_window(labels, args.window_seconds)

    # each window will receive a score of interestingness to make highlights later
    scored_windows = score_windows(windows, args.window_seconds)

    # only keep the interesting windows with minimum score 65
    candidates = make_candidates(
        scored_windows=scored_windows,
        min_score=args.min_score,
        pad_seconds=args.pad_seconds
    )

    # result json
    result = {
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "candidate_highlights": candidates,
        "all_scored_windows": scored_windows
    }

    # dump result to output file 
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print(f"Saved result to {args.output}")

    print("\nTop candidates:")
    for candidate in candidates[:10]:
        print(
            f"Rank {candidate['rank']}: "
            f"{candidate['start_sec']}s - {candidate['end_sec']}s | "
            f"score={candidate['score']} | "
            f"{candidate['event_type']}"
        )


if __name__ == "__main__":
    main()