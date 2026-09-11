# Fishing Highlight Generator

Python pipeline for turning fishing trip videos into downloadable highlight clips.

The product direction is intentionally simple: visitors choose a sample video or upload
their own video, choose whether captions should be included, and receive a set of
highlight clips to download. The analysis backend is an internal configuration choice,
not an end-user product control.

## Local Workflow

Analyze a video with the default backend:

```bash
PYTHONPATH=src python -m fishing_highlights --video vids/live_fishing.mp4
```

Run a specific internal analyzer:

```bash
PYTHONPATH=src python -m fishing_highlights --analyzer gemini --video vids/live_fishing.mp4
PYTHONPATH=src python -m fishing_highlights --analyzer twelvelabs --video vids/live_fishing.mp4
PYTHONPATH=src python -m fishing_highlights --analyzer rekognition --video vids/live_fishing.mp4 --bucket your-s3-bucket
```

Create highlights after analysis:

```bash
PYTHONPATH=src python -m fishing_highlights --video vids/live_fishing.mp4 --create-highlights
```

Create highlights with captions:

```bash
PYTHONPATH=src python -m fishing_highlights --video vids/live_fishing.mp4 --include-captions
```

After installing the package in editable mode, you can use the cleaner console command:

```bash
python -m pip install -e .
fishing-highlights --video vids/live_fishing.mp4
```

## Repository Layout

```text
src/fishing_highlights/
  analyzers/       Provider-specific analysis backends.
  pipeline/        Analysis, trimming, captioning, and local job orchestration.
  cli.py           Main command-line entry point.
  config.py        Product and local-development defaults.
  schema.py        Shared result contract helpers.
  settings.py      Dataclasses for pipeline configuration.
  video.py         Shared video utilities.

tests/             Standard-library unit tests for core behavior.
json_results/      Local generated analyzer output, ignored by git.
highlights/        Local generated highlight videos, ignored by git.
secrets/           Local key files, ignored by git.
vids/              Local input videos, ignored by git.
```

## Secrets

For local development, the scripts can read:

- `secrets/geminiAPI.key`
- `secrets/twelvelabsAPI.key`

For hosted production, use a managed secret store instead of files.
