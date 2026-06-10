# YouTube Processing Pipeline

Pipeline for downloading YouTube videos, creating low-resolution proxies, detecting scenes, extracting keyframes, running OCR, translating subtitles, and optionally creating Gemini summaries.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -e ".[ocr]"
```

Paddle GPU is installed separately when GPU OCR is needed.

## Full Pipeline

```powershell
.\.venv\Scripts\stage-pipeline.exe --urls-file urls.txt --db dataset\stage_pipeline.sqlite3 --reset-db --force --download-workers 4 --proxy-workers 1 --scene-workers 2 --keyframe-workers 4 --ocr-workers 1
```

Add subtitle translation:

```powershell
--translate-subtitles
```

## Gemini Summary

Set the API key in the current PowerShell session:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Run summaries after the DB has video URLs:

```powershell
.\.venv\Scripts\stage-pipeline.exe --db dataset\stage_pipeline.sqlite3 --start-at gemini --gemini-workers 1 --force
```

Outputs are written to:

```text
gemini_summary/<video_id>/summary.json
dataset/gemini_summaries.jsonl
```

## Outputs

Generated runtime outputs are ignored by Git:

```text
videos/
subtitles/
proxies/
scenes/
keyframes/
dataset/
logs/
gemini_summary/
```
