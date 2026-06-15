from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path.cwd()
SCENES_DIR = ROOT / "scenes"
KEYFRAMES_DIR = ROOT / "keyframes"
SUBTITLES_DIR = ROOT / "subtitles"
DATASET_DIR = ROOT / "dataset"
VIDEOS_DIR = ROOT / "videos"
DEFAULT_OUTPUT = DATASET_DIR / "documents.jsonl"
DEFAULT_MANIFEST = DATASET_DIR / "documents_manifest.json"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
FRAME_PATTERN = re.compile(r"frame-(-?\d+)", re.IGNORECASE)
TIMESTAMP_PATTERN = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)
TAG_PATTERN = re.compile(r"<[^>]+>")
SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class SubtitleCue:
    start_seconds: float
    end_seconds: float
    text: str
    language: str
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize scenes, keyframes, OCR, and subtitles into retrieval documents."
    )
    parser.add_argument("ids", nargs="*", help="Video IDs. Defaults to all scenes/* folders.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ocr-min-confidence", type=float, default=0.50)
    parser.add_argument(
        "--subtitle-preference",
        default="vi,en,vi-orig",
        help="Comma-separated subtitle language preference. Default: vi,en,vi-orig.",
    )
    parser.add_argument(
        "--watermark-ratio",
        type=float,
        default=0.50,
        help="Drop OCR lines repeated in at least this fraction of keyframes. Default: 0.50.",
    )
    parser.add_argument("--watermark-min-count", type=int, default=3)
    parser.add_argument("--keep-repeated-ocr", action="store_true")
    parser.add_argument("--no-repair-encoding", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_text(value: str) -> str:
    value = html.unescape(value.replace("\ufeff", " "))
    value = TAG_PATTERN.sub(" ", value)
    value = unicodedata.normalize("NFC", value)
    return SPACE_PATTERN.sub(" ", value).strip()


def repair_mojibake(value: str) -> str:
    markers = ("Ã", "Â", "Æ", "Ä", "áº", "á»", "â€")
    if not any(marker in value for marker in markers):
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    before = sum(value.count(marker) for marker in markers)
    after = sum(repaired.count(marker) for marker in markers)
    return repaired if after < before else value


def clean_text(value: str, repair_encoding: bool = True) -> str:
    value = repair_mojibake(value) if repair_encoding else value
    return normalize_text(value)


def normalized_key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in value if character.isalnum())


def parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def subtitle_language(path: Path) -> str:
    name = path.stem
    return name.split(".", 1)[1] if "." in name else "unknown"


def parse_subtitle(path: Path, repair_encoding: bool = True) -> list[SubtitleCue]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    cues: list[SubtitleCue] = []
    index = 0
    language = subtitle_language(path)
    while index < len(lines):
        match = TIMESTAMP_PATTERN.search(lines[index])
        if not match:
            index += 1
            continue
        start = parse_timestamp(match.group("start"))
        end = parse_timestamp(match.group("end"))
        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index].strip())
            index += 1
        text = clean_text(" ".join(text_lines), repair_encoding)
        if text:
            cues.append(SubtitleCue(start, max(end, start), text, language, relative(path)))
        index += 1
    return cues


def subtitle_files(video_id: str) -> list[Path]:
    directory = SUBTITLES_DIR / video_id
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".srt", ".vtt"})


def subtitle_tracks(video_id: str, repair_encoding: bool) -> dict[str, list[SubtitleCue]]:
    tracks: dict[str, list[SubtitleCue]] = defaultdict(list)
    for path in subtitle_files(video_id):
        tracks[subtitle_language(path)].extend(parse_subtitle(path, repair_encoding))
    return dict(tracks)


def overlap(start: float, end: float, cue: SubtitleCue) -> bool:
    return cue.end_seconds > start and cue.start_seconds < end


def deduplicate_texts(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalized_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def cues_for_scene(cues: list[SubtitleCue], start: float, end: float) -> list[SubtitleCue]:
    selected = [cue for cue in cues if overlap(start, end, cue)]
    output: list[SubtitleCue] = []
    seen: set[tuple[float, float, str]] = set()
    for cue in selected:
        key = (cue.start_seconds, cue.end_seconds, normalized_key(cue.text))
        if key in seen:
            continue
        seen.add(key)
        output.append(cue)
    return output


def cues_at_timestamp(cues: list[SubtitleCue], timestamp: float) -> list[SubtitleCue]:
    return [cue for cue in cues if cue.start_seconds <= timestamp <= cue.end_seconds]


def preferred_subtitle_text(
    scene_tracks: dict[str, list[SubtitleCue]], preferences: list[str]
) -> tuple[str | None, str]:
    for language in preferences:
        cues = scene_tracks.get(language) or []
        if cues:
            return language, " ".join(deduplicate_texts([cue.text for cue in cues]))
    for language in sorted(scene_tracks):
        cues = scene_tracks[language]
        if cues:
            return language, " ".join(deduplicate_texts([cue.text for cue in cues]))
    return None, ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def ocr_records(video_id: str) -> list[dict[str, Any]]:
    return load_jsonl(DATASET_DIR / video_id / "ocr.jsonl")


def frame_index(path: Path) -> int | None:
    match = FRAME_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def keyframe_paths(video_id: str, scene_id: str) -> list[Path]:
    directory = KEYFRAMES_DIR / video_id / scene_id
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".webp", ".png"})


def video_info(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": relative(path) if path else None, "fps": None, "frame_count": None, "frame_size": None}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"path": relative(path), "fps": None, "frame_count": None, "frame_size": None}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "path": relative(path),
        "fps": fps,
        "frame_count": frame_count,
        "frame_size": [width, height] if width and height else None,
    }


def source_video(video_id: str) -> Path | None:
    directory = VIDEOS_DIR / video_id
    if not directory.exists():
        return None
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    return paths[0] if paths else None


def line_confidence(line: dict[str, Any]) -> float:
    try:
        return float(line.get("confidence") or 0)
    except (TypeError, ValueError):
        return 0.0


def clean_ocr_lines(record: dict[str, Any], min_confidence: float, repair_encoding: bool) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in record.get("lines") or []:
        confidence = line_confidence(line)
        text = clean_text(str(line.get("text") or ""), repair_encoding)
        key = normalized_key(text)
        if confidence < min_confidence or not key or key in seen:
            continue
        seen.add(key)
        output.append({"text": text, "confidence": confidence, "bbox": line.get("bbox") or []})
    return output


def repeated_ocr_keys(
    records: list[dict[str, Any]], min_confidence: float, repair_encoding: bool, ratio: float, min_count: int
) -> set[str]:
    counts: Counter[str] = Counter()
    total = max(len(records), 1)
    for record in records:
        keys = {normalized_key(line["text"]) for line in clean_ocr_lines(record, min_confidence, repair_encoding)}
        counts.update(keys)
    threshold = max(min_count, int(total * ratio + 0.999999))
    return {key for key, count in counts.items() if count >= threshold}


def ocr_by_path(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        path = str(record.get("keyframe_path") or "").replace("/", "\\").casefold()
        if path:
            output[path] = record
    return output


def scene_keyframe_records(
    video_id: str,
    scene: dict[str, Any],
    proxy_fps: float,
    source_fps: float | None,
    records_by_path: dict[str, dict[str, Any]],
    repeated_keys: set[str],
    min_confidence: float,
    repair_encoding: bool,
    keep_repeated: bool,
    scene_tracks: dict[str, list[SubtitleCue]],
    subtitle_preferences: list[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scene_start = max(float(scene.get("start_seconds") or 0), 0.0)
    for path in keyframe_paths(video_id, str(scene["scene_id"])):
        proxy_index = frame_index(path)
        proxy_index = max(proxy_index or 0, 0)
        timestamp = proxy_index / proxy_fps if proxy_fps > 0 else scene_start
        timestamp_rounded = round(timestamp, 3)
        record = records_by_path.get(relative(path).replace("/", "\\").casefold(), {})
        raw_lines = clean_ocr_lines(record, min_confidence, repair_encoding)
        removed = [line for line in raw_lines if normalized_key(line["text"]) in repeated_keys]
        kept = raw_lines if keep_repeated else [line for line in raw_lines if normalized_key(line["text"]) not in repeated_keys]
        confidences = [line["confidence"] for line in kept]
        keyframe_tracks = {
            language: cues_at_timestamp(cues, timestamp) for language, cues in scene_tracks.items()
        }
        keyframe_tracks = {language: cues for language, cues in keyframe_tracks.items() if cues}
        subtitle_language_value, subtitle_text_value = preferred_subtitle_text(
            keyframe_tracks, subtitle_preferences
        )
        output.append(
            {
                "path": relative(path),
                "proxy_frame_index": proxy_index,
                "source_frame_index": round(timestamp * source_fps) if source_fps else None,
                "timestamp_seconds": timestamp_rounded,
                "scene_local_time_seconds": round(max(timestamp - scene_start, 0.0), 3),
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={max(round(timestamp), 0)}s",
                "subtitle_language": subtitle_language_value,
                "subtitle_text": subtitle_text_value,
                "subtitle_tracks": {
                    language: [
                        {
                            "start_seconds": round(cue.start_seconds, 3),
                            "end_seconds": round(cue.end_seconds, 3),
                            "text": cue.text,
                            "path": cue.path,
                        }
                        for cue in cues
                    ]
                    for language, cues in keyframe_tracks.items()
                },
                "ocr_text": "\n".join(line["text"] for line in kept),
                "ocr_confidence_mean": round(statistics.mean(confidences), 6) if confidences else None,
                "ocr_lines": kept,
                "removed_repeated_ocr": [line["text"] for line in removed],
            }
        )
    return output


def video_ids(requested: list[str]) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    if not SCENES_DIR.exists():
        raise SystemExit("No scenes directory found.")
    return sorted(path.name for path in SCENES_DIR.iterdir() if path.is_dir())


def build_video_documents(video_id: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    metadata_path = SCENES_DIR / video_id / "scenes.json"
    if not metadata_path.exists():
        print(f"[documents] skip {video_id}: missing {metadata_path}")
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    proxy_path = ROOT / metadata["video_path"]
    proxy_fps = float(metadata.get("fps") or 0)
    source = source_video(video_id)
    source_meta = video_info(source)
    proxy_meta = video_info(proxy_path)
    source_fps = source_meta["fps"]
    records = ocr_records(video_id)
    records_by_path = ocr_by_path(records)
    repeated_keys = repeated_ocr_keys(
        records,
        args.ocr_min_confidence,
        not args.no_repair_encoding,
        args.watermark_ratio,
        args.watermark_min_count,
    )
    tracks = subtitle_tracks(video_id, not args.no_repair_encoding)
    preferences = [value.strip() for value in args.subtitle_preference.split(",") if value.strip()]
    documents: list[dict[str, Any]] = []

    for scene in metadata.get("scenes", []):
        scene_id = str(scene["scene_id"])
        start = max(float(scene.get("start_seconds") or 0), 0.0)
        end = max(float(scene.get("end_seconds") or start), start)
        scene_tracks = {language: cues_for_scene(cues, start, end) for language, cues in tracks.items()}
        scene_tracks = {language: cues for language, cues in scene_tracks.items() if cues}
        preferred_language, subtitle_text = preferred_subtitle_text(scene_tracks, preferences)
        keyframes = scene_keyframe_records(
            video_id,
            scene,
            proxy_fps,
            source_fps,
            records_by_path,
            repeated_keys,
            args.ocr_min_confidence,
            not args.no_repair_encoding,
            args.keep_repeated_ocr,
            scene_tracks,
            preferences,
        )
        ocr_text = "\n".join(deduplicate_texts([item["ocr_text"] for item in keyframes if item["ocr_text"]]))
        combined_text = "\n".join(value for value in (subtitle_text, ocr_text) if value)
        proxy_start_frame = max(int(scene.get("start_frame") or 0), 0)
        proxy_end_frame = max(int(scene.get("end_frame") or 0), proxy_start_frame)
        source_start_frame = round(start * source_fps) if source_fps else None
        source_end_frame = round(end * source_fps) if source_fps else None
        documents.append(
            {
                "document_id": f"{video_id}:{scene_id}",
                "video_id": video_id,
                "scene_id": scene_id,
                "start_frame": proxy_start_frame,
                "end_frame": proxy_end_frame,
                "proxy_start_frame": proxy_start_frame,
                "proxy_end_frame": proxy_end_frame,
                "source_start_frame": source_start_frame,
                "source_end_frame": source_end_frame,
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "duration_seconds": round(max(end - start, 0.0), 3),
                "proxy_video": proxy_meta,
                "source_video": source_meta,
                "keyframes": keyframes,
                "subtitle_language": preferred_language,
                "subtitle_text": subtitle_text,
                "subtitle_tracks": {
                    language: [
                        {
                            "start_seconds": round(cue.start_seconds, 3),
                            "end_seconds": round(cue.end_seconds, 3),
                            "text": cue.text,
                            "path": cue.path,
                        }
                        for cue in cues
                    ]
                    for language, cues in scene_tracks.items()
                },
                "ocr_text": ocr_text,
                "combined_text": combined_text,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={max(round(start), 0)}s",
                "repeated_ocr_keys_removed": sorted(repeated_keys) if not args.keep_repeated_ocr else [],
                "generated_at": utc_now(),
            }
        )
    return documents


def write_documents(documents: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")


def write_manifest(documents: list[dict[str, Any]], args: argparse.Namespace) -> None:
    video_ids_present = sorted({document["video_id"] for document in documents})
    payload = {
        "generated_at": utc_now(),
        "output": relative(args.output),
        "document_count": len(documents),
        "video_count": len(video_ids_present),
        "video_ids": video_ids_present,
        "documents_with_subtitle": sum(bool(document["subtitle_text"]) for document in documents),
        "documents_with_ocr": sum(bool(document["ocr_text"]) for document in documents),
        "documents_with_keyframes": sum(bool(document["keyframes"]) for document in documents),
        "searchable_documents": sum(bool(document["combined_text"]) for document in documents),
        "keyframe_count": sum(len(document["keyframes"]) for document in documents),
        "settings": {
            "ocr_min_confidence": args.ocr_min_confidence,
            "subtitle_preference": args.subtitle_preference,
            "watermark_ratio": args.watermark_ratio,
            "watermark_min_count": args.watermark_min_count,
            "keep_repeated_ocr": args.keep_repeated_ocr,
            "repair_encoding": not args.no_repair_encoding,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    documents: list[dict[str, Any]] = []
    for video_id in video_ids(args.ids):
        video_documents = build_video_documents(video_id, args)
        documents.extend(video_documents)
        print(f"[documents] {video_id}: {len(video_documents)} scene document(s)")
    write_documents(documents, args.output)
    write_manifest(documents, args)
    print(f"[documents] wrote {len(documents)} document(s) -> {args.output}")
    print(f"[documents] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
