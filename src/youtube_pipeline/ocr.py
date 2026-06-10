from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2


ROOT = Path.cwd()
KEYFRAMES_DIR = ROOT / "keyframes"
DATASET_DIR = ROOT / "dataset"
FRAME_PATTERN = re.compile(r"frame-(\d+)", re.IGNORECASE)
SCENE_PATTERN = re.compile(r"scene-(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    bbox: list[list[float]]


@dataclass(frozen=True)
class OcrRecord:
    video_id: str
    scene_id: str
    frame_index: int | None
    scene_local_time_seconds: float | None
    keyframe_path: str
    text: str
    lines: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR on extracted keyframe images and export JSONL records for retrieval."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help="Video IDs to OCR. Defaults to all keyframes/* folders.",
    )
    parser.add_argument("--lang", default="vi", help="PaddleOCR language code. Default: vi")
    parser.add_argument("--device", default="auto", help="PaddleOCR device: auto, cpu, or gpu:0. Default: auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_DIR / "ocr.jsonl",
        help="Output JSONL file. Default: dataset/ocr.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSONL instead of appending.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of keyframe images to OCR for smoke tests.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum OCR confidence to keep a text line.",
    )
    parser.add_argument(
        "--image-glob",
        default="*.jpg",
        help="Image filename glob inside each scene folder. Default: *.jpg",
    )
    return parser.parse_args()


def video_ids(requested_ids: list[str]) -> list[str]:
    if requested_ids:
        return requested_ids
    if not KEYFRAMES_DIR.exists():
        raise SystemExit("No keyframes directory found. Run keyframe-pipeline first.")
    return sorted(path.name for path in KEYFRAMES_DIR.iterdir() if path.is_dir())


def keyframe_images(video_id: str, image_glob: str) -> list[Path]:
    video_dir = KEYFRAMES_DIR / video_id
    if not video_dir.exists():
        raise SystemExit(f"Keyframe folder not found: {video_dir}")
    return sorted(video_dir.glob(f"scene-*/*{image_glob.lstrip('*')}"))


def frame_index_from_path(path: Path) -> int | None:
    match = FRAME_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def scene_id_from_path(path: Path) -> str:
    match = SCENE_PATTERN.search(path.parent.name)
    return f"scene-{match.group(1)}" if match else path.parent.name


def scene_local_time(video_id: str, scene_id: str, frame_index: int | None) -> float | None:
    if frame_index is None:
        return None
    metadata_path = ROOT / "scenes" / video_id / "scenes.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        fps = float(payload.get("fps") or 0)
        for scene in payload.get("scenes", []):
            if scene.get("scene_id") == scene_id and fps > 0:
                return round(max((frame_index - int(scene["start_frame"])) / fps, 0.0), 3)
    scene_path = ROOT / "scenes" / video_id / f"{scene_id}.mp4"
    if not scene_path.exists():
        return None
    capture = cv2.VideoCapture(str(scene_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    capture.release()
    if fps <= 0:
        return None
    return round(frame_index / fps, 3)


def resolve_paddle_device(device: str) -> str:
    if device.lower() != "auto":
        return device
    try:
        import paddle
    except ImportError as exc:
        raise SystemExit("PaddlePaddle is not installed.") from exc
    return "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"


def validate_paddle_device(device: str) -> None:
    if not device.lower().startswith("gpu"):
        return
    try:
        import paddle
    except ImportError as exc:
        raise SystemExit("PaddlePaddle is not installed.") from exc
    if not paddle.device.is_compiled_with_cuda():
        raise SystemExit(
            "OCR GPU requested, but the installed PaddlePaddle is CPU-only. "
            "Install the PaddlePaddle GPU wheel matching your CUDA version, then rerun with --device gpu:0."
        )


def load_paddle_ocr(lang: str, device: str = "cpu") -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit(
            "PaddleOCR is not installed. Install the OCR dependencies in the project venv with: "
            "D:\\pipeline1\\.venv\\Scripts\\python.exe -m pip install paddlepaddle paddleocr"
        ) from exc
    resolved_device = resolve_paddle_device(device)
    validate_paddle_device(resolved_device)
    kwargs: dict[str, Any] = {"device": resolved_device}
    if resolved_device.lower().startswith("cpu"):
        kwargs["enable_mkldnn"] = False
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        **kwargs,
    )


def normalize_bbox(value: Any) -> list[list[float]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    bbox: list[list[float]] = []
    for point in value:
        if isinstance(point, Sequence) and not isinstance(point, (str, bytes)) and len(point) >= 2:
            bbox.append([float(point[0]), float(point[1])])
    return bbox


def extract_v3_lines(page: Mapping[str, Any], min_confidence: float) -> list[OcrLine]:
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []
    polygons = page.get("rec_polys") or []
    lines: list[OcrLine] = []

    for text_value, score_value, polygon in zip(texts, scores, polygons):
        text = str(text_value).strip()
        confidence = float(score_value)
        if text and confidence >= min_confidence:
            lines.append(OcrLine(text=text, confidence=confidence, bbox=normalize_bbox(polygon)))
    return lines


def extract_lines(result: Any, min_confidence: float) -> list[OcrLine]:
    lines: list[OcrLine] = []
    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if not page:
            continue
        if isinstance(page, Mapping):
            lines.extend(extract_v3_lines(page, min_confidence))
            continue
        for item in page:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            bbox = normalize_bbox(item[0])
            payload = item[1]
            if not isinstance(payload, (list, tuple)) or len(payload) < 2:
                continue
            text = str(payload[0]).strip()
            confidence = float(payload[1])
            if text and confidence >= min_confidence:
                lines.append(OcrLine(text=text, confidence=confidence, bbox=bbox))
    return lines


def ocr_image(ocr: Any, video_id: str, path: Path, args: argparse.Namespace) -> OcrRecord:
    scene_id = scene_id_from_path(path)
    frame_index = frame_index_from_path(path)
    result = ocr.predict(str(path))
    lines = extract_lines(result, args.min_confidence)
    return OcrRecord(
        video_id=video_id,
        scene_id=scene_id,
        frame_index=frame_index,
        scene_local_time_seconds=scene_local_time(video_id, scene_id, frame_index),
        keyframe_path=str(path.relative_to(ROOT)),
        text="\n".join(line.text for line in lines),
        lines=[asdict(line) for line in lines],
    )


def write_records(records: list[OcrRecord], output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with output_path.open(mode, encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    ocr = load_paddle_ocr(args.lang, args.device)
    records: list[OcrRecord] = []

    for video_id in video_ids(args.ids):
        for path in keyframe_images(video_id, args.image_glob):
            records.append(ocr_image(ocr, video_id, path, args))
            print(f"OCR {path}")
            if args.limit is not None and len(records) >= args.limit:
                write_records(records, args.output, args.overwrite)
                print(f"Wrote {len(records)} OCR record(s) -> {args.output}")
                return

    write_records(records, args.output, args.overwrite)
    print(f"Wrote {len(records)} OCR record(s) -> {args.output}")


if __name__ == "__main__":
    main()
