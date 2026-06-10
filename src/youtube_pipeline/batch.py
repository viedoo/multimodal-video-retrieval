from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from youtube_pipeline import cli, keyframes, ocr, scenes


ROOT = Path.cwd()
DATASET_DIR = ROOT / "dataset"
MANIFEST_PATH = DATASET_DIR / "batch_manifest.json"
LOGS_DIR = ROOT / "logs" / "batch"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


@dataclass(frozen=True)
class BatchResult:
    url: str
    status: str
    video_ids: list[str]
    timings: dict[str, Any]
    log_path: str | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full YouTube pipeline for many URLs with bounded parallel workers."
    )
    parser.add_argument("urls", nargs="*", help="YouTube video or playlist URLs")
    parser.add_argument(
        "--urls-file",
        type=Path,
        help="Text file containing one YouTube URL per line; blank lines and # comments are ignored",
    )
    parser.add_argument("--workers", type=int, default=2, help="Concurrent URL workers. Default: 2")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Batch status manifest path. Default: dataset/batch_manifest.json",
    )
    parser.add_argument("--force", action="store_true", help="Re-run stages even when outputs already exist.")
    parser.add_argument(
        "--with-thumbnails",
        dest="no_thumbnails",
        action="store_false",
        help="Generate thumbnails in addition to keyframes. Disabled by default.",
    )
    parser.add_argument("--no-thumbnails", dest="no_thumbnails", action="store_true", help="Skip thumbnail generation.")
    parser.set_defaults(no_thumbnails=True)
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR generation.")
    parser.add_argument(
        "--merge-ocr-output",
        type=Path,
        default=DATASET_DIR / "ocr.jsonl",
        help="Merged OCR JSONL output. Default: dataset/ocr.jsonl",
    )

    parser.add_argument("--no-auto-subs", action="store_true", help="Only download creator-provided subtitles.")
    parser.add_argument("--sub-langs", default="vi.*,en.*", help="Subtitle languages. Default: vi.*,en.*")
    parser.add_argument("--no-translate-subtitles", action="store_true", help="Do not translate subtitles.")
    parser.add_argument("--translate-server-url", default="http://localhost:5000")
    parser.add_argument("--translate-api-key", default=None)
    parser.add_argument("--translate-source-lang", default="auto")
    parser.add_argument("--translate-target-lang", default="en")
    parser.add_argument("--translate-batch-chars", type=int, default=4500)

    parser.add_argument("--scene-copy", action="store_true", help="Use fast copy mode for scene splitting.")
    parser.add_argument(
        "--split-scenes",
        action="store_true",
        help="Split detected scenes into mp4 clips. Disabled by default; metadata-only detection is faster.",
    )
    parser.add_argument(
        "--scene-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NVENC for scene split encoding when not using --scene-copy.",
    )
    parser.add_argument("--min-scene-len", default=None)
    parser.add_argument("--scene-downscale", type=int, default=4, help="Scene detection downscale factor. Default: 4")
    parser.add_argument("--scene-frame-skip", type=int, default=0, help="Frames to skip during scene detection. Default: 0")

    add_keyframe_args(parser)

    parser.add_argument("--ocr-lang", default="vi", help="PaddleOCR language code. Default: vi")
    parser.add_argument("--ocr-device", default="auto", help="PaddleOCR device: auto, cpu, or gpu:0. Default: auto")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-image-glob", default="*.jpg")
    return parser.parse_args()


def add_keyframe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-every-seconds", type=float, default=8.0)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--edge-margin", type=float, default=0.10)
    parser.add_argument("--candidate-multiplier", type=int, default=12)
    parser.add_argument(
        "--selection-mode",
        choices=["thumbnail", "action", "hybrid", "flow"],
        default="flow",
    )
    parser.add_argument("--motion-weight", type=float, default=0.55)
    parser.add_argument("--min-temporal-gap", type=float, default=0.16)
    parser.add_argument(
        "--diverse-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--unique-threshold", type=float, default=0.50)
    parser.add_argument("--min-edge-delta", type=float, default=0.006)
    parser.add_argument("--allow-duplicate-fill", action="store_true")
    parser.add_argument(
        "--reject-transition-blur",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--blur-neighbor-frames", type=int, default=2)
    parser.add_argument("--blur-ratio", type=float, default=0.65)
    parser.add_argument("--min-sharpness", type=float, default=35.0)
    parser.add_argument("--min-brightness", type=float, default=25.0)
    parser.add_argument("--max-brightness", type=float, default=235.0)
    parser.add_argument("--motion-method", choices=["absdiff", "farneback"], default="farneback")
    parser.add_argument("--flow-histogram-bins", type=int, default=8)
    parser.add_argument("--flow-unique-threshold", type=float, default=0.20)
    parser.add_argument("--flow-pyr-scale", type=float, default=0.5)
    parser.add_argument("--flow-levels", type=int, default=3)
    parser.add_argument("--flow-window-size", type=int, default=15)
    parser.add_argument("--flow-iterations", type=int, default=3)
    parser.add_argument("--flow-poly-n", type=int, default=5)
    parser.add_argument("--flow-poly-sigma", type=float, default=1.2)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--webp-quality", type=int, default=80)
    parser.add_argument("--jpg-quality", type=int, default=2)
    parser.add_argument(
        "--image-format",
        choices=["jpg", "webp"],
        default="jpg",
    )
    parser.add_argument("--short-scene-seconds", type=float, default=5.0)
    parser.add_argument(
        "--short-scene-strategy",
        choices=["first", "middle", "diverse"],
        default="diverse",
    )
    parser.add_argument("--short-scene-candidates", type=int, default=3)
    parser.add_argument(
        "--cross-scene-unique",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cross-scene-lookback", type=int, default=8)
    parser.add_argument("--cross-scene-color-threshold", type=float, default=0.25)
    parser.add_argument(
        "--skip-similar-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--from-scene-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def read_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls)
    if args.urls_file:
        for line in args.urls_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                urls.append(value)
    if not urls:
        raise SystemExit("Provide at least one URL or --urls-file.")
    return urls


def safe_print(*values: Any, **kwargs: Any) -> None:
    try:
        print(*values, **kwargs)
    except OSError:
        pass


def url_log_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return LOGS_DIR / f"{digest}.log"


def keyframe_args(args: argparse.Namespace, video_id: str) -> argparse.Namespace:
    values = {
        name: getattr(args, name)
        for name in (
            "sample_every_seconds",
            "min_frames",
            "max_frames",
            "edge_margin",
            "candidate_multiplier",
            "selection_mode",
            "motion_weight",
            "min_temporal_gap",
            "diverse_windows",
            "unique_threshold",
            "min_edge_delta",
            "allow_duplicate_fill",
            "reject_transition_blur",
            "blur_neighbor_frames",
            "blur_ratio",
            "min_sharpness",
            "min_brightness",
            "max_brightness",
            "motion_method",
            "flow_histogram_bins",
            "flow_unique_threshold",
            "flow_pyr_scale",
            "flow_levels",
            "flow_window_size",
            "flow_iterations",
            "flow_poly_n",
            "flow_poly_sigma",
            "height",
            "webp_quality",
            "jpg_quality",
            "image_format",
            "short_scene_seconds",
            "short_scene_strategy",
            "short_scene_candidates",
            "cross_scene_unique",
            "cross_scene_lookback",
            "cross_scene_color_threshold",
            "skip_similar_scenes",
            "from_scene_metadata",
        )
    }
    values["ids"] = [video_id]
    return argparse.Namespace(**values)


def thumbnail_args(args: argparse.Namespace, video_id: str) -> argparse.Namespace:
    values = vars(keyframe_args(args, video_id)).copy()
    values.update(
        {
            "min_frames": 1,
            "max_frames": 1,
            "image_format": "webp",
            "fixed_output_name": "thumbnail.webp",
        }
    )
    return argparse.Namespace(**values)


def ocr_args(args: argparse.Namespace, video_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        ids=[video_id],
        lang=args.ocr_lang,
        output=DATASET_DIR / video_id / "ocr.jsonl",
        overwrite=True,
        limit=None,
        min_confidence=args.ocr_min_confidence,
        image_glob=args.ocr_image_glob,
        device=args.ocr_device,
    )


def stage_done(path: Path, pattern: str) -> bool:
    return path.exists() and any(path.rglob(pattern))


def video_download_done(video_id: str) -> bool:
    video_dir = cli.VIDEOS_DIR / video_id
    return video_dir.exists() and any(path.suffix.lower() in VIDEO_EXTENSIONS for path in video_dir.iterdir())


def scenes_done(video_id: str, split_scenes: bool = False) -> bool:
    if split_scenes:
        return stage_done(scenes.SCENES_DIR / video_id, "*.mp4")
    return scenes.scene_metadata_path(video_id).exists()


def keyframes_done(video_id: str) -> bool:
    return stage_done(keyframes.KEYFRAMES_DIR / video_id, "*.jpg")


def thumbnails_done(video_id: str) -> bool:
    return stage_done(keyframes.THUMBNAILS_DIR / video_id, "thumbnail.webp")


def ocr_done(video_id: str) -> bool:
    return (DATASET_DIR / video_id / "ocr.jsonl").exists()


def clear_generated_dir(path: Path) -> None:
    root = ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to clear path outside workspace: {path}")
    if path.exists():
        shutil.rmtree(path)


def run_download(url: str, args: argparse.Namespace) -> list[str]:
    sub_langs = cli.subtitle_languages(args.sub_langs)
    video_ids = cli.download(url, include_auto_subs=not args.no_auto_subs, sub_langs=sub_langs)
    cli.organize_downloaded_subtitles(video_ids)
    return sorted(set(video_ids))


def run_translate_subtitles(video_ids: list[str], args: argparse.Namespace) -> None:
    if not args.no_translate_subtitles:
        cli.translate_downloaded_subtitles(video_ids, args)


def run_scenes(video_id: str, args: argparse.Namespace) -> None:
    for video_path in scenes.videos_for_id(video_id):
        if args.split_scenes:
            scenes.split_video(
                video_path,
                video_id,
                copy=args.scene_copy,
                gpu=args.scene_gpu,
                min_scene_len=args.min_scene_len,
            )
        else:
            scenes.detect_scene_metadata(
                video_path,
                video_id,
                min_scene_len=args.min_scene_len,
                downscale=args.scene_downscale,
                frame_skip=args.scene_frame_skip,
            )


def run_ocr(video_id: str, args: argparse.Namespace) -> None:
    run_args = ocr_args(args, video_id)
    engine = ocr.load_paddle_ocr(run_args.lang, run_args.device)
    records: list[ocr.OcrRecord] = []
    for path in ocr.keyframe_images(video_id, run_args.image_glob):
        records.append(ocr.ocr_image(engine, video_id, path, run_args))
        print(f"OCR {path}")
    ocr.write_records(records, run_args.output, overwrite=True)
    print(f"Wrote {len(records)} OCR record(s) -> {run_args.output}")


def elapsed_seconds(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 3)


def stage_timing(stage: str, started_at: float, skipped: bool = False, **extra: Any) -> dict[str, Any]:
    timing = {
        "stage": stage,
        "seconds": elapsed_seconds(started_at),
        "skipped": skipped,
    }
    timing.update(extra)
    return timing


def print_timing(url: str, timing: dict[str, Any], video_id: str | None = None) -> None:
    target = f" {video_id}" if video_id else ""
    skipped = " skipped" if timing.get("skipped") else ""
    safe_print(f"[TIMING]{target} {timing['stage']}: {timing['seconds']}s{skipped} ({url})")


def process_url(url: str, options: dict[str, Any]) -> BatchResult:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = url_log_path(url)
    with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            result = process_url_inner(url, options)
    return BatchResult(
        url=result.url,
        status=result.status,
        video_ids=result.video_ids,
        timings=result.timings,
        log_path=str(log_path),
        error=result.error,
    )


def process_url_inner(url: str, options: dict[str, Any]) -> BatchResult:
    args = argparse.Namespace(**options)
    url_started_at = time.perf_counter()
    timings: dict[str, Any] = {
        "url_seconds": None,
        "stages": [],
        "videos": {},
    }
    try:
        started_at = time.perf_counter()
        video_ids = run_download(url, args)
        timing = stage_timing("download", started_at, video_count=len(video_ids))
        timings["stages"].append(timing)
        print_timing(url, timing)
        if not video_ids:
            raise RuntimeError("No video IDs returned by yt-dlp.")

        started_at = time.perf_counter()
        run_translate_subtitles(video_ids, args)
        timing = stage_timing("translate_subtitles", started_at, skipped=args.no_translate_subtitles)
        timings["stages"].append(timing)
        print_timing(url, timing)

        for video_id in video_ids:
            video_timings: list[dict[str, Any]] = []
            timings["videos"][video_id] = video_timings

            if args.force or not scenes_done(video_id, split_scenes=args.split_scenes):
                started_at = time.perf_counter()
                run_scenes(video_id, args)
                timing = stage_timing("scenes", started_at)
            else:
                print(f"Skipping existing scenes: {video_id}")
                timing = stage_timing("scenes", time.perf_counter(), skipped=True)
            video_timings.append(timing)
            print_timing(url, timing, video_id)

            if args.force or not keyframes_done(video_id):
                if args.force:
                    clear_generated_dir(keyframes.KEYFRAMES_DIR / video_id)
                started_at = time.perf_counter()
                keyframes.run(keyframe_args(args, video_id))
                timing = stage_timing("keyframes", started_at)
            else:
                print(f"Skipping existing keyframes: {video_id}")
                timing = stage_timing("keyframes", time.perf_counter(), skipped=True)
            video_timings.append(timing)
            print_timing(url, timing, video_id)

            if not args.no_thumbnails:
                if args.force or not thumbnails_done(video_id):
                    started_at = time.perf_counter()
                    keyframes.run(thumbnail_args(args, video_id), output_root=keyframes.THUMBNAILS_DIR, label="thumbnails")
                    timing = stage_timing("thumbnails", started_at)
                else:
                    print(f"Skipping existing thumbnails: {video_id}")
                    timing = stage_timing("thumbnails", time.perf_counter(), skipped=True)
            else:
                timing = stage_timing("thumbnails", time.perf_counter(), skipped=True)
            video_timings.append(timing)
            print_timing(url, timing, video_id)

            if not args.no_ocr:
                if args.force or not ocr_done(video_id):
                    started_at = time.perf_counter()
                    run_ocr(video_id, args)
                    timing = stage_timing("ocr", started_at)
                else:
                    print(f"Skipping existing OCR: {video_id}")
                    timing = stage_timing("ocr", time.perf_counter(), skipped=True)
            else:
                timing = stage_timing("ocr", time.perf_counter(), skipped=True)
            video_timings.append(timing)
            print_timing(url, timing, video_id)

        timings["url_seconds"] = elapsed_seconds(url_started_at)
        safe_print(f"[TIMING] url total: {timings['url_seconds']}s ({url})")
        return BatchResult(url=url, status="done", video_ids=video_ids, timings=timings)
    except Exception as exc:
        timings["url_seconds"] = elapsed_seconds(url_started_at)
        return BatchResult(url=url, status="failed", video_ids=[], timings=timings, error=str(exc))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_ocr_outputs(video_ids: list[str], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as merged:
        for video_id in sorted(set(video_ids)):
            path = DATASET_DIR / video_id / "ocr.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    merged.write(line + "\n")
                    count += 1
    return count


def main() -> None:
    args = parse_args()
    urls = read_urls(args)
    if args.workers <= 0:
        raise SystemExit("--workers must be greater than 0.")

    options = vars(args).copy()
    options.pop("urls", None)
    options["urls_file"] = None

    manifest: dict[str, Any] = {
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "workers": args.workers,
        "jobs": {
            url: {"status": "queued", "video_ids": [], "log_path": str(url_log_path(url)), "error": None}
            for url in urls
        },
    }
    write_manifest(args.manifest, manifest)

    completed_video_ids: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for url in urls:
            manifest["jobs"][url]["status"] = "submitted"
            futures[executor.submit(process_url, url, options)] = url
        manifest["updated_at"] = utc_now()
        write_manifest(args.manifest, manifest)

        for future in as_completed(futures):
            url = futures[future]
            result = future.result()
            manifest["jobs"][url] = {
                "status": result.status,
                "video_ids": result.video_ids,
                "timings": result.timings,
                "log_path": result.log_path,
                "error": result.error,
            }
            manifest["updated_at"] = utc_now()
            write_manifest(args.manifest, manifest)
            completed_video_ids.extend(result.video_ids)
            safe_print(f"{result.status.upper()} {url}: {', '.join(result.video_ids) or result.error}")

    if not args.no_ocr:
        merged_count = merge_ocr_outputs(completed_video_ids, args.merge_ocr_output)
        manifest["ocr_merged_records"] = merged_count
        manifest["ocr_merged_output"] = str(args.merge_ocr_output)

    manifest["finished_at"] = utc_now()
    manifest["updated_at"] = utc_now()
    write_manifest(args.manifest, manifest)

    failed = [url for url, job in manifest["jobs"].items() if job["status"] != "done"]
    if failed:
        raise SystemExit(f"Batch finished with {len(failed)} failed job(s). See {args.manifest}")
    safe_print(f"Batch finished successfully. See {args.manifest}")


if __name__ == "__main__":
    main()
