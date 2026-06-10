from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from yt_dlp import YoutubeDL

from youtube_pipeline import cli, gemini_summary, keyframes, ocr, scenes


ROOT = Path.cwd()
DATASET_DIR = ROOT / "dataset"
DB_PATH = DATASET_DIR / "stage_pipeline.sqlite3"
PROXIES_DIR = ROOT / "proxies"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
STAGES = ("download", "proxy", "scenes", "keyframes", "ocr", "gemini")


@dataclass(frozen=True)
class StageResult:
    key: str
    status: str
    seconds: float
    video_ids: list[str] | None = None
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scalable staged YouTube pipeline with SQLite resume state."
    )
    parser.add_argument("urls", nargs="*", help="YouTube video or playlist URLs")
    parser.add_argument("--urls-file", type=Path)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--reset-db", action="store_true", help="Delete and recreate the stage DB.")
    parser.add_argument("--force", action="store_true", help="Re-run generated stages even if marked done.")
    parser.add_argument("--start-at", choices=STAGES, default="download")
    parser.add_argument("--stop-after", choices=STAGES, default=None)
    parser.add_argument("--ids", nargs="*", help="Existing downloaded video IDs to enqueue without downloading.")
    parser.add_argument("--ids-file", type=Path, help="Text file containing one existing video ID per line.")

    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--proxy-workers", type=int, default=1)
    parser.add_argument("--scene-workers", type=int, default=2)
    parser.add_argument("--keyframe-workers", type=int, default=2)
    parser.add_argument("--ocr-workers", type=int, default=1)
    parser.add_argument("--gemini-workers", type=int, default=1)

    parser.add_argument("--max-height", type=int, default=720, help="Download video-only up to this height.")
    parser.add_argument("--format", default=None, help="yt-dlp format override.")
    parser.add_argument("--no-auto-subs", action="store_true")
    parser.add_argument("--sub-langs", default="vi.*,en.*")
    parser.add_argument(
        "--translate-subtitles",
        action="store_true",
        help="Translate subtitles during download stage. Disabled by default for speed.",
    )
    parser.add_argument("--translate-server-url", default="http://localhost:5000")
    parser.add_argument("--translate-api-key", default=None)
    parser.add_argument("--translate-source-lang", default="auto")
    parser.add_argument("--translate-target-lang", default="en")
    parser.add_argument("--translate-batch-chars", type=int, default=4500)

    parser.add_argument("--proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proxy-height", type=int, default=360)
    parser.add_argument("--proxy-cq", type=int, default=28)
    parser.add_argument("--proxy-preset", default="p4")
    parser.add_argument("--proxy-mode", choices=["auto", "gpu", "cpu"], default="auto")

    parser.add_argument("--scene-downscale", type=int, default=2)
    parser.add_argument("--scene-frame-skip", type=int, default=1)
    parser.add_argument("--min-scene-len", default=None)
    parser.add_argument("--scene-detector", choices=["adaptive", "content", "histogram"], default="adaptive")
    parser.add_argument("--scene-adaptive-threshold", type=float, default=3.0)
    parser.add_argument("--scene-min-content-val", type=float, default=15.0)
    parser.add_argument("--scene-content-threshold", type=float, default=27.0)
    parser.add_argument("--scene-histogram-threshold", type=float, default=0.05)

    add_keyframe_args(parser)

    parser.add_argument("--ocr-lang", default="vi")
    parser.add_argument("--ocr-device", default="auto")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.35)
    parser.add_argument("--ocr-image-glob", default="*.jpg")
    parser.add_argument("--merge-ocr-output", type=Path, default=DATASET_DIR / "ocr.jsonl")

    parser.add_argument("--gemini-summary", action="store_true", help="Run Gemini YouTube summary stage.")
    parser.add_argument("--gemini-api-key", default=None, help="Defaults to GEMINI_API_KEY env var.")
    parser.add_argument("--gemini-model", default=gemini_summary.DEFAULT_MODEL)
    parser.add_argument("--gemini-prompt", default=None)
    parser.add_argument("--gemini-prompt-file", type=Path, default=None)
    parser.add_argument("--gemini-output-root", type=Path, default=gemini_summary.GEMINI_SUMMARY_DIR)
    parser.add_argument("--merge-gemini-output", type=Path, default=DATASET_DIR / "gemini_summaries.jsonl")
    parser.add_argument("--gemini-timeout", type=int, default=180)
    parser.add_argument("--gemini-temperature", type=float, default=0.2)
    parser.add_argument("--gemini-max-output-tokens", type=int, default=2048)
    return parser.parse_args()


def add_keyframe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-every-seconds", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=2)
    parser.add_argument("--edge-margin", type=float, default=0.10)
    parser.add_argument("--candidate-multiplier", type=int, default=1)
    parser.add_argument("--selection-mode", choices=["thumbnail", "action", "hybrid", "flow"], default="flow")
    parser.add_argument("--motion-weight", type=float, default=0.55)
    parser.add_argument("--min-temporal-gap", type=float, default=0.16)
    parser.add_argument("--diverse-windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unique-threshold", type=float, default=0.50)
    parser.add_argument("--min-edge-delta", type=float, default=0.006)
    parser.add_argument("--allow-duplicate-fill", action="store_true")
    parser.add_argument("--reject-transition-blur", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--analysis-height", type=int, default=240)
    parser.add_argument("--decode-mode", choices=["seek", "sequential"], default="seek")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--webp-quality", type=int, default=80)
    parser.add_argument("--jpg-quality", type=int, default=2)
    parser.add_argument("--image-format", choices=["jpg", "webp"], default="jpg")
    parser.add_argument("--short-scene-seconds", type=float, default=5.0)
    parser.add_argument("--short-scene-strategy", choices=["first", "middle", "diverse"], default="first")
    parser.add_argument("--short-scene-candidates", type=int, default=3)
    parser.add_argument("--cross-scene-unique", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cross-scene-lookback", type=int, default=8)
    parser.add_argument("--cross-scene-color-threshold", type=float, default=0.25)
    parser.add_argument("--skip-similar-scenes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--from-scene-metadata", action=argparse.BooleanOptionalAction, default=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 3)


def read_urls(args: argparse.Namespace, *, required: bool) -> list[str]:
    urls = list(args.urls)
    if args.urls_file:
        for line in args.urls_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                urls.append(value)
    if required and not urls:
        raise SystemExit("Provide URLs or --urls-file.")
    return list(dict.fromkeys(urls))


def read_ids(args: argparse.Namespace) -> list[str]:
    video_ids = list(args.ids or [])
    if args.ids_file:
        for line in args.ids_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                video_ids.append(value)
    return list(dict.fromkeys(video_ids))


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path, reset: bool) -> None:
    if reset and db_path.exists():
        db_path.unlink()
    with connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS url_jobs (
                url TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                video_ids_json TEXT NOT NULL DEFAULT '[]',
                seconds REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS video_jobs (
                video_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                download_status TEXT NOT NULL DEFAULT 'done',
                proxy_status TEXT NOT NULL DEFAULT 'pending',
                scene_status TEXT NOT NULL DEFAULT 'pending',
                keyframe_status TEXT NOT NULL DEFAULT 'pending',
                ocr_status TEXT NOT NULL DEFAULT 'pending',
                gemini_status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                timings_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(video_jobs)")}
        if "proxy_status" not in columns:
            connection.execute("ALTER TABLE video_jobs ADD COLUMN proxy_status TEXT NOT NULL DEFAULT 'pending'")
        if "gemini_status" not in columns:
            connection.execute("ALTER TABLE video_jobs ADD COLUMN gemini_status TEXT NOT NULL DEFAULT 'pending'")


def enqueue_urls(db_path: Path, urls: list[str]) -> None:
    now = utc_now()
    with connect(db_path) as connection:
        for url in urls:
            connection.execute(
                """
                INSERT OR IGNORE INTO url_jobs(url, status, created_at, updated_at)
                VALUES (?, 'pending', ?, ?)
                """,
                (url, now, now),
            )


def enqueue_video_ids(db_path: Path, video_ids: list[str], source: str = "manual") -> None:
    now = utc_now()
    with connect(db_path) as connection:
        for video_id in video_ids:
            connection.execute(
                """
                INSERT INTO video_jobs(video_id, url, download_status, created_at, updated_at)
                VALUES (?, ?, 'done', ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    download_status = 'done',
                    error = NULL,
                    updated_at = excluded.updated_at
                """,
                (video_id, source, now, now),
            )


def rows_for_stage(db_path: Path, stage: str, force: bool) -> list[sqlite3.Row]:
    with connect(db_path) as connection:
        if stage == "download":
            if force:
                return list(connection.execute("SELECT * FROM url_jobs ORDER BY created_at"))
            return list(connection.execute("SELECT * FROM url_jobs WHERE status != 'done' ORDER BY created_at"))
        column = f"{stage}_status"
        if force:
            return list(connection.execute("SELECT * FROM video_jobs ORDER BY created_at"))
        return list(
            connection.execute(
                f"SELECT * FROM video_jobs WHERE {column} != 'done' AND error IS NULL ORDER BY created_at"
            )
        )


def mark_url(db_path: Path, url: str, result: StageResult) -> None:
    now = utc_now()
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE url_jobs
            SET status = ?, error = ?, video_ids_json = ?, seconds = ?, updated_at = ?
            WHERE url = ?
            """,
            (
                result.status,
                result.error,
                json.dumps(result.video_ids or [], ensure_ascii=False),
                result.seconds,
                now,
                url,
            ),
        )
        if result.status == "done":
            for video_id in result.video_ids or []:
                connection.execute(
                    """
                    INSERT INTO video_jobs(video_id, url, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(video_id) DO UPDATE SET
                        url = excluded.url,
                        download_status = 'done',
                        proxy_status = CASE WHEN proxy_status = 'done' THEN 'done' ELSE 'pending' END,
                        gemini_status = CASE WHEN gemini_status = 'done' THEN 'done' ELSE 'pending' END,
                        error = NULL,
                        updated_at = excluded.updated_at
                    """,
                    (video_id, url, now, now),
                )


def timings_for(row: sqlite3.Row) -> dict[str, Any]:
    value = row["timings_json"] if "timings_json" in row.keys() else "{}"
    return json.loads(value or "{}")


def mark_video_stage(db_path: Path, video_id: str, stage: str, result: StageResult) -> None:
    now = utc_now()
    column = f"{stage}_status"
    with connect(db_path) as connection:
        row = connection.execute("SELECT * FROM video_jobs WHERE video_id = ?", (video_id,)).fetchone()
        timings = timings_for(row)
        timings[stage] = result.seconds
        connection.execute(
            f"""
            UPDATE video_jobs
            SET {column} = ?, error = ?, timings_json = ?, updated_at = ?
            WHERE video_id = ?
            """,
            (result.status, result.error, json.dumps(timings, ensure_ascii=False), now, video_id),
        )


def yt_dlp_format(max_height: int, override: str | None) -> str:
    if override:
        return override
    return f"bv*[height<={max_height}]/bv*/b[height<={max_height}]/b"


def download_video_only(url: str, options: dict[str, Any]) -> StageResult:
    started_at = time.perf_counter()
    try:
        sub_langs = cli.subtitle_languages(options["sub_langs"])
        ydl_options = {
            "format": yt_dlp_format(options["max_height"], options.get("format")),
            "outtmpl": {
                "default": str(cli.VIDEOS_DIR / "%(id)s" / "video.%(ext)s"),
                "subtitle": str(cli.SUBTITLES_DIR / "%(id)s" / "subtitles.%(ext)s"),
            },
            "writesubtitles": True,
            "writeautomaticsub": not options["no_auto_subs"],
            "subtitleslangs": sub_langs,
            "subtitlesformat": "srt/vtt/best",
            "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
            "ignoreerrors": True,
            "noplaylist": False,
            "continuedl": True,
            "retries": 10,
            "fragment_retries": 10,
            "socket_timeout": 30,
        }
        print(f"[download] {url}")
        with YoutubeDL(ydl_options) as ydl:
            info = ydl.extract_info(url, download=True)
        video_ids = sorted(set(cli.downloaded_video_ids(info or {})))
        cli.organize_downloaded_subtitles(video_ids)
        if options["translate_subtitles"]:
            cli.translate_downloaded_subtitles(video_ids, argparse.Namespace(**options))
        if not video_ids:
            raise RuntimeError("yt-dlp returned no video IDs.")
        return StageResult(url, "done", elapsed(started_at), video_ids=video_ids)
    except Exception as exc:
        return StageResult(url, "failed", elapsed(started_at), error=str(exc))


def video_files(video_id: str) -> list[Path]:
    video_dir = cli.VIDEOS_DIR / video_id
    if not video_dir.exists():
        raise RuntimeError(f"Missing video directory: {video_dir}")
    files = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No video files for: {video_id}")
    return files


def proxy_path(video_id: str) -> Path:
    return PROXIES_DIR / video_id / "proxy.mp4"


def proxy_files(video_id: str, use_proxy: bool) -> list[Path]:
    path = proxy_path(video_id)
    if use_proxy and path.exists():
        return [path]
    return video_files(video_id)


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def proxy_commands(video_path: Path, output_path: Path, options: dict[str, Any]) -> list[list[str]]:
    height = max(2, int(options["proxy_height"]))
    cq = str(int(options["proxy_cq"]))
    preset = str(options["proxy_preset"])
    mode = options["proxy_mode"]
    commands: list[list[str]] = []
    if mode in {"auto", "gpu"}:
        commands.append(
            [
                ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(video_path),
                "-an",
                "-sn",
                "-vf",
                f"scale_cuda=-2:{height}",
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-cq",
                cq,
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        commands.append(
            [
                ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-hwaccel",
                "cuda",
                "-i",
                str(video_path),
                "-an",
                "-sn",
                "-vf",
                f"scale=-2:{height}",
                "-c:v",
                "h264_nvenc",
                "-preset",
                preset,
                "-cq",
                cq,
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    if mode in {"auto", "cpu"}:
        commands.append(
            [
                ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-an",
                "-sn",
                "-vf",
                f"scale=-2:{height}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "28",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    return commands


def create_proxy(video_id: str, options: dict[str, Any]) -> StageResult:
    started_at = time.perf_counter()
    try:
        if not options["proxy"]:
            return StageResult(video_id, "done", elapsed(started_at))
        source = video_files(video_id)[0]
        output = proxy_path(video_id)
        if output.exists() and not options["force"]:
            return StageResult(video_id, "done", elapsed(started_at))
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_output = output.with_suffix(".tmp.mp4")
        if temp_output.exists():
            temp_output.unlink()
        errors: list[str] = []
        for command in proxy_commands(source, temp_output, options):
            try:
                subprocess.run(command, check=True)
                temp_output.replace(output)
                print(f"[proxy] {video_id}: {source} -> {output}")
                return StageResult(video_id, "done", elapsed(started_at))
            except subprocess.CalledProcessError as exc:
                errors.append(" ".join(command[:8]) + f" ... exit={exc.returncode}")
                if temp_output.exists():
                    temp_output.unlink()
        raise RuntimeError("; ".join(errors) or "No proxy command attempted.")
    except Exception as exc:
        return StageResult(video_id, "failed", elapsed(started_at), error=str(exc))


def run_scene_stage(video_id: str, options: dict[str, Any]) -> StageResult:
    started_at = time.perf_counter()
    try:
        for video_path in proxy_files(video_id, options["proxy"]):
            scenes.detect_scene_metadata(
                video_path,
                video_id,
                min_scene_len=options["min_scene_len"],
                downscale=options["scene_downscale"],
                frame_skip=options["scene_frame_skip"],
                detector=options["scene_detector"],
                adaptive_threshold=options["scene_adaptive_threshold"],
                min_content_val=options["scene_min_content_val"],
                content_threshold=options["scene_content_threshold"],
                histogram_threshold=options["scene_histogram_threshold"],
            )
        return StageResult(video_id, "done", elapsed(started_at))
    except Exception as exc:
        return StageResult(video_id, "failed", elapsed(started_at), error=str(exc))


def keyframe_args(options: dict[str, Any], video_id: str) -> argparse.Namespace:
    names = [
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
        "analysis_height",
        "decode_mode",
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
    ]
    values = {name: options[name] for name in names}
    values["ids"] = [video_id]
    return argparse.Namespace(**values)


def clear_generated_dir(path: Path) -> None:
    root = ROOT.resolve()
    resolved = path.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to clear outside workspace: {path}")
    if path.exists():
        shutil.rmtree(path)


def run_keyframe_stage(video_id: str, options: dict[str, Any]) -> StageResult:
    started_at = time.perf_counter()
    try:
        if options["force"]:
            clear_generated_dir(keyframes.KEYFRAMES_DIR / video_id)
        keyframes.run(keyframe_args(options, video_id))
        return StageResult(video_id, "done", elapsed(started_at))
    except Exception as exc:
        return StageResult(video_id, "failed", elapsed(started_at), error=str(exc))


def ocr_args(options: dict[str, Any], video_id: str) -> argparse.Namespace:
    return argparse.Namespace(
        ids=[video_id],
        lang=options["ocr_lang"],
        device=options["ocr_device"],
        output=DATASET_DIR / video_id / "ocr.jsonl",
        overwrite=True,
        limit=None,
        min_confidence=options["ocr_min_confidence"],
        image_glob=options["ocr_image_glob"],
    )


def run_ocr_chunk(video_ids: list[str], options: dict[str, Any]) -> list[StageResult]:
    engine = ocr.load_paddle_ocr(options["ocr_lang"], options["ocr_device"])
    results: list[StageResult] = []
    for video_id in video_ids:
        started_at = time.perf_counter()
        try:
            args = ocr_args(options, video_id)
            records = [ocr.ocr_image(engine, video_id, path, args) for path in ocr.keyframe_images(video_id, args.image_glob)]
            ocr.write_records(records, args.output, overwrite=True)
            print(f"[ocr] {video_id}: {len(records)} records")
            results.append(StageResult(video_id, "done", elapsed(started_at)))
        except Exception as exc:
            results.append(StageResult(video_id, "failed", elapsed(started_at), error=str(exc)))
    return results


def chunks(values: list[str], count: int) -> list[list[str]]:
    if not values:
        return []
    count = max(1, min(count, len(values)))
    chunk_size = math.ceil(len(values) / count)
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def run_downloads(args: argparse.Namespace, options: dict[str, Any]) -> None:
    rows = rows_for_stage(args.db, "download", args.force)
    print(f"[stage] download: {len(rows)} url(s), workers={args.download_workers}")
    with ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as executor:
        futures = {executor.submit(download_video_only, row["url"], options): row["url"] for row in rows}
        for future in as_completed(futures):
            result = future.result()
            mark_url(args.db, result.key, result)
            print(f"[download] {result.status} {result.key} {result.seconds}s {result.error or ''}")


def run_process_stage(args: argparse.Namespace, stage: str, worker_count: int, fn: Any, options: dict[str, Any]) -> None:
    rows = rows_for_stage(args.db, stage, args.force)
    print(f"[stage] {stage}: {len(rows)} video(s), workers={worker_count}")
    with ProcessPoolExecutor(max_workers=max(1, worker_count)) as executor:
        futures = {executor.submit(fn, row["video_id"], options): row["video_id"] for row in rows}
        for future in as_completed(futures):
            result = future.result()
            mark_video_stage(args.db, result.key, stage, result)
            print(f"[{stage}] {result.status} {result.key} {result.seconds}s {result.error or ''}")


def runnable_ocr_video_ids(db_path: Path, force: bool) -> list[str]:
    with connect(db_path) as connection:
        if force:
            rows = connection.execute(
                "SELECT video_id FROM video_jobs WHERE scene_status = 'done' AND keyframe_status = 'done' ORDER BY created_at"
            )
        else:
            rows = connection.execute(
                """
                SELECT video_id FROM video_jobs
                WHERE scene_status = 'done' AND keyframe_status = 'done' AND ocr_status != 'done' AND error IS NULL
                ORDER BY created_at
                """
            )
        return [row["video_id"] for row in rows]


def run_ocr_stage(args: argparse.Namespace, options: dict[str, Any]) -> None:
    video_ids = runnable_ocr_video_ids(args.db, args.force)
    print(f"[stage] ocr: {len(video_ids)} video(s), workers={args.ocr_workers}")
    with ProcessPoolExecutor(max_workers=max(1, args.ocr_workers)) as executor:
        futures = [executor.submit(run_ocr_chunk, chunk, options) for chunk in chunks(video_ids, args.ocr_workers)]
        for future in as_completed(futures):
            for result in future.result():
                mark_video_stage(args.db, result.key, "ocr", result)
                print(f"[ocr] {result.status} {result.key} {result.seconds}s {result.error or ''}")


def video_url(db_path: Path, video_id: str) -> str:
    with connect(db_path) as connection:
        row = connection.execute("SELECT url FROM video_jobs WHERE video_id = ?", (video_id,)).fetchone()
    if not row:
        raise RuntimeError(f"No DB row for video_id={video_id}")
    url = str(row["url"])
    if not url.startswith("http"):
        raise RuntimeError(f"No YouTube URL available for {video_id}. Current url field: {url}")
    return url


def run_gemini_stage(video_id: str, options: dict[str, Any]) -> StageResult:
    started_at = time.perf_counter()
    try:
        key = gemini_summary.api_key(options.get("gemini_api_key"))
        url = video_url(Path(options["db"]), video_id)
        prompt_template = gemini_summary.load_prompt(
            options.get("gemini_prompt"),
            Path(options["gemini_prompt_file"]) if options.get("gemini_prompt_file") else None,
        )
        result = gemini_summary.summarize_youtube(
            video_id=video_id,
            url=url,
            key=key,
            model=options["gemini_model"],
            prompt_template=prompt_template,
            output_root=Path(options["gemini_output_root"]),
            overwrite=options["force"],
            timeout=options["gemini_timeout"],
            temperature=options["gemini_temperature"],
            max_output_tokens=options["gemini_max_output_tokens"],
        )
        print(f"[gemini] {video_id}: {result.output_path}")
        return StageResult(video_id, "done", elapsed(started_at))
    except Exception as exc:
        return StageResult(video_id, "failed", elapsed(started_at), error=str(exc))


def merge_gemini_outputs(db_path: Path, output_root: Path, output: Path) -> int:
    with connect(db_path) as connection:
        video_ids = [
            row["video_id"]
            for row in connection.execute("SELECT video_id FROM video_jobs WHERE gemini_status = 'done' ORDER BY video_id")
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as merged:
        for video_id in video_ids:
            path = gemini_summary.output_path(video_id, output_root)
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            merged.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def merge_ocr_outputs(db_path: Path, output: Path) -> int:
    with connect(db_path) as connection:
        video_ids = [
            row["video_id"]
            for row in connection.execute("SELECT video_id FROM video_jobs WHERE ocr_status = 'done' ORDER BY video_id")
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as merged:
        for video_id in video_ids:
            path = DATASET_DIR / video_id / "ocr.jsonl"
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    merged.write(line + "\n")
                    count += 1
    return count


def print_summary(db_path: Path) -> None:
    with connect(db_path) as connection:
        total_urls = connection.execute("SELECT COUNT(*) AS count FROM url_jobs").fetchone()["count"]
        total_videos = connection.execute("SELECT COUNT(*) AS count FROM video_jobs").fetchone()["count"]
        print(f"[summary] urls={total_urls} videos={total_videos}")
        for stage in ("proxy", "scene", "keyframe", "ocr", "gemini"):
            column = f"{stage}_status"
            rows = connection.execute(f"SELECT {column} AS status, COUNT(*) AS count FROM video_jobs GROUP BY {column}")
            print(f"[summary] {stage}: " + ", ".join(f"{row['status']}={row['count']}" for row in rows))


def should_run(args: argparse.Namespace, stage: str) -> bool:
    start_index = STAGES.index(args.start_at)
    stop_index = STAGES.index(args.stop_after or STAGES[-1])
    stage_index = STAGES.index(stage)
    if stop_index < start_index:
        raise SystemExit("--stop-after must be the same as or later than --start-at.")
    return start_index <= stage_index <= stop_index


def main() -> None:
    args = parse_args()
    init_db(args.db, args.reset_db)
    urls = read_urls(args, required=args.start_at == "download")
    video_ids = read_ids(args)
    if urls:
        enqueue_urls(args.db, urls)
    if video_ids:
        enqueue_video_ids(args.db, video_ids)
    options = vars(args).copy()
    options["db"] = str(args.db)
    options["urls_file"] = None
    options["ids_file"] = None
    options["gemini_prompt_file"] = str(args.gemini_prompt_file) if args.gemini_prompt_file else None
    options["gemini_output_root"] = str(args.gemini_output_root)
    options["urls"] = []
    options["ids"] = []

    if should_run(args, "download"):
        run_downloads(args, options)
    if args.stop_after == "download":
        print_summary(args.db)
        return

    if should_run(args, "proxy"):
        run_process_stage(args, "proxy", args.proxy_workers, create_proxy, options)
    if args.stop_after == "proxy":
        print_summary(args.db)
        return

    if should_run(args, "scenes"):
        run_process_stage(args, "scene", args.scene_workers, run_scene_stage, options)
    if args.stop_after == "scenes":
        print_summary(args.db)
        return

    if should_run(args, "keyframes"):
        run_process_stage(args, "keyframe", args.keyframe_workers, run_keyframe_stage, options)
    if args.stop_after == "keyframes":
        print_summary(args.db)
        return

    if should_run(args, "ocr"):
        run_ocr_stage(args, options)
        merged = merge_ocr_outputs(args.db, args.merge_ocr_output)
        print(f"[merge] {merged} OCR record(s) -> {args.merge_ocr_output}")
    if args.stop_after == "ocr":
        print_summary(args.db)
        return

    if should_run(args, "gemini") and (args.gemini_summary or args.start_at == "gemini"):
        run_process_stage(args, "gemini", args.gemini_workers, run_gemini_stage, options)
        merged = merge_gemini_outputs(args.db, args.gemini_output_root, args.merge_gemini_output)
        print(f"[merge] {merged} Gemini summary record(s) -> {args.merge_gemini_output}")
    print_summary(args.db)


if __name__ == "__main__":
    main()
