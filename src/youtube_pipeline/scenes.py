from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg


ROOT = Path.cwd()
VIDEOS_DIR = ROOT / "videos"
SCENES_DIR = ROOT / "scenes"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect scenes in downloaded videos and split them into shorter clips."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help="YouTube video IDs to process, for example 3Nf1F1x2IDM. Defaults to all videos/* folders.",
    )
    parser.add_argument(
        "--copy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use fast copy mode for splitting instead of the default RTX/NVENC re-encode.",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use NVIDIA NVENC for scene clip encoding when not using --copy.",
    )
    parser.add_argument(
        "--min-scene-len",
        default=None,
        help="Minimum scene length, for example 2s or 00:00:02. Uses PySceneDetect default if omitted.",
    )
    parser.add_argument(
        "--metadata-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect scenes and write scenes/<id>/scenes.json without splitting mp4 files. Default: true.",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=4,
        help="Integer downscale factor for scene detection. Default: 4.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Frames to skip during scene detection. Default: 0.",
    )
    parser.add_argument(
        "--detector",
        choices=["adaptive", "content", "histogram"],
        default="adaptive",
        help="PySceneDetect detector to use. Default: adaptive.",
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=3.0,
        help="AdaptiveDetector adaptive threshold. Default: 3.0.",
    )
    parser.add_argument(
        "--min-content-val",
        type=float,
        default=15.0,
        help="AdaptiveDetector minimum content value. Default: 15.0.",
    )
    parser.add_argument(
        "--content-threshold",
        type=float,
        default=27.0,
        help="ContentDetector threshold. Default: 27.0.",
    )
    parser.add_argument(
        "--histogram-threshold",
        type=float,
        default=0.05,
        help="HistogramDetector threshold. Default: 0.05.",
    )
    return parser.parse_args()


def video_ids(requested_ids: list[str]) -> list[str]:
    if requested_ids:
        return requested_ids
    if not VIDEOS_DIR.exists():
        raise SystemExit("No videos directory found. Run youtube-pipeline first.")
    return sorted(path.name for path in VIDEOS_DIR.iterdir() if path.is_dir())


def videos_for_id(video_id: str) -> list[Path]:
    video_dir = VIDEOS_DIR / video_id
    if not video_dir.exists():
        raise SystemExit(f"Video folder not found: {video_dir}")
    videos = sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"No video files found in: {video_dir}")
    return videos


def ffmpeg_env() -> dict[str, str]:
    env = os.environ.copy()
    ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    return env


def split_video(video_path: Path, video_id: str, copy: bool, gpu: bool, min_scene_len: str | None) -> None:
    output_dir = SCENES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "scenedetect",
        "-i",
        str(video_path),
        "-o",
        str(output_dir),
    ]
    if min_scene_len:
        command.extend(["--min-scene-len", min_scene_len])
    command.extend(
        [
            "detect-adaptive",
            "split-video",
            "--filename",
            "scene-$SCENE_NUMBER",
        ]
    )
    if copy:
        command.append("--copy")
    elif gpu:
        command.extend(
            [
                "--args",
                "-map 0:v:0 -map 0:a? -map 0:s? -c:v h264_nvenc -preset p4 -cq 23 -c:a aac",
            ]
        )

    print(f"Detecting scenes: {video_path} -> {output_dir}")
    subprocess.run(command, env=ffmpeg_env(), check=True)


def scene_metadata_path(video_id: str) -> Path:
    return SCENES_DIR / video_id / "scenes.json"


def detect_scene_metadata(
    video_path: Path,
    video_id: str,
    min_scene_len: str | None,
    downscale: int,
    frame_skip: int,
    detector: str = "adaptive",
    adaptive_threshold: float = 3.0,
    min_content_val: float = 15.0,
    content_threshold: float = 27.0,
    histogram_threshold: float = 0.05,
) -> Path:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector, HistogramDetector

    output_dir = SCENES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.auto_downscale = False
    scene_manager.downscale = max(1, downscale)
    if detector == "adaptive":
        scene_detector = AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=min_scene_len or 15,
            min_content_val=min_content_val,
        )
    elif detector == "content":
        scene_detector = ContentDetector(threshold=content_threshold, min_scene_len=min_scene_len or 15)
    elif detector == "histogram":
        scene_detector = HistogramDetector(threshold=histogram_threshold, min_scene_len=min_scene_len or 15)
    else:
        raise ValueError(f"Unsupported scene detector: {detector}")
    scene_manager.add_detector(scene_detector)

    print(f"Detecting scene metadata ({detector}): {video_path} -> {scene_metadata_path(video_id)}")
    scene_manager.detect_scenes(video=video, frame_skip=max(0, frame_skip), show_progress=False)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        scene_list = [(video.base_timecode, video.duration)]

    fps = float(video.frame_rate)
    records = []
    for index, (start, end) in enumerate(scene_list, start=1):
        start_frame = int(start.frame_num)
        end_frame = max(int(end.frame_num), start_frame + 1)
        start_seconds = float(start.seconds)
        end_seconds = max(float(end.seconds), start_seconds)
        records.append(
            {
                "scene_id": f"scene-{index:03d}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "duration_seconds": round(max(end_seconds - start_seconds, (end_frame - start_frame) / fps), 3),
            }
        )

    payload = {
        "video_id": video_id,
        "video_path": str(video_path.relative_to(ROOT)),
        "fps": fps,
        "frame_size": list(video.frame_size),
        "downscale": max(1, downscale),
        "frame_skip": max(0, frame_skip),
        "detector": detector,
        "detector_options": {
            "adaptive_threshold": adaptive_threshold,
            "min_content_val": min_content_val,
            "content_threshold": content_threshold,
            "histogram_threshold": histogram_threshold,
            "min_scene_len": min_scene_len or 15,
        },
        "scenes": records,
    }
    output_path = scene_metadata_path(video_id)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{video_id}: wrote {len(records)} scene metadata record(s) -> {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    for video_id in video_ids(args.ids):
        for video_path in videos_for_id(video_id):
            if args.metadata_only:
                detect_scene_metadata(
                    video_path,
                    video_id,
                    min_scene_len=args.min_scene_len,
                    downscale=args.downscale,
                    frame_skip=args.frame_skip,
                    detector=args.detector,
                    adaptive_threshold=args.adaptive_threshold,
                    min_content_val=args.min_content_val,
                    content_threshold=args.content_threshold,
                    histogram_threshold=args.histogram_threshold,
                )
            else:
                split_video(video_path, video_id, copy=args.copy, gpu=args.gpu, min_scene_len=args.min_scene_len)


if __name__ == "__main__":
    main()
