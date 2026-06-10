from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path.cwd()
SCENES_DIR = ROOT / "scenes"
KEYFRAMES_DIR = ROOT / "keyframes"
THUMBNAILS_DIR = ROOT / "thumbnails"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
SCENE_NUMBER_PATTERN = re.compile(r"scene-(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Candidate:
    frame_index: int
    image: np.ndarray
    histogram: np.ndarray
    flow_histogram: np.ndarray
    edge_density: float
    sharpness: float
    brightness: float
    motion_score: float

    @property
    def quality(self) -> float:
        brightness_score = 1.0 - min(abs(self.brightness - 127.5) / 127.5, 1.0)
        return self.sharpness * (0.5 + brightness_score)


@dataclass(frozen=True)
class SceneRange:
    scene_id: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract diverse representative keyframes from scene clips."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help="YouTube video IDs to process, for example 3Nf1F1x2IDM. Defaults to all scenes/* folders.",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=8.0,
        help="Linear scaling factor for keyframe count. N = duration / this value.",
    )
    parser.add_argument("--min-frames", type=int, default=1, help="Minimum keyframes per scene.")
    parser.add_argument("--max-frames", type=int, default=8, help="Maximum keyframes per scene.")
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=0.10,
        help="Fraction of each scene to ignore at the start and end.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=12,
        help="How many candidate frames to inspect for each requested keyframe.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["thumbnail", "action", "hybrid", "flow"],
        default="flow",
        help="Keyframe strategy: thumbnail keeps the quality-first selector, action favors motion/change, hybrid balances both, flow favors dense optical flow.",
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=0.55,
        help="Motion/change weight for action and hybrid selection modes.",
    )
    parser.add_argument(
        "--min-temporal-gap",
        type=float,
        default=0.16,
        help="Minimum fraction of a clip between selected keyframes.",
    )
    parser.add_argument(
        "--diverse-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select keyframes from different temporal windows before global fallback.",
    )
    parser.add_argument(
        "--unique-threshold",
        type=float,
        default=0.50,
        help="Minimum HSV histogram distance between selected keyframes.",
    )
    parser.add_argument(
        "--min-edge-delta",
        type=float,
        default=0.006,
        help="Minimum edge-density difference for frames with similar colors.",
    )
    parser.add_argument(
        "--allow-duplicate-fill",
        action="store_true",
        help="Fill up to N with near-duplicates if a scene lacks enough unique frames.",
    )
    parser.add_argument(
        "--reject-transition-blur",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject frames that are much blurrier than nearby frames, which often happens around scene changes.",
    )
    parser.add_argument(
        "--blur-neighbor-frames",
        type=int,
        default=2,
        help="Neighbor distance used for transition-blur checks.",
    )
    parser.add_argument(
        "--blur-ratio",
        type=float,
        default=0.65,
        help="Reject a candidate when sharpness is below this ratio of the sharpest nearby frame.",
    )
    parser.add_argument("--min-sharpness", type=float, default=35.0, help="Minimum Laplacian variance.")
    parser.add_argument("--min-brightness", type=float, default=25.0, help="Minimum mean luminance.")
    parser.add_argument("--max-brightness", type=float, default=235.0, help="Maximum mean luminance.")
    parser.add_argument(
        "--motion-method",
        choices=["absdiff", "farneback"],
        default="farneback",
        help="Motion score method. farneback uses dense optical flow. Default: farneback.",
    )
    parser.add_argument("--flow-histogram-bins", type=int, default=8, help="Gradient histogram bins for optical flow.")
    parser.add_argument(
        "--flow-unique-threshold",
        type=float,
        default=0.20,
        help="Minimum optical-flow gradient histogram distance between selected frames. Default: 0.20.",
    )
    parser.add_argument("--flow-pyr-scale", type=float, default=0.5)
    parser.add_argument("--flow-levels", type=int, default=3)
    parser.add_argument("--flow-window-size", type=int, default=15)
    parser.add_argument("--flow-iterations", type=int, default=3)
    parser.add_argument("--flow-poly-n", type=int, default=5)
    parser.add_argument("--flow-poly-sigma", type=float, default=1.2)
    parser.add_argument(
        "--analysis-height",
        type=int,
        default=240,
        help="Resize frames to this height before quality, histogram, and optical-flow analysis. Use 0 to analyze full resolution. Default: 240.",
    )
    parser.add_argument(
        "--decode-mode",
        choices=["seek", "sequential"],
        default="seek",
        help="Frame access strategy. sequential avoids repeated random seeks only when candidate gaps are small. Default: seek.",
    )
    parser.add_argument("--height", type=int, default=360, help="Thumbnail height in pixels.")
    parser.add_argument("--webp-quality", type=int, default=80, help="Output WebP quality from 0 to 100.")
    parser.add_argument("--jpg-quality", type=int, default=2, help="Output JPG quality from 2 to 31, lower is better.")
    parser.add_argument(
        "--image-format",
        choices=["jpg", "webp"],
        default="jpg",
        help="Output image format for keyframes. Default: jpg",
    )
    parser.add_argument(
        "--short-scene-seconds",
        type=float,
        default=5.0,
        help="Scenes shorter than this use the short-scene strategy. Default: 5.",
    )
    parser.add_argument(
        "--short-scene-strategy",
        choices=["first", "middle", "diverse"],
        default="diverse",
        help="Short scene keyframe strategy. diverse samples a few cheap candidates and avoids duplicates. Default: diverse.",
    )
    parser.add_argument(
        "--short-scene-candidates",
        type=int,
        default=3,
        help="Number of candidate positions for short scenes when strategy=diverse. Default: 3.",
    )
    parser.add_argument(
        "--cross-scene-unique",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Avoid keyframes too similar to nearby previous scenes. Default: true.",
    )
    parser.add_argument(
        "--cross-scene-lookback",
        type=int,
        default=8,
        help="Number of previous selected keyframes to compare for cross-scene uniqueness. Default: 8.",
    )
    parser.add_argument(
        "--cross-scene-color-threshold",
        type=float,
        default=0.25,
        help="Minimum histogram distance from recent scene keyframes. Default: 0.25.",
    )
    parser.add_argument(
        "--skip-similar-scenes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a scene if all candidate keyframes are too similar to recent scenes. Default: true.",
    )
    parser.add_argument(
        "--from-scene-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read scenes/<id>/scenes.json and extract keyframes directly from the original video. Default: true.",
    )
    return parser.parse_args()


def scene_ids(requested_ids: list[str]) -> list[str]:
    if requested_ids:
        return requested_ids
    if not SCENES_DIR.exists():
        raise SystemExit("No scenes directory found. Run scene-pipeline first.")
    return sorted(path.name for path in SCENES_DIR.iterdir() if path.is_dir())


def scene_clips(scene_id: str) -> list[Path]:
    scene_dir = SCENES_DIR / scene_id
    if not scene_dir.exists():
        raise SystemExit(f"Scene folder not found: {scene_dir}")
    clips = sorted(path for path in scene_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not clips:
        raise SystemExit(f"No scene clips found in: {scene_dir}")
    return clips


def scene_metadata_path(video_id: str) -> Path:
    return SCENES_DIR / video_id / "scenes.json"


def load_scene_metadata(video_id: str) -> tuple[Path, list[SceneRange]]:
    path = scene_metadata_path(video_id)
    if not path.exists():
        raise SystemExit(f"Scene metadata not found: {path}. Run scene-pipeline first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    video_path = ROOT / payload["video_path"]
    scenes = [
        SceneRange(
            scene_id=str(item["scene_id"]),
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
            start_seconds=float(item["start_seconds"]),
            end_seconds=float(item["end_seconds"]),
            duration_seconds=float(item["duration_seconds"]),
        )
        for item in payload.get("scenes", [])
    ]
    if not scenes:
        raise SystemExit(f"No scenes in metadata: {path}")
    return video_path, scenes


def keyframe_count(duration_seconds: float, sample_every_seconds: float, minimum: int, maximum: int) -> int:
    if sample_every_seconds <= 0:
        raise SystemExit("--sample-every-seconds must be greater than 0.")
    return max(minimum, min(maximum, math.ceil(duration_seconds / sample_every_seconds)))


def sample_indices(total_frames: int, fps: float, args: argparse.Namespace) -> list[int]:
    duration = total_frames / fps
    target_count = keyframe_count(duration, args.sample_every_seconds, args.min_frames, args.max_frames)
    candidate_count = max(target_count * args.candidate_multiplier, target_count, 1)

    start = int(total_frames * args.edge_margin)
    end = int(total_frames * (1.0 - args.edge_margin)) - 1
    if start >= end:
        start, end = 0, max(total_frames - 1, 0)

    return sorted(set(int(round(index)) for index in np.linspace(start, end, candidate_count)))


def frame_histogram(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def normalized_histogram(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32).flatten()
    total = float(values.sum())
    if total <= 0:
        return values
    return values / total


def gray_frame(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def analysis_image(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    height = int(getattr(args, "analysis_height", 0) or 0)
    if height <= 0:
        return image
    return resize_to_height(image, height)


def frame_quality(gray: np.ndarray) -> tuple[float, float, float]:
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    edge_density = float((cv2.Canny(gray, 100, 200) > 0).mean())
    return sharpness, brightness, edge_density


def motion_score(previous_gray: np.ndarray | None, current_gray: np.ndarray) -> float:
    if previous_gray is None:
        return 0.0
    return float(cv2.absdiff(previous_gray, current_gray).mean())


def empty_flow_histogram(args: argparse.Namespace) -> np.ndarray:
    return np.zeros(max(int(args.flow_histogram_bins), 1), dtype=np.float32)


def flow_gradient_features(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    if previous_gray is None or args.motion_method != "farneback":
        return empty_flow_histogram(args), 0.0

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        pyr_scale=args.flow_pyr_scale,
        levels=args.flow_levels,
        winsize=args.flow_window_size,
        iterations=args.flow_iterations,
        poly_n=args.flow_poly_n,
        poly_sigma=args.flow_poly_sigma,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    grad_x = cv2.Sobel(magnitude, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(magnitude, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude, grad_angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
    bins = max(int(args.flow_histogram_bins), 1)
    histogram, _ = np.histogram(
        grad_angle,
        bins=bins,
        range=(0.0, float(2.0 * np.pi)),
        weights=grad_magnitude,
    )
    flow_strength = float(np.percentile(magnitude, 90) + grad_magnitude.mean())
    return normalized_histogram(histogram), flow_strength


def candidate_motion_features(
    previous_gray: np.ndarray | None,
    current_gray: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    if args.motion_method == "farneback":
        return flow_gradient_features(previous_gray, current_gray, args)
    return empty_flow_histogram(args), motion_score(previous_gray, current_gray)


def read_frame_at(capture: cv2.VideoCapture, frame_index: int, args: argparse.Namespace) -> np.ndarray | None:
    if getattr(args, "decode_mode", "seek") == "sequential":
        current = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if current <= frame_index:
            while current < frame_index:
                if not capture.grab():
                    return None
                current += 1
            ok, image = capture.read()
            return image if ok else None

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = capture.read()
    return image if ok else None


def read_gray_at(capture: cv2.VideoCapture, frame_index: int, args: argparse.Namespace) -> np.ndarray | None:
    image = read_frame_at(capture, frame_index, args)
    if image is None:
        return None
    return gray_frame(analysis_image(image, args))


def is_transition_blur(
    capture: cv2.VideoCapture,
    frame_index: int,
    total_frames: int,
    sharpness: float,
    args: argparse.Namespace,
) -> bool:
    if not args.reject_transition_blur or args.blur_neighbor_frames <= 0:
        return False

    neighbor_sharpness: list[float] = []
    for neighbor_index in (
        frame_index - args.blur_neighbor_frames,
        frame_index + args.blur_neighbor_frames,
    ):
        if neighbor_index < 0 or neighbor_index >= total_frames:
            continue
        neighbor_gray = read_gray_at(capture, neighbor_index, args)
        if neighbor_gray is not None:
            neighbor_sharpness.append(frame_quality(neighbor_gray)[0])

    return bool(neighbor_sharpness and sharpness < max(neighbor_sharpness) * args.blur_ratio)


def candidates_for_clip(clip: Path, args: argparse.Namespace) -> tuple[list[Candidate], int]:
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open scene clip: {clip}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise SystemExit(f"Unable to read frame count for: {clip}")

    indices = sample_indices(total_frames, fps, args)
    duration = total_frames / fps
    target_count = keyframe_count(duration, args.sample_every_seconds, args.min_frames, args.max_frames)
    candidates: list[Candidate] = []
    fallback_candidates: list[Candidate] = []
    previous_gray: np.ndarray | None = None

    for frame_index in indices:
        image = read_frame_at(capture, frame_index, args)
        if image is None:
            continue
        feature_image = analysis_image(image, args)
        gray = gray_frame(feature_image)
        sharpness, brightness, edge_density = frame_quality(gray)
        flow_histogram, motion = candidate_motion_features(previous_gray, gray, args)
        candidate = Candidate(
            frame_index=frame_index,
            image=image,
            histogram=frame_histogram(feature_image),
            flow_histogram=flow_histogram,
            edge_density=edge_density,
            sharpness=sharpness,
            brightness=brightness,
            motion_score=motion,
        )
        previous_gray = gray
        fallback_candidates.append(candidate)
        if (
            sharpness < args.min_sharpness
            or brightness < args.min_brightness
            or brightness > args.max_brightness
            or is_transition_blur(capture, frame_index, total_frames, sharpness, args)
        ):
            continue
        candidates.append(candidate)

    capture.release()
    return candidates or fallback_candidates, target_count


def sample_indices_for_range(scene: SceneRange, fps: float, args: argparse.Namespace) -> tuple[list[int], int]:
    total_frames = max(scene.end_frame - scene.start_frame, 1)
    if scene.duration_seconds < args.short_scene_seconds:
        if args.short_scene_strategy == "first":
            return [scene.start_frame], 1
        if args.short_scene_strategy == "middle":
            return [scene.start_frame + total_frames // 2], 1

        candidate_count = max(int(args.short_scene_candidates), 1)
        start = scene.start_frame
        end = max(scene.end_frame - 1, scene.start_frame)
        return sorted(set(int(round(index)) for index in np.linspace(start, end, candidate_count))), 1

    target_count = keyframe_count(scene.duration_seconds, args.sample_every_seconds, args.min_frames, args.max_frames)
    candidate_count = max(target_count * args.candidate_multiplier, target_count, 1)

    start = scene.start_frame + int(total_frames * args.edge_margin)
    end = scene.start_frame + int(total_frames * (1.0 - args.edge_margin)) - 1
    if start >= end:
        start, end = scene.start_frame, max(scene.end_frame - 1, scene.start_frame)

    return sorted(set(int(round(index)) for index in np.linspace(start, end, candidate_count))), target_count


def candidates_for_scene_range(
    capture: cv2.VideoCapture,
    scene: SceneRange,
    fps: float,
    total_video_frames: int,
    args: argparse.Namespace,
) -> tuple[list[Candidate], int]:
    indices, target_count = sample_indices_for_range(scene, fps, args)
    candidates: list[Candidate] = []
    fallback_candidates: list[Candidate] = []
    previous_gray: np.ndarray | None = None

    for frame_index in indices:
        image = read_frame_at(capture, frame_index, args)
        if image is None:
            continue
        feature_image = analysis_image(image, args)
        gray = gray_frame(feature_image)
        sharpness, brightness, edge_density = frame_quality(gray)
        flow_histogram, motion = candidate_motion_features(previous_gray, gray, args)
        candidate = Candidate(
            frame_index=frame_index,
            image=image,
            histogram=frame_histogram(feature_image),
            flow_histogram=flow_histogram,
            edge_density=edge_density,
            sharpness=sharpness,
            brightness=brightness,
            motion_score=motion,
        )
        previous_gray = gray
        fallback_candidates.append(candidate)
        if scene.duration_seconds < args.short_scene_seconds:
            candidates.append(candidate)
            continue
        if (
            sharpness < args.min_sharpness
            or brightness < args.min_brightness
            or brightness > args.max_brightness
            or is_transition_blur(capture, frame_index, total_video_frames, sharpness, args)
        ):
            continue
        candidates.append(candidate)

    return candidates or fallback_candidates, target_count


def histogram_distance(left: Candidate, right: Candidate) -> float:
    return float(cv2.compareHist(left.histogram, right.histogram, cv2.HISTCMP_BHATTACHARYYA))


def flow_histogram_distance(left: Candidate, right: Candidate) -> float:
    if left.flow_histogram.size == 0 or right.flow_histogram.size == 0:
        return 1.0
    return float(cv2.compareHist(left.flow_histogram.astype(np.float32), right.flow_histogram.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


def is_unique(
    candidate: Candidate,
    selected: list[Candidate],
    unique_threshold: float,
    min_edge_delta: float,
    flow_unique_threshold: float | None = None,
) -> bool:
    for current in selected:
        color_distance = histogram_distance(candidate, current)
        edge_delta = abs(candidate.edge_density - current.edge_density)
        if color_distance < unique_threshold and edge_delta < min_edge_delta:
            return False
        if flow_unique_threshold is not None:
            flow_distance = flow_histogram_distance(candidate, current)
            if flow_distance < flow_unique_threshold and color_distance < unique_threshold:
                return False
    return True


def is_candidate_unique(candidate: Candidate, selected: list[Candidate], args: argparse.Namespace) -> bool:
    return is_unique(
        candidate,
        selected,
        args.unique_threshold,
        args.min_edge_delta,
        args.flow_unique_threshold if args.motion_method == "farneback" else None,
    )


def recent_selected(selected: list[Candidate], lookback: int) -> list[Candidate]:
    if lookback <= 0:
        return selected
    return selected[-lookback:]


def is_globally_unique(candidate: Candidate, global_selected: list[Candidate], args: argparse.Namespace) -> bool:
    if not getattr(args, "cross_scene_unique", True):
        return True
    comparison_set = recent_selected(global_selected, args.cross_scene_lookback)
    for current in comparison_set:
        if histogram_distance(candidate, current) < args.cross_scene_color_threshold:
            return False
    return is_candidate_unique(candidate, comparison_set, args)


def diversity_distance(candidate: Candidate, selected: list[Candidate]) -> float:
    if not selected:
        return 1.0
    return min(histogram_distance(candidate, current) for current in selected)


def choose_best_global_candidate(
    candidates: list[Candidate],
    global_selected: list[Candidate],
    args: argparse.Namespace,
) -> Candidate | None:
    ranked = ranked_candidates(candidates, args)
    for candidate in ranked:
        if is_globally_unique(candidate, global_selected, args):
            return candidate

    if global_selected and args.skip_similar_scenes:
        return None

    comparison_set = recent_selected(global_selected, args.cross_scene_lookback)
    return max(
        ranked,
        key=lambda candidate: (
            diversity_distance(candidate, comparison_set),
            abs(candidate.edge_density - np.mean([current.edge_density for current in comparison_set])) if comparison_set else 0.0,
            candidate.quality,
        ),
    )


def normalized(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return min(max(value / maximum, 0.0), 1.0)


def selection_score(candidate: Candidate, max_quality: float, max_motion: float, args: argparse.Namespace) -> float:
    quality = normalized(candidate.quality, max_quality)
    motion = normalized(candidate.motion_score, max_motion)
    if args.selection_mode == "flow":
        return (0.75 * motion) + (0.25 * quality)
    if args.selection_mode == "action":
        return (args.motion_weight * motion) + ((1.0 - args.motion_weight) * quality)
    if args.selection_mode == "hybrid":
        return (0.5 * quality) + (args.motion_weight * 0.5 * motion)
    return candidate.quality


def ranked_candidates(candidates: list[Candidate], args: argparse.Namespace) -> list[Candidate]:
    max_quality = max((candidate.quality for candidate in candidates), default=0.0)
    max_motion = max((candidate.motion_score for candidate in candidates), default=0.0)
    return sorted(
        candidates,
        key=lambda candidate: selection_score(candidate, max_quality, max_motion, args),
        reverse=True,
    )


def is_temporally_spaced(candidate: Candidate, selected: list[Candidate], total_span: int, min_temporal_gap: float) -> bool:
    if not selected:
        return True
    min_gap = max(1, int(total_span * min_temporal_gap))
    return all(abs(candidate.frame_index - current.frame_index) >= min_gap for current in selected)


def temporal_window_candidates(candidates: list[Candidate], target_count: int) -> list[list[Candidate]]:
    if target_count <= 1:
        return [candidates]

    first = min(candidate.frame_index for candidate in candidates)
    last = max(candidate.frame_index for candidate in candidates)
    span = max(last - first + 1, 1)
    windows: list[list[Candidate]] = [[] for _ in range(target_count)]
    for candidate in candidates:
        window_index = min(int((candidate.frame_index - first) / span * target_count), target_count - 1)
        windows[window_index].append(candidate)
    return windows


def choose_from_ranked(
    ranked: list[Candidate],
    selected: list[Candidate],
    total_span: int,
    args: argparse.Namespace,
) -> Candidate | None:
    selected_indexes = {candidate.frame_index for candidate in selected}
    for candidate in ranked:
        if candidate.frame_index in selected_indexes:
            continue
        if not is_candidate_unique(candidate, selected, args):
            continue
        if not is_temporally_spaced(candidate, selected, total_span, args.min_temporal_gap):
            continue
        return candidate
    return None


def select_keyframes(candidates: list[Candidate], target_count: int, args: argparse.Namespace) -> list[Candidate]:
    total_span = max(candidate.frame_index for candidate in candidates) - min(candidate.frame_index for candidate in candidates)
    ranked = ranked_candidates(candidates, args)
    selected: list[Candidate] = []

    if args.diverse_windows and target_count > 1:
        for window in temporal_window_candidates(candidates, target_count):
            candidate = choose_from_ranked(ranked_candidates(window, args), selected, total_span, args)
            if candidate is not None:
                selected.append(candidate)
                if len(selected) == target_count:
                    break

    for candidate in ranked:
        if is_candidate_unique(candidate, selected, args) and is_temporally_spaced(
            candidate,
            selected,
            total_span,
            args.min_temporal_gap,
        ):
            selected.append(candidate)
            if len(selected) == target_count:
                break

    if args.allow_duplicate_fill and len(selected) < target_count:
        seen = {candidate.frame_index for candidate in selected}
        for candidate in ranked:
            if candidate.frame_index not in seen:
                selected.append(candidate)
                seen.add(candidate.frame_index)
                if len(selected) == target_count:
                    break

    return sorted(selected, key=lambda candidate: candidate.frame_index)


def select_keyframes_with_global_diversity(
    candidates: list[Candidate],
    target_count: int,
    args: argparse.Namespace,
    global_selected: list[Candidate],
) -> list[Candidate]:
    if not candidates:
        return []

    total_span = max(candidate.frame_index for candidate in candidates) - min(candidate.frame_index for candidate in candidates)
    ranked = ranked_candidates(candidates, args)
    selected: list[Candidate] = []

    for candidate in ranked:
        if not is_candidate_unique(candidate, selected, args):
            continue
        if not is_temporally_spaced(candidate, selected, total_span, args.min_temporal_gap):
            continue
        if not is_globally_unique(candidate, global_selected + selected, args):
            continue
        selected.append(candidate)
        if len(selected) == target_count:
            break

    if not selected:
        candidate = choose_best_global_candidate(candidates, global_selected, args)
        if candidate is not None:
            selected.append(candidate)

    if args.allow_duplicate_fill and len(selected) < target_count:
        seen = {candidate.frame_index for candidate in selected}
        for candidate in ranked:
            if candidate.frame_index not in seen:
                selected.append(candidate)
                seen.add(candidate.frame_index)
                if len(selected) == target_count:
                    break

    return sorted(selected, key=lambda candidate: candidate.frame_index)


def output_dir_for_clip(scene_id: str, clip: Path, output_root: Path) -> Path:
    match = SCENE_NUMBER_PATTERN.search(clip.stem)
    name = f"scene-{match.group(1)}" if match else clip.stem
    return output_root / scene_id / name


def validate_height(height: int) -> None:
    if height <= 0:
        raise SystemExit("--height must be greater than 0.")


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    validate_height(height)
    current_height, current_width = image.shape[:2]
    if current_height == height:
        return image
    width = max(2, int(round(current_width * (height / current_height))))
    if width % 2:
        width += 1
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def write_candidate_image(image: np.ndarray, output_file: Path, args: argparse.Namespace) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    resized = resize_to_height(image, args.height)
    suffix = output_file.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        quality = max(1, min(100, 101 - int(args.jpg_quality)))
        ok = cv2.imwrite(str(output_file), resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif suffix == ".webp":
        ok = cv2.imwrite(str(output_file), resized, [cv2.IMWRITE_WEBP_QUALITY, int(args.webp_quality)])
    else:
        ok = cv2.imwrite(str(output_file), resized)
    if not ok:
        raise SystemExit(f"Unable to write image: {output_file}")


def write_frame_image(clip: Path, frame_id: int, output_file: Path, args: argparse.Namespace) -> None:
    validate_height(args.height)
    frame_filter = f"select=eq(n\\,{frame_id}),scale=-2:{args.height}"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(clip),
        "-vf",
        frame_filter,
        "-frames:v",
        "1",
    ]
    if output_file.suffix.lower() in {".jpg", ".jpeg"}:
        command.extend(["-q:v", str(args.jpg_quality)])
    else:
        command.extend(["-c:v", "libwebp", "-quality", str(args.webp_quality), "-compression_level", "6"])
    command.append(str(output_file))
    subprocess.run(command, check=True)


def extract_keyframes(clip: Path, scene_id: str, args: argparse.Namespace, output_root: Path = KEYFRAMES_DIR, label: str = "keyframes") -> int:
    candidates, target_count = candidates_for_clip(clip, args)
    if not candidates:
        print(f"No usable keyframes found for: {clip}")
        return 0

    selected = select_keyframes(candidates, target_count, args)
    output_dir = output_dir_for_clip(scene_id, clip, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in selected:
        output_file = output_dir / getattr(args, "fixed_output_name", f"frame-{candidate.frame_index:06d}.{args.image_format}")
        write_candidate_image(candidate.image, output_file, args)

    print(f"{clip.name}: wrote {len(selected)} {label} -> {output_dir}")
    return len(selected)


def extract_keyframes_from_metadata(
    video_path: Path,
    video_id: str,
    scene: SceneRange,
    args: argparse.Namespace,
    global_selected: list[Candidate],
    output_root: Path = KEYFRAMES_DIR,
    label: str = "keyframes",
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        return extract_keyframes_from_metadata_capture(
            capture,
            video_path,
            video_id,
            scene,
            args,
            global_selected,
            fps,
            total_video_frames,
            output_root=output_root,
            label=label,
        )
    finally:
        capture.release()


def extract_keyframes_from_metadata_capture(
    capture: cv2.VideoCapture,
    video_path: Path,
    video_id: str,
    scene: SceneRange,
    args: argparse.Namespace,
    global_selected: list[Candidate],
    fps: float,
    total_video_frames: int,
    output_root: Path = KEYFRAMES_DIR,
    label: str = "keyframes",
) -> int:
    candidates, target_count = candidates_for_scene_range(capture, scene, fps, total_video_frames, args)
    if not candidates:
        print(f"No usable keyframes found for: {video_path} {scene.scene_id}")
        return 0

    if scene.duration_seconds < args.short_scene_seconds:
        candidate = choose_best_global_candidate(candidates, global_selected, args)
        selected = [candidate] if candidate is not None else []
    else:
        selected = select_keyframes_with_global_diversity(candidates, target_count, args, global_selected)
    if not selected:
        print(f"{scene.scene_id}: skipped similar {label}")
        return 0

    output_dir = output_root / video_id / scene.scene_id
    output_dir.mkdir(parents=True, exist_ok=True)

    for candidate in selected:
        output_file = output_dir / getattr(args, "fixed_output_name", f"frame-{candidate.frame_index:06d}.{args.image_format}")
        write_candidate_image(candidate.image, output_file, args)

    global_selected.extend(selected)
    print(f"{scene.scene_id}: wrote {len(selected)} {label} -> {output_dir}")
    return len(selected)


def run_from_metadata(args: argparse.Namespace, output_root: Path = KEYFRAMES_DIR, label: str = "keyframes") -> None:
    for video_id in scene_ids(args.ids):
        total = 0
        global_selected: list[Candidate] = []
        video_path, scene_ranges = load_scene_metadata(video_id)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise SystemExit(f"Unable to open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            for scene in scene_ranges:
                total += extract_keyframes_from_metadata_capture(
                    capture,
                    video_path,
                    video_id,
                    scene,
                    args,
                    global_selected,
                    fps,
                    total_video_frames,
                    output_root=output_root,
                    label=label,
                )
        finally:
            capture.release()
        print(f"video_id {video_id}: wrote {total} {label}")


def run(args: argparse.Namespace, output_root: Path = KEYFRAMES_DIR, label: str = "keyframes") -> None:
    if getattr(args, "from_scene_metadata", True):
        run_from_metadata(args, output_root=output_root, label=label)
        return
    for scene_id in scene_ids(args.ids):
        total = 0
        for clip in scene_clips(scene_id):
            total += extract_keyframes(clip, scene_id, args, output_root=output_root, label=label)
        print(f"video_id {scene_id}: wrote {total} {label}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
