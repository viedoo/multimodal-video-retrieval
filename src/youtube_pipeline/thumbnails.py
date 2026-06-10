from __future__ import annotations

from youtube_pipeline.keyframes import THUMBNAILS_DIR, parse_args, run


def main() -> None:
    args = parse_args()
    args.min_frames = 1
    args.max_frames = 1
    args.image_format = "webp"
    args.fixed_output_name = "thumbnail.webp"
    run(args, output_root=THUMBNAILS_DIR, label="thumbnails")


if __name__ == "__main__":
    main()
