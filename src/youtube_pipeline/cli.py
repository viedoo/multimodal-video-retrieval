from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
from yt_dlp import YoutubeDL

from youtube_pipeline.translate_subtitles import subtitle_files, translate_file


ROOT = Path.cwd()
VIDEOS_DIR = ROOT / "videos"
SUBTITLES_DIR = ROOT / "subtitles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos and subtitles into YouTube-ID folders."
    )
    parser.add_argument("urls", nargs="*", help="YouTube video or playlist URLs")
    parser.add_argument(
        "--urls-file",
        type=Path,
        help="Text file containing one YouTube URL per line; blank lines and # comments are ignored",
    )
    parser.add_argument(
        "--no-auto-subs",
        action="store_true",
        help="Only download creator-provided subtitles, not auto-generated subtitles",
    )
    parser.add_argument(
        "--sub-langs",
        default="vi.*,en.*",
        help="Comma-separated subtitle languages to download; default: vi.*,en.*",
    )
    parser.add_argument(
        "--no-translate-subtitles",
        action="store_true",
        help="Do not auto-translate downloaded subtitle files to English.",
    )
    parser.add_argument(
        "--translate-server-url",
        default="http://localhost:5000",
        help="LibreTranslate server URL for automatic subtitle translation.",
    )
    parser.add_argument("--translate-api-key", default=None, help="LibreTranslate API key, if required.")
    parser.add_argument("--translate-source-lang", default="auto", help="Subtitle source language. Default: auto")
    parser.add_argument("--translate-target-lang", default="en", help="Subtitle target language. Default: en")
    parser.add_argument(
        "--translate-batch-chars",
        type=int,
        default=4500,
        help="Maximum characters per LibreTranslate request. Default: 4500",
    )
    return parser.parse_args()


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


def subtitle_languages(value: str) -> list[str]:
    languages = [language.strip() for language in value.split(",") if language.strip()]
    if not languages:
        raise SystemExit("--sub-langs must include at least one language.")
    return languages


def downloaded_video_ids(info: dict) -> list[str]:
    if "entries" not in info:
        video_id = info.get("id")
        return [video_id] if isinstance(video_id, str) else []

    ids: list[str] = []
    for entry in info.get("entries") or []:
        if not entry:
            continue
        if "entries" in entry:
            ids.extend(downloaded_video_ids(entry))
            continue
        video_id = entry.get("id")
        if isinstance(video_id, str):
            ids.append(video_id)
    return ids


def download(url: str, include_auto_subs: bool, sub_langs: list[str]) -> list[str]:
    options = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": {
            "default": str(VIDEOS_DIR / "%(id)s" / "video.%(ext)s"),
            "subtitle": str(SUBTITLES_DIR / "%(id)s" / "subtitles.%(ext)s"),
        },
        "writesubtitles": True,
        "writeautomaticsub": include_auto_subs,
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

    print(f"Downloading {url} -> videos/<youtube_id> and subtitles/<youtube_id>")
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    return downloaded_video_ids(info or {})


def organize_downloaded_subtitles(video_ids: list[str]) -> None:
    for video_id in sorted(set(video_ids)):
        video_dir = VIDEOS_DIR / video_id
        if not video_dir.exists():
            continue
        subtitle_dir = SUBTITLES_DIR / video_id
        for path in video_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".srt", ".vtt"}:
                continue
            subtitle_dir.mkdir(parents=True, exist_ok=True)
            target = subtitle_dir / path.name.replace("video.", "subtitles.", 1)
            path.replace(target)


def translate_downloaded_subtitles(video_ids: list[str], args: argparse.Namespace) -> None:
    translate_args = argparse.Namespace(
        server_url=args.translate_server_url,
        api_key=args.translate_api_key,
        source_lang=args.translate_source_lang,
        target_lang=args.translate_target_lang,
        overwrite=False,
        batch_chars=args.translate_batch_chars,
    )
    extensions = {".srt", ".vtt"}

    for video_id in sorted(set(video_ids)):
        if not (SUBTITLES_DIR / video_id).exists():
            print(f"No subtitle folder to translate: {SUBTITLES_DIR / video_id}")
            continue
        for path in subtitle_files(video_id, extensions, args.translate_target_lang):
            translate_file(path, translate_args)


def main() -> None:
    args = parse_args()
    urls = read_urls(args)
    sub_langs = subtitle_languages(args.sub_langs)
    for url in urls:
        video_ids = download(url, include_auto_subs=not args.no_auto_subs, sub_langs=sub_langs)
        organize_downloaded_subtitles(video_ids)
        if not args.no_translate_subtitles:
            translate_downloaded_subtitles(video_ids, args)


if __name__ == "__main__":
    main()
