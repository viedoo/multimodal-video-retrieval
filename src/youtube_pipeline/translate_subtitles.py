from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path.cwd()
SUBTITLES_DIR = ROOT / "subtitles"
TIMESTAMP_PATTERN = re.compile(r"(-->|^\d\d:\d\d:\d\d[,.]\d\d\d)")
LANG_SUFFIX_PATTERN = re.compile(r"(?P<prefix>.*?)(?P<lang>[a-z]{2,3}(?:-[A-Za-z0-9]+)?)$", re.IGNORECASE)
DELIMITER = "\n\n<<<YOUTUBE_PIPELINE_SUBTITLE_BREAK>>>\n\n"


@dataclass(frozen=True)
class SubtitleBlock:
    prefix: list[str]
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate downloaded subtitle files to English with LibreTranslate."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help="YouTube video IDs to process. Defaults to all subtitles/* folders.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:5000",
        help="LibreTranslate server URL. Default: http://localhost:5000",
    )
    parser.add_argument("--api-key", default=None, help="LibreTranslate API key, if required.")
    parser.add_argument("--source-lang", default="auto", help="Source language. Default: auto")
    parser.add_argument("--target-lang", default="en", help="Target language. Default: en")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing translated subtitle files.",
    )
    parser.add_argument(
        "--batch-chars",
        type=int,
        default=4500,
        help="Maximum characters per translation request. Default: 4500",
    )
    parser.add_argument(
        "--extensions",
        default=".srt,.vtt",
        help="Comma-separated subtitle extensions to translate. Default: .srt,.vtt",
    )
    return parser.parse_args()


def subtitle_ids(requested_ids: list[str]) -> list[str]:
    if requested_ids:
        return requested_ids
    if not SUBTITLES_DIR.exists():
        raise SystemExit("No subtitles directory found. Run youtube-pipeline first.")
    return sorted(path.name for path in SUBTITLES_DIR.iterdir() if path.is_dir())


def subtitle_files(video_id: str, extensions: set[str], target_lang: str) -> list[Path]:
    subtitle_dir = SUBTITLES_DIR / video_id
    if not subtitle_dir.exists():
        raise SystemExit(f"Subtitle folder not found: {subtitle_dir}")
    return sorted(
        path
        for path in subtitle_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
        and not is_target_language_file(path, target_lang)
        and not is_original_backup_file(path)
    )


def is_target_language_file(path: Path, target_lang: str) -> bool:
    stem_parts = path.stem.split(".")
    if stem_parts and stem_parts[-1].lower() == target_lang.lower():
        return True
    return path.stem.lower().endswith(f"-{target_lang.lower()}")


def is_original_backup_file(path: Path) -> bool:
    return path.stem.lower().endswith("-orig")


def translated_path(path: Path, target_lang: str) -> Path:
    stem_parts = path.stem.split(".")
    if len(stem_parts) > 1 and LANG_SUFFIX_PATTERN.fullmatch(stem_parts[-1]):
        stem_parts[-1] = target_lang
        return path.with_name(".".join(stem_parts) + path.suffix)
    return path.with_name(f"{path.stem}.{target_lang}{path.suffix}")


def parse_subtitles(content: str) -> list[SubtitleBlock]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    blocks: list[SubtitleBlock] = []
    for raw_block in re.split(r"\n{2,}", normalized):
        lines = raw_block.splitlines()
        text_start = text_start_index(lines)
        if text_start >= len(lines):
            blocks.append(SubtitleBlock(prefix=lines, text=""))
            continue
        blocks.append(
            SubtitleBlock(prefix=lines[:text_start], text="\n".join(lines[text_start:]))
        )
    return blocks


def text_start_index(lines: list[str]) -> int:
    timestamp_index = next(
        (index for index, line in enumerate(lines) if TIMESTAMP_PATTERN.search(line)),
        None,
    )
    if timestamp_index is None:
        if lines and lines[0].strip().upper() == "WEBVTT":
            return len(lines)
        return 0
    return timestamp_index + 1


def render_subtitles(blocks: list[SubtitleBlock], translated_texts: list[str]) -> str:
    rendered_blocks: list[str] = []
    for block, translated_text in zip(blocks, translated_texts, strict=True):
        lines = [*block.prefix]
        if translated_text:
            lines.extend(translated_text.splitlines())
        rendered_blocks.append("\n".join(lines))
    return "\n\n".join(rendered_blocks).rstrip() + "\n"


def chunk_blocks(blocks: list[SubtitleBlock], batch_chars: int) -> list[list[int]]:
    if batch_chars <= len(DELIMITER):
        raise SystemExit("--batch-chars must be larger than the internal delimiter length.")

    chunks: list[list[int]] = []
    current: list[int] = []
    current_size = 0
    for index, block in enumerate(blocks):
        text_size = len(block.text)
        next_size = current_size + text_size + (len(DELIMITER) if current else 0)
        if current and next_size > batch_chars:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(index)
        current_size += text_size + (len(DELIMITER) if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


def libretranslate(
    text: str,
    server_url: str,
    source_lang: str,
    target_lang: str,
    api_key: str | None,
) -> str:
    payload = {
        "q": text,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }
    if api_key:
        payload["api_key"] = api_key

    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/translate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"LibreTranslate request failed: {exc}") from exc

    translated = data.get("translatedText")
    if not isinstance(translated, str):
        raise SystemExit(f"Unexpected LibreTranslate response: {data}")
    return translated


def translate_blocks(blocks: list[SubtitleBlock], args: argparse.Namespace) -> list[str]:
    translated_texts = [block.text for block in blocks]
    for indexes in chunk_blocks(blocks, args.batch_chars):
        source_texts = [blocks[index].text for index in indexes]
        if not any(text.strip() for text in source_texts):
            continue
        combined = DELIMITER.join(source_texts)
        translated = libretranslate(
            combined,
            server_url=args.server_url,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            api_key=args.api_key,
        )
        parts = translated.split(DELIMITER)
        if len(parts) != len(indexes):
            parts = [
                libretranslate(
                    blocks[index].text,
                    server_url=args.server_url,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    api_key=args.api_key,
                )
                for index in indexes
            ]
        for index, text in zip(indexes, parts, strict=True):
            translated_texts[index] = text.strip()
    return translated_texts


def translate_file(path: Path, args: argparse.Namespace) -> bool:
    output_path = translated_path(path, args.target_lang)
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing translation: {output_path}")
        return False

    content = path.read_text(encoding="utf-8-sig")
    blocks = parse_subtitles(content)
    if not blocks:
        print(f"Skipping empty subtitle file: {path}")
        return False

    translated_texts = translate_blocks(blocks, args)
    output_path.write_text(render_subtitles(blocks, translated_texts), encoding="utf-8")
    print(f"Translated {path} -> {output_path}")
    return True


def main() -> None:
    args = parse_args()
    extensions = {extension.strip().lower() for extension in args.extensions.split(",") if extension.strip()}
    if not extensions:
        raise SystemExit("--extensions must include at least one extension.")

    total = 0
    for video_id in subtitle_ids(args.ids):
        for path in subtitle_files(video_id, extensions, args.target_lang):
            total += int(translate_file(path, args))
    print(f"Translated {total} subtitle file(s).")


if __name__ == "__main__":
    main()
