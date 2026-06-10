from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
GEMINI_SUMMARY_DIR = ROOT / "gemini_summary"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_PROMPT = """Summarize this YouTube video for a video retrieval dataset.

Return Vietnamese JSON with this schema:
{
  "summary": "5-8 cau tom tat noi dung chinh",
  "topics": ["topic ngan gon"],
  "visual_keywords": ["doi tuong, boi canh, hanh dong nhin thay"],
  "timeline": [
    {"time": "MM:SS", "description": "su kien hoac noi dung dang chu y"}
  ],
  "search_queries": ["cac cum tu nguoi dung co the tim video nay"],
  "ocr_relevance": "nhan xet neu video co kha nang chua text tren man hinh"
}

Focus on factual visual/audio content. Avoid guessing details that are not visible or audible."""


@dataclass(frozen=True)
class GeminiSummaryResult:
    video_id: str
    output_path: Path
    seconds: float
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize YouTube videos with Gemini API.")
    parser.add_argument("items", nargs="*", help="Pairs in the form video_id=url, or raw YouTube URLs.")
    parser.add_argument("--api-key", default=None, help="Gemini API key. Defaults to GEMINI_API_KEY env var.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=GEMINI_SUMMARY_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args()
    if not args.items:
        raise SystemExit("Provide at least one video_id=url item or URL.")
    return args


def elapsed(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 3)


def api_key(value: str | None = None) -> str:
    key = value or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.")
    return key


def output_path(video_id: str, output_root: Path = GEMINI_SUMMARY_DIR) -> Path:
    return output_root / video_id / "summary.json"


def load_prompt(prompt: str | None = None, prompt_file: Path | None = None) -> str:
    if prompt_file:
        return prompt_file.read_text(encoding="utf-8")
    return prompt or DEFAULT_PROMPT


def render_prompt(template: str, video_id: str, url: str) -> str:
    return template.replace("{video_id}", video_id).replace("{url}", url)


def extract_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in response.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def maybe_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def request_summary(
    *,
    url: str,
    prompt: str,
    key: str,
    model: str,
    timeout: int,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"file_data": {"file_uri": url}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {body[:1000]}") from exc


def summarize_youtube(
    *,
    video_id: str,
    url: str,
    key: str,
    model: str = DEFAULT_MODEL,
    prompt_template: str = DEFAULT_PROMPT,
    output_root: Path = GEMINI_SUMMARY_DIR,
    overwrite: bool = False,
    timeout: int = 180,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
) -> GeminiSummaryResult:
    started_at = time.perf_counter()
    path = output_path(video_id, output_root)
    if path.exists() and not overwrite:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return GeminiSummaryResult(video_id, path, elapsed(started_at), payload.get("text", ""))

    prompt = render_prompt(prompt_template, video_id, url)
    response = request_summary(
        url=url,
        prompt=prompt,
        key=key,
        model=model,
        timeout=timeout,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    text = extract_text(response)
    payload = {
        "video_id": video_id,
        "url": url,
        "model": model,
        "prompt": prompt,
        "text": text,
        "json": maybe_json(text),
        "usage_metadata": response.get("usageMetadata"),
        "finish_reason": (response.get("candidates") or [{}])[0].get("finishReason"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return GeminiSummaryResult(video_id, path, elapsed(started_at), text)


def parse_item(item: str) -> tuple[str, str]:
    if "=" in item:
        video_id, url = item.split("=", 1)
        return video_id.strip(), url.strip()
    if "youtu" in item:
        video_id = item.rstrip("/").split("v=")[-1].split("&")[0].split("/")[-1]
        return video_id, item
    raise SystemExit(f"Invalid item: {item}. Use video_id=url or a YouTube URL.")


def main() -> None:
    args = parse_args()
    key = api_key(args.api_key)
    prompt_template = load_prompt(args.prompt, args.prompt_file)
    for item in args.items:
        video_id, url = parse_item(item)
        result = summarize_youtube(
            video_id=video_id,
            url=url,
            key=key,
            model=args.model,
            prompt_template=prompt_template,
            output_root=args.output_root,
            overwrite=args.overwrite,
            timeout=args.timeout,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"{video_id}: wrote {result.output_path} in {result.seconds}s")


if __name__ == "__main__":
    main()
