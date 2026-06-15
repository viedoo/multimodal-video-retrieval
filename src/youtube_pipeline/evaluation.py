from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .retrieval import (
    DEFAULT_DB,
    DEFAULT_FAISS_INDEX,
    DEFAULT_VECTOR_METADATA,
    TOKEN_PATTERN,
    bm25_search,
    entity_search,
    load_reranker,
    merge_hits,
    rerank_hits,
    semantic_search,
    tfidf_search,
)


DATASET_DIR = Path.cwd() / "dataset"
DEFAULT_QUERIES = DATASET_DIR / "evaluation_queries.jsonl"
DEFAULT_REPORT = DATASET_DIR / "evaluation_report.json"
DEFAULT_DETAILS = DATASET_DIR / "evaluation_details.jsonl"
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
GENERIC_WORDS = {
    "các", "của", "cho", "được", "đang", "đây", "đó", "khi", "là", "một", "những",
    "này", "nó", "ra", "sẽ", "thì", "trong", "và", "về", "với", "video", "chúng", "ta",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and evaluate video retrieval query sets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="Create a stratified bootstrap query set from indexed documents.")
    seed.add_argument("--db", type=Path, default=DEFAULT_DB)
    seed.add_argument("--output", type=Path, default=DEFAULT_QUERIES)
    seed.add_argument("--count", type=int, default=60)
    seed.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate query labels against the retrieval database.")
    validate.add_argument("--db", type=Path, default=DEFAULT_DB)
    validate.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)

    run = subparsers.add_parser("run", help="Compare BM25, semantic, and hybrid retrieval metrics.")
    run.add_argument("--db", type=Path, default=DEFAULT_DB)
    run.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    run.add_argument("--index", type=Path, default=DEFAULT_FAISS_INDEX)
    run.add_argument("--metadata", type=Path, default=DEFAULT_VECTOR_METADATA)
    run.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    run.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    run.add_argument("--modes", default="bm25,tfidf,semantic,entity,hybrid")
    run.add_argument("--candidate-limit", type=int, default=50)
    run.add_argument("--limit", type=int, default=10)
    run.add_argument("--reviewed-only", action="store_true")
    run.add_argument("--provider", choices=["gemini", "ollama", "hashing"], default=None)
    run.add_argument("--model", default=None)
    run.add_argument("--api-key", default=None)
    run.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run.add_argument("--bm25-weight", type=float, default=1.0)
    run.add_argument("--semantic-weight", type=float, default=1.0)
    run.add_argument("--entity-weight", type=float, default=0.35)
    run.add_argument("--rrf-k", type=int, default=60)
    run.add_argument("--reranker", choices=["none", "bge"], default="none")
    run.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    run.add_argument("--rerank-top", type=int, default=50)
    run.add_argument("--reranker-device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--reranker-batch-size", type=int, default=1)
    run.add_argument("--reranker-max-length", type=int, default=256)
    run.add_argument("--reranker-min-final-score", type=float, default=0.20)
    run.add_argument("--reranker-max-score-drop", type=float, default=6.0)
    return parser.parse_args()


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"File not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def choose_sentence(text: str) -> str:
    sentences = [" ".join(part.split()) for part in SENTENCE_PATTERN.split(text)]
    candidates = [part for part in sentences if 7 <= len(TOKEN_PATTERN.findall(part)) <= 35]
    if not candidates:
        candidates = [" ".join(text.split())]
    return max(candidates, key=lambda part: len(set(token.casefold() for token in TOKEN_PATTERN.findall(part))))


def document_frequencies(rows: list[sqlite3.Row]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for row in rows:
        frequencies.update(set(token.casefold() for token in TOKEN_PATTERN.findall(row["combined_text"])))
    return frequencies


def make_query(row: sqlite3.Row, frequencies: Counter[str], variant: int) -> tuple[str, str]:
    sentence = choose_sentence(row["combined_text"])
    tokens = TOKEN_PATTERN.findall(sentence)
    if variant == 0:
        query = " ".join(tokens[: min(14, len(tokens))])
        return query, "sentence_excerpt"

    ranked = sorted(
        {token for token in tokens if len(token) >= 3 and token.casefold() not in GENERIC_WORDS},
        key=lambda token: (frequencies[token.casefold()], -len(token), token.casefold()),
    )
    if variant == 1 and ranked:
        return " ".join(ranked[:6]), "distinct_keywords"

    entities = json.loads(row["entities_json"] or "[]")
    entity = next((item["text"] for item in entities if len(item.get("text", "")) >= 3), None)
    context = " ".join(ranked[:4])
    if entity and context:
        return f"{entity} {context}", "entity_context"
    return " ".join(tokens[: min(12, len(tokens))]), "sentence_excerpt"


def stratified_rows(rows: list[sqlite3.Row], count: int) -> list[sqlite3.Row]:
    by_video: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    selected: list[sqlite3.Row] = []
    video_ids = sorted(by_video)
    round_number = 0
    while len(selected) < count:
        added = False
        for video_id in video_ids:
            items = by_video[video_id]
            target_per_video = math.ceil(count / max(1, len(video_ids)))
            if round_number >= target_per_video or not items:
                continue
            position = int((round_number + 1) * len(items) / (target_per_video + 1))
            position = min(max(position, 0), len(items) - 1)
            selected.append(items[position])
            added = True
            if len(selected) == count:
                break
        if not added:
            break
        round_number += 1
    return selected


def seed_queries(db_path: Path, output: Path, count: int, force: bool) -> None:
    if output.exists() and not force:
        raise RuntimeError(f"Output already exists: {output}. Pass --force to replace it.")
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT document_id, video_id, scene_id, start_seconds, source_start_frame,
                   combined_text, entities_json, youtube_url, keyframe_json
            FROM documents
            WHERE length(combined_text) >= 120
            ORDER BY video_id, start_seconds
            """
        ).fetchall()
    if len(rows) < count:
        raise RuntimeError(f"Only {len(rows)} eligible documents for {count} requested queries.")
    frequencies = document_frequencies(rows)
    selected = stratified_rows(rows, count)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        query, query_type = make_query(row, frequencies, (index - 1) % 3)
        keyframe = json.loads(row["keyframe_json"]) if row["keyframe_json"] else None
        target_timestamp = keyframe.get("timestamp_seconds") if keyframe else row["start_seconds"]
        target_frame = keyframe.get("source_frame_index") if keyframe else row["source_start_frame"]
        records.append(
            {
                "query_id": f"q{index:03d}",
                "query": query,
                "relevant_document_ids": [row["document_id"]],
                "relevant_video_id": row["video_id"],
                "relevant_scene_id": row["scene_id"],
                "relevant_scene_start_seconds": row["start_seconds"],
                "relevant_scene_start_frame": row["source_start_frame"],
                "relevant_timestamp_seconds": target_timestamp,
                "relevant_source_frame": target_frame,
                "youtube_url": row["youtube_url"],
                "query_type": query_type,
                "source": "auto_generated_bootstrap",
                "reviewed": False,
                "notes": "Rewrite the query naturally and verify all relevant scenes before marking reviewed=true.",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    print(f"[seed] {len(records)} bootstrap queries -> {output}")
    print("[seed] These labels are not a human gold set. Review query wording and relevant scenes before final reporting.")


def validate_queries(db_path: Path, queries_path: Path) -> list[dict[str, Any]]:
    queries = load_jsonl(queries_path)
    errors: list[str] = []
    seen_ids: set[str] = set()
    with connect(db_path) as connection:
        known_ids = {row[0] for row in connection.execute("SELECT document_id FROM documents")}
    for index, item in enumerate(queries, start=1):
        query_id = item.get("query_id")
        if not query_id or query_id in seen_ids:
            errors.append(f"record {index}: missing or duplicate query_id={query_id!r}")
        seen_ids.add(query_id)
        if not str(item.get("query") or "").strip():
            errors.append(f"{query_id}: empty query")
        relevant = item.get("relevant_document_ids") or []
        if not relevant:
            errors.append(f"{query_id}: no relevant_document_ids")
        unknown = sorted(set(relevant) - known_ids)
        if unknown:
            errors.append(f"{query_id}: unknown document ids {unknown}")
    if errors:
        raise RuntimeError("Query validation failed:\n- " + "\n- ".join(errors))
    reviewed = sum(bool(item.get("reviewed")) for item in queries)
    print(f"[validate] {len(queries)} valid queries; reviewed={reviewed}; bootstrap={len(queries) - reviewed}")
    return queries


def search_args(args: argparse.Namespace, query: str) -> SimpleNamespace:
    return SimpleNamespace(
        query=query,
        db=args.db,
        index=args.index,
        metadata=args.metadata,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        ollama_url=args.ollama_url,
        bm25_weight=args.bm25_weight,
        semantic_weight=args.semantic_weight,
        entity_weight=args.entity_weight,
        rrf_k=args.rrf_k,
        reranker=args.reranker,
        reranker_model=args.reranker_model,
        rerank_top=args.rerank_top,
        reranker_device=args.reranker_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_min_final_score=args.reranker_min_final_score,
        reranker_max_score_drop=args.reranker_max_score_drop,
    )


def retrieve(mode: str, args: argparse.Namespace, query: str, reranker: Any | None = None) -> list[Any]:
    bm25_hits = bm25_search(args.db, query, args.candidate_limit) if mode in {"bm25", "hybrid"} else []
    tfidf_hits = tfidf_search(args.db, query, args.candidate_limit) if mode == "tfidf" else []
    semantic_args = search_args(args, query)
    semantic_hits = semantic_search(semantic_args, args.candidate_limit) if mode in {"semantic", "hybrid"} else []
    entity_hits = entity_search(args.db, query, args.candidate_limit) if mode in {"entity", "hybrid"} else []
    if mode == "bm25":
        return bm25_hits
    if mode == "tfidf":
        return tfidf_hits
    if mode == "semantic":
        return semantic_hits
    if mode == "entity":
        return entity_hits
    hits = merge_hits(semantic_args, bm25_hits, semantic_hits, entity_hits)
    return rerank_hits(semantic_args, hits, reranker)


def evaluate(args: argparse.Namespace) -> None:
    queries = validate_queries(args.db, args.queries)
    if args.reviewed_only:
        queries = [item for item in queries if item.get("reviewed")]
    if not queries:
        raise RuntimeError("No queries selected for evaluation.")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid_modes = sorted(set(modes) - {"bm25", "tfidf", "semantic", "entity", "hybrid"})
    if invalid_modes:
        raise RuntimeError(f"Unsupported modes: {invalid_modes}")

    detail_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "query_count": len(queries),
        "reviewed_query_count": sum(bool(item.get("reviewed")) for item in queries),
        "warning": "Bootstrap queries are retrieval-biased until manually reviewed.",
        "modes": {},
    }
    reranker = load_reranker(search_args(args, ""))
    for mode in modes:
        recalls_5: list[float] = []
        recalls_10: list[float] = []
        reciprocal_ranks: list[float] = []
        latencies_ms: list[float] = []
        for position, item in enumerate(queries, start=1):
            started = time.perf_counter()
            hits = retrieve(mode, args, item["query"], reranker)
            latency_ms = (time.perf_counter() - started) * 1000
            ranked_ids = [hit.document_id for hit in hits[: args.limit]]
            relevant = set(item["relevant_document_ids"])
            recall_5 = len(relevant & set(ranked_ids[:5])) / len(relevant)
            recall_10 = len(relevant & set(ranked_ids[:10])) / len(relevant)
            first_rank = next((rank for rank, document_id in enumerate(ranked_ids, 1) if document_id in relevant), None)
            reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
            recalls_5.append(recall_5)
            recalls_10.append(recall_10)
            reciprocal_ranks.append(reciprocal_rank)
            latencies_ms.append(latency_ms)
            detail_records.append(
                {
                    "mode": mode,
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "reviewed": bool(item.get("reviewed")),
                    "relevant_document_ids": sorted(relevant),
                    "ranked_document_ids": ranked_ids,
                    "first_relevant_rank": first_rank,
                    "recall_at_5": recall_5,
                    "recall_at_10": recall_10,
                    "reciprocal_rank": reciprocal_rank,
                    "latency_ms": latency_ms,
                }
            )
            if position % 10 == 0 or position == len(queries):
                print(f"[evaluate] {mode}: {position}/{len(queries)}")
        report["modes"][mode] = {
            "recall_at_5": statistics.fmean(recalls_5),
            "recall_at_10": statistics.fmean(recalls_10),
            "mrr": statistics.fmean(reciprocal_ranks),
            "latency_ms_mean": statistics.fmean(latencies_ms),
            "latency_ms_p50": percentile(latencies_ms, 0.5),
            "latency_ms_p95": percentile(latencies_ms, 0.95),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.details.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in detail_records), encoding="utf-8"
    )
    print(json.dumps(report["modes"], ensure_ascii=False, indent=2))
    print(f"[evaluate] report -> {args.report}")
    print(f"[evaluate] details -> {args.details}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.command == "seed":
        seed_queries(args.db, args.output, args.count, args.force)
    elif args.command == "validate":
        validate_queries(args.db, args.queries)
    elif args.command == "run":
        evaluate(args)


if __name__ == "__main__":
    main()
