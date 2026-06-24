from __future__ import annotations

import argparse
import json
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .retrieval import (
    DEFAULT_DB,
    DEFAULT_FAISS_INDEX,
    DEFAULT_KEYFRAME_FAISS_INDEX,
    DEFAULT_KEYFRAME_VECTOR_METADATA,
    DEFAULT_OLLAMA_URL,
    DEFAULT_VECTOR_METADATA,
    hit_payload,
    load_faiss_index,
    load_reranker,
    load_vector_metadata,
    run_search,
    semantic_args_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent HTTP server for video retrieval.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index", type=Path, default=DEFAULT_FAISS_INDEX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_VECTOR_METADATA)
    parser.add_argument("--keyframe-index", type=Path, default=DEFAULT_KEYFRAME_FAISS_INDEX)
    parser.add_argument("--keyframe-metadata", type=Path, default=DEFAULT_KEYFRAME_VECTOR_METADATA)
    parser.add_argument("--provider", choices=["gemini", "ollama", "hashing"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--reranker-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--preload", action="store_true", help="Load FAISS indexes and BGE before serving.")
    return parser.parse_args()


def request_args(server_args: argparse.Namespace, body: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        query=str(body.get("query") or ""),
        db=Path(body.get("db") or server_args.db),
        index=Path(body.get("index") or server_args.index),
        metadata=Path(body.get("metadata") or server_args.metadata),
        keyframe_index=Path(body.get("keyframe_index") or server_args.keyframe_index),
        keyframe_metadata=Path(body.get("keyframe_metadata") or server_args.keyframe_metadata),
        search_level=body.get("search_level") or "scene",
        mode=body.get("mode") or "hybrid",
        adaptive=bool(body.get("adaptive", True)),
        show_plan=bool(body.get("show_plan", False)),
        limit=int(body.get("limit", 10)),
        candidate_limit=int(body.get("candidate_limit", 50)),
        provider=body.get("provider", server_args.provider),
        model=body.get("model", server_args.model),
        api_key=body.get("api_key", server_args.api_key),
        ollama_url=body.get("ollama_url", server_args.ollama_url),
        aliases=Path(body.get("aliases") or "config/entity_aliases.json"),
        bm25_weight=float(body.get("bm25_weight", 1.0)),
        semantic_weight=float(body.get("semantic_weight", 1.0)),
        semantic_fallback=bool(body.get("semantic_fallback", True)),
        entity_weight=float(body.get("entity_weight", 0.35)),
        rrf_k=int(body.get("rrf_k", 60)),
        dedupe_seconds=float(body.get("dedupe_seconds", 30.0)),
        reranker=body.get("reranker") or "none",
        reranker_model=body.get("reranker_model") or "BAAI/bge-reranker-v2-m3",
        rerank_top=int(body.get("rerank_top", 50)),
        reranker_device=body.get("reranker_device") or server_args.reranker_device,
        reranker_batch_size=int(body.get("reranker_batch_size", 1)),
        reranker_max_length=int(body.get("reranker_max_length", 256)),
        reranker_min_final_score=float(body.get("reranker_min_final_score", 0.20)),
        reranker_max_score_drop=float(body.get("reranker_max_score_drop", 6.0)),
        reranker_fallback=bool(body.get("reranker_fallback", True)),
        json=True,
    )


class RetrievalHandler(BaseHTTPRequestHandler):
    server_version = "RetrievalServer/0.1"

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(HTTPStatus.OK, {"status": "ok"})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/search":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            args = request_args(self.server.server_args, body)  # type: ignore[attr-defined]
            if not args.query.strip():
                self._json(HTTPStatus.BAD_REQUEST, {"error": "query is required"})
                return
            started = time.perf_counter()
            hits = run_search(args)
            latency_ms = (time.perf_counter() - started) * 1000
            self._json(
                HTTPStatus.OK,
                {
                    "query": args.query,
                    "latency_ms": latency_ms,
                    "count": len(hits),
                    "results": [hit_payload(hit) for hit in hits],
                },
            )
        except Exception as exc:  # pragma: no cover - server boundary
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[retrieval-server] {self.address_string()} - {format % args}", file=sys.stderr)


def preload(args: argparse.Namespace) -> None:
    for metadata_path, index_path in ((args.metadata, args.index), (args.keyframe_metadata, args.keyframe_index)):
        if metadata_path.exists() and index_path.exists():
            load_vector_metadata(metadata_path)
            load_faiss_index(index_path)
    load_reranker(
        argparse.Namespace(
            reranker="bge",
            reranker_model="BAAI/bge-reranker-v2-m3",
            reranker_device=args.reranker_device,
            reranker_max_length=256,
            reranker_fallback=True,
        )
    )


def main() -> None:
    args = parse_args()
    if args.preload:
        preload(args)
    server = ThreadingHTTPServer((args.host, args.port), RetrievalHandler)
    server.server_args = args  # type: ignore[attr-defined]
    print(f"[retrieval-server] http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
