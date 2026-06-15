from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
DEFAULT_DOCUMENTS = DATASET_DIR / "documents.jsonl"
DEFAULT_DB = DATASET_DIR / "search_index.sqlite3"
DEFAULT_FAISS_INDEX = DATASET_DIR / "faiss.index"
DEFAULT_VECTOR_METADATA = DATASET_DIR / "vector_metadata.json"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_ALIASES = ROOT / "config" / "entity_aliases.json"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
ACRONYM_PATTERN = re.compile(r"\b[A-ZĐ][A-ZĐ0-9._-]{1,}\b")

ENTITY_PREFIXES = {
    "PERSON": ("ông ", "bà ", "anh ", "chị ", "nhà thơ ", "ca sĩ ", "diễn viên ", "tiến sĩ "),
    "ORG": ("công ty ", "tập đoàn ", "đại học ", "trường ", "đài ", "bộ ", "ủy ban ", "ngân hàng "),
    "LOCATION": ("thành phố ", "tỉnh ", "quận ", "huyện ", "xã ", "phường ", "nước ", "đảo "),
}
ENTITY_STOPWORDS = {
    "thưa quý", "quý vị", "các bạn", "chúng ta", "hiện nay", "tuy nhiên", "trong đó",
    "sau đó", "đây là", "điều này", "video này", "hôm nay", "vì vậy", "ngoài ra",
}
KNOWN_ORGANIZATIONS = {
    "anthropic", "openai", "google", "microsoft", "facebook", "meta", "nvidia", "openbsd",
    "youtube", "gemini", "claude", "chatgpt", "vtv", "bbc", "cnn",
}
VIETNAMESE_STOPWORDS = {
    "bao", "bằng", "bị", "các", "cái", "cần", "cho", "chúng", "có", "của", "đang",
    "đâu", "đây", "đến", "đoạn", "đó", "được", "gì", "hay", "hiện", "khi", "không", "là",
    "làm", "lúc", "mà", "một", "nào", "này", "nhắc", "những", "ở", "ra", "sao", "sẽ", "ta",
    "thì", "thời", "trong", "từ", "và", "vào", "về", "với", "xuất",
}
# search_key() strips Vietnamese accents, so query-time stopwords must use the
# same ASCII representation. Keep question scaffolding out of exact-overlap.
VIETNAMESE_STOPWORDS |= {
    "bao", "bang", "bi", "cac", "cai", "can", "cho", "chung", "co", "cua", "dang",
    "dau", "day", "den", "do", "duoc", "gi", "hay", "hien", "khi", "khac", "khong", "la",
    "lam", "luc", "ma", "mot", "nao", "nay", "nhu", "nhung", "o", "ra", "sao", "se", "ta",
    "the", "thi", "thoi", "trong", "tu", "va", "vao", "ve", "voi", "xuat",
}


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    video_id: str
    scene_id: str
    start_seconds: float
    end_seconds: float
    source_start_frame: int | None
    keyframe: dict[str, Any] | None
    keyframes: list[dict[str, Any]]
    subtitle_text: str
    ocr_text: str
    combined_text: str
    entities: list[dict[str, str]]
    youtube_url: str
    bm25_rank: int | None = None
    semantic_rank: int | None = None
    entity_rank: int | None = None
    bm25_score: float | None = None
    semantic_score: float | None = None
    entity_score: float | None = None
    entity_overlap: int = 0
    matched_passage: str = ""
    matched_timestamp: float | None = None
    matched_frame: int | None = None
    reranker_score: float | None = None
    final_score: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query BM25/FAISS video retrieval indexes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build SQLite FTS5/BM25 index and local NER fields.")
    build.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    build.add_argument("--db", type=Path, default=DEFAULT_DB)
    build.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    build.add_argument("--ner-provider", choices=["rules", "electra"], default="rules")
    build.add_argument("--ner-model", default="NlpHUST/ner-vietnamese-electra-base")
    build.add_argument("--ner-device", choices=["auto", "cpu", "cuda"], default="auto")
    build.add_argument("--ner-min-score", type=float, default=0.55)

    embed = subparsers.add_parser("embed", help="Create document embeddings and a FAISS cosine index.")
    embed.add_argument("--db", type=Path, default=DEFAULT_DB)
    embed.add_argument("--index", type=Path, default=DEFAULT_FAISS_INDEX)
    embed.add_argument("--metadata", type=Path, default=DEFAULT_VECTOR_METADATA)
    embed.add_argument("--provider", choices=["gemini", "ollama", "hashing"], default="gemini")
    embed.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    embed.add_argument("--dimension", type=int, default=768)
    embed.add_argument("--api-key", default=None, help="Defaults to GEMINI_API_KEY.")
    embed.add_argument("--timeout", type=int, default=120)
    embed.add_argument("--retries", type=int, default=5)
    embed.add_argument("--retry-delay", type=float, default=2.0)
    embed.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    embed.add_argument("--limit", type=int, default=None)
    embed.add_argument("--force", action="store_true")

    search = subparsers.add_parser("search", help="Search BM25, semantic, or hybrid indexes.")
    search.add_argument("query")
    search.add_argument("--db", type=Path, default=DEFAULT_DB)
    search.add_argument("--index", type=Path, default=DEFAULT_FAISS_INDEX)
    search.add_argument("--metadata", type=Path, default=DEFAULT_VECTOR_METADATA)
    search.add_argument("--mode", choices=["bm25", "tfidf", "semantic", "entity", "hybrid"], default="hybrid")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--candidate-limit", type=int, default=50)
    search.add_argument("--provider", choices=["gemini", "ollama", "hashing"], default=None)
    search.add_argument("--model", default=None)
    search.add_argument("--api-key", default=None)
    search.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    search.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    search.add_argument("--bm25-weight", type=float, default=1.0)
    search.add_argument("--semantic-weight", type=float, default=1.0)
    search.add_argument("--entity-weight", type=float, default=0.35)
    search.add_argument("--rrf-k", type=int, default=60)
    search.add_argument("--dedupe-seconds", type=float, default=30.0)
    search.add_argument("--reranker", choices=["none", "bge"], default="none")
    search.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    search.add_argument("--rerank-top", type=int, default=50)
    search.add_argument("--reranker-device", choices=["auto", "cpu", "cuda"], default="auto")
    search.add_argument("--reranker-batch-size", type=int, default=1)
    search.add_argument("--reranker-max-length", type=int, default=256)
    search.add_argument("--reranker-min-final-score", type=float, default=0.20)
    search.add_argument("--reranker-max-score-drop", type=float, default=6.0)
    search.add_argument("--json", action="store_true")
    return parser.parse_args()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Documents file not found: {path}")
    documents: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return documents


def search_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(character for character in value if not unicodedata.combining(character))
    value = value.replace("đ", "d")
    return " ".join(TOKEN_PATTERN.findall(value))


def load_aliases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"Entity aliases must be a JSON array: {path}")
    output: list[dict[str, Any]] = []
    for record in records:
        canonical = normalize_entity(str(record.get("canonical") or ""))
        if not canonical:
            continue
        aliases = [canonical, *(record.get("aliases") or [])]
        output.append(
            {
                "canonical": canonical,
                "type": str(record.get("type") or "MISC").upper(),
                "aliases": sorted({normalize_entity(str(alias)) for alias in aliases if str(alias).strip()}),
            }
        )
    return output


def alias_lookup(records: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    lookup: dict[str, list[dict[str, str]]] = {}
    for record in records:
        for alias in record["aliases"]:
            lookup.setdefault(search_key(alias), []).append(
                {"canonical": record["canonical"], "type": record["type"], "alias": alias}
            )
    return lookup


def canonicalize_entities(
    entities: list[dict[str, Any]], aliases: dict[str, list[dict[str, str]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        matches = aliases.get(search_key(entity["text"])) or []
        typed_matches = [match for match in matches if match["type"] == entity["type"]]
        candidates = typed_matches or [
            {"canonical": entity["text"], "type": entity["type"], "alias": entity["text"]}
        ]
        for candidate in candidates:
            key = (search_key(candidate["canonical"]), candidate["type"])
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    **entity,
                    "text": candidate["canonical"],
                    "type": candidate["type"],
                    "matched_text": entity["text"],
                }
            )
    return output


def normalize_entity(value: str) -> str:
    return " ".join(value.split()).strip(" .,:;!?()[]{}\"'")


def infer_entity_type(text: str, full_text: str) -> str:
    lowered = text.casefold()
    context = full_text.casefold()
    for entity_type, prefixes in ENTITY_PREFIXES.items():
        if any(f"{prefix}{lowered}" in context for prefix in prefixes):
            return entity_type
    if lowered in KNOWN_ORGANIZATIONS:
        return "ORG"
    if text.isupper() and len(text) <= 12:
        return "ORG"
    return "MISC"


def capitalized_sequences(text: str) -> Iterable[str]:
    matches = list(WORD_PATTERN.finditer(text))
    sequence: list[re.Match[str]] = []

    def flush() -> Iterable[str]:
        if not sequence:
            return []
        value = " ".join(match.group(0) for match in sequence[:6])
        sequence.clear()
        return [value]

    for match in matches:
        token = match.group(0)
        is_capitalized = token[0].isupper() and not token.isupper()
        is_known_name = token.casefold() in KNOWN_ORGANIZATIONS
        adjacent = not sequence or text[sequence[-1].end() : match.start()].isspace()
        if (is_capitalized or is_known_name) and adjacent:
            sequence.append(match)
            continue
        yield from flush()
        if is_capitalized or is_known_name:
            sequence.append(match)
    yield from flush()


def extract_entities(text: str) -> list[dict[str, str]]:
    candidates: list[tuple[str, str]] = []
    candidates.extend((match.group(0), "EMAIL") for match in EMAIL_PATTERN.finditer(text))
    candidates.extend((match.group(0).rstrip(".,;"), "URL") for match in URL_PATTERN.finditer(text))
    candidates.extend((match.group(0), infer_entity_type(match.group(0), text)) for match in ACRONYM_PATTERN.finditer(text))
    for candidate in capitalized_sequences(text):
        value = normalize_entity(candidate)
        lowered = value.casefold()
        tokens = lowered.split()
        if lowered in ENTITY_STOPWORDS or not tokens:
            continue
        if len(tokens) == 1 and lowered not in KNOWN_ORGANIZATIONS:
            continue
        candidates.append((value, infer_entity_type(value, text)))

    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value, entity_type in candidates:
        value = normalize_entity(value)
        key = (value.casefold(), entity_type)
        if len(value) < 2 or key in seen:
            continue
        seen.add(key)
        output.append({"text": value, "type": entity_type, "source": "rules"})
    return output


def best_keyframe(document: dict[str, Any]) -> dict[str, Any] | None:
    keyframes = document.get("keyframes") or []
    if not keyframes:
        return None
    with_ocr = [item for item in keyframes if item.get("ocr_text")]
    return (with_ocr or keyframes)[0]


def index_terms(text: str) -> Counter[str]:
    return Counter(
        token
        for token in search_key(text).split()
        if len(token) >= 2 and token not in VIETNAMESE_STOPWORDS
    )


def init_search_db(connection: sqlite3.Connection, overwrite: bool) -> None:
    if overwrite:
        connection.executescript(
            """
            DROP TABLE IF EXISTS documents_fts;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS document_entities;
            DROP TABLE IF EXISTS entity_aliases;
            DROP TABLE IF EXISTS document_terms;
            DROP TABLE IF EXISTS document_term_norms;
            DROP TABLE IF EXISTS term_stats;
            """
        )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            document_id TEXT NOT NULL UNIQUE,
            video_id TEXT NOT NULL,
            scene_id TEXT NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            source_start_frame INTEGER,
            source_end_frame INTEGER,
            subtitle_text TEXT NOT NULL,
            ocr_text TEXT NOT NULL,
            combined_text TEXT NOT NULL,
            entities_text TEXT NOT NULL,
            entities_json TEXT NOT NULL,
            youtube_url TEXT NOT NULL,
            keyframe_json TEXT,
            document_json TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            subtitle_text,
            ocr_text,
            combined_text,
            entities_text,
            content='documents',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE IF NOT EXISTS embeddings_cache (
            document_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            vector BLOB NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(document_id, provider, model, dimension)
        );
        CREATE TABLE IF NOT EXISTS document_entities (
            document_id TEXT NOT NULL,
            canonical TEXT NOT NULL,
            canonical_key TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            matched_text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'rules',
            score REAL,
            PRIMARY KEY(document_id, canonical_key, entity_type)
        );
        CREATE INDEX IF NOT EXISTS idx_document_entities_key ON document_entities(canonical_key);
        CREATE TABLE IF NOT EXISTS entity_aliases (
            canonical TEXT NOT NULL,
            canonical_key TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            alias TEXT NOT NULL,
            alias_key TEXT NOT NULL,
            PRIMARY KEY(canonical_key, entity_type, alias_key)
        );
        CREATE INDEX IF NOT EXISTS idx_entity_aliases_key ON entity_aliases(alias_key);
        CREATE TABLE IF NOT EXISTS document_terms (
            document_id TEXT NOT NULL,
            term TEXT NOT NULL,
            tf INTEGER NOT NULL,
            PRIMARY KEY(document_id, term)
        );
        CREATE INDEX IF NOT EXISTS idx_document_terms_term ON document_terms(term);
        CREATE TABLE IF NOT EXISTS term_stats (
            term TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL,
            inverse_document_frequency REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_term_norms (
            document_id TEXT PRIMARY KEY,
            vector_norm REAL NOT NULL
        );
        """
    )


def build_index(
    documents_path: Path,
    db_path: Path,
    overwrite: bool,
    aliases_path: Path = DEFAULT_ALIASES,
    ner_provider: str = "rules",
    ner_model: str = "NlpHUST/ner-vietnamese-electra-base",
    ner_device: str = "auto",
    ner_min_score: float = 0.55,
) -> None:
    documents = load_jsonl(documents_path)
    alias_records = load_aliases(aliases_path)
    aliases = alias_lookup(alias_records)
    model_ner = None
    if ner_provider == "electra":
        from .ner import ElectraVietnameseNER

        model_ner = ElectraVietnameseNER(ner_model, ner_device, ner_min_score)
    model_entities = (
        model_ner.extract_many([document.get("combined_text") or "" for document in documents])
        if model_ner is not None
        else [[] for _ in documents]
    )
    with connect(db_path) as connection:
        init_search_db(connection, overwrite)
        connection.execute("DELETE FROM documents")
        connection.execute("DELETE FROM document_entities")
        connection.execute("DELETE FROM entity_aliases")
        connection.execute("DELETE FROM document_terms")
        connection.execute("DELETE FROM document_term_norms")
        connection.execute("DELETE FROM term_stats")
        for record in alias_records:
            for alias in record["aliases"]:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO entity_aliases(
                        canonical, canonical_key, entity_type, alias, alias_key
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record["canonical"], search_key(record["canonical"]), record["type"],
                        alias, search_key(alias),
                    ),
                )
        document_frequencies: Counter[str] = Counter()
        document_term_counts: dict[str, Counter[str]] = {}
        for item_index, document in enumerate(documents, start=1):
            combined_text = document.get("combined_text") or ""
            entities: list[dict[str, Any]] = extract_entities(combined_text)
            entities.extend(model_entities[item_index - 1])
            normalized_text = f" {search_key(combined_text)} "
            for alias_key, matches in aliases.items():
                if alias_key and f" {alias_key} " in normalized_text:
                    for match in matches:
                        if (
                            match["canonical"] == "Hồ Chí Minh"
                            and match["type"] == "PERSON"
                            and " thanh pho ho chi minh " in normalized_text
                        ):
                            continue
                        entities.append(
                            {
                                "text": match["alias"],
                                "type": match["type"],
                                "source": "alias",
                                "score": 1.0,
                            }
                        )
            entities = canonicalize_entities(entities, aliases)
            has_ho_chi_minh_city = any(
                entity["type"] == "LOCATION"
                and search_key(entity["text"]) == "thanh pho ho chi minh"
                for entity in entities
            )
            if has_ho_chi_minh_city:
                entities = [
                    entity
                    for entity in entities
                    if not (
                        search_key(entity["text"]) == "ho chi minh"
                        and entity["type"] != "LOCATION"
                    )
                ]
            entities_text = " ".join(entity["text"] for entity in entities)
            keyframe = best_keyframe(document)
            connection.execute(
                """
                INSERT INTO documents(
                    document_id, video_id, scene_id, start_seconds, end_seconds,
                    source_start_frame, source_end_frame, subtitle_text, ocr_text,
                    combined_text, entities_text, entities_json, youtube_url,
                    keyframe_json, document_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document["document_id"],
                    document["video_id"],
                    document["scene_id"],
                    document["start_seconds"],
                    document["end_seconds"],
                    document.get("source_start_frame"),
                    document.get("source_end_frame"),
                    document.get("subtitle_text") or "",
                    document.get("ocr_text") or "",
                    combined_text,
                    entities_text,
                    json.dumps(entities, ensure_ascii=False),
                    document.get("youtube_url") or "",
                    json.dumps(keyframe, ensure_ascii=False) if keyframe else None,
                    json.dumps(document, ensure_ascii=False),
                ),
            )
            for entity in entities:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO document_entities(
                        document_id, canonical, canonical_key, entity_type,
                        matched_text, source, score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document["document_id"], entity["text"], search_key(entity["text"]),
                        entity["type"], entity.get("matched_text") or entity["text"],
                        entity.get("source") or ner_provider, entity.get("score"),
                    ),
                )
            terms = index_terms(combined_text)
            document_term_counts[document["document_id"]] = terms
            document_frequencies.update(terms.keys())
            connection.executemany(
                "INSERT INTO document_terms(document_id, term, tf) VALUES (?, ?, ?)",
                ((document["document_id"], term, frequency) for term, frequency in terms.items()),
            )
            if item_index % 100 == 0:
                print(f"[index] {item_index}/{len(documents)}")
        searchable_count = max(1, sum(bool(index_terms(document.get("combined_text") or "")) for document in documents))
        idf_values = {
            term: math.log((searchable_count + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequencies.items()
        }
        connection.executemany(
            "INSERT INTO term_stats(term, document_frequency, inverse_document_frequency) VALUES (?, ?, ?)",
            (
                (term, frequency, idf_values[term])
                for term, frequency in document_frequencies.items()
            ),
        )
        connection.executemany(
            "INSERT INTO document_term_norms(document_id, vector_norm) VALUES (?, ?)",
            (
                (
                    document_id,
                    math.sqrt(
                        sum(((1.0 + math.log(tf)) * idf_values[term]) ** 2 for term, tf in terms.items())
                    ) or 1.0,
                )
                for document_id, terms in document_term_counts.items()
            ),
        )
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        searchable = connection.execute("SELECT COUNT(*) FROM documents WHERE combined_text != ''").fetchone()[0]
    print(
        f"[index] {count} documents, {searchable} searchable, "
        f"ner={ner_provider}, aliases={len(alias_records)} -> {db_path}"
    )


def fts_query(value: str) -> str:
    tokens = list(index_terms(value))
    if not tokens:
        tokens = search_key(value).split()
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def resolve_query_entity_keys(connection: sqlite3.Connection, query: str) -> set[str]:
    normalized = f" {search_key(query)} "
    rows = connection.execute(
        "SELECT DISTINCT canonical_key, alias_key FROM entity_aliases ORDER BY length(alias_key) DESC"
    ).fetchall()
    keys = {row["canonical_key"] for row in rows if row["alias_key"] and f" {row['alias_key']} " in normalized}
    for entity in extract_entities(query):
        entity_key = search_key(entity["text"])
        alias_rows = connection.execute(
            "SELECT canonical_key FROM entity_aliases WHERE alias_key = ?", (entity_key,)
        ).fetchall()
        keys.update(row["canonical_key"] for row in alias_rows)
        if not alias_rows and len(entity_key.split()) >= 2:
            keys.add(entity_key)
    return {key for key in keys if key}


def row_to_hit(row: sqlite3.Row, **scores: Any) -> SearchHit:
    document = json.loads(row["document_json"])
    return SearchHit(
        document_id=row["document_id"],
        video_id=row["video_id"],
        scene_id=row["scene_id"],
        start_seconds=float(row["start_seconds"]),
        end_seconds=float(row["end_seconds"]),
        source_start_frame=row["source_start_frame"],
        keyframe=json.loads(row["keyframe_json"]) if row["keyframe_json"] else None,
        keyframes=document.get("keyframes") or [],
        subtitle_text=row["subtitle_text"],
        ocr_text=row["ocr_text"],
        combined_text=row["combined_text"],
        entities=json.loads(row["entities_json"]),
        youtube_url=row["youtube_url"],
        **scores,
    )


def best_passage(hit: SearchHit, query: str, max_chars: int = 1800) -> tuple[str, float, int | None]:
    query_terms = set(index_terms(query))
    candidates: list[tuple[float, str, float, int | None]] = []
    for keyframe in hit.keyframes or ([hit.keyframe] if hit.keyframe else []):
        subtitle = keyframe.get("subtitle_text") or ""
        ocr = keyframe.get("ocr_text") or ""
        parts = [f"SUBTITLE: {hit.subtitle_text}"] if hit.subtitle_text else []
        if subtitle and subtitle not in hit.subtitle_text:
            parts.append(f"LOCAL SUBTITLE: {subtitle}")
        if ocr:
            ocr_lines = []
            for line in keyframe.get("ocr_lines") or []:
                value = " ".join(str(line.get("text") or "").split())
                confidence = float(line.get("confidence") or 0.0)
                has_query_term = bool(query_terms & set(index_terms(value)))
                if value and (confidence >= 0.75 or has_query_term):
                    ocr_lines.append(value)
            cleaned_ocr = " | ".join(ocr_lines) if ocr_lines else " ".join(ocr.split())
            if cleaned_ocr:
                parts.append(f"OCR: {cleaned_ocr[:500]}")
        text = "\n".join(parts)
        ocr_overlap = len(query_terms & set(index_terms(ocr)))
        local_overlap = len(query_terms & set(index_terms(subtitle)))
        confidence = float(keyframe.get("ocr_confidence_mean") or 0.0)
        score = 2.0 * ocr_overlap + 0.5 * local_overlap + 0.1 * confidence
        candidates.append(
            (
                score,
                text,
                float(keyframe.get("timestamp_seconds") or hit.start_seconds),
                keyframe.get("source_frame_index"),
            )
        )
    if not candidates:
        return hit.combined_text[:max_chars], hit.start_seconds, hit.source_start_frame
    _, text, timestamp, frame = max(candidates, key=lambda item: (item[0], len(item[1])))
    return text[:max_chars], timestamp, frame


def attach_best_passages(hits: list[SearchHit], query: str) -> list[SearchHit]:
    output: list[SearchHit] = []
    for hit in hits:
        passage, timestamp, frame = best_passage(hit, query)
        output.append(
            SearchHit(
                **{
                    **hit.__dict__,
                    "matched_passage": passage,
                    "matched_timestamp": timestamp,
                    "matched_frame": frame,
                }
            )
        )
    return output


def load_reranker(args: argparse.Namespace) -> Any | None:
    if args.reranker == "none":
        return None
    from .reranker import TransformerReranker

    return TransformerReranker(
        args.reranker_model,
        args.reranker_device,
        args.reranker_max_length,
    )


def rerank_hits(args: argparse.Namespace, hits: list[SearchHit], reranker: Any | None = None) -> list[SearchHit]:
    hits = attach_best_passages(hits, args.query)
    if args.reranker == "none" or not hits:
        return hits

    rerank_count = min(args.rerank_top, len(hits))
    candidates = hits[:rerank_count]
    reranker = reranker or load_reranker(args)
    scores = reranker.score(
        args.query,
        [hit.matched_passage for hit in candidates],
        args.reranker_batch_size,
    )
    retrieval_scores = [hit.final_score for hit in candidates]
    retrieval_min = min(retrieval_scores)
    retrieval_span = max(retrieval_scores) - retrieval_min
    query_terms = set(index_terms(args.query))
    reranked = [
        SearchHit(
            **{
                **hit.__dict__,
                "reranker_score": score,
                "final_score": (
                    0.60 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))
                    + 0.25
                    * (
                        (hit.final_score - retrieval_min) / retrieval_span
                        if retrieval_span > 0
                        else 1.0
                    )
                    + 0.10
                    * (
                        len(query_terms & set(index_terms(hit.matched_passage))) / len(query_terms)
                        if query_terms
                        else 0.0
                    )
                    + 0.05 * float(hit.entity_overlap > 0)
                ),
            }
        )
        for hit, score in zip(candidates, scores)
    ]
    reranked.sort(
        key=lambda hit: hit.final_score,
        reverse=True,
    )
    best_reranker_score = reranked[0].reranker_score if reranked else None
    return [
        hit
        for hit in reranked
        if hit.final_score >= args.reranker_min_final_score
        and hit.reranker_score is not None
        and best_reranker_score is not None
        and hit.reranker_score >= best_reranker_score - args.reranker_max_score_drop
    ]


def bm25_search(db_path: Path, query: str, limit: int) -> list[SearchHit]:
    expression = fts_query(query)
    if not expression:
        return []
    sql = """
        SELECT d.*, bm25(documents_fts, 1.5, 0.7, 1.0, 2.5) AS score
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE documents_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """
    with connect(db_path) as connection:
        rows = connection.execute(sql, (expression, max(limit * 4, limit))).fetchall()
        query_terms = set(index_terms(query))
        entity_keys = resolve_query_entity_keys(connection, query)
        entity_documents: set[str] = set()
        if entity_keys:
            placeholders = ",".join("?" for _ in entity_keys)
            entity_documents = {
                row[0]
                for row in connection.execute(
                    f"SELECT DISTINCT document_id FROM document_entities WHERE canonical_key IN ({placeholders})",
                    tuple(entity_keys),
                )
            }
    ranked_rows = sorted(
        rows,
        key=lambda row: (
            row["document_id"] not in entity_documents,
            -len(query_terms & set(index_terms(row["combined_text"]))),
            float(row["score"]),
        ),
    )[:limit]
    return [
        row_to_hit(row, bm25_rank=index, bm25_score=float(row["score"]))
        for index, row in enumerate(ranked_rows, 1)
    ]


def tfidf_search(db_path: Path, query: str, limit: int) -> list[SearchHit]:
    query_counts = index_terms(query)
    if not query_counts:
        return []
    with connect(db_path) as connection:
        placeholders = ",".join("?" for _ in query_counts)
        stats = {
            row["term"]: float(row["inverse_document_frequency"])
            for row in connection.execute(
                f"SELECT term, inverse_document_frequency FROM term_stats WHERE term IN ({placeholders})",
                tuple(query_counts),
            )
        }
        query_weights = {
            term: (1.0 + math.log(frequency)) * stats[term]
            for term, frequency in query_counts.items()
            if term in stats
        }
        if not query_weights:
            return []
        query_norm = math.sqrt(sum(weight**2 for weight in query_weights.values())) or 1.0
        rows = connection.execute(
            f"""
            SELECT dt.document_id, dt.term, dt.tf, ts.inverse_document_frequency, n.vector_norm
            FROM document_terms dt
            JOIN term_stats ts ON ts.term = dt.term
            JOIN document_term_norms n ON n.document_id = dt.document_id
            WHERE dt.term IN ({placeholders})
            """,
            tuple(query_counts),
        ).fetchall()
        scores: dict[str, float] = Counter()
        norms: dict[str, float] = {}
        for row in rows:
            document_weight = (1.0 + math.log(row["tf"])) * row["inverse_document_frequency"]
            scores[row["document_id"]] += query_weights[row["term"]] * document_weight
            norms[row["document_id"]] = row["vector_norm"]
        ranked = sorted(
            ((document_id, score / (query_norm * norms[document_id])) for document_id, score in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]
        if not ranked:
            return []
        ids = [document_id for document_id, _ in ranked]
        id_placeholders = ",".join("?" for _ in ids)
        documents = connection.execute(
            f"SELECT * FROM documents WHERE document_id IN ({id_placeholders})", ids
        ).fetchall()
    by_id = {row["document_id"]: row for row in documents}
    return [
        row_to_hit(by_id[document_id], bm25_rank=rank, bm25_score=score)
        for rank, (document_id, score) in enumerate(ranked, 1)
        if document_id in by_id
    ]


def entity_search(db_path: Path, query: str, limit: int) -> list[SearchHit]:
    with connect(db_path) as connection:
        keys = resolve_query_entity_keys(connection, query)
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        ranked = connection.execute(
            f"""
            SELECT de.document_id, COUNT(DISTINCT de.canonical_key) AS matched_entities,
                   MAX(COALESCE(de.score, 0.5)) AS confidence
            FROM document_entities de
            JOIN documents d ON d.document_id = de.document_id
            WHERE de.canonical_key IN ({placeholders})
            GROUP BY de.document_id
            ORDER BY matched_entities DESC, confidence DESC, LENGTH(d.combined_text) DESC
            LIMIT ?
            """,
            (*keys, limit),
        ).fetchall()
        ids = [row["document_id"] for row in ranked]
        if not ids:
            return []
        id_placeholders = ",".join("?" for _ in ids)
        documents = connection.execute(
            f"SELECT * FROM documents WHERE document_id IN ({id_placeholders})", ids
        ).fetchall()
    by_id = {row["document_id"]: row for row in documents}
    return [
        row_to_hit(
            by_id[row["document_id"]],
            entity_rank=rank,
            entity_score=float(row["matched_entities"]) + float(row["confidence"]) / 10.0,
            entity_overlap=int(row["matched_entities"]),
        )
        for rank, row in enumerate(ranked, 1)
        if row["document_id"] in by_id
    ]


def api_key(value: str | None) -> str:
    key = value or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY for Gemini embeddings.")
    return key


def gemini_embedding(
    text: str,
    model: str,
    dimension: int,
    key: str,
    timeout: int = 120,
    retries: int = 5,
    retry_delay: float = 2.0,
) -> np.ndarray:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
    payload = {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "output_dimensionality": dimension,
    }
    request_data = json.dumps(payload).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(
            endpoint,
            data=request_data,
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            vector = result.get("embedding", {}).get("values")
            if vector is None:
                embeddings = result.get("embeddings") or []
                vector = embeddings[0].get("values") if embeddings else None
            if vector is None:
                raise RuntimeError(f"No embedding in Gemini response: {str(result)[:500]}")
            return normalize_vector(np.asarray(vector, dtype=np.float32))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise RuntimeError(f"Gemini embedding HTTP {exc.code}: {body[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == retries:
                raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc
        time.sleep(retry_delay * (2**attempt))
    raise RuntimeError("Gemini embedding failed after retries.")


def ollama_embedding(
    text: str,
    model: str,
    url: str,
    timeout: int = 120,
    keep_alive: str | int | None = None,
) -> np.ndarray:
    endpoint = f"{url.rstrip('/')}/api/embed"
    payload: dict[str, Any] = {"model": model, "input": text, "truncate": True}
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama embedding request failed at {endpoint}. Start Ollama and run: ollama pull {model}. Error: {exc}"
        ) from exc
    embeddings = result.get("embeddings") or []
    if not embeddings:
        raise RuntimeError(f"No embeddings in Ollama response: {str(result)[:500]}")
    return normalize_vector(np.asarray(embeddings[0], dtype=np.float32))


def hashing_embedding(text: str, dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = TOKEN_PATTERN.findall(normalized)
    features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    return normalize_vector(vector)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def prepare_document(document_id: str, text: str, provider: str = "gemini") -> str:
    if provider == "ollama":
        return f"search_document: {text}"
    return f"title: {document_id} | text: {text}"


def prepare_query(query: str, provider: str = "gemini") -> str:
    if provider == "ollama":
        return f"search_query: {query}"
    return f"task: search result | query: {query}"


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cached_vector(
    connection: sqlite3.Connection, document_id: str, provider: str, model: str, dimension: int, digest: str
) -> np.ndarray | None:
    row = connection.execute(
        """
        SELECT text_hash, vector FROM embeddings_cache
        WHERE document_id = ? AND provider = ? AND model = ? AND dimension = ?
        """,
        (document_id, provider, model, dimension),
    ).fetchone()
    if not row or row["text_hash"] != digest:
        return None
    return np.frombuffer(row["vector"], dtype=np.float32).copy()


def save_cached_vector(
    connection: sqlite3.Connection,
    document_id: str,
    provider: str,
    model: str,
    dimension: int,
    digest: str,
    vector: np.ndarray,
) -> None:
    connection.execute(
        """
        INSERT INTO embeddings_cache(document_id, provider, model, dimension, text_hash, vector)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, provider, model, dimension) DO UPDATE SET
            text_hash = excluded.text_hash,
            vector = excluded.vector,
            updated_at = CURRENT_TIMESTAMP
        """,
        (document_id, provider, model, dimension, digest, vector.astype(np.float32).tobytes()),
    )


def import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is not installed. Install project optional dependency: pip install -e .[retrieval]") from exc
    return faiss


def build_embeddings(args: argparse.Namespace) -> None:
    faiss = import_faiss()
    key = api_key(args.api_key) if args.provider == "gemini" else None
    with connect(args.db) as connection:
        init_search_db(connection, overwrite=False)
        query = "SELECT document_id, combined_text FROM documents WHERE combined_text != '' ORDER BY id"
        rows = connection.execute(query).fetchall()
        if args.limit is not None:
            rows = rows[: args.limit]
        vectors: list[np.ndarray] = []
        document_ids: list[str] = []
        for index, row in enumerate(rows, start=1):
            content = prepare_document(row["document_id"], row["combined_text"], args.provider)
            digest = text_hash(content)
            vector = None if args.force else cached_vector(
                connection, row["document_id"], args.provider, args.model, args.dimension, digest
            )
            if vector is None:
                if args.provider == "gemini":
                    vector = gemini_embedding(
                        content,
                        args.model,
                        args.dimension,
                        key or "",
                        args.timeout,
                        args.retries,
                        args.retry_delay,
                    )
                elif args.provider == "ollama":
                    vector = ollama_embedding(content, args.model, args.ollama_url, args.timeout)
                else:
                    vector = hashing_embedding(content, args.dimension)
                if len(vector) != args.dimension:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: provider returned {len(vector)}, "
                        f"but --dimension is {args.dimension}."
                    )
                save_cached_vector(
                    connection,
                    row["document_id"],
                    args.provider,
                    args.model,
                    args.dimension,
                    digest,
                    vector,
                )
                connection.commit()
            vectors.append(normalize_vector(vector.astype(np.float32)))
            document_ids.append(row["document_id"])
            if index % 25 == 0 or index == len(rows):
                print(f"[embed] {index}/{len(rows)}")

    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, args.dimension), dtype=np.float32)
    index = faiss.IndexFlatIP(args.dimension)
    if len(matrix):
        index.add(matrix)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index))
    metadata = {
        "provider": args.provider,
        "model": args.model,
        "dimension": args.dimension,
        "document_ids": document_ids,
        "count": len(document_ids),
        "metric": "cosine_via_normalized_inner_product",
    }
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[embed] FAISS index: {len(document_ids)} vectors -> {args.index}")


def load_vector_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Vector metadata not found: {path}. Run retrieval-pipeline embed first.")
    return json.loads(path.read_text(encoding="utf-8"))


def query_embedding(query: str, metadata: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    provider = args.provider or metadata["provider"]
    model = args.model or metadata["model"]
    dimension = int(metadata["dimension"])
    content = prepare_query(query, provider)
    if provider == "gemini":
        return gemini_embedding(content, model, dimension, api_key(args.api_key))
    if provider == "ollama":
        # A local cross-encoder may need the same GPU immediately after this call.
        return ollama_embedding(content, model, args.ollama_url, keep_alive=0)
    return hashing_embedding(content, dimension)


def semantic_search(args: argparse.Namespace, limit: int) -> list[SearchHit]:
    faiss = import_faiss()
    metadata = load_vector_metadata(args.metadata)
    index = faiss.read_index(str(args.index))
    vector = query_embedding(args.query, metadata, args).reshape(1, -1).astype(np.float32)
    scores, indexes = index.search(vector, min(limit, int(metadata["count"])))
    document_ids = [metadata["document_ids"][int(index_value)] for index_value in indexes[0] if index_value >= 0]
    if not document_ids:
        return []
    placeholders = ",".join("?" for _ in document_ids)
    with connect(args.db) as connection:
        rows = connection.execute(f"SELECT * FROM documents WHERE document_id IN ({placeholders})", document_ids).fetchall()
    by_id = {row["document_id"]: row for row in rows}
    hits: list[SearchHit] = []
    for rank, (document_id, score) in enumerate(zip(document_ids, scores[0]), start=1):
        row = by_id.get(document_id)
        if row:
            hits.append(row_to_hit(row, semantic_rank=rank, semantic_score=float(score)))
    return hits


def query_entity_keys(query: str) -> set[str]:
    entities = extract_entities(query)
    values = [entity["text"] for entity in entities]
    if not values:
        values = TOKEN_PATTERN.findall(query)
    return {value.casefold() for value in values if len(value) >= 2}


def merge_hits(
    args: argparse.Namespace,
    bm25_hits: list[SearchHit],
    semantic_hits: list[SearchHit],
    entity_hits: list[SearchHit] | None = None,
) -> list[SearchHit]:
    by_id: dict[str, SearchHit] = {}
    for hit in bm25_hits + semantic_hits + (entity_hits or []):
        current = by_id.get(hit.document_id)
        if current is None:
            by_id[hit.document_id] = hit
            continue
        by_id[hit.document_id] = SearchHit(
            **{
                **current.__dict__,
                "bm25_rank": current.bm25_rank or hit.bm25_rank,
                "bm25_score": current.bm25_score if current.bm25_score is not None else hit.bm25_score,
                "semantic_rank": current.semantic_rank or hit.semantic_rank,
                "semantic_score": current.semantic_score if current.semantic_score is not None else hit.semantic_score,
                "entity_rank": current.entity_rank or hit.entity_rank,
                "entity_score": current.entity_score if current.entity_score is not None else hit.entity_score,
                "entity_overlap": max(current.entity_overlap, hit.entity_overlap),
            }
        )

    with connect(args.db) as connection:
        query_entities = resolve_query_entity_keys(connection, args.query)
    ranked: list[SearchHit] = []
    for hit in by_id.values():
        entity_values = {search_key(entity["text"]) for entity in hit.entities}
        overlap = len(query_entities & entity_values)
        score = 0.0
        if hit.bm25_rank:
            score += args.bm25_weight / (args.rrf_k + hit.bm25_rank)
        if hit.semantic_rank:
            score += args.semantic_weight / (args.rrf_k + hit.semantic_rank)
        if hit.entity_rank:
            score += args.entity_weight / (args.rrf_k + hit.entity_rank)
        score += 0.005 * args.entity_weight * max(overlap, hit.entity_overlap)
        ranked.append(
            SearchHit(
                **{
                    **hit.__dict__,
                    "entity_overlap": max(overlap, hit.entity_overlap),
                    "final_score": score,
                }
            )
        )
    return sorted(ranked, key=lambda hit: hit.final_score, reverse=True)


def temporal_deduplicate(hits: list[SearchHit], seconds: float, limit: int) -> list[SearchHit]:
    if seconds <= 0:
        return hits[:limit]
    output: list[SearchHit] = []
    for hit in hits:
        hit_terms = set(index_terms(hit.matched_passage or hit.combined_text))
        duplicate = any(
            previous.video_id == hit.video_id
            and (
                abs(previous.start_seconds - hit.start_seconds) <= seconds
                or (
                    abs(previous.start_seconds - hit.start_seconds) <= max(seconds * 3, 90)
                    and bool(hit_terms)
                    and len(hit_terms & set(index_terms(previous.matched_passage or previous.combined_text)))
                    / max(1, len(hit_terms | set(index_terms(previous.matched_passage or previous.combined_text))))
                    >= 0.55
                )
            )
            for previous in output
        )
        if duplicate:
            continue
        output.append(hit)
        if len(output) == limit:
            break
    return output


def hit_payload(hit: SearchHit) -> dict[str, Any]:
    return hit.__dict__


def print_hits(hits: list[SearchHit], as_json: bool) -> None:
    if as_json:
        print(json.dumps([hit_payload(hit) for hit in hits], ensure_ascii=False, indent=2))
        return
    for index, hit in enumerate(hits, start=1):
        display_passage = (hit.matched_passage or hit.combined_text).replace("SUBTITLE: ", "").replace(
            "LOCAL SUBTITLE: ", ""
        ).replace("OCR: ", "")
        snippet = " ".join(display_passage.split())[:320]
        frame = hit.matched_frame
        if frame is None:
            frame = hit.keyframe.get("source_frame_index") if hit.keyframe else hit.source_start_frame
        timestamp = hit.matched_timestamp if hit.matched_timestamp is not None else hit.start_seconds
        reranker = f" rerank={hit.reranker_score:.4f}" if hit.reranker_score is not None else ""
        print(
            f"{index}. score={hit.final_score:.6f} video={hit.video_id} scene={hit.scene_id} "
            f"time={timestamp:.3f}s frame={frame}{reranker}"
        )
        print(f"   {snippet}")
        print(f"   https://www.youtube.com/watch?v={hit.video_id}&t={max(0, round(timestamp))}s")


def search(args: argparse.Namespace) -> None:
    # On Windows, loading FAISS before a Transformers model can crash inside
    # PyTorch weight loading. Construct the reranker before semantic_search.
    reranker = load_reranker(args)
    bm25_hits = bm25_search(args.db, args.query, args.candidate_limit) if args.mode in {"bm25", "hybrid"} else []
    tfidf_hits = tfidf_search(args.db, args.query, args.candidate_limit) if args.mode == "tfidf" else []
    semantic_hits = semantic_search(args, args.candidate_limit) if args.mode in {"semantic", "hybrid"} else []
    entity_hits = entity_search(args.db, args.query, args.candidate_limit) if args.mode in {"entity", "hybrid"} else []
    if args.mode == "bm25":
        hits = [SearchHit(**{**hit.__dict__, "final_score": 1.0 / (args.rrf_k + (hit.bm25_rank or 1))}) for hit in bm25_hits]
    elif args.mode == "tfidf":
        hits = [SearchHit(**{**hit.__dict__, "final_score": float(hit.bm25_score or 0)}) for hit in tfidf_hits]
    elif args.mode == "semantic":
        hits = [SearchHit(**{**hit.__dict__, "final_score": float(hit.semantic_score or 0)}) for hit in semantic_hits]
    elif args.mode == "entity":
        hits = [SearchHit(**{**hit.__dict__, "final_score": float(hit.entity_score or 0)}) for hit in entity_hits]
    else:
        hits = merge_hits(args, bm25_hits, semantic_hits, entity_hits)
    hits = rerank_hits(args, hits, reranker)
    print_hits(temporal_deduplicate(hits, args.dedupe_seconds, args.limit), args.json)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.command == "build":
        build_index(
            args.documents,
            args.db,
            args.overwrite,
            args.aliases,
            args.ner_provider,
            args.ner_model,
            args.ner_device,
            args.ner_min_score,
        )
    elif args.command == "embed":
        build_embeddings(args)
    elif args.command == "search":
        search(args)


if __name__ == "__main__":
    main()
