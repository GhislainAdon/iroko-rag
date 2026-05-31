from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import DEFAULT_EMBEDDING_DIMENSIONS
from .semantic_index import (
    SQLiteVecSemanticIndex,
    SemanticIndexError,
    SemanticIndexRecord,
    SemanticSearchResult,
)


INDEX_BY_CHANNEL = {
    "summary": "summary_only_vector",
    "entity": "entity_vectors",
    "relation": "relation_vectors",
}
SEMANTIC_TOOL_CHANNELS = ("summary", "entity", "relation")


@dataclass(frozen=True)
class QueryProjection:
    entities: list[str]
    relations: list[str]


@dataclass(frozen=True)
class SemanticProjectionCandidate:
    document_id: str
    score: float
    sources: list[dict[str, Any]]
    source_type: str
    title: str
    metadata: dict[str, Any]
    snippet: str


class SemanticProjectionSearchBackend:
    """Semantic channel retrieval over rebuildable projection indexes.

    The SQLite catalog remains the source of truth. This backend only reads
    external sqlite-vec projection indexes and returns candidate document ids
    for the catalog to resolve and filter.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedder: Any,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_cache_path: str | Path | None = None,
        fetch_multiplier: int = 100,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser()
        self.embedder = embedder
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.cache_model = embedding_cache_model_key(embedding_model, embedding_dimensions)
        self.embedding_cache = EmbeddingCache(
            Path(embedding_cache_path).expanduser()
            if embedding_cache_path is not None
            else self.index_dir / "embedding_cache.sqlite"
        )
        self.fetch_multiplier = fetch_multiplier
        self.indexes = {
            channel: SQLiteVecSemanticIndex(self.index_dir / f"{index_name}.sqlite")
            for channel, index_name in INDEX_BY_CHANNEL.items()
        }

    @classmethod
    def from_provider(
        cls,
        index_dir: str | Path,
        *,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_timeout: float = 60,
        **kwargs: Any,
    ) -> "SemanticProjectionSearchBackend":
        return cls(
            index_dir,
            embedder=make_embedder(
                embedding_provider,
                embedding_model,
                dimensions=embedding_dimensions,
                timeout=embedding_timeout,
            ),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            **kwargs,
        )

    def search_channel(
        self,
        channel: str,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SemanticProjectionCandidate]:
        if channel not in SEMANTIC_TOOL_CHANNELS:
            raise ValueError(f"unsupported semantic channel: {channel}")
        if channel not in self.available_channels():
            return []
        query = normalize_text(query)
        if not query:
            return []
        projection = heuristic_query_projection(query)
        vector = self.embedding_cache.embed_texts(
            [query_text_for_channel(channel, query, projection)],
            provider=self.embedding_provider,
            model=self.cache_model,
            embedder=self.embedder,
            batch_size=1,
        )[0]
        results = self.indexes[channel].search(
            vector,
            limit=limit,
            filters=filters,
            fetch_multiplier=self.fetch_multiplier,
        )
        return rank_single_semantic_channel(channel, results)

    def available_channels(self) -> tuple[str, ...]:
        return tuple(
            channel
            for channel in SEMANTIC_TOOL_CHANNELS
            if self._channel_document_count(channel) > 0
        )

    def info(self) -> dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "strategy": "semantic_channel_vector",
            "available_channels": list(self.available_channels()),
            "channels": {
                channel: self._safe_channel_info(channel)
                for channel in self.indexes
            },
        }

    def _channel_document_count(self, channel: str) -> int:
        info = self._safe_channel_info(channel)
        if not info.get("available"):
            return 0
        return int(info.get("document_count") or 0)

    def _safe_channel_info(self, channel: str) -> dict[str, Any]:
        index = self.indexes[channel]
        if not index.db_path.exists():
            return {
                "db_path": str(index.db_path),
                "available": False,
                "document_count": 0,
                "error": "index file is missing",
            }
        try:
            info = index.info()
        except (OSError, sqlite3.Error, SemanticIndexError) as exc:
            return {
                "db_path": str(index.db_path),
                "available": False,
                "document_count": 0,
                "error": str(exc),
            }
        return {**info, "available": int(info.get("document_count") or 0) > 0}


class SummaryProjectionIndexer:
    """Synchronous register-time summary projection indexer."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedder: Any,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_cache_path: str | Path | None = None,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.cache_model = embedding_cache_model_key(embedding_model, embedding_dimensions)
        self.embedding_cache = EmbeddingCache(
            Path(embedding_cache_path).expanduser()
            if embedding_cache_path is not None
            else self.index_dir / "embedding_cache.sqlite"
        )
        self.index = SQLiteVecSemanticIndex(
            self.index_dir / f"{INDEX_BY_CHANNEL['summary']}.sqlite"
        )
        self._ensure_index()

    @classmethod
    def from_provider(
        cls,
        index_dir: str | Path,
        *,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_timeout: float = 60,
        **kwargs: Any,
    ) -> "SummaryProjectionIndexer":
        cls._validate_existing_index_dimension(index_dir, embedding_dimensions)
        return cls(
            index_dir,
            embedder=make_embedder(
                embedding_provider,
                embedding_model,
                dimensions=embedding_dimensions,
                timeout=embedding_timeout,
            ),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            **kwargs,
        )

    def upsert_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        summary = str((record.get("metadata") or {}).get("summary") or "").strip()
        if not summary:
            return {"status": "skipped", "reason": "missing_summary"}
        vector = self.embedding_cache.embed_texts(
            [summary],
            provider=self.embedding_provider,
            model=self.cache_model,
            embedder=self.embedder,
            batch_size=1,
        )[0]
        metadata = dict(record.get("metadata") or {})
        count = self.index.upsert_many(
            [
                SemanticIndexRecord(
                    file_ref=str(record["file_ref"]),
                    vector=vector,
                    text=summary,
                    external_id=record.get("external_id"),
                    source_type=str(record.get("source_type") or ""),
                    title=str(record.get("title") or ""),
                    metadata=metadata,
                )
            ]
        )
        return {
            "status": "ready",
            "indexed_rows": count,
            "index_path": str(self.index.db_path),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
        }

    def delete_summary(self, file_ref: str) -> int:
        return self.index.delete_file_refs([file_ref])

    def _ensure_index(self) -> None:
        if not self.index.db_path.exists():
            self.index.reset(
                dimension=self.embedding_dimensions,
                metadata=self._index_metadata(),
            )
            return
        try:
            existing_dimension = self.index.dimension()
        except Exception as exc:
            raise RuntimeError(
                "could not validate existing summary projection index config; "
                f"refusing to reset {self.index.db_path}. Move the existing index "
                "aside or rebuild it intentionally before changing embedding config."
            ) from exc
        if existing_dimension != self.embedding_dimensions:
            raise self._dimension_mismatch_error(
                self.index.db_path,
                existing_dimension,
                self.embedding_dimensions,
            )

    def _index_metadata(self) -> dict[str, Any]:
        return {
            "channel": "summary",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
        }

    @classmethod
    def _validate_existing_index_dimension(
        cls,
        index_dir: str | Path,
        embedding_dimensions: int,
    ) -> None:
        index_path = (
            Path(index_dir).expanduser() / f"{INDEX_BY_CHANNEL['summary']}.sqlite"
        )
        if not index_path.exists():
            return
        index = SQLiteVecSemanticIndex(index_path)
        try:
            existing_dimension = index.dimension()
        except Exception as exc:
            raise RuntimeError(
                "could not validate existing summary projection index config; "
                f"refusing to reset {index_path}. Move the existing index "
                "aside or rebuild it intentionally before changing embedding config."
            ) from exc
        if existing_dimension != embedding_dimensions:
            raise cls._dimension_mismatch_error(
                index_path,
                existing_dimension,
                embedding_dimensions,
            )

    @staticmethod
    def _dimension_mismatch_error(
        index_path: Path,
        existing_dimension: int,
        embedding_dimensions: int,
    ) -> RuntimeError:
        return RuntimeError(
            "summary projection index dimension mismatch: "
            f"{index_path} was built with dimension {existing_dimension}, "
            f"but configured embedding_dimensions is {embedding_dimensions}. "
            "Use the matching embedding config, or rebuild the projection index "
            "at a new path after preserving the existing data."
        )


class EmbeddingCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_blob BLOB,
                    vector_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(provider, model, text_hash)
                )
                """
            )
            conn.commit()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def embed_texts(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        embedder: Any,
        batch_size: int,
    ) -> list[list[float]]:
        hashes = [SQLiteVecSemanticIndex.text_hash(text) for text in texts]
        cached: dict[str, list[float]] = {}
        with self.connect() as conn:
            for text_hash in sorted(set(hashes)):
                row = conn.execute(
                    """
                    SELECT vector_blob, vector_json
                    FROM embedding_cache
                    WHERE provider = ? AND model = ? AND text_hash = ?
                    """,
                    (provider, model, text_hash),
                ).fetchone()
                if row is not None:
                    cached[text_hash] = decode_vector(row["vector_blob"], row["vector_json"])
        missing_positions = [
            index for index, text_hash in enumerate(hashes) if text_hash not in cached
        ]
        for start in range(0, len(missing_positions), max(1, batch_size)):
            positions = missing_positions[start : start + max(1, batch_size)]
            batch_texts = [texts[index] for index in positions]
            vectors = embed_with_retry(embedder, batch_texts)
            with self.connect() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embedding_cache(
                        provider, model, text_hash, dimension, vector_blob, vector_json
                    )
                    VALUES (?, ?, ?, ?, ?, '')
                    """,
                    [
                        (
                            provider,
                            model,
                            hashes[index],
                            len(vector),
                            encode_vector(vector),
                        )
                        for index, vector in zip(positions, vectors)
                    ],
                )
                conn.commit()
            for index, vector in zip(positions, vectors):
                cached[hashes[index]] = vector
        return [cached[text_hash] for text_hash in hashes]


class EmbeddingClient:
    def __init__(self, *, provider: str, model: str, dimensions: int, timeout: float):
        self.provider = provider.lower()
        self.model = model
        self.dimensions = dimensions
        if self.provider != "openai":
            raise ValueError(f"unknown embedding provider: {provider}")
        from openai import OpenAI

        api_key = os.environ.get("PIFS_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("PIFS_EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError(
                "PIFS_EMBEDDING_API_KEY or OPENAI_API_KEY is required for PIFS embeddings"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions > 0:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


def make_embedder(provider: str, model: str, *, dimensions: int, timeout: float) -> Any:
    return EmbeddingClient(
        provider=provider,
        model=model,
        dimensions=dimensions,
        timeout=timeout,
    )


def query_text_for_channel(channel: str, query: str, projection: QueryProjection) -> str:
    if channel == "summary":
        return query
    if channel == "entity":
        return compact_join(projection.entities, limit=24) or query
    if channel == "relation":
        return "\n".join(projection.relations) or query
    raise ValueError(f"unknown semantic channel: {channel}")


def rank_single_semantic_channel(
    channel: str,
    results: list[SemanticSearchResult],
) -> list[SemanticProjectionCandidate]:
    rows: list[SemanticProjectionCandidate] = []
    seen: set[str] = set()
    for rank, result in enumerate(results, 1):
        doc_id = str(result.external_id or result.file_ref)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rows.append(
            SemanticProjectionCandidate(
                document_id=doc_id,
                score=1 / (60 + rank),
                sources=[{"channel": channel, "rank": rank, "distance": result.distance}],
                source_type=result.source_type,
                title=result.title,
                metadata=result.metadata,
                snippet=f"{channel}_vector rank={rank}",
            )
        )
    return rows


def heuristic_query_projection(question: str) -> QueryProjection:
    entities = dedupe(
        [
            *identifier_terms(question),
            *keyword_terms(question)[:16],
        ]
    )[:16]
    predicate = infer_query_predicate(question)
    subject = entities[0] if entities else "question"
    return QueryProjection(
        entities=entities,
        relations=[f"{subject} | {predicate} | {question}"],
    )


def compact_join(values: list[str], *, limit: int) -> str:
    return " | ".join(values[:limit])


def identifier_terms(text: str) -> list[str]:
    patterns = [
        r"\b[A-Z]{2,12}-\d{2,}\b",
        r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b\s*(?:=|:)\s*[A-Za-z0-9_.:/-]+",
        r"\b[A-Za-z][A-Za-z0-9_+-]+(?:[-_+][A-Za-z0-9]+)+\b",
        r"\b[A-Z]{2,}[A-Za-z0-9_-]*\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.strip() for match in re.findall(pattern, text))
    return found


def keyword_terms(text: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "for",
        "from",
        "how",
        "into",
        "the",
        "this",
        "that",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text)
        if term.lower() not in stopwords
    ]
    return dedupe(terms)


def infer_query_predicate(question: str) -> str:
    lowered = question.lower()
    rules = [
        ("asks_default", ["default", "defaults"]),
        ("asks_limit", ["limit", "maximum", "minimum", "size"]),
        ("asks_cause", ["caused", "cause", "why"]),
        ("asks_owner", ["who", "owner", "assigned"]),
        ("asks_deadline", ["when", "deadline", "date"]),
        ("asks_status", ["status", "state"]),
        ("asks_requirement", ["required", "requirement", "must"]),
    ]
    for predicate, needles in rules:
        if any(needle in lowered for needle in needles):
            return predicate
    return "asks_about"


def dedupe(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def embedding_cache_model_key(model: str, dimensions: int) -> str:
    return f"{model}:dimensions={dimensions}" if dimensions > 0 else model


def embed_with_retry(embedder: Any, texts: list[str], *, max_attempts: int = 8) -> list[list[float]]:
    for attempt in range(1, max_attempts + 1):
        try:
            return embedder.embed(texts)
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(min(120.0, 2.0 ** (attempt - 1)))
    raise RuntimeError("unreachable embedding retry state")


def encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(blob: bytes | None, vector_json: str | None) -> list[float]:
    if blob:
        if len(blob) % 4 != 0:
            raise ValueError("invalid cached vector blob length")
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))
    if vector_json:
        value = json.loads(vector_json)
        if isinstance(value, list):
            return [float(item) for item in value]
    raise ValueError("cached embedding row does not contain a vector")
