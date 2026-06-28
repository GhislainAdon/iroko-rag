from __future__ import annotations

import json
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


SUMMARY_CHANNEL = "summary"
SUMMARY_INDEX_NAME = "summary"


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
        self.summary_index = SQLiteVecSemanticIndex(
            self.index_dir / f"{SUMMARY_INDEX_NAME}.sqlite"
        )

    @classmethod
    def from_provider(
        cls,
        index_dir: str | Path,
        *,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        embedding_timeout: float = 60,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        **kwargs: Any,
    ) -> "SemanticProjectionSearchBackend":
        return cls(
            index_dir,
            embedder=make_embedder(
                embedding_provider,
                embedding_model,
                dimensions=embedding_dimensions,
                timeout=embedding_timeout,
                api_key=embedding_api_key,
                base_url=embedding_base_url,
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
        if channel != SUMMARY_CHANNEL:
            raise ValueError(f"unsupported semantic channel: {channel}")
        if channel not in self.available_channels():
            return []
        query = normalize_text(query)
        if not query:
            return []
        vector = self.embedding_cache.embed_texts(
            [query],
            provider=self.embedding_provider,
            model=self.cache_model,
            embedder=self.embedder,
            batch_size=1,
        )[0]
        results = self.summary_index.search(
            vector,
            limit=limit,
            filters=filters,
            fetch_multiplier=self.fetch_multiplier,
        )
        return rank_single_semantic_channel(SUMMARY_CHANNEL, results)

    def available_channels(self) -> tuple[str, ...]:
        return (SUMMARY_CHANNEL,) if self._summary_document_count() > 0 else ()

    def info(self) -> dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "strategy": "semantic_channel_vector",
            "available_channels": list(self.available_channels()),
            "channels": {SUMMARY_CHANNEL: self._safe_summary_info()},
        }

    def _summary_document_count(self) -> int:
        info = self._safe_summary_info()
        if not info.get("available"):
            return 0
        return int(info.get("document_count") or 0)

    def _safe_summary_info(self) -> dict[str, Any]:
        index = self.summary_index
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
            self.index_dir / f"{SUMMARY_INDEX_NAME}.sqlite"
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
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
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
                api_key=embedding_api_key,
                base_url=embedding_base_url,
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
            Path(index_dir).expanduser() / f"{SUMMARY_INDEX_NAME}.sqlite"
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
            if len(vectors) != len(positions):
                raise ValueError(
                    "embedding response length mismatch: "
                    f"requested {len(positions)}, received {len(vectors)}"
                )
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
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dimensions: int,
        timeout: float,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.dimensions = dimensions
        if self.provider != "openai":
            raise ValueError(f"unknown embedding provider: {provider}")
        from openai import OpenAI

        if not api_key:
            raise ValueError("embedding_api_key is required for PIFS embeddings")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions > 0:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


def make_embedder(
    provider: str,
    model: str,
    *,
    dimensions: int,
    timeout: float,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    return EmbeddingClient(
        provider=provider,
        model=model,
        dimensions=dimensions,
        timeout=timeout,
        api_key=api_key,
        base_url=base_url,
    )


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

def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def embedding_cache_model_key(model: str, dimensions: int) -> str:
    return f"{model}:dimensions={dimensions}" if dimensions > 0 else model


def embed_with_retry(embedder: Any, texts: list[str], *, max_attempts: int = 8) -> list[list[float]]:
    for attempt in range(1, max_attempts + 1):
        try:
            return embedder.embed(texts)
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_embedding_error(exc):
                raise
            time.sleep(min(120.0, 2.0 ** (attempt - 1)))
    raise RuntimeError("unreachable embedding retry state")


def is_retryable_embedding_error(exc: Exception) -> bool:
    retryable = getattr(exc, "retryable", None)
    if isinstance(retryable, bool):
        return retryable
    status_code = getattr(exc, "status_code", None)
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        status = None
    if status is not None:
        if status in {408, 409, 429}:
            return True
        if status >= 500:
            return True
        if 400 <= status < 500:
            return False
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "connection", "ratelimit"))


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
