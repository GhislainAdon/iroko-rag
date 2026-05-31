from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import DEFAULT_EMBEDDING_DIMENSIONS
from .hybrid_projection import (
    EmbeddingCache,
    INDEX_BY_CHANNEL,
    embedding_cache_model_key,
    make_embedder,
)
from .semantic_index import SQLiteVecSemanticIndex, SemanticIndexRecord


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
                    source_path=str(record.get("source_path") or ""),
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
