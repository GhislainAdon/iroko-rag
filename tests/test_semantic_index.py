import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pageindex.filesystem.semantic_index import (
    SemanticIndexRecord,
    SQLiteVecSemanticIndex,
)


class FixedDimensionEmbedder:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed(self, texts):
        return [[1.0, *([0.0] * (self.dimensions - 1))] for _ in texts]


def test_sqlite_vec_semantic_index_round_trip(tmp_path):
    index = SQLiteVecSemanticIndex(tmp_path / "semantic.sqlite")
    index.reset(dimension=3, metadata={"field_mode": "summary"})

    index.upsert_many(
        [
            SemanticIndexRecord(
                file_ref="file_a",
                external_id="doc_a",
                source_type="github",
                source_path="github/a.json",
                title="Multipart upload limits",
                text="multipart upload limits",
                vector=[1.0, 0.0, 0.0],
                metadata={"topic": "uploads"},
            ),
            SemanticIndexRecord(
                file_ref="file_b",
                external_id="doc_b",
                source_type="slack",
                source_path="slack/b.json",
                title="GPU cache issue",
                text="gpu cache issue",
                vector=[0.0, 1.0, 0.0],
                metadata={"topic": "runtime"},
            ),
        ]
    )

    assert index.info()["document_count"] == 2

    results = index.search([0.9, 0.1, 0.0], limit=2)
    assert [item.external_id for item in results] == ["doc_a", "doc_b"]

    filtered = index.search(
        [0.9, 0.1, 0.0],
        limit=2,
        filters={"source_type": "slack"},
    )
    assert [item.external_id for item in filtered] == ["doc_b"]


def test_sqlite_vec_semantic_index_file_ref_filter_not_limited_by_global_rank(tmp_path):
    index = SQLiteVecSemanticIndex(tmp_path / "semantic.sqlite")
    index.reset(dimension=2, metadata={"field_mode": "summary"})

    records = [
        SemanticIndexRecord(
            file_ref=f"file_off_{item:02d}",
            external_id=f"doc_off_{item:02d}",
            source_type="documents",
            source_path=f"other/{item:02d}.pdf",
            title=f"Off scope {item:02d}",
            text="off scope",
            vector=[1.0, 0.0],
        )
        for item in range(30)
    ]
    records.append(
        SemanticIndexRecord(
            file_ref="file_in_scope",
            external_id="doc_in_scope",
            source_type="documents",
            source_path="documents/in-scope.pdf",
            title="In scope",
            text="in scope",
            vector=[0.0, 1.0],
        )
    )
    index.upsert_many(records)

    results = index.search(
        [1.0, 0.0],
        limit=1,
        filters={"file_ref": ["file_in_scope"]},
    )

    assert [item.file_ref for item in results] == ["file_in_scope"]


def test_summary_projection_indexes_unified_metadata_summary(tmp_path):
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    indexer = SummaryProjectionIndexer(
        tmp_path / "projection",
        embedder=FixedDimensionEmbedder(3),
        embedding_provider="test",
        embedding_model="fake",
        embedding_dimensions=3,
    )

    result = indexer.upsert_summary(
        {
            "file_ref": "file_a",
            "external_id": "doc_a",
            "source_type": "documents",
            "source_path": "docs/a.pdf",
            "title": "A",
            "metadata": {
                "summary": "Unified metadata summary.",
                "department": "ops",
            },
        }
    )

    assert result["status"] == "ready"
    hits = indexer.index.search([1.0, 0.0, 0.0], limit=1)
    assert hits[0].external_id == "doc_a"
    assert hits[0].metadata["summary"] == "Unified metadata summary."
    assert hits[0].metadata["department"] == "ops"


def test_summary_projection_indexer_defaults_to_1024_dimensions(tmp_path):
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    indexer = SummaryProjectionIndexer(
        tmp_path / "projection",
        embedder=FixedDimensionEmbedder(1024),
        embedding_provider="test",
        embedding_model="fake",
    )

    info = indexer.index.info()

    assert info["dimension"] == 1024
    assert info["metadata"]["embedding_dimensions"] == 1024

    result = indexer.upsert_summary(
        {
            "file_ref": "file_a",
            "external_id": "doc_a",
            "source_type": "documents",
            "source_path": "docs/a.pdf",
            "title": "A",
            "metadata": {"summary": "Default dimension summary."},
        }
    )

    assert result["status"] == "ready"
    assert result["embedding_dimensions"] == 1024


def test_summary_projection_indexer_allows_explicit_256_dimensions(tmp_path):
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    indexer = SummaryProjectionIndexer(
        tmp_path / "projection",
        embedder=FixedDimensionEmbedder(256),
        embedding_provider="test",
        embedding_model="fake",
        embedding_dimensions=256,
    )

    assert indexer.index.info()["dimension"] == 256
    assert indexer.upsert_summary(
        {
            "file_ref": "file_a",
            "external_id": "doc_a",
            "source_type": "documents",
            "source_path": "docs/a.pdf",
            "title": "A",
            "metadata": {"summary": "Explicit 256 dimension summary."},
        }
    )["status"] == "ready"


def test_summary_projection_default_rejects_existing_256_index_for_writes(tmp_path):
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    index_dir = tmp_path / "projection"
    index = SQLiteVecSemanticIndex(index_dir / "summary_only_vector.sqlite")
    index.reset(
        dimension=256,
        metadata={
            "channel": "summary",
            "embedding_provider": "test",
            "embedding_model": "fake",
            "embedding_dimensions": 256,
        },
    )

    with pytest.raises(RuntimeError, match="configured embedding_dimensions is 1024"):
        SummaryProjectionIndexer(
            index_dir,
            embedder=FixedDimensionEmbedder(1024),
            embedding_provider="test",
            embedding_model="fake",
        )

    assert SQLiteVecSemanticIndex(index.db_path).info()["dimension"] == 256


def test_summary_projection_from_provider_rejects_dimension_mismatch_before_embedder(
    tmp_path, monkeypatch
):
    from pageindex.filesystem import projection_indexing
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    index_dir = tmp_path / "projection"
    index = SQLiteVecSemanticIndex(index_dir / "summary_only_vector.sqlite")
    index.reset(
        dimension=256,
        metadata={
            "channel": "summary",
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 256,
        },
    )

    def fail_make_embedder(*args, **kwargs):
        raise AssertionError("embedder should not be constructed before dimension validation")

    monkeypatch.setattr(projection_indexing, "make_embedder", fail_make_embedder)

    with pytest.raises(RuntimeError, match="configured embedding_dimensions is 1024"):
        SummaryProjectionIndexer.from_provider(index_dir)


def test_embedding_cache_key_separates_model_dimensions(tmp_path):
    from pageindex.filesystem.hybrid_projection import (
        EmbeddingCache,
        embedding_cache_model_key,
    )

    class CountingEmbedder:
        def __init__(self, dimensions: int):
            self.dimensions = dimensions
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[float(self.dimensions), *([0.0] * (self.dimensions - 1))] for _ in texts]

    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    embedder_256 = CountingEmbedder(256)
    embedder_1024 = CountingEmbedder(1024)
    key_256 = embedding_cache_model_key("fake", 256)
    key_1024 = embedding_cache_model_key("fake", 1024)

    assert key_256 != key_1024

    vector_256 = cache.embed_texts(
        ["same text"],
        provider="test",
        model=key_256,
        embedder=embedder_256,
        batch_size=1,
    )[0]
    vector_1024 = cache.embed_texts(
        ["same text"],
        provider="test",
        model=key_1024,
        embedder=embedder_1024,
        batch_size=1,
    )[0]

    assert len(vector_256) == 256
    assert len(vector_1024) == 1024
    assert embedder_256.calls == 1
    assert embedder_1024.calls == 1


def test_summary_projection_dimension_mismatch_preserves_existing_index(tmp_path):
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    index_dir = tmp_path / "projection"
    index = SQLiteVecSemanticIndex(index_dir / "summary_only_vector.sqlite")
    index.reset(
        dimension=3,
        metadata={
            "channel": "summary",
            "embedding_provider": "test",
            "embedding_model": "fake",
            "embedding_dimensions": 3,
        },
    )
    index.upsert_many(
        [
            SemanticIndexRecord(
                file_ref="file_a",
                external_id="doc_a",
                source_type="documents",
                source_path="docs/a.pdf",
                title="A",
                text="summary",
                vector=[1.0, 0.0, 0.0],
            )
        ]
    )

    with pytest.raises(RuntimeError, match="summary projection index dimension mismatch"):
        SummaryProjectionIndexer(
            index_dir,
            embedder=FixedDimensionEmbedder(4),
            embedding_provider="test",
            embedding_model="fake",
            embedding_dimensions=4,
        )

    preserved = SQLiteVecSemanticIndex(index.db_path)
    assert preserved.info()["dimension"] == 3
    assert preserved.info()["document_count"] == 1
    assert preserved.search([1.0, 0.0, 0.0], limit=1)[0].external_id == "doc_a"


def test_hash_embedding_provider_is_not_available():
    from pageindex.filesystem.hybrid_projection import make_embedder

    with pytest.raises(ValueError, match="unknown embedding provider: hash"):
        make_embedder("hash", "unused", dimensions=256, timeout=1)
