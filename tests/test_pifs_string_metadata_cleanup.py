import json
from pathlib import Path

import pytest


class StaticEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def _patch_summary_indexer(monkeypatch):
    from pageindex.filesystem.semantic_projection import SummaryProjectionIndexer

    def fake_from_provider(index_dir, **kwargs):
        return SummaryProjectionIndexer(
            Path(index_dir),
            embedder=StaticEmbedder(),
            embedding_provider="test",
            embedding_model="static",
            embedding_dimensions=3,
        )

    monkeypatch.setattr(SummaryProjectionIndexer, "from_provider", fake_from_provider)


def _patch_pageindex_client(monkeypatch, *, description="PageIndex summary"):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        doc_id = Path(file_path).stem + "_doc"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": Path(file_path).name,
            "doc_description": description,
            "line_count": 2,
            "structure": [{"title": "Doc", "node_id": "0001", "nodes": []}],
            "pages": [{"page": 1, "content": Path(file_path).read_text(encoding="utf-8")}],
        }
        self.documents[doc_id] = doc
        self._save_doc(doc_id)
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)


def _register_md(filesystem, tmp_path, monkeypatch, name, metadata=None):
    _patch_summary_indexer(monkeypatch)
    _patch_pageindex_client(monkeypatch, description=f"summary for {name}")
    path = tmp_path / name
    path.write_text(f"# {name}\n\nbody", encoding="utf-8")
    return filesystem.register_file(
        storage_uri=path.as_uri(),
        folder_path="/documents",
        title=name,
        content_type="text/markdown",
        metadata=metadata or {},
    )


def test_metadata_schema_is_string_registry_only(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    filesystem.metadata.register_schema(
        {"fields": {"ticker": {"description": "stock ticker"}}}
    )

    assert filesystem.metadata.export_schema() == {
        "fields": {"ticker": {"description": "stock ticker", "source": "manual"}}
    }
    with filesystem.store.connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        metadata_value_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(metadata_values)")
        }

    assert "metadata_schema" not in tables
    assert metadata_value_columns == {"file_ref", "field_id", "value_text", "created_at"}


def test_range_filters_are_removed(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem
    from pageindex.filesystem.metadata import MetadataQueryError

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    filesystem.metadata.register_schema({"fields": {"score": {}}})

    with pytest.raises(MetadataQueryError, match="Unsupported metadata operator: \\$gte"):
        filesystem.metadata.parse_filter({"score": {"$gte": 3}})


def test_numeric_metadata_filter_is_string_equality(tmp_path, monkeypatch):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    filesystem.metadata.register_schema({"fields": {"score": {}}})
    _register_md(filesystem, tmp_path, monkeypatch, "int.md", {"score": 3})
    _register_md(filesystem, tmp_path, monkeypatch, "float.md", {"score": 3.0})
    _register_md(filesystem, tmp_path, monkeypatch, "text.md", {"score": "3"})

    rows = filesystem.search(None, metadata_filter={"score": {"$eq": 3}})

    assert {row.title for row in rows} == {"int.md", "text.md"}


def test_register_uses_pageindex_description_as_summary_only(tmp_path, monkeypatch):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    file_ref = _register_md(filesystem, tmp_path, monkeypatch, "summary.md", {"ticker": "AAPL"})
    info = filesystem.store.file_info(file_ref)

    assert info["metadata"]["summary"] == "summary for summary.md"
    assert "summary" not in filesystem.metadata.export_schema()["fields"]
    with filesystem.store.connect() as conn:
        fields = [row["name"] for row in conn.execute("SELECT name FROM metadata_fields")]
    assert fields == ["ticker"]


def test_register_rejects_plain_text_without_pageindex(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    source = tmp_path / "plain.txt"
    source.write_text("plain text", encoding="utf-8")
    filesystem = PageIndexFileSystem(tmp_path / "workspace")

    with pytest.raises(ValueError, match="PDF/Markdown"):
        filesystem.register_file(
            storage_uri=source.as_uri(),
            folder_path="/documents",
            title="plain.txt",
            content_type="text/plain",
            content="plain text",
        )


def test_set_metadata_replaces_custom_metadata_without_touching_summary(tmp_path, monkeypatch):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    file_ref = _register_md(filesystem, tmp_path, monkeypatch, "meta.md", {"ticker": "AAPL"})

    info = filesystem.set_metadata(file_ref, {"ticker": "MSFT", "year": 2026})

    assert info["metadata"] == {
        "summary": "summary for meta.md",
        "ticker": "MSFT",
        "year": 2026,
    }
    assert {field.name for field in filesystem.store.list_metadata_fields()} == {
        "ticker",
        "year",
    }
    assert filesystem.set_metadata(file_ref, {}, clear=True)["metadata"] == {
        "summary": "summary for meta.md"
    }


def test_set_metadata_keeps_summary_out_of_ordinary_lexical_search(tmp_path, monkeypatch):
    from pageindex.filesystem import PageIndexFileSystem

    unique_summary_token = "summaryleaktoken"
    filesystem = PageIndexFileSystem(tmp_path / "workspace")
    _patch_summary_indexer(monkeypatch)
    _patch_pageindex_client(monkeypatch, description=unique_summary_token)
    path = tmp_path / "leak.md"
    path.write_text("# Leak\n\nordinary body", encoding="utf-8")
    file_ref = filesystem.register_file(
        storage_uri=path.as_uri(),
        folder_path="/documents",
        title="leak.md",
        content_type="text/markdown",
        metadata={"ticker": "AAPL"},
    )

    assert filesystem.search(unique_summary_token) == []

    filesystem.set_metadata(file_ref, {"ticker": "MSFT"})

    assert filesystem.search(unique_summary_token) == []
    assert {row.title for row in filesystem.search("MSFT")} == {"leak.md"}
