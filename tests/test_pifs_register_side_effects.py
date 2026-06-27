from pathlib import Path

import pytest

from tests.pifs_markdown_fixture import register_markdown


class RecordingSummaryIndexer:
    def __init__(self):
        self.upserted = []
        self.deleted = []

    def upsert_summary(self, record):
        self.upserted.append(dict(record))
        return {"status": "ready"}

    def delete_summary(self, file_ref):
        self.deleted.append(file_ref)


def test_register_insert_failure_cleans_owned_artifacts_and_skips_projection(
    tmp_path: Path, monkeypatch
):
    from pageindex.filesystem import PageIndexFileSystem

    workspace = tmp_path / "workspace"
    indexer = RecordingSummaryIndexer()
    filesystem = PageIndexFileSystem(workspace=workspace)
    filesystem.summary_projection_indexer = indexer

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        register_markdown(
            filesystem,
            tmp_path,
            "doc_insert_failure",
            "/documents",
            title="insert_failure.md",
            text="Markdown content for registration.",
        )

    assert indexer.upserted == []
    assert list((workspace / "artifacts" / "raw").glob("*.json")) == []
    assert list((workspace / "artifacts" / "text").glob("*.txt")) == []


def test_register_failure_after_catalog_insert_cleans_catalog_and_projection(
    tmp_path: Path, monkeypatch
):
    from pageindex.filesystem import PageIndexFileSystem

    workspace = tmp_path / "workspace"
    indexer = RecordingSummaryIndexer()
    filesystem = PageIndexFileSystem(workspace=workspace)
    filesystem.summary_projection_indexer = indexer

    def fail_sync(record):
        raise RuntimeError("raw sync failed")

    monkeypatch.setattr(filesystem, "_sync_owned_raw_artifact", fail_sync)

    with pytest.raises(RuntimeError, match="raw sync failed"):
        register_markdown(
            filesystem,
            tmp_path,
            "doc_sync_failure",
            "/documents",
            title="sync_failure.md",
            text="Markdown content for registration.",
        )

    assert filesystem.search(None) == []
    assert indexer.deleted == [indexer.upserted[0]["file_ref"]]
