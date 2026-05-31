import json
from pathlib import Path

import pytest


class GeneratedMetadata:
    def __init__(self):
        self.calls = []

    def generate(self, request, *, fields):
        self.calls.append((request, list(fields)))
        values = {
            "summary": f"Summary for {request.title}: {request.text[:60]}",
            "doc_type": "uploaded_file",
            "domain": "workspace",
            "topic": "pifs add",
        }
        return {field: values[field] for field in fields if field in values}


class StaticEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


def make_summary_indexer(workspace: Path):
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    return SummaryProjectionIndexer(
        workspace / "artifacts" / "projection_indexes",
        embedder=StaticEmbedder(),
        embedding_provider="test",
        embedding_model="static",
        embedding_dimensions=3,
    )


def make_filesystem(workspace: Path):
    from pageindex.filesystem import PageIndexFileSystem

    return PageIndexFileSystem(
        workspace=workspace,
        metadata_generator=GeneratedMetadata(),
        summary_projection_indexer=make_summary_indexer(workspace),
        summary_projection_embedding_provider="test",
        summary_projection_embedding_model="static",
        summary_projection_embedding_dimensions=3,
    )


def write_pageindex_client_doc(workspace: Path, doc_id: str, doc: dict) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / f"{doc_id}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = {
        doc_id: {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
            "line_count": doc.get("line_count"),
        }
    }
    (workspace / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_add_text_folder_target_copies_artifact_indexes_summary_and_is_readable(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor

    source = tmp_path / "filing.txt"
    source.write_text("alpha filing text for pifs add", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    info = filesystem.add_file(str(source), "/documents/reports")

    assert info["source_path"] == "documents/reports/filing.txt"
    assert info["folder_path"] == "/documents/reports"
    assert filesystem.folder_info("/documents/reports")["path"] == "/documents/reports"
    assert info["storage_uri"] != source.as_uri()
    assert "/artifacts/uploads/" in info["storage_uri"]
    copied_path = Path(info["storage_uri"].removeprefix("file://"))
    assert copied_path.read_text(encoding="utf-8") == "alpha filing text for pifs add"
    assert copied_path.resolve() != source.resolve()

    executor = PIFSCommandExecutor(filesystem, json_output=True)
    rendered = json.loads(executor.execute("cat /documents/reports/filing.txt --all"))

    assert rendered["data"]["text"] == "alpha filing text for pifs add"
    assert info["metadata"]["summary"].startswith("Summary for filing.txt")
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 1


def test_add_rejects_same_folder_same_basename_without_overwrite(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor

    source = tmp_path / "conflict.txt"
    source.write_text("first body", encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    filesystem.add_file(source, "/documents")
    source.write_text("second body must not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        filesystem.add_file(source, "/documents")

    executor = PIFSCommandExecutor(filesystem, json_output=True)
    rendered = json.loads(executor.execute("cat /documents/conflict.txt --all"))
    assert rendered["data"]["text"] == "first body"


def test_add_rejects_unsupported_type_before_registration(tmp_path):
    source = tmp_path / "payload.json"
    source.write_text('{"unsupported": true}', encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    with pytest.raises(ValueError, match="Unsupported file type"):
        filesystem.add_file(source, "/documents")

    assert filesystem.browse("/", recursive=True)["files"] == []
    assert not list((tmp_path / "workspace" / "artifacts" / "uploads").glob("**/*"))


def test_add_rejects_disabled_summary_projection_before_registration(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    source = tmp_path / "disabled.txt"
    source.write_text("must not register without summary vector", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = PageIndexFileSystem(
        workspace=workspace,
        metadata_generator=GeneratedMetadata(),
        summary_projection_index=False,
    )

    with pytest.raises(RuntimeError, match="summary projection index"):
        filesystem.add_file(source, "/documents")

    assert filesystem.browse("/", recursive=True)["files"] == []
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_configures_semantic_retrieval_in_same_filesystem_instance(tmp_path):
    source = tmp_path / "semantic.txt"
    source.write_text("alpha semantic recall text", encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    assert filesystem.semantic_retrieval_channels() == ()

    filesystem.add_file(source, "/documents")

    assert filesystem.semantic_retrieval_channels() == ("summary",)
    results = filesystem.search_semantic_channel(
        "summary",
        "semantic recall",
        scope={"folder_path": "/documents", "recursive": True},
        limit=5,
    )
    assert [result.source_path for result in results] == ["documents/semantic.txt"]


def test_add_markdown_builds_pageindex_tree_from_copied_artifact(tmp_path, monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor

    indexed_paths = []

    def fake_index(self, file_path, mode="auto"):
        indexed_paths.append(Path(file_path))
        doc_id = "doc_added_md"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "notes.md",
            "doc_description": "",
            "line_count": 3,
            "structure": [
                {
                    "title": "Notes",
                    "node_id": "0001",
                    "line_num": 1,
                    "text": "# Notes\n\ncopied markdown body",
                    "nodes": [],
                }
            ],
        }
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n\ncopied markdown body", encoding="utf-8")
    filesystem = make_filesystem(tmp_path / "workspace")

    info = filesystem.add_file(source, "/documents")
    executor = PIFSCommandExecutor(filesystem, json_output=True)
    structure = json.loads(executor.execute("cat /documents/notes.md --structure"))

    assert structure["data"]["available"] is True
    assert structure["data"]["structure"][0]["title"] == "Notes"
    assert indexed_paths == [Path(info["storage_uri"].removeprefix("file://"))]
    assert indexed_paths[0].resolve() != source.resolve()


def test_add_failure_does_not_leave_visible_catalog_or_artifacts(tmp_path, monkeypatch):
    source = tmp_path / "atomic.txt"
    source.write_text("atomic body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        filesystem.add_file(source, "/documents")

    assert filesystem.browse("/", recursive=True)["files"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_markdown_insert_failure_removes_pageindex_cache(tmp_path, monkeypatch):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        doc_id = "doc_failed_add_md"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "failed.md",
            "doc_description": "",
            "line_count": 3,
            "structure": [
                {
                    "title": "Failed",
                    "node_id": "0001",
                    "line_num": 1,
                    "text": "# Failed\n\nbody",
                    "nodes": [],
                }
            ],
        }
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "failed.md"
    source.write_text("# Failed\n\nbody", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        filesystem.add_file(source, "/documents/reports")

    pageindex_workspace = workspace / "artifacts" / "pageindex_client"
    assert not (pageindex_workspace / "doc_failed_add_md.json").exists()
    meta_path = pageindex_workspace / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "doc_failed_add_md" not in meta
    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert listing["folders"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_markdown_index_failure_removes_pageindex_cache_delta(tmp_path, monkeypatch):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        doc_id = "doc_partial_before_raise"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "partial.md",
            "doc_description": "",
            "line_count": 3,
            "structure": [{"title": "Partial", "node_id": "0001", "nodes": []}],
        }
        self.documents[doc_id] = doc
        self._save_doc(doc_id)
        raise RuntimeError("index failed after cache write")

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "partial.md"
    source.write_text("# Partial\n\nbody", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)
    pageindex_workspace = workspace / "artifacts" / "pageindex_client"

    with pytest.raises(RuntimeError, match="failed to build PageIndex tree"):
        filesystem.add_file(source, "/documents/reports")

    assert not (pageindex_workspace / "doc_partial_before_raise.json").exists()
    meta_path = pageindex_workspace / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "doc_partial_before_raise" not in meta
    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert listing["folders"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_markdown_failure_preserves_unrelated_pageindex_cache(tmp_path, monkeypatch):
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        doc_id = "doc_failed_add_md"
        doc = {
            "id": doc_id,
            "type": "md",
            "path": str(Path(file_path).resolve()),
            "doc_name": "failed.md",
            "doc_description": "",
            "line_count": 3,
            "structure": [{"title": "Failed", "node_id": "0001", "nodes": []}],
        }
        self.documents[doc_id] = doc
        self._save_doc(doc_id)
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    source = tmp_path / "failed.md"
    source.write_text("# Failed\n\nbody", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)
    pageindex_workspace = workspace / "artifacts" / "pageindex_client"
    write_pageindex_client_doc(
        pageindex_workspace,
        "doc_unrelated",
        {
            "id": "doc_unrelated",
            "type": "md",
            "path": str((tmp_path / "unrelated.md").resolve()),
            "doc_name": "unrelated.md",
            "doc_description": "",
            "line_count": 1,
            "structure": [{"title": "Unrelated", "node_id": "0001", "nodes": []}],
        },
    )

    def fail_insert(records):
        raise RuntimeError("catalog insert failed")

    monkeypatch.setattr(filesystem.store, "insert_files", fail_insert)

    with pytest.raises(RuntimeError, match="catalog insert failed"):
        filesystem.add_file(source, "/documents")

    assert not (pageindex_workspace / "doc_failed_add_md.json").exists()
    assert (pageindex_workspace / "doc_unrelated.json").exists()
    meta = json.loads((pageindex_workspace / "_meta.json").read_text(encoding="utf-8"))
    assert "doc_failed_add_md" not in meta
    assert "doc_unrelated" in meta


def test_add_failure_after_summary_vector_rolls_back_catalog_and_vector(
    tmp_path, monkeypatch
):
    source = tmp_path / "post_vector.txt"
    source.write_text("post vector rollback body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_status_update(*args, **kwargs):
        raise RuntimeError("metadata status update failed")

    monkeypatch.setattr(filesystem.store, "update_file_metadata_status", fail_status_update)

    with pytest.raises(RuntimeError, match="metadata status update failed"):
        filesystem.add_file(source, "/documents")

    assert filesystem.browse("/", recursive=True)["files"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_failure_removes_nested_folders_created_only_for_add(tmp_path, monkeypatch):
    source = tmp_path / "nested.txt"
    source.write_text("nested rollback body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)

    def fail_status_update(*args, **kwargs):
        raise RuntimeError("metadata status update failed")

    monkeypatch.setattr(filesystem.store, "update_file_metadata_status", fail_status_update)

    with pytest.raises(RuntimeError, match="metadata status update failed"):
        filesystem.add_file(source, "/documents/reports")

    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert listing["folders"] == []
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    assert not list((workspace / "artifacts" / "text").glob("*.txt"))
    assert not list((workspace / "artifacts" / "raw").glob("*.json"))


def test_add_failure_preserves_preexisting_parent_folder(tmp_path, monkeypatch):
    source = tmp_path / "nested.txt"
    source.write_text("nested rollback body", encoding="utf-8")
    workspace = tmp_path / "workspace"
    filesystem = make_filesystem(workspace)
    filesystem.create_folder("/documents")

    def fail_status_update(*args, **kwargs):
        raise RuntimeError("metadata status update failed")

    monkeypatch.setattr(filesystem.store, "update_file_metadata_status", fail_status_update)

    with pytest.raises(RuntimeError, match="metadata status update failed"):
        filesystem.add_file(source, "/documents/reports")

    listing = filesystem.browse("/", recursive=True)
    assert listing["files"] == []
    assert [folder["path"] for folder in listing["folders"]] == ["/documents"]
    assert filesystem.summary_projection_indexer.index.info()["document_count"] == 0


def test_cli_add_uses_workspace_and_prints_added_file(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    source = tmp_path / "cli.txt"
    source.write_text("cli body", encoding="utf-8")
    calls = []

    class FakeAddFileSystem:
        def __init__(self, workspace):
            self.workspace = Path(workspace)

        def configure_existing_projection_retrieval(self):
            return False

        def add_file(self, physical_path, virtual_target):
            calls.append((self.workspace, physical_path, virtual_target))
            return {
                "file_ref": "file_cli",
                "path": "/documents/cli.txt",
                "source_path": "documents/cli.txt",
                "storage_uri": "file:///workspace/artifacts/uploads/file_cli/cli.txt",
            }

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeAddFileSystem)

    status = cli.main(["--workspace", str(tmp_path / "workspace"), "add", str(source), "/documents"])

    assert status == 0
    assert calls == [(tmp_path / "workspace", str(source), "/documents")]
    assert capsys.readouterr().out == (
        "added: /documents/cli.txt\n"
        "file_ref: file_cli\n"
        "storage_uri: file:///workspace/artifacts/uploads/file_cli/cli.txt\n"
    )
