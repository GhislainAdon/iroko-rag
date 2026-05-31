import json
import tempfile
from pathlib import Path

import pytest


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
        }
    }
    if doc.get("type") == "pdf":
        meta[doc_id]["page_count"] = doc.get("page_count")
    elif doc.get("type") == "md":
        meta[doc_id]["line_count"] = doc.get("line_count")
    (workspace / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class RecordingMetadataGenerator:
    values = {
        "summary": "Generated retrieval summary.",
        "doc_type": "technical_note",
        "domain": "documentation",
        "topic": "pageindex extraction",
    }

    def __init__(self):
        self.calls = []

    def generate(self, request, *, fields):
        self.calls.append((request, list(fields)))
        return {field: self.values[field] for field in fields if field in self.values}


def test_pageindex_structure_options_report_failed_register_build(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.md"
        source.write_text("# Report\n\nCached structure is not built yet.", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")

        def fail_index(*args, **kwargs):
            raise RuntimeError("index failed: extractor unavailable")

        monkeypatch.setattr(PageIndexClient, "index", fail_index)
        filesystem.register_file(
            storage_uri=source.as_uri(),
            external_id="dsid_structural_missing",
            title="Structural report",
            content=source.read_text(encoding="utf-8"),
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_structural_missing --structure"))
        pages = json.loads(executor.execute("cat dsid_structural_missing --page 1-2"))
        stat = json.loads(executor.execute("stat dsid_structural_missing"))

        assert structure["data"]["mode"] == "structure"
        assert structure["data"]["available"] is False
        assert structure["data"]["status"] == "failed"
        assert "RuntimeError: index failed: extractor unavailable" in structure["data"]["message"]
        assert stat["data"]["pageindex_tree_status"] == "failed"
        assert stat["data"]["metadata_status"]["pageindex_tree"] == {
            "status": "failed",
            "owner": "pageindex",
            "source": "PageIndexClient.index",
            "error_type": "RuntimeError",
            "message": "index failed: extractor unavailable",
        }

        assert pages["data"]["mode"] == "page"
        assert pages["data"]["available"] is False
        assert pages["data"]["pages"] == "1-2"

        assert "cp" not in executor.allowed_commands()
        assert "mkdir" not in executor.allowed_commands()


def test_register_pdf_markdown_uses_pageindex_extracted_text_for_metadata_and_fts(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PageIndexFileSystem

    def fake_index(self, file_path, mode="auto"):
        suffix = Path(file_path).suffix.lower()
        doc_id = f"doc_{suffix.lstrip('.')}"
        if suffix == ".pdf":
            doc = {
                "id": doc_id,
                "type": "pdf",
                "path": str(Path(file_path).resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 2,
                "structure": [{"title": "Report", "node_id": "0001", "nodes": []}],
                "pages": [
                    {"page": 1, "content": "PageIndex PDF extracted alpha text."},
                    {"page": 2, "content": "Second PageIndex PDF extracted beta text."},
                ],
            }
        else:
            doc = {
                "id": doc_id,
                "type": "md",
                "path": str(Path(file_path).resolve()),
                "doc_name": "notes",
                "doc_description": "",
                "line_count": 3,
                "structure": [
                    {
                        "title": "Notes",
                        "node_id": "0001",
                        "line_num": 1,
                        "text": "# Notes\n\nPageIndex Markdown extracted gamma text.",
                        "nodes": [],
                    }
                ],
                "pages": [
                    {"page": 1, "content": "PageIndex Markdown extracted gamma text."}
                ],
            }
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    with tempfile.TemporaryDirectory() as tmp:
        source_pdf = Path(tmp) / "report.pdf"
        source_md = Path(tmp) / "notes.md"
        source_pdf.write_bytes(b"%PDF-1.4\n% test fixture\n")
        source_md.write_text("# Notes\n\nCaller markdown content", encoding="utf-8")
        generator = RecordingMetadataGenerator()
        filesystem = PageIndexFileSystem(
            workspace=Path(tmp) / "workspace",
            metadata_generator=generator,
        )

        filesystem.register_file(
            storage_uri=source_pdf.as_uri(),
            external_id="dsid_pdf_extracted",
            title="PDF extracted",
            content="CALLER PDF CONTENT MUST NOT REACH GENERATOR",
        )
        filesystem.register_file(
            storage_uri=source_md.as_uri(),
            external_id="dsid_md_extracted",
            title="Markdown extracted",
            content="CALLER MD CONTENT MUST NOT REACH GENERATOR",
        )

        pdf_request = generator.calls[0][0]
        md_request = generator.calls[1][0]
        pdf_entry = filesystem.store.get_file(
            filesystem.store.resolve_file_ref("dsid_pdf_extracted")
        )
        md_entry = filesystem.store.get_file(
            filesystem.store.resolve_file_ref("dsid_md_extracted")
        )

        assert "PageIndex PDF extracted alpha text" in pdf_request.text
        assert "Second PageIndex PDF extracted beta text" in pdf_request.text
        assert "CALLER PDF CONTENT" not in pdf_request.text
        assert "PageIndex Markdown extracted gamma text" in md_request.text
        assert "CALLER MD CONTENT" not in md_request.text
        assert "PageIndex PDF extracted alpha text" in Path(
            pdf_entry.text_artifact_path
        ).read_text(encoding="utf-8")
        assert "PageIndex Markdown extracted gamma text" in Path(
            md_entry.text_artifact_path
        ).read_text(encoding="utf-8")
        assert [r.external_id for r in filesystem.search("alpha beta", limit=5)] == [
            "dsid_pdf_extracted"
        ]
        assert [r.external_id for r in filesystem.search("gamma", limit=5)] == [
            "dsid_md_extracted"
        ]
        assert filesystem.search("CALLER", limit=5) == []


def test_register_text_metadata_generation_keeps_caller_content_without_pageindex(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PageIndexFileSystem

    def fail_index(*args, **kwargs):
        raise AssertionError("PageIndexClient.index should not be called for text files")

    monkeypatch.setattr(PageIndexClient, "index", fail_index)
    with tempfile.TemporaryDirectory() as tmp:
        generator = RecordingMetadataGenerator()
        filesystem = PageIndexFileSystem(
            workspace=Path(tmp) / "workspace",
            metadata_generator=generator,
        )

        filesystem.register_file(
            storage_uri="file:///tmp/readme.txt",
            external_id="dsid_text_generation",
            title="Text generation",
            content="Plain text caller content stays authoritative.",
            content_type="text/plain",
        )

        stat = filesystem.store.file_info("dsid_text_generation")
        entry = filesystem.store.get_file(
            filesystem.store.resolve_file_ref("dsid_text_generation")
        )

        assert generator.calls[0][0].text == "Plain text caller content stays authoritative."
        assert stat["pageindex_doc_id"] is None
        assert stat["pageindex_tree_status"] == "not_built"
        assert Path(entry.text_artifact_path).read_text(
            encoding="utf-8"
        ) == "Plain text caller content stays authoritative."


def test_register_pdf_markdown_cache_miss_invokes_pageindex_client_index(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PageIndexFileSystem

    calls: list[str] = []

    def fake_index(self, file_path, mode="auto"):
        calls.append(str(file_path))
        doc_id = f"doc_{Path(file_path).suffix.lstrip('.')}"
        doc_type = "pdf" if Path(file_path).suffix == ".pdf" else "md"
        doc = {
            "id": doc_id,
            "type": doc_type,
            "path": str(Path(file_path).resolve()),
            "doc_name": Path(file_path).name,
            "doc_description": "",
            "structure": [{"title": Path(file_path).stem, "node_id": "0001", "nodes": []}],
        }
        if doc_type == "pdf":
            doc["page_count"] = 1
            doc["pages"] = [{"page": 1, "content": "Page one text"}]
        else:
            doc["line_count"] = 1
        write_pageindex_client_doc(self.workspace, doc_id, doc)
        self.documents[doc_id] = doc
        return doc_id

    monkeypatch.setattr(PageIndexClient, "index", fake_index)
    with tempfile.TemporaryDirectory() as tmp:
        source_pdf = Path(tmp) / "report.pdf"
        source_md = Path(tmp) / "notes.md"
        source_pdf.write_bytes(b"%PDF-1.4\n% test fixture\n")
        source_md.write_text("# Notes", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")

        filesystem.register_file(
            storage_uri=str(source_pdf),
            external_id="dsid_pdf_build",
            title="PDF build",
            content="pdf text",
        )
        filesystem.register_file(
            storage_uri=source_md.as_uri(),
            external_id="dsid_md_build",
            title="Markdown build",
            content=source_md.read_text(encoding="utf-8"),
        )

        pdf_stat = filesystem.store.file_info("dsid_pdf_build")
        md_stat = filesystem.store.file_info("dsid_md_build")

        assert calls == [str(source_pdf.resolve()), str(source_md.resolve())]
        assert pdf_stat["pageindex_doc_id"] == "doc_pdf"
        assert pdf_stat["pageindex_tree_status"] == "built"
        assert md_stat["pageindex_doc_id"] == "doc_md"
        assert md_stat["pageindex_tree_status"] == "built"


def test_cat_structure_page_reuses_pageindex_client_cache_without_indexing(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.pdf"
        source.write_bytes(b"%PDF-1.4\n% test fixture\n")
        workspace = Path(tmp) / "workspace"
        filesystem = PageIndexFileSystem(workspace=workspace)
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_cached_pdf",
            {
                "id": "doc_cached_pdf",
                "type": "pdf",
                "path": str(source.resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 2,
                "structure": [
                    {
                        "title": "Introduction",
                        "node_id": "0001",
                        "text": "Intro section text",
                        "nodes": [
                            {
                                "title": "Findings",
                                "node_id": "0002",
                                "physical_index": 2,
                                "nodes": [],
                            }
                        ],
                    }
                ],
                "pages": [
                    {"page": 1, "content": "Page one text"},
                    {"page": 2, "content": "Page two text"},
                ],
            },
        )

        def fail_index(*args, **kwargs):
            raise AssertionError("PageIndexClient.index should not be called on cache hit")

        monkeypatch.setattr(PageIndexClient, "index", fail_index)
        filesystem.register_file(
            storage_uri=source.as_uri(),
            external_id="dsid_structural_cached",
            title="Cached structural report",
            content="text artifact remains available for grep, not cat --all",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_structural_cached --structure"))
        pages = json.loads(executor.execute("cat dsid_structural_cached --page 1-2"))
        stat = json.loads(executor.execute("stat dsid_structural_cached"))

        assert structure["data"]["available"] is True
        assert structure["data"]["pageindex_doc_id"] == "doc_cached_pdf"
        assert structure["data"]["structure"][0]["title"] == "Introduction"
        assert structure["data"]["structure"][0]["nodes"][0]["title"] == "Findings"
        assert "structure_pagination" not in structure["data"]
        assert "text" not in structure["data"]["structure"][0]
        assert "text" not in structure["data"]["structure"][0]["nodes"][0]

        assert pages["data"]["available"] is True
        assert pages["data"]["text"] == "Page one text\n\nPage two text"
        with pytest.raises(PIFSCommandError, match="target-first"):
            executor.execute("cat --page 1-2 dsid_structural_cached")
        with pytest.raises(PIFSCommandError, match="one file target"):
            executor.execute("cat dsid_structural_cached --page 1 2")

        assert stat["data"]["pageindex_doc_id"] == "doc_cached_pdf"
        assert stat["data"]["pageindex_tree_status"] == "built"


def test_cat_node_is_not_supported():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        filesystem.register_file(
            storage_uri="file:///tmp/notes.md",
            external_id="dsid_md_cached",
            title="Cached markdown notes",
            content="# Notes\n\nBody",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        with pytest.raises(PIFSCommandError, match="Unsupported cat option: --node"):
            executor.execute("cat dsid_md_cached --node 0001")


def test_cat_structure_page_and_text_outputs_are_hard_limited():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.pdf"
        source.write_bytes(b"%PDF-1.4\n% test fixture\n")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        structure_nodes = [
            {
                "title": f"Section {index}",
                "node_id": f"{index:04d}",
                "start_index": index,
                "end_index": index,
                "text": f"node {index} text",
                "nodes": [],
            }
            for index in range(1, 31)
        ]
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_limited_pdf",
            {
                "id": "doc_limited_pdf",
                "type": "pdf",
                "path": str(source.resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 10,
                "structure": structure_nodes,
                "pages": [
                    {"page": index, "content": f"Page {index} text"}
                    for index in range(1, 11)
                ],
            },
        )
        filesystem.register_file(
            storage_uri=source.as_uri(),
            external_id="dsid_limited_pdf",
            title="Limited structural report",
            content="text artifact remains available for grep",
        )
        text_content = "\n".join(f"line {index}" for index in range(1, 106))
        filesystem.register_file(
            storage_uri="file:///tmp/long.txt",
            external_id="dsid_long_text",
            title="Long text",
            content=text_content,
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_limited_pdf --structure"))
        assert len(structure["data"]["structure"]) == 30
        assert structure["data"]["structure"][25]["node_id"] == "0026"
        assert "text" not in structure["data"]["structure"][0]
        assert "structure_pagination" not in structure["data"]
        with pytest.raises(PIFSCommandError, match="Unsupported cat option: --offset"):
            executor.execute("cat dsid_limited_pdf --structure --offset 25")

        pages = json.loads(executor.execute("cat dsid_limited_pdf --page 1-5"))
        assert pages["data"]["text"] == (
            "Page 1 text\n\nPage 2 text\n\nPage 3 text\n\nPage 4 text\n\nPage 5 text"
        )
        assert pages["data"]["page_pagination"]["limit"] == 5
        with pytest.raises(PIFSCommandError, match="at most 5"):
            executor.execute("cat dsid_limited_pdf --page 1-6")
        with pytest.raises(PIFSCommandError, match="evidence is sufficient"):
            executor.execute("cat dsid_limited_pdf --page 1-6")

        with pytest.raises(PIFSCommandError, match="Unsupported cat option: --node"):
            executor.execute("cat dsid_limited_pdf --node 0001")

        with pytest.raises(PIFSCommandError, match="quote the whole target"):
            executor.execute("cat dsid_limited_pdf 0001")

        text = json.loads(executor.execute("cat dsid_long_text --all"))
        assert "line 100" in text["data"]["text"]
        assert "line 101" not in text["data"]["text"]
        assert text["data"]["pagination"]["has_more"] is True
        assert text["data"]["pagination"]["next_range"] == "101-105"
        with pytest.raises(PIFSCommandError, match="at most 100"):
            executor.execute("cat dsid_long_text --range 1-101")


def test_tree_folder_behavior_is_preserved():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    with tempfile.TemporaryDirectory() as tmp:
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        filesystem.register_file(
            storage_uri="file:///tmp/report.txt",
            folder_path="/docs/reports",
            external_id="dsid_folder_tree",
            title="Folder report",
            content="folder tree behavior remains intact",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        folder_tree = json.loads(executor.execute("tree /docs --depth 2"))

        assert folder_tree["data"]["path"] == "/docs"
        assert folder_tree["data"]["folders"][0]["path"] == "/docs/reports"


def test_tree_does_not_read_file_internal_pageindex_structure():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "report.pdf"
        source.write_bytes(b"%PDF-1.4\n% test fixture\n")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_tree_is_folder_only",
            {
                "id": "doc_tree_is_folder_only",
                "type": "pdf",
                "path": str(source.resolve()),
                "doc_name": "report.pdf",
                "doc_description": "",
                "page_count": 1,
                "structure": [
                    {"title": "Introduction", "node_id": "0001", "nodes": []}
                ],
                "pages": [{"page": 1, "content": "Page one text"}],
            },
        )
        filesystem.register_file(
            storage_uri=source.as_uri(),
            external_id="dsid_tree_is_folder_only",
            title="Cached structural report",
            content="text artifact remains available",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        with pytest.raises(PIFSCommandError):
            executor.execute("tree dsid_tree_is_folder_only")

        structure = json.loads(executor.execute("cat dsid_tree_is_folder_only --structure"))
        assert structure["data"]["structure"][0]["title"] == "Introduction"


def test_cat_all_is_limited_to_text_files():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        filesystem.register_file(
            storage_uri="file:///tmp/readme.txt",
            external_id="dsid_text_file",
            title="Text readme",
            content="plain text body",
        )
        filesystem.register_file(
            storage_uri="file:///tmp/report.pdf",
            external_id="dsid_pdf_file",
            title="PDF report",
            content="extracted text should not be served through cat --all",
        )
        filesystem.register_file(
            storage_uri="file:///tmp/notes.md",
            external_id="dsid_md_file",
            title="Markdown notes",
            content="markdown text should use PageIndex structure reads",
        )
        filesystem.register_file(
            storage_uri="file:///tmp/data.json",
            external_id="dsid_json_file",
            title="JSON record",
            content='{"body":"json"}',
            content_type="application/json",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        text = json.loads(executor.execute("cat dsid_text_file --all"))
        assert text["data"]["text"] == "plain text body"

        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_pdf_file --all")
        with pytest.raises(ValueError, match="not supported for PDF/Markdown"):
            filesystem.open("dsid_pdf_file")
        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_md_file --all")
        with pytest.raises(ValueError, match="not supported for PDF/Markdown"):
            filesystem.open("dsid_md_file")
        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_json_file --all")
        opened_json = filesystem.open("dsid_json_file")
        assert opened_json.text == '{"body":"json"}'

def test_pageindex_structure_commands_are_limited_to_pdf_and_markdown():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        filesystem.register_file(
            storage_uri="file:///tmp/readme.txt",
            external_id="dsid_text_only",
            title="Text readme",
            content="plain text body",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        for command in (
            "cat dsid_text_only --structure",
            "cat dsid_text_only --page 1",
        ):
            with pytest.raises(PIFSCommandError, match="only supported for PDF/Markdown"):
                executor.execute(command)

        with pytest.raises(PIFSCommandError, match="Unsupported cat option: --node"):
            executor.execute("cat dsid_text_only --node 0001")


def test_existing_pageindex_status_allows_legacy_record_without_format_suffix():
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "uploaded"
        source.write_text("# Uploaded\n\nBody", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
        file_ref = filesystem.register_file(
            storage_uri=source.as_uri(),
            external_id="dsid_legacy_pageindex",
            title="Legacy PageIndex record",
            content="text/plain is only a weak default here",
        )
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_legacy_pageindex",
            {
                "id": "doc_legacy_pageindex",
                "type": "md",
                "path": str(source.resolve()),
                "doc_name": "uploaded",
                "doc_description": "",
                "line_count": 3,
                "structure": [
                    {"title": "Uploaded", "node_id": "0001", "text": "Body", "nodes": []}
                ],
            },
        )
        filesystem.store.update_pageindex_pointer(
            file_ref,
            pageindex_doc_id="doc_legacy_pageindex",
            pageindex_tree_status="built",
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_legacy_pageindex --structure"))
        assert structure["data"]["structure"][0]["title"] == "Uploaded"
        with pytest.raises(PIFSCommandError, match="only supported for txt/text files"):
            executor.execute("cat dsid_legacy_pageindex --all")


def test_read_commands_do_not_link_pageindex_cache_when_pointer_is_missing(monkeypatch):
    from pageindex import PageIndexClient
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "late.md"
        source.write_text("# Late\n\nBody", encoding="utf-8")
        filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")

        def fail_index(*args, **kwargs):
            raise RuntimeError("index failed")

        monkeypatch.setattr(PageIndexClient, "index", fail_index)
        filesystem.register_file(
            storage_uri=source.as_uri(),
            external_id="dsid_late_cache",
            title="Late cache",
            content=source.read_text(encoding="utf-8"),
        )
        write_pageindex_client_doc(
            filesystem.pageindex_client_workspace,
            "doc_late_cache",
            {
                "id": "doc_late_cache",
                "type": "md",
                "path": str(source.resolve()),
                "doc_name": "late",
                "doc_description": "",
                "line_count": 3,
                "structure": [
                    {"title": "Late", "node_id": "0001", "text": "Body", "nodes": []}
                ],
            },
        )
        executor = PIFSCommandExecutor(filesystem, json_output=True)

        structure = json.loads(executor.execute("cat dsid_late_cache --structure"))
        stat = json.loads(executor.execute("stat dsid_late_cache"))

        assert structure["data"]["available"] is False
        assert stat["data"]["pageindex_doc_id"] is None
        assert stat["data"]["pageindex_tree_status"] == "failed"
