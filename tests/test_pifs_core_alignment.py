import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


class BrowseBackend:
    def __init__(self, document_ids, channels=("summary",), file_refs_by_document_id=None):
        self.document_ids = list(document_ids)
        self.channels = channels
        self.file_refs_by_document_id = dict(file_refs_by_document_id or {})
        self.calls = []

    def available_channels(self):
        return self.channels

    def search_channel(self, channel, query, *, limit=10, filters=None):
        self.calls.append((channel, query, limit, filters))
        file_ref_filter = set()
        if isinstance(filters, dict):
            raw_file_refs = filters.get("file_ref") or []
            if isinstance(raw_file_refs, str):
                file_ref_filter = {raw_file_refs}
            else:
                file_ref_filter = {str(item) for item in raw_file_refs}
        document_ids = self.document_ids
        if file_ref_filter and self.file_refs_by_document_id:
            document_ids = [
                document_id
                for document_id in document_ids
                if self.file_refs_by_document_id.get(document_id) in file_ref_filter
            ]
        return [
            SimpleNamespace(
                document_id=document_id,
                snippet=f"{channel} candidate {rank}: {query}",
                score=1.0 - rank * 0.01,
                sources=[{"channel": channel, "rank": rank}],
            )
            for rank, document_id in enumerate(document_ids[:limit], 1)
        ]


def _payload(output):
    return json.loads(output)


def _register_file(filesystem, root, external_id, folder_path, *, title=None, text=None):
    source = root / f"{external_id}.txt"
    source.write_text(text or f"{external_id} alpha evidence", encoding="utf-8")
    return filesystem.register_file(
        storage_uri=source.as_uri(),
        folder_path=folder_path,
        external_id=external_id,
        title=title or f"{external_id}.txt",
        content=source.read_text(encoding="utf-8"),
        metadata={"department": "finance"},
    )


class PIFSCoreAlignmentTest(unittest.TestCase):
    def test_aligned_command_surface_and_json_errors(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            _register_file(filesystem, root, "doc_a", "/documents")
            executor = PIFSCommandExecutor(filesystem)

            self.assertEqual(
                executor.allowed_commands(),
                {"browse", "cat", "grep", "ls", "stat", "tree"},
            )

            for command in (
                "find /documents",
                "grep -R alpha /documents",
                "grep alpha /documents",
                "stat --schema",
                "stat --field department doc_a",
                "stat doc_a doc_a",
                "cat doc_a",
                "cat doc_a --all",
                "tree /documents | grep doc",
                'browse /documents "alpha" --space entity',
                'browse /documents "alpha" --json',
            ):
                result = _payload(executor.execute(command))
                self.assertFalse(result["success"])
                self.assertEqual(result["error"]["code"], "invalid_command")
                self.assertEqual(result["next_steps"], [])

    def test_ls_is_tree_depth_one_alias(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            _register_file(filesystem, root, "doc_root", "/documents")
            _register_file(filesystem, root, "doc_child", "/documents/team")
            executor = PIFSCommandExecutor(filesystem)
            calls = []
            original = executor._cmd_tree

            def record_tree(args):
                calls.append(args)
                return original(args)

            executor._cmd_tree = record_tree

            ls_result = _payload(executor.execute("ls /documents"))
            tree_result = _payload(executor.execute("tree /documents -L 1"))

            self.assertEqual(calls[0], ["/documents", "-L", "1"])
            self.assertEqual(ls_result, tree_result)
            self.assertTrue(ls_result["success"])
            self.assertEqual(
                list(ls_result["data"]),
                ["tree", "total_folders", "depth", "truncated"],
            )
            self.assertEqual(ls_result["data"]["depth"], 1)

    def test_browse_summary_scoped_and_paginated(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            refs = {
                "doc_direct": _register_file(filesystem, root, "doc_direct", "/documents"),
                "doc_deep": _register_file(filesystem, root, "doc_deep", "/documents/reports"),
                "doc_other": _register_file(filesystem, root, "doc_other", "/other"),
            }
            backend = BrowseBackend(
                ["doc_other", "doc_deep", "doc_direct"],
                file_refs_by_document_id=refs,
            )
            filesystem.semantic_retrieval_backend = backend
            executor = PIFSCommandExecutor(filesystem)

            missing_query = _payload(executor.execute("browse /documents"))
            self.assertFalse(missing_query["success"])
            self.assertIn("requires a query", missing_query["error"]["message"])

            removed_space = _payload(
                executor.execute('browse /documents "alpha" --space entity')
            )
            self.assertFalse(removed_space["success"])
            self.assertIn("--space is removed", removed_space["error"]["message"])

            direct = _payload(executor.execute('browse /documents "alpha"'))
            self.assertTrue(direct["success"])
            self.assertEqual(
                [doc["document_id"] for doc in direct["data"]["documents"]],
                ["doc_direct"],
            )
            self.assertEqual(
                direct["data"]["scope"],
                {
                    "folder": "/documents",
                    "recursive": False,
                    "query": "alpha",
                    "where": None,
                    "retrieval": "summary",
                },
            )
            self.assertEqual(backend.calls[-1][0], "summary")

            recursive = _payload(executor.execute('browse -R /documents "alpha"'))
            self.assertEqual(
                [doc["document_id"] for doc in recursive["data"]["documents"]],
                ["doc_deep", "doc_direct"],
            )
            self.assertTrue(recursive["data"]["scope"]["recursive"])

    def test_stat_and_grep_are_single_document_only(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            _register_file(
                filesystem,
                root,
                "doc_a",
                "/documents",
                title="report.txt",
                text="alpha evidence\nbeta evidence",
            )
            executor = PIFSCommandExecutor(filesystem)

            stat = _payload(executor.execute("stat /documents/report.txt"))
            self.assertTrue(stat["success"])
            self.assertEqual(stat["data"]["document"]["document_id"], "doc_a")
            self.assertTrue(stat["data"]["document"]["file_ref"].startswith("file_"))

            grep = _payload(executor.execute("grep alpha /documents/report.txt"))
            self.assertTrue(grep["success"])
            self.assertEqual(grep["data"]["document"]["document_id"], "doc_a")
            self.assertEqual(
                grep["data"]["matches"],
                [{"line": 1, "text": "alpha evidence"}],
            )

            folder_grep = _payload(executor.execute("grep alpha /documents"))
            self.assertFalse(folder_grep["success"])
            self.assertIn("not a folder", folder_grep["error"]["message"])

    def test_cat_structure_and_page_use_aligned_json_shapes(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

        class FakePageClient:
            documents = {"pi_doc": {}}

            def get_document_structure(self, doc_id):
                assert doc_id == "pi_doc"
                return json.dumps(
                    [{"title": "Risk", "node_id": "0001", "text": "hidden", "nodes": []}]
                )

            def get_page_content(self, doc_id, pages):
                assert doc_id == "pi_doc"
                assert pages == "1-2"
                return json.dumps(
                    [
                        {"page": 1, "content": "page one evidence"},
                        {"page": 2, "content": "page two evidence"},
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "report"
            source.write_text("cached text", encoding="utf-8")
            filesystem = PageIndexFileSystem(workspace=root / "workspace")
            filesystem._pageindex_client = lambda: FakePageClient()
            file_ref = filesystem.register_file(
                storage_uri=source.as_uri(),
                folder_path="/documents",
                external_id="doc_pdf",
                title="report",
                content="cached text",
            )
            filesystem.store.update_pageindex_pointer(
                file_ref,
                pageindex_doc_id="pi_doc",
                pageindex_tree_status="built",
            )
            executor = PIFSCommandExecutor(filesystem)

            structure = _payload(executor.execute("cat doc_pdf --structure"))
            self.assertTrue(structure["success"])
            self.assertEqual(structure["data"]["document"]["document_id"], "doc_pdf")
            self.assertEqual(
                structure["data"]["structure"],
                [{"title": "Risk", "node_id": "0001", "nodes": []}],
            )
            self.assertEqual(structure["data"]["pagination"], {"available": True})

            page = _payload(executor.execute("cat doc_pdf --page 1-2"))
            self.assertTrue(page["success"])
            self.assertEqual(page["data"]["requested_pages"], "1-2")
            self.assertEqual(
                [item["page"] for item in page["data"]["returned_pages"]],
                [1, 2],
            )
            self.assertEqual(
                page["data"]["content"],
                {
                    "text": "page one evidence\n\npage two evidence",
                    "available": True,
                },
            )

    def test_agent_policy_mentions_only_aligned_surface(self):
        from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
        from pageindex.filesystem.agent import build_pifs_agent_instructions

        with tempfile.TemporaryDirectory() as tmp:
            filesystem = PageIndexFileSystem(workspace=Path(tmp) / "workspace")
            executor = PIFSCommandExecutor(filesystem)

            instructions = build_pifs_agent_instructions(filesystem, executor=executor)

            for expected in ("tree", "browse", "stat", "cat", "grep"):
                self.assertIn(expected, instructions)
            for removed in (
                "find --",
                "grep -R",
                "stat --schema",
                "stat --field",
                "cat <target> --all",
                "cat <target> --range",
            ):
                self.assertNotIn(removed, instructions)
            self.assertIn(
                "run cat <target> --structure before the first cat <target> --page",
                instructions,
            )

    def test_projection_surface_is_summary(self):
        from pageindex.filesystem.core import (
            DEFAULT_METADATA_GENERATION_FIELDS,
            SEMANTIC_PROJECTION_INDEX_NAMES,
            SEMANTIC_RETRIEVAL_CHANNELS,
        )
        from pageindex.filesystem.metadata_generation import GENERATED_METADATA_FIELDS
        semantic_projection_source = Path("pageindex/filesystem/semantic_projection.py").read_text(
            encoding="utf-8"
        )

        self.assertEqual(SEMANTIC_RETRIEVAL_CHANNELS, ("summary",))
        self.assertEqual(SEMANTIC_PROJECTION_INDEX_NAMES, {"summary": "summary"})
        self.assertNotIn("entity", DEFAULT_METADATA_GENERATION_FIELDS)
        self.assertNotIn("relation", DEFAULT_METADATA_GENERATION_FIELDS)
        self.assertEqual(GENERATED_METADATA_FIELDS, ("summary", "doc_type", "domain", "topic"))
        self.assertIn('SUMMARY_INDEX_NAME = "summary"', semantic_projection_source)
        self.assertNotIn("entity_vectors", semantic_projection_source)
        self.assertNotIn("relation_vectors", semantic_projection_source)


if __name__ == "__main__":
    unittest.main()
