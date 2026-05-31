import json
from types import SimpleNamespace

import pytest


def test_filesystem_lazy_exports_remain_public():
    import pageindex.filesystem as filesystem
    from pageindex.filesystem import (
        HybridProjectionSearchBackend,
        RebuildableSemanticIndex,
        SemanticIndexRecord,
        SemanticSearchResult,
        SQLiteVecSemanticIndex,
        SummaryProjectionIndexer,
    )

    for name in (
        "HybridProjectionSearchBackend",
        "RebuildableSemanticIndex",
        "SemanticIndexRecord",
        "SemanticSearchResult",
        "SQLiteVecSemanticIndex",
        "SummaryProjectionIndexer",
    ):
        assert name in filesystem.__all__
        assert name in dir(filesystem)

    assert HybridProjectionSearchBackend.__name__ == "HybridProjectionSearchBackend"
    assert RebuildableSemanticIndex.__name__ == "RebuildableSemanticIndex"
    assert SemanticIndexRecord.__name__ == "SemanticIndexRecord"
    assert SemanticSearchResult.__name__ == "SemanticSearchResult"
    assert SQLiteVecSemanticIndex.__name__ == "SQLiteVecSemanticIndex"
    assert SummaryProjectionIndexer.__name__ == "SummaryProjectionIndexer"


class SummaryBackend:
    def __init__(self, document_id):
        self.document_id = document_id
        self.calls = []

    def available_channels(self):
        return ("summary",)

    def search_channel(self, channel, query, *, limit=10, filters=None):
        self.calls.append((channel, query, filters))
        return [
            SimpleNamespace(
                document_id=self.document_id,
                snippet=f"summary candidate: {query}",
            )
        ]


class ChannelBackend:
    def __init__(self, document_id, channels=("summary", "entity", "relation")):
        self.document_id = document_id
        self.channels = channels
        self.calls = []

    def available_channels(self):
        return self.channels

    def search_channel(self, channel, query, *, limit=10, filters=None):
        self.calls.append((channel, query, limit, filters))
        return [
            SimpleNamespace(
                document_id=self.document_id,
                snippet=f"{channel} candidate: {query}",
            )
        ]


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
            raw_file_refs = filters.get("file_ref") or filters.get("file_refs") or []
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
                sources=[{"channel": channel, "rank": rank, "distance": rank / 10}],
            )
            for rank, document_id in enumerate(document_ids[:limit], 1)
        ]


def _register_browse_file(
    filesystem,
    external_id,
    folder_path,
    *,
    department="ops",
    summary=None,
):
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class SummaryGenerator:
        def generate(self, document, *, fields):
            values = {
                "summary": summary
                if summary is not None
                else f"summary for {document.external_id}",
                "doc_type": "memo",
                "domain": "finance",
                "topic": "risk",
            }
            return MetadataGenerationResult(
                values={field: values[field] for field in fields if field in values}
            )

    filesystem.metadata_generator = SummaryGenerator()
    return filesystem.register_file(
        storage_uri=f"file:///tmp/{external_id}.txt",
        source_path=f"documents/{external_id}.txt",
        folder_path=folder_path,
        external_id=external_id,
        title=f"{external_id}.txt",
        content=f"{external_id} discusses vector databases and retrieval.",
        metadata={"department": department},
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )


def test_browse_is_agent_visible_semantic_command(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    filesystem.semantic_retrieval_backend = ChannelBackend("dsid_report")
    executor = PIFSCommandExecutor(filesystem)

    allowed = executor.allowed_commands()
    surface = executor.describe_available_command_surfaces()

    assert "browse" in allowed
    assert 'browse [-R] <folder> "<query>"' in surface
    assert not {
        "search-summary",
        "search-entity",
        "search-relation",
        "semantic-grep",
    } & allowed
    for old_command in (
        "search-summary",
        "search-entity",
        "search-relation",
        "semantic-grep",
        "find --name: entity semantic",
        "find --relation: relation semantic",
    ):
        assert old_command not in surface
    assert executor.command_capabilities()["retrieval"]["semantic"]["commands"] == ["browse"]


def test_browse_requires_positional_query_and_rejects_removed_options(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    _register_browse_file(filesystem, "doc_direct", "/documents")
    filesystem.semantic_retrieval_backend = BrowseBackend(["doc_direct"])
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    with pytest.raises(PIFSCommandError, match="browse requires a query"):
        executor.execute("browse /documents")
    with pytest.raises(PIFSCommandError, match="--query"):
        executor.execute('browse /documents "vector database" --query "other"')
    with pytest.raises(PIFSCommandError, match="--limit"):
        executor.execute('browse /documents "vector database" --limit 10')
    with pytest.raises(PIFSCommandError, match="--offset"):
        executor.execute('browse /documents "vector database" --offset 10')
    with pytest.raises(PIFSCommandError, match="browse accepts a folder and one quoted query"):
        executor.execute("browse /documents vector database")


def test_browse_validates_space_availability_and_page(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    _register_browse_file(filesystem, "doc_direct", "/documents")
    filesystem.semantic_retrieval_backend = BrowseBackend(["doc_direct"], channels=("summary",))
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    with pytest.raises(PIFSCommandError, match="Unsupported browse --space: hybrid"):
        executor.execute('browse /documents "vector database" --space hybrid')
    with pytest.raises(PIFSCommandError, match="available spaces: summary"):
        executor.execute('browse /documents "vector database" --space entity')
    with pytest.raises(PIFSCommandError, match="browse --page must be at least 1"):
        executor.execute('browse /documents "vector database" --page 0')


def test_browse_default_summary_does_not_fallback_to_other_spaces(tmp_path):
    import json

    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    _register_browse_file(filesystem, "doc_direct", "/documents")
    backend = BrowseBackend(["doc_direct"], channels=("entity",))
    filesystem.semantic_retrieval_backend = backend
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    with pytest.raises(PIFSCommandError, match="available spaces: entity"):
        executor.execute('browse /documents "vector database"')
    assert backend.calls == []

    result = json.loads(
        executor.execute('browse /documents "vector database" --space entity')
    )["data"]
    assert [item["document_id"] for item in result["data"]] == ["doc_direct"]
    assert backend.calls[-1][0] == "entity"


def test_browse_non_recursive_searches_only_direct_files_and_recursive_is_global(tmp_path):
    import json

    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    _register_browse_file(filesystem, "doc_direct", "/documents")
    _register_browse_file(filesystem, "doc_deep", "/documents/reports")
    backend = BrowseBackend(["doc_deep", "doc_direct"])
    filesystem.semantic_retrieval_backend = backend
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    direct = json.loads(executor.execute('browse /documents "vector database"'))["data"]
    assert [item["document_id"] for item in direct["data"]] == ["doc_direct"]
    assert direct["recursive"] is False
    assert direct["space"] == "summary"
    assert direct["page"] == 1
    assert direct["page_size"] == 10
    assert backend.calls[-1][0] == "summary"

    recursive = json.loads(executor.execute('browse -R /documents "vector database"'))["data"]
    assert [item["document_id"] for item in recursive["data"]] == [
        "doc_deep",
        "doc_direct",
    ]
    assert [item["rank"] for item in recursive["data"]] == [1, 2]
    assert recursive["recursive"] is True


def test_browse_supports_fixed_size_one_based_pagination_and_metadata_filter(tmp_path):
    import json

    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    document_ids = []
    for index in range(12):
        external_id = f"doc_{index:02d}"
        document_ids.append(external_id)
        department = "finance" if index == 10 else "ops"
        _register_browse_file(filesystem, external_id, "/documents", department=department)
    filesystem.semantic_retrieval_backend = BrowseBackend(document_ids)
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    first_page = json.loads(executor.execute('browse /documents "vector database"'))["data"]
    assert len(first_page["data"]) == 10
    assert first_page["has_more"] is True
    assert first_page["data"][0]["rank"] == 1

    second_page = json.loads(
        executor.execute('browse /documents "vector database" --page 2')
    )["data"]
    assert [item["document_id"] for item in second_page["data"]] == ["doc_10", "doc_11"]
    assert [item["rank"] for item in second_page["data"]] == [11, 12]
    assert second_page["has_more"] is False

    filtered = json.loads(
        executor.execute(
            'browse /documents "vector database" --where \'{"department":"finance"}\''
        )
    )["data"]
    assert [item["document_id"] for item in filtered["data"]] == ["doc_10"]
    assert filtered["data"][0]["summary"] == "summary for doc_10"


def test_browse_scopes_semantic_search_before_candidate_limit(tmp_path):
    import json

    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    file_refs_by_document_id = {}
    candidate_ids = []
    for index in range(150):
        external_id = f"off_scope_{index:02d}"
        candidate_ids.append(external_id)
        file_refs_by_document_id[external_id] = _register_browse_file(
            filesystem,
            external_id,
            "/other",
        )
    file_refs_by_document_id["doc_deep"] = _register_browse_file(
        filesystem,
        "doc_deep",
        "/documents/reports",
    )
    file_refs_by_document_id["doc_direct"] = _register_browse_file(
        filesystem,
        "doc_direct",
        "/documents",
    )
    backend = BrowseBackend(
        [*candidate_ids, "doc_deep", "doc_direct"],
        file_refs_by_document_id=file_refs_by_document_id,
    )
    filesystem.semantic_retrieval_backend = backend
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    direct = json.loads(executor.execute('browse /documents "vector database"'))["data"]
    assert [item["document_id"] for item in direct["data"]] == ["doc_direct"]

    recursive = json.loads(executor.execute('browse -R /documents "vector database"'))["data"]
    assert [item["document_id"] for item in recursive["data"]] == [
        "doc_deep",
        "doc_direct",
    ]

def test_browse_shell_output_uses_fixed_blocks_with_pagination_command(tmp_path):
    import re

    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    document_ids = []
    for index in range(12):
        external_id = f"doc_{index:02d}"
        document_ids.append(external_id)
        _register_browse_file(
            filesystem,
            external_id,
            "/documents",
            department="finance",
            summary=(
                "first line\nsecond\tline   with spaces"
                if index == 0
                else f"summary for {external_id}"
            ),
        )
    filesystem.semantic_retrieval_backend = BrowseBackend(
        document_ids,
        channels=("summary", "entity"),
    )
    executor = PIFSCommandExecutor(filesystem)

    rendered = executor.execute(
        'browse -R /documents "vector database" --space entity '
        '--where \'{"department":"finance"}\''
    )
    lines = rendered.splitlines()

    assert lines[:6] == [
        "# page=1 page_size=10 has_more=true",
        "rank: 1",
        "similarity: 0.91",
        "path: /documents/doc_00.txt",
        "summary: first line second line with spaces",
        "",
    ]
    assert lines[6:10] == [
        "rank: 2",
        "similarity: 0.83",
        "path: /documents/doc_01.txt",
        "summary: summary for doc_01",
    ]
    similarity_lines = [line for line in lines if line.startswith("similarity: ")]
    assert len(similarity_lines) == 10
    assert all(re.fullmatch(r"similarity: [01]\.\d{2}", line) for line in similarity_lines)
    assert all(0.0 <= float(line.removeprefix("similarity: ")) <= 1.0 for line in similarity_lines)
    assert lines[-1] == (
        "# next: browse -R /documents 'vector database' --space entity "
        '--where \'{"department":"finance"}\' --page 2'
    )
    assert "mode:" not in rendered
    assert "data:" not in rendered
    assert "score:" not in rendered


def test_browse_shell_path_falls_back_to_unique_locator_when_source_collides(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class SummaryGenerator:
        def generate(self, document, *, fields):
            return MetadataGenerationResult(
                values={"summary": f"summary for {document.external_id}"}
            )

    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=SummaryGenerator(),
    )
    first_ref = filesystem.register_file(
        storage_uri="file:///tmp/first.json",
        source_path="shared/source.json",
        folder_path="/documents",
        external_id="dsid_first",
        title="First",
        content="first content",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    filesystem.register_file(
        storage_uri="file:///tmp/second.json",
        source_path="shared/source.json",
        folder_path="/documents",
        external_id="dsid_second",
        title="Second",
        content="second content",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    filesystem.semantic_retrieval_backend = BrowseBackend(["dsid_first"])
    executor = PIFSCommandExecutor(filesystem)

    rendered = executor.execute('browse /documents "first"')

    assert "path: dsid_first" in rendered
    assert "path: /shared/source.json" not in rendered
    assert filesystem.store.resolve_file_ref("dsid_first") == first_ref
    with pytest.raises(KeyError, match="Ambiguous file target"):
        filesystem.store.resolve_file_ref("/shared/source.json")


def test_browse_scope_keeps_ordinary_folders_out_of_source_type_filters(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class SummaryGenerator:
        def generate(self, document, *, fields):
            return MetadataGenerationResult(
                values={"summary": "Federal Reserve annual report summary"}
            )

    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=SummaryGenerator(),
    )
    file_ref = filesystem.register_file(
        storage_uri="file:///tmp/report.pdf",
        source_path="examples/documents/report.pdf",
        folder_path="/documents",
        external_id="dsid_report",
        title="report.pdf",
        metadata={"source_type": "examples-documents"},
        content="Federal Reserve supervision and regulation annual report.",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    backend = SummaryBackend("dsid_report")
    filesystem.semantic_retrieval_backend = backend
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    result = json.loads(
        executor.execute('browse /documents "Federal Reserve annual report"')
    )

    assert "source_type" not in backend.calls[0][2]
    assert "source_path" not in backend.calls[0][2]
    assert result["data"]["data"][0]["path"] == "/examples/documents/report.pdf"
    assert result["data"]["data"][0]["summary"] == "Federal Reserve annual report summary"
    assert filesystem.store.resolve_file_ref(result["data"]["data"][0]["path"]) == file_ref


def test_browse_path_is_unique_source_target_when_titles_collide(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class SummaryGenerator:
        def generate(self, document, *, fields):
            return MetadataGenerationResult(
                values={"summary": f"summary for {document.external_id}"}
            )

    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=SummaryGenerator(),
    )
    first_ref = filesystem.register_file(
        storage_uri="file:///tmp/first.json",
        source_path="slack/dsid_first.json",
        folder_path="/documents",
        external_id="dsid_first",
        title="announcements",
        content="first announcement mentions H200 reservations.",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    filesystem.register_file(
        storage_uri="file:///tmp/second.json",
        source_path="slack/dsid_second.json",
        folder_path="/documents",
        external_id="dsid_second",
        title="announcements",
        content="second announcement mentions unrelated maintenance.",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    filesystem.semantic_retrieval_backend = SummaryBackend("dsid_first")
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    result = json.loads(executor.execute('browse /documents "H200 reservations"'))

    assert result["data"]["data"][0]["path"] == "/slack/dsid_first.json"
    assert filesystem.store.resolve_file_ref(result["data"]["data"][0]["path"]) == first_ref
    with pytest.raises(KeyError, match="Ambiguous file target"):
        filesystem.store.resolve_file_ref("/documents/announcements")


def test_browse_path_falls_back_when_source_target_is_ambiguous(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class SummaryGenerator:
        def generate(self, document, *, fields):
            return MetadataGenerationResult(
                values={"summary": f"summary for {document.external_id}"}
            )

    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=SummaryGenerator(),
    )
    first_ref = filesystem.register_file(
        storage_uri="file:///tmp/first.json",
        source_path="shared/source.json",
        folder_path="/documents",
        external_id="dsid_first",
        title="First",
        content="first content",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    filesystem.register_file(
        storage_uri="file:///tmp/second.json",
        source_path="shared/source.json",
        folder_path="/documents",
        external_id="dsid_second",
        title="Second",
        content="second content",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    filesystem.semantic_retrieval_backend = SummaryBackend("dsid_first")
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    result = json.loads(executor.execute('browse /documents "first"'))

    assert result["data"]["data"][0]["path"] == "dsid_first"
    assert filesystem.store.resolve_file_ref(result["data"]["data"][0]["path"]) == first_ref


def test_old_semantic_commands_are_unsupported_even_when_indexes_exist(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class MetadataGenerator:
        def generate(self, document, *, fields):
            values = {
                "summary": "Risk and compliance summary",
                "entity": "Federal Reserve; Disney",
                "relation": "Federal Reserve affects Disney valuation",
            }
            return MetadataGenerationResult(values={field: values[field] for field in fields})

    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=MetadataGenerator(),
    )
    filesystem.register_file(
        storage_uri="file:///tmp/market-note.pdf",
        source_path="examples/documents/market-note.pdf",
        folder_path="/documents",
        external_id="dsid_market_note",
        title="market-note.pdf",
        content="Federal Reserve policy affects Disney valuation.",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
                "entity": True,
                "relation": True,
            }
        },
    )
    filesystem.semantic_retrieval_backend = ChannelBackend("dsid_market_note")
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    for command in (
        'search-summary "Federal Reserve" /documents',
        'search-entity "Federal Reserve" /documents',
        'search-relation "Disney valuation" /documents',
        'semantic-grep -R "Federal Reserve" /documents',
    ):
        with pytest.raises(PIFSCommandError, match="Unsupported command"):
            executor.execute(command)

    entity = json.loads(
        executor.execute('browse /documents "Federal Reserve" --space entity')
    )
    assert entity["data"]["data"][0]["summary"] == "Risk and compliance summary"
    assert entity["data"]["data"][0]["path"] == "/examples/documents/market-note.pdf"

    relation = json.loads(
        executor.execute('browse /documents "Disney valuation" --space relation')
    )
    assert relation["data"]["data"][0]["summary"] == "Risk and compliance summary"
    assert relation["data"]["data"][0]["path"] == "/examples/documents/market-note.pdf"


def test_find_name_is_lexical_and_find_relation_is_not_semantic_alias(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    filesystem.register_file(
        storage_uri="file:///tmp/report.pdf",
        source_path="examples/documents/report.pdf",
        folder_path="/documents",
        external_id="dsid_report",
        title="Annual report",
        content="Federal Reserve supervision and regulation annual report.",
    )
    backend = ChannelBackend("dsid_report", channels=("entity", "relation"))
    filesystem.semantic_retrieval_backend = backend
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    result = json.loads(executor.execute("find /documents --name Reserve"))["data"]

    assert result[0]["external_id"] == "dsid_report"
    assert backend.calls == []

    with pytest.raises(PIFSCommandError, match="find --relation is not supported"):
        executor.execute('find /documents --relation "Reserve regulates report"')


def test_broad_recursive_grep_suggests_browse_not_removed_semantic_commands(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    _register_browse_file(filesystem, "dsid_report", "/documents")
    filesystem.semantic_retrieval_backend = ChannelBackend("dsid_report")
    filesystem.store.folder_subtree_thresholds = lambda *args, **kwargs: {
        "depth_limit": 2,
        "file_limit": 10,
        "folder_depth_exceeds_limit": True,
        "file_count_exceeds_limit": False,
        "sampled_file_count": 11,
        "sample_deep_folder_path": "/documents/deep",
    }
    executor = PIFSCommandExecutor(filesystem)

    rendered = executor.execute('grep -R "Federal Reserve" /documents')

    assert "# suggested: browse -R /documents 'Federal Reserve'" in rendered
    assert "search-summary" not in rendered
    assert "search-entity" not in rendered
    assert "search-relation" not in rendered
    assert "semantic-grep" not in rendered


def test_semantic_search_scope_filters_explicit_source_type_facets():
    from pageindex.filesystem import PageIndexFileSystem

    assert PageIndexFileSystem._semantic_filters_for_scope(
        {"folder_path": "/source_type=google-drive"}
    ) == {"source_type": "google_drive"}
    assert PageIndexFileSystem._semantic_filters_for_scope(
        {"folder_path": "/semantic/source_type=google-drive"}
    ) == {"source_type": "google_drive"}
    assert PageIndexFileSystem._semantic_filters_for_scope(
        {"folder_path": "/documents"}
    ) == {}


def test_grep_source_file_requires_terms_on_same_line(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    source_dir = tmp_path / "source" / "documents"
    source_dir.mkdir(parents=True)
    source = source_dir / "split.json"
    source.write_text(
        '{\n  "first": "alpha evidence lives here",\n'
        '  "second": "omega evidence lives there"\n}\n',
        encoding="utf-8",
    )
    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    filesystem.register_file(
        storage_uri=str(source),
        source_path="documents/split.json",
        folder_path="/documents",
        external_id="doc_split_terms",
        title="Split source terms",
        content="registered artifact without the searched tokens",
    )
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    result = json.loads(executor.execute('grep -R "alpha omega" /documents'))

    assert result["data"]["mode"] == "files"
    assert result["data"]["data"] == []

    matched = json.loads(executor.execute('grep -R "alpha evidence" /documents'))

    assert matched["data"]["data"][0]["external_id"] == "doc_split_terms"
    assert matched["data"]["data"][0]["line"] == 2
    assert "alpha evidence" in matched["data"]["data"][0]["text"]


def test_existing_summary_projection_index_configures_retrieval_backend(tmp_path, monkeypatch):
    from pageindex.filesystem import PageIndexFileSystem
    from pageindex.filesystem.semantic_index import SemanticIndexRecord, SQLiteVecSemanticIndex

    workspace = tmp_path / "workspace"
    index_dir = workspace / "artifacts" / "projection_indexes"
    summary_index = SQLiteVecSemanticIndex(index_dir / "summary_only_vector.sqlite")
    summary_index.reset(
        dimension=3,
        metadata={
            "channel": "summary",
            "embedding_provider": "openai",
            "embedding_model": "test-embedding",
            "embedding_dimensions": 3,
        },
    )
    summary_index.upsert_many(
        [
            SemanticIndexRecord(
                file_ref="file_a",
                external_id="doc_a",
                source_type="documents",
                source_path="documents/a.pdf",
                title="A",
                text="summary",
                vector=[1.0, 0.0, 0.0],
            )
        ]
    )
    filesystem = PageIndexFileSystem(workspace)
    calls = []

    def fake_configure(index_dir_arg, **kwargs):
        calls.append((index_dir_arg, kwargs))
        filesystem.semantic_retrieval_backend = SummaryBackend("doc_a")
        return filesystem.semantic_retrieval_backend

    monkeypatch.setattr(
        filesystem,
        "configure_hybrid_projection_retrieval",
        fake_configure,
    )

    assert filesystem.configure_existing_projection_retrieval() is True
    assert calls == [
        (
            filesystem.summary_projection_index_dir,
            {
                "embedding_provider": "openai",
                "embedding_model": "test-embedding",
                "embedding_dimensions": 3,
                "embedding_timeout": 60,
            },
        )
    ]
    assert filesystem.semantic_retrieval_channels() == ("summary",)


def test_default_semantic_search_uses_summary_projection_when_only_summary_available(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem
    from pageindex.filesystem.hybrid_projection import HybridProjectionSearchBackend
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    class FixedEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    class SummaryGenerator:
        def generate(self, document, *, fields):
            return MetadataGenerationResult(
                values={"summary": "vendor renewal risk matrix"}
            )

    source = tmp_path / "source.txt"
    source.write_text("ordinary fixture body", encoding="utf-8")
    index_dir = tmp_path / "workspace" / "artifacts" / "projection_indexes"
    indexer = SummaryProjectionIndexer(
        index_dir,
        embedder=FixedEmbedder(),
        embedding_provider="test",
        embedding_model="fake",
        embedding_dimensions=3,
    )
    backend = HybridProjectionSearchBackend(
        index_dir,
        embedder=FixedEmbedder(),
        embedding_provider="test",
        embedding_model="fake",
        embedding_dimensions=3,
    )
    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=SummaryGenerator(),
        summary_projection_indexer=indexer,
        semantic_retrieval_backend=backend,
    )
    filesystem.register_file(
        storage_uri=source.as_uri(),
        source_path="docs/source.txt",
        folder_path="/documents",
        external_id="doc_summary_only",
        title="Operations note",
        content=source.read_text(encoding="utf-8"),
        metadata={"department": "ops"},
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )

    assert filesystem.search("purchase order exposure", semantic=False) == []

    results = filesystem.search("purchase order exposure", semantic=True)

    assert [result.external_id for result in results] == ["doc_summary_only"]
    assert results[0].snippet == "summary_vector rank=1"
