from importlib import import_module
from typing import TYPE_CHECKING

from .commands import PIFSCommandExecutor
from .core import PageIndexFileSystem
from .types import OpenResult, SearchResult

if TYPE_CHECKING:
    from .semantic_projection import SemanticProjectionSearchBackend
    from .semantic_projection import SummaryProjectionIndexer
    from .semantic_index import (
        RebuildableSemanticIndex,
        SemanticIndexRecord,
        SemanticSearchResult,
        SQLiteVecSemanticIndex,
    )

_LAZY_EXPORTS = {
    "SemanticProjectionSearchBackend": (".semantic_projection", "SemanticProjectionSearchBackend"),
    "RebuildableSemanticIndex": (".semantic_index", "RebuildableSemanticIndex"),
    "SemanticIndexRecord": (".semantic_index", "SemanticIndexRecord"),
    "SemanticSearchResult": (".semantic_index", "SemanticSearchResult"),
    "SQLiteVecSemanticIndex": (".semantic_index", "SQLiteVecSemanticIndex"),
    "SummaryProjectionIndexer": (".semantic_projection", "SummaryProjectionIndexer"),
}

__all__ = [
    "OpenResult",
    "SemanticProjectionSearchBackend",
    "PIFSCommandExecutor",
    "PageIndexFileSystem",
    "RebuildableSemanticIndex",
    "SearchResult",
    "SemanticIndexRecord",
    "SemanticSearchResult",
    "SummaryProjectionIndexer",
    "SQLiteVecSemanticIndex",
]


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))
