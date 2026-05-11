# pageindex/__init__.py
# Upstream exports (backward compatibility)
from .page_index import *
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content

# SDK exports
from .client import PageIndexClient, LocalClient, CloudClient
from .config import IndexConfig
from .collection import Collection
from .parser.protocol import ContentNode, ParsedDocument, DocumentParser
from .storage.protocol import StorageEngine
from .events import QueryEvent
from .errors import (
    PageIndexError,
    PageIndexAPIError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    IndexingError,
    CloudAPIError,
    FileTypeError,
)

__all__ = [
    "PageIndexClient",
    "LocalClient",
    "CloudClient",
    "IndexConfig",
    "Collection",
    "ContentNode",
    "ParsedDocument",
    "DocumentParser",
    "StorageEngine",
    "QueryEvent",
    "PageIndexError",
    "PageIndexAPIError",
    "CollectionNotFoundError",
    "DocumentNotFoundError",
    "IndexingError",
    "CloudAPIError",
    "FileTypeError",
]
