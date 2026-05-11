class PageIndexError(Exception):
    """Base exception for all PageIndex SDK errors."""
    pass


class CollectionNotFoundError(PageIndexError):
    """Collection does not exist."""
    pass


class DocumentNotFoundError(PageIndexError):
    """Document ID not found."""
    pass


class IndexingError(PageIndexError):
    """Indexing pipeline failure."""
    pass


class PageIndexAPIError(PageIndexError):
    """PageIndex cloud API returned an error.

    Kept for compatibility with the pageindex 0.2.x cloud SDK.
    """
    pass


class CloudAPIError(PageIndexAPIError):
    """Cloud API returned error."""
    pass


class FileTypeError(PageIndexError):
    """Unsupported file type."""
    pass
