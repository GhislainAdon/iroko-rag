# tests/sdk/test_collection.py
import pytest
from unittest.mock import MagicMock
from pageindex.collection import Collection


@pytest.fixture
def col():
    backend = MagicMock()
    backend.list_documents.return_value = [
        {"doc_id": "d1", "doc_name": "paper.pdf", "doc_type": "pdf"}
    ]
    backend.get_document.return_value = {"doc_id": "d1", "doc_name": "paper.pdf"}
    backend.add_document.return_value = "d1"
    return Collection(name="papers", backend=backend)


def test_add(col):
    doc_id = col.add("paper.pdf")
    assert doc_id == "d1"
    col._backend.add_document.assert_called_once_with("papers", "paper.pdf")


def test_list_documents(col):
    docs = col.list_documents()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == "d1"


def test_get_document(col):
    doc = col.get_document("d1")
    assert doc["doc_name"] == "paper.pdf"


def test_delete_document(col):
    col.delete_document("d1")
    col._backend.delete_document.assert_called_once_with("papers", "d1")


def test_name_property(col):
    assert col.name == "papers"


def test_query_without_doc_ids_warns_when_multidoc(col, monkeypatch):
    monkeypatch.delenv("PAGEINDEX_EXPERIMENTAL_MULTIDOC", raising=False)
    col._backend.list_documents.return_value = [
        {"doc_id": "d1", "doc_name": "a.pdf", "doc_type": "pdf"},
        {"doc_id": "d2", "doc_name": "b.pdf", "doc_type": "pdf"},
    ]
    col._backend.query.return_value = "answer"
    with pytest.warns(UserWarning, match="experimental"):
        result = col.query("what?")
    assert result == "answer"


def test_query_without_doc_ids_no_warning_when_single_doc(col, monkeypatch, recwarn):
    monkeypatch.delenv("PAGEINDEX_EXPERIMENTAL_MULTIDOC", raising=False)
    col._backend.query.return_value = "answer"
    col.query("what?")
    assert not any(issubclass(w.category, UserWarning) for w in recwarn)


def test_query_empty_collection_raises(col, monkeypatch):
    monkeypatch.delenv("PAGEINDEX_EXPERIMENTAL_MULTIDOC", raising=False)
    col._backend.list_documents.return_value = []
    with pytest.raises(ValueError, match="empty"):
        col.query("what?")


def test_query_with_doc_ids_no_warning(col, recwarn):
    col._backend.query.return_value = "answer"
    col.query("what?", doc_ids=["d1"])
    assert not any(issubclass(w.category, UserWarning) for w in recwarn)


def test_query_env_var_silences_warning(col, monkeypatch, recwarn):
    monkeypatch.setenv("PAGEINDEX_EXPERIMENTAL_MULTIDOC", "1")
    col._backend.list_documents.return_value = [
        {"doc_id": "d1", "doc_name": "a.pdf", "doc_type": "pdf"},
        {"doc_id": "d2", "doc_name": "b.pdf", "doc_type": "pdf"},
    ]
    col._backend.query.return_value = "answer"
    col.query("what?")
    assert not any(issubclass(w.category, UserWarning) for w in recwarn)
