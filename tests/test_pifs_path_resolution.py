import pytest

from tests.pifs_markdown_fixture import register_markdown


def test_root_virtual_file_path_resolves_without_double_slash(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    file_ref = register_markdown(
        filesystem, tmp_path, "doc_root_title", "/", title="Root Title.md", text="root content"
    )

    assert filesystem.store.resolve_file_ref("/Root Title.md") == file_ref


def test_nested_virtual_file_path_resolves_by_folder_and_title(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    first_ref = register_markdown(filesystem, tmp_path, "doc_first", "/a", title="First.md", text="first content")
    second_ref = register_markdown(filesystem, tmp_path, "doc_second", "/a/b", title="file.md", text="second content")

    assert filesystem.store.resolve_file_ref("/a/b/file.md") == second_ref

    assert first_ref != second_ref


def test_unknown_virtual_file_target_raises_clear_error(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    first_ref = register_markdown(filesystem, tmp_path, "doc_first", "/first", title="First.md", text="first content")
    second_ref = register_markdown(filesystem, tmp_path, "doc_second", "/second", title="Second.md", text="second content")

    with pytest.raises(KeyError, match="Unknown file target"):
        filesystem.store.resolve_file_ref("/shared/missing.txt")

    assert first_ref != second_ref
