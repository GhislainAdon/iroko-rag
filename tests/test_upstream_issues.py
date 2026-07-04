"""Regression tests for confirmed upstream PageIndex issues fixed in iroko-rag.

Each test reproduces the exact failure described in the upstream issue and
runs without any LLM call.
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.utils import extract_json, get_leaf_nodes, list_to_tree
from pageindex.retrieve import get_page_content
from pageindex.page_index_md import extract_node_text_content, extract_nodes_from_markdown


class TestIssue330GetLeafNodes:
    """get_leaf_nodes raised KeyError on leaf nodes because clean_node()
    deletes the 'nodes' key instead of leaving an empty list."""

    def test_leaf_nodes_after_list_to_tree(self):
        flat = [
            {'structure': '1', 'title': 'Chapter 1', 'start_index': 1, 'end_index': 5},
            {'structure': '2', 'title': 'Chapter 2', 'start_index': 6, 'end_index': 10},
        ]
        tree = list_to_tree(flat)
        leaves = get_leaf_nodes(tree)  # crashed with KeyError: 'nodes'
        assert len(leaves) == 2
        assert {l['title'] for l in leaves} == {'Chapter 1', 'Chapter 2'}


class TestIssue326ExtractJson:
    """extract_json returned {} when the model wraps JSON in prose, and
    callers doing direct key access then crashed with KeyError."""

    def test_json_wrapped_in_prose(self):
        response = 'Sure! Here is the answer:\n{"toc_detected": "yes"}\nHope this helps.'
        assert extract_json(response) == {"toc_detected": "yes"}

    def test_json_array_wrapped_in_prose(self):
        response = 'The structure is: [{"title": "A"}, {"title": "B"}] as requested.'
        assert extract_json(response) == [{"title": "A"}, {"title": "B"}]

    def test_strict_json_still_works(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json_still_works(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_garbage_returns_empty_dict(self):
        assert extract_json('no json here at all') == {}

    def test_braces_inside_strings_are_ignored(self):
        response = 'Note {this} is prose. {"k": "va{lu}e"} done.'
        assert extract_json(response) == {"k": "va{lu}e"}


class TestIssue279MdPageContent:
    """get_page_content treated a comma list '5,100' on Markdown docs as
    the inclusive window [5, 100], pulling in every node in between."""

    DOCS = {
        'md': {'type': 'md', 'structure': [
            {'line_num': 5, 'text': 'L5', 'nodes': []},
            {'line_num': 10, 'text': 'L10', 'nodes': []},
            {'line_num': 50, 'text': 'L50', 'nodes': []},
            {'line_num': 100, 'text': 'L100', 'nodes': []}]},
    }

    def test_comma_list_is_exact(self):
        result = json.loads(get_page_content(self.DOCS, 'md', '5,100'))
        assert [r['page'] for r in result] == [5, 100]

    def test_range_still_collects_window(self):
        result = json.loads(get_page_content(self.DOCS, 'md', '5-50'))
        assert [r['page'] for r in result] == [5, 10, 50]

    def test_single_page(self):
        result = json.loads(get_page_content(self.DOCS, 'md', '50'))
        assert [r['page'] for r in result] == [50]


class TestIssue245MarkdownEdgeCases:
    """The markdown node extraction silently dropped content before the
    first header and produced zero nodes for headerless documents."""

    def _nodes_for(self, content, fallback_title='Document'):
        node_list, lines = extract_nodes_from_markdown(content)
        return extract_node_text_content(node_list, lines,
                                         fallback_title=fallback_title)

    def test_preamble_is_captured(self):
        content = "Intro paragraph.\nImportant context.\n\n# Chapter 1\nBody."
        nodes = self._nodes_for(content)
        assert nodes[0]['title'] == 'Preamble'
        assert 'Intro paragraph.' in nodes[0]['text']
        assert nodes[1]['title'] == 'Chapter 1'

    def test_headerless_document_yields_one_node(self):
        content = "Just plain text\nwith no headers at all."
        nodes = self._nodes_for(content, fallback_title='my-doc')
        assert len(nodes) == 1
        assert nodes[0]['title'] == 'my-doc'
        assert 'Just plain text' in nodes[0]['text']

    def test_frontmatter_alone_is_not_a_preamble(self):
        content = "---\ntitle: Doc\n---\n\n# Chapter 1\nBody."
        nodes = self._nodes_for(content)
        assert nodes[0]['title'] == 'Chapter 1'

    def test_frontmatter_plus_intro_keeps_only_intro(self):
        content = "---\ntitle: Doc\n---\nReal intro text.\n\n# Chapter 1\nBody."
        nodes = self._nodes_for(content)
        assert nodes[0]['title'] == 'Preamble'
        assert 'Real intro text.' in nodes[0]['text']
        assert 'title: Doc' not in nodes[0]['text']

    def test_no_preamble_no_extra_node(self):
        content = "# Chapter 1\nBody text."
        nodes = self._nodes_for(content)
        assert nodes[0]['title'] == 'Chapter 1'
        assert len(nodes) == 1


class TestIssue283Throttling:
    """Concurrent LLM calls are now capped by a semaphore; unthrottled
    asyncio.gather fan-outs flooded rate-limited endpoints (HTTP 429)."""

    def test_concurrency_is_capped(self):
        from pageindex import utils as u

        peak = 0
        active = 0

        async def fake_acompletion(**kwargs):
            nonlocal peak, active
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1

            class R:
                class Choice:
                    class Msg:
                        content = "ok"
                    message = Msg()
                choices = [Choice()]
            return R()

        async def run():
            tasks = [u.llm_acompletion("test-model", f"p{i}") for i in range(40)]
            return await asyncio.gather(*tasks)

        with patch.object(u.litellm, "acompletion", side_effect=fake_acompletion):
            results = asyncio.run(run())

        assert all(r == "ok" for r in results)
        assert peak <= u.MAX_CONCURRENT_LLM_CALLS
