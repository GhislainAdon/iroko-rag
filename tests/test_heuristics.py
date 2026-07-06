"""Tests for the plain-text structure heuristics in ingest.py."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingest import heuristic_headers, llm_structure_headers


def _apply(text):
    lines, changed = heuristic_headers(text.split('\n'))
    return '\n'.join(lines), changed


class TestHeuristicHeaders:
    def test_all_caps_line_becomes_h2(self):
        out, changed = _apply("CONTEXTE DE LA MISSION\nDu texte normal.")
        assert changed
        assert out.startswith('## CONTEXTE DE LA MISSION')

    def test_short_colon_line_becomes_h3(self):
        out, changed = _apply("Bases de données :\nOracle\nPostgreSQL")
        assert changed
        assert out.startswith('### Bases de données')

    def test_normal_sentences_untouched(self):
        text = ("Le prestataire interviendra sur les technologies "
                "et environnements suivants :\nOracle\nRedHat 7 / 8 / 9")
        out, changed = _apply(text)
        assert not changed
        assert out == text

    def test_lowercase_colon_line_untouched(self):
        out, changed = _apply("les Business Line IT du SI CASA :")
        assert not changed

    def test_short_acronym_lines_are_a_known_tradeoff(self):
        # 'API REST' is promoted (all-caps): harmless over-segmentation
        # accepted to keep the heuristic simple.
        out, changed = _apply("API REST")
        assert changed

    def test_mixed_case_title_without_colon_untouched(self):
        out, changed = _apply("Livrables attendus\nScripts Ansible")
        assert not changed


DOC = ("Introduction du projet\nCe document décrit la mission.\n\n"
       "Les livrables attendus\nLe projet fournira un rapport complet.\n")


class TestLlmStructureHeaders:
    """Stage-2 structuring: LLM proposes heading lines, only verbatim
    matches are promoted (LangExtract-style grounding)."""

    def test_verbatim_headings_are_promoted(self):
        resp = ('{"headings": ["Introduction du projet", '
                '"Les livrables attendus"]}')
        with patch('pageindex.utils.llm_completion', return_value=resp):
            out = llm_structure_headers(DOC, model='test')
        assert '## Introduction du projet' in out
        assert '## Les livrables attendus' in out
        assert 'Ce document décrit la mission.' in out

    def test_hallucinated_headings_are_dropped(self):
        resp = ('{"headings": ["Résumé exécutif", '
                '"Les livrables attendus"]}')
        with patch('pageindex.utils.llm_completion', return_value=resp):
            out = llm_structure_headers(DOC, model='test')
        assert '## Les livrables attendus' in out
        assert 'Résumé exécutif' not in out

    def test_all_hallucinated_returns_none(self):
        resp = '{"headings": ["Titre inventé", "Autre invention"]}'
        with patch('pageindex.utils.llm_completion', return_value=resp):
            assert llm_structure_headers(DOC, model='test') is None

    def test_empty_or_failed_response_returns_none(self):
        with patch('pageindex.utils.llm_completion', return_value=''):
            assert llm_structure_headers(DOC, model='test') is None
        with patch('pageindex.utils.llm_completion',
                   side_effect=RuntimeError('boom')):
            assert llm_structure_headers(DOC, model='test') is None
