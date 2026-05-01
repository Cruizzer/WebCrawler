"""
test_search.py - Tests for src/search.py
"""

import sys
from io import StringIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import build_index, compute_pagerank
from search import find_word, find_query, print_word, print_query_results


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def index():
    pages = [
        {"url": "http://example.com/1", "text": "good life is good and great"},
        {"url": "http://example.com/2", "text": "life is beautiful life"},
        {"url": "http://example.com/3", "text": "great things happen"},
    ]
    return build_index(pages)


# ---------------------------------------------------------------------------
# find_word
# ---------------------------------------------------------------------------

class TestFindWord:
    def test_finds_existing_word(self, index):
        result = find_word("life", index)
        assert result is not None

    def test_returns_none_for_missing_word(self, index):
        assert find_word("xyznotaword", index) == None  # noqa: E711

    def test_case_insensitive_lookup(self, index):
        assert find_word("GOOD", index) == find_word("good", index)

    def test_returned_dict_has_frequency(self, index):
        result = find_word("life", index)
        for stats in result.values():
            assert "frequency" in stats

    def test_returned_dict_has_positions(self, index):
        result = find_word("life", index)
        for stats in result.values():
            assert "positions" in stats


# ---------------------------------------------------------------------------
# find_query
# ---------------------------------------------------------------------------

class TestFindQuery:
    def test_single_word_query_returns_results(self, index):
        results = find_query("life", index)
        assert len(results) > 0

    def test_multi_word_and_query(self, index):
        # "good" is only on page 1; "life" is on pages 1 and 2 → intersection = page 1
        results = find_query("good life", index)
        urls = [r["url"] for r in results]
        assert "http://example.com/1" in urls
        assert "http://example.com/2" not in urls

    def test_no_results_for_missing_word(self, index):
        assert find_query("xyznotaword", index) == []

    def test_empty_query_returns_empty(self, index):
        assert find_query("", index) == []

    def test_results_contain_expected_keys(self, index):
        results = find_query("life", index)
        for r in results:
            assert "url" in r
            assert "matched_words" in r
            assert "total_freq" in r
            assert "total_tfidf" in r

    def test_ranking_by_tfidf(self, index):
        """Page 2 has 'life' twice; page 1 once - page 2 should rank higher."""
        results = find_query("life", index)
        urls_in_order = [r["url"] for r in results]
        assert urls_in_order[0] == "http://example.com/2"

    def test_multi_word_no_common_page(self, index):
        # "great" is on pages 1 & 3; "beautiful" is only on page 2 → empty
        results = find_query("great beautiful", index)
        assert results == []

    def test_whitespace_only_query(self, index):
        assert find_query("   ", index) == []

    def test_page_rank_breaks_tie_when_tfidf_is_equal(self):
        pages = [
            {"url": "http://example.com/a", "text": "shared token", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "shared token", "links": ["http://example.com/a", "http://example.com/c"]},
            {"url": "http://example.com/c", "text": "shared token", "links": ["http://example.com/a"]},
        ]
        index = build_index(pages)
        results = find_query("shared", index)
        ranks = compute_pagerank(pages)
        expected_order = [url for url, _ in sorted(ranks.items(), key=lambda item: item[1], reverse=True)]
        assert [result["url"] for result in results] == expected_order


# ---------------------------------------------------------------------------
# print_word / print_query_results (smoke tests via stdout capture)
# ---------------------------------------------------------------------------

class TestPrintFunctions:
    def test_print_word_existing(self, index, capsys):
        print_word("life", index)
        captured = capsys.readouterr()
        assert "life" in captured.out
        assert "frequency" in captured.out

    def test_print_word_missing(self, index, capsys):
        print_word("xyznotaword", index)
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_print_query_results_with_results(self, index, capsys):
        results = find_query("life", index)
        print_query_results("life", results)
        captured = capsys.readouterr()
        assert "http://example.com" in captured.out

    def test_print_query_results_empty(self, index, capsys):
        print_query_results("xyznotaword", [])
        captured = capsys.readouterr()
        assert "No pages found" in captured.out
