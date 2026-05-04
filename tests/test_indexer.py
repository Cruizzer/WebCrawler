"""
test_indexer.py - Tests for src/indexer.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import (
    tokenise,
    build_index,
    compute_pagerank,
    save_index,
    load_index,
    STOP_WORDS,
)


# ---------------------------------------------------------------------------
# tokenise
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_returns_lowercase_tokens(self):
        assert tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenise("it's great!") == ["it", "s", "great"]

    def test_empty_string(self):
        assert tokenise("") == []

    def test_numbers_kept(self):
        tokens = tokenise("Page 2 of 10")
        assert "2" in tokens
        assert "10" in tokens

    def test_stop_word_removal(self):
        tokens = tokenise("the quick brown fox", remove_stop_words=True)
        assert "the" not in tokens
        assert "quick" in tokens
        assert "fox" in tokens

    def test_stop_words_kept_when_flag_false(self):
        tokens = tokenise("the quick fox", remove_stop_words=False)
        assert "the" in tokens


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    PAGES = [
        {"url": "http://example.com/1", "text": "good life is good", "links": ["http://example.com/2"]},
        {"url": "http://example.com/2", "text": "life and more life", "links": ["http://example.com/1"]},
    ]

    def test_index_contains_expected_words(self):
        index = build_index(self.PAGES)
        assert "good" in index
        assert "life" in index

    def test_case_insensitive(self):
        pages = [{"url": "http://example.com/1", "text": "Good GOOD good"}]
        index = build_index(pages)
        assert "good" in index
        assert "Good" not in index

    def test_frequency_count(self):
        pages = [{"url": "http://example.com/1", "text": "good life is good"}]
        index = build_index(pages)
        assert index["good"]["http://example.com/1"]["frequency"] == 2

    def test_position_tracking(self):
        pages = [{"url": "http://example.com/1", "text": "good life is good"}]
        index = build_index(pages)
        positions = index["good"]["http://example.com/1"]["positions"]
        assert positions == [0, 3]

    def test_multiple_pages(self):
        index = build_index(self.PAGES)
        assert "http://example.com/1" in index["life"]
        assert "http://example.com/2" in index["life"]

    def test_tfidf_score_present(self):
        index = build_index(self.PAGES)
        assert "tfidf" in index["good"]["http://example.com/1"]

    def test_tfidf_higher_for_rare_word(self):
        """A word on only one page should have a higher IDF than one on all."""
        pages = [
            {"url": "http://example.com/1", "text": "rare common"},
            {"url": "http://example.com/2", "text": "common common"},
        ]
        index = build_index(pages)
        # 'rare' appears on 1/2 pages; 'common' appears on 2/2 pages
        rare_idf_page1 = index["rare"]["http://example.com/1"]["tfidf"]
        common_idf_page1 = index["common"]["http://example.com/1"]["tfidf"]
        assert rare_idf_page1 > common_idf_page1

    def test_empty_pages_list(self):
        index = build_index([])
        assert index == {}

    def test_pagerank_is_attached_to_page_entries(self):
        index = build_index(self.PAGES)
        assert "pagerank" in index["good"]["http://example.com/1"]


class TestPageRank:
    def test_scores_sum_to_one(self):
        pages = [
            {"url": "http://example.com/a", "text": "a", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "b", "links": ["http://example.com/a", "http://example.com/c"]},
            {"url": "http://example.com/c", "text": "c", "links": []},
        ]
        ranks = compute_pagerank(pages)
        assert pytest.approx(sum(ranks.values()), rel=1e-6) == 1.0

    def test_rank_sinks_do_not_absorb_all_rank(self):
        pages = [
            {"url": "http://example.com/a", "text": "a", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "b", "links": ["http://example.com/a"]},
            {"url": "http://example.com/c", "text": "c", "links": []},
        ]
        ranks = compute_pagerank(pages)
        assert ranks["http://example.com/a"] > ranks["http://example.com/c"]
        assert ranks["http://example.com/b"] > ranks["http://example.com/c"]


# ---------------------------------------------------------------------------
# save_index / load_index
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "index.json")
        pages = [{"url": "http://example.com/1", "text": "hello world"}]
        index = build_index(pages)
        save_index(index, path)
        loaded = load_index(path)
        assert loaded == index

    def test_save_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "index.json")
        save_index({}, path)
        assert Path(path).exists()

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_index(str(tmp_path / "nonexistent.json"))

    def test_saved_file_is_valid_json(self, tmp_path):
        path = str(tmp_path / "index.json")
        save_index({"hello": {"http://x.com": {"frequency": 1, "positions": [0]}}}, path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "hello" in data
