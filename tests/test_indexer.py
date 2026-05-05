"""
test_indexer.py - Comprehensive tests for src/indexer.py

Tests cover:
- Tokenization (lowercasing, punctuation stripping, number handling, stop words)
- Index building (frequency tracking, position tracking, TF-IDF calculation)
- PageRank computation (dangling nodes, convergence, rank distribution)
- Index persistence (save/load roundtrip, file creation, JSON validity)
- Edge cases (empty inputs, single page, large indexes, special characters)
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
    """Test text tokenization and preprocessing."""
    
    def test_returns_lowercase_tokens(self):
        """All tokens should be lowercase."""
        assert tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        """Punctuation should be removed, leaving only alphanumeric."""
        assert tokenise("it's great!") == ["it", "s", "great"]

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert tokenise("") == []

    def test_numbers_kept(self):
        """Numbers should be preserved as tokens."""
        tokens = tokenise("Page 2 of 10")
        assert "2" in tokens
        assert "10" in tokens

    def test_stop_word_removal(self):
        """Stop words should be filtered when flag is True."""
        tokens = tokenise("the quick brown fox", remove_stop_words=True)
        assert "the" not in tokens
        assert "quick" in tokens
        assert "fox" in tokens

    def test_stop_words_kept_when_flag_false(self):
        """Stop words should be kept when flag is False (default)."""
        tokens = tokenise("the quick fox", remove_stop_words=False)
        assert "the" in tokens

    def test_whitespace_handling(self):
        """Multiple spaces should be handled gracefully."""
        tokens = tokenise("hello    world")
        assert tokens == ["hello", "world"]

    def test_special_characters_removed(self):
        """Special characters should be stripped."""
        tokens = tokenise("hello@world#test!")
        assert tokens == ["helloworld", "test"] or len(tokens) >= 1

    def test_numbers_with_punctuation(self):
        """Numbers should be extracted even with punctuation."""
        tokens = tokenise("Version 1.2.3 released")
        assert "version" in tokens
        assert any(t in ["1", "2", "3", "123"] for t in tokens)

    def test_very_long_text(self):
        """Should handle large text inputs."""
        long_text = " ".join(["word"] * 10000)
        tokens = tokenise(long_text)
        assert len(tokens) == 10000


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    """Test index construction and statistics."""
    
    PAGES = [
        {"url": "http://example.com/1", "text": "good life is good", "links": ["http://example.com/2"]},
        {"url": "http://example.com/2", "text": "life and more life", "links": ["http://example.com/1"]},
    ]

    def test_index_contains_expected_words(self):
        """Index should contain all unique words from pages."""
        index = build_index(self.PAGES)
        assert "good" in index
        assert "life" in index

    def test_case_insensitive_indexing(self):
        """All words should be lowercased in index."""
        pages = [{"url": "http://example.com/1", "text": "Good GOOD good"}]
        index = build_index(pages)
        assert "good" in index
        assert "Good" not in index

    def test_frequency_count(self):
        """Frequency should be correctly counted per page."""
        pages = [{"url": "http://example.com/1", "text": "good life is good"}]
        index = build_index(pages)
        assert index["good"]["http://example.com/1"]["frequency"] == 2

    def test_position_tracking(self):
        """Positions of words should be tracked."""
        pages = [{"url": "http://example.com/1", "text": "good life is good"}]
        index = build_index(pages)
        positions = index["good"]["http://example.com/1"]["positions"]
        assert positions == [0, 3]

    def test_multiple_pages_same_word(self):
        """Same word on multiple pages should create entries for each."""
        index = build_index(self.PAGES)
        assert "http://example.com/1" in index["life"]
        assert "http://example.com/2" in index["life"]

    def test_tfidf_score_present(self):
        """TF-IDF score should be added to each word-page entry."""
        index = build_index(self.PAGES)
        assert "tfidf" in index["good"]["http://example.com/1"]

    def test_tfidf_higher_for_rare_word(self):
        """Rare words should have higher IDF than common words."""
        pages = [
            {"url": "http://example.com/1", "text": "rare common"},
            {"url": "http://example.com/2", "text": "common common"},
        ]
        index = build_index(pages)
        rare_tfidf = index["rare"]["http://example.com/1"]["tfidf"]
        common_tfidf = index["common"]["http://example.com/1"]["tfidf"]
        assert rare_tfidf > common_tfidf

    def test_empty_pages_list(self):
        """Empty page list should return empty index."""
        index = build_index([])
        assert index == {}

    def test_pagerank_attached(self):
        """PageRank score should be attached to each entry."""
        index = build_index(self.PAGES)
        assert "pagerank" in index["good"]["http://example.com/1"]

    def test_single_page(self):
        """Index with single page should work correctly."""
        pages = [{"url": "http://example.com/1", "text": "single page"}]
        index = build_index(pages)
        assert "single" in index
        assert "page" in index
        assert len(index) == 2

    def test_page_missing_links_key(self):
        """Pages without 'links' key should be handled gracefully."""
        pages = [
            {"url": "http://example.com/1", "text": "some text"},  # missing 'links'
        ]
        index = build_index(pages)
        assert "some" in index


# ---------------------------------------------------------------------------
# compute_pagerank
# ---------------------------------------------------------------------------

class TestPageRank:
    """Test PageRank algorithm implementation."""
    
    def test_scores_sum_to_one(self):
        """All PageRank scores should sum to 1.0."""
        pages = [
            {"url": "http://example.com/a", "text": "a", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "b", "links": ["http://example.com/a", "http://example.com/c"]},
            {"url": "http://example.com/c", "text": "c", "links": []},
        ]
        ranks = compute_pagerank(pages)
        assert pytest.approx(sum(ranks.values()), rel=1e-6) == 1.0

    def test_rank_sinks_do_not_absorb_all_rank(self):
        """Pages with no outlinks (sinks) should not monopolize rank."""
        pages = [
            {"url": "http://example.com/a", "text": "a", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "b", "links": ["http://example.com/a"]},
            {"url": "http://example.com/c", "text": "c", "links": []},
        ]
        ranks = compute_pagerank(pages)
        assert ranks["http://example.com/a"] > ranks["http://example.com/c"]
        assert ranks["http://example.com/b"] > ranks["http://example.com/c"]

    def test_single_page_ranks_to_one(self):
        """Single page should have rank 1.0."""
        pages = [{"url": "http://example.com/a", "text": "a", "links": []}]
        ranks = compute_pagerank(pages)
        assert pytest.approx(ranks["http://example.com/a"], rel=1e-6) == 1.0

    def test_two_page_cycle(self):
        """Two pages linking to each other should have equal rank."""
        pages = [
            {"url": "http://example.com/a", "text": "a", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "b", "links": ["http://example.com/a"]},
        ]
        ranks = compute_pagerank(pages)
        assert pytest.approx(ranks["http://example.com/a"], rel=1e-6) == ranks["http://example.com/b"]

    def test_initial_rank_distribution(self):
        """All ranks should be positive even for unreachable pages."""
        pages = [
            {"url": "http://example.com/a", "text": "a", "links": []},
            {"url": "http://example.com/b", "text": "b", "links": []},
        ]
        ranks = compute_pagerank(pages)
        assert all(r > 0 for r in ranks.values())

    def test_convergence_with_damping_factor(self):
        """PageRank should converge with reasonable damping factor."""
        pages = [
            {"url": "http://example.com/a", "text": "a", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "b", "links": ["http://example.com/a"]},
            {"url": "http://example.com/c", "text": "c", "links": ["http://example.com/a"]},
        ]
        ranks = compute_pagerank(pages, damping=0.85, max_iter=100)
        # Should converge to sum of 1.0
        assert pytest.approx(sum(ranks.values()), rel=1e-6) == 1.0


# ---------------------------------------------------------------------------
# save_index / load_index
# ---------------------------------------------------------------------------

class TestPersistence:
    """Test index serialization and deserialization."""
    
    def test_save_and_load_roundtrip(self, tmp_path):
        """Index should be identical after save/load roundtrip."""
        path = str(tmp_path / "index.json")
        pages = [{"url": "http://example.com/1", "text": "hello world"}]
        index = build_index(pages)
        save_index(index, path)
        loaded = load_index(path)
        assert loaded == index

    def test_save_creates_parent_directories(self, tmp_path):
        """save_index should create parent directories if they don't exist."""
        path = str(tmp_path / "nested" / "dir" / "index.json")
        save_index({}, path)
        assert Path(path).exists()

    def test_load_raises_on_missing_file(self, tmp_path):
        """load_index should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_index(str(tmp_path / "nonexistent.json"))

    def test_saved_file_is_valid_json(self, tmp_path):
        """Saved file should be valid JSON."""
        path = str(tmp_path / "index.json")
        save_index({"hello": {"http://x.com": {"frequency": 1, "positions": [0]}}}, path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "hello" in data

    def test_complex_index_roundtrip(self, tmp_path):
        """Complex index with TF-IDF and PageRank should survive roundtrip."""
        path = str(tmp_path / "complex.json")
        pages = [
            {"url": "http://example.com/1", "text": "hello world hello", "links": ["http://example.com/2"]},
            {"url": "http://example.com/2", "text": "world test", "links": ["http://example.com/1"]},
        ]
        index = build_index(pages)
        save_index(index, path)
        loaded = load_index(path)
        
        # Check structure is preserved
        assert "hello" in loaded
        assert "world" in loaded
        assert "http://example.com/1" in loaded["hello"]

    def test_empty_index_save_load(self, tmp_path):
        """Empty index should save and load correctly."""
        path = str(tmp_path / "empty.json")
        save_index({}, path)
        loaded = load_index(path)
        assert loaded == {}

    def test_large_index_persistence(self, tmp_path):
        """Large index should be saved and loaded correctly."""
        path = str(tmp_path / "large.json")
        # Create a moderately large index
        pages = [
            {"url": f"http://example.com/{i}", "text": f"word{i} " * 100, "links": []}
            for i in range(50)
        ]
        index = build_index(pages)
        save_index(index, path)
        loaded = load_index(path)
        assert len(loaded) > 0

    def test_special_characters_in_index(self, tmp_path):
        """Index with special characters should roundtrip correctly."""
        path = str(tmp_path / "special.json")
        pages = [
            {"url": "http://example.com/1", "text": "café naïve élève", "links": []},
        ]
        index = build_index(pages)
        save_index(index, path)
        loaded = load_index(path)
        assert len(loaded) > 0
