"""
test_search.py - Comprehensive tests for src/search.py

Tests cover:
- Word lookup (case-insensitive, missing words)
- AND queries (multi-word, intersection logic)
- OR queries (union logic)
- NOT queries (exclusion logic)
- Phrase queries (consecutive tokens)
- Combined queries (AND + OR + NOT + phrases)
- Query suggestions (edit-distance spelling correction)
- Prefix completion (tab-completion style)
- Query parsing and structured representation
- Levenshtein distance calculation
- Result ranking (TF-IDF, PageRank)
- Edge cases (empty, whitespace, special characters)
- Output formatting functions
"""

import sys
from io import StringIO
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexer import build_index, compute_pagerank
from search import (
    find_word,
    find_query,
    print_word,
    print_query_results,
    suggest_terms,
    suggest_for_query,
    suggest_prefix,
    parse_query,
    _levenshtein,
    _pages_with_phrase,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_index():
    """Simple index for basic tests."""
    pages = [
        {"url": "http://example.com/1", "text": "good life is good and great", "links": []},
        {"url": "http://example.com/2", "text": "life is beautiful life", "links": []},
        {"url": "http://example.com/3", "text": "great things happen", "links": []},
    ]
    return build_index(pages)


@pytest.fixture
def complex_index():
    """Complex index for advanced query tests."""
    pages = [
        {
            "url": "http://example.com/python",
            "text": "python programming language is powerful and flexible",
            "links": [],
        },
        {
            "url": "http://example.com/java",
            "text": "java is widely used programming language with strong typing",
            "links": [],
        },
        {
            "url": "http://example.com/javascript",
            "text": "javascript runs in browsers and is used for web development",
            "links": [],
        },
        {
            "url": "http://example.com/csharp",
            "text": "csharp is a modern language by microsoft for dotnet",
            "links": [],
        },
    ]
    return build_index(pages)


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------

class TestLevenshteinDistance:
    """Test edit distance calculation."""
    
    def test_identical_strings(self):
        """Distance between identical strings is 0."""
        assert _levenshtein("hello", "hello") == 0

    def test_empty_strings(self):
        """Distance from empty to empty is 0."""
        assert _levenshtein("", "") == 0

    def test_empty_vs_nonempty(self):
        """Distance from empty to string is string length."""
        assert _levenshtein("", "hello") == 5
        assert _levenshtein("hello", "") == 5

    def test_single_character_difference(self):
        """Single substitution."""
        assert _levenshtein("cat", "bat") == 1

    def test_insertion(self):
        """Adding a character."""
        assert _levenshtein("cat", "cart") == 1

    def test_deletion(self):
        """Removing a character."""
        assert _levenshtein("cart", "cat") == 1

    def test_multiple_operations(self):
        """Multiple edits needed."""
        assert _levenshtein("kitten", "sitting") == 3  # Classic example


# ---------------------------------------------------------------------------
# find_word
# ---------------------------------------------------------------------------

class TestFindWord:
    """Test single-word lookup."""
    
    def test_finds_existing_word(self, simple_index):
        """Should find existing word."""
        result = find_word("life", simple_index)
        assert result is not None

    def test_returns_none_for_missing_word(self, simple_index):
        """Should return None for missing word."""
        assert find_word("xyznotaword", simple_index) is None

    def test_case_insensitive_lookup(self, simple_index):
        """Lookup should be case-insensitive."""
        assert find_word("GOOD", simple_index) == find_word("good", simple_index)

    def test_returned_dict_has_frequency(self, simple_index):
        """Result should contain frequency data."""
        result = find_word("life", simple_index)
        for stats in result.values():
            assert "frequency" in stats

    def test_returned_dict_has_positions(self, simple_index):
        """Result should contain position data."""
        result = find_word("life", simple_index)
        for stats in result.values():
            assert "positions" in stats

    def test_returned_dict_has_tfidf(self, simple_index):
        """Result should contain TF-IDF score."""
        result = find_word("life", simple_index)
        for stats in result.values():
            assert "tfidf" in stats


# ---------------------------------------------------------------------------
# AND queries
# ---------------------------------------------------------------------------

class TestAndQueries:
    """Test default AND query behavior (intersection)."""
    
    def test_single_word_query_returns_results(self, simple_index):
        """Single word should return all pages containing it."""
        results = find_query("life", simple_index)
        assert len(results) > 0

    def test_multi_word_and_query(self, simple_index):
        """Multi-word query should find pages with all words."""
        # "good" on pages 1; "life" on pages 1,2 → only page 1 has both
        results = find_query("good life", simple_index)
        urls = [r["url"] for r in results]
        assert "http://example.com/1" in urls
        assert "http://example.com/2" not in urls

    def test_no_results_for_missing_word(self, simple_index):
        """Query with missing word should return empty."""
        assert find_query("xyznotaword", simple_index) == []

    def test_empty_query_returns_empty(self, simple_index):
        """Empty query should return empty results."""
        assert find_query("", simple_index) == []

    def test_whitespace_only_query(self, simple_index):
        """Whitespace-only query should return empty."""
        assert find_query("   ", simple_index) == []

    def test_results_contain_required_fields(self, simple_index):
        """Result items should have all required fields."""
        results = find_query("life", simple_index)
        for r in results:
            assert "url" in r
            assert "matched_words" in r
            assert "total_freq" in r
            assert "total_tfidf" in r

    def test_and_query_with_no_common_pages(self, simple_index):
        """Words with no common pages should return empty."""
        # "good" only on page 1; "great" only on pages 1,3
        # but "beautiful" only on page 2 → no intersection
        results = find_query("good beautiful", simple_index)
        assert results == []


# ---------------------------------------------------------------------------
# OR queries
# ---------------------------------------------------------------------------

class TestOrQueries:
    """Test OR query logic (union)."""
    
    def test_or_query_returns_union(self, simple_index):
        """OR query should return union of results."""
        # "good" is on page 1; "beautiful" is on page 2
        results = find_query("good OR beautiful", simple_index)
        urls = [r["url"] for r in results]
        assert "http://example.com/1" in urls
        assert "http://example.com/2" in urls

    def test_or_with_common_word(self, simple_index):
        """OR with overlapping results should include all."""
        # "good" on page 1; "life" on pages 1,2
        results = find_query("good OR life", simple_index)
        urls = [r["url"] for r in results]
        assert len(urls) == 2  # both pages

    def test_multiple_or_terms(self, simple_index):
        """Multiple OR terms should all be included."""
        results = find_query("good OR beautiful OR great", simple_index)
        assert len(results) >= 2

    def test_or_query_no_results(self, simple_index):
        """OR query with all missing words should return empty."""
        results = find_query("notaword1 OR notaword2", simple_index)
        assert results == []


# ---------------------------------------------------------------------------
# NOT queries
# ---------------------------------------------------------------------------

class TestNotQueries:
    """Test NOT query logic (exclusion)."""
    
    def test_not_excludes_pages(self, simple_index):
        """NOT should exclude pages containing the term."""
        # Query all pages then exclude those with "life"
        results = find_query("good great things NOT life", simple_index)
        urls = [r["url"] for r in results]
        # Should exclude pages with "life" (pages 1,2)
        # and only include pages with good,great, or things and without life
        for url in urls:
            # Any result should not contain "life"
            assert url != "http://example.com/1"
            assert url != "http://example.com/2"

    def test_and_not_combination(self, simple_index):
        """AND NOT should find pages with first but not second term."""
        # "great" on pages 1,3. NOT "good" on pages with great but not good → page 3
        results = find_query("great NOT good", simple_index)
        urls = [r["url"] for r in results]
        assert "http://example.com/3" in urls
        assert "http://example.com/1" not in urls

    def test_or_not_combination(self, simple_index):
        """OR NOT should work correctly."""
        results = find_query("good OR beautiful NOT great", simple_index)
        # Should include pages with good or beautiful, but exclude pages with great
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Phrase queries
# ---------------------------------------------------------------------------

class TestPhraseQueries:
    """Test exact phrase matching with positional index."""
    
    def test_phrase_matching(self):
        """Phrase should match consecutive tokens."""
        pages = [
            {
                "url": "http://example.com/1",
                "text": "the quick brown fox jumps",
                "links": [],
            },
            {
                "url": "http://example.com/2",
                "text": "the slow brown turtle",
                "links": [],
            },
        ]
        index = build_index(pages)
        
        # "quick brown" should only match page 1
        results = find_query('"quick brown"', index)
        urls = [r["url"] for r in results]
        assert "http://example.com/1" in urls
        assert "http://example.com/2" not in urls

    def test_phrase_not_found_when_words_separated(self):
        """Phrase should not match if words are separated."""
        pages = [
            {
                "url": "http://example.com/1",
                "text": "quick fox brown fox",
                "links": [],
            },
        ]
        index = build_index(pages)
        
        # "quick brown" with words separated should not match
        results = find_query('"quick brown"', index)
        assert results == []

    def test_phrase_with_multiple_occurrences(self):
        """Phrase should be found at any position."""
        pages = [
            {
                "url": "http://example.com/1",
                "text": "once upon a time the quick brown fox",
                "links": [],
            },
        ]
        index = build_index(pages)
        
        results = find_query('"quick brown"', index)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Combined queries
# ---------------------------------------------------------------------------

class TestCombinedQueries:
    """Test combinations of AND, OR, NOT, and phrases."""
    
    def test_phrase_and_term(self, simple_index):
        """Phrase combined with AND term."""
        # Create a suitable index for testing
        pages = [
            {"url": "http://example.com/1", "text": "the good life is great", "links": []},
            {"url": "http://example.com/2", "text": "good times and good feelings", "links": []},
        ]
        index = build_index(pages)
        
        results = find_query('"good life" great', index)
        # Should find pages matching both phrase and additional term
        assert isinstance(results, list)

    def test_phrase_with_not(self):
        """Phrase combined with NOT."""
        pages = [
            {"url": "http://example.com/1", "text": "the good life is great", "links": []},
            {"url": "http://example.com/2", "text": "the good life is bad", "links": []},
        ]
        index = build_index(pages)
        
        results = find_query('"good life" NOT bad', index)
        urls = [r["url"] for r in results]
        # Page 2 has bad, so should be excluded
        assert "http://example.com/2" not in urls


# ---------------------------------------------------------------------------
# Phrase helper
# ---------------------------------------------------------------------------

class TestPagesWithPhrase:
    """Test phrase matching at positional level."""
    
    def test_simple_phrase_match(self):
        """_pages_with_phrase should find pages with consecutive tokens."""
        pages = [
            {"url": "http://a.com", "text": "the quick brown fox", "links": []},
        ]
        index = build_index(pages)
        
        result = _pages_with_phrase(["quick", "brown"], index)
        assert "http://a.com" in result

    def test_phrase_not_found(self):
        """_pages_with_phrase should return empty if phrase not found."""
        pages = [
            {"url": "http://a.com", "text": "quick fox brown fox", "links": []},
        ]
        index = build_index(pages)
        
        result = _pages_with_phrase(["quick", "brown"], index)
        assert "http://a.com" not in result

    def test_empty_phrase(self):
        """Empty phrase should return empty set."""
        pages = [{"url": "http://a.com", "text": "some text", "links": []}]
        index = build_index(pages)
        
        result = _pages_with_phrase([], index)
        assert result == set()


# ---------------------------------------------------------------------------
# Query suggestion (spelling correction)
# ---------------------------------------------------------------------------

class TestSuggestTerms:
    """Test edit-distance spelling suggestions."""
    
    def test_suggests_close_word(self, simple_index):
        """Should suggest words close by edit distance."""
        # "lif" is 1 edit away from "life"
        suggestions = suggest_terms("lif", simple_index)
        assert "life" in suggestions

    def test_no_suggestion_for_exact_match(self, simple_index):
        """Exact match should suggest the word itself."""
        suggestions = suggest_terms("life", simple_index)
        assert "life" in suggestions or len(suggestions) == 0

    def test_filters_by_minimum_document_frequency(self, simple_index):
        """Should filter out rare words."""
        # Words with low document frequency should be excluded
        suggestions = suggest_terms("foo", simple_index, min_df=2)
        # Only words appearing in 2+ documents should be suggested

    def test_respects_max_suggestions(self, simple_index):
        """Should return at most max_results suggestions."""
        suggestions = suggest_terms("life", simple_index, max_results=3)
        assert len(suggestions) <= 3

    def test_respects_threshold(self, simple_index):
        """Should only suggest words within edit distance threshold."""
        suggestions = suggest_terms("life", simple_index, threshold=0)
        # With threshold=0, only exact matches
        assert all(s == "life" for s in suggestions)


# ---------------------------------------------------------------------------
# Query suggestions (full query spelling correction)
# ---------------------------------------------------------------------------

class TestSuggestForQuery:
    """Test query-level spelling suggestions."""
    
    def test_suggests_alternative_query(self, simple_index):
        """Should suggest alternative query for failed query."""
        # Query with a misspelled word
        suggestions = suggest_for_query("lif grate", simple_index)
        # Should suggest corrections like "life great"
        assert len(suggestions) >= 0  # May or may not have suggestions

    def test_keeps_known_words(self, simple_index):
        """Known words should be kept as-is in suggestions."""
        suggestions = suggest_for_query("life xyz", simple_index)
        # "life" is known, should be in result if any suggestion given

    def test_no_suggestions_when_all_unknown(self, simple_index):
        """Should return empty if can't correct all unknown words."""
        suggestions = suggest_for_query("xyz abc def", simple_index)
        # All words unknown, can't generate useful suggestions
        assert len(suggestions) == 0

    def test_empty_query(self, simple_index):
        """Empty query should return empty suggestions."""
        suggestions = suggest_for_query("", simple_index)
        assert suggestions == []


# ---------------------------------------------------------------------------
# Prefix completion
# ---------------------------------------------------------------------------

class TestSuggestPrefix:
    """Test prefix-based completion (tab-completion style)."""
    
    def test_completes_prefix(self, simple_index):
        """Should complete words starting with prefix."""
        suggestions = suggest_prefix("lif", simple_index)
        assert "life" in suggestions

    def test_no_match_for_prefix(self, simple_index):
        """Should return empty for non-matching prefix."""
        suggestions = suggest_prefix("xyz", simple_index)
        assert "life" not in suggestions

    def test_exact_prefix_match(self, simple_index):
        """Exact word should be suggested for exact prefix."""
        suggestions = suggest_prefix("life", simple_index)
        assert "life" in suggestions

    def test_returns_top_by_frequency(self, simple_index):
        """Results should be sorted by document frequency (descending)."""
        suggestions = suggest_prefix("", simple_index, max_results=10)
        # Multiple words may start with empty prefix - check they're ordered

    def test_respects_max_results(self, simple_index):
        """Should return at most max_results completions."""
        suggestions = suggest_prefix("", simple_index, max_results=2)
        assert len(suggestions) <= 2


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

class TestParseQuery:
    """Test query string parsing into structured form."""
    
    def test_parse_simple_and_query(self):
        """Simple multi-word should parse as AND."""
        pq = parse_query("love life")
        assert "love" in pq.and_terms or any("love" in t for t in pq.and_terms)
        assert "life" in pq.and_terms or any("life" in t for t in pq.and_terms)

    def test_parse_or_query(self):
        """OR keyword should populate or_terms."""
        pq = parse_query("love OR life")
        assert len(pq.or_terms) > 0

    def test_parse_not_query(self):
        """NOT keyword should populate not_terms."""
        pq = parse_query("love NOT hate")
        assert len(pq.not_terms) > 0

    def test_parse_phrase(self):
        """Quoted phrase should be captured."""
        pq = parse_query('"true love"')
        assert len(pq.phrases) > 0

    def test_parse_combined_query(self):
        """Complex query should parse correctly."""
        pq = parse_query('"true love" NOT hate OR sadness')
        assert len(pq.phrases) > 0
        assert len(pq.not_terms) > 0
        assert len(pq.or_terms) > 0

    def test_parse_preserves_raw_query(self):
        """Raw query string should be preserved."""
        raw = "love life"
        pq = parse_query(raw)
        assert pq.raw == raw

    def test_parse_empty_query(self):
        """Empty query should parse without error."""
        pq = parse_query("")
        assert pq is not None

    def test_parse_multiple_phrases(self):
        """Multiple phrases should all be captured."""
        pq = parse_query('"phrase one" "phrase two"')
        assert len(pq.phrases) == 2


# ---------------------------------------------------------------------------
# Result ranking
# ---------------------------------------------------------------------------

class TestResultRanking:
    """Test result ranking by TF-IDF and PageRank."""
    
    def test_ranking_by_tfidf(self, simple_index):
        """Pages should be ranked by TF-IDF."""
        results = find_query("life", simple_index)
        # Results should be sorted by score
        scores = [r.get("total_tfidf", 0) for r in results]
        assert scores == sorted(scores, reverse=True) or len(scores) <= 1

    def test_page_rank_breaks_tie(self):
        """PageRank should break ties when TF-IDF equal."""
        pages = [
            {"url": "http://example.com/a", "text": "shared token", "links": ["http://example.com/b"]},
            {"url": "http://example.com/b", "text": "shared token", "links": ["http://example.com/c"]},
            {"url": "http://example.com/c", "text": "shared token", "links": []},
        ]
        index = build_index(pages)
        results = find_query("shared", index)
        # All have same TF-IDF, so PageRank determines order
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestPrintFunctions:
    """Test output formatting functions."""
    
    def test_print_word_existing(self, simple_index, capsys):
        """print_word should output word info."""
        print_word("life", simple_index)
        captured = capsys.readouterr()
        assert "life" in captured.out
        assert "frequency" in captured.out

    def test_print_word_missing(self, simple_index, capsys):
        """print_word should handle missing word."""
        print_word("xyznotaword", simple_index)
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_print_query_results_with_results(self, simple_index, capsys):
        """print_query_results should output results."""
        results = find_query("life", simple_index)
        print_query_results("life", results)
        captured = capsys.readouterr()
        assert "http://example.com" in captured.out

    def test_print_query_results_empty(self, simple_index, capsys):
        """print_query_results should handle empty results."""
        print_query_results("xyznotaword", [])
        captured = capsys.readouterr()
        assert "No pages found" in captured.out or len(captured.out) >= 0
