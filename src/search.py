"""
search.py - Query parsing, index searching, result ranking, and query suggestions.

Supports four query modes, all parsed from a single query string:

    AND (default)   find love life        → pages containing both 'love' AND 'life'
    OR              find love OR life     → pages containing 'love' OR 'life' (union)
    NOT             find love NOT hate    → pages with 'love' but NOT 'hate'
    Phrase          find "the world"      → pages where 'the world' appears consecutively

Modes may be combined:
    find "true love" NOT hate OR sadness

Query suggestions (edit-distance spelling correction) are offered automatically
when an AND query returns zero results, and are also available via the
``suggest`` command.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from indexer import tokenise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SUGGESTIONS = 5          # suggestions shown per failed query
_SUGGESTION_THRESHOLD = 2     # max edit distance for a suggestion to qualify
_MIN_SUGGESTION_DF = 2        # minimum document frequency for a suggestion word


# ---------------------------------------------------------------------------
# Single-word lookup
# ---------------------------------------------------------------------------

def find_word(word: str, index: dict) -> dict[str, Any] | None:
    """
    Return index entries for a single *word*, or None if not found.

    The lookup is case-insensitive.

    Args:
        word:  The word to look up.
        index: Inverted index dictionary.

    Returns:
        Dict mapping page URLs to their stats, or None.
    """
    return index.get(word.lower())


def print_word(word: str, index: dict) -> None:
    """
    Print formatted index data for *word* to stdout.

    Args:
        word:  The word to display.
        index: Inverted index dictionary.
    """
    entries = find_word(word, index)
    if entries is None:
        print(f"Word '{word}' not found in the index.")
        return

    print(f"\nWord: {word.lower()}\n")
    print("Found in:")
    for url, stats in entries.items():
        print(f"  - {url}")
        print(f"      frequency : {stats['frequency']}")
        print(f"      positions : {stats['positions']}")
        if "tfidf" in stats:
            print(f"      tfidf     : {stats['tfidf']}")
        if "pagerank" in stats:
            print(f"      pagerank  : {stats['pagerank']}")
    print()


# ---------------------------------------------------------------------------
# Query suggestion - edit-distance spelling correction
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """
    Compute the Levenshtein edit distance between strings *a* and *b*.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Integer edit distance (0 = identical).
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a          # ensure |a| >= |b|
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,               # deletion
                curr[j - 1] + 1,           # insertion
                prev[j - 1] + (ca != cb),  # substitution
            )
        prev = curr
    return prev[-1]


def suggest_terms(
    word: str,
    index: dict,
    max_results: int = _MAX_SUGGESTIONS,
    threshold: int = _SUGGESTION_THRESHOLD,
    min_df: int = _MIN_SUGGESTION_DF,
) -> list[str]:
    """
    Return vocabulary words close to *word* by edit distance.

    Candidates are filtered to those with document frequency >= *min_df* so
    that rare index noise (e.g. HTML artefacts) is not surfaced.  Results are
    sorted by (edit_distance, -document_frequency) so the most useful
    correction comes first.

    Args:
        word:        The misspelled (or unknown) query term.
        index:       Inverted index dictionary.
        max_results: Maximum number of suggestions to return.
        threshold:   Maximum edit distance to consider a candidate.
        min_df:      Minimum document frequency for a candidate.

    Returns:
        List of suggested vocabulary words, best match first.
    """
    word = word.lower()
    candidates: list[tuple[int, int, str]] = []   # (dist, -df, term)

    for term, pages in index.items():
        df = len(pages)
        if df < min_df:
            continue
        dist = _levenshtein(word, term)
        if dist <= threshold:
            candidates.append((dist, -df, term))

    candidates.sort()
    return [term for _, _, term in candidates[:max_results]]


def suggest_for_query(
    query: str,
    index: dict,
    max_results: int = _MAX_SUGGESTIONS,
) -> list[str]:
    """
    Return a list of alternative query strings for a failed *query*.

    Each unknown token is independently corrected via :func:`suggest_terms`.
    Known tokens are kept as-is.  Returns at most *max_results* alternatives.

    Args:
        query:       The original (failed) query string.
        index:       Inverted index dictionary.
        max_results: Maximum number of alternative queries to return.

    Returns:
        List of alternative query strings.
    """
    tokens = tokenise(query)
    if not tokens:
        return []

    # For each token, get up to max_results suggestions (or keep if known).
    per_token: list[list[str]] = []
    for token in tokens:
        if token in index:
            per_token.append([token])
        else:
            alts = suggest_terms(token, index, max_results=max_results)
            if not alts:
                return []          # can't suggest anything useful
            per_token.append(alts)

    # Build alternatives by taking the i-th suggestion for each token.
    alternatives: list[str] = []
    max_depth = max(len(opts) for opts in per_token)
    for i in range(min(max_depth, max_results)):
        parts: list[str] = []
        for opts in per_token:
            parts.append(opts[min(i, len(opts) - 1)])
        candidate = " ".join(parts)
        if candidate not in alternatives:
            alternatives.append(candidate)

    return alternatives[:max_results]


def suggest_prefix(
    prefix: str,
    index: dict,
    max_results: int = _MAX_SUGGESTIONS,
    min_df: int = _MIN_SUGGESTION_DF,
) -> list[str]:
    """
    Return index terms that begin with *prefix*, ranked by document frequency.

    Useful for tab-completion style ``suggest`` commands.

    Args:
        prefix:      The prefix string to match.
        index:       Inverted index dictionary.
        max_results: Maximum number of completions to return.
        min_df:      Minimum document frequency for a candidate.

    Returns:
        List of matching terms, most frequent first.
    """
    prefix = prefix.lower()
    hits: list[tuple[int, str]] = []

    for term, pages in index.items():
        df = len(pages)
        if df >= min_df and term.startswith(prefix):
            hits.append((-df, term))   # negative for descending sort

    hits.sort()
    return [term for _, term in hits[:max_results]]


# ---------------------------------------------------------------------------
# Advanced query parser
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """
    Structured representation of a parsed query.

    Attributes:
        phrases:    List of phrase constraints - each is a list of tokens that
                    must appear consecutively in that order.
        and_terms:  Tokens that must all appear on a matching page (AND logic).
        or_terms:   Tokens where at least one must appear (OR logic).
        not_terms:  Tokens that must NOT appear on a matching page.
        raw:        Original query string before parsing.
    """
    phrases: list[list[str]] = field(default_factory=list)
    and_terms: list[str] = field(default_factory=list)
    or_terms: list[str] = field(default_factory=list)
    not_terms: list[str] = field(default_factory=list)
    raw: str = ""


def parse_query(query: str) -> ParsedQuery:
    """
    Parse a query string into a :class:`ParsedQuery` structure.

    Grammar (evaluated left-to-right after phrase extraction)::

        query     := clause ( WS clause )*
        clause    := NOT term | OR term | AND term | "phrase" | term
        term      := alphanumeric token
        phrase    := '"' token ( WS token )* '"'

    Phrases are extracted first (before token splitting), so quoted
    multi-word expressions are treated as atomic.  ``OR`` and ``NOT``
    are case-insensitive operator keywords.  ``AND`` acts as a binary
    operator when it sits between two query terms, but it can still be
    searched as a literal word when it appears on its own.

    Args:
        query: Raw query string from the user.

    Returns:
        A :class:`ParsedQuery` capturing the structured intent.

    Examples:
        >>> parse_query('love life')
        ParsedQuery(and_terms=['love', 'life'], ...)
        >>> parse_query('"true love" NOT hate')
        ParsedQuery(phrases=[['true', 'love']], not_terms=['hate'], ...)
        >>> parse_query('love OR life')
        ParsedQuery(and_terms=['love'], or_terms=['life'], ...)
    """
    pq = ParsedQuery(raw=query)

    # 1. Extract quoted phrases and replace them with placeholders so the
    #    surrounding operator structure is preserved.
    phrase_re = re.compile(r'"([^"]+)"')
    phrase_tokens_by_placeholder: dict[str, list[str]] = {}

    def _replace_phrase(match: re.Match[str]) -> str:
        placeholder = f"__PHRASE{len(phrase_tokens_by_placeholder)}__"
        phrase_tokens = tokenise(match.group(1))
        if phrase_tokens:
            phrase_tokens_by_placeholder[placeholder] = phrase_tokens
        return placeholder

    remaining = phrase_re.sub(_replace_phrase, query).strip()

    # 2. Tokenise the remainder and classify by operator prefix.
    raw_tokens = remaining.split()
    i = 0
    while i < len(raw_tokens):
        token = raw_tokens[i]
        upper = token.upper()

        if token in phrase_tokens_by_placeholder:
            pq.phrases.append(phrase_tokens_by_placeholder[token])
            i += 1
            continue

        prev_token = raw_tokens[i - 1] if i > 0 else None
        next_token = raw_tokens[i + 1] if i + 1 < len(raw_tokens) else None

        if upper == "AND":
            # Treat AND as an operator only when it connects two operands.
            prev_is_operand = prev_token is not None and prev_token.upper() not in {"AND", "OR", "NOT"}
            next_is_operand = next_token is not None and next_token.upper() not in {"AND", "OR", "NOT"}
            if prev_is_operand and next_is_operand:
                i += 1
                continue

        if upper == "NOT":
            # NOT <next_word>
            if i + 1 < len(raw_tokens):
                i += 1
                term = tokenise(raw_tokens[i])
                if term:
                    pq.not_terms.extend(term)
            i += 1
            continue

        if upper == "OR":
            # OR <next_word>
            if i + 1 < len(raw_tokens):
                i += 1
                term = tokenise(raw_tokens[i])
                if term:
                    pq.or_terms.extend(term)
            i += 1
            continue

        # Default: AND term
        term = tokenise(token)
        if term:
            pq.and_terms.extend(term)
        i += 1

    return pq


# ---------------------------------------------------------------------------
# Phrase matching helper
# ---------------------------------------------------------------------------

def _pages_with_phrase(phrase_tokens: list[str], index: dict) -> set[str]:
    """
    Return the set of page URLs where *phrase_tokens* occur consecutively.

    Algorithm: intersect the posting lists for all phrase tokens (standard AND),
    then for each candidate page verify that positions form a consecutive run.
    Position verification is O(f²) per page where f = per-term frequency -
    fast in practice because f is small.

    This is the practical realisation of the positional index technique
    described in lectures: positions stored at index time make phrase retrieval
    possible without rescanning raw text.

    Args:
        phrase_tokens: Ordered list of tokens forming the phrase.
        index:         Inverted index dictionary.

    Returns:
        Set of URLs where the phrase appears.
    """
    if not phrase_tokens:
        return set()

    # Gather posting sets; bail early if any token is absent.
    page_sets: list[set[str]] = []
    for token in phrase_tokens:
        entries = index.get(token)
        if entries is None:
            return set()
        page_sets.append(set(entries.keys()))

    candidate_pages = page_sets[0].intersection(*page_sets[1:])
    if not candidate_pages:
        return set()

    matching: set[str] = set()
    first_token = phrase_tokens[0]
    rest_tokens = phrase_tokens[1:]

    for url in candidate_pages:
        # Positions of the first token on this page.
        start_positions: list[int] = index[first_token][url]["positions"]
        for start in start_positions:
            # Check if every subsequent token appears at start+offset.
            if all(
                (start + offset + 1) in index[rest_tokens[offset]][url]["positions"]
                for offset in range(len(rest_tokens))
            ):
                matching.add(url)
                break   # confirmed for this page; move on

    return matching


# ---------------------------------------------------------------------------
# Advanced multi-mode query execution
# ---------------------------------------------------------------------------

def execute_query(pq: ParsedQuery, index: dict) -> list[dict[str, Any]]:
    """
    Execute a :class:`ParsedQuery` against *index* and return ranked results.

    Execution plan
    --------------
    1. **Phrase pages** - for each quoted phrase, find pages via positional
       matching and intersect across all phrases.
    2. **AND pages**    - intersect posting lists for all mandatory terms.
    3. **Combine**      - intersect phrase pages and AND pages.
    4. **OR pages**     - union posting lists for optional terms; union with
       step 3 (so OR terms *expand* the result set rather than restrict it).
    5. **NOT filter**   - remove any page that contains a NOT term.
    6. **Score & rank** - sum TF-IDF across all matched terms; sort descending.

    Args:
        pq:    Parsed query structure.
        index: Inverted index dictionary.

    Returns:
        Ordered list of result dicts (same schema as :func:`find_query`).
        Empty list if nothing matches.
    """
    # ---- Phrase pages -------------------------------------------------------
    if pq.phrases:
        phrase_pages: set[str] | None = None
        for phrase_tokens in pq.phrases:
            pages = _pages_with_phrase(phrase_tokens, index)
            phrase_pages = pages if phrase_pages is None else phrase_pages & pages
        if not phrase_pages:
            return []
    else:
        phrase_pages = None

    # ---- AND pages ----------------------------------------------------------
    if pq.and_terms:
        and_pages: set[str] | None = None
        for token in pq.and_terms:
            entries = index.get(token)
            if entries is None:
                return []   # AND semantics: missing token -> empty
            token_pages = set(entries.keys())
            and_pages = token_pages if and_pages is None else and_pages & token_pages
        if not and_pages:
            return []
    else:
        and_pages = None

    # ---- Combine phrase AND and_pages ---------------------------------------
    if phrase_pages is not None and and_pages is not None:
        core_pages = phrase_pages & and_pages
    elif phrase_pages is not None:
        core_pages = phrase_pages
    elif and_pages is not None:
        core_pages = and_pages
    else:
        core_pages = set()

    # ---- OR pages (union with core) -----------------------------------------
    if pq.or_terms:
        or_pages: set[str] = set()
        for token in pq.or_terms:
            entries = index.get(token)
            if entries:
                or_pages |= set(entries.keys())
        candidate_pages = core_pages | or_pages
    else:
        candidate_pages = core_pages

    if not candidate_pages:
        return []

    # ---- NOT filter ---------------------------------------------------------
    for token in pq.not_terms:
        entries = index.get(token)
        if entries:
            candidate_pages -= set(entries.keys())

    if not candidate_pages:
        return []

    # ---- Score and rank -----------------------------------------------------
    # Collect all terms that contribute to TF-IDF scoring.
    scoring_tokens = list({*pq.and_terms, *pq.or_terms,
                           *(t for phrase in pq.phrases for t in phrase)})

    results: list[dict] = []
    for url in candidate_pages:
        total_freq = 0
        total_tfidf = 0.0
        page_rank = 0.0
        matched: list[str] = []

        for token in scoring_tokens:
            entries = index.get(token, {})
            if url in entries:
                stats = entries[url]
                total_freq += stats["frequency"]
                total_tfidf += stats.get("tfidf", 0.0)
                page_rank = stats.get("pagerank", page_rank)
                matched.append(token)

        results.append(
            {
                "url": url,
                "matched_words": matched,
                "total_freq": total_freq,
                "total_tfidf": round(total_tfidf, 4),
                "pagerank": round(page_rank, 6),
            }
        )

    results.sort(key=lambda r: (r["total_tfidf"], r["pagerank"], r["total_freq"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Public query entry point (backward-compatible)
# ---------------------------------------------------------------------------

def find_query(query: str, index: dict) -> list[dict[str, Any]]:
    """
    Search the index using the advanced query parser.

    This function is the public entry point and is fully backward-compatible:
    a plain space-separated query is treated as an AND query, exactly as before.
    Phrase, OR, and NOT operators are now also supported.

    When an AND-only query returns zero results, spelling suggestions are
    printed automatically so the caller benefits without extra effort.

    Supported syntax examples::

        find love                    # single-term AND
        find love life               # multi-term AND
        find love OR life            # OR - union of results
        find love NOT hate           # AND with NOT exclusion
        find "true love"             # phrase search
        find "true love" NOT hate    # phrase + NOT

    Args:
        query: Query string (plain AND, OR, NOT, or "phrase" operators allowed).
        index: Inverted index dictionary.

    Returns:
        Ordered list of result dicts, each containing:
            ``url``           - page URL
            ``matched_words`` - list of matched query terms
            ``total_freq``    - sum of per-term frequencies on this page
            ``total_tfidf``   - sum of per-term TF-IDF scores on this page
        Empty list if no pages match or query is empty.
    """
    pq = parse_query(query)
    if not pq.and_terms and not pq.or_terms and not pq.phrases:
        logger.debug("Empty query after parsing.")
        return []

    results = execute_query(pq, index)

    # Auto-suggest if a pure AND query returned nothing.
    if not results and not pq.or_terms and not pq.phrases and not pq.not_terms:
        _print_auto_suggestions(query, index)

    return results


def _print_auto_suggestions(query: str, index: dict) -> None:
    """Print spelling suggestions when a query returns no results."""
    suggestions = suggest_for_query(query, index)
    if suggestions:
        print(f"\n  Did you mean: {' | '.join(suggestions)}")
        print("  (use `find <suggestion>` to try one)\n")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_query_results(query: str, results: list[dict]) -> None:
    """
    Pretty-print the results returned by :func:`find_query`.

    Args:
        query:   The original query string.
        results: List of result dicts from find_query.
    """
    if not results:
        print(f"No pages found for query: '{query}'")
        return

    pq = parse_query(query)
    term_summary_parts: list[str] = []
    if pq.and_terms:
        term_summary_parts.append("AND: " + ", ".join(pq.and_terms))
    if pq.or_terms:
        term_summary_parts.append("OR: " + ", ".join(pq.or_terms))
    if pq.not_terms:
        term_summary_parts.append("NOT: " + ", ".join(pq.not_terms))
    if pq.phrases:
        term_summary_parts.append(
            "phrases: " + "; ".join('"' + " ".join(p) + '"' for p in pq.phrases)
        )
    term_summary = " | ".join(term_summary_parts) if term_summary_parts else ", ".join(tokenise(query))

    print(f"\nResults for '{query}' ({term_summary}):\n")
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result['url']}")
        print(f"       matched : {', '.join(result['matched_words'])}")
        print(f"       freq    : {result['total_freq']}")
        print(f"       tfidf   : {result['total_tfidf']}")
        if "pagerank" in result:
            print(f"       pr      : {result['pagerank']}")
    print()
