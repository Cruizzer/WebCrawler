"""
indexer.py - Inverted index builder, saver, and loader.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "it", "its", "by", "as", "be",
})


def tokenise(text: str, remove_stop_words: bool = False) -> list[str]:
    """Convert text into lowercase alphanumeric tokens."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if remove_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(
    page_data: list[dict],
    remove_stop_words: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Build an inverted index from a list of crawled pages.

    Structure::

        {
            "word": {
                "page_url": {
                    "frequency": 3,
                    "positions": [4, 15, 27]
                }
            }
        }

    Args:
        page_data: List of dicts with keys ``url`` and ``text``.
        remove_stop_words: Strip common English stop words from the index.

    Returns:
        Inverted index dictionary.
    """
    pages = list(page_data)
    index: dict[str, dict[str, Any]] = {}
    total_pages = len(pages)

    page_ranks = compute_pagerank(pages)

    for entry in pages:
        url: str = entry["url"]
        tokens: list[str] = tokenise(entry["text"], remove_stop_words)

        for position, token in enumerate(tokens):
            if token not in index:
                index[token] = {}
            if url not in index[token]:
                index[token][url] = {"frequency": 0, "positions": []}
            index[token][url]["frequency"] += 1
            index[token][url]["positions"].append(position)

    # Annotate each word→page entry with a TF-IDF score for ranking.
    _annotate_tfidf(index, total_pages)
    _annotate_pagerank(index, page_ranks)

    logger.info("Index built: %d unique tokens across %d pages.", len(index), total_pages)
    return index


def compute_pagerank(
    page_data: list[dict],
    damping: float = 0.15,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """
    Compute PageRank values for the crawl graph using the lecture algorithm.

    Pages with no outbound links are treated as rank sinks and assumed to link
    to all pages in the collection.

    Args:
        page_data: List of crawled pages, each optionally containing ``links``.
        damping:   Probability of clicking the "surprise me" button.
        max_iter:  Maximum number of power-iteration steps.
        tolerance: Stop when the total rank change falls below this threshold.

    Returns:
        Mapping of page URL to PageRank score. Scores sum to approximately 1.
    """
    pages = [entry["url"] for entry in page_data]
    if not pages:
        return {}

    page_set = set(pages)
    outgoing: dict[str, list[str]] = {}
    for entry in page_data:
        url = entry["url"]
        links = entry.get("links", [])
        seen: set[str] = set()
        filtered_links: list[str] = []
        for link in links:
            if link in page_set and link not in seen:
                seen.add(link)
                filtered_links.append(link)
        outgoing[url] = filtered_links

    n = len(pages)
    ranks = {url: 1.0 / n for url in pages}

    for _ in range(max_iter):
        sink_mass = sum(ranks[url] for url, links in outgoing.items() if not links)
        next_ranks = {
            url: (damping / n) + ((1 - damping) * sink_mass / n)
            for url in pages
        }

        for source, links in outgoing.items():
            if not links:
                continue
            share = ranks[source] / len(links)
            for target in links:
                next_ranks[target] += (1 - damping) * share

        delta = sum(abs(next_ranks[url] - ranks[url]) for url in pages)
        ranks = next_ranks
        if delta < tolerance:
            break

    total = sum(ranks.values())
    if total:
        ranks = {url: score / total for url, score in ranks.items()}

    return ranks


def _annotate_tfidf(index: dict, total_pages: int) -> None:
    """
    Add a ``tfidf`` score to every (word, page) pair in-place.

    TF  = frequency / total tokens on page  (approximated as frequency here
          since we do not store total page token counts separately).
    IDF = log(N / df) where N = total pages, df = pages containing word.
    """
    for word, pages in index.items():
        df = len(pages)
        idf = math.log((total_pages + 1) / (df + 1)) + 1  # smoothed
        for url, stats in pages.items():
            tf = stats["frequency"]
            stats["tfidf"] = round(tf * idf, 4)


def _annotate_pagerank(index: dict, page_ranks: dict[str, float]) -> None:
    """Add a ``pagerank`` score to each (word, page) pair in-place."""
    for pages in index.values():
        for url, stats in pages.items():
            stats["pagerank"] = round(page_ranks.get(url, 0.0), 6)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_index(index: dict, path: str) -> None:
    """
    Save the inverted index to a JSON file.

    Args:
        index: Inverted index dictionary.
        path:  File path to write (created/overwritten).
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
    logger.info("Index saved to %s (%d bytes).", path, file_path.stat().st_size)


def load_index(path: str) -> dict:
    """
    Load the inverted index from a JSON file.

    Args:
        path: File path of the previously saved index.

    Returns:
        Inverted index dictionary.

    Raises:
        FileNotFoundError: If the index file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"No index file found at '{path}'. Run `build` first."
        )
    with file_path.open("r", encoding="utf-8") as fh:
        index = json.load(fh)
    logger.info("Index loaded from %s (%d tokens).", path, len(index))
    return index
