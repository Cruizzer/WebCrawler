"""main.py - CLI entry point for the command-line search engine."""

import logging
import sys
from pathlib import Path

# Allow sibling imports when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from crawler import crawl_site, BASE_URL
from indexer import build_index, save_index, load_index
from search import (
    find_query,
    print_query_results,
    print_word,
    suggest_prefix,
    suggest_for_query,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_PATH = "data/index.json"
START_URL = BASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Minimal help output printed inline to avoid large multi-line strings.


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_build(index_store: dict) -> dict:
    """Crawl site, build index, save to disk, and return updated index."""
    print("Starting crawl - this may take a few minutes (6 s politeness delay)...")
    pages = list(crawl_site(START_URL))
    if not pages:
        print("Crawl returned no pages. Check your network connection.")
        return index_store

    print(f"Crawled {len(pages)} page(s). Building index…")
    index = build_index(pages)
    save_index(index, INDEX_PATH)
    print(f"Index built and saved to '{INDEX_PATH}' ({len(index)} unique tokens).")
    return index


def cmd_load(index_store: dict) -> dict:
    """Load index from disk and return it."""
    try:
        index = load_index(INDEX_PATH)
        print(f"Index loaded from '{INDEX_PATH}' ({len(index)} unique tokens).")
        return index
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return index_store


def cmd_print(args: list[str], index: dict) -> None:
    """Print index data for a word."""
    if not args:
        print("Usage: print <word>")
        return
    if not index:
        print("No index loaded. Run `build` or `load` first.")
        return
    print_word(args[0], index)


def cmd_suggest(args: list[str], index: dict) -> None:
    """Provide suggestions for a prefix or full query.

    Usage: `suggest <prefix>` for completions or `suggest <query...>` for
    query-level spelling suggestions.
    """
    if not args:
        print("Usage: suggest <prefix|query>")
        return
    if not index:
        print("No index loaded. Run `build` or `load` first.")
        return

    q = " ".join(args)
    # Single token -> prefix completions; otherwise try query suggestions.
    if len(args) == 1:
        hits = suggest_prefix(q, index)
        if hits:
            print("Completions:")
            for h in hits:
                print(f"  - {h}")
        else:
            print("No completions found.")
    else:
        alts = suggest_for_query(q, index)
        if alts:
            print("Query suggestions:")
            for a in alts:
                print(f"  - {a}")
        else:
            print("No suggestions available.")


def cmd_find(args: list[str], index: dict) -> None:
    """Search index for pages containing all query words."""
    if not args:
        print("Usage: find <word> [word …]")
        return
    if not index:
        print("No index loaded. Run `build` or `load` first.")
        return
    query = " ".join(args)
    results = find_query(query, index)
    print_query_results(query, results)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the interactive CLI loop."""
    index: dict = {}

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split()
        command, *args = parts

        normalized = command.lower()

        if normalized == "build":
            index = cmd_build(index)
        elif normalized == "load":
            index = cmd_load(index)
        elif normalized == "print":
            cmd_print(args, index)
        elif normalized == "find":
            cmd_find(args, index)
        elif normalized == "help":
            print("Commands: build, load, print <word>, find <query>, suggest <prefix|query>, help, exit")
        elif normalized == "suggest":
            cmd_suggest(args, index)
        elif normalized in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        else:
            print(f"Unknown command: '{command}'. Type 'help' for options.")


if __name__ == "__main__":
    main()
