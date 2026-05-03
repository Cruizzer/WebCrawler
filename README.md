# SearchEngine - Command-Line Inverted Index Search Tool

A fully-featured command-line search engine that crawls [quotes.toscrape.com](https://quotes.toscrape.com), builds a persistent inverted index, and lets you search it interactively.

---

## Project Overview

### Purpose
This tool demonstrates how a basic web search engine works under the hood - crawling pages, tokenising text, building an inverted index, and ranking search results using TF-IDF scoring.

### Features
- **Web crawling** with pagination support and a 6-second politeness delay
- **Inverted index** storing per-page word frequency and token positions
- **TF-IDF ranking** with PageRank-based tie-breaking for more relevant search results
- **Stop-word filtering** to reduce noise in the index
- **Advanced query search** - supports AND, OR, NOT, and quoted phrase queries
- **Suggestions** - prefix completion and spelling suggestions via `suggest`
- **Persistent storage** - index saved as JSON and reloaded across sessions
- **Interactive CLI** with `build`, `load`, `print`, `find`, `suggest`, and `help` commands

### Architecture

```
src/
├── crawler.py   HTTP fetching, pagination, HTML text extraction
├── indexer.py   Tokenisation, index building, TF-IDF, save/load
├── search.py    Query parsing, AND search, ranking, display
└── main.py      Interactive CLI loop
```

---

## Installation

```bash
git clone <your-repo-url>
cd search-engine
pip install -r requirements.txt
```

---

## Usage

```bash
python src/main.py
```

This opens an interactive prompt. Type one command per line.

### Commands

#### `build`
Crawl the target website, build the inverted index, and save it to `data/index.json`.

```
> build
Starting crawl - this may take a few minutes (6 s politeness delay)...
Crawled 10 page(s). Building index…
Index built and saved to 'data/index.json' (1842 unique tokens).
```

Use this when you want to rebuild the index from scratch after changing the crawler or indexer.

#### `load`
Load a previously saved index from disk (much faster than rebuilding).

```
> load
Index loaded from 'data/index.json' (1842 unique tokens).
```

Use this after `build` if you want to restart the program without crawling again.

#### `print <word>`
Display all index data for a single word.

```
> print life

Word: life

Found in:
  - https://quotes.toscrape.com/page/1/
      frequency : 3
      positions : [12, 45, 103]
      tfidf     : 2.197
      pagerank  : 0.084213
```

Example usage:

```text
> print life
> print nonsense
```

#### `find <query>`
Search for pages using the query parser. By default, space-separated terms are treated as AND, but you can also use OR, NOT, and quoted phrases.

```
> find good friends

Results for query 'good friends' (terms: good, friends):

  1. https://quotes.toscrape.com/page/3/
       matched : good, friends
       freq    : 5
       tfidf   : 3.841
       pr      : 0.093421
```

Example output can also include PageRank when it helps break ranking ties or surface authoritative pages.

Additional examples:

```text
> find love OR sadness
> find love NOT hate
> find "true love"
> find "true love" NOT hate
```

If a plain AND query returns no results, the program may print spelling suggestions automatically.

#### `suggest <word|query>`
Show prefix completions for a single word, or spelling suggestions for a multi-word query.

```text
> suggest lov
> suggest lvoe
```

Use this when you want help finding the intended search term before running `find`.

#### `help`
Show the list of available commands.

#### `exit` / `quit` / `q`
Quit the programme.

---

## Testing

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_indexer.py -v
```

---

## Dependencies

| Package          | Purpose                        |
|------------------|--------------------------------|
| `requests`       | HTTP requests for crawling     |
| `beautifulsoup4` | HTML parsing                   |
| `pytest`         | Test framework                 |

Install all:

```bash
pip install -r requirements.txt
```

---

## Advanced Features

- **TF-IDF scoring** - words rare across pages are weighted more heavily.
- **Stop-word filtering** - optional via the `remove_stop_words` flag in `src/indexer.py`.
- **Phrase search** - stored token positions let the search module check consecutive word order.
- **Query suggestions** - Levenshtein-based spelling suggestions and prefix completions.
- **Advanced query parsing** - AND, OR, NOT, and quoted phrases are supported in `find`.
- **Graceful error handling** - network failures and missing index files are caught with helpful messages.
- **Politeness delay** - exactly 6 seconds between HTTP requests to respect the target server.
- **PageRank** - computed from the crawl graph using the lecture-style iterative algorithm with a "surprise me" probability and rank-sink handling.

---

## Project Structure

```
search-engine/
│
├── src/
│   ├── crawler.py
│   ├── indexer.py
│   ├── search.py
│   └── main.py
│
├── tests/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   └── test_search.py
│
├── data/
│   └── index.json        ← generated by `build`
│
├── requirements.txt
├── README.md
└── .gitignore
```
