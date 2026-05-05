"""
crawler.py - Web crawler for quotes.toscrape.com

Handles HTTP requests, pagination, politeness delay, and text extraction.
"""

from __future__ import annotations

import time
import logging
from collections import deque
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://quotes.toscrape.com"
POLITENESS_DELAY = 6


def _normalise_url(url: str) -> str:
    """Return a canonical same-site URL form (lower host, no fragment)."""
    without_fragment, _ = urldefrag(url)
    parts = urlsplit(without_fragment)
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _extract_same_domain_links(html: str, current_url: str, base_netloc: str) -> list[str]:
    """Extract unique same-domain links from a page, preserving document order."""
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        absolute = _normalise_url(urljoin(current_url, anchor["href"]))
        if urlsplit(absolute).netloc != base_netloc:
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)

    return links


def extract_text(html: str) -> str:
    """Extract visible text content from HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def get_next_page(soup) -> str | None:
    """Find URL of next pagination page if it exists."""
    next_btn = soup.find("li", class_="next")
    if next_btn:
        anchor = next_btn.find("a")
        href = anchor.get("href") if anchor else None
        if href:
            if href.startswith("http://") or href.startswith("https://"):
                return href
            return BASE_URL + href
    return None


def crawl_site(start_url: str = BASE_URL):
    """Crawl all pages from start_url yielding page data with URL, text, and links."""
    logger = logging.getLogger(__name__)
    session = requests.Session()
    session.headers.update({"User-Agent": "SearchEngineCoursework/1.0"})

    start = _normalise_url(start_url)
    base_netloc = urlsplit(start).netloc
    to_visit = deque([start])
    queued: set[str] = {start}
    visited: set[str] = set()
    page_number = 1
    first_request = True

    while to_visit:
        current_url = to_visit.popleft()
        queued.discard(current_url)
        if current_url in visited:
            continue

        if not first_request:
            time.sleep(POLITENESS_DELAY)
        first_request = False

        visited.add(current_url)
        logger.info("Crawling page %d: %s", page_number, current_url)

        try:
            response = session.get(current_url, timeout=15)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Request failed for %s: %s", current_url, exc)
            page_number += 1
            continue

        text = extract_text(response.text)
        links = _extract_same_domain_links(response.text, current_url, base_netloc)
        yield {"url": current_url, "text": text, "links": links}

        for link in links:
            if link not in visited and link not in queued:
                to_visit.append(link)
                queued.add(link)
        page_number += 1
