"""
test_crawler.py - Tests for src/crawler.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import extract_text, get_next_page, crawl_site, POLITENESS_DELAY


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_returns_visible_text(self):
        html = "<html><body><p>Hello World</p></body></html>"
        result = extract_text(html)
        assert "Hello" in result
        assert "World" in result

    def test_strips_script_tags(self):
        html = "<html><body><script>var x=1;</script><p>Visible</p></body></html>"
        result = extract_text(html)
        assert "var" not in result
        assert "Visible" in result

    def test_strips_style_tags(self):
        html = "<html><head><style>body{color:red;}</style></head><body>Text</body></html>"
        result = extract_text(html)
        assert "color" not in result
        assert "Text" in result

    def test_normalises_whitespace(self):
        html = "<p>   lots   of   spaces   </p>"
        result = extract_text(html)
        assert "  " not in result  # no double spaces

    def test_empty_html(self):
        result = extract_text("")
        assert result == ""


# ---------------------------------------------------------------------------
# get_next_page
# ---------------------------------------------------------------------------

class TestGetNextPage:
    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_returns_url_when_next_exists(self):
        html = '<li class="next"><a href="/page/2/">Next</a></li>'
        result = get_next_page(self._soup(html))
        assert result == "https://quotes.toscrape.com/page/2/"

    def test_returns_none_when_no_next(self):
        html = "<html><body><p>Last page</p></body></html>"
        result = get_next_page(self._soup(html))
        assert result is None

    def test_returns_none_for_empty_html(self):
        result = get_next_page(self._soup(""))
        assert result is None

    def test_next_anchor_missing_href(self):
        html = '<li class="next"><a>Next</a></li>'
        result = get_next_page(self._soup(html))
        assert result is None


# ---------------------------------------------------------------------------
# crawl_site
# ---------------------------------------------------------------------------

def _make_response(text: str, status: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return resp


class TestCrawlSite:
    PAGE1_HTML = """
    <html><body>
      <p>Quote from page one.</p>
      <ul class="pager"><li class="next"><a href="/page/2/">Next</a></li></ul>
    </body></html>
    """
    PAGE2_HTML = """
    <html><body>
      <p>Quote from page two.</p>
    </body></html>
    """

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_yields_two_pages(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            _make_response(self.PAGE1_HTML),
            _make_response(self.PAGE2_HTML),
        ]

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 2

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_page_data_contains_url_and_text(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.return_value = _make_response(self.PAGE2_HTML)

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert "url" in pages[0]
        assert "text" in pages[0]
        assert "page two" in pages[0]["text"]

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_respects_politeness_delay(self, mock_session_cls, mock_sleep):
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            _make_response(self.PAGE1_HTML),
            _make_response(self.PAGE2_HTML),
        ]

        list(crawl_site("https://quotes.toscrape.com"))
        # sleep should be called once (between page 1 and page 2)
        mock_sleep.assert_called_once_with(POLITENESS_DELAY)

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_handles_network_error_gracefully(self, mock_session_cls, mock_sleep):
        import requests as req_lib
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = req_lib.RequestException("Connection refused")

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert pages == []

    @patch("crawler.time.sleep")
    @patch("crawler.requests.Session")
    def test_does_not_revisit_urls(self, mock_session_cls, mock_sleep):
        """Ensure visited-set prevents infinite loops.

        Page 2's "next" link resolves to the start URL, which is already in
        the visited set, so the crawler must stop after exactly 2 pages.
        """
        start = "https://quotes.toscrape.com"
        # /page/2/ is the next link from page 1
        page1 = """<html><body><p>P1</p>
            <ul class="pager"><li class="next"><a href="/page/2/">→</a></li></ul>
        </body></html>"""
        # page 2 links back to the root - same as start URL
        page2 = f"""<html><body><p>P2</p>
            <ul class="pager"><li class="next">
              <a href="{start}">→</a>
            </li></ul>
        </body></html>"""

        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            _make_response(page1),
            _make_response(page2),
        ]

        pages = list(crawl_site(start))
        assert len(pages) == 2
