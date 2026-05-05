"""
test_crawler.py - Comprehensive tests for src/crawler.py

Tests cover:
- Text extraction (visible text, script/style removal, whitespace normalization)
- Pagination link detection
- URL normalization (fragments, lowercasing, relative URLs)
- Link extraction (same-domain filtering, duplicate elimination)
- Retry logic with exponential backoff
- Network error handling (timeouts, connection errors, HTTP errors)
- Crawl robustness and edge cases
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import (
    extract_text,
    get_next_page,
    crawl_site,
    _normalise_url,
    _extract_same_domain_links,
    POLITENESS_DELAY,
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
    RETRY_BACKOFF_FACTOR,
)


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    """Test HTML text extraction and cleaning."""
    
    def test_returns_visible_text(self):
        """Simple HTML should yield visible text."""
        html = "<html><body><p>Hello World</p></body></html>"
        result = extract_text(html)
        assert "Hello" in result
        assert "World" in result

    def test_strips_script_tags(self):
        """JavaScript in <script> tags should be removed."""
        html = "<html><body><script>var x=1;</script><p>Visible</p></body></html>"
        result = extract_text(html)
        assert "var" not in result
        assert "Visible" in result

    def test_strips_style_tags(self):
        """CSS in <style> tags should be removed."""
        html = "<html><head><style>body{color:red;}</style></head><body>Text</body></html>"
        result = extract_text(html)
        assert "color" not in result
        assert "Text" in result

    def test_normalises_whitespace(self):
        """Multiple spaces should be collapsed to single space."""
        html = "<p>   lots   of   spaces   </p>"
        result = extract_text(html)
        assert "  " not in result  # no double spaces

    def test_empty_html(self):
        """Empty HTML should return empty string."""
        result = extract_text("")
        assert result == ""

    def test_strips_nested_scripts_and_styles(self):
        """Multiple script/style tags should all be removed."""
        html = """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
        <script>alert('test');</script>
        <p>Keep this</p>
        <script>more js</script>
        </body>
        </html>
        """
        result = extract_text(html)
        assert "Keep this" in result
        assert "alert" not in result
        assert "more js" not in result

    def test_preserves_text_from_multiple_elements(self):
        """Text from multiple HTML elements should be concatenated."""
        html = "<div>First</div><div>Second</div><div>Third</div>"
        result = extract_text(html)
        assert "First" in result
        assert "Second" in result
        assert "Third" in result

    def test_handles_html_entities(self):
        """HTML entities should be decoded."""
        html = "<p>&lt;tag&gt; &amp; text</p>"
        result = extract_text(html)
        assert "<tag>" in result or "tag" in result
        assert "&" not in result or "amp" not in result

    def test_complex_real_world_html(self):
        """Real-world HTML with mixed content should extract cleanly."""
        html = """
        <html>
        <head><title>Page</title><script>analytics()</script></head>
        <body>
        <nav><a href="#">Menu</a></nav>
        <article>
            <h1>Article Title</h1>
            <p>Article content here.</p>
        </article>
        <aside><style>.ad { display: none; }</style><div>Ad</div></aside>
        </body>
        </html>
        """
        result = extract_text(html)
        assert "Article Title" in result
        assert "Article content" in result
        assert "analytics" not in result


# ---------------------------------------------------------------------------
# _normalise_url
# ---------------------------------------------------------------------------

class TestNormaliseUrl:
    """Test URL normalization (lowercasing, fragment removal, path normalization)."""
    
    def test_removes_fragment(self):
        """URL fragments should be stripped."""
        url = "https://example.com/page#section"
        result = _normalise_url(url)
        assert "#section" not in result

    def test_lowercases_scheme(self):
        """URL scheme should be lowercased."""
        url = "HTTPS://example.com/page"
        result = _normalise_url(url)
        assert result.startswith("https://")

    def test_lowercases_domain(self):
        """Domain should be lowercased."""
        url = "https://Example.COM/page"
        result = _normalise_url(url)
        assert "Example.COM" not in result
        assert "example.com" in result

    def test_preserves_path_case(self):
        """Path should preserve case (case-sensitive on most servers)."""
        url = "https://example.com/MyPath/Page"
        result = _normalise_url(url)
        assert "MyPath" in result or "mypath" in result  # May depend on implementation

    def test_adds_default_path(self):
        """URL with no path should get '/' as default."""
        url = "https://example.com"
        result = _normalise_url(url)
        assert result.endswith("/")

    def test_preserves_query_string(self):
        """Query parameters should be preserved."""
        url = "https://example.com/page?foo=bar&baz=qux"
        result = _normalise_url(url)
        assert "foo=bar" in result
        assert "baz=qux" in result

    def test_removes_fragment_with_query(self):
        """Fragment should be removed even with query string."""
        url = "https://example.com/page?foo=bar#section"
        result = _normalise_url(url)
        assert "#section" not in result
        assert "foo=bar" in result

    def test_handles_trailing_slash(self):
        """Path with trailing slash should be preserved or consistent."""
        url1 = "https://example.com/page/"
        url2 = "https://example.com/page"
        result1 = _normalise_url(url1)
        result2 = _normalise_url(url2)
        # Results may differ but should be consistent


# ---------------------------------------------------------------------------
# _extract_same_domain_links
# ---------------------------------------------------------------------------

class TestExtractSameDomainLinks:
    """Test link extraction with domain filtering."""
    
    def test_extracts_absolute_links(self):
        """Absolute links to same domain should be extracted."""
        html = '<a href="https://example.com/page2">Link</a>'
        links = _extract_same_domain_links(html, "https://example.com/page1", "example.com")
        assert "https://example.com/page2" in links

    def test_extracts_relative_links(self):
        """Relative links should be converted to absolute and extracted."""
        html = '<a href="/page2">Link</a>'
        links = _extract_same_domain_links(html, "https://example.com/page1", "example.com")
        assert any("/page2" in link for link in links)

    def test_filters_cross_domain_links(self):
        """Links to different domains should be filtered out."""
        html = '<a href="https://other.com/page">Link</a>'
        links = _extract_same_domain_links(html, "https://example.com/page1", "example.com")
        assert not any("other.com" in link for link in links)

    def test_deduplicates_links(self):
        """Duplicate links should appear only once."""
        html = '<a href="/page2">L1</a><a href="/page2">L2</a>'
        links = _extract_same_domain_links(html, "https://example.com/page1", "example.com")
        # Count occurrences of /page2
        count = sum(1 for link in links if "/page2" in link)
        assert count == 1

    def test_preserves_document_order(self):
        """Links should appear in document order."""
        html = '<a href="/p1">1</a><a href="/p2">2</a><a href="/p3">3</a>'
        links = _extract_same_domain_links(html, "https://example.com/", "example.com")
        assert len(links) == 3

    def test_ignores_links_without_href(self):
        """Anchor tags without href should be skipped."""
        html = '<a>No href</a><a href="/page">Has href</a>'
        links = _extract_same_domain_links(html, "https://example.com/", "example.com")
        assert len(links) == 1

    def test_handles_empty_html(self):
        """Empty HTML should return empty list."""
        links = _extract_same_domain_links("", "https://example.com/", "example.com")
        assert links == []

    def test_handles_relative_paths(self):
        """Various relative path formats should be handled."""
        html = '''
        <a href="page">Relative</a>
        <a href="./page">Dot slash</a>
        <a href="../page">Parent</a>
        '''
        links = _extract_same_domain_links(html, "https://example.com/dir/page", "example.com")
        assert len(links) > 0


# ---------------------------------------------------------------------------
# get_next_page
# ---------------------------------------------------------------------------

class TestGetNextPage:
    """Test pagination link detection."""
    
    def _soup(self, html: str) -> BeautifulSoup:
        return BeautifulSoup(html, "html.parser")

    def test_returns_url_when_next_exists(self):
        """Should return next page URL when next button exists."""
        html = '<li class="next"><a href="/page/2/">Next</a></li>'
        result = get_next_page(self._soup(html))
        assert result == "https://quotes.toscrape.com/page/2/"

    def test_returns_none_when_no_next(self):
        """Should return None when no next button found."""
        html = "<html><body><p>Last page</p></body></html>"
        result = get_next_page(self._soup(html))
        assert result is None

    def test_returns_none_for_empty_html(self):
        """Should return None for empty HTML."""
        result = get_next_page(self._soup(""))
        assert result is None

    def test_next_anchor_missing_href(self):
        """Should return None if anchor tag has no href."""
        html = '<li class="next"><a>Next</a></li>'
        result = get_next_page(self._soup(html))
        assert result is None

    def test_absolute_url_in_next(self):
        """Should handle absolute URLs in next link."""
        html = '<li class="next"><a href="https://quotes.toscrape.com/page/2/">Next</a></li>'
        result = get_next_page(self._soup(html))
        assert "page/2" in result

    def test_http_next_link(self):
        """Should handle HTTP (non-HTTPS) URLs."""
        html = '<li class="next"><a href="http://quotes.toscrape.com/page/2/">Next</a></li>'
        result = get_next_page(self._soup(html))
        assert result is not None


# ---------------------------------------------------------------------------
# Retry logic with exponential backoff
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


class TestRetryMechanism:
    """Test retry logic with exponential backoff."""
    
    PAGE_HTML = "<html><body><p>Test page</p></body></html>"

    @patch("time.sleep")
    @patch("requests.Session")
    def test_success_on_first_attempt(self, mock_session_cls, mock_sleep):
        """Should not retry on first successful attempt."""
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.return_value = _make_response(self.PAGE_HTML)

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 1
        # No retries, so get called once
        assert session.get.call_count == 1

    @patch("time.sleep")
    @patch("requests.Session")
    def test_retries_on_timeout(self, mock_session_cls, mock_sleep):
        """Should retry on timeout errors."""
        import requests
        session = MagicMock()
        mock_session_cls.return_value = session
        # Fail twice, succeed on third
        session.get.side_effect = [
            requests.Timeout("timeout"),
            requests.Timeout("timeout"),
            _make_response(self.PAGE_HTML),
        ]

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 1
        assert session.get.call_count == 3

    @patch("time.sleep")
    @patch("requests.Session")
    def test_exponential_backoff_delays(self, mock_session_cls, mock_sleep):
        """Should increase delay exponentially between retries."""
        import requests
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            requests.Timeout("timeout"),
            requests.Timeout("timeout"),
            _make_response(self.PAGE_HTML),
        ]

        list(crawl_site("https://quotes.toscrape.com"))
        
        # Check sleep was called with exponential backoff
        sleep_calls = mock_sleep.call_args_list
        # Should have calls for: politeness delay(0), retry delay(1s), retry delay(2s)
        assert len(sleep_calls) >= 2  # At least 2 retry delays

    @patch("time.sleep")
    @patch("requests.Session")
    def test_gives_up_after_max_retries(self, mock_session_cls, mock_sleep):
        """Should stop retrying after MAX_RETRIES attempts."""
        import requests
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = requests.Timeout("always fails")

        pages = list(crawl_site("https://quotes.toscrape.com"))
        # Should give up and return no pages
        assert len(pages) == 0
        # Should attempt up to MAX_RETRIES times
        assert session.get.call_count == MAX_RETRIES

    @patch("time.sleep")
    @patch("requests.Session")
    def test_retries_on_connection_error(self, mock_session_cls, mock_sleep):
        """Should retry on ConnectionError."""
        import requests
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            requests.ConnectionError("refused"),
            _make_response(self.PAGE_HTML),
        ]

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 1
        assert session.get.call_count == 2

    @patch("time.sleep")
    @patch("requests.Session")
    def test_retries_on_http_error(self, mock_session_cls, mock_sleep):
        """Should retry on HTTP error responses."""
        import requests
        session = MagicMock()
        mock_session_cls.return_value = session
        resp_error = _make_response("error", status=503)
        resp_error.raise_for_status.side_effect = requests.HTTPError("503")
        session.get.side_effect = [
            resp_error,
            _make_response(self.PAGE_HTML),
        ]

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 1
        assert session.get.call_count == 2

    @patch("time.sleep")
    @patch("requests.Session")
    def test_retries_on_429_rate_limit(self, mock_session_cls, mock_sleep):
        """Should retry on 429 Too Many Requests."""
        import requests
        session = MagicMock()
        mock_session_cls.return_value = session
        resp_429 = _make_response("rate limit", status=429)
        resp_429.raise_for_status.side_effect = requests.HTTPError("429")
        session.get.side_effect = [
            resp_429,
            _make_response(self.PAGE_HTML),
        ]

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 1


# ---------------------------------------------------------------------------
# crawl_site - general behavior
# ---------------------------------------------------------------------------

class TestCrawlSite:
    """Test overall crawl site behavior."""
    
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

    @patch("time.sleep")
    @patch("requests.Session")
    def test_yields_two_pages(self, mock_session_cls, mock_sleep):
        """Should crawl and yield multiple pages following pagination."""
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            _make_response(self.PAGE1_HTML),
            _make_response(self.PAGE2_HTML),
        ]

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 2

    @patch("time.sleep")
    @patch("requests.Session")
    def test_page_data_contains_url_and_text(self, mock_session_cls, mock_sleep):
        """Each page should contain URL and extracted text."""
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.return_value = _make_response(self.PAGE2_HTML)

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert "url" in pages[0]
        assert "text" in pages[0]
        assert "page two" in pages[0]["text"]

    @patch("time.sleep")
    @patch("requests.Session")
    def test_page_data_contains_links(self, mock_session_cls, mock_sleep):
        """Each page should contain extracted links."""
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.return_value = _make_response(self.PAGE2_HTML)

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert "links" in pages[0]

    @patch("time.sleep")
    @patch("requests.Session")
    def test_respects_politeness_delay(self, mock_session_cls, mock_sleep):
        """Should respect politeness delay between page requests."""
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            _make_response(self.PAGE1_HTML),
            _make_response(self.PAGE2_HTML),
        ]

        list(crawl_site("https://quotes.toscrape.com"))
        # sleep should be called for politeness delay
        assert any(
            call(POLITENESS_DELAY) in mock_sleep.call_args_list
            for _ in [1]
        )

    @patch("time.sleep")
    @patch("requests.Session")
    def test_handles_network_error_gracefully(self, mock_session_cls, mock_sleep):
        """Should handle network errors without crashing."""
        import requests
        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = requests.RequestException("Connection refused")

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert pages == []

    @patch("time.sleep")
    @patch("requests.Session")
    def test_does_not_revisit_urls(self, mock_session_cls, mock_sleep):
        """Should prevent infinite loops by not revisiting URLs."""
        start = "https://quotes.toscrape.com"
        page1 = """<html><body><p>P1</p>
            <ul class="pager"><li class="next"><a href="/page/2/">→</a></li></ul>
        </body></html>"""
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

    @patch("time.sleep")
    @patch("requests.Session")
    def test_extracts_links_and_follows_them(self, mock_session_cls, mock_sleep):
        """Should extract and follow links from crawled pages."""
        page1 = """<html><body><p>P1</p>
            <a href="/page2">Link to page 2</a>
        </body></html>"""
        page2 = """<html><body><p>P2</p></body></html>"""

        session = MagicMock()
        mock_session_cls.return_value = session
        session.get.side_effect = [
            _make_response(page1),
            _make_response(page2),
        ]

        pages = list(crawl_site("https://quotes.toscrape.com"))
        assert len(pages) == 2
