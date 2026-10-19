"""Internet search and page fetching for the agents.

Search providers, first configured wins:

- ``PINNEAPPLE_SEARXNG_URL``: your own SearXNG instance (private, no key; ``docker run -p 8888:8080 searxng/searxng``);
- ``BRAVE_API_KEY``: the Brave Search API;
- ``TAVILY_API_KEY``: the Tavily search API;
- otherwise DuckDuckGo's HTML endpoint (no key; may rate-limit heavy use).

The search query leaves your machine; the model and your files do not.
"""
from __future__ import annotations

import html
import os
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse

__all__ = ["web_search", "fetch_text", "html_to_text"]

_UA = {"User-Agent": "Mozilla/5.0 (PINNeAPPle agent workbench)"}


def _searxng(base: str, q: str, n: int) -> list[dict]:
    import requests

    r = requests.get(base.rstrip("/") + "/search", params={"q": q, "format": "json"}, headers=_UA, timeout=20)
    r.raise_for_status()
    return [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": h.get("content", "")}
            for h in r.json().get("results", [])[:n]]


def _brave(key: str, q: str, n: int) -> list[dict]:
    import requests

    r = requests.get("https://api.search.brave.com/res/v1/web/search", params={"q": q, "count": n},
                     headers={"X-Subscription-Token": key, "Accept": "application/json"}, timeout=20)
    r.raise_for_status()
    return [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": html_to_text(h.get("description", ""))}
            for h in r.json().get("web", {}).get("results", [])[:n]]


def _tavily(key: str, q: str, n: int) -> list[dict]:
    import requests

    r = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": q, "max_results": n},
                      timeout=30)
    r.raise_for_status()
    return [{"title": h.get("title", ""), "url": h.get("url", ""), "snippet": h.get("content", "")}
            for h in r.json().get("results", [])[:n]]


def _ddg_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    q = parse_qs(urlparse(href).query)
    return unquote(q["uddg"][0]) if "uddg" in q else href


def _duckduckgo(q: str, n: int) -> list[dict]:
    import requests

    r = requests.post("https://html.duckduckgo.com/html/", data={"q": q}, headers=_UA, timeout=20)
    r.raise_for_status()
    return parse_duckduckgo(r.text)[:n]


def parse_duckduckgo(page: str) -> list[dict]:
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
                         page, flags=re.S):
        snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', m.group(3), flags=re.S)
        out.append({"title": html_to_text(m.group(2)), "url": _ddg_url(html.unescape(m.group(1))),
                    "snippet": html_to_text(snip.group(1)) if snip else ""})
    return out


def web_search(query: str, max_results: int = 8) -> list[dict]:
    """``[{title, url, snippet}]`` from the first configured provider."""
    n = max(1, min(int(max_results), 20))
    if os.environ.get("PINNEAPPLE_SEARXNG_URL"):
        return _searxng(os.environ["PINNEAPPLE_SEARXNG_URL"], query, n)
    if os.environ.get("BRAVE_API_KEY"):
        return _brave(os.environ["BRAVE_API_KEY"], query, n)
    if os.environ.get("TAVILY_API_KEY"):
        return _tavily(os.environ["TAVILY_API_KEY"], query, n)
    return _duckduckgo(query, n)


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(page: str) -> str:
    p = _Text()
    p.feed(page or "")
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _raw_github(url: str) -> str:
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else url


def fetch_text(url: str, max_chars: int = 12_000) -> str:
    """The readable text of a page (HTML stripped) or a raw text file; GitHub ``blob`` links are read raw."""
    import requests

    if not re.match(r"https?://", url):
        raise ValueError("only http(s) URLs can be fetched")
    r = requests.get(_raw_github(url), headers=_UA, timeout=30)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    text = html_to_text(r.text) if "html" in ctype else r.text
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [{len(text) - max_chars} more characters]"
    return f"{url}\n\n{text}"
