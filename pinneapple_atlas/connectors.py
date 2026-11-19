"""Read-only, bounded public metadata connectors. No network calls at import time."""
import json
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen


class PublicMetadataClient:
    def __init__(self, *, timeout=12, user_agent="PINNeAPPle-Atlas/0.1 (research metadata)", fetch=None):
        self.timeout = timeout
        self.user_agent = user_agent
        self._fetch = fetch or self._http_get

    def _http_get(self, url):
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def openalex(self, query, limit=10):
        limit = _limit(limit)
        url = "https://api.openalex.org/works?" + urlencode({"search": query, "per-page": limit})
        payload = self._fetch(url)
        return [{"id": w.get("id"), "title": w.get("title"), "doi": w.get("doi"),
                 "year": w.get("publication_year"), "source": "openalex",
                 "url": w.get("doi") or w.get("id")} for w in payload.get("results", [])]

    def crossref(self, query, limit=10):
        limit = _limit(limit)
        url = "https://api.crossref.org/works?" + urlencode({"query": query, "rows": limit})
        payload = self._fetch(url)
        return [{"id": w.get("DOI"), "title": (w.get("title") or [""])[0],
                 "doi": ("https://doi.org/" + w["DOI"]) if w.get("DOI") else None,
                 "source": "crossref", "url": w.get("URL")} for w in payload.get("message", {}).get("items", [])]

    def github_repositories(self, query, limit=10):
        limit = _limit(limit)
        url = "https://api.github.com/search/repositories?" + urlencode({"q": query, "per_page": limit})
        payload = self._fetch(url)
        return [{"id": str(w.get("id")), "title": w.get("full_name"), "url": w.get("html_url"),
                 "source": "github", "description": w.get("description"),
                 "revision": w.get("pushed_at")} for w in payload.get("items", [])]

    def github_inventory(self, owner="PINNeAPPle-Labs", repo="PINNeAPPle", ref="main"):
        # Inventory only: file existence is NOT proof of an implemented capability.
        url = ("https://api.github.com/repos/" + quote(owner, safe="") + "/" +
               quote(repo, safe="") + "/git/trees/" + quote(ref, safe="") + "?recursive=1")
        payload = self._fetch(url)
        return {"repository": f"{owner}/{repo}", "ref": ref,
                "truncated": bool(payload.get("truncated", False)),
                "paths": sorted(item["path"] for item in payload.get("tree", [])
                                if item.get("type") == "blob" and item.get("path"))}


def _limit(n):
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    return n
