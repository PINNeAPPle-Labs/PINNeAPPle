"""Additional bounded public-source metadata adapters (no import-time I/O)."""
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _get(url, accept):
    with urlopen(Request(url, headers={"User-Agent": "PINNeAPPle-Atlas/0.1", "Accept": accept}), timeout=12) as response:
        return response.read()


def _limit(limit):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return limit


class AdditionalSources:
    def __init__(self, fetch=None):
        self.fetch = fetch or _get

    def arxiv(self, query, limit=10):
        url = "https://export.arxiv.org/api/query?" + urlencode({"search_query": "all:" + query, "start": 0, "max_results": _limit(limit)})
        root = ET.fromstring(self.fetch(url, "application/atom+xml"))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        return [{"source": "arxiv", "id": e.findtext("a:id", default="", namespaces=ns),
                 "title": " ".join(e.findtext("a:title", default="", namespaces=ns).split()),
                 "url": e.findtext("a:id", default="", namespaces=ns)}
                for e in root.findall("a:entry", ns)]

    def huggingface_datasets(self, query, limit=10):
        url = "https://huggingface.co/api/datasets?" + urlencode({"search": query, "limit": _limit(limit)})
        data = json.loads(self.fetch(url, "application/json"))
        return [{"source": "huggingface", "id": d["id"], "title": d["id"],
                 "url": "https://huggingface.co/datasets/" + d["id"]} for d in data if d.get("id")]

    def zenodo(self, query, limit=10):
        url = "https://zenodo.org/api/records?" + urlencode({"q": query, "size": _limit(limit)})
        data = json.loads(self.fetch(url, "application/json"))
        return [{"source": "zenodo", "id": str(d["id"]),
                 "title": d.get("metadata", {}).get("title") or str(d["id"]),
                 "url": d.get("links", {}).get("html") or "https://zenodo.org/records/" + str(d["id"]),
                 "doi": d.get("doi")} for d in data.get("hits", {}).get("hits", []) if d.get("id")]
