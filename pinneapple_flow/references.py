"""The curated reference registry (``references.yaml``) and the check every cited reference goes through.

A reference is *registry* when its key is in the curated file, *verified* when a DOI / URL given by someone else (an
LLM, a user) resolved over the network, and *unverified* otherwise. Nothing here invents a citation: an unknown key
stays an unknown key, shown as such.

    ref("raissi2019pinn")["doi"]                     # '10.1016/j.jcp.2018.10.045'
    check({"key": "lu2021deeponet"})["status"]       # 'registry'
    check({"title": "Some paper", "doi": "10.1/x"})  # 'unverified' (offline) or 'verified' / 'not found' (online)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_FILE = Path(__file__).with_name("references.yaml")
KINDS = ("article", "book", "conference", "preprint")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")


_EXTRA: dict[str, dict[str, Any]] = {}          # entries registered by other packages (register_references)


@lru_cache(maxsize=1)
def registry() -> dict[str, dict[str, Any]]:
    import yaml
    data = yaml.safe_load(REGISTRY_FILE.read_text()) or {}
    return {**{k: {"key": k, **v} for k, v in data.items()}, **{k: {"key": k, **v} for k, v in _EXTRA.items()}}


def register_references(entries: dict[str, dict[str, Any]] | str | Path) -> list[str]:
    """Add curated references (a mapping key -> entry, or a YAML file of them) for another package's rules. Each
    entry must be well formed (:func:`problems`); an existing key is refused unless the entry is the same.
    Returns the keys added."""
    import yaml
    if isinstance(entries, (str, Path)):
        entries = yaml.safe_load(Path(entries).read_text()) or {}
    current = registry()
    bad = {k: problems(v) for k, v in entries.items() if problems(v)}
    if bad:
        raise ValueError(f"malformed references: {bad}")
    clash = [k for k, v in entries.items() if k in current
             and {kk: vv for kk, vv in current[k].items() if kk != "key"} != dict(v)]
    if clash:
        raise ValueError(f"references already registered with other data: {clash}")
    _EXTRA.update({k: dict(v) for k, v in entries.items()})
    registry.cache_clear()
    return list(entries)


def ref(key: str) -> dict[str, Any]:
    r = registry().get(key)
    if r is None:
        raise KeyError(f"{key!r} is not in the reference registry ({REGISTRY_FILE.name})")
    return dict(r)


def link(r: dict[str, Any]) -> str:
    """A clickable link for a reference: its DOI, else its URL, else ''."""
    if r.get("doi"):
        return f"https://doi.org/{r['doi']}"
    return r.get("url") or ""


def problems(entry: dict[str, Any]) -> list[str]:
    """What is wrong with a registry entry (empty when it is well formed)."""
    out = []
    for f in ("title", "authors", "year", "venue", "kind"):
        if not entry.get(f):
            out.append(f"missing {f}")
    if entry.get("kind") not in KINDS:
        out.append(f"kind must be one of {KINDS}")
    if not isinstance(entry.get("year"), int) or not 1600 < entry.get("year", 0) < 2100:
        out.append("year must be an integer")
    if not isinstance(entry.get("authors"), list):
        out.append("authors must be a list")
    if entry.get("doi") and not _DOI.match(str(entry["doi"])):
        out.append("doi is not of the form 10.xxxx/...")
    if entry.get("url") and not str(entry["url"]).startswith("https://"):
        out.append("url must be https")
    if entry.get("kind") != "book" and not (entry.get("doi") or entry.get("url")):
        out.append("an article / paper needs a doi or url")
    if not entry.get("topics"):
        out.append("missing topics")
    return out


def _online_ok() -> bool:
    return os.environ.get("PINNEAPPLE_FLOW_VERIFY_REFS", "").lower() in ("1", "true", "yes")


def _resolves(url: str, timeout: float = 6.0) -> bool | None:
    """True / False when the URL answered, None when it could not be checked."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pinneapple-flow-refcheck"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - https only, checked above
            return 200 <= r.status < 400
    except Exception as exc:  # noqa: BLE001 - network failure = cannot check
        code = getattr(exc, "code", None)
        if code in (404, 410):
            return False
        return None


def _get_json(url: str, timeout: float = 8.0) -> tuple[int | None, Any]:
    """(HTTP status, parsed JSON or None); (None, None) when the request could not be made."""
    try:
        import json as _json
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "pinneapple-flow-refcheck (mailto:none)",
                                                   "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - fixed https hosts
            return r.status, _json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001 - network failure = cannot check
        return getattr(exc, "code", None), None


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def verify_doi(doi: str, title: str = "") -> dict[str, Any]:
    """Resolve a DOI online: the doi.org handle API, then Crossref (which also gives the registered title, compared
    with ``title`` when given). ``status``: ``verified``, ``not found`` or ``unverified`` (no network)."""
    out: dict[str, Any] = {"doi": doi, "status": "unverified", "source": "", "title_match": None}
    if not _DOI.match(doi or ""):
        return {**out, "status": "not found", "note": "malformed DOI"}
    import urllib.parse
    code, js = _get_json(f"https://doi.org/api/handles/{urllib.parse.quote(doi, safe='/')}")
    if isinstance(js, dict) and js.get("responseCode") == 1:
        out.update(status="verified", source="doi.org")
    elif code == 404 or (isinstance(js, dict) and js.get("responseCode") == 100):
        out.update(status="not found", source="doi.org")
    code, js = _get_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}")
    if code == 200 and isinstance(js, dict):
        msg = js.get("message") or {}
        ct = (msg.get("title") or [""])[0]
        out.update(status="verified", source=",".join(x for x in (out["source"], "crossref") if x),
                   registered_title=ct)
        if title and ct:
            out["title_match"] = _norm_title(title) == _norm_title(ct) or _norm_title(ct).startswith(
                _norm_title(title)) or _norm_title(title).startswith(_norm_title(ct))
    elif code == 404 and out["status"] != "verified":
        out.update(status="not found", source=",".join(x for x in (out["source"], "crossref") if x))
    return out


def verify_registry() -> dict[str, dict[str, Any]]:
    """Check every registry entry with a DOI online (:func:`verify_doi`): key -> result."""
    return {k: verify_doi(str(v["doi"]), v.get("title", "")) for k, v in registry().items() if v.get("doi")}


def check(reference: dict[str, Any] | str, *, online: bool | None = None) -> dict[str, Any]:
    """A reference as stored on a decision, with ``status``: ``registry`` (curated key), ``verified`` (its DOI / URL
    resolved), ``not found`` (the DOI / URL did not resolve) or ``unverified`` (offline, or no DOI / URL)."""
    if isinstance(reference, str):
        reference = {"key": reference}
    r = dict(reference or {})
    key = r.get("key")
    if key and key in registry():
        out = {**ref(key), **{k: v for k, v in r.items() if k in ("section", "supports")}}
        out["status"] = "registry"
        out["link"] = link(out)
        return out
    if key:
        r["note"] = f"key {key!r} is not in the curated registry"
    doi, url = str(r.get("doi") or "").strip(), str(r.get("url") or "").strip()
    if doi and not _DOI.match(doi):
        r["doi"], doi = "", ""
        r["note"] = (r.get("note", "") + "; malformed DOI dropped").strip("; ")
    if url and not url.startswith("https://"):
        r["url"], url = "", ""
    target = f"https://doi.org/{doi}" if doi else url
    status = "unverified"
    if target and (online if online is not None else _online_ok()):
        if doi:
            v = verify_doi(doi, r.get("title", ""))
            status = v["status"]
            if v.get("title_match") is False:
                r["note"] = (r.get("note", "") + f"; the DOI is registered as {v.get('registered_title')!r}").strip("; ")
        else:
            ok = _resolves(target)
            status = "verified" if ok else ("not found" if ok is False else "unverified")
    r["status"] = status
    r["link"] = target
    return r


def cite(*keys: str, supports: str = "") -> list[dict[str, Any]]:
    """Registry references for ``keys`` (KeyError for an unknown key)."""
    out = []
    for k in keys:
        r = ref(k)
        r["status"] = "registry"
        r["link"] = link(r)
        if supports:
            r["supports"] = supports
        out.append(r)
    return out


def format_reference(r: dict[str, Any]) -> str:
    authors = r.get("authors") or []
    a = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    bits = [b for b in (a, f"({r['year']})" if r.get("year") else "", r.get("title"), r.get("venue")) if b]
    s = " ".join(bits[:2]) + (". " + ". ".join(bits[2:]) if len(bits) > 2 else "")
    if r.get("link"):
        s += f". {r['link']}"
    if r.get("status") and r["status"] != "registry":
        s += f" [{r['status']}]"
    return s
