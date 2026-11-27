"""One registry for the public physics datasets and the pre-trained models (#528).

Two catalogues already existed and answered different questions: :mod:`pinneapple_data.datasets.catalog` lists the
*external* public datasets (licence, whether the licence was verified, citation, homepage, the adapter that reads it)
and :mod:`pinneapple_hub.zoo` lists PINNeAPPle's own published artifacts (weights, SHA-256, version, Hugging Face
repo, model card). Each solved "how do I get this" its own way. This module is the single door in front of both:

    >>> from pinneapple_data import registry
    >>> [e.id for e in registry.list(domain="CFD")][:3]
    >>> registry.licence("flowbench")                  # printed before anything is downloaded
    >>> data = registry.load("airfoil_rans_cst_re4e6", split="test")     # zoo dataset, verified SHA-256
    >>> registry.load("pdebench")                      # NoReaderError: the call that does work, named
    >>> registry.metadata("airfoil_rans_cst_re4e6")    # what a run records about the data it used

It **composes, it does not re-describe**: nothing here carries its own copy of a licence, a citation or a checksum.
An entry's fields are read from the two sources at load time (``catalog.CATALOG`` and the zoo's ``manifest.json``),
so correcting the licence in one place corrects the registry, the docs table and every run record together.

The licence is shown **before** the download and the entry says whether a product may use the data: a non-commercial
licence (FlowBench's CC-BY-NC-4.0) and an access that needs registration are flagged on the entry, never discovered
after the files are on disk. The citation is printed after the load and returned by :func:`metadata`, which is the
dict a Lab run stores in its dataset card (:func:`record`).

What it will not do: invent a reader. ``airfrans`` is registered with its licence and citation and *no* loader --
the entry names the formats and the homepage, and :func:`load` raises :class:`NoReaderError` with the call that does
work today, because a registry that pretends to read ten datasets is a reading list (the argument of
:mod:`pinneapple_data.datasets.catalog`). Publishing to the Hub or to Zenodo (:func:`publish`) is declared, not
executed: it needs a token and an upload the CI cannot make, and the zoo already ships its own publisher.

Command line::

    python -m pinneapple_data.registry                       # every entry, one line each
    python -m pinneapple_data.registry --docs docs/core_concepts/datasets_and_models.md
    python -m pinneapple_data.registry --licence flowbench --citation flowbench
    python -m pinneapple_data.registry --json > registry.json
"""
from __future__ import annotations

import argparse
import builtins
import json
import sys
import warnings
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SCHEMA", "KINDS", "Entry", "NoReaderError", "RegistryError", "PublishError", "entries", "get", "info",
           "licence", "citation", "list", "load", "metadata", "publish", "record", "table", "to_markdown"]

SCHEMA = "pinneapple.registry/1"
KINDS = ("dataset", "model")


class RegistryError(RuntimeError):
    """The registry cannot answer: an unknown id, or a source that does not read."""


class NoReaderError(RegistryError):
    """The entry is registered (with its licence and citation) but this library has no reader for it."""

    def __init__(self, entry: Entry, how: str = "") -> None:
        message = (f"no reader registered for {entry.id!r}: {entry.title}. Licence {entry.licence}"
                   + (" (non-commercial)" if not entry.commercial_use else "")
                   + (f"; what it takes today: {how}" if how else "")
                   + (f"; homepage {entry.homepage}" if entry.homepage else ""))
        super().__init__(message)
        self.entry = entry
        self.how = how


class PublishError(RegistryError):
    """Publishing needs credentials and an upload; it is declared here and executed by the zoo's publisher."""


@dataclass(frozen=True)
class Entry:
    """One dataset or model: what it is, who may use it, how it is cited, and whether this library reads it."""
    id: str
    kind: str
    title: str
    domain: str = ""
    summary: str = ""
    homepage: str = ""
    licence: str = ""
    licence_verified: bool = False
    commercial_use: bool | None = None             # None: the licence does not say, and neither will this module
    gated: bool = False                            # needs registration or an API key
    citation: str = ""
    size: str = ""
    version: str = ""
    source: str = ""                               # "catalog" (external) or "zoo" (PINNeAPPle's own)
    reader: str = ""                               # the call registry.load uses, when the registry can load it
    adapter: str = ""                              # a reader in the library that takes a local path (not uniform)
    files: Mapping[str, Any] = field(default_factory=dict)     # name -> {sha256, size}, zoo entries only

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "domain": self.domain,
                "summary": self.summary, "homepage": self.homepage, "licence": self.licence,
                "licence_verified": self.licence_verified, "commercial_use": self.commercial_use,
                "gated": self.gated, "citation": self.citation, "size": self.size, "version": self.version,
                "source": self.source, "reader": self.reader, "adapter": self.adapter,
                "readable": bool(self.reader),
                "checksums": {k: v.get("sha256") for k, v in self.files.items() if isinstance(v, Mapping)}}

    @property
    def readable(self) -> bool:
        return bool(self.reader)

    @property
    def product_warning(self) -> str:
        """Why a product may not ship this data, in one line; empty when nothing is known against it."""
        if self.commercial_use is False:
            return f"non-commercial licence ({self.licence}): research use only"
        if self.gated:
            return f"gated access ({self.licence or 'terms of use'}): registration or a key is needed"
        if not self.licence_verified:
            return f"licence not verified ({self.licence or 'not stated'})"
        return ""


# ---------------------------------------------------------------------------------------------- the two sources
def _catalog_entries() -> Iterator[Entry]:
    from .datasets import catalog as C
    for e in C.entries():
        text = str(e.license or "")
        yield Entry(id=e.id, kind="dataset", title=e.title, domain=e.domain, summary=" ".join(e.use or ())[:300],
                    homepage=e.homepage, licence=text.split(" (")[0].strip(),
                    licence_verified=bool(e.license_verified),
                    commercial_use=_commercial_of(text), gated=e.access != "open", citation=str(e.citation or ""),
                    size=e.size, source="catalog", adapter=(e.adapter or ""))


def _commercial_of(licence_text: str) -> bool | None:
    """What the licence text in the catalogue allows. ``None`` where the text does not say: the module never
    upgrades "unverified" to "fine for a product"."""
    low = licence_text.lower()
    if low.startswith("code "):
        return None                                # the text covers the code, not the data beside it
    if "nc" in low.replace("-", " ").split() or "non-commercial" in low or "noncommercial" in low:
        return False
    if "public domain" in low:
        return True
    for spdx in ("apache-2.0", "mit", "bsd-3", "cc-by-4.0", "odbl"):
        if spdx in low:
            return True
    return None


def _zoo_entries() -> Iterator[Entry]:
    from pinneapple_hub.zoo import _artifacts as art  # the same manifest the zoo loads from
    man = art.manifest()
    for kind, plural in (("dataset", "datasets"), ("model", "models")):
        for name, item in (man.get(plural) or {}).items():
            licence = str(item.get("license") or "")
            cite = str(item.get("citation") or "")
            if kind == "model":                    # a model ships its card: the licence and citation are there
                card_licence, card_cite = _model_card(name)
                licence = licence or card_licence
                cite = cite or card_cite
            yield Entry(id=name, kind=kind, title=str(item.get("title") or item.get("task") or name),
                        domain=str(item.get("domain") or item.get("task") or "")[:120],
                        summary=str(item.get("description") or "")[:300],
                        homepage=f"https://huggingface.co/{item.get('hf_repo', '')}" if item.get("hf_repo") else "",
                        licence=licence.split(" (")[0].strip(), licence_verified=bool(licence),
                        commercial_use=_commercial_of(licence), gated=False,
                        citation=cite, size=str(item.get("size") or ""),
                        version=str(item.get("version") or ""), source="zoo",
                        reader=f"pinneapple_hub.zoo.{plural}:load", files=item.get("files") or {})


def _model_card(name: str) -> tuple[str, str]:
    """A model's licence and citation, from its own card, when the manifest does not repeat them (the card is what the
    model ships with, so it is the source; an unreadable card gives blanks rather than a guess)."""
    try:
        from pinneapple_hub.zoo import info as _info
        card = _info(name, offline=True)
        return str(getattr(card, "license", "") or ""), str(getattr(card, "citation", "") or "")
    except Exception:  # noqa: BLE001 - a card that is not bundled offline gives no licence here
        return "", ""


_CACHE: dict[str, list[Entry]] = {}


def entries(*, reload: bool = False) -> list[Entry]:
    """Every entry of both sources, the external datasets first, then the zoo's own. Cached until ``reload``."""
    if reload or "all" not in _CACHE:
        items: list[Entry] = []
        problems: list[str] = []
        for build in (_catalog_entries, _zoo_entries):
            try:
                items.extend(build())
            except Exception as e:  # noqa: BLE001 - one unreadable source must not empty the registry
                problems.append(f"{build.__name__}: {type(e).__name__}: {e}")
        _CACHE["all"] = sorted(items, key=lambda e: (e.kind, e.id))
        _CACHE["problems"] = problems                        # type: ignore[assignment]
    return _CACHE["all"]


def sources() -> list[str]:
    """The problems met while reading the sources, so a caller can see a catalogue that silently went empty."""
    entries()
    return builtins.list(_CACHE.get("problems") or [])


def list(domain: str | None = None, kind: str | None = None, *, commercial_only: bool = False,
         readable: bool | None = None, source: str | None = None) -> list[Entry]:  # noqa: A001
    """The entries, filtered: ``domain`` is a case-insensitive substring, ``kind`` is ``dataset`` or ``model``."""
    out = entries()
    if kind:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        out = [e for e in out if e.kind == kind]
    if domain:
        d = domain.lower()
        out = [e for e in out if d in (e.domain + " " + e.title + " " + e.id).lower()]
    if commercial_only:
        out = [e for e in out if e.commercial_use is not False]
    if readable is not None:
        out = [e for e in out if e.readable is readable]
    if source:
        out = [e for e in out if e.source == source]
    return out


def get(entry_id: str) -> Entry:
    """One entry by id."""
    for e in entries():
        if e.id == entry_id:
            return e
    close = [e.id for e in entries() if entry_id.lower() in e.id.lower()][:5]
    raise KeyError(f"{entry_id!r} is not in the registry" + (f"; did you mean {close}?" if close else
                                                             f"; {len(entries())} entries, see registry.list()"))


def info(entry_id: str) -> dict[str, Any]:
    """The entry as a JSON-ready dict."""
    return get(entry_id).to_dict()


# ---------------------------------------------------------------------------------------------- licence and citation
def licence(entry_id: str) -> str:
    """The banner shown **before** the download: what the licence is, whether a product may use it, and where to
    read the terms. One line, because it is printed."""
    e = get(entry_id)
    parts = [f"{e.id}: {e.kind}, licence {e.licence or 'not stated'}"
             + ("" if e.licence_verified else " (unverified)")
             + (f", version {e.version}" if e.version else "")]
    if e.product_warning:
        parts.append(e.product_warning)
    if e.homepage:
        parts.append(f"terms: {e.homepage}")
    return " | ".join(parts)


def citation(entry_id: str) -> str:
    """The citation to use, or a sentence saying it is not recorded (never a guessed one)."""
    e = get(entry_id)
    return e.citation or f"{e.id}: no citation recorded in the registry; see {e.homepage or 'the entry page'}"


def metadata(entry_id: str, **extra: Any) -> dict[str, Any]:
    """What a run records about the data or the model it used: id, licence, citation, version and checksums.

    This is the dict :func:`record` writes into a Lab dataset card and the one a report can quote; ``extra`` is
    merged last, so a caller may add its own split, seed or row count."""
    e = get(entry_id)
    out = {"schema": SCHEMA, "id": e.id, "kind": e.kind, "title": e.title, "source": e.source,
           "licence": e.licence, "licence_verified": e.licence_verified, "commercial_use": e.commercial_use,
           "citation": e.citation, "version": e.version, "checksums": e.to_dict()["checksums"]}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------------------------- loading
def load(entry_id: str, split: str | None = None, *, quiet: bool = False, offline: bool | None = None,
         **kw: Any):
    """Load an entry, with the licence printed before the download and the citation after it.

    ``split`` is passed to the zoo's dataset loader (``None`` loads every split as ``{name: split}``). A gated entry
    is refused unless the caller says it has access (``allow_gated=True``), which is the point of flagging it: the
    terms are somebody else's, and this module will not pretend it accepted them.
    """
    e = get(entry_id)
    allow_gated = bool(kw.pop("allow_gated", False))
    if not e.readable:
        raise NoReaderError(e, _reader_hint(e))
    if e.gated and not allow_gated:
        raise RegistryError(f"{e.id} needs registration or a key ({e.licence}); pass allow_gated=True once you have "
                            f"it, and see {e.homepage}")
    if not quiet:
        print(licence(entry_id), file=sys.stderr)
    if e.commercial_use is False:
        warnings.warn(f"{e.id}: {e.product_warning}; do not ship it inside a product", stacklevel=2)
    if e.source == "zoo":
        if e.kind == "dataset":
            from pinneapple_hub.zoo import datasets as Z
            data = Z.load(e.id, split=split, offline=offline, **kw)
        else:
            from pinneapple_hub.zoo import load as _load_model
            data = _load_model(e.id, offline=offline, **kw)
    else:                                                                     # pragma: no cover - no reader today
        raise NoReaderError(e, _reader_hint(e))
    if not quiet:
        print(f"cite: {citation(entry_id)}", file=sys.stderr)
    return data


def _reader_hint(e: Entry) -> str:
    """The call that reads this entry today, from the catalogue's adapter, or why there is none."""
    if e.reader:
        return f"{e.reader}({e.id!r})"
    if e.adapter:
        return (f"{e.adapter} reads it from a local copy (load_bundle / open_cases), not by name: the registry has no "
                f"uniform loader for a file the user must fetch first")
    return (f"the files are public ({e.homepage or 'see the entry page'}) but this library has no adapter for them; "
            f"the formats are the entry's, and the registry will not pretend to read them")


def record(ctx: Any, entry_id: str, name: str | None = None, **extra: Any) -> Any:
    """Register the use of an entry in a Lab run: the dataset card carries the licence and the citation.

        with experiment("my_study") as ex:
            data = registry.load("airfoil_rans_cst_re4e6", split="test")
            registry.record(ex.ctx, "airfoil_rans_cst_re4e6")

    The card's ``license`` field and its ``provenance.registry`` block are what the curation reads; nothing is
    written when the entry does not exist, so a typo fails here instead of in a report six months later.
    """
    e = get(entry_id)
    meta = metadata(entry_id, **extra)
    writer = ctx.dataset(name or e.id, description=f"{e.title}. {e.summary}".strip()[:400], license=e.licence,
                         provenance={"registry": meta})
    return writer


# ---------------------------------------------------------------------------------------------- publishing
def publish(entry_id: str, *, target: str = "hub", token: str | None = None, **_: Any) -> dict[str, Any]:
    """Declared, not executed (#528): pushing a dataset or a surrogate to the Hub or to Zenodo.

    The plan is returned so a caller sees exactly what would be uploaded (every file with its SHA-256, the model
    card, the licence), and :class:`PublishError` says what is missing to do it. The zoo already ships the publisher
    it was built with (``pinneapple_hub.zoo.publish``); this function exists so the registry has one place that says
    where a published artifact comes from, instead of a fourth downloader with its own conventions."""
    e = get(entry_id)
    if target not in ("hub", "zenodo"):
        raise ValueError(f"target must be 'hub' or 'zenodo', got {target!r}")
    plan = {"schema": SCHEMA, "id": e.id, "target": target, "licence": e.licence,
            "files": e.to_dict()["checksums"], "citation": e.citation, "card": "model_card.json"}
    raise PublishError(f"publishing {e.id} to {target} is not done here: it needs a token and an upload. The plan "
                       f"would be {json.dumps(plan, sort_keys=True)[:400]}")


# ---------------------------------------------------------------------------------------------- the docs table
_COLUMNS = ("id", "kind", "domain", "licence", "verified", "product", "citation", "read", "size")


def table(rows: Iterable[Entry] | None = None) -> list[list[str]]:
    """The entries as table cells, for the generated docs page (and for a terminal)."""
    out = [builtins.list(_COLUMNS)]
    for e in (rows if rows is not None else entries()):
        product = "no" if e.commercial_use is False else ("yes" if e.commercial_use else "?")
        out.append([e.id, e.kind, e.domain[:40], e.licence or "not stated", "yes" if e.licence_verified else "no",
                    product, (e.citation or "-")[:90],
                    "yes" if e.readable else ("adapter" if e.adapter else "no"), e.size[:30]])
    return out


def to_markdown(rows: Iterable[Entry] | None = None) -> str:
    """The generated documentation page: the table, the legend and where each fact comes from."""
    items = builtins.list(rows if rows is not None else entries())
    cells = table(items)
    head, body = cells[0], cells[1:]
    ds = sum(1 for e in items if e.kind == "dataset")
    ms = sum(1 for e in items if e.kind == "model")
    lines = ["# Datasets and models: the registry", "",
             f"Generated by `python -m pinneapple_data.registry --docs docs/core_concepts/datasets_and_models.md` "
             f"(schema `{SCHEMA}`, #528). Do not edit by hand.", "",
             f"{ds} datasets and {ms} models. Every field is read from the source that owns it: the external "
             "datasets from `pinneapple_data.datasets.catalog` (licence, whether it was verified, citation, "
             "homepage, adapter) and PINNeAPPle's own artifacts from `pinneapple_hub/zoo/manifest.json` (weights, "
             "SHA-256, version, Hugging Face repo, model card).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in body:
        lines.append("| " + " | ".join(c.replace("|", "/") for c in row) + " |")
    lines += ["",
              "**product**: `no` means the licence forbids commercial use (the entry is still usable for research), "
              "`?` means the licence does not say, `yes` means the licence text allows it -- which is not a legal "
              "opinion, and `verified` says whether the licence was read at the source. **read**: whether this "
              "library has a reader for it today; `no` is a documented gap, not a broken link.", "",
              "Usage: `pinneapple_data.registry.load(\"<id>\", split=\"test\")`, `registry.licence(id)` before a "
              "download, `registry.record(ctx, id)` inside a run so the licence and the citation land in the dataset "
              "card and therefore in the curation."]
    problems = sources()
    if problems:
        lines += ["", "## Sources that did not read", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


def cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the datasets and models registry (#528)")
    ap.add_argument("--docs", default=None, help="write the generated markdown page to this path")
    ap.add_argument("--json", action="store_true", help="print every entry as JSON")
    ap.add_argument("--licence", default=None, help="print the licence banner of one entry")
    ap.add_argument("--citation", default=None, help="print the citation of one entry")
    ap.add_argument("--kind", default=None, choices=builtins.list(KINDS))
    ap.add_argument("--domain", default=None)
    a = ap.parse_args(argv)
    if a.licence:
        print(licence(a.licence))
        return 0
    if a.citation:
        print(citation(a.citation))
        return 0
    if a.docs:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(a.docs)), exist_ok=True)
        with open(a.docs, "w") as f:
            f.write(to_markdown())
        print(a.docs)
        return 0
    if a.json:
        print(json.dumps({"schema": SCHEMA, "entries": [e.to_dict() for e in list(kind=a.kind, domain=a.domain)]},
                         indent=1))
        return 0
    for row in table(list(kind=a.kind, domain=a.domain)):
        print(" | ".join(row))
    return 0


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(cli())
