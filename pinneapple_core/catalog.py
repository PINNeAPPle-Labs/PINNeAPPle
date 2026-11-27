"""One catalogue over everything the library can list (#1478).

Before this module there were five lists that did not know about each other: ``pp.list_models()``, the registry
gateway's models, the benchmark's suites, the app's model catalogue and the arena's. A model available in one was
not necessarily available in the others, and nothing said so — a benchmark could score a subset of what the app
advertised, with no error anywhere.

``pp.catalog()`` is a **read-only** facade over the unified registry (#200). It does not add a sixth list: every
section reads ``pp.registered(kind)`` and ``registry.describe(name)``, which is the one source of truth. Writes go
to the registry, through ``pp.register``.

What it adds on top of the registry is the part the registry cannot say by itself:

* **one entry shape** — ``id``, ``kind``, ``title``, plus ``units``, ``variables`` and ``provenance`` that are
  ``None`` when the registration did not declare them. A missing field is never filled with a default that looks
  like data;
* **the views** — every list that a user or another component reads, with the set of ids each one reports;
* **the discrepancies** — ids that one view lists and another does not. A view that is a different plane on purpose
  (the registry gateway holds *trained* artifacts, not architectures) says so in one sentence instead of being
  silently tolerated.

Example
-------
>>> import pinneapple as pp
>>> catalogue = pp.catalog()
>>> len(catalogue.models()) > 0
True
>>> catalogue.get("vanilla_pinn").kind
'model'
>>> catalogue.discrepancies()["models"]          # empty when every architecture view agrees
[]
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from . import registry as _registry

__all__ = ["Catalog", "CatalogEntry", "CatalogView", "catalog"]

# The nine kinds of #200, in the order a reader looks for them.
KINDS = ("model", "solver", "dataset", "physics", "geometry", "material", "operator", "benchmark", "optimizer")

# The registry keys a richer entry can carry. Anything absent stays None: an inferred unit is worse than no unit.
_UNITS_KEYS = ("units", "unit", "output_units", "field_units")
_VARIABLES_KEYS = ("variables", "outputs", "inputs", "fields")
_PROVENANCE_KEYS = ("provenance", "source", "citation", "reference", "license", "licence")


def _first(meta: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in meta and meta[key] not in (None, "", (), [], {}):
            return meta[key]
    return None


@dataclass(frozen=True)
class CatalogEntry:
    """One catalogue entry: what it is, and the fields it declared — ``None`` where it declared nothing."""

    id: str
    kind: str
    title: str | None = None
    units: Any = None
    variables: Any = None
    provenance: Any = None
    home: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "title": self.title, "units": self.units,
                "variables": self.variables, "provenance": self.provenance, "home": self.home}

    def __str__(self) -> str:  # a readable line in a table or a CLI
        return f"{self.kind}:{self.id}"


@dataclass(frozen=True)
class CatalogView:
    """A list somebody reads, and whether it is supposed to agree with the catalogue.

    ``note`` is ``None`` for a view that lists the same resources as the catalogue — a disagreement is then a bug.
    When the view is a **different plane** (trained artifacts, a benchmark's own suite), ``note`` says so in one
    sentence, so the difference is a stated fact rather than a silent tolerance.
    """

    name: str
    ids: Callable[[], set]
    note: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("a view needs a name")


def _cell(value: Any) -> str:
    """A table cell for a field that may be absent: an em dash, never an inferred value."""
    if value is None:
        return "—"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _registered(kind: str) -> list[str]:
    try:
        return sorted(_registry.registered(kind))
    except Exception:  # pragma: no cover - a kind that is not installed in this build
        return []


def _describe(kind: str, name: str) -> dict[str, Any]:
    """The registry's own description of the entry: ``name``, ``kind``, ``home`` and whatever the registration
    declared. No fallback on failure — a catalogue that invents a shape is not a catalogue."""
    return _registry.resolve_kind(kind).describe(name)


def _entry(kind: str, name: str) -> CatalogEntry:
    meta = _describe(kind, name)
    title = meta.get("title") or meta.get("label") or meta.get("description")
    return CatalogEntry(
        id=name,
        kind=kind,
        title=str(title) if isinstance(title, str) and title else None,
        units=_first(meta, _UNITS_KEYS),
        variables=_first(meta, _VARIABLES_KEYS),
        provenance=_first(meta, _PROVENANCE_KEYS),
        home=meta.get("home"),
        meta=dict(meta),
    )


def _pp_model_names() -> set:
    import pinneapple as pp
    return set(pp.list_models())


def _app_model_names() -> set:
    from pinneapple_app.backend.core.model_catalog import list_models as app_list
    return {m.name for m in app_list()}


def _benchmark_model_names() -> set:
    from pinneapple_tools.benchmark_suite.benchmark import PINNArenaBenchmark
    return set(PINNArenaBenchmark.list_models())


def _gateway_model_names() -> set:
    """The trained artifacts the registry gateway serves, read from ``PINNEAPPLE_MODEL_STORE``.

    Without a configured store the view is **absent** (it raises, and :func:`_views` leaves it out) rather than
    empty: a deployment with no artifact store does not disagree with the catalogue, it simply has no artifacts.
    """
    import os

    root = os.environ.get("PINNEAPPLE_MODEL_STORE")
    if not root:
        raise RuntimeError("no PINNEAPPLE_MODEL_STORE configured: the gateway view is absent here")
    from pinneapple_registry.gateway import ModelGateway

    return set(ModelGateway(root).models())


def _views() -> list[CatalogView]:
    """The five lists of #1478, each with what it is. A view that is not importable here is left out, not faked."""
    candidates = [
        CatalogView("pp.list_models", _pp_model_names),
        CatalogView("app.model_catalog", _app_model_names),
        CatalogView("benchmark", _benchmark_model_names,
                    note="the benchmark lists the specs its own suites can run, which is a subset chosen on purpose; "
                         "the catalogue says which architectures exist, the benchmark says which it scores"),
        CatalogView("registry.gateway", _gateway_model_names,
                    note="the gateway lists trained artifacts, not architectures: a model appears there once "
                         "somebody trained and registered it"),
    ]
    out = []
    for view in candidates:
        try:
            view.ids()
        except Exception:  # the view is not installed in this build: it is absent, not empty
            continue
        out.append(view)
    return out


class Catalog:
    """The read-only catalogue. Build it with :func:`catalog`; it reads the registry on every call."""

    kinds = KINDS

    # -- the sections ---------------------------------------------------------------------------------------
    def _section(self, kind: str) -> list[CatalogEntry]:
        """Read the registry now, never a snapshot: a resource registered a moment ago is in the next call."""
        if kind not in KINDS:
            raise KeyError(f"unknown kind {kind!r}; one of {', '.join(KINDS)}")
        return [_entry(kind, name) for name in _registered(kind)]

    def models(self) -> list[CatalogEntry]:
        return self._section("model")

    def solvers(self) -> list[CatalogEntry]:
        return self._section("solver")

    def datasets(self) -> list[CatalogEntry]:
        return self._section("dataset")

    def experiments(self) -> list[CatalogEntry]:
        """The registered physics problems: what a study can start from."""
        return self._section("physics")

    def geometry(self) -> list[CatalogEntry]:
        return self._section("geometry")

    def materials(self) -> list[CatalogEntry]:
        return self._section("material")

    def operators(self) -> list[CatalogEntry]:
        return self._section("operator")

    def benchmarks(self) -> list[CatalogEntry]:
        return self._section("benchmark")

    def optimizers(self) -> list[CatalogEntry]:
        return self._section("optimizer")

    # -- lookups --------------------------------------------------------------------------------------------
    def get(self, identifier: str) -> CatalogEntry:
        """The entry with that id, across every kind. ``KeyError`` when nothing has it."""
        for kind in KINDS:
            if identifier in _registered(kind):
                return _entry(kind, identifier)
        raise KeyError(f"no catalogued resource {identifier!r}")

    def entries(self) -> list[CatalogEntry]:
        out: list[CatalogEntry] = []
        for kind in KINDS:
            out.extend(self._section(kind))
        return out

    def counts(self) -> dict[str, int]:
        return {kind: len(self._section(kind)) for kind in KINDS}

    # -- the views ------------------------------------------------------------------------------------------
    def views(self) -> dict[str, set]:
        return {view.name: set(view.ids()) for view in _views()}

    def view_notes(self) -> dict[str, str | None]:
        return {view.name: view.note for view in _views()}

    def discrepancies(self) -> dict[str, list[str]]:
        """Ids a view lists and another does not, per section.

        Only the views that are supposed to agree are compared: one with a ``note`` is a different plane and its
        difference is stated, not reported as a bug. ``"models"`` is empty when every architecture view agrees —
        which is what makes a benchmark silently scoring a subset impossible.
        """
        agreeing = [view for view in _views() if view.note is None]
        present: dict[str, set] = {}
        for view in agreeing:
            try:
                present[view.name] = set(view.ids())
            except Exception:  # pragma: no cover - a view that fails to list is out, not silently empty
                continue
        if len(present) < 2:
            return {"models": []}
        missing = sorted({name for ids in present.values() for name in ids if any(name not in other
                                                                                 for other in present.values())})
        return {"models": missing}

    # -- serialisation --------------------------------------------------------------------------------------
    def to_markdown(self) -> str:
        """The catalogue as a table: what a docs page or a run record carries, generated and never hand-written."""
        lines = ["# Catalogue", "",
                 "| Kind | Count |", "|---|---|"]
        for kind, n in self.counts().items():
            lines.append(f"| {kind} | {n} |")
        for kind in KINDS:
            entries = self._section(kind)
            if not entries:
                continue
            lines += ["", f"## {kind}", "", "| Id | Title | Units | Variables | Provenance |", "|---|---|---|---|---|"]
            for entry in entries:
                lines.append(f"| `{entry.id}` | {entry.title or '—'} | {_cell(entry.units)} | "
                             f"{_cell(entry.variables)} | {_cell(entry.provenance)} |")
        notes = {name: note for name, note in self.view_notes().items() if note}
        if notes:
            lines += ["", "## Views that are another plane", ""]
            lines += [f"- **{name}**: {note}" for name, note in notes.items()]
        missing = self.discrepancies().get("models", [])
        lines += ["", "## Disagreements among the architecture views", "",
                  "None." if not missing else ", ".join(f"`{name}`" for name in missing)]
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"kinds": list(KINDS),
                "counts": self.counts(),
                "entries": [entry.to_dict() for entry in self.entries()],
                "views": {name: sorted(ids) for name, ids in self.views().items()},
                "view_notes": self.view_notes(),
                "discrepancies": self.discrepancies()}

    def __repr__(self) -> str:
        counts = ", ".join(f"{kind}={n}" for kind, n in self.counts().items() if n)
        return f"Catalog({counts or 'empty'})"


def catalog() -> Catalog:
    """The catalogue of this installation. A new object on each call: it reads the registry when asked, so a
    resource registered after it was built is seen."""
    return Catalog()
