"""Choosing a catalogued public dataset and reading it — the dispatch behind the stage map's data half.

The catalogue answers "which archives exist and which have a reader". This module answers the next two questions a
pipeline asks: **which one should I use for this stage**, and **how do I actually get a bundle out of it**.

The choice is deterministic and its rule is written down, because a recommender whose rule is hidden cannot be
argued with:

1. the dataset must be tagged with the stage;
2. with ``bundle_only`` (the default) it must have an adapter that returns a :class:`PhysicsDataBundle` — a reader
   that stops at "raw trajectories" is not readable *into this library's contract*, and the catalogue's
   ``readable`` flag alone does not say that;
3. an archive whose licence text marks it non-commercial is dropped when ``commercial_ok`` is false — the text is
   quoted in the refusal rather than interpreted;
4. the rest are ordered by priority (P0 first), then by whether the licence was verified here, then by id. No score,
   no weights: the tie-break is name-stable so two runs choose the same archive.

    >>> from pinneapple_data.datasets.read import choose
    >>> choose("data")["id"]             # P0, licence verified, first by name among equals
    'cfdbench'
    >>> choose("inverse")["id"]          # the subsurface archive is the only one that feeds identification
    'spe10'
    >>> choose("data", prefer="pdebench")["id"]
    'pdebench'
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["Choice", "choose", "read", "readable_stages"]

#: licence texts that forbid a commercial use, matched as substrings; the refusal quotes the text it matched
_NON_COMMERCIAL = ("non-commercial", "noncommercial", "cc-by-nc", "research only")


@dataclass(frozen=True)
class Choice:
    """How one catalogued archive becomes a bundle: the callable and the arguments it insists on."""

    dataset: str
    reader: Callable[..., Any]
    needs: tuple[str, ...] = ()
    bundle: bool = True
    why: str = ""


def _choices() -> dict[str, Choice]:
    """The dispatch table, built lazily so importing this module costs nothing and a missing adapter is a reason."""
    from pinneapple_data.datasets import cfdbench, flowbench, pdebench, spe10, the_well

    def flowbench_reader(**kw: Any) -> Any:
        if "fpo" in kw:
            return flowbench.load_fpo(kw.pop("fpo"), **kw)
        return flowbench.load_tensors(**kw)

    choices = {
        "pdebench": Choice("pdebench", pdebench.load_bundle, needs=("source",),
                           why="PDEBench ships a bundle loader that reads the archive over HTTP or from disk"),
        "the_well": Choice("the_well", the_well.load_bundle, needs=("source", "license"),
                           why="The Well's data terms are per scenario, so the licence is an argument, not a default"),
        "cfdbench": Choice("cfdbench", cfdbench.load_bundle, needs=("case",),
                           why="one directory per case with u.npy and v.npy"),
        "spe10": Choice("spe10", spe10.load_bundle, needs=("perm",),
                        why="the permeability file is the dataset; the licence is the caller's"),
        "flowbench": Choice("flowbench", flowbench_reader, needs=("x", "y"),
                            why="stored as X/Y tensors per family, or one .npz per FPO sample"),
        "deepmind_mgn": Choice("deepmind_mgn", _no_bundle_reader, needs=("dataset",), bundle=False,
                               why="its reader returns trajectories and tensors, not a physics-data bundle, so it is "
                                   "catalogued as readable and not selectable here"),
    }
    return choices


def _no_bundle_reader(**_: Any) -> Any:
    raise NotImplementedError(
        "the meshgraphnets reader returns trajectories (download_prefix + read_trajectories), not a bundle; "
        "use it directly until an adapter to pinneapple_data.schema exists")


_CACHE: dict[str, Choice] | None = None


def _table() -> dict[str, Choice]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _choices()
    return _CACHE


def _non_commercial(license_text: str) -> bool:
    low = license_text.lower()
    return any(marker in low for marker in _NON_COMMERCIAL)


def choose(stage_id: str | None = None, *, bundle_only: bool = True, commercial_ok: bool = False,
           prefer: str | None = None) -> dict[str, Any]:
    """The archive this stage should use, with the rule that picked it and the candidates it beat.

    ``prefer`` names an archive to use if it survived the filters — the escape hatch for a caller that knows which
    case it wants, without falling back to "pass me the id and I will read it unvalidated"."""
    from pinneapple_data.datasets import catalog as core

    stage_id = stage_id or "data"
    entries = core.entries(stage=stage_id)
    table = _table()
    kept, dropped = [], []
    for entry in entries:
        if entry.id not in table:
            dropped.append({"id": entry.id, "why": "no bundle reader in this library" if entry.adapter is None
                            else "its reader does not return a physics-data bundle"})
            continue
        choice = table[entry.id]
        if bundle_only and not choice.bundle:
            dropped.append({"id": entry.id, "why": choice.why})
            continue
        if not commercial_ok and _non_commercial(entry.license):
            dropped.append({"id": entry.id, "why": f"licence: {entry.license}"})
            continue
        kept.append(entry)

    order = {"P0": 0, "P1": 1, "P2": 2}
    kept.sort(key=lambda e: (order.get(e.priority, 9), not e.license_verified, e.id))
    if prefer is not None:                                # after the sort, or the sort would undo it
        chosen = [e for e in kept if e.id == prefer]
        if not chosen:
            return {"id": None, "stage": stage_id, "why": f"{prefer!r} is not a readable candidate here",
                    "candidates": [e.id for e in kept], "dropped": dropped}
        kept = chosen + [e for e in kept if e.id != prefer]
    if not kept:
        return {"id": None, "stage": stage_id, "why": f"no archive tagged {stage_id!r} is readable into a bundle",
                "candidates": [], "dropped": dropped}
    best = kept[0]
    return {
        "id": best.id, "stage": stage_id, "title": best.title, "license": best.license,
        "license_verified": best.license_verified, "needs": list(table[best.id].needs),
        "why": (f"{best.priority} archive tagged {stage_id!r} with a bundle reader; picked over "
                f"{[e.id for e in kept[1:]] or 'nothing'} by priority, then licence verified here, then id"),
        "candidates": [e.id for e in kept], "dropped": dropped,
    }


def read(dataset_id: str, *, gates: bool = True, **kwargs: Any) -> dict[str, Any]:
    """Read a catalogued archive into a bundle, then gate it; refuses a dataset with no bundle reader.

    Returns ``{"bundle": ..., "gates": GateReport | None, "dataset": ...}``, or raises ``KeyError`` naming the
    datasets that *can* be read — the failure a caller can act on."""
    from pinneapple_data.datasets import catalog as core

    table = _table()
    if dataset_id not in table:
        raise KeyError(f"'{dataset_id}' has no bundle reader; readable: {sorted(k for k, c in table.items() if c.bundle)}")
    choice = table[dataset_id]
    if not choice.bundle:
        raise ValueError(f"'{dataset_id}': {choice.why}")
    entry = core.get(dataset_id)
    missing = [n for n in choice.needs if n not in kwargs]
    if missing:
        raise TypeError(f"reading '{dataset_id}' needs {missing}; it is read as: {choice.why}")
    bundle = choice.reader(**kwargs)
    report = None
    if gates:
        from pinneapple_data.schema import run_gates
        report = run_gates(bundle)
    return {"bundle": bundle, "gates": report, "dataset": dataset_id, "license": entry.license}


def readable_stages() -> dict[str, list[str]]:
    """Stage -> the catalogued datasets that a pipeline can read into a bundle today."""
    from pinneapple_data.datasets import catalog as core

    table = _table()
    out: dict[str, list[str]] = {}
    for entry in core.entries():
        if entry.id in table and table[entry.id].bundle:
            for stage in entry.stages:
                out.setdefault(stage, []).append(entry.id)
    return {k: sorted(v) for k, v in out.items()}
