"""Stage 7, second half: the index over the cards.

Not a service and not multi-tenant — a reader. Every model the campaign has ever fitted is already on disk as a
``card.json``; the registry turns that directory tree into the four questions an engineer actually asks: which families
have a model, which of them is worth using, which dataset it came from, and which gate it failed.

The answers come from the cards and nothing else, so the registry cannot drift from the evidence.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .card import read_card, registry_row

__all__ = ["best", "entries", "summary", "table"]


def entries(root: str | Path) -> list[dict[str, Any]]:
    """Every card under ``root``, one row each, newest first. A card that will not parse is reported, not skipped."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("*/model/card.json")):
        try:
            rows.append(registry_row(read_card(path), path))
        except (OSError, ValueError) as exc:
            rows.append({"family": path.parent.parent.name, "path": str(path), "error": str(exc)})
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return rows


def usable(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Those that beat their baseline and pass their tolerance — the only ones worth pointing a client at."""
    return [r for r in rows
            if r.get("relative_l2") is not None and r.get("baseline") is not None
            and r["relative_l2"] < r["baseline"] and r.get("within_tolerance")]


def best(rows: Iterable[dict[str, Any]], family: str | None = None) -> dict[str, Any] | None:
    """The lowest held-out error among the usable models (optionally of one family), or ``None``."""
    pool = [r for r in usable(rows) if family is None or r.get("family") == family]
    return min(pool, key=lambda r: r["relative_l2"]) if pool else None


def summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    ok = usable(rows)
    return {"n_models": len(rows), "n_usable": len(ok),
            "families": sorted({r.get("family", "") for r in rows}),
            "licences": sorted({r.get("licence", "") for r in rows if r.get("licence")}),
            "failed_gates": sorted({g for r in rows for g in r.get("gates_failed", [])})}


def table(rows: Iterable[dict[str, Any]]) -> str:
    """A plain table, because the first consumer of a registry is a person in a terminal."""
    rows = list(rows)
    if not rows:
        return "no cards found"
    head = f"{'family':16s} {'held-out':>8s} {'baseline':>8s} {'p90':>7s}  {'tolerance':>9s}  {'usage':14s} gates"
    lines = [head, "-" * len(head)]
    for r in rows:
        if "error" in r:
            lines.append(f"{r.get('family', '?'):16s} CARD UNREADABLE: {r['error'][:60]}")
            continue
        rel = "     -  " if r["relative_l2"] is None else f"{r['relative_l2']:8.3f}"
        base = "     -  " if r["baseline"] is None else f"{r['baseline']:8.3f}"
        p90 = "    -  " if r["p90"] is None else f"{r['p90']:7.3f}"
        tol = "   pass" if r["within_tolerance"] else "   FAIL"
        gates = ",".join(r.get("gates_failed", [])) or "ok"
        if r.get("gates_not_applicable"):
            gates += f" (na: {','.join(r['gates_not_applicable'])})"
        lines.append(f"{r['family']:16s} {rel} {base} {p90}  {tol:>9s}  {r.get('usage', ''):14s} {gates}")
    return "\n".join(lines)
