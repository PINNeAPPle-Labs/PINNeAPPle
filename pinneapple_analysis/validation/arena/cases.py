"""Stored regression cases of the arena and their replay (``pinneapple check --arena-cases``).

A cases file holds the problem (name and options), the hardened defence (check names and thresholds) and every
escape and false alarm the red team found, each as a few numbers: the seed and variation of the correct case, and
the generator with its parameters. Replaying regenerates each case, runs the stored defence and requires every
escape to be rejected and every false alarm to pass. A file is JSON::

    {"format": "pinneapple-arena-cases/1", "problem": "darcy_2d", "options": {...},
     "defence": {"checks": [...], "thresholds": {...}, ...},
     "cases": [{"kind": "escape", "seed": 123, "variation": {}, "generator": "coefficient_field",
                "params": {...}, "true_error": 0.14, ...}, ...]}
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..fault_injection.checks import check_library
from ..fault_injection.problems import get_problem
from .blue import Defence
from .red import Finding

__all__ = ["FORMAT", "save_cases", "load_cases", "replay", "ReplayResult", "builtin_cases", "load_defence"]

FORMAT = "pinneapple-arena-cases/1"
BUILTIN = Path(__file__).resolve().parent / "cases"


def save_cases(path: str | Path, problem, defence: Defence, findings: Sequence[Finding],
               meta: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": FORMAT, "problem": problem.name, "options": problem.options, "defence": defence.to_dict(),
           "cases": [f.to_dict() for f in findings], "meta": dict(meta or {})}
    path.write_text(json.dumps(doc, indent=1, default=float) + "\n", encoding="utf-8")
    return path


def load_cases(path: str | Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not an arena cases file (format {doc.get('format')!r}, expected {FORMAT!r})")
    return doc


def builtin_cases() -> list[Path]:
    """The cases files shipped with the library (one per problem)."""
    return sorted(BUILTIN.glob("*.json"))


@dataclass
class ReplayResult:
    path: str
    problem: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def regressions(self) -> list[dict[str, Any]]:
        return [r for r in self.rows if not r["ok"]]

    @property
    def ok(self) -> bool:
        return not self.regressions

    def lines(self) -> list[str]:
        out = [f"{self.path} ({self.problem}): {len(self.rows) - len(self.regressions)}/{len(self.rows)} cases hold"]
        for r in self.rows:
            want = "rejected" if r["kind"] == "escape" else "accepted"
            out.append(f"  [{'ok' if r['ok'] else 'REGRESSION'}] {r['kind']} seed={r['seed']} "
                       f"{r.get('generator') or 'correct case'}: must be {want}, "
                       f"{'rejected' if r['rejected'] else 'accepted'} (largest margin {r['margin']:+.3g})")
        return out


def load_defence(path: str | Path) -> tuple[Any, Defence]:
    """The problem and the hardened defence stored in a cases file (``builtin_cases()`` for the shipped ones)."""
    doc = load_cases(path)
    p = get_problem(doc["problem"], **doc.get("options", {}))
    lib = check_library(p)
    dd = doc["defence"]
    missing = [n for n in dd["checks"] if n not in lib]
    if missing:
        raise KeyError(f"{path}: checks not in the library of {p.name}: {missing}")
    return p, Defence([lib[n] for n in dd["checks"]], {n: float(t) for n, t in dd["thresholds"].items()},
                      dd.get("kind", "stored"), dd.get("alpha"), dd.get("alpha_cal"), dd.get("scales"),
                      dd.get("statistic_threshold"), int(dd.get("n_calibration", 0)), dd.get("alpha_guaranteed"))


def replay(path: str | Path) -> ReplayResult:
    """Regenerate every stored case of ``path`` and check it against the stored defence."""
    doc = load_cases(path)
    p, d = load_defence(path)
    res = ReplayResult(str(path), p.name)
    for c in doc["cases"]:
        f = Finding.from_dict(c)
        case, _ = f.case(p)
        vals = d.scores(case)
        rej = d.rejects_values(vals)
        marg = max(d.margins(vals).values())
        ok = rej if f.kind == "escape" else not rej
        res.rows.append({"kind": f.kind, "seed": f.seed, "generator": f.generator, "rejected": bool(rej),
                         "margin": float(marg) if math.isfinite(marg) else math.inf, "ok": bool(ok)})
    return res
