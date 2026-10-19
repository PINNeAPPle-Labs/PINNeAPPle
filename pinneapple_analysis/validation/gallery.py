"""The ending every example in ``examples/validation/`` shares (#705): the validation report of a good model, the
trust card, the outputs ``pinneapple check`` reads in CI, and one real failure the checks catch on a broken variant.

:func:`conclude` compares the two reports, requires the expected check to fail on the broken model and not on the
good one, writes ``report.json`` / ``trust_card.json`` / ``report.sarif`` / ``failure.json`` and returns a
:class:`GalleryResult`. The gallery test runs every example at smoke size and checks these results."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["GalleryResult", "conclude", "FAMILIES", "PHYSICS"]

FAMILIES = ("pinn", "operator", "graph", "rom", "classical-ml", "neural-ode", "generative", "external")
PHYSICS = ("incompressible-flow", "heat", "elasticity", "electromagnetics", "porous-flow", "reaction-diffusion",
           "particle-dynamics", "advection", "structural-dynamics")


@dataclass
class GalleryResult:
    name: str
    family: tuple[str, ...]
    physics: tuple[str, ...]
    good: Any                                  # ValidationReport of the model as trained
    bad: Any                                   # ValidationReport of the broken variant
    caught: list[str]                          # checks failing on the broken model and not on the good one
    expected: str
    files: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        g = self.good.to_trust_report()
        return (f"{self.name}: good model -> {self.good.decision} (trust card {g.decisao}); broken variant -> "
                f"{self.bad.decision}, caught by {', '.join(self.caught)}")


def conclude(name: str, good, bad, *, expect: str, family, physics, out_dir: str | Path | None = None
             ) -> GalleryResult:
    """``good`` / ``bad``: :class:`ValidationReport` of the model and of its broken variant. ``expect``: the id (or
    id prefix) of the check that must catch the failure. Raises ``AssertionError`` when it does not."""
    family, physics = tuple([family] if isinstance(family, str) else family), \
        tuple([physics] if isinstance(physics, str) else physics)
    unknown = [f for f in family if f not in FAMILIES] + [p for p in physics if p not in PHYSICS]
    if unknown:
        raise ValueError(f"unknown family / physics {unknown}")
    good_status = {c.id: c.status for c in good.checks}
    caught = [c.id for c in bad.checks if c.status == "fail" and good_status.get(c.id) != "fail"]
    if not any(c == expect or c.startswith(expect + ".") for c in caught):
        lines = [f"  {c.id}: good {good_status.get(c.id)}, broken {c.status}" for c in bad.checks]
        raise AssertionError(f"{name}: the broken variant was not caught by {expect!r}\n" + "\n".join(lines))
    res = GalleryResult(name, family, physics, good, bad, caught, expect)
    if out_dir is None:                     # the repository's examples/_out (a source checkout), else the cwd's
        root = Path(__file__).resolve().parents[2]
        out_dir = (root if (root / "examples").is_dir() else Path.cwd()) / "examples" / "_out" / "validation" / name
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    good.to_json(out / "report.json")
    card = good.to_trust_report()
    (out / "trust_card.json").write_text(json.dumps(card.to_dict(), indent=2, default=str))
    (out / "report.sarif").write_text(good.render("sarif"))
    (out / "failure.json").write_text(bad.to_json())
    res.files = {k: str(out / f) for k, f in (("report", "report.json"), ("trust_card", "trust_card.json"),
                                               ("sarif", "report.sarif"), ("failure", "failure.json"))}
    print(res.summary())
    return res
