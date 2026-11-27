"""Running a contract: the fidelity ladder, the checks between its rungs, the decision, and the files that let
someone else replay it.

The runner is deliberately dumb. It walks the ladder the contract carries, asks the family for each rung, and turns
the family's checklines into a :class:`~pinneapple_analysis.trust.trust_report.TrustReport`, which already owns the
library's decision rule (a failed critical check is ``REJECT``, a failed non-critical one is ``REVIEW``, and a rung
that did not run lowers the coverage). Nothing here decides on its own whether a result is good.

What a run writes is the point of the whole capability: ``contract.json`` (what was asked and compiled),
``run.json`` (what ran, with its numbers and checks), ``report.md`` and any figures. :mod:`pinneapple_intent.replay`
rebuilds the run from those files alone, without an agent and without the sentence that started it.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pinneapple_analysis.trust.trust_report import (
    CONTRADICTS,
    NOT_RUN,
    SUPPORTS,
    Check,
    TrustReport,
)

from .contract import READY, RUN_SCHEMA_ID, StudyContract
from .family import (
    ANALYTIC,
    CONVERGENCE,
    NUMERICAL,
    SURROGATE,
    Measurement,
    StudyFamily,
    get_family,
)

# The checks that make or break a study: if the solver does not reproduce the closed form, or the quantity of
# interest moves when the grid is refined, the answer is not one to act on, whatever the rest says.
CRITICAL = ("against the", "change under refinement", "slender enough", "laminar, where")


@dataclass
class StudyRun:
    """What one execution produced: the contract, the rungs that ran, the decision, and where the files are."""
    contract: StudyContract
    family: str
    measurements: dict[str, Measurement] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    trust: TrustReport | None = None
    wall_s: float = 0.0
    out_dir: str = ""
    files: dict[str, str] = field(default_factory=dict)
    seed: int = 0

    # ------------------------------------------------------------------ the answer
    @property
    def decision(self) -> str:
        return self.trust.decisao if self.trust is not None else "REVIEW"

    @property
    def qoi(self) -> float | None:
        """The quantity of interest from the most reliable rung that ran: the solver, else the closed form."""
        for rung in (CONVERGENCE, NUMERICAL, ANALYTIC):
            measurement = self.measurements.get(rung)
            if measurement is not None and measurement.ran:
                family = get_family(self.family)
                value = measurement.outputs.get(family.qoi)
                if value is not None:
                    return float(value)
        return None

    def outputs(self) -> dict[str, Any]:
        family = get_family(self.family)
        return {"qoi": dict(name=family.qoi, label=family.qoi_label, unit=family.qoi_unit,
                            value=self.qoi, decision=self.decision),
                "stages": {name: m.to_dict() for name, m in self.measurements.items()}}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA_ID,
            "contract_id": self.contract.id,
            "family": self.family,
            "mode": self.contract.mode,
            "seed": self.seed,
            "decision": self.decision,
            "trust": self.trust.to_dict() if self.trust is not None else None,
            "stages": [m.to_dict() for m in self.measurements.values()],
            "checks": self.checks,
            "wall_s": self.wall_s,
            "files": dict(self.files),
            "library_version": self.contract.provenance.library_version,
            "contract": self.contract.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=False)

    def summary(self) -> str:
        family = get_family(self.family)
        value = self.qoi
        shown = "not computed" if value is None else f"{value:.6g} {family.qoi_unit}"
        passed = sum(1 for c in self.checks if c.get("passed"))
        return (f"{family.title}: {family.qoi} = {shown} ({self.decision}); "
                f"{passed}/{len(self.checks)} checks passed in {self.wall_s:.1f} s")

    def report_markdown(self) -> str:
        return _markdown(self)

    # ------------------------------------------------------------------ files
    def write(self, out_dir: str | None = None) -> dict[str, str]:
        target = out_dir or self.out_dir or os.path.join("study_runs", self.contract.id)
        os.makedirs(target, exist_ok=True)
        self.out_dir = target
        self.files["contract"] = self.contract.write(os.path.join(target, "contract.json"))
        self.files["run"] = _write(os.path.join(target, "run.json"), self.to_json())
        self.files["report"] = _write(os.path.join(target, "report.md"), self.report_markdown())
        if self.trust is not None:
            self.files["trust"] = _write(os.path.join(target, "trust.md"), self.trust.to_markdown())
        return dict(self.files)


def run_study(contract: StudyContract, *, out_dir: str | None = None, rungs: tuple[str, ...] | None = None,
              progress: Callable[[str], None] | None = None) -> StudyRun:
    """Run the ladder of ``contract`` and return the :class:`StudyRun`.

    A contract that still has open questions is refused: the runner never guesses the value it was not given.
    ``rungs`` overrides the ladder (for example to re-run only the surrogate); the decision then reports the rungs
    that did not run as ``not_run``, lowering coverage rather than passing silently.
    """
    say = progress or (lambda message: None)
    if contract.status != READY:
        raise ValueError(f"the contract is not ready: {contract.summary()}")
    family = get_family(contract.family)
    ladder = tuple(rungs) if rungs is not None else contract.rung_names
    started = time.perf_counter()
    run = StudyRun(contract=contract, family=family.name, seed=contract.provenance.seed, out_dir=out_dir or "")
    for name in ladder:
        rung = next((r for r in contract.ladder if r.name == name), None)
        available, why = rung.availability() if rung is not None else (True, "")
        if not available:
            run.measurements[name] = Measurement(rung=name, ran=False, reason=why)
            say(f"{name}: not run ({why})")
            continue
        say(f"{name}: running")
        try:
            measurement = family.evaluate(name, contract, workdir=out_dir,
                                          options={"h": _spacing(family, name, contract)})
        except Exception as exc:  # noqa: BLE001 - a rung that fails is a reported failure, not a crash
            run.measurements[name] = Measurement(rung=name, ran=False,
                                                 reason=f"{type(exc).__name__}: {str(exc)[:300]}")
            say(f"{name}: failed ({type(exc).__name__})")
            continue
        run.measurements[name] = measurement
        say(f"{name}: {measurement.wall_s:.2f} s, {sum(1 for c in measurement.checks if c['passed'])}/"
            f"{len(measurement.checks)} checks")
    run.wall_s = time.perf_counter() - started
    run.checks = family.compare(run.measurements)
    run.checks += [c for m in run.measurements.values() for c in m.checks]
    run.checks = _without_repeats(run.checks)
    run.trust = _trust(run, contract, ladder)
    if out_dir:
        run.write(out_dir)
    return run


def _without_repeats(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per check name.

    A rung that runs the solver twice (the refinement rung does) reports its own checks twice, and the same energy
    balance counted four times reads as four verifications. The first failure is kept, since that is the one a reader
    has to act on; a check that passed everywhere keeps its first row.
    """
    kept: dict[str, dict[str, Any]] = {}
    for item in checks:
        name = str(item.get("name", ""))
        previous = kept.get(name)
        if previous is None or (previous.get("passed") and not item.get("passed")):
            kept[name] = item
    return list(kept.values())


def _spacing(family: StudyFamily, rung: str, contract: StudyContract) -> float | None:
    """The grid spacing a rung uses; the family derives it from the case, the runner only asks for it."""
    return family.spacing(contract, rung)


def _trust(run: StudyRun, contract: StudyContract, ladder: tuple[str, ...]) -> TrustReport:
    """The library's trust decision, fed by the study's own checks and by the rungs that did not run."""
    checks: list[Check] = []
    for item in run.checks:
        name = str(item.get("name", ""))
        value = item.get("value")
        limit = item.get("limit")
        checks.append(Check(
            name=name, status=SUPPORTS if item.get("passed") else CONTRADICTS,
            critical=any(marker in name for marker in CRITICAL),
            value=float(value) if isinstance(value, (int, float)) else None,
            limit=float(limit) if isinstance(limit, (int, float)) else None,
            reason=str(item.get("detail", "") or "")))
    for name in ladder:
        measurement = run.measurements.get(name)
        if measurement is None or not measurement.ran:
            checks.append(Check(name=f"{name}: the rung ran", status=NOT_RUN,
                                reason=(measurement.reason if measurement is not None else "not attempted")))
    return TrustReport(checks, model=f"{contract.family}/{contract.mode}",
                       details={"contract_id": contract.id, "family": contract.family, "mode": contract.mode,
                                "wall_s": run.wall_s})


def _markdown(run: StudyRun) -> str:
    """The report: what was asked, what every number is, what ran, what failed, and what the study does not cover."""
    family = get_family(run.family)
    contract = run.contract
    lines = [f"# {family.title}", "",
             f"**Contract** `{contract.id}` · mode `{contract.mode}` · family `{contract.family}` · "
             f"decision **{run.decision}**", ""]
    lines += ["## What was asked", "", f"> {contract.provenance.text}", ""]
    lines += ["## Quantity of interest", ""]
    value = run.qoi
    lines.append(f"- **{family.qoi_label}**: "
                 + ("not computed" if value is None else f"`{value:.6g} {family.qoi_unit}`")
                 + f" — from the {_best_rung(run)} rung")
    lines += ["", "## Every value the study used", "",
              "| quantity | value | where it came from |", "|---|---|---|"]
    labels = {s.name: s.label for s in family.slots}
    for quantity in contract.values.values():
        lines.append(f"| {labels.get(quantity.name, quantity.name)} | {quantity.display()} | "
                     f"{_provenance_text(quantity)} |")
    lines += ["", "## Equations", ""] + [f"- `{e}`" for e in contract.equations]
    lines += ["", "## Assumptions", ""] + [f"- {a}" for a in contract.assumptions]
    lines += ["", "## The ladder", "",
              "| rung | what it is | ran | time |", "|---|---|---|---|"]
    for rung in contract.ladder:
        measurement = run.measurements.get(rung.name)
        ran = "yes" if measurement is not None and measurement.ran else (
            f"no — {measurement.reason}" if measurement is not None else "not attempted")
        seconds = f"{measurement.wall_s:.2f} s" if measurement is not None and measurement.ran else "—"
        lines.append(f"| {rung.name} | {rung.label} | {ran} | {seconds} |")
    lines += ["", "## Checks", "", "| check | result | value | limit |", "|---|---|---|---|"]
    for item in run.checks:
        mark = "pass" if item.get("passed") else "**fail**"
        lines.append(f"| {item.get('name', '')} | {mark} | {item.get('value', '')} | {item.get('limit', '')} |")
    failures = [c for c in run.checks if not c.get("passed")]
    if failures:
        lines += ["", "### What failed, and what it means", ""]
        lines += [f"- **{c.get('name')}**: {c.get('detail', '')}" for c in failures]
    lines += ["", "## What this study does not cover", ""] + [f"- {lim}" for lim in contract.limitations]
    if contract.references:
        lines += ["", "## References", ""] + [f"- {ref}" for ref in contract.references]
    if contract.unsupported:
        lines += ["", "## Parts of the request that were not used", ""] + \
                 [f"- {item}" for item in contract.unsupported]
    lines += ["", "---", "",
              "Every number above comes from a closed form, a solver call, or a value the contract records with its "
              "source. To reproduce the study without an agent: `python -m pinneapple_intent replay "
              f"{run.out_dir or '<run dir>'}`.", ""]
    return "\n".join(lines)


def _best_rung(run: StudyRun) -> str:
    for name in (CONVERGENCE, NUMERICAL, ANALYTIC, SURROGATE):
        measurement = run.measurements.get(name)
        if measurement is not None and measurement.ran:
            return name
    return "no"


def _provenance_text(quantity) -> str:
    if quantity.provenance == "stated" and quantity.quote:
        return f'stated in the request: "{quantity.quote}"'
    if quantity.provenance == "answered":
        return "answered the follow-up question"
    if quantity.provenance == "defaulted":
        return f"default: {quantity.source}"
    return quantity.provenance


def _write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def study(text: str, *, out_dir: str | None = None, answers: dict[str, Any] | None = None,
          family: str | StudyFamily | None = None, mode: str = "simulation", llm: Any = None, seed: int = 0,
          rungs: tuple[str, ...] | None = None, progress: Callable[[str], None] | None = None):
    """A sentence in, a run out: ``compile_study`` and ``run_study`` in one call.

    Returns the :class:`StudyRun` when the request compiled, or the contract carrying its questions (status
    ``needs_clarification``) or the refusal, so a caller always gets an object that says what happened and why.
    """
    from .compile import compile_study
    contract = compile_study(text, answers=answers, family=family, mode=mode, llm=llm, seed=seed)
    if isinstance(contract, StudyContract) and contract.status != READY:
        return contract
    if not isinstance(contract, StudyContract):
        return contract
    return run_study(contract, out_dir=out_dir, rungs=rungs, progress=progress)


__all__ = ["StudyRun", "run_study", "study", "CRITICAL"]
